from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

IMAGE_PATH_SEMANTIC = "image.path"
IMAGE_URL_SEMANTIC = "image.url"
IMAGE_URI_SEMANTIC = "image.uri"
IMAGE_THUMBNAIL_SEMANTIC = "image.thumbnail"
RECORD_ID_SEMANTIC = "record_id"

LABEL_SEMANTICS = ("target_label", "label", "class", "category")
PREDICTION_SEMANTICS = ("prediction", "predicted_label", "ml.prediction", "model.prediction")
UNCERTAINTY_SEMANTICS = ("uncertainty", "ml.uncertainty", "prediction_uncertainty")
AL_STATE_SEMANTICS = ("al.label_state", "label_state", "annotation_state")


@dataclass(frozen=True)
class AssetRef:
    """Lightweight reference to an image/media asset."""

    dataset_id: str
    row_id: str
    role: str
    uri: str
    media_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImagePreview:
    """Display-ready preview result."""

    asset: AssetRef
    data_uri: str
    cache_path: str
    width: int
    height: int
    original_width: int
    original_height: int
    mode: str
    format: str
    media_type: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["asset"] = self.asset.to_dict()
        return result


class ImageAssetResolver:
    """
    Resolve image assets from AstronomicAL datasets.

    Preferred mappings:
      - image.uri
      - image.path
      - image.url
      - image.thumbnail

    Gallery callers can prefer image.thumbnail without changing the single-image
    viewer, which should generally use the full-resolution asset.
    """

    def __init__(self, context: Any, *, cache_dir: str | Path | None = None) -> None:
        self.context = context
        self.cache_dir = Path(cache_dir or ".astronomical_cache/images").expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def asset_for_row(
        self,
        dataset_id: str,
        row_id: Any,
        *,
        role: str = "image",
        prefer_thumbnail: bool = False,
    ) -> AssetRef:
        columns = self._columns_for_asset_lookup(dataset_id, prefer_thumbnail=prefer_thumbnail)
        row = self.get_row(dataset_id, row_id, columns=columns or None)
        if row is None or row.empty:
            raise KeyError(f"Could not find row {row_id!r} in dataset {dataset_id!r}.")

        record = row.iloc[0].to_dict()
        uri, source_semantic, source_column = self._resolve_uri_from_record(
            dataset_id,
            record,
            prefer_thumbnail=prefer_thumbnail,
        )
        if not uri:
            raise ValueError(
                "No image asset mapping was found. Map one of 'image.uri', "
                "'image.path', 'image.url', or 'image.thumbnail'."
            )

        media_type = self._guess_media_type(uri)
        metadata = self._metadata_from_record(dataset_id, record)
        metadata.update(
            {
                "source_semantic": source_semantic,
                "source_column": source_column,
                "prefer_thumbnail": bool(prefer_thumbnail),
            }
        )
        return AssetRef(
            dataset_id=str(dataset_id),
            row_id=str(row_id),
            role=str(role or "image"),
            uri=str(uri),
            media_type=media_type,
            metadata=metadata,
        )

    def load_preview(
        self,
        asset: AssetRef,
        *,
        max_size: int = 2048,
        format: str = "JPEG",
        cancel_token: Any = None,
    ) -> ImagePreview:
        """Load an image asset, create a bounded preview, and return a data URI."""

        if cancel_token is not None and cancel_token.cancelled():
            raise RuntimeError("Image preview load cancelled.")

        save_format = "JPEG" if format.upper() in {"JPG", "JPEG"} else format.upper()
        cache_path = self._preview_cache_path(asset, max_size=max_size, format=save_format)

        cached = self._load_cached_preview(asset, cache_path, save_format=save_format)
        if cached is not None:
            return cached

        from PIL import Image, ImageOps

        raw = self._read_asset_bytes(asset.uri, cancel_token=cancel_token)
        if cancel_token is not None and cancel_token.cancelled():
            raise RuntimeError("Image preview load cancelled.")

        with Image.open(BytesIO(raw)) as image:
            image = ImageOps.exif_transpose(image)
            original_width, original_height = image.size

            if save_format == "JPEG" and image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")

            preview = image.copy()
            preview.thumbnail((int(max_size), int(max_size)))

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            preview.save(cache_path, format=save_format)

            buffer = BytesIO()
            preview.save(buffer, format=save_format)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            media_type = "image/jpeg" if save_format == "JPEG" else f"image/{save_format.lower()}"
            width, height = preview.size

            result = ImagePreview(
                asset=asset,
                data_uri=f"data:{media_type};base64,{encoded}",
                cache_path=str(cache_path),
                width=int(width),
                height=int(height),
                original_width=int(original_width),
                original_height=int(original_height),
                mode=str(image.mode),
                format=save_format,
                media_type=media_type,
            )
            self._write_preview_metadata(result)
            return result

    def load_preview_for_row(
        self,
        dataset_id: str,
        row_id: Any,
        *,
        role: str = "image",
        max_size: int = 2048,
        prefer_thumbnail: bool = False,
        cancel_token: Any = None,
    ) -> ImagePreview:
        asset = self.asset_for_row(
            dataset_id,
            row_id,
            role=role,
            prefer_thumbnail=prefer_thumbnail,
        )
        return self.load_preview(asset, max_size=max_size, cancel_token=cancel_token)

    def get_row(
        self,
        dataset_id: str,
        row_id: Any,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Retrieve one row using DatasetSource where possible."""

        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return pd.DataFrame()

        id_column = self.get_mapping(dataset_id, RECORD_ID_SEMANTIC) or "record_id"

        targets = [datasets]
        try:
            get_source = getattr(datasets, "get_source", None)
            if callable(get_source):
                source = get_source(dataset_id)
                if source is not None:
                    targets.append(source)
        except Exception:
            pass

        for target in targets:
            method = getattr(target, "get_row_by_id", None)
            if callable(method):
                try:
                    if target is datasets:
                        return method(dataset_id, row_id, id_column=id_column, columns=columns)
                    return method(row_id, id_column=id_column, columns=columns)
                except Exception:
                    try:
                        if target is datasets:
                            return method(dataset_id, row_id, id_column=id_column, columns=None)
                        return method(row_id, id_column=id_column, columns=None)
                    except Exception:
                        pass

        try:
            df = datasets.get_df(dataset_id, columns=columns)
        except TypeError:
            try:
                df = datasets.get_df(dataset_id)
            except Exception:
                df = None
        except Exception:
            df = None

        if df is None or not hasattr(df, "columns"):
            return pd.DataFrame()

        if id_column == "Use Index":
            mask = df.index.astype(str) == str(row_id)
        elif id_column in df.columns:
            mask = df[id_column].astype(str) == str(row_id)
        else:
            return pd.DataFrame(columns=df.columns)

        matches = df.loc[mask]
        if columns is not None:
            existing = [col for col in columns if col in matches.columns]
            matches = matches.loc[:, existing]
        return matches.head(1).copy()

    def get_rows(
        self,
        dataset_id: str,
        row_ids: Sequence[Any],
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Retrieve multiple rows by record id using DatasetManager vector lookup."""

        datasets = getattr(self.context, "datasets", None)
        if datasets is None or not row_ids:
            return pd.DataFrame()

        id_column = self.get_mapping(dataset_id, RECORD_ID_SEMANTIC) or "record_id"
        method = getattr(datasets, "get_rows_by_ids", None)
        if callable(method):
            try:
                return method(dataset_id, row_ids, id_column=id_column, columns=columns)
            except Exception:
                pass

        frames = [self.get_row(dataset_id, row_id, columns=columns) for row_id in row_ids]
        frames = [frame for frame in frames if frame is not None and not frame.empty]
        if not frames:
            return pd.DataFrame(columns=list(columns or []))
        return pd.concat(frames, ignore_index=True)

    def get_mapping(
        self,
        dataset_id: str,
        semantic_name: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return default

        for method_name in ("get_mapping", "mapping", "get_column_mapping"):
            method = getattr(datasets, method_name, None)
            if not callable(method):
                continue
            try:
                value = method(dataset_id, semantic_name, default)
            except TypeError:
                try:
                    value = method(dataset_id, semantic_name)
                except Exception:
                    value = None
            except Exception:
                value = None
            if value is not None:
                return str(value)

        try:
            method = getattr(datasets, "get_mappings", None)
            if callable(method):
                mappings = method(dataset_id)
                if isinstance(mappings, Mapping):
                    value = mappings.get(semantic_name, default)
                    return str(value) if value is not None else default
        except Exception:
            pass

        return default

    def _columns_for_asset_lookup(
        self,
        dataset_id: str,
        *,
        prefer_thumbnail: bool = False,
    ) -> list[str]:
        columns: list[str] = []
        for semantic in (
            RECORD_ID_SEMANTIC,
            IMAGE_THUMBNAIL_SEMANTIC,
            IMAGE_URI_SEMANTIC,
            IMAGE_PATH_SEMANTIC,
            IMAGE_URL_SEMANTIC,
            *LABEL_SEMANTICS,
            *PREDICTION_SEMANTICS,
            *UNCERTAINTY_SEMANTICS,
            *AL_STATE_SEMANTICS,
        ):
            column = self.get_mapping(dataset_id, semantic)
            if column and column != "Use Index":
                columns.append(column)

        fallback_columns = [
            "image_uri",
            "image_path",
            "image_url",
            "thumbnail",
            "thumbnail_uri",
            "thumbnail_path",
            "path",
            "filepath",
            "file_path",
            "filename",
            "url",
            "href",
            "target_label",
            "label",
            "labels",
            "class_name",
            "class",
            "category",
            "prediction",
            "predicted_label",
            "probability",
            "confidence",
            "uncertainty",
            "score",
            "label_state",
            "al_label_state",
        ]
        columns.extend(fallback_columns)
        return list(dict.fromkeys(str(col) for col in columns if col))

    def _resolve_uri_from_record(
        self,
        dataset_id: str,
        record: Mapping[str, Any],
        *,
        prefer_thumbnail: bool = False,
    ) -> tuple[str, str, str]:
        semantic_names = (
            (
                IMAGE_THUMBNAIL_SEMANTIC,
                IMAGE_URI_SEMANTIC,
                IMAGE_PATH_SEMANTIC,
                IMAGE_URL_SEMANTIC,
            )
            if prefer_thumbnail
            else (
                IMAGE_URI_SEMANTIC,
                IMAGE_PATH_SEMANTIC,
                IMAGE_URL_SEMANTIC,
                IMAGE_THUMBNAIL_SEMANTIC,
            )
        )

        for semantic_name in semantic_names:
            column = self.get_mapping(dataset_id, semantic_name)
            if column and column in record:
                uri = self._clean_value(record.get(column))
                if uri:
                    if semantic_name != IMAGE_URL_SEMANTIC:
                        uri = self._normalise_uri(dataset_id, uri)
                    return uri, semantic_name, column

        fallback_columns = (
            "thumbnail_uri",
            "thumbnail_path",
            "thumbnail",
            "thumb_uri",
            "thumb_path",
            "image_uri",
            "image_path",
            "image_url",
            "path",
            "filepath",
            "file_path",
            "filename",
            "url",
            "href",
        ) if prefer_thumbnail else (
            "image_uri",
            "image_path",
            "image_url",
            "path",
            "filepath",
            "file_path",
            "filename",
            "url",
            "href",
            "thumbnail_uri",
            "thumbnail_path",
            "thumbnail",
            "thumb_uri",
            "thumb_path",
        )

        for column in fallback_columns:
            if column not in record:
                continue
            uri = self._clean_value(record.get(column))
            if not uri:
                continue
            if column in {"image_url", "url", "href"}:
                parsed = urllib.parse.urlparse(uri)
                if not parsed.scheme:
                    uri = self._normalise_uri(dataset_id, uri)
            else:
                uri = self._normalise_uri(dataset_id, uri)
            return uri, f"fallback.{column}", column

        return "", "", ""

    def _metadata_from_record(self, dataset_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
        label_value, label_column = self._resolve_first_value(
            dataset_id,
            record,
            LABEL_SEMANTICS,
            ("target_label_name", "target_label", "label_name", "label", "labels", "class_name", "class", "category"),
        )
        prediction_value, prediction_column = self._resolve_first_value(
            dataset_id,
            record,
            PREDICTION_SEMANTICS,
            ("predicted_label", "prediction", "pred", "ml_prediction", "class_prediction"),
        )
        probability_value, probability_column = self._resolve_first_value(
            dataset_id,
            record,
            (),
            ("probability", "confidence", "prediction_probability", "prediction_confidence", "score"),
        )
        uncertainty_value, uncertainty_column = self._resolve_first_value(
            dataset_id,
            record,
            UNCERTAINTY_SEMANTICS,
            ("uncertainty", "prediction_uncertainty", "entropy", "margin", "least_confidence"),
        )
        label_state_value, label_state_column = self._resolve_first_value(
            dataset_id,
            record,
            AL_STATE_SEMANTICS,
            ("al_label_state", "label_state", "annotation_state", "review_state"),
        )

        return {
            "target_label": label_value,
            "target_label_column": label_column,
            "prediction": prediction_value,
            "prediction_column": prediction_column,
            "probability": probability_value,
            "probability_column": probability_column,
            "uncertainty": uncertainty_value,
            "uncertainty_column": uncertainty_column,
            "label_state": label_state_value,
            "label_state_column": label_state_column,
        }

    def _resolve_first_value(
        self,
        dataset_id: str,
        record: Mapping[str, Any],
        semantics: Sequence[str],
        fallback_columns: Sequence[str],
    ) -> tuple[str, str]:
        for semantic_name in semantics:
            column = self.get_mapping(dataset_id, semantic_name)
            if column and column in record:
                value = self._clean_value(record.get(column))
                if value:
                    return value, column

        for column in fallback_columns:
            if column in record:
                value = self._clean_value(record.get(column))
                if value:
                    return value, column
        return "", ""

    def _normalise_uri(self, dataset_id: str, value: str) -> str:
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme:
            return value

        path = Path(value).expanduser()
        if not path.is_absolute():
            base_path = self._dataset_base_path(dataset_id)
            if base_path:
                path = Path(base_path).expanduser() / path
        return path.resolve().as_uri()

    def _dataset_base_path(self, dataset_id: str) -> Optional[str]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return None

        try:
            ds = datasets.get(dataset_id)
            meta = getattr(ds, "meta", {}) or {}
        except Exception:
            try:
                meta = datasets.get_meta(dataset_id)
            except Exception:
                meta = {}

        for key in ("base_path", "image_base_path", "source_dir", "source_path"):
            value = meta.get(key)
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if path.is_file():
                return str(path.parent)
            return str(path)
        return None

    def _read_asset_bytes(self, uri: str, *, cancel_token: Any = None) -> bytes:
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme in {"http", "https"}:
            if cancel_token is not None and cancel_token.cancelled():
                raise RuntimeError("Image download cancelled.")
            with urllib.request.urlopen(uri, timeout=30) as response:
                return response.read()

        if parsed.scheme == "file":
            path = Path(urllib.request.url2pathname(parsed.path))
        elif parsed.scheme == "":
            path = Path(uri)
        else:
            raise ValueError(
                f"Unsupported image URI scheme {parsed.scheme!r}. "
                "Supported schemes are file, http, and https."
            )

        with path.expanduser().open("rb") as handle:
            return handle.read()


    def _load_cached_preview(
        self,
        asset: AssetRef,
        cache_path: Path,
        *,
        save_format: str,
    ) -> Optional[ImagePreview]:
        if not cache_path.exists() or not cache_path.is_file():
            return None

        try:
            raw = cache_path.read_bytes()
            encoded = base64.b64encode(raw).decode("ascii")
            media_type = "image/jpeg" if save_format == "JPEG" else f"image/{save_format.lower()}"
            meta_path = self._preview_metadata_path(cache_path)
            metadata: dict[str, Any] = {}
            if meta_path.exists():
                try:
                    metadata = json.loads(meta_path.read_text())
                except Exception:
                    metadata = {}

            width = int(metadata.get("width") or 0)
            height = int(metadata.get("height") or 0)
            original_width = int(metadata.get("original_width") or width or 0)
            original_height = int(metadata.get("original_height") or height or 0)
            mode = str(metadata.get("mode") or "")

            if not width or not height:
                from PIL import Image

                with Image.open(cache_path) as image:
                    width, height = image.size
                    original_width = original_width or width
                    original_height = original_height or height
                    mode = mode or str(image.mode)

            return ImagePreview(
                asset=asset,
                data_uri=f"data:{media_type};base64,{encoded}",
                cache_path=str(cache_path),
                width=int(width),
                height=int(height),
                original_width=int(original_width),
                original_height=int(original_height),
                mode=mode,
                format=save_format,
                media_type=media_type,
            )
        except Exception:
            return None

    def _write_preview_metadata(self, preview: ImagePreview) -> None:
        try:
            path = self._preview_metadata_path(Path(preview.cache_path))
            path.write_text(
                json.dumps(
                    {
                        "width": preview.width,
                        "height": preview.height,
                        "original_width": preview.original_width,
                        "original_height": preview.original_height,
                        "mode": preview.mode,
                        "format": preview.format,
                        "media_type": preview.media_type,
                    },
                    separators=(",", ":"),
                )
            )
        except Exception:
            pass

    @staticmethod
    def _preview_metadata_path(cache_path: Path) -> Path:
        return cache_path.with_suffix(cache_path.suffix + ".json")

    def _preview_cache_path(self, asset: AssetRef, *, max_size: int, format: str) -> Path:
        suffix = ".jpg" if format.upper() in {"JPG", "JPEG"} else f".{format.lower()}"
        key = hashlib.sha256(
            f"{asset.dataset_id}|{asset.row_id}|{asset.role}|{asset.uri}|{max_size}|{format}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}{suffix}"

    @staticmethod
    def _guess_media_type(uri: str) -> str:
        guessed, _ = mimetypes.guess_type(uri)
        return guessed or "image/*"

    @staticmethod
    def _clean_value(value: Any) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        text = str(value).strip()
        return text if text and text.lower() not in {"nan", "none", "null"} else ""


def get_image_resolver(context: Any) -> ImageAssetResolver:
    """Fetch the registered resolver, or create a local fallback."""

    services = getattr(context, "services", None)
    if services is not None:
        for key in ("core.image.asset_resolver", "asset_resolver"):
            try:
                resolver = services.get(key)
            except Exception:
                resolver = None
            if resolver is not None:
                return resolver
    return ImageAssetResolver(context)
