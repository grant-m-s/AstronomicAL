from __future__ import annotations

import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

DEFAULT_VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".m4v",
    ".mkv",
    ".webm",
    ".avi",
    ".mpeg",
    ".mpg",
)

def slugify(value: str, *, fallback: str = "video_dataset") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("._-")
    return slug or fallback

def parse_extensions(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_VIDEO_EXTENSIONS
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = [str(part).strip() for part in value]
    result: list[str] = []
    for part in parts:
        if not part:
            continue
        if not part.startswith("."):
            part = "." + part
        result.append(part.lower())
    return tuple(result or DEFAULT_VIDEO_EXTENSIONS)

def build_video_manifest_dataframe(
    root: str | Path,
    *,
    recursive: bool = True,
    extensions: str | Iterable[str] | None = None,
    label_from_parent: bool = False,
    relative_paths: bool = True,
    record_id_strategy: str = "relative_path",
    cancel_token: Any = None,
) -> pd.DataFrame:
    """Build a metadata-only video manifest without probing or decoding media."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Video folder does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Video root is not a directory: {root_path}")

    allowed = set(parse_extensions(extensions))
    iterator = root_path.rglob("*") if recursive else root_path.glob("*")
    rows: list[dict[str, Any]] = []

    for path in iterator:
        if _cancelled(cancel_token):
            break
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        rel_path = path.relative_to(root_path)
        stored_path = str(rel_path if relative_paths else path).replace("\\", "/")
        if record_id_strategy == "uuid":
            record_id = uuid.uuid4().hex
        elif record_id_strategy == "filename":
            record_id = path.stem
        else:
            record_id = str(rel_path).replace("\\", "/")
        media_type, _ = mimetypes.guess_type(str(path))
        row: dict[str, Any] = {
            "record_id": record_id,
            "video_path": stored_path,
            "video_uri": path.as_uri(),
            "video_media_kind": "video_file",
            "filename": path.name,
            "extension": path.suffix.lower(),
            "media_type": media_type or "video/*",
            "source_dir": str(root_path),
        }
        if label_from_parent:
            row["target_label"] = path.parent.name
        rows.append(row)

    columns = [
        "record_id",
        "video_path",
        "video_uri",
        "video_media_kind",
        "filename",
        "extension",
        "media_type",
        "source_dir",
        "target_label",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df.sort_values("record_id", kind="stable").reset_index(drop=True)

def video_column_mappings(columns_or_df: pd.DataFrame | Sequence[str]) -> dict[str, str]:
    columns = (
        {str(column) for column in columns_or_df.columns}
        if isinstance(columns_or_df, pd.DataFrame)
        else {str(column) for column in columns_or_df}
    )
    mappings: dict[str, str] = {"record_id": "record_id"}
    if "video_uri" in columns:
        mappings["video.uri"] = "video_uri"
    elif "video_path" in columns:
        mappings["video.uri"] = "video_path"
    if "video_path" in columns:
        mappings["video.path"] = "video_path"
    if "video_url" in columns:
        mappings["video.url"] = "video_url"
    optional = {
        "video.thumbnail_uri": ("video_thumbnail_uri", "thumbnail_uri"),
        "video.duration": ("video_duration", "duration_seconds", "duration"),
        "video.fps": ("video_fps", "fps"),
        "video.frame_count": ("video_frame_count", "frame_count", "num_frames"),
        "video.width": ("video_width", "width"),
        "video.height": ("video_height", "height"),
        "video.media_kind": ("video_media_kind", "media_kind"),
        "video.frame_uri_template": (
            "video_frame_uri_template",
            "frame_uri_template",
        ),
        "video.frame_uris": (
            "video_frame_uris",
            "frame_uris",
        ),
        "video.frame_index_offset": (
            "video_frame_index_offset",
            "frame_index_offset",
        ),
        "video.frame_index_step": (
            "video_frame_index_step",
            "frame_index_step",
            "frame_stride",
        ),
        "target_label": ("target_label",),
    }
    for semantic, aliases in optional.items():
        for column in aliases:
            if column in columns:
                mappings[semantic] = column
                break
    return mappings

def register_video_manifest_dataset(
    context: Any,
    df: pd.DataFrame,
    *,
    dataset_id: str,
    name: Optional[str] = None,
    base_path: str | Path | None = None,
    set_active: bool = True,
    write_parquet: bool = True,
    source_format: str = "video_manifest",
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Register video rows and verify canonical DatasetManager semantic mappings."""

    dataset_id = slugify(dataset_id)
    name = name or dataset_id
    mappings = video_column_mappings(df)
    if "video.uri" not in mappings:
        raise ValueError("Video manifests require a video_uri or video_path column.")

    meta = _video_registration_metadata(
        columns=[str(column) for column in df.columns],
        mappings=mappings,
        row_count=int(len(df)),
        base_path=base_path,
        source_format=source_format,
        extra_metadata=extra_metadata,
    )

    datasets = _require_datasets(context)
    backend = "pandas"
    parquet_path: Optional[Path] = None
    if write_parquet and len(df) > 0:
        try:
            cache_dir = Path(".astronomical_cache/datasets").expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = cache_dir / f"{dataset_id}.parquet"
            df.to_parquet(parquet_path, index=False)
            register_parquet = getattr(datasets, "register_parquet", None)
            if not callable(register_parquet):
                raise AttributeError("DatasetManager has no register_parquet")
            register_parquet(dataset_id, parquet_path, name=name, **meta)
            backend = "duckdb_parquet"
        except Exception:
            datasets.register(dataset_id, df, name=name, **meta)
            backend = "pandas"
    else:
        datasets.register(dataset_id, df, name=name, **meta)

    _apply_and_verify_mappings(datasets, dataset_id, mappings)
    _publish_video_dataset_events(
        context,
        dataset_id=dataset_id,
        name=name,
        rows=int(len(df)),
        backend=backend,
        mappings=mappings,
        set_active=set_active,
    )
    return {
        "dataset_id": dataset_id,
        "name": name,
        "rows": int(len(df)),
        "backend": backend,
        "parquet_path": str(parquet_path) if parquet_path else None,
        "mappings": dict(mappings),
        "mapping_verification": "get_mappings",
    }

def register_video_manifest_path(
    context: Any,
    *,
    manifest_path: str | Path,
    dataset_id: str,
    name: Optional[str] = None,
    columns: Sequence[str],
    row_count: int,
    mappings: Mapping[str, str] | None = None,
    base_path: str | Path | None = None,
    set_active: bool = True,
    source_format: str = "video_manifest",
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Register an already-written Parquet video manifest without materialising it.

    This is the public exchange point for importer plugins. The producing plugin
    owns acquisition/parsing and writes a compact manifest sidecar; ``core.video``
    owns the semantic mapping contract and canonical DatasetManager registration.
    """

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video manifest does not exist: {path}")
    if path.suffix.casefold() not in {".parquet", ".pq"}:
        raise ValueError("core.video register_manifest currently requires a Parquet manifest.")

    dataset_id = slugify(dataset_id)
    name = name or dataset_id
    clean_columns = [str(column) for column in columns if str(column).strip()]
    if not clean_columns:
        raise ValueError("Video manifest registration requires an explicit column inventory.")
    clean_mappings = {
        str(key): str(value)
        for key, value in dict(mappings or video_column_mappings(clean_columns)).items()
        if str(key).strip() and str(value).strip()
    }
    _validate_video_mappings(clean_columns, clean_mappings)

    meta = _video_registration_metadata(
        columns=clean_columns,
        mappings=clean_mappings,
        row_count=max(0, int(row_count or 0)),
        base_path=base_path,
        source_format=source_format,
        extra_metadata={
            "source_path": str(path),
            **dict(extra_metadata or {}),
        },
    )
    datasets = _require_datasets(context)
    register_parquet = getattr(datasets, "register_parquet", None)
    if not callable(register_parquet):
        raise RuntimeError(
            "DatasetManager.register_parquet() is required for external video manifests; "
            "core.video will not materialise an importer-owned manifest merely to register it."
        )
    register_parquet(dataset_id, path, name=name, **meta)
    _apply_and_verify_mappings(datasets, dataset_id, clean_mappings)
    _publish_video_dataset_events(
        context,
        dataset_id=dataset_id,
        name=name,
        rows=max(0, int(row_count or 0)),
        backend="duckdb_parquet",
        mappings=clean_mappings,
        set_active=set_active,
    )
    return {
        "dataset_id": dataset_id,
        "name": name,
        "rows": max(0, int(row_count or 0)),
        "backend": "duckdb_parquet",
        "parquet_path": str(path),
        "mappings": dict(clean_mappings),
        "mapping_verification": "get_mappings",
        "registered_from_manifest": True,
    }

def _video_registration_metadata(
    *,
    columns: Sequence[str],
    mappings: Mapping[str, str],
    row_count: int,
    base_path: str | Path | None,
    source_format: str,
    extra_metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "domain": "generic",
        "modality": "video",
        "source_format": str(source_format or "video_manifest"),
        "base_path": str(Path(base_path).expanduser().resolve()) if base_path else None,
        "row_count": max(0, int(row_count or 0)),
        "columns": [str(column) for column in columns],
        "column_mappings": dict(mappings),
    }
    meta.update(dict(extra_metadata or {}))
    # Importers cannot override canonical video mappings by smuggling an older
    # metadata copy into extra_metadata.
    meta["column_mappings"] = dict(mappings)
    return meta

def _validate_video_mappings(columns: Sequence[str], mappings: Mapping[str, str]) -> None:
    column_set = {str(column) for column in columns}
    for required in ("record_id", "video.uri"):
        column = str(mappings.get(required) or "").strip()
        if not column:
            raise ValueError(f"Video manifest requires semantic mapping {required!r}.")
        if column not in column_set:
            raise ValueError(
                f"Video mapping {required!r} points to missing manifest column {column!r}."
            )
    missing = {
        semantic: column
        for semantic, column in mappings.items()
        if str(column) not in column_set
    }
    if missing:
        raise ValueError(
            "Video mappings point to missing manifest columns: "
            + ", ".join(f"{key} -> {value}" for key, value in sorted(missing.items()))
        )

def _require_datasets(context: Any) -> Any:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise RuntimeError("context.datasets is unavailable.")
    return datasets

def _apply_and_verify_mappings(
    datasets: Any,
    dataset_id: str,
    mappings: Mapping[str, str],
) -> None:
    set_mapping = getattr(datasets, "set_mapping", None)
    if not callable(set_mapping):
        raise RuntimeError("DatasetManager does not expose set_mapping().")
    for semantic_name, column_name in mappings.items():
        set_mapping(dataset_id, semantic_name, column_name)

    get_mappings = getattr(datasets, "get_mappings", None)
    if not callable(get_mappings):
        raise RuntimeError("DatasetManager does not expose get_mappings() for verification.")
    retained = dict(get_mappings(dataset_id) or {})
    missing = {
        semantic: column
        for semantic, column in mappings.items()
        if str(retained.get(semantic) or "") != str(column)
    }
    if missing:
        raise RuntimeError(
            "DatasetManager did not retain the requested video mappings: "
            + ", ".join(f"{key} -> {value}" for key, value in sorted(missing.items()))
        )

def _publish_video_dataset_events(
    context: Any,
    *,
    dataset_id: str,
    name: str,
    rows: int,
    backend: str,
    mappings: Mapping[str, str],
    set_active: bool,
) -> None:
    events = getattr(context, "events", None)
    if events is not None:
        events.publish(
            "dataset.loaded",
            {
                "dataset_id": dataset_id,
                "name": name,
                "modality": "video",
                "rows": int(rows),
                "backend": backend,
                "origin": "core.video",
            },
        )
        events.publish(
            "dataset.mapping.updated",
            {
                "dataset_id": dataset_id,
                "mappings": dict(mappings),
                "origin": "core.video",
            },
        )

    if set_active:
        _require_datasets(context).set_active(dataset_id, origin="core.video")

def _cancelled(cancel_token: Any) -> bool:
    if cancel_token is None:
        return False
    cancelled = getattr(cancel_token, "cancelled", None)
    if callable(cancelled):
        try:
            return bool(cancelled())
        except Exception:
            return False
    return bool(cancelled)
