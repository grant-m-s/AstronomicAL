from __future__ import annotations

import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

DEFAULT_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
)


def slugify(value: str, *, fallback: str = "image_dataset") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("._-")
    return slug or fallback


def parse_extensions(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_IMAGE_EXTENSIONS

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = [str(part).strip() for part in value]

    result = []
    for part in parts:
        if not part:
            continue
        if not part.startswith("."):
            part = "." + part
        result.append(part.lower())
    return tuple(result or DEFAULT_IMAGE_EXTENSIONS)


def build_image_manifest_dataframe(
    root: str | Path,
    *,
    recursive: bool = True,
    extensions: str | Iterable[str] | None = None,
    label_from_parent: bool = True,
    relative_paths: bool = True,
    record_id_strategy: str = "relative_path",
    cancel_token: Any = None,
) -> pd.DataFrame:
    """Scan a folder of image files and return a metadata-only manifest dataframe."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Image folder does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: {root_path}")

    allowed = set(parse_extensions(extensions))
    iterator = root_path.rglob("*") if recursive else root_path.glob("*")

    rows: list[dict[str, Any]] = []
    for path in iterator:
        if cancel_token is not None and cancel_token.cancelled():
            break
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in allowed:
            continue

        rel_path = path.relative_to(root_path)
        image_path = str(rel_path) if relative_paths else str(path)
        image_path = image_path.replace("\\", "/")

        if record_id_strategy == "uuid":
            record_id = uuid.uuid4().hex
        elif record_id_strategy == "filename":
            record_id = path.stem
        else:
            record_id = str(rel_path).replace("\\", "/")

        media_type, _ = mimetypes.guess_type(str(path))
        row: dict[str, Any] = {
            "record_id": record_id,
            "image_path": image_path,
            "image_uri": path.as_uri(),
            "filename": path.name,
            "extension": suffix,
            "media_type": media_type or "image/*",
            "source_dir": str(root_path),
        }
        if label_from_parent:
            row["target_label"] = path.parent.name
        rows.append(row)

    columns = [
        "record_id",
        "image_path",
        "image_uri",
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


def register_image_manifest_dataset(
    context: Any,
    df: pd.DataFrame,
    *,
    dataset_id: str,
    name: Optional[str] = None,
    base_path: str | Path | None = None,
    set_active: bool = True,
    write_parquet: bool = True,
) -> dict[str, Any]:
    """Register an image manifest as an AstronomicAL dataset."""

    dataset_id = slugify(dataset_id)
    name = name or dataset_id

    column_mappings: dict[str, str] = {
        "record_id": "record_id",
        "image.path": "image_path",
        "image.uri": "image_uri",
    }
    if "image_url" in df.columns:
        column_mappings["image.url"] = "image_url"
    if "thumbnail_uri" in df.columns:
        column_mappings["image.thumbnail"] = "thumbnail_uri"
    elif "thumbnail_path" in df.columns:
        column_mappings["image.thumbnail"] = "thumbnail_path"
    if "target_label" in df.columns:
        column_mappings["target_label"] = "target_label"

    meta: dict[str, Any] = {
        "domain": "generic",
        "modality": "image",
        "source_format": "image_manifest",
        "base_path": str(Path(base_path).expanduser().resolve()) if base_path else None,
        "row_count": int(len(df)),
        "columns": [str(col) for col in df.columns],
        "column_mappings": dict(column_mappings),
    }

    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise RuntimeError("context.datasets is unavailable.")

    registered_backend = "pandas"
    parquet_path = None
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
            registered_backend = "duckdb_parquet"
        except Exception:
            datasets.register(dataset_id, df, name=name, **meta)
            registered_backend = "pandas"
    else:
        datasets.register(dataset_id, df, name=name, **meta)
        registered_backend = "pandas"

    for semantic_name, column_name in column_mappings.items():
        try:
            datasets.set_mapping(dataset_id, semantic_name, column_name)
        except Exception:
            pass

    events = getattr(context, "events", None)
    if events is not None:
        events.publish(
            "dataset.loaded",
            {
                "dataset_id": dataset_id,
                "name": name,
                "modality": "image",
                "rows": int(len(df)),
                "backend": registered_backend,
                "origin": "core.image",
            },
        )
        events.publish(
            "dataset.mapping.updated",
            {
                "dataset_id": dataset_id,
                "mappings": dict(column_mappings),
                "origin": "core.image",
            },
        )

    if set_active:
        datasets.set_active(
            dataset_id,
            origin="core.image",
        )

    return {
        "dataset_id": dataset_id,
        "name": name,
        "rows": int(len(df)),
        "backend": registered_backend,
        "parquet_path": str(parquet_path) if parquet_path else None,
        "mappings": dict(column_mappings),
    }
