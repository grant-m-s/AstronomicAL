from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from service import HuggingFaceDatasetService
except ImportError:
    from .service import HuggingFaceDatasetService


def slugify(value: str, *, fallback: str = "hf_dataset") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("._-")
    return slug or fallback


def _emit_progress(progress_callback: Any, **payload: Any) -> None:
    if callable(progress_callback):
        try:
            progress_callback(payload)
        except Exception:
            pass

def import_hf_image_dataset_as_manifest(
    context: Any,
    *,
    repo_id: str,
    config_name: str | None,
    split: str,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    image_column: str | None = None,
    label_column: str | None = None,
    id_column: str | None = None,
    max_rows: int = 0,
    token: str | None = None,
    trust_remote_code: bool = False,
    write_parquet: bool = True,
    set_active: bool = True,
    cancel_token: Any = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    image_column = image_column or "__hf_file__"

    if image_column == "__hf_file__":
        return import_hf_files_as_manifest(
            context,
            repo_id=repo_id,
            split=split,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            max_rows=max_rows,
            token=token,
            write_parquet=write_parquet,
            set_active=set_active,
            cancel_token=cancel_token,
            progress_callback=progress_callback,
        )

    return import_hf_dataset_builder_as_manifest(
        context,
        repo_id=repo_id,
        config_name=config_name,
        split=split,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        image_column=image_column,
        label_column=label_column,
        id_column=id_column,
        max_rows=max_rows,
        token=token,
        trust_remote_code=trust_remote_code,
        write_parquet=write_parquet,
        set_active=set_active,
        cancel_token=cancel_token,
        progress_callback=progress_callback,
    )


def import_hf_files_as_manifest(
    context: Any,
    *,
    repo_id: str,
    split: str,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    max_rows: int = 0,
    token: str | None = None,
    write_parquet: bool = True,
    set_active: bool = True,
    cancel_token: Any = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """
    Preferred importer for ImageFolder-style Hugging Face datasets.

    Uses parallel hf_hub_download calls so the UI can show real file-count
    progress.
    """

    repo_id = str(repo_id or "").strip()
    if not repo_id:
        raise ValueError("repo_id is required.")

    split = str(split or "train").strip() or "train"

    service = _get_hf_service(context, token=token)

    dataset_id = slugify(
        dataset_id or f"hf_{repo_id}_files_{split}",
        fallback="hf_image_dataset",
    )
    dataset_name = dataset_name or f"HF {repo_id} [files / {split}]"

    _emit_progress(
        progress_callback,
        phase="listing",
        completed=0,
        total=0,
        percent=1,
        message=f"Listing image files for {repo_id} / {split}…",
    )

    files = service.list_image_files(
        repo_id,
        token=token,
        split=split,
        max_files=int(max_rows or 0),
    )

    if not files:
        raise ValueError(f"No image files found for split {split!r} in {repo_id!r}.")

    if cancel_token is not None and cancel_token.cancelled():
        return {
            "cancelled": True,
            "dataset_id": dataset_id,
            "rows": 0,
        }

    total = len(files)

    _emit_progress(
        progress_callback,
        phase="listed",
        completed=0,
        total=total,
        percent=5,
        message=f"Found {total} image file(s). Starting download…",
    )

    file_paths = [file.path for file in files]

    local_path_map = service.download_files_parallel(
        repo_id,
        file_paths,
        token=token,
        max_workers=16,
        progress_callback=progress_callback,
    )

    _emit_progress(
        progress_callback,
        phase="manifest",
        completed=0,
        total=total,
        percent=88,
        message="Building AstronomicAL image manifest…",
    )

    rows: list[dict[str, Any]] = []

    for index, file in enumerate(files):
        if cancel_token is not None and cancel_token.cancelled():
            return {
                "cancelled": True,
                "dataset_id": dataset_id,
                "rows": len(rows),
            }

        local_path = local_path_map.get(file.path)
        if not local_path:
            raise FileNotFoundError(
                f"Downloaded file was not found in local path map: {file.path}"
            )

        path_obj = Path(local_path).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(
                f"Downloaded local file does not exist: {path_obj}"
            )

        media_type = mimetypes.guess_type(str(path_obj))[0] or "image/*"
        record_id = f"{split}:{index}"
        label = file.label or ""

        rows.append(
            {
                "record_id": record_id,
                "image_path": str(path_obj),
                "image_uri": path_obj.as_uri(),
                "media_type": media_type,
                "target_label": label,
                "target_label_name": label,
                "hf_dataset_id": repo_id,
                "hf_config": "",
                "hf_split": split,
                "hf_row_index": index,
                "hf_file_path": file.path,
                "hf_file_size": file.size,
                "hf_image_column": "__hf_file__",
                "hf_label_column": "__hf_path_label__",
            }
        )

        if index % 250 == 0 or index == total - 1:
            _emit_progress(
                progress_callback,
                phase="manifest",
                completed=index + 1,
                total=total,
                percent=88 + int(((index + 1) / total) * 7),
                message=f"Building manifest row {index + 1}/{total}…",
            )

    manifest_df = pd.DataFrame(rows)

    _emit_progress(
        progress_callback,
        phase="registering",
        completed=total,
        total=total,
        percent=96,
        message="Registering dataset with AstronomicAL…",
    )

    result = register_manifest_dataset(
        context,
        manifest_df,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        repo_id=repo_id,
        config_name=None,
        split=split,
        image_column="__hf_file__",
        label_column="__hf_path_label__",
        write_parquet=write_parquet,
        set_active=set_active,
    )

    _emit_progress(
        progress_callback,
        phase="done",
        completed=total,
        total=total,
        percent=100,
        message=f"Registered {total} image file(s).",
    )

    result.update(
        {
            "repo_id": repo_id,
            "config_name": "",
            "split": split,
            "image_column": "__hf_file__",
            "label_column": "__hf_path_label__",
            "download_method": "parallel_hf_hub_download",
            "preview": manifest_df.head(25),
        }
    )
    return result


def import_hf_dataset_builder_as_manifest(
    context: Any,
    *,
    repo_id: str,
    config_name: str | None,
    split: str,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    image_column: str | None = None,
    label_column: str | None = None,
    id_column: str | None = None,
    max_rows: int = 0,
    token: str | None = None,
    trust_remote_code: bool = False,
    write_parquet: bool = True,
    set_active: bool = True,
    cancel_token: Any = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    
    repo_id = str(repo_id or "").strip()

    _emit_progress(
        progress_callback,
        phase="builder_loading",
        completed=0,
        total=0,
        percent=5,
        message=f"Loading Hugging Face dataset builder for {repo_id}…",
    )

    if not repo_id:
        raise ValueError("repo_id is required.")

    split = str(split or "train").strip() or "train"
    config_name = None if config_name in {None, "", "__default__", "default"} else str(config_name)

    service = _get_hf_service(context, token=token)
    dataset = service.load_split(
        repo_id=repo_id,
        config_name=config_name,
        split=split,
        token=token,
        trust_remote_code=trust_remote_code,
        streaming=False,
    )

    features = getattr(dataset, "features", {}) or {}
    image_column = image_column or service.infer_image_column(features, dataset=dataset)
    label_column = label_column or service.infer_label_column(features, dataset=dataset)

    if not image_column:
        raise ValueError(
            "Could not infer an image column. Provide image_column manually."
        )

    dataset = service.cast_image_decode_false(dataset, image_column)

    dataset_id = slugify(
        dataset_id
        or f"hf_{repo_id}_{config_name or 'default'}_{split}",
        fallback="hf_image_dataset",
    )
    dataset_name = dataset_name or f"HF {repo_id} [{config_name or 'default'} / {split}]"

    asset_cache_dir = (
        Path(".astronomical_cache/huggingface/assets")
        / slugify(repo_id)
        / slugify(config_name or "default")
        / slugify(split)
    ).expanduser()
    asset_cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    row_iterable = _iter_dataset_rows(dataset, max_rows=max_rows)
    for index, row in enumerate(row_iterable):
        if cancel_token is not None and cancel_token.cancelled():
            return {
                "cancelled": True,
                "dataset_id": dataset_id,
                "rows": len(rows),
            }

        record_id = service.record_id_for_row(
            row,
            index=index,
            split=split,
            id_column=id_column,
        )

        image_value = row.get(image_column)
        image_path, image_uri, media_type = _materialise_image_reference(
            image_value,
            asset_cache_dir=asset_cache_dir,
            row_index=index,
            record_id=record_id,
        )

        label_value = row.get(label_column) if label_column else None
        label_name = service.label_to_name(features, label_column, label_value)

        manifest_row: dict[str, Any] = {
            "record_id": str(record_id),
            "image_path": image_path,
            "image_uri": image_uri,
            "media_type": media_type,
            "target_label": _json_safe_scalar(label_value),
            "target_label_name": label_name,
            "hf_dataset_id": repo_id,
            "hf_config": config_name or "",
            "hf_split": split,
            "hf_row_index": index,
            "hf_image_column": image_column,
            "hf_label_column": label_column or "",
        }

        for key, value in row.items():
            if key == image_column:
                continue
            if key == label_column:
                continue
            if key in manifest_row:
                continue
            safe_value = _json_safe_scalar(value)
            if safe_value is not None:
                manifest_row[f"hf_{key}"] = safe_value

        rows.append(manifest_row)
        if max_rows and (index % 100 == 0 or index == max_rows - 1):
            _emit_progress(
                progress_callback,
                phase="builder_rows",
                completed=index + 1,
                total=max_rows,
                percent=10 + int(((index + 1) / max_rows) * 75),
                message=f"Materialising row {index + 1}/{max_rows}…",
            )

    manifest_df = pd.DataFrame(rows)

    _emit_progress(
        progress_callback,
        phase="registering",
        completed=len(rows),
        total=len(rows),
        percent=95,
        message="Registering dataset with AstronomicAL…",
    )

    result = register_manifest_dataset(
        context,
        manifest_df,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        repo_id=repo_id,
        config_name=config_name,
        split=split,
        image_column=image_column,
        label_column=label_column,
        write_parquet=write_parquet,
        set_active=set_active,
    )

    result.update(
        {
            "repo_id": repo_id,
            "config_name": config_name or "",
            "split": split,
            "image_column": image_column,
            "label_column": label_column or "",
            "download_method": "datasets_builder",
            "preview": manifest_df.head(25),
        }
    )

    _emit_progress(
        progress_callback,
        phase="done",
        completed=len(rows),
        total=len(rows),
        percent=100,
        message=f"Registered {len(rows)} row(s).",
    )

    return result


def register_manifest_dataset(
    context: Any,
    manifest_df: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_name: str,
    repo_id: str,
    config_name: str | None,
    split: str,
    image_column: str,
    label_column: str | None,
    write_parquet: bool = True,
    set_active: bool = True,
) -> dict[str, Any]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise RuntimeError("context.datasets is unavailable.")

    meta = {
        "domain": "generic",
        "modality": "image",
        "source_format": "huggingface_image_manifest",
        "hf_dataset_id": repo_id,
        "hf_config": config_name or "",
        "hf_split": split,
        "hf_image_column": image_column,
        "hf_label_column": label_column or "",
        "row_count": int(len(manifest_df)),
        "columns": [str(col) for col in manifest_df.columns],
        "column_mappings": {
            "record_id": "record_id",
            "image.path": "image_path",
            "image.uri": "image_uri",
            "target_label": "target_label_name" if "target_label_name" in manifest_df.columns else "target_label",
        },
    }

    backend = "pandas"
    parquet_path = None

    if write_parquet and len(manifest_df) > 0:
        try:
            cache_dir = Path(".astronomical_cache/datasets").expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = cache_dir / f"{dataset_id}.parquet"
            manifest_df.to_parquet(parquet_path, index=False)

            register_parquet = getattr(datasets, "register_parquet", None)
            if callable(register_parquet):
                register_parquet(
                    dataset_id,
                    parquet_path,
                    name=dataset_name,
                    **meta,
                )
                backend = "duckdb_parquet"
            else:
                raise AttributeError("DatasetManager has no register_parquet")
        except Exception:
            datasets.register(
                dataset_id,
                manifest_df,
                name=dataset_name,
                **meta,
            )
            backend = "pandas"
    else:
        datasets.register(
            dataset_id,
            manifest_df,
            name=dataset_name,
            **meta,
        )
        backend = "pandas"

    for semantic_name, column_name in meta["column_mappings"].items():
        try:
            datasets.set_mapping(dataset_id, semantic_name, column_name)
        except Exception:
            pass

    if set_active:
        try:
            datasets.set_active(dataset_id)
        except Exception:
            pass

    events = getattr(context, "events", None)
    if events is not None:
        events.publish(
            "dataset.loaded",
            {
                "dataset_id": dataset_id,
                "name": dataset_name,
                "modality": "image",
                "source": "huggingface",
                "hf_dataset_id": repo_id,
                "hf_config": config_name or "",
                "hf_split": split,
                "rows": int(len(manifest_df)),
                "backend": backend,
                "origin": "integrations.huggingface",
            },
        )
        events.publish(
            "dataset.mapping.updated",
            {
                "dataset_id": dataset_id,
                "mappings": dict(meta["column_mappings"]),
                "origin": "integrations.huggingface",
            },
        )
        if set_active:
            events.publish(
                "dataset.active.changed",
                {
                    "dataset_id": dataset_id,
                    "origin": "integrations.huggingface",
                },
            )

    return {
        "dataset_id": dataset_id,
        "name": dataset_name,
        "rows": int(len(manifest_df)),
        "backend": backend,
        "parquet_path": str(parquet_path) if parquet_path else None,
        "mappings": dict(meta["column_mappings"]),
    }


def _get_hf_service(context: Any, *, token: str | None = None) -> HuggingFaceDatasetService:
    services = getattr(context, "services", None)
    if services is not None:
        for key in (
            "integrations.huggingface.client",
            "huggingface.client",
        ):
            try:
                service = services.get(key)
                if service is not None:
                    if token:
                        service.token = token
                    return service
            except Exception:
                pass
    return HuggingFaceDatasetService(token=token)


def _iter_dataset_rows(dataset: Any, *, max_rows: int = 0):
    if max_rows and max_rows > 0:
        if hasattr(dataset, "select"):
            try:
                count = min(int(max_rows), len(dataset))
                return dataset.select(range(count))
            except Exception:
                pass
        return iter_limit(dataset, int(max_rows))
    return iter(dataset)


def iter_limit(dataset: Any, limit: int):
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        yield row


def _materialise_image_reference(
    value: Any,
    *,
    asset_cache_dir: Path,
    row_index: int,
    record_id: str,
) -> tuple[str, str, str]:
    if isinstance(value, dict):
        path = value.get("path")
        raw = value.get("bytes")

        if path:
            path_obj = Path(str(path)).expanduser()
            if path_obj.exists():
                media_type = mimetypes.guess_type(str(path_obj))[0] or "image/*"
                return str(path_obj), path_obj.resolve().as_uri(), media_type

        if raw is not None:
            return _write_image_bytes(
                raw,
                asset_cache_dir=asset_cache_dir,
                row_index=row_index,
                record_id=record_id,
            )

    if isinstance(value, (str, Path)):
        path_obj = Path(str(value)).expanduser()
        if path_obj.exists():
            media_type = mimetypes.guess_type(str(path_obj))[0] or "image/*"
            return str(path_obj), path_obj.resolve().as_uri(), media_type
        text = str(value)
        if text.startswith(("http://", "https://")):
            media_type = mimetypes.guess_type(text)[0] or "image/*"
            return text, text, media_type

    if hasattr(value, "save"):
        buffer = BytesIO()
        value.save(buffer, format="PNG")
        return _write_image_bytes(
            buffer.getvalue(),
            asset_cache_dir=asset_cache_dir,
            row_index=row_index,
            record_id=record_id,
            suffix=".png",
            media_type="image/png",
        )

    raise ValueError(
        f"Could not convert image value for row {row_index} into a path/URI."
    )


def _write_image_bytes(
    raw: bytes,
    *,
    asset_cache_dir: Path,
    row_index: int,
    record_id: str,
    suffix: str = ".img",
    media_type: str = "image/*",
) -> tuple[str, str, str]:
    digest = hashlib.sha256(raw).hexdigest()[:16]
    safe_id = slugify(str(record_id), fallback=f"row_{row_index}")
    path = asset_cache_dir / f"{row_index:08d}_{safe_id}_{digest}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.write_bytes(raw)

    return str(path), path.resolve().as_uri(), media_type


def _json_safe_scalar(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    if isinstance(value, bytes):
        return base64.b64encode(value[:128]).decode("ascii")

    return str(value)