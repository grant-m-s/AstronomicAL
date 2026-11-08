from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from .fiftyone_format import match_classification_annotation
from .manifest_cache import (
    StreamingParquetManifest, cache_identity, cached_preview, immutable_revision,
    manifest_paths, pyarrow_available, read_completed_manifest, write_completed_manifest,
)

HF_FIFTYONE_GROUND_TRUTH = "__hf_fiftyone_ground_truth__"
HF_PATH_LABEL = "__hf_path_label__"
HF_LABEL_SEMANTICS_VERSION = 1

try:
    from service import (
        HFIncompleteSnapshotError,
        HFTransferCancelled,
        HuggingFaceDatasetService,
    )
except ImportError:
    from .service import (
        HFIncompleteSnapshotError,
        HFTransferCancelled,
        HuggingFaceDatasetService,
    )

def slugify(value: str, *, fallback: str = "hf_dataset") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("._-")
    return slug or fallback

def _missing_snapshot_paths(snapshot_path: str | Path, file_paths: list[str]) -> list[str]:
    root = Path(snapshot_path).expanduser()
    missing: list[str] = []
    for raw_path in file_paths:
        relative = str(raw_path or "").strip()
        if not relative:
            continue
        try:
            present = (root / relative).is_file()
        except OSError:
            present = False
        if not present:
            missing.append(relative)
    return missing

def find_registered_hf_dataset(
    context: Any,
    *,
    repo_id: str,
    config_name: str | None,
    split: str,
    revision: str | None = None,
    required_label_semantics_version: int | None = None,
    required_builder_import_identity: dict[str, Any] | None = None,
    allow_legacy_builder: bool = False,
) -> str | None:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None

    try:
        dataset_ids = list(datasets.list_ids())
    except Exception:
        return None

    wanted_repo = str(repo_id or "")
    wanted_config = str(config_name or "")
    wanted_split = str(split or "train")
    wanted_revision = str(revision or "").strip()

    for dataset_id in dataset_ids:
        try:
            meta = dict(datasets.get_meta(dataset_id) or {})
        except Exception:
            continue

        if str(meta.get("hf_dataset_id") or "") != wanted_repo:
            continue
        if str(meta.get("hf_config") or "") != wanted_config:
            continue
        if str(meta.get("hf_split") or "") != wanted_split:
            continue
        if wanted_revision and str(meta.get("hf_revision") or "") != wanted_revision:
            continue
        if required_builder_import_identity is not None:
            recorded = meta.get("hf_builder_import_identity")
            if recorded is None:
                if not allow_legacy_builder:
                    continue
            elif recorded != required_builder_import_identity:
                continue
        if required_label_semantics_version is not None:
            try:
                semantics_version = int(meta.get("hf_label_semantics_version") or 0)
            except Exception:
                semantics_version = 0
            if semantics_version < int(required_label_semantics_version):
                continue
        return str(dataset_id)

    return None

def _unique_dataset_id(context: Any, base: str) -> str:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return slugify(base)

    candidate = slugify(base)
    try:
        existing = set(str(value) for value in datasets.list_ids())
    except Exception:
        existing = set()

    if candidate not in existing:
        return candidate

    index = 2
    while f"{candidate}_{index}" in existing:
        index += 1
    return f"{candidate}_{index}"

def _emit_progress(progress_callback: Any, **payload: Any) -> None:
    if callable(progress_callback):
        try:
            progress_callback(payload)
        except Exception:
            pass

def _format_bytes(value: int | None) -> str:
    amount = max(int(value or 0), 0)
    number = float(amount)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024.0 or unit == "TiB":
            return f"{int(number):,} {unit}" if unit == "B" else f"{number:.1f} {unit}"
        number /= 1024.0
    return f"{amount:,} B"

def _annotation_matches_split(annotation: Any, split: str) -> bool:
    requested = str(split or "train").strip().casefold()
    aliases = {
        "training": "train",
        "tr": "train",
        "val": "validation",
        "valid": "validation",
        "dev": "validation",
        "testing": "test",
        "eval": "test",
    }
    requested = aliases.get(requested, requested)
    tags = tuple(str(value).strip().casefold() for value in getattr(annotation, "tags", ()) if str(value).strip())
    if not tags:
        return requested == "train"
    normalised = {aliases.get(value, value) for value in tags}
    return requested in normalised

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
    revision: str | None = None,
    prefer_repository_files: bool = False,
    token: str | None = None,
    trust_remote_code: bool = False,
    write_parquet: bool = True,
    set_active: bool = True,
    cancel_token: Any = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    image_column = str(image_column or "").strip()

    if prefer_repository_files and image_column != "__hf_file__":
        try:
            service = _get_hf_service(context, token=token)
            resolved_revision = service.resolve_dataset_revision(
                repo_id, revision=revision, token=token
            )
            viability = service.repository_image_manifest_viability(
                repo_id,
                split=split,
                revision=resolved_revision,
                token=token,
            )
            if bool(viability.get("usable")):
                image_column = "__hf_file__"
                revision = resolved_revision
                _emit_progress(
                    progress_callback,
                    phase="strategy",
                    completed=0,
                    total=0,
                    percent=1,
                    message=(
                        (
                            f"Native FiftyOne annotations declare {viability.get('fiftyone_ground_truth')} "
                            "ground truth; using the filtered snapshot importer with exact sample labels."
                        )
                        if viability.get("label_source") == "fiftyone_samples"
                        else (
                            "Repository paths contain semantic labels; using the filtered "
                            "snapshot importer instead of constructing the dataset row-by-row."
                        )
                    ),
                )
        except Exception:
            # Falling back to the builder is safe when a real structured image
            # column was selected; an absent image column is rejected explicitly
            # below instead of being reinterpreted as repository-file mode.
            pass

    if not image_column:
        raise ValueError(
            "No image input was detected for this dataset. AstronomicAL only uses "
            "the repository-image importer when inspection found actual image files; "
            "structured video/annotation repositories must use their dedicated "
            "registration path instead."
        )

    if image_column == "__hf_file__":
        return import_hf_files_as_manifest(
            context,
            repo_id=repo_id,
            split=split,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            label_column=label_column,
            max_rows=max_rows,
            revision=revision,
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
        revision=revision,
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
    label_column: str | None = None,
    max_rows: int = 0,
    revision: str | None = None,
    token: str | None = None,
    write_parquet: bool = True,
    set_active: bool = True,
    cancel_token: Any = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """
    Preferred importer for ImageFolder-style Hugging Face datasets.

    The selected files are acquired as one filtered, revision-pinned Hub snapshot.
    Small previews may still use individual file resolution, but full imports never
    perform thousands of top-level hf_hub_download calls.
    """

    repo_id = str(repo_id or "").strip()
    if not repo_id:
        raise ValueError("repo_id is required.")

    split = str(split or "train").strip() or "train"

    service = _get_hf_service(context, token=token)
    resolved_revision = service.resolve_dataset_revision(
        repo_id, revision=revision, token=token
    )

    existing_id = find_registered_hf_dataset(
        context,
        repo_id=repo_id,
        config_name=None,
        split=split,
        revision=resolved_revision,
        required_label_semantics_version=HF_LABEL_SEMANTICS_VERSION,
    )
    if existing_id is not None:
        if set_active:
            context.datasets.set_active(
                existing_id,
                origin="integrations.huggingface",
            )
        return {
            "dataset_id": existing_id,
            "name": getattr(context.datasets.get(existing_id), "name", existing_id),
            "rows": context.datasets.row_count(existing_id),
            "backend": context.datasets.get_meta(existing_id).get("backend", "unknown"),
            "repo_id": repo_id,
            "config_name": "",
            "split": split,
            "revision": resolved_revision,
            "download_method": "existing_registration",
            "existing": True,
            "cache_reused": True,
            "preview": context.datasets.head(existing_id, n=25),
        }

    dataset_id = _unique_dataset_id(
        context,
        dataset_id or f"hf_{repo_id}_files_{split}",
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

    fiftyone_index = None
    fiftyone_error: BaseException | None = None
    try:
        fiftyone_index = service.get_fiftyone_classification_index(
            repo_id,
            revision=resolved_revision,
            token=token,
            cancel_token=cancel_token,
            progress_callback=progress_callback,
        )
    except HFTransferCancelled:
        return {
            "cancelled": True,
            "dataset_id": dataset_id,
            "rows": 0,
            "repo_id": repo_id,
            "config_name": "",
            "split": split,
            "revision": resolved_revision,
        }
    except Exception as exc:
        fiftyone_error = exc
        if label_column == HF_FIFTYONE_GROUND_TRUTH:
            raise RuntimeError(
                "The selected native FiftyOne ground-truth source could not be loaded: "
                + str(exc)
            ) from exc
        _emit_progress(
            progress_callback,
            phase="annotation_fallback",
            completed=0,
            total=0,
            percent=-1,
            progress_known=False,
            severity="warning",
            message=(
                "Native FiftyOne annotations were unavailable; AstronomicAL will "
                "fall back to conservative repository path-label inference. "
                + str(exc)
            ),
        )

    # Native FiftyOne sample manifests are authoritative about which repository
    # media files are actual dataset rows.  Listing without a path-derived split
    # filter avoids losing flat repositories whose train/test split lives in tags.
    files = service.list_image_files(
        repo_id,
        token=token,
        split=None if fiftyone_index is not None else split,
        max_files=0 if fiftyone_index is not None else int(max_rows or 0),
        revision=resolved_revision,
    )

    annotation_by_path: dict[str, Any] = {}
    if fiftyone_index is not None:
        matched_files = []
        for file in files:
            annotation = match_classification_annotation(fiftyone_index, file.path)
            if annotation is None or not _annotation_matches_split(annotation, split):
                continue
            matched_files.append(file)
            annotation_by_path[file.path] = annotation

        # by_path contains multiple suffix keys for some samples. Count unique
        # filepaths rather than dictionary entries when reporting expectations.
        expected_rows = len({
            annotation.filepath
            for annotation in fiftyone_index.by_path.values()
            if _annotation_matches_split(annotation, split)
        })
        coverage = len(matched_files) / max(expected_rows, 1)
        if expected_rows and coverage < 0.80:
            message = (
                "Native FiftyOne samples.json could only be matched to "
                f"{len(matched_files):,}/{expected_rows:,} expected {split} sample(s)."
            )
            if label_column == HF_FIFTYONE_GROUND_TRUTH:
                raise RuntimeError(message + " Refusing to claim exact ground-truth alignment.")
            _emit_progress(
                progress_callback,
                phase="annotation_fallback",
                completed=len(matched_files),
                total=expected_rows,
                percent=-1,
                progress_known=False,
                severity="warning",
                message=message + " Falling back to repository path labels.",
            )
            fiftyone_index = None
            annotation_by_path = {}
            files = [item for item in files if str(item.split or "train") == split]
        else:
            files = matched_files
            if max_rows and max_rows > 0:
                files = files[: int(max_rows)]
                annotation_by_path = {
                    item.path: annotation_by_path[item.path]
                    for item in files
                }
            _emit_progress(
                progress_callback,
                phase="annotation_manifest_ready",
                completed=len(files),
                total=expected_rows or len(files),
                percent=-1,
                progress_known=False,
                message=(
                    f"Native FiftyOne sample manifest matched {len(files):,} {split} row(s); "
                    f"target labels come from {fiftyone_index.target_field!r}. "
                    "Repository images not referenced by samples.json are excluded."
                ),
            )

    if not files:
        detail = (f" FiftyOne annotation error: {fiftyone_error}" if fiftyone_error else "")
        raise ValueError(
            f"No image files found for split {split!r} in {repo_id!r}." + detail
        )

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
        message=f"Found {total} image file(s). Resolving local cache before download…",
    )

    file_paths = [file.path for file in files]
    file_sizes = {file.path: file.size for file in files}
    plan = service.snapshot_download_plan(
        repo_id,
        file_paths,
        revision=resolved_revision,
        token=token,
        max_workers=16,
        file_sizes=file_sizes,
    )

    source_size = (
        _format_bytes(plan.total_bytes)
        if plan.total_size_complete
        else (f"at least {_format_bytes(plan.total_bytes)}" if plan.total_bytes else "unknown")
    )
    if plan.download_size_complete:
        remaining_size = _format_bytes(plan.download_bytes)
    elif not plan.cache_coverage_known and plan.total_size_complete:
        remaining_size = f"up to {_format_bytes(plan.total_bytes)}"
    elif plan.download_bytes:
        remaining_size = f"at least {_format_bytes(plan.download_bytes)}"
    else:
        remaining_size = "unknown"

    if plan.cache_coverage_known:
        cached_size = _format_bytes(plan.cached_bytes) if plan.cached_bytes else "0 B"
        cache_text = f"{plan.cached_files:,} cached ({cached_size})"
        remaining_files_text = f"{plan.download_files:,} remaining"
    else:
        cache_text = "local cache detected, but exact selected-file coverage is not measurable from this legacy/partial snapshot"
        remaining_files_text = f"up to {plan.total_files or total:,} remaining"

    plan_message = (
        f"Snapshot plan: {plan.total_files or total:,} selected file(s) · {source_size} source · "
        f"{cache_text} · {remaining_files_text} ({remaining_size} network)."
    )
    if not plan.dry_run_supported:
        plan_message += (
            " Sizes come from the pinned repository tree plus verified local cache coverage; "
            "no remote per-file dry-run was used."
        )
    _emit_progress(
        progress_callback,
        phase="snapshot_plan",
        completed=plan.cached_files,
        total=plan.total_files or total,
        percent=-1,
        progress_known=False,
        progress_basis="planning",
        selected_total_bytes=plan.total_bytes if plan.total_size_complete else 0,
        expected_download_bytes=plan.download_bytes if plan.download_size_complete else 0,
        cached_bytes=plan.cached_bytes,
        message=plan_message,
    )

    try:
        snapshot_path = Path(
            service.snapshot_download_files(
                repo_id,
                file_paths,
                revision=resolved_revision,
                token=token,
                max_workers=16,
                quiet=True,
                cancel_token=cancel_token,
                progress_callback=progress_callback,
                snapshot_plan=plan,
            )
        ).expanduser().resolve()
    except HFTransferCancelled:
        return {
            "cancelled": True,
            "dataset_id": dataset_id,
            "rows": 0,
            "repo_id": repo_id,
            "config_name": "",
            "split": split,
            "revision": resolved_revision,
            "snapshot_plan": plan.to_dict(),
        }
    missing_paths = _missing_snapshot_paths(snapshot_path, file_paths)
    if missing_paths:
        sample = ", ".join(missing_paths[:3])
        if len(missing_paths) > 3:
            sample += f", … ({len(missing_paths) - 3} more)"
        raise HFIncompleteSnapshotError(
            (
                "Pinned Hugging Face snapshot is incomplete after acquisition: "
                f"{len(missing_paths):,}/{len(file_paths):,} requested file(s) are missing"
                + (f" ({sample})" if sample else "")
                + ". AstronomicAL refused to register a partial dataset."
            ),
            snapshot_path=str(snapshot_path),
            missing_paths=missing_paths,
        )

    local_path_map = {
        path: str((snapshot_path / path).resolve())
        for path in file_paths
    }

    _emit_progress(
        progress_callback,
        phase="manifest",
        completed=0,
        total=total,
        percent=88,
        message="Snapshot ready. Building AstronomicAL image manifest…",
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
        annotation = annotation_by_path.get(file.path)
        record_id = (
            str(getattr(annotation, "sample_id", "") or "").strip()
            if annotation is not None
            else ""
        ) or f"{split}:{index}"
        exact_label = str(getattr(annotation, "target_label", "") or "").strip()
        label = exact_label or str(file.label or "").strip()
        label_source = (
            f"fiftyone:{getattr(annotation, 'target_field', '')}"
            if exact_label
            else str(file.label_source or "repository_path")
        )

        manifest_row: dict[str, Any] = {
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
            "hf_label_column": (
                HF_FIFTYONE_GROUND_TRUTH if exact_label else HF_PATH_LABEL
            ),
            "hf_label_source": label_source,
            "hf_path_label_primary": str(file.label or ""),
            "hf_path_label_secondary": str(file.label_secondary or ""),
            "hf_path_label_secondary_confidence": file.label_secondary_confidence,
            "hf_path_label_prefix_value": file.label_prefix_value,
            "hf_path_label_parser": str(file.label_parser or ""),
            "hf_path_label_ambiguous": bool(file.label_ambiguous),
            "hf_fiftyone_ground_truth_field": (
                str(getattr(annotation, "target_field", "") or "")
                if annotation is not None else ""
            ),
            "hf_fiftyone_sample_id": (
                str(getattr(annotation, "sample_id", "") or "")
                if annotation is not None else ""
            ),
        }
        if annotation is not None:
            for prediction_field, prediction in dict(
                getattr(annotation, "predictions", {}) or {}
            ).items():
                prefix = "hf_fiftyone_" + slugify(
                    prediction_field, fallback="prediction"
                )
                manifest_row[prefix + "_label"] = str(
                    prediction.get("label") or ""
                )
                manifest_row[prefix + "_confidence"] = prediction.get("confidence")

        rows.append(manifest_row)

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
        label_column=(
            HF_FIFTYONE_GROUND_TRUTH if fiftyone_index is not None else HF_PATH_LABEL
        ),
        revision=resolved_revision,
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
            "revision": resolved_revision,
            "image_column": "__hf_file__",
            "label_column": (
                HF_FIFTYONE_GROUND_TRUTH if fiftyone_index is not None else HF_PATH_LABEL
            ),
            "label_source": (
                f"fiftyone:{fiftyone_index.target_field}"
                if fiftyone_index is not None else "repository_path"
            ),
            "download_method": "filtered_snapshot_download",
            "snapshot_path": str(snapshot_path),
            "snapshot_plan": plan.to_dict(),
            "cache_reused": bool(plan.cache_reused),
            "fully_cached": bool(plan.fully_cached),
            "downloaded_files": int(plan.download_files),
            "downloaded_bytes": int(plan.download_bytes),
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
    revision: str | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
    write_parquet: bool = True,
    set_active: bool = True,
    cancel_token: Any = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Import bounded image references; stream metadata to Parquet when supported."""
    started = time.perf_counter()
    timings: dict[str, float] = {}
    repo_id = str(repo_id or "").strip()
    if not repo_id:
        raise ValueError("repo_id is required.")
    split = str(split or "train").strip() or "train"
    config_name = None if config_name in {None, "", "__default__", "default"} else str(config_name)
    max_rows = max(0, int(max_rows or 0))

    _emit_progress(
        progress_callback, phase="builder_loading", completed=0, total=0, percent=-1,
        progress_known=False, message=f"Checking cached dataset and prepared imports for {repo_id}…",
    )
    service = _get_hf_service(context, token=token)
    step = time.perf_counter()
    resolved_revision = service.resolve_dataset_revision(repo_id, revision=revision, token=token)
    timings["revision_lookup_seconds"] = round(time.perf_counter() - step, 3)

    assets_dir = (service.assets_dir / slugify(repo_id) / slugify(config_name or "default")
                  / slugify(split)).expanduser()
    identity = cache_identity(
        repo_id=repo_id, revision=resolved_revision, config_name=config_name, split=split,
        image_column=image_column, label_column=label_column, id_column=id_column,
        max_rows=max_rows, assets_dir=assets_dir, trust_remote_code=trust_remote_code,
        label_semantics_version=HF_LABEL_SEMANTICS_VERSION,
    )
    manifest_root = getattr(service, "cache_dir", Path(".astronomical_cache/huggingface").resolve())
    manifest_path, metadata_path = manifest_paths(manifest_root, identity)
    register_parquet = getattr(getattr(context, "datasets", None), "register_parquet", None)
    stream_enabled = bool(write_parquet and callable(register_parquet) and pyarrow_available())
    can_reuse = bool(stream_enabled and immutable_revision(resolved_revision))

    existing_id = find_registered_hf_dataset(
        context, repo_id=repo_id, config_name=config_name, split=split, revision=resolved_revision,
        required_builder_import_identity=identity, allow_legacy_builder=max_rows == 0,
    )
    if existing_id is not None:
        meta = dict(context.datasets.get_meta(existing_id) or {})
        if set_active:
            context.datasets.set_active(existing_id, origin="integrations.huggingface")
        return {
            "dataset_id": existing_id,
            "name": getattr(context.datasets.get(existing_id), "name", existing_id),
            "rows": context.datasets.row_count(existing_id),
            "backend": meta.get("backend", "unknown"),
            "repo_id": repo_id, "config_name": config_name or "", "split": split,
            "revision": resolved_revision, "download_method": "existing_registration",
            "existing": True, "cache_reused": True,
            "import_timings": {**timings, "total_seconds": round(time.perf_counter() - started, 3)},
            "preview": context.datasets.head(existing_id, n=25),
        }

    dataset_id = _unique_dataset_id(context, dataset_id or f"hf_{repo_id}_{config_name or 'default'}_{split}")
    dataset_name = dataset_name or f"HF {repo_id} [{config_name or 'default'} / {split}]"

    if can_reuse:
        step = time.perf_counter()
        cached = read_completed_manifest(manifest_path, metadata_path, identity)
        timings["manifest_cache_check_seconds"] = round(time.perf_counter() - step, 3)
        if cached is not None:
            count = int(cached["rows"])
            _emit_progress(
                progress_callback, phase="manifest_cache_hit", completed=count, total=count,
                percent=95, progress_known=True, progress_basis="rows",
                message=f"Reusing {count:,} prepared image references; no image extraction required.",
            )
            step = time.perf_counter()
            result = register_manifest_dataset(
                context, None, dataset_id=dataset_id, dataset_name=dataset_name,
                repo_id=repo_id, config_name=config_name, split=split,
                image_column=str(cached["image_column"]), label_column=str(cached["label_column"]) or None,
                revision=resolved_revision, manifest_path=manifest_path,
                manifest_rows=count, manifest_columns=list(cached["columns"]),
                builder_import_identity=identity, write_parquet=True, set_active=set_active,
            )
            timings["registration_seconds"] = round(time.perf_counter() - step, 3)
            result.update({
                "repo_id": repo_id, "config_name": config_name or "", "split": split,
                "revision": resolved_revision, "image_column": cached["image_column"],
                "label_column": cached["label_column"], "download_method": "prepared_manifest_cache",
                "cache_reused": True, "manifest_reused": True,
                "preview": cached_preview(manifest_path),
                "import_timings": {**timings, "total_seconds": round(time.perf_counter() - started, 3)},
            })
            _emit_progress(
                progress_callback, phase="done", completed=count, total=count,
                percent=100, message=f"Reused {count:,} cached image references.",
            )
            return result

    _emit_progress(
        progress_callback, phase="builder_loading", completed=0, total=0, percent=-1,
        progress_known=False, message=f"Loading {repo_id} / {split} from local Hugging Face cache if available…",
    )
    step = time.perf_counter()
    dataset = service.load_split(
        repo_id=repo_id, config_name=config_name, split=split, revision=resolved_revision,
        token=token, trust_remote_code=trust_remote_code, streaming=False,
    )
    timings["split_loading_seconds"] = round(time.perf_counter() - step, 3)
    features = getattr(dataset, "features", {}) or {}
    image_column = image_column or service.infer_image_column(features, dataset=dataset)
    label_column = label_column or service.infer_label_column(features, dataset=dataset)
    if not image_column:
        raise ValueError("Could not infer an image column. Provide image_column manually.")
    dataset = service.cast_image_decode_false(dataset, image_column)
    assets_dir.mkdir(parents=True, exist_ok=True)
    try:
        available_rows = len(dataset)
    except (TypeError, ValueError):
        available_rows = 0
    expected_rows = min(max_rows, available_rows) if max_rows and available_rows else available_rows
    if not stream_enabled and expected_rows >= 10_000:
        _emit_progress(
            progress_callback, phase="builder_backend_warning", completed=0, total=0,
            percent=-1, progress_known=False, severity="warning",
            message=("Streaming Parquet is unavailable; using the existing in-memory pandas "
                     "manifest path. Large imports may exceed available RAM."),
        )
    _emit_progress(
        progress_callback, phase="builder_rows", completed=0, total=expected_rows,
        percent=10 if expected_rows else -1, progress_known=bool(expected_rows),
        progress_basis="rows", message=(f"Preparing {expected_rows:,} image references…" if expected_rows
                                        else "Preparing image references (count unavailable)…"),
    )

    rows: list[dict[str, Any]] = []
    preview: list[dict[str, Any]] = []
    writer = StreamingParquetManifest(manifest_path) if stream_enabled else None
    completed = 0
    step = time.perf_counter()
    try:
        for index, row in enumerate(_iter_dataset_rows(dataset, max_rows=max_rows)):
            if cancel_token is not None and cancel_token.cancelled():
                return {"cancelled": True, "dataset_id": dataset_id, "rows": completed}
            record_id = service.record_id_for_row(row, index=index, split=split, id_column=id_column)
            image_path, image_uri, media_type = _materialise_image_reference(
                row.get(image_column), asset_cache_dir=assets_dir, row_index=index, record_id=record_id,
            )
            label_value = row.get(label_column) if label_column else None
            label_name = service.label_to_name(features, label_column, label_value)
            display_label = label_name
            if display_label in (None, "") and label_value not in (None, ""):
                display_label = str(_json_safe_scalar(label_value))
            manifest_row: dict[str, Any] = {
                "record_id": str(record_id), "image_path": image_path, "image_uri": image_uri,
                "media_type": media_type, "target_label": _json_safe_scalar(label_value),
                "target_label_name": display_label or "", "hf_dataset_id": repo_id,
                "hf_config": config_name or "", "hf_split": split, "hf_row_index": index,
                "hf_image_column": image_column, "hf_label_column": label_column or "",
            }
            for key, value in row.items():
                if key in {image_column, label_column} or key in manifest_row:
                    continue
                safe_value = _json_safe_scalar(value)
                if safe_value is not None:
                    manifest_row[f"hf_{key}"] = safe_value
            if writer is not None:
                writer.append(manifest_row)
            else:
                rows.append(manifest_row)
            if len(preview) < 25:
                preview.append(manifest_row)
            completed = index + 1
            if index == 0 or completed % 200 == 0 or (expected_rows and completed == expected_rows):
                _emit_progress(
                    progress_callback, phase="builder_rows", completed=completed, total=expected_rows,
                    percent=10 + int((completed / expected_rows) * 75) if expected_rows else -1,
                    progress_known=bool(expected_rows), progress_basis="rows",
                    message=(f"Preparing image {completed:,}/{expected_rows:,}…" if expected_rows
                             else f"Prepared {completed:,} images…"),
                )
        timings["image_extraction_seconds"] = round(time.perf_counter() - step, 3)
        step = time.perf_counter()
        if writer is not None:
            count, columns = writer.finish()
            manifest_df = None
            write_completed_manifest(
                metadata_path, identity=identity, rows=count, columns=columns,
                image_column=image_column, label_column=label_column,
            )
        else:
            manifest_df = pd.DataFrame(rows)
            count = len(rows)
            columns = list(manifest_df.columns)
        timings["manifest_finalization_seconds"] = round(time.perf_counter() - step, 3)
    finally:
        if writer is not None:
            writer.close()

    _emit_progress(
        progress_callback, phase="registering", completed=count, total=count, percent=95,
        message=f"Registering {count:,} image references with AstronomicAL…",
    )
    step = time.perf_counter()
    result = register_manifest_dataset(
        context, manifest_df, dataset_id=dataset_id, dataset_name=dataset_name,
        repo_id=repo_id, config_name=config_name, split=split, image_column=image_column,
        label_column=label_column, revision=resolved_revision,
        manifest_path=manifest_path if writer is not None else None,
        manifest_rows=count if writer is not None else None,
        manifest_columns=columns if writer is not None else None,
        builder_import_identity=identity, write_parquet=write_parquet, set_active=set_active,
    )
    timings["registration_seconds"] = round(time.perf_counter() - step, 3)
    result.update({
        "repo_id": repo_id, "config_name": config_name or "", "split": split,
        "revision": resolved_revision, "image_column": image_column,
        "label_column": label_column or "", "download_method": "datasets_builder_revision_pinned",
        "cache_reused": bool(service.cache_status(repo_id, revision=resolved_revision).cached),
        "manifest_reused": False, "preview": pd.DataFrame(preview),
        "import_timings": {**timings, "total_seconds": round(time.perf_counter() - started, 3)},
    })
    _emit_progress(
        progress_callback, phase="done", completed=count, total=count,
        percent=100, message=f"Registered {count:,} row(s).",
    )
    return result

def _read_dataset_mappings(
    datasets: Any,
    dataset_id: str,
    expected: dict[str, str],
) -> tuple[dict[str, str], str]:
    getter = getattr(datasets, "get_mappings", None)
    if callable(getter):
        try:
            return {
                str(key): str(value)
                for key, value in dict(getter(dataset_id) or {}).items()
                if value not in (None, "")
            }, "get_mappings"
        except Exception:
            pass

    single_getter = getattr(datasets, "get_mapping", None)
    if callable(single_getter):
        values: dict[str, str] = {}
        readable = False
        for semantic_name in expected:
            try:
                value = single_getter(dataset_id, semantic_name)
                readable = True
            except Exception:
                continue
            if value not in (None, ""):
                values[str(semantic_name)] = str(value)
        if readable:
            return values, "get_mapping"

    try:
        meta = dict(datasets.get_meta(dataset_id) or {})
        return {
            str(key): str(value)
            for key, value in dict(meta.get("column_mappings") or {}).items()
            if value not in (None, "")
        }, "metadata"
    except Exception:
        return {}, "unavailable"

def _ensure_dataset_mappings(
    datasets: Any,
    dataset_id: str,
    expected: dict[str, str],
    *,
    available_columns: list[str],
) -> tuple[dict[str, str], str]:
    columns = {str(value) for value in available_columns}
    invalid = {
        semantic: column
        for semantic, column in expected.items()
        if str(column) not in columns
    }
    if invalid:
        raise RuntimeError(
            "Hugging Face importer attempted to register semantic mappings to "
            f"missing physical columns: {invalid!r}."
        )

    # Reassert mappings *after* DatasetManager registration. Workspace restore can
    # legitimately apply persisted dataset state during register_source/register_parquet;
    # setting the importer's mappings afterwards makes the final canonical state
    # deterministic instead of relying on registration metadata surviving restore.
    setter = getattr(datasets, "set_mapping", None)
    if callable(setter):
        for semantic_name, column_name in expected.items():
            try:
                setter(dataset_id, semantic_name, column_name)
            except Exception as exc:
                raise RuntimeError(
                    "DatasetManager rejected the Hugging Face semantic mapping "
                    f"{semantic_name!r} -> {column_name!r}: {exc}"
                ) from exc

    actual, verification_mode = _read_dataset_mappings(
        datasets, dataset_id, expected
    )
    mismatched = {
        semantic: {"expected": column, "actual": actual.get(semantic)}
        for semantic, column in expected.items()
        if actual.get(semantic) != column
    }
    if mismatched:
        if not callable(setter):
            reason = "DatasetManager exposes no set_mapping() method"
        else:
            reason = "DatasetManager did not retain the mappings after set_mapping()"
        raise RuntimeError(
            f"{reason}; verified via {verification_mode}. Mismatches: {mismatched!r}."
        )

    return {semantic: actual[semantic] for semantic in expected}, verification_mode

def register_manifest_dataset(
    context: Any,
    manifest_df: pd.DataFrame | None,
    *,
    dataset_id: str,
    dataset_name: str,
    repo_id: str,
    config_name: str | None,
    split: str,
    image_column: str,
    label_column: str | None,
    revision: str | None = None,
    write_parquet: bool = True,
    set_active: bool = True,
    manifest_path: Path | None = None,
    manifest_rows: int | None = None,
    manifest_columns: list[str] | None = None,
    builder_import_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register either a pandas manifest or an already-written Parquet manifest."""
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise RuntimeError("context.datasets is unavailable.")
    external_manifest = manifest_path is not None
    if external_manifest:
        if not write_parquet or manifest_rows is None or manifest_columns is None:
            raise ValueError("External manifests require Parquet, explicit row count and column inventory.")
        if not Path(manifest_path).is_file():
            raise FileNotFoundError(f"Image manifest does not exist: {manifest_path}")
        row_count = int(manifest_rows)
        columns = [str(column) for column in manifest_columns]
    elif manifest_df is not None:
        row_count = int(len(manifest_df))
        columns = [str(column) for column in manifest_df.columns]
    else:
        raise ValueError("A manifest DataFrame or Parquet path is required.")

    meta = {
        "domain": "generic", "modality": "image", "source_format": "huggingface_image_manifest",
        "hf_dataset_id": repo_id, "hf_config": config_name or "", "hf_split": split,
        "hf_revision": str(revision or ""), "hf_image_column": image_column,
        "hf_label_column": label_column or "", "hf_label_semantics_version": HF_LABEL_SEMANTICS_VERSION,
        "hf_label_source": (
            "fiftyone_samples" if label_column == HF_FIFTYONE_GROUND_TRUTH
            else "repository_path_or_structured_feature"
        ),
        "loader_id": "integrations.huggingface", "row_count": row_count, "columns": columns,
        "column_mappings": {
            "record_id": "record_id", "image.path": "image_path", "image.uri": "image_uri",
            "target_label": "target_label_name" if "target_label_name" in columns else "target_label",
        },
    }
    if builder_import_identity is not None:
        meta["hf_builder_import_identity"] = dict(builder_import_identity)

    backend = "pandas"
    parquet_path = Path(manifest_path) if external_manifest else None
    if external_manifest:
        register_parquet = getattr(datasets, "register_parquet", None)
        if not callable(register_parquet):
            raise RuntimeError("Bounded Hugging Face imports require DatasetManager.register_parquet().")
        register_parquet(
            dataset_id, parquet_path, name=dataset_name, source_path=str(parquet_path), **meta,
        )
        backend = "duckdb_parquet"
    elif write_parquet and row_count:
        try:
            cache_dir = Path(".astronomical_cache/datasets").expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = cache_dir / f"{dataset_id}.parquet"
            manifest_df.to_parquet(parquet_path, index=False)
            register_parquet = getattr(datasets, "register_parquet", None)
            if not callable(register_parquet):
                raise AttributeError("DatasetManager has no register_parquet")
            register_parquet(
                dataset_id, parquet_path, name=dataset_name, source_path=str(parquet_path), **meta,
            )
            backend = "duckdb_parquet"
        except Exception:
            datasets.register(dataset_id, manifest_df, name=dataset_name, **meta)
            backend = "pandas"
    else:
        datasets.register(dataset_id, manifest_df, name=dataset_name, **meta)

    verified_mappings, verification = _ensure_dataset_mappings(
        datasets, dataset_id, dict(meta["column_mappings"]), available_columns=columns,
    )
    events = getattr(context, "events", None)
    if events is not None:
        events.publish("dataset.loaded", {
            "dataset_id": dataset_id, "name": dataset_name, "modality": "image",
            "source": "huggingface", "hf_dataset_id": repo_id, "hf_config": config_name or "",
            "hf_split": split, "hf_revision": str(revision or ""), "rows": row_count,
            "backend": backend, "origin": "integrations.huggingface",
        })
        events.publish("dataset.mapping.updated", {
            "dataset_id": dataset_id, "mappings": dict(verified_mappings),
            "verification": verification, "origin": "integrations.huggingface",
        })
    if set_active:
        datasets.set_active(dataset_id, origin="integrations.huggingface")
    return {
        "dataset_id": dataset_id, "name": dataset_name, "rows": row_count,
        "backend": backend, "parquet_path": str(parquet_path) if parquet_path else None,
        "mappings": dict(verified_mappings), "mapping_verification": verification,
        "existing": False,
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
        # Avoid retaining a truncated image when an import is interrupted.
        temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.partial")
        try:
            temp.write_bytes(raw)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

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
