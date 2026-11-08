from __future__ import annotations

"""Import Hugging Face repository-hosted video files via core.video's public API.

This adapter deliberately operates on repository media files (MP4/MOV/etc.), not
YTVIS annotations or datasets.Video features. Video bytes remain in HF's cache;
only local paths and provenance are written to a compact Parquet manifest.
"""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import pandas as pd

EPIC_KITCHENS_REPO_ID = "lightly-ai/epic-kitchens-100-clips"

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mpeg", ".mpg"})
DEFAULT_MAX_INVENTORY = 20_000
DEFAULT_MAX_BYTES = 2 * 1024**3

def _cancelled(cancel_token: Any) -> bool:
    if cancel_token is None:
        return False
    value = getattr(cancel_token, "cancelled", False)
    return bool(value() if callable(value) else value)

def _check_cancel(cancel_token: Any) -> None:
    if _cancelled(cancel_token):
        raise RuntimeError("Hugging Face video import cancelled.")

def _safe_repo_path(raw: str) -> str:
    value = str(raw or "").replace("\\", "/").strip()
    path = PurePosixPath(value)
    if not value or value.startswith("/") or any(part in {".", ".."} for part in value.split("/")):
        raise ValueError(f"Unsafe Hugging Face repository path: {raw!r}.")
    if path.suffix.casefold() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video file: {raw!r}.")
    return str(path)

def _slug(value: str, *, default: str = "hf_video") -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_-")[:88] or default

def _progress(callback: Any, *, phase: str, completed: int, total: int, message: str) -> None:
    if callable(callback):
        callback({
            "phase": phase,
            "completed": completed,
            "total": total,
            "percent": int(100 * completed / total) if total else -1,
            "progress_known": bool(total),
            "message": message,
        })

def _pinned_revision(repo_id: str, *, revision: str | None, token: str | None) -> str:
    from huggingface_hub import HfApi

    info = HfApi(token=token).dataset_info(repo_id=repo_id, revision=revision, token=token)
    sha = str(getattr(info, "sha", "") or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise RuntimeError("Hugging Face did not return a verifiable dataset commit SHA.")
    return sha

def select_repository_videos(
    service: Any,
    *,
    repo_id: str,
    revision: str,
    token: str | None,
    path_prefix: str = "",
    max_videos: int = 10,
    max_inventory: int = DEFAULT_MAX_INVENTORY,
    max_total_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[list[dict[str, Any]], bool]:
    """Select a deterministic bounded prefix without downloading the media."""
    if not 1 <= int(max_videos) <= 1000:
        raise ValueError("max_videos must be between 1 and 1000.")
    if not 1 <= int(max_inventory) <= 100_000:
        raise ValueError("max_inventory must be between 1 and 100000.")
    if int(max_total_bytes) <= 0:
        raise ValueError("max_total_bytes must be positive.")

    prefix = str(path_prefix or "").replace("\\", "/").strip("/")
    if any(part in {".", ".."} for part in prefix.split("/") if part):
        raise ValueError("path_prefix must be a repository-relative directory.")

    inventory = service.list_repository_files(
        repo_id,
        revision=revision,
        token=token,
        path_in_repo=prefix or None,
        max_files=int(max_inventory),
    )
    # The service returns a bounded inventory; don't claim full repository coverage.
    truncated = len(inventory) >= int(max_inventory)
    videos: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(inventory, key=lambda item: str(item.get("path", ""))):
        raw_path = str(row.get("path") or "")
        if PurePosixPath(raw_path).suffix.casefold() not in VIDEO_SUFFIXES:
            continue
        path = _safe_repo_path(raw_path)
        if prefix and not path.startswith(prefix + "/"):
            continue
        if path in seen:
            continue
        seen.add(path)
        size = row.get("size")
        if size is None or int(size) < 0:
            raise ValueError(f"Video {path!r} has unknown size; refusing an unbounded download.")
        videos.append({"path": path, "size": int(size)})
        if len(videos) >= int(max_videos):
            break

    if not videos:
        raise ValueError(
            "No supported encoded video files found in the inspected inventory. "
            "Specify path_prefix (e.g. clips/P01) for large repositories, or verify "
            "that the dataset stores MP4/MOV files rather than Video features in Parquet."
        )
    total = sum(row["size"] for row in videos)
    if total > max_total_bytes:
        raise ValueError(
            f"Selected video download size {total:,} bytes exceeds maximum "
            f"{int(max_total_bytes):,}; select fewer videos or raise max_total_bytes."
        )
    return videos, truncated

def _video_record(repo_path: str, local_path: str, *, repo_id: str, revision: str) -> dict[str, Any]:
    name = PurePosixPath(repo_path).stem
    stable_hash = hashlib.sha256(repo_path.encode("utf-8")).hexdigest()[:12]
    return {
        "record_id": f"{_slug(name)}_{stable_hash}",
        "source_clip_id": name,
        "video_uri": str(Path(local_path).resolve()),
        "video_path": str(Path(local_path).resolve()),
        "video_media_kind": "video_file",
        "filename": PurePosixPath(repo_path).name,
        "hf_repo_path": repo_path,
        "hf_repo_id": repo_id,
        "hf_revision": revision,
    }

def import_repository_videos(
    context: Any,
    *,
    manager: Any,
    service: Any,
    repo_id: str,
    path_prefix: str = "",
    revision: str | None = None,
    token: str | None = None,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    max_videos: int = 10,
    max_total_bytes: int = DEFAULT_MAX_BYTES,
    max_inventory: int = DEFAULT_MAX_INVENTORY,
    set_active: bool = True,
    annotation_mode: str = "auto",
    cancel_token: Any = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Download a bounded set of encoded videos and register a Parquet manifest.

    This must be invoked as a job by the UI/action caller. The platform dataset
    registry is not modified until all selected files and the manifest exist.
    """
    from huggingface_hub import hf_hub_download
    from astronomicAL.platform.plugins.specs import ActionRequest

    repo_id = str(repo_id or "").strip()
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must be a Hugging Face namespace/repository identifier.")
    if manager is None or not callable(getattr(manager, "run_action", None)):
        raise RuntimeError("Import requires a PluginManager with run_action().")
    manager.get_action("core.video.register_manifest")

    annotation_mode = str(annotation_mode or "auto").strip().lower()
    if annotation_mode not in {"auto", "none", "epic_kitchens_100"}:
        raise ValueError("annotation_mode must be auto, none, or epic_kitchens_100")
    is_epic = repo_id == EPIC_KITCHENS_REPO_ID
    epic_annotations = None
    if is_epic and annotation_mode != "none":
        # Domain semantics live in the adapter; other repositories import without it.
        from . import epic_kitchens_annotations as epic_annotations
    if annotation_mode == "epic_kitchens_100" and not is_epic:
        raise ValueError("EPIC-KITCHENS annotations can only be used with the matching repository")
    use_epic_annotations = is_epic and annotation_mode != "none"

    resolved_token = (
        service.effective_token(token)
        if callable(getattr(service, "effective_token", None))
        else token
    )
    _check_cancel(cancel_token)
    sha = _pinned_revision(repo_id, revision=revision, token=resolved_token)
    files, truncated = select_repository_videos(
        service,
        repo_id=repo_id,
        revision=sha,
        token=resolved_token,
        path_prefix=path_prefix,
        max_videos=max_videos,
        max_inventory=max_inventory,
        max_total_bytes=max_total_bytes,
    )
    selection_identity: list[Any] = [sha, [row["path"] for row in files]]
    if use_epic_annotations:
        # Preserve existing IDs for all video-only imports, while avoiding a
        # collision with prior EPIC datasets that lacked action annotations.
        selection_identity.append(epic_annotations.SCHEMA)
    selected_hash = hashlib.sha256(json.dumps(
        selection_identity, separators=(",", ":")
    ).encode("utf-8")).hexdigest()[:14]
    dataset_id = _slug(dataset_id or f"hf_{repo_id}_{selected_hash}")
    dataset_name = str(dataset_name or f"HF videos: {repo_id}" +
                       (" · EPIC actions" if use_epic_annotations else ""))

    datasets = getattr(context, "datasets", None)
    if datasets is not None:
        try:
            if dataset_id in datasets.list_ids():
                meta = datasets.get_meta(dataset_id)
                if (meta.get("hf_repo_id") == repo_id
                        and meta.get("hf_revision") == sha
                        and meta.get("hf_selection_hash") == selected_hash):
                    if set_active:
                        datasets.set_active(dataset_id, origin="integrations.huggingface")
                    return {"repo_id": repo_id, "revision": sha, "dataset_id": dataset_id,
                            "dataset_name": dataset_name, "rows": len(files), "existing": True,
                            "registration_contract": "core.video.register_manifest",
                            "annotation_schema": meta.get("annotation_schema"),
                            "annotation_matches": meta.get("annotation_matches", 0),
                            "inventory_truncated": truncated}
                raise ValueError(f"Dataset ID {dataset_id!r} already belongs to another source.")
        except (AttributeError, TypeError):
            pass

    rows = []
    for index, item in enumerate(files, start=1):
        _check_cancel(cancel_token)
        path = item["path"]
        # Pass the resolved commit SHA explicitly: never fall back to a cache hit
        # against the floating branch or a different requested revision.
        local_path = hf_hub_download(
            repo_id=repo_id, filename=path, repo_type="dataset", revision=sha,
            token=resolved_token,
            cache_dir=(str(service.configured_cache_dir)
                       if getattr(service, "configured_cache_dir", None) else None),
        )
        _check_cancel(cancel_token)
        local_file = Path(local_path).resolve()
        if not local_file.is_file():
            raise FileNotFoundError(f"Cached Hugging Face video is missing: {path}")
        if local_file.stat().st_size != item["size"]:
            raise IOError(f"Cached video size does not match repository metadata: {path}")
        rows.append(_video_record(path, str(local_file), repo_id=repo_id, revision=sha))
        _progress(progress_callback, phase="video_download", completed=index, total=len(files),
                  message=f"Cached {index}/{len(files)} video files")

    annotation_result: dict[str, Any] = {}
    annotation_checksums: dict[str, str] = {}
    if use_epic_annotations:
        _progress(progress_callback, phase="annotation_download", completed=0, total=2,
                  message="Retrieving pinned EPIC-KITCHENS action CSVs")
        annotation_files = epic_annotations.fetch_annotation_files(
            repo_id, revision=sha, token=resolved_token,
            cache_dir=(str(service.configured_cache_dir)
                       if getattr(service, "configured_cache_dir", None) else None),
            cancel_token=cancel_token,
        )
        requested_ids = {
            clip_id for row in rows
            if (clip_id := epic_annotations.clip_annotation_id(row["hf_repo_path"]))
        }
        annotations, annotation_checksums = epic_annotations.load_relevant_annotations(
            annotation_files, requested_ids=requested_ids, cancel_token=cancel_token,
        )
        annotation_result = epic_annotations.enrich_video_rows(
            rows, annotations=annotations, strict=True,
        )
        _progress(progress_callback, phase="annotation_join", completed=len(rows), total=len(rows),
                  message=f"Matched {annotation_result['matched']} EPIC-KITCHENS actions")

    _check_cancel(cancel_token)
    frame = pd.DataFrame.from_records(rows)
    cache_dir = Path(".astronomical_cache/huggingface/video_manifests").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / f"{dataset_id}_{selected_hash}.parquet"
    fd, temp_name = tempfile.mkstemp(dir=str(cache_dir), prefix=".hf-video-", suffix=".parquet")
    os.close(fd)
    try:
        frame.to_parquet(temp_name, index=False)
        _check_cancel(cancel_token)
        os.replace(temp_name, manifest_path)
    finally:
        if Path(temp_name).exists():
            Path(temp_name).unlink()

    params = {
        "manifest_path": str(manifest_path),
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "columns": list(frame.columns),
        "row_count": len(frame),
        "mappings": {
            "record_id": "record_id", "video.uri": "video_uri",
            "video.path": "video_path", "video.media_kind": "video_media_kind",
            **({"target_label": "target_label"} if use_epic_annotations else {}),
        },
        "source_format": "huggingface_repository_video_manifest",
        "extra_metadata": {
            "loader_id": "integrations.huggingface",
            "hf_repo_id": repo_id, "hf_revision": sha,
            "hf_selection_hash": selected_hash,
            "hf_path_prefix": path_prefix,
            "hf_inventory_truncated": truncated,
            "hf_selected_files": len(files),
            "hf_selected_bytes": sum(item["size"] for item in files),
            "video_media_representation": "video_file",
            **({
                "annotation_schema": epic_annotations.SCHEMA,
                "annotation_files": list(epic_annotations.ANNOTATION_FILES),
                "annotation_file_sha256": annotation_checksums,
                "annotation_matches": annotation_result["matched"],
                "annotation_missing": annotation_result["missing"],
                "annotation_timebase": "original_full_length_recording",
            } if use_epic_annotations else {}),
        },
        "set_active": bool(set_active),
    }
    _check_cancel(cancel_token)
    result = manager.run_action("core.video.register_manifest", context, ActionRequest(params=params))
    return {
        **dict(result or {}),
        "repo_id": repo_id,
        "revision": sha,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "rows": len(frame),
        "existing": False,
        "annotation_schema": epic_annotations.SCHEMA if use_epic_annotations else None,
        "annotation_matches": annotation_result.get("matched", 0),
        "inventory_truncated": truncated,
        "manifest_path": str(manifest_path),
        "registration_contract": "core.video.register_manifest",
        "preview": frame.head(10),
    }
