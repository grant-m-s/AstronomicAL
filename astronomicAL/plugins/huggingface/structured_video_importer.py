from __future__ import annotations

"""Bridge cached YTVIS-like annotations into the public ``core.video`` contract.

The Hugging Face integration owns repository acquisition and format adaptation. It
writes a compact, importer-owned Parquet manifest and asks ``core.video`` to
register that manifest through a public action. Segmentation payloads remain in the
source annotation JSON for the later ``vision.segmentation`` sidecar tranche.
"""

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import quote

import pandas as pd

from .structured_annotations import HFStructuredAnnotationProfile

VIDEO_MANIFEST_SCHEMA_VERSION = 4
NOMINAL_FRAME_SEQUENCE_PLAYBACK_FPS = 5.0

def iter_top_level_array_items(
    path: str | Path,
    key: str,
    *,
    chunk_size: int = 1 << 20,
    cancel_token: Any = None,
) -> Iterator[dict[str, Any]]:
    """Stream JSON objects from one top-level array and stop at its closing bracket.

    This intentionally avoids ``json.load`` and never reads later top-level arrays
    such as a very large ``annotations`` payload after ``videos`` has completed.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if chunk_size < 4096:
        chunk_size = 4096

    decoder = json.JSONDecoder()
    wanted = str(key)
    buffer = ""
    offset = 0
    found_array = False
    eof = False

    with source.open("r", encoding="utf-8") as handle:
        while True:
            _check_cancelled(cancel_token)
            if not eof and (not found_array or len(buffer) - offset < chunk_size // 2):
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer = buffer[offset:] + chunk
                    offset = 0
                else:
                    buffer = buffer[offset:]
                    offset = 0
                    eof = True

            if not found_array:
                match = re.search(rf'"{re.escape(wanted)}"\s*:\s*\[', buffer[offset:])
                if match is None:
                    if eof:
                        raise KeyError(f"Top-level JSON array {wanted!r} was not found in {source}.")
                    # Retain overlap in case the key spans chunks.
                    retain = min(len(buffer), max(256, len(wanted) + 64))
                    buffer = buffer[-retain:]
                    offset = 0
                    continue
                offset += match.end()
                found_array = True

            while True:
                _check_cancelled(cancel_token)
                while offset < len(buffer) and buffer[offset].isspace():
                    offset += 1
                if offset >= len(buffer):
                    break
                if buffer[offset] == "]":
                    return
                if buffer[offset] == ",":
                    offset += 1
                    continue

                try:
                    value, end = decoder.raw_decode(buffer, offset)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(
                            f"Could not decode an item from top-level array {wanted!r} in {source}."
                        )
                    break
                offset = end
                if isinstance(value, Mapping):
                    yield dict(value)

            if eof:
                raise ValueError(f"Top-level array {wanted!r} did not terminate in {source}.")

def build_ytvis_video_manifest(
    annotation_path: str | Path,
    *,
    repo_id: str,
    revision: str,
    split: str,
    annotation_file: str,
    annotation_variant: str,
    frame_media_root: str,
    annotation_fps: float,
    max_videos: int = 0,
    cancel_token: Any = None,
    progress_callback: Any = None,
) -> pd.DataFrame:
    """Create one compact row per annotated video without reading masklets."""
    root = _normalise_media_root(frame_media_root)
    fps = float(annotation_fps or 0.0)
    if fps <= 0:
        raise ValueError("A positive annotation-frame FPS is required for frame-sequence video import.")

    rows: list[dict[str, Any]] = []
    for index, video in enumerate(
        iter_top_level_array_items(annotation_path, "videos", cancel_token=cancel_token)
    ):
        _check_cancelled(cancel_token)
        if max_videos and len(rows) >= int(max_videos):
            break
        row = _video_row(
            video,
            repo_id=repo_id,
            revision=revision,
            split=split,
            annotation_file=annotation_file,
            annotation_variant=annotation_variant,
            frame_media_root=root,
            annotation_fps=fps,
        )
        rows.append(row)
        if callable(progress_callback) and (index == 0 or (index + 1) % 100 == 0):
            progress_callback(
                {
                    "phase": "video_manifest",
                    "completed": index + 1,
                    "total": 0,
                    "percent": -1,
                    "progress_known": False,
                    "progress_basis": "streaming_rows",
                    "message": f"Streaming video metadata: {index + 1:,} video(s)…",
                }
            )

    if not rows:
        raise ValueError(f"No video records were found in {annotation_path}.")
    frame = pd.DataFrame.from_records(rows)
    return frame

def import_structured_video_dataset(
    context: Any,
    *,
    manager: Any,
    service: Any,
    repo_id: str,
    splits: Sequence[str],
    variant: str = "auto",
    revision: str | None = None,
    token: str | None = None,
    profile: Mapping[str, Any] | None = None,
    dataset_id_base: str | None = None,
    dataset_name_base: str | None = None,
    media_root_override: str | None = None,
    annotation_fps: float | None = None,
    max_videos: int = 0,
    active_split: str | None = None,
    cancel_token: Any = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Acquire selected annotations, build manifests, and call ``core.video`` publicly."""
    if manager is None or not callable(getattr(manager, "run_action", None)):
        raise RuntimeError(
            "Structured video import requires PluginManager so integrations.huggingface can "
            "invoke the public core.video.register_manifest action."
        )
    _require_core_video_action(manager)

    profile_obj = (
        HFStructuredAnnotationProfile.from_dict(profile)
        if profile
        else None
    )
    if profile_obj is None:
        details = service.get_structured_annotation_details(
            repo_id,
            token=token,
            revision=revision,
        )
        if not details.structured_annotation_profile:
            raise ValueError("No supported structured video annotation profile was detected.")
        profile_obj = HFStructuredAnnotationProfile.from_dict(details.structured_annotation_profile)

    cache_result = service.cache_structured_annotations(
        repo_id,
        splits=list(splits),
        variant=variant,
        revision=revision,
        token=token,
        max_workers=4,
        cancel_token=cancel_token,
        progress_callback=progress_callback,
        profile=profile_obj.to_dict(),
    )
    resolved_revision = str(cache_result.get("revision") or revision or "")
    selected_meta = list(cache_result.get("annotation_files") or [])
    local_files = list(cache_result.get("local_files") or [])
    if len(selected_meta) != len(local_files):
        raise RuntimeError("Structured annotation cache returned mismatched metadata/local files.")

    base_id = _slugify(dataset_id_base or repo_id.replace("/", "_"), fallback="hf_video")
    base_name = str(dataset_name_base or repo_id)
    inferred_roots = dict(profile_obj.frame_media_roots_by_split or {})

    # The browser's frame-root field is an override, not a prerequisite. Re-resolve
    # any missing split roots inside the managed import job so stale inspection
    # state or a nested Hub layout cannot prevent a valid structured import.
    if not str(media_root_override or "").strip():
        missing_root_splits = [
            str(split)
            for split in splits
            if not str(inferred_roots.get(str(split)) or "").strip()
        ]
        resolver = getattr(service, "resolve_structured_frame_media_roots", None)
        if missing_root_splits and callable(resolver):
            resolved_roots = resolver(
                repo_id,
                profile=profile_obj.to_dict(),
                splits=missing_root_splits,
                revision=resolved_revision or revision,
                token=token,
            )
            inferred_roots.update(
                {
                    str(key): str(value)
                    for key, value in dict(resolved_roots or {}).items()
                    if str(key).strip() and str(value).strip()
                }
            )

    inferred_fps = profile_obj.frame_media_fps
    explicit_fps = _positive_float(annotation_fps)
    profile_fps = _positive_float(inferred_fps)
    if explicit_fps is not None:
        effective_fps = explicit_fps
        fps_source = "explicit_override"
        fps_estimated = False
    elif profile_fps is not None:
        effective_fps = profile_fps
        fps_source = "structured_profile"
        fps_estimated = False
    else:
        effective_fps = float(NOMINAL_FRAME_SEQUENCE_PLAYBACK_FPS)
        fps_source = "nominal_playback_default"
        fps_estimated = True
    results: list[dict[str, Any]] = []
    previews: list[pd.DataFrame] = []

    requested_active_split = str(active_split or "").strip()

    datasets = getattr(context, "datasets", None)
    initial_active_dataset_id = ""
    if datasets is not None:
        try:
            initial_active_dataset_id = str(
                datasets.active_id() or ""
            ).strip()
        except Exception:
            initial_active_dataset_id = ""

    for index, (metadata, local_path) in enumerate(zip(selected_meta, local_files)):
        _check_cancelled(cancel_token)
        split = str(metadata.get("split") or "").strip() or f"split_{index + 1}"
        root = str(media_root_override or inferred_roots.get(split) or "").strip()
        if not root:
            raise ValueError(
                f"AstronomicAL could not resolve frame media for split {split!r} from the pinned "
                "Hugging Face repository. The optional frame-media root override can be used for "
                "repositories whose media lives elsewhere."
            )
        fps = float(effective_fps)

        frame = build_ytvis_video_manifest(
            local_path,
            repo_id=repo_id,
            revision=resolved_revision,
            split=split,
            annotation_file=str(metadata.get("path") or ""),
            annotation_variant=str(metadata.get("variant") or variant),
            frame_media_root=root,
            annotation_fps=fps,
            max_videos=max_videos,
            cancel_token=cancel_token,
            progress_callback=_split_progress(progress_callback, index, len(local_files), split),
        )
        # Persist the exact local retained-annotation identity into the manifest rows.
        # Dataset metadata can outlive/reconcile across stable dataset IDs, whereas
        # these columns travel with the exact importer-owned Parquet source.  The
        # segmentation plugin can therefore resolve the annotation payload that
        # actually produced this manifest without importing Hugging Face internals.
        annotation_local_path = Path(local_path).expanduser().absolute()
        snapshot_root_path = Path(
            str(cache_result.get("snapshot_root") or annotation_local_path.parent)
        ).expanduser().absolute()
        annotation_local_uri = annotation_local_path.as_uri()
        snapshot_root_uri = snapshot_root_path.as_uri()
        frame["segmentation_annotation_uri"] = annotation_local_uri
        frame["segmentation_source_root_uri"] = snapshot_root_uri
        frame["segmentation_source_format"] = profile_obj.format_id
        dataset_id = _slugify(f"{base_id}_{split}", fallback=f"hf_video_{index + 1}")
        dataset_name = f"{base_name} / {split}"
        manifest_path = _write_manifest(frame, dataset_id=dataset_id, revision=resolved_revision)
        mappings = _video_mappings(frame.columns)
        set_active = bool(
            requested_active_split
            and requested_active_split == split
        )
        annotation_source = {
            "format_id": profile_obj.format_id,
            "uri": annotation_local_uri,
            "root_uri": snapshot_root_uri,
            "repo_id": repo_id,
            "revision": resolved_revision,
            "split": split,
            "annotation_file": str(metadata.get("path") or ""),
            "annotation_variant": str(metadata.get("variant") or variant),
            "record_id_column": "record_id",
            "video_source_id_column": "video_source_id",
            "timeline_fps": fps,
            "timeline_fps_source": fps_source,
            "timeline_fps_estimated": bool(fps_estimated),
        }
        params = {
            "manifest_path": str(manifest_path),
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "columns": [str(column) for column in frame.columns],
            "row_count": int(len(frame)),
            "mappings": mappings,
            "source_format": "huggingface_ytvis_video_manifest",
            "extra_metadata": {
                "loader_id": "integrations.huggingface",
                "hf_dataset_id": repo_id,
                "hf_revision": resolved_revision,
                "hf_split": split,
                "hf_annotation_file": str(metadata.get("path") or ""),
                "hf_annotation_variant": str(metadata.get("variant") or variant),
                "hf_structured_format": profile_obj.format_id,
                "video_media_representation": "frame_sequence",
                "annotation_frame_fps": fps,
                "annotation_frame_fps_source": fps_source,
                "annotation_frame_fps_estimated": bool(fps_estimated),
                "external_frame_media_root": root,
                "segmentation_annotations_retained_external": True,
                "segmentation_annotation_source": dict(annotation_source),
                "segmentation_annotation_sources": [dict(annotation_source)],
                "structured_video_manifest_schema_version": VIDEO_MANIFEST_SCHEMA_VERSION,
            },
            "set_active": set_active,
        }
        from astronomicAL.platform.plugins.specs import ActionRequest

        result = manager.run_action(
            "core.video.register_manifest",
            context,
            ActionRequest(params=params),
        )
        result = dict(result or {})
        result.update(
            {
                "split": split,
                "repo_id": repo_id,
                "revision": resolved_revision,
                "annotation_file": str(metadata.get("path") or ""),
                "annotation_variant": str(metadata.get("variant") or variant),
                "frame_media_root": root,
                "annotation_fps": fps,
                "annotation_fps_source": fps_source,
                "annotation_fps_estimated": bool(fps_estimated),
                "segmentation_annotation_source": dict(annotation_source),
            }
        )
        results.append(result)
        previews.append(frame.head(10))

    preview = (
        pd.concat(previews, ignore_index=True)
        if previews
        else pd.DataFrame()
    )

    effective_active_split = requested_active_split

    if datasets is not None:
        target_dataset_id = ""

        if requested_active_split:
            for item in results:
                if (
                    str(item.get("split") or "").strip()
                    == requested_active_split
                ):
                    target_dataset_id = str(
                        item.get("dataset_id") or ""
                    ).strip()
                    if target_dataset_id:
                        break
        elif initial_active_dataset_id:
            # "Do not change active dataset" means restore the dataset that was
            # active before registration. This also undoes any pending workspace
            # restore that happened to be applied while a later split registered.
            try:
                if initial_active_dataset_id in {
                    str(value)
                    for value in datasets.list_ids()
                }:
                    target_dataset_id = initial_active_dataset_id
            except Exception:
                target_dataset_id = ""

        if target_dataset_id:
            try:
                current_dataset_id = str(
                    datasets.active_id() or ""
                ).strip()
            except Exception:
                current_dataset_id = ""

            if current_dataset_id != target_dataset_id:
                datasets.set_active(
                    target_dataset_id,
                    origin="integrations.huggingface",
                )

    return {
        "repo_id": repo_id,
        "revision": resolved_revision,
        "format_id": profile_obj.format_id,
        "results": results,
        "active_split": effective_active_split,
        "rows": sum(int(item.get("rows") or 0) for item in results),
        "preview": preview,
        "snapshot_root": cache_result.get("snapshot_root"),
        "annotation_files": selected_meta,
        "registered": True,
        "registration_contract": "core.video.register_manifest",
        "segmentation_imported": False,
    }

def _video_row(
    video: Mapping[str, Any],
    *,
    repo_id: str,
    revision: str,
    split: str,
    annotation_file: str,
    annotation_variant: str,
    frame_media_root: str,
    annotation_fps: float,
) -> dict[str, Any]:
    file_names = [str(value) for value in (video.get("file_names") or ()) if str(value).strip()]
    if not file_names:
        raise ValueError(f"Video {video.get('id')!r} has no frame file_names.")
    pattern = infer_frame_sequence(file_names)
    first_name = file_names[0]
    video_name = str(video.get("video_name") or "").strip()
    raw_id = video.get("id")
    record_id = video_name or f"video_{raw_id}"
    # ``file_names`` is the authoritative logical annotation timeline.  A source
    # video may have additional unsampled frames; those belong in source_* metadata.
    frame_count = len(file_names)
    source_frame_count = _positive_int(video.get("video_num_frames"))
    source_fps = _positive_float(video.get("video_fps"))
    width = _positive_int(video.get("width"))
    height = _positive_int(video.get("height"))
    frame_dir = str(pattern.get("parent") or "")
    video_uri = _join_media_uri(frame_media_root, frame_dir, directory=True)
    thumbnail_uri = _join_media_uri(frame_media_root, first_name)

    explicit_names = list(pattern.get("frame_names") or [])
    if explicit_names:
        # Irregular/dropped frame numbering cannot be represented by one
        # offset+stride template. Preserve the annotation's exact ordered frame
        # identity instead; core.video resolves logical frame i to this URI i.
        template_uri = ""
        explicit_uris = [
            _join_media_uri(frame_media_root, name)
            for name in explicit_names
        ]
        frame_offset = 0
        frame_step = 1
    else:
        frame_token = (
            f"{{frame_index:0{pattern['width']}d}}"
            if pattern["width"] > 0
            else "{frame_index}"
        )
        template_name = f"{pattern['prefix']}{frame_token}{pattern['suffix']}"
        template_rel = str(PurePosixPath(frame_dir) / template_name) if frame_dir else template_name
        template_uri = _join_media_uri(frame_media_root, template_rel, preserve_template=True)
        explicit_uris = None
        frame_offset = int(pattern["offset"])
        frame_step = int(pattern["step"])
    duration = frame_count / annotation_fps if annotation_fps > 0 else None
    source_duration = (
        source_frame_count / source_fps
        if source_frame_count and source_fps
        else None
    )
    return {
        "record_id": record_id,
        "video_source_id": raw_id,
        "video_name": video_name or record_id,
        "video_uri": video_uri,
        "video_media_kind": "frame_sequence",
        "video_frame_uri_template": template_uri,
        "video_frame_uris": explicit_uris,
        "video_frame_index_offset": frame_offset,
        "video_frame_index_step": frame_step,
        "video_thumbnail_uri": thumbnail_uri,
        "video_fps": float(annotation_fps),
        "video_frame_count": int(frame_count),
        "video_duration": duration,
        "video_width": width,
        "video_height": height,
        "source_video_fps": source_fps,
        "source_video_frame_count": source_frame_count,
        "source_video_duration": source_duration,
        "video_creation_datetime": _safe_scalar(video.get("video_creation_datetime")),
        "location_id": _safe_scalar(video.get("location_id")),
        "hf_repo_id": repo_id,
        "hf_revision": revision,
        "hf_split": split,
        "hf_annotation_file": annotation_file,
        "hf_annotation_variant": annotation_variant,
        "annotation_frame_count": int(frame_count),
        "annotation_frame_fps": float(annotation_fps),
    }

def infer_frame_sequence(file_names: Sequence[str]) -> dict[str, Any]:
    """Describe a YTVIS frame sequence without inventing missing source frames.

    Regular numeric filenames use the compact public template contract, including a
    positive source-frame stride when applicable.  If the annotation's exact
    ``file_names`` are not one arithmetic numeric sequence, return an explicit
    ordered frame-name list instead.  That fallback preserves logical annotation
    frame identity for dropped/irregular source numbering without materialising
    media or guessing filenames.
    """
    clean = [str(value) for value in file_names if str(value).strip()]
    if not clean:
        raise ValueError("Frame sequence is empty.")

    parsed: list[tuple[int, str, str, int, int, str]] = []
    parseable = True
    for source_index, value in enumerate(clean):
        path = PurePosixPath(value)
        match = re.match(r"^(.*?)(\d+)(\D*)$", path.name)
        if not match:
            parseable = False
            break
        prefix, digits, suffix = match.groups()
        parsed.append(
            (
                source_index,
                str(path.parent) if str(path.parent) != "." else "",
                prefix,
                int(digits),
                len(digits),
                suffix,
            )
        )

    if parseable and parsed:
        parents = {item[1] for item in parsed}
        prefixes = {item[2] for item in parsed}
        widths = {item[4] for item in parsed}
        suffixes = {item[5] for item in parsed}
        stable_shape = (
            len(parents) == 1
            and len(prefixes) == 1
            and len(widths) == 1
            and len(suffixes) == 1
        )
        if stable_shape:
            offset = parsed[0][3]
            step = 1
            if len(parsed) > 1:
                step = parsed[1][3] - offset
            if step > 0 and all(
                number == offset + source_index * step
                for source_index, _parent, _prefix, number, _width, _suffix in parsed
            ):
                return {
                    "parent": parsed[0][1],
                    "prefix": parsed[0][2],
                    "offset": offset,
                    "step": step,
                    "width": parsed[0][4],
                    "suffix": parsed[0][5],
                }

    parents = {
        str(PurePosixPath(value).parent)
        if str(PurePosixPath(value).parent) != "."
        else ""
        for value in clean
    }
    common_parent = next(iter(parents)) if len(parents) == 1 else ""
    return {
        "parent": common_parent,
        "frame_names": clean,
    }

def _write_manifest(frame: pd.DataFrame, *, dataset_id: str, revision: str) -> Path:
    """Write a manifest whose cache identity includes media interpretation.

    A repository revision alone is not sufficient identity for an importer-owned
    manifest: frame-root inference and importer schema can improve without the
    upstream annotation revision changing. Derive a short fingerprint from the
    resolved media columns so a corrected root cannot silently reuse the path of
    a previously registered, semantically different manifest.
    """
    revision_token = _slugify(revision[:16] or "unresolved", fallback="unresolved")
    identity_parts = [
        f"schema={VIDEO_MANIFEST_SCHEMA_VERSION}",
        f"revision={revision}",
    ]
    if not frame.empty:
        first = frame.iloc[0]
        for column in (
            "video_uri",
            "video_frame_uri_template",
            "video_frame_index_step",
            "video_thumbnail_uri",
            "video_fps",
            "hf_annotation_file",
            "hf_annotation_variant",
            "segmentation_annotation_uri",
            "segmentation_source_root_uri",
        ):
            if column in frame.columns:
                identity_parts.append(f"{column}={first.get(column)}")
        if "video_frame_uris" in frame.columns:
            raw_uris = first.get("video_frame_uris")
            if hasattr(raw_uris, "tolist") and not isinstance(raw_uris, (str, bytes)):
                try:
                    raw_uris = raw_uris.tolist()
                except Exception:
                    pass
            if isinstance(raw_uris, (list, tuple)) and raw_uris:
                explicit_fingerprint = hashlib.sha256(
                    "\n".join(str(value) for value in raw_uris).encode("utf-8")
                ).hexdigest()
                identity_parts.append(f"video_frame_uris_sha256={explicit_fingerprint}")
    fingerprint = hashlib.sha256("\n".join(identity_parts).encode("utf-8")).hexdigest()[:12]

    root = Path(".astronomical_cache/datasets/huggingface_structured_video").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / (
        f"{_slugify(dataset_id)}_{revision_token}_v{VIDEO_MANIFEST_SCHEMA_VERSION}_{fingerprint}.parquet"
    )
    frame.to_parquet(path, index=False)
    return path.resolve()

def _video_mappings(columns: Iterable[str]) -> dict[str, str]:
    cols = {str(value) for value in columns}
    candidates = {
        "record_id": "record_id",
        "video.uri": "video_uri",
        "video.thumbnail_uri": "video_thumbnail_uri",
        "video.duration": "video_duration",
        "video.fps": "video_fps",
        "video.frame_count": "video_frame_count",
        "video.width": "video_width",
        "video.height": "video_height",
        "video.media_kind": "video_media_kind",
        "video.frame_uri_template": "video_frame_uri_template",
        "video.frame_uris": "video_frame_uris",
        "video.frame_index_offset": "video_frame_index_offset",
        "video.frame_index_step": "video_frame_index_step",
    }
    return {semantic: column for semantic, column in candidates.items() if column in cols}

def _require_core_video_action(manager: Any) -> None:
    getter = getattr(manager, "get_action", None)
    if callable(getter):
        try:
            getter("core.video.register_manifest")
            return
        except Exception as exc:
            raise RuntimeError(
                "Enable core.video >= 0.2.17 before importing structured video datasets."
            ) from exc

def _split_progress(callback: Any, index: int, total: int, split: str):
    if not callable(callback):
        return None

    def emit(payload: Mapping[str, Any]) -> None:
        data = dict(payload or {})
        data["phase"] = f"{data.get('phase', 'video_manifest')}:{split}"
        data["message"] = f"Split {index + 1}/{max(total, 1)} ({split}): {data.get('message', '')}"
        callback(data)

    return emit

def _normalise_media_root(value: str) -> str:
    root = str(value or "").strip()
    if not root:
        raise ValueError("frame_media_root is required.")
    return root.rstrip("/") + "/"

def _join_media_uri(root: str, relative: str, *, directory: bool = False, preserve_template: bool = False) -> str:
    parts = []
    for part in PurePosixPath(str(relative or "")).parts:
        if preserve_template and "{frame_index" in part:
            # Encode literal filename text while preserving the formatting token
            # consumed later by core.video.  This matters now that a valid frame
            # sequence can contain a prefix such as ``camera 1_``.
            token_match = re.search(r"\{frame_index(?::[^}]*)?\}", part)
            if token_match is None:
                parts.append(quote(part, safe="-._~"))
            else:
                token = token_match.group(0)
                sentinel = "__AL_FRAME_INDEX_TEMPLATE__"
                encoded = quote(part.replace(token, sentinel, 1), safe="-._~")
                parts.append(encoded.replace(sentinel, token, 1))
        else:
            parts.append(quote(part, safe="-._~"))
    result = root.rstrip("/") + "/" + "/".join(parts)
    if directory:
        result = result.rstrip("/") + "/"
    return result

def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except Exception:
        return None
    return result if result > 0 else None

def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) and result > 0 else None

def _safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        try:
            if isinstance(value, float) and math.isnan(value):
                return None
        except Exception:
            pass
        return value
    return str(value)

def _slugify(value: str, *, fallback: str = "dataset") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "")).strip("._-")
    return slug or fallback

def _check_cancelled(cancel_token: Any) -> None:
    if cancel_token is None:
        return
    method = getattr(cancel_token, "raise_if_cancelled", None)
    if callable(method):
        method()
        return
    cancelled = getattr(cancel_token, "cancelled", None)
    if callable(cancelled) and cancelled():
        raise RuntimeError("Structured video import was cancelled.")
    if isinstance(cancelled, bool) and cancelled:
        raise RuntimeError("Structured video import was cancelled.")

__all__ = [
    "VIDEO_MANIFEST_SCHEMA_VERSION",
    "NOMINAL_FRAME_SEQUENCE_PLAYBACK_FPS",
    "build_ytvis_video_manifest",
    "import_structured_video_dataset",
    "infer_frame_sequence",
    "iter_top_level_array_items",
]
