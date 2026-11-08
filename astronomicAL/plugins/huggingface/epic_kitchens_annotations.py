from __future__ import annotations

"""EPIC-KITCHENS-100 clip-level action annotation adapter.

Importers stay generic: only this module knows the published EPIC-KITCHENS CSV
columns and the `clips/{participant}/{narration_id}.mp4` convention. Times and
frame indices refer to the *source recording*, not the extracted MP4 clip.
"""

import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

REPO_ID = "lightly-ai/epic-kitchens-100-clips"
SCHEMA = "epic_kitchens_100_actions_v1"
ANNOTATION_FILES = (
    "epic-kitchens-100-annotations/EPIC_100_train.csv",
    "epic-kitchens-100-annotations/EPIC_100_validation.csv",
)
CLIP_RE = re.compile(r"^clips/(P\d{2,3})/(P\d{2,3}_\d+_\d+)\.mp4$", re.I)

# Stable Parquet schema even when some selected clips have no matching label.
ANNOTATION_COLUMNS = (
    "annotation_status", "annotation_split", "action_narration_id",
    "participant_id", "original_video_id", "narration", "verb", "noun",
    "all_nouns", "verb_class", "noun_class", "all_noun_classes",
    "original_narration_timestamp", "original_start_timestamp",
    "original_stop_timestamp", "original_start_seconds", "original_stop_seconds",
    "original_start_frame", "original_stop_frame", "target_label",
)

def _source_seconds(value: str) -> float | None:
    """Parse source-recording clock times; never interpret as clip-local times."""
    value = str(value or "").strip()
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid EPIC-KITCHENS timestamp: {value!r}")
    hours, minutes, seconds = parts
    if not (hours.isdigit() and minutes.isdigit() and re.fullmatch(r"\d+(?:\.\d+)?", seconds)):
        raise ValueError(f"Invalid EPIC-KITCHENS timestamp: {value!r}")
    m, s = int(minutes), float(seconds)
    if m >= 60 or s >= 60:
        raise ValueError(f"Invalid EPIC-KITCHENS timestamp: {value!r}")
    return int(hours) * 3600.0 + m * 60.0 + s

def _json_array(value: str) -> str:
    """Keep lists as JSON strings for a stable, portable Parquet column type."""
    import ast
    text = str(value or "").strip()
    if not text:
        return "[]"
    try:
        result = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid EPIC-KITCHENS list: {text[:80]!r}") from exc
    if not isinstance(result, (tuple, list)):
        raise ValueError("EPIC-KITCHENS list annotation is not an array")
    return json.dumps(list(result), ensure_ascii=False, separators=(",", ":"))

def _optional_int(value: str) -> int | None:
    text = str(value or "").strip()
    return int(text) if text else None

def clip_annotation_id(path: str) -> str | None:
    normalized = str(PurePosixPath(str(path or "")))
    match = CLIP_RE.fullmatch(normalized)
    if match and match.group(2).startswith(match.group(1) + "_"):
        return match.group(2)
    return None

def load_relevant_annotations(
    paths: Mapping[str, str | Path], *, requested_ids: set[str],
    cancel_token: Any = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Scan both published CSVs but retain only the requested clip annotations.

    Reject duplicates across annotation files and malformed matching records.
    No annotation CSV values are executed or interpreted as Python expressions
    beyond bounded literal list parsing.
    """
    found: dict[str, dict[str, Any]] = {}
    checksums: dict[str, str] = {}
    for name in ANNOTATION_FILES:
        path = Path(paths[name])
        checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"narration_id", "participant_id", "video_id", "start_timestamp", "stop_timestamp", "narration"}
            if not required.issubset(set(reader.fieldnames or ())):
                raise ValueError(f"Unexpected EPIC-KITCHENS CSV schema in {name}")
            split = "train" if name.endswith("_train.csv") else "validation"
            for i, item in enumerate(reader):
                if i % 2048 == 0:
                    _check_cancel(cancel_token)
                narration_id = str(item.get("narration_id") or "").strip()
                if narration_id not in requested_ids:
                    continue
                if narration_id in found:
                    raise ValueError(f"Duplicate EPIC-KITCHENS narration_id {narration_id!r}")
                start = _source_seconds(str(item.get("start_timestamp") or ""))
                stop = _source_seconds(str(item.get("stop_timestamp") or ""))
                if start is None or stop is None or stop < start:
                    raise ValueError(f"Invalid action time interval for {narration_id}")
                found[narration_id] = {
                    "annotation_status": "matched",
                    "annotation_split": split,
                    "action_narration_id": narration_id,
                    "participant_id": str(item.get("participant_id") or ""),
                    "original_video_id": str(item.get("video_id") or ""),
                    "narration": str(item.get("narration") or ""),
                    "verb": str(item.get("verb") or ""),
                    "noun": str(item.get("noun") or ""),
                    "all_nouns": _json_array(str(item.get("all_nouns") or "")),
                    "verb_class": _optional_int(str(item.get("verb_class") or "")),
                    "noun_class": _optional_int(str(item.get("noun_class") or "")),
                    "all_noun_classes": _json_array(str(item.get("all_noun_classes") or "")),
                    "original_narration_timestamp": str(item.get("narration_timestamp") or ""),
                    "original_start_timestamp": str(item.get("start_timestamp") or ""),
                    "original_stop_timestamp": str(item.get("stop_timestamp") or ""),
                    "original_start_seconds": start,
                    "original_stop_seconds": stop,
                    "original_start_frame": _optional_int(str(item.get("start_frame") or "")),
                    "original_stop_frame": _optional_int(str(item.get("stop_frame") or "")),
                    "target_label": str(item.get("verb") or "") or None,
                }
    return found, checksums

def enrich_video_rows(
    rows: list[dict[str, Any]], *,
    annotations: Mapping[str, Mapping[str, Any]],
    strict: bool = True,
) -> dict[str, Any]:
    """Modify new manifest rows in place; existing registered datasets stay immutable."""
    matched = 0
    missing: list[str] = []
    for row in rows:
        name = clip_annotation_id(str(row.get("hf_repo_path") or ""))
        annotation = annotations.get(name or "")
        row.update({key: None for key in ANNOTATION_COLUMNS})
        if annotation is None:
            row["annotation_status"] = "not_found"
            missing.append(str(row.get("source_clip_id") or row.get("hf_repo_path") or ""))
            continue
        row.update({key: annotation.get(key) for key in ANNOTATION_COLUMNS})
        matched += 1
    if missing and strict:
        raise ValueError(
            f"Could not match {len(missing)} EPIC-KITCHENS clip(s) to the pinned train/validation "
            f"annotation CSVs (first five: {missing[:5]!r}). Re-import with annotation_mode='none' "
            "to load unannotated media."
        )
    return {"matched": matched, "missing": len(missing), "missing_sample": missing[:10]}

def _check_cancel(cancel_token: Any) -> None:
    if cancel_token is None:
        return
    cancelled = getattr(cancel_token, "cancelled", False)
    if bool(cancelled() if callable(cancelled) else cancelled):
        raise RuntimeError("Hugging Face video annotation import cancelled")

def fetch_annotation_files(
    repo_id: str, *, revision: str, token: str | None,
    cache_dir: str | None, cancel_token: Any = None,
    download: Callable[..., str] | None = None,
) -> dict[str, str]:
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    files = {}
    for filename in ANNOTATION_FILES:
        _check_cancel(cancel_token)
        files[filename] = download(repo_id=repo_id, filename=filename, repo_type="dataset",
                                   revision=revision, token=token, cache_dir=cache_dir)
    _check_cancel(cancel_token)
    return files
