from __future__ import annotations

import base64
import copy
import hashlib
import html
import json
import logging
import os
import re
import traceback
import time
import subprocess
import sys
import queue
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any

from collections import Counter

from .fiftyone_format import (
    FiftyOneSampleIndex, FiftyOneSchema, build_classification_sample_index, parse_fiftyone_schema,
)
from .structured_annotations import (
    HFStructuredAnnotationProfile, detect_structured_annotation_profile, structured_annotation_candidates,
)

import threading
from contextlib import contextmanager

import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_TASK_FILTER = "image-classification"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

SPLIT_ALIASES = {
    "train": "train", "training": "train", "tr": "train", "val": "validation", "valid": "validation",
    "validation": "validation", "dev": "validation", "test": "test", "testing": "test", "eval": "test",
    "evaluation": "test",
}

GENERIC_FOLDERS = {
    "", ".", "data", "dataset", "datasets", "images", "image", "imgs", "jpg", "jpeg", "png", "files",
    "file", "raw", "samples", "sample", "records", "record", "examples", "example", "train", "training",
    "validation", "valid", "val", "test", "testing",
}

SNAPSHOT_DRY_RUN_MAX_FILES = 128
ANONYMOUS_SNAPSHOT_MAX_WORKERS = 4
AUTHENTICATED_SNAPSHOT_MAX_WORKERS = 8
FIFTYONE_METADATA_FILENAME = "metadata.json"
FIFTYONE_SAMPLES_FILENAME = "samples.json"
FIFTYONE_SAMPLES_MAX_BYTES = 128 * 1024 * 1024
MAX_RATE_LIMIT_RETRIES = 5
REPOSITORY_INSPECTION_MAX_FILES = 50_000
REPOSITORY_INSPECTION_SAMPLE_FILES = 512
STRUCTURED_ANNOTATION_PROBE_MAX_BYTES = 4 * 1024 * 1024
STRUCTURED_ANNOTATION_PROBE_MAX_FILES = 3
STRUCTURED_ANNOTATION_ROOT_SCAN_MAX_FILES = 3000
STRUCTURED_ANNOTATION_DIRECTORY_SCAN_MAX_FILES = 512
STRUCTURED_MEDIA_DIRECTORY_SCAN_MAX_FILES = 96

GENERIC_FOLDER_PATTERNS = [
    r"^data[_-]?\d+$", r"^part[_-]?\d+$", r"^chunk[_-]?\d+$", r"^shard[_-]?\d+$", r"^batch[_-]?\d+$",
    r"^images?[_-]?\d+$", r"^files?[_-]?\d+$", r"^records?[_-]?\d+$", r"^samples?[_-]?\d+$",
]

class SilentTqdm:
    """Suppress tqdm output while satisfying the concurrent snapshot-download progress interface."""

    _lock = threading.RLock()

    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total", None)
        self.n = 0
        self.disable = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def __iter__(self):
        return iter(())

    def update(self, n: int = 1) -> None:
        self.n += n

    def close(self) -> None:
        pass

    def clear(self, *args, **kwargs) -> None:
        pass

    def reset(self, *args, **kwargs) -> None:
        self.n = 0

    def refresh(self, *args, **kwargs) -> None:
        pass

    def set_description(self, *args, **kwargs) -> None:
        pass

    def set_description_str(self, *args, **kwargs) -> None:
        pass

    def set_postfix(self, *args, **kwargs) -> None:
        pass

    def set_postfix_str(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def get_lock(cls):
        return cls._lock

    @classmethod
    def set_lock(cls, lock) -> None:
        cls._lock = lock

    @classmethod
    def write(cls, *args, **kwargs) -> None:
        pass

    @classmethod
    @contextmanager
    def external_write_mode(cls, *args, **kwargs):
        yield

class HFTransferCancelled(RuntimeError):
    """Raised when an AstronomicAL cancellation request stops a Hub transfer."""

class HFIncompleteSnapshotError(RuntimeError):
    """Raised when Hugging Face returns a snapshot missing requested files."""

    def __init__(
        self, message: str, *, snapshot_path: str = "", missing_paths: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.snapshot_path = str(snapshot_path or "")
        self.missing_paths = list(missing_paths or [])

class HFRateLimitBackoff(RuntimeError):
    """Internal signal used to replace huggingface_hub's blocking 429 sleep."""

    def __init__(self, retry_after_seconds: float, message: str = "") -> None:
        self.retry_after_seconds = max(float(retry_after_seconds or 0.0), 0.0)
        super().__init__(message or f"Hugging Face rate limited; retry after {self.retry_after_seconds:.0f}s")

class _HFRateLimitInterceptor(logging.Handler):
    """Observe Hub 429 logs for one repository and abort before its blocking sleep."""

    def __init__(self, *, repo_id: str, progress_callback: Any = None) -> None:
        super().__init__(level=logging.WARNING)
        self.repo_id = str(repo_id or "")
        self.progress_callback = progress_callback
        self._rate_limited_threads: dict[int, float] = {}

    def _notify(self, *, retry_after_seconds: float | None, raw_message: str) -> None:
        if not callable(self.progress_callback):
            return
        seconds = None if retry_after_seconds is None else max(int(round(retry_after_seconds)), 0)
        if seconds is None:
            message = (
                "Hugging Face rate limit reached. Waiting for retry information; "
                "add an HF token to increase limits or cancel this import."
            )
        else:
            message = (
                f"Hugging Face rate limit reached. Retry window: {seconds}s. "
                "You can cancel now; adding an HF token usually increases the available quota."
            )
        try:
            self.progress_callback(
                {
                    "phase": "rate_limited", "completed": 0, "total": 0, "percent": 10, "message": message,
                    "current_file": "", "severity": "warning", "retry_after_seconds": seconds,
                    "raw_message": raw_message,
                }
            )
        except Exception:
            pass

    def emit(self, record: logging.LogRecord) -> None:
        message = str(record.getMessage() or "")
        thread_id = int(getattr(record, "thread", 0) or 0)
        now = time.monotonic()

        if "429" in message and ("HTTP Error" in message or "Too Many Requests" in message):
            raw_repo = self.repo_id
            encoded_repo = urllib.parse.quote(self.repo_id, safe="/")
            if raw_repo and (raw_repo in message or encoded_repo in message):
                self._rate_limited_threads[thread_id] = now
                self._notify(retry_after_seconds=None, raw_message=message)
            return

        if "Rate limited. Waiting" not in message:
            return

        seen_at = self._rate_limited_threads.get(thread_id)
        if seen_at is None or now - seen_at > 15.0:
            return

        match = re.search(r"Waiting\s+([0-9]+(?:\.[0-9]+)?)s", message)
        retry_after = float(match.group(1)) if match else 30.0
        self._notify(retry_after_seconds=retry_after, raw_message=message)

        # huggingface_hub logs this line immediately before time.sleep(). Raising
        # here returns control to AstronomicAL so the same wait can be performed
        # cooperatively with JobManager cancellation checks.
        raise HFRateLimitBackoff(retry_after, message)

class _SnapshotProgressState:
    """Aggregate concurrent snapshot progress; file count is authoritative because byte totals are
    discovered lazily.
    """

    def __init__(self, *, expected_files: int, progress_callback: Any = None) -> None:
        self.expected_files = max(int(expected_files or 0), 0)
        self.progress_callback = progress_callback
        self._lock = threading.RLock()
        self._files_completed = 0
        self._transfer_bytes = 0
        self._reconstructed_bytes = 0
        self._last_emitted: tuple[int, int, int] | None = None

    @staticmethod
    def _format_bytes(value: int) -> str:
        amount = max(int(value or 0), 0)
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        number = float(amount)
        for unit in units:
            if number < 1024.0 or unit == units[-1]:
                if unit == "B":
                    return f"{int(number):,} {unit}"
                return f"{number:.1f} {unit}"
            number /= 1024.0
        return f"{amount:,} B"

    def update_files(self, completed: int) -> None:
        with self._lock:
            self._files_completed = max(
                self._files_completed, min(max(int(completed or 0), 0), self.expected_files),
            )
        self._emit()

    def update_bytes(self, completed: int, *, transfer: bool = False) -> None:
        with self._lock:
            value = max(int(completed or 0), 0)
            if transfer:
                self._transfer_bytes = max(self._transfer_bytes, value)
            else:
                self._reconstructed_bytes = max(self._reconstructed_bytes, value)
        self._emit()

    def _emit(self) -> None:
        if not callable(self.progress_callback):
            return
        with self._lock:
            files_completed = self._files_completed
            transfer_bytes = self._transfer_bytes
            reconstructed_bytes = self._reconstructed_bytes
            byte_value = max(transfer_bytes, reconstructed_bytes)
            signature = (files_completed, transfer_bytes, reconstructed_bytes)
            if signature == self._last_emitted:
                return
            self._last_emitted = signature

        progress_known = bool(self.expected_files and files_completed > 0)
        percent = (
            int(min(files_completed / max(self.expected_files, 1), 1.0) * 100) if progress_known else -1
        )

        message = (
            f"Resolving snapshot cache: {files_completed:,}/{self.expected_files:,} files finalized"
            if self.expected_files else "Resolving snapshot cache"
        )
        if byte_value:
            message += f" · {self._format_bytes(byte_value)} processed"
        message += "…"

        try:
            self.progress_callback(
                {
                    "phase": "snapshot_download", "completed": files_completed,
                    "total": self.expected_files, "percent": percent, "progress_known": progress_known,
                    "progress_basis": "files_finalized" if progress_known else "indeterminate",
                    "message": message, "current_file": "", "severity": "info", "progress_unit": "files",
                    "transfer_bytes": byte_value,
                }
            )
        except HFTransferCancelled:
            raise
        except Exception:
            pass

def _cancellable_tqdm_class(
    cancel_token: Any = None, progress_callback: Any = None, *, expected_files: int = 0,
):
    state = _SnapshotProgressState(expected_files=expected_files, progress_callback=progress_callback)

    class CancellableSilentTqdm(SilentTqdm):
        def __init__(self, *args, **kwargs) -> None:
            self._progress_desc = str(kwargs.get("desc") or "")
            self._progress_unit = str(kwargs.get("unit") or "")
            self._progress_name = str(kwargs.get("name") or "")
            initial = int(kwargs.get("initial", 0) or 0)
            super().__init__(*args, **kwargs)
            if initial:
                self.n = initial
            self._is_byte_bar = self._progress_unit.casefold() in {"b", "bytes"}
            self._is_transfer_bar = (
                "transfer" in self._progress_name.casefold()
                or "downloading" in self._progress_desc.casefold()
                or "download" in self._progress_desc.casefold()
            )
            self._is_file_bar = (
                not self._is_byte_bar and state.expected_files > 0
                and int(self.total or 0) == state.expected_files
            )
            if self._is_file_bar and self.n:
                state.update_files(self.n)
            elif self._is_byte_bar and self.n:
                state.update_bytes(self.n, transfer=self._is_transfer_bar)

        def update(self, n: int = 1) -> None:
            if cancel_token is not None and cancel_token.cancelled():
                raise HFTransferCancelled("Hugging Face snapshot transfer cancelled.")
            super().update(n)
            if self._is_file_bar:
                state.update_files(self.n)
            elif self._is_byte_bar:
                state.update_bytes(self.n, transfer=self._is_transfer_bar)

        # Newer Xet-backed Hugging Face releases expose a separate transfer
        # callback.  Supporting it keeps the secondary byte counter useful while
        # leaving the stable file-count denominator unchanged.
        def update_transfer(self, n: int = 1) -> None:
            if cancel_token is not None and cancel_token.cancelled():
                raise HFTransferCancelled("Hugging Face snapshot transfer cancelled.")
            with state._lock:
                completed = state._transfer_bytes + max(int(n or 0), 0)
            state.update_bytes(completed, transfer=True)

        def close(self) -> None:
            if self._is_file_bar and self.total and self.n >= int(self.total):
                state.update_files(state.expected_files)
            super().close()

    return CancellableSilentTqdm

@dataclass
class HFDatasetSearchResult:
    repo_id: str
    author: str = ""
    downloads: int | None = None
    likes: int | None = None
    last_modified: str = ""
    tags: list[str] = field(default_factory=list)
    gated: str = ""
    private: bool = False
    card_summary: str = ""
    revision: str = ""
    access_state: str = "not_gated"
    accessible: bool = True
    token_available: bool = False
    authenticated_user: str = ""
    access_message: str = ""
    access_probe_file: str = ""
    access_error: str = ""

@dataclass
class HFDatasetAccessStatus:
    """Report gated access status without submitting consent. Users request access on Hugging Face; this
    service checks pinned-file readability.
    """

    repo_id: str
    gated: bool = False
    gating_mode: str = ""
    state: str = "not_gated"
    accessible: bool = True
    token_available: bool = False
    authenticated_user: str = ""
    access_url: str = ""
    probe_file: str = ""
    message: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id, "gated": self.gated, "gating_mode": self.gating_mode,
            "state": self.state, "accessible": self.accessible, "token_available": self.token_available,
            "authenticated_user": self.authenticated_user, "access_url": self.access_url,
            "probe_file": self.probe_file, "message": self.message, "error": self.error,
        }

@dataclass
class HFCacheStatus:
    repo_id: str
    revision: str = ""
    hub_cached: bool = False
    datasets_cached: bool = False
    hub_state: str = "not_cached"
    hub_cached_files: int = 0
    hub_cached_bytes: int = 0
    datasets_cache_paths: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)

    @property
    def cached(self) -> bool:
        return bool(self.hub_cached or self.datasets_cached)

    @property
    def label(self) -> str:
        if self.hub_cached and self.datasets_cached:
            return "Hub snapshot + prepared dataset"
        if self.datasets_cached:
            return "Prepared dataset"
        if self.hub_cached:
            if self.hub_state == "revision_snapshot":
                return "Pinned Hub snapshot"
            return "Hub files"
        return "Not detected"

@dataclass
class HFSnapshotPlan:
    repo_id: str
    revision: str = ""
    total_files: int = 0
    cached_files: int = 0
    download_files: int = 0
    total_bytes: int = 0
    cached_bytes: int = 0
    download_bytes: int = 0
    known_size_files: int = 0
    download_known_size_files: int = 0
    cache_coverage_known: bool = False
    dry_run_supported: bool = False
    planning_mode: str = "dry_run"

    @property
    def total_size_complete(self) -> bool:
        return self.total_files > 0 and self.known_size_files >= self.total_files

    @property
    def download_size_complete(self) -> bool:
        return self.cache_coverage_known and (
            self.download_files == 0 or (
                self.download_files > 0 and self.download_known_size_files >= self.download_files
            )
        )

    @property
    def cache_reused(self) -> bool:
        return self.cached_files > 0

    @property
    def fully_cached(self) -> bool:
        return self.total_files > 0 and self.download_files == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id, "revision": self.revision, "total_files": self.total_files,
            "cached_files": self.cached_files, "download_files": self.download_files,
            "total_bytes": self.total_bytes, "cached_bytes": self.cached_bytes,
            "download_bytes": self.download_bytes, "known_size_files": self.known_size_files,
            "download_known_size_files": self.download_known_size_files,
            "cache_coverage_known": self.cache_coverage_known,
            "total_size_complete": self.total_size_complete,
            "download_size_complete": self.download_size_complete,
            "dry_run_supported": self.dry_run_supported, "planning_mode": self.planning_mode,
            "cache_reused": self.cache_reused, "fully_cached": self.fully_cached,
        }

@dataclass(frozen=True)
class HFPathLabelInfo:
    """Retain primary path-label inference and separate filename prediction metadata to avoid label
    contamination.
    """

    label: str = ""
    source: str = ""
    primary_label: str = ""
    secondary_label: str = ""
    secondary_confidence: float | None = None
    prefix_value: float | None = None
    parser: str = ""
    ambiguous: bool = False

@dataclass
class HFImageFile:
    path: str
    split: str = "train"
    label: str = ""
    size: int | None = None
    label_source: str = ""
    label_secondary: str = ""
    label_secondary_confidence: float | None = None
    label_prefix_value: float | None = None
    label_parser: str = ""
    label_ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "split": self.split, "label": self.label, "size": self.size,
            "label_source": self.label_source, "label_secondary": self.label_secondary,
            "label_secondary_confidence": self.label_secondary_confidence,
            "label_prefix_value": self.label_prefix_value, "label_parser": self.label_parser,
            "label_ambiguous": self.label_ambiguous,
        }

@dataclass
class HFDatasetDetails:
    repo_id: str
    revision: str = ""
    inspection_source: str = "unknown"
    configs: list[str] = field(default_factory=lambda: ["default"])
    splits_by_config: dict[str, list[str]] = field(default_factory=dict)
    features_by_config: dict[str, dict[str, str]] = field(default_factory=dict)
    feature_specs_by_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    row_counts_by_config: dict[str, dict[str, int]] = field(default_factory=dict)
    parquet_files_by_config: dict[str, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)
    parquet_partial: bool = False
    task_categories: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    pretty_name: str = ""
    description: str = ""
    description_source: str = ""
    card_sections: list[dict[str, str]] = field(default_factory=list)
    card_source_url: str = ""
    card_detection_text: str = ""
    image_column_candidates: list[str] = field(default_factory=lambda: ["__hf_file__"])
    label_column_candidates: list[str] = field(default_factory=lambda: ["__hf_path_label__"])
    warnings: list[str] = field(default_factory=list)
    sample_files: list[HFImageFile] = field(default_factory=list)
    scanned_file_count: int = 0
    total_repo_files_hint: int | None = None
    total_rows_hint: int | None = None
    total_bytes_hint: int | None = None
    repository_stats_by_split: dict[str, dict[str, Any]] = field(default_factory=dict)
    repository_inventory_complete: bool = False
    repository_label_examples: list[str] = field(default_factory=list)
    inferred_task_hint: str = ""
    repository_format_hint: str = ""
    annotation_source_hint: str = ""
    ground_truth_field_hint: str = ""
    prediction_field_hints: list[str] = field(default_factory=list)
    fiftyone_schema: dict[str, Any] = field(default_factory=dict)
    declared_data_files_by_config: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    structured_annotation_profile: dict[str, Any] = field(default_factory=dict)

    @property
    def error(self) -> str:
        return "\n".join(self.warnings)

class HuggingFaceDatasetService:
    """Inspect datasets via lightweight Hub metadata, prefer local caches for previews, and use filtered
    snapshots for full imports.
    """

    def __init__(self, *, token: str | None = None, cache_dir: str | Path | None = None) -> None:
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        # Hugging Face's standard caches are the primary source.  Omitting
        # cache_dir from Hub/Datasets calls lets downloads made by notebooks,
        # the CLI, and other applications be reused automatically.
        self.configured_cache_dir = (
            Path(cache_dir).expanduser().resolve() if cache_dir not in (None, "") else None
        )

        # Plugin-owned generated assets and legacy downloads remain under the
        # AstronomicAL cache root.  Legacy locations are probed before network
        # access so existing installations do not redownload their data.
        self.cache_dir = Path(".astronomical_cache/huggingface").expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = self.cache_dir / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        # The plugin service is long-lived, so preview thumbnails, prepared local
        # Dataset handles, and repository file listings can be reused safely.
        self._preview_lock = threading.RLock()
        self._auth_lock = threading.RLock()
        self._auth_identity_cache: dict[str, tuple[float, str, str]] = {}
        self._preview_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._preview_dataset_cache: dict[tuple[str, str, str, str, str], Any] = {}
        self._image_file_cache: dict[str, tuple[list[HFImageFile], bool]] = {}
        self._repository_file_cache: dict[tuple[str, str, str], tuple[list[dict[str, Any]], bool]] = {}
        self._hub_scan_cache: dict[str, tuple[float, Any]] = {}
        self._fiftyone_schema_cache: dict[str, FiftyOneSchema | None] = {}
        self._fiftyone_sample_index_cache: dict[str, FiftyOneSampleIndex | None] = {}
        # Inspection metadata is safe, small, and expensive enough to justify a
        # service-lifetime cache.  Values are deep-copied on both sides so UI
        # code can add warnings or derive selections without mutating the cache.
        self._dataset_details_cache: dict[tuple[str, str, str, str], HFDatasetDetails] = {}

    def _preview_cache_get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        with self._preview_lock:
            cached = self._preview_cache.get(key)
            if cached is None:
                return None
            return {**cached, "items": [dict(item) for item in cached.get("items", [])]}

    def _preview_cache_put(self, key: tuple[Any, ...], result: dict[str, Any]) -> None:
        stored = {**dict(result), "items": [dict(item) for item in result.get("items", [])]}
        with self._preview_lock:
            self._preview_cache[key] = stored
            while len(self._preview_cache) > 24:
                oldest = next(iter(self._preview_cache))
                self._preview_cache.pop(oldest, None)

    def _load_split_local_only(
        self, *, repo_id: str, config_name: str | None, split: str, token: str | None,
        trust_remote_code: bool, streaming: bool, revision: str | None = None,
    ) -> Any | None:
        from datasets import load_dataset

        def load(kwargs: dict[str, Any]):
            try:
                if config_name:
                    return load_dataset(repo_id, name=config_name, split=split, **kwargs)
                return load_dataset(repo_id, split=split, **kwargs)
            except TypeError:
                retry = dict(kwargs)
                retry.pop("trust_remote_code", None)
                retry.pop("token", None)
                if config_name:
                    return load_dataset(repo_id, name=config_name, split=split, **retry)
                return load_dataset(repo_id, split=split, **retry)

        for cache_root in self._datasets_cache_roots():
            kwargs = self._datasets_kwargs_for_cache(
                token=token, trust_remote_code=trust_remote_code, cache_root=cache_root,
                local_files_only=True, streaming=streaming,
            )
            if revision:
                kwargs["revision"] = str(revision)
            try:
                return load(kwargs)
            except Exception:
                continue
        return None

    @staticmethod
    def _preview_result_to_html(result: dict[str, Any], *, thumb_size: int) -> str:
        items = list(result.get("items", []) or [])
        note = html.escape(str(result.get("note", "") or ""))
        if not items:
            return f"<p>{note or 'No preview rows were returned.'}</p>"

        cards: list[str] = []
        for item in items:
            data_uri = str(item.get("data_uri", "") or "")
            if data_uri:
                image_html = (
                    f'<img src="{data_uri}" style="display:block;width:100%;height:{thumb_size}px;object-fit:contain;background:#111;" />'
                )
            else:
                image_html = (
                    f'<div style="height:{thumb_size}px;display:flex;align-items:center;justify-content:center;background:#eef2f6;color:#687386;">Preview unavailable</div>'
                )
            title = html.escape(str(item.get("record_id", "") or ""))
            label = html.escape(str(item.get("label", "") or ""))
            image_col = html.escape(str(item.get("image_column", "") or ""))
            label_col = html.escape(str(item.get("label_column", "") or ""))
            cards.append(
                '<div style="border:1px solid #d8dee8;border-radius:8px;padding:8px;background:#fff;">'
                + image_html
                + f'<div style="margin-top:7px;font-size:12px;line-height:1.4;overflow-wrap:anywhere;"><strong>{title}</strong><br>{label}<br><span style="color:#687386;">image: {image_col}</span><br><span style="color:#687386;">label: {label_col}</span></div></div>'
            )
        return (
            f'<p style="font-size:11px;color:#687386;line-height:1.4;">{note}</p>'
            '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;width:100%;">'
            + ''.join(cards) + '</div>'
        )

    # ------------------------------------------------------------------
    # Cache discovery and local-first helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalised_repo_cache_name(repo_id: str) -> str:
        return "datasets--" + str(repo_id).strip().replace("/", "--")

    def _hub_cache_roots(self) -> list[Path | None]:
        roots: list[Path | None] = [None]

        if self.configured_cache_dir is not None:
            roots.extend(
                [
                    self.configured_cache_dir, self.configured_cache_dir / "hub",
                    self.configured_cache_dir / "hub_files", self.configured_cache_dir / "snapshots",
                ]
            )

        roots.extend([self.cache_dir / "hub", self.cache_dir / "hub_files", self.cache_dir / "snapshots"])

        seen: set[str] = set()
        out: list[Path | None] = []
        for root in roots:
            key = "<default>" if root is None else str(root.expanduser())
            if key in seen:
                continue
            seen.add(key)
            out.append(root)
        return out

    def _datasets_cache_roots(self) -> list[Path | None]:
        roots: list[Path | None] = [None]

        if self.configured_cache_dir is not None:
            roots.extend(
                [
                    self.configured_cache_dir, self.configured_cache_dir / "datasets",
                    self.configured_cache_dir / "datasets_cache",
                ]
            )

        roots.extend([self.cache_dir / "datasets", self.cache_dir / "datasets_cache"])

        seen: set[str] = set()
        out: list[Path | None] = []
        for root in roots:
            key = "<default>" if root is None else str(root.expanduser())
            if key in seen:
                continue
            seen.add(key)
            out.append(root)
        return out

    @staticmethod
    def _default_hub_cache_dir() -> Path:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            return Path(HF_HUB_CACHE).expanduser()
        except Exception:
            root = os.environ.get("HF_HOME")
            if root:
                return Path(root).expanduser() / "hub"
            root = os.environ.get("HF_HUB_CACHE")
            if root:
                return Path(root).expanduser()
            return Path.home() / ".cache" / "huggingface" / "hub"

    @staticmethod
    def _default_datasets_cache_dir() -> Path:
        try:
            import datasets

            value = getattr(getattr(datasets, "config", None), "HF_DATASETS_CACHE", None)
            if value:
                return Path(value).expanduser()
        except Exception:
            pass

        root = os.environ.get("HF_DATASETS_CACHE")
        if root:
            return Path(root).expanduser()
        root = os.environ.get("HF_HOME")
        if root:
            return Path(root).expanduser() / "datasets"
        return Path.home() / ".cache" / "huggingface" / "datasets"

    def _scan_hub_cache_cached(self, root: Path) -> Any | None:
        """Cache expensive scan_cache_dir results briefly for search-result rendering."""
        key = str(Path(root).expanduser())
        now = time.time()
        with self._preview_lock:
            cached = self._hub_scan_cache.get(key)
            if cached is not None and now - cached[0] <= 10.0:
                return cached[1]
        try:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir(cache_dir=Path(root).expanduser())
        except Exception:
            info = None
        with self._preview_lock:
            self._hub_scan_cache[key] = (now, info)
        return info

    def cache_status(self, repo_id: str, *, revision: str | None = None) -> HFCacheStatus:
        """Return a factual local cache inventory rather than a directory-exists guess."""
        repo_id = str(repo_id or "").strip()
        status = HFCacheStatus(repo_id=repo_id, revision=str(revision or ""))
        if not repo_id:
            return status

        locations: list[str] = []
        seen_cache_roots: set[str] = set()
        hub_roots: list[Path] = [self._default_hub_cache_dir()]
        hub_roots.extend(root for root in self._hub_cache_roots() if root is not None)

        for root in hub_roots:
            root = Path(root).expanduser()
            key = str(root)
            if key in seen_cache_roots or not root.exists():
                continue
            seen_cache_roots.add(key)
            info = self._scan_hub_cache_cached(root)
            if info is None:
                continue
            for cached_repo in getattr(info, "repos", ()) or ():
                if str(getattr(cached_repo, "repo_id", "")) != repo_id:
                    continue
                if str(getattr(cached_repo, "repo_type", "")) not in {"dataset", "datasets"}:
                    continue
                revisions = list(getattr(cached_repo, "revisions", ()) or ())
                selected = []
                if revision:
                    selected = [
                        item for item in revisions if str(getattr(item, "commit_hash", "")) == str(revision)
                    ]
                else:
                    selected = revisions
                if selected:
                    status.hub_cached = True
                    status.hub_state = "revision_snapshot" if revision else "cached_revisions"
                    for item in selected:
                        files = list(getattr(item, "files", ()) or ())
                        status.hub_cached_files += len(files)
                        status.hub_cached_bytes += sum(
                            int(getattr(file, "size_on_disk", 0) or 0) for file in files
                        )
                        snapshot_path = getattr(item, "snapshot_path", None)
                        if snapshot_path:
                            locations.append(str(snapshot_path))
                repo_path = getattr(cached_repo, "repo_path", None)
                if repo_path:
                    locations.append(str(repo_path))

        # Compatibility fallback for old huggingface_hub versions or legacy caches.
        if not status.hub_cached:
            repo_cache_name = self._normalised_repo_cache_name(repo_id)
            for root in hub_roots:
                root = Path(root).expanduser()
                candidates = [
                    root / repo_cache_name, root / "hub" / repo_cache_name,
                    root / "hub_files" / repo_cache_name, root / "snapshots" / repo_cache_name,
                ]
                for candidate in candidates:
                    if candidate.exists():
                        status.hub_cached = True
                        status.hub_state = "legacy_or_partial"
                        locations.append(str(candidate))
                        break

        datasets_roots: list[Path] = [self._default_datasets_cache_dir()]
        datasets_roots.extend(root for root in self._datasets_cache_roots() if root is not None)
        owner, _, name = repo_id.partition("/")
        search_tokens = {
            repo_id.replace("/", "___"), repo_id.replace("/", "--"), repo_id.replace("/", "_"), name,
        }
        if owner:
            search_tokens.add(owner)

        seen_dataset_roots: set[str] = set()
        for root in datasets_roots:
            root = Path(root).expanduser()
            root_key = str(root)
            if root_key in seen_dataset_roots or not root.exists():
                continue
            seen_dataset_roots.add(root_key)
            try:
                children = list(root.iterdir())
            except Exception:
                continue
            for child in children:
                child_name = child.name
                if name and name not in child_name:
                    continue
                if any(token and token in child_name for token in search_tokens):
                    status.datasets_cached = True
                    status.datasets_cache_paths.append(str(child))
                    locations.append(str(child))
                    break

        status.locations = list(dict.fromkeys(locations))
        return status

    @staticmethod
    def _looks_like_commit_revision(value: str | None) -> bool:
        text = str(value or "").strip()
        return bool(re.fullmatch(r"[0-9a-fA-F]{40,64}", text))

    def _details_cache_auth_scope(self, token: str | None = None) -> str:
        """Partition inspection cache keys by non-secret authentication fingerprint; never retain a raw
        token as a key.
        """
        resolved = str(self.effective_token(token) or "").strip()
        if not resolved:
            return "anonymous"
        digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
        return f"authenticated:{digest}"

    @staticmethod
    def _details_cache_key(
        repo_id: str, *, config_name: str | None = None, mode: str = "auto", auth_scope: str = "anonymous",
    ) -> tuple[str, str, str, str]:
        return (
            str(repo_id or "").strip(), str(config_name or "default"), str(mode or "auto"),
            str(auth_scope or "anonymous"),
        )

    def remember_dataset_details(
        self, details: HFDatasetDetails | None, *, config_name: str | None = None, mode: str = "auto",
        token: str | None = None,
    ) -> None:
        if details is None or not str(getattr(details, "repo_id", "") or "").strip():
            return
        auth_scope = self._details_cache_auth_scope(token)
        key = self._details_cache_key(
            str(details.repo_id), config_name=config_name, mode=mode, auth_scope=auth_scope,
        )
        auto_key = self._details_cache_key(
            str(details.repo_id), config_name=config_name, mode="auto", auth_scope=auth_scope,
        )
        with self._preview_lock:
            cached = copy.deepcopy(details)
            self._dataset_details_cache[key] = cached
            # Auto inspection is the normal path on restore.  Alias the most
            # recently successful explicit inspection into the auto key so a
            # panel close/reopen in the same plugin lifetime does not repeat the
            # Hub request merely because the resolved mode was viewer/files.
            self._dataset_details_cache[auto_key] = copy.deepcopy(cached)
            while len(self._dataset_details_cache) > 32:
                oldest = next(iter(self._dataset_details_cache))
                self._dataset_details_cache.pop(oldest, None)

    def _remembered_dataset_details(
        self, repo_id: str, *, config_name: str | None = None, mode: str = "auto", token: str | None = None,
    ) -> HFDatasetDetails | None:
        auth_scope = self._details_cache_auth_scope(token)
        requested = self._details_cache_key(
            repo_id, config_name=config_name, mode=mode, auth_scope=auth_scope,
        )
        fallback = self._details_cache_key(
            repo_id, config_name=config_name, mode="auto", auth_scope=auth_scope,
        )
        with self._preview_lock:
            details = self._dataset_details_cache.get(requested)
            if details is None and requested != fallback:
                details = self._dataset_details_cache.get(fallback)
            return copy.deepcopy(details) if details is not None else None

    def _local_snapshot_path(
        self, repo_id: str, *, revision: str | None = None, token: str | None = None,
    ) -> Path | None:
        """Resolve a Hub snapshot strictly offline, including legacy cache-root fallbacks."""
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            snapshot_download = None

        if snapshot_download is not None:
            for cache_root in self._hub_cache_roots():
                kwargs: dict[str, Any] = {
                    "repo_id": str(repo_id), "repo_type": "dataset", "local_files_only": True,
                    "token": self.effective_token(token), "tqdm_class": SilentTqdm,
                }
                if revision not in (None, ""):
                    kwargs["revision"] = str(revision)
                if cache_root is not None:
                    kwargs["cache_dir"] = str(cache_root)
                try:
                    value = self._call_with_compatible_kwargs(snapshot_download, kwargs)
                    path = Path(value).expanduser()
                    if path.is_dir():
                        return path
                except Exception:
                    continue

        status = self.cache_status(repo_id, revision=revision)
        candidates: list[Path] = []
        for raw in status.locations:
            path = Path(raw).expanduser()
            if path.is_dir() and path.name == str(revision or ""):
                candidates.insert(0, path)
            elif path.is_dir() and path.parent.name == "snapshots":
                candidates.append(path)
            elif path.is_dir() and (path / "snapshots").is_dir():
                try:
                    snapshots = [item for item in (path / "snapshots").iterdir() if item.is_dir()]
                except Exception:
                    snapshots = []
                if revision:
                    exact = [item for item in snapshots if item.name == str(revision)]
                    candidates.extend(exact)
                candidates.extend(sorted(snapshots, key=lambda item: item.stat().st_mtime, reverse=True))
        for path in candidates:
            try:
                if path.is_dir():
                    return path
            except OSError:
                continue
        return None

    @staticmethod
    def _local_snapshot_inventory(
        root: Path, *, max_files: int = REPOSITORY_INSPECTION_MAX_FILES,
    ) -> tuple[list[dict[str, Any]], bool]:
        rows: list[dict[str, Any]] = []
        exhausted = True
        root = Path(root).expanduser()
        for directory, _dirs, filenames in os.walk(root):
            base = Path(directory)
            for filename in filenames:
                path = base / filename
                try:
                    relative = path.relative_to(root).as_posix()
                    size = int(path.stat().st_size)
                except OSError:
                    continue
                rows.append({"path": relative, "size": size})
                if max_files and len(rows) >= max_files:
                    exhausted = False
                    return rows, exhausted
        return rows, exhausted

    @staticmethod
    def _local_card_text(root: Path) -> str:
        for name in ("README.md", "README.MD", "readme.md"):
            path = Path(root) / name
            try:
                if path.is_file() and path.stat().st_size <= 2 * 1024 * 1024:
                    return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        return ""

    @staticmethod
    def _local_annotation_probe(root: Path, row: dict[str, Any]) -> dict[str, Any] | None:
        path_text = str(row.get("path") or "").strip()
        try:
            advertised_size = int(row.get("size") or 0)
        except Exception:
            advertised_size = 0
        if (
            not path_text or advertised_size <= 0 or advertised_size > STRUCTURED_ANNOTATION_PROBE_MAX_BYTES
        ):
            return None
        path = Path(root) / path_text
        try:
            actual_size = int(path.stat().st_size)
            if actual_size <= 0 or actual_size > STRUCTURED_ANNOTATION_PROBE_MAX_BYTES:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        videos = payload.get("videos")
        annotations_present = "annotations" in payload
        annotations = payload.get("annotations")
        categories = payload.get("categories")
        video_sample = next(
            (item for item in videos if isinstance(item, dict)), None,
        ) if isinstance(videos, list) else None
        annotation_sample = next(
            (item for item in annotations if isinstance(item, dict)), None,
        ) if isinstance(annotations, list) else None
        first_file_names = []
        if isinstance(video_sample, dict):
            first_file_names = [
                str(value) for value in (video_sample.get("file_names") or [])[:3] if str(value).strip()
            ]
        return {
            "path": path_text, "size": actual_size, "root_fields": [str(key) for key in payload.keys()],
            "videos_is_array": isinstance(videos, list), "annotations_present": annotations_present,
            "annotations_is_array": isinstance(annotations, list),
            "annotations_is_null": annotations_present and annotations is None,
            "categories_is_array": isinstance(categories, list),
            "video_sampled": isinstance(video_sample, dict), "video_fields": (
                [str(key) for key in video_sample.keys()] if isinstance(video_sample, dict) else []
            ), "annotation_sampled": isinstance(annotation_sample, dict), "annotation_fields": (
                [str(key) for key in annotation_sample.keys()] if isinstance(annotation_sample, dict)
                else []
            ), "first_file_names": first_file_names,
        }

    def _local_frame_media_roots(
        self, *, snapshot_root: Path, profile: HFStructuredAnnotationProfile,
        inventory: list[dict[str, Any]], probes: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        roots = {
            str(key): str(value) for key, value in dict(profile.frame_media_roots_by_split or {}).items()
            if str(key).strip() and str(value).strip()
        }
        image_paths = [
            str(row.get("path") or "").strip() for row in inventory
            if Path(str(row.get("path") or "")).suffix.casefold() in IMAGE_EXTENSIONS
        ]
        probe_samples: dict[str, list[str]] = {}
        for annotation_path, payload in probes.items():
            split = self._infer_split_from_path(annotation_path)
            names = [
                str(value).strip().lstrip("/") for value in (payload.get("first_file_names") or [])
                if str(value).strip()
            ]
            if split and names:
                probe_samples.setdefault(split, []).extend(names)

        for split in profile.splits:
            if split in roots:
                continue
            samples = probe_samples.get(str(split), [])
            counts: Counter[str] = Counter()
            for path in image_paths[:REPOSITORY_INSPECTION_SAMPLE_FILES * 8]:
                inferred_split = self.infer_split_from_path(path)
                if inferred_split != split:
                    continue
                clean_path = str(path or "").strip().lstrip("/")
                normalised_samples = [
                    str(value or "").strip().lstrip("/") for value in samples if str(value or "").strip()
                ]
                # An annotation file_name can be repository-root-relative.  The
                # existing root helper represents that valid root as an empty
                # string, which is otherwise indistinguishable from "no guess".
                # Preserve the exact-match evidence explicitly for local caches.
                if clean_path and clean_path in normalised_samples:
                    counts[""] += 1
                    continue
                candidate = self._repository_media_root_from_image_path(
                    path, split=str(split), sample_names=samples,
                )
                if candidate:
                    counts[candidate] += 1
            if not counts:
                continue
            chosen = sorted(counts.items(), key=lambda item: (-int(item[1]), len(item[0]), item[0]))[0][0]
            local_root = (Path(snapshot_root) / chosen).resolve()
            if local_root.is_dir():
                roots[str(split)] = local_root.as_uri().rstrip("/") + "/"
        return roots

    def get_dataset_details_cached(
        self, repo_id: str, *, config_name: str | None = None, mode: str = "auto", token: str | None = None,
        trust_remote_code: bool = False, revision: str | None = None,
        cache_status: HFCacheStatus | None = None,
    ) -> HFDatasetDetails | None:
        """Inspect only locally cached metadata, snapshots or builders; return None if remote metadata is
        required.
        """
        repo_id = str(repo_id or "").strip()
        if not repo_id:
            return None

        remembered = self._remembered_dataset_details(
            repo_id, config_name=config_name, mode=mode, token=token,
        )
        if remembered is not None and revision:
            remembered_revision = str(getattr(remembered, "revision", "") or "").strip()
            if remembered_revision != str(revision).strip():
                remembered = None
        if remembered is not None:
            remembered.inspection_source = "memory_cache"
            return remembered

        cache = cache_status or self.cache_status(repo_id, revision=revision)
        if not cache.cached:
            return None

        snapshot_root = self._local_snapshot_path(repo_id, revision=revision, token=token)
        if snapshot_root is not None:
            inventory, complete = self._local_snapshot_inventory(snapshot_root)
            if inventory:
                resolved_revision = (
                    str(revision or "").strip()
                    or (snapshot_root.name if self._looks_like_commit_revision(snapshot_root.name) else "")
                    or str(cache.revision or "")
                )
                details = HFDatasetDetails(
                    repo_id=repo_id, revision=resolved_revision, inspection_source="local_hub_snapshot",
                )
                details.total_repo_files_hint = len(inventory) if complete else None
                card_text = self._local_card_text(snapshot_root)
                details.card_detection_text = card_text
                details.description_source = "local cached dataset card" if card_text else ""
                if card_text:
                    body = re.sub(r"^---.*?---\s*", "", card_text, count=1, flags=re.DOTALL).strip()
                    details.description = body[:12_000]

                candidates = structured_annotation_candidates(
                    repository_files=inventory, declared_data_files_by_split={},
                )
                probes: dict[str, dict[str, Any]] = {}
                eligible = sorted(
                    candidates, key=lambda row: (int(row.get("size") or 0), str(row.get("path") or "")),
                )
                for row in eligible[:STRUCTURED_ANNOTATION_PROBE_MAX_FILES]:
                    probe = self._local_annotation_probe(snapshot_root, row)
                    if probe is not None:
                        probes[str(row.get("path") or "")] = probe
                profile = detect_structured_annotation_profile(
                    card_text=card_text, repository_files=inventory, declared_data_files_by_split={},
                    probed_json_by_path=probes,
                )
                if profile is not None:
                    local_roots = self._local_frame_media_roots(
                        snapshot_root=snapshot_root, profile=profile, inventory=inventory, probes=probes,
                    )
                    if local_roots != dict(profile.frame_media_roots_by_split or {}):
                        evidence = list(profile.evidence or ())
                        evidence.append("local Hugging Face snapshot supplied cached frame media roots")
                        profile = replace(
                            profile, frame_media_roots_by_split=local_roots, evidence=tuple(evidence),
                        )
                    details.structured_annotation_profile = profile.to_dict()
                    details.repository_format_hint = profile.format_id
                    details.annotation_source_hint = profile.format_label
                    if profile.task_hints:
                        details.inferred_task_hint = profile.task_hints[0]
                    details.configs = [str(config_name or "default")]
                    details.splits_by_config[details.configs[0]] = list(profile.splits or ("train",))
                    details.image_column_candidates = []
                    details.label_column_candidates = []
                    self.remember_dataset_details(details, config_name=config_name, mode=mode, token=token)
                    return copy.deepcopy(details)

                image_rows = [row for row in inventory if self._is_image_path(str(row.get("path") or ""))]
                if image_rows:
                    files: list[HFImageFile] = []
                    for row in image_rows[:REPOSITORY_INSPECTION_SAMPLE_FILES]:
                        path_text = str(row.get("path") or "")
                        split = self.infer_split_from_path(path_text)
                        label_info = self.infer_label_details_from_path(path_text, split)
                        files.append(
                            HFImageFile(
                                path=path_text, split=split, label=label_info.label, size=row.get("size"),
                                label_source=label_info.source, label_secondary=label_info.secondary_label,
                                label_secondary_confidence=label_info.secondary_confidence,
                                label_prefix_value=label_info.prefix_value, label_parser=label_info.parser,
                                label_ambiguous=label_info.ambiguous,
                            )
                        )
                    details.sample_files = files
                    details.scanned_file_count = len(files)
                    splits = sorted({item.split for item in files if item.split}) or ["train"]
                    details.configs = [str(config_name or "default")]
                    details.splits_by_config[details.configs[0]] = splits
                    details.features_by_config[details.configs[0]] = {
                        "__hf_file__": "Cached Hugging Face image files",
                        "__hf_path_label__": "Inferred label from cached repository path",
                    }
                    details.image_column_candidates = ["__hf_file__"]
                    details.label_column_candidates = ["__hf_path_label__"]
                    cache_key = f"{repo_id}@{resolved_revision or 'default'}"
                    with self._preview_lock:
                        self._image_file_cache[cache_key] = (list(files), bool(complete))
                    self.remember_dataset_details(details, config_name=config_name, mode=mode, token=token)
                    return copy.deepcopy(details)

        if cache.datasets_cached:
            details = self.get_dataset_details_builder_local(
                repo_id, config_name=config_name, token=token, trust_remote_code=trust_remote_code,
                revision=revision,
            )
            if details is not None:
                self.remember_dataset_details(details, config_name=config_name, mode=mode, token=token)
                return copy.deepcopy(details)

        return None

    def _hub_download_kwargs(
        self, *, repo_id: str, filename: str | None = None, token: str | None = None,
        cache_root: Path | None = None, local_files_only: bool = False, revision: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "repo_id": repo_id, "repo_type": "dataset", "token": token or self.token,
            "local_files_only": bool(local_files_only),
        }
        if filename is not None:
            kwargs["filename"] = filename
        if revision not in (None, ""):
            kwargs["revision"] = str(revision)
        if cache_root is not None:
            kwargs["cache_dir"] = str(cache_root)
        return kwargs

    @staticmethod
    def _call_with_compatible_kwargs(func: Any, kwargs: dict[str, Any]) -> Any:
        attempt = dict(kwargs)
        removable = ("token", "tqdm_class", "max_workers")
        while True:
            try:
                return func(**attempt)
            except TypeError as exc:
                message = str(exc)
                removed = False
                for key in removable:
                    if key in attempt and key in message:
                        attempt.pop(key, None)
                        removed = True
                        break
                if not removed:
                    raise

    def _datasets_kwargs_for_cache(
        self, *, token: str | None, trust_remote_code: bool, cache_root: Path | None,
        local_files_only: bool, streaming: bool | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if cache_root is not None:
            kwargs["cache_dir"] = str(cache_root)
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        resolved_token = token or self.token
        if resolved_token:
            kwargs["token"] = resolved_token
        if streaming is not None:
            kwargs["streaming"] = bool(streaming)

        if local_files_only:
            try:
                from datasets import DownloadConfig

                kwargs["download_config"] = DownloadConfig(local_files_only=True)
            except Exception:
                pass

        return kwargs

    # ------------------------------------------------------------------
    # Dataset Viewer API / immutable revision inspection
    # ------------------------------------------------------------------

    def resolve_dataset_revision(
        self, repo_id: str, *, revision: str | None = None, token: str | None = None,
    ) -> str:
        """Resolve refs to an immutable revision, without a network request when already pinned."""
        if self._looks_like_commit_revision(revision):
            return str(revision).strip()

        from huggingface_hub import HfApi

        api = HfApi(token=token or self.token)
        kwargs: dict[str, Any] = {
            "repo_id": str(repo_id), "revision": revision, "token": token or self.token,
        }
        try:
            info = api.dataset_info(**kwargs)
        except TypeError:
            kwargs.pop("token", None)
            info = api.dataset_info(**kwargs)
        sha = str(getattr(info, "sha", "") or "").strip()
        return sha or str(revision or "main")

    def _viewer_request(
        self, endpoint: str, *, token: str | None = None, timeout: float = 15.0, **params: Any,
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {key: value for key, value in params.items() if value not in (None, "")}
        )
        url = f"https://datasets-server.huggingface.co/{endpoint.lstrip('/')}"
        if query:
            url += "?" + query
        headers = {"Accept": "application/json"}
        resolved_token = token or self.token
        if resolved_token:
            headers["Authorization"] = f"Bearer {resolved_token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"Dataset Viewer /{endpoint} failed with HTTP {exc.code}: {detail[:500]}"
            ) from exc
        if not isinstance(payload, dict):
            raise TypeError(f"Dataset Viewer /{endpoint} returned a non-object response.")
        return payload

    @staticmethod
    def _viewer_feature_to_string(feature_type: Any) -> str:
        if not isinstance(feature_type, dict):
            return str(feature_type)
        kind = str(feature_type.get("_type") or "").strip()
        if kind == "ClassLabel":
            names = list(feature_type.get("names") or [])
            return f"ClassLabel(names={names!r})"
        if kind == "Value":
            return f"Value({feature_type.get('dtype', 'unknown')})"
        if kind:
            return kind + (f"({feature_type.get('dtype')})" if feature_type.get("dtype") else "")
        return json.dumps(feature_type, sort_keys=True, default=str)

    @staticmethod
    def _task_metadata(card_data: Any) -> tuple[list[str], list[str]]:
        if not isinstance(card_data, dict):
            try:
                card_data = dict(card_data or {})
            except Exception:
                card_data = {}
        categories = card_data.get("task_categories") or []
        task_ids = card_data.get("task_ids") or []
        if isinstance(categories, str):
            categories = [categories]
        if isinstance(task_ids, str):
            task_ids = [task_ids]
        return (
            [str(value) for value in categories if str(value).strip()],
            [str(value) for value in task_ids if str(value).strip()],
        )

    @staticmethod
    def _card_data_dict(card_data: Any) -> dict[str, Any]:
        if isinstance(card_data, dict):
            return dict(card_data)
        try:
            return dict(card_data or {})
        except Exception:
            return {}

    @staticmethod
    def _strip_dataset_card_front_matter(value: Any) -> str:
        text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
        if not text.lstrip().startswith("---"):
            return text

        lines = text.splitlines()
        first = None
        for index, line in enumerate(lines):
            if line.strip():
                first = index
                break
        if first is None or lines[first].strip() != "---":
            return text

        for index in range(first + 1, len(lines)):
            if lines[index].strip() == "---":
                return "\n".join(lines[index + 1 :])
        return text

    @staticmethod
    def _normalise_card_heading(value: Any) -> str:
        text = str(value or "").strip().casefold()
        text = re.sub(r"[`*_#]+", "", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _clean_dataset_card_markdown(cls, value: Any, *, max_chars: int = 1200) -> str:
        """Keep readable prose and lists from a dataset card; discard presentation and installation
        snippets.
        """
        text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            return ""

        cleaned: list[str] = []
        in_fence = False
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue

            heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$", raw_line)
            if heading:
                title = re.sub(r"\s+", " ", heading.group(1)).strip()
                if title:
                    cleaned.append("")
                    cleaned.append(title + ":")
                continue

            line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", raw_line)
            line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
            line = re.sub(r"<[^>]+>", " ", line)
            line = line.replace("**", "").replace("__", "").replace("`", "")
            line = re.sub(r"^\s*[-*+]\s+", "• ", line)
            line = re.sub(r"^\s*\d+[.)]\s+", "• ", line)
            if re.match(r"^\s*\|?[-: ]+\|[-|: ]*\s*$", line):
                continue
            line = re.sub(r"\s+", " ", line).strip()
            cleaned.append(line)

        out: list[str] = []
        paragraph: list[str] = []
        for line in cleaned + [""]:
            if not line:
                if paragraph:
                    out.append(" ".join(paragraph).strip())
                    paragraph = []
                continue
            if line.startswith("• ") or line.endswith(":"):
                if paragraph:
                    out.append(" ".join(paragraph).strip())
                    paragraph = []
                out.append(line)
            else:
                paragraph.append(line)

        boilerplate_prefixes = (
            "this is a fiftyone dataset with ", "if you haven't already, install fiftyone", "pip install ",
            "import fiftyone", "dataset = fouh.load_from_hub", "session = fo.launch_app",
            "load the dataset", "launch the app",
        )
        normalised_prefixes = tuple(cls._normalise_card_heading(prefix) for prefix in boilerplate_prefixes)
        meaningful = [
            line for line in out if line
            and not cls._normalise_card_heading(line).startswith(normalised_prefixes)
        ]
        rendered_parts: list[str] = []
        for line in meaningful:
            if not rendered_parts:
                rendered_parts.append(line)
                continue
            previous = rendered_parts[-1]
            if line.startswith("• ") and previous.lstrip().startswith("• "):
                rendered_parts.append("\n" + line)
            elif line.startswith("• ") or previous.lstrip().startswith("• "):
                rendered_parts.append("\n" + line)
            elif line.endswith(":") or previous.rstrip().endswith(":"):
                rendered_parts.append("\n" + line)
            else:
                rendered_parts.append("\n\n" + line)

        result = "".join(rendered_parts).strip()
        if len(result) <= max_chars:
            return result

        clipped = result[: max(int(max_chars), 120)].rstrip()
        sentence_end = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
        if sentence_end >= int(max_chars * 0.55):
            clipped = clipped[: sentence_end + 1]
        else:
            line_end = clipped.rfind("\n")
            if line_end >= int(max_chars * 0.55):
                clipped = clipped[:line_end]
            else:
                word_end = clipped.rfind(" ")
                if word_end > 0:
                    clipped = clipped[:word_end]
            clipped += "…"
        return clipped.strip()

    @classmethod
    def _extract_dataset_card_sections(
        cls, value: Any, *, max_sections: int = 5, max_chars_per_section: int = 1100,
        max_total_chars: int = 4200,
    ) -> list[dict[str, str]]:
        """Extract relevant README sections structurally; skip boilerplate without repository-specific
        cases.
        """
        text = cls._strip_dataset_card_front_matter(value).strip()
        if not text:
            return []

        lines = text.splitlines()
        headings: list[tuple[int, int, str]] = []
        for index, line in enumerate(lines):
            match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if not match:
                continue
            headings.append((index, len(match.group(1)), match.group(2).strip()))

        if not headings:
            body = cls._clean_dataset_card_markdown(
                text, max_chars=min(max_total_chars, max_chars_per_section * 2),
            )
            return [{"title": "Overview", "text": body}] if body else []

        ignored = {
            "installation", "install", "usage", "example usage", "quickstart", "quick start", "citation",
            "citation information", "contributions", "table of contents",
        }
        priority_groups = [
            {
                "dataset description", "dataset summary", "summary", "overview", "about",
                "about this dataset",
            }, {
                "supported tasks and leaderboards", "supported tasks", "intended use", "intended uses",
                "uses", "applications", "evaluation", "benchmark",
            }, {
                "dataset structure", "dataset contents", "data splits", "dataset statistics", "statistics",
                "data instances", "data fields",
            }, {
                "dataset creation", "curation rationale", "source data", "dataset sources", "sources",
                "collection process", "annotations",
            }, {
                "considerations for using the data", "other known limitations", "known limitations",
                "limitations", "discussion of biases", "biases", "social impact of dataset",
            },
        ]

        raw_sections: list[dict[str, Any]] = []
        for pos, (line_index, level, title) in enumerate(headings):
            norm = cls._normalise_card_heading(title)
            if not norm or norm in ignored:
                continue
            end_index = len(lines)
            for next_index, next_level, _next_title in headings[pos + 1 :]:
                if next_level <= level:
                    end_index = next_index
                    break
            body = cls._clean_dataset_card_markdown(
                "\n".join(lines[line_index + 1 : end_index]), max_chars=max_chars_per_section,
            )
            if not body:
                continue
            raw_sections.append({"title": title, "norm": norm, "text": body, "source_index": line_index})

        if not raw_sections:
            return []

        def group_rank(norm: str) -> int:
            for rank, group in enumerate(priority_groups):
                if norm in group:
                    return rank
            return len(priority_groups) + 1

        administrative = {
            "dataset details", "dataset curators", "licensing information", "license",
            "additional information", "acknowledgements", "acknowledgments",
        }
        candidates = [
            section for section in raw_sections if section["norm"] not in administrative
            and not str(section["norm"]).startswith("dataset card for ")
        ]
        candidates.sort(
            key=lambda section: (group_rank(str(section["norm"])), int(section["source_index"]))
        )

        chosen: list[dict[str, str]] = []
        seen_text: set[str] = set()
        total_chars = 0
        for section in candidates:
            text_value = str(section["text"]).strip()
            fingerprint = re.sub(r"\W+", "", text_value.casefold())[:280]
            if not fingerprint or fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)

            if total_chars and total_chars + len(text_value) > max_total_chars:
                remaining = max_total_chars - total_chars
                if remaining < 260:
                    continue
                text_value = cls._clean_dataset_card_markdown(text_value, max_chars=remaining)
            if not text_value:
                continue

            chosen.append({"title": str(section["title"]).strip(), "text": text_value})
            total_chars += len(text_value)
            if len(chosen) >= max_sections or total_chars >= max_total_chars:
                break

        return chosen

    @classmethod
    def _extract_dataset_description_text(cls, value: Any, *, max_chars: int = 1800) -> str:
        """Return the best substantive description from card Markdown."""
        sections = cls._extract_dataset_card_sections(
            value, max_sections=5, max_chars_per_section=max_chars, max_total_chars=max_chars * 2,
        )
        if not sections:
            return ""

        preferred = {
            "dataset description", "dataset summary", "summary", "overview", "about", "about this dataset",
        }
        for section in sections:
            if cls._normalise_card_heading(section.get("title")) in preferred:
                return str(section.get("text") or "").strip()[:max_chars].strip()
        return str(sections[0].get("text") or "").strip()[:max_chars].strip()

    @staticmethod
    def _declared_data_files_from_card(card_data: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
        """Return config -> split -> declared repository paths from card YAML."""
        result: dict[str, dict[str, list[str]]] = {}
        configs = card_data.get("configs") if isinstance(card_data, dict) else None
        if isinstance(configs, dict):
            configs = [configs]
        if not isinstance(configs, (list, tuple)):
            configs = []

        for raw_config in configs:
            if not isinstance(raw_config, dict):
                continue
            config_name = str(
                raw_config.get("config_name") or raw_config.get("name") or "default"
            ).strip() or "default"
            data_files = raw_config.get("data_files")
            if isinstance(data_files, dict):
                data_files = [data_files]
            if not isinstance(data_files, (list, tuple)):
                continue
            for raw_item in data_files:
                if isinstance(raw_item, str):
                    split = "train"
                    paths = [raw_item]
                elif isinstance(raw_item, dict):
                    split = str(raw_item.get("split") or "train").strip() or "train"
                    raw_paths = raw_item.get("path")
                    if raw_paths is None:
                        raw_paths = raw_item.get("paths")
                    if isinstance(raw_paths, str):
                        paths = [raw_paths]
                    elif isinstance(raw_paths, (list, tuple)):
                        paths = [str(value) for value in raw_paths]
                    else:
                        paths = []
                else:
                    continue
                for path in paths:
                    path = str(path or "").strip()
                    if not path:
                        continue
                    bucket = result.setdefault(config_name, {}).setdefault(split, [])
                    if path not in bucket:
                        bucket.append(path)
        return result

    @staticmethod
    def _structured_detection_text(details: HFDatasetDetails) -> str:
        # ``card_detection_text`` is the canonical, pinned README text when it
        # is available. Preserve it byte-for-text rather than mixing in the
        # display-oriented description/section extracts below: those extracts
        # intentionally collapse whitespace, which destroys directory-tree
        # indentation used by structured-media path inference.
        if details.card_detection_text:
            return str(details.card_detection_text)

        parts: list[str] = []
        if details.description:
            parts.append(str(details.description))
        for section in details.card_sections or []:
            if not isinstance(section, dict):
                continue
            title = str(section.get("title") or "").strip()
            text = str(section.get("text") or "").strip()
            if title:
                parts.append(title)
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    @staticmethod
    def _flatten_declared_data_files(details: HFDatasetDetails) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        for _config, split_map in (details.declared_data_files_by_config or {}).items():
            for split, paths in (split_map or {}).items():
                bucket = merged.setdefault(str(split), [])
                for path in paths or []:
                    path = str(path or "").strip()
                    if path and path not in bucket:
                        bucket.append(path)
        return merged

    def _apply_dataset_card_metadata(
        self, details: HFDatasetDetails, info: Any, *, repo_id: str, token: str | None = None,
    ) -> None:
        card_data = self._card_data_dict(
            getattr(info, "card_data", None) or getattr(info, "cardData", None) or {}
        )
        details.declared_data_files_by_config = self._declared_data_files_from_card(card_data)
        if details.declared_data_files_by_config:
            declared_configs = list(details.declared_data_files_by_config)
            details.configs = declared_configs or details.configs
            for config_name, split_map in details.declared_data_files_by_config.items():
                declared_splits = [str(split) for split in split_map if str(split).strip()]
                if declared_splits:
                    details.splits_by_config.setdefault(config_name, [])
                    for split in declared_splits:
                        if split not in details.splits_by_config[config_name]:
                            details.splits_by_config[config_name].append(split)
        details.task_categories, details.task_ids = self._task_metadata(card_data)
        details.pretty_name = str(card_data.get("pretty_name") or "").strip()
        details.card_source_url = (
            "https://huggingface.co/datasets/" + urllib.parse.quote(str(repo_id or "").strip(), safe="/")
        )

        siblings = {
            str(getattr(item, "rfilename", "") or "").strip()
            for item in (getattr(info, "siblings", None) or [])
        }

        # The Hub page itself is rendered from README.md. Prefer that complete,
        # immutable card over DatasetInfo.description, which can be truncated or
        # reduced to generated Installation/Usage boilerplate depending on the
        # hub/client response shape.
        if "README.md" in siblings:
            try:
                readme = self.download_file(
                    repo_id, "README.md", token=token, revision=details.revision or None, quiet=True,
                )
                raw = Path(readme).read_text(encoding="utf-8", errors="replace")
                details.card_detection_text = raw[:262144]
                sections = self._extract_dataset_card_sections(raw)
                if sections:
                    details.card_sections = sections
                    details.description = self._extract_dataset_description_text(raw)
                    details.description_source = "Pinned Hugging Face dataset card (README.md)"
                    return
            except Exception:
                pass

        candidates: list[tuple[str, Any]] = [
            ("Hugging Face dataset card", getattr(info, "description", None)),
            ("Hugging Face dataset card metadata", card_data.get("description")),
            ("Hugging Face dataset card metadata", card_data.get("summary")),
            ("Hugging Face dataset card metadata", card_data.get("dataset_summary")),
        ]
        for source, raw in candidates:
            sections = self._extract_dataset_card_sections(raw)
            if not sections:
                continue
            details.card_detection_text = str(raw or "")[:262144]
            details.card_sections = sections
            details.description = self._extract_dataset_description_text(raw)
            details.description_source = source
            return

    def get_dataset_details_viewer(
        self, repo_id: str, *, config_name: str | None = None, token: str | None = None,
    ) -> HFDatasetDetails:
        """Inspect a dataset through Hugging Face's precomputed Dataset Viewer APIs."""
        repo_id = str(repo_id or "").strip()
        if not repo_id:
            raise ValueError("repo_id is required.")

        details = HFDatasetDetails(repo_id=repo_id, inspection_source="dataset_viewer")
        sibling_names: set[str] = set()
        try:
            info = self.dataset_info_lightweight(repo_id, token=token)
            details.revision = str(getattr(info, "sha", "") or "")
            siblings = list(getattr(info, "siblings", []) or [])
            sibling_names = {
                str(getattr(item, "rfilename", None) or getattr(item, "path", None) or "").strip()
                for item in siblings
            }
            sibling_names.discard("")
            details.total_repo_files_hint = len(siblings) if siblings else None
            self._apply_dataset_card_metadata(details, info, repo_id=repo_id, token=token)
        except Exception as exc:
            details.warnings.append("Could not resolve Hub revision/card metadata. " + self._short_exception(exc))

        try:
            self.enrich_structured_annotation_details(details, token=token)
        except Exception as exc:
            details.warnings.append(
                "Structured annotation inventory unavailable. " + self._short_exception(exc)
            )

        if {FIFTYONE_METADATA_FILENAME, FIFTYONE_SAMPLES_FILENAME}.issubset(sibling_names):
            schema = self.get_fiftyone_schema(repo_id, revision=details.revision or None, token=token)
            if schema is not None:
                details.repository_format_hint = "fiftyone"
                details.fiftyone_schema = schema.to_dict()
                details.ground_truth_field_hint = str(schema.ground_truth_field or "")
                details.prediction_field_hints = list(schema.prediction_fields)
                if schema.ground_truth_field:
                    details.annotation_source_hint = (
                        f"FiftyOne {schema.ground_truth_field} Classification"
                    )

        try:
            split_payload = self._viewer_request("splits", dataset=repo_id, token=token)
        except Exception as exc:
            if details.structured_annotation_profile:
                details.inspection_source = "dataset_card+repository_inventory"
                details.warnings.append(
                    "Dataset Viewer split metadata unavailable; structured annotation "
                    "inspection remains available from the pinned card/repository inventory. "
                    + self._short_exception(exc)
                )
                return details
            raise
        split_rows = list(split_payload.get("splits") or [])
        if not split_rows:
            failed = split_payload.get("failed") or []
            if details.structured_annotation_profile:
                details.inspection_source = "dataset_card+repository_inventory"
                if failed:
                    details.warnings.append(
                        "Dataset Viewer returned no usable splits for this structured annotation "
                        "repository; using dataset-card split declarations instead."
                    )
                return details
            raise RuntimeError(f"Dataset Viewer returned no usable splits. {failed!r}")

        configs: list[str] = []
        for row in split_rows:
            if not isinstance(row, dict):
                continue
            config = str(row.get("config") or "default")
            split = str(row.get("split") or "train")
            if config not in configs:
                configs.append(config)
            details.splits_by_config.setdefault(config, [])
            if split not in details.splits_by_config[config]:
                details.splits_by_config[config].append(split)
        details.configs = configs or list(details.configs or ["default"])

        if details.structured_annotation_profile:
            # The annotation JSON itself can be hundreds of MiB or larger. Once
            # the card + pinned repository inventory identifies a structured
            # video format, inspection remains metadata-only and does not ask
            # Dataset Viewer to materialise/preview rows from that JSON.
            details.inspection_source = "dataset_card+repository_inventory"
            return details

        selected_config = str(config_name or "")
        if not selected_config or selected_config not in details.configs:
            selected_config = "default" if "default" in details.configs else details.configs[0]
        selected_splits = details.splits_by_config.get(selected_config) or ["train"]
        selected_split = selected_splits[0]

        try:
            first_rows = self._viewer_request(
                "first-rows", dataset=repo_id, config=selected_config, split=selected_split, token=token,
            )
        except Exception as exc:
            if details.structured_annotation_profile:
                first_rows = {"features": [], "rows": []}
                details.warnings.append(
                    "Dataset Viewer row preview unavailable for the structured annotation "
                    "file; no annotation JSON was materialised during inspection. "
                    + self._short_exception(exc)
                )
            else:
                raise
        feature_specs: dict[str, Any] = {}
        feature_text: dict[str, str] = {}
        for feature in first_rows.get("features") or []:
            if not isinstance(feature, dict):
                continue
            name = str(feature.get("name") or "").strip()
            if not name:
                continue
            spec = feature.get("type") or {}
            feature_specs[name] = spec
            feature_text[name] = self._viewer_feature_to_string(spec)
        details.features_by_config[selected_config] = feature_text
        details.feature_specs_by_config[selected_config] = feature_specs
        image_candidates, label_candidates = self._feature_candidates_from_features(feature_text)
        details.image_column_candidates = image_candidates
        details.label_column_candidates = label_candidates

        try:
            size_payload = self._viewer_request("size", dataset=repo_id, token=token)
            size = size_payload.get("size") or {}
            dataset_size = size.get("dataset") or {}
            details.total_rows_hint = int(dataset_size.get("num_rows") or 0) or None
            details.total_bytes_hint = int(dataset_size.get("num_bytes_original_files") or 0) or None
            for config_row in size.get("configs") or []:
                if not isinstance(config_row, dict):
                    continue
                cfg = str(config_row.get("config") or "default")
                for split_row in config_row.get("splits") or []:
                    if not isinstance(split_row, dict):
                        continue
                    sp = str(split_row.get("split") or "train")
                    count = int(split_row.get("num_rows") or 0)
                    details.row_counts_by_config.setdefault(cfg, {})[sp] = count
        except Exception as exc:
            details.warnings.append("Dataset size metadata unavailable. " + self._short_exception(exc))

        try:
            parquet_payload = self._viewer_request("parquet", dataset=repo_id, token=token)
            details.parquet_partial = bool(parquet_payload.get("partial", False))
            for item in parquet_payload.get("parquet_files") or []:
                if not isinstance(item, dict):
                    continue
                cfg = str(item.get("config") or "default")
                sp = str(item.get("split") or "train")
                details.parquet_files_by_config.setdefault(cfg, {}).setdefault(sp, []).append(dict(item))
        except Exception as exc:
            details.warnings.append("Parquet inventory unavailable. " + self._short_exception(exc))

        # Dataset Viewer is intentionally the primary inspection source, but raw
        # image repositories can legitimately have no row/size/ClassLabel metadata.
        # In that case one pinned repository-tree listing gives us file counts,
        # byte sizes and path-label coverage without triggering a builder resolve or
        # one remote HEAD per file.
        if details.image_column_candidates and (
            details.total_rows_hint is None or details.total_bytes_hint is None
            or not details.label_column_candidates or not (details.task_categories or details.task_ids)
        ):
            self.enrich_details_from_repository_inventory(details, token=token)

        return details

    def preview_image_items_viewer(
        self, *, repo_id: str, config_name: str | None, split: str, image_column: str | None = None,
        label_column: str | None = None, id_column: str | None = None, limit: int = 12,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Preview image rows without downloading the dataset repository locally."""
        config = str(config_name or "default")
        split = str(split or "train")
        limit = max(1, min(100, int(limit or 12)))
        payload = self._viewer_request(
            "first-rows", dataset=repo_id, config=config, split=split, token=token
        )
        features = {
            str(item.get("name")): item.get("type") or {} for item in payload.get("features") or []
            if isinstance(item, dict) and item.get("name")
        }
        feature_text = {name: self._viewer_feature_to_string(spec) for name, spec in features.items()}
        image_candidates, label_candidates = self._feature_candidates_from_features(feature_text)
        image_column = image_column or (image_candidates[0] if image_candidates else None)
        label_column = label_column or (label_candidates[0] if label_candidates else None)
        if not image_column:
            raise ValueError("Dataset Viewer preview found no image column.")

        items: list[dict[str, Any]] = []
        for position, wrapper in enumerate((payload.get("rows") or [])[:limit]):
            if not isinstance(wrapper, dict):
                continue
            row = wrapper.get("row") or {}
            if not isinstance(row, dict):
                continue
            row_index = int(wrapper.get("row_idx") if wrapper.get("row_idx") is not None else position)
            record_id = row.get(id_column) if id_column and id_column in row else None
            if record_id is None:
                record_id = self.record_id_for_row(row, index=row_index, split=split, id_column=id_column)
            label_value = row.get(label_column) if label_column else None
            label_name = self._viewer_label_to_name(features.get(label_column or ""), label_value)
            image_value = row.get(image_column)
            image_src = ""
            if isinstance(image_value, dict):
                image_src = str(image_value.get("src") or "")
            items.append({
                "record_id": str(record_id), "label": label_name, "image_column": str(image_column),
                "label_column": str(label_column or ""), "image_src": image_src,
                "source": "Hugging Face Dataset Viewer",
            })
        return {
            "kind": "images", "items": items, "count": len(items),
            "cache_source": "Dataset Viewer (no local dataset download)",
            "note": "Preview rows were served by Hugging Face's precomputed Dataset Viewer.",
        }

    def preview_rows_viewer(
        self, *, repo_id: str, config_name: str | None, split: str, limit: int = 12,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Return a bounded generic preview for text/audio/tabular/nested datasets."""
        config = str(config_name or "default")
        split = str(split or "train")
        limit = max(1, min(100, int(limit or 12)))
        payload = self._viewer_request(
            "first-rows", dataset=repo_id, config=config, split=split, token=token
        )
        columns = [
            str(item.get("name")) for item in payload.get("features") or []
            if isinstance(item, dict) and item.get("name")
        ]

        def compact(value: Any, *, depth: int = 0) -> Any:
            if depth >= 2:
                text = str(value)
                return text if len(text) <= 180 else text[:177] + "…"
            if isinstance(value, dict):
                # Preserve media metadata but do not expose expiring signed URLs in tables.
                if "src" in value and set(value).issubset({"src", "height", "width", "sampling_rate"}):
                    summary = {key: value.get(key) for key in ("height", "width", "sampling_rate") if value.get(key) is not None}
                    summary["media"] = "available"
                    return summary
                return {str(k): compact(v, depth=depth + 1) for k, v in list(value.items())[:12]}
            if isinstance(value, (list, tuple)):
                return [compact(item, depth=depth + 1) for item in list(value)[:12]]
            text = str(value) if not isinstance(value, (str, int, float, bool)) and value is not None else value
            if isinstance(text, str) and len(text) > 240:
                return text[:237] + "…"
            return text

        rows: list[dict[str, Any]] = []
        for wrapper in (payload.get("rows") or [])[:limit]:
            if not isinstance(wrapper, dict):
                continue
            row = wrapper.get("row") or {}
            if not isinstance(row, dict):
                continue
            rows.append({str(key): compact(value) for key, value in row.items()})
        return {
            "kind": "table", "rows": rows, "columns": columns or list(rows[0]) if rows else columns,
            "count": len(rows), "cache_source": "Dataset Viewer (no local dataset download)",
            "note": "Preview rows were served by Hugging Face's precomputed Dataset Viewer.",
        }

    @staticmethod
    def _viewer_label_to_name(feature_spec: Any, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(feature_spec, dict) and str(feature_spec.get("_type") or "") == "ClassLabel":
            names = list(feature_spec.get("names") or [])
            try:
                return str(names[int(value)])
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value)
        return str(value)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _load_dataset_builder_local_only(
        self, repo_id: str, *, config_name: str | None = None, token: str | None = None,
        trust_remote_code: bool = False, revision: str | None = None,
    ):
        from datasets import load_dataset_builder

        def _load(kwargs: dict[str, Any]):
            try:
                if config_name:
                    return load_dataset_builder(repo_id, name=config_name, **kwargs)
                return load_dataset_builder(repo_id, **kwargs)
            except TypeError:
                retry = dict(kwargs)
                retry.pop("trust_remote_code", None)
                retry.pop("token", None)
                if config_name:
                    return load_dataset_builder(repo_id, name=config_name, **retry)
                return load_dataset_builder(repo_id, **retry)

        for cache_root in self._datasets_cache_roots():
            kwargs = self._datasets_kwargs_for_cache(
                token=token, trust_remote_code=trust_remote_code, cache_root=cache_root,
                local_files_only=True,
            )
            if revision not in (None, ""):
                kwargs["revision"] = str(revision)
            try:
                return _load(kwargs)
            except Exception:
                continue
        return None

    def load_dataset_builder_once(
        self, repo_id: str, *, config_name: str | None = None, token: str | None = None,
        trust_remote_code: bool = False,
    ):
        """Load a builder local-first, contacting the Hub only as fallback."""
        local = self._load_dataset_builder_local_only(
            repo_id, config_name=config_name, token=token, trust_remote_code=trust_remote_code,
        )
        if local is not None:
            return local

        from datasets import load_dataset_builder

        kwargs = self._datasets_kwargs_for_cache(
            token=token, trust_remote_code=trust_remote_code, cache_root=self.configured_cache_dir,
            local_files_only=False,
        )
        try:
            if config_name:
                return load_dataset_builder(repo_id, name=config_name, **kwargs)
            return load_dataset_builder(repo_id, **kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            kwargs.pop("token", None)
            if config_name:
                return load_dataset_builder(repo_id, name=config_name, **kwargs)
            return load_dataset_builder(repo_id, **kwargs)

    def get_dataset_details_builder_local(
        self, repo_id: str, *, config_name: str | None = None, token: str | None = None,
        trust_remote_code: bool = False, revision: str | None = None,
    ) -> HFDatasetDetails | None:
        builder = self._load_dataset_builder_local_only(
            repo_id, config_name=config_name, token=token, trust_remote_code=trust_remote_code,
            revision=revision,
        )
        if builder is None:
            return None

        config_key = str(
            getattr(getattr(builder, "config", None), "name", None) or config_name or "default"
        )
        details = HFDatasetDetails(
            repo_id=str(repo_id), revision=str(revision or ""), inspection_source="local_datasets_builder",
            configs=[config_key],
        )
        info = getattr(builder, "info", None)
        features = getattr(info, "features", None) or {}
        splits = getattr(info, "splits", None) or {}
        if splits:
            try:
                split_names = [str(name) for name in splits.keys()]
            except Exception:
                split_names = [str(name) for name in splits]
        else:
            split_names = ["train"]
        details.splits_by_config[config_key] = split_names or ["train"]
        details.features_by_config[config_key] = {
            str(name): self._feature_to_string(feature) for name, feature in dict(features).items()
        }
        image_candidates, label_candidates = self._feature_candidates_from_features(
            details.features_by_config[config_key]
        )
        details.image_column_candidates = image_candidates or (["image"] if "image" in details.features_by_config[config_key] else [])
        details.label_column_candidates = label_candidates
        if not details.features_by_config[config_key] and not details.image_column_candidates:
            return None
        return details

    def search_datasets(
        self, *, query: str = "", task: str = DEFAULT_TASK_FILTER, limit: int = 25, sort: str = "downloads",
        token: str | None = None,
    ) -> list[HFDatasetSearchResult]:
        from huggingface_hub import HfApi

        resolved_token = self.effective_token(token)
        api = HfApi(token=resolved_token)

        requested_limit = max(int(limit or 25), 1)

        base_kwargs: dict[str, Any] = {"limit": requested_limit, "full": True}
        if query:
            base_kwargs["search"] = query
        if sort:
            base_kwargs["sort"] = sort

        attempts: list[dict[str, Any]] = []

        if task:
            attempt = dict(base_kwargs)
            attempt["filter"] = task
            attempts.append(attempt)

            attempt = dict(base_kwargs)
            attempt["task_categories"] = task
            attempts.append(attempt)

        attempts.append(dict(base_kwargs))

        minimal: dict[str, Any] = {"limit": requested_limit}
        if query:
            minimal["search"] = query
        attempts.append(minimal)

        last_exc: BaseException | None = None
        infos = None

        for attempt in attempts:
            try:
                infos = list(api.list_datasets(**attempt))
                break
            except TypeError as exc:
                last_exc = exc
                continue

        if infos is None:
            if last_exc:
                raise last_exc
            infos = []

        if query and "/" in query:
            exact_query = str(query).strip()
            try:
                exact_info = api.dataset_info(exact_query, files_metadata=False, token=resolved_token)
            except TypeError:
                try:
                    exact_info = api.dataset_info(exact_query, token=resolved_token)
                except Exception:
                    exact_info = None
            except Exception:
                exact_info = None

            if exact_info is not None:
                exact_id = (
                    getattr(exact_info, "id", None) or getattr(exact_info, "repo_id", None) or exact_query
                )
                infos = [
                    exact_info, *[
                        info for info in infos if str(
                            getattr(info, "id", None) or getattr(info, "repo_id", None) or ""
                        ).casefold() != str(exact_id).casefold()
                    ],
                ]

        results: list[HFDatasetSearchResult] = []
        for info in infos:
            repo_id = (
                getattr(info, "id", None) or getattr(info, "repo_id", None) or getattr(info, "name", None)
                or ""
            )
            if not repo_id:
                continue

            tags = list(getattr(info, "tags", []) or [])

            last_modified = (getattr(info, "lastModified", None) or getattr(info, "last_modified", None))
            if last_modified is not None:
                last_modified = str(last_modified)

            card_data = (getattr(info, "cardData", None) or getattr(info, "card_data", None) or {})
            summary = ""
            if isinstance(card_data, dict):
                summary = str(card_data.get("summary") or card_data.get("description") or "")

            gating_mode = self._normalise_gating_mode(getattr(info, "gated", None))
            revision = str(getattr(info, "sha", "") or "").strip()
            access_status = (
                self.dataset_access_status(
                    str(repo_id), revision=revision or None, token=resolved_token, info=info,
                ) if gating_mode else HFDatasetAccessStatus(
                    repo_id=str(repo_id), gated=False, gating_mode="", state="not_gated", accessible=True,
                    token_available=bool(resolved_token), access_url=(
                        "https://huggingface.co/datasets/" + urllib.parse.quote(str(repo_id), safe="/")
                    ), message="This dataset is not gated.",
                )
            )

            results.append(
                HFDatasetSearchResult(
                    repo_id=str(repo_id), author=str(getattr(info, "author", "") or "").strip(),
                    downloads=getattr(info, "downloads", None), likes=getattr(info, "likes", None),
                    last_modified=str(last_modified or ""), tags=tags, gated=gating_mode,
                    private=bool(getattr(info, "private", False)), card_summary=summary, revision=revision,
                    access_state=access_status.state, accessible=bool(access_status.accessible),
                    token_available=bool(access_status.token_available),
                    authenticated_user=access_status.authenticated_user,
                    access_message=access_status.message, access_probe_file=access_status.probe_file,
                    access_error=access_status.error,
                )
            )

        if sort == "downloads":
            results.sort(key=lambda r: r.downloads or 0, reverse=True)
        elif sort == "likes":
            results.sort(key=lambda r: r.likes or 0, reverse=True)
        elif sort in {"lastModified", "last_modified"}:
            results.sort(key=lambda r: r.last_modified or "", reverse=True)

        return results[:requested_limit]

    def search_dataframe(
        self, *, query: str = "", task: str = DEFAULT_TASK_FILTER, limit: int = 25, sort: str = "downloads",
        token: str | None = None,
    ) -> pd.DataFrame:
        rows = []
        for result in self.search_datasets(query=query, task=task, limit=limit, sort=sort, token=token):
            cache = self.cache_status(result.repo_id)
            rows.append(
                {
                    "repo_id": result.repo_id, "cached": "Yes" if cache.cached else "",
                    "cache": cache.label, "downloads": result.downloads, "likes": result.likes,
                    "last_modified": result.last_modified, "gated": result.gated,
                    "gating_mode": result.gated, "access_state": result.access_state,
                    "accessible": result.accessible, "token_available": result.token_available,
                    "authenticated_user": result.authenticated_user,
                    "access_message": result.access_message, "access_probe_file": result.access_probe_file,
                    "access_error": result.access_error, "revision": result.revision,
                    "private": result.private, "tags": ", ".join(result.tags[:8]),
                    "summary": result.card_summary,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Lightweight inspect
    # ------------------------------------------------------------------

    def feature_text_for_column(self, details: Any, *, config: str, column: str) -> str:
        try:
            features = details.features_by_config.get(config, {}) or {}
            return str(features.get(column, ""))
        except Exception:
            return ""

    def feature_looks_like_shard_labels(self, feature_text: str) -> bool:
        """
        Detect ClassLabel names such as data_0, data_1, data_10, shard_0, part_0.

        These are usually storage shard folders, not semantic class labels.
        """

        text = str(feature_text or "")
        lowered = text.lower()

        if "classlabel" not in lowered and "names=" not in lowered:
            return False

        names = re.findall(r"[\"']([^\"']+)[\"']", text)
        if not names:
            return False

        if len(names) < 3:
            return False

        shard_like = 0
        for name in names:
            cleaned = str(name).strip().lower()
            if self._looks_like_non_label_folder(cleaned):
                shard_like += 1

        return shard_like / max(len(names), 1) >= 0.8

    def selected_builder_label_is_non_semantic(
        self, details: Any, *, config: str, label_column: str | None,
    ) -> bool:
        if not label_column:
            return True

        feature_text = self.feature_text_for_column(details, config=config, column=label_column)
        return self.feature_looks_like_shard_labels(feature_text)

    def _looks_like_non_label_folder(self, folder_name: str) -> bool:
        cleaned = str(folder_name or "").strip().lower()
        if cleaned in GENERIC_FOLDERS:
            return True

        for pattern in GENERIC_FOLDER_PATTERNS:
            if re.match(pattern, cleaned):
                return True

        return False

    def _feature_candidates_from_features(self, features: dict[str, Any]) -> tuple[list[str], list[str]]:
        image_candidates: list[str] = []
        label_candidates: list[str] = []

        for column, feature_type in dict(features or {}).items():
            col = str(column)
            text = str(feature_type).lower()
            col_lower = col.lower()

            if ("image" in text or col_lower in {"image", "img", "picture", "photo", "filepath", "path"}):
                if col not in image_candidates:
                    image_candidates.append(col)

            if (
                "classlabel" in text or "class label" in text or "label" in text
                or col_lower in {"label", "labels", "target", "class", "category"}
            ):
                if col not in label_candidates:
                    label_candidates.append(col)

        return image_candidates, label_candidates

    def get_dataset_details_builder(
        self, repo_id: str, *, config_name: str | None = None, token: str | None = None,
        trust_remote_code: bool = False,
    ) -> HFDatasetDetails:
        """Inspect the selected config with one Datasets builder call to avoid repeated data-file
        resolution.
        """

        repo_id = str(repo_id or "").strip()
        if not repo_id:
            raise ValueError("repo_id is required.")

        details = HFDatasetDetails(repo_id=repo_id, inspection_source="datasets_builder")
        try:
            info = self.dataset_info_lightweight(repo_id, token=token)
            details.revision = str(getattr(info, "sha", "") or "")
            self._apply_dataset_card_metadata(details, info, repo_id=repo_id, token=token)
        except Exception:
            pass

        config_key = config_name or "default"
        details.configs = [config_key]
        details.splits_by_config[config_key] = ["train"]
        details.scanned_file_count = 0

        print(
            f"[AL_DEBUG][HF][builder_inspect] START repo_id={repo_id!r} config={config_name!r}", flush=True,
        )

        try:
            builder = self.load_dataset_builder_once(
                repo_id, config_name=config_name, token=token, trust_remote_code=trust_remote_code,
            )

            actual_config = getattr(getattr(builder, "config", None), "name", None)
            if actual_config:
                config_key = str(actual_config)
                details.configs = [config_key]

            info = getattr(builder, "info", None)
            features = getattr(info, "features", None) or {}
            splits = getattr(info, "splits", None) or {}

            if splits:
                try:
                    split_names = [str(name) for name in splits.keys()]
                except Exception:
                    split_names = [str(name) for name in splits]
                details.splits_by_config[config_key] = split_names or ["train"]
            else:
                details.splits_by_config[config_key] = ["train"]

            details.features_by_config[config_key] = {
                str(name): self._feature_to_string(feature) for name, feature in dict(features).items()
            }

            image_candidates, label_candidates = self._feature_candidates_from_features(
                details.features_by_config[config_key]
            )

            details.image_column_candidates = image_candidates or ["image"]
            details.label_column_candidates = label_candidates

            # Strong fallback for common names.
            columns = list(details.features_by_config[config_key].keys())
            if "label" in columns and "label" not in details.label_column_candidates:
                details.label_column_candidates.insert(0, "label")
            if "labels" in columns and "labels" not in details.label_column_candidates:
                details.label_column_candidates.append("labels")

            print(
                f"[AL_DEBUG][HF][builder_inspect] DONE repo_id={repo_id!r} "
                f"config={config_key!r} splits={details.splits_by_config[config_key]!r} "
                f"columns={list(details.features_by_config[config_key].keys())!r}", flush=True,
            )

            return details

        except Exception as exc:
            details.warnings.append("Single-call builder inspection failed. " + self._short_exception(exc))
            print(f"[AL_DEBUG][HF][builder_inspect] ERROR repo_id={repo_id!r} error={exc!r}", flush=True)

        # Fallback: stream one row only. This may still trigger a resolve once, but
        # avoids the previous pattern of several separate builder helper calls.
        try:
            sample_dataset = self.load_split(
                repo_id=repo_id, config_name=config_name, split="train", token=token,
                trust_remote_code=trust_remote_code, streaming=True,
            )
            first_row = dict(next(iter(sample_dataset)))
            features = {str(key): type(value).__name__ for key, value in first_row.items()}
            details.features_by_config[config_key] = features
            details.splits_by_config[config_key] = ["train"]

            image_candidates, label_candidates = self._feature_candidates_from_features(features)
            details.image_column_candidates = image_candidates or ["image"]
            details.label_column_candidates = label_candidates

            details.warnings.append(
                "Builder metadata was unavailable; inferred columns from one streamed row."
            )
        except Exception as exc:
            details.features_by_config[config_key] = {"image": "Image fallback"}
            details.image_column_candidates = ["image"]
            details.label_column_candidates = []
            details.warnings.append(
                "Could not infer builder columns from a streamed sample. " + self._short_exception(exc)
            )

        return details

    def get_config_names_builder(
        self, repo_id: str, *, token: str | None = None, trust_remote_code: bool = False,
    ) -> list[str]:
        from datasets import get_dataset_config_names

        kwargs = self._hf_kwargs(token=token, trust_remote_code=trust_remote_code)

        try:
            configs = list(get_dataset_config_names(repo_id, **kwargs))
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            configs = list(get_dataset_config_names(repo_id, **kwargs))

        return [str(config) for config in configs] or ["default"]

    def get_split_names_builder(
        self, repo_id: str, *, config_name: str | None = None, token: str | None = None,
        trust_remote_code: bool = False,
    ) -> list[str]:
        from datasets import get_dataset_split_names

        kwargs = self._hf_kwargs(token=token, trust_remote_code=trust_remote_code)

        try:
            if config_name:
                splits = list(get_dataset_split_names(repo_id, config_name, **kwargs))
            else:
                splits = list(get_dataset_split_names(repo_id, **kwargs))
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            if config_name:
                splits = list(get_dataset_split_names(repo_id, config_name, **kwargs))
            else:
                splits = list(get_dataset_split_names(repo_id, **kwargs))

        return [str(split) for split in splits] or ["train"]

    def get_features_builder(
        self, repo_id: str, *, config_name: str | None = None, token: str | None = None,
        trust_remote_code: bool = False,
    ) -> dict[str, str]:
        from datasets import load_dataset_builder

        kwargs = self._hf_kwargs(token=token, trust_remote_code=trust_remote_code)

        try:
            if config_name:
                builder = load_dataset_builder(repo_id, name=config_name, **kwargs)
            else:
                builder = load_dataset_builder(repo_id, **kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            if config_name:
                builder = load_dataset_builder(repo_id, name=config_name, **kwargs)
            else:
                builder = load_dataset_builder(repo_id, **kwargs)

        features = getattr(getattr(builder, "info", None), "features", None) or {}
        return {str(name): self._feature_to_string(feature) for name, feature in features.items()}

    def preview_image_grid_html_builder(
        self, *, repo_id: str, config_name: str | None, split: str, image_column: str | None = None,
        label_column: str | None = None, id_column: str | None = None, limit: int = 12,
        token: str | None = None, trust_remote_code: bool = False, thumb_size: int = 180,
        revision: str | None = None,
    ) -> str:
        result = self.preview_image_items_builder(
            repo_id=repo_id, config_name=config_name, split=split, image_column=image_column,
            label_column=label_column, id_column=id_column, limit=limit, token=token,
            trust_remote_code=trust_remote_code, thumb_size=thumb_size, revision=revision,
        )
        return self._preview_result_to_html(result, thumb_size=thumb_size)

    def preview_image_items_builder(
        self, *, repo_id: str, config_name: str | None, split: str, image_column: str | None = None,
        label_column: str | None = None, id_column: str | None = None, limit: int = 12,
        token: str | None = None, trust_remote_code: bool = False, thumb_size: int = 180,
        revision: str | None = None,
    ) -> dict[str, Any]:
        """Preview structured rows, using revision-scoped local Arrow data first."""
        repo_id = str(repo_id or "").strip()
        config_key = str(config_name or "default")
        split = str(split or "train")
        limit = max(1, int(limit or 12))
        thumb_size = max(32, int(thumb_size or 180))
        auth_scope = self._details_cache_auth_scope(token)
        key = (
            "builder", repo_id, config_key, split, str(revision or ""), auth_scope,
            str(image_column or ""),
            str(label_column or ""), str(id_column or ""), limit, thumb_size,
        )
        cached = self._preview_cache_get(key)
        if cached is not None:
            cached["cache_source"] = "preview memory cache"
            return cached

        dataset_key = (repo_id, config_key, split, str(revision or ""), auth_scope)
        with self._preview_lock:
            dataset = self._preview_dataset_cache.get(dataset_key)
        cache_source = "prepared local Datasets cache"

        if dataset is None:
            dataset = self._load_split_local_only(
                repo_id=repo_id, config_name=config_name, split=split, token=token,
                trust_remote_code=trust_remote_code, streaming=False, revision=revision,
            )
            if dataset is not None:
                with self._preview_lock:
                    self._preview_dataset_cache[dataset_key] = dataset

        if dataset is None:
            cache_source = "streamed from Hugging Face"
            dataset = self.load_split(
                repo_id=repo_id, config_name=config_name, split=split, token=token,
                trust_remote_code=trust_remote_code, streaming=True, revision=revision,
            )

        features = getattr(dataset, "features", {}) or {}
        image_column = image_column or self.infer_image_column(features, dataset=dataset) or "image"
        label_column = label_column or self.infer_label_column(features, dataset=dataset)

        # Decode=False keeps row access cheap and lets thumbnail conversion run
        # concurrently below instead of decoding every image serially here.
        preview_dataset = self.cast_image_decode_false(dataset, image_column)

        import itertools
        if hasattr(preview_dataset, "select"):
            try:
                available = len(preview_dataset)
                rows_iter = (preview_dataset[index] for index in range(min(limit, available)))
            except Exception:
                rows_iter = itertools.islice(iter(preview_dataset), limit)
        else:
            rows_iter = itertools.islice(iter(preview_dataset), limit)

        raw_rows = [(index, dict(row)) for index, row in enumerate(rows_iter)]
        if not label_column:
            for _index, row in raw_rows:
                for candidate in ("label", "labels", "target", "class", "category"):
                    if candidate in row:
                        label_column = candidate
                        break
                if label_column:
                    break

        def build_item(index_and_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            index, row = index_and_row
            record_id = self.record_id_for_row(row, index=index, split=split, id_column=id_column)
            label_value = row.get(label_column) if label_column else ""
            label_name = self.label_to_name(features, label_column, label_value)
            label_display = label_name if label_name != "" else label_value
            try:
                data_uri = self.image_value_to_data_uri(row.get(image_column), max_size=thumb_size)
            except Exception:
                data_uri = ""
            return {
                "record_id": str(record_id), "label": "" if label_display is None else str(label_display),
                "image_column": str(image_column or ""), "label_column": str(label_column or ""),
                "source": cache_source, "data_uri": data_uri,
            }

        workers = min(8, max(1, len(raw_rows)))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                items = list(executor.map(build_item, raw_rows))
        else:
            items = [build_item(row) for row in raw_rows]

        result = {
            "items": items, "count": len(items), "cache_source": cache_source, "note": (
                "Prepared local dataset data were used." if cache_source.startswith("prepared")
                else "No prepared local copy was available, so preview rows were streamed."
            ),
        }
        self._preview_cache_put(key, result)
        return result

    def image_value_to_data_uri(self, value: Any, *, max_size: int = 180) -> str:
        from PIL import Image as PILImage

        if value is None:
            return ""
        if hasattr(value, "copy") and hasattr(value, "save"):
            try:
                return self.image_to_preview_data_uri(value.copy(), size=int(max_size or 180), fill=True)
            except Exception:
                pass

        raw: bytes | None = None
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                raw = value["bytes"]
            elif value.get("path"):
                path_text = str(value["path"])
                if path_text.startswith("hf://"):
                    return ""
                path = Path(path_text).expanduser()
                if path.exists():
                    return self.local_image_to_data_uri(path, max_size=max_size)
        elif isinstance(value, (str, Path)):
            text = str(value)
            if text.startswith("hf://"):
                return ""
            path = Path(text).expanduser()
            if path.exists():
                return self.local_image_to_data_uri(path, max_size=max_size)

        if raw is None:
            return ""
        with PILImage.open(BytesIO(raw)) as image:
            return self.image_to_preview_data_uri(image, size=int(max_size or 180), fill=True)

    def get_dataset_details(
        self, repo_id: str, *, token: str | None = None, trust_remote_code: bool = False,
    ) -> HFDatasetDetails:
        repo_id = str(repo_id or "").strip()
        if not repo_id:
            raise ValueError("repo_id is required.")

        details = HFDatasetDetails(repo_id=repo_id, inspection_source="repository_files")
        resolved_token = token or self.token

        try:
            info = self.dataset_info_lightweight(repo_id, token=resolved_token)
            details.revision = str(getattr(info, "sha", "") or "")
            self._apply_dataset_card_metadata(details, info, repo_id=repo_id, token=token)
            siblings = list(getattr(info, "siblings", []) or [])
            if siblings:
                details.total_repo_files_hint = len(siblings)
        except Exception as exc:
            details.warnings.append("Could not read Hub dataset metadata. " + self._short_exception(exc))

        try:
            self.enrich_structured_annotation_details(details, token=resolved_token)
        except Exception as exc:
            details.warnings.append(
                "Structured annotation inventory unavailable. " + self._short_exception(exc)
            )
        if details.structured_annotation_profile:
            details.inspection_source = "dataset_card+repository_inventory"
            if details.total_repo_files_hint is not None:
                details.warnings.append(
                    f"Hub metadata reports {details.total_repo_files_hint} top-level/listed sibling file(s)."
                )
            return details

        try:
            sample_files = self.list_image_files(repo_id, token=resolved_token, max_files=500)
            details.sample_files = sample_files
            details.scanned_file_count = len(sample_files)

            splits = sorted({file.split for file in sample_files if file.split})
            if not splits:
                splits = ["train"]
            details.splits_by_config["default"] = splits

            labels = sorted({file.label for file in sample_files if file.label})
            if labels:
                details.warnings.append(f"Detected {len(labels)} label folder(s) in sampled files.")
        except Exception as exc:
            details.splits_by_config["default"] = ["train"]
            details.warnings.append(
                "Could not sample repository image files. " + self._short_exception(exc)
            )

        details.features_by_config["default"] = {
            "__hf_file__": "Hub image files; batch downloaded with snapshot_download",
            "__hf_path_label__": "Inferred label from parent folder, if available",
        }
        details.image_column_candidates = ["__hf_file__"]
        details.label_column_candidates = ["__hf_path_label__"]

        if details.total_repo_files_hint is not None:
            details.warnings.append(
                f"Hub metadata reports {details.total_repo_files_hint} top-level/listed sibling file(s)."
            )

        return details

    def dataset_info_lightweight(self, repo_id: str, *, token: str | None = None):
        from huggingface_hub import HfApi

        resolved_token = token or self.token
        api = HfApi(token=resolved_token)

        try:
            return api.dataset_info(
                repo_id, files_metadata=False, expand=[
                    "author", "cardData", "description", "downloads", "gated", "lastModified", "likes",
                    "private", "siblings", "sha", "tags",
                ], token=resolved_token,
            )
        except TypeError:
            return api.dataset_info(repo_id, files_metadata=False, token=resolved_token)

    @staticmethod
    def _normalise_gating_mode(value: Any) -> str:
        if isinstance(value, bool):
            return "gated" if value else ""
        text = str(value or "").strip().casefold()
        if text in {"", "false", "none", "null", "0", "no"}:
            return ""
        if text in {"auto", "automatic"}:
            return "auto"
        if text in {"manual", "manual-review", "manual_review"}:
            return "manual"
        return "gated"

    @staticmethod
    def _access_probe_file(info: Any) -> str:
        names = []
        for sibling in getattr(info, "siblings", None) or []:
            name = str(
                getattr(sibling, "rfilename", None) or getattr(sibling, "filename", None)
                or getattr(sibling, "path", None) or ""
            ).strip()
            if name:
                names.append(name)

        if not names:
            return ""

        metadata_names = {"README.md", ".gitattributes", ".gitignore"}
        data_candidates = [
            name for name in names if name not in metadata_names and not name.startswith(".")
        ]
        if data_candidates:
            return data_candidates[0]
        if "README.md" in names:
            return "README.md"
        return names[0]

    @staticmethod
    def _classify_gated_access_error(exc: BaseException) -> tuple[str, str]:
        text = " ".join(str(exc or "").split())
        lowered = text.casefold()
        class_name = exc.__class__.__name__.casefold()

        if "rejected" in lowered:
            return (
                "access_rejected",
                "Your Hugging Face access request appears to have been rejected by the dataset owner.",
            )
        if (
            "awaiting a review" in lowered or "awaiting review" in lowered or "pending approval" in lowered
            or "request is pending" in lowered
        ):
            return ("pending_approval", "Your Hugging Face access request is pending owner approval.")
        if (
            "region" in lowered or "country" in lowered or "geo" in lowered
        ) and (
            "restrict" in lowered or "block" in lowered or "not available" in lowered
            or "disallow" in lowered
        ):
            return (
                "region_restricted",
                "Hugging Face reports that this gated dataset is unavailable from the current region.",
            )
        if (
            "401" in lowered or "invalid token" in lowered or "invalid user token" in lowered
            or "authentication" in lowered and "invalid" in lowered
        ):
            return ("authentication_invalid", "The current Hugging Face token could not be authenticated.")
        if (
            "gatedrepoerror" in class_name or "gated repo" in lowered or "gated dataset" in lowered
            or "access to this dataset is restricted" in lowered or "403" in lowered
            or "request access" in lowered
        ):
            return (
                "access_request_required",
                "This Hugging Face dataset is gated and the current account does not yet have file access.",
            )
        return (
            "access_unknown",
            "AstronomicAL could not verify gated-file access with the current Hugging Face credentials.",
        )

    def _authenticated_user_for_token(self, token: str) -> str:
        """Cache the HF username by token fingerprint to avoid repeated whoami requests; keep raw tokens
        out of cache keys.
        """

        resolved_token = str(token or "").strip()
        if not resolved_token:
            return ""

        key = hashlib.sha256(resolved_token.encode("utf-8")).hexdigest()
        now = time.time()
        with self._auth_lock:
            cached = self._auth_identity_cache.get(key)
            if cached is not None and now - cached[0] <= 300.0:
                _cached_at, username, error = cached
                if error:
                    raise RuntimeError(error)
                return username

        try:
            from huggingface_hub import HfApi

            who = HfApi(token=resolved_token).whoami(token=resolved_token)
            username = ""
            if isinstance(who, dict):
                username = str(who.get("name") or who.get("fullname") or who.get("user") or "").strip()
        except Exception as exc:
            error = self._short_exception(exc)
            with self._auth_lock:
                self._auth_identity_cache[key] = (now, "", error)
            raise RuntimeError(error) from exc

        with self._auth_lock:
            self._auth_identity_cache[key] = (now, username, "")
        return username

    def dataset_access_status(
        self, repo_id: str, *, revision: str | None = None, token: str | None = None, info: Any = None,
    ) -> HFDatasetAccessStatus:
        """Check read access to a pinned repository file without downloading it; gated consent stays on the
        Hub.
        """

        repo_id = str(repo_id or "").strip()
        if not repo_id:
            raise ValueError("repo_id is required.")

        access_url = ("https://huggingface.co/datasets/" + urllib.parse.quote(repo_id, safe="/"))
        resolved_token = self.effective_token(token)

        try:
            info = info or self.dataset_info_lightweight(repo_id, token=resolved_token)
        except Exception as exc:
            state, message = self._classify_gated_access_error(exc)
            definitely_gated = state in {
                "access_request_required", "pending_approval", "access_rejected", "region_restricted",
            }
            return HFDatasetAccessStatus(
                repo_id=repo_id, gated=definitely_gated, gating_mode="gated" if definitely_gated else "",
                state=state if definitely_gated else "metadata_unavailable",
                accessible=not definitely_gated, token_available=bool(resolved_token),
                access_url=access_url, message=(
                    message if definitely_gated
                    else "AstronomicAL could not preflight Hugging Face access metadata; normal inspection will continue."
                ), error=self._short_exception(exc),
            )

        gating_mode = self._normalise_gating_mode(getattr(info, "gated", None))
        if not gating_mode:
            return HFDatasetAccessStatus(
                repo_id=repo_id, gated=False, gating_mode="", state="not_gated", accessible=True,
                token_available=bool(resolved_token), access_url=access_url,
                message="This dataset is not gated.",
            )

        if not resolved_token:
            return HFDatasetAccessStatus(
                repo_id=repo_id, gated=True, gating_mode=gating_mode, state="authentication_required",
                accessible=False, token_available=False, access_url=access_url, message=(
                    "This dataset is gated. Request or accept access on the Hugging Face "
                    "dataset page, then authenticate AstronomicAL with the same Hugging Face account."
                ),
            )

        authenticated_user = ""
        try:
            authenticated_user = self._authenticated_user_for_token(resolved_token)
        except Exception as exc:
            state, message = self._classify_gated_access_error(exc)
            if state == "access_unknown":
                state = "authentication_invalid"
                message = "The current Hugging Face token could not be authenticated."
            return HFDatasetAccessStatus(
                repo_id=repo_id, gated=True, gating_mode=gating_mode, state=state, accessible=False,
                token_available=True, access_url=access_url, message=message,
                error=self._short_exception(exc),
            )

        probe_file = self._access_probe_file(info)
        if not probe_file:
            # Empty repositories have no file whose authorization can be probed.
            # Keep this conservative rather than claiming gated access that has
            # not actually been demonstrated.
            return HFDatasetAccessStatus(
                repo_id=repo_id, gated=True, gating_mode=gating_mode, state="access_unknown",
                accessible=False, token_available=True, authenticated_user=authenticated_user,
                access_url=access_url, message=(
                    "The account is authenticated, but this gated repository exposes no "
                    "file that AstronomicAL can use for a non-downloading access check."
                ),
            )

        resolved_revision = (
            str(revision or "").strip() or str(getattr(info, "sha", "") or "").strip() or "main"
        )

        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url

            probe_url = hf_hub_url(
                repo_id=repo_id, filename=probe_file, repo_type="dataset", revision=resolved_revision,
            )
            get_hf_file_metadata(probe_url, token=resolved_token, timeout=10)
        except Exception as exc:
            state, message = self._classify_gated_access_error(exc)
            suffix = (
                " Open the dataset page in your browser, complete any required form, "
                "then click Re-check access in AstronomicAL."
            )
            if state in {"access_request_required", "pending_approval"}:
                message += suffix
            return HFDatasetAccessStatus(
                repo_id=repo_id, gated=True, gating_mode=gating_mode, state=state, accessible=False,
                token_available=True, authenticated_user=authenticated_user, access_url=access_url,
                probe_file=probe_file, message=message, error=self._short_exception(exc),
            )

        user_text = f" for {authenticated_user}" if authenticated_user else ""
        return HFDatasetAccessStatus(
            repo_id=repo_id, gated=True, gating_mode=gating_mode, state="access_granted", accessible=True,
            token_available=True, authenticated_user=authenticated_user, access_url=access_url,
            probe_file=probe_file, message=(
                f"Gated dataset access is granted{user_text}. AstronomicAL can inspect "
                "and download repository files with the current Hugging Face token."
            ),
        )

    def get_dataset_access_preview(
        self, repo_id: str, *, token: str | None = None, info: Any = None,
    ) -> HFDatasetDetails:
        """Show public metadata before access without inventing gated-repository splits or schema."""

        repo_id = str(repo_id or "").strip()
        if not repo_id:
            raise ValueError("repo_id is required.")

        details = HFDatasetDetails(
            repo_id=repo_id, inspection_source="hub_metadata_gated", configs=["default"],
        )
        try:
            info = info or self.dataset_info_lightweight(repo_id, token=self.effective_token(token))
            details.revision = str(getattr(info, "sha", "") or "")
            siblings = list(getattr(info, "siblings", []) or [])
            details.total_repo_files_hint = len(siblings) if siblings else None
            self._apply_dataset_card_metadata(details, info, repo_id=repo_id, token=token)
        except Exception as exc:
            details.warnings.append(
                "Could not read public Hub metadata before access was granted. "
                + self._short_exception(exc)
            )
        return details

    # ------------------------------------------------------------------
    # Native FiftyOne repository metadata
    # ------------------------------------------------------------------

    def get_fiftyone_schema(
        self, repo_id: str, *, revision: str | None = None, token: str | None = None,
    ) -> FiftyOneSchema | None:
        """Read and cache a pinned native FiftyOne metadata.json without importing FiftyOne."""
        repo_id = str(repo_id or "").strip()
        if not repo_id:
            return None
        resolved_revision = str(
            revision or self.resolve_dataset_revision(repo_id, revision=revision, token=token) or "main"
        )
        cache_key = f"{repo_id}@{resolved_revision}"
        with self._preview_lock:
            if cache_key in self._fiftyone_schema_cache:
                return self._fiftyone_schema_cache[cache_key]

        schema: FiftyOneSchema | None = None
        try:
            path = self.download_file(
                repo_id, FIFTYONE_METADATA_FILENAME, token=token, revision=resolved_revision, quiet=True,
            )
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            schema = parse_fiftyone_schema(payload)
        except Exception:
            schema = None

        with self._preview_lock:
            self._fiftyone_schema_cache[cache_key] = schema
        return schema

    def get_fiftyone_classification_index(
        self, repo_id: str, *, revision: str | None = None, token: str | None = None,
        cancel_token: Any = None, progress_callback: Any = None,
    ) -> FiftyOneSampleIndex | None:
        """Load bounded FiftyOne sample labels through a cancellable snapshot worker; allow fallback for
        large manifests.
        """
        repo_id = str(repo_id or "").strip()
        if not repo_id:
            return None
        resolved_revision = str(
            revision or self.resolve_dataset_revision(repo_id, revision=revision, token=token) or "main"
        )
        cache_key = f"{repo_id}@{resolved_revision}"
        with self._preview_lock:
            if cache_key in self._fiftyone_sample_index_cache:
                return self._fiftyone_sample_index_cache[cache_key]

        schema = self.get_fiftyone_schema(repo_id, revision=resolved_revision, token=token)
        if schema is None or not schema.usable_classification_ground_truth:
            with self._preview_lock:
                self._fiftyone_sample_index_cache[cache_key] = None
            return None

        def annotation_progress(payload: dict[str, Any]) -> None:
            if not callable(progress_callback):
                return
            data = dict(payload or {})
            severity = str(data.get("severity") or "info")
            message = str(data.get("message") or "Preparing FiftyOne annotations…")
            progress_callback(
                {
                    **data, "phase": "annotation_manifest", "completed": 0, "total": 0, "percent": -1,
                    "progress_known": False, "progress_basis": "annotation_manifest", "severity": severity,
                    "message": "Preparing native FiftyOne sample annotations: " + message,
                }
            )

        snapshot_root = self.snapshot_download_files(
            repo_id, [FIFTYONE_METADATA_FILENAME, FIFTYONE_SAMPLES_FILENAME], revision=resolved_revision,
            token=token, max_workers=2, quiet=True, cancel_token=cancel_token,
            progress_callback=annotation_progress,
        )
        samples_path = Path(snapshot_root) / FIFTYONE_SAMPLES_FILENAME
        try:
            byte_count = int(samples_path.stat().st_size)
        except Exception:
            byte_count = 0
        if byte_count > FIFTYONE_SAMPLES_MAX_BYTES:
            raise RuntimeError(
                "Native FiftyOne samples.json is too large for the bounded "
                f"classification metadata adapter ({byte_count:,} bytes > "
                f"{FIFTYONE_SAMPLES_MAX_BYTES:,} bytes)."
            )
        payload = json.loads(samples_path.read_text(encoding="utf-8"))
        index = build_classification_sample_index(payload, schema)
        with self._preview_lock:
            self._fiftyone_sample_index_cache[cache_key] = index
        return index

    # ------------------------------------------------------------------
    # Hub file listing
    # ------------------------------------------------------------------

    def list_repository_files(
        self, repo_id: str, *, token: str | None = None, revision: str | None = None,
        path_in_repo: str | None = None, max_files: int = 5000,
    ) -> list[dict[str, Any]]:
        """List a bounded pinned repository inventory and file sizes without downloading content."""
        repo_id = str(repo_id or "").strip()
        if not repo_id:
            raise ValueError("repo_id is required.")
        root = str(path_in_repo or "").strip().strip("/")
        limit = max(0, int(max_files or 0))
        cache_key = (repo_id, str(revision or "default"), root)
        with self._preview_lock:
            cached = self._repository_file_cache.get(cache_key)
        if cached is not None:
            rows, complete = cached
            if complete or not limit or len(rows) >= limit:
                return [dict(row) for row in (rows[:limit] if limit else rows)]

        from huggingface_hub import HfApi

        resolved_token = token or self.token
        api = HfApi(token=resolved_token)
        kwargs: dict[str, Any] = {
            "repo_id": repo_id, "repo_type": "dataset", "recursive": True, "revision": revision,
            "token": resolved_token,
        }
        if root:
            kwargs["path_in_repo"] = root
        try:
            tree_iter = api.list_repo_tree(**kwargs)
        except TypeError:
            kwargs.pop("token", None)
            tree_iter = api.list_repo_tree(**kwargs)

        rows: list[dict[str, Any]] = []
        exhausted = True
        for item in tree_iter:
            class_name = item.__class__.__name__.casefold()
            if "folder" in class_name or "directory" in class_name:
                continue
            path = str(getattr(item, "path", None) or getattr(item, "rfilename", None) or "").strip()
            if not path:
                continue
            if root and path != root and not path.startswith(root + "/"):
                path = f"{root}/{path.lstrip('/')}"
            raw_size = getattr(item, "size", None)
            try:
                size = int(raw_size) if raw_size is not None else None
            except Exception:
                size = None
            rows.append({"path": path, "size": size})
            if limit and len(rows) >= limit:
                exhausted = False
                break

        with self._preview_lock:
            self._repository_file_cache[cache_key] = ([dict(row) for row in rows], exhausted)
        return [dict(row) for row in rows]

    def _probe_structured_annotation_candidates(
        self, details: HFDatasetDetails, candidates: list[dict[str, Any]], *, token: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Probe small annotation JSON candidates under both advertised-size and actual-size limits."""
        eligible: list[tuple[int, dict[str, Any]]] = []
        for row in candidates:
            path = str(row.get("path") or "").strip()
            raw_size = row.get("size")
            try:
                size = int(raw_size) if raw_size is not None else -1
            except Exception:
                size = -1
            if not path or size <= 0 or size > STRUCTURED_ANNOTATION_PROBE_MAX_BYTES:
                continue
            eligible.append((size, dict(row)))

        probes: dict[str, dict[str, Any]] = {}
        for _size, row in sorted(eligible, key=lambda item: (item[0], str(item[1].get("path") or "")))[
            :STRUCTURED_ANNOTATION_PROBE_MAX_FILES
        ]:
            path = str(row.get("path") or "").strip()
            try:
                local_path = Path(
                    self.download_file(
                        details.repo_id, path, token=token, revision=details.revision or None, quiet=True,
                    )
                ).expanduser()
                actual_size = int(local_path.stat().st_size)
                if actual_size <= 0 or actual_size > STRUCTURED_ANNOTATION_PROBE_MAX_BYTES:
                    continue
                payload = json.loads(local_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue
            videos = payload.get("videos")
            annotations_present = "annotations" in payload
            annotations = payload.get("annotations")
            categories = payload.get("categories")
            video_sample = next(
                (item for item in videos if isinstance(item, dict)), None,
            ) if isinstance(videos, list) else None
            annotation_sample = next(
                (item for item in annotations if isinstance(item, dict)), None,
            ) if isinstance(annotations, list) else None
            first_file_names = []
            if isinstance(video_sample, dict):
                first_file_names = [
                    str(value) for value in (video_sample.get("file_names") or [])[:3] if str(value).strip()
                ]

            probes[path] = {
                "path": path, "size": actual_size, "root_fields": [str(key) for key in payload.keys()],
                "videos_is_array": isinstance(videos, list), "annotations_present": annotations_present,
                "annotations_is_array": isinstance(annotations, list),
                "annotations_is_null": annotations_present and annotations is None,
                "categories_is_array": isinstance(categories, list),
                "video_sampled": isinstance(video_sample, dict), "video_fields": (
                    [str(key) for key in video_sample.keys()] if isinstance(video_sample, dict) else []
                ), "annotation_sampled": isinstance(annotation_sample, dict), "annotation_fields": (
                    [str(key) for key in annotation_sample.keys()] if isinstance(annotation_sample, dict)
                    else []
                ), "first_file_names": first_file_names,
            }
        return probes

    @staticmethod
    def _huggingface_dataset_resolve_root(repo_id: str, revision: str, path_in_repo: str) -> str:
        repo = urllib.parse.quote(str(repo_id or "").strip(), safe="/")
        rev = urllib.parse.quote(str(revision or "main").strip() or "main", safe="")
        root = urllib.parse.quote(str(path_in_repo or "").strip().strip("/"), safe="/")
        base = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/"
        return base + (root.rstrip("/") + "/" if root else "")

    @staticmethod
    def _repository_media_root_from_image_path(
        path: str, *, split: str, sample_names: list[str] | tuple[str, ...] = (),
    ) -> str:
        """Infer a frame root from an annotation suffix or split path without opening media."""
        clean_path = str(path or "").strip().strip("/")
        canonical_split = SPLIT_ALIASES.get(str(split or "").casefold(), str(split or "").casefold())
        if not clean_path or not canonical_split:
            return ""

        for raw_sample in sample_names:
            sample = str(raw_sample or "").strip().lstrip("/")
            if not sample:
                continue
            candidates = [sample]
            sample_parts = sample.split("/")
            if len(sample_parts) > 1:
                first = SPLIT_ALIASES.get(sample_parts[0].casefold(), "")
                if first == canonical_split:
                    candidates.append("/".join(sample_parts[1:]))
            for candidate in candidates:
                if clean_path == candidate:
                    return ""
                suffix = "/" + candidate
                if clean_path.endswith(suffix):
                    return clean_path[: -len(suffix)].rstrip("/")

        parts = clean_path.split("/")
        for index, part in enumerate(parts[:-1]):
            normalised = re.sub(r"[^a-z0-9]+", "", part.casefold())
            if SPLIT_ALIASES.get(normalised, "") == canonical_split:
                return "/".join(parts[: index + 1])
        return ""

    def _infer_repository_frame_media_roots(
        self, details: HFDatasetDetails, profile: HFStructuredAnnotationProfile, *,
        token: str | None = None, probes: dict[str, dict[str, Any]] | None = None,
        splits: list[str] | tuple[str, ...] | None = None,
    ) -> HFStructuredAnnotationProfile:
        """Find verified image-bearing split roots from bounded Hub metadata; prefer explicit card roots."""
        existing = {
            str(key): str(value) for key, value in dict(profile.frame_media_roots_by_split or {}).items()
            if str(key).strip() and str(value).strip()
        }
        target_splits = tuple(
            str(value) for value in (splits if splits is not None else profile.splits) if str(value).strip()
        )
        missing = [split for split in target_splits if split not in existing]
        if not missing:
            return profile

        probe_samples: dict[str, list[str]] = {}
        for path, payload in dict(probes or {}).items():
            split = SPLIT_ALIASES.get(
                self._infer_split_from_path(str(path or "")), self._infer_split_from_path(str(path or "")),
            )
            names = [
                str(value).strip().lstrip("/") for value in (payload.get("first_file_names") or [])
                if str(value).strip()
            ]
            if split and names:
                probe_samples.setdefault(split, []).extend(names)

        aliases = {
            "train": ("train", "training"), "validation": ("validation", "valid", "val", "dev"),
            "test": ("test", "testing", "eval", "evaluation"),
        }
        roots = dict(existing)
        inferred_splits: list[str] = []
        for split in missing:
            split_aliases = aliases.get(str(split), (str(split),))
            samples = probe_samples.get(str(split), [])
            chosen = ""
            for candidate_root in split_aliases:
                try:
                    rows = self.list_repository_files(
                        details.repo_id, token=token, revision=details.revision or None,
                        path_in_repo=candidate_root, max_files=STRUCTURED_MEDIA_DIRECTORY_SCAN_MAX_FILES,
                    )
                except Exception:
                    continue
                image_paths = [
                    str(row.get("path") or "").strip() for row in rows
                    if Path(str(row.get("path") or "")).suffix.casefold() in IMAGE_EXTENSIONS
                ]
                if not image_paths:
                    continue
                if samples:
                    relative = [
                        path[len(candidate_root) + 1 :] if path.startswith(candidate_root + "/") else path
                        for path in image_paths
                    ]
                    normalised_samples = [
                        sample[len(candidate_root) + 1 :] if sample.startswith(candidate_root + "/")
                        else sample for sample in samples
                    ]
                    if not any(
                        sample in relative or any(value.endswith("/" + sample) for value in relative)
                        for sample in normalised_samples
                    ):
                        continue
                chosen = candidate_root
                break

            # Some repositories place split media below a leading prefix or use a
            # layout that is visible to the generic image inventory but not at a
            # literal top-level ``valid/`` / ``validation/`` directory. Reuse the
            # existing image-file inventory as a second, still metadata-only source
            # of truth and derive the common repository prefix from concrete paths.
            if not chosen:
                try:
                    image_files = self.list_image_files(
                        details.repo_id, token=token, split=str(split), max_files=24,
                        revision=details.revision or None,
                    )
                except Exception:
                    image_files = []
                root_counts: Counter[str] = Counter()
                for image_file in image_files:
                    candidate = self._repository_media_root_from_image_path(
                        str(getattr(image_file, "path", "") or ""), split=str(split), sample_names=samples,
                    )
                    if candidate:
                        root_counts[candidate] += 1
                if root_counts:
                    chosen = sorted(
                        root_counts.items(), key=lambda item: (-int(item[1]), len(item[0]), item[0]),
                    )[0][0]

            if not chosen:
                continue
            roots[str(split)] = self._huggingface_dataset_resolve_root(
                details.repo_id, details.revision or "main", chosen,
            )
            inferred_splits.append(str(split))

        if not inferred_splits:
            return profile

        evidence = list(profile.evidence or ())
        evidence.append(
            "bounded Hugging Face repository-tree inspection resolved in-repository frame media for "
            + ", ".join(inferred_splits)
        )
        return replace(profile, frame_media_roots_by_split=roots, evidence=tuple(evidence))

    def resolve_structured_frame_media_roots(
        self, repo_id: str, *, profile: dict[str, Any] | HFStructuredAnnotationProfile,
        splits: list[str] | tuple[str, ...] | None = None, revision: str | None = None,
        token: str | None = None,
    ) -> dict[str, str]:
        """Resolve frame roots from pinned metadata without materialising images or assuming repository
        identity.
        """
        repo_id = str(repo_id or "").strip()
        if not repo_id:
            raise ValueError("repo_id is required.")
        profile_obj = (
            profile if isinstance(profile, HFStructuredAnnotationProfile)
            else HFStructuredAnnotationProfile.from_dict(dict(profile or {}))
        )
        resolved_revision = str(
            revision or self.resolve_dataset_revision(repo_id, revision=revision, token=token) or "main"
        )
        details = HFDatasetDetails(
            repo_id=repo_id, revision=resolved_revision, inspection_source="repository_media_resolution",
        )
        resolved = self._infer_repository_frame_media_roots(
            details, profile_obj, token=token, probes=None, splits=splits,
        )
        wanted = {
            str(value) for value in (splits if splits is not None else resolved.splits)
            if str(value).strip()
        }
        return {
            str(key): str(value) for key, value in dict(resolved.frame_media_roots_by_split or {}).items()
            if str(key) in wanted and str(value).strip()
        }

    @staticmethod
    def _infer_split_from_path(path: str) -> str:
        text = f"/{str(path or '').casefold()}/"
        for raw, canonical in SPLIT_ALIASES.items():
            if re.search(rf"(?:^|[/_.-]){re.escape(raw)}(?:$|[/_.-])", text):
                return canonical
        return ""

    def enrich_structured_annotation_details(
        self, details: HFDatasetDetails, *, token: str | None = None,
    ) -> dict[str, Any]:
        """Inspect structured-video formats with bounded small-JSON probes and no repository-specific
        special cases.
        """
        declared = self._flatten_declared_data_files(details)
        detection_text = self._structured_detection_text(details)
        folded_detection = detection_text.casefold()
        declared_json = any(
            str(path).casefold().endswith(".json") for paths in declared.values() for path in paths
        )
        broad_video_annotation_hint = bool(
            re.search(r"\bytvis\b", folded_detection) or "masklet" in folded_detection
            or "video_np_pairs" in folded_detection or (
                "video" in folded_detection and (
                    "segmentation" in folded_detection or "annotation" in folded_detection
                    or "tracking" in folded_detection
                )
            )
        )

        # Do not require the dataset card / Dataset Viewer schema to advertise
        # video semantics before checking the two conventional annotation roots.
        # Repositories can expose RGB/RGB-D frames as an Image dataset while
        # retaining their canonical video-instance JSON under ``annotations/``.
        # These directory-scoped Hub tree calls are bounded and metadata-only.
        # A potentially expensive repository-root scan is still reserved for
        # repositories which already provide stronger structured-media evidence.
        parents: list[str] = []
        for paths in declared.values():
            for raw_path in paths:
                if not str(raw_path).casefold().endswith(".json"):
                    continue
                parent = str(Path(str(raw_path)).parent).replace("\\", "/")
                if parent == ".":
                    parent = ""
                if parent not in parents:
                    parents.append(parent)

        scan_roots: list[str] = []
        for root in [*parents, "annotations", "annotation"]:
            clean = str(root or "").strip().strip("/")
            if clean not in scan_roots:
                scan_roots.append(clean)

        inventory: list[dict[str, Any]] = []
        seen: set[str] = set()

        def scan_root(root: str, *, limit: int) -> None:
            try:
                rows = self.list_repository_files(
                    details.repo_id, token=token, revision=details.revision or None,
                    path_in_repo=root or None, max_files=limit,
                )
            except Exception:
                return
            for row in rows:
                path = str(row.get("path") or "")
                if not path or path in seen:
                    continue
                seen.add(path)
                inventory.append(dict(row))

        for root in scan_roots[:10]:
            scan_root(
                root, limit=(
                    STRUCTURED_ANNOTATION_ROOT_SCAN_MAX_FILES if not root else (
                        5000 if root in parents else STRUCTURED_ANNOTATION_DIRECTORY_SCAN_MAX_FILES
                    )
                ),
            )

        # A repository-root tree walk can enumerate thousands of unrelated media
        # files. Only fall back to it when stronger card/declaration evidence exists
        # *and* the cheap annotation-directory probes found no annotation-like JSON.
        initial_candidates = structured_annotation_candidates(
            repository_files=inventory, declared_data_files_by_split=declared,
        )
        if (declared_json or broad_video_annotation_hint) and not initial_candidates and "" not in scan_roots:
            scan_root("", limit=STRUCTURED_ANNOTATION_ROOT_SCAN_MAX_FILES)

        if not inventory and not declared_json:
            return {}

        probes: dict[str, dict[str, Any]] = {}

        # First preserve the zero-download path for repositories whose card
        # already provides enough explicit format evidence. Card-declared JSON
        # paths remain usable here even when a gated repository tree could not be
        # enumerated: ``structured_annotation_candidates`` synthesises only the
        # declared path metadata, while the detector still requires independent
        # YTVIS/video-segmentation evidence before accepting the profile.
        profile = detect_structured_annotation_profile(
            card_text=detection_text, repository_files=inventory, declared_data_files_by_split=declared,
        )

        if profile is None:
            candidates = structured_annotation_candidates(
                repository_files=inventory, declared_data_files_by_split=declared,
            )
            probes = self._probe_structured_annotation_candidates(details, candidates, token=token)
            if probes:
                profile = detect_structured_annotation_profile(
                    card_text=detection_text, repository_files=inventory,
                    declared_data_files_by_split=declared, probed_json_by_path=probes,
                )

        if profile is None:
            return {}

        profile = self._infer_repository_frame_media_roots(details, profile, token=token, probes=probes)

        details.structured_annotation_profile = profile.to_dict()
        details.repository_format_hint = details.repository_format_hint or profile.format_id
        details.annotation_source_hint = details.annotation_source_hint or profile.format_label
        if not details.inferred_task_hint and profile.task_hints:
            details.inferred_task_hint = profile.task_hints[0]
        details.image_column_candidates = []
        details.label_column_candidates = []

        if not details.splits_by_config:
            details.splits_by_config["default"] = list(profile.splits or ("train",))
            details.configs = ["default"]
        elif profile.splits:
            # Preserve declared config identity, filling only missing split lists.
            for config_name in list(details.configs or details.splits_by_config or ["default"]):
                if not details.splits_by_config.get(config_name):
                    details.splits_by_config[config_name] = list(profile.splits)
        return dict(details.structured_annotation_profile)

    def get_structured_annotation_details(
        self, repo_id: str, *, token: str | None = None, revision: str | None = None,
    ) -> HFDatasetDetails:
        """Inspect structured annotation metadata without invoking Dataset Viewer/builder."""
        repo_id = str(repo_id or "").strip()
        if not repo_id:
            raise ValueError("repo_id is required.")
        info = self.dataset_info_lightweight(repo_id, token=token)
        details = HFDatasetDetails(
            repo_id=repo_id, revision=str(revision or getattr(info, "sha", "") or ""),
            inspection_source="dataset_card+repository_inventory",
        )
        self._apply_dataset_card_metadata(details, info, repo_id=repo_id, token=token)
        self.enrich_structured_annotation_details(details, token=token)
        return details

    def cache_structured_annotations(
        self, repo_id: str, *, splits: list[str] | tuple[str, ...] | None = None, variant: str = "auto",
        revision: str | None = None, token: str | None = None, max_workers: int = 4,
        cancel_token: Any = None, progress_callback: Any = None, profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Cache selected structured annotations, reusing complete local data first."""
        repo_id = str(repo_id or "").strip()
        if not repo_id:
            raise ValueError("repo_id is required.")

        profile_obj: HFStructuredAnnotationProfile | None = None
        resolved_revision = str(revision or "").strip()
        if profile:
            profile_obj = HFStructuredAnnotationProfile.from_dict(profile)
        else:
            cached_details = self.get_dataset_details_cached(repo_id, mode="auto", token=token)
            if cached_details is not None and cached_details.structured_annotation_profile:
                profile_obj = HFStructuredAnnotationProfile.from_dict(
                    cached_details.structured_annotation_profile
                )
                resolved_revision = resolved_revision or str(cached_details.revision or "")

        if profile_obj is not None:
            selected = profile_obj.files_for(splits=splits, variant=variant)
            if not selected:
                raise ValueError("No structured annotation files match the selected splits/variant.")
            paths = [item.path for item in selected]
            local_snapshot = self._local_snapshot_path(
                repo_id, revision=resolved_revision or None, token=token,
            )
            if local_snapshot is not None:
                missing = self._snapshot_missing_paths(local_snapshot, paths)
                if not missing:
                    if not resolved_revision and self._looks_like_commit_revision(local_snapshot.name):
                        resolved_revision = local_snapshot.name
                    sizes = {item.path: item.size for item in selected}
                    cached_bytes = 0
                    known = 0
                    for path_text in paths:
                        size = sizes.get(path_text)
                        if size is None:
                            try:
                                size = int((local_snapshot / path_text).stat().st_size)
                            except OSError:
                                size = None
                        if size is not None:
                            cached_bytes += int(size)
                            known += 1
                    plan = HFSnapshotPlan(
                        repo_id=repo_id, revision=resolved_revision, total_files=len(paths),
                        cached_files=len(paths), download_files=0, total_bytes=cached_bytes,
                        cached_bytes=cached_bytes, download_bytes=0, known_size_files=known,
                        download_known_size_files=0, cache_coverage_known=True, dry_run_supported=False,
                        planning_mode="local_snapshot_complete",
                    )
                    if callable(progress_callback):
                        progress_callback(
                            {
                                "phase": "snapshot_cached", "completed": len(paths), "total": len(paths),
                                "percent": 100, "progress_known": True, "progress_basis": "files_cached",
                                "message": "Selected structured annotations are already complete in the local Hugging Face cache.",
                                "current_file": "", "severity": "info",
                                "selected_total_bytes": cached_bytes, "expected_download_bytes": 0,
                                "cached_bytes": cached_bytes,
                            }
                        )
                    return {
                        "repo_id": repo_id, "revision": resolved_revision,
                        "format_id": profile_obj.format_id, "format_label": profile_obj.format_label,
                        "variant": str(variant or "auto"), "splits": list(splits or profile_obj.splits),
                        "annotation_files": [item.to_dict() for item in selected],
                        "snapshot_root": str(local_snapshot),
                        "local_files": [str(local_snapshot / path) for path in paths],
                        "snapshot_plan": plan.to_dict(), "registered": False,
                        "next_contract": "core.video + vision.segmentation",
                    }

        # Local state was not sufficient, so remote access/metadata is now
        # genuinely required.  Only at this point do we verify gated access.
        access = self.dataset_access_status(repo_id, token=token)
        if access.gated and not access.accessible:
            raise PermissionError(access.message or "Hugging Face dataset access is required.")

        resolved_revision = self.resolve_dataset_revision(
            repo_id, revision=resolved_revision or None, token=token,
        )
        if profile_obj is None:
            details = self.get_structured_annotation_details(
                repo_id, token=token, revision=resolved_revision,
            )
            if not details.structured_annotation_profile:
                raise ValueError(
                    "No supported structured annotation profile was detected for this repository."
                )
            profile_obj = HFStructuredAnnotationProfile.from_dict(details.structured_annotation_profile)

        selected = profile_obj.files_for(splits=splits, variant=variant)
        if not selected:
            raise ValueError("No structured annotation files match the selected splits/variant.")
        paths = [item.path for item in selected]
        sizes = {item.path: item.size for item in selected}
        plan = self.snapshot_download_plan(
            repo_id, paths, revision=resolved_revision, token=token, max_workers=max_workers,
            file_sizes=sizes,
        )
        if callable(progress_callback):
            selected_total = plan.total_bytes if plan.total_size_complete else 0
            expected_download = plan.download_bytes if plan.download_size_complete else 0
            progress_callback(
                {
                    "phase": "annotation_plan", "completed": plan.cached_files, "total": plan.total_files,
                    "percent": (100 if plan.fully_cached else -1),
                    "progress_known": bool(plan.fully_cached),
                    "progress_basis": "files_cached" if plan.fully_cached else "planning", "message": (
                        f"Structured annotation plan: {plan.total_files:,} file(s), "
                        f"{plan.cached_files:,} cached, {plan.download_files:,} remaining."
                    ), "selected_total_bytes": selected_total, "expected_download_bytes": expected_download,
                    "cached_bytes": plan.cached_bytes,
                }
            )

        snapshot_root = self.snapshot_download_files(
            repo_id, paths, revision=resolved_revision, token=token, max_workers=max_workers, quiet=True,
            cancel_token=cancel_token, progress_callback=progress_callback, snapshot_plan=plan,
        )
        root = Path(snapshot_root)
        local_files = [str(root / path) for path in paths]
        missing = [path for path in local_files if not Path(path).is_file()]
        if missing:
            raise HFIncompleteSnapshotError(
                "Structured annotation snapshot completed without all requested files: "
                + ", ".join(missing[:3])
            )
        return {
            "repo_id": str(repo_id), "revision": resolved_revision, "format_id": profile_obj.format_id,
            "format_label": profile_obj.format_label, "variant": str(variant or "auto"),
            "splits": list(splits or profile_obj.splits),
            "annotation_files": [item.to_dict() for item in selected], "snapshot_root": str(snapshot_root),
            "local_files": local_files, "snapshot_plan": plan.to_dict(), "registered": False,
            "next_contract": "core.video + vision.segmentation",
        }

    def list_image_files(
        self, repo_id: str, *, token: str | None = None, split: str | None = None, max_files: int = 0,
        revision: str | None = None,
    ) -> list[HFImageFile]:
        """List image files while reusing the listing produced by inspection."""
        repo_id = str(repo_id or "").strip()
        max_files = max(0, int(max_files or 0))
        wanted_split = self._normalise_split(split) if split else None

        cache_key = f"{repo_id}@{revision or 'default'}"
        with self._preview_lock:
            cached_entry = self._image_file_cache.get(cache_key)
        if cached_entry is not None:
            cached_files, complete = cached_entry
            filtered = [item for item in cached_files if wanted_split is None or item.split == wanted_split]
            if complete or (max_files and len(filtered) >= max_files):
                return list(filtered[:max_files] if max_files else filtered)

        from huggingface_hub import HfApi
        api = HfApi(token=token or self.token)
        try:
            tree_iter = api.list_repo_tree(
                repo_id=repo_id, repo_type="dataset", recursive=True, revision=revision,
                token=token or self.token,
            )
        except TypeError:
            tree_iter = api.list_repo_tree(
                repo_id=repo_id, repo_type="dataset", recursive=True, revision=revision,
            )

        files: list[HFImageFile] = []
        exhausted = True
        matching_count = 0
        for item in tree_iter:
            path = getattr(item, "path", None) or getattr(item, "rfilename", None) or ""
            if not path or not self._is_image_path(path):
                continue
            detected_split = self.infer_split_from_path(path)
            label_info = self.infer_label_details_from_path(path, detected_split)
            image_file = HFImageFile(
                path=str(path), split=detected_split, label=label_info.label,
                size=getattr(item, "size", None), label_source=label_info.source,
                label_secondary=label_info.secondary_label,
                label_secondary_confidence=label_info.secondary_confidence,
                label_prefix_value=label_info.prefix_value, label_parser=label_info.parser,
                label_ambiguous=label_info.ambiguous,
            )
            files.append(image_file)
            if wanted_split is None or detected_split == wanted_split:
                matching_count += 1
            if max_files and matching_count >= max_files:
                exhausted = False
                break

        with self._preview_lock:
            previous, previous_complete = self._image_file_cache.get(cache_key, ([], False))
            by_path = {item.path: item for item in previous}
            by_path.update({item.path: item for item in files})
            self._image_file_cache[cache_key] = (
                list(by_path.values()), bool(previous_complete or exhausted),
            )

        filtered = [item for item in files if wanted_split is None or item.split == wanted_split]
        return filtered[:max_files] if max_files else filtered

    def enrich_details_from_repository_inventory(
        self, details: HFDatasetDetails, *, token: str | None = None,
    ) -> HFDatasetDetails:
        """Fill inspection metadata gaps using cached pinned Hub file listings; sample oversized
        repositories.
        """
        if details is None:
            return details

        config = str(details.configs[0] if details.configs else "default")
        features = dict(
            details.features_by_config.get(config) or details.features_by_config.get("default") or {}
        )
        image_candidates = list(details.image_column_candidates or [])
        if not image_candidates:
            image_candidates, _ = self._feature_candidates_from_features(features)
        if not image_candidates:
            return details

        repo_hint = int(details.total_repo_files_hint or 0)
        full_inventory = not repo_hint or repo_hint <= REPOSITORY_INSPECTION_MAX_FILES
        max_files = 0 if full_inventory else REPOSITORY_INSPECTION_SAMPLE_FILES

        try:
            files = self.list_image_files(
                details.repo_id, token=token, max_files=max_files, revision=details.revision or None,
            )
        except Exception as exc:
            details.warnings.append("Repository inventory unavailable. " + self._short_exception(exc))
            return details

        if not files:
            return details

        details.repository_inventory_complete = bool(full_inventory)
        by_split: dict[str, list[HFImageFile]] = {}
        for item in files:
            by_split.setdefault(str(item.split or "train"), []).append(item)

        all_labels = [str(item.label).strip() for item in files if str(item.label or "").strip()]
        distinct_labels = sorted(set(all_labels))
        details.repository_label_examples = distinct_labels[:12]

        stats: dict[str, dict[str, Any]] = {}
        for split, split_files in by_split.items():
            known_sizes = [int(item.size) for item in split_files if item.size is not None]
            labelled = sum(1 for item in split_files if str(item.label or "").strip())
            ambiguous = sum(1 for item in split_files if bool(item.label_ambiguous))
            parser_counts = Counter(str(item.label_parser or "unknown") for item in split_files)
            stats[split] = {
                "files": len(split_files), "bytes": sum(known_sizes), "known_size_files": len(known_sizes),
                "size_complete": bool(full_inventory and len(known_sizes) == len(split_files)),
                "labelled_files": labelled, "label_coverage": labelled / max(len(split_files), 1),
                "label_count": len({str(item.label).strip() for item in split_files if str(item.label or "").strip()}),
                "ambiguous_filename_pairs": ambiguous, "label_parser_counts": dict(parser_counts),
            }
        details.repository_stats_by_split = stats

        if full_inventory:
            total_images = len(files)
            known_sizes = [int(item.size) for item in files if item.size is not None]
            if details.total_rows_hint is None:
                details.total_rows_hint = total_images
            if details.total_bytes_hint is None and len(known_sizes) == total_images:
                details.total_bytes_hint = sum(known_sizes)
            details.scanned_file_count = total_images

        label_coverage = len(all_labels) / max(len(files), 1)
        if (
            details.ground_truth_field_hint
            or (not details.label_column_candidates and label_coverage >= 0.80)
        ):
            # Native FiftyOne Classification metadata is stronger than filename
            # inference. Otherwise high path-label coverage is still sufficient
            # to identify an image-classification repository conservatively.
            details.inferred_task_hint = "image-classification"

        if not full_inventory:
            details.warnings.append(
                f"Repository is large; inspection sampled {len(files):,} image file(s) for path-label inference."
            )
        return details

    def repository_image_manifest_viability(
        self, repo_id: str, *, split: str, revision: str | None = None, token: str | None = None,
        sample_limit: int = 32,
    ) -> dict[str, Any]:
        """Check cheaply whether raw repository paths can preserve labels without a builder."""
        files = self.list_image_files(
            repo_id, token=token, split=split, max_files=max(3, int(sample_limit or 32)), revision=revision,
        )
        labelled = sum(1 for item in files if str(item.label or "").strip())
        coverage = labelled / max(len(files), 1)
        fiftyone_schema = self.get_fiftyone_schema(repo_id, revision=revision, token=token)
        exact_fiftyone_labels = bool(
            fiftyone_schema is not None and fiftyone_schema.usable_classification_ground_truth
        )
        return {
            "sampled_files": len(files), "labelled_files": labelled, "label_coverage": coverage,
            "fiftyone_ground_truth": (fiftyone_schema.ground_truth_field if fiftyone_schema else ""),
            "label_source": ("fiftyone_samples" if exact_fiftyone_labels else "repository_path"),
            "usable": bool(exact_fiftyone_labels or (len(files) >= 3 and coverage >= 0.80)),
        }

    def preview_image_grid_html(
        self, *, repo_id: str, config_name: str | None, split: str, image_column: str | None = None,
        label_column: str | None = None, id_column: str | None = None, limit: int = 12,
        token: str | None = None, trust_remote_code: bool = False, thumb_size: int = 180,
    ) -> str:
        del config_name, image_column, label_column, id_column, trust_remote_code
        result = self.preview_image_items(
            repo_id=repo_id, split=split, limit=limit, token=token, thumb_size=thumb_size,
        )
        return self._preview_result_to_html(result, thumb_size=thumb_size)

    def preview_image_items(
        self, *, repo_id: str, split: str, limit: int = 12, token: str | None = None, thumb_size: int = 180,
        revision: str | None = None,
    ) -> dict[str, Any]:
        """Resolve revision-scoped file previews and consult local caches first."""
        repo_id = str(repo_id or "").strip()
        split = str(split or "train")
        limit = max(1, int(limit or 12))
        thumb_size = max(32, int(thumb_size or 180))
        key = ("files", repo_id, split, str(revision or ""), self._details_cache_auth_scope(token),
               limit, thumb_size)
        cached = self._preview_cache_get(key)
        if cached is not None:
            cached["cache_source"] = "preview memory cache"
            return cached

        files = self.list_image_files(
            repo_id, token=token or self.token, split=split, max_files=limit, revision=revision,
        )
        if not files:
            result = {
                "items": [], "count": 0, "cache_source": "none",
                "note": "No image files were found for this split.",
            }
            self._preview_cache_put(key, result)
            return result

        resolved = self.download_files_parallel(
            repo_id, [item.path for item in files], token=token or self.token,
            max_workers=min(8, len(files)), revision=revision,
        )

        def build_item(index_and_file: tuple[int, HFImageFile]) -> dict[str, Any]:
            index, file = index_and_file
            local_path = resolved.get(file.path, "")
            try:
                data_uri = self.local_image_to_data_uri(local_path, max_size=thumb_size)
            except Exception:
                data_uri = ""
            return {
                "record_id": f"{file.split}:{index}", "label": str(file.label or "no inferred label"),
                "image_column": "Hub image file", "label_column": "folder/path inference",
                "path": file.path, "source": "local Hugging Face cache or resolved Hub file",
                "data_uri": data_uri,
            }

        workers = min(8, max(1, len(files)))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                items = list(executor.map(build_item, enumerate(files)))
        else:
            items = [build_item(item) for item in enumerate(files)]

        result = {
            "items": items, "count": len(items), "cache_source": "local-first Hugging Face file cache",
            "note": f"Resolved {len(items)} preview image(s) concurrently; cached files were reused first.",
        }
        self._preview_cache_put(key, result)
        return result

    def download_files_parallel(
        self, repo_id: str, file_paths: list[str], *, token: str | None = None, max_workers: int = 16,
        progress_callback: Any = None, revision: str | None = None,
    ) -> dict[str, str]:
        """Fetch files concurrently, reuse local caches first, and report file-count progress. Returns a
        mapping from Hub-relative path to local cache path.
        """

        clean_paths = [str(path) for path in file_paths if str(path).strip()]
        total = len(clean_paths)

        if total == 0:
            return {}

        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
        except Exception:
            pass

        def emit(
            *, phase: str, completed: int, total: int, percent: int, message: str, current_file: str = "",
        ) -> None:
            if callable(progress_callback):
                try:
                    progress_callback(
                        {
                            "phase": phase, "completed": int(completed), "total": int(total),
                            "percent": max(0, min(100, int(percent))), "message": message,
                            "current_file": current_file,
                        }
                    )
                except Exception:
                    pass

        emit(
            phase="download", completed=0, total=total, percent=5,
            message=f"Starting parallel download of {total} file(s)…",
        )

        def worker(path: str) -> tuple[str, str]:
            local_path = self.download_file(repo_id, path, token=token or self.token, revision=revision)
            return path, local_path

        results: dict[str, str] = {}
        completed = 0
        max_workers = max(1, int(max_workers or 16))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, path): path for path in clean_paths}

            for future in as_completed(futures):
                path = futures[future]
                hub_path, local_path = future.result()
                results[hub_path] = local_path
                completed += 1

                # Reserve 5-85% for file download.
                percent = 5 + int((completed / total) * 80)

                emit(
                    phase="download", completed=completed, total=total, percent=percent,
                    message=f"Downloaded {completed}/{total} file(s)…", current_file=path,
                )

        emit(
            phase="download", completed=total, total=total, percent=85,
            message=f"Downloaded {total}/{total} file(s).",
        )

        return results

    def download_file(
        self, repo_id: str, filename: str, *, token: str | None = None, revision: str | None = None,
        quiet: bool = True,
    ) -> str:
        from huggingface_hub import hf_hub_download

        # Probe all known caches without network access first.  This includes
        # the standard Hugging Face cache and AstronomicAL's legacy locations.
        for cache_root in self._hub_cache_roots():
            kwargs = self._hub_download_kwargs(
                repo_id=repo_id, filename=filename, token=token, cache_root=cache_root,
                local_files_only=True, revision=revision,
            )
            try:
                return str(self._call_with_compatible_kwargs(hf_hub_download, kwargs))
            except Exception:
                continue

        kwargs = self._hub_download_kwargs(
            repo_id=repo_id, filename=filename, token=token, cache_root=self.configured_cache_dir,
            local_files_only=False, revision=revision,
        )
        if quiet:
            kwargs["tqdm_class"] = SilentTqdm
        return str(self._call_with_compatible_kwargs(hf_hub_download, kwargs))

    # ------------------------------------------------------------------
    # Import: batch snapshot download
    # ------------------------------------------------------------------

    def effective_token(self, token: str | None = None) -> str | None:
        explicit = str(token or self.token or "").strip()
        if explicit:
            return explicit
        try:
            from huggingface_hub import get_token

            cached = get_token()
            if cached:
                return str(cached)
        except Exception:
            pass
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            value = str(os.environ.get(name) or "").strip()
            if value:
                return value
        return None

    def has_effective_token(self, token: str | None = None) -> bool:
        return bool(self.effective_token(token))

    def _snapshot_worker_count(self, requested: int, *, token: str | None = None) -> int:
        requested = max(int(requested or 1), 1)
        limit = (
            AUTHENTICATED_SNAPSHOT_MAX_WORKERS if self.has_effective_token(token)
            else ANONYMOUS_SNAPSHOT_MAX_WORKERS
        )
        return min(requested, limit)

    @staticmethod
    def _cancelled(cancel_token: Any = None) -> bool:
        try:
            return bool(cancel_token is not None and cancel_token.cancelled())
        except Exception:
            return False

    @contextmanager
    def _rate_limit_interceptor(self, *, repo_id: str, progress_callback: Any = None):
        logger = logging.getLogger("huggingface_hub")
        handler = _HFRateLimitInterceptor(repo_id=repo_id, progress_callback=progress_callback)
        logger.addHandler(handler)
        try:
            yield
        finally:
            try:
                logger.removeHandler(handler)
            except Exception:
                pass

    def _cooperative_rate_limit_wait(
        self, seconds: float, *, cancel_token: Any = None, progress_callback: Any = None, attempt: int = 1,
    ) -> None:
        remaining = max(float(seconds or 0.0), 0.0)
        deadline = time.monotonic() + remaining
        last_reported: int | None = None
        while True:
            if self._cancelled(cancel_token):
                raise HFTransferCancelled("Hugging Face import cancelled during rate-limit backoff.")
            remaining = max(deadline - time.monotonic(), 0.0)
            rounded = int(remaining + 0.999)
            if rounded != last_reported and callable(progress_callback):
                last_reported = rounded
                try:
                    progress_callback(
                        {
                            "phase": "rate_limited", "completed": 0, "total": 0, "percent": 10, "message": (
                                f"Hugging Face rate limit reached. Retrying in {rounded}s "
                                f"(retry {attempt}/{MAX_RATE_LIMIT_RETRIES}). "
                                "You can cancel the import now."
                            ), "current_file": "", "severity": "warning", "retry_after_seconds": rounded,
                        }
                    )
                except Exception:
                    pass
            if remaining <= 0:
                return
            time.sleep(min(0.25, remaining))

    def _best_local_snapshot_coverage(
        self, repo_id: str, file_paths: list[str], *, revision: str, token: str | None = None,
    ) -> tuple[set[str], bool]:
        """Find the largest locally verified snapshot file set without contacting the Hub."""
        from huggingface_hub import snapshot_download

        requested = list(dict.fromkeys(str(path) for path in file_paths if str(path).strip()))
        if not requested:
            return set(), True

        best: set[str] = set()
        resolved_any = False
        resolved_token = self.effective_token(token)
        for cache_root in self._hub_cache_roots():
            kwargs: dict[str, Any] = {
                "repo_id": repo_id, "repo_type": "dataset", "revision": revision,
                "allow_patterns": requested, "local_files_only": True, "token": resolved_token,
                "tqdm_class": SilentTqdm,
            }
            if cache_root is not None:
                kwargs["cache_dir"] = str(cache_root)
            try:
                snapshot_path = self._call_with_compatible_kwargs(snapshot_download, kwargs)
            except Exception:
                continue
            resolved_any = True
            root = Path(snapshot_path).expanduser()
            present: set[str] = set()
            for relative in requested:
                try:
                    if (root / relative).is_file():
                        present.add(relative)
                except OSError:
                    continue
            if len(present) > len(best):
                best = present
            if len(best) >= len(requested):
                break
        return best, resolved_any

    def snapshot_download_plan(
        self, repo_id: str, file_paths: list[str], *, revision: str | None = None, token: str | None = None,
        max_workers: int = 16, file_sizes: dict[str, int | None] | None = None,
    ) -> HFSnapshotPlan:
        """Estimate selected, cached and remaining bytes from pinned file sizes and verified local files;
        avoid large remote snapshot dry runs.
        """
        from huggingface_hub import snapshot_download

        clean_paths = list(dict.fromkeys(str(path) for path in file_paths if str(path).strip()))
        if not clean_paths:
            return HFSnapshotPlan(repo_id=str(repo_id), revision=str(revision or ""))

        resolved_revision = self.resolve_dataset_revision(repo_id, revision=revision, token=token)
        sizes = {str(key): value for key, value in dict(file_sizes or {}).items()}

        def _apply_tree_sizes(plan: HFSnapshotPlan, cached_paths: set[str] | None = None) -> HFSnapshotPlan:
            cached_paths = set(cached_paths or set())
            plan.total_files = len(clean_paths)
            plan.cached_files = len(cached_paths)
            plan.download_files = max(len(clean_paths) - len(cached_paths), 0)
            plan.total_bytes = 0
            plan.cached_bytes = 0
            plan.download_bytes = 0
            plan.known_size_files = 0
            plan.download_known_size_files = 0
            for path in clean_paths:
                raw_size = sizes.get(path)
                try:
                    size = int(raw_size) if raw_size is not None else None
                except Exception:
                    size = None
                if size is None or size < 0:
                    continue
                plan.known_size_files += 1
                plan.total_bytes += size
                if path in cached_paths:
                    plan.cached_bytes += size
                else:
                    plan.download_known_size_files += 1
                    plan.download_bytes += size
            return plan

        use_remote_dry_run = (
            len(clean_paths) <= SNAPSHOT_DRY_RUN_MAX_FILES and self.has_effective_token(token)
        )
        if not use_remote_dry_run:
            mode = (
                "large_snapshot_repository_tree" if len(clean_paths) > SNAPSHOT_DRY_RUN_MAX_FILES
                else "anonymous_repository_tree"
            )
            cached_paths, local_snapshot_resolved = self._best_local_snapshot_coverage(
                repo_id, clean_paths, revision=resolved_revision, token=token
            )
            cache_state = self.cache_status(repo_id, revision=resolved_revision)
            cache_coverage_known = bool(local_snapshot_resolved or not cache_state.hub_cached)
            return _apply_tree_sizes(
                HFSnapshotPlan(
                    repo_id=str(repo_id), revision=resolved_revision,
                    cache_coverage_known=cache_coverage_known, dry_run_supported=False, planning_mode=mode,
                ), cached_paths,
            )

        resolved_token = self.effective_token(token)
        workers = self._snapshot_worker_count(max_workers, token=resolved_token)
        kwargs: dict[str, Any] = {
            "repo_id": repo_id, "repo_type": "dataset", "revision": resolved_revision,
            "allow_patterns": clean_paths, "max_workers": workers, "token": resolved_token, "dry_run": True,
            "tqdm_class": SilentTqdm,
        }
        if self.configured_cache_dir is not None:
            kwargs["cache_dir"] = str(self.configured_cache_dir)
        try:
            infos = snapshot_download(**kwargs)
        except TypeError:
            cached_paths, local_snapshot_resolved = self._best_local_snapshot_coverage(
                repo_id, clean_paths, revision=resolved_revision, token=token
            )
            cache_state = self.cache_status(repo_id, revision=resolved_revision)
            cache_coverage_known = bool(local_snapshot_resolved or not cache_state.hub_cached)
            return _apply_tree_sizes(
                HFSnapshotPlan(
                    repo_id=str(repo_id), revision=resolved_revision,
                    cache_coverage_known=cache_coverage_known, dry_run_supported=False,
                    planning_mode="dry_run_unsupported_repository_tree",
                ), cached_paths,
            )

        plan = HFSnapshotPlan(
            repo_id=str(repo_id), revision=resolved_revision, total_files=len(infos or []),
            cache_coverage_known=True, dry_run_supported=True, planning_mode="dry_run",
        )
        for info in infos or []:
            size = int(getattr(info, "file_size", 0) or 0)
            cached = bool(getattr(info, "is_cached", False))
            will_download = bool(getattr(info, "will_download", not cached))
            plan.total_bytes += size
            plan.known_size_files += 1
            if cached:
                plan.cached_files += 1
                plan.cached_bytes += size
            if will_download:
                plan.download_files += 1
                plan.download_bytes += size
                plan.download_known_size_files += 1
        return plan

    @staticmethod
    def _snapshot_missing_paths(snapshot_path: str | Path, requested_paths: list[str]) -> list[str]:
        """Verify requested files exist in an incomplete or legacy snapshot before accepting cached
        coverage.
        """
        root = Path(snapshot_path).expanduser()
        missing: list[str] = []
        for raw_path in requested_paths:
            relative = str(raw_path or "").strip()
            if not relative:
                continue
            candidate = root / relative
            try:
                present = candidate.is_file()
            except OSError:
                present = False
            if not present:
                missing.append(relative)
        return missing

    @staticmethod
    def _snapshot_missing_message(
        *, repo_id: str, snapshot_path: str | Path, missing_paths: list[str], requested_count: int,
    ) -> str:
        sample = ", ".join(missing_paths[:3])
        if len(missing_paths) > 3:
            sample += f", … ({len(missing_paths) - 3} more)"
        return (
            f"Hugging Face snapshot for {repo_id} is incomplete: "
            f"{len(missing_paths):,}/{requested_count:,} requested file(s) are missing"
            + (f" ({sample})" if sample else "") + f". Snapshot: {snapshot_path}"
        )

    def _snapshot_worker_script_path(self) -> Path:
        return Path(__file__).with_name("snapshot_worker.py")

    @staticmethod
    def _terminate_transfer_process(process: Any) -> None:
        """Terminate an isolated snapshot worker without waiting on Hub threads."""
        if process is None:
            return
        try:
            if process.poll() is not None:
                return
        except Exception:
            pass
        try:
            process.terminate()
        except Exception:
            pass
        try:
            process.wait(timeout=1.5)
            return
        except Exception:
            pass
        try:
            process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=1.5)
        except Exception:
            pass

    def _snapshot_download_interruptible(
        self, repo_id: str, file_paths: list[str], *, revision: str, token: str | None, max_workers: int,
        cancel_token: Any = None, progress_callback: Any = None,
        snapshot_plan: HFSnapshotPlan | None = None,
    ) -> str:
        """Run Hub snapshots in a terminable subprocess so cancellation can stop retries and internal
        worker threads.
        """
        clean_paths = list(dict.fromkeys(str(path) for path in file_paths if str(path).strip()))
        if not clean_paths:
            raise ValueError("No file paths supplied for snapshot download.")

        worker_path = self._snapshot_worker_script_path()
        if not worker_path.is_file():
            raise RuntimeError(f"Hugging Face snapshot worker is missing: {worker_path}")

        plan = snapshot_plan
        payload = {
            "repo_id": str(repo_id), "revision": str(revision), "file_paths": clean_paths,
            "max_workers": max(1, int(max_workers or 1)), "token": token or None,
            "cache_dir": str(self.configured_cache_dir) if self.configured_cache_dir is not None else None,
            "selected_total_bytes": (
                int(plan.total_bytes) if plan is not None and plan.total_size_complete else 0
            ), "expected_download_bytes": (
                int(plan.download_bytes) if plan is not None and plan.download_size_complete else 0
            ), "expected_download_files": (
                int(plan.download_files) if plan is not None else len(clean_paths)
            ), "cached_files": int(plan.cached_files) if plan is not None else 0,
            "cached_bytes": int(plan.cached_bytes) if plan is not None else 0,
        }

        process = subprocess.Popen(
            [sys.executable, "-u", str(worker_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps(payload))
        process.stdin.close()

        messages: queue.Queue[str | None] = queue.Queue()

        def _reader() -> None:
            try:
                assert process.stdout is not None
                for raw_line in process.stdout:
                    messages.put(raw_line.rstrip("\n"))
            finally:
                messages.put(None)

        reader = threading.Thread(target=_reader, name=f"hf-snapshot-reader:{repo_id}", daemon=True)
        reader.start()

        prefix = "__ASTRONOMICAL_HF_SNAPSHOT__ "
        result_path = ""
        worker_error: dict[str, Any] | None = None
        reader_done = False

        try:
            while True:
                if self._cancelled(cancel_token):
                    if callable(progress_callback):
                        try:
                            progress_callback(
                                {
                                    "phase": "cancelling", "completed": 0, "total": len(clean_paths),
                                    "percent": -1, "progress_known": False, "message": (
                                        "Cancellation requested. Stopping the isolated Hugging Face "
                                        "snapshot process now; completed cache files will be kept for reuse."
                                    ), "current_file": "", "severity": "warning",
                                }
                            )
                        except Exception:
                            pass
                    self._terminate_transfer_process(process)
                    raise HFTransferCancelled("Hugging Face snapshot transfer cancelled.")

                drained = False
                while True:
                    try:
                        line = messages.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    if line is None:
                        reader_done = True
                        continue
                    if not line.startswith(prefix):
                        continue
                    try:
                        message = json.loads(line[len(prefix):])
                    except Exception:
                        continue
                    kind = str(message.get("type") or "")
                    if kind == "progress":
                        if callable(progress_callback):
                            payload_out = dict(message)
                            payload_out.pop("type", None)
                            try:
                                progress_callback(payload_out)
                            except Exception:
                                pass
                    elif kind == "result":
                        result_path = str(message.get("path") or "")
                    elif kind == "error":
                        worker_error = dict(message)

                return_code = process.poll()
                if return_code is not None and reader_done:
                    break
                if return_code is not None and not drained:
                    # Give the daemon reader a short chance to flush the final JSON line.
                    reader.join(timeout=0.05)
                time.sleep(0.05)
        finally:
            if self._cancelled(cancel_token):
                self._terminate_transfer_process(process)

        if worker_error is not None:
            error_type = str(worker_error.get("error_type") or "RuntimeError")
            message = str(worker_error.get("message") or "Hugging Face snapshot worker failed.")
            raise RuntimeError(f"{error_type}: {message}")
        if process.returncode not in (0, None):
            raise RuntimeError(
                f"Hugging Face snapshot worker exited with code {process.returncode} without a result."
            )
        if not result_path:
            raise RuntimeError("Hugging Face snapshot worker completed without returning a snapshot path.")
        return result_path

    def snapshot_download_files(
        self, repo_id: str, file_paths: list[str], *, revision: str | None = None, token: str | None = None,
        max_workers: int = 16, quiet: bool = True, cancel_token: Any = None, progress_callback: Any = None,
        snapshot_plan: HFSnapshotPlan | None = None,
    ) -> str:
        """Resolve a filtered immutable snapshot with visible, cancellable 429 backoff."""
        from huggingface_hub import snapshot_download

        clean_paths = list(dict.fromkeys(str(path) for path in file_paths if str(path).strip()))
        if not clean_paths:
            raise ValueError("No file paths supplied for snapshot download.")

        if self._cancelled(cancel_token):
            raise HFTransferCancelled("Hugging Face snapshot transfer cancelled.")

        resolved_revision = self.resolve_dataset_revision(repo_id, revision=revision, token=token)
        resolved_token = self.effective_token(token)
        workers = self._snapshot_worker_count(max_workers, token=resolved_token)
        tqdm_class = (
            _cancellable_tqdm_class(cancel_token, progress_callback, expected_files=len(clean_paths))
            if cancel_token is not None or callable(progress_callback) else SilentTqdm
        )

        def _kwargs(cache_root: Path | None, *, local_only: bool) -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                "repo_id": repo_id, "repo_type": "dataset", "revision": resolved_revision,
                "allow_patterns": clean_paths, "max_workers": workers, "token": resolved_token,
                "local_files_only": bool(local_only),
            }
            if cache_root is not None:
                kwargs["cache_dir"] = str(cache_root)
            if quiet:
                kwargs["tqdm_class"] = tqdm_class
            return kwargs

        # First try every known cache without any network traffic.
        for cache_root in self._hub_cache_roots():
            if self._cancelled(cancel_token):
                raise HFTransferCancelled("Hugging Face snapshot transfer cancelled.")
            try:
                path = self._call_with_compatible_kwargs(
                    snapshot_download, _kwargs(cache_root, local_only=True),
                )
                missing = self._snapshot_missing_paths(path, clean_paths)
                if missing:
                    if callable(progress_callback):
                        progress_callback(
                            {
                                "phase": "snapshot_partial_cache",
                                "completed": len(clean_paths) - len(missing), "total": len(clean_paths),
                                "percent": int(((len(clean_paths) - len(missing)) / max(len(clean_paths), 1)) * 100),
                                "progress_known": True, "progress_basis": "files_cached", "message": (
                                    f"Partial Hugging Face cache detected: {len(missing):,}/{len(clean_paths):,} "
                                    "requested file(s) are missing. AstronomicAL will complete the pinned "
                                    "snapshot from the Hub instead of using incomplete cached data."
                                ), "current_file": missing[0] if missing else "", "severity": "warning",
                            }
                        )
                    continue
                if callable(progress_callback):
                    progress_callback(
                        {
                            "phase": "snapshot_cached", "completed": len(clean_paths),
                            "total": len(clean_paths), "percent": 100, "progress_known": True,
                            "progress_basis": "files_cached",
                            "message": "Complete pinned snapshot found in the local Hugging Face cache.",
                            "current_file": "", "severity": "info",
                        }
                    )
                return str(path)
            except HFTransferCancelled:
                raise
            except Exception:
                continue

        if callable(progress_callback) and not resolved_token:
            progress_callback(
                {
                    "phase": "anonymous_download", "completed": 0, "total": len(clean_paths), "percent": 10,
                    "message": (
                        "No authenticated Hugging Face token was detected. Anonymous imports use fewer "
                        "parallel workers to reduce rate-limit bursts; add a token in Search options for larger downloads."
                    ), "current_file": "", "severity": "warning",
                }
            )

        attempt = 0
        while True:
            if self._cancelled(cancel_token):
                raise HFTransferCancelled("Hugging Face snapshot transfer cancelled.")
            try:
                path = self._snapshot_download_interruptible(
                    repo_id, clean_paths, revision=resolved_revision, token=resolved_token,
                    max_workers=workers, cancel_token=cancel_token, progress_callback=progress_callback,
                    snapshot_plan=snapshot_plan,
                )
                missing = self._snapshot_missing_paths(path, clean_paths)
                if missing:
                    message = self._snapshot_missing_message(
                        repo_id=repo_id, snapshot_path=path, missing_paths=missing,
                        requested_count=len(clean_paths),
                    )
                    if callable(progress_callback):
                        progress_callback(
                            {
                                "phase": "snapshot_incomplete",
                                "completed": len(clean_paths) - len(missing), "total": len(clean_paths),
                                "percent": int(((len(clean_paths) - len(missing)) / max(len(clean_paths), 1)) * 100),
                                "progress_known": True, "progress_basis": "files_present", "message": (
                                    message + ". The incomplete snapshot will not be registered. "
                                    "Retry with network access; completed cache entries will be reused."
                                ), "current_file": missing[0] if missing else "", "severity": "warning",
                            }
                        )
                    raise HFIncompleteSnapshotError(
                        message, snapshot_path=str(path), missing_paths=missing,
                    )
                return str(path)
            except HFTransferCancelled:
                raise
            except HFRateLimitBackoff as exc:
                attempt += 1
                if attempt > MAX_RATE_LIMIT_RETRIES:
                    raise RuntimeError(
                        f"Hugging Face rate limit persisted after {MAX_RATE_LIMIT_RETRIES} retries. "
                        "Add an authenticated HF token and retry the import."
                    ) from exc
                self._cooperative_rate_limit_wait(
                    exc.retry_after_seconds, cancel_token=cancel_token, progress_callback=progress_callback,
                    attempt=attempt,
                )
                if callable(progress_callback):
                    progress_callback(
                        {
                            "phase": "snapshot_retry", "completed": 0, "total": len(clean_paths),
                            "percent": 11,
                            "message": "Rate-limit window cleared. Retrying the pinned snapshot; completed cache entries will be reused.",
                            "current_file": "", "severity": "info",
                        }
                    )

    def image_to_preview_data_uri(self, image, *, size: int = 180, fill: bool = True) -> str:
        """Resize a PIL image into a preview data URI, upscaling small source images when needed."""

        from PIL import Image as PILImage
        from PIL import ImageOps

        size = max(32, int(size or 180))

        image = ImageOps.exif_transpose(image)

        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")

        if fill:
            # Fill the full square preview area. This may crop non-square images.
            image = ImageOps.fit(
                image, (size, size), method=PILImage.Resampling.NEAREST, centering=(0.5, 0.5),
            )
        else:
            # Preserve full image without cropping, but still upscale tiny images.
            image.thumbnail((size, size), resample=PILImage.Resampling.LANCZOS)

            canvas = PILImage.new("RGB", (size, size), (17, 17, 17))
            if image.mode != "RGB":
                image = image.convert("RGB")

            x = (size - image.width) // 2
            y = (size - image.height) // 2
            canvas.paste(image, (x, y))
            image = canvas

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def local_image_to_data_uri(self, path: str | Path, *, max_size: int = 180) -> str:
        from PIL import Image as PILImage
        from PIL import ImageOps

        path = Path(path).expanduser()
        if not path.exists():
            return ""

        with PILImage.open(path) as image:
            return self.image_to_preview_data_uri(image, size=int(max_size or 180), fill=True)

    # ------------------------------------------------------------------
    # Legacy Datasets API helpers
    # ------------------------------------------------------------------

    def load_split(
        self, *, repo_id: str, config_name: str | None, split: str, revision: str | None = None,
        token: str | None = None, trust_remote_code: bool = False, streaming: bool = False,
    ):
        from datasets import load_dataset

        def _load(kwargs: dict[str, Any]):
            try:
                if config_name:
                    return load_dataset(repo_id, name=config_name, split=split, **kwargs)
                return load_dataset(repo_id, split=split, **kwargs)
            except TypeError:
                retry = dict(kwargs)
                retry.pop("trust_remote_code", None)
                retry.pop("token", None)
                if config_name:
                    return load_dataset(repo_id, name=config_name, split=split, **retry)
                return load_dataset(repo_id, split=split, **retry)

        for cache_root in self._datasets_cache_roots():
            kwargs = self._datasets_kwargs_for_cache(
                token=token, trust_remote_code=trust_remote_code, cache_root=cache_root,
                local_files_only=True, streaming=streaming,
            )
            if revision:
                kwargs["revision"] = str(revision)
            try:
                return _load(kwargs)
            except Exception:
                continue

        kwargs = self._datasets_kwargs_for_cache(
            token=token, trust_remote_code=trust_remote_code, cache_root=self.configured_cache_dir,
            local_files_only=False, streaming=streaming,
        )
        if revision:
            kwargs["revision"] = str(revision)
        return _load(kwargs)

    def cast_image_decode_false(self, dataset: Any, image_column: str) -> Any:
        try:
            from datasets import Image

            return dataset.cast_column(image_column, Image(decode=False))
        except Exception:
            return dataset

    def infer_image_column(self, features: Any, *, dataset: Any = None) -> str | None:
        for name, feature in dict(features or {}).items():
            if "image" in self._feature_to_string(feature).lower():
                return str(name)

        column_names = self._column_names(dataset)
        for candidate in ("image", "img", "picture", "photo", "file", "filepath", "image_path", "path"):
            if candidate in column_names:
                return candidate

        return None

    def infer_label_column(self, features: Any, *, dataset: Any = None) -> str | None:
        for name, feature in dict(features or {}).items():
            text = self._feature_to_string(feature).lower()
            if "classlabel" in text:
                return str(name)

        column_names = self._column_names(dataset)
        for candidate in ("label", "labels", "target", "class", "category"):
            if candidate in column_names:
                return candidate

        return None

    def record_id_for_row(
        self, row: dict[str, Any], *, index: int, split: str, id_column: str | None = None,
    ) -> str:
        if id_column and id_column in row and row.get(id_column) is not None:
            return str(row.get(id_column))
        for candidate in ("id", "record_id", "image_id", "file_name", "filename", "path"):
            if candidate in row and row.get(candidate) is not None:
                return str(row.get(candidate))
        return f"{split}:{index}"

    def label_to_name(self, features: Any, label_column: str | None, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        feature = None
        if label_column is not None:
            try:
                feature = features[label_column]
            except Exception:
                feature = None

        if isinstance(value, (list, tuple)):
            names = [self.label_to_name(features, label_column, item) for item in value]
            return ", ".join(name for name in names if name)

        if feature is not None and hasattr(feature, "int2str"):
            try:
                return str(feature.int2str(int(value)))
            except Exception:
                pass

        names = getattr(feature, "names", None)
        if names is not None:
            try:
                return str(names[int(value)])
            except Exception:
                pass

        try:
            import numpy as np

            if isinstance(value, np.generic):
                value = value.item()
        except Exception:
            pass

        return str(value)

    def choose_builder_columns(
        self, details: HFDatasetDetails, *, config: str,
    ) -> tuple[str | None, str | None, str | None]:
        features = dict(details.features_by_config.get(config, {}) or {})
        columns = list(features)

        image_candidates = list(details.image_column_candidates or [])
        if not image_candidates:
            image_candidates, _ = self._feature_candidates_from_features(features)

        label_candidates = list(details.label_column_candidates or [])
        if not label_candidates:
            _, label_candidates = self._feature_candidates_from_features(features)

        image_column = image_candidates[0] if image_candidates else None

        label_column = None
        for candidate in label_candidates:
            if not self.feature_looks_like_shard_labels(features.get(candidate, "")):
                label_column = candidate
                break

        id_column = None
        for candidate in ("record_id", "id", "image_id", "file_name", "filename", "path"):
            if candidate in columns:
                id_column = candidate
                break

        return image_column, label_column, id_column

    # ------------------------------------------------------------------
    # Path inference helpers

    # ------------------------------------------------------------------

    @staticmethod
    def _is_image_path(path: str) -> bool:
        suffix = Path(path).suffix.lower()
        return suffix in IMAGE_EXTENSIONS

    @staticmethod
    def _normalise_split(value: str | None) -> str:
        if not value:
            return "train"
        cleaned = str(value).strip().lower()
        return SPLIT_ALIASES.get(cleaned, cleaned)

    def infer_split_from_path(self, path: str) -> str:
        parts = [part.lower() for part in Path(path).parts]
        stem = Path(path).stem.lower()

        for part in parts:
            cleaned = re.sub(r"[^a-z0-9]+", "", part)
            if cleaned in SPLIT_ALIASES:
                return SPLIT_ALIASES[cleaned]

        tokens = re.split(r"[^a-z0-9]+", stem)
        for token in tokens:
            if token in SPLIT_ALIASES:
                return SPLIT_ALIASES[token]

        for key, split in SPLIT_ALIASES.items():
            if re.search(rf"(^|[_\-.]){re.escape(key)}([_\-.]|$)", stem):
                return split

        return "train"

    def infer_label_from_path(self, path: str, split: str = "train") -> str:
        """Compatibility wrapper returning only the primary target candidate."""
        return self.infer_label_details_from_path(path, split).label

    def infer_label_details_from_path(self, path: str, split: str = "train") -> HFPathLabelInfo:
        """Infer primary labels from folders or filenames while keeping paired prediction labels separate;
        native FiftyOne annotations take precedence.
        """
        parts = list(Path(path).parts)

        if len(parts) >= 2:
            parent = parts[-2]
            parent_clean = parent.strip().lower()
            if (
                not self._looks_like_non_label_folder(parent_clean)
                and self._normalise_split(parent_clean) != split
            ):
                label = self._normalise_filename_label_candidate(parent)
                if label:
                    return HFPathLabelInfo(
                        label=label, source="parent_folder", primary_label=label, parser="parent_folder_v1",
                    )

        return self.infer_label_details_from_filename(path)

    def infer_label_from_filename(self, path: str) -> str:
        """Compatibility wrapper returning only the primary filename label."""
        return self.infer_label_details_from_filename(path).label

    def infer_label_details_from_filename(self, path: str) -> HFPathLabelInfo:
        """Parse label-bearing filenames without collapsing distinct roles."""
        stem = Path(path).stem.strip()
        if not stem:
            return HFPathLabelInfo()

        # FiftyOne repositories are commonly exported with a human-readable
        # ground-truth/prediction pair in the filename.  The first numeric value
        # is preserved as an opaque prefix value; no semantic meaning is assumed.
        paired = re.match(
            r"^\s*(?P<prefix>[+-]?(?:\d+(?:\.\d*)?|\.\d+))[_-]" r"(?P<primary>.+?)\s+_\s+"
            r"(?P<secondary>.+?)[_-](?P<confidence>[01]?(?:\.\d+))\s*$", stem,
        )
        if paired:
            primary = self._normalise_filename_label_candidate(paired.group("primary"))
            secondary = self._normalise_filename_label_candidate(paired.group("secondary"))
            if (
                self._looks_like_filename_label_candidate(primary)
                and self._looks_like_filename_label_candidate(secondary)
            ):
                try:
                    confidence = float(paired.group("confidence"))
                except Exception:
                    confidence = None
                try:
                    prefix_value = float(paired.group("prefix"))
                except Exception:
                    prefix_value = None
                return HFPathLabelInfo(
                    label=primary, source="paired_filename", primary_label=primary,
                    secondary_label=secondary, secondary_confidence=confidence, prefix_value=prefix_value,
                    parser="paired_filename_classification_v1", ambiguous=bool(
                        primary and secondary and primary.casefold() != secondary.casefold()
                    ),
                )

        has_double_separator = "__" in stem
        has_score_suffix = bool(re.search(r"[_\-.][01]?\.\d+$", stem))
        has_leading_numeric_id = bool(re.match(r"^\d+[_\-.]+[A-Za-z]", stem))

        if not (has_double_separator or has_score_suffix or has_leading_numeric_id):
            return HFPathLabelInfo()

        raw_candidates: list[str] = []
        if has_double_separator:
            raw_candidates.extend(part for part in stem.split("__") if part)
        raw_candidates.append(stem)

        normalised: list[str] = []
        for candidate in raw_candidates:
            label = self._normalise_filename_label_candidate(candidate)
            if self._looks_like_filename_label_candidate(label):
                normalised.append(label)

        if not normalised:
            return HFPathLabelInfo()

        counts = Counter(normalised)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
        label = ranked[0][0]
        return HFPathLabelInfo(
            label=label, source="filename", primary_label=label, parser="generic_filename_v1",
        )

    def _normalise_filename_label_candidate(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""

        # Remove leading numeric IDs.
        text = re.sub(r"^\d+[_\-.]+", "", text)

        # Remove trailing confidence-like scores.
        text = re.sub(r"([_\-.]+[01]?\.\d+)$", "", text)

        # Remove common score suffix forms.
        text = re.sub(r"([_\-.]+score[_\-.]*[01]?\.\d+)$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"([_\-.]+conf[_\-.]*[01]?\.\d+)$", "", text, flags=re.IGNORECASE)

        # Strip leftover separators.
        text = re.sub(r"[_\-.]+$", "", text)
        text = re.sub(r"^[_\-.]+", "", text)

        # Convert separators to readable labels.
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _looks_like_filename_label_candidate(self, value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False

        lowered = text.lower()

        if self._looks_like_non_label_folder(lowered):
            return False

        if lowered in {"train", "test", "validation", "valid", "val"}:
            return False

        if lowered.isdigit():
            return False

        # Avoid huge captions/paths being treated as labels.
        if len(text) > 80:
            return False

        # Require at least one alphabetic character. This intentionally rejects
        # pure MNIST digit filenames unless the label is provided elsewhere.
        if not re.search(r"[A-Za-z]", text):
            return False

        return True

    def _looks_like_non_label_folder(self, folder_name: str) -> bool:
        cleaned = str(folder_name or "").strip().lower()
        if cleaned in GENERIC_FOLDERS:
            return True

        for pattern in GENERIC_FOLDER_PATTERNS:
            if re.match(pattern, cleaned):
                return True

        return False

    # ------------------------------------------------------------------
    # Generic utilities
    # ------------------------------------------------------------------

    def _hf_kwargs(self, *, token: str | None = None, trust_remote_code: bool = False) -> dict[str, Any]:
        return self._datasets_kwargs_for_cache(
            token=token, trust_remote_code=trust_remote_code, cache_root=self.configured_cache_dir,
            local_files_only=False,
        )

    @staticmethod
    def _feature_to_string(feature: Any) -> str:
        try:
            return feature.__class__.__name__ + ": " + str(feature)
        except Exception:
            return str(feature)

    @staticmethod
    def _column_names(dataset: Any) -> list[str]:
        if dataset is None:
            return []
        try:
            return list(dataset.column_names)
        except Exception:
            pass
        try:
            features = getattr(dataset, "features", {}) or {}
            return list(features.keys())
        except Exception:
            return []

    @staticmethod
    def _short_exception(exc: BaseException, *, limit: int = 500) -> str:
        text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        text = " ".join(text.split())
        if len(text) > limit:
            return text[:limit] + "… [truncated]"
        return text
