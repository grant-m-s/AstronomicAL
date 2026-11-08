from __future__ import annotations

"""Isolated Hugging Face snapshot transfer worker.

This module is intentionally executable as a plain Python script.  AstronomicAL
runs network-backed ``snapshot_download`` calls in this child process so the
parent JobManager can terminate the transfer promptly on cancellation without
waiting for Hugging Face's internal thread pool or rate-limit sleep to unwind.
"""

import json
import logging
import os
import re
import sys
import traceback
from contextlib import contextmanager
from threading import RLock
from typing import Any

PROTOCOL_PREFIX = "__ASTRONOMICAL_HF_SNAPSHOT__ "
_EXPECTED_FILES = 0
_FILES_COMPLETED = 0
_TRANSFER_BYTES = 0
_SELECTED_TOTAL_BYTES = 0
_EXPECTED_DOWNLOAD_BYTES = 0
_EXPECTED_DOWNLOAD_FILES = 0
_CACHED_FILES = 0
_CACHED_BYTES = 0
_TRANSFER_ACTIVE_BARS: dict[int, int] = {}
_TRANSFER_COMPLETED_BYTES = 0
_TRANSFER_MANUAL_BYTES = 0
_LOCK = RLock()

def _emit(payload: dict[str, Any]) -> None:
    try:
        sys.stdout.write(PROTOCOL_PREFIX + json.dumps(payload, sort_keys=True, default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        pass

def _format_bytes(value: int) -> str:
    amount = max(int(value or 0), 0)
    number = float(amount)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024.0 or unit == "TiB":
            return f"{int(number):,} {unit}" if unit == "B" else f"{number:.1f} {unit}"
        number /= 1024.0
    return f"{amount:,} B"

def _refresh_transfer_bytes_locked() -> None:
    global _TRANSFER_BYTES
    _TRANSFER_BYTES = max(
        int(_TRANSFER_MANUAL_BYTES)
        + int(_TRANSFER_COMPLETED_BYTES)
        + sum(max(int(value), 0) for value in _TRANSFER_ACTIVE_BARS.values()),
        0,
    )

def _progress_snapshot() -> dict[str, Any]:
    with _LOCK:
        completed = min(max(int(_FILES_COMPLETED), 0), max(int(_EXPECTED_FILES), 0))
        total = max(int(_EXPECTED_FILES), 0)
        transferred = max(int(_TRANSFER_BYTES), 0)
        selected_total = max(int(_SELECTED_TOTAL_BYTES), 0)
        expected_download = max(int(_EXPECTED_DOWNLOAD_BYTES), 0)
        expected_download_files = max(int(_EXPECTED_DOWNLOAD_FILES), 0)
        cached_files = max(int(_CACHED_FILES), 0)
        cached_bytes = max(int(_CACHED_BYTES), 0)

    progress_known = False
    progress_basis = "indeterminate"
    percent = -1
    if expected_download > 0 and transferred > 0:
        percent = int(min(transferred / max(expected_download, 1), 1.0) * 100)
        progress_known = True
        progress_basis = "network_bytes"
    elif total > 0 and completed > 0:
        percent = int(min(completed / max(total, 1), 1.0) * 100)
        progress_known = True
        progress_basis = "files_finalized"

    return {
        "completed": completed,
        "total": total,
        "transfer_bytes": transferred,
        "selected_total_bytes": selected_total,
        "expected_download_bytes": expected_download,
        "expected_download_files": expected_download_files,
        "cached_files": cached_files,
        "cached_bytes": cached_bytes,
        "percent": percent,
        "progress_known": progress_known,
        "progress_basis": progress_basis,
    }

def _emit_progress() -> None:
    state = _progress_snapshot()
    completed = int(state["completed"])
    total = int(state["total"])
    transferred = int(state["transfer_bytes"])
    selected_total = int(state["selected_total_bytes"])
    expected_download = int(state["expected_download_bytes"])
    cached_files = int(state["cached_files"])
    cached_bytes = int(state["cached_bytes"])

    if total and completed == 0 and transferred == 0:
        message = (
            f"Starting snapshot transfer: {total:,} file(s) selected · "
            "waiting for the first measurable file/chunk progress"
        )
    else:
        message = (
            f"Downloading snapshot: {completed:,}/{total:,} files finalized"
            if total
            else "Downloading snapshot"
        )
    if expected_download:
        if transferred:
            message += f" · {_format_bytes(transferred)} / {_format_bytes(expected_download)} network"
        else:
            message += (
                f" · {_format_bytes(expected_download)} network planned"
                " · transfer telemetry has not started yet"
            )
    elif transferred:
        message += f" · {_format_bytes(transferred)} transferred this session"
    if selected_total:
        message += f" · {_format_bytes(selected_total)} selected"
    if cached_files or cached_bytes:
        cache_text = f"{cached_files:,} file(s) cached"
        if cached_bytes:
            cache_text += f" / {_format_bytes(cached_bytes)}"
        message += f" · {cache_text}"
    message += "…"

    payload = {
        "type": "progress",
        "phase": "snapshot_download",
        "message": message,
        "current_file": "",
        "severity": "info",
        "progress_unit": "files",
        **state,
    }
    _emit(payload)

class WorkerTqdm:
    """Small tqdm-compatible bridge that emits aggregate JSON progress.

    Hugging Face may create one byte bar per concurrent file. Track each bar's
    *network delta* independently and sum them rather than reporting the maximum
    individual bar. Initial/resumed bytes are excluded so the UI describes bytes
    transferred by this worker session rather than bytes already present in cache.
    """

    _tqdm_lock = RLock()
    _lock = _tqdm_lock

    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total")
        self.n = int(kwargs.get("initial", 0) or 0)
        self._initial_n = self.n
        self.disable = True
        self._desc = str(kwargs.get("desc") or "")
        self._unit = str(kwargs.get("unit") or "")
        self._name = str(kwargs.get("name") or "")
        self._is_byte_bar = self._unit.casefold() in {"b", "bytes"}
        self._is_transfer_bar = (
            "transfer" in self._name.casefold()
            or "download" in self._desc.casefold()
        )
        self._is_file_bar = (
            not self._is_byte_bar
            and _EXPECTED_FILES > 0
            and int(self.total or 0) == _EXPECTED_FILES
        )
        self._bar_id = id(self)
        self._closed = False
        if self.n:
            self._publish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __iter__(self):
        return iter(())

    def _publish(self) -> None:
        global _FILES_COMPLETED
        with _LOCK:
            if self._is_file_bar:
                _FILES_COMPLETED = max(_FILES_COMPLETED, int(self.n or 0))
            elif self._is_byte_bar:
                _TRANSFER_ACTIVE_BARS[self._bar_id] = max(
                    int(self.n or 0) - int(self._initial_n or 0),
                    0,
                )
                _refresh_transfer_bytes_locked()
        if self._is_file_bar or self._is_byte_bar:
            _emit_progress()

    def update(self, n: int | float | None = 1) -> None:
        self.n += int(n or 0)
        self._publish()

    def update_transfer(self, n: int | float | None = 1) -> None:
        global _TRANSFER_MANUAL_BYTES
        with _LOCK:
            _TRANSFER_MANUAL_BYTES += max(int(n or 0), 0)
            _refresh_transfer_bytes_locked()
        _emit_progress()

    def close(self) -> None:
        global _FILES_COMPLETED, _TRANSFER_COMPLETED_BYTES
        if self._closed:
            return
        self._closed = True
        emit_progress = False
        with _LOCK:
            if self._is_file_bar and self.total and self.n >= int(self.total):
                _FILES_COMPLETED = max(_FILES_COMPLETED, _EXPECTED_FILES)
                emit_progress = True
            if self._is_byte_bar:
                current = max(int(self.n or 0) - int(self._initial_n or 0), 0)
                active = _TRANSFER_ACTIVE_BARS.pop(self._bar_id, 0)
                _TRANSFER_COMPLETED_BYTES += max(current, active)
                _refresh_transfer_bytes_locked()
                emit_progress = True
        if emit_progress:
            _emit_progress()

    def clear(self, *args, **kwargs) -> None:
        pass

    def reset(self, *args, **kwargs) -> None:
        global _TRANSFER_COMPLETED_BYTES
        if self._is_byte_bar:
            with _LOCK:
                current = max(int(self.n or 0) - int(self._initial_n or 0), 0)
                active = _TRANSFER_ACTIVE_BARS.get(self._bar_id, 0)
                _TRANSFER_COMPLETED_BYTES += max(current, active)
                _TRANSFER_ACTIVE_BARS[self._bar_id] = 0
                _refresh_transfer_bytes_locked()
        self.n = 0
        self._initial_n = 0
        self._closed = False

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

    def set_transfer_postfix_str(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def get_lock(cls):
        return cls._tqdm_lock

    @classmethod
    def set_lock(cls, lock) -> None:
        cls._tqdm_lock = lock
        cls._lock = lock

    @classmethod
    def write(cls, *args, **kwargs) -> None:
        pass

    @classmethod
    @contextmanager
    def external_write_mode(cls, *args, **kwargs):
        yield

class RateLimitHandler(logging.Handler):
    """Forward only useful Hub rate-limit messages to the parent process."""

    def emit(self, record: logging.LogRecord) -> None:
        message = str(record.getMessage() or "")
        if "429" not in message and "Rate limited. Waiting" not in message:
            return
        match = re.search(r"Waiting\s+([0-9]+(?:\.[0-9]+)?)s", message)
        seconds = int(float(match.group(1))) if match else None
        text = "Hugging Face rate limit reached."
        if seconds is not None:
            text += f" Hub retry wait is about {seconds}s."
        text += " You can cancel the import immediately."
        state = _progress_snapshot()
        _emit(
            {
                "type": "progress",
                "phase": "rate_limited",
                "message": text,
                "current_file": "",
                "severity": "warning",
                "retry_after_seconds": seconds,
                **state,
            }
        )

def _configure_logging() -> None:
    logger = logging.getLogger("huggingface_hub")
    logger.addHandler(RateLimitHandler(level=logging.WARNING))
    logger.setLevel(logging.WARNING)
    logger.propagate = False

def main() -> int:
    global _EXPECTED_FILES, _SELECTED_TOTAL_BYTES, _EXPECTED_DOWNLOAD_BYTES
    global _EXPECTED_DOWNLOAD_FILES, _CACHED_FILES, _CACHED_BYTES
    global _TRANSFER_BYTES, _TRANSFER_COMPLETED_BYTES, _TRANSFER_MANUAL_BYTES
    try:
        with _LOCK:
            _TRANSFER_BYTES = 0
            _TRANSFER_COMPLETED_BYTES = 0
            _TRANSFER_MANUAL_BYTES = 0
            _TRANSFER_ACTIVE_BARS.clear()
        raw = sys.stdin.read()
        payload = json.loads(raw or "{}")
        file_paths = [str(path) for path in payload.get("file_paths") or [] if str(path).strip()]
        _EXPECTED_FILES = len(file_paths)
        _SELECTED_TOTAL_BYTES = max(int(payload.get("selected_total_bytes") or 0), 0)
        _EXPECTED_DOWNLOAD_BYTES = max(int(payload.get("expected_download_bytes") or 0), 0)
        _EXPECTED_DOWNLOAD_FILES = max(int(payload.get("expected_download_files") or 0), 0)
        _CACHED_FILES = max(int(payload.get("cached_files") or 0), 0)
        _CACHED_BYTES = max(int(payload.get("cached_bytes") or 0), 0)
        _configure_logging()
        _emit_progress()

        from huggingface_hub import snapshot_download

        kwargs: dict[str, Any] = {
            "repo_id": str(payload["repo_id"]),
            "repo_type": "dataset",
            "revision": str(payload.get("revision") or "main"),
            "allow_patterns": file_paths,
            "max_workers": max(1, int(payload.get("max_workers") or 1)),
            "local_files_only": False,
            "tqdm_class": WorkerTqdm,
        }
        token = payload.get("token")
        if token:
            kwargs["token"] = str(token)
        cache_dir = payload.get("cache_dir")
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)

        path = snapshot_download(**kwargs)
        _emit({"type": "result", "path": str(path)})
        return 0
    except BaseException as exc:
        _emit(
            {
                "type": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
        )
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
