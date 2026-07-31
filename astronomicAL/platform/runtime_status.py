from __future__ import annotations

import threading
import time
import traceback as traceback_mod
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional


@dataclass(frozen=True)
class UiLagRecord:
    timestamp: float
    lag: float
    expected_interval: float
    source: str = "ui_heartbeat"


@dataclass(frozen=True)
class RuntimeErrorRecord:
    timestamp: float
    source: str
    message: str
    details: Optional[str] = None
    traceback_text: Optional[str] = None


@dataclass(frozen=True)
class StatusNoteRecord:
    timestamp: float
    source: str
    message: str
    details: Optional[str] = None


@dataclass(frozen=True)
class RuntimeDiagnosticsSnapshot:
    """Consistent, read-only copy of RuntimeStatus diagnostic history."""

    captured_at: float
    ui_lags: tuple[UiLagRecord, ...]
    errors: tuple[RuntimeErrorRecord, ...]
    notes: tuple[StatusNoteRecord, ...]


class RuntimeStatus:
    """Small shared runtime diagnostics service.

    The service intentionally does not publish events when recording diagnostics,
    because it may be called from inside EventBus, Panel, or Bokeh callback paths.
    Runtime observability views poll this service directly.
    """

    def __init__(self, *, history_limit: int = 500) -> None:
        self._lock = threading.RLock()
        self._ui_lags: Deque[UiLagRecord] = deque(maxlen=history_limit)
        self._errors: Deque[RuntimeErrorRecord] = deque(maxlen=history_limit)
        self._notes: Deque[StatusNoteRecord] = deque(maxlen=history_limit)

    def record_ui_lag(
        self,
        lag: float,
        *,
        expected_interval: float,
        source: str = "ui_heartbeat",
    ) -> None:
        try:
            lag = float(lag)
            expected_interval = float(expected_interval)
        except Exception:
            return

        with self._lock:
            self._ui_lags.append(
                UiLagRecord(
                    timestamp=time.time(),
                    lag=lag,
                    expected_interval=expected_interval,
                    source=str(source),
                )
            )

    def recent_ui_lags(self, n: int = 50) -> List[UiLagRecord]:
        with self._lock:
            if n <= 0:
                return []
            return list(self._ui_lags)[-n:]

    def record_error(
        self,
        source: str,
        message: str,
        *,
        details: Optional[str] = None,
        exc: Optional[BaseException] = None,
        traceback_text: Optional[str] = None,
    ) -> None:
        if traceback_text is None and exc is not None:
            traceback_text = "".join(
                traceback_mod.format_exception(type(exc), exc, exc.__traceback__)
            )

        with self._lock:
            self._errors.append(
                RuntimeErrorRecord(
                    timestamp=time.time(),
                    source=str(source),
                    message=str(message),
                    details=details,
                    traceback_text=traceback_text,
                )
            )

    def recent_errors(self, n: int = 50) -> List[RuntimeErrorRecord]:
        with self._lock:
            if n <= 0:
                return []
            return list(self._errors)[-n:]

    def record_note(
        self,
        source: str,
        message: str,
        *,
        details: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._notes.append(
                StatusNoteRecord(
                    timestamp=time.time(),
                    source=str(source),
                    message=str(message),
                    details=details,
                )
            )

    def recent_notes(self, n: int = 50) -> List[StatusNoteRecord]:
        with self._lock:
            if n <= 0:
                return []
            return list(self._notes)[-n:]

    def diagnostic_snapshot(self, *, limit: int = 50) -> RuntimeDiagnosticsSnapshot:
        """Return runtime diagnostics from one lock-protected capture."""
        safe_limit = max(1, int(limit))
        with self._lock:
            return RuntimeDiagnosticsSnapshot(
                captured_at=time.time(),
                ui_lags=tuple(list(self._ui_lags)[-safe_limit:]),
                errors=tuple(list(self._errors)[-safe_limit:]),
                notes=tuple(list(self._notes)[-safe_limit:]),
            )

    def clear(self) -> None:
        with self._lock:
            self._ui_lags.clear()
            self._errors.clear()
            self._notes.clear()