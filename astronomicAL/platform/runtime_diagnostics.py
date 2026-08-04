from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

ACTIVE_PUBLISH_STALL_SECONDS = 1.0
RECENT_ISSUE_WINDOW_SECONDS = 300.0
RECENT_HEADER_WINDOW_SECONDS = 30.0
SLOW_EVENT_P95_MS = 50.0


@dataclass(frozen=True)
class RuntimeDiagnosticsData:
    captured_at: float
    active_jobs: tuple[Any, ...]
    recent_jobs: tuple[Any, ...]
    subscriptions: tuple[Any, ...]
    active_publishes: tuple[Any, ...]
    slow_callbacks: tuple[Any, ...]
    publish_timings: tuple[Any, ...]
    callback_errors: tuple[Any, ...]
    ui_lags: tuple[Any, ...]
    runtime_errors: tuple[Any, ...]
    runtime_notes: tuple[Any, ...]


@dataclass(frozen=True)
class RuntimeHealthSummary:
    status: str
    tone: str
    headline: str
    detail: str
    active_job_count: int
    active_publish_count: int
    stalled_publish_count: int
    recent_slow_count: int
    recent_callback_error_count: int
    recent_runtime_error_count: int
    recent_ui_lag_count: int
    recent_note_count: int
    event_p95_ms: float
    event_max_ms: float
    latest_ui_lag_ms: float
    max_ui_lag_ms: float
    subscriber_count: int

    @property
    def recent_error_count(self) -> int:
        return self.recent_callback_error_count + self.recent_runtime_error_count

    @property
    def issue_count(self) -> int:
        return (
            self.stalled_publish_count
            + self.recent_slow_count
            + self.recent_error_count
            + self.recent_ui_lag_count
        )


def value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def collect_runtime_diagnostics(
    context: Any,
    *,
    event_limit: int = 100,
    job_limit: int = 20,
    runtime_limit: int = 20,
) -> RuntimeDiagnosticsData:
    """Collect a bounded, read-only view of runtime diagnostics.

    EventBus and RuntimeStatus atomic snapshot APIs are preferred when available.
    Older implementations remain supported through their existing list methods.
    """
    captured_at = time.time()
    events = getattr(context, "events", None)
    jobs = getattr(context, "jobs", None)
    runtime_status = getattr(context, "runtime_status", None)

    active_jobs = tuple(_call_list(jobs, "active_jobs"))
    recent_jobs = tuple(_call_list(jobs, "recent_jobs", max(1, int(job_limit))))

    event_snapshot = _call(events, "diagnostic_snapshot", limit=max(1, int(event_limit)))
    if event_snapshot is not None:
        captured_at = float(value(event_snapshot, "captured_at", captured_at) or captured_at)
        subscriptions = tuple(value(event_snapshot, "subscriptions", ()) or ())
        active_publishes = tuple(value(event_snapshot, "active_publishes", ()) or ())
        slow_callbacks = tuple(value(event_snapshot, "slow_callbacks", ()) or ())
        publish_timings = tuple(value(event_snapshot, "publish_timings", ()) or ())
        callback_errors = tuple(value(event_snapshot, "callback_errors", ()) or ())
    else:
        subscriptions = tuple(_call_list(events, "list_subscriptions"))
        active_publishes = tuple(_call_list(events, "active_publishes"))
        slow_callbacks = tuple(
            _call_list(events, "recent_slow_callbacks", max(1, int(event_limit)))
        )
        publish_timings = tuple(
            _call_list(events, "recent_publish_timings", max(1, int(event_limit)))
        )
        callback_errors = tuple(
            _call_list(events, "recent_callback_errors", max(1, int(event_limit)))
        )

    runtime_snapshot = _call(
        runtime_status,
        "diagnostic_snapshot",
        limit=max(1, int(runtime_limit)),
    )
    if runtime_snapshot is not None:
        ui_lags = tuple(value(runtime_snapshot, "ui_lags", ()) or ())
        runtime_errors = tuple(value(runtime_snapshot, "errors", ()) or ())
        runtime_notes = tuple(value(runtime_snapshot, "notes", ()) or ())
    else:
        ui_lags = tuple(
            _call_list(runtime_status, "recent_ui_lags", max(1, int(runtime_limit)))
        )
        runtime_errors = tuple(
            _call_list(runtime_status, "recent_errors", max(1, int(runtime_limit)))
        )
        runtime_notes = tuple(
            _call_list(runtime_status, "recent_notes", max(1, int(runtime_limit)))
        )

    return RuntimeDiagnosticsData(
        captured_at=captured_at,
        active_jobs=active_jobs,
        recent_jobs=recent_jobs,
        subscriptions=subscriptions,
        active_publishes=active_publishes,
        slow_callbacks=slow_callbacks,
        publish_timings=publish_timings,
        callback_errors=callback_errors,
        ui_lags=ui_lags,
        runtime_errors=runtime_errors,
        runtime_notes=runtime_notes,
    )


def build_runtime_health(
    data: RuntimeDiagnosticsData,
    *,
    recent_window_seconds: float = RECENT_ISSUE_WINDOW_SECONDS,
) -> RuntimeHealthSummary:
    cutoff = data.captured_at - max(1.0, float(recent_window_seconds))
    header_cutoff = data.captured_at - RECENT_HEADER_WINDOW_SECONDS

    recent_slow = tuple(item for item in data.slow_callbacks if _timestamp(item) >= cutoff)
    recent_callback_errors = tuple(
        item for item in data.callback_errors if _timestamp(item) >= cutoff
    )
    recent_runtime_errors = tuple(
        item for item in data.runtime_errors if _timestamp(item) >= cutoff
    )
    recent_ui_lags = tuple(item for item in data.ui_lags if _timestamp(item) >= cutoff)
    recent_notes = tuple(item for item in data.runtime_notes if _timestamp(item) >= cutoff)
    stalled_publishes = tuple(
        item
        for item in data.active_publishes
        if _float(value(item, "elapsed", 0.0)) >= ACTIVE_PUBLISH_STALL_SECONDS
    )

    recent_publish_timings = tuple(
        item for item in data.publish_timings if _timestamp(item) >= cutoff
    )
    event_durations_ms = [
        _float(value(item, "duration", 0.0)) * 1000.0
        for item in recent_publish_timings
    ]
    ui_lags_ms = [_float(value(item, "lag", 0.0)) * 1000.0 for item in recent_ui_lags]

    latest_callback_error = _latest(data.callback_errors)
    latest_runtime_error = _latest(data.runtime_errors)
    latest_error = _latest((item for item in (latest_callback_error, latest_runtime_error) if item))
    latest_lag = _latest(data.ui_lags)
    latest_slow = _latest(data.slow_callbacks)

    if latest_error is not None and _timestamp(latest_error) >= header_cutoff:
        label = (
            value(latest_error, "owner_label")
            or value(latest_error, "source")
            or value(latest_error, "topic")
            or "runtime"
        )
        status, tone = "Error", "danger"
        headline = f"Error · {label}"
        detail = str(
            value(latest_error, "message")
            or value(latest_error, "error")
            or "A recent runtime operation failed."
        )
    elif stalled_publishes:
        worst = max(stalled_publishes, key=lambda item: _float(value(item, "elapsed", 0.0)))
        status, tone = "Blocked", "danger"
        headline = f"Event blocked · {value(worst, 'topic', 'unknown topic')}"
        detail = f"Synchronous delivery has been active for {format_seconds(value(worst, 'elapsed', 0.0))}."
    elif latest_lag is not None and _timestamp(latest_lag) >= header_cutoff:
        lag_ms = _float(value(latest_lag, "lag", 0.0)) * 1000.0
        status, tone = "UI lag", "warning"
        headline = f"UI lag · {format_duration_ms(lag_ms)}"
        detail = "The session heartbeat was delayed by work on the UI thread."
    elif latest_slow is not None and _timestamp(latest_slow) >= header_cutoff:
        label = (
            value(latest_slow, "owner_label")
            or value(latest_slow, "callback_name")
            or value(latest_slow, "module")
            or "callback"
        )
        duration_ms = _float(value(latest_slow, "duration", 0.0)) * 1000.0
        status, tone = "Slow", "warning"
        headline = f"Slow callback · {label}"
        detail = f"Last synchronous callback took {format_duration_ms(duration_ms)}."
    elif data.active_jobs or data.active_publishes:
        status, tone = "Working", "info"
        parts = []
        if data.active_jobs:
            parts.append(f"{len(data.active_jobs)} background job{'s' if len(data.active_jobs) != 1 else ''}")
        if data.active_publishes:
            parts.append(f"{len(data.active_publishes)} active publish{'es' if len(data.active_publishes) != 1 else ''}")
        headline = "Working · " + " · ".join(parts)
        detail = "Work is active; no recent heartbeat delay was detected."
    elif percentile(event_durations_ms, 95.0) >= SLOW_EVENT_P95_MS:
        status, tone = "Degraded", "warning"
        headline = f"Event p95 · {format_duration_ms(percentile(event_durations_ms, 95.0))}"
        detail = "Recent synchronous event delivery is slower than the platform target."
    else:
        status, tone = "Responsive", "success"
        headline = "UI responsive"
        detail = "No recent heartbeat delay or synchronous callback bottleneck."

    return RuntimeHealthSummary(
        status=status,
        tone=tone,
        headline=headline,
        detail=detail,
        active_job_count=len(data.active_jobs),
        active_publish_count=len(data.active_publishes),
        stalled_publish_count=len(stalled_publishes),
        recent_slow_count=len(recent_slow),
        recent_callback_error_count=len(recent_callback_errors),
        recent_runtime_error_count=len(recent_runtime_errors),
        recent_ui_lag_count=len(recent_ui_lags),
        recent_note_count=len(recent_notes),
        event_p95_ms=percentile(event_durations_ms, 95.0),
        event_max_ms=max(event_durations_ms, default=0.0),
        latest_ui_lag_ms=(
            _float(value(latest_lag, "lag", 0.0)) * 1000.0 if latest_lag else 0.0
        ),
        max_ui_lag_ms=max(ui_lags_ms, default=0.0),
        subscriber_count=len(data.subscriptions),
    )


def percentile(values: Sequence[float], percent: float) -> float:
    clean = sorted(float(item) for item in values if math.isfinite(float(item)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]

    rank = (max(0.0, min(100.0, float(percent))) / 100.0) * (len(clean) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return clean[lower]
    weight = rank - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def format_duration_ms(milliseconds: float) -> str:
    value_ms = max(0.0, _float(milliseconds))
    if value_ms >= 1000.0:
        return f"{value_ms / 1000.0:.2f}s"
    if value_ms >= 10.0:
        return f"{value_ms:.1f}ms"
    return f"{value_ms:.2f}ms"


def format_seconds(seconds: Any) -> str:
    return format_duration_ms(_float(seconds) * 1000.0)


def format_age(timestamp: Any, *, now: float | None = None) -> str:
    stamp = _float(timestamp)
    if stamp <= 0.0:
        return "—"
    if now is None:
        now = time.time()
    age = max(0.0, float(now) - stamp)
    if age < 1.0:
        return "now"
    if age < 60.0:
        return f"{age:.0f}s ago"
    if age < 3600.0:
        return f"{age / 60.0:.0f}m ago"
    return f"{age / 3600.0:.1f}h ago"


def format_clock(timestamp: Any) -> str:
    stamp = _float(timestamp)
    if stamp <= 0.0:
        return "—"
    return time.strftime("%H:%M:%S", time.localtime(stamp))


def _call(target: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    if target is None:
        return None
    method = getattr(target, method_name, None)
    if not callable(method):
        return None
    try:
        return method(*args, **kwargs)
    except Exception:
        return None


def _call_list(target: Any, method_name: str, *args: Any) -> Iterable[Any]:
    result = _call(target, method_name, *args)
    return () if result is None else result


def _timestamp(record: Any) -> float:
    return _float(value(record, "timestamp", 0.0))


def _float(raw: Any) -> float:
    try:
        result = float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _latest(records: Iterable[Any]) -> Any | None:
    available = [item for item in records if item is not None]
    if not available:
        return None
    return max(available, key=_timestamp)
