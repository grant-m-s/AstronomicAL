from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

MAX_HISTORY_LIMIT = 5000
ACTIVE_STALL_THRESHOLD_SECONDS = 1.0
DEFAULT_SUBSCRIPTION_ROW_LIMIT = 250

KNOWN_TOPIC_PREFIXES = (
    "dataset.",
    "selection.",
    "artifact.",
    "workflow.",
    "plugin.",
    "job.",
    "workspace.",
)

TOPIC_GROUPS: Mapping[str, str | None] = {
    "All topics": None,
    "Datasets": "dataset.",
    "Selections": "selection.",
    "Artifacts": "artifact.",
    "Workflows": "workflow.",
    "Plugins": "plugin.",
    "Jobs": "job.",
    "Workspace": "workspace.",
    "Other": "__other__",
}


@dataclass(frozen=True)
class EventMonitorSnapshot:
    captured_at: float
    subscriptions: tuple[Any, ...]
    active_publishes: tuple[Any, ...]
    slow_callbacks: tuple[Any, ...]
    publish_timings: tuple[Any, ...]
    callback_errors: tuple[Any, ...]


@dataclass(frozen=True)
class EventHealthSummary:
    status: str
    status_tone: str
    publishes_per_minute: int
    p95_publish_ms: float
    subscriber_count: int
    issue_count: int
    active_count: int
    stalled_count: int
    recent_error_count: int
    recent_slow_count: int


def value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def collect_snapshot(event_bus: Any, *, limit: int = 250) -> EventMonitorSnapshot:
    """Read the public EventBus diagnostics API without mutating bus state.

    A platform EventBus may expose ``diagnostic_snapshot(limit=...)`` to return
    all diagnostic collections atomically. The existing public inspection
    methods remain supported for compatibility with the current platform.
    """
    safe_limit = max(1, min(int(limit), MAX_HISTORY_LIMIT))
    snapshot_method = getattr(event_bus, "diagnostic_snapshot", None)
    if callable(snapshot_method):
        return _coerce_snapshot(snapshot_method(limit=safe_limit))

    return EventMonitorSnapshot(
        captured_at=time.time(),
        subscriptions=tuple(_call_list(event_bus, "list_subscriptions")),
        active_publishes=tuple(_call_list(event_bus, "active_publishes")),
        slow_callbacks=tuple(
            _call_list(event_bus, "recent_slow_callbacks", safe_limit)
        ),
        publish_timings=tuple(
            _call_list(event_bus, "recent_publish_timings", safe_limit)
        ),
        callback_errors=tuple(
            _call_list(event_bus, "recent_callback_errors", safe_limit)
        ),
    )


def build_health_summary(
    snapshot: EventMonitorSnapshot,
    *,
    recent_window_seconds: float = 300.0,
) -> EventHealthSummary:
    cutoff = snapshot.captured_at - max(1.0, float(recent_window_seconds))
    minute_cutoff = snapshot.captured_at - 60.0

    recent_errors = [
        item
        for item in snapshot.callback_errors
        if _timestamp(item) >= cutoff
    ]
    recent_slow = [
        item
        for item in snapshot.slow_callbacks
        if _timestamp(item) >= cutoff
    ]
    stalled = [
        item
        for item in snapshot.active_publishes
        if float(value(item, "elapsed", 0.0) or 0.0)
        >= ACTIVE_STALL_THRESHOLD_SECONDS
    ]
    publishes_last_minute = sum(
        1 for item in snapshot.publish_timings if _timestamp(item) >= minute_cutoff
    )
    p95_ms = percentile(
        [
            float(value(item, "duration", 0.0) or 0.0) * 1000.0
            for item in snapshot.publish_timings
        ],
        95.0,
    )

    if recent_errors or stalled:
        status, tone = "Action needed", "danger"
    elif recent_slow:
        status, tone = "Degraded", "warning"
    else:
        status, tone = "Healthy", "success"

    issue_count = len(recent_errors) + len(recent_slow) + len(stalled)
    return EventHealthSummary(
        status=status,
        status_tone=tone,
        publishes_per_minute=publishes_last_minute,
        p95_publish_ms=p95_ms,
        subscriber_count=len(snapshot.subscriptions),
        issue_count=issue_count,
        active_count=len(snapshot.active_publishes),
        stalled_count=len(stalled),
        recent_error_count=len(recent_errors),
        recent_slow_count=len(recent_slow),
    )


def build_activity_rows(
    snapshot: EventMonitorSnapshot,
    *,
    topic_group: str = "All topics",
    search: str = "",
) -> list[dict[str, Any]]:
    error_ids = {
        value(item, "publish_id") for item in snapshot.callback_errors
    }
    slow_ids = {
        value(item, "publish_id") for item in snapshot.slow_callbacks
    }
    rows: list[dict[str, Any]] = []

    for item in reversed(snapshot.publish_timings):
        topic = str(value(item, "topic", "") or "")
        summary = str(value(item, "payload_summary", "") or "")
        if not topic_matches(topic, topic_group):
            continue
        if not text_matches(search, topic, summary):
            continue

        publish_id = value(item, "publish_id")
        if publish_id in error_ids:
            status = "Error"
        elif publish_id in slow_ids:
            status = "Slow"
        else:
            status = "OK"

        rows.append(
            {
                "time": format_clock(_timestamp(item)),
                "topic": topic,
                "duration_ms": round(
                    float(value(item, "duration", 0.0) or 0.0) * 1000.0,
                    2,
                ),
                "callbacks": int(value(item, "callback_count", 0) or 0),
                "status": status,
                "summary": summary,
                "publish_id": publish_id,
            }
        )
    return rows


def build_subscription_rows(
    snapshot: EventMonitorSnapshot,
    *,
    topic_group: str = "All topics",
    search: str = "",
    limit: int = DEFAULT_SUBSCRIPTION_ROW_LIMIT,
) -> list[dict[str, Any]]:
    safe_limit = max(1, int(limit))
    rows: list[dict[str, Any]] = []
    for item in snapshot.subscriptions:
        topic = str(value(item, "topic", "") or "")
        owner = str(
            value(item, "owner_label")
            or value(item, "owner_id")
            or "Unidentified subscriber"
        )
        callback = str(value(item, "callback_name", "") or "")
        module = str(value(item, "module", "") or "")
        if not topic_matches(topic, topic_group):
            continue
        if not text_matches(search, topic, owner, callback, module):
            continue
        rows.append(
            {
                "owner": owner,
                "kind": str(value(item, "owner_kind", "subscriber") or "subscriber"),
                "topic": topic,
                "callback": callback,
                "module": module,
            }
        )

    rows.sort(
        key=lambda row: (
            row["topic"].lower(),
            row["owner"].lower(),
            row["callback"].lower(),
        )
    )
    return rows[:safe_limit]


def build_topic_rows(
    snapshot: EventMonitorSnapshot,
    *,
    topic_group: str = "All topics",
    search: str = "",
    limit: int = 8,
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    durations: dict[str, list[float]] = defaultdict(list)

    for item in snapshot.publish_timings:
        topic = str(value(item, "topic", "") or "")
        if not topic_matches(topic, topic_group):
            continue
        if not text_matches(search, topic):
            continue
        counts[topic] += 1
        durations[topic].append(
            float(value(item, "duration", 0.0) or 0.0) * 1000.0
        )

    return [
        {
            "topic": topic,
            "count": count,
            "average_ms": round(sum(durations[topic]) / max(1, count), 2),
            "max_ms": round(max(durations[topic]), 2),
        }
        for topic, count in counts.most_common(max(1, int(limit)))
    ]


def topic_matches(topic: str, group: str) -> bool:
    prefix = TOPIC_GROUPS.get(group)
    if prefix is None:
        return True
    if prefix == "__other__":
        return topic != "*" and not topic.startswith(KNOWN_TOPIC_PREFIXES)
    return topic.startswith(prefix)


def text_matches(search: str, *values: Any) -> bool:
    query = str(search or "").strip().lower()
    if not query:
        return True
    return any(query in str(item or "").lower() for item in values)


def percentile(values: Sequence[float], percent: float) -> float:
    clean = sorted(float(item) for item in values if math.isfinite(float(item)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]

    rank = (max(0.0, min(100.0, percent)) / 100.0) * (len(clean) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return clean[lower]
    weight = rank - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def format_clock(timestamp: float) -> str:
    if timestamp <= 0:
        return "—"
    return time.strftime("%H:%M:%S", time.localtime(timestamp))


def _coerce_snapshot(snapshot: Any) -> EventMonitorSnapshot:
    return EventMonitorSnapshot(
        captured_at=float(value(snapshot, "captured_at", time.time()) or time.time()),
        subscriptions=tuple(value(snapshot, "subscriptions", ()) or ()),
        active_publishes=tuple(value(snapshot, "active_publishes", ()) or ()),
        slow_callbacks=tuple(value(snapshot, "slow_callbacks", ()) or ()),
        publish_timings=tuple(value(snapshot, "publish_timings", ()) or ()),
        callback_errors=tuple(value(snapshot, "callback_errors", ()) or ()),
    )


def _timestamp(record: Any) -> float:
    try:
        return float(value(record, "timestamp", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _call_list(target: Any, method_name: str, *args: Any) -> Iterable[Any]:
    method = getattr(target, method_name, None)
    if not callable(method):
        return ()
    result = method(*args)
    return () if result is None else result