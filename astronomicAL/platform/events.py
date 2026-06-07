from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


EventCallback = Callable[[str, Any], None]


@dataclass(frozen=True)
class Subscription:
    id: str
    topic: str


@dataclass(frozen=True)
class SubscriptionInfo:
    id: str
    topic: str
    owner_id: Optional[str] = None
    owner_label: Optional[str] = None
    owner_kind: Optional[str] = None
    callback_name: Optional[str] = None
    module: Optional[str] = None


@dataclass(frozen=True)
class SlowCallbackRecord:
    timestamp: float
    publish_id: int
    topic: str
    duration: float
    owner_id: Optional[str]
    owner_label: Optional[str]
    owner_kind: Optional[str]
    callback_name: Optional[str]
    module: Optional[str]
    payload_summary: str


@dataclass(frozen=True)
class PublishTimingRecord:
    timestamp: float
    publish_id: int
    topic: str
    subscriber_count: int
    wildcard_count: int
    callback_count: int
    duration: float
    payload_summary: str


@dataclass(frozen=True)
class ActivePublishSnapshot:
    timestamp: float
    publish_id: int
    topic: str
    started_at: float
    elapsed: float
    callback_count: int
    payload_summary: str


@dataclass(frozen=True)
class CallbackErrorRecord:
    timestamp: float
    publish_id: int
    topic: str
    owner_id: Optional[str]
    owner_label: Optional[str]
    owner_kind: Optional[str]
    callback_name: Optional[str]
    module: Optional[str]
    error: str
    traceback_text: str
    payload_summary: str


@dataclass
class _ActivePublish:
    timestamp: float
    publish_id: int
    topic: str
    started_perf: float
    started_at: float
    callback_count: int
    payload_summary: str


class EventBus:
    """
    Minimal pub/sub event bus with optional tracing and lightweight diagnostics.

    - Topics are strings: "selection.changed", "dataset.loaded", etc.
    - Payloads can be any object, though dict payloads are recommended.
    - Subscriptions can carry ownership metadata for debugging.
    - publish() is synchronous.
    - Slow callbacks are stored in a diagnostics buffer and still printed.
    """

    def __init__(
        self,
        *,
        trace: bool = False,
        trace_limit: int = 2000,
        diagnostics_limit: int = 1000,
    ) -> None:
        self._lock = threading.RLock()
        self._subs: Dict[str, List[tuple[str, EventCallback, Dict[str, Any]]]] = {}

        self._trace_enabled: bool = trace
        self._trace_buf: Deque[Tuple[float, str, Any]] = deque(maxlen=trace_limit)

        self._publish_seq: int = 0
        self._active_publishes: Dict[int, _ActivePublish] = {}
        self._slow_callbacks: Deque[SlowCallbackRecord] = deque(
            maxlen=diagnostics_limit
        )
        self._publish_timings: Deque[PublishTimingRecord] = deque(
            maxlen=diagnostics_limit
        )
        self._callback_errors: Deque[CallbackErrorRecord] = deque(
            maxlen=diagnostics_limit
        )

    # --------------------
    # Tracing / inspection
    # --------------------

    def enable_trace(self, enabled: bool = True) -> None:
        self._trace_enabled = enabled

    def recent_events(self, n: int = 200) -> List[Tuple[float, str, Any]]:
        """
        Return last n traced publish events: (timestamp, topic, payload).
        """
        with self._lock:
            if n <= 0:
                return []
            return list(self._trace_buf)[-n:]

    def subscribers(self) -> Dict[str, int]:
        """
        Return subscriber count per topic.
        """
        with self._lock:
            return {topic: len(lst) for topic, lst in self._subs.items()}

    def list_subscriptions(self) -> List[SubscriptionInfo]:
        """
        Return flattened subscription metadata for debugging/monitoring.
        """
        out: List[SubscriptionInfo] = []

        with self._lock:
            for topic, entries in self._subs.items():
                for sub_id, callback, meta in entries:
                    info = self._normalise_meta(callback, meta)
                    out.append(
                        SubscriptionInfo(
                            id=sub_id,
                            topic=topic,
                            owner_id=info.get("owner_id"),
                            owner_label=info.get("owner_label"),
                            owner_kind=info.get("owner_kind"),
                            callback_name=info.get("callback_name"),
                            module=info.get("module"),
                        )
                    )

        out.sort(
            key=lambda x: (
                x.owner_kind or "",
                x.owner_label or "",
                x.topic or "",
                x.callback_name or "",
                x.id,
            )
        )
        return out

    def active_publishes(self) -> List[ActivePublishSnapshot]:
        """
        Return event publishes currently executing.

        If the UI thread is blocked by a synchronous callback, this method cannot
        repaint the UI during the blockage, but the snapshot is useful for
        polling/status views and post-stall diagnostics.
        """
        now_perf = time.perf_counter()

        with self._lock:
            snapshots = [
                ActivePublishSnapshot(
                    timestamp=active.timestamp,
                    publish_id=active.publish_id,
                    topic=active.topic,
                    started_at=active.started_at,
                    elapsed=max(0.0, now_perf - active.started_perf),
                    callback_count=active.callback_count,
                    payload_summary=active.payload_summary,
                )
                for active in self._active_publishes.values()
            ]

        snapshots.sort(key=lambda item: item.started_at)
        return snapshots

    def recent_slow_callbacks(self, n: int = 50) -> List[SlowCallbackRecord]:
        with self._lock:
            if n <= 0:
                return []
            return list(self._slow_callbacks)[-n:]

    def recent_publish_timings(self, n: int = 100) -> List[PublishTimingRecord]:
        with self._lock:
            if n <= 0:
                return []
            return list(self._publish_timings)[-n:]

    def recent_callback_errors(self, n: int = 50) -> List[CallbackErrorRecord]:
        with self._lock:
            if n <= 0:
                return []
            return list(self._callback_errors)[-n:]

    def clear_diagnostics(self) -> None:
        with self._lock:
            self._active_publishes.clear()
            self._slow_callbacks.clear()
            self._publish_timings.clear()
            self._callback_errors.clear()

    # --------------------
    # Metadata helpers
    # --------------------

    @staticmethod
    def _normalise_meta(
        callback: EventCallback,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fill in missing ownership metadata from a bound callback method.
        """
        meta = dict(meta or {})

        callback_name = (
            getattr(callback, "__qualname__", None)
            or getattr(callback, "__name__", None)
            or repr(callback)
        )
        module = getattr(callback, "__module__", None)

        meta.setdefault("callback_name", callback_name)
        meta.setdefault("module", module)

        owner = getattr(callback, "__self__", None)
        if owner is not None:
            if not meta.get("owner_id"):
                meta["owner_id"] = (
                    getattr(owner, "panel_id", None)
                    or getattr(owner, "name", None)
                    or getattr(owner, "title", None)
                    or f"{owner.__class__.__name__}:{id(owner)}"
                )

            if not meta.get("owner_label"):
                meta["owner_label"] = (
                    getattr(owner, "panel_name", None)
                    or getattr(owner, "title", None)
                    or getattr(owner, "name", None)
                    or owner.__class__.__name__
                )

            if not meta.get("owner_kind"):
                cls_name = owner.__class__.__name__.lower()
                if "dashboard" in cls_name:
                    meta["owner_kind"] = "dashboard"
                elif hasattr(owner, "panel_id") or hasattr(owner, "panel_name"):
                    meta["owner_kind"] = "panel"
                else:
                    meta["owner_kind"] = "subscriber"

        meta.setdefault("owner_id", None)
        meta.setdefault("owner_label", None)
        meta.setdefault("owner_kind", "subscriber")

        return meta

    @staticmethod
    def _payload_summary(payload: Any) -> str:
        """
        Summarise payloads without dumping huge selected ID lists.
        """
        if not isinstance(payload, dict):
            return type(payload).__name__

        parts: list[str] = []

        for key in (
            "dataset_id",
            "selection_id",
            "selection_set_id",
            "row_id",
            "focused_id",
            "origin_panel_id",
            "origin",
        ):
            if key in payload:
                parts.append(f"{key}={payload.get(key)!r}")

        for key in ("row_ids", "ids", "selected_ids"):
            value = payload.get(key)
            if isinstance(value, (list, tuple, set)):
                parts.append(f"{key}_len={len(value):,}")

        meta = payload.get("metadata")
        if isinstance(meta, dict):
            for key in (
                "total_matches",
                "published_ids",
                "truncated",
                "source",
                "geometry",
            ):
                if key in meta:
                    parts.append(f"metadata.{key}={meta.get(key)!r}")

        return " ".join(parts) if parts else f"dict_keys={sorted(payload.keys())!r}"

    @staticmethod
    def _slow_threshold_for_topic(topic: str) -> float:
        """
        Lower threshold for selection topics while debugging UI stalls.
        """
        if topic.startswith("selection."):
            return 0.01
        return 0.05

    # --------------------
    # Pub/Sub
    # --------------------

    def subscribe(
        self,
        topic: str,
        callback: EventCallback,
        *,
        owner_id: Optional[str] = None,
        owner_label: Optional[str] = None,
        owner_kind: Optional[str] = None,
    ) -> Subscription:
        """
        Subscribe to a topic.

        Special topic:
        - "*" subscribes to all events.
        """
        sub_id = uuid.uuid4().hex

        meta = self._normalise_meta(
            callback,
            {
                "owner_id": owner_id,
                "owner_label": owner_label,
                "owner_kind": owner_kind,
            },
        )

        with self._lock:
            self._subs.setdefault(topic, []).append((sub_id, callback, meta))

        return Subscription(id=sub_id, topic=topic)

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            lst = self._subs.get(sub.topic, [])
            self._subs[sub.topic] = [
                (sid, cb, meta) for (sid, cb, meta) in lst if sid != sub.id
            ]

            if not self._subs[sub.topic]:
                self._subs.pop(sub.topic, None)

    def publish(self, topic: str, payload: Any = None) -> None:
        """
        Publish synchronously to subscribers.

        If a subscriber raises, log it, store it in diagnostics, and continue.
        Also delivers to wildcard "*" subscribers.
        """
        publish_start = time.perf_counter()
        publish_wall_start = time.time()
        threshold = self._slow_threshold_for_topic(topic)

        with self._lock:
            self._publish_seq += 1
            publish_id = self._publish_seq

            if self._trace_enabled:
                self._trace_buf.append((time.time(), topic, payload))

            callbacks = list(self._subs.get(topic, []))
            wildcard_callbacks = list(self._subs.get("*", []))
            all_callbacks = callbacks + wildcard_callbacks

            payload_summary = self._payload_summary(payload)

            self._active_publishes[publish_id] = _ActivePublish(
                timestamp=publish_wall_start,
                publish_id=publish_id,
                topic=topic,
                started_perf=publish_start,
                started_at=publish_wall_start,
                callback_count=len(all_callbacks),
                payload_summary=payload_summary,
            )

        if topic.startswith("selection."):
            print(
                "[AstronomicAL events] publish start "
                f"id={publish_id} "
                f"topic={topic!r} "
                f"callbacks={len(all_callbacks)} "
                f"payload={payload_summary}",
                flush=True,
            )

        try:
            for _sid, cb, meta in all_callbacks:
                info = self._normalise_meta(cb, meta)
                cb_start = time.perf_counter()

                try:
                    cb(topic, payload)
                except Exception as exc:
                    tb_text = traceback.format_exc()
                    print(tb_text, flush=True)

                    error_record = CallbackErrorRecord(
                        timestamp=time.time(),
                        publish_id=publish_id,
                        topic=topic,
                        owner_id=info.get("owner_id"),
                        owner_label=info.get("owner_label"),
                        owner_kind=info.get("owner_kind"),
                        callback_name=info.get("callback_name"),
                        module=info.get("module"),
                        error=repr(exc),
                        traceback_text=tb_text,
                        payload_summary=payload_summary,
                    )

                    with self._lock:
                        self._callback_errors.append(error_record)

                finally:
                    duration = time.perf_counter() - cb_start

                    if duration >= threshold:
                        slow_record = SlowCallbackRecord(
                            timestamp=time.time(),
                            publish_id=publish_id,
                            topic=topic,
                            duration=duration,
                            owner_id=info.get("owner_id"),
                            owner_label=info.get("owner_label"),
                            owner_kind=info.get("owner_kind"),
                            callback_name=info.get("callback_name"),
                            module=info.get("module"),
                            payload_summary=payload_summary,
                        )

                        with self._lock:
                            self._slow_callbacks.append(slow_record)

                        print(
                            "[AstronomicAL events] slow subscriber "
                            f"id={publish_id} "
                            f"topic={topic!r} "
                            f"duration={duration:.3f}s "
                            f"owner={info.get('owner_label')!r} "
                            f"kind={info.get('owner_kind')!r} "
                            f"callback={info.get('callback_name')!r} "
                            f"module={info.get('module')!r}",
                            flush=True,
                        )

        finally:
            total = time.perf_counter() - publish_start

            timing = PublishTimingRecord(
                timestamp=time.time(),
                publish_id=publish_id,
                topic=topic,
                subscriber_count=len(callbacks),
                wildcard_count=len(wildcard_callbacks),
                callback_count=len(all_callbacks),
                duration=total,
                payload_summary=payload_summary,
            )

            with self._lock:
                self._active_publishes.pop(publish_id, None)
                self._publish_timings.append(timing)

            if total >= threshold:
                print(
                    "[AstronomicAL events] publish complete "
                    f"id={publish_id} "
                    f"topic={topic!r} "
                    f"subscribers={len(callbacks)} "
                    f"wildcards={len(wildcard_callbacks)} "
                    f"callbacks={len(all_callbacks)} "
                    f"duration={total:.3f}s "
                    f"payload={payload_summary}",
                    flush=True,
                )

    def clear(self) -> None:
        with self._lock:
            self._subs.clear()
            self._trace_buf.clear()
            self._active_publishes.clear()
            self._slow_callbacks.clear()
            self._publish_timings.clear()
            self._callback_errors.clear()