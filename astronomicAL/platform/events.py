import threading
import uuid
import traceback
import time
from dataclasses import dataclass
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Deque, Tuple


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


class EventBus:
    """
    Minimal pub/sub event bus with optional tracing.

    - Topics are strings ("selection.changed", "dataset.loaded", etc.)
    - Payloads can be any object (dict recommended).
    - Thread-safe subscribe/publish.
    - Trace buffer stores recent published events for debugging/monitoring.
    - Subscriptions carry lightweight ownership metadata for introspection.
    """

    def __init__(self, *, trace: bool = False, trace_limit: int = 2000) -> None:
        self._lock = threading.RLock()
        self._subs: Dict[str, List[tuple[str, EventCallback, Dict[str, Any]]]] = {}

        self._trace_enabled: bool = trace
        self._trace_buf: Deque[Tuple[float, str, Any]] = deque(maxlen=trace_limit)

    # --------------------
    # Tracing / inspection
    # --------------------
    def enable_trace(self, enabled: bool = True) -> None:
        self._trace_enabled = enabled

    def recent_events(self, n: int = 200) -> List[Tuple[float, str, Any]]:
        """Return last n traced publish events: (timestamp, topic, payload)."""
        with self._lock:
            if n <= 0:
                return []
            return list(self._trace_buf)[-n:]

    def subscribers(self) -> Dict[str, int]:
        """Return subscriber count per topic."""
        with self._lock:
            return {topic: len(lst) for topic, lst in self._subs.items()}

    def list_subscriptions(self) -> List[SubscriptionInfo]:
        """
        Return a flattened list of subscription metadata for debugging/monitoring.
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

    # --------------------
    # Metadata helpers
    # --------------------
    @staticmethod
    def _normalise_meta(callback: EventCallback, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fill in any missing ownership metadata from a bound callback method.
        """
        meta = dict(meta or {})

        callback_name = getattr(callback, "__qualname__", None) or getattr(callback, "__name__", None) or repr(callback)
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
        - "*" subscribes to all events (wildcard).
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
                (sid, cb, meta)
                for (sid, cb, meta) in lst
                if sid != sub.id
            ]
            if not self._subs[sub.topic]:
                self._subs.pop(sub.topic, None)

    def publish(self, topic: str, payload: Any = None) -> None:
        """Publish synchronously to subscribers.

        If a subscriber raises, log it and continue.
        Also delivers to wildcard '*' subscribers.
        """
        publish_start = time.perf_counter()

        if self._trace_enabled:
            with self._lock:
                self._trace_buf.append((time.time(), topic, payload))

        with self._lock:
            callbacks = list(self._subs.get(topic, []))
            wildcard_callbacks = list(self._subs.get("*", []))

        for _sid, cb, meta in callbacks + wildcard_callbacks:
            info = self._normalise_meta(cb, meta)

            cb_start = time.perf_counter()
            try:
                cb(topic, payload)
            except Exception:
                traceback.print_exc()
            finally:
                duration = time.perf_counter() - cb_start

                if duration >= 0.05:
                    print(
                        "[AstronomicAL events] slow subscriber "
                        f"topic={topic!r} "
                        f"duration={duration:.3f}s "
                        f"owner={info.get('owner_label')!r} "
                        f"kind={info.get('owner_kind')!r} "
                        f"callback={info.get('callback_name')!r} "
                        f"module={info.get('module')!r}",
                        flush=True,
                    )

        total = time.perf_counter() - publish_start

        if total >= 0.05:
            print(
                "[AstronomicAL events] publish complete "
                f"topic={topic!r} "
                f"subscribers={len(callbacks)} "
                f"wildcards={len(wildcard_callbacks)} "
                f"duration={total:.3f}s",
                flush=True,
            )

    def clear(self) -> None:
        with self._lock:
            self._subs.clear()
            self._trace_buf.clear()