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


class EventBus:
    """
    Minimal pub/sub event bus with optional tracing.

    - Topics are strings ("selection.changed", "dataset.loaded", etc.)
    - Payloads can be any object (dict recommended).
    - Thread-safe subscribe/publish.
    - Trace buffer stores recent published events for debugging/monitoring.
    """

    def __init__(self, *, trace: bool = False, trace_limit: int = 2000) -> None:
        self._lock = threading.RLock()
        self._subs: Dict[str, List[tuple[str, EventCallback]]] = {}

        # NEW: tracing
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

    # --------------------
    # Pub/Sub
    # --------------------
    def subscribe(self, topic: str, callback: EventCallback) -> Subscription:
        """
        Subscribe to a topic.

        Special topic:
        - "*" subscribes to all events (wildcard).
        """
        sub_id = uuid.uuid4().hex
        with self._lock:
            self._subs.setdefault(topic, []).append((sub_id, callback))
        return Subscription(id=sub_id, topic=topic)

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            lst = self._subs.get(sub.topic, [])
            self._subs[sub.topic] = [(sid, cb) for (sid, cb) in lst if sid != sub.id]
            if not self._subs[sub.topic]:
                self._subs.pop(sub.topic, None)

    def publish(self, topic: str, payload: Any = None) -> None:
        """
        Publish synchronously to subscribers. If a subscriber raises, log it and continue.
        Also delivers to wildcard '*' subscribers.
        """
        if self._trace_enabled:
            with self._lock:
                self._trace_buf.append((time.time(), topic, payload))

        with self._lock:
            callbacks = list(self._subs.get(topic, []))
            wildcard_callbacks = list(self._subs.get("*", []))

        for _sid, cb in callbacks + wildcard_callbacks:
            try:
                cb(topic, payload)
            except Exception:
                traceback.print_exc()

    def clear(self) -> None:
        with self._lock:
            self._subs.clear()
            self._trace_buf.clear()