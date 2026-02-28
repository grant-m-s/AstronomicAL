# astronomicAL/platform/events.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import threading
import uuid
import traceback


EventCallback = Callable[[str, Any], None]


@dataclass(frozen=True)
class Subscription:
    """Opaque handle used to unsubscribe."""
    id: str
    topic: str


class EventBus:
    """
    Minimal pub/sub event bus.

    - Topics are strings ("selection.changed", "dataset.loaded", etc.)
    - Payloads can be any object (dict recommended).
    - Thread-safe subscribe/publish.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: Dict[str, List[tuple[str, EventCallback]]] = {}

    def subscribe(self, topic: str, callback: EventCallback) -> Subscription:
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
        Publish synchronously to subscribers. If a subscriber raises, we log it
        (to stderr) and continue.
        """
        with self._lock:
            callbacks = list(self._subs.get(topic, []))

        for _sid, cb in callbacks:
            try:
                cb(topic, payload)
            except Exception:
                # Avoid crashing the app due to a single bad handler.
                traceback.print_exc()

    def clear(self) -> None:
        with self._lock:
            self._subs.clear()