from __future__ import annotations

from dataclasses import replace
from threading import RLock
from typing import Any, Optional

from .contracts import VideoTemporalFocus

TEMPORAL_CHANGED_TOPIC = "video.temporal.changed"
TEMPORAL_CLEARED_TOPIC = "video.temporal.cleared"


class VideoTemporalStateManager:
    """Own plugin-level temporal focus while platform selection owns row identity.

    The platform SelectionManager remains authoritative for ``dataset_id`` +
    ``row_id``. This service only owns state *inside* the focused video, such as
    frame index and playback state. The event payload is deliberately tiny.
    """

    service_key = "core.video.temporal_state"

    def __init__(self, context: Any) -> None:
        self.context = context
        self._lock = RLock()
        self._state = VideoTemporalFocus()
        self._subscriptions: list[Any] = []
        self._disposed = False
        self._subscribe()
        self._sync_from_platform_focus(publish=False)

    def get_state(self) -> VideoTemporalFocus:
        with self._lock:
            return replace(self._state)

    def set_frame(
        self,
        dataset_id: str,
        row_id: str,
        frame_index: int,
        *,
        fps: Optional[float] = None,
        frame_count: Optional[int] = None,
        origin: Optional[str] = None,
        panel_id: Optional[str] = None,
        publish: bool = True,
    ) -> VideoTemporalFocus:
        safe_frame_count = _positive_int(frame_count)
        safe_index = max(0, int(frame_index or 0))
        if safe_frame_count is not None:
            safe_index = min(safe_index, max(0, safe_frame_count - 1))

        safe_fps = _positive_float(fps)
        with self._lock:
            current = self._state
            same_row = (
                current.dataset_id == str(dataset_id)
                and current.row_id == str(row_id)
            )
            if safe_fps is None and same_row:
                safe_fps = current.fps
            if safe_frame_count is None and same_row:
                safe_frame_count = current.frame_count
            time_seconds = (
                float(safe_index) / float(safe_fps)
                if safe_fps is not None and safe_fps > 0
                else None
            )
            candidate = VideoTemporalFocus(
                dataset_id=str(dataset_id),
                row_id=str(row_id),
                frame_index=safe_index,
                time_seconds=time_seconds,
                playing=current.playing if same_row else False,
                playback_rate=current.playback_rate,
                fps=safe_fps,
                frame_count=safe_frame_count,
                origin=origin,
                panel_id=panel_id,
            )
            changed = _meaningful_state(candidate) != _meaningful_state(current)
            self._state = candidate
            state = replace(self._state)

        # `video.temporal.changed` is a state-change notification, not a render
        # pulse. Suppressing no-op publishes is important because multiple video
        # panels may subscribe and independently request bounded media work.
        if publish and changed:
            self._publish(TEMPORAL_CHANGED_TOPIC, state)
        return state

    def set_playing(
        self,
        playing: bool,
        *,
        playback_rate: Optional[float] = None,
        origin: Optional[str] = None,
        panel_id: Optional[str] = None,
        publish: bool = True,
    ) -> VideoTemporalFocus:
        with self._lock:
            current = self._state
            rate = _positive_float(playback_rate)
            if rate is None:
                rate = current.playback_rate or 1.0
            candidate = replace(
                current,
                playing=bool(playing),
                playback_rate=float(rate),
                origin=origin,
                panel_id=panel_id,
            )
            changed = _meaningful_state(candidate) != _meaningful_state(current)
            self._state = candidate
            state = replace(self._state)
        if publish and changed:
            self._publish(TEMPORAL_CHANGED_TOPIC, state)
        return state

    def advance(
        self,
        delta_frames: int = 1,
        *,
        loop: bool = False,
        origin: Optional[str] = None,
        panel_id: Optional[str] = None,
    ) -> VideoTemporalFocus:
        current = self.get_state()
        if current.dataset_id is None or current.row_id is None:
            return current

        target = max(0, int(current.frame_index) + int(delta_frames))
        if current.frame_count is not None and current.frame_count > 0:
            if loop:
                target %= int(current.frame_count)
            else:
                target = min(target, int(current.frame_count) - 1)

        state = self.set_frame(
            current.dataset_id,
            current.row_id,
            target,
            fps=current.fps,
            frame_count=current.frame_count,
            origin=origin,
            panel_id=panel_id,
            publish=True,
        )
        if (
            not loop
            and current.frame_count is not None
            and current.frame_count > 0
            and state.frame_index >= current.frame_count - 1
            and current.playing
        ):
            state = self.set_playing(
                False,
                origin=origin,
                panel_id=panel_id,
                publish=True,
            )
        return state

    def clear(
        self,
        *,
        origin: Optional[str] = None,
        panel_id: Optional[str] = None,
        publish: bool = True,
    ) -> VideoTemporalFocus:
        with self._lock:
            previous = self._state
            self._state = VideoTemporalFocus(origin=origin, panel_id=panel_id)
            state = replace(self._state)
        if publish:
            events = getattr(self.context, "events", None)
            if events is not None:
                events.publish(
                    TEMPORAL_CLEARED_TOPIC,
                    {
                        "dataset_id": previous.dataset_id,
                        "row_id": previous.row_id,
                        "origin": origin,
                        "panel_id": panel_id,
                    },
                )
        return state

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in self._subscriptions:
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions.clear()

    def _subscribe(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        self._subscriptions.append(
            events.subscribe(
                "selection.focus.changed",
                self._on_platform_focus_changed,
                owner_id="core.video.temporal_state",
                owner_label="Video temporal state",
                owner_kind="plugin_service",
            )
        )
        self._subscriptions.append(
            events.subscribe(
                "selection.focus.cleared",
                self._on_platform_focus_cleared,
                owner_id="core.video.temporal_state",
                owner_label="Video temporal state",
                owner_kind="plugin_service",
            )
        )

    def _sync_from_platform_focus(self, *, publish: bool) -> None:
        selection = getattr(self.context, "selection", None)
        get_focus = getattr(selection, "get_focus", None)
        if not callable(get_focus):
            return
        try:
            focus = get_focus()
        except Exception:
            return
        dataset_id = getattr(focus, "dataset_id", None)
        row_id = getattr(focus, "row_id", None)
        if dataset_id is None or row_id is None:
            return
        self.set_frame(
            str(dataset_id),
            str(row_id),
            0,
            origin="core.video.temporal_state",
            publish=publish,
        )

    def _on_platform_focus_changed(self, _topic: str, payload: Any) -> None:
        if self._disposed:
            return
        payload = payload if isinstance(payload, dict) else {}
        dataset_id = payload.get("dataset_id")
        row_id = payload.get("row_id")
        if dataset_id is None or row_id is None:
            return
        current = self.get_state()
        if current.dataset_id == str(dataset_id) and current.row_id == str(row_id):
            return
        self.set_frame(
            str(dataset_id),
            str(row_id),
            0,
            origin="selection.focus.changed",
            publish=True,
        )

    def _on_platform_focus_cleared(self, _topic: str, payload: Any) -> None:
        if self._disposed:
            return
        self.clear(origin="selection.focus.cleared", publish=True)

    def _publish(self, topic: str, state: VideoTemporalFocus) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        events.publish(topic, state.to_dict())


def _meaningful_state(state: VideoTemporalFocus) -> tuple[Any, ...]:
    """Fields that define temporal state independently of event provenance."""
    return (
        state.dataset_id,
        state.row_id,
        int(state.frame_index),
        state.time_seconds,
        bool(state.playing),
        float(state.playback_rate or 1.0),
        state.fps,
        state.frame_count,
    )


def get_video_temporal_state(context: Any) -> VideoTemporalStateManager:
    services = getattr(context, "services", None)
    if services is not None:
        try:
            state = services.get(VideoTemporalStateManager.service_key)
        except Exception:
            state = None
        if state is not None:
            return state
    return VideoTemporalStateManager(context)


def _positive_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except Exception:
        return None
    if result <= 0:
        return None
    return result


def _positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        result = int(value)
    except Exception:
        return None
    if result <= 0:
        return None
    return result
