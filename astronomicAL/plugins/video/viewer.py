from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Optional, Mapping

import panel as pn
from astronomicAL.platform.plugins.specs import ActionRequest

from .contracts import VideoFramePreview, VideoMetadata
from .resolver import get_video_resolver
from .overlays import (
    OVERLAY_CATALOG_CHANGED_TOPIC,
    OVERLAY_SELECTION_CHANGED_TOPIC,
    get_video_overlay_registry,
)
from .surface import (
    BrowserFrameSequencePattern,
    VideoFrameSurface,
    browser_frame_sequence_pattern,
)
from .temporal import (
    TEMPORAL_CHANGED_TOPIC,
    TEMPORAL_CLEARED_TOPIC,
    get_video_temporal_state,
)

class VideoViewerPanel:
    """Selection-aware bounded video-frame viewer.

    The panel is a controller/view only. Canonical dataset-row focus stays in the
    platform SelectionManager and frame/time focus stays in the core.video
    temporal-state service. The panel owns only transient UI/job handles.
    """

    state_version = 4
    panel_id = "core.video.viewer"

    def __init__(
        self,
        context: Any,
        *,
        manager: Any = None,
        max_size: int = 1280,
        preview_fps: float = 12.0,
        prefetch_count: int = 6,
    ) -> None:
        self.context = context
        self.manager = manager or getattr(context, "plugins", None)
        self.max_size = max(128, int(max_size or 1280))
        # ``preview_fps`` is now only the managed/server fallback ceiling.
        # Browser-direct frame sequences use their mapped source FPS exactly.
        self.preview_fps = max(1.0, min(30.0, float(preview_fps or 12.0)))
        self.prefetch_count = max(0, min(24, int(prefetch_count or 6)))
        self.resolver = get_video_resolver(context)
        self.temporal = get_video_temporal_state(context)
        self.overlay_registry = get_video_overlay_registry(context)

        self._disposed = False
        self._subscriptions: list[Any] = []
        self._frame_job: Any = None
        self._probe_job: Any = None
        self._summary_job: Any = None
        self._summary_event_subscription: Any = None
        self._summary_event_topic = ""
        # Overlay preparation uses two bounded lanes: an exact-frame foreground lane
        # for first paint/seeks and a single background producer lane that extends a
        # resident frame cache.  This mirrors the former Segmentation Viewer cache and
        # prevents browser playback from outrunning one large replacement window.
        self._overlay_job: Any = None
        self._overlay_prefetch_job: Any = None
        self._overlay_generation = 0
        self._overlay_catalog: dict[str, dict[str, Any]] = {}
        # None means no deliberate choice has been made; a dict with an empty
        # source_id means the user explicitly selected "None". Never overwrite
        # this preference when a particular row lacks the chosen source.
        self._preferred_overlay: Optional[dict[str, str]] = None
        self._overlay_layer_signature: tuple[tuple[str, str], ...] = ()
        self._overlay_layer_source_id = ""
        self._overlay_cache_signature: Optional[tuple[Any, ...]] = None
        self._overlay_cache_frames: dict[int, str] = {}
        self._overlay_prepared_frames: set[int] = set()
        self._overlay_prefetch_range: tuple[int, int] | None = None
        # Compatibility diagnostics retained for extensions/tests that inspect the old
        # moving-window attributes. They now describe the resident cache coverage.
        self._overlay_window_start = -1
        self._overlay_window_end = -1
        self._overlay_window_signature: Optional[tuple[Any, ...]] = None
        self._periodic: Any = None
        self._play_anchor_monotonic: Optional[float] = None
        self._play_anchor_frame = 0
        self._request_seq = 0
        self._pending_frame_key: Optional[tuple[str, str, int]] = None
        self._failed_frame_key: Optional[tuple[str, str, int]] = None
        self._current_asset: Any = None
        self._direct_pattern: Optional[BrowserFrameSequencePattern] = None
        self._direct_mode_active = False
        self._direct_mode_disabled = False
        self._updating_widgets = False
        self._updating_surface = False
        self._surface_watchers: list[Any] = []
        self._current_dataset_id: Optional[str] = None
        self._current_row_id: Optional[str] = None
        self._metadata = VideoMetadata()
        self._last_preview: Optional[VideoFramePreview] = None
        self._restored_frame_index: Optional[int] = None
        self._frame_access_available = False
        self._display_mode = "auto"

        self.title = pn.pane.Markdown("### Video Viewer", sizing_mode="stretch_width")
        self.status = pn.pane.Alert(
            "Select a record with `record_id` and `video.uri` mappings.",
            alert_type="info",
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
            styles={"padding": "6px 10px"},
        )
        self.frame_surface = VideoFrameSurface(
            sizing_mode="scale_width",
            min_height=16,
            margin=(0, 0, 8, 0),
            prefetch_count=self.prefetch_count,
            message="No video selected",
        )
        # Compatibility alias for callers/tests that refer to the historical
        # ``frame_pane`` attribute. The visible pane wraps only standard Bokeh
        # models so dynamically opened plugin panels never introduce an unknown
        # custom frontend model type.
        self.frame_pane = self.frame_surface.view

        self.previous_button = pn.widgets.Button(
            name="Previous", icon="chevron-left", width=105, height=32
        )
        self.play_button = pn.widgets.Button(
            name="Play", icon="player-play", width=88, height=32, button_type="primary"
        )
        self.next_button = pn.widgets.Button(
            name="Next", icon="chevron-right", width=96, height=32
        )
        self.probe_button = pn.widgets.Button(
            name="Probe", icon="info-circle", width=92, height=32
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh", icon="refresh", width=96, height=32
        )
        self.cache_frames_button = pn.widgets.Button(
            name="Cache next 60 frames", icon="download", width=175, height=32,
            button_type="light", disabled=True,
        )
        self.speed = pn.widgets.Select(
            name="Preview speed",
            options={"0.5×": 0.5, "1×": 1.0, "2×": 2.0},
            value=1.0,
            width=105,
            height=44,
        )
        self.display_channels = pn.widgets.Select(
            name="Display channels",
            options={
                "Auto": "auto",
                "Source/native": "source",
                "RGB": "rgb",
            },
            value="auto",
            width=180,
            height=44,
            disabled=True,
        )
        self.overlay_source = pn.widgets.Select(
            name="Segmentation map",
            options={"None": ""},
            value="",
            width=220,
            height=44,
            disabled=True,
        )
        self.summary_columns_button = pn.widgets.Button(
            name="Summary unavailable",
            icon="table-plus",
            button_type="light",
            width=190,
            height=32,
            disabled=True,
            visible=False,
        )
        self.overlay_layers = pn.widgets.MultiChoice(
            name="Visible objects / layers",
            options={},
            value=[],
            sizing_mode="stretch_width",
            height=44,
            disabled=True,
        )
        self.overlay_opacity = pn.widgets.FloatSlider(
            name="Overlay opacity",
            start=0.05,
            end=0.9,
            step=0.05,
            value=0.45,
            width=180,
            height=44,
            disabled=True,
        )
        self.frame_slider = pn.widgets.IntSlider(
            name="Frame",
            start=0,
            end=0,
            value=0,
            step=1,
            sizing_mode="stretch_width",
        )
        self.meta = pn.pane.Markdown("", sizing_mode="stretch_width", margin=0)
        self.details_button = pn.widgets.Button(
            name="Show details",
            icon="info-circle",
            button_type="light",
            width=112,
            height=30,
        )
        self.details_box = pn.Column(
            self.meta,
            visible=False,
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
            styles={
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "box-sizing": "border-box",
                "padding": "8px 10px",
                "border": "1px solid #d8e0e8",
                "border-radius": "5px",
                "background": "#f8fafc",
                "overflow-x": "auto",
            },
        )

        self.previous_button.on_click(lambda _event: self._step(-1))
        self.next_button.on_click(lambda _event: self._step(1))
        self.play_button.on_click(lambda _event: self._toggle_playback())
        self.probe_button.on_click(lambda _event: self._probe_media())
        self.refresh_button.on_click(lambda _event: self._refresh_current_frame())
        self.cache_frames_button.on_click(lambda _event: self._prepare_browser_cache())
        self.summary_columns_button.on_click(
            lambda _event: self._update_summary_columns()
        )
        self.details_button.on_click(lambda _event: self._toggle_details())
        self.frame_slider.param.watch(self._on_slider_changed, "value")
        self.speed.param.watch(self._on_speed_changed, "value")
        self.display_channels.param.watch(self._on_display_mode_changed, "value")
        self.overlay_source.param.watch(self._on_overlay_source_changed, "value")
        self.overlay_layers.param.watch(self._on_overlay_layers_changed, "value")
        self.overlay_opacity.param.watch(self._on_overlay_opacity_changed, "value")
        self._surface_watchers = [
            self.frame_surface.on_frame_changed(self._on_surface_frame_changed),
            self.frame_surface.on_playing_changed(self._on_surface_playing_changed),
            self.frame_surface.on_error(self._on_surface_error),
            self.frame_surface.on_image_info_changed(self._on_browser_image_info_changed),
            self.frame_surface.on_cache_prepare(self._on_cache_prepare),
        ]

        def responsive_control_row(*objects: Any, margin=(0, 0, 4, 0)) -> Any:
            """Keep every control inside the plugin panel at narrow workspace widths."""

            return pn.Row(
                *objects,
                sizing_mode="stretch_width",
                margin=margin,
                styles={
                    "width": "100%",
                    "max-width": "100%",
                    "min-width": "0",
                    "overflow-x": "hidden",
                    "overflow-y": "visible",
                    "box-sizing": "border-box",
                    "gap": "8px",
                    "align-items": "center",
                    "flex-wrap": "wrap",
                },
            )

        # Split navigation from secondary controls.  A single fixed-width seven-widget
        # row could exceed the workspace card before flex wrapping had been reflected in
        # Panel/Bokeh sizing.  Two bounded rows remain readable at narrow panel widths.
        self.navigation_controls = responsive_control_row(
            self.previous_button, self.play_button, self.next_button
        )
        self.use_as_input_button = pn.widgets.Button(
            name="Use videos as input", button_type="primary", width=150,
        )
        self.use_as_input_button.on_click(self._offer_to_inputs)
        self.utility_controls = responsive_control_row(
            self.speed,
            self.display_channels,
            self.probe_button,
            self.refresh_button,
            self.cache_frames_button,
            self.use_as_input_button,
            margin=(0, 0, 4, 0),
        )
        self.overlay_controls = pn.Column(
            responsive_control_row(
                self.overlay_source,
                self.overlay_opacity,
                self.summary_columns_button,
                margin=(0, 0, 2, 0),
            ),
            self.overlay_layers,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
        self.controls = pn.Column(
            self.frame_slider,
            self.navigation_controls,
            self.utility_controls,
            self.overlay_controls,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
            styles={
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "overflow-x": "hidden",
                "box-sizing": "border-box",
            },
        )
        self.view = pn.Column(
            self.status,
            self.frame_pane,
            self.controls,
            responsive_control_row(self.details_button, margin=(0, 0, 4, 0)),
            self.details_box,
            sizing_mode="stretch_both",
            styles={
                "height": "100%",
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "min-height": "0",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "padding": "8px",
                "box-sizing": "border-box",
            },
        )

        self._subscribe()
        self._refresh_from_platform_focus()
        self._schedule(self._refresh_from_platform_focus)

    def panel(self):
        return self.view

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._stop_periodic()
        self.frame_surface.dispose()
        self._cancel_job("_frame_job")
        self._cancel_job("_probe_job")
        self._cancel_job("_summary_job")
        self._cancel_job("_overlay_job")
        self._cancel_job("_overlay_prefetch_job")
        self._updating_surface = True
        try:
            self.frame_surface.playing = False
            self.frame_surface.client_clock = False
        finally:
            self._updating_surface = False
        for watcher in list(self._surface_watchers):
            try:
                self.frame_surface.remove_listener(watcher)
            except Exception:
                pass
        self._surface_watchers.clear()
        self._set_summary_event_topic("")
        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in self._subscriptions:
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions.clear()

    def _offer_to_inputs(self, _event=None) -> None:
        """Offer this dataset's video references without exporting decoded frames."""
        try:
            dataset_id = str(self._current_dataset_id or "")
            if not dataset_id:
                raise ValueError("No video dataset is selected")
            datasets = self.context.datasets
            get_mapping = getattr(datasets, "get_mapping", None)
            video_col = get_mapping(dataset_id, "video.uri") if callable(get_mapping) else None
            columns = list(datasets.get_source(dataset_id).columns())
            if not video_col:
                video_col = next((c for c in ("video_uri", "video_path", "video_url") if c in columns), None)
            if not video_col or video_col not in columns:
                raise ValueError("Map video.uri to a physical dataset column first")
            id_column = get_mapping(dataset_id, "record_id") if callable(get_mapping) else None
            if not id_column or id_column not in columns:
                id_column = next((c for c in ("record_id", "object_id", "sample_id", "row_id", "id") if c in columns), None)
            if not id_column:
                raise ValueError("Map record_id to a physical dataset column first")
            if self.manager is None:
                raise RuntimeError("Plugin manager unavailable")
            request = ActionRequest(params={
                "role": "video", "origin": self.panel_id, "sample_id_column": id_column,
                "binding": {"kind": "dataset_columns", "dataset_id": dataset_id,
                            "columns": [video_col],
                            "join": {"sample_column": id_column, "source_column": id_column,
                                     "cardinality": "one"}},
            }, origin=self.panel_id)
            self.manager.run_action("data.compose.offer_input", self.context, request,
                on_done=lambda _result: self._schedule(lambda: self._input_offer_notice("Video source sent to Inputs & Data Composition.")),
                on_error=lambda exc: self._schedule(lambda: self._input_offer_notice(f"Could not offer video input: {exc}", failed=True)))
        except Exception as exc:
            self._input_offer_notice(f"Could not offer video input: {exc}", failed=True)

    def _input_offer_notice(self, message: str, *, failed: bool = False) -> None:
        self.status.alert_type = "warning" if failed else "success"
        self.status.object = message

    def _toggle_details(self) -> None:
        visible = not bool(self.details_box.visible)
        self.details_box.visible = visible
        self.details_button.name = "Hide details" if visible else "Show details"

    def get_state(self) -> dict[str, Any]:
        state = self.temporal.get_state()
        return {
            "state_version": self.state_version,
            "max_size": self.max_size,
            "preview_fps": self.preview_fps,
            "prefetch_count": self.prefetch_count,
            "display_mode": str(self._display_mode or "auto"),
            "playback_rate": float(self.speed.value or 1.0),
            "overlay_source_id": str(self.overlay_source.value or ""),
            "preferred_overlay": (
                dict(self._preferred_overlay)
                if self._preferred_overlay is not None else None
            ),
            "overlay_opacity": float(self.overlay_opacity.value or 0.45),
            "dataset_id": state.dataset_id,
            "row_id": state.row_id,
            "frame_index": int(state.frame_index),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        try:
            self.max_size = max(128, int(state.get("max_size", self.max_size)))
            self.preview_fps = max(1.0, min(30.0, float(state.get("preview_fps", self.preview_fps))))
            self.prefetch_count = max(0, min(24, int(state.get("prefetch_count", self.prefetch_count))))
            self.frame_surface.prefetch_count = self.prefetch_count
            restored_display = str(state.get("display_mode") or self._display_mode or "auto")
            self._display_mode = restored_display
            self._updating_widgets = True
            try:
                if restored_display in set(self.display_channels.options.values()):
                    self.display_channels.value = restored_display
            finally:
                self._updating_widgets = False
        except Exception:
            pass
        rate = state.get("playback_rate")
        if rate in (0.5, 1.0, 2.0):
            self.speed.value = float(rate)
        try:
            self.overlay_opacity.value = min(0.9, max(0.05, float(state.get("overlay_opacity", self.overlay_opacity.value))))
        except Exception:
            pass
        if "preferred_overlay" in state:
            preferred = state.get("preferred_overlay")
            if isinstance(preferred, Mapping):
                self._preferred_overlay = {
                    "source_id": str(preferred.get("source_id") or ""),
                    "provider_id": str(preferred.get("provider_id") or ""),
                    "label": str(preferred.get("label") or ""),
                }
            else:
                # A saved active selection should still restore even if nobody
                # explicitly changed the dropdown in that session.
                active = str(state.get("overlay_source_id") or "")
                self._preferred_overlay = (
                    {"source_id": active, "provider_id": "", "label": ""}
                    if active else None
                )
        else:
            # Older saved layouts stored only the currently visible source.
            # Treat a nonempty legacy value as an initial preference.
            restored_overlay = str(state.get("overlay_source_id") or "")
            self._preferred_overlay = (
                {"source_id": restored_overlay, "provider_id": "", "label": ""}
                if restored_overlay else None
            )
        try:
            self._restored_frame_index = max(0, int(state.get("frame_index", 0)))
        except Exception:
            self._restored_frame_index = None
        self._schedule(self._apply_restored_frame)
        self._schedule(self._refresh_overlay_catalog)

    def _subscribe(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        for topic, callback in (
            ("selection.focus.changed", self._on_focus_event),
            ("selection.focus.cleared", self._on_focus_event),
            ("dataset.active.changed", self._on_dataset_event),
            ("dataset.mapping.updated", self._on_dataset_event),
            ("mapping.resolved", self._on_dataset_event),
            (TEMPORAL_CHANGED_TOPIC, self._on_temporal_event),
            (TEMPORAL_CLEARED_TOPIC, self._on_temporal_event),
            (OVERLAY_CATALOG_CHANGED_TOPIC, self._on_overlay_event),
            (OVERLAY_SELECTION_CHANGED_TOPIC, self._on_overlay_event),
        ):
            self._subscriptions.append(
                events.subscribe(
                    topic,
                    callback,
                    owner_id=self.panel_id,
                    owner_label="Video Viewer",
                    owner_kind="plugin_panel",
                )
            )

    def _on_focus_event(self, _topic: str, _payload: Any) -> None:
        self._schedule(self._refresh_from_platform_focus)

    def _on_dataset_event(self, _topic: str, payload: Any) -> None:
        payload = payload if isinstance(payload, dict) else {}
        dataset_id = payload.get("dataset_id")
        if dataset_id and self._current_dataset_id and str(dataset_id) != self._current_dataset_id:
            return
        self._schedule(self._refresh_from_platform_focus)

    def _on_temporal_event(self, _topic: str, payload: Any) -> None:
        payload = payload if isinstance(payload, dict) else {}
        if payload.get("dataset_id") != self._current_dataset_id or payload.get("row_id") != self._current_row_id:
            return

        # Browser-direct playback is authoritative for visual frame cadence. Feedback
        # from this panel must update shared temporal state and ordinary widgets, but
        # must not be reflected straight back into the browser surface: doing so would
        # re-run its CustomJS control callback, cancel/re-anchor requestAnimationFrame,
        # and turn every presented frame into a browser -> Python -> browser loop.
        local_direct_feedback = (
            self._direct_mode_active
            and str(payload.get("panel_id") or "") == self.panel_id
        )
        if local_direct_feedback:
            self._schedule(self._sync_widgets_from_temporal_state)
        else:
            self._schedule(self._sync_from_temporal_state)
        self._schedule(self._request_overlay_window)

    def _refresh_from_platform_focus(self) -> None:
        if self._disposed:
            return
        selection = getattr(self.context, "selection", None)
        get_focus = getattr(selection, "get_focus", None)
        if not callable(get_focus):
            self._show_empty("SelectionManager is unavailable.")
            return
        try:
            focus = get_focus()
        except Exception as exc:
            self._show_error(f"Could not read platform focus: {exc}")
            return
        dataset_id = getattr(focus, "dataset_id", None)
        row_id = getattr(focus, "row_id", None)
        if dataset_id is None or row_id is None:
            self._current_dataset_id = None
            self._current_row_id = None
            self._metadata = VideoMetadata()
            self._show_empty("No focused video record.")
            return

        changed = (
            str(dataset_id) != self._current_dataset_id
            or str(row_id) != self._current_row_id
        )
        self._current_dataset_id = str(dataset_id)
        self._current_row_id = str(row_id)
        try:
            self._display_mode = self.resolver.set_display_mode(
                self._current_dataset_id,
                self._current_row_id,
                self._display_mode,
                origin=self.panel_id,
                panel_id=self.panel_id,
            )
        except Exception:
            self._display_mode = "auto"
        if changed:
            self._stop_periodic()
            self._cancel_job("_frame_job")
            self._pending_frame_key = None
            self._failed_frame_key = None
            self._last_preview = None
            self._current_asset = None
            self._direct_pattern = None
            self._direct_mode_active = False
            self._direct_mode_disabled = False
            self.cache_frames_button.disabled = True
            self._play_anchor_monotonic = None
            self._play_anchor_frame = 0
            self._invalidate_overlay_window(clear_surface=True, cancel_job=True)
            self._updating_surface = True
            try:
                self.frame_surface.reset("Loading focused video frame…")
                self.frame_surface.prefetch_count = self.prefetch_count
            finally:
                self._updating_surface = False

        try:
            asset = self.resolver.asset_for_row(
                self._current_dataset_id,
                self._current_row_id,
            )
            # Asset resolution already used bounded row access. Use its mapped
            # metadata rather than making a second identical DatasetManager query.
            mapped = dict(getattr(asset, "metadata", {}) or {})
            self._metadata = (
                VideoMetadata(**mapped)
                if mapped and {"source", "extra"}.issubset(mapped)
                else self.resolver.metadata_for_row(self._current_dataset_id, self._current_row_id)
            )
            self._current_asset = asset
        except Exception as exc:
            text = str(exc)
            if "requires bounded DatasetManager.get_row_by_id" in text or "DatasetSource access" in text:
                self._show_empty(
                    "No usable video source is selected for the current focus yet. "
                    "Choose or register a video dataset/source, focus one of its rows, then use Refresh."
                )
            else:
                self._show_error(text)
            self._refresh_overlay_catalog()
            return

        candidate = self._direct_pattern_for_asset(asset)
        # A PNG validated by the browser may now have explicit RGB/RGBA metadata.
        # That refines the resolver's transform choice without changing the media.
        # Keep the already-visible canvas rather than restarting its playback.
        if (
            self._direct_mode_active and self._direct_pattern is not None
            and self._direct_pattern.display_transform == "rgb_unpremultiplied_png"
            and candidate is not None
            and candidate.display_transform in {"source", "rgb_unpremultiplied"}
            and replace(candidate, display_transform="rgb_unpremultiplied_png") == self._direct_pattern
        ):
            candidate = self._direct_pattern
        if candidate != self._direct_pattern:
            self._direct_pattern = candidate
            self._direct_mode_active = False
            self._direct_mode_disabled = False
            self._updating_surface = True
            try:
                self.frame_surface.client_clock = False
                self.frame_surface.generation += 1
            finally:
                self._updating_surface = False

        # This asset was just resolved with bounded row access.  Avoid a third
        # synchronous dataset lookup merely to answer whether its frames exist.
        if asset.media_kind == "frame_sequence":
            self._frame_access_available = bool(asset.frame_uris or asset.frame_uri_template)
        else:
            self._frame_access_available = bool(self.resolver.decoder_available())
        self._configure_controls_from_metadata(self._metadata)
        state = self.temporal.get_state()
        if state.dataset_id != self._current_dataset_id or state.row_id != self._current_row_id:
            self.temporal.set_frame(
                self._current_dataset_id,
                self._current_row_id,
                0,
                fps=self._metadata.fps,
                frame_count=self._metadata.frame_count,
                origin=self.panel_id,
                panel_id=self.panel_id,
            )
        else:
            self.temporal.set_frame(
                self._current_dataset_id,
                self._current_row_id,
                state.frame_index,
                fps=self._metadata.fps or state.fps,
                frame_count=self._metadata.frame_count or state.frame_count,
                origin=self.panel_id,
                panel_id=self.panel_id,
            )

        self._apply_restored_frame()
        current_state = self.temporal.get_state()
        self._sync_surface_from_temporal_state(current_state)
        self._update_metadata_display()

        # Passive focus/mapping refreshes must not erase the status of the current
        # managed frame request. Deterministic failures remain sticky until an
        # explicit refresh, frame change, or row change.
        current_frame_key = (
            self._current_dataset_id,
            self._current_row_id,
            int(current_state.frame_index),
        )
        frame_request_owns_status = (
            self._failed_frame_key == current_frame_key
            or self._pending_frame_key == current_frame_key
            or self._direct_mode_active
        )
        if not frame_request_owns_status:
            self.status.alert_type = "info"
            access_note = self.resolver.frame_access_description_for_row(
                self._current_dataset_id,
                self._current_row_id,
            )
            direct_note = (
                " Browser-direct playback is available; unsupported frames use the managed resolver."
                if self._direct_pattern is not None and not self._direct_mode_disabled
                else ""
            )
            self.status.object = f"Focused `{self._current_row_id}`. {access_note}{direct_note}"

        # The first paint takes precedence over overlay discovery and prefetch.
        # The resolver has already approved the browser transform, including a
        # verified-PNG mode which checks the IHDR before drawing scientific RGBD.
        # Any direct-load error triggers the existing managed-resolver fallback.
        if self._frame_access_available:
            if self._direct_pattern is not None and not self._direct_mode_disabled:
                if not self._direct_mode_active:
                    self._enable_direct_mode()
            else:
                self._request_current_frame(force=False)
        else:
            self._surface_show_message("Frame access unavailable")

        self._refresh_overlay_catalog()
        self._request_overlay_window()

    def _on_overlay_event(self, topic: str, payload: Any) -> None:
        data = dict(payload or {}) if isinstance(payload, Mapping) else {}
        dataset_id = str(data.get("dataset_id") or "")
        row_id = str(data.get("row_id") or "")
        if dataset_id and dataset_id != str(self._current_dataset_id or ""):
            return
        if row_id and row_id != str(self._current_row_id or ""):
            return

        # ``select_source`` publishes synchronously. This panel's own widget callback
        # has already invalidated/refreshed the selected source before that event can be
        # observed, so echoing our own selection notification through a scheduled
        # catalogue refresh starts a second prepare job for unprepared retained
        # annotations.
        if (
            str(topic or "") == OVERLAY_SELECTION_CHANGED_TOPIC
            and str(data.get("origin") or "") == self.panel_id
        ):
            return

        # A transient provider can replace pixels for exactly one already-selected
        # frame without changing its catalogue/layer metadata. Invalidate only that
        # resident frame; all neighbouring prepared frames remain hot.
        if str(topic or "") == OVERLAY_CATALOG_CHANGED_TOPIC and bool(data.get("content_changed")):
            selected = str(self.overlay_source.value or "")
            source_id = str(data.get("source_id") or "")
            if not source_id or source_id == selected:
                try:
                    frame = max(0, int(data.get("frame_index")))
                except (TypeError, ValueError):
                    frame = -1
                if frame >= 0:
                    self._overlay_prepared_frames.discard(frame)
                    self._overlay_cache_frames.pop(frame, None)
                    merge = getattr(self.frame_surface, "merge_overlay_frames", None)
                    if callable(merge):
                        merge({frame: ""})
                    self._schedule(self._request_overlay_window)
        self._schedule(self._refresh_overlay_catalog)

    def _refresh_overlay_catalog(self) -> None:
        registry = self.overlay_registry
        dataset_id = self._current_dataset_id
        row_id = self._current_row_id
        if registry is None or not dataset_id or row_id is None:
            self._overlay_catalog = {}
            self._updating_widgets = True
            try:
                self.overlay_source.options = {"None": ""}
                self.overlay_source.value = ""
                self.overlay_source.disabled = True
                self.overlay_layers.options = {}
                self.overlay_layers.value = []
                self.overlay_layers.disabled = True
                self.overlay_opacity.disabled = True
            finally:
                self._updating_widgets = False
            self._overlay_layer_signature = ()
            self._invalidate_overlay_window(clear_surface=True, cancel_job=True)
            return
        sources = list(registry.list_sources(dataset_id, row_id) or ())
        self._overlay_catalog = {
            str(value.get("source_id") or ""): dict(value)
            for value in sources
            if str(value.get("source_id") or "")
        }
        options = {"None": ""}
        for value in sources:
            source_id = str(value.get("source_id") or "")
            label = str(value.get("label") or source_id)
            options[label] = source_id
        registered = str(registry.selected_source(dataset_id, row_id) or "")
        if self._preferred_overlay is not None:
            # A missing choice temporarily displays None, but is *not* a new
            # preference. Reconsider it whenever the next catalogue arrives.
            selected = self._match_preferred_overlay(self._overlay_catalog)
        else:
            selected = registered if registered in self._overlay_catalog else ""
        if registered != selected:
            registry.select_source(dataset_id, row_id, selected or None, origin=self.panel_id)
        previous_selected = str(self.overlay_source.value or "")
        self._updating_widgets = True
        try:
            self.overlay_source.options = options
            self.overlay_source.value = selected
            self.overlay_source.disabled = not bool(sources)
            self.overlay_opacity.disabled = not bool(selected)
        finally:
            self._updating_widgets = False
        if selected != previous_selected:
            self._invalidate_overlay_window(clear_surface=True, cancel_job=True)
        self._refresh_overlay_layers()
        self._refresh_summary_button()

    def _match_preferred_overlay(self, catalog: Mapping[str, Mapping[str, Any]]) -> str:
        preferred = self._preferred_overlay
        if preferred is None:
            return ""
        source_id = str(preferred.get("source_id") or "")
        if not source_id:
            return ""  # Explicit user choice of None.
        provider_id = str(preferred.get("provider_id") or "")
        if source_id in catalog and (
            not provider_id
            or str(catalog[source_id].get("provider_id") or "") == provider_id
        ):
            return source_id
        label = str(preferred.get("label") or "").strip().casefold()
        if not provider_id or not label:
            return ""
        # Artifact identifiers may be per-video. Only remap when the provider
        # and human-visible map name identify exactly one available source.
        matches = [
            key for key, descriptor in catalog.items()
            if str(descriptor.get("provider_id") or "") == provider_id
            and str(descriptor.get("label") or "").strip().casefold() == label
        ]
        return matches[0] if len(matches) == 1 else ""

    def _refresh_overlay_layers(self) -> None:
        registry = self.overlay_registry
        source_id = str(self.overlay_source.value or "")
        if registry is None or not source_id or not self._current_dataset_id or self._current_row_id is None:
            self._updating_widgets = True
            try:
                self.overlay_layers.options = {}
                self.overlay_layers.value = []
                self.overlay_layers.disabled = True
            finally:
                self._updating_widgets = False
            self._overlay_layer_signature = ()
            self._overlay_layer_source_id = ""
            self._invalidate_overlay_window(clear_surface=True, cancel_job=True)
            self._refresh_summary_button()
            return
        layers = list(
            registry.list_layers(source_id, self._current_dataset_id, self._current_row_id) or ()
        )
        layer_pairs = [
            (
                str(value.get("layer_id") or ""),
                str(value.get("label") or value.get("layer_id") or ""),
            )
            for value in layers
            if str(value.get("layer_id") or "")
        ]
        options = {label: layer_id for layer_id, label in layer_pairs}
        same_source = self._overlay_layer_source_id == source_id
        previous_values = (
            tuple(str(value) for value in list(self.overlay_layers.value or ()) if str(value))
            if same_source else ()
        )
        current = [value for value in previous_values if value in options.values()]
        if not current:
            current = [
                str(value.get("layer_id"))
                for value in layers
                if str(value.get("layer_id") or "")
                and bool(value.get("enabled_by_default", True))
            ]
        elif same_source and bool(
            dict((self._overlay_catalog.get(source_id) or {}).get("metadata") or {}).get("workbench_preview")
        ):
            # The Workbench replaces its preview as additional objects are prompted.
            # Keep deliberate existing layer exclusions, but show newly introduced
            # object masks by default rather than retaining an arms-only selection.
            known_ids = {layer_id for layer_id, _label in self._overlay_layer_signature}
            current.extend(
                layer_id
                for layer in layers
                if (layer_id := str(layer.get("layer_id") or ""))
                and layer_id not in known_ids
                and layer_id not in current
                and bool(layer.get("enabled_by_default", True))
            )
        signature = tuple(layer_pairs)
        metadata_changed = signature != self._overlay_layer_signature
        self._updating_widgets = True
        try:
            # Label-only catalogue changes are presentation metadata; they must not
            # throw away already prepared overlay pixels.
            if metadata_changed and self.overlay_layers.value:
                self.overlay_layers.value = []
            self.overlay_layers.options = options
            self.overlay_layers.value = current
            self.overlay_layers.disabled = not bool(options)
            self.overlay_opacity.disabled = False
        finally:
            self._updating_widgets = False
        self._overlay_layer_signature = signature
        self._overlay_layer_source_id = source_id
        selected_changed = tuple(current) != previous_values
        if selected_changed:
            self._invalidate_overlay_window(clear_surface=True, cancel_job=True)
        self._request_overlay_window(force=selected_changed)

    def _on_overlay_source_changed(self, event: Any) -> None:
        if self._disposed or self._updating_widgets:
            return
        source_id = str(event.new or "")
        if self.overlay_registry is None or not self._current_dataset_id or self._current_row_id is None:
            return
        descriptor = self._overlay_catalog.get(source_id, {})
        self._preferred_overlay = {
            "source_id": source_id,
            "provider_id": str(descriptor.get("provider_id") or ""),
            "label": str(descriptor.get("label") or ""),
        }
        self.overlay_registry.select_source(
            self._current_dataset_id,
            self._current_row_id,
            source_id or None,
            origin=self.panel_id,
        )
        self._invalidate_overlay_window(clear_surface=True, cancel_job=True)
        self._refresh_overlay_layers()
        self._refresh_summary_button()

    def _selected_summary_spec(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        source_id = str(self.overlay_source.value or "")
        if not source_id:
            return {}, {}

        source = dict(self._overlay_catalog.get(source_id) or {})
        metadata = dict(source.get("metadata") or {})
        summary_spec = dict(metadata.get("dataset_summary") or {})
        return source, summary_spec

    def _set_summary_event_topic(self, topic: str) -> None:
        topic = str(topic or "").strip()
        if topic == self._summary_event_topic:
            return

        events = getattr(self.context, "events", None)
        unsubscribe = getattr(events, "unsubscribe", None)

        if self._summary_event_subscription is not None and callable(unsubscribe):
            try:
                unsubscribe(self._summary_event_subscription)
            except Exception:
                pass

        self._summary_event_subscription = None
        self._summary_event_topic = topic

        if not topic:
            return

        subscribe = getattr(events, "subscribe", None)
        if not callable(subscribe):
            return

        try:
            token = subscribe(
                topic,
                self._on_summary_event,
                owner_id=self.panel_id,
                owner_label="Video Viewer dataset summary",
                owner_kind="plugin_panel",
            )
        except TypeError:
            token = subscribe(topic, self._on_summary_event)

        self._summary_event_subscription = token

    def _on_summary_event(self, _topic: str, payload: Any) -> None:
        data = dict(payload or {}) if isinstance(payload, Mapping) else {}
        dataset_id = str(data.get("dataset_id") or "")
        if (
            dataset_id
            and self._current_dataset_id
            and dataset_id != str(self._current_dataset_id)
        ):
            return

        self._schedule(self._refresh_summary_button)

    def _refresh_summary_button(self) -> None:
        source_id = str(self.overlay_source.value or "")
        source, summary_spec = self._selected_summary_spec()

        # The control occupies the Segmentation map area only when a source exists.
        self.summary_columns_button.visible = bool(source_id)

        if not source_id:
            self._set_summary_event_topic("")
            self.summary_columns_button.disabled = True
            self.summary_columns_button.name = "Summary unavailable"
            return

        action_id = str(summary_spec.get("action_id") or "").strip()
        overlay_name = str(summary_spec.get("overlay_name") or "").strip()
        metadata_key = str(summary_spec.get("metadata_key") or "").strip()
        event_topic = str(summary_spec.get("event_topic") or "").strip()

        self._set_summary_event_topic(event_topic)

        available = bool(action_id and overlay_name and metadata_key)
        if not available:
            self.summary_columns_button.disabled = True
            self.summary_columns_button.name = "Summary unavailable"
            return

        attached = False
        current = False

        try:
            datasets = getattr(self.context, "datasets", None)
            if datasets is None or not self._current_dataset_id:
                raise RuntimeError("DatasetManager is unavailable")

            dataset_meta = dict(
                datasets.get_meta(self._current_dataset_id) or {}
            )
            overlay_names = {
                str(value)
                for value in (dataset_meta.get("column_overlays") or ())
            }
            feature_meta = dict(dataset_meta.get(metadata_key) or {})
            attached_columns = {
                str(value)
                for value in (feature_meta.get("columns") or ())
            }
            current_columns = {
                str(value)
                for value in datasets.list_columns(self._current_dataset_id)
            }

            attached = bool(
                overlay_name in overlay_names
                and (
                    not attached_columns
                    or attached_columns.issubset(current_columns)
                )
            )

            source_metadata = dict(source.get("metadata") or {})
            selected_artifact_id = str(
                source_metadata.get("artifact_id") or ""
            )
            stored_source_id = str(feature_meta.get("source_id") or "")
            stored_artifact_id = str(
                feature_meta.get("source_artifact_id") or ""
            )

            # Older summary overlays did not store source_id. Artifact equality keeps
            # those overlays backward-compatible without creating a second schema.
            same_source = bool(
                stored_source_id == source_id
                or (
                    not stored_source_id
                    and selected_artifact_id
                    and stored_artifact_id == selected_artifact_id
                )
            )

            current = bool(
                attached
                and same_source
                and (
                    not selected_artifact_id
                    or stored_artifact_id == selected_artifact_id
                )
            )
        except Exception:
            attached = False
            current = False

        busy = self._summary_job is not None
        source_preparing = bool(
            self._overlay_job is not None
            and not bool(source.get("prepared", True))
        )

        self.summary_columns_button.disabled = bool(
            busy
            or source_preparing
            or self.manager is None
        )

        self.summary_columns_button.name = (
            "Updating summary columns…"
            if busy
            else (
                "Refresh summary columns"
                if current
                else (
                    "Update summary columns"
                    if attached
                    else "Add summary columns"
                )
            )
        )

    def _update_summary_columns(self) -> None:
        if (
            self.manager is None
            or not self._current_dataset_id
            or self._current_row_id is None
            or self._summary_job is not None
        ):
            return

        source_id = str(self.overlay_source.value or "")
        source, summary_spec = self._selected_summary_spec()
        action_id = str(summary_spec.get("action_id") or "").strip()

        if not source_id or not action_id:
            self.status.alert_type = "warning"
            self.status.object = (
                "The selected video overlay does not expose dataset summary columns."
            )
            self._refresh_summary_button()
            return

        source_label = str(source.get("label") or source_id)

        request = ActionRequest(
            params={
                "dataset_id": self._current_dataset_id,
                "row_id": self._current_row_id,
                "source_id": source_id,
            },
            origin=self.panel_id,
        )

        self.status.alert_type = "info"
        self.status.object = (
            f"Updating segmentation summary columns from {source_label}…"
        )

        try:
            self._summary_job = self.manager.run_action(
                action_id,
                self.context,
                request,
                on_done=lambda result: self._schedule(
                    lambda: self._summary_columns_done(result)
                ),
                on_error=lambda exc: self._schedule(
                    lambda: self._summary_columns_failed(exc)
                ),
            )
        except Exception as exc:
            self._summary_columns_failed(exc)
            return

        self._refresh_summary_button()

    def _summary_columns_done(self, result: Any) -> None:
        self._summary_job = None
        data = dict(result or {}) if isinstance(result, Mapping) else {}

        # Retained dataset annotations may have been indexed as part of the action,
        # so refresh the catalogue before deriving the final button state.
        self._refresh_overlay_catalog()

        source_label = str(
            data.get("source_label")
            or "selected segmentation"
        )
        column_count = len(data.get("columns") or ())
        self.status.alert_type = "success"
        self.status.object = (
            f"Updated {column_count:,} segmentation summary columns from "
            f"{source_label} across "
            f"{int(data.get('row_count') or 0):,} dataset rows "
            f"({int(data.get('object_rows') or 0):,} with objects)."
        )

        self._refresh_summary_button()

    def _summary_columns_failed(self, exc: Exception) -> None:
        self._summary_job = None
        self.status.alert_type = "danger"
        self.status.object = (
            f"Could not update segmentation summary columns: {exc}"
        )
        self._refresh_summary_button()

    def _on_overlay_layers_changed(self, _event: Any) -> None:
        if self._disposed or self._updating_widgets:
            return
        self._invalidate_overlay_window(clear_surface=True, cancel_job=True)
        self._request_overlay_window(force=True)

    def _on_overlay_opacity_changed(self, _event: Any) -> None:
        if self._disposed or self._updating_widgets:
            return
        # Provider pixels are RGBA masks/detections. Opacity is purely a core.video
        # presentation transform and therefore belongs on the existing browser/server
        # surface, not in a new provider render job.
        opacity = min(1.0, max(0.0, float(self.overlay_opacity.value or 0.45)))
        setter = getattr(self.frame_surface, "set_overlay_opacity", None)
        if callable(setter):
            setter(opacity)
        elif self._overlay_cache_frames:
            self.frame_surface.set_overlay_frames(self._overlay_cache_frames, opacity=opacity)

    def _prepare_overlay_source_if_needed(self, source_id: str) -> None:
        source = self._overlay_catalog.get(source_id) or {}
        if bool(source.get("prepared", True)) or self.overlay_registry is None:
            self._request_overlay_window(force=True)
            return
        if self._overlay_job is not None:
            return
        dataset_id = str(self._current_dataset_id or "")
        row_id = str(self._current_row_id or "")
        generation = self._overlay_generation

        def work(*, cancel_token: Any):
            return self.overlay_registry.prepare_source(
                source_id, dataset_id, row_id, cancel_token=cancel_token
            )

        def done(_result: Any) -> None:
            if self._disposed or generation != self._overlay_generation:
                return
            self._overlay_job = None
            self._refresh_overlay_catalog()

        def error(exc: BaseException) -> None:
            if generation != self._overlay_generation:
                return
            self._overlay_job = None
            self.status.alert_type = "warning"
            self.status.object = f"Could not prepare the selected overlay source: {exc}"

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                done(work(cancel_token=None))
            except BaseException as exc:
                error(exc)
            return
        self._overlay_job = jobs.submit(
            work,
            title="Preparing video overlay",
            key=f"core.video.overlay.prepare:{id(self)}:{generation}",
            on_done=done,
            on_error=error,
        )

    def _invalidate_overlay_window(
        self,
        *,
        clear_surface: bool,
        cancel_job: bool,
    ) -> None:
        """Invalidate source/layer-dependent resident overlay presentation state."""

        self._overlay_generation += 1
        self._overlay_window_start = -1
        self._overlay_window_end = -1
        self._overlay_window_signature = None
        self._overlay_cache_signature = None
        self._overlay_cache_frames = {}
        self._overlay_prepared_frames = set()
        self._overlay_prefetch_range = None
        if cancel_job:
            self._cancel_job("_overlay_job")
            self._cancel_job("_overlay_prefetch_job")
        if clear_surface:
            self.frame_surface.clear_overlay_frames()

    @staticmethod
    def _overlay_fill_bounds(
        frame_index: int,
        *,
        fps: float,
        playback_rate: float,
        frame_count: int,
    ) -> tuple[int, int, int, int]:
        """Small producer horizon modelled on the former Segmentation Viewer cache."""

        anchor = max(0, int(frame_index or 0))
        safe_fps = max(1.0, float(fps or 10.0))
        rate = max(0.01, float(playback_rate or 1.0))
        ahead = max(24, min(720, int(round(safe_fps * rate * 6.0))))
        behind = max(8, min(240, int(round(safe_fps * 2.0))))
        start = max(0, anchor - behind)
        end = anchor + ahead
        if int(frame_count or 0) > 0:
            end = min(end, int(frame_count) - 1)
        return start, max(start, end), ahead, behind

    def _overlay_signature(
        self,
        *,
        source_id: str,
        dataset_id: str,
        row_id: str,
        layers: tuple[str, ...],
    ) -> tuple[Any, ...]:
        # Opacity is deliberately absent: it is presentation-only and can change
        # instantly while reusing exactly the same provider pixels.
        return (str(source_id), str(dataset_id), str(row_id), tuple(layers))

    def _overlay_request_context(self) -> tuple[str, str, str, tuple[str, ...], float, Any] | None:
        if self._disposed or self.overlay_registry is None:
            return None
        source_id = str(self.overlay_source.value or "")
        if not source_id or not self._current_dataset_id or self._current_row_id is None:
            return None
        source = self._overlay_catalog.get(source_id) or {}
        if not bool(source.get("prepared", True)):
            return None
        dataset_id = str(self._current_dataset_id)
        row_id = str(self._current_row_id)
        layers = tuple(str(value) for value in list(self.overlay_layers.value or ()) if str(value))
        opacity = min(1.0, max(0.0, float(self.overlay_opacity.value or 0.45)))
        return source_id, dataset_id, row_id, layers, opacity, self.temporal.get_state()

    def _merge_overlay_result(
        self,
        result: Any,
        *,
        request_start: int,
        request_end: int,
        signature: tuple[Any, ...],
        opacity: float,
    ) -> None:
        data = dict(result or {}) if isinstance(result, Mapping) else {}
        frames = {
            int(frame): str(uri)
            for frame, uri in dict(data.get("frames") or {}).items()
            if str(uri or "")
        }
        prepared_values = data.get("prepared_frames")
        if isinstance(prepared_values, (list, tuple, set)):
            prepared = {max(0, int(frame)) for frame in prepared_values}
        else:
            result_start = max(0, int(data.get("frame_start", request_start) or request_start))
            result_end = max(result_start, int(data.get("frame_end", request_end) or request_end))
            prepared = set(range(result_start, result_end + 1))
        self._overlay_cache_signature = signature
        self._overlay_window_signature = signature
        self._overlay_cache_frames.update(frames)
        self._overlay_prepared_frames.update(prepared)
        if self._overlay_prepared_frames:
            self._overlay_window_start = min(self._overlay_prepared_frames)
            self._overlay_window_end = max(self._overlay_prepared_frames)
        merge = getattr(self.frame_surface, "merge_overlay_frames", None)
        if callable(merge):
            merge(frames, opacity=opacity)
        else:
            self.frame_surface.set_overlay_frames(self._overlay_cache_frames, opacity=opacity)

    def _request_overlay_exact(
        self,
        *,
        frame: int,
        source_id: str,
        dataset_id: str,
        row_id: str,
        layers: tuple[str, ...],
        opacity: float,
        signature: tuple[Any, ...],
    ) -> None:
        if frame in self._overlay_prepared_frames or self._overlay_job is not None:
            return
        generation = self._overlay_generation

        def work(*, cancel_token: Any):
            return self.overlay_registry.render_window(
                source_id,
                dataset_id,
                row_id,
                frame_start=frame,
                frame_end=frame,
                layer_ids=layers,
                opacity=opacity,
                cancel_token=cancel_token,
            )

        def done(result: Any) -> None:
            if (
                self._disposed
                or generation != self._overlay_generation
                or source_id != str(self.overlay_source.value or "")
            ):
                return
            self._overlay_job = None
            self._merge_overlay_result(
                result,
                request_start=frame,
                request_end=frame,
                signature=signature,
                opacity=float(self.overlay_opacity.value or opacity),
            )
            # First paint is now available; immediately begin extending the resident
            # cache ahead of the current playhead.
            self._schedule(self._request_overlay_window)

        def error(exc: BaseException) -> None:
            if generation != self._overlay_generation:
                return
            self._overlay_job = None
            self.status.alert_type = "warning"
            self.status.object = f"Video overlay unavailable: {exc}"

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                done(work(cancel_token=None))
            except BaseException as exc:
                error(exc)
            return
        self._overlay_job = jobs.submit(
            work,
            title=f"Preparing video overlay frame {frame}",
            key=f"core.video.overlay.frame:{id(self)}:{generation}:{frame}",
            on_done=done,
            on_error=error,
        )

    def _next_overlay_missing_band(
        self,
        *,
        anchor: int,
        start: int,
        end: int,
        fps: float,
        playback_rate: float,
    ) -> tuple[int, int] | None:
        prepared = self._overlay_prepared_frames
        effective_fps = max(1.0, float(fps or 10.0) * max(0.01, float(playback_rate or 1.0)))
        # Keep each producer task short enough to reprioritise. About two seconds of
        # playback is a good compromise between SQLite/query overhead and latency.
        band_size = max(6, min(120, int(round(effective_fps * 2.0))))
        forward_start = max(start, anchor + 1)
        cursor = forward_start
        while cursor <= end:
            while cursor <= end and cursor in prepared:
                cursor += 1
            if cursor > end:
                break
            right = cursor
            while right + 1 <= end and right + 1 not in prepared and right - cursor + 1 < band_size:
                right += 1
            return cursor, right
        # Fill a small amount behind only after forward runway is hot.
        cursor = min(anchor - 1, end)
        while cursor >= start:
            while cursor >= start and cursor in prepared:
                cursor -= 1
            if cursor < start:
                break
            left = cursor
            while left - 1 >= start and left - 1 not in prepared and cursor - left + 1 < band_size:
                left -= 1
            return left, cursor
        return None

    def _request_overlay_prefetch(
        self,
        *,
        source_id: str,
        dataset_id: str,
        row_id: str,
        layers: tuple[str, ...],
        opacity: float,
        signature: tuple[Any, ...],
        state: Any,
    ) -> None:
        if self._overlay_prefetch_job is not None:
            return
        anchor = max(0, int(getattr(state, "frame_index", 0) or 0))
        fps = float(getattr(state, "fps", 0.0) or self._metadata.fps or 0.0)
        rate = max(0.01, float(getattr(state, "playback_rate", 1.0) or 1.0))
        count = int(getattr(state, "frame_count", 0) or self._metadata.frame_count or 0)
        start, end, ahead, _behind = self._overlay_fill_bounds(
            anchor, fps=fps, playback_rate=rate, frame_count=count
        )
        # Refill at 90% of the small producer lookahead, matching the old segmentation
        # panel's deliberately early low-water behaviour.
        forward_prepared = 0
        probe = anchor + 1
        while probe <= end and probe in self._overlay_prepared_frames:
            forward_prepared += 1
            probe += 1
        low_water = max(8, int(round(ahead * 0.90)))
        if forward_prepared > low_water and anchor in self._overlay_prepared_frames:
            return
        band = self._next_overlay_missing_band(
            anchor=anchor,
            start=start,
            end=end,
            fps=fps,
            playback_rate=rate,
        )
        if band is None:
            return
        request_start, request_end = band
        generation = self._overlay_generation
        self._overlay_prefetch_range = band

        def work(*, cancel_token: Any):
            return self.overlay_registry.render_window(
                source_id,
                dataset_id,
                row_id,
                frame_start=request_start,
                frame_end=request_end,
                layer_ids=layers,
                opacity=opacity,
                cancel_token=cancel_token,
            )

        def done(result: Any) -> None:
            if (
                self._disposed
                or generation != self._overlay_generation
                or source_id != str(self.overlay_source.value or "")
            ):
                return
            self._overlay_prefetch_job = None
            self._overlay_prefetch_range = None
            self._merge_overlay_result(
                result,
                request_start=request_start,
                request_end=request_end,
                signature=signature,
                opacity=float(self.overlay_opacity.value or opacity),
            )
            self._schedule(self._request_overlay_window)

        def error(exc: BaseException) -> None:
            if generation != self._overlay_generation:
                return
            self._overlay_prefetch_job = None
            self._overlay_prefetch_range = None
            self.status.alert_type = "warning"
            self.status.object = f"Video overlay preload unavailable: {exc}"

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                done(work(cancel_token=None))
            except BaseException as exc:
                error(exc)
            return
        self._overlay_prefetch_job = jobs.submit(
            work,
            title=f"Preloading video overlays {request_start}-{request_end}",
            key=(
                f"core.video.overlay.prefetch:{id(self)}:{generation}:"
                f"{request_start}:{request_end}"
            ),
            on_done=done,
            on_error=error,
        )

    def _request_overlay_window(self, force: bool = False) -> None:
        if self._disposed or self.overlay_registry is None:
            return
        source_id = str(self.overlay_source.value or "")
        if not source_id or not self._current_dataset_id or self._current_row_id is None:
            return
        source = self._overlay_catalog.get(source_id) or {}
        if not bool(source.get("prepared", True)):
            if force:
                self._prepare_overlay_source_if_needed(source_id)
            return

        dataset_id = str(self._current_dataset_id)
        row_id = str(self._current_row_id)
        layers = tuple(str(value) for value in list(self.overlay_layers.value or ()) if str(value))
        signature = self._overlay_signature(
            source_id=source_id,
            dataset_id=dataset_id,
            row_id=row_id,
            layers=layers,
        )
        if signature != self._overlay_cache_signature and self._overlay_cache_signature is not None:
            self._invalidate_overlay_window(clear_surface=True, cancel_job=True)
        self._overlay_cache_signature = signature
        self._overlay_window_signature = signature

        state = self.temporal.get_state()
        anchor = max(0, int(getattr(state, "frame_index", 0) or 0))
        opacity = min(1.0, max(0.0, float(self.overlay_opacity.value or 0.45)))
        # Exact-frame prime is intentionally independent from the background producer.
        # A seek/current frame therefore cannot wait behind a large preload.
        if anchor not in self._overlay_prepared_frames:
            self._request_overlay_exact(
                frame=anchor,
                source_id=source_id,
                dataset_id=dataset_id,
                row_id=row_id,
                layers=layers,
                opacity=opacity,
                signature=signature,
            )
        # Do not let initial background work compete with the first exact-frame paint.
        if anchor in self._overlay_prepared_frames:
            self._request_overlay_prefetch(
                source_id=source_id,
                dataset_id=dataset_id,
                row_id=row_id,
                layers=layers,
                opacity=opacity,
                signature=signature,
                state=state,
            )

    def _configure_controls_from_metadata(self, metadata: VideoMetadata) -> None:
        self.frame_surface.set_dimensions(metadata.width, metadata.height)
        count = metadata.frame_count
        self._updating_widgets = True
        try:
            if count is not None and count > 0:
                self.frame_slider.end = max(0, int(count) - 1)
                self.frame_slider.disabled = False
                self.previous_button.disabled = False
                self.next_button.disabled = False
                self.play_button.disabled = not self._frame_access_available
            else:
                self.frame_slider.end = max(0, int(self.frame_slider.value))
                self.frame_slider.disabled = True
                self.previous_button.disabled = True
                self.next_button.disabled = True
                self.play_button.disabled = True
            self.refresh_button.disabled = not self._frame_access_available
            self.probe_button.disabled = not self._frame_access_available
            self._configure_display_channel_options(metadata)
        finally:
            self._updating_widgets = False

    def _sync_surface_from_temporal_state(self, state: Any) -> None:
        if self._disposed:
            return
        self._updating_surface = True
        try:
            self.frame_surface.frame_count = max(0, int(state.frame_count or self._metadata.frame_count or 0))
            self.frame_surface.fps = max(0.0, float(state.fps or self._metadata.fps or 0.0))
            self.frame_surface.playback_rate = max(0.01, float(state.playback_rate or self.speed.value or 1.0))
            self.frame_surface.prefetch_count = self.prefetch_count
            self.frame_surface.frame_index = max(0, int(state.frame_index or 0))
            self.frame_surface.client_clock = bool(self._direct_mode_active)
            self.frame_surface.playing = bool(state.playing)
        finally:
            self._updating_surface = False

    def _sync_widgets_from_temporal_state(self) -> None:
        if self._disposed:
            return
        state = self.temporal.get_state()
        if state.dataset_id != self._current_dataset_id or state.row_id != self._current_row_id:
            return
        self._updating_widgets = True
        try:
            if not self.frame_slider.disabled:
                value = max(self.frame_slider.start, min(int(state.frame_index), self.frame_slider.end))
                self.frame_slider.value = value
            self.play_button.name = "Pause" if state.playing else "Play"
            self.play_button.icon = "player-pause" if state.playing else "player-play"
        finally:
            self._updating_widgets = False

    def _sync_from_temporal_state(self) -> None:
        if self._disposed:
            return
        state = self.temporal.get_state()
        if state.dataset_id != self._current_dataset_id or state.row_id != self._current_row_id:
            return
        self._sync_widgets_from_temporal_state()

        self._sync_surface_from_temporal_state(state)
        if state.playing and not self._direct_mode_active:
            self._ensure_periodic()
        else:
            self._stop_periodic()

        # Browser-direct frame sequences own frame rendering/read-ahead while active.
        # The managed fallback remains one-frame-at-a-time with JobManager
        # backpressure and never accumulates stale frame requests.
        if not self._direct_mode_active:
            self._request_current_frame(force=False)

    def _on_browser_image_info_changed(self, event: Any) -> None:
        """Promote verified browser PNG channel metadata into the normal video contract."""
        if self._disposed or not self._direct_mode_active:
            return
        if not self._current_dataset_id or self._current_row_id is None:
            return
        info = dict(event.new or {})
        mode = str(info.get("source_image_mode") or "")
        count = int(info.get("source_channel_count") or 0)
        if (mode, count) not in {("RGB", 3), ("RGBA", 4)}:
            return
        remember = getattr(self.resolver, "remember_browser_frame_info", None)
        if callable(remember):
            remember(self._current_dataset_id, self._current_row_id,
                     source_mode=mode, channel_count=count)
        bands = ["R", "G", "B"] + (["A"] if count == 4 else [])
        self._metadata = replace(self._metadata, extra={
            **dict(self._metadata.extra or {}), "source_image_mode": mode,
            "source_channel_count": count, "source_image_bands": bands,
        })
        self._updating_widgets = True
        try:
            self._configure_display_channel_options(self._metadata)
        finally:
            self._updating_widgets = False
        self._update_metadata_display()

    def _on_surface_frame_changed(self, event: Any) -> None:
        if self._disposed or self._updating_surface or not self._direct_mode_active:
            return
        if self._current_dataset_id is None or self._current_row_id is None:
            return
        state = self.temporal.get_state()
        frame_index = max(0, int(event.new or 0))
        if not bool(state.playing):
            # Paused seeks originate from this panel's slider/step controls, so Python
            # temporal state is authoritative. Browser presentation feedback may arrive
            # later after image decode and must never pull the paused frame backwards.
            return
        if int(state.frame_index) == frame_index:
            return
        self.temporal.set_frame(
            self._current_dataset_id,
            self._current_row_id,
            frame_index,
            fps=self._metadata.fps or state.fps,
            frame_count=self._metadata.frame_count or state.frame_count,
            origin=self.panel_id,
            panel_id=self.panel_id,
        )

    def _on_surface_playing_changed(self, event: Any) -> None:
        if self._disposed or self._updating_surface or not self._direct_mode_active:
            return
        state = self.temporal.get_state()
        playing = bool(event.new)
        if bool(state.playing) == playing:
            return
        self.temporal.set_playing(
            playing,
            playback_rate=float(self.speed.value or state.playback_rate or 1.0),
            origin=self.panel_id,
            panel_id=self.panel_id,
        )
        if not playing and state.frame_count and int(self.frame_surface.frame_index) >= int(state.frame_count) - 1:
            self.status.alert_type = "success"
            self.status.object = "Playback complete."

    def _on_surface_error(self, event: Any) -> None:
        if self._disposed or self._updating_surface or not self._direct_mode_active:
            return
        message = str(event.new or "").strip()
        if not message:
            return

        frame_index = max(0, int(getattr(self.frame_surface, "error_frame", -1) or 0))
        self._direct_mode_active = False
        self._direct_mode_disabled = True
        self._stop_periodic()
        self._updating_surface = True
        try:
            # Keep the last successfully displayed browser frame in place until
            # the managed fallback has decoded the requested frame.
            self.frame_surface.client_clock = False
            self.frame_surface.playing = False
        finally:
            self._updating_surface = False

        current = self.temporal.get_state()
        self.status.alert_type = "warning"
        self.status.object = (
            f"Browser-buffered frame {frame_index:,} could not be displayed; "
            "falling back to the managed core.video resolver."
        )
        self._update_metadata_display()
        # Start the managed request first. If stopping playback publishes a
        # temporal event synchronously, that subsequent normal request will see
        # the same pending frame and deduplicate rather than racing this fallback.
        self._request_current_frame(force=True)
        if current.playing:
            self.temporal.set_playing(
                False,
                origin=self.panel_id,
                panel_id=self.panel_id,
            )

    def _configure_display_channel_options(self, metadata: VideoMetadata) -> None:
        extra = dict(getattr(metadata, "extra", {}) or {})
        media_kind = str(extra.get("media_kind") or getattr(self._current_asset, "media_kind", "") or "")
        count = int(extra.get("source_channel_count") or 0)
        bands = [str(value) for value in (extra.get("source_image_bands") or ())]
        options: dict[str, str] = {
            "Auto": "auto",
            "Source/native": "source",
        }
        if count >= 3 or not count:
            options["RGB (channels 1–3)"] = "rgb"
        for index in range(1, count + 1):
            band = bands[index - 1] if index - 1 < len(bands) else str(index)
            suffix = f" ({band}/extra)" if index > 3 else f" ({band})"
            options[f"Channel {index}{suffix}"] = f"channel:{index}"
        current = str(self._display_mode or "auto")
        if current not in options.values():
            current = "auto"
            self._display_mode = current
        self.display_channels.options = options
        self.display_channels.value = current
        self.display_channels.disabled = media_kind != "frame_sequence"

    def _direct_pattern_for_asset(self, asset: Any) -> Optional[BrowserFrameSequencePattern]:
        if asset is None:
            return None
        direct = getattr(self.resolver, "browser_direct_frame_uri", None)
        transform_for = getattr(self.resolver, "browser_direct_display_transform", None)
        if not callable(direct):
            return browser_frame_sequence_pattern(asset)
        try:
            if not str(direct(asset, 0, display_mode=self._display_mode) or ""):
                return None
            transform = (
                str(transform_for(asset, display_mode=self._display_mode) or "source")
                if callable(transform_for)
                else "source"
            )
        except Exception:
            return None
        pattern = browser_frame_sequence_pattern(asset)
        if pattern is None:
            return None
        return replace(pattern, display_transform=transform)

    def _on_display_mode_changed(self, event: Any) -> None:
        if self._disposed or self._updating_widgets:
            return
        selected = str(event.new or "auto")
        self._display_mode = selected
        if self._current_dataset_id is None or self._current_row_id is None:
            return
        try:
            self._display_mode = self.resolver.set_display_mode(
                self._current_dataset_id,
                self._current_row_id,
                selected,
                origin=self.panel_id,
                panel_id=self.panel_id,
            )
        except Exception as exc:
            self._show_error(f"Could not change video display channels: {exc}")
            return
        self._direct_mode_active = False
        self._direct_mode_disabled = False
        self._stop_periodic()
        self._updating_surface = True
        try:
            self.frame_surface.client_clock = False
            self.frame_surface.playing = False
            self.frame_surface.generation += 1
        finally:
            self._updating_surface = False
        self._direct_pattern = self._direct_pattern_for_asset(self._current_asset)
        self._last_preview = None
        self._failed_frame_key = None
        self.status.alert_type = "info"
        self.status.object = "Applying video display-channel interpretation…"
        if self._direct_pattern is not None and not self._direct_mode_disabled:
            self._enable_direct_mode()
        else:
            self._request_current_frame(force=True)

    def _on_slider_changed(self, event: Any) -> None:
        if self._disposed or self._updating_widgets:
            return
        if self._current_dataset_id is None or self._current_row_id is None:
            return
        state = self.temporal.get_state()
        frame_index = int(event.new)
        if self._direct_mode_active:
            # Seek the browser surface directly before publishing shared state.
            # Local temporal feedback is intentionally not echoed back to the surface.
            self._updating_surface = True
            try:
                self.frame_surface.frame_index = frame_index
            finally:
                self._updating_surface = False
        elif state.playing:
            self._restart_play_clock(frame_index)
        new_state = self.temporal.set_frame(
            self._current_dataset_id,
            self._current_row_id,
            frame_index,
            fps=self._metadata.fps,
            frame_count=self._metadata.frame_count,
            origin=self.panel_id,
            panel_id=self.panel_id,
        )
        if not bool(new_state.playing):
            self.status.alert_type = "info"
            self.status.object = f"Paused at frame {int(new_state.frame_index):,}."

    def _on_speed_changed(self, event: Any) -> None:
        if self._disposed:
            return
        state = self.temporal.get_state()
        rate = float(event.new or 1.0)
        if self._direct_mode_active:
            self._updating_surface = True
            try:
                self.frame_surface.playback_rate = rate
            finally:
                self._updating_surface = False
        elif state.playing:
            self._restart_play_clock(state.frame_index)
            self._stop_periodic()
        self.temporal.set_playing(
            state.playing,
            playback_rate=rate,
            origin=self.panel_id,
            panel_id=self.panel_id,
        )
        if state.playing:
            mode = "browser-buffered" if self._direct_mode_active else "managed"
            self.status.alert_type = "success"
            self.status.object = f"Playing at {rate:g}× via {mode} frame playback."

    def _step(self, delta: int) -> None:
        if self._current_dataset_id is None or self._current_row_id is None:
            return
        if self._direct_mode_active:
            self._updating_surface = True
            try:
                self.frame_surface.playing = False
            finally:
                self._updating_surface = False
        self.temporal.set_playing(False, origin=self.panel_id, panel_id=self.panel_id)
        self._stop_periodic()
        state = self.temporal.advance(
            int(delta),
            loop=False,
            origin=self.panel_id,
            panel_id=self.panel_id,
        )
        if self._direct_mode_active:
            self._updating_surface = True
            try:
                self.frame_surface.frame_index = int(state.frame_index)
            finally:
                self._updating_surface = False
        self.status.alert_type = "info"
        self.status.object = f"Paused at frame {int(state.frame_index):,}."

    def _toggle_playback(self) -> None:
        state = self.temporal.get_state()
        if state.dataset_id is None or state.row_id is None:
            return
        playing = not state.playing
        if playing and state.frame_count and int(state.frame_index) >= int(state.frame_count) - 1:
            state = self.temporal.set_frame(
                state.dataset_id,
                state.row_id,
                0,
                fps=state.fps or self._metadata.fps,
                frame_count=state.frame_count or self._metadata.frame_count,
                origin=self.panel_id,
                panel_id=self.panel_id,
            )

        if self._direct_mode_active:
            # Stop/start the browser clock in the same server callback as the button
            # click.  Do not wait for the EventBus next-tick synchronization path.
            # VideoFrameSurface invalidates any unresolved presentation promise on
            # this playing transition, so a paused surface cannot publish a stale
            # playing frame afterwards.
            self._updating_surface = True
            try:
                if not playing:
                    # Use the most recent browser feedback known to Python as the
                    # shared pause position before the pause state is published.
                    surface_frame = max(0, int(self.frame_surface.frame_index))
                    if surface_frame != int(state.frame_index):
                        state = self.temporal.set_frame(
                            state.dataset_id,
                            state.row_id,
                            surface_frame,
                            fps=state.fps or self._metadata.fps,
                            frame_count=state.frame_count or self._metadata.frame_count,
                            origin=self.panel_id,
                            panel_id=self.panel_id,
                        )
                self.frame_surface.playing = playing
            finally:
                self._updating_surface = False
        elif playing:
            self._restart_play_clock(state.frame_index)

        new_state = self.temporal.set_playing(
            playing,
            playback_rate=float(self.speed.value or 1.0),
            origin=self.panel_id,
            panel_id=self.panel_id,
        )
        if not playing:
            self._stop_periodic()
        self.cache_frames_button.disabled = (
            playing or not self._direct_mode_active or self._direct_pattern is None
            or self._direct_pattern.display_transform not in
            {"rgb_unpremultiplied", "rgb_unpremultiplied_png"}
        )
        mode = "browser-buffered" if self._direct_mode_active else "managed"
        self.status.alert_type = "success" if playing else "info"
        self.status.object = (
            f"Playing at {float(self.speed.value or 1.0):g}× via {mode} frame playback."
            if playing
            else f"Paused at frame {int(new_state.frame_index):,}."
        )

    def _refresh_current_frame(self) -> None:
        if self._disposed:
            return
        state = self.temporal.get_state()
        if state.playing:
            self.temporal.set_playing(
                False,
                origin=self.panel_id,
                panel_id=self.panel_id,
            )
        if self._direct_mode_active:
            self._direct_mode_active = False
            self._updating_surface = True
            try:
                self.frame_surface.client_clock = False
                self.frame_surface.playing = False
            finally:
                self._updating_surface = False
        self._stop_periodic()
        self._request_current_frame(force=True)

    def _restart_play_clock(self, frame_index: int) -> None:
        self._play_anchor_frame = max(0, int(frame_index or 0))
        self._play_anchor_monotonic = time.monotonic()

    def _server_playback_period_ms(self) -> int:
        state = self.temporal.get_state()
        fps = float(state.fps or self._metadata.fps or self.preview_fps or 1.0)
        rate = max(0.01, float(state.playback_rate or 1.0))
        effective = max(1.0, min(float(self.preview_fps), fps * rate))
        return max(25, int(round(1000.0 / effective)))

    def _ensure_periodic(self) -> None:
        if self._direct_mode_active:
            self._stop_periodic()
            return
        if self._periodic is not None or self._disposed:
            return
        state = self.temporal.get_state()
        if self._play_anchor_monotonic is None:
            self._restart_play_clock(state.frame_index)
        try:
            self._periodic = pn.state.add_periodic_callback(
                self._play_tick,
                period=self._server_playback_period_ms(),
                start=True,
            )
        except Exception:
            self._periodic = None

    def _stop_periodic(self) -> None:
        periodic = self._periodic
        self._periodic = None
        if periodic is None:
            return
        stop = getattr(periodic, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass

    def _play_tick(self) -> None:
        if self._disposed:
            self._stop_periodic()
            return
        state = self.temporal.get_state()
        if not state.playing or self._direct_mode_active:
            self._stop_periodic()
            return
        if self._frame_job is not None:
            future = getattr(self._frame_job, "future", None)
            if future is not None and not future.done():
                return

        # Managed playback is readiness-driven, not wall-clock catch-up.  If one
        # frame takes longer than 1/fps to resolve, playback buffers at that frame and
        # advances exactly one source frame once the request is complete.  This avoids
        # the old freeze-then-jump behaviour and makes Pause authoritative immediately.
        target = int(state.frame_index) + 1
        if state.frame_count is not None and state.frame_count > 0:
            target = min(target, int(state.frame_count) - 1)
        if target <= int(state.frame_index):
            self.temporal.set_playing(
                False,
                origin=self.panel_id,
                panel_id=self.panel_id,
            )
            self.status.alert_type = "success"
            self.status.object = "Playback complete."
            return

        new_state = self.temporal.set_frame(
            state.dataset_id,
            state.row_id,
            target,
            fps=state.fps or self._metadata.fps,
            frame_count=state.frame_count or self._metadata.frame_count,
            origin=self.panel_id,
            panel_id=self.panel_id,
        )
        self._restart_play_clock(new_state.frame_index)
        if (
            new_state.frame_count is not None
            and new_state.frame_count > 0
            and int(new_state.frame_index) >= int(new_state.frame_count) - 1
        ):
            self.temporal.set_playing(
                False,
                origin=self.panel_id,
                panel_id=self.panel_id,
            )
            self.status.alert_type = "success"
            self.status.object = "Playback complete."

    def _prepare_browser_cache(self) -> None:
        """Pre-cache a bounded range of original video frames while paused."""
        if (not self._direct_mode_active or self._disposed
                or self.cache_frames_button.disabled):
            return
        self.cache_frames_button.disabled = True
        self.status.alert_type = "info"
        self.status.object = "Caching up to 60 upcoming frames in the browser…"
        self.frame_surface.prepare_browser_cache()

    def _on_cache_prepare(self, event: Any) -> None:
        if self._disposed or not self._direct_mode_active:
            return
        values = dict(event.new or {})
        stage = str(values.get("stage") or "")
        done = max(0, int(values.get("done") or 0))
        total = max(0, int(values.get("total") or 0))
        failed = max(0, int(values.get("failed") or 0))
        if stage == "cache_prepare_progress":
            self.status.object = f"Caching playback frames: {done:,}/{total:,} complete."
            return
        if stage == "cache_prepare_unavailable":
            self.cache_frames_button.disabled = True
            self.status.alert_type = "warning"
            self.status.object = "Persistent browser frame caching is unavailable in this browser."
            return
        if stage in {"cache_prepare_done", "cache_prepare_cancelled"}:
            if bool(self.temporal.get_state().playing):
                self.cache_frames_button.disabled = True
                return
            self.cache_frames_button.disabled = False
            self.status.alert_type = "success" if failed == 0 and stage == "cache_prepare_done" else "info"
            self.status.object = (
                f"Playback frame cache: {done:,}/{total:,} complete"
                f" ({failed:,} unavailable)."
                if stage == "cache_prepare_done" else
                f"Playback caching stopped after {done:,}/{total:,} frames."
            )

    def _enable_direct_mode(self) -> None:
        if self._direct_pattern is None or self._direct_mode_disabled:
            return
        state = self.temporal.get_state()
        self._direct_mode_active = True
        self.cache_frames_button.disabled = (
            self._direct_pattern.display_transform not in
            {"rgb_unpremultiplied", "rgb_unpremultiplied_png"}
            or bool(state.playing) or not bool(state.frame_count or self._metadata.frame_count)
        )
        self._stop_periodic()
        self._updating_surface = True
        try:
            self.frame_surface.configure_direct_sequence(
                self._direct_pattern,
                frame_index=int(state.frame_index),
                frame_count=state.frame_count or self._metadata.frame_count,
                fps=state.fps or self._metadata.fps,
                playback_rate=float(state.playback_rate or self.speed.value or 1.0),
                playing=bool(state.playing),
                prefetch_count=self.prefetch_count,
                dataset_id=str(self._current_dataset_id or ""),
                row_id=str(self._current_row_id or ""),
            )
        finally:
            self._updating_surface = False
        self._update_metadata_display()

    def _request_current_frame(self, *, force: bool) -> None:
        if self._disposed or not self._frame_access_available:
            return
        if self._current_dataset_id is None or self._current_row_id is None:
            return
        if self._direct_mode_active and not force:
            return

        state = self.temporal.get_state()
        frame_index = int(state.frame_index)
        dataset_id = str(self._current_dataset_id)
        row_id = str(self._current_row_id)
        frame_key = (dataset_id, row_id, frame_index)

        if not force:
            if (
                self._last_preview is not None
                and self._last_preview.asset.dataset_id == dataset_id
                and self._last_preview.asset.row_id == row_id
                and int(self._last_preview.requested_frame_index) == frame_index
            ):
                # A managed preview for the current frame can be promoted to the
                # browser-direct path without re-fetching it through Python.
                if self._direct_pattern is not None and not self._direct_mode_disabled:
                    self._enable_direct_mode()
                return
            if self._failed_frame_key == frame_key:
                return
            if self._pending_frame_key == frame_key and self._frame_job is not None:
                future = getattr(self._frame_job, "future", None)
                if future is None or not future.done():
                    return

        if force and self._failed_frame_key == frame_key:
            self._failed_frame_key = None

        self._request_seq += 1
        request_seq = self._request_seq
        self._cancel_job("_frame_job")
        self._pending_frame_key = frame_key
        if not state.playing:
            self.status.alert_type = "info"
            self.status.object = f"Loading frame {frame_index:,}…"

        def work(*, cancel_token: Any):
            return self.resolver.load_frame_for_row(
                dataset_id,
                row_id,
                frame_index=frame_index,
                max_size=self.max_size,
                display_mode=self._display_mode,
                cancel_token=cancel_token,
            )

        def done(preview: VideoFramePreview) -> None:
            if self._disposed or request_seq != self._request_seq:
                return
            self._frame_job = None
            if self._pending_frame_key == frame_key:
                self._pending_frame_key = None
            if self._failed_frame_key == frame_key:
                self._failed_frame_key = None
            self._last_preview = preview
            self._current_asset = preview.asset

            metadata_changed = self._metadata_signature(preview.metadata) != self._metadata_signature(self._metadata)
            display_metadata_changed = (
                self._display_metadata_signature(preview.metadata)
                != self._display_metadata_signature(self._metadata)
            )
            if metadata_changed or display_metadata_changed:
                self._metadata = preview.metadata
                self._configure_controls_from_metadata(self._metadata)
            candidate = self._direct_pattern_for_asset(preview.asset)
            if candidate != self._direct_pattern:
                self._direct_pattern = candidate
                self._direct_mode_active = False
                self._direct_mode_disabled = False

            self._updating_surface = True
            try:
                self.frame_surface.show_server_frame(
                    preview.data_uri,
                    frame_index=int(preview.requested_frame_index),
                    dataset_id=dataset_id,
                    row_id=row_id,
                )
                self.frame_surface.frame_count = max(0, int(self._metadata.frame_count or 0))
                self.frame_surface.fps = max(0.0, float(self._metadata.fps or 0.0))
                self.frame_surface.playback_rate = max(0.01, float(self.temporal.get_state().playback_rate or 1.0))
                self.frame_surface.playing = bool(self.temporal.get_state().playing)
            finally:
                self._updating_surface = False

            current_state = self.temporal.get_state()
            if not current_state.playing:
                self.status.alert_type = "success"
                self.status.object = (
                    f"Frame {preview.frame_index:,}"
                    + (f" · {preview.time_seconds:.3f}s" if preview.time_seconds is not None else "")
                )
            self._update_metadata_display()

            if self._direct_pattern is not None and not self._direct_mode_disabled:
                self._enable_direct_mode()
                if not current_state.playing:
                    self.status.alert_type = "success"
                    self.status.object += " · browser-buffered playback ready"
            elif current_state.playing:
                self._ensure_periodic()

        def error(exc: BaseException) -> None:
            if self._disposed or request_seq != self._request_seq:
                return
            self._frame_job = None
            if self._pending_frame_key == frame_key:
                self._pending_frame_key = None
            self._failed_frame_key = frame_key
            self._show_error(
                f"Could not load video frame {frame_index:,}: {exc}. "
                "Automatic retry is paused; use Refresh frame to retry explicitly."
            )
            current = self.temporal.get_state()
            if current.playing:
                self.temporal.set_playing(
                    False,
                    origin=self.panel_id,
                    panel_id=self.panel_id,
                )

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                done(work(cancel_token=None))
            except BaseException as exc:
                error(exc)
            return
        self._frame_job = jobs.submit(
            work,
            title=f"Loading video frame {frame_index}",
            key=f"core.video.frame:{id(self)}:{request_seq}",
            on_done=done,
            on_error=error,
        )

    def _probe_media(self) -> None:
        if self._disposed or self._current_dataset_id is None or self._current_row_id is None:
            return
        if not self._frame_access_available:
            self._show_error("Frame access is unavailable for the focused video row.")
            return
        current = self.temporal.get_state()
        if current.playing:
            self.temporal.set_playing(
                False,
                origin=self.panel_id,
                panel_id=self.panel_id,
            )
        if self._direct_mode_active:
            self._direct_mode_active = False
            self._updating_surface = True
            try:
                self.frame_surface.client_clock = False
                self.frame_surface.playing = False
            finally:
                self._updating_surface = False
        self._cancel_job("_probe_job")
        dataset_id = self._current_dataset_id
        row_id = self._current_row_id
        self.status.alert_type = "info"
        self.status.object = "Probing media metadata; remote media may be cached locally…"

        def work(*, cancel_token: Any):
            return self.resolver.probe_for_row(
                dataset_id,
                row_id,
                materialize_remote=True,
                cancel_token=cancel_token,
            )

        def done(metadata: VideoMetadata) -> None:
            if self._disposed or dataset_id != self._current_dataset_id or row_id != self._current_row_id:
                return
            self._probe_job = None
            self._metadata = metadata
            self._configure_controls_from_metadata(metadata)
            state = self.temporal.get_state()
            self.temporal.set_frame(
                dataset_id,
                row_id,
                state.frame_index,
                fps=metadata.fps,
                frame_count=metadata.frame_count,
                origin=self.panel_id,
                panel_id=self.panel_id,
            )
            try:
                asset = self.resolver.asset_for_row(dataset_id, row_id)
            except Exception:
                asset = None
            if asset is not None:
                self._current_asset = asset
                self._direct_pattern = self._direct_pattern_for_asset(asset)
            self._sync_surface_from_temporal_state(self.temporal.get_state())
            self._update_metadata_display()
            self.status.alert_type = "success"
            self.status.object = "Video metadata probe complete."
            self._request_current_frame(force=True)

        def error(exc: BaseException) -> None:
            if self._disposed:
                return
            self._probe_job = None
            self._show_error(f"Could not probe video media: {exc}")

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                done(work(cancel_token=None))
            except BaseException as exc:
                error(exc)
            return
        self._probe_job = jobs.submit(
            work,
            title="Probing video media",
            key=f"core.video.probe:{id(self)}:{dataset_id}:{row_id}",
            on_done=done,
            on_error=error,
        )

    def _apply_restored_frame(self) -> None:
        if self._restored_frame_index is None:
            return
        if self._current_dataset_id is None or self._current_row_id is None:
            return
        frame_index = self._restored_frame_index
        self._restored_frame_index = None
        self.temporal.set_frame(
            self._current_dataset_id,
            self._current_row_id,
            frame_index,
            fps=self._metadata.fps,
            frame_count=self._metadata.frame_count,
            origin="workspace.restore",
            panel_id=self.panel_id,
        )

    def _cancel_job(self, attribute: str) -> None:
        handle = getattr(self, attribute, None)
        setattr(self, attribute, None)
        if attribute == "_frame_job":
            self._pending_frame_key = None
        if handle is None:
            return
        try:
            handle.cancel()
        except Exception:
            pass

    def _surface_show_message(self, message: str) -> None:
        self._updating_surface = True
        try:
            self.frame_surface.client_clock = False
            self.frame_surface.playing = False
            self.frame_surface.mode = "server"
            self.frame_surface.source = ""
            self.frame_surface.message = str(message)
            self.frame_surface.error = ""
            self.frame_surface.error_frame = -1
            self.frame_surface.generation += 1
        finally:
            self._updating_surface = False

    def _show_empty(self, message: str) -> None:
        self._stop_periodic()
        self._cancel_job("_frame_job")
        self._direct_mode_active = False
        self._direct_mode_disabled = False
        self._direct_pattern = None
        self._surface_show_message(message)
        self.status.alert_type = "info"
        self.status.object = message
        self.meta.object = ""
        self._updating_widgets = True
        try:
            self.frame_slider.start = 0
            self.frame_slider.end = 0
            self.frame_slider.value = 0
            self.frame_slider.disabled = True
            self.previous_button.disabled = True
            self.next_button.disabled = True
            self.play_button.disabled = True
            self.refresh_button.disabled = True
            self.probe_button.disabled = True
            self.summary_columns_button.visible = False
            self.summary_columns_button.disabled = True
            self.summary_columns_button.name = "Summary unavailable"
        finally:
            self._updating_widgets = False
        self._set_summary_event_topic("")

    def _show_error(self, message: str) -> None:
        self.status.alert_type = "danger"
        self.status.object = str(message)
        if self._last_preview is None and not self._direct_mode_active:
            self._surface_show_message("Preview unavailable")

    @staticmethod
    def _display_metadata_signature(metadata: VideoMetadata) -> tuple[Any, ...]:
        extra = dict(getattr(metadata, "extra", {}) or {})
        return (
            extra.get("source_image_mode"),
            tuple(extra.get("source_image_bands") or ()),
            int(extra.get("source_channel_count") or 0),
            extra.get("display_mode_requested"),
            extra.get("display_mode_applied"),
        )

    @staticmethod
    def _metadata_signature(metadata: VideoMetadata) -> tuple[Any, ...]:
        # Per-frame provenance such as ``extra['frame_uri']`` intentionally does
        # not participate. Rebuilding controls/Markdown for that hot-path detail
        # caused avoidable Bokeh layout churn during playback.
        return (
            metadata.duration_seconds,
            metadata.fps,
            metadata.frame_count,
            metadata.width,
            metadata.height,
            metadata.thumbnail_uri,
        )

    def _update_metadata_display(self) -> None:
        asset = self._current_asset
        uri = str(getattr(asset, "uri", "") or "") if asset is not None else ""
        playback_note = (
            "browser-buffered HTTP(S) frame sequence"
            if self._direct_mode_active
            else "managed resolver preview"
        )
        self.meta.object = self._metadata_markdown(
            uri,
            self._metadata,
            asset=asset,
            playback_note=playback_note,
        )

    def _schedule(self, callback: Any) -> None:
        if self._disposed:
            return
        doc = getattr(pn.state, "curdoc", None)
        add_next_tick = getattr(doc, "add_next_tick_callback", None)
        if callable(add_next_tick):
            try:
                add_next_tick(callback)
                return
            except Exception:
                pass
        try:
            callback()
        except Exception:
            pass

    @staticmethod
    def _metadata_markdown(
        uri: str,
        metadata: VideoMetadata,
        *,
        asset: Any = None,
        playback_note: str = "",
    ) -> str:
        values = [
            f"**FPS:** {_fmt(metadata.fps)}",
            f"**Frames:** {_fmt_int(metadata.frame_count)}",
            f"**Duration:** {_fmt_seconds(metadata.duration_seconds)}",
            f"**Dimensions:** {_fmt_dimensions(metadata.width, metadata.height)}",
            f"**Metadata source:** `{metadata.source}`",
        ]
        if playback_note:
            values.append(f"**Playback:** `{playback_note}`")
        if asset is not None:
            kind = str(getattr(asset, "media_kind", "") or "")
            if kind:
                values.append(f"**Media kind:** `{kind}`")
            template = str(getattr(asset, "frame_uri_template", "") or "")
            if template:
                values.append(f"**Frame template:** `{template}`")
        extra = dict(getattr(metadata, "extra", {}) or {})
        if extra.get("source_channel_count"):
            bands = ", ".join(str(value) for value in (extra.get("source_image_bands") or ()))
            values.append(
                f"**Source channels:** {int(extra.get('source_channel_count') or 0)}"
                + (f" (`{bands}`)" if bands else "")
            )
            values.append(
                f"**Display interpretation:** `{extra.get('display_mode_applied') or extra.get('display_mode_requested') or 'auto'}`"
            )
        if uri:
            values.append(f"**Media:** `{uri}`")
        return "  \n".join(values)

def _fmt(value: Optional[float]) -> str:
    return "Unknown" if value is None else f"{float(value):.3f}".rstrip("0").rstrip(".")

def _fmt_int(value: Optional[int]) -> str:
    return "Unknown" if value is None else f"{int(value):,}"

def _fmt_seconds(value: Optional[float]) -> str:
    return "Unknown" if value is None else f"{float(value):.3f}s"

def _fmt_dimensions(width: Optional[int], height: Optional[int]) -> str:
    if width is None or height is None:
        return "Unknown"
    return f"{int(width):,} × {int(height):,}"
