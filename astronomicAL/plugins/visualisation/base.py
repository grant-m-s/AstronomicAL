from __future__ import annotations

import uuid

from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import OrderedDict

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

from .constants import (
    INTERNAL_ROW_ID,
    INTERNAL_X,
    INTERNAL_Y,
    PLOT_MIN_HEIGHT,
    SETTINGS_HEIGHT,
)
from .utils import (
    PreparedFrame,
    _active_dataset_id,
    _active_df,
    ensure_hv_extension,
    force_wheel_zoom_hook,
    prepare_plot_frame,
    prepared_cache_key,
)
from .widgets import (
    header_select,
    settings_box,
    settings_checkbox,
    settings_float_slider,
    settings_int_input,
    settings_multichoice,
    settings_select,
)

import time


class BaseVisualisationPanel(param.Parameterized):
    """Lifecycle-aware base class for visualisation plugin panels."""

    title = "Visualisation"

    def __init__(
        self,
        context,
        state,
        *,
        show_controls: bool = True,
        show_header: bool = True,
        **params,
    ):
        ensure_hv_extension()
        super().__init__(**params)

        self.context = context
        self.state = state
        self.show_controls = show_controls
        self.show_header = show_header

        self.panel_id = str(uuid.uuid4())
        self._subs: List[Any] = []
        self._watchers: List[Tuple[Any, Any]] = []
        self._stream_watchers: List[Tuple[Any, Any]] = []
        self._disposed = False
        self._refresh_scheduled = False
        self._refresh_request_count = 0
        self._last_refresh_requested_at = None
        self._last_refresh_reason = None


        self.settings_visible = False
        self._layout: Optional[pn.Column] = None
        self._settings_built = False

        self._prepared_cache: OrderedDict[Tuple[Any, ...], PreparedFrame] = OrderedDict()
        self._prepared_cache_limit = 6
        self._suppress_state_refresh = False
        
        self._row_index_cache_key = None
        self._row_index_cache = None
        
        self._interactive_sample_cache = {}

        self._last_x_range = None
        self._last_y_range = None

        self.plot_pane = pn.pane.HoloViews(
            sizing_mode="stretch_both",
            height_policy="max",
            min_height=PLOT_MIN_HEIGHT,
            margin=(0, 0, 0, 0),
            styles={"min-height": "0"},
        )

        self.status_pane = pn.pane.HTML(
            "",
            width=230,
            height=30,
            margin=(2, 8, 0, 0),
        )

        self.settings_pane = pn.Column(
            sizing_mode="stretch_width",
            height=SETTINGS_HEIGHT,
            min_height=SETTINGS_HEIGHT,
            max_height=SETTINGS_HEIGHT,
            height_policy="fixed",
            visible=False,
            margin=(0, 0, 0, 0),
            styles={
                "height": f"{SETTINGS_HEIGHT}px",
                "min-height": f"{SETTINGS_HEIGHT}px",
                "max-height": f"{SETTINGS_HEIGHT}px",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "box-sizing": "border-box",
            },
        )

        self.settings_button = pn.widgets.Button(
            name="⚙",
            width=32,
            height=32,
            button_type="light",
            margin=(14, 0, 0, 0),
            sizing_mode="fixed",
        )
        self.settings_button.on_click(self._toggle_settings)

        for topic in (
            "dataset.loaded",
            "dataset.updated",
            "dataset.active.changed",
            "dataset.mapping_updated",
            "labels.settings.updated",
        ):
            self._subscribe(topic, self._on_dataset_event)

        for topic in (
            "selection.focus.changed",
            "selection.focus.cleared",
            "selection.set.changed",
            "selection.set.cleared",
        ):
            self._subscribe(topic, self._on_selection_event)

        self._watch_state(
            [
                "x",
                "y",
                "label_filter",
                "color_by",
                "render_mode",
                "datashade_threshold",
                "interactive_sample_limit",
                "max_selection_ids",
                "point_size",
                "point_alpha",
                "log_x",
                "log_y",
                "bins",
                "density",
                "cumulative",
                "log_density",
            ]
        )

    def _ensure_settings_built(self) -> None:
        if self._settings_built:
            return

        self.settings_pane[:] = [self._settings_controls()]
        self._settings_built = True

    def _apply_settings_visibility(self) -> None:
        self._ensure_settings_built()
        self.settings_pane.visible = self.settings_visible
        self.settings_button.button_type = "primary" if self.settings_visible else "light"

    def _toggle_settings(self, _event=None) -> None:
        self.settings_visible = not self.settings_visible
        self._apply_settings_visibility()

    def _schedule_refresh(self, reason: str = "unknown") -> None:
        self._refresh_request_count += 1

        now = time.perf_counter()

        if self._refresh_scheduled:
            print(
                "[AstronomicAL visualisation] refresh already scheduled; "
                f"skipping duplicate request reason={reason!r}",
                flush=True,
            )
            return

        self._refresh_scheduled = True
        self._last_refresh_requested_at = now
        self._last_refresh_reason = reason

        print(
            "[AstronomicAL visualisation] scheduling refresh "
            f"reason={reason!r}",
            flush=True,
        )

        try:
            import panel as pn

            pn.state.curdoc.add_next_tick_callback(self._run_scheduled_refresh)
        except Exception:
            self._run_scheduled_refresh()

    def _run_scheduled_refresh(self) -> None:
        queued_for = None

        if self._last_refresh_requested_at is not None:
            queued_for = time.perf_counter() - self._last_refresh_requested_at

        reason = self._last_refresh_reason

        self._refresh_scheduled = False
        self._last_refresh_requested_at = None
        self._last_refresh_reason = None

        print(
            "[AstronomicAL visualisation] running scheduled refresh "
            f"reason={reason!r} "
            f"queued_for={queued_for:.2f}s" if queued_for is not None
            else "[AstronomicAL visualisation] running scheduled refresh",
            flush=True,
        )

        self.refresh()

    def _subscribe(self, topic: str, callback) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        try:
            sub = events.subscribe(
                topic,
                callback,
                owner_id=self.panel_id,
                owner_label=self.__class__.__name__,
                owner_kind="plugin-panel",
            )
        except TypeError:
            sub = events.subscribe(topic, callback)

        self._subs.append(sub)

    def _watch_state(self, names: Sequence[str]) -> None:
        for name in names:
            try:
                watcher = self.state.param.watch(self._on_state_changed, name)
                self._watchers.append((self.state, watcher))
            except Exception:
                pass

    def _watch_param(self, owner: Any, callback, parameter_name: str, *, render_scoped: bool = False) -> None:
        try:
            watcher = owner.param.watch(callback, parameter_name)
            if render_scoped:
                self._stream_watchers.append((owner, watcher))
            else:
                self._watchers.append((owner, watcher))
        except Exception:
            pass

    def _clear_stream_watchers(self) -> None:
        for owner, watcher in list(self._stream_watchers):
            try:
                owner.param.unwatch(watcher)
            except Exception:
                pass
        self._stream_watchers.clear()

    def _clear_prepared_cache(self) -> None:
        try:
            self._prepared_cache.clear()
        except Exception:
            self._prepared_cache = OrderedDict()

        self._row_index_cache_key = None
        self._row_index_cache = None

    def dispose(self) -> None:
        if self._disposed:
            return

        self._disposed = True

        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in list(self._subs):
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass
        self._subs.clear()

        self._clear_stream_watchers()

        for owner, watcher in list(self._watchers):
            try:
                owner.param.unwatch(watcher)
            except Exception:
                pass
        self._watchers.clear()

        self._clear_prepared_cache()

    def get_state(self) -> Dict[str, Any]:
        state = self.state.get_state()
        state["settings_visible"] = self.settings_visible
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        if isinstance(state, dict):
            self.settings_visible = bool(state.get("settings_visible", False))
            self.state.restore_state(state)
            self._clear_prepared_cache()
            self.refresh()
            self._apply_settings_visibility()

    def _df(self):
        return _active_df(self.context)

    def _dataset_id(self) -> Optional[str]:
        return _active_dataset_id(self.context)

    def _plot_data(self, *, require_y: bool) -> PreparedFrame:
        key = prepared_cache_key(self.context, self.state, require_y=require_y)

        cached = self._prepared_cache.get(key)
        if cached is not None:
            try:
                self._prepared_cache.move_to_end(key)
            except Exception:
                pass
            return cached

        data = prepare_plot_frame(self.context, self.state, require_y=require_y)

        self._prepared_cache[key] = data

        try:
            self._prepared_cache.move_to_end(key)
        except Exception:
            pass

        while len(self._prepared_cache) > int(self._prepared_cache_limit):
            try:
                self._prepared_cache.popitem(last=False)
            except TypeError:
                first_key = next(iter(self._prepared_cache))
                self._prepared_cache.pop(first_key, None)

        return data

    def _row_index_for(self, data: PreparedFrame):
        """Return a cached pandas Index for fast row-id lookup.

        Selection overlays should not repeatedly do full-array string isin()
        checks over a million rows. Build one index per prepared frame and use
        get_indexer() for selected/focused row IDs.
        """
        if data.empty or INTERNAL_ROW_ID not in data.frame.columns:
            return None

        key = (id(data.frame), len(data.frame))

        if self._row_index_cache_key == key and self._row_index_cache is not None:
            return self._row_index_cache

        try:
            index = pd.Index(data.frame[INTERNAL_ROW_ID].astype(str), copy=False)
        except Exception:
            return None

        self._row_index_cache_key = key
        self._row_index_cache = index
        return index

    def _rows_for_row_ids(
        self,
        data: PreparedFrame,
        row_ids,
        *,
        visible_only: bool = False,
        limit: Optional[int] = None,
    ):
        """Return rows matching row IDs using cached direct lookup.

        This is much faster than np.isin over the whole frame for every
        selection update.
        """
        if data.empty or not row_ids or INTERNAL_ROW_ID not in data.frame.columns:
            return data.frame.iloc[0:0]

        row_ids = [str(row_id) for row_id in row_ids if row_id is not None]
        if not row_ids:
            return data.frame.iloc[0:0]

        if limit is not None and len(row_ids) > int(limit):
            row_ids = row_ids[: int(limit)]

        index = self._row_index_for(data)

        try:
            if index is None or not index.is_unique:
                raise ValueError("row-id index unavailable or non-unique")

            positions = index.get_indexer(row_ids)
            positions = positions[positions >= 0]

            if len(positions) == 0:
                return data.frame.iloc[0:0]

            sub = data.frame.iloc[positions]
        except Exception:
            # Safe fallback for duplicate IDs or unusual index behaviour.
            selected = set(row_ids)
            mask = data.frame[INTERNAL_ROW_ID].astype(str).isin(selected)
            sub = data.frame.loc[mask]

        if visible_only and not sub.empty:
            if self._last_x_range is not None and INTERNAL_X in sub.columns:
                x0, x1 = self._last_x_range
                x = sub[INTERNAL_X].to_numpy(copy=False)
                sub = sub.loc[(x >= x0) & (x <= x1)]

            if self._last_y_range is not None and INTERNAL_Y in sub.columns:
                y0, y1 = self._last_y_range
                y = sub[INTERNAL_Y].to_numpy(copy=False)
                sub = sub.loc[(y >= y0) & (y <= y1)]

        return sub

    def _focus_point(self, data: PreparedFrame) -> Optional[Tuple[float, Optional[float]]]:
        selection = getattr(self.context, "selection", None)
        if selection is None or data.empty:
            return None

        try:
            focus = selection.get_focus()
        except Exception:
            return None

        if focus is None or getattr(focus, "dataset_id", None) != self._dataset_id():
            return None

        row_id = str(getattr(focus, "row_id", ""))
        if not row_id:
            return None

        row = self._rows_for_row_ids(
            data,
            [row_id],
            visible_only=False,
            limit=1,
        )

        if row.empty:
            return None

        x = float(row.iloc[0][INTERNAL_X])
        y = None
        if INTERNAL_Y in row.columns:
            y = float(row.iloc[0][INTERNAL_Y])

        return x, y

    def _active_selection_ids(self) -> List[str]:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return []

        try:
            active = selection.get_active_set()
        except Exception:
            return []

        if active is None or getattr(active, "dataset_id", None) != self._dataset_id():
            return []

        return [str(row_id) for row_id in list(getattr(active, "row_ids", []) or [])]

    def _selection_points(self, data: PreparedFrame):
        ids = self._active_selection_ids()
        if not ids or data.empty or INTERNAL_Y not in data.frame.columns:
            return None

        sub = self._rows_for_row_ids(
            data,
            ids,
            visible_only=False,
            limit=int(self.state.max_selection_ids),
        )

        if sub.empty:
            return None

        return hv.Points(
            sub,
            kdims=[INTERNAL_X, INTERNAL_Y],
        ).opts(
            marker="circle",
            size=max(float(self.state.point_size) + 4, 8),
            fill_alpha=0.0,
            line_color="orange",
            line_width=2,
            active_tools=[],
            hooks=[force_wheel_zoom_hook],
            logx=self.state.log_x,
            logy=self.state.log_y,
            shared_axes=False,
            axiswise=True,
            framewise=True,
            **self._current_range_opts(include_y=True),
        )

    def _focus_overlay(self, data: PreparedFrame, *, size: float = 14):
        point = self._focus_point(data)
        if point is None:
            return None

        x, y = point
        if y is None:
            return None

        return hv.Points(
            [(x, y)],
            kdims=[INTERNAL_X, INTERNAL_Y],
        ).opts(
            marker="circle",
            size=size,
            fill_alpha=0.0,
            line_color="black",
            line_width=3,
            active_tools=[],
            hooks=[force_wheel_zoom_hook],
            logx=self.state.log_x,
            logy=self.state.log_y,
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

    def _on_dataset_event(self, topic, payload) -> None:
        t0 = time.perf_counter()

        print(
            f"[AstronomicAL visualisation] dataset event received: {topic}",
            flush=True,
        )

        self._suppress_state_refresh = True

        try:
            if topic == "labels.settings.updated":
                self.state.apply_label_settings(payload)
            else:
                self.state.refresh_from_context()
        finally:
            self._suppress_state_refresh = False

        t1 = time.perf_counter()

        self._clear_prepared_cache()

        print(
            "[AstronomicAL visualisation] dataset event state refresh complete "
            f"topic={topic} "
            f"duration={t1 - t0:.2f}s",
            flush=True,
        )

        self._schedule_refresh(reason=f"dataset_event.{topic}")

    def _payload_value(self, payload, key: str, default=None):
        """Read a value from dict-like or object-like event payloads."""
        if payload is None:
            return default

        if isinstance(payload, dict):
            if key in payload:
                return payload.get(key)

            metadata = payload.get("metadata")
            if isinstance(metadata, dict) and key in metadata:
                return metadata.get(key)

            return default

        if hasattr(payload, key):
            return getattr(payload, key)

        metadata = getattr(payload, "metadata", None)
        if isinstance(metadata, dict) and key in metadata:
            return metadata.get(key)

        return default

    def _normalise_range(self, value):
        if value is None or len(value) != 2:
            return None

        lo, hi = value
        if lo is None or hi is None:
            return None

        try:
            lo = float(lo)
            hi = float(hi)
        except Exception:
            return None

        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return None

        return (min(lo, hi), max(lo, hi))

    def _remember_ranges(self, x_range=None, y_range=None) -> bool:
        """Remember the current plot ranges.

        Returns True if either remembered range changed.
        """
        changed = False

        x_range = self._normalise_range(x_range)
        y_range = self._normalise_range(y_range)

        if x_range is not None and x_range != self._last_x_range:
            self._last_x_range = x_range
            changed = True

        if y_range is not None and y_range != self._last_y_range:
            self._last_y_range = y_range
            changed = True

        return changed

    def _current_range_opts(self, *, include_y: bool = True) -> Dict[str, Any]:
        opts: Dict[str, Any] = {}

        if self._last_x_range is not None:
            opts["xlim"] = self._last_x_range

        if include_y and self._last_y_range is not None:
            opts["ylim"] = self._last_y_range

        return opts

    def _on_selection_event(self, topic, payload) -> None:
        event_panel_id = self._payload_value(payload, "panel_id")
        if event_panel_id is not None and str(event_panel_id) == str(self.panel_id):
            return

        self._schedule_refresh()

    def _on_state_changed(self, event) -> None:
        if getattr(self, "_suppress_state_refresh", False):
            print(
                "[AstronomicAL visualisation] suppressed state refresh "
                f"param={getattr(event, 'name', None)!r}",
                flush=True,
            )
            return

        name = getattr(event, "name", None)

        if name in {"x", "y", "log_x", "log_y"}:
            try:
                self._last_x_range = None
                self._last_y_range = None
            except Exception:
                pass

            try:
                self._interactive_sample_cache.clear()
            except Exception:
                pass

            try:
                self._row_id_lookup_cache.clear()
            except Exception:
                pass

            try:
                self._interactive_current_frame = None
            except Exception:
                pass

            print(
                "[AstronomicAL visualisation] reset plot ranges after axis change "
                f"param={name!r}",
                flush=True,
            )

        self._schedule_refresh(reason=f"state.{name or 'unknown'}")

    def refresh(self) -> None:
        t0 = time.perf_counter()

        print(
            f"[AstronomicAL visualisation] refresh start panel={type(self).__name__}",
            flush=True,
        )

        try:
            self._render()
        finally:
            print(
                f"[AstronomicAL visualisation] refresh end panel={type(self).__name__} "
                f"duration={time.perf_counter() - t0:.2f}s",
                flush=True,
            )

    def _render(self) -> None:
        raise NotImplementedError

    def _empty(self, message: str):
        return hv.Text(0.5, 0.5, message).opts(
            xlim=(0, 1),
            ylim=(0, 1),
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            toolbar=None,
            xaxis=None,
            yaxis=None,
            show_frame=False,
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

    def _base_opts(
        self,
        *,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        tools: Optional[List[str]] = None,
        active_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        opts = dict(
            xlabel=str(xlabel if xlabel is not None else self.state.x),
            ylabel=str(ylabel if ylabel is not None else self.state.y),
            logx=self.state.log_x,
            logy=self.state.log_y,
            tools=tools or ["pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=active_tools or ["wheel_zoom"],
            hooks=[force_wheel_zoom_hook],
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            show_grid=True,
            framewise=True,
            axiswise=True,
            shared_axes=False,
            toolbar="right",
        )

        opts.update(self._current_range_opts(include_y=True))
        return opts

    def _settings_controls(self):
        return settings_box(
            self.status_pane,
            settings_select(self.state.param.color_by, name="Colour", width=130),
            settings_multichoice(self.state.param.label_filter, name="Labels", width=210),
            settings_select(self.state.param.render_mode, name="Render", width=130),
            settings_int_input(self.state.param.datashade_threshold, name="Shade threshold", width=145),
            settings_int_input(self.state.param.interactive_sample_limit, name="Sample limit", width=130),
            settings_int_input(self.state.param.max_selection_ids, name="Max selected IDs", width=145),
            settings_float_slider(self.state.param.point_size, name="Size", width=175),
            settings_float_slider(self.state.param.point_alpha, name="Alpha", width=175),
            settings_checkbox(self.state.param.log_x, name="Log X"),
            settings_checkbox(self.state.param.log_y, name="Log Y"),
        )

    def _header(self):
        return pn.GridBox(
            header_select(self.state.param.x, name="X"),
            header_select(self.state.param.y, name="Y"),
            self.settings_button if self.show_controls else pn.Spacer(width=34, height=34),
            ncols=3,
            sizing_mode="stretch_width",
            height=48,
            margin=(0, 0, 0, 0),
            styles={
                "display": "grid",
                "grid-template-columns": "minmax(70px, 1fr) minmax(70px, 1fr) 34px",
                "gap": "4px",
                "align-items": "start",
            },
        )

    def _layout_children(self):
        children = []
        if self.show_header:
            children.append(self._header())
        if self.show_controls:
            children.append(self.settings_pane)
        children.append(self.plot_pane)
        return children

    def panel(self):
        self.refresh()

        if not self.show_header and not self.show_controls:
            return self.plot_pane

        self._ensure_settings_built()
        self._apply_settings_visibility()

        self._layout = pn.Column(
            *self._layout_children(),
            sizing_mode="stretch_both",
            height_policy="max",
            min_height=0,
            margin=(0, 0, 0, 0),
            styles={
                "min-height": "0",
                "overflow": "hidden",
            },
        )
        return self._layout