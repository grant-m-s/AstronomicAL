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
    _load_row_ids_array,
)
from .widgets import (
    header_select,
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
        self._last_focus_payload = None

        self._interactive_sample_cache = {}
        self._interactive_sample_cache_max = 30
        self._last_interactive_frame_id = None

        self._prepared_extent_cache_key = None
        self._prepared_extent_cache = None

        self._base_sample_cache = {}
        self._base_sample_cache_max = 16

        self._near_full_range_keys = set()
        self._near_full_range_keys_max = 64

        self._focus_point_cache = OrderedDict()
        self._focus_point_cache_max = 512

        self._async_focus_pending = set()
        self._async_focus_failed = set()

        # Never do synchronous source-backed focus lookup for large prepared frames.
        self._focus_source_lookup_row_limit = 500_000
        self._focus_frame_scan_limit = 500_000

        self.settings_visible = False
        self._layout: Optional[pn.Column] = None
        self._settings_built = False

        self._prepared_cache: OrderedDict[Tuple[Any, ...], PreparedFrame] = OrderedDict()
        self._prepared_cache_limit = 6
        self._shared_prepared_cache = self._get_shared_prepared_cache()
        self._suppress_state_refresh = False

        self._focus_point_cache = OrderedDict()
        self._focus_point_cache_max = 256
        self._focus_frame_scan_limit = 500_000
        
        self._allow_one_full_range_skip = False

        self._row_index_cache_key = None
        self._row_index_cache = None
        
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
            height=75,
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


    def _shared_visualisation_data_cache(self):
        services = getattr(self.context, "services", None)

        if services is None:
            return None

        for key in (
            "core.visualisation.data_cache",
            "visualisation.data_cache",
            "data_cache",
        ):
            try:
                cache = services.get(key)
                if cache is not None:
                    return cache
            except Exception:
                pass

        return None


    def _shared_focus_rows(self):
        cache = self._shared_visualisation_data_cache()

        if cache is None:
            return None

        store = getattr(cache, "focus_rows", None)

        if store is None:
            store = OrderedDict()
            try:
                cache.focus_rows = store
            except Exception:
                return None

        return store


    def _shared_focus_row_get(self, key):
        store = self._shared_focus_rows()

        if store is None:
            return None

        value = store.get(key)

        if value is not None and hasattr(store, "move_to_end"):
            store.move_to_end(key)

        return value


    def _shared_focus_row_set(self, key, value, *, max_items: int = 256):
        store = self._shared_focus_rows()

        if store is None:
            return

        if key in store:
            store.pop(key, None)

        store[key] = value

        while len(store) > max_items:
            store.popitem(last=False)


    def _focus_point_cache_get(self, key):
        value = self._focus_point_cache.get(key)

        if value is not None:
            self._focus_point_cache.move_to_end(key)

        return value


    def _focus_point_cache_set(self, key, value):
        if key in self._focus_point_cache:
            self._focus_point_cache.pop(key, None)

        self._focus_point_cache[key] = value

        while len(self._focus_point_cache) > self._focus_point_cache_max:
            self._focus_point_cache.popitem(last=False)

    def _focus_axis_value_from_row(
        self,
        row_df,
        column: str,
        *,
        log_enabled: bool,
        axis_name: str,
        row_id: str,
    ):
        try:
            value = float(row_df.iloc[0][column])
        except Exception as exc:
            print(
                "[AstronomicAL visualisation] focus axis value unreadable "
                f"row_id={row_id!r} axis={axis_name} column={column!r} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None

        if not np.isfinite(value):
            print(
                "[AstronomicAL visualisation] focus row not drawable on this panel "
                f"row_id={row_id!r} "
                f"axis={axis_name} "
                f"column={column!r} "
                f"reason=non_finite "
                f"value={value!r}",
                flush=True,
            )
            return None

        if log_enabled and value <= 0:
            print(
                "[AstronomicAL visualisation] focus row not drawable on this panel "
                f"row_id={row_id!r} "
                f"axis={axis_name} "
                f"column={column!r} "
                f"reason=invalid_for_log_axis "
                f"value={value!r}",
                flush=True,
            )
            return None

        # Important:
        # Return raw data-space value. HoloViews/Bokeh log axes transform display,
        # but glyph coordinates are still raw data values.
        return value

    def _focus_point_from_cached_focus_row(self, focus):
        dataset_id = self._dataset_id()

        if focus is None or getattr(focus, "dataset_id", None) != dataset_id:
            return None

        row_id = str(getattr(focus, "row_id", "") or "")
        record_id_col = getattr(self.state, "record_id_col", None)
        x_col = getattr(self.state, "x", None)
        y_col = getattr(self.state, "y", None)

        if not row_id or not record_id_col or record_id_col == "Use Index":
            return None

        if not x_col or not y_col:
            return None

        point_key = (
            str(dataset_id),
            str(record_id_col),
            str(row_id),
            str(x_col),
            str(y_col),
            bool(getattr(self.state, "log_x", False)),
            bool(getattr(self.state, "log_y", False)),
        )

        cached_point = self._focus_point_cache_get(point_key)
        if cached_point is not None:
            return cached_point

        row_key = (
            "focus_row",
            str(dataset_id),
            str(record_id_col),
            str(row_id),
        )

        row_df = self._shared_focus_row_get(row_key)

        if row_df is None or getattr(row_df, "empty", True):
            print(
                "[AstronomicAL visualisation] focus row cache miss "
                f"row_id={row_id!r} x={x_col!r} y={y_col!r}",
                flush=True,
            )
            return None

        if x_col not in row_df.columns or y_col not in row_df.columns:
            print(
                "[AstronomicAL visualisation] focus row missing axis columns "
                f"row_id={row_id!r} x={x_col!r} x_present={x_col in row_df.columns} "
                f"y={y_col!r} y_present={y_col in row_df.columns}",
                flush=True,
            )
            return None

        x = self._focus_axis_value_from_row(
            row_df,
            x_col,
            log_enabled=bool(getattr(self.state, "log_x", False)),
            axis_name="x",
            row_id=row_id,
        )

        y = self._focus_axis_value_from_row(
            row_df,
            y_col,
            log_enabled=bool(getattr(self.state, "log_y", False)),
            axis_name="y",
            row_id=row_id,
        )

        if x is None or y is None:
            return None

        point = (x, y)
        self._focus_point_cache_set(point_key, point)
        return point

    def _schedule_async_focus_lookup(self, focus, data: PreparedFrame) -> None:
        """
        Schedule a shared focused-row lookup without blocking the UI.

        Multiple panels requesting the same focused row dedupe through JobManager's
        key. Late joiners attach their own on_done callback.
        """
        jobs = getattr(self.context, "jobs", None)

        if jobs is None:
            return

        dataset_id = self._dataset_id()
        row_id = str(getattr(focus, "row_id", "") or "")
        record_id_col = getattr(self.state, "record_id_col", None)

        if not dataset_id or not row_id:
            return

        if not record_id_col or record_id_col == "Use Index":
            return

        row_key = (
            "focus_row",
            str(dataset_id),
            str(record_id_col),
            str(row_id),
        )

        failed_key = (
            "focus_row_failed",
            str(dataset_id),
            str(record_id_col),
            str(row_id),
        )

        if self._shared_focus_row_get(row_key) is not None:
            return

        if self._shared_focus_row_get(failed_key) is True:
            return

        if row_key in self._async_focus_pending:
            return

        self._async_focus_pending.add(row_key)

        # Capture the current visual state so stale async results do not refresh
        # the wrong panel after axes or dataset have changed.
        request_signature = self._visual_state_signature()
        panel_id = self.panel_id

        job_key = (
            f"visualisation:focus-row:"
            f"{dataset_id}:{record_id_col}:{row_id}"
        )

        def _on_done(row_df):
            self._async_focus_pending.discard(row_key)

            if self._disposed:
                return

            if self.panel_id != panel_id:
                return

            if self._dataset_id() != dataset_id:
                return

            if self._visual_state_signature() != request_signature:
                # Axes/state changed while the job was running. Cache the row for
                # later use, but do not force this stale panel refresh.
                if row_df is not None and not getattr(row_df, "empty", True):
                    self._shared_focus_row_set(row_key, row_df)
                return

            if row_df is None or getattr(row_df, "empty", True):
                self._shared_focus_row_set(failed_key, True)
                return

            self._shared_focus_row_set(row_key, row_df)

            print(
                "[AstronomicAL visualisation] async focus row ready "
                f"panel={type(self).__name__} "
                f"row_id={row_id!r}",
                flush=True,
            )

            try:
                self._interactive_sample_cache.clear()
            except Exception:
                pass

            try:
                self._base_sample_cache.clear()
            except Exception:
                pass

            # Re-render now that the focus point can be resolved from cache.
            self._schedule_refresh(reason="focus.async_resolved")

        def _on_error(exc):
            self._async_focus_pending.discard(row_key)
            self._shared_focus_row_set(failed_key, True)

            print(
                "[AstronomicAL visualisation] async focus row failed "
                f"panel={type(self).__name__} "
                f"row_id={row_id!r} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        try:
            jobs.submit(
                self._fetch_focus_row_for_async_job,
                title="Resolve visualisation focus row",
                key=job_key,
                on_done=_on_done,
                on_error=_on_error,
                dataset_id=str(dataset_id),
                row_id=str(row_id),
                record_id_col=str(record_id_col),
            )
        except Exception as exc:
            self._async_focus_pending.discard(row_key)
            print(
                "[AstronomicAL visualisation] async focus submit failed "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def _fetch_focus_row_for_async_job(
        self,
        *,
        dataset_id: str,
        row_id: str,
        record_id_col: str,
        cancel_token=None,
    ):
        """
        Worker-thread function.

        Important:
        - Do not touch Panel/Bokeh objects here.
        - Do not mutate panel state here.
        - Only use DatasetManager/DatasetSource.
        """
        if cancel_token is not None and cancel_token.cancelled():
            return None

        datasets = getattr(self.context, "datasets", None)

        if datasets is None:
            return None

        try:
            source = datasets.get_source(dataset_id)
        except Exception:
            return None

        if source is None:
            return None

        row_id_str = str(row_id)

        # Prefer DatasetManager-level APIs if available.
        for method_name in ("get_row_by_id", "row_by_id", "find_row_by_id"):
            method = getattr(datasets, method_name, None)

            if not callable(method):
                continue

            for args, kwargs in (
                ((dataset_id, row_id_str, record_id_col), {}),
                ((dataset_id, row_id_str), {"id_column": record_id_col}),
                ((dataset_id, row_id_str), {}),
            ):
                if cancel_token is not None and cancel_token.cancelled():
                    return None

                try:
                    value = method(*args, **kwargs)
                except TypeError:
                    continue
                except Exception:
                    continue

                row_df = self._coerce_single_row_frame(value)
                if row_df is not None and not row_df.empty:
                    return row_df

        # Source-level APIs.
        for method_name in ("get_row_by_id", "row_by_id", "find_row_by_id"):
            method = getattr(source, method_name, None)

            if not callable(method):
                continue

            for args, kwargs in (
                ((row_id_str, record_id_col), {}),
                ((row_id_str,), {"id_column": record_id_col}),
                ((record_id_col, row_id_str), {}),
                ((row_id_str,), {}),
            ):
                if cancel_token is not None and cancel_token.cancelled():
                    return None

                try:
                    value = method(*args, **kwargs)
                except TypeError:
                    continue
                except Exception:
                    continue

                row_df = self._coerce_single_row_frame(value)
                if row_df is not None and not row_df.empty:
                    return row_df

        # Generic Parquet/DuckDB-backed filtered read.
        to_pandas = getattr(source, "to_pandas", None)

        if callable(to_pandas):
            quoted_id_col = '"' + str(record_id_col).replace('"', '""') + '"'

            for where_sql in (
                f"CAST({quoted_id_col} AS VARCHAR) = ?",
                f"{quoted_id_col} = ?",
            ):
                if cancel_token is not None and cancel_token.cancelled():
                    return None

                try:
                    # Do not pass columns here. We want one shared full row so all
                    # scatter panels can resolve their own x/y from the same result.
                    value = to_pandas(
                        where_sql=where_sql,
                        params=[row_id_str],
                        limit=1,
                    )
                except TypeError:
                    continue
                except Exception:
                    continue

                row_df = self._coerce_single_row_frame(value)
                if row_df is not None and not row_df.empty:
                    return row_df

        return None

    def _get_shared_prepared_cache(self):
        services = getattr(self.context, "services", None)
        if services is None:
            return None

        for key in (
            "core.visualisation.prepared_cache",
            "core.visualisation.cache",
            "visualisation.prepared_cache",
        ):
            try:
                service = services.get(key)
                if service is not None:
                    return service
            except Exception:
                pass

        return None

    def _prepared_cache_get(self, key):
        cache = getattr(self, "_shared_prepared_cache", None)

        if cache is not None and hasattr(cache, "get"):
            try:
                value = cache.get(key)
                print(
                    "[AstronomicAL visualisation prepared-cache] "
                    f"{'HIT' if value is not None else 'MISS'} shared key={key!r}",
                    flush=True,
                )
                return value
            except Exception as exc:
                print(
                    "[AstronomicAL visualisation prepared-cache] shared lookup failed "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        value = self._prepared_cache.get(key)
        print(
            "[AstronomicAL visualisation prepared-cache] "
            f"{'HIT' if value is not None else 'MISS'} local key={key!r}",
            flush=True,
        )
        return value


    def _prepared_cache_set(self, key, data):
        cache = getattr(self, "_shared_prepared_cache", None)

        if cache is not None and hasattr(cache, "set"):
            try:
                print(
                    f"[AstronomicAL visualisation prepared-cache] SET shared key={key!r}",
                    flush=True,
                )
                cache.set(key, data)
                return
            except Exception as exc:
                print(
                    "[AstronomicAL visualisation prepared-cache] shared set failed "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        print(
            f"[AstronomicAL visualisation prepared-cache] SET local key={key!r}",
            flush=True,
        )
        self._prepared_cache.clear()
        self._prepared_cache[key] = data

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

    def _visual_state_signature(self):
        return (
            getattr(self.state, "dataset_id", None),
            getattr(self.state, "x", None),
            getattr(self.state, "y", None),
            getattr(self.state, "record_id_col", None),
            getattr(self.state, "label_col", None),
            getattr(self.state, "color_by", None),
            getattr(self.state, "log_x", None),
            getattr(self.state, "log_y", None),
            tuple(getattr(self.state, "label_filter", []) or []),
        )

    def _schedule_refresh(self, *, reason: str = "unknown", delay_ms: Optional[int] = None):
        if delay_ms is None:
            if reason in {"state.x", "state.y", "state.color_by", "state.label_col"}:
                delay_ms = 300
            else:
                delay_ms = 0

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
        self._prepared_cache.clear()

        self._row_index_cache_key = None
        self._row_index_cache = None

        try:
            self._focus_point_cache.clear()
        except Exception:
            pass

        try:
            self._focus_point_cache.clear()
        except Exception:
            pass

        try:
            self._interactive_sample_cache.clear()
        except Exception:
            pass

        try:
            self._base_sample_cache.clear()
        except Exception:
            pass

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

        try:
            self._async_focus_pending.clear()
            self._async_focus_failed.clear()
        except Exception:
            pass

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
        self._last_prepared_cache_key = key
        print(
            "[AstronomicAL visualisation cache] lookup "
            f"panel={type(self).__name__} "
            f"key={key!r}",
            flush=True,
        )

        cached = self._prepared_cache_get(key)
        if cached is not None:
            print(
                "[AstronomicAL visualisation cache] HIT "
                f"panel={type(self).__name__} rows={len(cached.frame):,}",
                flush=True,
            )
            return cached

        print(
            "[AstronomicAL visualisation cache] MISS "
            f"panel={type(self).__name__}",
            flush=True,
        )

        data = prepare_plot_frame(self.context, self.state, require_y=require_y)
        self._prepared_cache_set(key, data)
        return data

    def _same_underlying_frame(self, left, right) -> bool:
        return self._frame_for_cache(left) is self._frame_for_cache(right)

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
        if not row_ids or data.empty:
            return data.frame.iloc[0:0]

        scan_limit = int(getattr(self, "_focus_frame_scan_limit", 500_000))

        if not visible_only and len(data.frame) > scan_limit:
            print(
                "[AstronomicAL visualisation] blocked full-frame row-id lookup "
                f"panel={type(self).__name__} "
                f"rows={len(data.frame):,} "
                f"row_ids={len(row_ids)}",
                flush=True,
            )
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

    def _coerce_single_row_frame(self, value):
        if value is None:
            return None

        if isinstance(value, pd.DataFrame):
            if value.empty:
                return None
            return value.head(1)

        if isinstance(value, pd.Series):
            return value.to_frame().T

        if isinstance(value, dict):
            return pd.DataFrame([value])

        return None


    def _call_first_working(self, obj, method_names, call_variants):
        for method_name in method_names:
            method = getattr(obj, method_name, None)

            if not callable(method):
                continue

            for args, kwargs in call_variants:
                try:
                    return method(*args, **kwargs)
                except TypeError:
                    continue
                except Exception:
                    continue

        return None


    def _focus_row_from_dataset_source(self, dataset_id, row_id, columns):
        """
        Fetch one focused row through DatasetManager/DatasetSource.

        This avoids scanning the full prepared plotting frame on every focus event.
        """
        if columns is not None:
            columns = list(dict.fromkeys(columns))

        datasets = getattr(self.context, "datasets", None)

        if datasets is None:
            return None

        record_id_col = getattr(self.state, "record_id_col", None)

        if not record_id_col or record_id_col == "Use Index":
            return None

        row_id_str = str(row_id)

        # 1. Try DatasetManager-level row lookup APIs if they exist.
        manager_value = self._call_first_working(
            datasets,
            (
                "get_row_by_id",
                "row_by_id",
                "find_row_by_id",
            ),
            (
                ((dataset_id, row_id_str, record_id_col), {"columns": columns}),
                ((dataset_id, row_id_str), {"id_column": record_id_col, "columns": columns}),
                ((dataset_id, row_id_str), {"columns": columns}),
                ((dataset_id, row_id_str, record_id_col), {}),
                ((dataset_id, row_id_str), {}),
            ),
        )

        manager_df = self._coerce_single_row_frame(manager_value)
        if manager_df is not None:
            return manager_df

        # 2. Try DatasetSource-level row lookup APIs.
        try:
            source = datasets.get_source(dataset_id)
        except Exception:
            source = None

        if source is None:
            return None

        source_value = self._call_first_working(
            source,
            (
                "get_row_by_id",
                "row_by_id",
                "find_row_by_id",
            ),
            (
                ((row_id_str, record_id_col), {"columns": columns}),
                ((row_id_str,), {"id_column": record_id_col, "columns": columns}),
                ((row_id_str,), {"columns": columns}),
                ((record_id_col, row_id_str), {"columns": columns}),
                ((record_id_col, row_id_str), {}),
                ((row_id_str,), {}),
            ),
        )

        source_df = self._coerce_single_row_frame(source_value)
        if source_df is not None:
            return source_df

        # 3. Try a source-backed filtered to_pandas call.
        to_pandas = getattr(source, "to_pandas", None)

        if callable(to_pandas):
            for where_sql in (
                f'CAST("{record_id_col}" AS VARCHAR) = ?',
                f'"{record_id_col}" = ?',
            ):
                try:

                    kwargs = {
                        "where_sql": where_sql,
                        "params": [row_id_str],
                        "limit": 1,
                    }

                    if columns is not None:
                        kwargs["columns"] = columns

                    df = to_pandas(**kwargs)

                except TypeError:
                    continue
                except Exception:
                    continue

                df = self._coerce_single_row_frame(df)
                if df is not None:
                    return df

        # 4. Try position lookup + source row-by-position APIs.
        position = None
        find_position = getattr(datasets, "find_position_by_id", None)

        if callable(find_position):
            for args, kwargs in (
                ((dataset_id, row_id_str, record_id_col), {}),
                ((dataset_id, row_id_str), {"id_column": record_id_col}),
                ((dataset_id, row_id_str), {}),
            ):
                try:
                    position = find_position(*args, **kwargs)
                    break
                except TypeError:
                    continue
                except Exception:
                    continue

        if position is None:
            return None

        position_value = self._call_first_working(
            source,
            (
                "get_row_by_position",
                "row_by_position",
                "get_row",
                "row",
            ),
            (
                ((position,), {"columns": columns}),
                ((position,), {}),
            ),
        )

        position_df = self._coerce_single_row_frame(position_value)
        if position_df is not None:
            return position_df

        rows_value = self._call_first_working(
            source,
            (
                "rows_by_positions",
                "get_rows_by_positions",
                "take_rows",
            ),
            (
                (([position],), {"columns": columns}),
                (([position],), {}),
            ),
        )

        rows_df = self._coerce_single_row_frame(rows_value)
        if rows_df is not None:
            return rows_df

        return None


    def _focus_point_from_dataset_source(self, focus):
        """
        Resolve the focused point for this panel's current axes by fetching one
        source row, not by scanning the full prepared frame.

        Order:
        1. per-panel point cache
        2. shared one-row dataframe cache
        3. source lookup for a full focused row
        4. source lookup for just this panel's x/y columns
        """
        dataset_id = self._dataset_id()

        if focus is None or getattr(focus, "dataset_id", None) != dataset_id:
            return None

        row_id = getattr(focus, "row_id", None)
        if row_id is None:
            return None

        x_col = getattr(self.state, "x", None)
        y_col = getattr(self.state, "y", None)

        if not x_col or not y_col:
            return None

        record_id_col = getattr(self.state, "record_id_col", None)

        if not record_id_col or record_id_col == "Use Index":
            return None

        cache_key = (
            dataset_id,
            str(row_id),
            str(record_id_col),
            str(x_col),
            str(y_col),
            bool(getattr(self.state, "log_x", False)),
            bool(getattr(self.state, "log_y", False)),
        )

        cached = self._focus_point_cache_get(cache_key)
        if cached is not None:
            return cached

        shared_key = (
            "focus_row",
            dataset_id,
            str(record_id_col),
            str(row_id),
        )

        failed_shared_key = (
            "focus_row_failed",
            dataset_id,
            str(record_id_col),
            str(row_id),
        )

        row_df = self._shared_focus_row_get(shared_key)

        # Avoid every open panel repeating a failed "fetch full row" attempt.
        shared_fetch_failed = self._shared_focus_row_get(failed_shared_key) is True

        if row_df is None and not shared_fetch_failed:
            row_df = self._focus_row_from_dataset_source(
                dataset_id,
                row_id,
                None,  # fetch full row if DatasetSource supports it
            )

            if row_df is not None and not row_df.empty:
                self._shared_focus_row_set(shared_key, row_df)
            else:
                self._shared_focus_row_set(failed_shared_key, True)

        if (
            row_df is None
            or row_df.empty
            or x_col not in row_df.columns
            or y_col not in row_df.columns
        ):
            row_df = self._focus_row_from_dataset_source(
                dataset_id,
                row_id,
                [record_id_col, x_col, y_col],
            )

        if row_df is None or row_df.empty:
            return None

        if x_col not in row_df.columns or y_col not in row_df.columns:
            return None

        try:
            x = float(row_df.iloc[0][x_col])
            y = float(row_df.iloc[0][y_col])
        except Exception:
            return None

        if not np.isfinite(x) or not np.isfinite(y):
            return None

        if getattr(self.state, "log_x", False) and x <= 0:
            return None

        if getattr(self.state, "log_y", False) and y <= 0:
            return None

        point = (x, y)
        self._focus_point_cache_set(cache_key, point)
        return point


    def _focus_point_from_metadata(self, focus):
        metadata = getattr(focus, "metadata", None) or {}

        def _meta(*names, default=None):
            for name in names:
                if isinstance(metadata, dict) and name in metadata:
                    return metadata.get(name)
                if hasattr(focus, name):
                    return getattr(focus, name)
            return default

        focus_x_col = _meta("x_col", "x_variable")
        focus_y_col = _meta("y_col", "y_variable")
        focus_log_x = bool(_meta("log_x", "x_log", default=False))
        focus_log_y = bool(_meta("log_y", "y_log", default=False))

        current_x_col = str(getattr(self.state, "x", "") or "")
        current_y_col = str(getattr(self.state, "y", "") or "")
        current_log_x = bool(getattr(self.state, "log_x", False))
        current_log_y = bool(getattr(self.state, "log_y", False))

        if str(focus_x_col or "") != current_x_col:
            return None

        if str(focus_y_col or "") != current_y_col:
            return None

        if focus_log_x != current_log_x:
            return None

        if focus_log_y != current_log_y:
            return None

        try:
            x = float(_meta("x", "focus_x"))
            y = float(_meta("y", "focus_y"))
        except Exception:
            return None

        if not np.isfinite(x) or not np.isfinite(y):
            return None

        if current_log_x and x <= 0:
            return None

        if current_log_y and y <= 0:
            return None

        return x, y

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

        row_id = str(getattr(focus, "row_id", "") or "")

        if not row_id:
            return None

        # 1. Immediate fast path: focus event came from a panel with same axes.
        point = self._focus_point_from_metadata(focus)
        if point is not None:
            return point

        # 2. Immediate cache path: async job may already have fetched this row.
        point = self._focus_point_from_cached_focus_row(focus)
        if point is not None:
            return point

        # 3. For large datasets, schedule async lookup and return immediately.
        if len(data.frame) > int(getattr(self, "_focus_source_lookup_row_limit", 500_000)):
            self._schedule_async_focus_lookup(focus, data)
            return None

        # 4. Small/medium datasets can still do a direct source lookup.
        point_from_source = getattr(self, "_focus_point_from_dataset_source", None)

        if callable(point_from_source):
            point = point_from_source(focus)
            if point is not None:
                return point

        # 5. Small-frame fallback only.
        if len(data.frame) > int(getattr(self, "_focus_frame_scan_limit", 500_000)):
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

            # Add these:
            tools=[],
            active_tools=[],
            toolbar=None,

            # Remove this unless you specifically need it here:
            # hooks=[force_wheel_zoom_hook],

            logx=self.state.log_x,
            logy=self.state.log_y,
            shared_axes=False,
            axiswise=True,
            framewise=True,
            **self._current_range_opts(include_y=True),
        )

    def _focus_overlay(self, data: PreparedFrame, *, size: float = 14):
        
        t0 = time.perf_counter()
        point = self._focus_point(data)
        dt = time.perf_counter() - t0

        if dt > 0.25:
            print(
                "[AstronomicAL visualisation] slow focus_point "
                f"panel={type(self).__name__} "
                f"duration={dt:.3f}s "
                f"rows={len(data.frame):,} "
                f"x={getattr(self.state, 'x', None)!r} "
                f"y={getattr(self.state, 'y', None)!r}",
                flush=True,
            )

        if getattr(self, "_debug_visualisation", False):
            print(
                "[AstronomicAL visualisation] focus_overlay "
                f"panel={type(self).__name__} "
                f"x={getattr(self.state, 'x', None)!r} "
                f"y={getattr(self.state, 'y', None)!r} "
                f"point={point!r}",
                flush=True,
            )

        if point is None:
            return hv.Overlay([])

        x, y = point
        if y is None:
            return hv.Overlay([])

        return hv.Points(
            [(x, y)],
            kdims=[INTERNAL_X, INTERNAL_Y],
        ).opts(
            marker="circle",
            size=size,
            fill_alpha=0.0,
            line_color="black",
            line_width=3,

            # Add these:
            tools=[],
            active_tools=[],
            toolbar=None,

            # Remove this unless you specifically need it here:
            # hooks=[force_wheel_zoom_hook],

            logx=self.state.log_x,
            logy=self.state.log_y,
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

    def _event_topic(self, topic):
        return str(topic or "")


    def _uses_label_rendering(self) -> bool:
        color_by = str(getattr(self.state, "color_by", "") or "").strip().lower()
        label_filter = getattr(self.state, "label_filter", None) or []
        label_col = getattr(self.state, "label_col", None)

        if color_by == "labels":
            return True

        if label_col and label_col != "No Labels":
            if label_filter and "All" not in label_filter:
                return True

        return False

    def _maybe_prewarm_row_ids(self, *, topic: str, payload=None) -> None:
        """
        Warm the visualisation row-id cache after dataset or mapping changes.

        Do not run this for label settings or ordinary visual state changes.
        The goal is only to avoid the first expensive row-id-cache MISS after a
        dataset becomes active or the record-id mapping changes.
        """
        topic = str(topic or "")

        if topic not in {
            "dataset.loaded",
            "dataset.updated",
            "dataset.active.changed",
            "dataset.mapping_updated",
        }:
            return

        dataset_id = self._dataset_id()

        if dataset_id is None:
            return

        record_id_col = getattr(self.state, "record_id_col", None)

        if not record_id_col or record_id_col == "Use Index":
            return

        jobs = getattr(self.context, "jobs", None)

        if jobs is None:
            return

        key = f"visualisation:row_ids:{dataset_id}:{record_id_col}"

        def _work(*, cancel_token=None):
            if cancel_token is not None and cancel_token.cancelled():
                return None

            return _load_row_ids_array(self.context, self.state)

        try:
            jobs.submit(
                _work,
                title="Prewarm visualisation row IDs",
                key=key,
            )
        except TypeError:
            # Compatibility with simpler JobManager signatures.
            try:
                jobs.submit(_work, key=key)
            except Exception:
                pass
        except Exception:
            pass


    def _on_dataset_event(self, topic, payload) -> None:
        t0 = time.perf_counter()
        topic = self._event_topic(topic)

        print(
            f"[AstronomicAL visualisation] dataset event received: {topic}",
            flush=True,
        )

        if isinstance(payload, dict):
            dataset_id = payload.get("dataset_id")
            if dataset_id is not None and dataset_id != self._dataset_id():
                return

        before = self._visual_state_signature()

        was_using_labels = self._uses_label_rendering()

        self._suppress_state_refresh = True
        try:
            if topic == "labels.settings.updated":
                # Important:
                # Always apply label settings first. This event may be what changes
                # the panel from color_by="None" to color_by="Labels".
                self.state.apply_label_settings(payload)
            else:
                self.state.refresh_from_context()
        finally:
            self._suppress_state_refresh = False

        after = self._visual_state_signature()
        now_using_labels = self._uses_label_rendering()

        if topic == "dataset.mapping_updated" and before == after:
            print(
                "[AstronomicAL visualisation] mapping update did not change visual state; "
                "skipping refresh",
                flush=True,
            )
            return

        if topic == "labels.settings.updated":
            # Skip only if the event genuinely has no visual effect.
            # Do not skip before apply_label_settings(), because that prevents
            # newly applied labels from enabling label colouring.
            if before == after and not was_using_labels and not now_using_labels:
                print(
                    "[AstronomicAL visualisation] label settings did not affect this panel; "
                    "skipping refresh",
                    flush=True,
                )
                return

        self._clear_prepared_cache()

        # Only prewarm row ids for dataset/mapping changes, not label changes.
        self._maybe_prewarm_row_ids(topic=topic, payload=payload)

        print(
            "[AstronomicAL visualisation] dataset event state refresh complete "
            f"topic={topic} "
            f"duration={time.perf_counter() - t0:.2f}s",
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

        event_dataset_id = self._payload_value(payload, "dataset_id")
        active_dataset_id = self._dataset_id()

        if (
            event_dataset_id is not None
            and active_dataset_id is not None
            and str(event_dataset_id) != str(active_dataset_id)
        ):
            return

        # Selection focus updates should never block the tap interaction.
        # Schedule slightly later so the click/record-browser feedback can settle.
        def _run():
            self._schedule_refresh(reason=f"selection.{topic}")

        try:
            import panel as pn
            pn.state.curdoc.add_timeout_callback(_run, 50)
        except Exception:
            _run()

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
        if tools is None:
            tools = ["pan", "wheel_zoom", "box_zoom", "reset"]

        if active_tools is None:
            active_tools = ["wheel_zoom"]

        opts = dict(
            xlabel=str(xlabel if xlabel is not None else self.state.x),
            ylabel=str(ylabel if ylabel is not None else self.state.y),
            logx=self.state.log_x,
            logy=self.state.log_y,
            tools=tools,
            active_tools=active_tools,
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
        return pn.Column(
        pn.Row(
            self.status_pane,
            settings_select(self.state.param.color_by, name="Colour", width=130),
            settings_multichoice(self.state.param.label_filter, name="Labels", width=210),
            settings_select(self.state.param.render_mode, name="Render", width=130),
        ),
        pn.Row(
            settings_int_input(self.state.param.datashade_threshold, name="Shade threshold", width=145),
            settings_int_input(self.state.param.interactive_sample_limit, name="Sample limit", width=130),
            settings_int_input(self.state.param.max_selection_ids, name="Max selected IDs", width=145),
            settings_float_slider(self.state.param.point_size, name="Size", width=175),
            settings_float_slider(self.state.param.point_alpha, name="Alpha", width=175),
        ),
        pn.Row(
            settings_checkbox(self.state.param.log_x, name="Log X"),
            settings_checkbox(self.state.param.log_y, name="Log Y"),
        ),
        sizing_mode="stretch_width",
        height_policy="fit",
        margin=(5, 0, 0, 0),
        styles={
            "overflow": "visible",
            "align-content": "flex-start",
            "align-items": "flex-start",
            "gap": "2px 6px",
            "padding": "4px 6px 4px 6px",
            "border-top": "1px solid #ddd",
            "border-bottom": "1px solid #eee",
            "background": "#fafafa",
            "box-sizing": "border-box",
        },
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