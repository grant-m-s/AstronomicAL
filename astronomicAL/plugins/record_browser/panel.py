import html
import uuid
from typing import Any, Dict, List, Optional

from collections import OrderedDict

import numpy as np
import pandas as pd
import panel as pn
import param

import time

try:
    from astronomicAL.utils.optimise import get_series_type
except Exception:  # pragma: no cover - defensive fallback for plugin reuse.
    def get_series_type(series):
        if pd.api.types.is_bool_dtype(series):
            return "bool"
        if pd.api.types.is_integer_dtype(series):
            return "int"
        if pd.api.types.is_float_dtype(series):
            return "float"
        return "string"


def _safe_unwatch(watcher):
    for attr in ("obj", "inst"):
        owner = getattr(watcher, attr, None)
        if owner is not None and hasattr(owner, "param"):
            try:
                owner.param.unwatch(watcher)
                return
            except Exception:
                pass

class RecordBrowserPanel(param.Parameterized):
    """Plugin-native replacement for the generic part of ExplorationDashboard.

    This intentionally does not:
    - load datasets
    - register datasets
    - request mappings manually
    - require RA/Dec
    - generate ra_dec
    - generate training features
    - generate fake labels

    It only:
    - reads the active dataset
    - uses the mapped record_id column
    - browses records
    - displays metadata
    - publishes/consumes selection focus
    - optionally publishes label display settings
    """

    index = param.Integer(default=0, bounds=(0, 0))

    def __init__(self, context, **params):
        super().__init__(**params)

        self.context = context
        self.config = getattr(context, "config", None)

        self.panel_id = str(uuid.uuid4())

        self.df = pd.DataFrame()  # compatibility preview only
        self.source = None
        self.columns: List[str] = []
        self.row_count: int = 0

        self._event_subs: List[Any] = []
        self._watchers: List[Any] = []
        self._running_panels = set()
        self._built = False
        self._disposed = False

        self.main_layout = None
        self.labels_expanded = False
        self._preserve_sourceid_input_once = False

        self.visited_indices = [self.index]
        self.current_position = 0

        self.record_id_col: Optional[str] = None
        self.label_col = "No Labels"
        self.labels: List[Any] = []
        self.label_to_strings_param: Dict[str, Any] = {}
        self.colours_param: Dict[Any, Any] = {}

        self.extra_info_cols = self._initial_extra_info_cols()

        self._row_position_cache = OrderedDict()
        self._row_position_cache_max = 4096
        self._pending_focus_row_id = None
        self._focus_sync_scheduled = False

        self._dataset_refresh_scheduled = False
        self._pending_dataset_reset_history = False

        self._root = pn.Column(
            sizing_mode="stretch_both",
            scroll=True,
            margin=(0, 0, 0, 0),
            styles={
                "padding": "4px",
                "box-sizing": "border-box",
            },
        )

        self._subscribe_to_dataset_events()
        self._subscribe_to_selection_events()

        self._refresh_from_active_dataset(reset_history=True)


    def get_state(self):
        return {
            "extra_info_cols": list(self.extra_info_cols or []),
        }


    def restore_state(self, state):
        if not isinstance(state, dict):
            return

        # Defensive: if this was saved while still mapping-gated, the real state
        # may be nested inside the gate wrapper.
        if "extra_info_cols" not in state and isinstance(state.get("restore_state"), dict):
            state = state["restore_state"]

        if "extra_info_cols" not in state:
            return

        cols = state.get("extra_info_cols") or []
        if not isinstance(cols, (list, tuple)):
            return

        restored = [str(col) for col in cols if col]

        # If the dataset schema is already known, remove stale columns.
        # If not, keep them and let the next dataset refresh validate them.
        if self.columns:
            restored = [col for col in restored if col in self.columns]

        self.extra_info_cols = restored
        self._sync_legacy_settings()

        if hasattr(self, "extra_info_html"):
            self._refresh_extra_info_view()

        if getattr(self, "_built", False):
            self._rerender_main_layout()

    # ---------------------------------------------------------------------
    # Dataset / mapping helpers
    # ---------------------------------------------------------------------

    def _has_focus_for_active_dataset(self) -> bool:
        focus = self._get_focus_state()
        if focus is None:
            return False

        dataset_id = self._focus_value(focus, "dataset_id")
        row_id = self._focus_value(focus, "row_id")

        return dataset_id == self._dataset_id() and row_id is not None

    def _set_index_bounds(self, max_index: int) -> None:
        max_index = max(0, int(max_index))

        try:
            self.param["index"].bounds = (0, max_index)
        except Exception:
            self.param.index.bounds = (0, max_index)

        if hasattr(self, "index_input"):
            self.index_input.start = 0
            self.index_input.end = max_index

    def _dataset_id(self) -> Optional[str]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return None
        try:
            return datasets.active_id()
        except Exception:
            return None

    def _active_df(self) -> Optional[pd.DataFrame]:
        datasets = getattr(self.context, "datasets", None)
        dataset_id = self._dataset_id()
        if datasets is None or dataset_id is None:
            return None
        try:
            return datasets.get_df(dataset_id)
        except Exception:
            return None

    def _active_source(self):
        datasets = getattr(self.context, "datasets", None)
        dataset_id = self._dataset_id()

        if datasets is None or dataset_id is None:
            return None

        get_source = getattr(datasets, "get_source", None)
        if callable(get_source):
            try:
                return get_source(dataset_id)
            except Exception:
                return None

        return None

    def _active_columns(self) -> List[str]:
        datasets = getattr(self.context, "datasets", None)
        dataset_id = self._dataset_id()

        if datasets is not None and dataset_id is not None:
            list_columns = getattr(datasets, "list_columns", None)
            if callable(list_columns):
                try:
                    return list(list_columns(dataset_id))
                except Exception:
                    pass

        if self.source is not None:
            try:
                return list(self.source.columns())
            except Exception:
                pass

        return []

    def _active_row_count(self) -> int:
        datasets = getattr(self.context, "datasets", None)
        dataset_id = self._dataset_id()

        if datasets is not None and dataset_id is not None:
            row_count = getattr(datasets, "row_count", None)
            if callable(row_count):
                try:
                    return int(row_count(dataset_id))
                except Exception:
                    pass

        if self.source is not None:
            try:
                return int(self.source.row_count())
            except Exception:
                pass

        return 0

    def _column_exists(self, column: Optional[str]) -> bool:
        return column is not None and column in set(self.columns)


    def _get_mapping(self, semantic_name: str) -> Optional[str]:
        datasets = getattr(self.context, "datasets", None)
        dataset_id = self._dataset_id()
        if datasets is None or dataset_id is None:
            return None

        for method_name in ("get_mapping", "mapping", "get_column_mapping"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    value = method(dataset_id, semantic_name)
                except Exception:
                    value = None
                if value:
                    return value

        return None

    def _is_index_mapping(self, value: Optional[str]) -> bool:
        if value is None:
            return False
        return str(value).strip().lower() in {
            "use index",
            "index",
            "__index__",
            "_index",
        }

    def _resolve_record_id_col(self) -> Optional[str]:
        mapped = self._get_mapping("record_id")

        if self._is_index_mapping(mapped):
            return "Use Index"

        if mapped and mapped in self.columns:
            return mapped

        # Backward-compatible bridge only. The plugin does not depend on this.
        settings = getattr(self.config, "settings", {}) if self.config is not None else {}
        legacy = settings.get("id_col")

        if self._is_index_mapping(legacy):
            return "Use Index"

        if legacy and legacy in self.columns:
            return legacy

        return None

    def _resolve_label_col(self) -> str:
        mapped = self._get_mapping("target_label")

        if mapped and mapped in self.columns:
            return mapped

        if self.label_col and self.label_col in self.columns:
            return self.label_col

        settings = getattr(self.config, "settings", {}) if self.config is not None else {}
        legacy = settings.get("label_col")

        if legacy and legacy in self.columns:
            return legacy

        return "No Labels"

    def _initial_extra_info_cols(self) -> List[str]:
        settings = getattr(self.config, "settings", {}) if self.config is not None else {}
        existing = settings.get("extra_info_cols", [])
        if isinstance(existing, list):
            return list(existing)
        return []

    def _sync_legacy_settings(self) -> None:
        """Write small compatibility hints for old panels without owning config."""
        if self.config is None:
            return

        settings = getattr(self.config, "settings", None)
        if settings is None:
            return

        if self.record_id_col is not None:
            settings["id_col"] = self.record_id_col

        settings["label_col"] = self.label_col
        settings["extra_info_cols"] = list(self.extra_info_cols)

        # Compatibility only: schema frame, not the full dataset.
        try:
            self.config.main_df = pd.DataFrame(columns=self.columns)
        except Exception:
            pass

    def _get_unique_label_values(
        self,
        column: str,
        *,
        max_values: int = 21,
    ) -> tuple[list[Any], str, bool]:
        """
        Return unique non-null values for a label column.

        The boolean says whether the column exceeded max_values.
        This must not use self.df, because self.df is schema-only for
        Parquet-backed datasets.
        """
        if not column or column not in self.columns:
            return [], "string", False

        datasets = getattr(self.context, "datasets", None)
        dataset_id = self._dataset_id()

        # Preferred dataset-level API.
        if datasets is not None and dataset_id is not None:
            for method_name in (
                "unique_values",
                "get_unique_values",
                "column_unique_values",
            ):
                method = getattr(datasets, method_name, None)
                if callable(method):
                    try:
                        values = method(
                            dataset_id,
                            column,
                            max_values=max_values,
                            dropna=True,
                        )
                        values = list(values)
                        exceeded = len(values) > max_values - 1
                        return values[: max_values - 1], self._infer_label_type(values), exceeded
                    except TypeError:
                        try:
                            values = method(dataset_id, column)
                            values = list(values)
                            exceeded = len(values) > max_values - 1
                            return values[: max_values - 1], self._infer_label_type(values), exceeded
                        except Exception:
                            pass
                    except Exception:
                        pass

        # Preferred source-level API.
        if self.source is not None:
            for method_name in (
                "unique_values",
                "get_unique_values",
                "column_unique_values",
            ):
                method = getattr(self.source, method_name, None)
                if callable(method):
                    try:
                        values = method(
                            column,
                            max_values=max_values,
                            dropna=True,
                        )
                        values = list(values)
                        exceeded = len(values) > max_values - 1
                        return values[: max_values - 1], self._infer_label_type(values), exceeded
                    except TypeError:
                        try:
                            values = method(column)
                            values = list(values)
                            exceeded = len(values) > max_values - 1
                            return values[: max_values - 1], self._infer_label_type(values), exceeded
                        except Exception:
                            pass
                    except Exception:
                        pass

        # Fallback: read one column only. This is not ideal for huge columns, but
        # still avoids materialising the whole 274-column dataset.
        try:
            df = self.source.to_pandas(columns=[column])
        except Exception as exc:
            print(
                "[AstronomicAL record_browser] Could not read label column "
                f"{column!r}: {exc}",
                flush=True,
            )
            return [], "string", False

        if df.empty or column not in df.columns:
            return [], "string", False

        series = df[column]

        label_type = get_series_type(series)

        if label_type == "mixed":
            series = series.astype(str)
            label_type = "string"
        elif label_type == "bool":
            series = series.astype("Int64")
            label_type = "int"

        unique_values = list(series.dropna().unique())
        exceeded = len(unique_values) > max_values - 1

        return unique_values[: max_values - 1], label_type, exceeded


    def _infer_label_type(self, values: list[Any]) -> str:
        if not values:
            return "string"

        series = pd.Series(values)

        label_type = get_series_type(series)

        if label_type == "mixed":
            return "string"

        if label_type == "bool":
            return "int"

        return label_type

    def _row_position_cache_key(self, row_id):
        return (
            self._dataset_id(),
            str(getattr(self, "record_id_col", "") or ""),
            str(row_id),
        )


    def _row_position_cache_get(self, row_id):
        key = self._row_position_cache_key(row_id)
        value = self._row_position_cache.get(key)

        if value is not None:
            self._row_position_cache.move_to_end(key)

        return value


    def _row_position_cache_set(self, row_id, position):
        if position is None:
            return

        key = self._row_position_cache_key(row_id)

        if key in self._row_position_cache:
            self._row_position_cache.pop(key, None)

        self._row_position_cache[key] = int(position)

        while len(self._row_position_cache) > self._row_position_cache_max:
            self._row_position_cache.popitem(last=False)


    def _schedule_dataset_refresh(self, *, reset_history: bool) -> None:
        """Schedule dataset refresh outside the EventBus subscriber.

        Record browser refresh can rebuild widgets and resolve focus. That is
        too heavy to run synchronously inside dataset.updated publication.
        """
        self._pending_dataset_reset_history = (
            bool(self._pending_dataset_reset_history) or bool(reset_history)
        )

        if self._dataset_refresh_scheduled:
            return

        self._dataset_refresh_scheduled = True

        def _run():
            self._dataset_refresh_scheduled = False
            reset = bool(self._pending_dataset_reset_history)
            self._pending_dataset_reset_history = False

            if self._disposed:
                return

            self._refresh_from_active_dataset(reset_history=reset)

        try:
            pn.state.curdoc.add_next_tick_callback(_run)
        except Exception:
            _run()

    def _schedule_focus_from_selection(self, row_id: str) -> None:
        """
        Do not run potentially slow record-browser seeking directly inside the
        selection.focus.changed event callback. The EventBus should not be blocked
        by UI navigation.
        """
        self._pending_focus_row_id = str(row_id)

        if self._focus_sync_scheduled:
            return

        self._focus_sync_scheduled = True

        def _run():
            self._focus_sync_scheduled = False
            pending = self._pending_focus_row_id
            self._pending_focus_row_id = None

            if pending is None:
                return

            self._focus_row_from_selection(str(pending))

        try:
            import panel as pn
            pn.state.curdoc.add_next_tick_callback(_run)
        except Exception:
            _run()

    # ---------------------------------------------------------------------
    # Events
    # ---------------------------------------------------------------------

    def _subscribe(self, topic: str, callback) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        try:
            sub = events.subscribe(topic, callback)
            self._event_subs.append(sub)
        except Exception:
            pass

    def _subscribe_to_dataset_events(self) -> None:
        self._subscribe("dataset.active.changed", self._on_dataset_active_changed)
        self._subscribe("dataset.updated", self._on_dataset_updated)
        self._subscribe("dataset.mapping_updated", self._on_dataset_mapping_updated)

    def _subscribe_to_selection_events(self) -> None:
        self._subscribe("selection.focus.changed", self._on_selection_focus_changed)

    def _on_dataset_active_changed(self, _topic, payload) -> None:
        if isinstance(payload, dict):
            dataset_id = payload.get("dataset_id")
            if dataset_id is not None and dataset_id != self._dataset_id():
                # active_id may already have changed by this point, so do not
                # return solely because payload disagrees. This guard mainly
                # prevents unrelated dataset chatter.
                pass

        self._schedule_dataset_refresh(reset_history=True)

    def _on_dataset_updated(self, _topic, payload) -> None:
        dataset_id = payload.get("dataset_id") if isinstance(payload, dict) else None
        if dataset_id is not None and dataset_id != self._dataset_id():
            return

        # Keep the subscriber cheap. The scheduled refresh will update schema,
        # column counts, label selector options, and extra-info rendering.
        self._schedule_dataset_refresh(reset_history=False)

    def _on_dataset_mapping_updated(self, _topic, payload) -> None:
        if not isinstance(payload, dict):
            return

        dataset_id = payload.get("dataset_id")
        if dataset_id is not None and dataset_id != self._dataset_id():
            return

        semantic_name = payload.get("semantic_name")
        if semantic_name in {"record_id", "target_label"}:
            self._schedule_dataset_refresh(reset_history=False)

    def _on_selection_focus_changed(self, _topic, payload) -> None:
        if not isinstance(payload, dict):
            return

        if payload.get("panel_id") == self.panel_id:
            return

        dataset_id = payload.get("dataset_id")
        if dataset_id is not None and dataset_id != self._dataset_id():
            return

        row_id = payload.get("row_id")
        if row_id is None:
            return

        row_id = str(row_id)

        def _run():
            self._focus_row_from_selection(row_id)

        try:
            import panel as pn
            pn.state.curdoc.add_next_tick_callback(_run)
        except Exception:
            _run()


    # ---------------------------------------------------------------------
    # Data refresh / empty states
    # ---------------------------------------------------------------------

    def _refresh_from_active_dataset(self, reset_history: bool = True) -> None:
        if self._disposed:
            return

        t0 = time.perf_counter()

        print(
            "[AstronomicAL record_browser] refresh_from_active_dataset start "
            f"reset_history={reset_history}",
            flush=True,
        )

        self.source = self._active_source()

        if self.source is None:
            self.df = pd.DataFrame()
            self.columns = []
            self.row_count = 0
            self.record_id_col = None
            self._built = False
            self._render_no_dataset()
            return

        t_schema = time.perf_counter()

        self.columns = self._active_columns()
        self.row_count = self._active_row_count()

        # Schema-only compatibility frame. Do not materialise the dataset here.
        self.df = pd.DataFrame(columns=self.columns)

        print(
            "[AstronomicAL record_browser] schema loaded "
            f"columns={len(self.columns):,} rows={self.row_count:,} "
            f"duration={time.perf_counter() - t_schema:.3f}s",
            flush=True,
        )

        self.record_id_col = self._resolve_record_id_col()

        if self.record_id_col is None:
            self._built = False
            self._render_no_record_id_mapping()
            print(
                "[AstronomicAL record_browser] missing record_id mapping "
                f"total={time.perf_counter() - t0:.3f}s",
                flush=True,
            )
            return

        self.label_col = self._resolve_label_col()

        self.extra_info_cols = [
            col for col in self.extra_info_cols
            if col in self.columns
        ]

        self._sync_legacy_settings()

        if self.row_count == 0:
            self._set_index_bounds(0)
            self.index = 0
            self._built = False
            self._render_empty_dataset()
            return

        max_index = max(0, self.row_count - 1)
        self._set_index_bounds(max_index)
        self.index = min(max(self.index, 0), max_index)

        if reset_history:
            self.visited_indices = [self.index]
            self.current_position = 0

        if not self._built:
            self._build_dashboard_ui()
            self._root[:] = [self.main_layout]
            self._built = True

            if not self._sync_index_from_current_focus():
                if not self._has_focus_for_active_dataset():
                    self._publish_focus_for_current_index()
                else:
                    print(
                        "[AstronomicAL record_browser] existing external focus could not be "
                        "resolved yet; preserving it instead of overwriting focus",
                        flush=True,
                    )

            print(
                "[AstronomicAL record_browser] refresh built ui "
                f"total={time.perf_counter() - t0:.3f}s",
                flush=True,
            )
            return

        self._update_widget_ranges()
        self._sync_label_selector_options()

        if hasattr(self, "column_selector"):
            self.column_selector.value = None
            self.column_selector.visible = False

        if hasattr(self, "extra_info_html"):
            self._refresh_extra_info_view()

        self._update_navigation_flags()
        self._rerender_main_layout()

        if not self._sync_index_from_current_focus():
            if not self._has_focus_for_active_dataset():
                self._publish_focus_for_current_index()
            else:
                print(
                    "[AstronomicAL record_browser] existing external focus could not be "
                    "resolved yet; preserving it instead of overwriting focus",
                    flush=True,
                )

        print(
            "[AstronomicAL record_browser] refresh end "
            f"total={time.perf_counter() - t0:.3f}s",
            flush=True,
        )

    def _render_no_dataset(self) -> None:
        load_button = pn.widgets.Button(
            name="Load dataset",
            button_type="primary",
            width=120,
            height=32,
            margin=(8, 0, 0, 0),
        )

        def _request_dataset_load(_event):
            events = getattr(self.context, "events", None)
            if events is not None:
                events.publish(
                    "dataset.open_requested",
                    {
                        "source": "core.record_browser",
                        "panel_id": self.panel_id,
                    },
                )

        load_button.on_click(_request_dataset_load)

        self._root[:] = [
            pn.Column(
                pn.pane.Alert(
                    "No active dataset is loaded. Use the dataset header to load "
                    "or switch datasets.",
                    alert_type="info",
                    sizing_mode="stretch_width",
                ),
                load_button,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )
        ]

    def _render_no_record_id_mapping(self) -> None:
        self._root[:] = [
            pn.Column(
                pn.pane.Alert(
                    "Record Browser needs a record ID mapping before it can open. "
                    "Use the column-mapping header alert to map `record_id`.",
                    alert_type="warning",
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )
        ]

    def _render_empty_dataset(self) -> None:
        self._root[:] = [
            pn.Column(
                pn.pane.Alert(
                    "The active dataset has no rows to browse.",
                    alert_type="warning",
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )
        ]

    def _update_widget_ranges(self) -> None:
        if not hasattr(self, "index_input"):
            return

        max_index = max(0, self.row_count - 1)

        self.index_input.start = 0
        self.index_input.end = max_index
        self.index_input.value = self.index

        if hasattr(self, "sourceid_input"):
            self.sourceid_input.value = ""

    # ---------------------------------------------------------------------
    # Visual helpers copied/kept from the Exploration panel style
    # ---------------------------------------------------------------------

    def _escape_html(self, value: Any) -> str:
        return html.escape(str(value), quote=True)

    def _flex_row(
        self,
        *objects,
        gap="8px",
        margin=(0, 0, 8, 0),
        justify_content="flex-start",
    ):
        return pn.FlexBox(
            *objects,
            flex_direction="row",
            flex_wrap="wrap",
            gap=gap,
            align_items="center",
            justify_content=justify_content,
            sizing_mode="stretch_width",
            margin=margin,
        )

    def _vspace(self, height=8):
        return pn.Spacer(height=height)

    def _field_block(self, label, widget, help_text=None, width=None):
        items = [
            pn.pane.Markdown(f"**{label}**", margin=(0, 0, 4, 0)),
            widget,
        ]

        if help_text:
            items.append(
                pn.pane.Markdown(
                    f"<small>{help_text}</small>",
                    margin=(4, 0, 0, 0),
                )
            )

        return pn.Column(*items, width=width, margin=(0, 0, 0, 0))

    def _heading(
        self,
        text,
        size=15,
        weight=700,
        color="#2f2f2f",
        margin_bottom=0,
    ):
        return pn.pane.HTML(
            (
                f"<div style='font-size:{size}px; font-weight:{weight}; "
                f"color:{color}; line-height:1.2; margin-bottom:{margin_bottom}px;'>"
                f"{self._escape_html(text)}"
                "</div>"
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

    def _body(self, text):
        return pn.pane.HTML(
            (
                "<div style='font-size:12px; color:#666; line-height:1.35;'>"
                f"{self._escape_html(text)}"
                "</div>"
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

    def _field_label(self, text):
        return pn.pane.HTML(
            (
                "<div style='font-size:11px; font-weight:700; color:#555; "
                "line-height:1.2; margin-bottom:3px;'>"
                f"{self._escape_html(text)}"
                "</div>"
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

    def _divider(self):
        return pn.pane.HTML(
            "<div style='height:1px; background:#e8e8e8; width:100%;'></div>",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

    def _card(self, *objects):
        return pn.Column(
            *objects,
            sizing_mode="stretch_width",
            min_width=0,
            styles={
                "border": "1px solid #d9d9d9",
                "border-radius": "8px",
                "background": "#ffffff",
                "padding": "10px 12px",
                "box-sizing": "border-box",
                "width": "100%",
                "overflow": "hidden",
            },
            margin=(0, 0, 8, 0),
        )

    def _subsection(self, title, description, *content):
        items = [
            self._heading(title, size=14, weight=700, margin_bottom=4),
            self._body(description),
        ]

        for obj in content:
            items.extend([pn.Spacer(height=5), obj])

        return pn.Column(
            *items,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(0, 0, 0, 0),
        )

    # ---------------------------------------------------------------------
    # Dashboard build
    # ---------------------------------------------------------------------

    def _build_dashboard_ui(self) -> None:
        max_index = max(0, self.row_count - 1)
        self._set_index_bounds(max_index)
        self.index = min(max(self.index, 0), max_index)

        self._sync_index_from_current_focus()

        self.index_input = pn.widgets.IntInput(
            name="",
            value=self.index,
            start=0,
            end=max_index,
            width=88,
            height=28,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        )

        self.sourceid_input = pn.widgets.TextInput(
            name="",
            placeholder="Enter full or partial ID",
            value="",
            sizing_mode="stretch_width",
            max_width=320,
            margin=(0, 0, 0, 0),
        )

        self._watchers.append(
            self.index_input.param.watch(self._index_input_cb, "value")
        )
        self._watchers.append(
            self.param.watch(self._sync_index_widget_cb, "index")
        )

        self.prev_button = pn.widgets.Button(
            name="Previous",
            button_type="default",
            width=80,
            height=30,
            margin=(0, 0, 0, 0),
        )

        self.next_button = pn.widgets.Button(
            name="Next",
            button_type="primary",
            width=80,
            height=30,
            margin=(0, 0, 0, 0),
        )

        self.search_button = pn.widgets.Button(
            name="Find",
            button_type="primary",
            width=80,
            height=30,
            margin=(0, 0, 0, 0),
        )

        self.extra_info_html = pn.pane.HTML(
            self._get_extra_info_html(),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.prev_button.on_click(self._go_previous)
        self.next_button.on_click(self._go_next)
        self.search_button.on_click(self._search_button_cb)

        self._initialise_add_remove_columns_widgets()
        self._initialise_label_selector()
        self._update_navigation_flags()

        self.main_layout = self._build_main_layout()
        self._publish_focus_for_current_index()

    def _build_main_layout(self):
        browse_controls = pn.Row(
            self.search_button,
            self.prev_button,
            self.next_button,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(0, 0, 0, 0),
        )

        metadata_controls = pn.Row(
            self.add_column_button,
            self.remove_column_button,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(0, 0, 0, 0),
        )

        index_block = pn.Column(
            self._field_label("Record index"),
            pn.Row(
                self.index_input,
                sizing_mode="fixed",
                width=88,
                height=28,
                margin=(0, 0, 0, 0),
            ),
            sizing_mode="fixed",
            width=110,
            min_height=46,
            max_height=46,
            margin=(0, 0, 0, 0),
        )

        current_record_block = self._subsection(
            "Current record",
            "Review the selected record.",
            index_block,
            pn.Column(
                self._field_label("Visible record information"),
                self.extra_info_html,
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 0, 0),
            ),
        )

        browse_block = self._subsection(
            "Browse records",
            "Search for a record or move backward and forward through your "
            "navigation history.",
            pn.Column(
                self._field_label("Find record"),
                self.sourceid_input,
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 0, 0),
            ),
            browse_controls,
        )

        record_card = self._card(
            current_record_block,
            pn.Spacer(height=5),
            self._divider(),
            pn.Spacer(height=5),
            browse_block,
        )

        metadata_items = [metadata_controls]

        if self.column_selector.visible:
            metadata_items.extend(
                [
                    pn.Spacer(height=5),
                    pn.Column(
                        self._field_label("Field"),
                        self.column_selector,
                        sizing_mode="stretch_width",
                        min_width=0,
                        margin=(0, 0, 0, 0),
                    ),
                ]
            )

        metadata_block = self._subsection(
            "Visible metadata",
            "Choose which extra fields appear in the summary above.",
            *metadata_items,
        )

        settings_card = self._card(metadata_block)
        labels_card = self._build_labels_card()

        return pn.Column(
            record_card,
            settings_card,
            labels_card,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(0, 0, 0, 0),
        )

    def _rerender_main_layout(self):
        if not self._built:
            return
        self.main_layout = self._build_main_layout()
        self._root[:] = [self.main_layout]

    # ---------------------------------------------------------------------
    # Record navigation
    # ---------------------------------------------------------------------

    def _index_input_cb(self, event):
        if event.new != self.index:
            self.index = event.new

    def _sync_index_widget_cb(self, event):
        if hasattr(self, "index_input") and self.index_input.value != event.new:
            self.index_input.value = event.new

    @param.depends("index", watch=True)
    def _update_history(self):
        if not hasattr(self, "sourceid_input"):
            return

        if not self.visited_indices:
            self.visited_indices = [self.index]
            self.current_position = 0
            return

        current_position = min(self.current_position, len(self.visited_indices) - 1)

        if self.visited_indices[current_position] == self.index:
            return

        if self.index != self.visited_indices[-1]:
            self.visited_indices.append(self.index)

        self.current_position = len(self.visited_indices) - 1
        self._update_navigation_flags()

        if getattr(self, "_preserve_sourceid_input_once", False):
            self._preserve_sourceid_input_once = False
        else:
            self.sourceid_input.value = ""

    def _go_previous(self, _event):
        if self.current_position > 0:
            self.current_position -= 1
            self.index = self.visited_indices[self.current_position]
        self._update_navigation_flags()

    def _go_next(self, _event):
        if self.row_count <= 1:
            self.index = 0
            self._update_navigation_flags()
            return

        max_index = self.row_count - 1
        self._set_index_bounds(max_index)

        if self.current_position < len(self.visited_indices) - 1:
            self.current_position += 1
            self.index = min(self.visited_indices[self.current_position], max_index)
        else:
            new_index = self.index + 1
            if new_index <= max_index:
                self.index = new_index
            else:
                self.index = 0

        self._update_navigation_flags()

    def _search_button_cb(self, _event):
        sourceid = self.sourceid_input.value
        if sourceid:
            self._find_from_id(sourceid)

    def _get_id(self):
        selected = self._get_current_row_df()

        if selected.empty:
            return pd.Series(dtype="object")

        if self.record_id_col == "Use Index":
            return pd.Series(selected.index, index=selected.index)

        if self.record_id_col in selected.columns:
            return selected[self.record_id_col]

        return pd.Series(selected.index, index=selected.index)

    def _get_current_row_df(self):
        if self.source is None or self.row_count <= 0:
            return pd.DataFrame(columns=self.columns)

        if self.index < 0 or self.index >= self.row_count:
            return pd.DataFrame(columns=self.columns)

        wanted: List[str] = []

        if (
            self.record_id_col
            and self.record_id_col != "Use Index"
            and self.record_id_col in self.columns
        ):
            wanted.append(self.record_id_col)

        if (
            self.label_col
            and self.label_col != "No Labels"
            and self.label_col in self.columns
        ):
            wanted.append(self.label_col)

        for col in self.extra_info_cols:
            if col in self.columns and col not in wanted:
                wanted.append(col)

        # If there are no real columns to fetch, return a non-empty dummy row so
        # _get_extra_info_df still displays the index-based Record ID.
        if not wanted:
            return pd.DataFrame({"__record_index__": [self.index]})

        try:
            return self.source.get_row_by_position(
                self.index,
                columns=wanted,
            )
        except Exception:
            pass

        datasets = getattr(self.context, "datasets", None)
        dataset_id = self._dataset_id()

        if datasets is not None and dataset_id is not None:
            method = getattr(datasets, "get_row_by_position", None)
            if callable(method):
                try:
                    return method(dataset_id, self.index, columns=wanted)
                except Exception:
                    pass

        return pd.DataFrame(columns=wanted)

    def _get_selected_id(self):
        if self.row_count <= 0:
            return None

        if self.record_id_col == "Use Index":
            return str(self.index)

        selected = self._get_current_row_df()

        if selected.empty:
            return None

        if self.record_id_col in selected.columns:
            return str(selected[self.record_id_col].iloc[0])

        return None

    def _find_index_for_row_id(self, row_id):
        print(
            "[RecordBrowser] _find_index_for_row_id start "
            f"dataset_id={self._dataset_id()!r} "
            f"row_id={row_id!r} "
            f"row_id_type={type(row_id).__name__} "
            f"record_id_col={self.record_id_col!r}",
            flush=True,
        )

        if row_id is None or self.row_count <= 0:
            return None

        cached = self._row_position_cache_get(row_id)

        if cached is not None:
            return cached

        if self.record_id_col == "Use Index":
            try:
                position = int(row_id)
            except Exception:
                return None

            if 0 <= position < self.row_count:
                self._row_position_cache_set(row_id, position)
                return position

            return None

        if not self.record_id_col or self.record_id_col not in self.columns:
            return None

        datasets = getattr(self.context, "datasets", None)
        dataset_id = self._dataset_id()

        if datasets is not None and dataset_id is not None:
            method = getattr(datasets, "find_position_by_id", None)
            if callable(method):
                try:
                    position = method(
                        dataset_id,
                        row_id,
                        id_column=self.record_id_col,
                    )
                    print(
                        "[RecordBrowser] DatasetManager.find_position_by_id returned "
                        f"{position!r}",
                        flush=True,
                    )
                    if position is not None:
                        self._row_position_cache_set(row_id, position)
                        return int(position)
                except Exception as exc:
                    print(
                        "[RecordBrowser] DatasetManager.find_position_by_id failed "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

        if self.source is not None:
            method = getattr(self.source, "find_position_by_id", None)
            if callable(method):
                try:
                    position = method(
                        row_id,
                        id_column=self.record_id_col,
                    )
                    if position is not None:
                        self._row_position_cache_set(row_id, position)
                        return int(position)
                except Exception:
                    pass

        return None

    def _find_from_id(self, sourceid):
        sourceid = sourceid.strip()

        if not sourceid or self.row_count <= 0:
            return

        exact_position = self._find_index_for_row_id(sourceid)

        if exact_position is not None:
            self.index = exact_position
            return

        if self.record_id_col == "Use Index":
            print("No matches found")
            return

        if not self.record_id_col or self.record_id_col not in self.columns:
            print("No record ID column is mapped")
            return

        # Partial search fallback. This reads only the ID column, not the full table.
        try:
            id_df = self.source.to_pandas(
                columns=[self.record_id_col],
            )
        except Exception:
            print("Could not search record IDs")
            return

        if id_df.empty or self.record_id_col not in id_df.columns:
            print("No matches found")
            return

        id_series = id_df[self.record_id_col].astype(str)

        matches = id_series.str.contains(sourceid, case=True, na=False)
        n_matches = int(matches.sum())

        if n_matches == 1:
            self.index = int(np.flatnonzero(matches.to_numpy())[0])
            return

        if n_matches == 0:
            print("No matches found")
            return

        exact_matches = id_series == sourceid
        n_exact = int(exact_matches.sum())

        if n_exact == 1:
            self.index = int(np.flatnonzero(exact_matches.to_numpy())[0])
        elif n_exact > 1:
            print(
                f"There are {n_exact} records which exactly match the "
                "provided record ID."
            )
        else:
            print(
                f"There are {n_matches} records containing the provided "
                "record ID; be more specific."
            )

    def _update_navigation_flags(self):
        if not hasattr(self, "prev_button") or not hasattr(self, "next_button"):
            return

        any_running = bool(self._running_panels)
        self.prev_button.disabled = any_running or self.current_position == 0
        self.next_button.disabled = any_running

    def _multithread_running_cb(self, is_running, panel_name):
        """Compatibility callback kept for panels that may report running state."""

        if is_running:
            self._running_panels.add(panel_name)
        else:
            self._running_panels.discard(panel_name)

        self._update_navigation_flags()

    # ---------------------------------------------------------------------
    # Platform selection focus
    # ---------------------------------------------------------------------

    def _has_current_focus_for_active_dataset(self):
        focus = self._get_focus_state()
        if focus is None:
            return False

        return (
            self._focus_value(focus, "dataset_id") == self._dataset_id()
            and self._focus_value(focus, "row_id") is not None
        )

    def _focus_value(self, focus, key: str):
        if focus is None:
            return None

        if isinstance(focus, dict):
            return focus.get(key)

        return getattr(focus, key, None)

    def _get_focus_state(self):
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None

        try:
            return selection.get_focus()
        except Exception:
            return None

    def _sync_index_from_current_focus(self):
        focus = self._get_focus_state()
        if focus is None:
            return False

        if self._focus_value(focus, "dataset_id") != self._dataset_id():
            return False

        row_id = self._focus_value(focus, "row_id")
        if row_id is None:
            return False

        new_index = self._find_index_for_row_id(row_id)
        if new_index is None:
            return False

        if new_index != self.index:
            self.index = new_index

        return True

    def _publish_focus_for_current_index(self):
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return

        row_id = self._get_selected_id()
        if row_id is None:
            return

        dataset_id = self._dataset_id()
        if dataset_id is None:
            return

        focus = self._get_focus_state()
        if focus is not None:
            current_dataset_id = self._focus_value(focus, "dataset_id")
            current_row_id = self._focus_value(focus, "row_id")

            if current_dataset_id == dataset_id and str(current_row_id) == str(row_id):
                return

        selection.set_focus(
            dataset_id=dataset_id,
            row_id=str(row_id),
            origin="record_browser.index",
            panel_id=self.panel_id,
            metadata={
                "row_position": int(self.index),
                "id_column": self.record_id_col or "Use Index",
                "panel_type": "record_browser",
            },
        )

    def _focus_row_from_selection(self, row_id):
        new_index = self._find_index_for_row_id(row_id)
        if new_index is None:
            return

        if hasattr(self, "sourceid_input") and self.sourceid_input.value != str(row_id):
            self.sourceid_input.value = str(row_id)

        if new_index != self.index:
            self._preserve_sourceid_input_once = True
            self.index = new_index
        else:
            if hasattr(self, "extra_info_html"):
                self._refresh_extra_info_view()

    @param.depends("index", watch=True)
    def _update_focus_cb(self):
        self._publish_focus_for_current_index()

        if hasattr(self, "extra_info_html"):
            self._refresh_extra_info_view()

    # ---------------------------------------------------------------------
    # Extra info / metadata table
    # ---------------------------------------------------------------------

    def _format_value(self, value: Any) -> Any:
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and value < 1e4:
            return float(f"{value:.6g}")
        return value

    def _get_extra_info_df(self):
        selected = self._get_current_row_df()

        if selected.empty:
            cols = ["Record ID"] + self.extra_info_cols
            return pd.DataFrame(cols, columns=["Column"])

        row = selected.iloc[0]
        record_id = self._get_selected_id()

        extra_data_list = [["Record ID", record_id]]

        if self.label_col != "No Labels" and self.label_col in selected.columns:
            extra_data_list.append(
                ["Label", self._format_value(row[self.label_col])]
            )

        for col in self.extra_info_cols:
            if col not in selected.columns:
                continue

            value = self._format_value(row[col])
            extra_data_list.append([col, value])

        return pd.DataFrame(extra_data_list, columns=["Column", "Value"])

    def _get_extra_info_html(self):
        df = self._get_extra_info_df()

        if df.empty:
            return (
                "<div style='font-size:12px; color:#777;'>"
                "No record information available."
                "</div>"
            )

        rows = []

        for _, row in df.iterrows():
            key = self._escape_html(row.iloc[0])
            value = self._escape_html(row.iloc[1] if len(row) > 1 else "")

            rows.append(
                "<tr>"
                "<td style='padding:5px 8px; border-bottom:1px solid #eeeeee; "
                "font-weight:700; color:#444; width:38%; vertical-align:top;'>"
                f"{key}"
                "</td>"
                "<td style='padding:5px 8px; border-bottom:1px solid #eeeeee; "
                "color:#333; word-break:break-word; vertical-align:top;'>"
                f"{value}"
                "</td>"
                "</tr>"
            )

        height = self._extra_info_height()

        return (
            f"<div style='max-height:{height}px; overflow-y:auto; "
            "border:1px solid #e6e6e6; border-radius:7px; background:#fafafa;'>"
            "<table style='width:100%; border-collapse:collapse; font-size:12px;'>"
            f"{''.join(rows)}"
            "</table>"
            "</div>"
        )

    def _refresh_extra_info_view(self):
        self.extra_info_html.object = self._get_extra_info_html()

    def _extra_info_height(self):
        n_rows = len(self._get_extra_info_df())
        header = 30
        row_h = 28
        min_h = 56
        max_h = 180
        return max(min_h, min(max_h, header + n_rows * row_h))

    def _initialise_add_remove_columns_widgets(self):
        self.add_column_button = pn.widgets.Button(
            name="Add field",
            button_type="light",
            width=92,
            height=30,
            margin=(0, 0, 0, 0),
        )

        self.remove_column_button = pn.widgets.Button(
            name="Remove field",
            button_type="light",
            width=112,
            height=30,
            margin=(0, 0, 0, 0),
        )

        self.column_selector = pn.widgets.Select(
            name="",
            options=[],
            sizing_mode="stretch_width",
            max_width=320,
            visible=False,
            margin=(0, 0, 0, 0),
        )

        self.selector_watcher = None

        self.add_column_button.on_click(self._add_column_callback)
        self.remove_column_button.on_click(self._remove_column_callback)

    def _add_column_callback(self, _event):
        self.column_selector.value = None

        already = set(self.extra_info_cols)
        reserved = {"ra_dec"}

        options = [
            col for col in self.columns
            if col not in already and col not in reserved
        ]

        self.column_selector.visible = True
        self.column_selector.options = [""] + options

        if self.selector_watcher is not None:
            try:
                self.column_selector.param.unwatch(self.selector_watcher)
            except Exception:
                pass

        self.selector_watcher = self.column_selector.param.watch(
            self._add_extra_feature,
            "value",
        )

        self._rerender_main_layout()

    def _remove_column_callback(self, _event):
        self.column_selector.value = None
        self.column_selector.visible = True
        self.column_selector.options = [""] + list(self.extra_info_cols)

        if self.selector_watcher is not None:
            try:
                self.column_selector.param.unwatch(self.selector_watcher)
            except Exception:
                pass

        self.selector_watcher = self.column_selector.param.watch(
            self._remove_extra_feature,
            "value",
        )

        self._rerender_main_layout()

    def _add_extra_feature(self, event):
        column = event.new

        if column and column not in self.extra_info_cols:
            self.extra_info_cols.append(column)
            self._sync_legacy_settings()
            self._refresh_extra_info_view()

        self.column_selector.visible = False
        self._rerender_main_layout()

    def _remove_extra_feature(self, event):
        column = event.new

        if column and column in self.extra_info_cols:
            self.extra_info_cols.remove(column)
            self._sync_legacy_settings()
            self._refresh_extra_info_view()

        self.column_selector.visible = False
        self._rerender_main_layout()

    # ---------------------------------------------------------------------
    # Label display settings
    # ---------------------------------------------------------------------

    def _initialise_label_selector(self):
        label_column_options = ["No Labels"] + list(self.columns)

        if self.label_col not in label_column_options:
            self.label_col = "No Labels"

        self.label_selector = pn.widgets.Select(
            name="",
            options=label_column_options,
            value=self.label_col,
            sizing_mode="stretch_width",
            max_width=320,
            visible=True,
            margin=(0, 0, 0, 0),
        )

        self.label_editor_layout = pn.Column(
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.confirm_label_button = pn.widgets.Button(
            name="Apply label settings",
            button_type="success",
            width=130,
            height=30,
            visible=False,
            margin=(0, 0, 0, 0),
        )

        self.confirm_label_button.on_click(self._confirm_labels_change_cb)
        self._watchers.append(
            self.label_selector.param.watch(self._update_labels_cb, "value")
        )

        self.labels = []
        self._sync_label_editor_state(self.label_selector.value, keep_button_visible=True)

    def _sync_label_selector_options(self):
        if not hasattr(self, "label_selector"):
            return

        options = ["No Labels"] + list(self.columns)
        self.label_selector.options = options

        if self.label_col not in options:
            self.label_col = self._resolve_label_col()

        if self.label_col not in options:
            self.label_col = "No Labels"

        self.label_selector.value = self.label_col
        self._sync_label_editor_state(self.label_col, keep_button_visible=True)

    def _toggle_labels_section(self, _event=None):
        self.labels_expanded = not self.labels_expanded

        if self.main_layout is not None:
            self._rerender_main_layout()

    def _build_labels_card(self):
        self.labels_toggle_button = pn.widgets.Button(
            name="Labels ▾" if self.labels_expanded else "Labels ▸",
            button_type="light",
            sizing_mode="stretch_width",
            height=30,
            margin=(0, 0, 0, 0),
        )

        self.labels_toggle_button.on_click(self._toggle_labels_section)

        collapsed_styles = {
            "border": "1px solid #d9d9d9",
            "border-radius": "8px",
            "background": "#ffffff",
            "padding": "4px 8px",
            "box-sizing": "border-box",
            "width": "100%",
        }

        expanded_styles = {
            "border": "1px solid #d9d9d9",
            "border-radius": "8px",
            "background": "#ffffff",
            "padding": "10px 12px",
            "box-sizing": "border-box",
            "width": "100%",
        }

        if not self.labels_expanded:
            return pn.Column(
                self.labels_toggle_button,
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 8, 0),
                styles=collapsed_styles,
            )

        label_items = [
            self._body(
                "Choose a label column, then optionally rename labels and "
                "set their colours."
            ),
            pn.Spacer(height=6),
            pn.Column(
                self._field_label("Label column"),
                self.label_selector,
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 0, 0),
            ),
        ]

        if len(self.label_editor_layout.objects) > 0:
            label_items.extend(
                [
                    pn.Spacer(height=6),
                    pn.Column(
                        self._field_label("Label display settings"),
                        self.label_editor_layout,
                        sizing_mode="stretch_width",
                        min_width=0,
                        margin=(0, 0, 0, 0),
                    ),
                ]
            )

        if self.confirm_label_button.visible:
            label_items.extend(
                [
                    pn.Spacer(height=6),
                    pn.Row(
                        self.confirm_label_button,
                        sizing_mode="stretch_width",
                        min_width=0,
                        margin=(0, 0, 0, 0),
                    ),
                ]
            )

        labels_body = pn.Column(
            *label_items,
            sizing_mode="stretch_width",
            min_width=0,
            height=220,
            scroll=True,
            styles={
                "overflow-x": "hidden",
                "padding-right": "2px",
                "box-sizing": "border-box",
            },
            margin=(0, 0, 0, 0),
        )

        return pn.Column(
            self.labels_toggle_button,
            pn.Spacer(height=6),
            labels_body,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(0, 0, 8, 0),
            styles=expanded_styles,
        )

    def _update_labels_cb(self, event):
        self._sync_label_editor_state(event.new, keep_button_visible=True)
        self._rerender_main_layout()

    def _confirm_labels_change_cb(self, _event):
        self.label_col = self.label_selector.value
        self._sync_legacy_settings()

        labels_to_strings, strings_to_labels = self.get_label_strings()
        label_colours = self.get_label_colours()

        if self.config is not None and getattr(self.config, "settings", None) is not None:
            self.config.settings["label_col"] = self.label_col
            self.config.settings["labels"] = self.labels
            self.config.settings["labels_to_strings"] = labels_to_strings
            self.config.settings["strings_to_labels"] = strings_to_labels
            self.config.settings["label_colours"] = label_colours

        self.confirm_label_button.visible = bool(self.labels)
        self._publish_label_settings()
        self._refresh_extra_info_view()
        self._rerender_main_layout()

    def _label_editor_height(self):
        n = len(getattr(self, "labels", []))

        if n == 0:
            return 0

        header_h = 16
        row_h = 30
        gap_h = 3
        padding_h = 8

        total = header_h + padding_h + n * row_h + max(0, n - 1) * gap_h
        return min(118, total)

    def _sync_label_editor_state(self, selected_label_column, keep_button_visible=True):
        if (
            selected_label_column not in self.columns
            or selected_label_column == "No Labels"
        ):
            self.labels = []
            self.label_editor_layout[:] = []
            self.confirm_label_button.visible = False
            return

        unique_values, label_type, exceeded = self._get_unique_label_values(
            selected_label_column,
            max_values=21,
        )

        if exceeded:
            print(
                "You have chosen a column with too many unique values "
                "(possibly continuous); please choose a column with a smaller "
                "set of labels (<=20).",
                flush=True,
            )
            self.labels = []
            self.label_editor_layout[:] = []
            self.confirm_label_button.visible = False
            return

        if label_type == "mixed":
            unique_values = [str(value) for value in unique_values]
        elif label_type == "bool":
            unique_values = [int(value) for value in unique_values]

        try:
            self.labels = sorted(unique_values)
        except TypeError:
            self.labels = sorted(unique_values, key=lambda value: str(value))

        self._build_label_editor_rows()
        self.confirm_label_button.visible = keep_button_visible and bool(self.labels)

    def _build_label_editor_rows(self):
        self.label_to_strings_param = {}
        self.colours_param = {}

        colour_list = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        raw_w = 64
        colour_w = 88

        def header_cell(text):
            return pn.pane.HTML(
                (
                    "<div style='font-size:10px; font-weight:700; "
                    "color:#666; line-height:1.1;'>"
                    f"{self._escape_html(text)}"
                    "</div>"
                ),
                margin=(0, 0, 0, 0),
                sizing_mode="stretch_width",
                height=12,
            )

        rows = [
            pn.Row(
                pn.Column(
                    header_cell("Raw label"),
                    width=raw_w,
                    min_width=raw_w,
                    max_width=raw_w,
                    margin=(0, 0, 0, 0),
                ),
                pn.Column(
                    header_cell("Display name"),
                    sizing_mode="stretch_width",
                    min_width=0,
                    margin=(0, 0, 0, 0),
                ),
                pn.Column(
                    header_cell("Colour"),
                    width=colour_w,
                    min_width=colour_w,
                    max_width=colour_w,
                    margin=(0, 0, 0, 0),
                ),
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 6, 0),
            )
        ]

        for i, label in enumerate(self.labels):
            text_input = pn.widgets.TextInput(
                name="",
                value=str(label),
                placeholder="Display name",
                sizing_mode="stretch_width",
                min_width=0,
                height=26,
                margin=(0, 0, 0, 0),
            )

            picker = pn.widgets.ColorPicker(
                name="",
                value=colour_list[i % len(colour_list)],
                width=58,
                min_width=58,
                max_width=58,
                height=26,
                margin=(0, 0, 0, 0),
            )

            self.label_to_strings_param[f"{label}"] = text_input
            self.colours_param[label] = picker

            raw_chip = pn.pane.HTML(
                (
                    "<div style='font-size:11px; color:#333; background:#f3f3f3; "
                    "border-radius:5px; padding:5px 6px; overflow:hidden; "
                    "text-overflow:ellipsis; white-space:nowrap;'>"
                    f"{self._escape_html(label)}"
                    "</div>"
                ),
                width=raw_w,
                height=24,
                margin=(0, 0, 0, 0),
            )

            colour_cell = pn.Row(
                pn.Spacer(sizing_mode="stretch_width"),
                picker,
                pn.Spacer(sizing_mode="stretch_width"),
                width=colour_w,
                min_width=colour_w,
                max_width=colour_w,
                margin=(0, 0, 0, 0),
                align="center",
            )

            row = pn.Row(
                pn.Column(
                    raw_chip,
                    width=raw_w,
                    min_width=raw_w,
                    max_width=raw_w,
                    margin=(0, 0, 0, 0),
                ),
                pn.Column(
                    text_input,
                    sizing_mode="stretch_width",
                    min_width=0,
                    margin=(0, 0, 0, 0),
                ),
                pn.Column(
                    colour_cell,
                    width=colour_w,
                    min_width=colour_w,
                    max_width=colour_w,
                    margin=(0, 0, 0, 0),
                ),
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 4, 0),
                styles={
                    "width": "100%",
                    "max-width": "100%",
                    "border": "1px solid #e8e8e8",
                    "border-radius": "7px",
                    "padding": "6px 6px",
                    "background": "#ffffff",
                    "box-sizing": "border-box",
                    "align-items": "center",
                    "overflow": "hidden",
                },
            )

            rows.append(row)

        self.label_editor_layout[:] = rows

    def get_label_strings(self):
        labels_to_strings = {}
        strings_to_labels = {}

        for label in self.labels:
            value = self.label_to_strings_param[f"{label}"].value

            if value == "":
                value = str(label)

            labels_to_strings[f"{label}"] = value
            strings_to_labels[f"{value}"] = label

        return labels_to_strings, strings_to_labels

    def get_label_colours(self):
        return {key: param_obj.value for key, param_obj in self.colours_param.items()}

    def _publish_label_settings(self):
        events = getattr(self.context, "events", None)
        if events is None:
            return

        labels_to_strings, strings_to_labels = self.get_label_strings()
        label_colours = self.get_label_colours()

        events.publish(
            "labels.settings.updated",
            {
                "dataset_id": self._dataset_id(),
                "label_col": self.label_col,
                "labels": list(getattr(self, "labels", [])),
                "labels_to_strings": labels_to_strings,
                "strings_to_labels": strings_to_labels,
                "label_colours": label_colours,
                "source": "RecordBrowserPanel",
                "panel_id": self.panel_id,
            },
        )

    # ---------------------------------------------------------------------
    # Public plugin panel API
    # ---------------------------------------------------------------------

    def get_layout(self):
        return self.main_layout

    def get_toolbar(self):
        return None

    def panel(self):
        self._root.sizing_mode = "stretch_both"
        self._root.scroll = True
        self._root.margin = (0, 0, 0, 0)
        self._root.min_width = 0
        self._root.styles = {
            "padding": "4px",
            "box-sizing": "border-box",
        }
        return self._root

    def dispose(self):
        self._disposed = True

        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in list(getattr(self, "_event_subs", [])):
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass

        self._event_subs = []

        for watcher in list(getattr(self, "_watchers", [])):
            try:
                _safe_unwatch(watcher)
            except Exception:
                pass

        self._watchers = []

        if getattr(self, "selector_watcher", None) is not None:
            try:
                self.column_selector.param.unwatch(self.selector_watcher)
            except Exception:
                pass
            self.selector_watcher = None

        self._running_panels = []