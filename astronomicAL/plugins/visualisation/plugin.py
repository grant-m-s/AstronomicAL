from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import datashader as ds
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.palettes import Category10, Category20, Turbo256
from holoviews import streams
from holoviews.operation.datashader import datashade, dynspread, rasterize

from astronomicAL.platform.plugins import PluginManifest

try:
    hv.extension("bokeh")
except Exception:
    pass


PLUGIN_ID = "core.visualisation"
STATE_SERVICE_KEY = f"{PLUGIN_ID}.state"

PLOT_MIN_HEIGHT = 170
SETTINGS_HEIGHT = 118


manifest = PluginManifest(
    id=PLUGIN_ID,
    name="Visualisation",
    version="0.2.3",
    description=(
        "Generic HoloViews visualisation panels for the active dataset. "
        "Panels share X/Y/label state, publish focus and selection sets, "
        "and switch to datashader for large datasets."
    ),
    capabilities=["panel", "datasets", "selection", "visualisation"],
    tags=["core", "visualisation", "plots", "holoviews", "datashader"],
)


def register(api) -> None:
    api.register_service(
        key="state",
        factory=create_visualisation_state,
        lazy=True,
        replace=True,
        description="Shared visualisation state for X/Y variables, label filters, and rendering policy.",
    )

    api.register_panel(
        id="scatter",
        title="Scatter Plot",
        factory=create_scatter_panel,
        description="Interactive XY scatter plot with tap focus, box/lasso selection, labels, and datashader fallback.",
        category="Core / Visualisation",
        icon="scatter_plot",
        tags=["core", "visualisation", "scatter", "selection"],
        required_mappings=["record_id"],
        optional_mappings=["target_label"],
        produces=["selection.focus.changed", "selection.set.changed"],
        default_layout={"x": 0, "y": 0, "w": 3, "h": 4},
    )

    api.register_panel(
        id="histogram",
        title="Histogram",
        factory=create_histogram_panel,
        description="Label-aware histogram using the shared X variable and current focus overlay.",
        category="Core / Visualisation",
        icon="bar_chart",
        tags=["core", "visualisation", "histogram", "labels"],
        required_mappings=["record_id"],
        optional_mappings=["target_label"],
        default_layout={"x": 0, "y": 0, "w": 3, "h": 4},
    )

    api.register_panel(
        id="density",
        title="2D Density",
        factory=create_density_panel,
        description="Responsive 2D density plot using shared X/Y variables and label filters.",
        category="Core / Visualisation",
        icon="grid_on",
        tags=["core", "visualisation", "density", "datashader"],
        required_mappings=["record_id"],
        optional_mappings=["target_label"],
        default_layout={"x": 0, "y": 0, "w": 3, "h": 4},
    )

    api.register_panel(
        id="explorer",
        title="Linked Plot Explorer",
        factory=create_explorer_panel,
        description="Combined scatter, histogram, and density panels sharing one compact settings header.",
        category="Core / Visualisation",
        icon="dashboard",
        tags=["core", "visualisation", "linked", "dashboard"],
        required_mappings=["record_id"],
        optional_mappings=["target_label"],
        produces=["selection.focus.changed", "selection.set.changed"],
        default_layout={"x": 0, "y": 0, "w": 4, "h": 5},
    )


def create_visualisation_state(context, **kwargs):
    return VisualisationState(context=context)


def _get_state(context):
    services = getattr(context, "services", None)
    if services is None:
        return VisualisationState(context=context)

    try:
        state = services.get(STATE_SERVICE_KEY)
    except Exception:
        state = None

    if state is None:
        state = VisualisationState(context=context)
        try:
            services.set(STATE_SERVICE_KEY, state, replace=True, owner=PLUGIN_ID)
        except TypeError:
            services.set(STATE_SERVICE_KEY, state)
        except Exception:
            pass

    return state


def create_scatter_panel(context, **kwargs):
    controller = ScatterPanel(context=context, state=_get_state(context), show_controls=True)
    return controller.panel(), controller


def create_histogram_panel(context, **kwargs):
    controller = HistogramPanel(context=context, state=_get_state(context), show_controls=True)
    return controller.panel(), controller


def create_density_panel(context, **kwargs):
    controller = DensityPanel(context=context, state=_get_state(context), show_controls=True)
    return controller.panel(), controller


def create_explorer_panel(context, **kwargs):
    controller = LinkedExplorerPanel(context=context, state=_get_state(context))
    return controller.panel(), controller


class VisualisationState(param.Parameterized):
    """Shared live UI state for all visualisation plugin panels."""

    x = param.Selector(objects=[], default=None, allow_None=True)
    y = param.Selector(objects=[], default=None, allow_None=True)
    label_filter = param.ListSelector(objects=["All"], default=["All"])
    color_by = param.Selector(objects=["None", "Labels"], default="Labels")

    render_mode = param.Selector(objects=["auto", "interactive", "datashader"], default="auto")
    datashade_threshold = param.Integer(default=50_000, bounds=(1_000, 5_000_000))
    point_size = param.Number(default=5, bounds=(1, 20))
    point_alpha = param.Number(default=0.65, bounds=(0.05, 1.0))

    log_x = param.Boolean(default=False)
    log_y = param.Boolean(default=False)

    bins = param.Integer(default=35, bounds=(2, 300))
    density = param.Boolean(default=False)
    cumulative = param.Boolean(default=False)
    log_density = param.Boolean(default=False)

    def __init__(self, context, **params):
        super().__init__(**params)
        self.context = context
        self.dataset_id: Optional[str] = None
        self.numeric_columns: List[str] = []
        self.record_id_col: Optional[str] = None
        self.label_col: Optional[str] = None
        self.labels_to_strings: Dict[Any, str] = {}
        self.strings_to_labels: Dict[str, Any] = {}
        self.label_colours: Dict[Any, str] = {}
        self.refresh_from_context()

    def refresh_from_context(self) -> None:
        dataset_id = _active_dataset_id(self.context)
        self.dataset_id = dataset_id
        df = _active_df(self.context)

        if df is None or df.empty:
            self.numeric_columns = []
            self.param.x.objects = []
            self.param.y.objects = []
            self.x = None
            self.y = None
            self._reset_labels()
            return

        self.record_id_col = _mapped_column(self.context, dataset_id, "record_id", allow_index=True)
        self.label_col = _mapped_column(self.context, dataset_id, "target_label", allow_index=False)

        self.numeric_columns = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not self.numeric_columns:
            self.param.x.objects = []
            self.param.y.objects = []
            self.x = None
            self.y = None
            self._refresh_label_state(df)
            return

        self.param.x.objects = self.numeric_columns
        self.param.y.objects = self.numeric_columns

        preferred = [column for column in self.numeric_columns if column != self.record_id_col] or list(self.numeric_columns)

        if self.x not in self.numeric_columns:
            self.x = preferred[0]

        if self.y not in self.numeric_columns:
            self.y = preferred[1] if len(preferred) > 1 else preferred[0]

        if len(preferred) > 1 and self.x == self.y:
            self.y = next((c for c in preferred if c != self.x), self.y)

        self._refresh_label_state(df)

    def _reset_labels(self) -> None:
        self.labels_to_strings = {}
        self.strings_to_labels = {}
        self.label_colours = {}
        self.param.label_filter.objects = ["All"]
        self.label_filter = ["All"]
        self.param.color_by.objects = ["None"]
        self.color_by = "None"

    def _refresh_label_state(self, df: pd.DataFrame) -> None:
        cfg = getattr(self.context, "config", None)
        settings = getattr(cfg, "settings", {}) if cfg is not None else {}

        labels_to_strings = dict(settings.get("labels_to_strings", {}) or {})
        strings_to_labels = dict(settings.get("strings_to_labels", {}) or {})
        label_colours = dict(settings.get("label_colours", {}) or {})

        if self.label_col and self.label_col in df.columns:
            raw_labels = sorted(list(pd.Series(df[self.label_col].dropna().unique())), key=lambda value: str(value))
            if not labels_to_strings:
                labels_to_strings = {raw: str(raw) for raw in raw_labels}
            if not strings_to_labels:
                strings_to_labels = {str(display): raw for raw, display in labels_to_strings.items()}
            if not label_colours:
                label_colours = _default_colour_map(raw_labels)

        self.labels_to_strings = labels_to_strings
        self.strings_to_labels = strings_to_labels
        self.label_colours = label_colours

        label_options = ["All"] + sorted([str(k) for k in strings_to_labels.keys()])
        self.param.label_filter.objects = label_options

        current = [value for value in list(self.label_filter or []) if value in label_options]
        self.label_filter = current or ["All"]

        colour_options = ["None"] + (["Labels"] if len(label_options) > 1 else [])
        self.param.color_by.objects = colour_options

        if self.color_by not in colour_options:
            self.color_by = "Labels" if "Labels" in colour_options else "None"

    def apply_label_settings(self, payload: Optional[Dict[str, Any]]) -> None:
        """Apply label display settings published by Record Browser."""
        if not isinstance(payload, dict):
            return

        df = _active_df(self.context)

        label_col = (
            payload.get("label_col")
            or payload.get("label_column")
            or payload.get("active_label_col")
            or payload.get("active_label_column")
            or payload.get("target_label")
        )

        if label_col and df is not None and label_col in df.columns:
            self.label_col = label_col

        labels_to_strings = (
            payload.get("labels_to_strings")
            or payload.get("label_to_string")
            or payload.get("label_strings")
            or payload.get("labels")
            or {}
        )
        strings_to_labels = (
            payload.get("strings_to_labels")
            or payload.get("string_to_label")
            or payload.get("reverse_labels")
            or {}
        )
        label_colours = (
            payload.get("label_colours")
            or payload.get("label_colors")
            or payload.get("colours")
            or payload.get("colors")
            or {}
        )

        if labels_to_strings:
            self.labels_to_strings = dict(labels_to_strings)

        if strings_to_labels:
            self.strings_to_labels = dict(strings_to_labels)
        elif self.labels_to_strings:
            self.strings_to_labels = {str(display): raw for raw, display in self.labels_to_strings.items()}

        if label_colours:
            self.label_colours = dict(label_colours)
        elif self.labels_to_strings:
            self.label_colours = _default_colour_map(list(self.labels_to_strings.keys()))

        label_options = ["All"] + sorted([str(k) for k in self.strings_to_labels.keys()])
        self.param.label_filter.objects = label_options

        current = [value for value in list(self.label_filter or []) if value in label_options]
        self.label_filter = current or ["All"]

        colour_options = ["None"] + (["Labels"] if len(label_options) > 1 else [])
        self.param.color_by.objects = colour_options

        if "Labels" in colour_options:
            self.color_by = "Labels"
        elif self.color_by not in colour_options:
            self.color_by = "None"

    def label_display(self, raw: Any) -> str:
        return str(self.labels_to_strings.get(raw, self.labels_to_strings.get(str(raw), raw)))

    def label_colour(self, raw: Any) -> str:
        return str(self.label_colours.get(raw, self.label_colours.get(str(raw), "#1f77b4")))

    def selected_raw_labels(self) -> List[Any]:
        if "All" in (self.label_filter or []):
            return list(self.strings_to_labels.values())
        return [self.strings_to_labels[name] for name in self.label_filter if name in self.strings_to_labels]

    def get_state(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "label_filter": list(self.label_filter or []),
            "color_by": self.color_by,
            "render_mode": self.render_mode,
            "datashade_threshold": self.datashade_threshold,
            "point_size": self.point_size,
            "point_alpha": self.point_alpha,
            "log_x": self.log_x,
            "log_y": self.log_y,
            "bins": self.bins,
            "density": self.density,
            "cumulative": self.cumulative,
            "log_density": self.log_density,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        self.refresh_from_context()

        for key, value in state.items():
            if key in self.param:
                try:
                    setattr(self, key, value)
                except Exception:
                    pass


class BaseVisualisationPanel(param.Parameterized):
    """Lifecycle-aware base class for visualisation plugin panels."""

    title = "Visualisation"

    def __init__(
        self,
        context,
        state: VisualisationState,
        *,
        show_controls: bool = True,
        show_header: bool = True,
        **params,
    ):
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

        self.settings_visible = False
        self._layout: Optional[pn.Column] = None
        self._settings_built = False

        self.plot_pane = pn.pane.HoloViews(
            sizing_mode="stretch_both",
            height_policy="max",
            min_height=PLOT_MIN_HEIGHT,
            margin=(0, 0, 0, 0),
            styles={"min-height": "0"},
        )

        self.status_pane = pn.pane.HTML(
            "",
            width=185,
            height=34,
            margin=(0, 8, 2, 0),
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

    def _schedule_refresh(self) -> None:
        """Defer selection-driven redraws until HoloViews has finished its callback.

        Calling refresh() synchronously from a HoloViews tap/selection callback can
        replace the Bokeh model while HoloViews is still processing the browser
        message. That is what causes: 'NoneType' object has no attribute
        'document'.
        """
        if self._disposed or self._refresh_scheduled:
            return

        self._refresh_scheduled = True

        def _run():
            self._refresh_scheduled = False
            if not self._disposed:
                self.refresh()

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(_run)
            else:
                _run()
        except Exception:
            _run()

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

    def get_state(self) -> Dict[str, Any]:
        state = self.state.get_state()
        state["settings_visible"] = self.settings_visible
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        if isinstance(state, dict):
            self.settings_visible = bool(state.get("settings_visible", False))

        self.state.restore_state(state)
        self.refresh()
        self._apply_settings_visibility()

    def _df(self) -> Optional[pd.DataFrame]:
        return _active_df(self.context)

    def _dataset_id(self) -> Optional[str]:
        return _active_dataset_id(self.context)

    def _id_values(self, df: pd.DataFrame) -> np.ndarray:
        id_col = self.state.record_id_col
        if id_col == "Use Index" or not id_col or id_col not in df.columns:
            return df.index.astype(str).to_numpy()
        return df[id_col].astype(str).to_numpy()

    def _plot_df(self, require_y: bool = False) -> pd.DataFrame:
        df = self._df()
        if df is None or df.empty or not self.state.x:
            return pd.DataFrame()

        required = [self.state.x]
        if require_y:
            if not self.state.y:
                return pd.DataFrame()
            required.append(self.state.y)

        missing = [column for column in required if column not in df.columns]
        if missing:
            return pd.DataFrame()

        out = df.copy()

        for column in required:
            numeric_values = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            out = out.loc[numeric_values.notna()]

        if self.state.log_x:
            out = out.loc[out[self.state.x] > 0]

        if require_y and self.state.log_y:
            out = out.loc[out[self.state.y] > 0]

        label_col = self.state.label_col
        if label_col and label_col in out.columns and self.state.label_filter and "All" not in self.state.label_filter:
            out = out.loc[out[label_col].isin(self.state.selected_raw_labels())]

        return out

    def _focus_row(self) -> Optional[pd.Series]:
        selection = getattr(self.context, "selection", None)
        if selection is None:
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

        df = self._df()
        if df is None or df.empty:
            return None

        id_values = self._id_values(df)
        matches = np.where(id_values == row_id)[0]
        if len(matches) == 0:
            return None

        return df.iloc[int(matches[0])]

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

    def _on_dataset_event(self, topic, payload) -> None:
        if topic == "labels.settings.updated":
            self.state.apply_label_settings(payload)
        else:
            self.state.refresh_from_context()

        self.refresh()

    def _on_selection_event(self, topic, payload) -> None:
        self._schedule_refresh()

    def _on_state_changed(self, event) -> None:
        self.refresh()

    def refresh(self) -> None:
        if self._disposed:
            return

        try:
            self._render()
        except Exception as exc:
            self.status_pane.object = f"<span style='font-size:12px;color:#b00020'>Plot error: {exc}</span>"
            self.plot_pane.object = self._empty("Plot error")

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
        return dict(
            xlabel=xlabel or self.state.x,
            ylabel=ylabel or self.state.y,
            logx=self.state.log_x,
            logy=self.state.log_y,
            tools=tools or ["pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=active_tools or ["wheel_zoom"],
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            show_grid=True,
            framewise=True,
            axiswise=True,
            shared_axes=False,
            toolbar="right",
        )

    def _settings_controls(self):
        return _settings_box(
            self.status_pane,
            _settings_select(self.state.param.color_by, name="Colour", width=130),
            _settings_multichoice(self.state.param.label_filter, name="Labels", width=200),
            _settings_select(self.state.param.render_mode, name="Render", width=130),
            _settings_int_input(self.state.param.datashade_threshold, name="Shade threshold", width=145),
            _settings_float_slider(self.state.param.point_size, name="Size", width=180),
            _settings_float_slider(self.state.param.point_alpha, name="Alpha", width=180),
            _settings_checkbox(self.state.param.log_x, name="Log X"),
            _settings_checkbox(self.state.param.log_y, name="Log Y"),
        )

    def _header(self):
        return pn.GridBox(
            _header_select(self.state.param.x, name="X"),
            _header_select(self.state.param.y, name="Y"),
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


class ScatterPanel(BaseVisualisationPanel):
    title = "Scatter Plot"

    def _render(self) -> None:
        self._clear_stream_watchers()

        df = self._plot_df(require_y=True)
        if df.empty:
            self.plot_pane.object = self._empty("No finite X/Y data")
            self.status_pane.object = "<span style='font-size:12px'>0 plotted rows</span>"
            return

        use_shader = self.state.render_mode == "datashader" or (
            self.state.render_mode == "auto" and len(df) > int(self.state.datashade_threshold)
        )

        self.status_pane.object = (
            f"<span style='font-size:12px'>{len(df):,} plotted rows · "
            f"{'datashader' if use_shader else 'interactive'}</span>"
        )

        base = self._scatter_datashaded(df) if use_shader else self._scatter_interactive(df)
        overlays = [base, self._selection_overlay(), self._focus_overlay()]

        self.plot_pane.object = hv.Overlay([item for item in overlays if item is not None]).collate().opts(
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=self.state.x,
            ylabel=self.state.y,
            legend_position="right",
            show_grid=True,
            toolbar="right",
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

    def _scatter_interactive(self, df: pd.DataFrame):
        layers = []
        label_col = self.state.label_col
        use_labels = self.state.color_by == "Labels" and label_col and label_col in df.columns

        if use_labels:
            for raw_label, sub in df.groupby(label_col, dropna=False):
                label = self.state.label_display(raw_label)
                colour = self.state.label_colour(raw_label)
                layers.append(self._points_for_df(sub, label=label, colour=colour))
        else:
            layers.append(self._points_for_df(df, label="All", colour="#1f77b4"))

        return hv.Overlay(layers).collate().opts(
            legend_position="right",
            show_grid=True,
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

    def _points_for_df(self, df: pd.DataFrame, *, label: str, colour: str):
        ids = self._id_values(df)

        points = hv.Points(
            (df[self.state.x].to_numpy(), df[self.state.y].to_numpy(), ids),
            kdims=["x", "y"],
            vdims=["id"],
            label=label,
        ).opts(
            size=self.state.point_size,
            color=colour,
            alpha=self.state.point_alpha,
            line_alpha=0,
            selection_alpha=1.0,
            selection_color="orange",
            selection_line_color="black",
            nonselection_alpha=0.18,
            muted_alpha=0.03,
            **self._base_opts(
                xlabel=self.state.x,
                ylabel=self.state.y,
                tools=["tap", "box_select", "lasso_select", "hover", "pan", "wheel_zoom", "box_zoom", "reset"],
                active_tools=["wheel_zoom"],
            ),
        )

        selection_stream = streams.Selection1D(source=points)
        bounds_stream = streams.BoundsXY(source=points)
        last_bounds = {"value": None}

        def on_bounds(event):
            last_bounds["value"] = event.new

        def on_select(event, row_ids=ids):
            indices = list(event.new or [])
            if not indices:
                return

            selected_ids = [str(row_ids[i]) for i in indices if 0 <= i < len(row_ids)]
            self._publish_selection(selected_ids, bounds=last_bounds.get("value"))

        self._watch_param(bounds_stream, on_bounds, "bounds", render_scoped=True)
        self._watch_param(selection_stream, on_select, "index", render_scoped=True)

        return points

    def _scatter_datashaded(self, df: pd.DataFrame):
        label_col = self.state.label_col
        use_labels = self.state.color_by == "Labels" and label_col and label_col in df.columns

        if use_labels:
            plot_df = df[[self.state.x, self.state.y, label_col]].copy()
            plot_df["label_display"] = plot_df[label_col].map(self.state.label_display).astype("category")
            colour_key = {
                self.state.label_display(raw_label): self.state.label_colour(raw_label)
                for raw_label in df[label_col].dropna().unique()
            }
            points = hv.Points(plot_df, kdims=[self.state.x, self.state.y], vdims=["label_display"])
            shaded = datashade(points, aggregator=ds.count_cat("label_display"), color_key=colour_key)
        else:
            points = hv.Points(df, kdims=[self.state.x, self.state.y])
            shaded = datashade(points, aggregator=ds.count(), cmap=["#1f77b4"])

        return dynspread(shaded, threshold=0.75, how="saturate").opts(
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=self.state.x,
            ylabel=self.state.y,
            toolbar="right",
            show_grid=True,
            tools=["pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

    def _publish_selection(self, row_ids: List[str], bounds: Optional[Tuple[float, float, float, float]] = None) -> None:
        dataset_id = self._dataset_id()
        if not row_ids or not dataset_id or not getattr(self.context, "selection", None):
            return

        if len(row_ids) == 1:
            self.context.selection.set_focus(
                dataset_id=dataset_id,
                row_id=row_ids[0],
                origin="core.visualisation.scatter.tap",
                panel_id=self.panel_id,
            )
            return

        metadata = {
            "panel_type": "scatter",
            "x_variable": self.state.x,
            "y_variable": self.state.y,
        }

        if bounds and len(bounds) == 4:
            left, bottom, right, top = bounds
            metadata["geometry"] = {
                "kind": "box",
                "x_variable": self.state.x,
                "y_variable": self.state.y,
                "bounds": [left, right, bottom, top],
            }

        self.context.selection.set_selection_set(
            dataset_id=dataset_id,
            row_ids=row_ids,
            origin="core.visualisation.scatter.selection",
            panel_id=self.panel_id,
            mode="replace",
            metadata=metadata,
            create_artifact=True,
            update_focus_policy="preserve_or_first",
        )

    def _selection_overlay(self):
        ids = self._active_selection_ids()
        if not ids:
            return None

        df = self._df()
        if df is None or df.empty:
            return None

        id_values = self._id_values(df)
        mask = pd.Series(id_values).isin(ids).to_numpy()
        sub = df.loc[mask]

        if sub.empty or self.state.x not in sub.columns or self.state.y not in sub.columns:
            return None

        return hv.Points(sub, kdims=[self.state.x, self.state.y]).opts(
            marker="circle",
            size=max(float(self.state.point_size) + 4, 8),
            fill_alpha=0.0,
            line_color="orange",
            line_width=2,
            active_tools=[],
            logx=self.state.log_x,
            logy=self.state.log_y,
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

    def _focus_overlay(self):
        row = self._focus_row()
        if row is None or self.state.x not in row or self.state.y not in row:
            return None

        return hv.Points([(row[self.state.x], row[self.state.y])], kdims=[self.state.x, self.state.y]).opts(
            marker="circle",
            size=max(float(self.state.point_size) + 8, 12),
            fill_alpha=0.0,
            line_color="black",
            line_width=3,
            active_tools=[],
            logx=self.state.log_x,
            logy=self.state.log_y,
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )


class HistogramPanel(BaseVisualisationPanel):
    title = "Histogram"

    def _settings_controls(self):
        return _settings_box(
            self.status_pane,
            _settings_select(self.state.param.color_by, name="Colour", width=130),
            _settings_multichoice(self.state.param.label_filter, name="Labels", width=200),
            _settings_int_slider(self.state.param.bins, name="Bins", width=180),
            _settings_checkbox(self.state.param.density, name="Density"),
            _settings_checkbox(self.state.param.cumulative, name="Cumulative"),
            _settings_checkbox(self.state.param.log_x, name="Log X"),
            _settings_checkbox(self.state.param.log_y, name="Log Y"),
        )

    def _render(self) -> None:
        df = self._plot_df(require_y=False)
        if df.empty:
            self.plot_pane.object = self._empty("No finite X data")
            self.status_pane.object = "<span style='font-size:12px'>0 histogram rows</span>"
            return

        label_col = self.state.label_col
        use_labels = self.state.color_by == "Labels" and label_col and label_col in df.columns

        layers = []

        if use_labels:
            for raw_label, sub in df.groupby(label_col, dropna=False):
                name = self.state.label_display(raw_label)
                colour = self.state.label_colour(raw_label)
                hist = self._hist(sub[self.state.x].to_numpy(), name, colour)
                if hist is not None:
                    layers.append(hist)
        else:
            hist = self._hist(df[self.state.x].to_numpy(), "All", "#1f77b4")
            if hist is not None:
                layers.append(hist)

        focus = self._focus_row()
        if focus is not None and self.state.x in focus:
            try:
                layers.append(hv.VLine(float(focus[self.state.x])).opts(color="black", line_dash="dashed", line_width=1))
            except Exception:
                pass

        ylabel = "% of rows" if self.state.density else "# rows"

        self.plot_pane.object = hv.Overlay(layers).opts(
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=self.state.x,
            ylabel=ylabel,
            legend_position="right",
            show_grid=True,
            toolbar="right",
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

        self.status_pane.object = f"<span style='font-size:12px'>{len(df):,} histogram rows</span>"

    def _hist(self, values: np.ndarray, label: str, colour: str):
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]

        if self.state.log_x:
            x = x[x > 0]

        if len(x) == 0:
            return None

        if self.state.log_x and np.min(x) > 0 and np.max(x) > np.min(x):
            bins = np.geomspace(np.min(x), np.max(x), int(self.state.bins))
        else:
            bins = int(self.state.bins)

        weights = np.ones_like(x) / len(x) if self.state.density else None
        stats, edges = np.histogram(x, bins=bins, weights=weights)

        if self.state.cumulative:
            stats = np.cumsum(stats)

        return hv.Histogram((edges, stats), kdims=[self.state.x], vdims=["count"], label=label).opts(
            fill_color=colour,
            line_color=colour,
            fill_alpha=0.45,
            line_width=1.5,
            logx=self.state.log_x,
            logy=self.state.log_y,
            tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            show_grid=True,
            toolbar="right",
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )


class DensityPanel(BaseVisualisationPanel):
    title = "2D Density"

    def _settings_controls(self):
        return _settings_box(
            self.status_pane,
            _settings_multichoice(self.state.param.label_filter, name="Labels", width=200),
            _settings_int_slider(self.state.param.bins, name="Bins", width=180),
            _settings_select(self.state.param.render_mode, name="Render", width=130),
            _settings_int_input(self.state.param.datashade_threshold, name="Shade threshold", width=145),
            _settings_checkbox(self.state.param.log_x, name="Log X"),
            _settings_checkbox(self.state.param.log_y, name="Log Y"),
            _settings_checkbox(self.state.param.log_density, name="Log density"),
        )

    def _render(self) -> None:
        df = self._plot_df(require_y=True)
        if df.empty:
            self.plot_pane.object = self._empty("No finite X/Y data")
            self.status_pane.object = "<span style='font-size:12px'>0 density rows</span>"
            return

        use_shader = self.state.render_mode == "datashader" or (
            self.state.render_mode == "auto" and len(df) > int(self.state.datashade_threshold)
        )

        base = self._density_datashaded(df) if use_shader else self._density_hextiles(df)
        focus = self._focus_overlay()

        self.plot_pane.object = hv.Overlay([item for item in [base, focus] if item is not None]).collate().opts(
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=self.state.x,
            ylabel=self.state.y,
            show_grid=True,
            toolbar="right",
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

        self.status_pane.object = (
            f"<span style='font-size:12px'>{len(df):,} density rows · "
            f"{'datashader' if use_shader else 'hexbin'}</span>"
        )

    def _density_hextiles(self, df: pd.DataFrame):
        gridsize = max(5, min(100, int(self.state.bins)))

        return hv.HexTiles(df, kdims=[self.state.x, self.state.y]).opts(
            gridsize=gridsize,
            cmap=Turbo256,
            colorbar=True,
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=self.state.x,
            ylabel=self.state.y,
            logx=self.state.log_x,
            logy=self.state.log_y,
            tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            show_grid=True,
            toolbar="right",
            line_alpha=0.15,
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

    def _density_datashaded(self, df: pd.DataFrame):
        points = hv.Points(df, kdims=[self.state.x, self.state.y])

        return rasterize(points, aggregator=ds.count()).opts(
            cmap=Turbo256,
            colorbar=True,
            logz=self.state.log_density,
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=self.state.x,
            ylabel=self.state.y,
            logx=self.state.log_x,
            logy=self.state.log_y,
            tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            show_grid=True,
            toolbar="right",
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

    def _focus_overlay(self):
        row = self._focus_row()
        if row is None or self.state.x not in row or self.state.y not in row:
            return None

        return hv.Points([(row[self.state.x], row[self.state.y])], kdims=[self.state.x, self.state.y]).opts(
            marker="circle",
            size=14,
            fill_alpha=0.0,
            line_color="black",
            line_width=3,
            active_tools=[],
            logx=self.state.log_x,
            logy=self.state.log_y,
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )


class LinkedExplorerPanel(param.Parameterized):
    """One compact settings header and multiple linked plots sharing the same state service."""

    def __init__(self, context, state: VisualisationState, **params):
        super().__init__(**params)
        self.context = context
        self.state = state
        self.settings_visible = False
        self._layout: Optional[pn.Column] = None
        self._settings_built = False

        self.scatter = ScatterPanel(context=context, state=state, show_controls=False, show_header=False)
        self.histogram = HistogramPanel(context=context, state=state, show_controls=False, show_header=False)
        self.density = DensityPanel(context=context, state=state, show_controls=False, show_header=False)

        self.status_pane = pn.pane.HTML(
            "",
            width=230,
            height=34,
            margin=(0, 8, 2, 0),
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

        self.tabs: Optional[pn.Tabs] = None
        self._watchers: List[Tuple[Any, Any]] = []

        for name in [
            "x",
            "y",
            "label_filter",
            "color_by",
            "render_mode",
            "datashade_threshold",
            "log_x",
            "log_y",
            "bins",
        ]:
            try:
                watcher = self.state.param.watch(self._update_status, name)
                self._watchers.append((self.state, watcher))
            except Exception:
                pass

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

    def _update_status(self, _event=None) -> None:
        df = _active_df(self.context)
        rows = 0 if df is None else len(df)
        self.status_pane.object = (
            f"<span style='font-size:12px'>{rows:,} rows · "
            f"X: {self.state.x or '-'} · Y: {self.state.y or '-'}</span>"
        )

    def _settings_controls(self):
        return _settings_box(
            self.status_pane,
            _settings_select(self.state.param.color_by, name="Colour", width=130),
            _settings_multichoice(self.state.param.label_filter, name="Labels", width=200),
            _settings_select(self.state.param.render_mode, name="Scatter render", width=140),
            _settings_int_input(self.state.param.datashade_threshold, name="Shade threshold", width=145),
            _settings_float_slider(self.state.param.point_size, name="Size", width=180),
            _settings_float_slider(self.state.param.point_alpha, name="Alpha", width=180),
            _settings_int_slider(self.state.param.bins, name="Bins", width=180),
            _settings_checkbox(self.state.param.log_x, name="Log X"),
            _settings_checkbox(self.state.param.log_y, name="Log Y"),
            _settings_checkbox(self.state.param.log_density, name="Log density"),
        )

    def _header(self):
        return pn.GridBox(
            _header_select(self.state.param.x, name="X"),
            _header_select(self.state.param.y, name="Y"),
            self.settings_button,
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

    def _build_tabs(self):
        if self.tabs is None:
            self.tabs = pn.Tabs(
                ("Scatter", self.scatter.panel()),
                ("Histogram", self.histogram.panel()),
                ("2D Density", self.density.panel()),
                dynamic=True,
                sizing_mode="stretch_both",
                height_policy="max",
                margin=(0, 0, 0, 0),
                styles={
                    "min-height": "0",
                    "overflow": "hidden",
                },
            )
        return self.tabs

    def panel(self):
        self._update_status()
        self._ensure_settings_built()
        self._apply_settings_visibility()

        self._layout = pn.Column(
            self._header(),
            self.settings_pane,
            self._build_tabs(),
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

    def get_state(self) -> Dict[str, Any]:
        state = self.state.get_state()
        state["settings_visible"] = self.settings_visible
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        if isinstance(state, dict):
            self.settings_visible = bool(state.get("settings_visible", False))

        self.state.restore_state(state)

        for child in (self.scatter, self.histogram, self.density):
            try:
                child.refresh()
            except Exception:
                pass

        self._update_status()
        self._apply_settings_visibility()

    def dispose(self) -> None:
        for owner, watcher in list(self._watchers):
            try:
                owner.param.unwatch(watcher)
            except Exception:
                pass
        self._watchers.clear()

        for child in (self.scatter, self.histogram, self.density):
            try:
                child.dispose()
            except Exception:
                pass


def _header_select(parameter, *, name: str):
    return pn.widgets.Select.from_param(
        parameter,
        name=name,
        sizing_mode="stretch_width",
        height=44,
        margin=(0, 2, 0, 0),
    )


def _settings_select(parameter, *, name: str, width: int = 145):
    return pn.widgets.Select.from_param(
        parameter,
        name=name,
        width=width,
        height=44,
        sizing_mode="fixed",
        margin=(0, 8, 4, 0),
    )


def _settings_multichoice(parameter, *, name: str, width: int = 200):
    widget = pn.widgets.MultiChoice.from_param(
        parameter,
        name=name,
        width=width,
        height=44,
        sizing_mode="fixed",
        margin=(0, 8, 4, 0),
    )

    if "allow_html" in widget.param:
        try:
            widget.allow_html = False
        except Exception:
            pass

    return widget


def _settings_int_input(parameter, *, name: str, width: int = 140):
    return pn.widgets.IntInput.from_param(
        parameter,
        name=name,
        width=width,
        height=44,
        sizing_mode="fixed",
        margin=(0, 8, 4, 0),
    )


def _settings_float_slider(parameter, *, name: str, width: int = 180):
    return pn.widgets.FloatSlider.from_param(
        parameter,
        name=name,
        width=width,
        height=44,
        sizing_mode="fixed",
        margin=(0, 10, 4, 0),
    )


def _settings_int_slider(parameter, *, name: str, width: int = 180):
    return pn.widgets.IntSlider.from_param(
        parameter,
        name=name,
        width=width,
        height=44,
        sizing_mode="fixed",
        margin=(0, 10, 4, 0),
    )


def _settings_checkbox(parameter, *, name: str):
    return pn.widgets.Checkbox.from_param(
        parameter,
        name=name,
        width=110,
        height=30,
        sizing_mode="fixed",
        margin=(14, 10, 2, 0),
    )


def _settings_box(*controls):
    return pn.FlexBox(
        *controls,
        sizing_mode="stretch_width",
        height_policy="fit",
        margin=(0, 0, 0, 0),
        styles={
            "overflow": "visible",
            "align-content": "flex-start",
            "align-items": "flex-start",
            "gap": "4px 8px",
            "padding": "4px 6px 2px 6px",
            "border-top": "1px solid #ddd",
            "border-bottom": "1px solid #eee",
            "background": "#fafafa",
            "box-sizing": "border-box",
        },
    )


def _active_dataset_id(context) -> Optional[str]:
    try:
        return context.datasets.active_id()
    except Exception:
        return None


def _active_df(context) -> Optional[pd.DataFrame]:
    try:
        return context.datasets.get_df()
    except Exception:
        return None


def _mapped_column(context, dataset_id: Optional[str], semantic: str, *, allow_index: bool) -> Optional[str]:
    datasets = getattr(context, "datasets", None)

    if datasets is not None and dataset_id is not None:
        try:
            value = datasets.get_mapping(dataset_id, semantic)
            if value == "Use Index" and allow_index:
                return value

            df = datasets.get_df(dataset_id)
            if value in df.columns:
                return value
        except Exception:
            pass

    cfg = getattr(context, "config", None)
    settings = getattr(cfg, "settings", {}) if cfg is not None else {}
    df = _active_df(context)

    if semantic == "record_id":
        value = settings.get("id_col")
        if value == "Use Index" and allow_index:
            return value
        if df is not None and value in df.columns:
            return value

    if semantic == "target_label":
        value = settings.get("label_col")
        if df is not None and value in df.columns:
            return value

    return None


def _default_colour_map(raw_labels: Iterable[Any]) -> Dict[Any, str]:
    labels = list(raw_labels)
    if not labels:
        return {}

    palette = Category10[10] if len(labels) <= 10 else Category20[20]
    return {label: palette[index % len(palette)] for index, label in enumerate(labels)}