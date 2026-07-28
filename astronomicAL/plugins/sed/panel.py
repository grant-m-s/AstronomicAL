from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import html
import traceback

import pandas as pd
import panel as pn
from bokeh.models import (
    ColumnDataSource,
    CustomJSTickFormatter,
    HoverTool,
    LinearAxis,
    NormalHead,
    Whisker,
)
from bokeh.plotting import figure

from .service import SEDRuntime, SED_UNIT_OPTIONS, normalise_sed_unit

PLUGIN_ID = "astro.sed"
RUNTIME_SERVICE_KEY = f"{PLUGIN_ID}.runtime"
SED_ARTIFACT_TYPE = "astro.sed.broadband"
MIN_POINTS_TO_PLOT = 3
SHOW_SED_ERROR_BARS_BY_DEFAULT = True
SHOW_AB_MAGNITUDE_AXIS_BY_DEFAULT = True
UPPER_LIMIT_ARROW_SCALE = 0.6
PLOT_UNIT_MICROJY = "microJy"
PLOT_UNIT_NUFNU = "erg/s/cm² (νFν)"
PLOT_UNIT_OPTIONS = [PLOT_UNIT_MICROJY, PLOT_UNIT_NUFNU]
SPEED_OF_LIGHT_CM_S = 2.99792458e10

def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)

def _payload_records(payload: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    if not payload:
        return []

    records = payload.get("records", [])
    if isinstance(records, list):
        return [dict(item) for item in records if isinstance(item, Mapping)]

    return []

def _assign_sed_plot(
    pane: pn.pane.Bokeh,
    plot: Any,
    *,
    visible: bool,
) -> bool:
    """Assign a native Bokeh model after establishing its rendered visibility.

    The initial pane starts hidden. Setting ``visible`` before replacing ``object``
    ensures Panel creates or updates the mounted Bokeh child for the real plot,
    rather than retaining the empty model that was present during first layout
    construction.
    """
    pane.visible = bool(visible)
    pane.object = plot

    # Explicitly notify Param after a whole Bokeh model replacement. This is
    # harmless on a live server and also covers embedded/notebook contexts.
    try:
        pane.param.trigger("object")
    except Exception:
        pass

    return pane.object is plot and pane.visible is bool(visible)

def microjy_to_abmag(flux_uJy: float) -> float:
    """Convert microJy to AB magnitude."""
    return 23.9 - 2.5 * math.log10(float(flux_uJy))

def _microjy_to_nufnu(flux_uJy: float, wavelength_um: float) -> float:
    """Convert F_nu in microJy to nu F_nu in erg/s/cm^2."""
    return (SPEED_OF_LIGHT_CM_S / (float(wavelength_um) * 1.0e-4)) * float(flux_uJy) * 1.0e-29

def _normalise_plot_unit(value: Any) -> str:
    value = str(value or PLOT_UNIT_MICROJY).strip()
    aliases = {
        "microjy": PLOT_UNIT_MICROJY,
        "ujy": PLOT_UNIT_MICROJY,
        "µjy": PLOT_UNIT_MICROJY,
        "flux_density": PLOT_UNIT_MICROJY,
        "flux density": PLOT_UNIT_MICROJY,
        "nufnu": PLOT_UNIT_NUFNU,
        "nu fnu": PLOT_UNIT_NUFNU,
        "nuFnu": PLOT_UNIT_NUFNU,
        "erg/s": PLOT_UNIT_NUFNU,
        "erg/s/cm2": PLOT_UNIT_NUFNU,
        "erg/s/cm^2": PLOT_UNIT_NUFNU,
        "erg/s/cm²": PLOT_UNIT_NUFNU,
    }
    return aliases.get(value.lower(), value if value in PLOT_UNIT_OPTIONS else PLOT_UNIT_MICROJY)

def _add_ab_magnitude_axis(fig: Any) -> None:
    """Add a right-hand AB-magnitude axis sharing the flux-density range."""
    formatter = CustomJSTickFormatter(
        code="""
        if (tick <= 0 || !isFinite(tick)) {
            return "";
        }
        const mag = 23.9 - 2.5 * Math.log10(tick);
        return mag.toFixed(1);
        """
    )
    fig.add_layout(
        LinearAxis(
            axis_label="AB magnitude",
            formatter=formatter,
            major_label_text_color="black",
            axis_label_text_color="black",
        ),
        "right",
    )

def _positive_error_floor(values: pd.Series) -> float:
    positive = pd.to_numeric(values, errors="coerce")
    positive = positive[positive > 0]
    if positive.empty:
        return 1.0e-30
    return max(float(positive.min()) * 1.0e-6, 1.0e-300)

def _add_vertical_whiskers(
    fig: Any,
    data: pd.DataFrame,
    *,
    y_col: str,
    lower_col: str,
    upper_col: str,
    lower_head: Any = None,
    upper_head: Any = None,
) -> None:
    if data.empty:
        return
    source = ColumnDataSource(
        {
            "base": data["wavelength (µm)"].tolist(),
            "lower": data[lower_col].tolist(),
            "upper": data[upper_col].tolist(),
        }
    )
    fig.add_layout(
        Whisker(
            source=source,
            base="base",
            lower="lower",
            upper="upper",
            lower_head=lower_head,
            upper_head=upper_head,
            line_color="black",
            line_width=1.5,
        )
    )

def _add_horizontal_whiskers(fig: Any, data: pd.DataFrame, *, y_col: str) -> None:
    if data.empty:
        return
    half_fwhm = 0.5 * data["FWHM"]
    x_floor = _positive_error_floor(data["wavelength (µm)"])
    lower = (data["wavelength (µm)"] - half_fwhm).clip(lower=x_floor)
    upper = data["wavelength (µm)"] + half_fwhm
    source = ColumnDataSource(
        {
            "base": data[y_col].tolist(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        }
    )
    fig.add_layout(
        Whisker(
            source=source,
            base="base",
            lower="lower",
            upper="upper",
            dimension="width",
            lower_head=None,
            upper_head=None,
            line_color="black",
            line_width=1.5,
        )
    )

def create_sed_plot(
    records: Sequence[Mapping[str, Any]],
    *,
    show_error_bars: bool = SHOW_SED_ERROR_BARS_BY_DEFAULT,
    show_fwhm_error_bars: bool = True,
    show_magnitude_axis: bool = SHOW_AB_MAGNITUDE_AXIS_BY_DEFAULT,
    plot_unit: str = PLOT_UNIT_MICROJY,
    upper_limit_arrow_scale: float = UPPER_LIMIT_ARROW_SCALE,
) -> Any:
    """Create the broadband SED as a native Bokeh figure.

    The plot is built directly as a Bokeh model and assigned to a Bokeh pane.
    """
    plot_unit = _normalise_plot_unit(plot_unit)
    use_nufnu = plot_unit == PLOT_UNIT_NUFNU
    show_magnitude_axis = bool(show_magnitude_axis and not use_nufnu)

    y_col = "nuFnu_erg_s_cm2" if use_nufnu else "flux_uJy"
    y_err_col = "nuFnu_error_erg_s_cm2" if use_nufnu else "flux_error_uJy"
    y_label = "νFν (erg s⁻¹ cm⁻²)" if use_nufnu else "flux density (µJy)"

    fig = figure(
        x_axis_type="log",
        y_axis_type="log",
        x_axis_label="wavelength (µm)",
        y_axis_label=y_label,
        sizing_mode="stretch_width",
        height=320,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        toolbar_location="above",
    )
    fig.min_border_left = 65
    fig.min_border_right = 65 if show_magnitude_axis else 10

    if show_magnitude_axis:
        _add_ab_magnitude_axis(fig)

    data = pd.DataFrame([dict(item) for item in records or []])
    required = {"wavelength (µm)", "flux_uJy"}
    if data.empty or not required.issubset(data.columns):
        return fig

    for column in ("wavelength (µm)", "flux_uJy", "flux_error_uJy", "FWHM"):
        if column not in data.columns:
            data[column] = 0.0
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["wavelength (µm)", "flux_uJy"])
    data = data[(data["wavelength (µm)"] > 0) & (data["flux_uJy"] > 0)]
    data = data.sort_values("wavelength (µm)").reset_index(drop=True)
    if data.empty:
        return fig

    if use_nufnu:
        scale = SPEED_OF_LIGHT_CM_S / (data["wavelength (µm)"] * 1.0e-4) * 1.0e-29
        data[y_col] = data["flux_uJy"] * scale
        data[y_err_col] = data["flux_error_uJy"] * scale
    else:
        data[y_col] = data["flux_uJy"]
        data[y_err_col] = data["flux_error_uJy"]

    data = data.dropna(subset=[y_col])
    data = data[data[y_col] > 0].copy()
    if len(data) < MIN_POINTS_TO_PLOT:
        return fig

    data["is_upper_limit"] = data[y_err_col] < 0
    data["has_larger_error"] = data[y_err_col] > data[y_col]
    data["is_detection"] = ~(data["is_upper_limit"] | data["has_larger_error"])

    detection_data = data[data["is_detection"]]
    if len(detection_data) >= 2:
        fig.line(
            detection_data["wavelength (µm)"].tolist(),
            detection_data[y_col].tolist(),
            line_width=2,
        )

    source_data = data.copy()
    for column in (
        "band",
        "input_value",
        "input_error",
        "input_unit",
        "value_column",
        "error_column",
    ):
        if column not in source_data.columns:
            source_data[column] = ""
    source = ColumnDataSource(source_data)
    points = fig.scatter(
        x="wavelength (µm)",
        y=y_col,
        source=source,
        marker="circle",
        size=8,
        fill_alpha=0.8,
        line_alpha=0.8,
    )

    hover_tooltips = [
        ("band", "@band"),
        ("wavelength", "@{wavelength (µm)}{0.0000} µm"),
        ("flux", "@flux_uJy{0.0000} µJy"),
        ("flux error", "@flux_error_uJy{0.0000} µJy"),
        ("input", "@input_value{0.0000} @input_unit"),
        ("value column", "@value_column"),
        ("error column", "@error_column"),
    ]
    if use_nufnu:
        hover_tooltips.insert(3, ("νFν", f"@{{{y_col}}}{{0.000e}}"))
    fig.add_tools(HoverTool(renderers=[points], tooltips=hover_tooltips))

    if show_error_bars:
        floor = _positive_error_floor(data[y_col])

        good_error_data = data[
            data["is_detection"]
            & data[y_err_col].notna()
            & (data[y_err_col] > 0)
        ].copy()
        if not good_error_data.empty:
            good_error_data["error_lower"] = (
                good_error_data[y_col] - good_error_data[y_err_col]
            ).clip(lower=floor)
            good_error_data["error_upper"] = good_error_data[y_col] + good_error_data[y_err_col]
            _add_vertical_whiskers(
                fig,
                good_error_data,
                y_col=y_col,
                lower_col="error_lower",
                upper_col="error_upper",
            )

        upper_limit_data = data[data["is_upper_limit"]].copy()
        if not upper_limit_data.empty:
            upper_limit_data["error_lower"] = (
                upper_limit_data[y_col] * (1.0 - float(upper_limit_arrow_scale))
            ).clip(lower=floor)
            upper_limit_data["error_upper"] = upper_limit_data[y_col]
            _add_vertical_whiskers(
                fig,
                upper_limit_data,
                y_col=y_col,
                lower_col="error_lower",
                upper_col="error_upper",
                lower_head=NormalHead(size=8),
                upper_head=None,
            )

        larger_error_data = data[data["has_larger_error"]].copy()
        if not larger_error_data.empty:
            larger_error_data["error_lower"] = (
                larger_error_data[y_col] * (1.0 - float(upper_limit_arrow_scale))
            ).clip(lower=floor)
            larger_error_data["error_upper"] = (
                larger_error_data[y_col] + larger_error_data[y_err_col]
            )
            _add_vertical_whiskers(
                fig,
                larger_error_data,
                y_col=y_col,
                lower_col="error_lower",
                upper_col="error_upper",
                lower_head=NormalHead(size=8),
                upper_head=None,
            )

        if show_fwhm_error_bars:
            x_error_data = data[data["FWHM"].notna() & (data["FWHM"] > 0)].copy()
            _add_horizontal_whiskers(fig, x_error_data, y_col=y_col)

    return fig

class BroadbandSEDPanel:
    state_version = 3

    def __init__(
        self,
        context,
        *,
        instance_id: Optional[str] = None,
        restore_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.context = context
        self.instance_id = instance_id or f"{PLUGIN_ID}.panel"
        self.restore_state = dict(restore_state or {})
        self.subscriptions: List[Any] = []
        self.job_handles: List[Any] = []
        self._disposed = False
        self._ui_loaded = False
        self._pending_refresh = False
        self._build_generation = 0
        self._active_job_handle: Optional[Any] = None
        self.runtime: SEDRuntime = self._get_runtime()

        self.sed_file: Optional[str] = self.restore_state.get("sed_file")

        self.column_overrides: Dict[str, str] = dict(
            self.restore_state.get("column_overrides") or {}
        )
        self.pending_column_overrides: Dict[str, str] = dict(self.column_overrides)

        self.error_column_overrides: Dict[str, str] = dict(
            self.restore_state.get("error_column_overrides") or {}
        )
        self.pending_error_column_overrides: Dict[str, str] = dict(
            self.error_column_overrides
        )

        self.unit_overrides: Dict[str, str] = {
            str(key): normalise_sed_unit(value)
            for key, value in dict(self.restore_state.get("unit_overrides") or {}).items()
        }
        self.pending_unit_overrides: Dict[str, str] = dict(self.unit_overrides)

        restored_assign_all_unit = (
            self.restore_state.get("assign_all_unit_value")
            or self.restore_state.get("unit_for_all_filters")
        )
        self.assign_all_unit_value: Optional[str] = (
            normalise_sed_unit(restored_assign_all_unit)
            if restored_assign_all_unit
            else None
        )

        self.show_fwhm_error_bars: bool = bool(
            self.restore_state.get("show_fwhm_error_bars", True)
        )
        self.plot_unit: str = _normalise_plot_unit(
            self.restore_state.get("plot_unit", PLOT_UNIT_MICROJY)
        )
        self.show_flux_table: bool = bool(
            self.restore_state.get("show_flux_table", False)
        )
        self.settings_expanded: bool = bool(
            self.restore_state.get("settings_expanded", False)
        )
        self.latest_payload: Optional[Dict[str, Any]] = None
        self.latest_payload_artifact_id: Optional[str] = None

        self.current_dataset_id: Optional[str] = None
        self.current_row_id: Optional[str] = None
        self.latest_artifact_id: Optional[str] = self.restore_state.get("latest_artifact_id")
        self._latest_build_context: Dict[str, str] = {}

        self.file_select = pn.widgets.Select(
            name="SED photometry-band file",
            options=[],
            value=None,
            sizing_mode="stretch_width",
            min_width=360,
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="primary",
            width=90,
        )
        self.create_file_button = pn.widgets.Button(
            name="Create new SED data file",
            button_type="default",
            width=190,
        )
        self.clear_mappings_button = pn.widgets.Button(
            name="Clear local SED mappings",
            button_type="default",
            width=190,
        )
        self.settings_button = pn.widgets.Button(
            name="⚙",
            width=32,
            height=32,
            button_type="default",
            sizing_mode="fixed",
            margin=(14, 0, 0, 0),
        )
        self.show_fwhm_checkbox = pn.widgets.Checkbox(
            name="Show horizontal FWHM error bars",
            value=self.show_fwhm_error_bars,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        self.plot_unit_select = pn.widgets.Select(
            name="Plotting units",
            options=PLOT_UNIT_OPTIONS,
            value=self.plot_unit,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        self.show_flux_table_checkbox = pn.widgets.Checkbox(
            name="Show tabular flux data",
            value=self.show_flux_table,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        self.plot_settings_panel = pn.Column(
            self.show_fwhm_checkbox,
            self.plot_unit_select,
            self.show_flux_table_checkbox,
            sizing_mode="stretch_width",
            visible=self.settings_expanded,
            margin=(0, 0, 12, 0),
        )
        self.settings_box = self.plot_settings_panel
        self.status = pn.pane.Alert(
            "Select a SED photometry-band file and focus a row.",
            alert_type="info",
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        self.mapping_table: Optional[pn.widgets.Tabulator] = None
        self.mapping_reference_select: Optional[pn.widgets.Select] = None
        self.mapping_column_select: Optional[pn.widgets.Select] = None
        self.mapping_error_column_select: Optional[pn.widgets.Select] = None
        self.mapping_unit_select: Optional[pn.widgets.Select] = None
        self.assign_all_unit_select: Optional[pn.widgets.Select] = None
        self.mapping_content: Optional[pn.Column] = None
        self.mapping_toggle_button: Optional[pn.widgets.Button] = None

        self.last_valid_point_count: int = 0
        self.mapping_ui_signature: Optional[tuple] = None
        self.mapping_ui_changed_this_refresh: bool = False
        self.mapping_controls_expanded: bool = bool(
            self.restore_state.get("mapping_controls_expanded", True)
        )

        # Diagnostics should not appear before the user has actually tried to
        # apply mappings. Otherwise the panel looks broken before the user has
        # done anything.
        self.mapping_applied_once: bool = bool(
            self.restore_state.get("mapping_applied_once", False)
        )

        self.mapping_box = pn.Column(
            sizing_mode="stretch_width",
            margin=(14, 0, 18, 0),
            visible=False,
            styles={
                "border": "1px solid #d8d8d8",
                "border-radius": "6px",
                "padding": "12px",
                "background": "#fbfbfb",
                "box-sizing": "border-box",
                "overflow": "visible",
            },
        )
        self.plot_placeholder = pn.pane.Alert(
            (
                "SED plot will appear once at least "
                f"{MIN_POINTS_TO_PLOT} finite broadband points are available."
            ),
            alert_type="info",
            sizing_mode="stretch_width",
            visible=True,
            margin=(12, 0, 12, 0),
        )
        self.plot_pane = pn.pane.Bokeh(
            create_sed_plot(
                [],
                show_fwhm_error_bars=self.show_fwhm_error_bars,
                show_magnitude_axis=self.plot_unit == PLOT_UNIT_MICROJY,
                plot_unit=self.plot_unit,
            ),
            sizing_mode="stretch_width",
            height=320,
            min_height=260,
            margin=(12, 0, 14, 0),
            visible=False,
        )
        self.table_pane = pn.Column(
            sizing_mode="stretch_width",
            margin=(6, 0, 0, 0),
            visible=self.show_flux_table,
        )
        self.skipped_pane = pn.Column(
            sizing_mode="stretch_width",
            margin=(6, 0, 0, 0),
        )
        self.view = pn.Column(
            self.file_select,
            pn.Row(
                self.refresh_button,
                self.create_file_button,
                self.clear_mappings_button,
                self.settings_button,
                sizing_mode="stretch_width",
                margin=(8, 0, 10, 0),
            ),
            self.status,
            self.plot_pane,
            self.plot_settings_panel,
            self.mapping_box,
            self.plot_placeholder,
            self.table_pane,
            self.skipped_pane,
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
            styles={
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "padding": "0 8px 12px 8px",
            },
        )

        self._refresh_file_options()
        self.file_select.param.watch(self._on_file_selected, "value")
        self.refresh_button.on_click(lambda event: self.refresh(refresh_file_options=True))
        self.create_file_button.on_click(self._on_create_file)
        self.clear_mappings_button.on_click(self._on_clear_mappings)
        self.settings_button.on_click(self._on_toggle_settings)
        self.show_fwhm_checkbox.param.watch(self._on_plot_settings_changed, "value")
        self.plot_unit_select.param.watch(self._on_plot_settings_changed, "value")
        self.show_flux_table_checkbox.param.watch(self._on_plot_settings_changed, "value")
        self._subscribe()

        # Build the first focused SED only after the browser has mounted the
        # Panel document. A next-tick callback is too early during application
        # startup and can update the hidden placeholder model instead of the
        # mounted Bokeh pane. ``pn.state.onload`` also executes immediately when
        # this panel is opened after the session has already loaded.
        self._register_onload_callback()

    # ------------------------------------------------------------------
    # Platform access
    # ------------------------------------------------------------------

    def _get_runtime(self) -> SEDRuntime:
        services = getattr(self.context, "services", None)
        if services is not None:
            try:
                runtime = services.require(RUNTIME_SERVICE_KEY)
                if runtime is not None:
                    return runtime
            except Exception:
                pass

            try:
                runtime = services.get(RUNTIME_SERVICE_KEY)
                if runtime is not None:
                    return runtime
            except Exception:
                pass

        return SEDRuntime()

    def _get_focus(self):
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None

        try:
            return selection.get_focus()
        except Exception:
            return None

    def _schedule_ui_callback(self, callback: Callable[[], None]) -> None:
        def _run() -> None:
            if not self._disposed:
                callback()

        execute = getattr(pn.state, "execute", None)
        if callable(execute):
            try:
                execute(_run)
                return
            except Exception:
                pass

        document = getattr(pn.state, "curdoc", None)
        if document is not None:
            try:
                document.add_next_tick_callback(_run)
                return
            except Exception:
                pass

        _run()

    def _register_onload_callback(self) -> None:
        onload = getattr(pn.state, "onload", None)
        if callable(onload):
            try:
                onload(self._on_ui_loaded)
                return
            except Exception:
                pass

        self._schedule_ui_callback(self._on_ui_loaded)

    def _on_ui_loaded(self) -> None:
        if self._disposed or self._ui_loaded:
            return

        self._ui_loaded = True
        self._pending_refresh = False
        self._initialise_from_focus()

    def _initialise_from_focus(self) -> None:
        focus = self._get_focus()
        if (
            focus is not None
            and getattr(focus, "dataset_id", None)
            and getattr(focus, "row_id", None)
        ):
            self.current_dataset_id = str(focus.dataset_id)
            self.current_row_id = str(focus.row_id)
            self.refresh()
            return

        self._set_status("Focus a row to plot its SED.", "info")

    def _active_dataset_id(self) -> Optional[str]:
        try:
            return self.context.datasets.active_id()
        except Exception:
            return None

    def _dataset_columns(self, dataset_id: Optional[str]) -> List[str]:
        if not dataset_id:
            return []

        try:
            return [str(col) for col in self.context.datasets.list_columns(dataset_id)]
        except Exception:
            try:
                return [str(col) for col in self.context.datasets.get_df(dataset_id, limit=0).columns]
            except Exception:
                return []

    def _record_id_column(self, dataset_id: str) -> str:
        datasets = self.context.datasets

        for method_name in ("get_mapping", "mapping", "get_column_mapping"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    value = method(dataset_id, "record_id")
                except Exception:
                    value = None

                if value:
                    return str(value)

        # Fallbacks are here only for transitional use. The registered panel
        # declares required_mappings=["record_id"], so the mapping gate should
        # normally resolve this before construction.
        columns = set(self._dataset_columns(dataset_id))
        for candidate in ("id", "ID", "source_id", "object_id", "row_id"):
            if candidate in columns:
                return candidate

        raise KeyError("No `record_id` mapping is available for the active dataset.")

    def _get_row(self, dataset_id: str, row_id: str):
        id_column = self._record_id_column(dataset_id)

        if id_column == "Use Index":
            try:
                return self.context.datasets.get_row_by_position(dataset_id, int(row_id))
            except Exception:
                df = self.context.datasets.get_df(dataset_id, limit=None)
                return df.iloc[[int(row_id)]]

        try:
            return self.context.datasets.get_row_by_id(
                dataset_id,
                row_id,
                id_column=id_column,
            )
        except Exception:
            # Last-resort compatibility path.
            df = self.context.datasets.get_df(dataset_id, limit=None)
            if id_column not in df.columns:
                raise
            return df[df[id_column].astype(str) == str(row_id)].head(1)

    # ------------------------------------------------------------------
    # UI / state helpers
    # ------------------------------------------------------------------

    def _set_mapping_content_visible(self, visible: bool) -> None:
        self.mapping_controls_expanded = bool(visible)

        if self.mapping_content is not None:
            self.mapping_content.visible = bool(visible)

        if self.mapping_toggle_button is not None:
            self.mapping_toggle_button.name = (
                "Hide mapping controls" if visible else "Show mapping controls"
            )

    def _clear_mapping_controls(self) -> None:
        self.mapping_table = None
        self.mapping_reference_select = None
        self.mapping_column_select = None
        self.mapping_error_column_select = None
        self.mapping_unit_select = None
        self.assign_all_unit_select = None
        self.mapping_content = None
        self.mapping_toggle_button = None
        self.mapping_ui_signature = None
        self.mapping_ui_changed_this_refresh = True
        self.mapping_box.objects = []
        self.mapping_box.visible = False

    def _set_status(self, message: str, alert_type: str = "info") -> None:
        self.status.object = message
        self.status.alert_type = alert_type
        self.status.margin = (8, 0, 12, 0)

    def _sed_unit_value(self, value: Any, *, default: str = "ABmag") -> str:
        """Return a unit value that is safe to put into the SED unit dropdown."""
        unit = normalise_sed_unit(value or default)
        if unit in SED_UNIT_OPTIONS:
            return unit

        default_unit = normalise_sed_unit(default)
        if default_unit in SED_UNIT_OPTIONS:
            return default_unit

        return "ABmag"

    def _current_filter_unit(self, filter_name: Any) -> str:
        """Resolve the unit that should be shown for one filter.

        Priority:
        1. pending per-filter unit
        2. saved per-filter unit
        3. last assigned "unit for all filters"
        4. ABmag
        """
        key = str(filter_name or "")

        if key:
            pending = self.pending_unit_overrides.get(key)
            if pending:
                return self._sed_unit_value(pending)

            saved = self.unit_overrides.get(key)
            if saved:
                return self._sed_unit_value(saved)

        if self.assign_all_unit_value:
            return self._sed_unit_value(self.assign_all_unit_value)

        return "ABmag"

    def _inferred_assign_all_unit(self, filter_names: Sequence[str]) -> str:
        """Return the value that should be displayed by "Unit for all filters"."""
        if self.assign_all_unit_value:
            return self._sed_unit_value(self.assign_all_unit_value)

        units = []
        for filter_name in filter_names:
            key = str(filter_name)
            value = self.pending_unit_overrides.get(key) or self.unit_overrides.get(key)
            if value:
                units.append(self._sed_unit_value(value))

        if units and len(set(units)) == 1:
            return units[0]

        return "ABmag"

    def _publish(self, topic: str, payload: Optional[Dict[str, Any]] = None) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        try:
            events.publish(topic, payload or {})
        except Exception:
            pass

    def _event_identity(
        self,
        *,
        dataset_id: Optional[Any] = None,
        row_id: Optional[Any] = None,
        artifact_id: Optional[str] = None,
        sed_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "plugin_id": PLUGIN_ID,
            "origin": self.instance_id,
            "panel_id": self.instance_id,
        }

        if dataset_id is not None:
            payload["dataset_id"] = str(dataset_id)

        if row_id is not None:
            row_id_str = str(row_id)
            payload["row_id"] = row_id_str
            payload["row_ids"] = [row_id_str]

        if artifact_id is not None:
            payload["artifact_id"] = str(artifact_id)

        if sed_file is not None:
            payload["sed_file"] = str(sed_file)

        return payload

    def _publish_sed_running(
        self,
        running: bool,
        *,
        dataset_id: Optional[Any] = None,
        row_id: Optional[Any] = None,
        sed_file: Optional[str] = None,
        artifact_id: Optional[str] = None,
        reason: Optional[str] = None,
        error: Optional[Any] = None,
    ) -> None:
        payload = self._event_identity(
            dataset_id=dataset_id,
            row_id=row_id,
            artifact_id=artifact_id,
            sed_file=sed_file,
        )
        payload["running"] = bool(running)

        if reason:
            payload["reason"] = str(reason)

        if error is not None:
            payload["error"] = str(error)

        self._publish("astro.sed.running", payload)

    def _publish_artifact_created(
        self,
        *,
        artifact_id: Optional[str],
        dataset_id: Any,
        row_id: Any,
    ) -> None:
        if artifact_id is None:
            return

        payload = self._event_identity(
            dataset_id=dataset_id,
            row_id=row_id,
            artifact_id=artifact_id,
            sed_file=self.sed_file,
        )
        payload["type"] = SED_ARTIFACT_TYPE
        self._publish("artifact.created", payload)

    def _publish_plugin_error(
        self,
        *,
        stage: str,
        error: Any,
        dataset_id: Optional[Any] = None,
        row_id: Optional[Any] = None,
        sed_file: Optional[str] = None,
    ) -> None:
        payload = self._event_identity(
            dataset_id=dataset_id,
            row_id=row_id,
            sed_file=sed_file,
        )
        payload.update(
            {
                "stage": str(stage),
                "error": str(error),
                "error_type": type(error).__name__,
            }
        )
        self._publish("plugin.error", payload)

    def _on_toggle_settings(self, event) -> None:
        self.settings_expanded = not self.settings_expanded
        self.plot_settings_panel.visible = self.settings_expanded

    def _on_plot_settings_changed(self, event) -> None:
        self.show_fwhm_error_bars = bool(self.show_fwhm_checkbox.value)
        self.plot_unit = _normalise_plot_unit(self.plot_unit_select.value)
        self.show_flux_table = bool(self.show_flux_table_checkbox.value)
        self.table_pane.visible = self.show_flux_table
        if self.plot_unit_select.value != self.plot_unit:
            self.plot_unit_select.value = self.plot_unit
        if self.latest_payload is not None:
            payload = dict(self.latest_payload)
            artifact_id = self.latest_payload_artifact_id
            if not self._render_payload(
                payload,
                artifact_id=artifact_id,
                collapse_mapping=False,
            ):
                self._schedule_ui_callback(
                    lambda: self._render_payload(
                        payload,
                        artifact_id=artifact_id,
                        collapse_mapping=False,
                    )
                )

    def _refresh_file_options(self) -> None:
        files = self.runtime.list_band_files()
        options = [""] + files
        old_value = self.sed_file or self.file_select.value or ""

        if old_value and old_value not in options:
            options.append(old_value)

        self.file_select.options = options
        self.file_select.value = old_value if old_value in options else ""

    def _on_file_selected(self, event) -> None:
        value = event.new or ""
        self.sed_file = value or None
        self.latest_payload = None
        self.latest_payload_artifact_id = None
        # File context changed, so mapping requirements may change.
        self.mapping_ui_signature = None
        self.refresh(refresh_file_options=False)

    def _on_create_file(self, event) -> None:
        dataset_id = self._active_dataset_id()
        if not dataset_id:
            self._set_status("Load a dataset before creating a SED data file.", "warning")
            return

        columns = self._dataset_columns(dataset_id)
        if not columns:
            self._set_status("The active dataset has no visible columns.", "warning")
            return

        try:
            created = self.runtime.create_photometry_band_file(columns)
        except Exception as exc:
            self._set_status(f"Could not create SED data file: {html.escape(str(exc))}", "danger")
            return

        self.sed_file = created
        self._refresh_file_options()
        self.file_select.value = created
        self._set_status(f"Created SED data file: `{created}`", "success")
        self.mapping_ui_signature = None
        self.refresh(refresh_file_options=False)

    def _enabled_filter_names(
        self,
        bands: Mapping[str, Mapping[str, Any]],
    ) -> List[str]:
        names: List[str] = []

        for filter_name, spec in bands.items():
            try:
                if self.runtime._is_enabled_band(spec):
                    names.append(str(filter_name))
            except Exception:
                continue

        return names

    def _mapping_rows_for_display(
        self,
        dataset_id: str,
        filter_names: Sequence[str],
    ) -> pd.DataFrame:
        columns = set(self._dataset_columns(dataset_id))

        rows = []

        for filter_name in filter_names:
            filter_name = str(filter_name)

            value_col = (
                self.pending_column_overrides.get(filter_name)
                or self.column_overrides.get(filter_name)
                or ""
            )

            error_col = (
                self.pending_error_column_overrides.get(filter_name)
                or self.error_column_overrides.get(filter_name)
                or ""
            )

            unit = self._current_filter_unit(filter_name)

            if value_col:
                value_display = value_col
                status = "mapped"
            elif filter_name in columns:
                value_display = filter_name
                status = "direct column"
            else:
                value_display = ""
                status = "unmapped"

            rows.append(
                {
                    "Filter": filter_name,
                    "Flux/mag column": value_display,
                    "Error column": error_col,
                    "Unit": unit,
                    "Status": status,
                }
            )

        return pd.DataFrame(rows)

    def _on_mapping_reference_changed(self, event) -> None:
        filter_name = str(event.new or "")

        columns = self.mapping_column_select.options if self.mapping_column_select is not None else []

        error_columns = (
            self.mapping_error_column_select.options
            if self.mapping_error_column_select is not None
            else []
        )

        value_col = (
            self.pending_column_overrides.get(filter_name)
            or self.column_overrides.get(filter_name)
            or (filter_name if filter_name in columns else "")
        )

        error_col = (
            self.pending_error_column_overrides.get(filter_name)
            or self.error_column_overrides.get(filter_name)
            or ""
        )

        unit = self._current_filter_unit(filter_name)

        if self.mapping_column_select is not None:
            self.mapping_column_select.value = value_col if value_col in columns else ""

        if self.mapping_error_column_select is not None:
            self.mapping_error_column_select.value = error_col if error_col in error_columns else ""

        if self.mapping_unit_select is not None:
            self.mapping_unit_select.value = self._sed_unit_value(unit)

    def _on_set_single_mapping(self, event) -> None:
        if (
            self.mapping_reference_select is None
            or self.mapping_column_select is None
            or self.mapping_error_column_select is None
            or self.mapping_unit_select is None
        ):
            return

        filter_name = str(self.mapping_reference_select.value or "").strip()
        value_col = str(self.mapping_column_select.value or "").strip()
        error_col = str(self.mapping_error_column_select.value or "").strip()
        unit = self._sed_unit_value(self.mapping_unit_select.value or "ABmag")

        if not filter_name:
            return

        if value_col:
            self.pending_column_overrides[filter_name] = value_col
        else:
            self.pending_column_overrides.pop(filter_name, None)

        if error_col:
            self.pending_error_column_overrides[filter_name] = error_col
        else:
            self.pending_error_column_overrides.pop(filter_name, None)

        self.pending_unit_overrides[filter_name] = unit

        dataset_id = self.current_dataset_id or self._active_dataset_id()
        if dataset_id and self.mapping_table is not None:
            filters = list(self.mapping_reference_select.options)
            self.mapping_table.value = self._mapping_rows_for_display(dataset_id, filters)

    def _on_assign_unit_to_all(self, event) -> None:
        if self.assign_all_unit_select is None or self.mapping_reference_select is None:
            return

        unit = self._sed_unit_value(self.assign_all_unit_select.value or "ABmag")
        self.assign_all_unit_value = unit

        for filter_name in self.mapping_reference_select.options:
            self.pending_unit_overrides[str(filter_name)] = unit

        if self.mapping_unit_select is not None:
            self.mapping_unit_select.value = unit

        dataset_id = self.current_dataset_id or self._active_dataset_id()
        if dataset_id and self.mapping_table is not None:
            filters = list(self.mapping_reference_select.options)
            self.mapping_table.value = self._mapping_rows_for_display(dataset_id, filters)

    def _on_apply_column_mappings(self, event) -> None:
        self.column_overrides = {
            str(key): str(value)
            for key, value in self.pending_column_overrides.items()
            if value
        }

        self.error_column_overrides = {
            str(key): str(value)
            for key, value in self.pending_error_column_overrides.items()
            if value
        }

        self.unit_overrides = {
            str(key): self._sed_unit_value(value)
            for key, value in self.pending_unit_overrides.items()
            if value
        }

        self.pending_column_overrides = dict(self.column_overrides)
        self.pending_error_column_overrides = dict(self.error_column_overrides)
        self.pending_unit_overrides = dict(self.unit_overrides)

        if self.assign_all_unit_select is not None:
            self.assign_all_unit_value = self._sed_unit_value(
                self.assign_all_unit_select.value or self.assign_all_unit_value or "ABmag"
            )

        self.mapping_applied_once = True
        self.mapping_ui_signature = None

        self.refresh(refresh_file_options=False)

    def _on_clear_mappings(self, event) -> None:
        self.column_overrides.clear()
        self.pending_column_overrides.clear()

        self.error_column_overrides.clear()
        self.pending_error_column_overrides.clear()

        self.unit_overrides.clear()
        self.pending_unit_overrides.clear()
        self.assign_all_unit_value = None

        if self.assign_all_unit_select is not None:
            self.assign_all_unit_select.value = "ABmag"

        if self.mapping_unit_select is not None:
            self.mapping_unit_select.value = "ABmag"

        self.mapping_ui_signature = None

        self.refresh(refresh_file_options=False)

    def _load_bands(self) -> Dict[str, Dict[str, Any]]:
        if not self.sed_file:
            return {}

        return self.runtime.load_band_file(self.sed_file)

    def _unknown_tokens(self, dataset_id: str, bands: Mapping[str, Mapping[str, Any]]) -> List[str]:
        columns = set(self._dataset_columns(dataset_id))
        unknown: List[str] = []

        for token in self.runtime.referenced_column_tokens(dict(bands)):
            mapped = self.column_overrides.get(token)
            if token in columns:
                continue
            if mapped and mapped in columns:
                continue
            unknown.append(str(token))

        return unknown

    def _build_mapping_controls(
        self,
        *,
        dataset_id: str,
        filter_names: Sequence[str],
        unknown_tokens: Sequence[str],
    ) -> None:
        filters = [str(filter_name) for filter_name in filter_names]
        if not filters:
            self._clear_mapping_controls()
            return

        columns = [""] + self._dataset_columns(dataset_id)
        signature = (
            str(dataset_id),
            str(self.sed_file or ""),
            tuple(filters),
            tuple(columns),
            tuple(sorted(str(token) for token in unknown_tokens)),
        )

        if (
            self.mapping_ui_signature == signature
            and self.mapping_box.visible
            and self.mapping_content is not None
        ):
            self.mapping_ui_changed_this_refresh = False
            self._set_mapping_content_visible(self.mapping_controls_expanded)
            return

        self.mapping_ui_changed_this_refresh = True
        previous_signature = self.mapping_ui_signature
        self.mapping_ui_signature = signature

        # If the user changed dataset or SED file, open the controls because the
        # mapping context genuinely changed. If only the focused source changed,
        # this code path is not reached because the signature is unchanged.
        if previous_signature is None or previous_signature[:2] != signature[:2]:
            self.mapping_controls_expanded = True

        self.mapping_reference_select = pn.widgets.Select(
            name="Filter",
            options=filters,
            value=filters[0] if filters else None,
            sizing_mode="stretch_width",
            margin=(0, 0, 10, 0),
        )
        self.mapping_column_select = pn.widgets.Select(
            name="Flux / magnitude column",
            options=columns,
            value="",
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        )
        self.mapping_error_column_select = pn.widgets.Select(
            name="Error column",
            options=columns,
            value="",
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        )

        initial_filter = filters[0] if filters else ""
        initial_filter_unit = self._current_filter_unit(initial_filter)
        initial_assign_all_unit = self._inferred_assign_all_unit(filters)

        self.mapping_unit_select = pn.widgets.Select(
            name="Unit",
            options=SED_UNIT_OPTIONS,
            value=self._sed_unit_value(initial_filter_unit),
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        )

        self.assign_all_unit_select = pn.widgets.Select(
            name="Unit for all filters",
            options=SED_UNIT_OPTIONS,
            value=self._sed_unit_value(initial_assign_all_unit),
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        )

        # Populate the three dropdowns for the first filter.
        class _Event:
            def __init__(self, new):
                self.new = new

        self._on_mapping_reference_changed(_Event(self.mapping_reference_select.value))

        self.mapping_reference_select.param.watch(
            self._on_mapping_reference_changed,
            "value",
        )

        set_button = pn.widgets.Button(
            name="Set filter mapping",
            button_type="default",
            height=32,
            sizing_mode="stretch_width",
            margin=(0, 6, 0, 0),
        )
        set_button.on_click(self._on_set_single_mapping)

        apply_button = pn.widgets.Button(
            name="Apply mappings",
            button_type="primary",
            height=32,
            sizing_mode="stretch_width",
            margin=(0, 6, 0, 0),
        )
        apply_button.on_click(self._on_apply_column_mappings)

        assign_all_unit_button = pn.widgets.Button(
            name="Assign unit to all filters",
            button_type="default",
            height=32,
            sizing_mode="stretch_width",
            margin=(0, 6, 0, 0),
        )
        assign_all_unit_button.on_click(self._on_assign_unit_to_all)

        clear_pending_button = pn.widgets.Button(
            name="Clear pending",
            button_type="default",
            height=32,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        def _clear_pending(event) -> None:
            self.pending_column_overrides.clear()
            self.pending_error_column_overrides.clear()
            self.pending_unit_overrides.clear()

            if self.mapping_reference_select is not None:
                self._on_mapping_reference_changed(_Event(self.mapping_reference_select.value))

            if self.mapping_table is not None:
                self.mapping_table.value = self._mapping_rows_for_display(dataset_id, filters)

        clear_pending_button.on_click(_clear_pending)

        self.mapping_table = pn.widgets.Tabulator(
            self._mapping_rows_for_display(dataset_id, filters),
            show_index=False,
            disabled=True,
            pagination="local",
            page_size=6,
            height=170,
            sizing_mode="stretch_width",
            widths={
                "Filter": 140,
                "Flux/mag column": 220,
                "Error column": 220,
                "Unit": 75,
                "Status": 95,
            },
            margin=(10, 0, 0, 0),
        )

        mapped_count = 0
        dataset_columns = set(columns)
        for filter_name in filters:
            if (
                filter_name in dataset_columns
                or self.pending_column_overrides.get(filter_name)
                or self.column_overrides.get(filter_name)
            ):
                mapped_count += 1

        unknown_count = len(unknown_tokens)
        if unknown_count:
            attention_text = f"{unknown_count} filter value column(s) need attention."
        else:
            attention_text = "All filter value columns are directly available or mapped."

        summary = pn.pane.HTML(
            f"""
            <b>SED column mappings</b><br>
            {html.escape(attention_text)}<br>
            {mapped_count} / {len(filters)} filters currently have a direct, saved, or pending value-column mapping.<br>
            Choose one value column, optional error column, and unit per filter. Error columns are assumed to have the same unit.
            """,
            sizing_mode="stretch_width",
        )

        self.mapping_toggle_button = pn.widgets.Button(
            name="Hide mapping controls",
            button_type="light",
            height=30,
            width=170,
            margin=(0, 0, 12, 0),
        )

        def _toggle_mapping_controls(event) -> None:
            current = (
                bool(self.mapping_content.visible)
                if self.mapping_content is not None
                else False
            )
            self._set_mapping_content_visible(not current)

        self.mapping_toggle_button.on_click(_toggle_mapping_controls)

        self.mapping_content = pn.Column(
            pn.Row(
                self.mapping_reference_select,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            ),
            pn.Row(
                self.mapping_column_select,
                self.mapping_error_column_select,
                self.mapping_unit_select,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            ),
            pn.Row(
                set_button,
                clear_pending_button,
                apply_button,
                sizing_mode="stretch_width",
                margin=(4, 0, 10, 0),
            ),
            pn.Row(
                self.assign_all_unit_select,
                assign_all_unit_button,
                sizing_mode="stretch_width",
                margin=(6, 0, 10, 0),
            ),
            self.mapping_table,
            sizing_mode="stretch_width",
            visible=True,
            margin=(0, 0, 0, 0),
            styles={"overflow": "visible"},
        )

        self.mapping_box.objects = [
            summary,
            self.mapping_toggle_button,
            self.mapping_content,
        ]
        self.mapping_box.visible = True
        self._set_mapping_content_visible(self.mapping_controls_expanded)

    # ------------------------------------------------------------------
    # Event handling / rendering
    # ------------------------------------------------------------------

    def _subscribe(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        for topic, callback in (
            ("selection.focus.changed", self._on_focus_changed),
            ("selection.focus.cleared", self._on_focus_cleared),
            ("dataset.active.changed", self._on_dataset_changed),
            ("dataset.mapping.updated", self._on_dataset_changed),
        ):
            try:
                sub = events.subscribe(
                    topic,
                    callback,
                    owner_id=self.instance_id,
                    owner_label="Broadband SED",
                    owner_kind="panel",
                )
            except TypeError:
                sub = events.subscribe(topic, callback)
            self.subscriptions.append(sub)

    def _on_focus_changed(self, topic, payload) -> None:
        dataset_id = payload.get("dataset_id")
        row_id = payload.get("row_id")

        if dataset_id is None or row_id is None:
            return

        self.current_dataset_id = str(dataset_id)
        self.current_row_id = str(row_id)
        if not self._ui_loaded:
            self._pending_refresh = True
            return
        self.refresh()

    def _on_focus_cleared(self, topic, payload) -> None:
        self._cancel_active_build(reason="focus_cleared")
        self.current_dataset_id = None
        self.current_row_id = None
        self.latest_payload = None
        self.latest_payload_artifact_id = None
        self._clear_mapping_controls()
        _assign_sed_plot(
            self.plot_pane,
            create_sed_plot(
                [],
                show_fwhm_error_bars=self.show_fwhm_error_bars,
                show_magnitude_axis=self.plot_unit == PLOT_UNIT_MICROJY,
                plot_unit=self.plot_unit,
            ),
            visible=False,
        )
        self.plot_placeholder.visible = True
        self.plot_placeholder.object = (
            "Focus a row to build its SED. The plot will appear once at least "
            f"{MIN_POINTS_TO_PLOT} finite broadband points are available."
        )
        self.table_pane.clear()
        self.table_pane.visible = self.show_flux_table
        self.skipped_pane.clear()
        self._set_status("Focus a row to plot its SED.", "info")

    def _on_dataset_changed(self, topic, payload) -> None:
        self._refresh_file_options()
        focus = self._get_focus()
        if (
            focus is not None
            and getattr(focus, "dataset_id", None)
            and getattr(focus, "row_id", None)
        ):
            self.current_dataset_id = str(focus.dataset_id)
            self.current_row_id = str(focus.row_id)
            if not self._ui_loaded:
                self._pending_refresh = True
                return
            self.refresh()

    def refresh(self, *, refresh_file_options: bool = False, source_change: bool = False) -> None:
        if not self._ui_loaded:
            self._pending_refresh = True
            return
        if refresh_file_options:
            self._refresh_file_options()

        dataset_id = self.current_dataset_id or self._active_dataset_id()
        row_id = self.current_row_id

        if not dataset_id:
            self._cancel_active_build(reason="no_dataset")
            self._set_status("Load a dataset before using the Broadband SED panel.", "warning")
            return

        if not self.sed_file:
            self._cancel_active_build(reason="no_sed_file")
            self._clear_mapping_controls()
            self.plot_pane.visible = False
            self.plot_placeholder.visible = True
            self.plot_placeholder.object = (
                "Select a SED photometry-band JSON file. The plot will appear once at "
                f"least {MIN_POINTS_TO_PLOT} finite broadband points are available."
            )
            self.table_pane.clear()
            self.table_pane.visible = self.show_flux_table
            self.skipped_pane.clear()
            self._set_status(
                "Select a SED photometry-band JSON file or create a new one.",
                "info",
            )
            return

        if not Path(self.sed_file).is_file():
            self._cancel_active_build(reason="missing_sed_file")
            self._clear_mapping_controls()
            self.plot_pane.visible = False
            self.plot_placeholder.visible = True
            self.plot_placeholder.object = "SED plot is hidden because the selected band file is missing."
            self._set_status(
                f"SED photometry-band file does not exist: `{self.sed_file}`",
                "danger",
            )
            return

        try:
            bands = self._load_bands()
        except Exception as exc:
            self._cancel_active_build(reason="sed_file_load_error")
            self._clear_mapping_controls()
            self.plot_pane.visible = False
            self.plot_placeholder.visible = True
            self.plot_placeholder.object = "SED plot is hidden because the selected band file could not be loaded."
            self._set_status(f"Could not load SED band file: {html.escape(str(exc))}", "danger")
            return

        filter_names = self._enabled_filter_names(bands)
        unknown_tokens = self._unknown_tokens(dataset_id, bands)
        self._build_mapping_controls(
            dataset_id=dataset_id,
            filter_names=filter_names,
            unknown_tokens=unknown_tokens,
        )

        if unknown_tokens:
            if self.mapping_ui_changed_this_refresh:
                self._set_status(
                    (
                        "Some SED filter value columns are unmapped. The panel will still plot "
                        f"using any available or mapped filters once at least {MIN_POINTS_TO_PLOT} "
                        "finite points are available."
                    ),
                    "warning",
                )
        else:
            if self.mapping_ui_changed_this_refresh:
                self._set_status("SED mappings are resolved. Building plot...", "info")

        if not row_id:
            self._cancel_active_build(reason="no_focused_row")
            self.plot_pane.visible = False
            self.plot_placeholder.visible = True
            self.plot_placeholder.object = (
                "Focus a row to build its SED. The plot will appear once at least "
                f"{MIN_POINTS_TO_PLOT} finite broadband points are available."
            )
            self.table_pane.clear()
            self.table_pane.visible = self.show_flux_table
            self.skipped_pane.clear()
            self._set_status("Focus a row to plot its SED.", "info")
            return

        self._submit_build_job(dataset_id=dataset_id, row_id=str(row_id), sed_file=self.sed_file)

    def _forget_job_handle(self, handle: Any) -> None:
        if handle is None:
            return
        try:
            self.job_handles.remove(handle)
        except ValueError:
            pass

    def _cancel_active_build(self, *, reason: str) -> None:
        self._build_generation += 1
        handle = self._active_job_handle
        self._active_job_handle = None
        latest_context = dict(self._latest_build_context)
        self._latest_build_context = {}

        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass
            self._forget_job_handle(handle)

        if latest_context:
            self._publish_sed_running(
                False,
                dataset_id=latest_context.get("dataset_id"),
                row_id=latest_context.get("row_id"),
                sed_file=latest_context.get("sed_file"),
                reason=reason,
            )

    def _submit_build_job(self, *, dataset_id: str, row_id: str, sed_file: str) -> None:
        self._cancel_active_build(reason="superseded")
        generation = self._build_generation
        column_overrides = dict(self.column_overrides)
        error_column_overrides = dict(self.error_column_overrides)
        unit_overrides = dict(self.unit_overrides)

        if self.mapping_ui_changed_this_refresh or self.last_valid_point_count < MIN_POINTS_TO_PLOT:
            self._set_status(f"Building SED for row `{html.escape(str(row_id))}`...", "info")

        self._latest_build_context = {
            "dataset_id": str(dataset_id),
            "row_id": str(row_id),
            "sed_file": str(sed_file),
        }

        self._publish_sed_running(
            True,
            dataset_id=dataset_id,
            row_id=row_id,
            sed_file=sed_file,
            reason="build_started",
        )

        handle_box: Dict[str, Any] = {}

        def _worker(cancel_token=None) -> Dict[str, Any]:
            return self._build_sed_payload(
                cancel_token=cancel_token,
                dataset_id=dataset_id,
                row_id=row_id,
                sed_file=sed_file,
                column_overrides=column_overrides,
                error_column_overrides=error_column_overrides,
                unit_overrides=unit_overrides,
            )

        def _done(payload: Dict[str, Any]) -> None:
            self._forget_job_handle(handle_box.get("handle"))
            self._on_build_done(
                payload,
                generation=generation,
                dataset_id=dataset_id,
                row_id=row_id,
                sed_file=sed_file,
            )

        def _error(exc: BaseException) -> None:
            self._forget_job_handle(handle_box.get("handle"))
            self._on_build_error(
                exc,
                generation=generation,
                dataset_id=dataset_id,
                row_id=row_id,
                sed_file=sed_file,
            )

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                _done(_worker(cancel_token=None))
            except Exception as exc:
                _error(exc)
            return

        handle = jobs.submit(
            _worker,
            title="Build broadband SED",
            key=(
                f"broadband-sed:{self.instance_id}:{dataset_id}:"
                f"{row_id}:{generation}"
            ),
            on_done=_done,
            on_error=_error,
        )
        handle_box["handle"] = handle
        self._active_job_handle = handle
        self.job_handles.append(handle)

    def _build_sed_payload(
        self,
        *,
        cancel_token,
        dataset_id: str,
        row_id: str,
        sed_file: str,
        column_overrides: Mapping[str, str],
        error_column_overrides: Mapping[str, str],
        unit_overrides: Mapping[str, str],
    ) -> Dict[str, Any]:
        if cancel_token is not None and cancel_token.cancelled():
            return {}

        bands = self.runtime.load_band_file(sed_file)

        if cancel_token is not None and cancel_token.cancelled():
            return {}

        row = self._get_row(dataset_id, row_id)

        if cancel_token is not None and cancel_token.cancelled():
            return {}

        result = self.runtime.build_sed_table(
            row=row,
            bands=bands,
            column_overrides=column_overrides,
            error_column_overrides=error_column_overrides,
            unit_overrides=unit_overrides,
        )

        if cancel_token is not None and cancel_token.cancelled():
            return {}

        return self.runtime.build_artifact_payload(
            sed_df=result.dataframe,
            dataset_id=dataset_id,
            row_id=row_id,
            sed_file=sed_file,
            column_overrides=column_overrides,
            error_column_overrides=error_column_overrides,
            unit_overrides=unit_overrides,
            skipped=result.skipped,
        )

    def _on_build_done(
        self,
        payload: Dict[str, Any],
        *,
        generation: int,
        dataset_id: Optional[str] = None,
        row_id: Optional[str] = None,
        sed_file: Optional[str] = None,
    ) -> None:
        if self._disposed or generation != self._build_generation:
            return

        if not payload:
            self._active_job_handle = None
            self._latest_build_context = {}
            self._publish_sed_running(
                False,
                dataset_id=dataset_id,
                row_id=row_id,
                sed_file=sed_file,
                reason="cancelled",
            )
            self._set_status(
                "The SED build was cancelled before completion. Press Refresh to retry.",
                "warning",
            )
            return

        dataset_id = str(payload.get("dataset_id") or dataset_id or "")
        row_id = str(payload.get("row_id") or row_id or "")
        sed_file = str(payload.get("sed_file") or sed_file or self.sed_file or "")

        if (
            dataset_id != str(self.current_dataset_id or "")
            or row_id != str(self.current_row_id or "")
            or sed_file != str(self.sed_file or "")
        ):
            self._active_job_handle = None
            self._latest_build_context = {}
            self._publish_sed_running(
                False,
                dataset_id=dataset_id,
                row_id=row_id,
                sed_file=sed_file,
                reason="stale_result",
            )
            return

        artifact_id = None
        try:
            artifact_id = self.context.artifacts.put(
                SED_ARTIFACT_TYPE,
                payload,
                dataset_id=dataset_id,
                row_ids=[row_id],
                params={
                    "sed_file": payload.get("sed_file"),
                    "origin": self.instance_id,
                    "plugin_id": PLUGIN_ID,
                },
            )
            self.latest_artifact_id = artifact_id
        except Exception:
            artifact_id = None

        self._publish_artifact_created(
            artifact_id=artifact_id,
            dataset_id=dataset_id,
            row_id=row_id,
        )

        updated_payload = self._event_identity(
            dataset_id=dataset_id,
            row_id=row_id,
            artifact_id=artifact_id,
            sed_file=sed_file,
        )
        updated_payload["record_count"] = len(_payload_records(payload))
        self._publish("astro.sed.updated", updated_payload)

        def _render(attempt: int = 0) -> None:
            if self._disposed or generation != self._build_generation:
                return

            try:
                render_complete = self._render_payload(
                    payload,
                    artifact_id=artifact_id,
                )
            except Exception as exc:
                self._on_build_error(
                    exc,
                    generation=generation,
                    dataset_id=dataset_id,
                    row_id=row_id,
                    sed_file=sed_file,
                    stage="render_sed",
                    message="Could not render SED",
                )
                return

            if not render_complete and attempt == 0:
                self._schedule_ui_callback(lambda: _render(1))
                return

            self._active_job_handle = None
            self._latest_build_context = {}
            self._publish_sed_running(
                False,
                dataset_id=dataset_id,
                row_id=row_id,
                sed_file=sed_file,
                artifact_id=artifact_id,
                reason="completed",
            )

        self._schedule_ui_callback(_render)

    def _on_build_error(
        self,
        exc: BaseException,
        *,
        generation: int,
        dataset_id: Optional[str] = None,
        row_id: Optional[str] = None,
        sed_file: Optional[str] = None,
        stage: str = "build_sed",
        message: str = "Could not build SED",
    ) -> None:
        if self._disposed or generation != self._build_generation:
            return

        self._active_job_handle = None
        self._latest_build_context = {}
        self._publish_sed_running(
            False,
            dataset_id=dataset_id,
            row_id=row_id,
            sed_file=sed_file,
            reason="error",
            error=exc,
        )
        self._publish_plugin_error(
            stage=stage,
            error=exc,
            dataset_id=dataset_id,
            row_id=row_id,
            sed_file=sed_file,
        )

        self._set_status(f"{message}: {html.escape(str(exc))}", "danger")
        self.skipped_pane.clear()
        self.skipped_pane.append(
            pn.pane.HTML(
                f"""
                <pre style="white-space: pre-wrap; font-size: 11px;">
                {html.escape(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))}
                </pre>
                """,
                sizing_mode="stretch_width",
            )
        )

    def _render_payload(
        self,
        payload: Mapping[str, Any],
        *,
        artifact_id: Optional[str] = None,
        collapse_mapping: bool = True,
    ) -> bool:
        self.latest_payload = dict(payload)
        self.latest_payload_artifact_id = artifact_id
        records = _payload_records(payload)
        finite_records = []

        for record in records:
            try:
                wavelength = float(record.get("wavelength (µm)"))
                flux_uJy = float(record.get("flux_uJy"))
            except Exception:
                continue

            if pd.notna(wavelength) and pd.notna(flux_uJy) and wavelength > 0 and flux_uJy > 0:
                finite_records.append(record)

        self.last_valid_point_count = len(finite_records)
        self.table_pane.clear()
        self.table_pane.visible = self.show_flux_table
        self.skipped_pane.clear()

        if len(finite_records) >= MIN_POINTS_TO_PLOT:
            plot = create_sed_plot(
                finite_records,
                show_error_bars=SHOW_SED_ERROR_BARS_BY_DEFAULT,
                show_fwhm_error_bars=self.show_fwhm_error_bars,
                show_magnitude_axis=self.plot_unit == PLOT_UNIT_MICROJY,
                plot_unit=self.plot_unit,
            )
            render_complete = _assign_sed_plot(
                self.plot_pane,
                plot,
                visible=True,
            )
            self.plot_placeholder.visible = False

            plotted_unit_label = "νFν" if self.plot_unit == PLOT_UNIT_NUFNU else "µJy"
            msg = f"Plotted {len(finite_records)} broadband SED points in {plotted_unit_label}"
            if artifact_id:
                msg += f". Artifact: `{artifact_id}`"
            self._set_status(msg, "success")

            # Once the plot is useful, collapse the mapping controls so the
            # panel is not vertically crowded. The user can reopen them.
            if collapse_mapping and self.mapping_box.visible:
                self._set_mapping_content_visible(False)
        else:
            plot = create_sed_plot(
                [],
                show_fwhm_error_bars=self.show_fwhm_error_bars,
                show_magnitude_axis=self.plot_unit == PLOT_UNIT_MICROJY,
                plot_unit=self.plot_unit,
            )
            render_complete = _assign_sed_plot(
                self.plot_pane,
                plot,
                visible=False,
            )
            self.plot_placeholder.visible = True
            self.plot_placeholder.object = (
                f"Only {len(finite_records)} finite SED point"
                f"{'' if len(finite_records) == 1 else 's'} could be built. "
                f"The plot will appear after {MIN_POINTS_TO_PLOT} valid points are available."
            )
            self._set_status(
                (
                    f"{len(finite_records)} valid SED point"
                    f"{'' if len(finite_records) == 1 else 's'} built; "
                    f"{MIN_POINTS_TO_PLOT} required for plotting."
                ),
                "info",
            )

        if self.show_flux_table and records and len(finite_records) >= MIN_POINTS_TO_PLOT:
            sed_df = pd.DataFrame(records)
            self.table_pane.append(
                pn.widgets.Tabulator(
                    sed_df,
                    disabled=True,
                    show_index=False,
                    pagination="local",
                    page_size=5,
                    sizing_mode="stretch_width",
                    height=180,
                    margin=(14, 0, 12, 0),
                )
            )

        skipped = payload.get("skipped") or []
        show_skipped_diagnostics = (
            bool(skipped)
            and self.mapping_applied_once
            and len(finite_records) >= MIN_POINTS_TO_PLOT
        )

        if show_skipped_diagnostics:
            skipped_df = pd.DataFrame([dict(item) for item in skipped])
            skipped_summary = pn.pane.HTML(
                f"""
                Skipped references: {len(skipped)}. These were ignored because they were unmapped, disabled, non-finite, or unavailable for the focused row.
                """,
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            )
            skipped_table = pn.widgets.Tabulator(
                skipped_df,
                disabled=True,
                show_index=False,
                pagination="local",
                page_size=4,
                sizing_mode="stretch_width",
                height=130,
                margin=(0, 0, 0, 0),
            )
            skipped_content = pn.Column(
                skipped_table,
                sizing_mode="stretch_width",
                visible=False,
                margin=(8, 0, 0, 0),
            )
            skipped_toggle = pn.widgets.Button(
                name="Show skipped references",
                button_type="light",
                height=30,
                width=180,
                margin=(0, 0, 0, 0),
            )

            def _toggle_skipped(event) -> None:
                skipped_content.visible = not skipped_content.visible
                skipped_toggle.name = (
                    "Hide skipped references"
                    if skipped_content.visible
                    else "Show skipped references"
                )

            skipped_toggle.on_click(_toggle_skipped)
            self.skipped_pane.append(
                pn.Column(
                    skipped_summary,
                    skipped_toggle,
                    skipped_content,
                    sizing_mode="stretch_width",
                    margin=(16, 0, 12, 0),
                    styles={
                        "border-top": "1px solid #eeeeee",
                        "padding-top": "12px",
                    },
                )
            )

        return render_complete

    # ------------------------------------------------------------------
    # Persistence / cleanup
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "state_version": self.state_version,
            "sed_file": self.sed_file,
            "column_overrides": dict(self.column_overrides),
            "error_column_overrides": dict(self.error_column_overrides),
            "unit_overrides": dict(self.unit_overrides),
            "assign_all_unit_value": self.assign_all_unit_value,
            "latest_artifact_id": self.latest_artifact_id,
            "mapping_applied_once": self.mapping_applied_once,
            "mapping_controls_expanded": self.mapping_controls_expanded,
            "show_fwhm_error_bars": self.show_fwhm_error_bars,
            "plot_unit": self.plot_unit,
            "show_flux_table": self.show_flux_table,
            "settings_expanded": self.settings_expanded,
        }

    def dispose(self) -> None:
        if self._disposed:
            return

        self._cancel_active_build(reason="panel.dispose")
        self._disposed = True

        for handle in list(self.job_handles):
            try:
                handle.cancel()
            except Exception:
                pass
        self.job_handles.clear()

        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in self.subscriptions:
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass
        self.subscriptions.clear()

def create_broadband_sed_panel(context, **kwargs):
    controller = BroadbandSEDPanel(context, **kwargs)
    return controller.view, controller

class BroadbandSEDArtifactViewer:
    state_version = 1

    def __init__(self, context, *, artifact_id: Optional[str] = None, **kwargs) -> None:
        self.context = context
        self.artifact_id = artifact_id
        self.status = pn.pane.Alert("", alert_type="info", sizing_mode="stretch_width")
        self.plot_pane = pn.pane.Bokeh(
            create_sed_plot([]),
            sizing_mode="stretch_both",
        )
        self.table_pane = pn.Column(sizing_mode="stretch_width")
        self.view = pn.Column(self.status, self.plot_pane, self.table_pane, sizing_mode="stretch_both")
        onload = getattr(pn.state, "onload", None)
        if callable(onload):
            onload(self.render)
        else:
            self.render()

    def render(self) -> None:
        if not self.artifact_id:
            self.status.object = "No SED artifact id was provided."
            self.status.alert_type = "warning"
            return

        try:
            payload = self.context.artifacts.get(self.artifact_id)
        except Exception as exc:
            self.status.object = (
                f"Could not load SED artifact `{html.escape(str(self.artifact_id))}`: "
                f"{html.escape(str(exc))}"
            )
            self.status.alert_type = "danger"
            return

        records = _payload_records(payload)
        _assign_sed_plot(
            self.plot_pane,
            create_sed_plot(records),
            visible=True,
        )
        self.table_pane.clear()

        if records:
            self.table_pane.append(
                pn.widgets.Tabulator(
                    pd.DataFrame(records),
                    disabled=True,
                    pagination="local",
                    page_size=8,
                    sizing_mode="stretch_width",
                    height=220,
                )
            )

        dataset_id = _safe_str(payload.get("dataset_id") if isinstance(payload, Mapping) else "")
        row_id = _safe_str(payload.get("row_id") if isinstance(payload, Mapping) else "")
        self.status.object = (
            f"Broadband SED artifact `{html.escape(str(self.artifact_id))}`"
            f" for dataset `{html.escape(dataset_id)}`, row `{html.escape(row_id)}`."
        )
        self.status.alert_type = "success"

    def dispose(self) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {
            "state_version": self.state_version,
            "artifact_id": self.artifact_id,
        }

def create_broadband_sed_artifact_viewer(context, artifact_id=None, **kwargs):
    controller = BroadbandSEDArtifactViewer(context, artifact_id=artifact_id, **kwargs)
    return controller.view, controller