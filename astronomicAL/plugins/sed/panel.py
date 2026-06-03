from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import html
import traceback

import holoviews as hv
import pandas as pd
import panel as pn

from .service import SEDRuntime

PLUGIN_ID = "astro.sed"
RUNTIME_SERVICE_KEY = f"{PLUGIN_ID}.runtime"
SED_ARTIFACT_TYPE = "astro.sed.broadband"
MIN_POINTS_TO_PLOT = 3
SHOW_SED_ERROR_BARS_BY_DEFAULT = True


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _payload_records(payload: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    if not payload:
        return []
    records = payload.get("records", [])
    if isinstance(records, list):
        return [dict(item) for item in records if isinstance(item, Mapping)]
    return []


def create_sed_plot(
    records: Sequence[Mapping[str, Any]],
    *,
    show_error_bars: bool = SHOW_SED_ERROR_BARS_BY_DEFAULT,
) -> hv.Overlay | hv.Element:
    """Create the broadband SED plot.

    Error bars come from the legacy SED JSON fields:

    - FWHM  -> horizontal wavelength/filter-width bars
    - error -> vertical magnitude/flux uncertainty bars
    """

    data = pd.DataFrame([dict(item) for item in records or []])

    required = {"wavelength (µm)", "magnitude"}
    if data.empty or not required.issubset(set(data.columns)):
        return hv.Scatter(
            pd.DataFrame({"wavelength (µm)": [], "magnitude": []}),
            kdims=["wavelength (µm)"],
            vdims=["magnitude"],
        ).opts(
            responsive=True,
            active_tools=["pan", "wheel_zoom"],
            invert_yaxis=True,
            logx=True,
            xlabel="wavelength (µm)",
            ylabel="magnitude",
        )

    for column in ("wavelength (µm)", "magnitude", "FWHM", "error"):
        if column not in data.columns:
            data[column] = 0
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["wavelength (µm)", "magnitude"])
    data = data[data["wavelength (µm)"] > 0]
    data = data.sort_values("wavelength (µm)")

    if len(data) < MIN_POINTS_TO_PLOT:
        return hv.Scatter(
            pd.DataFrame({"wavelength (µm)": [], "magnitude": []}),
            kdims=["wavelength (µm)"],
            vdims=["magnitude"],
        ).opts(
            responsive=True,
            active_tools=["pan", "wheel_zoom"],
            invert_yaxis=True,
            logx=True,
            xlabel="wavelength (µm)",
            ylabel="magnitude",
        )

    line = hv.Curve(
        data,
        kdims=["wavelength (µm)"],
        vdims=["magnitude"],
    ).opts(
        responsive=True,
        active_tools=["pan", "wheel_zoom"],
        line_width=2,
    )

    points = hv.Scatter(
        data,
        kdims=["wavelength (µm)"],
        vdims=["magnitude", "band", "FWHM", "error"],
    ).opts(
        marker="circle",
        alpha=0.8,
        size=6,
        active_tools=["pan", "wheel_zoom"],
    )

    overlay = line * points

    if show_error_bars:
        y_error_rows = []
        for _, row in data.iterrows():
            err = float(row.get("error", 0) or 0)
            if err > 0:
                y_error_rows.append(
                    (
                        float(row["wavelength (µm)"]),
                        float(row["magnitude"]),
                        err,
                    )
                )

        if y_error_rows:
            overlay = overlay * hv.ErrorBars(y_error_rows)

        x_error_rows = []
        for _, row in data.iterrows():
            fwhm = float(row.get("FWHM", 0) or 0)
            if fwhm > 0:
                x_error_rows.append(
                    (
                        float(row["wavelength (µm)"]),
                        float(row["magnitude"]),
                        0.5 * fwhm,
                    )
                )

        if x_error_rows:
            overlay = overlay * hv.ErrorBars(x_error_rows, horizontal=True)

    return overlay.opts(
        invert_yaxis=True,
        logx=True,
        responsive=True,
        active_tools=["pan", "wheel_zoom"],
        xlabel="wavelength (µm)",
        ylabel="magnitude",
    )


class BroadbandSEDPanel:
    state_version = 1

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

        self.runtime: SEDRuntime = self._get_runtime()

        self.sed_file: Optional[str] = self.restore_state.get("sed_file")
        self.column_overrides: Dict[str, str] = dict(
            self.restore_state.get("column_overrides") or {}
        )

        self.pending_column_overrides: Dict[str, str] = dict(self.column_overrides)

        self.current_dataset_id: Optional[str] = None
        self.current_row_id: Optional[str] = None
        self.latest_artifact_id: Optional[str] = self.restore_state.get("latest_artifact_id")

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
            name="Clear local column mappings",
            button_type="default",
            width=190,
        )

        self.status = pn.pane.Alert(
            "Select a SED photometry-band file and focus a row.",
            alert_type="info",
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        self.mapping_table: Optional[pn.widgets.Tabulator] = None
        self.mapping_reference_select: Optional[pn.widgets.Select] = None
        self.mapping_column_select: Optional[pn.widgets.Select] = None
        self.mapping_content: Optional[pn.Column] = None
        self.mapping_toggle_button: Optional[pn.widgets.Button] = None
        self.last_valid_point_count: int = 0

        self.mapping_ui_signature: Optional[tuple] = None
        self.mapping_ui_changed_this_refresh: bool = False

        self.mapping_controls_expanded: bool = bool(
            self.restore_state.get("mapping_controls_expanded", True)
        )

        # Diagnostics should not appear before the user has actually tried to apply
        # mappings. Otherwise the panel looks broken before the user has done anything.
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

        self.plot_pane = pn.pane.HoloViews(
            create_sed_plot([]),
            sizing_mode="stretch_width",
            height=320,
            min_height=260,
            margin=(12, 0, 14, 0),
            visible=False,
        )

        self.table_pane = pn.Column(
            sizing_mode="stretch_width",
            margin=(6, 0, 0, 0),
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
                sizing_mode="stretch_width",
                margin=(8, 0, 10, 0),
            ),
            self.status,
            self.mapping_box,
            self.plot_placeholder,
            self.plot_pane,
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

        self._subscribe()

        focus = self._get_focus()
        if (
            focus is not None
            and getattr(focus, "dataset_id", None)
            and getattr(focus, "row_id", None)
        ):
            self.current_dataset_id = str(focus.dataset_id)
            self.current_row_id = str(focus.row_id)
            self.refresh()
        else:
            self._set_status("Focus a row to plot its SED.", "info")

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

    def _mapping_rows_for_display(
        self,
        dataset_id: str,
        tokens: Sequence[str],
    ) -> pd.DataFrame:
        columns = set(self._dataset_columns(dataset_id))

        rows = []
        for token in tokens:
            token = str(token)
            mapped = (
                self.pending_column_overrides.get(token)
                or self.column_overrides.get(token)
                or ""
            )

            if token in columns:
                status = "direct column"
                mapped_display = token
            elif mapped:
                status = "mapped"
                mapped_display = mapped
            else:
                status = "unmapped"
                mapped_display = ""

            rows.append(
                {
                    "SED reference": token,
                    "Mapped dataset column": mapped_display,
                    "Status": status,
                }
            )

        return pd.DataFrame(rows)


    def _on_mapping_reference_changed(self, event) -> None:
        if self.mapping_column_select is None:
            return

        token = str(event.new or "")
        mapped = (
            self.pending_column_overrides.get(token)
            or self.column_overrides.get(token)
            or ""
        )

        if mapped in self.mapping_column_select.options:
            self.mapping_column_select.value = mapped
        else:
            self.mapping_column_select.value = ""


    def _on_set_single_mapping(self, event) -> None:
        if self.mapping_reference_select is None or self.mapping_column_select is None:
            return

        token = str(self.mapping_reference_select.value or "").strip()
        mapped = str(self.mapping_column_select.value or "").strip()

        if not token:
            return

        if mapped:
            self.pending_column_overrides[token] = mapped
        else:
            self.pending_column_overrides.pop(token, None)

        dataset_id = self.current_dataset_id or self._active_dataset_id()
        if dataset_id and self.mapping_table is not None:
            tokens = list(self.mapping_reference_select.options)
            self.mapping_table.value = self._mapping_rows_for_display(dataset_id, tokens)


    def _on_apply_column_mappings(self, event) -> None:
        self.column_overrides = {
            str(key): str(value)
            for key, value in self.pending_column_overrides.items()
            if value
        }
        self.pending_column_overrides = dict(self.column_overrides)

        self.mapping_applied_once = True

        self.mapping_ui_signature = None

        self.refresh(refresh_file_options=False)

    def _on_clear_mappings(self, event) -> None:
        self.column_overrides.clear()
        self.pending_column_overrides.clear()
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
            unknown.append(token)

        return unknown

    def _build_mapping_controls(
        self,
        dataset_id: str,
        unknown_tokens: Sequence[str],
    ) -> None:
        if not unknown_tokens:
            self._clear_mapping_controls()
            return

        columns = [""] + self._dataset_columns(dataset_id)
        tokens = [str(token) for token in unknown_tokens]
        dataset_columns = set(columns)

        signature = (
            str(dataset_id),
            str(self.sed_file or ""),
            tuple(tokens),
            tuple(columns),
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
            name="SED reference",
            options=tokens,
            value=tokens[0] if tokens else None,
            sizing_mode="stretch_width",
            margin=(0, 0, 10, 0),
        )

        self.mapping_column_select = pn.widgets.Select(
            name="Dataset column",
            options=columns,
            value="",
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        )

        first_token = self.mapping_reference_select.value
        first_mapped = (
            self.pending_column_overrides.get(first_token)
            or self.column_overrides.get(first_token)
            or ""
        )
        if first_mapped in columns:
            self.mapping_column_select.value = first_mapped

        self.mapping_reference_select.param.watch(
            self._on_mapping_reference_changed,
            "value",
        )

        set_button = pn.widgets.Button(
            name="Set mapping",
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

        clear_pending_button = pn.widgets.Button(
            name="Clear pending",
            button_type="default",
            height=32,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        def _clear_pending(event) -> None:
            self.pending_column_overrides.clear()
            if self.mapping_column_select is not None:
                self.mapping_column_select.value = ""
            if self.mapping_table is not None:
                self.mapping_table.value = self._mapping_rows_for_display(dataset_id, tokens)

        clear_pending_button.on_click(_clear_pending)

        self.mapping_table = pn.widgets.Tabulator(
            self._mapping_rows_for_display(dataset_id, tokens),
            show_index=False,
            disabled=True,
            pagination="local",
            page_size=4,
            height=128,
            sizing_mode="stretch_width",
            widths={
                "SED reference": 140,
                "Mapped dataset column": 220,
                "Status": 95,
            },
            margin=(10, 0, 0, 0),
        )

        mapped_count = sum(
            1
            for token in tokens
            if (
                token in dataset_columns
                or self.pending_column_overrides.get(token)
                or self.column_overrides.get(token)
            )
        )

        summary = pn.pane.HTML(
            f"""
            <div style="
                font-size: 12px;
                line-height: 1.45;
                color: #444;
                margin: 0 0 10px 0;
            ">
                <b>{len(tokens)} SED references need attention.</b><br>
                {mapped_count} currently have a direct, saved, or pending mapping.
                The plot appears after at least <b>{MIN_POINTS_TO_PLOT}</b>
                finite SED points can be built.
            </div>
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
            self.mapping_table,
            sizing_mode="stretch_width",
            visible=True,
            margin=(0, 0, 0, 0),
            styles={
                "overflow": "visible",
            },
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
            ("dataset.mapping_updated", self._on_dataset_changed),
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
        self.refresh()

    def _on_focus_cleared(self, topic, payload) -> None:
        self.current_dataset_id = None
        self.current_row_id = None
        self._clear_mapping_controls()
        self.plot_pane.object = create_sed_plot([])
        self.plot_pane.visible = False
        self.plot_placeholder.visible = True
        self.plot_placeholder.object = (
            "Focus a row to build its SED. The plot will appear once at least "
            f"{MIN_POINTS_TO_PLOT} finite broadband points are available."
        )
        self.table_pane.clear()
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
        self.refresh()

    def refresh(self, *, refresh_file_options: bool = False, source_change: bool = False) -> None:
        if refresh_file_options:
            self._refresh_file_options()

        dataset_id = self.current_dataset_id or self._active_dataset_id()
        row_id = self.current_row_id

        if not dataset_id:
            self._set_status("Load a dataset before using the Broadband SED panel.", "warning")
            return

        if not self.sed_file:
            self._clear_mapping_controls()
            self.plot_pane.visible = False
            self.plot_placeholder.visible = True
            self.plot_placeholder.object = (
                "Select a SED photometry-band JSON file. The plot will appear once at "
                f"least {MIN_POINTS_TO_PLOT} finite broadband points are available."
            )
            self.table_pane.clear()
            self.skipped_pane.clear()
            self._set_status(
                "Select a SED photometry-band JSON file or create a new one.",
                "info",
            )
            return

        if not Path(self.sed_file).is_file():
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
            self._clear_mapping_controls()
            self.plot_pane.visible = False
            self.plot_placeholder.visible = True
            self.plot_placeholder.object = "SED plot is hidden because the selected band file could not be loaded."
            self._set_status(f"Could not load SED band file: {html.escape(str(exc))}", "danger")
            return

        unknown_tokens = self._unknown_tokens(dataset_id, bands)
        self._build_mapping_controls(dataset_id, unknown_tokens)

        if unknown_tokens:
            if self.mapping_ui_changed_this_refresh:
                self._set_status(
                    (
                        "Some SED references are unmapped. The panel will still plot using "
                        f"any available or mapped bands once at least {MIN_POINTS_TO_PLOT} "
                        "finite points are available."
                    ),
                    "warning",
                )
        else:
            if self.mapping_ui_changed_this_refresh:
                self._set_status("SED mappings are resolved. Building plot...", "info")

        if not row_id:
            self.plot_pane.visible = False
            self.plot_placeholder.visible = True
            self.plot_placeholder.object = (
                "Focus a row to build its SED. The plot will appear once at least "
                f"{MIN_POINTS_TO_PLOT} finite broadband points are available."
            )
            self.table_pane.clear()
            self.skipped_pane.clear()
            self._set_status("Focus a row to plot its SED.", "info")
            return

        self._submit_build_job(dataset_id=dataset_id, row_id=str(row_id), sed_file=self.sed_file)

    def _submit_build_job(self, *, dataset_id: str, row_id: str, sed_file: str) -> None:
        if self.mapping_ui_changed_this_refresh or self.last_valid_point_count < MIN_POINTS_TO_PLOT:
            self._set_status(f"Building SED for row `{html.escape(str(row_id))}`...", "info")

        events = getattr(self.context, "events", None)
        if events is not None:
            try:
                events.publish(
                    "astro.sed.running",
                    {
                        "dataset_id": dataset_id,
                        "row_id": str(row_id),
                        "sed_file": sed_file,
                        "origin": self.instance_id,
                    },
                )
            except Exception:
                pass

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                payload = self._build_sed_payload(
                    cancel_token=None,
                    dataset_id=dataset_id,
                    row_id=row_id,
                    sed_file=sed_file,
                )
                self._on_build_done(payload)
            except Exception as exc:
                self._on_build_error(exc)
            return

        handle = jobs.submit(
            self._build_sed_payload,
            title="Build broadband SED",
            key=f"broadband-sed:{self.instance_id}:{dataset_id}:{row_id}:{sed_file}",
            on_done=self._on_build_done,
            on_error=self._on_build_error,
            dataset_id=dataset_id,
            row_id=row_id,
            sed_file=sed_file,
        )
        self.job_handles.append(handle)

    def _build_sed_payload(
        self,
        *,
        cancel_token,
        dataset_id: str,
        row_id: str,
        sed_file: str,
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
            column_overrides=self.column_overrides,
        )

        if cancel_token is not None and cancel_token.cancelled():
            return {}

        return self.runtime.build_artifact_payload(
            sed_df=result.dataframe,
            dataset_id=dataset_id,
            row_id=row_id,
            sed_file=sed_file,
            column_overrides=self.column_overrides,
            skipped=result.skipped,
        )

    def _on_build_done(self, payload: Dict[str, Any]) -> None:
        if not payload:
            return

        dataset_id = str(payload.get("dataset_id") or "")
        row_id = str(payload.get("row_id") or "")

        # Ignore late results for an old focus.
        if self.current_dataset_id and dataset_id != self.current_dataset_id:
            return
        if self.current_row_id and row_id != self.current_row_id:
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
                },
            )
            self.latest_artifact_id = artifact_id
        except Exception:
            artifact_id = None

        events = getattr(self.context, "events", None)
        if events is not None:
            if artifact_id is not None:
                try:
                    events.publish(
                        "artifact.created",
                        {
                            "artifact_id": artifact_id,
                            "type": SED_ARTIFACT_TYPE,
                            "dataset_id": dataset_id,
                            "row_id": row_id,
                            "origin": self.instance_id,
                        },
                    )
                except Exception:
                    pass
            try:
                events.publish(
                    "astro.sed.updated",
                    {
                        "artifact_id": artifact_id,
                        "dataset_id": dataset_id,
                        "row_id": row_id,
                        "record_count": len(_payload_records(payload)),
                        "origin": self.instance_id,
                    },
                )
            except Exception:
                pass

        self._render_payload(payload, artifact_id=artifact_id)

    def _on_build_error(self, exc: BaseException) -> None:
        self._set_status(f"Could not build SED: {html.escape(str(exc))}", "danger")
        self.skipped_pane.clear()
        self.skipped_pane.append(
            pn.pane.HTML(
                f"<pre>{html.escape(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))}</pre>",
                sizing_mode="stretch_width",
            )
        )

    def _render_payload(self, payload: Mapping[str, Any], *, artifact_id: Optional[str] = None) -> None:
        records = _payload_records(payload)

        finite_records = []
        for record in records:
            try:
                wavelength = float(record.get("wavelength (µm)"))
                magnitude = float(record.get("magnitude"))
            except Exception:
                continue

            if pd.notna(wavelength) and pd.notna(magnitude) and wavelength > 0:
                finite_records.append(record)

        self.last_valid_point_count = len(finite_records)

        self.table_pane.clear()
        self.skipped_pane.clear()

        if len(finite_records) >= MIN_POINTS_TO_PLOT:
            self.plot_pane.object = create_sed_plot(
                finite_records,
                show_error_bars=SHOW_SED_ERROR_BARS_BY_DEFAULT,
            )
            self.plot_pane.visible = True
            self.plot_placeholder.visible = False

            msg = f"Plotted {len(finite_records)} broadband SED points"
            if artifact_id:
                msg += f". Artifact: `{artifact_id}`"
            self._set_status(msg, "success")

            # Once the plot is useful, collapse the mapping controls so the panel
            # is not vertically crowded. The user can reopen them.
            if self.mapping_box.visible:
                self._set_mapping_content_visible(False)

        else:
            self.plot_pane.object = create_sed_plot([])
            self.plot_pane.visible = False
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

        if records and len(finite_records) >= MIN_POINTS_TO_PLOT:
            sed_df = pd.DataFrame(records)
            self.table_pane.append(
                pn.widgets.Tabulator(
                    sed_df,
                    disabled=True,
                    show_index=False,
                    pagination="local",
                    page_size=5,
                    sizing_mode="stretch_width",
                    height=160,
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
                <div style="
                    padding: 8px 10px;
                    border: 1px solid #d8d8d8;
                    border-radius: 5px;
                    background: #fbfbfb;
                    color: #444;
                    font-size: 12px;
                    line-height: 1.4;
                    box-sizing: border-box;
                ">
                    <b>Skipped references:</b> {len(skipped)}.
                    These were ignored because they were unmapped, disabled,
                    non-finite, or unavailable for the focused row.
                </div>
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

    # ------------------------------------------------------------------
    # Persistence / cleanup
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "state_version": self.state_version,
            "sed_file": self.sed_file,
            "column_overrides": dict(self.column_overrides),
            "latest_artifact_id": self.latest_artifact_id,
            "mapping_applied_once": self.mapping_applied_once,
            "mapping_controls_expanded": self.mapping_controls_expanded,
        }

    def dispose(self) -> None:
        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in self.subscriptions:
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass
        self.subscriptions.clear()

        for handle in self.job_handles:
            try:
                handle.cancel()
            except Exception:
                pass
        self.job_handles.clear()


def create_broadband_sed_panel(context, **kwargs):
    controller = BroadbandSEDPanel(context, **kwargs)
    return controller.view, controller


class BroadbandSEDArtifactViewer:
    state_version = 1

    def __init__(self, context, *, artifact_id: Optional[str] = None, **kwargs) -> None:
        self.context = context
        self.artifact_id = artifact_id
        self.status = pn.pane.Alert("", alert_type="info", sizing_mode="stretch_width")
        self.plot_pane = pn.pane.HoloViews(create_sed_plot([]), sizing_mode="stretch_both")
        self.table_pane = pn.Column(sizing_mode="stretch_width")
        self.view = pn.Column(self.status, self.plot_pane, self.table_pane, sizing_mode="stretch_both")
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
        self.plot_pane.object = create_sed_plot(records)

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