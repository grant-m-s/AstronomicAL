# BUG: If reloading the same file again it starts create another parquet.
from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource

from astronomicAL.platform.application_chrome_styles import (
    APPLICATION_CHROME_CSS,
    HEADER_BUTTON_STYLESHEET,
    HEADER_PRIMARY_BUTTON_STYLESHEET,
    HEADER_SELECT_STYLESHEET,
)
from astronomicAL.platform.fits_import import register_fits_table
from astronomicAL.platform.modal_utils import close_template_modal, open_template_modal
from astronomicAL.settings.data_selection import DataSelection

_MODAL_SELECT_STYLESHEET = """
:host {
  color: #172B4D !important;
}
.bk-input-group {
  margin: 0 !important;
  color: #172B4D !important;
}
label {
  color: #172B4D !important;
}
select,
select.bk-input,
.bk-input {
  background-color: #ffffff !important;
  color: #172B4D !important;
  border: 1px solid #A6B1C2 !important;
  border-radius: 4px !important;
  box-shadow: none !important;
  outline: none !important;
  padding-right: 28px !important;
}
select:focus,
select.bk-input:focus,
.bk-input:focus {
  background-color: #ffffff !important;
  color: #172B4D !important;
  border-color: #4C9AFF !important;
  box-shadow: 0 0 0 2px rgba(76, 154, 255, 0.22) !important;
}
option {
  background-color: #ffffff !important;
  color: #172B4D !important;
}
"""

_MODAL_CHECKBOX_STYLESHEET = """
:host {
  color: #172B4D !important;
}
.bk-input-group,
label,
span {
  color: #172B4D !important;
}
input[type="checkbox"] {
  background-color: #ffffff !important;
}
"""

def _append_stylesheet(widget: Any, stylesheet: str) -> None:
    try:
        stylesheets = list(getattr(widget, "stylesheets", []) or [])
        if stylesheet not in stylesheets:
            stylesheets.append(stylesheet)
        widget.stylesheets = stylesheets
    except Exception:
        pass

def _install_application_chrome_css() -> None:
    marker = "--al-application-chrome-v7"
    if any(marker in css for css in pn.config.raw_css):
        return
    pn.config.raw_css.append(APPLICATION_CHROME_CSS)

class DatasetHeaderController:
    """Global dataset switcher and dataset-loading modal controller."""

    def __init__(self, context: Any, template: Any) -> None:
        self.context = context
        self.template = template
        self._subs: list[Any] = []
        self._updating_select = False

        _install_application_chrome_css()

        self.context_label = pn.pane.HTML(
            '<span aria-hidden="true">Dataset</span>',
            width=52,
            height=30,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-context-label"],
        )
        self.active_dataset_select = pn.widgets.Select(
            name="",
            options=OrderedDict({"No dataset loaded": ""}),
            value="",
            width=270,
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-header-dataset-select"],
            stylesheets=[HEADER_SELECT_STYLESHEET],
        )
        self.active_dataset_select.description = "Switch the active dataset"
        self.dataset_stats = pn.pane.HTML(
            "",
            width=178,
            height=30,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
            visible=True,
            css_classes=["al-dataset-stats"],
        )
        self.active_dataset_select.param.watch(
            self._on_active_dataset_select_changed,
            "value",
        )

        self.button = pn.widgets.Button(
            name="Add data",
            icon="database-plus",
            button_type="default",
            width=98,
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-header-add-data"],
            stylesheets=[HEADER_BUTTON_STYLESHEET],
        )
        self.button.description = "Load and register another dataset"
        self.button.on_click(self._open_modal)

        self.view = pn.Row(
            self.context_label,
            self.active_dataset_select,
            self.button,
            self.dataset_stats,
            sizing_mode="fixed",
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-header"],
            styles={"overflow": "visible", "min-width": "0"},
        )

        self.close_button = pn.widgets.Button(
            name="Close",
            button_type="light",
            width=120,
            height=38,
            margin=(8, 20, 8, 20),
        )
        self.close_button.on_click(
            lambda _event: close_template_modal(self.template)
        )

        self.data_selection = HeaderDataSelection(
            src=ColumnDataSource(data={}),
            mode="Exploring",
            context=self.context,
            close_settings_button=self.close_button,
            on_dataset_loaded=self._on_dataset_loaded_from_modal,
        )
        self.modal_root = self._build_modal_root()

        self._subscribe()
        self._refresh_header()

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------

    def _subscribe(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        for topic in (
            "dataset.loaded",
            "dataset.active.changed",
            "dataset.updated",
            "dataset.removed",
            "dataset.open_requested",
            "dataset.mapping.updated",
        ):
            try:
                sub = events.subscribe(
                    topic,
                    self._on_dataset_event,
                    owner_id="platform.dataset_header",
                    owner_label="Dataset Header",
                    owner_kind="platform-header",
                )
            except TypeError:
                sub = events.subscribe(topic, self._on_dataset_event)
            self._subs.append(sub)

    def _on_dataset_event(self, topic: str, payload: Any) -> None:
        del payload
        if topic == "dataset.open_requested":
            self._open_modal()
            return
        self._refresh_header()

    def dispose(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        for sub in list(self._subs):
            try:
                events.unsubscribe(sub)
            except Exception:
                pass
        self._subs.clear()

    # ------------------------------------------------------------------
    # Header state
    # ------------------------------------------------------------------

    def _refresh_header(self) -> None:
        ids = self._dataset_ids()
        active_id = self._active_dataset_id_or_none()

        self._updating_select = True
        try:
            if not ids:
                self.active_dataset_select.options = OrderedDict(
                    {"No dataset loaded": ""}
                )
                self.active_dataset_select.value = ""
                self.active_dataset_select.disabled = True
                self.active_dataset_select.description = "No dataset is loaded"
                self.dataset_stats.object = ""
                self.dataset_stats.visible = False
                self.button.button_type = "primary"
                self.button.stylesheets = [HEADER_PRIMARY_BUTTON_STYLESHEET]
                self.button.description = "Load the first dataset"
                return

            options = self._dataset_option_labels(ids)
            self.active_dataset_select.options = options
            self.active_dataset_select.disabled = False
            self.active_dataset_select.value = (
                active_id if active_id in ids else ids[0]
            )
            selected_id = str(self.active_dataset_select.value)
            name = self._dataset_name(selected_id)
            dimensions = self._dataset_dimensions(selected_id)
            self.active_dataset_select.description = (
                f"Active dataset: {name}"
                + (f" ({dimensions})" if dimensions else "")
            )
            self.dataset_stats.object = (
                dimensions
                if dimensions
                else '<span aria-hidden="true">&nbsp;</span>'
            )
            self.dataset_stats.visible = True
            self.button.button_type = "default"
            self.button.stylesheets = [HEADER_BUTTON_STYLESHEET]
            self.button.description = "Load and register another dataset"
        finally:
            self._updating_select = False

    def _on_active_dataset_select_changed(self, event: Any) -> None:
        if self._updating_select:
            return
        dataset_id = event.new
        if not dataset_id:
            return
        self._set_active_dataset(dataset_id)

    def _dataset_option_labels(self, dataset_ids: list[str]) -> OrderedDict:
        names = [self._dataset_name(dataset_id) for dataset_id in dataset_ids]
        counts: dict[str, int] = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1

        options: OrderedDict[str, str] = OrderedDict()
        for dataset_id, name in zip(dataset_ids, names):
            label = name if counts[name] == 1 else f"{name} · {dataset_id}"
            options[label] = dataset_id
        return options

    def _dataset_name(self, dataset_id: str) -> str:
        try:
            dataset = self.context.datasets.get(dataset_id)
            return str(getattr(dataset, "name", None) or dataset_id)
        except Exception:
            return str(dataset_id)

    def _dataset_dimensions(self, dataset_id: str) -> str:
        try:
            rows = self.context.datasets.row_count(dataset_id)
        except Exception:
            rows = None
        try:
            columns = self.context.datasets.list_columns(dataset_id)
            cols = len(columns)
        except Exception:
            cols = None

        if rows is not None and cols is not None:
            return f"{rows:,} rows · {cols:,} cols"
        if rows is not None:
            return f"{rows:,} rows"
        if cols is not None:
            return f"{cols:,} cols"
        return ""

    def _dataset_label(self, dataset_id: str) -> str:
        return self._dataset_name(dataset_id)

    @staticmethod
    def _cache_dir_for_file(filename: str) -> Path:
        try:
            return Path(filename).expanduser().resolve().parent / ".astronomical_cache"
        except Exception:
            return Path.cwd() / ".astronomical_cache"

    # ------------------------------------------------------------------
    # Modal
    # ------------------------------------------------------------------

    def _open_modal(self, _event: Any = None) -> None:
        open_template_modal(
            self.template,
            self.modal_root,
            close_on_backdrop=True,
        )

    def _on_dataset_loaded_from_modal(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        filename: str,
        optimise_data: bool,
        df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        result = self._register_loaded_dataset(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            filename=filename,
            df=df,
            optimise_data=optimise_data,
            set_active=True,
        )
        self._refresh_header()
        return result

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def _register_loaded_dataset(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        filename: str,
        df: pd.DataFrame | None = None,
        optimise_data: bool,
        set_active: bool = True,
    ) -> dict[str, Any]:
        lower_filename = str(filename).lower()
        cache_dir = self._cache_dir_for_file(filename)

        print(
            "[AstronomicAL loader] --------------------------------------------------",
            flush=True,
        )
        print(f"[AstronomicAL loader] Loading dataset: {dataset_name}", flush=True)
        print(f"[AstronomicAL loader] Dataset id: {dataset_id}", flush=True)
        print(f"[AstronomicAL loader] Source file: {filename}", flush=True)
        print(f"[AstronomicAL loader] Optimise data: {optimise_data}", flush=True)
        print(f"[AstronomicAL loader] Cache dir: {cache_dir}", flush=True)

        if lower_filename.endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
            print("[AstronomicAL loader] Detected FITS input.", flush=True)
            result = register_fits_table(
                self.context.datasets,
                filename,
                cache_dir=cache_dir,
                hdu=1,
                dataset_id=dataset_id,
                name=dataset_name,
                overwrite=False,
            )
        elif lower_filename.endswith((".parquet", ".pq")):
            print("[AstronomicAL loader] Detected Parquet input.", flush=True)
            print(
                "[AstronomicAL loader] Registering Parquet lazily with DuckDB.",
                flush=True,
            )
            self.context.datasets.register_parquet(
                dataset_id,
                filename,
                name=dataset_name,
                source_path=filename,
                loader_id="settings.data_selection",
                optimise_data=optimise_data,
            )
            result = {
                "dataset_id": dataset_id,
                "parquet_path": filename,
                "created": False,
            }
        else:
            print(
                "[AstronomicAL loader] Falling back to pandas registration.",
                flush=True,
            )
            if df is None:
                raise ValueError(
                    "This large-data loading path currently supports FITS and "
                    "Parquet directly. Non-FITS/non-Parquet loaders must pass a "
                    "DataFrame during the transition."
                )
            self.context.datasets.register(
                dataset_id,
                df,
                name=dataset_name,
                source_path=filename,
                loader_id="settings.data_selection",
                optimise_data=optimise_data,
                rows=len(df),
                columns=list(df.columns),
            )
            result = {
                "dataset_id": dataset_id,
                "pandas_fallback": True,
                "created": True,
            }

        print(
            "[AstronomicAL loader] Dataset backend registration complete.",
            flush=True,
        )

        try:
            rows = self.context.datasets.row_count(dataset_id)
        except Exception:
            rows = len(df) if df is not None else None
        try:
            columns = self.context.datasets.list_columns(dataset_id)
        except Exception:
            columns = list(df.columns) if df is not None else []
        self._publish(
            "dataset.loaded",
            {
                "dataset_id": dataset_id,
                "name": dataset_name,
                "rows": rows,
                "columns": columns,
                "source_path": filename,
                "loader_id": "settings.data_selection",
                "optimise_data": optimise_data,
                "backend": self.context.datasets.get_meta(dataset_id).get(
                    "backend"
                ),
            },
        )

        if set_active:
            self._clear_selection_for_dataset_switch()
            self.context.datasets.set_active(
                dataset_id,
                origin="platform.dataset_header",
            )
        return result

    def _set_active_dataset(self, dataset_id: str) -> None:
        self._clear_selection_for_dataset_switch()
        self.context.datasets.set_active(
            dataset_id,
            origin="platform.dataset_header",
        )
        self._refresh_header()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        events = getattr(self.context, "events", None)
        if events is not None:
            events.publish(topic, payload)

    def _clear_selection_for_dataset_switch(self) -> None:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return
        try:
            selection.clear_focus(origin="dataset.active.changed")
        except Exception:
            pass
        try:
            selection.clear_selection_set(origin="dataset.active.changed")
        except Exception:
            pass

    def has_active_dataset(self) -> bool:
        """Return whether an active dataset is registered with the manager."""
        dataset_id = self._active_dataset_id_or_none()
        if dataset_id in (None, ""):
            return False
        try:
            return dataset_id in self.context.datasets.list_ids()
        except Exception:
            return False

    def _dataset_ids(self) -> list[str]:
        try:
            return list(self.context.datasets.list_ids())
        except Exception:
            return []

    def _active_dataset_id_or_none(self) -> Optional[str]:
        try:
            return self.context.datasets.active_id()
        except Exception:
            return None

    def _build_modal_root(self) -> pn.Column:
        close_settings_button = pn.widgets.Button(
            name="Close",
            button_type="default",
            width=130,
            height=36,
            margin=(8, 20, 10, 0),
        )
        close_settings_button.on_click(
            lambda _event: close_template_modal(self.template)
        )
        self.modal_close_button = close_settings_button

        try:
            setattr(
                self.context,
                "_dataset_header_modal_close_button",
                self.modal_close_button,
            )
        except Exception:
            pass

        data_selection_view = self.data_selection.panel()
        data_selection_view.margin = (0, 0, 0, 0)

        header = pn.pane.HTML(
            """
            <div class="al-modal-heading">Select your data</div>
            <div class="al-modal-subheading">
              Load a dataset into the platform, optionally using an existing
              configuration or layout.
            </div>
            """,
            sizing_mode="stretch_width",
            height=58,
            margin=(0, 0, 10, 0),
        )
        body = pn.Column(
            data_selection_view,
            sizing_mode="fixed",
            width=660,
            height=390,
            scroll=True,
            margin=(0, 0, 0, 0),
            styles={
                "padding": "16px 20px",
                "box-sizing": "border-box",
                "overflow-y": "auto",
                "overflow-x": "hidden",
            },
            css_classes=["al-modal-body", "al-dataset-modal-body"],
        )
        footer = pn.Row(
            pn.layout.HSpacer(),
            self.modal_close_button,
            sizing_mode="fixed",
            width=660,
            height=64,
            margin=(14, 0, 0, 0),
            styles={
                "padding": "12px 20px 12px 0",
                "box-sizing": "border-box",
                "overflow": "visible",
            },
            css_classes=["al-modal-footer"],
        )
        return pn.Column(
            header,
            body,
            footer,
            sizing_mode="fixed",
            width=690,
            height=558,
            margin=(0, 0, 0, 0),
            styles={"box-sizing": "border-box", "overflow": "hidden"},
            css_classes=["al-modal-card", "al-dataset-modal-card"],
        )

class HeaderDataSelection(DataSelection):
    """DataSelection variant used inside the dataset-header modal."""

    def __init__(
        self,
        src: Any,
        mode: str,
        context: Any,
        close_settings_button: Any,
        on_dataset_loaded: Any,
    ) -> None:
        self._on_dataset_loaded_callback = on_dataset_loaded
        super().__init__(
            src=src,
            mode=mode,
            context=context,
            close_settings_button=close_settings_button,
        )

    def _refresh_layout(self) -> None:
        controls = self._build_controls()
        controls.margin = (0, 0, 0, 0)
        info = self._build_info()
        info.margin = (0, 20, 0, 20)
        divider = pn.pane.HTML(
            '<div style="height:1px;background:#e5e7eb;margin:11px 0 12px;"></div>',
            sizing_mode="stretch_width",
            height=24,
            margin=(0, 0, 0, 0),
        )

        self.view.sizing_mode = "stretch_width"
        self.view.margin = (0, 0, 0, 0)
        self.view.objects = [controls, divider, info]

    def _build_controls(self) -> pn.Column:
        self._compact_widget_layout()

        memory_row = pn.Row(
            self.memory_optimisation_check,
            self._memory_opt_tooltip,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            height=24,
        )
        dataset_block = pn.Column(
            self._field_label("Data file"),
            self.dataset_widget,
            sizing_mode="fixed",
            width=340,
            min_height=58,
            margin=(0, 0, 0, 0),
            styles={"overflow": "visible"},
        )
        workspace_file_block = pn.Column(
            self._field_label("Workspace file"),
            self.workspace_file_widget,
            sizing_mode="fixed",
            width=340,
            min_height=58,
            margin=(0, 0, 0, 0),
            styles={"overflow": "visible"},
        )
        button = (
            self.load_workspace_button
            if self.load_layout_check
            else self.load_data_button
        )
        button_block = pn.Row(
            button,
            sizing_mode="fixed",
            width=340,
            height=40,
            margin=(0, 0, 0, 0),
        )

        if self.load_layout_check:
            return pn.Column(
                self.load_layout_widget,
                pn.Spacer(height=6),
                workspace_file_block,
                pn.Spacer(height=10),
                button_block,
                sizing_mode="fixed",
                width=380,
                margin=(0, 20, 0, 20),
                styles={"overflow": "visible"},
            )

        return pn.Column(
            self.load_layout_widget,
            memory_row,
            pn.Spacer(height=10),
            dataset_block,
            pn.Spacer(height=10),
            button_block,
            pn.Spacer(height=10),
            sizing_mode="fixed",
            width=380,
            margin=(12, 20, 0, 20),
            css_classes=["al-dataset-modal-controls"],
            styles={"overflow": "visible", "color": "#172B4D"},
        )

    def _compact_widget_layout(self) -> None:
        widgets = [
            getattr(self, "load_layout_widget", None),
            getattr(self, "memory_optimisation_check", None),
            getattr(self, "dataset_widget", None),
            getattr(self, "workspace_file_widget", None),
            getattr(self, "load_data_button", None),
            getattr(self, "load_workspace_button", None),
        ]
        for widget in widgets:
            if widget is None:
                continue
            try:
                widget.margin = (0, 0, 0, 0)
            except Exception:
                pass

        for checkbox in (
            getattr(self, "load_layout_widget", None),
            getattr(self, "memory_optimisation_check", None),
        ):
            if checkbox is None:
                continue
            try:
                checkbox.styles = {
                    **dict(getattr(checkbox, "styles", {}) or {}),
                    "color": "#172B4D",
                }
            except Exception:
                pass
            _append_stylesheet(checkbox, _MODAL_CHECKBOX_STYLESHEET)

        for select_widget in (
            getattr(self, "dataset_widget", None),
            getattr(self, "workspace_file_widget", None),
        ):
            if select_widget is None:
                continue
            try:
                select_widget.name = ""
                select_widget.width = 320
                select_widget.height = 34
                select_widget.sizing_mode = "fixed"
                select_widget.margin = (0, 0, 0, 0)
                select_widget.styles = {
                    **dict(getattr(select_widget, "styles", {}) or {}),
                    "background": "#ffffff",
                    "color": "#172B4D",
                }
            except Exception:
                pass
            _append_stylesheet(select_widget, _MODAL_SELECT_STYLESHEET)

        for button in (
            getattr(self, "load_data_button", None),
            getattr(self, "load_workspace_button", None),
        ):
            if button is None:
                continue
            try:
                button.height = 38
                button.width = 220
                button.margin = (0, 0, 0, 0)
            except Exception:
                pass

    @staticmethod
    def _field_label(text: str) -> pn.pane.HTML:
        return pn.pane.HTML(
            f'<div style="font-weight:600;color:#172B4D;line-height:20px;">{text}</div>',
            height=20,
            margin=(0, 0, 4, 0),
            sizing_mode="fixed",
        )

    def _load_data_cb(self, event: Any) -> Any:
        del event
        self.load_data_button.disabled = True
        self.load_data_button.name = "Preparing dataset import..."
        print("[AstronomicAL loader] Load button clicked.", flush=True)

        try:
            filename = self.dataset
            optimise_data = bool(self.memory_optimisation_check.value)
            dataset_id = self._unique_dataset_id(
                self._normalise_dataset_id(Path(filename).stem)
            )
            dataset_name = (
                Path(filename).stem.replace("_", " ").replace("-", " ").title()
            )

            self.load_data_button.name = "Importing and caching dataset..."
            print(
                "[AstronomicAL loader] Starting dataset import callback.",
                flush=True,
            )
            result = self._on_dataset_loaded_callback(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                filename=filename,
                optimise_data=optimise_data,
                df=None,
            )

            self.load_data_button.name = "Finalising dataset..."
            print(
                "[AstronomicAL loader] Dataset import callback complete.",
                flush=True,
            )

            try:
                self.df = self.context.datasets.head(dataset_id, n=1)
            except Exception:
                try:
                    columns = self.context.datasets.list_columns(dataset_id)
                except Exception:
                    columns = []
                self.df = pd.DataFrame(columns=columns)

            self._initialise_src()
            self.ready = True
            self.load_data_button.name = "File Loaded."
            print("[AstronomicAL loader] File loaded successfully.", flush=True)

            self.close_settings_button.disabled = False
            self.close_settings_button.button_type = "success"
            self.close_settings_button.name = "Close Settings"

            try:
                modal_close_button = getattr(
                    self.context,
                    "_dataset_header_modal_close_button",
                    None,
                )
                if modal_close_button is not None:
                    modal_close_button.disabled = False
                    modal_close_button.button_type = "success"
                    modal_close_button.name = "Close Settings"
            except Exception:
                pass
            return result
        except Exception as exc:
            self.error_message = f"Unable to load data file: `{exc}`"
            self.load_data_button.name = "Unable to load file"
            print(f"[AstronomicAL loader] ERROR: {exc}", flush=True)
            self._refresh_layout()
            raise
        finally:
            self.load_data_button.disabled = False

    def _unique_dataset_id(self, base: str) -> str:
        try:
            existing = set(self.context.datasets.list_ids())
        except Exception:
            existing = set()

        if base not in existing:
            return base

        index = 2
        while f"{base}_{index}" in existing:
            index += 1
        return f"{base}_{index}"

    @staticmethod
    def _normalise_dataset_id(value: str) -> str:
        value = (value or "dataset").strip().lower()
        value = re.sub(r"[^a-z0-9_]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        return value or "dataset"