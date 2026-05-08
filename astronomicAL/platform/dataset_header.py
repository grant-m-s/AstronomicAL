from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import panel as pn

from astronomicAL.platform.modal_utils import open_template_modal
from astronomicAL.settings.data_selection import DataSelection


class DatasetHeaderController:
    """Global dataset header control.

    The header owns:
    - quick switching between loaded datasets
    - opening the dataset loading modal

    The modal body deliberately reuses the existing DataSelection settings UI
    so the loading experience matches the old Exploration entry screen.
    """

    def __init__(self, context, template) -> None:
        self.context = context
        self.template = template
        self._subs = []
        self._updating_select = False

        self.active_dataset_select = pn.widgets.Select(
            name="",
            options=OrderedDict({"No dataset loaded": ""}),
            value="",
            width=260,
            height=34,
            margin=(6, 4, 6, 8),
        )
        self.active_dataset_select.param.watch(
            self._on_active_dataset_select_changed,
            "value",
        )

        self.button = pn.widgets.Button(
            name="Add Data",
            button_type="default",
            width=92,
            height=38,
            margin=(6, 8, 6, 4),
        )
        self.button.on_click(self._open_modal)

        self.view = pn.Row(
            self.active_dataset_select,
            self.button,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        )

        self.close_button = pn.widgets.Button(
            name="Close",
            button_type="light",
            width=120,
            height=38,
            margin=(8, 20, 8, 20),
        )
        self.close_button.on_click(lambda _event: self.template.close_modal())

        self.data_selection = HeaderDataSelection(
            src=self._legacy_source(),
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
            "dataset.mapping_updated",
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
                return

            options = OrderedDict()
            for dataset_id in ids:
                options[self._dataset_label(dataset_id)] = dataset_id

            self.active_dataset_select.options = options
            self.active_dataset_select.disabled = False

            if active_id in ids:
                self.active_dataset_select.value = active_id
            else:
                self.active_dataset_select.value = ids[0]

        finally:
            self._updating_select = False

    def _on_active_dataset_select_changed(self, event) -> None:
        if self._updating_select:
            return

        dataset_id = event.new
        if not dataset_id:
            return

        self._set_active_dataset(dataset_id)

    def _dataset_label(self, dataset_id: str) -> str:
        try:
            dataset = self.context.datasets.get(dataset_id)
            rows = len(dataset.df)
            cols = len(dataset.df.columns)
            return f"{dataset.name} · {rows:,} × {cols:,}"
        except Exception:
            return dataset_id

    # ------------------------------------------------------------------
    # Modal
    # ------------------------------------------------------------------

    def _open_modal(self, _event=None) -> None:
        open_template_modal(self.template, self.modal_root)

    def _on_dataset_loaded_from_modal(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        filename: str,
        df: pd.DataFrame,
        optimise_data: bool,
    ) -> None:
        self._register_loaded_dataset(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            filename=filename,
            df=df,
            optimise_data=optimise_data,
            set_active=True,
        )
        self._refresh_header()

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def _register_loaded_dataset(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        filename: str,
        df: pd.DataFrame,
        optimise_data: bool,
        set_active: bool = True,
    ) -> None:
        previous_dataset_id = self._active_dataset_id_or_none()

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

        self._sync_legacy_config(
            dataset_id=dataset_id,
            filename=filename,
            df=df,
            optimise_data=optimise_data,
        )

        self._publish(
            "dataset.loaded",
            {
                "dataset_id": dataset_id,
                "name": dataset_name,
                "rows": len(df),
                "columns": list(df.columns),
                "source_path": filename,
                "loader_id": "settings.data_selection",
                "optimise_data": optimise_data,
            },
        )

        if set_active:
            self.context.datasets.set_active(dataset_id)
            self._clear_selection_for_dataset_switch()

            self._publish(
                "dataset.active.changed",
                {
                    "dataset_id": dataset_id,
                    "previous_dataset_id": previous_dataset_id,
                    "source": "DatasetHeaderController",
                },
            )

    def _set_active_dataset(self, dataset_id: str) -> None:
        previous_dataset_id = self._active_dataset_id_or_none()

        self.context.datasets.set_active(dataset_id)
        df = self.context.datasets.get_df(dataset_id)

        try:
            dataset = self.context.datasets.get(dataset_id)
            filename = dataset.meta.get("source_path", "")
            optimise_data = bool(dataset.meta.get("optimise_data", True))
        except Exception:
            filename = ""
            optimise_data = True

        self._sync_legacy_config(
            dataset_id=dataset_id,
            filename=filename,
            df=df,
            optimise_data=optimise_data,
        )

        self._clear_selection_for_dataset_switch()

        self._publish(
            "dataset.active.changed",
            {
                "dataset_id": dataset_id,
                "previous_dataset_id": previous_dataset_id,
                "source": "DatasetHeaderController",
            },
        )

        self._refresh_header()

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    def _legacy_source(self):
        config = getattr(self.context, "config", None)
        if config is not None:
            try:
                return config.source
            except Exception:
                pass
        return None

    def _sync_legacy_config(
        self,
        *,
        dataset_id: str,
        filename: str,
        df: pd.DataFrame,
        optimise_data: bool,
    ) -> None:
        config = getattr(self.context, "config", None)
        if config is None:
            return

        if not hasattr(config, "settings") or config.settings is None:
            config.settings = {}

        config.settings["dataset_filepath"] = filename
        config.settings["optimise_data"] = optimise_data
        config.settings["active_dataset_id"] = dataset_id

        try:
            config.main_df = df
        except Exception:
            pass

        try:
            config.source.data = {f"{col}": [] for col in df.columns}
        except Exception:
            pass

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


    def _build_modal_root(self):
        """Build a compact modal matching the old settings/data screen.

        Important: the scroll shell needs an explicit height. Using only
        max_height can collapse the modal body in Panel/Bokeh layouts.
        """

        close_settings_button = pn.widgets.Button(
            name="Close",
            button_type="default",
            width=150,
            height=36,
            margin=(0, 0, 0, 0),
        )
        close_settings_button.on_click(lambda _event: self.template.close_modal())

        self.modal_close_button = close_settings_button

        try:
            setattr(self.context, "_dataset_header_modal_close_button", self.modal_close_button)
        except Exception:
            pass

        data_selection_view = self.data_selection.panel()
        data_selection_view.margin = (0, 0, 0, 0)

        card = pn.Column(
            pn.Row(
                pn.layout.HSpacer(),
                self.modal_close_button,
                sizing_mode="stretch_width",
                margin=(0, 0, 12, 0),
            ),
            pn.pane.Markdown(
                "## Select Your Data",
                sizing_mode="stretch_width",
                margin=(0, 0, 10, 0),
                styles={
                    "color": "#172B4D",
                },
            ),
            data_selection_view,
            sizing_mode="stretch_width",
            width=620,
            max_width=620,
            margin=(0, 0, 0, 0),
            styles={
                "background": "#ffffff",
                "padding": "24px 32px 28px 32px",
                "box-sizing": "border-box",
                "overflow": "visible",
            },
        )

        scroll_shell = pn.Column(
            card,
            sizing_mode="fixed",
            width=660,
            height=640,
            scroll=True,
            margin=(0, 0, 0, 0),
            styles={
                "background": "#ffffff",
                "border-radius": "6px",
                "box-sizing": "border-box",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "box-shadow": "0 16px 48px rgba(15, 23, 42, 0.22)",
            },
        )

        return pn.Row(
            pn.layout.HSpacer(),
            scroll_shell,
            pn.layout.HSpacer(),
            sizing_mode="stretch_width",
            margin=(8, 0, 8, 0),
            styles={
                "box-sizing": "border-box",
                "overflow": "visible",
            },
        )



class HeaderDataSelection(DataSelection):
    """DataSelection variant used inside the dataset header modal.

    It keeps the original DataSelection UI/layout, but changes the load action
    so the dataframe is registered with the platform DatasetManager instead of
    only advancing the settings pipeline.
    """

    def __init__(
        self,
        src,
        mode,
        context,
        close_settings_button,
        on_dataset_loaded,
    ):
        self._on_dataset_loaded_callback = on_dataset_loaded
        super().__init__(
            src=src,
            mode=mode,
            context=context,
            close_settings_button=close_settings_button,
        )

    def _refresh_layout(self):
        """Compact DataSelection layout for the dataset header modal.

        Avoid pn.layout.Divider here. In the shared template modal it can become a
        flex item that expands vertically, creating a large blank gap before the
        information block.
        """

        controls = self._build_controls()
        controls.margin = (0, 0, 0, 0)

        info = self._build_info()
        info.margin = (0, 20, 0, 20)

        divider = pn.pane.HTML(
            """
            <div style="
                width: 100%;
                height: 1px;
                border-top: 2px solid #555;
                margin: 10px 20px 12px 20px;
                box-sizing: border-box;
            "></div>
            """,
            sizing_mode="stretch_width",
            height=24,
            margin=(0, 0, 0, 0),
        )

        self.view.sizing_mode = "stretch_width"
        self.view.margin = (0, 0, 0, 0)

        self.view.objects = [
            controls,
            divider,
            info,
        ]

    def _build_controls(self):
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
            styles={
                "overflow": "visible",
            },
        )

        config_file_block = pn.Column(
            self._field_label("Configuration file"),
            self.config_file_widget,
            sizing_mode="fixed",
            width=340,
            min_height=58,
            margin=(0, 0, 0, 0),
            styles={
                "overflow": "visible",
            },
        )

        load_option_block = pn.Column(
            self._field_label("Load config options"),
            self.load_config_select_widget,
            sizing_mode="fixed",
            width=340,
            min_height=58,
            margin=(0, 0, 0, 0),
            styles={
                "overflow": "visible",
            },
        )

        button = self.load_data_button if not self.load_layout_check else self.load_data_button_js

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
                config_file_block,
                pn.Spacer(height=8),
                load_option_block,
                pn.Spacer(height=10),
                button_block,
                sizing_mode="fixed",
                width=380,
                margin=(0, 20, 0, 20),
                styles={
                    "overflow": "visible",
                },
            )

        return pn.Column(
            self.load_layout_widget,
            memory_row,
            pn.Spacer(height=10),
            dataset_block,
            pn.Spacer(height=10),
            button_block,
            sizing_mode="fixed",
            width=380,
            margin=(0, 20, 0, 20),
            styles={
                "overflow": "visible",
            },
        )
    
    def _compact_widget_layout(self) -> None:
        """Remove margins inherited from the original settings screen widgets."""

        widgets = [
            getattr(self, "load_layout_widget", None),
            getattr(self, "memory_optimisation_check", None),
            getattr(self, "dataset_widget", None),
            getattr(self, "config_file_widget", None),
            getattr(self, "load_config_select_widget", None),
            getattr(self, "load_data_button", None),
            getattr(self, "load_data_button_js", None),
        ]

        for widget in widgets:
            if widget is None:
                continue

            try:
                widget.margin = (0, 0, 0, 0)
            except Exception:
                pass

        for select_widget in (
            getattr(self, "dataset_widget", None),
            getattr(self, "config_file_widget", None),
            getattr(self, "load_config_select_widget", None),
        ):
            if select_widget is None:
                continue

            try:
                select_widget.name = ""
                select_widget.width = 320
                select_widget.height = 34
                select_widget.sizing_mode = "fixed"
                select_widget.margin = (0, 0, 0, 0)
            except Exception:
                pass

        for button in (
            getattr(self, "load_data_button", None),
            getattr(self, "load_data_button_js", None),
        ):
            if button is None:
                continue

            try:
                button.height = 38
                button.width = 220
                button.margin = (0, 0, 0, 0)
            except Exception:
                pass

    def _field_label(self, text: str):
        return pn.pane.HTML(
            f"""
            <div style="
                font-size: 13px;
                font-weight: 700;
                color: #44546A;
                line-height: 18px;
                height: 20px;
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            ">
                {text}
            </div>
            """,
            height=20,
            margin=(0, 0, 4, 0),
            sizing_mode="fixed",
        )

    def _load_data_cb(self, event):
        self.load_data_button.disabled = True
        self.load_data_button.name = "Loading File..."

        try:
            filename = self.dataset
            optimise_data = bool(self.memory_optimisation_check.value)

            if self.config is not None:
                self.config.settings["dataset_filepath"] = filename

            df = self.get_dataframe_from_fits_file(
                filename,
                optimise_data=optimise_data,
            )

            self.df = df

            if self.config is not None:
                self.config.main_df = df

            self._initialise_src()

            dataset_id = self._unique_dataset_id(
                self._normalise_dataset_id(Path(filename).stem)
            )
            dataset_name = Path(filename).stem.replace("_", " ").replace("-", " ").title()

            self._on_dataset_loaded_callback(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                filename=filename,
                df=df,
                optimise_data=optimise_data,
            )

            self.ready = True
            self.load_data_button.name = "File Loaded."

            # Legacy/inner close button passed into DataSelection.
            self.close_settings_button.disabled = False
            self.close_settings_button.button_type = "success"
            self.close_settings_button.name = "Close Settings"

            # Visible modal close button owned by DatasetHeaderController.
            try:
                modal_close_button = getattr(self.context, "_dataset_header_modal_close_button", None)
                if modal_close_button is not None:
                    modal_close_button.disabled = False
                    modal_close_button.button_type = "success"
                    modal_close_button.name = "Close Settings"
            except Exception:
                pass

        except Exception as exc:
            self.error_message = f"Unable to load data file: `{exc}`"
            self.load_data_button.name = "Unable to load file"
            self._refresh_layout()
            raise

        finally:
            self.load_data_button.disabled = False

    def _unique_dataset_id(self, base: str) -> str:
        existing = set()

        try:
            existing = set(self.context.datasets.list_ids())
        except Exception:
            existing = set()

        if base not in existing:
            return base

        i = 2
        while f"{base}_{i}" in existing:
            i += 1

        return f"{base}_{i}"

    def _normalise_dataset_id(self, value: str) -> str:
        value = (value or "dataset").strip().lower()
        value = re.sub(r"[^a-z0-9_]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        return value or "dataset"