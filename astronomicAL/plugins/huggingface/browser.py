from __future__ import annotations

import html
import threading
import traceback
from collections import OrderedDict
from typing import Any

import pandas as pd
import panel as pn

from .importer import (
    find_registered_hf_dataset,
    import_hf_image_dataset_as_manifest,
    slugify,
)
from .service import HFDatasetDetails, HuggingFaceDatasetService
from .styles import (
    BUTTON_PRIMARY_STYLESHEET,
    BUTTON_SECONDARY_STYLESHEET,
    BUTTON_SUCCESS_STYLESHEET,
    CHECKBOX_STYLESHEET,
    FORM_CHECKBOX_STYLESHEET,
    EMPTY_STATE_STYLES,
    INSPECTOR_STYLES,
    NUMBER_INPUT_STYLESHEET,
    PROGRESS_STYLESHEET,
    RESULT_BUTTON_STYLESHEET,
    RESULTS_LIST_STYLES,
    ROOT_STYLES,
    SEARCH_BAR_INPUT_STYLESHEET,
    SEARCH_INPUT_STYLESHEET,
    SELECT_STYLESHEET,
    SHELL_STYLES,
    SUBSURFACE_STYLES,
    SURFACE_STYLES,
    TOOLBAR_STYLES,
    install_huggingface_styles,
)

DEFAULT_TASKS = OrderedDict(
    {
        "Image datasets": "image-classification",
        "All datasets": "",
        "Object detection": "object-detection",
        "Semantic segmentation": "semantic-segmentation",
        "Image-to-text": "image-to-text",
        "Depth estimation": "depth-estimation",
    }
)

SORT_OPTIONS = OrderedDict(
    {
        "Most downloaded": "downloads",
        "Most liked": "likes",
        "Recently updated": "last_modified",
    }
)

STRATEGY_OPTIONS = OrderedDict(
    {
        "Automatic": "auto",
        "Structured dataset rows": "builder",
        "Image files and folders": "files",
    }
)


class HuggingFaceBrowserPanel:
    """Search, inspect, preview, and import Hugging Face image datasets.

    The normal workflow is deliberately small:

    1. Search for a dataset or paste an exact ``owner/repository`` ID.
    2. Select a result and let AstronomicAL inspect its configs, splits,
       image column, and labels.
    3. Import the detected splits.

    Advanced controls remain available for unusual repositories, but cache
    reuse, strategy selection, and label inference are automatic by default.
    """

    state_version = 7

    def __init__(self, context: Any) -> None:
        self.context = context
        self._disposed = False
        self._doc = pn.state.curdoc
        self._job_handles: list[Any] = []
        self._busy = False
        self._updating_widgets = False
        self._search_df = pd.DataFrame()
        self._details: HFDatasetDetails | None = None
        self._details_mode: str | None = None
        self._details_config: str | None = None
        self._restored_state: dict[str, Any] = {}
        self._last_suggested_dataset_id = ""
        self._last_suggested_dataset_name = ""
        self._inspection_generation = 0
        self._pending_inspection_repo: str | None = None

        self._progress_lock = threading.RLock()
        self._progress_state: dict[str, Any] = {
            "active": False,
            "phase": "idle",
            "completed": 0,
            "total": 0,
            "percent": 0,
            "message": "",
            "current_file": "",
        }
        self._progress_periodic = None

        install_huggingface_styles()
        self._build_widgets()
        self._view = self._build_view()
        self._set_status(
            "Search by topic or paste an exact Hugging Face dataset ID.",
            "info",
        )

    # ------------------------------------------------------------------
    # Public panel lifecycle
    # ------------------------------------------------------------------

    def panel(self):
        return self._view

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True

        try:
            if self._progress_periodic is not None:
                self._progress_periodic.stop()
        except Exception:
            pass
        self._progress_periodic = None

        for handle in list(self._job_handles):
            try:
                handle.cancel()
            except Exception:
                pass
        self._job_handles.clear()

    def get_state(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "query": self.query.value,
            "task": self.task.value,
            "sort": self.sort.value,
            "limit": self.limit.value,
            "selected_repo": self.selected_repo,
            "config": self.config_select.value,
            "splits": list(self.split_choices.value or []),
            "preview_split": self.preview_split.value,
            "preview_limit": self.preview_limit.value,
            "strategy": self.strategy.value,
            "image_column": self.image_column.value,
            "label_column": self.label_column.value,
            "id_column": self.id_column.value,
            "dataset_id": self.dataset_id.value,
            "dataset_name": self.dataset_name.value,
            "max_rows": self.max_rows.value,
            "write_parquet": self.write_parquet.value,
            "active_after_import": self.active_after_import.value,
            "trust_remote_code": self.trust_remote_code.value,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        self._restored_state = dict(state)

        for key, widget in {
            "query": self.query,
            "task": self.task,
            "sort": self.sort,
            "limit": self.limit,
            "preview_limit": self.preview_limit,
            "strategy": self.strategy,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "max_rows": self.max_rows,
            "write_parquet": self.write_parquet,
            "trust_remote_code": self.trust_remote_code,
        }.items():
            if key not in state:
                continue
            try:
                widget.value = state[key]
            except Exception:
                pass

        selected_repo = str(state.get("selected_repo") or "").strip()
        if selected_repo:
            self.query.value = selected_repo
            self._select_repo(selected_repo, inspect=False)

    # ------------------------------------------------------------------
    # Widget and layout construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        self.query = pn.widgets.TextInput(
            name="",
            value="cifar10",
            placeholder="Search a topic or paste owner/dataset",
            height=38,
            sizing_mode="stretch_width",
            stylesheets=[SEARCH_BAR_INPUT_STYLESHEET],
        )
        self.search_button = pn.widgets.Button(
            name="Search",
            icon="search",
            button_type="primary",
            width=118,
            height=38,
            stylesheets=[BUTTON_PRIMARY_STYLESHEET],
        )
        self.search_button.on_click(self._on_search_clicked)

        self.search_options_toggle = pn.widgets.Toggle(
            name="Search options",
            value=False,
            button_type="default",
            width=126,
            height=38,
            stylesheets=[BUTTON_SECONDARY_STYLESHEET],
        )
        self.search_options_toggle.param.watch(
            self._on_search_options_toggled,
            "value",
        )

        self.task = pn.widgets.Select(
            name="Dataset type",
            options=DEFAULT_TASKS,
            value="image-classification",
            sizing_mode="stretch_width",
            stylesheets=[SELECT_STYLESHEET],
        )
        self.sort = pn.widgets.Select(
            name="Sort results",
            options=SORT_OPTIONS,
            value="downloads",
            sizing_mode="stretch_width",
            stylesheets=[SELECT_STYLESHEET],
        )
        self.limit = pn.widgets.IntInput(
            name="Maximum results",
            value=30,
            start=1,
            end=200,
            sizing_mode="stretch_width",
            stylesheets=[NUMBER_INPUT_STYLESHEET],
        )
        self.token = pn.widgets.PasswordInput(
            name="Hugging Face token",
            placeholder="Optional; your saved HF login is used when blank",
            sizing_mode="stretch_width",
            stylesheets=[SEARCH_INPUT_STYLESHEET],
        )

        self._result_items: dict[str, Any] = {}
        self.results_list = pn.Column(
            self._results_empty_html(
                "Search Hugging Face to see matching image datasets."
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles=RESULTS_LIST_STYLES,
        )
        self.results_meta = pn.pane.HTML(
            self._results_meta_html("No search yet"),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.selected_pane = pn.pane.HTML(
            self._selected_html("No dataset selected"),
            sizing_mode="stretch_width",
            height=64,
            margin=(0, 0, 0, 0),
        )
        self.inspect_button = pn.widgets.Button(
            name="Re-inspect",
            icon="refresh",
            button_type="default",
            height=34,
            width=126,
            margin=(0, 0, 0, 0),
            disabled=True,
            stylesheets=[BUTTON_SECONDARY_STYLESHEET],
        )
        self.inspect_button.on_click(self._on_inspect_clicked)

        self.cache_pane = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
        self.structure_pane = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
        self.notice_pane = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.config_select = pn.widgets.Select(
            name="Configuration",
            options=["default"],
            value="default",
            sizing_mode="stretch_width",
            visible=False,
            margin=(8, 0, 0, 0),
            stylesheets=[SELECT_STYLESHEET],
        )
        self.config_select.param.watch(
            self._on_config_changed,
            "value",
        )

        self.split_choices = pn.widgets.CheckBoxGroup(
            name="Splits to import",
            options=[],
            value=[],
            inline=True,
            margin=(8, 0, 0, 0),
            stylesheets=[CHECKBOX_STYLESHEET],
        )
        self.split_choices.param.watch(
            self._on_split_choices_changed,
            "value",
        )

        self.preview_split = pn.widgets.Select(
            name="Preview split",
            options=[],
            sizing_mode="stretch_width",
            margin=(8, 0, 0, 0),
            stylesheets=[SELECT_STYLESHEET],
        )
        self.preview_limit = pn.widgets.IntInput(
            name="Rows",
            value=10,
            start=1,
            end=50,
            height=58,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            stylesheets=[NUMBER_INPUT_STYLESHEET],
        )
        self.preview_button = pn.widgets.Button(
            name="Preview",
            icon="photo",
            button_type="default",
            height=38,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            disabled=True,
            stylesheets=[BUTTON_SECONDARY_STYLESHEET],
        )
        self.preview_button.on_click(self._on_preview_clicked)
        self.preview_grid = pn.Column(
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box",
                "display": "flex",
                "gap": "10px",
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "overflow": "visible",
            },
            css_classes=["al-hf-preview-grid"],
        )

        self.active_after_import = pn.widgets.Select(
            name="Active dataset after import",
            options=OrderedDict({"Do not change active dataset": ""}),
            value="",
            sizing_mode="stretch_width",
            margin=(8, 0, 0, 0),
            stylesheets=[SELECT_STYLESHEET],
        )

        self.dataset_id = pn.widgets.TextInput(
            name="Dataset ID base",
            value="",
            placeholder="Generated automatically",
            height=58,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            stylesheets=[SEARCH_INPUT_STYLESHEET],
        )
        self.dataset_name = pn.widgets.TextInput(
            name="Dataset name base",
            value="",
            placeholder="Generated automatically",
            height=58,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            stylesheets=[SEARCH_INPUT_STYLESHEET],
        )
        self.max_rows = pn.widgets.IntInput(
            name="Maximum rows per split; 0 = all",
            value=0,
            start=0,
            height=58,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            stylesheets=[NUMBER_INPUT_STYLESHEET],
        )
        self.write_parquet = pn.widgets.Checkbox(
            name="Store the AstronomicAL manifest as Parquet",
            value=True,
            height=38,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            stylesheets=[FORM_CHECKBOX_STYLESHEET],
        )
        self.strategy = pn.widgets.Select(
            name="Import strategy",
            options=STRATEGY_OPTIONS,
            value="auto",
            height=58,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            stylesheets=[SELECT_STYLESHEET],
        )
        self.strategy.param.watch(
            self._on_strategy_changed,
            "value",
        )
        self.trust_remote_code = pn.widgets.Checkbox(
            name="Trust repository dataset code",
            value=False,
            height=38,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            stylesheets=[FORM_CHECKBOX_STYLESHEET],
        )
        self.image_column = pn.widgets.Select(
            name="Image column",
            options=[],
            height=58,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            stylesheets=[SELECT_STYLESHEET],
        )
        self.label_column = pn.widgets.Select(
            name="Label column",
            options=OrderedDict({"No label": ""}),
            value="",
            height=58,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            stylesheets=[SELECT_STYLESHEET],
        )
        self.id_column = pn.widgets.Select(
            name="Record ID column",
            options=OrderedDict({"Generate IDs": ""}),
            value="",
            height=58,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            stylesheets=[SELECT_STYLESHEET],
        )

        self.advanced_toggle = pn.widgets.Toggle(
            name="Advanced options",
            value=False,
            button_type="default",
            height=34,
            width=142,
            margin=(8, 0, 0, 0),
            stylesheets=[BUTTON_SECONDARY_STYLESHEET],
        )
        self.advanced_toggle.param.watch(
            self._on_advanced_options_toggled,
            "value",
        )

        self.import_button = pn.widgets.Button(
            name="Import selected splits",
            icon="cloud-download",
            button_type="success",
            height=38,
            sizing_mode="stretch_width",
            disabled=True,
            margin=(10, 0, 0, 0),
            stylesheets=[BUTTON_SUCCESS_STYLESHEET],
        )
        self.import_button.on_click(self._on_import_clicked)

        self.import_summary = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
        self.import_preview = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            visible=False,
            margin=(0, 0, 0, 0),
        )

        self.status = pn.pane.HTML(
            self._status_html("Ready", "info"),
            sizing_mode="stretch_width",
            max_width=390,
            margin=(0, 0, 0, 0),
        )
        self.progress_bar = pn.widgets.Progress(
            name="Progress",
            value=0,
            max=100,
            visible=False,
            height=7,
            sizing_mode="stretch_width",
            stylesheets=[PROGRESS_STYLESHEET],
        )
        self.progress_text = pn.pane.HTML(
            "",
            visible=False,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
    def _build_view(self) -> pn.Column:
        heading = pn.pane.HTML(
            """
            <div style="min-width:0;">
              <div style="color:#263244;font-size:15px;font-weight:720;line-height:1.25;letter-spacing:-0.01em;">
                Find and import labelled image datasets
              </div>
              <div style="margin-top:3px;color:#687386;font-size:11px;line-height:1.4;">
                Search Hugging Face, review the detected structure, then import the splits you need.
                Existing Hugging Face caches and AstronomicAL registrations are reused first.
              </div>
            </div>
            """,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        search_label = pn.pane.HTML(
            """
            <div style="color:#526071;font-size:10px;font-weight:750;letter-spacing:.045em;text-transform:uppercase;line-height:1.2;">
              Search Hugging Face
            </div>
            """,
            sizing_mode="stretch_width",
            height=18,
            margin=(0, 0, 3, 0),
        )

        self.search_options_panel = pn.Column(
            pn.Row(
                self.task,
                self.sort,
                self.limit,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
                styles={"gap": "8px", "align-items": "end"},
            ),
            self.token,
            sizing_mode="stretch_width",
            visible=False,
            margin=(8, 0, 0, 0),
            styles={
                **SUBSURFACE_STYLES,
                "padding": "10px",
                "overflow": "visible",
            },
        )

        search_controls = pn.Column(
            search_label,
            pn.Row(
                self.query,
                self.search_button,
                self.search_options_toggle,
                sizing_mode="stretch_width",
                height=38,
                margin=(0, 0, 0, 0),
                styles={
                    "box-sizing": "border-box",
                    "gap": "8px",
                    "align-items": "center",
                    "overflow": "visible",
                    "min-width": "0",
                },
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box",
                "display": "flex",
                "gap": "4px",
                "min-width": "0",
                "overflow": "visible",
            },
        )

        self.progress_mount = pn.Column(
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box",
                "display": "flex",
                "gap": "4px",
                "min-width": "0",
                "overflow": "hidden",
            },
        )

        toolbar = pn.Column(
            pn.Row(
                heading,
                self.status,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
                styles={
                    "box-sizing": "border-box",
                    "gap": "14px",
                    "align-items": "flex-start",
                    "min-width": "0",
                    "overflow": "visible",
                },
            ),
            search_controls,
            self.search_options_panel,
            self.progress_mount,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                **TOOLBAR_STYLES,
                "display": "flex",
                "gap": "10px",
                "overflow": "visible",
            },
        )

        results_header = pn.Row(
            pn.pane.HTML(
                """
                <div>
                  <div style="color:#263244;font-size:13px;font-weight:720;line-height:1.25;">Search results</div>
                  <div style="margin-top:2px;color:#687386;font-size:10.5px;line-height:1.35;">
                    Select a dataset to inspect it automatically.
                  </div>
                </div>
                """,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            ),
            self.results_meta,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
            styles={"gap": "8px", "align-items": "center"},
        )

        results_surface = pn.Column(
            results_header,
            self.results_list,
            sizing_mode="stretch_width",
            width=300,
            min_width=260,
            max_width=360,
            margin=(0, 0, 0, 0),
            styles={
                **SURFACE_STYLES,
                "overflow": "visible",
            },
        )

        self.empty_state = pn.pane.HTML(
            self._empty_state_html(),
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
            styles=EMPTY_STATE_STYLES,
        )

        selected_header = pn.Column(
            self.selected_pane,
            pn.Row(
                pn.layout.HSpacer(),
                self.inspect_button,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
                styles={"min-height": "34px", "align-items": "center"},
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={"gap": "8px", "min-width": "0"},
        )

        split_section = pn.Column(
            self._section_heading(
                "Splits",
                "Detected splits are selected automatically. Clear any split you do not need.",
            ),
            self.config_select,
            self.split_choices,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles=SUBSURFACE_STYLES,
        )

        self.preview_result_mount = pn.Column(
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box",
                "display": "block",
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "overflow": "visible",
                "position": "relative",
            },
        )

        preview_section = pn.Column(
            self._section_heading(
                "Preview",
                "Preview only downloads or resolves the displayed examples.",
            ),
            self.preview_split,
            self.preview_limit,
            self.preview_button,
            self.preview_result_mount,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                **SUBSURFACE_STYLES,
                "display": "flex",
                "flex-direction": "column",
                "gap": "10px",
                "overflow": "visible",
                "position": "relative",
                "z-index": "0",
            },
        )

        source_options = pn.Column(
            self._advanced_group_heading(
                "Source interpretation",
                "Automatic detection is recommended. Override these values only for unusual repositories.",
            ),
            self.strategy,
            self.trust_remote_code,
            self.image_column,
            self.label_column,
            self.id_column,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box",
                "display": "flex",
                "gap": "10px",
                "min-width": "0",
                "overflow": "visible",
            },
        )

        registration_options = pn.Column(
            self._advanced_group_heading(
                "AstronomicAL registration",
                "Generated names remain tied to the repository and configuration selected for this import.",
            ),
            self.dataset_id,
            self.dataset_name,
            self.max_rows,
            self.write_parquet,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box",
                "display": "flex",
                "gap": "10px",
                "min-width": "0",
                "overflow": "visible",
            },
        )

        self.advanced_panel = pn.Column(
            source_options,
            pn.pane.HTML(
                '<div style="height:1px;background:#dfe5ed;width:100%;"></div>',
                height=1,
                sizing_mode="stretch_width",
                margin=(2, 0, 2, 0),
            ),
            registration_options,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                **SUBSURFACE_STYLES,
                "display": "flex",
                "gap": "12px",
                "padding": "12px",
                "background": "#f7f9fb",
                "overflow-x": "hidden",
                "overflow-y": "visible",
            },
        )
        self.advanced_mount = pn.Column(
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box",
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "overflow": "visible",
            },
        )

        import_section = pn.Column(
            self._section_heading(
                "Import into AstronomicAL",
                "Existing registrations are reused. Missing files are downloaded only when needed.",
            ),
            self.active_after_import,
            self.import_button,
            pn.Row(
                self.advanced_toggle,
                pn.layout.HSpacer(),
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            ),
            self.advanced_mount,
            self.import_summary,
            self.import_preview,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                **SUBSURFACE_STYLES,
                "background": "#ffffff",
                "position": "relative",
                "z-index": "0",
                "overflow": "visible",
            },
        )

        self.dataset_details = pn.Column(
            selected_header,
            self.cache_pane,
            self.notice_pane,
            self.structure_pane,
            split_section,
            preview_section,
            import_section,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                "display": "flex",
                "gap": "10px",
                "min-width": "0",
                "overflow": "visible",
            },
        )

        self.inspector_body = pn.Column(
            self.empty_state,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={"min-width": "0", "overflow": "visible"},
        )

        inspector_surface = pn.Column(
            self.inspector_body,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(0, 0, 0, 0),
            styles=INSPECTOR_STYLES,
        )

        workbench = pn.Row(
            results_surface,
            inspector_surface,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box",
                "min-width": "0",
                "gap": "10px",
                "align-items": "flex-start",
                "overflow": "visible",
            },
        )

        shell = pn.Column(
            toolbar,
            workbench,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                **SHELL_STYLES,
                "gap": "10px",
            },
        )

        return pn.Column(
            shell,
            sizing_mode="stretch_width",
            css_classes=["al-hf-root-v3"],
            margin=(0, 0, 0, 0),
            styles=ROOT_STYLES,
        )
    @staticmethod
    def _advanced_group_heading(title: str, copy: str) -> pn.pane.HTML:
        return pn.pane.HTML(
            f"""
            <div style="min-width:0;max-width:100%;">
              <div style="color:#263244;font-size:11px;font-weight:720;line-height:1.3;overflow-wrap:anywhere;">{html.escape(title)}</div>
              <div style="margin-top:2px;color:#687386;font-size:9.8px;line-height:1.4;overflow-wrap:anywhere;">{html.escape(copy)}</div>
            </div>
            """,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

    @staticmethod
    def _section_heading(title: str, copy: str) -> pn.pane.HTML:
        return pn.pane.HTML(
            f"""
            <div style="padding-bottom:8px;border-bottom:1px solid #e2e7ee;">
              <div style="color:#263244;font-size:12px;font-weight:720;line-height:1.25;">{html.escape(title)}</div>
              <div style="margin-top:2px;color:#687386;font-size:10px;line-height:1.4;">{html.escape(copy)}</div>
            </div>
            """,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
    @property
    def selected_repo(self) -> str:
        return str(getattr(self, "_selected_repo", "") or "").strip()

    def _on_search_clicked(self, _event: Any = None) -> None:
        query = str(self.query.value or "").strip()
        if not query:
            self._set_status("Enter a search term or dataset ID.", "warning")
            return

        if "/" in query:
            cache = self._service().cache_status(query)
            if cache.cached:
                self._select_repo(query, inspect=True)
                self._set_status(
                    "Local Hugging Face cache detected. Inspecting the exact "
                    "dataset ID before contacting the Hub for missing metadata.",
                    "info",
                )
                return

        self._run_job(
            self._search,
            title="Searching Hugging Face datasets",
            key=(
                f"hf.search:{query}:{self.task.value}:"
                f"{self.sort.value}:{self.limit.value}"
            ),
            on_done=self._on_search_done,
            on_error=self._on_error,
        )

    def _search(self, *, cancel_token: Any = None) -> pd.DataFrame:
        del cancel_token
        return self._service().search_dataframe(
            query=str(self.query.value or "").strip(),
            task=str(self.task.value or ""),
            limit=int(self.limit.value or 30),
            sort=str(self.sort.value or "downloads"),
            token=self._token(),
        )

    def _on_search_done(self, frame: pd.DataFrame) -> None:
        self._set_busy(False)
        self._finish_progress("Search complete.")

        self._search_df = frame if frame is not None else pd.DataFrame()

        if self._search_df.empty or "repo_id" not in self._search_df.columns:
            self.results_meta.object = self._results_meta_html("No matches")
            self.results_list[:] = [
                self._results_empty_html("No matching datasets were found.")
            ]
            exact = str(self.query.value or "").strip()
            if "/" in exact:
                self._select_repo(exact, inspect=True)
                self._set_status(
                    "No search row was returned. Inspecting the exact dataset ID directly.",
                    "info",
                )
                return

            self._set_status("No matching datasets were found.", "warning")
            return

        repo_ids = [
            str(value)
            for value in self._search_df["repo_id"].tolist()
            if str(value).strip()
        ]
        cached_count = 0
        if "cached" in self._search_df.columns:
            try:
                cached_count = sum(
                    1
                    for value in self._search_df["cached"].tolist()
                    if self._truthy(value)
                )
            except Exception:
                cached_count = 0

        label = f"{len(repo_ids)} datasets"
        if cached_count:
            label += f" · {cached_count} cached"
        self.results_meta.object = self._results_meta_html(label)
        self._render_search_results()

        exact = str(self.query.value or "").strip().casefold()
        exact_repo = next(
            (
                repo_id
                for repo_id in repo_ids
                if repo_id.casefold() == exact
            ),
            None,
        )

        if exact_repo is not None:
            self._select_repo(exact_repo, inspect=True)
            self._set_status(
                f"Found the exact dataset {exact_repo}.",
                "success",
            )
            return

        self._set_status(
            f"Found {len(repo_ids)} datasets. Select one to inspect it.",
            "success",
        )
    def _render_search_results(self) -> None:
        self._result_items = {}
        items: list[Any] = []

        for _index, row in self._search_df.iterrows():
            repo_id = str(row.get("repo_id") or "").strip()
            if not repo_id:
                continue

            button = pn.widgets.Button(
                name=repo_id,
                button_type="default",
                sizing_mode="stretch_width",
                height=24,
                margin=(0, 0, 0, 0),
                stylesheets=[RESULT_BUTTON_STYLESHEET],
            )
            button.description = f"Inspect {repo_id}"
            button.on_click(
                lambda _event, selected_repo=repo_id: self._select_repo(
                    selected_repo,
                    inspect=True,
                )
            )

            meta = pn.pane.HTML(
                self._result_meta_html(row),
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )
            item = pn.Column(
                button,
                meta,
                sizing_mode="stretch_width",
                margin=(0, 0, 6, 0),
                styles=self._result_item_styles(selected=False),
            )
            self._result_items[repo_id] = item
            items.append(item)

        if not items:
            items = [self._results_empty_html("No usable dataset rows were returned.")]

        self.results_list[:] = items
        self._set_selected_result(self.selected_repo)

    def _set_selected_result(self, repo_id: str) -> None:
        selected_repo = str(repo_id or "")
        for item_repo, item in self._result_items.items():
            try:
                item.styles = self._result_item_styles(
                    selected=item_repo == selected_repo
                )
            except Exception:
                pass

    @staticmethod
    def _result_item_styles(*, selected: bool) -> dict[str, str]:
        if selected:
            return {
                "box-sizing": "border-box",
                "padding": "8px 10px",
                "border": "1px solid #78acd0",
                "border-radius": "8px",
                "background": "#eef6fc",
                "box-shadow": "0 0 0 1px rgba(15, 111, 189, 0.06)",
                "overflow": "hidden",
            }
        return {
            "box-sizing": "border-box",
            "padding": "8px 10px",
            "border": "1px solid #e2e7ee",
            "border-radius": "8px",
            "background": "#ffffff",
            "overflow": "hidden",
        }

    @staticmethod
    def _result_meta_html(row: Any) -> str:
        cached = HuggingFaceBrowserPanel._truthy(row.get("cached"))
        downloads = HuggingFaceBrowserPanel._format_count(row.get("downloads"))
        likes = HuggingFaceBrowserPanel._format_count(row.get("likes"))
        modified = str(row.get("last_modified") or "").strip()
        if len(modified) > 10:
            modified = modified[:10]

        cache_badge = (
            "<span style='display:inline-flex;align-items:center;padding:2px 7px;border-radius:999px;"
            "background:#ecf8f1;color:#1f7a4d;font-size:9.5px;font-weight:700;'>Cached</span>"
            if cached
            else ""
        )
        gated_badge = (
            "<span style='display:inline-flex;align-items:center;padding:2px 7px;border-radius:999px;"
            "background:#fff7d6;color:#9a6700;font-size:9.5px;font-weight:700;'>Gated</span>"
            if HuggingFaceBrowserPanel._truthy(row.get("gated"))
            else ""
        )
        stats = []
        if downloads:
            stats.append(f"{downloads} downloads")
        if likes:
            stats.append(f"{likes} likes")
        if modified:
            stats.append(f"updated {html.escape(modified)}")
        stats_text = " · ".join(stats) or "Dataset repository"

        return (
            "<div style='display:flex;align-items:center;justify-content:space-between;gap:8px;min-width:0;'>"
            f"<div style='min-width:0;color:#687386;font-size:9.8px;line-height:1.35;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{stats_text}</div>"
            f"<div style='display:flex;gap:4px;flex:0 0 auto;'>{cache_badge}{gated_badge}</div>"
            "</div>"
        )

    @staticmethod
    def _format_count(value: Any) -> str:
        try:
            number = int(value or 0)
        except Exception:
            return ""
        if number >= 1_000_000:
            return f"{number / 1_000_000:.1f}m".rstrip("0").rstrip(".")
        if number >= 1_000:
            return f"{number / 1_000:.1f}k".rstrip("0").rstrip(".")
        return f"{number:,}" if number else ""

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value or "").strip().casefold() in {
            "1",
            "true",
            "yes",
            "y",
            "cached",
        }

    @staticmethod
    def _results_meta_html(text: str) -> str:
        return (
            "<div style='display:inline-flex;align-items:center;justify-content:center;"
            "min-height:24px;padding:3px 8px;border:1px solid #d8dee8;border-radius:999px;"
            "background:#f7f8fa;color:#687386;font-size:9.8px;font-weight:700;white-space:nowrap;'>"
            f"{html.escape(str(text))}</div>"
        )

    @staticmethod
    def _results_empty_html(message: str) -> pn.pane.HTML:
        return pn.pane.HTML(
            f"""
            <div style="padding:28px 18px;text-align:center;color:#687386;">
              <div style="font-size:22px;line-height:1;margin-bottom:8px;">⌕</div>
              <div style="font-size:11px;line-height:1.45;">{html.escape(message)}</div>
            </div>
            """,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

    def _on_search_options_toggled(self, event: Any) -> None:
        visible = bool(event.new)
        if hasattr(self, "search_options_panel"):
            self.search_options_panel.visible = visible
        self.search_options_toggle.name = (
            "Hide options" if visible else "Search options"
        )

    def _on_advanced_options_toggled(self, event: Any) -> None:
        visible = bool(event.new)

        # Mount and unmount the advanced form instead of relying on ``visible``.
        # Some Panel/Bokeh layout combinations keep a styled hidden Column in the
        # document flow, which made the form appear permanently open.
        if hasattr(self, "advanced_mount"):
            self.advanced_mount.objects = (
                [self.advanced_panel]
                if visible
                else []
            )

        self.advanced_toggle.name = (
            "Hide advanced" if visible else "Advanced options"
        )
    def _select_repo(self, repo_id: str, *, inspect: bool) -> None:
        repo_id = str(repo_id or "").strip()
        if not repo_id:
            return

        self._inspection_generation += 1

        self._selected_repo = repo_id
        self._set_selected_result(repo_id)
        self.selected_pane.object = self._selected_html(repo_id)
        self.inspect_button.disabled = False
        self.inspector_body[:] = [self.dataset_details]

        # A repository selection is a hard state boundary. Nothing derived from
        # the previous repository may remain visible while the new inspection is
        # pending, including configs, splits, inferred columns, previews, and
        # import results.
        self._clear_selected_dataset_details()
        self._refresh_cache_summary()
        self._suggest_names()

        if not inspect:
            return

        if self._busy:
            self._pending_inspection_repo = repo_id
            self._set_status(
                "Dataset selection changed. Previous details were cleared; "
                "inspection will begin when the current task finishes.",
                "info",
            )
            return

        self._pending_inspection_repo = None
        self._start_inspection()

    def _clear_selected_dataset_details(self) -> None:
        self._details = None
        self._details_mode = None
        self._details_config = None

        self._updating_widgets = True
        try:
            self.config_select.options = ["default"]
            self.config_select.value = "default"
            self.config_select.visible = False

            self.split_choices.options = []
            self.split_choices.value = []

            self.preview_split.options = []
            self.preview_split.value = None

            self.active_after_import.options = OrderedDict(
                {"Do not change active dataset": ""}
            )
            self.active_after_import.value = ""

            self.image_column.options = []
            self.image_column.value = None
            self.label_column.options = OrderedDict({"No label": ""})
            self.label_column.value = ""
            self.id_column.options = OrderedDict({"Generate IDs": ""})
            self.id_column.value = ""
        finally:
            self._updating_widgets = False

        self.structure_pane.object = self._structure_placeholder_html()
        self.notice_pane.object = ""
        self.preview_grid.objects = []
        if hasattr(self, "preview_result_mount"):
            self.preview_result_mount.objects = []
        self.import_summary.object = ""
        self.import_preview.object = ""
        self.import_preview.visible = False

        self.preview_button.disabled = True
        self.import_button.disabled = True
        self.import_button.name = "Import selected splits"
        self.import_button.icon = "cloud-download"

    def _on_inspect_clicked(self, _event: Any = None) -> None:
        self._start_inspection()

    def _start_inspection(self, *, config_name: str | None = None) -> None:
        repo_id = self.selected_repo
        if not repo_id:
            self._set_status("Select a dataset first.", "warning")
            return

        generation = self._inspection_generation
        selected_config = config_name
        if selected_config is None and self.config_select.visible:
            selected_config = self._normalise_config(self.config_select.value)

        desired = str(self.strategy.value or "auto")
        token = self._token()
        trust_remote_code = bool(self.trust_remote_code.value)

        self._run_job(
            lambda cancel_token=None: self._inspect(
                repo_id=repo_id,
                desired=desired,
                token=token,
                trust_remote_code=trust_remote_code,
                config_name=selected_config,
                cancel_token=cancel_token,
            ),
            title=f"Inspecting {repo_id}",
            key=f"hf.inspect:{repo_id}:{selected_config or 'default'}:{desired}",
            on_done=lambda payload: self._on_inspect_done(
                payload,
                expected_repo=repo_id,
                generation=generation,
            ),
            on_error=lambda exc: self._on_inspect_error(
                exc,
                expected_repo=repo_id,
                generation=generation,
            ),
        )

    def _inspect(
        self,
        *,
        repo_id: str,
        desired: str,
        token: str | None,
        trust_remote_code: bool,
        config_name: str | None,
        cancel_token: Any = None,
    ) -> dict[str, Any]:
        del cancel_token
        service = self._service()

        configs: list[str] = []
        try:
            configs = service.get_config_names_builder(
                repo_id,
                token=token,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            configs = []

        if config_name is None and configs:
            config_name = (
                "default"
                if "default" in configs
                else configs[0]
            )

        builder_details: HFDatasetDetails | None = None
        builder_error: BaseException | None = None
        if desired in {"auto", "builder"}:
            try:
                builder_details = service.get_dataset_details_builder(
                    repo_id,
                    config_name=self._normalise_config(config_name),
                    token=token,
                    trust_remote_code=trust_remote_code,
                )
            except BaseException as exc:
                builder_error = exc

        file_details: HFDatasetDetails | None = None
        file_error: BaseException | None = None
        builder_usable = self._builder_details_usable(
            service,
            builder_details,
        )
        if desired == "files" or builder_details is None or (
            desired == "auto" and not builder_usable
        ):
            try:
                file_details = service.get_dataset_details(
                    repo_id,
                    token=token,
                    trust_remote_code=trust_remote_code,
                )
            except BaseException as exc:
                file_error = exc

        if desired == "builder":
            if builder_details is None:
                raise RuntimeError(
                    "Structured dataset inspection failed: "
                    + str(builder_error or "unknown error")
                )
            mode = "builder"
            details = builder_details
        elif desired == "files":
            if file_details is None:
                raise RuntimeError(
                    "Image-file inspection failed: "
                    + str(file_error or "unknown error")
                )
            mode = "files"
            details = file_details
        else:
            mode, details = self._choose_automatic_strategy(
                service,
                builder_details=builder_details,
                file_details=file_details,
            )

        if not configs:
            configs = list(details.configs or [])
        if not configs:
            configs = ["default"]

        selected_config = (
            str(config_name)
            if config_name not in (None, "")
            else str(details.configs[0] if details.configs else "default")
        )

        cache = service.cache_status(repo_id)

        return {
            "repo_id": repo_id,
            "mode": mode,
            "details": details,
            "configs": configs,
            "config": selected_config,
            "cache": cache,
            "builder_error": str(builder_error or ""),
            "file_error": str(file_error or ""),
        }

    @staticmethod
    def _builder_details_usable(
        service: HuggingFaceDatasetService,
        details: HFDatasetDetails | None,
    ) -> bool:
        if details is None:
            return False

        config = str(details.configs[0] if details.configs else "default")
        features = dict(
            details.features_by_config.get(config)
            or details.features_by_config.get("default")
            or {}
        )
        image_column, _label_column, _id_column = service.choose_builder_columns(
            details,
            config=config,
        )
        fallback_only = (
            len(features) == 1
            and str(features.get("image", "")).lower().endswith("fallback")
        )
        return bool(image_column and not fallback_only)

    @staticmethod
    def _choose_automatic_strategy(
        service: HuggingFaceDatasetService,
        *,
        builder_details: HFDatasetDetails | None,
        file_details: HFDatasetDetails | None,
    ) -> tuple[str, HFDatasetDetails]:
        if HuggingFaceBrowserPanel._builder_details_usable(
            service,
            builder_details,
        ):
            # Structured rows are preferred even without labels because they
            # preserve declared configs and split names. Label inference then
            # falls back to the selected column's raw value.
            return "builder", builder_details

        if file_details is not None and (
            file_details.sample_files
            or int(file_details.scanned_file_count or 0) > 0
        ):
            return "files", file_details

        if builder_details is not None:
            return "builder", builder_details
        if file_details is not None:
            return "files", file_details

        raise RuntimeError(
            "AstronomicAL could not inspect the dataset through either "
            "structured rows or repository image files."
        )

    def _on_inspect_done(
        self,
        payload: dict[str, Any],
        *,
        expected_repo: str,
        generation: int,
    ) -> None:
        if not self._inspection_is_current(expected_repo, generation):
            self._finish_stale_inspection()
            return

        self._pending_inspection_repo = None
        self._set_busy(False)
        self._finish_progress("Inspection complete.")

        self._details_mode = str(payload.get("mode") or "builder")
        self._details = payload.get("details")
        self._details_config = str(payload.get("config") or "default")

        if self._details is None:
            self._set_status("Inspection returned no dataset details.", "danger")
            return

        configs = [
            str(value)
            for value in (payload.get("configs") or ["default"])
        ]
        if not configs:
            configs = ["default"]

        self._updating_widgets = True
        try:
            self.config_select.options = configs
            selected_config = (
                self._details_config
                if self._details_config in configs
                else configs[0]
            )
            self.config_select.value = selected_config
            self.config_select.visible = len(configs) > 1
            self._apply_details_to_widgets(
                self._details,
                mode=self._details_mode,
                config=selected_config,
            )
        finally:
            self._updating_widgets = False

        cache = payload.get("cache")
        self.cache_pane.object = self._cache_html(cache)
        self._refresh_structure_summary()
        self._refresh_existing_registration_summary()
        self._suggest_names()

        mode_label = self._mode_label(self._details_mode)
        self._set_status(
            f"Inspection complete. AstronomicAL selected {html.escape(mode_label)}.",
            "success",
        )

    def _on_inspect_error(
        self,
        exc: BaseException,
        *,
        expected_repo: str,
        generation: int,
    ) -> None:
        if not self._inspection_is_current(expected_repo, generation):
            self._finish_stale_inspection()
            return
        self._pending_inspection_repo = None
        self._on_error(exc)

    def _inspection_is_current(
        self,
        expected_repo: str,
        generation: int,
    ) -> bool:
        return (
            generation == self._inspection_generation
            and expected_repo == self.selected_repo
        )

    def _finish_stale_inspection(self) -> None:
        self._set_busy(False)
        self._finish_progress("Selection changed.")
        self._start_pending_inspection_if_ready()

    def _start_pending_inspection_if_ready(self) -> None:
        if self._busy:
            return

        pending_repo = str(self._pending_inspection_repo or "").strip()
        if not pending_repo or pending_repo != self.selected_repo:
            self._pending_inspection_repo = None
            return

        self._pending_inspection_repo = None
        self._start_inspection()

    def _apply_details_to_widgets(
        self,
        details: HFDatasetDetails,
        *,
        mode: str,
        config: str,
    ) -> None:
        if mode == "builder":
            service = self._service()
            image, label, record_id = service.choose_builder_columns(
                details,
                config=config,
            )
            features = dict(
                details.features_by_config.get(config)
                or details.features_by_config.get("default")
                or {}
            )
            columns = list(features)

            image_options = columns or list(details.image_column_candidates or [])
            self._set_select_options(
                self.image_column,
                image_options,
                preferred=image or (image_options[0] if image_options else None),
            )
            self._set_select_options(
                self.label_column,
                OrderedDict(
                    [("No label", "")]
                    + [(column, column) for column in columns]
                ),
                preferred=label or "",
            )
            self._set_select_options(
                self.id_column,
                OrderedDict(
                    [("Generate IDs", "")]
                    + [(column, column) for column in columns]
                ),
                preferred=record_id or "",
            )
        else:
            self._set_select_options(
                self.image_column,
                OrderedDict({"Repository image files": "__hf_file__"}),
                preferred="__hf_file__",
            )
            self._set_select_options(
                self.label_column,
                OrderedDict(
                    {
                        "Infer from folders and filenames": "__hf_path_label__",
                        "No label": "",
                    }
                ),
                preferred="__hf_path_label__",
            )
            self._set_select_options(
                self.id_column,
                OrderedDict({"Generate IDs": ""}),
                preferred="",
            )

        splits_by_config = details.splits_by_config or {}
        splits = list(
            splits_by_config.get(config)
            or splits_by_config.get("default")
            or []
        )
        if not splits and mode == "files":
            splits = sorted(
                {
                    str(file.split or "train")
                    for file in details.sample_files
                }
            )
        if not splits:
            splits = ["train"]

        restored_splits = [
            str(value)
            for value in (self._restored_state.get("splits") or [])
            if str(value) in splits
        ]
        if not restored_splits:
            legacy_split = str(self._restored_state.get("split") or "")
            legacy_all = bool(self._restored_state.get("download_all_splits", False))
            if legacy_all:
                restored_splits = list(splits)
            elif legacy_split in splits:
                restored_splits = [legacy_split]
        selected_splits = restored_splits or list(splits)

        self.split_choices.options = splits
        self.split_choices.value = selected_splits

        self.preview_split.options = splits
        restored_preview = str(
            self._restored_state.get("preview_split") or ""
        )
        self.preview_split.value = (
            restored_preview
            if restored_preview in splits
            else splits[0]
        )
        self.preview_button.disabled = False
        self._update_active_after_import()
        self.import_button.disabled = not bool(selected_splits)

    def _on_config_changed(self, event: Any) -> None:
        if self._updating_widgets or self._details is None:
            return

        value = self._normalise_config(event.new)
        if value == self._normalise_config(self._details_config):
            return
        self._start_inspection(config_name=value)

    def _on_strategy_changed(self, _event: Any) -> None:
        if self._updating_widgets or not self.selected_repo:
            return
        if self._details is not None:
            self._start_inspection(
                config_name=self._normalise_config(self.config_select.value),
            )

    def _on_split_choices_changed(self, _event: Any) -> None:
        if self._updating_widgets:
            return
        self._update_active_after_import()
        self.import_button.disabled = not bool(self.split_choices.value)
        self._refresh_existing_registration_summary()
        self._suggest_names()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _on_preview_clicked(self, _event: Any = None) -> None:
        if self._details is None:
            self._set_status("Inspect the dataset before previewing it.", "warning")
            return

        request = {
            "repo_id": str(self.selected_repo),
            "mode": str(self._details_mode or "builder"),
            "config_name": self._normalise_config(self.config_select.value),
            "split": str(self.preview_split.value or "train"),
            "image_column": self._empty_to_none(self.image_column.value),
            "label_column": self._empty_to_none(self.label_column.value),
            "id_column": self._empty_to_none(self.id_column.value),
            "limit": int(self.preview_limit.value or 10),
            "token": self._token(),
            "trust_remote_code": bool(self.trust_remote_code.value),
        }
        self._pending_preview_request = request
        self._run_job(
            self._preview,
            title=f"Previewing {request['repo_id']} / {request['split']}",
            key=(
                f"hf.preview:{request['repo_id']}:{request['mode']}:"
                f"{request['config_name']}:{request['split']}:{request['limit']}"
            ),
            on_done=self._on_preview_done,
            on_error=self._on_error,
        )

    def _preview(self, *, cancel_token: Any = None) -> dict[str, Any]:
        del cancel_token
        request = dict(getattr(self, "_pending_preview_request", {}) or {})
        if not request:
            raise RuntimeError("Preview request was not captured.")
        service = self._service()
        if request["mode"] == "files":
            return service.preview_image_items(
                repo_id=request["repo_id"],
                split=request["split"],
                limit=request["limit"],
                token=request["token"],
                thumb_size=180,
            )
        return service.preview_image_items_builder(
            repo_id=request["repo_id"],
            config_name=request["config_name"],
            split=request["split"],
            image_column=request["image_column"],
            label_column=request["label_column"],
            id_column=request["id_column"],
            limit=request["limit"],
            token=request["token"],
            trust_remote_code=request["trust_remote_code"],
            thumb_size=180,
        )

    def _on_preview_done(self, result: dict[str, Any]) -> None:
        self._set_busy(False)
        self._finish_progress("Preview ready.")
        result = dict(result or {})
        items = list(result.get("items", []) or [])
        note = str(result.get("note", "") or "")
        cache_source = str(result.get("cache_source", "") or "")

        objects: list[Any] = []
        if note or cache_source:
            detail = note
            if cache_source:
                detail = f"{detail} Source: {cache_source}.".strip()
            objects.append(
                pn.pane.HTML(
                    f'<div class="al-hf-preview-note">{html.escape(detail)}</div>',
                    sizing_mode="stretch_width",
                    margin=(0, 0, 0, 0),
                )
            )
        for item in items:
            objects.append(
                pn.pane.HTML(
                    self._preview_card_html(item),
                    sizing_mode="stretch_width",
                    height=260,
                    min_height=260,
                    margin=(0, 0, 0, 0),
                    css_classes=["al-hf-preview-card-pane"],
                )
            )
        if not items:
            objects.append(
                pn.pane.HTML(
                    '<div class="al-hf-preview-empty">No preview rows were returned.</div>',
                    sizing_mode="stretch_width",
                    height=62,
                    margin=(0, 0, 0, 0),
                )
            )
        self.preview_grid.objects = objects
        self.preview_result_mount.objects = [self.preview_grid]
        self._set_status(
            f"Preview ready: {len(items)} image(s). {cache_source or 'Local caches were checked first.'}",
            "success",
        )

    @staticmethod
    def _preview_card_html(item: dict[str, Any]) -> str:
        data_uri = str(item.get("data_uri", "") or "")
        record_id = html.escape(str(item.get("record_id", "") or ""))
        label = html.escape(str(item.get("label", "") or ""))
        image_column = html.escape(str(item.get("image_column", "") or ""))
        label_column = html.escape(str(item.get("label_column", "") or ""))
        path = html.escape(str(item.get("path", "") or ""))
        if data_uri:
            image_html = (
                f'<img src="{data_uri}" alt="{record_id}" '
                'style="display:block;width:100%;height:180px;object-fit:contain;background:#111;" />'
            )
        else:
            image_html = (
                '<div style="display:flex;align-items:center;justify-content:center;height:180px;'
                'background:#eef2f6;color:#687386;font-size:11px;">Preview unavailable</div>'
            )
        path_html = (
            f'<div class="al-hf-preview-meta" title="{path}">{path}</div>'
            if path else ""
        )
        return (
            '<div class="al-hf-preview-card">'
            + image_html
            + '<div class="al-hf-preview-card-copy">'
            + f'<div class="al-hf-preview-record">{record_id}</div>'
            + f'<div class="al-hf-preview-label">{label}</div>'
            + f'<div class="al-hf-preview-meta">image: {image_column}</div>'
            + f'<div class="al-hf-preview-meta">label: {label_column}</div>'
            + path_html
            + '</div></div>'
        )

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def _on_import_clicked(self, _event: Any = None) -> None:
        if self._details is None:
            self._set_status("Inspect the dataset before importing it.", "warning")
            return

        repo_id = self.selected_repo
        if not repo_id:
            self._set_status("Select a dataset before importing it.", "warning")
            return

        splits = tuple(
            str(value)
            for value in (self.split_choices.value or [])
            if str(value).strip()
        )
        if not splits:
            self._set_status("Select at least one split to import.", "warning")
            return

        config = self._normalise_config(self.config_select.value)
        expected_id = self._suggested_base_id_for(repo_id, config)
        expected_name = self._suggested_base_name_for(repo_id, config)

        base_id_value = str(self.dataset_id.value or "").strip()
        base_name_value = str(self.dataset_name.value or "").strip()
        base_id = (
            expected_id
            if not base_id_value
            or base_id_value == self._last_suggested_dataset_id
            else base_id_value
        )
        base_name = (
            expected_name
            if not base_name_value
            or base_name_value == self._last_suggested_dataset_name
            else base_name_value
        )

        request = {
            "repo_id": repo_id,
            "config": config,
            "splits": splits,
            "active_split": str(self.active_after_import.value or ""),
            "base_id": base_id,
            "base_name": base_name,
            "image_column": self._resolved_image_column(),
            "label_column": self._resolved_label_column(),
            "id_column": self._empty_to_none(self.id_column.value),
            "max_rows": int(self.max_rows.value or 0),
            "token": self._token(),
            "trust_remote_code": bool(self.trust_remote_code.value),
            "write_parquet": bool(self.write_parquet.value),
        }

        self._run_job(
            lambda cancel_token=None: self._import_selected_splits(
                request=request,
                cancel_token=cancel_token,
            ),
            title=f"Importing {repo_id}",
            key=(
                f"hf.import:{repo_id}:{config or 'default'}:"
                f"{','.join(splits)}:{request['max_rows']}"
            ),
            on_done=self._on_import_done,
            on_error=self._on_error,
        )

    def _import_selected_splits(
        self,
        *,
        request: dict[str, Any],
        cancel_token: Any = None,
    ) -> dict[str, Any]:
        repo_id = str(request["repo_id"])
        splits = tuple(str(value) for value in request["splits"])
        config = self._normalise_config(request.get("config"))
        active_split = str(request.get("active_split") or "")
        base_id = str(request.get("base_id") or "").strip()
        base_name = str(request.get("base_name") or "").strip()

        results: list[dict[str, Any]] = []
        previews: list[pd.DataFrame] = []
        total_rows = 0

        for index, split in enumerate(splits):
            if cancel_token is not None and cancel_token.cancelled():
                return {
                    "cancelled": True,
                    "results": results,
                    "rows": total_rows,
                }

            existing_id = find_registered_hf_dataset(
                self.context,
                repo_id=repo_id,
                config_name=config,
                split=split,
            )
            if existing_id is not None:
                if active_split == split:
                    self.context.datasets.set_active(
                        existing_id,
                        origin="integrations.huggingface",
                    )
                result = {
                    "dataset_id": existing_id,
                    "name": self._dataset_name(existing_id),
                    "rows": self.context.datasets.row_count(existing_id),
                    "backend": self.context.datasets.get_meta(existing_id).get(
                        "backend",
                        "unknown",
                    ),
                    "repo_id": repo_id,
                    "config_name": config or "",
                    "split": split,
                    "existing": True,
                    "cache_reused": True,
                    "download_method": "existing_registration",
                    "preview": self.context.datasets.head(existing_id, n=10),
                }
            else:
                split_dataset_id = slugify(
                    f"{base_id}_{split}",
                    fallback=f"hf_dataset_{index + 1}",
                )
                split_dataset_name = (
                    f"{base_name} / {split}"
                )

                def _split_progress(
                    payload: dict[str, Any],
                    *,
                    split_index: int = index,
                    split_name: str = split,
                ) -> None:
                    payload = dict(payload or {})
                    inner = max(
                        0,
                        min(100, int(payload.get("percent", 0) or 0)),
                    )
                    overall = int(
                        (
                            split_index + (inner / 100.0)
                        )
                        / max(len(splits), 1)
                        * 100
                    )
                    payload["percent"] = overall
                    payload["phase"] = (
                        f"{payload.get('phase', 'import')}:{split_name}"
                    )
                    payload["message"] = (
                        f"Split {split_index + 1}/{len(splits)} "
                        f"({split_name}): {payload.get('message', '')}"
                    )
                    self._set_progress_from_worker(payload)

                result = import_hf_image_dataset_as_manifest(
                    self.context,
                    repo_id=repo_id,
                    config_name=config,
                    split=split,
                    dataset_id=split_dataset_id,
                    dataset_name=split_dataset_name,
                    image_column=request.get("image_column"),
                    label_column=request.get("label_column"),
                    id_column=request.get("id_column"),
                    max_rows=int(request.get("max_rows", 0) or 0),
                    token=request.get("token"),
                    trust_remote_code=bool(request.get("trust_remote_code", False)),
                    write_parquet=bool(request.get("write_parquet", True)),
                    set_active=active_split == split,
                    cancel_token=cancel_token,
                    progress_callback=_split_progress,
                )

            results.append(result)
            total_rows += int(result.get("rows", 0) or 0)
            preview = result.get("preview")
            if preview is not None:
                try:
                    preview_df = preview.copy()
                    preview_df.insert(
                        0,
                        "astronomical_dataset_id",
                        result.get("dataset_id", ""),
                    )
                    preview_df.insert(1, "imported_split", split)
                    previews.append(preview_df.head(10))
                except Exception:
                    pass

        preview = pd.DataFrame(
            [
                {
                    "dataset_id": item.get("dataset_id", ""),
                    "split": item.get("split", ""),
                    "rows": item.get("rows", 0),
                    "backend": item.get("backend", ""),
                    "existing": bool(item.get("existing", False)),
                    "active": item.get("split", "") == active_split,
                }
                for item in results
            ]
        )
        if previews:
            try:
                preview = pd.concat(
                    [preview] + previews,
                    ignore_index=True,
                    sort=False,
                )
            except Exception:
                pass

        self._set_progress_from_worker(
            {
                "phase": "done",
                "completed": len(results),
                "total": len(splits),
                "percent": 100,
                "message": (
                    f"Prepared {len(results)} split dataset(s), "
                    f"{total_rows} total row(s)."
                ),
            }
        )

        return {
            "cancelled": False,
            "repo_id": repo_id,
            "config_name": config or "",
            "results": results,
            "rows": total_rows,
            "split_count": len(results),
            "active_split": active_split,
            "preview": preview,
        }

    def _on_import_done(self, result: dict[str, Any]) -> None:
        self._set_busy(False)

        if result.get("cancelled"):
            self._finish_progress("Import cancelled.")
            self._set_status("Import cancelled.", "warning")
            return

        items = list(result.get("results") or [])
        imported = sum(
            1
            for item in items
            if not bool(item.get("existing", False))
        )
        reused = len(items) - imported
        rows = int(result.get("rows", 0) or 0)
        active_split = str(result.get("active_split") or "")

        lines = []
        for item in items:
            dataset_id = html.escape(str(item.get("dataset_id") or ""))
            split_value = str(item.get("split") or "")
            split = html.escape(split_value)
            row_count = int(item.get("rows", 0) or 0)
            state = "reused" if item.get("existing") else "imported"
            active = " · active" if split_value == active_split else ""
            lines.append(
                "<div style='min-width:0;padding:7px 0;border-top:1px solid #dcebe2;'>"
                f"<code title='{dataset_id}' style='display:block;min-width:0;color:#1f6a45;font-size:9.5px;line-height:1.35;overflow-wrap:anywhere;word-break:break-word;'>{dataset_id}</code>"
                f"<span style='display:block;margin-top:2px;color:#526071;font-size:9.3px;line-height:1.35;overflow-wrap:anywhere;word-break:break-word;'>{split} · {row_count:,} rows · {state}{active}</span>"
                "</div>"
            )

        imported_repo = html.escape(str(result.get("repo_id") or ""))
        self.import_summary.object = f"""
        <div style="box-sizing:border-box;width:100%;max-width:100%;min-width:0;padding:10px;border:1px solid #b8dec9;border-radius:8px;background:#f2faf5;color:#1f6a45;overflow:hidden;">
          <div style="font-size:11px;font-weight:720;line-height:1.3;">Import complete</div>
          <div style="margin-top:2px;color:#39785a;font-size:9.4px;line-height:1.35;overflow-wrap:anywhere;word-break:break-word;">
            Source: <code style="color:#1f6a45;overflow-wrap:anywhere;word-break:break-word;">{imported_repo}</code>
          </div>
          <div style="margin-top:4px;font-size:9.8px;line-height:1.4;overflow-wrap:anywhere;word-break:break-word;">
            Imported {imported} new split dataset(s), reused {reused} existing registration(s),
            and prepared {rows:,} total rows.
          </div>
          <div style="min-width:0;margin-top:6px;overflow:hidden;">{''.join(lines)}</div>
        </div>
        """

        preview = result.get("preview")
        if preview is not None:
            try:
                self.import_preview.object = self._import_preview_html(preview)
                self.import_preview.visible = bool(self.import_preview.object)
            except Exception:
                self.import_preview.object = ""
                self.import_preview.visible = False

        self._finish_progress("Import complete.")
        self._set_status(
            "Import complete. Local caches and existing registrations were reused where available.",
            "success",
        )
        self._refresh_existing_registration_summary()
    def _refresh_cache_summary(self) -> None:
        if not self.selected_repo:
            self.cache_pane.object = ""
            return
        self.cache_pane.object = self._cache_html(
            self._service().cache_status(self.selected_repo)
        )

    @staticmethod
    def _cache_html(cache: Any) -> str:
        if cache is None:
            return ""

        cached = bool(getattr(cache, "cached", False))
        label = html.escape(str(getattr(cache, "label", "Not detected")))
        locations = list(getattr(cache, "locations", []) or [])

        if cached:
            icon = "✓"
            title = "Available locally"
            detail = (
                f"{label}. Cached files are resolved before any network request."
            )
            border = "#b8dec9"
            background = "#f2faf5"
            colour = "#1f6a45"
        else:
            icon = "↓"
            title = "Not cached locally"
            detail = (
                "Preview or import may download files that are not already available."
            )
            border = "#d8dee8"
            background = "#f8fafc"
            colour = "#526071"

        location_html = ""
        if locations:
            location_html = (
                "<div style='margin-top:5px;color:#687386;font-size:9px;line-height:1.35;"
                "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
                + html.escape(str(locations[0]))
                + "</div>"
            )

        return f"""
        <div style="display:flex;align-items:flex-start;gap:9px;padding:9px 10px;border:1px solid {border};border-radius:8px;background:{background};color:{colour};">
          <div style="display:flex;align-items:center;justify-content:center;flex:0 0 22px;width:22px;height:22px;border-radius:50%;background:#ffffff;font-size:11px;font-weight:800;">{icon}</div>
          <div style="min-width:0;">
            <div style="font-size:10.5px;font-weight:720;line-height:1.3;">{title}</div>
            <div style="margin-top:1px;font-size:9.8px;line-height:1.4;">{detail}</div>
            {location_html}
          </div>
        </div>
        """
    def _refresh_structure_summary(self) -> None:
        details = self._details
        if details is None:
            self.structure_pane.object = self._structure_placeholder_html()
            return

        config = str(self.config_select.value or "default")
        splits = list(
            details.splits_by_config.get(config)
            or details.splits_by_config.get("default")
            or self.split_choices.options
            or []
        )
        features = dict(
            details.features_by_config.get(config)
            or details.features_by_config.get("default")
            or {}
        )
        warnings = list(details.warnings or [])

        mode = self._mode_label(self._details_mode)
        image = str(self.image_column.value or "Not detected")
        label = str(self.label_column.value or "No label detected")

        items = [
            ("Import method", mode),
            ("Configuration", config),
            ("Splits", ", ".join(str(value) for value in splits) or "train"),
            ("Image source", image),
            ("Label source", label),
            ("Columns", str(len(features))),
        ]

        rows = "".join(
            f"""
            <div style="display:grid;grid-template-columns:minmax(92px,0.72fr) minmax(0,1.28fr);gap:10px;align-items:start;padding:7px 0;border-top:1px solid #edf1f5;">
              <div style="color:#7a8594;font-size:8.7px;font-weight:750;letter-spacing:.045em;text-transform:uppercase;line-height:1.35;">{html.escape(key)}</div>
              <div style="min-width:0;color:#263244;font-size:10.4px;font-weight:620;line-height:1.4;overflow-wrap:anywhere;word-break:break-word;">{html.escape(value)}</div>
            </div>
            """
            for key, value in items
        )
        self.structure_pane.object = f"""
        <div style="padding:10px 12px;border:1px solid #e2e7ee;border-radius:8px;background:#fbfcfe;overflow:hidden;">
          <div style="padding-bottom:8px;color:#263244;font-size:11.5px;font-weight:720;line-height:1.3;">Detected dataset structure</div>
          <div style="min-width:0;">{rows}</div>
        </div>
        """

        if warnings:
            self.notice_pane.object = self._status_html(
                " · ".join(str(warning) for warning in warnings),
                "warning",
            )
        else:
            self.notice_pane.object = ""
    def _refresh_existing_registration_summary(self) -> None:
        if not self.selected_repo or self._details is None:
            return

        config = self._normalise_config(self.config_select.value)
        splits = [
            str(value)
            for value in (self.split_choices.value or [])
        ]

        existing = []
        for split in splits:
            dataset_id = find_registered_hf_dataset(
                self.context,
                repo_id=self.selected_repo,
                config_name=config,
                split=split,
            )
            if dataset_id:
                existing.append((split, dataset_id))

        if not existing:
            self.import_button.name = "Import selected splits"
            self.import_button.icon = "cloud-download"
            return

        if len(existing) == len(splits) and splits:
            self.import_button.name = "Reuse registered splits"
            self.import_button.icon = "database-check"
        else:
            self.import_button.name = "Import missing and reuse existing"
            self.import_button.icon = "cloud-download"

    # ------------------------------------------------------------------
    # Job and progress helpers
    # ------------------------------------------------------------------

    def _run_job(
        self,
        func: Any,
        *,
        title: str,
        key: str,
        on_done: Any,
        on_error: Any,
    ) -> None:
        if self._busy:
            return

        self._set_busy(True)
        self._set_status(title + "…", "busy")
        self._start_progress(title + "…")
        self._ensure_progress_ticker(True)

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                result = func(cancel_token=None)
                on_done(result)
            except Exception as exc:
                on_error(exc)
            finally:
                self._ensure_progress_ticker(False)
            return

        handle = jobs.submit(
            func,
            title=title,
            key=key,
            on_done=self._job_done_on_ui(on_done),
            on_error=self._job_error_on_ui(on_error),
        )
        self._job_handles.append(handle)

    def _job_done_on_ui(self, on_done: Any):
        def _wrapped(result: Any) -> None:
            def _finish() -> None:
                on_done(result)
                self._sync_progress_ui()
                self._ensure_progress_ticker(False)
                self._start_pending_inspection_if_ready()

            self._schedule_ui(_finish)

        return _wrapped

    def _job_error_on_ui(self, on_error: Any):
        def _wrapped(exc: BaseException) -> None:
            def _finish() -> None:
                on_error(exc)
                self._sync_progress_ui()
                self._ensure_progress_ticker(False)
                self._start_pending_inspection_if_ready()

            self._schedule_ui(_finish)

        return _wrapped

    def _schedule_ui(self, callback: Any, *args: Any, **kwargs: Any) -> None:
        def _run() -> None:
            if self._disposed:
                return
            try:
                callback(*args, **kwargs)
            except Exception:
                traceback.print_exc()

        doc = self._doc or pn.state.curdoc
        if doc is None:
            _run()
            return

        try:
            doc.add_next_tick_callback(_run)
        except Exception:
            _run()

    def _ensure_progress_ticker(self, running: bool) -> None:
        if running:
            if self._progress_periodic is None:
                try:
                    self._progress_periodic = pn.state.add_periodic_callback(
                        self._sync_progress_ui,
                        period=250,
                        start=True,
                    )
                except Exception:
                    self._progress_periodic = None
            return

        try:
            if self._progress_periodic is not None:
                self._progress_periodic.stop()
        except Exception:
            pass
        self._progress_periodic = None

    def _set_progress_from_worker(self, payload: dict[str, Any]) -> None:
        with self._progress_lock:
            self._progress_state.update(
                {
                    "active": True,
                    "phase": str(payload.get("phase", "") or ""),
                    "completed": int(payload.get("completed", 0) or 0),
                    "total": int(payload.get("total", 0) or 0),
                    "percent": int(payload.get("percent", 0) or 0),
                    "message": str(payload.get("message", "") or ""),
                    "current_file": str(payload.get("current_file", "") or ""),
                }
            )

    def _start_progress(self, message: str) -> None:
        with self._progress_lock:
            self._progress_state = {
                "active": True,
                "phase": "starting",
                "completed": 0,
                "total": 0,
                "percent": 0,
                "message": message,
                "current_file": "",
            }

    def _finish_progress(self, message: str) -> None:
        with self._progress_lock:
            self._progress_state.update(
                {
                    "active": False,
                    "phase": "done",
                    "percent": 100,
                    "message": message,
                    "current_file": "",
                }
            )
        self._sync_progress_ui()
    def _fail_progress(self, message: str) -> None:
        with self._progress_lock:
            self._progress_state.update(
                {
                    "active": False,
                    "phase": "error",
                    "message": message,
                    "current_file": "",
                }
            )
        self._sync_progress_ui()
    def _sync_progress_ui(self) -> None:
        if self._disposed:
            return

        with self._progress_lock:
            state = dict(self._progress_state)

        active = bool(state.get("active", False))

        if not active:
            self.progress_bar.visible = False
            self.progress_text.visible = False
            self.progress_text.object = ""
            progress_mount = getattr(self, "progress_mount", None)
            if progress_mount is not None:
                progress_mount[:] = []
            return

        self.progress_bar.visible = True
        self.progress_text.visible = True
        progress_mount = getattr(self, "progress_mount", None)
        if progress_mount is not None:
            progress_mount[:] = [self.progress_bar, self.progress_text]

        percent = max(0, min(100, int(state.get("percent", 0) or 0)))
        completed = int(state.get("completed", 0) or 0)
        total = int(state.get("total", 0) or 0)
        phase = html.escape(str(state.get("phase", "") or "working"))
        message = html.escape(str(state.get("message", "") or ""))
        current_file = str(state.get("current_file", "") or "")

        self.progress_bar.value = percent

        count = f" {completed}/{total}" if total else ""
        file_line = ""
        if current_file:
            short = (
                current_file
                if len(current_file) <= 90
                else "…" + current_file[-89:]
            )
            file_line = (
                "<div style='margin-top:2px;color:#7a8594;font-family:ui-monospace,monospace;"
                f"font-size:8.8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{html.escape(short)}</div>"
            )

        self.progress_text.object = f"""
        <div style="padding:4px 1px 0;color:#526071;font-size:9.8px;line-height:1.35;">
          <strong style="color:#0f6fbd;">{percent}%</strong>
          <span style="color:#7a8594;"> · {phase}{count}</span>
          <span> · {message}</span>
          {file_line}
        </div>
        """
    def _service(self) -> HuggingFaceDatasetService:
        services = getattr(self.context, "services", None)
        if services is not None:
            try:
                service = services.get("integrations.huggingface.client")
                if service is not None:
                    token = self._token()
                    if token:
                        service.token = token
                    return service
            except Exception:
                pass
        return HuggingFaceDatasetService(token=self._token())

    def _resolved_image_column(self) -> str | None:
        if self._details_mode == "files":
            return "__hf_file__"
        return self._empty_to_none(self.image_column.value)

    def _resolved_label_column(self) -> str | None:
        if self._details_mode == "files":
            return "__hf_path_label__"
        return self._empty_to_none(self.label_column.value)

    def _update_active_after_import(self) -> None:
        splits = [
            str(value)
            for value in (self.split_choices.value or [])
            if str(value).strip()
        ]
        options = OrderedDict({"Do not change active dataset": ""})
        for split in splits:
            options[split] = split

        previous = str(self.active_after_import.value or "")
        self.active_after_import.options = options

        restored = str(
            self._restored_state.get("active_after_import")
            or self._restored_state.get("active_split_after_import")
            or ""
        )
        if not restored and bool(self._restored_state.get("set_active", False)):
            restored = str(self._restored_state.get("split") or "")
        if previous in options.values():
            self.active_after_import.value = previous
        elif restored in options.values():
            self.active_after_import.value = restored
        elif splits:
            self.active_after_import.value = splits[0]
        else:
            self.active_after_import.value = ""

    def _suggest_names(self) -> None:
        repo_id = self.selected_repo
        if not repo_id:
            return

        config = self._normalise_config(self.config_select.value)
        suggested_id = self._suggested_base_id_for(repo_id, config)
        suggested_name = self._suggested_base_name_for(repo_id, config)

        current_id = str(self.dataset_id.value or "").strip()
        current_name = str(self.dataset_name.value or "").strip()

        if not current_id or current_id == self._last_suggested_dataset_id:
            self.dataset_id.value = suggested_id
        if not current_name or current_name == self._last_suggested_dataset_name:
            self.dataset_name.value = suggested_name

        self._last_suggested_dataset_id = suggested_id
        self._last_suggested_dataset_name = suggested_name

    def _suggested_base_id(self) -> str:
        return self._suggested_base_id_for(
            self.selected_repo,
            self._normalise_config(self.config_select.value),
        )

    @staticmethod
    def _suggested_base_id_for(repo_id: str, config: str | None) -> str:
        resolved_config = str(config or "default")
        return slugify(
            f"hf_{repo_id}_{resolved_config}",
            fallback="hf_image_dataset",
        )

    def _suggested_base_name(self) -> str:
        return self._suggested_base_name_for(
            self.selected_repo,
            self._normalise_config(self.config_select.value),
        )

    @staticmethod
    def _suggested_base_name_for(repo_id: str, config: str | None) -> str:
        resolved_config = str(config or "default")
        return f"HF {repo_id} [{resolved_config}]"

    @staticmethod
    def _import_preview_html(preview: Any) -> str:
        if preview is None:
            return ""

        try:
            frame = preview if isinstance(preview, pd.DataFrame) else pd.DataFrame(preview)
        except Exception:
            return ""
        if frame.empty:
            return ""

        frame = frame.head(10)
        columns = [str(column) for column in frame.columns]
        header = "".join(
            "<th title='{}' style='padding:6px 7px;border-bottom:1px solid #d8dee8;background:#f7f8fa;color:#526071;font-size:9px;font-weight:750;letter-spacing:.025em;text-align:left;white-space:nowrap;'>"
            "{}</th>".format(html.escape(column), html.escape(column))
            for column in columns
        )

        rows = []
        for _, row in frame.iterrows():
            cells = []
            for column in frame.columns:
                value = row.get(column, "")
                try:
                    is_missing = bool(pd.isna(value))
                except Exception:
                    is_missing = False
                text = "" if is_missing else str(value)
                safe = html.escape(text)
                cells.append(
                    "<td title='{}' style='max-width:190px;padding:6px 7px;border-bottom:1px solid #edf1f5;color:#263244;font-size:9.2px;line-height:1.35;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{}</td>".format(
                        safe,
                        safe,
                    )
                )
            rows.append("<tr>" + "".join(cells) + "</tr>")

        return (
            "<div style='box-sizing:border-box;width:100%;max-width:100%;min-width:0;margin-top:10px;border:1px solid #d8dee8;border-radius:8px;background:#ffffff;overflow:hidden;'>"
            "<div style='padding:8px 10px;border-bottom:1px solid #e2e7ee;color:#263244;font-size:10.5px;font-weight:720;'>Manifest preview</div>"
            "<div style='box-sizing:border-box;width:100%;max-width:100%;overflow-x:auto;overflow-y:hidden;'>"
            "<table style='border-collapse:collapse;width:max-content;min-width:100%;table-layout:auto;'>"
            f"<thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody>"
            "</table></div></div>"
        )

    def _dataset_name(self, dataset_id: str) -> str:
        try:
            dataset = self.context.datasets.get(dataset_id)
            return str(getattr(dataset, "name", None) or dataset_id)
        except Exception:
            return str(dataset_id)

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        for button in (
            self.search_button,
            self.inspect_button,
            self.preview_button,
            self.import_button,
        ):
            button.disabled = bool(busy)

        if not busy:
            self.inspect_button.disabled = not bool(self.selected_repo)
            self.preview_button.disabled = self._details is None
            self.import_button.disabled = (
                self._details is None
                or not bool(self.split_choices.value)
            )

    def _set_status(self, message: str, status_type: str) -> None:
        self.status.object = self._status_html(message, status_type)
    def _on_error(self, exc: BaseException) -> None:
        self._set_busy(False)
        message = " ".join(str(exc).split())
        self._fail_progress(message)
        self._set_status(message, "danger")

    def _set_select_options(
        self,
        widget: Any,
        options: Any,
        *,
        preferred: Any = None,
    ) -> None:
        widget.options = options
        values = self._option_values(options)
        if preferred in values:
            widget.value = preferred
        elif values:
            widget.value = values[0]
        else:
            widget.value = None

    @staticmethod
    def _option_values(options: Any) -> list[Any]:
        if isinstance(options, dict):
            return list(options.values())
        return list(options or [])

    @staticmethod
    def _normalise_config(value: Any) -> str | None:
        text = str(value or "").strip()
        if text in {"", "default", "__default__", "None", "null"}:
            return None
        return text

    def _token(self) -> str | None:
        value = str(self.token.value or "").strip()
        return value or None

    @staticmethod
    def _empty_to_none(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if text in {"", "None", "null", "__default__"}:
            return None
        return text

    @staticmethod
    def _mode_label(mode: str | None) -> str:
        if mode == "builder":
            return "structured Hugging Face dataset rows"
        if mode == "files":
            return "repository image files"
        return "automatic detection"

    @staticmethod
    def _selected_html(repo_id: str) -> str:
        safe_repo = html.escape(str(repo_id))
        return f"""
        <div style="display:flex;align-items:center;gap:10px;width:100%;height:62px;min-width:0;padding:10px 12px;border:1px solid #bfd7ea;border-radius:8px;background:#f3f8fc;overflow:hidden;">
          <div style="display:flex;align-items:center;justify-content:center;flex:0 0 34px;width:34px;height:34px;border-radius:9px;background:#ffd21e;color:#263244;font-size:11px;font-weight:800;">HF</div>
          <div style="flex:1 1 auto;min-width:0;overflow:hidden;">
            <div style="color:#687386;font-size:8.8px;font-weight:750;letter-spacing:.05em;text-transform:uppercase;line-height:1.2;">Selected repository</div>
            <div title="{safe_repo}" style="margin-top:5px;min-width:0;color:#0f5f9a;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;font-weight:680;line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{safe_repo}</div>
          </div>
        </div>
        """

    @staticmethod
    def _empty_state_html() -> str:
        return """
        <div style="max-width:390px;text-align:center;">
          <div style="display:inline-flex;align-items:center;justify-content:center;width:44px;height:44px;border-radius:12px;background:#eaf2f8;color:#0f6fbd;font-size:19px;font-weight:800;">HF</div>
          <div style="margin-top:13px;color:#263244;font-size:13px;font-weight:720;line-height:1.3;">Select a dataset to inspect</div>
          <div style="margin-top:5px;color:#687386;font-size:10.5px;line-height:1.5;">
            AstronomicAL will detect configurations, splits, image columns, labels,
            existing registrations, and local cache availability.
          </div>
        </div>
        """

    @staticmethod
    def _structure_placeholder_html() -> str:
        return """
        <div style="padding:12px;border:1px solid #e2e7ee;border-radius:8px;background:#fbfcfe;color:#687386;font-size:10px;line-height:1.4;">
          Inspecting dataset structure…
        </div>
        """

    @staticmethod
    def _status_html(message: str, status_type: str) -> str:
        palettes = {
            "info": ("#e9f4fb", "#0f5f9a", "#bdd9ec", "i"),
            "busy": ("#eaf2ff", "#1d4ed8", "#bfd7ff", "…"),
            "success": ("#ecf8f1", "#1f6a45", "#b8dec9", "✓"),
            "warning": ("#fff7d6", "#8a5d00", "#ead58b", "!"),
            "danger": ("#fff0ee", "#a42920", "#efc1bc", "×"),
        }
        resolved_type = status_type or "info"
        background, colour, border, icon = palettes.get(
            resolved_type,
            palettes["info"],
        )
        safe_message = html.escape(str(message))

        if resolved_type in {"warning", "danger"}:
            return f"""
            <div style="box-sizing:border-box;display:flex;align-items:flex-start;gap:8px;width:100%;max-width:100%;min-width:0;padding:8px 10px;border:1px solid {border};border-radius:8px;background:{background};color:{colour};font-size:9.8px;font-weight:620;line-height:1.45;overflow:hidden;">
              <span style="display:inline-flex;align-items:center;justify-content:center;flex:0 0 18px;width:18px;height:18px;margin-top:1px;border-radius:50%;background:#ffffff;font-size:9px;font-weight:800;">{icon}</span>
              <span style="display:block;flex:1 1 auto;min-width:0;max-width:100%;white-space:normal;overflow-wrap:anywhere;word-break:break-word;">{safe_message}</span>
            </div>
            """

        return f"""
        <div title="{safe_message}" style="box-sizing:border-box;display:flex;align-items:center;gap:6px;width:100%;max-width:390px;min-width:0;padding:5px 8px;border:1px solid {border};border-radius:999px;background:{background};color:{colour};font-size:9.8px;font-weight:650;line-height:1.25;overflow:hidden;">
          <span style="display:inline-flex;align-items:center;justify-content:center;flex:0 0 16px;width:16px;height:16px;border-radius:50%;background:#ffffff;font-size:9px;font-weight:800;">{icon}</span>
          <span style="display:block;flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{safe_message}</span>
        </div>
        """