from __future__ import annotations

import html
import threading
import traceback
from typing import Any

import pandas as pd
import panel as pn

try:
    from importer import import_hf_image_dataset_as_manifest, slugify
    from service import HuggingFaceDatasetService
except ImportError:
    from .importer import import_hf_image_dataset_as_manifest, slugify
    from .service import HuggingFaceDatasetService


DEFAULT_TASKS = [
    "",
    "image-classification",
    "object-detection",
    "semantic-segmentation",
    "image-to-text",
    "depth-estimation",
    "image-segmentation",
]


_HF_BROWSER_FRONTEND_GUARD_INSTALLED = False
_HF_BROWSER_CSS_INSTALLED = False


def _install_hf_browser_frontend_guard_once() -> None:
    """Keep body-level platform menus clickable while this panel is open."""

    global _HF_BROWSER_FRONTEND_GUARD_INSTALLED
    if _HF_BROWSER_FRONTEND_GUARD_INSTALLED:
        return

    pn.config.raw_css.append(
        """
        .al-hf-browser-root {
            position: relative !important;
            z-index: 0 !important;
            isolation: isolate !important;
            overflow: auto !important;
            box-sizing: border-box !important;
        }

        body:has(.al-hmenu-body-popover) .al-hf-browser-root,
        body:has(.al-hmenu-body-popover) .al-hf-browser-root *,
        body.al-hmenu-open-from-hf .al-hf-browser-root,
        body.al-hmenu-open-from-hf .al-hf-browser-root * {
            pointer-events: none !important;
        }
        """
    )

    _HF_BROWSER_FRONTEND_GUARD_INSTALLED = True


def _make_hf_menu_pointer_guard_pane():
    """Fallback observer for browsers/environments where CSS :has is unreliable."""

    html_text = """
    <script>
    (function () {
        if (window.__astronomical_hf_menu_guard_installed__) {
            return;
        }
        window.__astronomical_hf_menu_guard_installed__ = true;

        const update = function () {
            try {
                const open = !!document.querySelector('.al-hmenu-body-popover');
                document.body.classList.toggle('al-hmenu-open-from-hf', open);
            } catch (err) {
                /* best-effort only */
            }
        };

        const observer = new MutationObserver(update);
        observer.observe(document.body, { childList: true, subtree: true });
        update();
    })();
    </script>
    """

    try:
        return pn.pane.HTML(
            html_text,
            sanitize_html=False,
            width=0,
            height=0,
            margin=0,
        )
    except TypeError:
        return pn.pane.HTML(
            html_text,
            width=0,
            height=0,
            margin=0,
        )


def _install_hf_browser_css_once() -> None:
    """Scoped styling for the redesigned Hugging Face importer."""

    global _HF_BROWSER_CSS_INSTALLED
    if _HF_BROWSER_CSS_INSTALLED:
        return

    pn.config.raw_css.append(
        """
        .al-hf-browser-root {
            height: 100% !important;
            width: 100% !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            box-sizing: border-box !important;
            min-height: 0 !important;
            padding: 6px !important;
            scrollbar-gutter: stable !important;
        }

        .al-hf-browser-root .al-hf-browser-header {
            position: sticky !important;
            top: 0 !important;
            z-index: 50 !important;
            background: white !important;
            border-bottom: 1px solid #d8d8d8 !important;
            padding: 4px 6px 6px 6px !important;
            margin: 0 0 6px 0 !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06) !important;
        }

        .al-hf-browser-root .al-hf-browser-header * {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }

        .al-hf-browser-root .al-hf-title h1,
        .al-hf-browser-root .al-hf-title h2,
        .al-hf-browser-root .al-hf-title h3 {
            margin: 0 !important;
            font-size: 15px !important;
            line-height: 1.2 !important;
            font-weight: 700 !important;
        }

        .al-hf-browser-root .al-hf-header-hint {
            margin: 1px 0 2px 0 !important;
            padding: 0 !important;
            color: #5f6b7a !important;
            font-size: 0.82em !important;
            line-height: 1.15 !important;
        }

        .al-hf-browser-root .al-hf-status-line {
            margin: 2px 0 4px 0 !important;
            padding: 2px 6px !important;
            border-radius: 3px !important;
            font-size: 0.82em !important;
            line-height: 1.15 !important;
            border: 1px solid transparent !important;
            min-height: 0 !important;
        }

        .al-hf-browser-root .al-hf-status-info {
            background: #f3fbfd !important;
            border-color: #cbeef5 !important;
            color: #075985 !important;
        }

        .al-hf-browser-root .al-hf-status-primary {
            background: #f3f7ff !important;
            border-color: #bfd7ff !important;
            color: #1d4ed8 !important;
        }

        .al-hf-browser-root .al-hf-status-success {
            background: #f0fdf4 !important;
            border-color: #bbf7d0 !important;
            color: #166534 !important;
        }

        .al-hf-browser-root .al-hf-status-warning {
            background: #fffbeb !important;
            border-color: #fde68a !important;
            color: #92400e !important;
        }

        .al-hf-browser-root .al-hf-status-danger {
            background: #fef2f2 !important;
            border-color: #fecaca !important;
            color: #991b1b !important;
        }

        .al-hf-browser-root .al-hf-step-nav {
            margin-top: 3px !important;
            padding-top: 3px !important;
            border-top: 1px solid #eeeeee !important;
        }

        .al-hf-browser-root .al-hf-step-nav .bk-btn-group {
            width: 100%;
        }

        .al-hf-browser-root .al-hf-step-nav button {
            font-weight: 600;
            padding-top: 3px !important;
            padding-bottom: 3px !important;
            min-height: 26px !important;
        }

        .al-hf-browser-root .al-hf-step-body {
            overflow: visible !important;
            box-sizing: border-box !important;
            min-height: 0 !important;
            padding: 0 2px 180px 2px !important;
        }

        .al-hf-browser-root .al-hf-step-body img[src^="data:image"] {
            width: 100% !important;
            height: 180px !important;
            max-width: none !important;
            max-height: none !important;
            object-fit: contain !important;
            object-position: center center !important;
            image-rendering: pixelated !important;
            display: block !important;
        }

        .al-hf-browser-root .al-hf-step-body div:has(> img[src^="data:image"]) {
            width: 100% !important;
            min-height: 180px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            overflow: hidden !important;
        }

        .al-hf-browser-root .al-hf-step-card {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 12px;
            background: #fff;
            margin-bottom: 12px;
        }

        .al-hf-browser-root .al-hf-muted {
            color: #666;
            font-size: 0.92em;
        }

        .al-hf-browser-root .al-hf-diagnosis {
            border-left: 4px solid #3f7fbf;
            padding: 10px 12px;
            background: #f7fbff;
        }

        .al-hf-browser-root .al-hf-warning-box {
            border-left: 4px solid #b7791f;
            padding: 10px 12px;
            background: #fff8e8;
        }

        .al-hf-browser-root .al-hf-success-box {
            border-left: 4px solid #2f855a;
            padding: 10px 12px;
            background: #f0fff4;
        }

        .al-hf-browser-root .bk-Column,
        .al-hf-browser-root .bk-Row,
        .al-hf-browser-root .bk-panel-models-esm-ReactComponent {
            z-index: auto !important;
        }
        """
    )

    _HF_BROWSER_CSS_INSTALLED = True


class HuggingFaceBrowserPanel:
    """Guided Hugging Face image-dataset importer.

    The backend service/importer contracts are intentionally preserved:
    - search uses HuggingFaceDatasetService.search_dataframe
    - diagnosis uses get_dataset_details / get_dataset_details_builder
    - preview uses preview_image_grid_html / preview_image_grid_html_builder
    - import uses import_hf_image_dataset_as_manifest
    """

    state_version = 4

    def __init__(self, context: Any) -> None:
        self.context = context

        _install_hf_browser_frontend_guard_once()
        _install_hf_browser_css_once()

        self._menu_pointer_guard = _make_hf_menu_pointer_guard_pane()
        self._disposed = False
        self._doc = pn.state.curdoc
        self._job_handles: list[Any] = []

        self._search_df = pd.DataFrame()
        self._details = None
        self._details_mode: str | None = None
        self._restored_state: dict[str, Any] = {}

        self._last_suggested_dataset_id = ""
        self._last_suggested_dataset_name = ""

        self._active_step = 0
        self._steps: list[tuple[str, Any]] = []
        self.step_body = pn.Column(
            sizing_mode="stretch_width",
            css_classes=["al-hf-step-body"],
            styles={
                "overflow": "visible",
                "min-height": "0",
            },
        )

        self.step_bottom_spacer = pn.Spacer(height=180, sizing_mode="stretch_width")

        self._progress_lock = threading.RLock()
        self._progress_state: dict[str, Any] = {
            "active": False,
            "phase": "idle",
            "completed": 0,
            "total": 0,
            "percent": 0,
            "message": "Idle",
            "current_file": "",
        }

        # ------------------------------------------------------------------
        # Search widgets
        # ------------------------------------------------------------------

        self.query = pn.widgets.TextInput(
            name="Search",
            value="cifar10",
            placeholder="Search Hugging Face datasets...",
            sizing_mode="stretch_width",
        )

        self.task = pn.widgets.Select(
            name="Task filter",
            options=DEFAULT_TASKS,
            value="image-classification",
            sizing_mode="stretch_width",
        )

        self.sort = pn.widgets.Select(
            name="Sort",
            options=["downloads", "likes", "lastModified"],
            value="downloads",
            sizing_mode="stretch_width",
        )

        self.limit = pn.widgets.IntInput(
            name="Max results",
            value=25,
            start=1,
            end=200,
            sizing_mode="stretch_width",
        )

        self.token = pn.widgets.PasswordInput(
            name="HF token, optional",
            placeholder="Only needed for gated/private datasets",
            sizing_mode="stretch_width",
        )

        self.search_button = pn.widgets.Button(
            name="Search datasets",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.search_button.on_click(self._on_search_clicked)

        self.dataset_select = pn.widgets.Select(
            name="Selected dataset",
            options=[],
            sizing_mode="stretch_width",
        )
        self.dataset_select.param.watch(self._on_dataset_changed, "value")

        self.inspect_button = pn.widgets.Button(
            name="Inspect / diagnose selected dataset",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.inspect_button.on_click(self._on_inspect_clicked)

        self.results_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            selectable=1,
            pagination="local",
            page_size=10,
            sizing_mode="stretch_width",
            height=320,
            disabled=True,
        )
        self.results_table.param.watch(self._on_result_row_selected, "selection")

        # ------------------------------------------------------------------
        # Diagnosis/config widgets
        # ------------------------------------------------------------------

        self.import_mode = pn.widgets.RadioButtonGroup(
            name="Import strategy",
            options={
                "Auto recommendation": "auto",
                "Fast Hub files": "files",
                "HF Datasets builder": "builder",
            },
            value="auto",
            button_type="default",
            sizing_mode="stretch_width",
        )
        self.import_mode.param.watch(self._on_strategy_changed, "value")

        self.trust_remote_code = pn.widgets.Checkbox(
            name="Trust remote dataset code for builder mode",
            value=False,
        )

        self.config_select = pn.widgets.Select(
            name="Config",
            options=["default"],
            value="default",
            sizing_mode="stretch_width",
        )
        self.config_select.param.watch(self._on_config_changed, "value")

        self.split_select = pn.widgets.Select(
            name="Split",
            options=["train"],
            value="train",
            sizing_mode="stretch_width",
        )
        self.split_select.param.watch(self._on_split_changed, "value")

        self.image_column = pn.widgets.Select(
            name="Image source",
            options={"Hub image files": "__hf_file__"},
            value="__hf_file__",
            sizing_mode="stretch_width",
        )

        self.label_column = pn.widgets.Select(
            name="Label source",
            options={
                "Inferred from path/folder if available": "__hf_path_label__",
                "No label": "",
            },
            value="__hf_path_label__",
            sizing_mode="stretch_width",
        )

        self.id_column = pn.widgets.Select(
            name="Record ID column",
            options=[""],
            value="",
            sizing_mode="stretch_width",
        )

        self.diagnosis_pane = pn.pane.Markdown(
            "Search for a dataset, select a result, then inspect it.",
            sizing_mode="stretch_width",
            css_classes=["al-hf-diagnosis"],
        )

        self.details_pane = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
        )

        # ------------------------------------------------------------------
        # Preview/import widgets
        # ------------------------------------------------------------------

        self.preview_limit = pn.widgets.IntInput(
            name="Preview rows",
            value=8,
            start=1,
            end=50,
            sizing_mode="stretch_width",
        )

        self.preview_button = pn.widgets.Button(
            name="Preview selected split",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.preview_button.on_click(self._on_preview_clicked)

        self.preview_grid = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
        )

        self.max_import_rows = pn.widgets.IntInput(
            name="Max import rows; 0 = all selected split",
            value=0,
            start=0,
            sizing_mode="stretch_width",
        )

        self.dataset_id = pn.widgets.TextInput(
            name="AstronomicAL dataset ID",
            value="",
            placeholder="Auto-generated if empty",
            sizing_mode="stretch_width",
        )

        self.dataset_name = pn.widgets.TextInput(
            name="AstronomicAL dataset name",
            value="",
            placeholder="Auto-generated if empty",
            sizing_mode="stretch_width",
        )

        self.write_parquet = pn.widgets.Checkbox(
            name="Write manifest to Parquet cache",
            value=True,
        )

        self.set_active = pn.widgets.Checkbox(
            name="Set as active dataset",
            value=True,
        )

        self.download_all_splits = pn.widgets.Checkbox(
            name="Download all available splits",
            value=False,
        )
        self.download_all_splits.param.watch(self._on_download_all_splits_changed, "value")

        self.active_split_after_import = pn.widgets.Select(
            name="Set active dataset after all-splits import",
            options={"None": ""},
            value="",
            disabled=True,
            visible=False,
            sizing_mode="stretch_width",
        )

        self.all_splits_note = pn.pane.Markdown(
            "When importing all splits, choose which imported split should become active, or choose **None**.",
            visible=False,
            sizing_mode="stretch_width",
            css_classes=["al-hf-muted"],
        )

        self.import_button = pn.widgets.Button(
            name="Download and register dataset",
            button_type="success",
            sizing_mode="stretch_width",
        )
        self.import_button.on_click(self._on_import_clicked)

        self.import_summary = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
        )

        self.import_preview = pn.pane.DataFrame(
            pd.DataFrame(),
            sizing_mode="stretch_width",
            height=240,
        )

        # ------------------------------------------------------------------
        # Sticky status/progress
        # ------------------------------------------------------------------

        self.activity = pn.pane.HTML(
            "<span><b>Activity:</b> idle</span>",
            sizing_mode="stretch_width",
            margin=0,
        )

        self.status = pn.pane.HTML(
            "Search Hugging Face for image datasets.",
            sizing_mode="stretch_width",
            css_classes=["al-hf-status-line", "al-hf-status-info"],
            margin=0,
        )

        self.progress_bar = pn.widgets.Progress(
            name="Progress",
            value=0,
            max=100,
            visible=False,
            sizing_mode="stretch_width",
        )

        self.progress_text = pn.pane.Markdown(
            "",
            visible=False,
            sizing_mode="stretch_width",
        )

        self.step_nav = pn.widgets.RadioButtonGroup(
            name="Importer step",
            options={
                "Search": 0,
                "Diagnose": 1,
                "Preview": 2,
                "Import": 3,
            },
            value=0,
            button_type="light",
            sizing_mode="stretch_width",
            css_classes=["al-hf-step-nav"],
        )
        self.step_nav.param.watch(self._on_step_nav_changed, "value")

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------

        self.search_step = pn.Column(
            pn.pane.Markdown("### 1. Search", height=9),
            pn.pane.Markdown(
                "Find an image-style Hugging Face dataset. Select a row or use the dataset dropdown, then inspect it.",
                css_classes=["al-hf-muted"],
                height=15
            ),
            pn.Row(self.query, self.task, self.sort, self.limit, sizing_mode="stretch_width"),
            self.token,
            self.search_button,
            pn.pane.Markdown("#### Results (Scroll down to see all) ",height=10),
            self.dataset_select,
            self.inspect_button,
            self.results_table,
            sizing_mode="stretch_width",
            css_classes=["al-hf-step-card"],
        )

        self.diagnosis_step = pn.Column(
            pn.pane.Markdown("### 2. Diagnose"),
            pn.pane.Markdown(
                "AstronomicAL detects the safest import strategy, then lets you configure the split, image source, labels, and record IDs.",
                css_classes=["al-hf-muted"],
            ),

            pn.pane.Markdown("#### Import strategy"),
            self.import_mode,
            self.trust_remote_code,
            self.diagnosis_pane,

            pn.pane.Markdown("#### Configure import"),
            pn.Row(self.config_select, self.split_select, sizing_mode="stretch_width"),
            pn.Row(self.image_column, self.label_column, self.id_column, sizing_mode="stretch_width"),

            self.details_pane,

            sizing_mode="stretch_width",
            css_classes=["al-hf-step-card"],
        )

        self.preview_step = pn.Column(
            pn.pane.Markdown("### 3. Preview"),
            pn.pane.Markdown(
                "Preview downloads only the displayed images. It does not import the full dataset.",
                css_classes=["al-hf-muted"],
            ),
            pn.Row(self.preview_limit, self.preview_button, sizing_mode="stretch_width"),
            self.preview_grid,
            sizing_mode="stretch_width",
            css_classes=["al-hf-step-card"],
        )

        self.import_step = pn.Column(
            pn.pane.Markdown("### 4. Import"),
            pn.pane.Markdown(
                "Register the selected split as an AstronomicAL image-manifest dataset with semantic mappings.",
                css_classes=["al-hf-muted"],
            ),
            pn.Row(self.max_import_rows, sizing_mode="stretch_width"),
            pn.Row(self.dataset_id, self.dataset_name, sizing_mode="stretch_width"),
            pn.Row(
                self.write_parquet,
                self.download_all_splits,
                self.set_active,
                sizing_mode="stretch_width",
            ),
            self.all_splits_note,
            self.active_split_after_import,
            self.import_button,
            self.import_summary,
            pn.pane.Markdown("#### Manifest preview"),
            self.import_preview,
            sizing_mode="stretch_width",
            css_classes=["al-hf-step-card"],
        )

        self._steps = [
            ("Search", self.search_step),
            ("Diagnose", self.diagnosis_step),
            ("Preview", self.preview_step),
            ("Import", self.import_step),
        ]
        self._set_active_step(self._active_step)

        self.header = pn.Column(
            pn.Row(
                pn.pane.Markdown(
                    "## Hugging Face Dataset Importer",
                    css_classes=["al-hf-title"],
                    margin=0,
                    sizing_mode="stretch_width",
                ),
                self.activity,
                sizing_mode="stretch_width",
            ),
            pn.pane.Markdown(
                "Search → diagnose → configure → preview → import.",
                css_classes=["al-hf-header-hint"],
                sizing_mode="stretch_width",
                margin=0,
            ),
            self.status,
            self.progress_bar,
            self.progress_text,
            self.step_nav,
            sizing_mode="stretch_width",
            css_classes=["al-hf-browser-header"],
            margin=0,
        )

        self._view = pn.Column(
            self._menu_pointer_guard,
            self.header,
            self.step_body,
            sizing_mode="stretch_both",
            css_classes=["al-hf-browser-root"],
            styles={
                "height": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "padding": "6px",
                "min-height": "0",
            },
        )

        try:
            self._progress_periodic = pn.state.add_periodic_callback(
                self._sync_progress_ui,
                period=250,
                start=True,
            )
        except Exception:
            self._progress_periodic = None

    # This is never called. It only prevents type checkers/static tools from
    # complaining in environments where Panel's Button.from_param branch above is
    # optimized differently. It is intentionally harmless.
    @property
    def param_placeholder(self):
        return None

    # ------------------------------------------------------------------
    # Panel lifecycle/state
    # ------------------------------------------------------------------

    def _on_step_nav_changed(self, event: Any) -> None:
        try:
            value = int(event.new)
        except Exception:
            value = 0

        self._set_active_step(value)

    def _set_active_step(self, index: int) -> None:
        try:
            index = int(index)
        except Exception:
            index = 0

        self._active_step = max(0, index)


        if not getattr(self, "_steps", None):
            return

        self._active_step = max(0, min(self._active_step, len(self._steps) - 1))

        step_nav = getattr(self, "step_nav", None)
        if step_nav is not None and step_nav.value != self._active_step:
            step_nav.value = self._active_step

        step_body = getattr(self, "step_body", None)
        if step_body is None:
            return

        _title, panel = self._steps[self._active_step]

        spacer = getattr(self, "step_bottom_spacer", None) or pn.Spacer(height=96)
        step_body[:] = [panel, spacer]

        try:
            step_body.scroll_position = 0
        except Exception:
            pass

    def panel(self):
        return self._view

    def dispose(self) -> None:
        self._disposed = True

        try:
            if getattr(self, "_progress_periodic", None) is not None:
                self._progress_periodic.stop()
        except Exception:
            pass

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
            "dataset": self.dataset_select.value,
            "config": self.config_select.value,
            "split": self.split_select.value,
            "image_column": self.image_column.value,
            "label_column": self.label_column.value,
            "id_column": self.id_column.value,
            "preview_limit": self.preview_limit.value,
            "max_import_rows": self.max_import_rows.value,
            "dataset_id": self.dataset_id.value,
            "dataset_name": self.dataset_name.value,
            "write_parquet": self.write_parquet.value,
            "set_active": self.set_active.value,
            "download_all_splits": self.download_all_splits.value,
            "active_split_after_import": self.active_split_after_import.value,
            "trust_remote_code": self.trust_remote_code.value,
            "import_mode": self.import_mode.value,
            "active_tab": self._active_step,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        self._restored_state = dict(state)

        widget_map = {
            "query": self.query,
            "task": self.task,
            "sort": self.sort,
            "limit": self.limit,
            "preview_limit": self.preview_limit,
            "max_import_rows": self.max_import_rows,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "write_parquet": self.write_parquet,
            "set_active": self.set_active,
            "download_all_splits": self.download_all_splits,
            "active_split_after_import": self.active_split_after_import,
            "trust_remote_code": self.trust_remote_code,
            "import_mode": self.import_mode,
        }

        for key, widget in widget_map.items():
            if key not in state:
                continue
            try:
                widget.value = state[key]
            except Exception:
                pass

        try:
            active_tab = int(state.get("active_tab", 0) or 0)
            self._set_active_step(active_tab)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _on_search_clicked(self, event: Any) -> None:
        self._run_job(
            self._search,
            title="Searching Hugging Face datasets",
            key=f"hf.search:{self.query.value}:{self.task.value}:{self.limit.value}",
            on_done=self._on_search_done,
            on_error=self._on_error,
        )

    def _search(self, *, cancel_token: Any = None):
        service = self._service()
        return service.search_dataframe(
            query=str(self.query.value or ""),
            task=str(self.task.value or ""),
            limit=int(self.limit.value or 25),
            sort=str(self.sort.value or "downloads"),
            token=self._token(),
        )

    def _on_search_done(self, df: pd.DataFrame) -> None:
        self._set_busy(False)
        self._finish_progress("Search complete.")

        self._search_df = df if df is not None else pd.DataFrame()
        self.results_table.value = self._search_df

        if self._search_df.empty or "repo_id" not in self._search_df.columns:
            self.dataset_select.options = []
            self._set_warning("No datasets found.")
            return

        repo_ids = [str(value) for value in self._search_df["repo_id"].tolist()]
        self.dataset_select.options = repo_ids

        restored_dataset = str(self._restored_state.get("dataset") or "").strip()
        if restored_dataset and restored_dataset in repo_ids:
            self.dataset_select.value = restored_dataset
            try:
                idx = repo_ids.index(restored_dataset)
                self.results_table.selection = [idx]
            except Exception:
                pass
        else:
            self.dataset_select.value = repo_ids[0]
            try:
                self.results_table.selection = [0]
            except Exception:
                pass

        self._set_success(f"Found {len(repo_ids)} dataset(s). Select one and inspect it.")
        self._set_active_step(0)

    def _on_result_row_selected(self, event: Any) -> None:
        selection = list(event.new or [])
        if not selection or self._search_df.empty or "repo_id" not in self._search_df.columns:
            return

        selected = selection[0]

        try:
            if selected in self._search_df.index:
                row = self._search_df.loc[selected]
            else:
                row = self._search_df.iloc[int(selected)]
            repo_id = str(row.get("repo_id") or "").strip()
        except Exception:
            return

        if repo_id:
            try:
                self.dataset_select.value = repo_id
            except Exception:
                pass

    def _on_dataset_changed(self, event: Any) -> None:
        self._details = None
        self._details_mode = None
        self._suggest_dataset_ids(overwrite_generated_only=True)

        repo_id = str(event.new or "").strip()
        if repo_id:
            self.diagnosis_pane.object = (
                f"Selected `{self._escape(repo_id)}`. Run **Inspect / diagnose** to detect "
                "configs, splits, image sources, and labels."
            )

    # ------------------------------------------------------------------
    # Diagnosis/configuration
    # ------------------------------------------------------------------

    def _on_strategy_changed(self, event: Any) -> None:
        desired = str(event.new or "auto")
        if self._details is None:
            return

        if desired in {"files", "builder"} and desired != self._details_mode:
            self._set_warning(
                "Import strategy changed. Run Inspect / diagnose again so the column and split choices match the selected strategy."
            )
            return

        self._refresh_diagnosis_text()

    def _on_inspect_clicked(self, event: Any) -> None:
        repo_id = str(self.dataset_select.value or "").strip()
        if not repo_id:
            self._set_error("Choose a dataset first.")
            return

        strategy = str(self.import_mode.value or "auto")
        self._run_job(
            self._inspect,
            title=f"Inspecting {repo_id}",
            key=f"hf.inspect:{strategy}:{repo_id}:{self.config_select.value}",
            on_done=self._on_inspect_done,
            on_error=self._on_error,
        )

    def _inspect(self, *, cancel_token: Any = None):
        service = self._service()
        repo_id = str(self.dataset_select.value)

        desired = str(self.import_mode.value or "auto")

        if desired in {"auto", "files"}:
            details = service.get_dataset_details(
                repo_id,
                token=self._token(),
                trust_remote_code=bool(self.trust_remote_code.value),
            )

            if desired == "files":
                return {"mode": "files", "details": details}

            has_file_images = bool(getattr(details, "sample_files", None)) or int(
                getattr(details, "scanned_file_count", 0) or 0
            ) > 0

            if has_file_images:
                return {"mode": "files", "details": details}

        details = service.get_dataset_details_builder(
            repo_id,
            config_name=self._config(),
            token=self._token(),
            trust_remote_code=bool(self.trust_remote_code.value),
        )
        return {"mode": "builder", "details": details}

    def _on_inspect_done(self, payload: Any) -> None:
        self._set_busy(False)
        self._finish_progress("Inspection complete.")

        if isinstance(payload, dict) and "details" in payload:
            self._details_mode = str(payload.get("mode") or "files")
            self._details = payload.get("details")
        else:
            self._details_mode = "files"
            self._details = payload

        details = self._details
        if details is None:
            self._set_error("Inspection returned no details.")
            return

        configs = list(getattr(details, "configs", None) or ["default"])
        if not configs:
            configs = ["default"]

        restored_config = str(self._restored_state.get("config") or "").strip()
        config_value = restored_config if restored_config in configs else configs[0]

        self.config_select.options = configs
        self.config_select.value = config_value

        self._update_split_options(config_value)
        self._update_column_controls_from_details(config_value)
        self._update_active_split_options()
        self._restore_config_values_if_possible()
        self._suggest_dataset_ids(overwrite_generated_only=True)
        self._refresh_diagnosis_text()

        mode_label = self._mode_label(self._details_mode)
        self._set_success(f"Inspection complete. Recommended strategy: {mode_label}. Feel free to continue to preview/import.")
        self._set_active_step(1)

    def _on_config_changed(self, event: Any) -> None:
        if self._details is None:
            return

        config = str(event.new or "default")
        self._update_split_options(config)
        self._update_column_controls_from_details(config)
        self._update_active_split_options()
        self._suggest_dataset_ids(overwrite_generated_only=True)
        self._refresh_diagnosis_text()

    def _on_split_changed(self, event: Any) -> None:
        self._update_active_split_options()
        self._suggest_dataset_ids(overwrite_generated_only=True)
        self._refresh_diagnosis_text()

    def _update_split_options(self, config: str) -> None:
        details = self._details
        if details is None:
            return

        splits_by_config = getattr(details, "splits_by_config", {}) or {}
        splits = list(splits_by_config.get(config) or splits_by_config.get("default") or ["train"])
        if not splits:
            splits = ["train"]

        restored_split = str(self._restored_state.get("split") or "").strip()
        split_value = restored_split if restored_split in splits else splits[0]

        self.split_select.options = splits
        self.split_select.value = split_value

    def _update_column_controls_from_details(self, config: str) -> None:
        details = self._details
        mode = self._details_mode or "files"

        if details is None:
            return

        if mode == "builder":
            features_by_config = getattr(details, "features_by_config", {}) or {}
            features = dict(features_by_config.get(config) or features_by_config.get("default") or {})
            columns = list(features.keys())

            image_candidates = list(getattr(details, "image_column_candidates", None) or [])
            if not image_candidates:
                image_candidates = [
                    col
                    for col in columns
                    if str(col).lower() in {"image", "img", "picture", "photo", "thumbnail"}
                ]
            if not image_candidates and columns:
                image_candidates = columns[:]
            if not image_candidates:
                image_candidates = ["image"]

            label_candidates = list(getattr(details, "label_column_candidates", None) or [])
            for candidate in ["label", "labels", "target", "class", "category"]:
                if candidate in columns and candidate not in label_candidates:
                    label_candidates.append(candidate)

            self._set_select_options(
                self.image_column,
                image_candidates,
                preferred=self.image_column.value if self.image_column.value in image_candidates else image_candidates[0],
            )

            label_options = [""] + columns
            preferred_label = ""
            for candidate in label_candidates:
                if candidate in columns:
                    preferred_label = candidate
                    break

            self._set_select_options(
                self.label_column,
                label_options,
                preferred=preferred_label,
            )

            self._set_select_options(
                self.id_column,
                [""] + columns,
                preferred="",
            )

        else:
            self._set_select_options(
                self.image_column,
                {"Hub image files": "__hf_file__"},
                preferred="__hf_file__",
            )
            self._set_select_options(
                self.label_column,
                {
                    "Inferred from path/folder if available": "__hf_path_label__",
                    "No label": "",
                },
                preferred="__hf_path_label__",
            )
            self._set_select_options(
                self.id_column,
                [""],
                preferred="",
            )

    def _restore_config_values_if_possible(self) -> None:
        state = self._restored_state
        if not state:
            return

        for key, widget in {
            "image_column": self.image_column,
            "label_column": self.label_column,
            "id_column": self.id_column,
        }.items():
            value = state.get(key)
            if value is None:
                continue

            values = self._option_values(widget.options)
            if value in values:
                try:
                    widget.value = value
                except Exception:
                    pass

    def _refresh_diagnosis_text(self) -> None:
        details = self._details
        if details is None:
            return

        self.diagnosis_pane.object = self._diagnosis_markdown(details)
        self.details_pane.object = self._details_markdown(details, config=str(self.config_select.value or "default"))

    def _diagnosis_markdown(self, details: Any) -> str:
        repo_id = self._escape(getattr(details, "repo_id", None) or self.dataset_select.value or "")
        mode = self._details_mode or self._selected_mode()
        mode_label = self._mode_label(mode)

        reasons: list[str] = []
        warnings: list[str] = []

        sample_count = int(getattr(details, "scanned_file_count", 0) or 0)
        sample_files = list(getattr(details, "sample_files", None) or [])

        if mode == "files":
            if sample_count or sample_files:
                reasons.append(
                    "Hub image files were detected, so AstronomicAL can import the dataset without loading the full HF Datasets builder."
                )
            else:
                reasons.append(
                    "Fast file mode was selected manually. Import may fail if no image files are discoverable from Hub file paths."
                )
            reasons.append("Labels will be inferred from folder/file paths when possible.")
        else:
            reasons.append(
                "Builder mode exposes HF Datasets features/columns and is better for datasets whose images are stored in structured dataset rows."
            )
            if self.trust_remote_code.value:
                warnings.append("Remote dataset code is trusted for builder access. Only use this for repositories you trust.")

            try:
                service = self._service()
                if service.selected_builder_label_is_non_semantic(
                    details,
                    config=str(self.config_select.value or "default"),
                    label_column=self._empty_to_none(self.label_column.value),
                ):
                    warnings.append(
                        "The selected builder label column looks non-semantic, such as shard/file labels. Fast Hub-file import may produce better class labels."
                    )
            except Exception:
                pass

        if getattr(details, "error", None):
            for line in str(details.error).splitlines():
                line = line.strip()
                if line:
                    warnings.append(line)

        selected_strategy = str(self.import_mode.value or "auto")
        if selected_strategy == "auto":
            strategy_line = "Auto recommendation is active."
        else:
            strategy_line = f"Manual strategy selected: `{self._escape(self._mode_label(selected_strategy))}`."

        reason_lines = "\n".join(f"- {self._escape(reason)}" for reason in reasons) or "- No diagnosis reason available."
        warning_lines = ""
        if warnings:
            warning_lines = "\n\n**Warnings / notes:**\n\n" + "\n".join(
                f"- {self._escape(warning)}" for warning in warnings
            )

        if bool(self.download_all_splits.value):
            splits_text = ", ".join(self._available_splits())
            active_after = self.active_split_after_import.value or "None"
            split_line = (
                f"**Import scope:** all detected splits: "
                f"`{self._escape(splits_text)}`  \n"
                f"**Active dataset after import:** `{self._escape(active_after)}`  \n"
            )
        else:
            split_line = (
                f"**Selected split:** `{self._escape(str(self.split_select.value or 'train'))}`  \n"
            )

        return (
            f"**Dataset:** `{repo_id}`  \n"
            f"**Recommended import strategy:** `{self._escape(mode_label)}`  \n"
            f"{split_line}"
            f"**Selected image source:** `{self._escape(str(self.image_column.value or ''))}`  \n"
            f"**Selected label source:** `{self._escape(str(self.label_column.value or ''))}`  \n\n"
            f"{strategy_line}\n\n"
            f"**Why:**\n\n{reason_lines}"
            f"{warning_lines}"
        )

    def _details_markdown(self, details: Any, *, config: str) -> str:
        features_by_config = getattr(details, "features_by_config", {}) or {}
        features = dict(features_by_config.get(config) or features_by_config.get("default") or {})

        splits_by_config = getattr(details, "splits_by_config", {}) or {}
        splits = list(splits_by_config.get(config) or splits_by_config.get("default") or [])

        feature_lines = "\n".join(
            f"- `{self._escape(str(name))}`: `{self._escape(str(kind))}`"
            for name, kind in features.items()
        ) or "_No feature columns found for this strategy/config._"

        sample_lines = ""
        sample_files = list(getattr(details, "sample_files", None) or [])
        if sample_files:
            sample_lines = "\n\n**Sample image files:**\n\n" + "\n".join(
                f"- `{self._escape(getattr(file, 'path', ''))}`"
                f" → split `{self._escape(getattr(file, 'split', ''))}`"
                f"{', label `' + self._escape(getattr(file, 'label', '')) + '`' if getattr(file, 'label', '') else ''}"
                for file in sample_files[:10]
            )

        configs = list(getattr(details, "configs", None) or [])
        scanned = int(getattr(details, "scanned_file_count", 0) or 0)

        source_heading = (
            "Available builder columns"
            if self._details_mode == "builder"
            else "Available lightweight sources"
        )

        return (
            f"#### Dataset structure\n\n"
            f"**Configs:** `{self._escape(', '.join(configs) if configs else 'default')}`  \n"
            f"**Current config:** `{self._escape(config)}`  \n"
            f"**Inferred splits:** `{self._escape(', '.join(splits) if splits else 'train')}`  \n"
            f"**Sampled image files:** `{scanned}`  \n\n"
            f"**{source_heading}:**\n\n{feature_lines}"
            f"{sample_lines}"
        )

    def _on_download_all_splits_changed(self, event: Any) -> None:
        enabled = bool(event.new)

        self.set_active.disabled = enabled
        self.active_split_after_import.disabled = not enabled
        self.active_split_after_import.visible = enabled
        self.all_splits_note.visible = enabled

        if enabled:
            self.set_active.value = False
            self._update_active_split_options()
        else:
            self.active_split_after_import.value = ""
            self.set_active.disabled = False

        self._suggest_dataset_ids(overwrite_generated_only=True)
        self._refresh_diagnosis_text()

    def _available_splits(self) -> list[str]:
        details = self._details
        if details is None:
            return [str(self.split_select.value or "train")]

        config = str(self.config_select.value or "default")
        splits_by_config = getattr(details, "splits_by_config", {}) or {}

        splits = list(
            splits_by_config.get(config)
            or splits_by_config.get("default")
            or []
        )

        if not splits:
            split = str(self.split_select.value or "train").strip() or "train"
            splits = [split]

        cleaned: list[str] = []
        for split in splits:
            split = str(split or "").strip()
            if split and split not in cleaned:
                cleaned.append(split)

        return cleaned or ["train"]

    def _update_active_split_options(self) -> None:
        splits = self._available_splits()
        options = {"None": ""}
        options.update({split: split for split in splits})

        previous = self.active_split_after_import.value
        self.active_split_after_import.options = options

        if previous in options.values():
            self.active_split_after_import.value = previous
        elif str(self.split_select.value or "") in options.values():
            self.active_split_after_import.value = str(self.split_select.value)
        else:
            self.active_split_after_import.value = ""

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _on_preview_clicked(self, event: Any) -> None:
        repo_id = str(self.dataset_select.value or "").strip()
        if not repo_id:
            self._set_error("Choose a dataset first.")
            return

        if self._details is None:
            self._set_error("Inspect the dataset before previewing it.")
            return

        if self._mode_needs_reinspect():
            self._set_error(
                "Import strategy changed after inspection. Run Inspect / diagnose again before previewing."
            )
            return

        self._run_job(
            self._preview,
            title=f"Previewing {repo_id}",
            key=f"hf.preview:{self._selected_mode()}:{repo_id}:{self.split_select.value}:{self.preview_limit.value}",
            on_done=self._on_preview_done,
            on_error=self._on_error,
        )

    def _preview(self, *, cancel_token: Any = None):
        service = self._service()
        mode = self._selected_mode()

        if mode == "builder":
            return service.preview_image_grid_html_builder(
                repo_id=str(self.dataset_select.value),
                config_name=self._config(),
                split=str(self.split_select.value or "train"),
                image_column=self._empty_to_none(self.image_column.value),
                label_column=self._empty_to_none(self.label_column.value),
                id_column=self._empty_to_none(self.id_column.value),
                limit=int(self.preview_limit.value or 8),
                token=self._token(),
                trust_remote_code=bool(self.trust_remote_code.value),
            )

        return service.preview_image_grid_html(
            repo_id=str(self.dataset_select.value),
            config_name=self._config(),
            split=str(self.split_select.value or "train"),
            image_column=self._empty_to_none(self.image_column.value),
            label_column=self._empty_to_none(self.label_column.value),
            id_column=self._empty_to_none(self.id_column.value),
            limit=int(self.preview_limit.value or 8),
            token=self._token(),
            trust_remote_code=bool(self.trust_remote_code.value),
        )

    def _on_preview_done(self, html_text: str) -> None:
        self._set_busy(False)
        self._finish_progress("Preview loaded.")

        self.preview_grid.object = html_text or "<em>No preview returned.</em>"
        self._set_success(
            f"Preview loaded. Only up to {int(self.preview_limit.value or 8)} displayed image file(s) were downloaded."
        )
        self._set_active_step(2)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def _on_import_clicked(self, event: Any) -> None:
        repo_id = str(self.dataset_select.value or "").strip()
        if not repo_id:
            self._set_error("Choose a dataset first.")
            return

        if self._details is None:
            self._set_error("Inspect the dataset before importing it.")
            return

        if self._mode_needs_reinspect():
            self._set_error(
                "Import strategy changed after inspection. Run Inspect / diagnose again before importing."
            )
            return

        self._run_job(
            self._import,
            title=f"Downloading/registering {repo_id}",
            key=f"hf.import:{self._selected_mode()}:{repo_id}:{self.split_select.value}:{self.max_import_rows.value}",
            on_done=self._on_import_done,
            on_error=self._on_error,
        )

    def _import(self, *, cancel_token: Any = None):
        mode = self._selected_mode()

        image_column = self._empty_to_none(self.image_column.value)
        label_column = self._empty_to_none(self.label_column.value)

        if bool(self.download_all_splits.value):
            return self._import_all_splits(
                mode=mode,
                image_column=image_column,
                label_column=label_column,
                cancel_token=cancel_token,
            )

        return self._import_single_split(
            mode=mode,
            split=str(self.split_select.value or "train"),
            dataset_id=self._empty_to_none(self.dataset_id.value),
            dataset_name=self._empty_to_none(self.dataset_name.value),
            image_column=image_column,
            label_column=label_column,
            set_active=bool(self.set_active.value),
            cancel_token=cancel_token,
        )

    def _import_single_split(
        self,
        *,
        mode: str,
        split: str,
        dataset_id: str | None,
        dataset_name: str | None,
        image_column: str | None,
        label_column: str | None,
        set_active: bool,
        cancel_token: Any = None,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        import_note = ""

        if mode == "builder":
            service = self._service()
            config = str(self.config_select.value or "default")
            if service.selected_builder_label_is_non_semantic(
                self._details,
                config=config,
                label_column=label_column,
            ):
                mode = "files"
                image_column = "__hf_file__"
                label_column = "__hf_path_label__"
                import_note = (
                    "Builder labels looked non-semantic, for example data_0/data_1 shard labels. "
                    "Used fast Hub-file import instead."
                )
            else:
                image_column = image_column or "image"

        if mode == "files":
            image_column = "__hf_file__"
            label_column = "__hf_path_label__"

        result = import_hf_image_dataset_as_manifest(
            self.context,
            repo_id=str(self.dataset_select.value),
            config_name=self._config(),
            split=split,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            image_column=image_column,
            label_column=label_column,
            id_column=self._empty_to_none(self.id_column.value),
            max_rows=int(self.max_import_rows.value or 0),
            token=self._token(),
            trust_remote_code=bool(self.trust_remote_code.value),
            write_parquet=bool(self.write_parquet.value),
            set_active=bool(set_active),
            cancel_token=cancel_token,
            progress_callback=progress_callback or self._set_progress_from_worker,
        )

        if import_note:
            result["import_note"] = import_note

        return result

    def _import_all_splits(
        self,
        *,
        mode: str,
        image_column: str | None,
        label_column: str | None,
        cancel_token: Any = None,
    ) -> dict[str, Any]:
        splits = self._available_splits()
        if not splits:
            raise ValueError("No splits are available to import.")

        active_split = str(self.active_split_after_import.value or "")
        if active_split and active_split not in splits:
            raise ValueError(
                f"Active split {active_split!r} is not one of the available splits: {splits!r}"
            )

        repo_id = str(self.dataset_select.value or "hf_dataset")
        config = self._config() or self._details_mode or "files"

        base_dataset_id = self._empty_to_none(self.dataset_id.value)
        if not base_dataset_id:
            base_dataset_id = slugify(
                f"hf_{repo_id}_{config}_all_splits",
                fallback="hf_image_dataset",
            )

        base_dataset_name = self._empty_to_none(self.dataset_name.value)
        if not base_dataset_name:
            base_dataset_name = f"HF {repo_id} [{config} / all splits]"

        results: list[dict[str, Any]] = []
        previews: list[pd.DataFrame] = []
        total_rows = 0

        for index, split in enumerate(splits):
            if cancel_token is not None and cancel_token.cancelled():
                return {
                    "cancelled": True,
                    "multi_split": True,
                    "results": results,
                    "rows": total_rows,
                    "total_rows": total_rows,
                }

            split_dataset_id = slugify(
                f"{base_dataset_id}_{split}",
                fallback=f"hf_image_dataset_{index + 1}",
            )
            split_dataset_name = f"{base_dataset_name} / {split}"
            split_set_active = bool(active_split and split == active_split)

            def _split_progress(payload: dict[str, Any], *, split_index=index, split_name=split) -> None:
                payload = dict(payload or {})
                inner_percent = max(0, min(100, int(payload.get("percent", 0) or 0)))
                overall_percent = int(((split_index + (inner_percent / 100.0)) / len(splits)) * 100)

                message = str(payload.get("message", "") or "")
                payload["percent"] = overall_percent
                payload["message"] = f"Split {split_index + 1}/{len(splits)} ({split_name}): {message}"
                payload["phase"] = f"{payload.get('phase', 'import')}:{split_name}"

                self._set_progress_from_worker(payload)

            result = self._import_single_split(
                mode=mode,
                split=split,
                dataset_id=split_dataset_id,
                dataset_name=split_dataset_name,
                image_column=image_column,
                label_column=label_column,
                set_active=split_set_active,
                cancel_token=cancel_token,
                progress_callback=_split_progress,
            )

            if result.get("cancelled"):
                return {
                    "cancelled": True,
                    "multi_split": True,
                    "results": results,
                    "rows": total_rows,
                    "total_rows": total_rows,
                }

            results.append(result)
            total_rows += int(result.get("rows", 0) or 0)

            preview = result.get("preview")
            if preview is not None:
                try:
                    preview_df = preview.copy()
                    preview_df.insert(0, "astronomical_dataset_id", result.get("dataset_id", ""))
                    preview_df.insert(1, "imported_split", split)
                    previews.append(preview_df.head(10))
                except Exception:
                    pass

        summary_preview = pd.DataFrame(
            [
                {
                    "dataset_id": result.get("dataset_id", ""),
                    "split": result.get("split", ""),
                    "rows": result.get("rows", 0),
                    "backend": result.get("backend", ""),
                    "download_method": result.get("download_method", ""),
                    "set_active": result.get("split", "") == active_split if active_split else False,
                }
                for result in results
            ]
        )

        combined_preview = summary_preview
        if previews:
            try:
                combined_preview = pd.concat([summary_preview] + previews, ignore_index=True, sort=False)
            except Exception:
                combined_preview = summary_preview

        self._set_progress_from_worker(
            {
                "phase": "done",
                "completed": len(results),
                "total": len(splits),
                "percent": 100,
                "message": f"Registered {len(results)} split dataset(s), {total_rows} total row(s).",
            }
        )

        return {
            "multi_split": True,
            "dataset_id": base_dataset_id,
            "name": base_dataset_name,
            "rows": total_rows,
            "total_rows": total_rows,
            "split_count": len(results),
            "requested_splits": splits,
            "active_split": active_split,
            "results": results,
            "backend": "multi_split",
            "download_method": "multi_split",
            "preview": combined_preview,
        }

    def _on_import_done(self, result: dict[str, Any]) -> None:
        self._set_busy(False)

        if result.get("cancelled"):
            self._finish_progress("Import cancelled.")
            self._set_warning("Import cancelled.")
            return

        preview = result.get("preview")
        if preview is not None:
            try:
                self.import_preview.object = preview
            except Exception:
                self.import_preview.object = pd.DataFrame()

        if result.get("multi_split"):
            rows = int(result.get("total_rows", result.get("rows", 0)) or 0)
            split_count = int(result.get("split_count", 0) or 0)
            active_split = result.get("active_split") or "None"
            results = list(result.get("results") or [])

            dataset_lines = "\n".join(
                (
                    f"- `{self._escape(item.get('dataset_id', ''))}` "
                    f"from split `{self._escape(item.get('split', ''))}` "
                    f"with `{self._escape(item.get('rows', 0))}` rows"
                    f"{' **active**' if item.get('split', '') == result.get('active_split') else ''}"
                )
                for item in results
            )

            self.import_summary.object = (
                f"Registered `{split_count}` split dataset(s) with `{rows}` total rows.  \n"
                f"Active split after import: `{self._escape(active_split)}`\n\n"
                f"**Imported datasets:**\n\n{dataset_lines}"
            )

            self._finish_progress(
                f"Registered {split_count} split dataset(s), {rows} total row(s)."
            )
            self._set_success(
                f"Registered {split_count} split dataset(s), {rows} total row(s)."
            )
            self._set_active_step(3)
            return

        dataset_id = result.get("dataset_id")
        rows = result.get("rows")
        backend = result.get("backend", "unknown")
        method = result.get("download_method", "unknown")
        note = result.get("import_note")

        message = (
            f"Registered `{self._escape(dataset_id)}` with `{rows}` rows.  \n"
            f"Backend: `{self._escape(backend)}`  \n"
            f"Download method: `{self._escape(method)}`"
        )

        if note:
            message += f"\n\n**Note:** {self._escape(note)}"

        self.import_summary.object = message
        self._finish_progress(f"Registered {dataset_id}.")
        self._set_success(f"Registered `{dataset_id}` with {rows} rows.")
        self._set_active_step(3)

    # ------------------------------------------------------------------
    # Job/progress helpers
    # ------------------------------------------------------------------

    def _run_job(self, func, *, title: str, key: str, on_done, on_error) -> None:
        self._set_busy(True, title)
        self._set_loading(title + "…")
        self._start_progress(title + "…")

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                result = func(cancel_token=None)
                on_done(result)
            except Exception as exc:
                on_error(exc)
            return

        handle = jobs.submit(
            func,
            title=title,
            key=key,
            on_done=self._job_done_on_ui(on_done),
            on_error=self._job_error_on_ui(on_error),
        )
        self._job_handles.append(handle)

    def _schedule_ui(self, callback, *args, **kwargs) -> None:
        def _run() -> None:
            if getattr(self, "_disposed", False):
                return
            try:
                callback(*args, **kwargs)
            except Exception:
                traceback.print_exc()

        doc = getattr(self, "_doc", None) or pn.state.curdoc
        if doc is not None:
            self._doc = doc

        if doc is None:
            _run()
            return

        try:
            doc.add_next_tick_callback(_run)
        except Exception:
            _run()

    def _job_done_on_ui(self, on_done):
        def _wrapped(result):
            self._schedule_ui(on_done, result)

        return _wrapped

    def _job_error_on_ui(self, on_error):
        def _wrapped(exc):
            self._schedule_ui(on_error, exc)

        return _wrapped

    def _set_progress_from_worker(self, payload: dict[str, Any]) -> None:
        with self._progress_lock:
            self._progress_state.update(
                {
                    "active": True,
                    "phase": payload.get("phase", self._progress_state.get("phase", "")),
                    "completed": int(payload.get("completed", self._progress_state.get("completed", 0)) or 0),
                    "total": int(payload.get("total", self._progress_state.get("total", 0)) or 0),
                    "percent": int(payload.get("percent", self._progress_state.get("percent", 0)) or 0),
                    "message": str(payload.get("message", self._progress_state.get("message", "")) or ""),
                    "current_file": str(payload.get("current_file", "") or ""),
                }
            )

    def _start_progress(self, message: str) -> None:
        with self._progress_lock:
            self._progress_state.update(
                {
                    "active": True,
                    "phase": "starting",
                    "completed": 0,
                    "total": 0,
                    "percent": 0,
                    "message": message,
                    "current_file": "",
                }
            )

    def _finish_progress(self, message: str = "Done") -> None:
        with self._progress_lock:
            self._progress_state.update(
                {
                    "active": True,
                    "phase": "done",
                    "percent": 100,
                    "message": message,
                    "current_file": "",
                }
            )

    def _fail_progress(self, message: str) -> None:
        with self._progress_lock:
            self._progress_state.update(
                {
                    "active": True,
                    "phase": "error",
                    "message": message,
                    "current_file": "",
                }
            )

    def _sync_progress_ui(self) -> None:
        with self._progress_lock:
            state = dict(self._progress_state)

        active = bool(state.get("active", False))
        percent = max(0, min(100, int(state.get("percent", 0) or 0)))
        completed = int(state.get("completed", 0) or 0)
        total = int(state.get("total", 0) or 0)
        message = str(state.get("message", "") or "")
        current_file = str(state.get("current_file", "") or "")
        phase = str(state.get("phase", "") or "")

        self.progress_bar.visible = active
        self.progress_text.visible = active

        if not active:
            self.progress_text.object = ""
            return

        self.progress_bar.value = percent

        count_text = ""
        if total > 0:
            count_text = f" ({completed}/{total})"

        file_text = ""
        if current_file:
            short_file = current_file
            if len(short_file) > 90:
                short_file = "…" + short_file[-89:]
            file_text = f"\n`{self._escape(short_file)}`"

        self.progress_text.object = (
            f"**{percent}%** — `{self._escape(phase)}`{count_text}: "
            f"{self._escape(message)}"
            f"{file_text}"
        )

    # ------------------------------------------------------------------
    # General helpers
    # ------------------------------------------------------------------

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

    def _selected_mode(self) -> str:
        selected = str(self.import_mode.value or "auto")
        if selected in {"files", "builder"}:
            return selected
        return self._details_mode or "files"

    def _mode_needs_reinspect(self) -> bool:
        selected = str(self.import_mode.value or "auto")
        if selected not in {"files", "builder"}:
            return False
        return self._details is not None and self._details_mode is not None and selected != self._details_mode

    @staticmethod
    def _mode_label(mode: str | None) -> str:
        if mode == "builder":
            return "HF Datasets builder"
        if mode == "files":
            return "Fast Hub files"
        return "Auto recommendation"

    def _config(self) -> str | None:
        value = str(self.config_select.value or "").strip()
        if value in {"", "__default__", "default"}:
            return None
        return value

    def _token(self) -> str | None:
        value = str(self.token.value or "").strip()
        return value or None

    @staticmethod
    def _empty_to_none(value: Any) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        if value in {"", "__default__", "None", "null"}:
            return None
        return value

    @staticmethod
    def _escape(value: Any) -> str:
        return html.escape(str(value))

    @staticmethod
    def _option_values(options: Any) -> list[Any]:
        if isinstance(options, dict):
            return list(options.values())
        return list(options or [])

    def _set_select_options(self, widget: Any, options: Any, *, preferred: Any = None) -> None:
        widget.options = options
        values = self._option_values(options)

        if preferred in values:
            widget.value = preferred
        elif values:
            widget.value = values[0]
        else:
            widget.value = None

    def _suggest_dataset_ids(self, *, overwrite_generated_only: bool = False) -> None:
        repo_id = str(self.dataset_select.value or "hf_dataset")
        config = self._config() or self._details_mode or "files"

        if bool(getattr(self, "download_all_splits", None) and self.download_all_splits.value):
            split_part = "all_splits"
        else:
            split_part = str(self.split_select.value or "train")

        suggested_id = slugify(
            f"hf_{repo_id}_{config}_{split_part}",
            fallback="hf_image_dataset",
        )

        if split_part == "all_splits":
            suggested_name = f"HF {repo_id} [{config} / all splits]"
        else:
            suggested_name = f"HF {repo_id} [{config} / {split_part}]"

        should_set_id = not self.dataset_id.value
        should_set_name = not self.dataset_name.value

        if overwrite_generated_only:
            should_set_id = should_set_id or self.dataset_id.value == self._last_suggested_dataset_id
            should_set_name = should_set_name or self.dataset_name.value == self._last_suggested_dataset_name

        if should_set_id:
            self.dataset_id.value = suggested_id
            self._last_suggested_dataset_id = suggested_id

        if should_set_name:
            self.dataset_name.value = suggested_name
            self._last_suggested_dataset_name = suggested_name

    def _set_busy(self, busy: bool, message: str = "") -> None:
        suffix = f" — {self._escape(message)}" if message else ""
        if busy:
            self.activity.object = f"<span><b>Activity:</b> working{suffix}</span>"
        else:
            self.activity.object = "<span><b>Activity:</b> idle</span>"

        for button in (
            self.search_button,
            self.inspect_button,
            self.preview_button,
            self.import_button,
        ):
            button.disabled = bool(busy)

    def _set_status(self, message: str, status_type: str = "info") -> None:
        status_type = status_type or "info"
        self.status.object = message
        self.status.css_classes = [
            "al-hf-status-line",
            f"al-hf-status-{status_type}",
        ]

    def _set_loading(self, message: str) -> None:
        self._set_status(message, "primary")

    def _set_success(self, message: str) -> None:
        self._set_status(message, "success")

    def _set_warning(self, message: str) -> None:
        self._set_status(message, "warning")

    def _set_error(self, message: str) -> None:
        self._set_busy(False)
        self._set_status(message, "danger")

    def _on_error(self, exc: BaseException) -> None:
        message = str(exc)
        self._fail_progress(message)
        self._set_error(message)