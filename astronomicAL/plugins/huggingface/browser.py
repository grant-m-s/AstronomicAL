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


def _install_hf_browser_frontend_guard_once() -> None:
    """
    Plugin-side guard for the platform Add Panel menu.

    The platform menu is intentionally appended to document.body with class
    .al-hmenu-body-popover. The Hugging Face browser should not intercept
    pointer events while that menu is open.

    This leaves menu.py unchanged.
    """

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

        .al-hf-browser-root .al-hf-browser-sticky {
            position: sticky !important;
            top: 0 !important;
            z-index: 1 !important;
            background: white !important;
        }

        /*
         * Main fix:
         * While the platform menu popover exists, make the HF browser ignore
         * pointer events so the body-level menu receives the click.
         *
         * body:has(...) handles modern browsers directly.
         * body.al-hmenu-open-from-hf is set by the small observer script below.
         */
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
    """
    Tiny frontend observer installed by the HF browser only.

    It watches for the platform menu's body-level popover and toggles a body
    class. This supports browsers/environments where :has(...) is unreliable.
    """

    html_text = """
    <script>
    (function () {
        if (window.__AL_HF_MENU_POINTER_GUARD_INSTALLED__) {
            return;
        }

        window.__AL_HF_MENU_POINTER_GUARD_INSTALLED__ = true;

        function syncAstronomicalMenuState() {
            try {
                const open = !!document.querySelector('.al-hmenu-body-popover');
                document.body.classList.toggle('al-hmenu-open-from-hf', open);
            } catch (err) {
                // Keep this guard silent in production.
            }
        }

        const observer = new MutationObserver(function () {
            syncAstronomicalMenuState();
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        document.addEventListener('pointerdown', function () {
            syncAstronomicalMenuState();
        }, true);

        document.addEventListener('click', function () {
            window.setTimeout(syncAstronomicalMenuState, 0);
        }, true);

        document.addEventListener('keydown', function () {
            window.setTimeout(syncAstronomicalMenuState, 0);
        }, true);

        syncAstronomicalMenuState();
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

_HF_BROWSER_CSS_INSTALLED = False


def _install_hf_browser_css_once() -> None:
    """
    Keep the Hugging Face browser visually contained inside its own grid tile.

    This is intentionally scoped to .al-hf-browser-root so it does not alter
    the platform menu, React grid, or other plugins.
    """

    global _HF_BROWSER_CSS_INSTALLED

    if _HF_BROWSER_CSS_INSTALLED:
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

        .al-hf-browser-root .bk-Column,
        .al-hf-browser-root .bk-Row,
        .al-hf-browser-root .bk-panel-models-esm-ReactComponent {
            z-index: auto !important;
        }

        .al-hf-browser-root .al-hf-browser-sticky {
            position: sticky !important;
            top: 0 !important;
            z-index: 2 !important;
            background: white !important;
        }
        """
    )

    _HF_BROWSER_CSS_INSTALLED = True


class HuggingFaceBrowserPanel:
    state_version = 3

    def __init__(self, context: Any) -> None:
        self.context = context
        _install_hf_browser_frontend_guard_once()
        self._menu_pointer_guard = _make_hf_menu_pointer_guard_pane()

        _install_hf_browser_css_once()

        self._disposed = False
        self._doc = pn.state.curdoc
        self._ui_lock = threading.RLock()
        self._job_handles: list[Any] = []
        self._search_df = pd.DataFrame()
        self._details = None

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
        self.trust_remote_code = pn.widgets.Checkbox(
            name="Trust remote dataset code; only used by legacy builder import",
            value=False,
        )

        self.import_mode = pn.widgets.RadioButtonGroup(
            name="Import mode",
            options={
                "Fast Hub files": "files",
                "HF Datasets builder, slower": "builder",
            },
            value="builder",
            button_type="default",
            sizing_mode="stretch_width",
        )

        self.search_button = pn.widgets.Button(
            name="Search datasets",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.search_button.on_click(self._on_search_clicked)

        self.dataset_select = pn.widgets.Select(
            name="Dataset",
            options=[],
            sizing_mode="stretch_width",
        )
        self.inspect_button = pn.widgets.Button(
            name="Inspect selected dataset",
            button_type="default",
            sizing_mode="stretch_width",
        )
        self.inspect_button.on_click(self._on_inspect_clicked)

        self.config_select = pn.widgets.Select(
            name="Config",
            options=["default"],
            value="default",
            sizing_mode="stretch_width",
        )
        self.split_select = pn.widgets.Select(
            name="Split",
            options=["train"],
            value="train",
            sizing_mode="stretch_width",
        )
        self.image_column = pn.widgets.Select(
            name="Image source",
            options={"Hub image files; lightweight": "__hf_file__"},
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
            name="ID column; legacy builder import only",
            options=[""],
            value="",
            sizing_mode="stretch_width",
        )

        self.preview_limit = pn.widgets.IntInput(
            name="Preview rows",
            value=8,
            start=1,
            end=50,
            sizing_mode="stretch_width",
        )
        self.max_import_rows = pn.widgets.IntInput(
            name="Max import rows; 0 = all selected split",
            value=100,
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

        self.preview_button = pn.widgets.Button(
            name="Preview selected split",
            button_type="default",
            sizing_mode="stretch_width",
        )
        self.preview_button.on_click(self._on_preview_clicked)

        self.import_button = pn.widgets.Button(
            name="Download/register selected split",
            button_type="success",
            sizing_mode="stretch_width",
        )
        self.import_button.on_click(self._on_import_clicked)

        self.activity = pn.pane.Markdown(
            "**Activity:** idle",
            sizing_mode="stretch_width",
        )
        self.status = pn.pane.Alert(
            "Search Hugging Face for image datasets.",
            alert_type="info",
            sizing_mode="stretch_width",
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

        self.sticky_status = pn.Column(
            self.activity,
            self.status,
            self.progress_bar,
            self.progress_text,
            sizing_mode="stretch_width",
            css_classes=["al-hf-browser-sticky"],
            styles={
                "position": "sticky",
                "top": "0px",
                "z-index": "2",
                "background": "white",
                "padding": "8px",
                "border": "1px solid #ddd",
                "box-shadow": "0 2px 6px rgba(0, 0, 0, 0.08)",
            },
        )

        self.results_table = pn.pane.DataFrame(
            pd.DataFrame(),
            sizing_mode="stretch_width",
            height=260,
        )
        self.details_pane = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
        )
        self.preview_grid = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
        )
        self.import_preview = pn.pane.DataFrame(
            None,
            sizing_mode="stretch_width",
            height=220,
        )

        self._view = pn.Column(
            self._menu_pointer_guard,
            pn.pane.Markdown("### Hugging Face Dataset Browser"),
            pn.pane.Alert(
                "Inspect uses Hub metadata and file-path sampling only. "
                "Preview downloads only the displayed images. "
                "Download/register is the first full import step.",
                alert_type="secondary",
                sizing_mode="stretch_width",
            ),
            self.sticky_status,
            pn.Row(self.query, self.task, self.sort, self.limit, sizing_mode="stretch_width"),
            self.token,
            self.trust_remote_code,
            self.import_mode,
            self.search_button,
            pn.pane.Markdown("#### Results"),
            self.results_table,
            self.dataset_select,
            self.inspect_button,
            pn.pane.Markdown("#### Dataset details"),
            self.details_pane,
            pn.Row(self.config_select, self.split_select, sizing_mode="stretch_width"),
            pn.Row(self.image_column, self.label_column, self.id_column, sizing_mode="stretch_width"),
            pn.Row(self.preview_limit, self.max_import_rows, sizing_mode="stretch_width"),
            pn.Row(self.dataset_id, self.dataset_name, sizing_mode="stretch_width"),
            pn.Row(self.write_parquet, self.set_active, sizing_mode="stretch_width"),
            pn.Row(self.preview_button, self.import_button, sizing_mode="stretch_width"),
            pn.pane.Markdown("#### Preview"),
            self.preview_grid,
            pn.pane.Markdown("#### Imported manifest preview"),
            self.import_preview,
            sizing_mode="stretch_both",
            css_classes=["al-hf-browser-root"],
            styles={
                "height": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "auto",
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

    def _schedule_ui(self, callback, *args, **kwargs) -> None:
        """
        Schedule a UI mutation on the Bokeh/Panel document thread.

        JobManager callbacks can run on worker threads. The Hugging Face browser
        must not mutate Panel widgets directly from those worker callbacks.
        """

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
            # Fallback for non-server/unit-test contexts.
            _run()
            return

        try:
            doc.add_next_tick_callback(_run)
        except Exception:
            # Last-resort fallback. Better to show the error than silently lose UI.
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
        """
        Thread-safe progress update from a background JobManager worker.

        Do not update Panel widgets directly from this method.
        """

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
        """
        Runs on Panel's event loop and mirrors thread-safe progress state into
        visible widgets.
        """

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

        if active:
            self.progress_bar.value = percent

            count_text = ""
            if total > 0:
                count_text = f" ({completed}/{total})"

            file_text = ""
            if current_file:
                short_file = current_file
                if len(short_file) > 90:
                    short_file = "…" + short_file[-89:]
                file_text = f"  \n`{html.escape(short_file)}`"

            self.progress_text.object = (
                f"**{percent}%** — `{html.escape(phase)}`{count_text}: "
                f"{html.escape(message)}"
                f"{file_text}"
            )
        else:
            self.progress_text.object = ""

    def panel(self):
        return self._view

    def dispose(self) -> None:
        self._disposed = True

        try:
            if getattr(self, "_progress_periodic", None) is not None:
                self._progress_periodic.stop()
        except Exception:
            pass

        for handle in self._job_handles:
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
            "trust_remote_code": self.trust_remote_code.value,
            "import_mode": self.import_mode.value,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

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
            "trust_remote_code": self.trust_remote_code,
            "import_mode": self.import_mode,
        }
        for key, widget in widget_map.items():
            if key in state:
                try:
                    widget.value = state[key]
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # User actions
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
        self.results_table.object = self._search_df

        if self._search_df.empty or "repo_id" not in self._search_df.columns:
            self.dataset_select.options = []
            self._set_warning("No datasets found.")
            return

        repo_ids = [str(value) for value in self._search_df["repo_id"].tolist()]
        self.dataset_select.options = repo_ids
        self.dataset_select.value = repo_ids[0]
        self._set_success(f"Found {len(repo_ids)} dataset(s). Select one to inspect.")

    def _on_inspect_clicked(self, event: Any) -> None:
        repo_id = str(self.dataset_select.value or "").strip()
        if not repo_id:
            self._set_error("Choose a dataset first.")
            return

        mode = str(getattr(self.import_mode, "value", "files") or "files")
        key = f"hf.inspect.{mode}:{repo_id}:{self.config_select.value}"

        self._run_job(
            self._inspect,
            title=f"Inspecting {repo_id} ({mode})",
            key=key,
            on_done=self._on_inspect_done,
            on_error=self._on_error,
        )

    def _inspect(self, *, cancel_token: Any = None):
        service = self._service()

        if self.import_mode.value == "builder":
            return service.get_dataset_details_builder(
                str(self.dataset_select.value),
                config_name=self._config(),
                token=self._token(),
                trust_remote_code=bool(self.trust_remote_code.value),
            )

        return service.get_dataset_details(
            str(self.dataset_select.value),
            token=self._token(),
            trust_remote_code=bool(self.trust_remote_code.value),
        )


    def _on_inspect_done(self, details) -> None:
        self._set_busy(False)
        self._finish_progress("Inspect complete.")
        self._details = details

        configs = details.configs or ["default"]
        self.config_select.options = configs
        self.config_select.value = configs[0]

        splits = details.splits_by_config.get(configs[0]) or ["train"]
        self.split_select.options = splits
        self.split_select.value = splits[0]

        features = details.features_by_config.get(configs[0], {})
        columns = list(features.keys())

        if self.import_mode.value == "builder":
            image_candidates = details.image_column_candidates or [
                col for col in columns
                if col.lower() in {"image", "img", "picture", "photo"}
            ] or ["image"]

            label_candidates = details.label_column_candidates or [
                col for col in columns
                if col.lower() in {"label", "labels", "target", "class", "category"}
            ]

            # Strong fallback: if a literal label column exists, select it by default.
            if "label" in columns and "label" not in label_candidates:
                label_candidates.insert(0, "label")
            if "labels" in columns and "labels" not in label_candidates:
                label_candidates.append("labels")

            self.image_column.options = image_candidates
            self.image_column.value = image_candidates[0]

            self.label_column.options = [""] + columns

            if label_candidates:
                self.label_column.value = label_candidates[0]
            elif "label" in columns:
                self.label_column.value = "label"
            elif "labels" in columns:
                self.label_column.value = "labels"
            else:
                self.label_column.value = ""

            self.id_column.options = [""] + columns
            self.id_column.value = ""
        else:
            self.image_column.options = {"Hub image files; lightweight": "__hf_file__"}
            self.image_column.value = "__hf_file__"

            self.label_column.options = {
                "Inferred from path/folder if available": "__hf_path_label__",
                "No label": "",
            }
            self.label_column.value = "__hf_path_label__"

            self.id_column.options = [""]
            self.id_column.value = ""

        self.details_pane.object = self._details_markdown(details, config=configs[0])
        self._suggest_dataset_ids()

        if self.import_mode.value == "builder":
            if self.label_column.value:
                self._set_warning(
                    f"Dataset inspected using Hugging Face Datasets builder. "
                    f"Selected label column `{self.label_column.value}`."
                )
            else:
                self._set_warning(
                    "Dataset inspected using Hugging Face Datasets builder, but no "
                    "label column was found. The dataset may not expose digit/class "
                    "labels through the standard HF Datasets interface."
                )
        elif details.error:
            self._set_warning(
                "Dataset inspected using lightweight Hub file sampling. "
                "Labels are only inferred from paths."
            )
        else:
            self._set_success(
                "Dataset inspected using lightweight Hub file sampling. "
                "Preview will download only the displayed images."
            )

    def _on_preview_clicked(self, event: Any) -> None:
        repo_id = str(self.dataset_select.value or "").strip()
        if not repo_id:
            self._set_error("Choose a dataset first.")
            return

        self._run_job(
            self._preview,
            title=f"Previewing {repo_id}",
            key=(
                f"hf.preview.files:{repo_id}:{self.split_select.value}:"
                f"{self.preview_limit.value}"
            ),
            on_done=self._on_preview_done,
            on_error=self._on_error,
        )

    def _preview(self, *, cancel_token: Any = None):
        service = self._service()

        if self.import_mode.value == "builder":
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
        self.preview_grid.object = html_text
        self._set_success(
            f"Preview loaded. It downloaded only up to {int(self.preview_limit.value or 8)} displayed image file(s)."
        )

    def _on_import_clicked(self, event: Any) -> None:
        repo_id = str(self.dataset_select.value or "").strip()
        if not repo_id:
            self._set_error("Choose a dataset first.")
            return

        self._run_job(
            self._import,
            title=f"Downloading/registering {repo_id}",
            key=(
                f"hf.import.files:{repo_id}:{self.split_select.value}:"
                f"{self.max_import_rows.value}"
            ),
            on_done=self._on_import_done,
            on_error=self._on_error,
        )

    def _import(self, *, cancel_token: Any = None):
        image_column = self._empty_to_none(self.image_column.value)
        label_column = self._empty_to_none(self.label_column.value)

        effective_mode = self.import_mode.value

        if effective_mode == "builder":
            service = self._service()
            config = self.config_select.value or "default"

            if service.selected_builder_label_is_non_semantic(
                self._details,
                config=str(config),
                label_column=label_column,
            ):
                effective_mode = "files"
                image_column = "__hf_file__"
                label_column = "__hf_path_label__"
            else:
                image_column = image_column or "image"

        if effective_mode == "files":
            image_column = "__hf_file__"
            label_column = "__hf_path_label__"

        result = import_hf_image_dataset_as_manifest(
            self.context,
            repo_id=str(self.dataset_select.value),
            config_name=self._config(),
            split=str(self.split_select.value or "train"),
            dataset_id=self._empty_to_none(self.dataset_id.value),
            dataset_name=self._empty_to_none(self.dataset_name.value),
            image_column=image_column,
            label_column=label_column,
            id_column=self._empty_to_none(self.id_column.value),
            max_rows=int(self.max_import_rows.value or 0),
            token=self._token(),
            trust_remote_code=bool(self.trust_remote_code.value),
            write_parquet=bool(self.write_parquet.value),
            set_active=bool(self.set_active.value),
            cancel_token=cancel_token,
            progress_callback=self._set_progress_from_worker,
        )

        if effective_mode == "files" and self.import_mode.value == "builder":
            result["import_note"] = (
                "Builder labels looked non-semantic, for example data_0/data_1 shard labels. "
                "Used fast Hub-file import instead."
            )

        return result

    def _on_import_done(self, result: dict[str, Any]) -> None:
        self._set_busy(False)

        if result.get("cancelled"):
            self._finish_progress("Import cancelled.")
            self._set_warning("Import cancelled.")
            return

        preview = result.pop("preview", None)
        if preview is not None:
            self.import_preview.object = preview

        note = result.get("import_note")
        method = result.get("download_method", "unknown")

        message = (
            f"Registered `{result.get('dataset_id')}` with "
            f"{result.get('rows')} rows using `{result.get('backend')}`. "
            f"Download method: `{method}`."
        )

        self._finish_progress(message)

        if note:
            self._set_warning(message + " " + note)
        else:
            self._set_success(message)

    # ------------------------------------------------------------------
    # Helpers
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

    def _run_job(self, func, *, title: str, key: str, on_done, on_error) -> None:
        """
        Submit a background job, but always marshal UI callbacks back onto the
        Bokeh/Panel document thread.

        This is the main plugin-side fix for menu instability after the HF browser
        has run jobs.
        """

        self._set_busy(True, title)
        self._set_loading(title + "…")

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

    def _details_markdown(self, details, *, config: str) -> str:
        features = details.features_by_config.get(config, {})
        splits = details.splits_by_config.get(config, [])

        feature_lines = "\n".join(
            f"- `{html.escape(str(name))}`: `{html.escape(str(kind))}`"
            for name, kind in features.items()
        ) or "_No features found._"

        sample_lines = ""
        if getattr(details, "sample_files", None):
            sample_lines = "\n\n**Sample image files:**\n\n" + "\n".join(
                f"- `{html.escape(file.path)}` → split `{html.escape(file.split)}`"
                + (f", label `{html.escape(file.label)}`" if file.label else "")
                for file in details.sample_files[:8]
            )

        warning_text = ""
        if details.error:
            warning_text = (
                "\n\n**Notes:**\n\n"
                + "\n".join(
                    f"- {html.escape(line)}"
                    for line in details.error.splitlines()
                    if line.strip()
                )
            )

        source_heading = (
            "Available builder columns"
            if self.import_mode.value == "builder"
            else "Available lightweight sources"
        )

        return (
            f"**Dataset:** `{html.escape(details.repo_id)}`  \n"
            f"**Configs:** `{', '.join(details.configs)}`  \n"
            f"**Current config:** `{html.escape(config)}`  \n"
            f"**Inferred splits:** `{', '.join(splits)}`  \n"
            f"**Sampled image files:** `{getattr(details, 'scanned_file_count', 0)}`  \n\n"
            f"**{source_heading}:**\n\n{feature_lines}"
            f"{sample_lines}"
            f"{warning_text}"
        )

    def _suggest_dataset_ids(self) -> None:
        repo_id = str(self.dataset_select.value or "hf_dataset")
        config = self._config() or "files"
        split = str(self.split_select.value or "train")
        suggested = slugify(f"hf_{repo_id}_{config}_{split}")
        if not self.dataset_id.value:
            self.dataset_id.value = suggested
        if not self.dataset_name.value:
            self.dataset_name.value = f"HF {repo_id} [{config} / {split}]"

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
        return value or None

    def _set_busy(self, busy: bool, message: str = "") -> None:
        suffix = f" — {html.escape(message)}" if message else ""
        self.activity.object = "**Activity:** working" + suffix if busy else "**Activity:** idle"

        for button in (
            self.search_button,
            self.inspect_button,
            self.preview_button,
            self.import_button,
        ):
            button.disabled = bool(busy)

    def _set_loading(self, message: str) -> None:
        self.status.alert_type = "primary"
        self.status.object = message

    def _set_success(self, message: str) -> None:
        self.status.alert_type = "success"
        self.status.object = message

    def _set_warning(self, message: str) -> None:
        self.status.alert_type = "warning"
        self.status.object = message

    def _set_error(self, message: str) -> None:
        self._set_busy(False)
        self.status.alert_type = "danger"
        self.status.object = message

    def _on_error(self, exc: BaseException) -> None:
        message = str(exc)
        self._fail_progress(message)
        self._set_error(message)