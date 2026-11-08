from __future__ import annotations

import html
import threading
import time
import traceback
from collections import OrderedDict
from typing import Any

import pandas as pd
import panel as pn

from .importer import (
    HF_FIFTYONE_GROUND_TRUTH, HF_LABEL_SEMANTICS_VERSION, HF_PATH_LABEL, find_registered_hf_dataset,
    import_hf_image_dataset_as_manifest, slugify,
)
from .service import HFDatasetAccessStatus, HFDatasetDetails, HuggingFaceDatasetService
from .manifest_cache import cache_identity
from .structured_annotations import HFStructuredAnnotationProfile
from .structured_video_importer import import_structured_video_dataset
from .video_importer import EPIC_KITCHENS_REPO_ID, import_repository_videos
from .styles import (
    BUTTON_PRIMARY_STYLESHEET, BUTTON_SECONDARY_STYLESHEET, BUTTON_SUCCESS_STYLESHEET, CHECKBOX_STYLESHEET,
    FORM_CHECKBOX_STYLESHEET, EMPTY_STATE_STYLES, INSPECTOR_STYLES, NUMBER_INPUT_STYLESHEET,
    PROGRESS_STYLESHEET, RESULT_BUTTON_STYLESHEET, RESULTS_LIST_STYLES, ROOT_STYLES,
    SEARCH_BAR_INPUT_STYLESHEET, SEARCH_INPUT_STYLESHEET, SELECT_STYLESHEET, SHELL_STYLES,
    SUBSURFACE_STYLES, SURFACE_STYLES, TOOLBAR_STYLES,
)

DEFAULT_TASKS = OrderedDict(
    {
        "All datasets": "", "Image classification": "image-classification",
        "Object detection": "object-detection", "Semantic segmentation": "semantic-segmentation",
        "Image-to-text": "image-to-text", "Depth estimation": "depth-estimation",
        "Text classification": "text-classification", "Audio classification": "audio-classification",
        "Automatic speech recognition": "automatic-speech-recognition",
        "Question answering": "question-answering", "Tabular classification": "tabular-classification",
        "Tabular regression": "tabular-regression",
    }
)

SORT_OPTIONS = OrderedDict(
    {"Most downloaded": "downloads", "Most liked": "likes", "Recently updated": "last_modified"}
)

STRATEGY_OPTIONS = OrderedDict(
    {
        "Automatic": "auto", "Dataset Viewer metadata": "viewer", "Structured dataset rows": "builder",
        "Repository image files": "files",
    }
)

class HuggingFaceBrowserPanel:
    """Browse Hugging Face datasets via metadata, bounded previews and image/video imports."""

    state_version = 15
    nominal_structured_playback_fps = 5.0

    def __init__(self, context: Any, manager: Any = None) -> None:
        self.context = context
        self.manager = manager or getattr(context, "plugins", None)
        self._disposed = False
        self._doc = pn.state.curdoc
        self._job_handles: list[Any] = []
        self._active_job_handle = None
        self._active_job_cancellable = False
        self._cancel_requested = False
        self._busy = False
        self._updating_widgets = False
        self._search_df = pd.DataFrame()
        self._details: HFDatasetDetails | None = None
        self._details_mode: str | None = None
        self._details_config: str | None = None
        self._access_status: HFDatasetAccessStatus | None = None
        self._restored_state: dict[str, Any] = {}
        self._last_suggested_dataset_id = ""
        self._last_suggested_dataset_name = ""
        self._inspection_generation = 0
        self._pending_inspection_repo: str | None = None

        self._progress_lock = threading.RLock()
        self._progress_state: dict[str, Any] = {
            "active": False, "phase": "idle", "completed": 0, "total": 0, "percent": -1,
            "progress_known": False, "progress_basis": "indeterminate", "message": "", "current_file": "",
            "severity": "info", "retry_after_seconds": None,
        }
        self._progress_periodic = None
        self._last_progress_notice = ""
        self._progress_started_at = 0.0
        self._progress_phase_started_at = 0.0
        self._structured_frame_fps_nominal = False

        self._build_widgets()
        self._view = self._build_view()
        self._set_status("Search by topic or paste an exact Hugging Face dataset ID.", "info")

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
            "state_version": self.state_version, "query": self.query.value, "task": self.task.value,
            "sort": self.sort.value, "limit": self.limit.value, "selected_repo": self.selected_repo,
            "revision": (
                str(getattr(self._details, "revision", "") or "") if self._details is not None
                else str(self._restored_state.get("revision") or "")
            ), "config": self.config_select.value, "splits": list(self.split_choices.value or []),
            "preview_split": self.preview_split.value, "preview_limit": self.preview_limit.value,
            "strategy": self.strategy.value, "image_column": self.image_column.value,
            "label_column": self.label_column.value, "id_column": self.id_column.value,
            "dataset_id": self.dataset_id.value, "dataset_name": self.dataset_name.value,
            "max_rows": self.max_rows.value, "write_parquet": self.write_parquet.value,
            "active_after_import": self.active_after_import.value,
            "trust_remote_code": self.trust_remote_code.value,
            "annotation_variant": self.annotation_variant.value,
            "structured_media_root": self.structured_media_root.value,
            "structured_frame_fps": self.structured_frame_fps.value,
            "structured_frame_fps_nominal": bool(self._structured_frame_fps_nominal),
            "repository_video_prefix": self.repository_video_prefix.value,
            "repository_video_limit": self.repository_video_limit.value,
            "repository_video_active": self.repository_video_active.value,
            "repository_video_annotations": self.repository_video_annotations.value,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        self._restored_state = dict(state)
        self._structured_frame_fps_nominal = bool(state.get("structured_frame_fps_nominal", False))

        for key, widget in {
            "query": self.query, "task": self.task, "sort": self.sort, "limit": self.limit,
            "preview_limit": self.preview_limit, "strategy": self.strategy, "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name, "max_rows": self.max_rows,
            "write_parquet": self.write_parquet, "trust_remote_code": self.trust_remote_code,
            "annotation_variant": self.annotation_variant,
            "structured_media_root": self.structured_media_root,
            "structured_frame_fps": self.structured_frame_fps,
            "repository_video_prefix": self.repository_video_prefix,
            "repository_video_limit": self.repository_video_limit,
            "repository_video_active": self.repository_video_active,
            "repository_video_annotations": self.repository_video_annotations,
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
            # Restored layouts already contain the exact repository identity.  Re-run
            # inspection immediately instead of leaving a half-restored panel that
            # requires a manual "Re-inspect" click.  Inspection itself is job-backed
            # and cache-first, so this remains non-blocking with the real JobManager.
            self._select_repo(selected_repo, inspect=True)

    # ------------------------------------------------------------------
    # Widget and layout construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        self.query = pn.widgets.TextInput(
            name="", value="cifar10", placeholder="Search a topic or paste owner/dataset", height=38,
            sizing_mode="stretch_width", stylesheets=[SEARCH_BAR_INPUT_STYLESHEET],
        )
        self.search_button = pn.widgets.Button(
            name="Search", icon="search", button_type="primary", width=118, height=38,
            stylesheets=[BUTTON_PRIMARY_STYLESHEET],
        )
        self.search_button.on_click(self._on_search_clicked)
        self.query.param.watch(self._on_search_clicked, "enter_pressed")

        self.search_options_toggle = pn.widgets.Toggle(
            name="Search options", value=False, button_type="default", width=126, height=38,
            stylesheets=[BUTTON_SECONDARY_STYLESHEET],
        )
        self.search_options_toggle.param.watch(self._on_search_options_toggled, "value")

        self.task = pn.widgets.Select(
            name="Dataset type", options=DEFAULT_TASKS, value="", sizing_mode="stretch_width",
            stylesheets=[SELECT_STYLESHEET],
        )
        self.sort = pn.widgets.Select(
            name="Sort results", options=SORT_OPTIONS, value="downloads", sizing_mode="stretch_width",
            stylesheets=[SELECT_STYLESHEET],
        )
        self.limit = pn.widgets.IntInput(
            name="Maximum results", value=30, start=1, end=200, sizing_mode="stretch_width",
            stylesheets=[NUMBER_INPUT_STYLESHEET],
        )
        self.token = pn.widgets.PasswordInput(
            name="Hugging Face token", placeholder="Optional; your saved HF login is used when blank",
            sizing_mode="stretch_width", stylesheets=[SEARCH_INPUT_STYLESHEET],
        )
        self.token.param.watch(self._on_token_changed, "value")

        self._result_items: dict[str, Any] = {}
        self.results_list = pn.Column(
            self._results_empty_html("Search Hugging Face to see matching datasets."),
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), styles=RESULTS_LIST_STYLES,
        )
        self.results_meta = pn.pane.HTML(
            self._results_meta_html("No search yet"), sizing_mode="stretch_width", margin=(0, 0, 0, 0),
        )

        self.selected_pane = pn.pane.HTML(
            self._selected_html("No dataset selected"), sizing_mode="stretch_width", height=42,
            margin=(0, 0, 0, 0),
        )
        self.inspect_button = pn.widgets.Button(
            name="Refresh", icon="refresh", button_type="default", height=38, width=104,
            margin=(0, 0, 0, 0), disabled=True, stylesheets=[BUTTON_SECONDARY_STYLESHEET],
        )
        self.inspect_button.on_click(self._on_inspect_clicked)

        self.access_check_button = pn.widgets.Button(
            name="Re-check access", icon="shield-check", button_type="default", height=38, width=142,
            margin=(0, 0, 0, 0), visible=False, stylesheets=[BUTTON_SECONDARY_STYLESHEET],
        )
        self.access_check_button.on_click(self._on_access_check_clicked)
        self.access_pane = pn.pane.HTML("", sizing_mode="stretch_width", margin=(0, 0, 0, 0))

        self.cache_pane = pn.pane.HTML("", sizing_mode="stretch_width", margin=(0, 0, 0, 0))
        self.structure_pane = pn.pane.HTML("", sizing_mode="stretch_width", margin=(0, 0, 0, 0))
        self.notice_pane = pn.pane.HTML("", sizing_mode="stretch_width", margin=(0, 0, 0, 0))

        self.config_select = pn.widgets.Select(
            name="Configuration", options=["default"], value="default", sizing_mode="stretch_width",
            visible=False, margin=(8, 0, 0, 0), stylesheets=[SELECT_STYLESHEET],
        )
        self.config_select.param.watch(self._on_config_changed, "value")

        self.split_choices = pn.widgets.CheckBoxGroup(
            name="Splits to import", options=[], value=[], inline=True, min_height=38,
            sizing_mode="stretch_width", margin=(8, 0, 0, 0), stylesheets=[CHECKBOX_STYLESHEET],
        )
        self.split_choices.param.watch(self._on_split_choices_changed, "value")

        self.preview_split = pn.widgets.Select(
            name="Preview split", options=[], height=54, sizing_mode="stretch_width", margin=(0, 0, 0, 0),
            stylesheets=[SELECT_STYLESHEET],
        )
        self.preview_limit = pn.widgets.IntInput(
            name="Rows", value=10, start=1, end=50, height=54, sizing_mode="stretch_width",
            margin=(0, 0, 0, 0), stylesheets=[NUMBER_INPUT_STYLESHEET],
        )
        self.preview_button = pn.widgets.Button(
            name="Preview", icon="photo", button_type="default", width=92, height=38, margin=(16, 0, 0, 0),
            disabled=True, stylesheets=[BUTTON_SECONDARY_STYLESHEET],
        )
        self.preview_button.on_click(self._on_preview_clicked)
        self.preview_grid = pn.pane.HTML("", sizing_mode="stretch_width", margin=(0, 0, 0, 0))

        self.active_after_import = pn.widgets.Select(
            name="Active dataset after import", options=OrderedDict({"Do not change active dataset": ""}),
            value="", height=58, sizing_mode="stretch_width", margin=(8, 0, 0, 0),
            stylesheets=[SELECT_STYLESHEET],
        )

        self.dataset_id = pn.widgets.TextInput(
            name="Dataset ID base", value="", placeholder="Generated automatically", height=58,
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), stylesheets=[SEARCH_INPUT_STYLESHEET],
        )
        self.dataset_name = pn.widgets.TextInput(
            name="Dataset name base", value="", placeholder="Generated automatically", height=58,
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), stylesheets=[SEARCH_INPUT_STYLESHEET],
        )
        self.max_rows = pn.widgets.IntInput(
            name="Maximum rows per split; 0 = all", value=0, start=0, height=58,
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), stylesheets=[NUMBER_INPUT_STYLESHEET],
        )
        self.write_parquet = pn.widgets.Checkbox(
            name="Store the AstronomicAL manifest as Parquet", value=True, height=38,
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), stylesheets=[FORM_CHECKBOX_STYLESHEET],
        )
        self.strategy = pn.widgets.Select(
            name="Import strategy", options=STRATEGY_OPTIONS, value="auto", height=58,
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), stylesheets=[SELECT_STYLESHEET],
        )
        self.strategy.param.watch(self._on_strategy_changed, "value")
        self.trust_remote_code = pn.widgets.Checkbox(
            name="Trust repository dataset code", value=False, height=38, sizing_mode="stretch_width",
            margin=(0, 0, 0, 0), stylesheets=[FORM_CHECKBOX_STYLESHEET],
        )
        self.image_column = pn.widgets.Select(
            name="Image column", options=[], height=58, sizing_mode="stretch_width", margin=(0, 0, 0, 0),
            stylesheets=[SELECT_STYLESHEET],
        )
        self.label_column = pn.widgets.Select(
            name="Label column", options=OrderedDict({"No label": ""}), value="", height=58,
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), stylesheets=[SELECT_STYLESHEET],
        )
        self.id_column = pn.widgets.Select(
            name="Record ID column", options=OrderedDict({"Generate IDs": ""}), value="", height=58,
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), stylesheets=[SELECT_STYLESHEET],
        )

        self.advanced_toggle = pn.widgets.Toggle(
            name="Advanced options", value=False, button_type="default", height=34, width=142,
            margin=(8, 0, 0, 0), stylesheets=[BUTTON_SECONDARY_STYLESHEET],
        )
        self.advanced_toggle.param.watch(self._on_advanced_options_toggled, "value")

        self.annotation_variant = pn.widgets.Select(
            name="Structured annotation variant", options=OrderedDict({
                "Automatic · prefer extended metadata": "auto", "Extended metadata": "extended",
                "Base annotations": "base",
            }), value="auto", height=58, sizing_mode="stretch_width", visible=False, margin=(0, 0, 0, 0),
            stylesheets=[SELECT_STYLESHEET],
        )
        self.annotation_variant.param.watch(lambda *_: self._refresh_structure_summary(), "value")
        self.structured_media_root = pn.widgets.TextInput(
            name="Frame media root override (optional)",
            placeholder="Leave blank for automatic repository/media-root detection", value="", height=58,
            sizing_mode="stretch_width", visible=False, margin=(0, 0, 0, 0),
            stylesheets=[SEARCH_INPUT_STYLESHEET],
        )
        self.structured_media_root.param.watch(
            lambda *_: self._refresh_existing_registration_summary(), "value",
        )
        self.structured_frame_fps = pn.widgets.FloatInput(
            name="Playback FPS override (auto when possible)", value=0.0, start=0.0, step=1.0, height=58,
            sizing_mode="stretch_width", visible=False, margin=(0, 0, 0, 0),
            stylesheets=[NUMBER_INPUT_STYLESHEET],
        )
        self.structured_frame_fps.param.watch(self._on_structured_frame_fps_changed, "value")
        self.structured_registration_hint = pn.pane.HTML(
            "", visible=False, sizing_mode="stretch_width", margin=(0, 0, 0, 0),
        )
        self.cache_annotations_button = pn.widgets.Button(
            name="Cache selected annotations", icon="download", button_type="primary", height=38,
            sizing_mode="stretch_width", disabled=True, visible=False, margin=(0, 0, 0, 0),
            stylesheets=[BUTTON_PRIMARY_STYLESHEET],
        )
        self.cache_annotations_button.on_click(self._on_cache_annotations_clicked)

        self.import_button = pn.widgets.Button(
            name="Import selected splits", icon="cloud-download", button_type="success", height=38,
            sizing_mode="stretch_width", disabled=True, margin=(0, 0, 0, 0),
            stylesheets=[BUTTON_SUCCESS_STYLESHEET],
        )
        self.import_button.on_click(self._on_import_clicked)
        self.cancel_button = pn.widgets.Button(
            name="Cancel import", icon="x", button_type="warning", height=38, width=138, disabled=True,
            visible=False, margin=(0, 0, 0, 0), stylesheets=[BUTTON_SECONDARY_STYLESHEET],
        )
        self.cancel_button.on_click(self._on_cancel_clicked)
        self.transfer_notice = pn.pane.Alert(
            "", alert_type="warning", visible=False, sizing_mode="stretch_width", margin=(0, 0, 0, 0),
        )
        self.image_column.param.watch(self._on_image_column_changed, "value")

        # Separate repository-media workflow. This is deliberately not driven by
        # image split selection: video MP4 files often live directly in Hub repos.
        self.repository_video_prefix = pn.widgets.TextInput(
            name="Video repository folder (optional)", placeholder="e.g. clips/P01", value="",
            sizing_mode="stretch_width",
        )
        self.repository_video_limit = pn.widgets.IntInput(
            name="Maximum videos", value=10, start=1, end=1000, sizing_mode="stretch_width",
        )
        self.repository_video_active = pn.widgets.Checkbox(
            name="Set video dataset active", value=True, sizing_mode="stretch_width",
        )
        self.repository_video_annotations = pn.widgets.Select(
            name="Clip action annotations", options={
                "Auto (EPIC-KITCHENS-100 only)": "auto", "Do not join annotations": "none",
                "Require EPIC-KITCHENS-100 actions": "epic_kitchens_100",
            }, value="auto", sizing_mode="stretch_width",
        )
        self.repository_video_button = pn.widgets.Button(
            name="Import repository video files", icon="video", button_type="primary", height=38,
            sizing_mode="stretch_width", disabled=True, stylesheets=[BUTTON_PRIMARY_STYLESHEET],
        )
        self.repository_video_button.on_click(self._on_repository_video_import_clicked)

        self.import_summary = pn.pane.HTML("", sizing_mode="stretch_width", margin=(0, 0, 0, 0))
        self.import_preview = pn.pane.HTML(
            "", sizing_mode="stretch_width", visible=False, margin=(0, 0, 0, 0),
        )

        self.status = pn.pane.HTML(
            self._status_html("Ready", "info"), sizing_mode="stretch_width", max_width=390,
            margin=(0, 0, 0, 0),
        )
        self.progress_bar = pn.widgets.Progress(
            name="Progress", value=-1, active=True, max=100, visible=False, height=7,
            sizing_mode="stretch_width", stylesheets=[PROGRESS_STYLESHEET],
        )
        self.progress_text = pn.pane.HTML(
            "", visible=False, sizing_mode="stretch_width", margin=(0, 0, 0, 0),
        )
    def _build_view(self) -> pn.Column:
        heading = pn.pane.HTML(
            """
            <div style="min-width:0;">
              <div style="color:#263244;font-size:15px;font-weight:720;line-height:1.25;letter-spacing:-0.01em;">
                Explore and import Hugging Face datasets
              </div>
              <div style="margin-top:3px;color:#687386;font-size:11px;line-height:1.4;">
                Inspect schema, splits, cache state, and samples before importing the splits you need.
                Existing Hugging Face caches and AstronomicAL registrations are reused first.
              </div>
            </div>
            """,
            sizing_mode="stretch_width", margin=(0, 0, 0, 0),
        )

        search_label = pn.pane.HTML(
            """
            <div style="color:#526071;font-size:10px;font-weight:750;letter-spacing:.045em;text-transform:uppercase;line-height:1.2;">
              Search Hugging Face
            </div>
            """,
            sizing_mode="stretch_width", height=18, margin=(0, 0, 3, 0),
        )

        self.search_options_panel = pn.Column(
            pn.Row(
                self.task, self.sort, self.limit, sizing_mode="stretch_width", margin=(0, 0, 0, 0),
                styles={"gap": "8px", "align-items": "end"},
            ), self.token, sizing_mode="stretch_width", visible=False, margin=(8, 0, 0, 0), styles={
                **SUBSURFACE_STYLES, "padding": "10px", "overflow-x": "hidden", "overflow-y": "visible",
            },
        )

        search_controls = pn.Column(
            search_label, pn.Row(
                self.query, self.search_button, self.search_options_toggle, sizing_mode="stretch_width",
                height=38, margin=(0, 0, 0, 0), styles={
                    "box-sizing": "border-box", "gap": "8px", "align-items": "center", "overflow": "hidden",
                    "min-width": "0",
                },
            ), sizing_mode="stretch_width", margin=(0, 0, 0, 0), styles={
                "box-sizing": "border-box", "display": "flex", "gap": "4px", "min-width": "0",
                "overflow-x": "hidden", "overflow-y": "visible",
            },
        )

        self.progress_mount = pn.Column(
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), styles={
                "box-sizing": "border-box", "display": "flex", "gap": "4px", "min-width": "0",
                "overflow": "hidden",
            },
        )

        toolbar = pn.Column(
            pn.Row(
                heading, self.status, sizing_mode="stretch_width", margin=(0, 0, 0, 0), styles={
                    "box-sizing": "border-box", "gap": "14px", "align-items": "flex-start",
                    "min-width": "0", "overflow-x": "hidden", "overflow-y": "visible",
                },
            ), search_controls, self.search_options_panel, self.progress_mount, sizing_mode="stretch_width",
            margin=(0, 0, 0, 0), styles={
                **TOOLBAR_STYLES, "display": "flex", "gap": "10px", "overflow-x": "hidden",
                "overflow-y": "visible",
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
                sizing_mode="stretch_width", margin=(0, 0, 0, 0),
            ), self.results_meta, sizing_mode="stretch_width", margin=(0, 0, 8, 0),
            styles={"gap": "8px", "align-items": "center"},
        )

        self.results_surface = pn.Column(
            results_header, self.results_list, sizing_mode="stretch_width", min_width=210, max_width=250,
            visible=False, margin=(0, 0, 0, 0), styles={
                **SURFACE_STYLES, "overflow-x": "hidden", "overflow-y": "visible", "flex": "0 1 220px",
            },
        )

        self.empty_state = pn.pane.HTML(
            self._empty_state_html(), sizing_mode="stretch_both", margin=(0, 0, 0, 0),
            styles=EMPTY_STATE_STYLES,
        )

        self.selected_header_actions = pn.Row(
            self.access_check_button, self.inspect_button, width=254, height=38, height_policy="fixed",
            margin=(0, 0, 0, 0), styles={
                "gap": "8px", "align-items": "center", "justify-content": "flex-end", "overflow": "visible",
            },
        )
        self.selected_header = pn.Row(
            self.selected_pane, self.selected_header_actions, sizing_mode="stretch_width", height=64,
            height_policy="fixed", margin=(0, 0, 0, 0), styles={
                "box-sizing": "border-box", "min-width": "0", "padding": "10px 12px",
                "border": "1px solid #bfd7ea", "border-radius": "8px", "background": "#f3f8fc",
                "gap": "10px", "align-items": "center", "overflow": "hidden",
            },
        )

        # Primary workflow: choose the split population, choose what should become
        # active, then register/import it.  Keep these controls together in the
        # historically stable split surface rather than forcing users to scroll
        # into the secondary acquisition/options section.
        self.primary_action_row = pn.Row(
            self.import_button, self.cancel_button, sizing_mode="stretch_width", height=38,
            height_policy="fixed", margin=(0, 0, 0, 0), styles={"gap": "8px", "overflow": "visible"},
        )
        self.split_section = pn.Column(
            self._section_heading(
                "Splits / register",
                "Choose the splits you need, select the active dataset, then register or import them.",
            ), self.config_select, self.split_choices, self.active_after_import, self.primary_action_row,
            sizing_mode="stretch_width",
            # Header + one split row + labelled active-dataset Select + primary
            # action need ~200 px before optional configuration/wrapping.  This is
            # a minimum, not a fixed height: the card can still grow naturally.
            min_height=220, margin=(0, 0, 0, 0), styles={
                **SUBSURFACE_STYLES,
                # The workspace gives panels a bounded viewport.  This card is a
                # flex item inside the Panel Column; allowing flex-shrink while
                # SUBSURFACE_STYLES has min-height:0 collapses the card and lets
                # its children visibly overflow underneath the next section.
                # Registration is a primary workflow, so preserve its intrinsic
                # height and let the inspector/workspace scroll instead.
                "flex": "0 0 auto", "flex-shrink": "0", "overflow": "visible",
            },
        )

        self.preview_result_mount = pn.Column(
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), styles={
                "box-sizing": "border-box", "display": "block", "width": "100%", "max-width": "100%",
                "min-width": "0", "overflow": "visible", "position": "relative",
            },
        )

        # Panel/Bokeh Row does not consistently honour CSS align-items for widget
        # models.  Give every preview control the same 54px row budget and offset
        # the label-less button by the label height so the actual inputs line up.
        self.preview_controls = pn.Row(
            self.preview_split, self.preview_limit, self.preview_button, sizing_mode="stretch_width",
            height=54, height_policy="fixed", margin=(0, 0, 0, 0), styles={
                "box-sizing": "border-box", "gap": "6px", "min-width": "0", "overflow": "visible",
            },
        )
        self.preview_section = pn.Column(
            self._section_heading("Preview", "Preview only downloads or resolves the displayed examples."),
            self.preview_controls, self.preview_result_mount, sizing_mode="stretch_width",
            margin=(0, 0, 0, 0), styles={
                **SUBSURFACE_STYLES, "flex": "0 0 auto", "flex-shrink": "0", "overflow": "visible",
                "position": "relative", "z-index": "0",
            },
        )

        source_options = pn.Column(
            self._advanced_group_heading(
                "Source interpretation",
                "Automatic detection is recommended. Override these values only for unusual repositories.",
            ), self.strategy, self.trust_remote_code, self.image_column, self.label_column, self.id_column,
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), styles={
                "box-sizing": "border-box", "display": "flex", "gap": "10px", "min-width": "0",
                "overflow": "visible",
            },
        )

        registration_options = pn.Column(
            self._advanced_group_heading(
                "AstronomicAL registration",
                "Generated names remain tied to the repository and configuration selected for this import.",
            ), self.dataset_id, self.dataset_name, self.max_rows, self.write_parquet,
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), styles={
                "box-sizing": "border-box", "display": "flex", "gap": "10px", "min-width": "0",
                "overflow": "visible",
            },
        )

        self.advanced_panel = pn.Column(
            source_options, pn.pane.HTML(
                '<div style="height:1px;background:#dfe5ed;width:100%;"></div>', height=1,
                sizing_mode="stretch_width", margin=(2, 0, 2, 0),
            ), registration_options, sizing_mode="stretch_width", margin=(0, 0, 0, 0), styles={
                **SUBSURFACE_STYLES, "display": "flex", "gap": "12px", "padding": "12px",
                "background": "#f7f9fb", "overflow-x": "hidden", "overflow-y": "visible",
            },
        )
        self.advanced_mount = pn.Column(
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), styles={
                "box-sizing": "border-box", "width": "100%", "max-width": "100%", "min-width": "0",
                "overflow": "visible",
            },
        )

        # Secondary acquisition/options surface.  The user's primary registration
        # controls live with Splits above; this section owns structured-annotation
        # acquisition details, cache controls, transfer state, and advanced import
        # settings only.
        self.repository_video_section = pn.Column(
            self._section_heading(
                "Repository video files",
                "Download a bounded selection of MP4/MOV-style files; register with core.video. "
                "For large repositories choose a folder such as clips/P01.",
            ), self.repository_video_prefix, self.repository_video_limit, self.repository_video_annotations,
            self.repository_video_active, self.repository_video_button, sizing_mode="stretch_width",
            visible=False, styles={"border-top": "1px solid #dfe5ed", "padding-top": "8px"},
        )
        self.import_section = pn.Column(
            self._section_heading(
                "Data source options",
                "Optional structured-annotation acquisition and import settings. Local cache and existing AstronomicAL registrations are reused first.",
            ), self.annotation_variant, self.structured_media_root, self.structured_frame_fps,
            self.structured_registration_hint, self.cache_annotations_button, self.repository_video_section,
            self.transfer_notice, pn.Row(
                self.advanced_toggle, pn.layout.HSpacer(), sizing_mode="stretch_width", margin=(0, 0, 0, 0),
            ), self.advanced_mount, self.import_summary, self.import_preview, sizing_mode="stretch_width",
            margin=(0, 0, 0, 0), styles={
                **SUBSURFACE_STYLES, "background": "#ffffff", "position": "relative", "z-index": "0",
                "flex": "0 0 auto", "flex-shrink": "0", "overflow": "visible",
            },
        )

        # Priority order without changing the proven individual section geometry:
        # repository/cache -> splits -> register/import -> diagnostics -> preview.
        self.dataset_details = pn.Column(
            self.selected_header, self.access_pane, self.cache_pane, self.split_section,
            self.import_section, self.notice_pane, self.structure_pane, self.preview_section,
            sizing_mode="stretch_width", margin=(0, 0, 0, 0), styles={
                # Do not override the Bokeh Column with a second CSS flex layout.
                # In a bounded workspace that made min-height:0 child cards shrink
                # to header height while their widgets overflowed over siblings.
                "min-width": "0", "overflow-x": "hidden", "overflow-y": "visible",
            },
        )

        self.inspector_body = pn.Column(
            self.empty_state, sizing_mode="stretch_width", margin=(0, 0, 0, 0),
            styles={"min-width": "0", "overflow-x": "hidden", "overflow-y": "visible"},
        )

        self.inspector_surface = pn.Column(
            self.inspector_body, sizing_mode="stretch_width", min_width=320, margin=(0, 0, 0, 0), styles={
                **INSPECTOR_STYLES, "flex": "1 1 440px", "overflow-x": "hidden", "overflow-y": "visible",
            },
        )

        self.workbench = pn.Row(
            self.results_surface, self.inspector_surface, sizing_mode="stretch_width", margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box", "min-width": "0", "gap": "10px", "align-items": "flex-start",
                "overflow-x": "hidden", "overflow-y": "visible",
            },
        )

        shell = pn.Column(
            toolbar, self.workbench, sizing_mode="stretch_width", margin=(0, 0, 0, 0), styles={
                **SHELL_STYLES, "gap": "10px", "overflow-x": "hidden", "overflow-y": "visible",
            },
        )

        return pn.Column(
            shell, sizing_mode="stretch_width", css_classes=["al-hf-root-v4"], margin=(0, 0, 0, 0), styles={
                **ROOT_STYLES, "overflow-x": "hidden", "overflow-y": "visible",
            },
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
            sizing_mode="stretch_width", margin=(0, 0, 0, 0),
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
            sizing_mode="stretch_width", margin=(0, 0, 0, 0),
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
                    "dataset ID before contacting the Hub for missing metadata.", "info",
                )
                return

        self._run_job(
            self._search, title="Searching Hugging Face datasets", key=(
                f"hf.search:{query}:{self.task.value}:"
                f"{self.sort.value}:{self.limit.value}"
            ), on_done=self._on_search_done, on_error=self._on_error,
        )

    def _search(self, *, cancel_token: Any = None) -> pd.DataFrame:
        del cancel_token
        return self._service().search_dataframe(
            query=str(self.query.value or "").strip(), task=str(self.task.value or ""),
            limit=int(self.limit.value or 30), sort=str(self.sort.value or "downloads"),
            token=self._token(),
        )

    def _on_search_done(self, frame: pd.DataFrame) -> None:
        self._set_busy(False)
        self._finish_progress("Search complete.")

        self._search_df = frame if frame is not None else pd.DataFrame()
        # Exact-ID/layout restores do not need an empty search-results rail.
        # Reveal it only after a real search has completed, when it contains
        # useful choice/error information for the user.
        if hasattr(self, "results_surface"):
            self.results_surface.visible = True

        if self._search_df.empty or "repo_id" not in self._search_df.columns:
            self.results_meta.object = self._results_meta_html("No matches")
            self.results_list[:] = [self._results_empty_html("No matching datasets were found.")]
            exact = str(self.query.value or "").strip()
            if "/" in exact:
                self._select_repo(exact, inspect=True)
                self._set_status(
                    "No search row was returned. Inspecting the exact dataset ID directly.", "info",
                )
                return

            self._set_status("No matching datasets were found.", "warning")
            return

        repo_ids = [str(value) for value in self._search_df["repo_id"].tolist() if str(value).strip()]
        cached_count = 0
        if "cached" in self._search_df.columns:
            try:
                cached_count = sum(1 for value in self._search_df["cached"].tolist() if self._truthy(value))
            except Exception:
                cached_count = 0

        gated_count = 0
        if "gated" in self._search_df.columns:
            try:
                gated_count = sum(
                    1 for value in self._search_df["gated"].tolist() if self._gating_mode(value)
                )
            except Exception:
                gated_count = 0

        granted_count = 0
        if "access_state" in self._search_df.columns:
            try:
                granted_count = sum(
                    1 for value in self._search_df["access_state"].tolist()
                    if str(value or "").strip().casefold() == "access_granted"
                )
            except Exception:
                granted_count = 0

        label = f"{len(repo_ids)} datasets"
        if cached_count:
            label += f" · {cached_count} cached"
        if gated_count:
            label += f" · {gated_count} gated"
        if granted_count:
            label += f" · {granted_count} accessible"
        self.results_meta.object = self._results_meta_html(label)
        self._render_search_results()

        exact = str(self.query.value or "").strip().casefold()
        exact_repo = next((repo_id for repo_id in repo_ids if repo_id.casefold() == exact), None)

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
                name=repo_id, button_type="default", sizing_mode="stretch_width", height=24,
                margin=(0, 0, 0, 0), stylesheets=[RESULT_BUTTON_STYLESHEET],
            )
            button.description = f"Inspect {repo_id}"
            button.on_click(
                lambda _event, selected_repo=repo_id: self._select_repo(selected_repo, inspect=True)
            )

            meta = pn.pane.HTML(
                self._result_meta_html(row), sizing_mode="stretch_width", margin=(0, 0, 0, 0),
            )
            item = pn.Column(
                button, meta, sizing_mode="stretch_width", margin=(0, 0, 6, 0),
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
                item.styles = self._result_item_styles(selected=item_repo == selected_repo)
            except Exception:
                pass

    @staticmethod
    def _result_item_styles(*, selected: bool) -> dict[str, str]:
        if selected:
            return {
                "box-sizing": "border-box", "padding": "8px 10px", "border": "1px solid #78acd0",
                "border-radius": "8px", "background": "#eef6fc",
                "box-shadow": "0 0 0 1px rgba(15, 111, 189, 0.06)", "overflow": "hidden",
            }
        return {
            "box-sizing": "border-box", "padding": "8px 10px", "border": "1px solid #e2e7ee",
            "border-radius": "8px", "background": "#ffffff", "overflow": "hidden",
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
            "background:#ecf8f1;color:#1f7a4d;font-size:9.5px;font-weight:700;'>Cached</span>" if cached
            else ""
        )
        gating_mode = HuggingFaceBrowserPanel._gating_mode(row.get("gating_mode") or row.get("gated"))
        gated_label = "Gated"
        if gating_mode == "auto":
            gated_label = "Gated · auto"
        elif gating_mode == "manual":
            gated_label = "Gated · manual"
        gated_badge = (
            "<span title='This Hugging Face dataset requires an access request.' "
            "style='display:inline-flex;align-items:center;padding:2px 7px;border-radius:999px;"
            f"background:#fff7d6;color:#9a6700;font-size:9.5px;font-weight:700;'>{html.escape(gated_label)}</span>"
            if gating_mode else ""
        )
        access_badge = HuggingFaceBrowserPanel._search_access_badge_html(row)
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
            f"<div style='display:flex;gap:4px;flex:0 0 auto;'>{cache_badge}{gated_badge}{access_badge}</div>"
            "</div>"
        )

    @staticmethod
    def _gating_mode(value: Any) -> str:
        if isinstance(value, bool):
            return "gated" if value else ""
        text = str(value or "").strip().casefold()
        if text in {"", "false", "none", "null", "0", "no"}:
            return ""
        if text in {"auto", "automatic"}:
            return "auto"
        if text in {"manual", "manual-review", "manual_review"}:
            return "manual"
        return "gated"

    @staticmethod
    def _search_access_badge_html(row: Any) -> str:
        gating_mode = HuggingFaceBrowserPanel._gating_mode(row.get("gating_mode") or row.get("gated"))
        if not gating_mode:
            return ""

        state = str(row.get("access_state") or "access_unknown").strip().casefold()
        token_available = HuggingFaceBrowserPanel._truthy(row.get("token_available"))
        authenticated_user = str(row.get("authenticated_user") or "").strip()

        labels = {
            "access_granted": "Access ✓", "authentication_required": "Sign in",
            "access_request_required": "No access", "pending_approval": "Pending",
            "access_rejected": "Rejected", "region_restricted": "Restricted",
            "authentication_invalid": "Token invalid", "access_unknown": "Access ?",
            "metadata_unavailable": "Access ?", "checking": "Checking…",
        }
        palettes = {
            "access_granted": ("#ecf8f1", "#1f7a4d"), "authentication_required": ("#eef3f8", "#526071"),
            "access_request_required": ("#fff2df", "#9a5a00"), "pending_approval": ("#fff7d6", "#8a6500"),
            "access_rejected": ("#fff0ee", "#a42920"), "region_restricted": ("#fff0ee", "#a42920"),
            "authentication_invalid": ("#fff0ee", "#a42920"), "access_unknown": ("#f5f7fa", "#687386"),
            "metadata_unavailable": ("#f5f7fa", "#687386"), "checking": ("#f5f7fa", "#687386"),
        }
        label = labels.get(state, "Access ?")
        background, colour = palettes.get(state, palettes["access_unknown"])

        if state == "access_granted":
            title = (
                f"Current Hugging Face credentials can read this gated dataset"
                + (f" as {authenticated_user}." if authenticated_user else ".")
            )
        elif state == "authentication_required" or not token_available:
            title = "A Hugging Face login/token is required to verify access to this gated dataset."
        elif state == "pending_approval":
            title = "The current Hugging Face account has requested access and is awaiting approval."
        elif state == "access_request_required":
            title = "The current Hugging Face account does not yet have file access."
        elif state == "access_rejected":
            title = "The current Hugging Face account's access request was rejected."
        elif state == "region_restricted":
            title = "Hugging Face reports that access is restricted for the current region."
        elif state == "authentication_invalid":
            title = "The configured Hugging Face token could not be authenticated."
        else:
            title = "AstronomicAL could not yet verify gated-file access for this result."

        return (
            "<span title='{}' style='display:inline-flex;align-items:center;padding:2px 7px;"
            "border-radius:999px;background:{};color:{};font-size:9.5px;font-weight:700;'>" "{}</span>"
        ).format(html.escape(title, quote=True), background, colour, html.escape(label))

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
        return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "cached"}

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
            sizing_mode="stretch_width", margin=(0, 0, 0, 0),
        )

    def _on_search_options_toggled(self, event: Any) -> None:
        visible = bool(event.new)
        if hasattr(self, "search_options_panel"):
            self.search_options_panel.visible = visible
        self.search_options_toggle.name = ("Hide options" if visible else "Search options")

    def _show_search_options_for_authentication(self) -> None:
        """Reveal the token field when authentication is blocking progress."""
        if bool(self.search_options_toggle.value):
            if hasattr(self, "search_options_panel"):
                self.search_options_panel.visible = True
            return
        self.search_options_toggle.value = True

    def _on_token_changed(self, event: Any) -> None:
        if self._updating_widgets or self._disposed:
            return

        status = self._access_status
        if (not self.selected_repo or status is None or not status.gated or status.accessible):
            return

        self._show_search_options_for_authentication()
        if self._busy:
            self._pending_inspection_repo = self.selected_repo
            return
        self._start_inspection(
            config_name=(
                self._normalise_config(self.config_select.value) if self.config_select.visible else None
            )
        )

    def _on_advanced_options_toggled(self, event: Any) -> None:
        visible = bool(event.new)

        # Mount and unmount the advanced form instead of relying on ``visible``.
        # Some Panel/Bokeh layout combinations keep a styled hidden Column in the
        # document flow, which made the form appear permanently open.
        if hasattr(self, "advanced_mount"):
            self.advanced_mount.objects = ([self.advanced_panel] if visible else [])

        self.advanced_toggle.name = ("Hide advanced" if visible else "Advanced options")
        if hasattr(self, "repository_video_section"):
            self._refresh_existing_registration_summary()

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
        self._prime_access_from_search(repo_id)
        self._refresh_cache_summary()
        self._suggest_names()

        if not inspect:
            return

        if self._busy:
            self._pending_inspection_repo = repo_id
            self._set_status(
                "Dataset selection changed. Previous details were cleared; "
                "inspection will begin when the current task finishes.", "info",
            )
            return

        self._pending_inspection_repo = None
        self._start_inspection()

    def _search_row_for_repo(self, repo_id: str) -> dict[str, Any]:
        frame = self._search_df
        if frame is None or getattr(frame, "empty", True) or "repo_id" not in frame.columns:
            return {}
        target = str(repo_id or "").strip().casefold()
        try:
            matches = frame[frame["repo_id"].astype(str).str.casefold() == target]
            if matches.empty:
                return {}
            return dict(matches.iloc[0].to_dict())
        except Exception:
            return {}

    def _prime_access_from_search(self, repo_id: str) -> None:
        row = self._search_row_for_repo(repo_id)
        gating_mode = self._gating_mode(row.get("gating_mode") or row.get("gated"))
        if not gating_mode:
            self._access_status = None
            self.access_pane.object = ""
            self.access_check_button.visible = False
            return

        state = str(row.get("access_state") or "checking").strip() or "checking"
        token_available = self._truthy(row.get("token_available"))
        accessible = self._truthy(row.get("accessible"))
        message = str(row.get("access_message") or "").strip()
        if not message:
            message = (
                "This dataset is gated. AstronomicAL is checking whether the current "
                "Hugging Face account can read its files."
            )

        self._access_status = HFDatasetAccessStatus(
            repo_id=repo_id, gated=True, gating_mode=gating_mode, state=state, accessible=accessible,
            token_available=token_available,
            authenticated_user=str(row.get("authenticated_user") or "").strip(),
            access_url=f"https://huggingface.co/datasets/{repo_id}",
            probe_file=str(row.get("access_probe_file") or "").strip(), message=message,
            error=str(row.get("access_error") or "").strip(),
        )
        self._render_access_status()

    def _on_access_check_clicked(self, _event: Any = None) -> None:
        if not self.selected_repo:
            self._set_status("Select a dataset first.", "warning")
            return
        self._start_inspection(
            config_name=(
                self._normalise_config(self.config_select.value) if self.config_select.visible else None
            )
        )

    def _render_access_status(self) -> None:
        status = self._access_status
        if status is None or not status.gated:
            self.access_pane.object = ""
            self.access_check_button.visible = False
            return

        self.access_check_button.visible = True
        self.access_pane.object = self._access_status_html(status)
        if str(status.state or "") in {"authentication_required", "authentication_invalid"}:
            self._show_search_options_for_authentication()

    def _access_allows_data(self) -> bool:
        status = self._access_status
        return not (status is not None and status.gated and not status.accessible)

    def _structured_profile(self) -> dict[str, Any]:
        details = self._details
        if details is None:
            return {}
        profile = getattr(details, "structured_annotation_profile", {}) or {}
        return dict(profile) if isinstance(profile, dict) else {}

    def _clear_selected_dataset_details(self) -> None:
        self._details = None
        self._details_mode = None
        self._details_config = None
        self._access_status = None

        self._updating_widgets = True
        try:
            self.config_select.options = ["default"]
            self.config_select.value = "default"
            self.config_select.visible = False

            self.split_choices.options = []
            self.split_choices.value = []

            self.preview_split.options = []
            self.preview_split.value = None

            self.active_after_import.options = OrderedDict({"Do not change active dataset": ""})
            self.active_after_import.value = ""

            self.image_column.options = []
            self.image_column.value = None
            self.label_column.options = OrderedDict({"No label": ""})
            self.label_column.value = ""
            self.id_column.options = OrderedDict({"Generate IDs": ""})
            self.id_column.value = ""
        finally:
            self._updating_widgets = False

        self.access_pane.object = ""
        self.access_check_button.visible = False
        self.structure_pane.object = self._structure_placeholder_html()
        self.notice_pane.object = ""
        self.preview_grid.object = ""
        if hasattr(self, "preview_result_mount"):
            self.preview_result_mount.objects = []
        self.import_summary.object = ""
        self.import_preview.object = ""
        self.import_preview.visible = False

        self.preview_button.disabled = True
        self.import_button.disabled = True
        self.import_button.name = "Import selected splits"
        self.import_button.icon = "cloud-download"
        self.repository_video_button.disabled = True
        self.repository_video_section.visible = False
        self.annotation_variant.visible = False
        self.structured_media_root.visible = False
        self.structured_frame_fps.visible = False
        self.structured_registration_hint.visible = False
        self.structured_registration_hint.object = ""
        self.cache_annotations_button.visible = False
        self.cache_annotations_button.disabled = True
        self.active_after_import.visible = True
        self.advanced_toggle.visible = True
        self.advanced_mount.objects = ([self.advanced_panel] if bool(self.advanced_toggle.value) else [])

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
                repo_id=repo_id, desired=desired, token=token, trust_remote_code=trust_remote_code,
                config_name=selected_config, cancel_token=cancel_token,
            ),
            title=f"Inspecting {repo_id}",
            key=f"hf.inspect:{repo_id}:{selected_config or 'default'}:{desired}",
            on_done=lambda payload: self._on_inspect_done(
                payload, expected_repo=repo_id, generation=generation,
            ), on_error=lambda exc: self._on_inspect_error(
                exc, expected_repo=repo_id, generation=generation,
            ),
        )

    def _inspect(
        self, *, repo_id: str, desired: str, token: str | None, trust_remote_code: bool,
        config_name: str | None, cancel_token: Any = None,
    ) -> dict[str, Any]:
        del cancel_token
        service = self._service()

        # Cache is the first inspection source, not merely a badge rendered after
        # remote metadata has already been fetched.  Repeated layout restores and
        # exact-ID selections should stay offline whenever local state is sufficient.
        restored_repo = str(self._restored_state.get("selected_repo") or "").strip()
        revision_hint = (
            str(self._restored_state.get("revision") or "").strip() if restored_repo == repo_id else ""
        )
        cache = service.cache_status(repo_id, revision=revision_hint or None)
        cached_details = service.get_dataset_details_cached(
            repo_id, config_name=config_name, mode=desired, token=token,
            trust_remote_code=trust_remote_code, revision=revision_hint or None, cache_status=cache,
        )
        if cached_details is not None:
            configs = list(cached_details.configs or []) or ["default"]
            selected_config = (
                str(config_name) if config_name not in (None, "") and str(config_name) in configs
                else ("default" if "default" in configs else configs[0])
            )
            source = str(getattr(cached_details, "inspection_source", "") or "")
            if cached_details.structured_annotation_profile:
                mode = "cache"
            elif source == "local_datasets_builder":
                mode = "builder"
            else:
                mode = "files"
            print(
                f"[AL_DEBUG][HF][inspect_cache_hit] repo_id={repo_id!r} "
                f"source={source!r} config={selected_config!r}",
                flush=True,
            )
            return {
                "repo_id": repo_id, "mode": mode, "details": cached_details, "configs": configs,
                "config": selected_config, "cache": cache, "viewer_error": "", "access_status": None,
                "cache_only": True,
            }

        access_status = service.dataset_access_status(repo_id, token=token)
        if access_status.gated and not access_status.accessible:
            details = service.get_dataset_access_preview(repo_id, token=token)
            cache = service.cache_status(repo_id, revision=details.revision or None)
            return {
                "repo_id": repo_id, "mode": "access_required", "details": details, "configs": ["default"],
                "config": "default", "cache": cache, "viewer_error": "", "access_status": access_status,
            }

        viewer_details: HFDatasetDetails | None = None
        viewer_error: BaseException | None = None
        if desired in {"auto", "viewer"}:
            try:
                viewer_details = service.get_dataset_details_viewer(
                    repo_id, config_name=config_name, token=token,
                )
            except BaseException as exc:
                viewer_error = exc

        if desired == "viewer":
            if viewer_details is None:
                raise RuntimeError(
                    "Dataset Viewer inspection failed: " + str(viewer_error or "unknown error")
                )
            mode = "viewer"
            details = viewer_details
        elif desired == "auto" and viewer_details is not None:
            # Dataset Viewer is the preferred inspection path even for non-image
            # datasets. It is precomputed server-side and avoids triggering a
            # local "Resolving data files" pass merely to understand the schema.
            mode = "viewer"
            details = viewer_details
        else:
            configs: list[str] = []
            try:
                configs = service.get_config_names_builder(
                    repo_id, token=token, trust_remote_code=trust_remote_code
                )
            except Exception:
                configs = []
            if config_name is None and configs:
                config_name = "default" if "default" in configs else configs[0]

            builder_details: HFDatasetDetails | None = None
            builder_error: BaseException | None = None
            if desired in {"auto", "builder"}:
                try:
                    builder_details = service.get_dataset_details_builder(
                        repo_id, config_name=self._normalise_config(config_name), token=token,
                        trust_remote_code=trust_remote_code,
                    )
                except BaseException as exc:
                    builder_error = exc

            file_details: HFDatasetDetails | None = None
            file_error: BaseException | None = None
            builder_usable = self._builder_details_usable(service, builder_details)
            if desired == "files" or builder_details is None or (desired == "auto" and not builder_usable):
                try:
                    file_details = service.get_dataset_details(
                        repo_id, token=token, trust_remote_code=trust_remote_code
                    )
                except BaseException as exc:
                    file_error = exc

            if desired == "builder":
                if builder_details is None:
                    raise RuntimeError(
                        "Structured dataset inspection failed: " + str(builder_error or "unknown error")
                    )
                mode, details = "builder", builder_details
            elif desired == "files":
                if file_details is None:
                    raise RuntimeError(
                        "Repository image-file inspection failed: " + str(file_error or "unknown error")
                    )
                mode, details = "files", file_details
            else:
                mode, details = self._choose_automatic_strategy(
                    service, builder_details=builder_details, file_details=file_details
                )
                if viewer_error:
                    details.warnings.insert(
                        0, "Dataset Viewer unavailable; used local fallback. " + str(viewer_error),
                    )

        configs = list(details.configs or []) or ["default"]
        selected_config = (
            str(config_name) if config_name not in (None, "") and str(config_name) in configs
            else ("default" if "default" in configs else configs[0])
        )
        cache = service.cache_status(repo_id, revision=details.revision or None)
        return {
            "repo_id": repo_id, "mode": mode, "details": details, "configs": configs,
            "config": selected_config, "cache": cache, "viewer_error": str(viewer_error or ""),
            "access_status": access_status,
        }

    @staticmethod
    def _builder_details_usable(
        service: HuggingFaceDatasetService, details: HFDatasetDetails | None,
    ) -> bool:
        if details is None:
            return False

        config = str(details.configs[0] if details.configs else "default")
        features = dict(
            details.features_by_config.get(config) or details.features_by_config.get("default") or {}
        )
        image_column, _label_column, _id_column = service.choose_builder_columns(details, config=config)
        fallback_only = (len(features) == 1 and str(features.get("image", "")).lower().endswith("fallback"))
        return bool(image_column and not fallback_only)

    @staticmethod
    def _choose_automatic_strategy(
        service: HuggingFaceDatasetService, *, builder_details: HFDatasetDetails | None,
        file_details: HFDatasetDetails | None,
    ) -> tuple[str, HFDatasetDetails]:
        if HuggingFaceBrowserPanel._builder_details_usable(service, builder_details):
            # Structured rows are preferred even without labels because they
            # preserve declared configs and split names. Label inference then
            # falls back to the selected column's raw value.
            return "builder", builder_details

        if file_details is not None and (
            file_details.sample_files or int(file_details.scanned_file_count or 0) > 0
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

    def _on_inspect_done(self, payload: dict[str, Any], *, expected_repo: str, generation: int) -> None:
        if not self._inspection_is_current(expected_repo, generation):
            self._finish_stale_inspection()
            return

        self._pending_inspection_repo = None
        self._set_busy(False)
        self._finish_progress("Inspection complete.")

        self._details_mode = str(payload.get("mode") or "builder")
        self._details = payload.get("details")
        self._details_config = str(payload.get("config") or "default")
        access_status = payload.get("access_status")
        self._access_status = (access_status if isinstance(access_status, HFDatasetAccessStatus) else None)
        self._render_access_status()

        if self._details is None:
            self._set_status("Inspection returned no dataset details.", "danger")
            return

        configs = [str(value) for value in (payload.get("configs") or ["default"])]
        if not configs:
            configs = ["default"]

        self._updating_widgets = True
        try:
            self.config_select.options = configs
            selected_config = (self._details_config if self._details_config in configs else configs[0])
            self.config_select.value = selected_config
            self.config_select.visible = len(configs) > 1
            self._apply_details_to_widgets(self._details, mode=self._details_mode, config=selected_config)
        finally:
            self._updating_widgets = False

        cache = payload.get("cache")
        self.cache_pane.object = self._cache_html(cache)
        access_blocked = bool(
            self._access_status is not None and self._access_status.gated
            and not self._access_status.accessible
        )
        if not access_blocked:
            try:
                self._service().remember_dataset_details(
                    self._details, config_name=self._details_config, mode=self._details_mode,
                    token=self._token(),
                )
            except Exception:
                pass
        self._refresh_structure_summary()
        self._refresh_existing_registration_summary()
        self._suggest_names()

        mode_label = self._mode_label(self._details_mode)
        if (
            self._access_status is not None and self._access_status.gated
            and not self._access_status.accessible
        ):
            self._set_status(self._access_status.message, "warning")
            return

        if bool(payload.get("cache_only", False)):
            self._set_status(
                f"Inspection complete from local cache ({html.escape(mode_label)}). No Hub request was needed.",
                "success",
            )
        else:
            self._set_status(
                f"Inspection complete. AstronomicAL selected {html.escape(mode_label)}.",
                "success",
            )

    def _on_inspect_error(self, exc: BaseException, *, expected_repo: str, generation: int) -> None:
        if not self._inspection_is_current(expected_repo, generation):
            self._finish_stale_inspection()
            return
        self._pending_inspection_repo = None
        self._on_error(exc)

    def _inspection_is_current(self, expected_repo: str, generation: int) -> bool:
        return (generation == self._inspection_generation and expected_repo == self.selected_repo)

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

    def _on_structured_frame_fps_changed(self, _event: Any) -> None:
        if not self._updating_widgets:
            self._structured_frame_fps_nominal = False
        self._refresh_existing_registration_summary()

    def _apply_details_to_widgets(self, details: HFDatasetDetails, *, mode: str, config: str) -> None:
        if mode == "access_required":
            self._set_select_options(
                self.image_column, OrderedDict({"Access required before image detection": ""}),
                preferred="",
            )
            self._set_select_options(
                self.label_column, OrderedDict({"Access required before label detection": ""}),
                preferred="",
            )
            self._set_select_options(
                self.id_column, OrderedDict({"Generate IDs after access is granted": ""}), preferred="",
            )
            self.split_choices.options = []
            self.split_choices.value = []
            self.preview_split.options = []
            self.preview_split.value = None
            self.active_after_import.options = OrderedDict({"Do not change active dataset": ""})
            self.active_after_import.value = ""
            self.preview_button.disabled = True
            self.import_button.disabled = True
            self.annotation_variant.visible = False
            self.structured_media_root.visible = False
            self.structured_frame_fps.visible = False
            self.structured_registration_hint.visible = False
            self.structured_registration_hint.object = ""
            self.cache_annotations_button.visible = False
            self.cache_annotations_button.disabled = True
            self.active_after_import.visible = True
            self.advanced_toggle.visible = True
            return

        profile = dict(getattr(details, "structured_annotation_profile", {}) or {})
        if profile:
            self.annotation_variant.visible = True
            self.structured_media_root.visible = True
            self.structured_frame_fps.visible = True
            declared_fps = float(profile.get("frame_media_fps") or 0.0)
            inferred_fps = declared_fps
            nominal = False
            restored_fps = self._restored_state.get("structured_frame_fps")
            if restored_fps not in (None, ""):
                try:
                    inferred_fps = float(restored_fps)
                    nominal = bool(self._restored_state.get("structured_frame_fps_nominal", False))
                except Exception:
                    pass
            elif inferred_fps <= 0:
                inferred_fps = float(self.nominal_structured_playback_fps)
                nominal = True
            self._structured_frame_fps_nominal = nominal
            self.structured_frame_fps.value = inferred_fps
            if "structured_media_root" in self._restored_state:
                self.structured_media_root.value = str(self._restored_state.get("structured_media_root") or "")
            self.cache_annotations_button.visible = True
            self.active_after_import.visible = True
            self.advanced_toggle.visible = False
            self.advanced_mount.objects = []
            self._set_select_options(
                self.image_column, OrderedDict({"Frame sequence resolved by core.video": ""}), preferred="",
            )
            self._set_select_options(
                self.label_column, OrderedDict({"Temporal annotations remain in structured sidecars": ""}),
                preferred="",
            )
            self._set_select_options(
                self.id_column, OrderedDict({"Video identity defined by annotation schema": ""}),
                preferred="",
            )
            split_map = details.splits_by_config.get(config) or details.splits_by_config.get("default") or []
            splits = [str(value) for value in split_map if str(value).strip()]
            if not splits:
                splits = [str(value) for value in (profile.get("splits") or []) if str(value).strip()]
            if not splits:
                splits = ["train"]
            restored_splits = [
                str(value) for value in (self._restored_state.get("splits") or []) if str(value) in splits
            ]
            selected_splits = restored_splits or list(splits)
            self.split_choices.options = splits
            self.split_choices.value = selected_splits
            self._update_active_after_import()
            self.preview_split.options = splits
            self.preview_split.value = splits[0]
            self.preview_button.disabled = True
            self.import_button.name = "Register selected video splits"
            self.import_button.icon = "video"
            self._refresh_structured_registration_controls()
            self.cache_annotations_button.disabled = (
                not bool(selected_splits) or not self._access_allows_data()
            )
            return

        self.annotation_variant.visible = False
        self.structured_media_root.visible = False
        self.structured_frame_fps.visible = False
        self.structured_registration_hint.visible = False
        self.structured_registration_hint.object = ""
        self.cache_annotations_button.visible = False
        self.cache_annotations_button.disabled = True
        self.active_after_import.visible = True
        self.advanced_toggle.visible = True
        self.advanced_mount.objects = ([self.advanced_panel] if bool(self.advanced_toggle.value) else [])

        if mode in {"builder", "viewer"}:
            service = self._service()
            image, label, record_id = service.choose_builder_columns(details, config=config)
            features = dict(
                details.features_by_config.get(config) or details.features_by_config.get("default") or {}
            )
            columns = list(features)

            image_options = OrderedDict(
                [("No image input detected", "")] + [(column, column) for column in columns]
            )
            self._set_select_options(self.image_column, image_options, preferred=image or "")
            repository_stats = dict(getattr(details, "repository_stats_by_split", {}) or {})
            best_repository_label_coverage = max(
                (float(row.get("label_coverage", 0.0) or 0.0) for row in repository_stats.values()),
                default=0.0,
            )
            label_options = [("No label", "")] + [(column, column) for column in columns]
            native_ground_truth = str(getattr(details, "ground_truth_field_hint", "") or "")
            exact_fiftyone_labels = bool(mode == "viewer" and native_ground_truth)
            if exact_fiftyone_labels:
                label_options.append((
                    f"Native FiftyOne ground truth · {native_ground_truth}",
                    HF_FIFTYONE_GROUND_TRUTH,
                ))
            inferred_path_labels = mode == "viewer" and best_repository_label_coverage >= 0.80
            if inferred_path_labels:
                label_options.append(("Repository filename/folder inference (fallback)", HF_PATH_LABEL))
            preferred_label = label or (
                HF_FIFTYONE_GROUND_TRUTH if exact_fiftyone_labels
                else (HF_PATH_LABEL if inferred_path_labels else "")
            )
            self._set_select_options(
                self.label_column, OrderedDict(label_options), preferred=preferred_label,
            )
            self._set_select_options(
                self.id_column, OrderedDict(
                    [("Generate IDs", "")] + [(column, column) for column in columns]
                ), preferred=record_id or "",
            )
        else:
            self._set_select_options(
                self.image_column, OrderedDict({"Repository image files": "__hf_file__"}),
                preferred="__hf_file__",
            )
            self._set_select_options(
                self.label_column, OrderedDict(
                    {"Infer from folders and filenames": HF_PATH_LABEL, "No label": ""}
                ), preferred=HF_PATH_LABEL,
            )
            self._set_select_options(self.id_column, OrderedDict({"Generate IDs": ""}), preferred="")

        splits_by_config = details.splits_by_config or {}
        splits = list(splits_by_config.get(config) or splits_by_config.get("default") or [])
        if not splits and mode == "files":
            splits = sorted({str(file.split or "train") for file in details.sample_files})
        if not splits:
            splits = ["train"]

        restored_splits = [
            str(value) for value in (self._restored_state.get("splits") or []) if str(value) in splits
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
        restored_preview = str(self._restored_state.get("preview_split") or "")
        self.preview_split.value = (restored_preview if restored_preview in splits else splits[0])
        has_image = bool(self._resolved_image_column())
        self.preview_button.disabled = False if mode == "viewer" else not has_image
        self._update_active_after_import()
        self.import_button.disabled = (
            not bool(selected_splits) or not has_image or not self._access_allows_data()
        )

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
            self._start_inspection(config_name=self._normalise_config(self.config_select.value))

    def _on_split_choices_changed(self, _event: Any) -> None:
        if self._updating_widgets:
            return
        if self._structured_profile():
            self.cache_annotations_button.disabled = (
                not bool(self.split_choices.value) or not self._access_allows_data()
            )
            self._update_active_after_import()
            self._refresh_structured_registration_controls()
            self._refresh_structure_summary()
            return
        self._update_active_after_import()
        self.import_button.disabled = (
            not bool(self.split_choices.value) or not bool(self._resolved_image_column())
            or not self._access_allows_data()
        )
        self._refresh_existing_registration_summary()
        self._suggest_names()

    def _on_image_column_changed(self, _event: Any) -> None:
        if self._updating_widgets or not hasattr(self, "import_button"):
            return
        self.import_button.disabled = (
            not bool(self.split_choices.value) or not bool(self._resolved_image_column())
            or not self._access_allows_data()
        )
        if self._details is not None:
            self._refresh_structure_summary()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _on_preview_clicked(self, _event: Any = None) -> None:
        if self._details is None:
            self._set_status("Inspect the dataset before previewing it.", "warning")
            return
        if not self._access_allows_data():
            message = (
                self._access_status.message if self._access_status is not None
                else "Hugging Face dataset access is required before previewing files."
            )
            self._set_status(message, "warning")
            return

        request = {
            "repo_id": str(self.selected_repo), "mode": str(self._details_mode or "builder"),
            "config_name": self._normalise_config(self.config_select.value),
            "split": str(self.preview_split.value or "train"),
            "image_column": self._empty_to_none(self.image_column.value),
            "label_column": self._empty_to_none(self.label_column.value),
            "id_column": self._empty_to_none(self.id_column.value),
            "limit": int(self.preview_limit.value or 10), "token": self._token(),
            "revision": str(getattr(self._details, "revision", "") or "") or None,
            "trust_remote_code": bool(self.trust_remote_code.value),
        }
        self._pending_preview_request = request
        self._run_job(
            self._preview,
            title=f"Previewing {request['repo_id']} / {request['split']}",
            key=(
                f"hf.preview:{request['repo_id']}:{request['mode']}:"
                f"{request['config_name']}:{request['split']}:{request['limit']}:{request['revision'] or 'latest'}"
            ), on_done=self._on_preview_done, on_error=self._on_error,
        )

    def _preview(self, *, cancel_token: Any = None) -> dict[str, Any]:
        del cancel_token
        request = dict(getattr(self, "_pending_preview_request", {}) or {})
        if not request:
            raise RuntimeError("Preview request was not captured.")
        service = self._service()
        if request["mode"] == "viewer":
            if request.get("image_column"):
                return service.preview_image_items_viewer(
                    repo_id=request["repo_id"], config_name=request["config_name"], split=request["split"],
                    image_column=request["image_column"], label_column=request["label_column"],
                    id_column=request["id_column"], limit=request["limit"], token=request["token"],
                )
            return service.preview_rows_viewer(
                repo_id=request["repo_id"], config_name=request["config_name"], split=request["split"],
                limit=request["limit"], token=request["token"],
            )
        if request["mode"] == "files":
            return service.preview_image_items(
                repo_id=request["repo_id"], split=request["split"], limit=request["limit"],
                token=request["token"], thumb_size=180, revision=request["revision"],
            )
        return service.preview_image_items_builder(
            repo_id=request["repo_id"], config_name=request["config_name"], split=request["split"],
            image_column=request["image_column"], label_column=request["label_column"],
            id_column=request["id_column"], limit=request["limit"], token=request["token"],
            trust_remote_code=request["trust_remote_code"], thumb_size=180,
            revision=request["revision"],
        )

    def _on_preview_done(self, result: dict[str, Any]) -> None:
        self._set_busy(False)
        self._finish_progress("Preview ready.")
        result = dict(result or {})
        items = list(result.get("items", []) or [])
        note = str(result.get("note", "") or "")
        cache_source = str(result.get("cache_source", "") or "")

        if str(result.get("kind") or "") == "table":
            self.preview_result_mount.objects = [
                pn.pane.HTML(
                    self._generic_preview_html(
                        list(result.get("rows") or []), list(result.get("columns") or []), note=note,
                        cache_source=cache_source,
                    ), sizing_mode="stretch_width", margin=(0, 0, 0, 0),
                )
            ]
            self._set_status(
                f"Preview ready: {int(result.get('count', 0) or 0)} row(s). {cache_source}",
                "success",
            )
            return

        self.preview_grid.object = self._preview_grid_html(items, note=note, cache_source=cache_source)
        self.preview_result_mount.objects = [self.preview_grid]
        self._set_status(
            f"Preview ready: {len(items)} image(s). {cache_source or 'Local caches were checked first.'}",
            "success",
        )

    @staticmethod
    def _preview_card_html(item: dict[str, Any]) -> str:
        data_uri = str(item.get("data_uri", "") or "")
        image_src = str(item.get("image_src", "") or "")
        record_id = html.escape(str(item.get("record_id", "") or ""))
        label = html.escape(str(item.get("label", "") or ""))
        path = html.escape(str(item.get("path", "") or ""))
        if data_uri or image_src:
            src = html.escape(data_uri or image_src, quote=True)
            image_html = (
                f'<img src="{src}" alt="{record_id}" '
                'style="display:block;width:100%;max-width:100%;height:120px;'
                'object-fit:contain;background:#111111;border-radius:5px;" />'
            )
        else:
            image_html = (
                '<div style="display:flex;align-items:center;justify-content:center;height:120px;'
                'background:#eef2f6;color:#687386;font-size:10.5px;border-radius:5px;">'
                'Preview unavailable</div>'
            )

        label_html = (
            f'<div class="al-hf-preview-label" title="{label}" '
            'style="min-width:0;max-width:100%;white-space:nowrap;overflow:hidden;'
            'text-overflow:ellipsis;color:#526071;font-size:9.8px;line-height:1.35;">'
            f'{label}</div>'
            if label else ""
        )
        path_html = (
            f'<div class="al-hf-preview-path" title="{path}" '
            'style="min-width:0;max-width:100%;white-space:nowrap;overflow:hidden;'
            'text-overflow:ellipsis;color:#8a94a2;font-size:8.8px;line-height:1.3;">'
            f'{path}</div>'
            if path else ""
        )
        return (
            '<div class="al-hf-preview-card" '
            'style="box-sizing:border-box;width:100%;min-width:0;padding:7px;'
            'border:1px solid #d8dee8;border-radius:8px;background:#ffffff;overflow:hidden;">' + image_html
            + '<div class="al-hf-preview-card-copy" style="min-width:0;padding-top:6px;overflow:hidden;">'
            + f'<div class="al-hf-preview-record" title="{record_id}" '
              'style="min-width:0;max-width:100%;white-space:nowrap;overflow:hidden;'
              'text-overflow:ellipsis;color:#263244;font-size:10.5px;font-weight:720;'
              f'line-height:1.35;">{record_id}</div>'
            + label_html + path_html + '</div></div>'
        )

    @classmethod
    def _preview_grid_html(
        cls, items: list[dict[str, Any]], *, note: str = "", cache_source: str = "",
    ) -> str:
        detail = str(note or "").strip()
        if cache_source:
            source_text = f"Source: {cache_source}."
            detail = f"{detail} {source_text}".strip()
        note_html = (
            '<div class="al-hf-preview-note" '
            'style="grid-column:1/-1;box-sizing:border-box;width:100%;padding:7px 9px;'
            'border:1px solid #d8e3ec;border-radius:7px;background:#f7fafc;color:#526071;'
            'font-size:9.4px;line-height:1.4;overflow-wrap:anywhere;">' + html.escape(detail) + '</div>'
            if detail else ""
        )
        cards = "".join(cls._preview_card_html(dict(item or {})) for item in items)
        if not cards:
            cards = (
                '<div class="al-hf-preview-empty" '
                'style="grid-column:1/-1;box-sizing:border-box;width:100%;padding:8px 10px;'
                'border:1px solid #d8e3ec;border-radius:7px;background:#f7fafc;color:#526071;'
                'font-size:9.8px;line-height:1.45;">No preview rows were returned.</div>'
            )
        return (
            '<div class="al-hf-preview-grid" ' 'style="box-sizing:border-box;display:grid;'
            'grid-template-columns:repeat(auto-fit,minmax(150px,1fr));'
            'gap:8px;width:100%;max-width:100%;min-width:0;align-items:start;">' + note_html + cards
            + '</div>'
        )

    @staticmethod
    def _compact_feature_summary(feature_type: Any, spec: Any = None) -> str:
        spec_map = dict(spec or {}) if isinstance(spec, dict) else {}
        kind = str(spec_map.get("_type") or "").strip()
        if kind == "ClassLabel":
            names = [str(value) for value in (spec_map.get("names") or [])]
            return f"ClassLabel · {len(names):,} classes" if names else "ClassLabel"
        if kind == "Image":
            return "Image"
        if kind == "Value":
            dtype = str(spec_map.get("dtype") or "").strip()
            return f"Value · {dtype}" if dtype else "Value"
        if kind == "Sequence":
            nested = spec_map.get("feature")
            nested_summary = HuggingFaceBrowserPanel._compact_feature_summary("", nested)
            return f"Sequence · {nested_summary}" if nested_summary else "Sequence"
        if kind:
            return kind

        text = str(feature_type or "").strip()
        if "ClassLabel" in text:
            return "ClassLabel"
        if len(text) > 96:
            return text[:93].rstrip() + "…"
        return text or "Unknown"

    @staticmethod
    def _generic_preview_html(
        rows: list[dict[str, Any]], columns: list[str], *, note: str = "", cache_source: str = "",
    ) -> str:
        columns = [str(value) for value in columns]
        if not columns and rows:
            columns = list(rows[0])
        columns = columns[:10]
        head = "".join(
            f"<th style='padding:6px;text-align:left;border-bottom:1px solid #dfe5ed;font-size:9px;'>{html.escape(col)}</th>"
            for col in columns
        )
        body_rows = []
        for row in rows[:25]:
            cells = []
            for col in columns:
                value = row.get(col, "") if isinstance(row, dict) else ""
                text = str(value)
                if len(text) > 240:
                    text = text[:237] + "…"
                cells.append(
                    f"<td style='padding:6px;vertical-align:top;border-bottom:1px solid #edf1f5;font-size:9px;line-height:1.35;overflow-wrap:anywhere;'>{html.escape(text)}</td>"
                )
            body_rows.append("<tr>" + "".join(cells) + "</tr>")
        detail = " ".join(value for value in (note, cache_source) if value)
        return (
            f"<div style='margin-bottom:6px;color:#687386;font-size:9.5px;'>{html.escape(detail)}</div>"
            "<div style='overflow:auto;max-height:440px;border:1px solid #dfe5ed;border-radius:8px;'>"
            "<table style='border-collapse:collapse;width:100%;background:white;'>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
        )

    def _core_video_registration_available(self) -> bool:
        manager = self.manager or getattr(self.context, "plugins", None)
        if manager is None or not callable(getattr(manager, "run_action", None)):
            return False
        getter = getattr(manager, "get_action", None)
        if callable(getter):
            try:
                getter("core.video.register_manifest")
            except Exception:
                return False
        return True

    def _structured_video_registration_action_available(self) -> bool:
        return bool(
            self._structured_profile() and self._details is not None and self._access_allows_data()
            and bool(self.split_choices.value) and self._core_video_registration_available()
            and not self._busy
        )

    def _structured_video_registration_missing_requirements(
        self, *, splits: tuple[str, ...] | None = None,
    ) -> list[str]:
        profile = self._structured_profile()
        if not profile:
            return ["a detected structured-video annotation profile"]

        selected_splits = tuple(
            str(value) for value in (splits if splits is not None else (self.split_choices.value or []))
            if str(value).strip()
        )
        missing: list[str] = []

        if not self._core_video_registration_available():
            missing.append("core.video with the public register_manifest action")

        # Frame media roots are optional user overrides. The structured importer
        # resolves missing roots from the pinned Hugging Face repository inside
        # the managed import job, so absence here must never block dispatch.

        try:
            fps = float(self.structured_frame_fps.value or profile.get("frame_media_fps") or 0.0)
        except Exception:
            fps = 0.0
        if fps <= 0:
            missing.append("annotation frame FPS greater than 0")

        return missing

    def _structured_video_registration_available(self) -> bool:
        return bool(
            self._structured_video_registration_action_available()
            and not self._structured_video_registration_missing_requirements()
        )

    @staticmethod
    def _structured_registration_hint_html(
        missing: list[str], *, actionable: bool, nominal_fps: bool = False,
        automatic_roots: tuple[str, ...] = (), pending_auto_roots: tuple[str, ...] = (),
    ) -> str:
        if not actionable:
            if any("core.video" in item for item in missing):
                message = (
                    "<strong>Video registration unavailable:</strong> enable core.video with the "
                    "public register_manifest action. <strong>Cache selected annotations</strong> "
                    "can still be used now."
                )
                border = "#ead7a8"
                background = "#fff9eb"
                colour = "#725719"
                return (
                    f'<div style="box-sizing:border-box;width:100%;padding:7px 9px;border:1px solid {border};'
                    f'border-radius:7px;background:{background};color:{colour};font-size:9.5px;line-height:1.4;'
                    f'overflow-wrap:anywhere;">{message}</div>'
                )
            message = "Select at least one accessible split before registering video rows."
            border = "#d8e3ec"
            background = "#f7fafc"
            colour = "#526071"
        elif missing:
            message = (
                "<strong>Video registration still needs:</strong> " + html.escape("; ".join(missing))
                + ". <strong>Cache selected annotations</strong> can be used now; "
                "the registration button stays clickable and will repeat these requirements."
            )
            border = "#ead7a8"
            background = "#fff9eb"
            colour = "#725719"
            return (
                f'<div style="box-sizing:border-box;width:100%;padding:7px 9px;border:1px solid {border};'
                f'border-radius:7px;background:{background};color:{colour};font-size:9.5px;line-height:1.4;'
                f'overflow-wrap:anywhere;">{message}</div>'
            )
        else:
            parts = ["<strong>Ready to register.</strong>"]
            if automatic_roots:
                parts.append(
                    "Repository media already resolved for " + html.escape(", ".join(automatic_roots)) + "."
                )
            if pending_auto_roots:
                parts.append(
                    "Frame media for " + html.escape(", ".join(pending_auto_roots))
                    + " will be resolved automatically from the pinned Hugging Face repository during registration."
                )
            if nominal_fps:
                parts.append(
                    "The repository does not declare one authoritative annotation FPS, so playback will use "
                    "a <strong>nominal 5 fps cadence</strong>. Frame/mask indices remain exact; only wall-clock "
                    "playback timing is estimated. Override the FPS only if you know a better value."
                )
            message = " ".join(parts)
            border = "#b8dec9"
            background = "#f2faf5"
            colour = "#1f6a45"
            return (
                f'<div style="box-sizing:border-box;width:100%;padding:7px 9px;border:1px solid {border};'
                f'border-radius:7px;background:{background};color:{colour};font-size:9.5px;line-height:1.4;'
                f'overflow-wrap:anywhere;">{message}</div>'
            )

        return (
            f'<div style="box-sizing:border-box;width:100%;padding:7px 9px;border:1px solid {border};'
            f'border-radius:7px;background:{background};color:{colour};font-size:9.5px;line-height:1.4;'
            f'overflow-wrap:anywhere;">{html.escape(message)}</div>'
        )

    def _refresh_structured_registration_controls(self) -> None:
        structured = bool(self._structured_profile())
        self.structured_registration_hint.visible = structured
        if not structured:
            self.structured_registration_hint.object = ""
            return

        actionable = self._structured_video_registration_action_available()
        missing = self._structured_video_registration_missing_requirements()
        self.import_button.disabled = not actionable
        profile = self._structured_profile()
        roots = dict(profile.get("frame_media_roots_by_split") or {})
        selected = tuple(str(value) for value in (self.split_choices.value or []) if str(value).strip())
        automatic_roots = tuple(split for split in selected if str(roots.get(split) or "").strip())
        override = str(self.structured_media_root.value or "").strip()
        pending_auto_roots = tuple(
            split for split in selected if not override and not str(roots.get(split) or "").strip()
        )
        self.structured_registration_hint.object = self._structured_registration_hint_html(
            missing, actionable=actionable, nominal_fps=bool(self._structured_frame_fps_nominal),
            automatic_roots=automatic_roots, pending_auto_roots=pending_auto_roots,
        )

    def _start_structured_video_import(self, *, repo_id: str, splits: tuple[str, ...]) -> None:
        if not self._structured_video_registration_action_available():
            self._refresh_structured_registration_controls()
            if not self._core_video_registration_available():
                message = (
                    "Video registration unavailable: enable core.video with the public "
                    "register_manifest action. Cache selected annotations can still be used now."
                )
            else:
                message = "Select at least one accessible structured-video split before registering it."
            self._set_status(message, "warning")
            return

        missing_requirements = self._structured_video_registration_missing_requirements(splits=splits)
        if missing_requirements:
            message = "Video registration still needs: " + "; ".join(missing_requirements) + "."
            self._refresh_structured_registration_controls()
            self._set_status(message, "warning")
            return

        expected_id = self._suggested_base_id_for(repo_id, self._normalise_config(self.config_select.value))
        expected_name = self._suggested_base_name_for(repo_id, self._normalise_config(self.config_select.value))
        base_id_value = str(self.dataset_id.value or "").strip()
        base_name_value = str(self.dataset_name.value or "").strip()
        base_id = expected_id if not base_id_value or base_id_value == self._last_suggested_dataset_id else base_id_value
        base_name = expected_name if not base_name_value or base_name_value == self._last_suggested_dataset_name else base_name_value
        profile = self._structured_profile()
        revision = str(getattr(self._details, "revision", "") or "")
        manager = self.manager or getattr(self.context, "plugins", None)

        self._run_job(
            lambda cancel_token=None: import_structured_video_dataset(
                self.context, manager=manager, service=self._service(), repo_id=repo_id, splits=splits,
                variant=str(self.annotation_variant.value or "auto"), revision=revision or None,
                token=self._token(), profile=profile, dataset_id_base=base_id, dataset_name_base=base_name,
                media_root_override=(str(self.structured_media_root.value or "").strip() or None),
                annotation_fps=(
                    None if self._structured_frame_fps_nominal
                    else float(self.structured_frame_fps.value or profile.get("frame_media_fps") or 0.0)
                ), max_videos=int(self.max_rows.value or 0),
                active_split=(str(self.active_after_import.value or "").strip() or None),
                cancel_token=cancel_token, progress_callback=self._set_progress_from_worker,
            ),
            title=f"Registering structured video dataset {repo_id}",
            key=f"hf.video-import:{repo_id}:{revision or 'latest'}:{','.join(splits)}",
            on_done=self._on_import_done, on_error=self._on_error, cancellable=True,
        )
        if self._active_job_handle is not None:
            self.cancel_button.name = "Cancel video import"

    # ------------------------------------------------------------------
    # Structured annotation acquisition
    # ------------------------------------------------------------------

    def _on_cache_annotations_clicked(self, _event: Any = None) -> None:
        profile = self._structured_profile()
        if not profile:
            self._set_status("Inspect a supported structured annotation repository first.", "warning")
            return
        if not self._access_allows_data():
            message = (
                self._access_status.message if self._access_status is not None
                else "Hugging Face dataset access is required before caching annotations."
            )
            self._set_status(message, "warning")
            return
        repo_id = self.selected_repo
        splits = tuple(str(value) for value in (self.split_choices.value or []) if str(value).strip())
        if not repo_id or not splits:
            self._set_status("Select at least one annotation split to cache.", "warning")
            return
        variant = str(self.annotation_variant.value or "auto")
        revision = str(getattr(self._details, "revision", "") or "")
        token = self._token()

        self._run_job(
            lambda cancel_token=None: self._cache_selected_annotations(
                repo_id=repo_id, splits=splits, variant=variant, revision=revision, token=token,
                profile=profile, cancel_token=cancel_token,
            ),
            title=f"Caching annotations for {repo_id}",
            key=f"hf.annotations:{repo_id}:{revision or 'latest'}:{variant}:{','.join(splits)}",
            on_done=self._on_cache_annotations_done, on_error=self._on_error, cancellable=True,
        )
        if self._active_job_handle is not None:
            self.cancel_button.name = "Cancel transfer"

    def _cache_selected_annotations(
        self, *, repo_id: str, splits: tuple[str, ...], variant: str, revision: str, token: str | None,
        profile: dict[str, Any], cancel_token: Any = None,
    ) -> dict[str, Any]:
        return self._service().cache_structured_annotations(
            repo_id, splits=splits, variant=variant, revision=revision or None, token=token, max_workers=4,
            cancel_token=cancel_token, progress_callback=self._set_progress_from_worker, profile=profile,
        )

    def _on_cache_annotations_done(self, result: dict[str, Any]) -> None:
        self._set_busy(False)
        result = dict(result or {})
        files = list(result.get("annotation_files") or [])
        plan = dict(result.get("snapshot_plan") or {})

        def fmt_bytes(value: Any) -> str:
            try:
                amount = max(int(value or 0), 0)
            except Exception:
                amount = 0
            if not amount:
                return "unknown size"
            number = float(amount)
            for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
                if number < 1024.0 or unit == "TiB":
                    return f"{int(number):,} {unit}" if unit == "B" else f"{number:.2f} {unit}"
                number /= 1024.0
            return f"{amount:,} B"

        file_rows = []
        for item in files:
            path = html.escape(str(item.get("path") or ""))
            split = html.escape(str(item.get("split") or "unspecified"))
            variant = html.escape(str(item.get("variant") or "base"))
            size = fmt_bytes(item.get("size"))
            file_rows.append(
                "<div style='padding:6px 0;border-top:1px solid #dcebe2;'>"
                f"<code style='display:block;color:#1f6a45;font-size:9.3px;overflow-wrap:anywhere;'>{path}</code>"
                f"<span style='display:block;margin-top:2px;color:#526071;font-size:9px;'>{split} · {variant} · {html.escape(size)}</span>"
                "</div>"
            )

        cached_files = int(plan.get("cached_files", 0) or 0)
        downloaded_files = int(plan.get("download_files", 0) or 0)
        snapshot_root = html.escape(str(result.get("snapshot_root") or ""))
        self.import_summary.object = f"""
        <div style="box-sizing:border-box;width:100%;max-width:100%;min-width:0;padding:10px;border:1px solid #b8dec9;border-radius:8px;background:#f2faf5;color:#1f6a45;overflow:hidden;">
          <div style="font-size:11px;font-weight:720;line-height:1.3;">Structured annotations ready locally</div>
          <div style="margin-top:3px;color:#39785a;font-size:9.5px;line-height:1.4;">
            {len(files):,} selected annotation file(s) · {cached_files:,} already cached · {downloaded_files:,} required by the plan.
          </div>
          <div style="margin-top:4px;color:#526071;font-size:9.2px;line-height:1.4;">
            This cache action does not register a dataset. Use <strong>Register selected video splits</strong> to create core.video rows; temporal masklets remain reserved for <code>vision.segmentation</code>.
          </div>
          <div style="margin-top:5px;color:#687386;font-size:8.8px;overflow-wrap:anywhere;">Pinned snapshot: <code>{snapshot_root}</code></div>
          <div style="min-width:0;margin-top:6px;overflow:hidden;">{''.join(file_rows)}</div>
        </div>
        """
        self.import_preview.object = ""
        self.import_preview.visible = False
        self.transfer_notice.visible = False
        self._finish_progress("Structured annotation cache ready.")
        self._set_status(
            "Structured annotations are cached at the pinned revision. You can now register video rows through core.video; masklets remain external.",
            "success",
        )
        self._refresh_cache_summary()

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def _on_repository_video_import_clicked(self, _event: Any = None) -> None:
        repo_id = self.selected_repo
        if self._busy:
            return
        if not repo_id or not self._access_allows_data():
            self._set_status("Inspect an accessible Hugging Face repository first.", "warning")
            return
        if not self._core_video_registration_available():
            self._set_status("Enable core.video to register repository video files.", "warning")
            return
        revision = str(getattr(self._details, "revision", "") or "").strip() or None
        prefix = str(self.repository_video_prefix.value or "").strip()
        limit = int(self.repository_video_limit.value or 10)
        set_active = bool(self.repository_video_active.value)
        annotation_mode = (
            str(self.repository_video_annotations.value or "auto")
            if repo_id == EPIC_KITCHENS_REPO_ID else "none"
        )

        def import_job(cancel_token=None):
            result = import_repository_videos(
                self.context, manager=self.manager or getattr(self.context, "plugins", None),
                service=self._service(), repo_id=repo_id, path_prefix=prefix, revision=revision,
                token=self._token(), max_videos=limit, set_active=set_active,
                annotation_mode=annotation_mode, cancel_token=cancel_token,
                progress_callback=self._set_progress_from_worker,
            )
            return {
                "repo_id": repo_id, "rows": int(result.get("rows") or 0),
                "results": [{**result, "split": "repository"}],
                "active_split": "repository" if set_active else "", "preview": result.get("preview"),
            }

        self._repository_video_running = True
        self._run_job(
            import_job,
            title=f"Importing videos from {repo_id}",
            key=f"hf.repo-videos:{repo_id}:{revision or 'latest'}:{prefix}:{limit}:{annotation_mode}",
            on_done=self._on_repository_video_done, on_error=self._on_repository_video_error,
            cancellable=True,
        )
        if self._active_job_handle is not None:
            self.cancel_button.name = "Cancel video import"

    def _on_repository_video_done(self, result: dict[str, Any]) -> None:
        self._repository_video_running = False
        self._on_import_done(result)

    def _on_repository_video_error(self, exc: BaseException) -> None:
        self._repository_video_running = False
        self._on_error(exc)

    def _on_import_clicked(self, _event: Any = None) -> None:
        if self._details is None:
            self._set_status("Inspect the dataset before importing it.", "warning")
            return
        if not self._access_allows_data():
            message = (
                self._access_status.message if self._access_status is not None
                else "Hugging Face dataset access is required before importing files."
            )
            self._set_status(message, "warning")
            return

        repo_id = self.selected_repo
        if not repo_id:
            self._set_status("Select a dataset before importing it.", "warning")
            return

        splits = tuple(str(value) for value in (self.split_choices.value or []) if str(value).strip())
        if not splits:
            self._set_status("Select at least one split to import.", "warning")
            return

        if self._structured_profile():
            self._start_structured_video_import(repo_id=repo_id, splits=splits)
            return

        config = self._normalise_config(self.config_select.value)
        expected_id = self._suggested_base_id_for(repo_id, config)
        expected_name = self._suggested_base_name_for(repo_id, config)

        base_id_value = str(self.dataset_id.value or "").strip()
        base_name_value = str(self.dataset_name.value or "").strip()
        base_id = (
            expected_id if not base_id_value or base_id_value == self._last_suggested_dataset_id
            else base_id_value
        )
        base_name = (
            expected_name if not base_name_value or base_name_value == self._last_suggested_dataset_name
            else base_name_value
        )

        request = {
            "repo_id": repo_id, "config": config, "splits": splits,
            "active_split": str(self.active_after_import.value or ""), "base_id": base_id,
            "base_name": base_name, "image_column": self._resolved_image_column(),
            "label_column": self._resolved_label_column(),
            "id_column": self._empty_to_none(self.id_column.value),
            "max_rows": int(self.max_rows.value or 0),
            "revision": str(getattr(self._details, "revision", "") or ""), "token": self._token(),
            "trust_remote_code": bool(self.trust_remote_code.value),
            "write_parquet": bool(self.write_parquet.value), "require_label_semantics": bool(
                str(self.label_column.value or "") in {HF_FIFTYONE_GROUND_TRUTH, HF_PATH_LABEL}
            ),
        }

        authenticated = self._service().has_effective_token(request.get("token"))
        self._run_job(
            lambda cancel_token=None: self._import_selected_splits(
                request=request, cancel_token=cancel_token,
            ),
            title=f"Importing {repo_id}",
            key=(
                f"hf.import:{repo_id}:{config or 'default'}:"
                f"{','.join(splits)}:{request['max_rows']}"
            ), on_done=self._on_import_done, on_error=self._on_error, cancellable=True,
        )
        if not authenticated:
            warning = (
                "Import started without an authenticated Hugging Face token. Anonymous Hub limits are lower; "
                "add a token in Search options for large datasets. You can cancel this import at any time."
            )
            self.transfer_notice.object = warning
            self.transfer_notice.alert_type = "warning"
            self.transfer_notice.visible = True
            self._set_status(warning, "warning")

    def _import_selected_splits(
        self, *, request: dict[str, Any], cancel_token: Any = None,
    ) -> dict[str, Any]:
        repo_id = str(request["repo_id"])
        splits = tuple(str(value) for value in request["splits"])
        config = self._normalise_config(request.get("config"))
        active_split = str(request.get("active_split") or "")
        revision = str(request.get("revision") or "").strip() or None
        base_id = str(request.get("base_id") or "").strip()
        base_name = str(request.get("base_name") or "").strip()

        results: list[dict[str, Any]] = []
        previews: list[pd.DataFrame] = []
        total_rows = 0

        for index, split in enumerate(splits):
            if cancel_token is not None and cancel_token.cancelled():
                return {"cancelled": True, "results": results, "rows": total_rows}

            builder_identity = None
            if str(request.get("image_column") or "") != "__hf_file__":
                # A split imported with a different row limit or column selection
                # must not be mistaken for an exact existing registration.
                assets_dir = (self._service().assets_dir / slugify(repo_id)
                              / slugify(config or "default") / slugify(split))
                builder_identity = cache_identity(
                    repo_id=repo_id, revision=revision or "", config_name=config, split=split,
                    image_column=request.get("image_column"), label_column=request.get("label_column"),
                    id_column=request.get("id_column"), max_rows=int(request.get("max_rows") or 0),
                    assets_dir=assets_dir,
                    trust_remote_code=bool(request.get("trust_remote_code", False)),
                    label_semantics_version=HF_LABEL_SEMANTICS_VERSION,
                )
            existing_id = find_registered_hf_dataset(
                self.context, repo_id=repo_id, config_name=config, split=split, revision=revision,
                required_label_semantics_version=(
                    HF_LABEL_SEMANTICS_VERSION if bool(request.get("require_label_semantics")) else None
                ), required_builder_import_identity=builder_identity,
                allow_legacy_builder=int(request.get("max_rows") or 0) == 0,
            )
            if existing_id is not None:
                if active_split == split:
                    self.context.datasets.set_active(existing_id, origin="integrations.huggingface")
                result = {
                    "dataset_id": existing_id, "name": self._dataset_name(existing_id),
                    "rows": self.context.datasets.row_count(existing_id),
                    "backend": self.context.datasets.get_meta(existing_id).get("backend", "unknown"),
                    "repo_id": repo_id, "config_name": config or "", "split": split, "existing": True,
                    "cache_reused": True, "download_method": "existing_registration",
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
                    payload: dict[str, Any], *, split_index: int = index, split_name: str = split,
                ) -> None:
                    payload = dict(payload or {})
                    raw_percent = int(payload.get("percent", -1) if payload.get("percent") is not None else -1)
                    progress_known = bool(
                        payload.get(
                            "progress_known", raw_percent >= 0 and int(payload.get("total", 0) or 0) > 0
                            and int(payload.get("completed", 0) or 0) > 0,
                        )
                    )
                    if progress_known:
                        inner = max(0, min(100, raw_percent))
                        overall = int((split_index + (inner / 100.0)) / max(len(splits), 1) * 100)
                        payload["percent"] = overall
                    else:
                        payload["percent"] = -1
                    payload["progress_known"] = progress_known
                    payload["phase"] = (
                        f"{payload.get('phase', 'import')}:{split_name}"
                    )
                    payload["message"] = (
                        f"Split {split_index + 1}/{len(splits)} "
                        f"({split_name}): {payload.get('message', '')}"
                    )
                    self._set_progress_from_worker(payload)

                result = import_hf_image_dataset_as_manifest(
                    self.context, repo_id=repo_id, config_name=config, split=split,
                    dataset_id=split_dataset_id, dataset_name=split_dataset_name,
                    image_column=request.get("image_column"), label_column=request.get("label_column"),
                    id_column=request.get("id_column"), max_rows=int(request.get("max_rows", 0) or 0),
                    revision=revision, prefer_repository_files=str(self._details_mode or "") == "viewer",
                    token=request.get("token"),
                    trust_remote_code=bool(request.get("trust_remote_code", False)),
                    write_parquet=bool(request.get("write_parquet", True)),
                    set_active=active_split == split, cancel_token=cancel_token,
                    progress_callback=_split_progress,
                )

                if result.get("cancelled"):
                    return {
                        "cancelled": True, "results": results, "rows": total_rows, "repo_id": repo_id,
                        "config_name": config or "",
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

        preview = pd.DataFrame(
            [
                {
                    "dataset_id": item.get("dataset_id", ""), "split": item.get("split", ""),
                    "rows": item.get("rows", 0), "backend": item.get("backend", ""),
                    "existing": bool(item.get("existing", False)),
                    "active": item.get("split", "") == active_split,
                } for item in results
            ]
        )
        if previews:
            try:
                preview = pd.concat([preview] + previews, ignore_index=True, sort=False)
            except Exception:
                pass

        self._set_progress_from_worker(
            {
                "phase": "done", "completed": len(results), "total": len(splits), "percent": 100,
                "message": (
                    f"Prepared {len(results)} split dataset(s), "
                    f"{total_rows} total row(s)."
                ),
            }
        )

        return {
            "cancelled": False, "repo_id": repo_id, "config_name": config or "", "results": results,
            "rows": total_rows, "split_count": len(results), "active_split": active_split,
            "preview": preview,
        }

    def _apply_active_split_after_import(self, result: dict[str, Any]) -> str:
        """Reapply the chosen active split after all registrations and workspace reconciliation."""

        active_split = str(result.get("active_split") or "").strip()
        if not active_split:
            return ""

        target_dataset_id = ""
        for item in result.get("results") or ():
            if not isinstance(item, dict):
                continue
            if str(item.get("split") or "").strip() != active_split:
                continue

            target_dataset_id = str(item.get("dataset_id") or "").strip()
            if target_dataset_id:
                break

        if not target_dataset_id:
            return ""

        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return ""

        try:
            current_dataset_id = str(datasets.active_id() or "").strip()
        except Exception:
            current_dataset_id = ""

        if current_dataset_id != target_dataset_id:
            datasets.set_active(target_dataset_id, origin="integrations.huggingface")

        return target_dataset_id

    def _on_import_done(self, result: dict[str, Any]) -> None:
        self._set_busy(False)

        if result.get("cancelled"):
            self.transfer_notice.object = "Import cancelled. Any files already written remain in the Hugging Face cache and can be reused next time."
            self.transfer_notice.alert_type = "warning"
            self.transfer_notice.visible = True
            self._finish_progress("Import cancelled.")
            self._set_status("Import cancelled.", "warning")
            return

        self._apply_active_split_after_import(result)

        items = list(result.get("results") or [])
        imported = sum(1 for item in items if not bool(item.get("existing", False)))
        reused = len(items) - imported
        prepared_reused = sum(1 for item in items if bool(item.get("manifest_reused", False)))
        rows = int(result.get("rows", 0) or 0)
        active_split = str(result.get("active_split") or "")

        lines = []
        for item in items:
            dataset_id = html.escape(str(item.get("dataset_id") or ""))
            split_value = str(item.get("split") or "")
            split = html.escape(split_value)
            row_count = int(item.get("rows", 0) or 0)
            state = (
                "registration reused" if item.get("existing") else
                "prepared manifest reused" if item.get("manifest_reused") else "imported"
            )
            timings = dict(item.get("import_timings") or {})
            timing_parts = []
            for key, label in (
                ("split_loading_seconds", "HF load"), ("image_extraction_seconds", "images"),
                ("manifest_cache_check_seconds", "cache check"),
                ("manifest_finalization_seconds", "manifest"), ("registration_seconds", "register"),
                ("total_seconds", "total"),
            ):
                if key in timings:
                    timing_parts.append(f"{label} {float(timings[key]):.1f}s")
            timing_text = (" · " + html.escape(" · ".join(timing_parts))) if timing_parts else ""
            active = " · active" if split_value == active_split else ""
            mappings = dict(item.get("mappings") or {})
            mapping_text = f" · {len(mappings)} semantic mappings" if mappings else ""
            label_source = str(item.get("label_source") or "")
            label_text = f" · labels: {html.escape(label_source)}" if label_source else ""
            annotation_count = int(item.get("annotation_matches") or 0)
            if annotation_count:
                label_text += f" · {annotation_count:,} EPIC action annotations"
            lines.append(
                "<div style='min-width:0;padding:7px 0;border-top:1px solid #dcebe2;'>"
                f"<code title='{dataset_id}' style='display:block;min-width:0;color:#1f6a45;font-size:9.5px;line-height:1.35;overflow-wrap:anywhere;word-break:break-word;'>{dataset_id}</code>"
                f"<span style='display:block;margin-top:2px;color:#526071;font-size:9.3px;line-height:1.35;overflow-wrap:anywhere;word-break:break-word;'>{split} · {row_count:,} rows · {state}{active}{mapping_text}{label_text}{timing_text}</span>"
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
            Registered {imported} new split dataset(s) ({prepared_reused} from prepared manifests),
            reused {reused} existing registration(s), and prepared {rows:,} total rows.
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

        self.transfer_notice.visible = False
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
        revision = str(getattr(self._details, "revision", "") or "") if self._details else ""
        self.cache_pane.object = self._cache_html(
            self._service().cache_status(self.selected_repo, revision=revision or None)
        )

    @staticmethod
    def _cache_html(cache: Any) -> str:
        if cache is None:
            return ""

        cached = bool(getattr(cache, "cached", False))
        label = html.escape(str(getattr(cache, "label", "Not detected")))
        locations = list(getattr(cache, "locations", []) or [])
        revision = str(getattr(cache, "revision", "") or "")
        hub_files = int(getattr(cache, "hub_cached_files", 0) or 0)
        hub_bytes = int(getattr(cache, "hub_cached_bytes", 0) or 0)

        if cached:
            icon = "✓"
            title = "Local cache detected"
            extras = []
            if hub_files:
                extras.append(f"{hub_files:,} Hub file(s)")
            if hub_bytes:
                extras.append(f"{hub_bytes / (1024 ** 2):.1f} MiB")
            detail = (
                f"{label}." + (" " + " · ".join(extras) + "." if extras else "")
                + " Cached content may be partial; the selected split may still require preparation."
            )
            border = "#b8dec9"
            background = "#f2faf5"
            colour = "#1f6a45"
        else:
            icon = "↓"
            title = "Not cached locally"
            detail = ("Preview or import may download files that are not already available.")
            border = "#d8dee8"
            background = "#f8fafc"
            colour = "#526071"

        location_html = ""
        if revision:
            location_html += (
                "<div style='margin-top:4px;color:#687386;font-family:ui-monospace,monospace;font-size:8.8px;'>"
                + "Revision " + html.escape(revision[:12]) + "</div>"
            )
        if locations:
            location_html += (
                "<div style='margin-top:5px;color:#687386;font-size:9px;line-height:1.35;"
                "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
                + html.escape(str(locations[0])) + "</div>"
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

        def format_bytes(value: int | None) -> str:
            amount = max(int(value or 0), 0)
            if not amount:
                return "Unknown"
            number = float(amount)
            for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
                if number < 1024.0 or unit == "TiB":
                    return f"{int(number):,} {unit}" if unit == "B" else f"{number:.2f} {unit}"
                number /= 1024.0
            return f"{amount:,} B"

        config = str(self.config_select.value or "default")
        splits = list(
            details.splits_by_config.get(config) or details.splits_by_config.get("default")
            or self.split_choices.options or []
        )
        selected_splits = [str(value) for value in (self.split_choices.value or splits)]
        features = dict(
            details.features_by_config.get(config) or details.features_by_config.get("default") or {}
        )
        warnings = list(details.warnings or [])
        repository_stats = dict(getattr(details, "repository_stats_by_split", {}) or {})
        repository_complete = bool(getattr(details, "repository_inventory_complete", False))

        mode = self._mode_label(self._details_mode)
        if self._details_mode == "viewer" and repository_stats:
            mode = "Hugging Face Dataset Viewer + pinned repository inventory"

        image = str(self.image_column.value or "Not detected")
        selected_label = str(self.label_column.value or "")
        revision = str(getattr(details, "revision", "") or "")
        tasks = list(getattr(details, "task_categories", []) or []) + list(getattr(details, "task_ids", []) or [])
        inferred_task = str(getattr(details, "inferred_task_hint", "") or "")
        task_text = ", ".join(tasks[:4])
        if not task_text and inferred_task:
            task_text = inferred_task.replace("-", " " ).title() + " (inferred)"
        if not task_text:
            task_text = "Not declared"
        structured_profile = self._structured_profile()
        structured_profile_obj = (
            HFStructuredAnnotationProfile.from_dict(structured_profile) if structured_profile else None
        )

        # Prefer split-specific authoritative counts. Dataset Viewer /size wins;
        # repository inventory fills gaps for raw image repositories.
        viewer_counts = dict(
            getattr(details, "row_counts_by_config", {}).get(config)
            or getattr(details, "row_counts_by_config", {}).get("default") or {}
        )
        selected_rows = 0
        selected_rows_known = True
        selected_source_bytes = 0
        selected_source_bytes_known = True
        selected_files = 0
        labelled_files = 0
        label_count = 0
        ambiguous_filename_pairs = 0
        label_coverage_values: list[float] = []

        for split in selected_splits:
            if split in viewer_counts and int(viewer_counts.get(split) or 0) > 0:
                selected_rows += int(viewer_counts[split])
            elif split in repository_stats and repository_complete:
                selected_rows += int(repository_stats[split].get("files", 0) or 0)
            else:
                selected_rows_known = False

            row = repository_stats.get(split)
            if row:
                selected_files += int(row.get("files", 0) or 0)
                labelled_files += int(row.get("labelled_files", 0) or 0)
                label_count = max(label_count, int(row.get("label_count", 0) or 0))
                ambiguous_filename_pairs += int(row.get("ambiguous_filename_pairs", 0) or 0)
                label_coverage_values.append(float(row.get("label_coverage", 0.0) or 0.0))
                if bool(row.get("size_complete", False)):
                    selected_source_bytes += int(row.get("bytes", 0) or 0)
                else:
                    selected_source_bytes_known = False
            elif repository_stats:
                selected_source_bytes_known = False

        native_ground_truth = str(getattr(details, "ground_truth_field_hint", "") or "")
        if not selected_rows_known:
            total_rows = getattr(details, "total_rows_hint", None)
            rows_text = f"{int(total_rows):,} total" if total_rows else "Unknown"
        elif native_ground_truth and not viewer_counts and repository_stats:
            rows_text = (
                f"Up to {selected_rows:,} repository image candidate(s); "
                "native samples.json determines exact row membership on import"
            )
        else:
            rows_text = f"{selected_rows:,} selected row(s)"

        if repository_stats and selected_files:
            files_text = f"{selected_files:,} selected image file(s)"
            if not repository_complete:
                files_text += " (sample/incomplete inventory)"
        else:
            repo_file_hint = getattr(details, "total_repo_files_hint", None)
            files_text = f"{int(repo_file_hint):,} repository file(s)" if repo_file_hint else "Not enumerated"

        if repository_stats and selected_source_bytes_known and selected_source_bytes:
            source_bytes_text = f"{format_bytes(selected_source_bytes)} selected source"
        else:
            total_bytes = getattr(details, "total_bytes_hint", None)
            source_bytes_text = f"{format_bytes(total_bytes)} total" if total_bytes else "Unknown"

        if selected_label == HF_FIFTYONE_GROUND_TRUTH and native_ground_truth:
            label_text = (
                f"FiftyOne {native_ground_truth} Classification · exact samples.json annotations"
            )
            if label_count:
                label_text += f" · {label_count:,} path-label classes observed as a cross-check"
        elif selected_label == HF_PATH_LABEL and repository_stats:
            coverage = labelled_files / max(selected_files, 1) if selected_files else 0.0
            label_text = (
                f"Repository path inference · {labelled_files:,}/{selected_files:,} labelled "
                f"({coverage * 100:.1f}%)"
            )
            if label_count:
                label_text += f" · {label_count:,} distinct primary labels"
            if ambiguous_filename_pairs:
                label_text += f" · {ambiguous_filename_pairs:,} paired filenames retain a distinct secondary label"
        elif selected_label:
            label_text = selected_label
        elif native_ground_truth:
            label_text = f"FiftyOne {native_ground_truth} Classification available"
        else:
            best_coverage = max(label_coverage_values, default=0.0)
            if repository_stats and best_coverage >= 0.80:
                label_text = "Repository labels available; select path inference"
            else:
                label_text = "No label detected"

        parquet_files = getattr(details, "parquet_files_by_config", {}) or {}
        parquet_text = (
            "Partial conversion" if bool(getattr(details, "parquet_partial", False))
            else ("Available" if parquet_files else "Not reported")
        )

        if native_ground_truth:
            import_strategy = (
                "Pinned filtered snapshot · native FiftyOne samples.json defines rows and ground truth"
            )
        elif repository_stats and max(label_coverage_values, default=0.0) >= 0.80:
            import_strategy = "Pinned filtered snapshot · repository primary labels preserved"
        elif self._details_mode == "viewer" and parquet_files:
            import_strategy = "Structured dataset path (Parquet metadata available)"
        elif self._details_mode == "viewer":
            import_strategy = "Structured dataset fallback if repository labels are insufficient"
        else:
            import_strategy = mode

        if self._details_mode == "access_required":
            access_state = (
                str(self._access_status.state or "").replace("_", " ").title()
                if self._access_status is not None else "Access required"
            )
            gating_mode = (
                str(self._access_status.gating_mode or "gated").replace("_", " ").title()
                if self._access_status is not None else "Gated"
            )
            items = [
                ("Inspection", mode), ("Revision", revision[:12] if revision else "Not resolved"),
                ("Task", task_text),
                ("Dataset access", f"{gating_mode} · {access_state}"),
                ("Configuration", "Available after gated-file access is granted"),
                ("Splits", "Available after gated-file access is granted"),
                ("Rows", "Available after gated-file access is granted"), ("Repository files", files_text),
                ("Selected source size", "Available after gated-file access is granted"),
                ("Import strategy", "Request/receive Hugging Face access, then re-inspect"),
            ]
        elif structured_profile_obj is not None:
            variant = str(self.annotation_variant.value or "auto")
            selected_annotations = structured_profile_obj.files_for(
                splits=selected_splits, variant=variant,
            )
            selected_annotation_bytes = (
                sum(int(item.size or 0) for item in selected_annotations)
                if selected_annotations and all(item.size is not None for item in selected_annotations)
                else None
            )
            external_media = structured_profile_obj.external_media_summary
            if not external_media and structured_profile_obj.external_media:
                external_media = "Dataset card indicates that media lives outside the Hugging Face dataset repository."
            if not external_media:
                external_media = "No separate external-media location was detected in the compact card metadata."
            variant_label = {
                "auto": "Automatic · prefer extended metadata per split", "extended": "Extended metadata",
                "base": "Base annotations",
            }.get(variant, variant)
            items = [
                ("Inspection", "Pinned dataset card + repository annotation inventory"),
                ("Revision", revision[:12] if revision else "Not resolved"),
                ("Modality", " + ".join(value.title() for value in structured_profile_obj.modalities)),
                ("Task", ", ".join(value.replace("-", " ").title() for value in structured_profile_obj.task_hints)),
                ("Annotation format", structured_profile_obj.format_label), ("Configuration", config),
                ("Splits", ", ".join(structured_profile_obj.splits) or ", ".join(selected_splits) or "Not declared"),
                ("Selected annotation variant", variant_label),
                ("Annotation files", f"{len(selected_annotations):,} selected of {len(structured_profile_obj.annotation_files):,} discovered"),
                ("Selected annotation size", format_bytes(selected_annotation_bytes) if selected_annotation_bytes else "Unknown"),
                ("Rows / videos", "Not parsed during inspection; large annotation JSON stays external until a bounded video adapter consumes it"),
                ("Root fields", ", ".join(structured_profile_obj.root_fields) or "Not declared"),
                ("External media", external_media),
                ("Frame-sequence FPS", (str(structured_profile_obj.frame_media_fps) if structured_profile_obj.frame_media_fps else "Not inferred")),
                ("Native AstronomicAL status", ("Video registration ready through core.video; temporal masklets remain for vision.segmentation" if self._structured_video_registration_available() else "Annotation acquisition ready · enable core.video and/or provide frame media root/FPS")),
            ]
            for split_name, media_root in sorted(structured_profile_obj.frame_media_roots_by_split.items()):
                items.append((f"{split_name.title()} frame media root", media_root))
            for annotation_file in selected_annotations[:6]:
                split_name = annotation_file.split or "Annotation"
                variant_name = annotation_file.variant or "base"
                file_value = annotation_file.path
                if annotation_file.size is not None:
                    file_value += f" · {format_bytes(annotation_file.size)}"
                items.append((
                    f"{split_name.title()} annotation ({variant_name})",
                    file_value,
                ))
            if structured_profile_obj.external_media_urls:
                items.append(("External media URL", structured_profile_obj.external_media_urls[0]))
            if structured_profile_obj.extended_metadata_fields:
                items.append((
                    "Extended metadata", ", ".join(structured_profile_obj.extended_metadata_fields),
                ))
        else:
            items = [
                ("Inspection", mode), ("Revision", revision[:12] if revision else "Not resolved"),
                ("Task", task_text), ("Configuration", config),
                ("Splits", ", ".join(str(value) for value in splits) or "train"), ("Rows", rows_text),
                ("Repository files", files_text), ("Selected source size", source_bytes_text),
                ("Image source", image), ("Label source", label_text), ("Import strategy", import_strategy),
                ("Viewer columns", str(len(features))), ("Parquet", parquet_text),
            ]
        if native_ground_truth and self._details_mode != "access_required":
            predictions = list(getattr(details, "prediction_field_hints", []) or [])
            items.append(("Native format", "FiftyOne Dataset · metadata.json + samples.json"))
            items.append(("Ground-truth field", native_ground_truth))
            if predictions:
                items.append(("Declared prediction fields", ", ".join(predictions[:4])))

        feature_specs = dict(
            getattr(details, "feature_specs_by_config", {}).get(config)
            or getattr(details, "feature_specs_by_config", {}).get("default") or {}
        )
        label_spec = feature_specs.get(selected_label) if selected_label else None
        class_names: list[str] = []
        if isinstance(label_spec, dict) and str(label_spec.get("_type") or "") == "ClassLabel":
            class_names = [str(value) for value in (label_spec.get("names") or [])]
            if class_names:
                items.append(("Classes", f"{len(class_names):,} classes"))
        elif selected_label in {HF_PATH_LABEL, HF_FIFTYONE_GROUND_TRUTH}:
            examples = list(getattr(details, "repository_label_examples", []) or [])
            if examples:
                items.append(("Label examples", f"{len(examples):,} observed"))

        priority_keys = {
            "Dataset access", "Modality", "Task", "Annotation format", "Splits", "Rows", "Rows / videos",
            "Image source", "Label source", "Classes", "Native AstronomicAL status", "Import strategy",
        }
        primary_items = [(key, value) for key, value in items if key in priority_keys]
        secondary_items = [(key, value) for key, value in items if key not in priority_keys]
        if not primary_items:
            primary_items = items[:6]
            secondary_items = items[6:]

        def metadata_rows(entries: list[tuple[str, str]]) -> str:
            return "".join(
                f"""
                <div style="display:grid;grid-template-columns:minmax(92px,.48fr) minmax(0,1.52fr);gap:8px;align-items:start;padding:6px 0;border-top:1px solid #edf1f5;">
                  <div style="color:#7a8594;font-size:8.5px;font-weight:750;letter-spacing:.04em;text-transform:uppercase;line-height:1.35;">{html.escape(str(key))}</div>
                  <div style="min-width:0;color:#263244;font-size:10.2px;font-weight:600;line-height:1.4;overflow-wrap:anywhere;word-break:break-word;">{html.escape(str(value))}</div>
                </div>
                """
                for key, value in entries
            )

        primary_rows = metadata_rows(primary_items)
        technical_html = (
            """
            <details style="margin-top:7px;border-top:1px solid #dfe5ed;padding-top:6px;">
              <summary style="cursor:pointer;color:#526071;font-size:9.3px;font-weight:700;line-height:1.35;">
                Technical details
              </summary>
              <div style="margin-top:4px;min-width:0;">
            """
            + metadata_rows(secondary_items) + "</div></details>" if secondary_items else ""
        )

        description = str(getattr(details, "description", "") or "").strip()
        description_source = str(
            getattr(details, "description_source", "") or "Hugging Face dataset card"
        ).strip()
        card_sections = [
            dict(section) for section in (getattr(details, "card_sections", []) or [])
            if isinstance(section, dict) and str(section.get("text") or "").strip()
        ]
        card_source_url = str(
            getattr(details, "card_source_url", "")
            or f"https://huggingface.co/datasets/{self.selected_repo}"
        ).strip()

        description_html = ""
        if card_sections or description:
            sections_to_render = card_sections[:5]
            if not sections_to_render and description:
                sections_to_render = [{"title": "Dataset description", "text": description}]

            first_section = sections_to_render[0]
            first_title = str(first_section.get("title") or "Overview").strip()
            first_body = str(first_section.get("text") or "").strip()
            first_summary = first_body
            if len(first_summary) > 420:
                first_summary = first_summary[:417].rstrip() + "…"
            escaped_summary = html.escape(first_summary)
            escaped_summary = escaped_summary.replace("\n\n", "<br><br>").replace("\n", "<br>")

            more_sections = []
            for section in sections_to_render:
                title = str(section.get("title") or "Overview").strip()
                body = str(section.get("text") or "").strip()
                if len(body) > 2500:
                    body = body[:2497].rstrip() + "…"
                escaped_body = html.escape(body)
                escaped_body = escaped_body.replace("\n\n", "<br><br>")
                escaped_body = escaped_body.replace("\n• ", "<br>• ")
                escaped_body = escaped_body.replace("\n", "<br>")
                more_sections.append(
                    f"""
                    <div style="padding:6px 0;border-top:1px solid #edf1f5;">
                      <div style="color:#526071;font-size:9px;font-weight:740;line-height:1.35;">{html.escape(title)}</div>
                      <div style="margin-top:3px;color:#263244;font-size:9.8px;font-weight:500;line-height:1.45;overflow-wrap:anywhere;word-break:break-word;">{escaped_body}</div>
                    </div>
                    """
                )

            safe_url = html.escape(card_source_url, quote=True)
            source_link = (
                f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" '
                'style="color:#0f67a8;text-decoration:none;font-weight:650;">Open full card ↗</a>'
                if card_source_url else ""
            )
            show_more = len(sections_to_render) > 1 or len(first_body) > len(first_summary)
            more_html = (
                """
                <details style="margin-top:6px;">
                  <summary style="cursor:pointer;color:#526071;font-size:9px;font-weight:700;line-height:1.35;">
                    More from dataset card
                  </summary>
                  <div style="margin-top:4px;">
                """
                + "".join(more_sections) + "</div></details>" if show_more else ""
            )
            description_html = f"""
            <div style="margin:0 0 8px 0;padding:9px 10px;border:1px solid #dfe5ed;border-radius:7px;background:#ffffff;">
              <div style="display:flex;gap:8px;align-items:center;justify-content:space-between;flex-wrap:wrap;">
                <div style="color:#526071;font-size:8.7px;font-weight:760;letter-spacing:.04em;text-transform:uppercase;line-height:1.3;">About this dataset</div>
                <div style="font-size:8.7px;line-height:1.3;">{source_link}</div>
              </div>
              <div style="margin-top:5px;color:#526071;font-size:9px;font-weight:720;line-height:1.35;">{html.escape(first_title)}</div>
              <div style="margin-top:3px;color:#263244;font-size:9.9px;font-weight:500;line-height:1.45;overflow-wrap:anywhere;word-break:break-word;">{escaped_summary}</div>
              {more_html}
              <div style="margin-top:6px;color:#8a94a2;font-size:8.3px;line-height:1.3;">Source: {html.escape(description_source)}</div>
            </div>
            """

        feature_rows = []
        for column, feature_type in list(features.items())[:10]:
            feature_summary = self._compact_feature_summary(feature_type, feature_specs.get(column))
            feature_rows.append(
                "<div style='display:grid;grid-template-columns:minmax(82px,.65fr) minmax(0,1.35fr);gap:7px;padding:4px 0;border-top:1px solid #edf1f5;'>"
                f"<code style='font-size:8.8px;color:#0f5f9a;overflow-wrap:anywhere;'>{html.escape(str(column))}</code>"
                f"<span style='font-size:8.9px;color:#526071;overflow-wrap:anywhere;'>{html.escape(feature_summary)}</span>"
                "</div>"
            )
        schema_html = (
            "<div style='margin-top:8px;padding-top:6px;border-top:1px solid #dfe5ed;'>"
            "<div style='color:#526071;font-size:8.8px;font-weight:750;text-transform:uppercase;letter-spacing:.04em;'>Viewer feature schema</div>"
            + "".join(feature_rows) + (
                f"<div style='margin-top:4px;color:#7a8594;font-size:8.5px;'>Showing 10 of {len(features)} columns.</div>"
                if len(features) > 10 else ""
            ) + "</div>" if feature_rows else ""
        )

        class_details_html = ""
        if class_names:
            sample_names = class_names[:24]
            class_sample = ", ".join(html.escape(value) for value in sample_names)
            suffix = " …" if len(class_names) > len(sample_names) else ""
            class_details_html = f"""
            <details style="margin-top:7px;padding-top:6px;border-top:1px solid #dfe5ed;">
              <summary style="cursor:pointer;color:#526071;font-size:9px;font-weight:700;line-height:1.35;">
                Class labels · {len(class_names):,} total
              </summary>
              <div style="margin-top:5px;color:#687386;font-size:8.8px;line-height:1.45;overflow-wrap:anywhere;word-break:break-word;">
                {class_sample}{suffix}
              </div>
            </details>
            """

        self.structure_pane.object = f"""
        <div style="padding:9px 10px;border:1px solid #e2e7ee;border-radius:8px;background:#fbfcfe;overflow:hidden;">
          <div style="padding-bottom:7px;color:#263244;font-size:11.5px;font-weight:720;line-height:1.3;">Detected dataset structure</div>
          {description_html}
          <div style="min-width:0;">{primary_rows}</div>
          {technical_html}
          {schema_html}
          {class_details_html}
        </div>
        """

        if structured_profile_obj is not None and self._access_allows_data():
            warnings.insert(
                0,
                "Structured annotations stay external. Video-level rows can be registered through core.video when a frame-media root and annotation FPS are available; masklets remain for vision.segmentation.",
            )
        elif (
            not self._resolved_image_column() and self._details_mode == "viewer"
            and self._access_allows_data()
        ):
            warnings.insert(
                0,
                "This dataset has been inspected successfully, but the current importer still only registers image datasets. Generic text/audio/tabular import adapters remain a later ingestion tranche.",
            )
        if warnings:
            self.notice_pane.object = self._status_html(
                " · ".join(str(warning) for warning in warnings), "warning",
            )
        else:
            self.notice_pane.object = ""

    def _repository_video_relevant(self) -> bool:
        """Expose repository video tools for video datasets or explicit advanced use."""
        details = self._details
        if details is None or self._structured_profile():
            return False
        if self.selected_repo == EPIC_KITCHENS_REPO_ID:
            return True
        hints = [
            *(getattr(details, "task_ids", ()) or ()),
            *(getattr(details, "task_categories", ()) or ()),
            getattr(details, "inferred_task_hint", ""),
            getattr(details, "repository_format_hint", ""),
        ]
        return bool(self.advanced_toggle.value) or any(
            "video" in str(hint).casefold() or "action-recognition" in str(hint).casefold()
            for hint in hints
        )

    def _refresh_existing_registration_summary(self) -> None:
        self.repository_video_section.visible = bool(
            self.selected_repo and self._details is not None and self._repository_video_relevant()
        )
        self.repository_video_annotations.visible = self.selected_repo == EPIC_KITCHENS_REPO_ID
        self.repository_video_button.disabled = bool(
            self._busy or not self.selected_repo or self._details is None or not self._access_allows_data()
            or not self._core_video_registration_available()
        )
        if not self.selected_repo or self._details is None:
            return
        if self._structured_profile():
            self.import_button.name = "Register selected video splits"
            self.import_button.icon = "video"
            self._refresh_structured_registration_controls()
            return

        config = self._normalise_config(self.config_select.value)
        revision = str(getattr(self._details, "revision", "") or "") or None
        splits = [str(value) for value in (self.split_choices.value or [])]

        existing = []
        for split in splits:
            dataset_id = find_registered_hf_dataset(
                self.context, repo_id=self.selected_repo, config_name=config, split=split,
                revision=revision, required_label_semantics_version=(
                    HF_LABEL_SEMANTICS_VERSION if str(self.label_column.value or "")
                    in {HF_FIFTYONE_GROUND_TRUTH, HF_PATH_LABEL} else None
                ),
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
        self, func: Any, *, title: str, key: str, on_done: Any, on_error: Any, cancellable: bool = False,
    ) -> None:
        if self._busy:
            return

        self._cancel_requested = False
        self._active_job_cancellable = bool(cancellable)
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
            func, title=title, key=key, on_done=self._job_done_on_ui(on_done),
            on_error=self._job_error_on_ui(on_error),
        )
        self._job_handles.append(handle)
        if cancellable:
            self._active_job_handle = handle
            self.cancel_button.visible = True
            self.cancel_button.disabled = False
            self.cancel_button.name = "Cancel import"

    def _job_done_on_ui(self, on_done: Any):
        def _wrapped(result: Any) -> None:
            def _finish() -> None:
                self._clear_active_job()
                on_done(result)
                self._sync_progress_ui()
                self._ensure_progress_ticker(False)
                self._start_pending_inspection_if_ready()

            self._schedule_ui(_finish)

        return _wrapped

    def _job_error_on_ui(self, on_error: Any):
        def _wrapped(exc: BaseException) -> None:
            def _finish() -> None:
                self._clear_active_job()
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
                        self._sync_progress_ui, period=250, start=True,
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
        raw_percent = payload.get("percent", -1)
        try:
            percent = int(raw_percent) if raw_percent is not None else -1
        except Exception:
            percent = -1
        completed = int(payload.get("completed", 0) or 0)
        total = int(payload.get("total", 0) or 0)
        progress_known = bool(payload.get("progress_known", percent >= 0 and total > 0 and completed > 0))
        with self._progress_lock:
            phase = str(payload.get("phase", "") or "")
            previous_phase = str(self._progress_state.get("phase", "") or "")
            previous_count = int(self._progress_state.get("completed", 0) or 0)
            if phase != previous_phase or completed < previous_count:
                self._progress_phase_started_at = time.monotonic()
            self._progress_state.update(
                {
                    "active": True, "phase": phase, "completed": completed,
                    "total": total, "percent": percent, "progress_known": progress_known,
                    "progress_basis": str(payload.get("progress_basis", "") or ""),
                    "message": str(payload.get("message", "") or ""),
                    "current_file": str(payload.get("current_file", "") or ""),
                    "severity": str(payload.get("severity", "info") or "info"),
                    "retry_after_seconds": payload.get("retry_after_seconds"),
                    "transfer_bytes": int(payload.get("transfer_bytes", 0) or 0),
                    "selected_total_bytes": int(payload.get("selected_total_bytes", 0) or 0),
                    "expected_download_bytes": int(payload.get("expected_download_bytes", 0) or 0),
                    "cached_bytes": int(payload.get("cached_bytes", 0) or 0),
                }
            )

    def _start_progress(self, message: str) -> None:
        with self._progress_lock:
            self._progress_started_at = time.monotonic()
            self._progress_phase_started_at = self._progress_started_at
            self._progress_state = {
                "active": True, "phase": "starting", "completed": 0, "total": 0, "percent": -1,
                "progress_known": False, "progress_basis": "indeterminate", "message": message,
                "current_file": "", "severity": "info", "retry_after_seconds": None,
            }

    def _finish_progress(self, message: str) -> None:
        with self._progress_lock:
            self._progress_state.update(
                {
                    "active": False, "phase": "done", "percent": 100, "progress_known": True,
                    "progress_basis": "complete", "message": message, "current_file": "",
                }
            )
        self._sync_progress_ui()
    def _fail_progress(self, message: str) -> None:
        with self._progress_lock:
            self._progress_state.update(
                {"active": False, "phase": "error", "message": message, "current_file": ""}
            )
        self._sync_progress_ui()
    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {secs:02d}s"

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
            if progress_mount is not None and progress_mount.objects:
                progress_mount[:] = []
            return

        self.progress_bar.visible = True
        self.progress_text.visible = True
        progress_mount = getattr(self, "progress_mount", None)
        if progress_mount is not None and list(progress_mount.objects) != [self.progress_bar, self.progress_text]:
            progress_mount[:] = [self.progress_bar, self.progress_text]

        raw_percent = int(state.get("percent", -1) if state.get("percent") is not None else -1)
        progress_known = bool(state.get("progress_known", raw_percent >= 0))
        percent = max(0, min(100, raw_percent)) if progress_known else -1
        completed = int(state.get("completed", 0) or 0)
        total = int(state.get("total", 0) or 0)
        progress_basis = str(state.get("progress_basis", "") or "")
        phase = html.escape(str(state.get("phase", "") or "working"))
        message = html.escape(str(state.get("message", "") or ""))
        current_file = str(state.get("current_file", "") or "")
        severity = str(state.get("severity", "info") or "info")
        now = time.monotonic()
        elapsed = max(0.0, now - self._progress_started_at)
        stage_elapsed = max(0.0, now - self._progress_phase_started_at)
        remaining = None
        if (
            phase.startswith("builder_rows") and total > completed > 0
            and completed >= 100 and stage_elapsed >= 5.0
        ):
            remaining = min(7 * 24 * 3600.0, stage_elapsed * (total - completed) / completed)

        if severity == "warning" and message:
            notice_key = f"{phase}:{message}"
            if notice_key != self._last_progress_notice:
                self._last_progress_notice = notice_key
                self.transfer_notice.object = html.unescape(message)
                self.transfer_notice.alert_type = "warning"
                self.transfer_notice.visible = True
                self._set_status(html.unescape(message), "warning")
        elif phase in {"snapshot_download", "snapshot_retry", "snapshot_cached"} and not self._cancel_requested:
            self._set_status("Transferring Hugging Face data…", "busy")

        self.progress_bar.value = percent if progress_known else -1
        if hasattr(self.progress_bar, "active"):
            self.progress_bar.active = not progress_known

        count = f" {completed:,}/{total:,}" if total else ""
        timing = f" · elapsed {self._format_elapsed(elapsed)}"
        if remaining is not None:
            timing += f" · approx. {self._format_elapsed(remaining)} for image extraction"
        file_line = ""
        if current_file:
            short = (current_file if len(current_file) <= 90 else "…" + current_file[-89:])
            file_line = (
                "<div style='margin-top:2px;color:#7a8594;font-family:ui-monospace,monospace;"
                f"font-size:8.8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{html.escape(short)}</div>"
            )

        self.progress_text.object = f"""
        <div style="padding:4px 1px 0;color:#526071;font-size:9.8px;line-height:1.35;">
          <strong style="color:#0f6fbd;">{f'{percent}%' if progress_known else 'Working…'}</strong>
          <span style="color:#7a8594;"> · {phase}{count}{(' · ' + html.escape(progress_basis.replace('_', ' '))) if progress_known and progress_basis else ''}</span>
          <span> · {message}{timing}</span>
          {file_line}
        </div>
        """
    def _on_cancel_clicked(self, _event: Any = None) -> None:
        handle = self._active_job_handle
        if handle is None or not self._active_job_cancellable:
            return
        self._cancel_requested = True
        self.cancel_button.disabled = True
        self.cancel_button.name = "Cancelling…"
        try:
            handle.cancel()
        except Exception as exc:
            self._set_status(f"Could not request cancellation: {exc}", "danger")
            return

        if getattr(self, "_repository_video_running", False):
            message = (
                "Cancellation requested. The current Hugging Face file transfer may complete "
                "before the importer stops; cached files can be reused on retry."
            )
        else:
            message = (
                "Cancellation requested. AstronomicAL is terminating the isolated Hugging Face "
                "snapshot process now; files already committed to the HF cache are kept for reuse."
            )
        self.transfer_notice.object = message
        self.transfer_notice.alert_type = "warning"
        self.transfer_notice.visible = True
        self._set_status(message, "warning")
        self._set_progress_from_worker(
            {
                "phase": "cancelling", "completed": 0, "total": 0, "percent": -1, "progress_known": False,
                "progress_basis": "cancelling", "message": message, "severity": "warning",
            }
        )

    def _clear_active_job(self) -> None:
        handle = self._active_job_handle
        if handle is not None:
            try:
                self._job_handles.remove(handle)
            except ValueError:
                pass
        self._active_job_handle = None
        self._active_job_cancellable = False
        self.cancel_button.disabled = True
        self.cancel_button.visible = False
        self.cancel_button.name = "Cancel import"

    def _service(self) -> HuggingFaceDatasetService:
        services = getattr(self.context, "services", None)
        if services is not None:
            try:
                service = services.get("integrations.huggingface.client")
                if service is not None:
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
        splits = [str(value) for value in (self.split_choices.value or []) if str(value).strip()]
        options = OrderedDict({"Do not change active dataset": ""})
        for split in splits:
            options[split] = split

        previous = str(self.active_after_import.value or "")
        self.active_after_import.options = options

        restored = str(
            self._restored_state.get("active_after_import")
            or self._restored_state.get("active_split_after_import") or ""
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
            self.selected_repo, self._normalise_config(self.config_select.value),
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
            self.selected_repo, self._normalise_config(self.config_select.value),
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
            "{}</th>".format(html.escape(column), html.escape(column)) for column in columns
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
                        safe, safe,
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
            self.search_button, self.inspect_button, self.access_check_button, self.preview_button,
            self.import_button, self.cache_annotations_button, self.repository_video_button,
        ):
            button.disabled = bool(busy)

        if busy:
            if self._active_job_cancellable and self._active_job_handle is not None:
                self.cancel_button.visible = True
                self.cancel_button.disabled = self._cancel_requested
        else:
            self.repository_video_button.disabled = bool(
                self._details is None or not self._access_allows_data()
                or not self._core_video_registration_available()
            )
            self.inspect_button.disabled = not bool(self.selected_repo)
            self.access_check_button.disabled = False
            self.preview_button.disabled = (self._details is None or not self._access_allows_data())
            structured = bool(self._structured_profile())
            if structured:
                self._refresh_structured_registration_controls()
            else:
                self.import_button.disabled = (
                    self._details is None or not bool(self.split_choices.value)
                    or not bool(self._resolved_image_column()) or not self._access_allows_data()
                )
            self.cache_annotations_button.disabled = (
                not structured or not bool(self.split_choices.value) or not self._access_allows_data()
            )

    def _set_status(self, message: str, status_type: str) -> None:
        self.status.object = self._status_html(message, status_type)
    def _on_error(self, exc: BaseException) -> None:
        self._set_busy(False)
        if self._cancel_requested:
            message = "Transfer cancelled."
            self.transfer_notice.object = (
                "Transfer cancelled. Any files already written remain in the Hugging Face cache and can be reused next time."
            )
            self.transfer_notice.alert_type = "warning"
            self.transfer_notice.visible = True
            self._finish_progress(message)
            self._set_status(message, "warning")
            return
        message = " ".join(str(exc).split())
        self._fail_progress(message)
        self._set_status(message, "danger")

    def _set_select_options(self, widget: Any, options: Any, *, preferred: Any = None) -> None:
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
        if mode == "viewer":
            return "Hugging Face Dataset Viewer metadata"
        if mode == "builder":
            return "structured Hugging Face dataset rows"
        if mode == "files":
            return "repository image files"
        if mode == "access_required":
            return "Hugging Face metadata (gated files not yet accessible)"
        if mode == "cache":
            return "local Hugging Face cache"
        return "automatic detection"

    @staticmethod
    def _selected_html(repo_id: str) -> str:
        safe_repo = html.escape(str(repo_id))
        return f"""
        <div style="display:flex;align-items:center;gap:10px;width:100%;height:42px;min-width:0;overflow:hidden;">
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
            AstronomicAL will inspect configurations, splits, typed features, task hints,
            existing registrations, revision, and local cache availability.
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
    def _access_status_html(status: HFDatasetAccessStatus) -> str:
        state = str(status.state or "access_unknown")
        gating_mode = str(status.gating_mode or "gated")
        gating_label = {
            "auto": "Gated · automatic approval", "manual": "Gated · manual approval",
            "gated": "Gated dataset",
        }.get(gating_mode, "Gated dataset")

        palettes = {
            "checking": ("#eef5fb", "#0f5f9a", "#bfd7ea", "…"),
            "access_granted": ("#ecf8f1", "#1f6a45", "#b8dec9", "✓"),
            "authentication_required": ("#fff7d6", "#8a5d00", "#ead58b", "!"),
            "authentication_invalid": ("#fff0ee", "#a42920", "#efc1bc", "×"),
            "access_request_required": ("#fff7d6", "#8a5d00", "#ead58b", "!"),
            "pending_approval": ("#fff7d6", "#8a5d00", "#ead58b", "…"),
            "access_rejected": ("#fff0ee", "#a42920", "#efc1bc", "×"),
            "region_restricted": ("#fff0ee", "#a42920", "#efc1bc", "×"),
            "access_unknown": ("#f5f7fa", "#526071", "#d8dee8", "?"),
        }
        background, colour, border, icon = palettes.get(state, palettes["access_unknown"])

        title = {
            "checking": "Checking dataset access", "access_granted": "Dataset access granted",
            "authentication_required": "Hugging Face sign-in required",
            "authentication_invalid": "Hugging Face token needs attention",
            "access_request_required": "Dataset access request required",
            "pending_approval": "Dataset access request pending",
            "access_rejected": "Dataset access request rejected",
            "region_restricted": "Dataset unavailable in this region",
            "access_unknown": "Dataset access could not be verified",
        }.get(state, "Dataset access")

        auth_text = (
            f"Authenticated as {status.authenticated_user}"
            if status.authenticated_user else (
                "Hugging Face token detected" if status.token_available
                else "No Hugging Face token detected"
            )
        )
        safe_url = html.escape(
            str(
                status.access_url
                or f"https://huggingface.co/datasets/{status.repo_id}"
            ), quote=True,
        )
        access_link = (
            f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" '
            'style="display:inline-flex;align-items:center;min-height:28px;padding:0 9px;'
            'border:1px solid currentColor;border-radius:6px;color:inherit;text-decoration:none;'
            'font-size:9.4px;font-weight:720;">Open access page ↗</a>'
        )
        token_link = (
            '<a href="https://huggingface.co/settings/tokens" target="_blank" '
            'rel="noopener noreferrer" style="display:inline-flex;align-items:center;'
            'min-height:28px;padding:0 9px;border:1px solid #c7d2df;border-radius:6px;'
            'color:#526071;text-decoration:none;font-size:9.4px;font-weight:680;'
            'background:#ffffff;">Token settings ↗</a>'
        )
        browser_note = (
            "Hugging Face requires user access requests and any repository-specific "
            "consent fields to be completed in the browser. AstronomicAL does not "
            "submit those terms on your behalf."
        )

        return f"""
        <div style="box-sizing:border-box;width:100%;padding:10px 11px;border:1px solid {border};border-radius:8px;background:{background};color:{colour};overflow:hidden;">
          <div style="display:flex;align-items:flex-start;gap:9px;">
            <div style="display:flex;align-items:center;justify-content:center;flex:0 0 22px;width:22px;height:22px;border-radius:50%;background:#ffffff;font-size:10px;font-weight:800;">{icon}</div>
            <div style="flex:1 1 auto;min-width:0;">
              <div style="display:flex;gap:8px;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;">
                <div>
                  <div style="font-size:10.6px;font-weight:760;line-height:1.3;">{html.escape(title)}</div>
                  <div style="margin-top:2px;font-size:9px;font-weight:690;line-height:1.35;opacity:.86;">{html.escape(gating_label)} · {html.escape(auth_text)}</div>
                </div>
              </div>
              <div style="margin-top:6px;font-size:9.8px;font-weight:560;line-height:1.45;overflow-wrap:anywhere;">{html.escape(str(status.message or ""))}</div>
              <div style="margin-top:6px;font-size:8.8px;line-height:1.4;opacity:.86;">{html.escape(browser_note)}</div>
              <div style="display:flex;gap:7px;flex-wrap:wrap;margin-top:8px;">
                {access_link}
                {token_link}
              </div>
            </div>
          </div>
        </div>
        """

    @staticmethod
    def _status_html(message: str, status_type: str) -> str:
        palettes = {
            "info": ("#e9f4fb", "#0f5f9a", "#bdd9ec", "i"), "busy": ("#eaf2ff", "#1d4ed8", "#bfd7ff", "…"),
            "success": ("#ecf8f1", "#1f6a45", "#b8dec9", "✓"),
            "warning": ("#fff7d6", "#8a5d00", "#ead58b", "!"),
            "danger": ("#fff0ee", "#a42920", "#efc1bc", "×"),
        }
        resolved_type = status_type or "info"
        background, colour, border, icon = palettes.get(resolved_type, palettes["info"])
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
