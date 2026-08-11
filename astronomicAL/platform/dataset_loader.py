from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any, Iterable, Optional
import re

import panel as pn
import param
from panel.reactive import ReactiveHTML

from astronomicAL.platform.dataset_loader_styles import (
    DATASET_LOADER_CSS,
    DATASET_LOADER_INPUT_STYLESHEET,
    DATASET_LOADER_LOG_STYLESHEET,
)
from astronomicAL.platform.dataset_sources import DuckDBParquetDatasetSource
from astronomicAL.platform.modal_utils import close_template_modal
from astronomicAL.platform.tabular_import import (
    CacheAssessment,
    DiskSpaceAssessment,
    assess_disk_space,
    assess_parquet_cache,
    convert_source_to_parquet,
    default_parquet_path,
    detect_source_format,
    estimate_parquet_size,
    inspect_source_subresources,
    is_convertible_format,
    metadata_path_for_parquet,
    source_format_label,
    source_stem,
)

_LOADER_ID = "platform.dataset_loader"


class DatasetProgressLog(ReactiveHTML):
    """Read-only progress log that tails output until the user scrolls away.

    The browser owns the scroll-follow state so rapid worker progress does not
    require a Python round-trip for every wheel/scroll event. While the viewport
    is at the bottom, new output follows the newest line. As soon as the user
    scrolls upward, their exact scrollTop is preserved across subsequent value
    updates. Returning to the bottom automatically resumes following.
    """

    value = param.String(default="")

    _template = """
<div
  id="viewport"
  class="al-dataset-loader-log-viewport"
  role="log"
  aria-live="off"
  aria-label="Dataset conversion progress"
  tabindex="0"
>
  <pre id="log" class="al-dataset-loader-log-pre"></pre>
</div>
"""

    _stylesheets = [DATASET_LOADER_LOG_STYLESHEET]

    _scripts = {
        "render": """
state.auto_follow = true;
state.scroll_tolerance = 18;
state.update_follow_state = () => {
  const distance = viewport.scrollHeight - viewport.clientHeight - viewport.scrollTop;
  state.auto_follow = distance <= state.scroll_tolerance;
};
state.on_scroll = () => state.update_follow_state();
viewport.addEventListener("scroll", state.on_scroll, {passive: true});
log.textContent = data.value || "";
requestAnimationFrame(() => {
  viewport.scrollTop = viewport.scrollHeight;
  state.update_follow_state();
});
""",
        "value": """
const previousTop = viewport.scrollTop;
const shouldFollow = state.auto_follow !== false;
log.textContent = data.value || "";
requestAnimationFrame(() => {
  if (shouldFollow) {
    viewport.scrollTop = viewport.scrollHeight;
  } else {
    viewport.scrollTop = previousTop;
  }
});
""",
        "remove": """
if (state.on_scroll) {
  viewport.removeEventListener("scroll", state.on_scroll);
}
""",
    }


@dataclass(frozen=True)
class DatasetSourceInfo:
    path: Path
    source_format: str
    size_bytes: Optional[int]
    modified_ns: Optional[int]

    @property
    def label(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class DatasetImportRequest:
    source_path: str
    source_format: str
    dataset_id: str
    dataset_name: str
    parquet_path: Optional[str] = None
    source_subresource: Optional[str] = None
    regenerate: bool = False
    replace_dataset_id: Optional[str] = None


@dataclass(frozen=True)
class DatasetImportResult:
    dataset_id: str
    dataset_name: str
    source_path: str
    source_format: str
    backend: Optional[str]
    rows: Optional[int]
    columns: tuple[str, ...]
    cache_path: Optional[str]
    cache_created: Optional[bool]
    regenerated: bool = False
    updated_existing: bool = False

    def event_payload(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "updated_dataset_id": self.dataset_id if self.updated_existing else None,
            "name": self.dataset_name,
            "rows": self.rows,
            "columns": list(self.columns),
            "source_path": self.source_path,
            "source_format": self.source_format,
            "cache_path": self.cache_path,
            "cache_created": self.cache_created,
            "regenerated": self.regenerated,
            "updated_existing": self.updated_existing,
            "loader_id": _LOADER_ID,
            "optimise_data": False,
            "backend": self.backend,
            "origin": _LOADER_ID,
            "change": "source.regenerated" if self.updated_existing else "dataset.loaded",
        }


class DatasetImportService:
    """Platform-owned conversion and DatasetManager registration service.

    The service deliberately keeps source conversion separate from registration:
    a cache can be rebuilt atomically, then an existing DatasetSource can be
    refreshed without creating a second dataset identity.
    """

    def __init__(self, context: Any) -> None:
        if context is None:
            raise ValueError("DatasetImportService requires an AppContext.")
        if getattr(context, "datasets", None) is None:
            raise ValueError("DatasetImportService requires context.datasets.")
        self.context = context

    def import_dataset(
        self,
        *,
        request: DatasetImportRequest,
        cancel_token: Any = None,
        progress_callback: Any = None,
        progress_state_callback: Any = None,
    ) -> DatasetImportResult:
        self._raise_if_cancelled(cancel_token)
        source = Path(request.source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Dataset source does not exist: {source}")

        detected_format = detect_source_format(source)
        if detected_format is None:
            raise ValueError(f"Unsupported dataset format: {source.name}")
        if detected_format != request.source_format:
            raise ValueError(
                f"Dataset format changed before import: expected {request.source_format}, "
                f"detected {detected_format}."
            )

        replace_dataset_id = request.replace_dataset_id or None
        requested_id = normalise_dataset_id(request.dataset_id)
        if replace_dataset_id:
            dataset_id = str(replace_dataset_id)
            if dataset_id not in set(self.context.datasets.list_ids()):
                raise KeyError(f"Dataset to refresh is no longer registered: {dataset_id}")
            dataset_name = self._dataset_name(dataset_id)
        else:
            dataset_id = requested_id
            if dataset_id in set(self.context.datasets.list_ids()):
                raise ValueError(f"Dataset id is already registered: {dataset_id}")
            dataset_name = str(request.dataset_name or source.stem).strip() or dataset_id

        self._raise_if_cancelled(cancel_token)

        cache_path: Optional[str] = None
        cache_created: Optional[bool] = None
        registration_meta: dict[str, Any] = {
            "source_path": str(source),
            "source_format": detected_format,
            "loader_id": _LOADER_ID,
            "optimise_data": False,
        }

        if detected_format == "parquet":
            if request.regenerate:
                raise ValueError("A Parquet source is already optimised and cannot be regenerated.")
            parquet = source
            cache_created = False
        else:
            if not request.parquet_path:
                raise ValueError("A Parquet destination is required for this source format.")
            parquet = Path(request.parquet_path).expanduser().resolve()
            cache_path = str(parquet)

            assessment = assess_parquet_cache(
                source,
                parquet,
                source_subresource=request.source_subresource,
            )
            if assessment.requires_regeneration and not request.regenerate:
                raise RuntimeError(
                    "The generated Parquet is stale. Use Regenerate Parquet before loading it."
                )

            should_convert = request.regenerate or not parquet.is_file()
            if should_convert:
                conversion = convert_source_to_parquet(
                    source,
                    parquet_path=parquet,
                    source_format=detected_format,
                    dataset_id=dataset_id,
                    source_subresource=request.source_subresource,
                    overwrite=bool(request.regenerate),
                    progress_callback=progress_callback,
                    progress_state_callback=progress_state_callback,
                    cancel_token=cancel_token,
                )
                cache_created = bool(conversion.created)
                registration_meta.update(dict(conversion.metadata or {}))
            else:
                metadata_path = metadata_path_for_parquet(parquet)
                registration_meta.update(_read_json(metadata_path))
                cache_created = False
                self._progress(
                    progress_callback,
                    f"Reusing verified/generated Parquet cache: {parquet}",
                )

            registration_meta.update(
                {
                    "source_path": str(source),
                    "source_format": detected_format,
                    "source_subresource": request.source_subresource or None,
                    "cache_path": str(parquet),
                    "parquet_path": str(parquet),
                    "backend": "duckdb_parquet",
                    "loader_id": _LOADER_ID,
                    "optimise_data": False,
                }
            )

        columns_hint = _metadata_columns(registration_meta)
        row_count_hint = _metadata_row_count(registration_meta)

        if replace_dataset_id:
            old_meta = _safe_meta(self.context.datasets, dataset_id)
            merged_meta = dict(old_meta)
            merged_meta.update(registration_meta)
            merged_meta.pop("dataset_id", None)
            merged_meta.pop("name", None)
            source_object = DuckDBParquetDatasetSource(
                parquet,
                dataset_name=dataset_name,
                columns_hint=columns_hint,
                row_count_hint=row_count_hint,
            )
            self.context.datasets.ensure_source_registered(
                dataset_id,
                source_object,
                name=dataset_name,
                **merged_meta,
            )
            self._progress(
                progress_callback,
                f"Refreshed existing dataset source without changing dataset id: {dataset_id}",
            )
        else:
            registration_meta.pop("dataset_id", None)
            registration_meta.pop("name", None)
            self.context.datasets.register_parquet(
                dataset_id,
                parquet,
                name=dataset_name,
                **registration_meta,
            )
            self._progress(progress_callback, f"Registered dataset: {dataset_id}")

        rows = _safe_row_count(self.context.datasets, dataset_id)
        columns = tuple(_safe_columns(self.context.datasets, dataset_id))
        meta = _safe_meta(self.context.datasets, dataset_id)
        backend = _optional_text(meta.get("backend"))

        return DatasetImportResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            source_path=str(source),
            source_format=detected_format,
            backend=backend,
            rows=rows,
            columns=columns,
            cache_path=cache_path,
            cache_created=cache_created,
            regenerated=bool(request.regenerate),
            updated_existing=bool(replace_dataset_id),
        )

    def _dataset_name(self, dataset_id: str) -> str:
        try:
            dataset = self.context.datasets.get(dataset_id)
            return str(getattr(dataset, "name", None) or dataset_id)
        except Exception:
            return str(dataset_id)

    @staticmethod
    def _progress(callback: Any, message: str) -> None:
        print(f"[AstronomicAL loader] {message}", flush=True)
        if callback is not None:
            try:
                callback(str(message))
            except Exception:
                pass

    @staticmethod
    def _raise_if_cancelled(cancel_token: Any) -> None:
        if cancel_token is None:
            return
        for method_name in (
            "raise_if_cancelled",
            "raise_if_cancellation_requested",
            "check_cancelled",
        ):
            method = getattr(cancel_token, method_name, None)
            if callable(method):
                method()
                return
        for attribute_name in ("cancelled", "is_cancelled", "cancellation_requested"):
            value = getattr(cancel_token, attribute_name, False)
            if callable(value):
                value = value()
            if value:
                raise RuntimeError("Dataset import was cancelled.")


class DatasetLoaderController:
    """Dataset modal for server-side discovery, conversion and registration."""

    def __init__(
        self,
        *,
        context: Any,
        template: Any,
        data_directory: str | Path = "data",
        browse_root: str | Path | None = None,
        upload_directory: str | Path | None = None,
    ) -> None:
        if context is None:
            raise ValueError("DatasetLoaderController requires an AppContext.")
        if getattr(context, "datasets", None) is None:
            raise ValueError("DatasetLoaderController requires context.datasets.")

        self.context = context
        self.template = template
        self.data_directory = Path(data_directory).expanduser()
        # Retain these constructor keywords for compatibility with any existing
        # application wiring, but do not use a browser FileInput/native-dialog
        # upload path. A browser cannot expose the client's real absolute path;
        # large/external data must already be visible to the AstronomicAL server.
        self.browse_root = (
            Path(browse_root).expanduser() if browse_root is not None else None
        )
        self.upload_directory = (
            Path(upload_directory).expanduser() if upload_directory is not None else None
        )
        self.service = DatasetImportService(context)

        self._disposed = False
        self._busy = False
        self._updating_fields = False
        self._updating_parquet_path = False
        self._parquet_path_user_edited = False
        self._regen_armed = False
        self._watchers: list[Any] = []
        self._job_handle: Any = None
        self._job_document: Any = None
        self._sources: dict[str, DatasetSourceInfo] = {}
        self._source_directory_error: Optional[str] = None
        self._effective_source_directory: Optional[Path] = None
        self._existing_dataset_id: Optional[str] = None
        self._last_result: Optional[DatasetImportResult] = None
        self._subresource_error: Optional[str] = None
        self._progress_lines: deque[str] = deque(maxlen=5000)
        self._conversion_progress: dict[str, Any] = {}
        self._cache_assessment_memo: dict[tuple[Any, ...], CacheAssessment] = {}

        self._install_css()
        self._build_widgets()
        self._build_view()
        self._wire_events()
        self.refresh_sources(preserve_selection=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        if self._disposed:
            return
        self.refresh_sources(preserve_selection=True)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for watcher in list(self._watchers):
            try:
                watcher.inst.param.unwatch(watcher)
            except Exception:
                try:
                    watcher.cls.param.unwatch(watcher)
                except Exception:
                    pass
        self._watchers.clear()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _install_css() -> None:
        marker = ".al-dataset-loader-card"
        if any(marker in css for css in pn.config.raw_css):
            return
        pn.config.raw_css.append(DATASET_LOADER_CSS)

    def _build_widgets(self) -> None:
        self.directory_path_input = pn.widgets.TextInput(
            name="",
            placeholder="Folder path — blank uses data/",
            sizing_mode="stretch_width",
            height=36,
            margin=(0, 8, 0, 0),
            stylesheets=[DATASET_LOADER_INPUT_STYLESHEET],
        )
        self.directory_path_input.description = (
            "Directory visible to the AstronomicAL server. Leave blank to scan the default data/ folder."
        )

        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            icon="refresh",
            button_type="default",
            width=100,
            height=36,
            margin=(0, 0, 0, 0),
        )
        self.refresh_button.description = "Rescan the current source directory."

        self.source_select = pn.widgets.Select(
            name="",
            options=OrderedDict({"No supported files found": ""}),
            value="",
            sizing_mode="stretch_width",
            height=36,
            margin=(8, 0, 0, 0),
            stylesheets=[DATASET_LOADER_INPUT_STYLESHEET],
        )
        self.source_select.description = "Choose a supported tabular file from the current directory."

        self.dataset_name_input = pn.widgets.TextInput(
            name="",
            placeholder="Dataset display name",
            sizing_mode="stretch_width",
            height=36,
            stylesheets=[DATASET_LOADER_INPUT_STYLESHEET],
        )
        self.dataset_id_input = pn.widgets.TextInput(
            name="",
            placeholder="dataset_id",
            sizing_mode="stretch_width",
            height=36,
            stylesheets=[DATASET_LOADER_INPUT_STYLESHEET],
        )
        self.dataset_id_input.description = (
            "Stable identifier used by panels, mappings, events, and workspaces."
        )

        self.subresource_select = pn.widgets.Select(
            name="",
            options=OrderedDict(),
            value=None,
            sizing_mode="stretch_width",
            height=36,
            visible=False,
            stylesheets=[DATASET_LOADER_INPUT_STYLESHEET],
        )

        self.parquet_path_input = pn.widgets.TextInput(
            name="",
            placeholder="Generated Parquet path",
            sizing_mode="stretch_width",
            height=36,
            stylesheets=[DATASET_LOADER_INPUT_STYLESHEET],
        )
        self.parquet_path_input.description = (
            "Generated Parquet location. Change this for large datasets or other storage volumes."
        )

        self.source_summary = pn.pane.HTML("", sizing_mode="stretch_width")
        self.behaviour_summary = pn.pane.HTML("", sizing_mode="stretch_width")
        self.cache_summary = pn.pane.HTML("", sizing_mode="stretch_width")
        self.disk_summary = pn.pane.HTML("", sizing_mode="stretch_width")

        self.validation_alert = pn.pane.Alert(
            "",
            alert_type="warning",
            visible=False,
            sizing_mode="stretch_width",
            margin=(8, 0, 0, 0),
        )
        self.cache_alert = pn.pane.Alert(
            "",
            alert_type="warning",
            visible=False,
            sizing_mode="stretch_width",
            margin=(8, 0, 0, 0),
        )

        self.status_pane = pn.pane.HTML(
            self._status_html(
                "Ready",
                "Choose a source to review its conversion and cache status.",
                "idle",
            ),
            sizing_mode="stretch_width",
        )
        self.result_pane = pn.pane.HTML(
            "",
            visible=False,
            sizing_mode="stretch_width",
            css_classes=["al-dataset-loader-result"],
        )

        self.conversion_summary = pn.pane.HTML(
            self._conversion_progress_html(None),
            visible=False,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-loader-conversion-summary-host"],
        )

        # The log is independently scrollable, so it does not need to reserve
        # 220px of vertical space in the loader before an import has started.
        # 120px still shows several progress lines while substantially reducing
        # the loader's baseline height.
        self.progress_log = DatasetProgressLog(
            value="",
            height=120,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.regen_button = pn.widgets.Button(
            name="Regenerate Parquet",
            icon="refresh-dot",
            button_type="warning",
            width=190,
            height=38,
            visible=False,
            disabled=True,
        )
        self.regen_button.description = (
            "Rebuild the generated Parquet from the selected source. Requires confirmation."
        )

        self.activate_existing_button = pn.widgets.Button(
            name="Activate existing dataset",
            icon="database-check",
            button_type="primary",
            width=220,
            height=38,
            visible=False,
        )
        self.close_button = pn.widgets.Button(
            name="Close",
            button_type="light",
            width=110,
            height=38,
            margin=(0, 10, 0, 0),
        )
        self.import_button = pn.widgets.Button(
            name="Import dataset",
            icon="database-import",
            button_type="primary",
            width=170,
            height=38,
            disabled=True,
        )

    def _build_view(self) -> None:
        heading = pn.pane.HTML(
            """
<div class="al-modal-titlebar">
  <div class="al-modal-heading">Add dataset</div>
  <div class="al-modal-subtitle">
    Load server-visible tabular data; non-Parquet sources are cached as lazy
    Parquet-backed DatasetSources.
  </div>
</div>
""",
            sizing_mode="stretch_width",
            height=60,
            margin=(0, 0, 0, 0),
        )

        self.subresource_label = self._field_label("Sheet / table")

        source_section = pn.Column(
            self._section_heading(
                "1. Choose source",
                "Choose a server-visible folder and supported file. Leave Path blank to use AstronomicAL's data/ folder.",
            ),
            pn.Row(
                self.directory_path_input,
                self.refresh_button,
                sizing_mode="stretch_width",
                margin=(8, 0, 0, 0),
            ),
            self.source_select,
            pn.Spacer(height=2),
            self._field_label("Dataset name"),
            self.dataset_name_input,
            pn.Spacer(height=2),
            self._field_label("Dataset ID"),
            self.dataset_id_input,
            self.subresource_label,
            self.subresource_select,
            self.validation_alert,
            sizing_mode="stretch_width",
            css_classes=["al-dataset-loader-section"],
        )
        self.subresource_label.visible = False

        cache_section = pn.Column(
            self._section_heading(
                "2. Parquet cache",
                "Review the output location, cache freshness, and disk-space estimate.",
            ),
            pn.Spacer(height=4),
            self._field_label("Parquet location"),
            self.parquet_path_input,
            pn.Spacer(height=4),
            self.cache_summary,
            self.cache_alert,
            pn.Spacer(height=2),
            self.disk_summary,
            pn.Row(
                self.regen_button,
                sizing_mode="stretch_width",
                margin=(8, 0, 0, 0),
            ),
            sizing_mode="stretch_width",
            css_classes=["al-dataset-loader-section"],
        )

        details_section = pn.Column(
            self._section_heading(
                "3. Source details",
                "Detected before registration.",
            ),
            pn.Spacer(height=6),
            self.source_summary,
            pn.Spacer(height=8),
            self.behaviour_summary,
            sizing_mode="stretch_width",
            css_classes=["al-dataset-loader-section"],
        )

        status_section = pn.Column(
            self._section_heading(
                "4. Status and conversion progress",
                "Progress is mirrored here, in the terminal, and in Runtime Status.",
            ),
            pn.Spacer(height=6),
            self.status_pane,
            pn.Spacer(height=6),
            self.conversion_summary,
            pn.Spacer(height=6),
            self.progress_log,
            pn.Spacer(height=6),
            self.result_pane,
            pn.Row(
                self.activate_existing_button,
                sizing_mode="stretch_width",
                margin=(8, 0, 0, 0),
            ),
            sizing_mode="stretch_width",
            css_classes=["al-dataset-loader-section"],
        )

        left = pn.Column(
            source_section,
            cache_section,
            sizing_mode="fixed",
            width=500,
            margin=(0, 7, 0, 0),
        )
        right = pn.Column(
            details_section,
            status_section,
            sizing_mode="fixed",
            width=520,
            margin=(0, 0, 0, 7),
        )

        # Intrinsic body height for normal desktop displays. CSS contracts this
        # further when the available viewport is shorter, while retaining
        # overflow-y on the body so the footer never disappears below the screen.
        body = pn.Row(
            left,
            right,
            sizing_mode="fixed",
            width=1060,
            height=626,
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-loader-body", "al-dataset-loader-grid"],
        )

        footer = pn.Row(
            pn.layout.HSpacer(),
            self.close_button,
            self.import_button,
            sizing_mode="fixed",
            width=1060,
            height=50,
            align="center",
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-loader-footer", "al-modal-footer"],
        )

        # 760px comprises:
        #   24px card vertical padding
        #   60px heading
        #   626px scrollable body
        #   50px footer
        #
        # This is also the concrete Bokeh layout height passed through
        # mount_template_modal(), avoiding the previous 1072px wrapper model.
        self.view = pn.Column(
            heading,
            body,
            footer,
            sizing_mode="fixed",
            width=1060,
            height=760,
            margin=(0, 0, 0, 0),
            css_classes=["al-modal-card", "al-dataset-loader-card"],
            styles={"box-sizing": "border-box", "overflow": "hidden"},
        )

    def _wire_events(self) -> None:
        self._watchers.append(self.directory_path_input.param.watch(self._on_directory_path_changed, "value"))
        self._watchers.append(self.source_select.param.watch(self._on_source_changed, "value"))
        self._watchers.append(self.dataset_name_input.param.watch(self._on_identity_changed, "value"))
        self._watchers.append(self.dataset_id_input.param.watch(self._on_identity_changed, "value"))
        self._watchers.append(self.parquet_path_input.param.watch(self._on_parquet_path_changed, "value"))
        self._watchers.append(self.subresource_select.param.watch(self._on_subresource_changed, "value"))
        self.refresh_button.on_click(self._on_refresh_clicked)
        self.import_button.on_click(self._on_import_clicked)
        self.regen_button.on_click(self._on_regen_clicked)
        self.activate_existing_button.on_click(self._on_activate_existing)
        self.close_button.on_click(self._on_close_clicked)

    # ------------------------------------------------------------------
    # Source discovery
    # ------------------------------------------------------------------

    def _resolve_source_directory(self) -> tuple[Optional[Path], Optional[str]]:
        raw = str(self.directory_path_input.value or "").strip()
        candidate = self.data_directory if not raw else Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate.absolute()

        if not resolved.exists():
            return None, f"Folder does not exist: {resolved}"
        if not resolved.is_dir():
            return None, f"Path is not a directory: {resolved}"
        return resolved, None

    def refresh_sources(self, *, preserve_selection: bool = True) -> None:
        previous = str(self.source_select.value or "") if preserve_selection else ""
        directory, error = self._resolve_source_directory()
        self._effective_source_directory = directory
        self._source_directory_error = error

        sources: list[DatasetSourceInfo] = []
        if directory is not None:
            try:
                sources = discover_dataset_sources(directory)
            except OSError as exc:
                self._source_directory_error = f"Could not read folder {directory}: {exc}"
                sources = []

        self._sources = {str(source.path): source for source in sources}

        self._updating_fields = True
        try:
            if self._source_directory_error:
                self.source_select.options = OrderedDict({"Folder unavailable": ""})
                self.source_select.value = ""
                self.source_select.disabled = True
                self.source_select.description = self._source_directory_error
            elif sources:
                options: OrderedDict[str, str] = OrderedDict(
                    (source.label, str(source.path)) for source in sources
                )
                self.source_select.options = options
                self.source_select.disabled = False
                self.source_select.value = (
                    previous if previous in self._sources else str(sources[0].path)
                )
                self.source_select.description = (
                    f"Supported tabular files in {directory}."
                )
            else:
                self.source_select.options = OrderedDict({"No supported files found": ""})
                self.source_select.value = ""
                self.source_select.disabled = True
                self.source_select.description = (
                    f"No supported tabular files were found in {directory}."
                    if directory is not None
                    else "No source directory is available."
                )

            if directory is not None:
                self.refresh_button.description = f"Rescan {directory}."
            else:
                self.refresh_button.description = "Rescan the current source directory."
        finally:
            self._updating_fields = False

        self._apply_source_defaults(force=not bool(previous))
        self._refresh_subresources()
        self._refresh_state()

    def _on_refresh_clicked(self, _event: Any) -> None:
        self._cache_assessment_memo.clear()
        self.refresh_sources(preserve_selection=True)

    def _on_directory_path_changed(self, _event: Any) -> None:
        if self._updating_fields or self._busy:
            return
        self._last_result = None
        self.result_pane.visible = False
        self._regen_armed = False
        self._parquet_path_user_edited = False
        self._cache_assessment_memo.clear()
        self.refresh_sources(preserve_selection=False)

    def _on_source_changed(self, _event: Any) -> None:
        if self._updating_fields:
            return
        self._last_result = None
        self.result_pane.visible = False
        self._regen_armed = False
        self._parquet_path_user_edited = False
        self._apply_source_defaults(force=True)
        self._refresh_subresources()
        self._refresh_state()

    def _on_identity_changed(self, _event: Any) -> None:
        if self._updating_fields:
            return
        if not self._parquet_path_user_edited:
            self._set_default_parquet_path()
        self._refresh_state()

    def _on_parquet_path_changed(self, _event: Any) -> None:
        if self._updating_parquet_path:
            return
        self._parquet_path_user_edited = True
        self._regen_armed = False
        self._refresh_state()

    def _on_subresource_changed(self, _event: Any) -> None:
        if self._updating_fields:
            return
        self._regen_armed = False
        self._refresh_state()

    def _apply_source_defaults(self, *, force: bool) -> None:
        source = self._selected_source()
        if source is None:
            if force:
                self._updating_fields = True
                try:
                    self.dataset_name_input.value = ""
                    self.dataset_id_input.value = ""
                    self.parquet_path_input.value = ""
                finally:
                    self._updating_fields = False
            return

        stem = source_stem(source.path)
        display_name = stem.replace("_", " ").replace("-", " ").title()
        dataset_id = self._next_dataset_id(normalise_dataset_id(stem))
        self._updating_fields = True
        try:
            if force or not self.dataset_name_input.value.strip():
                self.dataset_name_input.value = display_name
            if force or not self.dataset_id_input.value.strip():
                self.dataset_id_input.value = dataset_id
        finally:
            self._updating_fields = False
        self._set_default_parquet_path()

    def _set_default_parquet_path(self) -> None:
        source = self._selected_source()
        if source is None:
            return
        if source.source_format == "parquet":
            value = str(source.path)
        else:
            dataset_id = normalise_dataset_id(self.dataset_id_input.value or source_stem(source.path))
            value = str(default_parquet_path(source.path, dataset_id))
            existing_id = self._registered_dataset_for_source(source.path)
            if existing_id:
                meta = _safe_meta(self.context.datasets, existing_id)
                existing_cache = meta.get("cache_path") or meta.get("parquet_path")
                if existing_cache:
                    value = str(existing_cache)
        self._updating_parquet_path = True
        try:
            self.parquet_path_input.value = value
        finally:
            self._updating_parquet_path = False

    def _refresh_subresources(self) -> None:
        source = self._selected_source()
        self._subresource_error = None
        self._updating_fields = True
        try:
            if source is None:
                self.subresource_select.options = OrderedDict()
                self.subresource_select.value = None
                self.subresource_select.visible = False
                self.subresource_label.visible = False
                return
            if source.source_format == "fits":
                self.subresource_select.options = OrderedDict({"HDU 1": "1"})
                self.subresource_select.value = "1"
                self.subresource_select.visible = False
                self.subresource_label.visible = False
                return

            try:
                values = inspect_source_subresources(source.path, source.source_format)
            except Exception as exc:
                values = ()
                self._subresource_error = str(exc).strip() or type(exc).__name__

            if values:
                options = OrderedDict((str(value), str(value)) for value in values)
                self.subresource_select.options = options
                self.subresource_select.value = next(iter(options.values()))
                self.subresource_select.visible = True
                self.subresource_label.visible = True
            else:
                self.subresource_select.options = OrderedDict()
                self.subresource_select.value = None
                self.subresource_select.visible = False
                self.subresource_label.visible = False
        finally:
            self._updating_fields = False

    # ------------------------------------------------------------------
    # State / validation
    # ------------------------------------------------------------------

    def _refresh_state(self) -> None:
        source = self._selected_source()
        self._existing_dataset_id = (
            self._registered_dataset_for_source(source.path) if source is not None else None
        )
        if source is not None and not self._parquet_path_user_edited:
            self._set_default_parquet_path()

        cache = self._cache_assessment(source)
        disk = self._disk_assessment(source, cache)
        self.source_summary.object = self._source_summary_html(source)
        self.behaviour_summary.object = self._behaviour_html(source)
        self.cache_summary.object = self._cache_summary_html(source, cache)
        self.disk_summary.object = self._disk_summary_html(source, disk)

        duplicate = self._existing_dataset_id is not None
        problem = None if duplicate else self._validation_problem(source, disk)
        self.validation_alert.visible = problem is not None
        self.validation_alert.object = problem or ""

        stale = cache is not None and cache.requires_regeneration
        self.cache_alert.visible = bool(stale or self._regen_armed)
        if self._regen_armed:
            self.cache_alert.alert_type = "warning"
            self.cache_alert.object = (
                "Regeneration will replace the existing generated Parquet at the path above. "
                "Click ‘Confirm regenerate’ to continue. The source file is never modified."
            )
        elif stale:
            self.cache_alert.alert_type = "danger"
            self.cache_alert.object = cache.message
        else:
            self.cache_alert.object = ""

        existing_is_active = self._existing_dataset_is_active()
        self.activate_existing_button.visible = duplicate and not self._busy
        self.activate_existing_button.disabled = self._busy or existing_is_active or stale
        if duplicate:
            existing_name = self._dataset_name(self._existing_dataset_id)
            self.activate_existing_button.name = (
                f"{existing_name} is active" if existing_is_active else f"Activate {existing_name}"
            )

        convertible = source is not None and is_convertible_format(source.source_format)
        cache_exists = cache is not None and cache.exists
        self.regen_button.visible = bool(convertible)
        self.regen_button.disabled = (
            self._busy
            or not convertible
            or not cache_exists
            or (disk is not None and disk.enough_space is False)
        )
        self.regen_button.name = "Confirm regenerate" if self._regen_armed else "Regenerate Parquet"

        self.import_button.disabled = (
            self._busy
            or problem is not None
            or duplicate
            or stale
            or (disk is not None and disk.enough_space is False)
        )
        self.directory_path_input.disabled = self._busy
        self.refresh_button.disabled = self._busy
        self.source_select.disabled = self._busy or not bool(self._sources)
        self.dataset_name_input.disabled = self._busy or source is None or duplicate
        self.dataset_id_input.disabled = self._busy or source is None or duplicate
        self.parquet_path_input.disabled = self._busy or source is None or source.source_format == "parquet"
        self.subresource_select.disabled = self._busy

        if self._busy:
            return
        if duplicate and existing_is_active:
            copy = (
                f"This source is registered as {self._dataset_name(self._existing_dataset_id)} "
                f"({self._existing_dataset_id}) and is already active."
            )
            if stale:
                copy += " Its generated Parquet is stale; regenerate it before continuing analysis."
            self.status_pane.object = self._status_html("Dataset already active", copy, "warning" if stale else "success")
        elif duplicate:
            copy = (
                f"This source is already registered as {self._dataset_name(self._existing_dataset_id)} "
                f"({self._existing_dataset_id})."
            )
            if stale:
                copy += " Its generated Parquet is stale; regenerate it before activating."
            else:
                copy += " Activate the existing dataset rather than creating a duplicate registration."
            self.status_pane.object = self._status_html("Source already registered", copy, "warning")
        elif stale:
            self.status_pane.object = self._status_html(
                "Parquet regeneration required",
                cache.message,
                "warning",
            )
        elif problem:
            self.status_pane.object = self._status_html("Import not ready", problem, "warning")
        elif source is not None:
            self.status_pane.object = self._status_html(
                "Ready to import",
                "Review the cache and disk checks, then start the background import.",
                "idle",
            )
        else:
            directory = self._effective_source_directory
            if self._source_directory_error:
                copy = self._source_directory_error
            elif directory is not None:
                copy = f"No supported tabular files were found in {directory}."
            else:
                copy = f"Add supported data to {self.data_directory.resolve()} or choose another server-visible folder."
            self.status_pane.object = self._status_html(
                "No supported source",
                copy,
                "warning",
            )

    def _validation_problem(
        self,
        source: Optional[DatasetSourceInfo],
        disk: Optional[DiskSpaceAssessment] = None,
    ) -> Optional[str]:
        if source is None:
            if self._source_directory_error:
                return self._source_directory_error
            return "Choose a supported dataset source."
        if not source.path.is_file():
            return "The selected source is no longer available. Refresh the file list."
        if self._subresource_error:
            return self._subresource_error
        if self.subresource_select.visible and not self.subresource_select.value:
            return "Choose the spreadsheet sheet, HDF5 key, or SQLite table to import."

        name = self.dataset_name_input.value.strip()
        if not name:
            return "Enter a dataset name."
        raw_id = self.dataset_id_input.value.strip()
        if not raw_id:
            return "Enter a dataset ID."
        normalised = normalise_dataset_id(raw_id)
        if raw_id != normalised:
            return (
                "Dataset IDs may contain lowercase letters, numbers, and underscores. "
                f"Use `{normalised}`."
            )
        if normalised in set(self.context.datasets.list_ids()):
            return f"Dataset ID `{normalised}` is already registered."

        if is_convertible_format(source.source_format):
            raw_parquet = str(self.parquet_path_input.value or "").strip()
            if not raw_parquet:
                return "Choose a Parquet output location."
            parquet = Path(raw_parquet).expanduser()
            if canonical_path(parquet) == canonical_path(source.path):
                return "The generated Parquet path must differ from the source file."
            if parquet.suffix.lower() not in {".parquet", ".pq"}:
                return "The generated cache path must end in .parquet or .pq."
            if disk is not None and disk.enough_space is False:
                return "The selected Parquet destination does not have enough estimated free space."
        return None

    def _cache_assessment(self, source: Optional[DatasetSourceInfo]) -> Optional[CacheAssessment]:
        if source is None or not is_convertible_format(source.source_format):
            return None
        raw_parquet = str(self.parquet_path_input.value or "").strip()
        if not raw_parquet:
            return None
        parquet = Path(raw_parquet).expanduser()
        metadata = metadata_path_for_parquet(parquet)
        try:
            source_stat = source.path.stat()
            source_key = (source_stat.st_size, source_stat.st_mtime_ns)
        except OSError:
            source_key = (None, None)
        try:
            parquet_stat = parquet.stat()
            parquet_key = (parquet_stat.st_size, parquet_stat.st_mtime_ns)
        except OSError:
            parquet_key = (None, None)
        try:
            metadata_stat = metadata.stat()
            metadata_key = (metadata_stat.st_size, metadata_stat.st_mtime_ns)
        except OSError:
            metadata_key = (None, None)
        key = (
            str(source.path),
            source_key,
            canonical_path(parquet),
            parquet_key,
            metadata_key,
            self._source_subresource(),
        )
        cached = self._cache_assessment_memo.get(key)
        if cached is not None:
            return cached
        assessment = assess_parquet_cache(
            source.path,
            parquet,
            source_subresource=self._source_subresource(),
        )
        self._cache_assessment_memo = {key: assessment}
        return assessment

    def _disk_assessment(
        self,
        source: Optional[DatasetSourceInfo],
        cache: Optional[CacheAssessment],
    ) -> Optional[DiskSpaceAssessment]:
        if source is None or not is_convertible_format(source.source_format):
            return None
        raw_parquet = str(self.parquet_path_input.value or "").strip()
        if not raw_parquet:
            return None
        estimate = estimate_parquet_size(
            source.source_format,
            source.size_bytes,
            existing_parquet_path=raw_parquet if cache is not None and cache.exists else None,
        )
        return assess_disk_space(raw_parquet, estimate)

    # ------------------------------------------------------------------
    # Import / regeneration
    # ------------------------------------------------------------------

    def _on_import_clicked(self, _event: Any) -> None:
        self._start_import(regenerate=False)

    def _on_regen_clicked(self, _event: Any) -> None:
        if self._busy:
            return
        source = self._selected_source()
        cache = self._cache_assessment(source)
        if source is None or cache is None or not cache.exists:
            self._refresh_state()
            return
        if not self._regen_armed:
            self._regen_armed = True
            self._refresh_state()
            return
        self._start_import(regenerate=True)

    def _start_import(self, *, regenerate: bool) -> None:
        if self._busy:
            return
        source = self._selected_source()
        disk = self._disk_assessment(source, self._cache_assessment(source))
        problem = None if self._existing_dataset_id else self._validation_problem(source, disk)
        if source is None or problem is not None:
            self._refresh_state()
            return
        if disk is not None and disk.enough_space is False:
            self._refresh_state()
            return

        replace_dataset_id = self._existing_dataset_id if regenerate else None
        request = DatasetImportRequest(
            source_path=str(source.path),
            source_format=source.source_format,
            dataset_id=(replace_dataset_id or self.dataset_id_input.value.strip()),
            dataset_name=(self._dataset_name(replace_dataset_id) if replace_dataset_id else self.dataset_name_input.value.strip()),
            parquet_path=(
                None if source.source_format == "parquet" else str(self.parquet_path_input.value).strip()
            ),
            source_subresource=self._source_subresource(),
            regenerate=regenerate,
            replace_dataset_id=replace_dataset_id,
        )

        self._regen_armed = False
        self._clear_progress()
        self._set_conversion_progress(
            {
                "phase": "starting",
                "label": "Starting dataset conversion" if source.source_format != "parquet" else "Registering Parquet source",
                "rows_completed": 0,
                "rows_total": None,
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
                "rows_per_second": None,
                "average_rows_per_second": None,
                "tmp_size_bytes": 0,
                "rss_gib": None,
            }
        )
        self._append_progress(
            f"{'Regenerating' if regenerate else 'Importing'} {request.source_path}"
        )
        self._set_busy(True)
        self._job_document = self._current_document()
        self.status_pane.object = self._status_html(
            "Regenerating Parquet" if regenerate else "Import in progress",
            (
                "The operation is running through the platform job runner. This modal mirrors "
                "converter progress; Runtime Status remains the application-wide job monitor."
            ),
            "busy",
        )
        self.result_pane.visible = False

        jobs = getattr(self.context, "jobs", None)
        if jobs is not None and hasattr(jobs, "submit"):
            try:
                self._job_handle = jobs.submit(
                    self._run_import_job,
                    title=(
                        f"Regenerate dataset cache: {request.dataset_name}"
                        if regenerate
                        else f"Import dataset: {request.dataset_name}"
                    ),
                    key=self._job_key(request),
                    on_done=self._on_job_done,
                    on_error=self._on_job_error,
                    request=request,
                )
                return
            except Exception as exc:
                self._job_handle = None
                self._finish_error(exc)
                return

        try:
            result = self._run_import_job(request=request, cancel_token=None)
        except BaseException as exc:
            self._finish_error(exc)
        else:
            self._finish_success(result)

    def _run_import_job(
        self,
        *,
        request: DatasetImportRequest,
        cancel_token: Any = None,
    ) -> DatasetImportResult:
        return self.service.import_dataset(
            request=request,
            cancel_token=cancel_token,
            progress_callback=self._progress_from_worker,
            progress_state_callback=self._progress_state_from_worker,
        )

    def _progress_from_worker(self, message: str) -> None:
        if self._disposed:
            return
        text = str(message)

        def _apply() -> None:
            if not self._disposed:
                self._append_progress(text)

        document = self._job_document
        if document is None:
            _apply()
            return
        try:
            document.add_next_tick_callback(_apply)
        except Exception:
            _apply()

    def _progress_state_from_worker(self, state: dict[str, Any]) -> None:
        if self._disposed:
            return
        payload = dict(state or {})

        def _apply() -> None:
            if not self._disposed:
                self._set_conversion_progress(payload)

        document = self._job_document
        if document is None:
            _apply()
            return
        try:
            document.add_next_tick_callback(_apply)
        except Exception:
            _apply()

    def _on_job_done(self, *args: Any, **kwargs: Any) -> None:
        try:
            result = _extract_job_result(args, kwargs)
            if not isinstance(result, DatasetImportResult):
                if isinstance(result, dict):
                    result = DatasetImportResult(**result)
                else:
                    raise TypeError(
                        "Dataset import job returned an unexpected result: "
                        f"{type(result).__name__}"
                    )
        except BaseException as exc:
            self._finish_error(exc)
            return
        self._finish_success(result)

    def _on_job_error(self, *args: Any, **kwargs: Any) -> None:
        self._finish_error(_extract_job_error(args, kwargs))

    def _finish_success(self, result: DatasetImportResult) -> None:
        self._job_handle = None
        self._last_result = result
        self._cache_assessment_memo.clear()
        try:
            self._clear_selection_for_dataset_switch()
            topic = "dataset.updated" if result.updated_existing else "dataset.loaded"
            self._publish(topic, result.event_payload())
            self.context.datasets.set_active(result.dataset_id, origin=_LOADER_ID, force=result.updated_existing)
        except BaseException as exc:
            self._set_busy(False)
            self._finish_error(exc)
            return

        self._append_progress("Dataset ready.")
        final_progress = dict(self._conversion_progress)
        final_progress.update(
            {
                "phase": "complete",
                "label": "Dataset ready",
                "rows_completed": result.rows if result.rows is not None else final_progress.get("rows_completed"),
                "rows_total": result.rows if result.rows is not None else final_progress.get("rows_total"),
                "eta_seconds": 0.0,
            }
        )
        self._set_conversion_progress(final_progress)
        self._set_busy(False)
        self.refresh_sources(preserve_selection=True)
        self.status_pane.object = self._status_html(
            "Dataset refreshed" if result.updated_existing else "Dataset ready",
            (
                f"{result.dataset_name} is registered and active. "
                + (
                    "The existing dataset identity was preserved while its Parquet source was refreshed."
                    if result.updated_existing
                    else "Panels continue to access it through DatasetManager and semantic mappings."
                )
            ),
            "success",
        )
        self.result_pane.object = self._result_html(result)
        self.result_pane.visible = True

    def _finish_error(self, exc: BaseException) -> None:
        self._job_handle = None
        self._set_busy(False)
        message = str(exc).strip() or type(exc).__name__
        self._append_progress(f"ERROR: {message}")
        failed_progress = dict(self._conversion_progress)
        failed_progress.update({"phase": "error", "label": "Import failed", "eta_seconds": None})
        self._set_conversion_progress(failed_progress)
        self.status_pane.object = self._status_html("Import failed", message, "error")
        self.result_pane.visible = False
        self._refresh_state_controls_only()

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.import_button.name = "Importing dataset…" if busy else "Import dataset"
        # Keep the footer geometry stable. The previous busy-state label was much
        # wider than the fixed Close button and could paint underneath the import
        # button. The continuation behaviour is conveyed as a tooltip instead.
        self.close_button.name = "Close"
        self.close_button.description = (
            "Close this modal. The import continues as a background job and remains visible in Runtime Status."
            if busy
            else "Close the dataset loader."
        )
        self._refresh_state_controls_only()

    def _refresh_state_controls_only(self) -> None:
        source = self._selected_source()
        cache = self._cache_assessment(source)
        disk = self._disk_assessment(source, cache)
        duplicate = self._existing_dataset_id is not None
        problem = self._validation_problem(source, disk) if not self._busy and not duplicate else None
        stale = cache is not None and cache.requires_regeneration
        convertible = source is not None and is_convertible_format(source.source_format)

        self.import_button.disabled = self._busy or duplicate or problem is not None or stale
        self.directory_path_input.disabled = self._busy
        self.refresh_button.disabled = self._busy
        self.source_select.disabled = self._busy or not bool(self._sources)
        self.dataset_name_input.disabled = self._busy or source is None or duplicate
        self.dataset_id_input.disabled = self._busy or source is None or duplicate
        self.parquet_path_input.disabled = self._busy or source is None or source.source_format == "parquet"
        self.regen_button.disabled = (
            self._busy or not convertible or cache is None or not cache.exists or (disk is not None and disk.enough_space is False)
        )
        existing_is_active = self._existing_dataset_is_active()
        self.activate_existing_button.visible = duplicate and not self._busy
        self.activate_existing_button.disabled = self._busy or existing_is_active or stale

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def _clear_progress(self) -> None:
        self._progress_lines.clear()
        self.progress_log.value = ""
        self._conversion_progress = {}
        self.conversion_summary.object = self._conversion_progress_html(None)
        self.conversion_summary.visible = False

    def _append_progress(self, message: str) -> None:
        self._progress_lines.append(str(message))
        self.progress_log.value = "\n".join(self._progress_lines)

    def _set_conversion_progress(self, state: dict[str, Any]) -> None:
        merged = dict(self._conversion_progress)
        merged.update(dict(state or {}))
        self._conversion_progress = merged
        self.conversion_summary.object = self._conversion_progress_html(merged)
        self.conversion_summary.visible = True

    @staticmethod
    def _current_document() -> Any:
        try:
            return pn.state.curdoc
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Existing dataset / modal helpers
    # ------------------------------------------------------------------

    def _on_activate_existing(self, _event: Any) -> None:
        dataset_id = self._existing_dataset_id
        if not dataset_id:
            return
        cache = self._cache_assessment(self._selected_source())
        if cache is not None and cache.requires_regeneration:
            self._notify("Regenerate the stale Parquet before activating this dataset.", level="warning")
            return
        if self._active_dataset_id() == dataset_id:
            self._refresh_state()
            return
        try:
            self._clear_selection_for_dataset_switch()
            self.context.datasets.set_active(dataset_id, origin=_LOADER_ID)
        except BaseException as exc:
            self._finish_error(exc)
            return
        self._refresh_state()

    def _on_close_clicked(self, _event: Any) -> None:
        close_template_modal(self.template)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _source_summary_html(self, source: Optional[DatasetSourceInfo]) -> str:
        if source is None:
            return '<div class="al-dataset-loader-section-copy">No supported source is selected.</div>'
        modified = "Unknown"
        if source.modified_ns is not None:
            from datetime import datetime

            modified = datetime.fromtimestamp(source.modified_ns / 1_000_000_000).astimezone().isoformat(timespec="seconds")
        rows = [
            ("File", source.path.name, True),
            ("Format", source_format_label(source.source_format), False),
            ("Size", format_bytes(source.size_bytes), False),
            ("Modified", modified, False),
            ("Server path", str(source.path), True),
        ]
        if self._source_subresource() and source.source_format in {"excel", "hdf5", "sqlite"}:
            rows.append(("Sheet / table", str(self._source_subresource()), True))
        return self._kv_html(rows)

    def _behaviour_html(self, source: Optional[DatasetSourceInfo]) -> str:
        if source is None:
            return '<div class="al-dataset-loader-section-copy">Choose a source to see its import behaviour.</div>'
        if source.source_format == "parquet":
            return """
<div class="al-dataset-loader-section-copy">
  <p style="margin:0;"><strong>Existing Parquet source.</strong> It is registered directly and queried lazily through DatasetManager; no copy is created.</p>
</div>
"""
        streaming = source.source_format in {"fits", "csv", "tsv", "jsonl", "hdf5", "arrow", "sqlite"}
        caveat = ""
        if source.source_format == "excel":
            caveat = " Spreadsheet parsing itself is not streaming, so the selected sheet may temporarily materialise in memory before Parquet is written."
        elif source.source_format == "hdf5":
            caveat = " HDF5 table-format keys stream; fixed-format keys must materialise before conversion."
        return f"""
<div class="al-dataset-loader-section-copy">
  <p style="margin:0 0 8px;"><strong>{escape(source_format_label(source.source_format))} → Parquet.</strong></p>
  <p style="margin:0;">AstronomicAL {'streams/chunks' if streaming else 'reads'} the source, writes an atomic Parquet cache, then registers a lazy DuckDB-backed DatasetSource.{escape(caveat)}</p>
</div>
"""

    def _cache_summary_html(
        self,
        source: Optional[DatasetSourceInfo],
        cache: Optional[CacheAssessment],
    ) -> str:
        if source is None:
            return '<div class="al-dataset-loader-section-copy">Choose a source first.</div>'
        if source.source_format == "parquet":
            return '<div class="al-dataset-loader-section-copy">The selected source is already Parquet; no generated cache is needed.</div>'
        if cache is None:
            return '<div class="al-dataset-loader-section-copy">Enter a Parquet destination to check cache freshness.</div>'
        state_label = {
            "missing": "Not created",
            "fresh": "Fresh",
            "stale": "Stale",
            "unknown": "Needs review",
        }.get(cache.state, cache.state.title())
        fingerprint = (
            "Matches" if cache.fingerprint_matches is True else "Changed" if cache.fingerprint_matches is False else "Not needed / unavailable"
        )
        return self._kv_html(
            (
                ("Cache state", state_label, False),
                ("Reason", cache.message, False),
                ("Content check", fingerprint, False),
                ("Metadata", cache.metadata_path, True),
            )
        )

    def _disk_summary_html(
        self,
        source: Optional[DatasetSourceInfo],
        disk: Optional[DiskSpaceAssessment],
    ) -> str:
        if source is None or source.source_format == "parquet":
            return ""
        if disk is None:
            return '<div class="al-dataset-loader-section-copy">Disk-space estimate unavailable until a Parquet path is selected.</div>'
        state = "Enough space" if disk.enough_space is True else "Insufficient estimated space" if disk.enough_space is False else "Unknown"
        return self._kv_html(
            (
                ("Disk check", state, False),
                ("Free", format_bytes(disk.free_bytes), False),
                ("Estimated output", format_bytes(disk.estimated_output_bytes), False),
                ("Assessment", disk.message, False),
            )
        )

    def _conversion_progress_html(self, state: Optional[dict[str, Any]]) -> str:
        if not state:
            return ""

        phase = str(state.get("phase") or "working")
        label = str(state.get("label") or "Converting dataset")
        completed = _optional_int(state.get("rows_completed"))
        total = _optional_int(state.get("rows_total"))
        chunk_index = _optional_int(state.get("chunk_index"))
        chunk_count = _optional_int(state.get("chunk_count"))
        elapsed = _optional_float(state.get("elapsed_seconds"))
        eta = _optional_float(state.get("eta_seconds"))
        current_rate = _optional_float(state.get("rows_per_second"))
        average_rate = _optional_float(state.get("average_rows_per_second"))
        tmp_size = _optional_int(state.get("tmp_size_bytes"))
        rss = _optional_float(state.get("rss_gib"))

        percent: Optional[float] = None
        if total is not None and total > 0 and completed is not None:
            percent = max(0.0, min(100.0, 100.0 * completed / total))
        elif phase == "complete":
            percent = 100.0

        rows_text = (
            f"{completed:,} / {total:,}"
            if completed is not None and total is not None
            else f"{completed:,} / —" if completed is not None
            else "— / —"
        )
        chunk_text = (
            f"{chunk_index:,} / {chunk_count:,}"
            if chunk_index is not None and chunk_count is not None
            else f"{chunk_index:,}" if chunk_index is not None
            else "—"
        )
        rate_value = current_rate if current_rate is not None else average_rate
        rate_text = f"{rate_value:,.0f} rows/s" if rate_value is not None else "Calculating…"
        average_text = f"{average_rate:,.0f} rows/s" if average_rate is not None else "Calculating…"
        eta_text = (
            "Complete"
            if phase == "complete"
            else _format_duration(eta) if eta is not None
            else "Calculating…" if total is not None and completed not in (None, 0)
            else "—"
        )
        elapsed_text = _format_duration(elapsed) if elapsed is not None else "—"
        percent_text = f"{percent:.1f}%" if percent is not None else "—"
        fill_style = f' style="width:{percent:.3f}%;"' if percent is not None else ""
        fill_class = "al-dataset-loader-progress-fill" + (" is-indeterminate" if percent is None and phase not in {"complete", "error"} else "")
        state_class = "error" if phase == "error" else "complete" if phase == "complete" else "active"

        return f"""
<div class="al-dataset-loader-conversion-summary {state_class}">
  <div class="al-dataset-loader-conversion-head">
    <div>
      <div class="al-dataset-loader-conversion-eyebrow">Total conversion progress</div>
      <div class="al-dataset-loader-conversion-label">{escape(label)}</div>
    </div>
    <div class="al-dataset-loader-conversion-percent">{percent_text}</div>
  </div>
  <div class="al-dataset-loader-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100"{f' aria-valuenow="{percent:.1f}"' if percent is not None else ''}>
    <div class="{fill_class}"{fill_style}></div>
  </div>
  <dl class="al-dataset-loader-progress-kv">
    <dt>Rows</dt><dd>{rows_text}</dd>
    <dt>Chunk</dt><dd>{chunk_text}</dd>
    <dt>Elapsed</dt><dd>{elapsed_text}</dd>
    <dt>Est. remaining</dt><dd>{eta_text}</dd>
    <dt>Current rate</dt><dd>{rate_text}</dd>
    <dt>Average rate</dt><dd>{average_text}</dd>
    <dt>Temporary Parquet</dt><dd>{format_bytes(tmp_size)}</dd>
    <dt>RSS</dt><dd>{f'{rss:.2f} GiB' if rss is not None else '—'}</dd>
  </dl>
</div>
"""

    def _result_html(self, result: DatasetImportResult) -> str:
        if result.regenerated:
            cache_state = "Regenerated"
        elif result.cache_created is True:
            cache_state = "Created"
        elif result.cache_path:
            cache_state = "Reused"
        else:
            cache_state = "Not applicable"
        return f"""
<div style="margin-top:2px;">
  <div class="al-dataset-loader-section-title" style="margin-bottom:9px;">Registration result</div>
  {self._kv_html((
      ("Dataset", result.dataset_name, False),
      ("Dataset ID", result.dataset_id, True),
      ("Dimensions", format_dimensions(result.rows, len(result.columns)), False),
      ("Backend", result.backend or "Registered", True),
      ("Cache", cache_state, False),
      ("Cache path", result.cache_path or "Not applicable", True),
  ))}
</div>
"""

    @staticmethod
    def _section_heading(title: str, copy: str) -> pn.pane.HTML:
        return pn.pane.HTML(
            f'<div class="al-dataset-loader-section-title">{escape(title)}</div>'
            f'<div class="al-dataset-loader-section-copy">{escape(copy)}</div>',
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

    @staticmethod
    def _field_label(text: str) -> pn.pane.HTML:
        return pn.pane.HTML(
            f'<div class="al-dataset-loader-field-label">{escape(text)}</div>',
            sizing_mode="stretch_width",
            height=20,
            margin=(8, 0, 0, 0),
        )

    @staticmethod
    def _status_html(title: str, copy: str, state: str) -> str:
        safe_state = state if state in {"idle", "busy", "success", "error", "warning"} else "idle"
        return f"""
<div class="al-dataset-loader-status {safe_state}">
  <span class="al-dataset-loader-status-dot" aria-hidden="true"></span>
  <div style="min-width:0;">
    <div class="al-dataset-loader-status-title">{escape(str(title))}</div>
    <div class="al-dataset-loader-status-copy">{escape(str(copy))}</div>
  </div>
</div>
"""

    @staticmethod
    def _kv_html(rows: Iterable[tuple[str, str, bool]]) -> str:
        items: list[str] = []
        for label, value, code in rows:
            cls = ' class="al-dataset-loader-code"' if code else ""
            items.append(f"<dt>{escape(str(label))}</dt><dd{cls}>{escape(str(value))}</dd>")
        return '<dl class="al-dataset-loader-kv">' + "".join(items) + "</dl>"

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _selected_source(self) -> Optional[DatasetSourceInfo]:
        return self._sources.get(str(self.source_select.value or ""))

    def _source_subresource(self) -> Optional[str]:
        source = self._selected_source()
        if source is None:
            return None
        if source.source_format == "fits":
            return "1"
        value = self.subresource_select.value if self.subresource_select.visible else None
        return str(value) if value not in (None, "") else None

    def _registered_dataset_for_source(self, source_path: Path) -> Optional[str]:
        target = canonical_path(source_path)
        for dataset_id in self.context.datasets.list_ids():
            metadata = _safe_meta(self.context.datasets, str(dataset_id))
            candidate = metadata.get("source_path")
            if not candidate:
                try:
                    dataset = self.context.datasets.get(dataset_id)
                    candidate = getattr(dataset, "source_path", None)
                except Exception:
                    candidate = None
            if candidate and canonical_path(candidate) == target:
                return str(dataset_id)
        return None

    def _next_dataset_id(self, base: str) -> str:
        existing = set(self.context.datasets.list_ids())
        if base not in existing:
            return base
        index = 2
        while f"{base}_{index}" in existing:
            index += 1
        return f"{base}_{index}"

    def _dataset_name(self, dataset_id: Optional[str]) -> str:
        if not dataset_id:
            return ""
        try:
            dataset = self.context.datasets.get(dataset_id)
            return str(getattr(dataset, "name", None) or dataset_id)
        except Exception:
            return str(dataset_id)

    def _active_dataset_id(self) -> Optional[str]:
        try:
            dataset_id = self.context.datasets.active_id()
        except Exception:
            return None
        return str(dataset_id) if dataset_id not in (None, "") else None

    def _existing_dataset_is_active(self) -> bool:
        return self._existing_dataset_id is not None and self._active_dataset_id() == self._existing_dataset_id

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

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        events = getattr(self.context, "events", None)
        if events is not None:
            events.publish(topic, payload)

    def _notify(self, message: str, *, level: str = "info") -> None:
        notifications = getattr(pn.state, "notifications", None)
        if notifications is None:
            return
        method = getattr(notifications, level, None)
        if callable(method):
            try:
                method(str(message))
            except Exception:
                pass

    @staticmethod
    def _job_key(request: DatasetImportRequest) -> str:
        identity = (
            f"{canonical_path(request.source_path)}|{request.dataset_id}|{request.source_format}|"
            f"sub={request.source_subresource}|parquet={request.parquet_path}|regen={request.regenerate}"
        )
        digest = sha256(identity.encode("utf-8")).hexdigest()[:20]
        return f"dataset.import:{digest}"



# ----------------------------------------------------------------------
# Module helpers / compatibility exports
# ----------------------------------------------------------------------


def discover_dataset_sources(data_directory: str | Path) -> list[DatasetSourceInfo]:
    root = Path(data_directory).expanduser()
    if not root.is_dir():
        return []
    sources: list[DatasetSourceInfo] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        info = dataset_source_info(path)
        if info is not None:
            sources.append(info)
    return sources


def dataset_source_info(path: str | Path) -> Optional[DatasetSourceInfo]:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return None
    source_format = detect_source_format(candidate)
    if source_format is None:
        return None
    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate.absolute()
    try:
        stat = resolved.stat()
        size_bytes = int(stat.st_size)
        modified_ns = int(stat.st_mtime_ns)
    except OSError:
        size_bytes = None
        modified_ns = None
    return DatasetSourceInfo(
        path=resolved,
        source_format=source_format,
        size_bytes=size_bytes,
        modified_ns=modified_ns,
    )


def normalise_dataset_id(value: Any) -> str:
    text = str(value or "dataset").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "dataset"


def canonical_path(value: str | Path) -> str:
    try:
        return str(Path(value).expanduser().resolve())
    except Exception:
        return str(Path(value).expanduser().absolute())


def format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "Unknown"
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{value:,} B"


def _format_duration(value: Optional[float]) -> str:
    if value is None:
        return "—"
    seconds = max(0, int(round(float(value))))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minute = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minute:02d}m"
    days, hour = divmod(hours, 24)
    return f"{days}d {hour:02d}h"


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def format_dimensions(rows: Optional[int], columns: Optional[int]) -> str:
    if rows is not None and columns is not None:
        return f"{rows:,} rows · {columns:,} columns"
    if rows is not None:
        return f"{rows:,} rows"
    if columns is not None:
        return f"{columns:,} columns"
    return "Available after registration"


def _safe_row_count(datasets: Any, dataset_id: str) -> Optional[int]:
    try:
        value = datasets.row_count(dataset_id)
        return int(value) if value is not None else None
    except Exception:
        return None


def _safe_columns(datasets: Any, dataset_id: str) -> list[str]:
    try:
        return [str(column) for column in datasets.list_columns(dataset_id)]
    except Exception:
        return []


def _safe_meta(datasets: Any, dataset_id: str) -> dict[str, Any]:
    try:
        value = datasets.get_meta(dataset_id)
        return dict(value or {})
    except Exception:
        return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _metadata_columns(meta: dict[str, Any]) -> Optional[list[str]]:
    raw = meta.get("columns")
    if isinstance(raw, dict):
        return [str(value) for value in raw.keys()]
    if isinstance(raw, (list, tuple)):
        return [str(value) for value in raw]
    return None


def _metadata_row_count(meta: dict[str, Any]) -> Optional[int]:
    for key in ("row_count", "rows", "n_rows"):
        value = meta.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            pass
    return None


def _optional_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _extract_job_result(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "result" in kwargs:
        return kwargs["result"]
    for value in reversed(args):
        if isinstance(value, DatasetImportResult):
            return value
        if isinstance(value, dict) and "dataset_id" in value:
            return value
    for value in reversed(args):
        result_method = getattr(value, "result", None)
        if callable(result_method):
            return result_method()
    if args:
        return args[-1]
    raise RuntimeError("Dataset import completed without a result.")


def _extract_job_error(args: tuple[Any, ...], kwargs: dict[str, Any]) -> BaseException:
    for key in ("error", "exception", "exc"):
        value = kwargs.get(key)
        if isinstance(value, BaseException):
            return value
    for value in reversed(args):
        if isinstance(value, BaseException):
            return value
    if args:
        return RuntimeError(str(args[-1]))
    return RuntimeError("Dataset import failed without an error message.")


__all__ = [
    "DatasetImportRequest",
    "DatasetImportResult",
    "DatasetImportService",
    "DatasetLoaderController",
    "DatasetSourceInfo",
    "canonical_path",
    "dataset_source_info",
    "detect_source_format",
    "discover_dataset_sources",
    "format_bytes",
    "format_dimensions",
    "normalise_dataset_id",
    "source_format_label",
    "source_stem",
]

