from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any, Iterable, Optional
import re

import panel as pn

from astronomicAL.platform.dataset_loader_styles import (
    DATASET_LOADER_CSS,
    DATASET_LOADER_INPUT_STYLESHEET,
)
from astronomicAL.platform.fits_import import register_fits_table
from astronomicAL.platform.modal_utils import close_template_modal


_LOADER_ID = "platform.dataset_loader"


@dataclass(frozen=True)
class DatasetSourceInfo:
    path: Path
    source_format: str
    size_bytes: Optional[int]

    @property
    def label(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class DatasetImportRequest:
    source_path: str
    source_format: str
    dataset_id: str
    dataset_name: str
    fits_hdu: int = 1


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

    def event_payload(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.dataset_name,
            "rows": self.rows,
            "columns": list(self.columns),
            "source_path": self.source_path,
            "loader_id": _LOADER_ID,
            "optimise_data": False,
            "backend": self.backend,
        }


class DatasetImportService:
    """Register supported server-visible datasets through DatasetManager."""

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
    ) -> DatasetImportResult:
        self._raise_if_cancelled(cancel_token)

        source = Path(request.source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Dataset source does not exist: {source}")

        detected_format = detect_source_format(source)
        if detected_format is None:
            raise ValueError(
                "Unsupported dataset format. Supported files are FITS and Parquet."
            )
        if detected_format != request.source_format:
            raise ValueError(
                f"Dataset format changed before import: expected "
                f"{request.source_format}, detected {detected_format}."
            )

        dataset_id = normalise_dataset_id(request.dataset_id)
        if dataset_id in set(self.context.datasets.list_ids()):
            raise ValueError(f"Dataset id is already registered: {dataset_id}")

        dataset_name = str(request.dataset_name or source.stem).strip() or dataset_id
        cache_path: Optional[str] = None
        cache_created: Optional[bool] = None

        self._raise_if_cancelled(cancel_token)

        if detected_format == "fits":
            cache_dir = source.parent / ".astronomical_cache"
            raw_result = register_fits_table(
                self.context.datasets,
                str(source),
                cache_dir=cache_dir,
                hdu=int(request.fits_hdu),
                dataset_id=dataset_id,
                name=dataset_name,
                overwrite=False,
            )
            result = dict(raw_result or {})
            cache_path = _optional_text(result.get("parquet_path"))
            cache_created = _optional_bool(result.get("created"))
        else:
            self.context.datasets.register_parquet(
                dataset_id,
                str(source),
                name=dataset_name,
                source_path=str(source),
                loader_id=_LOADER_ID,
                optimise_data=False,
            )
            cache_path = None
            cache_created = False

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
        )

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

        for attribute_name in (
            "cancelled",
            "is_cancelled",
            "cancellation_requested",
        ):
            value = getattr(cancel_token, attribute_name, False)
            if callable(value):
                value = value()
            if value:
                raise RuntimeError("Dataset import was cancelled.")


class DatasetLoaderController:
    """Dataset-only modal for discovering and registering server-visible files."""

    def __init__(
        self,
        *,
        context: Any,
        template: Any,
        data_directory: str | Path = "data",
    ) -> None:
        if context is None:
            raise ValueError("DatasetLoaderController requires an AppContext.")
        if getattr(context, "datasets", None) is None:
            raise ValueError("DatasetLoaderController requires context.datasets.")

        self.context = context
        self.template = template
        self.data_directory = Path(data_directory).expanduser()
        self.service = DatasetImportService(context)

        self._disposed = False
        self._busy = False
        self._updating_fields = False
        self._watchers: list[Any] = []
        self._job_handle: Any = None
        self._sources: dict[str, DatasetSourceInfo] = {}
        self._existing_dataset_id: Optional[str] = None
        self._last_result: Optional[DatasetImportResult] = None

        self._install_css()
        self._build_widgets()
        self._build_view()
        self._wire_events()
        self.refresh_sources(preserve_selection=False)

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Refresh source and registration state immediately before modal open."""
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
        self.source_select = pn.widgets.Select(
            name="",
            options=OrderedDict({"No supported files found": ""}),
            value="",
            sizing_mode="stretch_width",
            height=36,
            margin=(0, 8, 0, 0),
            stylesheets=[DATASET_LOADER_INPUT_STYLESHEET],
        )
        self.source_select.description = (
            "Choose a FITS or Parquet file visible to the AstronomicAL server"
        )

        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            icon="refresh",
            button_type="default",
            width=100,
            height=36,
            margin=(0, 0, 0, 0),
        )
        self.refresh_button.description = "Rescan the data directory"

        self.dataset_name_input = pn.widgets.TextInput(
            name="",
            placeholder="Dataset display name",
            sizing_mode="stretch_width",
            height=36,
            margin=(0, 0, 0, 0),
            stylesheets=[DATASET_LOADER_INPUT_STYLESHEET],
        )

        self.dataset_id_input = pn.widgets.TextInput(
            name="",
            placeholder="dataset_id",
            sizing_mode="stretch_width",
            height=36,
            margin=(0, 0, 0, 0),
            stylesheets=[DATASET_LOADER_INPUT_STYLESHEET],
        )
        self.dataset_id_input.description = (
            "Stable identifier used by panels, mappings, events, and workspaces"
        )

        self.source_summary = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-loader-source-summary"],
        )
        self.behaviour_summary = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-loader-behaviour"],
        )
        self.validation_alert = pn.pane.Alert(
            "",
            alert_type="warning",
            visible=False,
            sizing_mode="stretch_width",
            margin=(10, 0, 0, 0),
        )
        self.result_pane = pn.pane.HTML(
            "",
            visible=False,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-loader-result"],
        )
        self.status_pane = pn.pane.HTML(
            self._status_html(
                "Ready",
                "Choose a supported server-visible source to review its import behaviour.",
                "idle",
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.activate_existing_button = pn.widgets.Button(
            name="Activate existing dataset",
            icon="database-check",
            button_type="primary",
            width=220,
            height=38,
            visible=False,
            margin=(0, 0, 0, 0),
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
            margin=(0, 0, 0, 0),
        )

    def _build_view(self) -> None:
        heading = pn.pane.HTML(
            """
            <div class="al-modal-titlebar">
              <div class="al-modal-heading">Add dataset</div>
              <div class="al-modal-subtitle">
                Register a FITS or Parquet file that is already visible to the
                AstronomicAL server. Workspace restoration and semantic column
                mappings are handled by their own platform controls.
              </div>
            </div>
            """,
            sizing_mode="stretch_width",
            height=76,
            margin=(0, 0, 0, 0),
        )

        source_section = pn.Column(
            self._section_heading(
                "1. Choose source",
                "The selector scans the server-side data directory; it is not a browser upload.",
            ),
            pn.Row(
                self.source_select,
                self.refresh_button,
                sizing_mode="stretch_width",
                margin=(10, 0, 0, 0),
            ),
            pn.Spacer(height=12),
            self._field_label("Dataset name"),
            self.dataset_name_input,
            pn.Spacer(height=10),
            self._field_label("Dataset ID"),
            self.dataset_id_input,
            self.validation_alert,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
            css_classes=["al-dataset-loader-section"],
        )

        source_details_section = pn.Column(
            self._section_heading(
                "2. Source details",
                "Detected from the selected file before registration.",
            ),
            pn.Spacer(height=10),
            self.source_summary,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-loader-section"],
        )

        behaviour_section = pn.Column(
            self._section_heading(
                "3. Import behaviour",
                "Only behaviour implemented by the active backend is described here.",
            ),
            pn.Spacer(height=10),
            self.behaviour_summary,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
            css_classes=["al-dataset-loader-section"],
        )

        status_section = pn.Column(
            self._section_heading(
                "4. Status",
                "Background imports also appear in the Runtime Status monitor.",
            ),
            pn.Spacer(height=10),
            self.status_pane,
            pn.Spacer(height=10),
            self.result_pane,
            pn.Row(
                self.activate_existing_button,
                sizing_mode="stretch_width",
                margin=(10, 0, 0, 0),
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-loader-section"],
        )

        left = pn.Column(
            source_section,
            source_details_section,
            sizing_mode="fixed",
            width=430,
            margin=(0, 7, 0, 0),
        )
        right = pn.Column(
            behaviour_section,
            status_section,
            sizing_mode="fixed",
            width=470,
            margin=(0, 0, 0, 7),
        )

        body = pn.Row(
            left,
            right,
            sizing_mode="fixed",
            width=960,
            height=540,
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-loader-body", "al-dataset-loader-grid"],
        )

        footer = pn.Row(
            pn.layout.HSpacer(),
            self.close_button,
            self.import_button,
            sizing_mode="fixed",
            width=960,
            height=64,
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-loader-footer", "al-modal-footer"],
        )

        self.view = pn.Column(
            heading,
            body,
            footer,
            sizing_mode="fixed",
            width=960,
            height=680,
            margin=(0, 0, 0, 0),
            css_classes=[
                "al-modal-card",
                "al-dataset-loader-card",
            ],
            styles={"box-sizing": "border-box", "overflow": "hidden"},
        )

    def _wire_events(self) -> None:
        self._watchers.append(
            self.source_select.param.watch(self._on_source_changed, "value")
        )
        self._watchers.append(
            self.dataset_name_input.param.watch(self._on_identity_changed, "value")
        )
        self._watchers.append(
            self.dataset_id_input.param.watch(self._on_identity_changed, "value")
        )
        self.refresh_button.on_click(self._on_refresh_clicked)
        self.import_button.on_click(self._on_import_clicked)
        self.activate_existing_button.on_click(self._on_activate_existing)
        self.close_button.on_click(self._on_close_clicked)

    # ------------------------------------------------------------------
    # Source discovery and validation
    # ------------------------------------------------------------------

    def refresh_sources(self, *, preserve_selection: bool = True) -> None:
        previous = str(self.source_select.value or "") if preserve_selection else ""
        sources = discover_dataset_sources(self.data_directory)
        self._sources = {str(source.path): source for source in sources}

        if sources:
            options: OrderedDict[str, str] = OrderedDict()
            duplicate_names: dict[str, int] = {}
            for source in sources:
                duplicate_names[source.label] = duplicate_names.get(source.label, 0) + 1
            for source in sources:
                label = source.label
                if duplicate_names[label] > 1:
                    label = f"{label} · {source.path.parent}"
                options[label] = str(source.path)

            self._updating_fields = True
            try:
                self.source_select.options = options
                self.source_select.disabled = False
                self.source_select.value = previous if previous in self._sources else str(sources[0].path)
            finally:
                self._updating_fields = False
        else:
            self._updating_fields = True
            try:
                self.source_select.options = OrderedDict(
                    {"No supported FITS or Parquet files found": ""}
                )
                self.source_select.value = ""
                self.source_select.disabled = True
            finally:
                self._updating_fields = False

        self._apply_source_defaults(force=not bool(previous))
        self._refresh_state()

    def _on_refresh_clicked(self, _event: Any) -> None:
        self.refresh_sources(preserve_selection=True)

    def _on_source_changed(self, _event: Any) -> None:
        if self._updating_fields:
            return
        self._last_result = None
        self.result_pane.visible = False
        self._apply_source_defaults(force=True)
        self._refresh_state()

    def _on_identity_changed(self, _event: Any) -> None:
        if self._updating_fields:
            return
        self._refresh_state()

    def _apply_source_defaults(self, *, force: bool) -> None:
        source = self._selected_source()
        if source is None:
            if force:
                self._updating_fields = True
                try:
                    self.dataset_name_input.value = ""
                    self.dataset_id_input.value = ""
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

    def _refresh_state(self) -> None:
        source = self._selected_source()
        self._existing_dataset_id = (
            self._registered_dataset_for_source(source.path) if source is not None else None
        )

        self.source_summary.object = self._source_summary_html(source)
        self.behaviour_summary.object = self._behaviour_html(source)

        duplicate = self._existing_dataset_id is not None
        problem = None if duplicate else self._validation_problem(source)
        self.validation_alert.visible = problem is not None
        self.validation_alert.object = problem or ""

        existing_is_active = self._existing_dataset_is_active()
        self.activate_existing_button.visible = duplicate and not self._busy
        self.activate_existing_button.disabled = self._busy or existing_is_active
        if duplicate:
            existing_name = self._dataset_name(self._existing_dataset_id)
            self.activate_existing_button.name = (
                f"{existing_name} is active"
                if existing_is_active
                else f"Activate {existing_name}"
            )

        self.import_button.disabled = self._busy or problem is not None or duplicate
        self.refresh_button.disabled = self._busy
        self.source_select.disabled = self._busy or not bool(self._sources)
        self.dataset_name_input.disabled = self._busy or source is None or duplicate
        self.dataset_id_input.disabled = self._busy or source is None or duplicate

        if self._busy:
            return

        if duplicate and existing_is_active:
            self.status_pane.object = self._status_html(
                "Dataset already active",
                (
                    f"This file is registered as "
                    f"{self._dataset_name(self._existing_dataset_id)} "
                    f"({self._existing_dataset_id}) and is already the active dataset."
                ),
                "success",
            )
        elif duplicate:
            self.status_pane.object = self._status_html(
                "Source already registered",
                (
                    f"This file is already registered as "
                    f"{self._dataset_name(self._existing_dataset_id)} "
                    f"({self._existing_dataset_id}). Activate that dataset instead "
                    "of creating another cache or registration."
                ),
                "warning",
            )
        elif problem:
            self.status_pane.object = self._status_html(
                "Import not ready",
                problem,
                "warning",
            )
        elif source is not None:
            self.status_pane.object = self._status_html(
                "Ready to import",
                "Review the detected behaviour, then start the background import.",
                "idle",
            )
        else:
            self.status_pane.object = self._status_html(
                "No supported source",
                (
                    f"Add a FITS or Parquet file to "
                    f"{self.data_directory.resolve()} and refresh the list."
                ),
                "warning",
            )

    def _validation_problem(self, source: Optional[DatasetSourceInfo]) -> Optional[str]:
        if source is None:
            return "Choose a supported dataset source."
        if not source.path.is_file():
            return "The selected source is no longer available. Refresh the file list."

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
        return None

    def _selected_source(self) -> Optional[DatasetSourceInfo]:
        return self._sources.get(str(self.source_select.value or ""))

    # ------------------------------------------------------------------
    # Import execution
    # ------------------------------------------------------------------

    def _on_import_clicked(self, _event: Any) -> None:
        if self._busy:
            return

        source = self._selected_source()
        problem = self._validation_problem(source)
        if problem is not None or source is None:
            self._refresh_state()
            return

        request = DatasetImportRequest(
            source_path=str(source.path),
            source_format=source.source_format,
            dataset_id=self.dataset_id_input.value.strip(),
            dataset_name=self.dataset_name_input.value.strip(),
            fits_hdu=1,
        )

        self._set_busy(True)
        jobs = getattr(self.context, "jobs", None)
        if jobs is not None and hasattr(jobs, "submit"):
            status_copy = (
                "The import has been submitted to the platform job runner. "
                "Runtime Status shows whether it is queued or running; this dialog "
                "may be closed while the job continues."
            )
        else:
            status_copy = (
                "The platform job runner is unavailable, so this import is running "
                "in the current session."
            )
        self.status_pane.object = self._status_html(
            "Import in progress",
            status_copy,
            "busy",
        )
        self.result_pane.visible = False

        if jobs is not None and hasattr(jobs, "submit"):
            try:
                self._job_handle = jobs.submit(
                    self._run_import_job,
                    title=f"Import dataset: {request.dataset_name}",
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
        )

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

        try:
            self._clear_selection_for_dataset_switch()
            self._publish("dataset.loaded", result.event_payload())
            self.context.datasets.set_active(
                result.dataset_id,
                origin=_LOADER_ID,
            )
        except BaseException as exc:
            self._set_busy(False)
            self._finish_error(exc)
            return

        self._set_busy(False)
        self.refresh_sources(preserve_selection=True)
        self.status_pane.object = self._status_html(
            "Dataset ready",
            (
                f"{result.dataset_name} is registered and active. Panels request "
                "semantic column mappings only when they need them."
            ),
            "success",
        )
        self.result_pane.object = self._result_html(result)
        self.result_pane.visible = True

    def _finish_error(self, exc: BaseException) -> None:
        self._job_handle = None
        self._set_busy(False)
        message = str(exc).strip() or type(exc).__name__
        self.status_pane.object = self._status_html(
            "Import failed",
            message,
            "error",
        )
        self.result_pane.visible = False
        self._refresh_state_controls_only()

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.import_button.name = "Importing dataset…" if busy else "Import dataset"
        self.close_button.name = "Close while import continues" if busy else "Close"
        self._refresh_state_controls_only()

    def _refresh_state_controls_only(self) -> None:
        source = self._selected_source()
        duplicate = self._existing_dataset_id is not None
        problem = self._validation_problem(source) if not self._busy else None

        self.import_button.disabled = self._busy or problem is not None or duplicate
        self.refresh_button.disabled = self._busy
        self.source_select.disabled = self._busy or not bool(self._sources)
        self.dataset_name_input.disabled = self._busy or source is None or duplicate
        self.dataset_id_input.disabled = self._busy or source is None or duplicate
        existing_is_active = self._existing_dataset_is_active()
        self.activate_existing_button.visible = duplicate and not self._busy
        self.activate_existing_button.disabled = self._busy or existing_is_active
        if duplicate:
            existing_name = self._dataset_name(self._existing_dataset_id)
            self.activate_existing_button.name = (
                f"{existing_name} is active"
                if existing_is_active
                else f"Activate {existing_name}"
            )

    def _on_activate_existing(self, _event: Any) -> None:
        dataset_id = self._existing_dataset_id
        if not dataset_id:
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
    # Rendering helpers
    # ------------------------------------------------------------------

    def _source_summary_html(self, source: Optional[DatasetSourceInfo]) -> str:
        if source is None:
            return (
                '<div class="al-dataset-loader-section-copy">'
                "No supported source is selected."
                "</div>"
            )

        size = format_bytes(source.size_bytes)
        return self._kv_html(
            (
                ("File", source.path.name, True),
                ("Format", source_format_label(source.source_format), False),
                ("Size", size, False),
                ("Server path", str(source.path), True),
            )
        )

    def _behaviour_html(self, source: Optional[DatasetSourceInfo]) -> str:
        if source is None:
            return (
                '<div class="al-dataset-loader-section-copy">'
                "Choose a source to see its exact registration behaviour."
                "</div>"
            )

        if source.source_format == "fits":
            dataset_id = normalise_dataset_id(self.dataset_id_input.value or source_stem(source.path))
            cache_path = source.path.parent / ".astronomical_cache" / f"{dataset_id}.parquet"
            return f"""
<div class="al-dataset-loader-section-copy">
  <p style="margin:0 0 9px;"><strong>FITS binary table, HDU 1.</strong></p>
  <p style="margin:0 0 9px;">
    Rows are streamed into a local Parquet cache without materialising the full
    table as a shared DataFrame. Numeric precision is preserved; the removed
    memory-downcast option is not applied.
  </p>
  <p style="margin:0 0 9px;">
    The resulting Parquet source is registered with DatasetManager and queried
    lazily by panels.
  </p>
  <dl class="al-dataset-loader-kv">
    <dt>Cache path</dt>
    <dd class="al-dataset-loader-code">{escape(str(cache_path))}</dd>
  </dl>
</div>
"""

        return """
<div class="al-dataset-loader-section-copy">
  <p style="margin:0 0 9px;"><strong>Existing Parquet source.</strong></p>
  <p style="margin:0 0 9px;">
    The file is registered directly and queried lazily through the dataset
    backend. It is not copied, rewritten, or downcast by this loader.
  </p>
  <p style="margin:0;">
    Keep the source file at the registered server path so the dataset remains
    available to future queries and restored workspaces.
  </p>
</div>
"""

    def _result_html(self, result: DatasetImportResult) -> str:
        dimensions = format_dimensions(result.rows, len(result.columns))
        cache_value = result.cache_path or "Not applicable"
        if result.cache_created is True:
            cache_state = "Created"
        elif result.cache_created is False and result.cache_path:
            cache_state = "Reused"
        else:
            cache_state = "Not applicable"

        return f"""
<div style="margin-top:2px;">
  <div class="al-dataset-loader-section-title" style="margin-bottom:9px;">Registration result</div>
  {self._kv_html((
      ("Dataset", result.dataset_name, False),
      ("Dataset ID", result.dataset_id, True),
      ("Dimensions", dimensions, False),
      ("Backend", result.backend or "Registered", True),
      ("Cache", cache_state, False),
      ("Cache path", cache_value, True),
  ))}
</div>
"""

    @staticmethod
    def _section_heading(title: str, copy: str) -> pn.pane.HTML:
        return pn.pane.HTML(
            f"""
<div class="al-dataset-loader-section-title">{escape(title)}</div>
<div class="al-dataset-loader-section-copy">{escape(copy)}</div>
""",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

    @staticmethod
    def _field_label(text: str) -> pn.pane.HTML:
        return pn.pane.HTML(
            f'<div class="al-dataset-loader-field-label">{escape(text)}</div>',
            sizing_mode="stretch_width",
            height=20,
            margin=(0, 0, 0, 0),
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
        items = []
        for label, value, code in rows:
            cls = ' class="al-dataset-loader-code"' if code else ""
            items.append(
                f"<dt>{escape(str(label))}</dt>"
                f"<dd{cls}>{escape(str(value))}</dd>"
            )
        return '<dl class="al-dataset-loader-kv">' + "".join(items) + "</dl>"

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

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

    def _dataset_name(self, dataset_id: str) -> str:
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
        if dataset_id in (None, ""):
            return None
        return str(dataset_id)

    def _existing_dataset_is_active(self) -> bool:
        return (
            self._existing_dataset_id is not None
            and self._active_dataset_id() == self._existing_dataset_id
        )

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

    @staticmethod
    def _job_key(request: DatasetImportRequest) -> str:
        identity = (
            f"{canonical_path(request.source_path)}|{request.dataset_id}|"
            f"{request.source_format}|hdu={request.fits_hdu}"
        )
        digest = sha256(identity.encode("utf-8")).hexdigest()[:20]
        return f"dataset.import:{digest}"


def discover_dataset_sources(data_directory: str | Path) -> list[DatasetSourceInfo]:
    root = Path(data_directory).expanduser()
    if not root.is_dir():
        return []

    sources: list[DatasetSourceInfo] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            continue
        source_format = detect_source_format(path)
        if source_format is None:
            continue
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path.absolute()
        try:
            size_bytes = resolved.stat().st_size
        except OSError:
            size_bytes = None
        sources.append(
            DatasetSourceInfo(
                path=resolved,
                source_format=source_format,
                size_bytes=size_bytes,
            )
        )
    return sources


def detect_source_format(path: str | Path) -> Optional[str]:
    name = str(path).lower()
    if name.endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        return "fits"
    if name.endswith((".parquet", ".pq")):
        return "parquet"
    return None


def source_stem(path: str | Path) -> str:
    name = Path(path).name
    lower = name.lower()
    for suffix in (".fits.gz", ".fit.gz", ".parquet", ".fits", ".fit", ".pq"):
        if lower.endswith(suffix):
            return name[: -len(suffix)] or "dataset"
    return Path(name).stem or "dataset"


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


def source_format_label(source_format: str) -> str:
    return "FITS binary table" if source_format == "fits" else "Parquet"


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


def _optional_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


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
    "detect_source_format",
    "discover_dataset_sources",
    "format_bytes",
    "format_dimensions",
    "normalise_dataset_id",
    "source_stem",
]
