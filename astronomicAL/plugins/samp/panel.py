from __future__ import annotations

import datetime as _dt
import re
import time
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import panel as pn

from astronomicAL.platform.parquet_cache import default_cache_dir_for_context, normalise_dataset_id

from .service import SAMPBridge

PLUGIN_ID = "astro.samp"
SERVICE_KEY = f"{PLUGIN_ID}.bridge"

_SECTION_STYLE = {
    "padding": "12px 14px",
    "border": "1px solid #d9d9d9",
    "border-radius": "8px",
    "background": "#fafafa",
    "box-sizing": "border-box",
    "min-width": "0",
}

_STATUS_STYLE = {
    "padding": "10px 12px",
    "border": "1px solid #d9d9d9",
    "border-radius": "8px",
    "background": "#ffffff",
    "box-sizing": "border-box",
    "min-width": "0",
}


def create_samp_receive_panel(context, data=None, **kwargs):
    controller = SampReceivePanel(context=context)
    return controller.view, controller


def create_samp_send_panel(context, **kwargs):
    controller = SampSendPanel(context=context)
    return controller.view, controller


def _fit_block(*objects):
    return pn.FlexBox(
        *objects,
        flex_direction="column",
        flex_wrap="nowrap",
        justify_content="flex-start",
        align_items="stretch",
        sizing_mode="stretch_width",
        styles={"min-width": "0", "gap": "0px"},
    )


def _section(title: str, *objects):
    return pn.FlexBox(
        pn.pane.HTML(f"<div style='font-weight:600; margin-bottom:10px;'>{title}</div>", sizing_mode="stretch_width"),
        _fit_block(*objects),
        flex_direction="column",
        flex_wrap="nowrap",
        justify_content="flex-start",
        align_items="stretch",
        sizing_mode="stretch_width",
        styles=dict(_SECTION_STYLE),
        margin=(0, 0, 12, 0),
    )


def _status_block(status_pane):
    return pn.FlexBox(
        pn.pane.HTML("<div style='font-weight:600; margin-bottom:10px;'>Status</div>", sizing_mode="stretch_width"),
        _fit_block(status_pane),
        flex_direction="column",
        flex_wrap="nowrap",
        justify_content="flex-start",
        align_items="stretch",
        sizing_mode="stretch_width",
        styles=dict(_STATUS_STYLE),
        margin=(0, 0, 12, 0),
    )


def _root_column(*objects):
    return pn.FlexBox(
        *objects,
        flex_direction="column",
        flex_wrap="nowrap",
        justify_content="flex-start",
        align_items="stretch",
        sizing_mode="stretch_both",
        styles={
            "overflow-x": "hidden",
            "overflow-y": "auto",
            "min-width": "0",
            "min-height": "0",
            "gap": "0px",
        },
    )


def _text_input(name: str, value: str = ""):
    return pn.widgets.TextInput(name=name, value=value, sizing_mode="stretch_width", margin=(0, 0, 10, 0))


def _select(name: str, options=None, value=None, size=None):
    kwargs = dict(name=name, options=options or {}, value=value, sizing_mode="stretch_width", margin=(0, 0, 10, 0))
    if size is not None:
        kwargs["size"] = size
    return pn.widgets.Select(**kwargs)


def _int_input(name: str, value: int, start: int = 0, step: int = 1):
    return pn.widgets.IntInput(name=name, value=value, start=start, step=step, sizing_mode="stretch_width", margin=(0, 0, 10, 0))


def _multiselect(name: str, options=None, value=None, size: int = 10):
    return pn.widgets.MultiSelect(name=name, options=options or [], value=value or [], size=size, sizing_mode="stretch_width", margin=(0, 0, 10, 0))


def _multichoice(name: str, options=None, value=None, height: int = 120):
    return pn.widgets.MultiChoice(name=name, options=options or [], value=value or [], sizing_mode="stretch_width", height=height, margin=(0, 0, 10, 0))


def _safe_unwatch(widget, watcher) -> None:
    try:
        widget.param.unwatch(watcher)
    except Exception:
        pass


def _safe_count(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _normalise_dataset_id(value: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-").lower()
    return base or "samp-import"


def _samp_debug(label: str, **values: Any) -> None:
    try:
        parts = " ".join(f"{key}={value!r}" for key, value in values.items())
        print(f"[AstronomicAL SAMP][{label}] thread={threading.current_thread().name} {parts}", flush=True)
    except Exception:
        print(f"[AstronomicAL SAMP][{label}]", flush=True)


class _SampPanelBase:
    panel_name = "SAMP"

    def __init_base__(self, context, *, panel_name: str):
        self.context = context
        self.panel_id = f"{PLUGIN_ID}.{panel_name}.{uuid.uuid4().hex[:8]}"
        self.panel_name = panel_name
        self.datasets = getattr(context, "datasets", None)
        self.services = getattr(context, "services", None)
        self.jobs = getattr(context, "jobs", None)
        self.events = getattr(context, "events", None)
        self.artifacts = getattr(context, "artifacts", None)
        self._subscriptions: list[Any] = []
        self._watchers: list[tuple[Any, Any]] = []
        self._job_handles: list[Any] = []
        self._periodic_callbacks: list[Any] = []
        self._ui_doc = self._find_bokeh_doc_from_state()
        self._ui_queue = deque()
        self._ui_drain_handle = None
        self._disposed = False

    def _watch(self, widget, callback, attr: str) -> None:
        try:
            watcher = widget.param.watch(callback, attr)
            self._watchers.append((widget, watcher))
        except Exception:
            pass

    def _subscribe_dataset_events(self) -> None:
        if self.events is None:
            return
        for topic in ("dataset.loaded", "dataset.active.changed", "dataset.updated", "dataset.mapping.updated"):
            try:
                sub = self.events.subscribe(
                    topic,
                    self._on_dataset_event,
                    owner_id=self.panel_id,
                    owner_label=self.panel_name,
                    owner_kind="plugin-panel",
                )
            except TypeError:
                sub = self.events.subscribe(topic, self._on_dataset_event)
            self._subscriptions.append(sub)

    def _on_dataset_event(self, _topic, _payload) -> None:
        pass

    def _publish(self, topic: str, payload: dict[str, Any] | None = None) -> None:
        payload = dict(payload or {})
        if topic.startswith("dataset.") or topic.startswith("mapping.") or topic.startswith("interop.samp"):
            if (topic.startswith("dataset.") or topic.startswith("mapping.")) and threading.current_thread() is not threading.main_thread():
                _samp_debug("publish_off_main_thread", topic=topic, payload=payload)
            _samp_debug("publish", topic=topic, payload=payload)
        if self.events is None:
            return
        try:
            self.events.publish(topic, payload)
        except Exception as exc:
            _samp_debug("publish_failed", topic=topic, error=repr(exc), payload=payload)

    def _put_artifact(
        self,
        artifact_type: str,
        payload: Any,
        *,
        dataset_id: str | None = None,
        params: dict[str, Any] | None = None,
        persist: bool = False,
    ) -> str | None:
        if self.artifacts is None:
            return None
        try:
            return self.artifacts.put(
                artifact_type,
                payload,
                dataset_id=dataset_id or "default",
                params=params or {},
                persist=persist,
            )
        except Exception:
            return None

    def _submit_job(self, fn, *, title: str, key: str | None = None, on_done=None, on_error=None, **kwargs):
        # JobManager callbacks may run from worker threads depending on the host
        # version/debug path. Keep all Panel/widget, DatasetManager registration,
        # and event publication work on the UI thread so mapping gates and open
        # panels observe a consistent active-dataset/mapping state.
        def _done(result):
            _samp_debug("job_done_callback", panel_id=self.panel_id, title=title, key=key, result_type=type(result).__name__)
            if on_done is not None:
                self._run_on_ui_thread(lambda: on_done(result))

        def _error(exc):
            _samp_debug("job_error_callback", panel_id=self.panel_id, title=title, key=key, error=repr(exc))
            if on_error is not None:
                self._run_on_ui_thread(lambda: on_error(exc))
            else:
                raise exc

        if self.jobs is not None and hasattr(self.jobs, "submit"):
            handle = self.jobs.submit(fn, title=title, key=key, on_done=_done, on_error=_error, **kwargs)
            self._job_handles.append(handle)
            return handle
        try:
            result = fn(cancel_token=None, **kwargs)
            _done(result)
        except BaseException as exc:
            _error(exc)
        return None

    def _looks_like_bokeh_doc(self, value: Any) -> bool:
        return value is not None and callable(getattr(value, "add_next_tick_callback", None))

    def _find_bokeh_doc_from_state(self):
        doc = getattr(pn.state, "curdoc", None)
        if self._looks_like_bokeh_doc(doc):
            return doc

        # Panel keeps rendered roots in private state. This is intentionally
        # defensive: different Panel/Bokeh versions store (view, model, doc, ...)
        # tuples slightly differently.
        for attr in ("_views", "_templates"):
            values = getattr(pn.state, attr, None)
            if not values:
                continue
            try:
                iterable = values.values() if hasattr(values, "values") else values
            except Exception:
                iterable = []
            stack = list(iterable)
            seen: set[int] = set()
            while stack:
                obj = stack.pop()
                oid = id(obj)
                if oid in seen:
                    continue
                seen.add(oid)
                if self._looks_like_bokeh_doc(obj):
                    return obj
                obj_doc = getattr(obj, "document", None)
                if self._looks_like_bokeh_doc(obj_doc):
                    return obj_doc
                if isinstance(obj, dict):
                    stack.extend(obj.values())
                elif isinstance(obj, (tuple, list, set)):
                    stack.extend(obj)
        return None

    def _find_bokeh_doc_from_panel(self):
        for obj in (getattr(self, "view", None), *[getattr(self, name, None) for name in (
            "status", "connection_status", "summary", "dataset_select", "inbox_select", "send_button", "register_button"
        )]):
            if obj is None:
                continue
            models = getattr(obj, "_models", None)
            if not models:
                continue
            try:
                values = list(models.values())
            except Exception:
                values = []
            for value in values:
                stack = list(value) if isinstance(value, (tuple, list)) else [value]
                while stack:
                    model = stack.pop()
                    if self._looks_like_bokeh_doc(model):
                        return model
                    doc = getattr(model, "document", None)
                    if self._looks_like_bokeh_doc(doc):
                        return doc
                    if isinstance(model, (tuple, list)):
                        stack.extend(model)
        return None

    def _find_bokeh_doc(self):
        if self._looks_like_bokeh_doc(getattr(self, "_ui_doc", None)):
            return self._ui_doc
        doc = self._find_bokeh_doc_from_panel() or self._find_bokeh_doc_from_state()
        if self._looks_like_bokeh_doc(doc):
            self._ui_doc = doc
            return doc
        return None

    def _call_ui_callback(self, fn) -> None:
        if self._disposed:
            return
        try:
            fn()
        except Exception as exc:
            _samp_debug("ui_callback_failed", panel_id=self.panel_id, error=repr(exc))
            raise

    def _drain_ui_queue(self) -> None:
        if self._disposed:
            return
        while self._ui_queue:
            fn = self._ui_queue.popleft()
            self._call_ui_callback(fn)

    def _ensure_ui_queue_drain(self) -> None:
        if self._ui_drain_handle is not None:
            return
        try:
            self._ui_drain_handle = pn.state.add_periodic_callback(self._drain_ui_queue, period=100, start=True)
            self._periodic_callbacks.append(self._ui_drain_handle)
            _samp_debug("ui_queue_drain_started", panel_id=self.panel_id)
        except Exception as exc:
            _samp_debug("ui_queue_drain_unavailable", panel_id=self.panel_id, error=repr(exc))

    def _run_on_ui_thread(self, fn, *, retries: int = 40, delay_seconds: float = 0.05) -> None:
        """Run ``fn`` on the Bokeh document thread when possible.

        Plugin panel factories and JobManager callbacks may execute in worker
        threads in the current development branch. Publishing dataset/mapping
        events and mutating Panel widgets from those threads can leave mapping
        gates stale even though DatasetManager state is correct. This method
        posts to the rendered Bokeh document once it is available; if the panel
        has not been attached to a document yet, it retries briefly instead of
        falling back to inline worker-thread execution.
        """
        if self._disposed:
            return

        current = threading.current_thread().name
        doc = self._find_bokeh_doc()
        if doc is not None:
            try:
                doc.add_next_tick_callback(lambda: self._call_ui_callback(fn))
                _samp_debug("ui_handoff_posted", panel_id=self.panel_id, from_thread=current)
                return
            except Exception as exc:
                _samp_debug("ui_handoff_post_failed", panel_id=self.panel_id, from_thread=current, error=repr(exc))

        # If we are genuinely on the main thread and there is no Bokeh document,
        # inline execution is safe enough for non-server contexts/tests.
        if threading.current_thread() is threading.main_thread():
            _samp_debug("ui_handoff_inline_main", panel_id=self.panel_id)
            self._call_ui_callback(fn)
            return

        if retries > 0:
            try:
                threading.Timer(delay_seconds, lambda: self._run_on_ui_thread(fn, retries=retries - 1, delay_seconds=delay_seconds)).start()
                _samp_debug("ui_handoff_retry_scheduled", panel_id=self.panel_id, from_thread=current, retries_left=retries)
                return
            except Exception as exc:
                _samp_debug("ui_handoff_retry_failed", panel_id=self.panel_id, from_thread=current, error=repr(exc))

        # Last-resort queue. Do not run inline on a worker thread unless the
        # periodic queue cannot be started at all; inline worker mutation is the
        # behaviour that caused the mapping gate to get stuck.
        self._ui_queue.append(fn)
        self._ensure_ui_queue_drain()

    def _get_samp_service(self) -> SAMPBridge:
        if self.services is None:
            raise RuntimeError("No ServiceRegistry available on context.")
        try:
            bridge = self.services.get(SERVICE_KEY)
        except Exception:
            bridge = None
        if bridge is None:
            bridge = SAMPBridge(client_name="AstronomicAL")
            try:
                self.services.set(SERVICE_KEY, bridge, owner=PLUGIN_ID)
            except TypeError:
                self.services.set(SERVICE_KEY, bridge)
        return bridge

    def _ensure_samp_service(self) -> SAMPBridge:
        bridge = self._get_samp_service()
        bridge.start()
        return bridge

    def _try_start_samp_service(self) -> tuple[SAMPBridge | None, bool, str | None]:
        try:
            bridge = self._get_samp_service()
        except Exception as exc:
            return None, False, str(exc)
        if bool(getattr(bridge, "started", False)):
            try:
                bridge.last_error = None
            except Exception:
                pass
            return bridge, True, None
        ok = bridge.try_start() if hasattr(bridge, "try_start") else False
        return bridge, bool(ok), getattr(bridge, "last_error", None)

    def _add_periodic_callback(self, callback, *, period: int):
        try:
            handle = pn.state.add_periodic_callback(callback, period=period, start=True)
            self._periodic_callbacks.append(handle)
            return handle
        except Exception:
            return None

    def _dataset_name(self, dataset_id: str) -> str:
        if self.datasets is None:
            return dataset_id
        try:
            dataset = self.datasets.get(dataset_id)
            return getattr(dataset, "name", None) or dataset_id
        except Exception:
            return dataset_id

    def _dataset_columns(self, dataset_id: str) -> list[str]:
        if self.datasets is None:
            return []
        try:
            return [str(col) for col in self.datasets.list_columns(dataset_id)]
        except Exception:
            try:
                return [str(col) for col in self.datasets.get_df(dataset_id, limit=0).columns]
            except Exception:
                return []

    def _dataset_row_count(self, dataset_id: str) -> int | None:
        if self.datasets is None:
            return None
        try:
            count = self.datasets.row_count(dataset_id)
            return None if count is None else int(count)
        except Exception:
            return None

    def _dataset_meta(self, dataset_id: str) -> dict[str, Any]:
        if self.datasets is None:
            return {}
        try:
            return dict(self.datasets.get_meta(dataset_id) or {})
        except Exception:
            try:
                dataset = self.datasets.get(dataset_id)
                return dict(getattr(dataset, "meta", {}) or {})
            except Exception:
                return {}

    def _samp_mtype_for_path(self, path: str | Path | None) -> str | None:
        if not path:
            return None
        suffix = Path(str(path)).suffix.lower()
        if suffix in {".fits", ".fit", ".fts"}:
            return "table.load.fits"
        if suffix in {".vot", ".votable", ".xml"}:
            return "table.load.votable"
        return None

    def _direct_samp_file_for_dataset(self, dataset_id: str, *, export_format: str) -> dict[str, Any] | None:
        meta = self._dataset_meta(dataset_id)
        candidates = [
            meta.get("local_table_path"),
            meta.get("local_votable_path"),
            meta.get("original_samp_local_path"),
        ]
        requested = str(export_format or "").strip().lower()
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(str(candidate)).expanduser()
            if not path.exists():
                continue
            mtype = str(meta.get("samp_mtype") or self._samp_mtype_for_path(path) or "")
            if mtype not in {"table.load.fits", "table.load.votable"}:
                continue
            if requested in {"fits"} and mtype != "table.load.fits":
                continue
            if requested in {"votable", "votable_binary", "votable_tabledata"} and mtype != "table.load.votable":
                continue
            return {"path": path, "mtype": mtype, "format": "original_file"}
        return None

    def _materialise_dataset(
        self,
        dataset_id: str,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        if self.datasets is None:
            raise RuntimeError("No DatasetManager available on context.")
        source = self.datasets.get_source(dataset_id) if hasattr(self.datasets, "get_source") else None
        if source is not None and hasattr(source, "to_pandas"):
            return source.to_pandas(columns=columns, limit=limit)
        return self.datasets.get_df(dataset_id, columns=columns, limit=limit)

    def _dispose_base(self) -> None:
        self._disposed = True
        for callback in list(getattr(self, "_periodic_callbacks", []) or []):
            try:
                callback.stop()
            except Exception:
                pass
        self._periodic_callbacks.clear()
        for handle in list(getattr(self, "_job_handles", []) or []):
            try:
                if not handle.future.done():
                    handle.cancel()
            except Exception:
                pass
        for widget, watcher in list(getattr(self, "_watchers", []) or []):
            _safe_unwatch(widget, watcher)
        self._watchers.clear()
        if self.events is not None:
            for sub in list(getattr(self, "_subscriptions", []) or []):
                try:
                    self.events.unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions.clear()


class SampSendPanel(_SampPanelBase):
    def __init__(self, context):
        self.__init_base__(context, panel_name="SAMP Send")

        self.status = pn.pane.Markdown("Ready.")
        self.summary = pn.pane.Markdown("")
        self.dataset_select = _select("Dataset", options={})
        self.table_name = _text_input("Table name", value="AstronomicAL table")
        self.target_mode = _select(
            "Send target",
            options={"TOPCAT": "topcat", "All SAMP clients": "all", "Specific client": "client"},
            value="topcat",
        )
        self.client_select = _select("Specific client", options={})
        self.refresh_clients_button = pn.widgets.Button(name="Refresh clients", button_type="default", sizing_mode="stretch_width", height=40, margin=(0, 0, 10, 0))
        self.all_columns = pn.widgets.Checkbox(name="Send all columns", value=True, margin=(0, 0, 10, 0))
        self.column_help = pn.pane.Markdown(
            "Column choices are source-projected before pandas materialisation. For very wide tables, select only the columns TOPCAT needs.",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        self.column_select = _multiselect("Columns", options=[], value=[], size=10)
        self.row_limit = _int_input("Row limit (0 = all)", value=10000, start=0)
        self.export_scope = _select(
            "Rows to send",
            options={"First N / all rows": "limit", "Current selection set": "selection"},
            value="limit",
        )
        self.export_format = _select(
            "Transfer format",
            options={
                "Original SAMP file if available (fastest)": "original",
                "FITS binary table (fast; TOPCAT/DS9)": "fits",
                "VOTable BINARY (portable; slower)": "votable_binary",
                "VOTable TABLEDATA XML (compatibility; slowest)": "votable_tabledata",
            },
            value="original",
        )
        self.send_button = pn.widgets.Button(name="Send to SAMP", button_type="primary", sizing_mode="stretch_width", height=44, margin=(0, 0, 0, 0))

        self.client_block = _fit_block(self.client_select)
        self.columns_block = _fit_block(self.column_help, self.column_select)

        self._watch(self.dataset_select, self._on_dataset_changed, "value")
        self._watch(self.target_mode, self._on_target_changed, "value")
        self._watch(self.all_columns, self._on_all_columns_changed, "value")
        self.refresh_clients_button.on_click(self._refresh_clients_clicked)
        self.send_button.on_click(self._send_clicked)

        self._subscribe_dataset_events()
        self._refresh_dataset_options()
        self._refresh_client_options()
        self._on_target_changed(None)
        self._on_all_columns_changed(None)
        self.view = self.get_layout()

    def _on_dataset_event(self, _topic, _payload) -> None:
        self._refresh_dataset_options()

    def _dataset_options(self) -> dict[str, str]:
        if self.datasets is None:
            return {}
        options = {}
        try:
            dataset_ids = self.datasets.list_ids()
        except Exception:
            return {}
        for dataset_id in dataset_ids:
            dataset_name = self._dataset_name(dataset_id)
            options[f"{dataset_name} ({dataset_id})"] = dataset_id
        return options

    def _refresh_dataset_options(self) -> None:
        options = self._dataset_options()
        self.dataset_select.options = options
        if not options:
            self.dataset_select.value = None
            self.summary.object = "No datasets available."
            self.column_select.options = []
            self.column_select.value = []
            return

        if self.dataset_select.value not in options.values():
            try:
                active_id = self.datasets.active_id()
                self.dataset_select.value = active_id if active_id in options.values() else next(iter(options.values()))
            except Exception:
                self.dataset_select.value = next(iter(options.values()))
        self._refresh_column_options()
        self._refresh_summary()

    def _refresh_column_options(self) -> None:
        dataset_id = self.dataset_select.value
        if not dataset_id:
            self.column_select.options = []
            self.column_select.value = []
            return
        cols = self._dataset_columns(dataset_id)
        self.column_select.options = cols
        if self.all_columns.value:
            # Keep the hidden selector light for very wide datasets. Export uses
            # selected_columns=None when "Send all columns" is checked.
            self.column_select.value = []
        else:
            self.column_select.value = [c for c in self.column_select.value if c in cols]

    def _refresh_summary(self) -> None:
        dataset_id = self.dataset_select.value
        if not dataset_id:
            self.summary.object = "No dataset selected."
            return
        columns = self._dataset_columns(dataset_id)
        rows = self._dataset_row_count(dataset_id)
        backend = "unknown"
        try:
            backend = getattr(self.datasets.get_source(dataset_id), "backend_name", "unknown")
        except Exception:
            pass
        self.summary.object = (
            f"**Rows:** {_safe_count(rows)}  \n"
            f"**Columns:** {len(columns):,}  \n"
            f"**Backend:** `{backend}`  \n"
            f"**Dataset id:** `{dataset_id}`"
        )
        if not self.table_name.value.strip():
            self.table_name.value = dataset_id

    def _refresh_client_options(self) -> None:
        try:
            bridge, ok, error = self._try_start_samp_service()
            if bridge is None or not ok:
                self.client_select.options = {}
                self.status.object = (
                    "No running SAMP hub detected yet. Open TOPCAT/DS9 or another SAMP-capable "
                    f"application, then refresh. Last error: `{error or 'hub unavailable'}`"
                )
                return
            clients = bridge.list_clients(ensure_started=False)
            options = {f"{c['name']} ({c['id']})": c["id"] for c in clients}
            self.client_select.options = options
            if options and self.client_select.value not in options.values():
                self.client_select.value = next(iter(options.values()))
            self.status.object = f"Connected to SAMP hub. Found {len(clients)} SAMP client(s)."
        except Exception as exc:
            self.client_select.options = {}
            self.status.object = f"Failed to list SAMP clients: `{exc}`"

    def _on_dataset_changed(self, _event) -> None:
        self._refresh_column_options()
        self._refresh_summary()

    def _on_target_changed(self, _event) -> None:
        self.client_block.visible = self.target_mode.value == "client"

    def _on_all_columns_changed(self, _event) -> None:
        self.columns_block.visible = not bool(self.all_columns.value)
        if self.all_columns.value:
            self._refresh_column_options()

    def _refresh_clients_clicked(self, _event) -> None:
        self.status.object = "Refreshing SAMP clients..."
        self._refresh_client_options()

    def _selected_columns(self) -> list[str] | None:
        if self.all_columns.value:
            return None
        selected = [str(col) for col in self.column_select.value]
        if not selected:
            raise ValueError("Pick at least one column to send.")
        return selected

    def _send_clicked(self, _event) -> None:
        dataset_id = self.dataset_select.value
        if not dataset_id:
            self.status.object = "No dataset selected."
            return
        try:
            selected_columns = self._selected_columns()
        except Exception as exc:
            self.status.object = str(exc)
            return

        self.send_button.disabled = True
        self.status.object = "Exporting a source-projected table and sending over SAMP..."
        self._submit_job(
            self._send_job,
            title="Send table over SAMP",
            key=f"{self.panel_id}:send:{uuid.uuid4().hex}",
            on_done=self._on_send_done,
            on_error=self._on_error,
            dataset_id=dataset_id,
            table_name=self.table_name.value.strip() or dataset_id,
            target_mode=self.target_mode.value,
            target_client_id=self.client_select.value,
            selected_columns=selected_columns,
            row_limit=int(self.row_limit.value or 0),
            export_scope=self.export_scope.value,
            export_format=self.export_format.value,
        )

    def _selection_frame(self, dataset_id: str, *, columns: list[str] | None) -> pd.DataFrame:
        selection = getattr(self.context, "selection", None)
        active_set = selection.get_active_set() if selection is not None else None
        if active_set is None or getattr(active_set, "dataset_id", None) != dataset_id:
            raise RuntimeError("No active selection set for the selected dataset.")
        row_ids = list(getattr(active_set, "row_ids", []) or [])
        if not row_ids:
            raise RuntimeError("The active selection set is empty.")

        id_column = None
        try:
            id_column = self.datasets.get_mapping(dataset_id, "record_id")
        except Exception:
            id_column = None
        if id_column and str(id_column).strip().lower() in {"use index", "index", "__index__"}:
            positions = [int(row_id) for row_id in row_ids]
            source = self.datasets.get_source(dataset_id)
            frames = [source.get_row_by_position(pos, columns=columns) for pos in positions]
            return pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if frames else pd.DataFrame()
        if not id_column:
            raise RuntimeError("Current selection export requires the dataset's `record_id` mapping.")
        return self.datasets.get_rows_by_ids(dataset_id, row_ids, id_column=id_column, columns=columns)

    def _send_job(
        self,
        *,
        cancel_token,
        dataset_id: str,
        table_name: str,
        target_mode: str,
        target_client_id: str | None,
        selected_columns: list[str] | None,
        row_limit: int,
        export_scope: str,
        export_format: str,
    ):
        if cancel_token and cancel_token.cancelled():
            return None
        limit = None if row_limit <= 0 else int(row_limit)
        bridge = self._ensure_samp_service()
        if export_scope == "limit" and selected_columns is None and limit is None:
            direct = self._direct_samp_file_for_dataset(dataset_id, export_format=export_format)
            if direct is not None:
                result = bridge.send_table_file(
                    direct["path"],
                    mtype=direct["mtype"],
                    table_name=table_name,
                    target_mode=target_mode,
                    target_client_id=target_client_id,
                    row_count=self._dataset_row_count(dataset_id),
                    columns=self._dataset_columns(dataset_id),
                    export_format=direct["format"],
                )
                result.update({"dataset_id": dataset_id, "backend_materialised_rows": 0, "export_scope": export_scope})
                return result

        t_materialise = time.perf_counter()
        if export_scope == "selection":
            df = self._selection_frame(dataset_id, columns=selected_columns)
            if limit is not None:
                df = df.head(limit)
        else:
            df = self._materialise_dataset(dataset_id, columns=selected_columns, limit=limit)
        materialise_seconds = round(time.perf_counter() - t_materialise, 3)
        if cancel_token and cancel_token.cancelled():
            return None
        result = bridge.send_dataframe(
            df,
            table_name=table_name,
            target_mode=target_mode,
            target_client_id=target_client_id,
            export_format=export_format,
        )
        timings = dict(result.get("timings") or {})
        timings["materialise_seconds"] = materialise_seconds
        timings["job_total_seconds"] = round(materialise_seconds + float(timings.get("total_seconds") or 0), 3)
        result["timings"] = timings
        result.update({"dataset_id": dataset_id, "backend_materialised_rows": int(len(df)), "export_scope": export_scope})
        return result

    def _on_send_done(self, result) -> None:
        self.send_button.disabled = False
        if result is None:
            self.status.object = "Send cancelled."
            return
        artifact_payload = {
            key: result.get(key)
            for key in ("table_name", "path", "url", "mtype", "export_format", "row_count", "column_count", "delivered_to", "coercions", "columns", "export_scope", "timings", "direct_file")
        }
        artifact_id = self._put_artifact(
            "interop.samp.export",
            artifact_payload,
            dataset_id=result["dataset_id"],
            params={"target_mode": self.target_mode.value, "mtype": result.get("mtype", "table.load.votable")},
            persist=True,
        )
        if artifact_id:
            self._publish("artifact.created", {"artifact_id": artifact_id, "type": "interop.samp.export", "dataset_id": result["dataset_id"]})
        self._publish(
            "interop.samp.table.sent",
            {"dataset_id": result["dataset_id"], "artifact_id": artifact_id, "table_name": result["table_name"], "target_mode": self.target_mode.value},
        )
        timings = result.get("timings") or {}
        _samp_debug(
            "send_done",
            dataset_id=result.get("dataset_id"),
            rows=result.get("row_count"),
            columns=result.get("column_count"),
            export_format=result.get("export_format"),
            direct_file=bool(result.get("direct_file")),
            path=result.get("path"),
            timings=timings,
        )
        if timings:
            timing_bits = []
            if "materialise_seconds" in timings:
                timing_bits.append(f"materialise: {timings.get('materialise_seconds')}s")
            if "coerce_seconds" in timings:
                timing_bits.append(f"coerce: {timings.get('coerce_seconds')}s")
            if "write_seconds" in timings:
                timing_bits.append(f"write: {timings.get('write_seconds')}s")
            if "notify_seconds" in timings:
                timing_bits.append(f"notify: {timings.get('notify_seconds')}s")
            if "job_total_seconds" in timings:
                timing_bits.append(f"job: {timings.get('job_total_seconds')}s")
            elif "total_seconds" in timings:
                timing_bits.append(f"service: {timings.get('total_seconds')}s")
            timing_note = " Timings: " + ", ".join(timing_bits) + "."
        else:
            timing_note = ""
        self.status.object = (
            f"Sent **{result['table_name']}** as `{result.get('export_format', 'table')}` "
            f"({result['row_count']:,} rows, {result['column_count']:,} columns).{timing_note}"
        )

    def _on_error(self, exc: BaseException) -> None:
        self.send_button.disabled = False
        self.status.object = f"Failed: `{exc}`"

    def get_layout(self):
        return _root_column(
            _section("Dataset summary", self.summary),
            _section("Table selection", self.dataset_select, self.table_name),
            _section("Target client", self.target_mode, self.client_block, self.refresh_clients_button),
            _section("Export options", self.export_scope, self.export_format, self.all_columns, self.row_limit, self.columns_block),
            _section("Send", self.send_button),
            _status_block(self.status),
        )

    def dispose(self):
        self._dispose_base()


class SampReceivePanel(_SampPanelBase):
    def __init__(self, context):
        self.__init_base__(context, panel_name="SAMP Receive")
        self._listener_token: str | None = None
        self._samp_active_seen = False
        self._receive_count = 0
        self._received_items: list[dict[str, Any]] = []

        self.status = pn.pane.Markdown("SAMP receive panel opened. Waiting for a SAMP hub...")
        self.connection_status = pn.pane.Markdown("")
        self.refresh_connection_button = pn.widgets.Button(
            name="Refresh SAMP connection",
            button_type="default",
            sizing_mode="stretch_width",
            height=36,
            margin=(0, 0, 10, 0),
        )
        self.summary = pn.pane.Markdown("No received table selected.")
        self.inbox_info = pn.pane.Markdown("No received tables yet.")
        self.empty_state = pn.pane.Markdown("No received tables yet. Send a table from TOPCAT or another SAMP client.")
        self.inbox_select = _select("Received tables", options={}, size=8)
        self.preview_rows = _int_input("Preview rows", value=20, start=1, step=5)
        self.preview_columns = _multichoice("Preview columns", options=[], value=[], height=100)
        self.preview_button = pn.widgets.Button(name="Load / refresh preview", button_type="default", sizing_mode="stretch_width", height=40, margin=(0, 0, 10, 0))
        self.dataset_name = _text_input("Dataset name", value="")
        self.dataset_id = _text_input("Dataset id", value="")
        self.make_active = pn.widgets.Checkbox(name="Make active after import", value=True, margin=(0, 0, 10, 0))
        self.register_button = pn.widgets.Button(name="Register as Parquet-backed dataset", button_type="primary", sizing_mode="stretch_width", height=40, margin=(0, 0, 10, 0))
        self.activate_button = pn.widgets.Button(name="Make selected dataset active", button_type="default", sizing_mode="stretch_width", height=40, margin=(0, 0, 10, 0))
        self.discard_button = pn.widgets.Button(name="Discard selected", button_type="warning", sizing_mode="stretch_width", height=40, margin=(0, 0, 10, 0))
        self.clear_button = pn.widgets.Button(name="Clear inbox", button_type="default", sizing_mode="stretch_width", height=40, margin=(0, 0, 0, 0))
        self.preview = pn.pane.DataFrame(pd.DataFrame(), index=False, sizing_mode="stretch_width", height=180, max_height=180)
        self.inbox_body = _fit_block()

        self._watch(self.inbox_select, self._on_inbox_changed, "value")
        self.refresh_connection_button.on_click(lambda _event: self._poll_samp_connection(force_status=True))
        self.preview_button.on_click(self._preview_clicked)
        self.register_button.on_click(self._register_clicked)
        self.activate_button.on_click(self._activate_clicked)
        self.discard_button.on_click(self._discard_clicked)
        self.clear_button.on_click(self._clear_clicked)

        self._subscribe_dataset_events()
        self._attach_listener()
        self._add_periodic_callback(self._poll_samp_connection, period=3000)
        self._poll_samp_connection()
        self._refresh_inbox_body()
        self._update_selection_view()
        self.view = self.get_layout()

    def _attach_listener(self) -> None:
        try:
            bridge = self._get_samp_service()
            if self._listener_token is None:
                self._listener_token = bridge.add_table_listener(self._on_table_received)
        except Exception as exc:
            self.connection_status.object = f"Could not create SAMP bridge service: `{exc}`"

    def _detach_listener(self) -> None:
        if self._listener_token is None:
            return
        try:
            bridge = self._get_samp_service()
            bridge.remove_table_listener(self._listener_token)
        finally:
            self._listener_token = None

    def _poll_samp_connection(self, force_status: bool = False) -> None:
        if self._disposed:
            return
        bridge, ok, error = self._try_start_samp_service()
        if bridge is None:
            self.connection_status.object = f"SAMP bridge unavailable: `{error}`"
            return
        if self._listener_token is None:
            self._listener_token = bridge.add_table_listener(self._on_table_received)
        if ok:
            self._samp_active_seen = True
            try:
                clients = bridge.list_clients(ensure_started=False)
            except Exception:
                clients = []
            names = ", ".join(str(c.get("name") or c.get("id")) for c in clients) or "no other clients yet"
            self.connection_status.object = (
                f"**SAMP hub:** connected. AstronomicAL is listening for `table.load.votable` and `table.load.fits` messages.  \n"
                f"**Detected clients:** {names}"
            )
            self.status.object = "Listening for incoming SAMP tables."
        elif self._samp_active_seen or bool(getattr(bridge, "activity_seen", False)):
            self.connection_status.object = (
                "**SAMP hub:** active/recently detected. AstronomicAL has received SAMP traffic in this session.  \n"
                f"**Last refresh warning:** `{error or 'client list refresh failed'}`"
            )
            if force_status:
                self.status.object = "SAMP was active recently, but the client-list refresh failed. Incoming messages can still be received."
        else:
            self.connection_status.object = (
                "**SAMP hub:** not detected yet. Open TOPCAT, DS9, or another SAMP-capable application; "
                "this panel will retry automatically.  \n"
                f"**Last error:** `{error or 'hub unavailable'}`"
            )
            self.status.object = "Waiting for a SAMP hub. The panel will connect automatically when one appears."

    def _mark_samp_active(self, *, reason: str) -> None:
        self._samp_active_seen = True
        try:
            bridge = self._get_samp_service()
            clients = bridge.list_clients(ensure_started=False) if getattr(bridge, "started", False) else []
        except Exception:
            clients = []
        names = ", ".join(str(c.get("name") or c.get("id")) for c in clients) or "hub active"
        self.connection_status.object = (
            f"**SAMP hub:** connected/active ({reason}). AstronomicAL is listening for `table.load.votable` "
            f"and `table.load.fits` messages.  \n"
            f"**Detected clients:** {names}"
        )

    def _on_table_received(self, payload: dict[str, Any]) -> None:
        self._run_on_ui_thread(lambda: self._handle_received_table(payload))

    def _handle_received_table(self, payload: dict[str, Any]) -> None:
        self._receive_count += 1
        artifact_payload = {
            key: payload.get(key)
            for key in ("sender_id", "msg_id", "is_call", "mtype", "params", "extra", "name", "url", "local_path", "local_url")
        }
        artifact_id = self._put_artifact(
            "interop.samp.import",
            artifact_payload,
            dataset_id=None,
            params={"mtype": payload.get("mtype"), "sender_id": payload.get("sender_id"), "url": payload.get("url")},
            persist=True,
        )
        item_id = f"samp-recv-{uuid.uuid4().hex[:10]}"
        item = {
            "id": item_id,
            "name": payload.get("name") or f"Incoming SAMP table {self._receive_count}",
            "sender_id": payload.get("sender_id") or "unknown",
            "artifact_id": artifact_id,
            "dataset_id": None,
            "url": payload.get("url"),
            "mtype": payload.get("mtype"),
            "local_path": payload.get("local_path"),
            "local_url": payload.get("local_url"),
            "row_count": payload.get("row_count"),
            "column_count": payload.get("column_count"),
            "columns": list(payload.get("columns") or []),
            "preview": None,
            "received_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._received_items.insert(0, item)
        if artifact_id:
            self._publish("artifact.created", {"artifact_id": artifact_id, "type": "interop.samp.import", "dataset_id": None})
        self._publish(
            "interop.samp.table.received",
            {
                "artifact_id": artifact_id,
                "dataset_id": None,
                "table_name": item["name"],
                "sender_id": item["sender_id"],
                "local_path": item["local_path"],
            },
        )
        self._refresh_inbox_options(select_item_id=item_id)
        self._mark_samp_active(reason=f"received table from `{item['sender_id']}`")
        self.status.object = f"Received **{item['name']}** from `{item['sender_id']}`. The table was mirrored locally; preview/import will parse it in a job."

    def _refresh_inbox_body(self) -> None:
        self.inbox_body.objects = [self.empty_state] if not self._received_items else [self.inbox_select]

    def _refresh_inbox_options(self, select_item_id: str | None = None) -> None:
        options = {}
        for item in self._received_items:
            dataset_part = f" → dataset `{item['dataset_id']}`" if item["dataset_id"] else ""
            shape = f"{_safe_count(item.get('row_count'))}x{_safe_count(item.get('column_count'))}"
            label = f"{item['name']} | {shape} | {item['sender_id']} | {item['received_at']}{dataset_part}"
            options[label] = item["id"]
        self.inbox_select.options = options
        self.inbox_info.object = f"**Received tables:** {len(self._received_items)}" if options else "No received tables yet."
        if not options:
            self.inbox_select.value = None
        elif select_item_id is not None:
            self.inbox_select.value = select_item_id
        elif self.inbox_select.value not in options.values():
            self.inbox_select.value = next(iter(options.values()))
        self._refresh_inbox_body()
        self._update_selection_view()

    def _get_selected_item(self) -> dict[str, Any] | None:
        item_id = self.inbox_select.value
        if not item_id:
            return None
        return next((item for item in self._received_items if item["id"] == item_id), None)

    def _build_default_dataset_name(self, item: dict[str, Any]) -> str:
        return item["name"] or f"SAMP import {self._receive_count}"

    def _build_default_dataset_id(self, item: dict[str, Any]) -> str:
        return f"{_normalise_dataset_id(self._build_default_dataset_name(item))}-{item['id'][-4:]}"

    def _update_selection_view(self) -> None:
        item = self._get_selected_item()
        if item is None:
            self.summary.object = "No received table selected."
            self.preview.object = pd.DataFrame()
            self.preview_columns.options = []
            self.preview_columns.value = []
            self.dataset_name.value = ""
            self.dataset_id.value = ""
            self.register_button.disabled = True
            self.activate_button.disabled = True
            self.discard_button.disabled = True
            self.clear_button.disabled = len(self._received_items) == 0
            return

        self.register_button.disabled = item["dataset_id"] is not None
        self.activate_button.disabled = item["dataset_id"] is None
        self.discard_button.disabled = False
        self.clear_button.disabled = False
        if not self.dataset_name.value:
            self.dataset_name.value = self._build_default_dataset_name(item)
        if not self.dataset_id.value:
            self.dataset_id.value = self._build_default_dataset_id(item)

        cols = list(item.get("columns") or [])
        self.preview_columns.options = cols
        self.preview_columns.value = [c for c in self.preview_columns.value if c in cols]
        dataset_line = f"**Registered dataset:** `{item['dataset_id']}`" if item["dataset_id"] else "**Registered dataset:** not yet imported"
        self.summary.object = (
            f"**Table:** {item['name']}  \n"
            f"**Sender:** `{item['sender_id']}`  \n"
            f"**Received:** {item['received_at']}  \n"
            f"**Rows:** {_safe_count(item.get('row_count'))}  \n"
            f"**Columns:** {_safe_count(item.get('column_count'))}  \n"
            f"**Artifact id:** `{item['artifact_id']}`  \n"
            f"{dataset_line}  \n"
            f"**Local mirror:** `{item.get('local_path')}`  \n"
            f"**Source URL:** `{item.get('url')}`"
        )
        preview = item.get("preview")
        self.preview.object = preview if isinstance(preview, pd.DataFrame) else pd.DataFrame()

    def _on_inbox_changed(self, _event) -> None:
        self.dataset_name.value = ""
        self.dataset_id.value = ""
        self._update_selection_view()

    def _preview_clicked(self, _event) -> None:
        item = self._get_selected_item()
        if item is None:
            self.status.object = "No received table selected."
            return
        self.preview_button.disabled = True
        self.status.object = "Parsing preview in a job..."
        self._submit_job(
            self._preview_job,
            title="Preview SAMP table",
            key=f"{self.panel_id}:preview:{item['id']}:{uuid.uuid4().hex}",
            on_done=self._on_preview_done,
            on_error=self._on_preview_error,
            item_id=item["id"],
            url_or_path=item.get("local_path") or item.get("local_url") or item.get("url"),
            preview_rows=int(self.preview_rows.value or 20),
            preview_columns=list(self.preview_columns.value or []),
        )

    def _preview_job(self, *, cancel_token, item_id: str, url_or_path: str, preview_rows: int, preview_columns: list[str]):
        if cancel_token and cancel_token.cancelled():
            return None
        bridge = self._get_samp_service()
        result = bridge.inspect_table(url_or_path, preview_rows=preview_rows, preview_columns=preview_columns)
        result["item_id"] = item_id
        return result

    def _on_preview_done(self, result) -> None:
        self.preview_button.disabled = False
        if result is None:
            self.status.object = "Preview cancelled."
            return
        item = next((x for x in self._received_items if x["id"] == result.get("item_id")), None)
        if item is None:
            self.status.object = "Preview completed, but the inbox item was removed."
            return
        item["row_count"] = result.get("row_count")
        item["column_count"] = result.get("column_count")
        item["columns"] = list(result.get("columns") or [])
        item["preview"] = result.get("preview")
        self._publish(
            "interop.samp.table.previewed",
            {"artifact_id": item["artifact_id"], "table_name": item["name"], "row_count": item["row_count"], "column_count": item["column_count"]},
        )
        self._refresh_inbox_options(select_item_id=item["id"])
        self.status.object = f"Preview loaded for **{item['name']}** ({item['row_count']:,} rows, {item['column_count']:,} columns)."

    def _on_preview_error(self, exc: BaseException) -> None:
        self.preview_button.disabled = False
        self.status.object = f"Preview failed: `{exc}`"

    def _register_clicked(self, _event) -> None:
        item = self._get_selected_item()
        if item is None:
            self.status.object = "No received table selected."
            return
        if item["dataset_id"] is not None:
            self.status.object = f"Selected table is already registered as `{item['dataset_id']}`."
            return

        dataset_name = self.dataset_name.value.strip() or self._build_default_dataset_name(item)
        dataset_id = self.dataset_id.value.strip() or self._build_default_dataset_id(item)
        if self.datasets is None:
            self.status.object = "No DatasetManager available on context."
            return
        try:
            existing_ids = set(self.datasets.list_ids())
        except Exception:
            existing_ids = set()
        if dataset_id in existing_ids:
            self.status.object = f"Dataset id `{dataset_id}` already exists. Choose a different id."
            return

        self.register_button.disabled = True
        self.status.object = "Importing SAMP table as a Parquet-backed dataset..."
        self._submit_job(
            self._register_job,
            title="Import SAMP table",
            key=f"{self.panel_id}:import:{item['id']}:{uuid.uuid4().hex}",
            on_done=self._on_register_done,
            on_error=self._on_register_error,
            item_id=item["id"],
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            url_or_path=item.get("local_path") or item.get("local_url") or item.get("url"),
            source_url=item.get("url"),
            local_path=item.get("local_path"),
        )

    def _infer_column_mappings(self, columns: list[str]) -> dict[str, str]:
        column_by_lower = {str(col).lower(): str(col) for col in columns}
        column_by_normalised = {re.sub(r"[^a-z0-9]+", "", str(col).lower()): str(col) for col in columns}
        mappings: dict[str, str] = {}
        # Conservative auto-mapping: avoid generic "name" because it is often not unique.
        for alias in ("record_id", "source_id", "sourceid", "object_id", "objectid", "obj_id", "objid", "row_id", "rowid", "id", "ID"):
            direct = column_by_lower.get(alias.lower())
            if direct is not None:
                mappings["record_id"] = direct
                break
            normalised = column_by_normalised.get(re.sub(r"[^a-z0-9]+", "", alias.lower()))
            if normalised is not None:
                mappings["record_id"] = normalised
                break
        return mappings

    def _register_job(
        self,
        *,
        cancel_token,
        item_id: str,
        dataset_id: str,
        dataset_name: str,
        url_or_path: str,
        source_url: str | None,
        local_path: str | None,
    ):
        if cancel_token and cancel_token.cancelled():
            return None
        bridge = self._get_samp_service()
        cache_dir = default_cache_dir_for_context(self.context) / "samp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = cache_dir / f"{normalise_dataset_id(dataset_id)}.parquet"
        conversion = bridge.convert_table_to_parquet(url_or_path, parquet_path)
        if cancel_token and cancel_token.cancelled():
            return None
        columns = [str(col) for col in conversion.get("columns") or []]
        mappings = self._infer_column_mappings(columns)
        return {
            "item_id": item_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "parquet_path": str(conversion["parquet_path"]),
            "row_count": int(conversion.get("row_count") or 0),
            "column_count": int(conversion.get("column_count") or len(columns)),
            "columns": columns,
            "column_mappings": mappings,
            "conversion": conversion.get("conversion"),
            "source_url": source_url,
            "local_path": local_path,
        }

    def _force_dataset_mappings(self, dataset_id: str, mappings: dict[str, str]) -> dict[str, str]:
        applied: dict[str, str] = {}
        if not mappings or self.datasets is None:
            return applied
        try:
            dataset = self.datasets.get(dataset_id)
            meta = getattr(dataset, "meta", None)
            if isinstance(meta, dict):
                meta.setdefault("column_mappings", {})
                if isinstance(meta["column_mappings"], dict):
                    meta["column_mappings"].update({str(k): str(v) for k, v in mappings.items()})
        except Exception as exc:
            _samp_debug("force_mapping_meta_failed", dataset_id=dataset_id, error=repr(exc))
        for semantic_name, column_name in mappings.items():
            try:
                self.datasets.set_mapping(dataset_id, str(semantic_name), str(column_name))
                applied[str(semantic_name)] = str(column_name)
            except Exception as exc:
                _samp_debug("set_mapping_failed", dataset_id=dataset_id, semantic_name=semantic_name, column_name=column_name, error=repr(exc))
        return applied

    def _publish_mapping_refresh_events(self, dataset_id: str, mappings: dict[str, str], *, origin: str, delayed: bool = False) -> None:
        suffix = ".delayed" if delayed else ""
        mappings = {str(k): str(v) for k, v in (mappings or {}).items()}
        if mappings:
            self._publish("dataset.mapping.updated", {"dataset_id": dataset_id, "mappings": mappings, "origin": origin + suffix, "changed": True})
            for semantic_name, column_name in mappings.items():
                payload = {
                    "dataset_id": dataset_id,
                    "semantic_name": semantic_name,
                    "column_name": column_name,
                    "mappings": {semantic_name: column_name},
                    "origin": origin + suffix,
                    "changed": True,
                    "required": True,
                }
                self._publish("mapping.resolved", dict(payload, source="astro.samp.receive"))
                self._publish("dataset.mapping.updated", payload)
        self._publish("dataset.updated", {"dataset_id": dataset_id, "source": "interop.samp", "origin": origin + suffix})

    def _schedule_post_import_refresh(self, dataset_id: str, mappings: dict[str, str]) -> None:
        def _emit(delay_label: str):
            try:
                active = None
                try:
                    active = self.datasets.active_id() if self.datasets is not None else None
                except Exception:
                    pass
                _samp_debug("post_import_refresh", dataset_id=dataset_id, active_id=active, mappings=mappings, delay=delay_label)
                if active == dataset_id:
                    self._publish("dataset.active.changed", {"dataset_id": dataset_id, "active_id": dataset_id, "source": "interop.samp", "origin": f"astro.samp.receive.{delay_label}"})
                self._publish_mapping_refresh_events(dataset_id, mappings, origin=f"astro.samp.receive.{delay_label}", delayed=True)
            except Exception as exc:
                _samp_debug("post_import_refresh_failed", dataset_id=dataset_id, error=repr(exc))
        doc = getattr(pn.state, "curdoc", None)
        if doc is not None:
            for delay in (150, 500, 1200):
                try:
                    doc.add_timeout_callback(lambda delay=delay: _emit(f"after_{delay}ms"), delay)
                    continue
                except Exception:
                    pass
        _emit("immediate_fallback")

    def _diagnose_dataset_state(self, dataset_id: str) -> dict[str, Any]:
        out: dict[str, Any] = {"dataset_id": dataset_id}
        try:
            out["active_id"] = self.datasets.active_id()
        except Exception as exc:
            out["active_error"] = repr(exc)
        try:
            out["columns_count"] = len(self.datasets.list_columns(dataset_id))
        except Exception as exc:
            out["columns_error"] = repr(exc)
        try:
            out["mappings"] = dict(self.datasets.get_mappings(dataset_id) or {})
        except Exception as exc:
            out["mappings_error"] = repr(exc)
        try:
            out["row_count"] = self.datasets.row_count(dataset_id)
        except Exception as exc:
            out["row_count_error"] = repr(exc)
        return out

    def _on_register_done(self, result) -> None:
        self.register_button.disabled = False
        if result is None:
            self.status.object = "Import cancelled."
            return
        item = next((x for x in self._received_items if x["id"] == result.get("item_id")), None)
        if item is None:
            self.status.object = "Import completed, but the inbox item was removed."
            return
        item["dataset_id"] = result["dataset_id"]
        item["row_count"] = result["row_count"]
        item["column_count"] = result["column_count"]
        item["columns"] = result["columns"]

        mappings = dict(result.get("column_mappings") or {})
        meta = {
            "domain": "interop.samp",
            "source_format": "samp_table_parquet",
            "source_url": result.get("source_url"),
            "original_samp_url": result.get("source_url"),
            "local_table_path": result.get("local_path"),
            "original_samp_local_path": result.get("local_path"),
            "samp_mtype": item.get("mtype") or self._samp_mtype_for_path(result.get("local_path")),
            "row_count": int(result["row_count"]),
            "rows": int(result["row_count"]),
            "columns": list(result["columns"]),
            "column_mappings": mappings,
            "conversion": result.get("conversion"),
        }
        try:
            self.datasets.register_parquet(
                result["dataset_id"],
                result["parquet_path"],
                name=result["dataset_name"],
                **meta,
            )
        except AttributeError:
            raise RuntimeError("DatasetManager.register_parquet is required for source-backed SAMP imports.")

        mappings = self._force_dataset_mappings(result["dataset_id"], mappings)
        _samp_debug("dataset_registered", **self._diagnose_dataset_state(result["dataset_id"]))

        self._publish(
            "dataset.loaded",
            {
                "dataset_id": result["dataset_id"],
                "name": result["dataset_name"],
                "source": "interop.samp",
                "backend": "duckdb_parquet",
                "rows": int(result["row_count"]),
                "origin": "astro.samp.receive",
            },
        )

        # Make the newly imported dataset active before publishing mapping updates.
        # MappingGatedPanel resolves against the active dataset, so this ordering
        # ensures a waiting Record Browser sees the right dataset when the mapping
        # event arrives.
        if self.make_active.value:
            try:
                self.datasets.set_active(result["dataset_id"])
            except Exception:
                pass
            self._publish(
                "dataset.active.changed",
                {"dataset_id": result["dataset_id"], "active_id": result["dataset_id"], "source": "interop.samp", "origin": "astro.samp.receive"},
            )
        _samp_debug("dataset_after_active", **self._diagnose_dataset_state(result["dataset_id"]))

        self._publish_mapping_refresh_events(result["dataset_id"], mappings, origin="astro.samp.receive")
        self._schedule_post_import_refresh(result["dataset_id"], mappings)
        self._publish(
            "interop.samp.table.imported",
            {
                "artifact_id": item["artifact_id"],
                "dataset_id": result["dataset_id"],
                "table_name": item["name"],
                "sender_id": item["sender_id"],
                "parquet_path": result["parquet_path"],
                "mappings": mappings,
            },
        )
        mapping_note = ""
        if mappings:
            mapping_note = " Auto-mapped " + ", ".join(f"`{k}` → `{v}`" for k, v in mappings.items()) + "."
        self.status.object = (
            f"Imported **{item['name']}** as Parquet-backed dataset `{result['dataset_id']}`."
            + (" It is now active." if self.make_active.value else "")
            + mapping_note
        )
        self._refresh_inbox_options(select_item_id=item["id"])

    def _on_register_error(self, exc: BaseException) -> None:
        self.register_button.disabled = False
        self.status.object = f"Import failed: `{exc}`"

    def _activate_clicked(self, _event) -> None:
        item = self._get_selected_item()
        if item is None or item["dataset_id"] is None:
            self.status.object = "Selected table has not been registered as a dataset yet."
            return
        self.datasets.set_active(item["dataset_id"])
        mappings = {}
        try:
            mappings = dict(self.datasets.get_mappings(item["dataset_id"]) or {})
        except Exception:
            pass
        self._publish("dataset.active.changed", {"dataset_id": item["dataset_id"], "active_id": item["dataset_id"], "source": "interop.samp", "origin": "astro.samp.receive.activate"})
        self._publish_mapping_refresh_events(item["dataset_id"], mappings, origin="astro.samp.receive.activate")
        self.status.object = f"Set dataset `{item['dataset_id']}` as active."

    def _discard_clicked(self, _event) -> None:
        item = self._get_selected_item()
        if item is None:
            self.status.object = "No received table selected."
            return
        self._received_items = [x for x in self._received_items if x["id"] != item["id"]]
        self.status.object = f"Discarded received table **{item['name']}**."
        self._refresh_inbox_options()

    def _clear_clicked(self, _event) -> None:
        self._received_items = []
        self.status.object = "Cleared received-table inbox."
        self._refresh_inbox_options()

    def get_layout(self):
        return _root_column(
            _section("SAMP connection", self.connection_status, self.refresh_connection_button),
            _section("Received tables", self.inbox_info, self.inbox_body),
            _section("Inbox actions", self.discard_button, self.clear_button),
            _section("Selected table", self.summary),
            _section("Preview", self.preview_rows, self.preview_columns, self.preview_button, self.preview),
            _section("Import actions", self.dataset_name, self.dataset_id, self.make_active, self.register_button, self.activate_button),
            _status_block(self.status),
        )

    def dispose(self):
        self._detach_listener()
        self._dispose_base()
