from __future__ import annotations

from html import escape
from typing import Any, Dict, Optional

import panel as pn

from astronomicAL.platform.mapping_requirements import (
    active_dataset_id,
    publish_mapping_requests,
    resolve_mapping_requirements,
)
from astronomicAL.platform.panel_state import get_controller_state


class MissingColumnsPanel:
    """Temporary placeholder shown while required plugin mappings are missing.

    Kept deliberately compact and scrollable because this panel is commonly
    shown inside small React grid cells.
    """

    def __init__(
        self,
        *,
        context: Any,
        title: str,
        dataset_id: str,
        source: str,
        panel_id: Optional[str],
        missing_required: list[Any],
        missing_optional: list[Any],
        resend_requests,
    ) -> None:
        self.context = context
        self.title = title
        self.dataset_id = dataset_id
        self.source = source
        self.panel_id = panel_id
        self.missing_required = list(missing_required or [])
        self.missing_optional = list(missing_optional or [])
        self._resend_requests = resend_requests

        self.open_button = pn.widgets.Button(
            name="Open column mapping",
            button_type="primary",
            height=32,
            min_width=150,
            sizing_mode="stretch_width",
        )
        self.open_button.on_click(self._open_mapping)

        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="light",
            height=32,
            min_width=90,
            sizing_mode="stretch_width",
        )
        self.refresh_button.on_click(lambda _event: self._resend_requests())

        self.header = pn.pane.HTML(
            self._header_html(),
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )
        self.body = pn.pane.HTML(
            self._body_html(),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
        self.button_bar = pn.Row(
            self.open_button,
            self.refresh_button,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        self.view = pn.Column(
            self.header,
            self.button_bar,
            self.body,
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box",
                "padding": "8px",
                "overflow": "auto",
                "height": "100%",
                "min-width": "0",
            },
        )

    def _open_mapping(self, _event=None) -> None:
        self._resend_requests()
        events = getattr(self.context, "events", None)
        if events is not None:
            events.publish(
                "mapping.open_requested",
                {
                    "source": self.source,
                    "panel_id": self.panel_id,
                    "dataset_id": self.dataset_id,
                },
            )

    def _header_html(self) -> str:
        title = escape(str(self.title))
        dataset_id = escape(str(self.dataset_id))
        return f"""
<div style="
  border: 1px solid #f0d98a;
  background: #fff4c7;
  color: #6b5100;
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 13px;
  line-height: 1.35;
  box-sizing: border-box;
">
  <div style="font-weight: 700; margin-bottom: 3px;">
    {title}
  </div>
  <div>
    This plugin needs dataset column mappings before it can open.
  </div>
  <div style="margin-top: 6px; font-size: 12px;">
    Dataset: <code>{dataset_id}</code>
  </div>
</div>
"""

    def _requirement_items_html(self, requirements: list[Any]) -> str:
        if not requirements:
            return """
<div style="color: #777; font-size: 12px; padding: 4px 0;">
  None
</div>
"""

        items = []
        for req in requirements:
            label = escape(str(req.display_name or req.semantic_name))
            semantic = escape(str(req.semantic_name))
            description = escape(str(req.description or ""))
            description_html = (
                f"""
<div style="
  color: #555;
  margin-top: 3px;
  overflow-wrap: anywhere;
  word-break: normal;
">
  {description}
</div>
"""
                if description
                else ""
            )
            items.append(
                f"""
<div style="
  border: 1px solid #e6e6e6;
  border-radius: 6px;
  padding: 7px 8px;
  margin-bottom: 6px;
  background: #fff;
  box-sizing: border-box;
  min-width: 0;
">
  <div style="
    font-weight: 700;
    font-size: 12px;
    overflow-wrap: anywhere;
  ">
    {label}
  </div>
  <div style="
    color: #557fd8;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 11px;
    margin-top: 2px;
    overflow-wrap: anywhere;
  ">
    {semantic}
  </div>
  {description_html}
</div>
"""
            )
        return "\n".join(items)

    def _body_html(self) -> str:
        required_html = self._requirement_items_html(self.missing_required)
        optional_html = self._requirement_items_html(self.missing_optional)
        return f"""
<div style="
  box-sizing: border-box;
  min-width: 0;
  font-size: 12px;
  line-height: 1.35;
">
  <div style="
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px;
    align-items: start;
  ">
    <section style="min-width: 0;">
      <div style="
        font-weight: 700;
        font-size: 13px;
        margin: 0 0 6px 0;
      ">
        Required columns
      </div>
      {required_html}
    </section>
    <section style="min-width: 0;">
      <div style="
        font-weight: 700;
        font-size: 13px;
        margin: 0 0 6px 0;
      ">
        Optional columns
      </div>
      {optional_html}
    </section>
  </div>
</div>
"""


class MappingGatedPanel:
    """Wrap a plugin panel and delay construction until mappings are ready.

    The wrapper owns the dataset-specific concrete controller lifecycle.  Its
    target dataset is synchronized from the live platform state on every
    refresh so mapping and active-dataset events may arrive in either order
    without causing duplicate materialisation of the same concrete panel.
    """

    state_version = 1

    def __init__(
        self,
        *,
        context: Any,
        manager: Any,
        registration: Any,
        kwargs: Optional[Dict[str, Any]] = None,
        instance_id: Optional[str] = None,
        restore_state: Optional[Dict[str, Any]] = None,
        restore_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.context = context
        self.manager = manager
        self.registration = registration
        self.kwargs = dict(kwargs or {})
        self.instance_id = instance_id
        self.restore_state = dict(restore_state or {})
        self.restore_metadata = dict(restore_metadata or {})

        self.view = pn.Column(
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
            styles={
                "height": "100%",
                "width": "100%",
                "overflow": "hidden",
                "box-sizing": "border-box",
                "min-width": "0",
            },
        )

        self._subscriptions: list[Any] = []
        self._mapping_requests_sent: set[tuple[str, str, str]] = set()
        self._real_view: Any = None
        self._real_controller: Any = None
        self._real_dataset_id: Optional[str] = None
        self._real_load_started = False
        self._real_load_handle: Any = None
        self._real_load_generation = 0
        self._real_load_dataset_id: Optional[str] = None

        # This is the dataset the wrapper itself represents.  It must advance
        # as soon as *any* refresh observes a new active dataset, not only when
        # dataset.active.changed happens to arrive.  Mapping updates can be
        # published first during dataset registration/import.
        self._panel_dataset_id = self._dataset_id()
        self._disposed = False
        self._document: Any = None
        self._refresh_scheduled = False
        self._refresh_pending = False

        self._subscribe()
        self._refresh()

    def _show_loading(self) -> None:
        title = getattr(self.registration, "title", "Plugin panel")
        self.view[:] = [
            pn.Column(
                pn.Spacer(height=8),
                pn.indicators.LoadingSpinner(
                    value=True,
                    width=42,
                    height=42,
                    sizing_mode="fixed",
                    margin=(8, 0, 8, 0),
                ),
                pn.pane.HTML(
                    f"""
<div style="text-align: center; padding: 0 12px;">
  <div style="font-weight: 700; font-size: 15px; margin-bottom: 6px;">
    Loading {escape(str(title))}
  </div>
  <div style="font-size: 12px; color: #666;">
    Preparing mapped panel...
  </div>
</div>
""",
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_both",
                align="center",
                margin=(0, 0, 0, 0),
                styles={
                    "height": "100%",
                    "width": "100%",
                    "box-sizing": "border-box",
                    "display": "flex",
                    "align-items": "center",
                    "justify-content": "center",
                    "overflow": "hidden",
                    "background": "#fafafa",
                },
            )
        ]

    def _subscribe(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        owner_id = f"mapping_gate:{self._mapping_panel_id()}"
        owner_label = (
            f"Mapping gate: "
            f"{getattr(self.registration, 'title', self._mapping_panel_id())}"
        )
        self._subscriptions.append(
            events.subscribe(
                "dataset.mapping.updated",
                self._on_dataset_mapping_updated,
                owner_id=owner_id,
                owner_label=owner_label,
                owner_kind="panel",
            )
        )
        self._subscriptions.append(
            events.subscribe(
                "dataset.active.changed",
                self._on_dataset_active_changed,
                owner_id=owner_id,
                owner_label=owner_label,
                owner_kind="panel",
            )
        )

    def _resolve_document(self) -> Any:
        """Return the live Bokeh document owning this wrapper, when attached.

        Mapping events are synchronous and may be published from JobManager worker
        threads during dataset/import registration.  Once this wrapper is attached to
        a server document, all Panel/Bokeh mutations and concrete panel construction
        must be marshalled back to that document instead of running in the publishing
        worker thread.
        """

        for panel_object in (self.view, self._real_view):
            models = getattr(panel_object, "_models", None)
            if not isinstance(models, dict):
                continue
            for value in models.values():
                model = value[0] if isinstance(value, tuple) and value else value
                doc = getattr(model, "document", None)
                if doc is not None:
                    self._document = doc
                    return doc

        doc = self._document
        if doc is not None:
            return doc
        try:
            doc = pn.state.curdoc
        except Exception:
            doc = None
        if doc is not None:
            self._document = doc
        return doc

    def _has_attached_models(self) -> bool:
        for panel_object in (self.view, self._real_view):
            models = getattr(panel_object, "_models", None)
            if not isinstance(models, dict):
                continue
            for value in models.values():
                model = value[0] if isinstance(value, tuple) and value else value
                if getattr(model, "document", None) is not None:
                    return True
        return False

    def _schedule_refresh(self) -> None:
        """Coalesce mapping/dataset events and refresh on the owning UI document.

        Importers may publish several mapping and active-dataset events while a single
        registration job is still running.  Coalescing means the UI observes the final
        live dataset state rather than constructing intermediate concrete panels, and
        prevents worker-thread callbacks from mutating an attached Bokeh document.
        """

        if self._disposed:
            return
        self._refresh_pending = True
        if self._refresh_scheduled:
            return

        def run() -> None:
            self._refresh_scheduled = False
            if self._disposed or not self._refresh_pending:
                return
            self._refresh_pending = False
            self._refresh()

        doc = self._resolve_document()
        add_next = getattr(doc, "add_next_tick_callback", None)
        if callable(add_next):
            self._refresh_scheduled = True
            try:
                add_next(run)
                return
            except Exception:
                self._refresh_scheduled = False
                # A closing document may reject callbacks.  Never fall back to
                # mutating models that are still attached from this worker thread.
                if self._has_attached_models():
                    self._refresh_pending = False
                    return

        # Detached construction/unit tests have no document yet, so direct refresh is
        # safe and preserves the existing synchronous behaviour.
        self._refresh_pending = False
        self._refresh()

    def _dataset_id(self) -> str:
        return active_dataset_id(self.context) or "main"

    def _show_waiting_for_dataset(self, dataset_id: str | None) -> None:
        label = dataset_id or "active dataset"
        if label == "main":
            message = """
### Waiting for dataset

This panel has column mapping requirements, but a dataset has not been loaded yet.
Saved workspace mappings will be applied automatically after the dataset is loaded.
"""
        else:
            message = f"""
### Waiting for dataset

This panel has column mapping requirements, but the dataset (`{label}`) is not loaded yet.
Saved workspace mappings will be applied automatically after the dataset is loaded.
"""

        self.view[:] = [
            pn.Column(
                pn.pane.Markdown(message, sizing_mode="stretch_width"),
                sizing_mode="stretch_both",
                margin=10,
            )
        ]

    def _mapping_source(self) -> str:
        return getattr(self.registration, "id", None) or getattr(
            self.registration,
            "plugin_id",
            "plugin",
        )

    def _mapping_panel_id(self) -> str:
        return getattr(self.registration, "id", self._mapping_source())

    def _withdraw_mapping_requests(
        self,
        *,
        dataset_id: Optional[str] = None,
    ) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        events.publish(
            "mapping.request.withdrawn",
            {
                "dataset_id": str(dataset_id or self._dataset_id()),
                "panel_id": self._mapping_panel_id(),
                "source": self._mapping_source(),
            },
        )

    def _resolve(self):
        return resolve_mapping_requirements(
            context=self.context,
            dataset_id=self._dataset_id(),
            required_mappings=getattr(self.registration, "required_mappings", []),
            optional_mappings=getattr(self.registration, "optional_mappings", []),
        )

    def _synchronise_panel_dataset(self, dataset_id: str) -> bool:
        """Synchronise the wrapper-owned dataset identity with live platform state.

        Dataset registration can publish ``dataset.mapping.updated`` before a
        corresponding ``dataset.active.changed`` callback reaches this panel.
        Prior to this guard, ``_refresh`` could begin constructing the concrete
        panel for the new active dataset while ``_panel_dataset_id`` still named
        the old one.  The later active-dataset callback then treated the same
        target as a new transition, invalidated the in-progress construction,
        and started it a second time.

        Centralising the transition here makes event order irrelevant.  A true
        dataset change still captures/disposes the old concrete controller and
        begins a fresh dataset-specific lifecycle exactly once.
        """

        new_dataset_id = str(dataset_id or "main")
        if new_dataset_id == self._panel_dataset_id:
            return False

        previous_dataset_id = self._panel_dataset_id
        if previous_dataset_id:
            self._withdraw_mapping_requests(dataset_id=previous_dataset_id)

        self._panel_dataset_id = new_dataset_id
        self._mapping_requests_sent.clear()
        self._reset_real_panel()
        return True

    def _refresh(self) -> None:
        if self._disposed:
            return

        dataset_id = self._dataset_id()

        # Synchronise before *any* mapping resolution or concrete construction.
        # This is the lifecycle ordering invariant that prevents duplicate panel
        # factories when mapping and active-dataset events arrive in either order.
        self._synchronise_panel_dataset(dataset_id)

        datasets = getattr(self.context, "datasets", None)
        dataset_loaded = False
        if datasets is not None:
            try:
                if hasattr(datasets, "has_dataset"):
                    dataset_loaded = bool(datasets.has_dataset(dataset_id))
                elif hasattr(datasets, "_datasets"):
                    dataset_loaded = dataset_id in datasets._datasets
                else:
                    datasets.get_df(dataset_id)
                    dataset_loaded = True
            except Exception:
                dataset_loaded = False

        if not dataset_loaded:
            self._show_waiting_for_dataset(dataset_id)
            return

        state = self._resolve()
        source = self._mapping_source()
        panel_id = self._mapping_panel_id()
        publish_mapping_requests(
            context=self.context,
            dataset_id=state.dataset_id,
            source=source,
            panel_id=panel_id,
            requirements=[*state.missing_required, *state.missing_optional],
            sent_keys=self._mapping_requests_sent,
        )

        if state.missing_required:
            self._show_missing_columns(state)
            return

        self._show_real_panel()

    def _show_missing_columns(self, state) -> None:
        placeholder = MissingColumnsPanel(
            context=self.context,
            title=getattr(self.registration, "title", "Plugin panel"),
            dataset_id=state.dataset_id,
            source=self._mapping_source(),
            panel_id=self._mapping_panel_id(),
            missing_required=state.missing_required,
            missing_optional=state.missing_optional,
            resend_requests=self._force_resend_requests,
        )
        self.view[:] = [placeholder.view]

    def _force_resend_requests(self) -> None:
        self._mapping_requests_sent.clear()
        self._refresh()

    def _show_real_panel(self) -> None:
        """Construct the mapped plugin panel on the Panel UI thread.

        Each scheduled construction is associated with both a monotonically
        increasing generation and the active dataset ID. A delayed callback is
        discarded when either value has changed before the callback runs.
        """

        if self._real_view is not None or self._real_load_started:
            return

        dataset_id = self._dataset_id()
        if dataset_id != self._panel_dataset_id:
            # A caller bypassed _refresh or platform state changed between the
            # refresh and this method.  Re-enter through _refresh so the dataset
            # lifecycle transition is applied before any factory is invoked.
            self._refresh()
            return

        self._real_load_generation += 1
        generation = self._real_load_generation
        self._real_load_started = True
        self._real_load_dataset_id = dataset_id
        self._show_loading()

        def _finish_on_ui() -> None:
            if generation == self._real_load_generation:
                self._real_load_handle = None
            if not self._load_is_current(generation, dataset_id):
                return

            try:
                view, controller = self.manager._create_panel_now(
                    self.registration,
                    self.context,
                    instance_id=self.instance_id,
                    restore_state=self.restore_state,
                    restore_metadata=self.restore_metadata,
                    **self.kwargs,
                )
            except BaseException as exc:
                if not self._load_is_current(generation, dataset_id):
                    return
                self._real_load_started = False
                self._real_load_dataset_id = None
                self.view[:] = [self._error_view(exc)]
                return

            if not self._load_is_current(generation, dataset_id):
                self._dispose_controller(controller)
                return

            self._real_load_started = False
            self._real_load_dataset_id = None
            self._real_dataset_id = dataset_id
            self._real_view = view
            self._real_controller = controller
            self.view[:] = [view]

        try:
            doc = pn.state.curdoc
        except Exception:
            doc = None

        if doc is not None:
            try:
                self._real_load_handle = doc.add_timeout_callback(
                    _finish_on_ui,
                    150,
                )
                return
            except Exception:
                try:
                    self._real_load_handle = doc.add_next_tick_callback(
                        _finish_on_ui,
                    )
                    return
                except Exception:
                    self._real_load_handle = None

        _finish_on_ui()

    def _load_is_current(self, generation: int, dataset_id: str) -> bool:
        """Return whether a scheduled construction still belongs to this panel."""

        if self._disposed:
            return False
        if not self._real_load_started:
            return False
        if generation != self._real_load_generation:
            return False
        if dataset_id != self._real_load_dataset_id:
            return False
        if dataset_id != self._panel_dataset_id:
            return False
        return dataset_id == self._dataset_id()

    def _error_view(self, exc: BaseException) -> pn.Column:
        title = escape(str(getattr(self.registration, "title", "Plugin panel")))
        return pn.Column(
            pn.pane.Alert(
                f"Could not load **{title}**.",
                alert_type="danger",
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            ),
            pn.pane.HTML(
                f"<pre style='white-space: pre-wrap'>{escape(str(exc))}</pre>",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
            styles={
                "height": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "padding": "10px",
                "overflow": "auto",
            },
        )

    def _capture_real_controller_state(self) -> None:
        """Preserve current user-controlled panel state before reconstruction."""

        controller = self._real_controller
        if controller is None:
            return
        try:
            state = get_controller_state(controller)
        except Exception:
            return
        if isinstance(state, dict):
            self.restore_state = dict(state)

    def _cancel_pending_real_load(self) -> None:
        """Invalidate and best-effort cancel the currently scheduled load."""

        self._real_load_generation += 1
        self._real_load_started = False
        self._real_load_dataset_id = None

        handle = self._real_load_handle
        self._real_load_handle = None
        if handle is not None and hasattr(handle, "cancel"):
            try:
                handle.cancel()
            except Exception:
                pass

    def _reset_real_panel(self) -> None:
        """Dispose the current dataset-specific controller and clear its view."""

        self._capture_real_controller_state()
        self._cancel_pending_real_load()
        self._dispose_real_controller()
        self._real_view = None
        self._real_dataset_id = None
        self.view[:] = []

    def _on_dataset_mapping_updated(self, _topic: str, payload: Any) -> None:
        if not payload:
            return
        if payload.get("dataset_id") != self._dataset_id():
            return
        self._schedule_refresh()

    def _on_dataset_active_changed(self, _topic: str, payload: Any) -> None:
        if self._disposed:
            return

        # Do not synchronise/reset Panel models in the EventBus publisher thread.
        # _refresh reads the authoritative live active dataset after the callback has
        # been marshalled/coalesced onto the owning Bokeh document.
        self._schedule_refresh()

    @staticmethod
    def _dispose_controller(controller: Any) -> None:
        if controller is None or not hasattr(controller, "dispose"):
            return
        try:
            controller.dispose()
        except Exception:
            pass

    def _dispose_real_controller(self) -> None:
        controller = self._real_controller
        self._real_controller = None
        self._dispose_controller(controller)

    def get_state(self) -> dict[str, Any]:
        if self._real_controller is not None:
            return get_controller_state(self._real_controller)
        return {
            "gated": True,
            "registration_id": getattr(self.registration, "id", None),
            "restore_state": dict(self.restore_state),
            "restore_metadata": dict(self.restore_metadata),
            "open_kwargs": dict(self.kwargs),
        }

    def dispose(self) -> None:
        if self._disposed:
            return

        self._withdraw_mapping_requests(dataset_id=self._panel_dataset_id)
        self._disposed = True
        self._refresh_pending = False
        self._refresh_scheduled = False
        self._cancel_pending_real_load()

        events = getattr(self.context, "events", None)
        if events is not None:
            for subscription in list(self._subscriptions):
                try:
                    events.unsubscribe(subscription)
                except Exception:
                    pass

        self._subscriptions.clear()
        self._mapping_requests_sent.clear()
        self._dispose_real_controller()
        self._real_view = None
        self._real_dataset_id = None
        self.view[:] = []
