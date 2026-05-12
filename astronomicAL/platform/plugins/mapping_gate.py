from __future__ import annotations

from typing import Any, Dict, Optional

from html import escape

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
    """Wraps a plugin panel and delays construction until mappings are ready."""

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
        self._disposed = False

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
        owner_label = f"Mapping gate: {getattr(self.registration, 'title', self._mapping_panel_id())}"

        self._subscriptions.append(
            events.subscribe(
                "dataset.mapping_updated",
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

    def _dataset_id(self) -> str:
        return active_dataset_id(self.context) or "main"

    def _show_waiting_for_dataset(self, dataset_id: str | None) -> None:
        import panel as pn

        label = dataset_id or "active dataset"

        self.view[:] = [
            pn.Column(
                pn.pane.Markdown(
                    f"""
### Waiting for dataset

This panel has column mapping requirements, but `{label}` is not loaded yet.

Saved workspace mappings will be applied automatically after the dataset is loaded.
                    """,
                    sizing_mode="stretch_width",
                ),
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

    def _resolve(self):
        return resolve_mapping_requirements(
            context=self.context,
            dataset_id=self._dataset_id(),
            required_mappings=getattr(self.registration, "required_mappings", []),
            optional_mappings=getattr(self.registration, "optional_mappings", []),
        )

    def _refresh(self) -> None:
        if self._disposed:
            return

        dataset_id = self._dataset_id()

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
        if self._real_view is not None:
            return

        self._show_loading()

        def _finish() -> None:
            if self._disposed or self._real_view is not None:
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
                self._real_view = view
                self._real_controller = controller
                self.view[:] = [self._real_view]

            except Exception as exc:
                self.view[:] = [
                    pn.Column(
                        pn.pane.Alert(
                            f"Could not load **{escape(str(getattr(self.registration, 'title', 'Plugin panel')))}**.",
                            alert_type="danger",
                            sizing_mode="stretch_width",
                            margin=(0, 0, 8, 0),
                        ),
                        pn.pane.HTML(
                            f"""
                            <div style="font-size: 12px; color: #555; white-space: pre-wrap;
                                        border: 1px solid #e1e1e1; border-radius: 6px;
                                        padding: 8px; background: #fff;">
                                {escape(str(exc))}
                            </div>
                            """,
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
                ]

        try:
            doc = pn.state.curdoc
        except Exception:
            doc = None

        if doc is not None:
            try:
                doc.add_timeout_callback(_finish, 50)
                return
            except Exception:
                try:
                    doc.add_next_tick_callback(_finish)
                    return
                except Exception:
                    pass

        _finish()

    def _on_dataset_mapping_updated(self, _topic: str, payload: Any) -> None:
        if not payload:
            return

        if payload.get("dataset_id") != self._dataset_id():
            return

        self._refresh()

    def _on_dataset_active_changed(self, _topic: str, _payload: Any) -> None:
        self._mapping_requests_sent.clear()
        self._real_view = None
        self._dispose_real_controller()
        self._refresh()

    def _dispose_real_controller(self) -> None:
        controller = self._real_controller
        self._real_controller = None

        if controller is not None and hasattr(controller, "dispose"):
            try:
                controller.dispose()
            except Exception:
                pass

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

        self._disposed = True

        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in list(self._subscriptions):
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass

        self._subscriptions.clear()
        self._dispose_real_controller()