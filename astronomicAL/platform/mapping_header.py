from __future__ import annotations

from html import escape
from typing import Any, Dict, List, Tuple

import panel as pn

from astronomicAL.platform.modal_utils import open_template_modal


PendingKey = Tuple[str, str]
CurrentKey = Tuple[str, str]

SHEET_STYLES = {
    "background": "#ffffff",
    "border-radius": "16px",
    "padding": "0px",
    "border": "1px solid #e5e7eb",
    "box-shadow": "0 8px 26px rgba(15, 23, 42, 0.12)",
    "box-sizing": "border-box",
    "overflow": "hidden",
}

ROOT_STYLES = {
    "box-sizing": "border-box",
    "padding": "12px 0 36px 0",
    "overflow": "visible",
}

INTRO_STYLES = {
    "color": "#475569",
    "font-size": "0.96rem",
    "line-height": "1.45",
}

DATASET_SECTION_STYLES = {
    "background": "#f8fafc",
    "border-radius": "14px",
    "padding": "16px",
    "border": "1px solid #e2e8f0",
    "box-sizing": "border-box",
    "overflow": "visible",
}

FIELD_BLOCK_STYLES = {
    "background": "#ffffff",
    "border-radius": "10px",
    "padding": "12px",
    "border": "1px solid #e5e7eb",
    "box-shadow": "0 1px 4px rgba(15, 23, 42, 0.04)",
    "box-sizing": "border-box",
    "overflow": "visible",
}

CURRENT_ROW_STYLES = {
    "background": "#ffffff",
    "border-radius": "10px",
    "padding": "10px 12px",
    "border": "1px solid #e5e7eb",
    "box-sizing": "border-box",
    "overflow": "visible",
}

FOOTER_STYLES = {
    "border-top": "1px solid #e5e7eb",
    "padding": "14px 20px 16px 20px",
    "background": "#ffffff",
    "box-sizing": "border-box",
}

BADGE_REQUIRED = """
<span style="display:inline-block;background:#fee2e2;color:#991b1b;border:1px solid #fecaca;border-radius:999px;padding:3px 10px;font-size:0.78rem;font-weight:700;line-height:1.2;white-space:nowrap;">Required</span>
"""

BADGE_OPTIONAL = """
<span style="display:inline-block;background:#dbeafe;color:#1d4ed8;border:1px solid #bfdbfe;border-radius:999px;padding:3px 10px;font-size:0.78rem;font-weight:700;line-height:1.2;white-space:nowrap;">Optional</span>
"""


class MappingAlertController:
    """Header/modal controller for semantic dataset-column mappings.

    Pending requirements are intentionally applied per mapping. Blank values are
    ignored, including during "Apply all selected". Existing mappings are shown
    separately so users can inspect, update, or clear earlier choices.
    """

    def __init__(self, context, template) -> None:
        self.context = context
        self.template = template

        self._pending: Dict[PendingKey, dict] = {}
        self._selectors: Dict[PendingKey, pn.widgets.Select] = {}
        self._pending_values: Dict[PendingKey, str] = {}
        self._current_selectors: Dict[CurrentKey, pn.widgets.Select] = {}
        self._notice: dict[str, str] | None = None
        self._subs = []

        self.button = pn.widgets.Button(
            name="Mappings: 0",
            button_type="default",
            width=150,
            height=34,
            margin=(0, 0, 0, 0),
        )
        self.button.on_click(self._open_modal)

        self.view = pn.Row(
            self.button,
            width=150,
            height=40,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
            align="center",
            styles={
                "display": "flex",
                "align-items": "center",
            },
        )

        self.modal_header = pn.Column(
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={"padding": "22px 24px 0 24px", "box-sizing": "border-box"},
        )
        self.modal_body = pn.Column(
            sizing_mode="stretch_width",
            height=430,
            scroll=True,
            margin=(0, 0, 0, 0),
            styles={
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "padding": "0 24px 10px 24px",
                "box-sizing": "border-box",
            },
        )
        self.modal_footer = pn.Row(
            sizing_mode="stretch_width",
            height=72,
            margin=(0, 0, 0, 0),
            styles=FOOTER_STYLES,
        )
        self.modal_sheet = pn.Column(
            self.modal_header,
            self.modal_body,
            self.modal_footer,
            sizing_mode="fixed",
            width=960,
            height=622,
            margin=(0, 0, 0, 0),
            styles=SHEET_STYLES,
        )
        self.modal_root = pn.Column(
            pn.Row(
                pn.layout.HSpacer(),
                self.modal_sheet,
                pn.layout.HSpacer(),
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles=ROOT_STYLES,
        )

        if getattr(self.context, "events", None) is not None:
            self._subs.append(
                self.context.events.subscribe("mapping.requested", self._on_mapping_requested)
            )
            self._subs.append(
                self.context.events.subscribe("mapping.resolved", self._on_mapping_resolved)
            )
            self._subs.append(
                self.context.events.subscribe(
                    "mapping.open_requested", self._on_mapping_open_requested
                )
            )

        self._refresh_button()

    # ------------------------------------------------------------------
    # Lifecycle and event intake
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        if getattr(self.context, "events", None) is None:
            return
        for sub in self._subs:
            try:
                self.context.events.unsubscribe(sub)
            except Exception:
                pass
        self._subs = []

    def _on_mapping_open_requested(self, _topic: str, _payload: Any) -> None:
        self._rebuild_modal()
        open_template_modal(self.template, self.modal_root)

    def _on_mapping_requested(self, _topic: str, payload: Any) -> None:
        if not payload:
            return

        payload = self._normalise_payload(payload)
        dataset_id = payload.get("dataset_id")
        semantic_name = payload.get("semantic_name")
        if not dataset_id or not semantic_name:
            return

        existing_mapping = self.context.datasets.get_mapping(dataset_id, semantic_name)
        if existing_mapping is not None:
            self._refresh_button()
            return

        key = self._key_from_payload(payload)
        existing = self._pending.get(key)
        self._pending[key] = payload if existing is None else self._merge_pending_item(existing, payload)
        self._refresh_button()

    def _on_mapping_resolved(self, _topic: str, payload: Any) -> None:
        if not payload:
            return
        dataset_id = payload.get("dataset_id")
        semantic_name = payload.get("semantic_name")
        if not dataset_id or not semantic_name:
            return
        key = (str(dataset_id), str(semantic_name))
        self._pending.pop(key, None)
        self._pending_values.pop(key, None)
        self._refresh_button()

    # ------------------------------------------------------------------
    # Normalisation and status
    # ------------------------------------------------------------------

    def _key_from_payload(self, payload: dict) -> PendingKey:
        return str(payload["dataset_id"]), str(payload["semantic_name"])

    def _normalise_payload(self, payload: dict) -> dict:
        payload = dict(payload)
        payload.setdefault("source", "unknown")
        payload.setdefault("required", True)
        payload.setdefault("candidates", [])
        payload.setdefault("display_name", payload.get("semantic_name", "Column"))
        payload.setdefault("description", "")
        payload.setdefault("config_key", None)
        payload.setdefault("suggested", None)
        payload.setdefault("panel_id", None)

        panel_id = payload.get("panel_id")
        config_key = payload.get("config_key")
        payload["sources"] = [str(payload.get("source") or "unknown")]
        payload["panel_ids"] = [panel_id] if panel_id else []
        payload["config_keys"] = [config_key] if config_key else []
        payload["dataset_id"] = str(payload["dataset_id"])
        payload["semantic_name"] = str(payload["semantic_name"])
        payload["required"] = bool(payload.get("required", True))
        payload["candidates"] = self._unique_list(payload.get("candidates", []))
        return payload

    def _merge_pending_item(self, existing: dict, incoming: dict) -> dict:
        existing_was_required = bool(existing.get("required", True))
        incoming_is_required = bool(incoming.get("required", True))

        existing["required"] = existing_was_required or incoming_is_required
        existing["sources"] = self._unique_list(
            list(existing.get("sources", [])) + list(incoming.get("sources", []))
        )
        existing["panel_ids"] = self._unique_list(
            list(existing.get("panel_ids", [])) + list(incoming.get("panel_ids", []))
        )
        existing["config_keys"] = self._unique_list(
            list(existing.get("config_keys", [])) + list(incoming.get("config_keys", []))
        )
        existing["candidates"] = self._unique_list(
            list(existing.get("candidates", [])) + list(incoming.get("candidates", []))
        )

        if incoming_is_required and not existing_was_required:
            for field in ("display_name", "description", "suggested", "config_key"):
                existing[field] = incoming.get(field) or existing.get(field)
            return existing

        for field in ("display_name", "description", "suggested", "config_key"):
            if not existing.get(field) and incoming.get(field):
                existing[field] = incoming.get(field)
        return existing

    def _unique_list(self, values: List[Any]) -> List[Any]:
        seen = set()
        out = []
        for value in values:
            if value in (None, ""):
                continue
            key = str(value)
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out

    def _required_count(self) -> int:
        return sum(1 for item in self._pending.values() if item.get("required", True))

    def _current_mapping_count(self) -> int:
        count = 0
        for dataset_id in self._dataset_ids():
            try:
                count += len(self.context.datasets.get_mappings(dataset_id))
            except Exception:
                pass
        return count

    def _refresh_button(self) -> None:
        pending_total = len(self._pending)
        required_total = self._required_count()
        current_total = self._current_mapping_count()

        self.button.disabled = False

        if required_total > 0:
            self.button.name = f"⚠ {required_total} Required"
            self.button.button_type = "warning"
        elif pending_total > 0:
            self.button.name = f"ℹ {pending_total} Optional"
            self.button.button_type = "default"
        elif current_total > 0:
            self.button.name = "View Mappings"
            self.button.button_type = "default"
        else:
            self.button.name = "Mappings: 0"
            self.button.button_type = "default"

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _dataset_ids(self) -> list[str]:
        try:
            return [str(dataset_id) for dataset_id in self.context.datasets.list_ids()]
        except Exception:
            return []

    def _dataset_name(self, dataset_id: str) -> str:
        try:
            dataset = self.context.datasets.get(dataset_id)
            return str(getattr(dataset, "name", dataset_id) or dataset_id)
        except Exception:
            return str(dataset_id)

    def _dataset_columns(self, dataset_id: str) -> list[str]:
        try:
            return [str(col) for col in self.context.datasets.list_columns(dataset_id)]
        except Exception:
            return []

    def _select_options(self, raw_options: list[Any], current: Any = None, selected: Any = None) -> list[str]:
        options = [""]
        for value in raw_options:
            if value in (None, ""):
                continue
            text = str(value)
            if text not in options:
                options.append(text)
        for value in (current, selected):
            if value not in (None, ""):
                text = str(value)
                if text not in options:
                    options.insert(1, text)
        return options

    def _sources_text(self, item: dict) -> str:
        sources = [str(source) for source in item.get("sources", []) if source]
        if not sources:
            return "unknown"
        if len(sources) <= 4:
            return ", ".join(sources)
        shown = ", ".join(sources[:4])
        return f"{shown}, +{len(sources) - 4} more"

    def _set_notice(self, message: str, alert_type: str = "info") -> None:
        self._notice = {"message": message, "type": alert_type}

    def _capture_pending_selector_values(self) -> None:
        for key, selector in list(self._selectors.items()):
            if key not in self._pending:
                self._pending_values.pop(key, None)
                continue
            value = selector.value
            self._pending_values[key] = "" if value in (None, "") else str(value)

    # ------------------------------------------------------------------
    # Modal build
    # ------------------------------------------------------------------

    def _open_modal(self, _event=None) -> None:
        self._rebuild_modal()
        open_template_modal(self.template, self.modal_root)

    def _rebuild_modal(self) -> None:
        self._capture_pending_selector_values()
        self._selectors = {}
        self._current_selectors = {}

        self.modal_header[:] = [
            pn.pane.HTML(
                """
<h2 style="margin:0 0 14px 0;color:#17202a;font-size:1.55rem;line-height:1.25;font-weight:700;">Resolve Dataset Requirements</h2>
<div style="color:#475569;font-size:0.96rem;line-height:1.45;margin:0 0 18px 0;">
Choose the dataset columns needed by active panels. Pending requirements are applied one at a time unless you use <strong>Apply all selected</strong>. Blank dropdowns are ignored and remain awaiting mapping.
</div>
""",
                margin=(0, 0, 0, 0),
                sizing_mode="stretch_width",
                styles=INTRO_STYLES,
            ),
        ]

        body = []
        if self._notice:
            body.append(
                pn.pane.Alert(
                    self._notice.get("message", ""),
                    alert_type=self._notice.get("type", "info"),
                    sizing_mode="stretch_width",
                    margin=(0, 0, 12, 0),
                )
            )

        pending_sections = self._build_pending_sections()
        if pending_sections:
            body.extend(pending_sections)
        else:
            body.append(
                pn.pane.Alert(
                    "No outstanding column mappings.",
                    alert_type="success",
                    margin=(0, 0, 16, 0),
                )
            )

        body.extend(self._build_current_mapping_sections())
        body.append(pn.Spacer(height=10))
        self.modal_body[:] = body
        self.modal_footer[:] = [self._build_action_row()]

    def _section_header(self, dataset_id: str) -> list[Any]:
        return [
            pn.pane.HTML(
                f"""
<div style="font-weight:700;color:#0f172a;font-size:1.02rem;line-height:1.3;margin:0;">{escape(self._dataset_name(dataset_id))}</div>
<div style="color:#94a3b8;font-size:0.85rem;line-height:1.35;margin:3px 0 12px 0;">{escape(str(dataset_id))}</div>
""",
                margin=(0, 0, 0, 0),
                sizing_mode="stretch_width",
            )
        ]

    def _build_pending_sections(self) -> list[Any]:
        if not self._pending:
            return []

        grouped: Dict[str, List[Tuple[PendingKey, dict]]] = {}
        for key, item in self._pending.items():
            grouped.setdefault(item["dataset_id"], []).append((key, item))

        blocks = []
        for dataset_id, items in grouped.items():
            section = self._section_header(dataset_id)
            for key, item in items:
                current = self.context.datasets.get_mapping(dataset_id, item["semantic_name"])
                selected = self._pending_values.get(key, "")
                options = self._select_options(
                    list(item.get("candidates", [])),
                    current=current,
                    selected=selected,
                )
                if selected not in (None, ""):
                    value = str(selected)
                elif current not in (None, ""):
                    value = str(current)
                else:
                    value = ""
                if value not in options:
                    options.insert(1, value)

                selector = pn.widgets.Select(
                    name="",
                    options=options,
                    value=value,
                    sizing_mode="stretch_width",
                    margin=(0, 12, 0, 0),
                )
                self._selectors[key] = selector
                section.append(self._build_pending_card(key, item, selector))

            blocks.append(
                pn.Column(
                    *section,
                    styles=DATASET_SECTION_STYLES,
                    sizing_mode="stretch_width",
                    margin=(0, 0, 16, 0),
                )
            )
        return blocks

    def _pending_card_html(self, item: dict) -> str:
        badge = BADGE_REQUIRED if item.get("required", True) else BADGE_OPTIONAL
        title = escape(str(item.get("display_name") or item["semantic_name"]))
        semantic_name = escape(str(item["semantic_name"]))
        description = escape(str(item.get("description", "") or ""))
        requested_by = escape(self._sources_text(item))
        suggested = item.get("suggested")

        description_html = ""
        if description:
            description_html = f"""
<div style="color:#475569;font-size:0.9rem;line-height:1.4;margin-top:10px;">{description}</div>
"""

        suggested_html = ""
        if suggested not in (None, ""):
            suggested_html = f"""
<div style="color:#64748b;font-size:0.86rem;line-height:1.35;margin-top:5px;">Suggested: <span style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;color:#2563eb;">{escape(str(suggested))}</span></div>
"""

        return f"""
<div style="box-sizing:border-box;width:100%;">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;box-sizing:border-box;width:100%;">
    <div style="min-width:0;box-sizing:border-box;">
      <div style="font-weight:700;color:#0f172a;font-size:1rem;line-height:1.3;">{title}</div>
      <div style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;font-size:0.82rem;color:#64748b;line-height:1.35;margin-top:3px;">{semantic_name}</div>
    </div>
    <div style="flex:0 0 auto;line-height:1.2;">{badge}</div>
  </div>
  {description_html}
  <div style="color:#475569;font-size:0.9rem;line-height:1.35;margin-top:9px;">Requested by: <span style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;color:#2563eb;">{requested_by}</span></div>
  {suggested_html}
</div>
"""

    def _build_pending_card(self, key: PendingKey, item: dict, selector: pn.widgets.Select):
        apply_button = pn.widgets.Button(
            name="✓ Apply",
            button_type="success",
            width=105,
            height=36,
            margin=(0, 0, 0, 0),
        )
        apply_button.on_click(lambda _event, key=key: self._apply_pending_mapping(key))

        return pn.Column(
            pn.pane.HTML(
                self._pending_card_html(item),
                margin=(0, 0, 12, 0),
                sizing_mode="stretch_width",
            ),
            pn.pane.HTML(
                '<div style="font-weight:600;color:#334155;font-size:0.9rem;line-height:1.3;">Choose dataset column</div>',
                margin=(0, 0, 6, 0),
                sizing_mode="stretch_width",
            ),
            pn.Row(
                selector,
                apply_button,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            ),
            styles=FIELD_BLOCK_STYLES,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        )

    def _current_mapping_header(self) -> pn.pane.HTML:
        return pn.pane.HTML(
            """
<div style="border-top:1px solid #e5e7eb;margin:18px 0 12px 0;padding-top:16px;box-sizing:border-box;width:100%;">
  <div style="font-weight:700;color:#17202a;font-size:1.12rem;line-height:1.35;margin:0 0 6px 0;">Current mappings</div>
  <div style="color:#475569;font-size:0.92rem;line-height:1.4;margin:0;">
    Review, edit, or clear mappings that have already been applied. Blank values are ignored here; use <strong>Clear</strong> to remove a mapping.
  </div>
</div>
""",
            margin=(0, 0, 0, 0),
            sizing_mode="stretch_width",
        )

    def _build_current_mapping_sections(self) -> list[Any]:
        blocks = [self._current_mapping_header()]

        any_mappings = False
        for dataset_id in self._dataset_ids():
            try:
                mappings = dict(self.context.datasets.get_mappings(dataset_id) or {})
            except Exception:
                mappings = {}
            if not mappings:
                continue

            any_mappings = True
            section = self._section_header(dataset_id)
            columns = self._dataset_columns(dataset_id)
            for semantic_name, column_name in mappings.items():
                options = self._select_options(columns, current=column_name)
                value = str(column_name) if column_name not in (None, "") else ""
                if value and value not in options:
                    options.insert(1, value)
                selector = pn.widgets.Select(
                    name="",
                    options=options,
                    value=value,
                    sizing_mode="stretch_width",
                    margin=(0, 12, 0, 0),
                )
                key = (str(dataset_id), str(semantic_name))
                self._current_selectors[key] = selector
                section.append(
                    self._build_current_mapping_row(
                        str(dataset_id), str(semantic_name), str(column_name), selector
                    )
                )
            blocks.append(
                pn.Column(
                    *section,
                    styles=DATASET_SECTION_STYLES,
                    sizing_mode="stretch_width",
                    margin=(0, 0, 16, 0),
                )
            )

        if not any_mappings:
            blocks.append(
                pn.pane.Alert(
                    "No mappings have been applied yet.",
                    alert_type="info",
                    margin=(0, 0, 16, 0),
                )
            )
        return blocks

    def _build_current_mapping_row(
        self,
        dataset_id: str,
        semantic_name: str,
        column_name: str,
        selector: pn.widgets.Select,
    ):
        apply_button = pn.widgets.Button(
            name="Apply", button_type="primary", width=88, height=36, margin=(0, 8, 0, 0)
        )
        clear_button = pn.widgets.Button(
            name="Clear", button_type="warning", width=78, height=36, margin=(0, 0, 0, 0)
        )
        apply_button.on_click(
            lambda _event, dataset_id=dataset_id, semantic_name=semantic_name: self._apply_current_mapping(
                dataset_id, semantic_name
            )
        )
        clear_button.on_click(
            lambda _event, dataset_id=dataset_id, semantic_name=semantic_name: self._clear_current_mapping(
                dataset_id, semantic_name
            )
        )

        label = pn.pane.HTML(
            f"""
<div style="font-weight:700;color:#0f172a;font-size:0.92rem;line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{escape(str(semantic_name))}</div>
<div style="color:#64748b;font-size:0.82rem;line-height:1.35;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">Current: <span style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;color:#2563eb;">{escape(str(column_name))}</span></div>
""",
            width=240,
            height=46,
            margin=(0, 14, 0, 0),
        )
        return pn.Row(
            label,
            selector,
            apply_button,
            clear_button,
            styles=CURRENT_ROW_STYLES,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
            height=66,
        )

    def _close_modal(self, _event=None) -> None:
        if self._notice and self._notice.get("type") == "success":
            self._notice = None
        self.template.close_modal()

    def _build_action_row(self):
        close_button = pn.widgets.Button(
            name="Close", button_type="light", width=120, height=42, margin=(0, 12, 0, 0)
        )
        apply_all_button = pn.widgets.Button(
            name="Apply all selected",
            button_type="primary",
            width=180,
            height=42,
            margin=(0, 0, 0, 0),
            disabled=not bool(self._pending),
        )
        close_button.on_click(self._close_modal)
        apply_all_button.on_click(self._apply_mappings)
        return pn.Row(
            pn.layout.HSpacer(),
            close_button,
            apply_all_button,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

    # ------------------------------------------------------------------
    # Apply/edit/clear actions
    # ------------------------------------------------------------------

    def _apply_one_pending_key(self, key: PendingKey) -> bool:
        selector = self._selectors.get(key)
        item = self._pending.get(key)
        if selector is None or item is None:
            return False

        chosen = selector.value
        if chosen in (None, ""):
            return False

        dataset_id = item["dataset_id"]
        semantic_name = item["semantic_name"]
        old_value = self.context.datasets.get_mapping(dataset_id, semantic_name)
        changed = self.context.datasets.set_mapping(dataset_id, semantic_name, str(chosen))

        for config_key in item.get("config_keys", []):
            if config_key and getattr(self.context, "config", None) is not None:
                self.context.config.settings[config_key] = str(chosen)

        payload = {
            "source": "mapping_header",
            "dataset_id": dataset_id,
            "semantic_name": semantic_name,
            "column_name": str(chosen),
            "old_column_name": old_value,
            "required": bool(item.get("required", True)),
            "changed": bool(changed),
        }

        self._pending.pop(key, None)
        self._pending_values.pop(key, None)

        if getattr(self.context, "events", None) is not None:
            for source in item.get("sources", []) or ["unknown"]:
                resolved_payload = dict(payload)
                resolved_payload["source"] = source
                self.context.events.publish("mapping.resolved", resolved_payload)
            if changed:
                self.context.events.publish("dataset.mapping.updated", payload)

        return True

    def _apply_pending_mapping(self, key: PendingKey) -> None:
        self._capture_pending_selector_values()
        selector = self._selectors.get(key)
        item = self._pending.get(key)
        if selector is None or item is None:
            self._set_notice("That mapping is no longer awaiting resolution.", "warning")
            self._refresh_button()
            self._rebuild_modal()
            return

        chosen = selector.value
        if chosen in (None, ""):
            name = item.get("display_name") or item.get("semantic_name")
            self._set_notice(f"Choose a dataset column before applying {name}.", "warning")
            self._rebuild_modal()
            return

        semantic_name = str(item["semantic_name"])
        applied = self._apply_one_pending_key(key)
        if applied:
            self._set_notice(f"Applied {semantic_name} → {chosen}.", "success")
        else:
            self._set_notice(f"No mapping was applied for {semantic_name}.", "warning")
        self._refresh_button()
        self._rebuild_modal()

    def _apply_mappings(self, _event=None) -> None:
        self._capture_pending_selector_values()
        applied = []
        skipped_blank = 0

        for key in list(self._selectors.keys()):
            selector = self._selectors.get(key)
            item = self._pending.get(key)
            if selector is None or item is None:
                continue
            chosen = selector.value
            if chosen in (None, ""):
                skipped_blank += 1
                continue
            if self._apply_one_pending_key(key):
                applied.append((str(item["semantic_name"]), str(chosen)))

        if len(applied) == 1:
            semantic_name, column_name = applied[0]
            self._set_notice(f"Applied {semantic_name} → {column_name}.", "success")
        elif len(applied) > 1:
            self._set_notice(
                f"Applied {len(applied)} mappings. Blank rows were left awaiting mapping.",
                "success",
            )
        elif skipped_blank:
            self._set_notice(
                "No mappings were applied because all selected rows were blank.",
                "warning",
            )
        else:
            self._set_notice("No pending mappings were available to apply.", "info")

        self._refresh_button()
        self._rebuild_modal()

    def _apply_current_mapping(self, dataset_id: str, semantic_name: str) -> None:
        key = (str(dataset_id), str(semantic_name))
        selector = self._current_selectors.get(key)
        if selector is None:
            self._set_notice("That current mapping is no longer available.", "warning")
            self._refresh_button()
            self._rebuild_modal()
            return

        chosen = selector.value
        if chosen in (None, ""):
            self._set_notice(
                "Blank current-mapping values are ignored. Use Clear to remove a mapping.",
                "warning",
            )
            self._rebuild_modal()
            return

        old_value = self.context.datasets.get_mapping(dataset_id, semantic_name)
        changed = self.context.datasets.set_mapping(dataset_id, semantic_name, str(chosen))

        if getattr(self.context, "events", None) is not None and changed:
            self.context.events.publish(
                "dataset.mapping.updated",
                {
                    "source": "mapping_header",
                    "dataset_id": dataset_id,
                    "semantic_name": semantic_name,
                    "column_name": str(chosen),
                    "old_column_name": old_value,
                    "required": False,
                    "changed": True,
                },
            )

        if changed:
            self._set_notice(f"Updated {semantic_name} → {chosen}.", "success")
        else:
            self._set_notice(f"{semantic_name} is already mapped to {chosen}.", "info")
        self._refresh_button()
        self._rebuild_modal()

    def _clear_current_mapping(self, dataset_id: str, semantic_name: str) -> None:
        try:
            mappings = self.context.datasets.get_mappings(dataset_id)
        except Exception:
            mappings = {}

        if semantic_name not in mappings:
            self._set_notice(f"{semantic_name} was not mapped.", "info")
            self._refresh_button()
            self._rebuild_modal()
            return

        old_value = mappings.get(semantic_name)
        del mappings[semantic_name]

        if getattr(self.context, "events", None) is not None:
            self.context.events.publish(
                "dataset.mapping.updated",
                {
                    "source": "mapping_header",
                    "dataset_id": dataset_id,
                    "semantic_name": semantic_name,
                    "column_name": None,
                    "old_column_name": old_value,
                    "required": False,
                    "changed": True,
                    "cleared": True,
                },
            )

        self._set_notice(f"Cleared mapping for {semantic_name}.", "success")
        self._refresh_button()
        self._rebuild_modal()