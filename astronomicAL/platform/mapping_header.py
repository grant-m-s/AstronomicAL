from __future__ import annotations

from html import escape
from typing import Any, Dict, List, Tuple

import panel as pn

from astronomicAL.platform.modal_utils import open_template_modal


SHEET_STYLES = {
    "background": "#ffffff",
    "border-radius": "16px",
    "padding": "0px",
    "border": "1px solid #e5e7eb",
    "box-shadow": "0 18px 50px rgba(15, 23, 42, 0.14)",
    "box-sizing": "border-box",
    "overflow": "hidden",
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
}

FIELD_BLOCK_STYLES = {
    "background": "#ffffff",
    "border-radius": "10px",
    "padding": "12px",
    "border": "1px solid #e5e7eb",
    "box-shadow": "0 1px 4px rgba(15, 23, 42, 0.04)",
    "box-sizing": "border-box",
}

DATASET_ID_STYLES = {
    "color": "#94a3b8",
    "font-size": "0.85rem",
}

SOURCE_STYLES = {
    "color": "#475569",
    "font-size": "0.9rem",
}

SOURCE_CODE_STYLES = {
    "font-family": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
    "font-size": "0.84rem",
    "color": "#2563eb",
}

BADGE_REQUIRED = """
<span style="
    display:inline-block;
    color:#c2410c;
    background:#fff7ed;
    border:1px solid #fdba74;
    border-radius:999px;
    padding:2px 9px;
    font-size:0.78rem;
    font-weight:600;
">
Required
</span>
"""

BADGE_OPTIONAL = """
<span style="
    display:inline-block;
    color:#2563eb;
    background:#eff6ff;
    border:1px solid #bfdbfe;
    border-radius:999px;
    padding:2px 9px;
    font-size:0.78rem;
    font-weight:600;
">
Optional
</span>
"""


PendingKey = Tuple[str, str]


class MappingAlertController:
    """UI-local mapping request controller.

    The controller listens to mapping requests from panels/plugins, collates
    duplicate semantic requirements, and writes the resolved mappings to the
    DatasetManager.

    Multiple active panels often request the same semantic column, for example
    ``record_id``. Those requests are collapsed into one modal block per:

        (dataset_id, semantic_name)

    Required requests outrank optional requests. If any active panel requires a
    semantic mapping, the combined modal row is treated as required.
    """

    def __init__(self, context, template) -> None:
        self.context = context
        self.template = template

        self._pending: Dict[PendingKey, dict] = {}
        self._selectors: Dict[PendingKey, pn.widgets.Select] = {}
        self._subs = []

        self.button = pn.widgets.Button(
            name="✅ 0",
            button_type="success",
            width=92,
            height=38,
            margin=(6, 8, 6, 8),
        )
        self.button.on_click(self._open_modal)

        self.view = pn.Row(
            pn.layout.HSpacer(),
            self.button,
            sizing_mode="stretch_width",
        )

        self.modal_root = pn.Column(
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.modal_header = pn.Column(
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                "padding": "20px 20px 0 20px",
                "box-sizing": "border-box",
            },
        )

        self.modal_body = pn.Column(
            sizing_mode="stretch_width",
            height=400,
            scroll=True,
            margin=(0, 0, 0, 0),
            styles={
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "padding": "0 20px 8px 20px",
                "box-sizing": "border-box",
            },
        )

        self.modal_footer = pn.Row(
            sizing_mode="stretch_width",
            height=0,
            max_height=0,
            margin=(0, 0, 0, 0),
            styles={
                "height": "0px",
                "max-height": "0px",
                "overflow": "hidden",
                "padding": "0px",
                "border": "0px",
                "box-sizing": "border-box",
            },
        )

        self.modal_sheet = pn.Column(
            self.modal_header,
            self.modal_body,
            self.modal_footer,
            sizing_mode="fixed",
            width=960,
            height=620,
            margin=(20, 0, 20, 0),
            styles=SHEET_STYLES,
        )

        self.modal_root[:] = [
            pn.Row(
                pn.layout.HSpacer(),
                self.modal_sheet,
                pn.layout.HSpacer(),
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )
        ]

        if getattr(self.context, "events", None) is not None:
            self._subs.append(
                self.context.events.subscribe(
                    "mapping.requested",
                    self._on_mapping_requested,
                )
            )
            self._subs.append(
                self.context.events.subscribe(
                    "mapping.resolved",
                    self._on_mapping_resolved,
                )
            )
            self._subs.append(
                self.context.events.subscribe(
                    "mapping.open_requested",
                    self._on_mapping_open_requested,
                )
            )

        self._refresh_button()

    def dispose(self) -> None:
        if getattr(self.context, "events", None) is None:
            return

        for sub in self._subs:
            try:
                self.context.events.unsubscribe(sub)
            except Exception:
                pass

        self._subs = []

    def _key_from_payload(self, payload: dict) -> PendingKey:
        return (
            str(payload["dataset_id"]),
            str(payload["semantic_name"]),
        )

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

        source = str(payload.get("source") or "unknown")
        panel_id = payload.get("panel_id")
        config_key = payload.get("config_key")

        payload["sources"] = [source]
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

        # Required metadata should win over optional metadata because it is the
        # blocking version of the requirement.
        if incoming_is_required and not existing_was_required:
            existing["display_name"] = incoming.get("display_name") or existing.get("display_name")
            existing["description"] = incoming.get("description") or existing.get("description")
            existing["suggested"] = incoming.get("suggested") or existing.get("suggested")
            existing["config_key"] = incoming.get("config_key") or existing.get("config_key")
            return existing

        if not existing.get("display_name"):
            existing["display_name"] = incoming.get("display_name")

        if not existing.get("description") and incoming.get("description"):
            existing["description"] = incoming.get("description")

        if not existing.get("suggested") and incoming.get("suggested"):
            existing["suggested"] = incoming.get("suggested")

        if not existing.get("config_key") and incoming.get("config_key"):
            existing["config_key"] = incoming.get("config_key")

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

    def _refresh_button(self) -> None:
        total = len(self._pending)
        required = self._required_count()

        if total == 0:
            self.button.name = "✅ 0"
            self.button.button_type = "success"
            self.button.disabled = True
            return

        self.button.disabled = False

        if required > 0:
            self.button.name = f"⚠ {required}"
            self.button.button_type = "danger"
        else:
            self.button.name = f"ℹ {total}"
            self.button.button_type = "primary"

    def _on_mapping_open_requested(self, _topic: str, _payload: Any) -> None:
        if self._pending:
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
            return

        key = self._key_from_payload(payload)
        existing = self._pending.get(key)

        if existing is None:
            self._pending[key] = payload
        else:
            self._pending[key] = self._merge_pending_item(existing, payload)

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
        self._refresh_button()

    def _status_badge(self, item: dict):
        html = BADGE_REQUIRED if item.get("required", True) else BADGE_OPTIONAL
        return pn.pane.HTML(
            html,
            margin=(0, 0, 8, 0),
            sizing_mode="stretch_width",
        )

    def _sources_text(self, item: dict) -> str:
        sources = [str(source) for source in item.get("sources", []) if source]

        if not sources:
            return "unknown"

        if len(sources) <= 4:
            return ", ".join(sources)

        shown = ", ".join(sources[:4])
        return f"{shown}, +{len(sources) - 4} more"

    def _build_requirement_block(self, item: dict, selector: pn.widgets.Select):
        title = pn.pane.HTML(
            f"""
            <div style="font-weight:700;font-size:0.98rem;color:#0f172a;">
                {escape(str(item.get("display_name") or item["semantic_name"]))}
            </div>
            """,
            margin=(0, 0, 6, 0),
            sizing_mode="stretch_width",
        )

        description_text = str(item.get("description", "") or "")
        description = pn.pane.HTML(
            f"""
            <div style="color:#64748b;font-size:0.88rem;line-height:1.35;">
                {escape(description_text)}
            </div>
            """,
            margin=(0, 0, 8, 0),
            sizing_mode="stretch_width",
            visible=bool(description_text),
        )

        requested_by = pn.pane.HTML(
            f"""
            <div style="color:#475569;font-size:0.86rem;margin-bottom:10px;">
                Requested by:
                <span style="
                    font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
                    color:#2563eb;
                    font-size:0.83rem;
                ">
                    {escape(self._sources_text(item))}
                </span>
            </div>
            """,
            margin=(0, 0, 6, 0),
            sizing_mode="stretch_width",
        )

        select_label = pn.pane.HTML(
            """
            <div style="font-weight:600;color:#1e293b;font-size:0.88rem;">
                Choose dataset column
            </div>
            """,
            margin=(0, 0, 6, 0),
            sizing_mode="stretch_width",
        )

        selector.name = ""
        selector.sizing_mode = "stretch_width"
        selector.margin = (0, 0, 0, 0)

        return pn.Column(
            title,
            self._status_badge(item),
            description,
            requested_by,
            select_label,
            selector,
            styles=FIELD_BLOCK_STYLES,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        )

    def _open_modal(self, _event=None) -> None:
        self._rebuild_modal()
        open_template_modal(self.template, self.modal_root)

    def _rebuild_modal(self) -> None:
        self._selectors = {}

        self.modal_header[:] = [
            pn.pane.Markdown(
                "# Resolve Dataset Requirements",
                margin=(0, 0, 10, 0),
                sizing_mode="stretch_width",
            ),
            pn.pane.HTML(
                """
                <p>
                    Choose the dataset columns needed by currently active panels.
                    Duplicate requests are combined, and required requests take
                    priority over optional requests.
                </p>
                """,
                margin=(0, 0, 16, 0),
                sizing_mode="stretch_width",
                styles=INTRO_STYLES,
            ),
        ]

        # Do not use a fixed footer for the apply/close buttons. In smaller
        # browser windows the template modal can clip the footer, making the
        # buttons unreachable. Instead, include the action row at the bottom of
        # the scrollable modal body.
        self.modal_footer[:] = []

        if not self._pending:
            self.modal_body[:] = [
                pn.pane.Alert(
                    "No outstanding column mappings.",
                    alert_type="success",
                    margin=(0, 0, 0, 0),
                )
            ]
            return

        grouped: Dict[str, List[Tuple[PendingKey, dict]]] = {}
        for key, item in self._pending.items():
            grouped.setdefault(item["dataset_id"], []).append((key, item))

        body_blocks = []

        for dataset_id, items in grouped.items():
            try:
                dataset = self.context.datasets.get(dataset_id)
                dataset_name = getattr(dataset, "name", dataset_id)
            except Exception:
                dataset_name = dataset_id

            section_blocks = [
                pn.pane.HTML(
                    f"""
                    <div style="font-weight:700;color:#0f172a;font-size:1rem;">
                        {escape(str(dataset_name))}
                    </div>
                    """,
                    margin=(0, 0, 2, 0),
                    sizing_mode="stretch_width",
                ),
                pn.pane.HTML(
                    f"""
                    <div style="color:#94a3b8;font-size:0.85rem;">
                        {escape(str(dataset_id))}
                    </div>
                    """,
                    margin=(0, 0, 12, 0),
                    sizing_mode="stretch_width",
                ),
            ]

            sorted_items = sorted(
                items,
                key=lambda pair: (
                    not bool(pair[1].get("required", True)),
                    str(pair[1].get("display_name") or pair[1].get("semantic_name", "")).casefold(),
                ),
            )

            for key, item in sorted_items:
                current = self.context.datasets.get_mapping(
                    dataset_id,
                    item["semantic_name"],
                )

                default_value = current or item.get("suggested")

                options = list(item.get("candidates", []))
                if default_value is not None and default_value not in options:
                    options = [default_value] + options

                if not options:
                    options = [""]

                selector = pn.widgets.Select(
                    options=options,
                    value=default_value if default_value in options else options[0],
                    sizing_mode="stretch_width",
                    margin=(0, 0, 0, 0),
                )

                self._selectors[key] = selector
                section_blocks.append(self._build_requirement_block(item, selector))

            body_blocks.append(
                pn.Column(
                    *section_blocks,
                    styles=DATASET_SECTION_STYLES,
                    sizing_mode="stretch_width",
                    margin=(0, 0, 16, 0),
                )
            )

        apply_button = pn.widgets.Button(
            name="Apply mappings",
            button_type="primary",
            width=170,
            height=42,
            margin=(0, 0, 0, 0),
        )

        close_button = pn.widgets.Button(
            name="Close",
            button_type="light",
            width=120,
            height=42,
            margin=(0, 12, 0, 0),
        )

        apply_button.on_click(self._apply_mappings)
        close_button.on_click(lambda _e: self.template.close_modal())

        action_row = pn.Row(
            pn.layout.HSpacer(),
            close_button,
            apply_button,
            sizing_mode="stretch_width",
            margin=(8, 0, 0, 0),
            styles={
                "border-top": "1px solid #e5e7eb",
                "padding": "14px 0 4px 0",
                "background": "#ffffff",
                "box-sizing": "border-box",
            },
        )

        body_blocks.append(action_row)
        body_blocks.append(pn.Spacer(height=8))

        self.modal_body[:] = body_blocks

    def _apply_mappings(self, _event=None) -> None:
        resolved_keys: List[PendingKey] = []

        for key, selector in self._selectors.items():
            item = self._pending.get(key)
            if item is None:
                continue

            chosen = selector.value
            if chosen in (None, ""):
                continue

            dataset_id = item["dataset_id"]
            semantic_name = item["semantic_name"]

            # self.context.datasets.set_mapping(dataset_id, semantic_name, chosen)

            old_value = self.context.datasets.get_mapping(dataset_id, semantic_name)
            changed = self.context.datasets.set_mapping(dataset_id, semantic_name, chosen)

            for config_key in item.get("config_keys", []):
                if config_key and getattr(self.context, "config", None) is not None:
                    self.context.config.settings[config_key] = chosen

            dataset_payload = {
                "source": "mapping_header",
                "dataset_id": dataset_id,
                "semantic_name": semantic_name,
                "column_name": chosen,
                "old_column_name": old_value,
                "required": bool(item.get("required", True)),
                "changed": True,
            }


            # Always clear the pending mapping row.
            resolved_keys.append(key)

            # If unchanged, do not fan out a dataset-level update.
            if not changed:
                continue

            for source in item.get("sources", []) or ["unknown"]:
                payload = dict(dataset_payload)
                payload["source"] = source
                self.context.events.publish("mapping.resolved", payload)

            self.context.events.publish("dataset.mapping_updated", dataset_payload)

        for key in resolved_keys:
            self._pending.pop(key, None)

        self._refresh_button()
        self._rebuild_modal()

        if not self._pending:
            self.template.close_modal()