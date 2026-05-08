from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Tuple

import panel as pn


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


class MappingAlertController:
    """
    UI-local state only.

    It listens to:
      - mapping.requested
      - mapping.resolved

    It writes mappings to DatasetManager and emits:
      - mapping.resolved
      - dataset.mapping_updated
    """

    def __init__(self, context, template) -> None:
        self.context = context
        self.template = template

        self._pending: Dict[Tuple[str, str, str], dict] = {}
        self._selectors: Dict[Tuple[str, str, str], pn.widgets.Select] = {}
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
            height=430,
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
            margin=(0, 0, 0, 0),
            styles={
                "border-top": "1px solid #e5e7eb",
                "padding": "12px 20px 16px 20px",
                "background": "transparent",
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

        self.template.modal[:] = [self.modal_root]

        if getattr(self.context, "events", None) is not None:
            self._subs.append(
                self.context.events.subscribe("mapping.requested", self._on_mapping_requested)
            )
            self._subs.append(
                self.context.events.subscribe("mapping.resolved", self._on_mapping_resolved)
            )
            self._subs.append(
                self.context.events.subscribe(
                    "mapping.open_requested",
                    self._on_mapping_open_requested,
                )
            )

        self._refresh_button()

    def _on_mapping_open_requested(self, _topic: str, _payload: Any) -> None:
        if self._pending:
            self._rebuild_modal()
            self.template.open_modal()

    def dispose(self) -> None:
        if getattr(self.context, "events", None) is None:
            return

        for sub in self._subs:
            try:
                self.context.events.unsubscribe(sub)
            except Exception:
                pass
        self._subs = []

    def _key_from_payload(self, payload: dict) -> Tuple[str, str, str]:
        return (
            payload["dataset_id"],
            payload.get("source", "unknown"),
            payload["semantic_name"],
        )

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

    def _on_mapping_requested(self, _topic: str, payload: Any) -> None:
        if not payload:
            return

        payload = dict(payload)

        dataset_id = payload.get("dataset_id")
        semantic_name = payload.get("semantic_name")
        if not dataset_id or not semantic_name:
            return

        existing = self.context.datasets.get_mapping(dataset_id, semantic_name)
        if existing is not None:
            return

        payload.setdefault("source", "unknown")
        payload.setdefault("required", True)
        payload.setdefault("candidates", [])
        payload.setdefault("display_name", semantic_name)
        payload.setdefault("description", "")
        payload.setdefault("config_key", None)
        payload.setdefault("suggested", None)

        self._pending[self._key_from_payload(payload)] = payload
        self._refresh_button()

    def _on_mapping_resolved(self, _topic: str, payload: Any) -> None:
        if not payload:
            return

        dataset_id = payload.get("dataset_id")
        semantic_name = payload.get("semantic_name")

        if not dataset_id or not semantic_name:
            return

        to_remove = []

        for key, item in self._pending.items():
            same_dataset = item["dataset_id"] == dataset_id
            same_semantic = item["semantic_name"] == semantic_name

            if same_dataset and same_semantic:
                to_remove.append(key)

        for key in to_remove:
            self._pending.pop(key, None)

        self._refresh_button()

    def _status_badge(self, item: dict):
        if item.get("required", True):
            html = """
            <span style="
                display:inline-block;
                background:#fff7ed;
                color:#c2410c;
                border:1px solid #fdba74;
                border-radius:999px;
                padding:3px 8px;
                font-size:0.76rem;
                font-weight:600;
                line-height:1.15;
            ">Required</span>
            """
        else:
            html = """
            <span style="
                display:inline-block;
                background:#eff6ff;
                color:#1d4ed8;
                border:1px solid #93c5fd;
                border-radius:999px;
                padding:3px 8px;
                font-size:0.76rem;
                font-weight:600;
                line-height:1.15;
            ">Optional</span>
            """

        return pn.pane.HTML(
            html,
            margin=(0, 0, 10, 0),
            sizing_mode="stretch_width",
        )

    def _build_requirement_block(self, item: dict, selector: pn.widgets.Select):
        title = pn.pane.HTML(
            f"<div style='font-size:0.98rem;font-weight:700;color:#0f172a;'>{item['display_name']}</div>",
            margin=(0, 0, 6, 0),
            sizing_mode="stretch_width",
        )

        description = pn.pane.HTML(
            f"<div style='color:#64748b;font-size:0.92rem;line-height:1.4;'>{item.get('description', '') or ''}</div>",
            margin=(0, 0, 10, 0),
            sizing_mode="stretch_width",
        )

        select_label = pn.pane.HTML(
            "<div style='color:#334155;font-size:0.88rem;font-weight:600;'>Choose dataset column</div>",
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
            select_label,
            selector,
            styles=FIELD_BLOCK_STYLES,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        )

    def _open_modal(self, _event=None) -> None:
        self._rebuild_modal()
        self.template.open_modal()

    def _rebuild_modal(self) -> None:
        self._selectors = {}

        header_blocks = [
            pn.pane.Markdown(
                "# Resolve Dataset Requirements",
                margin=(0, 0, 10, 0),
                sizing_mode="stretch_width",
            ),
            pn.pane.HTML(
                (
                    f"<div style='color:{INTRO_STYLES['color']};"
                    f"font-size:{INTRO_STYLES['font-size']};"
                    f"line-height:{INTRO_STYLES['line-height']};'>"
                    "Choose the dataset columns needed by currently active panels. "
                    "Required fields must be mapped before those panels can continue."
                    "</div>"
                ),
                margin=(0, 0, 16, 0),
                sizing_mode="stretch_width",
            ),
        ]

        self.modal_header[:] = header_blocks

        if not self._pending:
            self.modal_body[:] = [
                pn.pane.Alert(
                    "No outstanding column mappings.",
                    alert_type="success",
                    margin=(0, 0, 0, 0),
                )
            ]
            self.modal_footer[:] = []
            return

        grouped = defaultdict(list)
        for key, item in self._pending.items():
            grouped[item["dataset_id"]].append((key, item))

        body_blocks = []

        for dataset_id, items in grouped.items():
            dataset = self.context.datasets.get(dataset_id)

            section_blocks = [
                pn.pane.HTML(
                    f"<div style='font-size:1.05rem;font-weight:700;color:#0f172a;'>{dataset.name}</div>",
                    margin=(0, 0, 2, 0),
                    sizing_mode="stretch_width",
                ),
                pn.pane.HTML(
                    f"<div style='color:{DATASET_ID_STYLES['color']};font-size:{DATASET_ID_STYLES['font-size']};'>{dataset_id}</div>",
                    margin=(0, 0, 12, 0),
                    sizing_mode="stretch_width",
                ),
            ]

            by_source = defaultdict(list)
            for key, item in items:
                by_source[item.get("source", "unknown")].append((key, item))

            for source_name, source_items in by_source.items():
                section_blocks.append(
                    pn.pane.HTML(
                        (
                            "<div style='margin-bottom:12px;'>"
                            f"<span style='font-weight:600;color:{SOURCE_STYLES['color']};font-size:{SOURCE_STYLES['font-size']};'>Requested by:</span> "
                            f"<span style='color:#2563eb;font-family:monospace;'>{source_name}</span>"
                            "</div>"
                        ),
                        sizing_mode="stretch_width",
                    )
                )

                for key, item in source_items:
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

        body_blocks.append(pn.Spacer(height=4))

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

        self.modal_body[:] = body_blocks
        self.modal_footer[:] = [
            pn.layout.HSpacer(),
            close_button,
            apply_button,
        ]

    def _apply_mappings(self, _event=None) -> None:
        resolved_semantics = []

        for key, selector in self._selectors.items():
            item = self._pending.get(key)
            if item is None:
                continue

            chosen = selector.value
            if chosen in (None, ""):
                continue

            dataset_id = item["dataset_id"]
            semantic_name = item["semantic_name"]

            self.context.datasets.set_mapping(dataset_id, semantic_name, chosen)

            config_key = item.get("config_key")
            if config_key and getattr(self.context, "config", None) is not None:
                self.context.config.settings[config_key] = chosen

            payload = {
                "source": item.get("source"),
                "dataset_id": dataset_id,
                "semantic_name": semantic_name,
                "config_key": config_key,
                "column_name": chosen,
            }

            self.context.events.publish("mapping.resolved", payload)
            self.context.events.publish("dataset.mapping_updated", payload)

            resolved_semantics.append((dataset_id, semantic_name))

        for dataset_id, semantic_name in resolved_semantics:
            for key, item in list(self._pending.items()):
                if (
                    item.get("dataset_id") == dataset_id
                    and item.get("semantic_name") == semantic_name
                ):
                    self._pending.pop(key, None)

        self._refresh_button()
        self._rebuild_modal()

        if not self._pending:
            self.template.close_modal()