from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import panel as pn

from astronomicAL.platform.plugins import PluginManifest


manifest = PluginManifest(
    id="core.event_monitor",
    name="Event Monitor",
    version="0.2.0",
    description=(
        "Diagnostics panel for EventBus subscriptions, recent events, "
        "dataset-event coverage, and owner/topic relationships."
    ),
    capabilities=["panel", "diagnostics", "events", "subscriptions"],
    tags=["core", "debug", "events", "observability"],
)


def register(api) -> None:
    api.register_panel(
        id="panel",
        title="Event Monitor",
        factory=create_event_monitor_panel,
        description=(
            "Inspect EventBus subscriptions, dataset-event coverage, "
            "recent events, and owner/topic graph structure."
        ),
        category="Diagnostics",
        icon="activity",
        tags=["debug", "events", "subscriptions"],
        default_layout={"x": 0, "y": 0, "w": 8, "h": 6},
    )


def create_event_monitor_panel(context, **kwargs):
    controller = EventMonitorPanel(context=context)
    return controller.view, controller


class EventMonitorPanel:
    """Plugin version of the context-core EventMonitorClass.

    It intentionally does not subclass CustomPlotClass. Instead, it follows the
    plugin contract directly:

    - receives context
    - reads EventBus diagnostics through context.events
    - renders a Panel view
    - cleans periodic callbacks and watchers in dispose()
    """

    DATASET_TOPICS = [
        "dataset.loaded",
        "dataset.active.changed",
        "dataset.updated",
        "dataset.mapping_updated",
    ]

    TOPIC_FILTER_OPTIONS = [
        "*",
        "dataset.",
        "selection.",
        "artifact.",
        "labels.",
        "review.",
        "workflow.",
        "plugin.",
    ]

    def __init__(self, context):
        self.context = context
        self._disposed = False
        self._watchers: list[tuple[Any, Any]] = []

        events = getattr(self.context, "events", None)
        if events is not None:
            try:
                events.enable_trace(True)
            except Exception:
                pass

        self.refresh_btn = pn.widgets.Button(
            name="Refresh",
            button_type="primary",
            width=100,
            height=32,
        )

        self.auto_refresh_toggle = pn.widgets.Checkbox(
            name="Auto refresh",
            value=True,
            width=120,
        )

        self.trace_toggle = pn.widgets.Checkbox(
            name="Trace enabled",
            value=True,
            width=120,
        )

        self.follow_toggle = pn.widgets.Checkbox(
            name="Follow newest",
            value=False,
            width=130,
        )

        self.limit_input = pn.widgets.Select(
            name="Rows",
            options=[100, 250, 500, 1000, 2000, 5000],
            value=250,
            width=110,
        )

        self.topic_filter = pn.widgets.Select(
            name="Graph topic filter",
            options=self.TOPIC_FILTER_OPTIONS,
            value="dataset.",
            width=150,
        )

        self.show_wildcards_toggle = pn.widgets.Checkbox(
            name="Include wildcard (*)",
            value=True,
            width=150,
        )

        self.show_orphans_toggle = pn.widgets.Checkbox(
            name="Show owners with no edges",
            value=True,
            width=190,
        )

        self.publish_test_btn = pn.widgets.Button(
            name="Publish test event",
            button_type="default",
            width=150,
            height=32,
        )

        self.status = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
        )

        self.graph_pane = pn.pane.Bokeh(
            sizing_mode="stretch_width",
            min_height=500,
            margin=(0, 0, 0, 0),
        )

        self.coverage_table = pn.widgets.Tabulator(
            pd.DataFrame(
                columns=[
                    "owner_label",
                    "owner_kind",
                    "wildcard",
                    "dataset.loaded",
                    "dataset.active.changed",
                    "dataset.updated",
                    "dataset.mapping_updated",
                    "status",
                    "topics",
                ]
            ),
            height=500,
            sizing_mode="stretch_width",
            disabled=True,
            pagination="local",
            page_size=25,
        )

        self.subs_table = pn.widgets.Tabulator(
            pd.DataFrame(
                columns=[
                    "owner_label",
                    "owner_kind",
                    "topic",
                    "callback_name",
                    "module",
                    "owner_id",
                ]
            ),
            height=500,
            sizing_mode="stretch_width",
            disabled=True,
            pagination="local",
            page_size=25,
        )

        self.events_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["time", "topic", "payload"]),
            height=500,
            sizing_mode="stretch_width",
            disabled=True,
            pagination="local",
            page_size=25,
        )

        self.refresh_btn.on_click(lambda _event: self.refresh())
        self.publish_test_btn.on_click(self._publish_test_event)

        for widget in [
            self.auto_refresh_toggle,
            self.trace_toggle,
            self.follow_toggle,
            self.limit_input,
            self.topic_filter,
            self.show_wildcards_toggle,
            self.show_orphans_toggle,
        ]:
            self._watch(widget, self._controls_changed, "value")

        self._periodic_callback = pn.state.add_periodic_callback(
            self._periodic_refresh,
            period=2000,
            start=True,
        )

        self.view = self._build_layout()
        self.refresh()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):

        row_1 = pn.Row(
            self.refresh_btn,
            pn.Spacer(width=10),
            self.publish_test_btn,
            pn.Spacer(width=18),
            self.auto_refresh_toggle,
            pn.Spacer(width=18),
            self.trace_toggle,
            sizing_mode="stretch_width",
            height=42,
            align="center",
        )

        filter_group = pn.Column(
            self.topic_filter,
            pn.Spacer(height=12),
            self.show_wildcards_toggle,
            width=165,
            height=88,
            margin=(0, 12, 0, 0),
        )

        rows_group = pn.Column(
            self.limit_input,
            pn.Spacer(height=12),
            self.show_orphans_toggle,
            width=180,
            height=88,
            margin=(0, 12, 0, 0),
        )

        follow_group = pn.Column(
            pn.Spacer(height=31),
            self.follow_toggle,
            width=145,
            height=88,
            margin=(0, 0, 0, 0),
        )

        row_2 = pn.Row(
            filter_group,
            rows_group,
            follow_group,
            pn.Spacer(sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            height=88,
            align="start",
        )

        tabs = pn.Tabs(
            (
                "Graph",
                pn.Column(
                    self.graph_pane,
                    sizing_mode="stretch_width",
                    height=560,
                    scroll=False,
                ),
            ),
            (
                "Coverage",
                pn.Column(
                    self.coverage_table,
                    sizing_mode="stretch_width",
                    height=560,
                    scroll=False,
                ),
            ),
            (
                "Subscriptions",
                pn.Column(
                    self.subs_table,
                    sizing_mode="stretch_width",
                    height=560,
                    scroll=False,
                ),
            ),
            (
                "Events",
                pn.Column(
                    self.events_table,
                    sizing_mode="stretch_width",
                    height=560,
                    scroll=False,
                ),
            ),
            dynamic=True,
            sizing_mode="stretch_width",
            height=600,
        )

        title = pn.pane.HTML(
            "<h2 style='margin: 0; padding: 0; line-height: 1.2;'>Event Monitor</h2>",
            height=34,
            min_height=34,
            max_height=34,
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        return pn.Column(
            title,
            row_1,
            row_2,
            self.status,
            tabs,
            sizing_mode="stretch_both",
            scroll=True,
            margin=(0, 0, 0, 0),
            styles={
                "overflow": "hidden",
                "padding": "0 8px 8px 8px",
            },
        )

    # ------------------------------------------------------------------
    # Reactive helpers
    # ------------------------------------------------------------------

    def _watch(self, widget, callback, attr: str) -> None:
        try:
            watcher = widget.param.watch(callback, attr)
            self._watchers.append((widget, watcher))
        except Exception:
            pass

    def _controls_changed(self, _event=None) -> None:
        self.refresh()

    def _periodic_refresh(self) -> None:
        if self._disposed:
            return
        if bool(self.auto_refresh_toggle.value):
            self.refresh()

    def _publish_test_event(self, _event=None) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        events.publish(
            "debug.event_monitor.test",
            {
                "message": "Test event from Event Monitor plugin.",
                "plugin_id": manifest.id,
            },
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _short_text(value: Any, max_len: int = 28) -> str:
        text = "" if value is None else str(value)
        return text if len(text) <= max_len else text[: max_len - 1] + "…"

    @staticmethod
    def _payload_to_str(payload: Any, max_len: int = 320) -> str:
        try:
            if isinstance(payload, (dict, list, tuple)):
                text = json.dumps(payload, default=str)
            else:
                text = str(payload)
        except Exception:
            text = repr(payload)

        text = text.replace("\n", " ").replace("\r", " ")
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    @staticmethod
    def _normalise_df_for_compare(df: Any) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception:
                return pd.DataFrame()

        out = df.copy().reset_index(drop=True)
        out.columns = [str(c) for c in out.columns]
        for col in out.columns:
            out[col] = out[col].map(lambda x: "" if x is None else str(x))
        return out

    def _df_changed(self, old_df: Any, new_df: Any) -> bool:
        old_norm = self._normalise_df_for_compare(old_df)
        new_norm = self._normalise_df_for_compare(new_df)

        if list(old_norm.columns) != list(new_norm.columns):
            return True
        if old_norm.shape != new_norm.shape:
            return True

        return not old_norm.equals(new_norm)

    def _set_tabulator_df_if_changed(self, widget, df: pd.DataFrame) -> None:
        current = getattr(widget, "value", None)
        if self._df_changed(current, df):
            widget.value = df

    @staticmethod
    def _value(info: Any, name: str, default: Any = None) -> Any:
        return getattr(info, name, default)

    @staticmethod
    def _pretty_owner_label(value: Any) -> str:
        text = "" if value is None else str(value)
        if ".." in text:
            text = text.replace("..", "::")
        if "::" in text:
            left, right = text.split("::", 1)
            left = left.split(".")[0]
            right = right.split(".")[-1]
            text = f"{left}::{right}"
        return text or ""

    @staticmethod
    def _wrap_label(text: Any, max_line_len: int = 16, max_lines: int = 4) -> str:
        text = "" if text is None else str(text).strip()
        if not text:
            return ""

        text = text.replace("..", "::")
        raw_parts = re.split(r"(::|\.|_|-)", text)

        chunks: list[str] = []
        for part in raw_parts:
            if not part:
                continue
            if part in {"::", ".", "_", "-"}:
                if chunks:
                    chunks[-1] = chunks[-1] + part
                else:
                    chunks.append(part)
                continue

            camel_parts = re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+",
                part,
            )
            chunks.extend(camel_parts or [part])

        def pack(width: int) -> list[str]:
            lines: list[str] = []
            current = ""

            for chunk in chunks:
                if not current:
                    if len(chunk) <= width:
                        current = chunk
                    else:
                        while len(chunk) > width:
                            lines.append(chunk[:width])
                            chunk = chunk[width:]
                        current = chunk
                    continue

                if len(current) + len(chunk) <= width:
                    current += chunk
                else:
                    lines.append(current.rstrip())
                    current = chunk

            if current:
                lines.append(current.rstrip())

            return [line for line in lines if line]

        width = max(10, int(max_line_len))
        lines = pack(width)
        while len(lines) > max_lines and width < 30:
            width += 2
            lines = pack(width)

        return "\n".join(lines[:max_lines])

    @staticmethod
    def _rows_to_cds_data(rows: Sequence[Dict[str, Any]], columns: Optional[List[str]] = None):
        if not rows:
            return {col: [] for col in (columns or [])}

        if columns is None:
            columns = []
            seen = set()
            for row in rows:
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        columns.append(key)

        return {
            col: [row.get(col) for row in rows]
            for col in columns
        }

    # ------------------------------------------------------------------
    # Data gathering
    # ------------------------------------------------------------------

    def _get_subscription_infos(self) -> list[Any]:
        events = getattr(self.context, "events", None)
        if events is None:
            return []

        method = getattr(events, "list_subscriptions", None)
        if not callable(method):
            return []

        try:
            return list(method())
        except Exception:
            return []

    def _topic_matches_filter(self, topic: str, prefix: str) -> bool:
        topic = str(topic or "")
        if topic == "*":
            return bool(self.show_wildcards_toggle.value)
        if prefix in ("", "*", None):
            return True
        return topic.startswith(str(prefix))

    def _owner_key(self, info: Any) -> str:
        owner_id = self._value(info, "owner_id")
        if owner_id:
            return str(owner_id)

        parts = [
            str(self._value(info, "owner_label", "unknown")),
            str(self._value(info, "callback_name", "callback")),
            str(self._value(info, "id", "")),
        ]
        return "::".join(parts)

    def _owner_label(self, info: Any) -> str:
        raw = (
            self._value(info, "owner_label")
            or self._value(info, "owner_id")
            or self._value(info, "callback_name")
            or ""
        )
        return self._pretty_owner_label(raw)

    def _build_owner_groups(self, sub_infos: Iterable[Any]) -> dict[str, dict[str, Any]]:
        owners: dict[str, dict[str, Any]] = {}

        for info in sub_infos:
            key = self._owner_key(info)
            record = owners.setdefault(
                key,
                {
                    "owner_id": self._value(info, "owner_id"),
                    "owner_label": self._owner_label(info),
                    "owner_kind": self._value(info, "owner_kind", "subscriber") or "subscriber",
                    "subscriptions": [],
                    "topics": set(),
                },
            )
            record["subscriptions"].append(info)
            record["topics"].add(self._value(info, "topic", ""))

        return owners

    def _build_diagnostics_df(self, owners: dict[str, dict[str, Any]]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for owner in owners.values():
            topics = set(owner["topics"])
            wildcard = "*" in topics

            row = {
                "owner_label": owner["owner_label"],
                "owner_kind": owner["owner_kind"],
                "wildcard": "yes" if wildcard else "no",
                "dataset.loaded": "yes" if wildcard or "dataset.loaded" in topics else "no",
                "dataset.active.changed": (
                    "yes" if wildcard or "dataset.active.changed" in topics else "no"
                ),
                "dataset.updated": "yes" if wildcard or "dataset.updated" in topics else "no",
                "dataset.mapping_updated": (
                    "yes" if wildcard or "dataset.mapping_updated" in topics else "no"
                ),
                "topics": ", ".join(sorted(str(t) for t in topics))[:240],
            }

            dataset_any = wildcard or any(str(t).startswith("dataset.") for t in topics)

            if wildcard or all(t in topics for t in self.DATASET_TOPICS):
                row["status"] = "good"
            elif dataset_any:
                row["status"] = "partial"
            else:
                row["status"] = "missing dataset subscriptions"

            rows.append(row)

        columns = [
            "owner_label",
            "owner_kind",
            "wildcard",
            "dataset.loaded",
            "dataset.active.changed",
            "dataset.updated",
            "dataset.mapping_updated",
            "status",
            "topics",
        ]

        if not rows:
            return pd.DataFrame(columns=columns)

        status_order = {
            "missing dataset subscriptions": 0,
            "partial": 1,
            "good": 2,
        }

        df = pd.DataFrame(rows)
        df["_status_order"] = df["status"].map(status_order).fillna(99)
        df = (
            df.sort_values(
                by=["_status_order", "owner_kind", "owner_label"],
                ascending=[True, True, True],
            )
            .drop(columns=["_status_order"])
            .reset_index(drop=True)
        )
        return df[columns]

    def _build_subscription_df(self, sub_infos: Sequence[Any]) -> pd.DataFrame:
        rows = []

        for info in sub_infos:
            rows.append(
                {
                    "owner_label": self._owner_label(info),
                    "owner_kind": self._value(info, "owner_kind", "subscriber") or "subscriber",
                    "topic": self._value(info, "topic"),
                    "callback_name": self._value(info, "callback_name"),
                    "module": self._value(info, "module"),
                    "owner_id": self._value(info, "owner_id"),
                }
            )

        columns = [
            "owner_label",
            "owner_kind",
            "topic",
            "callback_name",
            "module",
            "owner_id",
        ]

        if not rows:
            return pd.DataFrame(columns=columns)

        return (
            pd.DataFrame(rows)
            .sort_values(
                by=["owner_kind", "owner_label", "topic", "callback_name"],
                ascending=[True, True, True, True],
            )
            .reset_index(drop=True)
        )

    def _build_events_df(self, prefix: str, n: int) -> pd.DataFrame:
        events = getattr(self.context, "events", None)
        if events is None:
            return pd.DataFrame(columns=["time", "topic", "payload"])

        recent = getattr(events, "recent_events", None)
        if not callable(recent):
            return pd.DataFrame(columns=["time", "topic", "payload"])

        try:
            event_rows = recent(n)
        except Exception:
            return pd.DataFrame(columns=["time", "topic", "payload"])

        rows = []
        for t, topic, payload in event_rows:
            if prefix not in ("", "*", None) and not str(topic).startswith(str(prefix)):
                continue

            rows.append(
                {
                    "time": time.strftime("%H:%M:%S", time.localtime(t)),
                    "topic": topic,
                    "payload": self._payload_to_str(payload),
                }
            )

        if not bool(self.follow_toggle.value):
            rows.reverse()

        if not rows:
            return pd.DataFrame(columns=["time", "topic", "payload"])

        return pd.DataFrame(rows).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Graph building
    # ------------------------------------------------------------------

    def _build_graph_figure(
        self,
        owners: dict[str, dict[str, Any]],
        matching_subs: Sequence[Any],
        prefix: str,
    ):
        from bokeh.models import ColumnDataSource, HoverTool, LabelSet
        from bokeh.plotting import figure

        owner_keys_with_edges = {self._owner_key(info) for info in matching_subs}

        owner_items = []
        for key, owner in owners.items():
            if not self.show_orphans_toggle.value and key not in owner_keys_with_edges:
                continue
            owner_items.append((key, owner))

        owner_items = sorted(owner_items, key=lambda kv: str(kv[1]["owner_label"]).lower())
        topics = sorted({str(self._value(info, "topic", "")) for info in matching_subs})

        def spread(items: Sequence[Any], step: float = 1.0) -> dict[Any, float]:
            if not items:
                return {}
            if len(items) == 1:
                return {items[0]: 0.0}
            half_span = ((len(items) - 1) * step) / 2.0
            return {item: half_span - i * step for i, item in enumerate(items)}

        owner_positions = spread([key for key, _owner in owner_items])
        topic_positions = spread(topics)

        topic_counts = {}
        for info in matching_subs:
            topic = str(self._value(info, "topic", ""))
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        owner_label_x = 0.45
        owner_node_x = 1.25
        topic_node_x = 2.25
        topic_label_x = 3.05
        total_width = 3.6

        owner_label_rows = []
        owner_node_rows = []
        orphan_node_rows = []

        for key, owner in owner_items:
            y = owner_positions[key]
            owner_label_rows.append(
                {
                    "x": owner_label_x,
                    "y": y,
                    "label": self._wrap_label(owner["owner_label"]),
                    "full_label": owner["owner_label"],
                }
            )

            node_row = {
                "x": owner_node_x,
                "y": y,
                "full_label": owner["owner_label"],
                "owner_id": owner["owner_id"],
                "owner_kind": owner["owner_kind"],
                "subscription_count": len(owner["subscriptions"]),
                "dataset_topic_count": len(
                    [t for t in owner["topics"] if str(t).startswith("dataset.")]
                ),
            }

            if key in owner_keys_with_edges:
                owner_node_rows.append(node_row)
            else:
                orphan_node_rows.append(node_row)

        topic_label_rows = []
        topic_node_rows = []

        for topic in topics:
            y = topic_positions[topic]
            topic_label_rows.append(
                {
                    "x": topic_label_x,
                    "y": y,
                    "label": self._wrap_label(topic),
                    "topic": topic,
                }
            )
            topic_node_rows.append(
                {
                    "x": topic_node_x,
                    "y": y,
                    "topic": topic,
                    "subscriber_count": topic_counts.get(topic, 0),
                }
            )

        edge_rows = []
        for info in matching_subs:
            key = self._owner_key(info)
            topic = str(self._value(info, "topic", ""))

            if key not in owner_positions or topic not in topic_positions:
                continue

            edge_rows.append(
                {
                    "x0": owner_node_x,
                    "y0": owner_positions[key],
                    "x1": topic_node_x,
                    "y1": topic_positions[topic],
                    "owner_label": self._owner_label(info),
                    "owner_id": self._value(info, "owner_id"),
                    "owner_kind": self._value(info, "owner_kind"),
                    "topic": topic,
                    "callback_name": self._value(info, "callback_name"),
                    "module": self._value(info, "module"),
                }
            )

        all_y = list(owner_positions.values()) + list(topic_positions.values())
        if not all_y:
            all_y = [0.0]

        y_min = min(all_y) - 0.8
        y_max = max(all_y) + 0.8
        height = int(max(260, min(900, 120 + (y_max - y_min) * 90)))

        p = figure(
            title=f"Subscription Graph ({prefix if prefix not in ('', None) else '*'})",
            height=height,
            x_range=(0.0, total_width),
            y_range=(y_min, y_max),
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="right",
            sizing_mode="stretch_width",
            min_border_left=6,
            min_border_right=6,
            min_border_top=6,
            min_border_bottom=6,
        )

        p.grid.visible = False
        p.axis.visible = False
        p.outline_line_color = None
        p.toolbar.logo = None

        if edge_rows:
            src = ColumnDataSource(
                self._rows_to_cds_data(
                    edge_rows,
                    columns=[
                        "x0",
                        "y0",
                        "x1",
                        "y1",
                        "owner_label",
                        "owner_id",
                        "owner_kind",
                        "topic",
                        "callback_name",
                        "module",
                    ],
                )
            )
            renderer = p.segment(
                x0="x0",
                y0="y0",
                x1="x1",
                y1="y1",
                source=src,
                line_width=1.7,
                line_alpha=0.35,
            )
            p.add_tools(
                HoverTool(
                    renderers=[renderer],
                    tooltips=[
                        ("owner", "@owner_label"),
                        ("kind", "@owner_kind"),
                        ("topic", "@topic"),
                        ("callback", "@callback_name"),
                        ("module", "@module"),
                    ],
                )
            )

        if owner_label_rows:
            src = ColumnDataSource(
                self._rows_to_cds_data(
                    owner_label_rows,
                    columns=["x", "y", "label", "full_label"],
                )
            )
            p.add_layout(
                LabelSet(
                    x="x",
                    y="y",
                    text="label",
                    source=src,
                    text_align="center",
                    text_baseline="middle",
                    text_font_size="9pt",
                    text_font_style="bold",
                )
            )

        if topic_label_rows:
            src = ColumnDataSource(
                self._rows_to_cds_data(
                    topic_label_rows,
                    columns=["x", "y", "label", "topic"],
                )
            )
            p.add_layout(
                LabelSet(
                    x="x",
                    y="y",
                    text="label",
                    source=src,
                    text_align="center",
                    text_baseline="middle",
                    text_font_size="9pt",
                    text_font_style="bold",
                )
            )

        if owner_node_rows:
            src = ColumnDataSource(
                self._rows_to_cds_data(
                    owner_node_rows,
                    columns=[
                        "x",
                        "y",
                        "full_label",
                        "owner_id",
                        "owner_kind",
                        "subscription_count",
                        "dataset_topic_count",
                    ],
                )
            )
            renderer = p.scatter(
                x="x",
                y="y",
                size=10,
                marker="circle",
                source=src,
                alpha=0.95,
            )
            p.add_tools(
                HoverTool(
                    renderers=[renderer],
                    tooltips=[
                        ("owner", "@full_label"),
                        ("kind", "@owner_kind"),
                        ("owner_id", "@owner_id"),
                        ("subscriptions", "@subscription_count"),
                        ("dataset topics", "@dataset_topic_count"),
                    ],
                )
            )

        if orphan_node_rows:
            src = ColumnDataSource(
                self._rows_to_cds_data(
                    orphan_node_rows,
                    columns=[
                        "x",
                        "y",
                        "full_label",
                        "owner_id",
                        "owner_kind",
                        "subscription_count",
                        "dataset_topic_count",
                    ],
                )
            )
            renderer = p.scatter(
                x="x",
                y="y",
                size=8,
                marker="circle",
                source=src,
                alpha=0.25,
            )
            p.add_tools(
                HoverTool(
                    renderers=[renderer],
                    tooltips=[
                        ("owner", "@full_label"),
                        ("kind", "@owner_kind"),
                        ("owner_id", "@owner_id"),
                        ("subscriptions", "@subscription_count"),
                        ("dataset topics", "@dataset_topic_count"),
                    ],
                )
            )

        if topic_node_rows:
            src = ColumnDataSource(
                self._rows_to_cds_data(
                    topic_node_rows,
                    columns=["x", "y", "topic", "subscriber_count"],
                )
            )
            renderer = p.scatter(
                x="x",
                y="y",
                size=10,
                marker="square",
                source=src,
                alpha=0.95,
            )
            p.add_tools(
                HoverTool(
                    renderers=[renderer],
                    tooltips=[
                        ("topic", "@topic"),
                        ("subscribers", "@subscriber_count"),
                    ],
                )
            )

        if not edge_rows and not owner_node_rows and not orphan_node_rows and not topic_node_rows:
            p.text(
                x=[total_width / 2.0],
                y=[0.0],
                text=["No subscriptions found for the current filter."],
                text_align="center",
                text_baseline="middle",
            )

        return p

    # ------------------------------------------------------------------
    # Public refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        if self._disposed:
            return

        events = getattr(self.context, "events", None)
        if events is None:
            self.status.object = "### Event bus not available on context."
            return

        try:
            events.enable_trace(bool(self.trace_toggle.value))
        except Exception:
            pass

        prefix = self.topic_filter.value or "*"
        limit = int(self.limit_input.value or 250)

        try:
            sub_infos = self._get_subscription_infos()
            owners = self._build_owner_groups(sub_infos)

            matching_subs = [
                info
                for info in sub_infos
                if self._topic_matches_filter(str(self._value(info, "topic", "")), prefix)
            ]

            self.graph_pane.object = self._build_graph_figure(
                owners,
                matching_subs,
                prefix,
            )

            coverage_df = self._build_diagnostics_df(owners)
            subs_df = self._build_subscription_df(matching_subs)
            events_df = self._build_events_df(prefix, limit)

            self._set_tabulator_df_if_changed(self.coverage_table, coverage_df)
            self._set_tabulator_df_if_changed(self.subs_table, subs_df)
            self._set_tabulator_df_if_changed(self.events_table, events_df)

            missing_dataset = 0
            partial_dataset = 0

            for owner in owners.values():
                topics = set(owner["topics"])
                wildcard = "*" in topics
                dataset_any = wildcard or any(str(t).startswith("dataset.") for t in topics)

                if wildcard or all(t in topics for t in self.DATASET_TOPICS):
                    continue
                if dataset_any:
                    partial_dataset += 1
                else:
                    missing_dataset += 1

            workspace_count = ""
            workspace = getattr(self.context, "workspace", None)
            if workspace is not None:
                try:
                    workspace_count = f" • workspace panels: {len(workspace.list_panels())}"
                except Exception:
                    workspace_count = ""

            visible_topics = len({self._value(info, "topic", "") for info in matching_subs})

            self.status.object = (
                f"### Owners: {len(owners)} • matching subscriptions: {len(matching_subs)} "
                f"• visible topics: {visible_topics} • missing dataset.* owners: {missing_dataset} "
                f"• partial dataset.* owners: {partial_dataset}{workspace_count}"
            )
        except Exception as exc:
            self.status.object = f"### Event Monitor render error: `{exc}`"
            try:
                self.graph_pane.object = None
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        if self._disposed:
            return

        self._disposed = True

        try:
            if self._periodic_callback is not None:
                self._periodic_callback.stop()
        except Exception:
            pass

        for widget, watcher in list(self._watchers):
            try:
                widget.param.unwatch(watcher)
            except Exception:
                pass

        self._watchers.clear()