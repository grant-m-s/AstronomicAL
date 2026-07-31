from __future__ import annotations

import html
from typing import Any, Callable

import panel as pn

from astronomicAL.plugins.event_monitor.diagnostics import (
    ACTIVE_STALL_THRESHOLD_SECONDS,
    DEFAULT_SUBSCRIPTION_ROW_LIMIT,
    TOPIC_GROUPS,
    EventHealthSummary,
    EventMonitorSnapshot,
    build_activity_rows,
    build_health_summary,
    build_subscription_rows,
    build_topic_rows,
    collect_snapshot,
    format_clock,
    value,
)
from astronomicAL.plugins.event_monitor.live_refresh import DocumentPeriodicRefresh
from astronomicAL.plugins.event_monitor.styles import EVENT_MONITOR_CSS

LIVE_REFRESH_PERIOD_MS = 2000
SUBSCRIPTION_DISPLAY_LIMIT = DEFAULT_SUBSCRIPTION_ROW_LIMIT


class EventMonitorPanel:
    """Read-only, user-facing event-bus observability panel.

    The panel polls the EventBus diagnostic buffers. It does not enable tracing,
    publish events, clear diagnostics, or register a wildcard subscription.
    """

    state_version = 1

    def __init__(self, context: Any, *, instance_id: str | None = None):
        self.context = context
        self.instance_id = str(instance_id or "")
        self._disposed = False
        self._restoring = False
        self._watchers: list[tuple[Any, Any]] = []
        self._event_subscriptions: list[tuple[Any, Any]] = []
        self._last_snapshot: EventMonitorSnapshot | None = None
        self._session_document: Any | None = None
        self._live_error: str | None = None
        self._live_refresh = DocumentPeriodicRefresh(
            self._periodic_refresh,
            period_ms=LIVE_REFRESH_PERIOD_MS,
        )

        self.search = pn.widgets.TextInput(
            name="Search",
            placeholder="Topic, owner, callback, or payload…",
            sizing_mode="stretch_width",
            margin=0,
        )
        self.topic_group = pn.widgets.Select(
            name="Topic group",
            options=list(TOPIC_GROUPS),
            value="All topics",
            sizing_mode="stretch_width",
            margin=0,
        )
        self.history_limit = pn.widgets.Select(
            name="History window",
            options={
                "50 events": 50,
                "100 events": 100,
                "250 events": 250,
                "500 events": 500,
            },
            value=100,
            sizing_mode="stretch_width",
            margin=0,
        )
        self.live_updates = pn.widgets.Checkbox(
            name="Live updates",
            value=True,
            width=110,
            margin=(4, 0, 0, 0),
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh now",
            button_type="primary",
            sizing_mode="stretch_width",
            height=32,
            margin=0,
        )

        self.header = self._html_pane()
        self.summary = self._html_pane()
        self.error_banner = self._html_pane()
        self.issues = self._html_pane()
        self.topic_traffic = self._html_pane()
        self.activity = self._html_pane()
        self.subscriptions = self._html_pane()

        self.refresh_button.on_click(self._on_refresh_click)
        self._watch(self.live_updates, self._on_live_change, "value")
        self._watch(self.topic_group, self._on_filter_change, "value")
        self._watch(self.history_limit, self._on_history_change, "value")
        self._watch(self.search, self._on_filter_change, "value")
        self._watch(self.search, self._on_filter_change, "value_input")

        self.view = self._build_view()
        self._subscribe_to_panel_opened()
        self.refresh()
        self._schedule_live_start()

    @staticmethod
    def _html_pane() -> pn.pane.HTML:
        return pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            stylesheets=[EVENT_MONITOR_CSS],
            margin=0,
        )

    def _build_view(self) -> pn.Column:
        controls = pn.Column(
            self.search,
            self.topic_group,
            self.history_limit,
            self.live_updates,
            self.refresh_button,
            sizing_mode="stretch_width",
            css_classes=["al-em-card", "al-em-controls"],
            styles={
                "background": "#ffffff",
                "border": "1px solid #dfe5ee",
                "border-radius": "9px",
                "box-shadow": "0 2px 8px rgba(27, 43, 65, 0.045)",
                "padding": "12px",
            },
            margin=(0, 0, 10, 0),
        )

        body = pn.Column(
            self.header,
            controls,
            self.error_banner,
            self.summary,
            self._section(
                "Needs attention",
                "Stalled publishes and recent callback issues",
                self.issues,
            ),
            self._section(
                "Topic traffic",
                "Most active topics in the selected history",
                self.topic_traffic,
            ),
            self._section(
                "Recent activity",
                "Newest matching completed publishes first",
                self.activity,
            ),
            self._section(
                "Subscriptions",
                f"First {SUBSCRIPTION_DISPLAY_LIMIT:,} matching live registrations",
                self.subscriptions,
            ),
            sizing_mode="stretch_both",
            min_width=220,
            scroll=True,
            css_classes=["al-event-monitor"],
            styles={
                "background": "transparent",
                "box-sizing": "border-box",
                "overflow-x": "hidden",
                "overflow-y": "auto",
                "padding": "8px",
            },
            stylesheets=[EVENT_MONITOR_CSS],
            margin=0,
        )
        return body

    @staticmethod
    def _section(title: str, detail: str, content: Any) -> pn.Column:
        heading = pn.pane.HTML(
            '<div class="al-em-section-title">'
            f"<h3>{html.escape(title)}</h3>"
            f"<span>{html.escape(detail)}</span>"
            "</div>",
            sizing_mode="stretch_width",
            stylesheets=[EVENT_MONITOR_CSS],
            margin=0,
        )
        return pn.Column(
            heading,
            content,
            sizing_mode="stretch_width",
            css_classes=["al-em-card"],
            styles={
                "background": "#ffffff",
                "border": "1px solid #dfe5ee",
                "border-radius": "9px",
                "box-shadow": "0 2px 8px rgba(27, 43, 65, 0.045)",
                "padding": "12px",
            },
            margin=(0, 0, 10, 0),
        )

    def refresh(self) -> None:
        if self._disposed:
            return

        events = getattr(self.context, "events", None)
        if events is None:
            self._render_unavailable("EventBus is not available on AppContext.")
            return

        try:
            snapshot = collect_snapshot(events, limit=int(self.history_limit.value))
            self._last_snapshot = snapshot
            self._render_snapshot(snapshot)
            self.error_banner.object = (
                self._issue_html("warning", self._live_error) if self._live_error else ""
            )
        except Exception as exc:
            self._render_error(f"Unable to read event diagnostics: {exc}")

    def _render_snapshot(self, snapshot: EventMonitorSnapshot) -> None:
        summary = build_health_summary(snapshot)
        topic_group = str(self.topic_group.value or "All topics")
        search = self._search_value()

        activity_rows = build_activity_rows(snapshot, topic_group=topic_group, search=search)
        subscription_rows = build_subscription_rows(
            snapshot,
            topic_group=topic_group,
            search=search,
            limit=SUBSCRIPTION_DISPLAY_LIMIT,
        )
        topic_rows = build_topic_rows(snapshot, topic_group=topic_group, search=search)

        self.header.object = self._header_html(snapshot, summary)
        self.summary.object = self._summary_html(snapshot, summary)
        self.issues.object = self._issues_html(snapshot)
        self.topic_traffic.object = self._topic_traffic_html(topic_rows)
        self.activity.object = self._activity_html(activity_rows)
        self.subscriptions.object = self._subscriptions_html(subscription_rows)

    def _render_unavailable(self, message: str) -> None:
        self.header.object = self._header_html(None, None)
        self.summary.object = self._empty_html("Event diagnostics are unavailable.")
        self.error_banner.object = self._issue_html("danger", message)
        self.issues.object = self._empty_html("No issue data.")
        self.topic_traffic.object = self._empty_html("No topic data.")
        self.activity.object = self._empty_html("No activity data.")
        self.subscriptions.object = self._empty_html("No subscription data.")

    def _render_error(self, message: str) -> None:
        self.error_banner.object = self._issue_html("danger", message)

    def _header_html(
        self,
        snapshot: EventMonitorSnapshot | None,
        summary: EventHealthSummary | None,
    ) -> str:
        live_requested = bool(self.live_updates.value)
        is_live = live_requested and self._live_refresh.running
        if is_live:
            state_label = "Live"
            state_class = ""
        elif live_requested:
            state_label = "Starting"
            state_class = " is-paused"
        else:
            state_label = "Paused"
            state_class = " is-paused"
        refreshed = format_clock(snapshot.captured_at) if snapshot else "—"
        issue_text = ""
        if summary is not None and summary.issue_count:
            issue_text = f"{summary.issue_count:,} recent signal(s) need review"

        return f"""
        <div class="al-em-header">
          <div class="al-em-header-main">
            <div class="al-em-eyebrow">Platform observability</div>
            <h2 class="al-em-title">Event Monitor</h2>
            <div class="al-em-subtitle">Completed throughput, synchronous latency, callback health, and live registrations.</div>
          </div>
          <div class="al-em-header-meta">
            <div class="al-em-live{state_class}"><span class="al-em-live-dot"></span>{state_label}</div>
            <div class="al-em-updated">Updated {html.escape(refreshed)}</div>
            {f'<div class="al-em-header-warning">{html.escape(issue_text)}</div>' if issue_text else ''}
          </div>
        </div>
        """

    def _summary_html(
        self,
        snapshot: EventMonitorSnapshot,
        summary: EventHealthSummary,
    ) -> str:
        topic_count = len({value(item, "topic", "") for item in snapshot.subscriptions})
        cards = [
            self._metric_html(
                "Event health",
                summary.status,
                self._health_detail(summary),
                summary.status_tone,
            ),
            self._metric_html(
                "Throughput",
                f"{summary.publishes_per_minute:,}/min",
                f"{len(snapshot.publish_timings):,} publishes in history",
            ),
            self._metric_html(
                "P95 latency",
                self._duration_label(summary.p95_publish_ms),
                "Synchronous delivery time",
                "warning" if summary.p95_publish_ms >= 50.0 else "",
            ),
            self._metric_html(
                "Subscriptions",
                f"{summary.subscriber_count:,}",
                f"{topic_count:,} registered topics",
            ),
        ]
        return '<div class="al-em-summary-grid">' + "".join(cards) + "</div>"

    @staticmethod
    def _metric_html(label: str, amount: str, detail: str, tone: str = "") -> str:
        return (
            '<div class="al-em-metric">'
            f'<div class="al-em-metric-label">{html.escape(label)}</div>'
            f'<div class="al-em-metric-value {html.escape(tone)}">{html.escape(amount)}</div>'
            f'<div class="al-em-metric-detail">{html.escape(detail)}</div>'
            "</div>"
        )

    def _activity_html(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return self._empty_html("No matching event activity.")

        blocks: list[str] = []
        for row in rows:
            status = str(row.get("status", "OK"))
            tone = {"Error": "danger", "Slow": "warning"}.get(status, "success")
            topic = html.escape(str(row.get("topic", "")))
            payload = html.escape(str(row.get("summary", "") or ""))
            duration = self._duration_label(float(row.get("duration_ms", 0.0) or 0.0))
            callbacks = int(row.get("callbacks", 0) or 0)
            payload_html = f'<div class="al-em-event-payload">{payload}</div>' if payload else ""
            blocks.append(
                '<article class="al-em-event">'
                '<div class="al-em-event-top">'
                f'<code title="{topic}">{topic}</code>'
                f'<span class="al-em-badge {tone}">{html.escape(status)}</span>'
                "</div>"
                f'<div class="al-em-event-meta">{html.escape(str(row.get("time", "—")))} · '
                f"{html.escape(duration)} · {callbacks:,} callback(s)</div>"
                f"{payload_html}</article>"
            )
        return '<div class="al-em-stack">' + "".join(blocks) + "</div>"

    def _subscriptions_html(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return self._empty_html("No matching subscriptions.")

        blocks: list[str] = []
        for row in rows:
            topic = html.escape(str(row.get("topic", "")))
            owner = html.escape(str(row.get("owner", "Unidentified subscriber")))
            kind = html.escape(str(row.get("kind", "subscriber")))
            callback = html.escape(str(row.get("callback", "") or "—"))
            module = html.escape(str(row.get("module", "") or "—"))
            blocks.append(
                '<article class="al-em-subscription">'
                f'<code title="{topic}">{topic}</code>'
                f'<div class="al-em-subscription-owner">{owner}</div>'
                f'<div class="al-em-subscription-meta">{kind} · {callback}</div>'
                f'<div class="al-em-subscription-module" title="{module}">{module}</div>'
                "</article>"
            )
        return '<div class="al-em-stack">' + "".join(blocks) + "</div>"

    def _issues_html(self, snapshot: EventMonitorSnapshot) -> str:
        cutoff = snapshot.captured_at - 300.0
        blocks: list[str] = []

        active = sorted(
            (
                item
                for item in snapshot.active_publishes
                if float(value(item, "elapsed", 0.0) or 0.0)
                >= ACTIVE_STALL_THRESHOLD_SECONDS
            ),
            key=lambda item: float(value(item, "elapsed", 0.0) or 0.0),
            reverse=True,
        )
        for item in active[:4]:
            topic = html.escape(str(value(item, "topic", "Unknown topic")))
            elapsed = float(value(item, "elapsed", 0.0) or 0.0)
            callbacks = int(value(item, "callback_count", 0) or 0)
            blocks.append(
                '<div class="al-em-issue danger">'
                f'<div class="al-em-issue-title">Stalled publish on {topic}</div>'
                f'<div class="al-em-issue-meta">Active for {elapsed:.2f}s · '
                f"{callbacks:,} callback(s)</div></div>"
            )

        errors = [
            item
            for item in snapshot.callback_errors
            if float(value(item, "timestamp", 0.0) or 0.0) >= cutoff
        ]
        for item in reversed(errors[-4:]):
            topic = html.escape(str(value(item, "topic", "Unknown topic")))
            owner = html.escape(self._owner_label(item))
            error = html.escape(str(value(item, "error", "Callback failed")))
            trace = html.escape(str(value(item, "traceback_text", "") or ""))
            trace_html = ""
            if trace:
                trace_html = (
                    '<details class="al-em-trace"><summary>Show traceback</summary>'
                    f"<pre>{trace}</pre></details>"
                )
            blocks.append(
                '<div class="al-em-issue danger">'
                f'<div class="al-em-issue-title">{topic}</div>'
                f'<div class="al-em-issue-meta">{owner} · {error}</div>'
                f"{trace_html}</div>"
            )

        slow = [
            item
            for item in snapshot.slow_callbacks
            if float(value(item, "timestamp", 0.0) or 0.0) >= cutoff
        ]
        for item in reversed(slow[-5:]):
            topic = html.escape(str(value(item, "topic", "Unknown topic")))
            owner = html.escape(self._owner_label(item))
            duration_ms = float(value(item, "duration", 0.0) or 0.0) * 1000.0
            callback = html.escape(str(value(item, "callback_name", "callback") or "callback"))
            blocks.append(
                '<div class="al-em-issue warning">'
                f'<div class="al-em-issue-title">Slow callback on {topic}</div>'
                f'<div class="al-em-issue-meta">{owner} · {callback} · '
                f"{duration_ms:.1f} ms</div></div>"
            )

        if not blocks:
            return self._empty_html(
                "No stalled publishes, callback errors, or slow callbacks in the last five minutes."
            )
        return '<div class="al-em-stack">' + "".join(blocks) + "</div>"

    def _topic_traffic_html(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return self._empty_html("No matching publish history.")

        maximum = max(int(row["count"]) for row in rows) or 1
        blocks: list[str] = []
        for row in rows:
            width = max(4.0, (int(row["count"]) / maximum) * 100.0)
            topic = html.escape(str(row["topic"]))
            count = int(row["count"])
            average = float(row["average_ms"])
            maximum_ms = float(row.get("max_ms", 0.0) or 0.0)
            blocks.append(
                '<div class="al-em-topic-row">'
                '<div class="al-em-topic-top">'
                f'<code title="{topic}">{topic}</code>'
                f'<span>{count:,} publish(es)</span>'
                "</div>"
                f'<div class="al-em-topic-bar"><span style="width:{width:.1f}%"></span></div>'
                f'<div class="al-em-topic-stat">avg {average:.1f} ms · max {maximum_ms:.1f} ms</div>'
                "</div>"
            )
        return '<div class="al-em-stack">' + "".join(blocks) + "</div>"

    @staticmethod
    def _empty_html(message: str) -> str:
        return f'<div class="al-em-empty">{html.escape(message)}</div>'

    @staticmethod
    def _issue_html(tone: str, message: str) -> str:
        return (
            f'<div class="al-em-issue {html.escape(tone)}">'
            f'<div class="al-em-issue-title">{html.escape(message)}</div></div>'
        )

    @staticmethod
    def _health_detail(summary: EventHealthSummary) -> str:
        if summary.issue_count == 0:
            if summary.active_count:
                return f"No recent callback issues · {summary.active_count:,} active"
            return "No recent callback issues"
        return (
            f"{summary.recent_error_count:,} error(s), "
            f"{summary.recent_slow_count:,} slow, "
            f"{summary.stalled_count:,} stalled"
        )

    @staticmethod
    def _duration_label(milliseconds: float) -> str:
        if milliseconds >= 1000.0:
            return f"{milliseconds / 1000.0:.2f}s"
        if milliseconds >= 10.0:
            return f"{milliseconds:.1f}ms"
        return f"{milliseconds:.2f}ms"

    @staticmethod
    def _owner_label(item: Any) -> str:
        return str(
            value(item, "owner_label")
            or value(item, "owner_id")
            or value(item, "module")
            or "Unidentified subscriber"
        )

    def _search_value(self) -> str:
        value_input = getattr(self.search, "value_input", None)
        current = value_input if value_input is not None else self.search.value
        return str(current or "").strip()

    def _subscribe_to_panel_opened(self) -> None:
        """Start from the platform's post-attachment lifecycle event.

        AstronomicAL constructs plugin panels through ``JobManager``. The factory
        therefore runs on a worker thread where no Bokeh session document exists.
        ``plugin.panel.opened`` is published only after the completed panel has
        been inserted into the workspace on the UI document thread, making it the
        reliable point at which to create the per-session periodic callback.
        """
        if not self.instance_id:
            return

        events = getattr(self.context, "events", None)
        subscribe = getattr(events, "subscribe", None)
        if not callable(subscribe):
            return

        subscription = subscribe(
            "plugin.panel.opened",
            self._on_panel_opened,
            owner_id=self.instance_id,
            owner_label="Event Monitor",
            owner_kind="panel",
        )
        self._event_subscriptions.append((events, subscription))

    def _on_panel_opened(self, _topic: str, payload: Any) -> None:
        if self._disposed or self._live_refresh.running:
            return
        if not bool(self.live_updates.value) or not isinstance(payload, dict):
            return
        if str(payload.get("panel_id") or "") != self.instance_id:
            return

        self._start_live_updates(document=getattr(pn.state, "curdoc", None))

    def _schedule_live_start(self) -> None:
        """Start now when already on a session document.

        Worker-thread construction intentionally leaves the panel in ``Starting``.
        The instance-specific ``plugin.panel.opened`` subscription completes
        startup after workspace attachment on the UI document thread.
        """
        if self._disposed or not bool(self.live_updates.value):
            self._update_header_only()
            return
        self._start_live_updates()

    def _start_live_updates(self, *, document: Any | None = None) -> None:
        if self._disposed or not bool(self.live_updates.value):
            self._update_header_only()
            return

        current_document = getattr(pn.state, "curdoc", None)
        target_document = document or current_document or self._session_document

        if self._live_refresh.running:
            if (
                target_document is not None
                and self._live_refresh.document is not target_document
            ):
                self._live_error = (
                    "Live updates are already attached to another session document."
                )
            else:
                self._live_error = None
            self._update_header_only()
            return

        if target_document is None:
            self._update_header_only()
            return

        self._session_document = target_document
        try:
            self._live_refresh.start(target_document)
            self._live_error = None
        except Exception as exc:
            self._live_error = f"Live updates could not start: {exc}"
        self._update_header_only()

    def _stop_live_updates(self) -> None:
        try:
            self._live_refresh.stop()
            self._live_error = None
        except Exception as exc:
            self._live_error = f"Live updates could not stop cleanly: {exc}"
        self._update_header_only()

    def _on_refresh_click(self, _event: Any = None) -> None:
        if self._disposed:
            return
        if self.live_updates.value and not self._live_refresh.running:
            self._schedule_live_start()
        self.refresh()

    def _periodic_refresh(self) -> None:
        if (
            self._disposed
            or not bool(self.live_updates.value)
            or not self._live_refresh.running
        ):
            return
        self.refresh()

    def _on_live_change(self, _event: Any = None) -> None:
        if self._restoring or self._disposed:
            return
        if self.live_updates.value:
            self._schedule_live_start()
            self.refresh()
        else:
            self._stop_live_updates()

    def _on_filter_change(self, _event: Any = None) -> None:
        if self._restoring or self._disposed:
            return
        if self._last_snapshot is not None:
            self._render_snapshot(self._last_snapshot)
        else:
            self.refresh()

    def _on_history_change(self, _event: Any = None) -> None:
        if self._restoring or self._disposed:
            return
        self.refresh()

    def _update_header_only(self) -> None:
        summary = build_health_summary(self._last_snapshot) if self._last_snapshot is not None else None
        self.header.object = self._header_html(self._last_snapshot, summary)

    def _watch(self, widget: Any, callback: Callable[..., Any], attribute: str) -> None:
        watcher = widget.param.watch(callback, attribute)
        self._watchers.append((widget, watcher))

    def get_state(self) -> dict[str, Any]:
        return {
            "live_updates": bool(self.live_updates.value),
            "topic_group": str(self.topic_group.value),
            "history_limit": int(self.history_limit.value),
            "search": self._search_value(),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self._restoring = True
        try:
            topic_group = str(state.get("topic_group", "All topics"))
            if topic_group in TOPIC_GROUPS:
                self.topic_group.value = topic_group

            history_limit = int(state.get("history_limit", 100))
            if history_limit in self.history_limit.options.values():
                self.history_limit.value = history_limit

            search = str(state.get("search", "") or "")
            self.search.value = search
            if hasattr(self.search, "value_input"):
                try:
                    self.search.value_input = search
                except Exception:
                    pass
            self.live_updates.value = bool(state.get("live_updates", True))
        finally:
            self._restoring = False
        if self.live_updates.value:
            self._schedule_live_start()
        else:
            self._stop_live_updates()
        self.refresh()

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True

        try:
            self._live_refresh.dispose()
        except Exception:
            pass

        for events, subscription in self._event_subscriptions:
            unsubscribe = getattr(events, "unsubscribe", None)
            if not callable(unsubscribe):
                continue
            try:
                unsubscribe(subscription)
            except Exception:
                pass
        self._event_subscriptions.clear()

        for widget, watcher in self._watchers:
            try:
                widget.param.unwatch(watcher)
            except Exception:
                pass
        self._watchers.clear()