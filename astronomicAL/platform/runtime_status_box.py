from __future__ import annotations

import html
import time
from typing import Any, Optional

import panel as pn

from astronomicAL.platform.modal_utils import close_template_modal, open_template_modal
from astronomicAL.platform.runtime_diagnostics import (
    ACTIVE_PUBLISH_STALL_SECONDS,
    RECENT_ISSUE_WINDOW_SECONDS,
    RuntimeDiagnosticsData,
    RuntimeHealthSummary,
    build_runtime_health,
    collect_runtime_diagnostics,
    format_age,
    format_clock,
    format_duration_ms,
    format_seconds,
    value,
)
from astronomicAL.platform.runtime_status_styles import RUNTIME_STATUS_CSS

class RuntimeStatusBox:
    """Compact runtime indicator and UI-latency diagnostics modal.

    One lightweight heartbeat callback owns both lag detection and view refresh.
    The header distinguishes background work from evidence that the UI thread is
    delayed. The modal polls bounded, read-only platform diagnostics and never
    publishes events or mutates jobs.
    """

    def __init__(
        self,
        context: Any,
        *,
        template: Optional[Any] = None,
        refresh_ms: int = 500,
        heartbeat_ms: int = 250,
        lag_threshold_s: float = 0.75,
        recent_slow_window_s: float = 12.0,
        modal_refresh_ms: int = 2500,
    ) -> None:
        self.context = context
        self.template = template
        self.refresh_ms = max(100, int(refresh_ms))
        self.heartbeat_ms = max(100, int(heartbeat_ms))
        self.lag_threshold_s = max(0.05, float(lag_threshold_s))
        self.recent_slow_window_s = max(1.0, float(recent_slow_window_s))
        self.modal_refresh_ms = max(500, int(modal_refresh_ms))

        self._disposed = False
        self._callback: Any | None = None
        self._heartbeat_expected: float | None = None
        self._next_summary_refresh = 0.0
        self._next_modal_refresh = 0.0
        self._details_open = False
        self._last_open_error: str | None = None
        self._last_data: RuntimeDiagnosticsData | None = None
        self._last_summary: RuntimeHealthSummary | None = None

        self.summary = self._html_pane(
            width=170,
            height=30,
            sizing_mode="fixed",
            margin=(0, 6, 0, 0),
        )
        self.toggle = pn.widgets.Button(
            name="Details",
            button_type="default",
            width=72,
            height=30,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        )
        self.toggle.on_click(self._open_details)

        self.modal_header = self._html_pane(
            height=92,
            sizing_mode="stretch_width",
            margin=0,
        )
        # Keep the scrolling element as a stable Panel layout model. Updating the
        # object of a scrollable HTML pane replaces the DOM element that owns
        # scrollTop, which forces the modal back to the top on every live refresh.
        # The persistent Column below owns scrolling; only its child HTML pane is
        # updated, so browser scroll position is retained while content changes.
        self.modal_content = self._html_pane(
            sizing_mode="stretch_width",
            margin=0,
        )
        self.modal_body = pn.Column(
            self.modal_content,
            min_height=320,
            sizing_mode="stretch_both",
            scroll=True,
            margin=0,
            css_classes=["al-rd-scroll"],
            stylesheets=[RUNTIME_STATUS_CSS],
        )
        self.modal_footer_status = self._html_pane(
            sizing_mode="stretch_width",
            height=34,
            margin=0,
        )

        self.refresh_button = pn.widgets.Button(
            name="Refresh now",
            button_type="default",
            width=104,
            height=32,
            margin=0,
        )
        self.refresh_button.on_click(self._manual_refresh_modal)
        self.close_button = pn.widgets.Button(
            name="Close",
            button_type="primary",
            width=86,
            height=32,
            margin=0,
        )
        self.close_button.on_click(self._close_details)

        footer = pn.Row(
            self.modal_footer_status,
            pn.Spacer(sizing_mode="stretch_width"),
            self.refresh_button,
            self.close_button,
            sizing_mode="stretch_width",
            height=50,
            margin=0,
            css_classes=["al-rd-footer"],
            stylesheets=[RUNTIME_STATUS_CSS],
        )

        # The shared overlay host is intentionally zero-sized in normal
        # template flow. Keep the modal root fixed so Bokeh has concrete layout
        # dimensions before CSS and viewport constraints are applied. Stretch
        # sizing here collapses against the host and renders as a thin line.
        self.modal_view = pn.Column(
            self.modal_header,
            self.modal_body,
            footer,
            width=1080,
            height=760,
            min_width=320,
            min_height=480,
            sizing_mode="fixed",
            margin=0,
            css_classes=["al-runtime-status-modal-card", "al-runtime-diagnostics"],
            stylesheets=[RUNTIME_STATUS_CSS],
        )

        self.view = pn.Row(
            self.summary,
            self.toggle,
            width=252,
            height=32,
            sizing_mode="fixed",
            margin=(0, 8, 0, 8),
            css_classes=["al-runtime-status-box"],
            stylesheets=[RUNTIME_STATUS_CSS],
        )

        self._install_css()
        self.refresh()
        self._start_callback()

    @staticmethod
    def _html_pane(**kwargs: Any) -> pn.pane.HTML:
        return pn.pane.HTML("", stylesheets=[RUNTIME_STATUS_CSS], **kwargs)

    @staticmethod
    def _install_css() -> None:
        try:
            if RUNTIME_STATUS_CSS not in pn.config.raw_css:
                pn.config.raw_css.append(RUNTIME_STATUS_CSS)
        except Exception:
            pass

    def _start_callback(self) -> None:
        if self._disposed or self._callback is not None:
            return
        now = time.monotonic()
        self._heartbeat_expected = now + (self.heartbeat_ms / 1000.0)
        self._next_summary_refresh = now + (self.refresh_ms / 1000.0)
        self._next_modal_refresh = now + (self.modal_refresh_ms / 1000.0)
        try:
            self._callback = pn.state.add_periodic_callback(
                self._tick,
                period=self.heartbeat_ms,
                start=True,
            )
        except Exception:
            self._callback = None

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True

        callback = self._callback
        self._callback = None
        if callback is not None:
            try:
                callback.stop()
            except Exception:
                pass

        if self._details_open:
            template = self._template()
            if template is not None:
                try:
                    close_template_modal(template, clear=True, notify=False)
                except Exception:
                    pass
        self._on_modal_closed()

    def _tick(self) -> None:
        if self._disposed:
            return

        now = time.monotonic()
        self._record_heartbeat(now)

        refresh_summary = now >= self._next_summary_refresh
        refresh_modal = self._details_open and now >= self._next_modal_refresh
        if not refresh_summary and not refresh_modal:
            return

        data = self._collect()
        summary = build_runtime_health(data)
        self._last_data = data
        self._last_summary = summary

        if refresh_summary:
            self.summary.object = self._summary_html(summary)
            self._next_summary_refresh = now + (self.refresh_ms / 1000.0)

        if refresh_modal:
            self._render_modal(data, summary)
            self._next_modal_refresh = now + (self.modal_refresh_ms / 1000.0)

    def _record_heartbeat(self, now: float) -> None:
        interval = self.heartbeat_ms / 1000.0
        expected = self._heartbeat_expected
        self._heartbeat_expected = now + interval
        if expected is None:
            return

        lag = now - expected
        if lag < self.lag_threshold_s:
            return

        runtime_status = getattr(self.context, "runtime_status", None)
        record = getattr(runtime_status, "record_ui_lag", None)
        if not callable(record):
            return
        try:
            record(
                lag,
                expected_interval=interval,
                source="runtime_status_box",
            )
        except Exception:
            pass

    def _template(self) -> Optional[Any]:
        if self.template is not None:
            return self.template

        workspace = getattr(self.context, "workspace", None)
        if workspace is not None:
            template = getattr(workspace, "react", None)
            if template is not None:
                return template

        return None

    def _open_details(self, _event: Any = None) -> None:
        if self._disposed or self._details_open:
            return

        template = self._template()
        if template is None:
            self._last_open_error = "No application template is available."
            self.refresh()
            return

        self._details_open = True
        self._last_open_error = None
        self.toggle.disabled = True
        self._manual_refresh_modal()

        try:
            open_template_modal(
                template,
                self.modal_view,
                close_on_backdrop=True,
                on_close=self._on_modal_closed,
            )
        except Exception as exc:
            self._details_open = False
            self.toggle.disabled = False
            self._last_open_error = str(exc)
            self.refresh()

    def _on_modal_closed(self) -> None:
        self._details_open = False
        self.toggle.disabled = False

    def _close_details(self, _event: Any = None) -> None:
        template = self._template()
        if template is None:
            self._on_modal_closed()
            return
        close_template_modal(template, clear=True, notify=True)

    def _manual_refresh_modal(self, _event: Any = None) -> None:
        if self._disposed:
            return
        data = self._collect()
        summary = build_runtime_health(data)
        self._last_data = data
        self._last_summary = summary
        self.summary.object = self._summary_html(summary)
        self._render_modal(data, summary)
        now = time.monotonic()
        self._next_summary_refresh = now + (self.refresh_ms / 1000.0)
        self._next_modal_refresh = now + (self.modal_refresh_ms / 1000.0)

    def refresh(self) -> None:
        if self._disposed:
            return
        data = self._collect()
        summary = build_runtime_health(data)
        self._last_data = data
        self._last_summary = summary
        self.summary.object = self._summary_html(summary)
        if self._details_open:
            self._render_modal(data, summary)

    def _collect(self) -> RuntimeDiagnosticsData:
        return collect_runtime_diagnostics(
            self.context,
            event_limit=100,
            job_limit=20,
            runtime_limit=20,
        )

    def _render_modal(
        self,
        data: RuntimeDiagnosticsData,
        summary: RuntimeHealthSummary,
    ) -> None:
        self.modal_header.object = self._modal_header_html(data, summary)
        self.modal_content.object = self._details_html(data, summary)
        self.modal_footer_status.object = (
            '<div class="al-rd-footer-status">'
            f"Live refresh every {self.modal_refresh_ms / 1000.0:g}s · "
            f"heartbeat {self.heartbeat_ms}ms · "
            f"lag threshold {format_seconds(self.lag_threshold_s)}"
            "</div>"
        )

    def _summary_html(self, summary: RuntimeHealthSummary) -> str:
        if self._last_open_error:
            label = "Modal error"
            detail = self._last_open_error
            tone = "danger"
        else:
            label = summary.status
            detail = summary.headline.replace(f"{summary.status} · ", "")
            tone = summary.tone

        legacy = {
            "success": "idle",
            "info": "busy",
            "warning": "slow",
            "danger": "error",
        }.get(tone, "idle")
        return (
            f'<span class="al-runtime-status-pill {legacy} {self._escape(tone)}" '
            f'title="{self._escape(summary.detail if not self._last_open_error else detail)}">'
            '<span class="al-rd-status-dot"></span>'
            f'<span class="al-rd-status-label">{self._escape(label)}</span>'
            f'<span class="al-rd-status-detail">{self._escape(detail)}</span>'
            "</span>"
        )

    def _modal_header_html(
        self,
        data: RuntimeDiagnosticsData,
        summary: RuntimeHealthSummary,
    ) -> str:
        warning_class = summary.tone if summary.tone in {"warning", "danger"} else ""
        return f"""
        <div class="al-rd-header">
          <div class="al-rd-header-main">
            <div class="al-rd-eyebrow">Platform observability</div>
            <h2 class="al-rd-title">Runtime diagnostics</h2>
            <div class="al-rd-subtitle">
              Find UI-thread stalls, synchronous event bottlenecks, background work,
              and recent failures without adding event subscriptions.
            </div>
          </div>
          <div class="al-rd-header-meta">
            <div class="al-rd-live"><span class="al-rd-live-dot"></span>Live</div>
            <div class="al-rd-updated">Updated {self._escape(format_clock(data.captured_at))}</div>
            <div class="al-rd-header-status {self._escape(warning_class)}">
              {self._escape(summary.headline)}
            </div>
          </div>
        </div>
        """

    def _details_html(
        self,
        data: RuntimeDiagnosticsData,
        summary: RuntimeHealthSummary,
    ) -> str:
        metrics = "".join(
            [
                self._metric_html(
                    "UI responsiveness",
                    summary.status,
                    self._ui_metric_detail(summary),
                    summary.tone,
                ),
                self._metric_html(
                    "Active work",
                    f"{summary.active_job_count} jobs · {summary.active_publish_count} events",
                    "Background jobs are not classified as UI lag.",
                    "info" if summary.active_job_count or summary.active_publish_count else "",
                ),
                self._metric_html(
                    "Event delivery p95",
                    format_duration_ms(summary.event_p95_ms),
                    f"Max {format_duration_ms(summary.event_max_ms)} · {summary.subscriber_count} subscriptions",
                    "warning" if summary.event_p95_ms >= 50.0 else "",
                ),
                self._metric_html(
                    "Signals in 5 minutes",
                    str(summary.issue_count),
                    f"{summary.recent_error_count} errors · {summary.recent_slow_count} slow · {summary.recent_ui_lag_count} lag",
                    "danger" if summary.recent_error_count or summary.stalled_publish_count else (
                        "warning" if summary.issue_count else "success"
                    ),
                ),
            ]
        )

        left = (
            self._section_html(
                "Needs attention",
                "Signals most likely to explain a laggy interaction",
                self._attention_html(data),
                summary.issue_count,
            )
            + self._section_html(
                "UI responsiveness",
                "Heartbeat delays recorded after the UI thread became responsive again",
                self._ui_lag_html(data),
                len(data.ui_lags),
            )
            + self._section_html(
                "Event delivery",
                "Synchronous callback latency; use Event Monitor for full topic history",
                self._event_delivery_html(data, summary),
                len(data.publish_timings),
            )
        )
        right = (
            self._section_html(
                "Active work",
                "Background jobs and publishes currently visible to the platform",
                self._active_work_html(data),
                len(data.active_jobs) + len(data.active_publishes),
            )
            + self._section_html(
                "Recent jobs",
                "Newest completed, failed, or cancelled work first",
                self._recent_jobs_html(data),
                len(data.recent_jobs),
            )
            + self._section_html(
                "Failures and runtime notes",
                "Recent platform messages with available details and tracebacks",
                self._messages_html(data),
                len(data.callback_errors) + len(data.runtime_errors) + len(data.runtime_notes),
            )
        )

        return (
            f'<div class="al-rd-metrics">{metrics}</div>'
            f'<div class="al-rd-grid"><div>{left}</div><div>{right}</div></div>'
        )

    def _attention_html(self, data: RuntimeDiagnosticsData) -> str:
        cutoff = data.captured_at - RECENT_ISSUE_WINDOW_SECONDS
        signals: list[tuple[int, float, str]] = []

        for item in data.active_publishes:
            elapsed = self._float(value(item, "elapsed", 0.0))
            if elapsed < ACTIVE_PUBLISH_STALL_SECONDS:
                continue
            topic = value(item, "topic", "Unknown topic")
            signals.append(
                (
                    0,
                    data.captured_at,
                    self._signal_html(
                        "danger",
                        f"Event publish still active: {topic}",
                        f"{format_seconds(elapsed)} · {value(item, 'callback_count', 0)} callbacks",
                        str(value(item, "payload_summary", "") or ""),
                    ),
                )
            )

        for item in data.callback_errors:
            timestamp = self._timestamp(item)
            if timestamp < cutoff:
                continue
            owner = value(item, "owner_label") or value(item, "module") or "Unidentified callback"
            signals.append(
                (
                    1,
                    timestamp,
                    self._signal_html(
                        "danger",
                        f"Callback error: {value(item, 'topic', 'Unknown topic')}",
                        f"{owner} · {format_age(timestamp, now=data.captured_at)}",
                        str(value(item, "error", "Callback failed") or "Callback failed"),
                        trace=str(value(item, "traceback_text", "") or ""),
                    ),
                )
            )

        for item in data.runtime_errors:
            timestamp = self._timestamp(item)
            if timestamp < cutoff:
                continue
            signals.append(
                (
                    1,
                    timestamp,
                    self._signal_html(
                        "danger",
                        f"Runtime error: {value(item, 'source', 'runtime')}",
                        format_age(timestamp, now=data.captured_at),
                        self._joined_detail(value(item, "message"), value(item, "details")),
                        trace=str(value(item, "traceback_text", "") or ""),
                    ),
                )
            )

        for item in data.ui_lags:
            timestamp = self._timestamp(item)
            if timestamp < cutoff:
                continue
            lag = self._float(value(item, "lag", 0.0))
            signals.append(
                (
                    2,
                    timestamp,
                    self._signal_html(
                        "warning",
                        f"UI heartbeat delayed by {format_seconds(lag)}",
                        f"{value(item, 'source', 'ui_heartbeat')} · {format_age(timestamp, now=data.captured_at)}",
                        "The event loop could not service the heartbeat while synchronous UI work was running.",
                    ),
                )
            )

        for item in data.slow_callbacks:
            timestamp = self._timestamp(item)
            if timestamp < cutoff:
                continue
            owner = value(item, "owner_label") or value(item, "module") or "Unidentified callback"
            signals.append(
                (
                    3,
                    timestamp,
                    self._signal_html(
                        "warning",
                        f"Slow callback: {value(item, 'topic', 'Unknown topic')}",
                        f"{owner} · {value(item, 'callback_name', 'callback')} · {format_seconds(value(item, 'duration', 0.0))}",
                        str(value(item, "payload_summary", "") or ""),
                    ),
                )
            )

        if not signals:
            return self._empty_html(
                "No recent UI lag, stalled publishes, callback errors, or slow callbacks.",
                tone="success",
            )

        signals.sort(key=lambda item: (item[0], -item[1]))
        return '<div class="al-rd-stack">' + "".join(item[2] for item in signals[:10]) + "</div>"

    def _active_work_html(self, data: RuntimeDiagnosticsData) -> str:
        rows: list[str] = []
        for job in data.active_jobs:
            status = str(value(job, "status", "running") or "running")
            title = str(value(job, "title", "Background job") or "Background job")
            detail = str(value(job, "key", "") or "Unkeyed job")
            rows.append(
                self._row_html(
                    title,
                    format_seconds(value(job, "elapsed", 0.0)),
                    detail,
                    badge=(status, "info" if status in {"queued", "running"} else "warning"),
                )
            )

        for publish in data.active_publishes:
            elapsed = self._float(value(publish, "elapsed", 0.0))
            tone = "danger" if elapsed >= ACTIVE_PUBLISH_STALL_SECONDS else "info"
            rows.append(
                self._row_html(
                    f"Event · {value(publish, 'topic', 'Unknown topic')}",
                    format_seconds(elapsed),
                    f"{value(publish, 'callback_count', 0)} callbacks",
                    str(value(publish, "payload_summary", "") or ""),
                    badge=("stalled" if tone == "danger" else "active", tone),
                )
            )

        if not rows:
            return self._empty_html("No active background jobs or observable publishes.")
        return '<div class="al-rd-stack">' + "".join(rows) + "</div>"

    def _ui_lag_html(self, data: RuntimeDiagnosticsData) -> str:
        rows: list[str] = []
        maximum = max((self._float(value(item, "lag", 0.0)) for item in data.ui_lags), default=0.0)
        for item in reversed(data.ui_lags[-10:]):
            lag = self._float(value(item, "lag", 0.0))
            width = 100.0 if maximum <= 0 else max(4.0, min(100.0, lag / maximum * 100.0))
            rows.append(
                self._row_html(
                    str(value(item, "source", "ui_heartbeat") or "ui_heartbeat"),
                    format_seconds(lag),
                    f"Expected heartbeat {format_seconds(value(item, 'expected_interval', 0.0))} · {format_age(value(item, 'timestamp'), now=data.captured_at)}",
                    progress=width,
                )
            )

        if not rows:
            content = self._empty_html("No heartbeat delay has crossed the configured threshold.", tone="success")
        else:
            content = '<div class="al-rd-stack">' + "".join(rows) + "</div>"
        return (
            content
            + '<div class="al-rd-footnote">Heartbeat records appear after the UI thread becomes responsive again; they identify that a stall occurred, not necessarily the exact callback that caused it.</div>'
        )

    def _event_delivery_html(
        self,
        data: RuntimeDiagnosticsData,
        summary: RuntimeHealthSummary,
    ) -> str:
        rows: list[str] = []
        for item in reversed(data.slow_callbacks[-6:]):
            owner = value(item, "owner_label") or value(item, "module") or "Unidentified callback"
            rows.append(
                self._row_html(
                    str(value(item, "topic", "Unknown topic")),
                    format_seconds(value(item, "duration", 0.0)),
                    f"{owner} · {value(item, 'callback_name', 'callback')} · {format_age(value(item, 'timestamp'), now=data.captured_at)}",
                    str(value(item, "payload_summary", "") or ""),
                    badge=("slow", "warning"),
                )
            )

        if not rows:
            for item in reversed(data.publish_timings[-6:]):
                duration_ms = self._float(value(item, "duration", 0.0)) * 1000.0
                tone = "warning" if duration_ms >= 50.0 else "success"
                rows.append(
                    self._row_html(
                        str(value(item, "topic", "Unknown topic")),
                        format_duration_ms(duration_ms),
                        f"{value(item, 'callback_count', 0)} callbacks · {format_age(value(item, 'timestamp'), now=data.captured_at)}",
                        str(value(item, "payload_summary", "") or ""),
                        badge=("slow" if tone == "warning" else "ok", tone),
                    )
                )

        if not rows:
            return self._empty_html("No completed publish timings are available.")
        return (
            '<div class="al-rd-stack">'
            + "".join(rows)
            + f'<div class="al-rd-footnote">p95 {self._escape(format_duration_ms(summary.event_p95_ms))} · max {self._escape(format_duration_ms(summary.event_max_ms))}. Event delivery is synchronous, so expensive callbacks should submit jobs instead of blocking the publisher.</div>'
            + "</div>"
        )

    def _recent_jobs_html(self, data: RuntimeDiagnosticsData) -> str:
        rows: list[str] = []
        for job in reversed(data.recent_jobs[-10:]):
            status = str(value(job, "status", "finished") or "finished")
            tone = {
                "finished": "success",
                "cancelled": "warning",
                "error": "danger",
            }.get(status, "info")
            rows.append(
                self._row_html(
                    str(value(job, "title", "Background job") or "Background job"),
                    format_seconds(value(job, "elapsed", 0.0)),
                    f"{format_age(value(job, 'finished_at'), now=data.captured_at)} · {value(job, 'key', '') or 'unkeyed'}",
                    str(value(job, "error", "") or ""),
                    badge=(status, tone),
                )
            )
        if not rows:
            return self._empty_html("No completed jobs are available.")
        return '<div class="al-rd-stack">' + "".join(rows) + "</div>"

    def _messages_html(self, data: RuntimeDiagnosticsData) -> str:
        messages: list[tuple[float, str]] = []
        for item in data.callback_errors:
            timestamp = self._timestamp(item)
            owner = value(item, "owner_label") or value(item, "module") or "Unidentified callback"
            messages.append(
                (
                    timestamp,
                    self._signal_html(
                        "danger",
                        f"{value(item, 'topic', 'Unknown topic')}",
                        f"Event callback · {owner} · {format_age(timestamp, now=data.captured_at)}",
                        str(value(item, "error", "Callback failed") or "Callback failed"),
                        trace=str(value(item, "traceback_text", "") or ""),
                    ),
                )
            )
        for item in data.runtime_errors:
            timestamp = self._timestamp(item)
            messages.append(
                (
                    timestamp,
                    self._signal_html(
                        "danger",
                        str(value(item, "message", "Runtime error") or "Runtime error"),
                        f"{value(item, 'source', 'runtime')} · {format_age(timestamp, now=data.captured_at)}",
                        str(value(item, "details", "") or ""),
                        trace=str(value(item, "traceback_text", "") or ""),
                    ),
                )
            )
        for item in data.runtime_notes:
            timestamp = self._timestamp(item)
            messages.append(
                (
                    timestamp,
                    self._signal_html(
                        "info",
                        str(value(item, "message", "Runtime note") or "Runtime note"),
                        f"{value(item, 'source', 'runtime')} · {format_age(timestamp, now=data.captured_at)}",
                        str(value(item, "details", "") or ""),
                    ),
                )
            )

        if not messages:
            return self._empty_html("No runtime failures or notes are available.")
        messages.sort(key=lambda item: item[0], reverse=True)
        return '<div class="al-rd-stack">' + "".join(item[1] for item in messages[:10]) + "</div>"

    @staticmethod
    def _ui_metric_detail(summary: RuntimeHealthSummary) -> str:
        if summary.recent_ui_lag_count:
            return f"Latest {format_duration_ms(summary.latest_ui_lag_ms)} · max {format_duration_ms(summary.max_ui_lag_ms)}"
        return "No heartbeat delay in the recent diagnostic window."

    @classmethod
    def _metric_html(cls, label: str, amount: str, detail: str, tone: str = "") -> str:
        return (
            '<div class="al-rd-metric">'
            f'<div class="al-rd-metric-label">{cls._escape(label)}</div>'
            f'<div class="al-rd-metric-value {cls._escape(tone)}">{cls._escape(amount)}</div>'
            f'<div class="al-rd-metric-detail">{cls._escape(detail)}</div>'
            "</div>"
        )

    @classmethod
    def _section_html(
        cls,
        title: str,
        detail: str,
        content: str,
        count: int | None = None,
    ) -> str:
        count_html = (
            f'<span class="al-rd-section-count">{int(count):,}</span>'
            if count is not None
            else ""
        )
        return (
            '<section class="al-rd-card">'
            '<div class="al-rd-section-heading">'
            f'<div><h3>{cls._escape(title)}</h3><p>{cls._escape(detail)}</p></div>'
            f"{count_html}</div>{content}</section>"
        )

    @classmethod
    def _empty_html(cls, message: str, *, tone: str = "") -> str:
        return f'<div class="al-rd-empty {cls._escape(tone)}">{cls._escape(message)}</div>'

    @classmethod
    def _signal_html(
        cls,
        tone: str,
        title: str,
        meta: str,
        detail: str = "",
        *,
        trace: str = "",
    ) -> str:
        detail_html = f'<div class="al-rd-row-detail">{cls._escape(detail)}</div>' if detail else ""
        trace_html = (
            '<details class="al-rd-trace"><summary>Show traceback</summary>'
            f'<pre>{cls._escape(trace)}</pre></details>'
            if trace
            else ""
        )
        return (
            f'<article class="al-rd-signal {cls._escape(tone)}">'
            '<div class="al-rd-signal-top">'
            f'<div class="al-rd-signal-title">{cls._escape(title)}</div>'
            f'<span class="al-rd-badge {cls._escape(tone)}">{cls._escape(tone)}</span>'
            "</div>"
            f'<div class="al-rd-signal-meta">{cls._escape(meta)}</div>'
            f"{detail_html}{trace_html}</article>"
        )

    @classmethod
    def _row_html(
        cls,
        title: str,
        amount: str,
        meta: str,
        detail: str = "",
        *,
        badge: tuple[str, str] | None = None,
        progress: float | None = None,
    ) -> str:
        badge_html = ""
        if badge is not None:
            label, tone = badge
            badge_html = f'<span class="al-rd-badge {cls._escape(tone)}">{cls._escape(label)}</span>'
        detail_html = f'<div class="al-rd-row-detail">{cls._escape(detail)}</div>' if detail else ""
        progress_html = ""
        if progress is not None:
            width = max(0.0, min(100.0, float(progress)))
            progress_html = f'<div class="al-rd-progress"><span style="width:{width:.1f}%"></span></div>'
        return (
            '<article class="al-rd-row">'
            '<div class="al-rd-row-top">'
            f'<div class="al-rd-row-title">{cls._escape(title)}</div>'
            f'<div class="al-rd-row-value">{cls._escape(amount)}</div>'
            f"{badge_html}</div>"
            f'<div class="al-rd-row-meta">{cls._escape(meta)}</div>'
            f"{progress_html}{detail_html}</article>"
        )

    @staticmethod
    def _joined_detail(*parts: Any) -> str:
        return " · ".join(str(item) for item in parts if item not in (None, ""))

    @staticmethod
    def _timestamp(record: Any) -> float:
        return RuntimeStatusBox._float(value(record, "timestamp", 0.0))

    @staticmethod
    def _float(raw: Any) -> float:
        try:
            return float(raw or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _escape(raw: Any) -> str:
        return html.escape("" if raw is None else str(raw), quote=True)