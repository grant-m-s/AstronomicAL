from __future__ import annotations

import html
import time
from typing import Any, List, Optional, Sequence

import panel as pn

from astronomicAL.platform.modal_utils import (
    open_template_modal,
    close_template_modal,
)

def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    try:
        seconds = float(seconds)
    except Exception:
        return ""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def _fmt_age(timestamp: Optional[float]) -> str:
    if not timestamp:
        return ""
    age = max(0.0, time.time() - float(timestamp))
    if age < 1.0:
        return "now"
    if age < 60.0:
        return f"{age:.0f}s ago"
    if age < 3600.0:
        return f"{age / 60.0:.0f}m ago"
    return f"{age / 3600.0:.1f}h ago"


def _table_html(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    max_rows: int = 12,
) -> str:
    if not rows:
        return (
            f"<div class='al-runtime-status-section'>"
            f"<div class='al-runtime-status-title'>{_escape(title)}</div>"
            f"<div class='al-runtime-status-muted'>None</div>"
            f"</div>"
        )

    shown = list(rows)[-max_rows:]
    header_html = "".join(f"<th>{_escape(h)}</th>" for h in headers)

    row_html = ""
    for row in shown:
        row_html += "<tr>"
        row_html += "".join(f"<td>{_escape(cell)}</td>" for cell in row)
        row_html += "</tr>"

    extra = ""
    if len(rows) > len(shown):
        extra = (
            f"<div class='al-runtime-status-muted'>"
            f"Showing latest {len(shown)} of {len(rows)}"
            f"</div>"
        )

    return (
        f"<div class='al-runtime-status-section'>"
        f"<div class='al-runtime-status-title'>{_escape(title)}</div>"
        f"<table class='al-runtime-status-table'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{row_html}</tbody>"
        f"</table>"
        f"{extra}"
        f"</div>"
    )


class RuntimeStatusBox:
    """
    Permanent compact runtime status widget.

    Header footprint stays fixed.

    Details open in the shared template modal. The header button does not turn
    into a close button, because template modals block clicks outside the modal.
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

        self.refresh_ms = int(refresh_ms)
        self.heartbeat_ms = int(heartbeat_ms)
        self.lag_threshold_s = float(lag_threshold_s)
        self.recent_slow_window_s = float(recent_slow_window_s)
        self.modal_refresh_ms = int(modal_refresh_ms)

        self._callbacks: List[Any] = []
        self._heartbeat_expected: Optional[float] = None

        self._details_open = False
        self._last_modal_update_monotonic = 0.0
        self._last_modal_html = ""
        self._last_open_error: Optional[str] = None

        self.summary = pn.pane.HTML(
            "",
            width=170,
            height=30,
            sizing_mode="fixed",
            margin=(0, 6, 0, 0),
        )

        self.toggle = pn.widgets.Button(
            name="details",
            button_type="default",
            width=72,
            height=30,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        )
        self.toggle.on_click(self._open_details)

        self.modal_body = pn.pane.HTML(
            "",
            width=760,
            height=500,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        )

        self.close_button = pn.widgets.Button(
            name="Close",
            button_type="primary",
            width=90,
            height=34,
            margin=(10, 0, 0, 0),
        )
        self.close_button.on_click(self._close_details)

        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="default",
            width=90,
            height=34,
            margin=(10, 8, 0, 0),
        )
        self.refresh_button.on_click(self._manual_refresh_modal)

        self.modal_view = pn.Column(
            pn.pane.HTML(
                """
                <div class="al-runtime-modal-title">
                    <div class="al-runtime-modal-heading">Runtime diagnostics</div>
                    <div class="al-runtime-modal-subtitle">
                        Active jobs, event pressure, slow callbacks, UI lag, and recent errors.
                    </div>
                </div>
                """,
                width=760,
                height=58,
                sizing_mode="fixed",
                margin=(0, 0, 8, 0),
            ),
            self.modal_body,
            pn.Row(
                pn.Spacer(sizing_mode="stretch_width"),
                self.refresh_button,
                self.close_button,
                sizing_mode="stretch_width",
                width=760,
                height=48,
                margin=(0, 0, 0, 0),
            ),
            width=790,
            height=650,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
            css_classes=["al-runtime-status-modal-card"],
        )

        self.view = pn.Row(
            self.summary,
            self.toggle,
            width=252,
            height=32,
            sizing_mode="fixed",
            margin=(0, 8, 0, 8),
            css_classes=["al-runtime-status-box"],
        )

        self._install_css()
        self._start_callbacks()
        self.refresh()

    def _install_css(self) -> None:
        css = """
        .al-runtime-status-box {
            font-size: 12px;
            line-height: 1.25;
            overflow: hidden;
        }

        .al-runtime-status-box * {
            box-sizing: border-box;
        }

        .al-runtime-status-pill {
            display: inline-block;
            box-sizing: border-box;
            border-radius: 999px;
            padding: 5px 10px;
            border: 1px solid rgba(0,0,0,0.18);
            background: rgba(255,255,255,0.72);
            white-space: nowrap;
            width: 170px;
            height: 28px;
            overflow: hidden;
            text-overflow: ellipsis;
            vertical-align: middle;
        }

        .al-runtime-status-pill.idle {
            background: rgba(80, 180, 80, 0.12);
        }

        .al-runtime-status-pill.busy {
            background: rgba(30, 120, 220, 0.12);
        }

        .al-runtime-status-pill.slow {
            background: rgba(255, 190, 0, 0.18);
        }

        .al-runtime-status-pill.error {
            background: rgba(220, 40, 40, 0.14);
        }

        .al-runtime-status-modal-card {
            background: #f3f5f7;
            border: 1px solid rgba(0,0,0,0.24);
            border-radius: 12px;
            padding: 14px;

            box-shadow: 0 8px 22px rgba(0,0,0,0.18);
            color: #111;
        }

        .al-runtime-modal-title {
            width: 100%;
            padding: 8px 12px;
            border-radius: 8px;
            background: #1f2933;
            color: white;
            box-sizing: border-box;
        }

        .al-runtime-modal-heading {
            font-size: 16px;
            font-weight: 700;
            line-height: 1.2;
        }

        .al-runtime-modal-subtitle {
            font-size: 12px;
            opacity: 0.78;
            margin-top: 2px;
        }

        .al-runtime-status-details {
            width: 760px;
            height: 500px;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 10px;
            border: 1px solid rgba(0,0,0,0.14);
            border-radius: 8px;
            background: white;
            color: #111;
            box-sizing: border-box;
        }

        .al-runtime-status-section {
            margin-bottom: 14px;
        }

        .al-runtime-status-title {
            font-weight: 700;
            margin-bottom: 5px;
            color: #111;
        }

        .al-runtime-status-muted {
            color: rgba(0,0,0,0.58);
            font-style: italic;
        }

        .al-runtime-status-table {
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;
            color: #111;
            background: white;
        }

        .al-runtime-status-table th,
        .al-runtime-status-table td {
            border: 1px solid rgba(0,0,0,0.12);
            padding: 4px 6px;
            text-align: left;
            vertical-align: top;
            word-wrap: break-word;
            color: #111;
            font-size: 12px;
        }

        .al-runtime-status-table th {
            background: #edf0f3;
            font-weight: 700;
        }

        .al-runtime-status-table tr:nth-child(even) td {
            background: #fafafa;
        }

        """

        try:
            if css not in pn.config.raw_css:
                pn.config.raw_css.append(css)
        except Exception:
            pass

    def _start_callbacks(self) -> None:
        try:
            refresh_cb = pn.state.add_periodic_callback(
                self.refresh,
                period=self.refresh_ms,
                start=True,
            )
            self._callbacks.append(refresh_cb)
        except Exception:
            pass

        try:
            self._heartbeat_expected = time.monotonic() + (
                self.heartbeat_ms / 1000.0
            )
            heartbeat_cb = pn.state.add_periodic_callback(
                self._heartbeat,
                period=self.heartbeat_ms,
                start=True,
            )
            self._callbacks.append(heartbeat_cb)
        except Exception:
            pass

    def dispose(self) -> None:
        for cb in list(self._callbacks):
            try:
                cb.stop()
            except Exception:
                pass
        self._callbacks.clear()

    def _template(self) -> Optional[Any]:
        if self.template is not None:
            return self.template

        workspace = getattr(self.context, "workspace", None)
        if workspace is not None:
            template = getattr(workspace, "react", None)
            if template is not None:
                return template

        config = getattr(self.context, "config", None)
        if config is not None:
            app_context = getattr(config, "app_context", None)
            workspace = getattr(app_context, "workspace", None)
            if workspace is not None:
                template = getattr(workspace, "react", None)
                if template is not None:
                    return template

        return None

    def _open_details(self, _event: Any = None) -> None:
        template = self._template()
        if template is None:
            self._last_open_error = "no template"
            print(
                "[RuntimeStatusBox] Cannot open diagnostics: no ReactTemplate found.",
                flush=True,
            )
            self.refresh()
            return

        self._details_open = True
        self._last_open_error = None

        # The modal blocks outside clicks, so the close control lives inside the
        # modal. Keep the header button as "details" and disable it while open.
        self.toggle.name = "details"
        self.toggle.disabled = True

        self._manual_refresh_modal()

        try:
            open_template_modal(
                template,
                self.modal_view,
                close_on_backdrop=True,
                on_close=self._on_modal_closed,
            )
            print("[RuntimeStatusBox] Opened runtime diagnostics modal.", flush=True)
        except Exception as exc:
            import traceback

            self._details_open = False
            self.toggle.disabled = False
            self._last_open_error = str(exc)
            print("[RuntimeStatusBox] Failed to open diagnostics modal:", exc, flush=True)
            traceback.print_exc()
            self.refresh()

    def _on_modal_closed(self) -> None:
        """
        Called by modal_utils whenever the diagnostics modal closes, including
        backdrop click.
        """
        self._details_open = False
        self.toggle.name = "details"
        self.toggle.disabled = False

    def _close_details(self, _event: Any = None) -> None:
        template = self._template()
        if template is None:
            self._on_modal_closed()
            return

        close_template_modal(template, clear=True, notify=True)

    def _manual_refresh_modal(self, _event: Any = None) -> None:
        data = self._collect()
        html_text = self._details_html(data)
        self._last_modal_html = html_text
        self._last_modal_update_monotonic = time.monotonic()
        self.modal_body.object = html_text
        self.summary.object = self._summary_html(data)

    def _maybe_refresh_modal(self, data: dict[str, list[Any]]) -> None:
        return

    def _heartbeat(self) -> None:
        now = time.monotonic()
        expected = self._heartbeat_expected

        interval = self.heartbeat_ms / 1000.0
        self._heartbeat_expected = now + interval

        if expected is None:
            return

        lag = now - expected
        if lag < self.lag_threshold_s:
            return

        runtime_status = getattr(self.context, "runtime_status", None)
        if runtime_status is None:
            return

        try:
            runtime_status.record_ui_lag(
                lag,
                expected_interval=interval,
                source="runtime_status_box",
            )
        except Exception:
            pass

    def _safe_list(self, obj: Any, method_name: str, *args: Any) -> list[Any]:
        method = getattr(obj, method_name, None)
        if not callable(method):
            return []
        try:
            value = method(*args)
        except Exception:
            return []
        if value is None:
            return []
        return list(value)

    def _collect(self) -> dict[str, list[Any]]:
        events = getattr(self.context, "events", None)
        jobs = getattr(self.context, "jobs", None)
        runtime_status = getattr(self.context, "runtime_status", None)

        return {
            "active_jobs": self._safe_list(jobs, "active_jobs")
            if jobs is not None
            else [],
            "recent_jobs": self._safe_list(jobs, "recent_jobs", 8)
            if jobs is not None
            else [],
            "active_events": self._safe_list(events, "active_publishes")
            if events is not None
            else [],
            "slow_callbacks": self._safe_list(events, "recent_slow_callbacks", 8)
            if events is not None
            else [],
            "publish_timings": self._safe_list(events, "recent_publish_timings", 8)
            if events is not None
            else [],
            "callback_errors": self._safe_list(events, "recent_callback_errors", 5)
            if events is not None
            else [],
            "ui_lags": self._safe_list(runtime_status, "recent_ui_lags", 5)
            if runtime_status is not None
            else [],
            "runtime_errors": self._safe_list(runtime_status, "recent_errors", 5)
            if runtime_status is not None
            else [],
        }

    def refresh(self) -> None:
        data = self._collect()
        self.summary.object = self._summary_html(data)
        self._maybe_refresh_modal(data)

    def _summary_html(self, data: dict[str, list[Any]]) -> str:
        active_jobs = data["active_jobs"]
        active_events = data["active_events"]
        slow_callbacks = data["slow_callbacks"]
        callback_errors = data["callback_errors"]
        runtime_errors = data["runtime_errors"]
        ui_lags = data["ui_lags"]

        now = time.time()

        if self._last_open_error:
            return (
                "<span class='al-runtime-status-pill error'>"
                f"● Modal error: {_escape(self._last_open_error)}"
                "</span>"
            )

        latest_error = None
        if callback_errors:
            latest_error = callback_errors[-1]
        if runtime_errors:
            latest_runtime_error = runtime_errors[-1]
            if latest_error is None or getattr(
                latest_runtime_error,
                "timestamp",
                0,
            ) >= getattr(latest_error, "timestamp", 0):
                latest_error = latest_runtime_error

        if latest_error is not None and now - getattr(latest_error, "timestamp", 0) < 30:
            source = getattr(latest_error, "source", None)
            owner = getattr(latest_error, "owner_label", None)
            label = owner or source or "runtime"
            message = getattr(latest_error, "message", None) or getattr(
                latest_error,
                "error",
                "",
            )
            return (
                "<span class='al-runtime-status-pill error'>"
                f"● Error: {_escape(label)}"
                f"{' — ' + _escape(message) if message else ''}"
                "</span>"
            )

        if active_jobs or active_events:
            bits = []

            if active_jobs:
                bits.append(
                    f"{len(active_jobs)} job{'s' if len(active_jobs) != 1 else ''}"
                )

            if active_events:
                topics = [
                    getattr(item, "topic", None)
                    for item in active_events[-2:]
                    if getattr(item, "topic", None)
                ]
                if topics:
                    bits.append("event " + ", ".join(str(t) for t in topics))
                else:
                    bits.append(
                        f"{len(active_events)} event"
                        f"{'s' if len(active_events) != 1 else ''}"
                    )

            latest_job = active_jobs[-1] if active_jobs else None
            latest_job_title = getattr(latest_job, "title", None)
            suffix = f" · {_escape(latest_job_title)}" if latest_job_title else ""

            return (
                "<span class='al-runtime-status-pill busy'>"
                f"● Busy: {_escape(' · '.join(bits))}{suffix}"
                "</span>"
            )

        recent_slow = None
        for item in reversed(slow_callbacks):
            if now - getattr(item, "timestamp", 0) <= self.recent_slow_window_s:
                recent_slow = item
                break

        if recent_slow is not None:
            owner = getattr(recent_slow, "owner_label", None) or getattr(
                recent_slow,
                "module",
                None,
            )
            callback = getattr(recent_slow, "callback_name", None)
            duration = _fmt_duration(getattr(recent_slow, "duration", None))
            label = owner or callback or "callback"
            return (
                "<span class='al-runtime-status-pill slow'>"
                f"● Slow: {_escape(label)}"
                f"{' · ' + _escape(duration) if duration else ''}"
                "</span>"
            )

        recent_lag = None
        for item in reversed(ui_lags):
            if now - getattr(item, "timestamp", 0) <= self.recent_slow_window_s:
                recent_lag = item
                break

        if recent_lag is not None:
            return (
                "<span class='al-runtime-status-pill slow'>"
                f"● UI lag: {_escape(_fmt_duration(getattr(recent_lag, 'lag', None)))}"
                "</span>"
            )

        return "<span class='al-runtime-status-pill idle'>● Idle</span>"

    def _details_html(self, data: dict[str, list[Any]]) -> str:
        active_jobs = data["active_jobs"]
        recent_jobs = data["recent_jobs"]
        active_events = data["active_events"]
        slow_callbacks = data["slow_callbacks"]
        publish_timings = data["publish_timings"]
        callback_errors = data["callback_errors"]
        runtime_errors = data["runtime_errors"]
        ui_lags = data["ui_lags"]

        active_job_rows = [
            (
                getattr(job, "title", ""),
                getattr(job, "status", ""),
                _fmt_duration(getattr(job, "elapsed", None)),
                getattr(job, "key", "") or "",
            )
            for job in active_jobs
        ]

        recent_job_rows = [
            (
                getattr(job, "title", ""),
                getattr(job, "status", ""),
                _fmt_duration(getattr(job, "elapsed", None)),
                _fmt_age(getattr(job, "finished_at", None)),
                getattr(job, "error", "") or "",
            )
            for job in recent_jobs
        ]

        active_event_rows = [
            (
                getattr(event, "topic", ""),
                _fmt_duration(getattr(event, "elapsed", None)),
                getattr(event, "callback_count", ""),
                getattr(event, "payload_summary", ""),
            )
            for event in active_events
        ]

        slow_rows = [
            (
                getattr(item, "topic", ""),
                getattr(item, "owner_label", "") or getattr(item, "module", ""),
                getattr(item, "callback_name", ""),
                _fmt_duration(getattr(item, "duration", None)),
                _fmt_age(getattr(item, "timestamp", None)),
            )
            for item in slow_callbacks
        ]

        publish_rows = [
            (
                getattr(item, "topic", ""),
                _fmt_duration(getattr(item, "duration", None)),
                getattr(item, "callback_count", ""),
                _fmt_age(getattr(item, "timestamp", None)),
            )
            for item in publish_timings
        ]

        error_rows = []

        for item in callback_errors:
            error_rows.append(
                (
                    "event",
                    getattr(item, "topic", ""),
                    getattr(item, "owner_label", "") or getattr(item, "module", ""),
                    getattr(item, "error", ""),
                    _fmt_age(getattr(item, "timestamp", None)),
                )
            )

        for item in runtime_errors:
            error_rows.append(
                (
                    "runtime",
                    getattr(item, "source", ""),
                    "",
                    getattr(item, "message", ""),
                    _fmt_age(getattr(item, "timestamp", None)),
                )
            )

        lag_rows = [
            (
                getattr(item, "source", ""),
                _fmt_duration(getattr(item, "lag", None)),
                _fmt_duration(getattr(item, "expected_interval", None)),
                _fmt_age(getattr(item, "timestamp", None)),
            )
            for item in ui_lags
        ]

        return (
            "<div class='al-runtime-status-details'>"
            + _table_html(
                "Active jobs",
                ["Title", "Status", "Elapsed", "Key"],
                active_job_rows,
            )
            + _table_html(
                "Active event publishes",
                ["Topic", "Elapsed", "Callbacks", "Payload"],
                active_event_rows,
            )
            + _table_html(
                "Recent slow callbacks",
                ["Topic", "Owner", "Callback", "Duration", "When"],
                slow_rows,
            )
            + _table_html(
                "Recent event publishes",
                ["Topic", "Duration", "Callbacks", "When"],
                publish_rows,
            )
            + _table_html(
                "Recent jobs",
                ["Title", "Status", "Elapsed", "Finished", "Error"],
                recent_job_rows,
            )
            + _table_html(
                "UI lag",
                ["Source", "Lag", "Expected", "When"],
                lag_rows,
            )
            + _table_html(
                "Errors",
                ["Type", "Topic/source", "Owner", "Error", "When"],
                error_rows,
            )
            + "</div>"
        )