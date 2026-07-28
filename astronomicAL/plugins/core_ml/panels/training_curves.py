from __future__ import annotations

import base64
from io import BytesIO
import time
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import panel as pn

from ..progress import PROGRESS_EVENT, progress_alert_type, progress_markdown, progress_percent

class MLTrainingCurvesPanel:
    """Panel for viewing training curves and Optuna trial history."""

    def __init__(self, *, context: Any, restore_state: Optional[Dict[str, Any]] = None) -> None:
        self.context = context

        self.log_select = pn.widgets.Select(name="", options={})
        self.refresh = pn.widgets.Button(name="Refresh logs", button_type="light")
        self.reset_curve_controls = pn.widgets.Button(name="Reset metric selections", button_type="light")

        self.summary = pn.pane.Markdown("")
        # Reserve a fixed live-status viewport. The progress text can gain or
        # lose context, timing, ETA and row/batch lines between updates. Without a
        # fixed viewport those changes resize the selector block and make the tabs
        # below visibly jump.
        self.live_status = pn.pane.Alert(
            "No active training run is selected.",
            alert_type="info",
            sizing_mode="stretch_width",
            height=184,
            min_height=184,
            max_height=184,
            styles={
                "box-sizing": "border-box",
                "overflow-y": "auto",
                "overflow-x": "hidden",
            },
        )
        self.live_progress = pn.indicators.Progress(
            name="",
            value=0,
            max=100,
            visible=True,
            sizing_mode="stretch_width",
            height=16,
            margin=(0, 0, 0, 0),
            styles={"visibility": "hidden"},
        )
        self.live_progress_slot = pn.Column(
            self.live_progress,
            sizing_mode="stretch_width",
            height=24,
            min_height=24,
            max_height=24,
            margin=(0, 0, 6, 0),
            styles={"box-sizing": "border-box", "overflow": "hidden"},
        )
        self.curve_help = pn.pane.Alert(
            "Select a training log to configure visible curves.",
            alert_type="info",
            sizing_mode="stretch_width",
        )

        self.loss_metric_select = pn.widgets.MultiChoice(
            name="",
            options=[],
            value=[],
            placeholder="Choose loss columns",
            sizing_mode="stretch_width",
        )
        self.metric_select = pn.widgets.MultiChoice(
            name="",
            options=[],
            value=[],
            placeholder="Choose metric columns",
            sizing_mode="stretch_width",
        )
        self.y_scale = pn.widgets.Select(
            name="",
            options={"Auto": "auto", "Linear": "linear", "Log": "log"},
            value="auto",
            sizing_mode="stretch_width",
        )
        self.x_scale = pn.widgets.Select(
            name="",
            options={"Linear": "linear", "Log": "log"},
            value="linear",
            sizing_mode="stretch_width",
        )
        self.smoothing_window = pn.widgets.IntInput(
            name="",
            value=1,
            start=1,
            end=500,
            sizing_mode="stretch_width",
        )
        self.epoch_start = pn.widgets.IntInput(
            name="",
            value=0,
            start=0,
            sizing_mode="stretch_width",
        )
        self.epoch_end = pn.widgets.IntInput(
            name="",
            value=0,
            start=0,
            sizing_mode="stretch_width",
        )
        self.show_points = pn.widgets.Checkbox(
            name="Show point markers",
            value=False,
            sizing_mode="stretch_width",
        )

        self.loss_plot_html = pn.pane.HTML(
            self._plot_message_html("Loss curves will appear after epoch metrics are produced."),
            height=380,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
        self.metric_plot_html = pn.pane.HTML(
            self._plot_message_html("Metric curves will appear after epoch metrics are produced."),
            height=380,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
        self.loss_plot = pn.Column(
            self.loss_plot_html,
            sizing_mode="stretch_width",
            styles={"min-height": "380px", "background": "#f7f7f7"},
        )
        self.metric_plot = pn.Column(
            self.metric_plot_html,
            sizing_mode="stretch_width",
            styles={"min-height": "380px", "background": "#f7f7f7"},
        )
        self.epoch_table = pn.pane.DataFrame(pd.DataFrame(), height=260, sizing_mode="stretch_width")

        self.tuning_summary = pn.pane.Markdown(
            "No tuning information for the selected run.",
            sizing_mode="stretch_width",
        )
        self.current_trial_summary = pn.pane.Markdown(
            "No current Optuna trial.",
            sizing_mode="stretch_width",
        )
        self.current_trial_plot = pn.Column(sizing_mode="stretch_width")
        self.current_trial_table = pn.pane.DataFrame(pd.DataFrame(), height=220, sizing_mode="stretch_width")
        self.tuning_plot = pn.Column(sizing_mode="stretch_width")
        self.tuning_table = pn.pane.DataFrame(pd.DataFrame(), height=280, sizing_mode="stretch_width")

        self.log_select.sizing_mode = "stretch_width"
        self.log_select.height = 38
        self.refresh.sizing_mode = "stretch_width"
        self.refresh.height = 34
        self.reset_curve_controls.sizing_mode = "stretch_width"
        self.reset_curve_controls.height = 34

        self.refresh.on_click(lambda *_: self._load_logs(force_render=True))
        self.reset_curve_controls.on_click(lambda *_: self._reset_curve_controls())
        self.log_select.param.watch(lambda *_: self._render_selected(force_render=True), "value")

        for widget in [
            self.loss_metric_select,
            self.metric_select,
            self.y_scale,
            self.x_scale,
            self.smoothing_window,
            self.epoch_start,
            self.epoch_end,
            self.show_points,
        ]:
            widget.param.watch(
                lambda *_: self._render_selected(sync_controls=False, force_render=True),
                "value",
            )

        self._pending_log_event_artifact_id: Optional[str] = None
        self._pending_log_event_force = False
        self._pending_log_event_scheduled = False
        self._last_log_refresh_at = 0.0
        self._last_rendered_epoch_count_by_artifact: Dict[str, int] = {}
        self._last_rendered_status_by_artifact: Dict[str, str] = {}
        self._latest_live_progress_by_artifact: Dict[str, Dict[str, Any]] = {}
        self._latest_live_progress_by_run: Dict[str, Dict[str, Any]] = {}

        # Event-driven curve redraw cadence. Manual widget changes and finished
        # events still render immediately.
        self._event_refresh_min_interval_seconds = 2.0
        self._event_refresh_epoch_step = 3

        self._has_loss_plot = False
        self._has_metric_plot = False

        self._subscriptions: List[Any] = []
        self._subscribe_to_training_log_events()
        self._load_logs(force_render=True)

        if restore_state:
            self.restore_state(restore_state)

    def _subscribe_to_training_log_events(self) -> None:
        events = getattr(self.context, "events", None)
        subscribe = getattr(events, "subscribe", None)
        if not callable(subscribe):
            return

        for topic in [
            "ml.training_log.created",
            "ml.training_log.updated",
            "ml.recipe_run.started",
            PROGRESS_EVENT,
            "ml.recipe_run.paused",
            "ml.recipe_run.finished",
            "ml.training.started",
            "ml.training.finished",
        ]:
            try:
                sub = subscribe(
                    topic,
                    self._on_training_log_event,
                    owner_label="ML Training Curves",
                    owner_kind="panel",
                )
            except TypeError:
                sub = subscribe(topic, self._on_training_log_event)
            except Exception:
                continue
            self._subscriptions.append(sub)

    def _on_training_log_event(self, topic: str, payload: Any) -> None:
        artifact_id = None
        if isinstance(payload, dict):
            artifact_id = (
                payload.get("artifact_id")
                or payload.get("training_log_artifact_id")
                or payload.get("final_training_log_artifact_id")
            )

        topic_text = str(topic or "")
        if topic_text == PROGRESS_EVENT and isinstance(payload, dict):
            progress_payload = dict(payload)

            def update_progress() -> None:
                artifact_key = str(artifact_id or "")
                run_key = str(progress_payload.get("run_id") or "")
                if artifact_key:
                    self._latest_live_progress_by_artifact[artifact_key] = dict(progress_payload)
                if run_key:
                    self._latest_live_progress_by_run[run_key] = dict(progress_payload)

                if (
                    artifact_key
                    and artifact_key in self.log_select.options.values()
                    and self.log_select.value != artifact_key
                ):
                    try:
                        self.log_select.value = artifact_key
                    except Exception:
                        pass

                # Refresh the durable summary first, but do not let a slightly
                # older artifact snapshot overwrite the event that just arrived.
                self._refresh_summary_only(
                    select_artifact_id=artifact_key or None,
                    render_progress=False,
                )
                self._render_live_progress(progress_payload)

            try:
                doc = pn.state.curdoc
                if doc is not None:
                    doc.add_next_tick_callback(update_progress)
                    return
            except Exception:
                pass
            update_progress()
            return

        force = topic_text.endswith(".finished") or topic_text.endswith(".created") or topic_text.endswith(".paused")

        if artifact_id:
            self._pending_log_event_artifact_id = str(artifact_id)
        self._pending_log_event_force = bool(self._pending_log_event_force or force)

        if self._pending_log_event_scheduled:
            return

        self._pending_log_event_scheduled = True

        def update() -> None:
            self._pending_log_event_scheduled = False

            selected = self._pending_log_event_artifact_id
            force_render = bool(self._pending_log_event_force)

            self._pending_log_event_artifact_id = None
            self._pending_log_event_force = False

            try:
                should_render = force_render or self._should_refresh_for_artifact(selected)
            except Exception:
                should_render = force_render

            if not should_render:
                self._refresh_summary_only(select_artifact_id=selected)
                return

            self._last_log_refresh_at = time.time()
            self._load_logs(select_artifact_id=selected, force_render=force_render)

        delay_ms = 0 if force else 500
        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_timeout_callback(update, delay_ms)
                return
        except Exception:
            pass

        update()

    def _should_refresh_for_artifact(self, artifact_id: Optional[str]) -> bool:
        now = time.time()
        if now - float(self._last_log_refresh_at or 0.0) < float(self._event_refresh_min_interval_seconds):
            return False

        artifact_id = str(artifact_id or self.log_select.value or "")
        if not artifact_id:
            return True

        try:
            payload = self.context.artifacts.get(artifact_id)
        except Exception:
            return True

        status = str(payload.get("status", "") or "")
        epochs = payload.get("epochs", []) or []
        epoch_count = len(epochs)

        previous_status = self._last_rendered_status_by_artifact.get(artifact_id)
        terminal_statuses = {"finished", "complete", "completed", "failed", "error", "cancelled"}
        if status and status != previous_status and status in terminal_statuses:
            return True

        previous_epoch_count = int(self._last_rendered_epoch_count_by_artifact.get(artifact_id, 0))
        if epoch_count <= 0:
            return previous_epoch_count <= 0 and not (self._has_loss_plot or self._has_metric_plot)

        return epoch_count - previous_epoch_count >= int(self._event_refresh_epoch_step)

    def _refresh_summary_only(
        self,
        select_artifact_id: Optional[str] = None,
        *,
        render_progress: bool = True,
    ) -> None:
        artifact_id = str(select_artifact_id or self.log_select.value or "")
        if not artifact_id:
            return

        try:
            payload = self.context.artifacts.get(artifact_id)
        except Exception:
            return

        if select_artifact_id and select_artifact_id in self.log_select.options.values():
            try:
                self.log_select.value = select_artifact_id
            except Exception:
                pass

        tuning = payload.get("tuning") or {}
        tuning_trials = payload.get("tuning_trials") or []
        try:
            self.summary.object = self._summary_markdown(payload, tuning, tuning_trials)
            if render_progress:
                self._render_progress_from_log_payload(payload, artifact_id=artifact_id)
        except Exception:
            pass

    def dispose(self) -> None:
        events = getattr(self.context, "events", None)
        unsubscribe = getattr(events, "unsubscribe", None)
        if callable(unsubscribe):
            for sub in self._subscriptions:
                try:
                    unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions = []

    def panel(self):
        header = pn.pane.HTML(
            "<h3 style='margin:0;'>ML training curves</h3>",
            height=32,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        selector_block = pn.Column(
            header,
            self._field("Training log", self.log_select),
            pn.Row(self.refresh, self.reset_curve_controls, sizing_mode="stretch_width"),
            self.live_status,
            self.live_progress_slot,
            sizing_mode="stretch_width",
            margin=(0, 0, 10, 0),
            styles={"box-sizing": "border-box", "overflow": "visible"},
        )

        summary_tab = pn.Column(
            self.summary,
            sizing_mode="stretch_width",
            styles={"box-sizing": "border-box", "padding": "8px 4px 14px 0", "overflow": "visible"},
        )

        controls_tab = pn.Column(
            self.curve_help,
            pn.Row(
                self._field("Loss curves", self.loss_metric_select),
                self._field("Metric curves", self.metric_select),
                sizing_mode="stretch_width",
            ),
            pn.Row(
                self._field("Y axis", self.y_scale),
                self._field("X axis", self.x_scale),
                sizing_mode="stretch_width",
            ),
            pn.Row(
                self._field("Smoothing window", self.smoothing_window),
                self._field("Start epoch, 0 = first", self.epoch_start),
                self._field("End epoch, 0 = last", self.epoch_end),
                sizing_mode="stretch_width",
            ),
            self.show_points,
            sizing_mode="stretch_width",
            styles={"box-sizing": "border-box", "padding": "8px 4px 14px 0", "overflow": "visible"},
        )

        loss_tab = pn.Column(
            self.loss_plot,
            sizing_mode="stretch_width",
            styles={"box-sizing": "border-box", "padding": "8px 4px 14px 0", "overflow": "visible"},
        )

        metrics_tab = pn.Column(
            self.metric_plot,
            sizing_mode="stretch_width",
            styles={"box-sizing": "border-box", "padding": "8px 4px 14px 0", "overflow": "visible"},
        )

        optuna_tab = pn.Column(
            self.tuning_summary,
            pn.layout.Divider(),
            pn.pane.Markdown("### Current Optuna trial"),
            self.current_trial_summary,
            self.current_trial_plot,
            self.current_trial_table,
            pn.layout.Divider(),
            pn.pane.Markdown("### Completed trials"),
            self.tuning_plot,
            self.tuning_table,
            sizing_mode="stretch_width",
            styles={"box-sizing": "border-box", "padding": "8px 4px 14px 0", "overflow": "visible"},
        )

        epochs_tab = pn.Column(
            self.epoch_table,
            sizing_mode="stretch_width",
            styles={"box-sizing": "border-box", "padding": "8px 4px 14px 0", "overflow": "visible"},
        )

        body_tabs = pn.Tabs(
            ("Summary", summary_tab),
            ("Curve controls", controls_tab),
            ("Loss", loss_tab),
            ("Metrics", metrics_tab),
            ("Optuna", optuna_tab),
            ("Epochs", epochs_tab),
            dynamic=True,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
            styles={"box-sizing": "border-box", "overflow": "visible"},
        )

        return pn.Column(
            selector_block,
            body_tabs,
            sizing_mode="stretch_both",
            scroll=True,
            styles={
                "box-sizing": "border-box",
                "padding": "10px 14px 14px 14px",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "height": "100%",
                "min-height": "0",
            },
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "training_log_artifact_id": self.log_select.value,
            "loss_metrics": list(self.loss_metric_select.value or []),
            "metrics": list(self.metric_select.value or []),
            "y_scale": self.y_scale.value,
            "x_scale": self.x_scale.value,
            "smoothing_window": self.smoothing_window.value,
            "epoch_start": self.epoch_start.value,
            "epoch_end": self.epoch_end.value,
            "show_points": self.show_points.value,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        artifact_id = state.get("training_log_artifact_id")
        if artifact_id in self.log_select.options.values():
            self.log_select.value = artifact_id

        for attr, key in [
            ("y_scale", "y_scale"),
            ("x_scale", "x_scale"),
            ("smoothing_window", "smoothing_window"),
            ("epoch_start", "epoch_start"),
            ("epoch_end", "epoch_end"),
            ("show_points", "show_points"),
        ]:
            widget = getattr(self, attr, None)
            if widget is None or key not in state:
                continue
            try:
                widget.value = state[key]
            except Exception:
                pass

        self._render_selected(sync_controls=True, force_render=True)

        for widget, key in [
            (self.loss_metric_select, "loss_metrics"),
            (self.metric_select, "metrics"),
        ]:
            values = state.get(key)
            if not values:
                continue
            allowed = set(widget.options or [])
            try:
                widget.value = [value for value in values if value in allowed]
            except Exception:
                pass

        self._render_selected(sync_controls=False, force_render=True)

    def _field(self, label: str, widget):
        return pn.Column(
            pn.pane.HTML(
                f"<div style='font-size:12px;font-weight:600;color:#555;margin-bottom:2px;'>{label}</div>",
                height=18,
                sizing_mode="stretch_width",
                margin=(0, 0, 2, 0),
            ),
            widget,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
            styles={"box-sizing": "border-box", "overflow": "visible"},
        )

    def _plot_message_html(self, message: str) -> str:
        safe = str(message).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"""
            <div style="
            min-height: 350px;
            height: 350px;
            width: 100%;
            box-sizing: border-box;
            border-radius: 12px;
            border: 1px solid rgba(0,0,0,0.08);
            background: linear-gradient(180deg, #fbfbfb, #f2f2f2);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #555;
            font-size: 13px;
            text-align: center;
            padding: 18px;
            ">
            <div>{safe}</div>
            </div>
        """

    def _figure_to_img_html(self, fig, *, alt: str) -> str:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        safe_alt = str(alt).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"""
            <div style="
            min-height: 350px;
            width: 100%;
            box-sizing: border-box;
            border-radius: 12px;
            border: 1px solid rgba(0,0,0,0.08);
            background: #f7f7f7;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            padding: 4px;
            ">
            <img
                src="data:image/png;base64,{data}"
                alt="{safe_alt}"
                style="
                display: block;
                width: 100%;
                height: auto;
                max-height: 350px;
                object-fit: contain;
                border-radius: 8px;
                background: #ffffff;
                "
            />
            </div>
        """

    def _load_logs(self, select_artifact_id: Optional[str] = None, *, force_render: bool = False) -> None:
        artifacts = getattr(self.context, "artifacts", None)
        find = getattr(artifacts, "find", None)
        if not callable(find):
            self.log_select.options = {}
            self.summary.object = "Artifact store is not available."
            return

        try:
            refs = find(type="ml.training_log")
        except Exception as exc:
            self.log_select.options = {}
            self.summary.object = f"Could not load training logs: `{exc}`"
            return

        options = {}
        for ref in refs:
            try:
                payload = artifacts.get(ref.artifact_id)
            except Exception:
                payload = {}

            run_id = str(payload.get("run_id", ref.artifact_id))
            model_title = str(payload.get("model_title", "unknown model"))
            dataset_id = str(payload.get("dataset_id", ref.dataset_id))
            status = str(payload.get("status", "unknown"))
            created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ref.created_at))
            tuning = payload.get("tuning") or {}
            tuning_flag = " | Optuna" if tuning.get("enabled") or payload.get("tuning_trials") else ""
            label = f"{created} | {model_title} | {dataset_id} | {status}{tuning_flag} | {run_id[:8]}"
            options[label] = ref.artifact_id

        current = self.log_select.value
        previous_options = dict(self.log_select.options or {})
        if previous_options != options:
            self.log_select.options = options

        if select_artifact_id in options.values():
            self.log_select.value = select_artifact_id
        elif current in options.values():
            self.log_select.value = current
        elif options and not self.log_select.value:
            self.log_select.value = next(iter(options.values()))
        elif not options:
            self.log_select.value = None

        self._render_selected(force_render=force_render)

    def _render_selected(self, sync_controls: bool = True, *, force_render: bool = False) -> None:
        artifact_id = self.log_select.value

        if not artifact_id:
            self.summary.object = "No training logs yet.\nTrain a model first."
            self.live_status.alert_type = "info"
            self.live_status.object = "No active training run is selected."
            self._set_live_progress_bar(None)
            if not self._has_loss_plot:
                self.loss_plot_html.object = self._plot_message_html("Loss curves will appear after training starts.")
            if not self._has_metric_plot:
                self.metric_plot_html.object = self._plot_message_html("Metric curves will appear after training starts.")
            self.tuning_plot.objects = []
            self.current_trial_plot.objects = []
            self.current_trial_summary.object = "No current Optuna trial."
            self.current_trial_table.object = pd.DataFrame()
            self.epoch_table.object = pd.DataFrame()
            self.tuning_table.object = pd.DataFrame()
            return

        try:
            payload = self.context.artifacts.get(artifact_id)
        except Exception as exc:
            self.summary.object = f"Could not read training log: `{exc}`"
            return

        epochs = payload.get("epochs", []) or []
        epoch_df = self._normalise_epoch_df(pd.DataFrame(epochs))

        if sync_controls:
            self._sync_curve_controls(epoch_df)

        filtered_epoch_df = self._filtered_epoch_df(epoch_df)
        self.epoch_table.object = filtered_epoch_df

        tuning = payload.get("tuning") or {}
        tuning_trials = payload.get("tuning_trials") or []
        tuning_df = pd.DataFrame(tuning_trials)
        self.tuning_table.object = tuning_df
        self.summary.object = self._summary_markdown(payload, tuning, tuning_trials)
        self._render_progress_from_log_payload(payload, artifact_id=str(artifact_id))

        status = str(payload.get("status", "unknown") or "unknown")
        self._last_rendered_status_by_artifact[str(artifact_id)] = status

        if filtered_epoch_df.empty or "epoch" not in filtered_epoch_df.columns:
            message = payload.get("message", "Waiting for epoch data...")

            if not self._has_loss_plot:
                self.loss_plot_html.object = self._plot_message_html(f"{status}: {message}")

            if not self._has_metric_plot:
                self.metric_plot_html.object = self._plot_message_html(
                    "Curves will appear after epoch-level metrics are produced."
                )

            self._render_tuning(payload, tuning, tuning_df)
            return

        latest_epoch_count = len(filtered_epoch_df)
        previous_epoch_count = int(self._last_rendered_epoch_count_by_artifact.get(str(artifact_id), 0))
        finished = status in {"finished", "complete", "completed", "failed", "error", "cancelled"}

        should_update_plots = (
            force_render
            or finished
            or previous_epoch_count <= 1
            or latest_epoch_count - previous_epoch_count >= int(self._event_refresh_epoch_step)
        )

        if not should_update_plots:
            self._render_tuning(payload, tuning, tuning_df)
            return

        loss_keys = list(self.loss_metric_select.value or [])
        metric_keys = list(self.metric_select.value or [])

        new_loss_plot = self._plot(
            filtered_epoch_df,
            keys=loss_keys,
            title="Loss over epochs",
            ylabel="Loss",
            y_scale=str(self.y_scale.value or "auto"),
            x_scale=str(self.x_scale.value or "linear"),
            smoothing_window=int(self.smoothing_window.value or 1),
            show_points=bool(self.show_points.value),
        )
        new_metric_plot = self._plot(
            filtered_epoch_df,
            keys=metric_keys,
            title="Metrics over epochs",
            ylabel="Metric",
            y_scale=str(self.y_scale.value or "auto"),
            x_scale=str(self.x_scale.value or "linear"),
            smoothing_window=int(self.smoothing_window.value or 1),
            show_points=bool(self.show_points.value),
        )

        self.loss_plot_html.object = new_loss_plot
        self.metric_plot_html.object = new_metric_plot
        self._has_loss_plot = True
        self._has_metric_plot = True
        self._last_rendered_epoch_count_by_artifact[str(artifact_id)] = latest_epoch_count

        self._render_tuning(payload, tuning, tuning_df)

    def _render_progress_from_log_payload(
        self,
        payload: Dict[str, Any],
        *,
        artifact_id: Optional[str] = None,
    ) -> None:
        durable_progress = payload.get("progress")
        candidate = dict(durable_progress) if isinstance(durable_progress, dict) else None

        artifact_key = str(artifact_id or "")
        run_key = str(payload.get("run_id") or "")
        status = str(payload.get("status") or "unknown")
        terminal = status.lower() in {
            "complete",
            "completed",
            "finished",
            "paused",
            "cancelled",
            "failed",
            "error",
        }

        if terminal:
            # A terminal durable state must never be replaced by the last cached
            # running event merely because the artifact omitted a progress block.
            if candidate is None:
                candidate = {
                    "status": status,
                    "stage": payload.get("stage") or status,
                    "message": payload.get("message") or f"Run {status}.",
                    "run_id": run_key,
                    "dataset_id": payload.get("dataset_id"),
                    "recipe_id": payload.get("recipe_id"),
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                    "updated_at": payload.get("updated_at") or time.time(),
                }
            if artifact_key:
                self._latest_live_progress_by_artifact.pop(artifact_key, None)
            if run_key:
                self._latest_live_progress_by_run.pop(run_key, None)
        else:
            live_candidate = None
            if artifact_key:
                live_candidate = self._latest_live_progress_by_artifact.get(artifact_key)
            if live_candidate is None and run_key:
                live_candidate = self._latest_live_progress_by_run.get(run_key)

            if isinstance(live_candidate, dict):
                live_updated = float(live_candidate.get("updated_at") or 0.0)
                durable_updated = float((candidate or {}).get("updated_at") or 0.0)
                if candidate is None or live_updated >= durable_updated:
                    candidate = dict(live_candidate)

        if candidate is not None:
            self._render_live_progress(candidate)
            return

        message = str(payload.get("message") or "Waiting for progress updates.")
        self._render_live_progress({
            "status": status,
            "stage": payload.get("stage") or status,
            "message": message,
            "epoch": payload.get("best_epoch"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
        })

    def _render_live_progress(self, payload: Dict[str, Any]) -> None:
        data = dict(payload or {})
        self.live_status.alert_type = progress_alert_type(data)
        self.live_status.object = progress_markdown(data)
        self._set_live_progress_bar(progress_percent(data))

    def _set_live_progress_bar(self, percent: Optional[float]) -> None:
        """Keep the progress slot mounted so status updates cannot move the UI."""
        if percent is None:
            self.live_progress.value = 0
            self.live_progress.styles = {"visibility": "hidden"}
            return
        self.live_progress.value = max(0, min(100, int(round(float(percent)))))
        self.live_progress.styles = {"visibility": "visible"}

    def _normalise_epoch_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested metric dictionaries and coerce plottable columns.

        Older recipe logs may store values under a nested `metrics` dict. Newer
        logs should already have flat columns. This method supports both.
        """
        if df.empty:
            return df

        work = df.copy()

        if "metrics" in work.columns:
            for idx, value in work["metrics"].items():
                if not isinstance(value, dict):
                    continue
                for key, metric_value in value.items():
                    if key not in work.columns or pd.isna(work.at[idx, key]):
                        work.at[idx, key] = metric_value

        if "epoch" not in work.columns and "step" in work.columns:
            work["epoch"] = work["step"]

        if "epoch" in work.columns:
            work["epoch"] = pd.to_numeric(work["epoch"], errors="coerce")

        for column in work.columns:
            if column in {"time", "elapsed_seconds", "step", "total", "epoch"}:
                work[column] = pd.to_numeric(work[column], errors="coerce")
                continue
            if column in {"status", "message", "metrics", "split", "trial_state"}:
                continue

            converted = pd.to_numeric(work[column], errors="coerce")
            if converted.notna().any():
                work[column] = converted

        if "epoch" in work.columns:
            work = work.dropna(subset=["epoch"]).sort_values("epoch")

        return work

    def _numeric_metric_columns(self, df: pd.DataFrame) -> List[str]:
        if df.empty:
            return []

        ignored = {"time", "elapsed_seconds", "step", "total", "epoch", "status", "message", "metrics"}
        columns: List[str] = []

        for column in df.columns:
            if column in ignored:
                continue
            series = pd.to_numeric(df[column], errors="coerce")
            if series.notna().any():
                columns.append(str(column))

        return columns

    def _loss_columns(self, columns: Sequence[str]) -> List[str]:
        preferred = ["train_loss", "validation_loss", "val_loss", "test_loss", "loss"]
        out: List[str] = []

        for key in preferred:
            if key in columns and key not in out:
                out.append(key)

        for key in columns:
            lowered = key.lower()
            if "loss" in lowered and key not in out:
                out.append(key)

        return out

    def _default_metric_columns(self, columns: Sequence[str]) -> List[str]:
        preferred = [
            "val_accuracy",
            "validation_accuracy",
            "test_accuracy",
            "train_accuracy",
            "val_balanced_accuracy",
            "val_f1_macro",
            "val_roc_auc",
            "val_roc_auc_ovr_macro",
            "val_r2",
            "val_mae",
            "val_rmse",
            "learning_rate",
        ]

        out: List[str] = []
        loss_columns = set(self._loss_columns(columns))

        for key in preferred:
            if key in columns and key not in out and key not in loss_columns:
                out.append(key)

        for key in columns:
            lowered = key.lower()
            if key in out or key in loss_columns:
                continue
            if any(
                token in lowered
                for token in [
                    "acc",
                    "f1",
                    "auc",
                    "r2",
                    "mae",
                    "rmse",
                    "precision",
                    "recall",
                    "lr",
                    "learning_rate",
                ]
            ):
                out.append(key)

        return out

    def _sync_curve_controls(self, epoch_df: pd.DataFrame) -> None:
        numeric_columns = self._numeric_metric_columns(epoch_df)
        loss_columns = self._loss_columns(numeric_columns)
        metric_columns = [column for column in numeric_columns if column not in set(loss_columns)]

        previous_loss = list(self.loss_metric_select.value or [])
        previous_metrics = list(self.metric_select.value or [])

        self.loss_metric_select.options = loss_columns
        self.metric_select.options = metric_columns

        if previous_loss:
            self.loss_metric_select.value = [column for column in previous_loss if column in loss_columns]
        else:
            self.loss_metric_select.value = loss_columns[:4]

        if previous_metrics:
            self.metric_select.value = [column for column in previous_metrics if column in metric_columns]
        else:
            self.metric_select.value = self._default_metric_columns(metric_columns)[:6]

        self.curve_help.object = (
            f"Detected `{len(loss_columns)}` loss column(s) and "
            f"`{len(metric_columns)}` other numeric metric column(s).\n"
            "Use the selectors to add or remove plotted curves."
        )

    def _filtered_epoch_df(self, epoch_df: pd.DataFrame) -> pd.DataFrame:
        if epoch_df.empty or "epoch" not in epoch_df.columns:
            return epoch_df

        work = epoch_df.copy()

        try:
            start = int(self.epoch_start.value or 0)
        except Exception:
            start = 0

        try:
            end = int(self.epoch_end.value or 0)
        except Exception:
            end = 0

        if start > 0:
            work = work[work["epoch"] >= start]
        if end > 0:
            work = work[work["epoch"] <= end]

        return work

    def _reset_curve_controls(self) -> None:
        self.loss_metric_select.value = []
        self.metric_select.value = []
        self.y_scale.value = "auto"
        self.x_scale.value = "linear"
        self.smoothing_window.value = 1
        self.epoch_start.value = 0
        self.epoch_end.value = 0
        self.show_points.value = True
        self._render_selected(sync_controls=True, force_render=True)

    def _summary_markdown(
        self,
        payload: Dict[str, Any],
        tuning: Dict[str, Any],
        tuning_trials: List[Dict[str, Any]],
    ) -> str:
        tuning_enabled = bool(tuning.get("enabled") or tuning_trials)
        progress = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
        lines = [
            f"**Model:** {payload.get('model_title', 'unknown')}",
            f"**Run:** `{payload.get('run_id', '')}`",
            f"**Framework:** `{payload.get('framework', '')}`",
            f"**Status:** `{payload.get('status', 'unknown')}`",
            f"**Stage:** `{progress.get('stage_label') or progress.get('stage') or payload.get('stage', '')}`",
            f"**Message:** {progress.get('message') or payload.get('message', '')}",
            f"**Optimised to:** `{payload.get('optimize_metric', tuning.get('metric', ''))}`",
            f"**Best epoch:** `{payload.get('best_epoch', '')}`",
            f"**Optuna:** {'enabled' if tuning_enabled else 'disabled'}",
        ]

        if tuning_enabled:
            lines.append(f"**Optuna metric:** `{tuning.get('metric', '')}`")
            lines.append(f"**Optuna trials:** `{len(tuning_trials)}`")

        return "  \n".join(lines)

    def _render_tuning(self, payload: Dict[str, Any], tuning: Dict[str, Any], tuning_df: pd.DataFrame) -> None:
        current_trial = payload.get("tuning_current_trial") or {}
        current_trial_epochs = payload.get("tuning_current_trial_epochs") or payload.get("tuning_last_trial_epochs") or []
        current_trial_df = pd.DataFrame(current_trial_epochs)

        self._render_current_optuna_trial(
            payload=payload,
            tuning=tuning,
            current_trial=current_trial,
            current_trial_df=current_trial_df,
        )

        if not tuning and tuning_df.empty and not current_trial:
            self.tuning_summary.object = "Optuna tuning was not enabled for this run."
            self.tuning_plot.objects = [pn.pane.Alert("No Optuna trials are available.", alert_type="info")]
            return

        best_params = payload.get("tuning_best_params") or {}
        best_value = payload.get("tuning_best_value")

        if not best_params and not tuning_df.empty and "value" in tuning_df.columns:
            metric = str(tuning.get("metric") or "")
            completed = tuning_df.dropna(subset=["value"]).copy()
            if not completed.empty:
                minimize = any(token in metric.lower() for token in ("loss", "error", "mae", "mse", "rmse"))
                idx = completed["value"].idxmin() if minimize else completed["value"].idxmax()
                row = completed.loc[idx]
                if isinstance(row.get("params"), dict):
                    best_params = row.get("params")
                    best_value = row.get("value")

        lines = [
            "### Optuna tuning",
            f"- **Backend:** `{tuning.get('backend', 'optuna')}`",
            f"- **Metric:** `{tuning.get('metric', '')}`",
            f"- **Configured trials:** `{tuning.get('n_trials', '')}`",
            f"- **Completed/recorded trials:** `{len(tuning_df)}`",
        ]

        if best_value is not None:
            lines.append(f"- **Best value:** `{best_value}`")

        if best_params:
            lines.append("- **Best parameters:**")
            for key, value in best_params.items():
                lines.append(f"  - `{key}` = `{value}`")

        search_space = tuning.get("search_space") or {}
        if search_space:
            lines.append("- **Search space:**")
            for key, spec in search_space.items():
                lines.append(f"  - `{key}`: `{spec}`")

        self.tuning_summary.object = "\n".join(lines)

        if tuning_df.empty or "number" not in tuning_df.columns or "value" not in tuning_df.columns:
            self.tuning_plot.objects = [pn.pane.Alert("No numeric Optuna trial values to plot yet.", alert_type="info")]
            return

        self.tuning_plot.objects = [
            self._plot_trials(
                tuning_df,
                title="Optuna objective value by trial",
                ylabel=str(tuning.get("metric") or "objective"),
            )
        ]

    def _render_current_optuna_trial(
        self,
        *,
        payload: Dict[str, Any],
        tuning: Dict[str, Any],
        current_trial: Dict[str, Any],
        current_trial_df: pd.DataFrame,
    ) -> None:
        status = str(payload.get("status", "unknown"))
        message = str(payload.get("message", ""))

        if not current_trial and current_trial_df.empty:
            if status in {"tuning", "running", "queued"}:
                self.current_trial_summary.object = (
                    f"**Status:** `{status}`  \n"
                    f"**Message:** {message or 'Waiting for Optuna trial progress...'}"
                )
                self.current_trial_plot.objects = [
                    pn.pane.Alert(
                        "Waiting for current-trial epoch metrics.\n"
                        "For ResNet tuning, this should update after each trial epoch.\n"
                        "If it does not, the image tuning code is not publishing "
                        "`tuning_current_trial_epochs` yet.",
                        alert_type="warning",
                    )
                ]
            else:
                self.current_trial_summary.object = "No current Optuna trial."
                self.current_trial_plot.objects = [pn.pane.Alert("No current-trial epoch data.", alert_type="info")]
            self.current_trial_table.object = pd.DataFrame()
            return

        lines = [
            f"**Trial:** `{current_trial.get('display_number', current_trial.get('number', ''))}`"
            + (
                f" / `{current_trial.get('total_trials')}`"
                if current_trial.get("total_trials") is not None
                else ""
            ),
            f"**State:** `{current_trial.get('state', status)}`",
            f"**Metric:** `{current_trial.get('metric', tuning.get('metric', ''))}`",
        ]

        if current_trial.get("epoch") is not None and current_trial.get("total_epochs") is not None:
            lines.append(f"**Epoch progress:** `{current_trial.get('epoch')}` / `{current_trial.get('total_epochs')}`")

        if current_trial.get("value") is not None:
            lines.append(f"**Latest objective value:** `{current_trial.get('value')}`")

        if current_trial.get("message"):
            lines.append(f"**Message:** {current_trial.get('message')}")
        elif message:
            lines.append(f"**Message:** {message}")

        params = current_trial.get("params") or {}
        if isinstance(params, dict) and params:
            lines.append("**Trial parameters:**")
            for key, value in params.items():
                lines.append(f"- `{key}` = `{value}`")

        self.current_trial_summary.object = "  \n".join(lines)
        self.current_trial_table.object = current_trial_df

        if current_trial_df.empty:
            self.current_trial_plot.objects = [
                pn.pane.Alert(
                    "Current trial is running.\nWaiting for the first epoch to finish...",
                    alert_type="info",
                )
            ]
            return

        self.current_trial_plot.objects = [
            self._plot_current_trial_epochs(
                current_trial_df,
                metric=str(current_trial.get("metric") or tuning.get("metric") or "objective"),
            )
        ]

    def _plot(
        self,
        df: pd.DataFrame,
        *,
        keys: List[str],
        title: str,
        ylabel: str,
        y_scale: str = "auto",
        x_scale: str = "linear",
        smoothing_window: int = 1,
        show_points: bool = True,
    ) -> str:
        try:
            from matplotlib.figure import Figure
        except Exception as exc:
            return self._plot_message_html(f"Matplotlib is not available: {exc}")

        if df.empty or "epoch" not in df.columns:
            return self._plot_message_html("No epoch data available.")

        available = []
        for key in keys:
            if key not in df.columns:
                continue
            series = pd.to_numeric(df[key], errors="coerce")
            if series.notna().any():
                available.append(key)

        if not available:
            return self._plot_message_html("No selected metric columns contain numeric data.")

        fig = Figure(figsize=(7.4, 3.4), facecolor="white")
        ax = fig.subplots()

        x = pd.to_numeric(df["epoch"], errors="coerce")
        marker = "o" if show_points else None
        plotted_series = []

        for key in available:
            y = pd.to_numeric(df[key], errors="coerce")
            if smoothing_window and smoothing_window > 1:
                y_plot = y.rolling(window=int(smoothing_window), min_periods=1, center=False).mean()
                label = f"{key} — smoothed {int(smoothing_window)}"
            else:
                y_plot = y
                label = key

            valid = x.notna() & y_plot.notna()
            if not valid.any():
                continue

            ax.plot(x[valid], y_plot[valid], marker=marker, linewidth=1.5, label=label)
            plotted_series.append(y_plot[valid])

        if not plotted_series:
            return self._plot_message_html("Selected metric columns did not contain plottable values.")

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        if x_scale == "log":
            try:
                if x.dropna().min() > 0:
                    ax.set_xscale("log")
                else:
                    return self._plot_message_html("Log x-axis requires epoch values greater than zero.")
            except Exception:
                pass

        try:
            y_values = pd.concat(plotted_series, ignore_index=True).dropna()
        except Exception:
            y_values = pd.Series(dtype=float)

        if y_scale == "log":
            try:
                if not y_values.empty and y_values.min() > 0:
                    ax.set_yscale("log")
                else:
                    return self._plot_message_html(
                        "Log y-axis requires all selected plotted values to be greater than zero."
                    )
            except Exception:
                pass
        elif y_scale == "auto":
            try:
                if not y_values.empty and y_values.min() > 0 and "loss" in ylabel.lower():
                    ax.set_yscale("log")
            except Exception:
                pass

        fig.tight_layout()
        return self._figure_to_img_html(fig, alt=title)

    def _plot_current_trial_epochs(self, df: pd.DataFrame, *, metric: str):
        try:
            from matplotlib.figure import Figure
        except Exception as exc:
            return pn.pane.Alert(f"Matplotlib is not available: {exc}", alert_type="danger")

        if df.empty:
            return pn.pane.Alert("No current-trial epoch metrics to plot.", alert_type="info")

        x_key = "epoch" if "epoch" in df.columns else "step" if "step" in df.columns else None
        if not x_key:
            return pn.pane.Alert("Current-trial rows do not include epoch or step columns.", alert_type="warning")

        preferred = [
            "value",
            metric,
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
            "val_balanced_accuracy",
            "val_f1_macro",
            "val_roc_auc",
            "val_roc_auc_ovr_macro",
            "val_r2",
            "val_mae",
            "val_rmse",
        ]

        available = []
        for key in preferred:
            if key in df.columns and key not in available:
                try:
                    series = pd.to_numeric(df[key], errors="coerce")
                    if series.notna().any():
                        available.append(key)
                except Exception:
                    pass

        if not available:
            return pn.pane.Alert("No numeric current-trial metric columns found.", alert_type="warning")

        work = df.copy()
        fig = Figure(figsize=(7, 3.2))
        ax = fig.subplots()

        for key in available:
            y = pd.to_numeric(work[key], errors="coerce")
            ax.plot(work[x_key], y, marker="o", linewidth=1.5, label=key)

        title = "Current Optuna trial performance"
        if "trial_display_number" in work.columns and work["trial_display_number"].notna().any():
            try:
                title += f" — trial {int(work['trial_display_number'].dropna().iloc[-1])}"
            except Exception:
                pass

        ax.set_title(title)
        ax.set_xlabel("Epoch" if x_key == "epoch" else "Step")
        ax.set_ylabel(metric or "Objective")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        try:
            y_values = pd.concat(
                [pd.to_numeric(work[key], errors="coerce") for key in available],
                ignore_index=True,
            ).dropna()
            y_scale = str(self.y_scale.value or "auto")
            if y_scale == "log":
                if not y_values.empty and y_values.min() > 0:
                    ax.set_yscale("log")
            elif y_scale == "auto":
                if not y_values.empty and y_values.min() > 0 and "loss" in metric.lower():
                    ax.set_yscale("log")
        except Exception:
            pass

        fig.tight_layout()
        return pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_width", height=330)

    def _plot_trials(self, df: pd.DataFrame, *, title: str, ylabel: str):
        try:
            from matplotlib.figure import Figure
        except Exception as exc:
            return pn.pane.Alert(f"Matplotlib is not available: {exc}", alert_type="danger")

        work = df.copy()
        work = work[pd.notna(work.get("value"))]
        if work.empty:
            return pn.pane.Alert("No completed Optuna trials to plot.", alert_type="info")

        fig = Figure(figsize=(7, 3.2))
        ax = fig.subplots()

        ax.plot(work["number"], work["value"], marker="o", linewidth=1.5, label="trial value")

        try:
            running_best = []
            metric = ylabel.lower()
            minimize = any(token in metric for token in ("loss", "error", "mae", "mse", "rmse"))
            best = None

            for value in work["value"]:
                value = float(value)
                if best is None:
                    best = value
                elif minimize:
                    best = min(best, value)
                else:
                    best = max(best, value)
                running_best.append(best)

            ax.plot(work["number"], running_best, marker=None, linewidth=1.2, label="best so far")
        except Exception:
            pass

        ax.set_title(title)
        ax.set_xlabel("Trial")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        try:
            y_values = pd.to_numeric(work["value"], errors="coerce").dropna()
            y_scale = str(self.y_scale.value or "auto")
            if y_scale == "log":
                if not y_values.empty and y_values.min() > 0:
                    ax.set_yscale("log")
            elif y_scale == "auto":
                if not y_values.empty and y_values.min() > 0 and "loss" in ylabel.lower():
                    ax.set_yscale("log")
        except Exception:
            pass

        fig.tight_layout()
        return pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_width", height=330)
