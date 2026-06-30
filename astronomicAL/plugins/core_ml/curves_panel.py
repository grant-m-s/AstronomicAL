#BUG: Doesnt provide result metrics even when groundtruth provided
#BUG: Flicker When training round is fast

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import panel as pn


class MLTrainingCurvesPanel:
    """Panel for viewing training curves and Optuna trial history."""

    def __init__(self, *, context: Any, restore_state: Optional[Dict[str, Any]] = None) -> None:
        self.context = context

        self.log_select = pn.widgets.Select(name="", options={})
        self.refresh = pn.widgets.Button(name="Refresh logs", button_type="light")
        self.reset_curve_controls = pn.widgets.Button(
            name="Reset metric selections",
            button_type="light",
        )

        self.summary = pn.pane.Markdown("")

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
            options={
                "Auto": "auto",
                "Linear": "linear",
                "Log": "log",
            },
            value="auto",
            sizing_mode="stretch_width",
        )

        self.x_scale = pn.widgets.Select(
            name="",
            options={
                "Linear": "linear",
                "Log": "log",
            },
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

        self.loss_plot = pn.Column(sizing_mode="stretch_width")
        self.metric_plot = pn.Column(sizing_mode="stretch_width")

        self.epoch_table = pn.pane.DataFrame(
            pd.DataFrame(),
            height=260,
            sizing_mode="stretch_width",
        )

        self.tuning_summary = pn.pane.Markdown(
            "No tuning information for the selected run.",
            sizing_mode="stretch_width",
        )
        self.current_trial_summary = pn.pane.Markdown(
            "No current Optuna trial.",
            sizing_mode="stretch_width",
        )
        self.current_trial_plot = pn.Column(sizing_mode="stretch_width")
        self.current_trial_table = pn.pane.DataFrame(
            pd.DataFrame(),
            height=220,
            sizing_mode="stretch_width",
        )
        self.tuning_plot = pn.Column(sizing_mode="stretch_width")
        self.tuning_table = pn.pane.DataFrame(
            pd.DataFrame(),
            height=280,
            sizing_mode="stretch_width",
        )

        self.log_select.sizing_mode = "stretch_width"
        self.log_select.height = 38

        self.refresh.sizing_mode = "stretch_width"
        self.refresh.height = 34

        self.reset_curve_controls.sizing_mode = "stretch_width"
        self.reset_curve_controls.height = 34

        self.refresh.on_click(lambda *_: self._load_logs())
        self.reset_curve_controls.on_click(lambda *_: self._reset_curve_controls())

        self.log_select.param.watch(lambda *_: self._render_selected(), "value")

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
            widget.param.watch(lambda *_: self._render_selected(sync_controls=False), "value")

        self._subscriptions = []
        self._subscribe_to_training_log_events()
        self._load_logs()

        if restore_state:
            self.restore_state(restore_state)

    def _subscribe_to_training_log_events(self) -> None:
        events = getattr(self.context, "events", None)
        subscribe = getattr(events, "subscribe", None)
        if not callable(subscribe):
            return

        for topic in ["ml.training_log.created", "ml.training_log.updated"]:
            try:
                sub = subscribe(
                    topic,
                    self._on_training_log_event,
                    owner_label="ML Training Curves",
                    owner_kind="panel",
                )
                self._subscriptions.append(sub)
            except Exception:
                pass

    def _on_training_log_event(self, topic: str, payload: Any) -> None:
        artifact_id = None
        if isinstance(payload, dict):
            artifact_id = payload.get("artifact_id") or payload.get("training_log_artifact_id")

        def update():
            self._load_logs(select_artifact_id=artifact_id)

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(update)
                return
        except Exception:
            pass

        update()

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
            "<h3 style='margin:0'>ML training curves</h3>",
            height=32,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        selector_block = pn.Column(
            header,
            self._field("Training log", self.log_select),
            pn.Row(
                self.refresh,
                self.reset_curve_controls,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 10, 0),
            styles={
                "box-sizing": "border-box",
                "overflow": "visible",
            },
        )

        summary_tab = pn.Column(
            self.summary,
            sizing_mode="stretch_width",
            styles={
                "box-sizing": "border-box",
                "padding": "8px 4px 14px 0",
                "overflow": "visible",
            },
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
            styles={
                "box-sizing": "border-box",
                "padding": "8px 4px 14px 0",
                "overflow": "visible",
            },
        )

        loss_tab = pn.Column(
            self.loss_plot,
            sizing_mode="stretch_width",
            styles={
                "box-sizing": "border-box",
                "padding": "8px 4px 14px 0",
                "overflow": "visible",
            },
        )

        metrics_tab = pn.Column(
            self.metric_plot,
            sizing_mode="stretch_width",
            styles={
                "box-sizing": "border-box",
                "padding": "8px 4px 14px 0",
                "overflow": "visible",
            },
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
            styles={
                "box-sizing": "border-box",
                "padding": "8px 4px 14px 0",
                "overflow": "visible",
            },
        )

        epochs_tab = pn.Column(
            self.epoch_table,
            sizing_mode="stretch_width",
            styles={
                "box-sizing": "border-box",
                "padding": "8px 4px 14px 0",
                "overflow": "visible",
            },
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
            styles={
                "box-sizing": "border-box",
                "overflow": "visible",
            },
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

        # Metric selections are restored after the selected log has populated
        # available metric options.
        self._render_selected(sync_controls=True)

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

        self._render_selected(sync_controls=False)

    def _field(self, label: str, widget):
        return pn.Column(
            pn.pane.HTML(
                f"<div style='font-size:12px;font-weight:600;margin-bottom:2px'>{label}</div>",
                height=18,
                sizing_mode="stretch_width",
                margin=(0, 0, 2, 0),
            ),
            widget,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
            styles={
                "box-sizing": "border-box",
                "overflow": "visible",
            },
        )

    def _load_logs(self, select_artifact_id: Optional[str] = None) -> None:
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
        self.log_select.options = options

        if select_artifact_id in options.values():
            self.log_select.value = select_artifact_id
        elif current in options.values():
            self.log_select.value = current
        elif options:
            self.log_select.value = next(iter(options.values()))
        else:
            self.log_select.value = None

        self._render_selected()

    def _render_selected(self, sync_controls: bool = True) -> None:
        artifact_id = self.log_select.value

        if not artifact_id:
            self.summary.object = "No training logs yet. Train a model first."
            self.loss_plot.objects = []
            self.metric_plot.objects = []
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

        if filtered_epoch_df.empty or "epoch" not in filtered_epoch_df.columns:
            status = payload.get("status", "unknown")
            message = payload.get("message", "Waiting for epoch data...")
            self.loss_plot.objects = [
                pn.pane.Alert(
                    f"{status}: {message}",
                    alert_type="info" if status in {"queued", "running", "tuning"} else "warning",
                )
            ]
            self.metric_plot.objects = [
                pn.pane.Alert(
                    "Curves will appear after epoch-level metrics are produced.",
                    alert_type="info",
                )
            ]
        else:
            loss_keys = list(self.loss_metric_select.value or [])
            metric_keys = list(self.metric_select.value or [])

            self.loss_plot.objects = [
                self._plot(
                    filtered_epoch_df,
                    keys=loss_keys,
                    title="Loss over epochs",
                    ylabel="Loss",
                    y_scale=str(self.y_scale.value or "auto"),
                    x_scale=str(self.x_scale.value or "linear"),
                    smoothing_window=int(self.smoothing_window.value or 1),
                    show_points=bool(self.show_points.value),
                )
            ]

            self.metric_plot.objects = [
                self._plot(
                    filtered_epoch_df,
                    keys=metric_keys,
                    title="Metrics over epochs",
                    ylabel="Metric",
                    y_scale=str(self.y_scale.value or "auto"),
                    x_scale=str(self.x_scale.value or "linear"),
                    smoothing_window=int(self.smoothing_window.value or 1),
                    show_points=bool(self.show_points.value),
                )
            ]

        self._render_tuning(payload, tuning, tuning_df)

    def _normalise_epoch_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested metric dictionaries and coerce plottable columns.

        Older recipe logs may store values under a nested `metrics` dict.
        Newer logs should already have flat columns. This method supports both.
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
            if column in {
                "time",
                "elapsed_seconds",
                "step",
                "total",
                "epoch",
            }:
                work[column] = pd.to_numeric(work[column], errors="coerce")
                continue

            if column in {
                "status",
                "message",
                "metrics",
                "split",
                "trial_state",
            }:
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

        ignored = {
            "time",
            "elapsed_seconds",
            "step",
            "total",
            "epoch",
            "status",
            "message",
            "metrics",
        }

        columns: List[str] = []
        for column in df.columns:
            if column in ignored:
                continue
            series = pd.to_numeric(df[column], errors="coerce")
            if series.notna().any():
                columns.append(str(column))

        return columns


    def _loss_columns(self, columns: Sequence[str]) -> List[str]:
        preferred = [
            "train_loss",
            "validation_loss",
            "val_loss",
            "test_loss",
            "loss",
        ]

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
            if any(token in lowered for token in ["acc", "f1", "auc", "r2", "mae", "rmse", "precision", "recall", "lr", "learning_rate"]):
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
            self.loss_metric_select.value = [
                column for column in previous_loss if column in loss_columns
            ]
        else:
            self.loss_metric_select.value = loss_columns[:4]

        if previous_metrics:
            self.metric_select.value = [
                column for column in previous_metrics if column in metric_columns
            ]
        else:
            self.metric_select.value = self._default_metric_columns(metric_columns)[:6]

        self.curve_help.object = (
            f"Detected `{len(loss_columns)}` loss column(s) and "
            f"`{len(metric_columns)}` other numeric metric column(s). "
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
        self._render_selected(sync_controls=True)

    def _summary_markdown(
        self,
        payload: Dict[str, Any],
        tuning: Dict[str, Any],
        tuning_trials: List[Dict[str, Any]],
    ) -> str:
        tuning_enabled = bool(tuning.get("enabled") or tuning_trials)

        lines = [
            f"**Model:** {payload.get('model_title', 'unknown')}",
            f"**Run:** `{payload.get('run_id', '')}`",
            f"**Framework:** `{payload.get('framework', '')}`",
            f"**Status:** `{payload.get('status', 'unknown')}`",
            f"**Message:** {payload.get('message', '')}",
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
        current_trial_epochs = (
            payload.get("tuning_current_trial_epochs")
            or payload.get("tuning_last_trial_epochs")
            or []
        )
        current_trial_df = pd.DataFrame(current_trial_epochs)

        self._render_current_optuna_trial(
            payload=payload,
            tuning=tuning,
            current_trial=current_trial,
            current_trial_df=current_trial_df,
        )

        if not tuning and tuning_df.empty and not current_trial:
            self.tuning_summary.object = "Optuna tuning was not enabled for this run."
            self.tuning_plot.objects = [
                pn.pane.Alert("No Optuna trials are available.", alert_type="info")
            ]
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
            self.tuning_plot.objects = [
                pn.pane.Alert("No numeric Optuna trial values to plot yet.", alert_type="info")
            ]
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
                        "Waiting for current-trial epoch metrics. "
                        "For ResNet tuning, this should update after each trial epoch. "
                        "If it does not, the image tuning code is not publishing "
                        "`tuning_current_trial_epochs` yet.",
                        alert_type="warning",
                    )
                ]
            else:
                self.current_trial_summary.object = "No current Optuna trial."
                self.current_trial_plot.objects = [
                    pn.pane.Alert("No current-trial epoch data.", alert_type="info")
                ]

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
            lines.append(
                f"**Epoch progress:** `{current_trial.get('epoch')}` / `{current_trial.get('total_epochs')}`"
            )

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
                    "Current trial is running. Waiting for the first epoch to finish...",
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
    ):
        try:
            from matplotlib.figure import Figure
        except Exception as exc:
            return pn.pane.Alert(f"Matplotlib is not available: {exc}", alert_type="danger")

        if df.empty or "epoch" not in df.columns:
            return pn.pane.Alert("No epoch data available.", alert_type="info")

        available = []
        for key in keys:
            if key not in df.columns:
                continue
            series = pd.to_numeric(df[key], errors="coerce")
            if series.notna().any():
                available.append(key)

        if not available:
            return pn.pane.Alert(
                "No selected metric columns contain numeric data.",
                alert_type="warning",
            )

        fig = Figure(figsize=(7.4, 3.4))
        ax = fig.subplots()

        x = pd.to_numeric(df["epoch"], errors="coerce")
        marker = "o" if show_points else None

        plotted_series = []

        for key in available:
            y = pd.to_numeric(df[key], errors="coerce")

            if smoothing_window and smoothing_window > 1:
                y_plot = y.rolling(
                    window=int(smoothing_window),
                    min_periods=1,
                    center=False,
                ).mean()
                label = f"{key} — smoothed {int(smoothing_window)}"
            else:
                y_plot = y
                label = key

            valid = x.notna() & y_plot.notna()
            if not valid.any():
                continue

            ax.plot(
                x[valid],
                y_plot[valid],
                marker=marker,
                linewidth=1.5,
                label=label,
            )
            plotted_series.append(y_plot[valid])

        if not plotted_series:
            return pn.pane.Alert(
                "Selected metric columns did not contain plottable values.",
                alert_type="warning",
            )

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
                    return pn.pane.Alert(
                        "Log x-axis requires epoch values greater than zero.",
                        alert_type="warning",
                    )
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
                    return pn.pane.Alert(
                        "Log y-axis requires all selected plotted values to be greater than zero.",
                        alert_type="warning",
                    )
            except Exception:
                pass
        elif y_scale == "auto":

            try:
                if not y_values.empty and y_values.min() > 0 and "loss" in ylabel.lower():
                    ax.set_yscale("log")
            except Exception as e:
                print(f"y_scale auto exception - {e}")
                pass

        fig.tight_layout()
        return pn.pane.Matplotlib(
            fig,
            tight=True,
            sizing_mode="stretch_width",
            height=350,
        )


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