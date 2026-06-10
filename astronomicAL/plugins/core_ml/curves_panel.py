from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
import panel as pn


class MLTrainingCurvesPanel:
    """Panel for viewing training curves and Optuna trial history."""

    def __init__(self, *, context: Any, restore_state: Optional[Dict[str, Any]] = None) -> None:
        self.context = context

        self.log_select = pn.widgets.Select(name="", options={})
        self.refresh = pn.widgets.Button(name="Refresh logs", button_type="light")

        self.summary = pn.pane.Markdown("")
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

        self.refresh.on_click(lambda *_: self._load_logs())
        self.log_select.param.watch(lambda *_: self._render_selected(), "value")

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
            self.refresh,
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
        return {"training_log_artifact_id": self.log_select.value}

    def restore_state(self, state: Dict[str, Any]) -> None:
        artifact_id = state.get("training_log_artifact_id")
        if artifact_id in self.log_select.options.values():
            self.log_select.value = artifact_id

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

    def _render_selected(self) -> None:
        artifact_id = self.log_select.value

        if not artifact_id:
            self.summary.object = "No training logs yet. Train a model first."
            self.loss_plot.objects = []
            self.metric_plot.objects = []
            self.tuning_plot.objects = []
            self.epoch_table.object = pd.DataFrame()
            self.tuning_table.object = pd.DataFrame()
            return

        try:
            payload = self.context.artifacts.get(artifact_id)
        except Exception as exc:
            self.summary.object = f"Could not read training log: `{exc}`"
            return

        epochs = payload.get("epochs", []) or []
        epoch_df = pd.DataFrame(epochs)
        self.epoch_table.object = epoch_df

        tuning = payload.get("tuning") or {}
        tuning_trials = payload.get("tuning_trials") or []
        tuning_df = pd.DataFrame(tuning_trials)
        self.tuning_table.object = tuning_df

        self.summary.object = self._summary_markdown(payload, tuning, tuning_trials)

        if epoch_df.empty or "epoch" not in epoch_df.columns:
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
            self.loss_plot.objects = [
                self._plot(
                    epoch_df,
                    keys=["train_loss", "val_loss"],
                    title="Loss over epochs",
                    ylabel="Loss",
                )
            ]
            self.metric_plot.objects = [
                self._plot(
                    epoch_df,
                    keys=[
                        "val_accuracy",
                        "val_balanced_accuracy",
                        "val_f1_macro",
                        "val_r2",
                        "val_mae",
                        "val_rmse",
                    ],
                    title="Validation metrics over epochs",
                    ylabel="Metric",
                )
            ]

        self._render_tuning(payload, tuning, tuning_df)

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
        if not tuning and tuning_df.empty:
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

    def _plot(self, df: pd.DataFrame, *, keys: List[str], title: str, ylabel: str):
        try:
            from matplotlib.figure import Figure
        except Exception as exc:
            return pn.pane.Alert(f"Matplotlib is not available: {exc}", alert_type="danger")

        available = [key for key in keys if key in df.columns and df[key].notna().any()]
        if not available:
            return pn.pane.Alert("No compatible metric columns found.", alert_type="warning")

        fig = Figure(figsize=(7, 3.2))
        ax = fig.subplots()

        for key in available:
            ax.plot(df["epoch"], df[key], marker="o", linewidth=1.5, label=key)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        try:
            y_values = pd.concat([df[key] for key in available], ignore_index=True).dropna()
            if not y_values.empty and y_values.min() > 0 and y_values.min() < 0.01:
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

        fig.tight_layout()
        return pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_width", height=330)