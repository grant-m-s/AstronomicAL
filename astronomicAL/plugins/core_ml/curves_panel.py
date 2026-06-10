from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
import panel as pn


class MLTrainingCurvesPanel:
    """Panel for viewing loss/accuracy/F1 curves from ml.training_log artifacts."""

    def __init__(self, *, context: Any, restore_state: Optional[Dict[str, Any]] = None) -> None:
        self.context = context
        self.log_select = pn.widgets.Select(name="", options={})
        self.refresh = pn.widgets.Button(name="Refresh logs", button_type="light")
        self.summary = pn.pane.Markdown("")
        self.loss_plot = pn.Column(sizing_mode="stretch_width")
        self.metric_plot = pn.Column(sizing_mode="stretch_width")
        self.table = pn.pane.DataFrame(pd.DataFrame(), height=260, sizing_mode="stretch_width")

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

    def panel(self):
        return pn.Column(
            pn.pane.HTML("<h3 style='margin:0 0 8px 0;'>ML training curves</h3>"),
            self._field("Training log", self.log_select),
            self.refresh,
            self.summary,
            pn.Tabs(
                ("Loss", self.loss_plot),
                ("Accuracy / F1 / R²", self.metric_plot),
                ("Table", self.table),
                dynamic=True,
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
            styles={
                "box-sizing": "border-box",
                "padding": "10px 14px 14px 14px",
                "overflow-y": "auto",
                "overflow-x": "hidden",
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
                f"<div style='font-size:12px;font-weight:600;margin:0 0 3px 0;'>{label}</div>",
                height=18,
                sizing_mode="stretch_width",
            ),
            widget,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
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
            created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ref.created_at))
            label = f"{created} | {model_title} | {dataset_id} | {run_id[:8]}"
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
            self.summary.object = "No training logs yet. Train a torch model first."
            self.loss_plot.objects = []
            self.metric_plot.objects = []
            self.table.object = pd.DataFrame()
            return

        try:
            payload = self.context.artifacts.get(artifact_id)
        except Exception as exc:
            self.summary.object = f"Could not read training log: `{exc}`"
            return

        epochs = payload.get("epochs", [])
        df = pd.DataFrame(epochs)

        self.summary.object = (
            f"**Model:** {payload.get('model_title', 'unknown')}  \n"
            f"**Run:** `{payload.get('run_id', '')}`  \n"
            f"**Framework:** `{payload.get('framework', '')}`  \n"
            f"**Status:** `{payload.get('status', 'unknown')}`  \n"
            f"**Message:** {payload.get('message', '')}  \n"
            f"**Optimised to:** `{payload.get('optimize_metric', '')}`  \n"
            f"**Best epoch:** `{payload.get('best_epoch', '')}`"
        )

        self.table.object = df

        if df.empty or "epoch" not in df.columns:
            status = payload.get("status", "unknown")
            message = payload.get("message", "Waiting for epoch data...")

            self.loss_plot.objects = [
                pn.pane.Alert(
                    f"{status}: {message}",
                    alert_type="info" if status in {"queued", "running"} else "warning",
                )
            ]
            self.metric_plot.objects = [
                pn.pane.Alert(
                    "Training has started. Curves will appear after the first epoch finishes.",
                    alert_type="info",
                )
            ]
            return

        self.loss_plot.objects = [
            self._plot(
                df,
                keys=["train_loss", "val_loss"],
                title="Loss over epochs",
                ylabel="Loss",
            )
        ]

        self.metric_plot.objects = [
            self._plot(
                df,
                keys=["val_accuracy", "val_f1_macro", "val_r2", "val_mae"],
                title="Validation metrics over epochs",
                ylabel="Metric",
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
        if min(df[key]) < 0.01:
            ax.set_yscale('log')
        fig.tight_layout()

        return pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_width", height=330)