from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import panel as pn

from ..resume import (
    discover_model_manifests,
    discover_resume_manifests,
    import_model_manifest,
    storage_locations,
)
from ..runtime import publish


class MLModelManagerPanel:
    """User-facing inventory and import surface for durable ML files."""

    def __init__(self, *, context: Any, restore_state: Optional[Dict[str, Any]] = None) -> None:
        self.context = context
        self._subscriptions: List[Any] = []

        self.storage_root = pn.widgets.TextInput(
            name="ML artifact root",
            value=storage_locations(context)["artifact_root"],
            sizing_mode="stretch_width",
        )
        self.refresh_button = pn.widgets.Button(name="Refresh catalogue", button_type="light")
        self.scan_button = pn.widgets.Button(name="Find saved manifests", button_type="primary")

        self.model = pn.widgets.Select(name="Loaded model", options={}, sizing_mode="stretch_width")
        self.select_for_prediction_button = pn.widgets.Button(
            name="Use selected model in Predictor",
            button_type="success",
            disabled=True,
            sizing_mode="stretch_width",
        )

        self.manifest = pn.widgets.Select(name="Saved model manifest", options={}, sizing_mode="stretch_width")
        self.manifest_path = pn.widgets.TextInput(
            name="Manifest path",
            placeholder="/path/to/model_manifest.json",
            sizing_mode="stretch_width",
        )

        self.trust_external_files = pn.widgets.Checkbox(
            name="Trust files outside the ML artifact root",
            value=False,
            sizing_mode="stretch_width",
        )

        self.load_manifest_button = pn.widgets.Button(
            name="Load model manifest",
            button_type="primary",
            sizing_mode="stretch_width",
        )

        self.resume_checkpoint = pn.widgets.Select(
            name="Paused training checkpoint",
            options={},
            sizing_mode="stretch_width",
        )
        self.select_for_resume_button = pn.widgets.Button(
            name="Resume selected checkpoint in Recipe Launcher",
            button_type="warning",
            disabled=True,
            sizing_mode="stretch_width",
        )

        self.storage_summary = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.model_details = pn.pane.JSON({}, depth=4, sizing_mode="stretch_both")
        self.status = pn.pane.Alert("Model catalogue ready.", alert_type="info", sizing_mode="stretch_width")

        self.refresh_button.on_click(lambda *_: self.refresh())
        self.scan_button.on_click(lambda *_: self.scan())
        self.load_manifest_button.on_click(lambda *_: self.load_manifest())
        self.select_for_prediction_button.on_click(lambda *_: self.select_for_prediction())
        self.select_for_resume_button.on_click(lambda *_: self.select_for_resume())
        self.model.param.watch(lambda *_: self._on_model_change(), "value")
        self.manifest.param.watch(lambda *_: self._on_manifest_change(), "value")
        self.resume_checkpoint.param.watch(lambda *_: self._on_resume_change(), "value")

        self._subscribe()
        self.refresh()
        self.scan()
        if restore_state:
            self.restore_state(restore_state)

    def panel(self):
        loaded = pn.Column(
            pn.pane.Markdown("### Loaded models"),
            self.model,
            self.select_for_prediction_button,
            pn.pane.Markdown(
                "Models listed here are registered `ml.model` artifacts and are immediately available to the Predictor."
            ),
            sizing_mode="stretch_width",
        )
        disk = pn.Column(
            pn.pane.Markdown("### Load a saved model"),
            self.manifest,
            self.manifest_path,
            self.trust_external_files,
            pn.pane.Alert(
                "Model and checkpoint files use Python deserialisation. "
                "Only load files created locally or obtained from a "
                "trusted source.",
                alert_type="warning",
                sizing_mode="stretch_width",
            ),
            self.load_manifest_button,
            pn.pane.Markdown(
                "Load the JSON `model_manifest.json` or paused "
                "`*.model_manifest.json`; the referenced model file "
                "stays on disk."
            ),
            sizing_mode="stretch_width",
        )
        paused = pn.Column(
            pn.pane.Markdown("### Paused training"),
            self.resume_checkpoint,
            self.select_for_resume_button,
            pn.pane.Markdown(
                "A paused checkpoint contains model, optimizer, scheduler, best-epoch, RNG, split, parameters, and log state."
            ),
            sizing_mode="stretch_width",
        )
        return pn.Column(
            pn.pane.Markdown("## ML Models and Checkpoints"),
            pn.Row(self.storage_root, self.refresh_button, self.scan_button, sizing_mode="stretch_width"),
            self.storage_summary,
            pn.Tabs(("Loaded", loaded), ("Load from disk", disk), ("Paused runs", paused), dynamic=True),
            self.status,
            pn.pane.Markdown("### Selected model details"),
            self.model_details,
            sizing_mode="stretch_both",
            scroll=True,
            styles={"padding": "10px"},
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "storage_root": self.storage_root.value,
            "model": self.model.value,
            "manifest_path": self.manifest_path.value,
            "resume_checkpoint": self.resume_checkpoint.value,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, Mapping):
            return
        if state.get("storage_root"):
            self.storage_root.value = str(state["storage_root"])
            self.scan()
        if state.get("model") in self._option_values(self.model.options):
            self.model.value = state["model"]
        if state.get("manifest_path"):
            self.manifest_path.value = str(state["manifest_path"])
        if state.get("resume_checkpoint") in self._option_values(self.resume_checkpoint.options):
            self.resume_checkpoint.value = state["resume_checkpoint"]

    def dispose(self) -> None:
        events = getattr(self.context, "events", None)
        unsubscribe = getattr(events, "unsubscribe", None)
        if callable(unsubscribe):
            for sub in list(self._subscriptions):
                try:
                    unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions.clear()

    def refresh(self) -> None:
        catalog = self._catalog()
        if catalog is not None:
            try:
                catalog.refresh()
                self.model.options = catalog.as_options()
            except Exception as exc:
                self.status.alert_type = "danger"
                self.status.object = f"Could not refresh the model catalogue: {exc}"
        self._refresh_resume_artifacts()
        self._update_storage_summary()
        self._on_model_change()

    def scan(self) -> None:
        params = {"ml_artifact_dir": self.storage_root.value}
        try:
            model_paths = discover_model_manifests(self.context, params)
            resume_paths = discover_resume_manifests(self.context, params)
        except Exception as exc:
            self.status.alert_type = "danger"
            self.status.object = f"Could not scan the ML artifact root: {exc}"
            return
        self.manifest.options = {self._path_label(path): path for path in model_paths}
        disk_resume = {f"disk — {self._path_label(path)}": f"manifest:{path}" for path in resume_paths}
        current = dict(self.resume_checkpoint.options) if isinstance(self.resume_checkpoint.options, dict) else {}
        current.update(disk_resume)
        self.resume_checkpoint.options = current
        self.status.alert_type = "success"
        self.status.object = (
            f"Found {len(model_paths)} model manifest(s) and {len(resume_paths)} paused checkpoint manifest(s) "
            f"under `{self.storage_root.value}`."
        )
        self._update_storage_summary()

    def load_manifest(self) -> None:
        path = str(self.manifest_path.value or self.manifest.value or "").strip()
        if not path:
            self.status.alert_type = "warning"
            self.status.object = "Choose or enter a model manifest path."
            return
        try:
            result = import_model_manifest(
                context=self.context,
                manifest_path=path,
                trusted_root=self.storage_root.value,
                allow_external=bool(
                    self.trust_external_files.value
                ),
            )
        except Exception as exc:
            self.status.alert_type = "danger"
            self.status.object = f"Model load failed: {exc}"
            return

        self.trust_external_files.value = False
        self.status.alert_type = "success"
        self.status.object = (
            f"Loaded model artifact `{result['model_artifact_id']}` from `{result['manifest_path']}`. "
            f"Model data: `{result['model_path']}`"
        )
        self.refresh()
        if result.get("model_artifact_id") in self._option_values(self.model.options):
            self.model.value = result["model_artifact_id"]

    def select_for_prediction(self) -> None:
        artifact_id = self.model.value
        if not artifact_id:
            return
        publish(
            self.context,
            "ml.model.selected",
            {"artifact_id": artifact_id, "model_artifact_id": artifact_id, "origin": "core.ml.model_manager"},
        )
        self.status.alert_type = "success"
        self.status.object = f"Selected model `{artifact_id}` for the Predictor."

    def select_for_resume(self) -> None:
        value = str(self.resume_checkpoint.value or "")
        if not value:
            return
        payload: Dict[str, Any] = {"origin": "core.ml.model_manager"}
        if value.startswith("artifact:"):
            payload["resume_checkpoint_artifact_id"] = value.split(":", 1)[1]
        elif value.startswith("manifest:"):
            payload["resume_manifest_path"] = value.split(":", 1)[1]
        publish(self.context, "ml.resume_checkpoint.selected", payload)
        self.status.alert_type = "success"
        self.status.object = "Sent the selected paused checkpoint to the Recipe Launcher."

    def _refresh_resume_artifacts(self) -> None:
        artifacts = getattr(self.context, "artifacts", None)
        find = getattr(artifacts, "find", None)
        get = getattr(artifacts, "get", None)
        options: Dict[str, str] = {}
        if callable(find) and callable(get):
            try:
                refs = find(type="ml.resume_checkpoint")
            except Exception:
                refs = []
            for ref in refs:
                artifact_id = getattr(ref, "artifact_id", None)
                if not artifact_id:
                    continue
                try:
                    payload = get(artifact_id)
                except Exception:
                    payload = {}
                epoch = payload.get("completed_epoch") if isinstance(payload, Mapping) else None
                recipe = payload.get("recipe_id") if isinstance(payload, Mapping) else None
                run_id = payload.get("run_id") if isinstance(payload, Mapping) else None
                label = f"{recipe or 'recipe'} — epoch {epoch if epoch is not None else '?'} — run {str(run_id or '')[:8]}"
                options[label] = f"artifact:{artifact_id}"
        disk = {
            key: value
            for key, value in (dict(self.resume_checkpoint.options).items() if isinstance(self.resume_checkpoint.options, dict) else [])
            if str(value).startswith("manifest:")
        }
        options.update(disk)
        self.resume_checkpoint.options = options
        self.select_for_resume_button.disabled = not bool(self.resume_checkpoint.value)

    def _on_model_change(self) -> None:
        artifact_id = self.model.value
        self.select_for_prediction_button.disabled = not bool(artifact_id)
        if not artifact_id:
            self.model_details.object = {}
            return
        try:
            payload = self.context.artifacts.get(str(artifact_id))
        except Exception as exc:
            self.model_details.object = {"error": str(exc)}
            return
        self.model_details.object = dict(payload) if isinstance(payload, Mapping) else {"value": str(payload)}

    def _on_manifest_change(self) -> None:
        if self.manifest.value:
            self.manifest_path.value = str(self.manifest.value)

    def _on_resume_change(self) -> None:
        self.select_for_resume_button.disabled = not bool(self.resume_checkpoint.value)

    def _update_storage_summary(self) -> None:
        locations = storage_locations(self.context, {"ml_artifact_dir": self.storage_root.value})
        self.storage_summary.object = "\n".join(
            [
                "**Current durable storage**",
                f"- Models and checkpoints: `{locations['artifact_root']}`",
                f"- Predictions: `{locations['predictions']}`",
                f"- Training logs: `{locations['training_logs']}`",
            ]
        )

    def _catalog(self) -> Any:
        get = getattr(getattr(self.context, "services", None), "get", None)
        if not callable(get):
            return None
        for key in ("core.ml.trained_model_catalog", "trained_model_catalog"):
            try:
                return get(key)
            except Exception:
                continue
        return None

    def _subscribe(self) -> None:
        subscribe = getattr(getattr(self.context, "events", None), "subscribe", None)
        if not callable(subscribe):
            return
        for topic in ("ml.model.saved", "ml.recipe_run.finished", "artifact.created", "workspace.restored"):
            try:
                self._subscriptions.append(subscribe(topic, lambda *_args, **_kwargs: self._schedule_refresh()))
            except Exception:
                pass

    def _schedule_refresh(self) -> None:
        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(self.refresh)
                return
        except Exception:
            pass
        self.refresh()

    @staticmethod
    def _path_label(path: str) -> str:
        p = Path(path)
        return f"{p.parent.parent.name}/{p.parent.name}/{p.name}"

    @staticmethod
    def _option_values(options: Any) -> set[Any]:
        if isinstance(options, Mapping):
            return set(options.values())
        return set(options or [])


def create_model_manager_panel(context: Any, **kwargs: Any):
    controller = MLModelManagerPanel(context=context, restore_state=kwargs.get("restore_state"))
    return controller.panel(), controller
