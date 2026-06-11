from __future__ import annotations

import importlib.util
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import panel as pn

from astronomicAL.platform.plugins.specs import ActionRequest

def _load_sibling(stem: str):
    module_name = f"{__name__}.{stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = Path(__file__).with_name(f"{stem}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {stem!r} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ml = _load_sibling("ml")
_image_ml = _load_sibling("image_ml")
_actions = _load_sibling("actions")

TrainConfig = _ml.TrainConfig
train_model = _ml.train_model

ImageTrainConfig = _image_ml.ImageTrainConfig
train_image_model = _image_ml.train_image_model


class MLWorkbenchPanel:
    """Unified ML workbench.

    Model Builder owns architecture + hyperparameters.
    Workbench owns dataset + target + modality-specific input binding + split + training.
    """

    def __init__(
        self,
        *,
        context: Any,
        registry: Any,
        restore_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.context = context
        self.registry = registry
        self.root_tabs = None
        self.input_tabs = None
        self._visible_feature_columns: List[str] = []
        self._model_definition_subscription = None

        self._disposed = False
        self._active_job_handle: Optional[Any] = None
        self._active_job_key: Optional[str] = None
        self._active_job_id: Optional[str] = None
        self._active_training_log_artifact_id: Optional[str] = None

        self.dataset = pn.widgets.Select(name="", options=self._dataset_options())
        self.refresh = pn.widgets.Button(name="Refresh datasets", button_type="light")

        self.task = pn.widgets.Select(
            name="",
            options=["classification", "regression", "segmentation"],
            value="classification",
        )
        self.target = pn.widgets.Select(name="", options=[])
        self.model = pn.widgets.Select(name="", options={})
        self.refresh_models = pn.widgets.Button(
            name="Refresh model definitions",
            button_type="light",
        )
        self.model_note = pn.pane.HTML("")

        # Tabular input binding.
        self.search = pn.widgets.TextInput(
            name="",
            placeholder="Filter columns, e.g. flux, mag, error...",
        )
        self.features = pn.widgets.MultiSelect(
            name="",
            options=[],
            value=[],
            size=10,
        )
        self.use_numeric = pn.widgets.Button(name="Use numeric", button_type="primary")
        self.add_filtered = pn.widgets.Button(name="Add filtered", button_type="light")
        self.use_all = pn.widgets.Button(name="Use all usable", button_type="light")
        self.clear = pn.widgets.Button(name="Clear", button_type="light")
        self.feature_count = pn.pane.HTML("0 feature columns selected.")

        # Image input binding.
        self.image_column = pn.widgets.Select(name="", options=[])
        self.image_note = pn.pane.HTML("")

        # Future segmentation input binding.
        self.mask_column = pn.widgets.Select(name="", options=[])
        self.segmentation_note = pn.pane.HTML("")

        # Split/training.
        self.validation_size = pn.widgets.FloatSlider(
            name="",
            start=0.05,
            end=0.4,
            step=0.05,
            value=0.1,
        )
        self.test_size = pn.widgets.FloatSlider(
            name="",
            start=0.05,
            end=0.5,
            step=0.05,
            value=0.1,
        )
        self.random_state = pn.widgets.IntInput(name="", value=42, start=0)
        self.stratify = pn.widgets.Checkbox(
            name="Stratify classification split where possible",
            value=True,
        )

        self.train_button = pn.widgets.Button(name="Train model", button_type="success")
        self.cancel_button = pn.widgets.Button(
            name="Cancel training",
            button_type="danger",
            disabled=True,
        )
        self.status = pn.pane.Alert("Choose a model definition and inputs, then train.", alert_type="info")
        self.metrics = pn.pane.JSON({}, depth=3, sizing_mode="stretch_width")
        self.predictions = pn.pane.DataFrame(
            pd.DataFrame(),
            height=260,
            sizing_mode="stretch_width",
        )

        self.tuning_summary = pn.pane.Markdown(
            "No tuning information yet.",
            sizing_mode="stretch_width",
        )
        self.tuning_trials = pn.pane.DataFrame(
            pd.DataFrame(),
            height=260,
            sizing_mode="stretch_width",
        )

        self.artifacts = pn.pane.Markdown("")

        self.dataset.param.watch(lambda *_: self._load_columns(), "value")
        self.task.param.watch(lambda *_: self._on_task_change(), "value")
        self.target.param.watch(lambda *_: self._on_target_change(), "value")
        self.model.param.watch(lambda *_: self._on_model_change(), "value")
        self.search.param.watch(lambda *_: self._update_feature_options(), "value")
        self.features.param.watch(lambda *_: self._update_feature_count(), "value")

        self.refresh.on_click(lambda *_: self._load_columns(prefer_active=True))
        self.refresh_models.on_click(lambda *_: None if self._disposed else self._load_models())
        self.use_numeric.on_click(lambda *_: self._select_numeric())
        self.add_filtered.on_click(lambda *_: self._add_filtered())
        self.use_all.on_click(lambda *_: self._select_all())
        self.clear.on_click(lambda *_: setattr(self.features, "value", []))
        self.train_button.on_click(self._train)
        self.cancel_button.on_click(self._cancel_training)

        self._apply_widget_sizing()

        try:
            _ml.sync_model_definitions_from_artifacts(self.context, self.registry)
        except Exception:
            pass

        self._load_models()
        self._load_columns(prefer_active=True)
        self._subscribe_to_model_definition_events()

        if restore_state:
            self.restore_state(restore_state)

        self._on_model_change()

    def _set_training_running(self, running: bool) -> None:
        self.train_button.disabled = running
        self.cancel_button.disabled = not running


    def _cancel_training(self, *_: Any) -> None:
        """Request cooperative cancellation of the active training job."""

        if self._disposed:
            return

        handle = self._active_job_handle

        if handle is None:
            self.status.alert_type = "warning"
            self.status.object = "No active training job to cancel."
            self._set_training_running(False)
            return

        self.cancel_button.disabled = True
        self.status.alert_type = "warning"
        self.status.object = "Cancellation requested. Waiting for the training loop to stop..."

        self._update_active_training_log(
            status="cancelling",
            message="Cancellation requested by user.",
        )

        try:
            cancel = getattr(handle, "cancel", None)
            if callable(cancel):
                cancel()
        except Exception as exc:
            self.status.alert_type = "danger"
            self.status.object = f"Could not request cancellation: {exc}"


    def _clear_active_training_state(self) -> None:
        self._active_job_handle = None
        self._active_job_key = None
        self._active_job_id = None
        self._active_training_log_artifact_id = None
        self._set_training_running(False)


    def _is_cancel_error(self, error: BaseException) -> bool:
        text = f"{type(error).__name__}: {error}".lower()
        return "cancel" in text or "cancelled" in text or "canceled" in text


    def _update_active_training_log(self, *, status: str, message: str) -> None:
        artifact_id = self._active_training_log_artifact_id

        if not artifact_id:
            return

        try:
            payload = self.context.artifacts.get(artifact_id)
        except Exception:
            payload = None

        if isinstance(payload, dict):
            payload.update(
                {
                    "status": status,
                    "message": message,
                    "updated_at": time.time(),
                }
            )

        self._publish(
            "ml.training_log.updated",
            {
                "artifact_id": artifact_id,
                "run_id": payload.get("run_id") if isinstance(payload, dict) else None,
                "status": status,
                "message": message,
            },
        )

    def dispose(self) -> None:
        """Called by WorkspaceManager when this panel/controller is closed.

        Keep context/registry references intact. Panel/Bokeh can still deliver
        late widget events after disposal or during layout restore, so callbacks
        must be able to fail safely instead of hitting None.list_models.
        """

        self._disposed = True

        try:
            handle = self._active_job_handle
            cancel = getattr(handle, "cancel", None)
            if callable(cancel):
                cancel()
            else:
                self._cancel_active_job()
        except Exception:
            pass

        try:
            self._update_active_training_log(
                status="cancelled",
                message="Training cancelled because the ML Workbench was closed.",
            )
        except Exception:
            pass

        try:
            events = getattr(self.context, "events", None)
            unsubscribe = getattr(events, "unsubscribe", None)
            if callable(unsubscribe) and self._model_definition_subscription is not None:
                subs = self._model_definition_subscription
                if not isinstance(subs, list):
                    subs = [subs]
                for sub in subs:
                    try:
                        unsubscribe(sub)
                    except Exception:
                        pass
        except Exception:
            pass

        self._model_definition_subscription = None
        self._active_job_handle = None
        self._active_job_key = None
        self._active_job_id = None
        self._active_training_log_artifact_id = None

        try:
            self.metrics.object = {}
            self.predictions.object = pd.DataFrame()
            self.artifacts.object = ""
        except Exception:
            pass

    def _cancel_active_job(self) -> None:
        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            return

        for method_name, value in [
            ("cancel", self._active_job_id),
            ("cancel", self._active_job_key),
            ("cancel_job", self._active_job_id),
            ("cancel_job", self._active_job_key),
            ("cancel_by_key", self._active_job_key),
        ]:
            if not value:
                continue

            method = getattr(jobs, method_name, None)
            if callable(method):
                try:
                    method(value)
                    return
                except TypeError:
                    continue

    def panel(self):
        data_tab = pn.Column(
            self._field("Dataset", self.dataset),
            self.refresh,
            pn.Spacer(height=6),
            self._field("Task", self.task),
            self._field("Target column", self.target),
            self._field("Model definition", self.model),
            self.refresh_models,
            self.model_note,
            sizing_mode="stretch_width",
            styles=self._tab_styles(),
        )

        self.input_tabs = pn.Tabs(
            ("Inputs", self._inputs_panel()),
            dynamic=True,
            sizing_mode="stretch_both",
            styles={
                "box-sizing": "border-box",
                "overflow": "hidden",
                "padding": "0",
                "margin": "0",
            },
        )

        training_tab = pn.Column(
            self._field("Validation size", self.validation_size),
            self._field("Test size", self.test_size),
            self._field("Random state", self.random_state),
            self.stratify,
            pn.Spacer(height=8),
            self.train_button,
            self.cancel_button,
            sizing_mode="stretch_width",
            styles=self._tab_styles(),
        )

        setup_tab = pn.Tabs(
            ("Data", data_tab),
            ("Inputs", self.input_tabs),
            ("Training", training_tab),
            dynamic=True,
            sizing_mode="stretch_both",
            styles={
                "box-sizing": "border-box",
                "overflow": "hidden",
                "padding": "0",
                "margin": "0",
            },
        )

        results_tab = pn.Column(
            self.status,
            pn.Tabs(
                ("Metrics", pn.Column(self.metrics, sizing_mode="stretch_width")),
                ("Prediction preview", pn.Column(self.predictions, sizing_mode="stretch_width")),
                (
                    "Optuna tuning",
                    pn.Column(
                        self.tuning_summary,
                        self.tuning_trials,
                        sizing_mode="stretch_width",
                    ),
                ),
                ("Artifacts", pn.Column(self.artifacts, sizing_mode="stretch_width")),
                dynamic=True,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_both",
            styles=self._tab_styles(),
        )

        self.root_tabs = pn.Tabs(
            ("Setup", setup_tab),
            ("Results", results_tab),
            dynamic=True,
            sizing_mode="stretch_both",
            styles={
                "box-sizing": "border-box",
                "overflow": "hidden",
            },
        )

        self._refresh_inputs_panel()
        return self.root_tabs

    def get_state(self) -> Dict[str, Any]:
        return {
            "task": self.task.value,
            "target": self.target.value,
            "model": self.model.value,
            "features": list(self.features.value or []),
            "image_column": self.image_column.value,
            "mask_column": self.mask_column.value,
            "validation_size": self.validation_size.value,
            "test_size": self.test_size.value,
            "random_state": self.random_state.value,
            "stratify": self.stratify.value,
            "follow_active_dataset": True,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return


        if state.get("task") in self.task.options:
            self.task.value = state["task"]

        self._load_columns(prefer_active=True)

        if state.get("target") in self.target.options:
            self.target.value = state["target"]

        self._load_models()

        model_values = set(self.model.options.values())
        if state.get("model") in model_values:
            self.model.value = state["model"]

        self._update_feature_options()

        restored_features = [
            c for c in state.get("features", [])
            if c in self._all_feature_columns()
        ]
        self.features.options = list(dict.fromkeys([*restored_features, *self.features.options]))
        self.features.value = restored_features

        if state.get("image_column") in self.image_column.options:
            self.image_column.value = state["image_column"]

        if state.get("mask_column") in self.mask_column.options:
            self.mask_column.value = state["mask_column"]

        self.validation_size.value = float(state.get("validation_size", self.validation_size.value))
        self.test_size.value = float(state.get("test_size", self.test_size.value))
        self.random_state.value = int(state.get("random_state", self.random_state.value))
        self.stratify.value = bool(state.get("stratify", self.stratify.value))

        self._on_model_change()

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

    def _tab_styles(self) -> Dict[str, str]:
        return {
            "box-sizing": "border-box",
            "padding": "10px 14px 14px 14px",
            "overflow-y": "auto",
            "overflow-x": "hidden",
        }

    def _apply_widget_sizing(self) -> None:
        for widget in [
            self.dataset,
            self.task,
            self.target,
            self.model,
            self.search,
            self.image_column,
            self.mask_column,
            self.random_state,
        ]:
            widget.sizing_mode = "stretch_width"
            widget.height = 38
            widget.margin = (0, 0, 0, 0)

        self.features.sizing_mode = "stretch_width"
        self.features.height = 240
        self.features.margin = (0, 0, 8, 0)

        for slider in [self.validation_size, self.test_size]:
            slider.sizing_mode = "stretch_width"
            slider.height = 42
            slider.margin = (0, 0, 0, 0)

        self.stratify.sizing_mode = "stretch_width"
        self.stratify.height = 30
        self.stratify.margin = (0, 0, 8, 0)

        for button in [
            self.refresh,
            self.refresh_models,
            self.use_numeric,
            self.add_filtered,
            self.use_all,
            self.clear,
            self.train_button,
            self.cancel_button,
        ]:
            button.sizing_mode = "stretch_width"
            button.height = 34
            button.margin = (0, 0, 8, 0)

        self.status.sizing_mode = "stretch_width"
        self.status.margin = (0, 0, 10, 0)

        self.metrics.sizing_mode = "stretch_width"
        self.predictions.sizing_mode = "stretch_width"
        self.predictions.height = 280
        self.artifacts.sizing_mode = "stretch_width"

        self.tuning_summary.sizing_mode = "stretch_width"
        self.tuning_trials.sizing_mode = "stretch_width"
        self.tuning_trials.height = 280

    def _active_dataset_id(self) -> Optional[str]:
        try:
            active_id = self.context.datasets.active_id()
            return str(active_id) if active_id is not None else None
        except Exception:
            return None

    def _choose_dataset_value(
        self,
        *,
        options: List[str],
        prefer_active: bool = False,
        fallback: Optional[str] = None,
    ) -> Optional[str]:
        if not options:
            return None

        active_id = self._active_dataset_id()
        current = self.dataset.value

        if prefer_active and active_id in options:
            return active_id

        if current in options:
            return current

        if fallback in options:
            return fallback

        if active_id in options:
            return active_id

        return options[0]

    def _dataset_options(self) -> List[str]:
        try:
            return list(self.context.datasets.list_ids())
        except Exception:
            return []

    def _dataset_id(self) -> str:
        if self.dataset.value:
            return str(self.dataset.value)
        return str(self.context.datasets.active_id())

    def _load_columns(self, *, prefer_active: bool = False) -> None:
        options = self._dataset_options()
        self.dataset.options = options

        chosen = self._choose_dataset_value(
            options=options,
            prefer_active=prefer_active,
        )

        if chosen != self.dataset.value:
            self.dataset.value = chosen

        columns = self._columns()
        self.target.options = columns
        self.image_column.options = columns
        self.mask_column.options = columns

        if self.target.value not in columns:
            self.target.value = self._guess_target(columns)

        if self.image_column.value not in columns:
            self.image_column.value = self._guess_image_column(columns)

        if self.mask_column.value not in columns:
            self.mask_column.value = self._guess_mask_column(columns)

        self._update_feature_options()

    def _columns(self) -> List[str]:
        try:
            source = self.context.datasets.get_source(self._dataset_id())
            return [str(c) for c in source.columns()]
        except Exception:
            return []

    def _guess_target(self, columns: List[str]) -> Optional[str]:
        try:
            mapped = self.context.datasets.get_mapping(self._dataset_id(), "target_label")
            if mapped in columns:
                return mapped
        except Exception:
            pass

        lowered = {c.lower(): c for c in columns}
        for candidate in ("target_label", "target", "label", "class", "classification", "y"):
            if candidate in lowered:
                return lowered[candidate]

        return columns[0] if columns else None

    def _guess_image_column(self, columns: List[str]) -> Optional[str]:
        lowered = {c.lower(): c for c in columns}

        for candidate in (
            "image",
            "img",
            "image_path",
            "image_uri",
            "image_url",
            "path",
            "file",
            "filename",
            "jpg",
            "png",
            "cutout",
            "thumbnail",
        ):
            if candidate in lowered:
                return lowered[candidate]

        for column in columns:
            low = column.lower()
            if any(token in low for token in ("image", "img", "path", "uri", "url", "cutout", "jpg", "png")):
                return column

        return columns[0] if columns else None

    def _guess_mask_column(self, columns: List[str]) -> Optional[str]:
        lowered = {c.lower(): c for c in columns}

        for candidate in ("mask", "mask_path", "mask_uri", "segmentation", "label_mask"):
            if candidate in lowered:
                return lowered[candidate]

        for column in columns:
            low = column.lower()
            if any(token in low for token in ("mask", "segmentation")):
                return column

        return columns[0] if columns else None

    def _on_task_change(self) -> None:
        self._load_models()
        self._refresh_inputs_panel()

    def _on_target_change(self) -> None:
        self._update_feature_options()

    def _on_model_change(self) -> None:
        self._update_model_note()
        self._refresh_inputs_panel()

    def _get_registry(self, *, force_reacquire: bool = False):
        """Return the canonical ML registry service.

        The Workbench should not permanently trust an old registry reference.
        Plugin reloads, workspace restore, or panel construction order can leave
        the builder and workbench with different handles. Reacquire from
        core.ml.registry whenever a refresh/save event happens.
        """

        if not force_reacquire and getattr(self, "registry", None) is not None:
            return self.registry

        context = getattr(self, "context", None)
        services = getattr(context, "services", None)
        get = getattr(services, "get", None)

        if callable(get):
            try:
                registry = get("core.ml.registry")
                if registry is not None:
                    self.registry = registry
                    return registry
            except Exception:
                pass

        return getattr(self, "registry", None)

    def _load_models(self, *, force_reacquire_registry: bool = False) -> None:
        if getattr(self, "_disposed", False):
            return

        registry = self._get_registry(force_reacquire=force_reacquire_registry)
        if registry is None:
            self.model.options = {}
            self.model.value = None
            self.model_note.object = (
                "<p>"
                "ML registry service is not available. Try closing and reopening "
                "the ML Workbench, or check that the core.ml plugin registered "
                "the core.ml.registry service."
                "</p>"
            )
            return

        sync_error = None
        try:
            _ml.sync_model_definitions_from_artifacts(self.context, registry)
        except Exception as exc:
            sync_error = exc

        previous = self.model.value

        try:
            models = registry.list_models(task=self.task.value)
        except Exception as exc:
            self.model.options = {}
            self.model.value = None
            self.model_note.object = (
                "<p>"
                f"Could not load model definitions: {exc}"
                "</p>"
            )
            return

        supported = []
        for model in models:
            modality = getattr(model, "modality", "tabular")
            task = getattr(model, "task", None)

            if modality == "tabular":
                supported.append(model)
            elif modality == "image" and task == "classification":
                supported.append(model)
            elif modality == "image" and task == "segmentation":
                # Keep segmentation definitions visible, but _train still blocks
                # with the existing "not implemented" message.
                supported.append(model)

        options = {
            f"{model.title} [{model.framework}/{getattr(model, 'modality', 'tabular')}]: {model.id}": model.id
            for model in supported
        }

        self.model.options = options
        model_ids = list(options.values())

        if previous in model_ids:
            self.model.value = previous
        elif model_ids:
            self.model.value = model_ids[0]
        else:
            self.model.value = None

        self._update_model_note()
        self._refresh_inputs_panel()

        if sync_error is not None and not supported:
            self.model_note.object = (
                "<p>"
                f"Could not sync saved model definitions from artifacts: {sync_error}"
                "</p>"
            )

    def _selected_model(self):
        if getattr(self, "_disposed", False):
            return None

        registry = self._get_registry()
        if registry is None:
            return None

        try:
            if not self.model.value:
                return None
            return registry.get_model(str(self.model.value))
        except Exception:
            return None

    def _selected_modality(self) -> str:
        model = self._selected_model()
        return str(getattr(model, "modality", "tabular")) if model is not None else "tabular"

    def _update_model_note(self) -> None:
        model = self._selected_model()
        if model is None:
            self.model_note.object = (
                "<div style='font-size:12px;color:#777'>"
                "No compatible model definitions available. Create one in "
                "ML Model Builder, then click refresh."
                "</div>"
            )
            return

        params = getattr(model, "default_params", {}) or {}
        params_preview = ", ".join(f"{k}={v}" for k, v in list(params.items())[:8])
        if len(params) > 8:
            params_preview += ", ..."

        tuning = self._selected_tuning(model)
        if tuning:
            search_space = tuning.get("search_space") or {}
            tuned_params = ", ".join(search_space.keys()) or "none"
            tuning_line = (
                f"<br><b>Optuna:</b> enabled &nbsp; "
                f"metric={tuning.get('metric', 'unknown')} &nbsp; "
                f"trials={tuning.get('n_trials', 'unknown')} &nbsp; "
                f"params={tuned_params}"
            )
        else:
            tuning_line = "<br><b>Optuna:</b> disabled"

        self.model_note.object = (
            "<div style='font-size:12px;color:#555'>"
            f"Framework: {model.framework} &nbsp; "
            f"Task: {model.task} &nbsp; "
            f"Modality: {getattr(model, 'modality', 'tabular')}<br>"
            f"Hyperparameters: {params_preview or 'defaults'}"
            f"{tuning_line}"
            "</div>"
        )

    def _selected_tuning(self, model: Any = None) -> Dict[str, Any]:
        model = model or self._selected_model()
        if model is None:
            return {}

        metadata = getattr(model, "metadata", {}) or {}
        tuning = metadata.get("tuning") or {}

        if not isinstance(tuning, dict):
            return {}
        if not tuning.get("enabled"):
            return {}
        if not tuning.get("search_space"):
            return {}

        return dict(tuning)

    def _inputs_panel(self):
        modality = self._selected_modality()
        model = self._selected_model()

        if modality == "image" and model is not None and model.task == "classification":
            return pn.Column(
                self._field("Image column", self.image_column),
                self.image_note,
                pn.pane.HTML(
                    "<div style='font-size:12px;opacity:0.75;margin-top:6px;'>"
                    "The selected model definition is image-based, so the workbench "
                    "uses an image column instead of tabular feature columns."
                    "</div>"
                ),
                sizing_mode="stretch_width",
                styles=self._tab_styles(),
            )

        if modality == "image" and model is not None and model.task == "segmentation":
            return pn.Column(
                self._field("Image column", self.image_column),
                self._field("Mask column", self.mask_column),
                pn.pane.Alert(
                    "Segmentation input binding is shown, but segmentation training is not implemented yet.",
                    alert_type="warning",
                ),
                sizing_mode="stretch_width",
                styles=self._tab_styles(),
            )

        return pn.Column(
            self._field("Filter feature columns", self.search),
            self.features,
            pn.Row(
                self.use_numeric,
                self.add_filtered,
                self.use_all,
                self.clear,
                sizing_mode="stretch_width",
            ),
            self.feature_count,
            sizing_mode="stretch_width",
            styles=self._tab_styles(),
        )

    def _refresh_inputs_panel(self) -> None:
        if self.input_tabs is None:
            return

        self.input_tabs.objects = [("Inputs", self._inputs_panel())]

    def _all_feature_columns(self) -> List[str]:
        return [c for c in self._columns() if c != self.target.value]

    def _update_feature_options(self) -> None:
        all_features = self._all_feature_columns()
        selected = [c for c in self.features.value or [] if c in all_features]

        query = (self.search.value or "").strip().lower()
        visible = [c for c in all_features if not query or query in c.lower()]
        self._visible_feature_columns = visible

        self.features.options = list(dict.fromkeys([*selected, *visible]))
        self.features.value = selected
        self._update_feature_count()

    def _select_numeric(self) -> None:
        columns = self._all_feature_columns()

        if not columns:
            self.features.value = []
            return

        df = self.context.datasets.get_df(
            self._dataset_id(),
            columns=columns,
            limit=1000,
        )

        numeric = [
            c for c in columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        self.features.options = columns
        self.features.value = numeric
        self._update_feature_count()

    def _add_filtered(self) -> None:
        current = list(self.features.value or [])
        added = list(self._visible_feature_columns or [])
        combined = list(dict.fromkeys([*current, *added]))

        self.features.options = list(dict.fromkeys([*combined, *self.features.options]))
        self.features.value = combined
        self._update_feature_count()

    def _select_all(self) -> None:
        columns = self._all_feature_columns()
        self.features.options = columns
        self.features.value = columns
        self._update_feature_count()

    def _update_feature_count(self) -> None:
        count = len(self.features.value or [])
        visible = len(self._visible_feature_columns or [])
        self.feature_count.object = (
            "<div style='font-size:12px;margin:0 0 8px 0;'>"
            f"<b>{count}</b> selected. "
            f"<span style='opacity:0.75'>{visible} currently visible from filter.</span>"
            "</div>"
        )

    def _submit_training_action(
        self,
        *,
        action_fn: Any,
        request: ActionRequest,
        title: str,
        key: str,
    ) -> None:
        jobs = getattr(self.context, "jobs", None)
        submit = getattr(jobs, "submit", None)

        if callable(submit):
            self._active_job_key = key
            submitted = submit(
                action_fn,
                title=title,
                key=key,
                on_done=self._on_train_done,
                on_error=self._on_train_error,
                context=self.context,
                request=request,
            )
            self._active_job_handle = submitted
            self._active_job_id = (
                getattr(submitted, "job_id", None)
                or getattr(submitted, "id", None)
            )
            return

        result = action_fn(context=self.context, request=request)
        self._on_train_done(result)

    def _tabular_action_job_key(self, request: ActionRequest) -> str:
        params = request.params or {}
        features_key = ",".join(params.get("feature_columns") or request.columns or [])
        return (
            f"core.ml.action.train_tabular:{params.get('dataset_id')}:{params.get('task')}:"
            f"{params.get('target_column')}:{features_key}:{params.get('model_id')}:"
            f"{params.get('validation_size')}:{params.get('test_size')}:{params.get('random_state')}"
        )

    def _image_action_job_key(self, request: ActionRequest) -> str:
        params = request.params or {}
        return (
            f"core.ml.action.train_image:{params.get('dataset_id')}:classification:"
            f"{params.get('image_column')}:{params.get('target_column')}:{params.get('model_id')}:"
            f"{params.get('validation_size')}:{params.get('test_size')}:{params.get('random_state')}"
        )

    def _format_tuning_summary(self, tuning: Dict[str, Any]) -> str:
        if not tuning or not tuning.get("enabled"):
            return "Optuna tuning was not enabled for this run."

        trials = list(tuning.get("trials") or [])
        completed = [
            trial for trial in trials
            if trial.get("value") is not None
            and str(trial.get("state", "complete")).lower() == "complete"
        ]

        best_params = tuning.get("best_params") or {}
        best_value = tuning.get("best_value")

        lines = [
            "### Optuna tuning",
            f"- **Backend:** `{tuning.get('backend', 'optuna')}`",
            f"- **Metric:** `{tuning.get('metric', '')}`",
            f"- **Trials completed:** `{len(completed)}` / `{tuning.get('n_trials') or len(trials)}`",
        ]

        if best_value is not None:
            lines.append(f"- **Best value:** `{best_value}`")

        if best_params:
            lines.append("- **Best parameters:**")
            for key, value in best_params.items():
                lines.append(f"  - `{key}` = `{value}`")

        message = tuning.get("message")
        if message:
            lines.append(f"- **Latest message:** {message}")

        return "\n".join(lines)

    def _train(self, *_: Any) -> None:
        try:
            model = self._selected_model()
            if model is None:
                raise ValueError("Choose a model definition.")

            modality = str(getattr(model, "modality", "tabular"))
            run_id = uuid.uuid4().hex
            training_log_artifact_id = self._create_training_log_stub(run_id, model)

            self._active_training_log_artifact_id = training_log_artifact_id
            self._set_training_running(True)
            self.status.alert_type = "info"

            tuning = self._selected_tuning(model)
            if tuning:
                self.status.object = (
                    "Training started. Optuna tuning is enabled; the Training Curves "
                    "panel will update as trials finish."
                )
            else:
                self.status.object = "Training started..."

            if modality == "tabular":
                feature_columns = list(self.features.value or [])
                if not feature_columns:
                    raise ValueError("Choose at least one feature column.")

                request = ActionRequest(
                    dataset_id=self._dataset_id(),
                    columns=feature_columns,
                    params={
                        "dataset_id": self._dataset_id(),
                        "target_column": str(self.target.value),
                        "feature_columns": feature_columns,
                        "model_id": str(self.model.value),
                        "task": str(self.task.value),
                        "test_size": float(self.test_size.value),
                        "validation_size": float(self.validation_size.value),
                        "random_state": int(self.random_state.value),
                        "stratify": bool(self.stratify.value),
                        "run_id": run_id,
                        "training_log_artifact_id": training_log_artifact_id,
                    },
                    origin="core.ml.workbench",
                )

                self._submit_training_action(
                    action_fn=_actions.train_tabular_action,
                    request=request,
                    title="Train tabular ML model",
                    key=self._tabular_action_job_key(request),
                )
                return

            if modality == "image" and model.task == "classification":
                if not self.image_column.value:
                    raise ValueError("Choose an image column.")

                request = ActionRequest(
                    dataset_id=self._dataset_id(),
                    columns=[str(self.image_column.value), str(self.target.value)],
                    params={
                        "dataset_id": self._dataset_id(),
                        "image_column": str(self.image_column.value),
                        "target_column": str(self.target.value),
                        "model_id": str(self.model.value),
                        "test_size": float(self.test_size.value),
                        "validation_size": float(self.validation_size.value),
                        "random_state": int(self.random_state.value),
                        "stratify": bool(self.stratify.value),
                        "run_id": run_id,
                        "training_log_artifact_id": training_log_artifact_id,
                    },
                    origin="core.ml.workbench",
                )

                self._submit_training_action(
                    action_fn=_actions.train_image_classifier_action,
                    request=request,
                    title="Train image ML model",
                    key=self._image_action_job_key(request),
                )
                return

            raise ValueError(
                f"Training is not implemented yet for modality `{modality}` "
                f"and task `{model.task}`."
            )

        except Exception as exc:
            self._on_train_error(exc)

    def _tabular_job_key(self, config: Any) -> str:
        features_key = ",".join(config.feature_columns)
        return (
            f"core.ml.train:{config.dataset_id}:{config.task}:"
            f"{config.target_column}:{features_key}:{config.model_id}:"
            f"{config.validation_size}:{config.test_size}:{config.random_state}"
        )

    def _image_job_key(self, config: Any) -> str:
        return (
            f"core.ml.image.train:{config.dataset_id}:classification:"
            f"{config.image_column}:{config.target_column}:{config.model_id}:"
            f"{config.validation_size}:{config.test_size}:{config.random_state}"
        )

    def _create_training_log_stub(self, run_id: str, model: Any) -> Optional[str]:
        params = getattr(model, "default_params", {}) or {}
        modality = str(getattr(model, "modality", "tabular"))
        tuning = self._selected_tuning(model)

        if tuning:
            message = (
                "Training queued. Optuna tuning will run before the final model "
                "is trained with the best parameters."
            )
            status = "queued"
        else:
            message = "Training queued. Waiting for worker..."
            status = "queued"

        payload = {
            "run_id": run_id,
            "status": status,
            "message": message,
            "task": str(model.task),
            "modality": modality,
            "model_id": str(self.model.value),
            "model_title": str(model.title),
            "framework": str(model.framework),
            "dataset_id": self._dataset_id(),
            "target_column": str(self.target.value),
            "feature_columns": list(self.features.value or []) if modality == "tabular" else [],
            "image_column": str(self.image_column.value) if modality == "image" else None,
            "params": params,
            "tuning": tuning,
            "tuning_trials": [],
            "tuning_best_params": {},
            "tuning_best_value": None,
            "optimize_metric": tuning.get("metric") or params.get("optimize_metric", "val_loss"),
            "best_epoch": None,
            "last_epoch": None,
            "epochs": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        artifact_id = None
        try:
            artifact_id = self.context.artifacts.put(
                "ml.training_log",
                payload,
                dataset_id=self._dataset_id(),
                params={"run_id": run_id},
            )
        except Exception:
            artifact_id = None

        if artifact_id:
            self._publish(
                "ml.training_log.created",
                {
                    "artifact_id": artifact_id,
                    "run_id": run_id,
                    "status": status,
                    "message": message,
                    "tuning_enabled": bool(tuning),
                },
            )

        return artifact_id

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        events = getattr(self.context, "events", None)
        publish = getattr(events, "publish", None)

        if callable(publish):
            publish(topic, payload)

    def _schedule_ui_update(self, callback) -> None:
        """Run a UI update safely on the next Bokeh/Panel tick when available."""

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(callback)
                return
        except Exception:
            pass

        callback()

    def _subscribe_to_model_definition_events(self) -> None:
        events = getattr(self.context, "events", None)
        subscribe = getattr(events, "subscribe", None)
        if not callable(subscribe):
            return

        self._model_definition_subscription = []

        for topic in [
            "ml.model_definition.created",
            "ml.model_definition.saved",
            "ml.model_definition.updated",
            "ml.registry.changed",
            "artifact.created",
            "workspace.restored",
        ]:
            try:
                sub = subscribe(
                    topic,
                    self._on_model_definition_event,
                    owner_label="ML Workbench",
                    owner_kind="panel",
                )
                self._model_definition_subscription.append(sub)
            except Exception:
                pass

    def _on_model_definition_event(self, topic: str, payload: Any) -> None:
        if getattr(self, "_disposed", False):
            return

        payload = payload if isinstance(payload, dict) else {}

        # Ignore unrelated artifact.created events.
        if topic == "artifact.created" and payload.get("type") != "ml.model_definition":
            return

        model_definition_id = payload.get("model_definition_id")
        task = payload.get("task")

        def refresh_and_select() -> None:
            if getattr(self, "_disposed", False):
                return

            # If a newly saved model belongs to a different visible task, switch
            # first. The task watcher will call _load_models().
            if task and task in self.task.options and self.task.value != task:
                self.task.value = task
                return

            self._load_models(force_reacquire_registry=True)

            if model_definition_id:
                try:
                    valid_ids = set(self.model.options.values())
                    if model_definition_id in valid_ids:
                        self.model.value = model_definition_id
                        self._update_model_note()
                except Exception:
                    pass

        self._schedule_ui_update(refresh_and_select)

    def _on_train_done(self, result: Dict[str, Any]) -> None:
        if self._disposed:
            return

        result = result or {}
        self._clear_active_training_state()

        self.status.alert_type = "success"
        self.status.object = f"Training complete. Run `{result.get('run_id')}`."

        tuning = result.get("tuning") or {}
        self.metrics.object = {
            "metrics": result.get("metrics", {}),
            "tuning": {
                key: value
                for key, value in tuning.items()
                if key != "trials"
            },
        }

        self.predictions.object = pd.DataFrame(result.get("prediction_preview", []))

        self.tuning_summary.object = self._format_tuning_summary(tuning)
        self.tuning_trials.object = pd.DataFrame(tuning.get("trials") or [])

        artifact_ids = result.get("artifact_ids", {}) or {}
        model_artifact_id = result.get("model_artifact_id")
        if model_artifact_id and "model" not in artifact_ids:
            artifact_ids["model"] = model_artifact_id

        lines = ["### Created artifacts"]
        if artifact_ids:
            for name, artifact_id in artifact_ids.items():
                lines.append(f"- `{name}`: `{artifact_id}`")
        else:
            lines.append("No artifact ids were returned.")

        if result.get("durable_model"):
            lines.append("")
            lines.append("Durable model sidecar saved for the `ml.model` artifact.")

        if tuning.get("enabled"):
            lines.append("")
            lines.append("Optuna tuning details are available in the **Optuna tuning** tab and the **ML Training Curves** panel.")

        self.artifacts.object = "\n".join(lines)

        if self.root_tabs is not None:
            self.root_tabs.active = 1

    def _on_train_error(self, error: BaseException) -> None:
        if self._disposed:
            return

        was_cancelled = self._is_cancel_error(error)

        if was_cancelled:
            self._update_active_training_log(
                status="cancelled",
                message="Training cancelled by user.",
            )
        else:
            self._update_active_training_log(
                status="error",
                message=f"Training failed: {error}",
            )

        self._clear_active_training_state()

        if was_cancelled:
            self.status.alert_type = "warning"
            self.status.object = "Training cancelled."
        else:
            self.status.alert_type = "danger"
            self.status.object = f"Training failed: {error}"