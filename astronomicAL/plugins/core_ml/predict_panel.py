from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
import panel as pn

from astronomicAL.platform.plugins.specs import ActionRequest


def _load_sibling_module(stem: str):
    module_name = f"{__name__}.{stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = Path(__file__).with_name(f"{stem}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sibling module {stem!r} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MLPredictPanel:
    """Predict with a trained ml.model artifact.

    This layout intentionally avoids nested Accordions, forced 100% heights,
    manual overflow CSS, and divider-heavy sections. Those combinations can
    collapse inside the dynamic workspace grid and cause labels/buttons to
    overlap.
    """

    def __init__(self, *, context: Any, restore_state: Optional[Dict[str, Any]] = None) -> None:
        self.context = context
        self._disposed = False
        self._subscriptions: List[Any] = []
        self._active_job_handle = None
        self._active_job_key = None
        self._active_job_id = None

        self.dataset = pn.widgets.Select(
            name="Dataset to predict on",
            options=[],
            sizing_mode="stretch_width",
        )
        self.model = pn.widgets.Select(
            name="Trained model",
            options={},
            sizing_mode="stretch_width",
        )
        self.target = pn.widgets.Select(
            name="Evaluation label column",
            options=[],
            sizing_mode="stretch_width",
        )
        self.image_column = pn.widgets.Select(
            name="Image column override",
            options=[],
            sizing_mode="stretch_width",
        )

        self.register_dataset = pn.widgets.Checkbox(
            name="Create a prediction-table dataset for plotting and colouring",
            value=True,
            sizing_mode="stretch_width",
        )
        self.require_target_compatible = pn.widgets.Checkbox(
            name="Block prediction if evaluation labels do not match the model classes",
            value=False,
            sizing_mode="stretch_width",
        )

        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="light",
            height=36,
            sizing_mode="stretch_width",
        )
        self.validate_button = pn.widgets.Button(
            name="Check compatibility",
            button_type="light",
            height=36,
            sizing_mode="stretch_width",
        )
        self.predict_button = pn.widgets.Button(
            name="Predict",
            button_type="primary",
            height=40,
            sizing_mode="stretch_width",
        )
        self.cancel_button = pn.widgets.Button(
            name="Cancel",
            button_type="warning",
            disabled=True,
            height=40,
            sizing_mode="stretch_width",
        )

        self.status = pn.pane.Alert(
            "Waiting for trained models. Train a model in the Workbench; this panel will update automatically.",
            alert_type="info",
            sizing_mode="stretch_width",
            margin=(0, 0, 10, 0),
        )

        self.compatibility_title = pn.pane.Markdown(
            "### Compatibility",
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
        )
        self.compatibility_summary = pn.pane.Markdown(
            "No trained model is selected yet.",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        self.show_details = pn.widgets.Checkbox(
            name="Show detailed compatibility JSON",
            value=False,
            sizing_mode="stretch_width",
            margin=(2, 0, 8, 0),
        )
        self.compatibility_json = pn.pane.JSON(
            {},
            depth=3,
            sizing_mode="stretch_width",
            height=260,
            visible=False,
            margin=(0, 0, 0, 0),
        )

        self.outputs = pn.pane.Markdown(
            "No predictions have been created yet.",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        self.preview = pn.widgets.Tabulator(
            pd.DataFrame(),
            height=260,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.refresh_button.on_click(lambda *_: self.refresh(reason="manual refresh"))
        self.validate_button.on_click(lambda *_: self.validate(show_feedback=True))
        self.predict_button.on_click(lambda *_: self.predict())
        self.cancel_button.on_click(lambda *_: self.cancel())

        self.dataset.param.watch(lambda *_: self._on_dataset_change(), "value")
        self.model.param.watch(lambda *_: self.validate(show_feedback=False), "value")
        self.target.param.watch(lambda *_: self.validate(show_feedback=False), "value")
        self.image_column.param.watch(lambda *_: self.validate(show_feedback=False), "value")
        self.show_details.param.watch(lambda event: self._toggle_details(bool(event.new)), "value")

        self._subscribe_to_events()
        self.refresh(reason="panel opened")

        if restore_state:
            self.restore_state(restore_state)

    def panel(self):
        content = pn.Column(
            self._header_card(),
            self._model_card(),
            self._compatibility_card(),
            self._advanced_card(),
            self._outputs_card(),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        scroll_area = pn.Column(
            content,
            sizing_mode="stretch_both",
            scroll=True,
            margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box",
                "height": "100%",
                "min-height": "0",
                "max-height": "100%",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "padding": "0 6px 12px 0",
            },
        )

        return pn.Column(
            scroll_area,
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box",
                "height": "100%",
                "min-height": "0",
                "overflow": "hidden",
            },
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset.value,
            "model": self.model.value,
            "target": self.target.value,
            "image_column": self.image_column.value,
            "register_dataset": bool(self.register_dataset.value),
            "require_target_compatible": bool(self.require_target_compatible.value),
            "show_details": bool(self.show_details.value),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        self.refresh(reason="restore state")

        if state.get("dataset") in self.dataset.options:
            self.dataset.value = state["dataset"]

        self._load_columns()

        model_values = self._model_values()
        if state.get("model") in model_values:
            self.model.value = state["model"]

        if state.get("target") in self.target.options:
            self.target.value = state["target"]

        if state.get("image_column") in self.image_column.options:
            self.image_column.value = state["image_column"]

        self.register_dataset.value = bool(state.get("register_dataset", self.register_dataset.value))
        self.require_target_compatible.value = bool(
            state.get("require_target_compatible", self.require_target_compatible.value)
        )
        self.show_details.value = bool(state.get("show_details", self.show_details.value))
        self.validate(show_feedback=True)

    def dispose(self) -> None:
        self._disposed = True
        self.cancel()

        events = getattr(self.context, "events", None)
        unsubscribe = getattr(events, "unsubscribe", None)
        if callable(unsubscribe):
            for sub in list(self._subscriptions):
                try:
                    unsubscribe(sub)
                except Exception:
                    pass

        self._subscriptions.clear()

    def _header_card(self):
        return self._card(
            pn.pane.Markdown(
                "## ML Predictor\n"
                "Use a trained model artifact to predict on the selected dataset.",
                sizing_mode="stretch_width",
                margin=(0, 0, 6, 0),
            ),
            self.status,
        )

    def _model_card(self):
        return self._card(
            pn.pane.Markdown("### Inputs", margin=(0, 0, 8, 0)),
            self.dataset,
            pn.Spacer(height=8),
            self.model,
            pn.Spacer(height=10),
            pn.GridBox(
                self.refresh_button,
                self.validate_button,
                ncols=2,
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            ),
            pn.GridBox(
                self.predict_button,
                self.cancel_button,
                ncols=2,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            ),
        )

    def _compatibility_card(self):
        return self._card(
            self.compatibility_title,
            self.compatibility_summary,
            self.show_details,
            self.compatibility_json,
        )

    def _advanced_card(self):
        return self._card(
            pn.pane.Markdown("### Advanced options", margin=(0, 0, 8, 0)),
            self.target,
            pn.Spacer(height=8),
            self.image_column,
            pn.Spacer(height=8),
            self.register_dataset,
            self.require_target_compatible,
        )

    def _outputs_card(self):
        return self._card(
            pn.pane.Markdown("### Prediction outputs", margin=(0, 0, 8, 0)),
            self.outputs,
            pn.pane.Markdown("#### Preview", margin=(8, 0, 6, 0)),
            self.preview,
        )

    def _card(self, *objects):
        return pn.Column(
            *objects,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
            styles={
                "box-sizing": "border-box",
                "padding": "12px",
                "border": "1px solid #d9dee8",
                "border-radius": "8px",
                "background": "white",
            },
        )

    def _toggle_details(self, visible: bool) -> None:
        self.compatibility_json.visible = bool(visible)

    def refresh(self, *, reason: str = "refresh", select_model_artifact_id: Optional[str] = None) -> None:
        if self._disposed:
            return

        previous_model = self.model.value

        self._load_datasets()
        self._load_columns()
        self._load_models()

        model_values = self._model_values()

        if select_model_artifact_id and select_model_artifact_id in model_values:
            self.model.value = select_model_artifact_id
            self.status.alert_type = "success"
            self.status.object = "New trained model detected and selected automatically."
        elif previous_model in model_values:
            self.model.value = previous_model

        self.validate(show_feedback=False)

    def validate(self, *, show_feedback: bool = True) -> None:
        if self._disposed:
            return

        dataset_id = self._dataset_id()
        model_id = self._model_artifact_id()

        if not dataset_id:
            self.compatibility_json.object = {}
            self.compatibility_summary.object = "Choose a dataset."
            self.predict_button.disabled = True
            return

        if not model_id:
            self.compatibility_json.object = {}
            self.compatibility_summary.object = (
                "No trained model is selected. Train a model in the Workbench; "
                "this panel will refresh automatically when training finishes."
            )
            self.predict_button.disabled = True
            return

        catalog = self._catalog()
        if catalog is None:
            self.status.alert_type = "danger"
            self.status.object = "Trained model catalog service is not available."
            self.compatibility_summary.object = "Cannot check compatibility because the catalog service is unavailable."
            self.predict_button.disabled = True
            return

        try:
            report = catalog.compatibility(
                model_id,
                dataset_id,
                target_column=self.target.value,
                image_column=self.image_column.value,
                require_target_compatible=bool(self.require_target_compatible.value),
            )
        except Exception as exc:
            self.status.alert_type = "danger"
            self.status.object = f"Compatibility check failed: {exc}"
            self.compatibility_summary.object = f"Compatibility check failed: `{exc}`"
            self.compatibility_json.object = {"error": str(exc)}
            self.predict_button.disabled = True
            return

        self.compatibility_json.object = report
        self.compatibility_summary.object = self._format_compatibility_report(report)

        status = report.get("status")
        if status == "compatible":
            self.predict_button.disabled = False
            if show_feedback:
                self.status.alert_type = "success"
                self.status.object = "Compatibility check passed. Prediction can run."
        elif status == "warning":
            self.predict_button.disabled = False
            if show_feedback:
                self.status.alert_type = "warning"
                self.status.object = "Prediction can run, but there are compatibility warnings."
        else:
            self.predict_button.disabled = True
            if show_feedback:
                self.status.alert_type = "danger"
                self.status.object = "Prediction is blocked by compatibility errors."

    def predict(self) -> None:
        dataset_id = self._dataset_id()
        model_id = self._model_artifact_id()

        if not dataset_id or not model_id:
            self.status.alert_type = "danger"
            self.status.object = "Choose both a dataset and a trained model."
            return

        self.validate(show_feedback=True)
        if self.predict_button.disabled:
            return

        request = ActionRequest(
            dataset_id=dataset_id,
            params={
                "dataset_id": dataset_id,
                "model_artifact_id": model_id,
                "target_column": self.target.value,
                "image_column": self.image_column.value,
                "register_prediction_dataset": bool(self.register_dataset.value),
                "require_target_compatible": bool(self.require_target_compatible.value),
                "run_id": uuid.uuid4().hex,
            },
            artifact_id=model_id,
            origin="core.ml.predict_panel",
        )

        self._set_running(True)
        self.status.alert_type = "info"
        self.status.object = "Prediction started..."

        prediction = _load_sibling_module("prediction")
        jobs = getattr(self.context, "jobs", None)
        submit = getattr(jobs, "submit", None)
        key = f"core.ml.predict:{dataset_id}:{model_id}"

        if callable(submit):
            self._active_job_key = key
            submitted = submit(
                prediction.predict_action,
                title="Predict with trained ML model",
                key=key,
                on_done=self._on_predict_done,
                on_error=self._on_predict_error,
                context=self.context,
                request=request,
            )
            self._active_job_handle = submitted
            self._active_job_id = getattr(submitted, "job_id", None) or getattr(submitted, "id", None)
            return

        try:
            result = prediction.predict_action(context=self.context, request=request)
            self._on_predict_done(result)
        except Exception as exc:
            self._on_predict_error(exc)

    def cancel(self) -> None:
        jobs = getattr(self.context, "jobs", None)

        for method_name, value in (
            ("cancel", self._active_job_id),
            ("cancel", self._active_job_key),
            ("cancel_job", self._active_job_id),
            ("cancel_job", self._active_job_key),
            ("cancel_by_key", self._active_job_key),
        ):
            if not value:
                continue

            method = getattr(jobs, method_name, None)
            if callable(method):
                try:
                    method(value)
                    break
                except TypeError:
                    continue
                except Exception:
                    break

        self._set_running(False)

    def _on_predict_done(self, result: Any) -> None:
        self._set_running(False)

        if not isinstance(result, dict):
            self.status.alert_type = "danger"
            self.status.object = f"Prediction returned unexpected result: {result!r}"
            return

        if not result.get("ok", True):
            self.status.alert_type = "danger"
            self.status.object = "Prediction failed validation."
            report = result.get("compatibility_report") or {}
            self.compatibility_json.object = report
            self.compatibility_summary.object = self._format_compatibility_report(report)
            return

        preview = result.get("prediction_preview") or []
        self.preview.value = pd.DataFrame(preview)

        self.status.alert_type = "success"
        self.status.object = f"Prediction complete for {result.get('count', 0)} rows."

        recommended = result.get("recommended_color_columns") or []
        derived_dataset_id = result.get("derived_dataset_id")

        lines = [
            "Prediction finished successfully.",
            "",
            f"- **Predictions artifact:** `{result.get('artifact_id')}`",
        ]

        if derived_dataset_id:
            lines.extend(
                [
                    f"- **Prediction table dataset:** `{derived_dataset_id}`",
                    "",
                    (
                        "The prediction-table dataset contains one row per predicted record and can be opened or used by "
                        "visualisation/table tools without changing the original source dataset."
                    ),
                ]
            )
        else:
            lines.append("- No prediction-table dataset was created.")

        if recommended:
            lines.extend(
                [
                    "",
                    "**Useful colour-by columns for visualisation:**",
                    *[f"- `{column}`" for column in recommended],
                    "",
                    (
                        "`predicted_label` colours points by the predicted class. "
                        "`prediction_confidence` colours by model certainty. "
                        "`entropy`, `least_confidence`, `margin_uncertainty`, and `prob_*` columns help find uncertain "
                        "or class-specific records for active-learning review."
                    ),
                ]
            )

        self.outputs.object = "\n".join(lines)
        self.refresh(reason="prediction complete")

    def _on_predict_error(self, error: Any) -> None:
        self._set_running(False)
        self.status.alert_type = "danger"
        self.status.object = f"Prediction failed: {error}"

    def _subscribe_to_events(self) -> None:
        events = getattr(self.context, "events", None)
        subscribe = getattr(events, "subscribe", None)

        if not callable(subscribe):
            return

        for topic in [
            "ml.run.finished",
            "ml.model.saved",
            "artifact.created",
            "artifact.updated",
            "workspace.restored",
            "dataset.loaded",
            "dataset.active.changed",
            "dataset.mapping.updated",
            "ml.predictions.created",
        ]:
            try:
                sub = subscribe(
                    topic,
                    self._on_platform_event,
                    owner_label="ML Predictor",
                    owner_kind="panel",
                )
            except TypeError:
                try:
                    sub = subscribe(topic, self._on_platform_event)
                except Exception:
                    continue
            except Exception:
                continue
            self._subscriptions.append(sub)

    def _on_platform_event(self, *args: Any, **kwargs: Any) -> None:
        if self._disposed:
            return

        topic, payload = self._normalise_event_args(*args, **kwargs)

        select_model_id = None

        if topic == "ml.run.finished":
            artifact_ids = payload.get("artifact_ids") or {}
            select_model_id = artifact_ids.get("model") or payload.get("model_artifact_id")

        elif topic in {"artifact.created", "artifact.updated"}:
            artifact_type = payload.get("type") or payload.get("artifact_type")
            if artifact_type not in {"ml.model", "ml.model_definition", None}:
                return
            select_model_id = payload.get("artifact_id") if artifact_type == "ml.model" else None

        elif topic == "ml.model.saved":
            select_model_id = payload.get("artifact_id") or payload.get("model_artifact_id")

        def update() -> None:
            if self._disposed:
                return
            self.refresh(reason=topic, select_model_artifact_id=select_model_id)

        self._schedule_ui_update(update)

    def _normalise_event_args(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        topic = str(kwargs.get("topic") or kwargs.get("event") or "")
        payload = kwargs.get("payload")

        if len(args) >= 2:
            topic = str(args[0])
            payload = args[1]
        elif len(args) == 1:
            if isinstance(args[0], dict):
                payload = args[0]
                topic = str(payload.get("topic") or payload.get("event") or "")
            else:
                topic = str(args[0])

        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        return topic, payload

    def _schedule_ui_update(self, callback) -> None:
        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(callback)
                return
        except Exception:
            pass
        callback()

    def _load_datasets(self) -> None:
        try:
            options = list(self.context.datasets.list_ids())
        except Exception:
            options = []

        current = self.dataset.value
        self.dataset.options = options

        if current in options:
            self.dataset.value = current
        elif options:
            try:
                active = self.context.datasets.active_id()
            except Exception:
                active = None
            self.dataset.value = active if active in options else options[0]
        else:
            self.dataset.value = None

    def _load_columns(self) -> None:
        dataset_id = self._dataset_id()
        columns: List[str] = []

        if dataset_id:
            try:
                columns = [str(c) for c in self.context.datasets.list_columns(dataset_id)]
            except Exception:
                try:
                    columns = [str(c) for c in self.context.datasets.get_source(dataset_id).columns()]
                except Exception:
                    columns = []

        self.target.options = [None] + columns
        self.image_column.options = [None] + columns

        if self.target.value not in self.target.options:
            self.target.value = self._guess_target(columns)

        if self.image_column.value not in self.image_column.options:
            self.image_column.value = self._guess_image_column(columns)

    def _load_models(self) -> None:
        catalog = self._catalog()
        options: Dict[str, str] = {}

        if catalog is not None:
            try:
                catalog.refresh()
                options = catalog.as_options()
            except Exception:
                options = {}

        current = self.model.value
        self.model.options = options

        values = set(options.values())
        if current in values:
            self.model.value = current
        elif options:
            self.model.value = next(iter(options.values()))
        else:
            self.model.value = None

    def _on_dataset_change(self) -> None:
        self._load_columns()
        self.validate(show_feedback=False)

    def _catalog(self) -> Any:
        services = getattr(self.context, "services", None)
        get = getattr(services, "get", None)

        if not callable(get):
            return None

        try:
            return get("core.ml.trained_model_catalog")
        except Exception:
            return None

    def _dataset_id(self) -> Optional[str]:
        return str(self.dataset.value) if self.dataset.value else None

    def _model_artifact_id(self) -> Optional[str]:
        return str(self.model.value) if self.model.value else None

    def _model_values(self) -> set:
        if isinstance(self.model.options, dict):
            return set(self.model.options.values())
        return set(self.model.options or [])

    def _set_running(self, running: bool) -> None:
        self.predict_button.disabled = bool(running)
        self.cancel_button.disabled = not bool(running)
        self.refresh_button.disabled = bool(running)
        self.validate_button.disabled = bool(running)

    def _format_compatibility_report(self, report: Mapping[str, Any]) -> str:
        if not report:
            return "No compatibility report yet."

        status = report.get("status", "unknown")
        model = report.get("model_summary") or {}
        outputs = report.get("output_schema") or {}
        binding = report.get("resolved_input_binding") or {}
        warnings = report.get("warnings") or []
        errors = report.get("errors") or []
        recommended = report.get("recommended_prediction_columns") or []

        icon = {
            "compatible": "✅",
            "warning": "⚠️",
            "needs_mapping": "🧭",
            "incompatible": "⛔",
        }.get(str(status), "ℹ️")

        lines = [
            f"{icon} **Status:** `{status}`",
            f"**Model:** {model.get('title', '') or 'Unknown model'}",
            f"**Task/modality:** `{model.get('task', '')}` / `{model.get('modality', '')}`",
        ]

        classes = outputs.get("classes") or []
        if classes:
            lines.append("**Predicts classes:** " + ", ".join(f"`{label}`" for label in classes))

        if binding.get("kind") == "tabular":
            features = binding.get("feature_columns") or []
            lines.append(f"**Resolved tabular features:** `{len(features)}` column(s)")
            if len(features) <= 12 and features:
                lines.append("`" + "`, `".join(features) + "`")

        if binding.get("kind") == "image":
            image_column = binding.get("image_column")
            lines.append(f"**Resolved image column:** `{image_column}`")

        target = binding.get("target_column_for_evaluation")
        if target:
            lines.append(f"**Evaluation label column:** `{target}`")
        else:
            lines.append("**Evaluation label column:** none selected; predictions can still run.")

        if warnings:
            lines.append("")
            lines.append("**Warnings:**")
            lines.extend(f"- {warning}" for warning in warnings)

        if errors:
            lines.append("")
            lines.append("**Errors:**")
            lines.extend(f"- {error}" for error in errors)

        if recommended:
            lines.append("")
            lines.append(
                "**Prediction columns available for visualisation after prediction:** "
                + ", ".join(f"`{column}`" for column in recommended)
            )

        return "  \n".join(lines)

    def _guess_target(self, columns: List[str]) -> Optional[str]:
        dataset_id = self._dataset_id()

        if dataset_id:
            try:
                mapped = self.context.datasets.get_mapping(dataset_id, "target_label")
                if mapped in columns:
                    return mapped
            except Exception:
                pass

        lowered = {c.lower(): c for c in columns}
        for candidate in ("target_label", "target", "label", "class", "classification", "y"):
            if candidate in lowered:
                return lowered[candidate]

        return None

    def _guess_image_column(self, columns: List[str]) -> Optional[str]:
        dataset_id = self._dataset_id()

        if dataset_id:
            for semantic in ("image.path", "image.uri", "image.url"):
                try:
                    mapped = self.context.datasets.get_mapping(dataset_id, semantic)
                    if mapped in columns:
                        return mapped
                except Exception:
                    pass

        lowered = {c.lower(): c for c in columns}
        for candidate in ("image", "image_path", "image_uri", "image_url", "img", "path", "file", "filename", "cutout"):
            if candidate in lowered:
                return lowered[candidate]

        for column in columns:
            low = column.lower()
            if any(token in low for token in ("image", "img", "path", "uri", "url", "cutout", "jpg", "png")):
                return column

        return None


def create_predict_panel(context: Any, **kwargs: Any):
    controller = MLPredictPanel(context=context, restore_state=kwargs.get("restore_state"))
    return controller.panel(), controller