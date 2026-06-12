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
        self._last_derived_dataset_id: Optional[str] = None
        self._last_predictions_artifact_id: Optional[str] = None

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

        self.prediction_mode = pn.widgets.RadioButtonGroup(
            name="Prediction mode",
            options=["Predict only", "Predict + evaluate"],
            value="Predict only",
            button_type="default",
            sizing_mode="stretch_width",
        )

        self.row_scope = pn.widgets.Select(
            name="Rows",
            options=["All rows", "Active selection", "Focused row"],
            value="All rows",
            sizing_mode="stretch_width",
        )

        self.max_rows = pn.widgets.IntInput(
            name="Maximum rows (0 = all)",
            value=0,
            start=0,
            sizing_mode="stretch_width",
        )

        self.device = pn.widgets.Select(
            name="Device",
            options=["auto", "cpu", "cuda", "mps"],
            value="auto",
            sizing_mode="stretch_width",
        )

        self.batch_size = pn.widgets.IntInput(
            name="Batch size",
            value=64,
            start=1,
            sizing_mode="stretch_width",
        )

        self.skip_bad_images = pn.widgets.Checkbox(
            name="Skip unreadable images",
            value=True,
            sizing_mode="stretch_width",
        )

        self.set_active_dataset = pn.widgets.Checkbox(
            name="Set prediction-table dataset active after run",
            value=True,
            sizing_mode="stretch_width",
        )

        self.model_summary = pn.pane.Markdown(
            "No model selected.",
            sizing_mode="stretch_width",
            margin=(6, 0, 8, 0),
        )

        self.activate_dataset_button = pn.widgets.Button(
            name="Set last prediction dataset active",
            button_type="light",
            disabled=True,
            height=36,
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
        self.activate_dataset_button.on_click(lambda *_: self._activate_last_prediction_dataset())

        self.dataset.param.watch(lambda *_: self._on_dataset_change(), "value")
        self.model.param.watch(lambda *_: self._on_model_change(), "value")
        self.prediction_mode.param.watch(lambda *_: self._sync_visibility_and_validate(), "value")
        self.row_scope.param.watch(lambda *_: self.validate(show_feedback=False), "value")
        self.max_rows.param.watch(lambda *_: self.validate(show_feedback=False), "value")
        self.target.param.watch(lambda *_: self.validate(show_feedback=False), "value")
        self.image_column.param.watch(lambda *_: self.validate(show_feedback=False), "value")
        self.device.param.watch(lambda *_: self.validate(show_feedback=False), "value")
        self.batch_size.param.watch(lambda *_: self.validate(show_feedback=False), "value")
        self.skip_bad_images.param.watch(lambda *_: self.validate(show_feedback=False), "value")
        self.show_details.param.watch(lambda event: self._toggle_details(bool(event.new)), "value")

        self._subscribe_to_events()
        self.refresh(reason="panel opened")

        if restore_state:
            self.restore_state(restore_state)

    def panel(self):
        """Return a compact, workspace-safe predictor layout.

        The predictor is often opened inside a constrained GridStack tile. Avoid
        nested 100%-height scroll containers and stacked bordered cards, because
        those can cause Bokeh/Panel to underestimate section heights and overlap
        widget labels.
        """
        setup_tab = pn.Column(
            self._model_card(),
            self._advanced_card(),
            self._run_card(),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        tabs = pn.Tabs(
            ("Setup", setup_tab),
            ("Compatibility", self._compatibility_card()),
            ("Results", self._outputs_card()),
            dynamic=True,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        return pn.Column(
            self._header_card(),
            tabs,
            sizing_mode="stretch_both",
            scroll=True,
            margin=(0, 0, 0, 0),
            styles={
                "box-sizing": "border-box",
                "padding": "10px",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "background": "white",
            },
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset.value,
            "model": self.model.value,
            "prediction_mode": self.prediction_mode.value,
            "row_scope": self.row_scope.value,
            "max_rows": int(self.max_rows.value or 0),
            "target": self.target.value,
            "image_column": self.image_column.value,
            "device": self.device.value,
            "batch_size": int(self.batch_size.value or 64),
            "skip_bad_images": bool(self.skip_bad_images.value),
            "register_dataset": bool(self.register_dataset.value),
            "set_active_dataset": bool(self.set_active_dataset.value),
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

        if state.get("prediction_mode") in self.prediction_mode.options:
            self.prediction_mode.value = state["prediction_mode"]

        if state.get("row_scope") in self.row_scope.options:
            self.row_scope.value = state["row_scope"]

        try:
            self.max_rows.value = int(state.get("max_rows", self.max_rows.value) or 0)
        except Exception:
            pass

        if state.get("target") in self.target.options:
            self.target.value = state["target"]

        if state.get("image_column") in self.image_column.options:
            self.image_column.value = state["image_column"]

        if state.get("device") in self.device.options:
            self.device.value = state["device"]

        try:
            self.batch_size.value = int(state.get("batch_size", self.batch_size.value) or 64)
        except Exception:
            pass

        self.skip_bad_images.value = bool(state.get("skip_bad_images", self.skip_bad_images.value))
        self.register_dataset.value = bool(state.get("register_dataset", self.register_dataset.value))
        self.set_active_dataset.value = bool(state.get("set_active_dataset", self.set_active_dataset.value))
        self.require_target_compatible.value = bool(
            state.get("require_target_compatible", self.require_target_compatible.value)
        )
        self.show_details.value = bool(state.get("show_details", self.show_details.value))

        self._sync_visibility()
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
        return pn.Column(
            pn.pane.HTML(
                """
                <div style="
                    font-size: 20px;
                    font-weight: 700;
                    line-height: 26px;
                    margin: 0 0 6px 0;
                ">
                    ML Predictor
                </div>
                <div style="
                    font-size: 13px;
                    line-height: 18px;
                    margin: 0 0 10px 0;
                    color: #2b2f36;
                ">
                    Use a trained model artifact to predict on the selected dataset.
                </div>
                """,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            ),
            self.status,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
            styles={
                "box-sizing": "border-box",
                "padding": "12px",
                "border": "1px solid #d9dee8",
                "border-radius": "8px",
                "background": "white",
                "overflow": "visible",
            },
        )
    
    def _run_card(self):
        return self._section(
            "Run",
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

    def _model_card(self):
        objects = [
            self.dataset,
            self.model,
        ]

        model_summary = getattr(self, "model_summary", None)
        if model_summary is not None:
            objects.append(model_summary)

        return self._section(
            "Model and dataset",
            *objects,
        )

    def _compatibility_card(self):
        return self._section(
            "Compatibility",
            self.compatibility_summary,
            self.show_details,
            self.compatibility_json,
        )

    def _advanced_card(self):
        main_controls = []

        prediction_mode = getattr(self, "prediction_mode", None)
        if prediction_mode is not None:
            main_controls.append(prediction_mode)

        row_scope = getattr(self, "row_scope", None)
        if row_scope is not None:
            main_controls.append(row_scope)

        max_rows = getattr(self, "max_rows", None)
        if max_rows is not None:
            main_controls.append(max_rows)

        if getattr(self, "target", None) is not None:
            main_controls.append(self.target)

        if getattr(self, "image_column", None) is not None:
            main_controls.append(self.image_column)

        runtime_controls = []

        for name in (
            "device",
            "batch_size",
            "skip_bad_images",
            "register_dataset",
            "set_active_dataset",
            "require_target_compatible",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                runtime_controls.append(widget)

        if runtime_controls:
            runtime_box = pn.Column(
                *runtime_controls,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )

            main_controls.append(
                pn.Accordion(
                    ("Runtime and output options", runtime_box),
                    active=[],
                    sizing_mode="stretch_width",
                    margin=(0, 0, 0, 0),
                )
            )

        return self._section(
            "Prediction setup",
            *main_controls,
        )

    def _outputs_card(self):
        objects = [
            self.outputs,
        ]

        activate_button = getattr(self, "activate_dataset_button", None)
        if activate_button is not None:
            objects.append(activate_button)

        objects.extend(
            [
                pn.pane.HTML(
                    """
                    <div style="
                        font-size: 13px;
                        font-weight: 700;
                        line-height: 18px;
                        margin: 10px 0 6px 0;
                    ">
                        Preview
                    </div>
                    """,
                    sizing_mode="stretch_width",
                    margin=(0, 0, 0, 0),
                ),
                self.preview,
            ]
        )

        return self._section(
            "Prediction outputs",
            *objects,
        )

    def _section_title(self, title: str):
        return pn.pane.HTML(
            f"""
            <div style="
                display: block;
                box-sizing: border-box;
                font-size: 14px;
                font-weight: 700;
                line-height: 20px;
                margin: 0;
                padding: 0 0 10px 0;
                color: #20242a;
            ">
                {title}
            </div>
            """,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )


    def _section(self, title: str, *objects):
        clean_objects = []

        for obj in objects:
            if obj is None:
                continue

            # Normalise margins so widget labels have enough vertical breathing room.
            try:
                obj.margin = (0, 0, 12, 0)
            except Exception:
                pass

            clean_objects.append(obj)

        return pn.Column(
            self._section_title(title),
            *clean_objects,
            sizing_mode="stretch_width",
            margin=(0, 0, 14, 0),
            styles={
                "box-sizing": "border-box",
                "display": "block",
                "padding": "14px",
                "border": "1px solid #d9dee8",
                "border-radius": "8px",
                "background": "white",
                "overflow": "visible",
                "min-height": "auto",
                "height": "auto",
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
        elif options := getattr(self.model, "options", None):
            if isinstance(options, dict) and options:
                self.model.value = next(iter(options.values()))
            elif isinstance(options, (list, tuple)) and options:
                self.model.value = options[0]

        self._update_model_summary()
        self._sync_visibility()
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
            self.status.alert_type = "info"
            self.status.object = "Choose a dataset to predict on."
            return

        if not model_id:
            self.compatibility_json.object = {}
            self.compatibility_summary.object = (
                "No trained model is selected. Train a model in the Workbench or Recipe Launcher; "
                "this panel will refresh automatically when training finishes."
            )
            self.predict_button.disabled = True
            self.status.alert_type = "info"
            self.status.object = "No trained model is selected."
            return

        catalog = self._catalog()

        if catalog is None:
            self.status.alert_type = "danger"
            self.status.object = "Trained model catalog service is not available."
            self.compatibility_summary.object = "Cannot check compatibility because the catalog service is unavailable."
            self.predict_button.disabled = True
            return

        evaluation_mode = self._is_evaluation_mode()

        try:
            report = catalog.compatibility(
                model_id,
                dataset_id,
                target_column=self._target_column_param(),
                image_column=self._image_column_param(),
                require_target_compatible=(
                    bool(self.require_target_compatible.value)
                    if evaluation_mode
                    else False
                ),
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
            self.status.alert_type = "success"
            self.status.object = (
                "Ready to predict and evaluate."
                if evaluation_mode
                else "Ready to predict."
            )
        elif status == "warning":
            self.predict_button.disabled = False
            self.status.alert_type = "warning"
            self.status.object = "Ready to predict, but check compatibility warnings."
        else:
            self.predict_button.disabled = True
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

        row_ids = self._row_ids_for_scope()

        params = {
            "dataset_id": dataset_id,
            "model_artifact_id": model_id,
            "target_column": self._target_column_param(),
            "image_column": self._image_column_param(),
            "register_prediction_dataset": bool(self.register_dataset.value),
            "require_target_compatible": (
                bool(self.require_target_compatible.value)
                if self._is_evaluation_mode()
                else False
            ),
            "prediction_scope": "evaluation" if self._is_evaluation_mode() else "inference",
            "device": str(self.device.value or "auto"),
            "image_batch_size": int(self.batch_size.value or 64),
            "batch_size": int(self.batch_size.value or 64),
            "skip_bad_images": bool(self.skip_bad_images.value),
            "max_rows": int(self.max_rows.value or 0),
            "run_id": uuid.uuid4().hex,
        }

        request = ActionRequest(
            dataset_id=dataset_id,
            row_ids=row_ids,
            params=params,
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

        artifact_id = result.get("artifact_id")
        derived_dataset_id = result.get("derived_dataset_id")

        self._last_predictions_artifact_id = str(artifact_id) if artifact_id else None
        self._last_derived_dataset_id = str(derived_dataset_id) if derived_dataset_id else None
        self.activate_dataset_button.disabled = not bool(self._last_derived_dataset_id)

        if self._last_derived_dataset_id and bool(self.set_active_dataset.value):
            self._set_dataset_active(self._last_derived_dataset_id)

        count = result.get("count", 0)
        failed_image_row_count = int(result.get("failed_image_row_count") or 0)

        self.status.alert_type = "success" if failed_image_row_count == 0 else "warning"
        self.status.object = (
            f"Prediction complete for {count} rows."
            if failed_image_row_count == 0
            else f"Prediction complete for {count} rows; skipped {failed_image_row_count} unreadable image row(s)."
        )

        recommended = result.get("recommended_color_columns") or []
        lines = [
            "Prediction finished successfully.",
            "",
            f"- **Predictions artifact:** `{artifact_id}`",
        ]

        if derived_dataset_id:
            lines.extend(
                [
                    f"- **Prediction table dataset:** `{derived_dataset_id}`",
                    "",
                    (
                        "The prediction-table dataset contains one row per predicted record "
                        "and can be used directly by visualisation/table tools."
                    ),
                ]
            )
        else:
            lines.append("- No prediction-table dataset was created.")

        if failed_image_row_count:
            lines.extend(
                [
                    "",
                    f"**Skipped image rows:** `{failed_image_row_count}`",
                ]
            )

            failed_rows = result.get("failed_image_rows") or []
            for failed in failed_rows[:5]:
                lines.append(
                    f"- `{failed.get('row_id')}`: {failed.get('error')}"
                )

        if recommended:
            lines.extend(
                [
                    "",
                    "**Useful colour-by columns for visualisation:**",
                    *[f"- `{column}`" for column in recommended],
                    "",
                    (
                        "`predicted_label` colours points by predicted class. "
                        "`prediction_confidence` colours by certainty. "
                        "`entropy`, `least_confidence`, `margin_uncertainty`, and `prob_*` "
                        "columns help find uncertain or class-specific records."
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
            "ml.recipe_run.finished",
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

        if topic in {"ml.run.finished", "ml.recipe_run.finished"}:
            artifact_ids = payload.get("artifact_ids") or {}
            select_model_id = (
                artifact_ids.get("model")
                or payload.get("model_artifact_id")
                or payload.get("artifact_id")
            )
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
            self.target.value = self._guess_target(columns) if self._is_evaluation_mode() else None

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
        self._sync_visibility()
        self.validate(show_feedback=False)

    def _on_model_change(self) -> None:
        self._update_model_summary()
        self._sync_visibility()
        self.validate(show_feedback=False)


    def _sync_visibility_and_validate(self) -> None:
        self._sync_visibility()
        self.validate(show_feedback=False)


    def _sync_visibility(self) -> None:
        evaluation_mode = self._is_evaluation_mode()
        descriptor = self._selected_model_descriptor()
        modality = str((descriptor or {}).get("modality") or "").lower()

        self.target.visible = evaluation_mode
        self.require_target_compatible.visible = evaluation_mode

        if not evaluation_mode:
            try:
                self.target.value = None
            except Exception:
                pass

        is_image_model = modality == "image" or not modality
        self.image_column.visible = is_image_model
        self.skip_bad_images.visible = is_image_model

        # Runtime controls are useful for image/Torch models and harmless for tabular.
        self.device.visible = True
        self.batch_size.visible = True


    def _is_evaluation_mode(self) -> bool:
        return str(self.prediction_mode.value or "").strip() == "Predict + evaluate"


    def _target_column_param(self) -> Optional[str]:
        if not self._is_evaluation_mode():
            return None

        value = self.target.value
        return str(value) if value else None


    def _image_column_param(self) -> Optional[str]:
        value = self.image_column.value
        return str(value) if value else None


    def _selected_model_descriptor(self) -> Dict[str, Any]:
        catalog = self._catalog()
        model_id = self._model_artifact_id()

        if catalog is None or not model_id:
            return {}

        try:
            descriptor = catalog.get(model_id)
        except Exception:
            return {}

        if hasattr(descriptor, "to_dict") and callable(descriptor.to_dict):
            try:
                return dict(descriptor.to_dict())
            except Exception:
                return {}

        if isinstance(descriptor, Mapping):
            return dict(descriptor)

        return {}


    def _update_model_summary(self) -> None:
        descriptor = self._selected_model_descriptor()

        if not descriptor:
            self.model_summary.object = "No model selected."
            return

        input_summary = descriptor.get("input_summary") or {}
        output_summary = descriptor.get("output_summary") or {}
        metrics = descriptor.get("metrics") or {}

        classes = output_summary.get("classes") or []
        lines = [
            f"**Model artifact:** `{descriptor.get('artifact_id')}`",
            f"**Task/modality:** `{descriptor.get('task')}` / `{descriptor.get('modality')}`",
            f"**Framework:** `{descriptor.get('framework')}`",
        ]

        if descriptor.get("run_id"):
            lines.append(f"**Run:** `{str(descriptor.get('run_id'))}`")

        if descriptor.get("trained_dataset_id"):
            lines.append(f"**Trained on:** `{descriptor.get('trained_dataset_id')}`")

        if descriptor.get("target_column"):
            lines.append(f"**Training label column:** `{descriptor.get('target_column')}`")

        if input_summary.get("image_column"):
            lines.append(f"**Training image column:** `{input_summary.get('image_column')}`")

        if input_summary.get("image_size"):
            lines.append(f"**Image size:** `{input_summary.get('image_size')}`")

        if classes:
            preview = ", ".join(f"`{label}`" for label in classes[:12])
            if len(classes) > 12:
                preview += ", ..."
            lines.append(f"**Classes:** {preview}")

        if metrics:
            metric_bits = []
            for key in ("best_val_accuracy", "best_score", "test_accuracy"):
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    metric_bits.append(f"`{key}`={value:.4f}")
            if metric_bits:
                lines.append("**Metrics:** " + ", ".join(metric_bits))

        self.model_summary.object = "  \n".join(lines)


    def _row_ids_for_scope(self) -> Optional[List[str]]:
        scope = str(self.row_scope.value or "All rows")

        if scope == "Focused row":
            focused = self._focused_row_id()
            return [focused] if focused else []

        if scope == "Active selection":
            return self._active_selection_row_ids()

        return None


    def _focused_row_id(self) -> Optional[str]:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None

        for method_name in (
            "focused_row_id",
            "get_focus",
            "get_focused_row_id",
            "focus_id",
        ):
            method = getattr(selection, method_name, None)
            if callable(method):
                try:
                    value = method()
                except Exception:
                    continue

                if isinstance(value, Mapping):
                    value = (
                        value.get("record_id")
                        or value.get("row_id")
                        or value.get("id")
                    )

                if value is not None:
                    return str(value)

        for attr_name in ("focus", "focused", "focused_id"):
            value = getattr(selection, attr_name, None)
            if isinstance(value, Mapping):
                value = value.get("record_id") or value.get("row_id") or value.get("id")
            if value is not None and not callable(value):
                return str(value)

        return None


    def _active_selection_row_ids(self) -> List[str]:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return []

        for method_name in (
            "active_row_ids",
            "selected_row_ids",
            "get_selected_row_ids",
            "current_selection_ids",
            "get_current_selection",
        ):
            method = getattr(selection, method_name, None)
            if callable(method):
                try:
                    value = method()
                except Exception:
                    continue

                row_ids = self._coerce_row_ids(value)
                if row_ids:
                    return row_ids

        for attr_name in ("row_ids", "selected_ids", "selection", "current_selection"):
            value = getattr(selection, attr_name, None)
            row_ids = self._coerce_row_ids(value)
            if row_ids:
                return row_ids

        return []


    def _coerce_row_ids(self, value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, Mapping):
            for key in ("row_ids", "record_ids", "ids"):
                if key in value:
                    return self._coerce_row_ids(value[key])

            one = value.get("record_id") or value.get("row_id") or value.get("id")
            return [str(one)] if one is not None else []

        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if item is not None]

        return [str(value)]


    def _activate_last_prediction_dataset(self) -> None:
        if self._last_derived_dataset_id:
            self._set_dataset_active(self._last_derived_dataset_id)


    def _set_dataset_active(self, dataset_id: str) -> None:
        datasets = getattr(self.context, "datasets", None)

        for method_name in ("set_active", "activate", "set_active_id"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    method(dataset_id)
                    self.status.alert_type = "success"
                    self.status.object = f"Prediction dataset `{dataset_id}` is now active."
                    return
                except Exception:
                    continue

        self.status.alert_type = "warning"
        self.status.object = (
            f"Prediction dataset `{dataset_id}` was created, but this DatasetManager "
            "does not expose a known set-active method."
        )

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

            if binding.get("image_size"):
                lines.append(f"**Image size:** `{binding.get('image_size')}`")

            if binding.get("normalization"):
                lines.append(f"**Normalisation:** `{binding.get('normalization')}`")

        target = binding.get("target_column_for_evaluation")

        if target:
            lines.append(f"**Evaluation label column:** `{target}`")
        else:
            lines.append("**Evaluation label column:** none selected; predictions can still run.")

        failed_count = binding.get("failed_image_row_count")
        if failed_count:
            lines.append(f"**Skipped image rows in last run:** `{failed_count}`")

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