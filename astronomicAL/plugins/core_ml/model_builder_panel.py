from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import panel as pn

import json


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
_templates = _load_sibling("model_templates")


class MLModelBuilderPanel:
    """Create reusable model definitions with explicit hyperparameters."""

    def __init__(
        self,
        *,
        context: Any,
        registry: Any,
        restore_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.context = context
        self.registry = registry
        self.templates = _templates.model_templates()
        self.param_widgets: Dict[str, Any] = {}

        self.task = pn.widgets.Select(
            name="",
            options=["classification", "regression", "segmentation"],
            value="classification",
        )
        self.modality = pn.widgets.Select(
            name="",
            options=["tabular", "image"],
            value="tabular",
        )
        self.framework = pn.widgets.Select(
            name="",
            options=["sklearn", "torch"],
            value="sklearn",
        )
        self.template = pn.widgets.Select(name="", options={})
        self.model_name = pn.widgets.TextInput(name="", placeholder="e.g. RF high-depth v1")
        self.description = pn.widgets.TextAreaInput(name="", placeholder="Optional notes", height=80)

        self.tuning_enabled = pn.widgets.Checkbox(
            name="Enable Optuna tuning for this model definition",
            value=False,
        )
        self.tuning_n_trials = pn.widgets.IntInput(name="", value=30, start=1)
        self.tuning_timeout_seconds = pn.widgets.IntInput(
            name="",
            value=0,
            start=0,
        )
        self.tuning_sampler = pn.widgets.Select(
            name="",
            options=["tpe", "random"],
            value="tpe",
        )
        self.tuning_metric = pn.widgets.Select(name="", options=[])
        self.tuning_note = pn.pane.Alert(
            "Optuna tuning is executed for sklearn tabular models and torch image models. "
            "Torch tabular tuning can be wired next.",
            alert_type="info",
        )

        self.tune_widgets: Dict[str, Dict[str, Any]] = {}

        self.tuning_trial_epochs = pn.widgets.IntInput(
            name="",
            value=3,
            start=1,
        )

        self.params_area = pn.Column(
            sizing_mode="stretch_width",
            height=430,
            styles={
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "box-sizing": "border-box",
                "padding-right": "6px",
            },
        )
        self.tuning_area = pn.Column(
            sizing_mode="stretch_width",
            height=430,
            styles={
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "box-sizing": "border-box",
                "padding-right": "6px",
            },
        )
        self.save_button = pn.widgets.Button(name="Save model definition", button_type="success")
        self.refresh_button = pn.widgets.Button(name="Refresh saved definitions", button_type="light")
        self.status = pn.pane.Alert("Choose a template and save a model definition.", alert_type="info")

        self.saved = pn.pane.DataFrame(
            pd.DataFrame(),
            height=260,
            sizing_mode="stretch_width",
        )

        for widget in [
            self.task,
            self.modality,
            self.framework,
            self.template,
            self.model_name,
            self.tuning_n_trials,
            self.tuning_timeout_seconds,
            self.tuning_sampler,
            self.tuning_metric,
            self.tuning_trial_epochs
        ]:
            widget.sizing_mode = "stretch_width"
            widget.height = 38
            widget.margin = (0, 0, 0, 0)

        self.tuning_enabled.sizing_mode = "stretch_width"
        self.tuning_enabled.height = 30
        self.tuning_enabled.margin = (0, 0, 8, 0)

        self.description.sizing_mode = "stretch_width"
        self.save_button.sizing_mode = "stretch_width"
        self.refresh_button.sizing_mode = "stretch_width"

        self.task.param.watch(lambda *_: self._refresh_templates(), "value")
        self.modality.param.watch(lambda *_: self._refresh_templates(), "value")
        self.framework.param.watch(lambda *_: self._refresh_templates(), "value")
        self.template.param.watch(lambda *_: self._render_params(), "value")

        self.tuning_enabled.param.watch(lambda *_: self._render_params(), "value")
        self.task.param.watch(lambda *_: self._update_tuning_metric_options(), "value")

        self.save_button.on_click(self._save_definition)
        self.refresh_button.on_click(lambda *_: self._load_saved_table())

        self._refresh_templates()
        self._update_tuning_metric_options()
        self._load_saved_table()

        if restore_state:
            self.restore_state(restore_state)

    def _update_tuning_metric_options(self) -> None:
        if self.task.value == "classification":
            options = [
                "val_f1_macro",
                "val_accuracy",
                "val_balanced_accuracy",
                "val_roc_auc",
                "val_loss",
            ]
            default = "val_f1_macro"
        elif self.task.value == "regression":
            options = [
                "val_r2",
                "val_mae",
                "val_rmse",
                "val_loss",
            ]
            default = "val_r2"
        else:
            options = ["val_loss"]
            default = "val_loss"

        current = self.tuning_metric.value
        self.tuning_metric.options = options
        self.tuning_metric.value = current if current in options else default


    def _tuning_widget_for_param(self, name: str, schema: Dict[str, Any]):
        kind = schema.get("type")
        default = schema.get("default")

        enabled = pn.widgets.Checkbox(
            name=f"Tune `{name}` with Optuna",
            value=False,
            sizing_mode="stretch_width",
            height=30,
        )

        widgets: Dict[str, Any] = {"enabled": enabled, "schema": schema}

        if kind in {"int", "int_or_none"}:
            low_default = schema.get("min", 1)
            high_default = max(int(default or low_default), int(low_default)) * 3

            low = pn.widgets.IntInput(name="", value=int(low_default))
            high = pn.widgets.IntInput(name="", value=int(high_default))
            step = pn.widgets.IntInput(name="", value=1, start=1)
            log = pn.widgets.Checkbox(name="Log scale", value=False)

            for widget in [low, high, step]:
                widget.sizing_mode = "stretch_width"
                widget.height = 46
                widget.margin = (0, 0, 0, 0)

            log.sizing_mode = "stretch_width"
            log.height = 30
            log.margin = (0, 0, 8, 0)

            widgets.update({"low": low, "high": high, "step": step, "log": log})
            self.tune_widgets[name] = widgets

            return self._tuning_card(
                name,
                enabled,
                [
                    self._field("Low", low),
                    self._field("High", high),
                    self._field("Step", step),
                    log,
                ],
            )

        if kind == "float":
            low_default = float(schema.get("min", 1e-6))
            current = float(default if default is not None else low_default)
            high_default = float(schema.get("max", max(current * 10.0, low_default * 10.0)))

            low = pn.widgets.FloatInput(name="", value=low_default)
            high = pn.widgets.FloatInput(name="", value=high_default)
            log = pn.widgets.Checkbox(name="Log scale", value=current > 0 and low_default > 0)

            for widget in [low, high]:
                widget.sizing_mode = "stretch_width"
                widget.height = 46
                widget.margin = (0, 0, 0, 0)

            log.sizing_mode = "stretch_width"
            log.height = 34
            log.margin = (0, 0, 14, 0)

            widgets.update({"low": low, "high": high, "log": log})
            self.tune_widgets[name] = widgets

            return self._tuning_card(
                name,
                enabled,
                [
                    self._field("Low", low),
                    self._field("High", high),
                    log,
                ],
            )

        if kind == "select":
            raw_options = list(schema.get("options", []))
            label_to_value = {
                "None" if option is None else str(option): option
                for option in raw_options
            }

            choices = pn.widgets.MultiSelect(
                name="",
                options=list(label_to_value.keys()),
                value=list(label_to_value.keys()),
                size=min(8, max(2, len(label_to_value))),
                sizing_mode="stretch_width",
                height=150,
                margin=(0, 0, 0, 0),
            )

            widgets.update({"choices": choices, "label_to_value": label_to_value})
            self.tune_widgets[name] = widgets

            return self._tuning_card(
                name,
                enabled,
                [
                    self._field("Candidate values", choices),
                ],
            )

        if kind == "bool":
            self.tune_widgets[name] = widgets

            return self._tuning_card(
                name,
                enabled,
                [
                    pn.pane.HTML(
                        "<div style='font-size:12px;opacity:0.75;margin:0 0 8px 0;'>"
                        "When enabled, Optuna will try both True and False."
                        "</div>"
                    ),
                ],
            )

        return None

    def _tuning_card(self, name: str, enabled_widget: Any, controls: List[Any]):
        enabled_widget.sizing_mode = "stretch_width"
        enabled_widget.height = 34
        enabled_widget.margin = (0, 0, 10, 0)

        return pn.Column(
            pn.pane.HTML(
                f"<div style='font-size:13px;font-weight:700;margin:0 0 8px 0;'>{name}</div>",
                height=26,
                sizing_mode="stretch_width",
            ),
            enabled_widget,
            *controls,
            pn.Spacer(height=10),
            sizing_mode="stretch_width",
            margin=(0, 0, 22, 0),
            styles={
                "box-sizing": "border-box",
                "padding": "10px",
                "border": "1px solid #ddd",
                "border-radius": "6px",
                "background": "rgba(0,0,0,0.025)",
                "overflow": "visible",
            },
        )

    def _collect_tuning(self, template: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(self.tuning_enabled.value):
            return {
                "enabled": False,
                "backend": "optuna",
                "search_space": {},
            }

        search_space: Dict[str, Any] = {}

        for name, widgets in self.tune_widgets.items():
            enabled = widgets.get("enabled")
            if not enabled or not bool(enabled.value):
                continue

            schema = widgets.get("schema", {})
            kind = schema.get("type")

            if kind in {"int", "int_or_none"}:
                low = int(widgets["low"].value)
                high = int(widgets["high"].value)
                step = int(widgets["step"].value or 1)

                if high < low:
                    low, high = high, low

                search_space[name] = {
                    "type": "int",
                    "low": low,
                    "high": high,
                    "step": max(1, step),
                    "log": bool(widgets["log"].value),
                }

            elif kind == "float":
                low = float(widgets["low"].value)
                high = float(widgets["high"].value)

                if high < low:
                    low, high = high, low

                search_space[name] = {
                    "type": "float",
                    "low": low,
                    "high": high,
                    "log": bool(widgets["log"].value),
                }

            elif kind == "select":
                selected_labels = list(widgets["choices"].value or [])
                label_to_value = widgets.get("label_to_value", {})
                choices = [label_to_value[label] for label in selected_labels if label in label_to_value]

                if choices:
                    search_space[name] = {
                        "type": "categorical",
                        "choices": choices,
                    }

            elif kind == "bool":
                search_space[name] = {
                    "type": "categorical",
                    "choices": [False, True],
                }

        timeout_value = int(self.tuning_timeout_seconds.value or 0)

        return {
            "enabled": bool(search_space),
            "backend": "optuna",
            "n_trials": int(self.tuning_n_trials.value or 1),
            "trial_epochs": int(self.tuning_trial_epochs.value or 3),
            "timeout_seconds": timeout_value if timeout_value > 0 else None,
            "sampler": str(self.tuning_sampler.value or "tpe"),
            "metric": str(self.tuning_metric.value or "val_loss"),
            "search_space": search_space,
        }

    def panel(self):
        builder = pn.Column(
            pn.pane.HTML("<h3 style='margin:0 0 8px 0;'>ML Model Builder</h3>"),
            self._field("Task", self.task),
            self._field("Modality", self.modality),
            self._field("Framework", self.framework),
            self._field("Template", self.template),
            self._field("Model definition name", self.model_name),
            self._field("Description", self.description),
            pn.Tabs(
                (
                    "Fixed hyperparameters",
                    pn.Column(
                        pn.pane.HTML("<b>Hyperparameters</b>", height=26),
                        self.params_area,
                        sizing_mode="stretch_width",
                        height=500,
                        styles={
                            "overflow": "hidden",
                            "box-sizing": "border-box",
                        },
                    ),
                ),
                (
                    "Optuna tuning",
                    pn.Column(
                        self.tuning_enabled,
                        self.tuning_note,
                        self._field("Optimisation metric", self.tuning_metric),
                        self._field("Number of trials", self.tuning_n_trials),
                        self._field("Trial epochs for torch/image tuning", self.tuning_trial_epochs),
                        self._field("Timeout seconds, 0 = no timeout", self.tuning_timeout_seconds),
                        self._field("Sampler", self.tuning_sampler),
                        pn.pane.HTML(
                            "<div style='font-size:12px;opacity:0.75;margin:4px 0 10px 0;'>"
                            "Enable tuning for individual hyperparameters below. "
                            "Fixed values are still used for parameters that are not tuned."
                            "</div>",
                            height=42,
                        ),
                        self.tuning_area,
                        sizing_mode="stretch_width",
                        height=500,
                        styles={
                            "overflow": "hidden",
                            "box-sizing": "border-box",
                        },
                    ),
                ),
                dynamic=True,
                sizing_mode="stretch_width",
                height=540,
            ),
            self.save_button,
            self.status,
            sizing_mode="stretch_width",
            styles=self._styles(),
        )

        saved = pn.Column(
            pn.pane.HTML("<h3 style='margin:0 0 8px 0;'>Saved model definitions</h3>"),
            self.refresh_button,
            self.saved,
            sizing_mode="stretch_width",
            styles=self._styles(),
        )

        return pn.Tabs(
            ("Builder", builder),
            ("Saved", saved),
            dynamic=True,
            sizing_mode="stretch_both",
            styles={
                "overflow": "hidden",
                "box-sizing": "border-box",
            },
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "task": self.task.value,
            "modality": self.modality.value,
            "framework": self.framework.value,
            "template": self.template.value,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if state.get("task") in self.task.options:
            self.task.value = state["task"]
        if state.get("modality") in self.modality.options:
            self.modality.value = state["modality"]
        if state.get("framework") in self.framework.options:
            self.framework.value = state["framework"]

        self._refresh_templates()

        if state.get("template") in self.template.options.values():
            self.template.value = state["template"]

    def _styles(self) -> Dict[str, str]:
        return {
            "box-sizing": "border-box",
            "padding": "10px 14px 14px 14px",
            "overflow-y": "auto",
            "overflow-x": "hidden",
        }

    def _field(self, label: str, widget):
        try:
            widget_height = int(getattr(widget, "height", None) or 44)
        except Exception:
            widget_height = 44

        return pn.Column(
            pn.pane.HTML(
                f"<div style='font-size:12px;font-weight:600;margin:0 0 6px 0;'>{label}</div>",
                height=24,
                sizing_mode="stretch_width",
            ),
            widget,
            sizing_mode="stretch_width",
            min_height=max(78, widget_height + 36),
            margin=(0, 0, 14, 0),
            styles={
                "box-sizing": "border-box",
                "overflow": "visible",
            },
        )

    def _refresh_templates(self) -> None:
        matches = [
            template
            for template in self.templates
            if template["task"] == self.task.value
            and template["modality"] == self.modality.value
            and template["framework"] == self.framework.value
        ]

        self.template.options = {
            template["title"]: template["id"]
            for template in matches
        }

        if matches:
            self.template.value = matches[0]["id"]
            if not self.model_name.value:
                self.model_name.value = matches[0]["title"]
        else:
            self.template.value = None

        self._render_params()

    def _current_template(self) -> Optional[Dict[str, Any]]:
        template_id = self.template.value

        for template in self.templates:
            if template["id"] == template_id:
                return template

        return None

    def _render_params(self) -> None:
        self.param_widgets = {}
        self.tune_widgets = {}
        self.params_area.objects = []
        self.tuning_area.objects = []

        template = self._current_template()
        if not template:
            warning = pn.pane.Alert("No compatible templates found.", alert_type="warning")
            self.params_area.append(warning)
            self.tuning_area.append(
                pn.pane.Alert("No compatible templates found.", alert_type="warning")
            )
            return

        for name, schema in template.get("params", {}).items():
            fixed_widget = self._widget_for_param(name, schema)
            self.param_widgets[name] = fixed_widget

            self.params_area.append(
                pn.Column(
                    self._field(name, fixed_widget),
                    sizing_mode="stretch_width",
                    margin=(0, 0, 10, 0),
                )
            )

            tune_block = self._tuning_widget_for_param(name, schema)
            if tune_block is not None:
                self.tuning_area.append(tune_block)

    def _widget_for_param(self, name: str, schema: Dict[str, Any]):
        kind = schema.get("type")
        default = schema.get("default")

        if kind == "int":
            return pn.widgets.IntInput(
                name="",
                value=int(default),
                start=schema.get("min"),
                end=schema.get("max"),
                sizing_mode="stretch_width",
                height=38,
            )

        if kind == "int_or_none":
            value = "" if default is None else str(default)
            return pn.widgets.TextInput(
                name="",
                value=value,
                placeholder="blank = None",
                sizing_mode="stretch_width",
                height=38,
            )

        if kind == "float":
            return pn.widgets.FloatInput(
                name="",
                value=float(default),
                sizing_mode="stretch_width",
                height=38,
            )

        if kind == "bool":
            return pn.widgets.Checkbox(
                name="",
                value=bool(default),
                sizing_mode="stretch_width",
                height=30,
            )

        if kind == "select":
            options = schema.get("options", [])
            labels = {
                "None" if option is None else str(option): option
                for option in options
            }
            return pn.widgets.Select(
                name="",
                options=labels,
                value=default,
                sizing_mode="stretch_width",
                height=38,
            )

        return pn.widgets.TextInput(
            name="",
            value="" if default is None else str(default),
            sizing_mode="stretch_width",
            height=38,
        )

    def _collect_params(self) -> Dict[str, Any]:
        template = self._current_template()
        if not template:
            return {}

        params = {}

        for name, widget in self.param_widgets.items():
            schema = template.get("params", {}).get(name, {})
            value = widget.value

            if schema.get("type") == "int_or_none":
                if value is None or str(value).strip() == "":
                    params[name] = None
                else:
                    params[name] = int(value)
            elif schema.get("type") == "text":
                params[name] = str(value)
            else:
                params[name] = value

        return params

    def _save_definition(self, *_: Any) -> None:
        template = self._current_template()

        if not template:
            self.status.alert_type = "danger"
            self.status.object = "No model template selected."
            return

        title = self.model_name.value.strip() or template["title"]
        params = self._collect_params()

        definition = {
            "id": f"user.{uuid.uuid4().hex}",
            "title": title,
            "description": self.description.value or "",
            "framework": template["framework"],
            "task": template["task"],
            "modality": template["modality"],
            "template_id": template["id"],
            "template": template,
            "params": params,
            "tuning": self._collect_tuning(template),
        }

        try:
            _ml.register_model_definition(self.registry, definition)
        except Exception as exc:
            self.status.alert_type = "danger"
            self.status.object = f"Could not register model definition: `{exc}`"
            return
        
        try:
            events = getattr(self.context, "events", None)
            publish = getattr(events, "publish", None)
            if callable(publish):
                publish(
                    "ml.model_definition.created",
                    {
                        "model_definition_id": definition["id"],
                        "title": definition["title"],
                        "task": definition["task"],
                        "framework": definition["framework"],
                        "modality": definition["modality"],
                    },
                )
        except Exception:
            pass

        artifact_id = None
        try:
            artifact_id = self.context.artifacts.put(
                "ml.model_definition",
                definition,
                dataset_id=None,
                params={"model_definition_id": definition["id"]},
            )
        except Exception:
            artifact_id = None

        self.status.alert_type = "success"
        if artifact_id:
            self.status.object = (
                f"Saved `{title}` and registered it for the workbench. "
                f"Artifact `{artifact_id}`."
            )
        else:
            self.status.object = f"Registered `{title}` for this session."

        self._load_saved_table()

    def _load_saved_table(self) -> None:
        rows = []

        try:
            _ml.sync_model_definitions_from_artifacts(self.context, self.registry)
        except Exception:
            pass

        artifacts = getattr(self.context, "artifacts", None)
        find = getattr(artifacts, "find", None)
        get = getattr(artifacts, "get", None)

        if callable(find) and callable(get):
            try:
                refs = find(type="ml.model_definition")
            except Exception:
                refs = []

            for ref in refs:
                try:
                    payload = get(ref.artifact_id)
                except Exception:
                    continue

                rows.append(
                    {
                        "title": payload.get("title"),
                        "task": payload.get("task"),
                        "modality": payload.get("modality"),
                        "framework": payload.get("framework"),
                        "template": payload.get("template_id"),
                        "artifact_id": ref.artifact_id,
                        "tuning": bool((payload.get("tuning") or {}).get("enabled")),
                    }
                )

        self.saved.object = pd.DataFrame(rows)