from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import panel as pn


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
        self.tune_widgets: Dict[str, Dict[str, Any]] = {}

        self._syncing_model_name = False
        self._model_name_manually_edited = False
        self._last_template_title: Optional[str] = None

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
        self.model_name = pn.widgets.TextInput(
            name="",
            placeholder="Model definition name",
        )
        self.description = pn.widgets.TextAreaInput(
            name="",
            placeholder="Optional notes",
            height=80,
        )

        self.tuning_enabled = pn.widgets.Checkbox(
            name="Enable Optuna tuning for this model definition",
            value=False,
        )
        self.tuning_n_trials = pn.widgets.IntInput(name="", value=30, start=1)
        self.tuning_timeout_seconds = pn.widgets.IntInput(name="", value=0, start=0)
        self.tuning_sampler = pn.widgets.Select(
            name="",
            options=["tpe", "random"],
            value="tpe",
        )
        self.tuning_metric = pn.widgets.Select(name="", options=[])
        self.tuning_trial_epochs = pn.widgets.IntInput(name="", value=3, start=1)

        self.tuning_note = pn.pane.Alert(
            "Optuna tuning is saved into the model definition. Select individual "
            "hyperparameters below to build the search space. Fixed values are used "
            "for parameters that are not tuned.",
            alert_type="info",
        )

        self.tune_common_button = pn.widgets.Button(
            name="Tune common parameters",
            button_type="primary",
        )
        self.clear_tuning_button = pn.widgets.Button(
            name="Clear tuned parameters",
            button_type="light",
        )

        self.params_area = pn.Column(
            sizing_mode="stretch_width",
            styles={
                "overflow": "visible",
                "box-sizing": "border-box",
            },
        )
        self.tuning_area = pn.Column(
            sizing_mode="stretch_width",
            styles={
                "overflow": "visible",
                "box-sizing": "border-box",
            },
        )

        self.save_button = pn.widgets.Button(
            name="Save model definition",
            button_type="success",
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh saved definitions",
            button_type="light",
        )
        self.status = pn.pane.Alert(
            "Choose a template and save a model definition.",
            alert_type="info",
        )
        self.saved = pn.pane.DataFrame(
            pd.DataFrame(),
            height=260,
            sizing_mode="stretch_width",
        )

        self._apply_widget_sizing()
        self._wire_events()

        self._refresh_templates()
        self._update_tuning_metric_options()
        self._load_saved_table()

        if restore_state:
            self.restore_state(restore_state)

    def _apply_widget_sizing(self) -> None:
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
            self.tuning_trial_epochs,
        ]:
            widget.sizing_mode = "stretch_width"
            widget.height = 38
            widget.margin = (0, 0, 0, 0)

        self.tuning_enabled.sizing_mode = "stretch_width"
        self.tuning_enabled.height = 30
        self.tuning_enabled.margin = (0, 0, 8, 0)

        self.description.sizing_mode = "stretch_width"
        self.description.margin = (0, 0, 0, 0)

        for button in [
            self.tune_common_button,
            self.clear_tuning_button,
            self.save_button,
            self.refresh_button,
        ]:
            button.sizing_mode = "stretch_width"
            button.height = 34
            button.margin = (0, 0, 8, 0)

        self.saved.sizing_mode = "stretch_width"

    def _wire_events(self) -> None:
        self.task.param.watch(lambda *_: self._on_template_filter_change(), "value")
        self.modality.param.watch(lambda *_: self._on_template_filter_change(), "value")
        self.framework.param.watch(lambda *_: self._on_template_filter_change(), "value")

        self.template.param.watch(lambda *_: self._on_template_change(), "value")
        self.model_name.param.watch(self._on_model_name_change, "value")

        # Important: do NOT re-render all params when tuning_enabled changes.
        # Re-rendering recreated the fixed hyperparameter widgets and reset
        # values such as pretrained=True after the user had changed them.
        self.task.param.watch(lambda *_: self._update_tuning_metric_options(), "value")

        self.tune_common_button.on_click(lambda *_: self._select_common_tuning_params())
        self.clear_tuning_button.on_click(lambda *_: self._clear_tuning_params())

        self.save_button.on_click(self._save_definition)
        self.refresh_button.on_click(lambda *_: self._load_saved_table())

    def panel(self):
        fixed_section = pn.Column(
            pn.pane.HTML(
                "<h4 style='margin:12px 0 4px 0'>Fixed hyperparameters</h4>"
                "<div style='font-size:12px;color:#666;margin-bottom:8px'>"
                "These values are used directly unless the same parameter is selected "
                "for Optuna tuning below."
                "</div>",
                height=64,
                sizing_mode="stretch_width",
            ),
            self.params_area,
            sizing_mode="stretch_width",
            styles={
                "box-sizing": "border-box",
                "overflow": "visible",
            },
        )

        optuna_section = pn.Column(
            pn.pane.HTML(
                "<h4 style='margin:18px 0 4px 0'>Optuna tuning</h4>"
                "<div style='font-size:12px;color:#666;margin-bottom:8px'>"
                "Enable tuning, choose the optimisation settings, then tick the "
                "hyperparameters Optuna should search over."
                "</div>",
                height=72,
                sizing_mode="stretch_width",
            ),
            self.tuning_enabled,
            self.tuning_note,
            self._field("Optimisation metric", self.tuning_metric),
            self._field("Number of trials", self.tuning_n_trials),
            self._field("Trial epochs for torch/image tuning", self.tuning_trial_epochs),
            self._field("Timeout seconds, 0 = no timeout", self.tuning_timeout_seconds),
            self._field("Sampler", self.tuning_sampler),
            pn.Row(
                self.tune_common_button,
                self.clear_tuning_button,
                sizing_mode="stretch_width",
            ),
            pn.pane.HTML(
                "<h4 style='margin:14px 0 4px 0'>Tunable hyperparameters</h4>"
                "<div style='font-size:12px;color:#666;margin-bottom:8px'>"
                "Unticked parameters stay fixed at the values above."
                "</div>",
                height=60,
                sizing_mode="stretch_width",
            ),
            self.tuning_area,
            sizing_mode="stretch_width",
            styles={
                "box-sizing": "border-box",
                "overflow": "visible",
            },
        )

        builder = pn.Column(
            pn.pane.HTML(
                "<h3 style='margin:0 0 8px 0'>ML Model Builder</h3>",
                height=34,
                sizing_mode="stretch_width",
            ),
            self._field("Task", self.task),
            self._field("Modality", self.modality),
            self._field("Framework", self.framework),
            self._field("Template", self.template),
            self._field("Model definition name", self.model_name),
            self._field("Description", self.description),
            fixed_section,
            optuna_section,
            pn.Spacer(height=8),
            self.save_button,
            self.status,
            sizing_mode="stretch_both",
            scroll=True,
            styles={
                "box-sizing": "border-box",
                "padding": "10px 14px 18px 14px",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "height": "100%",
                "min-height": "0",
            },
        )

        saved = pn.Column(
            pn.pane.HTML(
                "<h3 style='margin:0 0 8px 0'>Saved model definitions</h3>",
                height=34,
                sizing_mode="stretch_width",
            ),
            self.refresh_button,
            self.saved,
            sizing_mode="stretch_both",
            scroll=True,
            styles={
                "box-sizing": "border-box",
                "padding": "10px 14px 18px 14px",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "height": "100%",
                "min-height": "0",
            },
        )

        return pn.Tabs(
            ("Builder", builder),
            ("Saved", saved),
            dynamic=True,
            sizing_mode="stretch_both",
            styles={
                "overflow": "hidden",
                "box-sizing": "border-box",
                "height": "100%",
                "min-height": "0",
            },
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "task": self.task.value,
            "modality": self.modality.value,
            "framework": self.framework.value,
            "template": self.template.value,
            "model_name": self.model_name.value,
            "description": self.description.value,
            "model_name_manually_edited": self._model_name_manually_edited,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        if state.get("task") in self.task.options:
            self.task.value = state["task"]

        if state.get("modality") in self.modality.options:
            self.modality.value = state["modality"]

        if state.get("framework") in self.framework.options:
            self.framework.value = state["framework"]

        self._refresh_templates()

        if state.get("template") in self.template.options.values():
            self.template.value = state["template"]
            self._on_template_change()

        if state.get("model_name"):
            self._syncing_model_name = True
            try:
                self.model_name.value = str(state["model_name"])
            finally:
                self._syncing_model_name = False

        if state.get("description"):
            self.description.value = str(state["description"])

        self._model_name_manually_edited = bool(
            state.get("model_name_manually_edited", self._model_name_manually_edited)
        )

    def _styles(self) -> Dict[str, str]:
        return {
            "box-sizing": "border-box",
            "padding": "10px 14px 18px 14px",
            "overflow": "visible",
        }

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
            margin=(0, 0, 10, 0),
            styles={
                "box-sizing": "border-box",
                "overflow": "visible",
            },
        )

    def _on_template_filter_change(self) -> None:
        self._update_tuning_metric_options()
        self._refresh_templates()

    def _refresh_templates(self) -> None:
        previous_template = self.template.value

        matches = [
            template
            for template in self.templates
            if template["task"] == self.task.value
            and template["modality"] == self.modality.value
            and template["framework"] == self.framework.value
        ]

        self.template.options = {template["title"]: template["id"] for template in matches}

        if matches:
            match_ids = {template["id"] for template in matches}
            if previous_template in match_ids:
                self.template.value = previous_template
            else:
                self.template.value = matches[0]["id"]
        else:
            self.template.value = None

        self._on_template_change()

    def _on_template_change(self) -> None:
        template = self._current_template()

        if template:
            title = str(template.get("title") or template.get("id") or "Model")
            current_name = (self.model_name.value or "").strip()
            previous_title = (self._last_template_title or "").strip()

            should_sync_name = (
                not self._model_name_manually_edited
                or not current_name
                or current_name == previous_title
            )

            if should_sync_name:
                self._syncing_model_name = True
                try:
                    self.model_name.value = title
                finally:
                    self._syncing_model_name = False
                self._model_name_manually_edited = False

            self._last_template_title = title
        else:
            self._last_template_title = None

        self._render_params()

    def _on_model_name_change(self, event: Any) -> None:
        if self._syncing_model_name:
            return

        value = str(getattr(event, "new", self.model_name.value) or "").strip()
        template = self._current_template()
        template_title = str(template.get("title", "")).strip() if template else ""

        self._model_name_manually_edited = bool(value and value != template_title)

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
            self.tuning_area.append(warning.clone())
            return

        tunable_count = 0

        for name, schema in template.get("params", {}).items():
            fixed_widget = self._widget_for_param(name, schema)
            self.param_widgets[name] = fixed_widget

            self.params_area.append(
                pn.Column(
                    self._field(name, fixed_widget),
                    sizing_mode="stretch_width",
                    margin=(0, 0, 8, 0),
                )
            )

            tune_block = self._tuning_widget_for_param(name, schema)
            if tune_block is not None:
                tunable_count += 1
                self.tuning_area.append(tune_block)

        if tunable_count == 0:
            self.tuning_area.append(
                pn.pane.Alert(
                    "This template does not currently expose any Optuna-tunable "
                    "parameters. Parameters with free-text schemas are deliberately "
                    "not tuned yet because they need custom parsers.",
                    alert_type="warning",
                )
            )

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
            labels = {"None" if option is None else str(option): option for option in options}
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

    def _tuning_widget_for_param(self, name: str, schema: Dict[str, Any]):
        # Templates can explicitly remove parameters from the Optuna UI.
        # This prevents setup/runtime parameters such as image_size, pretrained,
        # epochs, patience, etc. from appearing just because they are numeric.
        if schema.get("tunable") is False:
            return None

        kind = schema.get("type")
        default = schema.get("default")
        tune_space = dict(schema.get("tune_space") or {})

        enabled = pn.widgets.Checkbox(
            name=f"Tune `{name}` with Optuna",
            value=bool(schema.get("tune_default", False)),
            sizing_mode="stretch_width",
            height=30,
        )

        widgets: Dict[str, Any] = {
            "enabled": enabled,
            "schema": schema,
        }

        if kind in {"int", "int_or_none"}:
            low_default = int(
                tune_space.get(
                    "low",
                    schema.get("min", 1) or 1,
                )
            )

            if "high" in tune_space:
                high_default = int(tune_space["high"])
            elif default is None:
                high_default = max(low_default * 3, low_default + 4)
            else:
                high_default = max(int(default), low_default) * 3

            step_default = int(tune_space.get("step", 1) or 1)
            log_default = bool(tune_space.get("log", False))

            low = pn.widgets.IntInput(name="", value=int(low_default))
            high = pn.widgets.IntInput(name="", value=int(high_default))
            step = pn.widgets.IntInput(name="", value=step_default, start=1)
            log = pn.widgets.Checkbox(name="Log scale", value=log_default)

            for widget in [low, high, step]:
                widget.sizing_mode = "stretch_width"
                widget.height = 38
                widget.margin = (0, 0, 0, 0)

            log.sizing_mode = "stretch_width"
            log.height = 30
            log.margin = (0, 0, 8, 0)

            widgets.update(
                {
                    "low": low,
                    "high": high,
                    "step": step,
                    "log": log,
                }
            )
            self.tune_widgets[name] = widgets

            return self._tuning_card(
                name,
                enabled,
                [
                    pn.Row(
                        self._field("Low", low),
                        self._field("High", high),
                        self._field("Step", step),
                        sizing_mode="stretch_width",
                    ),
                    log,
                ],
            )

        if kind == "float":
            low_default = float(tune_space.get("low", schema.get("min", 1e-6)))
            current = float(default if default is not None else low_default)
            high_default = float(
                tune_space.get(
                    "high",
                    schema.get("max", max(current * 10.0, low_default * 10.0)),
                )
            )

            if high_default <= low_default:
                high_default = low_default + 1.0

            log_default = bool(
                tune_space.get(
                    "log",
                    current > 0 and low_default > 0,
                )
            )

            low = pn.widgets.FloatInput(name="", value=low_default)
            high = pn.widgets.FloatInput(name="", value=high_default)
            log = pn.widgets.Checkbox(name="Log scale", value=log_default)

            for widget in [low, high]:
                widget.sizing_mode = "stretch_width"
                widget.height = 38
                widget.margin = (0, 0, 0, 0)

            log.sizing_mode = "stretch_width"
            log.height = 30
            log.margin = (0, 0, 8, 0)

            widgets.update(
                {
                    "low": low,
                    "high": high,
                    "log": log,
                }
            )
            self.tune_widgets[name] = widgets

            return self._tuning_card(
                name,
                enabled,
                [
                    pn.Row(
                        self._field("Low", low),
                        self._field("High", high),
                        sizing_mode="stretch_width",
                    ),
                    log,
                ],
            )

        if kind == "select":
            raw_options = list(tune_space.get("choices", schema.get("options", [])))
            label_to_value = {"None" if option is None else str(option): option for option in raw_options}

            default_labels = tune_space.get("default_choices")
            if default_labels is None:
                default_labels = list(label_to_value.keys())

            choices = pn.widgets.MultiSelect(
                name="",
                options=list(label_to_value.keys()),
                value=[label for label in default_labels if label in label_to_value],
                size=min(8, max(2, len(label_to_value))),
                sizing_mode="stretch_width",
                height=min(180, max(90, 28 * max(2, len(label_to_value)))),
                margin=(0, 0, 0, 0),
            )

            widgets.update(
                {
                    "choices": choices,
                    "label_to_value": label_to_value,
                }
            )
            self.tune_widgets[name] = widgets

            return self._tuning_card(
                name,
                enabled,
                [
                    self._field("Candidate values", choices),
                ],
            )

        if kind == "bool":
            choices = list(tune_space.get("choices", [False, True]))
            if len(choices) < 2:
                return None

            self.tune_widgets[name] = widgets

            return self._tuning_card(
                name,
                enabled,
                [
                    pn.pane.HTML(
                        "<div style='font-size:12px;color:#666'>"
                        f"Optuna will try: <code>{choices}</code>."
                        "</div>",
                        sizing_mode="stretch_width",
                    ),
                ],
            )

        return None

    def _tuning_card(self, name: str, enabled_widget: Any, controls: List[Any]):
        enabled_widget.sizing_mode = "stretch_width"
        enabled_widget.height = 34
        enabled_widget.margin = (0, 0, 8, 0)

        return pn.Column(
            pn.pane.HTML(
                f"<div style='font-size:13px;font-weight:700'>{name}</div>",
                height=24,
                sizing_mode="stretch_width",
            ),
            enabled_widget,
            *controls,
            sizing_mode="stretch_width",
            margin=(0, 0, 14, 0),
            styles={
                "box-sizing": "border-box",
                "padding": "10px",
                "border": "1px solid #ddd",
                "border-radius": "6px",
                "background": "rgba(0,0,0,0.025)",
                "overflow": "visible",
            },
        )

    def _select_common_tuning_params(self) -> None:
        selected = 0

        for name, widgets in self.tune_widgets.items():
            schema = widgets.get("schema", {}) or {}
            enabled = widgets.get("enabled")

            if enabled is not None and bool(schema.get("tune_default", False)):
                enabled.value = True
                selected += 1

        if selected:
            self.tuning_enabled.value = True
            self.status.alert_type = "info"
            self.status.object = (
                f"Selected {selected} recommended parameters for Optuna tuning."
            )
        else:
            self.status.alert_type = "warning"
            self.status.object = (
                "This template does not define any recommended Optuna parameters."
            )

    def _clear_tuning_params(self) -> None:
        for widgets in self.tune_widgets.values():
            enabled = widgets.get("enabled")
            if enabled is not None:
                enabled.value = False

        self.status.alert_type = "info"
        self.status.object = "Cleared Optuna parameter selections."

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
                choices = [
                    label_to_value[label]
                    for label in selected_labels
                    if label in label_to_value
                ]

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

    def _save_definition(self, *_: Any) -> None:
        template = self._current_template()
        if not template:
            self.status.alert_type = "danger"
            self.status.object = "No model template selected."
            return

        title = self.model_name.value.strip() or template["title"]
        params = self._collect_params()
        tuning = self._collect_tuning(template)

        if bool(self.tuning_enabled.value) and not tuning.get("search_space"):
            self.status.alert_type = "warning"
            self.status.object = (
                "Optuna tuning is enabled, but no hyperparameters were selected. "
                "Tick at least one parameter card in the Optuna tab, use "
                "`Tune common parameters`, or disable Optuna tuning."
            )
            return

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
            "tuning": tuning,
        }

        try:
            _ml.register_model_definition(self.registry, definition)
        except Exception as exc:
            self.status.alert_type = "danger"
            self.status.object = f"Could not register model definition: `{exc}`"
            return

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

        try:
            events = getattr(self.context, "events", None)
            publish = getattr(events, "publish", None)
            if callable(publish):
                event_payload = {
                    "model_definition_id": definition["id"],
                    "title": definition["title"],
                    "task": definition["task"],
                    "framework": definition["framework"],
                    "modality": definition["modality"],
                    "tuning_enabled": bool(tuning.get("enabled")),
                    "tuning_metric": tuning.get("metric"),
                    "tuning_n_trials": tuning.get("n_trials"),
                    "tuned_parameters": sorted((tuning.get("search_space") or {}).keys()),
                    "artifact_id": artifact_id,
                }

                publish("ml.model_definition.created", event_payload)
                publish("ml.registry.changed", event_payload)
        except Exception:
            pass

        self.status.alert_type = "success"

        if artifact_id:
            if tuning.get("enabled"):
                tuned = ", ".join(sorted((tuning.get("search_space") or {}).keys()))
                self.status.object = (
                    f"Saved `{title}` with Optuna tuning enabled. "
                    f"Metric `{tuning.get('metric')}`, trials `{tuning.get('n_trials')}`, "
                    f"parameters: {tuned}. Artifact `{artifact_id}`."
                )
            else:
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

                tuning = payload.get("tuning") or {}
                search_space = tuning.get("search_space") or {}

                rows.append(
                    {
                        "title": payload.get("title"),
                        "task": payload.get("task"),
                        "modality": payload.get("modality"),
                        "framework": payload.get("framework"),
                        "template": payload.get("template_id"),
                        "tuning": bool(tuning.get("enabled")),
                        "metric": tuning.get("metric") if tuning.get("enabled") else "",
                        "trials": tuning.get("n_trials") if tuning.get("enabled") else "",
                        "tuned_params": ", ".join(sorted(search_space.keys())),
                        "artifact_id": ref.artifact_id,
                    }
                )

        self.saved.object = pd.DataFrame(rows)