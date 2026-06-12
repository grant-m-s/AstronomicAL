from __future__ import annotations

import importlib.util
import json
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import panel as pn

from astronomicAL.platform.plugins.specs import ActionRequest


def _load_sibling(stem: str):
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


_registry_mod = _load_sibling("recipe_registry")
_runner = _load_sibling("recipe_runner")


class MLRecipeLauncherPanel:
    def __init__(
        self,
        *,
        context: Any,
        registry: Any,
        restore_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.context = context
        self.registry = registry
        self.param_widgets: Dict[str, Any] = {}
        self._active_thread: Optional[threading.Thread] = None
        self._cancel_token: Any = None

        self.recipe = pn.widgets.Select(name="", options={}, sizing_mode="stretch_width")
        self.dataset = pn.widgets.Select(name="", options=[], sizing_mode="stretch_width")
        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="light",
            sizing_mode="stretch_width",
        )
        self.run_button = pn.widgets.Button(
            name="Run recipe",
            button_type="success",
            sizing_mode="stretch_width",
        )

        self.cancel_button = pn.widgets.Button(
            name="Cancel run",
            button_type="danger",
            disabled=True,
            sizing_mode="stretch_width",
        )

        self.recipe_card = pn.pane.Markdown(
            "Choose a recipe.",
            sizing_mode="stretch_width",
        )
        self.params_area = pn.Column(sizing_mode="stretch_width")
        self.status = pn.pane.Alert(
            "Choose a recipe and dataset.",
            alert_type="info",
            sizing_mode="stretch_width",
        )
        self.result = pn.pane.JSON(
            {},
            depth=3,
            sizing_mode="stretch_both",
        )

        self.recipe.param.watch(lambda *_: self._on_recipe_change(), "value")
        self.dataset.param.watch(lambda *_: self._apply_inferred_defaults(), "value")
        self.refresh_button.on_click(lambda *_: self.refresh())
        self.run_button.on_click(self._run_clicked)
        self.cancel_button.on_click(self._cancel_clicked)

        self.refresh()

        if restore_state:
            self.restore_state(restore_state)

    def panel(self):
        left = pn.Column(
            pn.pane.Markdown("### ML Recipe Launcher"),
            self._field("Recipe", self.recipe),
            self.recipe_card,
            self._field("Dataset", self.dataset),
            pn.Row(
                self.refresh_button,
                self.run_button,
                self.cancel_button,
                sizing_mode="stretch_width",
            ),
            self.status,
            sizing_mode="stretch_both",
            scroll=True,
            styles={
                "padding": "10px 14px 18px 14px",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "height": "100%",
                "max-height": "100%",
                "min-height": "0",
            },
        )

        params = pn.Column(
            pn.pane.Markdown("### Parameters"),
            self.params_area,
            sizing_mode="stretch_both",
            scroll=True,
            styles={
                "padding": "10px 14px 18px 14px",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "height": "100%",
                "max-height": "100%",
                "min-height": "0",
            },
        )

        output = pn.Column(
            pn.pane.Markdown("### Last result"),
            self.result,
            sizing_mode="stretch_both",
            scroll=True,
            styles={
                "padding": "10px 14px 18px 14px",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "height": "100%",
                "max-height": "100%",
                "min-height": "0",
            },
        )

        launch_tab = pn.Column(
            pn.Row(
                left,
                params,
                sizing_mode="stretch_both",
                styles={
                    "height": "100%",
                    "max-height": "100%",
                    "min-height": "0",
                    "overflow": "hidden",
                },
            ),
            sizing_mode="stretch_both",
            scroll=True,
            styles={
                "height": "100%",
                "max-height": "100%",
                "min-height": "0",
                "overflow-y": "auto",
                "overflow-x": "hidden",
            },
        )

        result_tab = pn.Column(
            output,
            sizing_mode="stretch_both",
            scroll=True,
            styles={
                "height": "100%",
                "max-height": "100%",
                "min-height": "0",
                "overflow-y": "auto",
                "overflow-x": "hidden",
            },
        )

        return pn.Tabs(
            ("Launch", launch_tab),
            ("Result", result_tab),
            dynamic=True,
            sizing_mode="stretch_both",
            styles={
                "height": "100%",
                "max-height": "100%",
                "min-height": "0",
                "overflow": "hidden",
            },
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "recipe": self.recipe.value,
            "dataset": self.dataset.value,
            "params": self._params(),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        if state.get("recipe") in self.recipe.options.values():
            self.recipe.value = state["recipe"]
            self._on_recipe_change()

        if state.get("dataset") in self.dataset.options:
            self.dataset.value = state["dataset"]

        params = state.get("params") or {}
        if isinstance(params, dict):
            for name, value in params.items():
                widget = self.param_widgets.get(name)
                if widget is not None:
                    try:
                        widget.value = value
                    except Exception:
                        pass

    def refresh(self) -> None:
        recipes = self.registry.list()
        self.recipe.options = {recipe.title: recipe.id for recipe in recipes}

        if recipes and not self.recipe.value:
            self.recipe.value = recipes[0].id

        dataset_ids = _registry_mod.list_dataset_ids(self.context)
        self.dataset.options = dataset_ids

        active = _registry_mod.active_dataset_id(self.context)
        if active in dataset_ids:
            self.dataset.value = active
        elif self.dataset.value not in dataset_ids:
            self.dataset.value = dataset_ids[0] if dataset_ids else None

        self._on_recipe_change()
        self._apply_inferred_defaults()

    def _field(self, label: str, widget: Any):
        return pn.Column(
            pn.pane.Markdown(f"**{label}**", height=22, margin=(0, 0, 2, 0)),
            widget,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

    def _on_recipe_change(self) -> None:
        self.param_widgets = {}
        self.params_area.objects = []

        recipe_id = self.recipe.value
        if not recipe_id:
            self.recipe_card.object = "No recipe selected."
            return

        spec = self.registry.get(recipe_id)
        self.recipe_card.object = (
            f"**{spec.title}**  \n"
            f"`{spec.id}` v{spec.version}  \n\n"
            f"{spec.description}\n\n"
            f"- Task: `{spec.task}`\n"
            f"- Modality: `{spec.modality}`\n"
            f"- Complexity: `{spec.complexity}`\n"
            f"- Required mappings: `{', '.join(spec.required_mappings) or 'none'}`\n"
            f"- Produces: `{', '.join(spec.produces) or 'none'}`"
        )

        schema = spec.params_schema or {"type": "object", "properties": {}}
        properties = schema.get("properties", {}) or {}

        for name, param_schema in properties.items():
            widget = self._widget_for_schema(str(name), param_schema)
            self.param_widgets[str(name)] = widget
            self.params_area.append(self._field(str(param_schema.get("title") or name), widget))

        self._apply_inferred_defaults()

    def _empty_widget_value(self, value: Any) -> bool:
        return value is None or value == "" or value == [] or value == {}

    def _apply_inferred_defaults(self) -> None:
        recipe_id = self.recipe.value
        dataset_id = self.dataset.value

        if not recipe_id or not dataset_id:
            return

        try:
            spec = self.registry.get(recipe_id)
            inferred = _registry_mod.infer_recipe_params(self.context, dataset_id, spec)
        except Exception:
            inferred = {}

        if not inferred:
            return

        applied = []
        for name, value in inferred.items():
            widget = self.param_widgets.get(name)
            if widget is None:
                continue

            try:
                current = widget.value
            except Exception:
                continue

            if not self._empty_widget_value(current):
                continue

            try:
                widget.value = value
                applied.append(f"`{name}` = `{value}`")
            except Exception:
                pass

        train_dataset_widget = self.param_widgets.get("train_dataset_id")
        if train_dataset_widget is not None and dataset_id:
            try:
                if self._empty_widget_value(train_dataset_widget.value):
                    train_dataset_widget.value = dataset_id
                    applied.append(f"`train_dataset_id` = `{dataset_id}`")
            except Exception:
                pass

        if applied:
            self.status.alert_type = "info"
            self.status.object = (
                "Inferred recipe inputs from the selected dataset: "
                + ", ".join(applied)
            )

    def _widget_for_schema(self, name: str, schema: Mapping[str, Any]):
        kind = str(schema.get("type", "string"))
        default = schema.get("default", "")

        widget_kind = str(schema.get("x-widget") or schema.get("widget") or "")

        if widget_kind == "dataset_select" or name.endswith("_dataset_id"):
            dataset_ids = _registry_mod.list_dataset_ids(self.context)
            options = [""] + dataset_ids
            value = default if default in options else ""
            return pn.widgets.Select(
                name="",
                options=options,
                value=value,
                sizing_mode="stretch_width",
            )

        if "enum" in schema:
            values = list(schema.get("enum") or [])
            options = {str(v): v for v in values}
            value = default if default in values else (values[0] if values else None)
            return pn.widgets.Select(
                name="",
                options=options,
                value=value,
                sizing_mode="stretch_width",
            )

        if kind in {"integer", "int"}:
            return pn.widgets.IntInput(
                name="",
                value=int(default or 0),
                start=schema.get("minimum"),
                end=schema.get("maximum"),
                sizing_mode="stretch_width",
            )

        if kind in {"number", "float"}:
            return pn.widgets.FloatInput(
                name="",
                value=float(default or 0.0),
                start=schema.get("minimum"),
                end=schema.get("maximum"),
                sizing_mode="stretch_width",
            )

        if kind in {"boolean", "bool"}:
            return pn.widgets.Checkbox(
                name="",
                value=bool(default),
                sizing_mode="stretch_width",
            )

        if kind in {"array", "object"}:
            text = json.dumps(default if default not in ("", None) else ([] if kind == "array" else {}))
            return pn.widgets.TextAreaInput(
                name="",
                value=text,
                height=120,
                sizing_mode="stretch_width",
            )

        return pn.widgets.TextInput(
            name="",
            value="" if default is None else str(default),
            placeholder=str(schema.get("description") or ""),
            sizing_mode="stretch_width",
        )

    def _params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        spec = self.registry.get(self.recipe.value)
        properties = (spec.params_schema or {}).get("properties", {}) or {}

        for name, widget in self.param_widgets.items():
            value = widget.value
            schema = properties.get(name, {})
            kind = str(schema.get("type", "string"))

            if kind in {"array", "object"} and isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    # Keep the raw string; validation/error handling should happen in recipe.
                    pass

            params[name] = value

        params["recipe_id"] = self.recipe.value
        params["dataset_id"] = self.dataset.value
        return params

    def _run_clicked(self, *_: Any) -> None:
        if self._active_thread and self._active_thread.is_alive():
            self.status.alert_type = "warning"
            self.status.object = "A recipe is already running from this panel."
            return

        params = self._params()
        if not params.get("recipe_id"):
            self.status.alert_type = "danger"
            self.status.object = "Choose a recipe."
            return
        if not params.get("dataset_id"):
            self.status.alert_type = "danger"
            self.status.object = "Choose a dataset."
            return

        self._cancel_token = _registry_mod.CancellationToken()
        self._set_running_state(True)

        self.status.alert_type = "info"
        self.status.object = (
            "Recipe running. Use Cancel run to request a clean stop. "
            "Cancellation will happen at the next run.check_cancelled() point."
        )

        request = ActionRequest(
            dataset_id=params["dataset_id"],
            row_ids=None,
            columns=[],
            params=params,
            artifact_id=None,
            origin="core.ml.recipe_launcher",
        )

        def worker() -> None:
            try:
                result = _runner.run_ml_recipe_action(
                    self.context,
                    request,
                    cancel_token=self._cancel_token,
                )
                status = str(result.get("status") or "").lower()
                self._update_result(
                    result,
                    success=status != "cancelled",
                    cancelled=status == "cancelled",
                )
            except Exception as exc:
                self._update_result(
                    {
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    success=False,
                    cancelled=False,
                )
            finally:
                self._set_running_state(False)
                self._cancel_token = None

        self._active_thread = threading.Thread(target=worker, daemon=True)
        self._active_thread.start()

    def _cancel_clicked(self, *_: Any) -> None:
        if self._cancel_token is None:
            self.status.alert_type = "warning"
            self.status.object = "No recipe run is currently active from this panel."
            return

        try:
            self._cancel_token.cancel("Recipe run cancelled from the ML Recipe Launcher.")
        except Exception:
            pass

        self.cancel_button.disabled = True
        self.status.alert_type = "warning"
        self.status.object = (
            "Cancellation requested. The recipe will stop at the next safe cancellation point."
        )


    def _set_running_state(self, running: bool) -> None:
        def apply() -> None:
            self.run_button.disabled = running
            self.refresh_button.disabled = running
            self.cancel_button.disabled = not running

        try:
            pn.state.curdoc.add_next_tick_callback(apply)
        except Exception:
            apply()

    def _update_result(
        self,
        payload: Dict[str, Any],
        *,
        success: bool,
        cancelled: bool = False,
    ) -> None:
        def apply() -> None:
            self.result.object = payload

            if cancelled:
                self.status.alert_type = "warning"
                self.status.object = payload.get("message") or "Recipe run cancelled."
            elif success:
                self.status.alert_type = "success"
                self.status.object = "Recipe finished."
            else:
                self.status.alert_type = "danger"
                self.status.object = f"Recipe failed: {payload.get('error', 'unknown error')}"

        try:
            pn.state.curdoc.add_next_tick_callback(apply)
        except Exception:
            apply()


def create_recipe_launcher_panel(context: Any, **kwargs: Any):
    registry = context.services.get("core.ml.recipe_registry")
    controller = MLRecipeLauncherPanel(
        context=context,
        registry=registry,
        restore_state=kwargs.get("restore_state"),
    )
    return controller.panel(), controller