from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import panel as pn

from astronomicAL.platform.plugins.specs import ActionRequest

from .. import registry as _registry_mod
from ..data import dataset_access as _dataset_access
from .. import recipe_runner as _runner
from ..feature_columns import parse_column_list
from ..job_bridge import submit_job

from ..profiles import PROTOCOL_KEYS as _PROTOCOL_KEYS

_PROTOCOL_LABELS = {
    "protocol_split_strategy": "Split method for selected dataset",
    "protocol_validation_source": "Validation source",
    "protocol_validation_dataset_id": "Validation dataset",
    "protocol_test_source": "Test source",
    "protocol_test_dataset_id": "Test dataset",
    "protocol_group_column": "Group/time column",
    "protocol_split_column": "Predefined split column",
    "protocol_validation_size": "Validation fraction",
    "protocol_test_size": "Test fraction",
    "protocol_selection_metric": "Best-epoch metric",
    "protocol_random_state": "Random seed",
}

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
        self.param_fields: Dict[str, Any] = {}
        # Protocol section (managed recipes only). Empty for freeform recipes.
        self.protocol_widgets: Dict[str, Any] = {}
        self.protocol_fields: Dict[str, Any] = {}
        self._active_handle: Any = None

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

        self.profile = pn.widgets.Select(name="", options={}, sizing_mode="stretch_width")
        self.profile_name = pn.widgets.TextInput(name="", placeholder="Profile name", sizing_mode="stretch_width")
        self.save_profile_button = pn.widgets.Button(name="Save profile", button_type="primary", sizing_mode="stretch_width")
        self.load_profile_button = pn.widgets.Button(name="Load profile", button_type="light", sizing_mode="stretch_width")

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
        self.dataset.param.watch(lambda *_: self._on_dataset_change(), "value")
        self.refresh_button.on_click(lambda *_: self.refresh())
        self.run_button.on_click(self._run_clicked)
        self.cancel_button.on_click(self._cancel_clicked)
        self.save_profile_button.on_click(self._save_profile_clicked)
        self.load_profile_button.on_click(self._load_profile_clicked)

        self.refresh()

        if restore_state:
            self.restore_state(restore_state)

    def panel(self):
        left = pn.Column(
            pn.pane.Markdown("### ML Recipe Launcher"),
            self._field("Recipe", self.recipe),
            self.recipe_card,
            self._field("Saved profile", self.profile),
            pn.Row(self.load_profile_button, self.save_profile_button, sizing_mode="stretch_width"),
            self._field("Profile name", self.profile_name),
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
                widget = (
                    self.param_widgets.get(name)
                    or self.protocol_widgets.get(name)
                )
                if widget is not None:
                    try:
                        widget.value = value
                    except Exception:
                        pass

        self._sync_protocol_visibility()

    def _profile_store(self):
        try:
            return self.context.services.get("core.ml.recipe_profile_store")
        except Exception:
            return None

    def _refresh_profiles(self) -> None:
        store = self._profile_store()
        if store is None:
            self.profile.options = {}
            return

        try:
            profiles = store.list()
        except Exception:
            self.profile.options = {}
            return

        self.profile.options = {
            f"{p.get('name') or p.get('profile_id')} | {p.get('recipe_title') or p.get('recipe_id')}": p.get("profile_id")
            for p in profiles
        }

    def refresh(self) -> None:
        recipes = self.registry.list()
        self.recipe.options = {recipe.title: recipe.id for recipe in recipes}

        if recipes and not self.recipe.value:
            self.recipe.value = recipes[0].id

        dataset_ids = _dataset_access.list_dataset_ids(self.context)
        self.dataset.options = dataset_ids

        active = _dataset_access.active_dataset_id(self.context)
        if active in dataset_ids:
            self.dataset.value = active
        elif self.dataset.value not in dataset_ids:
            self.dataset.value = dataset_ids[0] if dataset_ids else None

        self._refresh_profiles()
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
        self.param_fields = {}
        self.protocol_widgets = {}
        self.protocol_fields = {}
        self.params_area.objects = []

        recipe_id = self.recipe.value
        if not recipe_id:
            self.recipe_card.object = "No recipe selected."
            return

        spec = self.registry.get(recipe_id)
        managed = str(getattr(spec.recipe_cls, "execution_mode", "freeform")) == "managed"
        protocol_note = (
            "Validation, best-epoch selection and test evaluation are enforced "
            "by AstronomicAL's protocol (below)."
            if managed
            else "Freeform recipe: manages its own splits and evaluation; no "
            "protocol is enforced."
        )
        self.recipe_card.object = (
            f"**{spec.title}** \n"
            f"`{spec.id}` v{spec.version} \n\n"
            f"{spec.description}\n\n"
            f"- Task: `{spec.task}`\n"
            f"- Modality: `{spec.modality}`\n"
            f"- Complexity: `{spec.complexity}`\n"
            f"- Mode: `{'managed' if managed else 'freeform'}` — {protocol_note}\n"
            f"- Required mappings: `{', '.join(spec.required_mappings) or 'none'}`\n"
            f"- Produces: `{', '.join(spec.produces) or 'none'}`"
        )

        schema = spec.params_schema or {"type": "object", "properties": {}}
        properties = schema.get("properties", {}) or {}

        if properties:
            self.params_area.append(pn.pane.Markdown("#### Recipe parameters"))
        for name, param_schema in properties.items():
            name = str(name)
            widget = self._widget_for_schema(name, param_schema)
            field = self._field(str(param_schema.get("title") or name), widget)

            self.param_widgets[name] = widget
            self.param_fields[name] = field
            self.params_area.append(field)

        # Protocol section is panel-owned and shown only for managed recipes.
        self._build_protocol_section(spec)

        self._apply_inferred_defaults()

    # ------------------------------------------------------------------
    # Protocol section (managed recipes only)
    # ------------------------------------------------------------------

    def _build_protocol_section(self, spec: Any) -> None:
        self.protocol_widgets = {}
        self.protocol_fields = {}

        if str(getattr(spec.recipe_cls, "execution_mode", "freeform")) != "managed":
            return

        cols = [""] + _dataset_access.list_dataset_columns(
            self.context,
            self.dataset.value,
        )

        dataset_ids = _dataset_access.list_dataset_ids(self.context)
        dataset_options = [""] + dataset_ids

        self.protocol_widgets = {
            "protocol_split_strategy": pn.widgets.Select(
                name="",
                options=[
                    "random",
                    "by_group",
                    "temporal",
                    "predefined",
                ],
                value="random",
                sizing_mode="stretch_width",
            ),
            "protocol_validation_source": pn.widgets.Select(
                name="",
                options={
                    "Split from selected dataset": "split",
                    "Use separate validation dataset": "dataset",
                },
                value="split",
                sizing_mode="stretch_width",
            ),
            "protocol_validation_dataset_id": pn.widgets.Select(
                name="",
                options=dataset_options,
                value="",
                sizing_mode="stretch_width",
            ),
            "protocol_test_source": pn.widgets.Select(
                name="",
                options={
                    "Split from selected dataset": "split",
                    "Use separate test dataset": "dataset",
                    "No test set": "none",
                },
                value="split",
                sizing_mode="stretch_width",
            ),
            "protocol_test_dataset_id": pn.widgets.Select(
                name="",
                options=dataset_options,
                value="",
                sizing_mode="stretch_width",
            ),
            "protocol_group_column": pn.widgets.Select(
                name="",
                options=cols,
                value="",
                sizing_mode="stretch_width",
            ),
            "protocol_split_column": pn.widgets.Select(
                name="",
                options=cols,
                value="",
                sizing_mode="stretch_width",
            ),
            "protocol_validation_size": pn.widgets.FloatInput(
                name="",
                value=0.1,
                start=0.01,
                end=0.8,
                step=0.01,
                sizing_mode="stretch_width",
            ),
            "protocol_test_size": pn.widgets.FloatInput(
                name="",
                value=0.2,
                start=0.0,
                end=0.8,
                step=0.01,
                sizing_mode="stretch_width",
            ),
            "protocol_selection_metric": pn.widgets.Select(
                name="",
                options=["val_accuracy", "val_f1_macro", "val_loss"],
                value="val_accuracy",
                sizing_mode="stretch_width",
            ),
            "protocol_random_state": pn.widgets.IntInput(
                name="",
                value=42,
                sizing_mode="stretch_width",
            ),
        }

        self.params_area.append(
            pn.pane.Markdown(
                "#### Validation/test protocol enforced by AstronomicAL"
            )
        )

        for name in _PROTOCOL_KEYS:
            widget = self.protocol_widgets[name]
            field = self._field(
                _PROTOCOL_LABELS.get(name, name),
                widget,
            )
            self.protocol_fields[name] = field
            self.params_area.append(field)

        for key in (
            "protocol_split_strategy",
            "protocol_validation_source",
            "protocol_test_source",
        ):
            try:
                self.protocol_widgets[key].param.watch(
                    lambda *_: self._sync_protocol_visibility(),
                    "value",
                )
            except Exception:
                pass

        self._sync_protocol_visibility()

    def _sync_protocol_visibility(self) -> None:
        if not self.protocol_widgets:
            return

        split_strategy = str(
            getattr(
                self.protocol_widgets.get("protocol_split_strategy"),
                "value",
                "",
            )
            or "random"
        ).strip()

        validation_source = str(
            getattr(
                self.protocol_widgets.get("protocol_validation_source"),
                "value",
                "",
            )
            or "split"
        ).strip()

        test_source = str(
            getattr(
                self.protocol_widgets.get("protocol_test_source"),
                "value",
                "",
            )
            or "split"
        ).strip()

        val_from_split = validation_source == "split"
        test_from_split = test_source == "split"
        any_from_split = val_from_split or test_from_split

        def _show(name: str, visible: bool) -> None:
            field = self.protocol_fields.get(name)
            if field is not None:
                field.visible = visible

        _show("protocol_validation_dataset_id", validation_source == "dataset")
        _show("protocol_test_dataset_id", test_source == "dataset")

        _show(
            "protocol_group_column",
            any_from_split and split_strategy in ("by_group", "temporal"),
        )

        _show(
            "protocol_split_column",
            any_from_split and split_strategy == "predefined",
        )

        _show(
            "protocol_validation_size",
            val_from_split and split_strategy != "predefined",
        )

        _show(
            "protocol_test_size",
            test_from_split and split_strategy != "predefined",
        )

        _show(
            "protocol_random_state",
            any_from_split and split_strategy in ("random", "by_group"),
        )

        _show("protocol_selection_metric", True)

    def _refresh_recipe_column_widgets(self) -> None:
        columns = list(_dataset_access.list_dataset_columns(self.context, self.dataset.value))
        column_options = [""] + columns

        for name, widget in list(self.param_widgets.items()):
            lower_name = str(name).lower()

            if isinstance(widget, pn.widgets.MultiChoice) and lower_name in {
                "feature_columns",
                "input_columns",
                "features",
                "x_columns",
            }:
                current = [
                    str(value)
                    for value in list(widget.value or [])
                    if str(value) in columns
                ]
                widget.options = columns
                widget.value = current
                continue

            if isinstance(widget, pn.widgets.Select):
                looks_like_column = (
                    lower_name.endswith("_column")
                    or lower_name
                    in {
                        "target",
                        "label",
                        "target_column",
                        "label_column",
                        "record_id_column",
                        "id_column",
                        "image_column",
                        "image_path_column",
                        "image_uri_column",
                        "group_column",
                        "split_column",
                    }
                )
                if looks_like_column:
                    current = widget.value
                    widget.options = column_options
                    widget.value = current if current in column_options else ""

    def _refresh_protocol_columns(self) -> None:
        if not self.protocol_widgets:
            return

        cols = [""] + _dataset_access.list_dataset_columns(
            self.context,
            self.dataset.value,
        )

        for key in ("protocol_group_column", "protocol_split_column"):
            widget = self.protocol_widgets.get(key)
            if widget is None:
                continue

            current = widget.value
            widget.options = cols
            widget.value = current if current in cols else ""

        dataset_ids = _dataset_access.list_dataset_ids(self.context)
        dataset_options = [""] + dataset_ids

        for key in (
            "protocol_validation_dataset_id",
            "protocol_test_dataset_id",
        ):
            widget = self.protocol_widgets.get(key)
            if widget is None:
                continue

            current = widget.value
            widget.options = dataset_options
            widget.value = current if current in dataset_options else ""

    # ------------------------------------------------------------------

    def _on_dataset_change(self, *_: Any) -> None:
        self._refresh_recipe_column_widgets()
        self._refresh_protocol_columns()
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

        applied = []

        if inferred:
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

        if applied:
            self.status.alert_type = "info"
            self.status.object = (
                "Inferred recipe inputs from the selected dataset: "
                + ", ".join(applied)
            )

    def _column_list_from_value(self, value: Any) -> List[str]:
        return parse_column_list(value)

    def _widget_for_schema(self, name: str, schema: Mapping[str, Any]):
        kind = str(schema.get("type", "string"))
        default = schema.get("default", "")

        widget_kind = str(schema.get("x-widget") or schema.get("widget") or "")

        if widget_kind == "dataset_select" or name.endswith("_dataset_id"):
            dataset_ids = _dataset_access.list_dataset_ids(self.context)
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

        # Recipe internals (schema-generated widgets).
        for name, widget in self.param_widgets.items():
            value = widget.value
            schema = properties.get(name, {})
            kind = str(schema.get("type", "string"))

            if kind in {"array", "object"} and isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    # Keep the raw string; the recipe/runner handles validation.
                    pass

            params[name] = value

        # Protocol controls (panel-owned; only present for managed recipes).
        for name, widget in self.protocol_widgets.items():
            params[name] = widget.value

        params["recipe_id"] = self.recipe.value
        params["dataset_id"] = self.dataset.value
        return params

    def _save_profile_clicked(self, *_: Any) -> None:
        if not self.recipe.value:
            self.status.alert_type = "danger"
            self.status.object = "Choose a recipe before saving a profile."
            return

        store = self._profile_store()
        if store is None:
            self.status.alert_type = "danger"
            self.status.object = "Recipe profile store is not available."
            return

        spec = self.registry.get(self.recipe.value)
        params = self._params()

        from ..profiles import split_profile_params
        recipe_params, protocol_params, binding_params = split_profile_params(params)

        name = str(self.profile_name.value or "").strip()
        if not name:
            name = f"{spec.title} profile"

        existing_profile_id = self.profile.value if self.profile.value else ""

        payload = {
            "profile_id": existing_profile_id,
            "name": name,
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "execution_mode": str(getattr(spec.recipe_cls, "execution_mode", "freeform") or "freeform"),
            "task": getattr(spec, "task", ""),
            "modality": getattr(spec, "modality", ""),
            "default_dataset_id": self.dataset.value or "",
            "recipe_params": recipe_params,
            "protocol_params": protocol_params,
            "binding_params": binding_params,
            "source": "recipe_launcher",
        }

        try:
            artifact_id = store.save(payload)
        except Exception as exc:
            self.status.alert_type = "danger"
            self.status.object = f"Could not save recipe profile: `{exc}`"
            return

        saved_profile_id = ""
        try:
            saved_payload = store.get(artifact_id)
            saved_profile_id = str(saved_payload.get("profile_id") or "")
        except Exception:
            saved_profile_id = str(payload.get("profile_id") or existing_profile_id or "")

        self._refresh_profiles()
        if saved_profile_id in self.profile.options.values():
            self.profile.value = saved_profile_id

        self.status.alert_type = "success"
        self.status.object = f"Saved recipe profile `{name}` as `{artifact_id}`."

    def _load_profile_clicked(self, *_: Any) -> None:
        profile_id = self.profile.value
        if not profile_id:
            self.status.alert_type = "warning"
            self.status.object = "Choose a saved profile first."
            return

        store = self._profile_store()
        if store is None:
            self.status.alert_type = "danger"
            self.status.object = "Recipe profile store is not available."
            return

        try:
            profile = store.get(profile_id)
        except Exception as exc:
            self.status.alert_type = "danger"
            self.status.object = f"Could not load recipe profile: `{exc}`"
            return

        recipe_id = profile.get("recipe_id")
        if recipe_id in self.recipe.options.values():
            self.recipe.value = recipe_id
            self._on_recipe_change()

        dataset_id = profile.get("default_dataset_id")
        if dataset_id in self.dataset.options:
            self.dataset.value = dataset_id

        merged = {}
        merged.update(profile.get("recipe_params") or {})
        merged.update(profile.get("protocol_params") or {})
        merged.update(profile.get("binding_params") or {})

        for name, value in merged.items():
            widget = self.param_widgets.get(name) or self.protocol_widgets.get(name)
            if widget is None:
                continue
            try:
                widget.value = value
            except Exception:
                pass

        self.profile_name.value = str(profile.get("name") or "")
        self._sync_protocol_visibility()

        self.status.alert_type = "success"
        self.status.object = f"Loaded recipe profile `{profile.get('name') or profile_id}`."

    def _run_clicked(self, *_: Any) -> None:
        if self._active_handle is not None:
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

        def on_done(result: Dict[str, Any]) -> None:
            status = str(result.get("status") or "").lower()
            self._active_handle = None
            self._set_running_state(False)
            self._update_result(result, success=status != "cancelled", cancelled=status == "cancelled")

        def on_error(exc: BaseException) -> None:
            self._active_handle = None
            self._set_running_state(False)
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self._update_result({"error": str(exc), "traceback": tb}, success=False, cancelled=False)

        self._active_handle = submit_job(
            self.context,
            _runner.run_ml_recipe_action,
            title=f"Run ML recipe: {self.recipe.value}",
            key=f"core.ml.recipe:{params['dataset_id']}:{params['recipe_id']}",
            on_done=on_done,
            on_error=on_error,
            context=self.context,
            request=request,
        )

    def _cancel_clicked(self, *_: Any) -> None:
        if self._active_handle is None:
            self.status.alert_type = "warning"
            self.status.object = "No recipe run is currently active from this panel."
            return

        try:
            self._active_handle.cancel()
        except Exception:
            token = getattr(self._active_handle, "token", None)
            if token is not None:
                try:
                    token.cancel()
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

    def _subscribe_to_profile_events(self) -> None:
        events = getattr(self.context, "events", None)
        subscribe = getattr(events, "subscribe", None)
        if not callable(subscribe):
            return
        for topic in ("ml.recipe_profile.saved", "ml.recipe_profiles.changed"):
            try:
                self._subscriptions.append(
                    subscribe(topic, self._on_profile_event, owner_label="ML Recipe Launcher", owner_kind="panel")
                )
            except Exception:
                pass

    def _on_profile_event(self, topic: str, payload: Any) -> None:
        def update():
            self._refresh_profiles()
        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(update)
                return
        except Exception:
            pass
        update()

def create_recipe_launcher_panel(context: Any, **kwargs: Any):
    registry = context.services.get("core.ml.recipe_registry")
    controller = MLRecipeLauncherPanel(
        context=context,
        registry=registry,
        restore_state=kwargs.get("restore_state"),
    )
    return controller.panel(), controller
