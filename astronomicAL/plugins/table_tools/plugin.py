from __future__ import annotations

import html
import uuid
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import panel as pn

from astronomicAL.platform.plugins import PluginManifest
from astronomicAL.platform.plugins.specs import (
    ActionRequest,
    ActionResult,
    EventResult,
    InputSpec,
)


manifest = PluginManifest(
    id="core.table_tools",
    name="Table Tools",
    version="0.2.0",
    description=(
        "Expression-based table transforms: add derived columns and create "
        "subset datasets from boolean pandas expressions."
    ),
    capabilities=["panel", "actions", "datasets", "table-transform"],
    tags=["core", "table", "datasets", "transforms"],
)


def register(api) -> None:
    api.register_panel(
        id="transform_panel",
        title="Table Transform",
        factory=create_table_transform_panel,
        description=(
            "Create derived columns with pandas-style expressions and create "
            "subset datasets with boolean expressions."
        ),
        category="Data tools",
        icon="table",
        tags=["table", "dataset", "transform"],
        default_layout={"x": 0, "y": 0, "w": 6, "h": 6},
    )

    api.register_action(
        id="add_column",
        title="Add derived column",
        handler=add_column_action,
        inputs=InputSpec(dataset=True, selection="none", columns="optional"),
        outputs=["dataset.updated"],
        params_schema={
            "type": "object",
            "properties": {
                "new_column": {"type": "string"},
                "expression": {"type": "string"},
            },
            "required": ["new_column", "expression"],
            "additionalProperties": False,
        },
        run_in_job=False,
        description="Evaluate a pandas expression and add it as a new column.",
        category="Data tools",
        tags=["table", "derived-column"],
    )

    api.register_action(
        id="create_subset",
        title="Create subset dataset",
        handler=create_subset_action,
        inputs=InputSpec(dataset=True, selection="none", columns="optional"),
        outputs=["dataset.loaded", "dataset.active.changed"],
        params_schema={
            "type": "object",
            "properties": {
                "subset_name": {"type": "string"},
                "expression": {"type": "string"},
                "set_active": {"type": "boolean", "default": True},
            },
            "required": ["subset_name", "expression"],
            "additionalProperties": False,
        },
        run_in_job=False,
        description="Evaluate a boolean pandas expression and register a subset dataset.",
        category="Data tools",
        tags=["table", "subset"],
    )


def create_table_transform_panel(context, data=None, **kwargs):
    controller = TableTransformPanel(
        context=context,
        data=data,
    )
    return controller.view, controller


# ---------------------------------------------------------------------
# Plugin actions
# ---------------------------------------------------------------------


def add_column_action(context, request: ActionRequest, **_kwargs) -> ActionResult:
    dataset_id = _request_dataset_id(context, request)
    new_column = str(request.params.get("new_column", "")).strip()
    expression = str(request.params.get("expression", "")).strip()

    if not new_column:
        raise ValueError("Please provide a new column name.")
    if not expression:
        raise ValueError("Please provide a column expression.")

    df = _active_df(context, dataset_id).copy()

    if new_column in df.columns:
        raise ValueError(f"Column `{new_column}` already exists.")

    result = _evaluate_expression(expression, df)
    new_df = df.copy()
    new_df[new_column] = result

    _sync_dataset(
        context,
        dataset_id,
        new_df,
        extra_meta={
            "last_transform_type": "add_column",
            "last_transform_column": new_column,
            "last_transform_expr": expression,
        },
    )

    return ActionResult(
        value={
            "dataset_id": dataset_id,
            "column": new_column,
            "rows": len(new_df),
        },
        events=[
            EventResult(
                "dataset.updated",
                {
                    "dataset_id": dataset_id,
                    "change": "column.added",
                    "column": new_column,
                    "origin": manifest.id,
                },
            )
        ],
    )


def create_subset_action(context, request: ActionRequest, **_kwargs) -> ActionResult:
    base_dataset_id = _request_dataset_id(context, request)
    subset_name = str(request.params.get("subset_name", "")).strip()
    expression = str(request.params.get("expression", "")).strip()
    set_active = bool(request.params.get("set_active", True))

    if not subset_name:
        raise ValueError("Please provide a subset dataset name.")
    if not expression:
        raise ValueError("Please provide a subset expression.")

    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise RuntimeError("DatasetManager is required to create subset datasets.")

    base_df = _active_df(context, base_dataset_id).copy()
    mask = _evaluate_boolean_expression(expression, base_df)
    filtered = base_df.loc[mask].copy()

    new_dataset_id = f"{base_dataset_id}__subset__{uuid.uuid4().hex[:8]}"

    datasets.register(
        new_dataset_id,
        filtered,
        name=subset_name,
        derived_from=base_dataset_id,
        filter=expression,
        created_by=manifest.id,
    )

    previous_dataset_id = base_dataset_id

    events = [
        EventResult(
            "dataset.loaded",
            {
                "dataset_id": new_dataset_id,
                "derived_from": base_dataset_id,
                "origin": manifest.id,
            },
        )
    ]

    if set_active:
        datasets.set_active(new_dataset_id)
        _sync_config_main_df(context, filtered)
        events.append(
            EventResult(
                "dataset.active.changed",
                {
                    "dataset_id": new_dataset_id,
                    "previous_dataset_id": previous_dataset_id,
                    "origin": manifest.id,
                },
            )
        )

    return ActionResult(
        value={
            "dataset_id": new_dataset_id,
            "name": subset_name,
            "rows": len(filtered),
            "derived_from": base_dataset_id,
            "set_active": set_active,
        },
        events=events,
    )


# ---------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------


class TableTransformPanel:
    """Plugin version of the original context-core TableTransformPanel.

    Provides the original interaction model:

    - list/filter active dataset columns
    - preview a derived-column expression
    - add the derived column to the active dataset
    - preview a boolean subset expression
    - register a subset dataset
    - optionally make that subset active
    """

    def __init__(self, context, data=None):
        self.context = context
        self.data = data
        self._disposed = False
        self._subscriptions = []
        self._watchers: list[tuple[Any, Any]] = []

        common_margin = (0, 0, 6, 0)

        self.help_text = pn.pane.Markdown(
            (
                "Create derived columns with pandas-style expressions and create "
                "subset datasets with boolean filters.\n\n"
                "**Examples**\n"
                "- `col1 + col2`\n"
                "- `col1 ** 2`\n"
                "- `(col1 + col2) / (col3 + col4)`\n"
                "- `(score > 0.9) & (redshift < 1.0)`"
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.dataset_info = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        self.status = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        self.column_filter = pn.widgets.TextInput(
            name="Filter columns",
            placeholder="Type to filter columns, e.g. mag",
            sizing_mode="stretch_width",
            margin=common_margin,
        )

        self.available_columns = pn.pane.HTML(
            "",
            height=160,
            sizing_mode="stretch_width",
            margin=common_margin,
            styles={
                "border": "1px solid #bfbfbf",
                "border-radius": "4px",
                "padding": "8px",
                "background": "#f5f5f5",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "font-family": "monospace",
                "font-size": "13px",
                "line-height": "1.35",
            },
        )

        self.new_column_name = pn.widgets.TextInput(
            name="New column name",
            placeholder="e.g. colour_index",
            sizing_mode="stretch_width",
            margin=common_margin,
        )

        self.new_column_expr = pn.widgets.TextAreaInput(
            name="Column formula",
            placeholder="e.g. mag_g - mag_r",
            height=70,
            sizing_mode="stretch_width",
            margin=common_margin,
        )

        self.preview_column_button = pn.widgets.Button(
            name="Preview Column",
            button_type="default",
            height=32,
            sizing_mode="stretch_width",
            margin=0,
        )

        self.add_column_button = pn.widgets.Button(
            name="Add Column",
            button_type="primary",
            height=32,
            sizing_mode="stretch_width",
            margin=0,
        )

        self.subset_name = pn.widgets.TextInput(
            name="Subset dataset name",
            placeholder="e.g. high_confidence_sources",
            sizing_mode="stretch_width",
            margin=common_margin,
        )

        self.subset_expr = pn.widgets.TextAreaInput(
            name="Subset boolean filter",
            placeholder="e.g. (score > 0.9) & (redshift < 1.0)",
            height=90,
            min_height=90,
            sizing_mode="stretch_width",
            margin=common_margin,
        )

        self.set_active_checkbox = pn.widgets.Checkbox(
            name="Set new subset as active dataset",
            value=True,
            margin=common_margin,
        )

        self.preview_subset_button = pn.widgets.Button(
            name="Preview Subset",
            button_type="default",
            height=32,
            sizing_mode="stretch_width",
            margin=0,
        )

        self.create_subset_button = pn.widgets.Button(
            name="Create Subset Dataset",
            button_type="primary",
            height=32,
            sizing_mode="stretch_width",
            margin=0,
        )

        self.preview = pn.pane.DataFrame(
            pd.DataFrame(),
            height=140,
            min_height=140,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self._watch(self.column_filter, self._on_column_filter_change, "value_input")
        self._watch(self.column_filter, self._on_column_filter_change, "value")

        self.preview_column_button.on_click(self._preview_column)
        self.add_column_button.on_click(self._add_column)
        self.preview_subset_button.on_click(self._preview_subset)
        self.create_subset_button.on_click(self._create_subset_dataset)

        self._subscribe_to_dataset_events()
        self.view = self._build_view()
        self._refresh_metadata_panes()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_view(self):

        active_section = self._section(
            "Active dataset",
            self.dataset_info,
            self.column_filter,
            self.available_columns,
        )

        derived_section = self._section(
            "Add derived column",
            self.new_column_name,
            self.new_column_expr,
            self._button_row(self.preview_column_button, self.add_column_button),
            height=205,
        )

        subset_section = self._section(
            "Create subset dataset",
            self.subset_name,
            self.subset_expr,
            self.set_active_checkbox,
            self._button_row(self.preview_subset_button, self.create_subset_button),
            height=245,
        )

        preview_section = self._section(
            "Preview / Output",
            self.status,
            self.preview,
        )

        title = pn.pane.HTML(
            "<h2 style='margin: 0; padding: 0; line-height: 1.2;'>Table Transform</h2>",
            height=34,
            min_height=34,
            max_height=34,
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        body = pn.Column(
            title,
            self.help_text,
            active_section,
            derived_section,
            subset_section,
            preview_section,
            sizing_mode="stretch_width",
            scroll=True,
            margin=(0, 0, 0, 0),
            styles={
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "padding": "0 8px 8px 8px",
            },
        )

        return pn.Column(
            body,
            sizing_mode="stretch_both",
            min_height=450,
        )

    @staticmethod
    def _button_row(left_button, right_button):
        return pn.Row(
            left_button,
            pn.Spacer(width=8),
            right_button,
            sizing_mode="stretch_width",
            margin=(2, 0, 0, 0),
        )

    @staticmethod
    def _section(title: str, *objects, height=None):
        return pn.Column(
            pn.pane.HTML(
                f"<h3 style='margin: 0 0 8px 0; font-size: 15px;'>{html.escape(title)}</h3>",
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            ),
            *objects,
            sizing_mode="stretch_width",
            height=height,
            margin=(0, 0, 8, 0),
            styles={
                "border": "1px solid #d9d9d9",
                "border-radius": "4px",
                "padding": "8px",
                "background": "#fafafa",
                "overflow": "visible",
                "flex": "0 0 auto",
            },
        )

    # ------------------------------------------------------------------
    # Dataset/event helpers
    # ------------------------------------------------------------------

    def _subscribe_to_dataset_events(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        for topic in (
            "dataset.loaded",
            "dataset.active.changed",
            "dataset.updated",
            "dataset.mapping_updated",
        ):
            try:
                sub = events.subscribe(
                    topic,
                    self._dataset_event_received,
                    owner_id="core.table_tools.transform_panel",
                    owner_label="Table Transform",
                    owner_kind="plugin-panel",
                )
            except TypeError:
                sub = events.subscribe(topic, self._dataset_event_received)

            self._subscriptions.append(sub)

    def _dataset_event_received(self, topic: str, payload: Any) -> None:
        if self._disposed:
            return

        def _refresh():
            if not self._disposed:
                self._refresh_metadata_panes()

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(_refresh)
            else:
                _refresh()
        except Exception:
            _refresh()

    def _watch(self, widget, callback, attr: str) -> None:
        try:
            watcher = widget.param.watch(callback, attr)
            self._watchers.append((widget, watcher))
        except Exception:
            pass

    def _active_dataset_id(self) -> str:
        return _active_dataset_id(self.context)

    def _active_df(self) -> pd.DataFrame:
        try:
            return _active_df(self.context, self._active_dataset_id()).copy()
        except Exception:
            if self.data is not None:
                return self.data.copy()
            return pd.DataFrame()

    def _active_dataset_name(self) -> str:
        return _active_dataset_name(self.context, self._active_dataset_id())

    def _refresh_metadata_panes(self) -> None:
        df = self._active_df()
        dataset_id = self._active_dataset_id()
        dataset_name = self._active_dataset_name()

        self.dataset_info.object = (
            f"**Active dataset:** `{dataset_name}` (`{dataset_id}`)  \n"
            f"**Rows:** {len(df)}  \n"
            f"**Columns:** {len(df.columns)}"
        )

        self._update_available_columns_view()

    def _on_column_filter_change(self, _event=None) -> None:
        self._update_available_columns_view()

    def _update_available_columns_view(self) -> None:
        df = self._active_df()
        all_columns = [str(c) for c in df.columns]
        query = (getattr(self.column_filter, "value_input", None) or self.column_filter.value or "")
        query = query.strip().lower()

        if query:
            shown_columns = [c for c in all_columns if query in c.lower()]
        else:
            shown_columns = all_columns

        if shown_columns:
            escaped_columns = [html.escape(c) for c in shown_columns]
            body = "<br>".join(escaped_columns)
        else:
            body = "<em>No matching columns</em>"

        self.available_columns.object = (
            f"<div><strong>{len(shown_columns)} / {len(all_columns)} columns shown</strong></div>"
            f"<div style='margin-top: 6px;'>{body}</div>"
        )

    def _set_preview_df(self, df: pd.DataFrame) -> None:
        self.preview.object = df

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        try:
            events.publish(topic, payload)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Actions: derived columns
    # ------------------------------------------------------------------

    def _preview_column(self, _event=None) -> None:
        try:
            df = self._active_df()
            result = _evaluate_expression(self.new_column_expr.value, df)
            preview_df = pd.DataFrame({"__preview_result__": result.head(20).values})
            self._set_preview_df(preview_df)
            self.status.object = "Preview generated successfully."
        except Exception as exc:
            self.status.object = f"Column preview failed: `{exc}`"

    def _add_column(self, _event=None) -> None:
        try:
            manager = getattr(self.context, "plugins", None)

            params = {
                "new_column": self.new_column_name.value,
                "expression": self.new_column_expr.value,
            }

            if manager is not None:
                result = manager.run_action(
                    "core.table_tools.add_column",
                    self.context,
                    ActionRequest(
                        dataset_id=self._active_dataset_id(),
                        params=params,
                        origin="core.table_tools.transform_panel",
                    ),
                    return_processed=True,
                )
                value = getattr(result, "value", None) or {}
                new_col = value.get("column", params["new_column"])
            else:
                request = ActionRequest(
                    dataset_id=self._active_dataset_id(),
                    params=params,
                    origin="core.table_tools.transform_panel",
                )
                add_column_action(self.context, request)
                new_col = params["new_column"]

            df = self._active_df()
            if new_col in df.columns:
                self._set_preview_df(df[[new_col]].head(20))

            self._refresh_metadata_panes()
            self.status.object = f"Added new column `{new_col}` to the active dataset."
        except Exception as exc:
            self.status.object = f"Add column failed: `{exc}`"

    # ------------------------------------------------------------------
    # Actions: subsets
    # ------------------------------------------------------------------

    def _preview_subset(self, _event=None) -> None:
        try:
            df = self._active_df()
            mask = _evaluate_boolean_expression(self.subset_expr.value, df)
            filtered = df.loc[mask].copy()

            self._set_preview_df(filtered.head(50))
            self.status.object = (
                f"Subset preview generated: **{len(filtered)} / {len(df)}** rows matched."
            )
        except Exception as exc:
            self.status.object = f"Subset preview failed: `{exc}`"

    def _create_subset_dataset(self, _event=None) -> None:
        try:
            manager = getattr(self.context, "plugins", None)

            params = {
                "subset_name": self.subset_name.value,
                "expression": self.subset_expr.value,
                "set_active": bool(self.set_active_checkbox.value),
            }

            if manager is not None:
                result = manager.run_action(
                    "core.table_tools.create_subset",
                    self.context,
                    ActionRequest(
                        dataset_id=self._active_dataset_id(),
                        params=params,
                        origin="core.table_tools.transform_panel",
                    ),
                    return_processed=True,
                )
                value = getattr(result, "value", None) or {}
            else:
                request = ActionRequest(
                    dataset_id=self._active_dataset_id(),
                    params=params,
                    origin="core.table_tools.transform_panel",
                )
                value = create_subset_action(self.context, request).value or {}

            df = self._active_df()
            self._set_preview_df(df.head(50))
            self._refresh_metadata_panes()

            subset_name = value.get("name", params["subset_name"])
            row_count = value.get("rows", len(df))

            self.status.object = (
                f"Created subset dataset `{subset_name}` with **{row_count}** rows."
            )
        except Exception as exc:
            self.status.object = f"Create subset dataset failed: `{exc}`"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        if self._disposed:
            return

        self._disposed = True

        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in list(self._subscriptions):
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass

        self._subscriptions.clear()

        for widget, watcher in list(self._watchers):
            try:
                widget.param.unwatch(watcher)
            except Exception:
                pass

        self._watchers.clear()


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


def _request_dataset_id(context, request: Optional[ActionRequest] = None) -> str:
    if request is not None and request.dataset_id:
        return str(request.dataset_id)
    return _active_dataset_id(context)


def _active_dataset_id(context) -> str:
    datasets = getattr(context, "datasets", None)
    if datasets is not None:
        try:
            return datasets.active_id()
        except Exception:
            pass
    return "default"


def _active_df(context, dataset_id: Optional[str] = None) -> pd.DataFrame:
    datasets = getattr(context, "datasets", None)
    if datasets is not None:
        try:
            return datasets.get_df(dataset_id)
        except Exception:
            pass

    config = getattr(context, "config", None)
    if config is not None and getattr(config, "main_df", None) is not None:
        return config.main_df

    return pd.DataFrame()


def _active_dataset_name(context, dataset_id: Optional[str] = None) -> str:
    datasets = getattr(context, "datasets", None)
    if datasets is not None:
        try:
            return datasets.get(dataset_id).name
        except Exception:
            pass
    return str(dataset_id or _active_dataset_id(context))


def _active_dataset_meta(context, dataset_id: Optional[str] = None) -> Dict[str, Any]:
    datasets = getattr(context, "datasets", None)
    if datasets is not None:
        try:
            return dict(datasets.get_meta(dataset_id))
        except Exception:
            pass
    return {}


def _sync_config_main_df(context, df: pd.DataFrame) -> None:
    config = getattr(context, "config", None)
    if config is not None:
        try:
            config.main_df = df
        except Exception:
            pass


def _sync_dataset(
    context,
    dataset_id: str,
    df: pd.DataFrame,
    *,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    extra_meta = dict(extra_meta or {})
    datasets = getattr(context, "datasets", None)

    if datasets is not None:
        dataset_name = _active_dataset_name(context, dataset_id)
        meta = _active_dataset_meta(context, dataset_id)
        meta.update(extra_meta)

        ensure_registered = getattr(datasets, "ensure_registered", None)
        if callable(ensure_registered):
            ensure_registered(
                dataset_id,
                df,
                name=dataset_name,
                **meta,
            )
        else:
            datasets.register(
                dataset_id,
                df,
                name=dataset_name,
                **meta,
            )

    if dataset_id == _active_dataset_id(context):
        _sync_config_main_df(context, df)


def _evaluate_expression(expr: str, df: pd.DataFrame) -> pd.Series:
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Expression is empty.")

    local_dict = {
        "np": np,
        "pd": pd,
    }

    result = df.eval(
        expr,
        parser="pandas",
        engine="python",
        local_dict=local_dict,
    )

    if isinstance(result, pd.DataFrame):
        raise ValueError("Expression returned a DataFrame. Expected a single column/series.")

    if isinstance(result, pd.Series):
        return result.reindex(df.index)

    if isinstance(result, np.ndarray):
        if len(result) != len(df):
            raise ValueError(
                f"Expression returned an array of length {len(result)}, expected {len(df)}."
            )
        return pd.Series(result, index=df.index)

    if isinstance(result, list):
        if len(result) != len(df):
            raise ValueError(
                f"Expression returned a list of length {len(result)}, expected {len(df)}."
            )
        return pd.Series(result, index=df.index)

    return pd.Series([result] * len(df), index=df.index)


def _evaluate_boolean_expression(expr: str, df: pd.DataFrame) -> pd.Series:
    result = _evaluate_expression(expr, df)

    if not pd.api.types.is_bool_dtype(result):
        raise ValueError(
            "Subset expression must evaluate to boolean values. "
            "Use comparisons like `(col > 0)` and combine them with `&`, `|`, `~`."
        )

    return result.fillna(False).astype(bool)