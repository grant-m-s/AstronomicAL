# BUG: On reload: Auto defaults to subset filter dataset (ignores the original dataset direct from file)
# BUG: On reload: Failed to create panel core.table_tools.transform_panel: 'Unknown dataset_id: default'

from __future__ import annotations

import html
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import panel as pn

from astronomicAL.platform.parquet_cache import (
    default_cache_dir_for_context,
    normalise_dataset_id,
    register_dataframe_as_parquet,
    replace_dataset_with_dataframe_parquet,
)

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
    params = request.params or {}

    new_column = str(params.get("new_column", "")).strip()
    expression = str(params.get("expression", "")).strip()

    if not new_column:
        raise ValueError("Please provide a new column name.")

    if not expression:
        raise ValueError("Please provide a column expression.")

    columns = _dataset_columns(context, dataset_id)

    if new_column in columns:
        raise ValueError(f"Column `{new_column}` already exists.")

    source = _active_source(context, dataset_id)

    # Important:
    # Use the relation predicate, not the plain-parquet predicate.
    # This supports:
    # - duckdb_parquet
    # - duckdb_parquet_filtered
    # - duckdb_parquet_derived_column
    if _is_duckdb_relation_source(source):
        row_count = _add_column_duckdb_parquet(
            context,
            dataset_id=dataset_id,
            source=source,
            new_column=new_column,
            expression=expression,
        )
        materialized = False
        backend = getattr(_active_source(context, dataset_id), "backend_name", None)
    else:
        row_count = _add_column_pandas_fallback(
            context,
            dataset_id=dataset_id,
            new_column=new_column,
            expression=expression,
        )
        materialized = True
        backend = getattr(_active_source(context, dataset_id), "backend_name", None)

    return ActionResult(
        value={
            "dataset_id": dataset_id,
            "column": new_column,
            "rows": row_count,
            "materialized": materialized,
            "backend": backend,
        },
        events=[
            EventResult(
                "dataset.updated",
                {
                    "dataset_id": dataset_id,
                    "change": "column.added",
                    "column": new_column,
                    "changed_columns": [new_column],
                    "added_columns": [new_column],
                    "schema_changed": True,
                    "row_filter_changed": False,
                    "filter_changed": False,
                    "row_count_changed": False,
                    "rows_changed": False,
                    "data_changed": False,
                    "origin": manifest.id,
                    "materialized": materialized,
                    "backend": backend,
                },
            )
        ],
    )

def create_subset_action(context, request: ActionRequest, **_kwargs) -> ActionResult:
    base_dataset_id = _request_dataset_id(context, request)

    params = request.params or {}
    subset_name = str(params.get("subset_name", "")).strip()
    expression = str(params.get("expression", "")).strip()
    set_active = bool(params.get("set_active", True))

    if not subset_name:
        raise ValueError("Please provide a subset dataset name.")

    if not expression:
        raise ValueError("Please provide a subset expression.")

    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise RuntimeError("DatasetManager is required to create subset datasets.")

    source = _active_source(context, base_dataset_id)
    new_dataset_id = f"{base_dataset_id}__subset__{uuid.uuid4().hex[:8]}"

    if _is_duckdb_relation_source(source):
        row_count = _create_subset_duckdb_parquet(
            context,
            source=source,
            base_dataset_id=base_dataset_id,
            new_dataset_id=new_dataset_id,
            subset_name=subset_name,
            expression=expression,
        )
        materialized = False
    else:
        row_count = _create_subset_pandas_fallback(
            context,
            base_dataset_id=base_dataset_id,
            new_dataset_id=new_dataset_id,
            subset_name=subset_name,
            expression=expression,
        )
        materialized = True

    _copy_dataset_mappings(
        datasets,
        source_dataset_id=base_dataset_id,
        target_dataset_id=new_dataset_id,
    )

    previous_dataset_id = base_dataset_id

    events = [
        EventResult(
            "dataset.loaded",
            {
                "dataset_id": new_dataset_id,
                "derived_from": base_dataset_id,
                "origin": manifest.id,
                "materialized": materialized,
            },
        )
    ]

    if set_active:
        datasets.set_active(new_dataset_id)
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
            "rows": row_count,
            "derived_from": base_dataset_id,
            "set_active": set_active,
            "materialized": materialized,
            "backend": getattr(_active_source(context, new_dataset_id), "backend_name", None),
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

    def _active_df(
        context,
        dataset_id: Optional[str] = None,
        *,
        limit: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:

        source = _active_source(context, dataset_id)

        if source is not None:
            backend = getattr(source, "backend_name", "unknown")

            if backend != "pandas" and limit is None:
                raise RuntimeError(
                    "Refusing to materialise the full non-pandas dataset. "
                    "Use DatasetSource/DuckDB paths or pass a small limit."
                )

            return source.to_pandas(columns=columns, limit=limit)

        config = getattr(context, "config", None)

        if config is not None and getattr(config, "main_df", None) is not None:
            df = config.main_df

            if columns is not None:
                existing = [col for col in columns if col in df.columns]
                df = df.loc[:, existing]

            if limit is not None:
                df = df.head(int(limit))

            return df.copy()

        return pd.DataFrame()

    def _active_dataset_name(self) -> str:
        return _active_dataset_name(self.context, self._active_dataset_id())

    def _refresh_metadata_panes(self) -> None:
        dataset_id = self._active_dataset_id()
        dataset_name = self._active_dataset_name()
        row_count = _dataset_row_count(self.context, dataset_id)
        columns = _dataset_columns(self.context, dataset_id)

        rows_text = "unknown" if row_count is None else f"{row_count:,}"

        self.dataset_info.object = (
            f"**Active dataset:** `{dataset_name}` (`{dataset_id}`)  \n"
            f"**Rows:** {rows_text}  \n"
            f"**Columns:** {len(columns):,}"
        )

        self._update_available_columns_view()

    def _on_column_filter_change(self, _event=None) -> None:
        self._update_available_columns_view()

    def _update_available_columns_view(self) -> None:
        all_columns = [str(c) for c in _dataset_columns(self.context, self._active_dataset_id())]

        query = (
            getattr(self.column_filter, "value_input", None)
            or self.column_filter.value
            or ""
        )
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
            preview_df = _preview_column_expression(
                self.context,
                dataset_id=self._active_dataset_id(),
                expression=self.new_column_expr.value,
                limit=20,
            )
            self._set_preview_df(preview_df)
            self.status.object = "Preview generated successfully."
        except Exception as exc:
            self.status.object = f"Column preview failed: `{exc}`"

    def _add_column(self, _event=None) -> None:
        try:
            manager = getattr(self.context, "plugins", None)
            dataset_id = self._active_dataset_id()

            params = {
                "new_column": self.new_column_name.value,
                "expression": self.new_column_expr.value,
            }

            new_col = str(params["new_column"]).strip()

            if not new_col:
                self.status.object = "Add column failed: please provide a new column name."
                return

            if not str(params["expression"]).strip():
                self.status.object = "Add column failed: please provide a column formula."
                return

            self.add_column_button.disabled = True
            self.status.object = f"Adding lazy derived column `{new_col}`..."
            self._set_preview_df(pd.DataFrame())

            request = ActionRequest(
                dataset_id=dataset_id,
                params=params,
                origin="core.table_tools.transform_panel",
            )

            if manager is not None:
                result = manager.run_action(
                    "core.table_tools.add_column",
                    self.context,
                    request,
                    return_processed=True,
                )
                value = getattr(result, "value", None) or {}
            else:
                action_result = add_column_action(self.context, request)
                value = action_result.value or {}

            new_col = value.get("column", new_col)
            row_count = value.get("rows")
            materialized = bool(value.get("materialized", False))
            backend = value.get("backend") or getattr(
                _active_source(self.context, dataset_id),
                "backend_name",
                "unknown",
            )

            self.status.object = f"Column `{new_col}` registered. Loading preview..."

            preview_df = _dataset_head(
                self.context,
                dataset_id,
                n=20,
                columns=[new_col],
            )

            self._set_preview_df(preview_df)
            self._refresh_metadata_panes()

            row_text = "unknown" if row_count is None else f"{int(row_count):,}"
            mode_text = "materialised" if materialized else "lazy"

            self.status.object = (
                f"Added `{new_col}` as a **{mode_text}** derived column "
                f"over **{row_text}** rows (`{backend}`)."
            )

        except Exception as exc:
            self.status.object = f"Add column failed: `{exc}`"

        finally:
            self.add_column_button.disabled = False

    # ------------------------------------------------------------------
    # Actions: subsets
    # ------------------------------------------------------------------

    def _preview_subset(self, _event=None) -> None:
        try:
            preview_df, matched_count, total_count = _preview_subset_expression(
                self.context,
                dataset_id=self._active_dataset_id(),
                expression=self.subset_expr.value,
                limit=50,
            )

            self._set_preview_df(preview_df)

            total_text = "unknown" if total_count is None else f"{total_count:,}"
            self.status.object = (
                f"Subset preview generated: **{matched_count:,} / {total_text}** rows matched."
            )
        except Exception as exc:
            self.status.object = f"Subset preview failed: `{exc}`"

    def _create_subset_dataset(self, _event=None) -> None:
        try:
            manager = getattr(self.context, "plugins", None)
            dataset_id = self._active_dataset_id()

            params = {
                "subset_name": self.subset_name.value,
                "expression": self.subset_expr.value,
                "set_active": bool(self.set_active_checkbox.value),
            }

            self.create_subset_button.disabled = True
            self.status.object = (
                f"Creating lazy subset dataset `{params['subset_name']}`..."
            )
            self._set_preview_df(pd.DataFrame())

            request = ActionRequest(
                dataset_id=dataset_id,
                params=params,
                origin="core.table_tools.transform_panel",
            )

            if manager is not None:
                result = manager.run_action(
                    "core.table_tools.create_subset",
                    self.context,
                    request,
                    return_processed=True,
                )
                value = getattr(result, "value", None) or {}
            else:
                action_result = create_subset_action(self.context, request)
                value = action_result.value or {}

            new_dataset_id = value.get("dataset_id")
            subset_name = value.get("name", params["subset_name"])
            row_count = value.get("rows")
            materialized = bool(value.get("materialized", False))
            backend = value.get("backend") or "unknown"

            if new_dataset_id:
                self.status.object = (
                    f"Subset dataset `{subset_name}` registered as `{new_dataset_id}`. "
                    f"Loading preview..."
                )

                self._set_preview_df(
                    _dataset_head(
                        self.context,
                        new_dataset_id,
                        n=50,
                    )
                )
            else:
                self._set_preview_df(pd.DataFrame())

            self._refresh_metadata_panes()

            row_text = "unknown" if row_count is None else f"{int(row_count):,}"
            mode_text = "materialised Parquet file" if materialized else "lazy filtered Parquet view"

            self.status.object = (
                f"Created subset dataset `{subset_name}` with **{row_text}** rows "
                f"as a **{mode_text}** (`{backend}`)."
            )

        except Exception as exc:
            self.status.object = f"Create subset dataset failed: `{exc}`"

        finally:
            self.create_subset_button.disabled = False

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

def _debug_source_chain(source: Any) -> str:
    parts = []
    seen = set()
    current = source

    while current is not None:
        obj_id = id(current)
        backend = getattr(current, "backend_name", type(current).__name__)
        parts.append(f"{backend}@{obj_id:x}")

        if obj_id in seen:
            parts.append("CYCLE")
            break

        seen.add(obj_id)
        current = getattr(current, "base_source", None)

    return " -> ".join(parts)

def _request_dataset_id(context, request: Optional[ActionRequest] = None) -> str:
    if request is not None and request.dataset_id:
        return str(request.dataset_id)
    return _active_dataset_id(context)


def _active_dataset_id(context) -> str:
    datasets = getattr(context, "datasets", None)
    if datasets is not None:
        try:
            active = datasets.active_id()
            if active:
                return str(active)
        except Exception:
            pass

        try:
            ids = list(datasets.list_ids())
            if ids:
                return str(ids[0])
        except Exception:
            pass

    return "default"

def _active_source(context, dataset_id: Optional[str] = None):
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None

    get_source = getattr(datasets, "get_source", None)
    if callable(get_source):
        return get_source(dataset_id)

    return None


def _active_df(
    context,
    dataset_id: Optional[str] = None,
    *,
    limit: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compatibility-only materialisation helper.

    Important: this refuses to materialise non-pandas datasets without a limit.
    That prevents accidental full Parquet -> pandas loads.
    """
    source = _active_source(context, dataset_id)

    if source is not None:
        backend = getattr(source, "backend_name", "unknown")

        if limit is None and backend != "pandas":
            raise RuntimeError(
                "Refusing to materialise the full non-pandas dataset. "
                "Use DatasetSource/DuckDB paths or pass a small limit."
            )

        return source.to_pandas(columns=columns, limit=limit)

    config = getattr(context, "config", None)
    if config is not None and getattr(config, "main_df", None) is not None:
        df = config.main_df

        if columns is not None:
            existing = [col for col in columns if col in df.columns]
            df = df.loc[:, existing]

        if limit is not None:
            df = df.head(int(limit))

        return df.copy()

    return pd.DataFrame()


def _dataset_columns(context, dataset_id: Optional[str] = None) -> list[str]:
    datasets = getattr(context, "datasets", None)
    if datasets is not None:
        list_columns = getattr(datasets, "list_columns", None)
        if callable(list_columns):
            try:
                return [str(col) for col in list_columns(dataset_id)]
            except Exception:
                pass

    source = _active_source(context, dataset_id)
    if source is not None:
        try:
            return [str(col) for col in source.columns()]
        except Exception:
            pass

    config = getattr(context, "config", None)
    df = getattr(config, "main_df", None) if config is not None else None
    if isinstance(df, pd.DataFrame):
        return [str(col) for col in df.columns]

    return []


def _dataset_row_count(context, dataset_id: Optional[str] = None) -> Optional[int]:
    datasets = getattr(context, "datasets", None)
    if datasets is not None:
        row_count = getattr(datasets, "row_count", None)
        if callable(row_count):
            try:
                value = row_count(dataset_id)
                if value is not None:
                    return int(value)
            except Exception:
                pass

    source = _active_source(context, dataset_id)
    if source is not None:
        try:
            value = source.row_count()
            if value is not None:
                return int(value)
        except Exception:
            pass

    config = getattr(context, "config", None)
    df = getattr(config, "main_df", None) if config is not None else None
    if isinstance(df, pd.DataFrame):
        return int(len(df))

    return None


def _dataset_head(
    context,
    dataset_id: Optional[str] = None,
    *,
    n: int = 5,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    datasets = getattr(context, "datasets", None)
    if datasets is not None:
        head = getattr(datasets, "head", None)
        if callable(head):
            try:
                return head(dataset_id, n=n, columns=columns)
            except Exception:
                pass

    source = _active_source(context, dataset_id)
    if source is not None:
        return source.head(n=n, columns=columns)

    return _active_df(context, dataset_id, limit=n, columns=columns)


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


# def _sync_config_main_df(
#     context,
#     df: Optional[pd.DataFrame] = None,
#     *,
#     dataset_id: Optional[str] = None,
# ) -> None:
#     """
#     Legacy bridge only.

#     For source-backed datasets, do not push the full table into config.main_df.
#     Store a zero-row schema frame instead so old code can still inspect columns
#     without forcing a full materialisation.
#     """
#     config = getattr(context, "config", None)
#     if config is None:
#         return

#     try:
#         if df is not None:
#             config.main_df = df
#             return

#         if dataset_id is not None:
#             config.main_df = _dataset_head(context, dataset_id, n=0)
#     except Exception:
#         pass


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _quote_sql_string(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _is_identifier_like(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(value)))


def _expression_to_sql(expr: str, columns: Sequence[str]) -> str:
    """
    Convert the small pandas-style expression subset used by Table Tools into
    DuckDB SQL.

    Supported well:
    - arithmetic: a + b, a - b, a * b, a / b
    - comparisons: >, >=, <, <=, ==, !=
    - boolean combinations: &, |, ~
    - backtick column references: `column with spaces`

    Not supported here:
    - np.* / pd.* expression calls
    - arbitrary Python functions
    """
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Expression is empty.")

    if re.search(r"\b(?:np|pd)\.", expr):
        raise ValueError(
            "np.* and pd.* expressions are not supported by the DuckDB-backed "
            "large-dataset path. Use simple arithmetic/comparison expressions, "
            "or run this on an in-memory pandas dataset."
        )

    column_set = {str(col) for col in columns}

    def replace_backtick(match: re.Match) -> str:
        column_name = match.group(1)
        if column_name not in column_set:
            raise ValueError(f"Unknown column in expression: `{column_name}`")
        return _quote_identifier(column_name)

    sql = re.sub(r"`([^`]+)`", replace_backtick, expr)

    # Quote identifier-like column names. Longer names first prevents partial
    # replacement where one column name is a prefix of another.
    identifier_columns = sorted(
        [col for col in column_set if _is_identifier_like(col)],
        key=len,
        reverse=True,
    )

    for column in identifier_columns:
        sql = re.sub(
            rf'(?<!["\w.]){re.escape(column)}(?!["\w])',
            _quote_identifier(column),
            sql,
        )

    # Pandas-style boolean/comparison operators to SQL.
    sql = re.sub(r"(?<![<>=!])==(?![=])", "=", sql)
    sql = re.sub(r"\s*&\s*", " AND ", sql)
    sql = re.sub(r"\s*\|\s*", " OR ", sql)
    sql = re.sub(r"~\s*", "NOT ", sql)

    return sql

def _is_duckdb_relation_source(source: Any) -> bool:
    """
    True for any source that can expose a DuckDB relation over Parquet-backed
    data.

    This includes:
    - plain DuckDBParquetDatasetSource
    - lazy filtered Parquet views
    - lazy derived-column Parquet views
    """
    if source is None:
        return False

    return (
        callable(getattr(source, "_connect", None))
        and callable(getattr(source, "_relation_sql", None))
        and callable(getattr(source, "_path_argument", None))
    )


def _duckdb_relation_params(source: Any) -> list[Any]:

    if source is None:
        return []

    seen: set[int] = set()

    def _walk(src: Any) -> list[Any]:
        if src is None:
            return []

        obj_id = id(src)
        if obj_id in seen:
            raise RuntimeError(
                "Cycle detected in DuckDB dataset source chain while building "
                "SQL parameters. A lazy source probably has base_source pointing "
                "to itself or to one of its descendants."
            )

        seen.add(obj_id)

        base_source = getattr(src, "base_source", None)

        if base_source is not None:
            params = _walk(base_source)
            where_params = getattr(src, "where_params", None)
            if where_params:
                params.extend(list(where_params))
            seen.remove(obj_id)
            return params

        path_argument = getattr(src, "_path_argument", None)
        if callable(path_argument):
            seen.remove(obj_id)
            return [path_argument()]

        seen.remove(obj_id)
        return []

    return _walk(source)

def _duckdb_relation_query_sql(source: Any) -> str:
    """Return a SELECT query for a DuckDB relation source."""
    seen: set[int] = set()

    def _query(src: Any) -> str:
        obj_id = id(src)
        if obj_id in seen:
            raise RuntimeError(
                "Cycle detected in DuckDB dataset source chain while building "
                "relation SQL. A lazy source probably has base_source pointing "
                "to itself or to one of its descendants."
            )

        seen.add(obj_id)

        relation_sql = str(src._relation_sql()).strip()
        lower = relation_sql.lower()

        seen.remove(obj_id)

        if lower.startswith("select ") or lower.startswith("with "):
            return relation_sql

        return f"SELECT * FROM {relation_sql}"

    return _query(source)

def _duckdb_rows_by_ids_from_relation(
    source: Any,
    row_ids: Sequence[Any],
    *,
    id_column: str,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    ids = [str(row_id) for row_id in (row_ids or [])]
    if not ids:
        return pd.DataFrame(columns=list(columns or []))

    if id_column == "Use Index" or id_column not in source.columns():
        return pd.DataFrame(columns=list(columns or source.columns()))

    seen: set[str] = set()
    ordered_ids: list[str] = []

    for row_id in ids:
        if row_id in seen:
            continue
        seen.add(row_id)
        ordered_ids.append(row_id)

    selected_columns = list(columns or [])
    if id_column not in selected_columns:
        selected_columns.insert(0, id_column)

    available_columns = set(source.columns())
    selected_columns = [col for col in selected_columns if col in available_columns]

    if not selected_columns:
        selected_columns = [id_column]

    select_sql = ", ".join(
        f"src.{_quote_identifier(col)}" for col in selected_columns
    )

    values_sql = ", ".join(["(?, ?)"] * len(ordered_ids))

    values_params: list[Any] = []
    for order, row_id in enumerate(ordered_ids):
        values_params.extend([order, row_id])

    sql = (
        "WITH requested(__astronomical_lookup_order, __astronomical_lookup_id) AS "
        f"(VALUES {values_sql}) "
        f"SELECT {select_sql} "
        f"FROM {_duckdb_relation_from_sql(source, alias='src')} "
        "JOIN requested "
        f"ON CAST(src.{_quote_identifier(id_column)} AS VARCHAR) = requested.__astronomical_lookup_id "
        "ORDER BY requested.__astronomical_lookup_order"
    )

    con = source._connect()
    try:
        return con.execute(
            sql,
            [*values_params, *_duckdb_relation_params(source)],
        ).df()
    finally:
        con.close()

def _duckdb_relation_from_sql(source: Any, *, alias: str = "base") -> str:
    """Return a safe FROM-clause target for a DuckDB relation source."""
    return f"({_duckdb_relation_query_sql(source)}) AS {_quote_identifier(alias)}"

def _duckdb_self_from_sql(source: Any, *, alias: str = "src") -> str:
    """Return a FROM target for a lazy source's own SELECT query.

    Use this inside LazyDerivedColumnDuckDBSource and
    FilteredDuckDBParquetDatasetSource methods.

    Do not call _duckdb_relation_from_sql(self) inside those classes because
    that asks the helper to rediscover self._relation_sql(), which can recurse.
    """
    return f"({source._relation_sql()}) AS {_quote_identifier(alias)}"

def _is_duckdb_parquet_source(source: Any) -> bool:
    if source is None:
        return False

    if getattr(source, "backend_name", None) != "duckdb_parquet":
        return False

    return (
        callable(getattr(source, "_connect", None))
        and callable(getattr(source, "_relation_sql", None))
        and callable(getattr(source, "_path_argument", None))
    )

class LazyDerivedColumnDuckDBSource:
    """
    Lazy derived-column view over a DuckDB/Parquet-compatible source.

    This adds a computed column without physically rewriting the underlying
    Parquet file. Panels see the new column through the DatasetSource API.
    """

    backend_name = "duckdb_parquet_derived_column"

    def __init__(
        self,
        *,
        base_source: Any,
        new_column: str,
        expression_sql: str,
        expression_original: str,
        dataset_name: Optional[str] = None,
        columns_hint: Optional[Sequence[str]] = None,
        row_count_hint: Optional[int] = None,
    ) -> None:
        self.base_source = base_source
        self.new_column = str(new_column)
        self.expression_sql = str(expression_sql)
        self.expression_original = str(expression_original)
        self.dataset_name = dataset_name

        base_columns = [str(col) for col in (columns_hint or [])]

        if self.new_column not in base_columns:
            base_columns.append(self.new_column)

        self._column_cache = base_columns
        self._row_count_cache = (
            int(row_count_hint)
            if row_count_hint is not None
            else None
        )

    def _connect(self):
        return self.base_source._connect()

    def _path_argument(self):
        return self.base_source._path_argument()

    def _relation_params(self) -> list[Any]:
        return _duckdb_relation_params(self.base_source)

    def _base_relation_sql(self) -> str:
        return self.base_source._relation_sql()

    def _relation_sql(self) -> str:
        return (
            "SELECT base.*, "
            f"{self.expression_sql} AS {_quote_identifier(self.new_column)} "
            f"FROM {_duckdb_relation_from_sql(self.base_source, alias='base')}"
        )

    def _select_sql(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> str:
        if columns is None:
            return "*"

        selected = [str(col) for col in columns]

        if not selected:
            return "*"

        return ", ".join(_quote_identifier(col) for col in selected)

    def columns(self) -> list[str]:
        return list(self._column_cache)

    def dtypes(self) -> dict[str, str]:
        con = self._connect()
        try:
            df = con.execute(
                f"DESCRIBE SELECT * FROM {_duckdb_self_from_sql(self, alias='src')} LIMIT 0",
                self._relation_params(),
            ).df()
        finally:
            con.close()

        return {
            str(row["column_name"]): str(row["column_type"])
            for _, row in df.iterrows()
        }

    def row_count(self) -> Optional[int]:
        if self._row_count_cache is not None:
            return int(self._row_count_cache)

        con = self._connect()
        try:
            result = con.execute(
                f"SELECT COUNT(*) FROM {_duckdb_self_from_sql(self, alias='src')}",
                self._relation_params(),
            ).fetchone()
        finally:
            con.close()

        self._row_count_cache = int(result[0]) if result is not None else 0
        return self._row_count_cache

    def to_pandas(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        sql = (
            f"SELECT {self._select_sql(columns=columns)} "
            f"FROM {_duckdb_self_from_sql(self, alias='src')}"
        )
        sql_params = self._relation_params()

        if where_sql:
            sql += f" WHERE ({where_sql})"
            if params:
                sql_params.extend(list(params))

        if limit is not None:
            sql += " LIMIT ?"
            sql_params.append(int(limit))

        con = self._connect()
        try:
            return con.execute(sql, sql_params).df()
        finally:
            con.close()

    def head(
        self,
        n: int = 5,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return self.to_pandas(columns=columns, limit=n)

    def get_rows_by_ids(
        self,
        row_ids: Sequence[Any],
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return _duckdb_rows_by_ids_from_relation(
            self,
            row_ids,
            id_column=id_column,
            columns=columns,
        )

    def get_row_by_position(
        self,
        position: int,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if position < 0:
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        sql = (
            f"SELECT {self._select_sql(columns=columns)} "
            f"FROM {_duckdb_self_from_sql(self, alias='src')} "
            "LIMIT 1 OFFSET ?"
        )

        con = self._connect()
        try:
            return con.execute(
                sql,
                [*self._relation_params(), int(position)],
            ).df()
        finally:
            con.close()

    def get_row_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if id_column == "Use Index" or id_column not in self.columns():
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        return self.to_pandas(
            columns=columns,
            limit=1,
            where_sql=f"CAST({_quote_identifier(id_column)} AS VARCHAR) = ?",
            params=[str(row_id)],
        )

    def find_position_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
    ) -> Optional[int]:
        if id_column == "Use Index" or id_column not in self.columns():
            return None

        sql = (
            "SELECT rn FROM ("
            " SELECT "
            f" ROW_NUMBER() OVER () - 1 AS rn, "
            f" {_quote_identifier(id_column)} AS rid "
            f" FROM {_duckdb_self_from_sql(self, alias='src')}"
            ") AS numbered "
            "WHERE CAST(rid AS VARCHAR) = ? "
            "LIMIT 1"
        )

        con = self._connect()
        try:
            result = con.execute(
                sql,
                [*self._relation_params(), str(row_id)],
            ).fetchone()
        finally:
            con.close()

        if result is None:
            return None

        return int(result[0])

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "base_backend": getattr(self.base_source, "backend_name", "unknown"),
            "dataset_name": self.dataset_name,
            "new_column": self.new_column,
            "expression": self.expression_original,
            "materialized": False,
        }

class FilteredDuckDBParquetDatasetSource:
    """
    Lazy filtered view over a DuckDB/Parquet source.

    This represents a subset dataset without physically copying rows to a new
    Parquet file. Queries are pushed down to DuckDB and only materialised when a
    panel asks for a preview, row lookup, selected columns, etc.
    """

    backend_name = "duckdb_parquet_filtered"

    def __init__(
        self,
        *,
        base_source: Any,
        where_sql: str,
        where_params: Optional[Sequence[Any]] = None,
        dataset_name: Optional[str] = None,
        columns_hint: Optional[Sequence[str]] = None,
        row_count_hint: Optional[int] = None,
    ) -> None:
        self.base_source = base_source
        self.where_sql = str(where_sql)
        self.where_params = list(where_params or [])
        self.dataset_name = dataset_name

        self._column_cache = (
            [str(col) for col in columns_hint]
            if columns_hint is not None
            else None
        )
        self._row_count_cache = (
            int(row_count_hint)
            if row_count_hint is not None
            else None
        )

    def _connect(self):
        return self.base_source._connect()

    def _path_argument(self):
        return self.base_source._path_argument()

    def _base_relation_sql(self) -> str:
        return self.base_source._relation_sql()

    def _relation_sql(self) -> str:
        return (
            "SELECT base.* "
            f"FROM {_duckdb_relation_from_sql(self.base_source, alias='base')} "
            f"WHERE ({self.where_sql})"
        )

    def _select_sql(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> str:
        selected_columns = None if columns is None else [str(col) for col in columns]

        if selected_columns is None or not selected_columns:
            return "*"

        return ", ".join(_quote_identifier(col) for col in selected_columns)

    def _relation_params(self) -> list[Any]:
        params = _duckdb_relation_params(self.base_source)
        params.extend(self.where_params)
        return params


    def _params(self, extra: Optional[Sequence[Any]] = None) -> list[Any]:
        params = self._relation_params()
        if extra:
            params.extend(list(extra))
        return params

    def columns(self) -> list[str]:
        if self._column_cache is not None:
            return list(self._column_cache)

        try:
            self._column_cache = [str(col) for col in self.base_source.columns()]
            return list(self._column_cache)
        except Exception:
            pass

        con = self._connect()
        try:
            df = con.execute(
                f"DESCRIBE SELECT * FROM {_duckdb_self_from_sql(self, alias="src")} LIMIT 0",
                self._relation_params(),
            ).df()
        finally:
            con.close()

        self._column_cache = [str(col) for col in df["column_name"].tolist()]
        return list(self._column_cache)

    def dtypes(self) -> dict[str, str]:
        con = self._connect()
        try:
            df = con.execute(
                f"DESCRIBE SELECT * FROM {_duckdb_self_from_sql(self, alias="src")} LIMIT 0",
                self._relation_params(),
            ).df()
        finally:
            con.close()

        return {
            str(row["column_name"]): str(row["column_type"])
            for _, row in df.iterrows()
        }

    def row_count(self) -> Optional[int]:
        if self._row_count_cache is not None:
            return int(self._row_count_cache)

        con = self._connect()
        try:
            result = con.execute(
                f"SELECT COUNT(*) FROM {_duckdb_self_from_sql(self, alias="src")}",
                self._relation_params(),
            ).fetchone()
        finally:
            con.close()

        self._row_count_cache = int(result[0]) if result is not None else 0
        return self._row_count_cache

    def to_pandas(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        select_sql = self._select_sql(columns=columns)

        sql = (
            f"SELECT {select_sql} "
            f"FROM {_duckdb_self_from_sql(self, alias="src")}"
        )
        sql_params = self._relation_params()

        if where_sql:
            sql += f" WHERE ({where_sql})"
            if params:
                sql_params.extend(list(params))

        if limit is not None:
            sql += " LIMIT ?"
            sql_params.append(int(limit))

        con = self._connect()
        try:
            return con.execute(sql, sql_params).df()
        finally:
            con.close()

    def head(
        self,
        n: int = 5,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return self.to_pandas(columns=columns, limit=n)

    def get_row_by_position(
        self,
        position: int,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if position < 0:
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        select_sql = self._select_sql(columns=columns)

        sql = (
            f"SELECT {select_sql} "
            f"FROM {_duckdb_self_from_sql(self, alias="src")} "
            "LIMIT 1 OFFSET ?"
        )

        con = self._connect()
        try:
            return con.execute(
                sql,
                [*self._relation_params(), int(position)],
            ).df()
        finally:
            con.close()

    def get_rows_by_ids(
        self,
        row_ids: Sequence[Any],
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return _duckdb_rows_by_ids_from_relation(
            self,
            row_ids,
            id_column=id_column,
            columns=columns,
        )

    def get_row_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if id_column == "Use Index":
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        if id_column not in self.columns():
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        return self.to_pandas(
            columns=columns,
            limit=1,
            where_sql=f"CAST({_quote_identifier(id_column)} AS VARCHAR) = ?",
            params=[str(row_id)],
        )

    def find_position_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
    ) -> Optional[int]:
        if id_column == "Use Index" or id_column not in self.columns():
            return None

        sql = (
            "SELECT rn FROM ("
            " SELECT "
            f" ROW_NUMBER() OVER () - 1 AS rn, "
            f" {_quote_identifier(id_column)} AS rid "
            f" FROM {_duckdb_self_from_sql(self, alias="src")}"
            ") AS numbered "
            "WHERE CAST(rid AS VARCHAR) = ? "
            "LIMIT 1"
        )

        con = self._connect()
        try:
            result = con.execute(
                sql,
                [*self._relation_params(), str(row_id)],
            ).fetchone()
        finally:
            con.close()

        if result is None:
            return None

        return int(result[0])

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "base_backend": getattr(self.base_source, "backend_name", "unknown"),
            "dataset_name": self.dataset_name,
            "filter_sql": self.where_sql,
            "row_count": self._row_count_cache,
        }

def _configure_duckdb_for_large_copy(con: Any) -> None:
    """
    Keep DuckDB COPY/SELECT operations from building avoidable large in-memory
    structures.

    These pragmas are best-effort. Older DuckDB versions may not support all of
    them, so failures are intentionally ignored.
    """
    for statement in (
        "PRAGMA preserve_insertion_order=false",
        "PRAGMA threads=4",
    ):
        try:
            con.execute(statement)
        except Exception:
            pass

def _duckdb_query_df(
    source: Any,
    sql: str,
    params: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    con = source._connect()
    try:
        return con.execute(sql, list(params or [])).df()
    finally:
        con.close()


def _duckdb_fetchone(
    source: Any,
    sql: str,
    params: Optional[Sequence[Any]] = None,
):
    con = source._connect()
    try:
        return con.execute(sql, list(params or [])).fetchone()
    finally:
        con.close()


def _duckdb_copy_query_to_parquet(
    source: Any,
    *,
    select_sql: str,
    params: Sequence[Any],
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    copy_sql = (
        f"COPY ({select_sql}) "
        f"TO {_quote_sql_string(output_path)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    con = source._connect()
    try:
        _configure_duckdb_for_large_copy(con)
        con.execute(copy_sql, list(params))
    finally:
        con.close()

def _duckdb_copy_query_to_parquet_on_connection(
    con: Any,
    *,
    select_sql: str,
    params: Sequence[Any],
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    copy_sql = (
        f"COPY ({select_sql}) "
        f"TO {_quote_sql_string(output_path)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    con.execute(copy_sql, list(params))

def _new_cache_path(context, dataset_id: str, suffix: str) -> Path:
    cache_dir = default_cache_dir_for_context(context)
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_id = normalise_dataset_id(f"{dataset_id}__{suffix}")
    return cache_dir / f"{safe_id}.parquet"


def _register_parquet_dataset(
    context,
    *,
    dataset_id: str,
    parquet_path: Path,
    name: Optional[str],
    columns: Sequence[str],
    row_count: Optional[int],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise RuntimeError("DatasetManager is required.")

    register_parquet = getattr(datasets, "register_parquet", None)
    if not callable(register_parquet):
        raise RuntimeError(
            "DatasetManager.register_parquet is required for source-backed table transforms."
        )

    registration_meta = dict(meta or {})
    registration_meta.setdefault("backend", "duckdb_parquet")
    registration_meta.setdefault("source_format", "parquet")
    registration_meta.setdefault("source_path", str(parquet_path))
    registration_meta.setdefault("columns", [str(col) for col in columns])

    if row_count is not None:
        registration_meta.setdefault("row_count", int(row_count))
        registration_meta.setdefault("rows", int(row_count))

    register_parquet(
        dataset_id,
        parquet_path,
        name=name or dataset_id,
        **registration_meta,
    )

def _register_lazy_filtered_subset(
    context,
    *,
    source: Any,
    base_dataset_id: str,
    new_dataset_id: str,
    subset_name: str,
    expression: str,
    where_sql: str,
    row_count: Optional[int] = None,
) -> int:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise RuntimeError("DatasetManager is required.")

    columns = _dataset_columns(context, base_dataset_id)
    base_meta = _active_dataset_meta(context, base_dataset_id)

    filtered_source = FilteredDuckDBParquetDatasetSource(
        base_source=source,
        where_sql=where_sql,
        where_params=[],
        dataset_name=subset_name,
        columns_hint=columns,
        row_count_hint=row_count,
    )

    meta = dict(base_meta)
    meta.update(
        {
            "backend": "duckdb_parquet_filtered",
            "source_format": "parquet_filtered_view",
            "derived_from": base_dataset_id,
            "filter": expression,
            "filter_sql": where_sql,
            "created_by": manifest.id,
            "derivation_type": "table_subset",
            "materialized": False,
            "columns": columns,
        }
    )

    if row_count is not None:
        meta["row_count"] = int(row_count)
        meta["rows"] = int(row_count)

    datasets.register_source(
        new_dataset_id,
        filtered_source,
        name=subset_name,
        **meta,
    )

    return int(row_count) if row_count is not None else -1

def _add_column_duckdb_parquet(
    context,
    *,
    dataset_id: str,
    source: Any,
    new_column: str,
    expression: str,
) -> int:
    """Register a lazy derived-column view over any DuckDB relation source."""
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise RuntimeError("DatasetManager is required.")

    if not _is_duckdb_relation_source(source):
        raise RuntimeError(
            "Cannot add a lazy DuckDB column to this dataset because it does "
            "not expose a DuckDB relation."
        )

    columns = _dataset_columns(context, dataset_id)

    if new_column in columns:
        raise ValueError(f"Column `{new_column}` already exists.")

    expr_sql = _expression_to_sql(expression, columns)
    from_sql = _duckdb_relation_from_sql(source, alias="base")

    preview = _duckdb_query_df(
        source,
        (
            f"SELECT {expr_sql} AS {_quote_identifier(new_column)} "
            f"FROM {from_sql} "
            "LIMIT 1"
        ),
        _duckdb_relation_params(source),
    )

    if new_column not in preview.columns:
        raise RuntimeError(f"Could not validate derived column `{new_column}`.")

    row_count = _dataset_row_count(context, dataset_id)
    dataset_name = _active_dataset_name(context, dataset_id)
    meta = _active_dataset_meta(context, dataset_id)

    new_columns = [*columns, new_column]

    derived_source = LazyDerivedColumnDuckDBSource(
        base_source=source,
        new_column=new_column,
        expression_sql=expr_sql,
        expression_original=expression,
        dataset_name=dataset_name,
        columns_hint=new_columns,
        row_count_hint=row_count,
    )

    if derived_source.base_source is derived_source:
        raise RuntimeError(
            "Internal error: lazy derived column source was created with itself "
            "as base_source."
        )

    meta.update(
        {
            "backend": derived_source.backend_name,
            "source_format": "parquet_derived_column_view",
            "columns": new_columns,
            "row_count": row_count,
            "rows": row_count,
            "last_transform_type": "add_column",
            "last_transform_column": new_column,
            "last_transform_expr": expression,
            "created_by": manifest.id,
            "materialized": False,
        }
    )

    datasets.register_source(
        dataset_id,
        derived_source,
        name=dataset_name,
        **meta,
    )

    return int(row_count) if row_count is not None else 0

def _add_column_pandas_fallback(
    context,
    *,
    dataset_id: str,
    new_column: str,
    expression: str,
) -> int:
    """
    Compatibility path for datasets that are already pandas-backed.

    Non-pandas sources must not fall back here because that can materialise
    millions of rows.
    """
    source = _active_source(context, dataset_id)
    backend = getattr(source, "backend_name", None)

    if backend != "pandas":
        raise RuntimeError(
            "Refusing to add a column by materialising a non-pandas dataset. "
            f"Dataset `{dataset_id}` has backend `{backend}`."
        )

    df = _require_pandas_dataframe(context, dataset_id).copy()

    if new_column in df.columns:
        raise ValueError(f"Column `{new_column}` already exists.")

    result = _evaluate_expression(expression, df)
    df[new_column] = result

    cache_dir = default_cache_dir_for_context(context)
    dataset_name = _active_dataset_name(context, dataset_id)
    meta = _active_dataset_meta(context, dataset_id)

    meta.update(
        {
            "last_transform_type": "add_column",
            "last_transform_column": new_column,
            "last_transform_expr": expression,
            "created_by": manifest.id,
            "materialized": True,
        }
    )

    replace_dataset_with_dataframe_parquet(
        context,
        dataset_id=dataset_id,
        df=df,
        name=dataset_name,
        cache_dir=cache_dir,
        **meta,
    )

    return int(len(df))

def _create_subset_duckdb_parquet(
    context,
    *,
    source: Any,
    base_dataset_id: str,
    new_dataset_id: str,
    subset_name: str,
    expression: str,
) -> int:
    """Register a lazy filtered subset over any DuckDB relation source.

    Works for:
    - base duckdb_parquet sources
    - duckdb_parquet_filtered sources
    - duckdb_parquet_derived_column sources

    The important part is that source._relation_sql() may be `read_parquet(?)`.
    That cannot be wrapped directly as `(read_parquet(?))`, so use
    _duckdb_relation_from_sql().
    """
    if not _is_duckdb_relation_source(source):
        raise RuntimeError(
            "Cannot create a DuckDB-backed subset because this dataset does "
            "not expose a DuckDB relation."
        )

    columns = _dataset_columns(context, base_dataset_id)
    where_sql = _expression_to_sql(expression, columns)

    print(
        "[TableTools] create subset source chain:",
        _debug_source_chain(source),
        flush=True,
    )

    from_sql = _duckdb_relation_from_sql(source, alias="base")
    relation_params = _duckdb_relation_params(source)

    count_result = _duckdb_fetchone(
        source,
        f"SELECT COUNT(*) FROM {from_sql} WHERE ({where_sql})",
        relation_params,
    )
    matched_count = int(count_result[0]) if count_result is not None else 0

    return _register_lazy_filtered_subset(
        context,
        source=source,
        base_dataset_id=base_dataset_id,
        new_dataset_id=new_dataset_id,
        subset_name=subset_name,
        expression=expression,
        where_sql=where_sql,
        row_count=matched_count,
    )


def _create_subset_pandas_fallback(
    context,
    *,
    base_dataset_id: str,
    new_dataset_id: str,
    subset_name: str,
    expression: str,
) -> int:
    """
    Compatibility path only.

    This is allowed for datasets whose canonical source is already pandas.
    It must not be used as a fallback for source-backed large datasets.
    """
    source = _active_source(context, base_dataset_id)
    backend = getattr(source, "backend_name", None)

    if backend != "pandas":
        raise RuntimeError(
            "Refusing to create a subset by materialising a non-pandas "
            f"dataset through pandas fallback. Dataset `{base_dataset_id}` "
            f"has backend `{backend}`. Convert/register it as a DuckDB/Parquet "
            "source first, or add a backend-specific subset implementation."
        )

    base_df = _require_pandas_dataframe(context, base_dataset_id)
    mask = _evaluate_boolean_expression(expression, base_df)
    filtered = base_df.loc[mask].copy()

    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise RuntimeError("DatasetManager is required.")

    cache_dir = default_cache_dir_for_context(context)

    register_dataframe_as_parquet(
        datasets,
        dataset_id=new_dataset_id,
        df=filtered,
        name=subset_name,
        cache_dir=cache_dir,
        overwrite=True,
        derived_from=base_dataset_id,
        filter=expression,
        created_by=manifest.id,
        derivation_type="table_subset",
    )

    return int(len(filtered))

def _preview_column_expression(
    context,
    *,
    dataset_id: str,
    expression: str,
    limit: int = 20,
) -> pd.DataFrame:
    source = _active_source(context, dataset_id)

    if _is_duckdb_relation_source(source):
        columns = _dataset_columns(context, dataset_id)
        expr_sql = _expression_to_sql(expression, columns)
        from_sql = _duckdb_relation_from_sql(source, alias="base")

        return _duckdb_query_df(
            source,
            (
                f"SELECT {expr_sql} AS {_quote_identifier('__preview_result__')} "
                f"FROM {from_sql} "
                "LIMIT ?"
            ),
            [*_duckdb_relation_params(source), int(limit)],
        )

    df = _active_df(context, dataset_id, limit=limit)
    result = _evaluate_expression(expression, df)
    return pd.DataFrame({"__preview_result__": result.head(limit).values})

def _preview_subset_expression(
    context,
    *,
    dataset_id: str,
    expression: str,
    limit: int = 50,
) -> tuple[pd.DataFrame, int, Optional[int]]:
    source = _active_source(context, dataset_id)

    if _is_duckdb_relation_source(source):
        columns = _dataset_columns(context, dataset_id)
        where_sql = _expression_to_sql(expression, columns)
        from_sql = _duckdb_relation_from_sql(source, alias="base")
        relation_params = _duckdb_relation_params(source)

        print(
            "[TableTools] preview source chain:",
            _debug_source_chain(source),
            flush=True,
        )

        count_sql = (
            f"SELECT COUNT(*) "
            f"FROM {from_sql} "
            f"WHERE ({where_sql})"
        )

        preview_sql = (
            f"SELECT * "
            f"FROM {from_sql} "
            f"WHERE ({where_sql}) "
            "LIMIT ?"
        )

        con = source._connect()
        try:
            _configure_duckdb_for_large_copy(con)

            count_result = con.execute(
                count_sql,
                relation_params,
            ).fetchone()

            matched_count = int(count_result[0]) if count_result is not None else 0

            preview_df = con.execute(
                preview_sql,
                [*relation_params, int(limit)],
            ).df()

        finally:
            con.close()

        total_count = _dataset_row_count(context, dataset_id)
        return preview_df, matched_count, total_count

    source_backend = getattr(source, "backend_name", None)

    if source_backend != "pandas":
        raise RuntimeError(
            "Refusing to preview subset by materialising a non-pandas dataset. "
            f"Dataset `{dataset_id}` has backend `{source_backend}`."
        )

    df = _require_pandas_dataframe(context, dataset_id)
    mask = _evaluate_boolean_expression(expression, df)
    filtered = df.loc[mask]

    return filtered.head(limit).copy(), int(len(filtered)), int(len(df))


def _require_pandas_dataframe(context, dataset_id: Optional[str] = None) -> pd.DataFrame:
    source = _active_source(context, dataset_id)

    if source is not None:
        backend = getattr(source, "backend_name", None)

        if backend == "pandas":
            df = getattr(source, "df", None)
            if isinstance(df, pd.DataFrame):
                return df

        raise RuntimeError(
            "This operation requires an already in-memory pandas dataset. "
            f"Dataset `{dataset_id or _active_dataset_id(context)}` has backend "
            f"`{backend}`. Refusing to materialise it through pandas fallback."
        )

    config = getattr(context, "config", None)
    df = getattr(config, "main_df", None) if config is not None else None

    if isinstance(df, pd.DataFrame):
        return df

    raise RuntimeError(
        "This operation requires either a DuckDB/Parquet-backed dataset or an "
        "already in-memory pandas dataset. Refusing to materialise the full "
        "dataset through get_df()."
    )


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

def _dataset_get_mappings(datasets, dataset_id: str) -> Dict[str, str]:
    try:
        get_mappings = getattr(datasets, "get_mappings", None)
        if callable(get_mappings):
            mappings = get_mappings(dataset_id)
            if isinstance(mappings, dict):
                return dict(mappings)
    except Exception:
        pass

    try:
        get = getattr(datasets, "get", None)
        if callable(get):
            record = get(dataset_id)

            mappings = getattr(record, "mappings", None)
            if isinstance(mappings, dict):
                return dict(mappings)

            metadata = getattr(record, "metadata", None)
            if isinstance(metadata, dict):
                for key in ("mappings", "column_mappings", "semantic_mappings"):
                    if isinstance(metadata.get(key), dict):
                        return dict(metadata[key])
    except Exception:
        pass

    return {}


def _copy_dataset_mappings(
    datasets,
    *,
    source_dataset_id: str,
    target_dataset_id: str,
) -> None:
    mappings = _dataset_get_mappings(datasets, source_dataset_id)
    if not mappings:
        return

    set_mapping = getattr(datasets, "set_mapping", None)
    if callable(set_mapping):
        for semantic_name, column_name in mappings.items():
            try:
                set_mapping(target_dataset_id, semantic_name, column_name)
            except Exception:
                pass