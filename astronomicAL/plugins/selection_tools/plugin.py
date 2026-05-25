from __future__ import annotations

import html
import traceback
import uuid
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import panel as pn

from astronomicAL.platform.dataset_sources import DatasetSource
from astronomicAL.platform.plugins import PluginManifest
from astronomicAL.platform.plugins.specs import ActionResult, EventResult, InputSpec


SELECTION_PREVIEW_LIMIT = 50
SELECTION_PREVIEW_EXTRA_COLUMNS = 4

manifest = PluginManifest(
    id="core.selection_tools",
    name="Selection Tools",
    version="0.1.0",
    description=(
        "Generic selection inspection tools for focused rows and active "
        "multi-row selection sets."
    ),
    capabilities=["panel", "selection", "datasets", "events", "actions"],
    tags=["core", "selection", "workflow", "review", "dataset"],
)


def register(api) -> None:
    api.register_panel(
        id="selection_set",
        title="Selection Set",
        factory=create_selection_set_panel,
        description=(
            "Inspect the active selection set, preview selected rows, clear the "
            "set, move focus through selected row IDs, and create a derived "
            "dataset from the active selection."
        ),
        category="Selection",
        icon="list-checks",
        tags=["selection", "focus", "dataset", "review"],
        optional_mappings=[
            {
                "semantic_name": "record_id",
                "display_name": "ID column",
                "description": (
                    "Optional. Used to match platform selection row IDs back to "
                    "dataframe rows. If unmapped, Selection Tools falls back to "
                    "the dataframe index."
                ),
                "aliases": [
                    "source_id",
                    "sourceid",
                    "object_id",
                    "objid",
                    "id",
                    "ID",
                    "row_id",
                ],
                "allow_index": True,
            }
        ],
        default_layout={"x": 0, "y": 0, "w": 6, "h": 5},
    )

    api.register_action(
        id="selection_to_dataset",
        title="Create dataset from active selection",
        handler=selection_to_dataset_action,
        inputs=InputSpec(
            dataset=True,
            selection="optional",
            columns="none",
            optional_mappings=[
                {
                    "semantic_name": "record_id",
                    "display_name": "ID column",
                    "description": (
                        "Optional. Used to match selection row IDs to dataframe rows. "
                        "If unmapped, the dataframe index is used."
                    ),
                    "aliases": [
                        "source_id",
                        "sourceid",
                        "object_id",
                        "objid",
                        "id",
                        "ID",
                        "row_id",
                    ],
                    "allow_index": True,
                }
            ],
        ),
        outputs=["dataset.loaded", "dataset.active.changed", "selection.dataset.created"],
        params_schema={
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "set_active": {"type": "boolean", "default": True},
            },
            "required": ["dataset_name"],
            "additionalProperties": False,
        },
        run_in_job=False,
        description="Register the active selection set as a new derived dataset.",
        category="Selection",
        tags=["selection", "dataset", "subset"],
    )


def create_selection_set_panel(context, data=None, **kwargs):
    controller = SelectionSetPanel(context=context, data=data)
    return controller.view, controller


def selection_to_dataset_action(context, request, **_kwargs) -> ActionResult:
    dataset_name = str(request.params.get("dataset_name", "")).strip()
    set_active = bool(request.params.get("set_active", True))

    if not dataset_name:
        raise ValueError("Please provide a dataset name.")

    result = materialise_active_selection_as_dataset(
        context,
        dataset_name=dataset_name,
        set_active=set_active,
        publish_events=False,
    )

    events = [
        EventResult(
            "dataset.loaded",
            {
                "dataset_id": result["dataset_id"],
                "name": result["name"],
                "rows": result["rows"],
                "derived_from": result["derived_from"],
                "selection_set_id": result["selection_set_id"],
                "origin": manifest.id,
            },
        ),
        EventResult(
            "selection.dataset.created",
            {
                "dataset_id": result["dataset_id"],
                "derived_from": result["derived_from"],
                "selection_set_id": result["selection_set_id"],
                "rows": result["rows"],
                "origin": manifest.id,
            },
        ),
    ]

    if set_active:
        events.append(
            EventResult(
                "dataset.active.changed",
                {
                    "dataset_id": result["dataset_id"],
                    "previous_dataset_id": result["derived_from"],
                    "origin": manifest.id,
                },
            )
        )

    return ActionResult(value=result, events=events)


def materialise_active_selection_as_dataset(
    context,
    *,
    dataset_name: str,
    set_active: bool = True,
    publish_events: bool = True,
) -> Dict[str, Any]:
    selection = getattr(context, "selection", None)
    datasets = getattr(context, "datasets", None)

    if selection is None:
        raise RuntimeError("SelectionManager is required.")
    if datasets is None:
        raise RuntimeError("DatasetManager is required.")

    state = _get_active_selection_state(selection)
    if state is None:
        raise ValueError("There is no active selection set.")

    row_ids = [str(r) for r in list(getattr(state, "row_ids", []) or [])]
    if not row_ids:
        raise ValueError("The active selection set is empty.")

    base_dataset_id = getattr(state, "dataset_id", None) or _active_dataset_id(datasets)
    if not base_dataset_id:
        raise RuntimeError("Could not resolve the source dataset.")

    columns = _dataset_columns(datasets, base_dataset_id)
    id_col = _resolve_id_column_from_columns(context, base_dataset_id, columns)

    if id_col is None:
        raise ValueError(
            "Could not create a derived dataset from this selection because no "
            "record ID column is mapped. Map `record_id` for the source dataset "
            "or use a selection whose row IDs are integer row positions."
        )

    new_dataset_id = f"{base_dataset_id}__selection__{uuid.uuid4().hex[:8]}"
    selection_set_id = getattr(state, "selection_set_id", None)
    metadata = dict(getattr(state, "metadata", {}) or {})
    base_mappings = _dataset_get_mappings(datasets, base_dataset_id)

    dataset_metadata = {
        "derived_from": base_dataset_id,
        "created_by": manifest.id,
        "derivation_type": "selection_set",
        "selection_set_id": selection_set_id,
        "selection_row_count": len(row_ids),
        "selection_row_ids_preview": row_ids[:SELECTION_PREVIEW_LIMIT],
        "selection_metadata": metadata,
        "rows": len(row_ids),
    }

    source = _dataset_get_source(datasets, base_dataset_id)

    if source is not None and hasattr(datasets, "register_source"):
        subset_source = SelectionSubsetDatasetSource(
            base_source=source,
            row_ids=row_ids,
            id_column=id_col,
            columns=columns,
        )
        _register_dataset_source_compat(
            datasets,
            dataset_id=new_dataset_id,
            source=subset_source,
            name=dataset_name,
            metadata=dataset_metadata,
            mappings=base_mappings,
        )
        row_count = len(row_ids)
    else:
        # Last-resort compatibility path for very old DatasetManager-like
        # objects. This materialises only the selected rows, never the whole
        # source dataset.
        filtered_df = _selection_rows_to_dataframe(
            context,
            base_dataset_id=base_dataset_id,
            row_ids=row_ids,
            columns=columns,
            id_col=id_col,
        )
        if filtered_df.empty:
            raise ValueError(
                "The active selection did not match any rows in the source dataset."
            )

        _register_dataset_compat(
            datasets,
            dataset_id=new_dataset_id,
            df=filtered_df,
            name=dataset_name,
            metadata=dataset_metadata,
            mappings=base_mappings,
        )
        row_count = len(filtered_df)

    previous_dataset_id = base_dataset_id

    if set_active:
        _set_active_dataset_compat(datasets, new_dataset_id)

    result = {
        "dataset_id": new_dataset_id,
        "name": dataset_name,
        "rows": row_count,
        "derived_from": base_dataset_id,
        "selection_set_id": selection_set_id,
        "set_active": set_active,
    }

    if publish_events:
        _publish_event(
            context,
            "dataset.loaded",
            {
                "dataset_id": new_dataset_id,
                "name": dataset_name,
                "rows": row_count,
                "derived_from": base_dataset_id,
                "selection_set_id": selection_set_id,
                "origin": manifest.id,
            },
        )

        if set_active:
            _publish_event(
                context,
                "dataset.active.changed",
                {
                    "dataset_id": new_dataset_id,
                    "previous_dataset_id": previous_dataset_id,
                    "origin": manifest.id,
                },
            )

        _publish_event(
            context,
            "selection.dataset.created",
            {
                "dataset_id": new_dataset_id,
                "derived_from": base_dataset_id,
                "selection_set_id": selection_set_id,
                "rows": row_count,
                "origin": manifest.id,
            },
        )

    return result


def _get_active_selection_state(selection):
    try:
        get_active_set = getattr(selection, "get_active_set", None)
        if callable(get_active_set):
            return get_active_set()
    except Exception:
        traceback.print_exc()

    try:
        return getattr(selection, "active_set", None)
    except Exception:
        return None


def _active_dataset_id(datasets) -> Optional[str]:
    try:
        active_id = getattr(datasets, "active_id", None)
        if callable(active_id):
            return active_id()
        if active_id:
            return str(active_id)
    except Exception:
        pass

    try:
        active_dataset_id = getattr(datasets, "active_dataset_id", None)
        if active_dataset_id:
            return str(active_dataset_id)
    except Exception:
        pass

    return None


def _dataset_get_source(datasets, dataset_id: str):
    get_source = getattr(datasets, "get_source", None)
    if callable(get_source):
        try:
            return get_source(dataset_id)
        except Exception:
            pass

    get = getattr(datasets, "get", None)
    if callable(get):
        record = get(dataset_id)

        source = getattr(record, "source", None)
        if source is not None:
            return source

        df = getattr(record, "df", None)
        if isinstance(df, pd.DataFrame):
            return None

        data = getattr(record, "data", None)
        if isinstance(data, pd.DataFrame):
            return None

    return None

def _dataset_get_mappings(datasets, dataset_id: str) -> Dict[str, str]:
    try:
        get_mappings = getattr(datasets, "get_mappings", None)
        if callable(get_mappings):
            try:
                mappings = get_mappings(dataset_id)
            except TypeError:
                mappings = get_mappings()

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


def _register_dataset_compat(
    datasets,
    *,
    dataset_id: str,
    df: pd.DataFrame,
    name: str,
    metadata: Dict[str, Any],
    mappings: Dict[str, str],
) -> None:
    register = getattr(datasets, "register", None)
    if not callable(register):
        raise RuntimeError("DatasetManager does not expose register(...).")

    attempts = [
        lambda: register(
            dataset_id,
            df,
            name=name,
            metadata=metadata,
            mappings=mappings,
            column_mappings=mappings,
        ),
        lambda: register(
            dataset_id,
            df,
            name=name,
            metadata=metadata,
            column_mappings=mappings,
        ),
        lambda: register(
            dataset_id,
            df,
            name=name,
            metadata=metadata,
            mappings=mappings,
        ),
        lambda: register(
            dataset_id,
            df,
            name=name,
            metadata=metadata,
        ),
        lambda: register(
            dataset_id,
            df,
            metadata=metadata,
        ),
        lambda: register(
            dataset_id,
            df,
            name=name,
            derived_from=metadata.get("derived_from"),
            created_by=metadata.get("created_by"),
            derivation_type=metadata.get("derivation_type"),
            selection_set_id=metadata.get("selection_set_id"),
            selection_row_ids=metadata.get("selection_row_ids"),
            selection_metadata=metadata.get("selection_metadata"),
            column_mappings=mappings,
        ),
        lambda: register(dataset_id, df),
    ]

    last_exc: Optional[Exception] = None

    for attempt in attempts:
        try:
            attempt()
            _try_set_dataset_metadata(datasets, dataset_id, metadata)
            _try_set_dataset_mappings(datasets, dataset_id, mappings)
            return
        except TypeError as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("Could not register derived selection dataset.")


def _try_set_dataset_metadata(
    datasets,
    dataset_id: str,
    metadata: Dict[str, Any],
) -> None:
    for method_name in ("set_metadata", "update_metadata"):
        try:
            method = getattr(datasets, method_name, None)
            if callable(method):
                method(dataset_id, metadata)
                return
        except Exception:
            pass

    try:
        get = getattr(datasets, "get", None)
        if callable(get):
            record = get(dataset_id)

            record_meta = getattr(record, "meta", None)
            if isinstance(record_meta, dict):
                record_meta.update(metadata)
                return

            record_metadata = getattr(record, "metadata", None)
            if isinstance(record_metadata, dict):
                record_metadata.update(metadata)
                return
    except Exception:
        pass


def _try_set_dataset_mappings(
    datasets,
    dataset_id: str,
    mappings: Dict[str, str],
) -> None:
    if not mappings:
        return

    for method_name in ("set_mappings", "update_mappings"):
        try:
            method = getattr(datasets, method_name, None)
            if callable(method):
                method(dataset_id, mappings)
                return
        except Exception:
            pass

    try:
        for semantic_name, column_name in mappings.items():
            set_mapping = getattr(datasets, "set_mapping", None)
            if callable(set_mapping):
                set_mapping(dataset_id, semantic_name, column_name)
    except Exception:
        pass


def _set_active_dataset_compat(datasets, dataset_id: str) -> None:
    for method_name in ("set_active", "set_active_dataset"):
        try:
            method = getattr(datasets, method_name, None)
            if callable(method):
                method(dataset_id)
                return
        except Exception:
            pass

def _publish_event(context, topic: str, payload: Dict[str, Any]) -> None:
    events = getattr(context, "events", None)
    if events is None:
        return

    try:
        publish = getattr(events, "publish", None)
        if callable(publish):
            publish(topic, payload)
            return
    except Exception:
        traceback.print_exc()

    try:
        emit = getattr(events, "emit", None)
        if callable(emit):
            emit(topic, payload)
            return
    except Exception:
        traceback.print_exc()


class SelectionSubsetDatasetSource(DatasetSource):
    """
    Lazy source representing a selection-set subset of another dataset.

    It does not materialise the base dataset. It fetches rows from the base
    source by record ID or row position as needed.
    """

    backend_name = "selection_subset"

    def __init__(
        self,
        *,
        base_source: DatasetSource,
        row_ids: Sequence[str],
        id_column: str,
        columns: Sequence[str],
    ) -> None:
        self.base_source = base_source
        self.row_ids = [str(row_id) for row_id in row_ids]
        self.id_column = id_column
        self._columns = [str(column) for column in columns]

    def columns(self) -> List[str]:
        return list(self._columns)

    def row_count(self) -> int:
        return len(self.row_ids)

    def dtypes(self) -> Dict[str, str]:
        try:
            base_dtypes = self.base_source.dtypes()
            return {
                str(column): str(base_dtypes.get(column, "object"))
                for column in self._columns
            }
        except Exception:
            return {str(column): "object" for column in self._columns}

    def to_pandas(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        if where_sql is not None:
            raise NotImplementedError(
                "SelectionSubsetDatasetSource does not support SQL filtering yet."
            )

        selected_ids = self.row_ids[: int(limit)] if limit is not None else self.row_ids
        return _rows_from_source_by_selection_ids(
            self.base_source,
            row_ids=selected_ids,
            id_col=self.id_column,
            columns=columns or self._columns,
        )

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
        if position < 0 or position >= len(self.row_ids):
            return pd.DataFrame(columns=list(columns or self._columns))

        return _rows_from_source_by_selection_ids(
            self.base_source,
            row_ids=[self.row_ids[position]],
            id_col=self.id_column,
            columns=columns or self._columns,
        )

    def get_row_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        row_id = str(row_id)

        if row_id not in set(self.row_ids):
            return pd.DataFrame(columns=list(columns or self._columns))

        return _rows_from_source_by_selection_ids(
            self.base_source,
            row_ids=[row_id],
            id_col=self.id_column,
            columns=columns or self._columns,
        )

    def find_position_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
    ) -> Optional[int]:
        row_id = str(row_id)
        try:
            return self.row_ids.index(row_id)
        except ValueError:
            return None

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "base_backend": getattr(self.base_source, "backend_name", "unknown"),
            "rows": len(self.row_ids),
            "id_column": self.id_column,
        }


def _dataset_get_source(datasets, dataset_id: str):
    try:
        get_source = getattr(datasets, "get_source", None)
        if callable(get_source):
            return get_source(dataset_id)
    except Exception:
        traceback.print_exc()

    try:
        get = getattr(datasets, "get", None)
        if callable(get):
            record = get(dataset_id)
            source = getattr(record, "source", None)
            if source is not None:
                return source
    except Exception:
        traceback.print_exc()

    return None


def _dataset_columns(datasets, dataset_id: str) -> List[str]:
    try:
        list_columns = getattr(datasets, "list_columns", None)
        if callable(list_columns):
            return [str(column) for column in list_columns(dataset_id)]
    except Exception:
        traceback.print_exc()

    source = _dataset_get_source(datasets, dataset_id)
    if source is not None:
        try:
            return [str(column) for column in source.columns()]
        except Exception:
            traceback.print_exc()

    return []


def _resolve_id_column_from_columns(
    context,
    dataset_id: str,
    columns: Sequence[str],
) -> Optional[str]:
    column_set = {str(column) for column in columns}
    original = {str(column): str(column) for column in columns}
    mappings = _dataset_get_mappings(getattr(context, "datasets", None), dataset_id)

    for semantic_name in ("record_id", "id", "id_col", "row_id"):
        mapped = mappings.get(semantic_name)

        if mapped in ("Use Index", "__index__", "index"):
            return "Use Index"

        if isinstance(mapped, str) and mapped in column_set:
            return original[mapped]

    config = getattr(context, "config", None)
    if config is not None:
        try:
            settings = getattr(config, "settings", None)
            if isinstance(settings, dict):
                configured = settings.get("id_col")
                if configured in ("Use Index", "__index__", "index"):
                    return "Use Index"
                if isinstance(configured, str) and configured in column_set:
                    return original[configured]
        except Exception:
            pass

    for candidate in (
        "source_id",
        "sourceid",
        "object_id",
        "objid",
        "id",
        "ID",
        "row_id",
    ):
        if candidate in column_set:
            return original[candidate]

    # If all selection row IDs are integer-like, callers can still use
    # row-position access with id_col="Use Index". We do not infer that here
    # because this helper does not know the row IDs.
    return None


def _resolve_label_column_from_columns(
    context,
    dataset_id: str,
    columns: Sequence[str],
) -> Optional[str]:
    column_set = {str(column) for column in columns}
    mappings = _dataset_get_mappings(getattr(context, "datasets", None), dataset_id)

    for semantic_name in ("target_label", "label", "label_col", "class"):
        mapped = mappings.get(semantic_name)
        if isinstance(mapped, str) and mapped in column_set:
            return mapped

    config = getattr(context, "config", None)
    if config is not None:
        try:
            settings = getattr(config, "settings", None)
            if isinstance(settings, dict):
                configured = settings.get("label_col")
                if isinstance(configured, str) and configured in column_set:
                    return configured
        except Exception:
            pass

    for candidate in ("label", "class", "target", "label_col"):
        if candidate in column_set:
            return candidate

    return None


def _safe_preview_columns(
    *,
    columns: Sequence[str],
    id_col: Optional[str],
    label_col: Optional[str],
) -> List[str]:
    wanted: List[str] = []

    for column in (id_col, label_col):
        if column and column != "Use Index" and column in columns and column not in wanted:
            wanted.append(column)

    for column in columns:
        if column not in wanted:
            wanted.append(column)
        if len(wanted) >= SELECTION_PREVIEW_EXTRA_COLUMNS + 2:
            break

    return wanted


def _rows_from_source_by_selection_ids(
    source,
    *,
    row_ids: Sequence[str],
    id_col: Optional[str],
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    selected_columns = None if columns is None else [str(column) for column in columns]

    for row_id in row_ids:
        row_id = str(row_id)

        try:
            if id_col == "Use Index":
                try:
                    position = int(row_id)
                except Exception:
                    continue

                row = source.get_row_by_position(position, columns=selected_columns)
            elif id_col:
                row = source.get_row_by_id(
                    row_id,
                    id_column=id_col,
                    columns=selected_columns,
                )
            else:
                continue

            if row is None or row.empty:
                continue

            row = row.head(1).copy()
            row["_selection_row_id"] = row_id
            rows.append(row)
        except Exception:
            traceback.print_exc()
            continue

    if not rows:
        return pd.DataFrame(columns=list(selected_columns or []))

    return pd.concat(rows, ignore_index=True, sort=False)


def _selection_rows_to_dataframe(
    context,
    *,
    base_dataset_id: str,
    row_ids: Sequence[str],
    columns: Optional[Sequence[str]] = None,
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise RuntimeError("DatasetManager is required.")

    source = _dataset_get_source(datasets, base_dataset_id)
    if source is None:
        raise RuntimeError(f"Could not read source for dataset `{base_dataset_id}`.")

    if columns is None:
        columns = _dataset_columns(datasets, base_dataset_id)

    if id_col is None:
        id_col = _resolve_id_column_from_columns(context, base_dataset_id, columns)

    if id_col is None:
        raise ValueError(
            "Could not resolve an ID column for the selected rows. "
            "Map `record_id` for the source dataset."
        )

    return _rows_from_source_by_selection_ids(
        source,
        row_ids=row_ids,
        id_col=id_col,
        columns=columns,
    )


def _register_dataset_source_compat(
    datasets,
    *,
    dataset_id: str,
    source,
    name: str,
    metadata: Dict[str, Any],
    mappings: Dict[str, str],
) -> None:
    register_source = getattr(datasets, "register_source", None)
    if not callable(register_source):
        raise RuntimeError("DatasetManager does not expose register_source(...).")

    meta = dict(metadata or {})
    meta.pop("name", None)

    meta["column_mappings"] = dict(mappings or {})

    meta.setdefault("mappings", dict(mappings or {}))

    register_source(
        dataset_id,
        source,
        name=name,
        **meta,
    )

    _try_set_dataset_metadata(datasets, dataset_id, meta)
    _try_set_dataset_mappings(datasets, dataset_id, mappings)


class SelectionSetPanel:
    """Plugin version of the legacy SelectionSetPanel.

    This panel intentionally does not subclass CustomPlotClass. It uses the
    platform services directly:

    - context.selection for focus and active selection-set state
    - context.events for refresh notifications
    - context.datasets for previewing selected rows and creating derived datasets
    - context.config only as a temporary compatibility fallback for id/label cols
    """

    PANEL_ID = "core.selection_tools.selection_set"

    def __init__(self, context, data=None):
        self.context = context
        self.data = data
        self.panel_id = self.PANEL_ID

        self._disposed = False
        self._subscriptions: List[Any] = []
        self._refresh_pending = False
        self._last_signature = None

        self.summary_pane = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        self.preview_table = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            height=220,
            margin=(0, 0, 0, 0),
            styles={
                "overflow-y": "auto",
                "overflow-x": "auto",
                "border": "1px solid #ddd",
                "border-radius": "6px",
                "background": "white",
                "padding": "0",
            },
        )

        self.prev_button = pn.widgets.Button(
            name="Focus prev",
            button_type="default",
            width=96,
            height=30,
            margin=(0, 4, 0, 0),
        )
        self.prev_button.on_click(self._focus_prev_cb)

        self.next_button = pn.widgets.Button(
            name="Focus next",
            button_type="primary",
            width=96,
            height=30,
            margin=(0, 4, 0, 0),
        )
        self.next_button.on_click(self._focus_next_cb)

        self.clear_button = pn.widgets.Button(
            name="Clear set",
            button_type="danger",
            width=88,
            height=30,
            margin=(0, 8, 0, 0),
        )
        self.clear_button.on_click(self._clear_selection_cb)

        self.dataset_name = pn.widgets.TextInput(
            name="Dataset name",
            placeholder="e.g. selected_sources",
            sizing_mode="stretch_width",
            height=32,
            margin=(0, 6, 0, 0),
        )

        self.set_active_dataset = pn.widgets.Checkbox(
            name="Set active",
            value=True,
            width=90,
            height=30,
            margin=(4, 8, 0, 0),
        )

        self.create_dataset_button = pn.widgets.Button(
            name="Create Dataset",
            button_type="success",
            width=128,
            height=30,
            margin=(0, 0, 0, 0),
        )
        self.create_dataset_button.on_click(self._create_dataset_cb)

        self.status_pane = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            margin=(2, 0, 4, 0),
        )

        self.view = self._build_view()

        self._subscribe_to_runtime_events()
        self.refresh(force=True)

    # ------------------------------------------------------------------
    # Dashboard/plugin compatibility
    # ------------------------------------------------------------------

    def get_toolbar(self):
        """Tell Dashboard not to inject its fixed Close toolbar.

        The ReactGrid tile already has its own close affordance, and the
        injected toolbar creates a large blank area above this plugin panel.
        """
        return pn.Spacer(height=1, min_height=1, max_height=1)

    def panel(self):
        return self.view

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_view(self):
        title = pn.pane.Markdown(
            "### Selection Set",
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
        )

        selection_controls = pn.Row(
            self.prev_button,
            self.next_button,
            self.clear_button,
            sizing_mode="stretch_width",
            height=34,
            margin=(0, 0, 6, 0),
        )

        dataset_controls = pn.Column(
            pn.Row(
                self.dataset_name,
                self.set_active_dataset,
                self.create_dataset_button,
                sizing_mode="stretch_width",
                height=36,
                margin=(0, 0, 2, 0),
                align="end",
            ),
            self.status_pane,
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
            styles={
                "border": "1px solid #ddd",
                "border-radius": "6px",
                "background": "#fafafa",
                "padding": "6px 8px",
            },
        )

        return pn.Column(
            title,
            self.summary_pane,
            selection_controls,
            dataset_controls,
            self.preview_table,
            sizing_mode="stretch_both",
            min_height=0,
            margin=(0, 0, 0, 0),
            styles={
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "padding": "4px 8px 8px 8px",
            },
        )

    # ------------------------------------------------------------------
    # Platform state helpers
    # ------------------------------------------------------------------

    @property
    def selection(self):
        return getattr(self.context, "selection", None)

    @property
    def datasets(self):
        return getattr(self.context, "datasets", None)

    @property
    def events(self):
        return getattr(self.context, "events", None)

    @property
    def config(self):
        return getattr(self.context, "config", None)

    def _get_active_selection_set_state(self):
        if self.selection is None:
            return None

        return _get_active_selection_state(self.selection)

    def _get_focus_state(self):
        if self.selection is None:
            return None

        try:
            get_focus = getattr(self.selection, "get_focus", None)
            if callable(get_focus):
                return get_focus()
        except Exception:
            traceback.print_exc()

        try:
            return getattr(self.selection, "focus", None)
        except Exception:
            return None

    def _get_focus_row_id(self) -> Optional[str]:
        focus = self._get_focus_state()
        if focus is None:
            return None

        row_id = getattr(focus, "row_id", None)
        return None if row_id is None else str(row_id)

    def _active_dataset_id(self) -> str:
        if self.datasets is not None:
            active = _active_dataset_id(self.datasets)
            if active:
                return str(active)

        return "default"

    def _get_dataset_df(self, dataset_id: Optional[str] = None) -> Optional[pd.DataFrame]:

        dataset_id = dataset_id or self._active_dataset_id()

        if self.datasets is not None:
            try:
                columns = _dataset_columns(self.datasets, dataset_id)
                return self.datasets.get_df(
                    dataset_id,
                    columns=columns[: min(len(columns), 8)],
                    limit=SELECTION_PREVIEW_LIMIT,
                )
            except Exception:
                pass

        if self.data is not None and isinstance(self.data, pd.DataFrame):
            return self.data.head(SELECTION_PREVIEW_LIMIT).copy()

        return None

    def _config_setting(self, key: str, default: Any = None) -> Any:
        if self.config is None:
            return default

        try:
            settings = getattr(self.config, "settings", None)
            if isinstance(settings, dict):
                return settings.get(key, default)
        except Exception:
            pass

        return default

    def _get_dataset_mappings(self, dataset_id: Optional[str] = None) -> Dict[str, str]:
        if self.datasets is not None:
            return _dataset_get_mappings(
                self.datasets,
                dataset_id or self._active_dataset_id(),
            )

        return {}

    def _resolve_column_name(
        self,
        requirement: str,
        *,
        df: Optional[pd.DataFrame] = None,
        dataset_id: Optional[str] = None,
        allow_direct: bool = True,
    ) -> Optional[str]:
        dataset_id = dataset_id or self._active_dataset_id()

        if df is not None:
            columns = [str(column) for column in getattr(df, "columns", [])]
        elif self.datasets is not None:
            columns = _dataset_columns(self.datasets, dataset_id)
        else:
            columns = []

        if not columns:
            return None

        column_set = set(columns)
        mappings = self._get_dataset_mappings(dataset_id)

        if allow_direct and requirement in column_set:
            return requirement

        mapped = mappings.get(requirement)
        if mapped in ("Use Index", "__index__", "index"):
            return "Use Index"
        if isinstance(mapped, str) and mapped in column_set:
            return mapped

        aliases = {
            "id": ["record_id", "id", "ids", "source_id", "object_id", "id_col", "row_id"],
            "id_col": ["record_id", "id_col", "id", "ids", "source_id", "object_id", "row_id"],
            "record_id": ["record_id", "id", "ids", "source_id", "object_id", "id_col", "row_id"],
            "label": ["target_label", "label", "class", "target", "label_col"],
            "label_col": ["target_label", "label_col", "label", "class", "target"],
            "target_label": ["target_label", "label", "class", "target", "label_col"],
        }

        for alias in aliases.get(requirement, []):
            mapped = mappings.get(alias)
            if mapped in ("Use Index", "__index__", "index"):
                return "Use Index"
            if isinstance(mapped, str) and mapped in column_set:
                return mapped
            if alias in column_set:
                return alias

        return None

    # ------------------------------------------------------------------
    # Refresh/event handling
    # ------------------------------------------------------------------

    def _state_signature(self):
        state = self._get_active_selection_set_state()
        focus = self._get_focus_state()

        return (
            self._active_dataset_id(),
            getattr(state, "selection_set_id", None) if state is not None else None,
            getattr(state, "dataset_id", None) if state is not None else None,
            tuple(str(r) for r in (getattr(state, "row_ids", []) or []))
            if state is not None
            else (),
            getattr(focus, "dataset_id", None) if focus is not None else None,
            getattr(focus, "row_id", None) if focus is not None else None,
        )

    def _subscribe_to_runtime_events(self) -> None:
        for topic in (
            "selection.focus.changed",
            "selection.focus.cleared",
            "selection.set.changed",
            "selection.set.cleared",
            "dataset.active.changed",
            "dataset.updated",
            "dataset.mapping_updated",
            "dataset.loaded",
        ):
            self._subscribe(topic, self._runtime_event_cb)

    def _subscribe(self, topic: str, callback) -> None:
        if self.events is None:
            return

        try:
            sub = self.events.subscribe(
                topic,
                callback,
                owner_id=self.panel_id,
                owner_label="Selection Set",
                owner_kind="plugin-panel",
            )
        except TypeError:
            sub = self.events.subscribe(topic, callback)

        self._subscriptions.append(sub)

    def _runtime_event_cb(self, topic: str, payload: Any) -> None:
        self.schedule_refresh()

    def schedule_refresh(self) -> None:
        if self._disposed or self._refresh_pending:
            return

        self._refresh_pending = True

        def _runner():
            if self._disposed:
                return
            self._refresh_pending = False
            self.refresh()

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(_runner)
            else:
                _runner()
        except Exception:
            _runner()

    def refresh(self, *, force: bool = False) -> None:
        if self._disposed:
            return

        signature = self._state_signature()
        if not force and signature == self._last_signature:
            return

        self._last_signature = signature
        self._refresh_panel_state()

    def _refresh_panel_state(self) -> None:
        state = self._get_active_selection_set_state()
        focus = self._get_focus_state()

        if state is None:
            self.summary_pane.object = self._summary_html(
                "<strong>No active selection set.</strong>"
            )
            self.preview_table.object = self._preview_df_to_html(
                pd.DataFrame(columns=["focus", "row_id"])
            )
            self.prev_button.disabled = True
            self.next_button.disabled = True
            self.clear_button.disabled = True
            self.create_dataset_button.disabled = True
            return

        row_ids = [str(r) for r in list(getattr(state, "row_ids", []) or [])]
        dataset_id = getattr(state, "dataset_id", None) or self._active_dataset_id()
        selection_set_id = getattr(state, "selection_set_id", None) or "—"
        focus_row_id = getattr(focus, "row_id", None) if focus is not None else None
        geometry_text = self._get_geometry_text()

        self.summary_pane.object = self._summary_html(
            f"""
            <div style="display:grid;grid-template-columns:120px 1fr;gap:2px 8px;">
                <strong>Dataset:</strong>
                <code>{html.escape(str(dataset_id))}</code>

                <strong>Selection set:</strong>
                <code>{html.escape(str(selection_set_id))}</code>

                <strong>Selected sources:</strong>
                <span>{len(row_ids)}</span>

                <strong>Focused source:</strong>
                <span>{html.escape(str(focus_row_id)) if focus_row_id is not None else "—"}</span>

                <strong>Geometry:</strong>
                <span style="white-space:pre-line;">{html.escape(geometry_text)}</span>
            </div>
            """
        )

        self.preview_table.object = self._preview_df_to_html(self._get_preview_df())

        has_rows = len(row_ids) > 0
        self.prev_button.disabled = not has_rows
        self.next_button.disabled = not has_rows
        self.clear_button.disabled = not has_rows
        self.create_dataset_button.disabled = not has_rows

        if has_rows and not (self.dataset_name.value or "").strip():
            safe_set_id = str(selection_set_id).replace(" ", "_").replace("/", "_")
            if safe_set_id and safe_set_id != "—":
                self.dataset_name.value = f"{safe_set_id}_dataset"
            else:
                self.dataset_name.value = "selection_dataset"

    # ------------------------------------------------------------------
    # Data preview
    # ------------------------------------------------------------------

    def _get_preview_df(self) -> pd.DataFrame:
        state = self._get_active_selection_set_state()
        if state is None:
            return pd.DataFrame(columns=["focus", "row_id"])

        dataset_id = getattr(state, "dataset_id", None) or self._active_dataset_id()
        row_ids = [str(r) for r in list(getattr(state, "row_ids", []) or [])]
        focus_row_id = self._get_focus_row_id()

        if not row_ids:
            return pd.DataFrame(columns=["focus", "row_id"])

        if self.datasets is None:
            return self._fallback_preview(row_ids, focus_row_id)

        try:
            columns = _dataset_columns(self.datasets, dataset_id)
            id_col = _resolve_id_column_from_columns(self.context, dataset_id, columns)
            label_col = _resolve_label_column_from_columns(self.context, dataset_id, columns)

            preview_columns = _safe_preview_columns(
                columns=columns,
                id_col=id_col,
                label_col=label_col,
            )

            source = _dataset_get_source(self.datasets, dataset_id)
            if source is None:
                return self._fallback_preview(row_ids, focus_row_id)

            preview_row_ids = row_ids[:SELECTION_PREVIEW_LIMIT]

            preview = _rows_from_source_by_selection_ids(
                source,
                row_ids=preview_row_ids,
                id_col=id_col,
                columns=preview_columns,
            )

            if preview.empty:
                return self._fallback_preview(row_ids, focus_row_id)

            preview["row_id"] = preview["_selection_row_id"].astype(str)
            preview = preview.drop(columns=["_selection_row_id"], errors="ignore")

            preview.insert(
                0,
                "focus",
                [
                    "◀" if str(row_id) == str(focus_row_id) else ""
                    for row_id in preview["row_id"]
                ],
            )

            preferred_cols = ["focus", "row_id"]

            if (
                label_col not in (None, "", "No Labels", "Use Index")
                and label_col in preview.columns
                and label_col not in preferred_cols
            ):
                preferred_cols.append(label_col)

            for column in preview.columns:
                if column not in preferred_cols:
                    preferred_cols.append(column)

            return preview[preferred_cols].head(SELECTION_PREVIEW_LIMIT).reset_index(drop=True)
        except Exception:
            traceback.print_exc()
            return self._fallback_preview(row_ids, focus_row_id)

    @staticmethod
    def _fallback_preview(
        row_ids: List[str],
        focus_row_id: Optional[str],
    ) -> pd.DataFrame:
        preview = pd.DataFrame({"row_id": row_ids})
        preview.insert(
            0,
            "focus",
            ["◀" if str(row_id) == str(focus_row_id) else "" for row_id in row_ids],
        )
        return preview.head(50)

    def _get_geometry_text(self) -> str:
        state = self._get_active_selection_set_state()
        if state is None:
            return "—"

        metadata = getattr(state, "metadata", {}) or {}
        geometry = metadata.get("geometry", {}) or {}

        if geometry.get("kind") != "box":
            return "—"

        bounds = geometry.get("bounds")
        x_var = geometry.get("x_variable")
        y_var = geometry.get("y_variable")

        if not bounds or len(bounds) != 4:
            return "—"

        try:
            x0, x1, y0, y1 = bounds
            return f"{x_var}: [{x0:.4g}, {x1:.4g}]\n{y_var}: [{y0:.4g}, {y1:.4g}]"
        except Exception:
            return f"{x_var}/{y_var}"

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------

    def _clear_selection_cb(self, event=None) -> None:
        if self.selection is None:
            return

        try:
            self.selection.clear_selection_set(
                origin="selection.set.panel.clear",
                panel_id=self.panel_id,
            )
        except TypeError:
            try:
                self.selection.clear_selection_set(origin="selection.set.panel.clear")
            except TypeError:
                self.selection.clear_selection_set()

        self.schedule_refresh()

    def _focus_relative(self, delta: int) -> None:
        if self.selection is None:
            return

        state = self._get_active_selection_set_state()
        if state is None:
            return

        row_ids = [str(r) for r in list(getattr(state, "row_ids", []) or [])]
        if not row_ids:
            return

        focus_row_id = self._get_focus_row_id()

        if focus_row_id in row_ids:
            idx = row_ids.index(focus_row_id)
            idx = (idx + delta) % len(row_ids)
        else:
            idx = 0 if delta >= 0 else len(row_ids) - 1

        dataset_id = getattr(state, "dataset_id", None) or self._active_dataset_id()
        selection_set_id = getattr(state, "selection_set_id", None)

        try:
            self.selection.set_focus(
                dataset_id=dataset_id,
                row_id=row_ids[idx],
                origin="selection.set.panel.focus",
                panel_id=self.panel_id,
                selection_set_id=selection_set_id,
            )
        except TypeError:
            try:
                self.selection.set_focus(
                    dataset_id=dataset_id,
                    row_id=row_ids[idx],
                    origin="selection.set.panel.focus",
                )
            except TypeError:
                self.selection.set_focus(dataset_id, row_ids[idx])

        self.schedule_refresh()

    def _focus_prev_cb(self, event=None) -> None:
        self._focus_relative(-1)

    def _focus_next_cb(self, event=None) -> None:
        self._focus_relative(1)

    def _create_dataset_cb(self, event=None) -> None:
        dataset_name = (self.dataset_name.value or "").strip()
        set_active = bool(self.set_active_dataset.value)

        if not dataset_name:
            self.status_pane.object = (
                "**Could not create dataset:** please provide a dataset name."
            )
            return

        try:
            result = materialise_active_selection_as_dataset(
                self.context,
                dataset_name=dataset_name,
                set_active=set_active,
                publish_events=True,
            )

            active_msg = " and set as active" if set_active else ""
            self.status_pane.object = (
                f"Created dataset `{result['name']}` "
                f"(`{result['dataset_id']}`) with **{result['rows']} rows**"
                f"{active_msg}."
            )

        except Exception as exc:
            traceback.print_exc()
            self.status_pane.object = f"**Could not create dataset:** `{exc}`"

        self.schedule_refresh()

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summary_html(body: str) -> str:
        return f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 6px;
            background: #fafafa;
            padding: 6px 8px;
            line-height: 1.35;
            font-size: 13px;
        ">
            {body}
        </div>
        """

    def _preview_df_to_html(self, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return """
            <div style="padding: 10px; color: #666;">
                No rows in active selection set.
            </div>
            """

        header_cells = "".join(
            f"""
            <th style="
                position: sticky;
                top: 0;
                background: #f3f3f3;
                border-bottom: 1px solid #ccc;
                padding: 6px 8px;
                text-align: left;
                font-weight: 600;
                white-space: nowrap;
            ">
                {html.escape(str(col))}
            </th>
            """
            for col in df.columns
        )

        body_rows = []

        for _, row in df.iterrows():
            cells = "".join(
                f"""
                <td style="
                    border-bottom: 1px solid #eee;
                    padding: 5px 8px;
                    white-space: nowrap;
                ">
                    {"" if pd.isna(val) else html.escape(str(val))}
                </td>
                """
                for val in row.values
            )
            body_rows.append(f"<tr>{cells}</tr>")

        return f"""
        <table style="
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
            line-height: 1.35;
        ">
            <thead>
                <tr>{header_cells}</tr>
            </thead>
            <tbody>
                {''.join(body_rows)}
            </tbody>
        </table>
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        if self._disposed:
            return

        self._disposed = True
        self._refresh_pending = False

        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in list(self._subscriptions):
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass

        self._subscriptions.clear()