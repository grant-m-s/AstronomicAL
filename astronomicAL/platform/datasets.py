from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from astronomicAL.platform.dataset_sources import (
    DatasetSource,
    DuckDBParquetDatasetSource,
    PandasDatasetSource,
    coerce_dataset_source,
)
from astronomicAL.utils.debug import (
    compact_list,
    dataset_debug_print,
    mapping_debug_print,
    summarize_pending_restore_items,
)


@dataclass
class Dataset:
    dataset_id: str
    name: str
    source: DatasetSource
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def df(self) -> pd.DataFrame:
        """
        Compatibility shim for older code.

        New code should use Dataset.source or DatasetManager.get_source().
        """
        return self.source.to_pandas()

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        """
        Compatibility shim for old code that mutates dataset.df.

        This intentionally converts the dataset back to an in-memory pandas
        source. Avoid using this in new code.
        """
        self.source = PandasDatasetSource(value)


def _json_safe_metadata(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(_json_safe_metadata(key)): _json_safe_metadata(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe_metadata(item) for item in value]

    return str(value)


class DatasetManager:
    """
    Manages datasets.

    The canonical data payload is now DatasetSource, not pd.DataFrame.

    Compatibility:
        - register(..., df=...) still works.
        - get_df(...) still works but materializes a pandas view.
        - Dataset.df still works but should be treated as legacy.
    """

    def __init__(self) -> None:
        self._datasets: Dict[str, Dataset] = {}
        self._active_id: Optional[str] = None
        self._pending_restore_items = []
        self._pending_active_id = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        *,
        name: Optional[str] = None,
        **meta: Any,
    ) -> None:
        """
        Backward-compatible pandas registration.

        New code should prefer register_source() or register_parquet().
        """
        self.register_source(
            dataset_id,
            PandasDatasetSource(df),
            name=name,
            **meta,
        )

    def register_source(
        self,
        dataset_id: str,
        source: DatasetSource,
        *,
        name: Optional[str] = None,
        **meta: Any,
    ) -> None:
        if name is None:
            name = dataset_id

        meta = dict(meta)
        meta.setdefault("column_mappings", {})
        meta.setdefault("backend", getattr(source, "backend_name", "unknown"))

        self._datasets[dataset_id] = Dataset(
            dataset_id=dataset_id,
            name=name,
            source=source,
            meta=meta,
        )

        if self._active_id is None:
            self._active_id = dataset_id

        self._ensure_pending_restore_state()

        columns = source.columns()
        dataset_debug_print(
            "register_source",
            {
                "dataset_id": dataset_id,
                "backend": getattr(source, "backend_name", "unknown"),
                "row_count": source.row_count(),
                "column_count": len(columns),
                "columns_preview": compact_list([str(col) for col in columns]),
                "pending_restore_items": summarize_pending_restore_items(
                    self._pending_restore_items
                ),
            },
        )

        self._apply_pending_restore_to_dataset(dataset_id)

        dataset_debug_print(
            "register_source after pending restore",
            {
                "dataset_id": dataset_id,
                "mappings": self.get_mappings(dataset_id),
            },
        )

    def register_parquet(
        self,
        dataset_id: str,
        path: str | Path | Sequence[str | Path],
        *,
        name: Optional[str] = None,
        **meta: Any,
    ) -> None:
        """
        Register a Parquet dataset lazily through DuckDB.

        Metadata hints are important because UI refreshes should not repeatedly
        inspect/count large Parquet files.
        """
        columns_hint = None
        row_count_hint = None

        raw_columns = meta.get("columns")
        if isinstance(raw_columns, dict):
            columns_hint = list(raw_columns.keys())
        elif isinstance(raw_columns, (list, tuple)):
            columns_hint = [str(col) for col in raw_columns]

        for key in ("row_count", "rows", "n_rows"):
            value = meta.get(key)
            if value is not None:
                try:
                    row_count_hint = int(value)
                    break
                except Exception:
                    pass

        source = DuckDBParquetDatasetSource(
            path,
            dataset_name=name or dataset_id,
            columns_hint=columns_hint,
            row_count_hint=row_count_hint,
        )

        meta = dict(meta)
        meta.setdefault("source_format", "parquet")
        meta.setdefault("source_path", str(path))
        meta.setdefault("backend", "duckdb_parquet")

        if columns_hint is not None:
            meta.setdefault("columns", columns_hint)

        if row_count_hint is not None:
            meta.setdefault("row_count", row_count_hint)

        self.register_source(dataset_id, source, name=name, **meta)

    def ensure_registered(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        *,
        name: Optional[str] = None,
        **meta: Any,
    ) -> None:
        """
        Register the dataset if missing; otherwise update the existing dataset.

        Backward-compatible pandas method.
        """
        if dataset_id not in self._datasets:
            self.register(dataset_id, df, name=name, **meta)
            return

        self.ensure_source_registered(
            dataset_id,
            PandasDatasetSource(df),
            name=name,
            **meta,
        )

    def ensure_source_registered(
        self,
        dataset_id: str,
        source: DatasetSource,
        *,
        name: Optional[str] = None,
        **meta: Any,
    ) -> None:
        if dataset_id not in self._datasets:
            self.register_source(dataset_id, source, name=name, **meta)
            return

        ds = self._datasets[dataset_id]
        ds.source = coerce_dataset_source(source)

        if name is not None:
            ds.name = name

        if meta:
            ds.meta.update(meta)

        ds.meta.setdefault("column_mappings", {})
        ds.meta.setdefault("backend", getattr(ds.source, "backend_name", "unknown"))

    # ------------------------------------------------------------------
    # Basic access
    # ------------------------------------------------------------------

    def list_ids(self) -> list[str]:
        return list(self._datasets.keys())

    def active_id(self) -> str:
        if self._active_id is None:
            raise RuntimeError("No active dataset set.")
        return self._active_id

    def set_active(self, dataset_id: str) -> None:
        if dataset_id not in self._datasets:
            raise KeyError(f"Unknown dataset_id: {dataset_id}")

        self._active_id = dataset_id

        apply_pending = getattr(self, "_apply_pending_restore_to_dataset", None)
        if callable(apply_pending):
            apply_pending(dataset_id)

    def get(self, dataset_id: Optional[str] = None) -> Dataset:
        if dataset_id is None:
            dataset_id = self.active_id()

        if dataset_id not in self._datasets:
            raise KeyError(f"Unknown dataset_id: {dataset_id}")

        return self._datasets[dataset_id]

    def get_source(self, dataset_id: Optional[str] = None) -> DatasetSource:
        return self.get(dataset_id).source

    def get_df(
        self,
        dataset_id: Optional[str] = None,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        """
        Compatibility method.

        New code should prefer get_source(), list_columns(), row_count(),
        get_row_by_position(), or source.to_pandas(columns=[...], limit=...).
        """
        return self.get_source(dataset_id).to_pandas(
            columns=columns,
            limit=limit,
            where_sql=where_sql,
            params=params,
        )

    def get_meta(self, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        return self.get(dataset_id).meta

    def list_columns(self, dataset_id: Optional[str] = None) -> list[str]:
        return self.get_source(dataset_id).columns()

    def row_count(self, dataset_id: Optional[str] = None) -> Optional[int]:
        return self.get_source(dataset_id).row_count()

    def head(
        self,
        dataset_id: Optional[str] = None,
        n: int = 5,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return self.get_source(dataset_id).head(n=n, columns=columns)

    def dtypes(self, dataset_id: Optional[str] = None) -> dict[str, str]:
        source = self.get_source(dataset_id)

        if hasattr(source, "dtypes"):
            return source.dtypes()

        preview = source.head(0)
        return {str(col): str(dtype) for col, dtype in preview.dtypes.items()}

    def get_row_by_position(
        self,
        dataset_id: Optional[str],
        position: int,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return self.get_source(dataset_id).get_row_by_position(
            position,
            columns=columns,
        )

    def get_row_by_id(
        self,
        dataset_id: Optional[str],
        row_id: Any,
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return self.get_source(dataset_id).get_row_by_id(
            row_id,
            id_column=id_column,
            columns=columns,
        )
    
    def get_rows_by_ids(
        self,
        dataset_id: str,
        row_ids: Sequence[Any],
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Return multiple rows by record id using the active DatasetSource.

        This is the preferred path for visual overlays and small selected
        subsets because it lets DuckDB/Parquet backends do one vectorised
        lookup instead of requiring full prepared-frame scans.
        """
        source = self.get_source(dataset_id)

        method = getattr(source, "get_rows_by_ids", None)
        if callable(method):
            return method(
                row_ids,
                id_column=id_column,
                columns=columns,
            )

        frames: list[pd.DataFrame] = []
        for row_id in row_ids or []:
            try:
                row = source.get_row_by_id(
                    row_id,
                    id_column=id_column,
                    columns=columns,
                )
            except Exception:
                continue
            if row is not None and not row.empty:
                frames.append(row)

        if not frames:
            try:
                return pd.DataFrame(columns=source.columns() if columns is None else list(columns))
            except Exception:
                return pd.DataFrame(columns=list(columns or []))

        return pd.concat(frames, ignore_index=True)

    def find_position_by_id(
        self,
        dataset_id: Optional[str],
        row_id: Any,
        *,
        id_column: str,
    ) -> Optional[int]:
        return self.get_source(dataset_id).find_position_by_id(
            row_id,
            id_column=id_column,
        )

    # ------------------------------------------------------------------
    # Pending restore helpers
    # ------------------------------------------------------------------

    def _ensure_pending_restore_state(self) -> None:
        if not hasattr(self, "_pending_restore_items"):
            self._pending_restore_items = []
        if not hasattr(self, "_pending_active_id"):
            self._pending_active_id = None

    def _dataset_columns(self, dataset_id: str) -> list[str]:
        try:
            return [str(col) for col in self.list_columns(dataset_id)]
        except Exception:
            return []

    def _dataset_meta(self, dataset_id: str) -> dict[str, Any]:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            return {}
        return dict(getattr(dataset, "meta", {}) or {})

    def _dataset_name(self, dataset_id: str) -> str | None:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            return None
        return getattr(dataset, "name", None)

    @staticmethod
    def _source_from_meta(meta: dict[str, Any]) -> str | None:
        for key in (
            "source",
            "source_path",
            "path",
            "filename",
            "file",
            "filepath",
            "url",
        ):
            value = meta.get(key)
            if value:
                return str(value)
        return None

    def _dataset_source(self, dataset_id: str) -> str | None:
        return self._source_from_meta(self._dataset_meta(dataset_id))

    @staticmethod
    def _normalise_mapping_column(value: Any) -> str | None:
        if value is None:
            return None

        value = str(value)
        if value.lower() in {"use index", "use_index", "__index__", "index"}:
            return "Use Index"

        return value

    def _mapping_columns_exist(
        self,
        mappings: dict[str, Any],
        columns: list[str],
    ) -> bool:
        if not mappings:
            return True

        column_set = {str(col) for col in columns}

        for semantic_name, column_name in mappings.items():
            column_name = self._normalise_mapping_column(column_name)

            if column_name is None:
                mapping_debug_print(
                    "restored mapping invalid: None column",
                    {
                        "semantic_name": semantic_name,
                        "column_name": column_name,
                    },
                )
                return False

            if column_name == "Use Index":
                continue

            if column_name not in column_set:
                mapping_debug_print(
                    "restored mapping invalid: missing column",
                    {
                        "semantic_name": semantic_name,
                        "column_name": column_name,
                        "available_column_count": len(column_set),
                        "available_columns_preview": compact_list(sorted(column_set)),
                    },
                )
                return False

        return True

    def _pending_item_matches_dataset(
        self,
        item: dict[str, Any],
        dataset_id: str,
    ) -> bool:
        """
        Decide whether a pending restored mapping item belongs to a newly
        registered dataset.

        Matching order:
            1. Exact dataset id
            2. Same source/path metadata
            3. Same dataset name
            4. If there is only one pending item, allow it for the first
               dataset as long as mapped columns exist.
        """
        item_id = item.get("id")
        if item_id and str(item_id) == str(dataset_id):
            dataset_debug_print(
                "pending restore match by dataset_id",
                {
                    "item_id": item_id,
                    "dataset_id": dataset_id,
                },
            )
            return True

        item_meta = dict(item.get("meta") or {})
        item_source = self._source_from_meta(item_meta)
        dataset_source = self._dataset_source(dataset_id)
        if item_source and dataset_source and str(item_source) == str(dataset_source):
            dataset_debug_print(
                "pending restore match by source",
                {
                    "item_source": item_source,
                    "dataset_source": dataset_source,
                },
            )
            return True

        item_name = item.get("name")
        dataset_name = self._dataset_name(dataset_id)
        if item_name and dataset_name and str(item_name) == str(dataset_name):
            dataset_debug_print(
                "pending restore match by name",
                {
                    "item_name": item_name,
                    "dataset_name": dataset_name,
                },
            )
            return True

        self._ensure_pending_restore_state()
        mappings = dict(item.get("mappings") or {})
        columns = self._dataset_columns(dataset_id)

        if len(self._pending_restore_items) == 1:
            valid = self._mapping_columns_exist(mappings, columns)
            dataset_debug_print(
                "pending restore single-item fallback",
                {
                    "dataset_id": dataset_id,
                    "item_id": item_id,
                    "mappings": mappings,
                    "column_count": len(columns),
                    "columns_preview": compact_list(columns),
                    "valid": valid,
                },
            )
            return valid

        dataset_debug_print(
            "pending restore did not match",
            {
                "dataset_id": dataset_id,
                "item_id": item_id,
                "item_name": item_name,
                "dataset_name": dataset_name,
                "item_source": item_source,
                "dataset_source": dataset_source,
                "pending_count": len(self._pending_restore_items),
                "mappings": mappings,
                "column_count": len(columns),
                "columns_preview": compact_list(columns),
            },
        )
        return False

    def _publish_mapping_restored(
        self,
        dataset_id: str,
        mappings: dict[str, Any],
    ) -> None:
        events = (
            getattr(self, "events", None)
            or getattr(self, "_events", None)
            or getattr(self, "event_bus", None)
            or getattr(self, "_event_bus", None)
        )

        if events is None:
            return

        try:
            events.publish(
                "dataset.mapping_updated",
                {
                    "dataset_id": dataset_id,
                    "mappings": dict(mappings),
                    "origin": "workspace.restore",
                },
            )
        except Exception:
            pass

    def _apply_restore_item_to_dataset(
        self,
        dataset_id: str,
        item: dict[str, Any],
    ) -> bool:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            return False

        meta = dict(item.get("meta") or {})
        mappings = dict(item.get("mappings") or {})
        columns = self._dataset_columns(dataset_id)
        column_set = {str(col) for col in columns}

        valid_mappings = {}
        invalid_mappings = {}

        for semantic_key, column_name in mappings.items():
            column_name = self._normalise_mapping_column(column_name)
            if column_name is None:
                continue

            if column_name == "Use Index" or column_name in column_set:
                valid_mappings[str(semantic_key)] = str(column_name)
            else:
                invalid_mappings[str(semantic_key)] = str(column_name)

        dataset_meta = dict(getattr(dataset, "meta", {}) or {})
        dataset_meta.update(meta)

        existing_mappings = dict(dataset_meta.get("column_mappings", {}) or {})
        existing_mappings.update(valid_mappings)
        dataset_meta["column_mappings"] = existing_mappings

        if invalid_mappings:
            dataset_meta["invalid_restored_column_mappings"] = invalid_mappings
        else:
            dataset_meta.pop("invalid_restored_column_mappings", None)

        dataset.meta = dataset_meta

        if item.get("name") is not None:
            try:
                dataset.name = item["name"]
            except Exception:
                pass

        mapping_debug_print(
            "restored mappings",
            {
                "dataset_id": dataset_id,
                "valid": valid_mappings,
                "invalid": invalid_mappings,
            },
        )

        self._publish_mapping_restored(dataset_id, valid_mappings)
        return True

    def _apply_pending_restore_to_dataset(self, dataset_id: str) -> None:
        self._ensure_pending_restore_state()

        if not self._pending_restore_items:
            return

        remaining = []
        for item in self._pending_restore_items:
            if self._pending_item_matches_dataset(item, dataset_id):
                self._apply_restore_item_to_dataset(dataset_id, item)
            else:
                remaining.append(item)

        self._pending_restore_items = remaining

        if (
            self._pending_active_id is not None
            and str(self._pending_active_id) == str(dataset_id)
        ):
            try:
                self.set_active(dataset_id)
            except Exception:
                pass
            self._pending_active_id = None

    # ------------------------------------------------------------------
    # Workspace snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        items = []

        for dataset_id, dataset in self._datasets.items():
            meta = dict(getattr(dataset, "meta", {}) or {})
            mappings = dict(meta.pop("column_mappings", {}) or {})

            try:
                columns = [str(col) for col in dataset.source.columns()]
            except Exception:
                columns = []

            items.append(
                {
                    "id": dataset_id,
                    "name": getattr(dataset, "name", dataset_id),
                    "meta": _json_safe_metadata(meta),
                    "mappings": _json_safe_metadata(mappings),
                    "columns": columns,
                }
            )

        return {
            "active_id": self._active_id,
            "items": items,
        }

    def restore_metadata_snapshot(self, snapshot: dict[str, Any]) -> None:
        """
        Restore dataset metadata/mappings from a workspace snapshot.

        If datasets are not loaded yet, queue mapping information. If matching
        datasets are already loaded, apply immediately.
        """
        self._ensure_pending_restore_state()

        if not snapshot:
            dataset_debug_print("restore_metadata_snapshot skipped: empty snapshot")
            return

        self._pending_active_id = snapshot.get("active_id")
        items = snapshot.get("items", []) or []

        dataset_debug_print(
            "restore_metadata_snapshot start",
            {
                "active_id": self._pending_active_id,
                "items_count": len(items),
                "existing_datasets": list(self._datasets.keys()),
            },
        )

        for item in items:
            if not isinstance(item, dict):
                continue

            dataset_debug_print(
                "restore_metadata_snapshot item",
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "mappings": dict(item.get("mappings") or {}),
                },
            )

            dataset_id = item.get("id")
            applied = False

            if dataset_id and str(dataset_id) in self._datasets:
                dataset_debug_print(
                    "restore_metadata_snapshot applying immediately",
                    dataset_id,
                )
                applied = self._apply_restore_item_to_dataset(str(dataset_id), item)

            if not applied:
                dataset_debug_print(
                    "restore_metadata_snapshot queueing",
                    {
                        "id": item.get("id"),
                        "mappings": dict(item.get("mappings") or {}),
                    },
                )
                self._pending_restore_items.append(dict(item))

        for dataset_id in list(self._datasets.keys()):
            dataset_debug_print(
                "restore_metadata_snapshot trying pending restore",
                dataset_id,
            )
            self._apply_pending_restore_to_dataset(dataset_id)

        active_id = snapshot.get("active_id")
        if active_id and str(active_id) in self._datasets:
            try:
                self.set_active(str(active_id))
            except Exception:
                pass

        dataset_debug_print(
            "restore_metadata_snapshot complete",
            {
                "pending": summarize_pending_restore_items(
                    self._pending_restore_items
                ),
            },
        )

    # ------------------------------------------------------------------
    # Column mappings
    # ------------------------------------------------------------------

    def get_mappings(self, dataset_id: Optional[str] = None) -> Dict[str, str]:
        ds = self.get(dataset_id)
        ds.meta.setdefault("column_mappings", {})
        return ds.meta["column_mappings"]

    def get_mapping(
        self,
        dataset_id: Optional[str],
        semantic_name: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        mappings = self.get_mappings(dataset_id)
        return mappings.get(semantic_name, default)

    def set_mapping(
        self,
        dataset_id: Optional[str],
        semantic_name: str,
        column_name: str,
    ) -> bool:
        mappings = self.get_mappings(dataset_id)

        old_value = mappings.get(semantic_name)

        # Normalise to string for UI-provided values.
        old_norm = None if old_value is None else str(old_value)
        new_norm = None if column_name is None else str(column_name)

        if old_norm == new_norm:
            return False

        mappings[semantic_name] = column_name
        return True

    def has_mapping(
        self,
        dataset_id: Optional[str],
        semantic_name: str,
    ) -> bool:
        mappings = self.get_mappings(dataset_id)
        return semantic_name in mappings