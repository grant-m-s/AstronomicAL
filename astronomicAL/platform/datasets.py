from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Deque, Dict, Iterator, Optional, Sequence

import pandas as pd

from astronomicAL.platform.dataset_sources import (
    DatasetBatch,
    DatasetCapabilities,
    DatasetColumnOverlaySource,
    DatasetColumnStatistics,
    DatasetDistinctResult,
    DatasetScan,
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


class UnsafeDatasetMaterializationError(RuntimeError):
    """Raised when compatibility access would materialise an entire lazy dataset."""


@dataclass
class Dataset:
    dataset_id: str
    name: str
    source: DatasetSource
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def df(self) -> pd.DataFrame:
        """Compatibility shim for older in-memory datasets.

        ``Dataset.df`` is intentionally blocked for lazy/non-pandas sources. A
        property access cannot express a row/column bound or record telemetry,
        so allowing it to materialise a large Parquet-backed dataset would let a
        plugin exhaust the host process. Use ``DatasetManager.get_df()`` with
        explicit bounds, or use the backend-neutral source/manager APIs instead.
        """

        if not isinstance(self.source, PandasDatasetSource):
            backend = str(getattr(self.source, "backend_name", "unknown"))
            raise UnsafeDatasetMaterializationError(
                f"Dataset.df is blocked for lazy dataset {self.dataset_id!r} "
                f"(backend={backend!r}). Use DatasetManager row lookup, scans, "
                "bounded get_df(columns=..., limit=...), or explicitly opt in via "
                "DatasetManager.get_df(..., allow_full_materialization=True)."
            )
        return self.source.to_pandas()

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        """Compatibility shim for old code that mutates ``dataset.df``.

        This intentionally converts the dataset back to an in-memory pandas
        source. Avoid using this in new code.
        """

        self.source = PandasDatasetSource(value)

@dataclass(frozen=True)
class MaterializationRecord:
    dataset_id: str
    backend: str
    created_at: float
    elapsed_seconds: float
    requested_columns: Optional[tuple[str, ...]]
    requested_limit: Optional[int]
    filtered: bool
    source_row_count: Optional[int]
    source_column_count: int
    output_row_count: Optional[int]
    output_column_count: Optional[int]
    output_bytes: Optional[int]
    full_row_scan: bool
    full_dataset_materialization: bool
    origin: str
    succeeded: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "backend": self.backend,
            "created_at": self.created_at,
            "elapsed_seconds": self.elapsed_seconds,
            "requested_columns": (
                list(self.requested_columns)
                if self.requested_columns is not None
                else None
            ),
            "requested_limit": self.requested_limit,
            "filtered": self.filtered,
            "source_row_count": self.source_row_count,
            "source_column_count": self.source_column_count,
            "output_row_count": self.output_row_count,
            "output_column_count": self.output_column_count,
            "output_bytes": self.output_bytes,
            "full_row_scan": self.full_row_scan,
            "full_dataset_materialization": self.full_dataset_materialization,
            "origin": self.origin,
            "succeeded": self.succeeded,
            "error": self.error,
        }

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
    """Own datasets and expose backend-neutral access contracts.

    The canonical data payload is ``DatasetSource``, not ``pd.DataFrame``.

    Compatibility:
    - ``register(..., df=...)`` still works.
    - ``get_df(...)`` still works for bounded views and records materialisation
      telemetry. Full-table materialisation of a lazy source requires explicit
      opt-in.
    - ``Dataset.df`` remains available for pandas-backed datasets only; lazy
      sources are blocked because a property cannot express safe bounds.
    """

    def __init__(
        self,
        *,
        events: Any = None,
        materialization_history_limit: int = 100,
    ) -> None:
        if int(materialization_history_limit) <= 0:
            raise ValueError("materialization_history_limit must be greater than zero")
        self._datasets: Dict[str, Dataset] = {}
        self._active_id: Optional[str] = None
        self._events = events
        self._last_published_active_id: Optional[str] = None
        self._pending_restore_items: list[dict[str, Any]] = []
        self._pending_active_id: Optional[str] = None
        self._materializations: Deque[MaterializationRecord] = deque(
            maxlen=int(materialization_history_limit)
        )

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
        """Backward-compatible pandas registration.

        New code should prefer ``register_source()`` or ``register_parquet()``.
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
        source = coerce_dataset_source(source)
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
        """Register a Parquet dataset lazily through DuckDB.

        Metadata hints are important because UI refreshes should not repeatedly
        inspect or count large Parquet files.
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
        """Register the pandas dataset if missing; otherwise update it."""

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
        ds.meta.setdefault(
            "backend",
            getattr(ds.source, "backend_name", "unknown"),
        )

    def upsert_column_overlay(
        self,
        dataset_id: str,
        *,
        overlay_name: str,
        overlay_source: DatasetSource,
        base_record_id_column: str,
        overlay_record_id_column: str,
        columns: Sequence[str],
        preserve_base_on_missing: bool = False,
        origin: str = "platform.datasets.column_overlay",
        **meta: Any,
    ) -> list[str]:
        """Add or replace sidecar columns without creating another dataset.

        The dataset identity, base source, row order, and active-dataset state are
        preserved. The overlay is platform-owned after registration and can be
        replaced by name on later rounds.
        """

        dataset = self.get(dataset_id)
        previous_columns = set(dataset.source.columns())
        source = DatasetColumnOverlaySource.from_source(
            dataset.source
        ).with_overlay(
            name=str(overlay_name),
            source=overlay_source,
            base_record_id_column=str(base_record_id_column),
            overlay_record_id_column=str(overlay_record_id_column),
            columns=columns,
            preserve_base_on_missing=preserve_base_on_missing,
        )
        dataset.source = source
        dataset.meta.update(dict(meta or {}))
        dataset.meta["backend"] = source.backend_name
        dataset.meta["column_overlays"] = source.overlay_names()
        current_columns = source.columns()
        changed_columns = list(dict.fromkeys(str(column) for column in columns))
        added_columns = [
            column for column in changed_columns if column not in previous_columns
        ]
        payload = {
            "dataset_id": str(dataset_id),
            "updated_dataset_id": str(dataset_id),
            "overlay_name": str(overlay_name),
            "added_columns": added_columns,
            "changed_columns": changed_columns,
            "schema_changed": bool(added_columns),
            "change": (
                "column.added" if added_columns else "column.updated"
            ),
            "column_count": len(current_columns),
            "origin": str(origin),
        }
        self._publish_dataset_event("dataset.columns.changed", payload)
        self._publish_dataset_event("dataset.columns.updated", payload)
        self._publish_dataset_event("dataset.updated", payload)
        return changed_columns

    def remove_column_overlay(
        self,
        dataset_id: str,
        overlay_name: str,
        *,
        origin: str = "platform.datasets.column_overlay",
    ) -> bool:
        dataset = self.get(dataset_id)
        source = dataset.source
        if not isinstance(source, DatasetColumnOverlaySource):
            return False
        previous_columns = set(source.columns())
        updated = source.without_overlay(str(overlay_name))
        if updated is source:
            return False
        dataset.source = updated
        if isinstance(updated, DatasetColumnOverlaySource):
            dataset.meta["column_overlays"] = updated.overlay_names()
        else:
            dataset.meta.pop("column_overlays", None)
            dataset.meta["backend"] = getattr(
                updated,
                "backend_name",
                "unknown",
            )
        removed_columns = sorted(previous_columns - set(updated.columns()))
        payload = {
            "dataset_id": str(dataset_id),
            "updated_dataset_id": str(dataset_id),
            "overlay_name": str(overlay_name),
            "removed_columns": removed_columns,
            "changed_columns": removed_columns,
            "schema_changed": bool(removed_columns),
            "change": "column.removed",
            "origin": str(origin),
        }
        self._publish_dataset_event("dataset.columns.changed", payload)
        self._publish_dataset_event("dataset.updated", payload)
        return True

    def _publish_dataset_event(
        self,
        topic: str,
        payload: Dict[str, Any],
    ) -> None:
        if self._events is None:
            return
        try:
            self._events.publish(str(topic), dict(payload))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Basic access
    # ------------------------------------------------------------------

    def list_ids(self) -> list[str]:
        return list(self._datasets.keys())

    def active_id(self) -> str:
        if self._active_id is None:
            raise RuntimeError("No active dataset set.")
        return self._active_id

    def _publish_active_changed(
        self,
        *,
        dataset_id: str,
        previous_dataset_id: Optional[str],
        origin: str,
    ) -> bool:
        events = getattr(self, "_events", None)
        if events is None:
            return False
        try:
            events.publish(
                "dataset.active.changed",
                {
                    "dataset_id": dataset_id,
                    "previous_dataset_id": previous_dataset_id,
                    "origin": origin,
                },
            )
        except Exception:
            return False
        self._last_published_active_id = dataset_id
        return True

    def set_active(
        self,
        dataset_id: str,
        *,
        origin: str = "platform.datasets",
        publish: bool = True,
        force: bool = False,
    ) -> bool:
        if dataset_id not in self._datasets:
            raise KeyError(f"Unknown dataset_id: {dataset_id}")
        previous_dataset_id = self._active_id
        changed = previous_dataset_id != dataset_id
        self._active_id = dataset_id

        apply_pending = getattr(self, "_apply_pending_restore_to_dataset", None)
        if callable(apply_pending):
            apply_pending(dataset_id)

        should_publish = publish and (
            changed
            or force
            or self._last_published_active_id != dataset_id
        )
        if should_publish:
            self._publish_active_changed(
                dataset_id=dataset_id,
                previous_dataset_id=previous_dataset_id,
                origin=origin,
            )
        return changed

    def get(self, dataset_id: Optional[str] = None) -> Dataset:
        if dataset_id is None:
            dataset_id = self.active_id()
        if dataset_id not in self._datasets:
            raise KeyError(f"Unknown dataset_id: {dataset_id}")
        return self._datasets[dataset_id]

    def get_source(self, dataset_id: Optional[str] = None) -> DatasetSource:
        return self.get(dataset_id).source

    def capabilities(
        self,
        dataset_id: Optional[str] = None,
    ) -> DatasetCapabilities:
        return self.get_source(dataset_id).capabilities()

    def iter_batches(
        self,
        dataset_id: Optional[str] = None,
        *,
        scan: Optional[DatasetScan] = None,
        columns: Optional[Sequence[str]] = None,
        batch_size: int = 8192,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
        limit: Optional[int] = None,
        shard_index: int = 0,
        shard_count: int = 1,
    ) -> Iterator[DatasetBatch]:
        """Yield source batches through the manager-owned access boundary."""

        if scan is not None:
            if (
                columns is not None
                or batch_size != 8192
                or where_sql is not None
                or params is not None
                or limit is not None
                or shard_index != 0
                or shard_count != 1
            ):
                raise ValueError(
                    "Pass either scan=DatasetScan(...) or individual scan arguments, "
                    "not both."
                )
            request = scan
        else:
            request = DatasetScan(
                columns=(
                    None
                    if columns is None
                    else tuple(str(column) for column in columns)
                ),
                batch_size=int(batch_size),
                where_sql=where_sql,
                params=tuple(params or ()),
                limit=limit,
                shard_index=int(shard_index),
                shard_count=int(shard_count),
            )
        return self.get_source(dataset_id).iter_batches(request)

    def count_where(
        self,
        dataset_id: Optional[str] = None,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> Optional[int]:
        return self.get_source(dataset_id).count_where(
            where_sql=where_sql,
            params=params,
        )

    def distinct_values(
        self,
        dataset_id: Optional[str],
        column: str,
        *,
        limit: int = 100,
        include_null: bool = False,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> DatasetDistinctResult:
        return self.get_source(dataset_id).distinct_values(
            column,
            limit=limit,
            include_null=include_null,
            where_sql=where_sql,
            params=params,
        )

    def column_statistics(
        self,
        dataset_id: Optional[str],
        column: str,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> DatasetColumnStatistics:
        return self.get_source(dataset_id).column_statistics(
            column,
            where_sql=where_sql,
            params=params,
        )

    def get_df(
        self,
        dataset_id: Optional[str] = None,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
        origin: str = "platform.datasets.get_df",
        allow_full_materialization: bool = False,
    ) -> pd.DataFrame:
        """Materialise a pandas view and record platform telemetry.

        New code should prefer ``iter_batches()``, ``list_columns()``,
        ``row_count()``, row lookup methods, or bounded source summaries.

        A request for every row and every column of a lazy/non-pandas source is
        blocked by default. This is a host-safety boundary: a plugin should not
        be able to OOM the AstronomicAL process through an accidental legacy
        ``get_df()`` call. Truly intentional compatibility code may opt in with
        ``allow_full_materialization=True``.
        """

        resolved_dataset_id = dataset_id or self.active_id()
        source = self.get_source(resolved_dataset_id)
        source_columns = source.columns()
        source_rows = self._known_row_count_for_telemetry(
            resolved_dataset_id,
            source,
        )
        requested_columns = (
            None
            if columns is None
            else tuple(str(column) for column in columns)
        )
        full_row_scan = limit is None and where_sql is None
        full_dataset_materialization = (
            full_row_scan
            and (
                requested_columns is None
                or set(requested_columns) == set(source_columns)
            )
        )
        started_at = time.perf_counter()
        created_at = time.time()
        output: Optional[pd.DataFrame] = None
        error: Optional[str] = None
        try:
            if (
                full_dataset_materialization
                and not isinstance(source, PandasDatasetSource)
                and not bool(allow_full_materialization)
            ):
                row_summary = (
                    "unknown rows"
                    if source_rows is None
                    else f"{int(source_rows):,} rows"
                )
                raise UnsafeDatasetMaterializationError(
                    "Blocked full materialisation of lazy dataset "
                    f"{resolved_dataset_id!r} "
                    f"(backend={getattr(source, 'backend_name', 'unknown')!r}, "
                    f"{row_summary}, {len(source_columns):,} columns). "
                    "Request only the columns/rows required, use row lookup or "
                    "iter_batches(), or explicitly pass "
                    "allow_full_materialization=True for an intentional legacy "
                    "operation."
                )

            output = source.to_pandas(
                columns=columns,
                limit=limit,
                where_sql=where_sql,
                params=params,
            )
            return output
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            elapsed = max(0.0, time.perf_counter() - started_at)
            output_rows = None if output is None else int(len(output))
            output_columns = None if output is None else int(len(output.columns))
            output_bytes = None
            if output is not None:
                try:
                    output_bytes = int(output.memory_usage(index=True, deep=True).sum())
                except Exception:
                    pass
            record = MaterializationRecord(
                dataset_id=resolved_dataset_id,
                backend=str(getattr(source, "backend_name", "unknown")),
                created_at=created_at,
                elapsed_seconds=elapsed,
                requested_columns=requested_columns,
                requested_limit=None if limit is None else int(limit),
                filtered=where_sql is not None,
                source_row_count=(
                    None if source_rows is None else int(source_rows)
                ),
                source_column_count=len(source_columns),
                output_row_count=output_rows,
                output_column_count=output_columns,
                output_bytes=output_bytes,
                full_row_scan=full_row_scan,
                full_dataset_materialization=full_dataset_materialization,
                origin=str(origin or "platform.datasets.get_df"),
                succeeded=output is not None,
                error=error,
            )
            self._record_materialization(record)

    def materialization_history(
        self,
        *,
        limit: Optional[int] = None,
    ) -> list[MaterializationRecord]:
        records = list(self._materializations)
        records.reverse()
        if limit is None:
            return records
        return records[: max(0, int(limit))]

    def clear_materialization_history(self) -> None:
        self._materializations.clear()

    def _record_materialization(self, record: MaterializationRecord) -> None:
        self._materializations.append(record)
        events = getattr(self, "_events", None)
        if events is None:
            return
        try:
            events.publish("dataset.materialized", record.to_dict())
        except Exception:
            pass

    def _known_row_count_for_telemetry(
        self,
        dataset_id: str,
        source: DatasetSource,
    ) -> Optional[int]:
        """Return an already-known count without triggering a source scan."""

        dataset = self._datasets.get(dataset_id)
        meta = dict(getattr(dataset, "meta", {}) or {}) if dataset else {}
        for key in ("row_count", "rows", "n_rows"):
            value = meta.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                pass

        if isinstance(source, PandasDatasetSource):
            return int(source.row_count())

        cached = getattr(source, "_row_count_cache", None)
        if cached is not None:
            try:
                return int(cached)
            except Exception:
                pass
        return None

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
        return source.dtypes()

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
        """Return multiple rows by record ID using the active source."""

        source = self.get_source(dataset_id)
        return source.get_rows_by_ids(
            row_ids,
            id_column=id_column,
            columns=columns,
        )

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

    def _dataset_name(self, dataset_id: str) -> Optional[str]:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            return None
        return getattr(dataset, "name", None)

    @staticmethod
    def _source_from_meta(meta: dict[str, Any]) -> Optional[str]:
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

    def _dataset_source(self, dataset_id: str) -> Optional[str]:
        return self._source_from_meta(self._dataset_meta(dataset_id))

    @staticmethod
    def _normalise_mapping_column(value: Any) -> Optional[str]:
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
                        "available_columns_preview": compact_list(
                            sorted(column_set)
                        ),
                    },
                )
                return False
        return True

    def _pending_item_matches_dataset(
        self,
        item: dict[str, Any],
        dataset_id: str,
    ) -> bool:
        """Decide whether pending restored mappings belong to a dataset."""

        item_id = item.get("id")
        if item_id and str(item_id) == str(dataset_id):
            dataset_debug_print(
                "pending restore match by dataset_id",
                {"item_id": item_id, "dataset_id": dataset_id},
            )
            return True

        item_meta = dict(item.get("meta") or {})
        item_source = self._source_from_meta(item_meta)
        dataset_source = self._dataset_source(dataset_id)
        if (
            item_source
            and dataset_source
            and str(item_source) == str(dataset_source)
        ):
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
                {"item_name": item_name, "dataset_name": dataset_name},
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
                "dataset.mapping.updated",
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
        valid_mappings: dict[str, str] = {}
        invalid_mappings: dict[str, str] = {}
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
                self.set_active(
                    dataset_id,
                    origin="workspace.restore",
                    publish=False,
                )
            except Exception:
                pass

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
        """Restore dataset metadata and mappings from a workspace snapshot."""

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
                self.set_active(
                    str(active_id),
                    origin="workspace.restore",
                    publish=False,
                )
            except Exception:
                pass

        dataset_debug_print(
            "restore_metadata_snapshot complete",
            {
                "pending": summarize_pending_restore_items(
                    self._pending_restore_items
                )
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