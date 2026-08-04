from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class DatasetCapabilities:
    """Backend features callers may rely on without probing by failure."""

    batch_scan: bool = False
    batch_lookup_by_id: bool = False
    filtered_scan: bool = False
    ordered_scan: bool = False
    sharded_scan: bool = False
    bounded_distinct: bool = False
    column_statistics: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "batch_scan": self.batch_scan,
            "batch_lookup_by_id": self.batch_lookup_by_id,
            "filtered_scan": self.filtered_scan,
            "ordered_scan": self.ordered_scan,
            "sharded_scan": self.sharded_scan,
            "bounded_distinct": self.bounded_distinct,
            "column_statistics": self.column_statistics,
        }

@dataclass(frozen=True)
class DatasetScan:
    """A bounded tabular scan request.

    ``where_sql`` is transitional and matches the existing DuckDB source API.
    A backend-neutral predicate tree can replace it without changing the batch
    result contract. ``shard_index`` and ``shard_count`` describe disjoint
    physical scan shards for worker-backed consumers. Sources that advertise
    ``sharded_scan`` must ensure the union of all shards equals the unsharded
    scan without overlap.
    """

    columns: Optional[tuple[str, ...]] = None
    batch_size: int = 8192
    where_sql: Optional[str] = None
    params: tuple[Any, ...] = ()
    limit: Optional[int] = None
    shard_index: int = 0
    shard_count: int = 1

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError("DatasetScan.batch_size must be greater than zero")
        if self.limit is not None and int(self.limit) < 0:
            raise ValueError("DatasetScan.limit must be zero or greater")
        if int(self.shard_count) <= 0:
            raise ValueError("DatasetScan.shard_count must be greater than zero")
        if int(self.shard_index) < 0 or int(self.shard_index) >= int(self.shard_count):
            raise ValueError(
                "DatasetScan.shard_index must be between zero and shard_count - 1"
            )

@dataclass(frozen=True)
class DatasetBatch:
    frame: pd.DataFrame
    batch_index: int
    row_offset: int

    @property
    def row_count(self) -> int:
        return int(len(self.frame))


@dataclass(frozen=True)
class DatasetDistinctResult:
    """Bounded distinct values for one source column."""

    column: str
    values: tuple[Any, ...]
    truncated: bool
    scanned_rows: Optional[int]
    null_count: Optional[int]
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "values": list(self.values),
            "truncated": self.truncated,
            "scanned_rows": self.scanned_rows,
            "null_count": self.null_count,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class DatasetColumnStatistics:
    """Backend-computed summary for one source column."""

    column: str
    row_count: int
    non_null_count: int
    null_count: int
    distinct_count: Optional[int]
    minimum: Any
    maximum: Any
    mean: Optional[float]
    stddev: Optional[float]
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "row_count": self.row_count,
            "non_null_count": self.non_null_count,
            "null_count": self.null_count,
            "distinct_count": self.distinct_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "stddev": self.stddev,
            "backend": self.backend,
        }

def _normalise_columns(columns: Optional[Sequence[str]]) -> Optional[list[str]]:
    if columns is None:
        return None
    return [str(col) for col in columns]

def _scan_columns(
    available: Sequence[str],
    requested: Optional[Sequence[str]],
) -> Optional[list[str]]:
    selected = _normalise_columns(requested)
    if selected is None or not selected:
        return selected

    available_set = {str(column) for column in available}
    missing = [column for column in selected if column not in available_set]
    if missing:
        raise KeyError(f"Unknown scan columns: {missing!r}")
    return selected

def _quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier for DuckDB.

    This is for column names, not values. Values should still be passed as
    query parameters.
    """

    return '"' + str(identifier).replace('"', '""') + '"'

def _is_numeric_dtype_name(dtype: str) -> bool:
    value = str(dtype or "").upper()
    return any(
        token in value
        for token in (
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
            "FLOAT",
            "DOUBLE",
            "REAL",
            "DECIMAL",
            "NUMERIC",
        )
    )


def _python_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _stable_distinct_values(values: Iterable[Any]) -> list[Any]:
    return sorted(
        (_python_scalar(value) for value in values),
        key=lambda value: (type(value).__name__, str(value)),
    )

class DatasetSource(ABC):
    """Backend-neutral dataset access contract.

    A source may be backed by pandas, Parquet/DuckDB, Polars, Dask, Arrow, or
    a future remote/catalogue backend. Pandas is allowed as a returned
    materialised view, but should not be the canonical stored dataset object.
    """

    backend_name: str = "unknown"

    @abstractmethod
    def columns(self) -> list[str]:
        """Return dataset column names without materialising the full dataset."""

    @abstractmethod
    def row_count(self) -> Optional[int]:
        """Return row count if cheaply available."""

    def capabilities(self) -> DatasetCapabilities:
        return DatasetCapabilities()

    def iter_batches(self, scan: DatasetScan) -> Iterator[DatasetBatch]:
        """Yield bounded pandas batches without materialising the full source.

        Sources must keep backend resources alive for the iterator and release
        them when iteration completes, fails, or is explicitly closed.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not implement iter_batches()"
        )

    def count_where(
        self,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> Optional[int]:
        """Return a filtered row count if the backend can do it cheaply."""

        if not where_sql:
            count = self.row_count()
            return None if count is None else int(count)
        raise NotImplementedError(
            f"{type(self).__name__} does not implement count_where()"
        )

    def distinct_values(
        self,
        column: str,
        *,
        limit: int = 100,
        include_null: bool = False,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> DatasetDistinctResult:
        """Return at most ``limit`` deterministic distinct values."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement distinct_values()"
        )

    def column_statistics(
        self,
        column: str,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> DatasetColumnStatistics:
        """Return bounded metadata without materialising the source table."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement column_statistics()"
        )

    def sample_points(
        self,
        *,
        x_col: str,
        y_col: str,
        record_id_col: Optional[str] = None,
        limit: int = 10_000,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        log_x: bool = False,
        log_y: bool = False,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Return a small sampled point cloud for scatter first-paint."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement sample_points()"
        )

    def aggregate_2d(
        self,
        *,
        x_col: str,
        y_col: str,
        bins: int,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        log_x: bool = False,
        log_y: bool = False,
    ) -> dict[str, Any]:
        """Return a small two-dimensional count grid for density rendering."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement aggregate_2d()"
        )

    @abstractmethod
    def to_pandas(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        """Materialise a pandas view.

        This method exists for compatibility with old plugins. New plugins
        should request only the columns and rows they need.
        """

    @abstractmethod
    def dtypes(self) -> dict[str, str]:
        """Return column dtype information without materialising the dataset."""

        preview = self.head(0)
        return {str(col): str(dtype) for col, dtype in preview.dtypes.items()}

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
        return self.to_pandas(
            columns=columns,
            limit=1,
            where_sql=None,
            params=None,
        ).iloc[0:0]

    def get_row_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_row_by_id()"
        )

    def get_rows_by_ids(
        self,
        row_ids: Sequence[Any],
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for row_id in row_ids or []:
            try:
                row = self.get_row_by_id(
                    row_id,
                    id_column=id_column,
                    columns=columns,
                )
            except Exception:
                continue
            if row is not None and not row.empty:
                frames.append(row)

        if not frames:
            return pd.DataFrame(
                columns=self.columns() if columns is None else list(columns)
            )
        return pd.concat(frames, ignore_index=True)

    def find_position_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
    ) -> Optional[int]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement find_position_by_id()"
        )

    def sample_for_display(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        max_rows: int = 100_000,
    ) -> pd.DataFrame:
        return self.to_pandas(columns=columns, limit=max_rows)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "capabilities": self.capabilities().to_dict(),
        }

class PandasDatasetSource(DatasetSource):
    """Compatibility source for existing in-memory pandas DataFrames."""

    backend_name = "pandas"

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def dtypes(self) -> dict[str, str]:
        return {str(col): str(dtype) for col, dtype in self._df.dtypes.items()}

    def columns(self) -> list[str]:
        return [str(col) for col in self._df.columns]

    def row_count(self) -> int:
        return int(len(self._df))

    def capabilities(self) -> DatasetCapabilities:
        return DatasetCapabilities(
            batch_scan=True,
            batch_lookup_by_id=True,
            filtered_scan=False,
            ordered_scan=True,
            sharded_scan=True,
            bounded_distinct=True,
            column_statistics=True,
        )

    def iter_batches(self, scan: DatasetScan) -> Iterator[DatasetBatch]:
        if scan.where_sql is not None:
            raise NotImplementedError(
                "PandasDatasetSource does not support where_sql batch scans."
            )
        if scan.params:
            raise ValueError("DatasetScan.params requires where_sql")

        selected_columns = _scan_columns(self.columns(), scan.columns)
        frame = self._df
        if selected_columns:
            frame = frame.loc[:, selected_columns]
        if scan.limit is not None:
            frame = frame.iloc[: int(scan.limit)]
        if int(scan.shard_count) > 1:
            frame = frame.iloc[
                int(scan.shard_index) :: int(scan.shard_count)
            ]

        total_rows = len(frame)
        batch_size = int(scan.batch_size)
        for batch_index, start in enumerate(range(0, total_rows, batch_size)):
            stop = min(start + batch_size, total_rows)
            yield DatasetBatch(
                frame=frame.iloc[start:stop].copy(),
                batch_index=batch_index,
                row_offset=start,
            )

    def count_where(
        self,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> Optional[int]:
        if where_sql:
            raise NotImplementedError(
                "PandasDatasetSource does not support SQL count_where()."
            )
        return int(len(self._df))

    def distinct_values(
        self,
        column: str,
        *,
        limit: int = 100,
        include_null: bool = False,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> DatasetDistinctResult:
        if where_sql:
            raise NotImplementedError(
                "PandasDatasetSource does not support SQL distinct filters."
            )
        if params:
            raise ValueError("params requires where_sql")
        if column not in self._df.columns:
            raise KeyError(f"Unknown distinct column: {column!r}")
        limit = int(limit)
        if limit < 0:
            raise ValueError("limit must be zero or greater")

        series = self._df[column]
        null_count = int(series.isna().sum())
        values = list(series.dropna().unique())
        if include_null and null_count:
            values.append(None)
        values = _stable_distinct_values(values)
        truncated = len(values) > limit
        return DatasetDistinctResult(
            column=str(column),
            values=tuple(values[:limit]),
            truncated=truncated,
            scanned_rows=int(len(series)),
            null_count=null_count,
            backend=self.backend_name,
        )

    def column_statistics(
        self,
        column: str,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> DatasetColumnStatistics:
        if where_sql:
            raise NotImplementedError(
                "PandasDatasetSource does not support SQL statistics filters."
            )
        if params:
            raise ValueError("params requires where_sql")
        if column not in self._df.columns:
            raise KeyError(f"Unknown statistics column: {column!r}")

        series = self._df[column]
        non_null = series.dropna()
        row_count = int(len(series))
        non_null_count = int(len(non_null))
        minimum = None
        maximum = None
        if non_null_count:
            try:
                minimum = _python_scalar(non_null.min())
                maximum = _python_scalar(non_null.max())
            except Exception:
                pass

        mean = None
        stddev = None
        if (
            non_null_count
            and pd.api.types.is_numeric_dtype(series.dtype)
            and not pd.api.types.is_bool_dtype(series.dtype)
        ):
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(numeric):
                mean = float(numeric.mean())
                stddev = float(numeric.std(ddof=0))

        return DatasetColumnStatistics(
            column=str(column),
            row_count=row_count,
            non_null_count=non_null_count,
            null_count=row_count - non_null_count,
            distinct_count=int(non_null.nunique(dropna=True)),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            stddev=stddev,
            backend=self.backend_name,
        )

    def sample_points(
        self,
        *,
        x_col: str,
        y_col: str,
        record_id_col: Optional[str] = None,
        limit: int = 10_000,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        log_x: bool = False,
        log_y: bool = False,
        seed: int = 0,
    ) -> dict[str, Any]:
        limit = max(1, int(limit))
        if x_col not in self._df.columns or y_col not in self._df.columns:
            raise KeyError(f"Unknown sample columns: {x_col!r}, {y_col!r}")

        x = pd.to_numeric(self._df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(self._df[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)

        if log_x:
            mask &= x > 0
        if log_y:
            mask &= y > 0
        if x_range is not None:
            x0, x1 = float(min(x_range)), float(max(x_range))
            mask &= x >= x0
            mask &= x <= x1
        if y_range is not None:
            y0, y1 = float(min(y_range)), float(max(y_range))
            mask &= y >= y0
            mask &= y <= y1

        positions = np.flatnonzero(mask)
        row_count = int(len(positions))
        if row_count == 0:
            return {
                "frame": pd.DataFrame(columns=["__x", "__y", "__row_id"]),
                "row_count": 0,
                "sampled_from": 0,
                "backend": self.backend_name,
            }

        if row_count > limit:
            take_idx = np.linspace(0, row_count - 1, num=limit, dtype=np.int64)
            positions = positions[take_idx]

        if (
            record_id_col
            and record_id_col != "Use Index"
            and record_id_col in self._df.columns
        ):
            row_ids = self._df[record_id_col].to_numpy(copy=False)[positions]
        else:
            row_ids = positions

        frame = pd.DataFrame(
            {
                "__x": x[positions],
                "__y": y[positions],
                "__row_id": row_ids,
            },
            copy=False,
        )
        return {
            "frame": frame,
            "row_count": row_count,
            "sampled_from": row_count,
            "backend": self.backend_name,
        }

    def aggregate_2d(
        self,
        *,
        x_col: str,
        y_col: str,
        bins: int,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        log_x: bool = False,
        log_y: bool = False,
    ) -> dict[str, Any]:
        bins = max(5, min(500, int(bins)))
        if x_col not in self._df.columns or y_col not in self._df.columns:
            raise KeyError(f"Unknown aggregate columns: {x_col!r}, {y_col!r}")

        x_raw = pd.to_numeric(self._df[x_col], errors="coerce").to_numpy(dtype=float)
        y_raw = pd.to_numeric(self._df[y_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x_raw) & np.isfinite(y_raw)

        if log_x:
            finite &= x_raw > 0
        if log_y:
            finite &= y_raw > 0
        if x_range is not None:
            x0_raw, x1_raw = float(x_range[0]), float(x_range[1])
            finite &= x_raw >= min(x0_raw, x1_raw)
            finite &= x_raw <= max(x0_raw, x1_raw)
        if y_range is not None:
            y0_raw, y1_raw = float(y_range[0]), float(y_range[1])
            finite &= y_raw >= min(y0_raw, y1_raw)
            finite &= y_raw <= max(y0_raw, y1_raw)

        x_raw = x_raw[finite]
        y_raw = y_raw[finite]
        if len(x_raw) == 0 or len(y_raw) == 0:
            raise ValueError("No finite X/Y rows available for 2D aggregation")

        if x_range is None:
            raw_x0 = float(np.nanmin(x_raw))
            raw_x1 = float(np.nanmax(x_raw))
        else:
            raw_x0 = float(min(x_range))
            raw_x1 = float(max(x_range))

        if y_range is None:
            raw_y0 = float(np.nanmin(y_raw))
            raw_y1 = float(np.nanmax(y_raw))
        else:
            raw_y0 = float(min(y_range))
            raw_y1 = float(max(y_range))

        if log_x:
            plot_x = np.log10(x_raw)
            plot_x0 = float(np.log10(raw_x0))
            plot_x1 = float(np.log10(raw_x1))
        else:
            plot_x = x_raw
            plot_x0 = raw_x0
            plot_x1 = raw_x1

        if log_y:
            plot_y = np.log10(y_raw)
            plot_y0 = float(np.log10(raw_y0))
            plot_y1 = float(np.log10(raw_y1))
        else:
            plot_y = y_raw
            plot_y0 = raw_y0
            plot_y1 = raw_y1

        if plot_x1 <= plot_x0 or plot_y1 <= plot_y0:
            raise ValueError("Invalid aggregate range")

        counts, y_edges, x_edges = np.histogram2d(
            plot_y,
            plot_x,
            bins=[bins, bins],
            range=[
                [plot_y0, plot_y1],
                [plot_x0, plot_x1],
            ],
        )
        return {
            "counts": counts.astype("float64", copy=False),
            "x_edges": np.asarray(x_edges, dtype=float),
            "y_edges": np.asarray(y_edges, dtype=float),
            "raw_x_range": (raw_x0, raw_x1),
            "raw_y_range": (raw_y0, raw_y1),
            "plot_x_range": (plot_x0, plot_x1),
            "plot_y_range": (plot_y0, plot_y1),
            "row_count": int(len(plot_x)),
            "backend": self.backend_name,
        }

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
                "PandasDatasetSource does not support where_sql. "
                "Use DatasetManager.get_source(...).to_pandas() on a SQL-capable "
                "source, or add a pandas filter method explicitly."
            )

        df = self._df
        selected_columns = _normalise_columns(columns)
        if selected_columns is not None:
            existing = [col for col in selected_columns if col in df.columns]
            df = df.loc[:, existing]
        if limit is not None:
            df = df.head(int(limit))
        return df.copy()

    def get_row_by_position(
        self,
        position: int,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if position < 0 or position >= len(self._df):
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        df = self._df
        selected_columns = _normalise_columns(columns)
        if selected_columns is not None:
            selected_columns = [col for col in selected_columns if col in df.columns]
            df = df.loc[:, selected_columns]
        return df.iloc[[position]].copy()

    def get_row_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if id_column == "Use Index":
            mask = self._df.index.astype(str) == str(row_id)
        else:
            if id_column not in self._df.columns:
                return pd.DataFrame(
                    columns=self.columns() if columns is None else columns
                )
            mask = self._df[id_column].astype(str) == str(row_id)

        matches = self._df.loc[mask]
        if matches.empty:
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        selected_columns = _normalise_columns(columns)
        if selected_columns is not None:
            selected_columns = [
                col for col in selected_columns if col in matches.columns
            ]
            matches = matches.loc[:, selected_columns]
        return matches.head(1).copy()

    def get_rows_by_ids(
        self,
        row_ids: Sequence[Any],
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        ids = [str(row_id) for row_id in (row_ids or [])]
        if not ids:
            return pd.DataFrame(
                columns=self.columns() if columns is None else list(columns)
            )

        requested_order = {row_id: i for i, row_id in enumerate(ids)}
        if id_column == "Use Index":
            values = self._df.index.astype(str)
            matches = self._df.loc[values.isin(set(ids))].copy()
            if matches.empty:
                return pd.DataFrame(
                    columns=self.columns() if columns is None else list(columns)
                )
            matches["__astronomical_lookup_id"] = matches.index.astype(str)
            order_source = matches["__astronomical_lookup_id"]
        else:
            if id_column not in self._df.columns:
                return pd.DataFrame(
                    columns=self.columns() if columns is None else list(columns)
                )
            values = self._df[id_column].astype(str)
            matches = self._df.loc[values.isin(set(ids))].copy()
            if matches.empty:
                return pd.DataFrame(
                    columns=self.columns() if columns is None else list(columns)
                )
            order_source = matches[id_column].astype(str)

        matches["__astronomical_lookup_order"] = order_source.map(requested_order)
        matches = matches.sort_values("__astronomical_lookup_order", kind="stable")
        selected_columns = _normalise_columns(columns)
        if selected_columns is not None:
            selected_columns = [
                col for col in selected_columns if col in matches.columns
            ]
            matches = matches.loc[:, selected_columns]
        return matches.reset_index(drop=True).copy()

    def find_position_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
    ) -> Optional[int]:
        if id_column == "Use Index":
            values = self._df.index.astype(str)
        else:
            if id_column not in self._df.columns:
                return None
            values = self._df[id_column].astype(str)

        matches = values == str(row_id)
        positions = np.flatnonzero(matches)
        if len(positions) == 0:
            return None
        return int(positions[0])

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "rows": len(self._df),
            "columns": len(self._df.columns),
            "capabilities": self.capabilities().to_dict(),
        }

class DuckDBParquetDatasetSource(DatasetSource):
    backend_name = "duckdb_parquet"

    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        *,
        dataset_name: Optional[str] = None,
        columns_hint: Optional[Sequence[str]] = None,
        row_count_hint: Optional[int] = None,
    ):
        self.path = path
        self.dataset_name = dataset_name
        self._column_cache: Optional[list[str]] = (
            [str(col) for col in columns_hint]
            if columns_hint is not None
            else None
        )
        self._row_count_cache: Optional[int] = (
            int(row_count_hint) if row_count_hint is not None else None
        )

    def capabilities(self) -> DatasetCapabilities:
        return DatasetCapabilities(
            batch_scan=True,
            batch_lookup_by_id=True,
            filtered_scan=True,
            ordered_scan=False,
            sharded_scan=True,
            bounded_distinct=True,
            column_statistics=True,
        )

    def dtypes(self) -> dict[str, str]:
        con = self._connect()
        try:
            df = con.execute(
                f"DESCRIBE SELECT * FROM {self._relation_sql()} LIMIT 0",
                [self._path_argument()],
            ).df()
        finally:
            con.close()

        result = {}
        for _, row in df.iterrows():
            result[str(row["column_name"])] = str(row["column_type"])
        return result

    def _path_argument(self) -> str | list[str]:
        if isinstance(self.path, (list, tuple)):
            return [str(Path(p)) for p in self.path]
        return str(Path(self.path))

    def _connect(self):
        import duckdb

        return duckdb.connect(database=":memory:", read_only=False)

    def _relation_sql(self) -> str:
        return "read_parquet(?)"

    def _select_sql(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> str:
        selected_columns = _normalise_columns(columns)
        if selected_columns is None or not selected_columns:
            return "*"
        return ", ".join(_quote_identifier(col) for col in selected_columns)

    def columns(self) -> list[str]:
        if self._column_cache is not None:
            return list(self._column_cache)

        con = self._connect()
        try:
            df = con.execute(
                f"DESCRIBE SELECT * FROM {self._relation_sql()} LIMIT 0",
                [self._path_argument()],
            ).df()
        finally:
            con.close()

        self._column_cache = [str(col) for col in df["column_name"].tolist()]
        return list(self._column_cache)

    def row_count(self) -> Optional[int]:
        if self._row_count_cache is not None:
            return self._row_count_cache

        con = self._connect()
        try:
            value = con.execute(
                f"SELECT COUNT(*) AS n FROM {self._relation_sql()}",
                [self._path_argument()],
            ).fetchone()[0]
        finally:
            con.close()

        self._row_count_cache = int(value)
        return self._row_count_cache

    def count_where(
        self,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> int:
        sql = f"SELECT COUNT(*) AS n FROM {self._relation_sql()}"
        sql_params: list[Any] = [self._path_argument()]
        if where_sql:
            sql += f" WHERE {where_sql}"
            if params:
                sql_params.extend(list(params))

        con = self._connect()
        try:
            value = con.execute(sql, sql_params).fetchone()[0]
        finally:
            con.close()
        return int(value or 0)

    def distinct_values(
        self,
        column: str,
        *,
        limit: int = 100,
        include_null: bool = False,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> DatasetDistinctResult:
        if column not in self.columns():
            raise KeyError(f"Unknown distinct column: {column!r}")
        limit = int(limit)
        if limit < 0:
            raise ValueError("limit must be zero or greater")

        quoted = _quote_identifier(column)
        where_parts: list[str] = []
        sql_params: list[Any] = [self._path_argument()]
        if where_sql:
            where_parts.append(f"({where_sql})")
            sql_params.extend(list(params or []))
        elif params:
            raise ValueError("params requires where_sql")
        if not include_null:
            where_parts.append(f"{quoted} IS NOT NULL")

        sql = f"SELECT DISTINCT {quoted} AS value FROM {self._relation_sql()}"
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        sql += " ORDER BY CAST(value AS VARCHAR) LIMIT ?"
        sql_params.append(limit + 1)

        null_sql = f"SELECT COUNT(*) FROM {self._relation_sql()} WHERE {quoted} IS NULL"
        null_params: list[Any] = [self._path_argument()]
        if where_sql:
            null_sql += f" AND ({where_sql})"
            null_params.extend(list(params or []))

        con = self._connect()
        try:
            rows = con.execute(sql, sql_params).fetchall()
            null_count = int(con.execute(null_sql, null_params).fetchone()[0] or 0)
        finally:
            con.close()

        values = [_python_scalar(row[0]) for row in rows]
        return DatasetDistinctResult(
            column=str(column),
            values=tuple(values[:limit]),
            truncated=len(values) > limit,
            scanned_rows=self.count_where(where_sql=where_sql, params=params),
            null_count=null_count,
            backend=self.backend_name,
        )

    def column_statistics(
        self,
        column: str,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> DatasetColumnStatistics:
        if column not in self.columns():
            raise KeyError(f"Unknown statistics column: {column!r}")
        if params and not where_sql:
            raise ValueError("params requires where_sql")

        quoted = _quote_identifier(column)
        dtype = self._column_dtype_for_lookup(column)
        numeric = _is_numeric_dtype_name(dtype) and "BOOL" not in dtype
        mean_sql = f"AVG(CAST({quoted} AS DOUBLE))" if numeric else "NULL"
        stddev_sql = (
            f"STDDEV_POP(CAST({quoted} AS DOUBLE))" if numeric else "NULL"
        )
        sql = (
            "SELECT "
            "COUNT(*) AS row_count, "
            f"COUNT({quoted}) AS non_null_count, "
            f"COUNT(DISTINCT {quoted}) AS distinct_count, "
            f"MIN({quoted}) AS minimum, "
            f"MAX({quoted}) AS maximum, "
            f"{mean_sql} AS mean, "
            f"{stddev_sql} AS stddev "
            f"FROM {self._relation_sql()}"
        )
        sql_params: list[Any] = [self._path_argument()]
        if where_sql:
            sql += f" WHERE {where_sql}"
            sql_params.extend(list(params or []))

        con = self._connect()
        try:
            row = con.execute(sql, sql_params).fetchone()
        finally:
            con.close()
        row_count = int(row[0] or 0)
        non_null_count = int(row[1] or 0)
        return DatasetColumnStatistics(
            column=str(column),
            row_count=row_count,
            non_null_count=non_null_count,
            null_count=row_count - non_null_count,
            distinct_count=int(row[2] or 0),
            minimum=_python_scalar(row[3]),
            maximum=_python_scalar(row[4]),
            mean=None if row[5] is None else float(row[5]),
            stddev=None if row[6] is None else float(row[6]),
            backend=self.backend_name,
        )

    def iter_batches(self, scan: DatasetScan) -> Iterator[DatasetBatch]:
        selected_columns = _scan_columns(self.columns(), scan.columns)
        select_sql = self._select_sql(columns=selected_columns)
        path_argument, physically_sharded = self._scan_path_argument(scan)
        if physically_sharded and not path_argument:
            return

        sql_params: list[Any] = [path_argument]
        if int(scan.shard_count) > 1 and not physically_sharded:
            position_column = "__astronomical_scan_position"
            inner_sql = (
                f"SELECT {select_sql}, "
                f"ROW_NUMBER() OVER () - 1 AS {_quote_identifier(position_column)} "
                f"FROM {self._relation_sql()}"
            )
            if scan.where_sql:
                inner_sql += f" WHERE {scan.where_sql}"
                sql_params.extend(scan.params)
            elif scan.params:
                raise ValueError("DatasetScan.params requires where_sql")
            if scan.limit is not None:
                inner_sql += " LIMIT ?"
                sql_params.append(int(scan.limit))
            outer_select_sql = (
                select_sql
                if select_sql != "*"
                else f"* EXCLUDE ({_quote_identifier(position_column)})"
            )
            sql = (
                f"SELECT {outer_select_sql} FROM ({inner_sql}) AS scan_source "
                f"WHERE MOD({_quote_identifier(position_column)}, ?) = ?"
            )
            sql_params.extend(
                [int(scan.shard_count), int(scan.shard_index)]
            )
        else:
            sql = f"SELECT {select_sql} FROM {self._relation_sql()}"
            if scan.where_sql:
                sql += f" WHERE {scan.where_sql}"
                sql_params.extend(scan.params)
            elif scan.params:
                raise ValueError("DatasetScan.params requires where_sql")
            if scan.limit is not None:
                sql += " LIMIT ?"
                sql_params.append(int(scan.limit))

        con = self._connect()
        try:
            cursor = con.execute(sql, sql_params)
            column_names = [
                str(description[0]) for description in cursor.description
            ]
            row_offset = 0
            batch_index = 0

            while True:
                rows = cursor.fetchmany(int(scan.batch_size))
                if not rows:
                    break
                frame = pd.DataFrame.from_records(rows, columns=column_names)
                yield DatasetBatch(
                    frame=frame,
                    batch_index=batch_index,
                    row_offset=row_offset,
                )
                row_offset += len(frame)
                batch_index += 1
        finally:
            con.close()

    def _scan_path_argument(
        self,
        scan: DatasetScan,
    ) -> tuple[str | list[str], bool]:
        """Return a path argument and whether file-level sharding was applied."""

        path_argument = self._path_argument()
        if (
            int(scan.shard_count) <= 1
            or scan.where_sql
            or scan.limit is not None
            or not isinstance(path_argument, list)
        ):
            return path_argument, False
        return (
            path_argument[
                int(scan.shard_index) :: int(scan.shard_count)
            ],
            True,
        )

    def sample_points(
        self,
        *,
        x_col: str,
        y_col: str,
        record_id_col: Optional[str] = None,
        limit: int = 10_000,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        log_x: bool = False,
        log_y: bool = False,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Return a bounded scatter sample directly from DuckDB/Parquet."""

        limit = max(1, int(limit))
        available = set(self.columns())
        if x_col not in available or y_col not in available:
            raise KeyError(f"Unknown sample columns: {x_col!r}, {y_col!r}")

        qx = _quote_identifier(str(x_col))
        qy = _quote_identifier(str(y_col))
        if (
            record_id_col
            and record_id_col != "Use Index"
            and record_id_col in available
        ):
            qid = _quote_identifier(str(record_id_col))
            row_id_expr = f"CAST({qid} AS VARCHAR)"
        else:
            row_id_expr = "CAST(__rownum AS VARCHAR)"

        where_clauses = ["isfinite(__x)", "isfinite(__y)"]
        params: list[Any] = []
        if log_x:
            where_clauses.append("__x > 0")
        if log_y:
            where_clauses.append("__y > 0")
        if x_range is not None:
            x0, x1 = float(min(x_range)), float(max(x_range))
            where_clauses.extend(["__x >= ?", "__x <= ?"])
            params.extend([x0, x1])
        if y_range is not None:
            y0, y1 = float(min(y_range)), float(max(y_range))
            where_clauses.extend(["__y >= ?", "__y <= ?"])
            params.extend([y0, y1])

        where_sql = " AND ".join(where_clauses)
        sql = (
            "WITH src AS ("
            " SELECT "
            f" CAST({qx} AS DOUBLE) AS __x, "
            f" CAST({qy} AS DOUBLE) AS __y, "
            " row_number() OVER () - 1 AS __rownum, "
            f" {row_id_expr} AS __row_id "
            f" FROM {self._relation_sql()}"
            "), filtered AS ("
            " SELECT __x, __y, __row_id, __rownum "
            " FROM src "
            f" WHERE {where_sql}"
            "), sampled AS ("
            " SELECT __x, __y, __row_id "
            " FROM filtered "
            " WHERE MOD(__rownum, ?) = 0 "
            " LIMIT ?"
            ") "
            "SELECT __x, __y, __row_id FROM sampled"
        )

        try:
            approx_total = int(self.row_count() or 0)
        except Exception:
            approx_total = 0
        stride = max(1, int(approx_total // max(1, limit)))
        sql_params: list[Any] = [
            self._path_argument(),
            *params,
            stride,
            limit,
        ]

        con = self._connect()
        try:
            df = con.execute(sql, sql_params).df()
        finally:
            con.close()

        if df is None or df.empty:
            return {
                "frame": pd.DataFrame(columns=["__x", "__y", "__row_id"]),
                "row_count": 0,
                "sampled_from": 0,
                "backend": self.backend_name,
            }

        row_count = approx_total
        frame = df[["__x", "__y", "__row_id"]].copy()
        return {
            "frame": frame,
            "row_count": row_count,
            "sampled_from": row_count,
            "backend": self.backend_name,
        }

    def aggregate_2d(
        self,
        *,
        x_col: str,
        y_col: str,
        bins: int,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        log_x: bool = False,
        log_y: bool = False,
    ) -> dict[str, Any]:
        bins = max(5, min(500, int(bins)))
        available = set(self.columns())
        if x_col not in available or y_col not in available:
            raise KeyError(f"Unknown aggregate columns: {x_col!r}, {y_col!r}")

        qx = _quote_identifier(str(x_col))
        qy = _quote_identifier(str(y_col))
        x_plot_expr = "log10(x_raw)" if log_x else "x_raw"
        y_plot_expr = "log10(y_raw)" if log_y else "y_raw"
        raw_where = ["isfinite(x_raw)", "isfinite(y_raw)"]
        raw_params: list[Any] = []

        if log_x:
            raw_where.append("x_raw > 0")
        if log_y:
            raw_where.append("y_raw > 0")
        if x_range is not None:
            x0_raw = float(min(x_range))
            x1_raw = float(max(x_range))
            raw_where.extend(["x_raw >= ?", "x_raw <= ?"])
            raw_params.extend([x0_raw, x1_raw])
        if y_range is not None:
            y0_raw = float(min(y_range))
            y1_raw = float(max(y_range))
            raw_where.extend(["y_raw >= ?", "y_raw <= ?"])
            raw_params.extend([y0_raw, y1_raw])

        raw_where_sql = " AND ".join(raw_where)
        extent_sql = (
            "WITH src AS ("
            f" SELECT CAST({qx} AS DOUBLE) AS x_raw, "
            f" CAST({qy} AS DOUBLE) AS y_raw "
            f" FROM {self._relation_sql()}"
            "), filtered AS ("
            f" SELECT x_raw, y_raw, {x_plot_expr} AS x_plot, "
            f" {y_plot_expr} AS y_plot "
            " FROM src "
            f" WHERE {raw_where_sql}"
            ") "
            "SELECT "
            " MIN(x_raw) AS raw_x_min, "
            " MAX(x_raw) AS raw_x_max, "
            " MIN(y_raw) AS raw_y_min, "
            " MAX(y_raw) AS raw_y_max, "
            " MIN(x_plot) AS plot_x_min, "
            " MAX(x_plot) AS plot_x_max, "
            " MIN(y_plot) AS plot_y_min, "
            " MAX(y_plot) AS plot_y_max, "
            " COUNT(*) AS n "
            "FROM filtered"
        )

        con = self._connect()
        try:
            extent_row = con.execute(
                extent_sql,
                [self._path_argument(), *raw_params],
            ).fetchone()
            if extent_row is None:
                raise ValueError("No finite X/Y rows available for 2D aggregation")

            (
                raw_x0,
                raw_x1,
                raw_y0,
                raw_y1,
                plot_x0,
                plot_x1,
                plot_y0,
                plot_y1,
                row_count,
            ) = extent_row
            row_count = int(row_count or 0)
            if row_count <= 0:
                raise ValueError("No finite X/Y rows available for 2D aggregation")

            raw_x0 = float(raw_x0)
            raw_x1 = float(raw_x1)
            raw_y0 = float(raw_y0)
            raw_y1 = float(raw_y1)
            plot_x0 = float(plot_x0)
            plot_x1 = float(plot_x1)
            plot_y0 = float(plot_y0)
            plot_y1 = float(plot_y1)

            if not all(
                np.isfinite(value)
                for value in (
                    raw_x0,
                    raw_x1,
                    raw_y0,
                    raw_y1,
                    plot_x0,
                    plot_x1,
                    plot_y0,
                    plot_y1,
                )
            ):
                raise ValueError("Aggregate ranges are not finite")
            if plot_x1 <= plot_x0 or plot_y1 <= plot_y0:
                raise ValueError("Invalid aggregate range")

            x_width = float(plot_x1 - plot_x0) / float(bins)
            y_width = float(plot_y1 - plot_y0) / float(bins)
            max_bin = int(bins - 1)
            aggregate_sql = (
                "WITH src AS ("
                f" SELECT CAST({qx} AS DOUBLE) AS x_raw, "
                f" CAST({qy} AS DOUBLE) AS y_raw "
                f" FROM {self._relation_sql()}"
                "), filtered AS ("
                f" SELECT x_raw, y_raw, {x_plot_expr} AS x_plot, "
                f" {y_plot_expr} AS y_plot "
                " FROM src "
                f" WHERE {raw_where_sql}"
                "), binned AS ("
                " SELECT "
                " LEAST(GREATEST(CAST(FLOOR((x_plot - ?) / ?) AS BIGINT), 0), ?) AS xb, "
                " LEAST(GREATEST(CAST(FLOOR((y_plot - ?) / ?) AS BIGINT), 0), ?) AS yb "
                " FROM filtered "
                " WHERE x_plot >= ? AND x_plot <= ? "
                " AND y_plot >= ? AND y_plot <= ?"
                ") "
                "SELECT xb, yb, COUNT(*) AS n "
                "FROM binned GROUP BY xb, yb"
            )
            aggregate_params = [
                self._path_argument(),
                *raw_params,
                plot_x0,
                x_width,
                max_bin,
                plot_y0,
                y_width,
                max_bin,
                plot_x0,
                plot_x1,
                plot_y0,
                plot_y1,
            ]
            binned_df = con.execute(aggregate_sql, aggregate_params).df()
        finally:
            con.close()

        counts = np.zeros((bins, bins), dtype="float64")
        if binned_df is not None and not binned_df.empty:
            for _, row in binned_df.iterrows():
                try:
                    xb = int(row["xb"])
                    yb = int(row["yb"])
                    if 0 <= xb < bins and 0 <= yb < bins:
                        counts[yb, xb] = float(row["n"])
                except Exception:
                    continue

        return {
            "counts": counts,
            "x_edges": np.linspace(plot_x0, plot_x1, bins + 1, dtype=float),
            "y_edges": np.linspace(plot_y0, plot_y1, bins + 1, dtype=float),
            "raw_x_range": (raw_x0, raw_x1),
            "raw_y_range": (raw_y0, raw_y1),
            "plot_x_range": (plot_x0, plot_x1),
            "plot_y_range": (plot_y0, plot_y1),
            "row_count": row_count,
            "backend": self.backend_name,
        }

    def to_pandas(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        select_sql = self._select_sql(columns=columns)
        sql = f"SELECT {select_sql} FROM {self._relation_sql()}"
        sql_params: list[Any] = [self._path_argument()]
        if where_sql:
            sql += f" WHERE {where_sql}"
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
            f"FROM {self._relation_sql()} "
            "LIMIT 1 OFFSET ?"
        )
        con = self._connect()
        try:
            return con.execute(
                sql,
                [self._path_argument(), int(position)],
            ).df()
        finally:
            con.close()

    def _column_dtype_for_lookup(self, column: str) -> str:
        cache = getattr(self, "_dtype_lookup_cache", None)
        if cache is None:
            try:
                cache = self.dtypes()
            except Exception:
                cache = {}
            try:
                self._dtype_lookup_cache = cache
            except Exception:
                pass
        return str(cache.get(str(column), "") or "").upper()

    def _coerce_lookup_value_for_column(
        self,
        column: str,
        value: Any,
    ) -> tuple[Any, bool]:
        """Return a value suitable for native DuckDB comparison."""

        dtype = self._column_dtype_for_lookup(column)
        raw = value
        try:
            if "INT" in dtype:
                return int(raw), True
            if any(
                token in dtype
                for token in ("DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC")
            ):
                return float(raw), True
            if "BOOL" in dtype:
                if isinstance(raw, str):
                    lowered = raw.strip().lower()
                    if lowered in {"true", "1", "yes", "y"}:
                        return True, True
                    if lowered in {"false", "0", "no", "n"}:
                        return False, True
                    return raw, False
                return bool(raw), True
            return str(raw), True
        except Exception:
            return str(raw), False

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

        quoted = _quote_identifier(id_column)
        lookup_value, native_ok = self._coerce_lookup_value_for_column(
            id_column,
            row_id,
        )
        if native_ok:
            try:
                result = self.to_pandas(
                    columns=columns,
                    limit=1,
                    where_sql=f"{quoted} = ?",
                    params=[lookup_value],
                )
                if result is not None and not result.empty:
                    return result
            except Exception:
                pass

        return self.to_pandas(
            columns=columns,
            limit=1,
            where_sql=f"CAST({quoted} AS VARCHAR) = ?",
            params=[str(row_id)],
        )

    def get_rows_by_ids(
        self,
        row_ids: Sequence[Any],
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        ids = [str(row_id) for row_id in (row_ids or [])]
        if not ids:
            return pd.DataFrame(
                columns=self.columns() if columns is None else list(columns)
            )
        if id_column == "Use Index" or id_column not in self.columns():
            return pd.DataFrame(
                columns=self.columns() if columns is None else list(columns)
            )

        seen: set[str] = set()
        ordered_ids: list[str] = []
        for row_id in ids:
            if row_id in seen:
                continue
            seen.add(row_id)
            ordered_ids.append(row_id)

        available_columns = set(self.columns())
        selected_columns = _normalise_columns(columns)
        if selected_columns is None:
            select_sql = "src.*"
        else:
            selected_columns = list(dict.fromkeys([id_column, *selected_columns]))
            selected_columns = [
                col for col in selected_columns if col in available_columns
            ]
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
            f"FROM {self._relation_sql()} AS src "
            "JOIN requested "
            f"ON CAST(src.{_quote_identifier(id_column)} AS VARCHAR) = "
            "requested.__astronomical_lookup_id "
            "ORDER BY requested.__astronomical_lookup_order"
        )

        con = self._connect()
        try:
            return con.execute(
                sql,
                [*values_params, self._path_argument()],
            ).df()
        finally:
            con.close()

    def find_position_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
    ) -> Optional[int]:
        if id_column == "Use Index" or id_column not in self.columns():
            return None

        quoted = _quote_identifier(id_column)
        lookup_value, native_ok = self._coerce_lookup_value_for_column(
            id_column,
            row_id,
        )
        con = self._connect()
        try:
            if native_ok:
                try:
                    result = con.execute(
                        (
                            "SELECT rn FROM ("
                            " SELECT "
                            f" ROW_NUMBER() OVER () - 1 AS rn, {quoted} AS rid "
                            f" FROM {self._relation_sql()}"
                            ") WHERE rid = ? LIMIT 1"
                        ),
                        [self._path_argument(), lookup_value],
                    ).fetchone()
                    if result is not None:
                        return int(result[0])
                except Exception:
                    pass

            result = con.execute(
                (
                    "SELECT rn FROM ("
                    " SELECT "
                    f" ROW_NUMBER() OVER () - 1 AS rn, {quoted} AS rid "
                    f" FROM {self._relation_sql()}"
                    ") WHERE CAST(rid AS VARCHAR) = ? LIMIT 1"
                ),
                [self._path_argument(), str(row_id)],
            ).fetchone()
        finally:
            con.close()

        if result is None:
            return None
        return int(result[0])

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "path": self._path_argument(),
            "dataset_name": self.dataset_name,
            "capabilities": self.capabilities().to_dict(),
        }


@dataclass(frozen=True)
class DatasetColumnOverlay:
    """One named, replaceable column overlay joined by record ID."""

    name: str
    source: DatasetSource
    base_record_id_column: str
    overlay_record_id_column: str
    columns: tuple[str, ...]
    preserve_base_on_missing: bool = False

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("DatasetColumnOverlay.name must not be empty")
        if not str(self.base_record_id_column).strip():
            raise ValueError(
                "DatasetColumnOverlay.base_record_id_column must not be empty"
            )
        if not str(self.overlay_record_id_column).strip():
            raise ValueError(
                "DatasetColumnOverlay.overlay_record_id_column must not be empty"
            )
        available = set(self.source.columns())
        if self.overlay_record_id_column not in available:
            raise KeyError(
                "Overlay source is missing record ID column "
                f"{self.overlay_record_id_column!r}."
            )
        missing = [column for column in self.columns if column not in available]
        if missing:
            raise KeyError(f"Overlay source is missing columns: {missing!r}")


class DatasetColumnOverlaySource(DatasetSource):
    """Expose replaceable sidecar columns through an existing dataset ID.

    The base dataset remains canonical and keeps its row count and order. Named
    overlays are joined in bounded batches by record ID. Replacing an overlay
    does not create a new dataset and does not nest wrapper sources.
    """

    backend_name = "column_overlay"

    def __init__(
        self,
        base_source: DatasetSource,
        overlays: Sequence[DatasetColumnOverlay] = (),
    ) -> None:
        base_source = coerce_dataset_source(base_source)
        inherited: list[DatasetColumnOverlay] = []
        if isinstance(base_source, DatasetColumnOverlaySource):
            inherited = list(base_source.overlays)
            base_source = base_source.base_source

        self.base_source = base_source
        merged: dict[str, DatasetColumnOverlay] = {
            overlay.name: overlay for overlay in inherited
        }
        for overlay in overlays:
            merged[str(overlay.name)] = overlay
        self.overlays = tuple(merged.values())
        self._validate_overlays()

    @classmethod
    def from_source(cls, source: DatasetSource) -> "DatasetColumnOverlaySource":
        if isinstance(source, cls):
            return source
        return cls(source)

    def with_overlay(
        self,
        *,
        name: str,
        source: DatasetSource,
        base_record_id_column: str,
        overlay_record_id_column: str,
        columns: Sequence[str],
        preserve_base_on_missing: bool = False,
    ) -> "DatasetColumnOverlaySource":
        overlay = DatasetColumnOverlay(
            name=str(name),
            source=coerce_dataset_source(source),
            base_record_id_column=str(base_record_id_column),
            overlay_record_id_column=str(overlay_record_id_column),
            columns=tuple(dict.fromkeys(str(column) for column in columns)),
            preserve_base_on_missing=bool(preserve_base_on_missing),
        )
        retained = [
            existing for existing in self.overlays if existing.name != overlay.name
        ]
        retained.append(overlay)
        return DatasetColumnOverlaySource(self.base_source, retained)

    def without_overlay(self, name: str) -> DatasetSource:
        name = str(name)
        if all(overlay.name != name for overlay in self.overlays):
            return self
        retained = [
            overlay for overlay in self.overlays if overlay.name != name
        ]
        if not retained:
            return self.base_source
        return DatasetColumnOverlaySource(self.base_source, retained)

    def overlay_names(self) -> list[str]:
        return [overlay.name for overlay in self.overlays]

    def columns(self) -> list[str]:
        result = list(self.base_source.columns())
        for overlay in self.overlays:
            for column in overlay.columns:
                if column not in result:
                    result.append(column)
        return result

    def row_count(self) -> Optional[int]:
        return self.base_source.row_count()

    def dtypes(self) -> dict[str, str]:
        result = dict(self.base_source.dtypes())
        for overlay in self.overlays:
            overlay_dtypes = overlay.source.dtypes()
            for column in overlay.columns:
                result[column] = overlay_dtypes.get(column, "object")
        return result

    def capabilities(self) -> DatasetCapabilities:
        base = self.base_source.capabilities()
        return DatasetCapabilities(
            batch_scan=bool(base.batch_scan),
            batch_lookup_by_id=bool(base.batch_lookup_by_id),
            filtered_scan=False,
            ordered_scan=bool(base.ordered_scan),
            sharded_scan=bool(base.sharded_scan),
            bounded_distinct=True,
            column_statistics=True,
        )

    def iter_batches(self, scan: DatasetScan) -> Iterator[DatasetBatch]:
        requested = self._validate_columns(scan.columns)
        base_columns = self._base_columns_for(requested)
        if scan.where_sql and self._predicate_mentions_overlay(scan.where_sql):
            raise NotImplementedError(
                "Filtering a column-overlay dataset by overlay columns is not "
                "supported by the generic source."
            )
        base_scan = DatasetScan(
            columns=tuple(base_columns),
            batch_size=int(scan.batch_size),
            where_sql=scan.where_sql,
            params=tuple(scan.params or ()),
            limit=scan.limit,
            shard_index=int(scan.shard_index),
            shard_count=int(scan.shard_count),
        )
        for batch in self.base_source.iter_batches(base_scan):
            yield DatasetBatch(
                frame=self._join_frame(
                    batch.frame,
                    row_offset=batch.row_offset,
                    requested=requested,
                ),
                batch_index=batch.batch_index,
                row_offset=batch.row_offset,
            )

    def to_pandas(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        requested = self._validate_columns(columns)
        frames = [
            batch.frame
            for batch in self.iter_batches(
                DatasetScan(
                    columns=tuple(requested) if requested is not None else None,
                    batch_size=8192,
                    where_sql=where_sql,
                    params=tuple(params or ()),
                    limit=limit,
                )
            )
        ]
        if not frames:
            return pd.DataFrame(columns=requested or self.columns())
        return pd.concat(frames, ignore_index=True)

    def get_row_by_position(
        self,
        position: int,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        requested = self._validate_columns(columns)
        frame = self.base_source.get_row_by_position(
            int(position),
            columns=self._base_columns_for(requested),
        )
        return self._join_frame(
            frame,
            row_offset=max(0, int(position)),
            requested=requested,
        )

    def get_row_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        requested = self._validate_columns(columns)
        frame = self.base_source.get_row_by_id(
            row_id,
            id_column=id_column,
            columns=self._base_columns_for(requested),
        )
        return self._join_frame(frame, row_offset=0, requested=requested)

    def get_rows_by_ids(
        self,
        row_ids: Sequence[Any],
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        requested = self._validate_columns(columns)
        frame = self.base_source.get_rows_by_ids(
            row_ids,
            id_column=id_column,
            columns=self._base_columns_for(requested),
        )
        return self._join_frame(frame, row_offset=0, requested=requested)

    def find_position_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
    ) -> Optional[int]:
        return self.base_source.find_position_by_id(
            row_id,
            id_column=id_column,
        )

    def count_where(
        self,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> Optional[int]:
        if where_sql and self._predicate_mentions_overlay(where_sql):
            raise NotImplementedError(
                "Filtering a column-overlay dataset by overlay columns is not "
                "supported by the generic source."
            )
        return self.base_source.count_where(
            where_sql=where_sql,
            params=params,
        )

    def distinct_values(
        self,
        column: str,
        *,
        limit: int = 100,
        include_null: bool = False,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> DatasetDistinctResult:
        column = str(column)
        overlay = self._overlay_for_column(column)
        if overlay is None:
            return self.base_source.distinct_values(
                column,
                limit=limit,
                include_null=include_null,
                where_sql=where_sql,
                params=params,
            )
        if where_sql:
            raise NotImplementedError(
                "Filtered distinct values are not supported for overlay columns."
            )
        result = overlay.source.distinct_values(
            column,
            limit=limit,
            include_null=include_null,
        )
        total = self.row_count()
        overlay_rows = overlay.source.row_count()
        missing = (
            None
            if total is None or overlay_rows is None
            else max(0, int(total) - int(overlay_rows))
        )
        values = list(result.values)
        null_count = result.null_count
        if missing:
            null_count = int(null_count or 0) + missing
            if include_null and None not in values and len(values) < int(limit):
                values.append(None)
        return DatasetDistinctResult(
            column=column,
            values=tuple(values[: max(0, int(limit))]),
            truncated=bool(result.truncated),
            scanned_rows=None if total is None else int(total),
            null_count=null_count,
            backend=self.backend_name,
        )

    def column_statistics(
        self,
        column: str,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> DatasetColumnStatistics:
        column = str(column)
        overlay = self._overlay_for_column(column)
        if overlay is None:
            return self.base_source.column_statistics(
                column,
                where_sql=where_sql,
                params=params,
            )
        if where_sql:
            raise NotImplementedError(
                "Filtered statistics are not supported for overlay columns."
            )
        result = overlay.source.column_statistics(column)
        total = self.row_count()
        row_count = int(total) if total is not None else int(result.row_count)
        non_null = int(result.non_null_count)
        return DatasetColumnStatistics(
            column=column,
            row_count=row_count,
            non_null_count=non_null,
            null_count=max(0, row_count - non_null),
            distinct_count=result.distinct_count,
            minimum=result.minimum,
            maximum=result.maximum,
            mean=result.mean,
            stddev=result.stddev,
            backend=self.backend_name,
        )

    def sample_points(
        self,
        *,
        x_col: str,
        y_col: str,
        record_id_col: Optional[str] = None,
        limit: int = 10_000,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        log_x: bool = False,
        log_y: bool = False,
        seed: int = 0,
    ) -> dict[str, Any]:
        if (
            self._overlay_for_column(x_col) is None
            and self._overlay_for_column(y_col) is None
        ):
            return self.base_source.sample_points(
                x_col=x_col,
                y_col=y_col,
                record_id_col=record_id_col,
                limit=limit,
                x_range=x_range,
                y_range=y_range,
                log_x=log_x,
                log_y=log_y,
                seed=seed,
            )
        limit = max(1, int(limit))
        columns = [x_col, y_col]
        if record_id_col and record_id_col != "Use Index":
            columns.append(str(record_id_col))
        rng = np.random.default_rng(int(seed))
        reservoir: list[tuple[float, float, Any]] = []
        eligible = 0
        for batch in self.iter_batches(
            DatasetScan(columns=tuple(dict.fromkeys(columns)), batch_size=8192)
        ):
            frame = batch.frame
            x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
            if record_id_col and record_id_col != "Use Index":
                row_ids = frame[record_id_col].tolist()
            else:
                row_ids = list(
                    range(batch.row_offset, batch.row_offset + len(frame))
                )
            mask = np.isfinite(x) & np.isfinite(y)
            if log_x:
                mask &= x > 0
            if log_y:
                mask &= y > 0
            if x_range is not None:
                lo, hi = sorted((float(x_range[0]), float(x_range[1])))
                mask &= (x >= lo) & (x <= hi)
            if y_range is not None:
                lo, hi = sorted((float(y_range[0]), float(y_range[1])))
                mask &= (y >= lo) & (y <= hi)
            for index in np.flatnonzero(mask):
                eligible += 1
                item = (float(x[index]), float(y[index]), row_ids[index])
                if len(reservoir) < limit:
                    reservoir.append(item)
                    continue
                replacement = int(rng.integers(0, eligible))
                if replacement < limit:
                    reservoir[replacement] = item
        return {
            "frame": pd.DataFrame(
                reservoir,
                columns=["__x", "__y", "__row_id"],
            ),
            "row_count": int(eligible),
            "sampled_from": int(eligible),
            "backend": self.backend_name,
        }

    def aggregate_2d(
        self,
        *,
        x_col: str,
        y_col: str,
        bins: int,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        log_x: bool = False,
        log_y: bool = False,
    ) -> dict[str, Any]:
        if (
            self._overlay_for_column(x_col) is None
            and self._overlay_for_column(y_col) is None
        ):
            return self.base_source.aggregate_2d(
                x_col=x_col,
                y_col=y_col,
                bins=bins,
                x_range=x_range,
                y_range=y_range,
                log_x=log_x,
                log_y=log_y,
            )
        frame = self.to_pandas(columns=[x_col, y_col])
        return PandasDatasetSource(frame).aggregate_2d(
            x_col=x_col,
            y_col=y_col,
            bins=bins,
            x_range=x_range,
            y_range=y_range,
            log_x=log_x,
            log_y=log_y,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "base_backend": getattr(
                self.base_source,
                "backend_name",
                "unknown",
            ),
            "overlays": [
                {
                    "name": overlay.name,
                    "backend": getattr(
                        overlay.source,
                        "backend_name",
                        "unknown",
                    ),
                    "columns": list(overlay.columns),
                    "base_record_id_column": overlay.base_record_id_column,
                    "overlay_record_id_column": (
                        overlay.overlay_record_id_column
                    ),
                    "preserve_base_on_missing": (
                        overlay.preserve_base_on_missing
                    ),
                }
                for overlay in self.overlays
            ],
            "capabilities": self.capabilities().to_dict(),
        }

    def _validate_overlays(self) -> None:
        base_columns = set(self.base_source.columns())
        for overlay in self.overlays:
            if overlay.base_record_id_column not in base_columns:
                raise KeyError(
                    "Base source is missing record ID column "
                    f"{overlay.base_record_id_column!r} for overlay "
                    f"{overlay.name!r}."
                )

    def _validate_columns(
        self,
        columns: Optional[Sequence[str]],
    ) -> Optional[list[str]]:
        if columns is None:
            return None
        requested = [str(column) for column in columns]
        missing = [
            column for column in requested if column not in set(self.columns())
        ]
        if missing:
            raise KeyError(f"Unknown dataset columns: {missing!r}")
        return requested

    def _overlay_for_column(
        self,
        column: str,
    ) -> Optional[DatasetColumnOverlay]:
        result = None
        for overlay in self.overlays:
            if str(column) in overlay.columns:
                result = overlay
        return result

    def _base_columns_for(
        self,
        requested: Optional[Sequence[str]],
    ) -> list[str]:
        base_columns = list(self.base_source.columns())
        if requested is None:
            selected = list(base_columns)
            needed_overlays = list(self.overlays)
        else:
            selected = [
                column
                for column in requested
                if column in base_columns
                and self._overlay_for_column(column) is None
            ]
            needed_overlays = [
                overlay
                for overlay in self.overlays
                if any(column in overlay.columns for column in requested)
            ]
            for column in requested:
                overlay = self._overlay_for_column(column)
                if (
                    overlay is not None
                    and overlay.preserve_base_on_missing
                    and column in base_columns
                ):
                    selected.append(column)
        for overlay in needed_overlays:
            selected.append(overlay.base_record_id_column)
        return list(dict.fromkeys(selected))

    def _join_frame(
        self,
        frame: pd.DataFrame,
        *,
        row_offset: int,
        requested: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        output_columns = (
            list(requested) if requested is not None else self.columns()
        )
        if frame is None or frame.empty:
            return pd.DataFrame(columns=output_columns)

        result = frame.copy()
        for overlay in self.overlays:
            wanted = [
                column
                for column in output_columns
                if column in overlay.columns
            ]
            if not wanted:
                continue
            row_ids = result[overlay.base_record_id_column].tolist()
            overlay_frame = overlay.source.get_rows_by_ids(
                row_ids,
                id_column=overlay.overlay_record_id_column,
                columns=[overlay.overlay_record_id_column, *wanted],
            )
            lookup: dict[str, dict[str, Any]] = {}
            if overlay_frame is not None and not overlay_frame.empty:
                keys = overlay_frame[overlay.overlay_record_id_column].map(str)
                if keys.duplicated().any():
                    overlay_frame = overlay_frame.loc[
                        ~keys.duplicated(keep="last")
                    ].copy()
                    keys = overlay_frame[
                        overlay.overlay_record_id_column
                    ].map(str)
                for index, key in enumerate(keys):
                    lookup[str(key)] = {
                        column: overlay_frame.iloc[index][column]
                        for column in wanted
                    }
            for column in wanted:
                existing = (
                    result[column].tolist()
                    if (
                        overlay.preserve_base_on_missing
                        and column in result.columns
                    )
                    else [None] * len(result)
                )
                values = []
                for index, row_id in enumerate(row_ids):
                    record = lookup.get(str(row_id))
                    values.append(
                        record[column]
                        if record is not None
                        else existing[index]
                    )
                result[column] = values

        for column in output_columns:
            if column not in result.columns:
                result[column] = None
        return result.loc[:, output_columns].reset_index(drop=True)

    def _predicate_mentions_overlay(self, where_sql: str) -> bool:
        text = str(where_sql)
        return any(
            column in text
            for overlay in self.overlays
            for column in overlay.columns
        )



def coerce_dataset_source(obj: Any) -> DatasetSource:
    """Convert known dataset-like objects into DatasetSource instances."""

    if isinstance(obj, DatasetSource):
        return obj
    if isinstance(obj, pd.DataFrame):
        return PandasDatasetSource(obj)
    raise TypeError(
        "Unsupported dataset source. Expected DatasetSource or pandas.DataFrame, "
        f"got {type(obj)!r}."
    )