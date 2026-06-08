from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def _normalise_columns(columns: Optional[Sequence[str]]) -> Optional[list[str]]:
    if columns is None:
        return None
    return [str(col) for col in columns]

def _quote_identifier(identifier: str) -> str:
    """
    Quote a SQL identifier for DuckDB.

    This is for column names, not values. Values should still be passed as
    query parameters.
    """
    return '"' + str(identifier).replace('"', '""') + '"'


class DatasetSource(ABC):
    """
    Backend-neutral dataset access contract.

    DatasetSource is the new internal platform contract. A source may be backed
    by pandas, Parquet/DuckDB, Polars, Dask, Arrow, or a future remote/catalogue
    backend.

    Important rule:
        pandas is allowed as a returned materialized view, but it should no
        longer be the canonical stored dataset object.
    """

    backend_name: str = "unknown"

    @abstractmethod
    def columns(self) -> list[str]:
        """Return dataset column names without materializing the full dataset."""

    @abstractmethod
    def row_count(self) -> Optional[int]:
        """Return row count if cheaply available."""

    def count_where(
        self,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> Optional[int]:
        """
        Return a filtered row count if the backend can do it cheaply.

        This is intentionally separate from to_pandas(...), because selection
        tools need counts without materialising selected rows.
        """
        if not where_sql:
            count = self.row_count()
            return None if count is None else int(count)

        raise NotImplementedError(
            f"{type(self).__name__} does not implement count_where()"
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
        """Return a small sampled point cloud for scatter first-paint.

        Returned dictionary keys:
        - frame: pandas.DataFrame with columns "__x", "__y", "__row_id"
        - row_count: number of finite rows matching filters/ranges
        - sampled_from: same as row_count when sampling was applied
        - backend: backend name

        This avoids materialising x/y/row-id arrays for the entire dataset just to
        draw an interactive sample.
        """
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
        """Return a small 2D count grid for density rendering.

        Returned dictionary keys:
        - counts: np.ndarray shaped as (y_bins, x_bins)
        - x_edges: np.ndarray length bins + 1 in plot coordinates
        - y_edges: np.ndarray length bins + 1 in plot coordinates
        - raw_x_range: original data-space x range
        - raw_y_range: original data-space y range
        - plot_x_range: displayed x range, log10-transformed when log_x=True
        - plot_y_range: displayed y range, log10-transformed when log_y=True
        - row_count: number of finite rows included in the aggregate
        - backend: backend name
        """
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
        """
        Materialize a pandas view.

        This method exists for compatibility with old plugins. New plugins
        should request only the columns/rows they need.
        """

    @abstractmethod
    def dtypes(self) -> dict[str, str]:
        """
        Return column dtype information without materialising the full dataset.
        """
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
            return pd.DataFrame(columns=self.columns() if columns is None else list(columns))

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
        return {"backend": self.backend_name}


class PandasDatasetSource(DatasetSource):
    """
    Compatibility source for existing in-memory pandas DataFrames.

    This lets the rest of the app migrate gradually.
    """

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
            # Deterministic, cheap, evenly spaced first-paint sample.
            take_idx = np.linspace(0, row_count - 1, num=limit, dtype=np.int64)
            positions = positions[take_idx]

        if record_id_col and record_id_col != "Use Index" and record_id_col in self._df.columns:
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

        # Return a copy so old plugins cannot mutate the canonical source
        # accidentally. For large data, new plugins should not call get_df().
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
                return pd.DataFrame(columns=self.columns() if columns is None else columns)
            mask = self._df[id_column].astype(str) == str(row_id)

        matches = self._df.loc[mask]
        if matches.empty:
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        selected_columns = _normalise_columns(columns)
        if selected_columns is not None:
            selected_columns = [col for col in selected_columns if col in matches.columns]
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
            return pd.DataFrame(columns=self.columns() if columns is None else list(columns))

        requested_order = {row_id: i for i, row_id in enumerate(ids)}

        if id_column == "Use Index":
            values = self._df.index.astype(str)
            matches = self._df.loc[values.isin(set(ids))].copy()
            if matches.empty:
                return pd.DataFrame(columns=self.columns() if columns is None else list(columns))
            matches["__astronomical_lookup_id"] = matches.index.astype(str)
            order_source = matches["__astronomical_lookup_id"]
        else:
            if id_column not in self._df.columns:
                return pd.DataFrame(columns=self.columns() if columns is None else list(columns))
            values = self._df[id_column].astype(str)
            matches = self._df.loc[values.isin(set(ids))].copy()
            if matches.empty:
                return pd.DataFrame(columns=self.columns() if columns is None else list(columns))
            order_source = matches[id_column].astype(str)

        matches["__astronomical_lookup_order"] = order_source.map(requested_order)
        matches = matches.sort_values("__astronomical_lookup_order", kind="stable")

        selected_columns = _normalise_columns(columns)
        if selected_columns is not None:
            selected_columns = [col for col in selected_columns if col in matches.columns]
            matches = matches.loc[:, selected_columns]

        return matches.reset_index(drop=True).copy()

    def find_position_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
    ) -> Optional[int]:
        import numpy as np

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
            [str(col) for col in columns_hint] if columns_hint is not None else None
        )
        self._row_count_cache: Optional[int] = (
            int(row_count_hint) if row_count_hint is not None else None
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
        if selected_columns is None:
            return "*"
        if not selected_columns:
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
        """Return a bounded scatter sample directly from DuckDB/Parquet.

        This intentionally avoids ORDER BY random(), which would sort millions of
        rows. Instead it scans once, counts the filtered rows, and takes a stable
        stride sample using row_number modulo stride.
        """
        limit = max(1, int(limit))

        available = set(self.columns())
        if x_col not in available or y_col not in available:
            raise KeyError(f"Unknown sample columns: {x_col!r}, {y_col!r}")

        qx = _quote_identifier(str(x_col))
        qy = _quote_identifier(str(y_col))

        if record_id_col and record_id_col != "Use Index" and record_id_col in available:
            qid = _quote_identifier(str(record_id_col))
            row_id_expr = f"CAST({qid} AS VARCHAR)"
        else:
            row_id_expr = "CAST(__rownum AS VARCHAR)"

        where_clauses = [
            "isfinite(__x)",
            "isfinite(__y)",
        ]
        params: list[Any] = []

        if log_x:
            where_clauses.append("__x > 0")
        if log_y:
            where_clauses.append("__y > 0")

        if x_range is not None:
            x0, x1 = float(min(x_range)), float(max(x_range))
            where_clauses.append("__x >= ?")
            where_clauses.append("__x <= ?")
            params.extend([x0, x1])

        if y_range is not None:
            y0, y1 = float(min(y_range)), float(max(y_range))
            where_clauses.append("__y >= ?")
            where_clauses.append("__y <= ?")
            params.extend([y0, y1])

        where_sql = " AND ".join(where_clauses)

        sql = (
            "WITH src AS ("
            "  SELECT "
            f"    CAST({qx} AS DOUBLE) AS __x, "
            f"    CAST({qy} AS DOUBLE) AS __y, "
            "    row_number() OVER () - 1 AS __rownum, "
            f"    {row_id_expr} AS __row_id "
            f"  FROM {self._relation_sql()}"
            "), filtered AS ("
            "  SELECT __x, __y, __row_id, __rownum "
            "  FROM src "
            f"  WHERE {where_sql}"
            "), sampled AS ("
            "  SELECT __x, __y, __row_id "
            "  FROM filtered "
            "  WHERE MOD(__rownum, ?) = 0 "
            "  LIMIT ?"
            ") "
            "SELECT __x, __y, __row_id "
            "FROM sampled"
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

        raw_where = [
            "isfinite(x_raw)",
            "isfinite(y_raw)",
        ]
        raw_params: list[Any] = []

        if log_x:
            raw_where.append("x_raw > 0")
        if log_y:
            raw_where.append("y_raw > 0")

        if x_range is not None:
            x0_raw = float(min(x_range))
            x1_raw = float(max(x_range))
            raw_where.append("x_raw >= ?")
            raw_where.append("x_raw <= ?")
            raw_params.extend([x0_raw, x1_raw])

        if y_range is not None:
            y0_raw = float(min(y_range))
            y1_raw = float(max(y_range))
            raw_where.append("y_raw >= ?")
            raw_where.append("y_raw <= ?")
            raw_params.extend([y0_raw, y1_raw])

        raw_where_sql = " AND ".join(raw_where)

        extent_sql = (
            "WITH src AS ("
            f"  SELECT CAST({qx} AS DOUBLE) AS x_raw, "
            f"         CAST({qy} AS DOUBLE) AS y_raw "
            f"  FROM {self._relation_sql()}"
            "), filtered AS ("
            f"  SELECT x_raw, y_raw, "
            f"         {x_plot_expr} AS x_plot, "
            f"         {y_plot_expr} AS y_plot "
            "  FROM src "
            f"  WHERE {raw_where_sql}"
            ") "
            "SELECT "
            "  MIN(x_raw) AS raw_x_min, "
            "  MAX(x_raw) AS raw_x_max, "
            "  MIN(y_raw) AS raw_y_min, "
            "  MAX(y_raw) AS raw_y_max, "
            "  MIN(x_plot) AS plot_x_min, "
            "  MAX(x_plot) AS plot_x_max, "
            "  MIN(y_plot) AS plot_y_min, "
            "  MAX(y_plot) AS plot_y_max, "
            "  COUNT(*) AS n "
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
                f"  SELECT CAST({qx} AS DOUBLE) AS x_raw, "
                f"         CAST({qy} AS DOUBLE) AS y_raw "
                f"  FROM {self._relation_sql()}"
                "), filtered AS ("
                f"  SELECT x_raw, y_raw, "
                f"         {x_plot_expr} AS x_plot, "
                f"         {y_plot_expr} AS y_plot "
                "  FROM src "
                f"  WHERE {raw_where_sql}"
                "), binned AS ("
                "  SELECT "
                "    LEAST(GREATEST(CAST(FLOOR((x_plot - ?) / ?) AS BIGINT), 0), ?) AS xb, "
                "    LEAST(GREATEST(CAST(FLOOR((y_plot - ?) / ?) AS BIGINT), 0), ?) AS yb "
                "  FROM filtered "
                "  WHERE x_plot >= ? AND x_plot <= ? "
                "    AND y_plot >= ? AND y_plot <= ?"
                ") "
                "SELECT xb, yb, COUNT(*) AS n "
                "FROM binned "
                "GROUP BY xb, yb"
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

        # DuckDB supports LIMIT/OFFSET. This is acceptable for the first refactor.
        # Later, for very large random-access browsing, add a row_number cache or
        # an indexed record-id lookup table.
        sql = (
            f"SELECT {select_sql} "
            f"FROM {self._relation_sql()} "
            "LIMIT 1 OFFSET ?"
        )

        con = self._connect()
        try:
            return con.execute(sql, [self._path_argument(), int(position)]).df()
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

    def _coerce_lookup_value_for_column(self, column: str, value: Any) -> tuple[Any, bool]:
        """Return a value suitable for native DuckDB comparison.

        The boolean says whether native comparison is safe. If coercion fails,
        callers should fall back to CAST(column AS VARCHAR) = ?.
        """
        dtype = self._column_dtype_for_lookup(column)
        raw = value

        try:
            if "INT" in dtype:
                return int(raw), True

            if any(token in dtype for token in ("DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC")):
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

            # Strings and unknown object-like values can still use direct
            # equality with a string parameter.
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
        lookup_value, native_ok = self._coerce_lookup_value_for_column(id_column, row_id)

        # Fast path: preserve the column's native type so DuckDB/Parquet can
        # use predicate pushdown/statistics where possible.
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

        # Compatibility fallback for mixed/stringified IDs.
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
            return pd.DataFrame(columns=self.columns() if columns is None else list(columns))

        if id_column == "Use Index" or id_column not in self.columns():
            return pd.DataFrame(columns=self.columns() if columns is None else list(columns))

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
            f"FROM {self._relation_sql()} AS src "
            "JOIN requested "
            f"ON CAST(src.{_quote_identifier(id_column)} AS VARCHAR) = requested.__astronomical_lookup_id "
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
        lookup_value, native_ok = self._coerce_lookup_value_for_column(id_column, row_id)

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
                            ") "
                            "WHERE rid = ? "
                            "LIMIT 1"
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
                    ") "
                    "WHERE CAST(rid AS VARCHAR) = ? "
                    "LIMIT 1"
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
        }


def coerce_dataset_source(obj: Any) -> DatasetSource:
    """
    Convert known dataset-like objects into DatasetSource instances.
    """
    if isinstance(obj, DatasetSource):
        return obj

    if isinstance(obj, pd.DataFrame):
        return PandasDatasetSource(obj)

    raise TypeError(
        "Unsupported dataset source. Expected DatasetSource or pandas.DataFrame, "
        f"got {type(obj)!r}."
    )