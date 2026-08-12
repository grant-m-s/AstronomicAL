from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from astronomicAL.platform.dataset_sources import (
    DatasetBatch,
    DatasetCapabilities,
    DatasetColumnStatistics,
    DatasetDistinctResult,
    DatasetScan,
    DatasetSource,
)


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _quote_sql_string(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _python_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


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


@dataclass(frozen=True)
class UnionBranch:
    """One lazily referenced input relation in a by-name vertical union."""

    dataset_id: str
    source: Any
    query_sql: str
    params: tuple[Any, ...]
    columns: tuple[str, ...]
    row_count: Optional[int] = None


class LazyUnionByNameDuckDBSource(DatasetSource):
    """Lazy ``UNION ALL BY NAME`` over existing DuckDB relation sources.

    No input table is materialised and no combined Parquet file is written.
    Each branch remains backed by its original DatasetSource. DuckDB aligns
    columns by name and supplies NULL for columns that are absent from a branch.
    """

    backend_name = "duckdb_union_by_name"

    def __init__(
        self,
        *,
        branches: Sequence[UnionBranch],
        dataset_name: Optional[str] = None,
        columns_hint: Optional[Sequence[str]] = None,
        row_count_hint: Optional[int] = None,
        source_column_name: Optional[str] = None,
    ) -> None:
        normalised = tuple(branches or ())
        if len(normalised) < 2:
            raise ValueError("A lazy union requires at least two input datasets.")

        for branch in normalised:
            if not str(branch.dataset_id or "").strip():
                raise ValueError("Union branch dataset IDs must be non-empty.")
            if branch.source is None:
                raise ValueError(
                    f"Union branch {branch.dataset_id!r} does not have a source."
                )
            if not str(branch.query_sql or "").strip():
                raise ValueError(
                    f"Union branch {branch.dataset_id!r} does not expose relation SQL."
                )

        self.branches = normalised
        self.dataset_name = dataset_name
        self.source_column_name = (
            str(source_column_name).strip()
            if source_column_name not in (None, "")
            else None
        )

        self._column_cache: Optional[list[str]] = (
            [str(column) for column in columns_hint]
            if columns_hint is not None
            else None
        )
        if (
            self.source_column_name
            and self._column_cache is not None
            and self.source_column_name not in self._column_cache
        ):
            self._column_cache.append(self.source_column_name)

        self._dtype_cache: Optional[dict[str, str]] = None
        self._row_count_cache: Optional[int] = (
            int(row_count_hint) if row_count_hint is not None else None
        )

    # ------------------------------------------------------------------
    # DuckDB relation contract used by Table Tools lazy transforms
    # ------------------------------------------------------------------

    def _connect(self):
        connect = getattr(self.branches[0].source, "_connect", None)
        if not callable(connect):
            raise RuntimeError(
                "The first union source does not expose a DuckDB connection."
            )
        return connect()

    def _relation_params(self) -> list[Any]:
        params: list[Any] = []
        for branch in self.branches:
            params.extend(list(branch.params))
        return params

    def _relation_sql(self) -> str:
        statements: list[str] = []

        for index, branch in enumerate(self.branches):
            alias = _quote_identifier(f"union_branch_{index}")
            select = f"SELECT {alias}.*"

            if self.source_column_name:
                select += (
                    f", {_quote_sql_string(branch.dataset_id)} "
                    f"AS {_quote_identifier(self.source_column_name)}"
                )

            select += f" FROM ({branch.query_sql}) AS {alias}"
            statements.append(select)

        return " UNION ALL BY NAME ".join(statements)

    def _from_sql(self, alias: str = "src") -> str:
        return f"({self._relation_sql()}) AS {_quote_identifier(alias)}"

    # ------------------------------------------------------------------
    # DatasetSource contract
    # ------------------------------------------------------------------

    def capabilities(self) -> DatasetCapabilities:
        return DatasetCapabilities(
            batch_scan=True,
            batch_lookup_by_id=True,
            filtered_scan=True,
            ordered_scan=False,
            sharded_scan=False,
            bounded_distinct=True,
            column_statistics=True,
        )

    def columns(self) -> list[str]:
        if self._column_cache is None:
            self._refresh_schema()
        return list(self._column_cache or [])

    def dtypes(self) -> dict[str, str]:
        if self._dtype_cache is None:
            self._refresh_schema()
        return dict(self._dtype_cache or {})

    def _refresh_schema(self) -> None:
        con = self._connect()
        try:
            frame = con.execute(
                f"DESCRIBE SELECT * FROM {self._from_sql('schema_src')} LIMIT 0",
                self._relation_params(),
            ).df()
        except Exception as exc:
            raise ValueError(
                "The selected datasets cannot be combined by name because their "
                f"schemas contain incompatible shared column types: {exc}"
            ) from exc
        finally:
            con.close()

        columns = [str(value) for value in frame["column_name"].tolist()]
        dtypes = {
            str(row["column_name"]): str(row["column_type"])
            for _, row in frame.iterrows()
        }
        self._column_cache = columns
        self._dtype_cache = dtypes

    def row_count(self) -> Optional[int]:
        if self._row_count_cache is not None:
            return int(self._row_count_cache)

        if all(branch.row_count is not None for branch in self.branches):
            self._row_count_cache = sum(
                int(branch.row_count or 0) for branch in self.branches
            )
            return int(self._row_count_cache)

        con = self._connect()
        try:
            value = con.execute(
                f"SELECT COUNT(*) FROM {self._from_sql('count_src')}",
                self._relation_params(),
            ).fetchone()
        finally:
            con.close()

        self._row_count_cache = int(value[0]) if value is not None else 0
        return int(self._row_count_cache)

    def _selected_columns_sql(
        self,
        columns: Optional[Sequence[str]],
    ) -> str:
        if columns is None:
            return "*"

        selected = [str(column) for column in columns]
        if not selected:
            return "*"

        available = set(self.columns())
        missing = [column for column in selected if column not in available]
        if missing:
            raise KeyError(f"Unknown union columns: {missing!r}")

        return ", ".join(_quote_identifier(column) for column in selected)

    def _select_query(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
        limit: Optional[int] = None,
        alias: str = "src",
    ) -> tuple[str, list[Any]]:
        sql = (
            f"SELECT {self._selected_columns_sql(columns)} "
            f"FROM {self._from_sql(alias)}"
        )
        sql_params = self._relation_params()

        if where_sql:
            sql += f" WHERE ({where_sql})"
            sql_params.extend(list(params or ()))
        elif params:
            raise ValueError("params requires where_sql")

        if limit is not None:
            sql += " LIMIT ?"
            sql_params.append(max(0, int(limit)))

        return sql, sql_params

    def to_pandas(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        sql, sql_params = self._select_query(
            columns=columns,
            where_sql=where_sql,
            params=params,
            limit=limit,
            alias="materialise_src",
        )
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
        return self.to_pandas(columns=columns, limit=max(0, int(n)))

    def iter_batches(self, scan: DatasetScan) -> Iterator[DatasetBatch]:
        if int(scan.shard_count) != 1 or int(scan.shard_index) != 0:
            raise NotImplementedError(
                "LazyUnionByNameDuckDBSource does not advertise sharded scans."
            )

        sql, sql_params = self._select_query(
            columns=scan.columns,
            where_sql=scan.where_sql,
            params=scan.params,
            limit=scan.limit,
            alias="batch_src",
        )

        con = self._connect()
        try:
            cursor = con.execute(sql, sql_params)
            columns = [str(item[0]) for item in (cursor.description or ())]
            batch_size = int(scan.batch_size)
            row_offset = 0
            batch_index = 0

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                frame = pd.DataFrame.from_records(rows, columns=columns)
                yield DatasetBatch(
                    frame=frame,
                    batch_index=batch_index,
                    row_offset=row_offset,
                )
                row_offset += int(len(frame))
                batch_index += 1
        finally:
            con.close()

    def count_where(
        self,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> Optional[int]:
        if not where_sql and not params and self._row_count_cache is not None:
            return int(self._row_count_cache)

        sql = f"SELECT COUNT(*) FROM {self._from_sql('count_where_src')}"
        sql_params = self._relation_params()

        if where_sql:
            sql += f" WHERE ({where_sql})"
            sql_params.extend(list(params or ()))
        elif params:
            raise ValueError("params requires where_sql")

        con = self._connect()
        try:
            result = con.execute(sql, sql_params).fetchone()
        finally:
            con.close()

        count = int(result[0]) if result is not None else 0
        if not where_sql and not params:
            self._row_count_cache = count
        return count

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
        if column not in self.columns():
            raise KeyError(f"Unknown distinct column: {column!r}")

        limit = int(limit)
        if limit < 0:
            raise ValueError("limit must be zero or greater")

        quoted = _quote_identifier(column)
        clauses: list[str] = []
        sql_params = self._relation_params()

        if where_sql:
            clauses.append(f"({where_sql})")
            sql_params.extend(list(params or ()))
        elif params:
            raise ValueError("params requires where_sql")

        if not include_null:
            clauses.append(f"{quoted} IS NOT NULL")

        sql = (
            f"SELECT DISTINCT {quoted} AS value "
            f"FROM {self._from_sql('distinct_src')}"
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " LIMIT ?"
        sql_params.append(limit + 1)

        con = self._connect()
        try:
            rows = con.execute(sql, sql_params).fetchall()
        finally:
            con.close()

        values = [_python_scalar(row[0]) for row in rows]
        truncated = len(values) > limit

        return DatasetDistinctResult(
            column=column,
            values=tuple(values[:limit]),
            truncated=truncated,
            scanned_rows=None,
            null_count=None,
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
        dtypes = self.dtypes()
        if column not in dtypes:
            raise KeyError(f"Unknown statistics column: {column!r}")

        quoted = _quote_identifier(column)
        numeric = _is_numeric_dtype_name(dtypes[column])
        sql_params = self._relation_params()

        select_parts = [
            "COUNT(*) AS row_count",
            f"COUNT({quoted}) AS non_null_count",
            f"COUNT(DISTINCT {quoted}) AS distinct_count",
            f"MIN({quoted}) AS minimum",
            f"MAX({quoted}) AS maximum",
        ]
        if numeric:
            select_parts.extend(
                [
                    f"AVG(CAST({quoted} AS DOUBLE)) AS mean",
                    f"STDDEV_POP(CAST({quoted} AS DOUBLE)) AS stddev",
                ]
            )
        else:
            select_parts.extend(["NULL AS mean", "NULL AS stddev"])

        sql = (
            "SELECT "
            + ", ".join(select_parts)
            + f" FROM {self._from_sql('stats_src')}"
        )
        if where_sql:
            sql += f" WHERE ({where_sql})"
            sql_params.extend(list(params or ()))
        elif params:
            raise ValueError("params requires where_sql")

        con = self._connect()
        try:
            try:
                result = con.execute(sql, sql_params).fetchone()
            except Exception:
                fallback_params = self._relation_params()
                fallback_sql = (
                    "SELECT "
                    "COUNT(*) AS row_count, "
                    f"COUNT({quoted}) AS non_null_count, "
                    f"COUNT(DISTINCT {quoted}) AS distinct_count "
                    f"FROM {self._from_sql('stats_fallback_src')}"
                )
                if where_sql:
                    fallback_sql += f" WHERE ({where_sql})"
                    fallback_params.extend(list(params or ()))
                result = con.execute(fallback_sql, fallback_params).fetchone()
                result = (
                    result[0],
                    result[1],
                    result[2],
                    None,
                    None,
                    None,
                    None,
                )
        finally:
            con.close()

        row_count = int(result[0]) if result is not None else 0
        non_null_count = int(result[1]) if result is not None else 0

        return DatasetColumnStatistics(
            column=column,
            row_count=row_count,
            non_null_count=non_null_count,
            null_count=row_count - non_null_count,
            distinct_count=(
                None if result is None or result[2] is None else int(result[2])
            ),
            minimum=None if result is None else _python_scalar(result[3]),
            maximum=None if result is None else _python_scalar(result[4]),
            mean=(
                None if result is None or result[5] is None else float(result[5])
            ),
            stddev=(
                None if result is None or result[6] is None else float(result[6])
            ),
            backend=self.backend_name,
        )

    def get_row_by_position(
        self,
        position: int,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        position = int(position)
        if position < 0:
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        known_count = self.row_count()
        if known_count is not None and position >= int(known_count):
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        sql = (
            f"SELECT {self._selected_columns_sql(columns)} "
            f"FROM {self._from_sql('position_src')} LIMIT 1 OFFSET ?"
        )
        sql_params = [*self._relation_params(), position]

        con = self._connect()
        try:
            return con.execute(sql, sql_params).df()
        finally:
            con.close()

    def get_row_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        id_column = str(id_column)
        if id_column == "Use Index" or id_column not in self.columns():
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        return self.to_pandas(
            columns=columns,
            limit=1,
            where_sql=f"CAST({_quote_identifier(id_column)} AS VARCHAR) = ?",
            params=[str(row_id)],
        )

    def get_rows_by_ids(
        self,
        row_ids: Sequence[Any],
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        id_column = str(id_column)
        ids = [str(row_id) for row_id in (row_ids or ())]

        if not ids:
            return pd.DataFrame(columns=self.columns() if columns is None else columns)
        if id_column == "Use Index" or id_column not in self.columns():
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        ordered_ids = list(dict.fromkeys(ids))
        selected_columns = list(columns or ())
        if id_column not in selected_columns:
            selected_columns.insert(0, id_column)
        select_sql = self._selected_columns_sql(selected_columns)

        values_sql = ", ".join(["(?, ?)"] * len(ordered_ids))
        values_params: list[Any] = []
        for order, row_id in enumerate(ordered_ids):
            values_params.extend([order, row_id])

        sql = (
            "WITH requested(__astronomical_lookup_order, __astronomical_lookup_id) AS "
            f"(VALUES {values_sql}) "
            f"SELECT {select_sql} "
            f"FROM {self._from_sql('lookup_src')} "
            "JOIN requested ON "
            f"CAST(lookup_src.{_quote_identifier(id_column)} AS VARCHAR) = "
            "requested.__astronomical_lookup_id "
            "ORDER BY requested.__astronomical_lookup_order"
        )

        con = self._connect()
        try:
            return con.execute(
                sql,
                [*values_params, *self._relation_params()],
            ).df()
        finally:
            con.close()

    def find_position_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
    ) -> Optional[int]:
        id_column = str(id_column)
        if id_column == "Use Index" or id_column not in self.columns():
            return None

        quoted = _quote_identifier(id_column)
        sql = (
            "SELECT rn FROM ("
            " SELECT ROW_NUMBER() OVER () - 1 AS rn, "
            f" {quoted} AS rid "
            f" FROM {self._from_sql('position_lookup_src')}"
            ") AS numbered "
            "WHERE CAST(rid AS VARCHAR) = ? LIMIT 1"
        )

        con = self._connect()
        try:
            result = con.execute(
                sql,
                [*self._relation_params(), str(row_id)],
            ).fetchone()
        finally:
            con.close()

        return None if result is None else int(result[0])

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
        del seed

        x_col = str(x_col)
        y_col = str(y_col)
        available = set(self.columns())
        if x_col not in available or y_col not in available:
            raise KeyError(f"Unknown sample columns: {x_col!r}, {y_col!r}")

        limit = max(1, int(limit))
        x_expr = f"TRY_CAST({_quote_identifier(x_col)} AS DOUBLE)"
        y_expr = f"TRY_CAST({_quote_identifier(y_col)} AS DOUBLE)"
        clauses = [f"{x_expr} IS NOT NULL", f"{y_expr} IS NOT NULL"]
        params: list[Any] = []

        if log_x:
            clauses.append(f"{x_expr} > 0")
        if log_y:
            clauses.append(f"{y_expr} > 0")
        if x_range is not None:
            x0, x1 = sorted((float(x_range[0]), float(x_range[1])))
            clauses.append(f"{x_expr} BETWEEN ? AND ?")
            params.extend([x0, x1])
        if y_range is not None:
            y0, y1 = sorted((float(y_range[0]), float(y_range[1])))
            clauses.append(f"{y_expr} BETWEEN ? AND ?")
            params.extend([y0, y1])

        where_sql = " AND ".join(clauses)
        row_count = int(self.count_where(where_sql=where_sql, params=params) or 0)
        if row_count == 0:
            return {
                "frame": pd.DataFrame(columns=["__x", "__y", "__row_id"]),
                "row_count": 0,
                "sampled_from": 0,
                "backend": self.backend_name,
            }

        if (
            record_id_col
            and record_id_col != "Use Index"
            and str(record_id_col) in available
        ):
            row_id_expr = _quote_identifier(str(record_id_col))
        else:
            row_id_expr = "ROW_NUMBER() OVER () - 1"

        sql = (
            f"SELECT {x_expr} AS __x, {y_expr} AS __y, "
            f"{row_id_expr} AS __row_id "
            f"FROM {self._from_sql('sample_src')} "
            f"WHERE ({where_sql}) LIMIT ?"
        )

        con = self._connect()
        try:
            frame = con.execute(
                sql,
                [*self._relation_params(), *params, limit],
            ).df()
        finally:
            con.close()

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
        x_col = str(x_col)
        y_col = str(y_col)
        available = set(self.columns())
        if x_col not in available or y_col not in available:
            raise KeyError(f"Unknown aggregate columns: {x_col!r}, {y_col!r}")

        bins = max(5, min(500, int(bins)))
        requested_x_range = (
            None
            if x_range is None
            else tuple(sorted((float(x_range[0]), float(x_range[1]))))
        )
        requested_y_range = (
            None
            if y_range is None
            else tuple(sorted((float(y_range[0]), float(y_range[1]))))
        )

        if log_x and requested_x_range is not None and requested_x_range[0] <= 0:
            raise ValueError("Log X aggregation requires a positive X range")
        if log_y and requested_y_range is not None and requested_y_range[0] <= 0:
            raise ValueError("Log Y aggregation requires a positive Y range")

        data_x_min = np.inf
        data_x_max = -np.inf
        data_y_min = np.inf
        data_y_max = -np.inf
        eligible_rows = 0

        def filtered_xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            if frame is None or frame.empty:
                return np.empty(0, dtype=float), np.empty(0, dtype=float)

            x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)

            if log_x:
                mask &= x > 0
            if log_y:
                mask &= y > 0
            if requested_x_range is not None:
                mask &= x >= requested_x_range[0]
                mask &= x <= requested_x_range[1]
            if requested_y_range is not None:
                mask &= y >= requested_y_range[0]
                mask &= y <= requested_y_range[1]

            return x[mask], y[mask]

        scan = DatasetScan(columns=(x_col, y_col), batch_size=8192)
        for batch in self.iter_batches(scan):
            x, y = filtered_xy(batch.frame)
            if len(x) == 0:
                continue
            eligible_rows += int(len(x))
            data_x_min = min(data_x_min, float(np.min(x)))
            data_x_max = max(data_x_max, float(np.max(x)))
            data_y_min = min(data_y_min, float(np.min(y)))
            data_y_max = max(data_y_max, float(np.max(y)))

        if eligible_rows <= 0:
            raise ValueError("No finite X/Y rows available for 2D aggregation")

        raw_x0, raw_x1 = (
            requested_x_range
            if requested_x_range is not None
            else (float(data_x_min), float(data_x_max))
        )
        raw_y0, raw_y1 = (
            requested_y_range
            if requested_y_range is not None
            else (float(data_y_min), float(data_y_max))
        )

        plot_x0 = float(np.log10(raw_x0)) if log_x else float(raw_x0)
        plot_x1 = float(np.log10(raw_x1)) if log_x else float(raw_x1)
        plot_y0 = float(np.log10(raw_y0)) if log_y else float(raw_y0)
        plot_y1 = float(np.log10(raw_y1)) if log_y else float(raw_y1)

        if plot_x1 <= plot_x0 or plot_y1 <= plot_y0:
            raise ValueError("Invalid aggregate range")

        counts = np.zeros((bins, bins), dtype="float64")
        for batch in self.iter_batches(scan):
            x, y = filtered_xy(batch.frame)
            if len(x) == 0:
                continue

            plot_x = np.log10(x) if log_x else x
            plot_y = np.log10(y) if log_y else y
            batch_counts, _, _ = np.histogram2d(
                plot_y,
                plot_x,
                bins=[bins, bins],
                range=[[plot_y0, plot_y1], [plot_x0, plot_x1]],
            )
            counts += batch_counts

        return {
            "counts": counts,
            "x_edges": np.linspace(plot_x0, plot_x1, bins + 1, dtype=float),
            "y_edges": np.linspace(plot_y0, plot_y1, bins + 1, dtype=float),
            "raw_x_range": (float(raw_x0), float(raw_x1)),
            "raw_y_range": (float(raw_y0), float(raw_y1)),
            "plot_x_range": (plot_x0, plot_x1),
            "plot_y_range": (plot_y0, plot_y1),
            "row_count": int(eligible_rows),
            "backend": self.backend_name,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "dataset_name": self.dataset_name,
            "input_dataset_ids": [branch.dataset_id for branch in self.branches],
            "source_column_name": self.source_column_name,
            "row_count": self._row_count_cache,
            "materialized": False,
            "capabilities": {
                "batch_scan": True,
                "batch_lookup_by_id": True,
                "filtered_scan": True,
                "ordered_scan": False,
                "sharded_scan": False,
                "bounded_distinct": True,
                "column_statistics": True,
            },
        }
