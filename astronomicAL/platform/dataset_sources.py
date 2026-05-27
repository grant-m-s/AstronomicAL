from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

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

    def get_row_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if id_column == "Use Index":
            # A Parquet-backed dataset does not have a stable pandas index.
            # Use a real record_id column for large datasets.
            return pd.DataFrame(columns=self.columns() if columns is None else columns)

        if id_column not in self.columns():
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

        con = self._connect()
        try:
            result = con.execute(
                (
                    "SELECT rn FROM ("
                    "  SELECT "
                    f"    ROW_NUMBER() OVER () - 1 AS rn, {_quote_identifier(id_column)} AS rid "
                    f"  FROM {self._relation_sql()}"
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