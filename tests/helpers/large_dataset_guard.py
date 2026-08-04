from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd

from astronomicAL.platform.dataset_sources import DatasetSource


class FullMaterialisationError(RuntimeError):
    """
    Raised by HugeDatasetSource when code tries to materialise the entire
    dataset into pandas.

    This is intentionally loud. In real usage, the equivalent bug would be
    a huge memory spike or a very slow UI/plugin load.
    """


class HugeDatasetSource(DatasetSource):
    """
    Test double for a huge Parquet/DuckDB-style dataset source.

    It permits metadata access and bounded pandas views, but raises if code
    tries to materialise the full dataset into pandas.

    Use this in plugin contract tests when you want to verify that a plugin is
    not accidentally doing:

        context.datasets.get_df()
        context.datasets.get().df
        source.to_pandas()

    without a column/row bound.
    """

    backend_name = "huge_guard_source"

    def __init__(
        self,
        *,
        row_count: int = 50_000_000,
        columns: Sequence[str] = ("id", "x", "y", "label", "ra", "dec"),
        id_column: str = "id",
    ) -> None:
        self._row_count = int(row_count)
        self._columns = [str(column) for column in columns]
        self._id_column = str(id_column)
        self.calls: list[dict[str, Any]] = []

    def columns(self) -> list[str]:
        self.calls.append({"method": "columns"})
        return list(self._columns)

    def row_count(self) -> int:
        self.calls.append({"method": "row_count"})
        return self._row_count

    def dtypes(self) -> dict[str, str]:
        self.calls.append({"method": "dtypes"})
        return {
            column: "object" if column in {self._id_column, "label"} else "float64"
            for column in self._columns
        }

    def to_pandas(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "method": "to_pandas",
                "columns": None if columns is None else list(columns),
                "limit": limit,
                "where_sql": where_sql,
                "params": None if params is None else list(params),
            }
        )

        unbounded_rows = limit is None and where_sql is None
        unbounded_columns = columns is None

        if unbounded_rows and unbounded_columns:
            raise FullMaterialisationError(
                "Attempted to materialise a huge dataset into pandas without "
                "a row limit, column subset, or SQL filter. Use DatasetSource "
                "methods, bounded to_pandas(...), get_row_by_id(...), "
                "get_row_by_position(...), or a Parquet/DuckDB-friendly query."
            )

        selected_columns = [str(column) for column in (columns or self._columns)]
        n_rows = int(limit if limit is not None else 1)

        if where_sql is not None and limit is None:
            # A filtered query might still produce many rows in real life, but
            # for this guard source we allow it because it is not an obviously
            # unconditional full materialisation.
            n_rows = 1

        rows = []
        for i in range(max(n_rows, 0)):
            row = {}
            for column in selected_columns:
                if column == self._id_column:
                    row[column] = f"r{i}"
                elif column == "label":
                    row[column] = -1
                else:
                    row[column] = float(i)
            rows.append(row)

        return pd.DataFrame(rows, columns=selected_columns)

    def head(
        self,
        n: int = 5,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "method": "head",
                "n": n,
                "columns": None if columns is None else list(columns),
            }
        )
        return self.to_pandas(columns=columns, limit=n)

    def get_row_by_position(
        self,
        position: int,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "method": "get_row_by_position",
                "position": position,
                "columns": None if columns is None else list(columns),
            }
        )

        selected_columns = [str(column) for column in (columns or self._columns)]
        row = {}
        for column in selected_columns:
            if column == self._id_column:
                row[column] = f"r{position}"
            elif column == "label":
                row[column] = -1
            else:
                row[column] = float(position)

        return pd.DataFrame([row], columns=selected_columns)

    def get_row_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "method": "get_row_by_id",
                "row_id": row_id,
                "id_column": id_column,
                "columns": None if columns is None else list(columns),
            }
        )

        selected_columns = [str(column) for column in (columns or self._columns)]
        row = {}
        for column in selected_columns:
            if column == id_column:
                row[column] = row_id
            elif column == "label":
                row[column] = -1
            else:
                row[column] = 0.0

        return pd.DataFrame([row], columns=selected_columns)

    def find_position_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
    ) -> Optional[int]:
        self.calls.append(
            {
                "method": "find_position_by_id",
                "row_id": row_id,
                "id_column": id_column,
            }
        )

        text = str(row_id)
        if text.startswith("r") and text[1:].isdigit():
            return int(text[1:])

        return None