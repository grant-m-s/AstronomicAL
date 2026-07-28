from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Optional

import pandas as pd

from astronomicAL.platform.dataset_sources import (
    DatasetBatch,
    DatasetCapabilities,
    DatasetScan,
    DatasetSource,
)


class SelectionSubsetDatasetSource(DatasetSource):
    """Lazy, ordered view over selected rows from another dataset source.

    The subset is represented by stable record IDs rather than by a materialised
    dataframe. Bounded scans fetch only the IDs required for each output batch.
    """

    backend_name = "selection_subset"

    def __init__(
        self,
        *,
        base_source: DatasetSource,
        row_ids: Sequence[Any],
        id_column: str,
        columns: Sequence[str],
    ) -> None:
        self.base_source = base_source
        self.id_column = str(id_column)
        self._columns = [str(column) for column in columns]

        if self.id_column != "Use Index" and self.id_column not in self._columns:
            raise ValueError(
                f"Selection subset ID column {self.id_column!r} is not present "
                "in the source columns."
            )

        self.row_ids = list(row_ids)
        self._row_id_by_key: dict[str, Any] = {}
        self._position_by_key: dict[str, int] = {}

        for position, row_id in enumerate(self.row_ids):
            key = self._row_id_key(row_id)
            if key in self._position_by_key:
                raise ValueError(
                    "Selection subset row IDs must be unique; duplicate ID "
                    f"{key!r} was found."
                )

            self._row_id_by_key[key] = row_id
            self._position_by_key[key] = position

    @staticmethod
    def _row_id_key(row_id: Any) -> str:
        """Return the comparison key used by platform record-ID lookups."""

        return str(row_id)

    def columns(self) -> list[str]:
        return list(self._columns)

    def row_count(self) -> int:
        return len(self.row_ids)

    def dtypes(self) -> dict[str, str]:
        base_dtypes = self.base_source.dtypes()
        return {
            column: str(base_dtypes.get(column, "object"))
            for column in self._columns
        }

    def capabilities(self) -> DatasetCapabilities:
        return DatasetCapabilities(
            batch_scan=True,
            batch_lookup_by_id=True,
            filtered_scan=False,
            ordered_scan=True,
            sharded_scan=True,
            bounded_distinct=False,
            column_statistics=False,
        )

    def iter_batches(self, scan: DatasetScan) -> Iterator[DatasetBatch]:
        """Yield the selected rows in bounded, deterministic batches.

        Limit semantics match the platform sources: the limit is applied before
        logical sharding. Each shard is therefore a disjoint slice of the same
        limited selection.
        """

        if scan.where_sql is not None:
            raise NotImplementedError(
                "SelectionSubsetDatasetSource does not support filtered scans."
            )
        if scan.params:
            raise ValueError("DatasetScan.params requires where_sql")

        requested_columns = self._validated_columns(scan.columns)
        selected_ids = self.row_ids

        if scan.limit is not None:
            selected_ids = selected_ids[: int(scan.limit)]

        if int(scan.shard_count) > 1:
            selected_ids = selected_ids[
                int(scan.shard_index) :: int(scan.shard_count)
            ]

        batch_size = int(scan.batch_size)
        row_offset = 0

        for batch_index, start in enumerate(
            range(0, len(selected_ids), batch_size)
        ):
            batch_ids = selected_ids[start : start + batch_size]
            frame = self._fetch_rows(
                batch_ids,
                columns=requested_columns,
            )

            yield DatasetBatch(
                frame=frame,
                batch_index=batch_index,
                row_offset=row_offset,
            )
            row_offset += len(frame)

    def to_pandas(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        """Materialise the subset through the bounded scan contract."""

        requested_columns = self._validated_columns(columns)
        frames = [
            batch.frame
            for batch in self.iter_batches(
                DatasetScan(
                    columns=tuple(requested_columns),
                    batch_size=8192,
                    where_sql=where_sql,
                    params=tuple(params or ()),
                    limit=limit,
                )
            )
        ]

        if not frames:
            return pd.DataFrame(columns=requested_columns)

        return pd.concat(frames, ignore_index=True)

    def get_row_by_position(
        self,
        position: int,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        requested_columns = self._validated_columns(columns)

        if position < 0 or position >= len(self.row_ids):
            return pd.DataFrame(columns=requested_columns)

        return self._fetch_rows(
            [self.row_ids[position]],
            columns=requested_columns,
        )

    def get_row_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        requested_columns = self._validated_columns(columns)
        self._validate_lookup_column(id_column)

        key = self._row_id_key(row_id)
        if key not in self._row_id_by_key:
            return pd.DataFrame(columns=requested_columns)

        return self._fetch_rows(
            [self._row_id_by_key[key]],
            columns=requested_columns,
        )

    def get_rows_by_ids(
        self,
        row_ids: Sequence[Any],
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Fetch requested rows that belong to this subset.

        Results retain the caller's requested order. Duplicate requested IDs are
        collapsed because record IDs are required to be unique dataset identity.
        """

        requested_columns = self._validated_columns(columns)
        self._validate_lookup_column(id_column)

        selected_ids: list[Any] = []
        seen: set[str] = set()

        for row_id in row_ids:
            key = self._row_id_key(row_id)

            if key in seen or key not in self._row_id_by_key:
                continue

            seen.add(key)
            selected_ids.append(self._row_id_by_key[key])

        if not selected_ids:
            return pd.DataFrame(columns=requested_columns)

        return self._fetch_rows(
            selected_ids,
            columns=requested_columns,
        )

    def find_position_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
    ) -> Optional[int]:
        self._validate_lookup_column(id_column)
        return self._position_by_key.get(self._row_id_key(row_id))

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "base_backend": getattr(
                self.base_source,
                "backend_name",
                "unknown",
            ),
            "rows": len(self.row_ids),
            "id_column": self.id_column,
            "capabilities": self.capabilities().to_dict(),
        }

    def _validated_columns(
        self,
        columns: Optional[Sequence[str]],
    ) -> list[str]:
        requested = (
            []
            if columns is None
            else [str(column) for column in columns]
        )

        # DatasetScan uses None or an empty tuple to mean all columns.
        if not requested:
            return list(self._columns)

        available = set(self._columns)
        missing = [
            column
            for column in requested
            if column not in available
        ]

        if missing:
            raise KeyError(f"Unknown scan columns: {missing!r}")

        return requested

    def _validate_lookup_column(self, id_column: str) -> None:
        if str(id_column) != self.id_column:
            raise ValueError(
                "Selection subset lookups must use the source identity column "
                f"{self.id_column!r}, not {id_column!r}."
            )

    def _fetch_rows(
        self,
        row_ids: Sequence[Any],
        *,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        if not row_ids:
            return pd.DataFrame(columns=list(columns))

        if self.id_column == "Use Index":
            return self._fetch_rows_by_position(
                row_ids,
                columns=columns,
            )

        # The identity column is needed internally to validate completeness and
        # restore deterministic selection order, even when the caller did not
        # request it in the output.
        fetch_columns = list(columns)
        if self.id_column not in fetch_columns:
            fetch_columns.append(self.id_column)

        frame = self.base_source.get_rows_by_ids(
            list(row_ids),
            id_column=self.id_column,
            columns=fetch_columns,
        )

        if frame is None or frame.empty:
            raise KeyError(
                "The base dataset did not return rows for selection IDs "
                f"{[self._row_id_key(value) for value in row_ids[:10]]!r}."
            )

        if self.id_column not in frame.columns:
            raise KeyError(
                "Selection subset lookup did not return its record ID column "
                f"{self.id_column!r}."
            )

        requested_keys = [
            self._row_id_key(value)
            for value in row_ids
        ]
        returned_keys = frame[self.id_column].map(self._row_id_key)

        if returned_keys.duplicated().any():
            duplicates = returned_keys[
                returned_keys.duplicated(keep=False)
            ].unique().tolist()

            raise ValueError(
                "The base dataset returned duplicate record IDs for the "
                f"selection subset: {duplicates[:10]!r}."
            )

        returned_set = set(returned_keys)
        missing = [
            key
            for key in requested_keys
            if key not in returned_set
        ]

        if missing:
            raise KeyError(
                "The base dataset is missing selection rows "
                f"{missing[:10]!r}."
            )

        requested_order = {
            key: position
            for position, key in enumerate(requested_keys)
        }

        ordered = frame.copy()
        ordered["__selection_subset_order"] = returned_keys.map(
            requested_order
        )
        ordered = ordered.sort_values(
            "__selection_subset_order",
            kind="stable",
        ).drop(columns=["__selection_subset_order"])

        return ordered.loc[
            :,
            list(columns),
        ].reset_index(drop=True).copy()

    def _fetch_rows_by_position(
        self,
        row_ids: Sequence[Any],
        *,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for row_id in row_ids:
            try:
                position = int(row_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Selection subset index row IDs must be integer positions; "
                    f"got {row_id!r}."
                ) from exc

            frame = self.base_source.get_row_by_position(
                position,
                columns=columns,
            )

            if frame is None or frame.empty:
                raise KeyError(
                    f"The base dataset is missing row position {position}."
                )

            frames.append(
                frame.head(1).loc[:, list(columns)].copy()
            )

        return pd.concat(frames, ignore_index=True)