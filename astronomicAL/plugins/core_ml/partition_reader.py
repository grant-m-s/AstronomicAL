from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Sequence

import pandas as pd

from astronomicAL.platform.dataset_sources import (
    DatasetBatch,
    DatasetScan,
    DatasetSource,
)
from .protocol import PartitionRef
from .split_manifest import iter_partition_row_batches

CancelCheck = Callable[[], None]


@dataclass(frozen=True)
class PartitionBatch:
    """One bounded batch retrieved for a manifest-backed partition."""

    frame: pd.DataFrame
    batch_index: int
    row_offset: int
    requested_count: int
    missing_record_ids: tuple[Any, ...] = ()

    @property
    def row_count(self) -> int:
        return int(len(self.frame))


class PartitionReader:
    """Read a partition without materialising the complete membership table.

    Dataset-backed partitions are scanned directly. Manifest-backed partitions
    retain the bounded ID-lookup path as a compatibility fallback for runs that
    explicitly disable split-dataset materialisation or use a source that cannot
    provide a direct partition dataset.
    """

    def __init__(
        self,
        source: DatasetSource,
        partition: PartitionRef,
        *,
        default_batch_size: int = 8192,
    ):
        if int(default_batch_size) <= 0:
            raise ValueError("default_batch_size must be greater than zero.")

        self.source = source
        self.partition = partition
        self.default_batch_size = int(default_batch_size)
        capabilities = source.capabilities()
        self._direct_scan = (
            str(partition.source).strip().lower() == "dataset"
            and bool(capabilities.batch_scan)
        )

        if self._direct_scan:
            self._validate_direct_source_count()
        elif not capabilities.batch_lookup_by_id:
            raise NotImplementedError(
                f"{type(source).__name__} cannot read partition "
                f"{partition.name!r}: the source supports neither a direct "
                "dataset scan nor bounded ID lookup."
            )

    @property
    def record_id_column(self) -> str:
        return self.partition.manifest.record_id_column

    @property
    def row_count(self) -> int:
        return int(self.partition.row_count)

    @property
    def dataset_id(self) -> str:
        return str(
            self.partition.dataset_id
            or self.partition.manifest.source_dataset_id
        )

    @property
    def access_mode(self) -> str:
        return "dataset_scan" if self._direct_scan else "manifest_lookup"

    def iter_record_ids(
        self,
        *,
        verify_checksum: bool = False,
    ) -> Iterator[Any]:
        if self._direct_scan:
            for batch in self.source.iter_batches(
                DatasetScan(
                    columns=(self.record_id_column,),
                    batch_size=self.default_batch_size,
                )
            ):
                _validate_direct_batch(
                    batch,
                    record_id_column=self.record_id_column,
                    dataset_id=self.dataset_id,
                )
                yield from batch.frame[self.record_id_column].tolist()
            return

        record_id_column = self.record_id_column
        for rows in iter_partition_row_batches(
            self.partition,
            batch_size=self.default_batch_size,
            verify_checksum=verify_checksum,
        ):
            for row in rows:
                yield row[record_id_column]

    def iter_batches(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
        strict: bool = True,
        verify_checksum: bool = False,
        cancel_check: Optional[CancelCheck] = None,
        shard_index: int = 0,
        shard_count: int = 1,
    ) -> Iterator[PartitionBatch]:
        resolved_batch_size = int(batch_size or self.default_batch_size)
        if resolved_batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        shard_index = int(shard_index)
        shard_count = int(shard_count)
        if shard_count <= 0:
            raise ValueError("shard_count must be greater than zero.")
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(
                "shard_index must be between zero and shard_count - 1."
            )

        if self._direct_scan and (
            shard_count == 1 or self.source.capabilities().sharded_scan
        ):
            yield from self._iter_direct_batches(
                columns=columns,
                batch_size=resolved_batch_size,
                strict=strict,
                cancel_check=cancel_check,
                shard_index=shard_index,
                shard_count=shard_count,
            )
            return

        yield from self._iter_manifest_lookup_batches(
            columns=columns,
            batch_size=resolved_batch_size,
            strict=strict,
            verify_checksum=verify_checksum,
            cancel_check=cancel_check,
            shard_index=shard_index,
            shard_count=shard_count,
        )

    def materialize(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        max_rows: int = 100_000,
        batch_size: Optional[int] = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """Compatibility helper with an explicit materialisation guardrail."""

        if max_rows < 0:
            raise ValueError("max_rows must be zero or greater.")
        if self.row_count > max_rows:
            raise MemoryError(
                f"Partition {self.partition.name!r} contains {self.row_count} rows; "
                f"the materialisation limit is {max_rows}."
            )

        frames = [
            batch.frame
            for batch in self.iter_batches(
                columns=columns,
                batch_size=batch_size,
                strict=strict,
            )
        ]
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(
            columns=_reader_columns(columns, self.record_id_column)
        )

    def _iter_direct_batches(
        self,
        *,
        columns: Optional[Sequence[str]],
        batch_size: int,
        strict: bool,
        cancel_check: Optional[CancelCheck],
        shard_index: int,
        shard_count: int,
    ) -> Iterator[PartitionBatch]:
        selected_columns = _reader_columns(columns, self.record_id_column)
        seen_rows = 0
        for source_batch in self.source.iter_batches(
            DatasetScan(
                columns=(
                    tuple(selected_columns)
                    if selected_columns is not None
                    else None
                ),
                batch_size=batch_size,
                shard_index=shard_index,
                shard_count=shard_count,
            )
        ):
            if cancel_check is not None:
                cancel_check()
            if strict:
                _validate_direct_batch(
                    source_batch,
                    record_id_column=self.record_id_column,
                    dataset_id=self.dataset_id,
                )
            frame = source_batch.frame.reset_index(drop=True)
            seen_rows += len(frame)
            yield PartitionBatch(
                frame=frame,
                batch_index=source_batch.batch_index,
                row_offset=source_batch.row_offset,
                requested_count=len(frame),
            )

        if strict and shard_count == 1 and seen_rows != self.row_count:
            raise ValueError(
                f"Dataset-backed partition {self.partition.name!r} expected "
                f"{self.row_count} rows but scanned {seen_rows} from "
                f"dataset {self.dataset_id!r}."
            )

    def _iter_manifest_lookup_batches(
        self,
        *,
        columns: Optional[Sequence[str]],
        batch_size: int,
        strict: bool,
        verify_checksum: bool,
        cancel_check: Optional[CancelCheck],
        shard_index: int,
        shard_count: int,
    ) -> Iterator[PartitionBatch]:
        selected_columns = _reader_columns(columns, self.record_id_column)
        manifest_offset = 0
        output_batch_index = 0
        for manifest_rows in iter_partition_row_batches(
            self.partition,
            batch_size=batch_size,
            verify_checksum=verify_checksum,
        ):
            if cancel_check is not None:
                cancel_check()

            requested_ids = [
                row[self.record_id_column]
                for local_index, row in enumerate(manifest_rows)
                if (manifest_offset + local_index) % shard_count == shard_index
            ]
            batch_manifest_offset = manifest_offset
            manifest_offset += len(manifest_rows)
            if not requested_ids:
                continue
            frame = self.source.get_rows_by_ids(
                requested_ids,
                id_column=self.record_id_column,
                columns=selected_columns,
            )
            ordered, missing = _validate_and_order_rows(
                frame,
                requested_ids=requested_ids,
                record_id_column=self.record_id_column,
                dataset_id=self.dataset_id,
            )
            if strict and missing:
                preview = ", ".join(repr(value) for value in missing[:5])
                suffix = "" if len(missing) <= 5 else ", ..."
                raise KeyError(
                    f"Partition {self.partition.name!r} references "
                    f"{len(missing)} row(s) missing from dataset "
                    f"{self.dataset_id!r}: {preview}{suffix}"
                )

            yield PartitionBatch(
                frame=ordered,
                batch_index=output_batch_index,
                row_offset=batch_manifest_offset,
                requested_count=len(requested_ids),
                missing_record_ids=tuple(missing),
            )
            output_batch_index += 1

    def _validate_direct_source_count(self) -> None:
        try:
            source_count = self.source.row_count()
        except Exception:
            source_count = None
        if source_count is None:
            return
        if int(source_count) != self.row_count:
            raise ValueError(
                f"Dataset-backed partition {self.partition.name!r} references "
                f"dataset {self.dataset_id!r} with {source_count} rows, but the "
                f"split manifest records {self.row_count}."
            )


def create_partition_reader(
    context: Any,
    partition: PartitionRef,
    *,
    default_batch_size: int = 8192,
) -> PartitionReader:
    """Resolve the source from ``AppContext`` and construct a reader."""

    dataset_id = str(
        partition.dataset_id or partition.manifest.source_dataset_id
    )
    datasets = getattr(context, "datasets", None)
    get_source = getattr(datasets, "get_source", None)
    if not callable(get_source):
        raise RuntimeError("context.datasets does not expose get_source().")

    source = get_source(dataset_id)
    if source is None:
        raise KeyError(f"Dataset {dataset_id!r} is not registered.")
    return PartitionReader(
        source,
        partition,
        default_batch_size=default_batch_size,
    )


def _reader_columns(
    columns: Optional[Sequence[str]],
    record_id_column: str,
) -> Optional[list[str]]:
    if columns is None:
        return None
    return list(
        dict.fromkeys(
            [record_id_column, *(str(value) for value in columns)]
        )
    )


def _validate_direct_batch(
    batch: DatasetBatch,
    *,
    record_id_column: str,
    dataset_id: str,
) -> None:
    frame = batch.frame
    if record_id_column not in frame.columns:
        raise KeyError(
            f"Dataset {dataset_id!r} did not return record-ID column "
            f"{record_id_column!r}."
        )
    ids = frame[record_id_column]
    missing = ids.isna()
    if bool(missing.any()):
        raise ValueError(
            f"Dataset-backed partition {dataset_id!r} contains null record IDs."
        )
    duplicate = ids.map(_lookup_key).duplicated(keep=False)
    if bool(duplicate.any()):
        values = ids.loc[duplicate].head(5).tolist()
        raise ValueError(
            f"Dataset-backed partition {dataset_id!r} contains duplicate record "
            f"IDs in a scan batch: {values!r}."
        )


def _validate_and_order_rows(
    frame: Optional[pd.DataFrame],
    *,
    requested_ids: Sequence[Any],
    record_id_column: str,
    dataset_id: str,
) -> tuple[pd.DataFrame, list[Any]]:
    if frame is None or frame.empty:
        columns = [] if frame is None else list(frame.columns)
        return pd.DataFrame(columns=columns), list(requested_ids)
    if record_id_column not in frame.columns:
        raise KeyError(
            f"Dataset {dataset_id!r} did not return record-ID column "
            f"{record_id_column!r}."
        )

    result = frame.copy()
    lookup_ids = result[record_id_column].map(_lookup_key)
    duplicated = lookup_ids.duplicated(keep=False)
    if bool(duplicated.any()):
        duplicate_values = result.loc[
            duplicated, record_id_column
        ].head(5).tolist()
        raise ValueError(
            f"Dataset {dataset_id!r} returned duplicate record IDs for a bounded "
            f"partition lookup: {duplicate_values!r}."
        )

    requested_order = {
        _lookup_key(record_id): index
        for index, record_id in enumerate(requested_ids)
    }
    result["__astronomical_partition_order"] = lookup_ids.map(
        requested_order
    )
    unexpected = result["__astronomical_partition_order"].isna()
    if bool(unexpected.any()):
        values = result.loc[unexpected, record_id_column].head(5).tolist()
        raise ValueError(
            f"Dataset {dataset_id!r} returned rows that were not requested: "
            f"{values!r}."
        )

    returned_keys = set(lookup_ids.tolist())
    missing = [
        record_id
        for record_id in requested_ids
        if _lookup_key(record_id) not in returned_keys
    ]
    result = result.sort_values(
        "__astronomical_partition_order",
        kind="stable",
    ).drop(columns=["__astronomical_partition_order"])
    return result.reset_index(drop=True), missing


def _lookup_key(value: Any) -> str:
    return str(value)