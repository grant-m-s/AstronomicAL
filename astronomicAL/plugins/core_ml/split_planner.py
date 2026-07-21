from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from astronomicAL.platform.dataset_sources import DatasetScan, DatasetSource
from .protocol import DataBinding, Partitions, ProtocolConfig
from .split_manifest import (
    PartitionManifestMetadata,
    SplitManifestWriter,
    json_scalar,
)

CancelCheck = Optional[Callable[[], None]]
_ROLE_ORDER = ("train", "validation", "test")

def create_streaming_partitions(
    *,
    context: Any,
    root: Path | str,
    run_id: str,
    source_dataset_id: str,
    protocol: ProtocolConfig,
    binding: DataBinding,
    task_kind: str,
    batch_size: int = 65_536,
    cancel_check: CancelCheck = None,
    selected_row_ids: Optional[Sequence[Any]] = None,
) -> Partitions:
    """Create train/validation/test memberships without loading feature data.

    Only record ID, target, and protocol-specific columns are scanned. A narrow
    temporary SQLite index provides deterministic global ranking and group/time
    operations while allowing the source scan and manifest write to remain
    bounded.
    """

    resolved_batch_size = int(batch_size)
    if resolved_batch_size <= 0:
        raise ValueError("split batch_size must be greater than zero.")

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    database_path = root_path / f".{_safe_name(run_id)}-{uuid.uuid4().hex}.split.db"
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    try:
        _configure_database(connection)
        _create_schema(connection)
        need_validation_split = protocol.validation_source == "split"
        need_test_split = protocol.test_source == "split"
        task = _canonical_task(task_kind)

        scan_stats: Dict[str, Dict[str, int]] = {}
        main_source = _get_source(context, source_dataset_id)
        main_columns = _required_columns(
            source=main_source,
            binding=binding,
            protocol=protocol,
            include_protocol_columns=need_validation_split or need_test_split,
        )
        selected_ids = _normalise_selected_row_ids(selected_row_ids)
        if selected_ids is None:
            scan_stats["train_source"] = _ingest_dataset(
                connection,
                source=main_source,
                dataset_id=source_dataset_id,
                source_kind="main",
                assigned_role=None,
                columns=main_columns,
                binding=binding,
                protocol=protocol,
                task_kind=task,
                batch_size=resolved_batch_size,
                cancel_check=cancel_check,
            )
        else:
            scan_stats["train_source"] = _ingest_selected_dataset(
                connection,
                source=main_source,
                dataset_id=source_dataset_id,
                row_ids=selected_ids,
                columns=main_columns,
                binding=binding,
                protocol=protocol,
                task_kind=task,
                batch_size=min(resolved_batch_size, 8192),
                cancel_check=cancel_check,
            )

        if need_validation_split or need_test_split:
            _assign_main_roles(
                connection,
                protocol=protocol,
                task_kind=task,
                need_validation=need_validation_split,
                need_test=need_test_split,
                cancel_check=cancel_check,
            )
        else:
            connection.execute(
                "UPDATE split_rows SET assigned_role = 'train' "
                "WHERE source_kind = 'main'"
            )
            connection.commit()

        validation_dataset_id = (
            str(protocol.validation_dataset_id)
            if protocol.validation_source == "dataset"
            else str(source_dataset_id)
        )
        if protocol.validation_source == "dataset":
            validation_source = _get_source(context, validation_dataset_id)
            validation_columns = _required_columns(
                source=validation_source,
                binding=binding,
                protocol=protocol,
                include_protocol_columns=False,
            )
            scan_stats["validation_source"] = _ingest_dataset(
                connection,
                source=validation_source,
                dataset_id=validation_dataset_id,
                source_kind="validation_dataset",
                assigned_role="validation",
                columns=validation_columns,
                binding=binding,
                protocol=protocol,
                task_kind=task,
                batch_size=resolved_batch_size,
                cancel_check=cancel_check,
            )

        test_dataset_id: Optional[str]
        if protocol.test_source == "dataset":
            test_dataset_id = str(protocol.test_dataset_id)
            test_source = _get_source(context, test_dataset_id)
            test_columns = _required_columns(
                source=test_source,
                binding=binding,
                protocol=protocol,
                include_protocol_columns=False,
            )
            scan_stats["test_source"] = _ingest_dataset(
                connection,
                source=test_source,
                dataset_id=test_dataset_id,
                source_kind="test_dataset",
                assigned_role="test",
                columns=test_columns,
                binding=binding,
                protocol=protocol,
                task_kind=task,
                batch_size=resolved_batch_size,
                cancel_check=cancel_check,
            )
        elif protocol.test_source == "split":
            test_dataset_id = str(source_dataset_id)
        else:
            test_dataset_id = None

        role_counts = _role_counts(connection)
        _validate_required_partitions(
            role_counts,
            require_validation=True,
            require_test=protocol.test_source != "none",
        )
        classes = _resolve_classes(connection, task_kind=task)
        if task == "classification":
            _validate_partition_classes(connection, classes)

        writer = SplitManifestWriter(
            root=root_path,
            run_id=run_id,
            source_dataset_id=source_dataset_id,
            protocol_id=protocol.protocol_id,
            record_id_column=binding.record_id_column,
            target_column=binding.target_column,
        )
        try:
            for role in _ROLE_ORDER:
                if role_counts.get(role, 0) <= 0:
                    continue
                for record_id, target in _iter_role_rows(
                    connection,
                    role=role,
                    cancel_check=cancel_check,
                ):
                    writer.write(role=role, record_id=record_id, target=target)

            metadata = {
                "train": PartitionManifestMetadata(
                    name="train",
                    dataset_id=str(source_dataset_id),
                    source="split",
                    classes=tuple(classes),
                ),
                "validation": PartitionManifestMetadata(
                    name="validation",
                    dataset_id=validation_dataset_id,
                    source=protocol.validation_source,
                    classes=tuple(classes),
                ),
            }
            if role_counts.get("test", 0) > 0:
                metadata["test"] = PartitionManifestMetadata(
                    name="test",
                    dataset_id=test_dataset_id,
                    source=protocol.test_source,
                    classes=tuple(classes),
                )
            manifest, refs = writer.finalize(metadata)
        except Exception:
            writer.abort()
            raise

        parts = Partitions(
            train=refs["train"],
            val=refs["validation"],
            test=refs.get("test"),
            strategy=protocol.split_strategy,
            validation_source=protocol.validation_source,
            test_source=protocol.test_source,
            group_column=protocol.group_column,
            random_state=protocol.random_state,
            protocol_id=protocol.protocol_id,
            target_column=str(binding.target_column or ""),
            record_id_column=binding.record_id_column,
            train_dataset_id=str(source_dataset_id),
            validation_dataset_id=validation_dataset_id,
            test_dataset_id=test_dataset_id,
            split_manifest=manifest,
            partition_refs=refs,
            split_generation={
                "mode": "streaming_sqlite",
                "batch_size": resolved_batch_size,
                "index_path_removed": True,
                "role_counts": dict(role_counts),
                "scan_stats": scan_stats,
                "selected_row_count": (
                    None if selected_ids is None else len(selected_ids)
                ),
            },
        )
        return parts
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
        for suffix in ("-journal", "-wal", "-shm"):
            Path(str(database_path) + suffix).unlink(missing_ok=True)

def _configure_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA cache_size = -65536")

def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE split_rows (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            record_id_json TEXT NOT NULL,
            record_id_key TEXT NOT NULL,
            target_json TEXT,
            target_key TEXT,
            stratum TEXT NOT NULL,
            group_key TEXT,
            temporal_value REAL,
            predefined_role TEXT,
            hash_key INTEGER NOT NULL,
            assigned_role TEXT
        );
        CREATE UNIQUE INDEX split_rows_dataset_record_id
            ON split_rows(dataset_id, record_id_key);
        CREATE INDEX split_rows_source_stratum_hash
            ON split_rows(source_kind, stratum, hash_key, record_id_key);
        CREATE INDEX split_rows_assigned_role
            ON split_rows(assigned_role, seq);
        CREATE INDEX split_rows_group_key
            ON split_rows(group_key);
        CREATE INDEX split_rows_temporal
            ON split_rows(temporal_value, record_id_key);
        """
    )

def _get_source(context: Any, dataset_id: str) -> DatasetSource:
    datasets = getattr(context, "datasets", None)
    get_source = getattr(datasets, "get_source", None)
    if not callable(get_source):
        raise RuntimeError("context.datasets does not expose get_source().")
    source = get_source(str(dataset_id))
    if source is None:
        raise KeyError(f"Dataset {dataset_id!r} is not registered.")
    if not source.capabilities().batch_scan:
        raise NotImplementedError(
            f"Dataset source {type(source).__name__} does not support bounded scans."
        )
    return source

def _required_columns(
    *,
    source: DatasetSource,
    binding: DataBinding,
    protocol: ProtocolConfig,
    include_protocol_columns: bool,
) -> list[str]:
    columns = [binding.record_id_column]
    if binding.target_column:
        columns.append(binding.target_column)
    if include_protocol_columns:
        if protocol.split_strategy in {"by_group", "temporal"} and protocol.group_column:
            columns.append(protocol.group_column)
        if protocol.split_strategy == "predefined" and protocol.split_column:
            columns.append(protocol.split_column)
    columns = list(dict.fromkeys(str(column) for column in columns if column))
    available = set(source.columns())
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(
            f"Dataset source is missing required split column(s): {', '.join(missing)}"
        )
    return columns


def _normalise_selected_row_ids(
    row_ids: Optional[Sequence[Any]],
) -> Optional[list[str]]:
    if row_ids is None:
        return None
    result: list[str] = []
    seen: set[str] = set()
    for value in row_ids:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    if not result:
        raise ValueError("The selected training row set is empty.")
    return result


def _ingest_selected_dataset(
    connection: sqlite3.Connection,
    *,
    source: DatasetSource,
    dataset_id: str,
    row_ids: Sequence[str],
    columns: Sequence[str],
    binding: DataBinding,
    protocol: ProtocolConfig,
    task_kind: str,
    batch_size: int,
    cancel_check: CancelCheck,
) -> Dict[str, int]:
    """Ingest a bounded record-ID selection without scanning the full source."""

    if not source.capabilities().batch_lookup_by_id:
        raise NotImplementedError(
            f"{type(source).__name__} does not support bounded record-ID lookup "
            "required by selected-row training."
        )

    sql = (
        "INSERT INTO split_rows ("
        "dataset_id, source_kind, record_id_json, record_id_key, "
        "target_json, target_key, stratum, group_key, temporal_value, "
        "predefined_role, hash_key, assigned_role"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    inserted = 0
    skipped_missing_target = 0
    records: list[tuple[Any, ...]] = []

    for start in range(0, len(row_ids), max(1, int(batch_size))):
        _check_cancel(cancel_check)
        requested = list(row_ids[start : start + max(1, int(batch_size))])
        frame = source.get_rows_by_ids(
            requested,
            id_column=binding.record_id_column,
            columns=columns,
        )
        if frame is None or frame.empty:
            raise KeyError(
                f"Dataset {dataset_id!r} did not return selected training rows "
                f"{requested[:10]!r}."
            )
        if binding.record_id_column not in frame.columns:
            raise KeyError(
                "Selected-row lookup did not return record ID column "
                f"{binding.record_id_column!r}."
            )

        returned_ids = frame[binding.record_id_column].map(str)
        if returned_ids.duplicated().any():
            duplicates = returned_ids[
                returned_ids.duplicated(keep=False)
            ].unique().tolist()
            raise ValueError(
                f"Dataset {dataset_id!r} returned duplicate selected record IDs: "
                f"{duplicates[:10]!r}"
            )
        returned = set(returned_ids)
        missing = [row_id for row_id in requested if row_id not in returned]
        if missing:
            raise KeyError(
                f"Dataset {dataset_id!r} is missing selected training rows: "
                f"{missing[:10]!r}"
            )

        order = {row_id: index for index, row_id in enumerate(requested)}
        ordered = frame.copy()
        ordered["__selected_order"] = returned_ids.map(order)
        ordered = ordered.sort_values(
            "__selected_order",
            kind="stable",
        ).drop(columns=["__selected_order"])

        for row in ordered.loc[:, list(columns)].itertuples(
            index=False,
            name=None,
        ):
            values = dict(zip(columns, row))
            record_id = values.get(binding.record_id_column)
            target = (
                values.get(binding.target_column)
                if binding.target_column
                else None
            )
            if binding.target_column and _is_missing(target):
                skipped_missing_target += 1
                continue

            record_scalar = json_scalar(record_id)
            target_scalar = json_scalar(target)
            record_key = _value_key(record_scalar)
            target_key = (
                _value_key(target_scalar) if binding.target_column else ""
            )
            stratum = (
                target_key if task_kind == "classification" else "__all__"
            )
            records.append(
                (
                    str(dataset_id),
                    "main",
                    json.dumps(
                        record_scalar,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    record_key,
                    (
                        json.dumps(
                            target_scalar,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        if binding.target_column
                        else None
                    ),
                    target_key,
                    stratum,
                    None,
                    None,
                    None,
                    _stable_hash(record_key, protocol.random_state),
                    None,
                )
            )
            if len(records) >= 4096:
                _insert_records(
                    connection,
                    sql,
                    records,
                    dataset_id=dataset_id,
                )
                inserted += len(records)
                records.clear()
        connection.commit()

    if records:
        _insert_records(
            connection,
            sql,
            records,
            dataset_id=dataset_id,
        )
        inserted += len(records)
    connection.commit()

    if inserted <= 0:
        raise ValueError(
            f"Dataset {dataset_id!r} has no usable selected rows after "
            "dropping missing targets."
        )
    if skipped_missing_target:
        raise ValueError(
            "Selected Active Learning rows are missing verified target labels: "
            f"{skipped_missing_target} row(s)."
        )
    return {
        "scanned_rows": int(len(row_ids)),
        "usable_rows": int(inserted),
        "skipped_missing_target": int(skipped_missing_target),
        "selected_lookup": 1,
    }


def _ingest_dataset(
    connection: sqlite3.Connection,
    *,
    source: DatasetSource,
    dataset_id: str,
    source_kind: str,
    assigned_role: Optional[str],
    columns: Sequence[str],
    binding: DataBinding,
    protocol: ProtocolConfig,
    task_kind: str,
    batch_size: int,
    cancel_check: CancelCheck,
) -> Dict[str, int]:
    scanned = 0
    inserted = 0
    skipped_missing_target = 0
    records: list[tuple[Any, ...]] = []
    sql = (
        "INSERT INTO split_rows ("
        "dataset_id, source_kind, record_id_json, record_id_key, "
        "target_json, target_key, stratum, group_key, temporal_value, "
        "predefined_role, hash_key, assigned_role"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    for batch in source.iter_batches(
        DatasetScan(columns=tuple(columns), batch_size=batch_size)
    ):
        _check_cancel(cancel_check)
        for row in batch.frame.itertuples(index=False, name=None):
            scanned += 1
            values = dict(zip(columns, row))
            record_id = values.get(binding.record_id_column)
            if _is_missing(record_id):
                raise ValueError(
                    f"Dataset {dataset_id!r} contains a missing record ID in "
                    f"column {binding.record_id_column!r}."
                )
            target = values.get(binding.target_column) if binding.target_column else None
            if binding.target_column and _is_missing(target):
                skipped_missing_target += 1
                continue

            record_scalar = json_scalar(record_id)
            target_scalar = json_scalar(target)
            record_key = _value_key(record_scalar)
            target_key = _value_key(target_scalar) if binding.target_column else ""
            stratum = target_key if task_kind == "classification" else "__all__"
            group_key = None
            temporal_value = None
            predefined_role = None
            if source_kind == "main":
                if protocol.split_strategy == "by_group":
                    group_value = values.get(protocol.group_column)
                    if _is_missing(group_value):
                        raise ValueError(
                            f"Grouped split column {protocol.group_column!r} contains "
                            f"a missing value for record {record_scalar!r}."
                        )
                    group_key = _value_key(json_scalar(group_value))
                elif protocol.split_strategy == "temporal":
                    temporal_value = _temporal_value(
                        values.get(protocol.group_column),
                        record_id=record_scalar,
                        column=protocol.group_column,
                    )
                elif protocol.split_strategy == "predefined":
                    predefined_role = _predefined_role(
                        values.get(protocol.split_column),
                        record_id=record_scalar,
                        column=protocol.split_column,
                    )

            records.append(
                (
                    str(dataset_id),
                    str(source_kind),
                    json.dumps(record_scalar, ensure_ascii=False, separators=(",", ":")),
                    record_key,
                    (
                        json.dumps(target_scalar, ensure_ascii=False, separators=(",", ":"))
                        if binding.target_column
                        else None
                    ),
                    target_key,
                    stratum,
                    group_key,
                    temporal_value,
                    predefined_role,
                    _stable_hash(record_key, protocol.random_state),
                    assigned_role,
                )
            )
            if len(records) >= 4096:
                _insert_records(connection, sql, records, dataset_id=dataset_id)
                inserted += len(records)
                records.clear()
        connection.commit()

    if records:
        _insert_records(connection, sql, records, dataset_id=dataset_id)
        inserted += len(records)
        records.clear()
    connection.commit()
    if inserted <= 0:
        raise ValueError(
            f"Dataset {dataset_id!r} has no usable rows after dropping missing targets."
        )
    return {
        "scanned_rows": int(scanned),
        "usable_rows": int(inserted),
        "skipped_missing_target": int(skipped_missing_target),
    }

def _insert_records(
    connection: sqlite3.Connection,
    sql: str,
    records: Sequence[tuple[Any, ...]],
    *,
    dataset_id: str,
) -> None:
    try:
        connection.executemany(sql, records)
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            f"Dataset {dataset_id!r} contains duplicate record IDs in the "
            "configured record-ID column."
        ) from exc

def _assign_main_roles(
    connection: sqlite3.Connection,
    *,
    protocol: ProtocolConfig,
    task_kind: str,
    need_validation: bool,
    need_test: bool,
    cancel_check: CancelCheck,
) -> None:
    strategy = protocol.split_strategy
    if strategy == "random":
        _assign_random_roles(
            connection,
            validation_size=protocol.validation_size if need_validation else 0.0,
            test_size=protocol.test_size if need_test else 0.0,
            stratified=task_kind == "classification",
            cancel_check=cancel_check,
        )
    elif strategy == "by_group":
        _assign_group_roles(
            connection,
            validation_size=protocol.validation_size if need_validation else 0.0,
            test_size=protocol.test_size if need_test else 0.0,
            cancel_check=cancel_check,
        )
    elif strategy == "temporal":
        _assign_temporal_roles(
            connection,
            validation_size=protocol.validation_size if need_validation else 0.0,
            test_size=protocol.test_size if need_test else 0.0,
            cancel_check=cancel_check,
        )
    elif strategy == "predefined":
        _assign_predefined_roles(
            connection,
            need_validation=need_validation,
            need_test=need_test,
        )
    else:
        raise ValueError(f"Unsupported split strategy {strategy!r}.")
    connection.commit()

def _assign_random_roles(
    connection: sqlite3.Connection,
    *,
    validation_size: float,
    test_size: float,
    stratified: bool,
    cancel_check: CancelCheck,
) -> None:
    if stratified:
        strata = connection.execute(
            "SELECT stratum, COUNT(*) AS n FROM split_rows "
            "WHERE source_kind = 'main' GROUP BY stratum ORDER BY stratum"
        )
    else:
        strata = [("__all__", _main_count(connection))]

    updates: list[tuple[str, int]] = []
    for row in strata:
        _check_cancel(cancel_check)
        stratum, count = str(row[0]), int(row[1])
        train_n, validation_n, test_n = _allocate_counts(
            count,
            validation_size=validation_size,
            test_size=test_size,
        )
        if stratified:
            cursor = connection.execute(
                "SELECT seq FROM split_rows WHERE source_kind = 'main' "
                "AND stratum = ? ORDER BY hash_key, record_id_key",
                (stratum,),
            )
        else:
            cursor = connection.execute(
                "SELECT seq FROM split_rows WHERE source_kind = 'main' "
                "ORDER BY hash_key, record_id_key"
            )
        for index, value in enumerate(cursor):
            if index < test_n:
                role = "test"
            elif index < test_n + validation_n:
                role = "validation"
            else:
                role = "train"
            updates.append((role, int(value[0])))
            if len(updates) >= 8192:
                connection.executemany(
                    "UPDATE split_rows SET assigned_role = ? WHERE seq = ?",
                    updates,
                )
                updates.clear()
        expected = train_n + validation_n + test_n
        if expected != count:
            raise RuntimeError("Random split count allocation is inconsistent.")
    if updates:
        connection.executemany(
            "UPDATE split_rows SET assigned_role = ? WHERE seq = ?",
            updates,
        )

def _assign_group_roles(
    connection: sqlite3.Connection,
    *,
    validation_size: float,
    test_size: float,
    cancel_check: CancelCheck,
) -> None:
    total = _main_count(connection)
    if total <= 0:
        return
    connection.execute(
        "CREATE TEMP TABLE group_roles (group_key TEXT PRIMARY KEY, role TEXT NOT NULL)"
    )
    cumulative = 0
    assignments: list[tuple[str, str]] = []
    cursor = connection.execute(
        "SELECT group_key, COUNT(*) AS n, MIN(hash_key) AS h "
        "FROM split_rows WHERE source_kind = 'main' GROUP BY group_key "
        "ORDER BY h, group_key"
    )
    for group_key, count, _hash in cursor:
        _check_cancel(cancel_check)
        count = int(count)
        midpoint = (cumulative + count / 2.0) / float(total)
        if midpoint < test_size:
            role = "test"
        elif midpoint < test_size + validation_size:
            role = "validation"
        else:
            role = "train"
        cumulative += count
        assignments.append((str(group_key), role))
        if len(assignments) >= 4096:
            connection.executemany(
                "INSERT INTO group_roles(group_key, role) VALUES (?, ?)",
                assignments,
            )
            assignments.clear()
    if assignments:
        connection.executemany(
            "INSERT INTO group_roles(group_key, role) VALUES (?, ?)", assignments
        )
    connection.execute(
        "UPDATE split_rows SET assigned_role = ("
        "SELECT role FROM group_roles WHERE group_roles.group_key = split_rows.group_key"
        ") WHERE source_kind = 'main'"
    )

def _assign_temporal_roles(
    connection: sqlite3.Connection,
    *,
    validation_size: float,
    test_size: float,
    cancel_check: CancelCheck,
) -> None:
    total = _main_count(connection)
    train_n, validation_n, test_n = _allocate_counts(
        total,
        validation_size=validation_size,
        test_size=test_size,
    )
    updates: list[tuple[str, int]] = []
    cursor = connection.execute(
        "SELECT seq FROM split_rows WHERE source_kind = 'main' "
        "ORDER BY temporal_value, record_id_key"
    )
    for index, row in enumerate(cursor):
        _check_cancel(cancel_check)
        if index < train_n:
            role = "train"
        elif index < train_n + validation_n:
            role = "validation"
        else:
            role = "test"
        updates.append((role, int(row[0])))
        if len(updates) >= 8192:
            connection.executemany(
                "UPDATE split_rows SET assigned_role = ? WHERE seq = ?", updates
            )
            updates.clear()
    if updates:
        connection.executemany(
            "UPDATE split_rows SET assigned_role = ? WHERE seq = ?", updates
        )
    if train_n + validation_n + test_n != total:
        raise RuntimeError("Temporal split count allocation is inconsistent.")

def _assign_predefined_roles(
    connection: sqlite3.Connection,
    *,
    need_validation: bool,
    need_test: bool,
) -> None:
    present = {
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT predefined_role FROM split_rows "
            "WHERE source_kind = 'main'"
        )
    }
    unsupported = set()
    if not need_validation and "validation" in present:
        unsupported.add("validation")
    if not need_test and "test" in present:
        unsupported.add("test")
    if unsupported:
        raise ValueError(
            "Predefined split contains role(s) not enabled by the protocol: "
            + ", ".join(sorted(unsupported))
        )
    connection.execute(
        "UPDATE split_rows SET assigned_role = predefined_role "
        "WHERE source_kind = 'main'"
    )

def _allocate_counts(
    total: int,
    *,
    validation_size: float,
    test_size: float,
) -> tuple[int, int, int]:
    total = int(total)
    if total <= 0:
        return 0, 0, 0
    test_n = int(math.floor(total * max(0.0, float(test_size))))
    validation_n = int(math.floor(total * max(0.0, float(validation_size))))
    if test_size > 0 and test_n == 0 and total >= 3:
        test_n = 1
    if validation_size > 0 and validation_n == 0 and total - test_n >= 2:
        validation_n = 1
    while test_n + validation_n >= total:
        if validation_n >= test_n and validation_n > 0:
            validation_n -= 1
        elif test_n > 0:
            test_n -= 1
        else:
            break
    train_n = total - validation_n - test_n
    return train_n, validation_n, test_n

def _main_count(connection: sqlite3.Connection) -> int:
    return int(
        connection.execute(
            "SELECT COUNT(*) FROM split_rows WHERE source_kind = 'main'"
        ).fetchone()[0]
    )

def _role_counts(connection: sqlite3.Connection) -> Dict[str, int]:
    return {
        str(role): int(count)
        for role, count in connection.execute(
            "SELECT assigned_role, COUNT(*) FROM split_rows "
            "WHERE assigned_role IS NOT NULL GROUP BY assigned_role"
        )
    }

def _validate_required_partitions(
    role_counts: Mapping[str, int],
    *,
    require_validation: bool,
    require_test: bool,
) -> None:
    required = ["train"]
    if require_validation:
        required.append("validation")
    if require_test:
        required.append("test")
    empty = [role for role in required if int(role_counts.get(role, 0)) <= 0]
    if empty:
        raise ValueError(
            "Split protocol produced no usable rows for: " + ", ".join(empty)
        )

def _resolve_classes(
    connection: sqlite3.Connection,
    *,
    task_kind: str,
) -> list[str]:
    if task_kind != "classification":
        return []
    classes = sorted(
        str(json.loads(row[0]))
        for row in connection.execute(
            "SELECT DISTINCT target_json FROM split_rows "
            "WHERE assigned_role = 'train' ORDER BY target_key"
        )
    )
    if len(classes) < 2:
        raise ValueError(
            "Classification training requires at least two distinct target classes."
        )
    return classes

def _validate_partition_classes(
    connection: sqlite3.Connection,
    classes: Sequence[str],
) -> None:
    allowed = {str(value) for value in classes}
    for role in ("validation", "test"):
        unknown = sorted(
            {
                str(json.loads(row[0]))
                for row in connection.execute(
                    "SELECT DISTINCT target_json FROM split_rows "
                    "WHERE assigned_role = ? ORDER BY target_key",
                    (role,),
                )
                if str(json.loads(row[0])) not in allowed
            }
        )
        if unknown:
            raise ValueError(
                f"{role.capitalize()} partition contains target classes not present "
                f"in training: {unknown!r}."
            )

def _iter_role_rows(
    connection: sqlite3.Connection,
    *,
    role: str,
    cancel_check: CancelCheck,
) -> Iterator[tuple[Any, Any]]:
    cursor = connection.execute(
        "SELECT record_id_json, target_json FROM split_rows "
        "WHERE assigned_role = ? ORDER BY seq",
        (str(role),),
    )
    for record_id_json, target_json in cursor:
        _check_cancel(cancel_check)
        yield (
            json.loads(record_id_json),
            json.loads(target_json) if target_json is not None else None,
        )

def _canonical_task(value: str) -> str:
    return "regression" if str(value).lower() in {"regression", "regressor"} else "classification"

def _stable_hash(value: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{int(seed)}\0{value}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big", signed=False) & ((1 << 63) - 1)

def _value_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except Exception:
        return False

def _temporal_value(value: Any, *, record_id: Any, column: Optional[str]) -> float:
    if _is_missing(value):
        raise ValueError(
            f"Temporal split column {column!r} contains a missing value for "
            f"record {record_id!r}."
        )
    if isinstance(value, bool):
        raise ValueError(f"Temporal split value {value!r} is not orderable.")
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp()
    try:
        parsed = pd.to_datetime(value, errors="raise", utc=True)
        return float(parsed.value)
    except Exception as exc:
        raise ValueError(
            f"Temporal split column {column!r} contains an invalid value "
            f"{value!r} for record {record_id!r}."
        ) from exc

def _predefined_role(value: Any, *, record_id: Any, column: Optional[str]) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "dev": "validation",
        "test": "test",
        "testing": "test",
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(
            f"Predefined split column {column!r} contains unknown role {value!r} "
            f"for record {record_id!r}."
        ) from exc

def _check_cancel(cancel_check: CancelCheck) -> None:
    if cancel_check is not None:
        cancel_check()

def _safe_name(value: Any) -> str:
    text = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(value or "")
    ).strip("._-")
    return text or "run"