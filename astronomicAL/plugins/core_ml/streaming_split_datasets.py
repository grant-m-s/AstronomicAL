from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from astronomicAL.platform.dataset_sources import DatasetScan
from .feature_columns import parse_column_list
from .paths import ml_run_artifact_dir
from .protocol import PartitionRef, Partitions
from .serialization import json_safe
from .split_manifest import (
    SPLIT_ROLE_COLUMN,
    iter_split_manifest,
    json_scalar,
)

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}
CancelCheck = Optional[Callable[[], None]]


@dataclass(frozen=True)
class MaterializedPartitionDataset:
    role: str
    paths: tuple[str, ...]
    row_count: int
    columns: tuple[str, ...]


def should_materialize_split_datasets(params: Mapping[str, Any]) -> bool:
    raw = dict(params or {}).get("protocol_materialize_split_datasets", True)
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _FALSE_VALUES:
        return False
    if text in _TRUE_VALUES:
        return True
    return True


def materialize_streaming_split_datasets(
    *,
    run: Any,
    protocol: Any,
    binding: Any,
    parts: Partitions,
) -> Dict[str, str]:
    """Write source-derived split datasets with one sequential source scan.

    The previous implementation opened a ``PartitionReader`` once per role and
    repeatedly joined bounded manifest ID batches back to the full source. This
    implementation builds one compact membership index, scans the source once,
    routes each row to its role writer, and registers role-specific Parquet
    datasets. Subsequent epochs can scan those physical datasets directly.
    """

    params = dict(getattr(run, "params", {}) or {})
    if not should_materialize_split_datasets(params):
        return {}

    datasets = getattr(getattr(run, "context", None), "datasets", None)
    if datasets is None:
        raise RuntimeError("Split dataset registration requires context.datasets.")
    register_parquet = getattr(datasets, "register_parquet", None)
    if not callable(register_parquet):
        raise RuntimeError(
            "DatasetManager does not expose register_parquet(); streamed split "
            "datasets cannot be registered safely."
        )

    source_dataset_id = str(run.dataset_id)
    source = datasets.get_source(source_dataset_id)
    columns = _split_columns(source, binding=binding, params=params)
    root = ml_run_artifact_dir(run, kind="split_datasets")
    prefix = _split_dataset_prefix(
        source_dataset_id=source_dataset_id,
        run_id=run.run_id,
        recipe_id=run.recipe_id,
        params=params,
    )
    batch_size = _split_dataset_batch_size(params)
    source_partitions = {
        role: partition
        for role in ("train", "validation", "test")
        if (
            (partition := parts.partition_ref(role)) is not None
            and _is_source_split(partition, source_dataset_id)
        )
    }
    if not source_partitions:
        return {}

    role_roots = {
        role: root / f"{_safe_id(role)}-{uuid.uuid4().hex[:8]}"
        for role in source_partitions
    }
    materialized = materialize_partition_datasets(
        context=run.context,
        source_dataset_id=source_dataset_id,
        partitions=source_partitions,
        columns=columns,
        role_roots=role_roots,
        batch_size=batch_size,
        cancel_check=run.check_cancelled,
    )

    created: Dict[str, str] = {}
    try:
        for role, result in materialized.items():
            partition = source_partitions[role]
            dataset_id = f"{prefix}__{role}"
            name = _split_dataset_name(
                context=run.context,
                source_dataset_id=source_dataset_id,
                role=role,
                run_id=run.run_id,
            )
            meta = {
                "domain": "ml",
                "origin": "core.ml.recipe_split",
                "source_kind": "ml_recipe_split",
                "source_dataset_id": source_dataset_id,
                "derived_from": source_dataset_id,
                "derived_role": role,
                "is_derived": True,
                "is_ml_split_dataset": True,
                "ml_split_role": role,
                "ml_run_id": str(run.run_id),
                "recipe_id": str(run.recipe_id),
                "recipe_version": str(run.recipe_version),
                "protocol_id": str(getattr(protocol, "protocol_id", "")),
                "protocol_split_strategy": str(
                    getattr(protocol, "split_strategy", "")
                ),
                "protocol_validation_source": str(
                    getattr(protocol, "validation_source", "")
                ),
                "protocol_test_source": str(
                    getattr(protocol, "test_source", "")
                ),
                "record_id_column": str(binding.record_id_column),
                "target_column": (
                    str(binding.target_column)
                    if binding.target_column
                    else None
                ),
                "row_count": int(result.row_count),
                "columns": list(result.columns),
                "split_manifest_uri": str(partition.manifest.uri),
                "split_manifest_sha256": str(partition.manifest.sha256),
                "split_manifest_role": str(partition.role),
                "physical_access": "sequential_parquet_scan",
                "created_at": time.time(),
            }
            register_parquet(
                dataset_id,
                list(result.paths),
                name=name,
                **meta,
            )
            _copy_column_mappings(
                context=run.context,
                source_dataset_id=source_dataset_id,
                split_dataset_id=dataset_id,
                available_columns=set(result.columns),
                binding=binding,
            )
            created[role] = dataset_id
            _publish_dataset_events(
                context=run.context,
                dataset_id=dataset_id,
                role=role,
                source_dataset_id=source_dataset_id,
                run_id=str(run.run_id),
                protocol_id=str(getattr(protocol, "protocol_id", "")),
                row_count=int(result.row_count),
            )
    except Exception:
        if not created:
            _delete_materialized_results(materialized)
        raise

    _publish(
        run.context,
        "ml.split_datasets.created",
        {
            "run_id": str(run.run_id),
            "source_dataset_id": source_dataset_id,
            "protocol_id": str(getattr(protocol, "protocol_id", "")),
            "dataset_ids": dict(created),
            "storage": "parquet",
            "physical_access": "sequential_parquet_scan",
        },
    )
    return created


def materialize_partition_datasets(
    *,
    context: Any,
    source_dataset_id: str,
    partitions: Mapping[str, PartitionRef],
    columns: Optional[Sequence[str]],
    role_roots: Mapping[str, Path | str],
    batch_size: int,
    cancel_check: CancelCheck = None,
) -> Dict[str, MaterializedPartitionDataset]:
    """Materialise several roles from one source scan.

    ``partitions`` maps output role names to references in one split manifest.
    Only references whose membership belongs to ``source_dataset_id`` should be
    supplied. Membership is indexed on disk, so neither the source dataset nor
    the complete set of record IDs is held in memory.
    """

    resolved_batch_size = int(batch_size)
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if not partitions:
        return {}

    datasets = getattr(context, "datasets", None)
    get_source = getattr(datasets, "get_source", None)
    if not callable(get_source):
        raise RuntimeError("context.datasets does not expose get_source().")
    source = get_source(str(source_dataset_id))
    if source is None:
        raise KeyError(f"Dataset {source_dataset_id!r} is not registered.")
    if not source.capabilities().batch_scan:
        raise NotImplementedError(
            f"{type(source).__name__} does not support bounded source scans."
        )

    refs = {str(role): ref for role, ref in dict(partitions).items()}
    record_id_column, manifest = _validate_partition_refs(
        source_dataset_id=str(source_dataset_id),
        partitions=refs,
    )
    selected_columns = _materialization_columns(
        source,
        columns=columns,
        record_id_column=record_id_column,
    )
    expected_counts = {
        role: int(ref.row_count) for role, ref in refs.items()
    }
    expected_total = sum(expected_counts.values())
    source_count = source.row_count()
    if source_count is not None and int(source_count) != expected_total:
        raise ValueError(
            f"Source dataset {source_dataset_id!r} contains {source_count} rows, "
            f"but the supplied partition roles contain {expected_total}."
        )

    writers = {
        role: _ParquetPartitionWriter(Path(role_roots[role]))
        for role in refs
    }
    membership_path = _membership_database_path(role_roots)
    connection: Optional[sqlite3.Connection] = None
    finalized: Dict[str, MaterializedPartitionDataset] = {}
    try:
        connection = sqlite3.connect(str(membership_path))
        _configure_membership_database(connection)
        _build_membership_index(
            connection,
            manifest=manifest,
            partitions=refs,
            cancel_check=cancel_check,
        )

        counts = {role: 0 for role in refs}
        for batch in source.iter_batches(
            DatasetScan(
                columns=tuple(selected_columns),
                batch_size=resolved_batch_size,
            )
        ):
            _check_cancel(cancel_check)
            frame = batch.frame
            if frame is None or frame.empty:
                continue
            if record_id_column not in frame.columns:
                raise KeyError(
                    f"Source scan did not return record-ID column "
                    f"{record_id_column!r}."
                )
            keys = [_record_id_key(value) for value in frame[record_id_column]]
            duplicate_keys = pd.Series(keys, dtype="string").duplicated(keep=False)
            if bool(duplicate_keys.any()):
                values = frame.loc[
                    duplicate_keys.to_numpy(), record_id_column
                ].head(5).tolist()
                raise ValueError(
                    f"Source dataset {source_dataset_id!r} contains duplicate "
                    f"record IDs in a scan batch: {values!r}."
                )
            roles = _lookup_batch_roles(connection, keys)
            missing_positions = [
                index for index, role in enumerate(roles) if role is None
            ]
            if missing_positions:
                values = frame.iloc[missing_positions][record_id_column]
                raise KeyError(
                    f"Split manifest does not contain {len(missing_positions)} "
                    f"row(s) from source dataset {source_dataset_id!r}; first "
                    f"missing IDs: {values.head(5).tolist()!r}."
                )

            for role, writer in writers.items():
                positions = [
                    index
                    for index, assigned_role in enumerate(roles)
                    if assigned_role == role
                ]
                if not positions:
                    continue
                role_frame = frame.iloc[positions].reset_index(drop=True)
                writer.write(role_frame)
                counts[role] += len(role_frame)

        for role, expected in expected_counts.items():
            actual = int(counts[role])
            if actual != expected:
                raise ValueError(
                    f"Partition role {role!r} expected {expected} rows but "
                    f"materialised {actual}."
                )

        for role, writer in writers.items():
            paths = writer.finalize()
            finalized[role] = MaterializedPartitionDataset(
                role=role,
                paths=tuple(paths),
                row_count=int(writer.row_count),
                columns=tuple(writer.columns),
            )
        return finalized
    except Exception:
        for writer in writers.values():
            writer.abort()
        _delete_materialized_results(finalized)
        raise
    finally:
        if connection is not None:
            connection.close()
        membership_path.unlink(missing_ok=True)
        for suffix in ("-journal", "-wal", "-shm"):
            Path(str(membership_path) + suffix).unlink(missing_ok=True)


class _ParquetPartitionWriter:
    def __init__(self, final_root: Path):
        self.final_root = Path(final_root)
        self.temp_root = self.final_root.with_name(
            self.final_root.name + ".tmp"
        )
        self.temp_root.mkdir(parents=True, exist_ok=False)
        self._part_count = 0
        self._row_count = 0
        self._columns: list[str] = []
        self._finalized = False

    @property
    def row_count(self) -> int:
        return int(self._row_count)

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    def write(self, frame: pd.DataFrame) -> None:
        if self._finalized:
            raise RuntimeError("Split dataset writer is already finalized.")
        if frame is None or frame.empty:
            return
        incoming_columns = [str(column) for column in frame.columns]
        if not self._columns:
            self._columns = incoming_columns
        elif incoming_columns != self._columns:
            raise ValueError(
                "Split dataset schema changed between batches: "
                f"expected {self._columns!r}, got {incoming_columns!r}."
            )
        path = self.temp_root / f"part-{self._part_count:06d}.parquet"
        _write_parquet_batch(frame.reset_index(drop=True), path)
        self._part_count += 1
        self._row_count += int(len(frame))

    def finalize(self) -> list[str]:
        if self._finalized:
            raise RuntimeError("Split dataset writer is already finalized.")
        if self._part_count <= 0:
            raise ValueError("Cannot register an empty split dataset.")
        self.final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.temp_root, self.final_root)
        self._finalized = True
        return [
            str(path)
            for path in sorted(self.final_root.glob("part-*.parquet"))
        ]

    def abort(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)
        shutil.rmtree(self.final_root, ignore_errors=True)


def _configure_membership_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.executescript(
        """
        CREATE TABLE membership (
            record_id_key TEXT PRIMARY KEY,
            output_role TEXT NOT NULL
        );
        CREATE TEMP TABLE requested_ids (
            position INTEGER PRIMARY KEY,
            record_id_key TEXT NOT NULL
        );
        """
    )


def _build_membership_index(
    connection: sqlite3.Connection,
    *,
    manifest: Any,
    partitions: Mapping[str, PartitionRef],
    cancel_check: CancelCheck,
) -> None:
    manifest_roles = {
        str(ref.role): str(output_role)
        for output_role, ref in partitions.items()
    }
    counts = {role: 0 for role in partitions}
    pending: list[tuple[str, str]] = []
    for row in iter_split_manifest(manifest, verify_checksum=True):
        _check_cancel(cancel_check)
        output_role = manifest_roles.get(str(row.get(SPLIT_ROLE_COLUMN)))
        if output_role is None:
            continue
        record_id = row[manifest.record_id_column]
        pending.append((_record_id_key(record_id), output_role))
        counts[output_role] += 1
        if len(pending) >= 8192:
            _insert_membership_rows(connection, pending)
            pending.clear()
    if pending:
        _insert_membership_rows(connection, pending)
    connection.commit()

    for output_role, ref in partitions.items():
        actual = int(counts[str(output_role)])
        expected = int(ref.row_count)
        if actual != expected:
            raise ValueError(
                f"Split manifest role {ref.role!r} expected {expected} rows but "
                f"indexed {actual}."
            )


def _insert_membership_rows(
    connection: sqlite3.Connection,
    rows: Sequence[tuple[str, str]],
) -> None:
    try:
        connection.executemany(
            "INSERT INTO membership(record_id_key, output_role) VALUES (?, ?)",
            rows,
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            "Split manifest contains duplicate record IDs across the supplied "
            "partition roles."
        ) from exc


def _lookup_batch_roles(
    connection: sqlite3.Connection,
    keys: Sequence[str],
) -> list[Optional[str]]:
    connection.execute("DELETE FROM requested_ids")
    connection.executemany(
        "INSERT INTO requested_ids(position, record_id_key) VALUES (?, ?)",
        enumerate(keys),
    )
    rows = connection.execute(
        "SELECT requested_ids.position, membership.output_role "
        "FROM requested_ids LEFT JOIN membership USING(record_id_key) "
        "ORDER BY requested_ids.position"
    )
    result: list[Optional[str]] = [None] * len(keys)
    for position, role in rows:
        result[int(position)] = str(role) if role is not None else None
    return result


def _validate_partition_refs(
    *,
    source_dataset_id: str,
    partitions: Mapping[str, PartitionRef],
) -> tuple[str, Any]:
    refs = list(partitions.values())
    first = refs[0]
    manifest = first.manifest
    record_id_column = str(manifest.record_id_column)
    seen_manifest_roles: set[str] = set()
    for output_role, ref in partitions.items():
        if str(ref.source) != "split":
            raise ValueError(
                f"Partition {output_role!r} is not source-derived and should not "
                "be materialised through the source router."
            )
        dataset_id = str(ref.dataset_id or ref.manifest.source_dataset_id)
        if dataset_id != str(source_dataset_id):
            raise ValueError(
                f"Partition {output_role!r} belongs to dataset {dataset_id!r}, "
                f"not {source_dataset_id!r}."
            )
        if str(ref.manifest.uri) != str(manifest.uri):
            raise ValueError("All partition roles must use the same split manifest.")
        if str(ref.manifest.sha256) != str(manifest.sha256):
            raise ValueError("Partition roles disagree on split-manifest identity.")
        manifest_role = str(ref.role)
        if manifest_role in seen_manifest_roles:
            raise ValueError(
                f"Split manifest role {manifest_role!r} was supplied more than once."
            )
        seen_manifest_roles.add(manifest_role)
    return record_id_column, manifest


def _materialization_columns(
    source: Any,
    *,
    columns: Optional[Sequence[str]],
    record_id_column: str,
) -> list[str]:
    available = [str(column) for column in source.columns()]
    available_set = set(available)
    if record_id_column not in available_set:
        raise KeyError(
            f"Source dataset is missing record-ID column {record_id_column!r}."
        )
    if columns is None:
        return available
    selected = list(
        dict.fromkeys(
            [record_id_column, *(str(column) for column in columns)]
        )
    )
    missing = [column for column in selected if column not in available_set]
    if missing:
        raise KeyError(f"Unknown split dataset columns: {missing!r}")
    return selected


def _membership_database_path(
    role_roots: Mapping[str, Path | str],
) -> Path:
    roots = [Path(value) for value in role_roots.values()]
    parent = roots[0].parent
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f".partition-membership-{uuid.uuid4().hex}.sqlite"


def _record_id_key(value: Any) -> str:
    return json.dumps(
        json_scalar(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _delete_materialized_results(
    results: Mapping[str, MaterializedPartitionDataset],
) -> None:
    for result in results.values():
        parents = {Path(path).parent for path in result.paths}
        for parent in parents:
            shutil.rmtree(parent, ignore_errors=True)


def _write_parquet_batch(frame: pd.DataFrame, path: Path) -> None:
    """Write one bounded frame using an available Parquet backend."""

    try:
        import duckdb
    except ImportError:
        duckdb = None

    if duckdb is not None:
        connection = duckdb.connect(database=":memory:")
        try:
            connection.register("_astronomical_split_batch", frame)
            escaped = str(path).replace("'", "''")
            connection.execute(
                "COPY _astronomical_split_batch TO "
                f"'{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        finally:
            try:
                connection.unregister("_astronomical_split_batch")
            except Exception:
                pass
            connection.close()
        return

    try:
        frame.to_parquet(path, index=False, compression="zstd")
    except Exception as exc:
        raise RuntimeError(
            "Materialising split datasets requires DuckDB, PyArrow, or "
            "Fastparquet so bounded batches can be written to Parquet."
        ) from exc


def _split_columns(
    source: Any,
    *,
    binding: Any,
    params: Mapping[str, Any],
) -> Optional[list[str]]:
    mode = str(params.get("protocol_split_dataset_columns") or "all").strip()
    if not mode or mode.lower() in {"all", "*"}:
        return None

    if mode.lower() == "training":
        selected = [binding.record_id_column]
        selected.extend(list(getattr(binding, "input_columns", None) or []))
        if getattr(binding, "image_column", None):
            selected.append(binding.image_column)
        if getattr(binding, "target_column", None):
            selected.append(binding.target_column)
    else:
        selected = [binding.record_id_column]
        if getattr(binding, "target_column", None):
            selected.append(binding.target_column)
        selected.extend(parse_column_list(mode))

    selected = list(
        dict.fromkeys(str(column) for column in selected if column)
    )
    available = set(str(column) for column in source.columns())
    missing = [column for column in selected if column not in available]
    if missing:
        raise KeyError(f"Unknown split dataset columns: {missing!r}")
    return selected


def _is_source_split(
    partition: PartitionRef,
    source_dataset_id: str,
) -> bool:
    return (
        str(partition.source) == "split"
        and str(partition.dataset_id or partition.manifest.source_dataset_id)
        == str(source_dataset_id)
    )


def _split_dataset_batch_size(params: Mapping[str, Any]) -> int:
    value = int(
        params.get("protocol_split_dataset_batch_size")
        or params.get("stream_source_batch_size")
        or 8192
    )
    if value <= 0:
        raise ValueError(
            "protocol_split_dataset_batch_size must be greater than zero."
        )
    return value


def _split_dataset_prefix(
    *,
    source_dataset_id: str,
    run_id: str,
    recipe_id: str,
    params: Mapping[str, Any],
) -> str:
    explicit = str(params.get("protocol_split_dataset_prefix") or "").strip()
    if explicit:
        return _safe_id(explicit)
    source = _safe_id(source_dataset_id)
    recipe = _safe_id(recipe_id)
    run = _safe_id(str(run_id)[:8] or "run")
    return f"{source}__ml_{recipe}_{run}"


def _split_dataset_name(
    *,
    context: Any,
    source_dataset_id: str,
    role: str,
    run_id: str,
) -> str:
    source_name = source_dataset_id
    try:
        dataset = context.datasets.get(source_dataset_id)
        source_name = str(
            getattr(dataset, "name", None) or source_dataset_id
        )
    except Exception:
        pass
    labels = {
        "train": "Train",
        "validation": "Validation",
        "test": "Test",
    }
    return (
        f"{source_name} — ML {labels.get(role, role.title())} "
        f"Split ({str(run_id)[:8]})"
    )


def _copy_column_mappings(
    *,
    context: Any,
    source_dataset_id: str,
    split_dataset_id: str,
    available_columns: set[str],
    binding: Any,
) -> None:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return
    mappings: Dict[str, Any] = {}
    try:
        mappings.update(dict(datasets.get_mappings(source_dataset_id) or {}))
    except Exception:
        pass
    if getattr(binding, "record_id_column", None):
        mappings.setdefault("record_id", binding.record_id_column)
    if getattr(binding, "target_column", None):
        mappings.setdefault("target_label", binding.target_column)
    if getattr(binding, "image_column", None):
        mappings.setdefault("image.path", binding.image_column)

    set_mapping = getattr(datasets, "set_mapping", None)
    if not callable(set_mapping):
        return
    for semantic_name, column_name in mappings.items():
        if not _mapping_is_valid(column_name, available_columns):
            continue
        try:
            set_mapping(
                split_dataset_id,
                str(semantic_name),
                str(column_name),
            )
        except Exception:
            pass


def _mapping_is_valid(column_name: Any, columns: set[str]) -> bool:
    if column_name is None:
        return False
    text = str(column_name)
    return text in columns or text.lower() in {
        "use index",
        "use_index",
        "__index__",
        "index",
    }


def _publish_dataset_events(
    *,
    context: Any,
    dataset_id: str,
    role: str,
    source_dataset_id: str,
    run_id: str,
    protocol_id: str,
    row_count: int,
) -> None:
    payload = {
        "dataset_id": dataset_id,
        "role": role,
        "source_dataset_id": source_dataset_id,
        "run_id": run_id,
        "protocol_id": protocol_id,
        "row_count": int(row_count),
        "origin": "core.ml",
    }
    for topic in (
        "dataset.created",
        "dataset.loaded",
        "dataset.updated",
        "ml.split_dataset.created",
    ):
        _publish(context, topic, payload)


def _publish(context: Any, topic: str, payload: Mapping[str, Any]) -> None:
    events = getattr(context, "events", None)
    publish = getattr(events, "publish", None)
    if callable(publish):
        publish(topic, json_safe(dict(payload)))


def _check_cancel(cancel_check: CancelCheck) -> None:
    if cancel_check is not None:
        cancel_check()


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    text = text.strip("._-")
    return text or "dataset"

