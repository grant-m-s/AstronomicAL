from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import pandas as pd

from . import state as al_state
from .streaming_io import (
    PredictionRecordBatch,
    check_cancelled,
    iter_prediction_record_batches,
    normalise_prediction_record,
    prediction_storage_ref,
)

MEMBERSHIP_TABLE_SCHEMA_VERSION = 1
MEMBERSHIP_COLUMNS = (
    "membership_id",
    "session_id",
    "dataset_id",
    "row_id",
    "state",
    "excluded",
    "labelled",
    "training",
    "queued",
    "reason",
    "round",
    "updated_at",
)


@dataclass(frozen=True)
class MembershipTableRef:
    schema_version: int
    storage: str
    uri: str
    format: str
    created_at: float
    session_id: str
    dataset_id: str
    record_id_column: str
    state_column: str
    excluded_column: str
    row_count: int
    excluded_count: int
    labelled_count: int
    training_count: int
    queued_count: int
    columns: list[str]
    sha256: str
    size_bytes: int
    parts: list[str]
    content_sha256: str
    parquet_uri: Optional[str] = None
    parquet_parts: list[str] = field(default_factory=list)
    parquet_sha256: Optional[str] = None
    parquet_size_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["parts"] = list(self.parts or [])
        payload["parquet_parts"] = list(self.parquet_parts or [])
        return payload

    @classmethod
    def from_value(
        cls,
        value: "MembershipTableRef | Mapping[str, Any]",
    ) -> "MembershipTableRef":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("membership table reference must be a mapping")
        return cls(
            schema_version=int(value.get("schema_version") or MEMBERSHIP_TABLE_SCHEMA_VERSION),
            storage=str(value.get("storage") or "local_file"),
            uri=str(value.get("uri") or ""),
            format=str(value.get("format") or "jsonl.gz"),
            created_at=float(value.get("created_at") or 0.0),
            session_id=str(value.get("session_id") or ""),
            dataset_id=str(value.get("dataset_id") or ""),
            record_id_column=str(value.get("record_id_column") or "row_id"),
            state_column=str(value.get("state_column") or "state"),
            excluded_column=str(value.get("excluded_column") or "excluded"),
            row_count=int(value.get("row_count") or 0),
            excluded_count=int(value.get("excluded_count") or 0),
            labelled_count=int(value.get("labelled_count") or 0),
            training_count=int(value.get("training_count") or 0),
            queued_count=int(value.get("queued_count") or 0),
            columns=[str(column) for column in value.get("columns") or []],
            sha256=str(value.get("sha256") or ""),
            size_bytes=int(value.get("size_bytes") or 0),
            parts=[str(path) for path in value.get("parts") or []],
            content_sha256=str(value.get("content_sha256") or value.get("sha256") or ""),
            parquet_uri=(None if value.get("parquet_uri") in (None, "") else str(value.get("parquet_uri"))),
            parquet_parts=[str(path) for path in value.get("parquet_parts") or []],
            parquet_sha256=(None if value.get("parquet_sha256") in (None, "") else str(value.get("parquet_sha256"))),
            parquet_size_bytes=(None if value.get("parquet_size_bytes") in (None, "") else int(value.get("parquet_size_bytes"))),
        )


@dataclass
class EligibilityScanStats:
    mode: str = "stream_filter"
    source_row_count: int = 0
    eligible_row_count: int = 0
    excluded_row_count: int = 0


@dataclass
class ExclusionPlan:
    membership_ref: Optional[MembershipTableRef]
    explicit_row_ids: set[str]
    legacy_row_ids: set[str]

    @property
    def in_memory_row_ids(self) -> set[str]:
        return set(self.explicit_row_ids) | set(self.legacy_row_ids)

    @property
    def estimated_excluded_count(self) -> int:
        durable = int(self.membership_ref.excluded_count) if self.membership_ref else 0
        return durable + len(self.explicit_row_ids | self.legacy_row_ids)


class MembershipTableWriter:
    def __init__(
        self,
        *,
        root: Path | str,
        session_id: str,
        dataset_id: str,
        storage_format: str = "auto",
    ) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_id = str(session_id)
        self.dataset_id = str(dataset_id)
        self.created_at = time.time()
        self.row_count = 0
        self.excluded_count = 0
        self.labelled_count = 0
        self.training_count = 0
        self.queued_count = 0
        self.columns = list(MEMBERSHIP_COLUMNS)
        self._closed = False

        requested = str(storage_format or "auto").strip().lower()
        if requested not in {"auto", "parquet", "jsonl", "jsonl.gz"}:
            raise ValueError("membership_storage_format must be auto, parquet, or jsonl.gz")
        self._write_parquet = requested not in {"jsonl", "jsonl.gz"} and parquet_available()
        if requested == "parquet" and not self._write_parquet:
            raise RuntimeError("Parquet membership output requires duckdb, pyarrow, or fastparquet.")

        token = f"{_safe_name(self.session_id)}-{uuid.uuid4().hex[:10]}"
        self.final_path = self.root / f"{token}.jsonl.gz"
        self.temp_path = self.root / f".{token}.jsonl.gz.tmp"
        self._json_handle = gzip.open(self.temp_path, "wt", encoding="utf-8", newline="\n")
        self.parquet_final_path = self.root / f"{token}.parquet.d"
        self.parquet_temp_path = self.root / f".{token}.parquet.d.tmp"
        self.parquet_parts: list[Path] = []
        if self._write_parquet:
            self.parquet_temp_path.mkdir(parents=True, exist_ok=False)

    def write_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self._closed:
            raise RuntimeError("Membership writer is closed.")
        if not rows:
            return
        normalized = [self._normalize_row(row) for row in rows]
        for row in normalized:
            self._json_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            self._json_handle.write("\n")
        if self._write_parquet:
            frame = pd.DataFrame.from_records(normalized, columns=self.columns)
            for column in ("membership_id", "session_id", "dataset_id", "row_id", "state", "reason"):
                frame[column] = frame[column].astype("string")
            for column in ("excluded", "labelled", "training", "queued"):
                frame[column] = frame[column].astype("boolean")
            frame["round"] = pd.to_numeric(frame["round"], errors="coerce").astype("Int64")
            frame["updated_at"] = pd.to_numeric(frame["updated_at"], errors="coerce").astype("Float64")
            part = self.parquet_temp_path / f"part-{len(self.parquet_parts):06d}.parquet"
            _write_parquet(frame, part)
            self.parquet_parts.append(part)
        self.row_count += len(normalized)
        self.excluded_count += sum(1 for row in normalized if bool(row["excluded"]))
        self.labelled_count += sum(1 for row in normalized if bool(row["labelled"]))
        self.training_count += sum(1 for row in normalized if bool(row["training"]))
        self.queued_count += sum(1 for row in normalized if bool(row["queued"]))

    def _normalize_row(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        row = dict(value or {})
        row_id = str(row.get("row_id") or "").strip()
        state = str(row.get("state") or al_state.ROW_UNLABELLED)
        excluded = bool(row.get("excluded", state in al_state.TERMINAL_QUERY_STATES))
        labelled = bool(row.get("labelled", state in {al_state.ROW_VERIFIED, al_state.ROW_TRAINING}))
        training = bool(row.get("training", state == al_state.ROW_TRAINING))
        queued = bool(row.get("queued", state == al_state.ROW_QUEUED))
        return {
            "membership_id": str(row.get("membership_id") or f"{self.session_id}:{row_id}"),
            "session_id": self.session_id,
            "dataset_id": self.dataset_id,
            "row_id": row_id,
            "state": state,
            "excluded": excluded,
            "labelled": labelled,
            "training": training,
            "queued": queued,
            "reason": str(row.get("reason") or state),
            "round": int(row.get("round") or 0),
            "updated_at": float(row.get("updated_at") or self.created_at),
        }

    def finalize(self, *, content_sha256: str) -> MembershipTableRef:
        if self._closed:
            raise RuntimeError("Membership writer is already closed.")
        try:
            self._json_handle.close()
            os.replace(self.temp_path, self.final_path)
            final_parquet_parts: list[Path] = []
            if self._write_parquet and self.parquet_parts:
                os.replace(self.parquet_temp_path, self.parquet_final_path)
                final_parquet_parts = sorted(
                    self.parquet_final_path.glob("part-*.parquet")
                )
            elif self._write_parquet:
                shutil.rmtree(self.parquet_temp_path, ignore_errors=True)
            self._closed = True
            return MembershipTableRef(
                schema_version=MEMBERSHIP_TABLE_SCHEMA_VERSION,
                storage="local_file",
                uri=str(self.final_path),
                format="jsonl.gz",
                created_at=self.created_at,
                session_id=self.session_id,
                dataset_id=self.dataset_id,
                record_id_column="row_id",
                state_column="state",
                excluded_column="excluded",
                row_count=self.row_count,
                excluded_count=self.excluded_count,
                labelled_count=self.labelled_count,
                training_count=self.training_count,
                queued_count=self.queued_count,
                columns=list(self.columns),
                sha256=_paths_sha256([self.final_path]),
                size_bytes=self.final_path.stat().st_size,
                parts=[str(self.final_path)],
                content_sha256=str(content_sha256),
                parquet_uri=str(self.parquet_final_path) if final_parquet_parts else None,
                parquet_parts=[str(path) for path in final_parquet_parts],
                parquet_sha256=_paths_sha256(final_parquet_parts) if final_parquet_parts else None,
                parquet_size_bytes=(sum(path.stat().st_size for path in final_parquet_parts) if final_parquet_parts else None),
            )
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        handle = getattr(self, "_json_handle", None)
        if handle is not None and not handle.closed:
            handle.close()
        self.temp_path.unlink(missing_ok=True)
        self.final_path.unlink(missing_ok=True)
        shutil.rmtree(self.parquet_temp_path, ignore_errors=True)
        shutil.rmtree(self.parquet_final_path, ignore_errors=True)
        self._closed = True

    def __enter__(self) -> "MembershipTableWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or not self._closed:
            self.abort()
        return False


def iter_membership_rows_from_session(
    session: Mapping[str, Any],
) -> Iterator[Dict[str, Any]]:
    labels = {
        str(key): dict(value or {})
        for key, value in dict(session.get("labels") or {}).items()
    }
    row_states = {
        str(key): str(value)
        for key, value in dict(session.get("row_states") or {}).items()
    }
    for row_id, entry in labels.items():
        status = str(entry.get("status") or "")
        row_states.setdefault(
            row_id,
            al_state.ROW_UNSURE if status == "unsure" else al_state.ROW_VERIFIED,
        )
    for row_id in session.get("training_row_ids") or []:
        row_states[str(row_id)] = al_state.ROW_TRAINING
    for row_id in session.get("ignored_row_ids") or []:
        row_states.setdefault(str(row_id), al_state.ROW_UNSURE)
    last_batch = session.get("last_batch") or {}
    if isinstance(last_batch, Mapping):
        for row_id in last_batch.get("row_ids") or []:
            row_states.setdefault(str(row_id), al_state.ROW_QUEUED)

    round_index = int(session.get("round") or 0)
    updated_at = float(session.get("updated_at") or time.time())
    for row_id in sorted(row_states):
        state = row_states[row_id]
        yield {
            "row_id": row_id,
            "state": state,
            "excluded": state in al_state.TERMINAL_QUERY_STATES,
            "labelled": state in {al_state.ROW_VERIFIED, al_state.ROW_TRAINING},
            "training": state == al_state.ROW_TRAINING,
            "queued": state == al_state.ROW_QUEUED,
            "reason": state,
            "round": round_index,
            "updated_at": updated_at,
        }


def membership_rows_from_session(session: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return list(iter_membership_rows_from_session(session))

def membership_content_sha256(session: Mapping[str, Any]) -> str:
    """Fingerprint membership semantics, excluding volatile save timestamps.

    Session revisions update ``updated_at`` even when no row changes state.  The
    durable membership snapshot should therefore be reused across no-op saves.
    A state, reason, flag, or round change still produces a new snapshot.
    """

    digest = hashlib.sha256()
    for row in iter_membership_rows_from_session(session):
        semantic = {
            key: row.get(key)
            for key in (
                "row_id",
                "state",
                "excluded",
                "labelled",
                "training",
                "queued",
                "reason",
                "round",
            )
        }
        digest.update(
            json.dumps(
                semantic,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def ensure_membership_table(
    context: Any,
    session: Mapping[str, Any],
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> tuple[MembershipTableRef, bool]:
    params = dict(params or {})
    content_sha = membership_content_sha256(session)
    existing_raw = session.get("membership_table_ref")
    if isinstance(existing_raw, Mapping):
        try:
            existing = MembershipTableRef.from_value(existing_raw)
        except Exception:
            existing = None
        if existing is not None and existing.content_sha256 == content_sha and Path(existing.uri).is_file():
            return existing, False

    with MembershipTableWriter(
        root=membership_output_root(context, params),
        session_id=str(session.get("session_id") or "session"),
        dataset_id=str(session.get("pool_dataset_id") or session.get("dataset_id") or ""),
        storage_format=str(params.get("membership_storage_format") or "auto"),
    ) as writer:
        batch_size = max(1, int(params.get("membership_storage_batch_size") or 8192))
        batch: list[Dict[str, Any]] = []
        for row in iter_membership_rows_from_session(session):
            batch.append(row)
            if len(batch) >= batch_size:
                writer.write_rows(batch)
                batch = []
        if batch:
            writer.write_rows(batch)
        ref = writer.finalize(content_sha256=content_sha)
    return ref, True


def membership_ref_from_session(session: Mapping[str, Any]) -> Optional[MembershipTableRef]:
    raw = session.get("membership_table_ref")
    if not isinstance(raw, Mapping):
        return None
    try:
        ref = MembershipTableRef.from_value(raw)
    except Exception:
        return None
    return ref if ref.uri else None


def exclusion_plan_from_session(
    session: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    legacy_row_ids: Optional[Iterable[Any]] = None,
) -> ExclusionPlan:
    explicit = {
        str(value).strip()
        for value in params.get("exclude_row_ids") or []
        if str(value).strip()
    }
    legacy = {
        str(value).strip()
        for value in (legacy_row_ids or [])
        if str(value).strip()
    }
    return ExclusionPlan(
        membership_ref=membership_ref_from_session(session),
        explicit_row_ids=explicit,
        legacy_row_ids=legacy,
    )


def iter_membership_rows(ref: MembershipTableRef | Mapping[str, Any]) -> Iterator[Dict[str, Any]]:
    resolved = MembershipTableRef.from_value(ref)
    path = Path(resolved.uri).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Membership sidecar does not exist: {path}")
    opener = gzip.open if path.suffix.lower() == ".gz" or resolved.format.lower() in {"jsonl.gz", "gzip-jsonl"} else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            if not isinstance(raw, Mapping):
                raise TypeError(f"Membership row {line_number} must be an object")
            yield dict(raw)


class MembershipLookup:
    """Disk-backed membership lookup used when a source-side anti-join is unavailable."""

    def __init__(
        self,
        ref: MembershipTableRef | Mapping[str, Any],
        *,
        excluded_only: bool = True,
    ) -> None:
        self.ref = MembershipTableRef.from_value(ref)
        self.excluded_only = bool(excluded_only)
        self._tempdir = tempfile.TemporaryDirectory(prefix="astronomical-al-membership-")
        self._connection = sqlite3.connect(str(Path(self._tempdir.name) / "membership.sqlite"))
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute("CREATE TABLE membership (row_id TEXT PRIMARY KEY, state TEXT, excluded INTEGER, labelled INTEGER, training INTEGER)")
        buffer: list[tuple[str, str, int, int, int]] = []
        for row in iter_membership_rows(self.ref):
            excluded = int(bool(row.get("excluded")))
            if self.excluded_only and not excluded:
                continue
            buffer.append(
                (
                    str(row.get("row_id") or ""),
                    str(row.get("state") or ""),
                    excluded,
                    int(bool(row.get("labelled"))),
                    int(bool(row.get("training"))),
                )
            )
            if len(buffer) >= 8192:
                self._connection.executemany("INSERT OR REPLACE INTO membership VALUES (?, ?, ?, ?, ?)", buffer)
                buffer = []
        if buffer:
            self._connection.executemany("INSERT OR REPLACE INTO membership VALUES (?, ?, ?, ?, ?)", buffer)
        self._connection.commit()

    def contains_many(self, row_ids: Sequence[str]) -> set[str]:
        """Return excluded IDs from ``row_ids``."""

        return self.matching_many(row_ids, excluded=True)

    def matching_many(
        self,
        row_ids: Sequence[str],
        *,
        excluded: Optional[bool] = None,
        labelled: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> set[str]:
        values = [str(value) for value in row_ids if str(value)]
        found: set[str] = set()
        conditions: list[str] = []
        if excluded is not None:
            conditions.append(f"excluded = {1 if excluded else 0}")
        if labelled is not None:
            conditions.append(f"labelled = {1 if labelled else 0}")
        if training is not None:
            conditions.append(f"training = {1 if training else 0}")
        chunk_size = 900
        for start in range(0, len(values), chunk_size):
            chunk = values[start : start + chunk_size]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT row_id FROM membership WHERE row_id IN ({placeholders})"
            if conditions:
                query += " AND " + " AND ".join(conditions)
            found.update(str(row[0]) for row in self._connection.execute(query, chunk))
        return found

    def row_ids_for(self, *, labelled: bool = False, training: bool = False) -> Iterator[str]:
        where: list[str] = []
        if labelled:
            where.append("labelled = 1")
        if training:
            where.append("training = 1")
        sql = "SELECT row_id FROM membership"
        if where:
            sql += " WHERE " + " AND ".join(where)
        for row in self._connection.execute(sql):
            yield str(row[0])

    def close(self) -> None:
        try:
            self._connection.close()
        finally:
            self._tempdir.cleanup()

    def __enter__(self) -> "MembershipLookup":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def iter_eligible_prediction_batches(
    predictions_payload: Mapping[str, Any],
    *,
    exclusion_plan: ExclusionPlan,
    batch_size: int = 8192,
    cancel_token: Any = None,
) -> tuple[Iterator[PredictionRecordBatch], EligibilityScanStats]:
    stats = EligibilityScanStats()
    storage = prediction_storage_ref(predictions_payload)
    if (
        storage
        and storage.get("parquet_parts")
        and exclusion_plan.membership_ref is not None
        and exclusion_plan.membership_ref.parquet_parts
        and duckdb_available()
    ):
        stats.mode = "duckdb_anti_join"
        return (
            _iter_duckdb_anti_join_batches(
                predictions_payload,
                storage=storage,
                exclusion_plan=exclusion_plan,
                batch_size=batch_size,
                cancel_token=cancel_token,
                stats=stats,
            ),
            stats,
        )
    stats.mode = "disk_index_filter" if exclusion_plan.membership_ref is not None else "in_memory_filter"
    return (
        _iter_filtered_prediction_batches(
            predictions_payload,
            exclusion_plan=exclusion_plan,
            batch_size=batch_size,
            cancel_token=cancel_token,
            stats=stats,
        ),
        stats,
    )


def _iter_filtered_prediction_batches(
    predictions_payload: Mapping[str, Any],
    *,
    exclusion_plan: ExclusionPlan,
    batch_size: int,
    cancel_token: Any,
    stats: EligibilityScanStats,
) -> Iterator[PredictionRecordBatch]:
    in_memory = exclusion_plan.in_memory_row_ids
    lookup_cm = MembershipLookup(exclusion_plan.membership_ref) if exclusion_plan.membership_ref else None
    try:
        output_buffer: list[Dict[str, Any]] = []
        output_offset = 0
        output_batch_index = 0
        for batch in iter_prediction_record_batches(
            predictions_payload,
            batch_size=batch_size,
            cancel_token=cancel_token,
        ):
            check_cancelled(cancel_token)
            ids = [str(record.get("row_id") or record.get("record_id") or "") for record in batch.records]
            durable_excluded = lookup_cm.contains_many(ids) if lookup_cm is not None else set()
            for record, row_id in zip(batch.records, ids):
                stats.source_row_count += 1
                if not row_id or row_id in in_memory or row_id in durable_excluded:
                    stats.excluded_row_count += 1
                    continue
                stats.eligible_row_count += 1
                output_buffer.append(dict(record))
                if len(output_buffer) >= batch_size:
                    yield PredictionRecordBatch(
                        records=output_buffer,
                        batch_index=output_batch_index,
                        row_offset=output_offset,
                    )
                    output_offset += len(output_buffer)
                    output_batch_index += 1
                    output_buffer = []
        if output_buffer:
            yield PredictionRecordBatch(
                records=output_buffer,
                batch_index=output_batch_index,
                row_offset=output_offset,
            )
    finally:
        if lookup_cm is not None:
            lookup_cm.close()


def _iter_duckdb_anti_join_batches(
    predictions_payload: Mapping[str, Any],
    *,
    storage: Mapping[str, Any],
    exclusion_plan: ExclusionPlan,
    batch_size: int,
    cancel_token: Any,
    stats: EligibilityScanStats,
) -> Iterator[PredictionRecordBatch]:
    import duckdb

    membership_ref = exclusion_plan.membership_ref
    if membership_ref is None:
        return
    prediction_paths = [str(path) for path in storage.get("parquet_parts") or []]
    membership_paths = [str(path) for path in membership_ref.parquet_parts]
    if not prediction_paths or not membership_paths:
        return

    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("CREATE TEMP TABLE explicit_exclusions(row_id VARCHAR PRIMARY KEY)")
        if exclusion_plan.in_memory_row_ids:
            connection.executemany(
                "INSERT OR IGNORE INTO explicit_exclusions VALUES (?)",
                [(row_id,) for row_id in sorted(exclusion_plan.in_memory_row_ids)],
            )
        pred_sql = _read_parquet_sql(prediction_paths)
        member_sql = _read_parquet_sql(membership_paths)
        total = connection.execute(f"SELECT COUNT(*) FROM {pred_sql}").fetchone()[0]
        stats.source_row_count = int(total or 0)
        sql = (
            f"SELECT p.* FROM {pred_sql} AS p "
            f"WHERE NOT EXISTS ("
            f"SELECT 1 FROM {member_sql} AS m "
            f"WHERE CAST(m.row_id AS VARCHAR) = CAST(p.row_id AS VARCHAR) "
            f"AND COALESCE(CAST(m.excluded AS BOOLEAN), FALSE)"
            f") AND NOT EXISTS ("
            f"SELECT 1 FROM explicit_exclusions e "
            f"WHERE e.row_id = CAST(p.row_id AS VARCHAR)"
            f")"
        )
        cursor = connection.execute(sql)
        columns = [str(description[0]) for description in cursor.description]
        offset = 0
        batch_index = 0
        while True:
            check_cancelled(cancel_token)
            rows = cursor.fetchmany(max(1, int(batch_size)))
            if not rows:
                break
            records = [normalise_prediction_record(dict(zip(columns, row))) for row in rows]
            stats.eligible_row_count += len(records)
            yield PredictionRecordBatch(records=records, batch_index=batch_index, row_offset=offset)
            offset += len(records)
            batch_index += 1
        stats.excluded_row_count = max(0, stats.source_row_count - stats.eligible_row_count)
    finally:
        connection.close()


def membership_output_root(context: Any, params: Mapping[str, Any]) -> Path:
    explicit = params.get("membership_output_dir") or params.get("artifact_root")
    if explicit:
        root = Path(str(explicit)).expanduser()
    else:
        cache_dir = getattr(getattr(context, "artifacts", None), "_cache_dir", None)
        root = Path(cache_dir).expanduser() / "active_learning" / "memberships" if cache_dir else Path.cwd() / ".astronomical" / "active_learning" / "memberships"
    root.mkdir(parents=True, exist_ok=True)
    return root


def artifact_row_ids_ref(ref: MembershipTableRef | Mapping[str, Any]) -> Dict[str, Any]:
    resolved = MembershipTableRef.from_value(ref)
    if resolved.parquet_parts:
        return {
            "storage": resolved.storage,
            "uri": str(resolved.parquet_uri or resolved.parquet_parts[0]),
            "format": "parquet",
            "row_count": resolved.row_count,
            "sha256": resolved.parquet_sha256,
            "id_column": resolved.record_id_column,
            "parts": list(resolved.parquet_parts),
            "params": {
                "schema_version": resolved.schema_version,
                "session_id": resolved.session_id,
                "excluded_column": resolved.excluded_column,
            },
        }
    return {
        "storage": resolved.storage,
        "uri": resolved.uri,
        "format": resolved.format,
        "row_count": resolved.row_count,
        "sha256": resolved.sha256,
        "id_column": resolved.record_id_column,
        "parts": list(resolved.parts),
        "params": {
            "schema_version": resolved.schema_version,
            "session_id": resolved.session_id,
            "excluded_column": resolved.excluded_column,
        },
    }


def excluded_row_ids_preview(
    session: Mapping[str, Any],
    *,
    limit: int = 1000,
) -> list[str]:
    limit = max(0, int(limit))
    preview: list[str] = []
    if limit <= 0:
        return preview
    for row in iter_membership_rows_from_session(session):
        if not bool(row.get("excluded")):
            continue
        preview.append(str(row["row_id"]))
        if len(preview) >= limit:
            break
    return preview


def parquet_available() -> bool:
    for module in ("duckdb", "pyarrow", "fastparquet"):
        try:
            __import__(module)
            return True
        except Exception:
            continue
    return False


def duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401

        return True
    except Exception:
        return False


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=False)
        return
    except ImportError:
        pass
    try:
        import duckdb
    except Exception as exc:
        raise RuntimeError("No Parquet writer is available.") from exc
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("membership_batch", frame)
        escaped = str(path).replace("'", "''")
        connection.execute(f"COPY membership_batch TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        connection.close()


def _paths_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: Any) -> str:
    text = "".join(character if character.isalnum() or character in {"-", "_", "."} else "-" for character in str(value or "session")).strip("-._")
    return text[:96] or "session"


def _read_parquet_sql(paths: Sequence[str]) -> str:
    quoted = ", ".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{quoted}])"
