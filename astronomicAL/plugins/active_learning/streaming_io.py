from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

@dataclass(frozen=True)
class PredictionRecordBatch:
    records: list[Dict[str, Any]]
    batch_index: int
    row_offset: int

    @property
    def row_count(self) -> int:
        return len(self.records)

PREDICTION_ID_METADATA_KEYS = (
    "record_id_column",
    "row_id_column",
    "id_column",
    "prediction_id_column",
)
PREDICTION_ID_ALIASES = (
    "row_id",
    "record_id",
    "id",
)
PREDICTION_METADATA_CONTAINER_KEYS = (
    "prediction_ref",
    "prediction_table",
    "storage",
    "metadata",
    "params",
    "request",
    "inputs",
    "dataset",
    "prediction_dataset",
    "binding",
    "data_binding",
    "provenance",
)


def prediction_record_id_column(
    payload: Mapping[str, Any],
    *,
    storage: Optional[Mapping[str, Any]] = None,
    available_columns: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Resolve the physical identity column declared by a prediction artifact.

    Prediction tables are allowed to retain the source dataset's record-id column
    name instead of renaming it to ``row_id``.  Consumers must therefore use the
    artifact metadata first and only fall back to conventional aliases.
    """

    candidates: list[str] = []
    seen_objects: set[int] = set()

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in candidates:
            candidates.append(text)

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 5 or not isinstance(value, Mapping):
            return
        object_id = id(value)
        if object_id in seen_objects:
            return
        seen_objects.add(object_id)
        for key in PREDICTION_ID_METADATA_KEYS:
            add(value.get(key))
        for key in PREDICTION_METADATA_CONTAINER_KEYS:
            nested = value.get(key)
            if isinstance(nested, Mapping):
                visit(nested, depth + 1)

    if isinstance(storage, Mapping):
        visit(storage)
    visit(payload)

    if available_columns is None:
        return candidates[0] if candidates else None

    columns = [str(column) for column in available_columns]
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        matched = lookup.get(candidate.lower())
        if matched:
            return matched
    for candidate in PREDICTION_ID_ALIASES:
        if candidate in columns:
            return candidate
        matched = lookup.get(candidate.lower())
        if matched:
            return matched
    return None


def prediction_parquet_columns(storage: Mapping[str, Any]) -> list[str]:
    """Return Parquet columns without materialising prediction rows."""

    paths = [str(path) for path in storage.get("parquet_parts") or [] if str(path)]
    if not paths:
        uri = str(storage.get("uri") or "").strip()
        format_name = str(storage.get("format") or "").strip().lower()
        if uri and (Path(uri).suffix.lower() == ".parquet" or "parquet" in format_name):
            paths = [uri]
    if not paths:
        return []

    try:
        import duckdb
    except Exception:
        return []

    quoted_paths = ", ".join(
        "'" + path.replace("'", "''") + "'"
        for path in paths
    )
    connection = duckdb.connect(database=":memory:")
    try:
        rows = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet([{quoted_paths}]) LIMIT 0"
        ).fetchall()
    except Exception:
        return []
    finally:
        connection.close()
    return [str(row[0]) for row in rows if row]


def _quote_identifier(value: Any) -> str:
    return '"' + str(value).replace('"', '""') + '"'

def iter_prediction_record_batches(
    payload: Mapping[str, Any],
    *,
    batch_size: int = 8192,
    cancel_token: Any = None,
) -> Iterator[PredictionRecordBatch]:
    """Yield prediction records without loading the complete prediction table.

    Stage 6 writes a canonical gzip JSONL table and may also write a Parquet
    mirror. The JSONL table is used here because it is always present and can
    be consumed without an optional Parquet dependency.
    """

    batch_size = max(1, int(batch_size))
    inline = [dict(value) for value in payload.get("records") or [] if isinstance(value, Mapping)]
    storage = prediction_storage_ref(payload)
    id_column = prediction_record_id_column(payload, storage=storage)
    inline_complete = bool(payload.get("records_inline_complete"))

    if inline and (inline_complete or storage is None):
        yield from _iter_sequence_batches(
            inline,
            batch_size=batch_size,
            cancel_token=cancel_token,
            id_column=id_column,
        )
        return

    if storage is None:
        table = payload.get("prediction_table") or {}
        rows = table.get("rows") if isinstance(table, Mapping) else None
        if rows:
            yield from _iter_sequence_batches(
                [dict(value) for value in rows if isinstance(value, Mapping)],
                batch_size=batch_size,
                cancel_token=cancel_token,
                id_column=id_column,
            )
            return
        raise ValueError(
            "Prediction artifact does not contain a complete inline table or a durable prediction_ref."
        )

    uri = str(storage.get("uri") or "").strip()
    if not uri:
        raise ValueError("Prediction storage reference does not contain a uri.")
    path = Path(uri).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Prediction sidecar does not exist: {path}")

    expected_rows = _optional_int(storage.get("row_count"))
    offset = 0
    batch_index = 0
    buffer: list[Dict[str, Any]] = []
    with _open_prediction_text(path, str(storage.get("format") or "")) as handle:
        for line_number, line in enumerate(handle, start=1):
            check_cancelled(cancel_token)
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid prediction JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(raw, Mapping):
                raise TypeError(
                    f"Prediction JSONL row {line_number} must be an object, got {type(raw).__name__}."
                )
            buffer.append(
                normalise_prediction_record(raw, id_column=id_column)
            )
            if len(buffer) >= batch_size:
                yield PredictionRecordBatch(
                    records=buffer,
                    batch_index=batch_index,
                    row_offset=offset,
                )
                offset += len(buffer)
                batch_index += 1
                buffer = []

    if buffer:
        yield PredictionRecordBatch(
            records=buffer,
            batch_index=batch_index,
            row_offset=offset,
        )
        offset += len(buffer)

    if expected_rows is not None and offset != expected_rows:
        raise ValueError(
            "Prediction sidecar row-count mismatch: "
            f"expected {expected_rows}, read {offset} from {path}."
        )

def iter_prediction_records(
    payload: Mapping[str, Any],
    *,
    batch_size: int = 8192,
    cancel_token: Any = None,
) -> Iterator[Dict[str, Any]]:
    for batch in iter_prediction_record_batches(
        payload,
        batch_size=batch_size,
        cancel_token=cancel_token,
    ):
        yield from batch.records

def prediction_records_by_ids(
    payload: Mapping[str, Any],
    row_ids: Sequence[Any],
    *,
    batch_size: int = 8192,
    cancel_token: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Return a small selected-row lookup without loading the prediction table."""

    requested = [str(value) for value in row_ids if str(value)]
    if not requested:
        return {}
    requested_set = set(requested)
    storage = prediction_storage_ref(payload)
    parquet_parts = list((storage or {}).get("parquet_parts") or [])
    parquet_columns = (
        prediction_parquet_columns(storage)
        if storage and parquet_parts
        else []
    )
    parquet_id_column = prediction_record_id_column(
        payload,
        storage=storage,
        available_columns=parquet_columns,
    )
    if parquet_parts and parquet_id_column:
        try:
            import duckdb

            connection = duckdb.connect(database=":memory:")
            try:
                connection.execute("CREATE TEMP TABLE requested_ids(row_id VARCHAR PRIMARY KEY)")
                connection.executemany(
                    "INSERT OR IGNORE INTO requested_ids VALUES (?)",
                    [(row_id,) for row_id in requested],
                )
                paths = ", ".join(
                    "'" + str(path).replace("'", "''") + "'"
                    for path in parquet_parts
                )
                quoted_id = _quote_identifier(parquet_id_column)
                frame = connection.execute(
                    f"SELECT p.*, CAST(p.{quoted_id} AS VARCHAR) "
                    f"AS __al_prediction_row_id "
                    f"FROM read_parquet([{paths}]) p "
                    f"JOIN requested_ids r ON "
                    f"r.row_id = CAST(p.{quoted_id} AS VARCHAR)"
                ).df()
            finally:
                connection.close()
            records: Dict[str, Dict[str, Any]] = {}
            if frame is not None and not frame.empty:
                for raw in frame.to_dict(orient="records"):
                    record = normalise_prediction_record(
                        raw,
                        id_column="__al_prediction_row_id",
                    )
                    record.pop("__al_prediction_row_id", None)
                    row_id = str(record.get("row_id") or "")
                    if row_id:
                        records[row_id] = record
                return records
        except Exception:
            # The canonical JSONL sidecar is the correctness fallback when an
            # optional Parquet mirror has an incomplete schema.
            pass

    found: Dict[str, Dict[str, Any]] = {}
    for batch in iter_prediction_record_batches(
        payload,
        batch_size=batch_size,
        cancel_token=cancel_token,
    ):
        for raw in batch.records:
            row_id = str(raw.get("row_id") or raw.get("record_id") or "")
            if row_id in requested_set:
                found[row_id] = dict(raw)
        if len(found) >= len(requested_set):
            break
    return found

def prediction_storage_ref(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    direct = payload.get("prediction_ref")
    if isinstance(direct, Mapping) and direct.get("uri"):
        return dict(direct)
    table = payload.get("prediction_table")
    if isinstance(table, Mapping):
        storage = table.get("storage")
        if isinstance(storage, Mapping) and storage.get("uri"):
            return dict(storage)
    return None

def normalise_prediction_record(
    value: Mapping[str, Any],
    *,
    id_column: Optional[str] = None,
) -> Dict[str, Any]:
    record = dict(value)
    row_id = (
        record.get(str(id_column))
        if id_column and str(id_column) in record
        else None
    )
    if row_id in (None, ""):
        row_id = record.get("row_id", record.get("record_id", record.get("id")))
    if row_id not in (None, ""):
        record["row_id"] = str(row_id)
        record.setdefault("record_id", str(row_id))

    if "confidence" not in record and record.get("prediction_confidence") is not None:
        record["confidence"] = record.get("prediction_confidence")
    if "y_true" not in record and record.get("true_label") is not None:
        record["y_true"] = record.get("true_label")

    probabilities = record.get("probabilities")
    if not probabilities:
        probability_columns = {
            str(key)[5:]: value
            for key, value in record.items()
            if str(key).startswith("prob_") and value is not None
        }
        if probability_columns:
            record["probabilities"] = probability_columns
    return record

def check_cancelled(cancel_token: Any) -> None:
    if cancel_token is None:
        return
    for method_name in ("raise_if_cancelled", "check_cancelled"):
        method = getattr(cancel_token, method_name, None)
        if callable(method):
            method()
            return
    cancelled = getattr(cancel_token, "cancelled", False)
    if callable(cancelled):
        cancelled = cancelled()
    if cancelled:
        raise RuntimeError("Operation cancelled")

def _iter_sequence_batches(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    cancel_token: Any,
    id_column: Optional[str] = None,
) -> Iterator[PredictionRecordBatch]:
    offset = 0
    for batch_index, start in enumerate(range(0, len(rows), batch_size)):
        check_cancelled(cancel_token)
        records = [
            normalise_prediction_record(value, id_column=id_column)
            for value in rows[start : start + batch_size]
        ]
        yield PredictionRecordBatch(
            records=records,
            batch_index=batch_index,
            row_offset=offset,
        )
        offset += len(records)

def _open_prediction_text(path: Path, format_name: str) -> TextIO:
    format_name = format_name.strip().lower()
    if path.suffix.lower() == ".gz" or format_name in {"jsonl.gz", "gzip-jsonl"}:
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")

def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)