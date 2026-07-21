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
    inline_complete = bool(payload.get("records_inline_complete"))

    if inline and (inline_complete or storage is None):
        yield from _iter_sequence_batches(
            inline,
            batch_size=batch_size,
            cancel_token=cancel_token,
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
            buffer.append(normalise_prediction_record(raw))
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


def normalise_prediction_record(value: Mapping[str, Any]) -> Dict[str, Any]:
    record = dict(value)
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
) -> Iterator[PredictionRecordBatch]:
    offset = 0
    for batch_index, start in enumerate(range(0, len(rows), batch_size)):
        check_cancelled(cancel_token)
        records = [
            normalise_prediction_record(value)
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
