from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import pandas as pd

LABEL_TABLE_SCHEMA_VERSION = 1
LABEL_COLUMNS = (
    "label_id",
    "session_id",
    "dataset_id",
    "row_id",
    "label",
    "display_label",
    "status",
    "source",
    "round",
    "timestamp",
    "updated_at",
    "task_type",
)


@dataclass(frozen=True)
class LabelTableRef:
    schema_version: int
    storage: str
    uri: str
    format: str
    created_at: float
    session_id: str
    dataset_id: str
    target_column: str
    task_type: str
    record_id_column: str
    row_count: int
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
    def from_value(cls, value: "LabelTableRef | Mapping[str, Any]") -> "LabelTableRef":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("label table reference must be a mapping")
        return cls(
            schema_version=int(value.get("schema_version") or LABEL_TABLE_SCHEMA_VERSION),
            storage=str(value.get("storage") or "local_file"),
            uri=str(value.get("uri") or ""),
            format=str(value.get("format") or "jsonl.gz"),
            created_at=float(value.get("created_at") or 0.0),
            session_id=str(value.get("session_id") or ""),
            dataset_id=str(value.get("dataset_id") or ""),
            target_column=str(value.get("target_column") or "al_label"),
            task_type=str(value.get("task_type") or "classification"),
            record_id_column=str(value.get("record_id_column") or "row_id"),
            row_count=int(value.get("row_count") or 0),
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


class LabelTableWriter:
    """Write a complete, versioned label snapshot without retaining a dataframe."""

    def __init__(
        self,
        *,
        root: Path | str,
        session_id: str,
        dataset_id: str,
        target_column: str,
        task_type: str,
        storage_format: str = "auto",
    ) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_id = str(session_id)
        self.dataset_id = str(dataset_id)
        self.target_column = str(target_column or "al_label")
        self.task_type = str(task_type or "classification")
        self.created_at = time.time()
        self.row_count = 0
        self._closed = False
        self.columns = list(LABEL_COLUMNS)
        if self.target_column not in self.columns:
            self.columns.append(self.target_column)

        requested = str(storage_format or "auto").strip().lower()
        if requested not in {"auto", "parquet", "jsonl", "jsonl.gz"}:
            raise ValueError("label_storage_format must be auto, parquet, or jsonl.gz")
        self._write_parquet = requested not in {"jsonl", "jsonl.gz"} and parquet_available()
        if requested == "parquet" and not self._write_parquet:
            raise RuntimeError("Parquet label output requires duckdb, pyarrow, or fastparquet.")

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
            raise RuntimeError("Label writer is closed.")
        if not rows:
            return
        normalized = [self._normalize_row(row) for row in rows]
        for row in normalized:
            self._json_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            self._json_handle.write("\n")
        if self._write_parquet:
            frame = pd.DataFrame.from_records(normalized, columns=self.columns)
            for column in ("label_id", "session_id", "dataset_id", "row_id", "display_label", "status", "source", "task_type"):
                frame[column] = frame[column].astype("string")
            frame["round"] = pd.to_numeric(frame["round"], errors="coerce").astype("Int64")
            for column in ("timestamp", "updated_at"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
            if self.task_type == "regression":
                frame["label"] = pd.to_numeric(frame["label"], errors="coerce").astype("Float64")
                frame[self.target_column] = pd.to_numeric(frame[self.target_column], errors="coerce").astype("Float64")
            else:
                frame["label"] = frame["label"].astype("string")
                frame[self.target_column] = frame[self.target_column].astype("string")
            part = self.parquet_temp_path / f"part-{len(self.parquet_parts):06d}.parquet"
            _write_parquet(frame, part)
            self.parquet_parts.append(part)
        self.row_count += len(normalized)

    def _normalize_row(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        row = dict(value or {})
        row_id = str(row.get("row_id") or "").strip()
        label = _json_value(row.get("label"))
        normalized = {
            "label_id": str(row.get("label_id") or f"{self.session_id}:{row_id}"),
            "session_id": self.session_id,
            "dataset_id": self.dataset_id,
            "row_id": row_id,
            "label": label,
            "display_label": str(
                row.get("display_label")
                if row.get("display_label") not in (None, "")
                else ("" if label is None else label)
            ),
            "status": str(row.get("status") or "verified"),
            "source": str(row.get("source") or "manual"),
            "round": int(row.get("round") or 0),
            "timestamp": float(row.get("timestamp") or self.created_at),
            "updated_at": float(row.get("updated_at") or row.get("timestamp") or self.created_at),
            "task_type": self.task_type,
        }
        normalized[self.target_column] = label
        return {column: normalized.get(column) for column in self.columns}

    def finalize(self, *, content_sha256: str) -> LabelTableRef:
        if self._closed:
            raise RuntimeError("Label writer is already closed.")
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
            return LabelTableRef(
                schema_version=LABEL_TABLE_SCHEMA_VERSION,
                storage="local_file",
                uri=str(self.final_path),
                format="jsonl.gz",
                created_at=self.created_at,
                session_id=self.session_id,
                dataset_id=self.dataset_id,
                target_column=self.target_column,
                task_type=self.task_type,
                record_id_column="row_id",
                row_count=self.row_count,
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

    def __enter__(self) -> "LabelTableWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or not self._closed:
            self.abort()
        return False


def iter_label_rows_from_session(
    session: Mapping[str, Any],
) -> Iterator[Dict[str, Any]]:
    labels = dict(session.get("labels") or {})
    for row_id in sorted(labels, key=str):
        entry = dict(labels[row_id] or {})
        entry["row_id"] = str(entry.get("row_id") or row_id)
        yield entry


def label_rows_from_session(session: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return list(iter_label_rows_from_session(session))

def label_content_sha256(session: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for row in iter_label_rows_from_session(session):
        payload = {
            "row_id": str(row.get("row_id") or ""),
            "label": _json_value(row.get("label")),
            "display_label": str(row.get("display_label") or ""),
            "status": str(row.get("status") or ""),
            "source": str(row.get("source") or ""),
            "round": int(row.get("round") or 0),
            "timestamp": float(row.get("timestamp") or 0.0),
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def ensure_label_table(
    context: Any,
    session: Mapping[str, Any],
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> tuple[LabelTableRef, bool]:
    params = dict(params or {})
    content_sha = label_content_sha256(session)
    existing_raw = session.get("label_table_ref")
    if isinstance(existing_raw, Mapping):
        try:
            existing = LabelTableRef.from_value(existing_raw)
        except Exception:
            existing = None
        if existing is not None and existing.content_sha256 == content_sha and Path(existing.uri).is_file():
            return existing, False

    root = label_output_root(context, params)
    with LabelTableWriter(
        root=root,
        session_id=str(session.get("session_id") or "session"),
        dataset_id=str(session.get("pool_dataset_id") or session.get("dataset_id") or ""),
        target_column=str(session.get("target_column") or "al_label"),
        task_type=str(session.get("task_type") or "classification"),
        storage_format=str(params.get("label_storage_format") or "auto"),
    ) as writer:
        batch_size = max(1, int(params.get("label_storage_batch_size") or 8192))
        batch: list[Dict[str, Any]] = []
        for row in iter_label_rows_from_session(session):
            batch.append(row)
            if len(batch) >= batch_size:
                writer.write_rows(batch)
                batch = []
        if batch:
            writer.write_rows(batch)
        ref = writer.finalize(content_sha256=content_sha)
    return ref, True


def iter_label_rows(ref: LabelTableRef | Mapping[str, Any]) -> Iterator[Dict[str, Any]]:
    resolved = LabelTableRef.from_value(ref)
    path = Path(resolved.uri).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Label sidecar does not exist: {path}")
    opener = gzip.open if path.suffix.lower() == ".gz" or resolved.format.lower() in {"jsonl.gz", "gzip-jsonl"} else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            if not isinstance(raw, Mapping):
                raise TypeError(f"Label row {line_number} must be an object")
            yield dict(raw)


def label_output_root(context: Any, params: Mapping[str, Any]) -> Path:
    explicit = params.get("label_output_dir") or params.get("artifact_root")
    if explicit:
        root = Path(str(explicit)).expanduser()
    else:
        cache_dir = getattr(getattr(context, "artifacts", None), "_cache_dir", None)
        root = Path(cache_dir).expanduser() / "active_learning" / "labels" if cache_dir else Path.cwd() / ".astronomical" / "active_learning" / "labels"
    root.mkdir(parents=True, exist_ok=True)
    return root


def artifact_row_ids_ref(ref: LabelTableRef | Mapping[str, Any]) -> Dict[str, Any]:
    resolved = LabelTableRef.from_value(ref)
    if resolved.parquet_parts:
        return {
            "storage": resolved.storage,
            "uri": str(resolved.parquet_uri or resolved.parquet_parts[0]),
            "format": "parquet",
            "row_count": resolved.row_count,
            "sha256": resolved.parquet_sha256,
            "id_column": resolved.record_id_column,
            "parts": list(resolved.parquet_parts),
            "params": {"schema_version": resolved.schema_version, "session_id": resolved.session_id},
        }
    return {
        "storage": resolved.storage,
        "uri": resolved.uri,
        "format": resolved.format,
        "row_count": resolved.row_count,
        "sha256": resolved.sha256,
        "id_column": resolved.record_id_column,
        "parts": list(resolved.parts),
        "params": {"schema_version": resolved.schema_version, "session_id": resolved.session_id},
    }


def parquet_available() -> bool:
    for module in ("duckdb", "pyarrow", "fastparquet"):
        try:
            __import__(module)
            return True
        except Exception:
            continue
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
        connection.register("label_batch", frame)
        escaped = str(path).replace("'", "''")
        connection.execute(f"COPY label_batch TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
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


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)
