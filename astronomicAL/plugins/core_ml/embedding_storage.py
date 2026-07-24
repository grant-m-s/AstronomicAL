from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .artifacts import StoredEmbeddingIndexRef, StoredEmbeddingRef
from .serialization import json_safe

EMBEDDING_SIDECAR_SCHEMA_VERSION = 1


class EmbeddingTableWriter:
    """Incrementally write a standard embedding sidecar.

    JSONL remains the canonical representation.  Each Parquet part stores one
    scalar float column per embedding dimension, which makes the sidecar usable
    by DuckDB without materialising nested Python lists.
    """

    def __init__(
        self,
        *,
        root: Path | str,
        run_id: str,
        dataset_id: str,
        model_artifact_id: str = "",
        dimensions: int,
        dtype: str = "float32",
        storage_format: str = "auto",
        metadata_columns: Optional[Sequence[str]] = None,
        index_ref: Optional[StoredEmbeddingIndexRef | Mapping[str, Any]] = None,
    ) -> None:
        dimensions = int(dimensions)
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero.")
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.dataset_id = str(dataset_id)
        self.model_artifact_id = str(model_artifact_id or "")
        self.dimensions = dimensions
        self.dtype = str(dtype or "float32")
        self.created_at = time.time()
        self.record_id_column = "row_id"
        self.embedding_column = "embedding"
        self.embedding_columns = [f"embedding_{index:06d}" for index in range(dimensions)]
        self.metadata_columns = [
            str(column)
            for column in dict.fromkeys(metadata_columns or [])
            if str(column) not in {self.record_id_column, self.embedding_column}
            and str(column) not in self.embedding_columns
        ]
        self.columns = [self.record_id_column, self.embedding_column, *self.metadata_columns]
        self.index_ref = (
            StoredEmbeddingIndexRef.from_value(index_ref)
            if isinstance(index_ref, Mapping)
            else index_ref
        )
        self.row_count = 0
        self._closed = False

        requested = str(storage_format or "auto").strip().lower()
        if requested not in {"auto", "parquet", "jsonl", "jsonl.gz"}:
            raise ValueError("embedding_storage_format must be auto, parquet, or jsonl.gz")
        self._write_parquet = requested not in {"jsonl", "jsonl.gz"} and parquet_available()
        if requested == "parquet" and not self._write_parquet:
            raise RuntimeError("Parquet embedding output requires duckdb, pyarrow, or fastparquet.")

        token = f"{_safe_name(self.dataset_id)}-{_safe_name(self.run_id)}-{uuid.uuid4().hex[:8]}"
        self.final_path = self.root / f"{token}.jsonl.gz"
        self.temp_path = self.root / f".{token}.jsonl.gz.tmp"
        self._json_handle = gzip.open(self.temp_path, "wt", encoding="utf-8", newline="\n")
        self.parquet_final_path = self.root / f"{token}.parquet.d"
        self.parquet_temp_path = self.root / f".{token}.parquet.d.tmp"
        self.parquet_parts: list[Path] = []
        if self._write_parquet:
            self.parquet_temp_path.mkdir(parents=True, exist_ok=False)

    def write_batch(
        self,
        *,
        row_ids: Sequence[Any],
        embeddings: Any,
        metadata_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Embedding writer is closed.")
        vectors = np.asarray(embeddings)
        if vectors.ndim != 2 or int(vectors.shape[1]) != self.dimensions:
            raise ValueError(
                "Embedding batch shape mismatch: expected (*, "
                f"{self.dimensions}), got {tuple(vectors.shape)!r}."
            )
        if len(row_ids) != int(vectors.shape[0]):
            raise ValueError("row_ids length does not match embedding batch rows")
        metadata_rows = list(metadata_rows or [{} for _ in row_ids])
        if len(metadata_rows) != len(row_ids):
            raise ValueError("metadata_rows length does not match embedding batch rows")
        vectors = np.asarray(vectors, dtype=self.dtype)

        json_rows: list[Dict[str, Any]] = []
        parquet_records: list[Dict[str, Any]] = []
        for row_id, vector, metadata in zip(row_ids, vectors, metadata_rows):
            row_id_text = str(row_id)
            metadata_payload = {
                column: json_safe(dict(metadata or {}).get(column))
                for column in self.metadata_columns
            }
            json_row = {
                self.record_id_column: row_id_text,
                self.embedding_column: [float(value) for value in vector.tolist()],
                **metadata_payload,
            }
            json_rows.append(json_row)
            parquet_row = {self.record_id_column: row_id_text, **metadata_payload}
            parquet_row.update(
                {
                    column: float(value)
                    for column, value in zip(self.embedding_columns, vector.tolist())
                }
            )
            parquet_records.append(parquet_row)

        for row in json_rows:
            self._json_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            self._json_handle.write("\n")
        if self._write_parquet and parquet_records:
            parquet_columns = [self.record_id_column, *self.embedding_columns, *self.metadata_columns]
            frame = pd.DataFrame.from_records(parquet_records, columns=parquet_columns)
            frame[self.record_id_column] = frame[self.record_id_column].astype("string")
            for column in self.embedding_columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float32")
            part = self.parquet_temp_path / f"part-{len(self.parquet_parts):06d}.parquet"
            _write_parquet(frame, part)
            self.parquet_parts.append(part)
        self.row_count += len(json_rows)

    def finalize(self, *, metadata: Optional[Mapping[str, Any]] = None) -> StoredEmbeddingRef:
        if self._closed:
            raise RuntimeError("Embedding writer is already closed.")
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
            return StoredEmbeddingRef(
                schema_version=EMBEDDING_SIDECAR_SCHEMA_VERSION,
                storage="local_file",
                uri=str(self.final_path),
                format="jsonl.gz",
                created_at=self.created_at,
                dataset_id=self.dataset_id,
                model_artifact_id=self.model_artifact_id,
                row_count=int(self.row_count),
                dimensions=self.dimensions,
                dtype=self.dtype,
                record_id_column=self.record_id_column,
                embedding_column=self.embedding_column,
                embedding_columns=list(self.embedding_columns),
                columns=list(self.columns),
                sha256=_paths_sha256([self.final_path]),
                size_bytes=self.final_path.stat().st_size,
                parts=[str(self.final_path)],
                parquet_uri=str(self.parquet_final_path) if final_parquet_parts else None,
                parquet_parts=[str(path) for path in final_parquet_parts],
                parquet_sha256=_paths_sha256(final_parquet_parts) if final_parquet_parts else None,
                parquet_size_bytes=(sum(path.stat().st_size for path in final_parquet_parts) if final_parquet_parts else None),
                index_ref=self.index_ref,
                metadata=dict(json_safe(dict(metadata or {}))),
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

    def __enter__(self) -> "EmbeddingTableWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or not self._closed:
            self.abort()
        return False


def embedding_output_root(context: Any, params: Optional[Mapping[str, Any]] = None) -> Path:
    params = dict(params or {})
    explicit = params.get("embedding_output_dir") or params.get("artifact_root")
    if explicit:
        root = Path(str(explicit)).expanduser()
    else:
        cache_dir = getattr(getattr(context, "artifacts", None), "_cache_dir", None)
        root = Path(cache_dir).expanduser() / "embeddings" if cache_dir else Path.cwd() / ".astronomical" / "ml_artifacts" / "embeddings"
    root.mkdir(parents=True, exist_ok=True)
    return root


def iter_embedding_batches(
    ref: StoredEmbeddingRef | Mapping[str, Any],
    *,
    batch_size: int = 8192,
    cancel_token: Any = None,
) -> Iterator[tuple[list[str], np.ndarray, list[Dict[str, Any]]]]:
    resolved = StoredEmbeddingRef.from_value(ref)
    path = Path(resolved.uri).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Embedding sidecar does not exist: {path}")
    batch_size = max(1, int(batch_size))
    row_ids: list[str] = []
    vectors: list[list[float]] = []
    metadata_rows: list[Dict[str, Any]] = []
    with _open_text(path, resolved.format) as handle:
        for line_number, line in enumerate(handle, start=1):
            _check_cancelled(cancel_token)
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            if not isinstance(raw, Mapping):
                raise TypeError(f"Embedding row {line_number} must be an object")
            row_id = str(raw.get(resolved.record_id_column) or raw.get("row_id") or "")
            vector = raw.get(resolved.embedding_column)
            if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes, bytearray)):
                raise TypeError(f"Embedding row {line_number} is missing a vector sequence")
            if len(vector) != resolved.dimensions:
                raise ValueError(
                    f"Embedding row {line_number} has {len(vector)} values; expected {resolved.dimensions}."
                )
            row_ids.append(row_id)
            vectors.append([float(value) for value in vector])
            metadata_rows.append(
                {
                    str(key): json_safe(value)
                    for key, value in raw.items()
                    if str(key) not in {resolved.record_id_column, resolved.embedding_column}
                }
            )
            if len(row_ids) >= batch_size:
                yield row_ids, np.asarray(vectors, dtype=resolved.dtype), metadata_rows
                row_ids, vectors, metadata_rows = [], [], []
    if row_ids:
        yield row_ids, np.asarray(vectors, dtype=resolved.dtype), metadata_rows


def delete_embedding_ref(ref: StoredEmbeddingRef | Mapping[str, Any] | None) -> None:
    if ref is None:
        return
    resolved = StoredEmbeddingRef.from_value(ref)
    for value in (resolved.uri, resolved.parquet_uri):
        if not value:
            continue
        path = Path(str(value))
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


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
        connection.register("embedding_batch", frame)
        escaped = str(path).replace("'", "''")
        connection.execute(f"COPY embedding_batch TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        connection.close()


def _open_text(path: Path, format_name: str):
    if path.suffix.lower() == ".gz" or str(format_name).lower() in {"jsonl.gz", "gzip-jsonl"}:
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")


def _check_cancelled(cancel_token: Any) -> None:
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


def _paths_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: Any) -> str:
    text = "".join(character if character.isalnum() or character in {"-", "_", "."} else "-" for character in str(value or "embedding")).strip("-._")
    return text[:96] or "embedding"
