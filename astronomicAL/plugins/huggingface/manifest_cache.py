from __future__ import annotations

"""Bounded, revision-scoped Parquet manifests for Hugging Face image-builder imports."""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_BATCH_SIZE = 512
PREVIEW_ROWS = 25


def immutable_revision(revision: str | None) -> bool:
    """Reuse completed imports only when the Hub revision is an immutable commit hash."""
    return bool(re.fullmatch(r"[a-fA-F0-9]{40,64}", str(revision or "")))


def cache_identity(
    *, repo_id: str, revision: str, config_name: str | None, split: str,
    image_column: str | None, label_column: str | None, id_column: str | None,
    max_rows: int, assets_dir: Path, trust_remote_code: bool = False,
    label_semantics_version: int = 1,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "repo_id": repo_id, "revision": revision, "config_name": config_name or "", "split": split,
        "image_column": image_column or "", "label_column": label_column or "",
        "id_column": id_column or "", "max_rows": int(max_rows),
        "trust_remote_code": bool(trust_remote_code),
        "label_semantics_version": int(label_semantics_version),
        "assets_dir": str(assets_dir.resolve()),
    }


def manifest_paths(cache_root: Path, identity: Mapping[str, Any]) -> tuple[Path, Path]:
    encoded = json.dumps(dict(identity), sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    root = Path(cache_root).expanduser().resolve() / "builder_manifests"
    return root / f"{digest}.parquet", root / f"{digest}.json"


def pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
        return True
    except Exception:
        return False


def read_completed_manifest(
    parquet_path: Path, metadata_path: Path, identity: Mapping[str, Any], *,
    verify_local_images: bool = True,
) -> dict[str, Any] | None:
    """Return a complete, matching manifest, or None without trusting partial outputs."""
    if not parquet_path.is_file() or not metadata_path.is_file():
        return None
    try:
        import pyarrow.parquet as pq

        manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != dict(identity):
            return None
        parquet = pq.ParquetFile(str(parquet_path))
        rows = int(parquet.metadata.num_rows)
        columns = list(parquet.schema_arrow.names)
        if not rows or rows != int(manifest.get("rows") or 0):
            return None
        if columns != list(manifest.get("columns") or []):
            return None
        if not {"record_id", "image_path", "image_uri"}.issubset(columns):
            return None
        if verify_local_images:
            # Validate path existence without decoding images or materializing the manifest.
            for batch in parquet.iter_batches(batch_size=4096, columns=["image_path"]):
                for value in batch.column(0).to_pylist():
                    path = str(value or "")
                    if path and "://" not in path and not Path(path).is_file():
                        return None
        return manifest
    except Exception:
        # A damaged or incompatible cache is a miss, never a partial dataset.
        return None


def cached_preview(parquet_path: Path, *, limit: int = PREVIEW_ROWS):
    """Read only the first bounded Arrow batch; never load the full Parquet table."""
    import pandas as pd
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(str(parquet_path))
    first = next(parquet.iter_batches(batch_size=max(1, limit)), None)
    return pd.DataFrame(first.to_pylist()[:limit]) if first is not None else pd.DataFrame()


def write_completed_manifest(
    metadata_path: Path, *, identity: Mapping[str, Any], rows: int,
    columns: list[str], image_column: str, label_column: str | None,
) -> None:
    """Publish completion only after a valid Parquet file has been committed."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "identity": dict(identity), "rows": int(rows), "columns": list(columns),
        "image_column": image_column, "label_column": label_column or "",
    }
    temp = metadata_path.with_name(f"{metadata_path.name}.{uuid.uuid4().hex}.partial")
    try:
        temp.write_text(json.dumps(info, sort_keys=True), encoding="utf-8")
        os.replace(temp, metadata_path)
    finally:
        temp.unlink(missing_ok=True)


class StreamingParquetManifest:
    """Write bounded row batches to a temporary Parquet file, then atomically commit."""

    def __init__(self, path: Path, *, batch_size: int = MANIFEST_BATCH_SIZE) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.pa = pa
        self.pq = pq
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.partial")
        self.batch_size = max(1, int(batch_size))
        self.rows = 0
        self.columns: list[str] = []
        self._batch: list[dict[str, Any]] = []
        self._writer: Any = None
        self._schema: Any = None
        self._finished = False
        self.preview: list[dict[str, Any]] = []

    def append(self, row: Mapping[str, Any]) -> None:
        record = dict(row)
        if len(self.preview) < PREVIEW_ROWS:
            self.preview.append(record)
        self._batch.append(record)
        self.rows += 1
        if len(self._batch) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._batch:
            return
        if self._writer is None:
            # PyArrow infers field names from the first row, not the union of rows.
            # Normalize the entire first batch so optional fields aren't lost.
            fields = dict.fromkeys(key for record in self._batch for key in record)
            normalized = [{key: record.get(key) for key in fields} for record in self._batch]
            table = self.pa.Table.from_pylist(normalized)
            self._schema = table.schema
            self.columns = list(self._schema.names)
            self._writer = self.pq.ParquetWriter(str(self.temp), self._schema)
        else:
            # Do not silently discard fields first seen in a later batch.
            unexpected = set().union(*(row.keys() for row in self._batch)) - set(self.columns)
            if unexpected:
                raise ValueError(
                    "New image-manifest fields appeared after the initial Parquet batch: "
                    + ", ".join(sorted(unexpected))
                )
            table = self.pa.Table.from_pylist(self._batch, schema=self._schema)
        self._writer.write_table(table)
        self._batch.clear()

    def finish(self) -> tuple[int, list[str]]:
        if self.rows <= 0:
            raise ValueError("No image rows were available to register.")
        self._flush()
        self._writer.close()
        self._writer = None
        os.replace(self.temp, self.path)
        self._finished = True
        return self.rows, self.columns

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if not self._finished:
            self.temp.unlink(missing_ok=True)

    def __enter__(self) -> "StreamingParquetManifest":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
