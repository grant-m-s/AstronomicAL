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
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from ..serialization import json_safe


@dataclass(frozen=True)
class PredictionTableRef:
    """Durable prediction table references.

    ``uri`` remains a gzip JSONL file so existing artifact readers continue to
    work. When a Parquet engine is available, ``parquet_parts`` exposes the
    same rows as a lazy dataset without requiring another in-memory table.
    """

    storage: str
    uri: str
    format: str
    created_at: float
    row_count: int
    columns: list[str]
    sha256: str
    size_bytes: int
    parts: list[str]
    parquet_uri: Optional[str] = None
    parquet_parts: list[str] = field(default_factory=list)
    parquet_sha256: Optional[str] = None
    parquet_size_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["parquet_parts"] = list(self.parquet_parts or [])
        return json_safe(payload)


def prediction_output_root(
    context: Any,
    params: Optional[Mapping[str, Any]] = None,
) -> Path:
    params = dict(params or {})
    explicit = params.get("prediction_output_dir") or params.get("artifact_root")
    if explicit:
        root = Path(str(explicit)).expanduser()
    else:
        root = None
        try:
            from .. import paths as path_utils

            resolver = getattr(path_utils, "ml_artifact_root", None)
            if callable(resolver):
                for args in ((context, params), (context,), (params,), ()):
                    try:
                        candidate = resolver(*args)
                    except TypeError:
                        continue
                    if candidate:
                        root = Path(candidate)
                        break
        except Exception:
            root = None
        if root is None:
            config = getattr(context, "config", None)
            configured = getattr(config, "ml_artifact_root", None)
            root = (
                Path(str(configured)).expanduser()
                if configured
                else Path.cwd() / ".astronomical" / "ml_artifacts"
            )
    output = root / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    return output


class PredictionTableWriter:
    """Incrementally write canonical JSONL and an optional Parquet mirror."""

    def __init__(
        self,
        *,
        root: Path | str,
        run_id: str,
        dataset_id: str,
        model_artifact_id: str,
        storage_format: str = "auto",
        columns: Optional[Sequence[str]] = None,
        column_types: Optional[Mapping[str, str]] = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.dataset_id = str(dataset_id)
        self.model_artifact_id = str(model_artifact_id)
        self.created_at = time.time()
        self.row_count = 0
        self.columns = list(dict.fromkeys(str(value) for value in columns or []))
        self.column_types = {
            str(column): str(dtype)
            for column, dtype in dict(column_types or {}).items()
        }
        self._closed = False
        requested = str(storage_format or "auto").strip().lower()
        if requested not in {"auto", "parquet", "jsonl", "jsonl.gz"}:
            raise ValueError(
                "prediction_storage_format must be auto, parquet, or jsonl.gz"
            )
        self._write_parquet = requested not in {"jsonl", "jsonl.gz"} and _parquet_available()
        if requested == "parquet" and not self._write_parquet:
            raise RuntimeError(
                "Parquet prediction output requires duckdb, pyarrow, or fastparquet."
            )

        token = (
            f"{_safe_name(self.dataset_id)}-{_safe_name(self.run_id)}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        self.final_path = self.root / f"{token}.jsonl.gz"
        self.temp_path = self.root / f".{token}.jsonl.gz.tmp"
        self._json_handle = gzip.open(
            self.temp_path,
            "wt",
            encoding="utf-8",
            newline="\n",
        )
        self.parquet_final_path = self.root / f"{token}.parquet.d"
        self.parquet_temp_path = self.root / f".{token}.parquet.d.tmp"
        self.parquet_parts: list[Path] = []
        if self._write_parquet:
            self.parquet_temp_path.mkdir(parents=True, exist_ok=False)

    def write_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self._closed:
            raise RuntimeError("Prediction writer is closed.")
        if not rows:
            return
        cleaned = [dict(json_safe(dict(row))) for row in rows]
        incoming_columns = list(
            dict.fromkeys(key for row in cleaned for key in row.keys())
        )
        if not self.columns:
            self.columns = incoming_columns
        else:
            new_columns = [
                column for column in incoming_columns if column not in self.columns
            ]
            if new_columns:
                raise ValueError(
                    "Prediction output schema changed between batches; new columns: "
                    + ", ".join(new_columns)
                )

        normalized = [
            {column: row.get(column) for column in self.columns}
            for row in cleaned
        ]
        for row in normalized:
            self._json_handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            )
            self._json_handle.write("\n")

        if self._write_parquet:
            frame = pd.DataFrame.from_records(normalized).reindex(columns=self.columns)
            frame = _apply_column_types(frame, self.column_types)
            path = self.parquet_temp_path / (
                f"part-{len(self.parquet_parts):06d}.parquet"
            )
            _write_parquet_frame(frame, path)
            self.parquet_parts.append(path)
        self.row_count += len(normalized)

    def finalize(self) -> PredictionTableRef:
        if self._closed:
            raise RuntimeError("Prediction writer is already closed.")
        try:
            self._json_handle.close()
            os.replace(self.temp_path, self.final_path)
            json_parts = [self.final_path]

            final_parquet_parts: list[Path] = []
            if self._write_parquet:
                os.replace(self.parquet_temp_path, self.parquet_final_path)
                final_parquet_parts = sorted(
                    self.parquet_final_path.glob("part-*.parquet")
                )

            self._closed = True
            json_digest = _paths_sha256(json_parts)
            json_size = sum(path.stat().st_size for path in json_parts)
            parquet_digest = (
                _paths_sha256(final_parquet_parts)
                if final_parquet_parts
                else None
            )
            parquet_size = (
                sum(path.stat().st_size for path in final_parquet_parts)
                if final_parquet_parts
                else None
            )
            return PredictionTableRef(
                storage="local_file",
                uri=str(self.final_path),
                format="jsonl.gz",
                created_at=self.created_at,
                row_count=int(self.row_count),
                columns=list(self.columns),
                sha256=json_digest,
                size_bytes=int(json_size),
                parts=[str(self.final_path)],
                parquet_uri=(
                    str(self.parquet_final_path)
                    if final_parquet_parts
                    else None
                ),
                parquet_parts=[str(path) for path in final_parquet_parts],
                parquet_sha256=parquet_digest,
                parquet_size_bytes=parquet_size,
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

    def __enter__(self) -> "PredictionTableWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or not self._closed:
            self.abort()
        return False


def delete_prediction_ref(
    ref: PredictionTableRef | Mapping[str, Any] | None,
) -> None:
    if ref is None:
        return
    payload = ref.to_dict() if isinstance(ref, PredictionTableRef) else dict(ref)
    for key in ("uri", "parquet_uri"):
        value = payload.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _safe_name(value: Any) -> str:
    text = "".join(
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "-"
        for character in str(value or "prediction")
    ).strip("-._")
    return text[:96] or "prediction"


def _parquet_available() -> bool:
    for module in ("duckdb", "pyarrow", "fastparquet"):
        try:
            __import__(module)
            return True
        except Exception:
            continue
    return False


def _apply_column_types(
    frame: pd.DataFrame,
    column_types: Mapping[str, str],
) -> pd.DataFrame:
    result = frame.copy()
    for column, dtype in column_types.items():
        if column not in result.columns:
            continue
        if dtype == "string":
            result[column] = result[column].astype("string")
        elif dtype == "float":
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            ).astype("Float64")
        elif dtype == "boolean":
            result[column] = result[column].astype("boolean")
        elif dtype == "integer":
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            ).astype("Int64")
    return result


def _write_parquet_frame(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=False)
        return
    except ImportError:
        pass
    try:
        import duckdb
    except Exception as exc:
        raise RuntimeError(
            "No Parquet writer is available. Install duckdb, pyarrow, or fastparquet."
        ) from exc
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("prediction_batch", frame)
        quoted_path = str(path).replace("'", "''")
        connection.execute(
            f"COPY prediction_batch TO '{quoted_path}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
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
