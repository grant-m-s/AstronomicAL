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


@dataclass(frozen=True)
class TrainingSetRef:
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
        return asdict(self)


class TrainingSetWriter:
    """Write bounded training batches to JSONL and a Parquet dataset directory."""

    def __init__(
        self,
        *,
        root: Path | str,
        session_id: str,
        round_index: int,
        require_parquet: bool = True,
    ):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        if require_parquet and not parquet_available():
            raise RuntimeError(
                "Large active-learning training sets require duckdb, pyarrow, or fastparquet."
            )
        self._write_parquet = parquet_available()
        self.created_at = time.time()
        self.row_count = 0
        self.columns: list[str] = []
        self._closed = False
        token = f"{_safe_name(session_id)}-r{int(round_index)}-{uuid.uuid4().hex[:10]}"
        self.final_path = self.root / f"{token}.jsonl.gz"
        self.temp_path = self.root / f".{token}.jsonl.gz.tmp"
        self._json_handle = gzip.open(self.temp_path, "wt", encoding="utf-8", newline="\n")
        self.parquet_final_path = self.root / f"{token}.parquet.d"
        self.parquet_temp_path = self.root / f".{token}.parquet.d.tmp"
        self.parquet_parts: list[Path] = []
        if self._write_parquet:
            self.parquet_temp_path.mkdir(parents=True, exist_ok=False)

    def write_frame(self, frame: pd.DataFrame) -> None:
        if self._closed:
            raise RuntimeError("Training-set writer is closed.")
        if frame is None or frame.empty:
            return
        incoming = [str(column) for column in frame.columns]
        if not self.columns:
            self.columns = incoming
        elif incoming != self.columns:
            raise ValueError(
                "Training-set schema changed between batches: "
                f"expected {self.columns!r}, got {incoming!r}."
            )

        clean = frame.copy()
        for row in clean.to_dict(orient="records"):
            payload = {key: _json_value(value) for key, value in row.items()}
            self._json_handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            self._json_handle.write("\n")
        if self._write_parquet:
            part = self.parquet_temp_path / f"part-{len(self.parquet_parts):06d}.parquet"
            _write_parquet(clean, part)
            self.parquet_parts.append(part)
        self.row_count += len(clean)

    def finalize(self) -> TrainingSetRef:
        if self._closed:
            raise RuntimeError("Training-set writer is already closed.")
        try:
            self._json_handle.close()
            os.replace(self.temp_path, self.final_path)
            final_parquet_parts: list[Path] = []
            if self._write_parquet:
                os.replace(self.parquet_temp_path, self.parquet_final_path)
                final_parquet_parts = sorted(self.parquet_final_path.glob("part-*.parquet"))
            self._closed = True
            return TrainingSetRef(
                storage="local_file",
                uri=str(self.final_path),
                format="jsonl.gz",
                created_at=self.created_at,
                row_count=self.row_count,
                columns=list(self.columns),
                sha256=_paths_sha256([self.final_path]),
                size_bytes=self.final_path.stat().st_size,
                parts=[str(self.final_path)],
                parquet_uri=str(self.parquet_final_path) if final_parquet_parts else None,
                parquet_parts=[str(path) for path in final_parquet_parts],
                parquet_sha256=_paths_sha256(final_parquet_parts) if final_parquet_parts else None,
                parquet_size_bytes=(
                    sum(path.stat().st_size for path in final_parquet_parts)
                    if final_parquet_parts
                    else None
                ),
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

    def __enter__(self) -> "TrainingSetWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or not self._closed:
            self.abort()
        return False


def training_output_root(context: Any, params: Mapping[str, Any]) -> Path:
    explicit = params.get("training_output_dir") or params.get("artifact_root")
    if explicit:
        root = Path(str(explicit)).expanduser()
    else:
        root = Path.cwd() / ".astronomical" / "active_learning" / "training_sets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def delete_training_ref(ref: TrainingSetRef | Mapping[str, Any] | None) -> None:
    if ref is None:
        return
    payload = ref.to_dict() if isinstance(ref, TrainingSetRef) else dict(ref)
    for key in ("uri", "parquet_uri"):
        value = payload.get(key)
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
        connection.register("training_batch", frame)
        escaped = str(path).replace("'", "''")
        connection.execute(f"COPY training_batch TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
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
    text = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "-"
        for character in str(value or "session")
    ).strip("-._")
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
