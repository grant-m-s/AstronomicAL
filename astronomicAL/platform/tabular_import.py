from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.feather as pafeather
import pyarrow.ipc as paipc
import pyarrow.parquet as pq

from astronomicAL.platform.fits_import import (
    ProgressStateCallback,
    import_fits_table_to_parquet,
)

ProgressCallback = Callable[[str], None]

_FULL_HASH_THRESHOLD = 256 * 1024**2
_HASH_SAMPLE_BYTES = 4 * 1024**2
_DISK_RESERVE_BYTES = 256 * 1024**2

_FORMAT_LABELS = {
    "fits": "FITS binary table",
    "parquet": "Parquet",
    "csv": "CSV",
    "tsv": "TSV",
    "jsonl": "JSON Lines",
    "excel": "Spreadsheet",
    "hdf5": "HDF5",
    "feather": "Feather",
    "arrow": "Arrow IPC",
    "ecsv": "Astropy ECSV",
    "sqlite": "SQLite",
}

_CONVERTIBLE_FORMATS = frozenset(
    {
        "fits",
        "csv",
        "tsv",
        "jsonl",
        "excel",
        "hdf5",
        "feather",
        "arrow",
        "ecsv",
        "sqlite",
    }
)


@dataclass(frozen=True)
class SourceFingerprint:
    size_bytes: int
    mtime_ns: int
    digest: str
    digest_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CacheAssessment:
    state: str
    message: str
    parquet_path: str
    metadata_path: str
    source_newer: bool = False
    fingerprint_checked: bool = False
    fingerprint_matches: Optional[bool] = None
    legacy_metadata: bool = False

    @property
    def exists(self) -> bool:
        return self.state != "missing"

    @property
    def requires_regeneration(self) -> bool:
        return self.state == "stale"


@dataclass(frozen=True)
class DiskSpaceAssessment:
    path: str
    free_bytes: Optional[int]
    estimated_output_bytes: Optional[int]
    required_bytes: Optional[int]
    enough_space: Optional[bool]
    message: str


@dataclass(frozen=True)
class ConversionResult:
    dataset_id: str
    source_path: str
    source_format: str
    parquet_path: str
    metadata_path: str
    created: bool
    metadata: dict[str, Any]


def emit_progress(message: str, progress_callback: Optional[ProgressCallback] = None) -> None:
    text = str(message)
    print(f"[AstronomicAL loader] {text}", flush=True)
    if progress_callback is None:
        return
    try:
        progress_callback(text)
    except Exception:
        # Progress reporting must never make a conversion fail.
        pass


def emit_progress_state(
    progress_state_callback: Optional[ProgressStateCallback],
    **state: Any,
) -> None:
    if progress_state_callback is None:
        return
    try:
        progress_state_callback(dict(state))
    except Exception:
        # Structured progress is a UI aid and must never make conversion fail.
        pass


def detect_source_format(path: str | Path) -> Optional[str]:
    name = str(path).lower()
    if name.endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        return "fits"
    if name.endswith((".parquet", ".pq")):
        return "parquet"
    if name.endswith((".csv", ".csv.gz")):
        return "csv"
    if name.endswith((".tsv", ".tsv.gz")):
        return "tsv"
    if name.endswith((".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")):
        return "jsonl"
    if name.endswith((".xlsx", ".xls", ".xlsm", ".ods")):
        return "excel"
    if name.endswith((".h5", ".hdf5", ".hdf")):
        return "hdf5"
    if name.endswith(".feather"):
        return "feather"
    if name.endswith((".arrow", ".ipc")):
        return "arrow"
    if name.endswith(".ecsv"):
        return "ecsv"
    if name.endswith((".sqlite", ".sqlite3", ".db")):
        return "sqlite"
    return None


def source_format_label(source_format: str) -> str:
    return _FORMAT_LABELS.get(str(source_format), str(source_format).upper())


def is_convertible_format(source_format: str) -> bool:
    return str(source_format) in _CONVERTIBLE_FORMATS


def source_stem(path: str | Path) -> str:
    name = Path(path).name
    lower = name.lower()
    suffixes = (
        ".fits.gz",
        ".fit.gz",
        ".csv.gz",
        ".tsv.gz",
        ".jsonl.gz",
        ".ndjson.gz",
        ".parquet",
        ".feather",
        ".sqlite3",
        ".sqlite",
        ".hdf5",
        ".xlsx",
        ".xlsm",
        ".jsonl",
        ".ndjson",
        ".arrow",
        ".fits",
        ".fit",
        ".csv",
        ".tsv",
        ".hdf",
        ".h5",
        ".ipc",
        ".ecsv",
        ".ods",
        ".xls",
        ".pq",
        ".db",
    )
    for suffix in suffixes:
        if lower.endswith(suffix):
            return name[: -len(suffix)] or "dataset"
    return Path(name).stem or "dataset"


def default_parquet_path(source_path: str | Path, dataset_id: str) -> Path:
    source = Path(source_path).expanduser().resolve()
    return source.parent / ".astronomical_cache" / f"{dataset_id}.parquet"


def metadata_path_for_parquet(parquet_path: str | Path) -> Path:
    path = Path(parquet_path).expanduser()
    return path.with_suffix(".metadata.json")


def inspect_source_subresources(
    source_path: str | Path,
    source_format: Optional[str] = None,
) -> tuple[str, ...]:
    path = Path(source_path).expanduser().resolve()
    fmt = source_format or detect_source_format(path)

    if fmt == "excel":
        try:
            with pd.ExcelFile(path) as workbook:
                return tuple(str(name) for name in workbook.sheet_names)
        except ImportError as exc:
            raise RuntimeError(
                "Spreadsheet support needs the pandas engine for this file type "
                "(for example openpyxl for .xlsx, xlrd for .xls, or odfpy for .ods)."
            ) from exc

    if fmt == "hdf5":
        pandas_keys: tuple[str, ...] = ()
        pandas_error: Optional[BaseException] = None
        try:
            with pd.HDFStore(path, mode="r") as store:
                pandas_keys = tuple(str(key) for key in store.keys())
        except BaseException as exc:
            pandas_error = exc

        if pandas_keys:
            return pandas_keys

        try:
            return _h5py_subresources(path)
        except ImportError as exc:
            if isinstance(pandas_error, ImportError):
                raise RuntimeError(
                    "HDF5 support requires either PyTables ('tables') for pandas HDFStore "
                    "files or h5py for generic HDF5 datasets."
                ) from exc
            if pandas_error is not None:
                raise RuntimeError(
                    f"Could not inspect HDF5 file with pandas ({pandas_error}) and h5py is not installed."
                ) from exc
            raise

    if fmt == "sqlite":
        with sqlite3.connect(str(path)) as con:
            rows = con.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    return ()


def _h5py_subresources(path: Path) -> tuple[str, ...]:
    import h5py

    candidates: list[str] = []
    with h5py.File(path, "r") as handle:
        def _visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                if obj.ndim == 1 and (obj.dtype.names or obj.dtype.kind in "biufcSUO"):
                    candidates.append("/" + name)
                elif obj.ndim == 2 and obj.shape[1] > 0:
                    candidates.append("/" + name)
                return
            if not isinstance(obj, h5py.Group):
                return
            lengths: list[int] = []
            usable = 0
            for child in obj.values():
                if not isinstance(child, h5py.Dataset) or child.ndim != 1:
                    continue
                if child.dtype.names is not None:
                    continue
                lengths.append(int(child.shape[0]))
                usable += 1
            if usable and len(set(lengths)) == 1:
                candidates.append("/" + name if name else "/")

        handle.visititems(_visit)

    # Prefer shallow paths and suppress duplicates while preserving readability.
    unique = sorted(set(candidates), key=lambda value: (value.count("/"), value.casefold()))
    return tuple(unique)


def fingerprint_source(
    source_path: str | Path,
    *,
    full_hash_threshold: int = _FULL_HASH_THRESHOLD,
    sample_bytes: int = _HASH_SAMPLE_BYTES,
) -> SourceFingerprint:
    path = Path(source_path).expanduser().resolve()
    stat = path.stat()
    size = int(stat.st_size)
    mtime_ns = int(stat.st_mtime_ns)
    digest = sha256()

    if size <= int(full_hash_threshold):
        mode = "sha256-full"
        with path.open("rb") as handle:
            while True:
                block = handle.read(8 * 1024**2)
                if not block:
                    break
                digest.update(block)
    else:
        mode = "sha256-sampled"
        digest.update(str(size).encode("ascii"))
        digest.update(b"|")
        sample = max(64 * 1024, int(sample_bytes))
        offsets = (0, max(0, (size // 2) - (sample // 2)), max(0, size - sample))
        with path.open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                digest.update(str(offset).encode("ascii"))
                digest.update(b":")
                digest.update(handle.read(sample))
                digest.update(b"|")

    return SourceFingerprint(
        size_bytes=size,
        mtime_ns=mtime_ns,
        digest=digest.hexdigest(),
        digest_mode=mode,
    )


def assess_parquet_cache(
    source_path: str | Path,
    parquet_path: str | Path,
    *,
    source_subresource: Optional[str] = None,
) -> CacheAssessment:
    source = Path(source_path).expanduser().resolve()
    parquet = Path(parquet_path).expanduser().resolve()
    metadata_path = metadata_path_for_parquet(parquet)

    if not parquet.is_file():
        return CacheAssessment(
            state="missing",
            message="No generated Parquet exists yet.",
            parquet_path=str(parquet),
            metadata_path=str(metadata_path),
        )

    try:
        source_stat = source.stat()
        parquet_stat = parquet.stat()
        source_newer = int(source_stat.st_mtime_ns) > int(parquet_stat.st_mtime_ns)
    except OSError:
        source_newer = False

    metadata = _read_json(metadata_path)
    stored_fingerprint = metadata.get("source_fingerprint")
    stored_source = str(metadata.get("source_path") or "")
    stored_subresource = metadata.get("source_subresource")

    if not metadata:
        if source_newer:
            return CacheAssessment(
                state="stale",
                message=(
                    "The source is newer than this legacy Parquet cache and no "
                    "content fingerprint is available. Regeneration is recommended."
                ),
                parquet_path=str(parquet),
                metadata_path=str(metadata_path),
                source_newer=True,
                legacy_metadata=True,
            )
        return CacheAssessment(
            state="unknown",
            message=(
                "This Parquet predates cache fingerprints. Its timestamp does not "
                "indicate staleness, but AstronomicAL cannot verify the content."
            ),
            parquet_path=str(parquet),
            metadata_path=str(metadata_path),
            source_newer=source_newer,
            legacy_metadata=True,
        )

    if stored_source and _canonical_path(stored_source) != _canonical_path(source):
        return CacheAssessment(
            state="stale",
            message="The cache metadata points at a different source file.",
            parquet_path=str(parquet),
            metadata_path=str(metadata_path),
            source_newer=source_newer,
        )

    current_subresource = source_subresource or None
    recorded_subresource: Optional[str] = None
    has_recorded_subresource = False
    if "source_subresource" in metadata:
        recorded_subresource = (
            str(metadata.get("source_subresource"))
            if metadata.get("source_subresource") not in (None, "")
            else None
        )
        has_recorded_subresource = True
    elif "source_hdu" in metadata and current_subresource is not None:
        recorded_subresource = str(metadata.get("source_hdu"))
        has_recorded_subresource = True

    if has_recorded_subresource and recorded_subresource != current_subresource:
        return CacheAssessment(
            state="stale",
            message=(
                "The selected sheet/table differs from the one used to create "
                "this Parquet cache."
            ),
            parquet_path=str(parquet),
            metadata_path=str(metadata_path),
            source_newer=source_newer,
        )

    if isinstance(stored_fingerprint, dict):
        stored_size = _safe_int(stored_fingerprint.get("size_bytes"))
        stored_mtime_ns = _safe_int(stored_fingerprint.get("mtime_ns"))
        current_size = _safe_int(getattr(source_stat, "st_size", None)) if "source_stat" in locals() else None
        current_mtime_ns = _safe_int(getattr(source_stat, "st_mtime_ns", None)) if "source_stat" in locals() else None

        if stored_size == current_size and stored_mtime_ns == current_mtime_ns:
            return CacheAssessment(
                state="fresh",
                message="Source size, modification time, and recorded fingerprint are unchanged.",
                parquet_path=str(parquet),
                metadata_path=str(metadata_path),
                source_newer=source_newer,
            )

        try:
            current_fingerprint = fingerprint_source(source)
        except OSError:
            current_fingerprint = None

        if current_fingerprint is not None:
            digest_matches = str(stored_fingerprint.get("digest") or "") == current_fingerprint.digest
            if digest_matches:
                return CacheAssessment(
                    state="fresh",
                    message=(
                        "The source timestamp/size changed, but its content fingerprint "
                        "still matches the cache."
                    ),
                    parquet_path=str(parquet),
                    metadata_path=str(metadata_path),
                    source_newer=source_newer,
                    fingerprint_checked=True,
                    fingerprint_matches=True,
                )
            return CacheAssessment(
                state="stale",
                message=(
                    "The source content fingerprint no longer matches the Parquet cache. "
                    "Regenerate it before loading."
                ),
                parquet_path=str(parquet),
                metadata_path=str(metadata_path),
                source_newer=source_newer,
                fingerprint_checked=True,
                fingerprint_matches=False,
            )

    if source_newer:
        return CacheAssessment(
            state="stale",
            message=(
                "The source is newer than the Parquet cache. The metadata does not "
                "contain a usable fingerprint, so regeneration is recommended."
            ),
            parquet_path=str(parquet),
            metadata_path=str(metadata_path),
            source_newer=True,
            legacy_metadata=True,
        )

    return CacheAssessment(
        state="unknown",
        message=(
            "The cache exists, but its metadata cannot fully verify source content. "
            "You may reuse it or regenerate it explicitly."
        ),
        parquet_path=str(parquet),
        metadata_path=str(metadata_path),
        source_newer=source_newer,
        legacy_metadata=True,
    )


def estimate_parquet_size(
    source_format: str,
    source_size_bytes: Optional[int],
    *,
    existing_parquet_path: Optional[str | Path] = None,
) -> Optional[int]:
    if existing_parquet_path:
        try:
            existing_size = Path(existing_parquet_path).expanduser().stat().st_size
            if existing_size > 0:
                return int(existing_size * 1.15)
        except OSError:
            pass

    if source_size_bytes is None:
        return None

    factor = {
        "fits": 0.5,
        "csv": 1.10,
        "tsv": 1.10,
        "jsonl": 1.10,
        "excel": 3.00,
        "hdf5": 2.00,
        "feather": 1.25,
        "arrow": 1.25,
        "ecsv": 1.10,
        "sqlite": 1.50,
    }.get(str(source_format), 1.50)
    return max(1, int(int(source_size_bytes) * factor))


def assess_disk_space(
    parquet_path: str | Path,
    estimated_output_bytes: Optional[int],
    *,
    reserve_bytes: int = _DISK_RESERVE_BYTES,
) -> DiskSpaceAssessment:
    path = Path(parquet_path).expanduser().resolve()
    parent = _nearest_existing_parent(path.parent)
    try:
        free = int(shutil.disk_usage(parent).free)
    except OSError:
        free = None

    if estimated_output_bytes is None or free is None:
        return DiskSpaceAssessment(
            path=str(path),
            free_bytes=free,
            estimated_output_bytes=estimated_output_bytes,
            required_bytes=None,
            enough_space=None,
            message=(
                "Disk space could not be estimated reliably. AstronomicAL will still "
                "check that the destination is writable before conversion."
            ),
        )

    required = int(estimated_output_bytes) + max(0, int(reserve_bytes))
    enough = free >= required
    if enough:
        message = (
            f"Approximately {_format_bytes(free)} free; estimated new Parquet "
            f"size {_format_bytes(estimated_output_bytes)}."
        )
    else:
        message = (
            f"Only {_format_bytes(free)} is free, while AstronomicAL estimates "
            f"about {_format_bytes(required)} is needed including working reserve."
        )

    return DiskSpaceAssessment(
        path=str(path),
        free_bytes=free,
        estimated_output_bytes=estimated_output_bytes,
        required_bytes=required,
        enough_space=enough,
        message=message,
    )


def convert_source_to_parquet(
    source_path: str | Path,
    *,
    parquet_path: str | Path,
    source_format: Optional[str] = None,
    dataset_id: Optional[str] = None,
    source_subresource: Optional[str] = None,
    overwrite: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    progress_state_callback: Optional[ProgressStateCallback] = None,
    cancel_token: Any = None,
    compression: str = "zstd",
    compression_level: int = 1,
) -> ConversionResult:
    source = Path(source_path).expanduser().resolve()
    parquet = Path(parquet_path).expanduser().resolve()
    fmt = source_format or detect_source_format(source)
    if fmt is None or not is_convertible_format(fmt):
        raise ValueError(f"Unsupported source format for Parquet conversion: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"Dataset source does not exist: {source}")

    dataset_id = str(dataset_id or source_stem(source))
    metadata_path = metadata_path_for_parquet(parquet)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    _assert_destination_writable(parquet)

    if parquet.exists() and not overwrite:
        metadata = _read_json(metadata_path)
        emit_progress(f"Reusing existing Parquet cache: {parquet}", progress_callback)
        return ConversionResult(
            dataset_id=dataset_id,
            source_path=str(source),
            source_format=fmt,
            parquet_path=str(parquet),
            metadata_path=str(metadata_path),
            created=False,
            metadata=metadata,
        )

    _raise_if_cancelled(cancel_token)
    source_fingerprint = fingerprint_source(source)
    emit_progress(
        f"Source fingerprint: {source_fingerprint.digest_mode} "
        f"{source_fingerprint.digest[:16]}…",
        progress_callback,
    )

    if fmt == "fits":
        raw = import_fits_table_to_parquet(
            source,
            parquet_path=parquet,
            metadata_path=metadata_path,
            hdu=int(source_subresource) if source_subresource not in (None, "") else 1,
            dataset_id=dataset_id,
            overwrite=overwrite,
            compression=compression,
            compression_level=compression_level,
            use_dictionary=False,
            strip_strings=False,
            progress_callback=progress_callback,
            progress_state_callback=progress_state_callback,
            source_fingerprint=source_fingerprint.to_dict(),
            cancel_token=cancel_token,
        )
        return ConversionResult(
            dataset_id=dataset_id,
            source_path=str(source),
            source_format=fmt,
            parquet_path=str(raw["parquet_path"]),
            metadata_path=str(raw["metadata_path"]),
            created=bool(raw.get("created")),
            metadata=dict(raw.get("metadata") or {}),
        )

    tmp = parquet.with_name(parquet.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    emit_progress(f"Preparing {source_format_label(fmt)} conversion: {source}", progress_callback)
    emit_progress(f"Parquet destination: {parquet}", progress_callback)
    started = time.perf_counter()
    emit_progress_state(
        progress_state_callback,
        phase="preparing",
        label=f"Preparing {source_format_label(fmt)} conversion",
        rows_completed=0,
        rows_total=None,
        chunk_index=0,
        chunk_count=None,
        elapsed_seconds=0.0,
        eta_seconds=None,
        rows_per_second=None,
        average_rows_per_second=None,
        tmp_size_bytes=0,
        rss_gib=None,
    )

    tables = _source_tables(
        source,
        fmt,
        source_subresource=source_subresource,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
    )
    row_count, columns = _write_tables_to_parquet(
        tables,
        tmp,
        progress_callback=progress_callback,
        progress_state_callback=progress_state_callback,
        conversion_started=started,
        cancel_token=cancel_token,
        compression=compression,
        compression_level=compression_level,
    )

    _raise_if_cancelled(cancel_token)
    os.replace(tmp, parquet)
    elapsed = time.perf_counter() - started
    emit_progress(
        f"Parquet conversion complete: {row_count:,} rows × {len(columns):,} columns "
        f"in {elapsed:.1f}s.",
        progress_callback,
    )
    emit_progress_state(
        progress_state_callback,
        phase="complete",
        label="Parquet conversion complete",
        rows_completed=row_count,
        rows_total=row_count,
        chunk_index=None,
        chunk_count=None,
        elapsed_seconds=elapsed,
        eta_seconds=0.0,
        rows_per_second=None,
        average_rows_per_second=(row_count / elapsed if elapsed else None),
        tmp_size_bytes=(parquet.stat().st_size if parquet.exists() else None),
        rss_gib=None,
    )

    metadata = {
        "cache_schema_version": 2,
        "dataset_id": dataset_id,
        "source_format": fmt,
        "source_path": str(source),
        "source_subresource": source_subresource or None,
        "parquet_path": str(parquet),
        "row_count": int(row_count),
        "column_count": len(columns),
        "columns": list(columns),
        "source_fingerprint": source_fingerprint.to_dict(),
        "parquet_compression": compression,
        "parquet_compression_level": compression_level,
        "conversion_elapsed_seconds": elapsed,
        "created_at_ns": time.time_ns(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    emit_progress(f"Metadata written: {metadata_path}", progress_callback)

    return ConversionResult(
        dataset_id=dataset_id,
        source_path=str(source),
        source_format=fmt,
        parquet_path=str(parquet),
        metadata_path=str(metadata_path),
        created=True,
        metadata=metadata,
    )


def _source_tables(
    source: Path,
    source_format: str,
    *,
    source_subresource: Optional[str],
    progress_callback: Optional[ProgressCallback],
    cancel_token: Any,
) -> Iterable[pa.Table | pa.RecordBatch]:
    if source_format in {"csv", "tsv"}:
        return _csv_batches(
            source,
            delimiter="," if source_format == "csv" else "\t",
            progress_callback=progress_callback,
            cancel_token=cancel_token,
        )
    if source_format == "jsonl":
        return _jsonl_batches(source, progress_callback=progress_callback, cancel_token=cancel_token)
    if source_format == "excel":
        return _excel_tables(source, sheet_name=source_subresource, progress_callback=progress_callback)
    if source_format == "hdf5":
        return _hdf_tables(
            source,
            key=source_subresource,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
        )
    if source_format == "feather":
        return _feather_tables(source, progress_callback=progress_callback)
    if source_format == "arrow":
        return _arrow_batches(source, progress_callback=progress_callback, cancel_token=cancel_token)
    if source_format == "ecsv":
        return _ecsv_tables(source, progress_callback=progress_callback)
    if source_format == "sqlite":
        return _sqlite_tables(
            source,
            table_name=source_subresource,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
        )
    raise ValueError(f"No converter is implemented for source format {source_format!r}.")


def _csv_batches(
    source: Path,
    *,
    delimiter: str,
    progress_callback: Optional[ProgressCallback],
    cancel_token: Any,
) -> Iterator[pa.RecordBatch]:
    # ``pyarrow.input_stream`` defaults to compression="detect". When the
    # source is a path ending in ``.gz`` Arrow therefore already presents a
    # decompressed stream. Wrapping that stream in CompressedInputStream again
    # attempts to inflate plain CSV/TSV bytes a second time and fails with
    # ``zlib inflate failed: incorrect header check``.
    stream = pa.input_stream(str(source), compression="detect")
    try:
        reader = pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(block_size=64 * 1024**2, use_threads=True),
            parse_options=pacsv.ParseOptions(delimiter=delimiter),
            convert_options=pacsv.ConvertOptions(strings_can_be_null=True),
        )
        for index, batch in enumerate(reader, start=1):
            _raise_if_cancelled(cancel_token)
            emit_progress(
                f"Input batch {index}: {batch.num_rows:,} rows × {batch.num_columns:,} columns.",
                progress_callback,
            )
            yield batch
    finally:
        stream.close()


def _jsonl_batches(
    source: Path,
    *,
    progress_callback: Optional[ProgressCallback],
    cancel_token: Any,
) -> Iterator[pa.Table]:
    try:
        iterator = pd.read_json(
            source,
            lines=True,
            chunksize=100_000,
            compression="infer",
        )
    except ValueError as exc:
        raise ValueError(f"Could not parse JSON Lines source {source}: {exc}") from exc

    for index, frame in enumerate(iterator, start=1):
        _raise_if_cancelled(cancel_token)
        emit_progress(
            f"Input batch {index}: {len(frame):,} rows × {len(frame.columns):,} columns.",
            progress_callback,
        )
        yield pa.Table.from_pandas(frame, preserve_index=False)


def _excel_tables(
    source: Path,
    *,
    sheet_name: Optional[str],
    progress_callback: Optional[ProgressCallback],
) -> Iterable[pa.Table]:
    selected: Any = sheet_name if sheet_name not in (None, "") else 0
    emit_progress(
        f"Reading spreadsheet sheet {selected!r}. Spreadsheet parsing is not streaming, "
        "so memory use can temporarily approach the uncompressed sheet size.",
        progress_callback,
    )
    try:
        frame = pd.read_excel(source, sheet_name=selected)
    except ImportError as exc:
        raise RuntimeError(
            "Spreadsheet support needs the pandas engine for this file type "
            "(for example openpyxl for .xlsx, xlrd for .xls, or odfpy for .ods)."
        ) from exc
    return (pa.Table.from_pandas(frame, preserve_index=False),)


def _hdf_tables(
    source: Path,
    *,
    key: Optional[str],
    progress_callback: Optional[ProgressCallback],
    cancel_token: Any,
) -> Iterable[pa.Table]:
    store: Any = None
    pandas_keys: list[str] = []
    try:
        store = pd.HDFStore(source, mode="r")
        pandas_keys = [str(value) for value in store.keys()]
    except BaseException:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        store = None

    selected = key
    if store is not None and pandas_keys:
        if selected is None:
            selected = pandas_keys[0]
        if selected not in pandas_keys and not str(selected).startswith("/"):
            selected = f"/{selected}"
        if selected in pandas_keys:
            storer = store.get_storer(selected)
            is_table = bool(getattr(storer, "is_table", False))

            def _pandas_iterator() -> Iterator[pa.Table]:
                try:
                    if is_table:
                        emit_progress(f"Streaming HDF5 table {selected} in chunks.", progress_callback)
                        for index, frame in enumerate(
                            store.select(selected, chunksize=100_000), start=1
                        ):
                            _raise_if_cancelled(cancel_token)
                            emit_progress(
                                f"Input batch {index}: {len(frame):,} rows.",
                                progress_callback,
                            )
                            yield pa.Table.from_pandas(frame, preserve_index=False)
                    else:
                        emit_progress(
                            f"HDF5 key {selected} uses pandas fixed storage; it must be "
                            "materialised before AstronomicAL can write Parquet.",
                            progress_callback,
                        )
                        frame = store.get(selected)
                        if not isinstance(frame, pd.DataFrame):
                            frame = pd.DataFrame(frame)
                        yield pa.Table.from_pandas(frame, preserve_index=False)
                finally:
                    store.close()

            return _pandas_iterator()

    # A generic HDF5 file may be openable by PyTables even though it contains
    # no pandas HDFStore keys. In that case we must close the PyTables handle
    # before falling through to the h5py reader, otherwise PyTables retains an
    # open file until interpreter shutdown and emits UnclosedFileWarning.
    if store is not None:
        store.close()
        store = None

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "This HDF5 file is not a pandas HDFStore. Install h5py to read generic "
            "HDF5 datasets/groups, or install PyTables ('tables') for pandas HDFStore files."
        ) from exc

    available = _h5py_subresources(source)
    if not available:
        raise ValueError(
            "No tabular HDF5 dataset/group was found. Supported generic layouts are "
            "1-D structured datasets, 2-D datasets, or groups of equal-length 1-D datasets."
        )
    selected = str(selected or available[0])
    if not selected.startswith("/"):
        selected = "/" + selected
    if selected not in available:
        raise KeyError(f"Unknown HDF5 table {selected!r}. Available tables: {list(available)}")

    def _h5py_iterator() -> Iterator[pa.Table]:
        with h5py.File(source, "r") as handle:
            obj = handle[selected]
            if isinstance(obj, h5py.Dataset):
                total = int(obj.shape[0]) if obj.ndim else 0
                chunk_rows = max(1, min(100_000, total or 1))
                for index, start_row in enumerate(range(0, total, chunk_rows), start=1):
                    _raise_if_cancelled(cancel_token)
                    stop_row = min(total, start_row + chunk_rows)
                    values = obj[start_row:stop_row]
                    if obj.dtype.names:
                        mapping = {name: values[name] for name in obj.dtype.names}
                    elif obj.ndim == 2:
                        mapping = {
                            f"column_{column_index}": values[:, column_index]
                            for column_index in range(values.shape[1])
                        }
                    else:
                        mapping = {Path(selected).name or "value": values}
                    table = pa.Table.from_pydict(mapping)
                    emit_progress(
                        f"Input batch {index}: {table.num_rows:,} rows from HDF5 {selected}.",
                        progress_callback,
                    )
                    yield table
                return

            datasets = {
                name: child
                for name, child in obj.items()
                if isinstance(child, h5py.Dataset)
                and child.ndim == 1
                and child.dtype.names is None
            }
            if not datasets:
                raise ValueError(f"HDF5 group {selected!r} has no usable 1-D columns.")
            lengths = {int(child.shape[0]) for child in datasets.values()}
            if len(lengths) != 1:
                raise ValueError(
                    f"HDF5 group {selected!r} contains columns with different lengths."
                )
            total = next(iter(lengths))
            chunk_rows = max(1, min(100_000, total or 1))
            for index, start_row in enumerate(range(0, total, chunk_rows), start=1):
                _raise_if_cancelled(cancel_token)
                stop_row = min(total, start_row + chunk_rows)
                mapping = {name: child[start_row:stop_row] for name, child in datasets.items()}
                table = pa.Table.from_pydict(mapping)
                emit_progress(
                    f"Input batch {index}: {table.num_rows:,} rows from HDF5 group {selected}.",
                    progress_callback,
                )
                yield table

    return _h5py_iterator()


def _feather_tables(
    source: Path,
    *,
    progress_callback: Optional[ProgressCallback],
) -> Iterable[pa.Table]:
    emit_progress("Reading Feather/Arrow table.", progress_callback)
    return (pafeather.read_table(source),)


def _arrow_batches(
    source: Path,
    *,
    progress_callback: Optional[ProgressCallback],
    cancel_token: Any,
) -> Iterable[pa.RecordBatch]:
    mapped = pa.memory_map(str(source), "r")
    try:
        reader = paipc.open_file(mapped)
        batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
    except pa.ArrowInvalid:
        mapped.seek(0)
        stream_reader = paipc.open_stream(mapped)
        batches = list(stream_reader)

    def _iterator() -> Iterator[pa.RecordBatch]:
        try:
            for index, batch in enumerate(batches, start=1):
                _raise_if_cancelled(cancel_token)
                emit_progress(f"Input batch {index}: {batch.num_rows:,} rows.", progress_callback)
                yield batch
        finally:
            mapped.close()

    return _iterator()


def _ecsv_tables(
    source: Path,
    *,
    progress_callback: Optional[ProgressCallback],
) -> Iterable[pa.Table]:
    from astropy.table import Table

    emit_progress("Reading Astropy ECSV table.", progress_callback)
    table = Table.read(source, format="ascii.ecsv")
    return (pa.Table.from_pandas(table.to_pandas(), preserve_index=False),)


def _sqlite_tables(
    source: Path,
    *,
    table_name: Optional[str],
    progress_callback: Optional[ProgressCallback],
    cancel_token: Any,
) -> Iterable[pa.Table]:
    con = sqlite3.connect(str(source))
    rows = con.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    names = [str(row[0]) for row in rows]
    if not names:
        con.close()
        raise ValueError(f"SQLite database contains no user tables or views: {source}")
    selected = table_name or names[0]
    if selected not in names:
        con.close()
        raise KeyError(f"Unknown SQLite table {selected!r}. Available tables: {names}")

    quoted = '"' + selected.replace('"', '""') + '"'

    def _iterator() -> Iterator[pa.Table]:
        try:
            emit_progress(f"Streaming SQLite table {selected!r} in chunks.", progress_callback)
            for index, frame in enumerate(
                pd.read_sql_query(f"SELECT * FROM {quoted}", con, chunksize=100_000),
                start=1,
            ):
                _raise_if_cancelled(cancel_token)
                emit_progress(f"Input batch {index}: {len(frame):,} rows.", progress_callback)
                yield pa.Table.from_pandas(frame, preserve_index=False)
        finally:
            con.close()

    return _iterator()


def _write_tables_to_parquet(
    tables: Iterable[pa.Table | pa.RecordBatch],
    tmp_path: Path,
    *,
    progress_callback: Optional[ProgressCallback],
    progress_state_callback: Optional[ProgressStateCallback],
    conversion_started: float,
    cancel_token: Any,
    compression: str,
    compression_level: int,
) -> tuple[int, tuple[str, ...]]:
    writer: Optional[pq.ParquetWriter] = None
    schema: Optional[pa.Schema] = None
    total_rows = 0
    columns: tuple[str, ...] = ()

    try:
        for batch_index, value in enumerate(tables, start=1):
            _raise_if_cancelled(cancel_token)
            table = value if isinstance(value, pa.Table) else pa.Table.from_batches([value])
            if table.num_rows == 0 and writer is not None:
                continue

            if writer is None:
                schema = table.schema
                columns = tuple(str(name) for name in schema.names)
                writer = pq.ParquetWriter(
                    tmp_path,
                    schema,
                    compression=compression,
                    compression_level=compression_level,
                    use_dictionary=False,
                    write_statistics=True,
                )
            elif schema is not None and not table.schema.equals(schema, check_metadata=False):
                try:
                    table = table.cast(schema)
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                    raise ValueError(
                        "A later input chunk has a schema incompatible with the first chunk. "
                        "Normalise the source column types before importing."
                    ) from exc

            batch_started = time.perf_counter()
            writer.write_table(table, row_group_size=max(1, table.num_rows))
            batch_elapsed = max(0.0, time.perf_counter() - batch_started)
            total_rows += int(table.num_rows)
            elapsed_total = max(0.0, time.perf_counter() - conversion_started)
            average_rows_per_second = total_rows / elapsed_total if elapsed_total else None
            tmp_size = tmp_path.stat().st_size if tmp_path.exists() else 0
            emit_progress(
                f"Parquet row group {batch_index}: wrote {table.num_rows:,} rows; "
                f"total={total_rows:,}.",
                progress_callback,
            )
            emit_progress_state(
                progress_state_callback,
                phase="writing",
                label=f"Writing Parquet row group {batch_index}",
                rows_completed=total_rows,
                rows_total=None,
                chunk_index=batch_index,
                chunk_count=None,
                elapsed_seconds=elapsed_total,
                eta_seconds=None,
                rows_per_second=(table.num_rows / batch_elapsed if batch_elapsed else None),
                average_rows_per_second=average_rows_per_second,
                tmp_size_bytes=tmp_size,
                rss_gib=None,
            )

        if writer is None or schema is None:
            raise ValueError("The selected source produced no tabular rows or columns.")
    finally:
        if writer is not None:
            writer.close()

    return total_rows, columns


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _assert_destination_writable(path: Path) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"Parquet destination is not writable: {parent}")


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _canonical_path(value: str | Path) -> str:
    try:
        return str(Path(value).expanduser().resolve())
    except Exception:
        return str(Path(value).expanduser().absolute())


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _format_bytes(value: int) -> str:
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{value:,} B"


def _raise_if_cancelled(cancel_token: Any) -> None:
    if cancel_token is None:
        return
    value = getattr(cancel_token, "cancelled", False)
    if callable(value):
        value = value()
    if value:
        raise RuntimeError("Dataset import was cancelled.")


__all__ = [
    "CacheAssessment",
    "ConversionResult",
    "DiskSpaceAssessment",
    "ProgressCallback",
    "SourceFingerprint",
    "assess_disk_space",
    "assess_parquet_cache",
    "convert_source_to_parquet",
    "default_parquet_path",
    "detect_source_format",
    "emit_progress",
    "emit_progress_state",
    "estimate_parquet_size",
    "fingerprint_source",
    "inspect_source_subresources",
    "is_convertible_format",
    "metadata_path_for_parquet",
    "source_format_label",
    "source_stem",
]

