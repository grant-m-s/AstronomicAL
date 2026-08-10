from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits

from astronomicAL.utils.optimise import rss_gib

ProgressCallback = Callable[[str], None]
ProgressStateCallback = Callable[[dict[str, Any]], None]

DEFAULT_FITS_TARGET_CHUNK_BYTES = 512 * 1024**2


def _log_loader(
    message: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> None:
    text = str(message)
    print(f"[AstronomicAL loader] {text}", flush=True)
    if progress_callback is None:
        return
    try:
        progress_callback(text)
    except Exception:
        # UI progress is best-effort and must not fail an import.
        pass


def _emit_progress_state(
    progress_state_callback: Optional[ProgressStateCallback],
    **state: Any,
) -> None:
    if progress_state_callback is None:
        return
    try:
        progress_state_callback(dict(state))
    except Exception:
        # Structured UI progress is best-effort and must never fail conversion.
        pass


def _raise_if_cancelled(cancel_token: Any) -> None:
    if cancel_token is None:
        return
    value = getattr(cancel_token, "cancelled", False)
    if callable(value):
        value = value()
    if value:
        raise RuntimeError("Dataset import was cancelled.")


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _load_existing_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _native_endian(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.byteorder in ("=", "|"):
        return arr
    return arr.byteswap().view(arr.dtype.newbyteorder("="))


def _arrow_array_from_fits_column(
    arr: np.ndarray,
    *,
    null_sentinel: Any = None,
    strip_strings: bool = True,
) -> pa.Array:
    arr = np.asarray(arr)
    if arr.ndim != 1 or arr.dtype.fields is not None:
        raise ValueError("Only scalar 1-D columns are supported")

    if arr.dtype.kind == "S":
        if strip_strings:
            arr = np.char.rstrip(arr)
        try:
            return pa.array(arr).cast(pa.string())
        except Exception:
            return pa.array(
                [
                    x.decode("utf-8", "replace").rstrip()
                    if isinstance(x, (bytes, bytearray, np.bytes_))
                    else x
                    for x in arr
                ],
                type=pa.string(),
            )

    if arr.dtype.kind == "U":
        if strip_strings:
            arr = np.char.rstrip(arr)
        return pa.array(arr, type=pa.string())

    mask = None
    if null_sentinel is not None and arr.dtype.kind in "iu":
        mask = arr == null_sentinel

    arr = _native_endian(arr)
    return pa.array(arr, mask=mask)


def _fits_rec_chunk_to_arrow_table(
    rec: np.ndarray,
    column_names: list[str],
    null_sentinels: dict[str, object],
    *,
    strip_strings: bool = True,
    log_every_columns: int = 25,
    progress_prefix: str = "",
    progress_callback: Optional[ProgressCallback] = None,
    cancel_token: Any = None,
) -> tuple[pa.Table, list[str]]:
    arrays: list[pa.Array] = []
    names: list[str] = []
    skipped: list[str] = []
    started = time.perf_counter()

    for index, name in enumerate(column_names, start=1):
        _raise_if_cancelled(cancel_token)
        arr = np.asarray(rec[name])
        if arr.ndim != 1 or arr.dtype.fields is not None:
            skipped.append(name)
            continue

        arrays.append(
            _arrow_array_from_fits_column(
                arr,
                null_sentinel=null_sentinels.get(name),
                strip_strings=strip_strings,
            )
        )
        names.append(name)

        if index == 1 or index % log_every_columns == 0 or index == len(column_names):
            _log_loader(
                f"{progress_prefix} converted {index:,}/{len(column_names):,} columns "
                f"in {time.perf_counter() - started:.1f}s; RSS≈{rss_gib():.2f} GiB",
                progress_callback,
            )

    if not arrays:
        raise ValueError("No scalar 1-D columns found to write to Parquet")
    return pa.Table.from_arrays(arrays, names=names), skipped


def _fits_hdu_column_metadata(hdu: Any) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    for col in hdu.columns:
        columns.append(
            {
                "name": col.name,
                "format": str(col.format),
                "unit": str(col.unit) if col.unit is not None else None,
                "dim": str(col.dim) if getattr(col, "dim", None) else None,
                "null": _json_safe(getattr(col, "null", None)),
                "bscale": _json_safe(getattr(col, "bscale", None)),
                "bzero": _json_safe(getattr(col, "bzero", None)),
            }
        )
    return columns


def import_fits_table_to_parquet(
    fits_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    parquet_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    hdu: int | str = 1,
    dataset_id: Optional[str] = None,
    overwrite: bool = False,
    target_chunk_bytes: int = DEFAULT_FITS_TARGET_CHUNK_BYTES,
    max_rows_per_chunk: Optional[int] = None,
    compression: str = "zstd",
    compression_level: int = 1,
    use_dictionary: bool | list[str] = True,
    strip_strings: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    progress_state_callback: Optional[ProgressStateCallback] = None,
    source_fingerprint: Optional[dict[str, Any]] = None,
    cancel_token: Any = None,
) -> dict[str, Any]:
    """Stream a FITS binary table to Parquet without full-table materialisation.

    ``parquet_path`` is the preferred modern API because the dataset loader lets
    users choose the generated Parquet location. ``cache_dir`` remains supported
    for existing callers.
    """
    source = Path(fits_path).expanduser().resolve()
    if dataset_id is None:
        dataset_id = source.stem

    if parquet_path is None:
        if cache_dir is None:
            raise ValueError("Either parquet_path or cache_dir is required.")
        cache = Path(cache_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        parquet = cache / f"{dataset_id}.parquet"
    else:
        parquet = Path(parquet_path).expanduser().resolve()
        parquet.parent.mkdir(parents=True, exist_ok=True)

    metadata_file = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else parquet.with_suffix(".metadata.json")
    )
    tmp_parquet = parquet.with_name(parquet.name + ".tmp")

    if parquet.exists() and not overwrite:
        metadata = _load_existing_metadata(metadata_file)
        metadata.setdefault("dataset_id", dataset_id)
        metadata.setdefault("source_format", "fits")
        metadata.setdefault("source_path", str(source))
        metadata.setdefault("source_hdu", hdu)
        metadata.setdefault("parquet_path", str(parquet))
        _log_loader(f"Reusing existing FITS Parquet cache: {parquet}", progress_callback)
        return {
            "dataset_id": dataset_id,
            "parquet_path": str(parquet),
            "metadata_path": str(metadata_file),
            "created": False,
            "metadata": metadata,
        }

    if tmp_parquet.exists():
        tmp_parquet.unlink()

    _raise_if_cancelled(cancel_token)
    _log_loader(f"Preparing streamed FITS import: {source}", progress_callback)
    _log_loader(f"Dataset id: {dataset_id}", progress_callback)
    _log_loader(f"Parquet cache path: {parquet}", progress_callback)

    conversion_started = time.perf_counter()
    writer: Optional[pq.ParquetWriter] = None
    schema: Optional[pa.Schema] = None
    n_rows: Optional[int] = None
    chunk_count: Optional[int] = None
    last_chunk_no = 0
    total_rows = 0
    skipped_columns: set[str] = set()
    column_metadata: list[dict[str, Any]] = []
    header_metadata: dict[str, Any] = {}

    try:
        with fits.open(source, memmap=True, lazy_load_hdus=True) as hdul:
            table_hdu = hdul[hdu]
            data = table_hdu.data
            if data is None:
                raise ValueError(f"HDU {hdu!r} does not contain table data")

            n_rows = int(table_hdu.header["NAXIS2"])
            row_bytes = int(table_hdu.header.get("NAXIS1", data.dtype.itemsize))
            column_names = list(data.names)
            null_sentinels = {
                col.name: getattr(col, "null", None)
                for col in table_hdu.columns
                if getattr(col, "null", None) is not None
            }

            rows_per_chunk = max(1, target_chunk_bytes // max(row_bytes, 1))
            if max_rows_per_chunk is not None:
                rows_per_chunk = min(rows_per_chunk, max_rows_per_chunk)
            rows_per_chunk = min(rows_per_chunk, max(n_rows, 1))

            _log_loader(
                f"FITS table: {n_rows:,} rows × {len(column_names):,} columns; "
                f"row≈{row_bytes:,} bytes; chunk≈{rows_per_chunk:,} rows.",
                progress_callback,
            )
            chunk_count = max(1, (n_rows + rows_per_chunk - 1) // rows_per_chunk)
            _emit_progress_state(
                progress_state_callback,
                phase="preparing",
                label="Preparing FITS chunks",
                rows_completed=0,
                rows_total=n_rows,
                chunk_index=0,
                chunk_count=chunk_count,
                elapsed_seconds=time.perf_counter() - conversion_started,
                eta_seconds=None,
                rows_per_second=None,
                average_rows_per_second=None,
                tmp_size_bytes=0,
                rss_gib=rss_gib(),
            )

            column_metadata = _fits_hdu_column_metadata(table_hdu)
            header_metadata = _json_safe(dict(table_hdu.header))

            for chunk_no, start in enumerate(range(0, n_rows, rows_per_chunk), start=1):
                _raise_if_cancelled(cancel_token)
                stop = min(start + rows_per_chunk, n_rows)
                chunk_started = time.perf_counter()

                _log_loader(
                    f"Chunk {chunk_no}: starting rows {start:,}–{stop:,} "
                    f"({stop - start:,} rows); RSS≈{rss_gib():.2f} GiB",
                    progress_callback,
                )

                copy_started = time.perf_counter()
                rec = np.array(data[start:stop], copy=True)
                _log_loader(
                    f"Chunk {chunk_no}: copied FITS rows in "
                    f"{time.perf_counter() - copy_started:.1f}s; RSS≈{rss_gib():.2f} GiB",
                    progress_callback,
                )

                convert_started = time.perf_counter()
                arrow_table, skipped = _fits_rec_chunk_to_arrow_table(
                    rec,
                    column_names,
                    null_sentinels,
                    strip_strings=strip_strings,
                    progress_prefix=f"Chunk {chunk_no}:",
                    progress_callback=progress_callback,
                    cancel_token=cancel_token,
                )
                skipped_columns.update(skipped)
                _log_loader(
                    f"Chunk {chunk_no}: built Arrow table "
                    f"{arrow_table.num_rows:,} rows × {arrow_table.num_columns:,} cols "
                    f"in {time.perf_counter() - convert_started:.1f}s; "
                    f"table≈{arrow_table.nbytes / 1024**3:.2f} GiB; "
                    f"RSS≈{rss_gib():.2f} GiB",
                    progress_callback,
                )

                if writer is None:
                    schema = arrow_table.schema
                    writer = pq.ParquetWriter(
                        tmp_parquet,
                        schema,
                        compression=compression,
                        compression_level=compression_level,
                        use_dictionary=use_dictionary,
                        write_statistics=True,
                    )
                elif schema is not None and not arrow_table.schema.equals(
                    schema, check_metadata=False
                ):
                    arrow_table = arrow_table.cast(schema)

                writer.write_table(arrow_table, row_group_size=arrow_table.num_rows)
                total_rows += int(arrow_table.num_rows)
                elapsed = time.perf_counter() - chunk_started
                rows_per_second = arrow_table.num_rows / elapsed if elapsed else 0.0
                tmp_size = tmp_parquet.stat().st_size if tmp_parquet.exists() else 0
                elapsed_total = time.perf_counter() - conversion_started
                average_rows_per_second = total_rows / elapsed_total if elapsed_total else 0.0
                remaining_rows = max(0, n_rows - total_rows)
                eta_seconds = (
                    remaining_rows / average_rows_per_second
                    if average_rows_per_second > 0 and remaining_rows > 0
                    else 0.0 if remaining_rows == 0 else None
                )
                current_rss_gib = rss_gib()

                _log_loader(
                    f"Chunk {chunk_no}: wrote rows {start:,}–{stop:,} in {elapsed:.1f}s "
                    f"({rows_per_second:,.0f} rows/s); total={total_rows:,}/{n_rows:,}; "
                    f"tmp_size≈{tmp_size / 1024**3:.2f} GiB; RSS≈{current_rss_gib:.2f} GiB",
                    progress_callback,
                )
                last_chunk_no = chunk_no
                _emit_progress_state(
                    progress_state_callback,
                    phase="writing",
                    label=f"Writing FITS chunk {chunk_no}/{chunk_count}",
                    rows_completed=total_rows,
                    rows_total=n_rows,
                    chunk_index=chunk_no,
                    chunk_count=chunk_count,
                    elapsed_seconds=elapsed_total,
                    eta_seconds=eta_seconds,
                    rows_per_second=rows_per_second,
                    average_rows_per_second=average_rows_per_second,
                    tmp_size_bytes=tmp_size,
                    rss_gib=current_rss_gib,
                )

                del rec, arrow_table
                if chunk_no % 4 == 0:
                    gc.collect()

    finally:
        if writer is not None:
            tmp_size = tmp_parquet.stat().st_size if tmp_parquet.exists() else 0
            elapsed_total = time.perf_counter() - conversion_started
            _emit_progress_state(
                progress_state_callback,
                phase="finalizing",
                label="Finalising Parquet writer",
                rows_completed=total_rows,
                rows_total=n_rows,
                chunk_index=last_chunk_no,
                chunk_count=chunk_count,
                elapsed_seconds=elapsed_total,
                eta_seconds=None,
                rows_per_second=None,
                average_rows_per_second=(total_rows / elapsed_total if elapsed_total else None),
                tmp_size_bytes=tmp_size,
                rss_gib=rss_gib(),
            )
            _log_loader(
                f"Closing Parquet writer... tmp_size≈{tmp_size / 1024**3:.2f} GiB; "
                f"RSS≈{rss_gib():.2f} GiB",
                progress_callback,
            )
            close_started = time.perf_counter()
            writer.close()
            writer = None
            tmp_size = tmp_parquet.stat().st_size if tmp_parquet.exists() else 0
            _log_loader(
                f"Closed Parquet writer in {time.perf_counter() - close_started:.1f}s; "
                f"tmp_size≈{tmp_size / 1024**3:.2f} GiB; RSS≈{rss_gib():.2f} GiB",
                progress_callback,
            )
        _log_loader("Releasing FITS mmap handle...", progress_callback)

    _raise_if_cancelled(cancel_token)
    _log_loader("Replacing tmp Parquet with final Parquet path...", progress_callback)
    replace_started = time.perf_counter()
    os.replace(tmp_parquet, parquet)
    _log_loader(
        f"Rename complete in {time.perf_counter() - replace_started:.1f}s: {parquet}",
        progress_callback,
    )

    columns = list(schema.names) if schema is not None else []
    metadata = {
        "cache_schema_version": 2,
        "dataset_id": dataset_id,
        "source_format": "fits",
        "source_path": str(source),
        "source_subresource": str(hdu),
        "source_hdu": hdu,
        "parquet_path": str(parquet),
        "row_count": total_rows,
        "column_count": len(columns),
        "columns": columns,
        "column_metadata": column_metadata,
        "skipped_columns": sorted(skipped_columns),
        "table_meta": header_metadata,
        "parquet_compression": compression,
        "parquet_compression_level": compression_level,
        "parquet_use_dictionary": use_dictionary,
        "target_chunk_bytes": target_chunk_bytes,
        "conversion_elapsed_seconds": time.perf_counter() - conversion_started,
        "created_at_ns": time.time_ns(),
    }

    if source_fingerprint:
        metadata["source_fingerprint"] = dict(source_fingerprint)

    _log_loader("Writing JSON metadata...", progress_callback)
    metadata_started = time.perf_counter()
    metadata_file.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    _log_loader(
        f"Metadata written in {time.perf_counter() - metadata_started:.1f}s: {metadata_file}",
        progress_callback,
    )
    _log_loader(f"Parquet cache written: {parquet}", progress_callback)
    completed_elapsed = time.perf_counter() - conversion_started
    final_size = parquet.stat().st_size if parquet.exists() else 0
    _emit_progress_state(
        progress_state_callback,
        phase="complete",
        label="Parquet conversion complete",
        rows_completed=total_rows,
        rows_total=n_rows if n_rows is not None else total_rows,
        chunk_index=last_chunk_no,
        chunk_count=chunk_count,
        elapsed_seconds=completed_elapsed,
        eta_seconds=0.0,
        rows_per_second=None,
        average_rows_per_second=(total_rows / completed_elapsed if completed_elapsed else None),
        tmp_size_bytes=final_size,
        rss_gib=rss_gib(),
    )

    return {
        "dataset_id": dataset_id,
        "parquet_path": str(parquet),
        "metadata_path": str(metadata_file),
        "created": True,
        "metadata": metadata,
    }


def register_fits_table(
    datasets: Any,
    fits_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    parquet_path: str | Path | None = None,
    hdu: int | str = 1,
    dataset_id: Optional[str] = None,
    name: Optional[str] = None,
    overwrite: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    progress_state_callback: Optional[ProgressStateCallback] = None,
    source_fingerprint: Optional[dict[str, Any]] = None,
    cancel_token: Any = None,
) -> dict[str, Any]:
    """Backward-compatible FITS registration helper.

    New loader code separates conversion from registration, but existing callers
    may continue using this helper.
    """
    _log_loader(f"Starting FITS registration -id: {dataset_id}", progress_callback)
    result = import_fits_table_to_parquet(
        fits_path,
        cache_dir=cache_dir,
        parquet_path=parquet_path,
        overwrite=overwrite,
        hdu=hdu,
        dataset_id=dataset_id,
        target_chunk_bytes=DEFAULT_FITS_TARGET_CHUNK_BYTES,
        compression="zstd",
        compression_level=1,
        use_dictionary=False,
        strip_strings=False,
        progress_callback=progress_callback,
        progress_state_callback=progress_state_callback,
        source_fingerprint=source_fingerprint,
        cancel_token=cancel_token,
    )

    ds_id = result["dataset_id"]
    metadata = dict(result.get("metadata") or {})
    metadata.setdefault("source_format", "fits")
    metadata.setdefault("source_path", str(fits_path))
    metadata.setdefault("source_hdu", hdu)
    metadata.setdefault("cache_path", result["parquet_path"])

    registration_meta = dict(metadata)
    registration_meta.pop("dataset_id", None)
    registration_meta.pop("name", None)

    _log_loader(f"Registering lazy Parquet-backed dataset: {ds_id}", progress_callback)
    _log_loader(f"Backend cache: {result['parquet_path']}", progress_callback)
    datasets.register_parquet(
        ds_id,
        result["parquet_path"],
        name=name or ds_id,
        **registration_meta,
    )
    _log_loader(f"Dataset registered successfully: {ds_id}", progress_callback)
    return result


__all__ = [
    "ProgressCallback",
    "ProgressStateCallback",
    "import_fits_table_to_parquet",
    "register_fits_table",
]

