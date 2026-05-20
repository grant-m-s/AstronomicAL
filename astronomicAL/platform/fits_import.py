from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table

import gc
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from astropy.io import fits

import time

from astronomicAL.utils.optimise import (
    rss_gib,
)

def _log_loader(message: str) -> None:
    print(f"[AstronomicAL loader] {message}", flush=True)


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


def _table_column_metadata(table: Table) -> dict[str, dict[str, Any]]:
    columns: dict[str, dict[str, Any]] = {}

    for name in table.colnames:
        col = table[name]
        columns[str(name)] = {
            "dtype": str(getattr(col, "dtype", "")),
            "unit": str(getattr(col, "unit", "")) if getattr(col, "unit", None) else None,
            "description": getattr(col, "description", None),
            "format": str(getattr(col, "format", "")) if getattr(col, "format", None) else None,
        }

    return columns


def _load_existing_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

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

    # FITS string columns commonly arrive as bytes.
    if arr.dtype.kind == "S":
        if strip_strings:
            arr = np.char.rstrip(arr)

        # pa.array(bytes) gives binary; cast decodes as UTF-8.
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
) -> tuple[pa.Table, list[str]]:
    arrays = []
    names = []
    skipped = []

    t0 = time.perf_counter()

    for i, name in enumerate(column_names, start=1):
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

        if i == 1 or i % log_every_columns == 0 or i == len(column_names):
            elapsed = time.perf_counter() - t0
            _log_loader(
                f"{progress_prefix} converted {i:,}/{len(column_names):,} columns "
                f"in {elapsed:.1f}s; RSS≈{rss_gib():.2f} GiB"
            )

    if not arrays:
        raise ValueError("No scalar 1-D columns found to write to Parquet")

    return pa.Table.from_arrays(arrays, names=names), skipped


def _fits_hdu_column_metadata(hdu) -> list[dict[str, Any]]:
    cols = []

    for col in hdu.columns:
        cols.append(
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

    return cols

def import_fits_table_to_parquet(
    fits_path: str | Path,
    *,
    cache_dir: str | Path,
    hdu: int | str = 1,
    dataset_id: Optional[str] = None,
    overwrite: bool = False,
    target_chunk_bytes: int = 256 * 1024**2,
    max_rows_per_chunk: Optional[int] = None,
    compression: str = "zstd",
    compression_level: int = 1,
    use_dictionary: bool | list[str] = True,
    strip_strings: bool = True,
) -> dict[str, Any]:
    """
    Import a FITS binary table to Parquet without materialising the full table.

    This preserves numeric precision: no float32 downcast, no optimisation pass.
    It writes row chunks as Parquet row groups.
    """
    fits_path = Path(fits_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if dataset_id is None:
        dataset_id = fits_path.stem

    parquet_path = cache_dir / f"{dataset_id}.parquet"
    metadata_path = cache_dir / f"{dataset_id}.metadata.json"
    tmp_parquet_path = parquet_path.with_suffix(".parquet.tmp")

    if parquet_path.exists() and not overwrite:
        metadata = _load_existing_metadata(metadata_path)
        metadata.setdefault("dataset_id", dataset_id)
        metadata.setdefault("source_format", "fits")
        metadata.setdefault("source_path", str(fits_path))
        metadata.setdefault("source_hdu", hdu)
        metadata.setdefault("parquet_path", str(parquet_path))

        return {
            "dataset_id": dataset_id,
            "parquet_path": str(parquet_path),
            "metadata_path": str(metadata_path),
            "created": False,
            "metadata": metadata,
        }

    if tmp_parquet_path.exists():
        tmp_parquet_path.unlink()

    _log_loader(f"Preparing streamed FITS import: {fits_path}")
    _log_loader(f"Dataset id: {dataset_id}")
    _log_loader(f"Parquet cache path: {parquet_path}")

    writer = None
    schema = None
    total_rows = 0
    skipped_columns: set[str] = set()

    with fits.open(fits_path, memmap=True, lazy_load_hdus=True) as hdul:
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
            f"row≈{row_bytes:,} bytes; chunk≈{rows_per_chunk:,} rows."
        )

        column_metadata = _fits_hdu_column_metadata(table_hdu)
        header_metadata = _json_safe(dict(table_hdu.header))

        try:
            chunk_no = 0

            for start in range(0, n_rows, rows_per_chunk):
                chunk_no += 1
                stop = min(start + rows_per_chunk, n_rows)

                chunk_t0 = time.perf_counter()

                _log_loader(
                    f"Chunk {chunk_no}: starting rows {start:,}–{stop:,} "
                    f"({stop - start:,} rows); RSS≈{rss_gib():.2f} GiB"
                )

                copy_t0 = time.perf_counter()
                rec = np.array(data[start:stop], copy=True)
                _log_loader(
                    f"Chunk {chunk_no}: copied FITS rows in "
                    f"{time.perf_counter() - copy_t0:.1f}s; RSS≈{rss_gib():.2f} GiB"
                )

                convert_t0 = time.perf_counter()
                arrow_table, skipped = _fits_rec_chunk_to_arrow_table(
                    rec,
                    column_names,
                    null_sentinels,
                    strip_strings=strip_strings,
                    progress_prefix=f"Chunk {chunk_no}:",
                )
                skipped_columns.update(skipped)

                _log_loader(
                    f"Chunk {chunk_no}: built Arrow table "
                    f"{arrow_table.num_rows:,} rows × {arrow_table.num_columns:,} cols "
                    f"in {time.perf_counter() - convert_t0:.1f}s; "
                    f"table≈{arrow_table.nbytes / 1024**3:.2f} GiB; "
                    f"RSS≈{rss_gib():.2f} GiB"
                )

                if writer is None:
                    schema = arrow_table.schema
                    writer = pq.ParquetWriter(
                        tmp_parquet_path,
                        schema,
                        compression=compression,
                        compression_level=compression_level,
                        use_dictionary=use_dictionary,
                        write_statistics=True,
                    )
                elif not arrow_table.schema.equals(schema, check_metadata=False):
                    arrow_table = arrow_table.cast(schema)

                write_t0 = time.perf_counter()
                writer.write_table(
                    arrow_table,
                    row_group_size=arrow_table.num_rows,
                )

                total_rows += arrow_table.num_rows

                elapsed = time.perf_counter() - chunk_t0
                rows_per_sec = arrow_table.num_rows / elapsed if elapsed else 0

                _log_loader(
                    f"Chunk {chunk_no}: wrote rows {start:,}–{stop:,} "
                    f"in {elapsed:.1f}s "
                    f"({rows_per_sec:,.0f} rows/s); "
                    f"total={total_rows:,}/{n_rows:,}; "
                    f"tmp_size≈{tmp_parquet_path.stat().st_size / 1024**3:.2f} GiB; "
                    f"RSS≈{rss_gib():.2f} GiB"
                )

                del rec, arrow_table

                if chunk_no % 4 == 0:
                    gc.collect()

        finally:
            if writer is not None:
                _log_loader(
                    f"Closing Parquet writer... "
                    f"tmp_size≈{tmp_parquet_path.stat().st_size / 1024**3:.2f} GiB; "
                    f"RSS≈{rss_gib():.2f} GiB"
                )

                close_t0 = time.perf_counter()
                writer.close()

                _log_loader(
                    f"Closed Parquet writer in {time.perf_counter() - close_t0:.1f}s; "
                    f"tmp_size≈{tmp_parquet_path.stat().st_size / 1024**3:.2f} GiB; "
                    f"RSS≈{rss_gib():.2f} GiB"
                )

            _log_loader("Releasing FITS mmap handle...")
            # del data

    _log_loader("Replacing tmp Parquet with final Parquet path...")
    replace_t0 = time.perf_counter()
    os.replace(tmp_parquet_path, parquet_path)
    _log_loader(
        f"Rename complete in {time.perf_counter() - replace_t0:.1f}s: {parquet_path}"
    )

    column_names = list(schema.names) if schema is not None else []

    metadata = {
        "dataset_id": dataset_id,
        "source_format": "fits",
        "source_path": str(fits_path),
        "source_hdu": hdu,
        "parquet_path": str(parquet_path),
        "row_count": total_rows,
        "column_count": len(column_names),

        # Keep legacy/downstream expectation:
        "columns": column_names,

        # Put detailed FITS info here instead:
        "column_metadata": column_metadata,

        "skipped_columns": sorted(skipped_columns),
        "table_meta": header_metadata,
        "parquet_compression": compression,
        "parquet_compression_level": compression_level,
        "parquet_use_dictionary": use_dictionary,
        "target_chunk_bytes": target_chunk_bytes,
    }

    _log_loader("Writing JSON metadata...")
    metadata_t0 = time.perf_counter()
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _log_loader(
        f"Metadata written in {time.perf_counter() - metadata_t0:.1f}s: {metadata_path}"
    )

    _log_loader(f"Parquet cache written: {parquet_path}")
    _log_loader(f"Metadata written: {metadata_path}")

    return {
        "dataset_id": dataset_id,
        "parquet_path": str(parquet_path),
        "metadata_path": str(metadata_path),
        "created": True,
        "metadata": metadata,
    }


def register_fits_table(
    datasets,
    fits_path: str | Path,
    *,
    cache_dir: str | Path,
    hdu: int | str = 1,
    dataset_id: Optional[str] = None,
    name: Optional[str] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Import a FITS table and register it with DatasetManager as a lazy Parquet
    source.
    """
    _log_loader(f"Starting FITS registration -id: {dataset_id}")

    result = import_fits_table_to_parquet(
        fits_path,
        cache_dir=cache_dir,
        overwrite=overwrite,
        dataset_id=dataset_id,
        target_chunk_bytes=4096 * 2 * 1024**2,
        compression="zstd",
        compression_level=1,
        use_dictionary=False,
        strip_strings=False,
    )

    ds_id = result["dataset_id"]

    metadata = dict(result.get("metadata") or {})
    metadata.setdefault("source_format", "fits")
    metadata.setdefault("source_path", str(fits_path))
    metadata.setdefault("source_hdu", hdu)
    metadata.setdefault("cache_path", result["parquet_path"])

    # Critical fix:
    # register_parquet receives dataset_id positionally, so remove any duplicate
    # keys that would also be passed through **metadata.
    registration_meta = dict(metadata)
    registration_meta.pop("dataset_id", None)
    registration_meta.pop("name", None)

    _log_loader(f"Registering lazy Parquet-backed dataset: {ds_id}")
    _log_loader(f"Backend cache: {result['parquet_path']}")

    datasets.register_parquet(
        ds_id,
        result["parquet_path"],
        name=name or ds_id,
        **registration_meta,
    )

    _log_loader(f"Dataset registered successfully: {ds_id}")

    return result