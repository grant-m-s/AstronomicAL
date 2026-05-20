from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def normalise_dataset_id(value: str) -> str:
    value = (value or "dataset").strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "dataset"


def default_cache_dir_for_context(context, *, fallback_name: str = ".astronomical_cache") -> Path:
    """
    Return a stable local cache directory for generated Parquet datasets.

    This intentionally avoids requiring users to know where Parquet/DuckDB are
    being used. Later, this can be replaced with a project/workspace cache path.
    """
    config = getattr(context, "config", None)

    try:
        settings = getattr(config, "settings", {}) or {}
        dataset_path = settings.get("dataset_filepath")
        if dataset_path:
            return Path(dataset_path).expanduser().resolve().parent / fallback_name
    except Exception:
        pass

    return Path.cwd() / fallback_name


def register_dataframe_as_parquet(
    datasets,
    *,
    dataset_id: str,
    df: pd.DataFrame,
    name: Optional[str] = None,
    cache_dir: str | Path,
    overwrite: bool = True,
    **meta: Any,
) -> Path:
    """
    Materialise a pandas DataFrame to Parquet, then register it as a lazy
    DuckDB-backed Parquet dataset.

    This is the transition bridge for plugins that still produce pandas results.
    """
    print(
        f"[AstronomicAL loader] Writing derived dataset to Parquet: {dataset_id}",
        flush=True,
    )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_id = normalise_dataset_id(dataset_id)
    parquet_path = cache_dir / f"{safe_id}.parquet"
    metadata_path = cache_dir / f"{safe_id}.metadata.json"

    if parquet_path.exists() and not overwrite:
        raise FileExistsError(f"Parquet cache already exists: {parquet_path}")

    print(
        f"[AstronomicAL loader] Derived dataset size: {len(df):,} rows × {len(df.columns):,} columns.",
        flush=True,
    )
    print(f"[AstronomicAL loader] Parquet path: {parquet_path}", flush=True)

    df.to_parquet(
        parquet_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    metadata = {
        "dataset_id": dataset_id,
        "name": name or dataset_id,
        "backend": "duckdb_parquet",
        "source_format": "parquet",
        "source_path": str(parquet_path),
        "rows": int(len(df)),
        "columns": [str(col) for col in df.columns],
        **meta,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    print(f"[AstronomicAL loader] Metadata path: {metadata_path}", flush=True)

    register_parquet = getattr(datasets, "register_parquet", None)
    if callable(register_parquet):
        # Critical fix:
        # register_parquet receives dataset_id positionally, and name as a
        # keyword argument, so remove duplicate keys from metadata.
        registration_meta = dict(metadata)
        registration_meta.pop("dataset_id", None)
        registration_meta.pop("name", None)

        datasets.register_parquet(
            dataset_id,
            parquet_path,
            name=name or dataset_id,
            **registration_meta,
        )
    else:
        datasets.register(
            dataset_id,
            df,
            name=name or dataset_id,
            **metadata,
        )

    print(
        f"[AstronomicAL loader] Registered derived Parquet-backed dataset: {dataset_id}",
        flush=True,
    )

    return parquet_path


def replace_dataset_with_dataframe_parquet(
    context,
    *,
    dataset_id: str,
    df: pd.DataFrame,
    name: Optional[str] = None,
    cache_dir: str | Path,
    **meta: Any,
) -> Path:
    """
    Replace/update an existing dataset with a Parquet-backed source.

    Used by derived-column workflows that update the active dataset.
    """
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise RuntimeError("DatasetManager is required.")

    parquet_path = register_dataframe_as_parquet(
        datasets,
        dataset_id=dataset_id,
        df=df,
        name=name or dataset_id,
        cache_dir=cache_dir,
        overwrite=True,
        **meta,
    )

    return parquet_path