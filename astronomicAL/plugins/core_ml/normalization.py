from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .image_sidecar import load_image
from .serialization import json_safe


AUTO_KEYS = {
    "calculate_mean_std",
    "calculate_normalization",
    "compute_mean_std",
    "compute_normalization",
    "auto_mean_std",
    "auto_normalization",
    "use_computed_normalization",
}

MEAN_KEYS = {
    "mean",
    "image_mean",
    "normalize_mean",
    "normalization_mean",
    "normalization_means",
}

STD_KEYS = {
    "std",
    "image_std",
    "normalize_std",
    "normalization_std",
    "normalization_stds",
}


def should_compute_train_split_normalization(params: Mapping[str, Any]) -> bool:
    params = dict(params or {})

    for key in AUTO_KEYS:
        if _as_bool(params.get(key), default=False):
            return True

    return False


def apply_train_split_image_normalization(
    *,
    context: Any,
    params: Dict[str, Any],
    source_dataset_id: str,
    train_dataset_id: Optional[str],
    train_row_ids: Sequence[Any],
    record_id_column: str,
    image_column: Optional[str],
    cancel_token: Any = None,
) -> Optional[Dict[str, Any]]:
    """Calculate image mean/std from the training split only.

    This intentionally runs after the managed harness has created the split.
    Calculating over the full source dataset leaks validation/test distribution
    information into preprocessing.
    """

    if not should_compute_train_split_normalization(params):
        return None

    image_column = str(
        image_column
        or params.get("image_column")
        or params.get("image_path_column")
        or ""
    ).strip()

    if not image_column:
        raise ValueError(
            "Cannot calculate image mean/std because no image column is selected."
        )

    frame = _load_train_frame(
        context=context,
        source_dataset_id=source_dataset_id,
        train_dataset_id=train_dataset_id,
        train_row_ids=train_row_ids,
        record_id_column=record_id_column,
        image_column=image_column,
    )

    sample_size = int(params.get("normalization_sample_size") or 0)
    if sample_size > 0 and len(frame) > sample_size:
        frame = frame.sample(sample_size, random_state=int(params.get("protocol_random_state", 42)))

    image_size = int(
        params.get("image_size")
        or params.get("input_size")
        or params.get("resize")
        or 224
    )

    mean, std, used = _compute_mean_std(
        frame[image_column].dropna().tolist(),
        image_size=image_size,
        cancel_token=cancel_token,
    )

    for key in MEAN_KEYS:
        if key in params:
            params[key] = mean
    for key in STD_KEYS:
        if key in params:
            params[key] = std

    # Always set canonical keys too, so new recipes have stable names.
    params["normalization_mean"] = mean
    params["normalization_std"] = std
    params["normalization_source"] = "train_split"
    params["normalization_sample_count"] = used

    info = {
        "source": "train_split",
        "image_column": image_column,
        "train_dataset_id": train_dataset_id,
        "source_dataset_id": source_dataset_id,
        "sample_count": used,
        "mean": mean,
        "std": std,
    }

    params["computed_normalization"] = json_safe(info)
    return json_safe(info)


def _load_train_frame(
    *,
    context: Any,
    source_dataset_id: str,
    train_dataset_id: Optional[str],
    train_row_ids: Sequence[Any],
    record_id_column: str,
    image_column: str,
) -> pd.DataFrame:
    columns = [image_column]

    if record_id_column and record_id_column != image_column:
        columns.insert(0, record_id_column)

    if train_dataset_id:
        try:
            return context.datasets.get_df(train_dataset_id, columns=columns)
        except TypeError:
            df = context.datasets.get_df(train_dataset_id)
            return df[[c for c in columns if c in df.columns]]

    try:
        return context.datasets.get_rows_by_ids(
            source_dataset_id,
            list(train_row_ids),
            id_column=record_id_column,
            columns=columns,
        )
    except Exception:
        df = context.datasets.get_df(source_dataset_id, columns=columns)
        if record_id_column not in df.columns:
            return df

        wanted = {str(row_id) for row_id in train_row_ids}
        return df[df[record_id_column].astype(str).isin(wanted)]


def _compute_mean_std(
    image_values: Iterable[Any],
    *,
    image_size: int,
    cancel_token: Any = None,
) -> Tuple[list[float], list[float], int]:
    sums = np.zeros(3, dtype=np.float64)
    sq_sums = np.zeros(3, dtype=np.float64)
    count = 0

    for value in image_values:
        _raise_if_cancelled(cancel_token)

        image = load_image(value).resize((int(image_size), int(image_size)))
        arr = np.asarray(image, dtype=np.float32) / 255.0

        if arr.ndim != 3 or arr.shape[2] < 3:
            continue

        arr = arr[:, :, :3]
        pixels = arr.reshape(-1, 3)

        sums += pixels.sum(axis=0)
        sq_sums += np.square(pixels).sum(axis=0)
        count += pixels.shape[0]

    if count <= 0:
        raise ValueError("Could not calculate mean/std because no valid images were loaded.")

    mean = sums / count
    variance = np.maximum((sq_sums / count) - np.square(mean), 0.0)
    std = np.sqrt(variance)

    return (
        [float(v) for v in mean.tolist()],
        [float(v) for v in std.tolist()],
        int(count),
    )


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False

    return default


def _raise_if_cancelled(cancel_token: Any) -> None:
    if cancel_token is None:
        return

    for name in ("raise_if_cancelled", "throw_if_cancelled", "check_cancelled"):
        method = getattr(cancel_token, name, None)
        if callable(method):
            method()
            return

    for name in ("cancelled", "is_cancelled"):
        value = getattr(cancel_token, name, None)
        if callable(value) and value():
            raise RuntimeError("Job was cancelled.")
        if isinstance(value, bool) and value:
            raise RuntimeError("Job was cancelled.")