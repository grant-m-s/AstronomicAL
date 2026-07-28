from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .image_sidecar import load_image
from .normalization import MEAN_KEYS, STD_KEYS, should_compute_train_split_normalization
from .partition_reader import PartitionReader
from .serialization import json_safe


def apply_partition_reader_image_normalization(
    *,
    reader: PartitionReader,
    params: Dict[str, Any],
    image_column: Optional[str],
    cancel_token: Any = None,
) -> Optional[Dict[str, Any]]:
    """Calculate train-only image statistics with bounded metadata memory."""

    if not should_compute_train_split_normalization(params):
        return None
    resolved_image_column = str(
        image_column
        or params.get("image_column")
        or params.get("image_path_column")
        or ""
    ).strip()
    if not resolved_image_column:
        raise ValueError(
            "Cannot calculate image mean/std because no image column is selected."
        )

    sample_size = int(params.get("normalization_sample_size") or 0)
    if sample_size < 0:
        raise ValueError("normalization_sample_size must be zero or greater.")
    image_size = int(
        params.get("image_size")
        or params.get("input_size")
        or params.get("resize")
        or 224
    )
    image_values = _iter_image_values(
        reader,
        image_column=resolved_image_column,
        cancel_token=cancel_token,
    )
    if sample_size > 0:
        image_values = iter(
            _reservoir_sample(
                image_values,
                sample_size=sample_size,
                seed=int(params.get("protocol_random_state", 42)),
            )
        )

    mean, std, image_count, pixel_count = _compute_mean_std_stream(
        image_values,
        image_size=image_size,
        cancel_token=cancel_token,
    )
    for key in MEAN_KEYS:
        if key in params:
            params[key] = mean
    for key in STD_KEYS:
        if key in params:
            params[key] = std
    params["normalization_mean"] = mean
    params["normalization_std"] = std
    params["normalization_source"] = "train_split"
    params["normalization_sample_count"] = image_count
    params["normalization_image_count"] = image_count
    params["normalization_pixel_count"] = pixel_count

    info = {
        "source": "train_split",
        "image_column": resolved_image_column,
        "train_dataset_id": reader.dataset_id,
        "source_dataset_id": reader.partition.manifest.source_dataset_id,
        "sample_count": image_count,
        "image_count": image_count,
        "pixel_count": pixel_count,
        "mean": mean,
        "std": std,
        "partition_fingerprint": reader.partition.fingerprint,
        "manifest_sha256": reader.partition.manifest.sha256,
    }
    params["computed_normalization"] = json_safe(info)
    return json_safe(info)


def _iter_image_values(
    reader: PartitionReader,
    *,
    image_column: str,
    cancel_token: Any,
) -> Iterator[Any]:
    for batch in reader.iter_batches(
        columns=[image_column],
        strict=True,
        cancel_check=lambda: _raise_if_cancelled(cancel_token),
    ):
        for value in batch.frame[image_column].dropna().tolist():
            yield value


def _reservoir_sample(
    values: Iterable[Any],
    *,
    sample_size: int,
    seed: int,
) -> list[Any]:
    rng = random.Random(int(seed))
    sample: list[Any] = []
    for index, value in enumerate(values):
        if index < sample_size:
            sample.append(value)
            continue
        replacement = rng.randint(0, index)
        if replacement < sample_size:
            sample[replacement] = value
    return sample


def _compute_mean_std_stream(
    image_values: Iterable[Any],
    *,
    image_size: int,
    cancel_token: Any = None,
) -> Tuple[list[float], list[float], int, int]:
    sums = np.zeros(3, dtype=np.float64)
    sq_sums = np.zeros(3, dtype=np.float64)
    image_count = 0
    pixel_count = 0

    for value in image_values:
        _raise_if_cancelled(cancel_token)
        image = load_image(value).resize((int(image_size), int(image_size)))
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim != 3 or array.shape[2] < 3:
            continue
        pixels = array[:, :, :3].reshape(-1, 3)
        sums += pixels.sum(axis=0)
        sq_sums += np.square(pixels).sum(axis=0)
        image_count += 1
        pixel_count += int(pixels.shape[0])

    if image_count <= 0 or pixel_count <= 0:
        raise ValueError(
            "Could not calculate mean/std because no valid images were loaded."
        )
    mean = sums / pixel_count
    variance = np.maximum((sq_sums / pixel_count) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    return (
        [float(value) for value in mean.tolist()],
        [float(value) for value in std.tolist()],
        int(image_count),
        int(pixel_count),
    )


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
