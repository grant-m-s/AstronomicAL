from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

AXIS_KIND_NUMERIC = "numeric"
AXIS_KIND_DATETIME = "datetime"
AXIS_KIND_CATEGORICAL = "categorical"
AXIS_KIND_UNSUPPORTED = "unsupported"

_NUMERIC_MARKERS = (
    "int",
    "integer",
    "bigint",
    "smallint",
    "tinyint",
    "hugeint",
    "uint",
    "float",
    "double",
    "real",
    "decimal",
    "numeric",
    "bool",
    "boolean",
)
_DATETIME_MARKERS = (
    "date",
    "time",
    "timestamp",
    "datetime",
    "duration",
    "interval",
)
_UNSUPPORTED_MARKERS = (
    "array",
    "list",
    "large_list",
    "fixed_size_list",
    "struct",
    "map",
    "union",
    "binary",
    "large_binary",
    "blob",
    "geometry",
)


@dataclass(frozen=True)
class AxisEncoding:
    values: np.ndarray
    kind: str
    categories: tuple[str, ...] = ()
    numeric_ratio: float = 0.0
    datetime_ratio: float = 0.0
    category_count: int = 0


def infer_axis_kind_from_dtype(dtype_name: Any) -> str:
    """Return the broad plotting kind represented by a schema dtype.

    Scalar strings are categorical rather than unsupported. This is the main
    difference from the former numeric-only X/Y discovery logic.
    """

    text = str(dtype_name or "").strip().lower()

    if any(marker in text for marker in _UNSUPPORTED_MARKERS):
        return AXIS_KIND_UNSUPPORTED
    if any(marker in text for marker in _DATETIME_MARKERS):
        return AXIS_KIND_DATETIME
    if any(marker in text for marker in _NUMERIC_MARKERS):
        return AXIS_KIND_NUMERIC

    # Arrow/Pandas strings, categories and generic objects are all renderable
    # through factor encoding. Runtime encoding still rejects wholly missing
    # values without breaking the panel.
    return AXIS_KIND_CATEGORICAL


def plottable_columns_from_schema(
    columns: Sequence[Any],
    dtypes: Mapping[str, Any],
) -> list[str]:
    return [
        str(column)
        for column in columns
        if infer_axis_kind_from_dtype(dtypes.get(str(column), ""))
        != AXIS_KIND_UNSUPPORTED
    ]


def _safe_numeric_array(series: pd.Series) -> np.ndarray:
    try:
        values = series.to_numpy(copy=False)
    except Exception:
        values = series.to_numpy(dtype=np.float64, copy=False, na_value=np.nan)

    values = np.asarray(values)
    if values.dtype == np.float32:
        return values

    try:
        values64 = values.astype(np.float64, copy=False)
    except (TypeError, ValueError):
        values64 = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)

    finite = np.isfinite(values64)
    if not finite.any():
        return values64

    max_abs = np.nanmax(np.abs(values64[finite]))
    if max_abs <= 1.0e20:
        return values64.astype(np.float32, copy=False)
    return values64


def _non_null_ratio(series: pd.Series, converted: pd.Series) -> float:
    source_non_null = int(series.notna().sum())
    if source_non_null <= 0:
        return 0.0
    return float(converted.notna().sum()) / float(source_non_null)


def encode_axis_values(
    values: Any,
    *,
    numeric_threshold: float = 0.95,
    datetime_threshold: float = 0.95,
    max_category_labels: int = 80,
) -> AxisEncoding:
    """Convert a scalar column into coordinates accepted by existing plots.

    Conversion order:
      1. Native bool/numeric values remain numeric.
      2. Native datetimes become Unix milliseconds.
      3. Numeric-like strings become continuous numbers when at least 95% of
         non-null values convert.
      4. Datetime-like strings become Unix milliseconds at the same threshold.
      5. Remaining scalar values are factor encoded in first-seen order.

    First-seen factor order is intentional. A bounded prefix can then provide
    correct tick labels for the categories it contains, while later unseen
    categories append without shifting earlier codes.
    """

    series = values if isinstance(values, pd.Series) else pd.Series(values)

    if pd.api.types.is_bool_dtype(series.dtype):
        return AxisEncoding(
            values=series.to_numpy(dtype=np.float32, copy=False, na_value=np.nan),
            kind=AXIS_KIND_NUMERIC,
        )

    if pd.api.types.is_numeric_dtype(series.dtype):
        return AxisEncoding(
            values=_safe_numeric_array(series),
            kind=AXIS_KIND_NUMERIC,
            numeric_ratio=1.0,
        )

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        raw = parsed.astype("int64", copy=False).to_numpy(dtype=np.float64)
        raw[parsed.isna().to_numpy()] = np.nan
        return AxisEncoding(
            values=raw / 1.0e6,
            kind=AXIS_KIND_DATETIME,
            datetime_ratio=1.0,
        )

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = _non_null_ratio(series, numeric)
    if numeric_ratio >= float(numeric_threshold):
        return AxisEncoding(
            values=_safe_numeric_array(numeric),
            kind=AXIS_KIND_NUMERIC,
            numeric_ratio=numeric_ratio,
        )

    # Do not attempt datetime parsing on arbitrary long free text. The length
    # guard also avoids expensive parsing of paths and URIs.
    as_string = series.astype("string")
    non_null_strings = as_string.dropna()
    median_length = (
        float(non_null_strings.str.len().median())
        if not non_null_strings.empty
        else 0.0
    )
    datetime_ratio = 0.0
    if median_length <= 64 and not non_null_strings.empty:
        date_shaped = non_null_strings.str.match(
            r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T].*)?$",
            na=False,
        )
        datetime_shape_ratio = float(date_shaped.mean())
        if datetime_shape_ratio >= float(datetime_threshold):
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            datetime_ratio = _non_null_ratio(series, parsed)
            if datetime_ratio >= float(datetime_threshold):
                raw = parsed.astype("int64", copy=False).to_numpy(dtype=np.float64)
                raw[parsed.isna().to_numpy()] = np.nan
                return AxisEncoding(
                    values=raw / 1.0e6,
                    kind=AXIS_KIND_DATETIME,
                    numeric_ratio=numeric_ratio,
                    datetime_ratio=datetime_ratio,
                )

    codes, uniques = pd.factorize(as_string, sort=False, use_na_sentinel=True)
    encoded = codes.astype(np.float32, copy=False)
    encoded[codes < 0] = np.nan

    category_count = int(len(uniques))
    categories = (
        tuple(str(value) for value in uniques.tolist())
        if category_count <= int(max_category_labels)
        else ()
    )

    return AxisEncoding(
        values=encoded,
        kind=AXIS_KIND_CATEGORICAL,
        categories=categories,
        numeric_ratio=numeric_ratio,
        datetime_ratio=datetime_ratio,
        category_count=category_count,
    )


def encode_axis_array(values: Any) -> np.ndarray:
    return encode_axis_values(values).values


def categorical_tick_overrides(
    values: Iterable[Any],
    *,
    max_labels: int = 40,
    max_label_length: int = 48,
) -> Dict[float, str]:
    """Return category-code labels when the axis is reasonably small."""

    encoding = encode_axis_values(values)
    if encoding.kind != AXIS_KIND_CATEGORICAL:
        return {}
    if not encoding.categories or len(encoding.categories) > int(max_labels):
        return {}

    out: Dict[float, str] = {}
    for index, label in enumerate(encoding.categories):
        if len(label) > int(max_label_length):
            label = label[: max(1, int(max_label_length) - 1)] + "…"
        out[float(index)] = label
    return out
