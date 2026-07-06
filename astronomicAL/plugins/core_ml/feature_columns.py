from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping, Optional, Sequence

_FEATURE_PARAM_KEYS = (
    "feature_columns",
    "input_columns",
    "features",
    "x_columns",
)

def parse_column_list(value: Any) -> List[str]:
    """Normalise a UI/API column-list value into a clean list of column names.

    Accepts:
    - list/tuple/set
    - JSON list string
    - comma/newline separated string
    - single scalar value
    """
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                value = parsed
            elif isinstance(parsed, str):
                value = [parsed]
            else:
                value = [text]
        except Exception:
            value = [
                part.strip()
                for chunk in text.splitlines()
                for part in chunk.split(",")
                if part.strip()
            ]

    elif isinstance(value, Mapping):
        value = value.keys()

    elif isinstance(value, (int, float)):
        value = [value]

    columns: List[str] = []
    for item in value or []:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text == "Use Index":
            continue
        if text not in columns:
            columns.append(text)

    return columns

def feature_columns_from_params(params: Mapping[str, Any]) -> List[str]:
    """Resolve feature columns from any accepted recipe/action parameter key."""
    params = dict(params or {})

    for key in _FEATURE_PARAM_KEYS:
        columns = parse_column_list(params.get(key))
        if columns:
            return columns

    return []

def list_dataset_columns(context: Any, dataset_id: Any) -> List[str]:
    dataset_id = str(dataset_id or "").strip()
    if not dataset_id:
        return []

    try:
        return [str(col) for col in context.datasets.list_columns(dataset_id)]
    except Exception:
        pass

    try:
        return [str(col) for col in context.datasets.get_df(dataset_id).columns]
    except Exception:
        return []

def default_excluded_columns(
    *,
    record_id_column: Optional[str] = None,
    target_column: Optional[str] = None,
    image_column: Optional[str] = None,
    extra: Optional[Iterable[str]] = None,
) -> set[str]:
    excluded = {
        "",
        "Use Index",
        "al_label",
        "label",
        "labels",
        "target",
        "target_label",
        "class",
        "class_label",
        "prediction",
        "predicted_label",
        "prediction_confidence",
        "confidence",
        "entropy",
        "least_confidence",
        "margin",
        "margin_uncertainty",
        "selection_rank",
        "rank",
        "active_learning_strategy",
        "source",
    }

    for value in (record_id_column, target_column, image_column):
        if value:
            excluded.add(str(value))

    for value in extra or []:
        if value:
            excluded.add(str(value))

    return excluded

def default_feature_columns(
    context: Any,
    dataset_id: Any,
    *,
    record_id_column: Optional[str] = None,
    target_column: Optional[str] = None,
    image_column: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Best-effort automatic feature list.

    This is intentionally conservative. It excludes obvious identity, target,
    image, prediction, and AL metadata columns. It does not force numeric-only
    columns because sklearn recipes can preprocess categorical columns. Torch
    tabular recipes should normally use explicit feature_columns.
    """
    columns = list_dataset_columns(context, dataset_id)
    params = dict(params or {})

    excluded = default_excluded_columns(
        record_id_column=record_id_column,
        target_column=target_column,
        image_column=image_column,
        extra=parse_column_list(params.get("exclude_feature_columns")),
    )

    return [
        str(column)
        for column in columns
        if str(column) not in excluded
    ]

