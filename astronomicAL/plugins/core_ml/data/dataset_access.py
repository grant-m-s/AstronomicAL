from __future__ import annotations

import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence

def active_dataset_id(context: Any) -> Optional[str]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None

    for method_name in ("active_id", "get_active_id", "active_dataset_id"):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                value = None
            if value:
                return str(value)

    value = getattr(datasets, "active", None)
    if isinstance(value, str):
        return value

    return None

def list_dataset_ids(context: Any) -> List[str]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return []

    candidates: List[str] = []

    for method_name in ("list_ids", "ids", "dataset_ids"):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                values = method()
                candidates.extend(str(v) for v in values if v)
            except Exception:
                pass

    for method_name in ("list", "list_datasets"):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                values = method()
            except Exception:
                values = []
            for value in values or []:
                if isinstance(value, str):
                    candidates.append(value)
                else:
                    dataset_id = (
                        getattr(value, "dataset_id", None)
                        or getattr(value, "id", None)
                        or getattr(value, "name", None)
                    )
                    if dataset_id:
                        candidates.append(str(dataset_id))

    active = active_dataset_id(context)
    if active:
        candidates.append(active)

    deduped: List[str] = []
    seen = set()
    for value in candidates:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped

def _accepts_keyword(
    method: Any,
    keyword: str,
) -> bool:
    """Return whether a callable explicitly accepts a keyword."""
    try:
        parameters = inspect.signature(
            method
        ).parameters.values()
    except (TypeError, ValueError):
        return False

    return any(
        parameter.name == keyword
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _call_frame_method(
    method: Any,
    dataset_id: str,
    columns: Optional[Sequence[str]],
):
    requested_columns = (
        list(columns)
        if columns is not None
        else None
    )

    if (
        requested_columns is not None
        and _accepts_keyword(method, "columns")
    ):
        return method(
            dataset_id,
            columns=requested_columns,
        )

    df = method(dataset_id)

    if requested_columns is not None:
        return df.loc[:, requested_columns]

    return df


def _call_bound_frame_method(
    method: Any,
    columns: Optional[Sequence[str]],
):
    requested_columns = (
        list(columns)
        if columns is not None
        else None
    )

    if (
        requested_columns is not None
        and _accepts_keyword(method, "columns")
    ):
        return method(columns=requested_columns)

    df = method()

    if requested_columns is not None:
        return df.loc[:, requested_columns]

    return df


def get_dataset_frame(
    context: Any,
    dataset_id: str,
    *,
    columns: Optional[Sequence[str]] = None,
):
    """Materialise a dataset as a pandas DataFrame.

    Prefer the platform's canonical ``get_df`` API. Compatibility
    fallbacks are used only when that API is absent; real backend
    errors are not swallowed.
    """
    datasets = getattr(context, "datasets", None)

    if datasets is None:
        raise ValueError(
            "No dataset manager is available on context."
        )

    dataset_id = str(dataset_id)

    # The current platform API. Let genuine data/backend errors
    # propagate instead of silently trying a different route.
    get_df = getattr(datasets, "get_df", None)
    if callable(get_df):
        return _call_frame_method(
            get_df,
            dataset_id,
            columns,
        )

    errors = []

    for method_name in (
        "get_frame",
        "get_dataframe",
        "materialize",
        "to_dataframe",
    ):
        method = getattr(datasets, method_name, None)

        if not callable(method):
            continue

        try:
            df = _call_frame_method(
                method,
                dataset_id,
                columns,
            )
        except Exception as exc:
            errors.append((method_name, exc))
            continue

        if df is not None:
            return df

    for method_name in ("get", "dataset"):
        method = getattr(datasets, method_name, None)

        if not callable(method):
            continue

        try:
            dataset = method(dataset_id)
        except Exception as exc:
            errors.append((method_name, exc))
            continue

        if dataset is None:
            continue

        for attr in ("df", "dataframe", "frame"):
            df = getattr(dataset, attr, None)

            if df is not None:
                return (
                    df.loc[:, list(columns)]
                    if columns is not None
                    else df
                )

        for object_method_name in (
            "get_frame",
            "get_dataframe",
            "get_df",
            "to_dataframe",
            "materialize",
        ):
            object_method = getattr(
                dataset,
                object_method_name,
                None,
            )

            if not callable(object_method):
                continue

            try:
                df = _call_bound_frame_method(
                    object_method,
                    columns,
                )
            except Exception as exc:
                errors.append(
                    (
                        f"{method_name}.{object_method_name}",
                        exc,
                    )
                )
                continue

            if df is not None:
                return df

    if errors:
        method_name, error = errors[0]

        raise RuntimeError(
            f"Could not materialise dataset {dataset_id!r}; "
            f"{method_name} failed: {error}"
        ) from error

    raise KeyError(
        f"Could not materialise dataset {dataset_id!r}."
    )

def list_dataset_columns(
    context: Any,
    dataset_id: Optional[str],
) -> List[str]:
    """Return available columns for a dataset."""

    if not dataset_id:
        return []

    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return []

    dataset_id = str(dataset_id)

    for method_name in (
        "columns",
        "list_columns",
        "get_columns",
        "dataset_columns",
    ):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                return [str(c) for c in method(dataset_id)]
            except Exception:
                pass

    try:
        df = get_dataset_frame(context, dataset_id)
    except Exception:
        return []

    try:
        return [str(c) for c in list(df.columns)]
    except Exception:
        return []

def mapped_column(
    context: Any,
    dataset_id: str,
    semantic_name: str,
) -> Optional[str]:
    """Resolve a semantic column mapping from DatasetManager if available."""

    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None

    dataset_id = str(dataset_id)
    semantic_name = str(semantic_name)

    for method_name in (
        "get_mapping",
        "mapped_column",
        "get_mapped_column",
        "resolve_mapping",
        "mapping_for",
    ):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                value = method(dataset_id, semantic_name)
                if value:
                    return str(value)
            except Exception:
                pass

    for method_name in (
        "get_mappings",
        "mappings",
        "column_mappings",
    ):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                mappings = method(dataset_id)
            except Exception:
                mappings = None

            if isinstance(mappings, Mapping):
                value = mappings.get(semantic_name)
                if value:
                    return str(value)

    return None

def infer_column_bindings(
    context: Any,
    dataset_id: str,
    recipe_spec: Any = None,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Infer record-id, target, image, and feature columns.

    The old method probably did not accept recipe_spec/params. The new runner
    calls it with both, so keep this widened signature.
    """

    params = dict(params or {})
    columns = list_dataset_columns(context, dataset_id)
    column_set = {str(c) for c in columns}
    lower_to_original = {str(c).lower(): str(c) for c in columns}

    def from_params(*names: str) -> Optional[str]:
        for name in names:
            value = params.get(name)
            if value:
                return str(value)
        return None

    def from_mapping(*semantic_names: str) -> Optional[str]:
        for semantic_name in semantic_names:
            value = mapped_column(context, dataset_id, semantic_name)
            if value:
                return value
        return None

    def from_aliases(*aliases: str) -> Optional[str]:
        for alias in aliases:
            value = lower_to_original.get(alias.lower())
            if value:
                return value
        return None

    record_id_column = (
        from_params("record_id_column", "record_id")
        or from_mapping("record_id")
        or from_aliases("record_id", "object_id", "source_id", "id")
    )

    target_column = (
        from_params("target_column", "label_column", "class_column")
        or from_mapping("target_label", "label", "class")
        or from_aliases(
            "target_label",
            "target",
            "label",
            "class",
            "y",
        )
    )

    image_column = (
        from_params("image_column", "image_path_column", "image_uri_column")
        or from_mapping("image.uri", "image", "cutout.path")
        or from_aliases(
            "image_uri",
            "image_path",
            "image",
            "path",
            "filepath",
            "file_path",
            "cutout_path",
        )
    )

    feature_columns = params.get("feature_columns") or params.get("input_columns")

    if isinstance(feature_columns, str):
        feature_columns = [
            part.strip()
            for part in feature_columns.replace("\n", ",").split(",")
            if part.strip()
        ]
    elif feature_columns is None:
        feature_columns = []
    else:
        feature_columns = [
            str(column)
            for column in feature_columns
            if column
        ]

    result: Dict[str, Any] = {}

    if record_id_column:
        result["record_id_column"] = record_id_column
        result["record_id"] = record_id_column

    if target_column:
        result["target_column"] = target_column
        result["target_label"] = target_column

    if image_column:
        result["image_column"] = image_column
        result["image_path"] = image_column
        result["image_uri"] = image_column

    result["feature_columns"] = list(feature_columns or [])
    result["input_columns"] = list(feature_columns or [])

    return result
