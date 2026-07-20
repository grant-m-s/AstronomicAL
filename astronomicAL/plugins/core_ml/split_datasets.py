# astronomicAL/plugins/core_ml/split_datasets.py
from __future__ import annotations

import re
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from .feature_columns import parse_column_list
from .serialization import json_safe

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}

def should_materialize_split_datasets(params: Mapping[str, Any]) -> bool:
    """Return whether managed recipe split partitions should become datasets.

    Default is True because explicit train/validation/test datasets are the
    safer platform behaviour: downstream panels/actions must choose a split
    dataset intentionally instead of accidentally operating on the full source.
    """

    raw = dict(params or {}).get("protocol_materialize_split_datasets", True)

    if isinstance(raw, bool):
        return raw

    text = str(raw).strip().lower()
    if text in _FALSE_VALUES:
        return False
    if text in _TRUE_VALUES:
        return True

    return True

def materialize_split_datasets(
    *,
    context: Any,
    source_dataset_id: str,
    run_id: str,
    recipe_id: str,
    recipe_version: str,
    protocol: Any,
    binding: Any,
    params: Mapping[str, Any],
    partitions: Mapping[str, Any],
    fallback_frames: Mapping[str, pd.DataFrame],
) -> Dict[str, str]:
    """Register split-derived partitions as first-class AstronomicAL datasets.

    Only pass partitions that were actually split from `source_dataset_id`.
    External validation/test datasets are already first-class datasets and should
    not be re-registered here.

    Returns role -> dataset_id, using roles: train, validation, test.
    """

    if not should_materialize_split_datasets(params):
        return {}

    roles = {
        role: partition
        for role, partition in dict(partitions or {}).items()
        if partition is not None
    }
    if not roles:
        return {}

    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return {}

    prefix = _split_dataset_prefix(
        source_dataset_id=source_dataset_id,
        run_id=run_id,
        recipe_id=recipe_id,
        params=params,
    )

    created: Dict[str, str] = {}

    for role, partition in roles.items():
        frame = _materialize_role_frame(
            context=context,
            source_dataset_id=source_dataset_id,
            role=role,
            partition=partition,
            fallback_frame=fallback_frames.get(role),
            record_id_column=binding.record_id_column,
            target_column=binding.target_column,
            params=params,
        )

        dataset_id = f"{prefix}__{role}"
        name = _split_dataset_name(
            context=context,
            source_dataset_id=source_dataset_id,
            role=role,
            run_id=run_id,
        )

        meta = {
            "domain": "ml",
            "origin": "core.ml.recipe_split",
            "source_kind": "ml_recipe_split",
            "source_dataset_id": source_dataset_id,
            "derived_from": source_dataset_id,
            "derived_role": role,
            "is_derived": True,
            "is_ml_split_dataset": True,
            "ml_split_role": role,
            "ml_run_id": run_id,
            "recipe_id": recipe_id,
            "recipe_version": recipe_version,
            "protocol_id": getattr(protocol, "protocol_id", ""),
            "protocol_split_strategy": getattr(protocol, "split_strategy", ""),
            "protocol_validation_source": getattr(protocol, "validation_source", ""),
            "protocol_test_source": getattr(protocol, "test_source", ""),
            "record_id_column": binding.record_id_column,
            "target_column": binding.target_column,
            "row_count": int(len(frame)),
            "created_at": time.time(),
        }

        _register_dataset(
            datasets=datasets,
            dataset_id=dataset_id,
            frame=frame,
            name=name,
            meta=meta,
        )

        _copy_column_mappings(
            context=context,
            source_dataset_id=source_dataset_id,
            split_dataset_id=dataset_id,
            frame=frame,
            binding=binding,
        )

        created[role] = dataset_id

        _publish_dataset_events(
            context=context,
            dataset_id=dataset_id,
            role=role,
            source_dataset_id=source_dataset_id,
            run_id=run_id,
            protocol_id=getattr(protocol, "protocol_id", ""),
            row_count=len(frame),
        )

    if created:
        _publish(
            context,
            "ml.split_datasets.created",
            {
                "run_id": run_id,
                "source_dataset_id": source_dataset_id,
                "protocol_id": getattr(protocol, "protocol_id", ""),
                "dataset_ids": dict(created),
            },
        )

    return created

def _split_dataset_prefix(
    *,
    source_dataset_id: str,
    run_id: str,
    recipe_id: str,
    params: Mapping[str, Any],
) -> str:
    explicit = str(params.get("protocol_split_dataset_prefix") or "").strip()
    if explicit:
        return _safe_id(explicit)

    source = _safe_id(source_dataset_id)
    recipe = _safe_id(recipe_id)
    run = _safe_id(str(run_id)[:8] or "run")
    return f"{source}__ml_{recipe}_{run}"

def _split_dataset_name(
    *,
    context: Any,
    source_dataset_id: str,
    role: str,
    run_id: str,
) -> str:
    source_name = source_dataset_id

    try:
        dataset = context.datasets.get(source_dataset_id)
        source_name = str(getattr(dataset, "name", None) or source_dataset_id)
    except Exception:
        pass

    labels = {
        "train": "Train",
        "validation": "Validation",
        "test": "Test",
    }

    return f"{source_name} — ML {labels.get(role, role.title())} Split ({str(run_id)[:8]})"

def _materialize_role_frame(
    *,
    context: Any,
    source_dataset_id: str,
    role: str,
    partition: Any,
    fallback_frame: Optional[pd.DataFrame],
    record_id_column: str,
    target_column: Optional[str],
    params: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the dataset frame for one split role.

    By default this materialises full source rows, so downstream inspection
    panels still have all non-ML columns. Set protocol_split_dataset_columns to:

    - "all" / empty: full source rows by record id
    - "training": only the columns already loaded by the harness
    - comma/JSON list: explicit columns, plus record/target columns
    """

    column_mode = str(params.get("protocol_split_dataset_columns") or "all").strip()

    if column_mode.lower() == "training":
        if fallback_frame is None:
            raise ValueError(f"Cannot materialize {role} split without a frame.")
        return fallback_frame.reset_index(drop=True).copy(deep=False)

    explicit_columns = parse_column_list(column_mode)
    columns = None

    if explicit_columns and column_mode.lower() not in {"all", "*"}:
        wanted = [record_id_column]
        if target_column:
            wanted.append(target_column)
        wanted.extend(explicit_columns)
        columns = list(dict.fromkeys([str(c) for c in wanted if c]))

    try:
        frame = context.datasets.get_rows_by_ids(
            source_dataset_id,
            list(partition.record_ids),
            id_column=record_id_column,
            columns=columns,
        )
        frame = _restore_partition_order(
            frame,
            record_id_column=record_id_column,
            row_ids=partition.record_ids,
        )
    except Exception:
        if fallback_frame is None:
            raise
        frame = fallback_frame.reset_index(drop=True).copy(deep=False)

    return frame.reset_index(drop=True)

def _restore_partition_order(
    frame: pd.DataFrame,
    *,
    record_id_column: str,
    row_ids: Sequence[Any],
) -> pd.DataFrame:
    if frame is None or frame.empty or record_id_column not in frame.columns:
        return frame

    order = {str(row_id): index for index, row_id in enumerate(row_ids)}
    ordered = frame.copy(deep=False)
    ordered["__astronomical_split_order__"] = (
        ordered[record_id_column].astype(str).map(order)
    )

    ordered = ordered[
        ordered["__astronomical_split_order__"].notna()
    ].sort_values(
        "__astronomical_split_order__",
        kind="stable",
    )

    return ordered.drop(columns=["__astronomical_split_order__"]).reset_index(drop=True)

def _register_dataset(
    *,
    datasets: Any,
    dataset_id: str,
    frame: pd.DataFrame,
    name: str,
    meta: Mapping[str, Any],
) -> None:
    register_meta = dict(meta)

    register_meta.pop("source", None)

    ensure_registered = getattr(datasets, "ensure_registered", None)
    if callable(ensure_registered):
        ensure_registered(dataset_id, frame, name=name, **register_meta)
        return

    register = getattr(datasets, "register", None)
    if callable(register):
        register(dataset_id, frame, name=name, **register_meta)
        return

    raise RuntimeError("DatasetManager does not expose register/ensure_registered.")

def _copy_column_mappings(
    *,
    context: Any,
    source_dataset_id: str,
    split_dataset_id: str,
    frame: pd.DataFrame,
    binding: Any,
) -> None:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return

    frame_columns = {str(column) for column in frame.columns}
    mappings: Dict[str, Any] = {}

    try:
        mappings.update(dict(datasets.get_mappings(source_dataset_id) or {}))
    except Exception:
        pass

    if getattr(binding, "record_id_column", None):
        mappings.setdefault("record_id", binding.record_id_column)

    if getattr(binding, "target_column", None):
        mappings.setdefault("target_label", binding.target_column)

    if getattr(binding, "image_column", None):
        mappings.setdefault("image.path", binding.image_column)

    set_mapping = getattr(datasets, "set_mapping", None)
    if not callable(set_mapping):
        meta = datasets.get_meta(split_dataset_id)
        existing = dict(meta.get("column_mappings", {}) or {})
        existing.update(
            {
                str(key): str(value)
                for key, value in mappings.items()
                if _mapping_is_valid(value, frame_columns)
            }
        )
        meta["column_mappings"] = existing
        return

    for semantic_name, column_name in mappings.items():
        if not _mapping_is_valid(column_name, frame_columns):
            continue
        try:
            set_mapping(split_dataset_id, str(semantic_name), str(column_name))
        except Exception:
            pass

def _mapping_is_valid(column_name: Any, frame_columns: set[str]) -> bool:
    if column_name is None:
        return False

    text = str(column_name)
    return text in frame_columns or text.lower() in {
        "use index",
        "use_index",
        "__index__",
        "index",
    }

def _publish_dataset_events(
    *,
    context: Any,
    dataset_id: str,
    role: str,
    source_dataset_id: str,
    run_id: str,
    protocol_id: str,
    row_count: int,
) -> None:
    payload = {
        "dataset_id": dataset_id,
        "role": role,
        "source_dataset_id": source_dataset_id,
        "run_id": run_id,
        "protocol_id": protocol_id,
        "row_count": int(row_count),
        "origin": "core.ml",
    }

    for topic in (
        "dataset.created",
        "dataset.loaded",
        "dataset.updated",
        "ml.split_dataset.created",
    ):
        _publish(context, topic, payload)

def _publish(context: Any, topic: str, payload: Mapping[str, Any]) -> None:
    events = getattr(context, "events", None)
    publish = getattr(events, "publish", None)

    if callable(publish):
        publish(topic, json_safe(dict(payload)))

def _safe_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    text = text.strip("._-")
    return text or "dataset"
