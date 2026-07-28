from __future__ import annotations

import shutil
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

from astronomicAL.plugins.core_ml.protocol import DataBinding, ProtocolConfig
from astronomicAL.plugins.core_ml.split_planner import create_streaming_partitions
from astronomicAL.plugins.core_ml.streaming_split_datasets import (
    MaterializedPartitionDataset,
    materialize_partition_datasets,
)

from . import acquisition
from . import actions
from . import state as al_state
from .selection_handoff import (
    apply_ranked_review_selection,
    disabled_review_selection,
)

ORIGIN = "core.active_learning"
HOLDOUT_SOURCES = frozenset({"split", "dataset"})

def start_session_action(
    context: Any,
    request: Any,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    """Create an AL session with fixed pool, validation, and test datasets.

    Validation and test are configured independently. Each role may either be
    split from the selected source dataset or reference an already registered
    dataset. Existing holdout datasets are reused by ID and are never copied or
    re-registered.
    """

    request = actions.coerce_request(request)
    params = dict(request.params or {})
    dataset_id = actions.resolve_dataset_id(context, request, params)
    if not dataset_id:
        raise ValueError("start_session requires dataset_id or an active dataset.")

    seed = int(params.get("seed", 42))
    initial_k = max(0, int(params.get("initial_k", 20)))
    target_column = str(
        params.get("target_column") or params.get("label_column") or "al_label"
    ).strip()
    record_id_column = acquisition.resolve_record_id_column(context, dataset_id)
    if not record_id_column or _uses_index(record_id_column):
        raise ValueError(
            "Active Learning requires a physical record_id column. "
            "Map record_id to a dataset column rather than Use Index."
        )

    validation_source = _holdout_source(
        params.get("validation_source"),
        role="validation",
    )
    test_source = _holdout_source(
        params.get("test_source"),
        role="test",
    )
    validation_dataset_id = _external_dataset_id(
        params,
        role="validation",
        source=validation_source,
    )
    test_dataset_id = _external_dataset_id(
        params,
        role="test",
        source=test_source,
    )

    validation_size = _split_fraction(
        params.get("session_validation_size", 0.1),
        role="validation",
        source=validation_source,
    )
    test_size = _split_fraction(
        params.get("session_test_size", 0.2),
        role="test",
        source=test_source,
    )
    split_fraction = (
        (validation_size if validation_source == "split" else 0.0)
        + (test_size if test_source == "split" else 0.0)
    )
    if split_fraction >= 1.0:
        raise ValueError(
            "The validation and test fractions split from the pool must sum "
            "to less than one."
        )

    _validate_dataset_protocol(
        context,
        source_dataset_id=str(dataset_id),
        validation_source=validation_source,
        validation_dataset_id=validation_dataset_id,
        test_source=test_source,
        test_dataset_id=test_dataset_id,
        record_id_column=str(record_id_column),
        target_column=target_column,
    )

    recipe_profile_id = str(
        params.get("recipe_profile_id")
        or params.get("recipe_profile_artifact_id")
        or ""
    ).strip()
    recipe_profile_name = str(params.get("recipe_profile_name") or "").strip()
    label_profile = dict(params.get("label_profile") or {})
    requested_task_type = al_state.parse_task_type(
        params.get("task_type")
        or params.get("problem_type")
        or label_profile.get("task_type")
        or "auto",
        default=al_state.TASK_UNKNOWN,
    )
    if requested_task_type == al_state.TASK_UNKNOWN and target_column:
        label_profile = actions.infer_label_profile_from_column(
            context,
            dataset_id=dataset_id,
            column=target_column,
        )
        task_type = al_state.parse_task_type(
            label_profile.get("task_type"),
            default=al_state.TASK_CLASSIFICATION,
        )
    else:
        task_type = (
            requested_task_type
            if requested_task_type != al_state.TASK_UNKNOWN
            else al_state.TASK_CLASSIFICATION
        )
    label_profile.setdefault("task_type", task_type)
    label_profile.setdefault("column", target_column)

    label_options = al_state.parse_label_options(params.get("label_options") or [])
    if task_type == al_state.TASK_REGRESSION:
        label_options = []
    elif (
        not label_options
        and bool(params.get("infer_labels_from_column", True))
        and target_column
    ):
        label_options = actions.infer_label_options_from_column(
            context,
            dataset_id=dataset_id,
            column=target_column,
        )

    protocol_result = create_session_protocol(
        context=context,
        source_dataset_id=str(dataset_id),
        record_id_column=str(record_id_column),
        target_column=target_column,
        task_type=task_type,
        validation_source=validation_source,
        validation_dataset_id=validation_dataset_id,
        test_source=test_source,
        test_dataset_id=test_dataset_id,
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
        params=params,
        cancel_token=cancel_token,
    )
    pool_dataset_id = str(protocol_result["pool_dataset_id"])
    validation_dataset_id = str(protocol_result["validation_dataset_id"])
    test_dataset_id = str(protocol_result["test_dataset_id"])

    pool_row_count = int(protocol_result["counts"]["pool"])
    initial_k = min(initial_k, pool_row_count)
    selected_row_ids = acquisition.sample_dataset_row_ids(
        context,
        dataset_id=pool_dataset_id,
        k=initial_k,
        seed=seed,
    )

    session_contract = dict(
        params.get("session_contract") or params.get("contract") or {}
    )
    session_contract["data_protocol"] = dict(protocol_result["data_protocol"])
    session = al_state.create_session(
        dataset_id=dataset_id,
        pool_dataset_id=pool_dataset_id,
        validation_dataset_id=validation_dataset_id,
        test_dataset_id=test_dataset_id,
        recipe_id=str(params.get("recipe_id") or ""),
        recipe_profile_id=recipe_profile_id,
        recipe_profile_name=recipe_profile_name,
        al_protocol=str(params.get("al_protocol") or "review"),
        label_options=label_options,
        seed=seed,
        target_column=target_column,
        task_type=task_type,
        label_profile=label_profile,
        contract=session_contract,
    )
    session["storage"] = {
        key: params[key]
        for key in (
            "artifact_root",
            "label_output_dir",
            "membership_output_dir",
            "label_storage_format",
            "membership_storage_format",
            "label_storage_batch_size",
            "membership_storage_batch_size",
        )
        if params.get(key) not in (None, "")
    }

    records = acquisition.build_initial_records(selected_row_ids)
    batch_payload = acquisition.create_batch_payload(
        dataset_id=pool_dataset_id,
        session=session,
        strategy_id="initial_random",
        records=records,
        params=params,
        predictions_artifact_id=None,
        kind="initial_random",
        rank_stats={
            "strategy_id": "initial_random",
            "ranked_count": len(records),
            "requested_k": initial_k,
            "eligible_pool_count": pool_row_count,
            "seed": seed,
        },
    )
    batch_artifact_id = context.artifacts.put(
        al_state.ARTIFACT_BATCH,
        batch_payload,
        dataset_id=pool_dataset_id,
        row_ids=selected_row_ids,
        params={
            "session_id": session["session_id"],
            "strategy_id": "initial_random",
            "seed": seed,
            "k": initial_k,
            "pool_row_count": pool_row_count,
        },
    )
    session = al_state.with_last_batch(
        session,
        batch_artifact_id=batch_artifact_id,
        strategy_id="initial_random",
        row_ids=selected_row_ids,
        kind="initial_random",
    )
    session_artifact_id = actions.put_session(context, session)

    if bool(params.get("make_selection", True)):
        selection_result = apply_ranked_review_selection(
            context,
            source_dataset_id=str(dataset_id),
            pool_dataset_id=pool_dataset_id,
            row_ids=selected_row_ids,
            session_artifact_id=session_artifact_id,
            batch_artifact_id=batch_artifact_id,
            strategy_id="initial_random",
            origin=f"{ORIGIN}.start_session",
        )
    else:
        selection_result = disabled_review_selection(selected_row_ids)

    payload = {
        "session_artifact_id": session_artifact_id,
        "session_id": session["session_id"],
        "dataset_id": dataset_id,
        "source_dataset_id": dataset_id,
        "pool_dataset_id": pool_dataset_id,
        "validation_dataset_id": validation_dataset_id,
        "test_dataset_id": test_dataset_id,
        "validation_source": validation_source,
        "test_source": test_source,
        "partition_counts": dict(protocol_result["counts"]),
        "initial_count": len(selected_row_ids),
        "count": len(selected_row_ids),
        "task_type": task_type,
        "label_profile": label_profile,
        "selection_applied": selection_result.applied,
        "selection_dataset_id": selection_result.dataset_id,
        "focused_row_id": selection_result.focused_row_id,
        "selection_reason": selection_result.reason,
        "data_protocol": dict(protocol_result["data_protocol"]),
    }
    actions.publish(context, "al.session.created", payload)
    actions.publish(context, "al.session.protocol.created", payload)
    return {
        "ok": True,
        **payload,
        "batch_artifact_id": batch_artifact_id,
        "initial_row_ids": selected_row_ids,
        "counts": al_state.counts(session),
    }

def create_session_protocol(
    *,
    context: Any,
    source_dataset_id: str,
    record_id_column: str,
    target_column: str,
    task_type: str,
    validation_source: str,
    validation_dataset_id: str,
    test_source: str,
    test_dataset_id: str,
    validation_size: float,
    test_size: float,
    seed: int,
    params: Mapping[str, Any],
    cancel_token: Any = None,
) -> Dict[str, Any]:
    """Resolve and, where requested, materialise the AL data protocol."""

    source = context.datasets.get_source(source_dataset_id)
    source_columns = [str(column) for column in source.columns()]
    source_row_count = _required_row_count(
        source,
        dataset_id=source_dataset_id,
    )
    split_roles = {
        role
        for role, mode in (
            ("validation", validation_source),
            ("test", test_source),
        )
        if mode == "split"
    }

    if not split_roles:
        counts = {
            "pool": source_row_count,
            "validation": _required_dataset_row_count(
                context,
                validation_dataset_id,
            ),
            "test": _required_dataset_row_count(context, test_dataset_id),
        }
        protocol = _data_protocol_payload(
            source_dataset_id=source_dataset_id,
            pool_dataset_id=source_dataset_id,
            validation_source=validation_source,
            validation_dataset_id=validation_dataset_id,
            test_source=test_source,
            test_dataset_id=test_dataset_id,
            validation_size=validation_size,
            test_size=test_size,
            counts=counts,
            seed=seed,
            manifest=None,
            root=None,
        )
        return {
            "source_dataset_id": source_dataset_id,
            "pool_dataset_id": source_dataset_id,
            "validation_dataset_id": validation_dataset_id,
            "test_dataset_id": test_dataset_id,
            "counts": counts,
            "manifest": None,
            "root": None,
            "data_protocol": protocol,
        }

    minimum_rows = len(split_roles) + 1
    if source_row_count < minimum_rows:
        raise ValueError(
            "The source dataset does not contain enough rows for the requested "
            f"split protocol: {source_row_count} rows for {len(split_roles)} "
            "split holdout role(s)."
        )

    split_token = uuid.uuid4().hex[:12]
    root = _session_split_root(context, params) / (
        f"{_safe_name(source_dataset_id)}-{split_token}"
    )
    root.mkdir(parents=True, exist_ok=False)
    protocol_config = ProtocolConfig.from_params(
        {
            "dataset_id": source_dataset_id,
            "protocol_split_strategy": "random",
            "protocol_validation_source": validation_source,
            "protocol_validation_dataset_id": (
                validation_dataset_id if validation_source == "dataset" else None
            ),
            "protocol_test_source": test_source,
            "protocol_test_dataset_id": (
                test_dataset_id if test_source == "dataset" else None
            ),
            "protocol_validation_size": validation_size,
            "protocol_test_size": test_size,
            "protocol_random_state": seed,
            "protocol_materialize_split_datasets": False,
        }
    )
    binding = DataBinding(
        record_id_column=record_id_column,
        target_column=target_column,
        input_columns=[],
        image_column=None,
    )
    check_cancel = lambda: _check_cancelled(cancel_token)
    parts = create_streaming_partitions(
        context=context,
        root=root / "manifest",
        run_id=f"al-session-{split_token}",
        source_dataset_id=source_dataset_id,
        protocol=protocol_config,
        binding=binding,
        task_kind=(
            "regression"
            if task_type == al_state.TASK_REGRESSION
            else "classification"
        ),
        batch_size=max(
            1,
            int(params.get("session_split_scan_batch_size") or 65536),
        ),
        cancel_check=check_cancel,
    )

    partition_refs: dict[str, Any] = {}
    train_ref = parts.partition_ref("train")
    if train_ref is None:
        raise RuntimeError("Split planner did not create the pool partition.")
    partition_refs["pool"] = train_ref
    for role in sorted(split_roles):
        ref = parts.partition_ref(role)
        if ref is None:
            raise RuntimeError(f"Split planner did not create {role!r}.")
        partition_refs[role] = ref

    dataset_ids = {
        "pool": _unique_dataset_id(
            context,
            f"{source_dataset_id}__al_pool_{split_token}",
        ),
        "validation": (
            validation_dataset_id
            if validation_source == "dataset"
            else _unique_dataset_id(
                context,
                f"{source_dataset_id}__al_validation_{split_token}",
            )
        ),
        "test": (
            test_dataset_id
            if test_source == "dataset"
            else _unique_dataset_id(
                context,
                f"{source_dataset_id}__al_test_{split_token}",
            )
        ),
    }

    materialized_roles = {
        role: ref for role, ref in partition_refs.items()
    }
    registered_count = 0
    try:
        materialized = materialize_partition_datasets(
            context=context,
            source_dataset_id=source_dataset_id,
            partitions=materialized_roles,
            columns=source_columns,
            role_roots={role: root / role for role in materialized_roles},
            batch_size=max(
                1,
                int(params.get("session_split_output_batch_size") or 8192),
            ),
            cancel_check=check_cancel,
        )
        for output_role, result in materialized.items():
            _register_partition_dataset(
                context=context,
                source_dataset_id=source_dataset_id,
                output_dataset_id=dataset_ids[output_role],
                output_role=output_role,
                materialized=result,
                partition_ref=partition_refs[output_role],
                columns=source_columns,
                record_id_column=record_id_column,
                target_column=target_column,
                task_type=task_type,
            )
            registered_count += 1
    except Exception:
        if registered_count == 0:
            shutil.rmtree(root, ignore_errors=True)
        raise

    source_owned_counts = {
        role: int(ref.row_count) for role, ref in partition_refs.items()
    }
    if sum(source_owned_counts.values()) != source_row_count:
        raise RuntimeError(
            "Active Learning split roles did not preserve the complete source "
            f"dataset: expected {source_row_count}, created "
            f"{sum(source_owned_counts.values())}."
        )

    counts = {
        "pool": source_owned_counts["pool"],
        "validation": (
            source_owned_counts["validation"]
            if validation_source == "split"
            else _required_dataset_row_count(context, validation_dataset_id)
        ),
        "test": (
            source_owned_counts["test"]
            if test_source == "split"
            else _required_dataset_row_count(context, test_dataset_id)
        ),
    }
    manifest = (
        parts.split_manifest.to_dict()
        if getattr(parts, "split_manifest", None) is not None
        else None
    )
    protocol = _data_protocol_payload(
        source_dataset_id=source_dataset_id,
        pool_dataset_id=dataset_ids["pool"],
        validation_source=validation_source,
        validation_dataset_id=dataset_ids["validation"],
        test_source=test_source,
        test_dataset_id=dataset_ids["test"],
        validation_size=validation_size,
        test_size=test_size,
        counts=counts,
        seed=seed,
        manifest=manifest,
        root=str(root),
    )
    return {
        "source_dataset_id": source_dataset_id,
        "pool_dataset_id": dataset_ids["pool"],
        "validation_dataset_id": dataset_ids["validation"],
        "test_dataset_id": dataset_ids["test"],
        "counts": counts,
        "manifest": manifest,
        "root": str(root),
        "data_protocol": protocol,
    }

def _data_protocol_payload(
    *,
    source_dataset_id: str,
    pool_dataset_id: str,
    validation_source: str,
    validation_dataset_id: str,
    test_source: str,
    test_dataset_id: str,
    validation_size: float,
    test_size: float,
    counts: Mapping[str, int],
    seed: int,
    manifest: Optional[Mapping[str, Any]],
    root: Optional[str],
) -> Dict[str, Any]:
    split_fraction = (
        (validation_size if validation_source == "split" else 0.0)
        + (test_size if test_source == "split" else 0.0)
    )
    return {
        "schema_version": 1,
        "source_dataset_id": source_dataset_id,
        "seed": int(seed),
        "pool": {
            "source": "split" if split_fraction else "dataset",
            "dataset_id": pool_dataset_id,
            "row_count": int(counts["pool"]),
            "fraction": 1.0 - split_fraction,
        },
        "validation": {
            "source": validation_source,
            "dataset_id": validation_dataset_id,
            "row_count": int(counts["validation"]),
            "fraction": (
                validation_size if validation_source == "split" else None
            ),
        },
        "test": {
            "source": test_source,
            "dataset_id": test_dataset_id,
            "row_count": int(counts["test"]),
            "fraction": test_size if test_source == "split" else None,
        },
        "manifest": None if manifest is None else dict(manifest),
        "materialization_root": root,
    }

def _validate_dataset_protocol(
    context: Any,
    *,
    source_dataset_id: str,
    validation_source: str,
    validation_dataset_id: str,
    test_source: str,
    test_dataset_id: str,
    record_id_column: str,
    target_column: str,
) -> None:
    registered = set(context.datasets.list_ids())
    if source_dataset_id not in registered:
        raise KeyError(f"Unknown source dataset_id: {source_dataset_id}")

    external_ids = []
    for role, source, dataset_id in (
        ("validation", validation_source, validation_dataset_id),
        ("test", test_source, test_dataset_id),
    ):
        if source != "dataset":
            continue
        if not dataset_id:
            raise ValueError(
                f"{role}_dataset_id is required when {role}_source='dataset'."
            )
        if dataset_id not in registered:
            raise KeyError(
                f"The selected {role} dataset is not registered: {dataset_id!r}."
            )
        if dataset_id == source_dataset_id:
            raise ValueError(
                f"The {role} dataset must be different from the pool source."
            )
        external_ids.append(dataset_id)
        _validate_required_columns(
            context,
            dataset_id=dataset_id,
            record_id_column=record_id_column,
            target_column=target_column,
            role=role,
        )

    if len(external_ids) == 2 and external_ids[0] == external_ids[1]:
        raise ValueError(
            "Validation and test must use different existing datasets."
        )

    _validate_required_columns(
        context,
        dataset_id=source_dataset_id,
        record_id_column=record_id_column,
        target_column=target_column,
        role="source",
    )

def _validate_required_columns(
    context: Any,
    *,
    dataset_id: str,
    record_id_column: str,
    target_column: str,
    role: str,
) -> None:
    source = context.datasets.get_source(dataset_id)
    columns = {str(column) for column in source.columns()}
    missing = [
        column
        for column in (record_id_column, target_column)
        if column not in columns
    ]
    if missing:
        raise ValueError(
            f"The {role} dataset {dataset_id!r} is missing required "
            f"column(s): {', '.join(missing)}."
        )

def _register_partition_dataset(
    *,
    context: Any,
    source_dataset_id: str,
    output_dataset_id: str,
    output_role: str,
    materialized: MaterializedPartitionDataset,
    partition_ref: Any,
    columns: list[str],
    record_id_column: str,
    target_column: str,
    task_type: str,
) -> None:
    source_dataset = context.datasets.get(source_dataset_id)
    mappings = dict(actions.dataset_mappings(context, source_dataset_id) or {})
    mappings = actions.filter_mappings_to_columns(mappings, columns)
    mappings["record_id"] = record_id_column
    context.datasets.register_parquet(
        output_dataset_id,
        list(materialized.paths),
        name=(
            f"{getattr(source_dataset, 'name', source_dataset_id)} — "
            f"AL {output_role.title()}"
        ),
        source_kind="active_learning.session_partition",
        origin="core.active_learning.start_session",
        source_dataset_id=source_dataset_id,
        derived_from=source_dataset_id,
        derived_role=output_role,
        is_derived=True,
        is_active_learning_partition=True,
        row_count=int(materialized.row_count),
        target_column=target_column,
        task_type=task_type,
        problem_type=task_type,
        columns=list(materialized.columns),
        column_mappings=mappings,
        split_manifest_uri=str(partition_ref.manifest.uri),
        split_manifest_sha256=str(partition_ref.manifest.sha256),
        split_manifest_role=str(partition_ref.role),
        physical_access="sequential_parquet_scan",
        created_at=time.time(),
    )
    event = {
        "dataset_id": output_dataset_id,
        "source_dataset_id": source_dataset_id,
        "role": output_role,
        "row_count": int(materialized.row_count),
        "origin": "core.active_learning.start_session",
        "physical_access": "sequential_parquet_scan",
    }
    actions.publish(context, "dataset.registered", event)
    actions.publish(context, "dataset.loaded", event)

def _session_split_root(context: Any, params: Mapping[str, Any]) -> Path:
    explicit = (
        params.get("session_split_output_dir")
        or params.get("artifact_root")
        or params.get("training_output_dir")
    )
    if explicit:
        root = Path(str(explicit)).expanduser()
    else:
        cache_dir = getattr(
            getattr(context, "artifacts", None),
            "_cache_dir",
            None,
        )
        root = (
            Path(str(cache_dir)).expanduser()
            if cache_dir
            else Path.cwd() / ".astronomical"
        )
    output = root / "active_learning" / "session_splits"
    output.mkdir(parents=True, exist_ok=True)
    return output

def _external_dataset_id(
    params: Mapping[str, Any],
    *,
    role: str,
    source: str,
) -> str:
    value = str(params.get(f"{role}_dataset_id") or "").strip()
    if source == "dataset" and not value:
        raise ValueError(
            f"{role}_dataset_id is required when {role}_source='dataset'."
        )
    return value

def _holdout_source(value: Any, *, role: str) -> str:
    source = str(value or "split").strip().lower()
    if source not in HOLDOUT_SOURCES:
        raise ValueError(
            f"{role}_source must be 'split' or 'dataset', got {value!r}."
        )
    return source

def _split_fraction(value: Any, *, role: str, source: str) -> float:
    if source != "split":
        return 0.0
    result = _fraction(value, name=f"session_{role}_size")
    if result <= 0.0:
        raise ValueError(
            f"session_{role}_size must be greater than zero when "
            f"{role}_source='split'."
        )
    return result

def _required_dataset_row_count(context: Any, dataset_id: str) -> int:
    return _required_row_count(
        context.datasets.get_source(dataset_id),
        dataset_id=dataset_id,
    )

def _required_row_count(source: Any, *, dataset_id: str) -> int:
    value = source.row_count()
    if value is None:
        raise ValueError(
            f"Dataset {dataset_id!r} does not expose a row count required for "
            "the Active Learning protocol."
        )
    count = int(value)
    if count <= 0:
        raise ValueError(f"Dataset {dataset_id!r} contains no rows.")
    return count

def _unique_dataset_id(context: Any, base: str) -> str:
    existing = set(context.datasets.list_ids())
    if base not in existing:
        return base
    for index in range(2, 10000):
        candidate = f"{base}_{index}"
        if candidate not in existing:
            return candidate
    return f"{base}_{uuid.uuid4().hex[:8]}"

def _fraction(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if result < 0 or result >= 1:
        raise ValueError(f"{name} must be between zero and one.")
    return result

def _uses_index(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        "use index",
        "use_index",
        "__index__",
        "index",
    }

def _check_cancelled(cancel_token: Any) -> None:
    if cancel_token is None:
        return

    for method_name in ("raise_if_cancelled", "check_cancelled"):
        checker = getattr(cancel_token, method_name, None)
        if callable(checker):
            checker()
            return

    cancelled = getattr(cancel_token, "cancelled", False)
    if callable(cancelled):
        cancelled = cancelled()
    if bool(cancelled):
        raise RuntimeError("Active Learning session creation was cancelled.")

def _safe_name(value: Any) -> str:
    text = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "-"
        for character in str(value or "dataset")
    ).strip("-._")
    return text[:96] or "dataset"