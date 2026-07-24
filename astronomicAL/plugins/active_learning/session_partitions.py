from __future__ import annotations

import shutil
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

from astronomicAL.plugins.core_ml.protocol import DataBinding, ProtocolConfig
from astronomicAL.plugins.core_ml.split_planner import create_streaming_partitions
from astronomicAL.plugins.core_ml.streaming_split_datasets import (
    MaterializedPartitionDataset,
    materialize_partition_datasets,
)

from . import acquisition
from . import actions as legacy_actions
from . import state as al_state
from .selection_handoff import (
    apply_ranked_review_selection,
    disabled_review_selection,
)

ORIGIN = "core.active_learning"
_LEGACY_START_SESSION = legacy_actions.start_session_action

def install_legacy_bridge() -> None:
    """Make the custom panel use partitioned session creation too."""

    legacy_actions.start_session_action = start_session_action

def start_session_action(
    context: Any,
    request: Any,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    """Create fixed pool/validation/test partitions before initial sampling.

    The source dataset selected by the user remains the application's active
    dataset. The session owns references to its derived pool, validation, and
    test datasets and uses the pool reference for sampling, training,
    prediction, and query selection.
    """

    request = legacy_actions.coerce_request(request)
    params = dict(request.params or {})
    if not _bool_param(params.get("partition_whole_dataset"), True):
        return _LEGACY_START_SESSION(
            context,
            request,
            cancel_token=cancel_token,
        )

    dataset_id = legacy_actions.resolve_dataset_id(context, request, params)
    if not dataset_id:
        raise ValueError("start_session requires dataset_id or an active dataset.")
    seed = int(params.get("seed", 42))
    initial_k = max(0, int(params.get("initial_k", 20)))
    target_column = str(
        params.get("target_column") or params.get("label_column") or "al_label"
    )
    record_id_column = acquisition.resolve_record_id_column(context, dataset_id)
    if not record_id_column or _uses_index(record_id_column):
        raise ValueError(
            "Partitioned Active Learning requires a real record_id column. "
            "Map record_id to a dataset column rather than Use Index."
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
        label_profile = legacy_actions.infer_label_profile_from_column(
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
        label_options = legacy_actions.infer_label_options_from_column(
            context,
            dataset_id=dataset_id,
            column=target_column,
        )

    validation_size = _fraction(
        params.get("session_validation_size", params.get("validation_size", 0.1)),
        name="session_validation_size",
    )
    test_size = _fraction(
        params.get("session_test_size", params.get("test_size", 0.2)),
        name="session_test_size",
    )
    if validation_size <= 0:
        raise ValueError("session_validation_size must be greater than zero.")
    if test_size <= 0:
        raise ValueError("session_test_size must be greater than zero.")
    if validation_size + test_size >= 1:
        raise ValueError(
            "session_validation_size and session_test_size must sum to less than 1."
        )

    partition_result = create_session_partitions(
        context=context,
        source_dataset_id=str(dataset_id),
        record_id_column=str(record_id_column),
        target_column=target_column,
        task_type=task_type,
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
        params=params,
        cancel_token=cancel_token,
    )
    pool_dataset_id = partition_result["pool_dataset_id"]
    validation_dataset_id = partition_result["validation_dataset_id"]
    test_dataset_id = partition_result["test_dataset_id"]

    pool_row_count = int(partition_result["counts"]["pool"])
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
    session_contract["session_split"] = {
        "schema_version": 1,
        "source_dataset_id": str(dataset_id),
        "pool_dataset_id": pool_dataset_id,
        "validation_dataset_id": validation_dataset_id,
        "test_dataset_id": test_dataset_id,
        "pool_fraction": 1.0 - validation_size - test_size,
        "validation_fraction": validation_size,
        "test_fraction": test_size,
        "seed": seed,
        "counts": dict(partition_result["counts"]),
        "manifest": partition_result["manifest"],
    }
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
            "eligible_pool_count": partition_result["counts"]["pool"],
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
            "pool_row_count": partition_result["counts"]["pool"],
        },
    )
    session = al_state.with_last_batch(
        session,
        batch_artifact_id=batch_artifact_id,
        strategy_id="initial_random",
        row_ids=selected_row_ids,
        kind="initial_random",
    )
    session_artifact_id = legacy_actions.put_session(context, session)

    # Starting an Active Learning session must not replace the user-selected
    # active dataset. The batch artifact remains pool-owned, while the platform
    # selection handoff is projected onto the active source or pool dataset.
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
        "partition_counts": dict(partition_result["counts"]),
        "initial_count": len(selected_row_ids),
        "count": len(selected_row_ids),
        "task_type": task_type,
        "label_profile": label_profile,
        "selection_applied": selection_result.applied,
        "selection_dataset_id": selection_result.dataset_id,
        "focused_row_id": selection_result.focused_row_id,
        "selection_reason": selection_result.reason,
    }
    legacy_actions.publish(context, "al.session.created", payload)
    legacy_actions.publish(context, "al.session.partitions.created", payload)
    return {
        "ok": True,
        **payload,
        "batch_artifact_id": batch_artifact_id,
        "initial_row_ids": selected_row_ids,
        "counts": al_state.counts(session),
        "session_split": session_contract["session_split"],
    }

def create_session_partitions(
    *,
    context: Any,
    source_dataset_id: str,
    record_id_column: str,
    target_column: str,
    task_type: str,
    validation_size: float,
    test_size: float,
    seed: int,
    params: Mapping[str, Any],
    cancel_token: Any = None,
) -> Dict[str, Any]:
    source = context.datasets.get_source(source_dataset_id)
    available_columns = [str(column) for column in source.columns()]
    if record_id_column not in available_columns:
        raise KeyError(
            f"Dataset {source_dataset_id!r} is missing record ID column "
            f"{record_id_column!r}."
        )
    if target_column not in available_columns:
        raise KeyError(
            f"Dataset {source_dataset_id!r} is missing target column "
            f"{target_column!r}."
        )
    total_rows = source.row_count()
    if total_rows is not None and int(total_rows) < 3:
        raise ValueError("Active Learning partitioning requires at least three rows.")

    split_token = uuid.uuid4().hex[:12]
    root = _session_split_root(context, params) / (
        f"{_safe_name(source_dataset_id)}-{split_token}"
    )
    root.mkdir(parents=True, exist_ok=False)
    protocol = ProtocolConfig.from_params(
        {
            "dataset_id": source_dataset_id,
            "protocol_split_strategy": "random",
            "protocol_validation_source": "split",
            "protocol_test_source": "split",
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
        protocol=protocol,
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

    role_map = {"pool": "train", "validation": "validation", "test": "test"}
    partition_refs = {}
    for output_role, manifest_role in role_map.items():
        ref = parts.partition_ref(manifest_role)
        if ref is None:
            raise RuntimeError(
                f"Split planner did not create {manifest_role!r}."
            )
        partition_refs[output_role] = ref

    requested_ids = {
        "pool": str(params.get("session_pool_dataset_id") or "").strip(),
        "validation": str(
            params.get("session_validation_dataset_id") or ""
        ).strip(),
        "test": str(params.get("session_test_dataset_id") or "").strip(),
    }
    dataset_ids = {
        role: requested_ids[role]
        or _unique_dataset_id(
            context,
            f"{source_dataset_id}__al_{role}_{split_token}",
        )
        for role in role_map
    }

    registered_count = 0
    try:
        materialized = materialize_partition_datasets(
            context=context,
            source_dataset_id=source_dataset_id,
            partitions=partition_refs,
            columns=available_columns,
            role_roots={role: root / role for role in role_map},
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
                columns=available_columns,
                record_id_column=record_id_column,
                target_column=target_column,
                task_type=task_type,
            )
            registered_count += 1
    except Exception:
        if registered_count == 0:
            shutil.rmtree(root, ignore_errors=True)
        raise

    counts = {
        role: int(partition_refs[role].row_count) for role in role_map
    }
    if total_rows is not None and sum(counts.values()) != int(total_rows):
        raise RuntimeError(
            "Active Learning session split did not preserve the complete dataset: "
            f"expected {total_rows}, created {sum(counts.values())}."
        )
    return {
        "source_dataset_id": source_dataset_id,
        "pool_dataset_id": dataset_ids["pool"],
        "validation_dataset_id": dataset_ids["validation"],
        "test_dataset_id": dataset_ids["test"],
        "counts": counts,
        "manifest": parts.split_manifest.to_dict() if parts.split_manifest else None,
        "root": str(root),
    }

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
    mappings = dict(
        legacy_actions.dataset_mappings(context, source_dataset_id) or {}
    )
    mappings = legacy_actions.filter_mappings_to_columns(mappings, columns)
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
    legacy_actions.publish(context, "dataset.registered", event)
    legacy_actions.publish(context, "dataset.loaded", event)

def _session_split_root(context: Any, params: Mapping[str, Any]) -> Path:
    explicit = (
        params.get("session_split_output_dir")
        or params.get("artifact_root")
        or params.get("training_output_dir")
    )
    if explicit:
        root = Path(str(explicit)).expanduser()
    else:
        config = getattr(context, "config", None)
        configured = getattr(config, "ml_artifact_root", None)
        root = (
            Path(str(configured)).expanduser()
            if configured
            else Path.home() / ".astronomical" / "ml_artifacts"
        )
    output = root / "active_learning" / "session_splits"
    output.mkdir(parents=True, exist_ok=True)
    return output

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

def _bool_param(value: Any, default: bool) -> bool:
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

def _uses_index(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        "use index",
        "use_index",
        "__index__",
        "index",
    }

def _check_cancelled(cancel_token: Any) -> None:
    """Raise only when the supplied job token reports cancellation.

    ``JobManager`` tokens may expose ``cancelled`` as either a boolean property
    or a zero-argument method. Treating the bound method itself as a boolean
    makes every fresh token appear cancelled because bound methods are truthy.
    Prefer the token's explicit checking methods when available, then resolve
    the compatibility attribute safely.
    """

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
