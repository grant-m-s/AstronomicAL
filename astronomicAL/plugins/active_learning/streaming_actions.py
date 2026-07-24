from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence

from . import acquisition
from . import actions
from . import state as al_state
from .score_storage import ScoreTableRef, ScoreTableWriter, delete_score_ref, score_output_root
from .streaming_acquisition import acquire_query_batch_streaming, iter_pool_score_rows
from .streaming_io import check_cancelled
from .label_overlay import attach_session_labels
from .selection_handoff import (
    apply_ranked_review_selection,
    disabled_review_selection,
    session_source_dataset_id,
)

ORIGIN = "core.active_learning"


def train_from_session_action(
    context: Any,
    request: Any,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    """Train labelled pool rows against the session's fixed data protocol."""

    from . import core_ml_bridge

    action_request = actions.coerce_request(request)
    params = dict(action_request.params or {})
    session_artifact_id = str(params.get("session_artifact_id") or "").strip()
    if not session_artifact_id:
        raise ValueError("train_from_session requires session_artifact_id.")

    session = al_state.coerce_session(context.artifacts.get(session_artifact_id))
    validation_dataset_id = str(
        session.get("validation_dataset_id") or ""
    ).strip()
    test_dataset_id = str(session.get("test_dataset_id") or "").strip()
    if not validation_dataset_id:
        raise ValueError(
            "The Active Learning session has no fixed validation dataset."
        )
    if not test_dataset_id:
        raise ValueError(
            "The Active Learning session has no fixed test dataset."
        )

    for role, fixed_id in (
        ("validation", validation_dataset_id),
        ("test", test_dataset_id),
    ):
        requested = str(params.get(f"{role}_dataset_id") or "").strip()
        if requested and requested != fixed_id:
            raise ValueError(
                f"The {role} dataset is fixed by the Active Learning session "
                f"protocol as {fixed_id!r}; received {requested!r}."
            )

    registered = set(context.datasets.list_ids())
    missing = [
        dataset_id
        for dataset_id in (validation_dataset_id, test_dataset_id)
        if dataset_id not in registered
    ]
    if missing:
        raise KeyError(
            "Active Learning holdout dataset(s) are not registered: "
            + ", ".join(missing)
        )

    recipe_params = dict(params.get("recipe_params") or {})
    recipe_params.update(
        {
            "protocol_validation_source": "dataset",
            "protocol_validation_dataset_id": validation_dataset_id,
            "protocol_test_source": "dataset",
            "protocol_test_dataset_id": test_dataset_id,
            "protocol_materialize_split_datasets": False,
        }
    )
    params.update(
        {
            "recipe_params": recipe_params,
            "validation_dataset_id": validation_dataset_id,
            "test_dataset_id": test_dataset_id,
        }
    )
    rewritten = actions.coerce_request(
        {
            "dataset_id": action_request.dataset_id,
            "row_ids": action_request.row_ids,
            "columns": list(action_request.columns or []),
            "params": params,
            "artifact_id": action_request.artifact_id,
            "origin": action_request.origin,
        }
    )
    return core_ml_bridge.run_training_round(
        context,
        rewritten,
        cancel_token=cancel_token,
    )

def query_batch_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    request = actions.coerce_request(request)
    params = dict(request.params or {})
    session_artifact_id = str(params.get("session_artifact_id") or "").strip()
    if not session_artifact_id:
        raise ValueError("query_batch requires session_artifact_id.")

    session = al_state.coerce_session(context.artifacts.get(session_artifact_id))
    dataset_id = acquisition.session_pool_dataset_id(session)
    if not dataset_id:
        raise ValueError("query_batch could not determine the AL session pool dataset.")

    predictions_artifact_id = str(
        params.get("predictions_artifact_id")
        or request.artifact_id
        or al_state.latest_reference(session, "predictions_artifact_id")
        or ""
    ).strip()
    if not predictions_artifact_id:
        raise ValueError(
            "query_batch requires predictions_artifact_id or a session with latest predictions."
        )
    predictions_payload = context.artifacts.get(predictions_artifact_id)
    if not isinstance(predictions_payload, Mapping):
        raise TypeError(f"{predictions_artifact_id!r} is not an ml.predictions payload.")

    strategy_id = str(params.get("strategy_id") or params.get("strategy") or "least_confidence")
    k = max(1, int(params.get("k", 200)))
    seed = int(params.get("seed", session.get("seed", 42)))
    registry = actions.get_strategy_registry(context)
    result, ranked_records = acquire_query_batch_streaming(
        context=context,
        registry=registry,
        session=session,
        predictions_payload=predictions_payload,
        predictions_artifact_id=predictions_artifact_id,
        strategy_id=strategy_id,
        k=k,
        seed=seed,
        params=params,
        cancel_token=cancel_token,
    )
    row_ids = [str(record["row_id"]) for record in ranked_records]
    batch_payload = acquisition.create_batch_payload(
        dataset_id=dataset_id,
        session=session,
        strategy_id=strategy_id,
        records=ranked_records,
        params=params,
        predictions_artifact_id=predictions_artifact_id,
        kind="query",
        rank_stats=result.stats,
    )
    batch_artifact_id = context.artifacts.put(
        al_state.ARTIFACT_BATCH,
        batch_payload,
        dataset_id=dataset_id,
        row_ids=row_ids,
        params={
            "session_id": session["session_id"],
            "strategy_id": strategy_id,
            "predictions_artifact_id": predictions_artifact_id,
            "k": k,
            "excluded_count": result.stats.get("excluded_count"),
            "streaming": True,
        },
    )
    updated = al_state.with_last_batch(
        session,
        batch_artifact_id=batch_artifact_id,
        strategy_id=strategy_id,
        row_ids=row_ids,
        predictions_artifact_id=predictions_artifact_id,
        kind="query",
    )
    new_session_artifact_id = actions.put_session(
        context,
        updated,
        previous_artifact_id=session_artifact_id,
    )
    if bool(params.get("make_selection", True)):
        selection_result = apply_ranked_review_selection(
            context,
            source_dataset_id=session_source_dataset_id(updated),
            pool_dataset_id=dataset_id,
            row_ids=row_ids,
            session_artifact_id=new_session_artifact_id,
            batch_artifact_id=batch_artifact_id,
            strategy_id=strategy_id,
            origin=f"{ORIGIN}.query_batch",
        )
    else:
        selection_result = disabled_review_selection(row_ids)

    payload = {
        "session_artifact_id": new_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": updated["session_id"],
        "dataset_id": dataset_id,
        "predictions_artifact_id": predictions_artifact_id,
        "batch_artifact_id": batch_artifact_id,
        "strategy_id": strategy_id,
        "count": len(row_ids),
        "rank_stats": result.stats,
        "selection_applied": selection_result.applied,
        "selection_dataset_id": selection_result.dataset_id,
        "focused_row_id": selection_result.focused_row_id,
        "selection_reason": selection_result.reason,
    }
    actions.publish(context, "al.query_batch.created", payload)
    return {
        "ok": True,
        **payload,
        "row_ids": row_ids,
        "counts": al_state.counts(updated),
    }

def score_pool_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    request = actions.coerce_request(request)
    params = dict(request.params or {})
    session_artifact_id = str(params.get("session_artifact_id") or "").strip()
    if not session_artifact_id:
        raise ValueError("score_pool requires session_artifact_id.")

    session = al_state.coerce_session(context.artifacts.get(session_artifact_id))
    dataset_id = acquisition.session_pool_dataset_id(session)
    if not dataset_id:
        raise ValueError("score_pool could not determine the AL session pool dataset.")
    predictions_artifact_id = str(
        params.get("predictions_artifact_id")
        or request.artifact_id
        or al_state.latest_reference(session, "predictions_artifact_id")
        or ""
    ).strip()
    if not predictions_artifact_id:
        raise ValueError(
            "score_pool requires predictions_artifact_id or a session with latest pool predictions."
        )
    predictions_payload = context.artifacts.get(predictions_artifact_id)
    if not isinstance(predictions_payload, Mapping):
        raise TypeError(f"{predictions_artifact_id!r} is not an ml.predictions payload.")

    registry = actions.get_strategy_registry(context)
    strategy_ids = _strategy_ids(registry, params)
    seed = int(params.get("seed", session.get("seed", 42)))
    rows_iter, score_state = iter_pool_score_rows(
        context=context,
        registry=registry,
        session=session,
        predictions_payload=predictions_payload,
        strategy_ids=strategy_ids,
        seed=seed,
        params=params,
        cancel_token=cancel_token,
    )

    preview_limit = max(0, int(params.get("score_preview_limit") or 1000))
    preview: list[Dict[str, Any]] = []
    score_ref: Optional[ScoreTableRef] = None
    writer: Optional[ScoreTableWriter] = None
    try:
        writer = ScoreTableWriter(
            root=score_output_root(context, params),
            session_id=str(session.get("session_id") or "session"),
            storage_format=str(params.get("score_storage_format") or "auto"),
        )
        for rows in rows_iter:
            check_cancelled(cancel_token)
            writer.write_rows(rows)
            if len(preview) < preview_limit:
                preview.extend(rows[: preview_limit - len(preview)])
        score_ref = writer.finalize()
        writer = None

        stats_by_strategy = dict(score_state["stats_by_strategy"])
        if score_ref.row_count == 0 and not any(
            value.get("error") for value in stats_by_strategy.values()
        ):
            raise ValueError("No query-strategy scores were calculated.")

        score_dataset_id = _register_score_dataset(
            context,
            dataset_id=dataset_id,
            predictions_artifact_id=predictions_artifact_id,
            session=session,
            score_ref=score_ref,
            params=params,
        )
        by_strategy_preview: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
        for record in preview:
            by_strategy_preview[str(record.get("strategy_id") or "")].append(record)
        preview_row_ids = list(
            dict.fromkeys(
                str(record.get("row_id"))
                for record in preview
                if record.get("row_id") not in (None, "")
            )
        )
        score_payload = {
            "schema_version": 2,
            "kind": "strategy_scores",
            "dataset_id": dataset_id,
            "score_dataset_id": score_dataset_id,
            "session_id": session.get("session_id"),
            "session_artifact_id": session_artifact_id,
            "predictions_artifact_id": predictions_artifact_id,
            "strategy_ids": list(stats_by_strategy),
            "records": preview if score_ref.row_count <= len(preview) else [],
            "preview": preview,
            "records_inline_complete": score_ref.row_count <= len(preview),
            "by_strategy": dict(by_strategy_preview),
            "stats_by_strategy": stats_by_strategy,
            "eligible_pool_count": int(score_state["eligible_count"]),
            "excluded_count": int(score_state["excluded_count"]),
            "prediction_record_count": int(score_state["prediction_record_count"]),
            "scored_record_count": int(score_ref.row_count),
            "score_ref": score_ref.to_dict(),
            "seed": seed,
        }
        score_artifact_id = context.artifacts.put(
            al_state.ARTIFACT_STRATEGY_SCORES,
            score_payload,
            dataset_id=dataset_id,
            row_ids=preview_row_ids,
            params={
                "session_id": session.get("session_id"),
                "predictions_artifact_id": predictions_artifact_id,
                "strategy_ids": list(stats_by_strategy),
                "eligible_pool_count": score_state["eligible_count"],
                "streaming": True,
            },
        )

        updated = al_state.coerce_session(session)
        latest = dict(updated.get("latest") or {})
        latest["strategy_scores_artifact_id"] = str(score_artifact_id)
        latest["predictions_artifact_id"] = predictions_artifact_id
        if score_dataset_id:
            latest["strategy_scores_dataset_id"] = score_dataset_id
        updated["latest"] = latest
        updated.setdefault("history", []).append(
            {
                "event": "strategy_scores_calculated",
                "strategy_scores_artifact_id": str(score_artifact_id),
                "strategy_scores_dataset_id": score_dataset_id,
                "predictions_artifact_id": predictions_artifact_id,
                "strategy_ids": list(stats_by_strategy),
                "eligible_pool_count": score_state["eligible_count"],
                "scored_record_count": score_ref.row_count,
                "timestamp": al_state.now(),
            }
        )
        new_session_artifact_id = actions.put_session(
            context,
            updated,
            previous_artifact_id=session_artifact_id,
        )
        payload = {
            "session_artifact_id": new_session_artifact_id,
            "previous_session_artifact_id": session_artifact_id,
            "session_id": updated["session_id"],
            "dataset_id": dataset_id,
            "predictions_artifact_id": predictions_artifact_id,
            "strategy_scores_artifact_id": str(score_artifact_id),
            "strategy_scores_dataset_id": score_dataset_id,
            "strategy_ids": list(stats_by_strategy),
            "eligible_pool_count": score_state["eligible_count"],
            "scored_record_count": score_ref.row_count,
            "stats_by_strategy": stats_by_strategy,
            "score_ref": score_ref.to_dict(),
        }
        actions.publish(context, "al.strategy_scores.calculated", payload)
        return {"ok": True, **payload}
    except Exception:
        if writer is not None:
            writer.abort()
        delete_score_ref(score_ref)
        raise

def materialize_training_set_action(
    context: Any,
    request: Any,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    """Prepare the labelled pool rows without registering another dataset."""

    from . import core_ml_bridge as bridge

    request = actions.coerce_request(request)
    params = dict(request.params or {})
    session_artifact_id = str(params.get("session_artifact_id") or "").strip()
    if not session_artifact_id:
        raise ValueError("materialize_training_set requires session_artifact_id.")

    session = al_state.coerce_session(
        context.artifacts.get(session_artifact_id)
    )
    clear_queued = getattr(al_state, "clear_queued_rows", None)
    if callable(clear_queued):
        session_without_queue = clear_queued(
            session,
            reason="training_started",
        )
        if session_without_queue != session:
            session_artifact_id = actions.put_session(
                context,
                session_without_queue,
                previous_artifact_id=session_artifact_id,
            )
            session = session_without_queue
            actions.publish(
                context,
                "al.query_batch.invalidated",
                {
                    "session_artifact_id": session_artifact_id,
                    "session_id": session.get("session_id"),
                    "reason": "training_started",
                },
            )

    dataset_id = acquisition.session_pool_dataset_id(session)
    if not dataset_id:
        raise ValueError("Could not determine source/pool dataset_id.")

    labelled_items = list(al_state.labelled_training_items(session) or [])
    if not labelled_items:
        raise ValueError("No verified labels are available for training.")

    round_index = int(session.get("round", 0)) + 1
    target_column = str(
        params.get("target_column")
        or session.get("target_column")
        or "al_label"
    )
    task_type = al_state.parse_task_type(
        params.get("task_type")
        or session.get("task_type")
        or session.get("problem_type")
    )

    profile_info = _profile_info(bridge, context, params)
    source = context.datasets.get_source(dataset_id)
    source_columns = [str(column) for column in source.columns()]
    recipe_params = _recipe_params(
        bridge,
        profile_info,
        params,
        source_columns,
    )
    image_column = _image_column(
        bridge,
        context,
        dataset_id,
        params,
        recipe_params,
        source_columns,
    )
    required_columns = _training_columns(
        bridge,
        params=params,
        recipe_params=recipe_params,
        source_columns=source_columns,
        image_column=image_column,
    )

    mappings = dict(
        actions.dataset_mappings(context, dataset_id) or {}
    )
    id_column = str(
        params.get("record_id_column")
        or mappings.get("record_id")
        or ""
    ).strip()
    if _uses_index(id_column):
        raise ValueError(
            "Active Learning training requires a physical record_id column "
            "mapping."
        )
    if id_column not in source_columns:
        raise KeyError(
            f"Mapped record_id column {id_column!r} is not present in "
            f"{dataset_id!r}."
        )

    label_map = {
        str(item.get("row_id")): item.get("label")
        for item in labelled_items
        if isinstance(item, Mapping)
        and item.get("row_id") not in (None, "")
    }
    row_ids = list(label_map)
    if not row_ids:
        raise ValueError(
            "No labelled row identifiers are available for training."
        )

    overlay = attach_session_labels(
        context,
        dataset_id=dataset_id,
        record_id_column=id_column,
        target_column=target_column,
        labelled_items=labelled_items,
        session_id=str(session.get("session_id") or ""),
        session_artifact_id=session_artifact_id,
        round_index=round_index,
        origin=f"{ORIGIN}.materialize_training_set",
    )
    source = context.datasets.get_source(dataset_id)
    available_columns = set(source.columns())
    missing_columns = [
        column
        for column in required_columns
        if column not in available_columns
    ]
    if missing_columns:
        raise ValueError(
            "Active Learning pool dataset is missing recipe columns: "
            + ", ".join(missing_columns)
        )
    if target_column not in available_columns:
        raise RuntimeError(
            f"Label overlay did not expose target column {target_column!r}."
        )

    lookup_batch_size = max(
        1,
        int(params.get("training_lookup_batch_size") or 4096),
    )
    verified_count = 0
    for start_index in range(0, len(row_ids), lookup_batch_size):
        check_cancelled(cancel_token)
        batch_ids = row_ids[
            start_index : start_index + lookup_batch_size
        ]
        frame = source.get_rows_by_ids(
            batch_ids,
            id_column=id_column,
            columns=[id_column, target_column],
        )
        if frame is None or frame.empty:
            raise KeyError(
                "The pool dataset did not return labelled training rows: "
                f"{batch_ids[:10]!r}"
            )
        returned = frame[id_column].map(str)
        if returned.duplicated().any():
            duplicates = returned[
                returned.duplicated(keep=False)
            ].unique().tolist()
            raise ValueError(
                "The pool dataset returned duplicate training record IDs: "
                f"{duplicates[:10]!r}"
            )
        returned_ids = set(returned)
        missing_ids = [
            row_id for row_id in batch_ids if row_id not in returned_ids
        ]
        if missing_ids:
            raise KeyError(
                "Labelled rows are missing from the pool dataset: "
                f"{missing_ids[:10]!r}"
            )
        missing_labels = frame[target_column].isna()
        if bool(missing_labels.any()):
            failed = frame.loc[missing_labels, id_column].head(10).tolist()
            raise ValueError(
                "The Active Learning label overlay is missing labels for "
                f"training rows: {failed!r}"
            )
        verified_count += len(frame)

    if verified_count != len(row_ids):
        raise RuntimeError(
            "Active Learning training membership verification failed: "
            f"expected {len(row_ids)}, verified {verified_count}."
        )

    class_labels = _class_labels(
        bridge,
        task_type=task_type,
        params=params,
        session=session,
        labelled_items=labelled_items,
    )
    preview_limit = max(
        0,
        int(params.get("training_artifact_preview_limit") or 1000),
    )
    labels_preview = labelled_items[:preview_limit]
    row_ids_preview = row_ids[:preview_limit]
    payload = {
        "schema_version": 5,
        "session_id": session["session_id"],
        "session_artifact_id": session_artifact_id,
        "source_dataset_id": dataset_id,
        "pool_dataset_id": dataset_id,
        "training_dataset_id": dataset_id,
        "training_dataset_reused": True,
        "training_row_ids": (
            row_ids if len(row_ids) <= preview_limit else row_ids_preview
        ),
        "training_row_ids_inline_complete": len(row_ids) <= preview_limit,
        "validation_dataset_id": str(
            params.get("validation_dataset_id")
            or session.get("validation_dataset_id")
            or ""
        ),
        "test_dataset_id": str(
            params.get("test_dataset_id")
            or session.get("test_dataset_id")
            or ""
        ),
        "recipe_id": str(
            profile_info.get("recipe_id")
            or session.get("recipe_id")
            or ""
        ),
        "recipe_profile_id": str(
            profile_info.get("recipe_profile_id")
            or session.get("recipe_profile_id")
            or ""
        ),
        "recipe_profile_name": str(
            profile_info.get("recipe_profile_name")
            or session.get("recipe_profile_name")
            or ""
        ),
        "target_column": target_column,
        "task_type": task_type,
        "problem_type": task_type,
        "label_profile": dict(session.get("label_profile") or {}),
        "record_id_column": id_column,
        "image_column": image_column,
        "class_labels": class_labels,
        "classes": class_labels,
        "round": round_index,
        "seed": int(params.get("seed", session.get("seed", 42))),
        "row_count": len(row_ids),
        "row_ids": (
            row_ids if len(row_ids) <= preview_limit else row_ids_preview
        ),
        "row_ids_inline_complete": len(row_ids) <= preview_limit,
        "labels": (
            labelled_items
            if len(labelled_items) <= preview_limit
            else labels_preview
        ),
        "labels_inline_complete": len(labelled_items) <= preview_limit,
        "labels_preview": labels_preview,
        "label_overlay": overlay,
        "training_columns": list(source.columns()),
        "required_training_columns": required_columns,
        "label_counts": al_state.label_counts(session),
        "counts": al_state.counts(session),
        "session_contract": dict(
            params.get("session_contract")
            or session.get("contract")
            or {}
        ),
    }
    training_artifact_id = context.artifacts.put(
        al_state.ARTIFACT_TRAINING_SET,
        payload,
        dataset_id=dataset_id,
        row_ids=row_ids_preview,
        row_count=len(row_ids),
        params={
            "session_id": session["session_id"],
            "round": round_index,
            "target_column": target_column,
            "row_count": len(row_ids),
            "training_dataset_reused": True,
        },
    )
    event_payload = {
        "dataset_id": dataset_id,
        "training_dataset_id": dataset_id,
        "training_artifact_id": training_artifact_id,
        "session_id": session["session_id"],
        "session_artifact_id": session_artifact_id,
        "round": round_index,
        "row_count": len(row_ids),
        "target_column": target_column,
        "label_overlay": overlay,
        "origin": f"{ORIGIN}.materialize_training_set",
    }
    actions.publish(
        context,
        "al.training_set.prepared",
        event_payload,
    )
    return {
        "ok": True,
        "session": session,
        "session_artifact_id": session_artifact_id,
        "source_dataset_id": dataset_id,
        "training_dataset_id": dataset_id,
        "training_dataset_reused": True,
        "training_artifact_id": training_artifact_id,
        "training_row_ids": row_ids,
        "target_column": target_column,
        "task_type": task_type,
        "problem_type": task_type,
        "label_profile": dict(session.get("label_profile") or {}),
        "record_id_column": id_column,
        "image_column": image_column,
        "class_labels": class_labels,
        "labelled_count": len(labelled_items),
        "recipe_id": str(
            profile_info.get("recipe_id")
            or session.get("recipe_id")
            or ""
        ),
        "recipe_profile_id": str(
            profile_info.get("recipe_profile_id")
            or session.get("recipe_profile_id")
            or ""
        ),
        "recipe_profile_name": str(
            profile_info.get("recipe_profile_name")
            or session.get("recipe_profile_name")
            or ""
        ),
        "round": round_index,
        "label_overlay": overlay,
    }

def _register_score_dataset(
    context: Any,
    *,
    dataset_id: str,
    predictions_artifact_id: str,
    session: Mapping[str, Any],
    score_ref: ScoreTableRef,
    params: Mapping[str, Any],
) -> Optional[str]:
    if not score_ref.parquet_parts:
        return None
    register = getattr(context.datasets, "register_parquet", None)
    if not callable(register):
        return None
    requested = str(params.get("score_dataset_id") or "").strip()
    if requested:
        score_dataset_id = requested
    else:
        unique_id = getattr(actions, "unique_dataset_id", None)
        base = f"{dataset_id}__al_scores"
        score_dataset_id = unique_id(base) if callable(unique_id) else f"{base}_{uuid.uuid4().hex[:8]}"
    register(
        score_dataset_id,
        score_ref.parquet_parts,
        name=str(params.get("score_dataset_name") or f"AL strategy scores: {dataset_id}"),
        source_kind="ml.active_learning_scores",
        origin="core.active_learning.score_pool",
        source_dataset_id=dataset_id,
        predictions_artifact_id=predictions_artifact_id,
        al_session_id=session.get("session_id"),
        row_count=score_ref.row_count,
        columns=list(score_ref.columns),
        column_mappings={"record_id": "score_id"},
        source_record_id_column="row_id",
    )
    return score_dataset_id

def _profile_info(bridge: Any, context: Any, params: Mapping[str, Any]) -> Dict[str, Any]:
    resolver = getattr(bridge, "resolve_recipe_profile_info", None)
    return dict(resolver(context, params) or {}) if callable(resolver) else {}

def _recipe_params(
    bridge: Any,
    profile_info: Mapping[str, Any],
    params: Mapping[str, Any],
    source_columns: Sequence[str],
) -> Dict[str, Any]:
    merger = getattr(bridge, "merged_profile_recipe_params", None)
    raw = merger(dict(profile_info.get("profile") or {}), params) if callable(merger) else dict(params.get("recipe_params") or {})
    sanitizer = getattr(bridge, "sanitize_recipe_params_for_al_training", None)
    return (
        dict(sanitizer(raw, available_columns=source_columns) or {})
        if callable(sanitizer)
        else dict(raw or {})
    )

def _image_column(
    bridge: Any,
    context: Any,
    dataset_id: str,
    params: Mapping[str, Any],
    recipe_params: Dict[str, Any],
    source_columns: Sequence[str],
) -> str:
    resolver = getattr(bridge, "resolve_image_column_for_al_training", None)
    image_column = str(
        resolver(
            context,
            dataset_id,
            params,
            recipe_params,
            available_columns=source_columns,
        )
        if callable(resolver)
        else params.get("image_column") or recipe_params.get("image_column") or ""
    ).strip()
    ensure = getattr(bridge, "ensure_image_params", None)
    if image_column and callable(ensure):
        ensure(recipe_params, image_column)
    return image_column

def _training_columns(
    bridge: Any,
    *,
    params: Mapping[str, Any],
    recipe_params: Mapping[str, Any],
    source_columns: Sequence[str],
    image_column: str,
) -> list[str]:
    prediction_check = getattr(bridge, "is_prediction_column", lambda value: False)
    parse_list = getattr(bridge, "parse_string_list", _parse_string_list)
    requested = [
        str(column)
        for column in parse_list(params.get("required_columns") or [])
        if str(column) in source_columns and not prediction_check(column)
    ]
    column_params = getattr(bridge, "column_like_recipe_params", lambda value: {})
    requested.extend(
        str(column)
        for column in dict(column_params(recipe_params) or {}).values()
        if str(column) in source_columns and not prediction_check(column)
    )
    clean_features = getattr(bridge, "clean_feature_columns", _parse_string_list)
    requested.extend(
        str(column)
        for column in clean_features(
            recipe_params.get("feature_columns") or recipe_params.get("input_columns") or []
        )
        if str(column) in source_columns and not prediction_check(column)
    )
    if image_column and image_column in source_columns:
        requested.append(image_column)
    requested = list(dict.fromkeys(requested))
    # Empty means the recipe is using auto feature selection; retain all source
    # columns, but stream them to disk instead of creating one DataFrame.
    return requested or [
        column for column in source_columns if not prediction_check(column)
    ]

def _class_labels(
    bridge: Any,
    *,
    task_type: str,
    params: Mapping[str, Any],
    session: Mapping[str, Any],
    labelled_items: Sequence[Mapping[str, Any]],
) -> list[Any]:
    if task_type == getattr(al_state, "TASK_REGRESSION", "regression"):
        return []
    resolver = getattr(bridge, "resolve_class_labels", None)
    if callable(resolver):
        return list(
            resolver(
                params=params,
                session=session,
                labelled_items=labelled_items,
            )
            or []
        )
    return sorted(
        {
            str(item.get("label"))
            for item in labelled_items
            if item.get("label") not in (None, "")
        }
    )

def _strategy_ids(registry: Any, params: Mapping[str, Any]) -> list[str]:
    raw = params.get("strategy_ids") or params.get("strategies") or "all"
    if isinstance(raw, str):
        if raw.strip().lower() in {"", "all", "*"}:
            return [str(value) for value in registry.ids()]
        return [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]
    if isinstance(raw, Sequence):
        return [str(value).strip() for value in raw if str(value).strip()]
    return [str(value) for value in registry.ids()]

def _parse_string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]

def _uses_index(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        "",
        "use index",
        "use_index",
        "__index__",
        "index",
    }