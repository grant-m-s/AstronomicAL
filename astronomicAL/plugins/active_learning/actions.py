from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from astronomicAL.platform.plugins.specs import ActionRequest

from . import acquisition
from . import state as al_state
from . import strategies as strategy_module

ORIGIN = "core.active_learning"

def start_session_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Create an AL session and optional initial random review batch.

    Unlike the old implementation, this action does not require a core.ml
    recipe.  Recipe/contract metadata may still be attached when provided.
    """

    request = coerce_request(request)
    params = dict(request.params or {})
    dataset_id = resolve_dataset_id(context, request, params)
    if not dataset_id:
        raise ValueError("start_session requires dataset_id or an active dataset.")

    seed = int(params.get("seed", 42))
    initial_k = max(0, int(params.get("initial_k", 20)))
    target_column = str(params.get("target_column") or params.get("label_column") or "al_label")
    recipe_profile_id = str(params.get("recipe_profile_id") or params.get("recipe_profile_artifact_id") or "").strip()
    recipe_profile_name = str(params.get("recipe_profile_name") or "").strip()
    label_profile = dict(params.get("label_profile") or {})
    requested_task_type = al_state.parse_task_type(params.get("task_type") or params.get("problem_type") or label_profile.get("task_type") or "auto", default=al_state.TASK_UNKNOWN)
    if requested_task_type == al_state.TASK_UNKNOWN and target_column:
        label_profile = infer_label_profile_from_column(context, dataset_id=dataset_id, column=target_column)
        task_type = al_state.parse_task_type(label_profile.get("task_type"), default=al_state.TASK_CLASSIFICATION)
    else:
        task_type = requested_task_type if requested_task_type != al_state.TASK_UNKNOWN else al_state.TASK_CLASSIFICATION
        label_profile.setdefault("task_type", task_type)
        label_profile.setdefault("column", target_column)

    label_options = al_state.parse_label_options(params.get("label_options") or [])
    if task_type == al_state.TASK_REGRESSION:
        label_options = []
    elif not label_options and bool(params.get("infer_labels_from_column", True)) and target_column:
        label_options = infer_label_options_from_column(context, dataset_id=dataset_id, column=target_column)
    make_selection = bool(params.get("make_selection", True))

    selected_row_ids = acquisition.sample_dataset_row_ids(context, dataset_id=dataset_id, k=initial_k, seed=seed)
    session = al_state.create_session(
        dataset_id=dataset_id,
        pool_dataset_id=str(params.get("pool_dataset_id") or dataset_id),
        validation_dataset_id=str(params.get("validation_dataset_id") or ""),
        test_dataset_id=str(params.get("test_dataset_id") or ""),
        recipe_id=str(params.get("recipe_id") or ""),
        recipe_profile_id=recipe_profile_id,
        recipe_profile_name=recipe_profile_name,
        al_protocol=str(params.get("al_protocol") or "review"),
        label_options=label_options,
        seed=seed,
        target_column=target_column,
        task_type=task_type,
        label_profile=label_profile,
        contract=params.get("session_contract") or params.get("contract") or {},
    )

    records = acquisition.build_initial_records(selected_row_ids)
    batch_payload = acquisition.create_batch_payload(
        dataset_id=dataset_id,
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
            "seed": seed,
        },
    )
    batch_artifact_id = context.artifacts.put(
        al_state.ARTIFACT_BATCH,
        batch_payload,
        dataset_id=dataset_id,
        row_ids=selected_row_ids,
        params={"session_id": session["session_id"], "strategy_id": "initial_random", "seed": seed, "k": initial_k},
    )
    session = al_state.with_last_batch(
        session,
        batch_artifact_id=batch_artifact_id,
        strategy_id="initial_random",
        row_ids=selected_row_ids,
        kind="initial_random",
    )
    session_artifact_id = put_session(context, session)

    if make_selection and selected_row_ids:
        acquisition.set_ranked_selection(
            context,
            dataset_id=dataset_id,
            row_ids=selected_row_ids,
            session_artifact_id=session_artifact_id,
            batch_artifact_id=batch_artifact_id,
            strategy_id="initial_random",
            origin=f"{ORIGIN}.start_session",
        )

    publish(
        context,
        "al.session.created",
        {
            "session_artifact_id": session_artifact_id,
            "session_id": session["session_id"],
            "dataset_id": dataset_id,
            "initial_count": len(selected_row_ids),
            "task_type": task_type,
            "label_profile": label_profile,
        },
    )
    return {
        "ok": True,
        "session_artifact_id": session_artifact_id,
        "session_id": session["session_id"],
        "dataset_id": dataset_id,
        "batch_artifact_id": batch_artifact_id,
        "initial_row_ids": selected_row_ids,
        "counts": al_state.counts(session),
        "task_type": task_type,
        "label_profile": label_profile,
    }

def query_batch_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    request = coerce_request(request)
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
        raise ValueError("query_batch requires predictions_artifact_id or a session with latest predictions.")

    predictions_payload = context.artifacts.get(predictions_artifact_id)
    if not isinstance(predictions_payload, Mapping):
        raise TypeError(f"{predictions_artifact_id!r} is not an ml.predictions payload.")

    strategy_id = str(params.get("strategy_id") or params.get("strategy") or "least_confidence")
    k = max(1, int(params.get("k", 200)))
    seed = int(params.get("seed", session.get("seed", 42)))
    make_selection = bool(params.get("make_selection", True))
    registry = get_strategy_registry(context)

    result, ranked_records = acquisition.acquire_query_batch(
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
    new_session_artifact_id = put_session(context, updated, previous_artifact_id=session_artifact_id)

    if make_selection and row_ids:
        acquisition.set_ranked_selection(
            context,
            dataset_id=dataset_id,
            row_ids=row_ids,
            session_artifact_id=new_session_artifact_id,
            batch_artifact_id=batch_artifact_id,
            strategy_id=strategy_id,
            origin=f"{ORIGIN}.query_batch",
        )

    publish(
        context,
        "al.query_batch.created",
        {
            "session_artifact_id": new_session_artifact_id,
            "previous_session_artifact_id": session_artifact_id,
            "session_id": updated["session_id"],
            "dataset_id": dataset_id,
            "predictions_artifact_id": predictions_artifact_id,
            "batch_artifact_id": batch_artifact_id,
            "strategy_id": strategy_id,
            "count": len(row_ids),
            "rank_stats": result.stats,
        },
    )
    return {
        "ok": True,
        "session_artifact_id": new_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": updated["session_id"],
        "dataset_id": dataset_id,
        "predictions_artifact_id": predictions_artifact_id,
        "batch_artifact_id": batch_artifact_id,
        "row_ids": row_ids,
        "count": len(row_ids),
        "strategy_id": strategy_id,
        "rank_stats": result.stats,
        "counts": al_state.counts(updated),
    }

def record_label_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    request = coerce_request(request)
    params = dict(request.params or {})
    session_artifact_id = str(params.get("session_artifact_id") or "").strip()
    if not session_artifact_id:
        raise ValueError("record_label requires session_artifact_id.")

    label = params.get("label")
    if label in (None, ""):
        raise ValueError("record_label requires label.")

    session = al_state.coerce_session(context.artifacts.get(session_artifact_id))
    row_id = str(params.get("row_id") or "").strip()
    if not row_id:
        _, focused = focused_row_ref(context)
        row_id = str(focused or "").strip()
    if not row_id:
        raise ValueError("record_label requires row_id or a focused row.")

    updated = al_state.record_label(
        session,
        row_id=row_id,
        label=label,
        source=str(params.get("source") or "manual"),
    )
    new_session_artifact_id = put_session(context, updated, previous_artifact_id=session_artifact_id)

    batch_row_ids = [str(value) for value in ((updated.get("last_batch") or {}).get("row_ids") or []) if str(value)]
    remaining_row_ids = remaining_review_row_ids(updated, row_ids=batch_row_ids) if batch_row_ids else []
    next_row_id = next_review_row_id(updated, row_ids=batch_row_ids, after_row_id=row_id) if batch_row_ids else ""
    focus_updated = False
    if next_row_id and bool(params.get("update_focus", True)):
        focus_updated = acquisition.set_focus_row(
            context,
            dataset_id=str(updated.get("pool_dataset_id") or updated.get("dataset_id") or ""),
            row_id=next_row_id,
            origin=f"{ORIGIN}.record_label",
            metadata={"session_artifact_id": new_session_artifact_id, "reason": "labelled_next_unverified"},
        )

    recorded_entry = dict((updated.get("labels") or {}).get(str(row_id)) or {})
    payload = {
        "session_artifact_id": new_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": updated["session_id"],
        "dataset_id": updated["dataset_id"],
        "row_id": row_id,
        "label": recorded_entry.get("label", al_state.normalise_label(label)),
        "display_label": recorded_entry.get("display_label", al_state.display_label(label)),
        "remaining_row_ids": remaining_row_ids,
        "remaining_count": len(remaining_row_ids),
        "next_row_id": next_row_id,
        "focus_updated": focus_updated,
        "counts": al_state.counts(updated),
    }
    publish(context, "al.label.recorded", payload)
    return {"ok": True, **payload}

def bulk_label_next_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Label the next N unlabelled review rows from the source label column.

    The selected label column is stored on the session as ``target_column``.
    Bulk review uses that column value per row, so a batch containing cat/dog/star
    rows records cat/dog/star respectively instead of repeating the current UI
    dropdown value.  Rows with blank labels, the special Unsure value, or labels
    outside the session label set are skipped and the scan continues until N rows
    are recorded or the current batch is exhausted.
    """

    request = coerce_request(request)
    params = dict(request.params or {})
    session_artifact_id = str(params.get("session_artifact_id") or "").strip()
    if not session_artifact_id:
        raise ValueError("bulk_label_next requires session_artifact_id.")

    n = max(1, int(params.get("n") or params.get("count") or 1))
    session = al_state.coerce_session(context.artifacts.get(session_artifact_id))
    dataset_id = str(session.get("pool_dataset_id") or session.get("dataset_id") or "").strip()
    label_column = str(params.get("label_column") or session.get("target_column") or "").strip()
    if not dataset_id:
        raise ValueError("bulk_label_next could not determine the session dataset.")
    if not label_column:
        raise ValueError("bulk_label_next requires a label_column or a session target_column.")

    row_ids = [str(row_id) for row_id in ((session.get("last_batch") or {}).get("row_ids") or []) if str(row_id)]
    if not row_ids:
        raise ValueError("bulk_label_next requires a session with a latest review/query batch.")

    start_row_id = str(params.get("row_id") or params.get("start_row_id") or "").strip()
    if not start_row_id:
        _, focused = focused_row_ref(context)
        start_row_id = str(focused or "").strip()

    candidate_row_ids = next_unlabelled_batch_row_ids(session, row_ids=row_ids, start_row_id=start_row_id, n=len(row_ids))
    labels_by_row_id = dataset_label_values_by_row_id(context, dataset_id=dataset_id, row_ids=candidate_row_ids, label_column=label_column)
    allowed_labels = set() if al_state.is_regression_task(session) else {al_state.normalise_label(value) for value in (session.get("label_options") or []) if value not in (None, "")}

    selected_row_ids: List[str] = []
    selected_labels: Dict[str, str] = {}
    skipped: List[Dict[str, str]] = []
    for row_id in candidate_row_ids:
        raw_label = labels_by_row_id.get(str(row_id))
        label = al_state.normalise_label(raw_label)
        if not label:
            skipped.append({"row_id": str(row_id), "reason": "blank_label"})
            continue
        if label == al_state.UNSURE_LABEL:
            skipped.append({"row_id": str(row_id), "reason": "unsure_label"})
            continue
        if allowed_labels and label not in allowed_labels:
            skipped.append({"row_id": str(row_id), "reason": "label_not_in_session", "label": label})
            continue
        selected_row_ids.append(str(row_id))
        selected_labels[str(row_id)] = label
        if len(selected_row_ids) >= n:
            break

    if not selected_row_ids:
        return {
            "ok": True,
            "session_artifact_id": session_artifact_id,
            "previous_session_artifact_id": session_artifact_id,
            "session_id": session["session_id"],
            "dataset_id": session["dataset_id"],
            "label_column": label_column,
            "row_ids": [],
            "labels_by_row_id": {},
            "skipped": skipped,
            "count": 0,
            "requested_count": n,
            "counts": al_state.counts(session),
        }

    updated = session
    for row_id in selected_row_ids:
        updated = al_state.record_label(
            updated,
            row_id=row_id,
            label=selected_labels[row_id],
            source=str(params.get("source") or "bulk_column"),
        )

    new_session_artifact_id = put_session(context, updated, previous_artifact_id=session_artifact_id)

    remaining_row_ids = remaining_review_row_ids(updated, row_ids=row_ids)
    next_row_id = next_review_row_id(updated, row_ids=row_ids, after_row_id=selected_row_ids[-1])
    selection_updated = False
    if bool(params.get("update_selection", True)):
        last_batch = dict(updated.get("last_batch") or session.get("last_batch") or {})
        acquisition.set_ranked_selection(
            context,
            dataset_id=dataset_id,
            row_ids=remaining_row_ids,
            session_artifact_id=new_session_artifact_id,
            batch_artifact_id=str(last_batch.get("batch_artifact_id") or ""),
            strategy_id=str(last_batch.get("strategy_id") or "bulk_review"),
            origin=f"{ORIGIN}.bulk_label_next",
            update_focus_policy="first",
        )
        selection_updated = True
        if next_row_id:
            acquisition.set_focus_row(
                context,
                dataset_id=dataset_id,
                row_id=next_row_id,
                origin=f"{ORIGIN}.bulk_label_next",
                metadata={"session_artifact_id": new_session_artifact_id, "reason": "bulk_labelled_next_unverified"},
            )

    recorded_entry = dict((updated.get("labels") or {}).get(str(row_id)) or {})
    payload = {
        "session_artifact_id": new_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": updated["session_id"],
        "dataset_id": updated["dataset_id"],
        "label_column": label_column,
        "row_ids": selected_row_ids,
        "labels_by_row_id": dict(selected_labels),
        "skipped": skipped,
        "count": len(selected_row_ids),
        "requested_count": n,
        "remaining_row_ids": remaining_row_ids,
        "remaining_count": len(remaining_row_ids),
        "next_row_id": next_row_id,
        "selection_updated": selection_updated,
        "counts": al_state.counts(updated),
    }
    publish(context, "al.labels.bulk_recorded", payload)
    return {"ok": True, **payload}


def score_pool_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Score every currently eligible pool row with one or more query strategies.

    This does not create a review queue.  It materialises an exploratory score
    artifact for XY diagnostics so users can inspect informativeness over the
    whole search space before selecting a top-N query batch.
    """

    request = coerce_request(request)
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
        raise ValueError("score_pool requires predictions_artifact_id or a session with latest pool predictions.")

    predictions_payload = context.artifacts.get(predictions_artifact_id)
    if not isinstance(predictions_payload, Mapping):
        raise TypeError(f"{predictions_artifact_id!r} is not an ml.predictions payload.")

    registry = get_strategy_registry(context)
    raw_strategy_ids = params.get("strategy_ids") or params.get("strategies") or "all"
    if isinstance(raw_strategy_ids, str):
        if raw_strategy_ids.strip().lower() in {"", "all", "*"}:
            strategy_ids = registry.ids()
        else:
            strategy_ids = [part.strip() for part in raw_strategy_ids.replace("\n", ",").split(",") if part.strip()]
    elif isinstance(raw_strategy_ids, Sequence):
        strategy_ids = [str(value).strip() for value in raw_strategy_ids if str(value).strip()]
    else:
        strategy_ids = registry.ids()
    if not strategy_ids:
        raise ValueError("score_pool did not receive any strategy ids to calculate.")

    seed = int(params.get("seed", session.get("seed", 42)))
    exclude = acquisition.session_query_exclude_row_ids(session, params)
    pool = acquisition.build_query_pool(
        context=context,
        dataset_id=dataset_id,
        session=session,
        predictions_payload=predictions_payload,
        exclude_row_ids=exclude,
    )
    if not pool.records:
        raise ValueError("Prediction artifact contains no records to score.")
    eligible_count = sum(
        1
        for raw in pool.records
        if isinstance(raw, Mapping)
        and str(raw.get("row_id") or raw.get("id") or "").strip()
        and str(raw.get("row_id") or raw.get("id") or "").strip() not in pool.excluded_row_ids
    )
    if eligible_count <= 0:
        raise ValueError("No eligible pool rows remain to score.")

    all_records: List[Dict[str, Any]] = []
    by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    stats_by_strategy: Dict[str, Dict[str, Any]] = {}
    row_ids: List[str] = []
    for strategy_id in strategy_ids:
        strategy_id = str(strategy_id or "").strip()
        if not strategy_id:
            continue
        try:
            result = registry.acquire(
                pool,
                strategy_id=strategy_id,
                k=eligible_count,
                seed=seed,
                params=dict(params.get("strategy_params") or params),
                cancel_token=cancel_token,
            )
        except Exception as exc:
            try:
                info = registry.get(strategy_id).info()
                title = str(info.title or strategy_id)
            except Exception:
                title = strategy_id
            by_strategy[strategy_id] = []
            stats_by_strategy[strategy_id] = {
                "strategy_id": strategy_id,
                "strategy_title": title,
                "eligible_pool_count": eligible_count,
                "scored_pool_count": 0,
                "selected_count": 0,
                "scored_for_visualisation": True,
                "error": str(exc),
            }
            continue

        rows: List[Dict[str, Any]] = []
        for rank, candidate in enumerate(result.candidates, start=1):
            row_id = str(candidate.row_id or "").strip()
            if not row_id:
                continue
            metadata = dict(candidate.metadata or {})
            record = {
                "row_id": row_id,
                "strategy_id": str(result.strategy_id),
                "strategy_title": result.stats.get("strategy_title") or str(result.strategy_id),
                "score": float(candidate.score),
                "informativeness_score": float(candidate.score),
                "active_learning_score": float(candidate.score),
                "rank": rank,
                "selection_rank": rank,
                "score_source": str(metadata.get("_al_score_source") or "direct_score"),
            }
            rows.append(record)
            all_records.append(record)
            row_ids.append(row_id)
        result.stats["scored_for_visualisation"] = True
        result.stats["eligible_pool_count"] = eligible_count
        result.stats["scored_pool_count"] = len(rows)
        result.stats["selected_count"] = len(rows)
        by_strategy[str(result.strategy_id)] = rows
        stats_by_strategy[str(result.strategy_id)] = dict(result.stats)

    if not all_records and not any((stats.get("error") for stats in stats_by_strategy.values())):
        raise ValueError("No query-strategy scores were calculated.")

    row_ids = list(dict.fromkeys(row_ids))
    score_payload = {
        "schema_version": 1,
        "kind": "strategy_scores",
        "dataset_id": dataset_id,
        "session_id": session.get("session_id"),
        "session_artifact_id": session_artifact_id,
        "predictions_artifact_id": predictions_artifact_id,
        "strategy_ids": list(by_strategy.keys()),
        "records": all_records,
        "by_strategy": by_strategy,
        "stats_by_strategy": stats_by_strategy,
        "eligible_pool_count": eligible_count,
        "excluded_count": len(exclude),
        "seed": seed,
    }
    score_artifact_id = context.artifacts.put(
        al_state.ARTIFACT_STRATEGY_SCORES,
        score_payload,
        dataset_id=dataset_id,
        row_ids=row_ids,
        params={
            "session_id": session.get("session_id"),
            "predictions_artifact_id": predictions_artifact_id,
            "strategy_ids": list(by_strategy.keys()),
            "eligible_pool_count": eligible_count,
        },
    )

    updated = al_state.coerce_session(session)
    latest = dict(updated.get("latest") or {})
    latest["strategy_scores_artifact_id"] = str(score_artifact_id)
    latest["predictions_artifact_id"] = predictions_artifact_id
    updated["latest"] = latest
    updated.setdefault("history", []).append(
        {
            "event": "strategy_scores_calculated",
            "strategy_scores_artifact_id": str(score_artifact_id),
            "predictions_artifact_id": predictions_artifact_id,
            "strategy_ids": list(by_strategy.keys()),
            "eligible_pool_count": eligible_count,
            "scored_record_count": len(all_records),
            "timestamp": al_state.now(),
        }
    )
    new_session_artifact_id = put_session(context, updated, previous_artifact_id=session_artifact_id)
    recorded_entry = dict((updated.get("labels") or {}).get(str(row_id)) or {})
    payload = {
        "session_artifact_id": new_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": updated["session_id"],
        "dataset_id": dataset_id,
        "predictions_artifact_id": predictions_artifact_id,
        "strategy_scores_artifact_id": str(score_artifact_id),
        "strategy_ids": list(by_strategy.keys()),
        "eligible_pool_count": eligible_count,
        "scored_record_count": len(all_records),
        "stats_by_strategy": stats_by_strategy,
    }
    publish(context, "al.strategy_scores.calculated", payload)
    return {"ok": True, **payload}

def materialize_training_set_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    from . import core_ml_bridge

    return core_ml_bridge.materialize_training_set_action(context, request, cancel_token=cancel_token)

def profile_data_contract_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    from . import core_ml_bridge

    return core_ml_bridge.profile_data_contract_action(context, request, cancel_token=cancel_token)

def train_from_session_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    from . import core_ml_bridge

    return core_ml_bridge.train_from_session_action(context, request, cancel_token=cancel_token)

def get_strategy_registry(context: Any):
    services = getattr(context, "services", None)
    if services is not None:
        try:
            registry = services.get("core.active_learning.query_strategy_registry")
            if registry is not None:
                return registry
        except Exception as exc:
            raise RuntimeError("Active-learning query strategy registry service is unavailable.") from exc
    return strategy_module.create_default_strategy_registry()

def put_session(context: Any, session: Mapping[str, Any], *, previous_artifact_id: Optional[str] = None) -> str:
    session_payload = al_state.with_revision(session, previous_session_artifact_id=previous_artifact_id)
    artifact_id = context.artifacts.put(
        al_state.ARTIFACT_SESSION,
        session_payload,
        dataset_id=str(session_payload.get("dataset_id") or "default"),
        row_ids=list(session_payload.get("ignored_row_ids") or []),
        params={
            "session_id": session_payload.get("session_id"),
            "round": session_payload.get("round"),
            "revision": session_payload.get("revision"),
            "previous_session_artifact_id": previous_artifact_id,
        },
    )
    publish(
        context,
        "al.session.saved",
        {
            "session_artifact_id": artifact_id,
            "previous_session_artifact_id": previous_artifact_id,
            "session_id": session_payload.get("session_id"),
            "dataset_id": session_payload.get("dataset_id"),
            "round": session_payload.get("round"),
            "revision": session_payload.get("revision"),
            "counts": al_state.counts(session_payload),
        },
    )
    return str(artifact_id)

def resolve_dataset_id(context: Any, request: ActionRequest, params: Mapping[str, Any]) -> str:
    dataset_id = str(params.get("dataset_id") or request.dataset_id or "").strip()
    if dataset_id:
        return dataset_id
    active_id = getattr(getattr(context, "datasets", None), "active_id", None)
    return str(active_id() if callable(active_id) else "").strip()

def focused_row_ref(context: Any) -> tuple[Optional[str], Optional[str]]:
    selection = getattr(context, "selection", None)
    if selection is None or not hasattr(selection, "get_focus"):
        return None, None
    focus = selection.get_focus()
    if focus is None:
        return None, None
    dataset_id = getattr(focus, "dataset_id", None)
    row_id = getattr(focus, "row_id", None)
    if isinstance(focus, Mapping):
        dataset_id = focus.get("dataset_id", dataset_id)
        row_id = focus.get("row_id", row_id)
    return (str(dataset_id) if dataset_id else None, str(row_id) if row_id else None)

def next_unlabelled_batch_row_ids(
    session_payload: Mapping[str, Any],
    *,
    row_ids: Sequence[Any],
    start_row_id: str = "",
    n: int = 1,
) -> List[str]:
    session = al_state.coerce_session(session_payload)
    ordered = [str(row_id) for row_id in row_ids if str(row_id)]
    if not ordered:
        return []

    start_index = 0
    if start_row_id and start_row_id in ordered:
        start_index = ordered.index(start_row_id)

    labels = {str(row_id) for row_id in (session.get("labels") or {}).keys()}
    ignored = {str(row_id) for row_id in session.get("ignored_row_ids") or []}
    training = {str(row_id) for row_id in session.get("training_row_ids") or []}
    already_done = labels | ignored | training

    selected: List[str] = []
    for row_id in ordered[start_index:]:
        if row_id in already_done:
            continue
        selected.append(row_id)
        if len(selected) >= n:
            break
    return selected

def remaining_review_row_ids(session_payload: Mapping[str, Any], *, row_ids: Sequence[Any]) -> List[str]:
    """Rows from the current review/query batch that are still worth reviewing."""

    session = al_state.coerce_session(session_payload)
    row_states = {str(row_id): str(state) for row_id, state in dict(session.get("row_states") or {}).items()}
    labelled = {str(row_id) for row_id in dict(session.get("labels") or {}).keys()}
    ignored = {str(row_id) for row_id in session.get("ignored_row_ids") or []}
    training = {str(row_id) for row_id in session.get("training_row_ids") or []}
    remove_states = {al_state.ROW_VERIFIED, al_state.ROW_UNSURE, al_state.ROW_TRAINING, al_state.ROW_DEFERRED, al_state.ROW_EXCLUDED}
    out: List[str] = []
    for raw in row_ids:
        row_id = str(raw or "").strip()
        if not row_id:
            continue
        if row_id in labelled or row_id in ignored or row_id in training:
            continue
        if row_states.get(row_id) in remove_states:
            continue
        out.append(row_id)
    return list(dict.fromkeys(out))

def next_review_row_id(session_payload: Mapping[str, Any], *, row_ids: Sequence[Any], after_row_id: Any = "") -> str:
    """Return the next unverified row in current batch order after ``after_row_id``."""

    ordered = [str(row_id) for row_id in row_ids if str(row_id)]
    remaining = remaining_review_row_ids(session_payload, row_ids=ordered)
    if not remaining:
        return ""
    remaining_set = set(remaining)
    after = str(after_row_id or "").strip()
    if after and after in ordered:
        start = ordered.index(after) + 1
        for row_id in ordered[start:] + ordered[:start]:
            if row_id in remaining_set:
                return row_id
    return remaining[0]

def coerce_request(request: Any) -> ActionRequest:
    if isinstance(request, ActionRequest):
        return request
    if isinstance(request, dict):
        return ActionRequest.from_dict(request)
    return ActionRequest(
        dataset_id=getattr(request, "dataset_id", None),
        row_ids=getattr(request, "row_ids", None),
        columns=list(getattr(request, "columns", []) or []),
        params=dict(getattr(request, "params", {}) or {}),
        artifact_id=getattr(request, "artifact_id", None),
        origin=getattr(request, "origin", None),
    )

def publish(context: Any, topic: str, payload: Mapping[str, Any]) -> None:
    events = getattr(context, "events", None)
    publish_fn = getattr(events, "publish", None)
    if callable(publish_fn):
        enriched = dict(payload)
        enriched.setdefault("event", str(topic))
        enriched.setdefault("topic", str(topic))
        publish_fn(topic, enriched)

def call_registered_action(context: Any, action_id: str, request: ActionRequest, *, cancel_token: Any = None) -> Dict[str, Any]:
    manager = getattr(context, "plugins", None)
    if manager is None or not hasattr(manager, "get_action"):
        raise RuntimeError(f"{action_id} requires the plugin manager.")
    try:
        registration = manager.get_action(action_id)
    except Exception as exc:
        raise RuntimeError(f"Required action {action_id!r} is not registered.") from exc
    handler = getattr(registration, "handler", None)
    if not callable(handler):
        raise RuntimeError(f"Registered action {action_id!r} has no callable handler.")
    raw = handler(context, request, cancel_token=cancel_token)
    if isinstance(raw, Mapping):
        return dict(raw)
    return {"ok": True, "result": raw}

def infer_label_profile_from_column(
    context: Any,
    *,
    dataset_id: str,
    column: str,
    max_classes: int = 50,
    sample_rows: int = 10000,
) -> Dict[str, Any]:
    """Infer whether a target column is classification or regression with bounded work.

    Numeric columns with many distinct non-null values are treated as regression.
    Numeric columns with a small number of distinct values remain classification,
    which keeps common 0/1 and 0/1/2 class-label columns working as expected.
    """

    dataset_id = str(dataset_id or "").strip()
    column = str(column or "").strip()
    if not dataset_id or not column:
        return {"task_type": al_state.TASK_CLASSIFICATION, "column": column, "reason": "missing_dataset_or_column"}

    values = bounded_column_values(context, dataset_id=dataset_id, column=column, limit=max(sample_rows, max_classes + 1))
    non_null_values = [value for value in values if value not in (None, "")]
    unique_values: List[Any] = []
    seen = set()
    for value in non_null_values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            unique_values.append(value)
        if len(unique_values) > max_classes:
            break

    numeric_count = 0
    finite_numeric_count = 0
    has_fractional = False
    for value in non_null_values[:sample_rows]:
        try:
            numeric_value = float(str(value).strip())
        except Exception:
            continue
        numeric_count += 1
        if numeric_value == numeric_value and numeric_value not in (float("inf"), float("-inf")):
            finite_numeric_count += 1
            if abs(numeric_value - round(numeric_value)) > 1e-12:
                has_fractional = True

    sample_count = len(non_null_values)
    unique_count = len(unique_values)
    numeric_ratio = (numeric_count / sample_count) if sample_count else 0.0
    is_numeric = bool(sample_count) and numeric_ratio >= 0.95 and finite_numeric_count == numeric_count
    unique_exceeds_class_limit = unique_count > max_classes

    if is_numeric and unique_exceeds_class_limit:
        task_type = al_state.TASK_REGRESSION
        reason = f"numeric target with more than {max_classes} distinct sampled values"
        labels: List[str] = []
    elif is_numeric and has_fractional and unique_count > 2:
        task_type = al_state.TASK_REGRESSION
        reason = "numeric target has fractional sampled values"
        labels = []
    else:
        task_type = al_state.TASK_CLASSIFICATION
        reason = "small/categorical target value set"
        labels = []
        for value in unique_values[:max_classes]:
            text = str(value).strip()
            if text and al_state.normalise_label(text) != al_state.UNSURE_LABEL and text not in labels:
                labels.append(text)

    return {
        "schema_version": 1,
        "column": column,
        "task_type": task_type,
        "problem_type": task_type,
        "is_numeric": is_numeric,
        "numeric_ratio": numeric_ratio,
        "sample_count": sample_count,
        "sample_limit": sample_rows,
        "unique_count_sampled": unique_count,
        "unique_exceeds_class_limit": unique_exceeds_class_limit,
        "class_limit": max_classes,
        "class_labels": labels,
        "label_options": labels,
        "reason": reason,
    }


def bounded_column_values(context: Any, *, dataset_id: str, column: str, limit: int) -> List[Any]:
    dataset_id = str(dataset_id or "").strip()
    column = str(column or "").strip()
    limit = max(1, int(limit or 1))
    datasets = getattr(context, "datasets", None)
    if datasets is None or not dataset_id or not column:
        return []

    source = getattr(datasets, "get_source", lambda *_: None)(dataset_id)
    for owner in (source, datasets):
        if owner is None:
            continue
        for method_name in ("sample", "sample_rows", "head", "take", "take_rows", "materialize"):
            method = getattr(owner, method_name, None)
            if not callable(method):
                continue
            attempts = (
                lambda: method(dataset_id=dataset_id, columns=[column], n=limit),
                lambda: method(dataset_id=dataset_id, columns=[column], limit=limit),
                lambda: method(dataset_id=dataset_id, columns=[column], max_rows=limit),
                lambda: method(columns=[column], n=limit),
                lambda: method(columns=[column], limit=limit),
                lambda: method(columns=[column], max_rows=limit),
                lambda: method(n=limit, columns=[column]),
                lambda: method(limit, columns=[column]),
            )
            for attempt in attempts:
                try:
                    raw = attempt()
                except TypeError:
                    continue
                except Exception:
                    raw = None
                values = values_from_column_payload(raw, column, limit=limit)
                if values:
                    return values

    try:
        try:
            df = datasets.get_df(dataset_id, columns=[column], limit=limit)
        except TypeError:
            try:
                df = datasets.get_df(dataset_id, columns=[column], max_rows=limit)
            except TypeError:
                try:
                    df = datasets.get_df(dataset_id, columns=[column])
                except TypeError:
                    df = datasets.get_df(dataset_id)
    except Exception:
        return []
    values = values_from_column_payload(df, column, limit=limit)
    return values[:limit]


def values_from_column_payload(raw: Any, column: str, *, limit: int) -> List[Any]:
    if raw is None:
        return []
    if hasattr(raw, "columns") and column in getattr(raw, "columns", []):
        try:
            series = raw[column]
            try:
                series = series.dropna()
            except Exception:
                pass
            try:
                series = series.head(limit)
            except Exception:
                pass
            return list(series)[:limit]
        except Exception:
            return []
    if isinstance(raw, Mapping):
        if column in raw:
            value = raw.get(column)
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
                return list(value)[:limit]
            return [value]
        rows = raw.get("records") or raw.get("rows") or raw.get("data")
        if isinstance(rows, Iterable) and not isinstance(rows, (str, bytes, bytearray, Mapping)):
            return [row.get(column) for row in rows if isinstance(row, Mapping) and column in row][:limit]
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes, bytearray, Mapping)):
        values: List[Any] = []
        for item in raw:
            if isinstance(item, Mapping):
                if column in item:
                    values.append(item.get(column))
            else:
                values.append(item)
            if len(values) >= limit:
                break
        return values
    return []

def infer_label_options_from_column(context: Any, *, dataset_id: str, column: str, max_labels: int = 500) -> List[str]:
    """Infer available AL labels from a dataset column with bounded work."""

    dataset_id = str(dataset_id or "").strip()
    column = str(column or "").strip()
    if not dataset_id or not column:
        return []
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return []

    profile = infer_label_profile_from_column(context, dataset_id=dataset_id, column=column, max_classes=min(int(max_labels), 50))
    if al_state.parse_task_type(profile.get("task_type")) == al_state.TASK_REGRESSION:
        return []
    profile_labels = profile.get("label_options") or profile.get("class_labels") or []
    if profile_labels:
        return [str(label) for label in profile_labels[: int(max_labels)] if str(label).strip()]

    values: List[Any] = []
    source = getattr(datasets, "get_source", lambda *_: None)(dataset_id)
    for owner in (datasets, source):
        if owner is None:
            continue
        for method_name in ("unique_values", "distinct_values", "value_counts"):
            method = getattr(owner, method_name, None)
            if not callable(method):
                continue
            attempts = (
                lambda: method(dataset_id=dataset_id, column=column, limit=max_labels),
                lambda: method(column=column, limit=max_labels),
                lambda: method(column, max_labels),
            )
            for attempt in attempts:
                try:
                    raw = attempt()
                except TypeError:
                    continue
                except Exception:
                    raw = None
                if raw is None:
                    continue
                values = list(raw.keys()) if isinstance(raw, Mapping) else list(raw)
                break
            if values:
                break
        if values:
            break

    if not values:
        try:
            try:
                df = datasets.get_df(dataset_id, columns=[column])
            except TypeError:
                df = datasets.get_df(dataset_id)
        except Exception:
            return []
        if column not in getattr(df, "columns", []):
            return []
        try:
            series = df[column].dropna()
            try:
                series = series.head(50000)
            except Exception:
                pass
            values = series.unique().tolist()
        except Exception:
            values = list(df[column])[:50000]
    labels: List[str] = []
    for value in values:
        text = str(value).strip()
        if text and al_state.normalise_label(text) != al_state.UNSURE_LABEL and text not in labels:
            labels.append(text)
        if len(labels) >= int(max_labels):
            break
    return labels

def dataset_label_values_by_row_id(
    context: Any,
    *,
    dataset_id: str,
    row_ids: Sequence[Any],
    label_column: str,
) -> Dict[str, Any]:
    """Return source label-column values keyed by canonical row id.

    Keep this narrow: only the id/label columns are requested, and pandas work
    is vectorised.  Do not iterate a full wide catalogue row-by-row during bulk
    labelling.
    """

    dataset_id = str(dataset_id or "").strip()
    label_column = str(label_column or "").strip()
    wanted = {str(row_id) for row_id in row_ids if str(row_id)}
    if not dataset_id or not label_column or not wanted:
        return {}

    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return {}
    id_column = acquisition.resolve_record_id_column(context, dataset_id)
    columns = [label_column]
    if id_column and id_column != label_column:
        columns.insert(0, id_column)

    source = getattr(datasets, "get_source", lambda *_: None)(dataset_id)
    # Prefer source-level row lookup APIs when available.
    for owner in (datasets, source):
        if owner is None:
            continue
        for method_name in ("rows_by_ids", "get_rows_by_ids", "take_ids", "lookup_rows"):
            method = getattr(owner, method_name, None)
            if not callable(method):
                continue
            attempts = (
                lambda: method(dataset_id=dataset_id, row_ids=list(wanted), columns=columns),
                lambda: method(row_ids=list(wanted), columns=columns),
                lambda: method(list(wanted), columns=columns),
            )
            rows = None
            for attempt in attempts:
                try:
                    rows = attempt()
                    break
                except TypeError:
                    continue
                except Exception:
                    rows = None
                    break
            if rows is None:
                continue
            try:
                if hasattr(rows, "to_pandas"):
                    rows = rows.to_pandas()
                if hasattr(rows, "columns"):
                    df = rows
                    if label_column not in getattr(df, "columns", []):
                        continue
                    if id_column and id_column in df.columns:
                        work = df[[id_column, label_column]].copy()
                        work["__al_row_id"] = work[id_column].astype(str)
                    else:
                        work = df[[label_column]].copy()
                        work["__al_row_id"] = [str(index) for index in work.index]
                    work = work[work["__al_row_id"].isin(wanted)]
                    return {str(row_id): value for row_id, value in zip(work["__al_row_id"], work[label_column])}
                if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
                    out: Dict[str, Any] = {}
                    for row in rows:
                        if not isinstance(row, Mapping):
                            continue
                        row_id = str(row.get(id_column) if id_column else row.get("row_id") or row.get("id") or "")
                        if row_id in wanted:
                            out[row_id] = row.get(label_column)
                    if out:
                        return out
            except Exception:
                continue

    try:
        try:
            df = datasets.get_df(dataset_id, columns=columns)
        except TypeError:
            df = datasets.get_df(dataset_id)
    except Exception:
        return {}
    if label_column not in getattr(df, "columns", []):
        return {}

    try:
        if id_column and id_column in df.columns:
            work = df[[id_column, label_column]].copy()
            work["__al_row_id"] = work[id_column].astype(str)
        else:
            work = df[[label_column]].copy()
            work["__al_row_id"] = [str(index) for index in work.index]
        work = work[work["__al_row_id"].isin(wanted)]
        return {str(row_id): value for row_id, value in zip(work["__al_row_id"], work[label_column])}
    except Exception:
        out: Dict[str, Any] = {}
        if id_column and id_column in df.columns:
            for _, row in df.iterrows():
                row_id = str(row.get(id_column))
                if row_id in wanted:
                    out[row_id] = row.get(label_column)
        else:
            for index, row in df.iterrows():
                row_id = str(index)
                if row_id in wanted:
                    out[row_id] = row.get(label_column)
        return out

def dataset_mappings(context: Any, dataset_id: str) -> Dict[str, str]:
    try:
        return dict(context.datasets.get_mappings(dataset_id) or {})
    except Exception:
        return {}

def filter_mappings_to_columns(mappings: Mapping[str, str], columns: Iterable[Any]) -> Dict[str, str]:
    available = {str(column) for column in columns}
    return {str(k): str(v) for k, v in dict(mappings or {}).items() if str(v) in available}

def unique_dataset_id(prefix: str) -> str:
    return str(prefix).replace(":", "_").replace("/", "_") + f"_{uuid.uuid4().hex[:6]}"
