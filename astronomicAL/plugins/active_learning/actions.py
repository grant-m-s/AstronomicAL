from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from astronomicAL.platform.plugins.specs import ActionRequest

from . import acquisition
from . import state as al_state
from . import strategies as strategy_module
from . import label_storage
from . import membership_storage

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
    pool_dataset_id = str(params.get("pool_dataset_id") or dataset_id).strip() or dataset_id

    selected_row_ids = acquisition.sample_dataset_row_ids(context, dataset_id=pool_dataset_id, k=initial_k, seed=seed)
    session = al_state.create_session(
        dataset_id=dataset_id,
        pool_dataset_id=pool_dataset_id,
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
            "seed": seed,
        },
    )
    batch_artifact_id = context.artifacts.put(
        al_state.ARTIFACT_BATCH,
        batch_payload,
        dataset_id=pool_dataset_id,
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
            dataset_id=pool_dataset_id,
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
            "pool_dataset_id": pool_dataset_id,
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

    Missing, ambiguous, unsure, invalid-regression, and out-of-session labels are
    skipped and left unverified. They are not recorded as Unsure automatically,
    because problematic labels are exactly what AL review is meant to preserve.
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

    candidate_row_ids = next_unlabelled_batch_row_ids(
        session,
        row_ids=row_ids,
        start_row_id=start_row_id,
        n=len(row_ids),
    )

    labels_by_row_id = dataset_label_values_by_row_id(
        context,
        dataset_id=dataset_id,
        row_ids=candidate_row_ids,
        label_column=label_column,
    )

    is_regression = al_state.is_regression_task(session)

    allowed_labels = set()
    if not is_regression:
        for value in session.get("label_options") or []:
            label = al_state.normalise_label(value)
            if label and label != al_state.UNSURE_LABEL:
                allowed_labels.add(label)

    selected_row_ids: List[str] = []
    selected_labels: Dict[str, Any] = {}
    skipped: List[Dict[str, str]] = []

    for row_id in candidate_row_ids:
        row_id = str(row_id)
        raw_label = labels_by_row_id.get(row_id)

        if al_state.is_missing_label_value(raw_label):
            skipped.append({"row_id": row_id, "reason": "missing_label"})
            continue

        label = al_state.normalise_label(raw_label)

        if not label:
            skipped.append({"row_id": row_id, "reason": "blank_label"})
            continue

        if label == al_state.UNSURE_LABEL:
            skipped.append({"row_id": row_id, "reason": "unsure_or_ambiguous_label"})
            continue

        if is_regression:
            try:
                selected_label: Any = al_state.normalise_regression_label(raw_label)
            except Exception:
                skipped.append(
                    {
                        "row_id": row_id,
                        "reason": "invalid_regression_label",
                        "label": al_state.label_text(raw_label),
                    }
                )
                continue
        else:
            if allowed_labels and label not in allowed_labels:
                skipped.append(
                    {
                        "row_id": row_id,
                        "reason": "label_not_in_session",
                        "label": label,
                    }
                )
                continue
            selected_label = label

        selected_row_ids.append(row_id)
        selected_labels[row_id] = selected_label

        if len(selected_row_ids) >= n:
            break

    if not selected_row_ids:
        remaining_row_ids = remaining_review_row_ids(session, row_ids=row_ids)
        next_row_id = next_review_row_id(session, row_ids=row_ids, after_row_id=start_row_id)

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
            "remaining_row_ids": remaining_row_ids,
            "remaining_count": len(remaining_row_ids),
            "next_row_id": next_row_id,
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

def profile_data_contract_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    from . import core_ml_bridge

    return core_ml_bridge.profile_data_contract_action(context, request, cancel_token=cancel_token)

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
    session_payload = al_state.with_revision(
        session,
        previous_session_artifact_id=previous_artifact_id,
    )
    session_payload = _persist_session_tables(context, session_payload)
    membership_ref = membership_storage.MembershipTableRef.from_value(
        session_payload["membership_table_ref"]
    )
    row_ids_preview = membership_storage.excluded_row_ids_preview(
        session_payload,
        limit=1000,
    )
    artifact_id = context.artifacts.put(
        al_state.ARTIFACT_SESSION,
        session_payload,
        dataset_id=str(session_payload.get("dataset_id") or "default"),
        row_ids=row_ids_preview,
        row_count=int(membership_ref.row_count),
        row_ids_ref=membership_storage.artifact_row_ids_ref(membership_ref),
        params={
            "session_id": session_payload.get("session_id"),
            "round": session_payload.get("round"),
            "revision": session_payload.get("revision"),
            "previous_session_artifact_id": previous_artifact_id,
            "label_table_artifact_id": (session_payload.get("latest") or {}).get("label_table_artifact_id"),
            "membership_table_artifact_id": (session_payload.get("latest") or {}).get("membership_table_artifact_id"),
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
            "label_table_artifact_id": (session_payload.get("latest") or {}).get("label_table_artifact_id"),
            "membership_table_artifact_id": (session_payload.get("latest") or {}).get("membership_table_artifact_id"),
            "counts": al_state.counts(session_payload),
        },
    )
    return str(artifact_id)

def _persist_session_tables(context: Any, session: Mapping[str, Any]) -> Dict[str, Any]:
    payload = al_state.coerce_session(session)
    storage_params = dict(payload.get("storage") or {})
    label_ref, label_changed = label_storage.ensure_label_table(
        context,
        payload,
        params=storage_params,
    )
    membership_ref, membership_changed = membership_storage.ensure_membership_table(
        context,
        payload,
        params=storage_params,
    )
    payload["label_table_ref"] = label_ref.to_dict()
    payload["membership_table_ref"] = membership_ref.to_dict()
    payload["labels_inline_complete"] = True
    payload["memberships_inline_complete"] = True

    latest = dict(payload.get("latest") or {})
    if label_changed or not latest.get("label_table_artifact_id"):
        label_artifact_id = context.artifacts.put(
            al_state.ARTIFACT_LABEL_TABLE,
            {
                "schema_version": label_ref.schema_version,
                "session_id": label_ref.session_id,
                "dataset_id": label_ref.dataset_id,
                "target_column": label_ref.target_column,
                "task_type": label_ref.task_type,
                "label_ref": label_ref.to_dict(),
            },
            dataset_id=label_ref.dataset_id or str(payload.get("dataset_id") or "default"),
            row_count=label_ref.row_count,
            row_ids_ref=label_storage.artifact_row_ids_ref(label_ref),
            params={
                "session_id": label_ref.session_id,
                "content_sha256": label_ref.content_sha256,
                "schema_version": label_ref.schema_version,
            },
        )
        latest["label_table_artifact_id"] = str(label_artifact_id)

    if membership_changed or not latest.get("membership_table_artifact_id"):
        membership_artifact_id = context.artifacts.put(
            al_state.ARTIFACT_MEMBERSHIP_TABLE,
            {
                "schema_version": membership_ref.schema_version,
                "session_id": membership_ref.session_id,
                "dataset_id": membership_ref.dataset_id,
                "membership_ref": membership_ref.to_dict(),
            },
            dataset_id=membership_ref.dataset_id or str(payload.get("dataset_id") or "default"),
            row_count=membership_ref.row_count,
            row_ids_ref=membership_storage.artifact_row_ids_ref(membership_ref),
            params={
                "session_id": membership_ref.session_id,
                "content_sha256": membership_ref.content_sha256,
                "excluded_count": membership_ref.excluded_count,
                "schema_version": membership_ref.schema_version,
            },
        )
        latest["membership_table_artifact_id"] = str(membership_artifact_id)

    payload["latest"] = latest
    return payload

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


def active_learning_dataset_ids(
    session: Mapping[str, Any],
    *,
    include_source: bool = True,
) -> Dict[str, str]:
    """Return the registered datasets participating in an AL session."""

    payload = dict(session or {})
    contract = dict(payload.get("contract") or {})
    protocol = dict(contract.get("data_protocol") or {})

    def protocol_dataset(role: str) -> str:
        value = protocol.get(role)
        if isinstance(value, Mapping):
            return str(value.get("dataset_id") or "").strip()
        return ""

    source_dataset_id = str(
        payload.get("dataset_id")
        or protocol.get("source_dataset_id")
        or ""
    ).strip()
    result: Dict[str, str] = {}
    if include_source and source_dataset_id:
        result["source"] = source_dataset_id

    for role, value in (
        (
            "pool",
            payload.get("pool_dataset_id")
            or protocol_dataset("pool")
            or source_dataset_id,
        ),
        (
            "validation",
            payload.get("validation_dataset_id")
            or protocol_dataset("validation"),
        ),
        (
            "test",
            payload.get("test_dataset_id")
            or protocol_dataset("test"),
        ),
    ):
        dataset_id = str(value or "").strip()
        if dataset_id:
            result[role] = dataset_id
    return result


def _dataset_columns(context: Any, dataset_id: str) -> List[str]:
    manager = getattr(context, "datasets", None)
    if manager is None:
        raise RuntimeError("Dataset manager is unavailable.")

    list_columns = getattr(manager, "list_columns", None)
    if callable(list_columns):
        return [str(column) for column in list_columns(dataset_id)]

    get_source = getattr(manager, "get_source", None)
    if callable(get_source):
        source = get_source(dataset_id)
        columns = getattr(source, "columns", None)
        if callable(columns):
            return [str(column) for column in columns()]

    raise RuntimeError(
        f"Dataset manager cannot list columns for {dataset_id!r}."
    )


def apply_image_mapping_to_session_datasets(
    context: Any,
    session: Mapping[str, Any],
    column_name: str,
    *,
    source: str = "core.active_learning",
    strict: bool = True,
) -> Dict[str, Any]:
    """Apply one authoritative image binding to every AL dataset registration."""

    column_name = str(column_name or "").strip()
    if not column_name:
        raise ValueError("An image column must be selected.")

    role_dataset_ids = active_learning_dataset_ids(
        session,
        include_source=True,
    )
    unique_dataset_roles: Dict[str, List[str]] = {}
    for role, dataset_id in role_dataset_ids.items():
        unique_dataset_roles.setdefault(str(dataset_id), []).append(str(role))

    columns_by_dataset: Dict[str, List[str]] = {}
    missing_by_dataset: Dict[str, List[str]] = {}
    for dataset_id, roles in unique_dataset_roles.items():
        try:
            columns = _dataset_columns(context, dataset_id)
        except Exception as exc:
            missing_by_dataset[dataset_id] = [
                *roles,
                f"inspection failed: {exc}",
            ]
            continue
        columns_by_dataset[dataset_id] = columns
        if column_name not in columns:
            missing_by_dataset[dataset_id] = list(roles)

    if strict and missing_by_dataset:
        details = []
        for dataset_id, roles in missing_by_dataset.items():
            details.append(
                f"{dataset_id!r} ({', '.join(str(role) for role in roles)})"
            )
        raise ValueError(
            f"Image column {column_name!r} is not available in every Active "
            "Learning dataset: "
            + "; ".join(details)
        )

    manager = getattr(context, "datasets", None)
    set_mapping = getattr(manager, "set_mapping", None)
    get_mapping = getattr(manager, "get_mapping", None)
    if not callable(set_mapping):
        raise RuntimeError(
            "DatasetManager.set_mapping() is required to persist the "
            "Active Learning image binding."
        )

    changed: List[Dict[str, Any]] = []
    unchanged: List[Dict[str, Any]] = []
    semantic_names = ("image.path", "image.uri")
    for dataset_id, roles in unique_dataset_roles.items():
        if dataset_id in missing_by_dataset:
            continue
        for semantic_name in semantic_names:
            old_value = (
                get_mapping(dataset_id, semantic_name)
                if callable(get_mapping)
                else dataset_mappings(context, dataset_id).get(semantic_name)
            )
            did_change = bool(
                set_mapping(dataset_id, semantic_name, column_name)
            )
            item = {
                "dataset_id": dataset_id,
                "roles": list(roles),
                "semantic_name": semantic_name,
                "column_name": column_name,
                "old_column_name": old_value,
                "changed": did_change,
            }
            if did_change:
                changed.append(item)
                publish(
                    context,
                    "dataset.mapping.updated",
                    {
                        **item,
                        "source": str(source),
                        "required": True,
                    },
                )
            else:
                unchanged.append(item)

    return {
        "ok": not missing_by_dataset,
        "column_name": column_name,
        "role_dataset_ids": role_dataset_ids,
        "dataset_ids": list(unique_dataset_roles),
        "columns_by_dataset": columns_by_dataset,
        "missing_by_dataset": missing_by_dataset,
        "changed": changed,
        "unchanged": unchanged,
    }


def save_session_image_binding(
    context: Any,
    *,
    session_artifact_id: str,
    session: Mapping[str, Any],
    column_name: str,
    source: str = "core.active_learning.train_tab",
) -> Dict[str, Any]:
    """Persist a Train-tab image binding and propagate dataset mappings."""

    session_artifact_id = str(session_artifact_id or "").strip()
    if not session_artifact_id:
        raise ValueError(
            "An Active Learning session artifact is required to save the "
            "image binding."
        )
    column_name = str(column_name or "").strip()
    if not column_name:
        raise ValueError("An image column must be selected.")

    mapping_result = apply_image_mapping_to_session_datasets(
        context,
        session,
        column_name,
        source=source,
        strict=True,
    )

    updated = al_state.coerce_session(session)
    current_column = str(
        updated.get("image_column")
        or updated.get("image_path_column")
        or ""
    ).strip()
    session_changed = current_column != column_name

    if session_changed:
        updated["image_column"] = column_name
        updated["image_path_column"] = column_name

        contract = dict(updated.get("contract") or {})
        contract["image_binding"] = {
            "schema_version": 1,
            "column_name": column_name,
            "semantic_names": ["image.path", "image.uri"],
            "role_dataset_ids": dict(
                mapping_result.get("role_dataset_ids") or {}
            ),
        }
        updated["contract"] = contract

        latest = dict(updated.get("latest") or {})
        latest["image_column"] = column_name
        updated["latest"] = latest

        updated.setdefault("history", []).append(
            {
                "event": "image_mapping_updated",
                "image_column": column_name,
                "dataset_ids": list(
                    mapping_result.get("dataset_ids") or []
                ),
                "timestamp": al_state.now(),
            }
        )

    mapping_changed = bool(mapping_result.get("changed"))
    if session_changed:
        new_session_artifact_id = put_session(
            context,
            updated,
            previous_artifact_id=session_artifact_id,
        )
    else:
        new_session_artifact_id = session_artifact_id

    payload = {
        "session_artifact_id": new_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": updated.get("session_id"),
        "image_column": column_name,
        "session_changed": session_changed,
        "mapping_changed": mapping_changed,
        "mapping_result": mapping_result,
        "source": str(source),
    }
    publish(context, "al.session.image_mapping.updated", payload)
    return {"ok": True, **payload}


def filter_mappings_to_columns(mappings: Mapping[str, str], columns: Iterable[Any]) -> Dict[str, str]:
    available = {str(column) for column in columns}
    return {str(k): str(v) for k, v in dict(mappings or {}).items() if str(v) in available}

def unique_dataset_id(prefix: str) -> str:
    return str(prefix).replace(":", "_").replace("/", "_") + f"_{uuid.uuid4().hex[:6]}"