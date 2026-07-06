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
    label_options = al_state.parse_label_options(params.get("label_options") or [])
    if not label_options and bool(params.get("infer_labels_from_column", True)) and target_column:
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
    publish(
        context,
        "al.label.recorded",
        {
            "session_artifact_id": new_session_artifact_id,
            "previous_session_artifact_id": session_artifact_id,
            "session_id": updated["session_id"],
            "dataset_id": updated["dataset_id"],
            "row_id": row_id,
            "label": al_state.normalise_label(label),
            "display_label": al_state.display_label(label),
            "counts": al_state.counts(updated),
        },
    )
    return {
        "ok": True,
        "session_artifact_id": new_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": updated["session_id"],
        "dataset_id": updated["dataset_id"],
        "row_id": row_id,
        "label": al_state.normalise_label(label),
        "display_label": al_state.display_label(label),
        "counts": al_state.counts(updated),
    }


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
    allowed_labels = {al_state.normalise_label(value) for value in (session.get("label_options") or []) if value not in (None, "")}

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
    publish(
        context,
        "al.labels.bulk_recorded",
        {
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
            "counts": al_state.counts(updated),
        },
    )
    return {
        "ok": True,
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
        "counts": al_state.counts(updated),
    }


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


def infer_label_options_from_column(context: Any, *, dataset_id: str, column: str, max_labels: int = 500) -> List[str]:
    """Infer available AL labels from a dataset column.

    This mirrors the panel behaviour so workflow/action callers do not need to
    pass label_options manually once they have chosen the label column.
    """

    dataset_id = str(dataset_id or "").strip()
    column = str(column or "").strip()
    if not dataset_id or not column:
        return []
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return []
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
        values = df[column].dropna().unique().tolist()
    except Exception:
        values = list(df[column])
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
    """Return source label-column values keyed by canonical row id."""

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
    try:
        try:
            df = datasets.get_df(dataset_id, columns=columns)
        except TypeError:
            df = datasets.get_df(dataset_id)
    except Exception:
        return {}
    if label_column not in getattr(df, "columns", []):
        return {}

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
