from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import state as al_state
from .strategies import QueryPool, QueryResult, QueryStrategyRegistry

def build_initial_records(row_ids: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": str(row_id),
            "selection_rank": idx,
            "rank": idx,
            "informativeness_score": None,
            "active_learning_strategy": "initial_random",
            "source": "initial_random",
        }
        for idx, row_id in enumerate(row_ids, start=1)
    ]

def create_batch_payload(
    *,
    dataset_id: str,
    session: Mapping[str, Any],
    strategy_id: str,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    predictions_artifact_id: Optional[str],
    kind: str,
    rank_stats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": str(kind),
        "dataset_id": str(dataset_id),
        "session_id": session.get("session_id"),
        "round": int(session.get("round", 0)),
        "predictions_artifact_id": str(predictions_artifact_id or ""),
        "strategy": str(strategy_id),
        "strategy_id": str(strategy_id),
        "records": [dict(record) for record in records],
        "row_ids": [str(record.get("row_id")) for record in records if record.get("row_id") is not None],
        "count": len(records),
        "ordered_by": "informativeness_desc" if kind == "query" else "random",
        "params": dict(params or {}),
        "rank_stats": dict(rank_stats or {}),
        "session_counts": al_state.counts(session),
    }

def build_query_pool(
    *,
    context: Any,
    dataset_id: str,
    session: Mapping[str, Any],
    predictions_payload: Mapping[str, Any],
    exclude_row_ids: Iterable[Any] = (),
) -> QueryPool:
    records = extract_prediction_records(predictions_payload)
    excluded = set(str(row_id) for row_id in exclude_row_ids)
    labelled = set(str(row_id) for row_id in (session.get("training_row_ids") or []))
    return QueryPool(
        dataset_id=str(dataset_id),
        records=records,
        predictions_payload=dict(predictions_payload or {}),
        session=dict(session or {}),
        labelled_row_ids=labelled,
        excluded_row_ids=excluded,
        artifacts=getattr(context, "artifacts", None),
        services=getattr(context, "services", None),
        context=context,
    )

def extract_prediction_records(predictions_payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for key in ("records", "predictions", "rows", "data"):
        value = predictions_payload.get(key)
        if isinstance(value, list):
            return [dict(record) for record in value if isinstance(record, Mapping)]
    return []

def dataset_row_count(context: Any, dataset_id: str) -> Optional[int]:
    datasets = getattr(context, "datasets", None)
    source = getattr(datasets, "get_source", lambda *_: None)(dataset_id) if datasets else None
    for owner in (datasets, source):
        if owner is None:
            continue
        for attr_name in ("row_count", "n_rows", "count", "__len__"):
            value = getattr(owner, attr_name, None)
            if value is None:
                continue
            try:
                result = value(dataset_id) if callable(value) and owner is datasets and attr_name != "__len__" else value() if callable(value) else value
                if result is not None:
                    return int(result)
            except Exception:
                continue
    return None

def _source_take_rows(source: Any, *, offsets: Sequence[int], columns: Sequence[str]) -> Any:
    if source is None or not offsets:
        return None
    for method_name in ("take", "take_rows", "rows_by_position", "get_rows", "materialize_rows"):
        method = getattr(source, method_name, None)
        if not callable(method):
            continue
        attempts = (
            lambda: method(offsets=offsets, columns=columns),
            lambda: method(row_numbers=offsets, columns=columns),
            lambda: method(indices=offsets, columns=columns),
            lambda: method(offsets, columns=columns),
            lambda: method(offsets),
        )
        for attempt in attempts:
            try:
                return attempt()
            except TypeError:
                continue
            except Exception:
                return None
    return None

PREDICTION_DATASET_ID_KEYS = (
    "dataset_id",
    "source_dataset_id",
    "input_dataset_id",
    "target_dataset_id",
    "pool_dataset_id",
    "original_dataset_id",
    "prediction_source_dataset_id",
    "predicted_dataset_id",
)

def prediction_dataset_identifiers(predictions_payload: Mapping[str, Any]) -> Set[str]:
    """Return dataset ids that identify what dataset a prediction artifact belongs to.

    core.ml prediction payloads can expose either the direct dataset id or a
    derived prediction-table dataset plus provenance back to the source dataset.
    Querying should be allowed when *any* of those identifiers matches the AL
    pool dataset, but should fail for training-dataset predictions.
    """

    identifiers: Set[str] = set()

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 3:
            return
        if not isinstance(value, Mapping):
            return
        for key in PREDICTION_DATASET_ID_KEYS:
            raw = value.get(key)
            if raw not in (None, ""):
                identifiers.add(str(raw).strip())
        for nested_key in ("metadata", "provenance", "params", "request", "inputs", "dataset", "prediction_dataset"):
            nested = value.get(nested_key)
            if isinstance(nested, Mapping):
                visit(nested, depth + 1)

    visit(predictions_payload)
    return {identifier for identifier in identifiers if identifier}

def prediction_payload_matches_dataset(predictions_payload: Mapping[str, Any], dataset_id: str) -> bool:
    expected = str(dataset_id or "").strip()
    if not expected:
        return True
    identifiers = prediction_dataset_identifiers(predictions_payload)
    return not identifiers or expected in identifiers

def acquire_query_batch(
    *,
    context: Any,
    registry: QueryStrategyRegistry,
    session: Mapping[str, Any],
    predictions_payload: Mapping[str, Any],
    predictions_artifact_id: str,
    strategy_id: str,
    k: int,
    seed: int,
    params: Mapping[str, Any],
    cancel_token: Any = None,
) -> Tuple[QueryResult, List[Dict[str, Any]]]:
    dataset_id = session_pool_dataset_id(session)
    prediction_dataset_ids = prediction_dataset_identifiers(predictions_payload)
    prediction_dataset_id = str(predictions_payload.get("dataset_id") or next(iter(prediction_dataset_ids), "")).strip()
    if prediction_dataset_ids and dataset_id and dataset_id not in prediction_dataset_ids:
        got = ", ".join(sorted(prediction_dataset_ids))
        raise ValueError(
            "query_batch predictions: dataset mismatch. "
            f"Expected pool dataset {dataset_id!r}, got prediction dataset identifiers [{got}]. "
            "Use predictions generated on the AL pool dataset, not the materialised AL training dataset."
        )
    exclude = session_query_exclude_row_ids(session, params)
    pool = build_query_pool(
        context=context,
        dataset_id=dataset_id,
        session=session,
        predictions_payload=predictions_payload,
        exclude_row_ids=exclude,
    )
    if not pool.records:
        raise ValueError("Prediction artifact contains no records to rank.")
    result = registry.acquire(
        pool,
        strategy_id=strategy_id,
        k=max(1, int(k)),
        seed=int(seed),
        params=dict(params.get("strategy_params") or params),
        cancel_token=cancel_token,
    )
    records = result.records()
    result.stats.setdefault("excluded_count", len(exclude))
    result.stats.setdefault("exclude_row_ids_count", len(exclude))
    result.stats.setdefault("prediction_record_count", len(pool.records))
    result.stats.setdefault("returned_count", len(records))
    result.stats.setdefault("query_dataset_id", dataset_id)
    result.stats.setdefault("predictions_dataset_id", prediction_dataset_id)
    result.stats.setdefault("predictions_artifact_id", predictions_artifact_id)
    return result, records

def session_pool_dataset_id(session: Mapping[str, Any]) -> str:
    return str(session.get("pool_dataset_id") or session.get("dataset_id") or "").strip()

def session_query_exclude_row_ids(session: Mapping[str, Any], params: Mapping[str, Any]) -> Set[str]:
    exclude = set(al_state.row_ids_in_states(session, states=tuple(al_state.TERMINAL_QUERY_STATES)))
    values = params.get("exclude_row_ids") or []
    if isinstance(values, Mapping):
        values = values.keys()
    elif isinstance(values, (str, bytes, bytearray, int, float)):
        values = [values]
    for row_id in values:
        if row_id not in (None, ""):
            exclude.add(str(row_id).strip())
    return exclude

def sample_dataset_row_ids(context: Any, *, dataset_id: str, k: int, seed: int) -> List[str]:
    k = max(0, int(k or 0))
    if k == 0:
        return []
    datasets = getattr(context, "datasets", None)
    source = getattr(datasets, "get_source", lambda *_: None)(dataset_id) if datasets else None
    for owner in (datasets, source):
        for method_name in ("sample_row_ids", "sample_ids"):
            method = getattr(owner, method_name, None)
            if callable(method):
                try:
                    values = method(dataset_id=dataset_id, k=k, seed=seed)
                except TypeError:
                    try:
                        values = method(k=k, seed=seed)
                    except TypeError:
                        values = method(k, seed)
                return [str(value) for value in values]
    # If the source can expose row count / positional take, sample positions
    # directly.  Avoid building a Python list of every row id for large pools.
    id_column = resolve_record_id_column(context, dataset_id)
    row_count = dataset_row_count(context, dataset_id)
    source = getattr(datasets, "get_source", lambda *_: None)(dataset_id) if datasets else None
    if row_count and id_column:
        rng = random.Random(seed)
        offsets = rng.sample(range(int(row_count)), min(k, int(row_count)))
        rows = _source_take_rows(source, offsets=offsets, columns=[id_column])
        if rows is not None:
            try:
                import pandas as pd
                if hasattr(rows, "to_pandas"):
                    rows = rows.to_pandas()
                if hasattr(rows, "columns") and id_column in rows.columns:
                    return [str(value) for value in rows[id_column].dropna().tolist()]
                if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
                    out = []
                    for row in rows:
                        if isinstance(row, Mapping):
                            value = row.get(id_column)
                            if value not in (None, ""):
                                out.append(str(value))
                    if out:
                        return out
            except Exception:
                pass
    pool_row_ids = dataset_row_ids(context, dataset_id)
    rng = random.Random(seed)
    return rng.sample(pool_row_ids, min(k, len(pool_row_ids)))

def dataset_row_ids(context: Any, dataset_id: str) -> List[str]:
    id_column = resolve_record_id_column(context, dataset_id)
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return []
    if id_column:
        try:
            df = datasets.get_df(dataset_id, columns=[id_column])
        except TypeError:
            df = datasets.get_df(dataset_id)
        if id_column not in df.columns:
            raise ValueError(f"record_id mapping points to missing column: {id_column}")
        return [str(value) for value in df[id_column].dropna().tolist()]
    df = datasets.get_df(dataset_id, columns=[])
    return [str(idx) for idx in df.index.tolist()]

def resolve_record_id_column(context: Any, dataset_id: str) -> Optional[str]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None
    for semantic_name in ("record_id", "id", "row_id"):
        try:
            column = datasets.get_mapping(dataset_id, semantic_name)
        except Exception:
            column = None
        if column:
            return str(column)
    try:
        columns = {str(col) for col in datasets.list_columns(dataset_id)}
    except Exception:
        try:
            columns = {str(col) for col in datasets.get_df(dataset_id).columns}
        except Exception:
            columns = set()
    for candidate in ("record_id", "id", "ID", "source_id", "object_id", "row_id"):
        if candidate in columns:
            return candidate
    return None

def set_ranked_selection(
    context: Any,
    *,
    dataset_id: str,
    row_ids: Sequence[Any],
    session_artifact_id: str,
    batch_artifact_id: str,
    strategy_id: str,
    origin: str,
    update_focus_policy: str = "first",
) -> None:
    selection = getattr(context, "selection", None)
    if selection is None or not hasattr(selection, "set_selection_set"):
        return
    ordered_row_ids = [str(row_id) for row_id in row_ids]
    selection.set_selection_set(
        dataset_id=dataset_id,
        row_ids=ordered_row_ids,
        origin=origin,
        mode="replace",
        metadata={
            "ordered": True,
            "order": "informativeness_desc",
            "strategy_id": strategy_id,
            "session_artifact_id": session_artifact_id,
            "active_learning_batch_artifact_id": batch_artifact_id,
            "count": len(ordered_row_ids),
            "note": "Rows are intentionally ordered; review navigation should follow this order.",
        },
        create_artifact=True,
        update_focus_policy=str(update_focus_policy or "first"),
    )


def set_focus_row(
    context: Any,
    *,
    dataset_id: str,
    row_id: Any,
    origin: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Best-effort focus update used after queue mutations.

    SelectionManager has had a couple of nearby method shapes during the
    platform refactor.  Keep this tolerant so the action still succeeds when
    only the review text input can be updated by the panel.
    """

    row_id_text = str(row_id or "").strip()
    if not row_id_text:
        return False
    selection = getattr(context, "selection", None)
    if selection is None:
        return False
    payload = {"dataset_id": str(dataset_id or ""), "row_id": row_id_text, "metadata": dict(metadata or {}), "origin": str(origin or "")}
    for method_name in ("set_focus", "set_focused_row", "focus", "focus_row"):
        method = getattr(selection, method_name, None)
        if not callable(method):
            continue
        attempts = (
            lambda: method(dataset_id=payload["dataset_id"], row_id=payload["row_id"], origin=payload["origin"], metadata=payload["metadata"]),
            lambda: method(payload["dataset_id"], payload["row_id"], origin=payload["origin"], metadata=payload["metadata"]),
            lambda: method(payload),
        )
        for attempt in attempts:
            try:
                attempt()
                return True
            except TypeError:
                continue
            except Exception:
                return False
    return False
