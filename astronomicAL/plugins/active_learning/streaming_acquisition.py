from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from . import acquisition
from . import state as al_state
from . import strategies
from . import embedding_acquisition
from .membership_storage import (
    ExclusionPlan,
    exclusion_plan_from_session,
    iter_eligible_prediction_batches,
)
from .streaming_io import check_cancelled

@dataclass
class _TopKEntry:
    score: float
    row_id: str
    metadata: Dict[str, Any] = field(compare=False)

    def __lt__(self, other: "_TopKEntry") -> bool:
        # heap root is the worst retained item: lower score, then larger id.
        if self.score != other.score:
            return self.score < other.score
        return self.row_id > other.row_id

@dataclass(frozen=True)
class StreamingScoreResult:
    records: list[Dict[str, Any]]
    stats_by_strategy: Dict[str, Dict[str, Any]]
    eligible_count: int
    excluded_count: int
    prediction_record_count: int

def acquire_query_batch_streaming(
    *,
    context: Any,
    registry: Any,
    session: Mapping[str, Any],
    predictions_payload: Mapping[str, Any],
    predictions_artifact_id: str,
    strategy_id: str,
    k: int,
    seed: int,
    params: Mapping[str, Any],
    cancel_token: Any = None,
):
    dataset_id = acquisition.session_pool_dataset_id(session)
    _validate_prediction_dataset(
        predictions_payload,
        dataset_id=dataset_id,
    )
    exclusion_plan = _resolve_exclusion_plan(session, params)
    strategy = registry.get(strategy_id)
    batch_size = max(1, int(params.get("prediction_scan_batch_size") or 8192))

    if bool(getattr(strategy, "batch_aware", False)):
        result = _bounded_batch_acquisition(
            context=context,
            registry=registry,
            strategy=strategy,
            strategy_id=strategy_id,
            session=session,
            predictions_payload=predictions_payload,
            exclusion_plan=exclusion_plan,
            k=k,
            seed=seed,
            params=params,
            batch_size=batch_size,
            cancel_token=cancel_token,
        )
    else:
        result = _streaming_top_k(
            context=context,
            strategy=strategy,
            strategy_id=strategy_id,
            session=session,
            predictions_payload=predictions_payload,
            exclusion_plan=exclusion_plan,
            k=k,
            seed=seed,
            params=params,
            batch_size=batch_size,
            cancel_token=cancel_token,
        )

    records = result.records()
    result.stats.setdefault("excluded_count", exclusion_plan.estimated_excluded_count)
    result.stats.setdefault("exclude_row_ids_count", exclusion_plan.estimated_excluded_count)
    result.stats.setdefault("returned_count", len(records))
    result.stats.setdefault("query_dataset_id", dataset_id)
    result.stats.setdefault("predictions_artifact_id", predictions_artifact_id)
    if exclusion_plan.membership_ref is not None:
        result.stats.setdefault(
            "membership_table_ref",
            exclusion_plan.membership_ref.to_dict(),
        )
    return result, records

def score_pool_streaming(
    *,
    context: Any,
    registry: Any,
    session: Mapping[str, Any],
    predictions_payload: Mapping[str, Any],
    strategy_ids: Sequence[str],
    seed: int,
    params: Mapping[str, Any],
    cancel_token: Any = None,
) -> StreamingScoreResult:
    exclusion_plan = _resolve_exclusion_plan(session, params)
    batch_size = max(1, int(params.get("prediction_scan_batch_size") or 8192))
    strategies_by_id: Dict[str, Any] = {}
    stats: Dict[str, Dict[str, Any]] = {}
    for raw_id in strategy_ids:
        strategy_id = str(raw_id or "").strip()
        if not strategy_id:
            continue
        strategy = registry.get(strategy_id)
        info = _strategy_info(strategy, strategy_id)
        if bool(getattr(strategy, "batch_aware", False)):
            stats[strategy_id] = {
                "strategy_id": strategy_id,
                "strategy_title": info["title"],
                "scored_pool_count": 0,
                "selected_count": 0,
                "streaming": True,
                "error": (
                    "Whole-pool streaming scores are only defined for per-record strategies. "
                    "Use query_batch for embedding-backed batch-aware acquisition."
                ),
            }
            continue
        strategies_by_id[strategy_id] = strategy
        stats[strategy_id] = {
            "strategy_id": strategy_id,
            "strategy_title": info["title"],
            "scored_pool_count": 0,
            "missing_score_count": 0,
            "selected_count": 0,
            "streaming": True,
        }

    if not strategies_by_id and not stats:
        raise ValueError("No query strategies were selected.")

    pool = _query_pool(
        context=context,
        session=session,
        predictions_payload=predictions_payload,
        excluded_row_ids=exclusion_plan.in_memory_row_ids,
    )
    output: list[Dict[str, Any]] = []
    batches, scan_stats = iter_eligible_prediction_batches(
        predictions_payload,
        exclusion_plan=exclusion_plan,
        batch_size=batch_size,
        cancel_token=cancel_token,
    )
    for batch in batches:
        for raw in batch.records:
            check_cancelled(cancel_token)
            row_id = _row_id(raw)
            if not row_id:
                continue
            for strategy_id, strategy in strategies_by_id.items():
                score = _score_record(
                    strategy,
                    raw,
                    pool=pool,
                    predictions_payload=predictions_payload,
                    seed=seed,
                    params=params,
                )
                if score is None:
                    stats[strategy_id]["missing_score_count"] += 1
                    continue
                title = stats[strategy_id]["strategy_title"]
                output.append(
                    {
                        "score_id": f"{strategy_id}:{row_id}",
                        "row_id": row_id,
                        "strategy_id": strategy_id,
                        "strategy_title": title,
                        "score": score,
                        "informativeness_score": score,
                        "active_learning_score": score,
                        "rank": None,
                        "selection_rank": None,
                        "score_source": str(raw.get("_al_score_source") or "direct_score"),
                    }
                )
                stats[strategy_id]["scored_pool_count"] += 1

    for strategy_stats in stats.values():
        strategy_stats["eligible_pool_count"] = scan_stats.eligible_row_count
        strategy_stats["prediction_record_count"] = scan_stats.source_row_count
        strategy_stats["excluded_count"] = scan_stats.excluded_row_count
        strategy_stats["anti_join_mode"] = scan_stats.mode
        strategy_stats["selected_count"] = strategy_stats.get("scored_pool_count", 0)
        strategy_stats["scored_for_visualisation"] = True

    return StreamingScoreResult(
        records=output,
        stats_by_strategy=stats,
        eligible_count=scan_stats.eligible_row_count,
        excluded_count=scan_stats.excluded_row_count,
        prediction_record_count=scan_stats.source_row_count,
    )

def iter_pool_score_rows(
    *,
    context: Any,
    registry: Any,
    session: Mapping[str, Any],
    predictions_payload: Mapping[str, Any],
    strategy_ids: Sequence[str],
    seed: int,
    params: Mapping[str, Any],
    cancel_token: Any = None,
):
    """Yield score-row batches and a mutable stats dictionary."""

    exclusion_plan = _resolve_exclusion_plan(session, params)
    batch_size = max(1, int(params.get("prediction_scan_batch_size") or 8192))
    score_batch_size = max(1, int(params.get("score_output_batch_size") or 8192))
    strategy_map: Dict[str, Any] = {}
    stats: Dict[str, Dict[str, Any]] = {}
    for raw_id in strategy_ids:
        strategy_id = str(raw_id or "").strip()
        if not strategy_id:
            continue
        strategy = registry.get(strategy_id)
        title = _strategy_info(strategy, strategy_id)["title"]
        if bool(getattr(strategy, "batch_aware", False)):
            stats[strategy_id] = {
                "strategy_id": strategy_id,
                "strategy_title": title,
                "streaming": True,
                "scored_pool_count": 0,
                "selected_count": 0,
                "error": "Whole-pool scores are not defined for batch-aware strategies.",
            }
            continue
        strategy_map[strategy_id] = strategy
        stats[strategy_id] = {
            "strategy_id": strategy_id,
            "strategy_title": title,
            "streaming": True,
            "scored_pool_count": 0,
            "missing_score_count": 0,
            "selected_count": 0,
        }

    state = {
        "stats_by_strategy": stats,
        "eligible_count": 0,
        "excluded_count": exclusion_plan.estimated_excluded_count,
        "prediction_record_count": 0,
        "anti_join_mode": "pending",
    }
    pool = _query_pool(
        context=context,
        session=session,
        predictions_payload=predictions_payload,
        excluded_row_ids=exclusion_plan.in_memory_row_ids,
    )

    def generator():
        buffer: list[Dict[str, Any]] = []
        batches, scan_stats = iter_eligible_prediction_batches(
            predictions_payload,
            exclusion_plan=exclusion_plan,
            batch_size=batch_size,
            cancel_token=cancel_token,
        )
        for batch in batches:
            for raw in batch.records:
                check_cancelled(cancel_token)
                row_id = _row_id(raw)
                if not row_id:
                    continue
                for strategy_id, strategy in strategy_map.items():
                    score = _score_record(
                        strategy,
                        raw,
                        pool=pool,
                        predictions_payload=predictions_payload,
                        seed=seed,
                        params=params,
                    )
                    if score is None:
                        stats[strategy_id]["missing_score_count"] += 1
                        continue
                    stats[strategy_id]["scored_pool_count"] += 1
                    buffer.append(
                        {
                            "score_id": f"{strategy_id}:{row_id}",
                            "row_id": row_id,
                            "strategy_id": strategy_id,
                            "strategy_title": stats[strategy_id]["strategy_title"],
                            "score": score,
                            "informativeness_score": score,
                            "active_learning_score": score,
                            "rank": None,
                            "selection_rank": None,
                            "score_source": str(raw.get("_al_score_source") or "direct_score"),
                        }
                    )
                    if len(buffer) >= score_batch_size:
                        yield buffer
                        buffer = []
        if buffer:
            yield buffer
        state["eligible_count"] = scan_stats.eligible_row_count
        state["excluded_count"] = scan_stats.excluded_row_count
        state["prediction_record_count"] = scan_stats.source_row_count
        state["anti_join_mode"] = scan_stats.mode
        for item in stats.values():
            item["eligible_pool_count"] = scan_stats.eligible_row_count
            item["prediction_record_count"] = scan_stats.source_row_count
            item["excluded_count"] = scan_stats.excluded_row_count
            item["anti_join_mode"] = scan_stats.mode
            item["selected_count"] = item.get("scored_pool_count", 0)
            item["scored_for_visualisation"] = True

    return generator(), state

def _streaming_top_k(
    *,
    context: Any,
    strategy: Any,
    strategy_id: str,
    session: Mapping[str, Any],
    predictions_payload: Mapping[str, Any],
    exclusion_plan: ExclusionPlan,
    k: int,
    seed: int,
    params: Mapping[str, Any],
    batch_size: int,
    cancel_token: Any,
):
    limit = max(0, int(k))
    heap: list[_TopKEntry] = []
    pool = _query_pool(
        context=context,
        session=session,
        predictions_payload=predictions_payload,
        excluded_row_ids=exclusion_plan.in_memory_row_ids,
    )
    scored_count = 0
    missing_count = 0
    invalid_count = 0

    batches, scan_stats = iter_eligible_prediction_batches(
        predictions_payload,
        exclusion_plan=exclusion_plan,
        batch_size=batch_size,
        cancel_token=cancel_token,
    )
    for batch in batches:
        for raw in batch.records:
            check_cancelled(cancel_token)
            row_id = _row_id(raw)
            if not row_id:
                continue
            score = _score_record(
                strategy,
                raw,
                pool=pool,
                predictions_payload=predictions_payload,
                seed=seed,
                params=params,
            )
            if score is None:
                missing_count += 1
                continue
            if not math.isfinite(score):
                invalid_count += 1
                continue
            scored_count += 1
            if limit <= 0:
                continue
            entry = _TopKEntry(score=score, row_id=row_id, metadata=dict(raw))
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif _is_better(entry, heap[0]):
                heapq.heapreplace(heap, entry)

    if str(strategy_id) == "learning_loss" and (missing_count or invalid_count):
        raise ValueError(
            "Learning Loss requires a valid model-emitted loss prediction for every "
            f"eligible row; missing={missing_count}, invalid={invalid_count}."
        )

    selected = sorted(heap, key=lambda item: (-item.score, item.row_id))
    candidates = [
        strategies.QueryCandidate(
            row_id=item.row_id,
            score=float(item.score),
            metadata=item.metadata,
        )
        for item in selected
    ]
    info = _strategy_info(strategy, strategy_id)
    return strategies.QueryResult(
        strategy_id=str(strategy_id),
        candidates=candidates,
        stats={
            "strategy_id": str(strategy_id),
            "strategy_title": info["title"],
            "streaming": True,
            "streaming_mode": "top_k_heap",
            "anti_join_mode": scan_stats.mode,
            "prediction_record_count": scan_stats.source_row_count,
            "eligible_pool_count": scan_stats.eligible_row_count,
            "excluded_count": scan_stats.excluded_row_count,
            "scored_pool_count": scored_count,
            "missing_score_count": missing_count,
            "invalid_score_count": invalid_count,
            "requested_k": limit,
            "selected_count": len(candidates),
        },
    )

def _bounded_batch_acquisition(
    *,
    context: Any,
    registry: Any,
    strategy: Any,
    strategy_id: str,
    session: Mapping[str, Any],
    predictions_payload: Mapping[str, Any],
    exclusion_plan: ExclusionPlan,
    k: int,
    seed: int,
    params: Mapping[str, Any],
    batch_size: int,
    cancel_token: Any,
):
    embedding_result = embedding_acquisition.acquire_global_embedding_batch(
        context=context,
        session=session,
        predictions_payload=predictions_payload,
        strategy=strategy,
        strategy_id=strategy_id,
        k=k,
        seed=seed,
        params=params,
        exclusion_plan=exclusion_plan,
        cancel_token=cancel_token,
    )
    if embedding_result is not None:
        candidates = [
            strategies.QueryCandidate(
                row_id=str(record.get("row_id") or ""),
                score=float(record.get("score") or 0.0),
                metadata=dict(record),
            )
            for record in embedding_result.records
            if str(record.get("row_id") or "")
        ]
        stats = dict(embedding_result.stats)
        stats["embedding_ref"] = embedding_result.embedding_ref
        return strategies.QueryResult(
            strategy_id=str(strategy_id),
            candidates=candidates,
            stats=stats,
        )

    max_rows = max(1, int(params.get("batch_strategy_max_rows") or 100_000))
    records: list[Dict[str, Any]] = []
    batches, scan_stats = iter_eligible_prediction_batches(
        predictions_payload,
        exclusion_plan=exclusion_plan,
        batch_size=batch_size,
        cancel_token=cancel_token,
    )
    for batch in batches:
        for raw in batch.records:
            row_id = _row_id(raw)
            if not row_id:
                continue
            records.append(dict(raw))
            if len(records) > max_rows:
                raise ValueError(
                    f"Strategy {strategy_id!r} is batch-aware and requires the eligible pool in memory. "
                    f"The pool exceeds batch_strategy_max_rows={max_rows}. Provide a standard "
                    "embedding_ref for external-memory global acquisition or use a per-record strategy."
                )

    pool = _query_pool(
        context=context,
        session=session,
        predictions_payload=predictions_payload,
        excluded_row_ids=exclusion_plan.in_memory_row_ids,
        records=records,
    )
    result = registry.acquire(
        pool,
        strategy_id=strategy_id,
        k=max(1, int(k)),
        seed=int(seed),
        params=dict(params.get("strategy_params") or params),
        cancel_token=cancel_token,
    )
    result.stats.setdefault("streaming", True)
    result.stats.setdefault("streaming_mode", "bounded_batch_materialization")
    result.stats.setdefault("anti_join_mode", scan_stats.mode)
    result.stats.setdefault("batch_strategy_max_rows", max_rows)
    result.stats.setdefault("prediction_record_count", scan_stats.source_row_count)
    result.stats.setdefault("eligible_pool_count", scan_stats.eligible_row_count)
    result.stats.setdefault("excluded_count", scan_stats.excluded_row_count)
    return result

def _query_pool(
    *,
    context: Any,
    session: Mapping[str, Any],
    predictions_payload: Mapping[str, Any],
    excluded_row_ids: set[str],
    records: Optional[list[Mapping[str, Any]]] = None,
):
    labelled = {
        str(item.get("row_id"))
        for item in al_state.labelled_training_items(session)
        if isinstance(item, Mapping) and item.get("row_id") not in (None, "")
    }
    return strategies.QueryPool(
        dataset_id=acquisition.session_pool_dataset_id(session),
        records=list(records or []),
        predictions_payload=predictions_payload,
        session=session,
        labelled_row_ids=labelled,
        excluded_row_ids=set(excluded_row_ids),
        artifacts=getattr(context, "artifacts", None),
        services=getattr(context, "services", None),
        context=context,
    )

def _resolve_exclusion_plan(
    session: Mapping[str, Any],
    params: Mapping[str, Any],
) -> ExclusionPlan:
    durable = (
        session.get("membership_table_ref")
        if isinstance(session.get("membership_table_ref"), Mapping)
        else None
    )
    legacy_ids: Iterable[Any] = ()
    if durable is None:
        legacy_ids = acquisition.session_query_exclude_row_ids(session, params)
    return exclusion_plan_from_session(
        session,
        params,
        legacy_row_ids=legacy_ids,
    )


def _score_record(
    strategy: Any,
    record: Mapping[str, Any],
    *,
    pool: Any,
    predictions_payload: Mapping[str, Any],
    seed: int,
    params: Mapping[str, Any],
) -> Optional[float]:
    try:
        raw = strategy.score(
            record,
            pool=pool,
            predictions_payload=predictions_payload,
            seed=seed,
            params=dict(params.get("strategy_params") or params),
        )
    except NotImplementedError:
        return None
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None

def _row_id(record: Mapping[str, Any]) -> str:
    return str(record.get("row_id") or record.get("record_id") or record.get("id") or "").strip()

def _is_better(candidate: _TopKEntry, worst: _TopKEntry) -> bool:
    return candidate.score > worst.score or (
        candidate.score == worst.score and candidate.row_id < worst.row_id
    )

def _strategy_info(strategy: Any, strategy_id: str) -> Dict[str, Any]:
    info_method = getattr(strategy, "info", None)
    if callable(info_method):
        try:
            info = info_method()
            return {
                "id": str(getattr(info, "id", strategy_id)),
                "title": str(getattr(info, "title", strategy_id)),
            }
        except Exception:
            pass
    return {
        "id": str(getattr(strategy, "id", strategy_id)),
        "title": str(getattr(strategy, "title", strategy_id)),
    }

def _validate_prediction_dataset(
    predictions_payload: Mapping[str, Any],
    *,
    dataset_id: str,
) -> None:
    identifiers_fn = getattr(acquisition, "prediction_dataset_identifiers", None)
    if not callable(identifiers_fn):
        return
    identifiers = set(identifiers_fn(predictions_payload) or [])
    if identifiers and dataset_id and dataset_id not in identifiers:
        got = ", ".join(sorted(str(value) for value in identifiers))
        raise ValueError(
            "query_batch predictions: dataset mismatch. "
            f"Expected pool dataset {dataset_id!r}, got [{got}]."
        )
