from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


@dataclass(frozen=True)
class StrategyInfo:
    id: str
    title: str
    description: str = ""
    requires_probabilities: bool = False
    deterministic: bool = True
    batch_aware: bool = False
    required_prediction_fields: Sequence[str] = ()
    required_record_fields: Sequence[str] = ()
    required_artifact_types: Sequence[str] = ()
    params_schema: Mapping[str, Any] = field(default_factory=dict)
    tags: Sequence[str] = ()


@dataclass(frozen=True)
class QueryCandidate:
    row_id: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_record(self, *, strategy_id: str, rank: int) -> Dict[str, Any]:
        record = dict(self.metadata or {})
        record["row_id"] = str(self.row_id)
        record["informativeness_score"] = float(self.score)
        record["active_learning_score"] = float(self.score)
        record["active_learning_strategy"] = str(strategy_id)
        record["selection_rank"] = int(rank)
        record["rank"] = int(rank)
        return record


@dataclass
class QueryPool:
    """Batch-acquisition context passed to query strategies.

    Advanced strategies such as BatchBALD, BADGE, QUEST, or Learning Loss can
    inspect the full candidate pool, prediction payload, session state,
    labelled/excluded ids, platform artifact store, and service registry.
    """

    dataset_id: str
    records: List[Mapping[str, Any]]
    predictions_payload: Mapping[str, Any] = field(default_factory=dict)
    session: Mapping[str, Any] = field(default_factory=dict)
    labelled_row_ids: Set[str] = field(default_factory=set)
    excluded_row_ids: Set[str] = field(default_factory=set)
    artifacts: Any = None
    services: Any = None
    context: Any = None


@dataclass
class QueryResult:
    strategy_id: str
    candidates: List[QueryCandidate]
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def row_ids(self) -> List[str]:
        return [candidate.row_id for candidate in self.candidates]

    def records(self) -> List[Dict[str, Any]]:
        return [candidate.as_record(strategy_id=self.strategy_id, rank=idx) for idx, candidate in enumerate(self.candidates, start=1)]


class QueryStrategy:
    """Base class for active-learning query strategies.

    The preferred extension point is ``acquire(pool, k=..., params=...)``.  This
    is intentionally batch-oriented so research strategies that reason about the
    whole candidate set can be implemented without modifying the AL plugin.

    Legacy/simple strategies may implement ``score(record, ...)`` only; the
    default ``acquire`` method handles exclusion, scoring, sorting, and top-k.
    """

    id = "base"
    title = "Base strategy"
    description = ""
    requires_probabilities = False
    deterministic = True
    batch_aware = False
    required_prediction_fields: Sequence[str] = ()
    required_record_fields: Sequence[str] = ()
    required_artifact_types: Sequence[str] = ()
    params_schema: Mapping[str, Any] = {}
    tags: Sequence[str] = ()

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            id=str(self.id),
            title=str(self.title),
            description=str(self.description or ""),
            requires_probabilities=bool(self.requires_probabilities),
            deterministic=bool(self.deterministic),
            batch_aware=bool(self.batch_aware),
            required_prediction_fields=tuple(self.required_prediction_fields or ()),
            required_record_fields=tuple(self.required_record_fields or ()),
            required_artifact_types=tuple(self.required_artifact_types or ()),
            params_schema=dict(self.params_schema or {}),
            tags=tuple(self.tags or ()),
        )

    def acquire(
        self,
        pool: QueryPool,
        *,
        k: int,
        params: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        cancel_token: Any = None,
    ) -> List[QueryCandidate]:
        candidates: List[QueryCandidate] = []
        for raw in pool.records:
            _check_cancelled(cancel_token)
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            row_id = str(row.get("row_id") or row.get("id") or "").strip()
            if not row_id or row_id in pool.excluded_row_ids:
                continue
            score = _safe_float(
                self.score(
                    row,
                    pool=pool,
                    predictions_payload=pool.predictions_payload,
                    seed=seed,
                    params=params,
                )
            )
            if score is None or math.isnan(score):
                continue
            candidates.append(QueryCandidate(row_id=row_id, score=float(score), metadata=row))

        candidates.sort(key=lambda item: (-float(item.score), str(item.row_id)))
        return candidates[: max(0, int(k))]

    def score(
        self,
        record: Mapping[str, Any],
        *,
        pool: Optional[QueryPool] = None,
        predictions_payload: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Optional[float]:
        raise NotImplementedError


class PerRecordScoringStrategy(QueryStrategy):
    """Compatibility alias for strategies that score one record at a time."""


class LeastConfidenceStrategy(PerRecordScoringStrategy):
    id = "least_confidence"
    title = "Least confidence"
    description = "Ranks rows by 1 - max class probability."
    requires_probabilities = True
    required_prediction_fields = ("probabilities",)
    tags = ("uncertainty", "classification")

    def score(self, record: Mapping[str, Any], **kwargs: Any) -> Optional[float]:
        for key in ("least_confidence", "least_confidence_score"):
            value = _safe_float(record.get(key))
            if value is not None:
                return value
        confidence = _safe_float(record.get("confidence"))
        if confidence is not None:
            return 1.0 - confidence
        max_probability = _safe_float(record.get("max_probability"))
        if max_probability is not None:
            return 1.0 - max_probability
        probs = probability_values(record)
        if probs:
            return 1.0 - max(probs)
        return _safe_float(record.get("uncertainty"))


class MarginStrategy(PerRecordScoringStrategy):
    id = "margin"
    title = "Smallest margin"
    description = "Ranks rows where the top two class probabilities are closest."
    requires_probabilities = True
    required_prediction_fields = ("probabilities",)
    tags = ("uncertainty", "classification")

    def score(self, record: Mapping[str, Any], **kwargs: Any) -> Optional[float]:
        for key in ("margin_uncertainty", "smallest_margin"):
            value = _safe_float(record.get(key))
            if value is not None:
                return value
        margin = _safe_float(record.get("margin"))
        if margin is not None:
            return 1.0 - margin
        probs = sorted(probability_values(record), reverse=True)
        if len(probs) < 2:
            return None
        return 1.0 - (probs[0] - probs[1])


class EntropyStrategy(PerRecordScoringStrategy):
    id = "entropy"
    title = "Entropy"
    description = "Ranks rows by predictive entropy."
    requires_probabilities = True
    required_prediction_fields = ("probabilities",)
    tags = ("uncertainty", "classification")

    def score(self, record: Mapping[str, Any], **kwargs: Any) -> Optional[float]:
        value = _safe_float(record.get("entropy"))
        if value is not None:
            return value
        probs = probability_values(record)
        if not probs:
            return None
        return float(-sum(p * math.log(max(p, 1.0e-12)) for p in probs))


class RandomStrategy(PerRecordScoringStrategy):
    id = "random"
    title = "Random"
    description = "Ranks rows with a deterministic row-id/seed hash."
    deterministic = True
    tags = ("baseline",)

    def score(self, record: Mapping[str, Any], *, seed: Optional[int] = None, **kwargs: Any) -> Optional[float]:
        row_id = str(record.get("row_id") or record.get("id") or "")
        raw = f"{int(seed or 0)}:{row_id}".encode("utf-8")
        digest = hashlib.sha1(raw).hexdigest()[:16]
        return int(digest, 16) / float(0xFFFFFFFFFFFFFFFF)


class QueryStrategyRegistry:
    def __init__(self) -> None:
        self._strategies: Dict[str, QueryStrategy] = {}
        # Retained only for backward compatibility with older callers.
        self.last_rank_stats: Dict[str, Any] = {}

    def register(self, strategy: QueryStrategy, *, replace: bool = False) -> None:
        strategy_id = str(getattr(strategy, "id", "") or "").strip()
        if not strategy_id:
            raise ValueError("QueryStrategy.id must be a non-empty string.")
        if strategy_id in self._strategies and not replace:
            raise ValueError(f"Query strategy already registered: {strategy_id}")
        self._strategies[strategy_id] = strategy

    def unregister(self, strategy_id: str) -> None:
        self._strategies.pop(str(strategy_id), None)

    def get(self, strategy_id: str) -> QueryStrategy:
        strategy_id = str(strategy_id or "").strip()
        if not strategy_id:
            raise ValueError("strategy_id is required.")
        if strategy_id not in self._strategies:
            raise KeyError(f"Unknown active-learning query strategy: {strategy_id}")
        return self._strategies[strategy_id]

    def list(self) -> List[StrategyInfo]:
        return [strategy.info() for strategy in self._strategies.values()]

    def ids(self) -> List[str]:
        return list(self._strategies.keys())

    def option_map(self) -> Dict[str, str]:
        return {info.title: info.id for info in self.list()}

    def acquire(
        self,
        pool: QueryPool,
        *,
        strategy_id: str,
        k: int,
        seed: Optional[int] = None,
        params: Optional[Mapping[str, Any]] = None,
        cancel_token: Any = None,
    ) -> QueryResult:
        strategy = self.get(strategy_id)
        total_records_seen = len(list(pool.records or []))
        candidates = strategy.acquire(
            pool,
            k=max(0, int(k or 0)),
            seed=seed,
            params=params,
            cancel_token=cancel_token,
        )
        stats = {
            "strategy_id": strategy.id,
            "strategy_title": strategy.title,
            "total_records_seen": total_records_seen,
            "exclude_row_ids_count": len(pool.excluded_row_ids),
            "ranked_count": len(candidates),
            "requested_k": int(k or 0),
            "batch_aware": bool(strategy.info().batch_aware),
        }
        self.last_rank_stats = dict(stats)
        return QueryResult(strategy_id=strategy.id, candidates=list(candidates), stats=stats)

    def rank(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        strategy_id: str,
        k: Optional[int] = None,
        exclude_row_ids: Optional[Iterable[Any]] = None,
        predictions_payload: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Backward-compatible wrapper returning ranked records."""

        pool = QueryPool(
            dataset_id=str((predictions_payload or {}).get("dataset_id") or ""),
            records=[dict(record) for record in records],
            predictions_payload=dict(predictions_payload or {}),
            excluded_row_ids={str(row_id) for row_id in (exclude_row_ids or [])},
        )
        result = self.acquire(pool, strategy_id=strategy_id, k=int(k if k is not None else 10**12), seed=seed, params=params)
        return result.records()


def create_default_strategy_registry() -> QueryStrategyRegistry:
    registry = QueryStrategyRegistry()
    registry.register(LeastConfidenceStrategy())
    registry.register(MarginStrategy())
    registry.register(EntropyStrategy())
    registry.register(RandomStrategy())
    return registry


def probability_values(record: Mapping[str, Any]) -> List[float]:
    candidates: List[Any] = [
        record.get("probabilities"),
        record.get("class_probabilities"),
        record.get("class_probability"),
        record.get("proba"),
        record.get("scores"),
    ]
    for candidate in candidates:
        values = _values_from_probability_candidate(candidate)
        if values:
            return _normalise_probabilities(values)

    prefix_values: List[float] = []
    for key, value in record.items():
        key_str = str(key)
        if key_str.startswith(("probability_", "prob_", "p_class_")):
            numeric = _safe_float(value)
            if numeric is not None:
                prefix_values.append(numeric)
    return _normalise_probabilities(prefix_values)


def _values_from_probability_candidate(candidate: Any) -> List[float]:
    if candidate is None:
        return []
    if isinstance(candidate, Mapping):
        values = [_safe_float(value) for value in candidate.values()]
        return [value for value in values if value is not None]
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
        values = [_safe_float(value) for value in candidate]
        return [value for value in values if value is not None]
    return []


def _normalise_probabilities(values: Sequence[float]) -> List[float]:
    out = [max(0.0, float(value)) for value in values]
    total = sum(out)
    if total <= 0:
        return []
    if total > 1.0 + 1.0e-6 or total < 1.0 - 1.0e-6:
        out = [value / total for value in out]
    return out


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _check_cancelled(cancel_token: Any) -> None:
    if cancel_token is None:
        return
    for attr in ("raise_if_cancelled", "throw_if_cancelled", "check_cancelled"):
        method = getattr(cancel_token, attr, None)
        if callable(method):
            method()
            return
    if getattr(cancel_token, "cancelled", False) or getattr(cancel_token, "is_cancelled", False):
        raise RuntimeError("Operation cancelled.")
