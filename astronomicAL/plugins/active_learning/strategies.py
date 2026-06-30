from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

@dataclass(frozen=True)
class StrategyInfo:
    id: str
    title: str
    description: str = ""
    requires_probabilities: bool = False
    deterministic: bool = True

class QueryStrategy:
    """Base class for active-learning query strategies.

    Add new research strategies by subclassing this class and registering an
    instance with:

        context.services.get("core.active_learning.query_strategy_registry").register(
            MyStrategy()
        )

    A strategy returns a scalar informativeness score. Larger means "review
    sooner". The registry handles exclusion, stable sorting, rank assignment,
    and payload enrichment.
    """

    id = "base"
    title = "Base strategy"
    description = ""
    requires_probabilities = False
    deterministic = True

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            id=self.id,
            title=self.title,
            description=self.description,
            requires_probabilities=bool(self.requires_probabilities),
            deterministic=bool(self.deterministic),
        )

    def score(
        self,
        record: Mapping[str, Any],
        *,
        predictions_payload: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Optional[float]:
        raise NotImplementedError

class LeastConfidenceStrategy(QueryStrategy):
    id = "least_confidence"
    title = "Least confidence"
    description = "Ranks rows by 1 - max class probability."
    requires_probabilities = True

    def score(
        self,
        record: Mapping[str, Any],
        *,
        predictions_payload: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Optional[float]:
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
        if not probs:
            return _safe_float(record.get("uncertainty"))

        return 1.0 - max(probs)

class MarginStrategy(QueryStrategy):
    id = "margin"
    title = "Smallest margin"
    description = "Ranks rows where the top two class probabilities are closest."
    requires_probabilities = True

    def score(
        self,
        record: Mapping[str, Any],
        *,
        predictions_payload: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Optional[float]:
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

class EntropyStrategy(QueryStrategy):
    id = "entropy"
    title = "Entropy"
    description = "Ranks rows by predictive entropy."
    requires_probabilities = True

    def score(
        self,
        record: Mapping[str, Any],
        *,
        predictions_payload: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Optional[float]:
        value = _safe_float(record.get("entropy"))
        if value is not None:
            return value

        probs = probability_values(record)
        if not probs:
            return None

        return float(-sum(p * math.log(max(p, 1.0e-12)) for p in probs))

class RandomStrategy(QueryStrategy):
    id = "random"
    title = "Random"
    description = "Ranks rows with a deterministic row-id/seed hash."
    requires_probabilities = False
    deterministic = True

    def score(
        self,
        record: Mapping[str, Any],
        *,
        predictions_payload: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Optional[float]:
        row_id = str(record.get("row_id") or record.get("id") or "")
        raw = f"{int(seed or 0)}:{row_id}".encode("utf-8")
        digest = hashlib.sha1(raw).hexdigest()[:16]
        return int(digest, 16) / float(0xFFFFFFFFFFFFFFFF)

class QueryStrategyRegistry:
    def __init__(self) -> None:
        self._strategies: Dict[str, QueryStrategy] = {}

    def register(self, strategy: QueryStrategy, *, replace: bool = False) -> None:
        strategy_id = str(getattr(strategy, "id", "") or "").strip()
        if not strategy_id:
            raise ValueError("QueryStrategy.id must be a non-empty string.")
        if strategy_id in self._strategies and not replace:
            raise ValueError(f"Query strategy already registered: {strategy_id}")
        self._strategies[strategy_id] = strategy

    def get(self, strategy_id: str) -> QueryStrategy:
        strategy_id = str(strategy_id or "").strip()
        if not strategy_id:
            raise ValueError("strategy_id is required.")
        if strategy_id not in self._strategies:
            raise KeyError(f"Unknown active-learning query strategy: {strategy_id}")
        return self._strategies[strategy_id]

    def list(self) -> List[StrategyInfo]:
        return [strategy.info() for strategy in self._strategies.values()]

    def option_map(self) -> Dict[str, str]:
        return {info.title: info.id for info in self.list()}

    def ids(self) -> List[str]:
        return list(self._strategies.keys())

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
        strategy = self.get(strategy_id)
        exclude = {str(row_id) for row_id in (exclude_row_ids or [])}

        ranked: List[Dict[str, Any]] = []
        total_seen = 0
        excluded_count = 0
        missing_row_id_count = 0
        unscorable_count = 0

        for raw in records:
            total_seen += 1
            if not isinstance(raw, Mapping):
                unscorable_count += 1
                continue

            row = dict(raw)
            row_id = row.get("row_id")
            if row_id is None:
                row_id = row.get("id")
            if row_id is None:
                missing_row_id_count += 1
                continue

            row_id = str(row_id)
            if row_id in exclude:
                excluded_count += 1
                continue

            score = strategy.score(
                row,
                predictions_payload=predictions_payload,
                seed=seed,
                params=params,
            )
            score = _safe_float(score)
            if score is None or math.isnan(score):
                unscorable_count += 1
                continue

            row["row_id"] = row_id
            row["informativeness_score"] = float(score)
            row["active_learning_score"] = float(score)
            row["active_learning_strategy"] = strategy.id
            ranked.append(row)

        ranked.sort(
            key=lambda item: (
                -float(item.get("informativeness_score", float("-inf"))),
                str(item.get("row_id") or ""),
            )
        )

        if k is not None:
            ranked = ranked[: max(0, int(k))]

        for idx, row in enumerate(ranked, start=1):
            row["selection_rank"] = idx
            row["rank"] = idx

        self.last_rank_stats = {
            "strategy_id": strategy.id,
            "total_records_seen": total_seen,
            "excluded_count": excluded_count,
            "missing_row_id_count": missing_row_id_count,
            "unscorable_count": unscorable_count,
            "ranked_count": len(ranked),
            "requested_k": None if k is None else int(k),
        }
        return ranked

def create_default_strategy_registry() -> QueryStrategyRegistry:
    registry = QueryStrategyRegistry()
    registry.register(LeastConfidenceStrategy())
    registry.register(MarginStrategy())
    registry.register(EntropyStrategy())
    registry.register(RandomStrategy())
    return registry

def probability_values(record: Mapping[str, Any]) -> List[float]:
    """Extract class probabilities from the shapes used by core_ml predictions."""

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
        if (
            key_str.startswith("probability_")
            or key_str.startswith("prob_")
            or key_str.startswith("p_class_")
        ):
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
