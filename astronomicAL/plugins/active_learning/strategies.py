from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import random
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

class LearningLossStrategy(PerRecordScoringStrategy):
    id = "learning_loss"
    title = "Learning loss"
    description = "Ranks rows by a model-emitted predicted/expected loss score. Requires a learning-loss head or loss-prediction output in the prediction artifact."
    required_prediction_fields = ("learning_loss",)
    tags = ("loss", "classification", "regression")

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
        missing_row_ids: List[str] = []
        invalid_row_ids: List[str] = []
        for raw in pool.records:
            _check_cancelled(cancel_token)
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            row_id = str(row.get("row_id") or row.get("id") or "").strip()
            if not row_id or row_id in pool.excluded_row_ids:
                continue
            score, source = self.score_with_source(row)
            if score is None:
                missing_row_ids.append(row_id)
                continue
            if math.isnan(float(score)) or math.isinf(float(score)):
                invalid_row_ids.append(row_id)
                continue
            row["_al_score_source"] = source
            candidates.append(QueryCandidate(row_id=row_id, score=float(score), metadata=row))
        if missing_row_ids or invalid_row_ids:
            parts: List[str] = []
            if missing_row_ids:
                parts.append(f"missing learning-loss output for {len(missing_row_ids)} rows")
            if invalid_row_ids:
                parts.append(f"invalid learning-loss output for {len(invalid_row_ids)} rows")
            raise ValueError(
                "Learning Loss requires a model-emitted per-row loss prediction for every eligible pool row; "
                + ", ".join(parts)
                + ". Generate pool predictions with a learning-loss/loss-prediction head before using this strategy."
            )
        candidates.sort(key=lambda item: (-float(item.score), str(item.row_id)))
        return candidates[: max(0, int(k))]

    def score(self, record: Mapping[str, Any], **kwargs: Any) -> Optional[float]:
        return self.score_with_source(record)[0]

    def score_with_source(self, record: Mapping[str, Any]) -> tuple[Optional[float], str]:
        for key in (
            "learning_loss",
            "learning_loss_score",
            "predicted_loss",
            "loss_prediction",
            "expected_loss",
            "expected_error",
            "loss_head",
            "loss_head_score",
        ):
            value = _safe_float(record.get(key))
            if value is not None:
                return value, key
        return None, "missing_learning_loss"

class BADGEStrategy(QueryStrategy):
    id = "badge"
    title = "BADGE"
    description = "Batch-aware k-means++ selection over BADGE gradient embeddings. Requires explicit gradient embeddings or probabilities plus model embeddings/features."
    requires_probabilities = True
    batch_aware = True
    required_prediction_fields = ("probabilities", "embedding")
    tags = ("batch", "diversity", "classification")

    def acquire(
        self,
        pool: QueryPool,
        *,
        k: int,
        params: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        cancel_token: Any = None,
    ) -> List[QueryCandidate]:
        rows = _candidate_rows(pool, cancel_token=cancel_token)
        if not rows:
            return []
        vectors: Dict[str, List[float]] = {}
        gradient_norms: Dict[str, float] = {}
        score_sources: Dict[str, str] = {}
        missing_row_ids: List[str] = []
        rows_by_id = {str(row.get("row_id") or row.get("id") or "").strip(): row for row in rows}

        expected_width: Optional[int] = None
        for row in rows:
            row_id = str(row.get("row_id") or row.get("id") or "").strip()
            if not row_id:
                continue
            vector, source = badge_gradient_embedding(row)
            if not vector:
                missing_row_ids.append(row_id)
                continue
            if expected_width is None:
                expected_width = len(vector)
            elif len(vector) != expected_width:
                raise ValueError(
                    "BADGE requires consistent gradient-embedding dimensions across the eligible pool; "
                    f"row {row_id!r} has dimension {len(vector)}, expected {expected_width}."
                )
            vectors[row_id] = vector
            gradient_norms[row_id] = _vector_norm(vector)
            score_sources[row_id] = source

        if missing_row_ids:
            raise ValueError(
                "BADGE requires a gradient embedding for every eligible pool row. Provide `badge_embedding`/`gradient_embedding`, "
                "or emit both class probabilities and model embeddings/features so the BADGE gradient embedding can be computed. "
                f"Missing usable inputs for {len(missing_row_ids)} of {len(rows)} rows."
            )
        if not vectors:
            raise ValueError("BADGE found no eligible rows with usable gradient embeddings.")

        selected_ids = _badge_kmeans_pp(vectors, gradient_norms, k=max(0, int(k)), seed=seed)
        candidates: List[QueryCandidate] = []
        for row_id in selected_ids:
            metadata = dict(rows_by_id.get(row_id, {}))
            metadata["_al_score_source"] = score_sources.get(row_id, "badge_gradient_embedding")
            metadata["_al_badge_gradient_norm"] = float(gradient_norms.get(row_id, 0.0))
            candidates.append(QueryCandidate(row_id=row_id, score=float(gradient_norms.get(row_id, 0.0)), metadata=metadata))
        return candidates

class CoreSetStrategy(QueryStrategy):
    id = "coreset"
    title = "Core-set"
    description = "Batch-aware farthest-first selection in embedding/feature space, seeded by labelled examples when available."
    batch_aware = True
    required_prediction_fields = ("embedding",)
    tags = ("batch", "diversity", "embedding")

    def acquire(
        self,
        pool: QueryPool,
        *,
        k: int,
        params: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        cancel_token: Any = None,
    ) -> List[QueryCandidate]:
        rows = _candidate_rows(pool, cancel_token=cancel_token)
        if not rows:
            return []
        candidate_vectors: Dict[str, List[float]] = {}
        labelled_vectors: List[List[float]] = []
        all_records = [dict(record) for record in (pool.records or []) if isinstance(record, Mapping)]
        for row in all_records:
            row_id = str(row.get("row_id") or row.get("id") or "").strip()
            vector = embedding_values(row)
            if not row_id or not vector:
                continue
            if row_id in pool.labelled_row_ids:
                labelled_vectors.append(vector)
        for row in rows:
            row_id = str(row.get("row_id") or row.get("id") or "").strip()
            vector = embedding_values(row)
            if row_id and vector:
                candidate_vectors[row_id] = vector
        if not candidate_vectors:
            candidates = RandomStrategy().acquire(pool, k=k, params=params, seed=seed, cancel_token=cancel_token)
            return [QueryCandidate(row_id=item.row_id, score=item.score, metadata={**dict(item.metadata or {}), "_al_score_source": "random_fallback_no_embeddings"}) for item in candidates]
        selected_ids = _coreset_farthest_first(candidate_vectors, labelled_vectors, k=max(0, int(k)), seed=seed)
        rows_by_id = {str(row.get("row_id") or row.get("id") or ""): row for row in rows}
        scores = _distance_scores(candidate_vectors, labelled_vectors)
        candidates: List[QueryCandidate] = []
        for row_id in selected_ids:
            metadata = dict(rows_by_id.get(row_id, {}))
            metadata["_al_score_source"] = "embedding_distance_to_labelled" if labelled_vectors else "embedding_distance_to_pool_centroid"
            candidates.append(QueryCandidate(row_id=row_id, score=float(scores.get(row_id, 0.0)), metadata=metadata))
        return candidates

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
        eligible_count = 0
        for raw in pool.records or []:
            if not isinstance(raw, Mapping):
                continue
            row_id = str(raw.get("row_id") or raw.get("id") or "").strip()
            if row_id and row_id not in pool.excluded_row_ids:
                eligible_count += 1
        score_sources: Dict[str, int] = {}
        for candidate in candidates:
            metadata = dict(candidate.metadata or {})
            source = str(metadata.get("_al_score_source") or "direct_score")
            score_sources[source] = score_sources.get(source, 0) + 1
        info = strategy.info()
        stats = {
            "strategy_id": strategy.id,
            "strategy_title": strategy.title,
            "strategy_description": info.description,
            "total_records_seen": total_records_seen,
            "eligible_pool_count": eligible_count,
            "scored_pool_count": eligible_count,
            "selection_pool_used_count": eligible_count,
            "exclude_row_ids_count": len(pool.excluded_row_ids),
            "ranked_count": len(candidates),
            "selected_count": len(candidates),
            "requested_k": int(k or 0),
            "batch_aware": bool(info.batch_aware),
            "requires_probabilities": bool(info.requires_probabilities),
            "required_prediction_fields": list(info.required_prediction_fields or ()),
            "score_source_counts": score_sources,
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
    registry.register(LearningLossStrategy())
    registry.register(BADGEStrategy())
    registry.register(CoreSetStrategy())
    return registry

def _candidate_rows(pool: QueryPool, *, cancel_token: Any = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in pool.records or []:
        _check_cancelled(cancel_token)
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        row_id = str(row.get("row_id") or row.get("id") or "").strip()
        if row_id and row_id not in pool.excluded_row_ids:
            rows.append(row)
    return rows

def embedding_values(record: Mapping[str, Any]) -> List[float]:
    for key in ("embedding", "embeddings", "latent", "representation", "features", "feature_vector", "penultimate", "image_embedding"):
        values = _values_from_vector_candidate(record.get(key))
        if values:
            return values
    prefix_values: List[tuple[str, float]] = []
    for key, value in record.items():
        name = str(key)
        if name.startswith(("embedding_", "embed_", "latent_", "feature_", "feat_")):
            numeric = _safe_float(value)
            if numeric is not None:
                prefix_values.append((name, numeric))
    prefix_values.sort(key=lambda item: item[0])
    return [value for _, value in prefix_values]

def gradient_embedding_values(record: Mapping[str, Any]) -> List[float]:
    for key in ("badge_embedding", "gradient_embedding", "grad_embedding", "expected_gradient", "loss_gradient", "classification_gradient"):
        values = _values_from_vector_candidate(record.get(key))
        if values:
            return values
    prefix_values: List[tuple[str, float]] = []
    for key, value in record.items():
        name = str(key)
        if name.startswith(("badge_", "grad_embedding_", "gradient_")):
            numeric = _safe_float(value)
            if numeric is not None:
                prefix_values.append((name, numeric))
    prefix_values.sort(key=lambda item: item[0])
    return [value for _, value in prefix_values]

def badge_gradient_embedding(record: Mapping[str, Any]) -> tuple[List[float], str]:
    explicit = gradient_embedding_values(record)
    if explicit:
        return explicit, "gradient_embedding"
    computed = compute_badge_gradient_embedding(record)
    if computed:
        return computed, "computed_badge_gradient"
    return [], "missing_badge_inputs"

def compute_badge_gradient_embedding(record: Mapping[str, Any]) -> List[float]:
    """Compute the BADGE hallucinated gradient embedding from probabilities and features.

    For a classification model with final linear layer, BADGE uses the gradient of
    the cross-entropy loss for the model's predicted class.  With predicted
    probabilities ``p`` and penultimate representation ``h``, the block for class
    ``c`` is ``(p_c - 1[c == argmax(p)]) * h``.
    """

    features = embedding_values(record)
    probs = probability_values(record)
    if not features or len(probs) < 2:
        return []
    pred_index = max(range(len(probs)), key=lambda index: probs[index])
    out: List[float] = []
    for class_index, prob in enumerate(probs):
        coeff = float(prob) - (1.0 if class_index == pred_index else 0.0)
        out.extend(coeff * float(value) for value in features)
    return out

def approximate_badge_gradient(record: Mapping[str, Any]) -> List[float]:
    # Backward-compatible alias.  This is not a fallback strategy; it is the
    # standard BADGE gradient embedding computed from probabilities and model
    # representations when an explicit gradient vector was not persisted.
    return compute_badge_gradient_embedding(record)

def _values_from_vector_candidate(candidate: Any) -> List[float]:
    if candidate is None:
        return []
    if isinstance(candidate, Mapping):
        items = sorted(candidate.items(), key=lambda item: str(item[0]))
        values = [_safe_float(value) for _, value in items]
        return [value for value in values if value is not None]
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
        values = [_safe_float(value) for value in candidate]
        return [value for value in values if value is not None]
    return []

def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))

def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    length = min(len(left), len(right))
    if length <= 0:
        return 0.0
    total = 0.0
    for idx in range(length):
        delta = float(left[idx]) - float(right[idx])
        total += delta * delta
    return total

def _distance_scores(vectors: Mapping[str, Sequence[float]], labelled_vectors: Sequence[Sequence[float]]) -> Dict[str, float]:
    if not vectors:
        return {}
    if not labelled_vectors:
        centroid = _centroid(list(vectors.values()))
        return {row_id: math.sqrt(_squared_distance(vector, centroid)) for row_id, vector in vectors.items()}
    return {row_id: math.sqrt(min(_squared_distance(vector, labelled) for labelled in labelled_vectors)) for row_id, vector in vectors.items()}

def _centroid(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    width = min(len(vector) for vector in vectors if vector)
    if width <= 0:
        return []
    return [sum(float(vector[idx]) for vector in vectors) / float(len(vectors)) for idx in range(width)]

def _coreset_farthest_first(vectors: Mapping[str, Sequence[float]], labelled_vectors: Sequence[Sequence[float]], *, k: int, seed: Optional[int] = None) -> List[str]:
    remaining = set(vectors.keys())
    selected: List[str] = []
    distances = _distance_scores(vectors, labelled_vectors)
    while remaining and len(selected) < k:
        row_id = max(remaining, key=lambda candidate: (float(distances.get(candidate, 0.0)), _stable_random(candidate, seed)))
        selected.append(row_id)
        remaining.remove(row_id)
        vector = vectors[row_id]
        for candidate in list(remaining):
            distances[candidate] = min(float(distances.get(candidate, float("inf"))), math.sqrt(_squared_distance(vectors[candidate], vector)))
    return selected


def _badge_kmeans_pp(vectors: Mapping[str, Sequence[float]], weights: Mapping[str, float], *, k: int, seed: Optional[int] = None) -> List[str]:
    """BADGE k-means++ initialisation over gradient embeddings.

    The first centre is the largest gradient-norm point, matching common BADGE
    implementations.  Subsequent centres are sampled with probability
    proportional to squared distance from the nearest selected centre.  The seed
    keeps the stochastic k-means++ step reproducible.
    """

    if not vectors or k <= 0:
        return []
    remaining = set(vectors.keys())
    first = max(remaining, key=lambda candidate: (float(weights.get(candidate, 0.0)), _stable_random(candidate, seed)))
    selected = [first]
    remaining.remove(first)
    rng = random.Random(int(seed or 0))

    while remaining and len(selected) < k:
        distances = {
            candidate: min(_squared_distance(vectors[candidate], vectors[chosen]) for chosen in selected)
            for candidate in remaining
        }
        total = sum(max(0.0, float(value)) for value in distances.values())
        if total <= 0.0:
            next_id = max(remaining, key=lambda candidate: (float(weights.get(candidate, 0.0)), _stable_random(candidate, seed)))
        else:
            threshold = rng.random() * total
            cumulative = 0.0
            next_id = ""
            for candidate in sorted(remaining):
                cumulative += max(0.0, float(distances.get(candidate, 0.0)))
                if cumulative >= threshold:
                    next_id = candidate
                    break
            if not next_id:
                next_id = max(remaining, key=lambda candidate: (float(distances.get(candidate, 0.0)), _stable_random(candidate, seed)))
        selected.append(next_id)
        remaining.remove(next_id)
    return selected

def _weighted_farthest_first(vectors: Mapping[str, Sequence[float]], weights: Mapping[str, float], *, k: int, seed: Optional[int] = None) -> List[str]:
    if not vectors or k <= 0:
        return []
    remaining = set(vectors.keys())
    first = max(remaining, key=lambda candidate: (float(weights.get(candidate, 0.0)), _stable_random(candidate, seed)))
    selected = [first]
    remaining.remove(first)
    min_distances = {candidate: math.sqrt(_squared_distance(vectors[candidate], vectors[first])) for candidate in remaining}
    while remaining and len(selected) < k:
        row_id = max(remaining, key=lambda candidate: (float(min_distances.get(candidate, 0.0)) * (1.0 + float(weights.get(candidate, 0.0))), _stable_random(candidate, seed)))
        selected.append(row_id)
        remaining.remove(row_id)
        for candidate in list(remaining):
            min_distances[candidate] = min(float(min_distances.get(candidate, float("inf"))), math.sqrt(_squared_distance(vectors[candidate], vectors[row_id])))
    return selected

def _stable_random(row_id: str, seed: Optional[int]) -> float:
    raw = f"{int(seed or 0)}:{row_id}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:16]
    return int(digest, 16) / float(0xFFFFFFFFFFFFFFFF)

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
