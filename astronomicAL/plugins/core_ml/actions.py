from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from astronomicAL.platform.plugins.specs import ActionRequest


def _load_sibling_module(stem: str):
    module_name = f"{__name__}.{stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = Path(__file__).with_name(f"{stem}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sibling module {stem!r} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def create_active_learning_batch_action(
    context: Any,
    request: Any,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    """Rank prediction records by uncertainty and optionally create a selection set.

    This action is generation-agnostic: it consumes an ml.predictions artifact
    (from the predict action or a harness) and produces an
    ml.active_learning_batch artifact, optionally promoting the most uncertain
    rows to the platform selection set for review/annotation.
    """

    _check_cancelled(cancel_token)

    artifact_utils = _load_sibling_module("artifacts")

    request = _coerce_request(request)
    params = dict(request.params or {})

    predictions_artifact_id = str(
        params.get("predictions_artifact_id")
        or request.artifact_id
        or ""
    ).strip()

    if not predictions_artifact_id:
        raise ValueError(
            "create_active_learning_batch requires params.predictions_artifact_id "
            "or request.artifact_id."
        )

    predictions_payload = context.artifacts.get(predictions_artifact_id)
    if not isinstance(predictions_payload, Mapping):
        raise TypeError(
            f"Artifact {predictions_artifact_id!r} is not an ml.predictions payload."
        )

    dataset_id = str(
        params.get("dataset_id")
        or predictions_payload.get("dataset_id")
        or request.dataset_id
        or ""
    )

    if not dataset_id:
        raise ValueError("Could not determine dataset_id for active-learning batch.")

    strategy = str(params.get("strategy") or "least_confidence")
    k = int(params.get("k", 50))
    make_selection = bool(params.get("make_selection", True))

    source_records = list(predictions_payload.get("records") or [])
    ranked = _rank_prediction_records(source_records, strategy=strategy)
    selected = ranked[: max(0, k)]
    row_ids = [str(row["row_id"]) for row in selected if row.get("row_id") is not None]

    payload = artifact_utils.active_learning_batch_payload(
        dataset_id=dataset_id,
        predictions_artifact_id=predictions_artifact_id,
        strategy=strategy,
        records=selected,
        params=params,
    )

    artifact_id = context.artifacts.put(
        artifact_utils.ARTIFACTS.ACTIVE_LEARNING_BATCH,
        payload,
        dataset_id=dataset_id,
        row_ids=row_ids,
        params=params,
    )

    if make_selection and row_ids and hasattr(context, "selection"):
        context.selection.set_selection_set(
            dataset_id=dataset_id,
            row_ids=row_ids,
            origin="core.ml.active_learning_batch",
            mode="replace",
            metadata={
                "strategy": strategy,
                "predictions_artifact_id": predictions_artifact_id,
                "active_learning_batch_artifact_id": artifact_id,
                "count": len(row_ids),
            },
            create_artifact=True,
            update_focus_policy="first",
        )

    _publish(
        context,
        "ml.active_learning_batch.created",
        {
            "artifact_id": artifact_id,
            "dataset_id": dataset_id,
            "predictions_artifact_id": predictions_artifact_id,
            "strategy": strategy,
            "count": len(row_ids),
        },
    )

    return {
        "artifact_id": artifact_id,
        "dataset_id": dataset_id,
        "predictions_artifact_id": predictions_artifact_id,
        "strategy": strategy,
        "row_ids": row_ids,
        "preview": selected[:25],
    }


def _rank_prediction_records(records: Iterable[Any], *, strategy: str) -> List[Dict[str, Any]]:
    strategy = strategy.lower().strip()
    ranked: List[Dict[str, Any]] = []

    for raw in records:
        if not isinstance(raw, Mapping):
            continue

        record = dict(raw)
        score = _uncertainty_score(record, strategy=strategy)
        if score is None:
            continue

        record["active_learning_score"] = float(score)
        record["active_learning_strategy"] = strategy
        ranked.append(record)

    ranked.sort(key=lambda row: row.get("active_learning_score", float("-inf")), reverse=True)
    return ranked


def _uncertainty_score(record: Mapping[str, Any], *, strategy: str) -> Optional[float]:
    if strategy in {"least_confidence", "least-confidence", "confidence"}:
        value = record.get("least_confidence")
        if value is not None:
            return float(value)

        max_probability = record.get("max_probability")
        if max_probability is not None:
            return 1.0 - float(max_probability)

        # Harness-emitted records expose `confidence`/`uncertainty` instead of
        # `least_confidence`/`max_probability`. Fall back to those so AL works
        # on both prediction shapes.
        confidence = record.get("confidence")
        if confidence is not None:
            return 1.0 - float(confidence)

        uncertainty = record.get("uncertainty")
        if uncertainty is not None:
            return float(uncertainty)

    if strategy in {"margin", "smallest_margin", "margin_uncertainty"}:
        value = record.get("margin_uncertainty")
        if value is not None:
            return float(value)

        margin = record.get("margin")
        if margin is not None:
            return 1.0 - float(margin)

    if strategy == "entropy":
        value = record.get("entropy")
        if value is not None:
            return float(value)

    return None


def _coerce_request(request: Any) -> ActionRequest:
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


def _check_cancelled(cancel_token: Any) -> None:
    if cancel_token is None:
        return

    for attr in ("raise_if_cancelled", "throw_if_cancelled", "check_cancelled"):
        method = getattr(cancel_token, attr, None)
        if callable(method):
            method()
            return

    for attr in ("cancelled", "is_cancelled", "cancel_requested"):
        value = getattr(cancel_token, attr, None)
        try:
            cancelled = value() if callable(value) else bool(value)
        except Exception:
            cancelled = False

        if cancelled:
            raise RuntimeError("ML action cancelled.")


def _publish(context: Any, topic: str, payload: Mapping[str, Any]) -> None:
    events = getattr(context, "events", None)
    publish = getattr(events, "publish", None)
    if callable(publish):
        publish(topic, dict(payload))