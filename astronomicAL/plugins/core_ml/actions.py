from __future__ import annotations

import importlib.util
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from astronomicAL.platform.plugins.specs import ActionRequest, ActionResult, ArtifactResult, EventResult


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


def train_tabular_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Train a tabular model through the platform action contract.

    This is safe to submit through JobManager directly while the platform action
    runner is still maturing.
    """

    _check_cancelled(cancel_token)

    ml = _load_sibling_module("ml")
    artifact_utils = _load_sibling_module("artifacts")

    request = _coerce_request(request)
    params = dict(request.params or {})
    dataset_id = _dataset_id(context, request, params)

    target_column = str(params.get("target_column") or "").strip()
    model_id = str(params.get("model_id") or "").strip()
    task = str(params.get("task") or "classification").strip()
    feature_columns = _as_str_list(params.get("feature_columns") or request.columns or [])

    if not dataset_id:
        raise ValueError("train_tabular requires a dataset_id.")
    if not target_column:
        raise ValueError("train_tabular requires params.target_column.")
    if not model_id:
        raise ValueError("train_tabular requires params.model_id.")
    if not feature_columns:
        raise ValueError("train_tabular requires params.feature_columns or request.columns.")

    registry = context.services.get("core.ml.registry")

    sync = getattr(ml, "sync_model_definitions_from_artifacts", None)
    if callable(sync):
        sync(context, registry)

    model_spec = registry.get_model(model_id)
    tuning = _model_tuning_config(model_spec)

    run_id = str(params.get("run_id") or uuid.uuid4().hex)
    optimize_metric = str(
        params.get("optimize_metric")
        or tuning.get("metric")
        or "val_loss"
    )

    config = ml.TrainConfig(
        dataset_id=dataset_id,
        task=task,
        target_column=target_column,
        feature_columns=feature_columns,
        model_id=model_id,
        test_size=float(params.get("test_size", 0.2)),
        validation_size=float(params.get("validation_size", 0.2)),
        random_state=int(params.get("random_state", 42)),
        stratify=bool(params.get("stratify", True)),
        optimize_metric=optimize_metric,
        torch_hidden_layers=str(params.get("torch_hidden_layers", "128,64")),
        torch_epochs=int(params.get("torch_epochs", 50)),
        torch_batch_size=int(params.get("torch_batch_size", 512)),
        torch_learning_rate=float(params.get("torch_learning_rate", 1e-3)),
        torch_weight_decay=float(params.get("torch_weight_decay", 0.0)),
        torch_patience=int(params.get("torch_patience", 12)),
        run_id=run_id,
        training_log_artifact_id=params.get("training_log_artifact_id"),
    )

    started = time.time()

    result = ml.train_model(
        context=context,
        registry=registry,
        config=config,
        cancel_token=cancel_token,
    )

    _check_cancelled(cancel_token)

    result = dict(result or {})
    result.setdefault("run_id", run_id)
    result.setdefault("duration_seconds", time.time() - started)

    artifact_ids = dict(result.get("artifact_ids") or {})
    model_artifact_id = artifact_ids.get("model")

    if model_artifact_id:
        normalized_model_payload = artifact_utils.persist_existing_model_artifact(
            context=context,
            artifact_id=model_artifact_id,
        )
        result["model_artifact_id"] = model_artifact_id
        result["durable_model"] = bool(normalized_model_payload.get("model_ref"))

        _publish(
            context,
            "ml.model.saved",
            {
                "artifact_id": model_artifact_id,
                "run_id": run_id,
                "model_ref": normalized_model_payload.get("model_ref"),
            },
        )
    else:
        result["model_artifact_id"] = None
        result["durable_model"] = False

    result["artifact_ids"] = artifact_ids
    result = _augment_result_from_training_log(
        context=context,
        result=result,
        training_log_artifact_id=(
            params.get("training_log_artifact_id")
            or artifact_ids.get("training_log")
        ),
        model_tuning=tuning,
    )

    return artifact_utils.json_safe(result)


def train_image_classifier_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Train a torch image classifier through the platform action contract."""

    _check_cancelled(cancel_token)

    image_ml = _load_sibling_module("image_ml")
    ml = _load_sibling_module("ml")
    artifact_utils = _load_sibling_module("artifacts")

    request = _coerce_request(request)
    params = dict(request.params or {})
    dataset_id = _dataset_id(context, request, params)

    image_column = str(params.get("image_column") or "").strip()
    target_column = str(params.get("target_column") or "").strip()
    model_id = str(params.get("model_id") or "").strip()

    if not dataset_id:
        raise ValueError("train_image_classifier requires a dataset_id.")
    if not image_column:
        raise ValueError("train_image_classifier requires params.image_column.")
    if not target_column:
        raise ValueError("train_image_classifier requires params.target_column.")
    if not model_id:
        raise ValueError("train_image_classifier requires params.model_id.")

    registry = context.services.get("core.ml.registry")

    sync = getattr(ml, "sync_model_definitions_from_artifacts", None)
    if callable(sync):
        sync(context, registry)

    model_spec = registry.get_model(model_id)
    tuning = _model_tuning_config(model_spec)

    run_id = str(params.get("run_id") or uuid.uuid4().hex)

    config = image_ml.ImageTrainConfig(
        dataset_id=dataset_id,
        image_column=image_column,
        target_column=target_column,
        model_id=model_id,
        test_size=float(params.get("test_size", 0.2)),
        validation_size=float(params.get("validation_size", 0.2)),
        random_state=int(params.get("random_state", 42)),
        stratify=bool(params.get("stratify", True)),
        run_id=run_id,
        training_log_artifact_id=params.get("training_log_artifact_id"),
    )

    started = time.time()

    result = image_ml.train_image_model(
        context=context,
        registry=registry,
        config=config,
        cancel_token=cancel_token,
    )

    _check_cancelled(cancel_token)

    result = dict(result or {})
    result.setdefault("run_id", run_id)
    result.setdefault("duration_seconds", time.time() - started)

    artifact_ids = dict(result.get("artifact_ids") or {})
    model_artifact_id = artifact_ids.get("model")

    if model_artifact_id:
        normalized_model_payload = artifact_utils.persist_existing_model_artifact(
            context=context,
            artifact_id=model_artifact_id,
        )
        result["model_artifact_id"] = model_artifact_id
        result["durable_model"] = bool(normalized_model_payload.get("model_ref"))

        _publish(
            context,
            "ml.model.saved",
            {
                "artifact_id": model_artifact_id,
                "run_id": run_id,
                "model_ref": normalized_model_payload.get("model_ref"),
            },
        )
    else:
        result["model_artifact_id"] = None
        result["durable_model"] = False

    result["artifact_ids"] = artifact_ids
    result = _augment_result_from_training_log(
        context=context,
        result=result,
        training_log_artifact_id=(
            params.get("training_log_artifact_id")
            or artifact_ids.get("training_log")
        ),
        model_tuning=tuning,
    )

    return artifact_utils.json_safe(result)


def predict_tabular_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Run sklearn tabular inference and store an ml.predictions artifact."""

    _check_cancelled(cancel_token)

    artifact_utils = _load_sibling_module("artifacts")

    request = _coerce_request(request)
    params = dict(request.params or {})
    dataset_id = _dataset_id(context, request, params)
    model_artifact_id = str(params.get("model_artifact_id") or request.artifact_id or "").strip()

    if not dataset_id:
        raise ValueError("predict_tabular requires a dataset_id.")
    if not model_artifact_id:
        raise ValueError("predict_tabular requires params.model_artifact_id or request.artifact_id.")

    model_payload = context.artifacts.get(model_artifact_id)
    if not isinstance(model_payload, Mapping):
        raise TypeError(f"Artifact {model_artifact_id!r} is not an ml.model payload.")

    if "model" in model_payload and "model_ref" not in model_payload:
        model_payload = artifact_utils.persist_existing_model_artifact(
            context=context,
            artifact_id=model_artifact_id,
        )

    framework = str(model_payload.get("framework") or "").lower()
    modality = str(model_payload.get("modality") or "tabular").lower()

    if framework != "sklearn" or modality != "tabular":
        raise NotImplementedError(
            "predict_tabular currently supports sklearn tabular ml.model artifacts only. "
            f"Got framework={framework!r}, modality={modality!r}."
        )

    model = artifact_utils.load_model_from_payload(model_payload)

    feature_columns = _as_str_list(
        params.get("feature_columns")
        or model_payload.get("feature_columns")
        or []
    )
    if not feature_columns:
        raise ValueError("Model artifact does not define feature_columns; pass params.feature_columns.")

    record_id_column = _mapped_column(context, dataset_id, "record_id")
    columns = list(
        dict.fromkeys(
            [
                *feature_columns,
                *([record_id_column] if record_id_column else []),
            ]
        )
    )

    df = context.datasets.get_df(dataset_id, columns=columns)
    df = _filter_rows(df, request, record_id_column)

    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset {dataset_id!r} is missing feature columns: {missing}")

    X = df[feature_columns]
    predictions = model.predict(X)

    probabilities = None
    classes = None
    if hasattr(model, "predict_proba") and callable(model.predict_proba):
        try:
            probabilities = model.predict_proba(X)
            classes = getattr(model, "classes_", None)
        except Exception:
            probabilities = None
            classes = None

    row_ids = _row_ids(df, record_id_column)

    records = _prediction_records(
        row_ids=row_ids,
        predictions=predictions,
        probabilities=probabilities,
        classes=classes,
    )

    payload = artifact_utils.prediction_payload(
        run_id=str(params.get("run_id") or uuid.uuid4().hex),
        dataset_id=dataset_id,
        model_artifact_id=model_artifact_id,
        model_payload=model_payload,
        records=records,
        row_ids=row_ids,
        params=params,
    )

    artifact_id = context.artifacts.put(
        artifact_utils.ARTIFACTS.PREDICTIONS,
        payload,
        dataset_id=dataset_id,
        row_ids=row_ids,
        params={"model_artifact_id": model_artifact_id, **params},
    )

    _publish(
        context,
        "ml.predictions.created",
        {
            "artifact_id": artifact_id,
            "dataset_id": dataset_id,
            "model_artifact_id": model_artifact_id,
            "count": len(row_ids),
        },
    )

    return {
        "artifact_id": artifact_id,
        "dataset_id": dataset_id,
        "model_artifact_id": model_artifact_id,
        "count": len(row_ids),
        "prediction_preview": records[:25],
    }


def create_active_learning_batch_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Rank prediction records by uncertainty and optionally create a selection set."""

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
            "create_active_learning_batch requires params.predictions_artifact_id or request.artifact_id."
        )

    predictions_payload = context.artifacts.get(predictions_artifact_id)
    if not isinstance(predictions_payload, Mapping):
        raise TypeError(f"Artifact {predictions_artifact_id!r} is not an ml.predictions payload.")

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


def _dataset_id(context: Any, request: ActionRequest, params: Mapping[str, Any]) -> Optional[str]:
    dataset_id = params.get("dataset_id") or request.dataset_id
    if dataset_id:
        return str(dataset_id)

    try:
        return str(context.datasets.active_id())
    except Exception:
        return None


def _model_tuning_config(model_spec: Any) -> Dict[str, Any]:
    metadata = getattr(model_spec, "metadata", {}) or {}
    tuning = metadata.get("tuning") or {}
    if not isinstance(tuning, dict):
        return {}
    if not tuning.get("enabled"):
        return {}
    if not tuning.get("search_space"):
        return {}
    return dict(tuning)


def _augment_result_from_training_log(
    *,
    context: Any,
    result: Dict[str, Any],
    training_log_artifact_id: Optional[str],
    model_tuning: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    tuning_config = dict(model_tuning or {})
    tuning_trials: List[Dict[str, Any]] = []
    best_params: Dict[str, Any] = {}
    best_value = None
    status = None
    message = None

    if training_log_artifact_id:
        try:
            payload = context.artifacts.get(training_log_artifact_id)
        except Exception:
            payload = None

        if isinstance(payload, Mapping):
            status = payload.get("status")
            message = payload.get("message")
            tuning_trials = list(payload.get("tuning_trials") or [])
            best_params = dict(
                payload.get("tuning_best_params")
                or payload.get("best_params")
                or {}
            )
            best_value = (
                payload.get("tuning_best_value")
                if payload.get("tuning_best_value") is not None
                else payload.get("best_value")
            )

            if not tuning_config:
                maybe_tuning = payload.get("tuning")
                if isinstance(maybe_tuning, Mapping):
                    tuning_config = dict(maybe_tuning)

    if not best_params and tuning_trials:
        completed = [
            trial
            for trial in tuning_trials
            if trial.get("value") is not None and str(trial.get("state", "complete")).lower() == "complete"
        ]
        if completed:
            metric = str(tuning_config.get("metric") or completed[0].get("metric") or "")
            minimize = any(token in metric.lower() for token in ("loss", "error", "mae", "mse", "rmse"))
            best = min(completed, key=lambda t: float(t["value"])) if minimize else max(completed, key=lambda t: float(t["value"]))
            best_params = dict(best.get("params") or {})
            best_value = best.get("value")

    result["tuning"] = {
        "enabled": bool(tuning_config or tuning_trials),
        "backend": tuning_config.get("backend", "optuna") if tuning_config or tuning_trials else None,
        "metric": tuning_config.get("metric"),
        "direction": tuning_config.get("direction"),
        "n_trials": tuning_config.get("n_trials"),
        "trial_epochs": tuning_config.get("trial_epochs"),
        "search_space": tuning_config.get("search_space", {}),
        "best_params": best_params,
        "best_value": best_value,
        "trials": tuning_trials,
        "status": status,
        "message": message,
        "training_log_artifact_id": training_log_artifact_id,
    }

    return result


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _mapped_column(context: Any, dataset_id: str, mapping: str) -> Optional[str]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None

    for method_name in ("get_mapping", "mapping", "get_column_mapping"):
        method = getattr(datasets, method_name, None)
        if not callable(method):
            continue

        try:
            value = method(dataset_id, mapping)
        except Exception:
            value = None

        if value:
            return str(value)

    return None


def _filter_rows(df: pd.DataFrame, request: ActionRequest, record_id_column: Optional[str]) -> pd.DataFrame:
    row_ids = list(request.row_ids or [])
    if not row_ids:
        return df

    if record_id_column and record_id_column in df.columns:
        wanted = {str(row_id) for row_id in row_ids}
        return df[df[record_id_column].astype(str).isin(wanted)]

    try:
        return df.loc[row_ids]
    except Exception:
        wanted = {str(row_id) for row_id in row_ids}
        return df[df.index.astype(str).isin(wanted)]


def _row_ids(df: pd.DataFrame, record_id_column: Optional[str]) -> List[str]:
    if record_id_column and record_id_column in df.columns:
        return [str(value) for value in df[record_id_column].tolist()]
    return [str(value) for value in df.index.tolist()]


def _prediction_records(
    *,
    row_ids: Sequence[str],
    predictions: Any,
    probabilities: Any = None,
    classes: Any = None,
) -> List[Dict[str, Any]]:
    preds = np.asarray(predictions).reshape(-1)
    prob_array = None if probabilities is None else np.asarray(probabilities)

    class_values = None
    if classes is not None:
        class_values = [str(c) for c in list(classes)]

    records: List[Dict[str, Any]] = []

    for idx, row_id in enumerate(row_ids):
        record: Dict[str, Any] = {
            "row_id": str(row_id),
            "prediction": _json_scalar(preds[idx] if idx < len(preds) else None),
        }

        if prob_array is not None and idx < prob_array.shape[0]:
            probs = np.asarray(prob_array[idx], dtype=float).reshape(-1)
            record["probabilities"] = [float(p) for p in probs]

            if class_values and len(class_values) == len(probs):
                record["probabilities_by_class"] = {
                    class_values[i]: float(probs[i]) for i in range(len(probs))
                }

            if probs.size:
                sorted_probs = np.sort(probs)[::-1]
                record["max_probability"] = float(sorted_probs[0])
                record["least_confidence"] = float(1.0 - sorted_probs[0])

                if sorted_probs.size >= 2:
                    record["margin"] = float(sorted_probs[0] - sorted_probs[1])
                    record["margin_uncertainty"] = float(1.0 - record["margin"])

                record["entropy"] = _entropy(probs)

        records.append(record)

    return records


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


def _entropy(probs: np.ndarray) -> float:
    clean = np.asarray([p for p in probs if p > 0.0], dtype=float)
    if clean.size == 0:
        return 0.0
    return float(-np.sum(clean * np.log(clean)))


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


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