from __future__ import annotations

import importlib.util
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

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

def predict_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Unified model inference action for trained ml.model artifacts.

    This replaces the old sklearn-only `predict_tabular_action` flow with a
    compatibility-aware dispatcher. It currently runs sklearn tabular models and
    torch tabular models saved by the prototype backend. Image prediction is
    intentionally blocked until image sidecars store enough architecture details
    to reconstruct ResNet models safely.
    """

    _check_cancelled(cancel_token)
    artifact_utils = _load_sibling_module("artifacts")
    contract_utils = _load_sibling_module("model_contract")

    request = _coerce_request(request)
    params = dict(request.params or {})
    dataset_id = _dataset_id(context, request, params)
    model_artifact_id = str(params.get("model_artifact_id") or request.artifact_id or "").strip()
    if not dataset_id:
        raise ValueError("predict requires a dataset_id.")
    if not model_artifact_id:
        raise ValueError("predict requires params.model_artifact_id or request.artifact_id.")

    model_payload = context.artifacts.get(model_artifact_id)
    if not isinstance(model_payload, Mapping):
        raise TypeError(f"Artifact {model_artifact_id!r} is not an ml.model payload.")

    if "model" in model_payload and "model_ref" not in model_payload:
        model_payload = artifact_utils.persist_existing_model_artifact(
            context=context,
            artifact_id=model_artifact_id,
        )

    compatibility = contract_utils.validate_model_for_dataset(
        context=context,
        model_artifact_id=model_artifact_id,
        dataset_id=dataset_id,
        target_column=params.get("target_column"),
        image_column=params.get("image_column"),
        feature_column_mapping=params.get("feature_column_mapping") or {},
        require_target_compatible=bool(params.get("require_target_compatible", False)),
    )
    if not compatibility.can_predict:
        return {
            "ok": False,
            "status": compatibility.status,
            "dataset_id": dataset_id,
            "model_artifact_id": model_artifact_id,
            "compatibility_report": compatibility.to_dict(),
            "errors": list(compatibility.errors),
            "warnings": list(compatibility.warnings),
        }

    contract = contract_utils.ensure_model_contract(
        context=context,
        model_artifact_id=model_artifact_id,
        model_payload=model_payload,
        persist=True,
    )
    framework = str(contract.get("framework") or model_payload.get("framework") or "").lower()
    modality = str(contract.get("modality") or model_payload.get("modality") or "tabular").lower()
    task = str(contract.get("task") or model_payload.get("task") or "classification").lower()

    run_id = str(params.get("run_id") or uuid.uuid4().hex)
    started = time.time()
    if modality == "tabular" and framework == "sklearn":
        result = _predict_sklearn_tabular(
            context=context,
            dataset_id=dataset_id,
            model_artifact_id=model_artifact_id,
            model_payload=model_payload,
            compatibility=compatibility,
            request=request,
            cancel_token=cancel_token,
        )
    elif modality == "tabular" and framework == "torch":
        result = _predict_torch_tabular(
            context=context,
            dataset_id=dataset_id,
            model_artifact_id=model_artifact_id,
            model_payload=model_payload,
            compatibility=compatibility,
            request=request,
            cancel_token=cancel_token,
        )
    elif modality == "image" and framework == "torch":
        result = _predict_torch_image_classifier(
            context=context,
            dataset_id=dataset_id,
            model_artifact_id=model_artifact_id,
            model_payload=model_payload,
            compatibility=compatibility,
            request=request,
            cancel_token=cancel_token,
        )
    else:
        raise NotImplementedError(
            f"Prediction is not implemented for framework={framework!r}, modality={modality!r}."
        )

    _check_cancelled(cancel_token)
    records = result["records"]
    row_ids = result["row_ids"]
    input_binding = result["input_binding"]

    payload = contract_utils.build_predictions_payload(
        context=context,
        run_id=run_id,
        dataset_id=dataset_id,
        model_artifact_id=model_artifact_id,
        model_payload=model_payload,
        records=records,
        row_ids=row_ids,
        input_binding=input_binding,
        compatibility_report=compatibility,
        params=params,
        prediction_scope=str(params.get("prediction_scope") or "inference"),
    )

    artifact_id = context.artifacts.put(
        artifact_utils.ARTIFACTS.PREDICTIONS,
        payload,
        dataset_id=dataset_id,
        row_ids=row_ids,
        params={"model_artifact_id": model_artifact_id, **params},
    )

    derived_dataset_id = None
    if bool(params.get("register_prediction_dataset", True)):
        derived_dataset_id = register_prediction_table_dataset(
            context=context,
            predictions_payload=payload,
            predictions_artifact_id=artifact_id,
        )
        if derived_dataset_id:
            payload.setdefault("derived_table", {})["dataset_id"] = derived_dataset_id
            payload.setdefault("derived_table", {})["join_key"] = "record_id"

    _publish(
        context,
        "ml.predictions.created",
        {
            "artifact_id": artifact_id,
            "dataset_id": dataset_id,
            "derived_dataset_id": derived_dataset_id,
            "model_artifact_id": model_artifact_id,
            "count": len(row_ids),
            "task": task,
            "modality": modality,
            "framework": framework,
            "recommended_color_columns": payload.get("visualisation", {}).get("recommended_color_columns", []),
        },
    )

    catalog = _get_trained_model_catalog(context)
    if catalog is not None:
        try:
            catalog.refresh()
        except Exception:
            pass

    return artifact_utils.json_safe(
        {
            "ok": True,
            "artifact_id": artifact_id,
            "derived_dataset_id": derived_dataset_id,
            "dataset_id": dataset_id,
            "model_artifact_id": model_artifact_id,
            "count": len(row_ids),
            "duration_seconds": time.time() - started,
            "compatibility_report": compatibility.to_dict(),
            "recommended_color_columns": payload.get("visualisation", {}).get("recommended_color_columns", []),
            "prediction_preview": payload.get("prediction_table", {}).get("rows", [])[:25],
        }
    )


def predict_tabular_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Backward-compatible alias for existing `predict_tabular` registrations."""

    return predict_action(context=context, request=request, cancel_token=cancel_token)


def register_prediction_table_dataset(
    *,
    context: Any,
    predictions_payload: Mapping[str, Any],
    predictions_artifact_id: str,
) -> Optional[str]:
    """Expose prediction rows as a lightweight dataset for visualisation/table tools."""

    rows = list((predictions_payload.get("prediction_table") or {}).get("rows") or [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    run_id = str(predictions_payload.get("prediction_run_id") or predictions_payload.get("run_id") or uuid.uuid4().hex)
    source_dataset_id = str(predictions_payload.get("dataset_id") or "dataset")
    model_title = str(predictions_payload.get("model_title") or predictions_payload.get("model_id") or "model")
    dataset_id = f"predictions_{source_dataset_id}_{run_id[:8]}"

    datasets = getattr(context, "datasets", None)
    register = getattr(datasets, "register", None)
    if not callable(register):
        return None
    try:
        register(
            dataset_id,
            df,
            name=f"Predictions: {model_title}",
            source="ml.predictions",
            source_dataset_id=source_dataset_id,
            predictions_artifact_id=predictions_artifact_id,
            model_artifact_id=predictions_payload.get("model_artifact_id"),
            row_count=len(df),
            columns=list(df.columns),
            column_mappings={"record_id": "record_id"},
        )
        try:
            datasets.set_mapping(dataset_id, "record_id", "record_id")
        except Exception:
            pass
        _publish(
            context,
            "dataset.loaded",
            {
                "dataset_id": dataset_id,
                "origin": "core.ml.predictions",
                "source_dataset_id": source_dataset_id,
                "predictions_artifact_id": predictions_artifact_id,
            },
        )
        return dataset_id
    except Exception:
        return None


def _predict_sklearn_tabular(
    *,
    context: Any,
    dataset_id: str,
    model_artifact_id: str,
    model_payload: Mapping[str, Any],
    compatibility: Any,
    request: ActionRequest,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    artifact_utils = _load_sibling_module("artifacts")
    model = artifact_utils.load_model_from_payload(model_payload)
    binding = dict(compatibility.resolved_input_binding or {})
    feature_columns = [str(c) for c in binding.get("feature_columns") or []]
    if not feature_columns:
        raise ValueError("Model compatibility did not resolve any feature columns.")

    record_id_column = binding.get("record_id_column")
    columns = list(dict.fromkeys([*feature_columns, *([record_id_column] if record_id_column else [])]))
    df = context.datasets.get_df(dataset_id, columns=columns)
    df = _filter_rows(df, request, record_id_column)
    _check_cancelled(cancel_token)

    X = df[feature_columns]
    predictions = model.predict(X)
    probabilities = _predict_proba(model, X)
    classes = _classes(model)
    row_ids = _row_ids(df, record_id_column)
    records = _prediction_records(
        row_ids=row_ids,
        predictions=predictions,
        probabilities=probabilities,
        classes=classes,
    )
    return {"row_ids": row_ids, "records": records, "input_binding": binding}


def _predict_torch_tabular(
    *,
    context: Any,
    dataset_id: str,
    model_artifact_id: str,
    model_payload: Mapping[str, Any],
    compatibility: Any,
    request: ActionRequest,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    artifact_utils = _load_sibling_module("artifacts")
    saved = artifact_utils.load_model_from_payload(model_payload)
    if not isinstance(saved, Mapping):
        raise TypeError("Torch tabular model sidecar must contain model/preprocessor metadata.")
    torch_model = saved.get("torch_model") or saved.get("model")
    preprocessor = saved.get("preprocessor")
    label_encoder = saved.get("label_encoder")
    if torch_model is None or preprocessor is None:
        raise ValueError(
            "Torch tabular prediction requires the saved sidecar to contain `torch_model` and `preprocessor`."
        )

    import torch

    binding = dict(compatibility.resolved_input_binding or {})
    feature_columns = [str(c) for c in binding.get("feature_columns") or []]
    record_id_column = binding.get("record_id_column")
    columns = list(dict.fromkeys([*feature_columns, *([record_id_column] if record_id_column else [])]))
    df = context.datasets.get_df(dataset_id, columns=columns)
    df = _filter_rows(df, request, record_id_column)
    _check_cancelled(cancel_token)

    X = np.asarray(preprocessor.transform(df[feature_columns]), dtype=np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    torch_model.eval()
    with torch.no_grad():
        output = torch_model(X_tensor).detach().cpu().numpy()
    task = str(model_payload.get("task") or "classification").lower()
    row_ids = _row_ids(df, record_id_column)
    if task == "classification":
        probabilities = _softmax(output)
        pred_idx = np.argmax(probabilities, axis=1)
        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(pred_idx)
            classes = [str(c) for c in getattr(label_encoder, "classes_", [])]
        else:
            predictions = pred_idx
            classes = [str(i) for i in range(probabilities.shape[1])]
        records = _prediction_records(
            row_ids=row_ids,
            predictions=predictions,
            probabilities=probabilities,
            classes=classes,
        )
    else:
        predictions = output.reshape(-1)
        records = _prediction_records(row_ids=row_ids, predictions=predictions)
    return {"row_ids": row_ids, "records": records, "input_binding": binding}

def _predict_torch_image_classifier(
    *,
    context: Any,
    dataset_id: str,
    model_artifact_id: str,
    model_payload: Mapping[str, Any],
    compatibility: Any,
    request: ActionRequest,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    artifact_utils = _load_sibling_module("artifacts")
    image_sidecar = _load_sibling_module("image_sidecar")

    saved = artifact_utils.load_model_from_payload(model_payload)
    if not isinstance(saved, Mapping):
        raise TypeError(
            "Torch image classifier sidecar must load to a mapping containing "
            "`torch_model`, `class_names`, `image_size`, and `normalization`."
        )

    torch_model = saved.get("torch_model")
    class_names = [str(c) for c in saved.get("class_names") or []]
    if torch_model is None:
        raise ValueError("Image model sidecar is missing `torch_model`.")
    if not class_names:
        raise ValueError("Image model sidecar is missing `class_names`.")

    import torch

    params = dict(request.params or {})
    binding = dict(compatibility.resolved_input_binding or {})
    image_column = binding.get("image_column") or params.get("image_column")
    record_id_column = binding.get("record_id_column")

    if not image_column:
        raise ValueError("Compatibility did not resolve an image column.")

    columns = [str(image_column)]
    if record_id_column:
        columns.append(str(record_id_column))
    columns = list(dict.fromkeys(columns))

    df = context.datasets.get_df(dataset_id, columns=columns)
    df = _filter_rows(df, request, record_id_column)

    image_size = int(saved.get("image_size") or 224)
    normalization = dict(saved.get("normalization") or image_sidecar.DEFAULT_NORMALIZATION)
    transform = image_sidecar.image_transform(
        image_size=image_size,
        normalization=normalization,
    )

    device_name = str(params.get("device") or "auto").lower()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    batch_size = int(params.get("image_batch_size") or params.get("batch_size") or 32)
    skip_bad_images = bool(params.get("skip_bad_images", True))

    torch_model.to(device)
    torch_model.eval()

    row_ids: List[str] = []
    tensors: List[Any] = []
    failed_rows: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    def flush_batch() -> None:
        if not tensors:
            return

        _check_cancelled(cancel_token)

        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            logits = torch_model(batch)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()

        pred_idx = probabilities.argmax(axis=1)
        predictions = [class_names[int(idx)] for idx in pred_idx]

        batch_records = _prediction_records(
            row_ids=row_ids[-len(tensors):],
            predictions=predictions,
            probabilities=probabilities,
            classes=class_names,
        )
        records.extend(batch_records)
        tensors.clear()

        del batch, logits

    for index, row in df.iterrows():
        _check_cancelled(cancel_token)

        if record_id_column and record_id_column in df.columns:
            row_id = str(row[record_id_column])
        else:
            row_id = str(index)

        try:
            image = image_sidecar.load_image(row[image_column])
            tensors.append(transform(image))
            row_ids.append(row_id)
        except Exception as exc:
            failed = {
                "row_id": row_id,
                "image_column": image_column,
                "error": str(exc),
            }
            failed_rows.append(failed)
            if not skip_bad_images:
                raise ValueError(f"Could not load image for row {row_id}: {exc}") from exc

        if len(tensors) >= batch_size:
            flush_batch()

    flush_batch()

    if failed_rows:
        binding["failed_image_rows"] = failed_rows[:100]
        binding["failed_image_row_count"] = len(failed_rows)

    return {
        "row_ids": [record["row_id"] for record in records],
        "records": records,
        "input_binding": {
            **binding,
            "kind": "image",
            "image_column": image_column,
            "image_size": image_size,
            "normalization": normalization,
            "class_names": class_names,
            "device": str(device),
            "batch_size": batch_size,
            "skipped_bad_images": len(failed_rows),
        },
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
    row_ids: Sequence[Any],
    predictions: Any,
    probabilities: Any = None,
    classes: Any = None,
) -> List[Dict[str, Any]]:
    preds = np.asarray(predictions).reshape(-1)
    prob_array = None if probabilities is None else np.asarray(probabilities)
    class_values = [str(c) for c in list(classes or [])]
    records: List[Dict[str, Any]] = []
    for idx, row_id in enumerate(row_ids):
        pred = preds[idx] if idx < len(preds) else None
        record: Dict[str, Any] = {
            "row_id": str(row_id),
            "prediction": _json_scalar(pred),
            "y_pred": _json_scalar(pred),
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
                record["confidence"] = float(sorted_probs[0])
                record["least_confidence"] = float(1.0 - sorted_probs[0])
                if sorted_probs.size >= 2:
                    record["margin"] = float(sorted_probs[0] - sorted_probs[1])
                    record["margin_uncertainty"] = float(1.0 - record["margin"])
                record["entropy"] = _entropy(probs)
        records.append(record)
    return records


def _predict_proba(model: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba") and callable(model.predict_proba):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    return None


def _classes(model: Any) -> List[str]:
    try:
        named_steps = getattr(model, "named_steps", None)
        if named_steps:
            estimator = named_steps.get("model") or named_steps.get("estimator")
            classes = getattr(estimator, "classes_", None)
            if classes is not None:
                return [str(c) for c in list(classes)]
    except Exception:
        pass
    classes = getattr(model, "classes_", None)
    if classes is not None:
        try:
            return [str(c) for c in list(classes)]
        except Exception:
            pass
    return []


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    exp = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


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
            raise RuntimeError("ML prediction cancelled.")


def _publish(context: Any, topic: str, payload: Mapping[str, Any]) -> None:
    events = getattr(context, "events", None)
    publish = getattr(events, "publish", None)
    if callable(publish):
        publish(topic, dict(payload))


def _get_trained_model_catalog(context: Any) -> Any:
    services = getattr(context, "services", None)
    get = getattr(services, "get", None)
    if not callable(get):
        return None
    for key in ("core.ml.trained_model_catalog", "trained_model_catalog"):
        try:
            return get(key)
        except Exception:
            continue
    return None