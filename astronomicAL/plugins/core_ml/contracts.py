from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .serialization import json_safe

MODEL_CONTRACT_SCHEMA_VERSION = 2
PREDICTION_SCHEMA_VERSION = 2

@dataclass
class ModelCompatibilityReport:
    """Result of checking whether a trained model can predict on a dataset."""

    status: str
    model_artifact_id: str
    dataset_id: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    needs_mapping: List[str] = field(default_factory=list)
    model_summary: Dict[str, Any] = field(default_factory=dict)
    dataset_summary: Dict[str, Any] = field(default_factory=dict)
    resolved_input_binding: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    recommended_prediction_columns: List[str] = field(default_factory=list)

    @property
    def can_predict(self) -> bool:
        return self.status in {"compatible", "warning"}

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

def safe_record_id(value: Any) -> str:
    """Return a stable string record id without turning large ints into floats.

    Important for Euclid/object_id style identifiers:
    large int64 ids must remain exact strings. If a float already reached this
    function, precision may already be lost, so do not invent integer precision
    for very large floats.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, np.integer):
        return str(int(value))

    if isinstance(value, int):
        return str(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""

        # Only stringify small integral floats as ints. Large floats may be
        # rounded scientific-notation versions of int64 ids, so keep their
        # float representation for fallback matching rather than pretending
        # they are exact.
        if value.is_integer() and abs(value) <= 9_007_199_254_740_991:
            return str(int(value))

        return repr(value)

    if isinstance(value, np.generic):
        try:
            return safe_record_id(value.item())
        except Exception:
            pass

    return str(value).strip()

def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}

def _metadata_with_input_contract(model_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge recipe input_contract into metadata-like lookup state.

    Older generic model artifacts put most fields directly under metadata.
    Recipe artifacts put image/target/normalisation fields under input_contract.
    Contract building should support both.
    """
    input_contract = _as_mapping(model_payload.get("input_contract"))
    metadata = _as_mapping(model_payload.get("metadata"))

    # Metadata wins if both exist, but input_contract provides the important
    # recipe defaults.
    merged = {**input_contract, **metadata}

    training = _as_mapping(model_payload.get("training"))
    if training:
        merged.setdefault("training", training)

    metrics = _as_mapping(model_payload.get("metrics"))
    if metrics:
        merged.setdefault("metrics", metrics)

    return merged

def _first_non_empty(*values: Any) -> Optional[str]:
    for value in values:
        text = _none_if_empty(value)
        if text:
            return text
    return None

def ensure_model_contract(
    *,
    context: Any,
    model_artifact_id: str,
    model_payload: Optional[Mapping[str, Any]] = None,
    model_object: Any = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """Ensure an ml.model artifact carries a self-describing prediction contract."""

    if model_payload is None:
        model_payload = _artifact_get(context, model_artifact_id)
    if not isinstance(model_payload, Mapping):
        raise TypeError(f"Artifact {model_artifact_id!r} is not an ml.model payload.")

    existing = model_payload.get("prediction_contract")
    if isinstance(existing, Mapping) and int(existing.get("schema_version", 0) or 0) >= MODEL_CONTRACT_SCHEMA_VERSION:
        return json_safe(dict(existing))

    contract = build_model_contract(
        context=context,
        model_artifact_id=model_artifact_id,
        model_payload=model_payload,
        model_object=model_object,
    )

    if persist and isinstance(model_payload, dict):
        model_payload["prediction_contract"] = contract
        model_payload["schema_version"] = max(
            int(model_payload.get("schema_version", 1) or 1),
            MODEL_CONTRACT_SCHEMA_VERSION,
        )
        model_payload["updated_at"] = time.time()

    return contract

def build_model_contract(
    *,
    context: Any,
    model_artifact_id: str,
    model_payload: Mapping[str, Any],
    model_object: Any = None,
) -> Dict[str, Any]:
    """Build the portable model contract from a trained-model artifact."""

    input_contract = _as_mapping(model_payload.get("input_contract"))
    metadata = _metadata_with_input_contract(model_payload)
    training = _as_mapping(model_payload.get("training"))
    metrics = _as_mapping(model_payload.get("metrics"))

    framework = str(
        model_payload.get("framework")
        or metadata.get("framework")
        or ""
    ).lower()

    modality = str(
        model_payload.get("modality")
        or metadata.get("modality")
        or (
            "image"
            if _first_non_empty(
                model_payload.get("image_column"),
                input_contract.get("image_column"),
                metadata.get("image_column"),
            )
            else "tabular"
        )
    ).lower()

    task = str(
        model_payload.get("task")
        or metadata.get("task")
        or "classification"
    ).lower()

    training_dataset_id = str(
        model_payload.get("train_dataset_id")
        or model_payload.get("training_dataset_id")
        or input_contract.get("train_dataset_id")
        or input_contract.get("dataset_id")
        or training.get("train_dataset_id")
        or training.get("dataset_id")
        or model_payload.get("dataset_id")
        or ""
    )

    source_dataset_id = str(
        model_payload.get("source_dataset_id")
        or input_contract.get("source_dataset_id")
        or training.get("source_dataset_id")
        or model_payload.get("dataset_id")
        or ""
    )

    target_column = _first_non_empty(
        model_payload.get("target_column"),
        input_contract.get("target_column"),
        training.get("target_column"),
        metadata.get("target_column"),
    )

    record_id_column = _first_non_empty(
        model_payload.get("record_id_column"),
        input_contract.get("record_id_column"),
        training.get("record_id_column"),
        metadata.get("record_id_column"),
    )

    if not record_id_column and training_dataset_id:
        record_id_column = _mapped_column(context, training_dataset_id, "record_id")

    if modality == "image":
        input_schema = _image_input_schema(
            context=context,
            model_payload=model_payload,
            metadata=metadata,
            training_dataset_id=training_dataset_id,
        )
    else:
        input_schema = _tabular_input_schema(
            context=context,
            model_payload=model_payload,
            metadata=metadata,
            training_dataset_id=training_dataset_id,
        )

    classes = _classes_from_payload_or_model(model_payload, model_object)

    output_schema = _output_schema(
        task=task,
        classes=classes,
        metadata=metadata,
    )

    hyperparameters = {}
    if isinstance(model_payload.get("params"), Mapping):
        hyperparameters.update(dict(model_payload.get("params") or {}))
    if training:
        hyperparameters.update(training)

    contract = {
        "schema_version": MODEL_CONTRACT_SCHEMA_VERSION,
        "created_at": time.time(),
        "model_artifact_id": model_artifact_id,
        "run_id": model_payload.get("run_id"),
        "recipe_id": model_payload.get("recipe_id"),
        "recipe_version": model_payload.get("recipe_version"),
        "model_definition_id": model_payload.get("model_id"),
        "model_title": (
            model_payload.get("model_title")
            or model_payload.get("model_id")
            or model_payload.get("recipe_id")
            or model_artifact_id
        ),
        "framework": framework,
        "task": task,
        "modality": modality,
        "trained_on": {
            "dataset_id": training_dataset_id,
            "source_dataset_id": source_dataset_id or None,
            "dataset_fingerprint": (
                dataset_fingerprint(context, training_dataset_id)
                if training_dataset_id
                else None
            ),
            "row_count": (
                _row_count(context, training_dataset_id)
                if training_dataset_id
                else None
            ),
            "target_column": target_column,
            "target_semantic": "target_label" if target_column else None,
            "record_id_column": record_id_column,
            "record_id_semantic": "record_id" if record_id_column else None,
        },
        "input_schema": input_schema,
        "output_schema": output_schema,
        "training_context": {
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "tuning": metadata.get("tuning") or model_payload.get("tuning") or {},
            "active_learning_strategy": metadata.get("active_learning_strategy"),
            "label_schema_hash": _stable_hash(output_schema.get("classes") or []),
            "feature_schema_hash": _stable_hash(input_schema),
        },
        "compatibility": {
            "allow_extra_columns": True,
            "allow_column_remapping": True,
            "block_on_missing_inputs": True,
            "warn_on_dtype_change": True,
            "warn_on_target_class_mismatch": True,
        },
        "source_payload_summary": {
            "kind": model_payload.get("kind"),
            "model_ref": model_payload.get("model_ref"),
            "architecture": model_payload.get("architecture"),
            "custom_model_import": model_payload.get("custom_model_import"),
            "input_contract": input_contract,
            "metrics": metrics,
            "created_at": model_payload.get("created_at"),
        },
    }

    return json_safe(contract)

def validate_model_for_dataset(
    *,
    context: Any,
    model_artifact_id: str,
    dataset_id: str,
    target_column: Optional[str] = None,
    image_column: Optional[str] = None,
    feature_column_mapping: Optional[Mapping[str, str]] = None,
    require_target_compatible: bool = False,
) -> ModelCompatibilityReport:
    """Validate a trained model against the dataset the user wants to predict on."""

    model_payload = _artifact_get(context, model_artifact_id)
    if not isinstance(model_payload, Mapping):
        raise TypeError(f"Artifact {model_artifact_id!r} is not an ml.model payload.")
    contract = ensure_model_contract(
        context=context,
        model_artifact_id=model_artifact_id,
        model_payload=model_payload,
        persist=True,
    )

    errors: List[str] = []
    warnings: List[str] = []
    needs_mapping: List[str] = []
    mapping = {str(k): str(v) for k, v in dict(feature_column_mapping or {}).items() if v is not None}

    dataset_columns = dataset_column_names(context, dataset_id)
    dataset_column_set = set(dataset_columns)
    dtypes = dataset_dtypes(context, dataset_id)
    modality = str(contract.get("modality") or "tabular").lower()
    task = str(contract.get("task") or "classification").lower()
    input_schema = dict(contract.get("input_schema") or {})
    output_schema = dict(contract.get("output_schema") or {})

    resolved_binding: Dict[str, Any] = {}

    record_id_column = _mapped_column(context, dataset_id, "record_id")
    if not record_id_column:
        warnings.append("Dataset has no mapped `record_id`; prediction rows will fall back to dataframe index ids.")
    elif record_id_column not in dataset_column_set and str(record_id_column).lower() not in {
        "use index",
        "use_index",
        "__index__",
        "index",
    }:
        warnings.append(f"Mapped record_id column `{record_id_column}` is not present; dataframe index ids will be used.")

    if modality == "tabular":
        expected_features = [str(c) for c in input_schema.get("feature_columns") or []]
        if not expected_features:
            errors.append("Model contract does not declare any tabular feature columns.")
        actual_features: List[str] = []
        missing_features: List[str] = []
        remapped_features: Dict[str, str] = {}
        for expected in expected_features:
            actual = mapping.get(expected, expected)
            if actual in dataset_column_set:
                actual_features.append(actual)
                if actual != expected:
                    remapped_features[expected] = actual
            else:
                missing_features.append(expected)
        if missing_features:
            errors.append(
                "Dataset is missing required feature columns: " + ", ".join(f"`{c}`" for c in missing_features)
            )
        expected_dtypes = dict(input_schema.get("feature_dtypes") or {})
        dtype_warnings = _dtype_warnings(expected_dtypes, dtypes, mapping)
        warnings.extend(dtype_warnings)
        resolved_binding.update(
            {
                "kind": "tabular",
                "feature_columns": actual_features,
                "expected_feature_columns": expected_features,
                "feature_column_mapping": remapped_features,
                "record_id_column": record_id_column,
            }
        )

    elif modality == "image":
        preferred_image_column = image_column or input_schema.get("image_column")
        resolved_image_column = _first_present(
            [
                preferred_image_column,
                _mapped_column(context, dataset_id, "image.uri"),
                _mapped_column(context, dataset_id, "image.path"),
                _mapped_column(context, dataset_id, "image.url"),
                _guess_image_column(dataset_columns),
            ],
            dataset_column_set,
        )
        if not resolved_image_column:
            needs_mapping.append("image.path")
            errors.append("Image model requires an image path/URI column, but none could be resolved on this dataset.")
        resolved_binding.update(
            {
                "kind": "image",
                "image_column": resolved_image_column,
                "expected_image_column": input_schema.get("image_column"),
                "record_id_column": record_id_column,
                "image_size": input_schema.get("image_size"),
                "normalization": input_schema.get("normalization"),
            }
        )
    else:
        errors.append(f"Unsupported model modality `{modality}`.")

    eval_target_column = target_column or _mapped_column(context, dataset_id, "target_label")
    if eval_target_column:
        if eval_target_column not in dataset_column_set:
            warnings.append(f"Selected evaluation target column `{eval_target_column}` is not present.")
        elif task == "classification":
            class_warnings, class_errors = _target_class_warnings(
                context=context,
                dataset_id=dataset_id,
                target_column=eval_target_column,
                model_classes=[str(c) for c in output_schema.get("classes") or []],
                require_target_compatible=require_target_compatible,
            )
            warnings.extend(class_warnings)
            errors.extend(class_errors)
        resolved_binding["target_column_for_evaluation"] = eval_target_column
    else:
        resolved_binding["target_column_for_evaluation"] = None

    status = "compatible"
    if needs_mapping and not errors:
        status = "needs_mapping"
    if warnings:
        status = "warning"
    if errors:
        status = "incompatible"

    recommended_columns = recommended_prediction_columns(output_schema)

    return ModelCompatibilityReport(
        status=status,
        model_artifact_id=model_artifact_id,
        dataset_id=dataset_id,
        errors=errors,
        warnings=warnings,
        needs_mapping=needs_mapping,
        model_summary={
            "title": contract.get("model_title"),
            "framework": contract.get("framework"),
            "task": task,
            "modality": modality,
            "trained_on_dataset_id": (contract.get("trained_on") or {}).get("dataset_id"),
            "target_column": (contract.get("trained_on") or {}).get("target_column"),
            "classes": output_schema.get("classes"),
            "n_classes": output_schema.get("n_classes"),
        },
        dataset_summary={
            "dataset_id": dataset_id,
            "row_count": _row_count(context, dataset_id),
            "column_count": len(dataset_columns),
            "columns": dataset_columns,
            "fingerprint": dataset_fingerprint(context, dataset_id),
        },
        resolved_input_binding=resolved_binding,
        output_schema=output_schema,
        recommended_prediction_columns=recommended_columns,
    )

def build_predictions_payload(
    *,
    context: Any,
    run_id: str,
    dataset_id: str,
    model_artifact_id: str,
    model_payload: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    row_ids: Iterable[Any],
    input_binding: Mapping[str, Any],
    compatibility_report: ModelCompatibilityReport,
    params: Optional[Mapping[str, Any]] = None,
    prediction_scope: str = "inference",
) -> Dict[str, Any]:
    """Build a visualisation-ready ml.predictions payload.

    Row identity is taken from the explicit row_ids iterable first. Prediction
    records may have been built through pandas iterrows/JSON paths that can
    coerce large int64 identifiers into float/scientific notation.
    """
    contract = ensure_model_contract(
        context=context,
        model_artifact_id=model_artifact_id,
        model_payload=model_payload,
        persist=True,
    )
    output_schema = dict(contract.get("output_schema") or {})

    records_list = [dict(row) for row in records]
    row_ids_list = [safe_record_id(row_id) for row_id in row_ids]

    # Attach exact row ids back onto the records before payload/table creation.
    # This keeps ml.predictions.records and prediction_table.rows aligned.
    for index, row_id in enumerate(row_ids_list):
        if index >= len(records_list):
            break
        if row_id:
            records_list[index]["row_id"] = row_id
            records_list[index]["record_id"] = row_id

    table_rows = prediction_table_rows(
        records=records_list,
        row_ids=row_ids_list,
        output_schema=output_schema,
        model_artifact_id=model_artifact_id,
        prediction_run_id=run_id,
    )

    payload = {
        "artifact_type": "ml.predictions",
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "run_id": run_id,
        "prediction_run_id": run_id,
        "prediction_scope": prediction_scope,
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint(context, dataset_id),
        "model_artifact_id": model_artifact_id,
        "model_id": model_payload.get("model_id"),
        "model_title": model_payload.get("model_title"),
        "framework": contract.get("framework"),
        "task": contract.get("task"),
        "modality": contract.get("modality"),
        "trained_on": contract.get("trained_on"),
        "input_binding": dict(input_binding or {}),
        "output_schema": output_schema,
        "row_ids": row_ids_list,
        "records": records_list,
        "prediction_table": {
            "join_key": "record_id",
            "rows": table_rows,
            "columns": list(table_rows[0].keys()) if table_rows else [],
        },
        "visualisation": {
            "recommended_color_columns": recommended_prediction_columns(output_schema),
            "join_key": "record_id",
        },
        "compatibility_report": compatibility_report.to_dict(),
        "params": dict(params or {}),
        "created_at": time.time(),
    }
    return json_safe(payload)

def prediction_table_rows(
    *,
    records: Iterable[Mapping[str, Any]],
    output_schema: Mapping[str, Any],
    model_artifact_id: str,
    prediction_run_id: str,
    row_ids: Optional[Iterable[Any]] = None,
) -> List[Dict[str, Any]]:
    """Flatten per-row prediction records into dataset/plot-friendly columns.

    Prefer explicit row_ids over row_id values embedded in records, because the
    embedded record can pass through pandas/JSON paths that coerce large int64
    ids into floats.
    """
    classes = [
        str(c)
        for c in output_schema.get("classes")
        or output_schema.get("class_order")
        or []
    ]
    probability_columns = dict(output_schema.get("probability_columns") or {})

    records_list = [dict(record) for record in records]
    explicit_row_ids = [safe_record_id(row_id) for row_id in row_ids or []]

    rows: List[Dict[str, Any]] = []

    for index, raw in enumerate(records_list):
        record = dict(raw)

        if index < len(explicit_row_ids) and explicit_row_ids[index]:
            row_id = explicit_row_ids[index]
        else:
            row_id = safe_record_id(record.get("row_id", record.get("record_id")))

        prediction = record.get(
            "prediction",
            record.get("y_pred", record.get("predicted_label")),
        )

        row: Dict[str, Any] = {
            "record_id": row_id or None,
            "predicted_label": json_safe(prediction),
            "prediction": json_safe(prediction),
            "prediction_run_id": prediction_run_id,
            "model_artifact_id": model_artifact_id,
        }

        if "y_true" in record:
            row["true_label"] = json_safe(record.get("y_true"))
            row["is_correct"] = bool(str(record.get("y_true")) == str(prediction))

        if "confidence" in record:
            row["prediction_confidence"] = json_safe(record.get("confidence"))

        if "max_probability" in record:
            row["prediction_confidence"] = json_safe(record.get("max_probability"))

        for key in ("confidence_source", "confidence_semantics"):
            if key in record:
                row[key] = json_safe(record.get(key))

        for key in (
            "least_confidence",
            "margin",
            "margin_uncertainty",
            "entropy",
            "active_learning_score",
        ):
            if key in record:
                row[key] = json_safe(record.get(key))

        by_class = record.get("probabilities_by_class")
        if isinstance(by_class, Mapping):
            for class_name, value in by_class.items():
                column = (
                    probability_columns.get(str(class_name))
                    or f"prob_{_safe_column_token(str(class_name))}"
                )
                row[column] = json_safe(value)

        probabilities = record.get("probabilities")
        if probabilities is not None and not isinstance(by_class, Mapping):
            probs = list(probabilities)
            for idx, value in enumerate(probs):
                class_name = classes[idx] if idx < len(classes) else str(idx)
                column = (
                    probability_columns.get(str(class_name))
                    or f"prob_{_safe_column_token(str(class_name))}"
                )
                row[column] = json_safe(value)

        for class_name in classes:
            legacy_key = f"proba_{class_name}"
            if legacy_key in record:
                column = (
                    probability_columns.get(str(class_name))
                    or f"prob_{_safe_column_token(str(class_name))}"
                )
                row[column] = json_safe(record[legacy_key])

        rows.append(json_safe(row))

    return rows

def recommended_prediction_columns(output_schema: Mapping[str, Any]) -> List[str]:
    task = str(output_schema.get("task") or output_schema.get("kind") or "classification").lower()
    if task == "regression":
        return ["prediction", "predicted_value"]
    columns = ["predicted_label", "prediction_confidence", "entropy", "least_confidence", "margin_uncertainty"]
    prob_cols = output_schema.get("probability_columns") or {}
    if isinstance(prob_cols, Mapping):
        columns.extend(str(col) for col in prob_cols.values())
    return list(dict.fromkeys(columns))

def dataset_column_names(context: Any, dataset_id: str) -> List[str]:
    datasets = getattr(context, "datasets", None)
    for method_name in ("list_columns",):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                return [str(c) for c in method(dataset_id)]
            except Exception:
                pass
    try:
        source = datasets.get_source(dataset_id)
        return [str(c) for c in source.columns()]
    except Exception:
        pass
    try:
        return [str(c) for c in datasets.get_df(dataset_id, limit=0).columns]
    except Exception:
        return []

def dataset_dtypes(context: Any, dataset_id: str) -> Dict[str, str]:
    datasets = getattr(context, "datasets", None)
    method = getattr(datasets, "dtypes", None)
    if callable(method):
        try:
            return {str(k): str(v) for k, v in method(dataset_id).items()}
        except Exception:
            pass
    try:
        df = datasets.get_df(dataset_id, limit=50)
        return {str(k): str(v) for k, v in df.dtypes.items()}
    except Exception:
        return {}

def dataset_fingerprint(context: Any, dataset_id: str) -> Optional[str]:
    if not dataset_id:
        return None
    cols = dataset_column_names(context, dataset_id)
    row_count = _row_count(context, dataset_id)
    meta = {}
    try:
        meta = dict(context.datasets.get_meta(dataset_id) or {})
    except Exception:
        pass
    source_key = meta.get("source_path") or meta.get("source") or meta.get("path") or meta.get("backend")
    return _stable_hash({"dataset_id": dataset_id, "row_count": row_count, "columns": cols, "source": source_key})

def _tabular_input_schema(
    *,
    context: Any,
    model_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    training_dataset_id: str,
) -> Dict[str, Any]:
    features = [
        str(c)
        for c in (
            model_payload.get("feature_columns")
            or metadata.get("feature_columns")
            or metadata.get("input_columns")
            or []
        )
    ]
    dtypes = {}
    if training_dataset_id and features:
        all_dtypes = dataset_dtypes(context, training_dataset_id)
        dtypes = {column: all_dtypes.get(column) for column in features if all_dtypes.get(column)}
    numeric = []
    categorical = []
    for column in features:
        dtype = str(dtypes.get(column) or "").lower()
        if any(token in dtype for token in ("int", "float", "double", "number")):
            numeric.append(column)
        elif dtype:
            categorical.append(column)
    return {
        "kind": "tabular",
        "feature_columns": features,
        "required_columns": list(features),
        "feature_dtypes": dtypes,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "preprocessing": {
            "type": "model_embedded" if str(model_payload.get("framework", "")).lower() == "sklearn" else "sidecar_or_metadata",
            "handles_unknown_categories": True,
        },
    }

def _image_input_schema(
    *,
    context: Any,
    model_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    training_dataset_id: str,
) -> Dict[str, Any]:
    image_column = _first_non_empty(
        model_payload.get("image_column"),
        metadata.get("image_column"),
    )

    target_column = _first_non_empty(
        model_payload.get("target_column"),
        metadata.get("target_column"),
    )

    record_id_column = _first_non_empty(
        model_payload.get("record_id_column"),
        metadata.get("record_id_column"),
    )

    if not record_id_column and training_dataset_id:
        record_id_column = _mapped_column(context, training_dataset_id, "record_id")

    image_size = (
        metadata.get("image_size")
        or metadata.get("input_size")
        or model_payload.get("image_size")
        or 224
    )

    channels = (
        metadata.get("channels")
        or model_payload.get("channels")
        or 3
    )

    normalization = (
        metadata.get("normalization")
        or model_payload.get("normalization")
        or {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    )

    transform = (
        metadata.get("transform")
        or metadata.get("transforms")
        or model_payload.get("transform")
        or model_payload.get("transforms")
        or {}
    )

    try:
        image_size = int(image_size)
    except Exception:
        pass

    try:
        channels = int(channels)
    except Exception:
        channels = 3

    return {
        "kind": "image",
        "image_column": image_column,
        "target_column": target_column,
        "record_id_column": record_id_column,
        "image_semantic": "image.path",
        "required_semantics": ["image.path"],
        "image_size": image_size,
        "channels": channels,
        "normalization": normalization,
        "transform": transform,
        "accepted_uri_schemes": ["file", "http", "https", "s3", "gs"],
    }

def _output_schema(*, task: str, classes: Sequence[str], metadata: Mapping[str, Any]) -> Dict[str, Any]:
    task = str(task or "classification").lower()
    if task == "regression":
        return {
            "kind": "regression",
            "task": "regression",
            "prediction_column": "predicted_value",
            "has_probabilities": False,
        }
    class_order = [str(c) for c in classes or metadata.get("class_names") or metadata.get("classes") or []]
    probability_columns = {class_name: f"prob_{_safe_column_token(class_name)}" for class_name in class_order}
    return {
        "kind": "classification",
        "task": "classification",
        "classes": class_order,
        "class_order": class_order,
        "n_classes": len(class_order) if class_order else None,
        "has_probabilities": True,
        "prediction_column": "predicted_label",
        "confidence_column": "prediction_confidence",
        "uncertainty_columns": ["least_confidence", "margin_uncertainty", "entropy"],
        "probability_columns": probability_columns,
    }

def _classes_from_payload_or_model(model_payload: Mapping[str, Any], model_object: Any = None) -> List[str]:
    # Recipe artifacts save class_names at the top level.
    for key in ("class_order", "class_names", "classes"):
        value = model_payload.get(key)
        if value:
            return [str(c) for c in value]

    output_schema = model_payload.get("output_schema")
    if isinstance(output_schema, Mapping):
        for key in ("class_order", "class_names", "classes"):
            if output_schema.get(key):
                return [str(c) for c in output_schema[key]]

    prediction_contract = model_payload.get("prediction_contract")
    if isinstance(prediction_contract, Mapping):
        contract_output = prediction_contract.get("output_schema")
        if isinstance(contract_output, Mapping):
            for key in ("class_order", "class_names", "classes"):
                if contract_output.get(key):
                    return [str(c) for c in contract_output[key]]

    metadata = dict(model_payload.get("metadata") or {})
    for key in ("class_order", "class_names", "classes"):
        if metadata.get(key):
            return [str(c) for c in metadata[key]]

    if model_object is not None:
        classes = _classes_from_model_object(model_object)
        if classes:
            return classes

    model = model_payload.get("model")
    if model is not None:
        classes = _classes_from_model_object(model)
        if classes:
            return classes

    return []

def _classes_from_model_object(model: Any) -> List[str]:
    if isinstance(model, Mapping):
        for key in ("class_names", "classes"):
            if model.get(key):
                return [str(c) for c in model[key]]
        encoder = model.get("label_encoder")
        classes = getattr(encoder, "classes_", None)
        if classes is not None:
            return [str(c) for c in list(classes)]
        nested = model.get("model") or model.get("estimator") or model.get("torch_model")
        if nested is not None and nested is not model:
            return _classes_from_model_object(nested)
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

def _target_class_warnings(
    *,
    context: Any,
    dataset_id: str,
    target_column: str,
    model_classes: Sequence[str],
    require_target_compatible: bool,
) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    if not model_classes:
        return warnings, errors
    try:
        df = context.datasets.get_df(dataset_id, columns=[target_column])
        target_values = sorted({str(v) for v in df[target_column].dropna().unique().tolist()})
    except Exception:
        return warnings, errors
    model_set = {str(c) for c in model_classes}
    target_set = set(target_values)
    extra = sorted(target_set - model_set)
    missing = sorted(model_set - target_set)
    if extra:
        message = (
            f"Evaluation target column `{target_column}` contains classes not seen by the model: "
            + ", ".join(f"`{c}`" for c in extra)
        )
        if require_target_compatible:
            errors.append(message)
        else:
            warnings.append(message)
    if missing:
        warnings.append(
            f"Evaluation target column `{target_column}` is missing model classes: "
            + ", ".join(f"`{c}`" for c in missing)
        )
    return warnings, errors

def _dtype_warnings(expected: Mapping[str, str], actual: Mapping[str, str], mapping: Mapping[str, str]) -> List[str]:
    warnings: List[str] = []
    for expected_col, expected_dtype in expected.items():
        actual_col = mapping.get(str(expected_col), str(expected_col))
        actual_dtype = actual.get(actual_col)
        if not expected_dtype or not actual_dtype:
            continue
        if _dtype_family(expected_dtype) != _dtype_family(actual_dtype):
            warnings.append(
                f"Column `{actual_col}` has dtype `{actual_dtype}`, but the model was trained with `{expected_dtype}` for `{expected_col}`."
            )
    return warnings

def _dtype_family(dtype: str) -> str:
    low = str(dtype).lower()
    if any(token in low for token in ("int", "float", "double", "decimal", "number")):
        return "numeric"
    if any(token in low for token in ("bool",)):
        return "boolean"
    if any(token in low for token in ("date", "time")):
        return "datetime"
    return "categorical"

def _artifact_get(context: Any, artifact_id: str) -> Any:
    return context.artifacts.get(artifact_id)

def _mapped_column(context: Any, dataset_id: str, semantic_name: str) -> Optional[str]:
    if not dataset_id:
        return None
    try:
        value = context.datasets.get_mapping(dataset_id, semantic_name)
        return str(value) if value else None
    except Exception:
        return None

def _row_count(context: Any, dataset_id: str) -> Optional[int]:
    if not dataset_id:
        return None
    try:
        value = context.datasets.row_count(dataset_id)
        return int(value) if value is not None else None
    except Exception:
        pass
    try:
        return int(len(context.datasets.get_df(dataset_id, limit=None)))
    except Exception:
        return None

def _first_present(candidates: Sequence[Any], column_set: set[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = str(candidate)
        if candidate in column_set:
            return candidate
    return None

def _guess_image_column(columns: Sequence[str]) -> Optional[str]:
    lowered = {str(c).lower(): str(c) for c in columns}
    for candidate in ("image", "image_path", "image_uri", "image_url", "img", "path", "file", "filename", "cutout"):
        if candidate in lowered:
            return lowered[candidate]
    for column in columns:
        low = str(column).lower()
        if any(token in low for token in ("image", "img", "path", "uri", "url", "cutout", "jpg", "png")):
            return str(column)
    return None

def _none_if_empty(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def _stable_hash(value: Any) -> str:
    import hashlib
    import json

    raw = json.dumps(json_safe(value), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def _safe_column_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_").lower()
    return token or "class"
