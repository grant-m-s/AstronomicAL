from __future__ import annotations

import hashlib
import json
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from astronomicAL.platform.plugins.specs import ActionRequest

from . import acquisition
from . import state as al_state
from . import actions as al_actions

ORIGIN = "core.active_learning"

def ml_event_payload(base: Mapping[str, Any], ml_result: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    payload = dict(base)
    if ml_result is not None:
        payload["ml_result"] = al_state.json_safe_summary(ml_result)
        payload["metrics"] = al_state.extract_metrics(ml_result)
        for key in (
            "training_log_artifact_id",
            "run_artifact_id",
            "model_artifact_id",
            "split_spec_artifact_id",
            "evaluation_report_artifact_id",
        ):
            value = al_state.find_nested_value(ml_result, key)
            if value not in (None, ""):
                payload[key] = str(value)
        training_predictions = al_state.find_nested_value(ml_result, "predictions_artifact_id")
        if training_predictions not in (None, ""):
            payload["training_predictions_artifact_id"] = str(training_predictions)
    return payload

def collect_prediction_artifact_ids(value: Any) -> List[str]:
    """Collect candidate prediction artifact ids from a core.ml predict result.

    Avoid blindly using the first nested artifact id: training runs can include
    predictions on the materialised AL training dataset, while acquisition needs
    predictions on the original AL pool dataset.
    """

    keys = {
        "predictions_artifact_id",
        "prediction_artifact_id",
        "ml_predictions_artifact_id",
        "artifact_id",
    }
    found: List[str] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, Mapping):
            for key, raw in obj.items():
                if str(key) in keys and raw not in (None, ""):
                    found.append(str(raw))
            for nested in obj.values():
                visit(nested)
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            for nested in obj:
                visit(nested)

    visit(value)
    return al_state.stable_unique(found)

def select_pool_predictions_artifact_id(
    context: Any,
    prediction_result: Mapping[str, Any],
    *,
    pool_dataset_id: str,
) -> str:
    """Return the prediction artifact generated for the AL pool dataset.

    Raises a clear error when core.ml returns only training-dataset predictions or
    another incompatible artifact.
    """

    candidates = collect_prediction_artifact_ids(prediction_result)
    if not candidates:
        raise ValueError("core.ml.predict did not return a predictions artifact id.")

    mismatches: List[str] = []
    metadata_unknown: List[str] = []
    for artifact_id in candidates:
        try:
            payload = context.artifacts.get(str(artifact_id))
        except Exception as exc:
            mismatches.append(f"{artifact_id}: could not read artifact ({exc})")
            continue
        if not isinstance(payload, Mapping):
            mismatches.append(f"{artifact_id}: artifact payload is not a mapping")
            continue
        identifiers = acquisition.prediction_dataset_identifiers(payload)
        if not identifiers:
            metadata_unknown.append(str(artifact_id))
            continue
        if str(pool_dataset_id or "").strip() in identifiers:
            return str(artifact_id)
        mismatches.append(f"{artifact_id}: {sorted(identifiers)}")

    if metadata_unknown:
        # Older prediction payloads may not include dataset metadata.  Accept one
        # only when no explicit incompatible candidates were found.
        if not mismatches:
            return metadata_unknown[0]

    raise ValueError(
        "core.ml.predict did not return predictions for the AL pool dataset "
        f"{pool_dataset_id!r}. Candidate prediction artifacts: "
        + ("; ".join(mismatches + [f"{artifact_id}: no dataset metadata" for artifact_id in metadata_unknown]) or "none")
    )

def prediction_result_for_pool(
    context: Any,
    prediction_result: Mapping[str, Any],
    *,
    pool_dataset_id: str,
) -> Dict[str, Any]:
    result = dict(prediction_result or {})
    artifact_id = select_pool_predictions_artifact_id(context, result, pool_dataset_id=pool_dataset_id)
    result["predictions_artifact_id"] = artifact_id
    try:
        payload = context.artifacts.get(artifact_id)
    except Exception:
        payload = None
    if isinstance(payload, Mapping):
        identifiers = acquisition.prediction_dataset_identifiers(payload)
        if identifiers:
            result.setdefault("prediction_dataset_identifiers", sorted(identifiers))
            if pool_dataset_id in identifiers:
                result.setdefault("prediction_dataset_id", pool_dataset_id)
    return result

def enrich_ml_result_with_referenced_artifacts(context: Any, ml_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Attach referenced core.ml metrics/artifact summaries for AL performance tracking.

    core.ml often returns artifact ids while the detailed metrics live in the
    evaluation-report, training-log, or run artifacts.  The training-curves panel
    can see those artifacts directly; AL performance needs the same metric lookup
    when recording a round.
    """

    enriched = dict(ml_result or {})
    artifact_payloads: Dict[str, Any] = {}
    merged_metrics: Dict[str, float] = {}

    candidate_ids: List[str] = []
    for key in (
        "evaluation_report_artifact_id",
        "final_evaluation_report_artifact_id",
        "training_log_artifact_id",
        "final_training_log_artifact_id",
        "run_artifact_id",
        "ml_run_artifact_id",
    ):
        artifact_id = al_state.find_nested_value(enriched, key)
        if artifact_id in (None, ""):
            continue
        candidate_ids.append(str(artifact_id))

    candidate_ids.extend(al_state.artifact_ids(enriched))

    for artifact_id in al_state.stable_unique(candidate_ids):
        try:
            payload = context.artifacts.get(str(artifact_id))
        except Exception:
            payload = None
        if payload in (None, "", {}, []):
            continue

        label = _artifact_summary_label(str(artifact_id), payload)
        artifact_payloads[label] = al_state.json_safe_summary(payload)
        for metric, value in al_state.extract_metrics(payload).items():
            merged_metrics.setdefault(metric, value)

    direct_metrics = al_state.extract_metrics(enriched)
    if direct_metrics:
        merged_metrics.update(direct_metrics)
    if merged_metrics:
        existing = dict(enriched.get("metrics") or {}) if isinstance(enriched.get("metrics"), Mapping) else {}
        existing.update(merged_metrics)
        enriched["metrics"] = existing
    if artifact_payloads:
        enriched.setdefault("referenced_artifacts", {}).update(artifact_payloads)
    return enriched

def _artifact_summary_label(artifact_id: str, payload: Any) -> str:
    if isinstance(payload, Mapping):
        artifact_type = str(payload.get("type") or payload.get("artifact_type") or payload.get("kind") or "").strip()
        if artifact_type:
            return artifact_type.replace(".", "_")
    return str(artifact_id)

def resolve_recipe_profile_info(context: Any, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve optional core.ml recipe-profile metadata without making AL depend on it."""

    profile_id = str(
        params.get("recipe_profile_id")
        or params.get("recipe_profile_artifact_id")
        or params.get("profile_id")
        or params.get("profile_artifact_id")
        or ""
    ).strip()
    recipe_id = str(params.get("recipe_id") or "").strip()
    profile_name = ""
    profile: Dict[str, Any] = {}

    if profile_id:
        services = getattr(context, "services", None)
        store = None
        if services is not None:
            for key in ("core.ml.recipe_profile_store", "recipe_profile_store"):
                try:
                    store = services.get(key)
                    if store is not None:
                        break
                except Exception:
                    pass
        if store is not None:
            try:
                raw = store.get(profile_id)
                if isinstance(raw, Mapping):
                    profile = dict(raw)
                    recipe_id = recipe_id or str(profile.get("recipe_id") or "").strip()
                    profile_id = str(profile.get("profile_id") or profile.get("artifact_id") or profile_id)
                    profile_name = str(profile.get("name") or profile.get("title") or "").strip()
            except Exception:
                # Leave profile unresolved here; core.ml.run_ml_recipe will report a
                # precise error if the selected profile cannot be loaded.
                pass

    return {
        "recipe_profile_id": profile_id,
        "recipe_profile_name": profile_name,
        "recipe_id": recipe_id,
        "profile": profile,
    }

def merged_profile_recipe_params(profile: Mapping[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge profile params for local AL needs such as training dataset materialisation."""

    merged: Dict[str, Any] = {}
    if isinstance(profile, Mapping):
        merged.update(dict(profile.get("recipe_params") or {}))
        merged.update(dict(profile.get("binding_params") or {}))
        merged.update(dict(profile.get("protocol_params") or {}))
    merged.update(dict(params.get("recipe_params") or {}))
    return merged

def profile_data_contract_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Inspect the optional core.ml-facing data contract.

    This deliberately returns a lightweight, explicit report instead of binding
    session creation to core.ml recipes.  The report is useful to panels and
    bridge workflows, but AL sessions can exist without it.
    """

    request = al_actions.coerce_request(request)
    params = dict(request.params or {})
    dataset_id = al_actions.resolve_dataset_id(context, request, params)
    profile_info = resolve_recipe_profile_info(context, params)
    recipe_id = str(profile_info.get("recipe_id") or "").strip()
    recipe_profile_id = str(profile_info.get("recipe_profile_id") or "").strip()
    recipe_profile_name = str(profile_info.get("recipe_profile_name") or "").strip()
    if not dataset_id:
        raise ValueError("profile_data_contract requires dataset_id or an active dataset.")

    columns = list_dataset_columns(context, dataset_id)
    record_id_column = acquisition.resolve_record_id_column(context, dataset_id)
    target_column = str(params.get("target_column") or params.get("label_column") or "").strip()
    feature_columns = parse_string_list(params.get("feature_columns") or params.get("input_columns") or [])
    image_column = str(params.get("image_column") or params.get("image_path_column") or "").strip()
    mask_column = str(params.get("mask_column") or "").strip()

    errors: List[str] = []
    warnings: List[str] = []
    if not record_id_column:
        errors.append("No record_id mapping/column could be resolved.")
    for name, column in {
        "target_column": target_column,
        "image_column": image_column,
        "mask_column": mask_column,
    }.items():
        if column and column not in columns:
            errors.append(f"{name} points to missing column {column!r}.")
    missing_features = [column for column in feature_columns if column not in columns]
    if missing_features:
        errors.append("Selected feature columns are missing: " + ", ".join(missing_features))
    if not recipe_id and not recipe_profile_id:
        warnings.append("No recipe_profile_id supplied; this is fine for AL-only sessions, but training needs a saved core.ml recipe profile.")

    contract = {
        "schema_version": 2,
        "dataset": {"id": dataset_id, "columns": columns, "record_id_column": record_id_column},
        "recipe": {
            "id": recipe_id,
            "profile_id": recipe_profile_id,
            "profile_name": recipe_profile_name,
            "params": dict(params.get("recipe_params") or {}),
        },
        "bindings": {
            "pool": {
                "dataset_id": dataset_id,
                "record_id_column": record_id_column,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "image_column": image_column,
                "mask_column": mask_column,
            },
            "validation": {"dataset_id": str(params.get("validation_dataset_id") or "")},
            "test": {"dataset_id": str(params.get("test_dataset_id") or "")},
        },
        "counts": {
            "labelled_count": int(params.get("labelled_count") or 0),
            "column_count": len(columns),
        },
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
    al_actions.publish(
        context,
        "al.data_contract.profiled",
        {
            "dataset_id": dataset_id,
            "recipe_id": recipe_id,
            "recipe_profile_id": recipe_profile_id,
            "recipe_profile_name": recipe_profile_name,
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
        },
    )
    return {"ok": not errors, "contract": contract, "errors": errors, "warnings": warnings}

PREDICTION_COLUMN_NAMES = {
    "pred_label",
    "pred_confidence",
    "pred_provenance",
    "pred_entropy",
    "pred_least_confidence",
    "pred_margin_uncertainty",
    "pred_true_label",
    "pred_correct",
}
PREDICTION_COLUMN_PREFIXES = ("pred_prob_", "pred_", "prediction_", "prob_", "uncertainty_")

def is_prediction_column(column: Any) -> bool:
    name = str(column or "").strip()
    lower = name.lower()
    return lower in PREDICTION_COLUMN_NAMES or lower.startswith(PREDICTION_COLUMN_PREFIXES)

def clean_feature_columns(value: Any, *, available_columns: Optional[Sequence[str]] = None) -> List[str]:
    """Remove stale prediction-derived columns from a configured feature list.

    Active-learning rounds train from a materialised labelled dataset.  Prediction
    columns from earlier rounds are model outputs, not stable input features, and
    should never be required by the next scratch training round.
    """

    columns = parse_string_list(value)
    # ``available_columns`` is accepted for API clarity, but non-prediction
    # missing columns are intentionally kept so core_ml can report a real profile
    # mismatch.  Only stale prediction/output columns are silently removed.
    out: List[str] = []
    for column in columns:
        if is_prediction_column(column):
            continue
        if column not in out:
            out.append(column)
    return out

IMAGE_COLUMN_PARAM_KEYS = (
    "image.uri",
    "image.path",
)

IMAGE_MAPPING_KEYS = (
    "image.uri",
    "image.path",
)

IMAGE_COLUMN_EXACT_NAMES = (
    "image_uri",
    "image_path",
    "file_path",
    "filepath",
)

def _normalised_column_lookup(columns: Sequence[str]) -> Dict[str, str]:
    return {str(column).strip().lower(): str(column) for column in columns if str(column).strip()}

def _first_existing_column(candidates: Iterable[Any], columns: Sequence[str]) -> str:
    lookup = _normalised_column_lookup(columns)
    for raw in candidates:
        candidate = str(raw or "").strip()
        if not candidate:
            continue
        if candidate in columns:
            return candidate
        matched = lookup.get(candidate.lower())
        if matched:
            return matched
    return ""

def _image_column_candidates_from_params(params: Mapping[str, Any]) -> List[str]:
    candidates: List[str] = []
    for key in IMAGE_COLUMN_PARAM_KEYS:
        value = params.get(key)
        if value not in (None, "", [], {}):
            for column in parse_string_list(value):
                if column and column not in candidates:
                    candidates.append(column)
    return candidates

def _image_column_candidates_from_mappings(context: Any, dataset_id: str) -> List[str]:
    mappings = al_actions.dataset_mappings(context, dataset_id)
    candidates: List[str] = []
    for key in IMAGE_MAPPING_KEYS:
        value = mappings.get(key)
        if value and str(value) not in candidates:
            candidates.append(str(value))
    return candidates

def _image_column_candidates_from_column_names(columns: Sequence[str]) -> List[str]:
    lookup = _normalised_column_lookup(columns)
    candidates: List[str] = []
    for exact in IMAGE_COLUMN_EXACT_NAMES:
        value = lookup.get(exact)
        if value and value not in candidates:
            candidates.append(value)
    for column in columns:
        lowered = str(column).lower()
        # Avoid a generic ``path`` guess unless it is already mapped/explicit.
        if "image" in lowered and any(token in lowered for token in ("path", "uri", "url", "file", "filename")):
            value = str(column)
            if value not in candidates:
                candidates.append(value)
    return candidates

def resolve_image_column_for_al_training(
    context: Any,
    dataset_id: str,
    params: Mapping[str, Any],
    recipe_params: Mapping[str, Any],
    *,
    available_columns: Optional[Sequence[str]] = None,
) -> str:
    """Resolve the image input column that must survive AL training-set materialisation.

    Before the large-data optimisation, AL materialised the entire pool dataframe for
    training, so image columns were accidentally carried through.  Now the training
    set is intentionally narrow; image recipes therefore need the image column to be
    resolved from explicit params, dataset semantic mappings, or obvious image-path
    column names and then added back deliberately.
    """

    columns = list(available_columns or [])
    if not columns:
        columns = list_dataset_columns(context, dataset_id)

    candidates: List[str] = []
    for source in (recipe_params, params):
        candidates.extend(_image_column_candidates_from_params(source))
    candidates.extend(_image_column_candidates_from_mappings(context, dataset_id))
    candidates.extend(_image_column_candidates_from_column_names(columns))
    return _first_existing_column(candidates, columns)

def ensure_image_params(recipe_params: Dict[str, Any], image_column: str) -> None:
    """Pass the resolved image column to core.ml using the aliases it accepts."""

    image_column = str(image_column or "").strip()
    if not image_column:
        return
    recipe_params.setdefault("image_column", image_column)
    recipe_params.setdefault("image_path_column", image_column)

def sanitize_recipe_params_for_al_training(params: Mapping[str, Any], *, available_columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Return recipe params safe for AL scratch retraining.

    This strips stale prediction/output columns from feature lists and column-like
    params.  It prevents second and later AL rounds from trying to train on
    columns such as ``pred_prob_cat`` that were produced by an earlier inference
    pass or a prediction dataset.
    """

    cleaned = dict(params or {})
    available = {str(column) for column in available_columns or []}

    for key in ("feature_columns", "input_columns", "columns"):
        if key not in cleaned:
            continue
        values = clean_feature_columns(cleaned.get(key), available_columns=available_columns)
        if values:
            cleaned[key] = values
        else:
            cleaned.pop(key, None)
            # Let managed recipes fall back to their auto feature selection.
            cleaned.setdefault("auto_feature_columns", True)

    for key, value in list(cleaned.items()):
        if value in (None, "", [], {}):
            continue
        key_lower = str(key).lower()
        if key_lower.endswith("_column") or key_lower in {"target", "label", "label_column", "target_column", "id_column", "record_id_column"}:
            value_text = str(value)
            if is_prediction_column(value_text):
                cleaned.pop(key, None)

    return cleaned

def _resolve_training_control(context: Any, control_id: str) -> Any:
    if not control_id:
        return None
    services = getattr(context, "services", None)
    if services is None:
        raise RuntimeError(
            "Active Learning pause control requires the platform service registry."
        )
    registry = services.get("core.active_learning.training_controls")
    control = registry.get(control_id)
    if control is None:
        raise KeyError(
            f"Unknown Active Learning training control: {control_id}"
        )
    return control


def _call_registered_training_action(
    context: Any,
    request: ActionRequest,
    *,
    cancel_token: Any,
    training_control: Any,
) -> Dict[str, Any]:
    """Invoke the registered core.ml trainer with both platform controls."""

    manager = getattr(context, "plugins", None)
    if manager is None or not hasattr(manager, "get_action"):
        raise RuntimeError(
            "core.ml.run_ml_recipe requires the platform plugin manager."
        )
    registration = manager.get_action("core.ml.run_ml_recipe")
    handler = getattr(registration, "handler", None)
    if not callable(handler):
        raise RuntimeError(
            "Registered action 'core.ml.run_ml_recipe' has no callable handler."
        )
    raw = handler(
        context,
        request,
        cancel_token=cancel_token,
        training_control=training_control,
    )
    if isinstance(raw, Mapping):
        return dict(raw)
    return {"ok": True, "result": raw}



_RESUME_REFERENCE_KEYS = (
    "resume_checkpoint_artifact_id",
    "resume_manifest_path",
    "resume_checkpoint_path",
)

ARTIFACT_RESUME_REQUEST = "al.resume_request"
ARTIFACT_RESUME_DIAGNOSTIC = "al.resume_diagnostic"


_RESUME_RUNTIME_OVERRIDE_KEYS = (
    "device",
    "ml_artifact_dir",
    "artifact_dir",
    "prediction_output_dir",
    "save_predictions",
    "keep_work_dir",
    "keep_failed_work_dir",
    "trust_external_checkpoint",
)



def _safe_resume_value(value: Any, *, depth: int = 0) -> Any:
    """Return a bounded, status-safe representation of resume metadata."""

    if depth >= 5:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for index, (raw_key, raw_value) in enumerate(value.items()):
            if index >= 80:
                result["__truncated_keys__"] = len(value) - index
                break
            key = str(raw_key)
            lowered = key.lower()
            if any(
                token in lowered
                for token in (
                    "dataset",
                    "split",
                    "partition",
                    "protocol",
                    "row",
                    "recipe",
                    "target",
                    "checkpoint",
                    "manifest",
                    "artifact",
                    "source",
                    "seed",
                )
            ):
                result[key] = _safe_resume_value(raw_value, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        values = list(value)
        if len(values) <= 20:
            return [
                _safe_resume_value(item, depth=depth + 1)
                for item in values
            ]
        return {
            "count": len(values),
            "first": [
                _safe_resume_value(item, depth=depth + 1)
                for item in values[:5]
            ],
            "last": [
                _safe_resume_value(item, depth=depth + 1)
                for item in values[-5:]
            ],
        }
    return repr(value)[:500]


def _artifact_payload(context: Any, artifact_id: str) -> Dict[str, Any]:
    artifact_id = str(artifact_id or "").strip()
    if not artifact_id:
        return {}
    try:
        payload = context.artifacts.get(artifact_id)
    except Exception as exc:
        return {
            "artifact_id": artifact_id,
            "read_error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, Mapping):
        return {
            "artifact_id": artifact_id,
            "payload_type": type(payload).__name__,
        }
    return dict(payload)


def _store_resume_request(
    context: Any,
    *,
    session: Mapping[str, Any],
    session_artifact_id: str,
    request: ActionRequest,
) -> str:
    """Persist the exact Core ML request required for a safe resume."""

    row_ids = [
        str(row_id)
        for row_id in (request.row_ids or [])
        if str(row_id).strip()
    ]
    payload = {
        "schema_version": 1,
        "session_id": str(session.get("session_id") or ""),
        "session_artifact_id": session_artifact_id,
        "dataset_id": str(request.dataset_id or ""),
        "row_ids": row_ids,
        "row_count": len(row_ids),
        "row_ids_sha256": _training_membership_signature(row_ids),
        "columns": list(request.columns or []),
        "params": dict(request.params or {}),
        "artifact_id": request.artifact_id,
        "origin": request.origin,
        "created_at": al_state.now(),
    }
    return str(
        context.artifacts.put(
            ARTIFACT_RESUME_REQUEST,
            payload,
            dataset_id=str(request.dataset_id or "") or None,
            row_ids=row_ids[:1000],
            row_count=len(row_ids),
            params={
                "session_id": payload["session_id"],
                "row_count": len(row_ids),
                "row_ids_sha256": payload["row_ids_sha256"],
            },
        )
    )


def _load_resume_request(
    context: Any,
    *,
    session: Mapping[str, Any],
    paused_event: Mapping[str, Any],
) -> tuple[str, Dict[str, Any]]:
    snapshot = dict(session.get("paused_training") or {})
    latest = dict(session.get("latest") or {})
    artifact_id = str(
        snapshot.get("resume_request_artifact_id")
        or paused_event.get("resume_request_artifact_id")
        or latest.get("resume_request_artifact_id")
        or ""
    ).strip()
    if not artifact_id:
        return "", {}
    payload = _artifact_payload(context, artifact_id)
    if payload.get("read_error"):
        raise RuntimeError(
            "The exact paused Core ML request could not be read from "
            f"artifact {artifact_id!r}: {payload['read_error']}"
        )
    if not payload:
        raise RuntimeError(
            f"Resume request artifact {artifact_id!r} is empty."
        )
    return artifact_id, payload


def _apply_resume_overrides(
    base_params: Mapping[str, Any],
    *,
    incoming_params: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> Dict[str, Any]:
    """Keep immutable saved params and apply only approved resume overrides."""

    merged_incoming = merged_profile_recipe_params(
        dict(profile or {}),
        dict(incoming_params or {}),
    )
    result = dict(base_params or {})
    for key in _RESUME_REFERENCE_KEYS + _RESUME_RUNTIME_OVERRIDE_KEYS:
        value = incoming_params.get(key)
        if value in (None, ""):
            value = merged_incoming.get(key)
        if value not in (None, ""):
            result[key] = value
    result.pop("resume_al_training", None)
    result.pop("training_control_id", None)
    return result


def _checkpoint_reference(params: Mapping[str, Any]) -> tuple[str, str]:
    for key in _RESUME_REFERENCE_KEYS:
        value = str(params.get(key) or "").strip()
        if value:
            return key, value
    return "", ""


def _resume_debug_snapshot(
    context: Any,
    *,
    session: Mapping[str, Any],
    session_artifact_id: str,
    materialized: Mapping[str, Any],
    ml_request: ActionRequest,
    recipe_id: str,
    recipe_profile_id: str,
    resume_request_artifact_id: str,
    exact_request_replayed: bool,
) -> Dict[str, Any]:
    row_ids = [
        str(row_id)
        for row_id in (ml_request.row_ids or [])
        if str(row_id).strip()
    ]
    params = dict(ml_request.params or {})
    reference_key, reference_value = _checkpoint_reference(params)
    checkpoint_payload = (
        _artifact_payload(context, reference_value)
        if reference_key == "resume_checkpoint_artifact_id"
        else {}
    )
    debug = {
        "schema_version": 1,
        "session_id": str(session.get("session_id") or ""),
        "session_artifact_id": session_artifact_id,
        "recipe_id": recipe_id,
        "recipe_profile_id": recipe_profile_id,
        "request_dataset_id": str(ml_request.dataset_id or ""),
        "materialized_training_dataset_id": str(
            materialized.get("training_dataset_id") or ""
        ),
        "request_row_count": len(row_ids),
        "request_row_ids_sha256": _training_membership_signature(row_ids),
        "request_first_row_ids": row_ids[:5],
        "request_last_row_ids": row_ids[-5:] if row_ids else [],
        "validation_dataset_id": str(
            session.get("validation_dataset_id") or ""
        ),
        "test_dataset_id": str(session.get("test_dataset_id") or ""),
        "checkpoint_reference_key": reference_key,
        "checkpoint_reference": reference_value,
        "resume_request_artifact_id": resume_request_artifact_id,
        "exact_request_replayed": bool(exact_request_replayed),
        "request_protocol": {
            key: _safe_resume_value(value)
            for key, value in params.items()
            if key.startswith("protocol_")
            or key
            in {
                "validation_dataset_id",
                "test_dataset_id",
                "dataset_id",
                "training_row_count",
                "training_row_ids",
                "target_column",
                "label_column",
                "task_type",
                "problem_type",
                "seed",
                "random_seed",
            }
        },
        "checkpoint_artifact_summary": _safe_resume_value(
            checkpoint_payload
        ),
    }
    diagnostic_artifact_id = context.artifacts.put(
        ARTIFACT_RESUME_DIAGNOSTIC,
        debug,
        dataset_id=str(ml_request.dataset_id or "") or None,
        row_ids=row_ids[:1000],
        row_count=len(row_ids),
        params={
            "session_id": debug["session_id"],
            "checkpoint_reference": reference_value,
            "exact_request_replayed": bool(exact_request_replayed),
        },
    )
    debug["diagnostic_artifact_id"] = str(diagnostic_artifact_id)
    return debug



def _exception_debug_chain(exc: BaseException) -> List[Dict[str, Any]]:
    """Capture bounded exception metadata without assuming Core ML types."""

    chain: List[Dict[str, Any]] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    relation = "raised"
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        attrs: Dict[str, Any] = {}
        try:
            raw_attrs = dict(getattr(current, "__dict__", {}) or {})
        except Exception:
            raw_attrs = {}
        for key, value in raw_attrs.items():
            if key in {"failure_payload", "traceback"}:
                attrs[key] = _safe_resume_value(value)
            elif any(
                token in str(key).lower()
                for token in (
                    "split",
                    "partition",
                    "protocol",
                    "dataset",
                    "row",
                    "fingerprint",
                    "signature",
                    "checkpoint",
                    "expected",
                    "actual",
                    "saved",
                    "current",
                    "diff",
                )
            ):
                attrs[str(key)] = _safe_resume_value(value)
        chain.append(
            {
                "relation": relation,
                "type": f"{type(current).__module__}.{type(current).__name__}",
                "message": str(current),
                "args": _safe_resume_value(list(getattr(current, "args", ()) or ())),
                "attributes": attrs,
            }
        )
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if cause is not None and id(cause) not in seen:
            current = cause
            relation = "cause"
        elif context is not None and id(context) not in seen:
            current = context
            relation = "context"
        else:
            current = None
    return chain


def _compact_debug_json(value: Any, *, limit: int = 12000) -> str:
    try:
        text = json.dumps(
            _safe_resume_value(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        text = repr(value)
    if len(text) > limit:
        return text[:limit] + "\n... <truncated>"
    return text


def _resume_debug_report(debug: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "=== Exact protocol submitted to Core ML ===",
            str(
                debug.get("request_protocol_json")
                or _compact_debug_json(debug.get("request_protocol"))
            ),
            "",
            "=== Paused checkpoint partition metadata ===",
            str(
                debug.get("checkpoint_artifact_summary_json")
                or _compact_debug_json(
                    debug.get("checkpoint_artifact_summary")
                )
            ),
            "",
            "=== Core ML exception chain and attributes ===",
            str(
                debug.get("exception_chain_json")
                or _compact_debug_json(debug.get("exception_chain"))
            ),
            "",
            "=== AL resume comparison values ===",
            _compact_debug_json(
                {
                    "diagnostic_artifact_id": debug.get(
                        "diagnostic_artifact_id"
                    ),
                    "exact_request_replayed": debug.get(
                        "exact_request_replayed"
                    ),
                    "request_dataset_id": debug.get("request_dataset_id"),
                    "materialized_training_dataset_id": debug.get(
                        "materialized_training_dataset_id"
                    ),
                    "request_row_count": debug.get("request_row_count"),
                    "request_row_ids_sha256": debug.get(
                        "request_row_ids_sha256"
                    ),
                    "validation_dataset_id": debug.get(
                        "validation_dataset_id"
                    ),
                    "test_dataset_id": debug.get("test_dataset_id"),
                    "checkpoint_reference_key": debug.get(
                        "checkpoint_reference_key"
                    ),
                    "checkpoint_reference": debug.get(
                        "checkpoint_reference"
                    ),
                    "resume_request_artifact_id": debug.get(
                        "resume_request_artifact_id"
                    ),
                }
            ),
        ]
    )

def _resume_debug_status(debug: Mapping[str, Any]) -> str:
    signature = str(debug.get("request_row_ids_sha256") or "")
    return (
        "Resume preflight: "
        f"dataset={debug.get('request_dataset_id')!r}; "
        f"rows={debug.get('request_row_count')} "
        f"sha256={signature[:16] or 'none'}; "
        f"validation={debug.get('validation_dataset_id')!r}; "
        f"test={debug.get('test_dataset_id')!r}; "
        f"checkpoint={debug.get('checkpoint_reference')!r}; "
        f"exact_request_replayed={bool(debug.get('exact_request_replayed'))}; "
        f"checkpoint_summary_present={bool(debug.get('checkpoint_artifact_summary'))}; "
        f"exception_chain_entries={len(debug.get('exception_chain') or [])}; "
        f"diagnostic_artifact={debug.get('diagnostic_artifact_id')!r}."
    )


def _resume_requested(params: Mapping[str, Any]) -> bool:
    return bool(params.get("resume_al_training")) or any(
        params.get(key) not in (None, "")
        for key in _RESUME_REFERENCE_KEYS
    )


def _training_membership_signature(row_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for raw_row_id in row_ids:
        encoded = str(raw_row_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _latest_paused_training_event(
    session: Mapping[str, Any],
) -> Dict[str, Any]:
    for raw_event in reversed(list(session.get("history") or [])):
        if not isinstance(raw_event, Mapping):
            continue
        event = str(raw_event.get("event") or "").strip().lower()
        if event in {"training_paused", "training_pause"}:
            return dict(raw_event)
    return {}


def _paused_training_materialization(
    context: Any,
    *,
    session: Mapping[str, Any],
    session_artifact_id: str,
    recipe_id: str,
    recipe_profile_id: str,
) -> Dict[str, Any]:
    """Rebuild AL metadata without rebuilding Core ML partitions.

    Core ML owns exact checkpoint resume.  AL must therefore reuse the paused
    training dataset, membership, holdouts, and training artifact rather than
    rerunning materialize_training_set_action(), which can alter dataset or
    partition identity.
    """

    session = al_state.coerce_session(session)
    snapshot = dict(session.get("paused_training") or {})
    paused_event = _latest_paused_training_event(session)
    resume_request_artifact_id, resume_request = _load_resume_request(
        context,
        session=session,
        paused_event=paused_event,
    )

    training_artifact_id = str(
        snapshot.get("training_artifact_id")
        or paused_event.get("training_artifact_id")
        or ""
    ).strip()
    training_payload: Dict[str, Any] = {}
    if training_artifact_id:
        try:
            raw_payload = context.artifacts.get(training_artifact_id)
        except Exception:
            raw_payload = None
        if isinstance(raw_payload, Mapping):
            training_payload = dict(raw_payload)

    labelled_items = list(al_state.labelled_training_items(session) or [])
    training_row_ids = [
        str(item.get("row_id"))
        for item in labelled_items
        if isinstance(item, Mapping)
        and item.get("row_id") not in (None, "")
    ]
    if not training_row_ids:
        raise ValueError(
            "The paused Active Learning run has no labelled training rows."
        )

    expected_count = snapshot.get("training_row_count")
    if expected_count not in (None, "") and int(expected_count) != len(
        training_row_ids
    ):
        raise ValueError(
            "The Active Learning labels changed after training was paused. "
            "Resume requires the exact labelled-row membership used by the "
            "checkpoint."
        )

    expected_signature = str(
        snapshot.get("training_row_ids_sha256") or ""
    ).strip()
    current_signature = _training_membership_signature(training_row_ids)
    if expected_signature and expected_signature != current_signature:
        raise ValueError(
            "The Active Learning training-row order or membership changed "
            "after pause. Resume requires the exact checkpoint membership."
        )

    inline_complete = bool(
        training_payload.get("training_row_ids_inline_complete")
        or training_payload.get("row_ids_inline_complete")
    )
    saved_inline_ids = [
        str(row_id)
        for row_id in (
            training_payload.get("training_row_ids")
            or training_payload.get("row_ids")
            or []
        )
        if str(row_id).strip()
    ]
    if inline_complete and saved_inline_ids and saved_inline_ids != training_row_ids:
        raise ValueError(
            "The current Active Learning training rows do not match the "
            "paused training artifact."
        )

    pool_dataset_id = acquisition.session_pool_dataset_id(session)
    training_dataset_id = str(
        snapshot.get("training_dataset_id")
        or training_payload.get("training_dataset_id")
        or pool_dataset_id
        or ""
    ).strip()
    if not training_dataset_id:
        raise ValueError(
            "The paused Active Learning run does not identify its training "
            "dataset."
        )
    if pool_dataset_id and training_dataset_id != pool_dataset_id:
        raise ValueError(
            "The Active Learning pool dataset changed after pause. Exact "
            "resume was refused."
        )

    registered = set(context.datasets.list_ids())
    if training_dataset_id not in registered:
        raise KeyError(
            f"The paused training dataset {training_dataset_id!r} is no "
            "longer registered."
        )

    paused_recipe_id = str(
        snapshot.get("recipe_id")
        or training_payload.get("recipe_id")
        or session.get("recipe_id")
        or ""
    ).strip()
    paused_profile_id = str(
        snapshot.get("recipe_profile_id")
        or training_payload.get("recipe_profile_id")
        or session.get("recipe_profile_id")
        or ""
    ).strip()
    if recipe_id and paused_recipe_id and recipe_id != paused_recipe_id:
        raise ValueError(
            "The selected recipe differs from the recipe saved in the paused "
            "checkpoint."
        )
    if (
        recipe_profile_id
        and paused_profile_id
        and recipe_profile_id != paused_profile_id
    ):
        raise ValueError(
            "The selected recipe profile differs from the profile used by "
            "the paused run."
        )

    for role in ("validation", "test"):
        current_id = str(session.get(f"{role}_dataset_id") or "").strip()
        paused_id = str(
            snapshot.get(f"{role}_dataset_id")
            or training_payload.get(f"{role}_dataset_id")
            or current_id
            or ""
        ).strip()
        if paused_id and current_id and paused_id != current_id:
            raise ValueError(
                f"The {role} dataset changed after pause. Exact resume was "
                "refused."
            )

    target_column = str(
        snapshot.get("target_column")
        or training_payload.get("target_column")
        or session.get("target_column")
        or ""
    ).strip()
    if not target_column:
        raise ValueError(
            "The paused Active Learning run does not identify its target "
            "column."
        )

    task_type = al_state.parse_task_type(
        snapshot.get("task_type")
        or training_payload.get("task_type")
        or session.get("task_type")
        or session.get("problem_type")
    )
    class_labels = list(
        snapshot.get("class_labels")
        or training_payload.get("class_labels")
        or training_payload.get("classes")
        or session.get("label_options")
        or []
    )
    round_index = int(
        snapshot.get("round")
        or paused_event.get("round")
        or training_payload.get("round")
        or int(session.get("round", 0)) + 1
    )

    return {
        "ok": True,
        "session": session,
        "session_artifact_id": session_artifact_id,
        "source_dataset_id": pool_dataset_id,
        "training_dataset_id": training_dataset_id,
        "training_dataset_reused": True,
        "training_artifact_id": training_artifact_id,
        "training_row_ids": training_row_ids,
        "target_column": target_column,
        "task_type": task_type,
        "problem_type": task_type,
        "record_id_column": (
            snapshot.get("record_id_column")
            or training_payload.get("record_id_column")
        ),
        "image_column": (
            snapshot.get("image_column")
            or training_payload.get("image_column")
        ),
        "class_labels": class_labels,
        "labelled_count": len(labelled_items),
        "recipe_id": paused_recipe_id or recipe_id,
        "recipe_profile_id": paused_profile_id or recipe_profile_id,
        "recipe_profile_name": str(
            snapshot.get("recipe_profile_name")
            or training_payload.get("recipe_profile_name")
            or session.get("recipe_profile_name")
            or ""
        ),
        "round": round_index,
        "validation_dataset_id": str(
            session.get("validation_dataset_id") or ""
        ),
        "test_dataset_id": str(session.get("test_dataset_id") or ""),
        "resume_request_artifact_id": resume_request_artifact_id,
        "resume_request": resume_request,
        "resumed_from_pause": True,
    }


def _exact_resume_recipe_params(
    *,
    params: Mapping[str, Any],
    profile: Mapping[str, Any],
    recipe_id: str,
    training_dataset_id: str,
) -> Dict[str, Any]:
    """Return only parameters that Core ML permits to vary on resume."""

    merged = merged_profile_recipe_params(dict(profile or {}), dict(params or {}))
    resume_params: Dict[str, Any] = {
        "dataset_id": training_dataset_id,
        "recipe_id": recipe_id,
    }
    for key in _RESUME_REFERENCE_KEYS + _RESUME_RUNTIME_OVERRIDE_KEYS:
        value = params.get(key)
        if value in (None, ""):
            value = merged.get(key)
        if value not in (None, ""):
            resume_params[key] = value
    return resume_params


def _interrupted_training_result(
    context: Any,
    *,
    status: str,
    session: Mapping[str, Any],
    session_artifact_id: str,
    dataset_id: str,
    materialized: Mapping[str, Any],
    training_row_ids: Sequence[str],
    recipe_id: str,
    recipe_profile_id: str,
    recipe_profile_name: str,
    seed: int,
    ml_result: Mapping[str, Any],
    submitted_request: ActionRequest,
) -> Dict[str, Any]:
    """Persist pause/cancel references without completing an AL round."""

    status = str(status).strip().lower()
    updated = al_state.coerce_session(session)
    latest = dict(updated.get("latest") or {})
    for key in (
        "resume_checkpoint_artifact_id",
        "resume_manifest_path",
        "resume_checkpoint_path",
        "training_log_artifact_id",
        "model_artifact_id",
        "run_artifact_id",
    ):
        value = al_state.find_nested_value(ml_result, key)
        if value not in (None, ""):
            latest[key] = str(value)
    latest["training_dataset_id"] = str(
        materialized.get("training_dataset_id") or ""
    )
    latest["training_artifact_id"] = str(
        materialized.get("training_artifact_id") or ""
    )
    resume_request_artifact_id = ""
    if status == "paused":
        resume_request_artifact_id = _store_resume_request(
            context,
            session=updated,
            session_artifact_id=session_artifact_id,
            request=submitted_request,
        )
        latest["resume_request_artifact_id"] = resume_request_artifact_id
    updated["latest"] = latest

    if status == "paused":
        updated["paused_training"] = {
            "schema_version": 1,
            "training_dataset_id": str(
                materialized.get("training_dataset_id") or dataset_id
            ),
            "training_artifact_id": str(
                materialized.get("training_artifact_id") or ""
            ),
            "training_row_count": len(training_row_ids),
            "training_row_ids_sha256": _training_membership_signature(
                training_row_ids
            ),
            "target_column": str(
                materialized.get("target_column")
                or updated.get("target_column")
                or ""
            ),
            "task_type": str(
                materialized.get("task_type")
                or updated.get("task_type")
                or ""
            ),
            "record_id_column": materialized.get("record_id_column"),
            "image_column": materialized.get("image_column"),
            "class_labels": list(
                materialized.get("class_labels")
                or updated.get("label_options")
                or []
            ),
            "round": int(
                materialized.get("round")
                or int(updated.get("round", 0)) + 1
            ),
            "validation_dataset_id": str(
                updated.get("validation_dataset_id") or ""
            ),
            "test_dataset_id": str(updated.get("test_dataset_id") or ""),
            "recipe_id": recipe_id,
            "recipe_profile_id": recipe_profile_id,
            "recipe_profile_name": recipe_profile_name,
            "resume_checkpoint_artifact_id": latest.get(
                "resume_checkpoint_artifact_id"
            ),
            "resume_request_artifact_id": resume_request_artifact_id,
            "resume_manifest_path": latest.get("resume_manifest_path"),
            "resume_checkpoint_path": latest.get("resume_checkpoint_path"),
        }
    elif status == "cancelled":
        updated.pop("paused_training", None)

    updated.setdefault("history", []).append(
        {
            "event": f"training_{status}",
            "round": int(materialized.get("round") or int(updated.get("round", 0)) + 1),
            "training_dataset_id": str(
                materialized.get("training_dataset_id") or dataset_id
            ),
            "training_artifact_id": str(
                materialized.get("training_artifact_id") or ""
            ),
            "training_row_count": len(training_row_ids),
            "training_row_ids_sha256": _training_membership_signature(
                training_row_ids
            ),
            "resume_checkpoint_artifact_id": latest.get(
                "resume_checkpoint_artifact_id"
            ),
            "resume_request_artifact_id": resume_request_artifact_id,
            "timestamp": al_state.now(),
        }
    )
    new_session_artifact_id = al_actions.put_session(
        context,
        updated,
        previous_artifact_id=session_artifact_id,
    )
    payload = {
        "status": status,
        "workflow_status": status,
        "session_artifact_id": new_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": updated.get("session_id"),
        "source_dataset_id": dataset_id,
        "dataset_id": dataset_id,
        "training_dataset_id": materialized.get("training_dataset_id"),
        "training_artifact_id": materialized.get("training_artifact_id"),
        "round": materialized.get("round"),
        "seed": seed,
        "recipe_id": recipe_id,
        "recipe_profile_id": recipe_profile_id,
        "recipe_profile_name": recipe_profile_name,
        "labelled_count": materialized.get("labelled_count"),
        "training_row_count": len(training_row_ids),
        "resume_checkpoint_artifact_id": latest.get(
            "resume_checkpoint_artifact_id"
        ),
        "resume_request_artifact_id": resume_request_artifact_id,
        "resume_manifest_path": latest.get("resume_manifest_path"),
        "resume_checkpoint_path": latest.get("resume_checkpoint_path"),
        "ml_result": al_state.json_safe_summary(ml_result),
        "origin": f"{ORIGIN}.train_from_session",
    }
    al_actions.publish(context, f"al.round.training_{status}", payload)
    return {"ok": True, **payload}


def _is_training_cancelled(exc: BaseException) -> bool:
    return type(exc).__name__ in {
        "CancelledError",
        "MLRecipeCancelled",
        "JobCancelled",
    }


def run_training_round(
    context: Any,
    request: Any,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    """Attach AL labels, run core.ml, then optionally predict and query."""

    from . import streaming_actions

    request = al_actions.coerce_request(request)
    params = dict(request.params or {})
    training_control_id = str(params.pop("training_control_id", "") or "").strip()
    training_control = _resolve_training_control(context, training_control_id)
    session_artifact_id = str(params.get("session_artifact_id") or "").strip()
    if not session_artifact_id:
        raise ValueError("train_from_session requires session_artifact_id.")

    session = al_state.coerce_session(context.artifacts.get(session_artifact_id))
    resume_requested = _resume_requested(params)
    if resume_requested:
        paused_snapshot = dict(session.get("paused_training") or {})
        params.setdefault(
            "recipe_profile_id",
            paused_snapshot.get("recipe_profile_id")
            or session.get("recipe_profile_id"),
        )
        params.setdefault(
            "recipe_id",
            paused_snapshot.get("recipe_id") or session.get("recipe_id"),
        )

    profile_info = resolve_recipe_profile_info(context, params)
    recipe_profile_id = str(profile_info.get("recipe_profile_id") or "").strip()
    recipe_profile_name = str(profile_info.get("recipe_profile_name") or "").strip()
    recipe_id = str(profile_info.get("recipe_id") or "").strip()
    if not recipe_profile_id and not recipe_id:
        raise ValueError("train_from_session requires recipe_profile_id or recipe_id.")

    dataset_id = acquisition.session_pool_dataset_id(session)
    seed = int(params.get("seed", session.get("seed", 42)))

    start_payload = {
        "session_artifact_id": session_artifact_id,
        "session_id": session["session_id"],
        "dataset_id": dataset_id,
        "round": int(session.get("round", 0)) + 1,
        "recipe_id": recipe_id,
        "recipe_profile_id": recipe_profile_id,
        "recipe_profile_name": recipe_profile_name,
        "seed": seed,
        "resumed": resume_requested,
        "origin": f"{ORIGIN}.train_from_session",
    }
    al_actions.publish(context, "al.round.training_started", start_payload)
    al_actions.publish(context, "ml.recipe_run.started", start_payload)
    al_actions.publish(context, "ml.training.started", start_payload)

    materialized: Dict[str, Any] = {}
    training_row_ids: List[str] = []
    recipe_params: Dict[str, Any] = {}
    ml_request: Optional[ActionRequest] = None
    resume_debug: Dict[str, Any] = {}

    try:
        if resume_requested:
            materialized = _paused_training_materialization(
                context,
                session=session,
                session_artifact_id=session_artifact_id,
                recipe_id=recipe_id,
                recipe_profile_id=recipe_profile_id,
            )
        else:
            materialized = streaming_actions.materialize_training_set_action(
                context,
                ActionRequest(
                    dataset_id=None,
                    row_ids=None,
                    columns=[],
                    params=params,
                    artifact_id=None,
                    origin=f"{ORIGIN}.materialize_training_set",
                ),
                cancel_token=cancel_token,
            )
        session = al_state.coerce_session(materialized.get("session") or session)
        session_artifact_id = str(
            materialized.get("session_artifact_id") or session_artifact_id
        )
        train_dataset_id = str(materialized["training_dataset_id"])
        training_artifact_id = str(materialized["training_artifact_id"])
        training_row_ids = [
            str(row_id)
            for row_id in materialized.get("training_row_ids") or []
            if str(row_id).strip()
        ]
        if not training_row_ids:
            raise ValueError(
                "The prepared Active Learning training set did not provide "
                "training_row_ids."
            )

        target_column = str(materialized["target_column"])
        task_type = al_state.parse_task_type(
            materialized.get("task_type")
            or session.get("task_type")
            or session.get("problem_type")
        )
        train_columns = list_dataset_columns(context, train_dataset_id)
        if resume_requested:
            saved_request = materialized.get("resume_request")
            if isinstance(saved_request, Mapping) and saved_request:
                saved_dataset_id = str(
                    saved_request.get("dataset_id") or train_dataset_id
                ).strip()
                saved_row_ids = [
                    str(row_id)
                    for row_id in (saved_request.get("row_ids") or [])
                    if str(row_id).strip()
                ]
                saved_signature = str(
                    saved_request.get("row_ids_sha256") or ""
                ).strip()
                if saved_dataset_id != train_dataset_id:
                    raise ValueError(
                        "The saved paused request targets dataset "
                        f"{saved_dataset_id!r}, but the session now targets "
                        f"{train_dataset_id!r}."
                    )
                if not saved_row_ids:
                    raise ValueError(
                        "The saved paused request contains no training rows."
                    )
                if (
                    saved_signature
                    and saved_signature
                    != _training_membership_signature(saved_row_ids)
                ):
                    raise ValueError(
                        "The saved paused request row signature is corrupt."
                    )
                if saved_row_ids != training_row_ids:
                    raise ValueError(
                        "The saved paused request no longer matches the "
                        "session's labelled training membership."
                    )
                recipe_params = _apply_resume_overrides(
                    dict(saved_request.get("params") or {}),
                    incoming_params=params,
                    profile=dict(profile_info.get("profile") or {}),
                )
                request_row_ids = saved_row_ids
                train_dataset_id = saved_dataset_id
                exact_request_replayed = True
            else:
                # A checkpoint created before exact-request persistence is
                # reconstructed with the same full AL request contract used by
                # a fresh round, then only checkpoint/runtime fields are added.
                recipe_params = sanitize_recipe_params_for_al_training(
                    merged_profile_recipe_params(
                        dict(profile_info.get("profile") or {}),
                        params,
                    ),
                    available_columns=train_columns,
                )
                image_column = str(
                    materialized.get("image_column") or ""
                ).strip()
                if image_column and image_column in train_columns:
                    ensure_image_params(recipe_params, image_column)
                recipe_params.update(
                    {
                        "dataset_id": train_dataset_id,
                        "recipe_profile_id": recipe_profile_id,
                        "recipe_id": recipe_id,
                        "target_column": target_column,
                        "label_column": target_column,
                        "al_session_id": session["session_id"],
                        "al_session_artifact_id": session_artifact_id,
                        "al_training_artifact_id": training_artifact_id,
                        "al_round": materialized["round"],
                        "training_row_ids": list(training_row_ids),
                        "training_row_count": len(training_row_ids),
                        "task_type": task_type,
                        "problem_type": task_type,
                        "warm_start": False,
                        "reset_model": True,
                        "reset_model_each_round": True,
                        "initialise_from_scratch": True,
                        "initialization_seed": seed,
                        "initialisation_seed": seed,
                        "seed": seed,
                        "random_seed": seed,
                    }
                )
                if task_type != al_state.TASK_REGRESSION:
                    recipe_params.update(
                        {
                            "label_options": materialized.get(
                                "class_labels"
                            )
                            or [],
                            "class_labels": materialized.get(
                                "class_labels"
                            )
                            or [],
                            "classes": materialized.get("class_labels") or [],
                        }
                    )
                id_column = materialized.get("record_id_column")
                if id_column:
                    recipe_params.setdefault(
                        "record_id_column",
                        id_column,
                    )
                recipe_params = _apply_resume_overrides(
                    recipe_params,
                    incoming_params=params,
                    profile=dict(profile_info.get("profile") or {}),
                )
                request_row_ids = list(training_row_ids)
                exact_request_replayed = False
        else:
            recipe_params = sanitize_recipe_params_for_al_training(
                merged_profile_recipe_params(
                    dict(profile_info.get("profile") or {}),
                    params,
                ),
                available_columns=train_columns,
            )
            image_column = str(materialized.get("image_column") or "").strip()
            if image_column and image_column in train_columns:
                ensure_image_params(recipe_params, image_column)
            recipe_params.update(
                {
                    "dataset_id": train_dataset_id,
                    "recipe_profile_id": recipe_profile_id,
                    "recipe_id": recipe_id,
                    "target_column": target_column,
                    "label_column": target_column,
                    "al_session_id": session["session_id"],
                    "al_session_artifact_id": session_artifact_id,
                    "al_training_artifact_id": training_artifact_id,
                    "al_round": materialized["round"],
                    "training_row_ids": training_row_ids,
                    "training_row_count": len(training_row_ids),
                    "task_type": task_type,
                    "problem_type": task_type,
                    "warm_start": False,
                    "reset_model": True,
                    "reset_model_each_round": True,
                    "initialise_from_scratch": True,
                    "initialization_seed": seed,
                    "initialisation_seed": seed,
                    "seed": seed,
                    "random_seed": seed,
                }
            )
            if task_type != al_state.TASK_REGRESSION:
                recipe_params.update(
                    {
                        "label_options": materialized.get("class_labels") or [],
                        "class_labels": materialized.get("class_labels") or [],
                        "classes": materialized.get("class_labels") or [],
                    }
                )
            id_column = materialized.get("record_id_column")
            if id_column:
                recipe_params.setdefault("record_id_column", id_column)
            request_row_ids = training_row_ids

        ml_request = ActionRequest(
            dataset_id=train_dataset_id,
            row_ids=request_row_ids,
            columns=[],
            params=recipe_params,
            artifact_id=None,
            origin=f"{ORIGIN}.train_from_session",
        )
        if resume_requested:
            resume_debug = _resume_debug_snapshot(
                context,
                session=session,
                session_artifact_id=session_artifact_id,
                materialized=materialized,
                ml_request=ml_request,
                recipe_id=recipe_id,
                recipe_profile_id=recipe_profile_id,
                resume_request_artifact_id=str(
                    materialized.get("resume_request_artifact_id") or ""
                ),
                exact_request_replayed=exact_request_replayed,
            )
            al_actions.publish(
                context,
                "al.round.resume_preflight",
                dict(resume_debug),
            )
        ml_result = _call_registered_training_action(
            context,
            ml_request,
            cancel_token=cancel_token,
            training_control=training_control,
        )
        ml_result = enrich_ml_result_with_referenced_artifacts(
            context,
            ml_result,
        )
        training_status = str(ml_result.get("status") or "complete").lower()
        if training_status in {"paused", "cancelled"}:
            return _interrupted_training_result(
                context,
                status=training_status,
                session=session,
                session_artifact_id=session_artifact_id,
                dataset_id=dataset_id,
                materialized=materialized,
                training_row_ids=training_row_ids,
                recipe_id=recipe_id,
                recipe_profile_id=recipe_profile_id,
                recipe_profile_name=recipe_profile_name,
                seed=seed,
                ml_result=ml_result,
                submitted_request=ml_request,
            )

    except Exception as exc:
        event = (
            "al.round.training_cancelled"
            if _is_training_cancelled(exc)
            else "al.round.training_failed"
        )
        failure_payload = {
            "session_artifact_id": session_artifact_id,
            "session_id": session.get("session_id"),
            "dataset_id": dataset_id,
            "recipe_id": recipe_id,
            "recipe_profile_id": recipe_profile_id,
            "recipe_profile_name": recipe_profile_name,
            "round": int(session.get("round", 0)) + 1,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "origin": f"{ORIGIN}.train_from_session",
        }
        if resume_requested:
            if not resume_debug and ml_request is not None and materialized:
                try:
                    resume_debug = _resume_debug_snapshot(
                        context,
                        session=session,
                        session_artifact_id=session_artifact_id,
                        materialized=materialized,
                        ml_request=ml_request,
                        recipe_id=recipe_id,
                        recipe_profile_id=recipe_profile_id,
                        resume_request_artifact_id=str(
                            materialized.get(
                                "resume_request_artifact_id"
                            )
                            or ""
                        ),
                        exact_request_replayed=bool(
                            materialized.get("resume_request")
                        ),
                    )
                except Exception as debug_exc:
                    resume_debug = {
                        "debug_error": (
                            f"{type(debug_exc).__name__}: {debug_exc}"
                        )
                    }
            resume_debug["exception_chain"] = _exception_debug_chain(exc)
            resume_debug["request_protocol_json"] = _compact_debug_json(
                resume_debug.get("request_protocol")
            )
            resume_debug["checkpoint_artifact_summary_json"] = (
                _compact_debug_json(
                    resume_debug.get("checkpoint_artifact_summary")
                )
            )
            resume_debug["exception_chain_json"] = _compact_debug_json(
                resume_debug.get("exception_chain")
            )
            failure_payload["resume_debug"] = dict(resume_debug)
            failure_payload["resume_debug_status"] = _resume_debug_status(
                resume_debug
            )
            failure_payload["resume_debug_report"] = _resume_debug_report(
                resume_debug
            )
        al_actions.publish(context, event, failure_payload)
        if not _is_training_cancelled(exc):
            al_actions.publish(context, "ml.recipe_run.failed", failure_payload)
            al_actions.publish(context, "ml.training.failed", failure_payload)
        if resume_requested and not _is_training_cancelled(exc):
            wrapped = RuntimeError(
                "\n\n".join(
                    part
                    for part in (
                        str(exc),
                        str(
                            failure_payload.get(
                                "resume_debug_status"
                            )
                            or ""
                        ),
                        str(
                            failure_payload.get(
                                "resume_debug_report"
                            )
                            or ""
                        ),
                    )
                    if part
                )
            )
            setattr(wrapped, "failure_payload", failure_payload)
            raise wrapped from exc
        raise

    session = al_state.coerce_session(session)
    session.pop("paused_training", None)
    updated_session = al_state.with_completed_training_round(
        session,
        training_dataset_id=materialized["training_dataset_id"],
        training_artifact_id=materialized["training_artifact_id"],
        ml_result=ml_result,
    )
    updated_session["recipe_id"] = recipe_id
    updated_session["recipe_profile_id"] = recipe_profile_id
    updated_session["recipe_profile_name"] = recipe_profile_name
    training_session_artifact_id = al_actions.put_session(
        context,
        updated_session,
        previous_artifact_id=session_artifact_id,
    )
    final_session_artifact_id = training_session_artifact_id
    prediction_result: Dict[str, Any] = {}
    query_result: Dict[str, Any] = {}
    workflow_errors: List[Dict[str, str]] = []

    if bool(params.get("auto_predict", True)):
        try:
            model_artifact_id = al_state.latest_reference(
                updated_session,
                "model_artifact_id",
            )
            if not model_artifact_id:
                raise ValueError(
                    "The training result did not provide model_artifact_id."
                )
            prediction_params = {
                "dataset_id": dataset_id,
                "model_artifact_id": model_artifact_id,
                "scope": "inference",
                "require_target_compatible": False,
                "register_prediction_dataset": True,
            }
            prediction_params.update(dict(params.get("prediction_params") or {}))
            prediction_request = ActionRequest(
                dataset_id=dataset_id,
                row_ids=None,
                columns=[],
                params=prediction_params,
                artifact_id=model_artifact_id,
                origin=f"{ORIGIN}.auto_predict",
            )
            prediction_result = al_actions.call_registered_action(
                context,
                "core.ml.predict",
                prediction_request,
                cancel_token=cancel_token,
            )
            prediction_result = prediction_result_for_pool(
                context,
                prediction_result,
                pool_dataset_id=dataset_id,
            )
            updated_session = al_state.with_prediction_result(
                updated_session,
                prediction_result=prediction_result,
            )
            prediction_session_artifact_id = al_actions.put_session(
                context,
                updated_session,
                previous_artifact_id=training_session_artifact_id,
            )
            final_session_artifact_id = prediction_session_artifact_id

            if bool(params.get("auto_query", False)):
                query_request = ActionRequest(
                    artifact_id=al_state.latest_reference(
                        updated_session,
                        "predictions_artifact_id",
                    ),
                    params={
                        "session_artifact_id": prediction_session_artifact_id,
                        "strategy_id": str(
                            params.get("query_strategy_id")
                            or params.get("strategy_id")
                            or "least_confidence"
                        ),
                        "k": max(
                            1,
                            int(params.get("query_k", params.get("k", 200))),
                        ),
                        "seed": seed,
                        "make_selection": bool(
                            params.get("make_selection", True)
                        ),
                    },
                    origin=f"{ORIGIN}.auto_query",
                )
                query_result = streaming_actions.query_batch_action(
                    context,
                    query_request,
                    cancel_token=cancel_token,
                )
                final_session_artifact_id = str(
                    query_result["session_artifact_id"]
                )
        except Exception as exc:
            workflow_errors.append(
                {"stage": "prediction_or_query", "error": str(exc)}
            )
            al_actions.publish(
                context,
                "al.round.prediction_or_query_failed",
                {
                    "session_artifact_id": final_session_artifact_id,
                    "session_id": updated_session.get("session_id"),
                    "dataset_id": dataset_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    finish_base = {
        "session_artifact_id": final_session_artifact_id,
        "training_session_artifact_id": training_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": updated_session["session_id"],
        "source_dataset_id": dataset_id,
        "dataset_id": dataset_id,
        "training_dataset_id": materialized["training_dataset_id"],
        "training_artifact_id": materialized["training_artifact_id"],
        "round": materialized["round"],
        "recipe_id": recipe_id,
        "recipe_profile_id": recipe_profile_id,
        "recipe_profile_name": recipe_profile_name,
        "seed": seed,
        "workflow_status": "complete" if not workflow_errors else "partial",
        "workflow_errors": workflow_errors,
        "labelled_count": materialized["labelled_count"],
        "training_row_count": len(training_row_ids),
        "training_dataset_reused": bool(
            materialized.get("training_dataset_reused", False)
        ),
        "origin": f"{ORIGIN}.train_from_session",
    }
    finish_payload = ml_event_payload(finish_base, ml_result)
    for key, value in {
        "prediction_result": al_state.json_safe_summary(prediction_result),
        "query_result": al_state.json_safe_summary(query_result),
        "predictions_artifact_id": al_state.latest_reference(
            updated_session,
            "predictions_artifact_id",
        ),
        "model_artifact_id": al_state.latest_reference(
            updated_session,
            "model_artifact_id",
        ),
    }.items():
        if value not in (None, "", {}, []):
            finish_payload[key] = value
    al_actions.publish(context, "al.round.training_finished", finish_payload)
    al_actions.publish(context, "ml.recipe_run.finished", finish_payload)
    al_actions.publish(context, "ml.training.finished", finish_payload)

    return {
        "ok": True,
        "status": "complete",
        "workflow_status": "complete" if not workflow_errors else "partial",
        "workflow_errors": workflow_errors,
        "session_artifact_id": final_session_artifact_id,
        "training_session_artifact_id": training_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": updated_session["session_id"],
        "source_dataset_id": dataset_id,
        "training_dataset_id": materialized["training_dataset_id"],
        "training_artifact_id": materialized["training_artifact_id"],
        "round": materialized["round"],
        "seed": seed,
        "recipe_id": recipe_id,
        "recipe_profile_id": recipe_profile_id,
        "recipe_profile_name": recipe_profile_name,
        "labelled_count": materialized["labelled_count"],
        "training_row_count": len(training_row_ids),
        "training_dataset_reused": bool(
            materialized.get("training_dataset_reused", False)
        ),
        "ml_result": al_state.json_safe_summary(ml_result),
        "prediction_result": al_state.json_safe_summary(prediction_result),
        "query_result": al_state.json_safe_summary(query_result),
    }

def training_dataframe(
    context: Any,
    *,
    dataset_id: str,
    labelled_items: Sequence[Mapping[str, Any]],
    target_column: str,
    required_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    id_column = acquisition.resolve_record_id_column(context, dataset_id)
    row_ids = al_state.stable_unique(str(item["row_id"]) for item in labelled_items if str(item.get("row_id") or ""))
    if not row_ids:
        raise ValueError("No labelled row ids were available to materialise an AL training dataset.")

    required = al_state.stable_unique(str(column) for column in (required_columns or []) if column)
    columns: List[str] = []
    if id_column:
        columns.append(id_column)
    for column in required:
        if column not in columns and column != target_column:
            columns.append(column)

    # Prefer source-level row lookup APIs so small labelled sets do not scan a
    # million-row pool.  Fall back to a narrow dataframe materialisation only for
    # older DatasetManager/Source implementations that do not expose row lookup.
    source_df = _lookup_training_rows_by_id(context, dataset_id=dataset_id, row_ids=row_ids, columns=columns)
    if source_df is None or len(source_df.index) == 0:
        source_df = _materialise_training_columns(context, dataset_id=dataset_id, columns=columns)

    df, id_column = _normalise_training_rows(source_df, id_column=id_column, row_ids=row_ids)
    if df.empty:
        raise ValueError("No labelled rows could be matched in the source dataset.")

    labels_by_id = {str(item["row_id"]): item.get("label") for item in labelled_items}
    if id_column and id_column in df.columns:
        df[target_column] = df[id_column].astype(str).map(labels_by_id)
    else:
        df[target_column] = [labels_by_id.get(str(idx)) for idx in df.index]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Derived AL training dataset is missing required columns: " + ", ".join(missing))
    return df, id_column

def _lookup_training_rows_by_id(
    context: Any,
    *,
    dataset_id: str,
    row_ids: Sequence[str],
    columns: Sequence[str],
) -> Optional[pd.DataFrame]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None
    try:
        source = getattr(datasets, "get_source", lambda *_: None)(dataset_id)
    except Exception:
        source = None
    owners = (datasets, source)
    methods = ("rows_by_ids", "get_rows_by_ids", "take_ids", "lookup_rows")
    for owner in owners:
        if owner is None:
            continue
        for method_name in methods:
            method = getattr(owner, method_name, None)
            if not callable(method):
                continue
            attempts = (
                lambda: method(dataset_id=dataset_id, row_ids=list(row_ids), columns=list(columns)),
                lambda: method(dataset_id=dataset_id, ids=list(row_ids), columns=list(columns)),
                lambda: method(row_ids=list(row_ids), columns=list(columns)),
                lambda: method(ids=list(row_ids), columns=list(columns)),
                lambda: method(list(row_ids), columns=list(columns)),
            )
            for attempt in attempts:
                try:
                    df = _coerce_rows_dataframe(attempt())
                except TypeError:
                    continue
                except Exception:
                    df = None
                if df is not None:
                    return df
    return None

def _coerce_rows_dataframe(rows: Any) -> Optional[pd.DataFrame]:
    if rows is None:
        return None
    if hasattr(rows, "to_pandas"):
        rows = rows.to_pandas()
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        records = [dict(row) for row in rows if isinstance(row, Mapping)]
        if records:
            return pd.DataFrame.from_records(records)
    return None

def _materialise_training_columns(context: Any, *, dataset_id: str, columns: Sequence[str]) -> pd.DataFrame:
    try:
        return context.datasets.get_df(dataset_id, columns=list(columns)) if columns else context.datasets.get_df(dataset_id)
    except TypeError:
        return context.datasets.get_df(dataset_id)

def _normalise_training_rows(source_df: pd.DataFrame, *, id_column: Optional[str], row_ids: Sequence[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    row_id_set = set(row_ids)
    row_order = {row_id: idx for idx, row_id in enumerate(row_ids)}
    if id_column and id_column in source_df.columns:
        df = source_df[source_df[id_column].astype(str).isin(row_id_set)].copy()
        df["__al_row_id_order"] = df[id_column].astype(str).map(row_order)
        df = df.sort_values("__al_row_id_order").drop(columns=["__al_row_id_order"])
        return df, id_column

    index_lookup = {str(index_value): index_value for index_value in source_df.index.tolist()}
    matched_index = [index_lookup[row_id] for row_id in row_ids if row_id in index_lookup]
    if matched_index:
        df = source_df.loc[matched_index].copy()
        if not id_column:
            id_column = "al_record_id"
        if id_column not in df.columns:
            df[id_column] = [str(idx) for idx in df.index]
        return df, id_column

    # Some source-level lookup methods return rows in requested order without an
    # explicit id column.  Preserve those ids instead of discarding a valid narrow
    # lookup result.
    if len(source_df) == len(row_ids):
        df = source_df.copy()
        if not id_column:
            id_column = "al_record_id"
        if id_column not in df.columns:
            df[id_column] = list(row_ids)
        return df, id_column

    return source_df.iloc[0:0].copy(), id_column

def resolve_class_labels(*, params: Mapping[str, Any], session: Mapping[str, Any], labelled_items: Sequence[Mapping[str, Any]]) -> List[str]:
    values = parse_string_list(
        params.get("class_labels")
        or params.get("classes")
        or params.get("label_options")
        or session.get("label_options")
        or []
    )
    for item in labelled_items:
        label = str(item.get("label") or "")
        if label and label not in values and label != al_state.UNSURE_LABEL:
            values.append(label)
    return values

def list_dataset_columns(context: Any, dataset_id: str) -> List[str]:
    try:
        return [str(column) for column in context.datasets.list_columns(dataset_id)]
    except Exception:
        return [str(column) for column in context.datasets.get_df(dataset_id).columns]

def parse_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for chunk in value.splitlines() for part in chunk.split(",")]
    elif isinstance(value, Mapping):
        parts = [str(key).strip() for key in value.keys()]
    elif isinstance(value, Iterable):
        parts = [str(item).strip() for item in value]
    else:
        parts = [str(value).strip()]
    out: List[str] = []
    for item in parts:
        if item and item not in out:
            out.append(item)
    return out

def column_like_recipe_params(params: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in dict(params or {}).items():
        if value in (None, "", [], {}):
            continue
        key_lower = str(key).lower()
        if key_lower.endswith("_column") or key_lower in {
            "target",
            "label",
            "label_column",
            "target_column",
            "image_column",
            "image_path_column",
            "image_uri_column",
            "record_id_column",
            "id_column",
            "group_column",
            "split_column",
            "mask_column",
        }:
            out[str(key)] = str(value)
    return out