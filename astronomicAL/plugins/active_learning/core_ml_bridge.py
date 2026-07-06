from __future__ import annotations

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

def materialize_training_set_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    request = al_actions.coerce_request(request)
    params = dict(request.params or {})
    session_artifact_id = str(params.get("session_artifact_id") or "").strip()
    if not session_artifact_id:
        raise ValueError("materialize_training_set requires session_artifact_id.")
    session = al_state.coerce_session(context.artifacts.get(session_artifact_id))
    session_without_queue = al_state.clear_queued_rows(session, reason="training_started")
    if session_without_queue != session:
        session_artifact_id = al_actions.put_session(context, session_without_queue, previous_artifact_id=session_artifact_id)
        session = session_without_queue
        al_actions.publish(
            context,
            "al.query_batch.invalidated",
            {
                "session_artifact_id": session_artifact_id,
                "session_id": session.get("session_id"),
                "reason": "training_started",
            },
        )
    dataset_id = acquisition.session_pool_dataset_id(session)
    if not dataset_id:
        raise ValueError("Could not determine source/pool dataset_id.")
    labelled_items = al_state.labelled_training_items(session)
    if not labelled_items:
        raise ValueError("No verified labels are available for training.")

    round_index = int(session.get("round", 0)) + 1
    target_column = str(params.get("target_column") or session.get("target_column") or "al_label")
    train_dataset_id = str(params.get("train_dataset_id") or "").strip()
    if not train_dataset_id:
        train_dataset_id = al_actions.unique_dataset_id(f"{dataset_id}__al_train_r{round_index}")

    required_columns = [column for column in parse_string_list(params.get("required_columns") or []) if not is_prediction_column(column)]
    profile_info = resolve_recipe_profile_info(context, params)
    recipe_params = sanitize_recipe_params_for_al_training(
        merged_profile_recipe_params(dict(profile_info.get("profile") or {}), params)
    )
    required_columns.extend(column for column in column_like_recipe_params(recipe_params).values() if not is_prediction_column(column))
    required_columns.extend(clean_feature_columns(recipe_params.get("feature_columns") or recipe_params.get("input_columns") or []))
    train_df, id_column = training_dataframe(
        context,
        dataset_id=dataset_id,
        labelled_items=labelled_items,
        target_column=target_column,
        required_columns=required_columns,
    )
    mappings = al_actions.filter_mappings_to_columns(al_actions.dataset_mappings(context, dataset_id), train_df.columns)
    if id_column:
        mappings["record_id"] = id_column
    mappings["target_label"] = target_column

    context.datasets.register(
        train_dataset_id,
        train_df,
        name=f"AL training round {round_index}",
        column_mappings=mappings,
        al_session_id=session["session_id"],
        al_session_artifact_id=session_artifact_id,
        al_round=round_index,
        source_dataset_id=dataset_id,
        pool_dataset_id=dataset_id,
        label_column=target_column,
    )
    al_actions.publish(
        context,
        "dataset.registered",
        {
            "dataset_id": train_dataset_id,
            "source_dataset_id": dataset_id,
            "origin": f"{ORIGIN}.materialize_training_set",
            "kind": "active_learning_training_dataset",
            "session_id": session["session_id"],
            "session_artifact_id": session_artifact_id,
            "round": round_index,
        },
    )

    class_labels = resolve_class_labels(params=params, session=session, labelled_items=labelled_items)
    payload = {
        "schema_version": 3,
        "session_id": session["session_id"],
        "session_artifact_id": session_artifact_id,
        "source_dataset_id": dataset_id,
        "pool_dataset_id": dataset_id,
        "training_dataset_id": train_dataset_id,
        "validation_dataset_id": str(params.get("validation_dataset_id") or session.get("validation_dataset_id") or ""),
        "test_dataset_id": str(params.get("test_dataset_id") or session.get("test_dataset_id") or ""),
        "recipe_id": str(profile_info.get("recipe_id") or session.get("recipe_id") or ""),
        "recipe_profile_id": str(profile_info.get("recipe_profile_id") or session.get("recipe_profile_id") or ""),
        "recipe_profile_name": str(profile_info.get("recipe_profile_name") or session.get("recipe_profile_name") or ""),
        "target_column": target_column,
        "record_id_column": id_column,
        "class_labels": class_labels,
        "classes": class_labels,
        "round": round_index,
        "seed": int(params.get("seed", session.get("seed", 42))),
        "row_ids": [str(item["row_id"]) for item in labelled_items],
        "labels": labelled_items,
        "label_counts": al_state.label_counts(session),
        "counts": al_state.counts(session),
        "session_contract": dict(params.get("session_contract") or session.get("contract") or {}),
    }
    training_artifact_id = context.artifacts.put(
        al_state.ARTIFACT_TRAINING_SET,
        payload,
        dataset_id=train_dataset_id,
        row_ids=payload["row_ids"],
        params={"session_id": session["session_id"], "round": round_index, "target_column": target_column},
    )
    return {
        "ok": True,
        "session": session,
        "session_artifact_id": session_artifact_id,
        "source_dataset_id": dataset_id,
        "training_dataset_id": train_dataset_id,
        "training_artifact_id": training_artifact_id,
        "target_column": target_column,
        "record_id_column": id_column,
        "class_labels": class_labels,
        "labelled_count": len(labelled_items),
        "recipe_id": str(profile_info.get("recipe_id") or session.get("recipe_id") or ""),
        "recipe_profile_id": str(profile_info.get("recipe_profile_id") or session.get("recipe_profile_id") or ""),
        "recipe_profile_name": str(profile_info.get("recipe_profile_name") or session.get("recipe_profile_name") or ""),
        "round": round_index,
    }


def train_from_session_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Optional core.ml bridge: materialise labels, train, optionally predict/query."""

    request = al_actions.coerce_request(request)
    params = dict(request.params or {})
    session_artifact_id = str(params.get("session_artifact_id") or "").strip()
    profile_info = resolve_recipe_profile_info(context, params)
    recipe_profile_id = str(profile_info.get("recipe_profile_id") or "").strip()
    recipe_profile_name = str(profile_info.get("recipe_profile_name") or "").strip()
    recipe_id = str(profile_info.get("recipe_id") or "").strip()
    if not session_artifact_id:
        raise ValueError("train_from_session requires session_artifact_id.")
    if not recipe_profile_id and not recipe_id:
        raise ValueError("train_from_session requires recipe_profile_id or legacy recipe_id.")

    session = al_state.coerce_session(context.artifacts.get(session_artifact_id))
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
        "origin": f"{ORIGIN}.train_from_session",
    }
    al_actions.publish(context, "al.round.training_started", start_payload)
    # Publish generic ML lifecycle events as well so core ML/curve panels that
    # listen for recipe/training lifecycle events can react to bridge-driven AL runs.
    al_actions.publish(context, "ml.recipe_run.started", start_payload)
    al_actions.publish(context, "ml.training.started", start_payload)

    try:
        materialized = materialize_training_set_action(context, ActionRequest(dataset_id=None, row_ids=None, columns=[], params=params, artifact_id=None, origin=f"{ORIGIN}.materialize_training_set"), cancel_token=cancel_token)
        train_dataset_id = str(materialized["training_dataset_id"])
        training_artifact_id = str(materialized["training_artifact_id"])
        target_column = str(materialized["target_column"])
        train_columns = list(context.datasets.get_df(train_dataset_id).columns)
        recipe_params: Dict[str, Any] = sanitize_recipe_params_for_al_training(
            merged_profile_recipe_params(dict(profile_info.get("profile") or {}), params),
            available_columns=train_columns,
        )
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
                "label_options": materialized["class_labels"],
                "class_labels": materialized["class_labels"],
                "classes": materialized["class_labels"],
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
        id_column = materialized.get("record_id_column")
        if id_column:
            recipe_params.setdefault("record_id_column", id_column)

        ml_request = ActionRequest(
            dataset_id=train_dataset_id,
            row_ids=None,
            columns=[],
            params=recipe_params,
            artifact_id=None,
            origin=f"{ORIGIN}.train_from_session",
        )
        ml_result = al_actions.call_registered_action(context, "core.ml.run_ml_recipe", ml_request, cancel_token=cancel_token)
        ml_result = enrich_ml_result_with_referenced_artifacts(context, ml_result)

    except Exception as exc:
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
        al_actions.publish(context, "al.round.training_failed", failure_payload)
        al_actions.publish(context, "ml.recipe_run.failed", failure_payload)
        al_actions.publish(context, "ml.training.failed", failure_payload)
        raise

    updated_session = al_state.with_completed_training_round(
        session,
        training_dataset_id=materialized["training_dataset_id"],
        training_artifact_id=materialized["training_artifact_id"],
        ml_result=ml_result,
    )
    updated_session["recipe_id"] = recipe_id
    updated_session["recipe_profile_id"] = recipe_profile_id
    updated_session["recipe_profile_name"] = recipe_profile_name
    training_session_artifact_id = al_actions.put_session(context, updated_session, previous_artifact_id=session_artifact_id)
    final_session_artifact_id = training_session_artifact_id
    prediction_result: Dict[str, Any] = {}
    query_result: Dict[str, Any] = {}
    workflow_errors: List[Dict[str, str]] = []

    if bool(params.get("auto_predict", True)):
        try:
            model_artifact_id = al_state.latest_reference(updated_session, "model_artifact_id")
            if not model_artifact_id:
                raise ValueError("The training result did not provide model_artifact_id.")
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
            prediction_result = al_actions.call_registered_action(context, "core.ml.predict", prediction_request, cancel_token=cancel_token)
            prediction_result = prediction_result_for_pool(context, prediction_result, pool_dataset_id=dataset_id)
            updated_session = al_state.with_prediction_result(updated_session, prediction_result=prediction_result)
            prediction_session_artifact_id = al_actions.put_session(context, updated_session, previous_artifact_id=training_session_artifact_id)
            final_session_artifact_id = prediction_session_artifact_id

            if bool(params.get("auto_query", False)):
                query_request = ActionRequest(
                    artifact_id=al_state.latest_reference(updated_session, "predictions_artifact_id"),
                    params={
                        "session_artifact_id": prediction_session_artifact_id,
                        "strategy_id": str(params.get("query_strategy_id") or params.get("strategy_id") or "least_confidence"),
                        "k": max(1, int(params.get("query_k", params.get("k", 200)))),
                        "seed": seed,
                        "make_selection": bool(params.get("make_selection", True)),
                    },
                    origin=f"{ORIGIN}.auto_query",
                )
                query_result = al_actions.query_batch_action(context, query_request, cancel_token=cancel_token)
                final_session_artifact_id = str(query_result["session_artifact_id"])
        except Exception as exc:
            workflow_errors.append({"stage": "prediction_or_query", "error": str(exc)})
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
        "origin": f"{ORIGIN}.train_from_session",
    }
    finish_payload = ml_event_payload(finish_base, ml_result)
    # Include latest prediction/query outputs too, when the bridge generated them.
    for key, value in {
        "prediction_result": al_state.json_safe_summary(prediction_result),
        "query_result": al_state.json_safe_summary(query_result),
        "predictions_artifact_id": al_state.latest_reference(updated_session, "predictions_artifact_id"),
        "model_artifact_id": al_state.latest_reference(updated_session, "model_artifact_id"),
    }.items():
        if value not in (None, "", {}, []):
            finish_payload[key] = value
    al_actions.publish(context, "al.round.training_finished", finish_payload)
    al_actions.publish(context, "ml.recipe_run.finished", finish_payload)
    al_actions.publish(context, "ml.training.finished", finish_payload)

    return {
        "ok": True,
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
    source_df = context.datasets.get_df(dataset_id)
    row_ids = [str(item["row_id"]) for item in labelled_items]
    row_id_set = set(row_ids)

    if id_column and id_column in source_df.columns:
        df = source_df[source_df[id_column].astype(str).isin(row_id_set)].copy()
        df["__al_row_id_order"] = df[id_column].astype(str).map({row_id: idx for idx, row_id in enumerate(row_ids)})
        df = df.sort_values("__al_row_id_order").drop(columns=["__al_row_id_order"])
    else:
        index_lookup = {str(index_value): index_value for index_value in source_df.index.tolist()}
        matched_index = [index_lookup[row_id] for row_id in row_ids if row_id in index_lookup]
        df = source_df.loc[matched_index].copy()
        if not id_column:
            id_column = "al_record_id"
            df[id_column] = [str(idx) for idx in df.index]

    if df.empty:
        raise ValueError("No labelled rows could be matched in the source dataset.")

    labels_by_id = {str(item["row_id"]): str(item["label"]) for item in labelled_items}
    if id_column and id_column in df.columns:
        df[target_column] = df[id_column].astype(str).map(labels_by_id)
    else:
        df[target_column] = [labels_by_id.get(str(idx)) for idx in df.index]

    required = [str(column) for column in (required_columns or []) if column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Derived AL training dataset is missing required columns: " + ", ".join(missing))
    return df, id_column


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
