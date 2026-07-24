from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from astronomicAL.platform.plugins.specs import ActionRequest

from .data.binding import build_data_binding
from .recipe_base import make_run_context
from .registry import schema_defaults, validate_required_params
from .resource_estimates import (
    ensure_recipe_resources_allowed,
    estimate_recipe_resources,
)
from .protocol import ProtocolConfig
from .serialization import json_safe
from .runtime import (
    MLRecipeCancelled,
    MLRecipePaused,
    MLRecipeExecutionError,
    cleanup_ml_runtime,
    coerce_action_request,
    is_out_of_memory_error,
    publish,
    put_artifact,
    request_dataset_id,
)

def _coerce_request(request: Any) -> ActionRequest:
    return coerce_action_request(request)

def _dataset_id(context: Any, request: ActionRequest, params: Mapping[str, Any]) -> Optional[str]:
    return request_dataset_id(context, request, params)

def _collect_artifact_ids(
    result: Mapping[str, Any],
    *,
    training_log_artifact_id: Optional[str],
) -> Dict[str, Any]:
    artifact_ids: Dict[str, Any] = {}

    for key, value in dict(result or {}).items():
        if key.endswith("_artifact_id") and value:
            artifact_ids[key] = value

    if training_log_artifact_id:
        artifact_ids["training_log_artifact_id"] = training_log_artifact_id

    return artifact_ids

def _recipe_framework(spec: Any) -> str:
    """Resolve the recipe's framework so make_harness() can pick a harness.

    Prefers an explicit `framework` class attribute, then falls back to tags /
    recipe id. Kept here (rather than only in make_harness) so the value is also
    recorded in params and the training-log artifact.
    """
    framework = str(getattr(spec.recipe_cls, "framework", "") or "").lower()
    if framework:
        return framework

    tags = {str(t).lower() for t in (getattr(spec, "tags", []) or [])}
    recipe_id = str(getattr(spec, "id", "")).lower()
    if "torch" in tags or "pytorch" in tags or "torch" in recipe_id:
        return "torch"
    if "sklearn" in tags or "scikit-learn" in tags or "sklearn" in recipe_id:
        return "sklearn"
    if "tensorflow" in tags or "keras" in tags:
        return "tensorflow"
    return ""

def _build_data_binding(
    context: Any,
    dataset_id: str,
    spec: Any,
    params: Mapping[str, Any],
):
    return build_data_binding(context, dataset_id, spec, params)

def _resolve_profile_params(context: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    profile_key = (
        params.get("recipe_profile_id")
        or params.get("recipe_profile_artifact_id")
        or params.get("profile_id")
        or params.get("profile_artifact_id")
    )
    if not profile_key:
        return params

    store = context.services.get("core.ml.recipe_profile_store")
    profile = store.get(str(profile_key))

    merged: Dict[str, Any] = {}
    merged.update(profile.get("recipe_params") or {})
    merged.update(profile.get("protocol_params") or {})
    merged.update(profile.get("binding_params") or {})

    merged["recipe_id"] = profile.get("recipe_id")
    merged["recipe_profile_id"] = profile.get("profile_id")
    merged["recipe_profile_name"] = profile.get("name")

    # Dataset is intentionally overridable. AL must pass the current
    # round's materialised training dataset here.
    if profile.get("default_dataset_id"):
        merged["dataset_id"] = profile.get("default_dataset_id")

    merged.update(params)
    return merged

def _bool_param(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

def _resolve_resume_state(context: Any, params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    artifact_id = params.get("resume_checkpoint_artifact_id")
    manifest_path = params.get("resume_manifest_path")
    sidecar_path = params.get("resume_checkpoint_path")
    if not any((artifact_id, manifest_path, sidecar_path)):
        return params, {}

    from .resume import load_resume_checkpoint, merge_resume_params

    state = load_resume_checkpoint(
        context=context,
        artifact_id=str(artifact_id) if artifact_id else None,
        manifest_path=str(manifest_path) if manifest_path else None,
        sidecar_path=str(sidecar_path) if sidecar_path else None,
        trusted_root=(
            params.get("ml_artifact_dir")
            or params.get("artifact_dir")
        ),
        allow_external=_bool_param(
            params.get("trust_external_checkpoint", False)
        ),
    )
    saved_params = dict(state.get("params") or {})
    merged = merge_resume_params(saved_params, params)
    merged.update(
        {
            "recipe_id": state.get("recipe_id") or saved_params.get("recipe_id"),
            "recipe_version": state.get("recipe_version") or saved_params.get("recipe_version"),
            "dataset_id": state.get("dataset_id") or saved_params.get("dataset_id"),
            "run_id": state.get("run_id") or saved_params.get("run_id"),
            "training_log_artifact_id": state.get("training_log_artifact_id") or saved_params.get("training_log_artifact_id"),
            "resume_checkpoint_artifact_id": artifact_id or state.get("_resume_artifact_id"),
            "resume_manifest_path": manifest_path or state.get("_resume_manifest_path"),
            "resume_checkpoint_path": sidecar_path or (state.get("_resume_ref") or {}).get("uri"),
        }
    )
    return merged, state

def run_ml_recipe_action(
    context: Any,
    request: Any,
    *,
    cancel_token: Any = None,
    training_control: Any = None,
) -> Dict[str, Any]:
    request = _coerce_request(request)
    params = _resolve_profile_params(context, dict(request.params or {}))
    params, resume_state = _resolve_resume_state(context, params)

    recipe_id = str(params.get("recipe_id") or "").strip()
    if not recipe_id:
        raise ValueError("Missing required recipe_id or recipe_profile_id.")

    dataset_id = _dataset_id(context, request, params)
    if not dataset_id:
        raise ValueError("Missing required dataset_id.")

    registry = context.services.get("core.ml.recipe_registry")
    require_available = getattr(registry, "require_available", None)
    spec = require_available(recipe_id) if callable(require_available) else registry.get(recipe_id)

    defaults = schema_defaults(
        spec.params_schema or {}
    )

    merged_params: Dict[str, Any] = {}
    merged_params.update(defaults)
    merged_params.update(params)

    merged_params["recipe_id"] = spec.id
    merged_params["recipe_version"] = spec.version
    merged_params["recipe_title"] = spec.title
    merged_params["dataset_id"] = dataset_id

    if not merged_params.get("framework"):
        merged_params["framework"] = (
            getattr(spec, "framework", "")
            or _recipe_framework(spec)
        )

    if not merged_params.get("task"):
        merged_params["task"] = getattr(spec, "task", "")

    if not merged_params.get("modality"):
        merged_params["modality"] = getattr(spec, "modality", "")

    validate_required_params(
        spec.params_schema or {},
        merged_params,
    )

    protocol = ProtocolConfig.from_params(merged_params)
    binding = _build_data_binding(
        context,
        dataset_id,
        spec,
        merged_params,
    )

    resource_preflight = estimate_recipe_resources(
        context=context,
        dataset_id=dataset_id,
        recipe_spec=spec,
        params=merged_params,
        feature_columns=list(binding.input_columns or []),
    )
    merged_params["recipe_data_access"] = spec.data_access.to_dict()
    merged_params["resource_preflight"] = resource_preflight.to_dict()
    ensure_recipe_resources_allowed(resource_preflight)

    run = make_run_context(
        context=context,
        dataset_id=dataset_id,
        recipe_spec=spec,
        params=merged_params,
        cancel_token=cancel_token,
        training_control=training_control,
        resume_state=resume_state,
        run_id=merged_params.get("run_id"),
    )

    run.protocol = protocol
    run.binding = binding

    if getattr(run, "logger", None) is not None:
        run.logger.set_run_metadata(
            protocol=protocol,
            binding=binding,
        )

    recipe = spec.recipe_cls()

    start_payload = {
        "run_id": run.run_id,
        "dataset_id": dataset_id,
        "recipe_id": spec.id,
        "recipe_version": spec.version,
        "recipe_title": spec.title,
        "training_log_artifact_id": run.training_log_artifact_id,
        "launcher_session_id": merged_params.get("launcher_session_id"),
        "resumed": bool(resume_state),
        "resume_checkpoint_artifact_id": merged_params.get("resume_checkpoint_artifact_id"),
        "recipe_data_access": spec.data_access.to_dict(),
        "resource_preflight": resource_preflight.to_dict(),
    }

    if protocol is not None:
        start_payload["protocol_id"] = protocol.protocol_id

    publish(
        context,
        "ml.recipe_run.started",
        start_payload,
    )

    if getattr(run, "progress", None) is not None:
        run.progress.report(
            stage="initializing",
            message=f"Recipe `{spec.title}` has started. Preparing the managed training harness.",
            detail=(
                f"Dataset `{dataset_id}` passed resource preflight using "
                f"`{spec.data_access.mode.value}` data access."
            ),
            force=True,
        )
    else:
        run.log(
            message=f"Recipe `{spec.title}` started.",
            status="running",
            step=None,
            metrics={},
            extra={"phase": "started"},
        )

    run_status = "running"
    cleanup_error: Optional[BaseException] = None

    try:
        if getattr(run, "progress", None) is not None:
            run.progress.report(
                stage="preflight",
                message="Recipe, dataset bindings, protocol, and resource limits are valid. Dispatching the recipe harness.",
                detail=(
                    f"Framework `{merged_params.get('framework') or 'unknown'}`, "
                    f"task `{merged_params.get('task') or 'unknown'}`, "
                    f"modality `{merged_params.get('modality') or 'unknown'}`."
                ),
                current=1,
                total=1,
                unit="checks",
                force=True,
            )
        result = recipe.run(run)

        if result is None:
            result = {}

        if not isinstance(result, dict):
            result = {"result": result}

        result = json_safe(dict(result))
        result.setdefault("recipe_data_access", spec.data_access.to_dict())
        result.setdefault("resource_preflight", resource_preflight.to_dict())

        split_dataset_ids = dict(result.get("split_dataset_ids") or {})

        train_dataset_id = (
            result.get("train_dataset_id")
            or split_dataset_ids.get("train")
        )
        validation_dataset_id = (
            result.get("validation_dataset_id")
            or split_dataset_ids.get("validation")
            or split_dataset_ids.get("val")
        )
        test_dataset_id = (
            result.get("test_dataset_id")
            or split_dataset_ids.get("test")
        )

        if train_dataset_id and "train" not in split_dataset_ids:
            split_dataset_ids["train"] = train_dataset_id
        if validation_dataset_id and "validation" not in split_dataset_ids:
            split_dataset_ids["validation"] = validation_dataset_id
        if test_dataset_id and "test" not in split_dataset_ids:
            split_dataset_ids["test"] = test_dataset_id

        artifact_ids = _collect_artifact_ids(
            result,
            training_log_artifact_id=run.training_log_artifact_id,
        )

        run_artifact_payload = {
            "schema_version": 2,
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "source_dataset_id": dataset_id,
            "train_dataset_id": train_dataset_id,
            "validation_dataset_id": validation_dataset_id,
            "test_dataset_id": test_dataset_id,
            "split_dataset_ids": json_safe(split_dataset_ids),
            "profile_id": merged_params.get("recipe_profile_id"),
            "profile_name": merged_params.get("recipe_profile_name"),
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "status": "complete",
            "params": json_safe(merged_params),
            "result": result,
            "artifact_ids": artifact_ids,
            "training_log_artifact_id": run.training_log_artifact_id,
        }

        if protocol is not None:
            run_artifact_payload["protocol"] = {
                "protocol_id": protocol.protocol_id,
                "split_strategy": protocol.split_strategy,
                "validation_source": protocol.validation_source,
                "test_source": protocol.test_source,
                "validation_dataset_id": protocol.validation_dataset_id,
                "test_dataset_id": protocol.test_dataset_id,
                "materialize_split_datasets": getattr(
                    protocol,
                    "materialize_split_datasets",
                    True,
                ),
                "split_dataset_prefix": getattr(
                    protocol,
                    "split_dataset_prefix",
                    None,
                ),
                "group_column": protocol.group_column,
                "split_column": protocol.split_column,
                "validation_size": protocol.validation_size,
                "test_size": protocol.test_size,
                "selection_metric": protocol.selection_metric,
                "selection_mode": protocol.selection_mode,
                "resolved_mode": protocol.resolved_mode(),
                "random_state": protocol.random_state,
            }

        if binding is not None:
            run_artifact_payload["binding"] = {
                "record_id_column": binding.record_id_column,
                "target_column": binding.target_column,
                "input_columns": list(binding.input_columns or []),
                "image_column": binding.image_column,
            }

        if getattr(run, "progress", None) is not None:
            run.progress.report(
                stage="finalizing",
                message="Training outputs are complete. Saving the durable run summary and final training log.",
                force=True,
            )

        run_artifact_id = put_artifact(
            context,
            "ml.run",
            run_artifact_payload,
            dataset_id=dataset_id,
            params=merged_params,
        )

        artifact_ids["run_artifact_id"] = run_artifact_id
        artifact_ids["training_log_artifact_id"] = run.training_log_artifact_id

        if getattr(run, "logger", None) is not None:
            run.logger.update_summary(
                status="complete",
                message=f"Recipe `{spec.title}` complete.",
                run_artifact_id=run_artifact_id,
                artifact_ids=artifact_ids,
                best_epoch=result.get("best_epoch"),
                selection_metric=(
                    result.get("selection_metric")
                    or getattr(protocol, "selection_metric", None)
                ),
                protocol_id=(
                    result.get("protocol_id")
                    or getattr(protocol, "protocol_id", None)
                ),
                split_spec_artifact_id=result.get("split_spec_artifact_id"),
                split_dataset_ids=split_dataset_ids,
                train_dataset_id=train_dataset_id,
                validation_dataset_id=validation_dataset_id,
                test_dataset_id=test_dataset_id,
                model_artifact_id=result.get("model_artifact_id"),
                evaluation_report_artifact_id=result.get(
                    "evaluation_report_artifact_id"
                ),
                predictions_artifact_id=result.get("predictions_artifact_id"),
                test_metrics=result.get("test_metrics"),
            )

        if getattr(run, "progress", None) is not None:
            run.progress.terminal(
                status="complete",
                message=f"Recipe `{spec.title}` completed and all durable outputs were saved.",
            )
        else:
            run.log(
                message=f"Recipe `{spec.title}` complete.",
                status="complete",
                metrics={},
                extra={
                    "phase": "complete",
                    "artifact_ids": artifact_ids,
                    "split_dataset_ids": split_dataset_ids,
                    "train_dataset_id": train_dataset_id,
                    "validation_dataset_id": validation_dataset_id,
                    "test_dataset_id": test_dataset_id,
                },
            )

        final_training_log_artifact_id = None

        if getattr(run, "logger", None) is not None:
            final_training_log_artifact_id = run.logger.persist_final(
                status="complete",
                message=f"Recipe `{spec.title}` complete.",
                extra_summary={
                    "run_artifact_id": run_artifact_id,
                    "artifact_ids": artifact_ids,
                    "best_epoch": result.get("best_epoch"),
                    "selection_metric": (
                        result.get("selection_metric")
                        or getattr(protocol, "selection_metric", None)
                    ),
                    "protocol_id": (
                        result.get("protocol_id")
                        or getattr(protocol, "protocol_id", None)
                    ),
                    "split_spec_artifact_id": result.get("split_spec_artifact_id"),
                    "split_dataset_ids": split_dataset_ids,
                    "train_dataset_id": train_dataset_id,
                    "validation_dataset_id": validation_dataset_id,
                    "test_dataset_id": test_dataset_id,
                    "model_artifact_id": result.get("model_artifact_id"),
                    "evaluation_report_artifact_id": result.get(
                        "evaluation_report_artifact_id"
                    ),
                    "predictions_artifact_id": result.get("predictions_artifact_id"),
                    "test_metrics": result.get("test_metrics"),
                },
            )

            if final_training_log_artifact_id:
                artifact_ids["final_training_log_artifact_id"] = (
                    final_training_log_artifact_id
                )

        finished_payload = {
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "source_dataset_id": dataset_id,
            "train_dataset_id": train_dataset_id,
            "validation_dataset_id": validation_dataset_id,
            "test_dataset_id": test_dataset_id,
            "split_dataset_ids": split_dataset_ids,
            "recipe_profile_id": merged_params.get("recipe_profile_id"),
            "recipe_profile_name": merged_params.get("recipe_profile_name"),
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "status": "complete",
            "result": result,
            "artifact_ids": artifact_ids,
            "run_artifact_id": run_artifact_id,
            "training_log_artifact_id": run.training_log_artifact_id,
            "final_training_log_artifact_id": final_training_log_artifact_id,
            "launcher_session_id": merged_params.get("launcher_session_id"),
        }

        publish(
            context,
            "ml.recipe_run.finished",
            finished_payload,
        )

        run_status = "complete"
        return {
            "status": "complete",
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "source_dataset_id": dataset_id,
            "train_dataset_id": train_dataset_id,
            "validation_dataset_id": validation_dataset_id,
            "test_dataset_id": test_dataset_id,
            "split_dataset_ids": split_dataset_ids,
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "recipe_data_access": spec.data_access.to_dict(),
            "resource_preflight": resource_preflight.to_dict(),
            "result": result,
            "artifact_ids": artifact_ids,
            "run_artifact_id": run_artifact_id,
            "training_log_artifact_id": run.training_log_artifact_id,
            "final_training_log_artifact_id": final_training_log_artifact_id,
        }

    except MLRecipePaused as exc:
        run_status = "paused"
        cleanup_error = exc
        payload = dict(getattr(exc, "pause_payload", {}) or {})
        payload.setdefault("status", "paused")
        payload.setdefault("message", str(exc) or "Training paused.")
        payload.setdefault("run_id", run.run_id)
        payload.setdefault("dataset_id", dataset_id)
        payload.setdefault("recipe_id", spec.id)
        payload.setdefault("recipe_version", spec.version)
        payload.setdefault("recipe_title", spec.title)
        payload.setdefault("training_log_artifact_id", run.training_log_artifact_id)
        payload.setdefault("launcher_session_id", merged_params.get("launcher_session_id"))

        if getattr(run, "progress", None) is not None:
            run.progress.terminal(
                status="paused",
                message=str(payload.get("message") or "Training paused and resumable state was saved."),
            )

        final_training_log_artifact_id = None
        if getattr(run, "logger", None) is not None:
            run.logger.update_summary(**payload)
            final_training_log_artifact_id = run.logger.persist_final(
                status="paused",
                message=str(payload.get("message") or "Training paused."),
                extra_summary=payload,
            )
        payload["final_training_log_artifact_id"] = final_training_log_artifact_id

        run_artifact_payload = {
            "schema_version": 2,
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "source_dataset_id": dataset_id,
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "status": "paused",
            "resumable": True,
            "params": json_safe(merged_params),
            "result": json_safe(payload),
            "artifact_ids": json_safe(payload.get("artifact_ids") or {}),
            "training_log_artifact_id": run.training_log_artifact_id,
            "final_training_log_artifact_id": final_training_log_artifact_id,
        }
        run_artifact_id = put_artifact(
            context,
            "ml.run",
            run_artifact_payload,
            dataset_id=dataset_id,
            params=merged_params,
        )
        payload["run_artifact_id"] = run_artifact_id
        payload.setdefault("artifact_ids", {})["run_artifact_id"] = run_artifact_id

        publish(context, "ml.recipe_run.finished", payload)
        return payload

    except MLRecipeCancelled as exc:
        run_status = "cancelled"
        cleanup_error = exc
        message = str(exc) or "Recipe run cancelled."

        if getattr(run, "logger", None) is not None:
            run.logger.update_summary(
                status="cancelled",
                message=message,
                cancelled=True,
            )

        if getattr(run, "progress", None) is not None:
            run.progress.terminal(status="cancelled", message=message)
        else:
            run.log(
                message=message,
                status="cancelled",
                metrics={},
                extra={
                    "phase": "cancelled",
                    "cancelled": True,
                },
            )

        final_training_log_artifact_id = None

        if getattr(run, "logger", None) is not None:
            final_training_log_artifact_id = run.logger.persist_final(
                status="cancelled",
                message=message,
                extra_summary={
                    "cancelled": True,
                },
            )

        payload = {
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "status": "cancelled",
            "message": message,
            "training_log_artifact_id": run.training_log_artifact_id,
            "final_training_log_artifact_id": final_training_log_artifact_id,
            "launcher_session_id": merged_params.get("launcher_session_id"),
        }

        publish(
            context,
            "ml.recipe_run.finished",
            payload,
        )

        return payload

    except Exception as exc:
        run_status = "failed"
        cleanup_error = exc

        tb = traceback.format_exc()
        message = str(exc)

        if getattr(run, "logger", None) is not None:
            run.logger.update_summary(
                status="failed",
                message=message,
                error=message,
                traceback=tb,
            )

        final_training_log_artifact_id = None

        if getattr(run, "progress", None) is not None:
            run.progress.terminal(status="failed", message=message)

        try:
            if getattr(run, "progress", None) is None:
                run.log(
                    message=message,
                    status="failed",
                    metrics={},
                    extra={
                        "phase": "failed",
                        "error": message,
                        "traceback": tb,
                    },
                )

            if getattr(run, "logger", None) is not None:
                final_training_log_artifact_id = run.logger.persist_final(
                    status="failed",
                    message=message,
                    extra_summary={
                        "error": message,
                        "traceback": tb,
                    },
                )

        except Exception:
            pass

        payload = {
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "status": "failed",
            "error": message,
            "traceback": tb,
            "training_log_artifact_id": run.training_log_artifact_id,
            "final_training_log_artifact_id": final_training_log_artifact_id,
            "launcher_session_id": merged_params.get("launcher_session_id"),
        }

        publish(
            context,
            "ml.recipe_run.finished",
            payload,
        )

        # Futures retain exception tracebacks. A CUDA OOM traceback can therefore
        # keep model/batch tensors alive after the job has failed. Strip the
        # original traceback and raise a small failure object whose detailed
        # diagnostics remain in the durable training-log artifact.
        try:
            exc.__traceback__ = None
            exc.__cause__ = None
            exc.__context__ = None
        except Exception:
            pass
        raise MLRecipeExecutionError(message, failure_payload=payload) from None

    finally:
        aggressive = bool(cleanup_error and is_out_of_memory_error(cleanup_error))

        keep_work_dir = bool(merged_params.get("keep_work_dir", False))
        if run_status == "failed":
            keep_work_dir = keep_work_dir or bool(
                merged_params.get("keep_failed_work_dir", False)
            )

        work_dir = getattr(run, "work_dir", None)
        if work_dir and not keep_work_dir:
            try:
                shutil.rmtree(Path(work_dir), ignore_errors=True)
            except Exception:
                pass

        recipe = None
        cleanup_ml_runtime(
            reason=f"recipe run {run_status}",
            aggressive=aggressive or run_status == "failed",
        )