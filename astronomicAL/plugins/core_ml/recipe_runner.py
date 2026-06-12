from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from astronomicAL.platform.plugins.specs import ActionRequest


def _load_sibling(stem: str):
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


_registry_mod = _load_sibling("recipe_registry")


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
    return _registry_mod.active_dataset_id(context)


def run_ml_recipe_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Run a registered ML recipe through the platform action/job contract."""
    request = _coerce_request(request)
    params = dict(request.params or {})

    recipe_id = str(params.get("recipe_id") or "").strip()
    if not recipe_id:
        raise ValueError("run_ml_recipe requires params.recipe_id.")

    dataset_id = _dataset_id(context, request, params)
    if not dataset_id:
        raise ValueError("run_ml_recipe requires params.dataset_id or an active dataset.")

    registry = context.services.get("core.ml.recipe_registry")
    spec = registry.get(recipe_id)

    merged_params = _registry_mod.schema_defaults(spec.params_schema)

    # Infer bindings from the active/current dataset before user params are applied.
    # Mappings win inside infer_recipe_params(); user-supplied params then win over
    # all inferred/default values.
    inferred_params = _registry_mod.infer_recipe_params(context, dataset_id, spec)
    merged_params.update(inferred_params)
    merged_params.update(params)

    # If the UI submitted blank values, fill those blanks from the current dataset.
    for key, value in inferred_params.items():
        if _registry_mod.is_empty_param_value(merged_params.get(key)):
            merged_params[key] = value

    merged_params["recipe_id"] = recipe_id
    merged_params["dataset_id"] = dataset_id

    _registry_mod.validate_required_params(spec.params_schema, merged_params)

    started = time.time()
    run = _registry_mod.make_run_context(
        context=context,
        dataset_id=dataset_id,
        recipe_spec=spec,
        params=merged_params,
        cancel_token=cancel_token,
        run_id=merged_params.get("run_id"),
    )

    run.log(
        message=f"Starting recipe `{spec.title}`.",
        status="running",
        extra={
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "task": spec.task,
            "modality": spec.modality,
        },
    )
    run.publish(
        "ml.recipe_run.started",
        {
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "training_log_artifact_id": run.training_log_artifact_id,
        },
    )

    try:
        recipe = spec.recipe_cls()
        result = recipe.run(run)
        run.check_cancelled()

        result = dict(result or {})
        result.setdefault("run_id", run.run_id)
        result.setdefault("dataset_id", dataset_id)
        result.setdefault("recipe_id", spec.id)
        result.setdefault("recipe_version", spec.version)
        result.setdefault("training_log_artifact_id", run.training_log_artifact_id)
        result.setdefault("duration_seconds", time.time() - started)

        run_payload = {
            "schema_version": 1,
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "task": spec.task,
            "modality": spec.modality,
            "params": merged_params,
            "status": "complete",
            "started_at": started,
            "finished_at": time.time(),
            "duration_seconds": time.time() - started,
            "training_log_artifact_id": run.training_log_artifact_id,
            "result": _registry_mod.json_safe(result),
        }
        run_artifact_id = run.put_artifact("ml.run", run_payload)
        result["run_artifact_id"] = run_artifact_id

        run.log(message=f"Recipe `{spec.title}` complete.", status="complete")
        run.publish(
            "ml.recipe_run.finished",
            {
                "run_id": run.run_id,
                "dataset_id": dataset_id,
                "recipe_id": spec.id,
                "run_artifact_id": run_artifact_id,
                "status": "complete",
            },
        )
        return _registry_mod.json_safe(result)

    except _registry_mod.MLRecipeCancelled as exc:
        cancelled_payload = {
            "schema_version": 1,
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "task": spec.task,
            "modality": spec.modality,
            "params": merged_params,
            "status": "cancelled",
            "started_at": started,
            "finished_at": time.time(),
            "duration_seconds": time.time() - started,
            "training_log_artifact_id": run.training_log_artifact_id,
            "message": str(exc),
        }

        run_artifact_id = run.put_artifact("ml.run", cancelled_payload)

        run.log(
            message=str(exc),
            status="cancelled",
        )

        run.publish(
            "ml.recipe_run.cancelled",
            {
                "run_id": run.run_id,
                "dataset_id": dataset_id,
                "recipe_id": spec.id,
                "run_artifact_id": run_artifact_id,
                "training_log_artifact_id": run.training_log_artifact_id,
                "status": "cancelled",
                "message": str(exc),
            },
        )

        return _registry_mod.json_safe(
            {
                "run_id": run.run_id,
                "dataset_id": dataset_id,
                "recipe_id": spec.id,
                "recipe_version": spec.version,
                "training_log_artifact_id": run.training_log_artifact_id,
                "run_artifact_id": run_artifact_id,
                "status": "cancelled",
                "message": str(exc),
                "duration_seconds": time.time() - started,
            }
        )

    except Exception as exc:
        error_payload = {
            "schema_version": 1,
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "task": spec.task,
            "modality": spec.modality,
            "params": merged_params,
            "status": "failed",
            "started_at": started,
            "finished_at": time.time(),
            "duration_seconds": time.time() - started,
            "training_log_artifact_id": run.training_log_artifact_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        error_artifact_id = run.put_artifact("ml.run", error_payload)
        run.log(message=f"Recipe failed: {exc}", status="failed")
        run.publish(
            "ml.recipe_run.failed",
            {
                "run_id": run.run_id,
                "dataset_id": dataset_id,
                "recipe_id": spec.id,
                "run_artifact_id": error_artifact_id,
                "status": "failed",
                "error": str(exc),
            },
        )
        raise