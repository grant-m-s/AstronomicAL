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


def _execution_mode(spec: Any) -> str:
    """Managed recipes get protocol guarantees; freeform ones keep their freedom."""
    return str(getattr(spec.recipe_cls, "execution_mode", "freeform") or "freeform")

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
    """Resolve dataset columns for a recipe run."""

    inferred = _registry_mod.infer_column_bindings(
        context,
        dataset_id,
        spec,
        params=params,
    )

    columns = _registry_mod.list_dataset_columns(context, dataset_id)

    record_id_column = (
        params.get("record_id_column")
        or inferred.get("record_id_column")
        or inferred.get("record_id")
    )

    if not record_id_column and "id" in columns:
        record_id_column = "id"

    target_column = (
        params.get("target_column")
        or params.get("label_column")
        or inferred.get("target_column")
        or inferred.get("target_label")
    )

    image_column = (
        params.get("image_column")
        or params.get("image_path_column")
        or inferred.get("image_column")
        or inferred.get("image_path")
        or inferred.get("image_uri")
    )

    input_columns = list(
        params.get("input_columns")
        or params.get("feature_columns")
        or inferred.get("input_columns")
        or []
    )

    if image_column and image_column not in input_columns:
        input_columns.append(image_column)

    record_id_column = str(record_id_column) if record_id_column else ""
    target_column = str(target_column) if target_column else None
    image_column = str(image_column) if image_column else None

    binding = _registry_mod.DataBinding(
        record_id_column=record_id_column,
        target_column=target_column,
        input_columns=[str(column) for column in input_columns if column],
        image_column=image_column,
    )


    execution_mode = str(
        getattr(spec, "execution_mode", "freeform") or "freeform"
    )
    task = str(getattr(spec, "task", "") or "").lower()
    modality = str(getattr(spec, "modality", "") or "").lower()

    if execution_mode == "managed":
        if not binding.record_id_column:
            raise ValueError(
                "Managed recipes require a record-id column."
            )

        if task == "classification" and not binding.target_column:
            raise ValueError(
                "Managed classification recipes require a target column. "
                "Set `target_column`, map `target_label`, or add a matching "
                "target/label column to the dataset."
            )

        if modality == "image" and not binding.image_column:
            raise ValueError(
                "Managed image recipes require an image column. "
                "Set `image_column`, map `image.path`/`image.uri`, or add a "
                "matching image path column to the dataset."
            )

    missing_columns = []

    for column in (
        binding.record_id_column,
        binding.target_column,
        binding.image_column,
    ):
        if column and column not in columns:
            missing_columns.append(column)

    for column in binding.input_columns:
        if column and column not in columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(
            "Resolved recipe column(s) are not present in the dataset: "
            + ", ".join(sorted(set(missing_columns)))
        )

    return binding


def run_ml_recipe_action(
    context: Any,
    request: Any,
    *,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    request = _coerce_request(request)
    params = dict(request.params or {})

    recipe_id = str(params.get("recipe_id") or "").strip()
    if not recipe_id:
        raise ValueError("Missing required recipe_id.")

    dataset_id = (
        params.get("dataset_id")
        or getattr(request, "dataset_id", None)
    )
    if not dataset_id:
        raise ValueError("Missing required dataset_id.")

    registry = context.services.get("core.ml.recipe_registry")
    spec = registry.get(recipe_id)

    defaults = _registry_mod.schema_defaults(
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

    _registry_mod.validate_required_params(
        spec.params_schema or {},
        merged_params,
    )

    execution_mode = str(
        getattr(spec, "execution_mode", "freeform") or "freeform"
    )

    protocol = None
    binding = None

    if execution_mode == "managed":
        protocol = _registry_mod.ProtocolConfig.from_params(
            merged_params
        )
        binding = _build_data_binding(
            context,
            dataset_id,
            spec,
            merged_params,
        )

    run = _registry_mod.make_run_context(
        context=context,
        dataset_id=dataset_id,
        recipe_spec=spec,
        params=merged_params,
        cancel_token=cancel_token,
        run_id=merged_params.get("run_id"),
    )

    if protocol is not None:
        run.protocol = protocol

    if binding is not None:
        run.binding = binding

    if getattr(run, "logger", None) is not None:
        run.logger.set_run_metadata(
            protocol=protocol,
            binding=binding,
            execution_mode=execution_mode,
        )

    recipe = spec.recipe_cls()

    start_payload = {
        "run_id": run.run_id,
        "dataset_id": dataset_id,
        "recipe_id": spec.id,
        "recipe_version": spec.version,
        "recipe_title": spec.title,
        "execution_mode": execution_mode,
        "training_log_artifact_id": run.training_log_artifact_id,
    }

    if protocol is not None:
        start_payload["protocol_id"] = protocol.protocol_id

    _registry_mod.publish(
        context,
        "ml.recipe_run.started",
        start_payload,
    )

    run.log(
        message=f"Recipe `{spec.title}` started.",
        status="running",
        step=0,
        metrics={},
        extra={
            "phase": "started",
            "execution_mode": execution_mode,
        },
    )

    try:
        result = recipe.run(run)

        if result is None:
            result = {}

        if not isinstance(result, dict):
            result = {"result": result}

        result = _registry_mod.json_safe(dict(result))

        artifact_ids = _collect_artifact_ids(
            result,
            training_log_artifact_id=run.training_log_artifact_id,
        )

        run_artifact_payload = {
            "schema_version": 2,
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "execution_mode": execution_mode,
            "status": "complete",
            "params": _registry_mod.json_safe(merged_params),
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

        run_artifact_id = _registry_mod.put_artifact(
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
                model_artifact_id=result.get("model_artifact_id"),
                evaluation_report_artifact_id=result.get(
                    "evaluation_report_artifact_id"
                ),
                predictions_artifact_id=result.get("predictions_artifact_id"),
                test_metrics=result.get("test_metrics"),
            )

        run.log(
            message=f"Recipe `{spec.title}` complete.",
            status="complete",
            metrics={},
            extra={
                "phase": "complete",
                "artifact_ids": artifact_ids,
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
                    "model_artifact_id": result.get("model_artifact_id"),
                    "evaluation_report_artifact_id": result.get(
                        "evaluation_report_artifact_id"
                    ),
                    "predictions_artifact_id": result.get("predictions_artifact_id"),
                    "test_metrics": result.get("test_metrics"),
                },
            )

            if final_training_log_artifact_id:
                artifact_ids["final_training_log_artifact_id"] = final_training_log_artifact_id

        finished_payload = {
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "execution_mode": execution_mode,
            "status": "complete",
            "result": result,
            "artifact_ids": artifact_ids,
            "run_artifact_id": run_artifact_id,
            "training_log_artifact_id": run.training_log_artifact_id,
            "final_training_log_artifact_id": final_training_log_artifact_id,
        }

        _registry_mod.publish(
            context,
            "ml.recipe_run.finished",
            finished_payload,
        )

        return {
            "status": "complete",
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "recipe_id": spec.id,
            "recipe_version": spec.version,
            "recipe_title": spec.title,
            "execution_mode": execution_mode,
            "result": result,
            "artifact_ids": artifact_ids,
            "run_artifact_id": run_artifact_id,
            "training_log_artifact_id": run.training_log_artifact_id,
            "final_training_log_artifact_id": final_training_log_artifact_id,
        }

    except _registry_mod.MLRecipeCancelled as exc:
        message = str(exc) or "Recipe run cancelled."

        if getattr(run, "logger", None) is not None:
            run.logger.update_summary(
                status="cancelled",
                message=message,
                cancelled=True,
            )

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
            "execution_mode": execution_mode,
            "status": "cancelled",
            "message": message,
            "training_log_artifact_id": run.training_log_artifact_id,
            "final_training_log_artifact_id": final_training_log_artifact_id,
        }

        _registry_mod.publish(
            context,
            "ml.recipe_run.finished",
            payload,
        )

        return payload

    except Exception as exc:
        import traceback

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

        try:
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
            "execution_mode": execution_mode,
            "status": "failed",
            "error": message,
            "traceback": tb,
            "training_log_artifact_id": run.training_log_artifact_id,
            "final_training_log_artifact_id": final_training_log_artifact_id,
        }

        _registry_mod.publish(
            context,
            "ml.recipe_run.finished",
            payload,
        )

        raise



