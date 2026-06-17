# =============================================================================
# ADD TO recipe_registry.py
#
# These pieces turn the recipe framework into a "harness owns the protocol,
# recipe owns the internals" system that ANY recipe can use:
#
#   - ProtocolConfig / Partition / Partitions / DataBinding / TrainingComponents
#   - new optional hook methods on MLRecipe (internals + a loop that delegates)
#   - ManagedMLRecipe: run() delegates to a harness; recipe never owns protocol
#   - RunHarness: abstract skeleton owning partition/select/audit
#   - TorchClassificationHarness: the concrete torch+classification core
#   - make_harness(): factory so the runner stays modality-agnostic
#
# Freeform recipes (ExternalPythonRecipe) keep overriding run() directly — the
# escape hatch is untouched. Only Managed recipes get the protocol guarantees.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field

import json
import math
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import os
import re

class MLRunLogger:
    """Artifact-backed logger for long recipe runs.

    The logger records progress and keeps the live ml.training_log artifact
    curve-friendly. It does not own the scientific protocol.

    Managed recipes:
        - RunHarness owns split, validation selection, best_epoch, test metrics.
        - Logger mirrors those facts when the runner/harness provides them.

    Freeform recipes:
        - Logger can infer optimize_metric/best_epoch as a fallback.
    """

    def __init__(
        self,
        *,
        context: Any,
        run_id: str,
        dataset_id: Optional[str],
        training_log_artifact_id: Optional[str],
        recipe_spec: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        protocol: Any = None,
        binding: Any = None,
        execution_mode: str = "freeform",
    ) -> None:
        self.context = context
        self.run_id = str(run_id)
        self.dataset_id = dataset_id
        self.training_log_artifact_id = training_log_artifact_id
        self.recipe_spec = recipe_spec
        self.params = dict(params or {})
        self.protocol = protocol
        self.binding = binding
        self.execution_mode = str(execution_mode or "freeform")
        self.started_at = time.time()

        # Full chronological event stream.
        self.events: List[Dict[str, Any]] = []

        # Curve-friendly rows, merged by epoch.
        self.epochs_by_epoch: Dict[int, Dict[str, Any]] = {}

        # Authoritative facts supplied later by runner/harness.
        self.summary: Dict[str, Any] = {}

        self._ensure_training_log_artifact()

    def persist_final(
        self,
        *,
        status: str,
        message: str,
        extra_summary: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        """Write a final durable ml.training_log artifact.

        The live training log is intentionally persist=False for fast updates.
        This method creates a final persisted copy at the end of the run.

        Returns:
            The final persisted training-log artifact id, or the live id if a
            separate persisted artifact could not be created.
        """

        if extra_summary:
            self.summary.update(json_safe(dict(extra_summary)))

        self.summary.update(
            {
                "status": status,
                "message": message,
                "finalized_at": time.time(),
            }
        )

        payload = self._payload(
            status=status,
            message=message,
        )

        payload["final"] = True
        payload["live_training_log_artifact_id"] = self.training_log_artifact_id

        artifacts = getattr(self.context, "artifacts", None)
        put = getattr(artifacts, "put", None)

        if not callable(put):
            return self.training_log_artifact_id
        
        log_dir = ml_artifact_root(self.context, self.params) / "training_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / f"{_safe_path_part(self.run_id)}.training_log.json"
        log_path.write_text(
            json.dumps(json_safe(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        payload["log_ref"] = {
            "storage": "local_file",
            "uri": str(log_path),
            "path": str(log_path),
            "format": "json",
        }

        try:
            final_id = put(
                "ml.training_log",
                json_safe(payload),
                dataset_id=self.dataset_id,
                params=json_safe(self.params),
                persist=True,
            )
        except TypeError:
            # Compatibility with smaller ArtifactStore signatures.
            try:
                final_id = put(
                    "ml.training_log",
                    json_safe(payload),
                    dataset_id=self.dataset_id,
                    params=json_safe(self.params),
                )
            except TypeError:
                final_id = put(
                    "ml.training_log",
                    json_safe(payload),
                )
        except Exception:
            return self.training_log_artifact_id

        self.summary["final_training_log_artifact_id"] = final_id

        publish(
            self.context,
            "ml.training_log.finalized",
            {
                "run_id": self.run_id,
                "dataset_id": self.dataset_id,
                "artifact_id": final_id,
                "training_log_artifact_id": final_id,
                "live_training_log_artifact_id": self.training_log_artifact_id,
                "status": status,
                "message": message,
            },
        )

        return final_id

    def set_run_metadata(
        self,
        *,
        protocol: Any = None,
        binding: Any = None,
        execution_mode: Optional[str] = None,
    ) -> None:
        """Attach protocol/binding once the runner has resolved them."""

        if protocol is not None:
            self.protocol = protocol

        if binding is not None:
            self.binding = binding

        if execution_mode:
            self.execution_mode = str(execution_mode)

        self._update_training_log(
            self._payload(
                status=str(self.summary.get("status", "queued")),
                message=str(
                    self.summary.get(
                        "message",
                        "Recipe run initialised.",
                    )
                ),
            )
        )

    def update_summary(self, **kwargs: Any) -> None:
        """Store authoritative run facts from the runner or harness.

        Examples:
            best_epoch
            selection_metric
            protocol_id
            split_spec_artifact_id
            model_artifact_id
            evaluation_report_artifact_id
            predictions_artifact_id
            test_metrics
            error
        """

        clean = json_safe(dict(kwargs))
        self.summary.update(clean)

        self._update_training_log(
            self._payload(
                status=str(self.summary.get("status", "running")),
                message=str(self.summary.get("message", "Recipe run updated.")),
            )
        )

    def log(
        self,
        *,
        message: str,
        status: str = "running",
        step: Optional[int] = None,
        total: Optional[int] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Append a progress event and update the live training-log artifact."""

        now = time.time()

        metrics_dict = json_safe(dict(metrics or {}))
        extra_dict = json_safe(dict(extra or {}))

        event: Dict[str, Any] = {
            "time": now,
            "elapsed_seconds": now - self.started_at,
            "status": status,
            "message": message,
            "step": step,
            "total": total,
            "metrics": metrics_dict,
        }

        # Flatten metrics for table/curve consumers.
        if isinstance(metrics_dict, Mapping):
            for key, value in metrics_dict.items():
                event[str(key)] = value

        # Flatten small scalar extras.
        if isinstance(extra_dict, Mapping):
            for key, value in extra_dict.items():
                if self._is_scalar(value):
                    event[str(key)] = value
                else:
                    event[str(key)] = json_safe(value)

        # The curves panel uses epoch. Promote step if needed.
        if event.get("epoch") is None and step is not None:
            event["epoch"] = step

        self.events.append(event)

        epoch = self._coerce_epoch(event)
        if epoch is not None:
            self._upsert_epoch(epoch, event)

        self.summary.update(
            {
                "status": status,
                "message": message,
                "updated_at": now,
            }
        )

        payload = self._payload(status=status, message=message)
        self._update_training_log(payload)

        self._publish_progress(
            status=status,
            message=message,
            step=step,
            total=total,
            metrics=metrics_dict,
        )

    def metric(
        self,
        *,
        epoch: int,
        split: str,
        metrics: Mapping[str, Any],
    ) -> None:
        """Convenience helper for freeform recipes.

        Example:
            logger.metric(epoch=3, split="val", metrics={"loss": 0.2})

        Produces:
            epoch=3, val_loss=0.2
        """

        split_name = str(split or "").strip()
        prefixed = {
            f"{split_name}_{key}": value
            for key, value in dict(metrics or {}).items()
        }
        prefixed["epoch"] = int(epoch)

        self.log(
            message=f"{split_name} metrics for epoch {epoch}.",
            status="running",
            step=int(epoch),
            metrics=prefixed,
            extra={
                "epoch": int(epoch),
                "split": split_name,
            },
        )

    def finish(
        self,
        *,
        status: str = "complete",
        message: str = "Recipe run complete.",
        persist: bool = True,
        **summary: Any,
    ) -> Optional[str]:
        """Mark the run finished and optionally persist a final training log."""

        if summary:
            self.update_summary(**summary)

        self.log(
            message=message,
            status=status,
        )

        if persist:
            return self.persist_final(
                status=status,
                message=message,
                extra_summary=summary,
            )

        return self.training_log_artifact_id

    def _coerce_epoch(self, row: Mapping[str, Any]) -> Optional[int]:
        value = row.get("epoch")

        if value is None:
            value = row.get("step")

        if value is None:
            return None

        try:
            return int(value)
        except Exception:
            return None

    def _upsert_epoch(
        self,
        epoch: int,
        row: Mapping[str, Any],
    ) -> None:
        """Merge metrics for the same epoch into one curve row."""

        existing = self.epochs_by_epoch.setdefault(epoch, {"epoch": epoch})

        for key, value in row.items():
            key = str(key)

            if key == "metrics" and isinstance(value, Mapping):
                for metric_key, metric_value in value.items():
                    existing[str(metric_key)] = metric_value
                existing["metrics"] = json_safe(dict(value))
                continue

            if key in {
                "time",
                "elapsed_seconds",
                "status",
                "message",
                "step",
                "total",
                "split",
            }:
                existing[key] = value
                continue

            if self._is_scalar(value):
                existing[key] = value

    def _is_scalar(self, value: Any) -> bool:
        return value is None or isinstance(value, (str, int, float, bool))

    def _epoch_rows(self) -> List[Dict[str, Any]]:
        return [
            json_safe(dict(row))
            for _, row in sorted(
                self.epochs_by_epoch.items(),
                key=lambda item: item[0],
            )
        ]

    def _payload(
        self,
        *,
        status: str,
        message: str,
    ) -> Dict[str, Any]:
        epochs = self._epoch_rows()
        optimize_metric = self._optimize_metric(epochs)
        best_epoch = self._authoritative_best_epoch(
            epochs,
            optimize_metric,
        )

        payload: Dict[str, Any] = {
            "schema_version": 2,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "recipe_id": getattr(
                self.recipe_spec,
                "id",
                self.params.get("recipe_id", ""),
            ),
            "recipe_version": getattr(
                self.recipe_spec,
                "version",
                self.params.get("recipe_version", ""),
            ),
            "recipe_title": getattr(
                self.recipe_spec,
                "title",
                self.params.get("recipe_title", ""),
            ),
            "model_title": self._model_title(),
            "framework": self._framework(),
            "task": getattr(
                self.recipe_spec,
                "task",
                self.params.get("task", ""),
            ),
            "modality": getattr(
                self.recipe_spec,
                "modality",
                self.params.get("modality", ""),
            ),
            "execution_mode": self.execution_mode,
            "status": status,
            "message": message,
            "optimize_metric": optimize_metric,
            "selection_metric": optimize_metric,
            "best_epoch": best_epoch,
            "started_at": self.started_at,
            "updated_at": time.time(),
            "params": json_safe(self.params),
            "protocol": self._protocol_payload(),
            "binding": self._binding_payload(),
            "summary": json_safe(dict(self.summary)),
            "events": json_safe(self.events),
            "epochs": json_safe(epochs),
        }

        # Promote common summary fields to top level for panels.
        for key in (
            "protocol_id",
            "split_spec_artifact_id",
            "model_artifact_id",
            "evaluation_report_artifact_id",
            "predictions_artifact_id",
            "run_artifact_id",
            "test_metrics",
            "artifact_ids",
            "error",
            "traceback",
            "cancelled",
            "tuning",
            "tuning_trials",
            "tuning_current_trial",
            "tuning_current_trial_epochs",
            "tuning_best_params",
            "tuning_best_value",
        ):
            if key in self.summary:
                payload[key] = json_safe(self.summary[key])

        return payload

    def _protocol_payload(self) -> Dict[str, Any]:
        protocol = self.protocol

        if protocol is None:
            return {}

        resolved_mode = None
        try:
            resolved_mode = protocol.resolved_mode()
        except Exception:
            resolved_mode = None

        return {
            "protocol_id": getattr(protocol, "protocol_id", ""),
            "split_strategy": getattr(protocol, "split_strategy", ""),
            "validation_source": getattr(protocol, "validation_source", ""),
            "test_source": getattr(protocol, "test_source", ""),
            "validation_dataset_id": getattr(
                protocol,
                "validation_dataset_id",
                None,
            ),
            "test_dataset_id": getattr(
                protocol,
                "test_dataset_id",
                None,
            ),
            "group_column": getattr(protocol, "group_column", None),
            "split_column": getattr(protocol, "split_column", None),
            "validation_size": getattr(protocol, "validation_size", None),
            "test_size": getattr(protocol, "test_size", None),
            "selection_metric": getattr(protocol, "selection_metric", ""),
            "selection_mode": getattr(protocol, "selection_mode", ""),
            "resolved_mode": resolved_mode,
            "random_state": getattr(protocol, "random_state", None),
        }

    def _binding_payload(self) -> Dict[str, Any]:
        binding = self.binding

        if binding is None:
            return {}

        return {
            "record_id_column": getattr(binding, "record_id_column", ""),
            "target_column": getattr(binding, "target_column", None),
            "input_columns": list(
                getattr(binding, "input_columns", []) or []
            ),
            "image_column": getattr(binding, "image_column", None),
        }

    def _model_title(self) -> str:
        for key in (
            "model_title",
            "model_id",
            "architecture",
            "recipe_title",
        ):
            value = self.params.get(key)
            if value:
                return str(value)

        title = getattr(self.recipe_spec, "title", None)
        if title:
            return str(title)

        recipe_id = (
            getattr(self.recipe_spec, "id", None)
            or self.params.get("recipe_id")
        )
        return str(recipe_id or "unknown")

    def _framework(self) -> str:
        value = self.params.get("framework")
        if value:
            return str(value)

        value = getattr(self.recipe_spec, "framework", "")
        if value:
            return str(value)

        tags = set(
            str(tag).lower()
            for tag in getattr(self.recipe_spec, "tags", []) or []
        )
        recipe_id = str(getattr(self.recipe_spec, "id", "")).lower()

        if "torch" in tags or "pytorch" in tags or "torch" in recipe_id:
            return "torch"

        if (
            "sklearn" in tags
            or "scikit-learn" in tags
            or "sklearn" in recipe_id
        ):
            return "sklearn"

        if "tensorflow" in tags or "keras" in tags:
            return "tensorflow"

        return ""

    def _optimize_metric(
        self,
        epochs: Sequence[Mapping[str, Any]],
    ) -> str:
        # Managed recipes: protocol is authoritative.
        protocol_metric = getattr(self.protocol, "selection_metric", None)
        if protocol_metric:
            return str(protocol_metric)

        # Runner/harness may explicitly set this.
        for key in ("selection_metric", "optimize_metric"):
            value = self.summary.get(key)
            if value:
                return str(value)

        # Freeform fallback.
        value = self.params.get("optimize_metric")
        if value:
            return str(value)

        # Last resort: infer from available epoch columns.
        for key in (
            "val_accuracy",
            "validation_accuracy",
            "val_f1_macro",
            "val_balanced_accuracy",
            "val_loss",
            "validation_loss",
            "train_loss",
            "loss",
        ):
            if any(row.get(key) is not None for row in epochs):
                return key

        return ""

    def _authoritative_best_epoch(
        self,
        epochs: Sequence[Mapping[str, Any]],
        optimize_metric: str,
    ) -> Optional[int]:
        # Managed harness should set this.
        value = self.summary.get("best_epoch")
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass

        # Freeform fallback.
        return self._infer_best_epoch(
            epochs,
            optimize_metric,
        )

    def _infer_best_epoch(
        self,
        epochs: Sequence[Mapping[str, Any]],
        optimize_metric: str,
    ) -> Optional[int]:
        if not epochs or not optimize_metric:
            return None

        minimize = any(
            token in optimize_metric.lower()
            for token in ("loss", "error", "mae", "mse", "rmse")
        )

        best_value = None
        best_epoch = None

        for row in epochs:
            value = row.get(optimize_metric)
            epoch = row.get("epoch")

            if value is None or epoch is None:
                continue

            try:
                value_float = float(value)
                epoch_int = int(epoch)
            except Exception:
                continue

            if best_value is None:
                best_value = value_float
                best_epoch = epoch_int
            elif minimize and value_float < best_value:
                best_value = value_float
                best_epoch = epoch_int
            elif not minimize and value_float > best_value:
                best_value = value_float
                best_epoch = epoch_int

        return best_epoch

    def _ensure_training_log_artifact(self) -> None:
        if self.training_log_artifact_id:
            return

        artifacts = getattr(self.context, "artifacts", None)
        put = getattr(artifacts, "put", None)

        if not callable(put):
            return

        payload = {
            "schema_version": 2,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "recipe_id": getattr(
                self.recipe_spec,
                "id",
                self.params.get("recipe_id", ""),
            ),
            "recipe_version": getattr(
                self.recipe_spec,
                "version",
                "",
            ),
            "recipe_title": getattr(
                self.recipe_spec,
                "title",
                "",
            ),
            "model_title": self._model_title(),
            "framework": self._framework(),
            "task": getattr(
                self.recipe_spec,
                "task",
                self.params.get("task", ""),
            ),
            "modality": getattr(
                self.recipe_spec,
                "modality",
                self.params.get("modality", ""),
            ),
            "execution_mode": self.execution_mode,
            "status": "queued",
            "message": "Recipe run initialised.",
            "started_at": self.started_at,
            "updated_at": time.time(),
            "final": False,
            "events": [],
            "epochs": [],
        }

        self.training_log_artifact_id = put(
            "ml.training_log",
            json_safe(payload),
            dataset_id=self.dataset_id,
            params=json_safe(self.params),
            persist=False,
        )

        publish(
            self.context,
            "ml.training_log.created",
            {
                "run_id": self.run_id,
                "dataset_id": self.dataset_id,
                "artifact_id": self.training_log_artifact_id,
                "training_log_artifact_id": self.training_log_artifact_id,
            },
        )

    def _update_training_log(
        self,
        payload: Mapping[str, Any],
    ) -> None:
        if not self.training_log_artifact_id:
            self._ensure_training_log_artifact()

        if not self.training_log_artifact_id:
            return

        artifacts = getattr(self.context, "artifacts", None)
        clean_payload = json_safe(dict(payload))

        # Preferred future API, if you add it.
        update = getattr(artifacts, "update", None)
        if callable(update):
            update(self.training_log_artifact_id, clean_payload)
            return

        # Current in-memory artifact behavior.
        get = getattr(artifacts, "get", None)
        if callable(get):
            try:
                existing = get(self.training_log_artifact_id)
            except Exception:
                existing = None

            if isinstance(existing, dict):
                existing.clear()
                existing.update(clean_payload)
                return

        # Fallback for current ArtifactStore internals.
        payloads = getattr(artifacts, "_payloads", None)
        if isinstance(payloads, dict):
            payloads[self.training_log_artifact_id] = clean_payload

    def _publish_progress(
        self,
        *,
        status: str,
        message: str,
        step: Optional[int],
        total: Optional[int],
        metrics: Mapping[str, Any],
    ) -> None:
        payload = {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "artifact_id": self.training_log_artifact_id,
            "training_log_artifact_id": self.training_log_artifact_id,
            "status": status,
            "message": message,
            "step": step,
            "total": total,
            "metrics": json_safe(dict(metrics or {})),
        }

        publish(
            self.context,
            "ml.recipe_run.progress",
            payload,
        )

        publish(
            self.context,
            "ml.training_log.updated",
            payload,
        )


def json_safe(value: Any) -> Any:
    """Best-effort conversion of numpy/pandas/path/scalar values to JSON-safe values."""
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


def _safe_path_part(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")

    return text or fallback


def ml_artifact_root(
    context: Any,
    params: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Stable local root for ML sidecar files.

    Do NOT use run.work_dir for durable trained models. run.work_dir is temp.

    Priority:
        1. params["ml_artifact_dir"]
        2. params["artifact_dir"]
        3. env ASTRONOMICAL_ML_ARTIFACT_DIR
        4. context persistence-ish directory, if discoverable
        5. ~/.astronomical/ml_artifacts
    """

    params = dict(params or {})

    explicit = (
        params.get("ml_artifact_dir")
        or params.get("artifact_dir")
        or os.environ.get("ASTRONOMICAL_ML_ARTIFACT_DIR")
    )

    if explicit:
        root = Path(str(explicit)).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root

    persistence = getattr(context, "persistence", None)

    for attr in (
        "root_dir",
        "base_dir",
        "workspace_dir",
        "artifact_dir",
        "path",
    ):
        value = getattr(persistence, attr, None)
        if value:
            root = Path(str(value)).expanduser() / "ml_artifacts"
            root.mkdir(parents=True, exist_ok=True)
            return root

    root = Path.home() / ".astronomical" / "ml_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def ml_run_artifact_dir(
    run: Any,
    *,
    kind: str,
) -> Path:
    """Stable directory for files belonging to one ML recipe run."""

    recipe_id = _safe_path_part(
        getattr(run, "recipe_id", None) or run.params.get("recipe_id"),
        fallback="recipe",
    )
    run_id = _safe_path_part(
        getattr(run, "run_id", None) or run.params.get("run_id"),
        fallback=uuid.uuid4().hex,
    )
    kind = _safe_path_part(kind, fallback="artifact")

    root = ml_artifact_root(
        getattr(run, "context", None),
        getattr(run, "params", {}),
    )

    path = root / recipe_id / run_id / kind
    path.mkdir(parents=True, exist_ok=True)
    return path



def schema_defaults(schema: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract default values from a JSON-schema-like parameter schema."""
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    defaults: Dict[str, Any] = {}
    for name, spec in properties.items():
        if isinstance(spec, Mapping) and "default" in spec:
            defaults[str(name)] = spec["default"]
    return defaults


def validate_required_params(schema: Mapping[str, Any], params: Mapping[str, Any]) -> None:
    required = list(schema.get("required", []) or []) if isinstance(schema, Mapping) else []
    missing = []
    for key in required:
        value = params.get(key)
        if value is None or value == "":
            missing.append(str(key))
    if missing:
        raise ValueError(f"Missing required recipe parameter(s): {', '.join(missing)}")

def infer_recipe_params(
    context: Any,
    dataset_id: str,
    recipe_spec: Any,
) -> Dict[str, Any]:
    """Return inferred values only for parameters exposed by the recipe schema."""
    bindings = infer_column_bindings(context, dataset_id)
    schema = getattr(recipe_spec, "params_schema", None) or {}
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}

    inferred: Dict[str, Any] = {}
    for key in (
        "record_id_column",
        "image_column",
        "target_column",
        "mask_column",
        "feature_columns",
    ):
        if key in properties and key in bindings:
            inferred[key] = bindings[key]

    # Common aliases used by some recipe/action schemas.
    alias_map = {
        "label_column": "target_column",
        "class_column": "target_column",
        "image_path_column": "image_column",
        "image_uri_column": "image_column",
        "features": "feature_columns",
    }
    for alias, source_key in alias_map.items():
        if alias in properties and source_key in bindings:
            inferred[alias] = bindings[source_key]

    return inferred


def is_empty_param_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}

def make_run_context(
    *,
    context: Any,
    dataset_id: str,
    recipe_spec: Any,
    params: Mapping[str, Any],
    cancel_token: Any = None,
    run_id: Optional[str] = None,
) -> MLRunContext:
    run_id = str(run_id or params.get("run_id") or uuid.uuid4().hex)

    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f"astronomical-ml-recipe-{run_id[:8]}-"
        )
    )

    training_log_artifact_id = (
        params.get("training_log_artifact_id")
        or None
    )

    logger = MLRunLogger(
        context=context,
        run_id=run_id,
        dataset_id=dataset_id,
        training_log_artifact_id=training_log_artifact_id,
        recipe_spec=recipe_spec,
        params=params,
        execution_mode=getattr(recipe_spec, "execution_mode", "freeform"),
    )

    return MLRunContext(
        context=context,
        dataset_id=dataset_id,
        recipe_id=recipe_spec.id,
        recipe_version=recipe_spec.version,
        params=dict(params),
        run_id=run_id,
        work_dir=work_dir,
        cancel_token=cancel_token,
        training_log_artifact_id=logger.training_log_artifact_id,
        logger=logger,
    )

def active_dataset_id(context: Any) -> Optional[str]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None

    for method_name in ("active_id", "get_active_id", "active_dataset_id"):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                value = None
            if value:
                return str(value)

    value = getattr(datasets, "active", None)
    if isinstance(value, str):
        return value

    return None

def list_dataset_ids(context: Any) -> List[str]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return []

    candidates: List[str] = []

    for method_name in ("list_ids", "ids", "dataset_ids"):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                values = method()
                candidates.extend(str(v) for v in values if v)
            except Exception:
                pass

    for method_name in ("list", "list_datasets"):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                values = method()
            except Exception:
                values = []
            for value in values or []:
                if isinstance(value, str):
                    candidates.append(value)
                else:
                    dataset_id = (
                        getattr(value, "dataset_id", None)
                        or getattr(value, "id", None)
                        or getattr(value, "name", None)
                    )
                    if dataset_id:
                        candidates.append(str(dataset_id))

    active = active_dataset_id(context)
    if active:
        candidates.append(active)

    deduped: List[str] = []
    seen = set()
    for value in candidates:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped




class MLRecipeCancelled(RuntimeError):
    """Raised when a recipe run is cancelled by the user."""

class CancellationToken:
    """Thread-safe cancellation token for recipe runs.

    The recipe launcher owns this token. Recipes should call
    run.check_cancelled() inside long loops so cancellation can happen cleanly.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = "ML recipe run was cancelled."

    def cancel(self, reason: str = "ML recipe run was cancelled.") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def is_set(self) -> bool:
        return self._event.is_set()


def check_cancelled(cancel_token: Any = None) -> None:
    """Raise MLRecipeCancelled if a cancellation token has been set."""

    if cancel_token is None:
        return

    cancelled = False

    for attr in ("cancelled", "is_cancelled", "is_set"):
        value = getattr(cancel_token, attr, None)

        try:
            if callable(value):
                cancelled = bool(value())
            elif value is not None:
                cancelled = bool(value)
        except Exception:
            cancelled = False

        if cancelled:
            break

    if cancelled:
        reason = getattr(
            cancel_token,
            "reason",
            "ML recipe run was cancelled.",
        )
        raise MLRecipeCancelled(str(reason))


def publish(
    context: Any,
    event_type: str,
    payload: Optional[Mapping[str, Any]] = None,
) -> None:
    """Best-effort platform event publish."""

    events = getattr(context, "events", None)
    if events is None:
        return

    payload_dict = json_safe(dict(payload or {}))

    for method_name in ("publish", "emit"):
        method = getattr(events, method_name, None)
        if callable(method):
            try:
                method(event_type, payload_dict)
                return
            except TypeError:
                try:
                    method(
                        {
                            "type": event_type,
                            "payload": payload_dict,
                        }
                    )
                    return
                except Exception:
                    return
            except Exception:
                return


def put_artifact(
    context: Any,
    artifact_type: str,
    payload: Mapping[str, Any],
    *,
    dataset_id: Optional[str] = None,
    row_ids: Optional[Sequence[Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    persist: bool = True,
) -> Optional[str]:
    """Create an artifact using the platform ArtifactStore."""

    artifacts = getattr(context, "artifacts", None)
    if artifacts is None:
        return None

    put = getattr(artifacts, "put", None)
    if not callable(put):
        return None

    clean_payload = json_safe(dict(payload or {}))

    kwargs = {
        "dataset_id": dataset_id,
        "row_ids": list(row_ids) if row_ids is not None else None,
        "params": json_safe(dict(params or {})),
        "persist": persist,
    }

    # Different ArtifactStore versions may not accept every kwarg.
    while True:
        try:
            return put(
                artifact_type,
                clean_payload,
                **kwargs,
            )
        except TypeError as exc:
            message = str(exc)

            removed = False
            for key in list(kwargs):
                if key in message:
                    kwargs.pop(key, None)
                    removed = True
                    break

            if not removed:
                # Fallback to the smallest likely signature.
                try:
                    return put(artifact_type, clean_payload)
                except Exception:
                    return None
        except Exception:
            return None


def get_dataset_frame(
    context: Any,
    dataset_id: str,
    *,
    columns: Optional[Sequence[str]] = None,
):
    """Materialise a dataset as a pandas DataFrame.

    This supports both old DataFrame-style DatasetManager APIs and the newer
    source-backed manager APIs.
    """

    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise ValueError("No dataset manager is available on context.")

    dataset_id = str(dataset_id)

    # Preferred explicit APIs.
    for method_name in (
        "get_frame",
        "get_dataframe",
        "get_df",
        "materialize",
        "to_dataframe",
    ):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                df = method(dataset_id, columns=list(columns or []))
                if df is not None:
                    return df
            except TypeError:
                try:
                    df = method(dataset_id)
                    if df is not None:
                        if columns:
                            return df.loc[:, list(columns)]
                        return df
                except Exception:
                    pass
            except Exception:
                pass

    # Object-returning APIs.
    for method_name in ("get", "dataset"):
        method = getattr(datasets, method_name, None)
        if not callable(method):
            continue

        try:
            dataset = method(dataset_id)
        except Exception:
            dataset = None

        if dataset is None:
            continue

        for attr in ("df", "dataframe", "frame"):
            df = getattr(dataset, attr, None)
            if df is not None:
                if columns:
                    return df.loc[:, list(columns)]
                return df

        for obj_method_name in (
            "get_frame",
            "get_dataframe",
            "get_df",
            "to_dataframe",
            "materialize",
        ):
            obj_method = getattr(dataset, obj_method_name, None)
            if callable(obj_method):
                try:
                    df = obj_method(columns=list(columns or []))
                    if df is not None:
                        return df
                except TypeError:
                    df = obj_method()
                    if columns:
                        return df.loc[:, list(columns)]
                    return df

    raise KeyError(f"Could not materialise dataset {dataset_id!r}.")


def list_dataset_columns(
    context: Any,
    dataset_id: Optional[str],
) -> List[str]:
    """Return available columns for a dataset."""

    if not dataset_id:
        return []

    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return []

    dataset_id = str(dataset_id)

    for method_name in (
        "columns",
        "list_columns",
        "get_columns",
        "dataset_columns",
    ):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                return [str(c) for c in method(dataset_id)]
            except Exception:
                pass

    try:
        df = get_dataset_frame(context, dataset_id)
    except Exception:
        return []

    try:
        return [str(c) for c in list(df.columns)]
    except Exception:
        return []


def mapped_column(
    context: Any,
    dataset_id: str,
    semantic_name: str,
) -> Optional[str]:
    """Resolve a semantic column mapping from DatasetManager if available."""

    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None

    dataset_id = str(dataset_id)
    semantic_name = str(semantic_name)

    for method_name in (
        "mapped_column",
        "get_mapped_column",
        "resolve_mapping",
        "mapping_for",
    ):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                value = method(dataset_id, semantic_name)
                if value:
                    return str(value)
            except Exception:
                pass

    for method_name in (
        "get_mappings",
        "mappings",
        "column_mappings",
    ):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                mappings = method(dataset_id)
            except Exception:
                mappings = None

            if isinstance(mappings, Mapping):
                value = mappings.get(semantic_name)
                if value:
                    return str(value)

    return None


def infer_column_bindings(
    context: Any,
    dataset_id: str,
    recipe_spec: Any = None,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Infer record-id, target, image, and feature columns.

    The old method probably did not accept recipe_spec/params. The new runner
    calls it with both, so keep this widened signature.
    """

    params = dict(params or {})
    columns = list_dataset_columns(context, dataset_id)
    column_set = {str(c) for c in columns}
    lower_to_original = {str(c).lower(): str(c) for c in columns}

    def from_params(*names: str) -> Optional[str]:
        for name in names:
            value = params.get(name)
            if value:
                return str(value)
        return None

    def from_mapping(*semantic_names: str) -> Optional[str]:
        for semantic_name in semantic_names:
            value = mapped_column(context, dataset_id, semantic_name)
            if value:
                return value
        return None

    def from_aliases(*aliases: str) -> Optional[str]:
        for alias in aliases:
            value = lower_to_original.get(alias.lower())
            if value:
                return value
        return None

    record_id_column = (
        from_params("record_id_column", "record_id")
        or from_mapping("record_id")
        or from_aliases("record_id", "object_id", "source_id", "id")
    )

    target_column = (
        from_params("target_column", "label_column", "class_column")
        or from_mapping("target_label", "label", "class")
        or from_aliases(
            "target_label",
            "target",
            "label",
            "class",
            "y",
        )
    )

    image_column = (
        from_params("image_column", "image_path_column", "image_uri_column")
        or from_mapping("image.path", "image.uri", "image", "cutout.path")
        or from_aliases(
            "image_path",
            "image_uri",
            "image",
            "path",
            "filepath",
            "file_path",
            "cutout_path",
        )
    )

    feature_columns = params.get("feature_columns") or params.get("input_columns")
    if isinstance(feature_columns, str):
        feature_columns = [
            part.strip()
            for part in feature_columns.split(",")
            if part.strip()
        ]

    if not feature_columns:
        excluded = {
            value
            for value in (
                record_id_column,
                target_column,
                image_column,
            )
            if value
        }
        feature_columns = [
            c
            for c in columns
            if c not in excluded
        ]

    result: Dict[str, Any] = {}

    if record_id_column:
        result["record_id_column"] = record_id_column
        result["record_id"] = record_id_column

    if target_column:
        result["target_column"] = target_column
        result["target_label"] = target_column

    if image_column:
        result["image_column"] = image_column
        result["image_path"] = image_column
        result["image_uri"] = image_column

    result["feature_columns"] = list(feature_columns or [])
    result["input_columns"] = list(feature_columns or [])

    return result

# -----------------------------------------------------------------------------
# Protocol + binding value types
# -----------------------------------------------------------------------------

def resolve_selection_mode(metric: str, mode: str = "auto") -> str:
    if mode in ("max", "min"):
        return mode
    m = str(metric or "").lower()
    return "min" if any(t in m for t in ("loss", "error", "mae", "mse", "rmse")) else "max"

@dataclass
class ProtocolConfig:
    """Experiment protocol enforced by the harness.

    split_strategy controls only how the selected/main dataset is split when
    validation_source or test_source is "split".

    validation_source:
        split   -> create validation from selected dataset
        dataset -> use protocol_validation_dataset_id

    test_source:
        split   -> create test from selected dataset
        dataset -> use protocol_test_dataset_id
        none    -> no test set
    """

    split_strategy: str = "random"  # random | by_group | temporal | predefined

    validation_source: str = "split"  # split | dataset
    test_source: str = "split"        # split | dataset | none

    validation_dataset_id: Optional[str] = None
    test_dataset_id: Optional[str] = None

    group_column: Optional[str] = None
    split_column: Optional[str] = None

    validation_size: float = 0.1
    test_size: float = 0.2

    selection_metric: str = "val_accuracy"
    selection_mode: str = "auto"
    random_state: int = 42
    protocol_id: str = ""

    def resolved_mode(self) -> str:
        return resolve_selection_mode(
            self.selection_metric,
            self.selection_mode,
        )

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "ProtocolConfig":
        def num(key, default):
            try:
                return float(params.get(key, default))
            except Exception:
                return default

        cfg = cls(
            split_strategy=str(
                params.get("protocol_split_strategy", "random")
            ),
            validation_source=str(
                params.get("protocol_validation_source", "split")
            ),
            test_source=str(
                params.get("protocol_test_source", "split")
            ),
            validation_dataset_id=(
                params.get("protocol_validation_dataset_id") or None
            ),
            test_dataset_id=(
                params.get("protocol_test_dataset_id") or None
            ),
            group_column=(
                params.get("protocol_group_column") or None
            ),
            split_column=(
                params.get("protocol_split_column") or None
            ),
            validation_size=num("protocol_validation_size", 0.1),
            test_size=num("protocol_test_size", 0.2),
            selection_metric=str(
                params.get("protocol_selection_metric", "val_accuracy")
            ),
            selection_mode=str(
                params.get("protocol_selection_mode", "auto")
            ),
            random_state=int(params.get("protocol_random_state", 42)),
        )

        allowed_split_strategies = {
            "random",
            "by_group",
            "temporal",
            "predefined",
        }
        if cfg.split_strategy not in allowed_split_strategies:
            raise ValueError(
                f"Unknown protocol_split_strategy {cfg.split_strategy!r}. "
                f"Expected one of {sorted(allowed_split_strategies)}."
            )

        if cfg.validation_source not in {"split", "dataset"}:
            raise ValueError(
                "protocol_validation_source must be 'split' or 'dataset'."
            )

        if cfg.test_source not in {"split", "dataset", "none"}:
            raise ValueError(
                "protocol_test_source must be 'split', 'dataset', or 'none'."
            )

        train_dataset_id = params.get("dataset_id")

        if cfg.validation_source == "dataset":
            if not cfg.validation_dataset_id:
                raise ValueError(
                    "Validation source is 'dataset', but no validation dataset "
                    "was selected."
                )

            if cfg.validation_dataset_id == train_dataset_id:
                raise ValueError(
                    "Validation dataset must be different from the selected "
                    "training dataset."
                )

        if cfg.test_source == "dataset":
            if not cfg.test_dataset_id:
                raise ValueError(
                    "Test source is 'dataset', but no test dataset was selected."
                )

            if cfg.test_dataset_id == train_dataset_id:
                raise ValueError(
                    "Test dataset must be different from the selected training dataset."
                )

        if (
            cfg.validation_source == "dataset"
            and cfg.test_source == "dataset"
            and cfg.validation_dataset_id
            and cfg.test_dataset_id
            and cfg.validation_dataset_id == cfg.test_dataset_id
        ):
            raise ValueError(
                "Validation and test datasets must be different."
            )

        val_from_split = cfg.validation_source == "split"
        test_from_split = cfg.test_source == "split"

        if val_from_split and cfg.split_strategy != "predefined":
            if cfg.validation_size <= 0:
                raise ValueError(
                    "protocol_validation_size must be > 0 when validation "
                    "is split from the selected dataset."
                )

        if test_from_split and cfg.split_strategy != "predefined":
            if cfg.test_size <= 0:
                raise ValueError(
                    "protocol_test_size must be > 0 when test is split from "
                    "the selected dataset. Choose 'No test set' instead."
                )

        if cfg.split_strategy != "predefined":
            split_total = 0.0
            if val_from_split:
                split_total += cfg.validation_size
            if test_from_split:
                split_total += cfg.test_size

            if split_total >= 1.0:
                raise ValueError(
                    "Fractions split from the selected dataset must sum to < 1.0."
                )

        if cfg.split_strategy in {"by_group", "temporal"} and (
            val_from_split or test_from_split
        ):
            if not cfg.group_column:
                raise ValueError(
                    f"{cfg.split_strategy!r} split requires protocol_group_column."
                )

        if cfg.split_strategy == "predefined" and (
            val_from_split or test_from_split
        ):
            if not cfg.split_column:
                raise ValueError(
                    "predefined split requires protocol_split_column."
                )

        cfg.protocol_id = _stable_protocol_id(cfg)
        return cfg


def _stable_protocol_id(cfg: ProtocolConfig) -> str:
    import hashlib

    raw = "|".join(
        str(x)
        for x in (
            cfg.split_strategy,
            cfg.validation_source,
            cfg.test_source,
            cfg.validation_dataset_id,
            cfg.test_dataset_id,
            cfg.group_column,
            cfg.split_column,
            cfg.validation_size,
            cfg.test_size,
            cfg.selection_metric,
            cfg.resolved_mode(),
            cfg.random_state,
        )
    )

    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class DataBinding:
    """Resolved column binding. The harness resolves this (mappings/inference);
    the recipe only READS it inside load_sample. Data binding is not a recipe
    internal."""
    record_id_column: str
    target_column: Optional[str]
    input_columns: List[str] = field(default_factory=list)
    image_column: Optional[str] = None


@dataclass
class Partition:
    name: str
    record_ids: List[str]
    labels: List[str]
    classes: List[str]
    dataset_id: Optional[str] = None
    source: str = "split"  # split | dataset | none

    def __len__(self) -> int:
        return len(self.record_ids)


@dataclass
class Partitions:
    train: Partition
    val: Partition
    test: Optional[Partition]
    strategy: str
    validation_source: str
    test_source: str
    group_column: Optional[str]
    random_state: int
    protocol_id: str
    target_column: str
    record_id_column: str
    train_dataset_id: Optional[str] = None
    validation_dataset_id: Optional[str] = None
    test_dataset_id: Optional[str] = None


@dataclass
class TrainingComponents:
    """What configure_training returns — all the expert's, none of it protocol."""
    optimizer: Any
    scheduler: Any = None
    criterion: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetSpec:
    """What 'the model output' means for a run.

    The harness derives this once (via _target_spec) and threads it everywhere
    that used to assume a class count. Classification carries the class list;
    regression carries the number of continuous outputs. New task kinds add a
    new `kind` plus a matching harness, without touching the protocol flow.
    """

    kind: str = "classification"        # classification | regression
    classes: List[str] = field(default_factory=list)
    n_outputs: int = 1

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_outputs(self) -> int:
        """Width of the model's output layer."""
        if self.kind == "classification":
            return len(self.classes)
        return int(self.n_outputs)

# -----------------------------------------------------------------------------
# MLRecipe — base class for ALL recipes.
#
# Defined here, AFTER the value-type dataclasses (so the "TrainingComponents"
# annotations resolve) and BEFORE ManagedMLRecipe (which subclasses it) and the
# harness (which calls recipe hooks). Replaces the commented-out `# class
# MLRecipe:` documentation block.
#
# Identity attributes are read by the registry to build a RecipeSpec, and a few
# are read directly off the class:
#   - recipe_runner._recipe_framework / make_harness  -> recipe_cls.framework
#   - recipe_runner._execution_mode / panel           -> recipe_cls.execution_mode
#   - TorchClassificationHarness._write_model_artifact -> recipe.task / .modality
# -----------------------------------------------------------------------------

class MLRecipe:
    # --- identity / spec (defaults; concrete recipes override) --------------
    id: str = ""
    title: str = ""
    version: str = "0.0.0"
    task: str = "custom"
    modality: str = "custom"
    framework: str = ""          # "" -> runner falls back to tags/id inference
    complexity: str = "expert"
    author: str = ""
    description: str = ""
    tags: list = []
    required_mappings: list = []
    optional_mappings: list = []
    produces: list = []
    params_schema: Dict[str, Any] = {"type": "object", "properties": {}}

    # Freeform by default: the recipe owns its own run() and makes no
    # scientific-validity promise. ManagedMLRecipe overrides this to "managed".
    execution_mode: str = "freeform"

    @classmethod
    def spec(cls) -> "RecipeSpec":
        return RecipeSpec.from_recipe_cls(cls)

    # --- top-level entry point ----------------------------------------------
    def run(self, run) -> Dict[str, Any]:
        """Execute the recipe and return a result dict.

        The base does not implement this. Freeform recipes (e.g.
        ExternalPythonRecipe) override run() directly; managed recipes inherit
        ManagedMLRecipe.run(), which delegates to a harness.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run(). Override run() "
            "for a freeform recipe, or subclass ManagedMLRecipe and implement "
            "the internals (build_model / configure_training / load_sample / fit)."
        )

    # --- internals: the expert's contribution (managed recipes) -------------
    # The harness calls these; it never lets the recipe touch the protocol.

    def build_model(self, run, *, num_classes: int):
        raise NotImplementedError(
            "build_model(run, *, num_classes) must be implemented by a managed recipe."
        )

    def configure_training(self, run, model) -> "TrainingComponents":
        raise NotImplementedError(
            "configure_training(run, model) must be implemented by a managed recipe."
        )

    def train_transform(self, run):
        return None      # applied to TRAIN rows only

    def eval_transform(self, run):
        return None      # applied to VAL and TEST rows

    def load_sample(self, run, row):
        """Map one dataframe row to a raw input. The harness handles record-id
        keying, batching, and the data binding (run.binding); the recipe only
        knows how to read a single sample."""
        raise NotImplementedError(
            "load_sample(run, row) must be implemented by a managed recipe."
        )

    def eval_forward(self, model, batch_inputs):
        """Default forward used by the harness for val/test eval and the
        output-dimension check. Inherited as-is by recipes (e.g.
        CIFARResNetRecipe) that do a plain model(x) forward pass."""
        return model(batch_inputs)

    # --- the loop: the expert owns it; selection is delegated ---------------
    def fit(self, run, *, model, components: "TrainingComponents", train_loader, harness):
        """Run the training loop. The recipe sees ONLY train_loader and must
        call harness.report_epoch(epoch, model, train_metrics=...) at least
        once so the harness can evaluate the validation partition and select the
        best epoch. It receives no val/test loader and computes no selection
        metric, by design."""
        raise NotImplementedError(
            "fit(run, *, model, components, train_loader, harness) must be "
            "implemented by a managed recipe."
        )

@dataclass
class MLRunContext:
    context: Any
    dataset_id: str
    recipe_id: str
    recipe_version: str
    params: Dict[str, Any]
    run_id: str
    work_dir: Path
    cancel_token: Any = None
    training_log_artifact_id: Optional[str] = None
    logger: Optional[MLRunLogger] = None

    # Set by the runner after construction (run.protocol / run.binding) and read
    # by the harness via getattr(run, "protocol"/"binding", None). Declared here
    # so the contract is explicit and typed. String annotations avoid coupling to
    # the definition order of ProtocolConfig / DataBinding in this module.
    protocol: Optional["ProtocolConfig"] = None
    binding: Optional["DataBinding"] = None

    def check_cancelled(self) -> None:
        check_cancelled(self.cancel_token)

    def publish(self, event_type: str, payload: Optional[Mapping[str, Any]] = None) -> None:
        publish(self.context, event_type, payload)

    def put_artifact(
        self,
        artifact_type: str,
        payload: Mapping[str, Any],
        *,
        row_ids: Optional[Sequence[Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        return put_artifact(
            self.context,
            artifact_type,
            payload,
            dataset_id=self.dataset_id,
            row_ids=row_ids,
            params=params or self.params,
        )

    def log(
        self,
        *,
        message: str,
        status: str = "running",
        step: Optional[int] = None,
        total: Optional[int] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self.logger is not None:
            self.logger.log(
                message=message,
                status=status,
                step=step,
                total=total,
                metrics=metrics,
                extra=extra,
            )

class ManagedMLRecipe(MLRecipe):  # noqa: F821  (MLRecipe defined above in this file)
    """Recipe whose run() is the harness, not hand-written.

    A managed recipe implements only internals (build_model / configure_training
    / transforms / load_sample / fit). It NEVER partitions, evaluates val/test,
    selects the best epoch, or writes the scientific artifacts. The harness does
    all of that and refuses to delegate it.
    """

    execution_mode = "managed"

    def run(self, run) -> Dict[str, Any]:
        harness = make_harness(run, self)
        return harness.execute()

# -----------------------------------------------------------------------------
# RecipeSpec + MLRecipeRegistry
#
# Add to recipe_registry.py. No forward dependency on MLRecipe (it reads recipe
# classes purely by attribute), so it can live anywhere after the typing imports
# — logically it belongs just after the MLRecipe base class.
#
# Consumed by:
#   recipe_panel.py  -> registry.list() (spec.title/.id), registry.get(id)
#                       (spec.recipe_cls/.title/.id/.version/.description/.task/
#                        .modality/.complexity/.required_mappings/.produces/
#                        .params_schema)
#   recipe_runner.py -> context.services.get("core.ml.recipe_registry"),
#                       registry.get(id) (spec.recipe_cls/.params_schema/.id/
#                        .version/.title/.task/.modality/.tags)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RecipeSpec:
    recipe_cls: type
    id: str
    title: str
    version: str
    task: str
    modality: str
    framework: str
    complexity: str
    author: str
    description: str
    tags: List[str]
    required_mappings: List[str]
    optional_mappings: List[str]
    produces: List[str]
    params_schema: Dict[str, Any]
    execution_mode: str

    @classmethod
    def from_recipe_cls(cls, recipe_cls: type) -> "RecipeSpec":
        if not isinstance(recipe_cls, type):
            raise TypeError(
                f"RecipeSpec expects a recipe class, got {recipe_cls!r}."
            )

        def g(name: str, default: Any) -> Any:
            return getattr(recipe_cls, name, default)

        rid = str(g("id", "") or "").strip()
        if not rid:
            raise ValueError(
                f"Recipe {recipe_cls.__name__} has no `id`; cannot register it."
            )

        return cls(
            recipe_cls=recipe_cls,
            id=rid,
            title=str(g("title", "") or rid),
            version=str(g("version", "0.0.0")),
            task=str(g("task", "custom")),
            modality=str(g("modality", "custom")),
            framework=str(g("framework", "") or ""),
            complexity=str(g("complexity", "expert")),
            author=str(g("author", "")),
            description=str(g("description", "")),
            tags=list(g("tags", []) or []),
            required_mappings=list(g("required_mappings", []) or []),
            optional_mappings=list(g("optional_mappings", []) or []),
            produces=list(g("produces", []) or []),
            params_schema=dict(
                g("params_schema", {"type": "object", "properties": {}}) or {}
            ),
            execution_mode=str(g("execution_mode", "freeform") or "freeform"),
        )


class MLRecipeRegistry:
    """In-memory registry of recipe specs, keyed by recipe id.

    Registered as the `core.ml.recipe_registry` service by the plugin's
    register(). `list()` preserves registration order, so the first recipe the
    plugin registers becomes the launcher's default selection.
    """

    def __init__(self, recipes: Optional[Sequence[type]] = None) -> None:
        self._specs: Dict[str, RecipeSpec] = {}
        if recipes:
            self.register_many(recipes)

    def register(self, recipe_cls: type, *, replace: bool = True) -> RecipeSpec:
        spec = RecipeSpec.from_recipe_cls(recipe_cls)
        if spec.id in self._specs and not replace:
            raise ValueError(f"Recipe id {spec.id!r} is already registered.")
        self._specs[spec.id] = spec
        return spec

    def register_many(self, recipe_classes: Sequence[type]) -> None:
        for recipe_cls in recipe_classes:
            self.register(recipe_cls)

    def unregister(self, recipe_id: str) -> None:
        self._specs.pop(str(recipe_id), None)

    def get(self, recipe_id: str) -> RecipeSpec:
        try:
            return self._specs[str(recipe_id)]
        except KeyError:
            raise KeyError(
                f"No recipe registered with id {recipe_id!r}. "
                f"Known recipe ids: {sorted(self._specs)}"
            ) from None

    def list(self) -> List[RecipeSpec]:
        return list(self._specs.values())

    def ids(self) -> List[str]:
        return list(self._specs.keys())

    def __contains__(self, recipe_id: object) -> bool:
        return str(recipe_id) in self._specs

    def __len__(self) -> int:
        return len(self._specs)

# -----------------------------------------------------------------------------
# RunHarness — abstract skeleton; owns the protocol, calls modality hooks
# -----------------------------------------------------------------------------

class RunHarness:
    """Owns partitioning, selection, test evaluation, and the audit artifacts.

    The recipe is handed ONLY the train loader and a report_epoch() callback.
    It never receives val/test loaders and never computes the selection metric,
    so 'test used in training' and 'best epoch chosen on test' are not
    expressible by a recipe, correct or malicious.

    Subclasses implement the framework/modality specifics:
        _make_loader, _evaluate, _snapshot, _restore,
        _assert_output_dim, _write_model_artifact
    """

    def __init__(self, run, recipe: "MLRecipe"):
        self.run = run
        self.recipe = recipe
        self.protocol: ProtocolConfig = getattr(run, "protocol", None) or ProtocolConfig()
        self.binding: DataBinding = getattr(run, "binding", None)
        self._frame = None
        self._partition_frames: Dict[str, Any] = {}
        self._val_loader = None
        self._best_score = None
        self._best_epoch = None
        self._best_state = None
        self._history: List[Dict[str, Any]] = []

    # ---- the ONE primitive the recipe calls each epoch ---------------------
    def report_epoch(
        self,
        epoch: int,
        model,
        *,
        train_metrics: Dict[str, Any],
    ):
        """Called by managed recipes once per epoch.

        The recipe reports training metrics only. The harness evaluates validation,
        chooses the best epoch, snapshots best weights, and logs a merged curve row.
        """

        self.run.check_cancelled()

        val_metrics = self._evaluate(model, self._val_loader)[0]

        row: Dict[str, Any] = {"epoch": int(epoch)}

        row.update(
            {
                f"train_{key}": value
                for key, value in dict(train_metrics or {}).items()
            }
        )

        row.update(
            {
                f"val_{key}": value
                for key, value in dict(val_metrics or {}).items()
            }
        )

        self._history.append(row)

        score = row.get(self.protocol.selection_metric)

        if score is not None:
            try:
                score_float = float(score)
            except Exception:
                score_float = None

            if score_float is not None and self._is_better(score_float):
                self._best_score = score_float
                self._best_epoch = int(epoch)
                self._best_state = self._snapshot(model)

        self.run.log(
            message=f"epoch {epoch}",
            status="running",
            step=int(epoch),
            metrics=row,
            extra={
                "phase": "epoch",
                "selection_metric": self.protocol.selection_metric,
                "selection_mode": self.protocol.resolved_mode(),
                "best_epoch_so_far": self._best_epoch,
                "best_score_so_far": self._best_score,
            },
        )

        return val_metrics

# ---- task abstraction (overridable; default classification) ------------
    def _task_kind(self) -> str:
        task = str(
            getattr(self.recipe, "task", "")
            or self.run.params.get("task", "")
            or ""
        ).lower()
        if task in {"regression", "regressor"}:
            return "regression"
        return "classification"

    def _target_spec(self, parts: Partitions) -> TargetSpec:
        """Derive what the output means from the partitions + task kind.

        Classification reads the class list discovered during partitioning;
        regression reports the number of continuous outputs (default 1, override
        via params['n_outputs']). Subclasses override for exotic outputs.
        """
        if self._task_kind() == "regression":
            return TargetSpec(
                kind="regression",
                classes=[],
                n_outputs=int(self.run.params.get("n_outputs", 1) or 1),
            )
        return TargetSpec(
            kind="classification",
            classes=list(parts.train.classes),
        )

    def _build_model(self, parts: Partitions, target: TargetSpec):
        """Call the recipe's build_model with task-correct kwargs.

        New recipes may accept `target=`; legacy recipes accept `num_classes=`
        (interpreted as output width). The output-dim check then verifies the
        head matches `target` regardless of which signature was used.
        """
        recipe = self.recipe
        try:
            return recipe.build_model(self.run, target=target)
        except TypeError:
            return recipe.build_model(self.run, num_classes=target.num_outputs)

# ---- the protocol flow (NOT overridable by recipes) --------------------
    def execute(self) -> Dict[str, Any]:
        recipe, run = self.recipe, self.run

        parts = self._partition()                               # PROTOCOL
        split_spec_id = self._write_split_spec(parts)           # AUDIT: row-ids on disk

        target = self._target_spec(parts)                       # what 'output' means
        model = self._build_model(parts, target)
        components = recipe.configure_training(run, model)

        train_loader = self._make_loader(parts.train, train=True)
        self._val_loader = self._make_loader(parts.val, train=False)

        self._assert_output_dim(model, target, train_loader)    # kuangliu-10 trap

        # Expert's loop. It only sees train_loader + report_epoch(harness).
        recipe.fit(run, model=model, components=components,
                   train_loader=train_loader, harness=self)

        if self._best_state is None:
            raise RuntimeError(
                "Recipe completed without calling harness.report_epoch(...). "
                "A managed recipe must report at least one epoch so the harness "
                "can select on the validation partition.")
        self._restore(model, self._best_state)                  # best-epoch weights

        test_metrics, test_records = {}, []
        if parts.test is not None and len(parts.test) > 0:       # test LAST, ONCE
            test_loader = self._make_loader(parts.test, train=False)
            test_metrics, test_records = self._evaluate(
                model, test_loader, return_records=True)

        model_artifact_id = self._write_model_artifact(
            model,
            parts,
            target,
            split_spec_artifact_id=split_spec_id,
        )

        eval_id = self._write_evaluation_report(
            parts=parts,
            split_spec_id=split_spec_id,
            test_metrics=test_metrics,
            model_artifact_id=model_artifact_id,
        )

        predictions_id = None
        if test_records:
            predictions_id = self._write_predictions(
                test_records,
                parts,
                model_artifact_id,
            )

        result = {
            "status": "complete",
            "model_artifact_id": model_artifact_id,
            "evaluation_report_artifact_id": eval_id,
            "predictions_artifact_id": predictions_id,
            "split_spec_artifact_id": split_spec_id,
            "best_epoch": self._best_epoch,
            "best_score": self._best_score,
            "selection_metric": self.protocol.selection_metric,
            "selection_mode": self.protocol.resolved_mode(),
            "protocol_id": self.protocol.protocol_id,
            "task_kind": target.kind,
            "test_metrics": test_metrics,
            "history": self._history,
        }

        if getattr(self.run, "logger", None) is not None:
            self.run.logger.update_summary(**result)

        return result


    def _load_partition_frame(
        self,
        dataset_id: str,
        *,
        columns: Sequence[str],
        role: str,
    ):
        b = self.binding

        if not dataset_id:
            raise ValueError(f"{role} dataset id is missing.")

        df = get_dataset_frame(
            self.run.context,
            dataset_id,
            columns=list(columns),
        )

        required = [b.record_id_column]

        if b.target_column:
            required.append(b.target_column)

        missing = [
            column
            for column in required
            if column and column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{role} dataset {dataset_id!r} is missing required column(s): "
                + ", ".join(missing)
            )

        if b.target_column:
            df = df.dropna(subset=[b.target_column])

        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"{role} dataset {dataset_id!r} has no usable rows after "
                "dropping missing targets."
            )

        duplicate_count = int(
            df[b.record_id_column].astype(str).duplicated().sum()
        )

        if duplicate_count:
            raise ValueError(
                f"{role} dataset {dataset_id!r} has {duplicate_count} duplicate "
                f"record IDs in column {b.record_id_column!r}."
            )

        return df


    def _partition_from_frame(
        self,
        *,
        name: str,
        frame,
        dataset_id: str,
        classes: List[str],
        source: str,
    ) -> Partition:
        b = self.binding

        record_ids = frame[b.record_id_column].astype(str).tolist()

        labels = (
            frame[b.target_column].astype(str).tolist()
            if b.target_column
            else ["" for _ in record_ids]
        )

        return Partition(
            name=name,
            record_ids=record_ids,
            labels=labels,
            classes=list(classes),
            dataset_id=dataset_id,
            source=source,
        )


    def _base_split_indices(
        self,
        df,
        labels: Sequence[str],
        *,
        need_val: bool,
        need_test: bool,
    ):
        p = self.protocol

        if p.split_strategy == "predefined":
            return self._predefined_base_indices(
                df,
                need_val=need_val,
                need_test=need_test,
            )

        if p.split_strategy == "by_group":
            return self._grouped_base_indices(
                df,
                labels,
                need_val=need_val,
                need_test=need_test,
            )

        if p.split_strategy == "temporal":
            return self._temporal_base_indices(
                df,
                need_val=need_val,
                need_test=need_test,
            )

        return self._random_base_indices(
            labels,
            need_val=need_val,
            need_test=need_test,
        )


    def _random_base_indices(
        self,
        labels: Sequence[str],
        *,
        need_val: bool,
        need_test: bool,
    ):
        import numpy as np
        from sklearn.model_selection import train_test_split

        p = self.protocol
        idx = np.arange(len(labels))
        labels_arr = np.asarray(labels)

        rest = idx
        test = np.array([], dtype=int)

        if need_test:
            strat = labels_arr if _stratifiable(labels_arr) else None
            rest, test = train_test_split(
                idx,
                test_size=p.test_size,
                random_state=p.random_state,
                stratify=strat,
            )

        val = np.array([], dtype=int)

        if need_val:
            rest_labels = labels_arr[rest]
            strat_rest = rest_labels if _stratifiable(rest_labels) else None

            if need_test:
                rel_val = p.validation_size / max(
                    1e-9,
                    1.0 - p.test_size,
                )
            else:
                rel_val = p.validation_size

            rest, val = train_test_split(
                rest,
                test_size=rel_val,
                random_state=p.random_state,
                stratify=strat_rest,
            )

        train = rest

        return list(train), list(val), list(test)


    def _grouped_base_indices(
        self,
        df,
        labels: Sequence[str],
        *,
        need_val: bool,
        need_test: bool,
    ):
        import numpy as np
        from sklearn.model_selection import GroupShuffleSplit

        p = self.protocol

        if not p.group_column or p.group_column not in df.columns:
            raise ValueError(
                "Grouped split requires a valid protocol_group_column."
            )

        groups = df[p.group_column].astype(str).to_numpy()
        idx = np.arange(len(df))

        rest = idx
        test = np.array([], dtype=int)

        if need_test:
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=p.test_size,
                random_state=p.random_state,
            )
            rest_local, test_local = next(
                gss.split(idx, labels, groups)
            )
            rest = idx[rest_local]
            test = idx[test_local]

        val = np.array([], dtype=int)

        if need_val:
            if need_test:
                rel_val = p.validation_size / max(
                    1e-9,
                    1.0 - p.test_size,
                )
            else:
                rel_val = p.validation_size

            gss2 = GroupShuffleSplit(
                n_splits=1,
                test_size=rel_val,
                random_state=p.random_state,
            )
            tr_local, va_local = next(
                gss2.split(
                    rest,
                    [labels[i] for i in rest],
                    groups[rest],
                )
            )
            train = rest[tr_local]
            val = rest[va_local]
        else:
            train = rest

        return list(train), list(val), list(test)


    def _temporal_base_indices(
        self,
        df,
        *,
        need_val: bool,
        need_test: bool,
    ):
        import numpy as np

        p = self.protocol

        if not p.group_column or p.group_column not in df.columns:
            raise ValueError(
                "Temporal split requires protocol_group_column as the time column."
            )

        order = np.argsort(
            df[p.group_column].to_numpy(),
            kind="stable",
        )

        n = len(order)
        n_test = int(round(n * p.test_size)) if need_test else 0
        n_val = int(round(n * p.validation_size)) if need_val else 0

        if n_val <= 0 and need_val:
            raise ValueError("Temporal validation split is empty.")

        if n_test <= 0 and need_test:
            raise ValueError("Temporal test split is empty.")

        test = (
            order[n - n_test:]
            if n_test
            else np.array([], dtype=int)
        )

        val_end = n - n_test
        val_start = val_end - n_val

        val = (
            order[val_start:val_end]
            if n_val
            else np.array([], dtype=int)
        )

        train = order[:val_start]

        return list(train), list(val), list(test)


    def _predefined_base_indices(
        self,
        df,
        *,
        need_val: bool,
        need_test: bool,
    ):
        p = self.protocol

        if not p.split_column or p.split_column not in df.columns:
            raise ValueError(
                "Predefined split requires a valid protocol_split_column."
            )

        col = df[p.split_column].astype(str).str.lower()

        train = df.index[col.isin(["train", "training"])].tolist()

        val = (
            df.index[col.isin(["val", "valid", "validation"])].tolist()
            if need_val
            else []
        )

        test = (
            df.index[col.isin(["test", "testing", "holdout"])].tolist()
            if need_test
            else []
        )

        if not train:
            raise ValueError(
                "Predefined split column has no train/training rows."
            )

        if need_val and not val:
            raise ValueError(
                "Predefined split column has no val/valid/validation rows."
            )

        if need_test and not test:
            raise ValueError(
                "Predefined split column has no test/testing/holdout rows."
            )

        return train, val, test


    def _external_dataset_partitions(
        self,
        columns: Sequence[str],
    ) -> Partitions:
        b = self.binding
        p = self.protocol

        if not p.validation_dataset_id:
            raise ValueError(
                "External split strategy requires a validation dataset."
            )

        train_dataset_id = self.run.dataset_id
        val_dataset_id = p.validation_dataset_id
        test_dataset_id = p.test_dataset_id

        train_df = self._load_partition_frame(
            train_dataset_id,
            columns=columns,
            role="training",
        )

        val_df = self._load_partition_frame(
            val_dataset_id,
            columns=columns,
            role="validation",
        )

        test_df = None
        if test_dataset_id:
            test_df = self._load_partition_frame(
                test_dataset_id,
                columns=columns,
                role="test",
            )

        train_labels = (
            train_df[b.target_column].astype(str).tolist()
            if b.target_column
            else ["" for _ in range(len(train_df))]
        )

        classes = sorted(set(train_labels))

        if len(classes) < 2:
            raise ValueError(
                f"Training dataset has <2 classes after filtering: {classes}"
            )

        train_p = self._partition_from_frame(
            name="train",
            frame=train_df,
            dataset_id=train_dataset_id,
            classes=classes,
        )

        val_p = self._partition_from_frame(
            name="val",
            frame=val_df,
            dataset_id=val_dataset_id,
            classes=classes,
        )

        test_p = None
        if test_df is not None:
            test_p = self._partition_from_frame(
                name="test",
                frame=test_df,
                dataset_id=test_dataset_id,
                classes=classes,
            )

        self._assert_disjoint(train_p, val_p, test_p)
        self._assert_label_subset(classes, val_p, test_p)

        if len(val_p) == 0:
            raise ValueError("External validation dataset is empty.")

        self._frame = train_df
        self._partition_frames = {
            "train": train_df,
            "val": val_df,
        }

        if test_df is not None:
            self._partition_frames["test"] = test_df

        return Partitions(
            train=train_p,
            val=val_p,
            test=test_p,
            strategy=p.split_strategy,
            group_column=None,
            random_state=p.random_state,
            protocol_id=p.protocol_id,
            target_column=b.target_column or "",
            record_id_column=b.record_id_column,
            train_dataset_id=train_dataset_id,
            validation_dataset_id=val_dataset_id,
            test_dataset_id=test_dataset_id,
        )


# ---- partitioning (modality-agnostic) ----------------------------------
    def _partition(self) -> Partitions:
        b = self.binding
        p = self.protocol
        regression = self._task_kind() == "regression"

        cols = [b.record_id_column]

        if b.target_column:
            cols.append(b.target_column)

        cols += [c for c in b.input_columns if c]

        if p.group_column:
            cols.append(p.group_column)

        if p.split_column:
            cols.append(p.split_column)

        cols = list(dict.fromkeys([c for c in cols if c]))

        base_df = self._load_partition_frame(
            self.run.dataset_id,
            columns=cols,
            role="selected training",
        )

        base_labels = (
            base_df[b.target_column].astype(str).tolist()
            if b.target_column
            else ["" for _ in range(len(base_df))]
        )

        val_from_split = p.validation_source == "split"
        test_from_split = p.test_source == "split"

        train_idx, val_idx, test_idx = self._base_split_indices(
            base_df,
            base_labels,
            need_val=val_from_split,
            need_test=test_from_split,
        )

        train_df = base_df.iloc[train_idx].reset_index(drop=True)

        train_labels = (
            train_df[b.target_column].astype(str).tolist()
            if b.target_column
            else ["" for _ in range(len(train_df))]
        )

        if regression:
            classes: List[str] = []
        else:
            classes = sorted(set(train_labels))
            if len(classes) < 2:
                raise ValueError(
                    f"Training partition has <2 classes after protocol split: {classes}"
                )

        train_p = self._partition_from_frame(
            name="train",
            frame=train_df,
            dataset_id=self.run.dataset_id,
            classes=classes,
            source="selected",
        )

        if p.validation_source == "split":
            val_df = base_df.iloc[val_idx].reset_index(drop=True)
            val_p = self._partition_from_frame(
                name="val",
                frame=val_df,
                dataset_id=self.run.dataset_id,
                classes=classes,
                source="split",
            )
        else:
            val_df = self._load_partition_frame(
                p.validation_dataset_id,
                columns=cols,
                role="validation",
            )
            val_p = self._partition_from_frame(
                name="val",
                frame=val_df,
                dataset_id=p.validation_dataset_id,
                classes=classes,
                source="dataset",
            )

        test_p = None
        test_df = None

        if p.test_source == "split":
            test_df = base_df.iloc[test_idx].reset_index(drop=True)
            test_p = self._partition_from_frame(
                name="test",
                frame=test_df,
                dataset_id=self.run.dataset_id,
                classes=classes,
                source="split",
            )
        elif p.test_source == "dataset":
            test_df = self._load_partition_frame(
                p.test_dataset_id,
                columns=cols,
                role="test",
            )
            test_p = self._partition_from_frame(
                name="test",
                frame=test_df,
                dataset_id=p.test_dataset_id,
                classes=classes,
                source="dataset",
            )

        self._assert_disjoint(train_p, val_p, test_p)
        if not regression:
            self._assert_label_subset(classes, val_p, test_p)

        if len(val_p) == 0:
            raise ValueError("Validation partition is empty.")

        self._frame = train_df
        self._partition_frames = {
            "train": train_df,
            "val": val_df,
        }

        if test_df is not None:
            self._partition_frames["test"] = test_df

        return Partitions(
            train=train_p,
            val=val_p,
            test=test_p,
            strategy=p.split_strategy,
            validation_source=p.validation_source,
            test_source=p.test_source,
            group_column=p.group_column,
            random_state=p.random_state,
            protocol_id=p.protocol_id,
            target_column=b.target_column or "",
            record_id_column=b.record_id_column,
            train_dataset_id=self.run.dataset_id,
            validation_dataset_id=val_p.dataset_id,
            test_dataset_id=test_p.dataset_id if test_p is not None else None,
        )

    def _random_indices(self, idx, labels, p):
        from sklearn.model_selection import train_test_split
        import numpy as np
        labels = np.asarray(labels)
        strat = labels if _stratifiable(labels) else None
        rest, test = (idx, np.array([], int))
        if p.test_size > 0:
            rest, test = train_test_split(
                idx, test_size=p.test_size, random_state=p.random_state, stratify=strat)
        strat_rest = labels[rest] if (strat is not None) else None
        if strat_rest is not None and not _stratifiable(strat_rest):
            strat_rest = None
        rel_val = p.validation_size / max(1e-9, 1.0 - p.test_size)
        train, val = train_test_split(
            rest, test_size=rel_val, random_state=p.random_state, stratify=strat_rest)
        return list(train), list(val), list(test)

    def _grouped_indices(self, df, labels, p):
        # Same group never spans partitions — the astronomy cutout/object case.
        from sklearn.model_selection import GroupShuffleSplit
        import numpy as np
        if not p.group_column or p.group_column not in df.columns:
            raise ValueError("by_group split requires a valid protocol_group_column.")
        groups = df[p.group_column].astype(str).to_numpy()
        idx = np.arange(len(df))
        rest, test = idx, np.array([], int)
        if p.test_size > 0:
            gss = GroupShuffleSplit(n_splits=1, test_size=p.test_size,
                                    random_state=p.random_state)
            rest, test = next(gss.split(idx, labels, groups))
        rel_val = p.validation_size / max(1e-9, 1.0 - p.test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val,
                                 random_state=p.random_state)
        tr_local, va_local = next(gss2.split(rest, [labels[i] for i in rest],
                                             groups[rest]))
        return list(rest[tr_local]), list(rest[va_local]), list(test)

    def _temporal_indices(self, df, p):
        import numpy as np
        if not p.group_column or p.group_column not in df.columns:
            raise ValueError("temporal split requires protocol_group_column (a time column).")
        order = np.argsort(df[p.group_column].to_numpy(), kind="stable")
        n = len(order)
        n_test = int(round(n * p.test_size))
        n_val = int(round(n * p.validation_size))
        test = order[n - n_test:] if n_test else np.array([], int)
        val = order[n - n_test - n_val:n - n_test] if n_val else np.array([], int)
        train = order[:n - n_test - n_val]
        return list(train), list(val), list(test)

    def _predefined_indices(self, df, p):
        if not p.split_column or p.split_column not in df.columns:
            raise ValueError("predefined split requires a valid protocol_split_column.")
        col = df[p.split_column].astype(str).str.lower()
        tr = df.index[col.isin(["train", "training"])].tolist()
        va = df.index[col.isin(["val", "valid", "validation"])].tolist()
        te = df.index[col.isin(["test", "testing", "holdout"])].tolist()
        return tr, va, te

    def _assert_disjoint(self, train_p, val_p, test_p):
        s_tr, s_va = set(train_p.record_ids), set(val_p.record_ids)
        bad = s_tr & s_va
        if bad:
            raise RuntimeError(f"train/val share {len(bad)} record-ids (leakage).")
        if test_p is not None:
            s_te = set(test_p.record_ids)
            if s_tr & s_te or s_va & s_te:
                raise RuntimeError("test partition overlaps train/val (leakage).")

    def _assert_label_subset(self, classes, val_p, test_p):
        cset = set(classes)
        for part in (val_p, test_p):
            if part is None:
                continue
            extra = sorted(set(part.labels) - cset)
            if extra:
                raise ValueError(
                    f"{part.name} contains classes unseen in training: {extra}")

    def _is_better(self, score: float) -> bool:
        if self._best_score is None:
            return True
        return (score > self._best_score
                if self.protocol.resolved_mode() == "max"
                else score < self._best_score)

    # ---- audit artifacts (protocol-owned) ----------------------------------
    def _write_split_spec(self, parts: Partitions) -> Optional[str]:
        return self.run.put_artifact(
            "ml.split_spec",
            {
                "schema_version": 2,
                "run_id": self.run.run_id,
                "dataset_id": self.run.dataset_id,
                "train_dataset_id": parts.train_dataset_id,
                "validation_dataset_id": parts.validation_dataset_id,
                "test_dataset_id": parts.test_dataset_id,
                "protocol_id": parts.protocol_id,
                "split_strategy": parts.strategy,
                "validation_source": parts.validation_source,
                "test_source": parts.test_source,
                "group_column": parts.group_column,
                "random_state": parts.random_state,
                "record_id_column": parts.record_id_column,
                "target_column": parts.target_column,
                "train_row_ids": parts.train.record_ids,
                "validation_row_ids": parts.val.record_ids,
                "test_row_ids": parts.test.record_ids if parts.test else [],
                "classes": parts.train.classes,
                "isolation": {
                    "partitions_disjoint": True,
                    "test_used_in_training": False,
                    "test_used_in_selection": False,
                    "selection_partition": "validation",
                },
            },
        )

    def _write_evaluation_report(
        self,
        *,
        parts,
        split_spec_id,
        test_metrics,
        model_artifact_id,
    ) -> Optional[str]:
        return self.run.put_artifact(
            "ml.evaluation_report",
            {
                "schema_version": 2,
                "run_id": self.run.run_id,
                "dataset_id": self.run.dataset_id,
                "train_dataset_id": parts.train_dataset_id,
                "validation_dataset_id": parts.validation_dataset_id,
                "test_dataset_id": parts.test_dataset_id,
                "validation_source": parts.validation_source,
                "test_source": parts.test_source,
                "model_artifact_id": model_artifact_id,
                "split_spec_artifact_id": split_spec_id,
                "protocol_id": parts.protocol_id,
                "selection_metric": self.protocol.selection_metric,
                "selected_on_partition": "validation",
                "best_epoch": self._best_epoch,
                "best_score": self._best_score,
                "history": self._history,
                "metrics": test_metrics,
                "reported_metrics": test_metrics,
                "reported_partition": "test" if parts.test else None,
                "validity": {
                    "metric_partition": "test" if parts.test else "none",
                    "selection_partition": "validation",
                    "selection_touched_reported_partition": False,
                },
            },
        )

    def _write_predictions(self, records, parts, model_artifact_id) -> Optional[str]:
        return self.run.put_artifact("ml.predictions", {
            "schema_version": 2,
            "run_id": self.run.run_id,
            "dataset_id": self.run.dataset_id,
            "model_artifact_id": model_artifact_id,
            "protocol_id": parts.protocol_id,
            "prediction_scope": "test",
            "record_id_column": parts.record_id_column,
            "class_names": parts.train.classes,
            "records": records,
        }, row_ids=[r["record_id"] for r in records])

# ---- modality/framework specifics (subclasses implement) ---------------
    def _make_loader(self, partition: Partition, *, train: bool): raise NotImplementedError
    def _evaluate(self, model, loader, *, return_records: bool = False): raise NotImplementedError
    def _snapshot(self, model): raise NotImplementedError
    def _restore(self, model, state): raise NotImplementedError
    def _assert_output_dim(self, model, target: TargetSpec, train_loader): raise NotImplementedError
    def _write_model_artifact(self, model, parts, target: TargetSpec, *, split_spec_artifact_id=None) -> Optional[str]: raise NotImplementedError

def _stratifiable(labels) -> bool:
    import numpy as np
    vals, counts = np.unique(np.asarray(labels), return_counts=True)
    return len(vals) > 1 and counts.min() >= 2


# -----------------------------------------------------------------------------
# TorchClassificationHarness — the concrete torch+classification core
# -----------------------------------------------------------------------------

class TorchClassificationHarness(RunHarness):

    @property
    def device(self):
        return self._device()

    def _device(self):
        import torch
        req = str(self.run.params.get("device", "auto")).lower()
        if req == "cpu":
            return torch.device("cpu")
        if req == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _frame_for(self, partition: Partition):
        b = self.binding

        frame = (
            self._partition_frames.get(partition.name)
            if hasattr(self, "_partition_frames")
            else None
        )

        if frame is None:
            frame = self._frame

        wanted = set(str(record_id) for record_id in partition.record_ids)

        return frame[
            frame[b.record_id_column].astype(str).isin(wanted)
        ]

    def _make_loader(self, partition: Partition, *, train: bool):
        from torch.utils.data import DataLoader, Dataset

        recipe, run, b = self.recipe, self.run, self.binding
        frame = self._frame_for(partition).reset_index(drop=True)
        transform = (recipe.train_transform(run) if train else recipe.eval_transform(run))
        class_to_idx = {c: i for i, c in enumerate(partition.classes)}
        self._eval_classes = list(partition.classes)
        harness = self

        class _DS(Dataset):
            def __len__(self): return len(frame)
            def __getitem__(self, i):
                row = frame.iloc[i]
                x = recipe.load_sample(run, row)          # recipe: row -> raw input
                if transform is not None:
                    x = transform(x)
                y = class_to_idx[str(row[b.target_column])] if b.target_column else -1
                return x, int(y), str(row[b.record_id_column])

        return DataLoader(
            _DS(),
            batch_size=int(run.params.get("batch_size", 128)),
            shuffle=bool(train),
            num_workers=int(run.params.get("num_workers", 0)),
            pin_memory=str(self._device()).startswith("cuda"),
        )

    def _assert_output_dim(self, model, target, train_loader):
        # Catches the kuangliu ResNet18()-ignores-num_classes trap loudly,
        # before training a wrong-width head.
        import torch
        expected = int(target.num_outputs)
        model.eval()
        device = self._device()
        model.to(device)
        x, _y, _ids = next(iter(train_loader))
        with torch.no_grad():
            logits = self.recipe.eval_forward(model, x.to(device))
        out = int(logits.shape[1])
        if out != expected:
            raise ValueError(
                f"Model produces {out} outputs but the run expects {expected} "
                f"({target.kind}). The architecture is ignoring the output width "
                f"(common with hard-coded CIFAR-10 model factories). Pass the "
                f"output width through, or wrap the final layer.")

    def _evaluate(self, model, loader, *, return_records: bool = False):
        import numpy as np
        import torch
        from sklearn.metrics import accuracy_score, f1_score

        device = self._device()
        model.to(device)
        model.eval()
        crit = torch.nn.CrossEntropyLoss()
        loss_sum = seen = 0
        y_true, y_pred = [], []
        records = []
        classes = None
        with torch.no_grad():
            for x, y, ids in loader:
                x = x.to(device)
                y = y.to(device).long()
                logits = self.recipe.eval_forward(model, x)
                loss_sum += float(crit(logits, y)) * int(y.size(0))
                seen += int(y.size(0))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(preds.tolist())
                if return_records:
                    if classes is None:
                        classes = self._eval_classes
                    for rid, p_idx, prob in zip(ids, preds, probs):
                        prob = [float(v) for v in prob]
                        top = float(max(prob)) if prob else None
                        records.append({
                            "record_id": str(rid),
                            "prediction": classes[int(p_idx)],
                            "confidence": top,
                            "uncertainty": None if top is None else 1.0 - top,
                            "probabilities": prob,
                        })
        metrics = {
            "loss": (loss_sum / seen) if seen else None,
            "accuracy": float(accuracy_score(y_true, y_pred)) if seen else None,
            "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                       zero_division=0)) if seen else None,
        }
        return metrics, records

    def _snapshot(self, model):
        m = model.module if hasattr(model, "module") else model
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    def _restore(self, model, state):
        m = model.module if hasattr(model, "module") else model
        m.load_state_dict(state)


    def _write_model_artifact(
        self,
        model,
        parts: Partitions,
        target: TargetSpec,
        *,
        split_spec_artifact_id: Optional[str] = None,
    ) -> Optional[str]:
        import torch

        num_classes = int(target.num_outputs)

        model_dir = ml_run_artifact_dir(
            self.run,
            kind="model",
        )

        checkpoint_path = model_dir / "model.pt"
        tmp_checkpoint_path = model_dir / "model.pt.tmp"
        manifest_path = model_dir / "model_manifest.json"

        m = model.module if hasattr(model, "module") else model

        architecture = str(self.run.params.get("architecture") or "resnet18")
        custom_model_import = str(self.run.params.get("custom_model_import") or "").strip()

        if architecture in {"resnet18", "resnet34"}:
            architecture_ref = f"torchvision.{architecture}"
        else:
            architecture_ref = architecture

        checkpoint_payload = {
            "schema_version": 2,
            "state_dict": m.state_dict(),
            "class_names": list(parts.train.classes),
            "num_classes": int(num_classes),
            "params": dict(self.run.params),
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "run_id": self.run.run_id,
            "dataset_id": self.run.dataset_id,
            "protocol_id": parts.protocol_id,
            "framework": "torch",
            "task": self.recipe.task,
            "modality": self.recipe.modality,
            "train_dataset_id": parts.train_dataset_id,
            "validation_dataset_id": parts.validation_dataset_id,
            "test_dataset_id": parts.test_dataset_id,
            "validation_source": parts.validation_source,
            "test_source": parts.test_source,
            "input_contract": {
                "record_id_column": parts.record_id_column,
                "target_column": parts.target_column,
                "image_column": self.binding.image_column,
                "input_columns": list(self.binding.input_columns or []),
            },
            "architecture": architecture_ref,
            "custom_model_import": custom_model_import,
            "epoch": self._best_epoch,
            "metrics": {
                "best_score": self._best_score,
                "selection_metric": self.protocol.selection_metric,
            },
            "transform": {
                "image_size": 32,
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2023, 0.1994, 0.2010],
            },
        }

        # Atomic-ish write: write temp, then replace final checkpoint.
        torch.save(checkpoint_payload, tmp_checkpoint_path)
        os.replace(tmp_checkpoint_path, checkpoint_path)

        manifest_payload = {
            "schema_version": 2,
            "kind": "torch_classifier",
            "framework": "torch",
            "task": self.recipe.task,
            "modality": self.recipe.modality,
            "run_id": self.run.run_id,
            "dataset_id": self.run.dataset_id,
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "protocol_id": parts.protocol_id,
            "split_spec_artifact_id": split_spec_artifact_id,
            "created_at": time.time(),
            "class_names": list(parts.train.classes),
            "num_classes": int(num_classes),
            "train_dataset_id": parts.train_dataset_id,
            "validation_dataset_id": parts.validation_dataset_id,
            "test_dataset_id": parts.test_dataset_id,
            "validation_source": parts.validation_source,
            "test_source": parts.test_source,
            "files": {
                "checkpoint": str(checkpoint_path),
                "manifest": str(manifest_path),
            },
            "model_ref": {
                "storage": "local_file",
                "uri": str(checkpoint_path),
                "path": str(checkpoint_path),
                "format": "torch_checkpoint",
                "framework": "torch",
                "metadata": {
                    "class_names": list(parts.train.classes),
                    "num_classes": int(num_classes),
                    "recipe_id": self.run.recipe_id,
                    "recipe_version": self.run.recipe_version,
                    "run_id": self.run.run_id,
                    "architecture": architecture_ref,
                    "custom_model_import": custom_model_import,
                    "image_size": 32,
                    "normalization": {
                        "mean": [0.4914, 0.4822, 0.4465],
                        "std": [0.2023, 0.1994, 0.2010],
                    },
                }
            },
            "input_contract": {
                "record_id_column": parts.record_id_column,
                "target_column": parts.target_column,
                "image_column": self.binding.image_column,
                "input_columns": list(self.binding.input_columns or []),
            },
            "protocol": {
                "protocol_id": parts.protocol_id,
                "split_strategy": parts.strategy,
                "group_column": parts.group_column,
                "random_state": parts.random_state,
                "selection_metric": self.protocol.selection_metric,
                "selection_mode": self.protocol.resolved_mode(),
                "validation_dataset_id": parts.validation_dataset_id,
                "test_dataset_id": parts.test_dataset_id,
                "validation_source": parts.validation_source,
                "test_source": parts.test_source,
            },
            "model_title": getattr(self.recipe, "title", self.run.recipe_id),
            "architecture": architecture_ref,
            "custom_model_import": custom_model_import,
            "metrics": {
                "best_score": self._best_score,
                "selection_metric": self.protocol.selection_metric,
                "best_epoch": self._best_epoch,
            },
        }

        manifest_path.write_text(
            json.dumps(
                json_safe(manifest_payload),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        self._eval_classes = list(parts.train.classes)

        return self.run.put_artifact(
            "ml.model",
            manifest_payload,
            params=self.run.params,
        )


# -----------------------------------------------------------------------------
# Harness factory — keeps the runner modality-agnostic
# -----------------------------------------------------------------------------

_HARNESSES = []   # list of (predicate, harness_cls); first match wins

def register_harness(predicate, harness_cls):
    """Extension point: other plugins can register harnesses for new
    (framework, modality, task) combinations without touching this file."""
    _HARNESSES.insert(0, (predicate, harness_cls))

def make_harness(
    run,
    recipe: "MLRecipe",
) -> RunHarness:
    framework = (
        run.params.get("framework")
        or getattr(recipe, "framework", "")
        or ""
    )
    framework = str(framework).lower()

    task = str(
        getattr(recipe, "task", "")
        or run.params.get("task", "")
        or ""
    ).lower()

    modality = str(
        getattr(recipe, "modality", "")
        or run.params.get("modality", "")
        or ""
    ).lower()

    for predicate, harness_cls in list(_HARNESSES):
        try:
            if predicate(
                framework=framework,
                task=task,
                modality=modality,
                run=run,
                recipe=recipe,
            ):
                return harness_cls(run, recipe)
        except TypeError:
            try:
                if predicate(framework, task, modality):
                    return harness_cls(run, recipe)
            except Exception:
                pass
        except Exception:
            pass

    if framework == "torch" and task == "classification":
        return TorchClassificationHarness(run, recipe)

    raise ValueError(
        "No managed ML harness is available for "
        f"framework={framework!r}, task={task!r}, modality={modality!r}."
    )

# =============================================================================
# TorchRegressionHarness — the concrete torch+regression core.
#
# Append to recipe_registry.py AFTER make_harness / register_harness (it calls
# register_harness at import time). Parallels TorchClassificationHarness but for
# continuous targets. It owns none of the scientific protocol — RunHarness does.
# TargetSpec(kind="regression") already threads through the base partition /
# build_model / output-dim path, so this harness only implements the leaf
# methods: loader, evaluate, snapshot/restore, output-dim check, model artifact.
#
# Conventions:
#   * Output width comes from TargetSpec.num_outputs == params["n_outputs"]
#     (default 1). The model's final layer must emit that many units; the
#     base's _assert_output_dim call enforces it before training.
#   * One target column. Scalar regression is the common case. For multi-output
#     regression, store a vector per row in the target column (Python list,
#     numpy array, or comma/semicolon-separated string) and set n_outputs.
#   * The harness reports loss/mse/rmse/mae/r2 on the VALIDATION partition each
#     epoch and selects the best epoch with the protocol's selection metric
#     (defaulting to val_loss, minimised).
#
# Deliberate non-feature: the target is NOT scaled here. The predict side
# reconstructs raw model outputs with no inverse transform, so standardising the
# target inside the harness would make training and inference disagree. Pre-scale
# the target in the dataset, or scale inside the recipe AND persist+apply the
# scaler at inference, if you need it.
# =============================================================================


class TorchRegressionHarness(RunHarness):

    # val-partition metrics this harness emits. A stray classification default
    # (e.g. val_accuracy) is coerced to val_loss so epoch selection can't
    # silently no-op on a metric the rows never contain.
    _REGRESSION_VAL_METRICS = (
        "val_loss", "val_mse", "val_rmse", "val_mae", "val_r2",
    )
    _CLASSIFICATION_TOKENS = (
        "accuracy", "f1", "auc", "precision", "recall", "balanced",
    )

    def __init__(self, run, recipe: "MLRecipe"):
        super().__init__(run, recipe)
        if self.binding is None or not self.binding.target_column:
            raise ValueError(
                "Regression runs require a target column. Set `target_column`, "
                "map `target_label`, or add a numeric target column to the "
                "dataset. (The runner only auto-requires a target for "
                "classification, so regression must assert it here.)"
            )
        self._coerce_selection_metric()

    # ---- selection-metric sanity ------------------------------------------
    def _coerce_selection_metric(self) -> None:
        metric = str(self.protocol.selection_metric or "").strip()
        lowered = metric.lower()

        needs_default = (
            not metric
            or any(tok in lowered for tok in self._CLASSIFICATION_TOKENS)
        )
        if not needs_default:
            return

        self.protocol.selection_metric = "val_loss"
        self.protocol.selection_mode = "min"            # val_loss must minimise
        # Provenance must stay truthful: protocol_id is a hash of the config.
        self.protocol.protocol_id = _stable_protocol_id(self.protocol)

        self.run.log(
            message=(
                f"Regression run: selection metric {metric or '(unset)'!r} is not "
                "a regression metric; selecting best epoch on 'val_loss' (min)."
            ),
            status="running",
            extra={"phase": "protocol", "selection_metric": "val_loss"},
        )

    # ---- task identity (drives base _target_spec / regression branches) ----
    def _task_kind(self) -> str:
        return "regression"

    def _n_outputs(self) -> int:
        try:
            return max(1, int(self.run.params.get("n_outputs", 1) or 1))
        except Exception:
            return 1

    # ---- torch plumbing (parallels TorchClassificationHarness) -------------
    @property
    def device(self):
        return self._device()

    def _device(self):
        import torch
        req = str(self.run.params.get("device", "auto")).lower()
        if req == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _frame_for(self, partition: Partition):
        b = self.binding
        frame = (
            self._partition_frames.get(partition.name)
            if hasattr(self, "_partition_frames")
            else None
        )
        if frame is None:
            frame = self._frame
        wanted = set(str(record_id) for record_id in partition.record_ids)
        return frame[frame[b.record_id_column].astype(str).isin(wanted)]

    def _snapshot(self, model):
        m = model.module if hasattr(model, "module") else model
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    def _restore(self, model, state):
        m = model.module if hasattr(model, "module") else model
        m.load_state_dict(state)

    # ---- target reading ----------------------------------------------------
    def _read_target(self, value, n_outputs: int):
        import numpy as np
        if isinstance(value, (list, tuple, np.ndarray)):
            vec = [float(v) for v in np.asarray(value).reshape(-1)]
        elif isinstance(value, str) and any(s in value for s in (",", ";")):
            vec = [float(p) for p in re.split(r"[,;]", value) if p.strip() != ""]
        else:
            vec = [float(value)]
        if len(vec) != n_outputs:
            raise ValueError(
                f"Target has {len(vec)} value(s) but the run expects "
                f"n_outputs={n_outputs}. For multi-output regression, store a "
                f"vector per row in {self.binding.target_column!r} (list or "
                f"comma-separated) and set params['n_outputs']."
            )
        return vec

    # ---- modality/framework leaf methods -----------------------------------
    def _make_loader(self, partition: Partition, *, train: bool):
        import torch
        from torch.utils.data import DataLoader, Dataset

        recipe, run, b = self.recipe, self.run, self.binding
        frame = self._frame_for(partition).reset_index(drop=True)
        transform = recipe.train_transform(run) if train else recipe.eval_transform(run)
        n_outputs = self._n_outputs()
        read_target = self._read_target
        target_col = b.target_column

        class _DS(Dataset):
            def __len__(self):
                return len(frame)

            def __getitem__(self, i):
                row = frame.iloc[i]
                x = recipe.load_sample(run, row)          # recipe: row -> raw input
                if transform is not None:
                    x = transform(x)
                vec = read_target(row[target_col], n_outputs) if target_col else [0.0] * n_outputs
                y = torch.tensor(vec, dtype=torch.float32)  # shape [n_outputs]
                return x, y, str(row[b.record_id_column])

        return DataLoader(
            _DS(),
            batch_size=int(run.params.get("batch_size", 128)),
            shuffle=bool(train),
            num_workers=int(run.params.get("num_workers", 0)),
            pin_memory=str(self._device()).startswith("cuda"),
        )

    def _assert_output_dim(self, model, target: TargetSpec, train_loader):
        # Regression analogue of the kuangliu wrong-width-head trap: catch a head
        # that emits the wrong number of continuous outputs before training.
        import torch
        expected = int(target.num_outputs)
        device = self._device()
        model.to(device)
        model.eval()
        x, _y, _ids = next(iter(train_loader))
        if hasattr(x, "to"):
            x = x.to(device)
        with torch.no_grad():
            out = self.recipe.eval_forward(model, x)
        out_dim = 1 if out.dim() == 1 else int(out.shape[1])
        if out_dim != expected:
            raise ValueError(
                f"Model produces {out_dim} output(s) but the run expects "
                f"{expected} (regression, n_outputs={expected}). Give the model a "
                f"final layer with {expected} unit(s), or pass the output width "
                f"through your build_model."
            )

    def _evaluate(self, model, loader, *, return_records: bool = False):
        import numpy as np
        import torch

        device = self._device()
        model.to(device)
        model.eval()

        preds_all, true_all, ids_all = [], [], []
        with torch.no_grad():
            for x, y, ids in loader:
                if hasattr(x, "to"):
                    x = x.to(device)
                out = self.recipe.eval_forward(model, x)
                out = out.detach().cpu().float().numpy()
                if out.ndim == 1:
                    out = out.reshape(-1, 1)
                yb = (
                    y.detach().cpu().float().numpy()
                    if hasattr(y, "detach")
                    else np.asarray(y, dtype=float)
                )
                if yb.ndim == 1:
                    yb = yb.reshape(-1, 1)
                preds_all.append(out)
                true_all.append(yb)
                ids_all.extend(str(i) for i in ids)

        if not preds_all:
            return (
                {"loss": None, "mse": None, "rmse": None, "mae": None, "r2": None},
                [],
            )

        preds = np.concatenate(preds_all, axis=0)
        true = np.concatenate(true_all, axis=0)
        metrics = self._regression_metrics(true, preds)

        records: List[Dict[str, Any]] = []
        if return_records:
            single = preds.shape[1] == 1
            for rid, p_row, t_row in zip(ids_all, preds, true):
                pred_val = float(p_row[0]) if single else [float(v) for v in p_row]
                true_val = float(t_row[0]) if single else [float(v) for v in t_row]
                rec = {"record_id": rid, "prediction": pred_val, "y_true": true_val}
                if single:
                    rec["abs_error"] = abs(pred_val - true_val)
                records.append(rec)

        return metrics, records

    def _regression_metrics(self, y_true, y_pred) -> Dict[str, Any]:
        import numpy as np
        yt = np.asarray(y_true, dtype=float).reshape(len(y_true), -1)
        yp = np.asarray(y_pred, dtype=float).reshape(len(y_pred), -1)
        diff = yp - yt
        mse = float(np.mean(diff ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(diff)))
        try:
            from sklearn.metrics import r2_score
            r2 = float(r2_score(yt, yp, multioutput="uniform_average"))
        except Exception:
            ss_res = float(np.sum(diff ** 2))
            ss_tot = float(np.sum((yt - yt.mean(axis=0)) ** 2))
            r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
        # 'loss' == mse so selecting on val_loss (min) is well defined and
        # comparable regardless of the recipe's training criterion.
        return {"loss": mse, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def _write_model_artifact(
        self,
        model,
        parts: Partitions,
        target: TargetSpec,
        *,
        split_spec_artifact_id: Optional[str] = None,
    ) -> Optional[str]:
        import torch

        n_outputs = int(target.num_outputs)

        model_dir = ml_run_artifact_dir(self.run, kind="model")
        checkpoint_path = model_dir / "model.pt"
        tmp_checkpoint_path = model_dir / "model.pt.tmp"
        manifest_path = model_dir / "model_manifest.json"

        m = model.module if hasattr(model, "module") else model

        architecture = str(self.run.params.get("architecture") or "custom")
        custom_model_import = str(self.run.params.get("custom_model_import") or "").strip()

        # Only record transform hints the recipe actually supplied; do not invent
        # CIFAR defaults — a regression recipe may be tabular, not image.
        transform_meta: Dict[str, Any] = {}
        image_size = self.run.params.get("image_size")
        normalization = self.run.params.get("normalization")
        if image_size is not None:
            transform_meta["image_size"] = image_size
        if isinstance(normalization, Mapping):
            transform_meta["normalization"] = dict(normalization)

        target_columns = [parts.target_column] if parts.target_column else []

        checkpoint_payload = {
            "schema_version": 2,
            "state_dict": m.state_dict(),
            "class_names": [],
            "num_classes": int(n_outputs),      # output width; kept for loader parity
            "num_outputs": int(n_outputs),
            "params": dict(self.run.params),
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "run_id": self.run.run_id,
            "dataset_id": self.run.dataset_id,
            "protocol_id": parts.protocol_id,
            "framework": "torch",
            "task": self.recipe.task,
            "modality": self.recipe.modality,
            "train_dataset_id": parts.train_dataset_id,
            "validation_dataset_id": parts.validation_dataset_id,
            "test_dataset_id": parts.test_dataset_id,
            "validation_source": parts.validation_source,
            "test_source": parts.test_source,
            "input_contract": {
                "record_id_column": parts.record_id_column,
                "target_column": parts.target_column,
                "image_column": self.binding.image_column,
                "input_columns": list(self.binding.input_columns or []),
            },
            "architecture": architecture,
            "custom_model_import": custom_model_import,
            "epoch": self._best_epoch,
            "metrics": {
                "best_score": self._best_score,
                "selection_metric": self.protocol.selection_metric,
            },
            "transform": transform_meta,
        }

        # Atomic-ish write: temp then replace.
        torch.save(checkpoint_payload, tmp_checkpoint_path)
        os.replace(tmp_checkpoint_path, checkpoint_path)

        manifest_payload = {
            "schema_version": 2,
            "kind": "torch_regressor",
            "framework": "torch",
            "task": self.recipe.task,
            "modality": self.recipe.modality,
            "run_id": self.run.run_id,
            "dataset_id": self.run.dataset_id,
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "protocol_id": parts.protocol_id,
            "split_spec_artifact_id": split_spec_artifact_id,
            "created_at": time.time(),
            "class_names": [],
            "num_classes": int(n_outputs),
            "num_outputs": int(n_outputs),
            "target_columns": target_columns,
            "train_dataset_id": parts.train_dataset_id,
            "validation_dataset_id": parts.validation_dataset_id,
            "test_dataset_id": parts.test_dataset_id,
            "validation_source": parts.validation_source,
            "test_source": parts.test_source,
            "files": {
                "checkpoint": str(checkpoint_path),
                "manifest": str(manifest_path),
            },
            "model_ref": {
                "storage": "local_file",
                "uri": str(checkpoint_path),
                "path": str(checkpoint_path),
                "format": "torch_checkpoint",
                "framework": "torch",
                "metadata": {
                    "class_names": [],
                    "num_classes": int(n_outputs),
                    "num_outputs": int(n_outputs),
                    "task": "regression",
                    "recipe_id": self.run.recipe_id,
                    "recipe_version": self.run.recipe_version,
                    "run_id": self.run.run_id,
                    "architecture": architecture,
                    "custom_model_import": custom_model_import,
                    **transform_meta,
                },
            },
            "input_contract": {
                "record_id_column": parts.record_id_column,
                "target_column": parts.target_column,
                "image_column": self.binding.image_column,
                "input_columns": list(self.binding.input_columns or []),
            },
            "protocol": {
                "protocol_id": parts.protocol_id,
                "split_strategy": parts.strategy,
                "group_column": parts.group_column,
                "random_state": parts.random_state,
                "selection_metric": self.protocol.selection_metric,
                "selection_mode": self.protocol.resolved_mode(),
                "validation_dataset_id": parts.validation_dataset_id,
                "test_dataset_id": parts.test_dataset_id,
                "validation_source": parts.validation_source,
                "test_source": parts.test_source,
            },
            "model_title": getattr(self.recipe, "title", self.run.recipe_id),
            "architecture": architecture,
            "custom_model_import": custom_model_import,
            "metrics": {
                "best_score": self._best_score,
                "selection_metric": self.protocol.selection_metric,
                "best_epoch": self._best_epoch,
            },
        }

        manifest_path.write_text(
            json.dumps(json_safe(manifest_payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return self.run.put_artifact(
            "ml.model",
            manifest_payload,
            params=self.run.params,
        )


# -----------------------------------------------------------------------------
# Register so make_harness() resolves regression without editing its fallback.
# _HARNESSES is consulted first (first match wins); the predicate accepts both
# the kwargs call and the positional fallback make_harness uses.
# -----------------------------------------------------------------------------

def _is_torch_regression(
    framework="", task="", modality="", run=None, recipe=None, **_kwargs
) -> bool:
    return (
        str(framework).lower() == "torch"
        and str(task).lower() in {"regression", "regressor"}
    )


register_harness(_is_torch_regression, TorchRegressionHarness)