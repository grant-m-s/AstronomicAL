from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Mapping, Optional

from .paths import _safe_path_part, ml_artifact_root
from .runtime import publish
from .serialization import json_safe

class MLRunLogger:
    """
    Records recipe progress, metrics, selection information and final run state.
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
    ) -> None:
        self.context = context
        self.run_id = str(run_id)
        self.dataset_id = dataset_id
        self.training_log_artifact_id = training_log_artifact_id
        self.recipe_spec = recipe_spec
        self.params = dict(params or {})
        self.protocol = protocol
        self.binding = binding
        self.started_at = time.time()

        # Full chronological event stream.
        self.events: List[Dict[str, Any]] = []

        # Curve-friendly rows, merged by epoch.
        self.epochs_by_epoch: Dict[int, Dict[str, Any]] = {}

        # Authoritative facts supplied later by runner/harness.
        self.summary: Dict[str, Any] = {}

        self._ensure_training_log_artifact()

    def checkpoint_state(self) -> Dict[str, Any]:
        """Return the in-memory log state required to continue one run."""
        return json_safe(
            {
                "started_at": self.started_at,
                "events": list(self.events),
                "epochs": self._epoch_rows(),
                "summary": dict(self.summary),
                "training_log_artifact_id": self.training_log_artifact_id,
            }
        )

    def restore_from_checkpoint(self, state: Mapping[str, Any]) -> None:
        """Restore chronological events and curve rows before a resumed epoch."""
        if not isinstance(state, Mapping) or not state:
            return
        try:
            self.started_at = float(state.get("started_at") or self.started_at)
        except Exception:
            pass
        self.events = [dict(row) for row in state.get("events") or [] if isinstance(row, Mapping)]
        self.epochs_by_epoch = {}
        for row in state.get("epochs") or []:
            if not isinstance(row, Mapping):
                continue
            epoch = self._coerce_epoch(row)
            if epoch is not None:
                self.epochs_by_epoch[int(epoch)] = dict(row)
        self.summary.update(dict(state.get("summary") or {}))
        self.summary.update({"status": "resuming", "message": "Recipe run resumed from checkpoint."})

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
    ) -> None:
        """Attach protocol/binding once the runner has resolved them."""

        if protocol is not None:
            self.protocol = protocol

        if binding is not None:
            self.binding = binding

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
