from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..data.dataset_access import get_dataset_frame
from ..progress import ProgressIterable
from ..protocol import DataBinding, Partition, Partitions, ProtocolConfig, TargetSpec
from ..runtime import MLRecipePaused, check_cancelled, cleanup_ml_runtime, publish, put_artifact
from ..serialization import json_safe
from ..split_datasets import materialize_split_datasets

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
        self._parts: Optional[Partitions] = None
        self._target: Optional[TargetSpec] = None
        self._split_spec_artifact_id: Optional[str] = None
        self._last_completed_epoch: int = 0
        self.progress = getattr(run, "progress", None)

    def _progress_report(self, **kwargs: Any) -> None:
        if self.progress is None:
            return
        try:
            self.progress.report(**kwargs)
        except Exception:
            pass

    def _progress_activity(self, **kwargs: Any):
        if self.progress is None:
            return nullcontext()
        try:
            return self.progress.activity(**kwargs)
        except Exception:
            return nullcontext()

    def _instrument_training_loader(self, loader: Any) -> Any:
        framework = str(
            self.run.params.get("framework")
            or getattr(self.recipe, "framework", "")
            or ""
        ).lower()
        if framework == "torch" and hasattr(loader, "__iter__"):
            # Cancellation must not depend on progress UI availability. The
            # wrapper checks the main-process token between every DataLoader batch.
            return ProgressIterable(loader, reporter=self.progress, run=self.run)
        return loader

    def _training_step_label(self) -> str:
        return "epoch"

    # ---- the ONE primitive the recipe calls each epoch ---------------------
    def report_epoch(
        self,
        epoch: int,
        model,
        *,
        train_metrics: Dict[str, Any],
    ):
        """Evaluate validation data, select the best epoch, and publish progress."""

        self.run.check_cancelled()
        total_epochs = _configured_total_epochs(self.run)
        step_label = self._training_step_label()
        step_title = step_label.capitalize()
        self._progress_report(
            stage="validation",
            message=f"{step_title} {int(epoch)} training is complete. Evaluating the validation partition.",
            epoch=int(epoch),
            total_epochs=total_epochs,
            force=True,
        )
        with self._progress_activity(
            stage="validation",
            message=f"Evaluating validation data for {step_label} {int(epoch)}.",
            detail="Computing validation predictions and selection metrics without exposing the validation loader to the recipe.",
        ):
            val_metrics = self._evaluate(model, self._val_loader)[0]

        row: Dict[str, Any] = {"epoch": int(epoch)}
        row.update(
            {f"train_{key}": value for key, value in dict(train_metrics or {}).items()}
        )
        row.update(
            {f"val_{key}": value for key, value in dict(val_metrics or {}).items()}
        )
        self._history.append(row)

        score = row.get(self.protocol.selection_metric)
        improved = False
        if score is not None:
            try:
                score_float = float(score)
            except Exception:
                score_float = None
            if score_float is not None and self._is_better(score_float):
                self._best_score = score_float
                self._best_epoch = int(epoch)
                self._best_state = self._snapshot(model)
                improved = True

        selection_detail = (
            f"Selection metric `{self.protocol.selection_metric}` = `{score}`. "
            f"Best epoch is now `{self._best_epoch}` with score `{self._best_score}`."
        )
        if not improved and self._best_epoch is not None:
            selection_detail = (
                f"Selection metric `{self.protocol.selection_metric}` = `{score}`. "
                f"The existing best remains epoch `{self._best_epoch}` with score `{self._best_score}`."
            )
        self._progress_report(
            stage="selecting_checkpoint",
            message=f"Validation for {step_label} {int(epoch)} is complete. Comparing it with the best checkpoint.",
            detail=selection_detail,
            epoch=int(epoch),
            total_epochs=total_epochs,
            metrics=row,
            force=True,
        )

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

        self._progress_report(
            stage="training",
            message=(
                f"{step_title} {int(epoch)} is complete. "
                + (
                    f"Preparing {step_label} {int(epoch) + 1}."
                    if total_epochs is None or int(epoch) < total_epochs
                    else "All configured epochs are complete."
                )
            ),
            epoch=int(epoch),
            total_epochs=total_epochs,
            current=int(epoch),
            total=total_epochs,
            unit="epochs",
            force=True,
        )
        return val_metrics

    def check_pause_boundary(self, epoch: int, model: Any, components: Any) -> None:
        """Pause only after a complete epoch and scheduler update.

        Built-in recipes call this immediately after their scheduler step. The
        method persists both a full resumable checkpoint and an ml.model artifact
        that can be selected by the Predictor while training is paused.
        """
        self._last_completed_epoch = int(epoch)
        if not bool(getattr(self.run, "pause_requested", False)):
            return
        payload = self._persist_pause(model=model, components=components, completed_epoch=int(epoch))
        raise MLRecipePaused(
            f"Training paused after epoch {int(epoch)}.",
            pause_payload=payload,
        )

    def _persist_pause(self, *, model: Any, components: Any, completed_epoch: int) -> Dict[str, Any]:
        from ..resume import save_resume_checkpoint, write_paused_model_manifest

        if self._parts is None or self._target is None:
            raise RuntimeError("Cannot pause before partitions and target metadata are available.")

        framework = str(self.run.params.get("framework") or getattr(self.recipe, "framework", "")).lower()
        pause_reason = str(
            getattr(getattr(self.run, "training_control", None), "pause_reason", "")
            or "Training pause requested."
        )
        # Record the boundary before serialising logger state. The later paused
        # event includes artifact ids, but this marker ensures a resumed training
        # log still shows where and why execution stopped.
        self._progress_report(
            stage="saving_checkpoint",
            message=f"{pause_reason} Capturing complete state after epoch {completed_epoch}.",
            detail="Saving model, optimiser, scheduler, RNG, split identity, history, and best-checkpoint state.",
            epoch=completed_epoch,
            total_epochs=_configured_total_epochs(self.run),
            force=True,
        )
        if self.progress is None:
            self.run.log(
                message=f"{pause_reason} Saving epoch {completed_epoch} state.",
                status="pausing",
                step=completed_epoch,
                metrics={},
                extra={"phase": "pause_boundary", "completed_epoch": completed_epoch},
            )
        framework_state = self._capture_framework_resume_state(model, components, framework=framework)
        state = {
            "schema_version": 1,
            "status": "paused",
            "run_id": self.run.run_id,
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "dataset_id": self.run.dataset_id,
            "framework": framework,
            "task": self._task_kind(),
            "modality": str(getattr(self.recipe, "modality", "") or self.run.params.get("modality", "")),
            "completed_epoch": int(completed_epoch),
            "next_epoch": int(completed_epoch) + 1,
            "params": dict(self.run.params),
            "protocol": self._protocol_state(),
            "binding": self._binding_state(),
            "target": self._target_state(self._target),
            "partition_signatures": self._partition_signatures(self._parts),
            "split_spec_artifact_id": self._split_spec_artifact_id,
            "history": list(self._history),
            "best_epoch": self._best_epoch,
            "best_score": self._best_score,
            "best_state": self._best_state,
            "framework_state": framework_state,
            "logger_state": (
                self.run.logger.checkpoint_state()
                if getattr(self.run, "logger", None) is not None
                and callable(getattr(self.run.logger, "checkpoint_state", None))
                else {}
            ),
            "rng_state": self._capture_rng_state(framework=framework),
            "training_log_artifact_id": self.run.training_log_artifact_id,
            "created_at": time.time(),
        }
        # Torch prediction reconstruction already understands a top-level
        # state_dict checkpoint. Keep the alias in addition to framework_state.
        if framework == "torch":
            state["state_dict"] = framework_state.get("model_state_dict")
            state["model_state_dict"] = framework_state.get("model_state_dict")
            state["class_names"] = list(self._target.classes)
            state["num_classes"] = int(self._target.num_outputs)
            state["num_outputs"] = int(self._target.num_outputs)
            state["input_contract"] = self._binding_state()

        with self._progress_activity(
            stage="saving_checkpoint",
            message=f"Writing the epoch {completed_epoch} resumable checkpoint and manifest.",
        ):
            resume_ref, resume_manifest_path = save_resume_checkpoint(
                run=self.run,
                state=state,
                framework=framework,
                completed_epoch=completed_epoch,
            )
        resume_payload = {
            "artifact_type": "ml.resume_checkpoint",
            "schema_version": 1,
            "status": "paused",
            "resumable": True,
            "run_id": self.run.run_id,
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "dataset_id": self.run.dataset_id,
            "completed_epoch": int(completed_epoch),
            "next_epoch": int(completed_epoch) + 1,
            "resume_ref": resume_ref,
            "manifest_path": str(resume_manifest_path),
            "params": json_safe(self.run.params),
            "protocol": json_safe(self._protocol_state()),
            "binding": json_safe(self._binding_state()),
            "split_spec_artifact_id": self._split_spec_artifact_id,
            "training_log_artifact_id": self.run.training_log_artifact_id,
            "created_at": time.time(),
        }
        resume_artifact_id = self.run.put_artifact(
            "ml.resume_checkpoint",
            resume_payload,
            params=self.run.params,
            required=True,
        )

        self._progress_report(
            stage="saving_checkpoint",
            message="The resumable checkpoint is saved. Writing a prediction-ready paused model artifact.",
            current=1,
            total=2,
            unit="pause artifacts",
            epoch=completed_epoch,
            total_epochs=_configured_total_epochs(self.run),
            force=True,
        )
        model_payload = self._paused_model_payload(
            resume_ref=resume_ref,
            resume_artifact_id=resume_artifact_id,
            resume_manifest_path=resume_manifest_path,
            completed_epoch=completed_epoch,
        )
        model_dir = Path(str(resume_manifest_path)).parent.parent / "model"
        model_manifest_path = model_dir / f"paused-epoch-{completed_epoch:06d}.model_manifest.json"
        model_payload.setdefault("files", {})["manifest"] = str(model_manifest_path)
        with self._progress_activity(
            stage="saving_checkpoint",
            message="Writing the paused model manifest and registering the model artifact.",
        ):
            write_paused_model_manifest(model_manifest_path, model_payload)
            model_artifact_id = self.run.put_artifact(
                "ml.model",
                model_payload,
                params=self.run.params,
                required=True,
            )

        self._progress_report(
            stage="saving_checkpoint",
            message="Pause checkpoint and prediction-ready model are both saved.",
            current=2,
            total=2,
            unit="pause artifacts",
            epoch=completed_epoch,
            total_epochs=_configured_total_epochs(self.run),
            force=True,
        )
        payload = {
            "status": "paused",
            "message": f"Training paused after epoch {completed_epoch}.",
            "run_id": self.run.run_id,
            "dataset_id": self.run.dataset_id,
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "completed_epoch": int(completed_epoch),
            "next_epoch": int(completed_epoch) + 1,
            "resume_checkpoint_artifact_id": resume_artifact_id,
            "resume_manifest_path": str(resume_manifest_path),
            "resume_checkpoint_path": str(resume_ref.get("uri") or resume_ref.get("path")),
            "model_artifact_id": model_artifact_id,
            "model_manifest_path": str(model_manifest_path),
            "model_path": str(resume_ref.get("uri") or resume_ref.get("path")),
            "training_log_artifact_id": self.run.training_log_artifact_id,
            "artifact_ids": {
                "resume_checkpoint_artifact_id": resume_artifact_id,
                "model_artifact_id": model_artifact_id,
                "training_log_artifact_id": self.run.training_log_artifact_id,
            },
        }
        if getattr(self.run, "logger", None) is not None:
            self.run.logger.update_summary(**payload)
        self.run.log(
            message=payload["message"],
            status="paused",
            step=completed_epoch,
            metrics={},
            extra={"phase": "paused", **payload["artifact_ids"]},
        )
        publish(self.run.context, "ml.model.saved", {
            "artifact_id": model_artifact_id,
            "model_artifact_id": model_artifact_id,
            "run_id": self.run.run_id,
            "status": "paused",
            "model_path": payload["model_path"],
            "manifest_path": payload["model_manifest_path"],
        })
        publish(self.run.context, "ml.recipe_run.paused", payload)
        return payload

    def _capture_framework_resume_state(self, model: Any, components: Any, *, framework: str) -> Dict[str, Any]:
        if framework == "torch":
            module = model.module if hasattr(model, "module") else model
            state = {
                "model_state_dict": self._to_cpu(module.state_dict()),
                "parameter_requires_grad": {
                    str(name): bool(parameter.requires_grad)
                    for name, parameter in module.named_parameters()
                },
                "optimizer_state_dict": self._to_cpu(
                    components.optimizer.state_dict() if getattr(components, "optimizer", None) is not None else None
                ),
                "scheduler_state_dict": self._to_cpu(
                    components.scheduler.state_dict() if getattr(components, "scheduler", None) is not None else None
                ),
                "extra_state_dicts": {},
            }
            for key, value in dict(getattr(components, "extra", {}) or {}).items():
                state_dict = getattr(value, "state_dict", None)
                if callable(state_dict):
                    try:
                        state["extra_state_dicts"][str(key)] = self._to_cpu(state_dict())
                    except Exception:
                        pass
            return state
        if framework == "sklearn":
            from sklearn.pipeline import Pipeline

            fitted_preprocessor = deepcopy(getattr(self, "_fitted_preprocessor", None))
            fitted_model = deepcopy(model)
            prediction_model = Pipeline([
                ("preprocess", fitted_preprocessor),
                ("model", deepcopy(fitted_model)),
            ])
            return {
                "model": fitted_model,
                "fitted_preprocessor": fitted_preprocessor,
                "prediction_model": prediction_model,
            }
        raise ValueError(f"Pause/resume is not supported for framework {framework!r}.")

    def _restore_resume_state(self, model: Any, components: Any, parts: Partitions, target: TargetSpec):
        state = dict(getattr(self.run, "resume_state", {}) or {})
        if not state:
            return model, components
        self._validate_resume_identity(state, parts, target)
        self._history = [dict(row) for row in state.get("history") or [] if isinstance(row, Mapping)]
        self._best_epoch = state.get("best_epoch")
        self._best_score = state.get("best_score")
        self._best_state = state.get("best_state")
        self._last_completed_epoch = int(state.get("completed_epoch", 0) or 0)
        self.run.start_epoch = self._last_completed_epoch + 1

        framework = str(state.get("framework") or self.run.params.get("framework") or "").lower()
        framework_state = dict(state.get("framework_state") or {})
        if framework == "torch":
            module = model.module if hasattr(model, "module") else model
            model_state = framework_state.get("model_state_dict") or state.get("model_state_dict") or state.get("state_dict")
            if model_state is None:
                raise ValueError("Resume checkpoint is missing the torch model state.")
            module.load_state_dict(model_state, strict=True)
            requires_grad = dict(framework_state.get("parameter_requires_grad") or {})
            if requires_grad:
                for name, parameter in module.named_parameters():
                    if name in requires_grad:
                        parameter.requires_grad = bool(requires_grad[name])
            optimizer = getattr(components, "optimizer", None)
            optimizer_state = framework_state.get("optimizer_state_dict")
            if optimizer is not None and optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
                self._move_optimizer_state(optimizer, getattr(self, "device", "cpu"))
            scheduler = getattr(components, "scheduler", None)
            scheduler_state = framework_state.get("scheduler_state_dict")
            if scheduler is not None and scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            for key, extra_state in dict(framework_state.get("extra_state_dicts") or {}).items():
                value = dict(getattr(components, "extra", {}) or {}).get(key)
                load_state_dict = getattr(value, "load_state_dict", None)
                if callable(load_state_dict):
                    load_state_dict(extra_state)
        elif framework == "sklearn":
            saved_model = framework_state.get("model")
            if saved_model is None:
                raise ValueError("Resume checkpoint is missing the fitted sklearn estimator.")
            model = saved_model
            self._fitted_preprocessor = framework_state.get("fitted_preprocessor")
            self._preprocessor = self._fitted_preprocessor
        else:
            raise ValueError(f"Resume checkpoint uses unsupported framework {framework!r}.")

        self._restore_rng_state(state.get("rng_state") or {}, framework=framework)
        self.run.log(
            message=f"Resuming after epoch {self._last_completed_epoch}.",
            status="running",
            step=self._last_completed_epoch,
            metrics={},
            extra={"phase": "resumed", "next_epoch": self.run.start_epoch},
        )
        return model, components

    @staticmethod
    def _resume_selected_row_ids(
        params: Mapping[str, Any],
    ) -> Optional[List[str]]:
        """Normalise the exact action-level training selection."""

        raw = (
            params.get("training_row_ids")
            or params.get("selected_row_ids")
            or params.get("row_ids")
        )
        if raw is None:
            return None

        if isinstance(raw, str):
            values = [
                part.strip()
                for part in raw.replace("\n", ",").split(",")
                if part.strip()
            ]
        else:
            values = [
                str(value).strip()
                for value in raw
                if str(value).strip()
            ]

        return list(dict.fromkeys(values))

    @staticmethod
    def _resume_selected_rows_sha256(
        row_ids: Sequence[str],
    ) -> str:
        """Hash an ordered row selection without delimiter ambiguity."""

        digest = hashlib.sha256()
        for raw_row_id in row_ids:
            encoded = str(raw_row_id).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    @staticmethod
    def _resume_protocol_identity(
        protocol: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return every protocol field that can alter training semantics."""

        keys = (
            "protocol_id",
            "split_strategy",
            "validation_source",
            "test_source",
            "validation_dataset_id",
            "test_dataset_id",
            "group_column",
            "split_column",
            "validation_size",
            "test_size",
            "selection_metric",
            "selection_mode",
            "random_state",
        )
        return {
            key: protocol.get(key)
            for key in keys
        }

    @staticmethod
    def _resume_partition_identity(
        signatures: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return stable resume identity for each partition.

        ``PartitionRef.fingerprint`` is generated by the streaming split
        manifest implementation and is run-local: a fresh run can produce a
        different value for the same immutable dataset and deterministic
        protocol. Streaming resume safety is therefore established from the
        immutable dataset IDs, exact action-level selected rows, full protocol,
        partition role, and row count in ``_validate_resume_identity``.

        Ordinary in-memory ``Partition`` signatures remain membership hashes and
        retain their SHA-256 comparison.
        """

        volatile_fields = {
            "manifest_sha256",
            "manifest_uri",
            "manifest_path",
            "split_spec_artifact_id",
        }
        normalized: Dict[str, Any] = {}

        for role, raw_signature in dict(signatures or {}).items():
            role_name = str(role)
            if raw_signature is None:
                normalized[role_name] = None
                continue

            if not isinstance(raw_signature, Mapping):
                normalized[role_name] = raw_signature
                continue

            signature = {
                str(key): value
                for key, value in raw_signature.items()
                if str(key) not in volatile_fields
            }

            if "manifest_role" in signature:
                # Streaming PartitionRef.fingerprint is not a stable membership
                # digest across runs. Keep structural identity only; dataset,
                # selection and protocol identity are compared separately.
                normalized[role_name] = {
                    "count": signature.get("count"),
                    "manifest_role": signature.get("manifest_role"),
                }
            else:
                # Materialised/in-memory partitions use a true ordered
                # record-membership digest and remain strictly hash-checked.
                normalized[role_name] = signature

        return normalized

    def _validate_resume_identity(
        self,
        state: Mapping[str, Any],
        parts: Partitions,
        target: TargetSpec,
    ) -> None:
        checks = {
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "dataset_id": self.run.dataset_id,
        }
        for key, expected in checks.items():
            saved = state.get(key)
            if saved not in (None, "", expected):
                raise ValueError(
                    f"Resume checkpoint {key}={saved!r} does not match "
                    f"current {expected!r}."
                )

        saved_protocol = self._resume_protocol_identity(
            dict(state.get("protocol") or {})
        )
        current_protocol = self._resume_protocol_identity(
            self._protocol_state()
        )
        if saved_protocol != current_protocol:
            differing_fields = [
                key
                for key in saved_protocol
                if saved_protocol.get(key)
                != current_protocol.get(key)
            ]
            details = "; ".join(
                (
                    f"{key}: saved={saved_protocol.get(key)!r}, "
                    f"current={current_protocol.get(key)!r}"
                )
                for key in differing_fields
            )
            raise ValueError(
                "The data protocol no longer matches the paused run. "
                f"Differing fields: {', '.join(differing_fields)}. "
                f"{details}"
            )

        saved_binding = dict(state.get("binding") or {})
        current_binding = self._binding_state()
        if saved_binding and saved_binding != current_binding:
            raise ValueError(
                "The data binding no longer matches the paused run. "
                f"saved={saved_binding!r}; current={current_binding!r}."
            )

        saved_params = dict(state.get("params") or {})
        current_params = dict(self.run.params or {})
        saved_selection = self._resume_selected_row_ids(saved_params)
        current_selection = self._resume_selected_row_ids(current_params)

        if saved_selection != current_selection:
            saved_count = (
                len(saved_selection)
                if saved_selection is not None
                else None
            )
            current_count = (
                len(current_selection)
                if current_selection is not None
                else None
            )
            saved_hash = (
                self._resume_selected_rows_sha256(saved_selection)
                if saved_selection is not None
                else None
            )
            current_hash = (
                self._resume_selected_rows_sha256(current_selection)
                if current_selection is not None
                else None
            )
            raise ValueError(
                "The selected training rows no longer match the paused "
                "run. "
                f"saved_count={saved_count!r}; "
                f"current_count={current_count!r}; "
                f"saved_sha256={saved_hash!r}; "
                f"current_sha256={current_hash!r}."
            )

        saved_signatures = dict(
            state.get("partition_signatures") or {}
        )
        current_signatures = self._partition_signatures(parts)
        saved_identity = self._resume_partition_identity(
            saved_signatures
        )
        current_identity = self._resume_partition_identity(
            current_signatures
        )

        if saved_signatures and saved_identity != current_identity:
            differing_roles = [
                role
                for role in ("train", "validation", "test")
                if saved_identity.get(role)
                != current_identity.get(role)
            ]
            details = "; ".join(
                (
                    f"{role}: saved={saved_identity.get(role)!r}, "
                    f"current={current_identity.get(role)!r}"
                )
                for role in differing_roles
            )
            raise ValueError(
                "The dataset partitions no longer match the paused run. "
                "Resume was refused to avoid training on a different "
                "train/validation/test split. "
                f"Differing roles: {', '.join(differing_roles)}. "
                f"{details}"
            )

        saved_target = dict(state.get("target") or {})
        if saved_target and saved_target != self._target_state(target):
            raise ValueError(
                "The target/class schema no longer matches the paused run."
            )

    def _paused_model_payload(
        self,
        *,
        resume_ref: Mapping[str, Any],
        resume_artifact_id: str,
        resume_manifest_path: Path,
        completed_epoch: int,
    ) -> Dict[str, Any]:
        parts, target = self._parts, self._target
        assert parts is not None and target is not None
        title = str(getattr(self.recipe, "title", None) or self.run.recipe_id)
        return json_safe({
            "artifact_type": "ml.model",
            "schema_version": 3,
            "kind": "paused_model",
            "framework": str(self.run.params.get("framework") or getattr(self.recipe, "framework", "")),
            "task": self._task_kind(),
            "modality": str(getattr(self.recipe, "modality", "") or self.run.params.get("modality", "")),
            "run_id": self.run.run_id,
            "source_dataset_id": self.run.dataset_id,
            "dataset_id": parts.train_dataset_id or self.run.dataset_id,
            "train_dataset_id": parts.train_dataset_id,
            "validation_dataset_id": parts.validation_dataset_id,
            "test_dataset_id": parts.test_dataset_id,
            "split_dataset_ids": dict(parts.materialized_split_dataset_ids or {}),
            "split_spec_artifact_id": self._split_spec_artifact_id,
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "protocol_id": parts.protocol_id,
            "created_at": time.time(),
            "training_status": "paused",
            "resumable": True,
            "paused_epoch": int(completed_epoch),
            "resume_checkpoint_artifact_id": resume_artifact_id,
            "resume_manifest_path": str(resume_manifest_path),
            "training_log_artifact_id": self.run.training_log_artifact_id,
            "model_title": f"{title} (paused at epoch {completed_epoch})",
            "model_id": f"{self.run.recipe_id}.paused.{self.run.run_id}.{completed_epoch}",
            "model_ref": dict(resume_ref),
            "class_names": list(target.classes),
            "num_classes": int(target.num_classes),
            "num_outputs": int(target.num_outputs),
            "feature_columns": list(self.binding.input_columns or []),
            "input_contract": self._binding_state(),
            "protocol": self._protocol_state(),
            "params": dict(self.run.params),
            "metrics": {
                "best_score": self._best_score,
                "selection_metric": self.protocol.selection_metric,
                "best_epoch": self._best_epoch,
                "paused_epoch": int(completed_epoch),
            },
            "files": {
                "checkpoint": str(resume_ref.get("uri") or resume_ref.get("path")),
                "resume_manifest": str(resume_manifest_path),
            },
        })

    def _partition_signatures(self, parts: Partitions) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name, partition in (("train", parts.train), ("validation", parts.val), ("test", parts.test)):
            if partition is None:
                result[name] = None
                continue
            encoded = "\0".join(str(value) for value in partition.record_ids).encode("utf-8")
            result[name] = {"count": len(partition.record_ids), "sha256": hashlib.sha256(encoded).hexdigest()}
        return result

    def _protocol_state(self) -> Dict[str, Any]:
        p = self.protocol
        return {
            "protocol_id": p.protocol_id,
            "split_strategy": p.split_strategy,
            "validation_source": p.validation_source,
            "test_source": p.test_source,
            "validation_dataset_id": p.validation_dataset_id,
            "test_dataset_id": p.test_dataset_id,
            "group_column": p.group_column,
            "split_column": p.split_column,
            "validation_size": p.validation_size,
            "test_size": p.test_size,
            "selection_metric": p.selection_metric,
            "selection_mode": p.selection_mode,
            "random_state": p.random_state,
        }

    def _binding_state(self) -> Dict[str, Any]:
        b = self.binding
        return {
            "record_id_column": b.record_id_column,
            "target_column": b.target_column,
            "image_column": b.image_column,
            "feature_columns": list(b.input_columns or []),
            "input_columns": list(b.input_columns or []),
        }

    @staticmethod
    def _target_state(target: TargetSpec) -> Dict[str, Any]:
        return {"kind": target.kind, "classes": list(target.classes), "n_outputs": int(target.n_outputs)}

    def _capture_rng_state(self, *, framework: str) -> Dict[str, Any]:
        import numpy as np
        state: Dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
        if framework == "torch":
            import torch
            state["torch_cpu"] = torch.get_rng_state()
            if torch.cuda.is_available():
                try:
                    state["torch_cuda"] = torch.cuda.get_rng_state_all()
                except Exception:
                    pass
        return state

    def _restore_rng_state(self, state: Mapping[str, Any], *, framework: str) -> None:
        import numpy as np
        if state.get("python") is not None:
            random.setstate(state["python"])
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
        if framework == "torch":
            import torch
            if state.get("torch_cpu") is not None:
                torch.set_rng_state(state["torch_cpu"])
            if state.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _to_cpu(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: self._to_cpu(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._to_cpu(item) for item in value)
        if isinstance(value, list):
            return [self._to_cpu(item) for item in value]
        detach = getattr(value, "detach", None)
        cpu = getattr(value, "cpu", None)
        if callable(detach) and callable(cpu):
            try:
                return value.detach().cpu()
            except Exception:
                return value
        return value

    def _move_optimizer_state(self, optimizer: Any, device: Any) -> None:
        for state in getattr(optimizer, "state", {}).values():
            for key, value in list(state.items()):
                to = getattr(value, "to", None)
                if callable(to):
                    try:
                        state[key] = value.to(device)
                    except Exception:
                        pass

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
        parts = None
        model = None
        components = None
        train_loader = None
        test_loader = None
        completed = False

        try:
            run.check_cancelled()
            with self._progress_activity(
                stage="partitioning",
                message="Scanning the dataset and creating train, validation, and test membership.",
                detail="Only protocol-required columns are scanned by streaming harnesses; materialised recipes follow their declared compatibility path.",
            ):
                parts = self._partition()
            run.check_cancelled()

            partition_counts = {
                "train": _partition_length(parts.train),
                "validation": _partition_length(parts.val),
                "test": _partition_length(parts.test),
            }
            self._progress_report(
                stage="partitioning",
                message="Dataset partitioning is complete.",
                detail=(
                    f"Train `{partition_counts['train']:,}`, validation `{partition_counts['validation']:,}`, "
                    f"test `{partition_counts['test']:,}` rows."
                ),
                current=sum(partition_counts.values()),
                total=sum(partition_counts.values()),
                unit="rows",
                force=True,
            )

            self._progress_report(
                stage="split_manifest",
                message="Saving split membership, checksums, and protocol provenance.",
                force=True,
            )
            split_spec_id = self._write_split_spec(parts)
            run.check_cancelled()
            self._parts = parts
            self._split_spec_artifact_id = split_spec_id

            target = self._target_spec(parts)
            self._target = target
            self._progress_report(
                stage="building_model",
                message="Building the model with the resolved target/output shape.",
                detail=(
                    f"Task `{target.kind}`, output width `{target.num_outputs}`, "
                    f"recipe `{self.run.recipe_id}`."
                ),
                force=True,
            )
            with self._progress_activity(
                stage="building_model",
                message="Constructing model architecture and initial parameters.",
            ):
                model = self._build_model(parts, target)
            run.check_cancelled()

            self._progress_report(
                stage="configuring_training",
                message="Creating loss, optimiser, scheduler, and recipe-specific training components.",
                force=True,
            )
            components = recipe.configure_training(run, model)
            run.check_cancelled()

            self._progress_report(
                stage="loading_data",
                message="Preparing bounded train and validation data access.",
                force=True,
            )
            with self._progress_activity(
                stage="loading_data",
                message="Preparing train and validation loaders/readers.",
            ):
                train_loader = self._make_loader(parts.train, train=True)
                self._val_loader = self._make_loader(parts.val, train=False)
            run.check_cancelled()

            if getattr(self.run, "resume_state", None):
                self._progress_report(
                    stage="restoring",
                    message="Restoring model, optimiser, scheduler, RNG, history, and best-checkpoint state.",
                    force=True,
                )
            model, components = self._restore_resume_state(model, components, parts, target)
            run.check_cancelled()

            self._progress_report(
                stage="output_validation",
                message="Checking that model outputs match the resolved task and target schema.",
                force=True,
            )
            self._assert_output_dim(model, target, train_loader)
            run.check_cancelled()
            if getattr(self.run, "resume_state", None):
                framework = str(
                    self.run.resume_state.get("framework")
                    or self.run.params.get("framework")
                    or getattr(self.recipe, "framework", "")
                ).lower()
                self._restore_rng_state(
                    self.run.resume_state.get("rng_state") or {},
                    framework=framework,
                )

            train_loader = self._instrument_training_loader(train_loader)
            total_epochs = _configured_total_epochs(run)
            self._progress_report(
                stage="training",
                message="Starting the recipe training loop.",
                detail=(
                    f"Training will run for `{total_epochs}` epoch(s)."
                    if total_epochs
                    else "The recipe controls the number of training passes and reports each completed epoch."
                ),
                current=max(0, int(getattr(run, "start_epoch", 1) or 1) - 1),
                total=total_epochs,
                unit="epochs",
                epoch=max(1, int(getattr(run, "start_epoch", 1) or 1)),
                total_epochs=total_epochs,
                force=True,
            )
            with self._progress_activity(
                stage="training",
                message="Training is active. Reading batches and updating model parameters.",
                detail="Batch/row counters appear when the selected harness exposes them.",
            ):
                run.check_cancelled()
                recipe.fit(
                    run,
                    model=model,
                    components=components,
                    train_loader=train_loader,
                    harness=self,
                )
            run.check_cancelled()

            if self._best_state is None:
                raise RuntimeError(
                    "Recipe completed without calling harness.report_epoch(...). "
                    "A managed recipe must report at least one epoch so the harness "
                    "can select on the validation partition."
                )

            self._progress_report(
                stage="selecting_checkpoint",
                message=f"Restoring the best model state from epoch {self._best_epoch}.",
                detail=f"Best `{self.protocol.selection_metric}` score: `{self._best_score}`.",
                force=True,
            )
            self._restore(model, self._best_state)

            test_metrics, test_records = {}, []
            if parts.test is not None and len(parts.test) > 0:
                self._progress_report(
                    stage="test_evaluation",
                    message="Preparing the held-out test partition for its one-time evaluation.",
                    current=0,
                    total=_partition_length(parts.test),
                    unit="rows",
                    force=True,
                )
                test_loader = self._make_loader(parts.test, train=False)
                with self._progress_activity(
                    stage="test_evaluation",
                    message="Running one-time test prediction and metric calculation.",
                ):
                    test_metrics, test_records = self._evaluate(
                        model,
                        test_loader,
                        return_records=True,
                    )
                self._progress_report(
                    stage="test_evaluation",
                    message="Held-out test evaluation is complete.",
                    detail=f"Metrics: `{json_safe(test_metrics)}`",
                    current=_partition_length(parts.test),
                    total=_partition_length(parts.test),
                    unit="rows",
                    metrics=test_metrics,
                    force=True,
                )

            self._progress_report(
                stage="saving_model",
                message="Serialising the selected model and writing its durable manifest.",
                force=True,
            )
            with self._progress_activity(
                stage="saving_model",
                message="Writing the trained model sidecar and compatibility metadata.",
            ):
                model_artifact_id = self._write_model_artifact(
                    model,
                    parts,
                    target,
                    split_spec_artifact_id=split_spec_id,
                )
            if not model_artifact_id:
                raise RuntimeError("The trained model could not be persisted as an ml.model artifact.")

            self._progress_report(
                stage="saving_evaluation",
                message="Writing the evaluation report and experiment provenance.",
                force=True,
            )
            eval_id = self._write_evaluation_report(
                parts=parts,
                split_spec_id=split_spec_id,
                test_metrics=test_metrics,
                model_artifact_id=model_artifact_id,
            )

            predictions_id = None
            if test_records:
                self._progress_report(
                    stage="saving_predictions",
                    message="Writing row-keyed held-out predictions and their bounded artifact preview.",
                    current=0,
                    total=len(test_records),
                    unit="predictions",
                    force=True,
                )
                predictions_id = self._write_predictions(
                    test_records,
                    parts,
                    model_artifact_id,
                )
                self._progress_report(
                    stage="saving_predictions",
                    message="Held-out predictions were saved.",
                    current=len(test_records),
                    total=len(test_records),
                    unit="predictions",
                    force=True,
                )

            result = {
                "status": "complete",
                "model_artifact_id": model_artifact_id,
                "evaluation_report_artifact_id": eval_id,
                "predictions_artifact_id": predictions_id,
                "split_spec_artifact_id": split_spec_id,
                "source_dataset_id": self.run.dataset_id,
                "train_dataset_id": parts.train_dataset_id,
                "validation_dataset_id": parts.validation_dataset_id,
                "test_dataset_id": parts.test_dataset_id,
                "split_dataset_ids": dict(parts.materialized_split_dataset_ids or {}),
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

            self._progress_report(
                stage="finalizing",
                message="Model, metrics, and prediction artifacts are complete. Returning the run result.",
                force=True,
            )
            completed = True
            return result
        finally:
            cleanup = getattr(recipe, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup(run)
                except Exception:
                    pass

            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
                try:
                    for parameter in model.parameters():
                        parameter.grad = None
                except Exception:
                    pass

            optimizer = getattr(components, "optimizer", None)
            if optimizer is not None:
                try:
                    optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
                try:
                    optimizer.state.clear()
                except Exception:
                    pass

            if components is not None:
                for attribute in ("optimizer", "scheduler", "criterion"):
                    try:
                        setattr(components, attribute, None)
                    except Exception:
                        pass
                try:
                    components.extra.clear()
                except Exception:
                    pass

            self._val_loader = None
            self._best_state = None
            self._frame = None
            self._partition_frames.clear()
            self._parts = None
            self._target = None
            test_loader = None
            train_loader = None
            optimizer = None
            components = None
            model = None
            parts = None
            with self._progress_activity(
                stage="finalizing",
                message="Releasing loaders, temporary tensors, accelerator caches, and run-local resources.",
            ):
                cleanup_ml_runtime(
                    reason="managed recipe harness cleanup",
                    aggressive=not completed,
                )

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
            raise ValueError("External split strategy requires a validation dataset.")

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

        if self._task_kind() == "regression":
            classes: List[str] = []
        else:
            classes = self._classification_classes(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
            )

        train_p = self._partition_from_frame(
            name="train",
            frame=train_df,
            dataset_id=train_dataset_id,
            classes=classes,
            source="selected",
        )
        val_p = self._partition_from_frame(
            name="val",
            frame=val_df,
            dataset_id=val_dataset_id,
            classes=classes,
            source="dataset",
        )

        test_p = None
        if test_df is not None:
            test_p = self._partition_from_frame(
                name="test",
                frame=test_df,
                dataset_id=test_dataset_id,
                classes=classes,
                source="dataset",
            )

        self._assert_disjoint(train_p, val_p, test_p)

        if self._task_kind() != "regression":
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
            validation_source=p.validation_source,
            test_source=p.test_source,
            group_column=None,
            random_state=p.random_state,
            protocol_id=p.protocol_id,
            target_column=b.target_column or "",
            record_id_column=b.record_id_column,
            train_dataset_id=train_dataset_id,
            validation_dataset_id=val_dataset_id,
            test_dataset_id=test_dataset_id,
            materialized_split_dataset_ids={},
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

        if p.validation_source == "split":
            val_df = base_df.iloc[val_idx].reset_index(drop=True)
            val_dataset_id = self.run.dataset_id
            val_source = "split"
        else:
            val_df = self._load_partition_frame(
                p.validation_dataset_id,
                columns=cols,
                role="validation",
            )
            val_dataset_id = p.validation_dataset_id
            val_source = "dataset"

        test_df = None
        test_dataset_id = None
        test_source = "none"

        if p.test_source == "split":
            test_df = base_df.iloc[test_idx].reset_index(drop=True)
            test_dataset_id = self.run.dataset_id
            test_source = "split"
        elif p.test_source == "dataset":
            test_df = self._load_partition_frame(
                p.test_dataset_id,
                columns=cols,
                role="test",
            )
            test_dataset_id = p.test_dataset_id
            test_source = "dataset"

        if regression:
            classes: List[str] = []
        else:
            classes = self._classification_classes(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
            )

        train_p = self._partition_from_frame(
            name="train",
            frame=train_df,
            dataset_id=self.run.dataset_id,
            classes=classes,
            source="selected",
        )

        val_p = self._partition_from_frame(
            name="val",
            frame=val_df,
            dataset_id=val_dataset_id,
            classes=classes,
            source=val_source,
        )

        test_p = None
        if test_df is not None:
            test_p = self._partition_from_frame(
                name="test",
                frame=test_df,
                dataset_id=test_dataset_id,
                classes=classes,
                source=test_source,
            )

        self._assert_disjoint(train_p, val_p, test_p)

        if not regression:
            self._assert_label_subset(classes, val_p, test_p)

        if len(val_p) == 0:
            raise ValueError("Validation partition is empty.")

        split_dataset_ids: Dict[str, str] = {}

        # Only materialise partitions that were actually split from the selected
        # source dataset. External validation/test datasets are already explicit
        # platform datasets.
        split_partitions = {}
        split_frames = {}

        if val_from_split or test_from_split:
            split_partitions["train"] = train_p
            split_frames["train"] = train_df

        if val_from_split:
            split_partitions["validation"] = val_p
            split_frames["validation"] = val_df

        if test_from_split and test_p is not None and test_df is not None:
            split_partitions["test"] = test_p
            split_frames["test"] = test_df

        if split_partitions:
            split_dataset_ids = materialize_split_datasets(
                context=self.run.context,
                source_dataset_id=self.run.dataset_id,
                run_id=self.run.run_id,
                recipe_id=self.run.recipe_id,
                recipe_version=self.run.recipe_version,
                protocol=p,
                binding=b,
                params=self.run.params,
                partitions=split_partitions,
                fallback_frames=split_frames,
            )

            if split_dataset_ids.get("train"):
                train_p.dataset_id = split_dataset_ids["train"]
                train_p.source = "materialized_split"

            if split_dataset_ids.get("validation"):
                val_p.dataset_id = split_dataset_ids["validation"]
                val_p.source = "materialized_split"

            if test_p is not None and split_dataset_ids.get("test"):
                test_p.dataset_id = split_dataset_ids["test"]
                test_p.source = "materialized_split"

            self.run.params["train_dataset_id"] = train_p.dataset_id
            self.run.params["validation_dataset_id"] = val_p.dataset_id
            self.run.params["test_dataset_id"] = (
                test_p.dataset_id if test_p is not None else None
            )
            self.run.params["split_dataset_ids"] = dict(split_dataset_ids)

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
            train_dataset_id=train_p.dataset_id,
            validation_dataset_id=val_p.dataset_id,
            test_dataset_id=test_p.dataset_id if test_p is not None else None,
            materialized_split_dataset_ids=split_dataset_ids,
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

    def _is_active_learning_run(self) -> bool:
        params = dict(getattr(self.run, "params", {}) or {})
        if params.get("al_session_id") or params.get("al_session_artifact_id"):
            return True

        protocol = str(params.get("al_protocol") or "").strip().lower()
        return protocol in {"review", "active_learning", "active-learning", "al", "benchmark"}

    def _normalise_class_list(self, value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            parts = [part.strip() for part in value.replace("\n", ",").split(",")]
        elif isinstance(value, Mapping):
            parts = [str(key).strip() for key in value.keys()]
        elif isinstance(value, Iterable):
            parts = [str(item).strip() for item in value]
        else:
            parts = [str(value).strip()]

        out: List[str] = []
        for item in parts:
            if not item:
                continue
            if item not in out:
                out.append(item)
        return out

    def _labels_from_frame(self, frame) -> List[str]:
        b = self.binding
        if frame is None or not b.target_column or b.target_column not in frame.columns:
            return []

        values: List[str] = []
        try:
            raw = frame[b.target_column].dropna().astype(str).tolist()
        except Exception:
            raw = []

        for value in raw:
            value = str(value).strip()
            if value and value not in values:
                values.append(value)
        return values

    def _declared_class_universe(self) -> List[str]:
        """Return the run-declared class universe, preserving user order.

        Active-learning runs should pass this from the AL session's label_options.
        The harness also accepts common aliases because recipes/plugins may use
        different names.
        """
        params = dict(getattr(self.run, "params", {}) or {})

        for key in (
            "label_options",
            "class_labels",
            "classes",
            "class_names",
            "known_classes",
            "target_classes",
        ):
            labels = self._normalise_class_list(params.get(key))
            if labels:
                return labels

        return []

    def _classification_classes(
        self,
        *,
        train_df,
        val_df=None,
        test_df=None,
    ) -> List[str]:
        """Resolve model class universe for classification.

        Non-AL default:
            Use classes present in the training partition.

        AL default:
            Prefer declared label_options/classes from the AL session. If those
            are missing, include labels from train/val/test so evaluation can
            cover every known label instead of crashing on labels absent from
            the initial training subset.
        """
        train_labels = self._labels_from_frame(train_df)
        val_labels = self._labels_from_frame(val_df)
        test_labels = self._labels_from_frame(test_df)

        declared = self._declared_class_universe()
        is_al = self._is_active_learning_run()

        if declared:
            classes = list(declared)
            source = "declared"
        elif is_al:
            classes = list(dict.fromkeys([*train_labels, *val_labels, *test_labels]))
            source = "active_learning_eval_partitions"
        else:
            classes = sorted(set(train_labels))
            source = "training_partition"

        class_set = set(str(label) for label in classes)

        missing_train_labels = sorted(set(str(label) for label in train_labels) - class_set)
        if missing_train_labels:
            raise ValueError(
                "Training labels contain value(s) outside the model class universe: "
                f"{missing_train_labels}. Known classes are {classes}."
            )

        if len(classes) < 2:
            raise ValueError(
                f"Classification requires at least two known classes. Resolved {classes} "
                f"from source={source!r}."
            )

        train_seen = sorted(set(str(label) for label in train_labels))
        val_seen = sorted(set(str(label) for label in val_labels))
        test_seen = sorted(set(str(label) for label in test_labels))

        self._class_universe_info = {
            "source": source,
            "active_learning": bool(is_al),
            "classes": list(classes),
            "train_seen_classes": train_seen,
            "validation_seen_classes": val_seen,
            "test_seen_classes": test_seen,
            "classes_without_training_examples": [
                label for label in classes if str(label) not in set(train_seen)
            ],
            "evaluation_includes_classes_absent_from_training": bool(
                (set(val_seen) | set(test_seen)) - set(train_seen)
            ),
        }

        return classes

    def _assert_label_subset(self, classes, val_p, test_p):
        cset = set(str(label) for label in classes)

        for part in (val_p, test_p):
            if part is None:
                continue

            extra = sorted(set(str(label) for label in part.labels) - cset)
            if extra:
                raise ValueError(
                    f"{part.name} contains labels that are not in the model class "
                    f"universe: {extra}. Known classes are {sorted(cset)}. "
                    "For active learning, pass the full class list through "
                    "label_options/classes/class_labels when training."
                )

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
                "schema_version": 3,
                "run_id": self.run.run_id,
                "dataset_id": self.run.dataset_id,
                "source_dataset_id": self.run.dataset_id,
                "train_dataset_id": parts.train_dataset_id,
                "validation_dataset_id": parts.validation_dataset_id,
                "test_dataset_id": parts.test_dataset_id,
                "split_dataset_ids": dict(parts.materialized_split_dataset_ids or {}),
                "partition_dataset_ids": {
                    "train": parts.train_dataset_id,
                    "validation": parts.validation_dataset_id,
                    "test": parts.test_dataset_id,
                },
                "partition_sources": {
                    "train": parts.train.source,
                    "validation": parts.val.source,
                    "test": parts.test.source if parts.test else "none",
                },
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
                "class_universe": json_safe(
                    dict(getattr(self, "_class_universe_info", {}) or {})
                ),
                "isolation": {
                    "partitions_disjoint": True,
                    "test_used_in_training": False,
                    "test_used_in_selection": False,
                    "selection_partition": "validation",
                    "explicit_split_datasets": bool(parts.materialized_split_dataset_ids),
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
                "classes": parts.train.classes,
                "class_universe": json_safe(
                    dict(getattr(self, "_class_universe_info", {}) or {})
                ),
                "evaluation_scope": "all_known_classes",
                "validity": {
                    "metric_partition": "test" if parts.test else "none",
                    "selection_partition": "validation",
                    "selection_touched_reported_partition": False,
                },
            },
        )

    def _write_predictions(self, records, parts, model_artifact_id) -> Optional[str]:
        from pathlib import Path

        from ..artifacts import save_predictions_sidecar

        rows = [dict(record) for record in records]
        prediction_ref = None
        if bool(self.run.params.get("save_predictions", True)):
            prediction_ref = save_predictions_sidecar(
                context=self.run.context,
                run_id=self.run.run_id,
                dataset_id=self.run.dataset_id,
                model_artifact_id=str(model_artifact_id or "model"),
                rows=rows,
                params=self.run.params,
            )

        inline_limit_value = self.run.params.get(
            "prediction_inline_limit"
        )
        inline_limit = max(
            0,
            (
                1000
                if inline_limit_value in (None, "")
                else int(inline_limit_value)
            ),
        )

        compact = (
            prediction_ref is not None
            and len(rows) > inline_limit
        )

        payload = {
            "schema_version": 3,
            "run_id": self.run.run_id,
            "dataset_id": self.run.dataset_id,
            "model_artifact_id": model_artifact_id,
            "protocol_id": parts.protocol_id,
            "prediction_scope": "test",
            "record_id_column": parts.record_id_column,
            "class_names": parts.train.classes,
            "row_count": len(rows),
            "records": [] if compact else rows,
            "records_preview": rows[:25] if compact else [],
            "records_inline_complete": not compact,
            "prediction_ref": (
                prediction_ref.to_dict()
                if prediction_ref is not None
                else None
            ),
        }
        try:
            return self.run.put_artifact(
                "ml.predictions",
                payload,
                row_ids=[record["record_id"] for record in rows],
                required=True,
            )
        except Exception:
            if prediction_ref is not None:
                try:
                    Path(prediction_ref.uri).unlink(missing_ok=True)
                except Exception:
                    pass
            raise

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

def _partition_length(partition: Any) -> int:
    if partition is None:
        return 0
    try:
        return max(0, int(len(partition)))
    except Exception:
        pass
    try:
        return max(0, int(getattr(partition, "row_count", 0) or 0))
    except Exception:
        return 0

def _configured_total_epochs(run: Any) -> Optional[int]:
    params = dict(getattr(run, "params", {}) or {})
    for key in ("epochs", "num_epochs", "max_epochs", "n_estimators"):
        try:
            value = int(params.get(key) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    return None