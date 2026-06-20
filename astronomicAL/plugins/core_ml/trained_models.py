from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

@dataclass(frozen=True)
class TrainedModelDescriptor:
    artifact_id: str
    title: str
    task: str
    modality: str
    framework: str
    model_id: Optional[str] = None
    run_id: Optional[str] = None
    trained_dataset_id: Optional[str] = None
    target_column: Optional[str] = None
    input_summary: Dict[str, Any] = field(default_factory=dict)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[float] = None
    model_ref: Optional[Dict[str, Any]] = None
    al_strategy: Optional[str] = None
    hyperparameter_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        from . import model_contract as contract
        return contract.json_safe(asdict(self))


class TrainedModelCatalog:
    """Indexes durable ml.model artifacts for reuse in prediction workflows."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self._models: Dict[str, TrainedModelDescriptor] = {}
        self.refresh()
        self._subscriptions: List[Any] = []
        self._subscribe()

    def dispose(self) -> None:
        events = getattr(self.context, "events", None)
        unsubscribe = getattr(events, "unsubscribe", None)
        if callable(unsubscribe):
            for sub in list(self._subscriptions):
                try:
                    unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions.clear()

    def refresh(self) -> int:
        from . import model_contract as contract_utils

        artifacts = getattr(self.context, "artifacts", None)
        find = getattr(artifacts, "find", None)
        get = getattr(artifacts, "get", None)
        if not callable(find) or not callable(get):
            self._models = {}
            return 0
        descriptors: Dict[str, TrainedModelDescriptor] = {}
        try:
            refs = find(type="ml.model")
        except Exception:
            refs = []
        for ref in refs:
            artifact_id = getattr(ref, "artifact_id", None)
            if not artifact_id:
                continue
            try:
                payload = get(artifact_id)
                if not isinstance(payload, Mapping):
                    continue
                contract = contract_utils.ensure_model_contract(
                    context=self.context,
                    model_artifact_id=str(artifact_id),
                    model_payload=payload,
                    persist=True,
                )
                descriptors[str(artifact_id)] = self._descriptor_from_payload(
                    artifact_id=str(artifact_id),
                    payload=payload,
                    contract=contract,
                    ref=ref,
                )
            except Exception:
                continue
        self._models = descriptors
        return len(descriptors)

    def list_models(
        self,
        *,
        task: Optional[str] = None,
        modality: Optional[str] = None,
        framework: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> List[TrainedModelDescriptor]:
        models = list(self._models.values())
        if task:
            models = [m for m in models if m.task == task]
        if modality:
            models = [m for m in models if m.modality == modality]
        if framework:
            models = [m for m in models if m.framework == framework]
        if dataset_id:
            models = [m for m in models if m.trained_dataset_id == dataset_id]
        return sorted(models, key=lambda m: (-(m.created_at or 0), m.title, m.artifact_id))

    def get(self, artifact_id: str) -> TrainedModelDescriptor:
        if artifact_id not in self._models:
            self.refresh()
        return self._models[artifact_id]

    def as_options(self, *, task: Optional[str] = None, modality: Optional[str] = None) -> Dict[str, str]:
        """Return Panel-friendly {label: artifact_id} options."""
        options: Dict[str, str] = {}

        for model in self.list_models(task=task, modality=modality):
            label_bits = [model.title]

            label_bits.append(f"{model.modality}/{model.task}")

            classes = model.output_summary.get("classes") or []
            if classes:
                label_bits.append(f"{len(classes)} classes")

            if model.target_column:
                label_bits.append(f"target={model.target_column}")

            best_metric = (
                model.metrics.get("best_val_accuracy")
                or model.metrics.get("best_score")
                or model.metrics.get("test_accuracy")
            )

            if isinstance(best_metric, (int, float)):
                label_bits.append(f"best={best_metric:.3f}")

            if model.run_id:
                label_bits.append(f"run={str(model.run_id)[:8]}")

            options[" — ".join(str(bit) for bit in label_bits if bit)] = model.artifact_id

        return options

    def compatibility(self, artifact_id: str, dataset_id: str, **kwargs: Any) -> Dict[str, Any]:
        from . import model_contract as contract_utils

        report = contract_utils.validate_model_for_dataset(
            context=self.context,
            model_artifact_id=artifact_id,
            dataset_id=dataset_id,
            **kwargs,
        )
        return report.to_dict()

    def _descriptor_from_payload(
        self,
        *,
        artifact_id: str,
        payload: Mapping[str, Any],
        contract: Mapping[str, Any],
        ref: Any,
    ) -> TrainedModelDescriptor:
        output = dict(contract.get("output_schema") or {})
        trained_on = dict(contract.get("trained_on") or {})
        input_schema = dict(contract.get("input_schema") or {})
        training_context = dict(contract.get("training_context") or {})
        metrics = self._find_metrics_for_run(payload.get("run_id"), artifact_id)
        return TrainedModelDescriptor(
            artifact_id=artifact_id,
            title=str(contract.get("model_title") or payload.get("model_title") or artifact_id),
            task=str(contract.get("task") or payload.get("task") or "classification"),
            modality=str(contract.get("modality") or payload.get("modality") or "tabular"),
            framework=str(contract.get("framework") or payload.get("framework") or ""),
            model_id=payload.get("model_id"),
            run_id=payload.get("run_id"),
            trained_dataset_id=trained_on.get("dataset_id") or payload.get("dataset_id"),
            target_column=trained_on.get("target_column") or payload.get("target_column"),
            input_summary={
                "kind": input_schema.get("kind"),
                "feature_columns": input_schema.get("feature_columns"),
                "image_column": input_schema.get("image_column"),
                "target_column": input_schema.get("target_column"),
                "record_id_column": input_schema.get("record_id_column"),
                "image_size": input_schema.get("image_size"),
                "normalization": input_schema.get("normalization"),
                "transform": input_schema.get("transform"),
                "feature_schema_hash": training_context.get("feature_schema_hash"),
            },
            output_summary={
                "kind": output.get("kind"),
                "classes": output.get("classes"),
                "class_order": output.get("class_order"),
                "n_classes": output.get("n_classes"),
                "has_probabilities": output.get("has_probabilities"),
                "prediction_column": output.get("prediction_column"),
                "confidence_column": output.get("confidence_column"),
                "uncertainty_columns": output.get("uncertainty_columns"),
                "probability_columns": output.get("probability_columns"),
            },
            metrics=metrics,
            created_at=payload.get("created_at") or getattr(ref, "created_at", None),
            model_ref=payload.get("model_ref"),
            al_strategy=training_context.get("active_learning_strategy"),
            hyperparameter_summary=_summarise_hyperparameters(training_context.get("hyperparameters") or {}),
        )

    def _find_metrics_for_run(self, run_id: Any, model_artifact_id: str) -> Dict[str, Any]:
        artifacts = getattr(self.context, "artifacts", None)
        find = getattr(artifacts, "find", None)
        get = getattr(artifacts, "get", None)
        if not callable(find) or not callable(get):
            return {}
        refs = []
        if run_id:
            try:
                refs = find(type="ml.evaluation_report", params_subset={"run_id": run_id})
            except Exception:
                refs = []
        if not refs:
            try:
                refs = find(type="ml.evaluation_report")
            except Exception:
                refs = []
        for ref in refs:
            try:
                payload = get(ref.artifact_id)
            except Exception:
                continue
            if not isinstance(payload, Mapping):
                continue
            if run_id and payload.get("run_id") != run_id:
                continue
            metrics = payload.get("metrics")
            if isinstance(metrics, Mapping):
                return dict(metrics)
        return {}

    def _subscribe(self) -> None:
        events = getattr(self.context, "events", None)
        subscribe = getattr(events, "subscribe", None)
        if not callable(subscribe):
            return
        for topic in ("ml.model.saved", "ml.run.finished", "artifact.created", "workspace.restored"):
            try:
                self._subscriptions.append(subscribe(topic, self._on_event))
            except Exception:
                pass

    def _on_event(self, *_: Any, **__: Any) -> None:
        try:
            self.refresh()
        except Exception:
            pass


def create_trained_model_catalog(context: Any) -> TrainedModelCatalog:
    return TrainedModelCatalog(context)


def _summarise_hyperparameters(params: Mapping[str, Any], *, max_items: int = 6) -> str:
    if not isinstance(params, Mapping) or not params:
        return ""
    bits = []
    for idx, (key, value) in enumerate(params.items()):
        if idx >= max_items:
            bits.append("...")
            break
        bits.append(f"{key}={value}")
    return ", ".join(bits)