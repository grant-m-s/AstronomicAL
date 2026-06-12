from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

import importlib.util
import sys

ML_ARTIFACT_SCHEMA_VERSION = 1

def _load_sibling_module(stem: str):
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

@dataclass(frozen=True)
class StoredModelRef:
    """JSON-safe pointer to a model persisted outside ArtifactStore payloads."""

    storage: str
    uri: str
    format: str
    framework: str
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MLArtifactContract:
    """Documented ML artifact types produced and consumed by core.ml."""

    MODEL_DEFINITION: str = "ml.model_definition"
    FEATURE_SPEC: str = "ml.feature_spec"
    IMAGE_SPEC: str = "ml.image_spec"
    SPLIT_SPEC: str = "ml.split_spec"
    MODEL: str = "ml.model"
    EVALUATION_REPORT: str = "ml.evaluation_report"
    PREDICTIONS: str = "ml.predictions"
    TRAINING_LOG: str = "ml.training_log"
    RUN: str = "ml.run"
    ACTIVE_LEARNING_BATCH: str = "ml.active_learning_batch"


ARTIFACTS = MLArtifactContract()


def json_safe(value: Any) -> Any:
    """Convert common scientific Python values into JSON-safe objects."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return value

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]

    if isinstance(value, pd.Series):
        return [json_safe(v) for v in value.tolist()]

    if isinstance(value, pd.Index):
        return [json_safe(v) for v in value.tolist()]

    if isinstance(value, pd.DataFrame):
        return [json_safe(row) for row in value.to_dict(orient="records")]

    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]

    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


def ensure_json_safe(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe shallow/deep copy of an artifact payload."""

    safe = json_safe(dict(payload))
    if not isinstance(safe, dict):
        raise TypeError("Expected payload to coerce to a JSON object.")
    safe.setdefault("schema_version", ML_ARTIFACT_SCHEMA_VERSION)
    return safe


def ml_storage_dir(context: Any, *, kind: str = "models") -> Path:
    """Return a writable storage directory for ML sidecar files.

    ArtifactStore currently persists JSON payloads, but it does not yet have a
    first-class sidecar-file API. Store heavy model files beside the artifact
    cache when possible, otherwise fall back to a local project directory.
    """

    artifacts = getattr(context, "artifacts", None)
    cache_dir = getattr(artifacts, "_cache_dir", None)
    if cache_dir:
        root = Path(cache_dir)
    else:
        root = Path.cwd() / ".astronomical" / "ml_artifacts"

    path = root / kind
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_model_sidecar(
    *,
    context: Any,
    model: Any,
    framework: str,
    run_id: str,
    model_id: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> StoredModelRef:
    """Persist a live model object and return a JSON-safe reference.

    Supported:
    - sklearn objects via joblib
    - generic torch modules/state dictionaries via torch.save
    - torch image classifiers via architecture + state_dict + class metadata
    """

    framework = str(framework or "").lower()
    metadata = dict(metadata or {})

    image_sidecar = None
    if framework == "torch":
        try:
            image_sidecar = _load_sibling_module("image_sidecar")
        except Exception:
            image_sidecar = None

    if (
        framework == "torch"
        and image_sidecar is not None
        and image_sidecar.is_torch_image_bundle(model, metadata)
    ):
        filename = f"{run_id}-{_safe_filename(model_id)}-{uuid.uuid4().hex[:8]}.pt"
        path = ml_storage_dir(context, kind="models") / filename
        image_metadata = image_sidecar.save_torch_image_sidecar_file(
            path=path,
            model=model,
            metadata=metadata,
        )
        combined_metadata = dict(metadata)
        combined_metadata.update(image_metadata)
        return StoredModelRef(
            storage="local_file",
            uri=str(path),
            format=image_sidecar.TORCH_IMAGE_CLASSIFIER_FORMAT,
            framework=framework,
            created_at=time.time(),
            metadata=json_safe(combined_metadata),
        )

    suffix = "joblib" if framework == "sklearn" else "pt"
    filename = f"{run_id}-{_safe_filename(model_id)}-{uuid.uuid4().hex[:8]}.{suffix}"
    path = ml_storage_dir(context, kind="models") / filename

    if framework == "sklearn":
        import joblib

        joblib.dump(model, path)
        fmt = "joblib"
        extra = {"python_type": f"{type(model).__module__}.{type(model).__name__}"}

    elif framework == "torch":
        import torch

        if hasattr(model, "state_dict") and callable(model.state_dict):
            payload = {
                "state_dict": model.state_dict(),
                "python_type": f"{type(model).__module__}.{type(model).__name__}",
            }
        else:
            payload = model

        torch.save(payload, path)
        fmt = "torch"
        extra = {"python_type": f"{type(model).__module__}.{type(model).__name__}"}

    else:
        raise ValueError(f"Cannot persist unsupported ML framework {framework!r}.")

    combined_metadata = dict(metadata)
    combined_metadata.update(extra)

    return StoredModelRef(
        storage="local_file",
        uri=str(path),
        format=fmt,
        framework=framework,
        created_at=time.time(),
        metadata=json_safe(combined_metadata),
    )



def load_model_sidecar(model_ref: Mapping[str, Any]) -> Any:
    """Load a persisted model sidecar referenced by an ml.model payload."""
    fmt = str(model_ref.get("format") or "").lower()

    # Older recipe artifacts used `path`; durable model refs use `uri`.
    uri = model_ref.get("uri") or model_ref.get("path")
    if not uri:
        raise ValueError("Model reference is missing `uri` or legacy `path`.")

    path = Path(str(uri)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model sidecar not found: {path}")

    if fmt == "joblib":
        import joblib

        return joblib.load(path)

    if fmt == "torch":
        import torch

        return torch.load(path, map_location="cpu")

    if fmt == "torch_image_classifier":
        image_sidecar = _load_sibling_module("image_sidecar")
        return image_sidecar.load_torch_image_sidecar_file(
            path=path,
            metadata=model_ref.get("metadata") or {},
            map_location="cpu",
        )

    if fmt in {"torch_checkpoint", "recipe_torch_checkpoint", "recipe_torch_image_checkpoint"}:
        return _load_recipe_torch_image_checkpoint(path=path, model_ref=model_ref)

    raise ValueError(f"Unsupported model sidecar format {fmt!r}.")

def _load_recipe_torch_image_checkpoint(*, path: Path, model_ref: Mapping[str, Any]) -> Dict[str, Any]:
    """Load image-classifier checkpoints written by the recipe system.

    These checkpoints are not the same as image_sidecar.TORCH_IMAGE_CLASSIFIER_FORMAT.
    They contain a raw state_dict plus recipe metadata.
    """
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Torch checkpoint {path} did not contain a mapping payload.")

    metadata = dict(model_ref.get("metadata") or {})

    architecture = str(
        checkpoint.get("architecture")
        or metadata.get("architecture")
        or "torchvision.resnet18"
    )

    custom_model_import = str(
        checkpoint.get("custom_model_import")
        or metadata.get("custom_model_import")
        or ""
    ).strip()

    class_names = [
        str(value)
        for value in (
            checkpoint.get("class_names")
            or metadata.get("class_names")
            or []
        )
    ]

    if not class_names:
        raise ValueError(f"Torch checkpoint {path} is missing class_names.")

    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Torch checkpoint {path} is missing state_dict.")

    model = _build_recipe_image_model(
        architecture=architecture,
        custom_model_import=custom_model_import,
        num_classes=len(class_names),
    )

    model.load_state_dict(state_dict)
    model.eval()

    transform = dict(
        checkpoint.get("transform")
        or metadata.get("transform")
        or metadata.get("normalization")
        or {}
    )

    normalization = {
        "mean": transform.get("mean") or [0.4914, 0.4822, 0.4465],
        "std": transform.get("std") or [0.2023, 0.1994, 0.2010],
    }

    image_size = int(
        transform.get("image_size")
        or metadata.get("image_size")
        or 32
    )

    return {
        "torch_model": model,
        "class_names": class_names,
        "image_size": image_size,
        "normalization": normalization,
        "architecture": architecture,
        "custom_model_import": custom_model_import,
        "checkpoint": {
            "path": str(path),
            "epoch": checkpoint.get("epoch"),
            "metrics": json_safe(checkpoint.get("metrics") or {}),
            "params": json_safe(checkpoint.get("params") or {}),
            "transform": json_safe(transform),
        },
    }

def _build_recipe_image_model(
    *,
    architecture: str,
    custom_model_import: str,
    num_classes: int,
) -> Any:
    """Rebuild the image model architectures supported by core_ml.recipes."""
    import importlib

    import torch.nn as nn
    from torchvision import models

    architecture = str(architecture or "torchvision.resnet18").strip()

    if architecture == "custom_import":
        if not custom_model_import:
            raise ValueError("custom_model_import is required for custom_import checkpoints.")

        module_name, object_name = custom_model_import.rsplit(".", 1)
        factory = getattr(importlib.import_module(module_name), object_name)

        try:
            return factory(num_classes=num_classes)
        except TypeError:
            return factory()

    if architecture in {"torchvision.resnet18", "resnet18"}:
        model = models.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
        return model

    if architecture in {"torchvision.resnet34", "resnet34"}:
        model = models.resnet34(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
        return model

    if architecture in {"torchvision.mobilenet_v3_small", "mobilenet_v3_small"}:
        return models.mobilenet_v3_small(weights=None, num_classes=num_classes)

    raise ValueError(f"Unsupported recipe image architecture: {architecture!r}")


def normalize_model_artifact_payload(
    *,
    context: Any,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Convert an ml.model payload into the durable JSON-safe contract.

    Existing prototype training code may put a live Python model in
    payload["model"]. This function persists that object and replaces it with a
    `model_ref` dictionary. Already-normalized payloads pass through unchanged.
    """

    mutable = dict(payload)
    mutable.setdefault("schema_version", ML_ARTIFACT_SCHEMA_VERSION)
    mutable.setdefault("artifact_type", ARTIFACTS.MODEL)

    if "model_ref" in mutable and "model" not in mutable:
        return ensure_json_safe(mutable)

    model = mutable.pop("model", None)
    if model is None:
        return ensure_json_safe(mutable)

    framework = str(mutable.get("framework") or "").lower()
    run_id = str(mutable.get("run_id") or uuid.uuid4().hex)
    model_id = str(mutable.get("model_id") or mutable.get("model_title") or "model")
    metadata = dict(mutable.get("metadata") or {})

    ref = save_model_sidecar(
        context=context,
        model=model,
        framework=framework,
        run_id=run_id,
        model_id=model_id,
        metadata=metadata,
    )

    mutable["model_ref"] = ref.to_dict()
    mutable["model_object_stored"] = False
    mutable["updated_at"] = time.time()

    return ensure_json_safe(mutable)


def persist_existing_model_artifact(*, context: Any, artifact_id: str) -> Dict[str, Any]:
    """Normalize an existing ml.model artifact in-place where possible."""

    payload = context.artifacts.get(artifact_id)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Artifact {artifact_id!r} does not contain a model payload object.")

    normalized = normalize_model_artifact_payload(context=context, payload=payload)

    # ArtifactStore.get returns the live in-memory dict for normal payloads. Mutate
    # it when possible so existing refs keep working without a new artifact id.
    if isinstance(payload, dict):
        payload.clear()
        payload.update(normalized)

    return normalized


def load_model_from_payload(payload: Mapping[str, Any]) -> Any:
    """Load a model from either the new durable payload or an old in-memory payload."""

    if "model" in payload:
        return payload["model"]

    if "model_ref" in payload:
        return load_model_sidecar(payload["model_ref"])

    raise ValueError("ml.model payload has neither `model` nor `model_ref`.")


def prediction_payload(
    *,
    run_id: str,
    dataset_id: str,
    model_artifact_id: str,
    model_payload: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    row_ids: Optional[Iterable[Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a JSON-safe ml.predictions payload."""

    return ensure_json_safe(
        {
            "artifact_type": ARTIFACTS.PREDICTIONS,
            "schema_version": ML_ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "dataset_id": dataset_id,
            "model_artifact_id": model_artifact_id,
            "model_id": model_payload.get("model_id"),
            "model_title": model_payload.get("model_title"),
            "framework": model_payload.get("framework"),
            "task": model_payload.get("task"),
            "target_column": model_payload.get("target_column"),
            "feature_columns": list(model_payload.get("feature_columns") or []),
            "row_ids": list(row_ids or []),
            "records": list(records),
            "params": dict(params or {}),
            "created_at": time.time(),
        }
    )


def active_learning_batch_payload(
    *,
    dataset_id: str,
    predictions_artifact_id: str,
    strategy: str,
    records: Iterable[Mapping[str, Any]],
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a JSON-safe ml.active_learning_batch payload."""

    rows = list(records)

    return ensure_json_safe(
        {
            "artifact_type": ARTIFACTS.ACTIVE_LEARNING_BATCH,
            "schema_version": ML_ARTIFACT_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "predictions_artifact_id": predictions_artifact_id,
            "strategy": strategy,
            "count": len(rows),
            "row_ids": [row.get("row_id") for row in rows if row.get("row_id") is not None],
            "records": rows,
            "params": dict(params or {}),
            "created_at": time.time(),
        }
    )


def _safe_filename(value: str) -> str:
    safe = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "model"