from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import importlib.util
import sys

ML_ARTIFACT_SCHEMA_VERSION = 1

DEFAULT_INLINE_PREDICTION_ROWS = 1000
DEFAULT_PREDICTION_PREVIEW_ROWS = 25

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
class StoredPredictionRef:
    """JSON-safe pointer to a durable prediction table sidecar."""

    storage: str
    uri: str
    format: str
    created_at: float
    row_count: int
    columns: List[str]
    sha256: str
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class StoredEmbeddingIndexRef:
    """Optional indexed representation for global embedding acquisition."""

    storage: str
    uri: str
    format: str
    created_at: float
    table: Optional[str] = None
    record_id_column: str = "row_id"
    embedding_columns: List[str] = field(default_factory=list)
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: "StoredEmbeddingIndexRef | Mapping[str, Any]") -> "StoredEmbeddingIndexRef":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("embedding index reference must be a mapping")
        return cls(
            storage=str(value.get("storage") or "local_file"),
            uri=str(value.get("uri") or ""),
            format=str(value.get("format") or "duckdb"),
            created_at=float(value.get("created_at") or 0.0),
            table=(None if value.get("table") in (None, "") else str(value.get("table"))),
            record_id_column=str(value.get("record_id_column") or "row_id"),
            embedding_columns=embedding_columns,
            sha256=(None if value.get("sha256") in (None, "") else str(value.get("sha256"))),
            size_bytes=(None if value.get("size_bytes") in (None, "") else int(value.get("size_bytes"))),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class StoredEmbeddingRef:
    """JSON-safe pointer to a partitioned embedding table sidecar.

    The canonical JSONL representation stores one ``embedding`` list per row.
    The optional Parquet mirror stores fixed, ordered scalar columns named by
    ``embedding_columns`` so DuckDB can scan and anti-join parts globally.
    """

    schema_version: int
    storage: str
    uri: str
    format: str
    created_at: float
    dataset_id: str
    model_artifact_id: str
    row_count: int
    dimensions: int
    dtype: str
    record_id_column: str
    embedding_column: str
    embedding_columns: List[str]
    columns: List[str]
    sha256: str
    size_bytes: int
    parts: List[str]
    parquet_uri: Optional[str] = None
    parquet_parts: List[str] = field(default_factory=list)
    parquet_sha256: Optional[str] = None
    parquet_size_bytes: Optional[int] = None
    index_ref: Optional[StoredEmbeddingIndexRef] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["index_ref"] = None if self.index_ref is None else self.index_ref.to_dict()
        return payload

    @classmethod
    def from_value(cls, value: "StoredEmbeddingRef | Mapping[str, Any]") -> "StoredEmbeddingRef":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("embedding reference must be a mapping")
        index_raw = value.get("index_ref")
        index_ref = (
            StoredEmbeddingIndexRef.from_value(index_raw)
            if isinstance(index_raw, Mapping)
            else None
        )
        embedding_columns = [
            str(column) for column in value.get("embedding_columns") or []
        ]
        dimensions = int(value.get("dimensions") or len(embedding_columns))
        if dimensions <= 0:
            raise ValueError("embedding reference dimensions must be greater than zero")
        if embedding_columns and len(embedding_columns) != dimensions:
            raise ValueError(
                "embedding_columns length does not match embedding dimensions: "
                f"{len(embedding_columns)} != {dimensions}"
            )
        row_count = int(value.get("row_count") or 0)
        if row_count < 0:
            raise ValueError("embedding reference row_count must be zero or greater")
        uri = str(value.get("uri") or "")
        if not uri:
            raise ValueError("embedding reference uri is required")
        parts = [str(path) for path in value.get("parts") or []]
        if not parts:
            parts = [uri]
        return cls(
            schema_version=int(value.get("schema_version") or 1),
            storage=str(value.get("storage") or "local_file"),
            uri=uri,
            format=str(value.get("format") or "jsonl.gz"),
            created_at=float(value.get("created_at") or 0.0),
            dataset_id=str(value.get("dataset_id") or ""),
            model_artifact_id=str(value.get("model_artifact_id") or ""),
            row_count=row_count,
            dimensions=dimensions,
            dtype=str(value.get("dtype") or "float32"),
            record_id_column=str(value.get("record_id_column") or "row_id"),
            embedding_column=str(value.get("embedding_column") or "embedding"),
            embedding_columns=[str(column) for column in value.get("embedding_columns") or []],
            columns=[str(column) for column in value.get("columns") or []],
            sha256=str(value.get("sha256") or ""),
            size_bytes=int(value.get("size_bytes") or 0),
            parts=parts,
            parquet_uri=(None if value.get("parquet_uri") in (None, "") else str(value.get("parquet_uri"))),
            parquet_parts=[str(path) for path in value.get("parquet_parts") or []],
            parquet_sha256=(None if value.get("parquet_sha256") in (None, "") else str(value.get("parquet_sha256"))),
            parquet_size_bytes=(None if value.get("parquet_size_bytes") in (None, "") else int(value.get("parquet_size_bytes"))),
            index_ref=index_ref,
            metadata=dict(value.get("metadata") or {}),
        )


def embedding_ref_from_payload(payload: Mapping[str, Any]) -> Optional[StoredEmbeddingRef]:
    """Resolve the standard embedding reference from an artifact-like payload."""

    candidates = [
        payload.get("embedding_ref"),
        payload.get("embeddings_ref"),
        payload.get("feature_embedding_ref"),
    ]
    table = payload.get("embedding_table")
    if isinstance(table, Mapping):
        candidates.extend([table.get("storage"), table.get("embedding_ref")])
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not candidate.get("uri"):
            continue
        try:
            return StoredEmbeddingRef.from_value(candidate)
        except Exception:
            continue
    return None


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
    EMBEDDINGS: str = "ml.embeddings"
    TRAINING_LOG: str = "ml.training_log"
    RUN: str = "ml.run"
    RESUME_CHECKPOINT: str = "ml.resume_checkpoint"
    ACTIVE_LEARNING_BATCH: str = "ml.active_learning_batch"

ARTIFACTS = MLArtifactContract()

from .serialization import ensure_json_object, json_safe

def ensure_json_safe(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe shallow/deep copy of an artifact payload."""

    return ensure_json_object(payload, schema_version=ML_ARTIFACT_SCHEMA_VERSION)

def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _torch_load(path: Path, *, map_location: str = "cpu") -> Any:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def verify_file_ref(model_ref: Mapping[str, Any], path: Path) -> None:
    metadata = dict(model_ref.get("metadata") or {})
    expected = str(model_ref.get("sha256") or metadata.get("sha256") or "").strip()
    if expected:
        actual = file_sha256(path)
        if actual.lower() != expected.lower():
            raise ValueError(
                f"Model sidecar checksum mismatch for {path}. "
                f"Expected {expected}, got {actual}."
            )

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
            from . import image_sidecar
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
        combined_metadata.update({"sha256": file_sha256(path), "size_bytes": path.stat().st_size})
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

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            joblib.dump(model, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        fmt = "joblib"
        extra = {"python_type": f"{type(model).__module__}.{type(model).__name__}"}

    elif framework == "torch":
        import torch

        # Managed recipes use reconstructable state-dict checkpoints in their
        # harnesses. This compatibility path handles older artifacts containing a
        # live torch module. Persist the complete object because a bare state_dict
        # is not reloadable without constructor arguments.
        is_module = hasattr(model, "state_dict") and callable(model.state_dict)
        payload = model
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        fmt = "torch_module" if is_module else "torch_object"
        extra = {"python_type": f"{type(model).__module__}.{type(model).__name__}"}

    else:
        raise ValueError(f"Cannot persist unsupported ML framework {framework!r}.")

    combined_metadata = dict(metadata)
    combined_metadata.update(extra)
    combined_metadata.update({"sha256": file_sha256(path), "size_bytes": path.stat().st_size})

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

    verify_file_ref(model_ref, path)

    if fmt == "joblib":
        import joblib

        return joblib.load(path)

    if fmt == "sklearn_resume_checkpoint":
        import joblib

        checkpoint = joblib.load(path)
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"Sklearn resume checkpoint {path} did not contain a mapping payload.")
        prediction_model = checkpoint.get("prediction_model")
        if prediction_model is None:
            raise ValueError(f"Sklearn resume checkpoint {path} is missing prediction_model.")
        return prediction_model

    if fmt == "torch_resume_checkpoint":
        checkpoint = _torch_load(path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"Torch resume checkpoint {path} did not contain a mapping payload.")
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if state_dict is None:
            raise ValueError(f"Torch resume checkpoint {path} is missing model_state_dict.")
        normalized = dict(checkpoint)
        normalized.setdefault("state_dict", state_dict)
        return {
            "checkpoint_payload": normalized,
            "state_dict": state_dict,
            "checkpoint_path": str(path),
            "checkpoint_sha256": file_sha256(path),
            "metadata": json_safe(model_ref.get("metadata") or {}),
        }

    if fmt in {"torch", "torch_module", "torch_object"}:
        loaded = _torch_load(path, map_location="cpu")
        # Compatibility with the short-lived state-dict wrapper format.
        if fmt == "torch" and isinstance(loaded, Mapping) and set(loaded) == {"state_dict", "python_type"}:
            return dict(loaded)
        return loaded

    if fmt == "torch_image_classifier":
        from . import image_sidecar

        return image_sidecar.load_torch_image_sidecar_file(
            path=path,
            metadata=model_ref.get("metadata") or {},
            map_location="cpu",
        )

    if fmt in {"torch_checkpoint", "recipe_torch_checkpoint"}:
        checkpoint = _torch_load(path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"Torch checkpoint {path} did not contain a mapping payload.")
        if checkpoint.get("state_dict") is None:
            raise ValueError(f"Torch checkpoint {path} is missing state_dict.")
        return {
            "checkpoint_payload": dict(checkpoint),
            "state_dict": checkpoint.get("state_dict"),
            "checkpoint_path": str(path),
            "checkpoint_sha256": file_sha256(path),
            "metadata": json_safe(model_ref.get("metadata") or {}),
        }

    if fmt == "recipe_torch_image_checkpoint":
        return _load_recipe_torch_image_checkpoint(path=path, model_ref=model_ref)

    raise ValueError(f"Unsupported model sidecar format {fmt!r}.")

def _load_recipe_torch_image_checkpoint(*, path: Path, model_ref: Mapping[str, Any]) -> Dict[str, Any]:
    """Load image-classifier checkpoints written by the recipe system.

    These checkpoints are not the same as image_sidecar.TORCH_IMAGE_CLASSIFIER_FORMAT.
    They contain a raw state_dict plus recipe metadata.
    """
    import torch

    checkpoint = _torch_load(path, map_location="cpu")
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

def save_predictions_sidecar(
    *,
    context: Any,
    run_id: str,
    dataset_id: str,
    model_artifact_id: str,
    rows: Sequence[Mapping[str, Any]],
    params: Optional[Mapping[str, Any]] = None,
) -> StoredPredictionRef:
    """Atomically persist row-keyed predictions as compressed JSON Lines."""
    from .paths import ml_artifact_root

    params = dict(params or {})
    explicit = params.get("prediction_output_dir")
    if explicit:
        root = Path(str(explicit)).expanduser()
    else:
        root = ml_artifact_root(context, params) / "predictions"

    path = (
        root
        / _safe_filename(dataset_id)
        / _safe_filename(model_artifact_id)
        / _safe_filename(run_id)
        / "predictions.jsonl.gz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    clean_rows = [json_safe(dict(row)) for row in rows]
    columns = list(dict.fromkeys(key for row in clean_rows for key in row.keys()))

    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8", newline="\n") as handle:
            for row in clean_rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    return StoredPredictionRef(
        storage="local_file",
        uri=str(path),
        format="jsonl.gz",
        created_at=time.time(),
        row_count=len(clean_rows),
        columns=columns,
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
    )

def load_predictions_sidecar(prediction_ref: Mapping[str, Any]) -> List[Dict[str, Any]]:
    uri = prediction_ref.get("uri") or prediction_ref.get("path")
    if not uri:
        raise ValueError("Prediction reference is missing `uri` or legacy `path`.")
    path = Path(str(uri)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Prediction sidecar not found: {path}")

    expected = str(prediction_ref.get("sha256") or "").strip()
    if expected:
        actual = file_sha256(path)
        if actual.lower() != expected.lower():
            raise ValueError(
                f"Prediction sidecar checksum mismatch for {path}. "
                f"Expected {expected}, got {actual}."
            )

    fmt = str(prediction_ref.get("format") or "jsonl.gz").lower()
    if fmt != "jsonl.gz":
        raise ValueError(f"Unsupported prediction sidecar format {fmt!r}.")

    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, Mapping):
                    rows.append(dict(value))
    return rows

def compact_predictions_payload(
    payload: Mapping[str, Any],
    *,
    prediction_ref: Mapping[str, Any],
    inline_limit: int = DEFAULT_INLINE_PREDICTION_ROWS,
    preview_limit: int = DEFAULT_PREDICTION_PREVIEW_ROWS,
) -> Dict[str, Any]:
    """Replace large inline prediction rows with a durable sidecar."""
    mutable = dict(payload)
    table = dict(mutable.get("prediction_table") or {})

    table_rows = [
        dict(row)
        for row in table.get("rows") or []
    ]
    records = [
        dict(row)
        for row in mutable.get("records") or []
    ]

    inline_limit = max(0, int(inline_limit))
    preview_limit = max(0, int(preview_limit))

    mutable["prediction_ref"] = dict(prediction_ref)
    mutable["row_count"] = len(table_rows)
    table["row_count"] = len(table_rows)

    if len(table_rows) > inline_limit:
        mutable["records_preview"] = records[:preview_limit]
        mutable["records"] = []

        table["preview"] = table_rows[:preview_limit]
        table["rows"] = []
        table["inline_complete"] = False
        table["storage"] = "prediction_ref"
    else:
        table["inline_complete"] = True
        table["storage"] = "inline_and_prediction_ref"

    mutable["prediction_table"] = table
    return ensure_json_safe(mutable)

def prediction_rows_from_payload(
    payload: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Return complete prediction rows from inline storage or sidecar."""
    table = dict(payload.get("prediction_table") or {})
    table_rows = [
        dict(row)
        for row in table.get("rows") or []
    ]

    if table_rows and bool(table.get("inline_complete", True)):
        return table_rows

    records = [
        dict(row)
        for row in payload.get("records") or []
    ]

    if records and bool(payload.get("records_inline_complete", True)):
        return records

    prediction_ref = payload.get("prediction_ref")
    if isinstance(prediction_ref, Mapping):
        return load_predictions_sidecar(prediction_ref)

    return table_rows or records

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

def _safe_filename(value: str) -> str:
    safe = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "model"
