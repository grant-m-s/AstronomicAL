from __future__ import annotations

import json
import math
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


JSONDict = Dict[str, Any]


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


def publish(context: Any, event_type: str, payload: Optional[Mapping[str, Any]] = None) -> None:
    events = getattr(context, "events", None)
    if events is None:
        return

    payload_dict = dict(payload or {})
    for method_name in ("publish", "emit", "trigger"):
        method = getattr(events, method_name, None)
        if callable(method):
            try:
                method(event_type, payload_dict)
                return
            except TypeError:
                try:
                    method({"type": event_type, "payload": payload_dict})
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
) -> Optional[str]:
    artifacts = getattr(context, "artifacts", None)
    put = getattr(artifacts, "put", None)
    if not callable(put):
        return None

    safe_payload = json_safe(dict(payload))

    try:
        return put(
            artifact_type,
            safe_payload,
            dataset_id=dataset_id,
            row_ids=list(row_ids or []),
            params=dict(params or {}),
        )
    except TypeError:
        try:
            return put(artifact_type, safe_payload)
        except Exception:
            return None
    except Exception:
        return None


def get_dataset_frame(
    context: Any,
    dataset_id: str,
    *,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
):
    """Return a pandas DataFrame through the DatasetSource path when available.

    This is intentionally a compatibility helper. Recipe authors should use
    DatasetSource directly when they need streaming/lazy access.
    """
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        raise ValueError("context.datasets is not available.")

    # Preferred plugin-system path.
    get_source = getattr(datasets, "get_source", None)
    if callable(get_source):
        source = get_source(dataset_id)
        to_pandas = getattr(source, "to_pandas", None)
        if callable(to_pandas):
            try:
                return to_pandas(columns=list(columns) if columns else None, limit=limit)
            except TypeError:
                try:
                    return to_pandas(columns=list(columns) if columns else None)
                except TypeError:
                    return to_pandas()

        head = getattr(source, "head", None)
        if callable(head) and limit is not None:
            return head(limit)

    # Compatibility path.
    for method_name in ("get_df", "dataframe", "df"):
        method = getattr(datasets, method_name, None)
        if callable(method):
            df = method(dataset_id)
            if columns:
                df = df[list(columns)]
            if limit is not None:
                df = df.head(limit)
            return df

    # Some registry implementations expose dataset objects.
    get = getattr(datasets, "get", None)
    if callable(get):
        dataset = get(dataset_id)
        df = getattr(dataset, "df", None)
        if df is not None:
            if columns:
                df = df[list(columns)]
            if limit is not None:
                df = df.head(limit)
            return df

    raise ValueError(f"Could not materialise dataset `{dataset_id}` as a DataFrame.")


def mapped_column(context: Any, dataset_id: str, mapping: str) -> Optional[str]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None

    for method_name in ("get_mapping", "mapping", "get_column_mapping"):
        method = getattr(datasets, method_name, None)
        if not callable(method):
            continue
        try:
            value = method(dataset_id, mapping)
        except Exception:
            value = None
        if value:
            return str(value)

    return None

def list_dataset_columns(context: Any, dataset_id: str) -> List[str]:
    """Best-effort column discovery for active DatasetSource/DataFrame datasets."""
    datasets = getattr(context, "datasets", None)
    if datasets is None or not dataset_id:
        return []

    # Preferred dataset service APIs.
    for method_name in ("list_columns", "columns", "get_columns"):
        method = getattr(datasets, method_name, None)
        if callable(method):
            try:
                values = method(dataset_id)
                if values:
                    return [str(v) for v in values]
            except Exception:
                pass

    # DatasetSource path.
    get_source = getattr(datasets, "get_source", None)
    if callable(get_source):
        try:
            source = get_source(dataset_id)
        except Exception:
            source = None

        if source is not None:
            for method_name in ("columns", "list_columns", "get_columns"):
                method = getattr(source, method_name, None)
                if callable(method):
                    try:
                        values = method()
                        if values:
                            return [str(v) for v in values]
                    except Exception:
                        pass

            to_pandas = getattr(source, "to_pandas", None)
            if callable(to_pandas):
                try:
                    frame = to_pandas(limit=0)
                    return [str(c) for c in frame.columns]
                except Exception:
                    pass

            head = getattr(source, "head", None)
            if callable(head):
                try:
                    frame = head(0)
                    return [str(c) for c in frame.columns]
                except Exception:
                    pass

    # Compatibility DataFrame path.
    try:
        frame = get_dataset_frame(context, dataset_id, limit=0)
        return [str(c) for c in frame.columns]
    except Exception:
        return []


def _normalise_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def _choose_column(
    columns: Sequence[str],
    *,
    exact: Sequence[str] = (),
    contains: Sequence[str] = (),
    excludes: Sequence[str] = (),
) -> Optional[str]:
    if not columns:
        return None

    normalised = {column: _normalise_column_name(column) for column in columns}
    exact_set = {_normalise_column_name(value) for value in exact}
    contains_values = [_normalise_column_name(value) for value in contains]
    exclude_values = [_normalise_column_name(value) for value in excludes]

    for column, norm in normalised.items():
        if norm in exact_set:
            return column

    for wanted in contains_values:
        for column, norm in normalised.items():
            if wanted in norm and not any(excluded in norm for excluded in exclude_values):
                return column

    return None


def infer_column_bindings(context: Any, dataset_id: str) -> Dict[str, Any]:
    """Infer common ML column bindings from mappings and column-name heuristics.

    Mappings win. Column-name heuristics are only used when semantic mappings
    are missing.
    """
    columns = list_dataset_columns(context, dataset_id)

    record_id = (
        mapped_column(context, dataset_id, "record_id")
        or mapped_column(context, dataset_id, "id")
        or _choose_column(
            columns,
            exact=[
                "record_id",
                "object_id",
                "source_id",
                "target_id",
                "id",
                "uid",
                "uuid",
                "index",
                "idx",
            ],
            contains=[
                "record_id",
                "object_id",
                "source_id",
                "target_id",
            ],
        )
    )

    image_column = (
        mapped_column(context, dataset_id, "image.path")
        or mapped_column(context, dataset_id, "image.uri")
        or mapped_column(context, dataset_id, "image")
        or _choose_column(
            columns,
            exact=[
                "image",
                "image_path",
                "image_uri",
                "img",
                "img_path",
                "path",
                "file_path",
                "filepath",
                "filename",
                "uri",
                "url",
            ],
            contains=[
                "image_path",
                "image_uri",
                "image",
                "img_path",
                "filepath",
                "file_path",
                "filename",
                "jpg",
                "jpeg",
                "png",
                "fits",
                "uri",
                "url",
            ],
            excludes=[
                "label",
                "target",
                "class",
                "mask",
                "size",
                "bytes",
                "width",
                "height",
            ],
        )
    )

    target_column = (
        mapped_column(context, dataset_id, "target_label")
        or mapped_column(context, dataset_id, "label")
        or mapped_column(context, dataset_id, "target")
        or mapped_column(context, dataset_id, "class")
        or _choose_column(
            columns,
            exact=[
                "target_label",
                "label",
                "labels",
                "target",
                "class",
                "class_label",
                "category",
                "y",
                "truth",
                "annotation",
                "review_label",
            ],
            contains=[
                "target",
                "label",
                "class",
                "category",
                "truth",
                "annotation",
                "review",
            ],
            excludes=[
                "prediction",
                "predicted",
                "probability",
                "confidence",
                "uncertainty",
                "image",
                "path",
                "uri",
            ],
        )
    )

    mask_column = (
        mapped_column(context, dataset_id, "mask.path")
        or mapped_column(context, dataset_id, "mask.uri")
        or mapped_column(context, dataset_id, "mask")
        or _choose_column(
            columns,
            exact=[
                "mask",
                "mask_path",
                "mask_uri",
                "segmentation_mask",
                "segmentation_mask_path",
            ],
            contains=[
                "mask",
                "segmentation",
            ],
            excludes=[
                "image",
                "label",
                "target",
                "class",
            ],
        )
    )

    excluded = {
        value
        for value in [record_id, image_column, target_column, mask_column]
        if value
    }

    feature_columns = [
        column
        for column in columns
        if column not in excluded
        and not _normalise_column_name(column).startswith("__")
        and not any(
            bad in _normalise_column_name(column)
            for bad in [
                "prediction",
                "probability",
                "confidence",
                "uncertainty",
                "uri",
                "url",
                "path",
                "filename",
            ]
        )
    ]

    inferred: Dict[str, Any] = {
        "available_columns": columns,
        "feature_columns": feature_columns,
    }

    if record_id:
        inferred["record_id_column"] = record_id
    if image_column:
        inferred["image_column"] = image_column
    if target_column:
        inferred["target_column"] = target_column
    if mask_column:
        inferred["mask_column"] = mask_column

    return inferred


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

def check_cancelled(cancel_token: Any) -> None:
    if cancel_token is None:
        return

    for attr in ("cancelled", "is_cancelled", "cancel_requested", "is_set"):
        value = getattr(cancel_token, attr, None)
        try:
            cancelled = bool(value()) if callable(value) else bool(value)
        except Exception:
            cancelled = False

        if cancelled:
            reason = getattr(cancel_token, "reason", None) or "ML recipe run was cancelled."
            raise MLRecipeCancelled(str(reason))


@dataclass
class MLRecipeSpec:
    id: str
    title: str
    version: str
    task: str
    modality: str
    description: str
    recipe_cls: type
    params_schema: Dict[str, Any] = field(default_factory=dict)
    required_mappings: List[str] = field(default_factory=list)
    optional_mappings: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    complexity: str = "advanced"
    author: str = "unknown"
    notes: str = ""


class MLRunLogger:
    """Small artifact-backed logger for long recipe runs.

    The ML Training Curves panel expects epoch metrics to be flat columns, e.g.

        epoch, train_loss, val_loss, train_accuracy, val_accuracy

    Recipes may still pass metrics={...}; this logger flattens those metrics
    into each epoch row while preserving the nested metrics dict for debugging.
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
    ) -> None:
        self.context = context
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.training_log_artifact_id = training_log_artifact_id
        self.recipe_spec = recipe_spec
        self.params = dict(params or {})
        self.started_at = time.time()
        self.rows: List[Dict[str, Any]] = []

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
        now = time.time()

        metrics_dict = json_safe(dict(metrics or {}))
        extra_dict = json_safe(dict(extra or {}))

        row: Dict[str, Any] = {
            "time": now,
            "elapsed_seconds": now - self.started_at,
            "status": status,
            "message": message,
            "step": step,
            "total": total,
            "metrics": metrics_dict,
        }

        # Flatten metrics so MLTrainingCurvesPanel can plot them directly.
        if isinstance(metrics_dict, Mapping):
            for key, value in metrics_dict.items():
                row[str(key)] = value

        if isinstance(extra_dict, Mapping):
            for key, value in extra_dict.items():
                row[str(key)] = value

        # The curves panel uses an `epoch` column. If a recipe only supplied
        # step/total, promote step to epoch when it looks epoch-like.
        if row.get("epoch") is None and step is not None:
            row["epoch"] = step

        self.rows.append(row)

        payload = self._payload(status=status, message=message)
        self._update_training_log(payload)

        publish(
            self.context,
            "ml.recipe_run.progress",
            {
                "run_id": self.run_id,
                "dataset_id": self.dataset_id,
                "artifact_id": self.training_log_artifact_id,
                "training_log_artifact_id": self.training_log_artifact_id,
                "status": status,
                "message": message,
                "step": step,
                "total": total,
                "metrics": metrics_dict,
            },
        )

        # The existing ML Training Curves panel subscribes to these events.
        publish(
            self.context,
            "ml.training_log.updated",
            {
                "run_id": self.run_id,
                "dataset_id": self.dataset_id,
                "artifact_id": self.training_log_artifact_id,
                "training_log_artifact_id": self.training_log_artifact_id,
                "status": status,
                "message": message,
            },
        )

    def metric(self, *, epoch: int, split: str, metrics: Mapping[str, Any]) -> None:
        prefixed = {f"{split}_{k}": v for k, v in dict(metrics).items()}
        prefixed["epoch"] = epoch
        self.log(
            message=f"{split} metrics for epoch {epoch}.",
            step=epoch,
            metrics=prefixed,
            extra={"epoch": epoch, "split": split},
        )

    def finish(self, *, status: str = "complete", message: str = "Recipe run complete.") -> None:
        self.log(message=message, status=status)

    def _payload(self, *, status: str, message: str) -> Dict[str, Any]:
        epochs = self._epoch_rows()
        optimize_metric = self._optimize_metric(epochs)
        best_epoch = self._best_epoch(epochs, optimize_metric)

        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "recipe_id": getattr(self.recipe_spec, "id", self.params.get("recipe_id", "")),
            "recipe_version": getattr(
                self.recipe_spec,
                "version",
                self.params.get("recipe_version", ""),
            ),
            "model_title": self._model_title(),
            "framework": self._framework(),
            "task": getattr(self.recipe_spec, "task", self.params.get("task", "")),
            "modality": getattr(self.recipe_spec, "modality", self.params.get("modality", "")),
            "status": status,
            "message": message,
            "optimize_metric": optimize_metric,
            "best_epoch": best_epoch,
            "started_at": self.started_at,
            "updated_at": time.time(),
            "events": json_safe(self.rows),
            "epochs": json_safe(epochs),
        }

        return payload

    def _epoch_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for row in self.rows:
            if row.get("epoch") is None and row.get("step") is None:
                continue

            flat = dict(row)

            metrics = flat.get("metrics")
            if isinstance(metrics, Mapping):
                for key, value in metrics.items():
                    flat.setdefault(str(key), value)

            if flat.get("epoch") is None and flat.get("step") is not None:
                flat["epoch"] = flat.get("step")

            rows.append(flat)

        return rows

    def _model_title(self) -> str:
        for key in ("model_title", "model_id", "architecture", "recipe_title"):
            value = self.params.get(key)
            if value:
                return str(value)

        title = getattr(self.recipe_spec, "title", None)
        if title:
            return str(title)

        recipe_id = getattr(self.recipe_spec, "id", None) or self.params.get("recipe_id")
        if recipe_id:
            return str(recipe_id)

        return "unknown"

    def _framework(self) -> str:
        value = self.params.get("framework")
        if value:
            return str(value)

        tags = set(str(tag).lower() for tag in getattr(self.recipe_spec, "tags", []) or [])
        recipe_id = str(getattr(self.recipe_spec, "id", "")).lower()

        if "torch" in tags or "pytorch" in tags or "torch" in recipe_id:
            return "torch"

        if "sklearn" in tags or "scikit-learn" in tags or "sklearn" in recipe_id:
            return "sklearn"

        if "tensorflow" in tags or "keras" in tags:
            return "tensorflow"

        return ""

    def _optimize_metric(self, epochs: Sequence[Mapping[str, Any]]) -> str:
        value = self.params.get("optimize_metric")
        if value:
            return str(value)

        # Prefer validation accuracy for classification recipes.
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

    def _best_epoch(
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

    def _update_training_log(self, payload: Mapping[str, Any]) -> None:
        if not self.training_log_artifact_id:
            return

        try:
            existing = self.context.artifacts.get(self.training_log_artifact_id)
        except Exception:
            existing = None

        if isinstance(existing, dict):
            existing.update(json_safe(dict(payload)))


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


class MLRecipe:
    """Base class for code-backed ML recipes.

    Recipe implementations can use arbitrary Python/PyTorch/sklearn code. The
    platform only needs the declared metadata and the run() contract.
    """

    id = "recipe.unset"
    title = "Unset recipe"
    version = "0.1.0"
    task = "classification"
    modality = "tabular"
    description = ""
    params_schema: Dict[str, Any] = {"type": "object", "properties": {}}
    required_mappings: List[str] = []
    optional_mappings: List[str] = []
    produces: List[str] = []
    tags: List[str] = []
    complexity = "advanced"
    author = "unknown"
    notes = ""

    @classmethod
    def spec(cls) -> MLRecipeSpec:
        return MLRecipeSpec(
            id=str(cls.id),
            title=str(cls.title),
            version=str(cls.version),
            task=str(cls.task),
            modality=str(cls.modality),
            description=str(cls.description),
            recipe_cls=cls,
            params_schema=dict(cls.params_schema or {}),
            required_mappings=list(cls.required_mappings or []),
            optional_mappings=list(cls.optional_mappings or []),
            produces=list(cls.produces or []),
            tags=list(cls.tags or []),
            complexity=str(cls.complexity),
            author=str(cls.author),
            notes=str(cls.notes),
        )

    def run(self, run: MLRunContext) -> Dict[str, Any]:
        raise NotImplementedError


class MLRecipeRegistry:
    def __init__(self) -> None:
        self._recipes: Dict[str, MLRecipeSpec] = {}

    def register(self, recipe: type[MLRecipe] | MLRecipeSpec) -> MLRecipeSpec:
        spec = recipe if isinstance(recipe, MLRecipeSpec) else recipe.spec()
        if not spec.id:
            raise ValueError("Recipe id cannot be empty.")
        self._recipes[spec.id] = spec
        return spec

    def get(self, recipe_id: str) -> MLRecipeSpec:
        return self._recipes[str(recipe_id)]

    def list(
        self,
        *,
        task: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> List[MLRecipeSpec]:
        recipes = list(self._recipes.values())
        if task:
            recipes = [r for r in recipes if r.task == task]
        if modality:
            recipes = [r for r in recipes if r.modality == modality]
        return sorted(recipes, key=lambda r: (r.modality, r.task, r.title))

    def to_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": spec.id,
                "title": spec.title,
                "version": spec.version,
                "task": spec.task,
                "modality": spec.modality,
                "complexity": spec.complexity,
                "author": spec.author,
                "tags": ", ".join(spec.tags),
                "description": spec.description,
            }
            for spec in self.list()
        ]


def make_run_context(
    *,
    context: Any,
    dataset_id: str,
    recipe_spec: MLRecipeSpec,
    params: Mapping[str, Any],
    cancel_token: Any = None,
    run_id: Optional[str] = None,
) -> MLRunContext:
    run_id = run_id or uuid.uuid4().hex

    base = (
        Path(tempfile.gettempdir())
        / "astronomicAL"
        / "ml_recipe_runs"
        / str(run_id)
    )
    base.mkdir(parents=True, exist_ok=True)

    framework = str(params.get("framework") or "")
    if not framework:
        tags = set(str(tag).lower() for tag in getattr(recipe_spec, "tags", []) or [])
        recipe_id_lower = str(recipe_spec.id).lower()
        if "torch" in tags or "pytorch" in tags or "torch" in recipe_id_lower:
            framework = "torch"
        elif "sklearn" in tags or "scikit-learn" in tags or "sklearn" in recipe_id_lower:
            framework = "sklearn"

    model_title = (
        params.get("model_title")
        or params.get("model_id")
        or params.get("architecture")
        or recipe_spec.title
    )

    training_log_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "recipe_id": recipe_spec.id,
        "recipe_version": recipe_spec.version,
        "model_title": str(model_title or "unknown"),
        "framework": framework,
        "task": recipe_spec.task,
        "modality": recipe_spec.modality,
        "status": "created",
        "message": "Recipe run created.",
        "optimize_metric": str(params.get("optimize_metric") or ""),
        "best_epoch": None,
        "started_at": time.time(),
        "updated_at": time.time(),
        "events": [],
        "epochs": [],
    }

    training_log_artifact_id = put_artifact(
        context,
        "ml.training_log",
        training_log_payload,
        dataset_id=dataset_id,
        params=params,
    )

    publish(
        context,
        "ml.training_log.created",
        {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "artifact_id": training_log_artifact_id,
            "training_log_artifact_id": training_log_artifact_id,
            "recipe_id": recipe_spec.id,
        },
    )

    logger = MLRunLogger(
        context=context,
        run_id=run_id,
        dataset_id=dataset_id,
        training_log_artifact_id=training_log_artifact_id,
        recipe_spec=recipe_spec,
        params=params,
    )

    return MLRunContext(
        context=context,
        dataset_id=dataset_id,
        recipe_id=recipe_spec.id,
        recipe_version=recipe_spec.version,
        params=dict(params),
        run_id=run_id,
        work_dir=base,
        cancel_token=cancel_token,
        training_log_artifact_id=training_log_artifact_id,
        logger=logger,
    )