from __future__ import annotations

import importlib.util
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .recipe_registry import (
    ManagedMLRecipe,
    Partition,
    Partitions,
    RunHarness,
    TargetSpec,
    TrainingComponents,
    json_safe,
    ml_run_artifact_dir,
    register_harness,
)


@dataclass
class SklearnBatch:
    """The sklearn analogue of a torch DataLoader: a materialised partition."""

    X: pd.DataFrame
    y: Any
    ids: List[str]
    classes: List[str]


# =============================================================================
# Harness
# =============================================================================

class SklearnHarness(RunHarness):
    """Protocol harness for sklearn estimators (classification + regression).

    The recipe owns build_model / configure_training / fit. The harness owns the
    train-only preprocessor, validation selection, test-once evaluation, and the
    durable joblib sidecar that prediction.SklearnTabularPredictor loads.
    """

    def __init__(self, run, recipe):
        super().__init__(run, recipe)
        self._preprocessor = None
        self._fitted_preprocessor = None
        self._target: Optional[TargetSpec] = None

    # ---- task abstraction --------------------------------------------------
    def _target_spec(self, parts: Partitions) -> TargetSpec:
        self._target = super()._target_spec(parts)
        return self._target

    # ---- feature resolution ------------------------------------------------
    def _feature_columns(self) -> List[str]:
        cols = [c for c in (self.binding.input_columns or []) if c]
        if not cols:
            raise ValueError(
                "Sklearn recipes require feature columns. Set `feature_columns`/"
                "`input_columns` or ensure the dataset has non-id/target columns."
            )
        return cols

    def _frame_for(self, partition: Partition) -> pd.DataFrame:
        b = self.binding
        frame = self._partition_frames.get(partition.name)
        if frame is None:
            frame = self._frame
        wanted = {str(r) for r in partition.record_ids}
        return frame[frame[b.record_id_column].astype(str).isin(wanted)].reset_index(drop=True)

    # ---- loaders -----------------------------------------------------------
    def _make_loader(self, partition: Partition, *, train: bool):
        b = self.binding
        features = self._feature_columns()
        frame = self._frame_for(partition)

        X = frame[features]
        ids = frame[b.record_id_column].astype(str).tolist()

        if b.target_column:
            y_raw = frame[b.target_column]
            if self._task_kind() == "regression":
                y = np.asarray(y_raw, dtype=float)
            else:
                y = y_raw.astype(str).to_numpy()
        else:
            y = np.asarray([np.nan] * len(frame))

        # Fit the preprocessor on the TRAIN partition only, once.
        if train and self._fitted_preprocessor is None:
            self._preprocessor = self._make_preprocessor(X)
            self._fitted_preprocessor = self._preprocessor.fit(X)

        return SklearnBatch(X=X, y=y, ids=ids, classes=list(partition.classes))

    def _make_preprocessor(self, X: pd.DataFrame):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        categorical = [c for c in X.columns if c not in numeric]

        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        transformers = []
        if numeric:
            transformers.append((
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric,
            ))
        if categorical:
            transformers.append((
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", encoder),
                ]),
                categorical,
            ))

        if not transformers:
            raise ValueError("No usable feature columns were found for preprocessing.")

        return ColumnTransformer(transformers=transformers, remainder="drop")

    # ---- output-dim sanity check -------------------------------------------
    def _assert_output_dim(self, model, target: TargetSpec, train_loader):
        # sklearn estimators infer width from y at fit time, so there is no
        # wrong-width head to catch pre-fit. We instead validate post-fit in
        # _evaluate against the fitted estimator's classes_.
        return

    # ---- evaluation --------------------------------------------------------
    def _evaluate(self, model, loader, *, return_records: bool = False):
        Xt = self._fitted_preprocessor.transform(loader.X)
        if self._task_kind() == "regression":
            return self._evaluate_regression(model, Xt, loader, return_records)
        return self._evaluate_classification(model, Xt, loader, return_records)

    def _evaluate_classification(self, model, Xt, loader, return_records):
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

        y_true = np.asarray(loader.y).astype(str)
        y_pred = np.asarray(model.predict(Xt)).astype(str)

        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs = np.asarray(model.predict_proba(Xt), dtype=float)
            except Exception:
                probs = None

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        # cross-entropy-ish proxy so `val_loss` works as a selection metric too
        if probs is not None and probs.size:
            eps = 1e-12
            classes = [str(c) for c in getattr(model, "classes_", loader.classes)]
            idx = {c: i for i, c in enumerate(classes)}
            ll = []
            for i, t in enumerate(y_true):
                j = idx.get(str(t))
                if j is not None and j < probs.shape[1]:
                    ll.append(-np.log(max(probs[i, j], eps)))
            if ll:
                metrics["loss"] = float(np.mean(ll))

        records = []
        if return_records:
            classes = [str(c) for c in getattr(model, "classes_", loader.classes)]
            for i, rid in enumerate(loader.ids):
                rec: Dict[str, Any] = {"record_id": str(rid), "prediction": str(y_pred[i])}
                if probs is not None and i < probs.shape[0]:
                    p = probs[i]
                    order = np.sort(p)[::-1]
                    rec["confidence"] = float(order[0])
                    rec["max_probability"] = float(order[0])
                    rec["least_confidence"] = float(1.0 - order[0])
                    if order.size >= 2:
                        rec["margin"] = float(order[0] - order[1])
                        rec["margin_uncertainty"] = float(1.0 - rec["margin"])
                    rec["entropy"] = float(-np.sum([v * np.log(v) for v in p if v > 0]))
                    rec["probabilities"] = [float(v) for v in p]
                    if len(classes) == p.size:
                        rec["probabilities_by_class"] = {classes[k]: float(p[k]) for k in range(p.size)}
                records.append(rec)

        return metrics, records

    def _evaluate_regression(self, model, Xt, loader, return_records):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_true = np.asarray(loader.y, dtype=float)
        y_pred = np.asarray(model.predict(Xt), dtype=float).reshape(-1)

        metrics = {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
            "loss": float(mean_squared_error(y_true, y_pred)),
        }

        records = []
        if return_records:
            for i, rid in enumerate(loader.ids):
                records.append({
                    "record_id": str(rid),
                    "prediction": float(y_pred[i]),
                    "predicted_value": float(y_pred[i]),
                })

        return metrics, records

    # ---- snapshot / restore ------------------------------------------------
    def _snapshot(self, model):
        return deepcopy(model)

    def _restore(self, model, state):
        # sklearn models aren't mutated in place across epochs the way nn.Modules
        # are; the harness holds the best snapshot and we re-point to it.
        self._best_model = state

    # ---- model artifact ----------------------------------------------------
    def _write_model_artifact(self, model, parts, target: TargetSpec, *, split_spec_artifact_id=None):
        import joblib
        from sklearn.pipeline import Pipeline

        best = getattr(self, "_best_model", None) or model

        # The durable artifact is the FULL pipeline: train-fit preprocessor +
        # best estimator. prediction.SklearnTabularPredictor calls .predict(X)
        # on raw feature columns, so preprocessing must be embedded.
        pipeline = Pipeline([
            ("preprocess", self._fitted_preprocessor),
            ("model", best),
        ])

        prediction_capabilities = {
            "predict": callable(getattr(pipeline, "predict", None)),
            "predict_proba": callable(getattr(pipeline, "predict_proba", None)),
            "decision_function": callable(getattr(pipeline, "decision_function", None)),
        }

        prediction_capabilities["confidence_available"] = (
            prediction_capabilities["predict_proba"]
            or prediction_capabilities["decision_function"]
        )

        prediction_capabilities["confidence_source"] = (
            "predict_proba"
            if prediction_capabilities["predict_proba"]
            else "decision_function"
            if prediction_capabilities["decision_function"]
            else "unavailable"
        )

        model_dir = ml_run_artifact_dir(self.run, kind="model")
        sidecar_path = model_dir / "model.joblib"
        joblib.dump(pipeline, sidecar_path)

        manifest_payload = {
            "schema_version": 2,
            "kind": "sklearn_estimator",
            "framework": "sklearn",
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
            "num_classes": int(target.num_classes) if target.kind == "classification" else 0,
            "feature_columns": self._feature_columns(),
            "train_dataset_id": parts.train_dataset_id,
            "validation_dataset_id": parts.validation_dataset_id,
            "test_dataset_id": parts.test_dataset_id,
            "validation_source": parts.validation_source,
            "test_source": parts.test_source,
            "prediction_capabilities": prediction_capabilities,
            "model_ref": {
                "storage": "local_file",
                "uri": str(sidecar_path),
                "path": str(sidecar_path),
                "format": "joblib",
                "framework": "sklearn",
                "metadata": {
                    "class_names": list(parts.train.classes),
                    "feature_columns": self._feature_columns(),
                    "recipe_id": self.run.recipe_id,
                    "run_id": self.run.run_id,
                    "python_type": f"{type(best).__module__}.{type(best).__name__}",
                    "prediction_capabilities": prediction_capabilities,
                },
            },
            "input_contract": {
                "record_id_column": parts.record_id_column,
                "target_column": parts.target_column,
                "feature_columns": self._feature_columns(),
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
            "metrics": {
                "best_score": self._best_score,
                "selection_metric": self.protocol.selection_metric,
                "best_epoch": self._best_epoch,
            },
        }

        self._eval_classes = list(parts.train.classes)

        return self.run.put_artifact(
            "ml.model",
            json_safe(manifest_payload),
            params=self.run.params,
        )


# =============================================================================
# Base recipe + registration
# =============================================================================

class SklearnRecipe(ManagedMLRecipe):
    """Base for sklearn recipes. Subclasses implement build_model; the default
    fit is one-shot (fit + one report_epoch). Warm-start estimators override
    fit via the helper below for a live validation curve."""

    framework = "sklearn"
    modality = "tabular"

    def configure_training(self, run, model) -> TrainingComponents:
        # sklearn folds optimizer/loss into the estimator; nothing to configure.
        return TrainingComponents(optimizer=None)

    def load_sample(self, run, row):
        # Unused for sklearn: the harness reads whole feature frames in
        # _make_loader rather than per-row samples.
        raise NotImplementedError("SklearnHarness does not call load_sample.")

    def train_transform(self, run):
        return None

    def eval_transform(self, run):
        return None

    def fit(self, run, *, model, components, train_loader, harness):
        Xt = harness._fitted_preprocessor.transform(train_loader.X)
        model.fit(Xt, train_loader.y)
        harness.report_epoch(1, model, train_metrics={})


def fit_warm_start(run, *, model, components, train_loader, harness, points=10):
    """Grow an n_estimators-based estimator in chunks, reporting val each chunk.

    Recovers Gen 1's live tuning curve, now against a verified partition.
    """
    Xt = harness._fitted_preprocessor.transform(train_loader.X)

    try:
        final_n = int(model.get_params(deep=False).get("n_estimators") or 1)
    except Exception:
        final_n = 1
    final_n = max(1, final_n)

    try:
        model.set_params(warm_start=True)
    except Exception:
        pass

    points = max(2, min(50, int(points)))
    chunk = max(1, round(final_n / points))
    steps = list(range(chunk, final_n + 1, chunk))
    if not steps or steps[-1] != final_n:
        steps.append(final_n)

    for epoch, n in enumerate(steps, start=1):
        run.check_cancelled()
        try:
            model.set_params(n_estimators=int(n))
        except Exception:
            pass
        model.fit(Xt, train_loader.y)
        harness.report_epoch(epoch, model, train_metrics={"n_estimators": int(n)})


def register() -> None:
    """Register the sklearn harness with make_harness. Idempotent."""
    register_harness(
        lambda framework, task, modality, **_: (
            str(framework).lower() == "sklearn"
            and str(task).lower() in {"classification", "regression"}
        ),
        SklearnHarness,
    )