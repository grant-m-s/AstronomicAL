from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterator, List, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from ..partition_reader import create_partition_reader
from ..protocol import PartitionRef, TargetSpec, TrainingComponents
from ..resource_estimates import RecipeDataAccess, RecipeDataAccessMode
from . import register_harness
from .sklearn import SklearnHarness, SklearnRecipe
from .streaming_partitions import StreamingPartitionHarnessMixin


class IncrementalNumericPreprocessor(BaseEstimator, TransformerMixin):
    """Joblib-safe numeric preprocessor that supports bounded partial fitting."""

    def __init__(self, feature_columns: Optional[List[str]] = None) -> None:
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self._fitted = False

    def partial_fit(self, X: Any, y: Any = None):
        values = self._values(X)
        if len(values):
            self.scaler.partial_fit(values)
            self._fitted = True
        return self

    def fit(self, X: Any, y: Any = None):
        values = self._values(X)
        self.scaler.fit(values)
        self._fitted = True
        return self

    def transform(self, X: Any):
        values = self._values(X)
        if not self._fitted:
            raise RuntimeError("Incremental numeric preprocessor has not been fitted.")
        return self.scaler.transform(values)

    def _values(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            columns = list(self.feature_columns or [])
            frame = X.loc[:, columns] if columns else X
            values = frame.to_numpy(dtype=np.float64, copy=True)
        else:
            values = np.asarray(X, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


class IncrementalSklearnHarness(StreamingPartitionHarnessMixin, SklearnHarness):
    """Protocol harness for numeric estimators implementing ``partial_fit``."""

    def __init__(self, run: Any, recipe: Any) -> None:
        super().__init__(run, recipe)
        self._fitted_preprocessor = IncrementalNumericPreprocessor(self._feature_columns())
        self._preprocessor = self._fitted_preprocessor

    def _make_loader(self, partition: Any, *, train: bool):
        if not isinstance(partition, PartitionRef):
            raise TypeError(
                "Incremental sklearn recipes require manifest-backed partitions."
            )
        return partition

    def _assert_output_dim(self, model: Any, target: TargetSpec, train_loader: Any) -> None:
        if not callable(getattr(model, "partial_fit", None)):
            raise TypeError(
                f"{type(model).__name__} does not implement partial_fit and cannot "
                "run through the incremental sklearn harness."
            )

    def fit_incremental(
        self,
        *,
        model: Any,
        train_partition: PartitionRef,
        components: TrainingComponents,
        epochs: int,
    ) -> None:
        features = self._feature_columns()
        self._validate_numeric_features(train_partition, features)

        if not bool(getattr(self._fitted_preprocessor, "_fitted", False)):
            total_rows = int(getattr(train_partition, "row_count", 0) or 0)
            rows_seen = 0
            self._progress_report(
                stage="preprocessing",
                message="Fitting incremental numeric scaling from bounded training batches.",
                detail=f"Features: `{len(features)}`. No complete feature matrix is materialised.",
                current=0,
                total=total_rows,
                unit="rows",
                force=True,
            )
            for frame in self._iter_frames(train_partition, features):
                self.run.check_cancelled()
                self._fitted_preprocessor.partial_fit(frame.loc[:, features])
                rows_seen += len(frame)
                self._progress_report(
                    stage="preprocessing",
                    message="Updating incremental feature means and variances from training batches.",
                    current=rows_seen,
                    total=total_rows,
                    unit="rows",
                )
            self._progress_report(
                stage="preprocessing",
                message="Incremental numeric preprocessing is fitted.",
                current=rows_seen,
                total=total_rows or rows_seen,
                unit="rows",
                force=True,
            )

        total_epochs = max(1, int(epochs))
        classes = np.asarray(list(train_partition.classes), dtype=str)
        already_fitted = hasattr(model, "classes_") or hasattr(model, "coef_")

        partition_rows = int(getattr(train_partition, "row_count", 0) or 0)
        for epoch in self.run.epoch_range(total_epochs):
            rows_seen = 0
            batches_seen = 0
            self._progress_report(
                stage="training",
                message=f"Incremental sklearn epoch {epoch} of {total_epochs}: reading batches and calling partial_fit().",
                current=0,
                total=partition_rows,
                unit="rows",
                epoch=epoch,
                total_epochs=total_epochs,
                force=True,
            )
            for frame in self._iter_frames(train_partition, features):
                self.run.check_cancelled()
                X = self._fitted_preprocessor.transform(frame.loc[:, features])
                y = self._target_values(frame)
                kwargs: Dict[str, Any] = {}
                if (
                    self._task_kind() == "classification"
                    and not already_fitted
                    and rows_seen == 0
                ):
                    kwargs["classes"] = classes
                model.partial_fit(X, y, **kwargs)
                already_fitted = True
                rows_seen += len(frame)
                batches_seen += 1
                self._progress_report(
                    stage="training",
                    message=f"Incremental sklearn epoch {epoch} of {total_epochs}: estimator update {batches_seen} complete.",
                    current=rows_seen,
                    total=partition_rows,
                    unit="rows",
                    epoch=epoch,
                    total_epochs=total_epochs,
                )

            if rows_seen == 0:
                raise ValueError("The incremental training partition has no usable rows.")

            self.report_epoch(
                epoch,
                model,
                train_metrics={
                    "rows_seen": int(rows_seen),
                    "batches_seen": int(batches_seen),
                },
            )
            self.check_pause_boundary(epoch, model, components)

    def _evaluate(self, model: Any, loader: Any, *, return_records: bool = False):
        if not isinstance(loader, PartitionRef):
            return super()._evaluate(model, loader, return_records=return_records)

        features = self._feature_columns()
        if self._task_kind() == "regression":
            metrics = _OnlineRegressionMetrics()
        else:
            metrics = _OnlineClassificationMetrics(list(loader.classes))

        records: List[Dict[str, Any]] = []
        classes = [str(value) for value in getattr(model, "classes_", loader.classes)]
        rows_seen = 0
        total_rows = int(getattr(loader, "row_count", 0) or 0)
        role = str(getattr(loader, "role", "validation"))
        stage = "test_evaluation" if role == "test" else "validation"
        for frame in self._iter_frames(loader, features):
            self.run.check_cancelled()
            X = self._fitted_preprocessor.transform(frame.loc[:, features])
            y_true = self._target_values(frame)
            y_pred = np.asarray(model.predict(X))
            probabilities = _predict_probabilities(model, X)
            metrics.update(y_true, y_pred, probabilities, classes)
            rows_seen += len(frame)
            self._progress_report(
                stage=stage,
                message=f"Evaluating `{role}` rows with the incremental sklearn model.",
                current=rows_seen,
                total=total_rows,
                unit="rows",
            )

            if return_records:
                row_ids = frame[self.binding.record_id_column].astype(str).tolist()
                records.extend(
                    _prediction_records(
                        task=self._task_kind(),
                        row_ids=row_ids,
                        y_true=y_true,
                        y_pred=y_pred,
                        probabilities=probabilities,
                        classes=classes,
                    )
                )

        return metrics.finalize(), records

    def _iter_frames(
        self,
        partition: PartitionRef,
        features: List[str],
    ) -> Iterator[pd.DataFrame]:
        columns = [*features, self.binding.record_id_column]
        if self.binding.target_column:
            columns.append(self.binding.target_column)
        reader = create_partition_reader(
            self.run.context,
            partition,
            default_batch_size=int(
                self.run.params.get("stream_source_batch_size") or 8192
            ),
        )
        for batch in reader.iter_batches(
            columns=list(dict.fromkeys(columns)),
            strict=True,
            cancel_check=self.run.check_cancelled,
        ):
            frame = batch.frame
            if self.binding.target_column:
                frame = frame.dropna(subset=[self.binding.target_column])
            if not frame.empty:
                yield frame.reset_index(drop=True)

    def _validate_numeric_features(
        self,
        partition: PartitionRef,
        features: List[str],
    ) -> None:
        for frame in self._iter_frames(partition, features):
            non_numeric = [
                column
                for column in features
                if not pd.api.types.is_numeric_dtype(frame[column])
            ]
            if non_numeric:
                raise ValueError(
                    "Incremental sklearn recipes currently require numeric feature "
                    "columns. Unsupported columns: " + ", ".join(non_numeric)
                )
            return

    def _target_values(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.binding.target_column]
        if self._task_kind() == "regression":
            return values.to_numpy(dtype=float)
        return values.astype(str).to_numpy()


class IncrementalSklearnRecipe(SklearnRecipe):
    """Base class for managed sklearn recipes using bounded ``partial_fit``."""

    data_access = RecipeDataAccess(
        mode=RecipeDataAccessMode.INCREMENTAL,
        description=(
            "Fits numeric preprocessing and the estimator incrementally from "
            "bounded DatasetSource batches."
        ),
        requires_batch_scan=True,
        requires_numeric_features=True,
        supports_partial_fit=True,
        memory_multiplier=0.0,
        working_set_multiplier=4.0,
        disk_multiplier=1.15,
        blocking_bytes=0,
    )

    def fit(self, run, *, model, components, train_loader, harness):
        harness.fit_incremental(
            model=model,
            train_partition=train_loader,
            components=components,
            epochs=int(run.params.get("epochs") or 5),
        )


class _OnlineClassificationMetrics:
    def __init__(self, classes: List[str]) -> None:
        self.classes = list(dict.fromkeys(str(value) for value in classes))
        self.total = 0
        self.correct = 0
        self.support: Dict[str, int] = defaultdict(int)
        self.predicted: Dict[str, int] = defaultdict(int)
        self.true_positive: Dict[str, int] = defaultdict(int)
        self.loss_sum = 0.0
        self.loss_count = 0

    def update(
        self,
        y_true: Any,
        y_pred: Any,
        probabilities: Optional[np.ndarray],
        probability_classes: List[str],
    ) -> None:
        true_values = np.asarray(y_true).astype(str)
        pred_values = np.asarray(y_pred).astype(str)
        for actual, predicted in zip(true_values, pred_values):
            self.total += 1
            self.support[str(actual)] += 1
            self.predicted[str(predicted)] += 1
            if str(actual) == str(predicted):
                self.correct += 1
                self.true_positive[str(actual)] += 1

        if probabilities is not None and len(probabilities):
            class_index = {
                str(value): index
                for index, value in enumerate(probability_classes)
            }
            for index, actual in enumerate(true_values):
                column = class_index.get(str(actual))
                if column is None or column >= probabilities.shape[1]:
                    continue
                probability = float(probabilities[index, column])
                self.loss_sum += -float(np.log(max(probability, 1e-12)))
                self.loss_count += 1

    def finalize(self) -> Dict[str, float]:
        if not self.total:
            return {}
        labels = sorted(
            set(self.classes)
            | set(self.support)
            | set(self.predicted)
        )
        recalls: List[float] = []
        f1_values: List[float] = []
        for label in labels:
            tp = float(self.true_positive[label])
            support = float(self.support[label])
            predicted = float(self.predicted[label])
            recall = tp / support if support else 0.0
            precision = tp / predicted if predicted else 0.0
            if support:
                recalls.append(recall)
            f1_values.append(
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        accuracy = self.correct / self.total
        result = {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
            "f1_macro": float(np.mean(f1_values)) if f1_values else 0.0,
            "loss": (
                float(self.loss_sum / self.loss_count)
                if self.loss_count
                else float(1.0 - accuracy)
            ),
        }
        return result


class _OnlineRegressionMetrics:
    def __init__(self) -> None:
        self.count = 0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.sum_y = 0.0
        self.sum_y_squared = 0.0

    def update(
        self,
        y_true: Any,
        y_pred: Any,
        probabilities: Optional[np.ndarray] = None,
        probability_classes: Optional[List[str]] = None,
    ) -> None:
        true_values = np.asarray(y_true, dtype=float).reshape(-1)
        pred_values = np.asarray(y_pred, dtype=float).reshape(-1)
        residual = true_values - pred_values
        self.count += len(true_values)
        self.absolute_error += float(np.abs(residual).sum())
        self.squared_error += float(np.square(residual).sum())
        self.sum_y += float(true_values.sum())
        self.sum_y_squared += float(np.square(true_values).sum())

    def finalize(self) -> Dict[str, float]:
        if not self.count:
            return {}
        mse = self.squared_error / self.count
        total_variance = self.sum_y_squared - (self.sum_y**2 / self.count)
        r2 = 1.0 - self.squared_error / total_variance if total_variance > 0 else 0.0
        return {
            "r2": float(r2),
            "mae": float(self.absolute_error / self.count),
            "rmse": float(np.sqrt(mse)),
            "loss": float(mse),
        }


def _predict_probabilities(model: Any, X: Any) -> Optional[np.ndarray]:
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        try:
            values = np.asarray(predict_proba(X), dtype=float)
            if values.ndim == 1:
                values = np.column_stack([1.0 - values, values])
            return values
        except Exception:
            return None
    return None


def _prediction_records(
    *,
    task: str,
    row_ids: List[str],
    y_true: Any,
    y_pred: Any,
    probabilities: Optional[np.ndarray],
    classes: List[str],
) -> List[Dict[str, Any]]:
    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)
    records: List[Dict[str, Any]] = []
    for index, row_id in enumerate(row_ids):
        if task == "regression":
            records.append(
                {
                    "record_id": str(row_id),
                    "prediction": float(pred_values[index]),
                    "predicted_value": float(pred_values[index]),
                    "y_true": float(true_values[index]),
                }
            )
            continue

        record: Dict[str, Any] = {
            "record_id": str(row_id),
            "prediction": str(pred_values[index]),
            "y_true": str(true_values[index]),
        }
        if probabilities is not None and index < len(probabilities):
            values = np.asarray(probabilities[index], dtype=float)
            order = np.sort(values)[::-1]
            record["confidence"] = float(order[0])
            record["max_probability"] = float(order[0])
            record["least_confidence"] = float(1.0 - order[0])
            if len(order) >= 2:
                record["margin"] = float(order[0] - order[1])
                record["margin_uncertainty"] = float(1.0 - record["margin"])
            record["entropy"] = float(
                -np.sum([value * np.log(value) for value in values if value > 0])
            )
            record["probabilities"] = [float(value) for value in values]
            if len(classes) == len(values):
                record["probabilities_by_class"] = {
                    classes[position]: float(values[position])
                    for position in range(len(values))
                }
        records.append(record)
    return records


def register() -> None:
    register_harness(
        lambda framework, task, modality, recipe, **_: (
            str(framework).lower() == "sklearn"
            and str(task).lower() in {"classification", "regression"}
            and str(modality).lower() == "tabular"
            and RecipeDataAccess.coerce(
                getattr(recipe, "data_access", None),
                framework=framework,
            ).mode
            == RecipeDataAccessMode.INCREMENTAL
        ),
        IncrementalSklearnHarness,
    )
