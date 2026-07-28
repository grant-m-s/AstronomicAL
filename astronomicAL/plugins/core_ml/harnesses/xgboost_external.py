from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..partition_reader import create_partition_reader
from ..protocol import PartitionRef, TargetSpec, TrainingComponents
from ..resource_estimates import RecipeDataAccess, RecipeDataAccessMode
from . import register_harness
from .sklearn import SklearnHarness, SklearnRecipe
from .streaming_partitions import StreamingPartitionHarnessMixin


class NumericFrameTransformer(BaseEstimator, TransformerMixin):
    """Prediction-time numeric conversion matching external-memory training."""

    def __init__(self, feature_columns: Optional[List[str]] = None) -> None:
        self.feature_columns = feature_columns

    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        if isinstance(X, pd.DataFrame):
            columns = list(self.feature_columns or [])
            frame = X.loc[:, columns] if columns else X
            values = frame.to_numpy(dtype=np.float32, copy=True)
        else:
            values = np.asarray(X, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


class ExternalMemoryXGBoostModel(BaseEstimator):
    """Joblib-safe sklearn-style adapter around an XGBoost Booster."""

    def __init__(self, *, task: str, classes: Optional[List[str]] = None) -> None:
        self.task = str(task)
        self.classes = classes
        self.classes_ = np.asarray(list(classes or []), dtype=object)
        self.booster: Any = None

    def fit(self, X: Any, y: Any = None):
        raise RuntimeError(
            "ExternalMemoryXGBoostModel is fitted only by the managed "
            "external-memory harness."
        )

    def __deepcopy__(self, memo: Dict[int, Any]):
        copied = type(self)(
            task=self.task,
            classes=[str(value) for value in self.classes_],
        )
        copied.booster = _copy_booster(self.booster)
        return copied

    def __getattribute__(self, name: str):
        if name == "predict_proba":
            task = object.__getattribute__(self, "task")
            if task != "classification":
                raise AttributeError(
                    "Regression XGBoost models do not expose predict_proba."
                )
        return object.__getattribute__(self, name)

    def _matrix(self, X: Any):
        import xgboost as xgb

        values = np.asarray(X, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return xgb.DMatrix(values)

    def _raw_prediction(self, X: Any) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("XGBoost model has not been fitted.")
        raw = np.asarray(self.booster.predict(self._matrix(X)))
        if (
            self.task == "classification"
            and len(self.classes_) > 2
            and raw.ndim == 1
        ):
            raw = raw.reshape(-1, len(self.classes_))
        return raw

    def predict(self, X: Any):
        raw = self._raw_prediction(X)
        if self.task == "regression":
            return raw.reshape(-1)
        probabilities = _classification_probabilities(raw, len(self.classes_))
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: Any):
        if self.task != "classification":
            raise AttributeError("Regression models do not expose predict_proba.")
        return _classification_probabilities(
            self._raw_prediction(X),
            len(self.classes_),
        )


@dataclass
class XGBoostExternalBatch:
    role: str
    matrix: Any
    ids: List[str]
    classes: List[str]
    row_count: int
    cache_prefix: str


class ExternalMemoryXGBoostHarness(
    StreamingPartitionHarnessMixin,
    SklearnHarness,
):
    """Numeric XGBoost harness backed by XGBoost external-memory pages."""

    def __init__(self, run: Any, recipe: Any) -> None:
        super().__init__(run, recipe)
        self._matrix_cache: Dict[str, XGBoostExternalBatch] = {}
        self._fitted_preprocessor = NumericFrameTransformer(
            self._feature_columns()
        )
        self._preprocessor = self._fitted_preprocessor

    def _training_step_label(self) -> str:
        return "boosting round"

    def execute(self) -> Dict[str, Any]:
        try:
            return super().execute()
        finally:
            # DMatrix objects own the external-memory page files. Release them
            # before the runner removes the run working directory.
            self._val_loader = None
            self._matrix_cache.clear()
            gc.collect()

    def _make_loader(self, partition: Any, *, train: bool):
        if not isinstance(partition, PartitionRef):
            raise TypeError(
                "External-memory XGBoost requires manifest-backed partitions."
            )
        return self._external_batch(
            partition,
            collect_ids=str(partition.role).lower() == "test",
        )

    def _assert_output_dim(
        self,
        model: Any,
        target: TargetSpec,
        train_loader: Any,
    ) -> None:
        if not isinstance(model, ExternalMemoryXGBoostModel):
            raise TypeError(
                "External-memory XGBoost recipes must build "
                "ExternalMemoryXGBoostModel."
            )
        if (
            target.kind == "classification"
            and len(model.classes_) != target.num_outputs
        ):
            raise ValueError(
                "XGBoost model class metadata does not match the training "
                "partition."
            )

    def fit_external(
        self,
        *,
        model: ExternalMemoryXGBoostModel,
        train_loader: XGBoostExternalBatch,
        components: TrainingComponents,
    ) -> None:
        import xgboost as xgb

        final_rounds = max(
            1,
            int(self.run.params.get("n_estimators") or 800),
        )
        completed = max(
            0,
            int(getattr(self.run, "start_epoch", 1) or 1) - 1,
        )
        if completed >= final_rounds:
            raise ValueError(
                f"Resume checkpoint already contains {completed} boosting "
                f"rounds, which is not below requested "
                f"n_estimators={final_rounds}."
            )

        points = max(
            1,
            min(50, int(self.run.params.get("curve_points") or 10)),
        )
        remaining = final_rounds - completed
        chunk_size = max(1, int(np.ceil(remaining / points)))
        booster = model.booster

        self._progress_report(
            stage="training",
            message="Training XGBoost from disk-backed external-memory pages.",
            detail=f"Configured boosting rounds: `{final_rounds}`; reporting in chunks of up to `{chunk_size}` rounds.",
            current=completed,
            total=final_rounds,
            unit="boosting rounds",
            epoch=completed or 1,
            total_epochs=final_rounds,
            force=True,
        )
        while completed < final_rounds:
            self.run.check_cancelled()
            step = min(chunk_size, final_rounds - completed)
            self._progress_report(
                stage="training",
                message=f"Training XGBoost boosting rounds {completed + 1}–{completed + step} of {final_rounds}.",
                current=completed,
                total=final_rounds,
                unit="boosting rounds",
                epoch=completed + 1,
                total_epochs=final_rounds,
                force=True,
            )
            booster = xgb.train(
                params=self._booster_params(train_loader),
                dtrain=train_loader.matrix,
                num_boost_round=step,
                xgb_model=booster,
                verbose_eval=False,
            )
            completed += step
            model.booster = booster
            self._progress_report(
                stage="training",
                message=f"XGBoost has completed {completed} of {final_rounds} boosting rounds.",
                current=completed,
                total=final_rounds,
                unit="boosting rounds",
                epoch=completed,
                total_epochs=final_rounds,
                force=True,
            )
            self.report_epoch(
                completed,
                model,
                train_metrics={"boosting_rounds": int(completed)},
            )
            self.check_pause_boundary(completed, model, components)

    def _booster_params(
        self,
        loader: XGBoostExternalBatch,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "eta": float(self.run.params.get("learning_rate") or 0.05),
            "max_depth": int(self.run.params.get("max_depth") or 6),
            "subsample": float(self.run.params.get("subsample") or 0.8),
            "colsample_bytree": float(
                self.run.params.get("colsample_bytree") or 0.8
            ),
            "min_child_weight": float(
                self.run.params.get("min_child_weight") or 1.0
            ),
            "alpha": float(self.run.params.get("reg_alpha") or 0.0),
            "lambda": float(self.run.params.get("reg_lambda") or 1.0),
            "tree_method": str(
                self.run.params.get("tree_method") or "hist"
            ),
            "seed": int(self.run.params.get("random_state") or 42),
            "nthread": int(self.run.params.get("n_jobs") or -1),
        }
        if self._task_kind() == "regression":
            params.update(
                {
                    "objective": str(
                        self.run.params.get("objective")
                        or "reg:squarederror"
                    ),
                    "eval_metric": "rmse",
                }
            )
        elif len(loader.classes) <= 2:
            params.update(
                {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                }
            )
        else:
            params.update(
                {
                    "objective": "multi:softprob",
                    "num_class": len(loader.classes),
                    "eval_metric": "mlogloss",
                }
            )
        return params

    def _external_batch(
        self,
        partition: PartitionRef,
        *,
        collect_ids: bool,
    ) -> XGBoostExternalBatch:
        cache_key = f"{partition.role}:{partition.fingerprint}"
        existing = self._matrix_cache.get(cache_key)
        if existing is not None:
            return existing

        cache_root = Path(self.run.work_dir) / "xgboost-cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_prefix = str(
            cache_root
            / f"{_safe_token(partition.role)}-{partition.fingerprint[:12]}"
        )
        data_iter = self._make_data_iter(
            partition,
            cache_prefix=cache_prefix,
            collect_ids=collect_ids,
        )

        import xgboost as xgb

        role = str(partition.role)
        self._progress_report(
            stage="external_memory",
            message=f"Building XGBoost external-memory pages for the `{role}` partition.",
            detail=(
                f"Rows `{int(partition.row_count):,}`, features `{len(self._feature_columns()):,}`. "
                "Rows are streamed into XGBoost cache pages rather than retained as one matrix."
            ),
            current=0,
            total=int(partition.row_count),
            unit="rows",
            force=True,
        )
        with self._progress_activity(
            stage="external_memory",
            message=f"XGBoost is scanning `{role}` rows and writing external-memory cache pages.",
            detail=f"Cache prefix `{cache_prefix}`.",
        ):
            matrix = xgb.DMatrix(
                data_iter,
                nthread=int(self.run.params.get("n_jobs") or -1),
            )
        row_count = int(matrix.num_row())
        if row_count <= 0:
            raise ValueError(
                f"The {partition.role} partition has no usable rows for "
                "XGBoost."
            )

        self._progress_report(
            stage="external_memory",
            message=f"External-memory pages for `{role}` are ready.",
            current=row_count,
            total=int(partition.row_count) or row_count,
            unit="rows",
            force=True,
        )

        batch = XGBoostExternalBatch(
            role=str(partition.role),
            matrix=matrix,
            ids=list(data_iter.record_ids),
            classes=list(partition.classes),
            row_count=row_count,
            cache_prefix=cache_prefix,
        )
        self._matrix_cache[cache_key] = batch
        return batch

    def _make_data_iter(
        self,
        partition: PartitionRef,
        *,
        cache_prefix: str,
        collect_ids: bool,
    ):
        import xgboost as xgb

        harness = self
        features = self._feature_columns()
        class_index = {
            str(value): index
            for index, value in enumerate(partition.classes)
        }
        columns = [*features, self.binding.record_id_column]
        if self.binding.target_column:
            columns.append(self.binding.target_column)
        columns = list(dict.fromkeys(columns))

        class PartitionDataIter(xgb.DataIter):
            def __init__(self) -> None:
                super().__init__(
                    cache_prefix=cache_prefix,
                    release_data=True,
                )
                self.record_ids: List[str] = []
                self._iterator: Optional[Iterator[Any]] = None
                self._pass_index = 0
                self._started = False
                self._rows_seen = 0
                self._batches_seen = 0

            def reset(self) -> None:
                if self._started:
                    self._pass_index += 1
                self._iterator = None
                self._started = False
                self._rows_seen = 0
                self._batches_seen = 0

            def next(self, input_data) -> bool:
                if self._iterator is None:
                    reader = create_partition_reader(
                        harness.run.context,
                        partition,
                        default_batch_size=int(
                            harness.run.params.get(
                                "stream_source_batch_size"
                            )
                            or 8192
                        ),
                    )
                    self._iterator = iter(
                        reader.iter_batches(
                            columns=columns,
                            strict=True,
                            cancel_check=harness.run.check_cancelled,
                        )
                    )

                for batch in self._iterator:
                    harness.run.check_cancelled()
                    frame = batch.frame
                    if harness.binding.target_column:
                        frame = frame.dropna(
                            subset=[harness.binding.target_column]
                        )
                    if frame.empty:
                        continue
                    harness._validate_numeric(frame, features)
                    values = np.nan_to_num(
                        frame.loc[:, features].to_numpy(
                            dtype=np.float32,
                            copy=True,
                        ),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    if harness._task_kind() == "regression":
                        labels = frame[
                            harness.binding.target_column
                        ].to_numpy(dtype=float)
                    else:
                        labels = np.asarray(
                            [
                                class_index[str(value)]
                                for value in frame[
                                    harness.binding.target_column
                                ].astype(str)
                            ],
                            dtype=np.int64,
                        )
                    if collect_ids and self._pass_index == 0:
                        self.record_ids.extend(
                            frame[
                                harness.binding.record_id_column
                            ].astype(str).tolist()
                        )
                    self._started = True
                    self._rows_seen += len(frame)
                    self._batches_seen += 1
                    harness._progress_report(
                        stage="external_memory",
                        message=(
                            f"Streaming `{partition.role}` rows into XGBoost cache pages "
                            f"(scan pass {self._pass_index + 1}, batch {self._batches_seen})."
                        ),
                        current=self._rows_seen,
                        total=int(partition.row_count),
                        unit="rows",
                    )
                    input_data(
                        data=values,
                        label=labels,
                        feature_names=features,
                    )
                    return True
                return False

        return PartitionDataIter()

    def _validate_numeric(
        self,
        frame: pd.DataFrame,
        features: List[str],
    ) -> None:
        non_numeric = [
            column
            for column in features
            if not pd.api.types.is_numeric_dtype(frame[column])
        ]
        if non_numeric:
            raise ValueError(
                "External-memory XGBoost currently requires numeric feature "
                "columns. Unsupported columns: " + ", ".join(non_numeric)
            )

    def _evaluate(
        self,
        model: ExternalMemoryXGBoostModel,
        loader: Any,
        *,
        return_records: bool = False,
    ):
        if not isinstance(loader, XGBoostExternalBatch):
            return super()._evaluate(
                model,
                loader,
                return_records=return_records,
            )
        raw = np.asarray(model.booster.predict(loader.matrix))
        labels = np.asarray(loader.matrix.get_label())
        if self._task_kind() == "regression":
            return _regression_metrics_and_records(
                labels,
                raw,
                loader.ids,
                return_records=return_records,
            )

        probabilities = _classification_probabilities(
            raw,
            len(loader.classes),
        )
        prediction_indices = np.argmax(probabilities, axis=1)
        class_values = np.asarray(loader.classes, dtype=object)
        y_pred = class_values[prediction_indices]
        y_true = class_values[labels.astype(int)]
        return _classification_metrics_and_records(
            y_true,
            y_pred,
            probabilities,
            loader.ids,
            loader.classes,
            return_records=return_records,
        )

    def _snapshot(self, model: ExternalMemoryXGBoostModel):
        snapshot = ExternalMemoryXGBoostModel(
            task=model.task,
            classes=[str(value) for value in model.classes_],
        )
        snapshot.booster = _copy_booster(model.booster)
        return snapshot

    def _restore(
        self,
        model: ExternalMemoryXGBoostModel,
        state: ExternalMemoryXGBoostModel,
    ) -> None:
        model.booster = _copy_booster(state.booster)
        self._best_model = model


class ExternalMemoryXGBoostRecipe(SklearnRecipe):
    """Base recipe for XGBoost training through disk-backed DMatrix pages."""

    data_access = RecipeDataAccess(
        mode=RecipeDataAccessMode.EXTERNAL_MEMORY,
        description=(
            "Streams numeric rows into XGBoost disk-backed pages and trains "
            "without a full in-memory feature matrix."
        ),
        requires_batch_scan=True,
        requires_numeric_features=True,
        uses_external_memory=True,
        memory_multiplier=0.0,
        working_set_multiplier=5.0,
        disk_multiplier=2.2,
        fixed_overhead_bytes=512 * 1024**2,
        blocking_bytes=0,
    )

    def fit(self, run, *, model, components, train_loader, harness):
        harness.fit_external(
            model=model,
            train_loader=train_loader,
            components=components,
        )


def _copy_booster(booster: Any):
    if booster is None:
        return None
    import xgboost as xgb

    try:
        raw = booster.save_raw(raw_format="ubj")
    except TypeError:
        raw = booster.save_raw()
    copied = xgb.Booster()
    copied.load_model(bytearray(raw))
    return copied


def _classification_probabilities(
    raw: np.ndarray,
    class_count: int,
) -> np.ndarray:
    values = np.asarray(raw, dtype=float)
    if class_count <= 2:
        probabilities = values.reshape(-1)
        return np.column_stack([1.0 - probabilities, probabilities])
    if values.ndim == 1:
        values = values.reshape(-1, class_count)
    return values


def _classification_metrics_and_records(
    y_true: Any,
    y_pred: Any,
    probabilities: np.ndarray,
    row_ids: List[str],
    classes: List[str],
    *,
    return_records: bool,
):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
    )

    true_values = np.asarray(y_true).astype(str)
    pred_values = np.asarray(y_pred).astype(str)
    metrics = {
        "accuracy": float(accuracy_score(true_values, pred_values)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_values, pred_values)
        ),
        "f1_macro": float(
            f1_score(
                true_values,
                pred_values,
                average="macro",
                zero_division=0,
            )
        ),
    }
    class_index = {
        str(value): index
        for index, value in enumerate(classes)
    }
    losses = []
    for index, actual in enumerate(true_values):
        column = class_index.get(str(actual))
        if column is not None and column < probabilities.shape[1]:
            losses.append(
                -np.log(max(float(probabilities[index, column]), 1e-12))
            )
    if losses:
        metrics["loss"] = float(np.mean(losses))

    records: List[Dict[str, Any]] = []
    if return_records:
        if len(row_ids) != len(true_values):
            raise RuntimeError(
                "XGBoost prediction row IDs are incomplete; refusing to emit "
                "misaligned prediction records."
            )
        for index, row_id in enumerate(row_ids):
            values = probabilities[index]
            order = np.sort(values)[::-1]
            record: Dict[str, Any] = {
                "record_id": str(row_id),
                "prediction": str(pred_values[index]),
                "y_true": str(true_values[index]),
                "confidence": float(order[0]),
                "max_probability": float(order[0]),
                "least_confidence": float(1.0 - order[0]),
                "entropy": float(
                    -np.sum(
                        [
                            value * np.log(value)
                            for value in values
                            if value > 0
                        ]
                    )
                ),
                "probabilities": [float(value) for value in values],
                "probabilities_by_class": {
                    str(classes[position]): float(values[position])
                    for position in range(len(values))
                },
            }
            if len(order) >= 2:
                record["margin"] = float(order[0] - order[1])
                record["margin_uncertainty"] = float(
                    1.0 - record["margin"]
                )
            records.append(record)
    return metrics, records


def _regression_metrics_and_records(
    y_true: Any,
    y_pred: Any,
    row_ids: List[str],
    *,
    return_records: bool,
):
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    pred_values = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = float(mean_squared_error(true_values, pred_values))
    metrics = {
        "r2": float(r2_score(true_values, pred_values)),
        "mae": float(mean_absolute_error(true_values, pred_values)),
        "rmse": float(np.sqrt(mse)),
        "loss": mse,
    }
    if return_records and len(row_ids) != len(true_values):
        raise RuntimeError(
            "XGBoost prediction row IDs are incomplete; refusing to emit "
            "misaligned prediction records."
        )
    records = (
        [
            {
                "record_id": str(row_id),
                "prediction": float(prediction),
                "predicted_value": float(prediction),
                "y_true": float(actual),
            }
            for row_id, prediction, actual in zip(
                row_ids,
                pred_values,
                true_values,
            )
        ]
        if return_records
        else []
    )
    return metrics, records


def _safe_token(value: Any) -> str:
    token = "".join(
        character
        if str(character).isalnum() or character in {"-", "_"}
        else "_"
        for character in str(value)
    )
    return token.strip("_") or "partition"


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
            == RecipeDataAccessMode.EXTERNAL_MEMORY
        ),
        ExternalMemoryXGBoostHarness,
    )
