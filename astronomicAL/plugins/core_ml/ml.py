from __future__ import annotations

import importlib.util
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    """Convert numpy/pandas scalar values into JSON-safe Python values."""

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, np.generic):
        return value.item()

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass

    return value

def _import_object(path: str):
    import importlib

    module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _coerce_template_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert UI-friendly values into constructor-friendly values."""

    coerced = dict(params)

    if isinstance(coerced.get("hidden_layer_sizes"), str):
        coerced["hidden_layer_sizes"] = tuple(
            int(part.strip())
            for part in coerced["hidden_layer_sizes"].split(",")
            if part.strip()
        )

    return coerced


def model_spec_from_definition(definition: Dict[str, Any]) -> ModelSpec:
    """Create a live ModelSpec from a saved ml.model_definition artifact."""

    model_id = str(definition["id"])
    title = str(definition["title"])
    framework = str(definition["framework"])
    task = str(definition["task"])
    modality = str(definition.get("modality", "tabular"))
    params = dict(definition.get("params", {}))
    template = dict(definition.get("template", {}))

    metadata = {
        "template_id": definition.get("template_id"),
        "template": template,
        "tuning": definition.get("tuning", {}),
    }

    if framework == "sklearn":
        estimator_path = template["estimator"]

        def factory(runtime_params: Dict[str, Any]):
            estimator_cls = _import_object(estimator_path)
            merged = {**params, **runtime_params}
            merged = _coerce_template_params(merged)

            try:
                return estimator_cls(**merged)
            except TypeError:
                merged.pop("random_state", None)
                return estimator_cls(**merged)

        return ModelSpec(
            id=model_id,
            title=title,
            task=task,
            factory=factory,
            framework="sklearn",
            modality=modality,
            default_params=params,
            description=str(definition.get("description", "")),
            metadata=metadata,
        )

    def torch_factory(runtime_params: Dict[str, Any]):
        return {**params, **runtime_params}

    return ModelSpec(
        id=model_id,
        title=title,
        task=task,
        factory=torch_factory,
        framework="torch",
        modality=modality,
        default_params=params,
        description=str(definition.get("description", "")),
        metadata=metadata,
    )


def register_model_definition(registry: MLRegistry, definition: Dict[str, Any]) -> None:
    registry.register_model(model_spec_from_definition(definition))


def sync_model_definitions_from_artifacts(context: Any, registry: MLRegistry) -> int:
    artifacts = getattr(context, "artifacts", None)
    find = getattr(artifacts, "find", None)
    get = getattr(artifacts, "get", None)

    if not callable(find) or not callable(get):
        return 0

    count = 0

    try:
        refs = find(type="ml.model_definition")
    except Exception:
        return 0

    for ref in refs:
        try:
            definition = get(ref.artifact_id)
            register_model_definition(registry, definition)
            count += 1
        except Exception:
            continue

    return count

@dataclass(frozen=True)
class ModelSpec:
    id: str
    title: str
    task: str
    factory: Callable[[Dict[str, Any]], Any]
    framework: str = "sklearn"
    modality: str = "tabular"
    default_params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLRegistry:
    """Live registry of available ML model providers."""

    def __init__(self) -> None:
        self._models: Dict[str, ModelSpec] = {}

    def register_model(self, spec: ModelSpec) -> None:
        if not spec.id:
            raise ValueError("ModelSpec.id cannot be empty.")
        self._models[spec.id] = spec

    def get_model(self, model_id: str) -> ModelSpec:
        return self._models[model_id]

    def list_models(
        self,
        task: str | None = None,
        modality: str | None = None,
    ) -> List[ModelSpec]:
        models = list(self._models.values())

        if task:
            models = [m for m in models if m.task == task]

        if modality:
            models = [m for m in models if getattr(m, "modality", "tabular") == modality]

        return sorted(models, key=lambda m: (m.framework, m.modality, m.task, m.title))


def _tuning_config(model_spec: ModelSpec) -> Dict[str, Any]:
    metadata = getattr(model_spec, "metadata", {}) or {}
    tuning = metadata.get("tuning") or {}

    if not isinstance(tuning, dict):
        return {}

    if not tuning.get("enabled"):
        return {}

    if tuning.get("backend") != "optuna":
        return {}

    if not tuning.get("search_space"):
        return {}

    return tuning


def _optuna_direction(metric: str) -> str:
    metric = metric.lower()

    if (
        "loss" in metric
        or "error" in metric
        or "mae" in metric
        or "mse" in metric
        or "rmse" in metric
    ):
        return "minimize"

    return "maximize"


def _make_optuna_sampler(name: str, random_state: int):
    import optuna

    if name == "random":
        return optuna.samplers.RandomSampler(seed=int(random_state))

    return optuna.samplers.TPESampler(seed=int(random_state))


def _sample_optuna_params(trial: Any, search_space: Dict[str, Any]) -> Dict[str, Any]:
    sampled: Dict[str, Any] = {}

    for name, spec in search_space.items():
        kind = spec.get("type")

        if kind == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            log = bool(spec.get("log", False))

            if log:
                sampled[name] = trial.suggest_int(name, low, high, log=True)
            else:
                step = int(spec.get("step", 1) or 1)
                sampled[name] = trial.suggest_int(name, low, high, step=step)

        elif kind == "float":
            low = float(spec["low"])
            high = float(spec["high"])
            log = bool(spec.get("log", False))
            sampled[name] = trial.suggest_float(name, low, high, log=log)

        elif kind == "categorical":
            choices = list(spec.get("choices", []))
            if not choices:
                continue
            sampled[name] = trial.suggest_categorical(name, choices)

    return sampled


def _extract_objective_value(metrics: Dict[str, float], metric: str) -> float:
    if metric in metrics:
        return float(metrics[metric])

    # Friendly fallback for common naming differences.
    aliases = {
        "val_loss": ["loss", "val_log_loss"],
        "val_accuracy": ["accuracy"],
        "val_f1_macro": ["f1_macro"],
        "val_r2": ["r2"],
        "val_mae": ["mae"],
        "val_rmse": ["rmse"],
    }

    for alias in aliases.get(metric, []):
        if alias in metrics:
            return float(metrics[alias])

    raise ValueError(
        f"Optuna metric `{metric}` was not produced by evaluation. "
        f"Available metrics: {sorted(metrics.keys())}"
    )


def _publish_tuning_update(
    context: Any,
    *,
    artifact_id: Optional[str],
    run_id: str,
    message: str,
    trials: List[Dict[str, Any]],
) -> None:
    if not artifact_id:
        return

    try:
        payload = context.artifacts.get(artifact_id)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        payload.update(
            {
                "status": "tuning",
                "message": message,
                "tuning_trials": list(trials),
                "updated_at": time.time(),
            }
        )

    _publish(
        context,
        "ml.training_log.updated",
        {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "status": "tuning",
            "message": message,
        },
    )

def _optuna_tune_sklearn_model(
    *,
    context: Any,
    config: TrainConfig,
    model_spec: ModelSpec,
    split: Dict[str, Any],
    run_id: str,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    import optuna
    from sklearn.pipeline import Pipeline

    tuning = _tuning_config(model_spec)
    if not tuning:
        return {
            "enabled": False,
            "best_params": {},
            "best_value": None,
            "trials": [],
        }

    metric = str(tuning.get("metric", "val_f1_macro"))
    direction = _optuna_direction(metric)
    search_space = dict(tuning.get("search_space") or {})
    n_trials = int(tuning.get("n_trials", 30) or 30)
    timeout_seconds = tuning.get("timeout_seconds")
    sampler_name = str(tuning.get("sampler", "tpe"))

    sampler = _make_optuna_sampler(sampler_name, config.random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)

    trial_records: List[Dict[str, Any]] = []

    _publish_tuning_update(
        context,
        artifact_id=config.training_log_artifact_id,
        run_id=run_id,
        message=f"Starting Optuna tuning: {n_trials} trials, metric `{metric}`.",
        trials=trial_records,
    )

    def objective(trial: Any) -> float:
        _check_cancelled(cancel_token)

        trial_params = _sample_optuna_params(trial, search_space)

        preprocessor = _make_preprocessor(split["X_train"])
        estimator = model_spec.factory(trial_params)
        candidate = Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )

        candidate.fit(split["X_train"], split["y_train"])

        val_eval = _evaluate_estimator(
            task=config.task,
            estimator=candidate,
            X=split["X_val"],
            y=split["y_val"],
            rows=split["rows_val"],
        )

        prefixed_metrics = _prefix_metrics("val", val_eval["metrics"])
        value = _extract_objective_value(prefixed_metrics, metric)

        record = {
            "number": int(trial.number),
            "value": float(value),
            "metric": metric,
            "params": dict(trial_params),
            "state": "complete",
        }
        trial_records.append(record)

        _publish_tuning_update(
            context,
            artifact_id=config.training_log_artifact_id,
            run_id=run_id,
            message=f"Finished Optuna trial {trial.number + 1}/{n_trials}.",
            trials=trial_records,
        )

        return float(value)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        catch=(Exception,),
    )

    completed = [trial for trial in study.trials if trial.value is not None]

    if not completed:
        raise RuntimeError("Optuna tuning did not complete any valid trials.")

    best_params = dict(study.best_trial.params)
    best_value = float(study.best_value)

    _publish_tuning_update(
        context,
        artifact_id=config.training_log_artifact_id,
        run_id=run_id,
        message=f"Optuna tuning complete. Best {metric}={best_value:.6g}.",
        trials=trial_records,
    )

    return {
        "enabled": True,
        "backend": "optuna",
        "metric": metric,
        "direction": direction,
        "best_params": best_params,
        "best_value": best_value,
        "trials": trial_records,
    }

def build_default_registry() -> MLRegistry:
    """Built-in sklearn models, plus torch models if torch is installed."""

    registry = MLRegistry()

    def random_forest_classifier(params: Dict[str, Any]):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1,
        )

    def extra_trees_classifier(params: Dict[str, Any]):
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1,
        )

    def logistic_regression(params: Dict[str, Any]):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 1000)),
        )

    def random_forest_regressor(params: Dict[str, Any]):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1,
        )

    def extra_trees_regressor(params: Dict[str, Any]):
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1,
        )

    def ridge_regression(params: Dict[str, Any]):
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(params.get("alpha", 1.0)))

    registry.register_model(
        ModelSpec(
            id="sklearn.random_forest_classifier",
            title="Random Forest Classifier",
            task="classification",
            factory=random_forest_classifier,
            framework="sklearn",
            default_params={"n_estimators": 300},
            modality="tabular",
        )
    )
    registry.register_model(
        ModelSpec(
            id="sklearn.extra_trees_classifier",
            title="Extra Trees Classifier",
            task="classification",
            factory=extra_trees_classifier,
            framework="sklearn",
            default_params={"n_estimators": 300},
            modality="tabular",
        )
    )
    registry.register_model(
        ModelSpec(
            id="sklearn.logistic_regression",
            title="Logistic Regression",
            task="classification",
            factory=logistic_regression,
            framework="sklearn",
            default_params={"C": 1.0, "max_iter": 1000},
            modality="tabular",
        )
    )
    registry.register_model(
        ModelSpec(
            id="sklearn.random_forest_regressor",
            title="Random Forest Regressor",
            task="regression",
            factory=random_forest_regressor,
            framework="sklearn",
            default_params={"n_estimators": 300},
            modality="tabular",
        )
    )
    registry.register_model(
        ModelSpec(
            id="sklearn.extra_trees_regressor",
            title="Extra Trees Regressor",
            task="regression",
            factory=extra_trees_regressor,
            framework="sklearn",
            default_params={"n_estimators": 300},
            modality="tabular",
        )
    )
    registry.register_model(
        ModelSpec(
            id="sklearn.ridge_regression",
            title="Ridge Regression",
            task="regression",
            factory=ridge_regression,
            framework="sklearn",
            default_params={"alpha": 1.0},
            modality="tabular",
        )
    )

    if importlib.util.find_spec("torch") is not None:
        registry.register_model(
            ModelSpec(
                id="torch.mlp_classifier",
                title="Torch MLP Classifier",
                task="classification",
                factory=lambda params: params,
                framework="torch",
                default_params={},
                description="Simple tabular PyTorch MLP classifier.",
                modality="tabular",
            )
        )
        registry.register_model(
            ModelSpec(
                id="torch.mlp_regressor",
                title="Torch MLP Regressor",
                task="regression",
                factory=lambda params: params,
                framework="torch",
                default_params={},
                description="Simple tabular PyTorch MLP regressor.",
                modality="tabular",
            )
        )

    return registry


@dataclass
class TrainConfig:
    dataset_id: str
    task: str
    target_column: str
    feature_columns: List[str]
    model_id: str
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

    optimize_metric: str = "val_loss"
    torch_hidden_layers: str = "128,64"
    torch_epochs: int = 50
    torch_batch_size: int = 512
    torch_learning_rate: float = 1e-3
    torch_weight_decay: float = 0.0
    torch_patience: int = 12

    run_id: Optional[str] = None
    training_log_artifact_id: Optional[str] = None


def train_model(
    *,
    context: Any,
    registry: MLRegistry,
    config: TrainConfig,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    _check_cancelled(cancel_token)

    if not config.feature_columns:
        raise ValueError("Choose at least one feature column.")
    if config.target_column in config.feature_columns:
        raise ValueError("Target column cannot also be a feature column.")
    if config.test_size <= 0 or config.validation_size <= 0:
        raise ValueError("Test size and validation size must both be greater than zero.")
    if config.test_size + config.validation_size >= 0.9:
        raise ValueError("Test size + validation size must leave at least 10% for training.")

    run_id = config.run_id or uuid.uuid4().hex

    model_spec = registry.get_model(config.model_id)
    
    if getattr(model_spec, "modality", "tabular") != "tabular":
        raise ValueError(
            f"Model `{model_spec.title}` has modality "
            f"`{getattr(model_spec, 'modality', 'unknown')}`. "
            "The tabular backend only trains tabular models."
        )

    record_id_column = _mapped_column(context, config.dataset_id, "record_id")
    columns = list(dict.fromkeys([*config.feature_columns, config.target_column]))
    if record_id_column and record_id_column not in columns:
        columns.append(record_id_column)

    df = context.datasets.get_df(config.dataset_id, columns=columns)
    df = df.dropna(subset=[config.target_column])

    if df.empty:
        raise ValueError("No rows remain after dropping missing targets.")

    X = df[config.feature_columns]
    y = df[config.target_column]
    row_ids = _row_ids(context, config.dataset_id, df)

    split = _split_train_validation_test(
        X=X,
        y=y,
        row_ids=row_ids,
        task=config.task,
        test_size=config.test_size,
        validation_size=config.validation_size,
        random_state=config.random_state,
        stratify=config.stratify,
    )

    started = time.time()

    if model_spec.framework == "torch":
        result = _train_torch_model(
            context=context,
            config=config,
            model_spec=model_spec,
            split=split,
            run_id=run_id,
            cancel_token=cancel_token,
        )
    else:
        result = _train_sklearn_model(
            context=context,
            config=config,
            model_spec=model_spec,
            split=split,
            run_id=run_id,
            cancel_token=cancel_token,
        )

    finished = time.time()

    feature_payload = {
        "run_id": run_id,
        "modality": "tabular",
        "dataset_id": config.dataset_id,
        "target_column": config.target_column,
        "feature_columns": list(config.feature_columns),
    }

    split_payload = {
        "run_id": run_id,
        "test_size": config.test_size,
        "validation_size": config.validation_size,
        "random_state": config.random_state,
        "stratify": bool(config.task == "classification" and config.stratify),
        "train_row_ids": split["rows_train"],
        "validation_row_ids": split["rows_val"],
        "test_row_ids": split["rows_test"],
    }

    model_payload = {
        "run_id": run_id,
        "model": result["model"],
        "model_id": model_spec.id,
        "model_title": model_spec.title,
        "framework": model_spec.framework,
        "task": config.task,
        "dataset_id": config.dataset_id,
        "target_column": config.target_column,
        "feature_columns": list(config.feature_columns),
        "created_at": finished,
        "metadata": result.get("model_metadata", {}),
    }

    evaluation_payload = {
        "run_id": run_id,
        "task": config.task,
        "metrics": result["metrics"],
        "extra": result.get("extra", {}),
        "model_id": model_spec.id,
        "model_title": model_spec.title,
        "framework": model_spec.framework,
        "dataset_id": config.dataset_id,
    }

    predictions_payload = {
        "run_id": run_id,
        "task": config.task,
        "records": result["prediction_records"],
        "model_id": model_spec.id,
        "model_title": model_spec.title,
        "framework": model_spec.framework,
        "dataset_id": config.dataset_id,
    }

    training_log_payload = {
        "run_id": run_id,
        "task": config.task,
        "model_id": model_spec.id,
        "model_title": model_spec.title,
        "framework": model_spec.framework,
        "dataset_id": config.dataset_id,
        "optimize_metric": config.optimize_metric,
        "best_epoch": result.get("best_epoch"),
        "epochs": result.get("training_log", []),
    }

    training_log_artifact_id = config.training_log_artifact_id

    if training_log_artifact_id:
        _update_training_log_artifact(
            context,
            training_log_artifact_id,
            {
                **training_log_payload,
                "status": "finished",
                "message": "Training complete.",
                "finished_at": finished,
                "updated_at": finished,
            },
        )
    else:
        training_log_artifact_id = _put(
            context,
            "ml.training_log",
            {
                **training_log_payload,
                "status": "finished",
                "message": "Training complete.",
                "started_at": started,
                "finished_at": finished,
                "updated_at": finished,
            },
            config.dataset_id,
        )

    artifact_ids = {
        "feature_spec": _put(context, "ml.feature_spec", feature_payload, config.dataset_id),
        "split_spec": _put(context, "ml.split_spec", split_payload, config.dataset_id),
        "model": _put(context, "ml.model", model_payload, config.dataset_id),
        "evaluation": _put(context, "ml.evaluation_report", evaluation_payload, config.dataset_id),
        "predictions": _put(
            context,
            "ml.predictions",
            predictions_payload,
            config.dataset_id,
            split["rows_test"],
        ),
        "training_log": training_log_artifact_id,
    }

    _publish(
        context,
        "ml.training_log.updated",
        {
            "artifact_id": training_log_artifact_id,
            "run_id": run_id,
            "status": "finished",
        },
    )

    run_payload = {
        "run_id": run_id,
        "status": "finished",
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": finished - started,
        "config": asdict(config),
        "metrics": result["metrics"],
        "artifact_ids": dict(artifact_ids),
    }

    artifact_ids["run"] = _put(context, "ml.run", run_payload, config.dataset_id)

    _publish(
        context,
        "ml.run.finished",
        {
            "run_id": run_id,
            "artifact_ids": artifact_ids,
            "training_log_artifact_id": artifact_ids.get("training_log"),
        },
    )

    return {
        "run_id": run_id,
        "metrics": result["metrics"],
        "extra": result.get("extra", {}),
        "prediction_preview": result["prediction_records"][:25],
        "artifact_ids": artifact_ids,
    }


def _train_sklearn_model(
    *,
    context: Any,
    config: TrainConfig,
    model_spec: ModelSpec,
    split: Dict[str, Any],
    run_id: str,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    from sklearn.pipeline import Pipeline

    _check_cancelled(cancel_token)

    params = {**model_spec.default_params, "random_state": config.random_state}
    estimator = model_spec.factory(params)
    
    tuning_result = _optuna_tune_sklearn_model(
        context=context,
        config=config,
        model_spec=model_spec,
        split=split,
        run_id=run_id,
        cancel_token=cancel_token,
    )

    runtime_params = {}
    if tuning_result.get("enabled"):
        runtime_params = dict(tuning_result.get("best_params") or {})

    preprocessor = _make_preprocessor(split["X_train"])
    estimator = model_spec.factory(runtime_params)
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(split["X_train"], split["y_train"])

    _check_cancelled(cancel_token)

    val_eval = _evaluate_estimator(
        task=config.task,
        estimator=pipeline,
        X=split["X_val"],
        y=split["y_val"],
        rows=split["rows_val"],
    )
    test_eval = _evaluate_estimator(
        task=config.task,
        estimator=pipeline,
        X=split["X_test"],
        y=split["y_test"],
        rows=split["rows_test"],
    )

    metrics = _prefix_metrics("val", val_eval["metrics"])
    metrics.update(_prefix_metrics("test", test_eval["metrics"]))

    training_log = [
        {
            "epoch": 1,
            "train_loss": None,
            "val_loss": None,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        }
    ]

    return {
        "model": pipeline,
        "metrics": metrics,
        "extra": {
            "validation": val_eval.get("extra", {}),
            "test": test_eval.get("extra", {}),
            "tuning": tuning_result,
        },
        "prediction_records": test_eval["prediction_records"],
        "training_log": training_log,
        "best_epoch": 1,
        "model_metadata": {
            "framework": "sklearn",
            "estimator": type(estimator).__name__,
            "tuning": tuning_result,
        },
    }


def _train_torch_model(
    *,
    context: Any,
    config: TrainConfig,
    model_spec: ModelSpec,
    split: Dict[str, Any],
    run_id: str,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    model = None

    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import LabelEncoder

        torch.manual_seed(int(config.random_state))

        preprocessor = _make_preprocessor(split["X_train"], scale_numeric=True)
        X_train = np.asarray(preprocessor.fit_transform(split["X_train"]), dtype=np.float32)
        X_val = np.asarray(preprocessor.transform(split["X_val"]), dtype=np.float32)
        X_test = np.asarray(preprocessor.transform(split["X_test"]), dtype=np.float32)

        input_dim = int(X_train.shape[1])
        model_params = dict(model_spec.default_params or {})

        hidden_layers = _parse_hidden_layers(
            model_params.get("hidden_layers", config.torch_hidden_layers)
        )
        dropout = float(model_params.get("dropout", 0.1))
        optimize_metric = str(model_params.get("optimize_metric", config.optimize_metric))

        label_encoder = None

        if config.task == "classification":
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(split["y_train"]).astype(np.int64)
            y_val = label_encoder.transform(split["y_val"]).astype(np.int64)
            y_test = label_encoder.transform(split["y_test"]).astype(np.int64)

            output_dim = int(len(label_encoder.classes_))
            model = _TorchMLP(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                output_dim=output_dim,
                dropout=dropout,
            )
            criterion = nn.CrossEntropyLoss()
        else:
            y_train = np.asarray(split["y_train"], dtype=np.float32).reshape(-1, 1)
            y_val = np.asarray(split["y_val"], dtype=np.float32).reshape(-1, 1)
            y_test = np.asarray(split["y_test"], dtype=np.float32).reshape(-1, 1)

            output_dim = 1
            model = _TorchMLP(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                output_dim=output_dim,
                dropout=dropout,
            )
            criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(model_params.get("learning_rate", config.torch_learning_rate)),
            weight_decay=float(model_params.get("weight_decay", config.torch_weight_decay)),
        )

        train_loader = _torch_loader(
            X_train,
            y_train,
            task=config.task,
            batch_size=int(model_params.get("batch_size", config.torch_batch_size)),
            shuffle=True,
        )

        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        if config.task == "classification":
            y_val_t = torch.tensor(y_val, dtype=torch.long)
        else:
            y_val_t = torch.tensor(y_val, dtype=torch.float32)

        best_score = None
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0
        training_log: List[Dict[str, Any]] = []

        epochs = int(model_params.get("epochs", config.torch_epochs))
        patience = int(model_params.get("patience", config.torch_patience))

        _publish_training_log_update(
            context,
            artifact_id=config.training_log_artifact_id,
            run_id=run_id,
            status="running",
            message="Torch training initialised. Waiting for first epoch...",
            epochs=training_log,
            best_epoch=best_epoch,
        )

        for epoch in range(1, epochs + 1):
            _check_cancelled(cancel_token)

            model.train()
            batch_losses = []

            for X_batch, y_batch in train_loader:
                _check_cancelled(cancel_token)

                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

            model.eval()
            with torch.no_grad():
                val_output = model(X_val_t)
                val_loss = float(criterion(val_output, y_val_t).detach().cpu().item())

            val_metrics = _torch_metrics(
                task=config.task,
                output=val_output.detach().cpu().numpy(),
                y_true_raw=split["y_val"],
                label_encoder=label_encoder,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            }
            training_log.append(row)

            score = _score_for_optimisation(row, optimize_metric)
            if _is_better(score, best_score, optimize_metric):
                best_score = score
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            _publish_training_log_update(
                context,
                artifact_id=config.training_log_artifact_id,
                run_id=run_id,
                status="running",
                message=f"Finished epoch {epoch}.",
                epochs=training_log,
                best_epoch=best_epoch,
            )

            if patience > 0 and epochs_without_improvement >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        _publish_training_log_update(
            context,
            artifact_id=config.training_log_artifact_id,
            run_id=run_id,
            status="running",
            message="Evaluating best epoch on validation/test sets...",
            epochs=training_log,
            best_epoch=best_epoch,
        )

        val_eval = _evaluate_torch_model(
            task=config.task,
            model=model,
            X_tensor=X_val_t,
            y_raw=split["y_val"],
            rows=split["rows_val"],
            label_encoder=label_encoder,
        )
        test_eval = _evaluate_torch_model(
            task=config.task,
            model=model,
            X_tensor=X_test_t,
            y_raw=split["y_test"],
            rows=split["rows_test"],
            label_encoder=label_encoder,
        )

        metrics = _prefix_metrics("val", val_eval["metrics"])
        metrics.update(_prefix_metrics("test", test_eval["metrics"]))
        metrics["best_epoch"] = float(best_epoch)

        return {
            "model": {
                "torch_model": model,
                "preprocessor": preprocessor,
                "label_encoder": label_encoder,
                "input_dim": input_dim,
                "hidden_layers": hidden_layers,
                "output_dim": output_dim,
            },
            "metrics": metrics,
            "extra": {
                "validation": val_eval.get("extra", {}),
                "test": test_eval.get("extra", {}),
            },
            "prediction_records": test_eval["prediction_records"],
            "training_log": training_log,
            "best_epoch": best_epoch,
            "model_metadata": {
                "framework": "torch",
                "input_dim": input_dim,
                "hidden_layers": hidden_layers,
                "output_dim": output_dim,
            },
        }

    finally:
        _release_torch_cuda(model)


class _TorchMLP:
    def __new__(
        cls,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        import torch.nn as nn

        layers: List[Any] = []
        last_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))

            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))

            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

def _torch_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    batch_size: int,
    shuffle: bool,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X_t = torch.tensor(X, dtype=torch.float32)

    if task == "classification":
        y_t = torch.tensor(y, dtype=torch.long)
    else:
        y_t = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _torch_metrics(
    *,
    task: str,
    output: np.ndarray,
    y_true_raw: pd.Series,
    label_encoder: Any,
) -> Dict[str, float]:
    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score

        pred_idx = np.argmax(output, axis=1)
        pred_labels = label_encoder.inverse_transform(pred_idx)

        return {
            "val_accuracy": float(accuracy_score(y_true_raw, pred_labels)),
            "val_f1_macro": float(f1_score(y_true_raw, pred_labels, average="macro", zero_division=0)),
        }

    from sklearn.metrics import mean_absolute_error, r2_score

    y_pred = output.reshape(-1)
    y_true = np.asarray(y_true_raw, dtype=np.float32)

    return {
        "val_mae": float(mean_absolute_error(y_true, y_pred)),
        "val_r2": float(r2_score(y_true, y_pred)),
    }


def _evaluate_torch_model(
    *,
    task: str,
    model: Any,
    X_tensor: Any,
    y_raw: pd.Series,
    rows: List[Any],
    label_encoder: Any,
) -> Dict[str, Any]:
    import torch

    model.eval()
    with torch.no_grad():
        output = model(X_tensor).detach().cpu().numpy()

    if task == "classification":
        exp = np.exp(output - np.max(output, axis=1, keepdims=True))
        proba = exp / np.sum(exp, axis=1, keepdims=True)
        pred_idx = np.argmax(proba, axis=1)
        y_pred = label_encoder.inverse_transform(pred_idx)
        labels = [str(c) for c in label_encoder.classes_]

        metrics, extra = _metrics(
            task=task,
            y_true=y_raw,
            y_pred=y_pred,
            y_proba=proba,
        )

        records = _prediction_records(
            task=task,
            row_ids=rows,
            y_true=y_raw,
            y_pred=y_pred,
            y_proba=proba,
            labels=labels,
        )

        return {"metrics": metrics, "extra": extra, "prediction_records": records}

    y_pred = output.reshape(-1)

    metrics, extra = _metrics(
        task=task,
        y_true=y_raw,
        y_pred=y_pred,
        y_proba=None,
    )

    records = _prediction_records(
        task=task,
        row_ids=rows,
        y_true=y_raw,
        y_pred=y_pred,
        y_proba=None,
        labels=[],
    )

    return {"metrics": metrics, "extra": extra, "prediction_records": records}


def _evaluate_estimator(
    *,
    task: str,
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    rows: List[Any],
) -> Dict[str, Any]:
    y_pred = estimator.predict(X)
    y_proba = _predict_proba(estimator, X)
    labels = _classes(estimator)

    metrics, extra = _metrics(
        task=task,
        y_true=y,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    records = _prediction_records(
        task=task,
        row_ids=rows,
        y_true=y,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=labels,
    )

    return {"metrics": metrics, "extra": extra, "prediction_records": records}


def _make_preprocessor(X: pd.DataFrame, *, scale_numeric: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_columns = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []

    if numeric_columns:
        transformers.append(("num", Pipeline(numeric_steps), numeric_columns))

    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _split_train_validation_test(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    row_ids: List[Any],
    task: str,
    test_size: float,
    validation_size: float,
    random_state: int,
    stratify: bool,
) -> Dict[str, Any]:
    from sklearn.model_selection import train_test_split

    stratify_full = _stratify_or_none(y, task=task, enabled=stratify)

    X_remaining, X_test, y_remaining, y_test, rows_remaining, rows_test = train_test_split(
        X,
        y,
        row_ids,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=stratify_full,
    )

    validation_fraction_of_remaining = float(validation_size) / (1.0 - float(test_size))
    stratify_remaining = _stratify_or_none(y_remaining, task=task, enabled=stratify)

    X_train, X_val, y_train, y_val, rows_train, rows_val = train_test_split(
        X_remaining,
        y_remaining,
        rows_remaining,
        test_size=validation_fraction_of_remaining,
        random_state=int(random_state),
        stratify=stratify_remaining,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "rows_train": rows_train,
        "rows_val": rows_val,
        "rows_test": rows_test,
    }


def _stratify_or_none(y: pd.Series, *, task: str, enabled: bool):
    if task != "classification" or not enabled:
        return None

    counts = y.value_counts(dropna=False)
    if len(counts) <= 1:
        return None
    if counts.min() < 2:
        return None

    return y


def _metrics(
    *,
    task: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if task == "classification":
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

        extra: Dict[str, Any] = {
            "classification_report": classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics["roc_auc_ovr_macro"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                    )
            except Exception:
                pass

        return metrics, extra

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return (
        {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        },
        {},
    )


def _predict_proba(estimator: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    if hasattr(estimator, "predict_proba"):
        try:
            return estimator.predict_proba(X)
        except Exception:
            return None
    return None


def _classes(estimator: Any) -> List[str]:
    try:
        model = estimator.named_steps.get("model")
        classes = getattr(model, "classes_", [])
    except Exception:
        classes = getattr(estimator, "classes_", [])

    return [str(c) for c in classes]


def _prediction_records(
    *,
    task: str,
    row_ids: List[Any],
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    labels: List[str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for i, row_id in enumerate(row_ids):
        record: Dict[str, Any] = {
            "row_id": _json_value(row_id),
            "y_true": _json_value(y_true.iloc[i]),
            "y_pred": _json_value(y_pred[i]),
        }

        if task == "classification" and y_proba is not None:
            for j, label in enumerate(labels):
                record[f"proba_{label}"] = float(y_proba[i, j])
            record["confidence"] = float(np.max(y_proba[i]))

        records.append(record)

    return records


def _mapped_column(context: Any, dataset_id: str, mapping_name: str) -> Optional[str]:
    try:
        return context.datasets.get_mapping(dataset_id, mapping_name)
    except Exception:
        return None


def _row_ids(context: Any, dataset_id: str, df: pd.DataFrame) -> List[Any]:
    record_id_column = _mapped_column(context, dataset_id, "record_id")

    if record_id_column and record_id_column in df.columns:
        return [_json_value(v) for v in df[record_id_column].tolist()]

    return [_json_value(v) for v in df.index.tolist()]


def _put(
    context: Any,
    artifact_type: str,
    payload: Dict[str, Any],
    dataset_id: str,
    row_ids: Optional[List[Any]] = None,
) -> Optional[str]:
    artifacts = getattr(context, "artifacts", None)
    put = getattr(artifacts, "put", None)

    if not callable(put):
        return None

    return put(
        artifact_type,
        payload,
        dataset_id=dataset_id,
        row_ids=row_ids,
        params={"run_id": payload.get("run_id")},
    )

def _update_training_log_artifact(
    context: Any,
    artifact_id: Optional[str],
    updates: Dict[str, Any],
) -> None:
    if not artifact_id:
        return

    artifacts = getattr(context, "artifacts", None)
    get = getattr(artifacts, "get", None)

    if not callable(get):
        return

    try:
        payload = get(artifact_id)
    except Exception:
        return

    if isinstance(payload, dict):
        payload.update(updates)


def _publish_training_log_update(
    context: Any,
    *,
    artifact_id: Optional[str],
    run_id: str,
    status: str,
    message: str,
    epochs: Optional[List[Dict[str, Any]]] = None,
    best_epoch: Optional[int] = None,
) -> None:
    if not artifact_id:
        return

    now = time.time()

    updates: Dict[str, Any] = {
        "status": status,
        "message": message,
        "updated_at": now,
    }

    if epochs is not None:
        updates["epochs"] = list(epochs)
        updates["last_epoch"] = epochs[-1].get("epoch") if epochs else None

    if best_epoch is not None:
        updates["best_epoch"] = best_epoch

    _update_training_log_artifact(context, artifact_id, updates)

    _publish(
        context,
        "ml.training_log.updated",
        {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "status": status,
            "message": message,
            "best_epoch": best_epoch,
            "last_epoch": updates.get("last_epoch"),
        },
    )

def _publish(context: Any, topic: str, payload: Dict[str, Any]) -> None:
    events = getattr(context, "events", None)
    publish = getattr(events, "publish", None)

    if callable(publish):
        publish(topic, payload)


def _check_cancelled(cancel_token: Any) -> None:
    if cancel_token is None:
        return

    is_cancelled = getattr(cancel_token, "is_cancelled", None)
    if callable(is_cancelled) and is_cancelled():
        raise RuntimeError("Training cancelled.")

    cancelled = getattr(cancel_token, "cancelled", None)
    if callable(cancelled):
        if cancelled():
            raise RuntimeError("Training cancelled.")
        return

    if isinstance(cancelled, bool) and cancelled:
        raise RuntimeError("Training cancelled.")


def _parse_hidden_layers(value: str) -> List[int]:
    layers = []

    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        layers.append(max(1, int(part)))

    return layers or [128, 64]


def _prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _score_for_optimisation(row: Dict[str, Any], metric: str) -> Optional[float]:
    value = row.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _is_better(new_score: Optional[float], best_score: Optional[float], metric: str) -> bool:
    if new_score is None:
        return False
    if best_score is None:
        return True

    direction = _metric_direction(metric)
    if direction == "min":
        return new_score < best_score
    return new_score > best_score


def _metric_direction(metric: str) -> str:
    lowered = metric.lower()
    if "loss" in lowered or "error" in lowered or "mae" in lowered or "rmse" in lowered:
        return "min"
    return "max"

def _release_torch_cuda(model: Any = None) -> None:
    """Best-effort torch cleanup for completion, error, or cancellation."""

    try:
        if model is not None and hasattr(model, "to"):
            try:
                model.to("cpu")
            except Exception:
                pass

        import gc
        import torch

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass