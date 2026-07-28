from __future__ import annotations

from typing import Any

from ..harnesses.sklearn import SklearnRecipe, fit_warm_start
from ..harnesses.sklearn_incremental import IncrementalSklearnRecipe
from ..harnesses.xgboost_external import (
    ExternalMemoryXGBoostModel,
    ExternalMemoryXGBoostRecipe,
)


_FEATURE_COLUMNS_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
    "title": "Input feature columns",
    "description": (
        "Dataset columns used as model inputs. Choose these explicitly for "
        "reproducible ML/AL runs."
    ),
    "default": [],
    "x-widget": "column_multichoice",
}

_RECORD_ID_COLUMN_SCHEMA = {
    "type": "string",
    "title": "Record ID column",
    "description": "Stable row/object identifier column.",
    "default": "",
    "x-widget": "column_select",
}

_TARGET_COLUMN_SCHEMA = {
    "type": "string",
    "title": "Target / label column",
    "description": "Training target column.",
    "default": "",
    "x-widget": "column_select",
}

_AUTO_FEATURE_COLUMNS_SCHEMA = {
    "type": "boolean",
    "title": "Auto-select feature columns if none are chosen",
    "description": (
        "Fallback only. Explicit feature selection is recommended, especially "
        "for active learning."
    ),
    "default": False,
}

_COMMON_BINDING_PROPERTIES = {
    "record_id_column": _RECORD_ID_COLUMN_SCHEMA,
    "target_column": _TARGET_COLUMN_SCHEMA,
    "feature_columns": _FEATURE_COLUMNS_SCHEMA,
    "auto_feature_columns": _AUTO_FEATURE_COLUMNS_SCHEMA,
}

_COMMON_OUTPUTS = [
    "ml.split_spec",
    "ml.model",
    "ml.evaluation_report",
    "ml.predictions",
    "ml.training_log",
    "ml.run",
]


class SklearnTabularClassifierRecipe(SklearnRecipe):
    id = "core.ml.sklearn_tabular_classifier"
    title = "Sklearn tabular classifier"
    version = "0.2.0"
    task = "classification"
    modality = "tabular"
    framework = "sklearn"
    complexity = "baseline"
    author = "AstronomicAL"
    description = (
        "Materialised sklearn baseline for tabular classification. Supports "
        "Random Forest, Extra Trees, HistGradientBoosting, and Logistic "
        "Regression. The launcher estimates peak memory and blocks unsafe runs."
    )
    tags = [
        "sklearn",
        "tabular",
        "classification",
        "baseline",
        "materialized",
    ]
    required_mappings = ["record_id"]
    optional_mappings = ["target_label"]
    produces = list(_COMMON_OUTPUTS)

    params_schema = {
        "type": "object",
        "required": ["feature_columns"],
        "properties": {
            **_COMMON_BINDING_PROPERTIES,
            "model_type": {
                "type": "string",
                "title": "Sklearn model",
                "enum": [
                    "random_forest",
                    "extra_trees",
                    "hist_gradient_boosting",
                    "logistic_regression",
                ],
                "default": "random_forest",
            },
            "n_estimators": {
                "type": "integer",
                "title": "Trees / estimators",
                "default": 300,
                "minimum": 1,
            },
            "max_depth": {
                "type": "integer",
                "title": "Max depth",
                "default": 0,
                "minimum": 0,
                "description": "0 means sklearn default / unlimited where supported.",
            },
            "max_iter": {
                "type": "integer",
                "title": "Max iterations",
                "default": 300,
                "minimum": 1,
            },
            "learning_rate": {
                "type": "number",
                "title": "Learning rate",
                "default": 0.1,
            },
            "C": {
                "type": "number",
                "title": "Logistic regression C",
                "default": 1.0,
            },
            "class_weight": {
                "type": "string",
                "title": "Class weight",
                "enum": ["", "balanced"],
                "default": "",
            },
            "n_jobs": {
                "type": "integer",
                "title": "CPU jobs",
                "default": -1,
            },
            "random_state": {
                "type": "integer",
                "title": "Random state",
                "default": 42,
            },
            "live_curve": {
                "type": "boolean",
                "title": "Report validation curve for tree ensembles",
                "default": True,
            },
            "curve_points": {
                "type": "integer",
                "title": "Curve points",
                "default": 10,
                "minimum": 2,
            },
        },
    }

    def build_model(self, run, *, num_classes: int):
        model_type = str(run.params.get("model_type") or "random_forest")
        random_state = int(run.params.get("random_state", 42))
        n_jobs = int(run.params.get("n_jobs", -1))
        n_estimators = int(run.params.get("n_estimators", 300))
        max_depth_value = int(run.params.get("max_depth", 0) or 0)
        max_depth = None if max_depth_value <= 0 else max_depth_value
        class_weight = str(run.params.get("class_weight") or "").strip() or None

        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=n_jobs,
            )

        if model_type == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier

            return ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=n_jobs,
            )

        if model_type == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier

            kwargs: dict[str, Any] = {
                "max_iter": int(run.params.get("max_iter", 300)),
                "learning_rate": float(run.params.get("learning_rate", 0.1)),
                "random_state": random_state,
            }
            if max_depth is not None:
                kwargs["max_depth"] = max_depth
            return HistGradientBoostingClassifier(**kwargs)

        if model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                C=float(run.params.get("C", 1.0)),
                max_iter=int(run.params.get("max_iter", 300)),
                class_weight=class_weight,
                n_jobs=n_jobs,
                random_state=random_state,
            )

        raise ValueError(f"Unknown sklearn classifier model_type: {model_type!r}")

    def fit(self, run, *, model, components, train_loader, harness):
        try:
            has_n_estimators = "n_estimators" in model.get_params(deep=False)
        except Exception:
            has_n_estimators = False

        if bool(run.params.get("live_curve", True)) and has_n_estimators:
            return fit_warm_start(
                run,
                model=model,
                components=components,
                train_loader=train_loader,
                harness=harness,
                points=int(run.params.get("curve_points", 10)),
            )

        return super().fit(
            run,
            model=model,
            components=components,
            train_loader=train_loader,
            harness=harness,
        )


class SklearnTabularRegressorRecipe(SklearnRecipe):
    id = "core.ml.sklearn_tabular_regressor"
    title = "Sklearn tabular regressor"
    version = "0.2.0"
    task = "regression"
    modality = "tabular"
    framework = "sklearn"
    complexity = "baseline"
    author = "AstronomicAL"
    description = (
        "Materialised sklearn baseline for tabular regression. Supports Random "
        "Forest, Extra Trees, HistGradientBoosting, and Ridge regression. The "
        "launcher estimates peak memory and blocks unsafe runs."
    )
    tags = [
        "sklearn",
        "tabular",
        "regression",
        "baseline",
        "photoz",
        "materialized",
    ]
    required_mappings = ["record_id"]
    optional_mappings = ["target_label"]
    produces = list(_COMMON_OUTPUTS)

    params_schema = {
        "type": "object",
        "required": ["feature_columns"],
        "properties": {
            **_COMMON_BINDING_PROPERTIES,
            "model_type": {
                "type": "string",
                "title": "Sklearn model",
                "enum": [
                    "random_forest",
                    "extra_trees",
                    "hist_gradient_boosting",
                    "ridge",
                ],
                "default": "random_forest",
            },
            "n_estimators": {
                "type": "integer",
                "title": "Trees / estimators",
                "default": 300,
                "minimum": 1,
            },
            "max_depth": {
                "type": "integer",
                "title": "Max depth",
                "default": 0,
                "minimum": 0,
                "description": "0 means sklearn default / unlimited where supported.",
            },
            "max_iter": {
                "type": "integer",
                "title": "Max iterations",
                "default": 300,
                "minimum": 1,
            },
            "learning_rate": {
                "type": "number",
                "title": "Learning rate",
                "default": 0.1,
            },
            "alpha": {
                "type": "number",
                "title": "Ridge alpha",
                "default": 1.0,
            },
            "n_jobs": {
                "type": "integer",
                "title": "CPU jobs",
                "default": -1,
            },
            "random_state": {
                "type": "integer",
                "title": "Random state",
                "default": 42,
            },
            "live_curve": {
                "type": "boolean",
                "title": "Report validation curve for tree ensembles",
                "default": True,
            },
            "curve_points": {
                "type": "integer",
                "title": "Curve points",
                "default": 10,
                "minimum": 2,
            },
        },
    }

    def build_model(self, run, *, num_classes: int):
        model_type = str(run.params.get("model_type") or "random_forest")
        random_state = int(run.params.get("random_state", 42))
        n_jobs = int(run.params.get("n_jobs", -1))
        n_estimators = int(run.params.get("n_estimators", 300))
        max_depth_value = int(run.params.get("max_depth", 0) or 0)
        max_depth = None if max_depth_value <= 0 else max_depth_value

        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=n_jobs,
            )

        if model_type == "extra_trees":
            from sklearn.ensemble import ExtraTreesRegressor

            return ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=n_jobs,
            )

        if model_type == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingRegressor

            kwargs: dict[str, Any] = {
                "max_iter": int(run.params.get("max_iter", 300)),
                "learning_rate": float(run.params.get("learning_rate", 0.1)),
                "random_state": random_state,
            }
            if max_depth is not None:
                kwargs["max_depth"] = max_depth
            return HistGradientBoostingRegressor(**kwargs)

        if model_type == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=float(run.params.get("alpha", 1.0)))

        raise ValueError(f"Unknown sklearn regressor model_type: {model_type!r}")

    def fit(self, run, *, model, components, train_loader, harness):
        try:
            has_n_estimators = "n_estimators" in model.get_params(deep=False)
        except Exception:
            has_n_estimators = False

        if bool(run.params.get("live_curve", True)) and has_n_estimators:
            return fit_warm_start(
                run,
                model=model,
                components=components,
                train_loader=train_loader,
                harness=harness,
                points=int(run.params.get("curve_points", 10)),
            )

        return super().fit(
            run,
            model=model,
            components=components,
            train_loader=train_loader,
            harness=harness,
        )


_INCREMENTAL_COMMON_PROPERTIES = {
    **_COMMON_BINDING_PROPERTIES,
    "epochs": {
        "type": "integer",
        "title": "Passes over the training partition",
        "default": 5,
        "minimum": 1,
    },
    "alpha": {
        "type": "number",
        "title": "Regularisation strength",
        "default": 0.0001,
        "minimum": 0.0,
    },
    "penalty": {
        "type": "string",
        "enum": ["l2", "l1", "elasticnet"],
        "default": "l2",
    },
    "l1_ratio": {
        "type": "number",
        "default": 0.15,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "learning_rate": {
        "type": "string",
        "enum": ["optimal", "constant", "invscaling", "adaptive"],
        "default": "optimal",
    },
    "eta0": {
        "type": "number",
        "default": 0.01,
        "minimum": 0.0,
    },
    "average": {
        "type": "boolean",
        "default": False,
    },
    "random_state": {
        "type": "integer",
        "default": 42,
    },
    "stream_source_batch_size": {
        "type": "integer",
        "title": "Source batch rows",
        "default": 8192,
        "minimum": 1,
    },
}


class IncrementalSGDClassifierRecipe(IncrementalSklearnRecipe):
    id = "core.ml.sklearn_incremental_sgd_classifier"
    title = "Incremental SGD classifier"
    version = "0.1.0"
    task = "classification"
    modality = "tabular"
    framework = "sklearn"
    complexity = "baseline"
    author = "AstronomicAL"
    description = (
        "Numeric tabular classifier trained with StandardScaler.partial_fit and "
        "SGDClassifier.partial_fit over bounded DatasetSource batches."
    )
    tags = [
        "sklearn",
        "incremental",
        "partial-fit",
        "tabular",
        "classification",
    ]
    required_mappings = ["record_id"]
    optional_mappings = ["target_label"]
    produces = list(_COMMON_OUTPUTS)
    params_schema = {
        "type": "object",
        "required": ["feature_columns"],
        "properties": {
            **_INCREMENTAL_COMMON_PROPERTIES,
            "loss": {
                "type": "string",
                "enum": ["log_loss", "modified_huber"],
                "default": "log_loss",
            },
        },
    }

    def build_model(self, run, *, num_classes: int):
        from sklearn.linear_model import SGDClassifier

        return SGDClassifier(
            loss=str(run.params.get("loss") or "log_loss"),
            penalty=str(run.params.get("penalty") or "l2"),
            alpha=float(run.params.get("alpha") or 0.0001),
            l1_ratio=float(run.params.get("l1_ratio") or 0.15),
            learning_rate=str(run.params.get("learning_rate") or "optimal"),
            eta0=float(run.params.get("eta0") or 0.01),
            average=bool(run.params.get("average", False)),
            random_state=int(run.params.get("random_state") or 42),
        )


class IncrementalSGDRegressorRecipe(IncrementalSklearnRecipe):
    id = "core.ml.sklearn_incremental_sgd_regressor"
    title = "Incremental SGD regressor"
    version = "0.1.0"
    task = "regression"
    modality = "tabular"
    framework = "sklearn"
    complexity = "baseline"
    author = "AstronomicAL"
    description = (
        "Numeric tabular regressor trained with StandardScaler.partial_fit and "
        "SGDRegressor.partial_fit over bounded DatasetSource batches."
    )
    tags = [
        "sklearn",
        "incremental",
        "partial-fit",
        "tabular",
        "regression",
    ]
    required_mappings = ["record_id"]
    optional_mappings = ["target_label"]
    produces = list(_COMMON_OUTPUTS)
    params_schema = {
        "type": "object",
        "required": ["feature_columns"],
        "properties": {
            **_INCREMENTAL_COMMON_PROPERTIES,
            "loss": {
                "type": "string",
                "enum": ["squared_error", "huber", "epsilon_insensitive"],
                "default": "squared_error",
            },
            "epsilon": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
            },
        },
    }

    def build_model(self, run, *, num_classes: int):
        from sklearn.linear_model import SGDRegressor

        return SGDRegressor(
            loss=str(run.params.get("loss") or "squared_error"),
            penalty=str(run.params.get("penalty") or "l2"),
            alpha=float(run.params.get("alpha") or 0.0001),
            l1_ratio=float(run.params.get("l1_ratio") or 0.15),
            learning_rate=str(run.params.get("learning_rate") or "optimal"),
            eta0=float(run.params.get("eta0") or 0.01),
            epsilon=float(run.params.get("epsilon") or 0.1),
            average=bool(run.params.get("average", False)),
            random_state=int(run.params.get("random_state") or 42),
        )


_XGBOOST_COMMON_PROPERTIES = {
    **_COMMON_BINDING_PROPERTIES,
    "n_estimators": {
        "type": "integer",
        "title": "Boosting rounds",
        "default": 800,
        "minimum": 1,
    },
    "curve_points": {
        "type": "integer",
        "title": "Validation curve points",
        "default": 10,
        "minimum": 1,
        "maximum": 50,
    },
    "max_depth": {"type": "integer", "default": 6, "minimum": 1},
    "learning_rate": {
        "type": "number",
        "default": 0.05,
        "minimum": 1e-6,
    },
    "subsample": {
        "type": "number",
        "default": 0.8,
        "minimum": 0.05,
        "maximum": 1.0,
    },
    "colsample_bytree": {
        "type": "number",
        "default": 0.8,
        "minimum": 0.05,
        "maximum": 1.0,
    },
    "min_child_weight": {"type": "number", "default": 1.0, "minimum": 0.0},
    "reg_alpha": {"type": "number", "default": 0.0, "minimum": 0.0},
    "reg_lambda": {"type": "number", "default": 1.0, "minimum": 0.0},
    "tree_method": {
        "type": "string",
        "enum": ["hist", "approx"],
        "default": "hist",
    },
    "n_jobs": {"type": "integer", "default": -1},
    "random_state": {"type": "integer", "default": 42},
    "stream_source_batch_size": {
        "type": "integer",
        "title": "Source-to-cache batch rows",
        "default": 8192,
        "minimum": 1,
    },
}


class XGBoostTabularClassifierRecipe(ExternalMemoryXGBoostRecipe):
    id = "core.ml.xgboost_tabular_classifier"
    title = "XGBoost external-memory classifier"
    version = "0.2.0"
    task = "classification"
    modality = "tabular"
    framework = "sklearn"
    complexity = "advanced"
    author = "AstronomicAL"
    description = (
        "Numeric gradient-boosted-tree classifier trained from disk-backed "
        "XGBoost DMatrix caches rather than a fully materialised pandas matrix."
    )
    tags = [
        "xgboost",
        "external-memory",
        "gbdt",
        "tabular",
        "classification",
    ]
    required_imports = ["sklearn", "joblib", "xgboost"]
    source_urls = ["https://github.com/dmlc/xgboost"]
    source_reference = (
        "Uses XGBoost's official external-memory DMatrix cache format."
    )
    required_mappings = ["record_id"]
    optional_mappings = ["target_label"]
    produces = list(_COMMON_OUTPUTS)
    params_schema = {
        "type": "object",
        "required": ["feature_columns"],
        "properties": dict(_XGBOOST_COMMON_PROPERTIES),
    }

    def build_model(self, run, *, target=None, num_classes: int = 0):
        classes = list(getattr(target, "classes", []) or [])
        return ExternalMemoryXGBoostModel(
            task="classification",
            classes=classes,
        )


class XGBoostTabularRegressorRecipe(ExternalMemoryXGBoostRecipe):
    id = "core.ml.xgboost_tabular_regressor"
    title = "XGBoost external-memory regressor"
    version = "0.2.0"
    task = "regression"
    modality = "tabular"
    framework = "sklearn"
    complexity = "advanced"
    author = "AstronomicAL"
    description = (
        "Numeric XGBoost regressor trained from disk-backed DMatrix caches "
        "under AstronomicAL's managed split and evaluation protocol."
    )
    tags = [
        "xgboost",
        "external-memory",
        "gbdt",
        "tabular",
        "regression",
    ]
    required_imports = ["sklearn", "joblib", "xgboost"]
    source_urls = ["https://github.com/dmlc/xgboost"]
    source_reference = (
        "Uses XGBoost's official external-memory DMatrix cache format."
    )
    required_mappings = ["record_id"]
    optional_mappings = ["target_label"]
    produces = list(_COMMON_OUTPUTS)
    params_schema = {
        "type": "object",
        "required": ["feature_columns"],
        "properties": {
            **_XGBOOST_COMMON_PROPERTIES,
            "objective": {
                "type": "string",
                "enum": [
                    "reg:squarederror",
                    "reg:pseudohubererror",
                    "reg:absoluteerror",
                ],
                "default": "reg:squarederror",
            },
        },
    }

    def build_model(self, run, *, target=None, num_classes: int = 0):
        return ExternalMemoryXGBoostModel(
            task="regression",
            classes=[],
        )
