from __future__ import annotations

from typing import Any

import numpy as np

from ..harnesses.sklearn import SklearnRecipe, fit_warm_start

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

class SklearnTabularClassifierRecipe(SklearnRecipe):
    id = "core.ml.sklearn_tabular_classifier"
    title = "Sklearn tabular classifier"
    version = "0.1.0"
    task = "classification"
    modality = "tabular"
    framework = "sklearn"
    complexity = "baseline"
    author = "AstronomicAL"
    description = (
        "Protocol-managed sklearn baseline for tabular classification. "
        "Supports Random Forest, Extra Trees, HistGradientBoosting, and "
        "Logistic Regression. The harness owns train/validation/test protocol, "
        "preprocessing, metrics, artifacts, and prediction contracts."
    )
    tags = ["sklearn", "tabular", "classification", "baseline"]
    required_mappings = ["record_id"]
    optional_mappings = ["target_label"]
    produces = [
        "ml.split_spec",
        "ml.model",
        "ml.evaluation_report",
        "ml.predictions",
        "ml.training_log",
        "ml.run",
    ]

    params_schema = {
        "type": "object",
        "required": ["feature_columns"],
        "properties": {
            "record_id_column": _RECORD_ID_COLUMN_SCHEMA,
            "target_column": _TARGET_COLUMN_SCHEMA,
            "feature_columns": _FEATURE_COLUMNS_SCHEMA,
            "auto_feature_columns": _AUTO_FEATURE_COLUMNS_SCHEMA,
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
    version = "0.1.0"
    task = "regression"
    modality = "tabular"
    framework = "sklearn"
    complexity = "baseline"
    author = "AstronomicAL"
    description = (
        "Protocol-managed sklearn baseline for tabular regression. Supports "
        "Random Forest, Extra Trees, HistGradientBoosting, and Ridge regression. "
        "The harness owns train/validation/test protocol, preprocessing, metrics, "
        "artifacts, and prediction contracts."
    )
    tags = ["sklearn", "tabular", "regression", "baseline", "photoz"]
    required_mappings = ["record_id"]
    optional_mappings = ["target_label"]
    produces = [
        "ml.split_spec",
        "ml.model",
        "ml.evaluation_report",
        "ml.predictions",
        "ml.training_log",
        "ml.run",
    ]

    params_schema = {
        "type": "object",
        "required": ["feature_columns"],
        "properties": {
            "record_id_column": _RECORD_ID_COLUMN_SCHEMA,
            "target_column": _TARGET_COLUMN_SCHEMA,
            "feature_columns": _FEATURE_COLUMNS_SCHEMA,
            "auto_feature_columns": _AUTO_FEATURE_COLUMNS_SCHEMA,
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


# =============================================================================
# XGBoost recipes use the sklearn-compatible harness and preprocessing path.
# =============================================================================

_XGBOOST_COMMON_PROPERTIES = {
    "record_id_column": _RECORD_ID_COLUMN_SCHEMA,
    "target_column": _TARGET_COLUMN_SCHEMA,
    "feature_columns": _FEATURE_COLUMNS_SCHEMA,
    "auto_feature_columns": _AUTO_FEATURE_COLUMNS_SCHEMA,
    "n_estimators": {"type": "integer", "default": 800, "minimum": 1},
    "max_depth": {"type": "integer", "default": 6, "minimum": 1},
    "learning_rate": {"type": "number", "default": 0.05, "minimum": 1e-6},
    "subsample": {"type": "number", "default": 0.8, "minimum": 0.05, "maximum": 1.0},
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
        "enum": ["hist", "approx", "exact"],
        "default": "hist",
    },
    "n_jobs": {"type": "integer", "default": -1},
    "random_state": {"type": "integer", "default": 42},
}

class StringLabelXGBClassifier:
    """Small sklearn-compatible wrapper that preserves string class labels."""

    def __init__(self, **params):
        self.params = dict(params)
        self.model = None
        self.classes_ = np.asarray([], dtype=object)

    def get_params(self, deep: bool = True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(self, X, y):
        from xgboost import XGBClassifier

        labels = np.asarray(y).astype(str)
        self.classes_ = np.asarray(sorted(set(labels.tolist())), dtype=object)
        mapping = {label: index for index, label in enumerate(self.classes_)}
        encoded = np.asarray([mapping[label] for label in labels], dtype=np.int64)
        params = dict(self.params)
        params["objective"] = "binary:logistic" if len(self.classes_) == 2 else "multi:softprob"
        params["eval_metric"] = "logloss" if len(self.classes_) == 2 else "mlogloss"
        if len(self.classes_) > 2:
            params["num_class"] = len(self.classes_)
        self.model = XGBClassifier(**params)
        self.model.fit(X, encoded)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("XGBoost classifier has not been fitted.")
        encoded = np.asarray(self.model.predict(X), dtype=np.int64)
        return self.classes_[encoded]

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("XGBoost classifier has not been fitted.")
        return self.model.predict_proba(X)


class XGBoostTabularClassifierRecipe(SklearnRecipe):
    id = "core.ml.xgboost_tabular_classifier"
    title = "XGBoost tabular classifier"
    version = "0.1.0"
    task = "classification"
    modality = "tabular"
    framework = "sklearn"
    complexity = "advanced"
    author = "AstronomicAL"
    description = (
        "Strong gradient-boosted-tree classifier using XGBoost's sklearn API. "
        "AstronomicAL fits preprocessing on the training split only and owns all "
        "validation/test evaluation and artifact generation."
    )
    tags = ["xgboost", "gbdt", "tabular", "classification"]
    source_urls = ["https://github.com/dmlc/xgboost"]
    source_reference = "Uses XGBoost's histogram tree method and regularized subsampled boosting defaults."
    required_mappings = ["record_id"]
    optional_mappings = ["target_label"]
    produces = ["ml.split_spec", "ml.model", "ml.evaluation_report", "ml.predictions", "ml.training_log", "ml.run"]
    params_schema = {
        "type": "object",
        "required": ["feature_columns"],
        "properties": dict(_XGBOOST_COMMON_PROPERTIES),
    }

    def build_model(self, run, *, num_classes: int):
        p = run.params
        return StringLabelXGBClassifier(
            n_estimators=int(p.get("n_estimators", 800)),
            max_depth=int(p.get("max_depth", 6)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            subsample=float(p.get("subsample", 0.8)),
            colsample_bytree=float(p.get("colsample_bytree", 0.8)),
            min_child_weight=float(p.get("min_child_weight", 1.0)),
            reg_alpha=float(p.get("reg_alpha", 0.0)),
            reg_lambda=float(p.get("reg_lambda", 1.0)),
            tree_method=str(p.get("tree_method", "hist")),
            n_jobs=int(p.get("n_jobs", -1)),
            random_state=int(p.get("random_state", 42)),
        )


class XGBoostTabularRegressorRecipe(SklearnRecipe):
    id = "core.ml.xgboost_tabular_regressor"
    title = "XGBoost tabular regressor"
    version = "0.1.0"
    task = "regression"
    modality = "tabular"
    framework = "sklearn"
    complexity = "advanced"
    author = "AstronomicAL"
    description = (
        "Regularized histogram-based XGBoost regressor for strong tabular "
        "performance under AstronomicAL's managed split and evaluation protocol."
    )
    tags = ["xgboost", "gbdt", "tabular", "regression"]
    source_urls = ["https://github.com/dmlc/xgboost"]
    source_reference = "Uses the official XGBoost sklearn regressor with conservative high-performing defaults."
    required_mappings = ["record_id"]
    optional_mappings = ["target_label"]
    produces = ["ml.split_spec", "ml.model", "ml.evaluation_report", "ml.predictions", "ml.training_log", "ml.run"]
    params_schema = {
        "type": "object",
        "required": ["feature_columns"],
        "properties": {
            **_XGBOOST_COMMON_PROPERTIES,
            "objective": {
                "type": "string",
                "enum": ["reg:squarederror", "reg:pseudohubererror", "reg:absoluteerror"],
                "default": "reg:squarederror",
            },
        },
    }

    def build_model(self, run, *, num_classes: int):
        from xgboost import XGBRegressor

        p = run.params
        return XGBRegressor(
            n_estimators=int(p.get("n_estimators", 800)),
            max_depth=int(p.get("max_depth", 6)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            subsample=float(p.get("subsample", 0.8)),
            colsample_bytree=float(p.get("colsample_bytree", 0.8)),
            min_child_weight=float(p.get("min_child_weight", 1.0)),
            reg_alpha=float(p.get("reg_alpha", 0.0)),
            reg_lambda=float(p.get("reg_lambda", 1.0)),
            tree_method=str(p.get("tree_method", "hist")),
            objective=str(p.get("objective", "reg:squarederror")),
            n_jobs=int(p.get("n_jobs", -1)),
            random_state=int(p.get("random_state", 42)),
        )
