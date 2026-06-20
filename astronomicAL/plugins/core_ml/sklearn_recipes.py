from __future__ import annotations

from typing import Any

from .sklearn_harness import SklearnRecipe, fit_warm_start


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