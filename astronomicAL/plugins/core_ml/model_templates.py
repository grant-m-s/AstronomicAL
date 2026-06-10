from __future__ import annotations

from typing import Any, Dict, List


def model_templates() -> List[Dict[str, Any]]:
    """Declarative model templates for the Model Builder panel.

    Keep this file dependency-light. Import sklearn/torch only when training.
    """

    return [
        # ------------------------------------------------------------------
        # sklearn classification
        # ------------------------------------------------------------------
        {
            "id": "sklearn.logistic_regression",
            "title": "Logistic Regression",
            "framework": "sklearn",
            "task": "classification",
            "modality": "tabular",
            "estimator": "sklearn.linear_model.LogisticRegression",
            "params": {
                "C": {"type": "float", "default": 1.0, "min": 1e-6},
                "max_iter": {"type": "int", "default": 1000, "min": 1},
                "solver": {
                    "type": "select",
                    "default": "lbfgs",
                    "options": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                },
                "class_weight": {
                    "type": "select",
                    "default": None,
                    "options": [None, "balanced"],
                },
            },
        },
        {
            "id": "sklearn.random_forest_classifier",
            "title": "Random Forest Classifier",
            "framework": "sklearn",
            "task": "classification",
            "modality": "tabular",
            "estimator": "sklearn.ensemble.RandomForestClassifier",
            "params": {
                "n_estimators": {"type": "int", "default": 300, "min": 1},
                "max_depth": {"type": "int_or_none", "default": None, "min": 1},
                "min_samples_split": {"type": "int", "default": 2, "min": 2},
                "min_samples_leaf": {"type": "int", "default": 1, "min": 1},
                "max_features": {
                    "type": "select",
                    "default": "sqrt",
                    "options": ["sqrt", "log2", None],
                },
                "class_weight": {
                    "type": "select",
                    "default": None,
                    "options": [None, "balanced", "balanced_subsample"],
                },
            },
        },
        {
            "id": "sklearn.extra_trees_classifier",
            "title": "Extra Trees Classifier",
            "framework": "sklearn",
            "task": "classification",
            "modality": "tabular",
            "estimator": "sklearn.ensemble.ExtraTreesClassifier",
            "params": {
                "n_estimators": {"type": "int", "default": 300, "min": 1},
                "max_depth": {"type": "int_or_none", "default": None, "min": 1},
                "min_samples_split": {"type": "int", "default": 2, "min": 2},
                "min_samples_leaf": {"type": "int", "default": 1, "min": 1},
                "max_features": {
                    "type": "select",
                    "default": "sqrt",
                    "options": ["sqrt", "log2", None],
                },
                "class_weight": {
                    "type": "select",
                    "default": None,
                    "options": [None, "balanced", "balanced_subsample"],
                },
            },
        },
        {
            "id": "sklearn.gradient_boosting_classifier",
            "title": "Gradient Boosting Classifier",
            "framework": "sklearn",
            "task": "classification",
            "modality": "tabular",
            "estimator": "sklearn.ensemble.GradientBoostingClassifier",
            "params": {
                "n_estimators": {"type": "int", "default": 1500, "min": 1},
                "learning_rate": {"type": "float", "default": 0.05, "min": 1e-6},
                "max_depth": {"type": "int", "default": 3, "min": 1},
                "subsample": {"type": "float", "default": 1.0, "min": 0.01, "max": 1.0},
            },
        },
        {
            "id": "sklearn.hist_gradient_boosting_classifier",
            "title": "Histogram Gradient Boosting Classifier",
            "framework": "sklearn",
            "task": "classification",
            "modality": "tabular",
            "estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
            "params": {
                "learning_rate": {"type": "float", "default": 0.1, "min": 1e-6},
                "max_iter": {"type": "int", "default": 1500, "min": 1},
                "max_leaf_nodes": {"type": "int_or_none", "default": 31, "min": 2},
                "l2_regularization": {"type": "float", "default": 0.0, "min": 0.0},
            },
        },
        {
            "id": "sklearn.svc",
            "title": "Support Vector Classifier",
            "framework": "sklearn",
            "task": "classification",
            "modality": "tabular",
            "estimator": "sklearn.svm.SVC",
            "params": {
                "C": {"type": "float", "default": 1.0, "min": 1e-6},
                "kernel": {
                    "type": "select",
                    "default": "rbf",
                    "options": ["linear", "poly", "rbf", "sigmoid"],
                },
                "gamma": {
                    "type": "select",
                    "default": "scale",
                    "options": ["scale", "auto"],
                },
                "probability": {"type": "bool", "default": True},
                "class_weight": {
                    "type": "select",
                    "default": None,
                    "options": [None, "balanced"],
                },
            },
        },
        {
            "id": "sklearn.knn_classifier",
            "title": "K Neighbours Classifier",
            "framework": "sklearn",
            "task": "classification",
            "modality": "tabular",
            "estimator": "sklearn.neighbors.KNeighborsClassifier",
            "params": {
                "n_neighbors": {"type": "int", "default": 5, "min": 1},
                "weights": {
                    "type": "select",
                    "default": "uniform",
                    "options": ["uniform", "distance"],
                },
                "p": {"type": "int", "default": 2, "min": 1},
            },
        },
        {
            "id": "sklearn.mlp_classifier",
            "title": "sklearn MLP Classifier",
            "framework": "sklearn",
            "task": "classification",
            "modality": "tabular",
            "estimator": "sklearn.neural_network.MLPClassifier",
            "params": {
                "hidden_layer_sizes": {"type": "text", "default": "128,64"},
                "activation": {
                    "type": "select",
                    "default": "relu",
                    "options": ["identity", "logistic", "tanh", "relu"],
                },
                "alpha": {"type": "float", "default": 0.0001, "min": 0.0},
                "learning_rate_init": {"type": "float", "default": 0.001, "min": 1e-8},
                "max_iter": {"type": "int", "default": 300, "min": 1},
            },
        },

        # ------------------------------------------------------------------
        # sklearn regression
        # ------------------------------------------------------------------
        {
            "id": "sklearn.ridge_regression",
            "title": "Ridge Regression",
            "framework": "sklearn",
            "task": "regression",
            "modality": "tabular",
            "estimator": "sklearn.linear_model.Ridge",
            "params": {
                "alpha": {"type": "float", "default": 1.0, "min": 0.0},
            },
        },
        {
            "id": "sklearn.lasso_regression",
            "title": "Lasso Regression",
            "framework": "sklearn",
            "task": "regression",
            "modality": "tabular",
            "estimator": "sklearn.linear_model.Lasso",
            "params": {
                "alpha": {"type": "float", "default": 1.0, "min": 0.0},
                "max_iter": {"type": "int", "default": 1000, "min": 1},
            },
        },
        {
            "id": "sklearn.elastic_net_regression",
            "title": "Elastic Net Regression",
            "framework": "sklearn",
            "task": "regression",
            "modality": "tabular",
            "estimator": "sklearn.linear_model.ElasticNet",
            "params": {
                "alpha": {"type": "float", "default": 1.0, "min": 0.0},
                "l1_ratio": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
                "max_iter": {"type": "int", "default": 1000, "min": 1},
            },
        },
        {
            "id": "sklearn.random_forest_regressor",
            "title": "Random Forest Regressor",
            "framework": "sklearn",
            "task": "regression",
            "modality": "tabular",
            "estimator": "sklearn.ensemble.RandomForestRegressor",
            "params": {
                "n_estimators": {"type": "int", "default": 300, "min": 1},
                "max_depth": {"type": "int_or_none", "default": None, "min": 1},
                "min_samples_split": {"type": "int", "default": 2, "min": 2},
                "min_samples_leaf": {"type": "int", "default": 1, "min": 1},
                "max_features": {
                    "type": "select",
                    "default": 1.0,
                    "options": [1.0, "sqrt", "log2", None],
                },
            },
        },
        {
            "id": "sklearn.extra_trees_regressor",
            "title": "Extra Trees Regressor",
            "framework": "sklearn",
            "task": "regression",
            "modality": "tabular",
            "estimator": "sklearn.ensemble.ExtraTreesRegressor",
            "params": {
                "n_estimators": {"type": "int", "default": 300, "min": 1},
                "max_depth": {"type": "int_or_none", "default": None, "min": 1},
                "min_samples_split": {"type": "int", "default": 2, "min": 2},
                "min_samples_leaf": {"type": "int", "default": 1, "min": 1},
                "max_features": {
                    "type": "select",
                    "default": 1.0,
                    "options": [1.0, "sqrt", "log2", None],
                },
            },
        },
        {
            "id": "sklearn.gradient_boosting_regressor",
            "title": "Gradient Boosting Regressor",
            "framework": "sklearn",
            "task": "regression",
            "modality": "tabular",
            "estimator": "sklearn.ensemble.GradientBoostingRegressor",
            "params": {
                "n_estimators": {"type": "int", "default": 1500, "min": 1},
                "learning_rate": {"type": "float", "default": 0.05, "min": 1e-6},
                "max_depth": {"type": "int", "default": 3, "min": 1},
                "subsample": {"type": "float", "default": 1.0, "min": 0.01, "max": 1.0},
            },
        },
        {
            "id": "sklearn.hist_gradient_boosting_regressor",
            "title": "Histogram Gradient Boosting Regressor",
            "framework": "sklearn",
            "task": "regression",
            "modality": "tabular",
            "estimator": "sklearn.ensemble.HistGradientBoostingRegressor",
            "params": {
                "learning_rate": {"type": "float", "default": 0.1, "min": 1e-6},
                "max_iter": {"type": "int", "default": 1500, "min": 1},
                "max_leaf_nodes": {"type": "int_or_none", "default": 31, "min": 2},
                "l2_regularization": {"type": "float", "default": 0.0, "min": 0.0},
            },
        },
        {
            "id": "sklearn.svr",
            "title": "Support Vector Regressor",
            "framework": "sklearn",
            "task": "regression",
            "modality": "tabular",
            "estimator": "sklearn.svm.SVR",
            "params": {
                "C": {"type": "float", "default": 1.0, "min": 1e-6},
                "epsilon": {"type": "float", "default": 0.1, "min": 0.0},
                "kernel": {
                    "type": "select",
                    "default": "rbf",
                    "options": ["linear", "poly", "rbf", "sigmoid"],
                },
                "gamma": {
                    "type": "select",
                    "default": "scale",
                    "options": ["scale", "auto"],
                },
            },
        },
        {
            "id": "sklearn.knn_regressor",
            "title": "K Neighbours Regressor",
            "framework": "sklearn",
            "task": "regression",
            "modality": "tabular",
            "estimator": "sklearn.neighbors.KNeighborsRegressor",
            "params": {
                "n_neighbors": {"type": "int", "default": 5, "min": 1},
                "weights": {
                    "type": "select",
                    "default": "uniform",
                    "options": ["uniform", "distance"],
                },
                "p": {"type": "int", "default": 2, "min": 1},
            },
        },
        {
            "id": "sklearn.mlp_regressor",
            "title": "sklearn MLP Regressor",
            "framework": "sklearn",
            "task": "regression",
            "modality": "tabular",
            "estimator": "sklearn.neural_network.MLPRegressor",
            "params": {
                "hidden_layer_sizes": {"type": "text", "default": "128,64"},
                "activation": {
                    "type": "select",
                    "default": "relu",
                    "options": ["identity", "logistic", "tanh", "relu"],
                },
                "alpha": {"type": "float", "default": 0.0001, "min": 0.0},
                "learning_rate_init": {"type": "float", "default": 0.001, "min": 1e-8},
                "max_iter": {"type": "int", "default": 300, "min": 1},
            },
        },

        # ------------------------------------------------------------------
        # torch tabular templates currently trainable by the workbench
        # ------------------------------------------------------------------
        {
            "id": "torch.tabular_mlp_classifier",
            "title": "Torch Tabular MLP Classifier",
            "framework": "torch",
            "task": "classification",
            "modality": "tabular",
            "template": "tabular_mlp",
            "params": {
                "optimize_metric": {
                    "type": "select",
                    "default": "val_loss",
                    "options": ["val_loss", "val_f1_macro", "val_accuracy"],
                },
                "hidden_layers": {"type": "text", "default": "128,64"},
                "epochs": {"type": "int", "default": 50, "min": 1},
                "batch_size": {"type": "int", "default": 512, "min": 1},
                "learning_rate": {"type": "float", "default": 0.001, "min": 1e-8},
                "weight_decay": {"type": "float", "default": 0.0, "min": 0.0},
                "patience": {"type": "int", "default": 12, "min": 0},
                "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9},
            },
        },
        {
            "id": "torch.tabular_mlp_regressor",
            "title": "Torch Tabular MLP Regressor",
            "framework": "torch",
            "task": "regression",
            "modality": "tabular",
            "template": "tabular_mlp",
            "params": {
                "optimize_metric": {
                    "type": "select",
                    "default": "val_loss",
                    "options": ["val_loss", "val_r2", "val_mae"],
                },                
                "hidden_layers": {"type": "text", "default": "128,64"},
                "epochs": {"type": "int", "default": 50, "min": 1},
                "batch_size": {"type": "int", "default": 512, "min": 1},
                "learning_rate": {"type": "float", "default": 0.001, "min": 1e-8},
                "weight_decay": {"type": "float", "default": 0.0, "min": 0.0},
                "patience": {"type": "int", "default": 12, "min": 0},
                "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9},
            },
        },

        # ------------------------------------------------------------------
        # torch image templates: definable now, trainable once image workflow exists
        # ------------------------------------------------------------------
        {
            "id": "torch.resnet18_classifier",
            "title": "Torch ResNet-18 Image Classifier",
            "framework": "torch",
            "task": "classification",
            "modality": "image",
            "template": "resnet18",
            "params": {
                "pretrained": {"type": "bool", "default": True},
                "freeze_backbone": {"type": "bool", "default": False},
                "epochs": {"type": "int", "default": 150, "min": 1},
                "batch_size": {"type": "int", "default": 64, "min": 1},
                "learning_rate": {"type": "float", "default": 0.0003, "min": 1e-8},
                "weight_decay": {"type": "float", "default": 0.0001, "min": 0.0},
                "image_size": {"type": "int", "default": 224, "min": 16},
                "optimize_metric": {
                    "type": "select",
                    "default": "val_loss",
                    "options": ["val_loss", "val_f1_macro", "val_accuracy"],
                },
                "patience": {"type": "int", "default": 8, "min": 0},
            },
        },
        {
            "id": "torch.resnet50_classifier",
            "title": "Torch ResNet-50 Image Classifier",
            "framework": "torch",
            "task": "classification",
            "modality": "image",
            "template": "resnet50",
            "params": {
                "pretrained": {"type": "bool", "default": True},
                "freeze_backbone": {"type": "bool", "default": False},
                "epochs": {"type": "int", "default": 150, "min": 1},
                "batch_size": {"type": "int", "default": 32, "min": 1},
                "learning_rate": {"type": "float", "default": 0.0003, "min": 1e-8},
                "weight_decay": {"type": "float", "default": 0.0001, "min": 0.0},
                "image_size": {"type": "int", "default": 224, "min": 16},
                "optimize_metric": {
                    "type": "select",
                    "default": "val_loss",
                    "options": ["val_loss", "val_f1_macro", "val_accuracy"],
                },
                "patience": {"type": "int", "default": 8, "min": 0},
            },
        },
        {
            "id": "torch.unet_segmentation",
            "title": "Torch U-Net Segmentation",
            "framework": "torch",
            "task": "segmentation",
            "modality": "image",
            "template": "unet",
            "params": {
                "base_channels": {"type": "int", "default": 32, "min": 1},
                "depth": {"type": "int", "default": 4, "min": 1},
                "epochs": {"type": "int", "default": 40, "min": 1},
                "batch_size": {"type": "int", "default": 16, "min": 1},
                "learning_rate": {"type": "float", "default": 0.0003, "min": 1e-8},
                "weight_decay": {"type": "float", "default": 0.0001, "min": 0.0},
                "image_size": {"type": "int", "default": 256, "min": 16},
            },
        },
    ]


def get_template(template_id: str) -> Dict[str, Any]:
    for template in model_templates():
        if template["id"] == template_id:
            return template
    raise KeyError(template_id)