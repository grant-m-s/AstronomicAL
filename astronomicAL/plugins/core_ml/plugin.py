from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from astronomicAL.platform.plugins import PluginManifest


manifest = PluginManifest(
    id="core.ml",
    name="ML Core",
    version="0.4.0",
    description=(
        "Machine-learning workbench: model definitions, adaptive data binding, "
        "tabular/image training, durable model artifacts, predictions, active-learning "
        "batches, metrics, and training curves."
    ),
    requires=["scikit-learn>=1.2"],
    optional_requires=["torch", "torchvision", "pillow", "matplotlib", "optuna", "joblib"],
    capabilities=["panel", "action", "machine-learning", "training", "inference", "artifacts"],
    tags=[
        "core",
        "ml",
        "sklearn",
        "torch",
        "classification",
        "regression",
        "image",
        "tabular",
        "active-learning",
    ],
)


def _load_sibling_module(stem: str):
    module_name = f"{__name__}.{stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = Path(__file__).with_name(f"{stem}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sibling module {stem!r} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def register(api) -> None:
    actions = _load_sibling_module("actions")
    prediction = _load_sibling_module("prediction")
    trained_models = _load_sibling_module("trained_models")
    predict_panel = _load_sibling_module("predict_panel")
    recipe_runner = _load_sibling_module("recipe_runner")

    api.register_service(
        key="registry",
        factory=create_ml_registry,
        lazy=True,
        replace=True,
        description="ML registry for model/provider specs.",
    )

    api.register_service(
        key="trained_model_catalog",
        factory=lambda context: trained_models.create_trained_model_catalog(context),
        lazy=True,
        replace=True,
        description="Indexes trained ml.model artifacts and validates model/dataset compatibility.",
    )

    api.register_service(
        key="recipe_registry",
        factory=create_ml_recipe_registry,
        lazy=True,
        replace=True,
        description=(
            "Registry of code-backed ML recipes. Recipes expose typed UI parameters "
            "but keep expert dataloaders/training loops in Python."
        ),
    )

    api.register_action(
        id="train_tabular",
        title="Train Tabular Model",
        handler=actions.train_tabular_action,
        description=(
            "Train a saved tabular model definition and persist the trained model "
            "as a durable ml.model artifact."
        ),
        category="Machine Learning",
        icon="psychology",
        tags=["ml", "training", "tabular", "sklearn", "torch"],
        inputs={
            "dataset": True,
            "selection": "optional",
            "columns": "many",
            "numeric_columns": "none",
            "required_mappings": ["record_id"],
            "optional_mappings": ["target_label"],
        },
        outputs=[
            {"type": "ml.feature_spec", "description": "Feature/target binding used for the run."},
            {"type": "ml.split_spec", "description": "Train/validation/test split row ids."},
            {"type": "ml.model", "description": "Durable trained model artifact."},
            {"type": "ml.evaluation_report", "description": "Validation/test metrics."},
            {"type": "ml.predictions", "description": "Held-out predictions."},
            {"type": "ml.training_log", "description": "Training epochs or one-step sklearn log."},
            {"type": "ml.run", "description": "Run summary tying artifacts together."},
        ],
        params_schema={
            "type": "object",
            "required": ["target_column", "model_id"],
            "properties": {
                "dataset_id": {"type": "string"},
                "target_column": {"type": "string", "minLength": 1},
                "feature_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "model_id": {"type": "string", "minLength": 1},
                "task": {
                    "type": "string",
                    "enum": ["classification", "regression"],
                    "default": "classification",
                },
                "test_size": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 0.8,
                    "default": 0.2,
                },
                "validation_size": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 0.8,
                    "default": 0.2,
                },
                "random_state": {"type": "integer", "default": 42},
                "stratify": {"type": "boolean", "default": True},
                "optimize_metric": {"type": "string", "default": "val_loss"},
                "torch_hidden_layers": {"type": "string", "default": "128,64"},
                "torch_epochs": {"type": "integer", "minimum": 1, "default": 50},
                "torch_batch_size": {"type": "integer", "minimum": 1, "default": 512},
                "torch_learning_rate": {"type": "number", "minimum": 0.0, "default": 0.001},
                "torch_weight_decay": {"type": "number", "minimum": 0.0, "default": 0.0},
                "torch_patience": {"type": "integer", "minimum": 1, "default": 12},
                "run_id": {"type": "string"},
                "training_log_artifact_id": {"type": "string"},
            },
        },
        run_in_job=True,
        requires=["scikit-learn>=1.2"],
        optional_requires=["torch", "optuna", "joblib"],
    )

    api.register_action(
        id="train_image_classifier",
        title="Train Image Classifier",
        handler=actions.train_image_classifier_action,
        description=(
            "Train a saved torch image-classification model definition and persist "
            "the trained model as a durable ml.model artifact."
        ),
        category="Machine Learning",
        icon="image",
        tags=["ml", "training", "image", "torch", "classification"],
        inputs={
            "dataset": True,
            "selection": "optional",
            "columns": "many",
            "numeric_columns": "none",
            "required_mappings": ["record_id"],
            "optional_mappings": ["target_label"],
        },
        outputs=[
            {"type": "ml.image_spec", "description": "Image/target binding used for the run."},
            {"type": "ml.split_spec", "description": "Train/validation/test split row ids."},
            {"type": "ml.model", "description": "Durable trained model artifact."},
            {"type": "ml.evaluation_report", "description": "Validation/test metrics."},
            {"type": "ml.predictions", "description": "Held-out image predictions."},
            {"type": "ml.training_log", "description": "Training epochs and tuning log."},
            {"type": "ml.run", "description": "Run summary tying artifacts together."},
        ],
        params_schema={
            "type": "object",
            "required": ["image_column", "target_column", "model_id"],
            "properties": {
                "dataset_id": {"type": "string"},
                "image_column": {"type": "string", "minLength": 1},
                "target_column": {"type": "string", "minLength": 1},
                "model_id": {"type": "string", "minLength": 1},
                "test_size": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 0.8,
                    "default": 0.2,
                },
                "validation_size": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 0.8,
                    "default": 0.2,
                },
                "random_state": {"type": "integer", "default": 42},
                "stratify": {"type": "boolean", "default": True},
                "run_id": {"type": "string"},
                "training_log_artifact_id": {"type": "string"},
            },
        },
        run_in_job=True,
        requires=[],
        optional_requires=["torch", "torchvision", "pillow", "optuna"],
    )

    api.register_action(
        id="run_ml_recipe",
        title="Run ML Recipe",
        handler=recipe_runner.run_ml_recipe_action,
        description=(
            "Run a code-backed ML recipe. This is the extension point for expert "
            "training loops with custom transforms, dataloaders, schedulers, losses, "
            "callbacks, checkpointing, and prediction logic."
        ),
        category="Machine Learning",
        icon="integration_instructions",
        tags=["ml", "recipe", "training", "expert", "torch", "workflow"],
        inputs={
            "dataset": True,
            "selection": "optional",
            "columns": "optional",
            "numeric_columns": "none",
            "required_mappings": ["record_id"],
            "optional_mappings": ["target_label", "image.path", "image.uri"],
        },
        outputs=[
            {"type": "ml.training_log", "description": "Live recipe progress and metrics."},
            {"type": "ml.model", "description": "Optional durable trained model artifact."},
            {"type": "ml.predictions", "description": "Optional prediction artifact."},
            {"type": "ml.evaluation_report", "description": "Optional evaluation report."},
            {"type": "ml.run", "description": "Run summary and provenance."},
        ],
        params_schema={
            "type": "object",
            "required": ["recipe_id"],
            "properties": {
                "dataset_id": {"type": "string"},
                "recipe_id": {"type": "string", "minLength": 1},
                "run_id": {"type": "string"},
            },
        },
        run_in_job=True,
        requires=[],
        optional_requires=["torch", "torchvision", "pillow", "scikit-learn"],
    )

    api.register_action(
        id="predict",
        title="Predict with Trained Model",
        handler=prediction.predict_action,
        description=(
            "Run compatibility-checked inference using any durable ml.model artifact. "
            "Creates an ml.predictions artifact and, by default, a prediction-table "
            "dataset that can be used for plotting and colouring."
        ),
        category="Machine Learning",
        icon="batch_prediction",
        tags=["ml", "prediction", "inference", "trained-model", "visualisation"],
        inputs={
            "dataset": True,
            "selection": "optional",
            "columns": "optional",
            "numeric_columns": "none",
            "required_mappings": ["record_id"],
            "accepts_artifact_types": ["ml.model"],
        },
        outputs=[
            {
                "type": "ml.predictions",
                "description": "Self-describing predictions, probabilities, uncertainty scores, and visualisation hints.",
            },
            {
                "type": "dataset",
                "description": "Optional row-id keyed prediction table dataset for visualisation/colouring.",
            },
        ],
        params_schema={
            "type": "object",
            "required": ["model_artifact_id"],
            "properties": {
                "dataset_id": {"type": "string"},
                "model_artifact_id": {"type": "string", "minLength": 1},
                "target_column": {"type": "string"},
                "image_column": {"type": "string"},
                "feature_column_mapping": {"type": "object"},
                "require_target_compatible": {"type": "boolean", "default": False},
                "register_prediction_dataset": {"type": "boolean", "default": True},
                "run_id": {"type": "string"},
            },
        },
        run_in_job=True,
        requires=["scikit-learn>=1.2"],
        optional_requires=["torch", "joblib", "pillow"],
    )

    api.register_action(
        id="predict_tabular",
        title="Predict with Tabular Model",
        handler=prediction.predict_tabular_action,
        description=(
            "Run inference over a dataset or selection using a durable sklearn "
            "tabular ml.model artifact."
        ),
        category="Machine Learning",
        icon="batch_prediction",
        tags=["ml", "prediction", "inference", "tabular", "sklearn"],
        inputs={
            "dataset": True,
            "selection": "optional",
            "columns": "optional",
            "numeric_columns": "none",
            "required_mappings": ["record_id"],
            "accepts_artifact_types": ["ml.model"],
        },
        outputs=[
            {
                "type": "ml.predictions",
                "description": "Predictions and probabilities for each row.",
            }
        ],
        params_schema={
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "model_artifact_id": {"type": "string"},
                "feature_columns": {"type": "array", "items": {"type": "string"}},
                "run_id": {"type": "string"},
            },
        },
        run_in_job=True,
        requires=["scikit-learn>=1.2"],
        optional_requires=["joblib"],
    )

    api.register_action(
        id="create_active_learning_batch",
        title="Create Active-Learning Batch",
        handler=actions.create_active_learning_batch_action,
        description=(
            "Rank prediction records by uncertainty and optionally promote the top "
            "rows to the platform selection set for review/annotation."
        ),
        category="Machine Learning",
        icon="rule",
        tags=["ml", "active-learning", "selection", "uncertainty"],
        inputs={
            "dataset": False,
            "selection": "none",
            "columns": "none",
            "numeric_columns": "none",
            "accepts_artifact_types": ["ml.predictions"],
        },
        outputs=[
            {
                "type": "ml.active_learning_batch",
                "description": "Ranked uncertain rows for review.",
            },
            {
                "type": "selection.ids",
                "optional": True,
                "description": "Selection set created when make_selection is true.",
            },
        ],
        params_schema={
            "type": "object",
            "properties": {
                "predictions_artifact_id": {"type": "string"},
                "dataset_id": {"type": "string"},
                "strategy": {
                    "type": "string",
                    "enum": ["least_confidence", "margin", "entropy"],
                    "default": "least_confidence",
                },
                "k": {"type": "integer", "minimum": 1, "default": 50},
                "make_selection": {"type": "boolean", "default": True},
            },
        },
        run_in_job=False,
    )

    api.register_panel(
        id="model_builder",
        title="ML Model Builder",
        factory=create_model_builder_panel,
        description=(
            "Create reusable sklearn and torch model definitions with explicit "
            "hyperparameters for use in the ML Workbench."
        ),
        category="Machine Learning",
        icon="tune",
        tags=["ml", "models", "hyperparameters", "sklearn", "torch"],
        uses_services=["core.ml.registry"],
        produces=["ml.model_definition"],
        default_layout={"x": 0, "y": 0, "w": 5, "h": 7},
    )

    api.register_panel(
        id="workbench",
        title="ML Workbench",
        factory=create_ml_workbench_panel,
        description=(
            "Train saved model definitions. The workbench adapts its input binding "
            "to the selected model definition modality."
        ),
        category="Machine Learning",
        icon="psychology",
        tags=["ml", "training", "classification", "regression", "tabular", "image"],
        required_mappings=["record_id"],
        optional_mappings=["target_label"],
        uses_services=["core.ml.registry"],
        produces=[
            "ml.feature_spec",
            "ml.image_spec",
            "ml.split_spec",
            "ml.model",
            "ml.predictions",
            "ml.evaluation_report",
            "ml.training_log",
            "ml.run",
            "ml.run.finished",
        ],
        default_layout={"x": 5, "y": 0, "w": 5, "h": 7},
    )

    api.register_panel(
        id="recipe_launcher",
        title="ML Recipe Launcher",
        factory=create_ml_recipe_launcher_panel,
        description=(
            "Launch code-backed ML recipes. Use this for high-performance setups "
            "that need real Python dataloaders, transforms, schedulers, callbacks, "
            "checkpointing, or custom training loops."
        ),
        category="Machine Learning",
        icon="integration_instructions",
        tags=["ml", "recipes", "training", "torch", "expert", "no-code"],
        required_mappings=["record_id"],
        optional_mappings=["target_label", "image.path", "image.uri"],
        uses_services=["core.ml.recipe_registry"],
        produces=[
            "ml.training_log",
            "ml.model",
            "ml.predictions",
            "ml.evaluation_report",
            "ml.run",
            "ml.recipe_run.started",
            "ml.recipe_run.progress",
            "ml.recipe_run.finished",
        ],
        default_layout={"x": 10, "y": 0, "w": 5, "h": 7},
    )

    api.register_panel(
        id="predictor",
        title="ML Predictor",
        factory=create_ml_predict_panel,
        description=(
            "Choose any trained model artifact, validate it against a dataset, "
            "run prediction, and expose prediction outputs for visualisation."
        ),
        category="Machine Learning",
        icon="batch_prediction",
        tags=["ml", "prediction", "inference", "trained-model", "visualisation"],
        required_mappings=["record_id"],
        optional_mappings=["target_label", "image.path", "image.uri"],
        uses_services=["core.ml.trained_model_catalog"],
        produces=["ml.predictions", "dataset"],
        default_layout={"x": 5, "y": 7, "w": 5, "h": 5},
    )

    api.register_panel(
        id="training_curves",
        title="ML Training Curves",
        factory=create_training_curves_panel,
        description="View loss, accuracy, F1, and regression curves from training-log artifacts.",
        category="Machine Learning",
        icon="show_chart",
        tags=["ml", "training", "curves", "torch", "metrics"],
        produces=[],
        default_layout={"x": 0, "y": 7, "w": 5, "h": 5},
    )


def create_ml_registry():
    ml = _load_sibling_module("ml")
    return ml.build_default_registry()


def create_model_builder_panel(context, **kwargs):
    builder_module = _load_sibling_module("model_builder_panel")
    registry = context.services.get("core.ml.registry")

    controller = builder_module.MLModelBuilderPanel(
        context=context,
        registry=registry,
        restore_state=kwargs.get("restore_state"),
    )

    return controller.panel(), controller


def create_ml_workbench_panel(context, **kwargs):
    panel_module = _load_sibling_module("panel")
    registry = context.services.get("core.ml.registry")

    controller = panel_module.MLWorkbenchPanel(
        context=context,
        registry=registry,
        restore_state=kwargs.get("restore_state"),
    )

    return controller.panel(), controller


def create_training_curves_panel(context, **kwargs):
    curves_module = _load_sibling_module("curves_panel")

    controller = curves_module.MLTrainingCurvesPanel(
        context=context,
        restore_state=kwargs.get("restore_state"),
    )

    return controller.panel(), controller

def create_ml_predict_panel(context, **kwargs):
    predict_panel_module = _load_sibling_module("predict_panel")
    return predict_panel_module.create_predict_panel(context=context, **kwargs)

def create_ml_recipe_registry(context=None):
    registry_module = _load_sibling_module("recipe_registry")
    recipes_module = _load_sibling_module("recipes")

    registry = registry_module.MLRecipeRegistry()
    registry.register(recipes_module.ExternalPythonRecipe)
    registry.register(recipes_module.CIFARResNetRecipe)
    return registry


def create_ml_recipe_launcher_panel(context, **kwargs):
    recipe_panel_module = _load_sibling_module("recipe_panel")
    return recipe_panel_module.create_recipe_launcher_panel(context=context, **kwargs)