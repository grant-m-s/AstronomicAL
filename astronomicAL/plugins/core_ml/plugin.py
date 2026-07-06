from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="core.ml",
    name="ML Core",
    version="0.5.0",
    description=(
        "Machine-learning core: code-backed recipes, reusable recipe profiles, "
        "protocol-enforced training, durable model artifacts, compatibility-checked "
        "prediction, active-learning batches, trained-model cataloguing, and training curves."
    ),
    requires=["scikit-learn>=1.2"],
    optional_requires=["torch", "torchvision", "pillow", "matplotlib", "optuna", "joblib"],
    capabilities=["panel", "action", "machine-learning", "training", "inference", "artifacts", "recipe-profile",],
    tags=[
        "core",
        "ml",
        "recipe",
        "torch",
        "sklearn",
        "classification",
        "regression",
        "image",
        "tabular",
        "active-learning",
    ],
)

def register(api) -> None:

    from . import actions
    from . import prediction
    from . import trained_models
    from . import recipe_runner
    from . import recipe_profiles

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

    api.register_service(
        key="recipe_profile_store",
        factory=lambda context: recipe_profiles.create_recipe_profile_store(context),
        lazy=True,
        replace=True,
        description="Stores reusable ML recipe profiles/run templates for recipe launcher and active learning.",
    )

    api.register_action(
        id="run_ml_recipe",
        title="Run ML Recipe",
        handler=recipe_runner.run_ml_recipe_action,
        description=(
            "Run a code-backed ML recipe. This is the extension point for expert "
            "training loops with custom transforms, dataloaders, schedulers, losses, "
            "callbacks, checkpointing, and prediction logic. Managed recipes run "
            "under a protocol-enforcing harness that owns splitting, validation "
            "selection, test evaluation, and provenance."
        ),
        category="Machine Learning",
        icon="integration_instructions",
        tags=["ml", "recipe", "training", "expert", "torch", "sklearn", "workflow"],
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
            {"type": "ml.split_spec", "description": "Train/validation/test split row ids and protocol."},
            {"type": "ml.model", "description": "Durable trained model artifact."},
            {"type": "ml.predictions", "description": "Optional prediction artifact."},
            {"type": "ml.evaluation_report", "description": "Optional evaluation report."},
            {"type": "ml.run", "description": "Run summary and provenance."},
        ],
        params_schema={
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "recipe_id": {"type": "string"},
                "recipe_profile_id": {"type": "string"},
                "recipe_profile_artifact_id": {"type": "string"},
                "run_id": {"type": "string"},
            },
        },
        run_in_job=True,
        requires=[],
        optional_requires=["torch", "torchvision", "pillow", "scikit-learn"],
    )

    api.register_action(
        id="save_recipe_profile",
        title="Save ML Recipe Profile",
        handler=recipe_profiles.save_recipe_profile_action,
        description="Persist a reusable recipe/profile configuration for later launcher or active-learning runs.",
        category="Machine Learning",
        icon="save",
        tags=["ml", "recipe", "profile", "template"],
        inputs={"dataset": False, "selection": "none", "columns": "none", "numeric_columns": "none"},
        outputs=[{"type": "ml.recipe_profile", "description": "Saved reusable recipe profile."}],
        params_schema={
            "type": "object",
            "required": ["name", "recipe_id"],
            "properties": {
                "profile_id": {"type": "string"},
                "name": {"type": "string"},
                "recipe_id": {"type": "string"},
                "default_dataset_id": {"type": "string"},
                "recipe_params": {"type": "object"},
                "protocol_params": {"type": "object"},
                "binding_params": {"type": "object"},
                "notes": {"type": "string"},
            },
        },
        run_in_job=False,
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
        tags=["ml", "recipes", "training", "torch", "sklearn", "expert"],
        required_mappings=["record_id"],
        optional_mappings=["target_label", "image.path", "image.uri"],
        uses_services=["core.ml.recipe_registry"],
        produces=[
            "ml.training_log",
            "ml.split_spec",
            "ml.model",
            "ml.predictions",
            "ml.evaluation_report",
            "ml.run",
            "ml.recipe_run.started",
            "ml.recipe_run.progress",
            "ml.recipe_run.finished",
        ],
        default_layout={"x": 0, "y": 0, "w": 5, "h": 7},
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


def create_training_curves_panel(context, **kwargs):
    from . import curves_panel as curves_module

    controller = curves_module.MLTrainingCurvesPanel(
        context=context,
        restore_state=kwargs.get("restore_state"),
    )

    return controller.panel(), controller


def create_ml_predict_panel(context, **kwargs):
    from . import predict_panel as predict_panel_module

    return predict_panel_module.create_predict_panel(context=context, **kwargs)


def create_ml_recipe_registry(context=None):
    from . import recipe_registry as registry_module
    from . import recipes as recipes_module
    from . import sklearn_harness
    from . import sklearn_recipes

    # Register sklearn harness selection with recipe_registry.make_harness().
    # Idempotent in sklearn_harness.register().
    sklearn_harness.register()

    registry = registry_module.MLRecipeRegistry()

    registry.register(recipes_module.ExternalPythonRecipe)
    registry.register(recipes_module.CIFARResNetRecipe)
    registry.register(recipes_module.TimmImageClassifierRecipe)
    registry.register(recipes_module.WideResNetCIFARRecipe)
    registry.register(recipes_module.TimmImageRegressorRecipe)
    registry.register(recipes_module.TabularMLPRegressorRecipe)

    registry.register(sklearn_recipes.SklearnTabularClassifierRecipe)
    registry.register(sklearn_recipes.SklearnTabularRegressorRecipe)

    return registry

def create_ml_recipe_launcher_panel(context, **kwargs):
    from . import recipe_panel as recipe_panel_module

    return recipe_panel_module.create_recipe_launcher_panel(context=context, **kwargs)