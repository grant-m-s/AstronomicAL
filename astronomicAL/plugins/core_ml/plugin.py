from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from astronomicAL.platform.plugins import PluginManifest


manifest = PluginManifest(
    id="core.ml",
    name="ML Core",
    version="0.3.0",
    description=(
        "Machine-learning workbench: model definitions, adaptive data binding, "
        "tabular/image training, metrics, artifacts, and training curves."
    ),
    requires=["scikit-learn>=1.2"],
    optional_requires=["torch", "torchvision", "pillow", "matplotlib", "optuna"],
    capabilities=["panel", "machine-learning", "training", "artifacts"],
    tags=["core", "ml", "sklearn", "torch", "classification", "regression", "image", "tabular"],
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
    api.register_service(
        key="registry",
        factory=create_ml_registry,
        lazy=True,
        replace=True,
        description="ML registry for model/provider specs.",
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