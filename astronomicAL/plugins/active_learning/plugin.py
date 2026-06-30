from __future__ import annotations

from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="core.active_learning",
    name="Active Learning",
    version="0.3.0",
    description=(
        "Active-learning session manager built on ML predictions, ranked query strategies, "
        "AstronomicAL selection sets, and scratch retraining through core.ml recipes when available."
    ),
    requires_plugins=[],
    requires=[],
    optional_requires=[],
    capabilities=["panel", "action", "active-learning", "selection", "machine-learning"],
    tags=["core", "active-learning", "ml", "selection", "query-strategy"],
)

def register(api) -> None:
    from . import actions
    from . import strategies

    api.register_service(
        key="query_strategy_registry",
        factory=lambda context: strategies.create_default_strategy_registry(),
        lazy=True,
        replace=True,
        description=(
            "Registry for active-learning query strategies. Other plugins can retrieve this "
            "service and register new QueryStrategy instances."
        ),
    )

    api.register_action(
        id="start_session",
        title="Start Active-Learning Session",
        handler=actions.start_session_action,
        description=(
            "Create an active-learning session and draw an ordered initial random sample "
            "from the unlabelled pool."
        ),
        category="Active Learning",
        icon="playlist_add",
        tags=["active-learning", "random", "selection", "session"],
        inputs={
            "dataset": True,
            "selection": "none",
            "columns": "none",
            "numeric_columns": "none",
            "required_mappings": ["record_id"],
        },
        outputs=[
            {"type": "al.session", "description": "Active-learning session state."},
            {
                "type": "ml.active_learning_batch",
                "description": "Initial random review batch.",
            },
            {
                "type": "selection.ids",
                "optional": True,
                "description": "Ordered selection set.",
            },
        ],
        params_schema={
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "label_options": {"type": "array", "items": {"type": "string"}},
                "target_column": {"type": "string", "default": "al_label"},
                "initial_k": {"type": "integer", "minimum": 0, "default": 20},
                "seed": {"type": "integer", "default": 42},
                "make_selection": {"type": "boolean", "default": True},
                "session_contract": {"type": "object"},
                "image_column": {"type": "string"},
                "mask_column": {"type": "string"},
                "feature_columns": {"type": "array", "items": {"type": "string"}},
            },
        },
        run_in_job=False,
    )

    api.register_action(
        id="profile_data_contract",
        title="Inspect Active-Learning Data Contract",
        handler=actions.profile_data_contract_action,
        description=(
            "Resolve recipe-driven image/tabular/mask bindings, validate pool and holdout "
            "datasets, and inspect a small image sample."
        ),
        category="Active Learning",
        icon="fact_check",
        tags=["active-learning", "recipe", "data-contract", "image", "tabular"],
        inputs={
            "dataset": True,
            "selection": "none",
            "columns": "optional",
            "numeric_columns": "none",
            "required_mappings": ["record_id"],
        },
        outputs=[],
        params_schema={
            "type": "object",
            "required": ["recipe_id"],
            "properties": {
                "recipe_id": {"type": "string"},
                "dataset_id": {"type": "string"},
                "validation_dataset_id": {"type": "string"},
                "test_dataset_id": {"type": "string"},
                "target_column": {"type": "string"},
                "feature_columns": {"type": "array", "items": {"type": "string"}},
                "image_column": {"type": "string"},
                "mask_column": {"type": "string"},
                "recipe_params": {"type": "object"},
                "protocol_params": {"type": "object"},
                "labelled_count": {"type": "integer", "minimum": 0},
                "inspect_images": {"type": "boolean", "default": True},
                "image_sample_size": {"type": "integer", "minimum": 0, "default": 8},
            },
        },
        run_in_job=True,
    )

    api.register_action(
        id="query_batch",
        title="Create Active-Learning Query Batch",
        handler=actions.query_batch_action,
        description=(
            "Rank prediction records with a registered query strategy, ignore already "
            "labelled/unsure rows, and promote the top-k rows to the AstronomicAL selection "
            "set in informativeness order."
        ),
        category="Active Learning",
        icon="rule",
        tags=["active-learning", "query", "selection", "uncertainty"],
        inputs={
            "dataset": False,
            "selection": "none",
            "columns": "none",
            "numeric_columns": "none",
            "accepts_artifact_types": ["ml.predictions"],
        },
        outputs=[
            {"type": "al.session", "description": "Updated active-learning session state."},
            {
                "type": "ml.active_learning_batch",
                "description": "Ranked review batch.",
            },
            {
                "type": "selection.ids",
                "optional": True,
                "description": "Ordered selection set.",
            },
        ],
        params_schema={
            "type": "object",
            "properties": {
                "session_artifact_id": {"type": "string"},
                "predictions_artifact_id": {"type": "string"},
                "strategy_id": {"type": "string", "default": "least_confidence"},
                "k": {"type": "integer", "minimum": 1, "default": 200},
                "seed": {"type": "integer", "default": 42},
                "make_selection": {"type": "boolean", "default": True},
            },
        },
        run_in_job=True,
    )

    api.register_action(
        id="record_label",
        title="Record Active-Learning Label",
        handler=actions.record_label_action,
        description=(
            "Record a label/verification for the focused row. The special Unsure label "
            "removes a row from the pool but does not add it to the training set."
        ),
        category="Active Learning",
        icon="label",
        tags=["active-learning", "label", "annotation", "selection"],
        inputs={
            "dataset": False,
            "selection": "optional",
            "columns": "none",
            "numeric_columns": "none",
        },
        outputs=[
            {"type": "al.session", "description": "Updated active-learning session state."},
        ],
        params_schema={
            "type": "object",
            "required": ["session_artifact_id", "label"],
            "properties": {
                "session_artifact_id": {"type": "string"},
                "row_id": {"type": "string"},
                "label": {"type": "string"},
                "source": {"type": "string", "default": "manual"},
            },
        },
        run_in_job=False,
    )

    api.register_action(
        id="train_from_session",
        title="Train Active-Learning Round From Scratch",
        handler=actions.train_from_session_action,
        description=(
            "Materialise verified labels, train from scratch through core.ml, then by default "
            "predict over the original pool and create the next ranked review batch."
        ),
        category="Active Learning",
        icon="model_training",
        tags=["active-learning", "training", "ml", "recipe", "scratch"],
        inputs={
            "dataset": False,
            "selection": "none",
            "columns": "none",
            "numeric_columns": "none",
        },
        outputs=[
            {
                "type": "al.session",
                "description": "Updated session with the completed AL round.",
            },
            {
                "type": "al.training_set",
                "description": "Training rows and label manifest.",
            },
            {
                "type": "ml.run",
                "optional": True,
                "description": "core.ml run summary.",
            },
            {
                "type": "ml.model",
                "optional": True,
                "description": "Trained model artifact from core.ml.",
            },
            {
                "type": "ml.predictions",
                "optional": True,
                "description": "Pool predictions generated after training.",
            },
            {
                "type": "ml.active_learning_batch",
                "optional": True,
                "description": "Next ranked review batch generated after prediction.",
            },
        ],
        params_schema={
            "type": "object",
            "required": ["session_artifact_id", "recipe_id"],
            "properties": {
                "session_artifact_id": {"type": "string"},
                "recipe_id": {"type": "string"},
                "recipe_params": {"type": "object"},
                "target_column": {"type": "string", "default": "al_label"},
                "train_dataset_id": {"type": "string"},
                "seed": {"type": "integer"},
                "session_contract": {"type": "object"},
                "auto_predict": {"type": "boolean", "default": True},
                "auto_query": {"type": "boolean", "default": True},
                "query_strategy_id": {"type": "string", "default": "least_confidence"},
                "query_k": {"type": "integer", "minimum": 1, "default": 200},
                "make_selection": {"type": "boolean", "default": True},
            },
        },
        run_in_job=True,
    )

    api.register_panel(
        id="panel",
        title="Active Learning",
        factory=create_active_learning_panel,
        description=(
            "Manage active-learning sessions, initial random sampling, ranked query batches, "
            "manual labels including Unsure, and scratch retraining through selectable core.ml "
            "recipes. Recipes, datasets, recipe-driven image/tabular/mask inputs, labels, "
            "validation/test sets, and schema-defined recipe parameters are populated from "
            "platform state where available."
        ),
        category="Active Learning",
        icon="psychology",
        tags=["active-learning", "ml", "selection", "annotation"],
        required_mappings=["record_id"],
        optional_mappings=[
            "target_label",
            "image.path",
            "image.uri",
            "mask.path",
            "mask",
        ],
        # Descriptive only: the panel should still load when core.ml is disabled.
        uses_services=[
            "core.active_learning.query_strategy_registry",
            "core.ml.recipe_registry",
        ],
        produces=[
            "al.session",
            "al.training_set",
            "ml.active_learning_batch",
            "selection.ids",
            "al.session.created",
            "al.query_batch.created",
            "al.label.recorded",
            "al.round.training_started",
            "al.round.training_finished",
        ],
        default_layout={"x": 0, "y": 0, "w": 6, "h": 8},
        state_version=3,
        persist_layout=True,
        persist_state=True,
        restore_policy="best_effort",
    )

def create_active_learning_panel(context, **kwargs):
    from . import panel as panel_module

    controller = panel_module.ActiveLearningPanel(
        context=context,
        restore_state=kwargs.get("restore_state"),
    )
    return controller.panel(), controller
