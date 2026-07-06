from __future__ import annotations

from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="core.active_learning",
    name="Active Learning",
    version="0.4.3",
    description=(
        "Small active-learning core: sessions, labels, batch-capable query strategies, "
        "selection handoff, and an optional core.ml bridge for train/predict/query loops."
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
            "Registry for active-learning query strategies. Other plugins can retrieve "
            "core.active_learning.query_strategy_registry and register QueryStrategy instances."
        ),
    )

    api.register_action(
        id="start_session",
        title="Start Active-Learning Session",
        handler=actions.start_session_action,
        description="Create an active-learning session and optionally draw an initial random review batch.",
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
            {"type": "ml.active_learning_batch", "description": "Initial random review batch."},
            {"type": "selection.ids", "optional": True, "description": "Ordered selection set."},
        ],
        params_schema={
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "pool_dataset_id": {"type": "string"},
                "label_options": {"type": "array", "items": {"type": "string"}, "description": "Optional override; otherwise inferred from target_column/label_column."},
                "target_column": {"type": "string", "description": "Dataset column used to infer labels and as the session target column."},
                "label_column": {"type": "string", "description": "Alias for target_column."},
                "task_type": {"type": "string", "enum": ["classification", "regression", "auto"], "default": "auto", "description": "Detected/declared target type."},
                "problem_type": {"type": "string", "enum": ["classification", "regression", "auto"], "description": "Alias for task_type."},
                "label_profile": {"type": "object", "description": "Bounded target-column profile used to infer classification vs regression."},
                "infer_labels_from_column": {"type": "boolean", "default": True},
                "initial_k": {"type": "integer", "minimum": 0, "default": 20},
                "seed": {"type": "integer", "default": 42},
                "make_selection": {"type": "boolean", "default": True},
                "recipe_profile_id": {"type": "string", "description": "Optional saved core.ml recipe profile metadata."},
                "recipe_id": {"type": "string", "description": "Legacy optional metadata only; not required for sessions."},
                "session_contract": {"type": "object"},
            },
        },
        run_in_job=False,
    )

    api.register_action(
        id="profile_data_contract",
        title="Inspect Active-Learning Data Contract",
        handler=actions.profile_data_contract_action,
        description="Lightweight optional inspection of dataset/recipe bindings for the core.ml bridge.",
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
            "properties": {
                "recipe_profile_id": {"type": "string"},
                "recipe_profile_artifact_id": {"type": "string"},
                "recipe_id": {"type": "string", "description": "Legacy fallback; prefer recipe_profile_id."},
                "dataset_id": {"type": "string"},
                "validation_dataset_id": {"type": "string"},
                "test_dataset_id": {"type": "string"},
                "target_column": {"type": "string"},
                "task_type": {"type": "string", "enum": ["classification", "regression", "auto"]},
                "feature_columns": {"type": "array", "items": {"type": "string"}},
                "image_column": {"type": "string"},
                "mask_column": {"type": "string"},
                "recipe_params": {"type": "object"},
                "labelled_count": {"type": "integer", "minimum": 0},
            },
        },
        run_in_job=True,
    )

    api.register_action(
        id="query_batch",
        title="Create Active-Learning Query Batch",
        handler=actions.query_batch_action,
        description=(
            "Acquire a ranked query batch from ml.predictions using a registered strategy. "
            "Strategies can be per-record or full batch-aware."
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
            {"type": "ml.active_learning_batch", "description": "Ranked review batch."},
            {"type": "selection.ids", "optional": True, "description": "Ordered selection set."},
        ],
        params_schema={
            "type": "object",
            "properties": {
                "session_artifact_id": {"type": "string"},
                "predictions_artifact_id": {"type": "string"},
                "strategy_id": {"type": "string", "default": "least_confidence"},
                "strategy_params": {"type": "object"},
                "k": {"type": "integer", "minimum": 1, "default": 200},
                "seed": {"type": "integer", "default": 42},
                "make_selection": {"type": "boolean", "default": True},
                "exclude_row_ids": {"type": "array", "items": {"type": "string"}},
            },
        },
        run_in_job=True,
    )

    api.register_action(
        id="score_pool",
        title="Calculate Query-Strategy Scores",
        handler=actions.score_pool_action,
        description=(
            "Score every currently eligible pool row with one or more registered query strategies "
            "for XY diagnostics, without creating a review queue."
        ),
        category="Active Learning",
        icon="analytics",
        tags=["active-learning", "query", "strategy", "diagnostics", "visualisation"],
        inputs={
            "dataset": False,
            "selection": "none",
            "columns": "none",
            "numeric_columns": "none",
            "accepts_artifact_types": ["ml.predictions"],
        },
        outputs=[
            {"type": "al.session", "description": "Updated session with the latest strategy-score artifact."},
            {"type": "ml.active_learning_scores", "description": "Whole-pool strategy scores for diagnostics."},
        ],
        params_schema={
            "type": "object",
            "properties": {
                "session_artifact_id": {"type": "string"},
                "predictions_artifact_id": {"type": "string"},
                "strategy_ids": {"type": "array", "items": {"type": "string"}, "description": "Use all registered strategies when omitted."},
                "seed": {"type": "integer", "default": 42},
                "exclude_row_ids": {"type": "array", "items": {"type": "string"}},
            },
        },
        run_in_job=True,
    )

    api.register_action(
        id="record_label",
        title="Record Active-Learning Label",
        handler=actions.record_label_action,
        description="Record a label/verification for a row. The special Unsure label excludes without training.",
        category="Active Learning",
        icon="label",
        tags=["active-learning", "label", "annotation", "selection"],
        inputs={"dataset": False, "selection": "optional", "columns": "none", "numeric_columns": "none"},
        outputs=[{"type": "al.session", "description": "Updated active-learning session state."}],
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
        id="bulk_label_next",
        title="Bulk Label Next Review Rows",
        handler=actions.bulk_label_next_action,
        description="Record each next review row's pre-assigned source value from the session label column.",
        category="Active Learning",
        icon="playlist_add_check",
        tags=["active-learning", "label", "annotation", "bulk"],
        inputs={"dataset": False, "selection": "optional", "columns": "none", "numeric_columns": "none"},
        outputs=[{"type": "al.session", "description": "Updated active-learning session state."}],
        params_schema={
            "type": "object",
            "required": ["session_artifact_id"],
            "properties": {
                "session_artifact_id": {"type": "string"},
                "row_id": {"type": "string", "description": "Optional start row; defaults to focused row or first unlabelled batch row."},
                "label_column": {"type": "string", "description": "Optional override; defaults to the session target/label column."},
                "n": {"type": "integer", "minimum": 1, "default": 5},
                "source": {"type": "string", "default": "bulk_column"},
            },
        },
        run_in_job=False,
    )

    api.register_action(
        id="materialize_training_set",
        title="Materialize Active-Learning Training Set",
        handler=actions.materialize_training_set_action,
        description="Create a derived dataset/artifact from verified labels without training a model.",
        category="Active Learning",
        icon="dataset",
        tags=["active-learning", "training-set", "dataset"],
        inputs={"dataset": False, "selection": "none", "columns": "none", "numeric_columns": "none"},
        outputs=[{"type": "al.training_set", "description": "Training rows and label manifest."}],
        params_schema={
            "type": "object",
            "required": ["session_artifact_id"],
            "properties": {
                "session_artifact_id": {"type": "string"},
                "target_column": {"type": "string", "default": "al_label"},
                "task_type": {"type": "string", "enum": ["classification", "regression", "auto"]},
                "train_dataset_id": {"type": "string"},
                "required_columns": {"type": "array", "items": {"type": "string"}},
                "recipe_params": {"type": "object"},
            },
        },
        run_in_job=True,
    )

    api.register_action(
        id="train_from_session",
        title="Train Active-Learning Round Through core.ml",
        handler=actions.train_from_session_action,
        description=(
            "Optional bridge: materialise verified labels, train through core.ml, and optionally "
            "predict over the pool. Querying is normally performed from the AL Query tab."
        ),
        category="Active Learning",
        icon="model_training",
        tags=["active-learning", "training", "ml", "recipe-profile", "scratch"],
        inputs={"dataset": False, "selection": "none", "columns": "none", "numeric_columns": "none"},
        outputs=[
            {"type": "al.session", "description": "Updated session with the completed AL round."},
            {"type": "al.training_set", "description": "Training rows and label manifest."},
            {"type": "ml.run", "optional": True, "description": "core.ml run summary."},
            {"type": "ml.model", "optional": True, "description": "Trained model artifact from core.ml."},
            {"type": "ml.predictions", "optional": True, "description": "Pool predictions generated after training."},
        ],
        params_schema={
            "type": "object",
            "required": ["session_artifact_id"],
            "properties": {
                "session_artifact_id": {"type": "string"},
                "recipe_profile_id": {"type": "string", "description": "Saved core.ml recipe profile to use for this AL round."},
                "recipe_profile_artifact_id": {"type": "string", "description": "Alias for recipe_profile_id when the profile artifact id is used."},
                "recipe_id": {"type": "string", "description": "Legacy fallback; prefer recipe_profile_id."},
                "recipe_params": {"type": "object"},
                "prediction_params": {"type": "object"},
                "target_column": {"type": "string", "default": "al_label"},
                "task_type": {"type": "string", "enum": ["classification", "regression", "auto"]},
                "problem_type": {"type": "string", "enum": ["classification", "regression", "auto"]},
                "label_profile": {"type": "object"},
                "train_dataset_id": {"type": "string"},
                "seed": {"type": "integer"},
                "auto_predict": {"type": "boolean", "default": True},
                "auto_query": {"type": "boolean", "default": False, "description": "Legacy/advanced option. The panel leaves this false so querying happens only from the Query tab."},
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
            "Manage AL sessions, labels, query batches, query strategies, and optional core.ml "
            "train/predict/query loops."
        ),
        category="Active Learning",
        icon="psychology",
        tags=["active-learning", "ml", "selection", "annotation", "query-strategy"],
        required_mappings=["record_id"],
        optional_mappings=["target_label", "image.path", "image.uri", "mask.path", "mask"],
        uses_services=["core.active_learning.query_strategy_registry", "core.ml.recipe_profile_store"],
        produces=[
            "al.session",
            "al.training_set",
            "ml.active_learning_batch",
            "ml.active_learning_scores",
            "selection.ids",
            "al.session.created",
            "al.query_batch.created",
            "al.strategy_scores.calculated",
            "al.label.recorded",
            "al.labels.bulk_recorded",
            "al.round.training_started",
            "al.round.training_finished",
            "al.round.training_failed",
            "ml.recipe_run.started",
            "ml.recipe_run.finished",
            "ml.training.started",
            "ml.training.finished",
        ],
        default_layout={"x": 0, "y": 0, "w": 6, "h": 8},
        state_version=6,
        persist_layout=True,
        persist_state=True,
        restore_policy="best_effort",
    )

def create_active_learning_panel(context, **kwargs):
    from . import panel as panel_module

    controller = panel_module.ActiveLearningPanel(context=context, restore_state=kwargs.get("restore_state"))
    return controller.panel(), controller
