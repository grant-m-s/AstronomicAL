from __future__ import annotations

from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="core.active_learning",
    name="Active Learning",
    version="0.5.4",
    description=(
        "Active-learning sessions, labels, streaming query strategies, selection "
        "handoff, and an optional core.ml train/predict bridge."
    ),
    requires_plugins=[],
    requires=[],
    optional_requires=[],
    capabilities=["panel", "action", "active-learning", "selection", "machine-learning"],
    tags=["core", "active-learning", "ml", "selection", "query-strategy"],
)

def register(api) -> None:
    from . import actions
    from . import session_partitions
    from . import strategies
    from . import streaming_actions

    session_partitions.install_legacy_bridge()
    streaming_actions.install_legacy_bridge()

    api.register_service(
        key="query_strategy_registry",
        factory=lambda context: strategies.create_default_strategy_registry(),
        lazy=True,
        replace=True,
        description="Registry for active-learning query strategies.",
    )
    api.register_action(
        id="start_session",
        title="Start Active-Learning Session",
        handler=session_partitions.start_session_action,
        description=(
            "Create pool, validation, and test partitions in a cancellable background "
            "job, retain the selected source as the active dataset, and draw the "
            "initial review batch from the session pool."
        ),
        category="Active Learning",
        icon="playlist_add",
        tags=["active-learning", "random", "selection", "session"],
        inputs={"dataset": True, "selection": "none", "columns": "none", "numeric_columns": "none", "required_mappings": ["record_id"]},
        outputs=[{"type": "al.session"}, {"type": "ml.active_learning_batch"}, {"type": "selection.ids", "optional": True}],
        params_schema={
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "pool_dataset_id": {"type": "string"},
                "target_column": {"type": "string"},
                "label_column": {"type": "string"},
                "label_options": {"type": "array", "items": {"type": "string"}},
                "task_type": {"type": "string", "enum": ["classification", "regression", "auto"], "default": "auto"},
                "problem_type": {"type": "string", "enum": ["classification", "regression", "auto"]},
                "label_profile": {"type": "object"},
                "infer_labels_from_column": {"type": "boolean", "default": True},
                "initial_k": {"type": "integer", "minimum": 0, "default": 20},
                "seed": {"type": "integer", "default": 42},
                "make_selection": {"type": "boolean", "default": True},
                "partition_whole_dataset": {"type": "boolean", "default": True},
                "session_validation_size": {"type": "number", "minimum": 0.0, "exclusiveMaximum": 1.0, "default": 0.1},
                "session_test_size": {"type": "number", "minimum": 0.0, "exclusiveMaximum": 1.0, "default": 0.2},
                "session_pool_dataset_id": {"type": "string"},
                "session_validation_dataset_id": {"type": "string"},
                "session_test_dataset_id": {"type": "string"},
                "session_split_output_dir": {"type": "string"},
                "session_split_scan_batch_size": {"type": "integer", "minimum": 1, "default": 65536},
                "session_split_output_batch_size": {"type": "integer", "minimum": 1, "default": 8192},
                "recipe_profile_id": {"type": "string"},
                "recipe_id": {"type": "string"},
                "session_contract": {"type": "object"},
            },
        },
        run_in_job=True,
    )
    api.register_action(
        id="profile_data_contract",
        title="Inspect Active-Learning Data Contract",
        handler=actions.profile_data_contract_action,
        description="Inspect dataset and recipe bindings for the optional core.ml bridge.",
        category="Active Learning",
        icon="fact_check",
        tags=["active-learning", "recipe", "data-contract"],
        inputs={"dataset": True, "selection": "none", "columns": "optional", "numeric_columns": "none", "required_mappings": ["record_id"]},
        outputs=[],
        params_schema={
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "validation_dataset_id": {"type": "string"},
                "test_dataset_id": {"type": "string"},
                "recipe_profile_id": {"type": "string"},
                "recipe_profile_artifact_id": {"type": "string"},
                "recipe_id": {"type": "string"},
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
        handler=streaming_actions.query_batch_action,
        description=(
            "Stream prediction rows, retain only the ranked top-k candidates, "
            "and focus the first review record on the active source or pool dataset."
        ),
        category="Active Learning",
        icon="rule",
        tags=["active-learning", "query", "selection", "uncertainty", "streaming"],
        inputs={"dataset": False, "selection": "none", "columns": "none", "numeric_columns": "none", "accepts_artifact_types": ["ml.predictions"]},
        outputs=[{"type": "al.session"}, {"type": "ml.active_learning_batch"}, {"type": "selection.ids", "optional": True}],
        params_schema={
            "type": "object",
            "required": ["session_artifact_id"],
            "properties": {
                "session_artifact_id": {"type": "string"},
                "predictions_artifact_id": {"type": "string"},
                "strategy_id": {"type": "string", "default": "least_confidence"},
                "strategy_params": {"type": "object"},
                "k": {"type": "integer", "minimum": 1, "default": 200},
                "seed": {"type": "integer", "default": 42},
                "make_selection": {"type": "boolean", "default": True},
                "exclude_row_ids": {"type": "array", "items": {"type": "string"}},
                "prediction_scan_batch_size": {"type": "integer", "minimum": 1, "default": 8192},
                "batch_strategy_max_rows": {"type": "integer", "minimum": 1, "default": 100000},
            },
        },
        run_in_job=True,
    )
    api.register_action(
        id="score_pool",
        title="Calculate Query-Strategy Scores",
        handler=streaming_actions.score_pool_action,
        description="Stream whole-pool per-record scores into a durable score table.",
        category="Active Learning",
        icon="analytics",
        tags=["active-learning", "query", "strategy", "diagnostics", "streaming"],
        inputs={"dataset": False, "selection": "none", "columns": "none", "numeric_columns": "none", "accepts_artifact_types": ["ml.predictions"]},
        outputs=[{"type": "al.session"}, {"type": "ml.active_learning_scores"}],
        params_schema={
            "type": "object",
            "required": ["session_artifact_id"],
            "properties": {
                "session_artifact_id": {"type": "string"},
                "predictions_artifact_id": {"type": "string"},
                "strategy_ids": {"type": "array", "items": {"type": "string"}},
                "seed": {"type": "integer", "default": 42},
                "exclude_row_ids": {"type": "array", "items": {"type": "string"}},
                "prediction_scan_batch_size": {"type": "integer", "minimum": 1, "default": 8192},
                "score_output_batch_size": {"type": "integer", "minimum": 1, "default": 8192},
                "score_storage_format": {"type": "string", "enum": ["auto", "parquet", "jsonl.gz"], "default": "auto"},
                "score_preview_limit": {"type": "integer", "minimum": 0, "default": 1000},
            },
        },
        run_in_job=True,
    )
    api.register_action(
        id="record_label",
        title="Record Active-Learning Label",
        handler=actions.record_label_action,
        description="Record a label or verification for one row.",
        category="Active Learning",
        icon="label",
        tags=["active-learning", "label", "annotation"],
        inputs={"dataset": False, "selection": "optional", "columns": "none", "numeric_columns": "none"},
        outputs=[{"type": "al.session"}],
        params_schema={"type": "object", "required": ["session_artifact_id", "label"], "properties": {"session_artifact_id": {"type": "string"}, "row_id": {"type": "string"}, "label": {}, "source": {"type": "string", "default": "manual"}}},
        run_in_job=False,
    )
    api.register_action(
        id="bulk_label_next",
        title="Bulk Label Next Review Rows",
        handler=actions.bulk_label_next_action,
        description="Record source-column labels for the next review rows.",
        category="Active Learning",
        icon="playlist_add_check",
        tags=["active-learning", "label", "bulk"],
        inputs={"dataset": False, "selection": "optional", "columns": "none", "numeric_columns": "none"},
        outputs=[{"type": "al.session"}],
        params_schema={"type": "object", "required": ["session_artifact_id"], "properties": {"session_artifact_id": {"type": "string"}, "row_id": {"type": "string"}, "label_column": {"type": "string"}, "n": {"type": "integer", "minimum": 1, "default": 5}, "source": {"type": "string", "default": "bulk_column"}}},
        run_in_job=False,
    )
    api.register_action(
        id="materialize_training_set",
        title="Prepare Active-Learning Training Rows",
        handler=streaming_actions.materialize_training_set_action,
        description=(
            "Attach verified labels to the existing session pool dataset and "
            "publish the labelled training-row membership without registering "
            "a round-specific dataset."
        ),
        category="Active Learning",
        icon="dataset",
        tags=["active-learning", "training-set", "column-overlay", "streaming"],
        inputs={"dataset": False, "selection": "none", "columns": "none", "numeric_columns": "none"},
        outputs=[{"type": "al.training_set"}],
        params_schema={
            "type": "object",
            "required": ["session_artifact_id"],
            "properties": {
                "session_artifact_id": {"type": "string"},
                "target_column": {"type": "string", "default": "al_label"},
                "task_type": {"type": "string", "enum": ["classification", "regression", "auto"]},
                "problem_type": {"type": "string", "enum": ["classification", "regression", "auto"]},
                "label_profile": {"type": "object"},
                "validation_dataset_id": {"type": "string"},
                "test_dataset_id": {"type": "string"},
                "required_columns": {"type": "array", "items": {"type": "string"}},
                "recipe_params": {"type": "object"},
                "record_id_column": {"type": "string"},
                "training_lookup_batch_size": {"type": "integer", "minimum": 1, "default": 4096},
                "training_artifact_preview_limit": {"type": "integer", "minimum": 0, "default": 1000},
            },
        },
        run_in_job=True,
    )
    api.register_action(
        id="train_from_session",
        title="Train Active-Learning Round Through core.ml",
        handler=streaming_actions.train_from_session_action,
        description=(
            "Reuse the session pool dataset, train core.ml only on the verified "
            "labelled row IDs, and optionally predict over the pool."
        ),
        category="Active Learning",
        icon="model_training",
        tags=["active-learning", "training", "ml", "streaming"],
        inputs={"dataset": False, "selection": "none", "columns": "none", "numeric_columns": "none"},
        outputs=[{"type": "al.session"}, {"type": "al.training_set"}, {"type": "ml.run", "optional": True}, {"type": "ml.model", "optional": True}, {"type": "ml.predictions", "optional": True}],
        params_schema={
            "type": "object",
            "required": ["session_artifact_id"],
            "properties": {
                "session_artifact_id": {"type": "string"},
                "recipe_profile_id": {"type": "string"},
                "recipe_profile_artifact_id": {"type": "string"},
                "recipe_id": {"type": "string"},
                "recipe_params": {"type": "object"},
                "prediction_params": {"type": "object"},
                "target_column": {"type": "string", "default": "al_label"},
                "task_type": {"type": "string", "enum": ["classification", "regression", "auto"]},
                "problem_type": {"type": "string", "enum": ["classification", "regression", "auto"]},
                "label_profile": {"type": "object"},
                "seed": {"type": "integer"},
                "auto_predict": {"type": "boolean", "default": True},
                "auto_query": {"type": "boolean", "default": False},
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
        description="Manage sessions, labels, training rounds, predictions, and query batches.",
        category="Active Learning",
        icon="psychology",
        tags=["active-learning", "ml", "selection", "annotation", "query-strategy"],
        required_mappings=["record_id"],
        optional_mappings=[],
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
        state_version=8,
        persist_layout=True,
        persist_state=True,
        restore_policy="best_effort",
    )

def create_active_learning_panel(context, **kwargs):
    from .job_backed_panel import JobBackedActiveLearningPanel

    controller = JobBackedActiveLearningPanel(
        context=context,
        restore_state=kwargs.get("restore_state"),
    )
    return controller.panel(), controller