from __future__ import annotations

import pytest

from astronomicAL.platform.context import AppContext
from astronomicAL.platform.plugins.errors import PluginValidationError
from astronomicAL.platform.plugins.manager import PluginManager

from .conftest import recent_events_for


ACTIONS_PLUGIN = """
manifest = {
    "id": "test.outputs",
    "name": "Action Output Test Plugin",
    "version": "0.1.0",
}


def make_score(context, request, **kwargs):
    return {
        "dataset_id": request.dataset_id,
        "row_ids": list(request.row_ids or []),
        "params": dict(request.params or {}),
    }


def make_dataset(context, request, **kwargs):
    import pandas as pd
    from astronomicAL.platform.plugins.specs import DatasetResult

    df = pd.DataFrame(
        {
            "id": ["derived-1", "derived-2"],
            "score": [0.9, 0.1],
        }
    )

    return DatasetResult(
        id="derived_scores",
        dataframe=df,
        name="Derived Scores",
        set_active=True,
        metadata={"source_action": "make-dataset"},
    )


def publish_event(context, request, **kwargs):
    from astronomicAL.platform.plugins.specs import EventResult

    return EventResult(
        topic="test.outputs.custom_event",
        payload={"dataset_id": request.dataset_id, "ok": True},
    )


def numeric_summary(context, request, **kwargs):
    return {
        "dataset_id": request.dataset_id,
        "columns": list(request.columns or []),
        "row_ids": list(request.row_ids or []),
    }


def register(api):
    api.register_action(
        id="make-score",
        title="Make Score",
        handler=make_score,
        inputs={"dataset": True, "selection": "optional"},
        outputs=["test.outputs.score"],
        run_in_job=False,
        params_schema={
            "type": "object",
            "properties": {
                "scale": {"type": "number", "default": 1.0}
            },
            "additionalProperties": False,
        },
    )

    api.register_action(
        id="make-dataset",
        title="Make Dataset",
        handler=make_dataset,
        inputs={"dataset": True},
        run_in_job=False,
    )

    api.register_action(
        id="publish-event",
        title="Publish Event",
        handler=publish_event,
        inputs={"dataset": True},
        run_in_job=False,
    )

    api.register_action(
        id="numeric-summary",
        title="Numeric Summary",
        handler=numeric_summary,
        inputs={
            "dataset": True,
            "selection": "required",
            "columns": "one",
            "numeric_columns": "one",
        },
        outputs=["test.outputs.summary"],
        run_in_job=False,
    )
"""


def _enabled_manager(app_context: AppContext, make_manager) -> PluginManager:
    manager: PluginManager = make_manager("actions_plugin", ACTIONS_PLUGIN)
    manager.enable("test.outputs", app_context)
    return manager


def test_raw_action_result_is_promoted_to_artifact(
    app_context: AppContext,
    make_manager,
) -> None:
    manager = _enabled_manager(app_context, make_manager)

    result = manager.run_action(
        "test.outputs.make-score",
        app_context,
        request={
            "dataset_id": "main",
            "row_ids": ["a"],
            "params": {},
        },
        return_processed=True,
    )

    assert result.artifact_ids
    artifact_id = result.artifact_ids[0]

    artifact_ref = app_context.artifacts.ref(artifact_id)
    assert artifact_ref.type == "test.outputs.score"
    assert artifact_ref.dataset_id == "main"
    assert artifact_ref.row_ids == ["a"]

    payload = app_context.artifacts.get(artifact_id)
    assert payload["dataset_id"] == "main"
    assert payload["row_ids"] == ["a"]
    assert payload["params"]["scale"] == 1.0

    artifact_events = recent_events_for(app_context, "artifact.created")
    assert any(event["artifact_id"] == artifact_id for event in artifact_events)


def test_action_params_schema_rejects_unknown_parameter(
    app_context: AppContext,
    make_manager,
) -> None:
    manager = _enabled_manager(app_context, make_manager)

    with pytest.raises(PluginValidationError):
        manager.run_action(
            "test.outputs.make-score",
            app_context,
            request={
                "dataset_id": "main",
                "row_ids": ["a"],
                "params": {"unknown": 123},
            },
        )


def test_dataset_result_registers_and_can_activate_dataset(
    app_context: AppContext,
    make_manager,
) -> None:
    manager = _enabled_manager(app_context, make_manager)

    result = manager.run_action(
        "test.outputs.make-dataset",
        app_context,
        request={"dataset_id": "main"},
        return_processed=True,
    )

    assert result.dataset_ids == ["derived_scores"]
    assert app_context.datasets.active_id() == "derived_scores"

    derived = app_context.datasets.get_df("derived_scores")
    assert list(derived["id"]) == ["derived-1", "derived-2"]
    assert list(derived["score"]) == [0.9, 0.1]


def test_event_result_publishes_event(
    app_context: AppContext,
    make_manager,
) -> None:
    manager = _enabled_manager(app_context, make_manager)

    result = manager.run_action(
        "test.outputs.publish-event",
        app_context,
        request={"dataset_id": "main"},
        return_processed=True,
    )

    assert len(result.events) == 1
    assert result.events[0].topic == "test.outputs.custom_event"
    assert result.events[0].payload == {"dataset_id": "main", "ok": True}

    events = recent_events_for(app_context, "test.outputs.custom_event")
    assert events[-1] == {"dataset_id": "main", "ok": True}


def test_action_input_validation_uses_selection_and_numeric_column_rules(
    app_context: AppContext,
    make_manager,
) -> None:
    manager = _enabled_manager(app_context, make_manager)

    with pytest.raises(PluginValidationError):
        manager.run_action(
            "test.outputs.numeric-summary",
            app_context,
            request={"dataset_id": "main", "columns": ["value"]},
        )

    app_context.selection.set_focus("main", "b", origin="test")

    result = manager.run_action(
        "test.outputs.numeric-summary",
        app_context,
        request={"dataset_id": "main", "columns": ["value"]},
        return_processed=True,
    )

    payload = app_context.artifacts.get(result.artifact_ids[0])
    assert payload["dataset_id"] == "main"
    assert payload["columns"] == ["value"]
    assert payload["row_ids"] == ["b"]

    with pytest.raises(PluginValidationError):
        manager.run_action(
            "test.outputs.numeric-summary",
            app_context,
            request={"dataset_id": "main", "columns": ["name"]},
        )