from __future__ import annotations

import pytest

from astronomicAL.platform.context import AppContext
from astronomicAL.platform.plugins.errors import PluginValidationError
from astronomicAL.platform.plugins.manager import PluginManager

from .conftest import recent_events_for


MAPPING_SELECTION_PLUGIN = """
manifest = {
    "id": "test.mapping_selection",
    "name": "Mapping and Selection Test Plugin",
    "version": "0.1.0",
}


class SelectionRecorder:
    def __init__(self, context):
        self.context = context
        self.focused_rows = []
        self.disposed = False
        self.sub = context.events.subscribe(
            "selection.focus.changed",
            self.on_focus,
            owner_id="test.mapping_selection.recorder",
            owner_kind="panel",
            owner_label="SelectionRecorder",
        )

    def on_focus(self, topic, payload):
        self.focused_rows.append(payload["row_id"])

    def panel(self):
        return self

    def dispose(self):
        if not self.disposed:
            self.context.events.unsubscribe(self.sub)
            self.disposed = True


def make_panel(context):
    return SelectionRecorder(context)


def needs_domain_mapping(context, request, **kwargs):
    return {
        "dataset_id": request.dataset_id,
        "mapping": context.datasets.get_mapping(request.dataset_id, "domain_key"),
    }


def register(api):
    api.register_panel(
        id="selection-recorder",
        title="Selection Recorder",
        factory=make_panel,
        required_mappings=["record_id"],
    )

    api.register_action(
        id="needs-domain-mapping",
        title="Needs Domain Mapping",
        handler=needs_domain_mapping,
        inputs={
            "dataset": True,
            "required_mappings": ["domain_key"],
        },
        outputs=["test.mapping_selection.mapping_check"],
        run_in_job=False,
    )
"""


def _enabled_manager(app_context: AppContext, make_manager) -> PluginManager:
    manager: PluginManager = make_manager(
        "mapping_selection_plugin",
        MAPPING_SELECTION_PLUGIN,
    )
    manager.enable("test.mapping_selection", app_context)
    return manager


def test_action_required_mapping_fails_until_dataset_mapping_is_resolved(
    app_context: AppContext,
    make_manager,
) -> None:
    manager = _enabled_manager(app_context, make_manager)

    with pytest.raises(PluginValidationError):
        manager.run_action(
            "test.mapping_selection.needs-domain-mapping",
            app_context,
            request={"dataset_id": "main"},
        )

    app_context.datasets.set_mapping("main", "domain_key", "name")

    result = manager.run_action(
        "test.mapping_selection.needs-domain-mapping",
        app_context,
        request={"dataset_id": "main"},
        return_processed=True,
    )

    payload = app_context.artifacts.get(result.artifact_ids[0])
    assert payload["dataset_id"] == "main"
    assert payload["mapping"] == "name"


def test_selection_aware_panel_receives_focus_events_and_cleans_up(
    app_context: AppContext,
    make_manager,
) -> None:
    manager = _enabled_manager(app_context, make_manager)

    registration = manager.get_panel("test.mapping_selection.selection-recorder")
    controller = registration.factory(app_context)

    assert controller.focused_rows == []

    app_context.selection.set_focus("main", "b", origin="test")
    assert controller.focused_rows == ["b"]

    controller.dispose()
    app_context.selection.set_focus("main", "c", origin="test")

    assert controller.focused_rows == ["b"]
    assert controller.disposed is True


def test_selection_set_deduplicates_rows_creates_artifact_and_publishes_event(
    app_context: AppContext,
) -> None:
    state = app_context.selection.set_selection_set(
        "main",
        ["a", "b", "a"],
        origin="lasso-test",
    )

    assert state.dataset_id == "main"
    assert state.row_ids == ["a", "b"]
    assert state.artifact_id is not None

    artifact_payload = app_context.artifacts.get(state.artifact_id)
    assert artifact_payload["dataset_id"] == "main"
    assert artifact_payload["row_ids"] == ["a", "b"]
    assert artifact_payload["origin"] == "lasso-test"

    events = recent_events_for(app_context, "selection.set.changed")
    assert events
    assert events[-1]["dataset_id"] == "main"
    assert events[-1]["row_ids"] == ["a", "b"]


def test_focus_can_be_cleared_and_is_announced(
    app_context: AppContext,
) -> None:
    app_context.selection.set_focus("main", "a", origin="test")
    assert app_context.selection.get_focus().row_id == "a"

    cleared_return = app_context.selection.clear_focus(origin="test-clear")
    current_focus = app_context.selection.get_focus()

    assert cleared_return is None

    assert current_focus.dataset_id is None
    assert current_focus.row_id is None
    assert current_focus.origin == "test-clear"

    events = recent_events_for(app_context, "selection.focus.cleared")
    assert events
    assert events[-1]["dataset_id"] == "main"
    assert events[-1]["origin"] == "test-clear"