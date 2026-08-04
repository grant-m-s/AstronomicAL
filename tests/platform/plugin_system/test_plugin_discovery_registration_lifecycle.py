from __future__ import annotations

import pytest

from astronomicAL.platform.context import AppContext
from astronomicAL.platform.plugins.errors import PluginLoadError, PluginValidationError
from astronomicAL.platform.plugins.manager import PluginManager
from astronomicAL.platform.plugins.specs import PluginStatus

from .conftest import recent_events_for


CONTRACT_PLUGIN = """
manifest = {
    "id": "test.contract",
    "name": "Contract Test Plugin",
    "version": "0.1.0",
    "description": "Exercises plugin registration against platform services.",
    "tags": ["test", "contract"],
}


class DetailController:
    def __init__(self, context):
        self.context = context
        self.focused_rows = []
        self.disposed = False
        self.sub = context.events.subscribe(
            "selection.focus.changed",
            self.on_focus,
            owner_id="test.contract.detail",
            owner_kind="panel",
            owner_label="DetailController",
        )

    def on_focus(self, topic, payload):
        self.focused_rows.append(payload["row_id"])

    def panel(self):
        return self

    def dispose(self):
        if not self.disposed:
            self.context.events.unsubscribe(self.sub)
            self.disposed = True


def make_client(context=None, manager=None, settings=None):
    return {
        "client": "ready",
        "settings": dict(settings or {}),
        "manager_attached": manager is not None,
    }


def make_detail_panel(context):
    return DetailController(context)


def run_noop(context, request, **kwargs):
    return {"dataset_id": request.dataset_id, "row_ids": request.row_ids}


def workflow_builder(context, **kwargs):
    context.events.publish("test.contract.workflow_built", {"ok": True})
    return {"workflow": "built"}


def artifact_viewer(context, artifact_ref, payload=None, **kwargs):
    return {
        "artifact_id": artifact_ref.id,
        "artifact_type": artifact_ref.type,
        "payload": payload,
    }


def register(api):
    api.register_service(
        key="client",
        factory=make_client,
        lazy=True,
        description="Test client service",
    )

    api.register_panel(
        id="detail",
        title="Detail",
        factory=make_detail_panel,
        required_mappings=["record_id"],
        optional_mappings=["target_label"],
        uses_services=["test.contract.client"],
        produces=["test.contract.detail"],
        persist_layout=True,
        persist_state=True,
        state_version=1,
    )

    api.register_action(
        id="noop",
        title="No-op",
        handler=run_noop,
        inputs={"dataset": True, "selection": "optional"},
        outputs=["test.contract.noop"],
        run_in_job=False,
    )

    api.register_workflow(
        id="demo",
        title="Demo workflow",
        builder=workflow_builder,
        description="Builds a test workflow",
    )

    api.register_artifact_viewer(
        id="detail-viewer",
        artifact_type="test.contract.detail",
        viewer_factory=artifact_viewer,
        title="Detail artifact viewer",
    )
"""


BROKEN_ON_ENABLE_PLUGIN = """
manifest = {
    "id": "test.broken_on_enable",
    "name": "Broken On Enable",
    "version": "0.1.0",
}


def make_client(context=None, manager=None, settings=None):
    return {"client": "should be rolled back"}


def panel_factory(context):
    return "panel"


def register(api):
    api.register_service(
        key="client",
        factory=make_client,
        lazy=True,
    )
    api.register_panel(
        id="panel",
        title="Panel",
        factory=panel_factory,
    )


def on_enable(context, manager, settings):
    raise RuntimeError("boom during on_enable")
"""


MISSING_DEP_PLUGIN = """
manifest = {
    "id": "test.missing_dep",
    "name": "Missing Dependency Plugin",
    "version": "0.1.0",
    "requires": ["definitely-not-installed-astronomical-test-package==0.0.1"],
}


def register(api):
    pass
"""


def test_local_plugin_discovery_and_full_registration(
    app_context: AppContext,
    make_manager,
) -> None:
    manager: PluginManager = make_manager("contract_plugin", CONTRACT_PLUGIN)

    discovered_ids = {info.id for info in manager.list_plugins()}
    assert "test.contract" in discovered_ids

    manager.enable("test.contract", app_context)
    info = manager.plugin_info("test.contract")
    assert info.status == PluginStatus.ENABLED

    panel_ids = {registration.id for registration in manager.list_panels()}
    action_ids = {registration.id for registration in manager.list_actions()}
    service_ids = {registration.key for registration in manager.list_services()}
    workflow_ids = {registration.id for registration in manager.list_workflows()}
    viewer_ids = {registration.id for registration in manager.list_artifact_viewers()}

    assert "test.contract.detail" in panel_ids
    assert "test.contract.noop" in action_ids
    assert "test.contract.client" in service_ids
    assert "test.contract.demo" in workflow_ids
    assert "test.contract.detail-viewer" in viewer_ids

    panel = manager.get_panel("test.contract.detail")
    assert panel.required_mappings == ["record_id"]
    assert panel.optional_mappings == ["target_label"]
    assert panel.uses_services == ["test.contract.client"]
    assert panel.produces == ["test.contract.detail"]
    assert panel.persist_layout is True
    assert panel.persist_state is True
    assert panel.state_version == 1

    client = app_context.services.get("test.contract.client")
    assert client["client"] == "ready"
    assert client["manager_attached"] is True

    enabled_events = recent_events_for(app_context, "plugin.enabled")
    assert any(event["plugin_id"] == "test.contract" for event in enabled_events)


def test_plugin_disable_removes_registrations_services_and_emits_event(
    app_context: AppContext,
    make_manager,
) -> None:
    manager: PluginManager = make_manager("contract_plugin", CONTRACT_PLUGIN)
    manager.enable("test.contract", app_context)

    assert app_context.services.get("test.contract.client") is not None

    manager.disable("test.contract", app_context)
    info = manager.plugin_info("test.contract")
    assert info.status == PluginStatus.DISABLED

    assert all(
        not registration.id.startswith("test.contract.")
        for registration in manager.list_panels()
    )
    assert all(
        not registration.id.startswith("test.contract.")
        for registration in manager.list_actions()
    )
    assert all(
        not registration.id.startswith("test.contract.")
        for registration in manager.list_services()
    )
    assert app_context.services.get("test.contract.client") is None

    disabled_events = recent_events_for(app_context, "plugin.disabled")
    assert any(event["plugin_id"] == "test.contract" for event in disabled_events)


def test_plugin_enable_rolls_back_registrations_when_on_enable_fails(
    app_context: AppContext,
    make_manager,
) -> None:
    manager: PluginManager = make_manager(
        "broken_on_enable_plugin",
        BROKEN_ON_ENABLE_PLUGIN,
    )

    with pytest.raises(PluginLoadError):
        manager.enable("test.broken_on_enable", app_context)

    assert app_context.services.get("test.broken_on_enable.client") is None
    assert all(
        not registration.id.startswith("test.broken_on_enable.")
        for registration in manager.list_panels()
    )
    assert all(
        not registration.id.startswith("test.broken_on_enable.")
        for registration in manager.list_services()
    )


def test_missing_dependency_plugin_fails_validation_without_crashing_discovery(
    app_context: AppContext,
    make_manager,
) -> None:
    manager: PluginManager = make_manager("missing_dep_plugin", MISSING_DEP_PLUGIN)

    assert "test.missing_dep" in {info.id for info in manager.list_plugins()}

    with pytest.raises(PluginValidationError):
        manager.enable("test.missing_dep", app_context)