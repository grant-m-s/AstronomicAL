from __future__ import annotations

from astronomicAL.platform.context import AppContext
from astronomicAL.platform.plugins.manager import PluginManager

VIEWER_WORKFLOW_PLUGIN = """
manifest = {
    "id": "test.viewer_workflow",
    "name": "Viewer and Workflow Test Plugin",
    "version": "0.1.0",
}


def make_payload(context, request, **kwargs):
    return {"message": "hello", "dataset_id": request.dataset_id}


def view_test_artifact(context, artifact_id, **kwargs):
    artifact_ref = context.artifacts.ref(artifact_id)
    payload = context.artifacts.get(artifact_id)
    return {
        "viewed_id": artifact_ref.artifact_id,
        "viewed_type": artifact_ref.type,
        "payload": payload,
    }


def build_workflow(context, request=None, **kwargs):
    request = dict(request or {})
    context.events.publish(
        "test.viewer_workflow.workflow_built",
        {"request": request},
    )
    return {"status": "built", "request": request}


def register(api):
    api.register_action(
        id="make-payload",
        title="Make Payload",
        handler=make_payload,
        inputs={"dataset": True},
        outputs=["test.viewer_workflow.payload"],
        run_in_job=False,
    )

    api.register_artifact_viewer(
        id="payload-viewer",
        artifact_type="test.viewer_workflow.payload",
        viewer_factory=view_test_artifact,
        title="Payload Viewer",
    )

    api.register_workflow(
        id="demo-workflow",
        title="Demo Workflow",
        builder=build_workflow,
        description="Small workflow used in tests",
    )
"""

def _enabled_manager(app_context: AppContext, make_manager) -> PluginManager:
    manager: PluginManager = make_manager(
        "viewer_workflow_plugin",
        VIEWER_WORKFLOW_PLUGIN,
    )
    manager.enable("test.viewer_workflow", app_context)
    return manager


def test_artifact_viewer_registration_can_render_matching_artifact(
    app_context: AppContext,
    make_manager,
) -> None:
    manager = _enabled_manager(app_context, make_manager)

    result = manager.run_action(
        "test.viewer_workflow.make-payload",
        app_context,
        request={"dataset_id": "main"},
        return_processed=True,
    )

    artifact_id = result.artifact_ids[0]
    view, controller = manager.create_artifact_viewer(
        "test.viewer_workflow.payload",
        app_context,
        artifact_id,
    )

    assert controller is view
    assert view["viewed_id"] == artifact_id
    assert view["viewed_type"] == "test.viewer_workflow.payload"
    assert view["payload"] == {"message": "hello", "dataset_id": "main"}


def test_workflow_registration_can_be_invoked_without_core_coupling(
    app_context: AppContext,
    make_manager,
) -> None:
    manager = _enabled_manager(app_context, make_manager)

    workflow = manager.get_workflow("test.viewer_workflow.demo-workflow")
    assert workflow.title == "Demo Workflow"

    output = manager.build_workflow(
        "test.viewer_workflow.demo-workflow",
        app_context,
        request={"mode": "test"},
    )
    assert output == {"status": "built", "request": {"mode": "test"}}

    workflow_events = [
        payload
        for _, topic, payload in app_context.events.recent_events(100)
        if topic == "test.viewer_workflow.workflow_built"
    ]
    assert workflow_events[-1] == {"request": {"mode": "test"}}