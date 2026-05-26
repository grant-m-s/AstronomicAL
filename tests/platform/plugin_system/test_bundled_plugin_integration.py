from __future__ import annotations

from typing import Any

import pytest

from astronomicAL.platform.context import AppContext
from astronomicAL.platform.plugins.manager import PluginManager
from astronomicAL.platform.plugins.specs import PluginStatus

from .bundled_plugin_helpers import (
    EXPECTED_CORE_PLUGIN_IDS,
    core_panel_registrations,
    enable_core_plugins,
    ensure_required_mappings_for_panel,
    format_subscription_leaks,
    get_workspace_record,
    leaked_subscriptions_after,
    make_bundled_plugin_manager,
    make_fake_workspace_manager_compatible,
    plugin_panel_registrations,
    prepare_context_for_bundled_plugins,
    subscription_counter,
    subscription_count,
    wait_for_workspace_panel_kind,
)


@pytest.fixture
def bundled_manager(app_context: AppContext) -> PluginManager:
    prepare_context_for_bundled_plugins(app_context)
    make_fake_workspace_manager_compatible(app_context)

    manager = make_bundled_plugin_manager(app_context)
    enable_core_plugins(manager, app_context)

    return manager


def test_bundled_core_plugins_discover_and_enable(
    app_context: AppContext,
) -> None:
    prepare_context_for_bundled_plugins(app_context)

    manager = make_bundled_plugin_manager(app_context)
    enabled_ids = enable_core_plugins(manager, app_context)

    assert set(enabled_ids) == EXPECTED_CORE_PLUGIN_IDS

    for plugin_id in EXPECTED_CORE_PLUGIN_IDS:
        info = manager.plugin_info(plugin_id)
        assert info.status == PluginStatus.ENABLED
        assert info.name
        assert info.version

    registered_panel_plugin_ids = {
        registration.plugin_id for registration in manager.list_panels()
    }

    assert EXPECTED_CORE_PLUGIN_IDS <= registered_panel_plugin_ids


def test_bundled_core_panel_registrations_are_workspace_ready(
    bundled_manager: PluginManager,
) -> None:
    panels = core_panel_registrations(bundled_manager)

    assert panels, "No bundled core panel registrations were found"

    for registration in panels:
        assert registration.plugin_id in EXPECTED_CORE_PLUGIN_IDS
        assert registration.id.startswith(f"{registration.plugin_id}.")
        assert registration.title
        assert callable(registration.factory)

        # These are important for workspace persistence and plugin UI menus.
        assert registration.state_version >= 1
        assert isinstance(registration.persist_layout, bool)
        assert isinstance(registration.persist_state, bool)

        # Mapping requirements should be declarative, not hidden in factories.
        assert registration.required_mappings is not None
        assert registration.optional_mappings is not None


def test_open_and_close_every_bundled_core_panel_through_plugin_manager(
    app_context: AppContext,
    bundled_manager: PluginManager,
) -> None:
    panels = core_panel_registrations(bundled_manager)
    assert panels, "No core plugin panels are available to open"

    opened: list[str] = []

    for registration in panels:
        ensure_required_mappings_for_panel(app_context, registration)

        workspace_id = bundled_manager.open_panel(
            registration.id,
            context=app_context,
            instance_id=f"test-open:{registration.id}",
        )
        opened.append(workspace_id)

        record = wait_for_workspace_panel_kind(app_context, workspace_id)

        assert record.panel_id == workspace_id
        assert record.plugin_id == registration.plugin_id
        assert record.registration_id == registration.id
        assert record.controller is not None

        metadata = record.metadata or {}
        assert metadata.get("kind") == "plugin_panel"

        nested_metadata = metadata.get("metadata") or {}
        assert nested_metadata.get("panel_registration_title") or registration.title
        assert nested_metadata.get("restore_policy") or registration.restore_policy

        app_context.workspace.remove_panel(workspace_id)
        assert get_workspace_record(app_context, workspace_id) is None

    assert opened


def test_open_core_panels_then_emit_selection_and_dataset_events(
    app_context: AppContext,
    bundled_manager: PluginManager,
) -> None:
    panels = core_panel_registrations(bundled_manager)
    assert panels

    opened: list[str] = []

    for registration in panels:
        ensure_required_mappings_for_panel(app_context, registration)

        workspace_id = bundled_manager.open_panel(
            registration.id,
            context=app_context,
            instance_id=f"test-events:{registration.id}",
        )
        wait_for_workspace_panel_kind(app_context, workspace_id)
        opened.append(workspace_id)

    app_context.selection.set_focus("main", "b", origin="integration-test")
    focus = app_context.selection.get_focus()

    assert focus.dataset_id == "main"
    assert focus.row_id == "b"
    assert focus.origin == "integration-test"

    app_context.selection.set_selection_set(
        "main",
        ["a", "c"],
        origin="integration-test",
    )
    active_set = app_context.selection.get_active_set()

    assert active_set.dataset_id == "main"
    assert active_set.row_ids == ["a", "c"]

    app_context.events.publish(
        "dataset.updated",
        {
            "dataset_id": "main",
            "origin": "integration-test",
        },
    )

    failed_events = [
        payload
        for _, topic, payload in app_context.events.recent_events(500)
        if topic == "plugin.panel.open_failed"
    ]
    assert not failed_events

    for workspace_id in opened:
        app_context.workspace.remove_panel(workspace_id)
        assert get_workspace_record(app_context, workspace_id) is None


def test_record_browser_opens_and_publishes_initial_focus(
    app_context: AppContext,
    bundled_manager: PluginManager,
) -> None:
    registration_id = "core.record_browser.panel"

    registration = bundled_manager.get_panel(registration_id)
    ensure_required_mappings_for_panel(app_context, registration)

    workspace_id = bundled_manager.open_panel(
        registration_id,
        context=app_context,
        instance_id="test-record-browser",
    )
    record = wait_for_workspace_panel_kind(app_context, workspace_id)

    assert record.plugin_id == "core.record_browser"
    assert record.registration_id == registration_id

    focus = app_context.selection.get_focus()

    assert focus.dataset_id == "main"
    assert focus.row_id == "a"

    app_context.workspace.remove_panel(workspace_id)
    assert get_workspace_record(app_context, workspace_id) is None


@pytest.mark.parametrize(
    "plugin_id",
    sorted(EXPECTED_CORE_PLUGIN_IDS),
)
def test_disabling_each_core_plugin_removes_its_open_panels_and_registrations(
    app_context: AppContext,
    plugin_id: str,
) -> None:
    prepare_context_for_bundled_plugins(app_context)
    make_fake_workspace_manager_compatible(app_context)

    manager = make_bundled_plugin_manager(app_context)
    enable_core_plugins(manager, app_context)

    panels = plugin_panel_registrations(manager, [plugin_id])
    assert panels, f"{plugin_id} registered no panels"

    opened: list[str] = []

    for registration in panels:
        ensure_required_mappings_for_panel(app_context, registration)

        workspace_id = manager.open_panel(
            registration.id,
            context=app_context,
            instance_id=f"test-disable:{registration.id}",
        )
        wait_for_workspace_panel_kind(app_context, workspace_id)
        opened.append(workspace_id)

    manager.disable(plugin_id, app_context)

    info = manager.plugin_info(plugin_id)
    assert info.status == PluginStatus.DISABLED

    for workspace_id in opened:
        assert get_workspace_record(app_context, workspace_id) is None

    assert all(
        registration.plugin_id != plugin_id
        for registration in manager.list_panels()
    )
    assert all(
        registration.plugin_id != plugin_id
        for registration in manager.list_actions()
    )
    assert all(
        registration.plugin_id != plugin_id
        for registration in manager.list_workflows()
    )
    assert all(
        registration.plugin_id != plugin_id
        for registration in manager.list_services()
    )


def test_panel_close_does_not_leave_event_subscription_leaks(
    app_context: AppContext,
    bundled_manager: PluginManager,
) -> None:
    """
    Open panels, trigger events, close panels, and ensure the EventBus subscriber
    set returns to its previous state.

    This catches real plugin controllers that subscribe to events but forget to
    unsubscribe in dispose().
    """
    panels = core_panel_registrations(bundled_manager)
    assert panels

    before_counter = subscription_counter(app_context)
    before_count = subscription_count(app_context)

    opened: list[str] = []

    for registration in panels:
        ensure_required_mappings_for_panel(app_context, registration)

        workspace_id = bundled_manager.open_panel(
            registration.id,
            context=app_context,
            instance_id=f"test-leak:{registration.id}",
        )
        wait_for_workspace_panel_kind(app_context, workspace_id)
        opened.append(workspace_id)

    during_count = subscription_count(app_context)
    assert during_count >= before_count

    app_context.selection.set_focus("main", "c", origin="leak-test")
    app_context.selection.set_selection_set("main", ["a", "b"], origin="leak-test")
    app_context.events.publish("dataset.updated", {"dataset_id": "main"})

    for workspace_id in opened:
        app_context.workspace.remove_panel(workspace_id)

    after_counter = subscription_counter(app_context)
    after_count = subscription_count(app_context)

    leaks = leaked_subscriptions_after(before_counter, after_counter)

    assert not leaks, (
        "Closing bundled core plugin panels left EventBus subscriptions behind: "
        f"before={before_count}, during={during_count}, after={after_count}\n"
        f"{format_subscription_leaks(leaks)}"
    )


def test_core_plugin_manager_panel_can_inspect_enabled_plugins(
    app_context: AppContext,
    bundled_manager: PluginManager,
) -> None:
    """
    The plugin-manager plugin is a useful canary because it depends on
    context.plugins being present and populated.
    """
    registration_id = "core.plugin_manager.panel"

    registration = bundled_manager.get_panel(registration_id)
    ensure_required_mappings_for_panel(app_context, registration)

    workspace_id = bundled_manager.open_panel(
        registration_id,
        context=app_context,
        instance_id="test-plugin-manager-panel",
    )
    record = wait_for_workspace_panel_kind(app_context, workspace_id)

    assert record.plugin_id == "core.plugin_manager"
    assert record.registration_id == registration_id
    assert app_context.plugins is bundled_manager

    enabled_plugin_ids = {
        info.id
        for info in bundled_manager.list_plugins()
        if info.status == PluginStatus.ENABLED
    }
    assert EXPECTED_CORE_PLUGIN_IDS <= enabled_plugin_ids

    app_context.workspace.remove_panel(workspace_id)


def test_opened_core_panels_have_json_safe_state_when_supported(
    app_context: AppContext,
    bundled_manager: PluginManager,
) -> None:
    """
    Persistence depends on controllers exposing JSON-safe state. This test does
    not require every panel to be stateful; it only verifies that panels that do
    expose get_state() return dictionary-like state.
    """
    import json

    panels = core_panel_registrations(bundled_manager)
    assert panels

    for registration in panels:
        ensure_required_mappings_for_panel(app_context, registration)

        workspace_id = bundled_manager.open_panel(
            registration.id,
            context=app_context,
            instance_id=f"test-state:{registration.id}",
        )
        record = wait_for_workspace_panel_kind(app_context, workspace_id)

        controller = record.controller
        get_state = getattr(controller, "get_state", None)

        if callable(get_state):
            state = get_state()
            assert isinstance(state, dict), (
                f"{registration.id} get_state() should return a dict"
            )

            try:
                json.dumps(state, default=str)
            except TypeError as exc:
                raise AssertionError(
                    f"{registration.id} returned non-JSON-safe state: {state!r}"
                ) from exc

        app_context.workspace.remove_panel(workspace_id)


def test_bundled_core_actions_are_registered_with_platform_contracts(
    bundled_manager: PluginManager,
) -> None:
    """
    This intentionally does not execute every action because some actions may
    need UI-supplied params. It verifies that real bundled actions are registered
    through the platform contract rather than hidden behind panels.
    """
    actions = [
        action
        for action in bundled_manager.list_actions()
        if action.plugin_id in EXPECTED_CORE_PLUGIN_IDS
    ]

    for action in actions:
        assert action.id.startswith(f"{action.plugin_id}.")
        assert action.title
        assert callable(action.handler)
        assert action.inputs is not None
        assert action.outputs is not None
        assert isinstance(action.run_in_job, bool)
        assert isinstance(action.requires, list)
        assert isinstance(action.optional_requires, list)


def test_bundled_core_artifact_viewers_are_registered_with_platform_contracts(
    bundled_manager: PluginManager,
) -> None:
    viewers = [
        viewer
        for viewer in bundled_manager.list_artifact_viewers()
        if viewer.plugin_id in EXPECTED_CORE_PLUGIN_IDS
    ]

    for viewer in viewers:
        assert viewer.plugin_id in EXPECTED_CORE_PLUGIN_IDS
        assert viewer.artifact_type
        assert callable(viewer.viewer_factory)
        assert viewer.id is None or viewer.id.startswith(f"{viewer.plugin_id}.")
        assert isinstance(viewer.priority, int)
        assert isinstance(viewer.default, bool)


def test_no_core_panel_opening_produces_panel_open_failed_event(
    app_context: AppContext,
    bundled_manager: PluginManager,
) -> None:
    for registration in core_panel_registrations(bundled_manager):
        ensure_required_mappings_for_panel(app_context, registration)

        workspace_id = bundled_manager.open_panel(
            registration.id,
            context=app_context,
            instance_id=f"test-no-open-failed:{registration.id}",
        )
        wait_for_workspace_panel_kind(app_context, workspace_id)

    failed_events: list[dict[str, Any]] = [
        payload
        for _, topic, payload in app_context.events.recent_events(1000)
        if topic == "plugin.panel.open_failed"
    ]

    assert failed_events == []