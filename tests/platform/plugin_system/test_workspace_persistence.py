from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from astronomicAL.platform.context import AppContext
from astronomicAL.platform.persistence import SCHEMA_NAME, SCHEMA_VERSION, WorkspacePersistence
from astronomicAL.platform.plugins.manager import PluginManager
from astronomicAL.platform.plugins.specs import PluginStatus

from .bundled_plugin_helpers import (
    EXPECTED_CORE_PLUGIN_IDS,
    enable_core_plugins,
    ensure_required_mappings_for_panel,
    get_workspace_record,
    make_bundled_plugin_manager,
    make_fake_workspace_manager_compatible,
    prepare_context_for_bundled_plugins,
    wait_for_workspace_panel_kind,
)
from .conftest import recent_events_for


def _prepare_context_and_manager(context: AppContext) -> PluginManager:
    prepare_context_for_bundled_plugins(context)
    make_fake_workspace_manager_compatible(context)

    manager = make_bundled_plugin_manager(context)
    enable_core_plugins(manager, context)

    context.persistence = WorkspacePersistence(context)
    return manager


def _open_panel(
    context: AppContext,
    manager: PluginManager,
    registration_id: str,
    *,
    instance_id: str,
    open_kwargs: dict[str, Any] | None = None,
) -> str:
    registration = manager.get_panel(registration_id)
    ensure_required_mappings_for_panel(context, registration)

    workspace_id = manager.open_panel(
        registration_id,
        context=context,
        instance_id=instance_id,
        open_kwargs=open_kwargs or {},
    )

    wait_for_workspace_panel_kind(context, workspace_id)
    return workspace_id


def _panel_snapshot_by_instance(
    snapshot: dict[str, Any],
    instance_id: str,
) -> dict[str, Any]:
    for panel_snapshot in snapshot["workspace"]["panels"]:
        if panel_snapshot["instance_id"] == instance_id:
            return panel_snapshot

    raise AssertionError(f"Snapshot does not contain panel {instance_id!r}")


def _event_topics(context: AppContext) -> list[str]:
    return [topic for _, topic, _ in context.events.recent_events(500)]


def test_workspace_snapshot_contains_plugins_datasets_selection_and_panels(
    app_context: AppContext,
) -> None:
    manager = _prepare_context_and_manager(app_context)

    app_context.datasets.set_mapping("main", "x", "value")
    app_context.datasets.set_mapping("main", "y", "other")

    _open_panel(
        app_context,
        manager,
        "core.plugin_manager.panel",
        instance_id="persist-plugin-manager",
    )
    _open_panel(
        app_context,
        manager,
        "core.event_monitor.panel",
        instance_id="persist-event-monitor",
    )

    selection_set = app_context.selection.set_selection_set(
        "main",
        ["a", "c"],
        origin="persistence-test",
        metadata={"kind": "round-trip"},
        create_artifact=False,
    )

    app_context.selection.set_focus(
        "main",
        "b",
        origin="persistence-test",
        metadata={"reason": "snapshot"},
    )

    snapshot = app_context.persistence.snapshot()

    assert snapshot["schema"] == SCHEMA_NAME
    assert snapshot["schema_version"] == SCHEMA_VERSION
    assert "created_at" in snapshot

    assert set(snapshot["plugins"]["enabled"]) == EXPECTED_CORE_PLUGIN_IDS

    required_plugin_ids = {
        item["id"] for item in snapshot["plugins"]["required"]
    }
    assert required_plugin_ids == {
        "core.plugin_manager",
        "core.event_monitor",
    }

    dataset_snapshot = snapshot["datasets"]
    assert dataset_snapshot["active_id"] == "main"

    main_dataset = next(
        item for item in dataset_snapshot["items"] if item["id"] == "main"
    )
    assert main_dataset["mappings"]["record_id"] == "id"
    assert main_dataset["mappings"]["target_label"] == "label"
    assert main_dataset["mappings"]["x"] == "value"
    assert main_dataset["mappings"]["y"] == "other"
    assert main_dataset["columns"] == ["id", "value", "other", "name", "label"]

    focus_snapshot = snapshot["selection"]["focus"]
    assert focus_snapshot["dataset_id"] == "main"
    assert focus_snapshot["row_id"] == "b"
    assert focus_snapshot["origin"] == "persistence-test"
    assert focus_snapshot["metadata"] == {"reason": "snapshot"}

    set_snapshot = snapshot["selection"]["selection_set"]
    assert set_snapshot["dataset_id"] == "main"
    assert set_snapshot["row_ids"] == ["a", "c"]
    assert set_snapshot["selection_set_id"] == selection_set.selection_set_id
    assert set_snapshot["metadata"] == {"kind": "round-trip"}

    panel_ids = {
        panel["instance_id"] for panel in snapshot["workspace"]["panels"]
    }
    assert {
        "persist-plugin-manager",
        "persist-event-monitor",
    } <= panel_ids

    plugin_manager_panel = _panel_snapshot_by_instance(
        snapshot,
        "persist-plugin-manager",
    )
    assert plugin_manager_panel["kind"] == "plugin_panel"
    assert plugin_manager_panel["plugin_id"] == "core.plugin_manager"
    assert plugin_manager_panel["registration_id"] == "core.plugin_manager.panel"
    assert isinstance(plugin_manager_panel["state"], dict)

    event_monitor_panel = _panel_snapshot_by_instance(
        snapshot,
        "persist-event-monitor",
    )
    assert event_monitor_panel["kind"] == "plugin_panel"
    assert event_monitor_panel["plugin_id"] == "core.event_monitor"
    assert event_monitor_panel["registration_id"] == "core.event_monitor.panel"
    assert isinstance(event_monitor_panel["state"], dict)


def test_workspace_save_and_load_round_trip_json_file(
    app_context: AppContext,
    tmp_path: Path,
) -> None:
    manager = _prepare_context_and_manager(app_context)

    _open_panel(
        app_context,
        manager,
        "core.plugin_manager.panel",
        instance_id="save-load-plugin-manager",
    )

    app_context.selection.set_focus("main", "c", origin="save-test")

    path = tmp_path / "workspace" / "workspace.json"

    saved_snapshot = app_context.persistence.save(path)

    assert path.exists()
    assert "workspace.saved" in _event_topics(app_context)

    loaded_snapshot = app_context.persistence.load(path)

    assert loaded_snapshot["schema"] == saved_snapshot["schema"]
    assert loaded_snapshot["schema_version"] == saved_snapshot["schema_version"]
    assert loaded_snapshot["plugins"] == saved_snapshot["plugins"]
    assert loaded_snapshot["datasets"] == saved_snapshot["datasets"]
    assert loaded_snapshot["selection"] == saved_snapshot["selection"]
    assert loaded_snapshot["workspace"] == saved_snapshot["workspace"]

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    assert raw == loaded_snapshot


def test_restore_round_trip_reopens_panels_and_restores_selection(
    make_app_context: Callable[[str], AppContext],
) -> None:
    source_context = make_app_context("restore-source")
    source_manager = _prepare_context_and_manager(source_context)

    _open_panel(
        source_context,
        source_manager,
        "core.record_browser.panel",
        instance_id="restore-record-browser",
    )
    _open_panel(
        source_context,
        source_manager,
        "core.plugin_manager.panel",
        instance_id="restore-plugin-manager",
    )

    source_context.selection.set_focus(
        "main",
        "b",
        origin="restore-source",
        metadata={"from": "source"},
    )
    source_context.selection.set_selection_set(
        "main",
        ["a", "b"],
        origin="restore-source",
        metadata={"set": "source"},
        create_artifact=False,
    )

    snapshot = source_context.persistence.snapshot()

    restored_context = make_app_context("restore-target")
    restored_manager = _prepare_context_and_manager(restored_context)

    # Start from a deliberately different state so the restore has something
    # visible to change.
    restored_context.workspace.clear()
    restored_context.selection.clear_focus(origin="before-restore")
    restored_context.selection.clear_selection_set(origin="before-restore")

    issues = restored_context.persistence.restore(snapshot)

    assert issues == []

    record_browser = wait_for_workspace_panel_kind(
        restored_context,
        "restore-record-browser",
        expected_kind="plugin_panel",
    )

    plugin_manager = wait_for_workspace_panel_kind(
        restored_context,
        "restore-plugin-manager",
        expected_kind="plugin_panel",
    )

    assert record_browser.plugin_id == "core.record_browser"
    assert record_browser.registration_id == "core.record_browser.panel"

    assert plugin_manager.plugin_id == "core.plugin_manager"
    assert plugin_manager.registration_id == "core.plugin_manager.panel"

    focus = restored_context.selection.get_focus()
    assert focus.dataset_id == "main"
    assert focus.row_id == "b"
    assert focus.origin == "restore-source"
    assert focus.metadata == {"from": "source"}

    active_set = restored_context.selection.get_active_set()
    assert active_set is not None
    assert active_set.dataset_id == "main"
    assert active_set.row_ids == ["a", "b"]
    assert active_set.metadata == {"set": "source"}

    assert "workspace.restore.completed" in _event_topics(restored_context)

    enabled_ids = {
        info.id
        for info in restored_manager.list_plugins()
        if info.status == PluginStatus.ENABLED
    }
    assert EXPECTED_CORE_PLUGIN_IDS <= enabled_ids


def test_restore_applies_dataset_mappings_before_mapping_gated_panels_open(
    make_app_context: Callable[[str], AppContext],
) -> None:
    source_context = make_app_context("mapping-source")
    source_manager = _prepare_context_and_manager(source_context)

    source_context.datasets.set_mapping("main", "record_id", "id")
    source_context.datasets.set_mapping("main", "target_label", "label")
    source_context.datasets.set_mapping("main", "x", "value")
    source_context.datasets.set_mapping("main", "y", "other")

    _open_panel(
        source_context,
        source_manager,
        "core.record_browser.panel",
        instance_id="mapping-record-browser",
    )

    snapshot = source_context.persistence.snapshot()

    restored_context = make_app_context("mapping-target")
    restored_manager = _prepare_context_and_manager(restored_context)

    # Remove mappings after the manager is ready to prove restore applies the
    # snapshot's dataset metadata before panel restore.
    restored_context.datasets.set_mapping("main", "record_id", "")
    restored_context.datasets.set_mapping("main", "target_label", "")
    restored_context.datasets.set_mapping("main", "x", "")
    restored_context.datasets.set_mapping("main", "y", "")

    issues = restored_context.persistence.restore(snapshot)

    assert issues == []

    assert restored_context.datasets.get_mapping("main", "record_id") == "id"
    assert restored_context.datasets.get_mapping("main", "target_label") == "label"
    assert restored_context.datasets.get_mapping("main", "x") == "value"
    assert restored_context.datasets.get_mapping("main", "y") == "other"

    record = wait_for_workspace_panel_kind(
        restored_context,
        "mapping-record-browser",
        expected_kind="plugin_panel",
    )

    assert record.registration_id == "core.record_browser.panel"


def test_restore_missing_plugin_creates_missing_panel_and_reports_issue(
    make_app_context: Callable[[str], AppContext],
) -> None:
    source_context = make_app_context("missing-plugin-source")
    source_manager = _prepare_context_and_manager(source_context)

    _open_panel(
        source_context,
        source_manager,
        "core.plugin_manager.panel",
        instance_id="missing-plugin-panel",
    )

    snapshot = source_context.persistence.snapshot()

    broken_snapshot = copy.deepcopy(snapshot)
    broken_snapshot["plugins"]["enabled"] = ["missing.fake_plugin"]
    broken_snapshot["plugins"]["required"] = [
        {
            "id": "missing.fake_plugin",
            "version": "0.0.0",
            "panels": ["missing.fake_plugin.panel"],
        }
    ]

    for panel in broken_snapshot["workspace"]["panels"]:
        if panel["instance_id"] == "missing-plugin-panel":
            panel["plugin_id"] = "missing.fake_plugin"
            panel["registration_id"] = "missing.fake_plugin.panel"
            panel["title"] = "Missing Fake Plugin Panel"

    restored_context = make_app_context("missing-plugin-target")
    _prepare_context_and_manager(restored_context)

    issues = restored_context.persistence.restore(broken_snapshot, strict=False)

    issue_types = {issue["type"] for issue in issues}
    assert "plugin_enable_failed" in issue_types
    assert "panel_restore_failed" in issue_types

    record = get_workspace_record(restored_context, "missing-plugin-panel")
    assert record is not None
    assert record.kind == "missing_panel"
    assert record.plugin_id == "missing.fake_plugin"
    assert record.registration_id == "missing.fake_plugin.panel"
    assert "restore_reason" in record.metadata

    restore_events = recent_events_for(
        restored_context,
        "workspace.restore.completed",
    )
    assert restore_events
    assert restore_events[-1]["issues"] == issues


def test_restore_strict_mode_raises_when_required_plugin_is_missing(
    make_app_context: Callable[[str], AppContext],
) -> None:
    source_context = make_app_context("strict-source")
    source_manager = _prepare_context_and_manager(source_context)

    _open_panel(
        source_context,
        source_manager,
        "core.plugin_manager.panel",
        instance_id="strict-plugin-panel",
    )

    snapshot = source_context.persistence.snapshot()

    broken_snapshot = copy.deepcopy(snapshot)
    broken_snapshot["plugins"]["enabled"] = ["missing.strict_plugin"]
    broken_snapshot["plugins"]["required"] = [
        {
            "id": "missing.strict_plugin",
            "version": "0.0.0",
            "panels": ["missing.strict_plugin.panel"],
        }
    ]

    for panel in broken_snapshot["workspace"]["panels"]:
        if panel["instance_id"] == "strict-plugin-panel":
            panel["plugin_id"] = "missing.strict_plugin"
            panel["registration_id"] = "missing.strict_plugin.panel"

    restored_context = make_app_context("strict-target")
    _prepare_context_and_manager(restored_context)

    with pytest.raises(RuntimeError, match="Workspace restore completed with issues"):
        restored_context.persistence.restore(broken_snapshot, strict=True)


def test_restore_unsupported_panel_kind_creates_missing_panel(
    make_app_context: Callable[[str], AppContext],
) -> None:
    source_context = make_app_context("unsupported-kind-source")
    source_manager = _prepare_context_and_manager(source_context)

    _open_panel(
        source_context,
        source_manager,
        "core.plugin_manager.panel",
        instance_id="unsupported-kind-panel",
    )

    snapshot = source_context.persistence.snapshot()
    broken_snapshot = copy.deepcopy(snapshot)

    for panel in broken_snapshot["workspace"]["panels"]:
        if panel["instance_id"] == "unsupported-kind-panel":
            panel["kind"] = "legacy_non_plugin_panel"

    restored_context = make_app_context("unsupported-kind-target")
    _prepare_context_and_manager(restored_context)

    issues = restored_context.persistence.restore(broken_snapshot)

    assert any(issue["type"] == "unsupported_panel_kind" for issue in issues)

    record = get_workspace_record(restored_context, "unsupported-kind-panel")
    assert record is not None
    assert record.kind == "missing_panel"
    assert record.registration_id == "core.plugin_manager.panel"
    assert "Unsupported panel kind" in record.metadata["restore_reason"]


def test_non_persistent_workspace_panel_is_not_in_snapshot(
    app_context: AppContext,
) -> None:
    manager = _prepare_context_and_manager(app_context)

    _open_panel(
        app_context,
        manager,
        "core.plugin_manager.panel",
        instance_id="persistent-panel",
    )

    app_context.workspace.add_panel(
        "temporary-panel",
        view={"view": "temporary"},
        title="Temporary Panel",
        controller={"controller": "temporary"},
        kind="plugin_panel",
        plugin_id="test.temporary",
        registration_id="test.temporary.panel",
        persistent=False,
    )

    snapshot = app_context.persistence.snapshot()

    panel_ids = {
        panel["instance_id"] for panel in snapshot["workspace"]["panels"]
    }

    assert "persistent-panel" in panel_ids
    assert "temporary-panel" not in panel_ids


def test_workspace_snapshot_is_json_serialisable(
    app_context: AppContext,
) -> None:
    manager = _prepare_context_and_manager(app_context)

    _open_panel(
        app_context,
        manager,
        "core.plugin_manager.panel",
        instance_id="json-safe-plugin-manager",
    )

    app_context.selection.set_focus(
        "main",
        "a",
        origin="json-safe",
        metadata={"nested": {"value": 1}},
    )

    snapshot = app_context.persistence.snapshot()

    encoded = json.dumps(snapshot, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded["schema"] == SCHEMA_NAME
    assert decoded["schema_version"] == SCHEMA_VERSION
    assert decoded["workspace"]["panels"][0]["instance_id"] == (
        "json-safe-plugin-manager"
    )


def test_restore_invalid_schema_is_rejected(
    app_context: AppContext,
) -> None:
    _prepare_context_and_manager(app_context)

    bad_snapshot = {
        "schema": "not.astronomical.workspace",
        "schema_version": SCHEMA_VERSION,
    }

    with pytest.raises(ValueError, match="Unsupported workspace schema"):
        app_context.persistence.restore(bad_snapshot)


def test_restore_future_schema_version_is_rejected(
    app_context: AppContext,
) -> None:
    _prepare_context_and_manager(app_context)

    bad_snapshot = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION + 1,
    }

    with pytest.raises(ValueError, match="newer than supported"):
        app_context.persistence.restore(bad_snapshot)