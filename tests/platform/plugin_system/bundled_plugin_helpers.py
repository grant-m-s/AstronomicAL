from __future__ import annotations

import time

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import pytest

from astronomicAL.platform.context import AppContext
from astronomicAL.platform.plugins.manager import PluginManager
from astronomicAL.platform.plugins.specs import PanelRegistration, PluginStatus


EXPECTED_CORE_PLUGIN_IDS: set[str] = {
    "core.record_browser",
    "core.visualisation",
    "core.selection_tools",
    "core.table_tools",
    "core.annotations",
    "core.event_monitor",
    "core.plugin_manager",
}


def bundled_plugins_root() -> Path:
    """
    Return the package directory containing bundled plugin folders.

    PluginManager local-dir discovery expects a directory whose children are
    plugin folders containing plugin.py.
    """
    import astronomicAL.plugins as bundled_plugins_package

    package_paths = list(getattr(bundled_plugins_package, "__path__", []))
    if not package_paths:
        pytest.skip("astronomicAL.plugins is not a package in this checkout")

    return Path(package_paths[0])


def make_bundled_plugin_manager(context: AppContext) -> PluginManager:
    """
    Discover bundled plugins from the checkout rather than package entry points.

    The entry point groups are deliberately made unique so this test does not
    accidentally double-discover installed package entry points plus the local
    checkout plugin folders.
    """
    manager = PluginManager(
        entry_point_group="astronomical.plugins.__disabled_for_tests__",
        static_manifest_entry_point_group=(
            "astronomical.plugin_manifests.__disabled_for_tests__"
        ),
        local_plugin_dirs=[bundled_plugins_root()],
    )

    context.plugins = manager
    manager.discover()

    discovery_errors = manager.list_discovery_errors()
    assert not discovery_errors, f"Bundled plugin discovery errors: {discovery_errors}"

    return manager


def enable_core_plugins(
    manager: PluginManager,
    context: AppContext,
    *,
    expected_ids: set[str] = EXPECTED_CORE_PLUGIN_IDS,
) -> list[str]:
    discovered_ids = {info.id for info in manager.list_plugins()}
    missing = expected_ids - discovered_ids
    assert not missing, f"Expected bundled core plugins were not discovered: {missing}"

    enabled: list[str] = []

    # A stable order gives cleaner failure output if one plugin breaks.
    for plugin_id in sorted(expected_ids):
        manager.enable(plugin_id, context)
        info = manager.plugin_info(plugin_id)
        assert info.status == PluginStatus.ENABLED
        enabled.append(plugin_id)

    return enabled


def prepare_context_for_bundled_plugins(context: AppContext) -> None:
    """
    Fill in compatibility config fields that old transitional panels may still
    read while ensuring platform services remain the source of truth.
    """
    if context.config is None:
        context.config = SimpleNamespace()

    if not hasattr(context.config, "settings"):
        context.config.settings = {}

    context.config.settings.setdefault("id_col", "id")
    context.config.settings.setdefault("label_col", "label")
    context.config.settings.setdefault("extra_info_cols", ["value", "other", "name"])

    dataset_id = context.datasets.active_id()

    # Canonical generic mappings.
    context.datasets.set_mapping(dataset_id, "record_id", "id")
    context.datasets.set_mapping(dataset_id, "target_label", "label")

    # Common semantic names used or likely to be introduced by core plugins.
    context.datasets.set_mapping(dataset_id, "x", "value")
    context.datasets.set_mapping(dataset_id, "y", "other")
    context.datasets.set_mapping(dataset_id, "value", "value")
    context.datasets.set_mapping(dataset_id, "name", "name")


def _semantic_name(requirement: Any) -> str:
    """
    Extract a semantic mapping name from either the current simple string form
    or future MappingRequirement-like dataclass/object forms.
    """
    if isinstance(requirement, str):
        return requirement

    for attr in (
        "semantic_name",
        "semantic",
        "name",
        "id",
        "key",
        "mapping",
        "mapping_name",
    ):
        value = getattr(requirement, attr, None)
        if value:
            return str(value)

    if isinstance(requirement, dict):
        for key in (
            "semantic_name",
            "semantic",
            "name",
            "id",
            "key",
            "mapping",
            "mapping_name",
        ):
            value = requirement.get(key)
            if value:
                return str(value)

    return str(requirement)


def ensure_required_mappings_for_panel(
    context: AppContext,
    registration: PanelRegistration,
) -> None:
    """
    Satisfy a panel's declared mappings using the small test dataset.

    This intentionally uses semantic mappings, not hard-coded plugin internals.
    """
    dataset_id = context.datasets.active_id()

    for requirement in list(registration.required_mappings or []):
        semantic = _semantic_name(requirement)

        existing = context.datasets.get_mapping(dataset_id, semantic)
        if existing:
            continue

        lowered = semantic.lower()

        if "label" in lowered or "class" in lowered:
            column = "label"
        elif "id" in lowered or "record" in lowered or "source" in lowered:
            column = "id"
        elif lowered in {"x", "x_col", "x_column"}:
            column = "value"
        elif lowered in {"y", "y_col", "y_column"}:
            column = "other"
        elif "name" in lowered:
            column = "name"
        else:
            # Generic numeric fallback for future core-plugin requirements.
            column = "value"

        context.datasets.set_mapping(dataset_id, semantic, column)


def make_fake_workspace_manager_compatible(context: AppContext) -> None:
    """
    The earlier unit-test FakeWorkspace returned panel records from list_panels().
    PluginManager's workspace-facing path expects membership checks against panel
    ids. Patch only this test instance so open_panel(...) can see its loading tile.
    """
    workspace = context.workspace

    if hasattr(workspace, "_panels"):
        workspace.list_panels = lambda: list(workspace._panels.keys())  # type: ignore[attr-defined, method-assign]


def get_workspace_record(context: AppContext, panel_id: str) -> Any:
    workspace = context.workspace

    get_panel_record = getattr(workspace, "get_panel_record", None)
    if callable(get_panel_record):
        return get_panel_record(panel_id)

    panels = getattr(workspace, "_panels", {})
    return panels.get(panel_id)


def wait_for_workspace_panel_kind(
    context: AppContext,
    panel_id: str,
    *,
    expected_kind: str = "plugin_panel",
    timeout: float = 5.0,
) -> Any:
    """
    Wait for PluginManager.open_panel(...) to replace the loading panel with the
    final plugin panel. Fail early if the manager installs an error panel.
    """
    deadline = time.time() + timeout
    last_record = None

    while time.time() < deadline:
        record = get_workspace_record(context, panel_id)
        last_record = record

        if record is None:
            time.sleep(0.02)
            continue

        metadata = getattr(record, "metadata", {}) or {}
        kind = metadata.get("kind")

        if kind == expected_kind:
            return record

        if kind == "error_panel":
            nested = metadata.get("metadata", {}) or {}
            error = nested.get("error") or metadata.get("error") or "unknown error"
            raise AssertionError(
                f"Panel {panel_id!r} opened as error_panel: {error}"
            )

        time.sleep(0.02)

    raise AssertionError(
        f"Timed out waiting for panel {panel_id!r} to become {expected_kind!r}; "
        f"last record={last_record!r}"
    )


def subscription_count(context: AppContext) -> int:
    return len(context.events.list_subscriptions())


def core_panel_registrations(manager: PluginManager) -> list[PanelRegistration]:
    return [
        registration
        for registration in manager.list_panels()
        if registration.plugin_id in EXPECTED_CORE_PLUGIN_IDS
    ]


def plugin_panel_registrations(
    manager: PluginManager,
    plugin_ids: Iterable[str],
) -> list[PanelRegistration]:
    wanted = set(plugin_ids)
    return [
        registration
        for registration in manager.list_panels()
        if registration.plugin_id in wanted
    ]

def subscription_signature(subscription: object) -> tuple:
    """
    Stable-ish identity for comparing EventBus subscription snapshots.

    SubscriptionInfo is a dataclass in the platform. This helper is defensive so
    the test remains useful if the dataclass gains/renames diagnostic fields.
    """
    return (
        getattr(subscription, "topic", None),
        getattr(subscription, "owner_id", None),
        getattr(subscription, "owner_kind", None),
        getattr(subscription, "owner_label", None),
        getattr(subscription, "callback_repr", None),
        getattr(subscription, "callback_module", None),
        getattr(subscription, "callback_qualname", None),
    )


def subscription_counter(context: AppContext) -> Counter:
    return Counter(
        subscription_signature(subscription)
        for subscription in context.events.list_subscriptions()
    )


def leaked_subscriptions_after(
    before: Counter,
    after: Counter,
) -> list[tuple[tuple, int]]:
    leaked = after - before
    return sorted(leaked.items(), key=lambda item: repr(item[0]))


def format_subscription_leaks(leaks: list[tuple[tuple, int]]) -> str:
    if not leaks:
        return ""

    lines = ["Leaked EventBus subscriptions:"]

    for signature, count in leaks:
        (
            topic,
            owner_id,
            owner_kind,
            owner_label,
            callback_repr,
            callback_module,
            callback_qualname,
        ) = signature

        lines.append(
            "  "
            f"count={count}, "
            f"topic={topic!r}, "
            f"owner_id={owner_id!r}, "
            f"owner_kind={owner_kind!r}, "
            f"owner_label={owner_label!r}, "
            f"callback={callback_repr or callback_qualname!r}, "
            f"module={callback_module!r}"
        )

    return "\n".join(lines)