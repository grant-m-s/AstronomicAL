from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

_COMMUNITY_ORIGINS = {"user", "entry_point"}

@dataclass(frozen=True)
class Contribution:
    kind: str
    id: str
    title: str
    description: str = ""

@dataclass(frozen=True)
class OpenPanelInstance:
    instance_id: str
    title: str
    panel_id: str
    source: str = ""

@dataclass(frozen=True)
class DiscoveryIssue:
    candidate: str
    error: str

@dataclass(frozen=True)
class PluginSnapshot:
    id: str
    name: str
    version: str
    status: str
    description: str = ""
    source: str = ""
    origin: str = "unknown"
    configured_enabled: bool | None = None
    path: str | None = None
    error: str | None = None
    capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    optional_requires: tuple[str, ...] = ()
    requires_plugins: tuple[str, ...] = ()
    panels: tuple[Contribution, ...] = ()
    actions: tuple[Contribution, ...] = ()
    workflows: tuple[Contribution, ...] = ()
    services: tuple[Contribution, ...] = ()
    artifact_viewers: tuple[Contribution, ...] = ()
    open_instances: tuple[OpenPanelInstance, ...] = ()
    settings: Mapping[str, Any] = field(default_factory=dict)
    managed_install: bool = False
    installed_version: str | None = None
    installed_source: str | None = None
    installed_at: str | None = None
    updated_at: str | None = None
    archive_name: str | None = None
    package_sha256: str | None = None

    @property
    def is_managed(self) -> bool:
        return self.managed_install

    @property
    def status_label(self) -> str:
        return {
            "enabled": "Enabled",
            "disabled": "Disabled",
            "discovered": "Available",
            "error": "Needs attention",
        }.get(self.status, self.status.replace("_", " ").title() or "Unknown")

    @property
    def status_tone(self) -> str:
        return {
            "enabled": "success",
            "discovered": "info",
            "disabled": "muted",
            "error": "danger",
        }.get(self.status, "muted")

    @property
    def has_issue(self) -> bool:
        return self.status == "error" or bool(self.error)

    @property
    def is_available(self) -> bool:
        return self.status in {"disabled", "discovered"}

    @property
    def is_community(self) -> bool:
        return self.origin in _COMMUNITY_ORIGINS

    @property
    def is_bundled(self) -> bool:
        return self.origin == "bundled"

    @property
    def is_development(self) -> bool:
        return self.origin == "development"

    @property
    def is_runtime(self) -> bool:
        return self.origin == "runtime"

    @property
    def is_reloadable(self) -> bool:
        return self.is_development

    @property
    def contribution_count(self) -> int:
        return sum(
            len(items)
            for items in (
                self.panels,
                self.actions,
                self.workflows,
                self.services,
                self.artifact_viewers,
            )
        )

@dataclass(frozen=True)
class PluginManagerSnapshot:
    captured_at: datetime
    plugins: tuple[PluginSnapshot, ...]
    discovery_issues: tuple[DiscoveryIssue, ...]
    community_plugins_enabled: bool = False
    plugin_state_error: str | None = None
    installed_store_error: str | None = None

    @property
    def enabled_count(self) -> int:
        return sum(plugin.status == "enabled" for plugin in self.plugins)

    @property
    def available_count(self) -> int:
        return sum(plugin.is_available for plugin in self.plugins)

    @property
    def issue_count(self) -> int:
        return (
            sum(plugin.has_issue for plugin in self.plugins)
            + len(self.discovery_issues)
            + int(bool(self.installed_store_error))
        )

    @property
    def open_panel_count(self) -> int:
        return sum(len(plugin.open_instances) for plugin in self.plugins)

    @property
    def community_count(self) -> int:
        return sum(plugin.is_community for plugin in self.plugins)

    @property
    def configured_community_count(self) -> int:
        return sum(
            plugin.is_community and plugin.configured_enabled is True
            for plugin in self.plugins
        )

    @property
    def active_community_count(self) -> int:
        return sum(
            plugin.is_community and plugin.status == "enabled"
            for plugin in self.plugins
        )

def collect_snapshot(context: Any) -> PluginManagerSnapshot:
    manager = getattr(context, "plugins", None)
    if manager is None:
        raise RuntimeError("PluginManager is not available on AppContext.")

    state = _plugin_state(context)
    community_plugins_enabled = bool(
        getattr(state, "community_plugins_enabled", False)
    )
    plugin_state_error = _optional_text(getattr(state, "load_error", None))

    installed_store = getattr(context, "installed_plugins", None)
    installed_store_error = _optional_text(
        getattr(installed_store, "load_error", None)
    )

    infos = _safe_sequence(manager, "list_plugins")
    open_by_plugin = _open_instances_by_plugin(context, manager)

    contributions = {
        "panels": _contributions_by_plugin(
            _safe_sequence(manager, "list_panels"),
            kind="Panel",
            id_attr="id",
            title_attr="title",
        ),
        "actions": _contributions_by_plugin(
            _safe_sequence(manager, "list_actions"),
            kind="Action",
            id_attr="id",
            title_attr="title",
        ),
        "workflows": _contributions_by_plugin(
            _safe_sequence(manager, "list_workflows"),
            kind="Workflow",
            id_attr="id",
            title_attr="title",
        ),
        "services": _contributions_by_plugin(
            _safe_sequence(manager, "list_services"),
            kind="Service",
            id_attr="key",
            title_attr="key",
        ),
        "artifact_viewers": _contributions_by_plugin(
            _safe_sequence(manager, "list_artifact_viewers"),
            kind="Viewer",
            id_attr="id",
            title_attr="title",
            fallback_id_attr="artifact_type",
            fallback_title_attr="artifact_type",
        ),
    }

    plugins: list[PluginSnapshot] = []
    for info in infos:
        plugin_id = str(getattr(info, "id", "") or "")
        if not plugin_id:
            continue

        origin = _origin_text(getattr(info, "origin", "unknown"))
        configured_enabled: bool | None = None
        if origin in _COMMUNITY_ORIGINS and state is not None:
            is_enabled = getattr(state, "is_enabled", None)
            if callable(is_enabled):
                try:
                    configured_enabled = bool(is_enabled(plugin_id))
                except Exception:
                    configured_enabled = None

        installed_record = None
        if installed_store is not None:
            get_installed = getattr(installed_store, "get", None)
            if callable(get_installed):
                try:
                    installed_record = get_installed(plugin_id)
                except Exception:
                    installed_record = None

        settings: Mapping[str, Any] = {}
        get_settings = getattr(manager, "get_plugin_settings", None)
        if callable(get_settings):
            try:
                settings = dict(get_settings(plugin_id) or {})
            except Exception:
                settings = {}

        plugins.append(
            PluginSnapshot(
                id=plugin_id,
                name=str(getattr(info, "name", "") or plugin_id),
                version=str(getattr(info, "version", "") or "unknown"),
                status=_status_text(getattr(info, "status", "")),
                description=str(getattr(info, "description", "") or ""),
                source=str(getattr(info, "source", "") or ""),
                origin=origin,
                configured_enabled=configured_enabled,
                path=_optional_text(getattr(info, "path", None)),
                error=_optional_text(getattr(info, "error", None)),
                capabilities=_strings(getattr(info, "capabilities", ())),
                tags=_strings(getattr(info, "tags", ())),
                requires=_strings(getattr(info, "requires", ())),
                optional_requires=_strings(getattr(info, "optional_requires", ())),
                requires_plugins=_strings(getattr(info, "requires_plugins", ())),
                panels=tuple(contributions["panels"].get(plugin_id, ())),
                actions=tuple(contributions["actions"].get(plugin_id, ())),
                workflows=tuple(contributions["workflows"].get(plugin_id, ())),
                services=tuple(contributions["services"].get(plugin_id, ())),
                artifact_viewers=tuple(
                    contributions["artifact_viewers"].get(plugin_id, ())
                ),
                open_instances=tuple(open_by_plugin.get(plugin_id, ())),
                settings=settings,
                managed_install=installed_record is not None,
                installed_version=(
                    _optional_text(getattr(installed_record, "version", None))
                    if installed_record is not None
                    else None
                ),
                installed_source=(
                    _optional_text(getattr(installed_record, "source", None))
                    if installed_record is not None
                    else None
                ),
                installed_at=(
                    _optional_text(getattr(installed_record, "installed_at", None))
                    if installed_record is not None
                    else None
                ),
                updated_at=(
                    _optional_text(getattr(installed_record, "updated_at", None))
                    if installed_record is not None
                    else None
                ),
                archive_name=(
                    _optional_text(getattr(installed_record, "archive_name", None))
                    if installed_record is not None
                    else None
                ),
                package_sha256=(
                    _optional_text(getattr(installed_record, "sha256", None))
                    if installed_record is not None
                    else None
                ),
            )
        )

    plugins.sort(key=_plugin_sort_key)

    issues: list[DiscoveryIssue] = []
    list_errors = getattr(manager, "list_discovery_errors", None)
    if callable(list_errors):
        try:
            errors = dict(list_errors() or {})
        except Exception:
            errors = {}
        issues = [
            DiscoveryIssue(candidate=str(candidate), error=str(error))
            for candidate, error in sorted(
                errors.items(), key=lambda item: str(item[0]).lower()
            )
        ]

    return PluginManagerSnapshot(
        captured_at=datetime.now(timezone.utc),
        plugins=tuple(plugins),
        discovery_issues=tuple(issues),
        community_plugins_enabled=community_plugins_enabled,
        plugin_state_error=plugin_state_error,
        installed_store_error=installed_store_error,
    )

def filter_plugins(
    plugins: Iterable[PluginSnapshot],
    *,
    query: str = "",
    status_filter: str = "all",
) -> list[PluginSnapshot]:
    query = str(query or "").strip().lower()
    status_filter = str(status_filter or "all").strip().lower()

    filtered: list[PluginSnapshot] = []
    for plugin in plugins:
        if status_filter == "enabled" and plugin.status != "enabled":
            continue
        if status_filter == "available" and not plugin.is_available:
            continue
        if status_filter == "issues" and not plugin.has_issue:
            continue

        if query:
            values: list[str] = [
                plugin.id,
                plugin.name,
                plugin.description,
                plugin.source,
                plugin.origin,
                source_label(plugin.origin, plugin.source),
                plugin.error or "",
                plugin.installed_source or "",
                plugin.archive_name or "",
                *plugin.capabilities,
                *plugin.tags,
            ]
            for items in (
                plugin.panels,
                plugin.actions,
                plugin.workflows,
                plugin.services,
                plugin.artifact_viewers,
            ):
                for item in items:
                    values.extend((item.id, item.title, item.description))

            if query not in " ".join(values).lower():
                continue

        filtered.append(plugin)

    return filtered

def source_label(origin: str, source: str = "") -> str:
    origin_value = _origin_text(origin)
    labels = {
        "bundled": "Bundled plugin",
        "user": "Community plugin",
        "development": "Development plugin",
        "entry_point": "Python package",
        "runtime": "Runtime registration",
    }
    if origin_value in labels:
        return labels[origin_value]

    source_value = str(source or "").strip().lower()
    if source_value == "runtime_registration":
        return "Runtime registration"
    if "entry_point" in source_value:
        return "Installed package"
    if "static_manifest" in source_value:
        return "Static manifest"
    if "local" in source_value:
        return "Local plugin"
    if source_value:
        return source_value.replace("_", " ").title()
    return "Unknown source"

def provides_summary(plugin: PluginSnapshot) -> str:
    parts: list[str] = []
    for count, singular in (
        (len(plugin.panels), "panel"),
        (len(plugin.actions), "action"),
        (len(plugin.workflows), "workflow"),
        (len(plugin.services), "service"),
        (len(plugin.artifact_viewers), "viewer"),
    ):
        if count:
            parts.append(f"{count} {singular}{'' if count == 1 else 's'}")
    return " · ".join(parts) if parts else "No registered features"

def _plugin_state(context: Any) -> Any:
    state = getattr(context, "plugin_state", None)
    if state is not None:
        return state

    activation = getattr(context, "plugin_activation", None)
    return getattr(activation, "state", None)

def _safe_sequence(obj: Any, method_name: str) -> list[Any]:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return []
    try:
        return list(method() or [])
    except Exception:
        return []

def _contributions_by_plugin(
    registrations: Sequence[Any],
    *,
    kind: str,
    id_attr: str,
    title_attr: str,
    fallback_id_attr: str | None = None,
    fallback_title_attr: str | None = None,
) -> dict[str, list[Contribution]]:
    grouped: dict[str, list[Contribution]] = {}
    for registration in registrations:
        plugin_id = str(getattr(registration, "plugin_id", "") or "")
        if not plugin_id:
            continue

        contribution_id = getattr(registration, id_attr, None)
        if not contribution_id and fallback_id_attr:
            contribution_id = getattr(registration, fallback_id_attr, None)

        title = getattr(registration, title_attr, None)
        if not title and fallback_title_attr:
            title = getattr(registration, fallback_title_attr, None)

        contribution_id = str(contribution_id or title or kind)
        title = str(title or contribution_id)
        description = str(getattr(registration, "description", "") or "")

        grouped.setdefault(plugin_id, []).append(
            Contribution(
                kind=kind,
                id=contribution_id,
                title=title,
                description=description,
            )
        )

    for items in grouped.values():
        items.sort(key=lambda item: (item.title.lower(), item.id.lower()))
    return grouped

def _open_instances_by_plugin(
    context: Any,
    manager: Any,
) -> dict[str, list[OpenPanelInstance]]:
    grouped: dict[str, list[OpenPanelInstance]] = {}

    list_instances = getattr(manager, "list_panel_instances", None)
    if callable(list_instances):
        try:
            raw_instances = list(list_instances() or [])
        except TypeError:
            raw_instances = list(list_instances(plugin_id=None) or [])
        except Exception:
            raw_instances = []

        for item in raw_instances:
            if not isinstance(item, Mapping):
                continue
            plugin_id = str(item.get("plugin_id", "") or "")
            if not plugin_id:
                continue
            grouped.setdefault(plugin_id, []).append(
                OpenPanelInstance(
                    instance_id=str(item.get("instance_id", "") or ""),
                    title=str(
                        item.get("title", "") or item.get("panel_id", "") or "Panel"
                    ),
                    panel_id=str(item.get("panel_id", "") or ""),
                    source=str(item.get("source", "") or ""),
                )
            )

        if grouped:
            return _sort_open_instances(grouped)

    workspace = getattr(context, "workspace", None)
    list_panels = getattr(workspace, "list_panels", None)
    if not callable(list_panels):
        return grouped

    try:
        records = list_panels() or {}
    except Exception:
        return grouped

    if isinstance(records, Mapping):
        iterable = records.items()
    else:
        iterable = enumerate(records)

    for instance_key, record in iterable:
        plugin_id = str(getattr(record, "plugin_id", "") or "")
        if not plugin_id:
            continue
        grouped.setdefault(plugin_id, []).append(
            OpenPanelInstance(
                instance_id=str(getattr(record, "panel_id", "") or instance_key),
                title=str(getattr(record, "title", "") or "Panel"),
                panel_id=str(
                    getattr(record, "registration_id", "")
                    or getattr(record, "panel_id", "")
                    or instance_key
                ),
                source=str(getattr(record, "kind", "") or ""),
            )
        )

    return _sort_open_instances(grouped)

def _sort_open_instances(
    grouped: dict[str, list[OpenPanelInstance]],
) -> dict[str, list[OpenPanelInstance]]:
    for items in grouped.values():
        items.sort(key=lambda item: (item.title.lower(), item.instance_id.lower()))
    return grouped

def _plugin_sort_key(plugin: PluginSnapshot) -> tuple[int, str, str]:
    status_rank = {
        "error": 0,
        "enabled": 1,
        "discovered": 2,
        "disabled": 3,
    }.get(plugin.status, 4)
    return status_rank, plugin.name.lower(), plugin.id.lower()

def _status_text(status: Any) -> str:
    value = getattr(status, "value", status)
    return str(value or "").strip().lower()

def _origin_text(origin: Any) -> str:
    value = getattr(origin, "value", origin)
    return str(value or "unknown").strip().lower() or "unknown"

def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def _strings(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    try:
        return tuple(str(value) for value in values if str(value))
    except Exception:
        return (str(values),)