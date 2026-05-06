from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import panel as pn

from astronomicAL.platform.plugins import PluginManifest


manifest = PluginManifest(
    id="core.plugin_manager",
    name="Plugin Manager",
    version="0.2.0",
    description=(
        "Inspect and manage discovered AstronomicAL plugins, including panels, "
        "actions, services, workflows, artifact viewers, open panel instances, "
        "and discovery errors."
    ),
    capabilities=["panel", "diagnostics", "plugins", "management"],
    tags=["core", "debug", "plugins", "management"],
)


def register(api) -> None:
    api.register_panel(
        id="panel",
        title="Plugin Manager",
        factory=create_plugin_manager_panel,
        description="Inspect discovered plugins and their registered contributions.",
        category="Diagnostics",
        icon="plug",
        tags=["plugins", "debug", "management"],
        default_layout={"x": 0, "y": 0, "w": 5, "h": 7},
    )


def create_plugin_manager_panel(context, **kwargs):
    controller = PluginManagerPanel(context=context)
    return controller.view, controller


class PluginManagerPanel:
    """Narrow-layout plugin diagnostics and management panel."""

    SELF_PLUGIN_ID = manifest.id

    def __init__(self, context):
        self.context = context
        self.manager = getattr(context, "plugins", None)
        self._disposed = False
        self._subscriptions: list[Any] = []
        self._watchers: list[tuple[Any, Any]] = []

        self.title = pn.pane.HTML(
            "<h2 style='margin: 0; padding: 0; line-height: 1.2;'>Plugin Manager</h2>",
            height=32,
            min_height=32,
            max_height=32,
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        self.status = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        self.warning = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        self.selected_plugin = pn.widgets.Select(
            name="Selected plugin",
            options=[],
            value=None,
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        self.force_disable = pn.widgets.Checkbox(
            name="Allow disable/reload with open panel instances",
            value=False,
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="default",
            height=32,
            sizing_mode="stretch_width",
        )

        self.discover_button = pn.widgets.Button(
            name="Discover",
            button_type="default",
            height=32,
            sizing_mode="stretch_width",
        )

        self.broadcast_button = pn.widgets.Button(
            name="Refresh Menus",
            button_type="default",
            height=32,
            sizing_mode="stretch_width",
        )

        self.enable_button = pn.widgets.Button(
            name="Enable",
            button_type="primary",
            height=32,
            sizing_mode="stretch_width",
        )

        self.disable_button = pn.widgets.Button(
            name="Disable",
            button_type="warning",
            height=32,
            sizing_mode="stretch_width",
        )

        self.reload_button = pn.widgets.Button(
            name="Reload",
            button_type="success",
            height=32,
            sizing_mode="stretch_width",
        )

        self.plugins_table = self._make_table(page_size=8)
        self.instances_table = self._make_table(page_size=8)
        self.panels_table = self._make_table(page_size=8)
        self.actions_table = self._make_table(page_size=8)
        self.workflows_table = self._make_table(page_size=8)
        self.services_table = self._make_table(page_size=8)
        self.artifact_viewers_table = self._make_table(page_size=8)
        self.discovery_errors_table = self._make_table(page_size=8)

        self.details_pane = pn.pane.Markdown(
            "Select a plugin to inspect details.",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.refresh_button.on_click(self._refresh_clicked)
        self.discover_button.on_click(self._discover_clicked)
        self.broadcast_button.on_click(self._broadcast_clicked)
        self.enable_button.on_click(self._enable_clicked)
        self.disable_button.on_click(self._disable_clicked)
        self.reload_button.on_click(self._reload_clicked)

        self._watch(self.selected_plugin, self._selected_plugin_changed, "value")
        self._watch(self.force_disable, self._selected_plugin_changed, "value")

        self._subscribe_to_plugin_events()

        self.view = self._build_view()
        self.refresh()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _make_table(self, *, page_size: int = 8):
        return pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=False,
            disabled=True,
            pagination="local",
            page_size=page_size,
            sizing_mode="stretch_width",
            height=260,
            min_height=220,
            margin=(0, 0, 0, 0),
            configuration={
                "layout": "fitDataStretch",
                "responsiveLayout": "collapse",
            },
        )

    def _build_view(self):
        action_grid = pn.GridBox(
            self.refresh_button,
            self.discover_button,
            self.broadcast_button,
            self.enable_button,
            self.disable_button,
            self.reload_button,
            ncols=2,
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        contribution_tabs = pn.Tabs(
            ("Panels", self._section(self.panels_table)),
            ("Actions", self._section(self.actions_table)),
            ("Workflows", self._section(self.workflows_table)),
            ("Services", self._section(self.services_table)),
            ("Viewers", self._section(self.artifact_viewers_table)),
            dynamic=True,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        tabs = pn.Tabs(
            ("Plugins", self._section(self.plugins_table)),
            ("Open", self._section(self.instances_table)),
            ("Contrib", contribution_tabs),
            ("Errors", self._section(self.discovery_errors_table)),
            ("Details", pn.Column(self.details_pane, sizing_mode="stretch_width", scroll=True)),
            dynamic=True,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        return pn.Column(
            self.title,
            self.selected_plugin,
            self.force_disable,
            action_grid,
            self.status,
            self.warning,
            tabs,
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
            styles={
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "padding": "0 8px 8px 8px",
            },
        )

    @staticmethod
    def _section(obj):
        return pn.Column(
            obj,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            styles={
                "overflow-x": "auto",
                "overflow-y": "hidden",
            },
        )

    # ------------------------------------------------------------------
    # Event/watch lifecycle
    # ------------------------------------------------------------------

    def _watch(self, widget, callback, attr: str) -> None:
        try:
            watcher = widget.param.watch(callback, attr)
            self._watchers.append((widget, watcher))
        except Exception:
            pass

    def _subscribe_to_plugin_events(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        for topic in (
            "plugin.enabled",
            "plugin.disabled",
            "plugin.reloaded",
            "plugin.registry.changed",
        ):
            try:
                sub = events.subscribe(
                    topic,
                    self._on_plugin_event,
                    owner_id="core.plugin_manager.panel",
                    owner_label="Plugin Manager",
                    owner_kind="plugin-panel",
                )
            except TypeError:
                sub = events.subscribe(topic, self._on_plugin_event)

            self._subscriptions.append(sub)

    def _on_plugin_event(self, topic: str, payload: Any) -> None:
        if self._disposed:
            return
        self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        def _refresh():
            if not self._disposed:
                self.refresh()

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(_refresh)
            else:
                _refresh()
        except Exception:
            _refresh()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _refresh_clicked(self, _event=None) -> None:
        self.refresh()

    def _discover_clicked(self, _event=None) -> None:
        if self.manager is None:
            self.status.object = "PluginManager is not available on context."
            return

        try:
            self.manager.discover()
            self._publish_registry_changed(operation="discovered")
            self.status.object = "Discovery completed."
        except Exception as exc:
            self.status.object = f"Discovery failed: `{exc}`"
            traceback.print_exc()

        self.refresh()

    def _broadcast_clicked(self, _event=None) -> None:
        self._publish_registry_changed(operation="manual_refresh")
        self.status.object = "Published `plugin.registry.changed` for menu refresh."
        self.refresh()

    def _enable_clicked(self, _event=None) -> None:
        plugin_id = self.selected_plugin.value
        if not plugin_id:
            self.status.object = "Select a plugin first."
            return

        if self.manager is None:
            self.status.object = "PluginManager is not available on context."
            return

        try:
            self.manager.enable(plugin_id, self.context)
            self._publish_registry_changed(plugin_id=plugin_id, operation="enabled")
            self.status.object = f"Enabled `{plugin_id}`."
        except Exception as exc:
            self.status.object = f"Enable failed for `{plugin_id}`: `{exc}`"
            traceback.print_exc()

        self.refresh()

    def _disable_clicked(self, _event=None) -> None:
        plugin_id = self.selected_plugin.value
        if not plugin_id:
            self.status.object = "Select a plugin first."
            return

        if not self._can_disable_or_reload(plugin_id, action="disable"):
            self.refresh()
            return

        if self.manager is None:
            self.status.object = "PluginManager is not available on context."
            return

        try:
            self.manager.disable(plugin_id, context=self.context)
            self._publish_registry_changed(plugin_id=plugin_id, operation="disabled")
            self.status.object = f"Disabled `{plugin_id}`."
        except Exception as exc:
            self.status.object = f"Disable failed for `{plugin_id}`: `{exc}`"
            traceback.print_exc()

        self.refresh()

    def _reload_clicked(self, _event=None) -> None:
        plugin_id = self.selected_plugin.value
        if not plugin_id:
            self.status.object = "Select a plugin first."
            return

        if not self._can_disable_or_reload(plugin_id, action="reload"):
            self.refresh()
            return

        if self.manager is None:
            self.status.object = "PluginManager is not available on context."
            return

        try:
            self.manager.reload(plugin_id, self.context)
            self._publish_registry_changed(plugin_id=plugin_id, operation="reloaded")
            self.status.object = f"Reloaded `{plugin_id}`."
        except Exception as exc:
            self.status.object = f"Reload failed for `{plugin_id}`: `{exc}`"
            traceback.print_exc()

        self.refresh()

    def _can_disable_or_reload(self, plugin_id: str, *, action: str) -> bool:
        if plugin_id == self.SELF_PLUGIN_ID:
            self.status.object = (
                f"Refusing to {action} Plugin Manager from inside its own panel. "
                "Use a different management surface or restart the server."
            )
            return False

        open_count = self._open_instance_count(plugin_id)

        if open_count and not bool(self.force_disable.value):
            self.status.object = (
                f"Refusing to {action} `{plugin_id}` while it has "
                f"{open_count} open panel instance(s). Close them first or enable "
                "the force checkbox."
            )
            return False

        return True

    def _selected_plugin_changed(self, _event=None) -> None:
        self._render_details()
        self._update_button_state()

    def _publish_registry_changed(
        self,
        *,
        plugin_id: Optional[str] = None,
        operation: str,
    ) -> None:
        if self.manager is not None:
            method = getattr(self.manager, "publish_registry_changed", None)
            if callable(method):
                try:
                    method(
                        self.context,
                        plugin_id=plugin_id,
                        operation=operation,
                    )
                    return
                except Exception:
                    traceback.print_exc()

        events = getattr(self.context, "events", None)
        if events is not None:
            events.publish(
                "plugin.registry.changed",
                {
                    "plugin_id": plugin_id,
                    "operation": operation,
                },
            )

    # ------------------------------------------------------------------
    # Refresh/render
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        if self._disposed:
            return

        if self.manager is None:
            self.status.object = "PluginManager is not available on context."
            self._clear_all_tables()
            return

        plugin_infos = self._safe_list_plugins()

        self._refresh_selected_plugin_options(plugin_infos)

        self.plugins_table.value = self._plugins_dataframe(plugin_infos)
        self.instances_table.value = self._instances_dataframe()
        self.panels_table.value = self._panels_dataframe()
        self.actions_table.value = self._actions_dataframe()
        self.workflows_table.value = self._workflows_dataframe()
        self.services_table.value = self._services_dataframe()
        self.artifact_viewers_table.value = self._artifact_viewers_dataframe()
        self.discovery_errors_table.value = self._discovery_errors_dataframe()

        self._render_details()
        self._update_button_state()
        self._render_summary(plugin_infos)

    def _render_summary(self, plugin_infos: Iterable[Any]) -> None:
        plugin_infos = list(plugin_infos)

        enabled_count = len(
            [
                info
                for info in plugin_infos
                if self._status_text(getattr(info, "status", "")) == "enabled"
            ]
        )
        error_count = len(
            [
                info
                for info in plugin_infos
                if self._status_text(getattr(info, "status", "")) == "error"
            ]
        )
        open_count = len(self._safe_list_panel_instances())
        discovery_errors = len(self._safe_discovery_errors())

        self.status.object = (
            f"Plugins: **{len(plugin_infos)}** · "
            f"Enabled: **{enabled_count}** · "
            f"Errors: **{error_count}** · "
            f"Open panels: **{open_count}** · "
            f"Discovery errors: **{discovery_errors}**"
        )

    def _clear_all_tables(self) -> None:
        empty = pd.DataFrame()
        self.plugins_table.value = empty
        self.instances_table.value = empty
        self.panels_table.value = empty
        self.actions_table.value = empty
        self.workflows_table.value = empty
        self.services_table.value = empty
        self.artifact_viewers_table.value = empty
        self.discovery_errors_table.value = pd.DataFrame(columns=["candidate", "error"])
        self.details_pane.object = "PluginManager is not available."

    def _refresh_selected_plugin_options(self, plugin_infos: Iterable[Any]) -> None:
        ids = [getattr(info, "id", "") for info in plugin_infos if getattr(info, "id", "")]
        current = self.selected_plugin.value

        self.selected_plugin.options = ids

        if current in ids:
            self.selected_plugin.value = current
        elif ids:
            self.selected_plugin.value = ids[0]
        else:
            self.selected_plugin.value = None

    def _update_button_state(self) -> None:
        plugin_id = self.selected_plugin.value
        has_plugin = bool(plugin_id)
        open_count = self._open_instance_count(plugin_id) if plugin_id else 0
        force = bool(self.force_disable.value)

        self.enable_button.disabled = not has_plugin
        self.disable_button.disabled = (
            not has_plugin
            or plugin_id == self.SELF_PLUGIN_ID
            or (open_count > 0 and not force)
        )
        self.reload_button.disabled = (
            not has_plugin
            or plugin_id == self.SELF_PLUGIN_ID
            or (open_count > 0 and not force)
        )

        if not has_plugin:
            self.warning.object = ""
        elif plugin_id == self.SELF_PLUGIN_ID:
            self.warning.object = (
                "⚠️ Plugin Manager cannot disable or reload itself from this panel."
            )
        elif open_count > 0 and not force:
            self.warning.object = (
                f"⚠️ `{plugin_id}` has **{open_count}** open panel instance(s). "
                "Close them before disabling/reloading, or enable the force checkbox."
            )
        elif open_count > 0 and force:
            self.warning.object = (
                f"⚠️ Force mode enabled. `{plugin_id}` has **{open_count}** open panel instance(s)."
            )
        else:
            self.warning.object = ""

    def _render_details(self) -> None:
        plugin_id = self.selected_plugin.value

        if not plugin_id or self.manager is None:
            self.details_pane.object = "Select a plugin to inspect details."
            return

        try:
            info = self.manager.plugin_info(plugin_id)
        except Exception as exc:
            self.details_pane.object = f"Could not load plugin info for `{plugin_id}`: `{exc}`"
            return

        settings = {}
        try:
            settings = self.manager.get_plugin_settings(plugin_id)
        except Exception:
            settings = {}

        open_instances = self._safe_list_panel_instances(plugin_id=plugin_id)

        lines = [
            f"## {getattr(info, 'name', plugin_id)}",
            "",
            f"**ID:** `{getattr(info, 'id', '')}`",
            f"**Version:** `{getattr(info, 'version', '')}`",
            f"**Status:** `{self._status_text(getattr(info, 'status', ''))}`",
            f"**Source:** `{getattr(info, 'source', '')}`",
            f"**Path:** `{getattr(info, 'path', '')}`",
            f"**Open panel instances:** `{len(open_instances)}`",
            "",
            "### Description",
            "",
            getattr(info, "description", "") or "_No description provided._",
            "",
            "### Contributions",
            "",
            f"- Panels: `{', '.join(getattr(info, 'panels', []) or []) or 'none'}`",
            f"- Actions: `{', '.join(getattr(info, 'actions', []) or []) or 'none'}`",
            f"- Workflows: `{', '.join(getattr(info, 'workflows', []) or []) or 'none'}`",
            f"- Services: `{', '.join(getattr(info, 'services', []) or []) or 'none'}`",
            f"- Artifact viewers: `{', '.join(getattr(info, 'artifact_viewers', []) or []) or 'none'}`",
            "",
            "### Dependencies",
            "",
            f"- Requires: `{', '.join(getattr(info, 'requires', []) or []) or 'none'}`",
            f"- Optional requires: `{', '.join(getattr(info, 'optional_requires', []) or []) or 'none'}`",
            f"- Requires plugins: `{', '.join(getattr(info, 'requires_plugins', []) or []) or 'none'}`",
            "",
            "### Tags / capabilities",
            "",
            f"- Capabilities: `{', '.join(getattr(info, 'capabilities', []) or []) or 'none'}`",
            f"- Tags: `{', '.join(getattr(info, 'tags', []) or []) or 'none'}`",
            "",
        ]

        error = getattr(info, "error", None)
        if error:
            lines.extend(
                [
                    "### Error",
                    "",
                    str(error),
                    "",
                ]
            )

        if settings:
            lines.extend(
                [
                    "### Settings",
                    "",
                    json.dumps(settings, indent=2, default=str),
                    "",
                ]
            )

        self.details_pane.object = "\n".join(lines)

    # ------------------------------------------------------------------
    # Dataframe builders
    # ------------------------------------------------------------------

    def _plugins_dataframe(self, plugin_infos: Iterable[Any]) -> pd.DataFrame:
        rows = []

        for info in plugin_infos:
            plugin_id = getattr(info, "id", "")

            rows.append(
                {
                    "id": plugin_id,
                    "status": self._status_text(getattr(info, "status", "")),
                    "open": self._open_instance_count(plugin_id),
                    "panels": len(getattr(info, "panels", []) or []),
                    "actions": len(getattr(info, "actions", []) or []),
                    "services": len(getattr(info, "services", []) or []),
                    "error": self._short(getattr(info, "error", "")),
                }
            )

        return pd.DataFrame(
            rows,
            columns=[
                "id",
                "status",
                "open",
                "panels",
                "actions",
                "services",
                "error",
            ],
        )

    def _instances_dataframe(self) -> pd.DataFrame:
        rows = []

        for instance in self._safe_list_panel_instances():
            rows.append(
                {
                    "plugin": instance.get("plugin_id", ""),
                    "title": instance.get("title", ""),
                    "source": instance.get("source", ""),
                    "instance": instance.get("instance_id", ""),
                    "panel": instance.get("panel_id", ""),
                }
            )

        return pd.DataFrame(
            rows,
            columns=["plugin", "title", "source", "instance", "panel"],
        )

    def _panels_dataframe(self) -> pd.DataFrame:
        rows = []

        for reg in self._safe_list_panels():
            rows.append(
                {
                    "title": getattr(reg, "title", ""),
                    "plugin": getattr(reg, "plugin_id", ""),
                    "category": getattr(reg, "category", ""),
                    "id": getattr(reg, "id", ""),
                }
            )

        return pd.DataFrame(rows, columns=["title", "plugin", "category", "id"])

    def _actions_dataframe(self) -> pd.DataFrame:
        rows = []

        for reg in self._safe_list_actions():
            inputs = getattr(reg, "inputs", None)
            outputs = getattr(reg, "outputs", []) or []

            rows.append(
                {
                    "title": getattr(reg, "title", ""),
                    "plugin": getattr(reg, "plugin_id", ""),
                    "job": getattr(reg, "run_in_job", ""),
                    "selection": getattr(inputs, "selection", "") if inputs else "",
                    "outputs": self._join([getattr(o, "type", str(o)) for o in outputs]),
                    "id": getattr(reg, "id", ""),
                }
            )

        return pd.DataFrame(
            rows,
            columns=["title", "plugin", "job", "selection", "outputs", "id"],
        )

    def _workflows_dataframe(self) -> pd.DataFrame:
        rows = []

        for reg in self._safe_list_workflows():
            rows.append(
                {
                    "title": getattr(reg, "title", ""),
                    "plugin": getattr(reg, "plugin_id", ""),
                    "category": getattr(reg, "category", ""),
                    "id": getattr(reg, "id", ""),
                }
            )

        return pd.DataFrame(rows, columns=["title", "plugin", "category", "id"])

    def _services_dataframe(self) -> pd.DataFrame:
        rows = []

        for reg in self._safe_list_services():
            key = getattr(reg, "key", "")
            initialized = ""

            services = getattr(self.context, "services", None)
            if services is not None and hasattr(services, "is_initialized"):
                try:
                    initialized = services.is_initialized(key)
                except Exception:
                    initialized = ""

            rows.append(
                {
                    "key": key,
                    "plugin": getattr(reg, "plugin_id", ""),
                    "lazy": getattr(reg, "lazy", ""),
                    "init": initialized,
                }
            )

        return pd.DataFrame(rows, columns=["key", "plugin", "lazy", "init"])

    def _artifact_viewers_dataframe(self) -> pd.DataFrame:
        rows = []

        for reg in self._safe_list_artifact_viewers():
            rows.append(
                {
                    "artifact": getattr(reg, "artifact_type", ""),
                    "title": getattr(reg, "title", ""),
                    "plugin": getattr(reg, "plugin_id", ""),
                    "default": getattr(reg, "default", ""),
                }
            )

        return pd.DataFrame(
            rows,
            columns=["artifact", "title", "plugin", "default"],
        )

    def _discovery_errors_dataframe(self) -> pd.DataFrame:
        errors = self._safe_discovery_errors()

        rows = [
            {
                "candidate": candidate,
                "error": error,
            }
            for candidate, error in errors.items()
        ]

        return pd.DataFrame(rows, columns=["candidate", "error"])

    # ------------------------------------------------------------------
    # Safe manager wrappers
    # ------------------------------------------------------------------

    def _safe_list_plugins(self) -> List[Any]:
        if self.manager is None:
            return []
        try:
            return list(self.manager.list_plugins())
        except Exception:
            traceback.print_exc()
            return []

    def _safe_list_panels(self) -> List[Any]:
        if self.manager is None:
            return []
        try:
            return list(self.manager.list_panels())
        except Exception:
            traceback.print_exc()
            return []

    def _safe_list_actions(self) -> List[Any]:
        if self.manager is None:
            return []
        try:
            return list(self.manager.list_actions())
        except Exception:
            traceback.print_exc()
            return []

    def _safe_list_workflows(self) -> List[Any]:
        if self.manager is None:
            return []
        try:
            return list(self.manager.list_workflows())
        except Exception:
            traceback.print_exc()
            return []

    def _safe_list_services(self) -> List[Any]:
        if self.manager is None:
            return []
        try:
            return list(self.manager.list_services())
        except Exception:
            traceback.print_exc()
            return []

    def _safe_list_artifact_viewers(self) -> List[Any]:
        if self.manager is None:
            return []
        try:
            return list(self.manager.list_artifact_viewers())
        except Exception:
            traceback.print_exc()
            return []

    def _safe_list_panel_instances(
        self,
        plugin_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.manager is None:
            return []

        method = getattr(self.manager, "list_panel_instances", None)
        if not callable(method):
            return []

        try:
            return list(method(plugin_id=plugin_id))
        except Exception:
            traceback.print_exc()
            return []

    def _safe_discovery_errors(self) -> Dict[str, str]:
        if self.manager is None:
            return {}

        try:
            return dict(self.manager.list_discovery_errors())
        except Exception:
            traceback.print_exc()
            return {}

    def _open_instance_count(self, plugin_id: Optional[str]) -> int:
        if not plugin_id:
            return 0
        return len(self._safe_list_panel_instances(plugin_id=plugin_id))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _status_text(status: Any) -> str:
        value = getattr(status, "value", status)
        return str(value)

    @staticmethod
    def _join(values: Any) -> str:
        if values is None:
            return ""
        if isinstance(values, str):
            return values
        try:
            return ", ".join(str(v) for v in values)
        except Exception:
            return str(values)

    @staticmethod
    def _short(value: Any, limit: int = 180) -> str:
        if value is None:
            return ""
        text = str(value)
        return text if len(text) <= limit else text[: limit - 1] + "…"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        if self._disposed:
            return

        self._disposed = True

        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in list(self._subscriptions):
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass

        self._subscriptions.clear()

        for widget, watcher in list(self._watchers):
            try:
                widget.param.unwatch(watcher)
            except Exception:
                pass

        self._watchers.clear()
