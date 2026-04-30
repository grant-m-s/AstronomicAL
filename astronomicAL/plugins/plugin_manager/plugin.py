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
    version="0.1.0",
    description=(
        "Read and manage discovered AstronomicAL plugins, including their panels, "
        "actions, services, workflows, artifact viewers, and discovery errors."
    ),
    capabilities=["panel", "diagnostics", "plugins"],
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
        default_layout={"x": 0, "y": 0, "w": 8, "h": 6},
    )


def create_plugin_manager_panel(context, **kwargs):
    controller = PluginManagerPanel(context=context)
    return controller.view, controller


class PluginManagerPanel:
    """Plugin diagnostics and management panel.

    This panel intentionally uses the platform plugin manager as its data source.
    It does not own plugin state itself.

    It can:

    - list discovered plugins
    - show plugin status/errors
    - show registered panels/actions/workflows/services/artifact viewers
    - show discovery errors
    - run discovery again
    - enable, disable, or reload a selected plugin

    It guards against disabling itself, because doing so would remove the current
    panel registration while the panel is still open.
    """

    SELF_PLUGIN_ID = manifest.id

    def __init__(self, context):
        self.context = context
        self.manager = getattr(context, "plugins", None)
        self._disposed = False
        self._subscriptions: list[Any] = []
        self._watchers: list[tuple[Any, Any]] = []

        self.status = pn.pane.Markdown(
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

        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="default",
            width=110,
            height=32,
        )

        self.discover_button = pn.widgets.Button(
            name="Discover again",
            button_type="default",
            width=140,
            height=32,
        )

        self.enable_button = pn.widgets.Button(
            name="Enable",
            button_type="primary",
            width=100,
            height=32,
        )

        self.disable_button = pn.widgets.Button(
            name="Disable",
            button_type="warning",
            width=100,
            height=32,
        )

        self.reload_button = pn.widgets.Button(
            name="Reload",
            button_type="success",
            width=100,
            height=32,
        )

        self.plugins_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=False,
            disabled=True,
            pagination="local",
            page_size=20,
            sizing_mode="stretch_both",
        )

        self.panels_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=False,
            disabled=True,
            pagination="local",
            page_size=20,
            sizing_mode="stretch_both",
        )

        self.actions_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=False,
            disabled=True,
            pagination="local",
            page_size=20,
            sizing_mode="stretch_both",
        )

        self.workflows_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=False,
            disabled=True,
            pagination="local",
            page_size=20,
            sizing_mode="stretch_both",
        )

        self.services_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=False,
            disabled=True,
            pagination="local",
            page_size=20,
            sizing_mode="stretch_both",
        )

        self.artifact_viewers_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=False,
            disabled=True,
            pagination="local",
            page_size=20,
            sizing_mode="stretch_both",
        )

        self.discovery_errors_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["candidate", "error"]),
            show_index=False,
            disabled=True,
            pagination="local",
            page_size=20,
            sizing_mode="stretch_both",
        )

        self.details_pane = pn.pane.Markdown(
            "Select a plugin to inspect details.",
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
        )

        self.refresh_button.on_click(self._refresh_clicked)
        self.discover_button.on_click(self._discover_clicked)
        self.enable_button.on_click(self._enable_clicked)
        self.disable_button.on_click(self._disable_clicked)
        self.reload_button.on_click(self._reload_clicked)

        self._watch(self.selected_plugin, self._selected_plugin_changed, "value")
        self._subscribe_to_plugin_events()

        self.view = self._build_view()
        self.refresh()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_view(self):
        title = pn.pane.HTML(
            "<h2 style='margin: 0; padding: 0; line-height: 1.2;'>Plugin Manager</h2>",
            height=34,
            min_height=34,
            max_height=34,
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )

        controls = pn.Column(
            self.selected_plugin,
            pn.Row(
                self.refresh_button,
                self.discover_button,
                self.enable_button,
                self.disable_button,
                self.reload_button,
                sizing_mode="stretch_width",
                align="center",
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        contributions_tabs = pn.Tabs(
            ("Panels", self._table_section(self.panels_table)),
            ("Actions", self._table_section(self.actions_table)),
            ("Workflows", self._table_section(self.workflows_table)),
            ("Services", self._table_section(self.services_table)),
            ("Artifact Viewers", self._table_section(self.artifact_viewers_table)),
            dynamic=True,
            sizing_mode="stretch_both",
        )

        tabs = pn.Tabs(
            ("Plugins", self._table_section(self.plugins_table)),
            ("Contributions", contributions_tabs),
            ("Discovery Errors", self._table_section(self.discovery_errors_table)),
            ("Details", pn.Column(self.details_pane, sizing_mode="stretch_both", scroll=True)),
            dynamic=True,
            sizing_mode="stretch_both",
        )

        return pn.Column(
            title,
            controls,
            self.status,
            tabs,
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
            styles={
                "overflow": "hidden",
                "padding": "0 8px 8px 8px",
            },
        )

    @staticmethod
    def _table_section(table):
        return pn.Column(
            table,
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
            styles={"overflow": "hidden"},
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

        for topic in ("plugin.enabled", "plugin.disabled"):
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
            self.status.object = "Discovery completed."
        except Exception as exc:
            self.status.object = f"Discovery failed: `{exc}`"
            traceback.print_exc()

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

        if plugin_id == self.SELF_PLUGIN_ID:
            self.status.object = (
                "Refusing to disable Plugin Manager from inside its own panel. "
                "Use a different management surface or restart with the plugin disabled."
            )
            return

        if self.manager is None:
            self.status.object = "PluginManager is not available on context."
            return

        try:
            self.manager.disable(plugin_id, context=self.context)
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

        if plugin_id == self.SELF_PLUGIN_ID:
            self.status.object = (
                "Refusing to reload Plugin Manager from inside its own panel. "
                "Reload another plugin or restart the server."
            )
            return

        if self.manager is None:
            self.status.object = "PluginManager is not available on context."
            return

        try:
            self.manager.reload(plugin_id, self.context)
            self.status.object = f"Reloaded `{plugin_id}`."
        except Exception as exc:
            self.status.object = f"Reload failed for `{plugin_id}`: `{exc}`"
            traceback.print_exc()

        self.refresh()

    def _selected_plugin_changed(self, _event=None) -> None:
        self._render_details()
        self._update_button_state()

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
        self.panels_table.value = self._panels_dataframe()
        self.actions_table.value = self._actions_dataframe()
        self.workflows_table.value = self._workflows_dataframe()
        self.services_table.value = self._services_dataframe()
        self.artifact_viewers_table.value = self._artifact_viewers_dataframe()
        self.discovery_errors_table.value = self._discovery_errors_dataframe()

        self._render_details()
        self._update_button_state()

        enabled_count = len([info for info in plugin_infos if str(info.status).endswith("ENABLED") or str(info.status) == "enabled"])
        error_count = len([info for info in plugin_infos if str(info.status).endswith("ERROR") or str(info.status) == "error"])

        self.status.object = (
            f"Plugins: **{len(plugin_infos)}** • "
            f"Enabled: **{enabled_count}** • "
            f"Errors: **{error_count}** • "
            f"Panels: **{len(self._safe_list_panels())}** • "
            f"Actions: **{len(self._safe_list_actions())}**"
        )

    def _clear_all_tables(self) -> None:
        empty = pd.DataFrame()
        self.plugins_table.value = empty
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

        self.enable_button.disabled = not has_plugin
        self.disable_button.disabled = not has_plugin or plugin_id == self.SELF_PLUGIN_ID
        self.reload_button.disabled = not has_plugin or plugin_id == self.SELF_PLUGIN_ID

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

        lines = [
            f"## {getattr(info, 'name', plugin_id)}",
            "",
            f"**ID:** `{getattr(info, 'id', '')}`",
            f"**Version:** `{getattr(info, 'version', '')}`",
            f"**Status:** `{self._status_text(getattr(info, 'status', ''))}`",
            f"**Source:** `{getattr(info, 'source', '')}`",
            f"**Path:** `{getattr(info, 'path', '')}`",
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
                    "```text",
                    str(error),
                    "```",
                    "",
                ]
            )

        if settings:
            lines.extend(
                [
                    "### Settings",
                    "",
                    "```json",
                    json.dumps(settings, indent=2, default=str),
                    "```",
                ]
            )

        self.details_pane.object = "\n".join(lines)

    # ------------------------------------------------------------------
    # Dataframe builders
    # ------------------------------------------------------------------

    def _plugins_dataframe(self, plugin_infos: Iterable[Any]) -> pd.DataFrame:
        rows = []

        for info in plugin_infos:
            rows.append(
                {
                    "id": getattr(info, "id", ""),
                    "name": getattr(info, "name", ""),
                    "version": getattr(info, "version", ""),
                    "status": self._status_text(getattr(info, "status", "")),
                    "source": getattr(info, "source", ""),
                    "path": getattr(info, "path", ""),
                    "panels": len(getattr(info, "panels", []) or []),
                    "actions": len(getattr(info, "actions", []) or []),
                    "workflows": len(getattr(info, "workflows", []) or []),
                    "services": len(getattr(info, "services", []) or []),
                    "artifact_viewers": len(getattr(info, "artifact_viewers", []) or []),
                    "error": self._short(getattr(info, "error", "")),
                }
            )

        return pd.DataFrame(
            rows,
            columns=[
                "id",
                "name",
                "version",
                "status",
                "source",
                "path",
                "panels",
                "actions",
                "workflows",
                "services",
                "artifact_viewers",
                "error",
            ],
        )

    def _panels_dataframe(self) -> pd.DataFrame:
        rows = []

        for reg in self._safe_list_panels():
            rows.append(
                {
                    "id": getattr(reg, "id", ""),
                    "title": getattr(reg, "title", ""),
                    "plugin_id": getattr(reg, "plugin_id", ""),
                    "category": getattr(reg, "category", ""),
                    "required_mappings": self._join(getattr(reg, "required_mappings", [])),
                    "uses_services": self._join(getattr(reg, "uses_services", [])),
                    "produces": self._join(getattr(reg, "produces", [])),
                    "requires": self._join(getattr(reg, "requires", [])),
                    "description": getattr(reg, "description", ""),
                }
            )

        return pd.DataFrame(
            rows,
            columns=[
                "id",
                "title",
                "plugin_id",
                "category",
                "required_mappings",
                "uses_services",
                "produces",
                "requires",
                "description",
            ],
        )

    def _actions_dataframe(self) -> pd.DataFrame:
        rows = []

        for reg in self._safe_list_actions():
            inputs = getattr(reg, "inputs", None)
            outputs = getattr(reg, "outputs", []) or []

            rows.append(
                {
                    "id": getattr(reg, "id", ""),
                    "title": getattr(reg, "title", ""),
                    "plugin_id": getattr(reg, "plugin_id", ""),
                    "category": getattr(reg, "category", ""),
                    "run_in_job": getattr(reg, "run_in_job", ""),
                    "selection": getattr(inputs, "selection", "") if inputs else "",
                    "columns": getattr(inputs, "columns", "") if inputs else "",
                    "numeric_columns": getattr(inputs, "numeric_columns", "") if inputs else "",
                    "outputs": self._join([getattr(o, "type", str(o)) for o in outputs]),
                    "requires": self._join(getattr(reg, "requires", [])),
                    "description": getattr(reg, "description", ""),
                }
            )

        return pd.DataFrame(
            rows,
            columns=[
                "id",
                "title",
                "plugin_id",
                "category",
                "run_in_job",
                "selection",
                "columns",
                "numeric_columns",
                "outputs",
                "requires",
                "description",
            ],
        )

    def _workflows_dataframe(self) -> pd.DataFrame:
        rows = []

        for reg in self._safe_list_workflows():
            rows.append(
                {
                    "id": getattr(reg, "id", ""),
                    "title": getattr(reg, "title", ""),
                    "plugin_id": getattr(reg, "plugin_id", ""),
                    "category": getattr(reg, "category", ""),
                    "requires": self._join(getattr(reg, "requires", [])),
                    "description": getattr(reg, "description", ""),
                }
            )

        return pd.DataFrame(
            rows,
            columns=["id", "title", "plugin_id", "category", "requires", "description"],
        )

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
                    "plugin_id": getattr(reg, "plugin_id", ""),
                    "lazy": getattr(reg, "lazy", ""),
                    "replace": getattr(reg, "replace", ""),
                    "initialized": initialized,
                    "requires": self._join(getattr(reg, "requires", [])),
                    "description": getattr(reg, "description", ""),
                }
            )

        return pd.DataFrame(
            rows,
            columns=[
                "key",
                "plugin_id",
                "lazy",
                "replace",
                "initialized",
                "requires",
                "description",
            ],
        )

    def _artifact_viewers_dataframe(self) -> pd.DataFrame:
        rows = []

        for reg in self._safe_list_artifact_viewers():
            rows.append(
                {
                    "artifact_type": getattr(reg, "artifact_type", ""),
                    "id": getattr(reg, "id", ""),
                    "title": getattr(reg, "title", ""),
                    "plugin_id": getattr(reg, "plugin_id", ""),
                    "priority": getattr(reg, "priority", ""),
                    "default": getattr(reg, "default", ""),
                    "requires": self._join(getattr(reg, "requires", [])),
                    "description": getattr(reg, "description", ""),
                }
            )

        return pd.DataFrame(
            rows,
            columns=[
                "artifact_type",
                "id",
                "title",
                "plugin_id",
                "priority",
                "default",
                "requires",
                "description",
            ],
        )

    def _discovery_errors_dataframe(self) -> pd.DataFrame:
        if self.manager is None:
            return pd.DataFrame(columns=["candidate", "error"])

        try:
            errors = self.manager.list_discovery_errors()
        except Exception:
            errors = {}

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
    def _short(value: Any, limit: int = 240) -> str:
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