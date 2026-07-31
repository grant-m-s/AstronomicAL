from __future__ import annotations

import html
import json
import traceback
from datetime import datetime
from typing import Any, Callable, Mapping

import pandas as pd
import panel as pn

from .diagnostics import (
    PluginManagerSnapshot,
    PluginSnapshot,
    collect_snapshot,
    filter_plugins,
    provides_summary,
    source_label,
)
from .styles import PLUGIN_MANAGER_CSS


class PluginManagerPanel:
    """Simple, user-facing management surface for AstronomicAL plugins."""

    state_version = 1
    SELF_PLUGIN_ID = "core.plugin_manager"

    def __init__(self, context: Any):
        self.context = context
        self.manager = getattr(context, "plugins", None)
        self._disposed = False
        self._restoring = False
        self._refresh_scheduled = False
        self._syncing_selection = False
        self._subscriptions: list[Any] = []
        self._watchers: list[tuple[Any, Any]] = []
        self._snapshot: PluginManagerSnapshot | None = None
        self._visible_plugins: list[PluginSnapshot] = []
        self._selected_plugin_id: str | None = None

        self.search = pn.widgets.TextInput(
            name="Search plugins",
            placeholder="Name, feature, capability, or plugin ID…",
            sizing_mode="stretch_width",
            margin=0,
        )
        self.status_filter = pn.widgets.Select(
            name="Show",
            options={
                "All plugins": "all",
                "Enabled": "enabled",
                "Available": "available",
                "Needs attention": "issues",
            },
            value="all",
            sizing_mode="stretch_width",
            margin=0,
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="default",
            sizing_mode="stretch_width",
            height=32,
            margin=0,
        )
        self.discover_button = pn.widgets.Button(
            name="Scan for plugins",
            button_type="primary",
            sizing_mode="stretch_width",
            height=32,
            margin=0,
        )

        self.enable_button = pn.widgets.Button(
            name="Enable plugin",
            button_type="primary",
            sizing_mode="stretch_width",
            height=34,
            margin=0,
        )
        self.disable_button = pn.widgets.Button(
            name="Disable plugin",
            button_type="warning",
            sizing_mode="stretch_width",
            height=34,
            margin=0,
        )
        self.reload_button = pn.widgets.Button(
            name="Reload local plugin",
            button_type="default",
            sizing_mode="stretch_width",
            height=34,
            margin=0,
            visible=False,
        )

        self.plugin_table = pn.widgets.Tabulator(
            pd.DataFrame(
                columns=["Plugin", "Status", "Version", "Open", "Provides", "plugin_id"]
            ),
            show_index=False,
            selectable=1,
            pagination="local",
            page_size=12,
            sizing_mode="stretch_width",
            height=360,
            min_height=250,
            margin=0,
            hidden_columns=["plugin_id"],
            widths={
                "Plugin": 180,
                "Status": 115,
                "Version": 80,
                "Open": 60,
            },
            configuration={
                "layout": "fitColumns",
                "responsiveLayout": "collapse",
                "placeholder": "No plugins match this view.",
            },
            css_classes=["al-pm-table"],
            stylesheets=[PLUGIN_MANAGER_CSS],
        )

        self.header = self._html_pane()
        self.summary = self._html_pane()
        self.operation_banner = self._html_pane()
        self.selected_details = self._html_pane()
        self.action_note = self._html_pane()
        self.discovery_issues = self._html_pane()

        self.refresh_button.on_click(self._refresh_clicked)
        self.discover_button.on_click(self._discover_clicked)
        self.enable_button.on_click(self._enable_clicked)
        self.disable_button.on_click(self._disable_clicked)
        self.reload_button.on_click(self._reload_clicked)

        self._watch(self.search, self._filters_changed, "value")
        self._watch(self.search, self._filters_changed, "value_input")
        self._watch(self.status_filter, self._filters_changed, "value")
        self._watch(self.plugin_table, self._table_selection_changed, "selection")
        self._subscribe_to_plugin_events()

        self.discovery_section = self._section(
            "Discovery issues",
            "Plugins that could not be inspected during the latest scan",
            self.discovery_issues,
        )
        self.discovery_section.visible = False

        self.view = self._build_view()
        self.refresh()

    @staticmethod
    def _html_pane() -> pn.pane.HTML:
        return pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=0,
        )

    def _build_view(self) -> pn.Column:
        controls = pn.Column(
            self.search,
            self.status_filter,
            pn.GridBox(
                self.refresh_button,
                self.discover_button,
                ncols=2,
                sizing_mode="stretch_width",
                margin=0,
            ),
            sizing_mode="stretch_width",
            css_classes=["al-pm-card"],
            styles={
                "background": "#ffffff",
                "border": "1px solid #dfe5ee",
                "border-radius": "9px",
                "box-shadow": "0 2px 8px rgba(27, 43, 65, 0.045)",
                "padding": "12px",
            },
            margin=(0, 0, 10, 0),
        )

        plugin_list = self._section(
            "Plugins",
            "Select a plugin to see what it adds and manage its availability",
            self.plugin_table,
        )

        actions = pn.GridBox(
            self.enable_button,
            self.disable_button,
            self.reload_button,
            ncols=2,
            sizing_mode="stretch_width",
            margin=(10, 0, 0, 0),
        )

        selected = pn.Column(
            pn.pane.HTML(
                '<div class="al-pm-section-title"><h3>Selected plugin</h3>'
                '<span>Health, useful features, open panels, and lifecycle controls</span></div>',
                sizing_mode="stretch_width",
                stylesheets=[PLUGIN_MANAGER_CSS],
                margin=0,
            ),
            self.selected_details,
            self.action_note,
            actions,
            sizing_mode="stretch_width",
            css_classes=["al-pm-card"],
            styles={
                "background": "#ffffff",
                "border": "1px solid #dfe5ee",
                "border-radius": "9px",
                "box-shadow": "0 2px 8px rgba(27, 43, 65, 0.045)",
                "padding": "12px",
            },
            margin=(0, 0, 10, 0),
        )

        return pn.Column(
            self.header,
            self.operation_banner,
            self.summary,
            controls,
            plugin_list,
            selected,
            self.discovery_section,
            sizing_mode="stretch_both",
            min_width=220,
            scroll=True,
            css_classes=["al-plugin-manager"],
            styles={
                "background": "transparent",
                "box-sizing": "border-box",
                "overflow-x": "hidden",
                "overflow-y": "auto",
                "padding": "8px",
            },
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=0,
        )

    @staticmethod
    def _section(title: str, detail: str, content: Any) -> pn.Column:
        heading = pn.pane.HTML(
            '<div class="al-pm-section-title">'
            f"<h3>{html.escape(title)}</h3>"
            f"<span>{html.escape(detail)}</span>"
            "</div>",
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=0,
        )
        return pn.Column(
            heading,
            content,
            sizing_mode="stretch_width",
            css_classes=["al-pm-card"],
            styles={
                "background": "#ffffff",
                "border": "1px solid #dfe5ee",
                "border-radius": "9px",
                "box-shadow": "0 2px 8px rgba(27, 43, 65, 0.045)",
                "padding": "12px",
            },
            margin=(0, 0, 10, 0),
        )

    # ------------------------------------------------------------------
    # Snapshot and filtering
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        if self._disposed:
            return

        try:
            snapshot = collect_snapshot(self.context)
        except Exception as exc:
            self._snapshot = None
            self.header.object = self._header_html(None)
            self.summary.object = self._empty_html("Plugin information is unavailable.")
            self.selected_details.object = self._empty_html("No plugin can be selected.")
            self.discovery_issues.object = self._empty_html("No discovery information.")
            self.discovery_section.visible = False
            self.plugin_table.value = pd.DataFrame()
            self._set_operation_message("danger", f"Unable to read plugin state: {exc}")
            self._update_action_state(None)
            return

        self._snapshot = snapshot
        self.header.object = self._header_html(snapshot)
        self.summary.object = self._summary_html(snapshot)
        self.discovery_issues.object = self._discovery_issues_html(snapshot)
        self.discovery_section.visible = bool(snapshot.discovery_issues)
        self._apply_filters()

    def _apply_filters(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return

        visible = filter_plugins(
            snapshot.plugins,
            query=self._search_value(),
            status_filter=str(self.status_filter.value or "all"),
        )
        self._visible_plugins = visible

        rows = [
            {
                "Plugin": plugin.name,
                "Status": self._table_status(plugin),
                "Version": plugin.version,
                "Open": len(plugin.open_instances),
                "Provides": provides_summary(plugin),
                "plugin_id": plugin.id,
            }
            for plugin in visible
        ]
        self.plugin_table.value = pd.DataFrame(
            rows,
            columns=["Plugin", "Status", "Version", "Open", "Provides", "plugin_id"],
        )

        visible_ids = [plugin.id for plugin in visible]
        if self._selected_plugin_id not in visible_ids:
            self._selected_plugin_id = visible_ids[0] if visible_ids else None

        self._sync_table_selection()
        self._render_selected_plugin()

    def _sync_table_selection(self) -> None:
        self._syncing_selection = True
        try:
            if self._selected_plugin_id is None:
                self.plugin_table.selection = []
                return
            for index, plugin in enumerate(self._visible_plugins):
                if plugin.id == self._selected_plugin_id:
                    self.plugin_table.selection = [index]
                    return
            self.plugin_table.selection = []
        finally:
            self._syncing_selection = False

    def _table_selection_changed(self, event: Any) -> None:
        if self._disposed or self._syncing_selection or self._restoring:
            return
        selection = list(getattr(event, "new", None) or [])
        if not selection:
            return
        try:
            index = int(selection[0])
            plugin = self._visible_plugins[index]
        except Exception:
            return
        self._selected_plugin_id = plugin.id
        self._render_selected_plugin()

    def _filters_changed(self, _event: Any = None) -> None:
        if self._disposed or self._restoring:
            return
        self._apply_filters()

    def _search_value(self) -> str:
        value_input = getattr(self.search, "value_input", None)
        if value_input is not None:
            return str(value_input or "")
        return str(self.search.value or "")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _header_html(self, snapshot: PluginManagerSnapshot | None) -> str:
        if snapshot is None:
            health_label = "Unavailable"
            health_tone = "danger"
            refreshed = "—"
            total = 0
        else:
            total = len(snapshot.plugins)
            refreshed = self._clock(snapshot.captured_at)
            if snapshot.issue_count:
                health_label = f"{snapshot.issue_count} issue{'s' if snapshot.issue_count != 1 else ''}"
                health_tone = "warning"
            else:
                health_label = "Healthy"
                health_tone = ""

        return f"""
        <div class="al-pm-header">
          <div class="al-pm-header-main">
            <div class="al-pm-eyebrow">Platform management</div>
            <h2 class="al-pm-title">Plugin Manager</h2>
            <div class="al-pm-subtitle">Choose which features are available, understand what each plugin adds, and surface problems without navigating technical registry tables.</div>
          </div>
          <div class="al-pm-header-meta">
            <div class="al-pm-health {health_tone}"><span class="al-pm-health-dot"></span>{html.escape(health_label)}</div>
            <div class="al-pm-updated">{total:,} plugins · updated {html.escape(refreshed)}</div>
          </div>
        </div>
        """

    def _summary_html(self, snapshot: PluginManagerSnapshot) -> str:
        return (
            '<div class="al-pm-summary-grid">'
            + self._metric_html(
                "Enabled",
                str(snapshot.enabled_count),
                "Currently available in menus and workflows",
                "success",
            )
            + self._metric_html(
                "Available",
                str(snapshot.available_count),
                "Discovered but not currently enabled",
            )
            + self._metric_html(
                "Open panels",
                str(snapshot.open_panel_count),
                "Live plugin panel instances",
            )
            + self._metric_html(
                "Needs attention",
                str(snapshot.issue_count),
                "Plugin or discovery problems",
                "danger" if snapshot.issue_count else "success",
            )
            + "</div>"
        )

    @staticmethod
    def _metric_html(label: str, value: str, detail: str, tone: str = "") -> str:
        tone_class = f" {tone}" if tone else ""
        return f"""
        <div class="al-pm-metric">
          <div class="al-pm-metric-label">{html.escape(label)}</div>
          <div class="al-pm-metric-value{tone_class}">{html.escape(value)}</div>
          <div class="al-pm-metric-detail">{html.escape(detail)}</div>
        </div>
        """

    def _render_selected_plugin(self) -> None:
        plugin = self._selected_plugin()
        self._update_action_state(plugin)

        if plugin is None:
            self.selected_details.object = self._empty_html(
                "No plugin matches the current search and filter."
            )
            self.action_note.object = ""
            return

        validation = self._validation_result(plugin.id)
        self.selected_details.object = self._plugin_details_html(plugin, validation)
        self.action_note.object = self._action_note_html(plugin)

    def _plugin_details_html(self, plugin: PluginSnapshot, validation: Any) -> str:
        description = plugin.description or "No description was provided for this plugin."
        panels = self._contribution_chips("Panel", plugin.panels)
        actions = self._contribution_chips("Action", plugin.actions)
        workflows = self._contribution_chips("Workflow", plugin.workflows)
        open_instances = self._open_instances_html(plugin)
        readiness = self._readiness_html(plugin, validation)
        registration_note = ""
        if plugin.status != "enabled" and plugin.contribution_count == 0:
            registration_note = self._banner_html(
                "info",
                "This plugin registers its panels and actions when it is enabled.",
            )
        technical = self._technical_html(plugin)

        return f"""
        <div class="al-pm-plugin-head">
          <div>
            <h3 class="al-pm-plugin-name">{html.escape(plugin.name)}</h3>
            <div class="al-pm-plugin-version">Version {html.escape(plugin.version)}</div>
          </div>
          <span class="al-pm-status-pill {plugin.status_tone}">{html.escape(plugin.status_label)}</span>
        </div>
        <p class="al-pm-description">{html.escape(description)}</p>
        <div class="al-pm-mini-grid">
          <div class="al-pm-mini"><div class="al-pm-mini-label">Source</div><div class="al-pm-mini-value">{html.escape(source_label(plugin.source))}</div></div>
          <div class="al-pm-mini"><div class="al-pm-mini-label">Open panels</div><div class="al-pm-mini-value">{len(plugin.open_instances):,}</div></div>
          <div class="al-pm-mini"><div class="al-pm-mini-label">Panels</div><div class="al-pm-mini-value">{len(plugin.panels):,}</div></div>
          <div class="al-pm-mini"><div class="al-pm-mini-label">Actions</div><div class="al-pm-mini-value">{len(plugin.actions):,}</div></div>
        </div>
        {readiness}
        {registration_note}
        {self._optional_group("Panels added to the workspace", panels)}
        {self._optional_group("Actions and tools", actions)}
        {self._optional_group("Workflows", workflows)}
        {self._optional_group("Currently open", open_instances)}
        {technical}
        """

    def _readiness_html(self, plugin: PluginSnapshot, validation: Any) -> str:
        errors = list(getattr(validation, "errors", None) or []) if validation is not None else []
        warnings = list(getattr(validation, "warnings", None) or []) if validation is not None else []

        if plugin.error:
            errors.insert(0, plugin.error)

        if not errors and not warnings:
            if plugin.status == "enabled":
                return '<div class="al-pm-banner success al-pm-readiness">Plugin is enabled and ready.</div>'
            return '<div class="al-pm-banner info al-pm-readiness">No dependency problems were found.</div>'

        chunks: list[str] = []
        if errors:
            items = "".join(f"<li>{html.escape(str(item))}</li>" for item in errors)
            chunks.append(
                '<div class="al-pm-banner danger al-pm-readiness"><strong>Cannot start cleanly</strong>'
                f"<ul>{items}</ul></div>"
            )
        if warnings:
            items = "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings)
            chunks.append(
                '<div class="al-pm-banner warning al-pm-readiness"><strong>Optional items unavailable</strong>'
                f"<ul>{items}</ul></div>"
            )
        return "".join(chunks)

    @staticmethod
    def _optional_group(title: str, content: str) -> str:
        if not content:
            return ""
        return f'<div class="al-pm-subheading">{html.escape(title)}</div>{content}'

    @staticmethod
    def _contribution_chips(kind: str, contributions: Any) -> str:
        items = list(contributions or [])
        if not items:
            return ""
        chips = "".join(
            '<span class="al-pm-chip">'
            f"<strong>{html.escape(kind)}</strong>{html.escape(item.title)}"
            "</span>"
            for item in items
        )
        return f'<div class="al-pm-chip-list">{chips}</div>'

    @staticmethod
    def _open_instances_html(plugin: PluginSnapshot) -> str:
        if not plugin.open_instances:
            return ""
        return "".join(
            '<div class="al-pm-instance">'
            f'<div class="al-pm-instance-title">{html.escape(item.title)}</div>'
            f'<div class="al-pm-instance-meta">Instance {html.escape(item.instance_id)} · {html.escape(item.panel_id)}</div>'
            "</div>"
            for item in plugin.open_instances
        )

    def _technical_html(self, plugin: PluginSnapshot) -> str:
        services = ", ".join(item.title for item in plugin.services) or "None"
        viewers = ", ".join(item.title for item in plugin.artifact_viewers) or "None"
        dependencies = ", ".join(plugin.requires) or "None"
        optional_dependencies = ", ".join(plugin.optional_requires) or "None"
        plugin_dependencies = ", ".join(plugin.requires_plugins) or "None"
        capabilities = ", ".join(plugin.capabilities) or "None"
        tags = ", ".join(plugin.tags) or "None"
        path = plugin.path or "Not provided"
        settings = json.dumps(dict(plugin.settings), indent=2, default=str) if plugin.settings else "None"

        rows = [
            ("Plugin ID", plugin.id),
            ("Source", plugin.source or "Unknown"),
            ("Path", path),
            ("Capabilities", capabilities),
            ("Tags", tags),
            ("Services", services),
            ("Artifact viewers", viewers),
            ("Required packages", dependencies),
            ("Optional packages", optional_dependencies),
            ("Required plugins", plugin_dependencies),
            ("Settings", settings),
        ]
        body = "".join(
            '<div class="al-pm-technical-key">'
            f"{html.escape(key)}</div>"
            '<div class="al-pm-technical-value">'
            f"{html.escape(str(value))}</div>"
            for key, value in rows
        )
        return (
            '<details class="al-pm-technical">'
            "<summary>Technical details</summary>"
            f'<div class="al-pm-technical-grid">{body}</div>'
            "</details>"
        )

    def _discovery_issues_html(self, snapshot: PluginManagerSnapshot) -> str:
        if not snapshot.discovery_issues:
            return self._empty_html("No discovery problems were reported.")
        return "".join(
            '<div class="al-pm-issue">'
            f'<div class="al-pm-issue-title">{html.escape(issue.candidate)}</div>'
            f'<div class="al-pm-issue-meta">{html.escape(issue.error)}</div>'
            "</div>"
            for issue in snapshot.discovery_issues
        )

    def _action_note_html(self, plugin: PluginSnapshot) -> str:
        if plugin.id == self.SELF_PLUGIN_ID:
            return self._banner_html(
                "info",
                "Plugin Manager cannot disable or reload itself from inside this panel.",
            )
        if plugin.status == "enabled" and plugin.open_instances:
            count = len(plugin.open_instances)
            return self._banner_html(
                "warning",
                f"Disabling or reloading this plugin closes {count} open panel{'s' if count != 1 else ''} and cancels its tracked jobs.",
            )
        if plugin.is_local:
            return self._banner_html(
                "info",
                "Reload is intended for local development after editing plugin code.",
            )
        return ""

    @staticmethod
    def _table_status(plugin: PluginSnapshot) -> str:
        marker = {
            "enabled": "●",
            "disabled": "○",
            "discovered": "◌",
            "error": "⚠",
        }.get(plugin.status, "•")
        return f"{marker} {plugin.status_label}"

    @staticmethod
    def _empty_html(message: str) -> str:
        return f'<div class="al-pm-empty">{html.escape(message)}</div>'

    @staticmethod
    def _banner_html(tone: str, message: str) -> str:
        return f'<div class="al-pm-banner {html.escape(tone)}">{html.escape(message)}</div>'

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _refresh_clicked(self, _event: Any = None) -> None:
        self.refresh()

    def _discover_clicked(self, _event: Any = None) -> None:
        if self.manager is None:
            self._set_operation_message("danger", "PluginManager is not available.")
            return
        self._run_operation(
            label="Scanning for plugins",
            callback=self.manager.discover,
            success_message="Plugin scan completed.",
            registry_operation="discovered",
        )

    def _enable_clicked(self, _event: Any = None) -> None:
        plugin = self._selected_plugin()
        if plugin is None or self.manager is None:
            return
        self._run_operation(
            label=f"Enabling {plugin.name}",
            callback=lambda: self.manager.enable(plugin.id, self.context),
            success_message=f"Enabled {plugin.name}.",
            plugin_id=plugin.id,
            registry_operation="enabled",
        )

    def _disable_clicked(self, _event: Any = None) -> None:
        plugin = self._selected_plugin()
        if plugin is None or self.manager is None or plugin.id == self.SELF_PLUGIN_ID:
            return
        self._run_operation(
            label=f"Disabling {plugin.name}",
            callback=lambda: self.manager.disable(plugin.id, context=self.context),
            success_message=f"Disabled {plugin.name}.",
            plugin_id=plugin.id,
            registry_operation="disabled",
        )

    def _reload_clicked(self, _event: Any = None) -> None:
        plugin = self._selected_plugin()
        if (
            plugin is None
            or self.manager is None
            or plugin.id == self.SELF_PLUGIN_ID
            or not plugin.is_local
        ):
            return
        self._run_operation(
            label=f"Reloading {plugin.name}",
            callback=lambda: self.manager.reload(plugin.id, self.context),
            success_message=f"Reloaded {plugin.name}.",
            plugin_id=plugin.id,
            registry_operation="reloaded",
        )

    def _run_operation(
        self,
        *,
        label: str,
        callback: Callable[[], Any],
        success_message: str,
        registry_operation: str,
        plugin_id: str | None = None,
    ) -> None:
        self._set_busy(True)
        self._set_operation_message("info", f"{label}…")
        try:
            callback()
            self._publish_registry_changed(plugin_id, registry_operation)
        except Exception as exc:
            traceback.print_exc()
            self._set_operation_message("danger", f"{label} failed: {exc}")
        else:
            self._set_operation_message("success", success_message)
        finally:
            self._set_busy(False)
            self.refresh()

    def _set_busy(self, busy: bool) -> None:
        for button in (
            self.refresh_button,
            self.discover_button,
            self.enable_button,
            self.disable_button,
            self.reload_button,
        ):
            try:
                button.loading = busy
            except Exception:
                pass
        self.search.disabled = busy
        self.status_filter.disabled = busy
        if not busy:
            self._update_action_state(self._selected_plugin())

    def _set_operation_message(self, tone: str, message: str) -> None:
        self.operation_banner.object = self._banner_html(tone, message) if message else ""

    def _update_action_state(self, plugin: PluginSnapshot | None) -> None:
        if plugin is None:
            self.enable_button.disabled = True
            self.disable_button.disabled = True
            self.reload_button.disabled = True
            self.reload_button.visible = False
            return

        is_self = plugin.id == self.SELF_PLUGIN_ID
        self.enable_button.disabled = plugin.status == "enabled"
        self.disable_button.disabled = is_self or plugin.status != "enabled"
        self.reload_button.visible = plugin.is_local and not is_self
        self.reload_button.disabled = plugin.status != "enabled"

        open_count = len(plugin.open_instances)
        self.disable_button.name = (
            f"Disable and close {open_count} panel{'s' if open_count != 1 else ''}"
            if plugin.status == "enabled" and open_count
            else "Disable plugin"
        )
        self.reload_button.name = (
            f"Reload and close {open_count} panel{'s' if open_count != 1 else ''}"
            if open_count
            else "Reload local plugin"
        )

    def _publish_registry_changed(self, plugin_id: str | None, operation: str) -> None:
        if self.manager is not None:
            publish = getattr(self.manager, "publish_registry_changed", None)
            if callable(publish):
                try:
                    publish(
                        self.context,
                        plugin_id=plugin_id,
                        operation=operation,
                    )
                    return
                except Exception:
                    traceback.print_exc()

        events = getattr(self.context, "events", None)
        if events is not None:
            try:
                events.publish(
                    "plugin.registry.changed",
                    {"plugin_id": plugin_id, "operation": operation},
                )
            except Exception:
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Event lifecycle
    # ------------------------------------------------------------------

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
                subscription = events.subscribe(
                    topic,
                    self._on_plugin_event,
                    owner_id="core.plugin_manager.panel",
                    owner_label="Plugin Manager",
                    owner_kind="plugin-panel",
                )
            except TypeError:
                subscription = events.subscribe(topic, self._on_plugin_event)
            self._subscriptions.append(subscription)

    def _on_plugin_event(self, _topic: str, _payload: Any) -> None:
        if not self._disposed:
            self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        if self._refresh_scheduled or self._disposed:
            return
        self._refresh_scheduled = True

        def _refresh() -> None:
            self._refresh_scheduled = False
            if not self._disposed:
                self.refresh()

        try:
            document = pn.state.curdoc
        except Exception:
            document = None

        if document is not None:
            try:
                document.add_next_tick_callback(_refresh)
                return
            except Exception:
                pass
        _refresh()

    def _watch(self, widget: Any, callback: Callable[..., Any], attr: str) -> None:
        try:
            watcher = widget.param.watch(callback, attr)
            self._watchers.append((widget, watcher))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers and persisted state
    # ------------------------------------------------------------------

    def _selected_plugin(self) -> PluginSnapshot | None:
        if self._snapshot is None or self._selected_plugin_id is None:
            return None
        for plugin in self._snapshot.plugins:
            if plugin.id == self._selected_plugin_id:
                return plugin
        return None

    def _validation_result(self, plugin_id: str) -> Any:
        if self.manager is None:
            return None
        validate = getattr(self.manager, "validate", None)
        if not callable(validate):
            return None
        try:
            return validate(plugin_id)
        except Exception:
            return None

    @staticmethod
    def _clock(value: datetime) -> str:
        try:
            local = value.astimezone()
            return local.strftime("%H:%M:%S")
        except Exception:
            return "—"

    def get_state(self) -> dict[str, Any]:
        return {
            "search": self._search_value(),
            "status_filter": str(self.status_filter.value or "all"),
            "selected_plugin_id": self._selected_plugin_id,
        }

    def restore_state(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        self._restoring = True
        try:
            self.search.value = str(state.get("search", "") or "")
            status_filter = str(state.get("status_filter", "all") or "all")
            if status_filter in {"all", "enabled", "available", "issues"}:
                self.status_filter.value = status_filter
            selected = state.get("selected_plugin_id")
            self._selected_plugin_id = str(selected) if selected else None
        finally:
            self._restoring = False
        self._apply_filters()

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True

        events = getattr(self.context, "events", None)
        if events is not None:
            for subscription in list(self._subscriptions):
                try:
                    events.unsubscribe(subscription)
                except Exception:
                    pass
        self._subscriptions.clear()

        for widget, watcher in list(self._watchers):
            try:
                widget.param.unwatch(watcher)
            except Exception:
                pass
        self._watchers.clear()

