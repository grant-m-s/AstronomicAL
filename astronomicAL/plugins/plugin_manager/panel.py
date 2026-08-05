from __future__ import annotations

import html
import json
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterator, Mapping

import panel as pn

from .diagnostics import (
    PluginManagerSnapshot,
    PluginSnapshot,
    collect_snapshot,
    filter_plugins,
    provides_summary,
    source_label,
)
from .marketplace_panel import PluginMarketplacePanel
from .styles import PLUGIN_MANAGER_CSS


class PluginManagerPanel:
    """User-facing management surface for AstronomicAL plugins."""

    state_version = 2
    SELF_PLUGIN_ID = "core.plugin_manager"

    def __init__(self, context: Any):
        self.context = context
        self.manager = getattr(context, "plugins", None)
        self.activation = getattr(context, "plugin_activation", None)
        self.installer = getattr(context, "plugin_installer", None)
        self.installed_store = getattr(context, "installed_plugins", None)

        self._disposed = False
        self._restoring = False
        self._refresh_scheduled = False
        self._syncing_community_toggle = False
        self._subscriptions: list[Any] = []
        self._watchers: list[tuple[Any, Any]] = []
        self._row_buttons: list[pn.widgets.Button] = []
        self._row_uninstall_buttons: dict[str, pn.widgets.Button] = {}
        self._snapshot: PluginManagerSnapshot | None = None
        self._visible_plugins: list[PluginSnapshot] = []
        self._selected_plugin_id: str | None = None
        self._expanded_plugin_id: str | None = None
        self._focus_plugin_id: str | None = None
        self._lifecycle_message_plugin_id: str | None = None
        self._package_message_plugin_id: str | None = None
        self._pending_uninstall_plugin_id: str | None = None
        self._busy = False

        self.search = pn.widgets.TextInput(
            name="Search installed plugins",
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
        self.community_toggle = pn.widgets.Switch(
            name="",
            value=False,
            width=46,
            margin=(1, 0, 0, 8),
        )

        self.enable_button = pn.widgets.Button(
            name="Enable plugin",
            button_type="primary",
            sizing_mode="stretch_width",
            height=34,
            margin=0,
            css_classes=["al-pm-detail-action", "al-pm-detail-action-primary"],
            stylesheets=[PLUGIN_MANAGER_CSS],
        )
        self.disable_button = pn.widgets.Button(
            name="Disable plugin",
            button_type="default",
            sizing_mode="stretch_width",
            height=34,
            margin=0,
            css_classes=["al-pm-detail-action", "al-pm-detail-action-secondary"],
            stylesheets=[PLUGIN_MANAGER_CSS],
        )
        self.reload_button = pn.widgets.Button(
            name="Reload development plugin", button_type="default",
            sizing_mode="stretch_width", height=34, margin=0, visible=False,
        )

        self.install_file = pn.widgets.FileInput(
            name="", accept=".alplugin", multiple=False,
            sizing_mode="stretch_width", height=38, margin=(0, 0, 12, 0),
        )
        self.install_button = pn.widgets.Button(
            name="Install", button_type="primary",
            sizing_mode="stretch_width", height=34, margin=(4, 0, 4, 0),
        )
        self.install_file_toggle = pn.widgets.Button(
            name="＋ Install from file",
            button_type="default",
            width=150,
            height=30,
            margin=0,
            css_classes=["al-pm-inline-action"],
            stylesheets=[PLUGIN_MANAGER_CSS],
        )
        self.details_back_button = pn.widgets.Button(
            name="← Back to plugins",
            button_type="default",
            width=132,
            height=30,
            margin=(0, 0, 8, 0),
            css_classes=["al-pm-inline-action", "al-pm-back-action"],
            stylesheets=[PLUGIN_MANAGER_CSS],
        )
        self.update_file = pn.widgets.FileInput(
            name="", accept=".alplugin", multiple=False,
            sizing_mode="stretch_width", height=38, margin=(0, 0, 12, 0),
        )
        self.update_button = pn.widgets.Button(
            name="Update selected plugin", button_type="primary",
            sizing_mode="stretch_width", height=34, margin=0,
        )
        self.uninstall_button = pn.widgets.Button(
            name="Uninstall plugin",
            button_type="default",
            sizing_mode="stretch_width",
            height=34,
            margin=(2, 0, 4, 0),
            css_classes=["al-pm-detail-action", "al-pm-detail-action-danger"],
            stylesheets=[PLUGIN_MANAGER_CSS],
        )

        self.header = self._html_pane()
        self.summary = self._html_pane()
        self.community_status = self._html_pane()
        self.operation_banner = self._html_pane()
        self.lifecycle_banner = self._html_pane()
        self.install_status = self._html_pane()
        self.install_banner = self._html_pane()
        self.package_details = self._html_pane()
        self.package_banner = self._html_pane()
        self.uninstall_banner = self._html_pane()
        self.uninstall_note = self._html_pane()
        self.selected_details = self._html_pane()
        self.action_note = self._html_pane()
        self.discovery_issues = self._html_pane()

        self.community_status.margin = (4, 0, 0, 0)
        self.install_status.min_height = 46
        self.install_status.margin = (4, 0, 10, 0)
        self.install_banner.margin = (8, 0, 10, 0)
        self.package_details.margin = (4, 0, 10, 0)
        self.package_banner.margin = (8, 0, 10, 0)
        self.uninstall_banner.margin = (6, 0, 8, 0)

        self.core_list = pn.Column(
            sizing_mode="stretch_width", margin=0,
            css_classes=["al-pm-plugin-list"], stylesheets=[PLUGIN_MANAGER_CSS],
        )
        self.community_list = pn.Column(
            sizing_mode="stretch_width", margin=0,
            css_classes=["al-pm-plugin-list"], stylesheets=[PLUGIN_MANAGER_CSS],
        )
        self.other_list = pn.Column(
            sizing_mode="stretch_width", margin=0,
            css_classes=["al-pm-plugin-list"], stylesheets=[PLUGIN_MANAGER_CSS],
        )

        self.refresh_button.on_click(self._refresh_clicked)
        self.discover_button.on_click(self._discover_clicked)
        self.enable_button.on_click(self._enable_clicked)
        self.disable_button.on_click(self._disable_clicked)
        self.reload_button.on_click(self._reload_clicked)
        self.install_button.on_click(self._install_clicked)
        self.install_file_toggle.on_click(self._toggle_install_from_file)
        self.details_back_button.on_click(self._close_plugin_details)
        self.update_button.on_click(self._update_clicked)
        self.uninstall_button.on_click(self._uninstall_clicked)

        self._watch(self.search, self._filters_changed, "value")
        self._watch(self.search, self._filters_changed, "value_input")
        self._watch(self.status_filter, self._filters_changed, "value")
        self._watch(self.community_toggle, self._community_toggle_changed, "value")
        self._watch(self.install_file, self._package_file_changed, "value")
        self._watch(self.update_file, self._package_file_changed, "value")
        self._subscribe_to_plugin_events()

        self.discovery_section = self._section(
            "Discovery issues",
            "Plugins that could not be inspected during the latest scan",
            self.discovery_issues,
        )
        self.discovery_section.visible = False

        self.marketplace_panel = PluginMarketplacePanel(
            context,
            on_changed=self.refresh,
            on_manage=self._marketplace_manage_plugin,
        )

        self.view = self._build_view()
        self.refresh()

    @staticmethod
    def _html_pane() -> pn.pane.HTML:
        return pn.pane.HTML(
            "", sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS], margin=0,
        )

    def _build_view(self) -> pn.Column:
        installed_controls = pn.Column(
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
            styles=self._card_styles(),
            margin=(0, 0, 10, 0),
        )

        community_heading = pn.pane.HTML(
            '<div class="al-pm-section-title al-pm-inline-heading">'
            '<div><h3>Community plugins</h3>'
            '<span>Allow installed third-party plugins to execute in AstronomicAL</span></div>'
            '</div>',
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=0,
        )
        community_gate = pn.Column(
            pn.Row(
                community_heading,
                self.community_toggle,
                sizing_mode="stretch_width",
                height=48,
                margin=0,
                styles={"align-items": "center"},
            ),
            self.community_status,
            sizing_mode="stretch_width",
            css_classes=["al-pm-community-gate"],
            styles={"flex": "0 0 auto", "height": "auto"},
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=(0, 0, 12, 0),
        )

        install_file_label = pn.pane.HTML(
            '<div style="font-size:12px; font-weight:600; line-height:1.35;">'
            'Plugin package (.alplugin)</div>',
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            height=22,
            margin=(2, 0, 4, 0),
        )
        self.install_file_panel = pn.Column(
            pn.pane.HTML(
                '<div class="al-pm-file-install-title">Install a local plugin package</div>'
                '<div class="al-pm-file-install-copy">Use this for development or a '
                'downloaded <code>.alplugin</code> file. Marketplace installation is recommended '
                'for normal use.</div>',
                sizing_mode="stretch_width",
                stylesheets=[PLUGIN_MANAGER_CSS],
                margin=(0, 0, 8, 0),
            ),
            self.install_status,
            install_file_label,
            self.install_file,
            self.install_banner,
            self.install_button,
            sizing_mode="stretch_width",
            css_classes=["al-pm-file-install-panel"],
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=(6, 0, 0, 0),
            visible=False,
        )
        self.install_file_controls = pn.Column(
            self.install_file_toggle,
            self.install_file_panel,
            sizing_mode="stretch_width",
            css_classes=["al-pm-file-install-controls"],
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=(4, 0, 10, 0),
            styles={"flex": "0 0 auto", "height": "auto", "min-height": "0"},
        )

        details_actions = pn.Column(
            pn.Row(
                self.enable_button,
                self.disable_button,
                sizing_mode="stretch_width",
                css_classes=["al-pm-detail-lifecycle-actions"],
                stylesheets=[PLUGIN_MANAGER_CSS],
                margin=0,
            ),
            self.reload_button,
            sizing_mode="stretch_width",
            margin=(8, 0, 0, 0),
        )
        self.managed_uninstall_controls = pn.Column(
            self.uninstall_note,
            self.uninstall_banner,
            self.uninstall_button,
            sizing_mode="stretch_width",
            css_classes=["al-pm-uninstall-controls"],
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=(12, 0, 4, 0),
            visible=False,
        )
        update_file_label = pn.pane.HTML(
            '<div style="font-size:12px; font-weight:600; line-height:1.35;">'
            'Update package (.alplugin)</div>',
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            height=22,
            margin=(4, 0, 4, 0),
        )
        self.package_management = pn.Column(
            pn.pane.HTML(
                '<div class="al-pm-section-title"><h3>Package management</h3>'
                '<span>Update or remove plugins installed by AstronomicAL</span></div>',
                sizing_mode="stretch_width",
                stylesheets=[PLUGIN_MANAGER_CSS],
                margin=(16, 0, 8, 0),
            ),
            self.package_details,
            update_file_label,
            self.update_file,
            self.package_banner,
            self.update_button,
            sizing_mode="stretch_width",
            margin=(4, 0, 4, 0),
            visible=False,
        )
        self.installed_details_view = pn.Column(
            self.details_back_button,
            self.selected_details,
            self.action_note,
            self.lifecycle_banner,
            details_actions,
            self.managed_uninstall_controls,
            self.package_management,
            sizing_mode="stretch_width",
            css_classes=["al-pm-detail-view"],
            stylesheets=[PLUGIN_MANAGER_CSS],
            styles={"flex": "0 0 auto", "height": "auto", "min-height": "0"},
            margin=(8, 0, 0, 0),
            visible=False,
        )

        self.other_heading = self._list_heading(
            "Development and local plugins",
            "Development and runtime registrations",
            css_class="al-pm-other-heading",
        )

        core_tab = pn.Column(
            self._list_heading("Core plugins", "Plugins bundled with AstronomicAL"),
            self.core_list,
            sizing_mode="stretch_width",
            margin=(8, 0, 0, 0),
        )
        community_tab = pn.Column(
            community_gate,
            self.install_file_controls,
            self._list_heading("Installed community plugins", "Marketplace, package, and manually installed third-party plugins"),
            self.community_list,
            self.other_heading,
            self.other_list,
            sizing_mode="stretch_width",
            styles={"height": "auto", "min-height": "0", "flex": "0 0 auto"},
            margin=(8, 0, 0, 0),
        )

        self.installed_tabs = pn.Tabs(
            ("Core plugins", core_tab),
            ("Community plugins", community_tab),
            active=0,
            dynamic=True,
            sizing_mode="stretch_width",
            styles={"height": "auto", "min-height": "0", "flex": "0 0 auto"},
            margin=0,
        )
        installed_view = pn.Column(
            installed_controls,
            self.installed_tabs,
            self.installed_details_view,
            self.discovery_section,
            sizing_mode="stretch_width",
            margin=0,
        )
        self.main_tabs = pn.Tabs(
            ("Installed", installed_view),
            ("Marketplace", self.marketplace_panel.view),
            active=0,
            dynamic=True,
            sizing_mode="stretch_width",
            styles={"height": "auto", "min-height": "0", "flex": "0 0 auto"},
            margin=0,
        )

        return pn.Column(
            self.header,
            self.operation_banner,
            self.summary,
            self.main_tabs,
            sizing_mode="stretch_both",
            min_width=220,
            scroll=True,
            css_classes=["al-plugin-manager"],
            styles={
                "background": "transparent",
                "box-sizing": "border-box",
                "overflow-x": "hidden",
                "overflow-y": "auto",
                "overflow-anchor": "none",
                "padding": "8px",
            },
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=0,
        )

    @staticmethod
    def _list_heading(title: str, detail: str, *, css_class: str = "") -> pn.pane.HTML:
        extra = f" {css_class}" if css_class else ""
        return pn.pane.HTML(
            f'<div class="al-pm-list-heading{extra}"><h3>{html.escape(title)}</h3>'
            f'<span>{html.escape(detail)}</span></div>',
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=(4, 0, 6, 0),
        )

    @staticmethod
    def _card_styles() -> dict[str, str]:
        return {
            "background": "#ffffff",
            "border": "1px solid #dfe5ee",
            "border-radius": "9px",
            "box-shadow": "0 2px 8px rgba(27, 43, 65, 0.045)",
            "padding": "12px",
        }

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
            styles=PluginManagerPanel._card_styles(),
            margin=(0, 0, 10, 0),
        )

    # ------------------------------------------------------------------
    # Snapshot, filtering, and compact plugin rows
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
            self.community_status.object = self._banner_html(
                "danger", "Community plugin state is unavailable."
            )
            self.install_status.object = self._banner_html(
                "danger", "Plugin installation state is unavailable."
            )
            self.selected_details.object = self._empty_html("No plugin can be selected.")
            self.discovery_issues.object = self._empty_html("No discovery information.")
            self.discovery_section.visible = False
            self.core_list.objects = [self._empty_pane("Core plugins are unavailable.")]
            self.community_list.objects = [self._empty_pane("Community plugins are unavailable.")]
            self.other_list.objects = []
            self._set_operation_message("danger", f"Unable to read plugin state: {exc}")
            self._update_action_state(None)
            self._update_package_action_state(None)
            self._update_install_action_state()
            self.community_toggle.disabled = True
            return

        self._snapshot = snapshot
        self.header.object = self._header_html(snapshot)
        self.summary.object = self._summary_html(snapshot)
        self.discovery_issues.object = self._discovery_issues_html(snapshot)
        self.discovery_section.visible = bool(snapshot.discovery_issues)
        self._sync_community_controls(snapshot)
        self._sync_install_controls(snapshot)
        self._apply_filters()
        self.marketplace_panel.refresh()

    def _apply_filters(self, *, reset_page: bool = False) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return

        self._visible_plugins = filter_plugins(
            snapshot.plugins,
            query=self._search_value(),
            status_filter=str(self.status_filter.value or "all"),
        )

        visible_ids = {plugin.id for plugin in self._visible_plugins}
        if self._selected_plugin_id not in visible_ids:
            self._selected_plugin_id = None
            self._expanded_plugin_id = None
        if self._focus_plugin_id not in visible_ids:
            self._focus_plugin_id = None

        self._render_plugin_lists()
        self._render_selected_plugin()
        self._sync_installed_detail_visibility()

    def _render_plugin_lists(self) -> None:
        if self._snapshot is None:
            return

        # Dynamic detail panels contain live widgets. Detach the old row tree before
        # constructing the replacement so one widget model is never temporarily
        # mounted in two plugin rows during a refresh.
        self.core_list.objects = []
        self.community_list.objects = []
        self.other_list.objects = []

        core = [plugin for plugin in self._visible_plugins if plugin.is_bundled]
        community = [plugin for plugin in self._visible_plugins if plugin.is_community]
        other = [
            plugin for plugin in self._visible_plugins
            if not plugin.is_bundled and not plugin.is_community
        ]

        core = self._promote_focus(core)
        community = self._promote_focus(community)
        other = self._promote_focus(other)

        self._row_buttons = []
        self._row_uninstall_buttons = {}
        self.core_list.objects = self._plugin_group_objects(
            core, empty_message="No core plugins match this view."
        )
        self.community_list.objects = self._plugin_group_objects(
            community, empty_message="No installed community plugins match this view."
        )
        self.other_list.objects = self._plugin_group_objects(other, empty_message="")

        self.other_heading.visible = bool(other)
        self.other_list.visible = bool(other)

    def _promote_focus(self, plugins: list[PluginSnapshot]) -> list[PluginSnapshot]:
        focus = self._focus_plugin_id
        if not focus:
            return plugins
        return sorted(plugins, key=lambda plugin: (plugin.id != focus, plugin.name.lower(), plugin.id.lower()))

    def _plugin_group_objects(
        self,
        plugins: list[PluginSnapshot],
        *,
        empty_message: str,
    ) -> list[Any]:
        if not plugins:
            return [self._empty_pane(empty_message)] if empty_message else []

        objects: list[Any] = []
        for plugin in plugins:
            objects.append(
                pn.Column(
                    self._plugin_row(plugin),
                    sizing_mode="stretch_width",
                    css_classes=["al-pm-plugin-item"],
                    stylesheets=[PLUGIN_MANAGER_CSS],
                    styles={
                        "flex": "0 0 auto",
                        "position": "relative",
                        "height": "auto",
                        "min-height": "0",
                    },
                    margin=(0, 0, 7, 0),
                )
            )
        return objects

    def _plugin_row(self, plugin: PluginSnapshot) -> pn.Row:
        status_label, status_tone = self._display_status(plugin)
        update_version = self._marketplace_update_version(plugin.id)
        source = source_label(plugin.origin, plugin.source)
        if plugin.installed_source == "marketplace":
            source = "Marketplace"
        update_html = (
            f'<span class="al-pm-row-update">Update {html.escape(update_version)}</span>'
            if update_version else ""
        )
        description = plugin.description or "No description was provided."
        info_html = (
            '<div class="al-pm-row-info">'
            '<div class="al-pm-row-title-line">'
            f'<strong class="al-pm-row-name">{html.escape(plugin.name)}</strong>'
            f'<span class="al-pm-status-pill {html.escape(status_tone)}">{html.escape(status_label)}</span>'
            '</div>'
            f'<div class="al-pm-row-meta">Version {html.escape(plugin.version)} · {html.escape(source)} {update_html}</div>'
            f'<div class="al-pm-row-description">{html.escape(description)}</div>'
            f'<div class="al-pm-row-provides">{html.escape(provides_summary(plugin))}</div>'
            '</div>'
        )
        info = pn.pane.HTML(
            info_html,
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=0,
        )

        details_button = pn.widgets.Button(
            name="Details",
            button_type="default",
            width=74,
            height=30,
            margin=0,
            css_classes=["al-pm-row-action", "al-pm-row-action-quiet"],
            stylesheets=[PLUGIN_MANAGER_CSS],
        )
        details_button.on_click(lambda _event, plugin_id=plugin.id: self._toggle_plugin_details(plugin_id))
        self._row_buttons.append(details_button)

        lifecycle = self._row_lifecycle_button(plugin)
        buttons = [lifecycle, details_button] if lifecycle is not None else [details_button]
        primary_actions = pn.Row(
            *buttons,
            width=156 if len(buttons) > 1 else 76,
            height=30,
            margin=0,
            css_classes=["al-pm-row-actions"],
            stylesheets=[PLUGIN_MANAGER_CSS],
        )

        if plugin.is_community:
            uninstall_button = self._row_uninstall_button(plugin)
            actions = pn.Column(
                primary_actions,
                uninstall_button,
                width=156,
                margin=0,
                css_classes=["al-pm-row-actions-stack"],
                stylesheets=[PLUGIN_MANAGER_CSS],
                styles={"flex": "0 0 auto", "height": "auto", "min-height": "0"},
            )
        else:
            actions = primary_actions

        return pn.Row(
            info,
            actions,
            sizing_mode="stretch_width",
            css_classes=["al-pm-list-row"],
            stylesheets=[PLUGIN_MANAGER_CSS],
            styles={
                "flex": "0 0 auto",
                "position": "relative",
                "height": "auto",
                "min-height": "0",
                "align-items": "flex-start",
            },
            margin=0,
        )

    def _row_lifecycle_button(self, plugin: PluginSnapshot) -> pn.widgets.Button | None:
        if plugin.is_runtime or plugin.origin == "unknown":
            return None

        is_self = plugin.id == self.SELF_PLUGIN_ID
        if plugin.status == "enabled":
            button = pn.widgets.Button(
                name="Disable",
                button_type="default",
                width=74,
                height=30,
                margin=0,
                css_classes=["al-pm-row-action", "al-pm-row-action-secondary"],
                stylesheets=[PLUGIN_MANAGER_CSS],
            )
            button.disabled = self._busy or self.activation is None or is_self
            if not is_self:
                button.on_click(lambda _event, plugin_id=plugin.id: self._row_disable(plugin_id))
        else:
            button = pn.widgets.Button(
                name="Enable",
                button_type="primary",
                width=74,
                height=30,
                margin=0,
                css_classes=["al-pm-row-action", "al-pm-row-action-primary"],
                stylesheets=[PLUGIN_MANAGER_CSS],
            )
            community_allowed = bool(
                self._snapshot is not None and self._snapshot.community_plugins_enabled
            )
            button.disabled = (
                self._busy
                or self.activation is None
                or (plugin.is_community and not community_allowed)
            )
            button.on_click(lambda _event, plugin_id=plugin.id: self._row_enable(plugin_id))
        self._row_buttons.append(button)
        return button

    def _row_uninstall_button(self, plugin: PluginSnapshot) -> pn.widgets.Button:
        managed = bool(plugin.is_managed and plugin.origin == "user")
        snapshot = self._snapshot
        store_error = bool(snapshot is not None and snapshot.installed_store_error)
        unavailable = (
            self._busy
            or self.installer is None
            or self.installed_store is None
            or store_error
        )
        confirming = self._pending_uninstall_plugin_id == plugin.id
        button = pn.widgets.Button(
            name="Confirm uninstall" if confirming else "Uninstall",
            button_type="default",
            width=156,
            height=28,
            margin=(5, 0, 0, 0),
            disabled=(not managed) or unavailable,
            css_classes=["al-pm-row-action", "al-pm-row-action-danger", "al-pm-row-uninstall"],
            stylesheets=[PLUGIN_MANAGER_CSS],
        )
        if managed:
            button.on_click(
                lambda _event, plugin_id=plugin.id: self._row_uninstall(plugin_id)
            )
        self._row_buttons.append(button)
        self._row_uninstall_buttons[plugin.id] = button
        return button

    def _row_uninstall(self, plugin_id: str) -> None:
        if self._disposed or self._busy:
            return

        plugin = next(
            (
                item
                for item in (self._snapshot.plugins if self._snapshot else ())
                if item.id == plugin_id
            ),
            None,
        )
        if plugin is None:
            return

        self._selected_plugin_id = plugin.id
        if not plugin.is_managed or plugin.origin != "user":
            self._set_operation_message(
                "info",
                f"{plugin.name} was not installed by AstronomicAL, so its files cannot "
                "be uninstalled safely here.",
            )
            return

        was_pending = self._pending_uninstall_plugin_id == plugin.id
        self._uninstall_clicked(message_target="row")

        # First click is confirmation-only. Update the existing button model in
        # place instead of rebuilding the plugin list; rebuilding here destroys
        # and recreates the row DOM and makes the scroll container jump.
        if not was_pending and self._pending_uninstall_plugin_id == plugin.id:
            button = self._row_uninstall_buttons.get(plugin.id)
            if button is not None:
                button.name = "Confirm uninstall"

    def _toggle_plugin_details(self, plugin_id: str) -> None:
        """Open a dedicated installed-plugin detail view.

        Details deliberately live outside the plugin list. Moving a shared tree of
        live Panel widgets in and out of dynamically rebuilt rows caused stale
        browser layout heights and overlapping models. A dedicated detail route is
        both simpler and closer to the narrow-panel equivalent of Obsidian's detail
        page.
        """

        if self._disposed or self._busy:
            return

        plugin_id = str(plugin_id or "").strip()
        if not plugin_id:
            return

        plugin = next(
            (item for item in (self._snapshot.plugins if self._snapshot else ()) if item.id == plugin_id),
            None,
        )
        if plugin is None:
            return

        self._selected_plugin_id = plugin_id
        self._expanded_plugin_id = plugin_id
        self._focus_plugin_id = None
        self.installed_tabs.active = 1 if plugin.is_community else 0
        self._render_selected_plugin()
        self._sync_installed_detail_visibility()

    def _close_plugin_details(self, _event: Any = None) -> None:
        if self._disposed:
            return
        self._expanded_plugin_id = None
        self._sync_installed_detail_visibility()

    def _sync_installed_detail_visibility(self) -> None:
        detail_plugin = self._selected_plugin()
        show_details = bool(
            self._expanded_plugin_id
            and detail_plugin is not None
            and detail_plugin.id == self._expanded_plugin_id
        )
        self.installed_details_view.visible = show_details
        self.installed_tabs.visible = not show_details

    def _row_enable(self, plugin_id: str) -> None:
        self._selected_plugin_id = plugin_id
        self._enable_clicked(message_target="global")

    def _row_disable(self, plugin_id: str) -> None:
        self._selected_plugin_id = plugin_id
        self._disable_clicked(message_target="global")

    def _toggle_install_from_file(self, _event: Any = None) -> None:
        if self._disposed or self._busy:
            return
        visible = not bool(self.install_file_panel.visible)
        self.install_file_panel.visible = visible
        self.install_file_toggle.name = (
            "− Hide file installer" if visible else "＋ Install from file"
        )

    def _marketplace_manage_plugin(self, plugin_id: str) -> None:
        if self._disposed:
            return

        plugin_id = str(plugin_id or "").strip()
        if not plugin_id:
            return

        self.main_tabs.active = 0
        self.installed_tabs.active = 1
        self.search.value = ""
        try:
            self.search.value_input = ""
        except Exception:
            pass
        self.status_filter.value = "all"
        self._selected_plugin_id = plugin_id
        self._expanded_plugin_id = plugin_id
        self._focus_plugin_id = plugin_id
        self.refresh()
        self._sync_installed_detail_visibility()

    def _marketplace_update_version(self, plugin_id: str) -> str | None:
        updates = getattr(self.context, "marketplace_updates", None)
        list_updates = getattr(updates, "list_updates", None)
        if not callable(list_updates):
            return None
        try:
            infos = list(list_updates() or [])
        except Exception:
            return None
        for info in infos:
            if str(getattr(info, "plugin_id", "") or "") != plugin_id:
                continue
            target = str(getattr(info, "target_version", "") or "").strip()
            return target or None
        return None

    def _filters_changed(self, _event: Any = None) -> None:
        if self._disposed or self._restoring:
            return
        self._focus_plugin_id = None
        self._apply_filters()

    def _search_value(self) -> str:
        value_input = getattr(self.search, "value_input", None)
        if value_input is not None:
            return str(value_input or "")
        return str(self.search.value or "")

    @staticmethod
    def _empty_pane(message: str) -> pn.pane.HTML:
        return pn.pane.HTML(
            f'<div class="al-pm-empty">{html.escape(message)}</div>',
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=0,
        )

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
                health_label = (
                    f"{snapshot.issue_count} issue"
                    f"{'s' if snapshot.issue_count != 1 else ''}"
                )
                health_tone = "warning"
            else:
                health_label = "Healthy"
                health_tone = ""

        return f"""
        <div class="al-pm-header">
          <div class="al-pm-header-main">
            <div class="al-pm-eyebrow">Platform management</div>
            <h2 class="al-pm-title">Plugin Manager</h2>
            <div class="al-pm-subtitle">Manage bundled and community plugins, install from the marketplace, and control what is allowed to run.</div>
          </div>
          <div class="al-pm-header-meta">
            <div class="al-pm-health {health_tone}"><span class="al-pm-health-dot"></span>{html.escape(health_label)}</div>
            <div class="al-pm-updated">{total:,} plugins · updated {html.escape(refreshed)}</div>
          </div>
        </div>
        """

    def _summary_html(self, snapshot: PluginManagerSnapshot) -> str:
        community_installed = sum(plugin.is_community for plugin in snapshot.plugins)
        return (
            '<div class="al-pm-compact-summary">'
            f'<span><strong>{snapshot.enabled_count}</strong> enabled</span>'
            f'<span><strong>{snapshot.available_count}</strong> available</span>'
            f'<span><strong>{community_installed}</strong> community</span>'
            f'<span class="{"danger" if snapshot.issue_count else "success"}">'
            f'<strong>{snapshot.issue_count}</strong> issue'
            f'{"s" if snapshot.issue_count != 1 else ""}</span>'
            '</div>'
        )

    def _sync_community_controls(self, snapshot: PluginManagerSnapshot) -> None:
        self._syncing_community_toggle = True
        try:
            self.community_toggle.value = bool(snapshot.community_plugins_enabled)
        finally:
            self._syncing_community_toggle = False

        self.community_toggle.disabled = (
            self.activation is None or bool(snapshot.plugin_state_error)
        )
        self.community_status.object = self._community_status_html(snapshot)

    def _community_status_html(self, snapshot: PluginManagerSnapshot) -> str:
        if snapshot.plugin_state_error:
            return self._banner_html(
                "danger",
                "Plugin state could not be loaded. Community plugins remain disabled "
                f"until the state file is repaired: {snapshot.plugin_state_error}",
            )

        if self.activation is None:
            return self._banner_html(
                "danger",
                "PluginActivationService is not available on AppContext.",
            )

        if snapshot.community_plugins_enabled:
            return self._banner_html(
                "success",
                "Community plugins are enabled. Third-party plugins configured as "
                "enabled may execute in this AstronomicAL session.",
            )

        configured = snapshot.configured_community_count
        if configured:
            return self._banner_html(
                "warning",
                f"Community plugins are disabled. {configured} plugin"
                f"{'s are' if configured != 1 else ' is'} configured to run but "
                "blocked by the global community-plugin setting.",
            )

        return self._banner_html(
            "info",
            "Community plugins are disabled. Installed community plugins can still "
            "be discovered from their static manifests, but their Python code will not run.",
        )

    def _sync_install_controls(self, snapshot: PluginManagerSnapshot) -> None:
        if self.installer is None or self.installed_store is None:
            self.install_status.object = self._banner_html(
                "danger",
                "PluginInstaller / InstalledPluginStore is not available on AppContext.",
            )
        elif snapshot.installed_store_error:
            self.install_status.object = self._banner_html(
                "danger",
                "Installed plugin database could not be loaded. Package changes are "
                f"blocked until it is repaired: {snapshot.installed_store_error}",
            )
        else:
            self.install_status.object = self._banner_html(
                "info",
                "Choose an .alplugin file. AstronomicAL validates and installs it "
                "without importing or enabling the plugin.",
            )
        self._update_install_action_state()

    def _render_selected_plugin(self) -> None:
        plugin = self._selected_plugin()
        if (
            self._pending_uninstall_plugin_id is not None
            and (plugin is None or plugin.id != self._pending_uninstall_plugin_id)
        ):
            self._pending_uninstall_plugin_id = None

        self._update_action_state(plugin)
        self._update_package_action_state(plugin)

        if (
            self._lifecycle_message_plugin_id is not None
            and (plugin is None or plugin.id != self._lifecycle_message_plugin_id)
        ):
            self._clear_lifecycle_message()

        if (
            self._package_message_plugin_id is not None
            and (plugin is None or plugin.id != self._package_message_plugin_id)
        ):
            self._clear_package_message()

        if (
            self._pending_uninstall_plugin_id is None
            or plugin is None
            or plugin.id != self._pending_uninstall_plugin_id
        ):
            self.uninstall_banner.object = ""

        if plugin is None:
            self.selected_details.object = self._empty_html(
                "No plugin matches the current search and filter."
            )
            self.action_note.object = ""
            self.package_details.object = ""
            self._pending_uninstall_plugin_id = None
            return

        validation = self._validation_result(plugin.id)
        self.selected_details.object = self._plugin_details_html(plugin, validation)
        self.action_note.object = self._action_note_html(plugin)
        self.package_details.object = self._package_details_html(plugin)

    def _plugin_details_html(self, plugin: PluginSnapshot, validation: Any) -> str:
        description = plugin.description or "No description was provided for this plugin."
        panels = self._contribution_chips("Panel", plugin.panels)
        actions = self._contribution_chips("Action", plugin.actions)
        workflows = self._contribution_chips("Workflow", plugin.workflows)
        open_instances = self._open_instances_html(plugin)
        readiness = self._readiness_html(plugin, validation)
        activation_policy = self._activation_policy_html(plugin)

        registration_note = ""
        if plugin.status != "enabled" and plugin.contribution_count == 0:
            registration_note = self._banner_html(
                "info",
                "This plugin registers its panels and actions when it is enabled.",
            )

        configured_card = ""
        if plugin.is_community:
            configured_card = (
                '<div class="al-pm-mini">'
                '<div class="al-pm-mini-label">Configured</div>'
                '<div class="al-pm-mini-value">'
                f"{'Enabled' if plugin.configured_enabled else 'Disabled'}"
                "</div></div>"
            )

        technical = self._technical_html(plugin)
        status_label, status_tone = self._display_status(plugin)

        return f"""
        <div class="al-pm-plugin-head">
          <div>
            <h3 class="al-pm-plugin-name">{html.escape(plugin.name)}</h3>
            <div class="al-pm-plugin-version">Version {html.escape(plugin.version)}</div>
          </div>
          <span class="al-pm-status-pill {status_tone}">{html.escape(status_label)}</span>
        </div>
        <p class="al-pm-description">{html.escape(description)}</p>
        <div class="al-pm-mini-grid">
          <div class="al-pm-mini"><div class="al-pm-mini-label">Source</div><div class="al-pm-mini-value">{html.escape(source_label(plugin.origin, plugin.source))}</div></div>
          <div class="al-pm-mini"><div class="al-pm-mini-label">Open panels</div><div class="al-pm-mini-value">{len(plugin.open_instances):,}</div></div>
          <div class="al-pm-mini"><div class="al-pm-mini-label">Panels</div><div class="al-pm-mini-value">{len(plugin.panels):,}</div></div>
          <div class="al-pm-mini"><div class="al-pm-mini-label">Actions</div><div class="al-pm-mini-value">{len(plugin.actions):,}</div></div>
          {configured_card}
        </div>
        {readiness}
        {activation_policy}
        {registration_note}
        {self._optional_group("Panels added to the workspace", panels)}
        {self._optional_group("Actions and tools", actions)}
        {self._optional_group("Workflows", workflows)}
        {self._optional_group("Currently open", open_instances)}
        {technical}
        """

    def _display_status(self, plugin: PluginSnapshot) -> tuple[str, str]:
        snapshot = self._snapshot
        if (
            plugin.is_community
            and plugin.status != "enabled"
            and plugin.configured_enabled is True
            and snapshot is not None
            and not snapshot.community_plugins_enabled
        ):
            return "Blocked", "muted"
        return plugin.status_label, plugin.status_tone

    def _readiness_html(self, plugin: PluginSnapshot, validation: Any) -> str:
        errors = (
            list(getattr(validation, "errors", None) or [])
            if validation is not None
            else []
        )
        warnings = (
            list(getattr(validation, "warnings", None) or [])
            if validation is not None
            else []
        )

        if plugin.error:
            errors.insert(0, plugin.error)

        if not errors and not warnings:
            if plugin.status == "enabled":
                return (
                    '<div class="al-pm-banner success al-pm-readiness">'
                    "Plugin is enabled and ready.</div>"
                )
            return (
                '<div class="al-pm-banner info al-pm-readiness">'
                "No dependency problems were found.</div>"
            )

        chunks: list[str] = []
        if errors:
            items = "".join(f"<li>{html.escape(str(item))}</li>" for item in errors)
            chunks.append(
                '<div class="al-pm-banner danger al-pm-readiness">'
                "<strong>Cannot start cleanly</strong>"
                f"<ul>{items}</ul></div>"
            )
        if warnings:
            items = "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings)
            chunks.append(
                '<div class="al-pm-banner warning al-pm-readiness">'
                "<strong>Optional items unavailable</strong>"
                f"<ul>{items}</ul></div>"
            )
        return "".join(chunks)

    def _activation_policy_html(self, plugin: PluginSnapshot) -> str:
        snapshot = self._snapshot
        if snapshot is None:
            return ""

        if plugin.is_community:
            if not snapshot.community_plugins_enabled:
                if plugin.configured_enabled:
                    return self._banner_html(
                        "warning",
                        "This community plugin is configured as enabled, but it is "
                        "blocked while Community plugins are turned off.",
                    )
                return self._banner_html(
                    "info",
                    "Community plugins are turned off. Enable the global community "
                    "plugin setting before enabling this plugin.",
                )

            if plugin.configured_enabled and plugin.status != "enabled":
                return self._banner_html(
                    "warning",
                    "This plugin is configured as enabled but is not currently running. "
                    "Retry enabling it and review any validation errors above.",
                )

            if not plugin.configured_enabled and plugin.status != "enabled":
                return self._banner_html(
                    "info",
                    "This community plugin is installed and available, but disabled.",
                )

        if plugin.is_bundled and plugin.status != "enabled":
            return self._banner_html(
                "info",
                "Bundled plugins are part of AstronomicAL and are enabled automatically "
                "on application startup.",
            )

        return ""

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
        settings = (
            json.dumps(dict(plugin.settings), indent=2, default=str)
            if plugin.settings
            else "None"
        )
        configured = (
            "Enabled" if plugin.configured_enabled else "Disabled"
            if plugin.configured_enabled is not None
            else "Not applicable"
        )

        rows = [
            ("Plugin ID", plugin.id),
            ("Origin", plugin.origin or "unknown"),
            (
                "Package management",
                "AstronomicAL-managed" if plugin.is_managed else "External / unmanaged",
            ),
            ("Installed package version", plugin.installed_version or "Not managed"),
            ("Package archive", plugin.archive_name or "Not recorded"),
            ("Discovery source", plugin.source or "Unknown"),
            ("Configured enabled", configured),
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

        snapshot = self._snapshot
        if plugin.is_community and snapshot is not None and not snapshot.community_plugins_enabled:
            if plugin.configured_enabled:
                return self._banner_html(
                    "warning",
                    "This plugin remains configured as enabled, but the global Community "
                    "plugins switch is preventing it from running.",
                )
            return self._banner_html(
                "info",
                "Turn on Community plugins before enabling this third-party plugin.",
            )

        if plugin.status == "enabled" and plugin.open_instances:
            count = len(plugin.open_instances)
            action = "Disabling or reloading" if plugin.is_reloadable else "Disabling"
            return self._banner_html(
                "warning",
                f"{action} this plugin closes {count} open panel"
                f"{'s' if count != 1 else ''} and cancels its tracked jobs.",
            )

        if plugin.origin == "user" and not plugin.is_managed:
            return self._banner_html(
                "info",
                "This community plugin was added manually. AstronomicAL can enable "
                "or disable it, but will not update or uninstall its files.",
            )

        if plugin.is_development:
            return self._banner_html(
                "info",
                "Reload is available only for development plugins after editing their code.",
            )

        if plugin.is_bundled and plugin.status != "enabled":
            return self._banner_html(
                "info",
                "This bundled plugin is disabled for the current session and will be "
                "enabled again on the next AstronomicAL startup.",
            )

        if plugin.is_runtime:
            return self._banner_html(
                "info",
                "Runtime registrations do not expose install or lifecycle controls here.",
            )

        return ""

    def _package_details_html(self, plugin: PluginSnapshot) -> str:
        if not plugin.is_managed or plugin.origin != "user":
            return ""

        source = plugin.installed_source or "file"
        version = plugin.installed_version or plugin.version
        archive = plugin.archive_name or "Not recorded"
        details = (
            '<div class="al-pm-mini-grid">'
            '<div class="al-pm-mini"><div class="al-pm-mini-label">Managed</div>'
            '<div class="al-pm-mini-value">AstronomicAL</div></div>'
            '<div class="al-pm-mini"><div class="al-pm-mini-label">Installed version</div>'
            f'<div class="al-pm-mini-value">{html.escape(version)}</div></div>'
            '<div class="al-pm-mini"><div class="al-pm-mini-label">Install source</div>'
            f'<div class="al-pm-mini-value">{html.escape(source)}</div></div>'
            '<div class="al-pm-mini"><div class="al-pm-mini-label">Archive</div>'
            f'<div class="al-pm-mini-value">{html.escape(archive)}</div></div>'
            '</div>'
        )

        if plugin.status == "enabled":
            details += self._banner_html(
                "warning",
                "Updating this plugin will disable it first. It will remain disabled "
                "after the update so the new code is never executed implicitly.",
            )

        return details

    def _table_status(self, plugin: PluginSnapshot) -> str:
        snapshot = self._snapshot
        if (
            plugin.is_community
            and plugin.status != "enabled"
            and plugin.configured_enabled is True
            and snapshot is not None
            and not snapshot.community_plugins_enabled
        ):
            return "⊘ Blocked"

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
        return (
            f'<div class="al-pm-banner {html.escape(tone)}">'
            f"{html.escape(message)}</div>"
        )

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
            message_target="global",
        )

    def _community_toggle_changed(self, event: Any) -> None:
        if self._disposed or self._restoring or self._syncing_community_toggle:
            return

        if self.activation is None:
            self._set_operation_message(
                "danger", "PluginActivationService is not available."
            )
            self.refresh()
            return

        enabled = bool(getattr(event, "new", False))
        label = "Enabling community plugins" if enabled else "Disabling community plugins"
        self._set_busy(True)
        self._set_operation_message("info", f"{label}…")

        try:
            failures = self.activation.set_community_plugins_enabled(
                enabled,
                context=self.context,
            )
            self._publish_registry_changed(
                None,
                "community_enabled" if enabled else "community_disabled",
            )
        except Exception as exc:
            traceback.print_exc()
            self._set_operation_message("danger", f"{label} failed: {exc}")
        else:
            if failures:
                details = "; ".join(
                    f"{plugin_id}: {message}"
                    for plugin_id, message in sorted(failures.items())
                )
                self._set_operation_message(
                    "warning",
                    f"Community plugin setting changed, but some plugins could not "
                    f"be reconciled: {details}",
                )
            else:
                self._set_operation_message(
                    "success",
                    "Community plugins enabled."
                    if enabled
                    else "Community plugins disabled.",
                )
        finally:
            self._set_busy(False)
            self.refresh()

    def _package_file_changed(self, event: Any = None) -> None:
        if self._disposed or self._restoring:
            return

        changed_widget = getattr(event, "obj", None)
        if changed_widget is self.install_file:
            # A newly selected package starts a fresh install attempt. Do not
            # carry the previous package's success/failure message or retry-style
            # button label into the next selection.
            self.install_banner.object = ""
            self.install_button.name = "Install"
        elif changed_widget is self.update_file:
            # Apply the same principle to updates so a previous package error is
            # not shown against the newly selected archive.
            self._clear_package_message()

        self._update_install_action_state()
        self._update_package_action_state(self._selected_plugin())

    def _install_clicked(self, _event: Any = None) -> None:
        if self.installer is None:
            self.install_banner.object = self._banner_html(
                "danger", "PluginInstaller is not available."
            )
            return

        self._set_busy(True)
        self.install_banner.object = self._banner_html(
            "info", "Installing plugin package…"
        )
        result = None
        try:
            with self._uploaded_package_path(self.install_file) as archive_path:
                result = self.installer.install(archive_path)
            self._publish_registry_changed(result.plugin_id, "installed")
        except Exception as exc:
            traceback.print_exc()
            self.install_banner.object = self._banner_html(
                "danger", f"Installation failed: {exc}"
            )
            # Keep the selected archive available for an explicit retry, while
            # making it clear that this action applies to the current selection.
            self.install_button.name = "Install selected package"
        else:
            message = (
                f"Installed {result.plugin_id} {result.version}. The plugin remains disabled."
            )
            warnings = list(getattr(result, "warnings", ()) or ())
            if warnings:
                message += " " + " ".join(str(item) for item in warnings)
            self.install_banner.object = self._banner_html(
                "warning" if warnings else "success", message
            )
            self._clear_file_input(self.install_file)
            self.install_button.name = "Install"
        finally:
            self._set_busy(False)
            self.refresh()

        if result is not None:
            self._select_plugin_after_refresh(result.plugin_id)

    def _update_clicked(self, _event: Any = None) -> None:
        plugin = self._selected_plugin()
        if (
            plugin is None
            or not plugin.is_managed
            or plugin.origin != "user"
            or self.installer is None
        ):
            return

        self._pending_uninstall_plugin_id = None
        self._set_busy(True)
        self._set_package_message(plugin.id, "info", f"Preparing update for {plugin.name}…")
        result = None
        disabled_for_update = False
        try:
            with self._uploaded_package_path(self.update_file) as archive_path:
                inspection = self.installer.inspect(archive_path)
                package_id = str(getattr(inspection.manifest, "id", "") or "")
                if package_id != plugin.id:
                    raise ValueError(
                        f"Selected package is for plugin {package_id!r}, not {plugin.id!r}."
                    )

                if plugin.status == "enabled":
                    if self.activation is None:
                        raise RuntimeError(
                            "PluginActivationService is not available to disable the "
                            "running plugin before update."
                        )
                    self.activation.disable(plugin.id, context=self.context)
                    disabled_for_update = True

                result = self.installer.update(archive_path)

            self._publish_registry_changed(plugin.id, "updated")
        except Exception as exc:
            traceback.print_exc()
            suffix = (
                " The plugin was disabled before the update attempt and remains disabled."
                if disabled_for_update
                else ""
            )
            self._set_package_message(
                plugin.id, "danger", f"Update failed: {exc}{suffix}"
            )
        else:
            warnings = list(getattr(result, "warnings", ()) or ())
            message = (
                f"Updated {plugin.name} to version {result.version}. "
                "The updated plugin is not started automatically."
            )
            if warnings:
                message += " " + " ".join(str(item) for item in warnings)
            self._set_package_message(
                plugin.id, "warning" if warnings else "success", message
            )
            self._clear_file_input(self.update_file)
        finally:
            self._set_busy(False)
            self.refresh()

        self._select_plugin_after_refresh(plugin.id)

    def _uninstall_clicked(
        self,
        _event: Any = None,
        *,
        message_target: str = "detail",
    ) -> None:
        plugin = self._selected_plugin()
        if (
            plugin is None
            or not plugin.is_managed
            or plugin.origin != "user"
            or self.installer is None
        ):
            return

        if self._pending_uninstall_plugin_id != plugin.id:
            self._pending_uninstall_plugin_id = plugin.id
            message = (
                "Uninstall removes the AstronomicAL-managed plugin code. If the plugin "
                "is running it will be disabled first. Plugin data is preserved. "
                "Click Confirm uninstall to continue."
            )
            if message_target == "global":
                self._set_operation_message("warning", message)
            elif message_target == "detail":
                self.uninstall_banner.object = self._banner_html("warning", message)
            # For compact-row confirmation the button text itself is the prompt.
            # Do not add a banner above the list, because changing content above
            # the current viewport shifts the user's scroll position.
            self._update_package_action_state(plugin)
            return

        plugin_name = plugin.name
        plugin_id = plugin.id
        self._set_busy(True)
        if message_target in {"global", "row"}:
            self._set_operation_message("info", f"Uninstalling {plugin_name}…")
        else:
            self.uninstall_banner.object = self._banner_html(
                "info", f"Uninstalling {plugin_name}…"
            )
        result = None
        try:
            result = self.installer.uninstall(plugin_id, context=self.context)
            self._publish_registry_changed(plugin_id, "uninstalled")
        except Exception as exc:
            traceback.print_exc()
            self._pending_uninstall_plugin_id = None
            if message_target in {"global", "row"}:
                self._set_operation_message("danger", f"Uninstall failed: {exc}")
            else:
                self.uninstall_banner.object = self._banner_html(
                    "danger", f"Uninstall failed: {exc}"
                )
        else:
            warnings = list(getattr(result, "warnings", ()) or ())
            message = f"Uninstalled {plugin_name}."
            if warnings:
                message += " " + " ".join(str(item) for item in warnings)
            self._set_operation_message(
                "warning" if warnings else "success", message
            )
            self._pending_uninstall_plugin_id = None
            self._selected_plugin_id = None
            self.uninstall_banner.object = ""
            self._clear_package_message()
        finally:
            self._set_busy(False)
            self.refresh()

        if result is not None:
            self._selected_plugin_id = None
            self._expanded_plugin_id = None
            self._focus_plugin_id = None
            self._render_plugin_lists()
            self._render_selected_plugin()
            self._sync_installed_detail_visibility()

    def _enable_clicked(self, _event: Any = None, *, message_target: str = "plugin") -> None:
        plugin = self._selected_plugin()
        if plugin is None:
            return
        if self.activation is None:
            self._set_lifecycle_message(
                plugin.id,
                "danger",
                "PluginActivationService is not available.",
            )
            return
        if plugin.is_runtime or plugin.origin == "unknown":
            return

        snapshot = self._snapshot
        if (
            plugin.is_community
            and snapshot is not None
            and not snapshot.community_plugins_enabled
        ):
            self._set_lifecycle_message(
                plugin.id,
                "warning",
                "Community plugins are disabled. Turn on the Community plugins switch first.",
            )
            return

        self._run_operation(
            label=f"Enabling {plugin.name}",
            callback=lambda: self.activation.enable(
                plugin.id,
                context=self.context,
            ),
            success_message=f"Enabled {plugin.name}.",
            plugin_id=plugin.id,
            registry_operation="enabled",
            message_target=message_target,
        )

    def _disable_clicked(self, _event: Any = None, *, message_target: str = "plugin") -> None:
        plugin = self._selected_plugin()
        if plugin is None or plugin.id == self.SELF_PLUGIN_ID:
            return
        if self.activation is None:
            self._set_lifecycle_message(
                plugin.id,
                "danger",
                "PluginActivationService is not available.",
            )
            return
        if plugin.is_runtime or plugin.origin == "unknown":
            return

        self._run_operation(
            label=f"Disabling {plugin.name}",
            callback=lambda: self.activation.disable(
                plugin.id,
                context=self.context,
            ),
            success_message=f"Disabled {plugin.name}.",
            plugin_id=plugin.id,
            registry_operation="disabled",
            message_target=message_target,
        )

    def _reload_clicked(self, _event: Any = None) -> None:
        plugin = self._selected_plugin()
        if (
            plugin is None
            or self.manager is None
            or plugin.id == self.SELF_PLUGIN_ID
            or not plugin.is_reloadable
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
        message_target: str = "plugin",
    ) -> None:
        def set_message(tone: str, message: str) -> None:
            if message_target == "plugin" and plugin_id:
                self._set_lifecycle_message(plugin_id, tone, message)
            else:
                self._set_operation_message(tone, message)

        self._set_busy(True)
        set_message("info", f"{label}…")
        try:
            callback()
            self._publish_registry_changed(plugin_id, registry_operation)
        except Exception as exc:
            traceback.print_exc()
            set_message("danger", f"{label} failed: {exc}")
        else:
            set_message("success", success_message)
        finally:
            self._set_busy(False)
            self.refresh()

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        for button in (
            self.refresh_button,
            self.discover_button,
            self.enable_button,
            self.disable_button,
            self.reload_button,
            self.install_button,
            self.update_button,
            self.uninstall_button,
            self.install_file_toggle,
        ):
            try:
                button.loading = busy
            except Exception:
                pass

        self.search.disabled = busy
        self.status_filter.disabled = busy
        for button in list(self._row_buttons):
            try:
                button.disabled = busy
            except Exception:
                pass
        self.install_file.disabled = busy
        self.update_file.disabled = busy
        self.community_toggle.disabled = busy or self.activation is None

        if not busy:
            state_error = bool(
                self._snapshot is not None and self._snapshot.plugin_state_error
            )
            self.community_toggle.disabled = self.activation is None or state_error
            self._update_action_state(self._selected_plugin())
            self._update_install_action_state()
            self._update_package_action_state(self._selected_plugin())
            self._render_plugin_lists()

    def _set_operation_message(self, tone: str, message: str) -> None:
        self.operation_banner.object = self._banner_html(tone, message) if message else ""

    def _set_lifecycle_message(
        self,
        plugin_id: str,
        tone: str,
        message: str,
    ) -> None:
        self._lifecycle_message_plugin_id = str(plugin_id or "") or None
        self.lifecycle_banner.object = (
            self._banner_html(tone, message) if message else ""
        )

    def _clear_lifecycle_message(self) -> None:
        self._lifecycle_message_plugin_id = None
        self.lifecycle_banner.object = ""

    def _set_package_message(
        self,
        plugin_id: str,
        tone: str,
        message: str,
    ) -> None:
        self._package_message_plugin_id = str(plugin_id or "") or None
        self.package_banner.object = self._banner_html(tone, message) if message else ""

    def _clear_package_message(self) -> None:
        self._package_message_plugin_id = None
        self.package_banner.object = ""

    def _update_install_action_state(self) -> None:
        snapshot = self._snapshot
        store_error = bool(
            snapshot is not None and snapshot.installed_store_error
        )
        self.install_button.disabled = (
            self._busy
            or self.installer is None
            or self.installed_store is None
            or store_error
            or not self._file_input_has_value(self.install_file)
        )

    def _update_package_action_state(
        self,
        plugin: PluginSnapshot | None,
    ) -> None:
        is_community = bool(plugin is not None and plugin.is_community)
        managed = bool(
            plugin is not None
            and plugin.is_community
            and plugin.is_managed
            and plugin.origin == "user"
        )

        self.package_management.visible = managed
        self.managed_uninstall_controls.visible = is_community

        if plugin is None or not is_community:
            self.uninstall_note.object = ""
            self.update_button.disabled = True
            self.uninstall_button.disabled = True
            self.uninstall_button.name = "Uninstall plugin"
            self._pending_uninstall_plugin_id = None
            self.uninstall_banner.object = ""
            return

        if managed:
            self.uninstall_note.object = (
                '<div class="al-pm-uninstall-title">Installed plugin</div>'
                '<div class="al-pm-uninstall-copy">AstronomicAL installed this plugin '
                'and can remove its managed code safely.</div>'
            )
        else:
            self.uninstall_note.object = (
                '<div class="al-pm-uninstall-title">Uninstall</div>'
                '<div class="al-pm-uninstall-copy">This community plugin was not '
                'installed by AstronomicAL, so its files cannot be removed safely '
                'from Plugin Manager.</div>'
            )
            self.update_button.disabled = True
            self.uninstall_button.disabled = True
            self.uninstall_button.name = "Uninstall plugin"
            self._pending_uninstall_plugin_id = None
            self.uninstall_banner.object = ""
            return

        snapshot = self._snapshot
        store_error = bool(
            snapshot is not None and snapshot.installed_store_error
        )
        unavailable = (
            self._busy
            or self.installer is None
            or self.installed_store is None
            or store_error
        )
        self.update_button.disabled = (
            unavailable or not self._file_input_has_value(self.update_file)
        )
        self.update_button.name = (
            "Disable and update selected plugin"
            if plugin.status == "enabled"
            else "Update selected plugin"
        )
        self.uninstall_button.disabled = unavailable
        self.uninstall_button.name = (
            "Confirm uninstall"
            if self._pending_uninstall_plugin_id == plugin.id
            else "Uninstall plugin"
        )

    def _update_action_state(self, plugin: PluginSnapshot | None) -> None:
        if plugin is None:
            self.enable_button.name = "Enable plugin"
            self.disable_button.name = "Disable plugin"
            self.reload_button.name = "Reload development plugin"
            self.enable_button.disabled = True
            self.disable_button.disabled = True
            self.reload_button.disabled = True
            self.reload_button.visible = False
            return

        is_self = plugin.id == self.SELF_PLUGIN_ID
        activation_missing = self.activation is None
        runtime_enabled = plugin.status == "enabled"
        snapshot = self._snapshot
        community_allowed = bool(
            snapshot is not None and snapshot.community_plugins_enabled
        )

        self.enable_button.name = (
            "Retry enabling plugin"
            if plugin.is_community
            and plugin.configured_enabled
            and not runtime_enabled
            and community_allowed
            else "Enable plugin"
        )

        if plugin.is_runtime or plugin.origin == "unknown":
            self.enable_button.disabled = True
            self.disable_button.disabled = True
        elif plugin.is_community:
            self.enable_button.disabled = (
                activation_missing or runtime_enabled or not community_allowed
            )
            self.disable_button.disabled = (
                is_self
                or activation_missing
                or not (runtime_enabled or plugin.configured_enabled is True)
            )
        else:
            self.enable_button.disabled = activation_missing or runtime_enabled
            self.disable_button.disabled = (
                is_self or activation_missing or not runtime_enabled
            )

        self.reload_button.visible = plugin.is_reloadable and not is_self
        self.reload_button.disabled = not runtime_enabled

        open_count = len(plugin.open_instances)
        self.disable_button.name = (
            f"Disable and close {open_count} panel{'s' if open_count != 1 else ''}"
            if runtime_enabled and open_count
            else "Disable plugin"
        )
        self.reload_button.name = (
            f"Reload and close {open_count} panel{'s' if open_count != 1 else ''}"
            if open_count
            else "Reload development plugin"
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

    @staticmethod
    def _file_input_has_value(widget: Any) -> bool:
        value = getattr(widget, "value", None)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return len(value) > 0
        return bool(value)

    @staticmethod
    def _clear_file_input(widget: Any) -> None:
        try:
            widget.value = None
        except Exception:
            try:
                widget.value = b""
            except Exception:
                pass

    @contextmanager
    def _uploaded_package_path(self, widget: Any) -> Iterator[Path]:
        value = getattr(widget, "value", None)
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("Select exactly one .alplugin package.")
            value = value[0]

        if not isinstance(value, (bytes, bytearray, memoryview)) or not value:
            raise ValueError("Select an .alplugin package first.")

        filename = getattr(widget, "filename", None)
        if isinstance(filename, (list, tuple)):
            filename = filename[0] if filename else None
        filename = str(filename or "plugin.alplugin").replace("\\", "/")
        safe_name = filename.rsplit("/", 1)[-1].strip() or "plugin.alplugin"
        if not safe_name.lower().endswith(".alplugin"):
            raise ValueError("Plugin package filename must end with .alplugin.")

        with TemporaryDirectory(prefix="astronomical-plugin-upload-") as temporary:
            archive_path = Path(temporary) / safe_name
            archive_path.write_bytes(bytes(value))
            yield archive_path

    def _select_plugin_after_refresh(self, plugin_id: str) -> None:
        plugin_id = str(plugin_id or "").strip()
        if not plugin_id:
            return
        plugin = next(
            (item for item in (self._snapshot.plugins if self._snapshot else ()) if item.id == plugin_id),
            None,
        )
        if plugin is None:
            return
        self._selected_plugin_id = plugin_id
        self._expanded_plugin_id = plugin_id
        self._focus_plugin_id = plugin_id
        self.main_tabs.active = 0
        self.installed_tabs.active = 1 if plugin.is_community else 0
        self._apply_filters()
        self._sync_installed_detail_visibility()

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
            "expanded_plugin_id": self._expanded_plugin_id,
            "main_tab": int(getattr(self.main_tabs, "active", 0) or 0),
            "installed_tab": int(getattr(self.installed_tabs, "active", 0) or 0),
            "marketplace": self.marketplace_panel.get_state(),
        }

    def restore_state(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        self._restoring = True
        try:
            search_value = str(state.get("search", "") or "")
            self.search.value = search_value
            # Panel keeps the live TextInput text in value_input. The
            # search helpers intentionally prefer it so filtering responds
            # while the user types. Keep both parameters aligned when
            # restoring state programmatically.
            try:
                self.search.value_input = search_value
            except Exception:
                pass
            status_filter = str(state.get("status_filter", "all") or "all")
            if status_filter in {"all", "enabled", "available", "issues"}:
                self.status_filter.value = status_filter
            selected = state.get("selected_plugin_id")
            self._selected_plugin_id = str(selected) if selected else None
            expanded = state.get("expanded_plugin_id", selected)
            self._expanded_plugin_id = str(expanded) if expanded else None
            try:
                self.main_tabs.active = max(0, min(1, int(state.get("main_tab", 0) or 0)))
            except Exception:
                self.main_tabs.active = 0
            try:
                self.installed_tabs.active = max(0, min(1, int(state.get("installed_tab", 0) or 0)))
            except Exception:
                self.installed_tabs.active = 0
        finally:
            self._restoring = False
        self.marketplace_panel.restore_state(state.get("marketplace"))
        self._apply_filters()

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self.marketplace_panel.dispose()

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
        self._row_buttons.clear()