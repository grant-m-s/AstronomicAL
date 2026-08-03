from __future__ import annotations

import html
import json
import math
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterator, Mapping

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
    """User-facing management surface for AstronomicAL plugins."""

    state_version = 1
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
        self._syncing_selection = False
        self._syncing_page = False
        self._syncing_community_toggle = False
        self._subscriptions: list[Any] = []
        self._watchers: list[tuple[Any, Any]] = []
        self._snapshot: PluginManagerSnapshot | None = None
        self._visible_plugins: list[PluginSnapshot] = []
        self._selected_plugin_id: str | None = None
        self._lifecycle_message_plugin_id: str | None = None
        self._package_message_plugin_id: str | None = None
        self._pending_uninstall_plugin_id: str | None = None
        self._busy = False

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
        )
        self.disable_button = pn.widgets.Button(
            name="Disable plugin",
            button_type="warning",
            sizing_mode="stretch_width",
            height=34,
            margin=0,
        )
        self.reload_button = pn.widgets.Button(
            name="Reload development plugin",
            button_type="default",
            sizing_mode="stretch_width",
            height=34,
            margin=0,
            visible=False,
        )

        self.install_file = pn.widgets.FileInput(
            name="",
            accept=".alplugin",
            multiple=False,
            sizing_mode="stretch_width",
            height=38,
            margin=(0, 0, 12, 0),
        )
        self.install_button = pn.widgets.Button(
            name="Install",
            button_type="primary",
            sizing_mode="stretch_width",
            height=36,
            margin=(4, 0, 4, 0),
        )
        self.update_file = pn.widgets.FileInput(
            name="",
            accept=".alplugin",
            multiple=False,
            sizing_mode="stretch_width",
            height=38,
            margin=(0, 0, 12, 0),
        )
        self.update_button = pn.widgets.Button(
            name="Update selected plugin",
            button_type="primary",
            sizing_mode="stretch_width",
            height=34,
            margin=0,
        )
        self.uninstall_button = pn.widgets.Button(
            name="Uninstall installed plugin",
            button_type="danger",
            sizing_mode="stretch_width",
            height=38,
            margin=(2, 0, 4, 0),
        )

        self.plugin_table = pn.widgets.Tabulator(
            pd.DataFrame(
                columns=["Plugin", "Status", "Version", "Open", "Provides", "plugin_id"]
            ),
            show_index=False,
            selectable=1,
            pagination="local",
            # Keep a complete page inside the fixed table viewport. With
            # responsiveLayout="collapse", each plugin can occupy a normal row
            # plus a collapsed "Provides" row. A 12-row page therefore created
            # a second vertical scroller inside the already-scrollable Plugin
            # Manager panel. Tabulator restores that internal scroll during page
            # changes, which causes the visible jump/snap ("rubber band"). Five
            # plugins fit in this viewport without an internal vertical scroll.
            page_size=5,
            sizing_mode="stretch_width",
            height=400,
            min_height=400,
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
                # Tabulator otherwise preserves selection across pagination. A
                # selected row on another page can be restored while the page DOM
                # is being rebuilt, which is what caused the surrounding plugin
                # manager scroller to jump and then snap back when returning to a
                # previous page.
                "selectableRowsPersistence": False,
            },
            css_classes=["al-pm-table"],
            styles={"overflow-anchor": "none"},
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
        self.selected_details = self._html_pane()
        self.action_note = self._html_pane()
        self.discovery_issues = self._html_pane()

        # Dynamic HTML panes and Bokeh file-input labels can otherwise be laid out
        # too tightly inside nested Columns. Reserve comfortable vertical space
        # around the always-present status banners and package feedback.
        self.community_status.min_height = 50
        self.community_status.margin = (0, 0, 6, 0)
        self.install_status.min_height = 46
        self.install_status.margin = (4, 0, 10, 0)
        self.install_banner.margin = (8, 0, 10, 0)
        self.package_details.margin = (4, 0, 10, 0)
        self.package_banner.margin = (8, 0, 10, 0)
        self.uninstall_banner.margin = (6, 0, 8, 0)

        self.refresh_button.on_click(self._refresh_clicked)
        self.discover_button.on_click(self._discover_clicked)
        self.enable_button.on_click(self._enable_clicked)
        self.disable_button.on_click(self._disable_clicked)
        self.reload_button.on_click(self._reload_clicked)
        self.install_button.on_click(self._install_clicked)
        self.update_button.on_click(self._update_clicked)
        self.uninstall_button.on_click(self._uninstall_clicked)

        self._watch(self.search, self._filters_changed, "value")
        self._watch(self.search, self._filters_changed, "value_input")
        self._watch(self.status_filter, self._filters_changed, "value")
        self._watch(self.community_toggle, self._community_toggle_changed, "value")
        self._watch(self.plugin_table, self._table_selection_changed, "selection")
        self._watch(self.plugin_table, self._table_page_changed, "page")
        self._watch(self.install_file, self._package_file_changed, "value")
        self._watch(self.update_file, self._package_file_changed, "value")
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
            styles=self._card_styles(),
            margin=(0, 0, 10, 0),
        )

        community_heading = pn.pane.HTML(
            '<div class="al-pm-section-title">'
            "<h3>Community plugins</h3>"
            "<span>Allow third-party plugins to execute in AstronomicAL</span>"
            "</div>",
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            min_height=44,
            margin=0,
        )
        community_controls = pn.Row(
            community_heading,
            self.community_toggle,
            sizing_mode="stretch_width",
            min_height=50,
            margin=(0, 0, 8, 0),
        )
        community = pn.Column(
            community_controls,
            pn.Spacer(height=4, sizing_mode="stretch_width", margin=0),
            self.community_status,
            sizing_mode="stretch_width",
            css_classes=["al-pm-card"],
            styles=self._card_styles(),
            margin=(0, 0, 16, 0),
        )

        install_file_label = pn.pane.HTML(
            '<div style="font-size:12px; font-weight:600; line-height:1.35;">'
            'Plugin package (.alplugin)</div>',
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            height=22,
            margin=(2, 0, 4, 0),
        )
        install_from_file = self._section(
            "Install from file",
            "Install a local .alplugin package. Installation never enables or executes the plugin.",
            pn.Column(
                self.install_status,
                install_file_label,
                self.install_file,
                self.install_banner,
                self.install_button,
                sizing_mode="stretch_width",
                margin=(4, 0, 4, 0),
            ),
        )
        install_from_file.margin = (0, 0, 16, 0)

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

        self.managed_uninstall_controls = pn.Column(
            pn.pane.HTML(
                '<div style="font-size:12px; font-weight:600; line-height:1.35;">'
                'Installed plugin</div>'
                '<div style="font-size:11px; color:#667085; line-height:1.4; margin-top:2px;">'
                'AstronomicAL installed this plugin and can remove its managed code safely.'
                '</div>',
                sizing_mode="stretch_width",
                stylesheets=[PLUGIN_MANAGER_CSS],
                margin=(0, 0, 6, 0),
            ),
            self.uninstall_banner,
            self.uninstall_button,
            sizing_mode="stretch_width",
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
                "<span>Update or remove plugins installed by AstronomicAL</span></div>",
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

        selected = pn.Column(
            pn.pane.HTML(
                '<div class="al-pm-section-title"><h3>Selected plugin</h3>'
                "<span>Health, useful features, open panels, and lifecycle controls</span></div>",
                sizing_mode="stretch_width",
                stylesheets=[PLUGIN_MANAGER_CSS],
                margin=0,
            ),
            self.selected_details,
            self.action_note,
            # Plugin lifecycle feedback belongs next to the lifecycle controls.
            # Keeping enable/disable/reload failures here prevents an important
            # refusal from appearing far above the button the user just clicked.
            self.lifecycle_banner,
            actions,
            # Managed community plugins get an explicit destructive action directly
            # below the lifecycle controls. This keeps Uninstall visible without
            # forcing the user to hunt through the update-package section.
            self.managed_uninstall_controls,
            self.package_management,
            sizing_mode="stretch_width",
            css_classes=["al-pm-card"],
            styles=self._card_styles(),
            margin=(0, 0, 10, 0),
        )

        return pn.Column(
            self.header,
            self.operation_banner,
            self.summary,
            community,
            install_from_file,
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
                # Prevent browser scroll anchoring from reacting to Tabulator's
                # paginated row DOM replacement inside this nested scroller.
                "overflow-anchor": "none",
                "padding": "8px",
            },
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=0,
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
            self.community_status.object = self._banner_html(
                "danger", "Community plugin state is unavailable."
            )
            self.install_status.object = self._banner_html(
                "danger", "Plugin installation state is unavailable."
            )
            self.selected_details.object = self._empty_html("No plugin can be selected.")
            self.discovery_issues.object = self._empty_html("No discovery information.")
            self.discovery_section.visible = False
            self._replace_table_value(pd.DataFrame())
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

    def _apply_filters(self, *, reset_page: bool = False) -> None:
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

        current_page = 1 if reset_page else self._current_page()

        # Preserve the selected plugin by identity across lifecycle refreshes. The
        # diagnostics list is intentionally status-ranked, so enabling/disabling a
        # plugin can move it to a different page. Previously the refresh stayed on
        # the old page and silently selected that page's first plugin instead.
        selected_index = None
        if self._selected_plugin_id is not None:
            selected_index = next(
                (
                    index
                    for index, plugin in enumerate(visible)
                    if plugin.id == self._selected_plugin_id
                ),
                None,
            )

        self._replace_table_value(
            pd.DataFrame(
                rows,
                columns=["Plugin", "Status", "Version", "Open", "Provides", "plugin_id"],
            )
        )

        if selected_index is not None and not reset_page:
            target_page = (selected_index // self._page_size()) + 1
        else:
            target_page = min(current_page, self._max_page(len(visible)))

        self._set_table_page(target_page)

        page_plugins = self._plugins_on_page(target_page)
        page_ids = {plugin.id for plugin in page_plugins}
        if self._selected_plugin_id not in page_ids:
            self._selected_plugin_id = page_plugins[0].id if page_plugins else None

        self._sync_table_selection()
        self._render_selected_plugin()

    def _replace_table_value(self, value: pd.DataFrame) -> None:
        # Clearing the selection before replacing the data prevents the frontend
        # from trying to restore a selected row while Tabulator is rebuilding a
        # paginated page. Suppress page callbacks too because replacing the value
        # may reset Tabulator's current page before we restore the intended page.
        self._syncing_selection = True
        self._syncing_page = True
        try:
            self.plugin_table.selection = []
            self.plugin_table.value = value
        finally:
            self._syncing_page = False
            self._syncing_selection = False

    def _sync_table_selection(self) -> None:
        self._syncing_selection = True
        try:
            if self._selected_plugin_id is None:
                self.plugin_table.selection = []
                return

            page = self._current_page()
            start, end = self._page_bounds(page)
            for index in range(start, min(end, len(self._visible_plugins))):
                if self._visible_plugins[index].id == self._selected_plugin_id:
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

    def _table_page_changed(self, event: Any) -> None:
        if self._disposed or self._restoring or self._syncing_page:
            return

        try:
            page = max(1, int(getattr(event, "new", None) or 1))
        except Exception:
            page = 1

        page_plugins = self._plugins_on_page(page)
        if not page_plugins:
            self._selected_plugin_id = None
        elif self._selected_plugin_id not in {plugin.id for plugin in page_plugins}:
            # Update the details pane to the new page, but deliberately do not
            # programmatically select the row in Tabulator. Setting selection
            # while a page is being rebuilt makes Tabulator scroll the selected
            # row into view, which is the second source of the page-change jump.
            self._selected_plugin_id = page_plugins[0].id

        self._clear_table_selection()
        self._render_selected_plugin()

    def _clear_table_selection(self) -> None:
        self._syncing_selection = True
        try:
            if self.plugin_table.selection:
                self.plugin_table.selection = []
        finally:
            self._syncing_selection = False

    def _filters_changed(self, _event: Any = None) -> None:
        if self._disposed or self._restoring:
            return
        self._apply_filters(reset_page=True)

    def _search_value(self) -> str:
        value_input = getattr(self.search, "value_input", None)
        if value_input is not None:
            return str(value_input or "")
        return str(self.search.value or "")

    def _current_page(self) -> int:
        try:
            return max(1, int(getattr(self.plugin_table, "page", 1) or 1))
        except Exception:
            return 1

    def _page_size(self) -> int:
        try:
            return max(1, int(getattr(self.plugin_table, "page_size", 5) or 5))
        except Exception:
            return 5

    def _max_page(self, row_count: int) -> int:
        return max(1, math.ceil(max(0, int(row_count)) / self._page_size()))

    def _page_bounds(self, page: int) -> tuple[int, int]:
        page_size = self._page_size()
        start = (max(1, int(page)) - 1) * page_size
        return start, start + page_size

    def _plugins_on_page(self, page: int) -> list[PluginSnapshot]:
        start, end = self._page_bounds(page)
        return self._visible_plugins[start:end]

    def _set_table_page(self, page: int) -> None:
        self._syncing_page = True
        try:
            if getattr(self.plugin_table, "page", 1) != page:
                self.plugin_table.page = page
        finally:
            self._syncing_page = False

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

    def _uninstall_clicked(self, _event: Any = None) -> None:
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
            self.uninstall_banner.object = self._banner_html(
                "warning",
                "Uninstall removes the AstronomicAL-managed plugin code. If the plugin "
                "is running it will be disabled first. Plugin data is preserved. "
                "Click Confirm uninstall to continue.",
            )
            self._update_package_action_state(plugin)
            return

        plugin_name = plugin.name
        plugin_id = plugin.id
        self._set_busy(True)
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
            # refresh() normally selects the first row on the visible page. After an
            # uninstall, deliberately leave the detail card empty instead so the UI
            # does not make a different plugin look like the one just removed.
            self._selected_plugin_id = None
            self._clear_table_selection()
            self._render_selected_plugin()

    def _enable_clicked(self, _event: Any = None) -> None:
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
        )

    def _disable_clicked(self, _event: Any = None) -> None:
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
        ):
            try:
                button.loading = busy
            except Exception:
                pass

        self.search.disabled = busy
        self.status_filter.disabled = busy
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
        managed = bool(
            plugin is not None
            and plugin.is_managed
            and plugin.origin == "user"
        )
        self.package_management.visible = managed
        self.managed_uninstall_controls.visible = managed

        if not managed or plugin is None:
            self.update_button.disabled = True
            self.uninstall_button.disabled = True
            self.uninstall_button.name = "Uninstall installed plugin"
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
            else "Uninstall installed plugin"
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

        for index, plugin in enumerate(self._visible_plugins):
            if plugin.id != plugin_id:
                continue
            page = (index // self._page_size()) + 1
            self._selected_plugin_id = plugin_id
            self._set_table_page(page)
            self._sync_table_selection()
            self._render_selected_plugin()
            return

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
        self._apply_filters(reset_page=True)

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