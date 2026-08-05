from __future__ import annotations

import html
from typing import Any, Callable, Mapping

import panel as pn

from .marketplace_view import MarketplaceBrowseEntry, MarketplaceViewSnapshot, collect_marketplace_view
from .styles import PLUGIN_MANAGER_CSS


class PluginMarketplacePanel:
    """Marketplace browser embedded inside Plugin Manager."""

    def __init__(
        self,
        context: Any,
        *,
        on_changed: Callable[[], None] | None = None,
        on_manage: Callable[[str], None] | None = None,
    ):
        self.context = context
        self.marketplace = getattr(context, "marketplace", None)
        self.planner = getattr(context, "marketplace_planner", None)
        self.updates = getattr(context, "marketplace_updates", None)
        self.installer = getattr(context, "marketplace_installer", None)
        self._on_changed = on_changed
        self._on_manage = on_manage

        self._disposed = False
        self._restoring = False
        self._busy = False
        self._snapshot: MarketplaceViewSnapshot | None = None
        self._source_id: str | None = None
        self._selected_plugin_id: str | None = None
        self._refresh_job: Any | None = None
        self._install_job: Any | None = None
        self._generation = 0
        self._watchers: list[tuple[Any, Any]] = []
        self._card_buttons: list[pn.widgets.Button] = []

        self.source = pn.widgets.Select(
            name="Marketplace",
            options={},
            sizing_mode="stretch_width",
            margin=0,
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh catalogue",
            button_type="default",
            width=132,
            height=32,
            margin=0,
        )
        self.search = pn.widgets.TextInput(
            name="Search marketplace",
            placeholder="Name, description, tag, or plugin ID…",
            sizing_mode="stretch_width",
            margin=0,
        )

        self.status = self._html_pane()
        self.operation_banner = self._html_pane()
        self.plugin_list = pn.Column(
            sizing_mode="stretch_width",
            margin=0,
            css_classes=["al-pm-marketplace-list"],
            stylesheets=[PLUGIN_MANAGER_CSS],
        )

        self.status.margin = (0, 0, 8, 0)
        self.operation_banner.margin = (0, 0, 8, 0)

        self.refresh_button.on_click(self._refresh_clicked)
        self._watch(self.source, self._source_changed, "value")
        self._watch(self.search, self._search_changed, "value")
        self._watch(self.search, self._search_changed, "value_input")

        self.view = self._build_view()
        self.refresh()

    def _build_view(self) -> pn.Column:
        source_controls = pn.Row(
            self.source,
            self.refresh_button,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        return pn.Column(
            self.operation_banner,
            self.search,
            source_controls,
            self.status,
            self.plugin_list,
            sizing_mode="stretch_width",
            margin=(8, 0, 0, 0),
        )

    def refresh(self) -> None:
        if self._disposed:
            return

        if self.marketplace is None:
            self.source.options = {}
            self.source.disabled = True
            self.refresh_button.disabled = True
            self.status.object = self._banner_html(
                "info",
                "Marketplace services are unavailable.",
            )
            self.plugin_list.objects = [
                self._empty_pane("Marketplace plugins are unavailable.")
            ]
            return

        try:
            sources = list(self.marketplace.sources() or [])
        except Exception as exc:
            sources = []
            self.status.object = self._banner_html(
                "danger", f"Marketplace sources are unavailable: {exc}"
            )

        options: dict[str, str] = {}
        for source in sources:
            source_id = str(getattr(source, "id", "") or "")
            if not source_id:
                continue
            try:
                catalogue = self.marketplace.catalogue(source_id)
            except Exception:
                catalogue = None
            label = str(
                getattr(getattr(catalogue, "marketplace", None), "name", "")
                or source_id
            )
            if label in options:
                label = f"{label} ({source_id})"
            options[label] = source_id

        current = self._source_id
        if current not in set(options.values()):
            current = next(iter(options.values()), None)
        self._source_id = current

        self._restoring = True
        try:
            self.source.options = options
            if current is not None:
                self.source.value = current
        finally:
            self._restoring = False

        # The official marketplace is currently the only source. Preserve the
        # source widget for future multi-source support without wasting panel space.
        self.source.visible = len(options) > 1
        self.source.disabled = self._busy or not bool(options)
        self.refresh_button.disabled = self._busy or current is None

        view = collect_marketplace_view(
            self.context,
            source_id=current,
            query=self._search_value(),
        )
        self._snapshot = view
        self._render_source_status(view)

        visible_ids = {entry.plugin_id for entry in view.browse}
        if self._selected_plugin_id not in visible_ids:
            self._selected_plugin_id = None

        self._render_cards()

    def _render_source_status(self, view: MarketplaceViewSnapshot) -> None:
        if view.source_id is None:
            self.status.object = self._banner_html(
                "info",
                "No marketplace source is available.",
            )
        elif view.source_error:
            self.status.object = self._banner_html(
                "warning",
                f"{view.source_name or view.source_id} is using limited catalogue "
                f"state: {view.source_error}",
            )
        elif not view.catalogue_loaded:
            self.status.object = self._banner_html(
                "info",
                f"No catalogue is loaded for {view.source_name or view.source_id}. "
                "Refresh the catalogue to browse plugins.",
            )
        else:
            status = view.source_status or "loaded"
            self.status.object = self._banner_html(
                "success",
                f"{view.source_name or view.source_id} · {len(view.browse)} plugin"
                f"{'s' if len(view.browse) != 1 else ''} · catalogue {status}.",
            )

    def _render_cards(self) -> None:
        view = self._snapshot
        self._card_buttons = []
        if view is None or view.source_id is None:
            self.plugin_list.objects = [
                self._empty_pane("No marketplace plugins are available.")
            ]
            return
        if not view.browse:
            self.plugin_list.objects = [
                self._empty_pane("No marketplace plugins match this search.")
            ]
            return

        self.plugin_list.objects = [self._plugin_card(entry) for entry in view.browse]

    def _plugin_card(self, entry: MarketplaceBrowseEntry) -> pn.Column:
        selected = self._selected_plugin_id == entry.plugin_id
        tone = self._status_tone(entry.status)
        installed_badge = ""
        if entry.installed_version:
            installed_badge = (
                '<span class="al-pm-market-badge installed">Installed '
                f'{html.escape(entry.installed_version)}</span>'
            )
        update_badge = ""
        if entry.update_version:
            update_badge = (
                '<span class="al-pm-market-badge update">Update '
                f'{html.escape(entry.update_version)}</span>'
            )
        tags = " · ".join(entry.tags)
        tags_html = (
            f'<div class="al-pm-market-tags">{html.escape(tags)}</div>' if tags else ""
        )
        info = pn.pane.HTML(
            (
                '<div class="al-pm-market-card-info">'
                '<div class="al-pm-row-title-line">'
                f'<strong class="al-pm-row-name">{html.escape(entry.name)}</strong>'
                f'<span class="al-pm-status-pill {tone}">{html.escape(entry.status)}</span>'
                '</div>'
                f'<div class="al-pm-row-meta">Latest {html.escape(entry.version)} '
                f'{installed_badge}{update_badge}</div>'
                f'<div class="al-pm-row-description">'
                f'{html.escape(entry.description or "No description was provided.")}</div>'
                f'{tags_html}'
                '</div>'
            ),
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=0,
        )

        action = self._action_button(entry)
        details = pn.widgets.Button(
            name=(
                "Details"
                if entry.installed_version
                else ("Hide" if selected else "Details")
            ),
            button_type="default",
            width=78,
            height=30,
            margin=0,
            css_classes=["al-pm-row-action", "al-pm-row-action-quiet"],
            stylesheets=[PLUGIN_MANAGER_CSS],
        )
        details.disabled = self._busy
        if entry.installed_version:
            details.on_click(
                lambda _event, plugin_id=entry.plugin_id: self._manage_plugin(plugin_id)
            )
        else:
            details.on_click(
                lambda _event, plugin_id=entry.plugin_id: self._toggle_details(plugin_id)
            )
        self._card_buttons.append(details)

        buttons = [button for button in (action, details) if button is not None]
        action_row = pn.Row(
            *buttons,
            sizing_mode="stretch_width",
            css_classes=["al-pm-card-actions"],
            stylesheets=[PLUGIN_MANAGER_CSS],
            styles={"justify-content": "flex-end"},
            margin=(8, 0, 0, 0),
        )
        objects: list[Any] = [info, action_row]
        if selected and not entry.installed_version:
            objects.append(self._details_for(entry))

        return pn.Column(
            *objects,
            sizing_mode="stretch_width",
            css_classes=["al-pm-market-card"],
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=(0, 0, 8, 0),
        )

    def _action_button(self, entry: MarketplaceBrowseEntry) -> pn.widgets.Button | None:
        status = str(entry.status or "")
        if entry.update_version and status != "Installed elsewhere":
            name = "Update"
            button_type = "primary"
            callback = lambda _event, plugin_id=entry.plugin_id: self._update_plugin(plugin_id)
        elif entry.installed_version:
            return None
        else:
            name = "Install"
            button_type = "primary"
            callback = lambda _event, plugin_id=entry.plugin_id: self._install_plugin(plugin_id)

        button = pn.widgets.Button(
            name=name,
            button_type=button_type,
            width=78,
            height=30,
            margin=0,
            css_classes=["al-pm-row-action", "al-pm-row-action-primary"],
            stylesheets=[PLUGIN_MANAGER_CSS],
        )
        button.disabled = self._busy or (
            name in {"Install", "Update"} and self.installer is None
        )
        button.on_click(callback)
        self._card_buttons.append(button)
        return button

    def _details_for(self, entry: MarketplaceBrowseEntry) -> pn.Column:
        installed = entry.installed_version or "Not installed"
        update = entry.update_version or "None"
        license_value = entry.license or "Not specified"
        details = pn.pane.HTML(
            (
                '<div class="al-pm-mini-grid">'
                '<div class="al-pm-mini"><div class="al-pm-mini-label">Plugin ID</div>'
                f'<div class="al-pm-mini-value">{html.escape(entry.plugin_id)}</div></div>'
                '<div class="al-pm-mini"><div class="al-pm-mini-label">Installed</div>'
                f'<div class="al-pm-mini-value">{html.escape(installed)}</div></div>'
                '<div class="al-pm-mini"><div class="al-pm-mini-label">Update</div>'
                f'<div class="al-pm-mini-value">{html.escape(update)}</div></div>'
                '<div class="al-pm-mini"><div class="al-pm-mini-label">License</div>'
                f'<div class="al-pm-mini-value">{html.escape(license_value)}</div></div>'
                '</div>'
            ),
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=(6, 0, 0, 0),
        )

        plan_pane = self._html_pane()
        try:
            plan = self._plan_for_entry(entry)
        except Exception as exc:
            plan_pane.object = self._banner_html(
                "warning", f"No safe marketplace plan is available: {exc}"
            )
        else:
            plan_pane.object = self._plan_html(plan)
            blockers, preflight_error = self._plan_preflight(plan)
            if preflight_error:
                plan_pane.object += self._banner_html(
                    "warning", f"Marketplace preflight failed: {preflight_error}"
                )
            elif blockers:
                plan_pane.object += self._banner_html(
                    "warning",
                    "Disable these managed plugins before continuing: "
                    + ", ".join(blockers),
                )

        return pn.Column(
            details,
            plan_pane,
            sizing_mode="stretch_width",
            css_classes=["al-pm-market-details"],
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=0,
        )

    def _plan_for_entry(self, entry: MarketplaceBrowseEntry) -> Any:
        view = self._snapshot
        if view is None or view.source_id is None:
            raise RuntimeError("Marketplace source is unavailable.")
        if entry.update_version:
            if self.updates is None:
                raise RuntimeError("MarketplaceUpdatesFacade is not available.")
            return self.updates.plan_update(entry.plugin_id)
        if self.planner is None:
            raise RuntimeError("MarketplacePlanningService is not available.")
        return self.planner.plan_install(view.source_id, entry.plugin_id)

    def _toggle_details(self, plugin_id: str) -> None:
        if self._disposed or self._busy:
            return
        self._selected_plugin_id = (
            None if self._selected_plugin_id == plugin_id else str(plugin_id or "") or None
        )
        self._render_cards()

    def _manage_plugin(self, plugin_id: str) -> None:
        if callable(self._on_manage):
            self._on_manage(plugin_id)

    def _install_plugin(self, plugin_id: str) -> None:
        view = self._snapshot
        if view is None or view.source_id is None or self.planner is None:
            return
        try:
            plan = self.planner.plan_install(view.source_id, plugin_id)
        except Exception as exc:
            self.operation_banner.object = self._banner_html(
                "warning", f"No safe install plan is available: {exc}"
            )
            return
        self._execute_plan(plan, label=f"Installing {plugin_id}", focus_plugin_id=plugin_id)

    def _update_plugin(self, plugin_id: str) -> None:
        if self.updates is None:
            return
        try:
            plan = self.updates.plan_update(plugin_id)
        except Exception as exc:
            self.operation_banner.object = self._banner_html(
                "warning", f"No safe update plan is available: {exc}"
            )
            return
        self._execute_plan(plan, label=f"Updating {plugin_id}", focus_plugin_id=plugin_id)

    def _refresh_clicked(self, _event: Any = None) -> None:
        source_id = self._source_id
        if self.marketplace is None or not source_id:
            return
        submit = self._job_submitter()
        if submit is None:
            self.operation_banner.object = self._banner_html(
                "danger", "JobManager is not available for marketplace refresh."
            )
            return

        self._generation += 1
        generation = self._generation
        self._set_busy(True)
        self.operation_banner.object = self._banner_html(
            "info", f"Refreshing marketplace {source_id}…"
        )

        def work(*, cancel_token):
            if cancel_token.cancelled():
                return None
            return self.marketplace.refresh(source_id)

        def on_done(result):
            self._refresh_job = None
            if self._disposed or generation != self._generation:
                return
            self._set_busy(False)
            if result is None:
                self.operation_banner.object = self._banner_html(
                    "warning", "Marketplace refresh was cancelled."
                )
            else:
                status = str(getattr(result, "status", "updated") or "updated")
                error = str(getattr(result, "error", "") or "")
                message = f"Marketplace catalogue {status}."
                if error:
                    message += f" {error}"
                self.operation_banner.object = self._banner_html(
                    "warning" if error else "success", message
                )
            self.refresh()

        def on_error(exc):
            self._refresh_job = None
            if self._disposed or generation != self._generation:
                return
            self._set_busy(False)
            self.operation_banner.object = self._banner_html(
                "danger", f"Marketplace refresh failed: {exc}"
            )
            self.refresh()

        try:
            self._refresh_job = submit(
                work,
                title=f"Refresh marketplace {source_id}",
                key=f"core.plugin_manager.marketplace.refresh:{source_id}",
                on_done=on_done,
                on_error=on_error,
            )
        except Exception as exc:
            self._refresh_job = None
            self._set_busy(False)
            self.operation_banner.object = self._banner_html(
                "danger", f"Could not submit marketplace refresh: {exc}"
            )

    def _execute_plan(
        self,
        plan: Any,
        *,
        label: str,
        focus_plugin_id: str,
    ) -> None:
        if plan is None or self.installer is None:
            return
        blockers, preflight_error = self._plan_preflight(plan)
        if preflight_error is not None:
            self.operation_banner.object = self._banner_html(
                "warning", f"Marketplace preflight failed: {preflight_error}"
            )
            return
        if blockers:
            self.operation_banner.object = self._banner_html(
                "warning",
                "Disable these managed plugins before continuing: " + ", ".join(blockers),
            )
            return
        submit = self._job_submitter()
        if submit is None:
            self.operation_banner.object = self._banner_html(
                "danger", "JobManager is not available for marketplace installation."
            )
            return

        requested = str(getattr(plan, "requested_plugin_id", focus_plugin_id) or focus_plugin_id)
        self._set_busy(True)
        self.operation_banner.object = self._banner_html("info", f"{label}…")

        def work(*, cancel_token):
            if cancel_token.cancelled():
                return None
            return self.installer.execute(plan)

        def on_done(execution):
            self._install_job = None
            if self._disposed:
                return
            self._set_busy(False)
            if execution is None:
                self.operation_banner.object = self._banner_html(
                    "warning", "Marketplace operation was cancelled before it started."
                )
                return
            results = list(getattr(execution, "results", ()) or ())
            for result in results:
                self._publish_registry_changed(
                    str(getattr(result, "plugin_id", "") or "") or None,
                    str(getattr(result, "operation", "updated") or "updated"),
                )
            summary = ", ".join(
                f"{result.plugin_id} {result.version}" for result in results
            ) or "No package changes were required"
            if callable(self._on_changed):
                self._on_changed()
            self.refresh()
            self.operation_banner.object = self._banner_html(
                "success",
                f"Marketplace operation completed: {summary}. "
                "You can continue browsing, or press Details to manage the installed plugin.",
            )

        def on_error(exc):
            self._install_job = None
            if self._disposed:
                return
            self._set_busy(False)
            self.operation_banner.object = self._banner_html(
                "danger", f"Marketplace operation failed: {exc}"
            )
            self.refresh()
            if callable(self._on_changed):
                self._on_changed()

        try:
            self._install_job = submit(
                work,
                title=f"Marketplace install {requested}",
                key=f"core.plugin_manager.marketplace.install:{requested}",
                on_done=on_done,
                on_error=on_error,
            )
        except Exception as exc:
            self._install_job = None
            self._set_busy(False)
            self.operation_banner.object = self._banner_html(
                "danger", f"Could not submit marketplace operation: {exc}"
            )

    def _source_changed(self, event: Any) -> None:
        if self._disposed or self._restoring:
            return
        value = str(getattr(event, "new", "") or "").strip()
        self._source_id = value or None
        self._selected_plugin_id = None
        self.refresh()

    def _search_changed(self, _event: Any = None) -> None:
        if self._disposed or self._restoring:
            return
        self._selected_plugin_id = None
        self.refresh()

    def _plan_preflight(self, plan: Any) -> tuple[list[str], str | None]:
        preflight = getattr(self.installer, "preflight", None)
        if callable(preflight):
            try:
                result = preflight(plan)
            except Exception as exc:
                return [], str(exc)
            return list(getattr(result, "blockers", ()) or ()), None
        return self._running_update_blockers(plan), None

    def _running_update_blockers(self, plan: Any) -> list[str]:
        manager = getattr(self.context, "plugins", None)
        list_plugins = getattr(manager, "list_plugins", None)
        if not callable(list_plugins):
            return []
        try:
            infos = list(list_plugins() or [])
        except Exception:
            return []
        enabled = {
            str(getattr(info, "id", "") or "")
            for info in infos
            if getattr(
                getattr(info, "status", None),
                "value",
                getattr(info, "status", None),
            ) == "enabled"
        }
        return sorted(
            item.plugin_id
            for item in (getattr(plan, "changes", ()) or ())
            if getattr(item, "action", None) == "update" and item.plugin_id in enabled
        )

    def _job_submitter(self) -> Callable[..., Any] | None:
        jobs = getattr(self.context, "jobs", None)
        submit = getattr(jobs, "submit", None)
        return submit if callable(submit) else None

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        try:
            self.refresh_button.loading = busy
        except Exception:
            pass
        self.search.disabled = busy
        self.source.disabled = busy or not bool(self.source.options)
        self.refresh_button.disabled = busy or self._source_id is None
        for button in list(self._card_buttons):
            try:
                button.disabled = busy
                button.loading = busy
            except Exception:
                pass
        if not busy:
            self.refresh()

    def _publish_registry_changed(self, plugin_id: str | None, operation: str) -> None:
        manager = getattr(self.context, "plugins", None)
        publish = getattr(manager, "publish_registry_changed", None)
        if callable(publish):
            try:
                publish(self.context, plugin_id=plugin_id, operation=operation)
                return
            except Exception:
                pass
        events = getattr(self.context, "events", None)
        if events is not None:
            try:
                events.publish(
                    "plugin.registry.changed",
                    {"plugin_id": plugin_id, "operation": operation},
                )
            except Exception:
                pass

    def _plan_html(self, plan: Any) -> str:
        items = list(getattr(plan, "items", ()) or ())
        if not items:
            return self._banner_html("info", "No package changes are required.")
        rows = []
        for item in items:
            action = str(getattr(item, "action", "") or "")
            version = str(getattr(item, "version", "") or "")
            existing = str(getattr(item, "existing_version", "") or "")
            required_by = ", ".join(getattr(item, "required_by", ()) or ()) or "Requested"
            change = f"{existing} → {version}" if action == "update" and existing else version
            rows.append(
                '<div class="al-pm-plan-row">'
                f'<span class="al-pm-plan-action {html.escape(action)}">'
                f'{html.escape(action.upper())}</span>'
                f'<span class="al-pm-plan-plugin">'
                f'{html.escape(str(getattr(item, "plugin_id", "")))}</span>'
                f'<span class="al-pm-plan-version">{html.escape(change)}</span>'
                f'<span class="al-pm-plan-reason">{html.escape(required_by)}</span>'
                '</div>'
            )
        return (
            '<div class="al-pm-subheading">Package plan</div>'
            '<div class="al-pm-plan">'
            '<div class="al-pm-plan-head"><span>Action</span><span>Plugin</span>'
            '<span>Version</span><span>Required by</span></div>'
            + "".join(rows)
            + "</div>"
        )

    @staticmethod
    def _status_tone(status: str) -> str:
        value = str(status or "").lower()
        if "update" in value or "withdrawn" in value:
            return "warning"
        if "installed" in value:
            return "success"
        return "info"

    def _search_value(self) -> str:
        value_input = getattr(self.search, "value_input", None)
        if value_input is not None:
            return str(value_input or "")
        return str(self.search.value or "")

    def get_state(self) -> dict[str, Any]:
        return {
            "source_id": self._source_id,
            "search": self._search_value(),
            "selected_plugin_id": self._selected_plugin_id,
        }

    def restore_state(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        self._restoring = True
        try:
            source_id = state.get("source_id")
            self._source_id = str(source_id) if source_id else None
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
            plugin_id = state.get("selected_plugin_id")
            if plugin_id is None:
                # Backward compatibility with the previous marketplace panel state.
                plugin_id = state.get("selected_update_id")
            self._selected_plugin_id = str(plugin_id) if plugin_id else None
        finally:
            self._restoring = False
        self.refresh()

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._generation += 1
        for handle in (self._refresh_job, self._install_job):
            if handle is not None:
                try:
                    handle.cancel()
                except Exception:
                    pass
        self._refresh_job = None
        self._install_job = None
        for widget, watcher in list(self._watchers):
            try:
                widget.param.unwatch(watcher)
            except Exception:
                pass
        self._watchers.clear()
        self._card_buttons.clear()

    def _watch(self, widget: Any, callback: Callable[..., Any], attr: str) -> None:
        try:
            watcher = widget.param.watch(callback, attr)
            self._watchers.append((widget, watcher))
        except Exception:
            pass

    @staticmethod
    def _html_pane() -> pn.pane.HTML:
        return pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=0,
        )

    @staticmethod
    def _empty_pane(message: str) -> pn.pane.HTML:
        return pn.pane.HTML(
            f'<div class="al-pm-empty">{html.escape(message)}</div>',
            sizing_mode="stretch_width",
            stylesheets=[PLUGIN_MANAGER_CSS],
            margin=0,
        )

    @staticmethod
    def _banner_html(tone: str, message: str) -> str:
        return (
            f'<div class="al-pm-banner {html.escape(tone)}">'
            f"{html.escape(message)}</div>"
        )