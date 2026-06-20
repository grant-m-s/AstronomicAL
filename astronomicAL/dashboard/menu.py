from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import html
import json
import traceback
import uuid

import panel as pn
from bokeh.models import TextInput

from astronomicAL.extensions import custom_plots, extension_plots


@dataclass(frozen=True)
class MenuEntry:
    """Small normalized entry for the hierarchical Add Panel menu."""

    title: str
    value: str
    source: str
    domain: str
    category: str
    plugin_id: Optional[str] = None
    plugin_name: Optional[str] = None


class MenuDashboard:
    """Hierarchical 'Add Panel' menu.

    The popover lives INSIDE the menu root as an absolutely-positioned child of
    the trigger button. It therefore moves with the button when the ReactGrid
    reflows the tile -- there is no body portal, no manual positioning, no
    backdrop, and no host registry to lose. ReactGrid clipping is handled by
    temporarily setting overflow:visible up the ancestor chain while the menu is
    open, and by raising the enclosing grid item's z-index.
    """

    NATIVE_ENTRIES: Tuple[MenuEntry, ...] = ()

    LEGACY_HINTS: Dict[str, Tuple[str, str]] = {
        "SAMP Send": ("Astro", "Interop"),
        "SAMP Receive": ("Astro", "Interop"),
    }

    DOMAIN_ORDER: Dict[str, int] = {
        "Core": 0,
        "Active Learning": 1,
        "ML": 2,
        "Astro": 3,
        "Integration": 4,
        "Integrations": 4,
        "Extensions": 8,
        "Legacy": 9,
        "Plugins": 10,
        "User Plugins": 11,
        "Other": 99,
    }

    def __init__(self, main, context=None):
        self.main = main
        self.context = context if context is not None else getattr(main, "context", None)

        self._disposed = False
        self._subscriptions: List[Any] = []

        self._token = uuid.uuid4().hex[:10]
        self._root_class = f"al-hmenu-root-{self._token}"
        self._target_name = f"al_hmenu_target_{self._token}"

        self._target = TextInput(
            name=self._target_name,
            value="",
            visible=False,
            width=1,
            height=1,
        )
        self._target.on_change("value", self._on_target_changed)

        self._html = self._make_html_pane()

        self._subscribe_to_plugin_events()
        self._refresh_menu()

    # ------------------------------------------------------------------
    # Dashboard API
    # ------------------------------------------------------------------

    def get_toolbar(self):
        return pn.Spacer(height=1, min_height=1, max_height=1)

    def panel(self):
        return pn.Column(
            self._target,
            self._html,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

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

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _subscribe_to_plugin_events(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        for topic in ("plugin.enabled", "plugin.disabled", "plugin.reloaded"):
            try:
                sub = events.subscribe(
                    topic,
                    self._on_plugin_registry_changed,
                    owner_id=f"menu.dashboard.{self._token}",
                    owner_label="Hierarchical Panel Menu",
                    owner_kind="dashboard",
                )
            except TypeError:
                sub = events.subscribe(topic, self._on_plugin_registry_changed)
            self._subscriptions.append(sub)

    def _on_plugin_registry_changed(self, topic: str, payload: Any) -> None:
        if self._disposed:
            return

        def _refresh() -> None:
            if not self._disposed:
                self._refresh_menu()

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(_refresh)
            else:
                _refresh()
        except Exception:
            _refresh()

    def _workspace_grid(self):
        workspace = getattr(self.context, "workspace", None)
        if workspace is None:
            return None
        return getattr(workspace, "grid", None)

    def _current_workspace_panel_id(self) -> Optional[str]:
        panel_id = getattr(self.main, "_al_panel_id", None)
        if panel_id:
            return str(panel_id)

        workspace = getattr(self.context, "workspace", None)
        if workspace is None:
            return None
        try:
            for candidate_id, record in workspace.list_panels().items():
                if record.controller is self.main:
                    return str(candidate_id)
        except Exception:
            return None
        return None

    def _layout_items_for_existing_tile(self, panel_id: str) -> Dict[str, Dict[str, Any]]:
        grid = self._workspace_grid()
        if grid is None:
            return {}

        layout_items: Dict[str, Dict[str, Any]] = {}
        for breakpoint, breakpoint_layout in (getattr(grid, "layouts", None) or {}).items():
            for item in breakpoint_layout or []:
                if str(item.get("i")) != str(panel_id):
                    continue
                item_copy = dict(item)
                item_copy["i"] = str(panel_id)
                layout_items[str(breakpoint)] = item_copy
                break
        return layout_items

    @staticmethod
    def _rects_overlap(a, b) -> bool:
        return not (
            a["x"] + a["w"] <= b["x"]
            or b["x"] + b["w"] <= a["x"]
            or a["y"] + a["h"] <= b["y"]
            or b["y"] + b["h"] <= a["y"]
        )

    def _find_first_fit(self, layout_items, *, cols: int, w: int, h: int):
        items = []
        for item in layout_items or []:
            try:
                items.append(
                    {
                        "x": int(item.get("x", 0)),
                        "y": int(item.get("y", 0)),
                        "w": int(item.get("w", 1)),
                        "h": int(item.get("h", 1)),
                    }
                )
            except Exception:
                continue

        max_y = 0
        for item in items:
            max_y = max(max_y, item["y"] + item["h"])

        for y in range(0, max_y + 100):
            for x in range(0, max(1, cols - w + 1)):
                candidate = {"x": x, "y": y, "w": w, "h": h}
                if not any(self._rects_overlap(candidate, item) for item in items):
                    return x, y
        return 0, max_y

    def _new_panel_layout_items(self, *, default_w=6, default_h=8):
        grid = self._workspace_grid()
        if grid is None:
            return None

        layouts = dict(getattr(grid, "layouts", None) or {})
        cols_by_breakpoint = dict(
            getattr(grid, "cols_by_breakpoint", None) or {"lg": 12, "md": 12, "sm": 12}
        )

        layout_items = {}
        for breakpoint, cols in cols_by_breakpoint.items():
            cols = int(cols)
            w = min(int(default_w), cols)
            h = int(default_h)
            if breakpoint == "sm":
                w = cols
            existing = list(layouts.get(breakpoint, []))
            x, y = self._find_first_fit(existing, cols=cols, w=w, h=h)
            layout_items[breakpoint] = {"x": x, "y": y, "w": w, "h": h}
        return layout_items

    def _open_plugin_panel(self, registration_id: str) -> None:
        manager = getattr(self.context, "plugins", None)
        if manager is None:
            print("[MenuDashboard] Cannot open plugin panel: context.plugins is unavailable")
            return

        current_panel_id = self._current_workspace_panel_id()
        if current_panel_id:
            layout_items = self._layout_items_for_existing_tile(current_panel_id)
            print(
                "[MenuDashboard] replacing menu tile "
                f"panel_id={current_panel_id} with plugin panel "
                f"registration_id={registration_id}"
            )
            manager.open_panel(
                registration_id,
                context=self.context,
                instance_id=current_panel_id,
                layout_items=layout_items,
            )
            return

        print(
            "[MenuDashboard] menu tile id unavailable; opening plugin panel "
            f"as a new tile registration_id={registration_id}"
        )
        manager.open_panel(
            registration_id,
            context=self.context,
            layout_items=self._new_panel_layout_items(default_w=6, default_h=8),
        )

    def _on_target_changed(self, attr: str, old: str, new: str) -> None:
        if not new:
            return

        try:
            payload = json.loads(new)
            value = payload.get("value", "")
        except Exception:
            value = new

        try:
            self._target.value = ""
        except Exception:
            pass

        if not value:
            return

        if value == "__refresh__":
            self._refresh_menu()
            return

        try:
            if value.startswith("plugin:"):
                registration_id = value.split("plugin:", 1)[1]
                self._open_plugin_panel(registration_id)
                return
            self.main.set_contents(value)
        except Exception:
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _make_html_pane(self) -> pn.pane.HTML:
        try:
            return pn.pane.HTML(
                "",
                sanitize_html=False,
                sizing_mode="stretch_width",
                margin=(6, 8, 6, 8),
            )
        except TypeError:
            return pn.pane.HTML(
                "",
                sizing_mode="stretch_width",
                margin=(6, 8, 6, 8),
            )

    def _refresh_menu(self) -> None:
        entries = self._build_entries()
        self._html.object = self._render_html(entries)

    def _build_entries(self) -> List[MenuEntry]:
        entries: List[MenuEntry] = list(self.NATIVE_ENTRIES)

        plugin_entries = self._plugin_entries()
        plugin_titles = {entry.title for entry in plugin_entries}

        try:
            legacy_dict = custom_plots.get_customplot_dict(context=self.context)
        except Exception:
            traceback.print_exc()
            legacy_dict = {}

        for title in legacy_dict.keys():
            if title in plugin_titles:
                continue
            domain, category = self.LEGACY_HINTS.get(title, ("Legacy", "Custom Panels"))
            entries.append(
                MenuEntry(
                    title=title,
                    value=title,
                    source="Legacy",
                    domain=domain,
                    category=category,
                )
            )

        try:
            extension_plots.get_plot_dict()
        except Exception:
            traceback.print_exc()

        entries.extend(plugin_entries)

        seen = set()
        deduped: List[MenuEntry] = []
        for entry in entries:
            key = (entry.value, entry.source)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)

        return sorted(
            deduped,
            key=lambda entry: (
                self.DOMAIN_ORDER.get(entry.domain, 50),
                entry.domain.casefold(),
                entry.category.casefold(),
                entry.title.casefold(),
            ),
        )

    def _plugin_entries(self) -> List[MenuEntry]:
        manager = getattr(self.context, "plugins", None)
        if manager is None:
            return []

        plugin_info_by_id: Dict[str, Any] = {}
        try:
            for info in manager.list_plugins():
                plugin_info_by_id[getattr(info, "id", "")] = info
        except Exception:
            plugin_info_by_id = {}

        panel_regs: List[Any] = []
        try:
            panel_regs.extend(list(manager.list_panels()))
        except Exception:
            traceback.print_exc()

        try:
            raw_panels = getattr(manager, "_panels", {}) or {}
            for reg in raw_panels.values():
                if reg not in panel_regs:
                    panel_regs.append(reg)
        except Exception:
            pass

        deduped_regs: List[Any] = []
        seen_ids = set()
        for reg in panel_regs:
            reg_id = getattr(reg, "id", None)
            if not reg_id or reg_id in seen_ids:
                continue
            seen_ids.add(reg_id)
            deduped_regs.append(reg)

        entries: List[MenuEntry] = []
        for reg in deduped_regs:
            plugin_id = getattr(reg, "plugin_id", "") or ""
            info = plugin_info_by_id.get(plugin_id)
            title = getattr(reg, "title", None) or getattr(reg, "id", "Plugin Panel")
            category = getattr(reg, "category", None) or "Panels"
            registration_id = getattr(reg, "id", "") or title
            entries.append(
                MenuEntry(
                    title=title,
                    value=f"plugin:{registration_id}",
                    source="Plugin",
                    domain=self._domain_for_plugin(reg, info),
                    category=category,
                    plugin_id=plugin_id,
                    plugin_name=getattr(info, "name", None) if info is not None else plugin_id,
                )
            )
        return entries

    def _domain_for_plugin(self, reg: Any, info: Any) -> str:
        plugin_id = (getattr(reg, "plugin_id", "") or "").lower()
        tags = {str(t).lower() for t in (getattr(reg, "tags", None) or [])}
        capabilities = {str(c).lower() for c in (getattr(info, "capabilities", None) or [])}
        required_mappings = {str(m).lower() for m in (getattr(reg, "required_mappings", None) or [])}

        prefix = plugin_id.split(".", 1)[0] if plugin_id else ""

        if prefix == "core":
            return "Core"
        if prefix in {"astro", "astronomy"}:
            return "Astro"
        if prefix in {"ml", "model", "models"}:
            return "ML"
        if prefix in {"active", "active_learning", "al"}:
            return "Active Learning"
        if prefix in {"user", "local"}:
            return "User Plugins"

        if {"astro", "astronomy"} & tags:
            return "Astro"
        if {"ra", "dec"} <= required_mappings:
            return "Astro"
        if {"active-learning", "active_learning", "labelling", "labeling"} & tags:
            return "Active Learning"
        if {"ml", "model", "classifier"} & tags:
            return "ML"
        if {"diagnostics", "debug"} & tags or "diagnostics" in capabilities:
            return "Core"
        if prefix in {"integration", "integrations"}:
            return "Integration"
        if prefix:
            return prefix.replace("_", " ").replace("-", " ").title()
        return "Plugins"

    def _render_html(self, entries: List[MenuEntry]) -> str:
        grouped: Dict[str, Dict[str, List[MenuEntry]]] = {}
        for entry in entries:
            grouped.setdefault(entry.domain, {}).setdefault(entry.category, []).append(entry)

        domain_names = sorted(
            grouped.keys(),
            key=lambda d: (self.DOMAIN_ORDER.get(d, 50), d.casefold()),
        )

        menu_items = []
        for domain in domain_names:
            category_html = []
            for category in sorted(grouped[domain].keys(), key=str.casefold):
                panel_html = [self._render_leaf(e) for e in grouped[domain][category]]
                category_html.append(
                    f"""
                    <li class="al-hmenu-item al-hmenu-has-submenu">
                        <div class="al-hmenu-row">
                            <span>{self._esc(category)}</span>
                            <span class="al-hmenu-arrow">&rsaquo;</span>
                        </div>
                        <ul class="al-hmenu-menu al-hmenu-submenu">
                            {''.join(panel_html)}
                        </ul>
                    </li>
                    """
                )
            menu_items.append(
                f"""
                <li class="al-hmenu-item al-hmenu-has-submenu">
                    <div class="al-hmenu-row">
                        <span>{self._esc(domain)}</span>
                        <span class="al-hmenu-arrow">&rsaquo;</span>
                    </div>
                    <ul class="al-hmenu-menu al-hmenu-submenu">
                        {''.join(category_html)}
                    </ul>
                </li>
                """
            )

        total = len(entries)

        popover = (
            f'<ul class="al-hmenu-menu al-hmenu-body-popover" data-menu-token="{self._token}">'
            f"{''.join(menu_items) if menu_items else '<li class=\"al-hmenu-empty\">No panels available</li>'}"
            f"</ul>"
        )

        return f"""
        <style>
            .{self._root_class},
            .{self._root_class} * {{ box-sizing: border-box; }}

            .{self._root_class} {{
                position: relative;
                display: inline-block;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                font-size: 13px;
                line-height: 1.25;
                color: #222;
                overflow: visible;
            }}

            .{self._root_class} .al-hmenu-title {{
                font-size: 14px;
                font-weight: 600;
                margin: 0 0 6px 0;
            }}

            .{self._root_class} .al-hmenu-bar {{
                display: flex;
                align-items: center;
                gap: 6px;
                overflow: visible;
            }}

            .{self._root_class} .al-hmenu-anchor {{
                position: relative;
                display: inline-block;
            }}

            .{self._root_class} .al-hmenu-button,
            .{self._root_class} .al-hmenu-refresh {{
                border: 1px solid #b8b8b8;
                background: #f7f7f7;
                border-radius: 4px;
                min-height: 28px;
                padding: 4px 10px;
                color: #222;
                cursor: default;
                user-select: none;
                font-size: 13px;
            }}

            .{self._root_class} .al-hmenu-button:hover,
            .{self._root_class}.al-hmenu-open .al-hmenu-button,
            .{self._root_class} .al-hmenu-refresh:hover {{
                background: #ececec;
                border-color: #999;
            }}

            .{self._root_class} .al-hmenu-count {{
                color: #666;
                font-size: 12px;
                margin-left: 2px;
                white-space: nowrap;
            }}

            /* ---- in-root popover ---- */
            .al-hmenu-body-popover,
            .al-hmenu-body-popover * {{ box-sizing: border-box; }}

            .al-hmenu-body-popover {{
                display: none;
                position: absolute;
                top: calc(100% + 3px);
                left: 0;

                /*
                 * Keep the Add Panel menu above normal dashboard/grid content,
                 * but below Panel/Bootstrap modal/dialog layers.
                 *
                 * Bootstrap modal backdrop/modal commonly occupy ~1040/1050.
                 * The previous implementation lifted ancestors to 2147483000,
                 * which allowed this menu to render above active modals.
                 */
                z-index: 900;

                list-style: none;
                margin: 0;
                padding: 4px 0;
                min-width: 210px;
                background: #fff;
                border: 1px solid rgba(0,0,0,0.24);
                border-radius: 4px;
                box-shadow: 0 6px 18px rgba(0,0,0,0.18);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                font-size: 13px;
                line-height: 1.25;
                color: #222;
            }}

            .al-hmenu-body-popover .al-hmenu-menu {{
                list-style: none;
                margin: 0;
                padding: 4px 0;
                min-width: 210px;
                background: #fff;
                border: 1px solid rgba(0,0,0,0.24);
                border-radius: 4px;
                box-shadow: 0 6px 18px rgba(0,0,0,0.18);
            }}

            .al-hmenu-body-popover .al-hmenu-item {{
                position: relative;
                margin: 0;
                padding: 0;
                min-height: 26px;
                white-space: nowrap;
                list-style: none;
            }}

            .al-hmenu-body-popover .al-hmenu-row {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 18px;
                padding: 6px 10px;
                min-height: 26px;
                cursor: default;
            }}

            .al-hmenu-body-popover .al-hmenu-item:hover > .al-hmenu-row {{
                background: #e9f2ff;
            }}

            .al-hmenu-body-popover .al-hmenu-submenu {{
                display: none;
                position: absolute;
                top: -5px;
                left: calc(100% - 1px);
                margin: 0;
                z-index: 901;
            }}

            .al-hmenu-body-popover .al-hmenu-item:hover > .al-hmenu-submenu,
            .al-hmenu-body-popover .al-hmenu-item:focus-within > .al-hmenu-submenu {{
                display: block;
            }}

            .al-hmenu-body-popover .al-hmenu-has-submenu.al-hmenu-flip > .al-hmenu-submenu {{
                left: auto;
                right: calc(100% - 1px);
            }}

            .al-hmenu-body-popover .al-hmenu-arrow {{
                color: #777;
                font-size: 16px;
                line-height: 1;
            }}

            .al-hmenu-body-popover .al-hmenu-leaf-button {{
                appearance: none;
                border: 0;
                background: transparent;
                width: 100%;
                min-height: 26px;
                padding: 6px 10px;
                text-align: left;
                color: #222;
                font: inherit;
                cursor: default;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 14px;
            }}

            .al-hmenu-body-popover .al-hmenu-leaf-button:hover,
            .al-hmenu-body-popover .al-hmenu-leaf-button:focus {{
                outline: none;
                background: #e9f2ff;
            }}

            .al-hmenu-body-popover .al-hmenu-source {{
                color: #777;
                font-size: 11px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            }}

            .al-hmenu-body-popover .al-hmenu-empty {{
                padding: 8px 10px;
                color: #777;
            }}
        </style>

        <div class="al-hmenu-root {self._root_class}" data-menu-token="{self._token}">
            <div class="al-hmenu-title">Add Panel</div>

            <div class="al-hmenu-bar">
                <div class="al-hmenu-anchor">
                    <button type="button" class="al-hmenu-button" onclick="{self._toggle_js()}">
                        Panel &#9662;
                    </button>
                    {popover}
                </div>

                <button
                    type="button"
                    class="al-hmenu-refresh"
                    title="Refresh panel list"
                    data-value="__refresh__"
                    onclick="{self._click_js()}"
                >
                    &#8635;
                </button>

                <span class="al-hmenu-count">{total} available</span>
            </div>
        </div>
        """

    def _render_leaf(self, entry: MenuEntry) -> str:
        value = self._attr(entry.value)
        title = self._esc(entry.title)
        source = self._esc(entry.source)

        plugin_title = ""
        if entry.source == "Plugin":
            plugin_label = entry.plugin_name or entry.plugin_id or "Plugin"
            plugin_title = f' title="{self._attr(plugin_label)}"'

        return f"""
        <li class="al-hmenu-item">
            <button
                type="button"
                class="al-hmenu-leaf-button"
                data-value="{value}"
                {plugin_title}
                onclick="{self._click_js()}"
            >
                <span>{title}</span>
                <span class="al-hmenu-source">{source}</span>
            </button>
        </li>
        """

    def _toggle_js(self) -> str:
        js = """
        event.preventDefault();
        event.stopPropagation();

        const root = this.closest('.al-hmenu-root');
        if (!root) { return; }

        const popover = root.querySelector('.al-hmenu-body-popover');
        if (!popover) { return; }

        // These values intentionally sit below modal/dialog layers.
        // Bootstrap/Panel modal stacks commonly use backdrop/modal around
        // 1040/1050+. The menu only needs to beat ReactGrid tile stacking.
        const AL_HMENU_TILE_Z = '900';
        const AL_HMENU_POPOVER_Z = '901';

        function alModalIsOpen() {
            const selectors = [
                '.modal.show',
                '.modal.in',
                '.bk-modal',
                '.bk-dialog',
                '.bk-Dialog',
                '.pn-modal',
                '.pn-modal-content',
                '.modal-backdrop',
                '.modal-backdrop.show',
                '[role="dialog"][aria-modal="true"]',
                '[aria-modal="true"]'
            ];

            for (let i = 0; i < selectors.length; i++) {
                const nodes = document.querySelectorAll(selectors[i]);
                for (let j = 0; j < nodes.length; j++) {
                    const el = nodes[j];
                    const cs = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    if (
                        cs.display !== 'none' &&
                        cs.visibility !== 'hidden' &&
                        rect.width > 0 &&
                        rect.height > 0
                    ) {
                        return true;
                    }
                }
            }

            return false;
        }

        // Walk ancestors INCLUDING across shadow-DOM boundaries. BokehJS renders
        // widget/HTML-pane content inside a shadow root, so a plain parentElement
        // walk stops at the shadow host and never reaches the ReactGrid tile that
        // actually clips the deep submenus.
        function alParent(node) {
            if (!node) { return null; }
            if (node.parentElement) { return node.parentElement; }
            const r = (node.getRootNode && node.getRootNode());
            if (r && r.host) { return r.host; }
            const pn = node.parentNode;
            if (pn && pn.host) { return pn.host; }
            return null;
        }

        function restoreFor(pop) {
            const fixes = pop.__alFixes;
            if (fixes) {
                for (let i = 0; i < fixes.length; i++) {
                    const f = fixes[i];
                    try {
                        f.el.style.overflow = f.overflow;
                        f.el.style.overflowX = f.overflowX;
                        f.el.style.overflowY = f.overflowY;
                    } catch (e) {}
                }
                pop.__alFixes = null;
            }

            const zlifts = pop.__alZLifts;
            if (zlifts) {
                for (let i = 0; i < zlifts.length; i++) {
                    const z = zlifts[i];
                    try {
                        z.el.style.zIndex = z.zIndex;
                        z.el.style.position = z.position;
                    } catch (e) {}
                }
                pop.__alZLifts = null;
            }
        }

        function closeMenu(pop) {
            restoreFor(pop);
            pop.style.display = 'none';
            const r = pop.closest('.al-hmenu-root');
            if (r) { r.classList.remove('al-hmenu-open'); }
        }

        function closeAllMenus() {
            document.querySelectorAll('.al-hmenu-body-popover').forEach(function(pop) {
                if (pop.style.display === 'block') { closeMenu(pop); }
            });
        }

        function openMenu(pop, r) {
            // Never open the dashboard menu over an active modal/dialog.
            if (alModalIsOpen()) {
                closeAllMenus();
                return;
            }

            // Lift overflow clipping up the ancestor chain, crossing shadow
            // boundaries, so the in-root popover and nested submenus are not
            // cropped by the ReactGrid tile, shadow host, or pure clipping
            // containers.
            //
            // IMPORTANT: skip ancestors that actually own a scroll position.
            // Forcing overflow:visible on a scrolled element resets scrollTop /
            // scrollLeft to 0.
            const fixes = [];
            let node = alParent(pop);
            while (node && node !== document.documentElement && node !== document.body) {
                const cs = window.getComputedStyle(node);
                const clips =
                    cs.overflow !== 'visible' ||
                    cs.overflowX !== 'visible' ||
                    cs.overflowY !== 'visible';

                const scrollableY =
                    (cs.overflowY === 'auto' || cs.overflowY === 'scroll' ||
                     cs.overflow  === 'auto' || cs.overflow  === 'scroll') &&
                    node.scrollHeight > node.clientHeight + 1;

                const scrollableX =
                    (cs.overflowX === 'auto' || cs.overflowX === 'scroll' ||
                     cs.overflow  === 'auto' || cs.overflow  === 'scroll') &&
                    node.scrollWidth > node.clientWidth + 1;

                if (clips && !scrollableY && !scrollableX) {
                    fixes.push({
                        el: node,
                        overflow: node.style.overflow,
                        overflowX: node.style.overflowX,
                        overflowY: node.style.overflowY
                    });
                    node.style.overflow = 'visible';
                    node.style.overflowX = 'visible';
                    node.style.overflowY = 'visible';
                }

                node = alParent(node);
            }
            pop.__alFixes = fixes;

            /*
             * Raise only the local dashboard/grid ancestor stack, and cap it
             * below modal layers. The old value, 2147483000, caused the Add
             * Panel menu to render above active modals.
             */
            const zlifts = [];
            let zn = alParent(pop);
            while (zn && zn !== document.body && zn !== document.documentElement) {
                zlifts.push({
                    el: zn,
                    zIndex: zn.style.zIndex,
                    position: zn.style.position
                });

                const zcs = window.getComputedStyle(zn);
                if (zcs.position === 'static') {
                    zn.style.position = 'relative';
                }

                zn.style.zIndex = AL_HMENU_TILE_Z;
                zn = alParent(zn);
            }
            pop.__alZLifts = zlifts;

            pop.style.zIndex = AL_HMENU_POPOVER_Z;
            pop.style.display = 'block';

            r.classList.add('al-hmenu-open');

            window.requestAnimationFrame(function() {
                if (alModalIsOpen()) {
                    closeMenu(pop);
                    return;
                }

                const pr = pop.getBoundingClientRect();
                const anchor = pop.parentElement;
                const ar = anchor ? anchor.getBoundingClientRect() : null;

                // If the menu is pushed off the right edge, flip the root menu.
                if (pr.right > window.innerWidth - 8 && anchor) {
                    pop.style.left = 'auto';
                    pop.style.right = '0';
                } else {
                    pop.style.left = '0';
                    pop.style.right = 'auto';
                }

                installSubmenuFlipHandlers(pop);
            });
        }

        function installSubmenuFlipHandlers(scope) {
            const items = scope.querySelectorAll('.al-hmenu-has-submenu');
            for (let i = 0; i < items.length; i++) {
                const item = items[i];
                if (item.__alFlipBound) { continue; }
                item.__alFlipBound = true;

                item.addEventListener('mouseenter', function() {
                    item.classList.remove('al-hmenu-flip');

                    window.requestAnimationFrame(function() {
                        let submenu = null;
                        for (let k = 0; k < item.children.length; k++) {
                            const child = item.children[k];
                            if (
                                child.classList &&
                                child.classList.contains('al-hmenu-submenu')
                            ) {
                                submenu = child;
                                break;
                            }
                        }

                        if (!submenu) { return; }

                        const rect = submenu.getBoundingClientRect();
                        if (rect.right > window.innerWidth - 8) {
                            item.classList.add('al-hmenu-flip');
                        }
                    });
                });
            }
        }

        function installGlobalHandlers() {
            if (window.__alHmenuGlobal) { return; }
            window.__alHmenuGlobal = true;

            document.addEventListener('click', function(evt) {
                if (Date.now() - (window.__alHmenuOpenedAt || 0) < 150) {
                    return;
                }

                if (evt.target.closest && evt.target.closest('.al-hmenu-root')) {
                    return;
                }

                closeAllMenus();
            }, true);

            document.addEventListener('keydown', function(evt) {
                if (evt.key === 'Escape') {
                    closeAllMenus();
                }
            }, true);

            // If a modal/dialog appears after the menu has opened, immediately
            // close and restore the menu stack.
            const observer = new MutationObserver(function() {
                if (alModalIsOpen()) {
                    closeAllMenus();
                }
            });

            try {
                observer.observe(document.body, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    attributeFilter: ['class', 'style', 'aria-hidden', 'aria-modal']
                });
                window.__alHmenuModalObserver = observer;
            } catch (e) {}
        }

        installGlobalHandlers();

        if (popover.style.display === 'block') {
            closeMenu(popover);
            return;
        }

        if (alModalIsOpen()) {
            closeAllMenus();
            return;
        }

        closeAllMenus();
        window.__alHmenuOpenedAt = Date.now();
        openMenu(popover, root);
        """
        return self._attr(js)

    def _click_js(self) -> str:
        target_name = self._js_string(self._target_name)

        js = f"""
        event.preventDefault();
        event.stopPropagation();

        const selectedValue = this.getAttribute('data-value') || '';

        const payload = JSON.stringify({{
            value: selectedValue,
            event_id: Date.now().toString() + ':' + Math.random().toString()
        }});

        const targetName = {target_name};
        let updated = false;

        if (window.Bokeh && Array.isArray(window.Bokeh.documents)) {{
            for (const doc of window.Bokeh.documents) {{
                if (!doc) {{ continue; }}
                let model = null;
                if (typeof doc.get_model_by_name === 'function') {{
                    model = doc.get_model_by_name(targetName);
                }}
                if (!model && doc._all_models) {{
                    const models = doc._all_models;
                    const values = typeof models.values === 'function'
                        ? Array.from(models.values())
                        : Object.values(models);
                    for (const candidate of values) {{
                        if (candidate && candidate.name === targetName) {{
                            model = candidate;
                            break;
                        }}
                    }}
                }}
                if (model) {{
                    if (typeof model.setv === 'function') {{
                        model.setv({{ value: payload }});
                    }} else {{
                        model.value = payload;
                    }}
                    if (model.change && typeof model.change.emit === 'function') {{
                        model.change.emit();
                    }}
                    updated = true;
                    break;
                }}
            }}
        }}

        if (!updated) {{
            const selector =
                'input[name="' + targetName + '"], ' +
                '[name="' + targetName + '"] input, ' +
                '[name="' + targetName + '"] textarea';
            const target = document.querySelector(selector);
            if (target) {{
                target.value = payload;
                target.dispatchEvent(new Event('input', {{ bubbles: true }}));
                target.dispatchEvent(new Event('change', {{ bubbles: true }}));
                updated = true;
            }}
        }}

        // Close this menu and restore the overflow/z-index fixes it applied.
        const pop = this.closest('.al-hmenu-body-popover');
        if (pop) {{
            const fixes = pop.__alFixes;
            if (fixes) {{
                for (let i = 0; i < fixes.length; i++) {{
                    const f = fixes[i];
                    try {{
                        f.el.style.overflow = f.overflow;
                        f.el.style.overflowX = f.overflowX;
                        f.el.style.overflowY = f.overflowY;
                    }} catch (e) {{}}
                }}
                pop.__alFixes = null;
            }}
            const zlifts = pop.__alZLifts;
            if (zlifts) {{
                for (let i = 0; i < zlifts.length; i++) {{
                    const z = zlifts[i];
                    try {{
                        z.el.style.zIndex = z.zIndex;
                        z.el.style.position = z.position;
                    }} catch (e) {{}}
                }}
                pop.__alZLifts = null;
            }}
            pop.style.display = 'none';
            const r = pop.closest('.al-hmenu-root');
            if (r) {{ r.classList.remove('al-hmenu-open'); }}
        }}

        if (!updated && window.console) {{
            console.error('AstronomicAL menu could not find hidden Bokeh target model:', targetName);
        }}
        """
        return self._attr(js)

    # ------------------------------------------------------------------
    # Escaping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _esc(value: Any) -> str:
        return html.escape(str(value), quote=False)

    @staticmethod
    def _attr(value: Any) -> str:
        return html.escape(str(value), quote=True)

    @staticmethod
    def _js_string(value: Any) -> str:
        text = str(value)
        text = text.replace("\\", "\\\\")
        text = text.replace("'", "\\'")
        text = text.replace("\n", "\\n")
        text = text.replace("\r", "\\r")
        return f"'{text}'"