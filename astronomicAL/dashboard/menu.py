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
    """Dashboard used to dynamically choose which view to display.

    Compact HTML/CSS hierarchical menu.

    Behaviour:

    - top-level Panel menu opens on click, not hover
    - visible menu is appended to document.body to avoid ReactGrid clipping
    - top-level menu stays open until outside click, Escape, refresh, or selection
    - submenus open on hover
    - selected panels open through Dashboard.set_contents(...)
    - plugin panels continue to open through the current plugin/custom-plot bridge
    """

    NATIVE_ENTRIES: Tuple[MenuEntry, ...] = (
    )

    # Transitional grouping while these panels still live in legacy custom_plots.
    LEGACY_HINTS: Dict[str, Tuple[str, str]] = {
        "Notes Panel": ("Core", "Annotation"),
        "Selection Set": ("Core", "Selection"),

        "Euclid Cutout": ("Astro", "Images / Cutouts"),
        "VLASS Cutout": ("Astro", "Images / Cutouts"),
        "LoTSS Cutout": ("Astro", "Images / Cutouts"),

        "DESI Spectra": ("Astro", "Spectra"),
        "Euclid Spectra": ("Astro", "Spectra"),
        "SDSS Spectra": ("Astro", "Spectra"),
        "spec_analyser": ("Astro", "Spectra"),

        "BroadBand SED": ("Astro", "SED / Photometry"),
        "Aladin Lite": ("Astro", "Sky Viewers"),

        "SAMP Send": ("Astro", "Interop"),
        "SAMP Receive": ("Astro", "Interop"),
    }

    DOMAIN_ORDER: Dict[str, int] = {
        "Core": 0,
        "Active Learning": 1,
        "ML": 2,
        "Astro": 3,
        "Integration": 4,
        "Integrations": 4,  # optional backward-compatible alias
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

        # Real Bokeh model name. JS finds this through Bokeh.documents and
        # updates its value. Python receives the change through on_change.
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
        """
        Return the workspace tile id that contains this MenuDashboard.

        self.main is the parent Dashboard controller created in add_menu_panel().
        """
        panel_id = getattr(self.main, "_al_panel_id", None)
        if panel_id:
            return str(panel_id)

        # Fallback: try to find this Dashboard controller in workspace records.
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
        """
        Capture the current tile position for every breakpoint.

        The replacement plugin panel will reuse these layout items.
        """
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
        """
        Fallback only.

        Used if the menu is somehow not inside a tracked workspace tile.
        Normal menu behaviour should replace the existing menu tile.
        """
        grid = self._workspace_grid()
        if grid is None:
            return None

        layouts = dict(getattr(grid, "layouts", None) or {})
        cols_by_breakpoint = dict(
            getattr(grid, "cols_by_breakpoint", None)
            or {"lg": 12, "md": 12, "sm": 12}
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

            layout_items[breakpoint] = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }

        return layout_items

    def _open_plugin_panel(self, registration_id: str) -> None:
        """
        Replace this Menu tile with the selected plugin panel.

        This intentionally reuses the menu tile's workspace panel id so the
        selected panel appears exactly where the menu was.
        """
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

        # Defensive fallback. This should rarely happen.
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

        # Reset immediately so choosing the same item again still fires later.
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

            # Non-plugin entries retain the old behaviour for now: replace
            # this Dashboard's internal contents.
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

            domain, category = self.LEGACY_HINTS.get(
                title,
                ("Legacy", "Custom Panels"),
            )

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
            extension_dict = extension_plots.get_plot_dict()
        except Exception:
            traceback.print_exc()
            extension_dict = {}

        for title in extension_dict.keys():
            entries.append(
                MenuEntry(
                    title=title,
                    value=title,
                    source="Extension",
                    domain="Extensions",
                    category="Extension Plots",
                )
            )

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

        # Defensive fallback: if the public list_panels() view misses a panel
        # registration for any reason, also read the registry directly.
        #
        # This is intentionally tolerant because the panel menu is transitional
        # bridge code while plugin-driven menus settle.
        try:
            raw_panels = getattr(manager, "_panels", {}) or {}
            for reg in raw_panels.values():
                if reg not in panel_regs:
                    panel_regs.append(reg)
        except Exception:
            pass

        # Deduplicate by registration id while preserving first-seen order.
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
        capabilities = {
            str(c).lower()
            for c in (getattr(info, "capabilities", None) or [])
        }
        required_mappings = {
            str(m).lower()
            for m in (getattr(reg, "required_mappings", None) or [])
        }

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
            key=lambda d: (
                self.DOMAIN_ORDER.get(d, 50),
                d.casefold(),
            ),
        )

        menu_items = []

        for domain in domain_names:
            category_html = []

            for category in sorted(grouped[domain].keys(), key=str.casefold):
                panel_html = []

                for entry in grouped[domain][category]:
                    panel_html.append(self._render_leaf(entry))

                category_html.append(
                    f"""
                    <li class="al-hmenu-item al-hmenu-has-submenu">
                        <div class="al-hmenu-row">
                            <span>{self._esc(category)}</span>
                            <span class="al-hmenu-arrow">›</span>
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
                        <span class="al-hmenu-arrow">›</span>
                    </div>
                    <ul class="al-hmenu-menu al-hmenu-submenu">
                        {''.join(category_html)}
                    </ul>
                </li>
                """
            )

        total = len(entries)

        menu_markup = f"""
        <ul class="al-hmenu-menu al-hmenu-body-popover" data-menu-token="{self._token}">
            {''.join(menu_items) if menu_items else '<li class="al-hmenu-empty">No panels available</li>'}
        </ul>
        """

        return f"""
        <style>
            .{self._root_class},
            .{self._root_class} * {{
                box-sizing: border-box;
            }}

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
        </style>

        <div
            class="al-hmenu-root {self._root_class}"
            data-menu-token="{self._token}"
            data-menu-html="{self._attr(menu_markup)}"
        >
            <div class="al-hmenu-title">Add Panel</div>

            <div class="al-hmenu-bar">
                <button
                    type="button"
                    class="al-hmenu-button"
                    onclick="{self._toggle_js()}"
                >
                    Panel ▾
                </button>

                <button
                    type="button"
                    class="al-hmenu-refresh"
                    title="Refresh panel list"
                    data-value="__refresh__"
                    onclick="{self._click_js()}"
                >
                    ↻
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
        token = self._js_string(self._token)
        body_css = self._js_string(self._body_menu_css())

        js = f"""
        event.preventDefault();
        event.stopPropagation();

        const token = {token};
        const css = {body_css};
        const root = this.closest('.al-hmenu-root');

        if (!root) {{
            return;
        }}

        function ensureBodyMenuCss() {{
            const styleId = 'astronomical-hmenu-body-css';
            let style = document.getElementById(styleId);

            if (!style) {{
                style = document.createElement('style');
                style.id = styleId;
                document.head.appendChild(style);
            }}

            style.textContent = css;
        }}

        function closeAllMenus() {{
            document.querySelectorAll('.al-hmenu-body-popover').forEach((menu) => {{
                menu.remove();
            }});

            document.querySelectorAll('.al-hmenu-root.al-hmenu-open').forEach((menuRoot) => {{
                menuRoot.classList.remove('al-hmenu-open');
            }});
        }}

        function installGlobalHandlers() {{
            if (window.__astronomicalHMenuBodyHandlersInstalled) {{
                return;
            }}

            window.__astronomicalHMenuBodyHandlersInstalled = true;

            document.addEventListener('click', function(evt) {{
                if (
                    evt.target.closest('.al-hmenu-body-popover') ||
                    evt.target.closest('.al-hmenu-root')
                ) {{
                    return;
                }}

                closeAllMenus();
            }}, true);

            document.addEventListener('keydown', function(evt) {{
                if (evt.key !== 'Escape') {{
                    return;
                }}

                closeAllMenus();
            }}, true);
        }}

        function firstDirectSubmenu(item) {{
            for (let i = 0; i < item.children.length; i++) {{
                const child = item.children[i];
                if (child.classList && child.classList.contains('al-hmenu-submenu')) {{
                    return child;
                }}
            }}

            return null;
        }}

        function installSubmenuFlipHandlers(menu) {{
            const items = menu.querySelectorAll('.al-hmenu-has-submenu');

            for (let i = 0; i < items.length; i++) {{
                const item = items[i];

                item.addEventListener('mouseenter', function() {{
                    item.classList.remove('al-hmenu-flip');

                    window.requestAnimationFrame(function() {{
                        const submenu = firstDirectSubmenu(item);
                        if (!submenu) {{
                            return;
                        }}

                        const rect = submenu.getBoundingClientRect();
                        const pad = 8;

                        if (rect.right > window.innerWidth - pad) {{
                            item.classList.add('al-hmenu-flip');
                        }}
                    }});
                }});
            }}
        }}

        function positionMenu(menu, button) {{
            const rect = button.getBoundingClientRect();
            const pad = 8;

            menu.style.visibility = 'hidden';
            document.body.appendChild(menu);

            const menuRect = menu.getBoundingClientRect();

            let left = rect.left;
            let top = rect.bottom + 3;

            if (left + menuRect.width > window.innerWidth - pad) {{
                left = window.innerWidth - menuRect.width - pad;
            }}

            if (left < pad) {{
                left = pad;
            }}

            if (top + menuRect.height > window.innerHeight - pad) {{
                const above = rect.top - menuRect.height - 3;
                if (above > pad) {{
                    top = above;
                }}
            }}

            menu.style.left = left + 'px';
            menu.style.top = top + 'px';
            menu.style.visibility = '';
        }}

        ensureBodyMenuCss();
        installGlobalHandlers();

        const existing = document.querySelector(
            '.al-hmenu-body-popover[data-menu-token="' + token + '"]'
        );

        if (existing) {{
            closeAllMenus();
            return;
        }}

        closeAllMenus();

        const menuHTML = root.getAttribute('data-menu-html') || '';
        const wrapper = document.createElement('div');
        wrapper.innerHTML = menuHTML.trim();

        const menu = wrapper.querySelector(
            '.al-hmenu-body-popover[data-menu-token="' + token + '"]'
        );

        if (!menu) {{
            return;
        }}

        installSubmenuFlipHandlers(menu);
        positionMenu(menu, this);

        root.classList.add('al-hmenu-open');
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

        /*
         * Bokeh 3.x reliable path: find the hidden Bokeh TextInput model by
         * model.name and set its value directly.
         */
        if (window.Bokeh && Array.isArray(window.Bokeh.documents)) {{
            for (const doc of window.Bokeh.documents) {{
                if (!doc) {{
                    continue;
                }}

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

        /*
         * Fallback for non-shadow DOM render paths.
         */
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

        if (updated) {{
            document.querySelectorAll('.al-hmenu-body-popover').forEach((menu) => {{
                menu.remove();
            }});

            document.querySelectorAll('.al-hmenu-root.al-hmenu-open').forEach((menuRoot) => {{
                menuRoot.classList.remove('al-hmenu-open');
            }});
        }} else if (window.console) {{
            console.error('AstronomicAL menu could not find hidden Bokeh target model:', targetName);
        }}
        """
        return self._attr(js)

    def _body_menu_css(self) -> str:
        return """
        .al-hmenu-body-popover,
        .al-hmenu-body-popover * {
            box-sizing: border-box;
        }

        .al-hmenu-body-popover {
            position: fixed;
            z-index: 2147483000;
            list-style: none;
            margin: 0;
            padding: 4px 0;
            min-width: 210px;
            background: #fff;
            border: 1px solid rgba(0, 0, 0, 0.24);
            border-radius: 4px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
            overflow: visible;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 13px;
            line-height: 1.25;
            color: #222;
        }

        .al-hmenu-body-popover .al-hmenu-menu {
            list-style: none;
            margin: 0;
            padding: 4px 0;
            min-width: 210px;
            background: #fff;
            border: 1px solid rgba(0, 0, 0, 0.24);
            border-radius: 4px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
            overflow: visible;
        }

        .al-hmenu-body-popover .al-hmenu-item {
            position: relative;
            margin: 0;
            padding: 0;
            min-height: 26px;
            white-space: nowrap;
            list-style: none;
        }

        .al-hmenu-body-popover .al-hmenu-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 6px 10px;
            min-height: 26px;
            cursor: default;
        }

        .al-hmenu-body-popover .al-hmenu-item:hover > .al-hmenu-row {
            background: #e9f2ff;
        }

        .al-hmenu-body-popover .al-hmenu-submenu {
            display: none;
            position: absolute;
            top: -5px;
            left: calc(100% - 1px);
            margin: 0;
            z-index: 2147483001;
        }

        .al-hmenu-body-popover .al-hmenu-item:hover > .al-hmenu-submenu,
        .al-hmenu-body-popover .al-hmenu-item:focus-within > .al-hmenu-submenu {
            display: block;
        }

        .al-hmenu-body-popover .al-hmenu-has-submenu.al-hmenu-flip > .al-hmenu-submenu {
            left: auto;
            right: calc(100% - 1px);
        }

        .al-hmenu-body-popover .al-hmenu-arrow {
            color: #777;
            font-size: 16px;
            line-height: 1;
        }

        .al-hmenu-body-popover .al-hmenu-leaf-button {
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
        }

        .al-hmenu-body-popover .al-hmenu-leaf-button:hover,
        .al-hmenu-body-popover .al-hmenu-leaf-button:focus {
            outline: none;
            background: #e9f2ff;
        }

        .al-hmenu-body-popover .al-hmenu-source {
            color: #777;
            font-size: 11px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }

        .al-hmenu-body-popover .al-hmenu-empty {
            padding: 8px 10px;
            color: #777;
        }
        """

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