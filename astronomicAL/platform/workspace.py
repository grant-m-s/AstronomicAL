# astronomicAL/platform/workspace.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import panel as pn

from astronomicAL.extensions.dynamic_react_layout import DynamicReactGrid  # PR#23 component :contentReference[oaicite:1]{index=1}


@dataclass
class PanelRecord:
    panel_id: str
    title: str
    view: Any              # the thing in grid.objects (pn viewable)
    controller: Any = None # optional controller with dispose()


class WorkspaceManager:
    """
    Thin wrapper over ReactTemplate + DynamicReactGrid.
    Owns add/remove panel logic and the close-key watcher.
    """

    def __init__(self, react_template: pn.template.ReactTemplate, grid: DynamicReactGrid) -> None:
        self.react = react_template
        self.grid = grid
        self._panels: Dict[str, PanelRecord] = {}

        # IMPORTANT: normalize keys/layout IDs so close_key matches
        self._normalize_grid_state()

        self._ensure_close_watcher()

    @staticmethod
    def _sid(x: Any) -> str:
        """Normalize any panel id/key to a string."""
        return str(x)

    def _sync_grid(self) -> None:
        """Ensure WorkspaceManager is operating on the live grid attached to the template."""
        live = getattr(self.react, "_dynamic_grid", None)
        if live is not None and live is not self.grid:
            # swap to the live one
            self.grid = live
            # reattach close watcher on the new grid
            self._ensure_close_watcher()


    def _normalize_grid_state(self) -> None:
        keys = [str(k) for k in (self.grid.keys or [])]

        layouts = dict(self.grid.layouts or {})
        for bp, bp_layout in list(layouts.items()):
            new_bp = []
            for it in (bp_layout or []):
                if it is None:
                    continue
                it2 = dict(it)
                if "i" in it2:
                    it2["i"] = str(it2["i"])
                new_bp.append(it2)
            layouts[bp] = new_bp

        if keys != (self.grid.keys or []) or layouts != (self.grid.layouts or {}):
            self.grid.param.update(keys=keys, layouts=layouts)


    def _ensure_close_watcher(self) -> None:
        self._sync_grid()

        if getattr(self.grid, "_close_watcher_attached", False):
            return

        def _close_from_js(event) -> None:
            tile_id = event.new
            if not tile_id:
                return
            self.remove_panel(str(tile_id))

        self.grid.param.watch(_close_from_js, "close_key")
        self.grid._close_watcher_attached = True

    def register_existing(self) -> None:
        """
        Optional helper: if grid already populated externally,
        call this to record panel ids for tracking.
        """
        self._sync_grid()
        self._normalize_grid_state()

        for pid, view in zip(self.grid.keys or [], self.grid.objects or []):
            spid = str(pid)
            if spid not in self._panels:
                ctrl = getattr(view, "_al_controller", None)
                self._panels[spid] = PanelRecord(panel_id=spid, title=spid, view=view, controller=ctrl)

    def add_panel(self, panel_id, view, *, title=None, controller=None, layout_item=None, layout_items=None) -> None:
        self._sync_grid()
        self._normalize_grid_state()

        panel_id = str(panel_id)
        if title is None:
            title = panel_id

        self._panels[panel_id] = PanelRecord(panel_id=panel_id, title=title, view=view, controller=controller)

        new_keys = [*list(self.grid.keys or []), panel_id]
        new_objs = [*list(self.grid.objects or []), view]
        new_layouts = {**(self.grid.layouts or {})}

        if layout_items is not None:
            for bp, item in layout_items.items():
                bp_layout = list(new_layouts.get(bp, []))
                it = dict(item)
                it["i"] = panel_id
                bp_layout.append(it)
                new_layouts[bp] = bp_layout

        elif layout_item is not None:
            bps = list(new_layouts.keys()) or ["lg", "md", "sm"]
            for bp in bps:
                bp_layout = list(new_layouts.get(bp, []))
                it = dict(layout_item)
                it["i"] = panel_id
                bp_layout.append(it)
                new_layouts[bp] = bp_layout

        self.grid.param.update(keys=new_keys, objects=new_objs, layouts=new_layouts)

    def remove_panel(self, panel_id: str) -> None:
        self._sync_grid()
        self._normalize_grid_state()

        panel_id = str(panel_id)
        if panel_id not in (self.grid.keys or []):
            try:
                self.grid.close_key = ""
            except Exception:
                pass
            self._panels.pop(panel_id, None)
            return

        idx = list(self.grid.keys).index(panel_id)

        rec = self._panels.get(panel_id)
        view = rec.view if rec else None
        controller = rec.controller if rec else None

        # If controller not explicitly stored, try to recover it from the view
        if controller is None and view is not None:
            controller = getattr(view, "_al_controller", None)

        target = controller if controller is not None else view

        if target is not None and hasattr(target, "dispose"):
            try:
                target.dispose()
            except Exception:
                pass

        # Remove from layouts
        new_layouts = {}
        for bp, bp_layout in (self.grid.layouts or {}).items():
            new_layouts[bp] = [it for it in (bp_layout or []) if str(it.get("i")) != panel_id]

        self._panels.pop(panel_id, None)

        self.grid.param.update(
            keys=[k for k in self.grid.keys if k != panel_id],
            objects=[o for i, o in enumerate(self.grid.objects) if i != idx],
            layouts=new_layouts,
            close_key="",
        )

    def list_panels(self) -> Dict[str, PanelRecord]:
        return dict(self._panels)