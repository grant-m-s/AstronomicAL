from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import panel as pn

from astronomicAL.extensions.dynamic_react_layout import DynamicReactGrid
from astronomicAL.platform.panel_state import (
    get_controller_state,
    get_controller_state_version,
)

from astronomicAL.utils.debug import workspace_debug_print


@dataclass
class PanelRecord:
    panel_id: str
    title: str
    view: Any
    controller: Any = None

    kind: str = "plugin_panel"
    plugin_id: Optional[str] = None
    registration_id: Optional[str] = None
    plugin_version: Optional[str] = None

    state_version: int = 1
    persistent: bool = True

    open_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkspaceManager:
    """
    Owns visible panel instances and the DynamicReactGrid state.

    The workspace owns:
    - which panel instances are open
    - where they are in the grid
    - lifecycle cleanup on close/remove
    - the metadata needed for workspace save/load

    Plugin-specific widget state is optional and lives on the controller via
    get_state() / restore_state().
    """

    def __init__(
        self,
        react_template: pn.template.ReactTemplate,
        grid: DynamicReactGrid,
    ) -> None:
        self.react = react_template
        self.grid = grid
        self._panels: Dict[str, PanelRecord] = {}

        self._normalize_grid_state()
        self._ensure_close_watcher()

    @staticmethod
    def _sid(value: Any) -> str:
        return str(value)

    def _sync_grid(self) -> None:
        live = getattr(self.react, "_dynamic_grid", None)
        if live is not None and live is not self.grid:
            self.grid = live
            self._ensure_close_watcher()

    def _normalize_grid_state(self) -> None:
        keys = [str(key) for key in (self.grid.keys or [])]
        layouts = {}

        for breakpoint, breakpoint_layout in dict(self.grid.layouts or {}).items():
            new_breakpoint_layout = []
            for item in breakpoint_layout or []:
                if item is None:
                    continue
                item_copy = dict(item)
                if "i" in item_copy:
                    item_copy["i"] = str(item_copy["i"])
                new_breakpoint_layout.append(item_copy)
            layouts[breakpoint] = new_breakpoint_layout

        if keys != (self.grid.keys or []) or layouts != (self.grid.layouts or {}):
            self.grid.param.update(keys=keys, layouts=layouts)

    def _ensure_close_watcher(self) -> None:
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
        Register panels already present in the grid.

        This is mostly useful during startup or while old code is being removed.
        """
        self._sync_grid()
        self._normalize_grid_state()

        for panel_id, view in zip(self.grid.keys or [], self.grid.objects or []):
            panel_id = str(panel_id)
            if panel_id in self._panels:
                continue

            controller = getattr(view, "_al_controller", None)
            record = PanelRecord(
                panel_id=panel_id,
                title=getattr(view, "_al_title", panel_id),
                view=view,
                controller=controller,
                kind=getattr(view, "_al_kind", "unknown"),
                plugin_id=getattr(view, "_al_plugin_id", None),
                registration_id=getattr(view, "_al_registration_id", None),
                plugin_version=getattr(view, "_al_plugin_version", None),
                state_version=getattr(view, "_al_state_version", 1),
                persistent=getattr(view, "_al_persistent", True),
                open_kwargs=dict(getattr(view, "_al_open_kwargs", {}) or {}),
                metadata=dict(getattr(view, "_al_metadata", {}) or {}),
            )
            self._panels[panel_id] = record

    def add_panel(
        self,
        panel_id: Any,
        view: Any,
        *,
        title: Optional[str] = None,
        controller: Any = None,
        layout_item: Optional[dict[str, Any]] = None,
        layout_items: Optional[dict[str, dict[str, Any]]] = None,
        kind: str = "plugin_panel",
        plugin_id: Optional[str] = None,
        registration_id: Optional[str] = None,
        plugin_version: Optional[str] = None,
        state_version: int = 1,
        persistent: bool = True,
        open_kwargs: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._sync_grid()
        self._normalize_grid_state()

        panel_id = str(panel_id)
        title = title or panel_id

        if panel_id in self._panels or panel_id in [str(k) for k in (self.grid.keys or [])]:
            self.remove_panel(panel_id)

        record = PanelRecord(
            panel_id=panel_id,
            title=title,
            view=view,
            controller=controller,
            kind=kind,
            plugin_id=plugin_id,
            registration_id=registration_id,
            plugin_version=plugin_version,
            state_version=state_version,
            persistent=persistent,
            open_kwargs=dict(open_kwargs or {}),
            metadata=dict(metadata or {}),
        )
        self._panels[panel_id] = record

        self._attach_metadata_to_view(record)

        new_keys = [*list(self.grid.keys or []), panel_id]
        new_objects = [*list(self.grid.objects or []), view]
        new_layouts = deepcopy(dict(self.grid.layouts or {}))

        if layout_items is not None:
            for breakpoint, item in layout_items.items():
                breakpoint_layout = list(new_layouts.get(breakpoint, []))
                item_copy = dict(item)
                item_copy["i"] = panel_id
                breakpoint_layout = [
                    existing
                    for existing in breakpoint_layout
                    if str(existing.get("i")) != panel_id
                ]
                breakpoint_layout.append(item_copy)
                new_layouts[breakpoint] = breakpoint_layout
        elif layout_item is not None:
            breakpoints = list(new_layouts.keys()) or ["lg", "md", "sm"]
            for breakpoint in breakpoints:
                breakpoint_layout = list(new_layouts.get(breakpoint, []))
                item_copy = dict(layout_item)
                item_copy["i"] = panel_id
                breakpoint_layout = [
                    existing
                    for existing in breakpoint_layout
                    if str(existing.get("i")) != panel_id
                ]
                breakpoint_layout.append(item_copy)
                new_layouts[breakpoint] = breakpoint_layout

        self.grid.param.update(
            keys=new_keys,
            objects=new_objects,
            layouts=new_layouts,
        )

        workspace_debug_print(
            "add_panel",
            {
                "panel_id": panel_id,
                "title": title,
                "kind": kind,
                "plugin_id": plugin_id,
                "registration_id": registration_id,
                "persistent": persistent,
            },
        )

    def _attach_metadata_to_view(self, record: PanelRecord) -> None:
        common_attrs = {
            "_al_panel_id": record.panel_id,
            "_al_title": record.title,
            "_al_kind": record.kind,
            "_al_plugin_id": record.plugin_id,
            "_al_registration_id": record.registration_id,
            "_al_plugin_version": record.plugin_version,
            "_al_state_version": record.state_version,
            "_al_persistent": record.persistent,
            "_al_open_kwargs": dict(record.open_kwargs),
            "_al_metadata": dict(record.metadata),
            "_al_panel_record": record,
        }

        targets = [record.view]

        if record.controller is not None and record.controller is not record.view:
            targets.append(record.controller)

        for target in targets:
            if target is None:
                continue

            for name, value in common_attrs.items():
                try:
                    setattr(target, name, value)
                except Exception:
                    pass

        try:
            setattr(record.view, "_al_controller", record.controller)
        except Exception:
            pass

    def remove_panel(self, panel_id: str) -> None:
        self._sync_grid()
        self._normalize_grid_state()

        panel_id = str(panel_id)
        keys = [str(key) for key in (self.grid.keys or [])]

        if panel_id not in keys:
            self._panels.pop(panel_id, None)
            try:
                self.grid.close_key = ""
            except Exception:
                pass
            return

        index = keys.index(panel_id)
        record = self._panels.get(panel_id)
        view = record.view if record else None
        controller = record.controller if record else None

        if controller is None and view is not None:
            controller = getattr(view, "_al_controller", None)

        self._safe_dispose(controller)

        if view is not None and view is not controller:
            self._safe_dispose(view)

        new_layouts = {}
        for breakpoint, breakpoint_layout in (self.grid.layouts or {}).items():
            new_layouts[breakpoint] = [
                item
                for item in (breakpoint_layout or [])
                if str(item.get("i")) != panel_id
            ]

        self._panels.pop(panel_id, None)

        self.grid.param.update(
            keys=[key for key in keys if key != panel_id],
            objects=[
                obj
                for obj_index, obj in enumerate(self.grid.objects or [])
                if obj_index != index
            ],
            layouts=new_layouts,
            close_key="",
        )

        workspace_debug_print(
            "remove_panel",
            {
                "panel_id": panel_id,
            },
        )

    @staticmethod
    def _safe_dispose(obj: Any) -> None:
        if obj is None:
            return

        dispose = getattr(obj, "dispose", None)
        if not callable(dispose):
            return

        try:
            dispose()
        except Exception:
            pass

    def clear(self) -> None:
        for panel_id in list(self._panels.keys()):
            self.remove_panel(panel_id)

        self._panels.clear()
        self.grid.param.update(
            keys=[],
            objects=[],
            layouts={},
            close_key="",
        )

    def list_panels(self) -> Dict[str, PanelRecord]:
        return dict(self._panels)

    def get_panel_record(self, panel_id: str) -> PanelRecord:
        return self._panels[str(panel_id)]

    def snapshot_grid(self) -> dict[str, Any]:
        self._sync_grid()
        self._normalize_grid_state()

        return {
            "keys": [str(key) for key in (self.grid.keys or [])],
            "layouts": deepcopy(dict(self.grid.layouts or {})),
            "breakpoints": dict(self.grid.breakpoints or {}),
            "cols_by_breakpoint": dict(self.grid.cols_by_breakpoint or {}),
            "current_breakpoint": self.grid.current_breakpoint,
            "current_layout": deepcopy(list(self.grid.current_layout or [])),
            "row_height": self.grid.row_height,
            "margin": list(self.grid.margin or []),
            "compact_type": self.grid.compact_type,
            "resize_handles": list(self.grid.resize_handles or []),
        }

    def apply_grid_snapshot(self, grid_snapshot: dict[str, Any]) -> None:
        self._sync_grid()

        update = {}

        if "breakpoints" in grid_snapshot:
            update["breakpoints"] = grid_snapshot["breakpoints"]

        if "cols_by_breakpoint" in grid_snapshot:
            update["cols_by_breakpoint"] = grid_snapshot["cols_by_breakpoint"]

        if "row_height" in grid_snapshot:
            update["row_height"] = grid_snapshot["row_height"]

        if "margin" in grid_snapshot:
            update["margin"] = grid_snapshot["margin"]

        if "compact_type" in grid_snapshot:
            update["compact_type"] = grid_snapshot["compact_type"]

        if "resize_handles" in grid_snapshot:
            update["resize_handles"] = grid_snapshot["resize_handles"]

        if update:
            self.grid.param.update(**update)

    def snapshot_panels(self) -> list[dict[str, Any]]:
        panels = []

        for panel_id, record in self.list_panels().items():
            if not record.persistent:
                continue

            controller = record.controller
            if controller is None:
                controller = getattr(record.view, "_al_controller", None)

            state = get_controller_state(controller)
            state_version = get_controller_state_version(
                controller,
                default=record.state_version,
            )

            panels.append(
                {
                    "instance_id": panel_id,
                    "kind": record.kind,
                    "plugin_id": record.plugin_id,
                    "registration_id": record.registration_id,
                    "plugin_version": record.plugin_version,
                    "title": record.title,
                    "state_version": state_version,
                    "state": state,
                    "open_kwargs": dict(record.open_kwargs or {}),
                    "metadata": dict(record.metadata or {}),
                }
            )

        return panels

    def snapshot(self) -> dict[str, Any]:
        workspace_debug_print(
            "snapshot",
            {
                "panel_count": len(self._panels),
                "grid_keys": list(self.grid.keys or []),
            },
        )
        
        return {
            "grid": self.snapshot_grid(),
            "panels": self.snapshot_panels(),
        }