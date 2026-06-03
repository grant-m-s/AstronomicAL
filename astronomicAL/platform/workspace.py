from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import panel as pn

import threading

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


def workspace_panel_debug(label: str, **values) -> None:
    try:
        parts = " ".join(f"{key}={value!r}" for key, value in values.items())
        print(
            f"[AL_DEBUG][Workspace][{label}] "
            f"thread={threading.current_thread().name} {parts}",
            flush=True,
        )
    except Exception:
        print(f"[AL_DEBUG][Workspace][{label}] <print failed>", flush=True)

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

        self._close_watchers: list[tuple[Any, Any]] = []
        self._close_watched_grid: Any = None

        self._normalize_grid_state()
        self._ensure_close_watcher()

    def _grid_titles(self) -> dict[str, str]:
        """Return visible tile titles keyed by stable panel id."""

        keys = [str(key) for key in (self.grid.keys or [])]
        existing = dict(getattr(self.grid, "titles", {}) or {})

        titles: dict[str, str] = {}

        for key in keys:
            record = self._panels.get(key)
            if record is not None and record.title:
                titles[key] = str(record.title)
            else:
                titles[key] = str(existing.get(key, key))

        return titles

    def _sync_grid(self) -> None:
        live = getattr(self.react, "_dynamic_grid", None)

        if live is not None and live is not self.grid:
            self._detach_close_watchers()

            self.grid = live

            self._normalize_grid_state()
            self._ensure_close_watcher()

            workspace_debug_print(
                "sync_grid.swapped_live_grid",
                {"grid_id": hex(id(self.grid))},
            )
            return

        self._ensure_close_watcher()


    def _detach_close_watchers(self) -> None:
        for grid, watcher in list(self._close_watchers):
            try:
                grid.param.unwatch(watcher)
            except Exception:
                pass

        self._close_watchers.clear()
        self._close_watched_grid = None

    def _breakpoint_names(self, layouts: Optional[dict[str, Any]] = None) -> list[str]:
        names: list[str] = []

        for source in (
            getattr(self.grid, "cols_by_breakpoint", {}) or {},
            layouts or {},
            getattr(self.grid, "layouts", {}) or {},
        ):
            for key in source.keys():
                key = str(key)

                if key not in names:
                    names.append(key)

        for key in ("lg", "md", "sm"):
            if key not in names:
                names.append(key)

        return names
    
    def _merge_current_layout_into_layouts(self) -> None:

        current_layout = list(getattr(self.grid, "current_layout", None) or [])
        if not current_layout:
            return

        keys = [str(key) for key in (getattr(self.grid, "keys", None) or [])]
        if not keys:
            return

        key_set = set(keys)

        cleaned_current = []
        for item in current_layout:
            if not isinstance(item, dict):
                continue

            item_id = item.get("i")
            if item_id is None:
                continue

            item_id = str(item_id)
            if item_id not in key_set:
                continue

            cleaned_current.append(
                self._sanitize_layout_item(
                    dict(item),
                    breakpoint=str(getattr(self.grid, "current_breakpoint", None) or "lg"),
                )
            )

        current_ids = {str(item.get("i")) for item in cleaned_current}

        # Only merge when current_layout is a complete layout for the existing
        # open panels. This avoids partially replacing layouts during startup or
        # during transient key/object mismatch states.
        if current_ids != key_set:
            return

        breakpoint = str(getattr(self.grid, "current_breakpoint", None) or "lg")
        layouts = deepcopy(dict(getattr(self.grid, "layouts", None) or {}))

        if layouts.get(breakpoint) == cleaned_current:
            return

        layouts[breakpoint] = cleaned_current

        self.grid.param.update(
            layouts=layouts,
            current_layout=deepcopy(cleaned_current),
        )

    def _current_breakpoint_layout(
        self,
        layouts: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        breakpoint = getattr(self.grid, "current_breakpoint", None) or "lg"

        if breakpoint in layouts:
            return deepcopy(layouts[breakpoint])

        for fallback in ("lg", "md", "sm"):
            if fallback in layouts:
                return deepcopy(layouts[fallback])

        return []

    def _cols_for_breakpoint(self, breakpoint: str) -> int:
        try:
            return max(1, int((self.grid.cols_by_breakpoint or {}).get(breakpoint, 12)))
        except Exception:
            return 12

    @staticmethod
    def _layout_bottom_y(layout: list[dict[str, Any]]) -> int:
        bottom = 0

        for item in layout or []:
            try:
                y = int(item.get("y", 0))
                h = int(item.get("h", 1))
            except Exception:
                y = 0
                h = 1

            bottom = max(bottom, y + h)

        return bottom

    @staticmethod
    def _items_collide(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_x = int(left.get("x", 0))
        left_y = int(left.get("y", 0))
        left_w = int(left.get("w", 1))
        left_h = int(left.get("h", 1))

        right_x = int(right.get("x", 0))
        right_y = int(right.get("y", 0))
        right_w = int(right.get("w", 1))
        right_h = int(right.get("h", 1))

        return not (
            left_x + left_w <= right_x
            or right_x + right_w <= left_x
            or left_y + left_h <= right_y
            or right_y + right_h <= left_y
        )

    def _collides_with_layout(
        self,
        item: dict[str, Any],
        layout: list[dict[str, Any]],
    ) -> bool:
        return any(self._items_collide(item, existing) for existing in layout or [])

    def _place_item_without_colliding(
        self,
        item: dict[str, Any],
        existing_layout: list[dict[str, Any]],
        *,
        breakpoint: str,
    ) -> dict[str, Any]:

        item = self._sanitize_layout_item(
            dict(item),
            breakpoint=breakpoint,
        )

        guard = 0

        while self._collides_with_layout(item, existing_layout) and guard < 500:
            item["y"] = self._layout_bottom_y(existing_layout)
            item = self._sanitize_layout_item(
                item,
                breakpoint=breakpoint,
            )

            # If bottom_y still collides because of unusual saved layouts,
            # nudge down one row and try again.
            if self._collides_with_layout(item, existing_layout):
                item["y"] = int(item.get("y", 0)) + 1

            guard += 1

        return item

    def _sanitize_layout_item(
        self,
        item: dict[str, Any],
        *,
        panel_id: Optional[str] = None,
        breakpoint: str = "lg",
    ) -> dict[str, Any]:
        item = dict(item or {})

        if panel_id is not None:
            item["i"] = str(panel_id)
        elif "i" in item:
            item["i"] = str(item["i"])

        cols = self._cols_for_breakpoint(breakpoint)

        def as_int(value: Any, default: int, minimum: int = 0) -> int:
            try:
                out = int(value)
            except Exception:
                out = default

            return max(minimum, out)

        w = as_int(item.get("w", 4), 4, minimum=1)
        h = as_int(item.get("h", 4), 4, minimum=1)
        x = as_int(item.get("x", 0), 0, minimum=0)
        y = as_int(item.get("y", 0), 0, minimum=0)

        w = min(w, cols)
        x = min(x, max(0, cols - w))

        clean = {
            "i": str(item.get("i", panel_id or "")),
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        }

        if "static" in item:
            clean["static"] = bool(item["static"])

        if "minW" in item:
            clean["minW"] = as_int(item.get("minW"), 1, minimum=1)

        if "minH" in item:
            clean["minH"] = as_int(item.get("minH"), 1, minimum=1)

        if "maxW" in item:
            clean["maxW"] = as_int(item.get("maxW"), cols, minimum=1)

        if "maxH" in item:
            clean["maxH"] = as_int(item.get("maxH"), h, minimum=1)

        return clean

    def _new_default_layout_item(
        self,
        panel_id: str,
        *,
        breakpoint: str,
        existing_layout: list[dict[str, Any]],
        layout_item: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:

        hint = dict(layout_item or {})

        item = {
            "i": panel_id,
            "x": hint.get("x", 0),
            "y": self._layout_bottom_y(existing_layout),
            "w": hint.get("w", 4),
            "h": hint.get("h", 4),
        }

        # Preserve constraints, but do not preserve y from plugin default_layout.
        for key in ("minW", "minH", "maxW", "maxH", "static"):
            if key in hint:
                item[key] = hint[key]

        item = self._sanitize_layout_item(
            item,
            panel_id=panel_id,
            breakpoint=breakpoint,
        )

        return self._place_item_without_colliding(
            item,
            existing_layout,
            breakpoint=breakpoint,
        )

    def _normalize_grid_state(self) -> None:

        keys = [str(key) for key in (self.grid.keys or [])]
        key_set = set(keys)

        layouts = {}

        for breakpoint, breakpoint_layout in dict(self.grid.layouts or {}).items():
            new_breakpoint_layout = []

            for item in breakpoint_layout or []:
                if item is None:
                    continue

                item_copy = dict(item)

                if "i" not in item_copy:
                    continue

                if str(item_copy["i"]) not in key_set:
                    continue

                item_copy["i"] = str(item_copy["i"])

                new_breakpoint_layout.append(
                    self._sanitize_layout_item(
                        item_copy,
                        breakpoint=str(breakpoint),
                    )
                )

            layouts[str(breakpoint)] = new_breakpoint_layout

        if keys != (self.grid.keys or []) or layouts != (self.grid.layouts or {}):
            workspace_debug_print(
                "normalize_grid_state.updated",
                {
                    "keys": keys,
                    "breakpoints": list(layouts.keys()),
                },
            )
            self.grid.param.update(keys=keys, layouts=layouts)

    def _ensure_close_watcher(self) -> None:
        if self._close_watched_grid is self.grid:
            return

        self._detach_close_watchers()

        def _close_from_close_event(event) -> None:
            if not event.new:
                return

            panel_id = getattr(self.grid, "close_key", "")

            if not panel_id:
                return

            self.remove_panel(str(panel_id))

        def _close_from_key(event) -> None:
            if not event.new:
                return

            self.remove_panel(str(event.new))

        if hasattr(self.grid, "close_click_count"):
            watcher = self.grid.param.watch(_close_from_close_event, "close_click_count")
            self._close_watchers.append((self.grid, watcher))
        elif hasattr(self.grid, "close_key"):
            watcher = self.grid.param.watch(_close_from_key, "close_key")
            self._close_watchers.append((self.grid, watcher))

        self._close_watched_grid = self.grid

    def register_existing(self) -> None:
        """
        Register panels already present in the grid.

        This is mostly useful during startup or while old code is being removed.
        """
        self._sync_grid()
        self._merge_current_layout_into_layouts()
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

    def _replace_panel_in_place(
        self,
        panel_id: str,
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
    ) -> bool:

        panel_id = str(panel_id)

        workspace_panel_debug(
            "replace_panel_in_place ENTER",
            panel_id=panel_id,
            title=title,
            kind=kind,
            plugin_id=plugin_id,
            registration_id=registration_id,
            current_keys=[str(k) for k in (self.grid.keys or [])],
        )


        keys = [str(key) for key in (self.grid.keys or [])]

        if panel_id not in keys:
            return False

        index = keys.index(panel_id)

        old_record = self._panels.get(panel_id)
        old_view = old_record.view if old_record else None
        old_controller = old_record.controller if old_record else None

        if old_controller is None and old_view is not None:
            old_controller = getattr(old_view, "_al_controller", None)

        record = PanelRecord(
            panel_id=panel_id,
            title=title or panel_id,
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

        new_objects = list(self.grid.objects or [])

        # Keep objects aligned with keys defensively.
        while len(new_objects) < len(keys):
            new_objects.append(pn.Spacer(sizing_mode="stretch_both"))

        new_objects[index] = view

        # Preserve existing layout items. Only add a layout item if one is missing.
        new_layouts = deepcopy(dict(self.grid.layouts or {}))
        breakpoints = self._breakpoint_names(new_layouts)

        for breakpoint in breakpoints:
            breakpoint = str(breakpoint)
            existing_layout = [
                self._sanitize_layout_item(
                    dict(item),
                    breakpoint=breakpoint,
                )
                for item in list(new_layouts.get(breakpoint, []) or [])
                if item is not None and str(item.get("i")) in keys
            ]

            has_item = any(str(item.get("i")) == panel_id for item in existing_layout)

            if not has_item:
                supplied_item = None
                if layout_items is not None:
                    supplied_item = layout_items.get(breakpoint)

                if supplied_item is not None:
                    item_copy = self._sanitize_layout_item(
                        dict(supplied_item),
                        panel_id=panel_id,
                        breakpoint=breakpoint,
                    )
                else:
                    item_copy = self._new_default_layout_item(
                        panel_id,
                        breakpoint=breakpoint,
                        existing_layout=[
                            item for item in existing_layout
                            if str(item.get("i")) != panel_id
                        ],
                        layout_item=layout_item,
                    )

                existing_layout.append(item_copy)

            new_layouts[breakpoint] = existing_layout

        update = {
            "objects": new_objects,
            "layouts": new_layouts,
            "current_layout": self._current_breakpoint_layout(new_layouts),
        }

        if hasattr(self.grid, "titles"):
            new_titles = dict(getattr(self.grid, "titles", {}) or {})
            new_titles[panel_id] = str(title or panel_id)
            update["titles"] = new_titles

        self.grid.param.update(**update)

        workspace_panel_debug(
            "replace_panel_in_place AFTER",
            panel_id=panel_id,
            grid_keys=[str(k) for k in (self.grid.keys or [])],
            object_count=len(self.grid.objects or []),
        )

        # Dispose the old panel after the grid no longer references it.
        if old_controller is not None and old_controller is not controller:
            self._safe_dispose(old_controller)

        if (
            old_view is not None
            and old_view is not view
            and old_view is not old_controller
        ):
            self._safe_dispose(old_view)

        workspace_debug_print(
            "replace_panel_in_place",
            {
                "panel_id": panel_id,
                "title": title or panel_id,
                "kind": kind,
                "plugin_id": plugin_id,
                "registration_id": registration_id,
                "persistent": persistent,
            },
        )

        return True

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
        self._merge_current_layout_into_layouts()
        self._normalize_grid_state()

        panel_id = str(panel_id)

        workspace_panel_debug(
            "add_panel ENTER",
            panel_id=panel_id,
            title=title,
            kind=kind,
            plugin_id=plugin_id,
            registration_id=registration_id,
            existing_keys=[str(k) for k in (self.grid.keys or [])],
        )

        title = title or panel_id

        existing_keys = [str(k) for k in (self.grid.keys or [])]

        if panel_id in existing_keys:
            replaced = self._replace_panel_in_place(
                panel_id,
                view,
                title=title,
                controller=controller,
                layout_item=layout_item,
                layout_items=layout_items,
                kind=kind,
                plugin_id=plugin_id,
                registration_id=registration_id,
                plugin_version=plugin_version,
                state_version=state_version,
                persistent=persistent,
                open_kwargs=open_kwargs,
                metadata=metadata,
            )
            if replaced:
                return

        if panel_id in self._panels:
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
        breakpoints = self._breakpoint_names(new_layouts)

        for breakpoint in breakpoints:
            breakpoint_layout = [
                self._sanitize_layout_item(
                    dict(existing),
                    breakpoint=breakpoint,
                )
                for existing in list(new_layouts.get(breakpoint, []) or [])
                if str(existing.get("i")) != panel_id
            ]

            supplied_item = None

            if layout_items is not None:
                supplied_item = layout_items.get(breakpoint)

            if supplied_item is not None:

                item_copy = self._sanitize_layout_item(
                    dict(supplied_item),
                    panel_id=panel_id,
                    breakpoint=breakpoint,
                )
            else:

                item_copy = self._new_default_layout_item(
                    panel_id,
                    breakpoint=breakpoint,
                    existing_layout=breakpoint_layout,
                    layout_item=layout_item,
                )

            breakpoint_layout.append(item_copy)
            new_layouts[breakpoint] = breakpoint_layout

        new_current_layout = self._current_breakpoint_layout(new_layouts)

        update = {
            "keys": new_keys,
            "objects": new_objects,
            "layouts": new_layouts,
            "current_layout": self._current_breakpoint_layout(new_layouts),
        }

        if hasattr(self.grid, "titles"):
            new_titles = dict(getattr(self.grid, "titles", {}) or {})
            new_titles = {
                str(key): str(new_titles.get(str(key), str(key)))
                for key in new_keys
            }
            new_titles[panel_id] = str(title)
            update["titles"] = new_titles

        workspace_panel_debug(
            "grid.update BEFORE",
            panel_id=panel_id,
            update_keys=list(update.keys()),
            new_keys=[str(k) for k in update.get("keys", [])],
            object_count=len(update.get("objects", []) or []),
        )

        self.grid.param.update(**update)

        workspace_panel_debug(
            "grid.update AFTER",
            panel_id=panel_id,
            grid_keys=[str(k) for k in (self.grid.keys or [])],
            object_count=len(self.grid.objects or []),
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
        self._merge_current_layout_into_layouts()
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

        remaining_keys = [key for key in keys if key != panel_id]

        update = {
            "keys": remaining_keys,
            "objects": [
                obj
                for obj_index, obj in enumerate(self.grid.objects or [])
                if obj_index != index
            ],
            "layouts": new_layouts,
            "current_layout": self._current_breakpoint_layout(new_layouts),
            "close_key": "",
        }

        if hasattr(self.grid, "titles"):
            existing_titles = dict(getattr(self.grid, "titles", {}) or {})
            update["titles"] = {
                str(key): str(existing_titles.get(str(key), str(key)))
                for key in remaining_keys
            }

        if hasattr(self.grid, "close_click_count"):
            update["close_click_count"] = getattr(self.grid, "close_click_count", 0)

        self.grid.param.update(**update)

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

        self._sync_grid()
        self._merge_current_layout_into_layouts()
        self._normalize_grid_state()

        for panel_id in list(self._panels.keys()):
            self.remove_panel(panel_id)

        self._panels.clear()

        update = {
            "keys": [],
            "objects": [],
            "layouts": {},
            "current_layout": [],
            "close_key": "",
        }

        if hasattr(self.grid, "titles"):
            update["titles"] = {}

        if hasattr(self.grid, "close_click_count"):
            update["close_click_count"] = getattr(self.grid, "close_click_count", 0)

        self.grid.param.update(**update)

    def list_panels(self) -> Dict[str, PanelRecord]:
        return dict(self._panels)

    def get_panel_record(self, panel_id: str) -> PanelRecord:
        return self._panels[str(panel_id)]

    def snapshot_grid(self) -> dict[str, Any]:

        self._sync_grid()
        self._merge_current_layout_into_layouts()
        self._normalize_grid_state()

        snapshot = {
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

        if hasattr(self.grid, "prevent_collision"):
            snapshot["prevent_collision"] = self.grid.prevent_collision

        if hasattr(self.grid, "titles"):
            snapshot["titles"] = dict(self.grid.titles or {})

        return snapshot

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

        if "prevent_collision" in grid_snapshot and hasattr(self.grid, "prevent_collision"):
            update["prevent_collision"] = grid_snapshot["prevent_collision"]

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