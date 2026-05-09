from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn

from astronomicAL.config import get_save_layout_button
from astronomicAL.dashboard.dashboard import Dashboard
from astronomicAL.extensions.dynamic_react_layout import DynamicReactGrid
from astronomicAL.platform.dataset_header import DatasetHeaderController
from astronomicAL.platform.mapping_header import MappingAlertController
from astronomicAL.platform.modal_utils import ensure_template_modal_host


DEFAULT_BREAKPOINTS = {"lg": 1500, "md": 1050, "sm": 0}
DEFAULT_COLS_BY_BREAKPOINT = {"lg": 12, "md": 12, "sm": 12}
DEFAULT_RESIZE_HANDLES = ["s", "w", "e", "n", "sw", "nw", "se"]


def _publish(context: Any, topic: str, payload: dict[str, Any]) -> None:
    events = getattr(context, "events", None)
    if events is None:
        return

    try:
        events.publish(topic, payload)
    except Exception:
        pass


def create_layout_skeleton(
    react: pn.template.ReactTemplate,
    *,
    return_grid: bool = False,
):
    """
    Create and attach the empty DynamicReactGrid.

    main.py calls this before AppContext exists so the WorkspaceManager can be
    constructed with a real grid object.
    """
    grid = DynamicReactGrid(
        keys=[],
        objects=[],
        layouts={},
        sizing_mode="stretch_both",
        height=900,
        breakpoints=dict(DEFAULT_BREAKPOINTS),
        cols_by_breakpoint=dict(DEFAULT_COLS_BY_BREAKPOINT),
        resize_handles=list(DEFAULT_RESIZE_HANDLES),
        compact_type="vertical",
    )

    react._dynamic_grid = grid
    react.main[:12, :12] = grid

    if return_grid:
        return react, grid

    return react


def create_header(
    react: pn.template.ReactTemplate,
    grid: DynamicReactGrid,
    context: Any,
):
    """
    Build the app header.

    The save button now calls context.persistence through save_config.save_workspace.
    """
    if context is None:
        raise ValueError("create_header requires context.")

    config = getattr(context, "config", None)
    if config is None:
        raise ValueError("create_header requires context.config.")

    if not hasattr(react, "_header_box"):
        react._header_box = pn.Row(sizing_mode="stretch_width")
        react.header.append(react._header_box)

    react.config.raw_css.append(
        """
        #pn-Modal {
            background: transparent !important;
        }

        #pn-Modal .pn-modal-content {
            background: transparent !important;
            box-shadow: none !important;
            border: none !important;
            padding: 0 !important;
            width: auto !important;
            max-width: none !important;
            overflow: visible !important;
            display: flex !important;
            justify-content: center !important;
            align-items: flex-start !important;
        }

        #pn-Modal .pn-modalclose {
            display: none !important;
        }
        """
    )

    ensure_template_modal_host(react)

    dataset_header = DatasetHeaderController(context=context, template=react)
    mapping_alert = MappingAlertController(context=context, template=react)

    react._dataset_header = dataset_header
    react._mapping_alert = mapping_alert

    save_button = get_save_layout_button(
        enable_button=True,
        from_main=True,
        context=context,
    )

    add_menu_btn = pn.widgets.Button(
        name="+",
        button_type="default",
        width=38,
        height=34,
    )
    add_menu_btn.styles = {
        "font-size": "26px",
        "font-weight": "700",
        "line-height": "1",
        "padding": "0",
    }
    add_menu_btn.css_classes = ["al-add-menu-btn"]
    add_menu_btn.description = "Add Panel"

    def _on_add_menu(_event) -> None:
        try:
            add_menu_panel(grid, context=context)
        except Exception as exc:
            import traceback

            print("[add_menu_panel] ERROR:", exc)
            traceback.print_exc()

    add_menu_btn.on_click(_on_add_menu)

    export_fits_file_button = _build_export_labelled_data_button(context)

    header_row = pn.Row(
        save_button,
        dataset_header.view,
        mapping_alert.view,
        add_menu_btn,
        # export_fits_file_button,
        sizing_mode="stretch_width",
    )

    react._header_box[:] = [header_row]
    return react


def _build_export_labelled_data_button(context: Any):
    """
    Kept as a helper so it can be re-enabled in create_header if needed.
    """
    button = pn.widgets.Button(name="Export Labelled Data to Fits File")

    def export_fits_file_cb(_event) -> None:
        config = context.config
        settings = getattr(config, "settings", {}) or {}

        list_ids: list[str] = []
        list_labels: list[str] = []

        if settings.get("confirmed"):
            classifiers = settings.get("classifiers") or {}
            for _label, entry in classifiers.items():
                if isinstance(entry, dict) and ("id" in entry) and ("y" in entry):
                    list_ids.extend(entry["id"])
                    list_labels.extend(entry["y"])

            test_set_file = settings.get("test_set_file")
            if test_set_file and os.path.exists("data/test_set.json"):
                with open("data/test_set.json", "r", encoding="utf-8") as handle:
                    orig_labelled_data = json.load(handle)

                for source_id, label in orig_labelled_data.items():
                    list_ids.append(source_id)
                    list_labels.append(label)

        if not list_ids:
            button.disabled = True
            button.name = "No Labelled Data Found"
            button.disabled = False
            button.name = "Export Labelled Data to Fits File"
            return

        exported_labels = pd.DataFrame(
            {"id": list_ids, "label": list_labels},
            dtype="string",
        )

        from astronomicAL.utils.save_config import save_dataframe_to_fits

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"data/labelled_data_{timestamp}.fits"

        save_dataframe_to_fits(exported_labels, path)

        button.disabled = True
        button.name = f"{len(list_ids)} labelled sources saved to '{path}'"
        button.disabled = False
        button.name = "Export Labelled Data to Fits File"

    button.on_click(export_fits_file_cb)
    return button


def bind_controller(view: Any, controller: Any):
    """
    Attach a controller to a Panel view.

    WorkspaceManager already tracks controllers explicitly, but this is useful
    for panels that are created by older dashboard code while you are migrating.
    """
    if view is None or controller is None:
        return view

    try:
        setattr(view, "_al_controller", controller)
        if hasattr(controller, "dispose"):
            setattr(view, "dispose", controller.dispose)
    except Exception:
        pass

    return view


def _overlaps(a: dict[str, int], b: dict[str, int]) -> bool:
    return not (
        a["x"] + a["w"] <= b["x"]
        or b["x"] + b["w"] <= a["x"]
        or a["y"] + a["h"] <= b["y"]
        or b["y"] + b["h"] <= a["y"]
    )


def _find_first_fit(
    layout_items: list[dict[str, Any]],
    *,
    cols: int,
    w: int,
    h: int,
) -> tuple[int, int]:
    items = [
        {
            "x": int(item.get("x", 0)),
            "y": int(item.get("y", 0)),
            "w": int(item.get("w", 1)),
            "h": int(item.get("h", 1)),
        }
        for item in (layout_items or [])
        if item is not None
    ]

    max_y = 0
    for item in items:
        max_y = max(max_y, item["y"] + item["h"])

    for y in range(0, max_y + 100):
        for x in range(0, cols - w + 1):
            candidate = {"x": x, "y": y, "w": w, "h": h}
            if not any(_overlaps(candidate, item) for item in items):
                return x, y

    return 0, max_y


def _menu_geometry_for_breakpoint(breakpoint: str) -> tuple[int, int]:
    if breakpoint == "lg":
        return 4, 6

    if breakpoint == "md":
        return 6, 6

    return 12, 6


def _next_platform_panel_id(context: Any, prefix: str = "platform") -> str:
    settings = getattr(context.config, "settings", None)
    if settings is None:
        context.config.settings = {}
        settings = context.config.settings

    counter_key = "_panel_id_counter"
    current = int(settings.get(counter_key, 0))

    live_keys = [
        str(key)
        for key in (getattr(context.workspace.grid, "keys", None) or [])
    ]

    numeric_suffixes = []
    for key in live_keys:
        if key.startswith(f"{prefix}:"):
            try:
                numeric_suffixes.append(int(key.split(":", 1)[1]))
            except Exception:
                pass

    if numeric_suffixes:
        current = max(current, max(numeric_suffixes))

    current += 1
    settings[counter_key] = current

    return f"{prefix}:{current}"


def _layout_items_for_new_tile(
    grid: DynamicReactGrid,
    *,
    default_w_by_breakpoint: dict[str, int],
    default_h_by_breakpoint: dict[str, int],
) -> dict[str, dict[str, Any]]:
    layouts = dict(grid.layouts or {})
    cols_by_breakpoint = dict(grid.cols_by_breakpoint or DEFAULT_COLS_BY_BREAKPOINT)

    breakpoints = list(cols_by_breakpoint.keys()) or ["lg", "md", "sm"]
    layout_items: dict[str, dict[str, Any]] = {}

    for breakpoint in breakpoints:
        cols = int(cols_by_breakpoint.get(breakpoint, 12))
        w = int(default_w_by_breakpoint.get(breakpoint, 12))
        h = int(default_h_by_breakpoint.get(breakpoint, 6))

        w = max(1, min(w, cols))

        existing = list(layouts.get(breakpoint, []))
        x, y = _find_first_fit(existing, cols=cols, w=w, h=h)

        layout_items[breakpoint] = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        }

    return layout_items


def add_menu_panel(
    grid: DynamicReactGrid | None = None,
    context: Any | None = None,
) -> str:
    """
    Add the current Menu dashboard as a non-persistent platform panel.

    Once the menu itself is moved into a plugin, this function can simply call
    context.plugins.open_panel("core.menu.panel", ...).
    """
    if context is None:
        raise ValueError("add_menu_panel requires context.")

    if getattr(context, "workspace", None) is None:
        raise ValueError("add_menu_panel requires context.workspace.")

    if getattr(context, "config", None) is None:
        raise ValueError("add_menu_panel requires context.config.")

    grid = context.workspace.grid

    panel_id = _next_platform_panel_id(context, prefix="menu")
    dashboard = Dashboard(
        src=context.config.source,
        contents="Menu",
        context=context,
    )

    # Important: MenuDashboard receives this Dashboard as self.main.
    # It needs to know which workspace tile it lives in so selecting a plugin
    # panel can replace this tile rather than append a new one.
    dashboard._al_panel_id = panel_id
    dashboard._al_kind = "platform_panel"
    dashboard._al_registration_id = "platform.menu"
    dashboard._al_persistent = False

    try:
        view = dashboard.panel(in_grid=True)
    except TypeError:
        view = dashboard.panel()

    view = bind_controller(view, dashboard)

    layout_items = _layout_items_for_new_tile(
        grid,
        default_w_by_breakpoint={"lg": 4, "md": 6, "sm": 12},
        default_h_by_breakpoint={"lg": 6, "md": 6, "sm": 6},
    )

    context.workspace.add_panel(
        panel_id,
        view,
        title="Menu",
        controller=dashboard,
        layout_items=layout_items,
        kind="platform_panel",
        plugin_id=None,
        registration_id="platform.menu",
        persistent=False,
        metadata={"description": "Temporary add-panel menu."},
    )

    return panel_id


def create_layout_from_file(
    react: pn.template.ReactTemplate,
    context: Any = None,
    *,
    return_grid: bool = False,
):
    """
    Load a new-system workspace JSON file through context.persistence.

    This intentionally does not load old AstronomicAL config files.
    """
    if context is None:
        raise ValueError("create_layout_from_file requires context.")

    if getattr(context, "persistence", None) is None:
        raise RuntimeError("context.persistence is not configured.")

    config = getattr(context, "config", None)
    if config is None:
        raise ValueError("create_layout_from_file requires context.config.")

    layout_file = getattr(config, "layout_file", None)
    if not layout_file:
        raise ValueError("context.config.layout_file is not set.")

    grid = getattr(react, "_dynamic_grid", None)

    if grid is None:
        react, grid = create_layout_skeleton(react, return_grid=True)

    context.workspace.react = react
    context.workspace.grid = grid

    layout_path = Path(layout_file).expanduser()

    if not layout_path.exists():
        raise FileNotFoundError(f"Workspace file does not exist: {layout_path}")

    snapshot = context.persistence.load(layout_path)
    issues = context.persistence.restore(snapshot, strict=False)

    if len(context.workspace.list_panels()) == 0:
        print(
            "[create_layout_from_file] Restored workspace contains no panels; "
            "adding temporary Menu panel."
        )
        try:
            add_menu_panel(context.workspace.grid, context=context)
        except Exception as exc:
            print("[create_layout_from_file] Could not add fallback Menu panel:", exc)

    react = create_header(react, context.workspace.grid, context=context)

    _publish(
        context,
        "workspace.loaded",
        {
            "path": str(layout_path),
            "issues": issues,
        },
    )

    if return_grid:
        return react, context.workspace.grid

    return react


def create_default_layout(
    react: pn.template.ReactTemplate,
    context: Any = None,
    *,
    return_grid: bool = False,
):
    """
    Create a default empty plugin workspace.

    The header plus button is enough to add panels. A temporary Menu panel is
    also opened so the workspace is discoverable on first launch.
    """
    if context is None:
        raise ValueError("create_default_layout requires context.")

    react, grid = create_layout_skeleton(react, return_grid=True)

    context.workspace.react = react
    context.workspace.grid = grid

    react = create_header(react, grid, context=context)

    try:
        add_menu_panel(grid, context=context)
    except Exception as exc:
        print("[create_default_layout] could not add menu panel:", exc)

    _publish(
        context,
        "workspace.default.created",
        {
            "panel_count": len(context.workspace.list_panels()),
        },
    )

    if return_grid:
        return react, grid

    return react