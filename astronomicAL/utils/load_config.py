from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn

from astronomicAL.dashboard.dashboard import Dashboard
from astronomicAL.extensions.dynamic_react_layout import DynamicReactGrid
from astronomicAL.platform.application_chrome_styles import (
    APPLICATION_CHROME_CSS,
    HEADER_BUTTON_STYLESHEET,
    HEADER_PRIMARY_BUTTON_STYLESHEET,
    HEADER_RUNTIME_BUTTON_STYLESHEET,
    HEADER_STATUS_BUTTON_STYLESHEET,
)
from astronomicAL.platform.application_toolbar import ApplicationToolbarController
from astronomicAL.platform.dataset_header import DatasetHeaderController
from astronomicAL.platform.layout_controls import create_layout_controls
from astronomicAL.platform.mapping_header import MappingAlertController
from astronomicAL.platform.modal_utils import ensure_template_modal_host
from astronomicAL.platform.runtime_status_box import RuntimeStatusBox


DEFAULT_BREAKPOINTS = {"lg": 1500, "md": 1050, "sm": 0}
DEFAULT_COLS_BY_BREAKPOINT = {"lg": 12, "md": 12, "sm": 12}
DEFAULT_RESIZE_HANDLES = ["s", "w", "e", "n", "sw", "nw", "se"]

_MODAL_HOST_CSS = """
#pn-Modal {
  background: transparent !important;
}
#pn-Modal .pn-modal-content {
  display: flex !important;
  align-items: flex-start !important;
  justify-content: center !important;
  width: auto !important;
  max-width: none !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  overflow: visible !important;
}
#pn-Modal .pn-modalclose {
  display: none !important;
}
"""

_RUNTIME_DETAILS_STYLESHEET = """
:host {
  background: #ffffff !important;
  border-radius: 0 999px 999px 0 !important;
  overflow: hidden !important;
}

button,
.bk-btn {
  width: 100% !important;
  height: 28px !important;
  border: 0 !important;
  border-left: 1px solid #d8dee8 !important;
  border-radius: 0 999px 999px 0 !important;
  background: #ffffff !important;
  background-color: #ffffff !important;
  opacity: 1 !important;
  color: #263244 !important;
}

button:hover,
.bk-btn:hover {
  background: #f7f8fa !important;
  background-color: #f7f8fa !important;
}
"""

def _publish(context: Any, topic: str, payload: dict[str, Any]) -> None:
    events = getattr(context, "events", None)
    if events is None:
        return
    try:
        events.publish(topic, payload)
    except Exception:
        pass


def _append_raw_css(css: str, marker: str) -> None:
    if any(marker in item for item in pn.config.raw_css):
        return
    pn.config.raw_css.append(css)


def _append_stylesheet(widget: Any, stylesheet: str) -> None:
    try:
        stylesheets = list(getattr(widget, "stylesheets", []) or [])
        if stylesheet not in stylesheets:
            stylesheets.append(stylesheet)
        widget.stylesheets = stylesheets
    except Exception:
        pass


def _extend_css_classes(obj: Any, *classes: str) -> None:
    try:
        current = list(getattr(obj, "css_classes", []) or [])
        for class_name in classes:
            if class_name and class_name not in current:
                current.append(class_name)
        obj.css_classes = current
    except Exception:
        pass


def _prepare_layout_controls_mount(layout_controls: Any) -> list[Any]:
    """Hide the original buttons while keeping their drawer/toast models mounted."""
    actions: list[Any] = []
    for obj in list(getattr(layout_controls, "objects", []) or []):
        if isinstance(obj, pn.widgets.Button):
            actions.append(obj)
            obj.visible = False

    _extend_css_classes(layout_controls, "al-layout-controls-mount")
    try:
        layout_controls.width = 0
        layout_controls.height = 0
        layout_controls.sizing_mode = "fixed"
        layout_controls.margin = (0, 0, 0, 0)
        layout_controls.styles = {
            **dict(getattr(layout_controls, "styles", {}) or {}),
            "width": "0px",
            "height": "0px",
            "min-width": "0px",
            "min-height": "0px",
            "overflow": "visible",
            "padding": "0",
            "margin": "0",
        }
    except Exception:
        pass

    stack = list(getattr(layout_controls, "objects", []) or [])
    while stack:
        obj = stack.pop()
        stack.extend(list(getattr(obj, "objects", []) or []))
        styles = dict(getattr(obj, "styles", {}) or {})
        if (
            styles.get("position") == "fixed"
            and int(getattr(obj, "width", 0) or 0) >= 500
        ):
            try:
                obj.styles = {
                    **styles,
                    "top": (
                        "calc(var(--al-application-header-height) + "
                        "var(--al-application-toolbar-height) - 2px)"
                    ),
                    "right": "12px",
                    "max-height": (
                        "calc(100vh - var(--al-application-header-height) - "
                        "var(--al-application-toolbar-height) - 14px)"
                    ),
                }
            except Exception:
                pass
    return actions


def _layout_menu_heading(label: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        label,
        height=22,
        sizing_mode="stretch_width",
        margin=(0, 0, 0, 0),
        css_classes=["al-toolbar-popover-heading"],
    )


def _attach_layout_actions_to_toolbar(
    application_toolbar: Any,
    layout_actions: list[Any],
) -> None:
    """Expose save/load layout actions inside the navbar Layouts popover."""
    if not layout_actions:
        return

    descriptions = {
        "quick": "Save the current workspace immediately",
        "save": "Save the current workspace with a chosen name",
        "load": "Load a saved or uploaded workspace layout",
    }
    labels = {
        "quick": "Quick save layout",
        "save": "Save layout as…",
        "load": "Load layout…",
    }

    ordered: list[tuple[str, Any]] = []
    for action in layout_actions:
        name = str(getattr(action, "name", "") or "").casefold()
        key = (
            "quick"
            if "quick" in name
            else "load"
            if "load" in name
            else "save"
        )
        ordered.append((key, action))
    priority = {"quick": 0, "save": 1, "load": 2}
    ordered.sort(key=lambda item: priority[item[0]])

    proxies: list[Any] = []
    for key, action in ordered:
        proxy = application_toolbar._popup_button(
            name=labels[key],
            description=descriptions[key],
        )

        def _trigger(_event: Any, *, target: Any = action) -> None:
            application_toolbar._close_popups()
            try:
                target.clicks = int(getattr(target, "clicks", 0)) + 1
            except Exception as exc:
                application_toolbar._notify(
                    f"Could not open the layout action: {exc}",
                    level="error",
                )

        proxy.on_click(_trigger)
        proxies.append(proxy)

    current = list(getattr(application_toolbar.quick_layout_popup, "objects", []) or [])
    application_toolbar.quick_layout_popup.objects = [
        _layout_menu_heading("Quick workspaces"),
        *current,
        application_toolbar._popup_divider(),
        _layout_menu_heading("Workspace files"),
        *proxies,
    ]
    application_toolbar.quick_layout_popup.width = 224
    application_toolbar.quick_layout_popup.styles = {
        **dict(application_toolbar.quick_layout_popup.styles or {}),
        "left": "auto",
        "right": "0",
        "box-sizing": "border-box",
        "overflow": "hidden",
    }
    # The trigger intentionally has no Panel description. A description
    # creates a Bokeh tooltip model whose empty callout can remain visible
    # beside an open popover.


def _set_css_class(obj: Any, class_name: str, enabled: bool) -> None:
    try:
        classes = list(getattr(obj, "css_classes", []) or [])
        classes = [item for item in classes if item != class_name]
        if enabled:
            classes.append(class_name)
        obj.css_classes = classes
    except Exception:
        pass


def _clear_widget_tooltip(widget: Any) -> None:
    """Remove Panel/Bokeh tooltip state before the widget is first rendered."""
    try:
        widget.description = None
    except Exception:
        pass

    # Some Panel versions expose a second tooltip-style parameter. Avoid
    # assuming it exists, but clear it when available.
    try:
        if "tooltip" in widget.param:
            widget.param.update(tooltip=None)
    except Exception:
        pass


def _configure_toolbar_popovers(application_toolbar: Any) -> None:
    """Keep both toolbar menus inside the viewport without tooltip callouts."""
    overflow_popup = application_toolbar.overflow_popup
    overflow_popup.width = 188
    overflow_popup.styles = {
        **dict(getattr(overflow_popup, "styles", {}) or {}),
        "position": "absolute",
        "top": "34px",
        "left": "auto",
        "right": "0",
        "inset-inline-start": "auto",
        "inset-inline-end": "0",
        "width": "188px",
        "max-width": "calc(100vw - 24px)",
        "box-sizing": "border-box",
        "overflow": "hidden",
        "transform": "none",
        "z-index": "2400",
    }

    overflow_menu = application_toolbar.overflow_menu
    overflow_menu.styles = {
        **dict(getattr(overflow_menu, "styles", {}) or {}),
        "position": "relative",
        "overflow": "visible",
    }

    # Panel/Bokeh descriptions render tooltip callouts. On the two menu
    # triggers that callout can remain as a detached arrow while the popup is
    # open, so these controls intentionally rely on their visible labels and
    # accessible names instead of hover descriptions.
    _clear_widget_tooltip(application_toolbar.quick_layout_button)
    _clear_widget_tooltip(application_toolbar.overflow_button)
    _extend_css_classes(
        application_toolbar.quick_layout_button,
        "al-toolbar-menu-trigger",
    )
    _extend_css_classes(
        application_toolbar.overflow_button,
        "al-toolbar-menu-trigger",
    )


def create_layout_skeleton(
    react: pn.template.ReactTemplate,
    *,
    return_grid: bool = False,
):
    """Create and attach the empty DynamicReactGrid."""
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
    """Build the coordinated application header and record toolbar."""
    if context is None:
        raise ValueError("create_header requires context.")
    if getattr(context, "config", None) is None:
        raise ValueError("create_header requires context.config.")

    _append_raw_css(APPLICATION_CHROME_CSS, "--al-application-chrome-connected")
    _append_raw_css(_MODAL_HOST_CSS, "#pn-Modal .pn-modal-content")

    # RuntimeStatusBox is the sole application activity indicator. The template
    # spinner otherwise renders at the far-right edge, outside the header card.
    try:
        react.busy_indicator.visible = False
    except Exception:
        pass

    if not hasattr(react, "_header_box"):
        react._header_box = pn.Row(
            sizing_mode="stretch_width",
            height=50,
            margin=(0, 0, 0, 0),
            css_classes=["al-application-header-shell"],
            styles={"overflow": "visible", "min-width": "0"},
        )
        react.header.append(react._header_box)
    else:
        _extend_css_classes(react._header_box, "al-application-header-shell")
        react._header_box.height = 50
        react._header_box.margin = (0, 0, 0, 0)

    ensure_template_modal_host(react)

    dataset_header = DatasetHeaderController(context=context, template=react)
    old_mapping_alert = getattr(react, "_mapping_alert", None)
    if old_mapping_alert is not None and hasattr(old_mapping_alert, "dispose"):
        try:
            old_mapping_alert.dispose()
        except Exception:
            pass

    mapping_alert = MappingAlertController(context=context, template=react)
    react._dataset_header = dataset_header
    react._mapping_alert = mapping_alert

    # Keep the existing mapping controller and modal behaviour. The header
    # only controls whether its already-mounted view is shown for the active
    # dataset; it does not call the controller's private refresh methods.
    mapping_alert.button.width = 120
    mapping_alert.button.height = 30
    mapping_alert.button.margin = (0, 0, 0, 0)
    mapping_alert.view.width = 120
    mapping_alert.view.height = 30
    mapping_alert.view.margin = (0, 0, 0, 0)
    mapping_alert.view.sizing_mode = "fixed"
    _extend_css_classes(mapping_alert.view, "al-header-mapping-slot")

    layout_controls = create_layout_controls(context=context, template=react)
    layout_actions = _prepare_layout_controls_mount(layout_controls)

    _append_stylesheet(mapping_alert.button, HEADER_STATUS_BUTTON_STYLESHEET)
    _extend_css_classes(mapping_alert.button, "al-header-mapping-control")

    add_menu_btn = pn.widgets.Button(
        name="Add panel",
        icon="square-plus",
        button_type="primary",
        width=104,
        height=30,
        margin=(0, 0, 0, 0),
        css_classes=["al-add-menu-btn"],
        stylesheets=[HEADER_PRIMARY_BUTTON_STYLESHEET],
    )
    add_menu_btn.description = "Open the panel catalogue"

    def _on_add_menu(_event: Any) -> None:
        try:
            add_menu_panel(grid, context=context)
        except Exception as exc:
            import traceback

            print("[add_menu_panel] ERROR:", exc)
            traceback.print_exc()

    add_menu_btn.on_click(_on_add_menu)

    # Retained for compatibility; it is currently not placed in the header.
    _export_fits_file_button = _build_export_labelled_data_button(context)

    old_runtime_status_box = getattr(react, "_runtime_status_box", None)
    if old_runtime_status_box is not None and hasattr(
        old_runtime_status_box, "dispose"
    ):
        try:
            old_runtime_status_box.dispose()
        except Exception:
            pass

    runtime_status_box = RuntimeStatusBox(context=context, template=react)
    react._runtime_status_box = runtime_status_box
    _extend_css_classes(runtime_status_box.view, "al-header-runtime-control")
    try:
        runtime_status_box.view.width = 252
        runtime_status_box.view.margin = (0, 0, 0, 0)
        runtime_status_box.view.height = 30

        runtime_status_box.summary.width = 180
        runtime_status_box.summary.margin = (0, 0, 0, 0)
        runtime_status_box.summary.height = 28

        runtime_status_box.toggle.name = "Details"
        runtime_status_box.toggle.width = 72
        runtime_status_box.toggle.height = 28
        runtime_status_box.toggle.margin = (0, 0, 0, 0)
    except Exception:
        pass

    _append_stylesheet(runtime_status_box.toggle, HEADER_RUNTIME_BUTTON_STYLESHEET)
    _append_stylesheet(runtime_status_box.toggle, _RUNTIME_DETAILS_STYLESHEET)


    old_application_toolbar = getattr(react, "_application_toolbar", None)
    if old_application_toolbar is not None and hasattr(
        old_application_toolbar, "dispose"
    ):
        try:
            old_application_toolbar.dispose()
        except Exception:
            pass

    old_toolbar_view = getattr(old_application_toolbar, "view", None)
    if old_toolbar_view is not None:
        try:
            react.header.remove(old_toolbar_view)
        except (ValueError, AttributeError):
            pass

    application_toolbar = ApplicationToolbarController(
        context=context,
        template=react,
    )
    react._application_toolbar = application_toolbar
    _clear_widget_tooltip(application_toolbar.quick_layout_button)
    _clear_widget_tooltip(application_toolbar.overflow_button)
    _attach_layout_actions_to_toolbar(application_toolbar, layout_actions)
    _configure_toolbar_popovers(application_toolbar)
    react.header.append(application_toolbar.view)

    brand = pn.pane.HTML(
        """
        <div class="al-brand-lockup" aria-label="AstronomicAL">
          <span class="al-brand-wordmark">Astronomic<strong>AL</strong></span>
        </div>
        """,
        width=176,
        height=32,
        margin=(0, 0, 0, 0),
        css_classes=["al-application-brand"],
    )

    data_group = pn.Row(
        dataset_header.view,
        mapping_alert.view,
        sizing_mode="fixed",
        height=32,
        margin=(0, 0, 0, 0),
        css_classes=["al-header-group", "al-header-data-group"],
        styles={"overflow": "visible", "min-width": "0"},
    )
    status_group = pn.Row(
        runtime_status_box.view,
        sizing_mode="fixed",
        height=32,
        margin=(0, 0, 0, 0),
        css_classes=["al-header-group", "al-header-status-group"],
        styles={"overflow": "visible", "min-width": "0"},
    )
    workspace_group = pn.Row(
        add_menu_btn,
        sizing_mode="fixed",
        height=32,
        margin=(0, 0, 0, 0),
        css_classes=["al-header-group", "al-header-workspace-group"],
        styles={"overflow": "visible", "min-width": "0"},
    )

    header_row = pn.Row(
        brand,
        _header_divider(),
        data_group,
        pn.layout.HSpacer(),
        status_group,
        _header_divider(),
        workspace_group,
        layout_controls,
        sizing_mode="stretch_width",
        height=50,
        margin=(0, 0, 0, 0),
        align="center",
        css_classes=["al-application-header"],
        styles={
            "display": "flex",
            "align-items": "center",
            "gap": "8px",
            "overflow": "visible",
            "min-width": "0",
        },
    )
    react._header_box[:] = [header_row]

    old_state_subscriptions = list(
        getattr(react, "_header_state_subscriptions", []) or []
    )
    for subscription in old_state_subscriptions:
        try:
            context.events.unsubscribe(subscription)
        except Exception:
            pass

    def _sync_dataset_controls(_topic: str = "", _payload: Any = None) -> None:
        has_dataset = dataset_header.has_active_dataset()

        def _apply() -> None:
            # Visibility belongs to the application header. MappingAlertController
            # continues to own the button label, warning state, modal, and mapping
            # request lifecycle. Keeping the view mounted avoids Panel dropping an
            # initially hidden child from the rendered header.
            mapping_alert.view.visible = has_dataset
            mapping_alert.button.visible = has_dataset

            _set_css_class(
                application_toolbar.navigation_group,
                "al-toolbar-empty-state",
                not has_dataset,
            )
            _set_css_class(
                application_toolbar.scope_group,
                "al-toolbar-empty-state",
                not has_dataset,
            )

        try:
            doc = pn.state.curdoc
        except Exception:
            doc = None
        if doc is None or not _topic:
            _apply()
        else:
            try:
                doc.add_next_tick_callback(_apply)
            except Exception:
                _apply()

    subscriptions: list[Any] = []
    events = getattr(context, "events", None)
    if events is not None:
        for topic in (
            "dataset.loaded",
            "dataset.active.changed",
            "dataset.removed",
            "dataset.updated",
            "dataset.mapping.updated",
        ):
            try:
                subscriptions.append(
                    events.subscribe(
                        topic,
                        _sync_dataset_controls,
                        owner_id="platform.application_header",
                        owner_label="Application header",
                        owner_kind="application_chrome",
                    )
                )
            except TypeError:
                subscriptions.append(events.subscribe(topic, _sync_dataset_controls))
    react._header_state_subscriptions = subscriptions
    _sync_dataset_controls()
    return react


def _header_divider() -> pn.pane.HTML:
    return pn.pane.HTML(
        "",
        width=1,
        height=20,
        margin=(5, 4, 5, 4),
        css_classes=["al-header-inner-divider"],
    )


def _build_export_labelled_data_button(context: Any):
    """Build the legacy labelled-data export control."""
    button = pn.widgets.Button(name="Export Labelled Data to Fits File")

    def export_fits_file_cb(_event: Any) -> None:
        config = context.config
        settings = getattr(config, "settings", {}) or {}
        list_ids: list[str] = []
        list_labels: list[str] = []

        if settings.get("confirmed"):
            classifiers = settings.get("classifiers") or {}
            for _label, entry in classifiers.items():
                if isinstance(entry, dict) and "id" in entry and "y" in entry:
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
    """Attach a legacy controller to a Panel view."""
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
    numeric_suffixes: list[int] = []
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
    cols_by_breakpoint = dict(
        grid.cols_by_breakpoint or DEFAULT_COLS_BY_BREAKPOINT
    )
    breakpoints = list(cols_by_breakpoint.keys()) or ["lg", "md", "sm"]

    layout_items: dict[str, dict[str, Any]] = {}
    for breakpoint in breakpoints:
        cols = int(cols_by_breakpoint.get(breakpoint, 12))
        w = int(default_w_by_breakpoint.get(breakpoint, 12))
        h = int(default_h_by_breakpoint.get(breakpoint, 6))
        w = max(1, min(w, cols))
        existing = list(layouts.get(breakpoint, []))
        x, y = _find_first_fit(existing, cols=cols, w=w, h=h)
        layout_items[breakpoint] = {"x": x, "y": y, "w": w, "h": h}
    return layout_items


def add_menu_panel(
    grid: DynamicReactGrid | None = None,
    context: Any | None = None,
) -> str:
    """Add the current Menu dashboard as a non-persistent platform panel."""
    if context is None:
        raise ValueError("add_menu_panel requires context.")
    if getattr(context, "workspace", None) is None:
        raise ValueError("add_menu_panel requires context.workspace.")
    if getattr(context, "config", None) is None:
        raise ValueError("add_menu_panel requires context.config.")

    grid = context.workspace.grid
    context.workspace._sync_grid()
    context.workspace._merge_current_layout_into_layouts()
    context.workspace._normalize_grid_state()

    panel_id = _next_platform_panel_id(context, prefix="menu")
    dashboard = Dashboard(
        src=context.config.source,
        contents="Menu",
        context=context,
    )
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
    """Load a new-system workspace JSON file through context.persistence."""
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
            print(
                "[create_layout_from_file] Could not add fallback Menu panel:",
                exc,
            )

    react = create_header(react, context.workspace.grid, context=context)
    _publish(
        context,
        "workspace.loaded",
        {"path": str(layout_path), "issues": issues},
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
    """Create the default empty plugin workspace."""
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
        {"panel_count": len(context.workspace.list_panels())},
    )
    if return_grid:
        return react, grid
    return react