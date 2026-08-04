from __future__ import annotations

from pathlib import Path
from typing import Any

import panel as pn
from astronomicAL.platform.panel_catalogue import PanelCatalogueController
from astronomicAL.platform.workspace_grid import DynamicReactGrid
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
            add_panel_catalogue(context=context)
        except Exception as exc:
            import traceback

            print("[add_panel_catalogue] ERROR:", exc)
            traceback.print_exc()

    add_menu_btn.on_click(_on_add_menu)

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

PANEL_CATALOGUE_LAYOUT_ITEMS = {
    "lg": {"x": 0, "y": 0, "w": 4, "h": 6},
    "md": {"x": 0, "y": 0, "w": 6, "h": 6},
    "sm": {"x": 0, "y": 0, "w": 12, "h": 6},
}


def _next_platform_panel_id(context: Any, prefix: str) -> str:
    workspace = getattr(context, "workspace", None)
    if workspace is None:
        raise ValueError("_next_platform_panel_id requires context.workspace.")

    try:
        live_ids = {str(panel_id) for panel_id in workspace.list_panels()}
    except Exception:
        live_ids = set()

    index = 1
    while f"{prefix}:{index}" in live_ids:
        index += 1
    return f"{prefix}:{index}"


def add_panel_catalogue(*, context: Any) -> str:
    """Add a non-persistent platform panel for registered plugin discovery."""

    if context is None:
        raise ValueError("add_panel_catalogue requires context.")
    if getattr(context, "workspace", None) is None:
        raise ValueError("add_panel_catalogue requires context.workspace.")
    if getattr(context, "plugins", None) is None:
        raise ValueError("add_panel_catalogue requires context.plugins.")

    panel_id = _next_platform_panel_id(context, prefix="panel-catalogue")
    controller = PanelCatalogueController(context=context)

    context.workspace.add_panel(
        panel_id,
        controller.panel(),
        title="Add Panel",
        controller=controller,
        layout_items={
            breakpoint: dict(item)
            for breakpoint, item in PANEL_CATALOGUE_LAYOUT_ITEMS.items()
        },
        kind="platform_panel",
        plugin_id=None,
        registration_id="platform.panel_catalogue",
        persistent=False,
        metadata={
            "description": "Temporary catalogue of enabled plugin panels.",
        },
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

    layout_file = getattr(context, "layout_file", None)
    if not layout_file:
        raise ValueError("context.layout_file is not set.")

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
            "adding temporary panel catalogue."
        )
        try:
            add_panel_catalogue(context=context)
        except Exception as exc:
            print(
                "[create_layout_from_file] Could not add fallback panel catalogue:",
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
        add_panel_catalogue(context=context)
    except Exception as exc:
        print("[create_default_layout] could not add panel catalogue:", exc)

    _publish(
        context,
        "workspace.default.created",
        {"panel_count": len(context.workspace.list_panels())},
    )
    if return_grid:
        return react, grid
    return react