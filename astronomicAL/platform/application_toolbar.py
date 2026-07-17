from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import panel as pn

from astronomicAL.platform.application_toolbar_styles import (
    TOOLBAR_CSS,
    _TOOLBAR_ICON_BUTTON_STYLESHEET,
    _TOOLBAR_MENU_STYLESHEET,
    _TOOLBAR_POPUP_ITEM_STYLESHEET,
    _TOOLBAR_SELECTION_STYLESHEET,
    _TOOLBAR_SELECT_STYLESHEET,
    _TOOLBAR_TEXT_INPUT_STYLESHEET,
)

from astronomicAL.platform.record_navigation import (
    NavigationError,
    NavigationState,
    RecordNavigationManager,
)

@dataclass(frozen=True)
class PanelSelector:
    registration_ids: tuple[str, ...] = ()
    titles: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuickWorkspaceTemplate:
    id: str
    title: str
    description: str
    panels: tuple[PanelSelector, ...]


BUILT_IN_TEMPLATES: tuple[QuickWorkspaceTemplate, ...] = (
    QuickWorkspaceTemplate(
        id="exploration",
        title="Exploration",
        description="Record details, scatter visualisation and a table view.",
        panels=(
            PanelSelector(
                registration_ids=("core.record_browser.panel",),
                titles=("Record Browser", "Record Details"),
            ),
            PanelSelector(
                registration_ids=("core.visualisation.scatter",),
                titles=("Scatter", "Scatter Plot"),
            ),
            PanelSelector(
                registration_ids=("core.table_tools.table",),
                titles=("Table", "Dataset Table"),
            ),
        ),
    ),
    QuickWorkspaceTemplate(
        id="image_review",
        title="Image review",
        description="Selection gallery, image viewer and annotations.",
        panels=(
            PanelSelector(
                registration_ids=("core.image.selection_gallery",),
                titles=("Selection Gallery",),
            ),
            PanelSelector(
                registration_ids=("core.image.viewer",),
                titles=("Image Viewer",),
            ),
            PanelSelector(
                registration_ids=("core.annotations.panel",),
                titles=("Annotations",),
            ),
        ),
    ),
    QuickWorkspaceTemplate(
        id="machine_learning",
        title="Machine learning",
        description="Recipe launcher, curves, predictor and resource monitor.",
        panels=(
            PanelSelector(
                registration_ids=("core.ml.recipe_launcher",),
                titles=("Recipe Launcher",),
            ),
            PanelSelector(
                registration_ids=("core.ml.training_curves",),
                titles=("Training Curves",),
            ),
            PanelSelector(
                registration_ids=("core.ml.predictor",),
                titles=("Predictor",),
            ),
            PanelSelector(
                registration_ids=("core.resources.monitor",),
                titles=("Resource Monitor",),
            ),
        ),
    ),
    QuickWorkspaceTemplate(
        id="active_learning",
        title="Active-learning review",
        description="Active-learning workflow with image and training context.",
        panels=(
            PanelSelector(
                registration_ids=("core.active_learning.panel",),
                titles=("Active Learning",),
            ),
            PanelSelector(
                registration_ids=("core.image.selection_gallery",),
                titles=("Selection Gallery",),
            ),
            PanelSelector(
                registration_ids=("core.ml.training_curves",),
                titles=("Training Curves",),
            ),
        ),
    ),
)


class ApplicationToolbarController:
    """Second-row application toolbar for record-level workflows."""

    def __init__(self, *, context: Any, template: Any = None) -> None:
        self.context = context
        self.template = template
        self.navigation: RecordNavigationManager = context.navigation
        self._subscriptions: list[Any] = []
        self._watchers: list[Any] = []
        self._disposed = False
        self._search_handle: Any = None
        self._last_displayed_row_id = ""
        self._position_resolution_key: tuple[str, str, str] | None = None
        self._opening_panel_registrations: set[str] = set()
        self._template_by_id = {item.id: item for item in BUILT_IN_TEMPLATES}

        self._install_css()
        self._build_widgets()
        self._build_view()
        self._subscribe()
        self._render_state(self.navigation.refresh())

    def _build_widgets(self) -> None:
        self.previous_button = self._icon_button(
            icon="chevron-left",
            description="Previous record in the current scope",
        )
        self.next_button = self._icon_button(
            icon="chevron-right",
            description="Next record in the current scope",
        )
        self.previous_button.on_click(lambda _event: self._run_navigation(self.navigation.previous))
        self.next_button.on_click(lambda _event: self._run_navigation(self.navigation.next))

        self.search_input = pn.widgets.TextInput(
            name="",
            placeholder="Go to record ID…",
            min_width=110,
            max_width=260,
            height=30,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            css_classes=["al-toolbar-search"],
            stylesheets=[_TOOLBAR_TEXT_INPUT_STYLESHEET],
        )
        self.search_input.description = "Enter an exact record ID and press Enter"
        self.search_button = self._icon_button(
            icon="search",
            description="Find record ID",
        )
        self.search_button.on_click(lambda _event: self._submit_search())
        self._watchers.append(
            self.search_input.param.watch(
                lambda _event: self._submit_search(),
                "enter_pressed",
            )
        )

        self.position_pane = pn.pane.HTML(
            "",
            width=82,
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-toolbar-position"],
        )
        self.scope_select = pn.widgets.Select(
            name="",
            options={"All records": "dataset", "Selection": "selection"},
            value="dataset",
            width=112,
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-toolbar-scope"],
            stylesheets=[_TOOLBAR_SELECT_STYLESHEET],
        )
        self._watchers.append(
            self.scope_select.param.watch(self._on_scope_changed, "value")
        )

        self.selection_button = pn.widgets.Button(
            name="Select",
            icon="square-plus",
            button_type="default",
            width=86,
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-toolbar-selection-button", "al-toolbar-secondary"],
            stylesheets=[_TOOLBAR_SELECTION_STYLESHEET],
        )
        self.selection_button.on_click(lambda _event: self._toggle_selection())

        self.quick_layout_button = pn.widgets.Button(
            name="Layouts",
            icon="layout-dashboard",
            button_type="default",
            width=98,
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-toolbar-menu", "al-toolbar-secondary"],
            stylesheets=[_TOOLBAR_MENU_STYLESHEET],
        )
        self.quick_layout_button.description = "Open a standard set of panels"
        self.quick_layout_button.on_click(
            lambda _event: self._toggle_popup("layouts")
        )
        self.quick_layout_buttons: list[Any] = []
        for item in BUILT_IN_TEMPLATES:
            button = self._popup_button(
                name=item.title,
                description=item.description,
            )
            button.on_click(
                lambda _event, template_id=item.id: self._apply_quick_layout(
                    template_id
                )
            )
            self.quick_layout_buttons.append(button)
        self.quick_layout_popup = pn.Column(
            *self.quick_layout_buttons,
            visible=False,
            width=204,
            margin=(0, 0, 0, 0),
            css_classes=["al-toolbar-popover", "al-toolbar-layout-popover"],
            styles={
                "position": "absolute",
                "top": "34px",
                "right": "0",
                "z-index": "2400",
                "overflow": "visible",
            },
        )
        self.quick_layout_menu = pn.Column(
            self.quick_layout_button,
            self.quick_layout_popup,
            width=98,
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-toolbar-menu-wrapper"],
            styles={"position": "relative", "overflow": "visible"},
        )

        self.overflow_button = self._icon_button(
            icon="dots-vertical",
            description="More record and selection actions",
        )
        self.overflow_button.on_click(
            lambda _event: self._toggle_popup("overflow")
        )
        self.first_record_button = self._popup_button(name="First record")
        self.last_record_button = self._popup_button(name="Last record")
        self.clear_selection_button = self._popup_button(name="Clear selection")
        self.open_record_browser_button = self._popup_button(
            name="Open Record Browser"
        )
        self.first_record_button.on_click(
            lambda _event: self._run_popup_action(
                self.navigation.first,
                popup="overflow",
            )
        )
        self.last_record_button.on_click(
            lambda _event: self._run_popup_action(
                self.navigation.last,
                popup="overflow",
            )
        )
        self.clear_selection_button.on_click(
            lambda _event: self._run_popup_action(
                self.navigation.clear_selection,
                popup="overflow",
            )
        )
        self.open_record_browser_button.on_click(
            lambda _event: self._run_popup_callback(
                lambda: self._open_panel_registration(
                    "core.record_browser.panel"
                ),
                popup="overflow",
            )
        )
        self.overflow_popup = pn.Column(
            self.first_record_button,
            self.last_record_button,
            self._popup_divider(),
            self.clear_selection_button,
            self.open_record_browser_button,
            visible=False,
            width=188,
            margin=(0, 0, 0, 0),
            css_classes=["al-toolbar-popover", "al-toolbar-overflow-popover"],
            styles={
                "position": "absolute",
                "top": "34px",
                "right": "0",
                "z-index": "2400",
                "overflow": "visible",
            },
        )
        self.overflow_menu = pn.Column(
            self.overflow_button,
            self.overflow_popup,
            width=30,
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-toolbar-menu-wrapper"],
            styles={"position": "relative", "overflow": "visible"},
        )

        # Retained as an internal status target for search progress. Notifications
        # carry user-facing feedback, so this is intentionally not in the layout.
        self.status_pane = pn.pane.HTML("", visible=False)

    def _build_view(self) -> None:
        self.navigation_group = pn.Row(
            self.previous_button,
            self.next_button,
            self._divider(),
            self.search_input,
            self.search_button,
            self._divider(),
            self.position_pane,
            css_classes=["al-toolbar-group", "al-toolbar-navigation"],
            sizing_mode="stretch_width",
            height=30,
            margin=(0, 0, 0, 0),
            styles={"min-width": "0", "overflow": "visible"},
        )
        self.scope_group = pn.Row(
            self.scope_select,
            self.selection_button,
            css_classes=["al-toolbar-group", "al-toolbar-scope-group"],
            width=201,
            height=30,
            margin=(0, 0, 0, 8),
            styles={"min-width": "0", "overflow": "visible"},
        )
        self.workspace_group = pn.Row(
            self.quick_layout_menu,
            self.overflow_menu,
            css_classes=["al-toolbar-group", "al-toolbar-workspace-group"],
            width=136,
            height=30,
            margin=(0, 0, 0, 0),
            styles={"min-width": "0", "overflow": "visible", "margin-left": "auto"},
        )
        toolbar_card = pn.Row(
            self.navigation_group,
            self.scope_group,
            self.workspace_group,
            sizing_mode="stretch_width",
            height=44,
            margin=(0, 0, 0, 0),
            align="center",
            css_classes=["al-application-toolbar"],
            styles={
                "display": "flex",
                "align-items": "center",
                "gap": "0",
                "min-width": "0",
                "max-width": "100%",
                "overflow": "visible",
            },
        )
        self.view = pn.Row(
            toolbar_card,
            sizing_mode="stretch_width",
            height=62,
            margin=(0, 0, 0, 0),
            css_classes=["al-application-toolbar-shell"],
            styles={
                "min-width": "0",
                "max-width": "100%",
                "overflow": "visible",
                "position": "fixed",
                "top": "var(--al-application-header-height)",
                "left": "0",
                "right": "0",
                "z-index": "1200",
            },
        )

    def _subscribe(self) -> None:
        self._subscriptions.append(
            self.context.events.subscribe(
                "navigation.state.changed",
                self._on_navigation_state,
                owner_id="platform.application_toolbar",
                owner_label="Application toolbar",
                owner_kind="application_chrome",
            )
        )

    def _on_navigation_state(self, _topic: str, payload: Any) -> None:
        if self._disposed:
            return
        if isinstance(payload, NavigationState):
            state = payload
        elif isinstance(payload, dict):
            try:
                state = NavigationState(**payload)
            except TypeError:
                state = self.navigation.get_state()
        else:
            state = self.navigation.get_state()
        self._schedule_ui(lambda: self._render_state(state))

    def _render_state(self, state: NavigationState) -> None:
        if self._disposed:
            return
        has_dataset = state.dataset_id is not None
        has_mapping = state.id_column is not None
        can_start = has_dataset and has_mapping and state.row_count > 0

        self.navigation_group.visible = has_dataset
        self.scope_group.visible = has_dataset

        self.previous_button.disabled = not (
            can_start and (state.can_previous or state.position is None)
        )
        self.next_button.disabled = not (
            can_start and (state.can_next or state.position is None)
        )
        self.search_input.disabled = not (has_dataset and has_mapping)
        self.search_button.disabled = self.search_input.disabled
        self.scope_select.disabled = not has_dataset
        self.scope_select.value = state.scope
        self.selection_button.disabled = not state.has_focus
        self.selection_button.visible = state.has_focus
        self.first_record_button.disabled = not (
            can_start and (state.can_previous or state.position is None)
        )
        self.last_record_button.disabled = not (
            can_start and (state.can_next or state.position is None)
        )
        self.clear_selection_button.disabled = state.selection_count == 0

        if state.can_remove_from_selection:
            self.selection_button.name = "Selected"
            self.selection_button.icon = "square-minus"
            self.selection_button.description = "Remove the focused record from the active selection"
            self.selection_button.css_classes = [
                "al-toolbar-selection-button",
                "al-is-selected",
            ]
        else:
            self.selection_button.name = "Select"
            self.selection_button.icon = "square-plus"
            self.selection_button.description = "Add the focused record to the active selection"
            self.selection_button.css_classes = [
                "al-toolbar-selection-button",
            ]

        if state.position is None:
            current = "—"
        else:
            current = f"{state.position + 1:,}"

        total = (
            "—"
            if state.row_count <= 0
            else f"{state.row_count:,}"
        )
        self.position_pane.object = (
            '<span class="al-position-current">'
            + current
            + '</span><span class="al-position-separator"> / </span>'
            + '<span class="al-position-total">'
            + total
            + "</span>"
        )
        row_id_text = "" if state.row_id is None else str(state.row_id)
        current_input = str(getattr(self.search_input, "value_input", "") or "")
        if not current_input or current_input == self._last_displayed_row_id:
            self.search_input.value = row_id_text
            try:
                self.search_input.value_input = row_id_text
            except Exception:
                pass
            self._last_displayed_row_id = row_id_text

        self.search_input.description = state.message
        self.scope_select.description = (
            f"Navigate all {state.dataset_row_count:,} records"
            if state.scope == "dataset"
            else f"Navigate the {state.selection_count:,} selected records"
        )
        self._ensure_focus_position(state)

    @staticmethod
    def _record_id_token(row_id: Any) -> str:
        return (
            f"{type(row_id).__module__}."
            f"{type(row_id).__qualname__}:"
            f"{row_id!r}"
        )

    def _ensure_focus_position(self, state: NavigationState) -> None:
        if (
            self._disposed
            or not state.has_focus
            or state.position is not None
            or state.dataset_id is None
            or state.row_id is None
            or state.id_column is None
        ):
            self._position_resolution_key = None
            return
        row_id_token = self._record_id_token(state.row_id)
        key = (state.dataset_id, row_id_token, state.scope)
        if self._position_resolution_key == key:
            return
        self._position_resolution_key = key

        def _resolve(*, cancel_token: Any) -> NavigationState:
            if cancel_token is not None and cancel_token.cancelled():
                return self.navigation.get_state()
            return self.navigation.resolve_focus_position()

        def _finish(_result: Any = None) -> None:
            if self._position_resolution_key == key:
                self._position_resolution_key = None

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                _finish(_resolve(cancel_token=None))
            except Exception:
                _finish()
            return
        jobs.submit(
            _resolve,
            title="Resolve focused record position",
            key=f"application-toolbar:position:{state.dataset_id}:{state.scope}:{row_id_token}",
            on_done=_finish,
            on_error=lambda _exc: _finish(),
        )

    def _submit_search(self) -> None:
        query = str(
            getattr(self.search_input, "value_input", None)
            or self.search_input.value
            or ""
        ).strip()
        if not query:
            self._notify("Enter a record ID.", level="warning")
            return

        if self._search_handle is not None:
            try:
                self._search_handle.cancel()
            except Exception:
                pass
        self._set_search_busy(True, f"Finding {query}…")

        def _find(*, cancel_token: Any) -> Optional[int]:
            if cancel_token is not None and cancel_token.cancelled():
                return None
            return self.navigation.find_position(query)

        def _done(position: Optional[int]) -> None:
            self._search_handle = None
            self._set_search_busy(False)
            if position is None:
                self._notify(
                    f"No record with ID {query!r} was found in the current scope.",
                    level="warning",
                )
                return
            try:
                self.navigation.go_to_position(position, origin="application_toolbar.search")
            except Exception as exc:
                self._notify(str(exc), level="error")

        def _error(exc: BaseException) -> None:
            self._search_handle = None
            self._set_search_busy(False)
            self._notify(f"Record search failed: {exc}", level="error")

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                _done(_find(cancel_token=None))
            except BaseException as exc:
                _error(exc)
            return
        self._search_handle = jobs.submit(
            _find,
            title=f"Find record {query}",
            key="application-toolbar:record-search",
            on_done=_done,
            on_error=_error,
        )

    def _set_search_busy(self, busy: bool, status: str = "") -> None:
        self.search_button.loading = busy
        self.search_input.disabled = busy or self.navigation.get_state().id_column is None
        self.search_button.disabled = self.search_input.disabled
        classes = ["al-application-toolbar-shell"]
        if busy:
            classes.append("al-toolbar-busy")
        self.view.css_classes = classes
        if status:
            self.status_pane.object = html.escape(status)

    def _on_scope_changed(self, event: Any) -> None:
        if self._disposed:
            return
        try:
            self.navigation.set_scope(event.new, origin="application_toolbar.scope")
        except Exception as exc:
            self._notify(str(exc), level="error")

    def _toggle_selection(self) -> None:
        state = self.navigation.get_state()
        if state.can_remove_from_selection:
            self._run_navigation(self.navigation.remove_focus_from_selection)
        else:
            self._run_navigation(self.navigation.add_focus_to_selection)

    def _run_navigation(self, operation: Any) -> None:
        try:
            operation(origin="application_toolbar")
        except NavigationError as exc:
            self._notify(str(exc), level="warning")
        except Exception as exc:
            self._notify(str(exc), level="error")

    def _toggle_popup(self, popup: str) -> None:
        if self._disposed:
            return
        if popup == "layouts":
            show = not self.quick_layout_popup.visible
            self.overflow_popup.visible = False
            self.quick_layout_popup.visible = show
            return
        if popup == "overflow":
            show = not self.overflow_popup.visible
            self.quick_layout_popup.visible = False
            self.overflow_popup.visible = show

    def _close_popups(self) -> None:
        self.quick_layout_popup.visible = False
        self.overflow_popup.visible = False

    def _run_popup_action(self, operation: Any, *, popup: str) -> None:
        self._close_popups()
        self._run_navigation(operation)

    def _run_popup_callback(self, callback: Any, *, popup: str) -> None:
        self._close_popups()
        callback()

    def _apply_quick_layout(self, template_id: str) -> None:
        self._close_popups()
        template = self._template_by_id.get(str(template_id))
        if template is None:
            return
        self.quick_layout_button.disabled = True
        for button in self.quick_layout_buttons:
            button.disabled = True
        try:
            opened, missing = self._apply_template(template)
        except Exception as exc:
            self._notify(f"Could not apply {template.title}: {exc}", level="error")
            return
        finally:
            self.quick_layout_button.disabled = False
            for button in self.quick_layout_buttons:
                button.disabled = False

        if opened:
            message = f"Opened {opened} panel{'s' if opened != 1 else ''} for {template.title}."
            if missing:
                message += " Some optional panels are unavailable."
            self._notify(message, level="success")
        elif missing:
            self._notify(
                f"No panels for {template.title} are available in the enabled plugins.",
                level="warning",
            )
        else:
            self._notify(f"{template.title} panels are already open.", level="info")

    def _apply_template(self, template: QuickWorkspaceTemplate) -> tuple[int, int]:
        plugins = getattr(self.context, "plugins", None)
        workspace = getattr(self.context, "workspace", None)
        if plugins is None or workspace is None:
            raise RuntimeError("Plugin manager and workspace are required for quick workspaces.")

        registrations = list(plugins.list_panels())
        opened = 0
        missing = 0
        for selector in template.panels:
            registration = self._resolve_panel_selector(selector, registrations)
            if registration is None:
                missing += 1
                continue
            if self._is_panel_open(registration, selector=selector):
                continue

            registration_id = str(registration.id)
            self._opening_panel_registrations.add(registration_id)
            try:
                # open_panel() normally registers the panel synchronously. The
                # in-flight set also prevents duplicate opens if the user clicks
                # the layout menu repeatedly while a mapping-gated panel is
                # still being constructed.
                plugins.open_panel(
                    registration_id,
                    context=self.context,
                    layout_item=self._layout_hint_without_position(registration),
                )
                opened += 1
            finally:
                self._schedule_ui(
                    lambda registration_id=registration_id: (
                        self._opening_panel_registrations.discard(registration_id)
                    )
                )
        return opened, missing

    @staticmethod
    def _layout_hint_without_position(registration: Any) -> dict[str, Any]:
        """Preserve panel size constraints while discarding template geometry.

        Quick workspaces are panel bundles, not saved layouts. WorkspaceManager
        should therefore place every new panel through its normal append/
        collision-avoidance path instead of honoring a registration's x/y hint.
        """
        default_layout = getattr(registration, "default_layout", None)
        if not isinstance(default_layout, dict):
            return {}
        return {
            key: default_layout[key]
            for key in ("w", "h", "minW", "minH", "maxW", "maxH", "static")
            if key in default_layout
        }

    def _is_panel_open(
        self,
        registration: Any,
        *,
        selector: Optional[PanelSelector] = None,
    ) -> bool:
        registration_id = str(getattr(registration, "id", "") or "")
        if not registration_id:
            return False
        if registration_id in self._opening_panel_registrations:
            return True

        expected_titles = {
            str(getattr(registration, "title", "") or "").strip().casefold()
        }
        if selector is not None:
            expected_titles.update(title.strip().casefold() for title in selector.titles)
        expected_titles.discard("")

        workspace = getattr(self.context, "workspace", None)
        if workspace is None:
            return False

        try:
            records = list(workspace.list_panels().values())
        except Exception:
            records = []

        for record in records:
            targets = (
                record,
                getattr(record, "view", None),
                getattr(record, "controller", None),
            )
            record_registration_ids = {
                str(value)
                for target in targets
                if target is not None
                for value in (
                    getattr(target, "registration_id", None),
                    getattr(target, "_al_registration_id", None),
                )
                if value
            }
            if registration_id in record_registration_ids:
                return True

            # Older or startup-created panels may not yet have registration
            # metadata. Use title matching only as a compatibility fallback so
            # two unrelated registered panels with similar titles are not
            # treated as the same panel.
            if not record_registration_ids and expected_titles:
                record_title = str(
                    getattr(record, "title", None)
                    or getattr(getattr(record, "view", None), "_al_title", None)
                    or ""
                ).strip().casefold()
                if record_title in expected_titles:
                    return True

        # A panel can briefly exist in the live grid before register_existing()
        # has populated WorkspaceManager._panels. Inspect view metadata as a
        # final race-safe check.
        grid = getattr(workspace, "grid", None)
        for view in list(getattr(grid, "objects", None) or []):
            if str(getattr(view, "_al_registration_id", "") or "") == registration_id:
                return True
        return False

    @staticmethod
    def _resolve_panel_selector(selector: PanelSelector, registrations: Sequence[Any]) -> Any:
        by_id = {str(reg.id): reg for reg in registrations}
        for registration_id in selector.registration_ids:
            if registration_id in by_id:
                return by_id[registration_id]
        wanted_titles = {title.casefold() for title in selector.titles}
        for registration in registrations:
            if str(getattr(registration, "title", "")).casefold() in wanted_titles:
                return registration
        return None

    def _open_panel_registration(self, registration_id: str) -> None:
        try:
            registrations = list(self.context.plugins.list_panels())
            registration = next(
                (item for item in registrations if str(item.id) == str(registration_id)),
                None,
            )
            if registration is not None and self._is_panel_open(registration):
                self._notify("That panel is already open.", level="info")
                return
            self.context.plugins.open_panel(registration_id, context=self.context)
        except Exception as exc:
            self._notify(f"Could not open panel: {exc}", level="error")

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        try:
            self._close_popups()
        except Exception:
            pass
        if self._search_handle is not None:
            try:
                self._search_handle.cancel()
            except Exception:
                pass
        for subscription in list(self._subscriptions):
            try:
                self.context.events.unsubscribe(subscription)
            except Exception:
                pass
        self._subscriptions.clear()
        for watcher in list(self._watchers):
            try:
                watcher.inst.param.unwatch(watcher)
            except Exception:
                try:
                    watcher.obj.param.unwatch(watcher)
                except Exception:
                    pass
        self._watchers.clear()

    @staticmethod
    def _popup_button(*, name: str, description: str = "") -> Any:
        button = pn.widgets.Button(
            name=name,
            button_type="default",
            height=32,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            css_classes=["al-toolbar-popup-item"],
            stylesheets=[_TOOLBAR_POPUP_ITEM_STYLESHEET],
        )
        if description:
            button.description = description
        return button

    @staticmethod
    def _popup_divider() -> Any:
        return pn.pane.HTML(
            "",
            height=1,
            sizing_mode="stretch_width",
            margin=(4, 6, 4, 6),
            css_classes=["al-toolbar-popover-divider"],
        )

    @staticmethod
    def _icon_button(*, icon: str, description: str) -> Any:
        button = pn.widgets.Button(
            name="",
            icon=icon,
            button_type="default",
            width=30,
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-toolbar-icon-button", "al-toolbar-icon-only"],
            stylesheets=[_TOOLBAR_ICON_BUTTON_STYLESHEET],
        )
        button.description = description
        return button

    @staticmethod
    def _divider() -> Any:
        return pn.pane.HTML(
            "",
            width=1,
            height=20,
            margin=(5, 2, 5, 2),
            css_classes=["al-toolbar-inner-divider"],
        )

    @staticmethod
    def _schedule_ui(callback: Any) -> None:
        try:
            doc = pn.state.curdoc
        except Exception:
            doc = None
        if doc is None:
            callback()
            return
        try:
            doc.add_next_tick_callback(callback)
        except Exception:
            callback()

    @staticmethod
    def _notify(message: str, *, level: str = "info") -> None:
        notifications = getattr(pn.state, "notifications", None)
        method = getattr(notifications, level, None) if notifications is not None else None
        if callable(method):
            try:
                method(str(message), duration=4500)
                return
            except Exception:
                pass
        print(f"[ApplicationToolbar][{level}] {message}")

    @staticmethod
    def _install_css() -> None:
        marker = "--al-toolbar-v8"
        if any(marker in css for css in pn.config.raw_css):
            return
        pn.config.raw_css.append(TOOLBAR_CSS)