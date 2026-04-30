from __future__ import annotations

from functools import partial

import panel as pn
from bokeh.models.widgets import Dropdown
from bokeh.models import InlineStyleSheet

from astronomicAL.extensions import extension_plots
from astronomicAL.extensions import custom_plots
from astronomicAL.utils.debug import boot_print


class MenuDashboard:
    """Dashboard used to dynamically choose which view to display."""

    def __init__(self, main, context=None):
        self.main = main
        self.context = context if context is not None else getattr(main, "context", None)
        self.row = pn.Row(pn.pane.Str("loading"))
        self._subscriptions = []
        self._disposed = False

        boot_print("MenuDashboard.__init__: start")
        boot_print(f"MenuDashboard.__init__: context_present={self.context is not None}")
        boot_print(
            "MenuDashboard.__init__: plugins_present="
            f"{getattr(self.context, 'plugins', None) is not None if self.context is not None else False}"
        )

        self._dropdown = Dropdown(
            label="Choose Plot Type:",
            menu=self._build_plot_options(),
        )

        self._dropdown.stylesheets = [
            InlineStyleSheet(
                css="""
                /* Button */
                .bk-btn {
                    font-size: 14px;
                    padding: 10px 14px;
                    border-radius: 12px;
                    border: 1px solid rgba(0,0,0,.18);
                    background: rgba(255,255,255,.95);
                    box-shadow: 0 1px 2px rgba(0,0,0,.06);
                    min-height: 40px;
                }

                .bk-btn:hover {
                    border-color: rgba(0,0,0,.28);
                    box-shadow: 0 2px 6px rgba(0,0,0,.10);
                }

                .bk-btn:active {
                    transform: translateY(1px);
                }

                .bk-caret {
                    margin-left: 10px;
                    opacity: .75;
                }

                /* Menu */
                .bk-menu {
                    border-radius: 12px;
                    border: 1px solid rgba(0,0,0,.18);
                    box-shadow: 0 10px 24px rgba(0,0,0,.14);
                    padding: 6px;
                    min-width: 50px;
                    max-height: 150px;
                    overflow-y: auto;
                }

                .bk-menu a {
                    font-size: 14px;
                    padding: 10px 10px;
                    border-radius: 10px;
                }

                .bk-menu a:hover {
                    background: rgba(0,0,0,.06);
                }
                """
            )
        ]

        self._dropdown.on_click(
            partial(
                self._update_main_contents,
                main=main,
            )
        )

        self._plot_selection = pn.pane.Bokeh(self._dropdown)

        self._subscribe_to_plugin_events()

    # ------------------------------------------------------------------
    # Menu construction / refresh
    # ------------------------------------------------------------------

    def _build_plot_options(self):
        """Build menu options from static plots, legacy custom plots, and plugins."""

        base_options = [
            "Basic Plot",
            "Histogram Plot",
            "Density Plot",
            "Selected Source Info",
        ]

        custom_options = list(
            custom_plots.get_customplot_dict(context=self.context).keys()
        )

        extension_options = list(extension_plots.get_plot_dict().keys())

        seen = set()
        options = []

        for item in base_options + custom_options + extension_options:
            if item not in seen:
                seen.add(item)
                options.append(item)

        boot_print(f"MenuDashboard._build_plot_options: base_options={base_options}")
        boot_print(f"MenuDashboard._build_plot_options: custom_options={custom_options}")
        boot_print(f"MenuDashboard._build_plot_options: extension_options={extension_options}")
        boot_print(f"MenuDashboard._build_plot_options: final_options={options}")

        return options

    def refresh_plot_options(self) -> None:
        """Refresh an already-created Bokeh Dropdown menu."""

        if self._disposed:
            return

        options = self._build_plot_options()

        try:
            self._dropdown.menu = options
        except Exception:
            # Defensive fallback. If Bokeh rejects a live menu update, recreate
            # the pane object. This should be rare.
            self._dropdown = Dropdown(label="Choose Plot Type:", menu=options)
            self._dropdown.on_click(
                partial(
                    self._update_main_contents,
                    main=self.main,
                )
            )
            self._plot_selection.object = self._dropdown

    def _subscribe_to_plugin_events(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        for topic in ("plugin.enabled", "plugin.disabled", "plugin.reloaded"):
            try:
                sub = events.subscribe(
                    topic,
                    self._on_plugin_registry_changed,
                    owner_id="menu.dashboard",
                    owner_label="Menu Dashboard",
                    owner_kind="dashboard",
                )
            except TypeError:
                sub = events.subscribe(topic, self._on_plugin_registry_changed)

            self._subscriptions.append(sub)

    def _on_plugin_registry_changed(self, topic, payload) -> None:
        if self._disposed:
            return

        def _refresh():
            if not self._disposed:
                self.refresh_plot_options()

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(_refresh)
            else:
                _refresh()
        except Exception:
            _refresh()

    # ------------------------------------------------------------------
    # Toolbar/content
    # ------------------------------------------------------------------

    def get_toolbar(self):
        return pn.Spacer(height=1)

    def _update_main_contents(self, event, main):
        selected = event.item

        # Critical stale-menu guard:
        # The menu may have been constructed before a plugin was disabled.
        # Rebuild the options at click-time and refuse invalid selections.
        current_options = set(self._build_plot_options())

        if selected not in current_options:
            self.refresh_plot_options()
            self._dropdown.label = "Choose Plot Type:"
            main.show_message(
                title="Plot unavailable",
                message=(
                    f"`{selected}` is no longer available. "
                    "It may belong to a plugin that has been disabled."
                ),
            )
            return

        self._dropdown.label = "Loading..."
        main.set_contents(selected)
        self._dropdown.label = "Choose Plot Type:"

    def panel(self):
        """Render the current view."""

        self.row[0] = pn.Column(
            pn.layout.VSpacer(min_height=20, max_height=20),
            pn.Row(
                pn.Spacer(min_width=20, max_width=500),
                self._plot_selection,
                pn.Spacer(min_width=20, max_width=500),
            ),
            pn.layout.VSpacer(min_height=20, max_height=60),
            sizing_mode="stretch_height",
        )

        return self.row

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