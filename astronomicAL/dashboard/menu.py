from __future__ import annotations

from functools import partial

import panel as pn
from bokeh.models.widgets import Dropdown
from bokeh.models import InlineStyleSheet

from astronomicAL.extensions import extension_plots
from astronomicAL.extensions import custom_plots


class MenuDashboard:
    """Dashboard used to dynamically choose which view to display."""

    def __init__(self, main, context=None):
        self.context = context if context is not None else getattr(main, "context", None)
        self.row = pn.Row(pn.pane.Str("loading"))

        plot_options = self._build_plot_options()

        dd = Dropdown(label="Choose Plot Type:", menu=plot_options)

        dd.stylesheets = [
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

        dd.on_click(
            partial(
                self._update_main_contents,
                main=main,
            )
        )

        self._plot_selection = pn.pane.Bokeh(dd)

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

        # Preserve order while removing duplicates.
        seen = set()
        options = []
        for item in base_options + custom_options + extension_options:
            if item not in seen:
                seen.add(item)
                options.append(item)

        return options

    def get_toolbar(self):
        return pn.Spacer(height=1)

    def _update_main_contents(self, event, main):
        self._plot_selection.label = "Loading..."
        main.set_contents(event.item)

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