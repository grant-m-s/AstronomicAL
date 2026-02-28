from astronomicAL.extensions import extension_plots
from astronomicAL.extensions import custom_plots
from bokeh.models.widgets import Dropdown
from functools import partial

import panel as pn


class MenuDashboard:
    """A Dashboard used to dynamically choose which view to display.

    Parameters
    ----------
    main : Dashboard
        The parent Dashboard view required for updating which dashboard is
        rendered.

    Attributes
    ----------
    row : Panel Row
        The panel is housed in a row which can then be rendered by the
        parent Dashboard.

    """

    def __init__(self, main, context=None):
        
        self.context = context
        self.row = pn.Row(pn.pane.Str("loading"))

        plot_options = [
            "Basic Plot",
            "Histogram Plot",
            "Density Plot",
            "Selected Source Info",
        ] + list(custom_plots.get_customplot_dict().keys()) +  list(extension_plots.get_plot_dict().keys())

        dd = Dropdown(label="Choose Plot Type:", menu=plot_options)

        from bokeh.models import InlineStyleSheet

        dd.stylesheets = [InlineStyleSheet(css="""
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
        .bk-btn:hover{
        border-color: rgba(0,0,0,.28);
        box-shadow: 0 2px 6px rgba(0,0,0,.10);
        }
        .bk-btn:active{
        transform: translateY(1px);
        }
        .bk-caret{
        margin-left: 10px;
        opacity: .75;
        }

        /* Menu */
        .bk-menu{
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,.18);
        box-shadow: 0 10px 24px rgba(0,0,0,.14);
        padding: 6px;
        min-width: 50px;
        max-height: 150px;
        overflow-y: auto;
        }
        .bk-menu a{
        font-size: 14px;
        padding: 10px 10px;
        border-radius: 10px;
        }
        .bk-menu a:hover{
        background: rgba(0,0,0,.06);
        }
        """)]

        dd.on_click(
            partial(
                self._update_main_contents,
                main=main,
            ),
        )

        self._plot_selection = pn.pane.Bokeh(dd)


    def _update_main_contents(self, event, main):
        self._plot_selection.label = "Loading..."

        main.set_contents(event.item)

    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which can then be rendered by the
            parent Dashboard.

        """
        self.row[0] = pn.Column(
            pn.layout.VSpacer(min_height=20,max_height=20),
            pn.Row(pn.Spacer(min_width=20,max_width=500),self._plot_selection,pn.Spacer(min_width=20,max_width=500)),
            pn.layout.VSpacer(min_height=20,max_height=60),
            sizing_mode='stretch_height'
        )
        return self.row
