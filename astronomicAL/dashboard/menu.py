from astronomicAL.extensions import extension_plots
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

    def __init__(self, main):

        self.row = pn.Row(pn.pane.Str("loading"))

        plot_options = [
            "Basic Plot",
            "Histogram Plot",
            "Test Plot",
            "Selected Source Info",
        ] + list(extension_plots.get_plot_dict().keys())

        self._plot_selection = Dropdown(label="Choose plot type:", menu=plot_options)

        self._plot_selection.on_click(
            partial(
                self._update_main_contents,
                main=main,
            ),
        )

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
            pn.layout.VSpacer(),
            self._plot_selection,
            max_height=100,
        )
        return self.row
