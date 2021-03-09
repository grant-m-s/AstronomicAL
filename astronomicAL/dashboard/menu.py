from astronomicAL.extensions import extension_plots
from functools import partial

import panel as pn
import param


class MenuDashboard(param.Parameterized):
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

    def __init__(self, main, **params):
        super(MenuDashboard, self).__init__(**params)

        self.row = pn.Row(pn.pane.Str("loading"))

        plot_options = [
            "Basic Plot",
            "Selected Source Info",
        ] + list(extension_plots.get_plot_dict().keys())

        self._plot_selection = pn.widgets.Select(
            name="Choose plot type:", options=plot_options
        )

        self._add_plot_button = pn.widgets.Button(name="Add Plot")
        self._add_plot_button.on_click(
            partial(
                self._update_main_contents,
                main=main,
                button=self._add_plot_button,
            )
        )

        # self._add_selected_info_button = pn.widgets.Button(
        #     name="Add Selected Source Info"
        # )
        # self._add_selected_info_button.on_click(
        #     partial(
        #         self._update_main_contents,
        #         main=main,
        #         updated="Selected Source",
        #         button=self._add_selected_info_button,
        #     )
        # )

    def _update_main_contents(self, event, main, button):
        # print(updated)
        button.name = "Loading..."

        main.set_contents(self._plot_selection.value)

    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which can then be rendered by the
            parent Dashboard.

        """
        self.row[0] = pn.Column(
            self._plot_selection, self._add_plot_button, max_height=100
        )
        return self.row
