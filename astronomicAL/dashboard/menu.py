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
        The panel is housed in a row which will can then be rendered by the
        parent Dashboard.

    """

    def __init__(self, main, **params):
        super(MenuDashboard, self).__init__(**params)

        self.row = pn.Row(pn.pane.Str("loading"))

        self._add_plot_button = pn.widgets.Button(name="Add Plot")
        self._add_plot_button.on_click(
            partial(self.update_main_contents, main=main,
                    updated="Plot", button=self._add_plot_button))

        self._add_selected_info_button = pn.widgets.Button(
            name="Add Selected Source Info"
        )
        self._add_selected_info_button.on_click(
            partial(self.update_main_contents,
                    main=main,
                    updated="Selected Source",
                    button=self._add_selected_info_button)
                    )

    def _update_main_contents(self, event, main, updated, button):
        # print(updated)
        button.name = "Loading..."
        main.set_contents(updated)

    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which will can then be rendered by the
            parent Dashboard.

        """
        self.row[0] = pn.Column(
            self._add_plot_button,
            self._add_selected_info_button,
        )
        return self.row
