from functools import partial

import panel as pn
import param


class MenuDashboard(param.Parameterized):

    contents = param.String()

    def __init__(self, main, **params):
        super(MenuDashboard, self).__init__(**params)

        self.row = pn.Row(pn.pane.Str("loading"))

        self.add_plot_button = pn.widgets.Button(name="Add Plot")
        self.add_plot_button.on_click(
            partial(self.update_main_contents, main=main,
                    updated="Plot", button=self.add_plot_button))

        self.add_selected_info_button = pn.widgets.Button(
            name="Add Selected Source Info"
        )
        self.add_selected_info_button.on_click(
            partial(self.update_main_contents,
                    main=main,
                    updated="Selected Source",
                    button=self.add_selected_info_button)
                    )

    def update_main_contents(self, event, main, updated, button):
        # print(updated)
        button.name = "Loading..."
        main.set_contents(updated)

    def panel(self):

        self.row[0] = pn.Column(
            self.add_plot_button,
            self.add_selected_info_button,
        )
        return self.row
