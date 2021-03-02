from bokeh.models import ColumnDataSource
from extensions import extension_plots
from dashboard.active_learning import ActiveLearningDashboard
from dashboard.menu import MenuDashboard
from dashboard.plot import PlotDashboard
from dashboard.selected_source import SelectedSourceDashboard
from dashboard.settings_dashboard import SettingsDashboard

import config
import panel as pn
import param


class Dashboard(param.Parameterized):
    """Top-level Dashboard which holds an instance of any other Dashboard.

    This will initialize and render any of the other instances of the
    dashboards and handles the interactions of switching from one dashboard
    view to another.

    Parameters
    ----------
    src : ColumnDataSource
        The shared data source which holds the current selected source.
    contents : param.String, default = "Menu"
        The identifier for which of the dashboard views should be initialised
        and rendered.

    Attributes
    ----------
    row : Panel Row
        The panel is housed in a row which can then be rendered by the
        Panel layout.
    df : DataFrame
        The shared dataframe which holds all the data.

    """

    src = ColumnDataSource(data={"0": [], "1": []})

    contents = param.String()

    def __init__(self, src, contents="Menu", **params):
        super(Dashboard, self).__init__(**params)

        self.src = src
        self.src.on_change("data", self._update_extension_plots_cb)
        self.row = pn.Row(pn.pane.Str("loading"))
        self.df = config.main_df

        self._close_button = pn.widgets.Button(name="Close", max_width=100)
        self._close_button.on_click(self._close_button_cb)

        self.contents = contents

    def _close_button_cb(self, event):
        self.contents = "Menu"

    def _update_extension_plots_cb(self, attr, old, new):
        plot_dict = extension_plots.get_plot_dict()
        if self.contents in list(plot_dict.keys()):
            self.panel_contents = plot_dict[self.contents](config.main_df, self.src)
            self.panel()

    @param.depends("contents", watch=True)
    def _update_contents(self):

        print("Updating contents")

        if self.contents == "Settings":

            self.panel_contents = SettingsDashboard(self, self.src, self.df)

        elif self.contents == "Menu":

            self.panel_contents = MenuDashboard(self)

        elif self.contents == "Active Learning":

            self.df = config.main_df
            self.panel_contents = ActiveLearningDashboard(self.src, self.df)

        elif self.contents == "Basic Plot":
            if not config.settings["confirmed"]:
                self.contents = "Menu"
                print("Please Complete Settings before accessing this view.")
                return
            self.panel_contents = PlotDashboard(self.src)

        elif self.contents == "Selected Source Info":
            if not config.settings["confirmed"]:
                self.contents = "Menu"
                print("Please Complete Settings before accessing this view.")
                return
            self.panel_contents = SelectedSourceDashboard(self.src)
        else:
            plot_dict = extension_plots.get_plot_dict()
            self.panel_contents = plot_dict[self.contents](config.main_df, self.src)

        print("Successfully updated contents")

        self.panel()

    def set_contents(self, updated):
        """Update the current dashboard by setting a new `contents`.

        Parameters
        ----------
        updated : str
            The new contents view required.

        Returns
        -------
        None

        """
        print(updated)
        self.contents = updated

    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel contents is housed in a row which can then be
            rendered by the Panel layout.

        """
        if hasattr(self.panel_contents, "panel"):
            self.row[0] = self.panel_contents.panel()
        else:
            self.row[0] = pn.Card(
                self.panel_contents,
                header=pn.Row(self._close_button),
                collapsible=False,
            )

        print("Returned panel")

        return self.row
