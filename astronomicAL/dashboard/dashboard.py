from astronomicAL.dashboard.active_learning import ActiveLearningDashboard
from astronomicAL.dashboard.labelling import LabellingDashboard
from astronomicAL.dashboard.menu import MenuDashboard
from astronomicAL.dashboard.plot import PlotDashboard
from astronomicAL.dashboard.selected_source import SelectedSourceDashboard
from astronomicAL.dashboard.settings_dashboard import SettingsDashboard
from astronomicAL.extensions import extension_plots
from bokeh.models import ColumnDataSource

import astronomicAL.config as config
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

    def __init__(self, src, contents="Menu"):
        super(Dashboard, self).__init__()

        self.src = src
        self.src.on_change("data", self._update_extension_plots_cb)
        self.row = pn.Row(pn.pane.Str("loading"))
        self.df = config.main_df

        self._close_button = pn.widgets.Button(name="Close", max_width=100)
        self._close_button.on_click(self._close_button_cb)

        self._submit_button = pn.widgets.Button(name="Submit Column Names")
        self._submit_button.on_click(self._submit_button_cb)

        self.contents = contents

    def _submit_button_cb(self, event):
        self._submit_button.name = "Loading Plot..."
        self._submit_button.disabled = True
        if hasattr(self.plot_dict[self.contents], "col_selection"):
            for i, row in enumerate(self.plot_dict[self.contents].col_selection):
                if i == len(self.plot_dict[self.contents].col_selection) - 1:
                    continue
                for selector in row:
                    config.settings[selector.name] = selector.value
        else:
            self._submit_button.name = "Submit updated column names"
            self._submit_button.disabled = False
            self.plot_dict[self.contents]._load_file()
        self._update_contents()

    def _close_button_cb(self, event):
        self.contents = "Menu"

    def _update_extension_plots_cb(self, attr, old, new):
        plot_dict = extension_plots.get_plot_dict()
        if self.contents in list(plot_dict.keys()):
            self.panel_contents = plot_dict[self.contents].plot(self._submit_button)(
                config.main_df, self.src
            )
            self.panel()

    @param.depends("contents", watch=True)
    def _update_contents(self):

        if self.contents == "Settings":

            self.mode = ""
            self.panel_contents = SettingsDashboard(self, self.src)

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
            self.panel_contents = PlotDashboard(self.src, self._close_button)

        elif self.contents == "Labelling":

            self.df = config.main_df
            self.panel_contents = LabellingDashboard(self.src, self.df)

        elif self.contents == "Selected Source Info":
            if not config.settings["confirmed"]:
                self.contents = "Menu"
                print("Please Complete Settings before accessing this view.")
                return
            self.panel_contents = SelectedSourceDashboard(self.src, self._close_button)
        else:
            self.plot_dict = extension_plots.get_plot_dict()
            self.panel_contents = self.plot_dict[self.contents].plot(
                self._submit_button
            )(config.main_df, self.src)

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

        print("self.row dashboard:", self.row)

        return self.row
