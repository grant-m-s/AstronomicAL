from bokeh.models import ColumnDataSource
from dashboard.active_learning import ActiveLearningDashboard
from dashboard.menu import MenuDashboard
from dashboard.plot import PlotDashboard
from dashboard.selected_source import SelectedSourceDashboard
from dashboard.settings_dashboard import SettingsDashboard

import config
import panel as pn
import param


class DataDashboard(param.Parameterized):

    src = ColumnDataSource(data={"0": [], "1": []})

    contents = param.String()

    def __init__(self, src, contents="Menu", main_plot=None, **params):
        super(DataDashboard, self).__init__(**params)

        self.src = src
        self.row = pn.Row(pn.pane.Str("loading"))
        self.df = config.main_df

        self.contents = contents

    def update_panel_contents_src(self, event):
        self.panel_contents.src.data = self.src.data

    @param.depends("contents", watch=True)
    def update_contents(self):

        print("Updating contents")

        if self.contents == "Settings":

            self.panel_contents = SettingsDashboard(
                self, self.src, self.df)

        elif self.contents == "Menu":

            self.panel_contents = MenuDashboard(self)

        elif self.contents == "Plot":

            self.panel_contents = PlotDashboard(self.src)

        elif self.contents == "Active Learning":

            self.df = config.main_df
            self.panel_contents = ActiveLearningDashboard(self.src, self.df)

        elif self.contents == "Selected Source":

            self.panel_contents = SelectedSourceDashboard(self.src)

        self.panel()

    def set_contents(self, updated):
        self.contents = updated

    def panel(self):

        self.row[0] = self.panel_contents.panel()

        return self.row
