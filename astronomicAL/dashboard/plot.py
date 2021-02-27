from holoviews.operation.datashader import (
    datashade,
    dynspread,
)

import datashader as ds
import holoviews as hv

import config
import numpy as np
import pandas as pd
import panel as pn
import param

import dashboard.extension_plots as plots


class PlotDashboard(param.Parameterized):
    """A Dashboard used for rendering dynamic plots of the data.

    Parameters
    ----------
    src : ColumnDataSource
        The shared data source which holds the current selected source.

    Attributes
    ----------
    X_variable : param.Selector
        A Dropdown list of columns the user can use for the x-axis of the plot.
    Y_variable : DataFrame
        A Dropdown list of columns the user can use for the x-axis of the plot.
    row : Panel Row
        The panel is housed in a row which can then be rendered by the
        parent Dashboard.
    df : DataFrame
        The shared dataframe which holds all the data.

    """

    X_variable = param.Selector(
        objects=["0"],
        default="0",
        doc="Selection box for the X axis of the plot.")

    Y_variable = param.Selector(
        objects=["1"],
        default="1",
        doc="Selection box for the Y axis of the plot.")

    def __init__(self, src, **params):
        super(PlotDashboard, self).__init__(**params)

        self.row = pn.Row(pn.pane.Str("loading"))
        self.src = src
        self.src.on_change("data", self._panel_cb)
        self.df = config.main_df
        self.update_variable_lists()

    def _update_variable_lists_cb(self, attr, old, new):
        self.update_variable_lists()

    def update_variable_lists(self):
        """Update the list of options used inside `X_variable` and `Y_variable`.

        This method retrieves an up-to-date list of columns inside `df` and
        assigns them to both Selector objects.

        Returns
        -------
        None

        """
        print(f"x_var currently is: {self.X_variable}")

        cols = list(self.df.columns)

        if config.settings["id_col"] in cols:
            cols.remove(config.settings["id_col"])
        if config.settings["label_col"] in cols:
            cols.remove(config.settings["label_col"])

        self.param.X_variable.objects = cols
        self.param.Y_variable.objects = cols
        self.param.X_variable.default = config.settings["default_vars"][1]
        self.param.Y_variable.default = config.settings["default_vars"][0]
        self.Y_variable = config.settings["default_vars"][1]
        self.X_variable = config.settings["default_vars"][0]

        print(f"x_var has now changed to: {self.X_variable}")

    def _panel_cb(self, attr, old, new):
        self.panel()

    @param.depends("X_variable", "Y_variable")
    def plot(self):
        """Create a basic scatter plot of the data with the selected axis.

        The data is represented as a Holoviews Datashader object allowing for
        large numbers of points to be rendered at once. Plotted using a Bokeh
        renderer, the user has full manuverabilty of the data in the plot.

        Returns
        -------
        plot : Holoviews Object
            A Holoviews plot

        """

        p = hv.Points(
            self.df,
            [self.X_variable, self.Y_variable],
        ).opts(active_tools=["pan", "wheel_zoom"])

        cols = list(self.df.columns)

        if len(self.src.data[cols[0]]) == 1:
            selected = pd.DataFrame(
                self.src.data, columns=cols, index=[0])
        else:
            selected = pd.DataFrame(columns=cols)

        selected_plot = hv.Scatter(
            selected,
            self.X_variable,
            self.Y_variable,
        ).opts(
            fill_color="black",
            marker="circle",
            size=10,
            tools=["box_select"],
            active_tools=["pan", "wheel_zoom"],
        )

        color_key = config.settings["label_colours"]

        color_points = hv.NdOverlay(
            {
                config.settings['labels_to_strings'][f"{n}"]:
                hv.Points([0, 0],
                          label=config.settings['labels_to_strings'][f"{n}"]).opts(
                    style=dict(color=color_key[n], size=0)
                )
                for n in color_key
            }
        )

        max_x = np.max(self.df[self.X_variable])
        min_x = np.min(self.df[self.X_variable])

        max_y = np.max(self.df[self.Y_variable])
        min_y = np.min(self.df[self.Y_variable])

        x_sd = np.std(self.df[self.X_variable])
        x_mu = np.mean(self.df[self.X_variable])
        y_sd = np.std(self.df[self.Y_variable])
        y_mu = np.mean(self.df[self.Y_variable])

        max_x = np.min([x_mu + 4*x_sd, max_x])
        min_x = np.max([x_mu - 4*x_sd, min_x])

        max_y = np.min([y_mu + 4*y_sd, max_y])
        min_y = np.max([y_mu - 4*y_sd, min_y])

        if selected.shape[0] > 0:

            max_x = np.max([max_x, np.max(selected[self.X_variable])])
            min_x = np.min([min_x, np.min(selected[self.X_variable])])

            max_y = np.max([max_y, np.max(selected[self.Y_variable])])
            min_y = np.min([min_y, np.min(selected[self.Y_variable])])

        plot = (
            dynspread(
                datashade(
                    p,
                    color_key=color_key,
                    aggregator=ds.by(config.settings["label_col"], ds.count()),
                ).opts(xlim=(min_x, max_x), ylim=(min_y, max_y), responsive=True),
                threshold=0.75,
                how="saturate",
            )
            * selected_plot
            * color_points
        )
        return plot

    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which can then be rendered by the
            parent Dashboard.

        """

        print("Panel Plot")

        self.row[0] = pn.Card(
            pn.Row(self.plot, sizing_mode="stretch_both"),
            header=pn.Row(
                pn.Spacer(width=25, sizing_mode="fixed"),
                self.param.X_variable,
                self.param.Y_variable,
                width=200,
                sizing_mode="fixed",
            ),
            collapsible=False,
            sizing_mode="stretch_both",
        )

        return self.row
