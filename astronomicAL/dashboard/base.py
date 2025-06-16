from holoviews.operation.datashader import (
    datashade,
    dynspread,
)

import datashader as ds
import holoviews as hv

import astronomicAL.config as config
import numpy as np
import pandas as pd
import panel as pn
import param


class BasePlotClass(param.parametrized):

    def  __init__(self,  src, close_button):
        super.__init__()
        self.src = src
        self.df = config.main_df
        self.close_button = close_button
        self.figure = pn.pane.HoloViews()
        self.counter = 0

    def update_df(self):
        self.df = config.main_df

    def get_variable_list(self, excluded_columns = ["id_col", "ra_dec", "label_col"]):
        """Returns the list of options used inside `X_variable` or `Y_variable`.
        This method retrieves an up-to-date list of columns inside `df` to be assigned 
        to param.X_variable or param.Y_variable
        
        Returns
        -------
        List of columns name 

        """
        self.update_df()
        cols = list(self.df.columns)

        for exluded_col in excluded_columns:
            if config.settings[exluded_col] in cols:
                cols.remove(config.settings[exluded_col])
            elif exluded_col in cols:
                cols.remove(exluded_col)
        
        return cols
    
    def _update_plot()


class ScatterDashboard(BasePlotClass):
    """A Dashboard used for rendering dynamic scatter plots of the data.
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
    df : DataFrame
        The shared dataframe which holds all the data.

    """

    X_variable = param.Selector(
        objects=["0"], default="0", doc="Selection box for the X axis of the plot."
    )

    Y_variable = param.Selector(
        objects=["1"], default="1", doc="Selection box for the Y axis of the plot."
    )

    def __init__(self, src, close_button):
        super().__init__(self, src)
        self.src.on_change("data", self._change_src_cb)
        self.counter = 0
        self.update_variable_lists(excluded_columns = ["id_col", "label_col", "ra_dec"])

    def update_variable_lists(self, excluded_columns = ["id_col", "label_col", "ra_dec"] ):
        self.param.X_variable.objects = self.get_variable_list(excluded_columns=excluded_columns)
        self.param.Y_variable.objects = self.get_variable_list(excluded_columns=excluded_columns)
        self.param.X_variable.default = config.settings["default_vars"][0]
        self.param.Y_variable.default = config.settings["default_vars"][1]
        self.X_variable = config.settings["default_vars"][0]
        self.Y_variable = config.settings["default_vars"][1]

    def _panel_cb(self, attr, old, new):
        selected_src_plot = self.plot_selected(self.X_variable, self.Y_variable)
        if selected_src_plot is not None:
            overlay = hv.Overlay(self.main_plot + selected_src_plot).collate()
            self.figure.object = overlay

    @staticmethod
    def get_axis_limits(x_var, Nsigma = 3):
        
        x = x_var[np.isfinite(x_var)]
        max_x = np.max(x)
        min_x = np.min(x)
        x_sd = np.std(x)
        x_mu = np.mean(x)
        max_x = np.min([x_mu + Nsigma * x_sd, max_x])
        min_x = np.max([x_mu - Nsigma * x_sd, min_x])
        return min_x, max_x
    

    @param.depends("X_variable", "Y_variable")
    def _update_plot(self):
        self.main_plot = self.plot()
        selected_src_plot = self.plot_selected(self.X_variable, self.Y_variable)
        if selected_src_plot is not None:
            overlay = hv.Overlay(self.main_plot + selected_src_plot).collate()
            self.figure.object = overlay
        else:
            self.figure.object = self.main_plot
    
    def plot(self, x_var = None, y_var = None):
        """Create a basic scatter plot of the data with the selected axis.

        The data is represented as a Holoviews Datashader object allowing for
        large numbers of points to be rendered at once. Plotted using a Bokeh
        renderer, the user has full manuverabilty of the data in the plot.

        Returns
        -------
        plot : Holoviews Object
            A Holoviews plot

        """
        if x_var is None:
            x_var = self.X_variable

        if y_var is None:
            y_var = self.Y_variable

        if (self.df[[x_var, y_var]].dtypes == "object").any():
            print("One of the selected columns is not of float or int type")
            return
        
        color_key = config.settings["label_colours"]

        p = hv.Points(
            self.df,
            [x_var, y_var], 
        ).opts()
        
        min_x, max_x = self.get_axis_limits(self.df[x_var])
        min_y, max_y = self.get_axis_limits(self.df[y_var])

        plot = (
            dynspread(
                datashade(
                    p,
                    color_key=color_key,
                    aggregator=ds.by(config.settings["label_col"], ds.count()),
                ).opts(
                    xlim=(min_x, max_x),
                    ylim=(min_y, max_y),
                    active_tools = [], 
                ),
                threshold=0.75,
                how="saturate",
                legend_position="bottom_right"
            )
        )
        self.counter +=1
        print(f"Called Scatter plot function overall {self.counter} times")
        return plot
    
    def plot_selected(self, x_var, y_var):
        cols = list(self.df.columns)
        if len(self.src.data[cols[0]]) == 1:
            selected = pd.DataFrame(self.src.data, columns=cols, index=[0])
        else:
            return None
        if selected.shape[0] > 0:
            selected_plot = hv.Scatter(selected, x_var, y_var,).opts(
                fill_color="black",
                marker="circle",
                size=10,
                active_tools=[])
            return selected_plot
        

    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which can then be rendered by the
            parent Dashboard.

        """
        self._update_plot()
        return pn.Card(
            pn.Row(self.figure, sizing_mode="stretch_both"),
            header=pn.Row(
                    pn.Spacer(width=25,),
                self.close_button,
                pn.Row(self.param.X_variable, max_width=100),
                pn.Row(self.param.Y_variable, max_width=100),
                max_width=400,

            ),
            collapsible=False,
            sizing_mode="stretch_both",
        )




    



