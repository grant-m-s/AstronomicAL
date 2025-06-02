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
from matplotlib.figure import Figure


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
        objects=["0"], default="0", doc="Selection box for the X axis of the plot."
    )

    Y_variable = param.Selector(
        objects=["1"], default="1", doc="Selection box for the Y axis of the plot."
    )

    def __init__(self, src, close_button):
        super(PlotDashboard, self).__init__()

        self.row = pn.Row(pn.pane.Str("loading"))
        self.src = src
        self.src.on_change("data", self._panel_cb)
        self.df = config.main_df
        self.close_button = close_button
        self.update_variable_lists()

    def _update_variable_lists_cb(self, attr, old, new):
        self.update_variable_lists()

    def update_df(self):
        self.df = config.main_df

    def update_variable_lists(self):
        """Update the list of options used inside `X_variable` and `Y_variable`.

        This method retrieves an up-to-date list of columns inside `df` and
        assigns them to both Selector objects.

        Returns
        -------
        None

        """

        self.update_df()

        cols = list(self.df.columns)

        if config.settings["id_col"] in cols:
            cols.remove(config.settings["id_col"])
        if config.settings["label_col"] in cols:
            cols.remove(config.settings["label_col"])

        self.param.X_variable.objects = cols
        self.param.Y_variable.objects = cols
        self.param.X_variable.default = config.settings["default_vars"][0]
        self.param.Y_variable.default = config.settings["default_vars"][1]
        self.X_variable = config.settings["default_vars"][0]
        self.Y_variable = config.settings["default_vars"][1]

    def _panel_cb(self, attr, old, new):
        cols = list(self.df.columns)

        if config.settings["id_col"] in cols:
            cols.remove(config.settings["id_col"])
        if config.settings["label_col"] in cols:
            cols.remove(config.settings["label_col"])

        for i in config.dashboards.keys():
            if config.dashboards[i].contents == "Basic Plot":
                curr_x = config.dashboards[i].panel_contents.X_variable
                curr_y = config.dashboards[i].panel_contents.Y_variable
                if (curr_x == self.X_variable) and (curr_y == self.Y_variable):
                    try:
                        config.dashboards[i].panel_contents.X_variable = curr_x
                        config.dashboards[i].panel_contents.Y_variable = curr_y
                        config.dashboards[i].panel_contents.panel()
                    except:
                        config.dashboards[i].set_contents = "Menu"

                    break

        self.panel()


    @param.depends("X_variable", "Y_variable")
    def plot(self, x_var=None, y_var=None):
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

        p = hv.Points(
            self.df,
            [x_var, y_var], 
        ).opts()

        cols = list(self.df.columns)

        if len(self.src.data[cols[0]]) == 1:
            selected = pd.DataFrame(self.src.data, columns=cols, index=[0])
        else:
            selected = pd.DataFrame(columns=cols)

        selected_plot = hv.Scatter(selected, x_var, y_var,).opts(
            fill_color="black",
            marker="circle",
            size=10,
            #active_tools=["pan", "wheel_zoom"],
        )

        color_key = config.settings["label_colours"]

        # color_points = hv.NdOverlay(
        #     {
        #         config.settings["labels_to_strings"][f"{n}"]: hv.Points(
        #             [0, 0], label=config.settings["labels_to_strings"][f"{n}"]
        #         ).opts(style=dict(color=color_key[n], size=0))
        #         for n in color_key
        #     }
        # )

        max_x = np.max(self.df[x_var])
        min_x = np.min(self.df[x_var])

        max_y = np.max(self.df[y_var])
        min_y = np.min(self.df[y_var])

        x_sd = np.std(self.df[x_var])
        x_mu = np.mean(self.df[x_var])
        y_sd = np.std(self.df[y_var])
        y_mu = np.mean(self.df[y_var])

        max_x = np.min([x_mu + 4 * x_sd, max_x])
        min_x = np.max([x_mu - 4 * x_sd, min_x])

        max_y = np.min([y_mu + 4 * y_sd, max_y])
        min_y = np.max([y_mu - 4 * y_sd, min_y])

        if selected.shape[0] > 0:

            max_x = np.max([max_x, np.max(selected[x_var])])
            min_x = np.min([min_x, np.min(selected[x_var])])

            max_y = np.max([max_y, np.max(selected[y_var])])
            min_y = np.min([min_y, np.min(selected[y_var])])

        plot = (
            dynspread(
                datashade(
                    p,
                    color_key=color_key,
                    aggregator=ds.by(config.settings["label_col"], ds.count()),
                ).opts(
                    xlim=(min_x, max_x),
                    ylim=(min_y, max_y),
                    #responsive=True,
                    #shared_axes=False,
                    framewise=False,          
                    axiswise=False, 
                    default_tools = [],     
                    tools = [],
                ),
                threshold=0.75,
                how="saturate",
            )
            * selected_plot
            # * color_points
        ).opts(legend_position="bottom_right", 
               #shared_axes=False
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

        self.row[0] = pn.Card(
            pn.Row(self.plot, sizing_mode="stretch_both"),
            header=pn.Row(
                pn.Spacer(width=25,
                        #    sizing_mode="fixed"
                           ),
                self.close_button,
                pn.Row(self.param.X_variable, max_width=100),
                pn.Row(self.param.Y_variable, max_width=100),
                max_width=400,
                # sizing_mode="fixed",
            ),
            collapsible=False,
            sizing_mode="stretch_both",
        )

        return self.row
    
#######################################

class HistoDashboard(param.Parameterized):
    """A Dashboard used for rendering histograms of the data.

    Parameters
    ----------
    src : ColumnDataSource
        The shared data source which holds the current selected source.

    Attributes
    ----------
    X_variable : param.Selector
        A Dropdown list of columns the user can use for the x-axis of the plot.
    row : Panel Row
        The panel is housed in a row which can then be rendered by the
        parent Dashboard.
    df : DataFrame
        The shared dataframe which holds all the data.

    """

    X_variable = param.Selector(
        objects=["0"], default="0", doc= "Selection box for the X axis of the plot.")


    def __init__(self, src, close_button):
        super(HistoDashboard, self).__init__()

        self.row = pn.Row(pn.pane.Str("loading"))
        self.src = src
        self._initialize_widgets()
        self.src.on_change("data", self._panel_cb)
        self.df = config.main_df
        self.close_button = close_button
        self.update_variable_lists()

    def _update_variable_lists_cb(self, attr, old, new):
        self.update_variable_lists()

    def update_df(self):
        self.df = config.main_df

    def update_variable_lists(self):
        """Update the list of options used inside `X_variable`.

        This method retrieves an up-to-date list of columns inside `df` and
        assigns them to both Selector objects.

        Returns
        -------
        None

        """

        self.update_df()

        cols = list(self.df.columns)

        if config.settings["id_col"] in cols:
            cols.remove(config.settings["id_col"])

        self.param.X_variable.objects = cols
        self.param.X_variable.default = config.settings["default_vars"][0]
        self.X_variable = config.settings["default_vars"][0]


    def _panel_cb(self, attr, old, new):
        cols = list(self.df.columns)

        if config.settings["id_col"] in cols:
            cols.remove(config.settings["id_col"])

        for i in config.dashboards.keys():
            if config.dashboards[i].contents == "Histogram Plot":   
                curr_x = config.dashboards[i].panel_contents.X_variable
                if curr_x == self.X_variable:
                    try:
                        config.dashboards[i].panel_contents.X_variable = curr_x
                        config.dashboards[i].panel_contents.panel()
                    except:
                        config.dashboards[i].set_contents = "Menu"

                    break

        self.panel()

    def _initialize_widgets(self):
        self.log_xscale = pn.widgets.Checkbox(name = "x Log")
        self.log_yscale = pn.widgets.Checkbox(name = "y Log")
        self.density = pn.widgets.Checkbox(name = "Density")
        self.cumulative = pn.widgets.Checkbox(name = "Cumulative")
        self.Nbins_slider = pn.widgets.IntSlider(name='N bins', start=2, end=1000, step=1, value=10, value_throttled = 10)
        self.label_selector = pn.widgets.MultiChoice(name='Label to Plot', value=['All'],
                                                       options = ["All"] + list(config.settings["strings_to_labels"].keys()),
                                                       height = 300)
        self.strings_to_plot = ["All"]
        self.log_xscale.param.watch(self._update_plot,  "value")
        self.log_yscale.param.watch(self._update_plot,  "value")
        self.cumulative.param.watch(self._update_plot,  "value")
        #self.Nbins_slider.param.watch(self._update_plot, "value")
        self.Nbins_slider.param.watch(self._update_plot, "value_throttled")
        self.density.param.watch(self._update_plot,  "value")
        self.label_selector.param.watch(self._update_label, "value")
    

    @param.depends("X_variable")
    def plot(self, x_var=None):
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
            x_var_name = self.X_variable
            x_var = self.df[self.X_variable].to_numpy()
        
        x_var = x_var[np.isfinite(x_var)]
        if self.log_xscale.value:
            x_var = np.log10(x_var)
            x_var = x_var[np.isfinite(x_var)]

        stats, edges  =  np.histogram(x_var, bins = self.Nbins_slider.value)
        if self.cumulative.value:
            stats = np.cumsum(stats)
        if self.log_yscale.value: ylim = (1,None) 
        else: ylim = (0,None)
        plot = hv.Histogram((edges, stats)).opts(color = "blue", logy = self.log_yscale.value, line_color="blue", ylim = ylim)

        cols = list(self.df.columns)

        if len(self.src.data[cols[0]]) == 1:
            selected = pd.DataFrame(self.src.data, columns=cols, index=[0])
            selected_plot = hv.VLine(selected[x_var_name].iloc[0]).opts(
            color="black",
            line_dash = "dashed",
            line_width = 1,
            ylim = ylim
            )
            plot = (plot* selected_plot)
        
        plot = plot.opts(
        xlabel=x_var_name,
        ylabel="# Sources",
        hooks=[tools])

        return plot
    
    @staticmethod
    def get_histogram(ax, x_var, Nbins = 10, log_x = False, log_y = False, density = False, cumulative = False,
                      **kwargs):
        
        x = x_var[np.isfinite(x_var)]
        if density: 
            weights = np.ones(len(x))/len(x)
            ax.set_ylabel(r"% of sources")
        else:
            weights = None
            ax.set_ylabel("# of sources")
        
        
        if log_x:
            xmin, xmax = np.min(x), np.max(x)

            if xmin > 0: # both positive
                bins  = np.geomspace(xmin , xmax, Nbins) if xmin != xmax else Nbins
                ax.hist(x, bins = bins, log = log_y, 
                        cumulative = cumulative, weights = weights,
                        **kwargs);  
                ax.set_xscale("log")
           
            elif xmax < 0:  #both  negative
                bins = -np.geomspace(-xmax, -xmin, Nbins) if xmin != xmax else Nbins
                ax.hist(x, bins = bins[::-1], log = log_y, 
                        cumulative = cumulative, weights = weights,
                        **kwargs);  
                ax.set_xscale("symlog",  linthresh = np.abs(xmax), linscale = 0.15)
            
            elif xmin * xmax < 0:    #x has positive and negative values
                x_thresh = np.min(np.abs(x[x != 0]))
                bins_positive = np.geomspace(x_thresh, xmax, int(Nbins/2))  #for the moment same number of bins for positive 
                bins_negative = np.geomspace(x_thresh, -xmin, int(Nbins/2))  #and negative values
                bins = np.concatenate([-bins_negative[::-1], bins_positive])
                ax.hist(x, bins = bins, log = log_y, 
                        cumulative = cumulative, weights = weights,
                        **kwargs);  
                ax.set_xscale("symlog", linthresh=x_thresh, linscale = 0.15 )

            elif xmin * xmax == 0:   #at least one of the two is 0
                try:
                    x_thresh = np.min(np.abs(x[x != 0]))
                    bins_positive = np.geomspace(x_thresh, xmax, int(Nbins/2))  if xmax > 0 else np.array([0, x_thresh])
                    bins_negative = np.geomspace(x_thresh, -xmin, int(Nbins/2)) if xmin < 0 else np.array([0, x_thresh])
                    bins = np.concatenate([-bins_negative[::-1], bins_positive])
                    ax.hist(x, bins = bins, log = log_y, 
                       cumulative = cumulative, weights = weights,
                       **kwargs);  
                    ax.set_xscale("symlog", linthresh=x_thresh, linscale = 0.15 )
                except ValueError:  #all values are 0 so x[x != 0] is an empty array
                    ax.hist(x, bins = Nbins, log = log_y, 
                       cumulative = cumulative, weights = weights,
                       **kwargs); 

        else: 
            ax.hist(x, bins = Nbins, log = log_y, 
                cumulative = cumulative, weights = weights,
                **kwargs);  
  
    @param.depends("X_variable")
    def plot_mplt(self, x_var=None, strings_to_plot = ["All"]):
        """Create a basic histogram plot of the data with the selected axis.
        
        Returns
        -------
        plot : Matplotlib plot

        """
        if x_var is None:
            x_var_name = self.X_variable
            x_var = self.df[self.X_variable].to_numpy()
        
        if bool(strings_to_plot) and ("All" not in strings_to_plot or len(strings_to_plot)>1):
           #not sure we need this if statement
           labels = self.df[config.settings["label_col"]]
           labels_to_plot = [config.settings["strings_to_labels"][i] for i in strings_to_plot if i != "All"]
        
        else:
            labels_to_plot = []
        
        fig = Figure()
        ax = fig.subplots()
        
        if "All" in strings_to_plot:
            self.get_histogram(ax, x_var, Nbins = self.Nbins_slider.value, 
                           log_x = self.log_xscale.value, log_y = self.log_yscale.value,
                           cumulative = self.cumulative.value, density=self.density.value,
                           **{"color" : "blue", "edgecolor" : "blue", "label" : "All"})
            
        for i, label_to_plot in enumerate(labels_to_plot):
            self.get_histogram(ax, x_var[labels == label_to_plot], Nbins = self.Nbins_slider.value, 
                            log_x = self.log_xscale.value, log_y = self.log_yscale.value,
                            cumulative = self.cumulative.value, density=self.density.value,
                            **{"color" : config.settings["label_colours"][label_to_plot],
                               "edgecolor" : config.settings["label_colours"][label_to_plot],
                               "lw" : 2, "alpha" : 0.7, 
                               "label" : config.settings["labels_to_strings"][str(label_to_plot)],
                               "histtype" : "stepfilled" if i < 2 else "step"} ### maybe avoids confusion
                            )
                              
            
                           
        cols = list(self.df.columns)

        if len(self.src.data[cols[0]]) == 1:
            selected = pd.DataFrame(self.src.data, columns=cols, index=[0])
            ax.axvline(selected[x_var_name].iloc[0], c = "k", ls = ':', lw =1)
        
        ax.legend()
    
        return fig
    
    def _update_label(self, event):
        self.strings_to_plot = event.new
        self.panel()
        

    def _update_plot(self, event):
        self.panel()

    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which can then be rendered by the
            parent Dashboard.

        """

        self.row[0] = pn.Card(pn.Column(pn.WidgetBox(pn.Row(self.log_xscale, self.log_yscale, self.density, self.cumulative),
                           pn.Row(self.Nbins_slider, self.label_selector), width = 500, height = 150,),
                           pn.Row(self.plot_mplt(strings_to_plot=self.strings_to_plot)), 
                           sizing_mode="scale_both"),
            header=pn.Row(
                pn.Spacer(width=25,
                        #    sizing_mode="fixed"
                           ),
                self.close_button,
                pn.Row(self.param.X_variable, max_width=100),
                max_width=400,
                # sizing_mode="fixed",
            ),
            collapsible=False,
            sizing_mode="stretch_both",
        )

        return self.row


def tools(plot, element):
    plot.handles['plot'].toolbar.active_drag = None
    plot.handles['plot'].toolbar.active_scroll = None
