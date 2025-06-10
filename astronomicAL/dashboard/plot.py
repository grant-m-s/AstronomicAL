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

        return  pn.Card(
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
    
#######################################

class HistoDashboard_not_parametrized(param.Parameterized):
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
        super(HistoDashboard_not_parametrized, self).__init__()

        self.row = pn.Row(pn.pane.Str("loading"))
        self.src = src
        self._initialize_widgets()
        self.src.on_change("data", self._panel_cb)
        self.df = config.main_df
        self.close_button = close_button
        self.update_variable_lists()
        self.counter = 0
        self.panel()

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
        self.Nbins_slider = pn.widgets.IntSlider(name='N bins', start=2, end=200, step=1, value=10, value_throttled = 10)
        self.label_selector = pn.widgets.MultiChoice(name='Label to Plot', value=['All'],
                                                       options = ["All"] + list(config.settings["strings_to_labels"].keys()),
                                                       )
        
        self.range_min = pn.widgets.FloatInput(value = None, start=-np.inf, end=np.inf, step = 1, 
                              placeholder = "None", name = "range min",)
        self.range_max = pn.widgets.FloatInput(value = None, start=-np.inf, end=np.inf, step = 1, 
                              placeholder = "None", name = "range max")
        

        self.settings_panel = pn.Column(self.log_xscale, 
                                    self.log_yscale, 
                                    self.density, 
                                    self.cumulative, 
                                    self.Nbins_slider,
                                    pn.Row(self.range_min, self.range_max),
                                    self.label_selector,
                                    visible=False, 
                                    styles={'border': '1px solid lightgray', 'padding': '10px',
                                            },)
        
        self.settings_button = pn.widgets.Button(name="Settings ▾", button_type="primary")

        self.settings_button.on_click(self._toggle_dropdown)
        

        self.strings_to_plot = ["All"]
        self.log_xscale.param.watch(self._update_plot,  "value")
        self.log_yscale.param.watch(self._update_plot,  "value")
        self.cumulative.param.watch(self._update_plot,  "value")
        self.Nbins_slider.param.watch(self._update_plot, "value_throttled")
        self.density.param.watch(self._update_plot,  "value")
        self.range_min.param.watch(self._update_plot, "value")
        self.range_max.param.watch(self._update_plot, "value")
        self.label_selector.param.watch(self._update_label, "value")
        
    
    def _toggle_dropdown(self, event):
        self.settings_panel.visible = not self.settings_panel.visible
        self.settings_button.name = "Settings ▴" if self.settings_panel.visible else "Settings ▾"


    @staticmethod
    def get_histogram_hv(x_var, Nbins = 10, log_x = False, log_y = False, density = False, cumulative = False, 
                      range = (None, None), label = "",
                      **kwargs):
        
        x = x_var[np.isfinite(x_var)]
        xmin = np.min(x) if range[0] is None else range[0]
        xmax = np.max(x) if range[1] is None else range[1]
        ylim = (0.2,None) if log_y else (0,None)   #holoviews doesn't like no ylim passed with log yscale
        weights = np.ones_like(x)/len(x) if density else None 
            
        if log_x:
            if xmin > 0: # both positive
                bins  = np.geomspace(xmin , xmax, Nbins) if xmin != xmax else Nbins
            else:
                print("Negative values for log x scale not yet supported, removing values <=0")
                xmin = np.min(x[x>0])
                bins  = np.geomspace(xmin , xmax, Nbins) if xmin != xmax else Nbins
            #TODO implement case where all values are negative
        else:
            bins = np.linspace(xmin , xmax, Nbins) if xmin != xmax else Nbins
            
        stats, edges = np.histogram(x, bins = bins, weights = weights)
        if cumulative:
            stats = np.cumsum(stats)
        
        histogram = hv.Histogram((stats, edges), label = label).opts(logy = log_y,
                                                              logx = log_x,
                                                              ylim = ylim,
                                                              **kwargs)
        return histogram

    
    @param.depends("X_variable")
    def plot_hv(self, x_var=None):
        
        """Create a basic histogram plot of the data with the selected axis.
        Returns
        -------
        plot : holoviews plot

        """
        if x_var is None:
            x_var_name = self.X_variable
            x_var = self.df[self.X_variable].to_numpy()
        
        strings_to_plot = self.strings_to_plot
        
        if bool(strings_to_plot) and ("All" not in strings_to_plot or len(strings_to_plot)>1):
           labels = self.df[config.settings["label_col"]]
           labels_to_plot = [config.settings["strings_to_labels"][i] for i in strings_to_plot if i != "All"]
        
        else:
            labels_to_plot = []
        
        self.overlays = []
        
        if "All" in strings_to_plot:
            h = self.get_histogram_hv(x_var, Nbins = self.Nbins_slider.value, 
                            log_x = self.log_xscale.value, log_y = self.log_yscale.value,
                            cumulative = self.cumulative.value, density = self.density.value,
                            range = (self.range_min.value, self.range_max.value),
                            label = "All",
                            **{"fill_color" : "blue", "line_color" : "blue"})
            self.overlays.append(h)
            
        for i, label_to_plot in enumerate(labels_to_plot):
            h = self.get_histogram_hv(x_var[labels == label_to_plot], Nbins = self.Nbins_slider.value, 
                            log_x = self.log_xscale.value, log_y = self.log_yscale.value,
                            cumulative = self.cumulative.value, density=self.density.value,
                            range = (self.range_min.value, self.range_max.value),
                            xlabel = self.X_variable,
                            label = config.settings["labels_to_strings"][str(label_to_plot)],
                            **{"fill_color" : config.settings["label_colours"][label_to_plot] if i < 2 else "none",
                               "line_color" : config.settings["label_colours"][label_to_plot],
                               "line_width" : 1.5,
                               "fill_alpha" : 0.7,
                            }
                            )
            self.overlays.append(h)
            
                              
        cols = list(self.df.columns)

        if len(self.src.data[cols[0]]) == 1:
            selected = pd.DataFrame(self.src.data, columns=cols, index=[0])
            self.overlays.append(hv.VLine(selected[x_var_name].iloc[0]).opts(
                                  color="black",
                                  line_dash = "dashed",
                                  line_width = 1,
                                  )
            )
    
        xlabel=x_var_name
        ylabel= "% of Sources" if self.density.value else "# Sources" 
        plot = hv.Overlay(self.overlays).opts(active_tools = [],
                                              xlabel = xlabel,
                                              ylabel = ylabel,
                                              )
        
        self.counter +=1 
        print(f"called plot function  {self.counter} times")
        
        return plot
        
    def _update_label(self, event):
        self.strings_to_plot = event.new
        self.panel()
        
    def _update_plot(self, event):
        self.panel()

    def panel(self):
        return  pn.Card(pn.Column(
                            pn.Row(self.plot_hv, sizing_mode="scale_both"),
                            self.settings_panel, scroll = True),
                        header = pn.Row(
                                    pn.Spacer(width=25),
                                    self.close_button,
                                    pn.Row(self.param.X_variable, max_width=100),
                                    self.settings_button,
                                ),
                                collapsible=False,
                                sizing_mode="stretch_both",
                        )


class HistoDashboard(param.Parameterized):
    
    X_variable = param.Selector(objects=["0"], default="0", doc="X axis variable")
    log_xscale = param.Boolean(default=False, doc = None)
    log_yscale = param.Boolean(default=False, doc = None)
    density = param.Boolean(default=False, doc = None )
    cumulative = param.Boolean(default=False, doc = None)
    Nbins = param.Integer(default=10, bounds=(2, 200), doc = "Number of bins")
    range_min = param.Number(default= -np.inf, bounds=(-np.inf, np.inf), doc= "Range min")
    range_max = param.Number(default= np.inf, bounds=(-np.inf, np.inf), doc= "Range max")
    label_selector = param.ListSelector(default=["All"], objects=["All"], doc="Labels to plot")
   
    def __init__(self, src, close_button):
        super(HistoDashboard, self).__init__()

        self.row = pn.Row(pn.pane.Str("loading"))
        self.src = src
        self.close_button = close_button
        self.src.on_change("data", self._panel_cb)
        self.df = config.main_df
        self.update_variable_lists()
        self.counter = 0
        self.settings_button = pn.widgets.Button(name="Settings ▾", button_type="primary")
        self.settings_panel = pn.Column(
            pn.Param(
                self,
                parameters=[
                    "log_xscale", "log_yscale", "density", "cumulative",
                    "Nbins", "range_min", "range_max", "label_selector"
                ],
                widgets={
                    "label_selector": {"width": 200, "height": 80, "size": 10},
                    "range_min": {"type": pn.widgets.FloatInput, "placeholder": "None"},
                    "range_max": {"type": pn.widgets.FloatInput, "placeholder": "None"},
                    "Nbins": {"throttled": True},
                    "label_selector": {"type": pn.widgets.MultiChoice}
                },
                show_name=False,
                sizing_mode="stretch_width"
            ),
            visible=False,
            margin=(10, 0, 0, 0)
        )
        self.settings_button.on_click(self._toggle_settings_panel)
        self.panel()
    
    def _toggle_settings_panel(self, event):
        self.settings_panel.visible = not self.settings_panel.visible
        self.settings_button.name = "Settings ▴" if self.settings_panel.visible else "Settings ▾"

    def _update_variable_lists_cb(self, attr, old, new):
        self.update_variable_lists()

    def update_df(self):
        self.df = config.main_df

    def update_variable_lists(self):
        """Update the list of options used inside `X_variable and label selector`.
        This method retrieves an up-to-date list of columns inside `df` and
        assigns them to X-axis Selector object and label selector.

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
        self.param.label_selector.objects = ["All"] + list(config.settings["strings_to_labels"].keys())


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

    @staticmethod
    def get_histogram_hv(x_var, Nbins = 10, log_x = False, log_y = False, density = False, cumulative = False, 
                      range = (-np.inf, np.inf), label = "",
                      **kwargs):
        
        x = x_var[np.isfinite(x_var)]
        xmin = max(np.min(x),range[0])
        xmax = min(np.max(x),range[1])
        
        #if range[1] < xmin or range[0] > xmax i get an error due to bins not increasing
        if xmin > xmax:
            print("Warning, Range max < than minimum value spanned by the data"
                  "or Range min > than maximum value spanned by the data")
            xmin = xmax

        ylim = (0.2,None) if log_y else (0,None)   #holoviews doesn't like no ylim passed with log yscale
        weights = np.ones_like(x)/len(x) if density else None 
            
        if log_x:
            if xmin > 0: # both positive
                bins  = np.geomspace(xmin , xmax, Nbins) if xmin != xmax else Nbins
            else:
                print("Negative values for log x scale not yet supported, removing values <=0")
                xmin = np.min(x[x>0])
                bins  = np.geomspace(xmin , xmax, Nbins) if xmin != xmax else Nbins
            #TODO implement case where all values are negative
        else:
            bins = np.linspace(xmin , xmax, Nbins) if xmin != xmax else Nbins
            
        stats, edges = np.histogram(x, bins = bins, weights = weights)
        if cumulative:
            stats = np.cumsum(stats)
        
        histogram = hv.Histogram((stats, edges), label = label).opts(logy = log_y,
                                                              logx = log_x,
                                                              ylim = ylim,
                                                              xlim = (xmin, xmax),
                                                              active_tools = [],
                                                              **kwargs)
        
        return histogram, xmin, xmax

    
    @param.depends(
        "X_variable", "log_xscale", "log_yscale", "density", "cumulative",
        "Nbins", "range_min", "range_max", "label_selector"
    )
    def plot_hv(self, x_var=None):
        
        """Create a basic histogram plot of the data with the selected axis.
        Returns
        -------
        plot : holoviews plot

        """
        if x_var is None:
            x_var_name = self.X_variable
            x_var = self.df[self.X_variable].to_numpy()
        
        strings_to_plot = self.label_selector
        
        if bool(strings_to_plot) and ("All" not in strings_to_plot or len(strings_to_plot)>1):
           labels = self.df[config.settings["label_col"]]
           labels_to_plot = [config.settings["strings_to_labels"][i] for i in strings_to_plot if i != "All"]
        
        else:
            labels_to_plot = []
        
        self.overlays = []
        xmin, xmax = -np.inf, np.inf 
        if "All" in strings_to_plot:
            h, xmin_temp, xmax_temp = self.get_histogram_hv(x_var, Nbins = self.Nbins, 
                            log_x = self.log_xscale, log_y = self.log_yscale,
                            cumulative = self.cumulative, density = self.density,
                            range = (self.range_min, self.range_max),
                            label = "All",
                            **{"fill_color" : "blue", "line_color" : "blue"})
            self.overlays.append(h)
            xmin = max(xmin, xmin_temp)
            xmax = min(xmax, xmax_temp)
            
        for i, label_to_plot in enumerate(labels_to_plot):
            h, xmin_temp, xmax_temp = self.get_histogram_hv(x_var[labels == label_to_plot], Nbins = self.Nbins, 
                            log_x = self.log_xscale, log_y = self.log_yscale,
                            cumulative = self.cumulative, density=self.density,
                            range = (self.range_min, self.range_max),
                            label = config.settings["labels_to_strings"][str(label_to_plot)],
                            **{"fill_color" : config.settings["label_colours"][label_to_plot] if i < 2 else "none",
                               "line_color" : config.settings["label_colours"][label_to_plot],
                               "line_width" : 1.5,
                               "fill_alpha" : 0.7,
                            }
                            )
            self.overlays.append(h)
            xmin = max(xmin, xmin_temp)
            xmax = min(xmax, xmax_temp)
            
                              
        cols = list(self.df.columns)

        if len(self.src.data[cols[0]]) == 1:
            selected = pd.DataFrame(self.src.data, columns=cols, index=[0])
            self.overlays.append(hv.VLine(selected[x_var_name].iloc[0]).opts(
                                  color="black",
                                  line_dash = "dashed",
                                  line_width = 1,
                                  active_tools = [],
                                  )
            )
    
        xlabel=x_var_name
        ylabel= "% of Sources" if self.density else "# Sources" 
        plot = hv.Overlay(self.overlays).opts(active_tools = [],
                                              xlabel = xlabel,
                                              ylabel = ylabel,
                                              xlim = (xmin, xmax),
                                              )
        self.counter +=1 
        print(f"called plot function  {self.counter} times")
        return plot

    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which can then be rendered by the
            parent Dashboard.
        """
        
        return pn.Card(
                        pn.Column(
                               pn.Row(self.plot_hv, sizing_mode="scale_both"),
                               self.settings_panel, scroll = True),
                        header= pn.Row(
                                   pn.Spacer(width=25),
                                   self.close_button,
                                   pn.Row(self.param.X_variable, max_width=100),
                                   self.settings_button,
                                   ),
                                collapsible=False,
                                sizing_mode="stretch_both",
                        )

        

