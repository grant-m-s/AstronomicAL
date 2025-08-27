from holoviews.operation.datashader import (
    datashade,
    dynspread,
)

import datashader as ds
import holoviews as hv
from holoviews import streams

import uuid
import numpy as np
import pandas as pd
import panel as pn
import param

import astronomicAL.config as config
from astronomicAL.extensions.shared_data import shared_data


class BasePlotClass(param.Parameterized):

    def  __init__(self,  src, close_button):
        super().__init__()
        self.panel_id = str(uuid.uuid4()) 
        self.src = src
        self.df = config.main_df
        self.close_button = close_button
        self.figure = pn.pane.HoloViews(sizing_mode="stretch_both")
        self.settings_button = pn.widgets.Button(name="Settings ▾", button_type="primary",  max_height = 40, max_width=100)
        self.settings_button.on_click(self._toggle_settings_panel)
    
    def update_df(self):
        self.df = config.main_df

    def _toggle_settings_panel(self, event):
        self.settings_panel.visible = not self.settings_panel.visible
        self.settings_button.name = "Close Settings" if self.settings_panel.visible else "Open Settings"

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

        for excluded_col in excluded_columns:
            col_name = config.settings.get(excluded_col, excluded_col)
            if col_name in cols:
               cols.remove(col_name)
        
        cols = [col for col in cols if self.df[col].dtype != "object"]

        return cols
    
    def get_id(self):
        id_col = config.settings["id_col"]
        if id_col == "Use Index":
            ids = self.df.index.values
        else:
            ids = self.df[id_col].values
        return ids
        
    
    def remove_shared_data(self):
        """Removes subscriptions and published data from the shared data"""
        shared_data.cleanup_extension_panel(self.panel_id)
        print(f"[{self.panel_id}] removed from shared data")

    def remove_src_listener(self):
        """Removes the callback to a change in the selected source"""
        if self.src is not None and hasattr(self, "_src_callback"):
            try:
                self.src.remove_on_change("data", self._src_callback)
                print(f"[{self.panel_id}] Listener removed")
            except Exception as e:
                print(f"[{self.panel_id}] Error removing src listener: {e}")

    def cleanup_panel_plot(self):
        self.remove_shared_data()
        self.remove_src_listener()
    


class ScatterPlotDashboard(BasePlotClass):
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

    X_variable = param.Selector(objects=["0"], default="0", doc="Selection box for the X axis of the plot.")
    Y_variable = param.Selector(objects=["1"], default="1", doc="Selection box for the Y axis of the plot.")
    log_xscale = param.Boolean(default=False, doc = "Use log for x axis")
    log_yscale = param.Boolean(default=False, doc = "Use log for y axis")
    label_selector = param.ListSelector(default=["All"], objects=["All"], doc="Labels to plot")
    plot_mode = param.Selector(default="tap", objects=["tap", "rasterized"], doc= "Plot Mode")
    


    def __init__(self, src, close_button):
        super().__init__(src, close_button)
        self._src_callback = self._change_source_cb
        self.src.on_change("data", self._src_callback)
        self.update_variable_lists(excluded_columns = ["id_col", "label_col", "ra_dec"])
        
        self.settings_panel = pn.Column(
            pn.Param(
                self,
                parameters=[
                    "log_xscale", "log_yscale", "label_selector", "plot_mode"
                ],
                widgets={
                        "label_selector": {"type": pn.widgets.MultiChoice, "width": 200, "height": 80},
                        "plot_mode":  {"type" : pn.widgets.RadioBoxGroup}
                },
                show_name=False,
                sizing_mode="stretch_width"
            ),
            visible=False,
            margin=(10, 0, 0, 0)
        )



    def update_variable_lists(self, excluded_columns = ["id_col", "label_col", "ra_dec"] ):
        self.param.X_variable.objects = self.get_variable_list(excluded_columns=excluded_columns)
        self.param.Y_variable.objects = self.get_variable_list(excluded_columns=excluded_columns)
        self.param.X_variable.default = config.settings["default_vars"][0]
        self.param.Y_variable.default = config.settings["default_vars"][1]
        self.X_variable = config.settings["default_vars"][0]
        self.Y_variable = config.settings["default_vars"][1]
        self.param.label_selector.objects = ["All"] + list(config.settings["strings_to_labels"].keys())

    def _change_source_cb(self, attr, old, new):
        selected_src_plot = self.plot_selected(self.X_variable, self.Y_variable)
        if selected_src_plot is not None:
            self.figure.object = hv.Overlay(self.main_plot + selected_src_plot).collate()

    @staticmethod
    def get_axis_limits(x_var, Nsigma = 3):
        x = x_var[np.isfinite(x_var)]
        if len(x) == 0:
            return 0, 1 
        max_x = np.max(x)
        min_x = np.min(x)
        x_sd = np.std(x)
        x_mu = np.mean(x)
        max_x = np.min([x_mu + Nsigma * x_sd, max_x])
        min_x = np.max([x_mu - Nsigma * x_sd, min_x])
        return min_x, max_x
    

    @param.depends("X_variable", "Y_variable", "label_selector", "log_xscale",
                   "log_yscale", "plot_mode",
                   watch=True)
    def _update_plot(self):
        self.main_plot = self.plot()
        selected_src_plot = self.plot_selected(self.X_variable, self.Y_variable)
        if selected_src_plot is not None:
            self.figure.object = hv.Overlay(self.main_plot + selected_src_plot).collate()
        else:
            self.figure.object = self.main_plot
    

    def get_scatter_hv(self, x, y, sourceid = None,  plot_mode = "tap", color = "blue"):

        min_x, max_x = self.get_axis_limits(x)
        min_y, max_y = self.get_axis_limits(y)
        
        if plot_mode == "tap" and (sourceid is not None):
            points = hv.Points((x, y, sourceid), kdims=["x", "y"], vdims=["id"]).opts(
                     size = 5,
                     xlim=(min_x, max_x),
                     ylim=(min_y, max_y),
                     tools = ["tap", "box_select"],
                     active_tools=["tap"],
                     selection_fill_color="red",
                     nonselection_alpha=0.4,
                     logx = self.log_xscale,
                     logy = self.log_yscale,
                     xlabel=self.X_variable,
                     ylabel=self.Y_variable,
                     color=color,)
            
            sel_stream = streams.Selection1D(source=points)

            def tap_callback(event):
                if event.new:
                   shared_data.publish(self.panel_id, "selected_sourceid", str(sourceid[event.new[0]]))
                   for idx in event.new:
                       print(sourceid[idx])

            sel_stream.param.watch(tap_callback, 'index')
                   
        else:
            points = hv.Points((x, y), kdims=["x", "y"]).opts( logx = self.log_xscale,
                     logy = self.log_yscale)
            points = dynspread(datashade(points, 
                                       aggregator = ds.count(),
                                       cmap = [color],
                            ).opts(
                            xlim=(min_x, max_x),
                            ylim=(min_y, max_y),
                           active_tools = [], 
                ),
                threshold=0.75,
                how="saturate").opts(legend_position="bottom_right")
            
        return points
   
    
    def plot(self, x_var = None, y_var = None):

        if x_var is None:
            x_var = self.df[self.X_variable].to_numpy()
        if y_var is None:
            y_var = self.df[self.Y_variable].to_numpy()
        
        strings_to_plot = self.label_selector
       
        sourceid = self.get_id().astype(str) if self.plot_mode == "tap" else None
        
        if bool(strings_to_plot) and ("All" not in strings_to_plot or len(strings_to_plot)>1):
           labels = self.df[config.settings["label_col"]]
           labels_to_plot = [config.settings["strings_to_labels"][i] for i in strings_to_plot if i != "All"]
        
        else:
            labels_to_plot = []
       
        self.overlays = []
        if "All" in strings_to_plot:
            h = self.get_scatter_hv(x_var, y_var,  sourceid = sourceid,  plot_mode = self.plot_mode, color = "blue")
            self.overlays.append(h)
            
            
        for i, label_to_plot in enumerate(labels_to_plot):
            select = labels == label_to_plot
            label_sourceid = sourceid[select] if sourceid is not None else None
            h = self.get_scatter_hv(x_var[select], y_var[select],
                                    sourceid = label_sourceid,
                                    plot_mode = self.plot_mode,
                                    color = config.settings["label_colours"][label_to_plot])
                                    
            self.overlays.append(h)          
        plot = hv.Overlay(self.overlays).opts(active_tools = [], xlabel=self.X_variable,
                                            ylabel=self.Y_variable)
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
                active_tools=[],
                logx = self.log_xscale,
                logy = self.log_yscale)
            return selected_plot
        

    def panel(self):
        self._update_plot()
        return pn.Card(
                  pn.Column(
                      pn.Row(self.figure, sizing_mode="scale_both"),
                        self.settings_panel, scroll = True),
                  header=pn.Row(
                        pn.Spacer(width=25,),
                        self.close_button,
                        pn.Row(self.param.X_variable, max_width=100),
                        pn.Row(self.param.Y_variable, max_width=100),
                        self.settings_button,
                        max_width=400,
                    ),
            collapsible=False,
            sizing_mode="stretch_both",
        )


class HistoDashboard(BasePlotClass):
    
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
        
        super().__init__(src, close_button)
        self._src_callback = self._change_source_cb
        self.src.on_change("data", self._src_callback)
        self.update_variable_lists(excluded_columns = ["id_col", "ra_dec"])

        self.settings_panel = pn.Column(
            pn.Param(
                self,
                parameters=[
                    "log_xscale", "log_yscale", "density", "cumulative",
                    "Nbins", "range_min", "range_max", "label_selector"
                ],
                widgets={
                    "range_min": {"type": pn.widgets.FloatInput, "placeholder": "None"},
                    "range_max": {"type": pn.widgets.FloatInput, "placeholder": "None"},
                    "Nbins": {"throttled": True},
                    "label_selector": {"type": pn.widgets.MultiChoice, "width": 200, "height": 80}
                },
                show_name=False,
                sizing_mode="stretch_width"
            ),
            visible=False,
            margin=(10, 0, 0, 0)
        )
    

    def update_variable_lists(self, excluded_columns = ["id_col", "ra_dec"] ):
        self.param.X_variable.objects = self.get_variable_list(excluded_columns=excluded_columns)
        self.param.X_variable.default = config.settings["default_vars"][0]
        self.X_variable = config.settings["default_vars"][0]
        self.param.label_selector.objects = ["All"] + list(config.settings["strings_to_labels"].keys())

    def _change_source_cb(self, attr, old, new):
        selected_src_plot = self.plot_selected(self.X_variable)
        if selected_src_plot is not None:
            self.figure.object = hv.Overlay(self.main_plot + selected_src_plot).collate()


    @param.depends(
        "X_variable", "log_xscale", "log_yscale", "density", "cumulative",
        "Nbins", "range_min", "range_max", "label_selector",
        watch = True)
    def _update_plot(self):
        self.main_plot = self.plot_hv()
        selected_src_plot = self.plot_selected(self.X_variable)
        if selected_src_plot is not None:
            self.figure.object = hv.Overlay(self.main_plot + selected_src_plot).collate()
        else:
            self.figure.object = hv.Overlay(self.main_plot)

    @staticmethod
    def get_histogram_hv(x_var, Nbins = 10, log_x = False, log_y = False, density = False, cumulative = False, 
                      range = (-np.inf, np.inf), label = "", xlabel = "x", ylabel = "frequency",
                      **kwargs):
        
        x = x_var[np.isfinite(x_var)]
        xmin = max(np.min(x),range[0])
        xmax = min(np.max(x),range[1])
        
        #if range[1] < xmin or range[0] > xmax i get an error due to bins not increasing
        if xmin > xmax:
            print("Warning, Range max < than minimum value spanned by the data"
                  "or Range min > than maximum value spanned by the data")
            xmin = xmax

        
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
        
        ylim = (0.2,None) if log_y else (0,None)   #holoviews doesn't like no ylim passed with log yscale
        ylim = (np.min(stats[stats>0])/5, None) if (density and log_y) else ylim 
        
        histogram = hv.Histogram((edges, stats), kdims= [xlabel], 
                                 vdims=[ylabel], label = label).opts(logy = log_y,
                                                                logx = log_x,
                                                                ylim = ylim,
                                                                xlim = (xmin, xmax),
                                                                active_tools = [],
                                                                **kwargs)
        
        return histogram, xmin, xmax

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
        xmin, xmax = np.inf, -np.inf 
        
        xlabel=x_var_name
        ylabel= "% of Sources" if self.density else "# Sources" 


        if "All" in strings_to_plot:
            h, xmin_temp, xmax_temp = self.get_histogram_hv(x_var, Nbins = self.Nbins, 
                            log_x = self.log_xscale, log_y = self.log_yscale,
                            cumulative = self.cumulative, density = self.density,
                            range = (self.range_min, self.range_max),
                            label = "All", xlabel=xlabel, ylabel=ylabel,
                            **{"fill_color" : "blue", "line_color" : "blue"})
            self.overlays.append(h)
            xmin = min(xmin, xmin_temp)
            xmax = max(xmax, xmax_temp)
            
        for i, label_to_plot in enumerate(labels_to_plot):
            h, xmin_temp, xmax_temp = self.get_histogram_hv(x_var[labels == label_to_plot], Nbins = self.Nbins, 
                            log_x = self.log_xscale, log_y = self.log_yscale,
                            cumulative = self.cumulative, density=self.density,
                            range = (self.range_min, self.range_max),
                            label = config.settings["labels_to_strings"][str(label_to_plot)],
                            xlabel=xlabel, ylabel=ylabel,
                            **{"fill_color" : config.settings["label_colours"][label_to_plot] if i < 2 else "none",
                               "line_color" : config.settings["label_colours"][label_to_plot],
                               "line_width" : 1.5,
                               "fill_alpha" : 0.7,
                            }
                            )
            self.overlays.append(h)
            xmin = min(xmin, xmin_temp)
            xmax = max(xmax, xmax_temp)

        plot = hv.Overlay(self.overlays).opts(active_tools = [],
                                              xlim = (xmin, xmax),
                                              )
        return plot
    
    
    def plot_selected(self, x_var):
        cols = list(self.df.columns)
        if len(self.src.data[cols[0]]) == 1:
            selected = pd.DataFrame(self.src.data, columns=cols, index=[0])
        else:
            return None
        if selected.shape[0] > 0:
            selected_plot = hv.VLine(selected[x_var].iloc[0]).opts(
                                  color="black",
                                  line_dash = "dashed",
                                  line_width = 1,
                                  active_tools = [],
                                  )
            return selected_plot

    def panel(self):
        self._update_plot()
        return pn.Card(
                        pn.Column(
                               pn.Row(self.figure, sizing_mode="scale_both"),
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
        self.counter = 0
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