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

from astronomicAL.utils.optimise import matches_type


class BasePlotClass(param.Parameterized):

    X_variable = param.Selector(objects=["0"], default="0", doc= "Selection box for the X axis of the plot")
    log_xscale = param.Boolean(default=False, label = "log x",  doc = "Use log for x axis")
    log_yscale = param.Boolean(default=False, label = "log y", doc = "Use log for y axis")
    label_selector = param.ListSelector(default=["All"], objects=["All"], doc= "Labels to plot")
    
    selector_params = ("X_variable",)

    def  __init__(self,  src, close_button, context = None):
        super().__init__()

        self.context = context
        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config
        self.df = self.config.main_df

        self._disposed = False
        self._event_subs = []      # EventBus Subscription handles
        self._periodic_cbs = []    # pn.state periodic callbacks (if you ever add them)
        self._bokeh_on_change = []  # list of (model, attr, callback)

        self.panel_id = str(uuid.uuid4()) 
        self.src = src
        self.close_button = close_button
        self.figure = pn.pane.HoloViews(sizing_mode="stretch_both")
        self.settings_button = pn.widgets.Button(name="Open Settings", button_type="primary",  max_height = 40, max_width=100)
        self.settings_button.on_click(self._toggle_settings_panel)
    
    def watch_bokeh(self, model, attr: str, callback):
        """Register and track Bokeh model.on_change callbacks for unified disposal."""
        if model is None:
            return
        try:
            model.on_change(attr, callback)
            self._bokeh_on_change.append((model, attr, callback))
        except Exception:
            pass

    def unwatch_all_bokeh(self):
        """Remove all tracked Bokeh callbacks (idempotent)."""
        for model, attr, callback in list(getattr(self, "_bokeh_on_change", [])):
            try:
                model.remove_on_change(attr, callback)
            except Exception as e:
                # Ignore double-remove noise
                if "list.remove(x): x not in list" in str(e):
                    pass
            # continue regardless
        self._bokeh_on_change = []

    def subscribe_event(self, topic: str, callback):
        """
        Subscribe to EventBus and track the subscription so dispose() can unsubscribe.
        """
        if not self.context or not getattr(self.context, "events", None):
            return None
        sub = self.context.events.subscribe(topic, callback)
        self._event_subs.append(sub)
        return sub
    
    def _dispose_impl(self):
        """Subclass-specific cleanup hook (override if needed)."""
        return

    def dispose(self):
        if getattr(self, "_disposed", False):
            return
        self._disposed = True

        # 1) subclass cleanup first
        try:
            self._dispose_impl()
        except Exception:
            pass

        # 2) unwatch bokeh callbacks (including src.on_change if registered via watch_bokeh)
        try:
            self.unwatch_all_bokeh()
        except Exception:
            pass

        # 3) Unsubscribe EventBus
        if self.context and getattr(self.context, "events", None):
            for sub in list(getattr(self, "_event_subs", [])):
                try:
                    self.context.events.unsubscribe(sub)
                except Exception:
                    pass
        self._event_subs = []

        # 4) Stop periodic callbacks
        for cb in list(getattr(self, "_periodic_cbs", [])):
            try:
                cb.stop()
            except Exception:
                pass
        self._periodic_cbs = []


    def update_df(self):
        self.df = self.config.main_df

    def _toggle_settings_panel(self, event):
        self.settings_panel.visible = not self.settings_panel.visible
        self.settings_button.name = "Close Settings" if self.settings_panel.visible else "Open Settings"
    
    def get_column_list(self, excluded_columns = ["id_col", "ra_dec", "label_col"],
                              excluded_types = ["object"], allowed_types = None):
        """
        Returns the list of columns used for panel.widgets.Selector according to their type
        -----
        excluded_columns : list of columns which are removed regardless of their type
        allowed_types : list ["float", "numeric", "int"], if not None, only columns with this type are kept
        excluded_types = list ["float", "object"] list, columns with this types are removed

        It is a bit tricky, in the sense that if you pass an empty/None allowed_types, all types are kept
        """

        cols = list(self.df.columns)
        
        for excluded_col in excluded_columns:
            col_name = self.config.settings.get(excluded_col, excluded_col)
            if col_name in cols:
               cols.remove(col_name)
        
        if allowed_types:
            cols = [col for col in cols if matches_type(self.df[col].dtype, allowed_types)]
        if excluded_types:
            cols = [col for col in cols if not matches_type(self.df[col].dtype, excluded_types)]
        return cols

    def get_id(self):
        id_col = self.config.settings["id_col"]
        if id_col == "Use Index":
            ids = self.df.index.values
        else:
            ids = self.df[id_col].values
        return ids
    
    def _initialise_settings_dictionary(self, key_name, default_values):
        """
        key_name = 'Histogram_plot_settings', 'Scatter_plot_settings' or 'Density_plot_settings'
        default_values = Dictionary with key-values to be used as default ones
        """
        settings_dict = self.config.settings.setdefault(key_name, {})
        for key, value in default_values.items():
            if key not in settings_dict:
                settings_dict[key] = value

    def _initialise_selector_options(self):
        """Initilaises the available options for params objects which allow selection"""
        for name in self.selector_params:
            self.param[name].objects = self.available_columns

    def _initialise_param_objects(self,  **extra_params):
        """
        Method to initialise the param object which goverrn the behaviour of the plot, setting their initial values 
        and the allowed options.

        Parameters:
        -------------
        extra_params: param_name = value, for param objects which are not used in both Scatter and histogram plot
        """

        self._initialise_selector_options()
        self.param.label_selector.objects = ["All"] + list(self.config.settings["strings_to_labels"].keys())

        self.update_df()

        self.param.update(
                    X_variable = self._get_from_settings_dictionary("X_variable", self.available_columns[0]),
                    label_selector = self._get_from_settings_dictionary("label", ['All']),
                    log_xscale = self._get_from_settings_dictionary("log_x", False),
                    log_yscale = self._get_from_settings_dictionary("log_y", False),
                    **extra_params
                )
    
    def get_toolbar(self):

        toolbar = pn.Row(
                        pn.Spacer(width=25,),
                        self.close_button,
                        pn.Row(self.param.X_variable, max_width=100),
                        self.settings_button,
                        max_width=400, max_height=50
                    )

        return toolbar

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

    Y_variable = param.Selector(objects=["1"], default="1", doc="Selection box for the Y axis of the plot")
    plot_mode = param.Selector(default="tap", objects=["tap", "rasterized"], doc= "Plot Mode")
    selector_params = ("X_variable", "Y_variable")


    def __init__(self, src, close_button, context = None):
        super().__init__(src, close_button, context = context)

        self.context = context

        self._src_callback = self._change_source_cb
        self.watch_bokeh(self.src, "data", self._src_callback)
        self.available_columns = self.get_column_list(excluded_columns = ["id_col", "label_col", "ra_dec"])
        
        #In exploring mode there is no default variable in settings. Kept the config.settings.get for consistency
        self._initialise_settings_dictionary(key_name = "Scatter_plot_settings",
                                             default_values =  {
                                             "X_variable" : self.config.settings.get("default_vars", self.available_columns[:2])[0],
                                             "Y_variable" : self.config.settings.get("default_vars", self.available_columns[:2])[1],
                                             "log_x" : False,
                                             "log_y" : False,
                                             "labels" : ["All"],
                                             "mode" : "tap"})

        self._initialise_param_objects(
                                       Y_variable = self._get_from_settings_dictionary("Y_variable", self.available_columns[0]),
                                       plot_mode = self._get_from_settings_dictionary("mode", "tap"),
                                       )
        
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
    

    def _get_from_settings_dictionary(self, key, default):
        value = self.config.settings["Scatter_plot_settings"].get(key, default)
        return value
    
    
    def _update_all_settings_dictionary(self):
        new_values =  {"X_variable" : self.X_variable,
                       "Y_variable" : self.Y_variable,
                       "log_x" : self.log_xscale,
                       "log_y" : self.log_yscale,
                       "labels" : self.label_selector,
                       "mode" : self.plot_mode,
        }
        self.config.settings["Scatter_plot_settings"].update(new_values)


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
        self._update_all_settings_dictionary()
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
                size=4,

                # Make dense clouds readable
                alpha=0.25,

                # Remove default outline (big visual improvement)
                line_alpha=0.0,

                # Keep axis limits etc
                xlim=(min_x, max_x),
                ylim=(min_y, max_y),
                tools=["tap", "box_select", "wheel_zoom", "pan", "reset"],
                active_tools=["wheel_zoom"],

                # Better selection styling
                selection_alpha=1.0,
                selection_color="orange",
                selection_line_color="black",
                selection_line_width=2,

                nonselection_alpha=0.08,
                nonselection_color=color,   # keep same hue but faded
                nonselection_line_alpha=0.0,

                logx=self.log_xscale,
                logy=self.log_yscale,
                xlabel=self.X_variable,
                ylabel=self.Y_variable,
                color=color,
            )
            
            sel_stream = streams.Selection1D(source=points)

            def tap_callback(event):
                if not event.new:
                    return

                src_id = str(sourceid[event.new[0]])

                if getattr(self, "context", None) and getattr(self.context, "events", None):
                    self.context.events.publish(
                        "selection.sourceid.changed",
                        {"sourceId": src_id, "origin": "ScatterPlotDashboard", "panel_id": self.panel_id},
                    )

                for idx in event.new:
                    print(sourceid[idx])

            sel_stream.param.watch(tap_callback, "index")
                   
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
           labels = self.df[self.config.settings["label_col"]]
           labels_to_plot = [self.config.settings["strings_to_labels"][i] for i in strings_to_plot if i != "All"]
        
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
                                    color = self.config.settings["label_colours"][label_to_plot])
                                    
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
            selected_plot = hv.Scatter(selected, x_var, y_var).opts(
                marker="circle",
                size=14,
                fill_alpha=0.0,       # hollow
                line_color="black",
                line_width=3,

                active_tools=[],
                logx=self.log_xscale,
                logy=self.log_yscale,
            )
            return selected_plot


    def get_toolbar(self):

        toolbar = pn.Row(
                        pn.Spacer(width=25,),
                        self.close_button,
                        pn.Row(self.param.X_variable, max_width=100),
                        pn.Row(self.param.Y_variable, max_width=100),
                        self.settings_button,
                        max_width=400, max_height=50
                    )

        return toolbar

    def panel(self):
        self._update_plot()

        toolbar = self.get_toolbar()

        body = pn.Column(
                      pn.Row(self.figure, sizing_mode="scale_both"),
                        self.settings_panel, scroll = True)
        return pn.Column(
            toolbar, body,
            sizing_mode="stretch_both",
        )


class HistoDashboard(BasePlotClass):
    
    density = param.Boolean(default=False, doc = None )
    cumulative = param.Boolean(default=False, doc = None)
    Nbins = param.Integer(default=10, bounds=(2, 200), doc = "Number of bins")
    range_min = param.Number(default= None, bounds=(-np.inf, np.inf), allow_None= True,  doc= "Range min")
    range_max = param.Number(default= None, bounds=(-np.inf, np.inf), allow_None= True, doc= "Range max")

    def __init__(self, src, close_button, context = None):
        super().__init__(src, close_button, context = context)

        self.context = context

        self._src_callback = self._change_source_cb
        self.watch_bokeh(self.src, "data", self._src_callback)
        self.available_columns = self.get_column_list(excluded_columns = ["id_col", "ra_dec"])
        
        self._initialise_settings_dictionary(key_name = "Histogram_plot_settings",
                                             default_values =  {
                                             "X_variable" : self.config.settings.get("default_vars", self.available_columns[:2])[0],
                                             "log_x" : False,
                                             "log_y" : False,
                                             "density" : False,
                                             "cumulative" : False,
                                             "Nbins" : 10,
                                             "range" : (-np.inf, np.inf),
                                             "labels" : ["All"],
                                             })

        self._initialise_param_objects(
                                    cumulative = self._get_from_settings_dictionary("cumulative", False),
                                    density = self._get_from_settings_dictionary("density", False),
                                    Nbins = self._get_from_settings_dictionary("Nbins", 10),
                                    range_min = self._get_from_settings_dictionary("range", (-np.inf, np.inf))[0],
                                    range_max = self._get_from_settings_dictionary("range", (-np.inf, np.inf))[1],
                                    )

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
    
        
    def _change_source_cb(self, attr, old, new):
        selected_src_plot = self.plot_selected(self.X_variable)
        if selected_src_plot is not None:
            self.figure.object = hv.Overlay(self.main_plot + selected_src_plot).collate()

    def _get_from_settings_dictionary(self, key, default):
        value = self.config.settings["Histogram_plot_settings"].get(key, default)
        return value
    
    def _update_all_settings_dictionary(self):
        new_values =  {"X_variable" : self.X_variable,
                       "log_x" : self.log_xscale,
                       "log_y" : self.log_yscale,
                       "labels" : self.label_selector,
                       "cumulative" : self.cumulative,
                       "density" : self.density,
                       "Nbins" : self.Nbins,
                       "range" : (self.range_min, self.range_max),
        }
        self.config.settings["Histogram_plot_settings"].update(new_values)


    @param.depends(
        "X_variable", "log_xscale", "log_yscale", "density", "cumulative",
        "Nbins", "range_min", "range_max", "label_selector",
        watch = True)
    def _update_plot(self):
        self._update_all_settings_dictionary()
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
        
        xmin, xmax = range
        xmin = -np.inf if xmin is None else xmin
        xmax =  np.inf if xmax is None else xmax
        
        x = x_var[np.isfinite(x_var)]
        xmin = max(np.min(x), xmin)
        xmax = min(np.max(x), xmax)
        
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
           labels = self.df[self.config.settings["label_col"]]
           labels_to_plot = [self.config.settings["strings_to_labels"][i] for i in strings_to_plot if i != "All"]
        
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
                            label = self.config.settings["labels_to_strings"][str(label_to_plot)],
                            xlabel=xlabel, ylabel=ylabel,
                            **{"fill_color" : self.config.settings["label_colours"][label_to_plot] if i < 2 else "none",
                               "line_color" : self.config.settings["label_colours"][label_to_plot],
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

    def get_toolbar(self):
        toolbar = pn.Row(
                    pn.Spacer(width=25),
                    self.close_button,
                    pn.Row(self.param.X_variable, max_width=100),
                    self.settings_button, max_height=50
                )
        
        return toolbar

    def panel(self):
        self._update_plot()

        toolbar = self.get_toolbar()

        body = pn.Column(
                    pn.Row(self.figure, sizing_mode="scale_both"),
                    self.settings_panel, scroll = True)
        return pn.Column(
                    toolbar, body,
                        sizing_mode="stretch_both",
                    )

class DensityPlotDashboard(BasePlotClass):

    Y_variable = param.Selector(objects=["1"], default="1", doc="Selection box for the Y axis of the plot")
    selector_params = ("X_variable", "Y_variable")
    Nbins = param.Integer(default=10, bounds=(2, 200), doc = "Number of bins per axis")
    x_range_min = param.Number(default = None, bounds=(-np.inf, np.inf), allow_None= True, doc = "X variable range min")
    x_range_max = param.Number(default = None, bounds=(-np.inf, np.inf), allow_None= True, doc = "X variable range max")
    y_range_min = param.Number(default = None, bounds=(-np.inf, np.inf), allow_None= True, doc = "Y variable range min")
    y_range_max = param.Number(default = None, bounds=(-np.inf, np.inf), allow_None= True, doc = "Y variable range max")
    log_zscale = param.Boolean(default=False, label = "log density",  doc = "Use log for density color")

    clim = param.Integer(default=10, bounds=(2, 1000), doc = "Number of bins per axis")

    def __init__(self, src, close_button, context = None):
        super().__init__(src, close_button, context = context)

        self.context = context

        self._src_callback = self._change_source_cb
        self.watch_bokeh(self.src, "data", self._src_callback)
        self.available_columns = self.get_column_list(excluded_columns = ["id_col", "label_col", "ra_dec"])
        
        self._initialise_settings_dictionary(key_name = "Density_plot_settings",
                                             default_values =  {
                                             "X_variable" : self.config.settings.get("default_vars", self.available_columns[:2])[0],
                                             "Y_variable" : self.config.settings.get("default_vars", self.available_columns[:2])[1],
                                             "log_x" : False,
                                             "log_y" : False,
                                             "labels" : ["All"],
                                             "x_range" : (-np.inf, np.inf),
                                             "y_range" : (-np.inf, np.inf),
                                             "Nbins" : 20,
                                             "log_z" : False})

        self._initialise_param_objects(
                                       Y_variable = self._get_from_settings_dictionary("Y_variable", self.available_columns[0]),                                   
                                       Nbins = self._get_from_settings_dictionary("Nbins", 10),
                                       log_zscale = self._get_from_settings_dictionary("log_z", False),
                                       x_range_min = self._get_from_settings_dictionary("x_range", (-np.inf, np.inf))[0],
                                       x_range_max = self._get_from_settings_dictionary("x_range", (-np.inf, np.inf))[1],
                                       y_range_min = self._get_from_settings_dictionary("y_range", (-np.inf, np.inf))[0],
                                       y_range_max = self._get_from_settings_dictionary("y_range", (-np.inf, np.inf))[1],
                                    )


        self.param_widgets = {
                              "log_xscale": pn.widgets.Checkbox.from_param(self.param.log_xscale),
                              "log_yscale": pn.widgets.Checkbox.from_param(self.param.log_yscale),
                              "log_zscale": pn.widgets.Checkbox.from_param(self.param.log_zscale),
                              "Nbins": pn.widgets.IntSlider.from_param(self.param.Nbins, throttled=True),
                              "x_range_min" : pn.widgets.FloatInput.from_param(self.param.x_range_min),
                              "x_range_max" : pn.widgets.FloatInput.from_param(self.param.x_range_max),
                              "y_range_min" : pn.widgets.FloatInput.from_param(self.param.y_range_min),
                              "y_range_max" : pn.widgets.FloatInput.from_param(self.param.y_range_max),
                              "label_selector": pn.widgets.MultiChoice.from_param(self.param.label_selector, width=200, height=80),
                             }
        

        self.settings_panel = pn.Column(pn.Row(self.param_widgets["log_xscale"], self.param_widgets["log_yscale"], self.param_widgets["log_zscale"] ),
                                        self.param_widgets["Nbins"],
                                        pn.Column(pn.Row(self.param_widgets["x_range_min"], self.param_widgets["x_range_max"]),
                                                  pn.Row(self.param_widgets["y_range_min"], self.param_widgets["y_range_max"])),
                                        self.param_widgets["label_selector"],
                                        sizing_mode="stretch_width",
                                        visible=False,
                                        margin=(10, 0, 0, 0)       
                                        )


    def _get_from_settings_dictionary(self, key, default):
        value = self.config.settings["Density_plot_settings"].get(key, default)
        return value
    
    
    def _update_all_settings_dictionary(self):
        new_values =  {"X_variable" : self.X_variable,
                       "Y_variable" : self.Y_variable,
                       "log_x" : self.log_xscale,
                       "log_y" : self.log_yscale,
                       "labels" : self.label_selector,
                       "Nbins" : self.Nbins,
                       "x_range" : (self.x_range_min, self.x_range_max),
                       "y_range" : (self.y_range_min, self.y_range_max)
        }
        self.config.settings["Density_plot_settings"].update(new_values)


    def _change_source_cb(self, attr, old, new):
        selected_src_plot = self.plot_selected(self.X_variable, self.Y_variable)
        if selected_src_plot is not None:
            self.figure.object = hv.Overlay(self.main_plot + selected_src_plot).collate()



    @param.depends("X_variable", "Y_variable", "label_selector", "log_xscale",
                   "log_yscale", "log_zscale",
                   "Nbins", "x_range_min", "x_range_max", "y_range_min",
                   "y_range_max",
                   watch=True)
    
    def _update_plot(self):
        self._update_all_settings_dictionary()
        self.main_plot = self.plot()
        selected_src_plot = self.plot_selected(self.X_variable, self.Y_variable)
        if selected_src_plot is not None:
            self.figure.object = hv.Overlay(self.main_plot + selected_src_plot).collate()
        else:
            self.figure.object = self.main_plot
    

    def get_density_hv(self, x_var, y_var, log_x = False, log_y = False, 
                       x_range = (-np.inf, np.inf), y_range = (-np.inf, np.inf),
                       log_z = False,
                       Nbins = 25, cmap = "viridis"):
        
        

        xmin, xmax = x_range
        xmin = -np.inf if xmin is None else xmin
        xmax =  np.inf if xmax is None else xmax

        ymin, ymax = y_range
        ymin = -np.inf if ymin is None else ymin
        ymax =  np.inf if ymax is None else ymax
       
        select = np.logical_and.reduce([np.isfinite(x_var), np.isfinite(y_var), 
                                        x_var >= xmin, x_var < xmax,
                                        y_var >= ymin, y_var < ymax])
        
        x, y = x_var[select], y_var[select]
        if log_x:
            x = np.log10(x) 
            xmin, xmax = np.log10(xmin), np.log10(xmax)
        if log_y:
            y = np.log10(y)
            ymin, ymax = np.log10(ymin), np.log10(ymax)

    
    
        density_plot= hv.HexTiles((x, y), kdims=["x", "y"]).opts(
                     gridsize = Nbins,
                     tools = ["hover"],
                     active_tools=[],
                     xlabel=self.X_variable,
                     ylabel=self.Y_variable,
                     xlim = (np.min(x), np.max(x)),
                     ylim = (np.min(y), np.max(y)),
                     logz = log_z,
                     colorbar = True,
                     cmap = cmap)
         
        return density_plot
   
    
    def plot(self, x_var = None, y_var = None):

        if x_var is None:
            x_var = self.df[self.X_variable].to_numpy()
        if y_var is None:
            y_var = self.df[self.Y_variable].to_numpy()
        
        strings_to_plot = self.label_selector
       
        if bool(strings_to_plot) and ("All" not in strings_to_plot or len(strings_to_plot)>1):
           labels = self.df[self.config.settings["label_col"]]
           labels_to_plot = [self.config.settings["strings_to_labels"][i] for i in strings_to_plot if i != "All"]
        
        else:
            labels_to_plot = []
       
        self.overlays = []
        if "All" in strings_to_plot:
            h = self.get_density_hv(x_var, y_var, Nbins = self.Nbins,  
                                    log_x = self.log_xscale, log_y = self.log_yscale,
                                    x_range = (self.x_range_min, self.x_range_max),
                                    y_range = (self.y_range_min, self.y_range_max),
                                    log_z = self.log_zscale,
                                    cmap= "Blues")
            self.overlays.append(h)

        for i, label_to_plot in enumerate(labels_to_plot):
            select = labels == label_to_plot
            h = self.get_density_hv(x_var[select], y_var[select], Nbins = self.Nbins,  
                                    log_x = self.log_xscale, log_y = self.log_yscale,
                                    x_range = (self.x_range_min, self.x_range_max),
                                    y_range = (self.y_range_min, self.y_range_max),
                                    log_z = self.log_zscale,
                                    cmap= "Reds")
                                    
            self.overlays.append(h)          
        plot = hv.Overlay(self.overlays).opts(active_tools = [], xlabel=self.X_variable,
                                            ylabel=self.Y_variable)
        return plot
    
    def plot_selected(self, x_var, y_var):
        selected_plot = hv.Scatter((4, 3))
        return selected_plot

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
    
    def get_toolbar(self):

        toolbar = pn.Row(
                        pn.Spacer(width=25,),
                        self.close_button,
                        pn.Row(self.param.X_variable, max_width=100),
                        pn.Row(self.param.Y_variable, max_width=100),
                        self.settings_button,
                        max_width=400, max_height=50
                    )
        
        return toolbar

    def panel(self):
        self._update_plot()

        toolbar = self.get_toolbar()

        body = pn.Column(
                    pn.Row(self.figure, sizing_mode="scale_both"),
                    self.settings_panel, scroll = True)
        return pn.Column(
            toolbar, body,
            sizing_mode="stretch_both",
        )