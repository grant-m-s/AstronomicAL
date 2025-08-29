from holoviews.operation.datashader import (
    datashade,
    dynspread,
)

import datashader as ds
import holoviews as hv

import astronomicAL.config as config
import numpy as np
import os
import pandas as pd
import panel as pn
import json
import param
import uuid
import matplotlib.pyplot as plt
import concurrent.futures 
import pandas.api.types as pdt
from bokeh.document import without_document_lock
from bokeh.models import  NormalHead
from bokeh.models import Range1d, LinearAxis
from astronomicAL.extensions.shared_data import shared_data
from astronomicAL.extensions.astro_data_utility import DESISpectraClass, EuclidCutoutsClass, EuclidSpectraClass
from astronomicAL.extensions.astro_data_utility import VLASS_cutout, LoTSS_cutout



def get_customplot_dict():

    plot_dict = {
        
        "Euclid Cutout" : lambda data, src, close_button : EuclidPlotClass(data, src, close_button,
                                                            extra_features=[]),

        "DESI Spectra"  : lambda data, src, close_button : SpectrumPlotClass(data, src, close_button,
                                                            extra_features=[], dataset="DESI"), 

        "Euclid Spectra"  : lambda data, src, close_button : SpectrumPlotClass(data, src, close_button,
                                                            extra_features=[], dataset="EuclidSpec"), 

        "SDSS Spectra"  : lambda data, src, close_button : SpectrumPlotClass(data, src, close_button,
                                                            extra_features=[], dataset="SDSS"),

        "BroadBand SED"  : lambda data, src, close_button : SEDPlotClass(data, src, close_button,
                                                            extra_features=["Do not skip to Plot stage"]),
        
        "VLASS Cutout"  : lambda data, src, close_button : RadioClass(data, src, close_button,
                                                            extra_features=[], dataset="VLASS"),
        
        "LoTSS Cutout"  : lambda data, src, close_button : RadioClass(data, src, close_button,
                                                            extra_features=[], dataset="LoTSS")                                                  

    }

    return plot_dict


class CustomPlotClass(param.Parameterized):

    available_stages = ["columns_selection","plot"]

    stage = param.ObjectSelector(default="columns_selection", objects = available_stages)
    
    def __init__(self, data, src, close_button, extra_features):
        super().__init__()
        self.df = data
        self.src = src
        self.extra_features = extra_features
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.close_button = close_button
        self.panel_id = str(uuid.uuid4()) 
        self._get_unknown_columns(self.extra_features)
        self.figure = pn.pane.HoloViews(sizing_mode="stretch_both")
        self.message_pane = pn.pane.Markdown("## Loading...", sizing_mode="stretch_both", max_height = 30)
        self.plot_settings_button = pn.widgets.Button(name="Open Settings", button_type="primary", max_height = 40, max_width=100, sizing_mode="stretch_both" )
        self.plot_settings_button.on_click(self._toggle_settings_panel)
        self.plot_settings_panel = pn.Column(visible = False)

    def _submit_button_cb(self, event):
        for col, widget in self.select_widgets.items():
            selected_value = widget.value
            print(f"{col} --> {selected_value}")
            config.settings[col] = selected_value
        self.stage = "plot"

    def _skip_button_cb(self, event):
        current_index = self.available_stages.index(self.stage)
        self.stage = self.available_stages[current_index + 1]
    
    def _toggle_settings_panel(self, event):
        self.plot_settings_panel.visible = not self.plot_settings_panel.visible
        self.plot_settings_button.name = "Close Settings" if self.plot_settings_panel.visible else "Open Settings"

    def get_selected_source(self):
        if self.src is None:
            return None
        cols = list(self.df.columns)
        if len(self.src.data[cols[0]]) == 1:
            return pd.DataFrame(self.src.data, columns=cols, index=[0])
        return None
    
    def get_value_from_df(self, column):
        selected_source = self.get_selected_source()
        if (selected_source is not None) and self.check_required_column(column):
            return selected_source[column][0]
        return None
            
    def get_ra_dec(self, err_message = "No ra and dec available for this source"):
        ra_dec = self.get_value_from_df("ra_dec")
        if ra_dec is not None:
            ra = float(ra_dec[: ra_dec.index(",")])
            dec = float(ra_dec[ra_dec.index(",") + 1 :])
        else:
            print(err_message)
            ra, dec = None, None
        return ra, dec

    def check_required_column(self, column):
        return column in list(self.df.columns)
    
    def get_column_list(self, excluded_columns = ["id_col", "ra_dec", "label_col"],
                              excluded_types = ["object"], allowed_types = None):
        """Returns the list of columns used for panel.widgets.Selector according to their type
        allowed_types : list ["float", "numeric", "int"] """

        if allowed_types is None:  
            allowed_types = []

        cols = list(self.df.columns)
        
        for excluded_col in excluded_columns:
            col_name = config.settings.get(excluded_col, excluded_col)
            if col_name in cols:
               cols.remove(col_name)
        
        def matches_type(dtype, type_list):
            """Check if a dtype matches any keyword in type_list"""
            for t in type_list:
                if t == "float" and (pdt.is_float_dtype(dtype)):
                    return True
                if t == "int" and (pdt.is_integer_dtype(dtype)):
                    return True
                if t == "number" and (pdt.is_numeric_dtype(dtype)):
                    return True
                if t == "object" and (pdt.is_object_dtype(dtype)):
                    return True
            return False

        if allowed_types:
            cols = [col for col in cols if matches_type(self.df[col].dtype, allowed_types)]
        
        cols = [col for col in cols if not matches_type(self.df[col].dtype, excluded_types)]

        return cols



    def _get_selection_widgets_grid(self, columns_to_select, options = None, allowed_types = None):
        settings_grid = pn.GridBox(ncols=3, sizing_mode = "stretch_width", scroll = True)  
        self.select_widgets = {}
        if options is None:
            options = self.get_column_list(excluded_columns = ["ra_dec", "label_col"],
                              excluded_types = ["object"], allowed_types = allowed_types)
        if len(columns_to_select) > 0:
            for col in columns_to_select:
                select_widget = pn.widgets.Select(name= col, options=options, max_height=120, sizing_mode = "stretch_width")
                settings_grid.append(select_widget)
                self.select_widgets[col] = select_widget
        return settings_grid
        

    def columns_selection_panel(self, columns_to_select, skippable = False,
                                options = None, allowed_types = None,
                                info_text = None):
        settings_grid = self._get_selection_widgets_grid(columns_to_select, options = options, 
                                                         allowed_types = allowed_types)
        
        submit_button = pn.widgets.Button(name='Confirm', button_type='primary', max_height=120)
        submit_button.on_click(self._submit_button_cb)
        skip_button = pn.widgets.Button(name='Skip', button_type='primary', max_height=120)
        skip_button.on_click(self._skip_button_cb)
        if not skippable:
            skip_button.disabled = True
        if info_text is not None:
           card_content = pn.Column(pn.pane.Markdown(info_text, sizing_mode="stretch_width", margin=(15,0,15,15)), 
                                           settings_grid)
        else:
           card_content = settings_grid
        return pn.Card(card_content, header = pn.Row(pn.Spacer(width=25), self.close_button, skip_button, submit_button),
                                sizing_mode="stretch_both", scroll=True, collapsible = False, min_height = 300 )

    
    def _get_unknown_columns(self, columns_needed, change_stage = False,
                            settings_key = None,
                            unknown_stage  = "columns_selection",
                            ready_stage = "plot"):
        """
        Check if required columns exist in config.settings or config.main_df.

        columns_needed : list
            Columns that are required.
        change_stage : bool, optional
            Change stage when unknown columns are found (default: False).
        settings_key : str or None, optional
            If provided, checks within config.settings[settings_key].keys().
            Otherwise, checks directly against config.settings.
        unknown_stage : str, optional
            Stage to set if unknown columns are present (default: 'columns_selection').
        ready_stage : str, optional
            Stage to set if all columns are known (default: 'plot').
        """
        
        current_cols = getattr(config.main_df, "columns", [])
        self.unknown_columns = []

        if settings_key is not None:
            if settings_key not in config.settings:
                config.settings[settings_key] = {}
            settings_dict = config.settings[settings_key]
        else:
            settings_dict = config.settings

        for col in columns_needed:
            if col not in settings_dict:
                print(f"{col} not in config")
                if col not in current_cols:
                    self.unknown_columns.append(col)
                else:
                    settings_dict[col] = col
                            
        if self.unknown_columns:
            if change_stage and (self.stage != unknown_stage):
                self.stage = unknown_stage
        else:
            self.stage = ready_stage
    
    
    def run_multithread(self, function, func_kwargs=None, callback=None, allowed_exceptions=(Exception,)):
        if func_kwargs is None:
            func_kwargs = {}

        def wrapper():
            try:
                result = function(**func_kwargs)
                return result
            except allowed_exceptions as e:
                 print(f"[{self.__class__.__name__}] Exception in thread: {e}")
            return None
        
        future = self.executor.submit(wrapper)

        if callback:
            current_doc = pn.state.curdoc
            if current_doc is not None:
                future.add_done_callback(lambda fut: current_doc.add_next_tick_callback(lambda: callback(fut)))
            else:
                print(f"[{self.__class__.__name__}] Warning: pn.state.curdoc was None when scheduling callback.")

        return future
    
    @staticmethod
    def get_empty_image():
        """Just a white image to update the previous one if the query fails"""
        return hv.Image(np.ones((10,10))).opts(active_tools =[], 
                                            clim = (0,1), toolbar=None,
                                            padding = 0,border = 0,framewise = True, xaxis=None, 
                                         yaxis=None, cmap = "grey")
    
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

    def remove_column_selection(self):
        if hasattr(self, "unknown_columns"):
            for col in self.unknown_columns:
                if col in config.settings:
                    del config.settings[col]
            print(f"[{self.panel_id}] unknown columns selected removed from config")
            
    def cleanup_panel_plot(self):
        self.remove_shared_data()
        self.remove_src_listener()
        self.remove_column_selection()
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
            print(f"[{self.panel_id}] Thread executor shutdown.")


    ##Example method
    def plot(self, N=20):
        self.message_pane.visible = True
        coords = [(i, np.random.random()) for i in range(N)]
        scatter = hv.Scatter(coords).opts(color='black', marker='+')
        self.figure.object = scatter
        self.message_pane.visible = False

    ##Example method
    def get_layout(self):
        points_input = pn.widgets.IntInput(name="Number of points", value=20, start=1, sizing_mode = "stretch_width" )
        def update_points(event):
            N = points_input.value
            self.plot(N)
        self.plot_settings_panel.objects = [points_input]
        points_input.param.watch(update_points, 'value')
        self.plot(points_input.value)
        return pn.Column(self.message_pane, self.figure, self.plot_settings_panel, 
                         sizing_mode="stretch_both", min_height = 450, styles={'background': 'lightgreen'})
    
    def plot_panel(self):
        self.layout = self.get_layout()
        return pn.Card(self.layout, header = pn.Row(pn.Spacer(width=25,),self.close_button, self.plot_settings_button),
                       collapsible = False, sizing_mode="stretch_both", min_height =450,)
    
    @param.depends("stage")                        
    def mypanel(self):
        if self.stage == "columns_selection":
            return self.columns_selection_panel(self.unknown_columns)
        else:
            return self.plot_panel()
        


class EuclidPlotClass(CustomPlotClass):
    def __init__(self, data, src, close_button, extra_features ):
        super().__init__(data, src, close_button, extra_features)
        self._src_callback = self._change_source_cb
        self.src.on_change("data", self._src_callback)
        self.euclid_pane = pn.pane.HoloViews(width=400, height=400) #euclid_pane = Euclid cutout, figure = euclid_pane+overplotted_coordinates
        self.filter = "stacked"
        self.radius = shared_data.get_data("Euclid_radius", 5.0)
        
   
    def _change_source_cb(self, attr, old, new):
        self._initialise_euclid_object()
        self._run_euclid()

    def get_layout(self):
        self._initialise_widgets()
        self._initialise_euclid_object()
        self._subscribe_to_shared()
        self._run_euclid()
        return  pn.Column(self.message_pane, self.figure, self.plot_settings_panel, 
                          scroll = True, sizing_mode = "stretch_both")

    def _initialise_euclid_object(self):
        self.ra, self.dec = self.get_ra_dec()
        if (self.ra is None) or (self.dec is None):
            raise ValueError("RA or DEC is missing")
        else:
            self.euclid_object = EuclidCutoutsClass(self.ra, self.dec, 
                             euclid_filters= ["VIS", "NIR_Y", "NIR_J", "NIR_H"],
                             client = shared_data.get_data("Euclid_client", None))
            self.euclid_object.check_coverage()
            self.overplotted_coordinates = []
            

    def _initialise_widgets(self):

        self.radius_input = pn.widgets.FloatInput(name = "Radius [arcsec]", value = self.radius, 
                                                  step = 0.5, start = 1, end = 100, max_width = 200,
                                                  sizing_mode="stretch_both")
        self.radius_input.param.watch(self._update_radius, "value")

        self.stretching_input = pn.widgets.Select(name = "Stretching function", 
                                                options=  ['Linear', 'Sqrt', 'Log', 'Asinh', 'PowerLaw'],
                                                value = "Linear",
                                                sizing_mode = "stretch_both")
        self.stretching_input.param.watch(self._update_stretching, "value")

        self.contrast_scaler = pn.widgets.RangeSlider(name = "Image scaling", 
                                                    start = 0, end = 1, value = (0,1), step = 0.004, 
                                                    sizing_mode = "stretch_both")
        self.contrast_scaler.param.watch(self._update_intensity_scaling, "value")  

        self.filter_input = pn.widgets.Select(name = "Euclid Filter", 
                                              options =  {"VIS" : "VIS", 'Y' : "NIR_Y", 'J' : "NIR_J", 
                                                        'H' : "NIR_H", 'Color' : "stacked"},
                                                        value  = "stacked",
                                                sizing_mode = "stretch_both")
        self.filter_input.param.watch(self._update_filter, "value")
        
        self.overplot_source_coords_widget = pn.widgets.Checkbox(name = "Source Coordinates")
        self.overplot_source_coords_widget.param.watch(self._overplot_source_coordinates_callback, "value")

        self.overplot_coords_widget = pn.widgets.Checkbox(name = "Spectrum Coordinates")
        self.overplot_coords_widget.param.watch(self._overplot_coordinates_callback, "value")


        self.environment_input = pn.widgets.Select(name = "Euclid Science Archive Environment", 
                                              options =  {"Public Data Release" : "PDR", "Internal Data Release" : "IDR", 
                                                          "On The Fly" : "OTF", "REG" : "REG"},
                                                           value  = "PDR",
                                                           disabled_options=["REG"],
                                                sizing_mode = "stretch_both")
        self.environment_input.param.watch(self._change_euclid_environment, "value")

        self.user_input = pn.widgets.TextInput(name = 'Euclid Science Archive username', 
                                               placeholder = 'Enter your Euclid Science Archive username here',
                                               sizing_mode = "stretch_both")
        self.password_input = pn.widgets.PasswordInput(name = "Password", 
                                                       placeholder = 'Enter your Euclid Science Archive password here',
                                                       sizing_mode = "stretch_both")

        self.confirm_login_button = pn.widgets.Button(name = "Confirm", sizing_mode = "stretch_both", max_height = 30, 
                                                       max_width = 80, button_type= "primary")
        self.confirm_login_button.on_click(self._confirm_login_credentials_cb)

        self.login_column = pn.Column(self.user_input, self.password_input, self.confirm_login_button, visible = False)

        
        
        
        self.plot_settings_panel = pn.Column(self.contrast_scaler, self.radius_input, self.stretching_input, 
                                             self.filter_input,
                                             pn.Row(self.overplot_source_coords_widget,self.overplot_coords_widget), 
                                             self.environment_input, 
                                             self.login_column, 
                                             scroll = True, visible = False)
        


     
    def _update_radius(self, event):
        if event.new: #avoid passing None
            self.radius = event.new
            shared_data.publish(self.panel_id, "Euclid_radius", self.radius)
            self._run_euclid()
        else:
            print("Input a valid value for radius")

    @staticmethod
    def change_intensity_range(image, low, high):
        image = np.clip(image, low, high)
        image = (image-low)/(high-low)
        return np.clip(image, 0,1)

    def _update_intensity_scaling(self, event):
        low, high = event.new
        scaled_image = self.change_intensity_range(self.euclid_object.plot_data[self.filter], 
                                                   low, high)
        self.get_euclid_figure(scaled_image, show_coordinates= self.overplot_source_coords_widget.value)
        self._update_image()
    
    def _overplot_source_coordinates_callback(self, event):
        self.get_euclid_figure(self.euclid_object.plot_data[self.filter], 
                               show_coordinates=event.new)
        self._update_image()

    def _update_filter(self, event):
        self.filter = event.new
        self.get_euclid_figure(self.euclid_object.plot_data[self.filter],
                               show_coordinates= self.overplot_source_coords_widget.value)
        self._update_image()
        
    def _update_stretching(self, event):
        stretch = event.new
        self.euclid_object.get_plot_data(stretch = stretch)
        self.get_euclid_figure(self.euclid_object.plot_data[self.filter], show_coordinates= self.overplot_source_coords_widget.value)
        self._update_image()

    
    def _update_image(self): 
        try:
             self.figure.object = hv.Overlay(self.euclid_fig + self.overplotted_coordinates)
             self.message_pane.visible = False
        except Exception as e:         #too generic
            print("Euclid image unavailable")
            print(e)

    def _add_coordinates(self, coordinates, dataset):
        """Storing Coordinates from DESI/SDSS
           coordinates : dict : {"ra" : [...], "dec" : [...]} 
           dataset : string, key of the dictionary storing the coordinates
        """

        if not coordinates or "ra" not in coordinates or "dec" not in coordinates:
            print("Wrong passed coordinates")
            return
        ra, dec  = coordinates["ra"], coordinates["dec"]
        if not hasattr(self, "stored_spectrum_coordinates"):
            self.stored_spectrum_coordinates = {}
        self.stored_spectrum_coordinates[dataset] = {"ra" : ra, "dec" : dec}

        self.overplot_coords_widget.name = "Spectrum Coordinates"
        if self.overplot_coords_widget.value:
            self._show_overplot_coordinates()
    
    def _show_overplot_coordinates(self):

        if hasattr(self, "stored_spectrum_coordinates"):
            self.overplot_coords_widget.name = "Spectrum Coordinates"
            if self.overplot_coords_widget.value:
                self.overplotted_coordinates = []
                for dataset in self.stored_spectrum_coordinates:
                    print(f"overplotting coordinates for {dataset}")
                    N = len(self.stored_spectrum_coordinates[dataset]["ra"])
                    colors = plt.get_cmap("gist_rainbow", max(N,2))
                    marker = "+" if dataset == "DESI" else "*" #TODO improve
                    label = "Euclid Spectra" if dataset == "EuclidSpec" else f"{dataset} Spectra"
                    for i, (x, y) in enumerate(self.euclid_object.world_2_pix(ra =  self.stored_spectrum_coordinates[dataset]["ra"],
                                                                              dec = self.stored_spectrum_coordinates[dataset]["dec"],
                                                                              filtro = self.filter)):
                        if (0 <= x < self.image_width) and (0 <= y < self.image_height):
                            points = hv.Points([(x,y)], label = label if i == 0 else "")
                            points = points.opts(color = colors(i),
                                                marker = marker, 
                                                size = 20)
                            self.overplotted_coordinates.append(points)
                
                self._update_image()
    
    def _overplot_coordinates_callback(self, event):
        if event.new:
            if not hasattr(self, "stored_spectrum_coordinates"):
                print("No spectrum coordinates available")
                event.obj.name = "Spectrum Coordinates [Not Currently Avaliable]"
                self.overplotted_coordinates = []
                return None
            
            event.obj.name = "Spectrum Coordinates"
            self._show_overplot_coordinates()

        elif not event.new:
            self.overplotted_coordinates = []
        self._update_image()

    def _change_euclid_environment(self, event):
        self.environment = event.new
        if self.environment in  ["IDR", "OTF", "REG"]:
            if os.path.isfile("euclid_credentials.login"):
                print("I found the credential file")
                self.euclid_object.change_environment(environment=self.environment,
                                                      user = None, password = None, 
                                                      credentials_filepath = "euclid_credentials.login")
                
            else:
                user = config.settings.get("EuclidAccountUser", None)
                password = config.settings.get("EuclidAccountUser", None)
                if (user is None) or (password is None):
                    self.login_column.visible = True
                else:
                    self.euclid_object.change_environment(environment=self.environment,
                                                      user = user, password = password)
        else:
            self.euclid_object.change_environment(environment=self.environment)        


    def _confirm_login_credentials_cb(self, event):
        self.login_column.visible = False
        config.settings["EuclidAccountUser"] = self.user_input.value
        config.settings["EuclidAccountPassword"] = self.password_input.value
        self.euclid_object.change_environment(environment=self.environment,
                                                      user = config.settings["EuclidAccountUser"], 
                                                      passwsord = config.settings["EuclidAccountPassword"])



    def get_plot_scale(self):
        bar_length_arcsecond = self.bar_length_pixels * self.euclid_object.arcsec_per_pix[self.filter]
        return bar_length_arcsecond

    
    def get_euclid_figure(self, data, show_coordinates = False, show_scale = True):
        
        self.image_height, self.image_width,  = data.shape[:2]
        bounds = (0, 0, self.image_height, self.image_width)
        
        if len(data.shape) == 3:
            image = hv.RGB(data[::-1,...], bounds=bounds).opts(
                                         active_tools =[], toolbar=None,
                                         padding = 0,
                                         border = 0,
                                         framewise = True,
                                         xaxis=None, 
                                         yaxis=None,
                                         )
        else:
            image = hv.Image(data[::-1,...], bounds=bounds).opts(
                                         active_tools =[], toolbar=None,
                                         padding = 0,
                                         border = 0,
                                         framewise = True,
                                         xaxis=None, 
                                         yaxis=None,
                                         cmap = "grey",
                                         )

        self.euclid_fig = [image]
        
        if show_scale:
            self.bar_length_pixels = self.image_width * 0.2  #always shows a bar 1/5 of the plot 
            x0, y0 = 0.1*self.image_width, 0.1*self.image_height
            x1 = x0 + self.bar_length_pixels
            scale_bar = hv.Curve(([x0, x1], [y0, y0])).opts(color='red', line_width=3)
            scale_text = hv.Text(x=(x0 + x1)/2, y=y0 + y0/2,
                            text=f'{self.get_plot_scale():.1f}"').opts(
                            text_color='red', text_align='center',
                            text_baseline='bottom', fontsize=14
                            )
            self.euclid_fig.extend([scale_bar, scale_text])
        
        if show_coordinates:
            label = f"{np.round(self.ra,3)}, {np.round(self.dec,3)}"
            (x, y) = self.euclid_object.world_2_pix(ra = [self.ra], dec = [self.dec], filtro=self.filter)[0]
            if (0 <= x < self.image_width) and (0 <= y < self.image_height):
                points = hv.Points([(x,y)], label = label)
                points = points.opts(color = "blue",
                                    marker = "+", 
                                    size = 30)
                self.euclid_fig.append(points)
            
        if self.overplot_coords_widget.value:
            self._show_overplot_coordinates()

    def _run_euclid(self):
        """Wrapper for multithreading"""
        self.message_pane.object = "## Loading..."
        self.message_pane.visible = True
        if not self.euclid_object.has_coverage:
            print("The Source is not contained in Euclid mocs")
            self.message_pane.object = "## The source is not in the Euclid covered area"
            self.figure.object =  self.get_empty_image()

        shared_data.publish(self.panel_id, "EuclidCutout_running", True)
 
        def callback(future_obj=None):
            shared_data.publish(self.panel_id, "EuclidCutout_running", False)
            result = future_obj.result() #result = self.euclid_object.plot_data[self.filter] or None
            if result is None:
                self.message_pane.object = "## The Euclid cutout query failed"
                self.message_pane.visible = True #probably already visible
                self.figure.object = self.get_empty_image()
                return

            self.overplot_coords_widget.value = False
            if self.contrast_scaler.value != (0,1):
                low, high =  self.contrast_scaler.value
                scaled_image = self.change_intensity_range(self.euclid_object.plot_data[self.filter], low, high)
                self.get_euclid_figure(scaled_image, show_coordinates = self.overplot_source_coords_widget.value)
            else:
                self.get_euclid_figure( self.euclid_object.plot_data[self.filter], show_coordinates= self.overplot_source_coords_widget.value)
            self._update_image()
            self.message_pane.visible = False
     
        
        self.run_multithread(self.euclid_object.get_final_cutout,
                             func_kwargs = {"radius" : self.radius, "stretch" : self.stretching_input.value, 
                              "filtro" : self.filter_input.value,
                              "reference" : "VIS", "verbose" : True, "return_object" : True}, 
                              callback = callback)
        
    
    def _subscribe_to_shared(self):
        """It manages all the subscriptions to the shared dictionary. not very flexible but it works"""
        desi_callback = lambda coords: self._add_coordinates(coords, "DESI")
        sdss_callback = lambda coords: self._add_coordinates(coords, "SDSS")
        euclid_callback = lambda coords: self._add_coordinates(coords, "EuclidSpec")
        shared_data.replace_subscribe(self.panel_id, "DESI_coordinates", desi_callback)
        shared_data.replace_subscribe(self.panel_id, "SDSS_coordinates", sdss_callback)
        shared_data.replace_subscribe(self.panel_id, "EuclidSpec_coordinates", euclid_callback)
        
        #If DESI/SDSS panel are already initialized, I need to pass the coordinates directly
        if shared_data.get_data("DESI_coordinates"):
            self._add_coordinates(shared_data.get_data("DESI_coordinates"), "DESI")
        if shared_data.get_data("SDSS_coordinates"):
            self._add_coordinates(shared_data.get_data("SDSS_coordinates"), "SDSS")
        if shared_data.get_data("EuclidSpec_coordinates"):
            self._add_coordinates(shared_data.get_data("EuclidSpec_coordinates"), "EuclidSpec")
      


class SpectrumPlotClass(CustomPlotClass):
    

    def __init__(self, data, src, close_button, extra_features, dataset = "DESI"):
        super().__init__(data, src, close_button, extra_features)
        self.figure = pn.Column(scroll = True, sizing_mode = "stretch_both")
        self.dataset = dataset
        self._is_euclid_spec = self.dataset == "EuclidSpec" 
        self._src_callback = self._change_source_cb
        self.src.on_change("data", self._src_callback)
        self.from_sourceId = False
        self.plot_settings_panel = pn.Column(visible = False, scroll = True)
        self.mode_options = ["Use TargetId", "Cone Search"]
        self.chosen_mode = self.mode_options[1]


    def get_layout(self):
        self._initialize_settings_panel()
        self._initialize_spectrum_object()
        self._run_spectrum()
        return pn.Column(self.message_pane, self.figure, self.plot_settings_panel,  scroll = True)
    
    def _change_source_cb(self, attr, old, new):
        #TODO maybe add a method to reload the same object
        if self.stage == "plot":
            self._initialize_spectrum_object()
            self._run_spectrum()

    def _subscribe_to_shared(self):
        if not self.from_sourceId:
            if not shared_data.is_subscribed(self.panel_id, "Euclid_radius"):
               shared_data.subscribe(self.panel_id, "Euclid_radius", self._update_max_separation)

    def _initialize_spectrum_object(self):
        
        self.max_separation = shared_data.get_data("Euclid_radius", 0.5)

        if self.from_sourceId:
            self.selected_source = self.get_selected_source()
            try:
                self.sourceId = int(self.selected_source[config.settings[f"{self.dataset}_TargetID"]].iloc[0])
                self.ra, self.dec = None, None
            except KeyError:
                raise KeyError("Missing column with target ID")
            except ValueError:
                raise ValueError(f"{self.dataset}_TargetID must be convertible to int")
        else:
            self.sourceId = None
            self.ra, self.dec = self.get_ra_dec()
            if (self.ra is None) or (self.dec is None):
                raise ValueError("Missing ra and dec")

        if self._is_euclid_spec:
            self.spectrum_object = EuclidSpectraClass(self.ra, self.dec, max_separation = self.max_separation,
                                             sourceId = self.sourceId)
        else:
            datasets = (["DESI-DR1"] if self.dataset == "DESI"
                else ["BOSS-DR16", "SDSS-DR16"] if self.dataset == "SDSS"
                else None)
            self.spectrum_object = DESISpectraClass(self.ra, self.dec, datasets = datasets ,
                                        sourceId = self.sourceId, max_separation = self.max_separation,
                                        client = shared_data.get_data("Sparcl_client", None))

    def _add_coordinates_to_shared(self, ra, dec):
        """
        ra and dec are lists
        """
        shared_data.publish(self.panel_id, f"{self.dataset}_coordinates", {"ra": ra, "dec": dec})
        return None
        
    def _run_spectrum(self, max_separation = None):
        self.message_pane.object = "## Loading..."
        self.message_pane.visible = True
        shared_data.publish(self.panel_id, f"{self.dataset}_running", True)
        if max_separation is None:
            max_separation = self.max_separation
        
        def callback(future_result = None):
            shared_data.publish(self.panel_id, f"{self.dataset}_running", False)
            if self.spectrum_object.spectra is not None:
                if self.redshift_column_selector.value != "None":
                    redshift_value = self.get_value_from_df(self.redshift_column_selector.value)
                    if redshift_value is not None:
                       self.redshift_input.value = redshift_value
                self._update_plot()
                self._add_coordinates_to_shared(*self.spectrum_object.get_coordinates())
                self.message_pane.visible = False
            else:
                self.message_pane.object = "## Spectrum unavailable"
                self.message_pane.visible = True #probably already visible
                self.figure.objects = [self.get_empty_image()]
                

        self.run_multithread(self.spectrum_object.get_spectra, 
                             func_kwargs = {"max_separation" : max_separation, "return_object" : True},
                             callback=callback)
        

    def _initialize_settings_panel(self):
        self.retrieve_mode_button = pn.widgets.RadioButtonGroup(name="How to retrieve spectrum", options=self.mode_options, 
                                            value = self.chosen_mode, sizing_mode = "stretch_both", max_height = 40)
        
        self.max_separation_input = pn.widgets.FloatInput(name = "Cone Radius [arcsec]", value = shared_data.get_data("Euclid_radius", 0.5), 
                                                          step = 0.5, start = 1, end = 100, max_width = 200, max_height = 40,
                                                          sizing_mode="stretch_both")
        self.link_to_cutout_checkbox = pn.widgets.Checkbox(name = "Use radius from Euclid cutout",  value = False, align = "center")
        self.max_separation_input.disabled = (self.chosen_mode == "Use TargetId")
        self.link_to_cutout_checkbox.disabled = (self.chosen_mode == "Use TargetId")

        self.plot_lines_checkbox = pn.widgets.Checkbox(name = "Plot Emission/Absorption Lines positions",  value = not self._is_euclid_spec, align = "center")
        self.plot_lines_checkbox.disabled = self._is_euclid_spec
        
        self.redshift_input = pn.widgets.FloatInput(name = "Assign Redshift (Same for all Sources)", start = 0.0, end = 15, 
                                                    max_width = 200, max_height = 40, sizing_mode="stretch_both")
        self.query_redshift_button  = pn.widgets.Button(name = "Query Redshift", align = "center", button_type = "primary",
                                                       max_width = 200, max_height = 40, sizing_mode="stretch_both")
        self.redshift_column_selector  = pn.widgets.Select(name = "Redshift Column", align = "center", 
                                                           options = ["None"] + self.get_column_list(allowed_types=["float"]),
                                                           value = "None",
                                                         max_width = 200, max_height = 40, sizing_mode="stretch_both")
        self.redshift_input.disabled = not self._is_euclid_spec
        self.query_redshift_button.disabled = not self._is_euclid_spec
        self.redshift_column_selector.disabled = not self._is_euclid_spec

      
        self.retrieve_mode_button.param.watch(self._retrieve_mode_cb, "value")
        self.max_separation_input.param.watch(self._max_separation_input_cb, "value")
        self.link_to_cutout_checkbox.param.watch(self._link_to_cutout_cb, "value")

        
        self.plot_lines_checkbox.param.watch(self._plot_lines_cb, "value") 
        self.query_redshift_button.on_click(self._query_redshift_cb)
        self.redshift_input.param.watch(self._redshift_input_cb, "value")
        self.redshift_column_selector.param.watch(self._redshift_column_selector_cb, "value")
        

        self.plot_settings_panel = pn.Column(self.retrieve_mode_button, 
                                             pn.Row(self.max_separation_input, pn.Column(pn.Spacer(height=23), self.link_to_cutout_checkbox), align = "center"),
                                             self.plot_lines_checkbox,
                                             pn.Row(self.redshift_input,  self.query_redshift_button, 
                                             self.redshift_column_selector, align = "center"),
                                             scroll = True, visible = False)
        
    
    def _retrieve_mode_cb(self, event):
        if event.new == "Use TargetId":
            self.from_sourceId = True
            self.chosen_mode = event.new
            self.link_to_cutout_checkbox.disabled = True
            self.max_separation_input.disabled = True
            self._get_unknown_columns([f"{self.dataset}_TargetID"], change_stage=True)
        elif event.new == "Cone Search":
            self.from_sourceId = False
            self.chosen_mode = event.new
            self.link_to_cutout_checkbox.disabled = False
            self.max_separation_input.disabled = False
            self.get_layout()
    
    def _max_separation_input_cb(self, event):
        if event.new is not None:
            self.max_separation = event.new
            self._run_spectrum(self.max_separation)

    def _link_to_cutout_cb(self, event):
        if event.new:
            self._subscribe_to_shared()
        else:
            shared_data.unsubscribe(self.panel_id, "Euclid_radius")
    
    def _plot_lines_cb(self, event):
        if self.spectrum_object.spectra is not None:
            self._update_plot()
    
    def _redshift_input_cb(self, event):
        redshift = event.new
        if redshift is not None:
            spectype = "galaxy" if redshift > 0 else "star"
            self.spectrum_object._update_info_spectra("spectype", spectype)
            self.spectrum_object._update_info_spectra("redshift", redshift)
            self.plot_lines_checkbox.disabled = False
            if self.plot_lines_checkbox.value:
                self._update_plot()

    def _query_redshift_cb(self, event):
        if self.spectrum_object.spectra is not None:
            self.query_redshift_button.name = "Query Redshift [Running...]"
            self.spectrum_object.query_specz_table(verbose = True)
            self.spectrum_object.update_info_from_query()
            self.plot_lines_checkbox.disabled = False
            if self.plot_lines_checkbox.value:
                self._update_plot()
        self.query_redshift_button.name = "Query Redshift"
    
    def _redshift_column_selector_cb(self, event):
        column = event.new
        if column == "None":
            return
        redshift_value = self.get_value_from_df(column)
        if redshift_value is not None:
            self.redshift_input.value = redshift_value
    
    def _update_plot(self):
        plot_model = False if self._is_euclid_spec else True
        plot_lines = "class" if self.plot_lines_checkbox.value else False
        kwargs = {"aspect" : 3.8 if self.spectrum_object.available_spectra > 1 else 3.17, "responsive" : True}
        plot = self.spectrum_object.plot_all_spectra_hv(plot_model = plot_model, plot_lines = plot_lines,
                                                                **kwargs)
        self.figure.objects = [plot]

    
    def _update_max_separation(self, new_separation):
         self.max_separation_input.value = new_separation

    
    @param.depends("stage")                        
    def mypanel(self):
        if self.stage == "columns_selection":
            return self.columns_selection_panel(self.unknown_columns, allowed_types = ["int"],
                                                info_text= "## Select column with TargetID")
        else:
            return self.plot_panel()


class SEDPlotClass(CustomPlotClass):
    """A class used to plot the Broadband SED of the selected source. At the moment all fluxes should 
    have the same units"""

    
    available_stages = ["filters_selection", "columns_selection",
                       "error_columns_selection", "units_selection", "plot"]

    stage = param.ObjectSelector(default = available_stages[0], objects=available_stages)

    def __init__(self, data, src, close_button, extra_features):
        super().__init__(data, src, close_button, extra_features)
        self._src_callback = self._change_source_cb
        self.src.on_change("data", self._src_callback)
        self.conversion_dictionary = {"AB magnitudes" : lambda f, e : self.mag_to_flux(f,e),
                                      "milliJy" : lambda f, e : (f * 1000, e * 1000),
                                       "microJy" : lambda f, e : (f,e),
                                       "nanoJy"  : lambda f, e : (f / 1000, e / 1000),
                                       "cgs (erg/s/Hz/cm2)" : lambda f, e : (f * 1e23, e * 1e23)
                                     }

    def _change_source_cb(self, attr, old, new):
        if self.stage == "plot":
            self._update_plot(new)
    
    
    def filters_selection_panel(self):
        self.filter_data = self.read_photometric_file()
        self._initialize_checkboxes()
        self.checkbox_pane = pn.Column(*self.checkbox_group)
        self._initialize_add_band()
        submit_button = pn.widgets.Button(name='Confirm', button_type='primary', max_height=120)
        submit_button.on_click(self._submit_button_cb)

        return pn.Card(pn.Row(pn.Column("## Available Bands", self.checkbox_pane, scroll = True),
                        pn.Column(self.add_band_button, self.add_band_pane, scroll = True)),
                        header = pn.Row(pn.Spacer(width=25), self.close_button, submit_button),
                        sizing_mode="stretch_both",  scroll = True, collapsible = False, min_height = 300 )


    @staticmethod
    def read_photometric_file(extra_path = ""):
        filepath = os.path.join(extra_path, "data/sed_data/photometric_bands.json")
        with open(filepath, 'r') as f:
            filter_data = json.load(f)
        return filter_data
    
    def create_checkbox_tooltip(self, band, name, wavlen, fwhm, value = False):
        checkbox = pn.widgets.Checkbox(name=band, value=value, width=150)
        tooltip_text = f"{name},  (Wavelength: {wavlen} Å, FWHM: {fwhm} Å)"
        tooltip_icon = pn.widgets.TooltipIcon(value=tooltip_text, margin=(0, 0, 0, 0))
        return checkbox, tooltip_icon

    def _initialize_checkboxes(self):
        self.checkboxes = {} #dictionary storing all the checkboxs available
        self.checkbox_group = [] #List storing all pairs of checkbox-tooltip
        for band, info in self.filter_data.items():
            checkbox, tooltip_icon = self.create_checkbox_tooltip(band, info["name"], 
                                                                  info["wavelength"], info["FWHM"],
                                                                   value = band in config.settings["bands_used_SED"])
            self.checkbox_group.append(pn.Row(checkbox, tooltip_icon, align='center'))
            self.checkboxes[band] = checkbox

    def _initialize_add_band(self):
        self.short_name_input = pn.widgets.TextInput(name = "Short Filter Name")
        self.full_name_input = pn.widgets.TextInput(name = "Full Filter Name", value = "")
        self.wavelength_input = pn.widgets.FloatInput(name = "Effective Wavelength [Å]")
        self.fwhm_input = pn.widgets.FloatInput(name= "FWHM [Å]", value = 0)
        confirm_button = pn.widgets.Button(name="Confirm", button_type="primary")

        self.add_band_pane = pn.Column(self.short_name_input, self.full_name_input, 
                                       self.wavelength_input, self.fwhm_input, confirm_button, visible=False)
        self.add_band_button = pn.widgets.Button(name="Add Band ▾", button_type="success", max_height = 50)

        self.add_band_button.on_click(self._toggle_add_band_cb)
        confirm_button.on_click(self.add_new_band)

    def update_photometric_file(self, new_band, name, wavlen, fwhm):
        self.filter_data[new_band] = {"name" : name, 
                                      "wavelength" : wavlen,
                                      "FWHM" : fwhm}
        
    def _toggle_add_band_cb(self, event):
        self.add_band_pane.visible = not self.add_band_pane.visible
        self.add_band_button.name = "Add Band ▴" if self.add_band_pane.visible else "Add Band ▾"
        
    
    def add_new_band(self, event):
        try:
            new_band = self.short_name_input.value.strip()
        except AttributeError:
            self.short_name_input.value = "Insert a valid name (no empty string)"
            return
        try:
            name = self.full_name_input.value.strip()
        except AttributeError:
            self.full_name_input.value = "Insert a valid name (no empty string)"
            return
        wavlen = self.wavelength_input.value
        if wavlen <=0:
            print("Insert a valid effective wavelength (>0)")
            return 
        fwhm = self.fwhm_input.value
        if fwhm  < 0:
            print("Insert a valid full width half maximum for the filter (>=0)")
            return 

        if new_band:
            if new_band not in self.checkboxes:
                checkbox, tooltip_icon = self.create_checkbox_tooltip(new_band, name, wavlen, fwhm)
                self.checkbox_group.append(pn.Row(checkbox, tooltip_icon, align='center'))
                self.checkboxes[new_band] = checkbox
                self.checkbox_pane.objects = [*self.checkbox_group]
            self.update_photometric_file(new_band, name, wavlen, fwhm)

        self.short_name_input.value = ""
        self.full_name_input = ""
        self.wavelength_input.value = 0.0
        self.fwhm_input.value = 0.0
        self.add_band_pane.visible = False


    def _submit_button_cb(self, event):
        if self.stage == self.available_stages[0]:
            self._filters_selection_continue_cb()
        elif self.stage in (self.available_stages[1], self.available_stages[2]):
            self._columns_selection_continue_cb()
        elif self.stage ==  self.available_stages[3]:
            self._units_selection_continue_cb()
        else:
            self.stage = "plot"

    def _filters_selection_continue_cb(self):
        self.bands_to_plot = [band for band in self.checkboxes.keys() if  self.checkboxes[band].value]
        self.error_bands_to_plot = [f"err_{band}" for band in self.bands_to_plot]
        self._get_unknown_columns(self.bands_to_plot+self.error_bands_to_plot, ready_stage=self.available_stages[3],
                                 change_stage=True)


    def _columns_selection_continue_cb(self):
        
        if "SED_bands" not in config.settings:
            config.settings["SED_bands"] = {}
        for col, widget in self.select_widgets.items():
            selected_value = widget.value
            print(f"{col} --> {selected_value}")
            config.settings["SED_bands"][col] = selected_value

        current_idx = self.available_stages.index(self.stage)
        print(f"Moving to stage {current_idx + 1}, i.e. {self.available_stages[current_idx + 1]}")
        self.stage = self.available_stages[current_idx + 1]

    def _units_selection_continue_cb(self):
        config.settings["band_units"] = {band: widget.value for band, widget in self.select_widgets.items()}
        self.stage = "plot"

    def get_filter_information(self):
        self.wavlen = np.array([self.filter_data[band]["wavelength"] for band in self.bands_to_plot]).flatten()
        self.fwhm = np.array([self.filter_data[band]["FWHM"] for band in self.bands_to_plot]).flatten()
        self.fwhm = np.where(np.logical_and(np.isfinite(self.fwhm), self.fwhm>0), self.fwhm, np.nan) #avoid potential issues

    def units_selection_panel(self, columns_to_select):
        available_units = list(self.conversion_dictionary.keys())
        settings_grid = self._get_selection_widgets_grid(columns_to_select, options = available_units)
        
        submit_button = pn.widgets.Button(name='Confirm', button_type='primary', max_height=120)
        submit_button.on_click(self._submit_button_cb)
        skip_button = pn.widgets.Button(name='Skip', button_type='primary', max_height=120)
        skip_button.disabled = True

        def change_all_selections(event):
            value = event.new
            for col in columns_to_select:
                self.select_widgets[col].value = value

        master_select_widget = pn.widgets.Select(name= "Apply same units to all columns", options=available_units, max_height=120, sizing_mode = "stretch_width")
        master_select_widget.param.watch(change_all_selections, "value")
         
        
        return pn.Card(pn.Column(pn.pane.Markdown("## Select the columns units", sizing_mode = "stretch_width",
                                                  margin=(15,0,15,15)),
                                master_select_widget,
                                settings_grid, scroll = True),
                                header = pn.Row(pn.Spacer(width=25), self.close_button, skip_button, submit_button),
                                sizing_mode="stretch_both", scroll=True, collapsible = False, min_height = 300 )
    


    def get_fluxes_from_selected_source(self):
        selected_source = self.get_selected_source()
        flux = selected_source[[config.settings["SED_bands"][col] for col in self.bands_to_plot]].to_numpy().flatten()
        flux_err = []
        for col in  self.error_bands_to_plot:
            try:
                flux_err.append(selected_source[config.settings["SED_bands"][col]].iloc[0])
            except KeyError:
                flux_err.append(np.nan)
        return flux, np.array(flux_err).flatten()
 
    
    def get_layout(self):
        self.get_filter_information()
        self._initialize_settings_panel()
        self.flux, self.flux_err = self.get_fluxes_from_selected_source()
        self.clean_fluxes()
        y, y_err = self.convert_to_microjy(self.flux, self.flux_err)
        self.figure.object = self.plot_SED(self.wavlen, y, y_err, self.fwhm)
        self.message_pane.visible = False
        return pn.Column(self.message_pane, self.figure, self.plot_settings_panel, scroll = True, sizing_mode = "stretch_both")

    def clean_fluxes(self):
        """Removing missing/strange fluxes """
        cleaned_flux = []
        cleaned_err = []

        for band, f, e in zip(self.bands_to_plot, self.flux, self.flux_err):
            unit = config.settings["band_units"][band]
            if unit == "AB magnitudes":
                if f > 40 or f < -40:
                   f, e = np.nan, np.nan
            else:
                if f < 0:
                    f, e = np.nan, np.nan
            cleaned_flux.append(f)
            cleaned_err.append(e)

        self.flux = np.array(cleaned_flux)
        self.flux_err = np.array(cleaned_err)
    
    @staticmethod
    def plot_SED(wavlen, flux, flux_err, fwhm, redshift=0, output_units = "fnu"):

        mask = np.logical_and(np.isfinite(wavlen), np.isfinite(flux))
        if np.sum(mask) < 1:
            return pn.pane.Markdown(f"## There are no available points to plot. All specified bands have NaN values") 
        x = wavlen[mask] / (1 + redshift)
        y = flux[mask]
        err_y = flux_err[mask]
        fwhm = fwhm[mask]
    
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
    
        scatter = hv.Scatter((x, y), kdims='wavelength', vdims="Flux").opts(
            color="red", fill_color = None, marker="o", size=9, active_tools=[])
      
        xerrbars = hv.ErrorBars((x, y, fwhm / 2, fwhm / 2), kdims= 'wavelength', vdims=["Flux", "xneg", "xpos"], horizontal = True).opts(color = "black",
                                                                                        lower_head = None, upper_head = None, active_tools =[])
                                                                                                                   
        is_upper_limit = err_y < 0
        has_larger_errors = err_y > y #These could also be considered a upper limits...
        good_measure = np.logical_and(~is_upper_limit, ~has_larger_errors)
    
        ybars = hv.ErrorBars((x[good_measure ], y[good_measure ], err_y[good_measure ], err_y[good_measure ]), kdims='wavelength', vdims=["Flux", "yneg", "ypos"]).opts(
                color="black", line_width = 1.5, active_tools =[])
        
    
        arrow_length = 0.8 * y[has_larger_errors] #all arrows have the same length in logy scale
        larger_errors = hv.ErrorBars(
            (x[has_larger_errors], y[has_larger_errors], np.full(np.sum(has_larger_errors), arrow_length), err_y[has_larger_errors]),
            kdims='wavelength', vdims=["Flux", "yneg", "ypos"]).opts(color="black",lower_head = NormalHead(size=8),
                                                                     line_width = 1.5, active_tools =[])
     
        arrow_length = 0.8* y[is_upper_limit] #all arrows have the same length in logy scale
        upper_limits = hv.ErrorBars(
            (x[is_upper_limit], y[is_upper_limit], np.full(np.sum(is_upper_limit), arrow_length), np.zeros(np.sum(is_upper_limit))),
            kdims='wavelength', vdims=["Flux", "yneg", "ypos"]).opts(color="black",lower_head = NormalHead(size=8),
                                                                     line_width = 1.5, active_tools =[])
        
    
        plot = scatter * xerrbars * ybars * upper_limits * larger_errors
        xlabel = "Rest-Frame Wavelength" if redshift > 0 else "Observed Wavelength"
        ylabel = "Flux [erg/s cm-2]" if output_units == "nufnu" else "Flux density [μJy]" 
        hooks = [] if output_units == "nufnu" else [SEDPlotClass.add_magnitude_axis]

        return plot.opts(
            xlabel=xlabel,
            logx = True, logy = True, 
            xlim = (xmin/2, xmax*2),
            ylim = (ymin/3, ymax*3),
            ylabel = ylabel, show_grid=True,
            active_tools =[],
            hooks = hooks
        )
    
    @staticmethod
    def add_magnitude_axis(plot, element):
        """Bokeh hook to add secondary Y-axis with magnitude scale"""
        fig = plot.state
        y_start, y_end = fig.y_range.start, fig.y_range.end
        mag_start, _ = SEDPlotClass.flux_to_mag(y_start, 0)
        mag_end, _  =  SEDPlotClass.flux_to_mag(y_end, 0)
        
        #Note mag_start and end are reversed compared to fluxes
        fig.extra_y_ranges = {"mag": Range1d(start=mag_start, end=mag_end)}
        mag_axis = LinearAxis(y_range_name="mag", axis_label="AB Magnitude",
                              major_label_text_color="black", axis_label_text_color="black")
        fig.add_layout(mag_axis, 'right')


    @staticmethod
    def mag_to_flux(mag, err_mag):
        """Converts AB magnitudes in flux densities in microJy"""    
        flux = 10**((23.9 - mag)/2.5)
        err_flux = flux * err_mag * np.log(10)
        return flux, err_flux
    
    @staticmethod
    def flux_to_mag(flux, err_flux):
        """Converts fluxes in microJansky to AB magnitudes"""
        mag = -2.5*np.log10(flux) + 23.9
        err_mag = (err_flux/flux)/np.log(10)
        return mag, err_mag
    
    
    def convert_to_microjy(self, flux, err_flux):
        flux_converted =[]
        err_converted = []
        for band, f ,e in zip(self.bands_to_plot, flux, err_flux):
            unit = config.settings["band_units"][band]
            try:
                fc, ec = self.conversion_dictionary[unit](f, e)
                flux_converted.append(fc)
                err_converted.append(ec)
            except KeyError:
                print(f"I cannot find this unit: {unit}")
                flux_converted.append(np.nan)
                err_converted.append(np.nan)

        return np.array(flux_converted), np.array(err_converted)
    
    def convert_to_output_units(self, wav, flux, flux_err, output_units):
        """wav in angstrom, flux in microJy"""
        #TODO do it more genereal
        if output_units == "nufnu": 
            flux, flux_err = flux /1e23, flux_err/ 1e23
            flux, flux_err = flux*2.998e18/wav, flux_err*2.998e18/wav
        return flux, flux_err
    

    def _initialize_settings_panel(self):     
        output_units = {"microJy" : "fnu", "erg/s/cm2" : "nufnu"}  #first one should be always microJy                                                                           
        self.unit_selector = pn.widgets.Select(name = "Output Units", options = output_units, max_width = 200, max_height = 40, 
                                               sizing_mode="stretch_both")
        self.unit_selector.param.watch(self._update_plot, "value")                                                           
        self.plot_settings_panel = pn.Column(self.unit_selector, visible = False)
                                                                                                                                                   

    def _update_plot(self, event):
        self.flux, self.flux_err = self.get_fluxes_from_selected_source()
        self.clean_fluxes()
        y, y_err = self.convert_to_microjy(self.flux, self.flux_err)
        y, y_err = self.convert_to_output_units(self.wavlen, y, y_err, output_units =self.unit_selector.value)
        self.figure.object = self.plot_SED(self.wavlen, y, y_err, self.fwhm, output_units =self.unit_selector.value)
        self.message_pane.visible = False
        
    
    def plot_panel(self):
        self.layout = self.get_layout()
        return pn.Card(self.layout, header = pn.Row(pn.Spacer(width=25,),self.close_button, self.plot_settings_button),
                       collapsible = False, sizing_mode="stretch_both", min_height =450,)  

    @param.depends("stage")
    def mypanel(self):
        if self.stage == self.available_stages[0]:
            return self.filters_selection_panel()
        elif self.stage == self.available_stages[1]:
            columns_to_select = [i for i in self.bands_to_plot if i in self.unknown_columns]
            if len(columns_to_select) > 0:
                return self.columns_selection_panel(columns_to_select, skippable=False, allowed_types = ["float"],
                                            info_text= "## Select columns with flux values")
            else:
                self.stage = self.available_stages[2]
        
        elif self.stage == self.available_stages[2]:
            columns_to_select = [i for i in self.error_bands_to_plot if i in self.unknown_columns]
            if len(columns_to_select) > 0:
                return self.columns_selection_panel(columns_to_select, skippable=True, allowed_types = ["float"],
                                        info_text= "## Select columns with flux error values")
            else:
                self.stage = self.available_stages[3]

        elif self.stage == self.available_stages[3]:
            return self.units_selection_panel(self.bands_to_plot)
        
        else:
            return self.plot_panel()


class RadioClass(CustomPlotClass):
    
    def __init__(self, data, src, close_button, extra_features, dataset):
        super().__init__(data, src, close_button, extra_features)
        self._src_callback = self._change_source_cb
        self.src.on_change("data", self._src_callback)
        self.dataset = dataset
        self._initialize_source()
        self.radius = 20

    def _initialize_source(self):
        self.ra, self.dec = self.get_ra_dec()
        if (self.ra is None) or (self.dec is None):
            self.message_pane.visible = True
            self.message_pane.object = ["## Missing Ra and Dec"]   

    def _change_source_cb(self, attr, old, new):
        self._initialize_source()
        self._run_radio(radius = self.radius)

    def get_layout(self):
        self._initialise_widgets()
        self._run_radio(radius = self.radius)
        return  pn.Column(self.message_pane, self.figure, self.plot_settings_panel, 
                          scroll = True, sizing_mode = "stretch_both")

    def _initialise_widgets(self):

        self.radius_input = pn.widgets.FloatInput(name = "Radius [arcsec]", value = self.radius, 
                                                  step = 1, start = 1, end = 100, max_width = 200,
                                                  sizing_mode="stretch_both", max_height =30)
        self.radius_input.param.watch(self._update_radius, "value")
        
        self.plot_settings_panel = pn.Column(self.radius_input, 
                                             scroll = True, visible = False)
        
    def _update_radius(self, event):
        if event.new: 
            self.radius = event.new
            self._run_radio(radius = self.radius)
        else:
            print("Input a valid value for radius")

    def _run_radio(self, radius = 20):
        self.message_pane.visible = True
        shared_data.publish(self.panel_id, f"Radio_running", True)

        def callback(future_obj = None):
            shared_data.publish(self.panel_id, "Radio_running", False)
            print("I am calling the radio callback ")
            result = future_obj.result() 
            if result is None:
                self.message_pane.object = f"## {self.dataset} cutout query failed"
                self.message_pane.visible = True #probably already visible
            elif result is not None:
                print("i am obtaining the image")
                print(result.shape)
                self.figure.object = self.get_radio_figure(result)
                self.message_pane.visible = False

        if self.dataset == "VLASS":
            print("I am running VLASS")
            self.run_multithread(VLASS_cutout, 
                             func_kwargs = {"ra" : self.ra, "dec" : self.dec,
                                            "radius" : radius},
                             callback=callback)
        elif self.dataset == "LoTSS":
            self.run_multithread(LoTSS_cutout, 
                             func_kwargs = {"ra" : self.ra, "dec" : self.dec,
                                            "radius" : self.radius},
                             callback=callback)

    
    def get_radio_figure(self, data):
        self.image_height, self.image_width,  = data.shape[:2]
        bounds = (0, 0, self.image_height, self.image_width)
        image = hv.Image(data[::-1,...], bounds=bounds).opts(
                                         active_tools =[], toolbar=None,
                                         padding = 0,
                                         border = 0,
                                         framewise = True,
                                         xaxis=None, 
                                         yaxis=None,
                                         )
        return image

    
    