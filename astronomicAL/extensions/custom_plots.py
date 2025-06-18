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
import threading
import time
import param
import uuid
import concurrent.futures 
from bokeh.document import without_document_lock
from astronomicAL.extensions.shared_data import shared_data
from astronomicAL.extensions.astro_data_utility import DESISpectraClass, EuclidCutoutsClass, EuclidSpectraClass
import matplotlib.pyplot as plt


class CustomPlotClass(param.Parameterized):

    stage = param.ObjectSelector(default="column_selection", objects=["column_selection", "plot"])
    
    def __init__(self, data, src, extra_features, close_button):
        super().__init__()
        self.df = data
        self.src = src
        self.extra_features = extra_features
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.close_button = close_button
        self.panel_id = str(uuid.uuid4()) 
        self.get_unknown_columns()
        self.figure = pn.pane.HoloViews(sizing_mode="stretch_both", min_height = 400)
        self.loading_pane = pn.pane.Markdown("## Loading...", sizing_mode="stretch_both", max_height = 30)

    def _confirm_button_cb(self, event):
        for col, widget in self.select_widgets.items():
            print(col, widget.value)
        self.stage = "plot"

    def get_selected_source(self):
        if self.src is None:
            return None
        cols = list(self.df.columns)
        if len(self.src.data[cols[0]]) == 1:
            return pd.DataFrame(self.src.data, columns=cols, index=[0])
        return None
    
    def get_ra_dec(self, err_message = "No ra and dec available for this source"):
        selected_source = self.get_selected_source()
        if (selected_source is not None) and self.check_required_column("ra_dec"):
            ra_dec = selected_source["ra_dec"][0]
            ra = float(ra_dec[: ra_dec.index(",")])
            dec = float(ra_dec[ra_dec.index(",") + 1 :])
        else:
            print(err_message)
            ra, dec = None, None
        return ra, dec

    def check_required_column(self, column):
        return column in list(self.df.columns)


    def column_selection_panel(self):
        settings_grid = pn.GridBox(ncols=3, sizing_mode = "stretch_width", scroll = True)  
        options = list(config.main_df.columns)
        submit_button = pn.widgets.Button(name='Confirm', button_type='primary', max_height=120)
        submit_button.on_click(self._confirm_button_cb)
        for col in self.unknown_columns:
            select_widget = pn.widgets.Select(name= col, options=options, max_height=120, sizing_mode = "stretch_width")
            settings_grid.append(select_widget)
            self.select_widgets[col] = select_widget
        return pn.Card(settings_grid, header = pn.Row(pn.Spacer(width=25), self.close_button, submit_button),
                                sizing_mode="stretch_both", scroll=True, collapsible = False, min_height = 300 )
    
    def get_unknown_columns(self):
        current_cols = config.main_df.columns
        self.unknown_columns = []
        for col in self.extra_features:
            if col not in list(config.settings.keys()):
                print(f"{col} not in config")
                if col not in current_cols:
                    print(f"{col} not in df")
                    self.unknown_columns.append(col)
                else:
                    config.settings[col] = col
        if len(self.unknown_columns) > 0:
            self.select_widgets = {}
        else:
            self.stage = "plot"
    
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
    
    def cleanup_subscriptions(self):
        """Removes subscriptions to the shared data
           Used in a previous version"""
        shared_data.unsubscribe_panel(self.panel_id)
        print(f"CustomPlot {self.panel_id} removed from subscriptions")
    
    def cleanup_shared_data(self):
        """Removes subscriptions and published data from the shared data"""
        shared_data.cleanup_extension_panel(self.panel_id)
        print(f"CustomPlot {self.panel_id} removed from shared data")


    def get_layout(self):
        points_input = pn.widgets.IntInput(name="Number of points", value=20, start=1, sizing_mode = "stretch_width" )
        def update_points(event):
            N = points_input.value
            self.plot_async()
        points_input.param.watch(update_points, 'value')
        self.plot(points_input.value)
        return pn.Column(self.loading_pane, self.figure, points_input, sizing_mode="stretch_both", min_height = 450,
                          styles={'background': 'lightgreen'})
    
    def plot(self, N=20):
        self.loading_pane.visible = True
        coords = [(i, np.random.random()) for i in range(N)]
        scatter = hv.Scatter(coords).opts(color='black', marker='+')
        self.figure.object = scatter
        self.loading_pane.visible = False

    def plot_panel(self):
        self.layout = self.get_layout()
        return pn.Card(self.layout, header = pn.Row(pn.Spacer(width=25,),self.close_button),
                       collapsible = False, sizing_mode="stretch_both", min_height =450,
                       styles={'background': 'lightblue'})
    
    @param.depends("stage")                        
    def mypanel(self):
        if self.stage == "column_selection":
            return self.column_selection_panel()
        else:
            return self.plot_panel()
        


class EuclidPlotClass(CustomPlotClass):
    def __init__(self, data, src, extra_features, close_button):
        super().__init__(data, src, extra_features, close_button)
        self.euclid_pane = pn.pane.HoloViews(width=400, height=400) #euclid_pane = Euclid cutout, figure = euclid_pane+overplotted_coordinates
        self.src.on_change("data", self._change_source_cb)
        self.radius = shared_data.get_data("Euclid_radius", 5.0)
        self.get_layout()
        self._subscribe_to_shared()
   
    def _change_source_cb(self, attr, old, new):
        self._initialise_euclid_object()
        self._run_euclid()

    def get_layout(self):
        self._initialise_widgets()
        self._initialise_euclid_object()
        self._run_euclid()
        return  pn.Column(self.loading_pane, pn.Row(self.figure, pn.Column(self.stretching_input, 
                                                self.radius_input,
                                                self.contrast_scaler,
                                                self.overplot_coords_widget))
        )

    def _initialise_euclid_object(self):
        self.ra, self.dec = self.get_ra_dec()
        if (self.ra is None) or (self.dec is None):
            print("no ra or dec available")
            #TODO Raise esception
        else:
            self.euclid_object = EuclidCutoutsClass(self.ra, self.dec, 
                             euclid_filters= ["VIS", "NIR_Y", "NIR_H"])
            self.overplotted_coordinates = []
            

    def _initialise_widgets(self):
        self.radius_input = pn.widgets.FloatInput(name = "Radius [arcsec]", value = self.radius,
                                                  step = 0.5, start = 1, end = 100, 
                                                  sizing_mode="scale_width")
        self.radius_input.param.watch(self._update_radius, "value")

        self.stretching_input = pn.widgets.Select(name = "Stretching function", 
                                                options=  ['Linear', 'Sqrt', 'Log', 'Asinh', 'PowerLaw'],
                                                sizing_mode = "scale_width")
        self.stretching_input.param.watch(self._update_stretching, "value")

        self.contrast_scaler = pn.widgets.RangeSlider(name = "Image scaling", 
                                                    start = 0, end = 0.996,value = (0,1), step = 0.004, 
                                                    sizing_mode = "scale_both")
        self.contrast_scaler.param.watch(self._update_intensity_scaling, "value")  

        self.overplot_coords_widget = pn.widgets.Checkbox(name = "Spectrum Coordinates")
        self.overplot_coords_widget.param.watch(self._overplot_coordinates_callback, "value")

     
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
        scaled_image = self.change_intensity_range(self.euclid_object.reprojected_data["stacked"], 
                                                   low, high)
        self.get_euclid_figure(scaled_image)

        self._update_image()
    
    def _update_stretching(self, event):
        stretch = event.new
        self.euclid_object.stack_cutouts(stretch = stretch)
        self.get_euclid_figure(self.euclid_object.reprojected_data["stacked"])
        self._update_image()

    
    def _update_image(self): 
        try:
             self.figure.object = hv.Overlay(self.euclid_fig + self.overplotted_coordinates)
             self.loading_pane.visible = False
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
                for dataset in self.stored_spectrum_coordinates:
                    print(f"overplotting coordinates for {dataset}")
                    N = len(self.stored_spectrum_coordinates[dataset]["ra"])
                    colors = plt.get_cmap("gist_rainbow", max(N,2))
                    marker = "+" if dataset == "DESI" else "*" #TODO improve
                    self.overplotted_coordinates = []
                    for i, (x, y) in enumerate(self.euclid_object.world_2_pix(ra =  self.stored_spectrum_coordinates[dataset]["ra"],
                                                                              dec = self.stored_spectrum_coordinates[dataset]["dec"],
                                                                              filtro = "stacked" )):

                        if (0 <= x < self.image_width) and (0 <= y < self.image_height):
                            self.overplotted_coordinates.append(hv.Points([(x,y)]).opts(
                                                               color = colors(i),
                                                                 marker = marker, 
                                                                 size = 20,
                                                                 ))
    

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

    def get_plot_scale(self):
        bar_length_arcsecond = self.bar_length_pixels * self.euclid_object.arcsec_per_pix["stacked"]
        return bar_length_arcsecond

    
    def get_euclid_figure(self, data, show_scale = True):
        
        self.image_height, self.image_width = data.shape[:2]
        bounds = (0, 0, self.image_height, self.image_width)

        image = hv.RGB(data[::-1,...], bounds=bounds).opts(
                                    active_tools =[], toolbar=None,
                                    padding = 0,
                                    border = 0,
                                    framewise = True,
                                    xaxis=None, 
                                    yaxis=None,
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
            
        if self.overplot_coords_widget.value:
            self._show_overplot_coordinates()

    def _run_euclid(self):
        """Wrapper for multithreading"""
        self.loading_pane.visible = True
 
        def callback(future_obj=None):
            self.overplot_coords_widget.value = False
            self.contrast_scaler.value = (0,1)
            self.get_euclid_figure(self.euclid_object.reprojected_data["stacked"])
            self._update_image()
        
        self.run_multithread(self.euclid_object.get_final_cutout,
                             {"radius" : self.radius, "stretch" : self.stretching_input.value, 
                              "reference" : "VIS", "verbose" : True}, 
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
    def __init__(self, data, src, extra_features, close_button):
        super().__init__(data, src, extra_features, close_button)
        self.figure = pn.pane.HoloViews(width=400, height=400) #euclid_pane = Euclid cutout, figure = euclid_pane+overplotted_coordinates
        self.src.on_change("data", self._change_source_cb)
        self.radius = shared_data.get_data("Euclid_radius", 5.0)
        self.get_layout()
        self._subscribe_to_shared()



    def spectrum_plot(data, selected = None, plot_instance = None, dataset = "DESI", from_sourceId = False):

        if not hasattr(plot_instance, "container"):
        plot_instance.container = pn.Column(scroll = True)
    
    def update_plot(new_radius, panel_id = None):

        selected_source = get_selected_source(data=data, selected = selected)
    
        if from_sourceId:
            if config.settings[f"{dataset}_TargetID"] in selected_source.columns:
                sourceId = int(selected_source[config.settings[f"{dataset}_TargetID"]].iloc[0])
                ra, dec = None, None
            else:
                print("Missing column with target ID")
                plot_instance.container.objects = [pn.pane.Markdown("Missing Target ID")]
                return
        elif not from_sourceId:
            ra, dec = get_ra_dec(selected_source)
            if (ra is None) or (dec is None):
                plot_instance.container.objects = [pn.pane.Markdown("Missing RA and Dec")]
                return 
            sourceId = None 
    
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_spectrum, 
                                     ra = ra, dec = dec, dataset = dataset,
                                     sourceId = sourceId,
                                     max_separation = new_radius,
                                     check_coverage = not sourceId,
                                     )
        
        spectrum_object = future.result()   
    
        if spectrum_object is not None:
            _add_coordinates_to_shared(*spectrum_object.get_coordinates(), panel_id = panel_id,  key_name = dataset)
            plot_model = True if dataset != "EuclidSpec" else False
            kwargs = {"width" : 950,  "height" : 250 if spectrum_object.available_spectra > 1 else 300}
            plot = spectrum_object.plot_all_spectra_hv(plot_model = plot_model, **kwargs)
            plot_instance.container.objects = [plot] 

        else:
            plot_instance.container.objects = [pn.pane.Markdown("No spectrum found.")]
        
        
        initial_radius = shared_data.get_data("Euclid_radius", 0.5)
        update_plot(initial_radius, panel_id = plot_instance.panel_id)
        
        if not from_sourceId:
        if not shared_data.is_subscribed(plot_instance.panel_id, "Euclid_radius"):
            shared_data.subscribe(plot_instance.panel_id, "Euclid_radius", partial(update_plot, panel_id = plot_instance.panel_id))
        return plot_instance.container
   

    def _run_spectrum(self):
        self.loading_pane.visible = True
        if self.dataset == "EuclidSpec":
            spectrum_object = EuclidSpectraClass(self.ra, self.dec, max_separation = self.max_separation,
                                             sourceId = self.sourceId)
        else:
            datasets = (["DESI-DR1"] if self.dataset == "DESI"
            else ["BOSS-DR16", "SDSS-DR16"] if self.dataset == "SDSS"
            else None)
            spectrum_object = DESISpectraClass(self.ra, self.dec, datasets = datasets ,
                                        sourceId = self.sourceId, max_separation = self.max_separation,
                                        client = shared_data.get_data("Sparcl_client", None))
        
        def callback(future_result = None):
            if future_result is not None:

            spectrum_object = future.result()   
    
        if spectrum_object is not None:
            _add_coordinates_to_shared(*spectrum_object.get_coordinates(), panel_id = panel_id,  key_name = dataset)
            plot_model = True if dataset != "EuclidSpec" else False
            kwargs = {"width" : 950,  "height" : 250 if spectrum_object.available_spectra > 1 else 300}
            plot = spectrum_object.plot_all_spectra_hv(plot_model = plot_model, **kwargs)
            plot_instance.container.objects = [plot] 

        
    
  
    if spectrum_object.spectra is not None:
        spectrum_object.get_smoothed_spectra(kernel = "Box1dkernel", 
                                             window = 5 if dataset == "EuclidSpec" else 10)
        return spectrum_object
    return None

def _add_coordinates_to_shared(ra, dec, panel_id, key_name):
    """
    ra and dec are lists
    """
    shared_data.publish(panel_id, f"{key_name}_coordinates", {"ra": ra, "dec": dec})
    return 
        
    


       