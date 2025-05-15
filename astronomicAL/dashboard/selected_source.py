from functools import partial
import astronomicAL.config as config
import numpy as np
import pandas as pd
import panel as pn
import time
import threading



import matplotlib.pyplot as plt
from matplotlib.figure import Figure
#from astronomicAL.dashboard.astro_cutouts import euclid_cutouts_class, DESI_spectra_class, spectrum_class
from astropy import units as u
import time
#pn.extension()

import holoviews as hv 

#import io
#from PIL import Image


class SelectedSourceDashboard:
    """A Dashboard used for showing detailed information about a source.

    Optical and Radio images of the source are provided free when the user has
    Right Ascension (RA) and Declination (Dec) information. The user can specify
    what extra information they want to display when a source is selected.

    Parameters
    ----------
    src : ColumnDataSource
        The shared data source which holds the current selected source.

    Attributes
    ----------
    df : DataFrame
        The shared dataframe which holds all the data.
    row : Panel Row
        The panel is housed in a row which can then be rendered by the
        parent Dashboard.
    selected_history : List of str
        List of source ids that have been selected.
    src : ColumnDataSource
        The shared data source which holds the current selected source.

    """


    def __init__(self, src, close_button):

        self.df = config.main_df
        self.src = src
        self.src.on_change("data", self._panel_cb)

        self.close_button = close_button

        print("reloading information")

        self.row = pn.Row(pn.pane.Str("loading"))

        self.selected_history = []

        self._search_status = ""

        self.desi_pane = pn.pane.Matplotlib(
                         alt_text="Image Unavailable",
                         min_width=300,
                         min_height=300,
                         sizing_mode="scale_both",
                         interactive = False)
        
        self.euclid_pane = pn.pane.Matplotlib(
                         alt_text="Image Unavailable",
                         min_width=300,
                         min_height=300,
                         sizing_mode="scale_both",
                         interactive = False)
        
        
        
        self.image_pane = self.euclid_pane
        self._get_ra_dec()
        self.radius = 5.0
        
        desi_thread = threading.Thread(target=self.run_desi_init)
        desi_thread.start()


        self.get_euclid_object()
        self.get_euclid_cutout()
        self.get_euclid_figure()

        desi_thread.join()
        self.get_desi_figure()
        self.add_desi_coordinates()


        self._initialise_radius_scaling_widgets()
        self._add_selected_info()
        self._remove_loading_screen()
        self.panel()

    
    def _add_loading_screen(self):
        return None
        self.image_pane = self.loading_pane
        self.image_pane.object = "# Loading image"
        
    
    def _remove_loading_screen(self):
        return None 
        self.image_pane = self.euclid_pane
        self.panel()
    
    
    
    def _add_selected_info(self):
        self.contents = "Selected Source"
        self.search_id = pn.widgets.TextInput(
            name="Select ID",
            placeholder="Select a source by ID",
            max_height=50,
        )
        self.search_id.param.watch(self._change_selected, "value")

        self.panel()

    def _change_selected(self, event):

        if event.new == "":
            self._search_status = ""
            self.panel()
            return

        self._search_status = "Searching..."

        if event.new not in list(self.df[config.settings["id_col"]].values):
            self._search_status = "ID not found in dataset"
            self.panel()
            return

        selected_source = self.df[self.df[config.settings["id_col"]] == event.new]

        selected_dict = selected_source.set_index(config.settings["id_col"]).to_dict(
            "list"
        )
        selected_dict[config.settings["id_col"]] = [event.new]
        self.src.data = selected_dict

        self.panel()

    def _panel_cb(self, attr, old, new):
        self._get_ra_dec()
        self.radius = 5
        desi_thread = threading.Thread(target=self.run_desi)
        desi_thread.start()
        self.get_euclid_object()
        if self.radius_input.value == self.radius:
                self.get_euclid_cutout()
                self.get_euclid_figure()
                self.contrast_scaler.value = (0,1)
        else:
            self.radius_input.value = self.radius  ###this already calls self.get_euclid_cutout and self.get_euclid_figure()
                                                    ####and self.contrast_scaler.value = (0,255)
        desi_thread.join() 
        #self.desi_object.set_ra_dec(self.ra, self.dec)
        #self.get_desi_spectrum()
        self.add_desi_coordinates()
        self.get_desi_figure()
        self._remove_loading_screen()
        self.panel()

    def _check_valid_selected(self):
        selected = False

        if config.settings["id_col"] in list(self.src.data.keys()):
            if len(self.src.data[config.settings["id_col"]]) > 0:
                if self.src.data[config.settings["id_col"]][0] in list(
                    self.df[config.settings["id_col"]].values
                ):
                    selected = True

        return selected

    def _add_selected_to_history(self):

        add_source_to_list = True

        if len(self.selected_history) > 0:
            selected_id = self.src.data[config.settings["id_col"]][0]
            top_of_history = self.selected_history[0]
            if selected_id == top_of_history:
                add_source_to_list = False
            elif selected_id == "":
                add_source_to_list = False

        if add_source_to_list:
            self.selected_history = [
                self.src.data[config.settings["id_col"]][0]
            ] + self.selected_history

    def _deselect_source_cb(self, event):
        self.deselect_button.disabled = True
        self.deselect_button.name = "Deselecting..."
        print("deselecting...")
        self.empty_selected()
        print("deselected...")
        self.search_id.value = ""
        self._search_status = ""
        print("blank...")
        self.deselect_button.disabled = False
        self.deselect_button.name = "Deselect"

    def empty_selected(self):
        """Deselect sources by emptying `src.data`.

        Returns
        -------
        None

        """
        empty = {}
        for key in list(self.src.data.keys()):
            empty[key] = []

        self.src.data = empty


    def _update_image(self): 
        try:
             self.euclid_pane.object = self.euclid_fig
        except Exception as e:         #too generic
            print("Euclid image unavailable")
            print(e)
        try:
             self.desi_pane.object = self.desi_fig
        except Exception as e:         #too generic
            print("DESI/SDSS spectrum unavailable")
            print(e)
        return None
    

    def check_required_column(self, column):
        """Check `df` has the required column.

        Parameters
        ----------
        column : str
            Check whether this column is in `df`

        Returns
        -------
        has_required : bool
            Whether the column is in `df`.

        """
        has_required = False
        if column in list(self.df.columns):
            has_required = True

        return has_required
    
    def _initialise_radius_scaling_widgets(self):
        self.radius_input = pn.widgets.FloatInput(name = "Radius [arcsec]", value = self.radius,
                                                  step = 0.5, start = 1, end = 100, height=50, width =100,
                                                  sizing_mode="fixed")
        self.radius_input.param.watch(self.update_radius, "value")

        
        self.stretching_input = pn.widgets.MenuButton(name = "Stretching function", 
                                                     items =  [('Linear', 'Linear'), ('Sqrt', 'Sqrt'), ('Log', 'Log'), ('Asinh', 'Asinh'), ('PowerLaw', 'PowerLaw')],
                                                     button_type = "primary", width = 250,
                                                     sizing_mode="fixed")
        
        self.stretching_input.on_click(self.update_stretching)

        
        self.contrast_scaler = pn.widgets.RangeSlider(name = "Image scaling", 
                                                               start = 0, end = 1,value = (0,1), 
                                                               step = 0.004, height=70, width =400,
                                                               sizing_mode="fixed")
        
        self.contrast_scaler.param.watch(self.update_intensity_scaling, "value")


    
    def update_radius(self, event):
        self.radius = event.new
        self._add_loading_screen()
        self.get_euclid_cutout()
        self.contrast_scaler.value = (0,1)
        print(self.image_pane)
        self.get_euclid_figure()
        self.add_desi_coordinates()
        self._remove_loading_screen()
        self._update_image()

    @staticmethod
    def change_intensity_range(image, low, high):
        image = np.clip(image, low, high)
        image = (image-low)/(high-low)
        return np.clip(image, 0,1)

    def update_intensity_scaling(self, event):
        low, high = event.new
        scaled_image = self.change_intensity_range(self.euclid_object.reprojected_data["stacked"], 
                                                   low, high)
        self.euclid_image.set_data(scaled_image)
        self.euclid_fig.canvas.draw()
        self._update_image()
    
    def update_stretching(self, event):
        stretch = event.new
        self.euclid_object.stack_cutouts(stretch = stretch)
        self.euclid_image.set_data(self.euclid_object.reprojected_data["stacked"])
        self.euclid_fig.canvas.draw()
        self._update_image()


    
    def get_euclid_object(self):
        self.euclid_object = euclid_cutouts_class(self.ra, self.dec, 
                            euclid_filters= ["VIS", "NIR_Y", "NIR_H"])
        
        self.euclid_object.get_cone(verbose = True)
        return None
    
    def get_euclid_cutout(self, stretch = "Linear", verbose = True):
        if not hasattr(self, "euclid_object"):
            self.get_euclid_object()
        if len(self.euclid_object.cone_results) > 2:
            self.euclid_object.get_cutouts(radius = self.radius, verbose = verbose)
            self.euclid_object.read_cutouts()
            self.euclid_object.reproject_cutouts(reference = "VIS")
            self.euclid_object.stack_cutouts(stretch = stretch)
        else:
            print("No sources in Euclid dataset with those coordinates")
            return None 
        
    
        
    def get_plot_scale(self):
        bar_length_arcsecond = self.bar_length_pixels * self.euclid_object.arcsec_per_pix["stacked"]
        return bar_length_arcsecond

    
    def get_euclid_figure(self, show_scale = True, figsize = (8,8)):
        
        if hasattr(self, "euclid_image"):
            self.euclid_image.remove()
            self.euclid_image = self.ax.imshow(self.euclid_object.reprojected_data["stacked"], origin = "lower")


        self.euclid_fig = plt.Figure(figsize=figsize)
        self.ax = self.euclid_fig.add_subplot(1,1,1)
        self.ax.set_axis_off()
        self.euclid_image = self.ax.imshow(self.euclid_object.reprojected_data["stacked"], origin = "lower")
        
        if show_scale:
            image_height, image_width = self.euclid_image.get_size()
            self.bar_length_pixels = image_width * 0.2  #always show a bar 1/5 of the plot 
            x0, y0 = 0.1*image_width, 0.1*image_height
            x1 = x0 + self.bar_length_pixels
            self.ax.plot([x0, x1], [y0, y0], color='red', lw=3)
            self.scale_text =  self.ax.text((x0 + x1) / 2, y0 + y0/2, f'{self.get_plot_scale():.1f}"',
                                        color='red', ha='center', va='bottom', fontsize=14, fontweight='bold')
        

    def get_desi_object(self):
        #This initializes the SparcClient which takes time so better to call it once
        self.desi_object = DESI_spectra_class(self.ra, self.dec, 
                                              distance_radius=1)
    
    def get_desi_spectrum(self):
        self.desi_object.query_main_table(verbose = True)
        self.desi_object.query_spectra(verbose = True, max_sep=1)
        if self.desi_object.spectrum:
           self.desi_spectrum = spectrum_class(self.desi_object.spectrum.wavelength,
                                               self.desi_object.spectrum.flux,
                                               self.desi_object.spectrum.redshift)
           self.desi_spectrum.get_smoothed_spectrum(kernel = "Box1dkernel", window = 10)
        
    
    def run_desi(self):
        """Just a wrapper for threading"""
        self.desi_object.set_ra_dec(self.ra, self.dec)
        self.get_desi_spectrum()

    def run_desi_init(self):
        """Just a wrapper for threading, to use in initialization"""
        self.get_desi_object()
        self.get_desi_spectrum()

    def get_desi_figure(self):
        if hasattr(self, "desi_fig"):
            pass
        if not self.desi_object.spectrum:
            self.desi_fig = None
            return None 
        self.desi_fig, self.desi_ax= plt.subplots(figsize = (15,5))
        self.desi_spectrum.plot_spectrum(self.desi_ax)
        self.desi_ax.plot(self.desi_object.spectrum.wavelength, self.desi_object.spectrum.model, lw =1, c ='r')
        self.desi_ax.text(0, 1.02, f"Dataset = {self.desi_object.spectrum.data_release}", fontsize = 15, transform = self.desi_ax.transAxes)
        self.desi_ax.text(0.33, 1.02, f"SpecType = {self.desi_object.spectrum.spectype}", fontsize = 15, transform = self.desi_ax.transAxes)
        self.desi_ax.text(0.66, 1.02, f"z = {np.round(self.desi_object.spectrum.redshift,4)}", fontsize = 15,transform = self.desi_ax.transAxes)
        
    def add_desi_coordinates(self):
        if self.desi_object.spectrum and hasattr(self, "euclid_image"):
            xpos, ypos = self.euclid_object.wcs["stacked"].world_to_pixel(self.desi_object.coordinates_spectrum)
            self.ax.scatter(xpos, ypos, c = 'r', marker = '+', s=500)
       

    def get_desi_hv(self):
        if hasattr(self, "desi_hv"):
            pass
        if not self.desi_object.spectrum:
            self.desi_hv = None
            return None 
        self.desi_hv = self.desi_spectrum.plot_spectrum_hv()
        self.desi_hv = hv.Curve((self.desi_object.spectrum.wavelength, self.desi_object.spectrum.model))
        

    def get_info_pane(self):
        extra_data_list = [
        ["Source ID", self.src.data[config.settings["id_col"]][0]]]
        for i, col in enumerate(config.settings["extra_info_cols"]):
            extra_data_list.append([col, self.src.data[f"{col}"][0]])
            extra_data_df = pd.DataFrame(extra_data_list, columns=["Column", "Value"])
            extra_data_pn = pn.pane.DataFrame(
                            extra_data_df, index=False, sizing_mode="stretch_both")
        return extra_data_pn
           
    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which can then be rendered by the
            parent Dashboard.

        """
        # CHANGED :: Remove need to rerender with increases + decreases

        selected = self._check_valid_selected()

        if selected:

            self._add_selected_to_history()

            self.deselect_button = pn.widgets.Button(name="Deselect")
            self.deselect_button.on_click(self._deselect_source_cb)

            self.row[0] = self._organize_panel()
            self._update_image()

        else:
            self.row[0] = pn.Card(
                pn.Column(
                    self.search_id,
                    self._search_status,
                    pn.Row(
                        pn.widgets.DataFrame(
                            pd.DataFrame(
                                self.selected_history, columns=["Selected IDs"]
                            ),
                            show_index=False,
                        ),
                        max_width=300,
                    ),
                ),
                header=pn.Row(self.close_button, max_width=300),
            )

        return self.row


    def _get_ra_dec(self):
        if self.check_required_column("ra_dec"):
            ra_dec = self.src.data["ra_dec"][0]
            self.ra = float(ra_dec[: ra_dec.index(",")])
            self.dec = float(ra_dec[ra_dec.index(",") + 1 :])
        else:
            print("No ra and dec available for this source")
            self.ra, self.dec = None, None

    def _organize_panel(self):
        card = pn.Card(
                pn.Column(
                    pn.Row(
                            pn.Column(self.image_pane,
                                      pn.Row(self.radius_input, self.stretching_input, align = "center"),
                                      self.contrast_scaler),
                            pn.Row(self.get_info_pane(), max_height=250, max_width=300),
                          ),
                    self.desi_pane
                        ),
                collapsible=False,
                header=pn.Row(self.close_button, self.deselect_button, max_width=300),
                )
        return card






