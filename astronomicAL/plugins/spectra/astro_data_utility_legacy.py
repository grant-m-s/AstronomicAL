import os
import time
from string import Template
import requests 
from requests.exceptions import ReadTimeout, ConnectTimeout
import concurrent.futures 
from io import BytesIO
import numpy as np
import pandas as pd
import warnings
from astropy import units as u
from astroquery.esa.euclid import EuclidClass
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp 
from astropy.visualization import  PowerStretch, SqrtStretch, LogStretch
from astropy.visualization import AsinhStretch, LinearStretch, AsymmetricPercentileInterval
from astropy.coordinates import SkyCoord 
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import mocpy

from sparcl.client import SparclClient 
from astronomicAL.utils.error_tracker import ErrorTracker

import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts

class EuclidCutoutsClass:
    
    def __init__(self, ra, dec, 
                 euclid_filters = ["VIS", "NIR_Y", "NIR_J", "NIR_H"],
                 save_dir = "data/cutouts",
                 context = None):

        self.context = context
        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        svc = getattr(self.context, "services", None) if self.context is not None else None
        key = "euclid.client"

        if svc is not None and svc.has(key):
            self.client = svc.get(key)
        else:
            self.client = EuclidClass(environment="PDR")
            print("Initialized EuclidClass")
            if svc is not None:
                svc.set(key, self.client)
        
        self.moc = load_moc("Euclid_Q1")
        self.error_tracker = ErrorTracker()
        self.coordinates = SkyCoord(ra, dec, unit = "degree", frame = "icrs")   
        self.euclid_filters = euclid_filters
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok = True)

    def reset_data(self, ra, dec):
        self.coordinates = SkyCoord(ra, dec, unit = "degree", frame = "icrs")
        self.error_tracker.reset()
        self._remove_source_attributes()
    
    def _remove_source_attributes(self):
        """Removes all attributes specific to a source"""
        attributes = ["cone_results", "data", "wcs", "arcsec_per_pix", "reprojected_data",
                      "plot_data", "overplot_coordinates"]
        for attribute in attributes:
            if hasattr(self, attribute):
                delattr(self, attribute)


    def change_environment(self, environment, user = None, password = None, credentials_filepath = None):
        """This function handles the change of the environment of the client EuclidClass
            Note that at the moment there is no way to automatically recognize if the login failed"""
        assert environment in ("PDR", "IDR", "OTF", "REG"), "environment must be  'PDR', 'IDR,, 'OTF', 'REG'"
        self.client =  EuclidClass(environment = environment)
        if environment != "PDR":
            if credentials_filepath is not None:
                self.client.login(user = None, password = None, credentials_file = credentials_filepath)
            else:
                self.client.login(user = user, password = password, credentials_file = None)

    
    def get_cone(self, initial_radius = 0.5*u.degree, async_job= False, verbose = True):
        """Performs a cone search and retrieves a table with information about where the image
           containing the source are stored"""
        
        try:
            tic = time.perf_counter()
            job = self.client.cone_search(self.coordinates, initial_radius, table_name = "sedm.mosaic_product", ra_column_name="ra",
                                      dec_column_name="dec", columns="*", async_job= async_job)
            self.cone_results = job.get_results()
            toc = time.perf_counter()
            if verbose:
                print(f"Cone search required {toc-tic} seconds")
        except ConnectionError as e:
            self.error_tracker.log_error(e, "Failed to connect to ESA Science Archive")
        
    
    @staticmethod
    def get_info_cutout(cone_results, filter_name):
        line = cone_results[cone_results["filter_name"]==filter_name][0]
        file_path = os.path.join(line["file_path"], line["file_name"])
        instrument = line["instrument_name"]
        obs_id = line["tile_index"]
        return file_path, instrument, obs_id
    
    def get_band_cutout(self, band, fname = None):
        file_path, instrument, obs_id = self.get_info_cutout(self.cone_results, band)
        if fname is None:
            fname = f"{obs_id}_{band}"
        else:
            fname = f"{fname}_{band}" #Need a different fname in each of the bands
        output_file = os.path.join(self.save_dir, f"{fname}.fits")  
        try:
            return self.client.get_cutout(file_path=file_path, instrument=instrument, id=obs_id, 
                                          coordinate=self.coordinates, radius = self.cutout_radius, 
                                          output_file=output_file)[0]
        except ConnectionError as e:
             self.error_tracker.log_error(e, "Failed to connect to ESA Science Archive")

    def get_cutouts(self, radius, verbose = False):
        
        self.cutout_radius = radius*u.arcsec
        self.cutouts_paths = {}
        
        tic = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = { 
                 executor.submit(self.get_band_cutout, band, fname = "tmp") : band
                 for band in self.euclid_filters
            }
        for future in concurrent.futures.as_completed(futures):
            band = futures[future]
            save_path = future.result()
            self.cutouts_paths[band] = save_path  
        
        toc = time.perf_counter()
        if verbose:
            print(f"Retrieving all cutouts requiered {toc-tic} seconds")

    def read_cutouts(self):
        self.data = {}
        self.wcs = {}
        self.arcsec_per_pix ={}
        for band in self.cutouts_paths:
            try:
                with fits.open(self.cutouts_paths[band]) as hdul:
                    self.data[band] = hdul[0].data
                    self.wcs[band] = WCS(hdul[0].header)  
                    self.arcsec_per_pix[band] = np.abs(hdul[0].header["CD1_1"]*3600)
            except OSError as e:
                self.error_tracker.log_error(e, "Downloaded Corrupted FITS file")
                continue

    def reproject_cutouts(self, reference = "VIS"):
        """Aligns and resizes VIS and NISP images so that can be stacked
        Reference can be either the name of the filter or its index"""

        if isinstance(reference, int):
            reference = self.euclid_filters[reference]
        
        ref_wcs = self.wcs[reference]  
        ref_shape = self.data[reference].shape
        self.arcsec_per_pix |= {"Color" : self.arcsec_per_pix[reference]}
        self.wcs  |= {"Color" : ref_wcs} 
        
        self.reprojected_data = {}
        for band in self.euclid_filters:
            reprojected, _ = reproject_interp((self.data[band], self.wcs[band]), ref_wcs, shape_out=ref_shape)
            self.reprojected_data |= {band : reprojected}
        
        self.data["Color"] = self.get_color_cutout(r_img = "NIR_H", g_img = "NIR_Y", b_img = "VIS", stretch=None,
                                                        stretch_interval = None)


    def get_plot_data(self, stretch = "Linear", 
                        stretch_scale = None,
                        stretch_interval = AsymmetricPercentileInterval(lower_percentile = 0.1, upper_percentile=100),
                        ):
        """This is just a convenient method which initializes the cutouts to be plotted for all bands plus 
           the color image by calling self._stretch_image"""
        
        stretch_map = {"Linear": lambda: LinearStretch(slope = stretch_scale if stretch_scale is not None else 1),
                      "Sqrt": lambda: SqrtStretch(),
                      "Log" : lambda: LogStretch(a = stretch_scale if stretch_scale is not None else 1000),
                      "Asinh": lambda: AsinhStretch(a = stretch_scale if stretch_scale is not None else 0.1),
                      "PowerLaw": lambda: PowerStretch(a = stretch_scale if stretch_scale is not None else 2)}
            
        if isinstance(stretch, str):
            stretch = stretch_map.get(stretch)()

        self.plot_data = {}
        self.plot_data_info = {}
        
        for band in self.euclid_filters:
            self.plot_data[band] =  self._stretch_image(self.data[band], stretch = stretch, 
                                                        stretch_interval = stretch_interval)
            self.plot_data_info[band] =  {"min_value" : np.nanmin(self.plot_data[band]),
                                          "max_value" : np.nanmax(self.plot_data[band]) }
        
        
        self.plot_data["Color"] = self.get_color_cutout(r_img = "NIR_H", g_img = "NIR_Y", b_img = "VIS", stretch=stretch,
                                                        stretch_interval = stretch_interval)

        self.plot_data_info["Color"] = { "min_value" : [np.nanmin(i) for i in self.plot_data["Color"]], 
                                         "max_value" : [np.nanmax(i) for i in self.plot_data["Color"]]}


    def get_color_cutout(self, r_img = "NIR_H", g_img = "NIR_Y", b_img = "VIS",
                        stretch = None, 
                        stretch_interval = None):
        
        images = [self._stretch_image(self.reprojected_data[band], stretch = stretch, 
                 stretch_interval = stretch_interval) for band in [r_img, g_img, b_img]]
         
        return  np.dstack(images)
  
    def transform_image_range(self, band, low, high, gamma = 1, scale_method = "MinMax",
                              scale_by_channel = False):
        """Clip and scales the plot. This is used to update the plot due 
           to a change of parameters in CustomPlot
        """
        
        if  band != "Color":
            image = self.plot_data[band]
            image_min = self.plot_data_info[band].get("min_value", None)
            image_max = self.plot_data_info[band].get("max_value", None)
            clipped_image, new_min, new_max = self._clip_image(image, low, high, 
                                            image_min = image_min, image_max = image_max)
            scaled_image = self._scale_image(clipped_image, 
                                            scale_method = scale_method,
                                            image_min = new_min, image_max = new_max)
        else:
            ##band == Color
            if not np.iterable(low):
                low = [low] * 3
            if not np.iterable(high):
                high = [high] * 3
            if not np.iterable(gamma):
                gamma = [gamma] * 3
            clipped_images = []
            abs_min, abs_max = np.inf, -np.inf
            image_min = np.min(self.plot_data_info[band]["min_value"])
            image_max = np.max(self.plot_data_info[band]["max_value"])
            for i in range(3):
                image = self.plot_data[band][:, :, i]
                clipped_image, new_min, new_max = self._clip_image(image, low[i], high[i], 
                                            image_min = image_min, image_max = image_max)
                clipped_image = clipped_image**gamma[i]
                abs_min = min(abs_min, new_min)
                abs_max = max(abs_max, new_max)
                if scale_by_channel:
                    clipped_image = self._scale_image(clipped_image, 
                                            scale_method = scale_method,
                                            image_min = new_min, image_max = new_max)
                clipped_images.append(clipped_image)
            
            scaled_image = self._scale_image(np.dstack(clipped_images), scale_method = "minmax",
                                             image_min = abs_min, image_max = abs_max)

        return scaled_image
    
    @staticmethod
    def _stretch_image(image, stretch, stretch_interval):
        if stretch is None:
            stretch = LinearStretch()
        if stretch_interval is None:
            transform = stretch
        else:
            transform = stretch + stretch_interval 
        return transform(image)
    
    @staticmethod
    def _clip_image(image, low, high, image_min = None, image_max = None, clip = False):
        
        if (low == 0) and (high == 1):
            return image, image_min, image_max
        if image_min is None:
            image_min = np.nanmin(image)
        if image_max is None:
            image_max = np.nanmax(image)
        
        image_range = image_max - image_min
        absolute_low = image_min + low * image_range
        absolute_high = image_min + high * image_range

        return np.clip(image, absolute_low, absolute_high), absolute_low, absolute_high
    
    @staticmethod
    def _scale_image(image, scale_method= "minmax", image_min = None, image_max = None):
        if scale_method.lower() =="minmax":
            if image_min is None:
                image_min = np.nanmin(image)
            if image_max is None:
                image_max = np.nanmax(image)
            scaled_image = (image-image_min)/(image_max-image_min)
            scaled_image = np.clip(scaled_image, 0,1)
    
        elif scale_method.lower() == "expand":
            print("using expand scale")
            mid_value = np.nanmedian(image)
            sigma = np.nanstd(image)
            scaled_image = np.where(image>mid_value+(1*sigma), image * 2, image / 2)
        
        else:
            raise ValueError(f"Unknown scale_method: {scale_method}") 
        return scaled_image


    def _add_overplot_coordinates(self, ra, dec, dataset = "default"):
        """
        Creates a dictionary to store coordinates from different datasets which can
        be then overplotted n the cutout.
        Parameters:
        ra, dec: float or list of floats, icrs coordinates
        dataset : str, allows to store independently coordinates from different datasets
        """
        if not hasattr(self, "overplot_coordinates"):
            self.overplot_coordinates = {}
        self.overplot_coordinates[dataset] = {"ra" : ra, "dec" : dec}

    def _convert_overplot_coordinates(self, filtro = "Color", dataset = "default", zipped = True):
        """
        Converts the stored coordinates into pixel coordinates for a given filter.
        If zipped == True returns a list of (x,y) poais of pixel coordinates
        otherwise returns x and y 
        Parameters:
        filtro : str, WCS key (default is "Color")
        dataset : str, datasets coordinates to be transformed into pixels
        """
        if not hasattr(self, "overplot_coordinates"):
            warnings.warn("No stored coordinates", UserWarning)
            return [] if zipped else (None, None)

        coords = SkyCoord(ra = self.overplot_coordinates[dataset]["ra"],
                          dec = self.overplot_coordinates[dataset]["dec"],
                          unit="deg", frame="icrs")
        x_pix, y_pix = self.wcs[filtro].world_to_pixel(coords)
        return list(zip(x_pix, y_pix)) if zipped else (x_pix, y_pix)
            

    def world_2_pix(self, ra, dec, filtro = "Color", zipped = True):
        """
        Same as _convert_overplot_coordinates but for external coordinates
        """
        coords = SkyCoord(ra = ra, dec = dec, unit="deg", frame="icrs")
        x_pix, y_pix = self.wcs[filtro].world_to_pixel(coords)
        return list(zip(x_pix, y_pix)) if zipped else (x_pix, y_pix)
    

    def export_cutouts_to_fits(self, bands_to_export, directory_path = "data/saved_sources"):
        """Saves the fits file, Fits file have already been downloaded/saved so it might actually be 
            better to just copy them into the required directory.
        """
        for band in bands_to_export:
            try:
                with fits.open(self.cutouts_paths[band]) as hdul:
                    data = hdul[0].data       
                    header =hdul[0].header
                hdu = fits.PrimaryHDU(data = data, header=header)
                hdul = fits.HDUList([hdu])
                
                filename = f"{band}_cutout.fits"
                hdul.writeto(os.path.join(directory_path, filename), overwrite=True)
            except KeyError:
                print("The required band is not available")
            except OSError as e:
                print(e)
            except FileNotFoundError as e:
                print(f"I could not find {self.cutouts_paths[band]}\n {e}")

        
    def get_final_cutout(self, radius, 
                         stretch =  "Linear", 
                         filtro = "Color", 
                         reference = "VIS", 
                         stretch_scale = None,
                         verbose = False,
                         return_object = False):
        """
        Method which calls sequentially all the other methods to get a cutout. return_object returns 
        the required cutout in addition to storing it as an attribute for multithread purposes.
        """
        
        self.error_tracker.reset()
        if not check_isin_survey(ra = self.coordinates.ra.value,
                                 dec = self.coordinates.dec.value,
                                 moc  = self.moc):
            self.error_tracker.log_error("Source not in the survey", 
                                         "The selected source is outside the survey coverage area")
            return None

        if not hasattr(self, "cone_results"):
            self.get_cone(verbose = verbose, async_job= False)
        
        if len(self.cone_results) <= 2:
            if verbose:
                print("Initial Cone Results failed, trying with a 1 deg^2 search radius")
            self.get_cone(initial_radius = 1*u.degree,  verbose = verbose, async_job= False)
        
        if self.error_tracker.has_error:
            return None
        
        elif len(self.cone_results) <= 2:
            self.error_tracker.log_error(
                                       "Cone search failed",
                                       "No sources found within search radius")
            return None
        
        self.get_cutouts(radius=radius, verbose=verbose)
        if self.error_tracker.has_error:
            return None
        self.read_cutouts()
        if self.error_tracker.has_error:
            return None
            
        self.reproject_cutouts(reference=reference)
        self.get_plot_data(stretch=stretch, stretch_scale = stretch_scale)
        
        if return_object:
            return self.plot_data.get(filtro, None)
         
    def clean_space(self):
        """Free quota of queries to Euclid Science Archive by removing asinchronous jobs. 
         It takes a couple of minutes"""
        joblist = self.client.list_async_jobs()
        to_remove = [j.jobid for j in joblist]
        self.client.remove_jobs(to_remove)


class BaseSpectraClass:
    """Base class to store methods which are used both for Euclid and DESI/SDSS spectra"""
    def __init__(self, ra, dec, max_separation = 1, sourceId = None, context = None):

        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config


        self.ra = ra
        self.dec = dec
        self.error_tracker = ErrorTracker()
        self.max_separation = max_separation / 3600  # arcsec to deg
        self.spectra = None
        self.sourceId = sourceId

    def reset_data(self, ra, dec, max_separation = 1, sourceId = None):
        self.ra = ra
        self.dec = dec 
        self.spectra = None
        self.max_separation = max_separation / 3600
        self.error_tracker.reset()
        self._remove_source_attributes()


    def get_coordinates(self):
        if self.spectra is not None:
            ra = [getattr(spectrum, "ra", np.nan) for spectrum in self.spectra]
            dec = [getattr(spectrum, "dec", np.nan) for spectrum in self.spectra]
            return ra, dec
        return np.nan, np.nan
    
    def get_smoothed_spectra(self, kernel = "Box1DKernel", window = 10):
        
        if isinstance(kernel, str):
            kernel_dict = {"box1dkernel" : lambda : Box1DKernel(window),
                           "gaussian1dkernel" : lambda : Gaussian1DKernel(window)}
            kernel = kernel_dict[kernel.casefold()]()
        else:
            kernel = kernel(window)
        
        self.smoothed_fluxes = [convolve(spectrum.flux, kernel, mask = spectrum.mask, boundary = "extend") 
                                for spectrum in self.spectra]
    
    def get_emline_table(self, primary = True, extra_path = "data"):
        """Emission lines for galaxies/AGN. 
           Table from  https://github.com/d-i-an-a/inspec-z
           Lines flagged as primary are the ~20 strongest in QSO spectra
           according to Vanden Berk+2001 """
        path = os.path.join(extra_path, "utility_data/EmLines_air_vac.csv")
        self.emline_table = pd.read_csv(path)
        if primary:
            self.emline_table = self.emline_table[self.emline_table["primary"]==1]
        self.emline_table["Name"] = self.emline_table["Name"].replace(np.nan, "")
        
    def get_absline_table(self, primary = True, extra_path = "data"):
        """Absorption lines for Stars """
        path = os.path.join(extra_path, "utility_data/AbsLines_air_vac.csv")
        self.absline_table = pd.read_csv(path)
        if primary:
            self.absline_table = self.absline_table[self.absline_table["primary"]==1]
        self.absline_table["Name"] = self.absline_table["Name"].replace(np.nan, "")

    @staticmethod
    def find_masked_regions(mask, min_width = 1):
        """Returns indexes indicating the start and the end of a masked region fom a boolean mask
           Returnin indexes only if the mask is wider than min_width (to avoid messy plots)"""
        mask = mask.astype(bool)
        diff = np.diff(mask.astype(int))
        mask_start_idx = np.where(diff == 1)[0] + 1 #True - False i.e. begin of the mask
        mask_end_idx = np.where(diff == -1)[0]   # False - True  i.e. end of the mask
        if mask[0]:
            mask_start_idx = np.insert(mask_start_idx, 0, 0)
        if mask[-1]:
            mask_end_idx = np.append(mask_end_idx, len(mask) - 1)
        
        mask_width = mask_end_idx-mask_start_idx + 1 #number of masked elements per region
        select = mask_width >= min_width
        return mask_start_idx[select], mask_end_idx[select]

    
    def plot_spectrum(self, ax, idx=0, plot_model=True, 
                      plot_lines = 'class',
                      plot_emlines=True, annotate_emlines=True,
                      plot_abslines=True, annotate_abslines=True,
                      plot_mask = True,
                      show_xlabel=True, show_ylabel=True,
                      plot_info = True,
                      model_kwargs = {"line_width" : 2, "color" : "red"},
                      smoothed_kwargs = {"line_width" : 1, "color" : "black"},
                      ):
        
        """This only plots one spectrum. Ideally all plotting routines should be outside of this 
           class. However having on plot routine is useful for managing em/abs lines and the different 
           spectra to plot
        """
        
        assert idx < self.available_spectra, "Index larger than number of available spectra"
        
        wavlen = self.spectra[idx].wavelength
        flux = self.spectra[idx].flux
        smoothed = self.smoothed_fluxes[idx]
        redshift = self.spectra[idx].redshift
        
        if plot_lines == "class":
            is_extragal = self.spectra[idx].spectype.casefold() in ["galaxy", "qso"]
            plot_emlines = is_extragal and plot_abslines
            plot_abslines = self.spectra[idx].spectype.casefold() == "star" and plot_abslines
        
        elif not plot_lines:
            plot_abslines = False
            plot_emlines = False
        
        ax.plot(wavlen, flux, c = 'grey', lw = 0.3, label = "Flux")
        ax.plot(wavlen, smoothed, **smoothed_kwargs, label = "Smoothed Flux")

        if plot_mask:
            start_idx, end_idx = self.find_masked_regions(mask = self.spectra[idx].mask, min_width=5)
            for s_idx, e_idx in zip(start_idx, end_idx):
                ax.axvspan(wavlen[s_idx], wavlen[e_idx], facecolor = "lightgrey", edgecolor = "none");
        
        if plot_model:
            ax.plot(wavlen, self.spectra[idx].model, **model_kwargs, label = "Model")
        
        ymin, ymax = np.nanmin(smoothed), np.nanmax(smoothed)
        ymin = ymin / 3 if ymin >= 0 else ymin * 1.5
        ymax = ymax * 1.5 if ymax >= 0 else ymax / 3 ##Sometimes Euclid Fluxes are negative
        xmin, xmax = np.nanmin(wavlen), np.nanmax(wavlen)
        
        if plot_emlines and np.isfinite(redshift):
            if not hasattr(self, "emline_table"):
                self.get_emline_table()
            obs_wav = self.emline_table["wave_vac"] * (redshift +1)
            logic = np.logical_and(obs_wav >= xmin, obs_wav <= xmax)
            obs_wav = obs_wav[logic]
            ax.vlines(obs_wav, ymin, ymax, color="r", lw =1, ls ="dotted")
          
            if annotate_emlines:
                y = (ymin + 0.8 * (ymax-ymin)) 
                names = self.emline_table["Name"][logic].astype(str)
                for wav, name in zip(obs_wav, names):
                    ax.text(wav, y,  name, fontsize = 8, color = "k")
                                   

        if plot_abslines and np.isfinite(redshift):
            if not hasattr(self, "absline_table"):
                self.get_absline_table()
            obs_wav = self.absline_table["wave_vac"] * (redshift +1)
            logic = np.logical_and(obs_wav >= xmin, obs_wav <= xmax)
            obs_wav = obs_wav[logic]
            ax.vlines(obs_wav, ymin, ymax, color="blue", lw =1, ls ="dotted")
            if annotate_abslines:
                y = (ymin + 0.8 * (ymax-ymin)) 
                names = self.emline_table["Name"][logic].astype(str)
                for wav, name in zip(obs_wav, names):
                    ax.text(wav, y,  name, fontsize = 8, color = "k")
        if plot_info:
            spectype = self.spectra[idx].spectype
            if np.isfinite(redshift) and len(spectype)>0:
                y = (ymin + 0.1 * (ymax-ymin)) 
                x = (xmin + 0.6 * (xmax-xmin)) 
                text = f"z = {np.round(redshift,4)}, Type = {spectype.upper()}"
                ax.text(x, y, text, fontsize =15, color = "k")

        ax.set_xlabel(r'$\lambda_{obs} ~[{Å}] $' if show_xlabel else '', fontsize = 15)
        ax.set_ylabel(r'$ F_{\lambda}~[10^{-17}~erg~s^{-1}~cm^{-2}~{Å}^{-1}] $' if show_ylabel else '', fontsize = 15)
        ax.set_xscale('log')
        ax.set_xlim(xmin, xmax * 1.02),
        ax.set_ylim(ymin, ymax),
        ax.legend(loc = "lower left")
 
    
    def plot_spectrum_hv(self, idx=0, plot_model=True, 
                         plot_lines = 'class',
                         plot_emlines=True, annotate_emlines=True,
                         plot_abslines=True, annotate_abslines=True,
                         plot_mask = True,
                         show_xlabel=True, show_ylabel=True,
                         plot_info = True,
                         model_kwargs = {"line_width" : 2, "color" : "red"},
                         smoothed_kwargs = {"line_width" : 1, "color" : "black"},
                         **kwargs):
        """Same as above but for holoviews
           plot_lines = True, False, "class" overrides plot_emlines and plot_abslines
        """

        assert idx < self.available_spectra, "Index larger than number of available spectra"
        
        wavlen = self.spectra[idx].wavelength
        flux = self.spectra[idx].flux
        smoothed = self.smoothed_fluxes[idx]
        redshift = self.spectra[idx].redshift


        if plot_lines == "class":
            is_extragal = self.spectra[idx].spectype.casefold() in ["galaxy", "qso"]
            plot_emlines = is_extragal and plot_abslines
            plot_abslines = self.spectra[idx].spectype.casefold() == "star" and plot_abslines
        
        elif not plot_lines:
            plot_abslines = False
            plot_emlines = False

        flux_curve = hv.Curve((wavlen, flux), label = "Flux").opts(color='grey', line_width=0.3)
        smoothed_curve = hv.Curve((wavlen, smoothed), label = "Smoothed Flux").opts(**smoothed_kwargs)

        if plot_mask:
            start_idx, end_idx = self.find_masked_regions(mask = self.spectra[idx].mask, min_width=5)
            masked_regions =hv.VSpans((wavlen[start_idx], wavlen[end_idx])).opts(line_color=None, 
                                                                                 color = "lightgrey")
            overlays = [masked_regions, flux_curve, smoothed_curve]
        else:
            overlays = [flux_curve, smoothed_curve]
        
        if plot_model:
            model = self.spectra[idx].model
            model_curve = hv.Curve((wavlen, model), label = "Model").opts(**model_kwargs)
            overlays.append(model_curve)
        
        ymin, ymax = np.nanmin(smoothed), np.nanmax(smoothed)
        ymin = ymin / 3 if ymin >= 0 else ymin * 1.5
        ymax = ymax * 1.5 if ymax >= 0 else ymax / 3 ##Sometimes Euclid Fluxes are negative
        xmin, xmax = np.nanmin(wavlen), np.nanmax(wavlen)
    
        if plot_emlines and np.isfinite(redshift):
            if not hasattr(self, "emline_table"):
                self.get_emline_table()
            obs_wav = self.emline_table["wave_vac"] * (redshift +1)
            logic = np.logical_and(obs_wav >= xmin, obs_wav <= xmax)
            obs_wav = obs_wav[logic]
            overlays.append(hv.VLines(obs_wav).opts(color='red', line_width=1, line_dash='dotted'))
            if annotate_emlines:
                y = (ymin + 0.8 * (ymax-ymin)) * np.ones_like(obs_wav)
                names = self.emline_table["Name"][logic].astype(str)
                overlays.append(hv.Labels((obs_wav, y,  names), vdims = "names").opts(
                                       text_font_size='8pt', text_color = "black"))

        if plot_abslines and np.isfinite(redshift):
            if not hasattr(self, "absline_table"):
                self.get_absline_table()
            obs_wav = self.absline_table["wave_vac"] * (redshift +1)
            logic = np.logical_and(obs_wav >= xmin, obs_wav <= xmax)
            obs_wav = obs_wav[logic]
            overlays.append(hv.VLines(obs_wav).opts(color='blue', line_width=1, line_dash='dotted'))
            if annotate_abslines:
                y = (ymin + 0.8 * (ymax-ymin)) * np.ones_like(obs_wav)
                names = self.absline_table["Name"][logic].astype(str)
                overlays.append(hv.Labels((obs_wav, y,  names), vdims = "names").opts(
                                        text_font_size='8pt', text_color = "black"))
        if plot_info:
           spectype = self.spectra[idx].spectype
           if np.isfinite(redshift) and len(spectype)>0:
               y = (ymin + 0.1 * (ymax-ymin)) 
               x = (xmin + 0.6 * (xmax-xmin)) 
               text = f"z = {np.round(redshift,4)}, Type = {spectype.upper()}"
               overlays.append(hv.Text(x, y, text).opts(text_font_size = "15pt", text_color = "black"))
        
        xlabel = r'$$ \lambda_{obs} ~[{Å}] $$' if show_xlabel else ''
        ylabel = r'$$ F_{\lambda}~[10^{-17}~erg~s^{-1}~cm^{-2}~{Å}^{-1}] $$' if show_ylabel else ''

        spectrum_overlay = hv.Overlay(overlays).opts(
                opts.Overlay(
                    xlabel=xlabel,
                    ylabel=ylabel,
                    logx=True,
                    xlim=(xmin, xmax * 1.02),
                    ylim=(ymin, ymax),
                    active_tools=[],
                    show_legend=True,
                    legend_position = 'bottom_left',
                    **kwargs,
                    )
                )
        return spectrum_overlay
    
    def plot_all_spectra_hv(self, plot_lines = "class",
                            plot_model = True,  cmap = "gist_rainbow", plot_mask = True, **kwargs):
        N = self.available_spectra
        nrows, ncols = N, 1
        hv_plots = []
        colors = plt.get_cmap(cmap, max(N,2))
        for idx in range(N):
            
            plot = self.plot_spectrum_hv(
                        idx=idx,
                        plot_lines = plot_lines,
                        plot_model = plot_model,
                        plot_mask = plot_mask,
                        model_kwargs = {"line_width" : 2, "color" : colors(idx)},
                        smoothed_kwargs = {"line_width" : 1 if plot_model else 2, "color" :  "black" if plot_model else colors(idx)},
                        **kwargs
                    )
            hv_plots.append(plot)
        full_plot  = hv.Layout(hv_plots).cols(ncols)
        full_plot = full_plot.opts(opts.Layout(shared_axes=False))
        return full_plot
    
    
    def plot_all_spectra(self, plot_lines = "class",
                         plot_model = True,  cmap = "gist_rainbow", 
                         plot_mask = True):
        N = self.available_spectra
        colors = plt.get_cmap(cmap, max(N,2))
        nrows, ncols = N, 1
        ax_height = 5 #height of the single ax
        ratio = 3.8 if N > 1 else 3.17 # width = ax_height*ratio
        fig, axs = plt.subplots(nrows = nrows, ncols =ncols, figsize =(ratio*ax_height, N*ax_height))
        for idx in range(N):
            ax = axs[idx] if N>1 else axs
            self.plot_spectrum(ax, plot_lines = plot_lines, 
                               plot_model = plot_model,
                               plot_mask = plot_mask,
                               model_kwargs = {"lw" : 2, "color" : colors(idx)},
                               smoothed_kwargs = {"lw" : 1 if plot_model else 2, "color" :  "black" if plot_model else colors(idx)},
                               )
        return fig
        
    def export_spectra_to_fits(self, fname,
                               directory_path = "data/saved_sources"):
        if self.available_spectra > 0:
            hdus = [fits.PrimaryHDU()]
            for spectrum in self.spectra:
                wavlen = spectrum.wavelength
                flux = spectrum.flux
                model = spectrum.model if hasattr(spectrum, "model") else np.full_like(wavlen, np.nan)
                mask = spectrum.mask 

                cols = [fits.Column(name="wavlen", array=wavlen, format="E", unit = "ANG"),  
                        fits.Column(name="flux", array=flux, format="E", unit = "1e-17"),
                        fits.Column(name="model", array=model, format="E", unit = "1e-17"),
                        fits.Column(name="mask", array=mask, format="L")]
                hdu = fits.BinTableHDU.from_columns(cols)
                hdu.header["SourceId"] = spectrum.sourceid
                hdu.header["RA"] = spectrum.ra
                hdu.header["DEC"] = spectrum.dec
                hdu.header["redshift"] = spectrum.redshift
                hdu.header["spectype"] = spectrum.spectype
                hdus.append(hdu)
            
            fname = fname + ".fits"
            filename = os.path.join(directory_path, fname)
            hdulist = fits.HDUList(hdus)
            hdulist.writeto(filename, overwrite = True)
                

class DESISpectraClass(BaseSpectraClass):
    """
    This class handles the queries of spectra from DESI and SDSS/BOSS.
    Query by sparclid and by specId are formally the same. In this class the first it is used to query all
    spectra within the max_separation distance (by providing multiple sparclid). specID instead is passed as a unique 
    int value and returns a single spectrum.
    """
    def __init__(self, ra, dec, max_separation = 1, 
                 datasets = ["DESI-DR1", "DESI-EDR", "BOSS-DR17", "SDSS-DR17"],
                 sourceId = None, context = None):
        super().__init__(ra, dec, max_separation=max_separation, sourceId=sourceId, context = context)
        
        self.context = context
        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        if isinstance(datasets, str): 
            datasets = [datasets]
        self.datasets = datasets

        svc = getattr(self.context, "services", None) if self.context is not None else None
        key = "sparcl.client"

        if svc is not None and svc.has(key):
            self.client = svc.get(key)
        else:
            self.client = SparclClient(read_timeout=60)
            print("Initialized SparcClient")
            if svc is not None:
                svc.set(key, self.client)

        if ("DESI-DR1" in self.datasets) | ("DESI-EDR" in self.datasets):
            self.moc = load_moc(survey = "DESI")
        elif ("BOSS-DR17" in self.datasets) | ("SDSS-DR17" in self.datasets):
            self.moc = load_moc(survey = "SDSS")


    def get_spectra(self, max_separation = None, return_object = False):
        """
        Method which calls sequentially all the other methods to get the spectra. return_object returns 
        the required spectra in addition to storing it as an attribute for multithread purposes.
        """
        self.error_tracker.reset()

        #if not check_isin_survey(ra = self.ra,
        #                         dec = self.dec,
        #                         moc  = self.moc):
        #    self.error_tracker.log_error("Source not in the survey", 
        #                                 "The selected source is outside the survey coverage area")
        #    return None
        
        if max_separation is not None:
            self.max_separation = max_separation / 3600

        if self.sourceId is not None:
            self.query_spectra_specid(verbose = True)

        else:
            self.query_main_table(verbose = True)
            if not self.error_tracker.has_error:
                self.query_spectra_sparclid(verbose = True)

        if not self.error_tracker.has_error:
            self.get_smoothed_spectra(kernel = "Box1dkernel",  window = 10)
        
        if return_object:
            return getattr(self, "spectra", None)
        
    def _remove_source_attributes(self):
        """Removes all attributes specific to a source"""
        attributes = ["table_results", "available_spectra", "spectrum_query"]
        for attribute in attributes:
            if hasattr(self, attribute):
                delattr(self, attribute)

    def query_main_table(self, verbose = False):
        
        """Query using SparcClient. It does not accept cone queries, so we first perform a box search within
        [ra-radius, ra+radius]* [dec-radius, dec+radius] and then we keep only sources effectively within the cone"""
        
        constraints  = {"ra" : [self.ra-self.max_separation, self.ra+self.max_separation],
                       "dec" : [self.dec - self.max_separation, self.dec+self.max_separation],
                       "data_release": self.datasets,
                       "specprimary" : [1]}
        outfields = ['sparcl_id','specid', 'ra', 'dec', "data_release"]

        self.coordinates = SkyCoord(ra = self.ra, dec = self.dec, unit = "deg", frame = "icrs")
        
        tic = time.perf_counter()
        try:
            found = self.client.find(outfields = outfields, constraints = constraints, limit = 200)
            self.table_results = pd.DataFrame.from_records(found.records)
            self.table_results = self.table_results.drop_duplicates(subset = "specid")
        except ReadTimeout as e:
            self.error_tracker.log_error(e, "Could not connect to Sparcl server before reaching timeout")
            return
        
        toc = time.perf_counter()
        if verbose:
            print(f"Querying table with Sparclient required {toc-tic} seconds")
        
        if len(self.table_results) > 1:
            self.table_results["separation"] = self.coordinates.separation(
                          SkyCoord(self.table_results["ra"], self.table_results["dec"], unit = "deg")).value
            self.table_results = self.table_results[self.table_results["separation"]<= self.max_separation].sort_values("separation")
        
        self.available_spectra = len(self.table_results)
        if self.available_spectra < 1:
            self.error_tracker.log_error("No spectrum available", 
                                         "No spectrum found around the provided coordinates")
            
  
    def get_info_spectra(self):
        for dataset in self.datasets:
            logic = (self.table_results["data_release"]==dataset) & (self.table_results["specprimary"]<= 1)
            if np.sum(logic) >= 1:
                selected = self.table_results[logic].reset_index(drop = True)
                return [selected.loc[0, "sparcl_id"]]   #first one is the closest
        return [self.table_results.loc[0, "sparcl_id"]]
 
    
    def query_spectra_sparclid(self, verbose = False):
        include = ['sparcl_id', 'specid', 'data_release', 'redshift', 'flux',
                   'wavelength', 'model', 'spectype', "ra", "dec", "mask"]
        
        if self.available_spectra >= 1:
            sparcl_id = list(self.table_results["sparcl_id"])
            tic = time.perf_counter()
            try:
                self.spectrum_query = self.client.retrieve(uuid_list = sparcl_id, dataset_list = self.datasets,
                                                          include = include)
                if self.spectrum_query.info["status"]["success"]:
                    self.spectrum_query = self.spectrum_query.reorder(sparcl_id)
                    self.spectra = self.spectrum_query.records
                else:
                    self.error_tracker.log_error("Failed to retrieve spectra", 
                                                 "The Sparcl query to retrive spectra failed")
            except (ConnectionError, ReadTimeout) as e:
                self.error_tracker.log_error(e, "Could not connect to Sparcl server before reaching timeout")
            
            toc = time.perf_counter()
            if verbose:
                print(f"Retrieving spectrum required {toc-tic} seconds")

    
    def query_spectra_specid(self, verbose = False):
        include = ['sparcl_id', 'specid', 'data_release', 'redshift', 'flux',
                    'wavelength', 'model', 'spectype', "ra", "dec", "mask"]
        
        if self.sourceId is not None:
            tic = time.perf_counter()
    
            try:
                self.spectrum_query = self.client.retrieve_by_specid([self.sourceId], include = include,
                                             dataset_list = self.datasets)
                if self.spectrum_query.info["status"]["success"]:
                    self.spectra = [self.spectrum_query.records[0]] #Same spectrum could be in both DESI DR1 and DESI EDR 
                    self.available_spectra = 1
                else:
                    self.error_tracker.log_error("Failed to retrieve spectra", 
                                                 "The Sparcl query to retrive spectra failed")
            except (ConnectionError, ReadTimeout) as e:
                self.error_tracker.log_error(e, "Could not connect to Sparcl server before reaching timeout")
            
            toc = time.perf_counter()
            if verbose:
                print(f"Retrieving spectrum required {toc-tic} seconds")

    
class SpectrumContainer:
    """Utility class to store retrieved Euclid Spectra in a similar way to DESI ones"""

    def __init__(self, wavelength, flux, mask, sourceId, **kwargs):
        self.wavelength = wavelength
        self.flux = flux
        self.mask = mask
        self.sourceId = sourceId
        self.allowed_attributes = ['model', "redshift", "ra", "dec", "spectype"]
        for name, value in kwargs.items():
            self.set_attribute(name, value)

    def set_attribute(self, attribute_name, attribute_value):
        if attribute_name in self.allowed_attributes:
            setattr(self, attribute_name, attribute_value)
        else:
            raise AttributeError(f"{attribute_name} is not allowed to be added")
        

class EuclidSpectraClass(BaseSpectraClass):

    def __init__(self, ra, dec, max_separation =1, sourceId = None, context = None):
        super().__init__(ra, dec, max_separation = max_separation, sourceId = sourceId, context = context)

        self.context = context
        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        svc = getattr(self.context, "services", None) if self.context is not None else None
        key = "euclid.client"

        if svc is not None and svc.has(key):
            self.client = svc.get(key)
        else:
            self.client = EuclidClass(environment="PDR")
            print("Initialized EuclidClass")
            if svc is not None:
                svc.set(key, self.client)
        self.moc = load_moc("Euclid_Q1")
    
    def _remove_source_attributes(self):
        """Removes all attributes specific to a source"""
        attributes = ["table_results", "available_spectra", "specz_results", "specz_table"]
        for attribute in attributes:
            if hasattr(self, attribute):
                delattr(self, attribute)

    def get_spectra(self, max_separation = None, return_object = False,
                    smooth_kernel = "Box1dkernel",  smooth_window = 5):
        """Method which calls sequentially all the other methods to get the spectra. return_object returns 
        the required spectra in addition to storing it as an attribute for multithread purposes.
        """
        
        self.error_tracker.reset()

        if not check_isin_survey(ra = self.ra,
                                 dec = self.dec,
                                 moc  = self.moc):
            self.error_tracker.log_error("Source not in the survey", 
                                         "The selected source is outside the survey coverage area")
        if max_separation is not None:
            self.max_separation = max_separation / 3600
        
        if self.sourceId is not None:
            self.query_spectra_sourceId(verbose=True)
            if not self.error_tracker.has_error:
                for spectrum in self.spectra:
                    spectrum.set_attribute("ra", np.nan)
                    spectrum.set_attribute("dec", np.nan)
                    spectrum.set_attribute("redshift", np.nan) 
                    spectrum.set_attribute("spectype", "")
                    
        else:
            self.query_table(verbose = True)
            if not self.error_tracker.has_error:
                self.query_spectra_sourceId(verbose=True)
                if not self.error_tracker.has_error:
                    source_id = list(self.table_results["source_id"])
                    self.spectra = self._reorder_spectra(self.spectra, source_id)
                    self._add_info_spectra()

        if not self.error_tracker.has_error:
            self.get_smoothed_spectra(kernel = smooth_kernel,  window = smooth_window)
        
        if return_object:
            return getattr(self, "spectra", None)
        
    def query_table(self, verbose = False):
        query = f"""SELECT TOP 400
                    spec.file_name, spec.file_path, spec.source_id, spec.spectra_source_oid, spec.ra_obj, spec.dec_obj,
                    DISTANCE(spec.ra_obj, spec.dec_obj, {self.ra}, {self.dec})*3600 AS separation
                    FROM q1.spectra_source AS spec
                    WHERE DISTANCE(ra_obj, dec_obj, {self.ra}, {self.dec}) < {self.max_separation}
                    ORDER BY separation
                """
        print(f"query ra dec: ({self.ra},{self.dec})")
        tic = time.perf_counter()
        job = self.client.launch_job(query)
            
        if (job is None) or (job.get_phase() in ("ERROR", "ABORTED")):
            self.error_tracker.log_error("Connection Error", "Failed to connect to ESA Science Archive")
            return
        
        self.table_results = job.get_results()
        self.available_spectra = len(self.table_results)
        if self.available_spectra < 1:
            self.error_tracker.log_error("No spectra in the field", f"No spectra found around the requested coordinates - ra: {self.ra}, dec:{self.dec}")
        toc = time.perf_counter()
        if verbose:
                print(f"Querying Euclid spectra_source table required {toc-tic} seconds")
        
  
    @staticmethod
    def _get_euclid_url(source_id, retrieval_type = "SPECTRA_RGS" ):
        """retrieval_type : str either  'SPECTRA_RGS', 'SPECTRA_BGS' or 'ALL'
        Type of spectrum to be retrieved, Red, or Blue grism, ALL returns a .zip file.
        In Q1 only Red Grism Spectra are available"""
        if not isinstance(source_id, list):
            source_id =[source_id]
        url = "https://eas.esac.esa.int/sas-dd/data?ID="
        id_list = ",".join(f"sedm+{s_id}" for s_id in source_id)
        url =  url + id_list + f"&RETRIEVAL_TYPE={retrieval_type}"
        return url
    
    @staticmethod
    def _reorder_spectra(spectra, source_id):
        """Retrieved spectra are not in the same order as the queried main table, i.e. they are not ordered by separation"""
        spectra_dict = {entry.sourceId : entry for entry in spectra}
        ordered_spectra = [spectra_dict[s_id] for s_id in source_id if s_id in spectra_dict]
        return ordered_spectra
    
    @staticmethod
    def _get_Euclid_mask(euclid_mask):
        """Converts Euclid Mask Flags convention into a boolean mask.
           odd flags and >=64 flags mean bad pixels
           Following https://caltech-ipac.github.io/irsa-tutorials/tutorials/euclid_access/3_Euclid_intro_1D_spectra.html"""
        mask = np.where((euclid_mask % 2 ==1) | (euclid_mask >= 64 ), 1, 0)
        return mask.astype(bool)

    def _add_info_spectra(self):
        """Spectra and main table must be ordered """
        if len(self.spectra) == self.available_spectra:
            for spectrum, ra, dec, s_id in zip(self.spectra,
                                            self.table_results["ra_obj"], 
                                            self.table_results["dec_obj"],
                                            self.table_results["source_id"]):
                if  spectrum.sourceId == s_id:
                    spectrum.set_attribute("ra", ra)
                    spectrum.set_attribute("dec", dec)
                    spectrum.set_attribute("redshift", np.nan) 
                    spectrum.set_attribute("spectype", "") 
        

    def query_spectra_sourceId(self, verbose = False):
        source_id = list(self.table_results["source_id"])
        url = self._get_euclid_url(source_id=source_id, retrieval_type= "SPECTRA_RGS")
        tic = time.perf_counter()
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            self.error_tracker.log_error(e, "Failed to retrieve spectra from ESA URL")
            return
        retrieved_content_type = r.headers.get("Content-Type", "")
        if retrieved_content_type.endswith("fits"):  #in the future we might incur in .zip files
            try:    
                self.spectra = []
                with fits.open(BytesIO(r.content)) as hdus:
                    for hdu in hdus[1:]:  #first one is empty
                        spectrum  = SpectrumContainer(wavelength = hdu.data['WAVELENGTH'],
                                                      flux = hdu.data["SIGNAL"] * hdu.header["FSCALE"] * 1e17,
                                                      mask = self._get_Euclid_mask(hdu.data["MASK"]),
                                                      sourceId = hdu.header["SOURC_ID"])
                        self.spectra.append(spectrum)
            except OSError as e:
                self.error_tracker.log_error(e, "Downloaded Corrupted FITS file")
               
        else:
            self.error_tracker.log_error(f"Received {retrieved_content_type}, only FITS supported for now", f"Unsupported Format: {retrieved_content_type}")
        toc = time.perf_counter()
        if verbose:
            print(f"Retrieving Euclid spectra required {toc-tic} seconds")


    def query_specz_table(self, verbose = False):
        """It queries the table with fitted specz and classification. If classification == "star", redshift is set to 0,
         else the one derived from galaxies with the highest probability is used. QSO redshift not available at the moment"""

        if self.spectra is not None:
            sourceid_list = [spectrum.sourceId for spectrum in self.spectra]
            sourceids= ",".join([str(s) for s in sourceid_list])
            query = f"""SELECT 
                    class.object_id, class.spe_class, gal.spe_z AS gal_z, gal.spe_z_prob
                    FROM catalogue.spectro_zcatalog_spe_classification as class
                    LEFT JOIN catalogue.spectro_zcatalog_spe_galaxy_candidates AS gal 
                    ON class.object_id = gal.object_id
                    WHERE class.object_id IN ({sourceids})
                    """
           
            tic = time.perf_counter()
            job = self.client.launch_job(query)
            if (job is None) or (job.get_phase() in ("ERROR", "ABORTED")):
                self.specz_table = None 
                return
            try:
                self.specz_results = job.get_results()
            
                #keeping only galaxy redshift with highest probability, reordering to match the sourceid_list
                self.specz_table = (self.specz_results.to_pandas().sort_values(["object_id", "spe_z_prob"], 
                                                                   ascending=[True, False]).drop_duplicates("object_id"))
                self.specz_table= self.specz_table.set_index("object_id").reindex(sourceid_list).reset_index()
                
                self.specz_table['redshift'] = np.select([self.specz_table["spe_class"] == "galaxy",
                                                                    self.specz_table["spe_class"] == "qso",
                                                                    self.specz_table["spe_class"].isna()],
                                                                   [self.specz_table["gal_z"],
                                                                    np.nan,
                                                                    np.nan], 
                                                                    default=0)
            except AttributeError:
                self.specz_table = None 
            toc = time.perf_counter()
            if verbose:
                print(f"Querying Euclid spectroscopic redshift table required {toc-tic} seconds")
    
    def _update_info_spectra(self, attribute, values):
        """Update the attributes of spectra in self.spectra. Same as _add_info_spectra but more general"""
        if hasattr(values, "__len__") and (not isinstance(values, str)) and (len(values) == self.available_spectra):
            for spectrum, value in zip(self.spectra, values):
                spectrum.set_attribute(attribute, value)
        else:
            if hasattr(values, "__len__") and not isinstance(values, str) and len(values) == 1:
                value = values[0]  
            else:
                value = values  # scalar or string
            for spectrum in self.spectra:
                spectrum.set_attribute(attribute, value)
        
    def update_info_from_query(self):
        if self.specz_table is not None:
            for attribute in ["spectype", "redshift"]:
                col_name = "spe_class" if attribute == "spectype" else attribute 
                self._update_info_spectra(attribute, self.specz_table[col_name].values)

def load_moc(survey, path = "data/mocs"):
    surveys = {"Euclid_Q1" : "Euclid_Q1_color.fits",
               "Euclid_DR1": "Euclid_Q1_color.fits",
               "DESI" : "DESI_from_query.fits",
               "SDSS" : "SDSS_color.fits",
               "VLASS" : "VLASS_QL.fits",
               "LoTSS" : "LoTSS_dr2.fits",
               }
    assert survey in surveys, f"No Moc file available for {survey}"
    return mocpy.MOC.from_fits(os.path.join(path, surveys[survey]))

def check_isin_survey(ra, dec, moc):
    return moc.contains_lonlat(ra*u.deg, dec*u.deg)