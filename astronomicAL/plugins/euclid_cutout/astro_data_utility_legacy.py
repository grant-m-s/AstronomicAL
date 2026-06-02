import os
import time
import concurrent.futures 
import numpy as np

import warnings
from astropy import units as u
from astroquery.esa.euclid import EuclidClass
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp 
from astropy.visualization import  PowerStretch, SqrtStretch, LogStretch
from astropy.visualization import AsinhStretch, LinearStretch, AsymmetricPercentileInterval
from astropy.coordinates import SkyCoord 
import mocpy

from astronomicAL.utils.error_tracker import ErrorTracker


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