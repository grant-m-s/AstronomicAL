import os
import time
import requests 
import concurrent.futures 
from io import BytesIO
import numpy as np
import pandas as pd
from astropy import units as u
from astroquery.esa.euclid import Euclid
from astroquery.cadc import Cadc
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp 
from astropy.visualization import  PowerStretch, SqrtStretch, LogStretch
from astropy.visualization import AsinhStretch, LinearStretch, AsymmetricPercentileInterval
from astropy.coordinates import SkyCoord 
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import mocpy

from sparcl.client import SparclClient 
from astronomicAL.extensions.shared_data import shared_data
#from dl import queryClient as qc

import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts

#from scipy.ndimage import zoom



class EuclidCutoutsClass:
    
    def __init__(self, ra, dec, 
                 euclid_filters = ["VIS", "NIR_Y", "NIR_J", "NIR_H"],
                 save_dir = "data/cutouts"):
        self.coordinates = SkyCoord(ra, dec, unit = "degree", frame = "icrs")   
        self.euclid_filters = euclid_filters
        self.save_dir = save_dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    
    def get_cone(self, initial_radius = 0.5*u.degree, async_job= False, verbose = True):
        tic = time.perf_counter()
        job = Euclid.cone_search(self.coordinates, initial_radius, table_name="sedm.mosaic_product", ra_column_name="ra",
                                      dec_column_name="dec", columns="*", async_job= async_job)
        self.cone_results = job.get_results()
        toc = time.perf_counter()
        if verbose:
            print(f"Cone search required {toc-tic} seconds")
        
    
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
            fname = f"{fname}_{band}" #need a different fname in each of the bands
        output_file = os.path.join(self.save_dir, f"{fname}.fits")  #This is not unique, cutous might be overwritten
        return Euclid.get_cutout(file_path=file_path, instrument=instrument, id=obs_id, 
                                coordinate=self.coordinates, radius = self.cutout_radius, output_file=output_file)[0]


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
        for band in self.cutouts_paths.keys():
            with fits.open(self.cutouts_paths[band]) as hdul:
                self.data |= {band : hdul[0].data}         #probably doesn't work in 3.8--> use .update
                self.wcs  |= {band : WCS(hdul[0].header)}  #Required for stacking images}  
                self.arcsec_per_pix  |= {band : np.abs(hdul[0].header["CD1_1"]*3600)}  
    
    
    
    def reproject_cutouts(self, reference = "VIS"):
        """Aligns and resizes VIS and NISP images so that can be stacked
        Reference can be either the name of the filter or its index"""

        if isinstance(reference, int):
            reference = self.euclid_filters[reference]
        
        ref_wcs = self.wcs[reference]  
        ref_shape = self.data[reference].shape
        self.arcsec_per_pix |= {"stacked" : self.arcsec_per_pix[reference]}
        self.wcs  |= {"stacked" : ref_wcs} #just in case
        
        self.reprojected_data = {}
        for band in self.euclid_filters:
            reprojected, _ = reproject_interp((self.data[band], self.wcs[band]), ref_wcs, shape_out=ref_shape)
            self.reprojected_data |= {band : reprojected}
        

    
    @staticmethod
    def transform_image(image, 
                        stretch = LinearStretch(slope =1), 
                        interval = AsymmetricPercentileInterval(lower_percentile = 1, upper_percentile=99)):
        transform = stretch + interval 
        return transform(image)
    

    def stack_cutouts(self, r_img = "NIR_H", g_img = "NIR_Y", b_img = "VIS",
                        stretch = LinearStretch(slope =1), 
                        interval = AsymmetricPercentileInterval(lower_percentile = 1, upper_percentile=99) ):
        
        stretch_map = {"Linear": lambda: LinearStretch(slope=1),
                      "Sqrt": lambda: SqrtStretch(),
                      "Log" : lambda: LogStretch(),
                      "Asinh": lambda: AsinhStretch(),
                      "PowerLaw": lambda: PowerStretch(a=2)}


        if isinstance(stretch, str):
            stretch = stretch_map.get(stretch)()

        norm_images = [self.transform_image(self.reprojected_data[band], stretch = stretch, interval = interval)
                   for band in [r_img, g_img, b_img]]
        self.reprojected_data |= {"stacked" : np.dstack(norm_images)}
 
    
    def _add_overplot_coordinates(self, ra, dec, dataset = "default"):
        """
        Creates a dictionary to store coordinates from different dataset which can
        be then overplotted n the cutout.
        Parameters:
        ra, dec: float or list of floats, icrs coordinates
        dataset : str, allows to store independently coordinates from different datasets
        """
        if not hasattr(self, "overplot_coordinates"):
            self.overplot_coordinates = {}
        self.overplot_coordinates[dataset] = {"ra" : ra, "dec" : dec}

        
    
    def _convert_overplot_coordinates(self, filtro = "stacked", dataset = "default"):
        """
        Converts the stored coordinates into pixel coordinates for a given filter.
        Returnz a list of (x,y) poais of pixel coordinates
        Parameters:
        filtro : str, WCS key (default is "stacked")
        dataset : str, datasets coordinates to be transformed into pixels
        """
        if hasattr(self, "overplot_coordinates"):
            coords = SkyCoord(ra = self.overplot_coordinates[dataset]["ra"],
                             dec = self.overplot_coordinates[dataset]["dec"],
                            unit="deg", frame="icrs")
            x_pix, y_pix = self.wcs[filtro].world_to_pixel(coords)
            return list(zip(x_pix, y_pix))
        
        print("No stored coordinates")
        return []
    

    def world_2_pix(self, ra, dec, filtro = "stacked"):
        """
        Same as _convert_overplot_coordinates but for external coordinates
        """
        coords = SkyCoord(ra = ra, dec = dec, unit="deg", frame="icrs")
        x_pix, y_pix = self.wcs[filtro].world_to_pixel(coords)
        return list(zip(x_pix, y_pix))
        
        
    def get_final_cutout(self, radius, stretch =  "Linear", reference = "VIS", verbose = False) :
        """
        This method just calls all the other methods to obtain a color cutout which can be 
        rendered in the Euclid Cutout extension plot panel
        """
        if not hasattr(self, "cone_results"):
            self.get_cone(verbose = verbose, async_job= False)
        
        
        if len(self.cone_results) > 2:
            self.get_cutouts(radius = radius, verbose = verbose)
            self.read_cutouts()
            self.reproject_cutouts(reference = reference)
            self.stack_cutouts(stretch = stretch)
        else:
            print(f"No sources in Euclid dataset with {self.coordinates} coordinates")
    

    def check_coverage(self, path = "data/mocs"):
        self.has_coverage = check_isin_survey(ra = self.coordinates.ra.value,
                                              dec = self.coordinates.dec.value,
                                            survey = "Euclid",  path = path)
        
    def clean_space(self):
        """free quota of queries to Euclid Science Archibe by removing asinchronous jobs 
         It takes a couple of minutes"""
        joblist = Euclid.list_async_jobs()
        to_remove = [j.jobid for j in joblist]
        Euclid.remove_jobs(to_remove)



    @staticmethod
    def zoom_image(image, scale, same_shape = True, zooming_order = 1):
        if scale >= 1:
            return image
        if scale <= 0:
            return None
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        
        top = (height - new_height) // 2
        left = (width - new_width) // 2
        bottom = top + new_height
        right = left + new_width
        
        cropped_image = image[top:bottom, left:right]
        if not same_shape:
            return cropped_image
        zoom_factors = (height / new_height, width / new_width) if image.ndim == 2 else (height / new_height, width / new_width, 1)
        #return zoom(cropped_image, zoom_factors, order=zooming_order)
    
    def get_scaled_cutout(self, scale, same_shape = True, zooming_order = 1, verbose = False, filtro = "stacked"):
        """here scale refers to the current scaled image"""

        tic = time.perf_counter()
        if not hasattr(self,"scaled_cutouts"):
            self.scaled_cutouts = self.data.copy()
            self.scales = {band : 1 for band in self.scaled_cutouts}   #currently same scale for all bands
            self.scaled_arcsec_per_pix = self.arcsec_per_pix.copy()
        try:
            self.scales[filtro] = self.scales[filtro]*scale
            self.scaled_cutouts[filtro] = self.zoom_image(self.data[filtro], self.scales[filtro], same_shape = same_shape,
                                                         zooming_order=zooming_order)
            if same_shape:
                    self.scaled_arcsec_per_pix[filtro] = self.scaled_arcsec_per_pix[filtro] * scale
        except KeyError:    
            for band, image in self.data.items():
                self.scales[band] = self.scales[band]*scale
                self.scaled_cutouts[band] = self.zoom_image(image, self.scales[band], same_shape = same_shape,
                                                       zooming_order=zooming_order)
                if same_shape:
                    self.scaled_arcsec_per_pix[band] = self.scaled_arcsec_per_pix[band] * scale

        toc = time.perf_counter()
        
        if verbose:
            print(f"Scaling cutouts required {toc-tic} seconds")
   


class BaseSpectraClass:
    """Base class to store methods which are used both for Euclid and DESI/SDSS spectra"""
    def __init__(self, ra, dec, max_separation = 1, sourceId = None):
        self.ra = ra
        self.dec = dec
        self.max_separation = max_separation / 3600  # arcsec to deg
        self.spectra = None
        self.available_spectra = 0
        self.sourceId = sourceId
    
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
        
        self.smoothed_fluxes  = [convolve(spectrum.flux, kernel) for spectrum in self.spectra]
    
    def get_emline_table(self, primary = True, extra_path = "data"):
        """Emission lines for galaxies/AGN. 
           Table from  https://github.com/d-i-an-a/inspec-z
           Lines flagged as primary are the ~20 strongest in QSO spectra
           according to Vanden Berk+2001 """
        path = os.path.join(extra_path, "utility_data/EmLines_air_vac.csv")
        self.emline_table = pd.read_csv(path)
        if primary:
            self.emline_table = self.emline_table[self.emline_table["primary"]==1]
        
    def get_absline_table(self, primary = True, extra_path = "data"):
        """Absorption lines for Stars """
        path = os.path.join(extra_path, "utility_data/AbsLines_air_vac.csv")
        self.absline_table = pd.read_csv(path)
        if primary:
            self.absline_table = self.absline_table[self.absline_table["primary"]==1]

    
    def plot_spectrum(self,  ax, idx = 0, plot_model = True, 
                      plot_emlines = True, annotate_emlines = True,  
                      plot_abslines = True, annotate_abslines = True,
                      set_ylabel = True, model_kwargs = {"lw" : 2, "color" : "r"},
                      smoothed_kwargs = {"lw" : 1, "color" : "k"}):
        
        """This only plots one spectrum. Ideally all plotting routines should be outside of this 
           class. However having on plot routine is useful for managing em/abs lines and the different 
           spectra to plot
        """
        
        assert idx < self.available_spectra, "Index larger than number of available spectra"
        
        wavlen = self.spectra[idx].wavelength
        flux = self.spectra[idx].flux
        smoothed = self.smoothed_fluxes[idx]
        redshift = self.spectra[idx].redshift
        
        ax.plot(wavlen, flux, c = 'grey', lw = 0.1)
        ax.plot(wavlen, smoothed, **smoothed_kwargs)
        
        if plot_model:
            ax.plot(wavlen, self.spectra[idx].model, **model_kwargs)
        
        ymin, ymax = np.min(smoothed),  np.max(smoothed)*1.5
        #sometimes ymin is < 0 so by dividing we are cutting out part of the spectrum
        ymin = ymin/3 if ymin >=0 else ymin*1.5
        xmin, xmax = np.min(wavlen), np.max(wavlen) 
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax*1.02)
        ax.set_xscale('log')
    
        if plot_emlines and np.isfinite(redshift):
            if not hasattr(self, "emline_table"):
                self.get_emline_table()
            transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            for name, wav in zip(self.emline_table["Name"],self.emline_table["wave_vac"]):
                obs_wav = wav*(redshift+1)
                if obs_wav > xmax:
                    break
                elif obs_wav < xmin:
                    continue
                ax.axvline(obs_wav, c = 'r', lw = 0.5, ls = ':')
                if annotate_emlines:
                    ax.text(obs_wav, 0.8, name, rotation = 90, transform = transform, fontsize = 10)
        
        if plot_abslines and np.isfinite(redshift):
            if not hasattr(self, "absline_table"):
                self.get_absline_table()
            transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            for name, wav in zip(self.absline_table["Name"],self.absline_table["wave_vac"]):
                obs_wav = wav*(redshift+1)
                if (obs_wav > xmax) or (obs_wav < xmin):
                    continue
                ax.axvline(obs_wav, c = 'b', lw = 0.5, ls = ':')
                if annotate_abslines:
                    ax.text(obs_wav, 0.2, name, rotation = 90, transform = transform, fontsize = 10)
        
        
        ax.set_xlabel(r'$\lambda_{obs}~[\AA]$', fontsize = 15)
        if set_ylabel:
            ax.set_ylabel(r'$F_{\lambda}~[10^{-17}~ergs~s^{-1}~cm^{-2}~{\AA}^{-1}]$', fontsize =12)
    
    
    def plot_spectrum_hv(self, idx=0, plot_model=True, 
                         plot_lines = 'class',
                         plot_emlines=True, annotate_emlines=True,
                         plot_abslines=True, annotate_abslines=True,
                         show_xlabel=True, show_ylabel=True,
                         width=400, height=250, 
                         model_kwargs = {"line_width" : 2, "color" : "red"},
                         smoothed_kwargs = {"line_width" : 1, "color" : "black"}):
        """Same as above but for holoviews
           plot_lines = True, False, "class" overrides plot_emlines and plot_abslines
        """

        assert idx < self.available_spectra, "Index larger than number of available spectra"
        
        wavlen = self.spectra[idx].wavelength
        flux = self.spectra[idx].flux
        smoothed = self.smoothed_fluxes[idx]
        redshift = self.spectra[idx].redshift

        if plot_lines == "class":
            is_extragal = self.spectra[idx].spectype == "GALAXY" or self.spectra[idx].spectype == "QSO"
            plot_emlines = is_extragal and plot_abslines
            plot_abslines = self.spectra[idx].spectype == "STAR" and plot_abslines
        
        elif not plot_lines:
            plot_abslines = False
            plot_emlines = False

        flux_curve = hv.Curve((wavlen, flux)).opts(color='grey', line_width=0.1)
        smoothed_curve = hv.Curve((wavlen, smoothed)).opts(**smoothed_kwargs)
        
        overlays = [flux_curve, smoothed_curve]
        
        if plot_model:
            model = self.spectra[idx].model
            model_curve = hv.Curve((wavlen, model)).opts(**model_kwargs)
            overlays.append(model_curve)
        
        ymin, ymax = np.min(smoothed), np.max(smoothed)*1.5
        ymin = ymin / 3 if ymin >= 0 else ymin * 1.5
        xmin, xmax =  xmin, xmax = np.min(wavlen), np.max(wavlen)
        
        if plot_emlines and np.isfinite(redshift):
            if not hasattr(self, "emline_table"):
                self.get_emline_table()
            for name, wav in zip(self.emline_table["Name"], self.emline_table["wave_vac"]):
                obs_wav = wav * (redshift + 1)
                if obs_wav > xmax: #emission lines are sorted
                    break
                if obs_wav < xmin:
                    continue
                overlays.append(hv.VLine(obs_wav).opts(color='red', line_width=0.5, line_dash='dotted'))
                if annotate_emlines:
                    overlays.append(hv.Text(obs_wav, ymax * 0.8, str(name)).opts(
                                      yrotation=90, text_font_size='8pt'))
        if plot_abslines and np.isfinite(redshift):
            if not hasattr(self, "absline_table"):
                self.get_absline_table()
            for name, wav in zip(self.absline_table["Name"], self.absline_table["wave_vac"]):
                obs_wav = wav * (redshift + 1)
                if (obs_wav > xmax) or (obs_wav < xmin):
                    continue
                overlays.append(hv.VLine(obs_wav).opts(color='blue', line_width=0.5, line_dash='dotted'))
                if annotate_abslines:
                    overlays.append(hv.Text(obs_wav, ymax * 0.2, str(name)).opts(
                                         yrotation=90, text_font_size='8pt'))
        
        xlabel = r'$$ \lambda_{obs} ~{Å} $$' if show_xlabel else ''
        ylabel = r'$$ F_{\lambda}~[10^{-17}~ergs~s^{-1}~cm^{-2}~{Å}^{-1}] $$' if show_ylabel else ''

        spectrum_overlay = hv.Overlay(overlays).opts(
                opts.Overlay(
                    width=width, 
                    height=height,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    logx=True,
                    xlim=(xmin, xmax * 1.02),
                    ylim=(ymin, ymax),
                    active_tools=[],
                    show_legend=False
                    )
                )
        return spectrum_overlay
    
    def plot_all_spectra_hv(self, plot_lines = "class",
                            plot_model = True,  cmap = "gist_rainbow", **kwargs):
        N = self.available_spectra
        nrows, ncols = N, 1
        hv_plots = []
        colors = plt.get_cmap(cmap, max(N,2))
        for idx in range(N):
            
            plot = self.plot_spectrum_hv(
                        idx=idx,
                        plot_lines = plot_lines,
                        plot_model = plot_model,
                        model_kwargs = {"line_width" : 2, "color" : colors(idx)},
                        smoothed_kwargs = {"line_width" : 1 if plot_model else 2, "color" :  "black" if plot_model else colors(idx)},
                        **kwargs
                    )
            hv_plots.append(plot)
        full_plot  = hv.Layout(hv_plots).cols(ncols)
        full_plot = full_plot.opts(opts.Layout(shared_axes=False))
        return full_plot

class DESISpectraClass(BaseSpectraClass):
    """
    This class handles the queries of spectra from DESI and SDSS/BOSS.
    Query by sparclid and by specId are formally the same. In this class the first it is used to query all
    spectra within the max_separation distance (by providing multiple sparclid). specID instead is passed as a unique 
    int value and returns a single spectrum.
    """
    #actually queries also BOSS and SDSS DR16
    def __init__(self, ra, dec, max_separation = 1, 
                 datasets = ["DESI-DR1", "DESI-EDR", "BOSS-DR16", "SDSS-DR16"],
                 sourceId = None, client = None):
        super().__init__(ra, dec, max_separation=max_separation, sourceId=sourceId)
        if isinstance(datasets, str): 
            datasets = [datasets]
        self.datasets = datasets

        if client is None:
            shared_data.set_data("Sparcl_client", SparclClient(read_timeout=60))
            print("Initialized SparcClient")
            self.client = shared_data.get_data("Sparcl_client")
        else:
            self.client = client
    
    def get_spectra(self):
        """Call all methods to get spectra"""
        if self.sourceId is not None:
            self.query_spectra_specid(verbose = True)
        else:
            self.query_main_table(verbose = True)
            self.query_spectra_sparclid(verbose = True)

    
    def query_main_table(self, verbose = False):
        """Query using SparcClient. It does not accept circular queries, so we first perform a box search within
        [ra-radius, ra+radius]* [dec-radius, dec+radius] and then we keep only sources effectively within the cone"""
        
        constraints  = {"ra" : [self.ra-self.max_separation, self.ra+self.max_separation],
                       "dec" : [self.dec - self.max_separation, self.dec+self.max_separation],
                       "data_release": self.datasets,
                       "specprimary" : [1]}
        outfields = ['sparcl_id','specid', 'ra', 'dec', "data_release"]

        self.coordinates = SkyCoord(ra = self.ra, dec = self.dec, unit = "deg", frame = "icrs")

        tic = time.perf_counter()
        found = self.client.find(outfields = outfields, constraints = constraints, limit = 200)
        self.table_results = pd.DataFrame.from_records(found.records)
        self.table_results = self.table_results.drop_duplicates(subset = "specid")
        
        if len(self.table_results) > 1:
            self.table_results["separation"] = self.coordinates.separation(
                          SkyCoord(self.table_results["ra"], self.table_results["dec"], unit = "deg")).value
            self.table_results = self.table_results[self.table_results["separation"]<= self.max_separation].sort_values("separation")

        self.available_spectra = len(self.table_results)
        toc = time.perf_counter()
        if verbose:
            print(f"Querying table with Sparclient required {toc-tic} seconds")
  
    def get_info_spectra(self):
        for dataset in self.datasets:
            logic = (self.table_results["data_release"]==dataset) & (self.table_results["specprimary"]<= 1)
            if np.sum(logic) >= 1:
                selected = self.table_results[logic].reset_index(drop = True)
                return [selected.loc[0, "sparcl_id"]]   #first one is the closest
        return [self.table_results.loc[0, "sparcl_id"]]
 
    
    def query_spectra_sparclid(self, verbose = False):
        include = ['sparcl_id', 'specid', 'data_release', 'redshift', 'flux',
                   'wavelength', 'model', 'spectype', "ra", "dec"]
        
        if self.available_spectra >= 1:
            sparcl_id = list(self.table_results["sparcl_id"])

            tic = time.perf_counter()
            self.spectrum_query = self.client.retrieve(uuid_list = sparcl_id, dataset_list = self.datasets,
                                              include = include)
            if self.spectrum_query.info["status"]["success"]:
                self.spectrum_query = self.spectrum_query.reorder(sparcl_id)
                self.spectra = self.spectrum_query.records
            else:
                print("Something went wrong")
            toc = time.perf_counter()
            if verbose:
                print(f"Retrieving spectrum required {toc-tic} seconds")
        else:
            print("No available spectra")

    
    def query_spectra_specid(self, verbose = False):
        include = ['sparcl_id', 'specid', 'data_release', 'redshift', 'flux',
                            'wavelength', 'model', 'spectype', "ra", "dec"]
        tic = time.perf_counter()
        if self.sourceId is not None:
            self.spectrum_query = self.client.retrieve_by_specid([self.sourceId], include = include,
                                             dataset_list = self.datasets)
            if self.spectrum_query.info["status"]["success"]:
                self.spectra = [self.spectrum_query.records[0]] #Same spectrum could be in both DESI DR1 and DESI EDR 
                self.available_spectra = 1
            else:
                print("Something went wrong")
            toc = time.perf_counter()
            if verbose:
                print(f"Retrieving DESI/SDSS spectra required {toc-tic} seconds")
        else:
             print("No target specId provided ")

    def check_coverage(self, path = "data/mocs"):
        self.has_coverage = False
        if ("DESI-DR1" in self.datasets) | ("DESI-EDR" in self.datasets):
            self.has_coverage = self.has_coverage | check_isin_survey(self.ra, self.dec, survey = "DESI", path = path)
        if ("BOSS-DR16" in self.datasets) | ("SDSS-DR16" in self.datasets):
            self.has_coverage = self.has_coverage | check_isin_survey(self.ra, self.dec, survey = "SDSS", path = path)
    
    
class SpectrumContainer:
    """Utility class to store retrieved Euclid Spectra in a simial way to DESI ones"""

    def __init__(self, wavelength, flux, sourceId, **kwargs):
        self.wavelength = wavelength
        self.flux = flux
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

    def __init__(self, ra, dec, max_separation =1, sourceId = None):
        super().__init__(ra, dec, max_separation = max_separation, sourceId = sourceId)

    
    def get_spectra(self):
        """Call all methods to get spectra"""
        if self.sourceId is not None:
            self.query_spectra_sourceId(verbose=True)
        else:
            self.query_table(verbose = True)
            if self.available_spectra > 0:
                self.query_spectra_sourceId(verbose=True)
                source_id = list(self.table_results["source_id"])
                self.spectra = self._reorder_spectra(self.spectra, source_id)
                self._add_info_spectra()
            else:
                print("No spectra available spectra")

    def query_table(self, verbose = False):
        query = f"""SELECT TOP 400
                        spec.file_name, spec.file_path, spec.source_id, spec.spectra_source_oid, spec.ra_obj, spec.dec_obj,
                    DISTANCE(spec.ra_obj, spec.dec_obj, {self.ra}, {self.dec})*3600 AS separation
                    FROM spectra_source AS spec
                    WHERE DISTANCE(ra_obj, dec_obj, {self.ra}, {self.dec}) < {self.max_separation}
                    ORDER BY separation
                """
        
        #10 times slower query to get also spectro-z
        #query_z = f"""SELECT TOP 400  
        #        spec.file_name, spec.file_path, spec.source_id, spec.spectra_source_oid, spec.ra_obj, spec.dec_obj,
        #    DISTANCE(spec.ra_obj, spec.dec_obj, {self.ra}, {self.dec})*3600 AS separation, 
        #    class.spe_class, gal.spe_z AS gal_z
        #    FROM spectra_source AS spec
        #    LEFT JOIN catalogue.spectro_zcatalog_spe_classification AS class on spec.source_id = class.object_id
        #    LEFT JOIN catalogue.spectro_zcatalog_spe_galaxy_candidates AS gal on spec.source_id = gal.object_id
        #    WHERE DISTANCE(ra_obj, dec_obj, {self.ra}, {self.dec}) < {self.max_separation}
        #    ORDER BY separation
        #    """
        #    #QSOs have bad spec-z
        #    #LEFT JOIN catalogue.spectro_zcatalog_spe_qso_candidates AS qso on spec.source_id = qso.object_id

        tic = time.perf_counter()
        job = Euclid.launch_job(query)
        try:
            self.table_results = job.get_results()
            self.available_spectra = len(self.table_results)
        except AttributeError:
            print("Query returned None")
            self.available_spectra = 0
        
        toc = time.perf_counter()
        if verbose:
            print(f"Querying Euclid spectra_source table required {toc-tic} seconds")
    
    @staticmethod
    def get_url(source_id, retrieval_type = "SPECTRA_RGS" ):
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
    

    def _add_info_spectra(self):
        """Spectra and main table must be ordered """
        if len(self.spectra) == len(self.table_results):
            for spectrum, ra, dec, s_id in zip(self.spectra,
                                            self.table_results["ra_obj"], 
                                            self.table_results["dec_obj"],
                                            self.table_results["source_id"]):
                if  spectrum.sourceId == s_id:
                    spectrum.set_attribute("ra", ra)
                    spectrum.set_attribute("dec", dec)
                    spectrum.set_attribute("redshift", np.nan) #avoid issues with plot
                    spectrum.set_attribute("spectype", "") #avoid issues with plot


    def query_spectra_sourceId(self, verbose = False):
         
        source_id = list(self.table_results["source_id"])
        url = self.get_url(source_id=source_id, retrieval_type= "SPECTRA_RGS")
        tic = time.perf_counter()
        r = requests.get(url)
        if r.status_code == 200:
            self.spectra = []
            if r.headers.get("Content-Type","").endswith("fits"): ##in the future we might incur in .zip files
                hdus = fits.open(BytesIO(r.content))
                for hdu in hdus[1:]:  #first one is empty
                    spectrum  = SpectrumContainer(wavelength = hdu.data['WAVELENGTH'],
                                                  flux = hdu.data["SIGNAL"],
                                                  sourceId = hdu.header["SOURC_ID"])
                    self.spectra.append(spectrum)
            else:
                print("For the  moment only fits format are supported")
        else:
            print(f"Euclid spectra request failed with status code {r.status_code}")
        toc = time.perf_counter()
        if verbose:
            print(f"Retrieving Euclid spectra required {toc-tic} seconds")
    
    def get_spectra(self):
        """Call all methods to get spectra"""
        self.spectra = None
        if self.sourceId is not None:
            self.query_spectra_sourceId(verbose=True)
            for spectrum in self.spectra:
                    spectrum.set_attribute("ra", np.nan)
                    spectrum.set_attribute("dec", np.nan)
                    spectrum.set_attribute("redshift", np.nan) 
                    spectrum.set_attribute("spectype", "")
            #Euclid Spectra fits file do not have position of the source
        else:
            self.query_table(verbose = True)
            if self.available_spectra > 0:
                self.query_spectra_sourceId(verbose=True)
                source_id = list(self.table_results["source_id"])
                self.spectra = self._reorder_spectra(self.spectra, source_id)
                self._add_info_spectra()
            else:
                print("No spectra available spectra")
 

def LoTSS_cutout(ra, dec, radius = 10, check_coverage = True):
    """radius in arcsec"""
    if check_coverage:
        has_coverage = check_isin_survey(ra, dec, survey = "LoTSS")
    else: 
        has_coverage = True
    if has_coverage:
        coordinates = SkyCoord(ra*u.deg, dec*u.deg, frame = "icrs")
        stringa = coordinates.to_string(style = "hmsdms").replace("h",":").replace("d",":").replace("m",":").replace("s","")
        url = f"https://lofar-surveys.org/dr2-cutout.fits?pos={stringa}&size={radius/60}"
        response =requests.get(url)
        try:
            image = fits.open(BytesIO(response.content))[0].data
            return image
        except Exception as e:
            print(e)
            return None
    else:
        print("LOFAR-LoTSS does not cover these coordinates")
        return None
   


def VLASS_cutout(ra, dec, radius = 10, verbose = False, check_coverage = True):
    if check_coverage:
        has_coverage = check_isin_survey(ra, dec, survey = "VLASS")
    else: 
        has_coverage = True
    if has_coverage:
        cadc = Cadc()
        coordinates = SkyCoord(ra*u.deg, dec*u.deg, frame = "icrs")
        tic = time.perf_counter()
        query_results = cadc.query_region(coordinates = coordinates, radius = radius*u.arcsec, 
                                        collection = "VLASS")
        query_results = query_results[query_results['requirements_flag'] != "fail"]
        query_results.sort("proposal_id", reverse = True)   ##first 3.1 then 2.1 then 1.1
        toc = time.perf_counter()
        if verbose:
            print(f"Queried coordinates in {toc-tic} seconds")
        if len(query_results)>0:
            urls = cadc.get_image_list(query_result=query_results[0:1], coordinates = coordinates, 
                                        radius = radius*u.arcsec)
            urls = [url for url in urls if "tt0.rms" not in url and "tt1.rms" not in url and "se.alpha" not in url] 
            response =requests.get(urls[0])
            try:
                image = fits.open(BytesIO(response.content))[0].data
                return image[0,0,:,:]
            except Exception as e:
                print(e)
                return None
    print("VLA-VLASS does not cover these coordinates")
    return None


def check_isin_survey(ra, dec, survey, path = "data/mocs"):
    surveys = {"Euclid" : "Euclid_Q1_color.fits",
               "DESI"   : "DESI_from_query.fits",
               "SDSS"   : "SDSS_color.fits",
               "VLASS"  : "VLASS_QL.fits",
               "LoTSS"  : "LoTSS_dr2.fits",
               }
    assert survey in surveys, f"No Moc file available for {survey}"
    moc = mocpy.MOC.from_fits(os.path.join(path, surveys[survey]))
    return moc.contains_lonlat(ra*u.deg, dec*u.deg)



def get_ra_dec_DESI():
    """"
    Utility function to get ra and dec for a source with DESI spectra.
    Only useful for testing the routines
    """
    client = SparclClient(read_timeout=60)
    outs = ['sparcl_id', 'ra', 'dec']
    cons = {'data_release': ['DESI-DR1']}
    found = client.find(outfields=outs, constraints=cons)
    ra = found.records[0]["ra"]
    dec = found.records[0]["dec"]
    return ra, dec


def get_ra_dec_Euclid():
    ra = 265.94946 
    dec = 65.83025    
    return ra, dec  


#########previous stuff 


class radio_cutouts_class:

    def __init__(self, ra, dec):
        self.ra = float(ra)
        self.dec = float(dec)
        self.get_url()
        return None
    
    def get_url(self):
        h = np.floor(self.ra / 15.0)
        d = self.ra - h * 15
        m = np.floor(d / 0.25)
        d = d - m * 0.25
        s = d / (0.25 / 60.0)
        s = np.round(s)
        ra_conv = f"{h} {m} {s}"
        sign = 1
        if self.dec < 0:
            sign = -1
        g = np.abs(self.dec)
        d = np.floor(g) * sign
        g = g - np.floor(g)
        m = np.floor(g * 60.0)
        g = g - m / 60.0
        s = g * 3600.0

        s = np.round(s)
        dec_conv = f"{d} {m} {s}"

        url1 = "https://third.ucllnl.org/cgi-bin/firstimage?RA="
        url2 = "&Equinox=J2000&ImageSize=2.5&MaxInt=200&GIF=1"
        self.url = f"{url1}{ra_conv} {dec_conv}{url2}"
        return None

    def get_image(self):
        r = requests.get(self.url)
        self.image = BytesIO(r.content).seek(0)
        return None
    
    def reset(self, ra, dec):
        self.ra = float(ra)
        self.dec = float(dec)
        self.get_url()
        return None



class sdss_cutouts_class:
        
        def __init__(self, ra, dec, scale = 0.2):
            self.ra = ra
            self.dec = dec 
            self.scale = scale
            self.get_url()

        def get_url(self):
            url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra="

            self.url = f"{url}{self.ra}&dec={self.dec}&opt=G&scale={self.scale}"
            return None

        def update_scale(self, scale):
            url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra="
            self.scale = scale
            self.url = f"{url}{self.ra}&dec={self.dec}&opt=G&scale={self.scale}"
            return None
        
        def reset(self, ra, dec, scale = 0.2):
            self.ra = ra
            self.dec = dec 
            self.scale = scale
            self.get_url()
            return None
         
    #def query_main_table(self, verbose = False):
    #    """Astro Data Lab does not accept Circle or Point functions,
    #      referred to https://datalab.noirlab.edu/help/index.php?qa=366&qa_1=dont-adql-functions-like-point-circle-work-query-interface
    #      for cone search.
    #      LIMIT = 100 just to avoid strange results"""
    #    
    #    datasets_str = ', '.join(f"'{ds}'" for ds in self.datasets)
    #
    #    query = f"""SELECT sparcl_id, specid, ra, dec, redshift, spectype, 
    #             data_release, redshift_err, specprimary, redshift_warning,
    #             3600.0 * q3c_dist(ra, dec, {self.ra}, {self.dec}) AS separation_arcsec
    #             FROM sparcl.main
    #             WHERE Q3C_RADIAL_QUERY(ra, dec, {self.ra},{self.dec}, {self.max_separation})
    #             AND data_release IN ({datasets_str})
    #             ORDER BY separation_arcsec ASC
    #             LIMIT 100"""
    #    
    #    tic = time.perf_counter()
    #    self.table_results = qc.query(sql=query, fmt='pandas')
    #    self.table_results = self.table_results.drop_duplicates(subset = "specid")
    #    self.available_spectra = len(self.table_results)
    #    toc = time.perf_counter()
    #    if verbose:
    #        print(f"Querying Noirlab table required {toc-tic} seconds")