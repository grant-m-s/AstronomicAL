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
from dl import queryClient as qc

import matplotlib.transforms as transforms
import holoviews as hv
from holoviews import opts

#from scipy.ndimage import zoom



class EuclidCutoutsClass:
    
    def __init__(self, ra, dec, 
                 euclid_filters = ["VIS", "NIR_Y", "NIR_J", "NIR_H"],
                 save_dir = "data/cutouts", check_coverage = True):
        self.coordinates = SkyCoord(ra, dec, unit = "degree", frame = "icrs")   
        self.euclid_filters = euclid_filters
        self.save_dir = save_dir
        if check_coverage:
            self.has_coverage = check_isin_survey(ra, dec, survey= "Euclid")
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        return None
    
    def get_cone(self, initial_radius = 0.3*u.degree, async_job=True, verbose = True):
        tic = time.perf_counter()
        job = Euclid.cone_search(self.coordinates, initial_radius, table_name="sedm.mosaic_product", ra_column_name="ra",
                                      dec_column_name="dec", columns="*", async_job= async_job)
        self.cone_results = job.get_results()
        toc = time.perf_counter()
        if verbose:
            print(f"Cone search required {toc-tic} seconds")
        return None
    
    @staticmethod
    def get_info_cutout(cone_results, filter_name):
        line = cone_results[cone_results["filter_name"]==filter_name][0]
        file_path = os.path.join(line["file_path"], line["file_name"])
        instrument = line["instrument_name"]
        obs_id = line["tile_index"]
        return file_path, instrument, obs_id
    
    def get_band_cutout(self, band):
        file_path, instrument, obs_id = self.get_info_cutout(self.cone_results, band)
        output_file = os.path.join(self.save_dir, f"{obs_id}_{band}.fits")  #This is not unique, cutous might be overwritten
        return Euclid.get_cutout(file_path=file_path, instrument=instrument, id=obs_id, 
                                coordinate=self.coordinates, radius = self.cutout_radius, output_file=output_file)[0]


    def get_cutouts(self, radius, verbose = False):
        
        self.cutout_radius = radius*u.arcsec
        self.cutouts_paths = {}
        
        tic = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = { 
                 executor.submit(self.get_band_cutout, band) : band
                 for band in self.euclid_filters
            }
        for future in concurrent.futures.as_completed(futures):
            band = futures[future]
            save_path = future.result()
            self.cutouts_paths[band] = save_path  
        
        toc = time.perf_counter()
        if verbose:
                print(f"Retrieving all cutouts requiered {toc-tic} seconds")
        return None 
    

    def read_cutouts(self):
        self.data = {}
        self.wcs = {}
        self.arcsec_per_pix ={}
        for band in self.cutouts_paths.keys():
            with fits.open(self.cutouts_paths[band]) as hdul:
                self.data |= {band : hdul[0].data}         #probably doesn't work in 3.8--> use .update
                self.wcs  |= {band : WCS(hdul[0].header)}  #Required for stacking images}  
                self.arcsec_per_pix  |= {band : np.abs(hdul[0].header["CD1_1"]*3600)}  
        return None 
    
    
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
        
        return None
    
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
        return None
    
    def _add_overplot_coordinates(self, ra, dec, filtro="stacked"):
        """Convert RA/Dec to pixel coordinates and store them.
        Parameters:
        ra, dec: float or list of floats
        filtro: WCS key (default is "stacked")
        """
        if not hasattr(self, "overplot_coordinates"):
            self.overplot_coordinates = {f : [] for  f in self.wcs.keys()}

        if isinstance(ra, (float, int)):
            ra = [ra]
            dec = [dec]
        
        coords = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        x_pix, y_pix = self.wcs[filtro].world_to_pixel(coords)
        self.overplot_coordinates[filtro].extend(zip(x_pix, y_pix))
        return None
        


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
        return None
    



class DESISpectraClass:
    #actually queries also BOSS and SDSS DR16
    def __init__(self, ra, dec, max_separation = 1, 
                 datasets = ["DESI-DR1", "DESI-EDR", "BOSS-DR16", "SDSS-DR16"],
                 specId = None, check_coverage = True):
        self.ra = ra
        self.dec = dec
        self.max_separation = max_separation/3600 #arcsec--> degreee
        self.spectrum = None 
        
        if isinstance(datasets, str): 
            datasets = [datasets]
        self.datasets = datasets
        
        self.has_coverage = False
        if check_coverage:
            if ("DESI-DR1" in self.datasets) | ("DESI-EDR" in self.datasets):
                self.has_coverage = self.has_coverage | check_isin_survey(self.ra, self.dec, survey = "DESI")
            if ("BOSS-DR16" in self.datasets) | ("SDSS-DR16" in self.datasets):
                self.has_coverage = self.has_coverage | check_isin_survey(self.ra, self.dec, survey = "SDSS")
        
        self.specId = specId
        
        self.client = SparclClient()

    def set_ra_dec(self, ra, dec):
        """To update class without recalling the SparcClient"""
        self.ra = ra
        self.dec = dec
        self.spectra = None
        return None
    
    def get_spectrum(self):
        """Call all methods to get a spectrum"""
        if self.specId is not None:
            self.query_spectra_specid(verbose = True)
        else:
            self.query_main_table(verbose = True)
            self.query_spectra_sparclid(verbose = True)
        return None

    def query_main_table(self, verbose = False):
        """Astro Data Lab does not accept Circle or Point functions,
          referred to https://datalab.noirlab.edu/help/index.php?qa=366&qa_1=dont-adql-functions-like-point-circle-work-query-interface
          for cone search.
          LIMIT = 100 just for avoiding strange results"""
        
        datasets_str = ', '.join(f"'{ds}'" for ds in self.datasets)

        query = f"""SELECT sparcl_id, specid, ra, dec, redshift, spectype, 
                 data_release, redshift_err, specprimary, redshift_warning,
                 3600.0 * q3c_dist(ra, dec, {self.ra}, {self.dec}) AS separation_arcsec
                 FROM sparcl.main
                 WHERE Q3C_RADIAL_QUERY(ra, dec, {self.ra},{self.dec}, {self.max_separation})
                 AND data_release IN ({datasets_str})
                 ORDER BY separation_arcsec ASC
                 LIMIT 100"""
        
        tic = time.perf_counter()
        self.table_results = qc.query(sql=query, fmt='pandas')
        self.table_results = self.table_results.drop_duplicates(subset = "specid")
        self.available_spectra = len(self.table_results)
        toc = time.perf_counter()
        if verbose:
            print(f"Querying Noirlab table required {toc-tic} seconds")
        return None
    
  
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
        return None
    
    def query_spectra_specid(self, verbose = False):
        include = ['sparcl_id', 'specid', 'data_release', 'redshift', 'flux',
                             'wavelength', 'model', 'spectype', "ra", "dec"]
        tic = time.perf_counter()
        if self.specId is not None:
            self.spectrum_query = self.client.retrieve_by_specid([self.specId], include = include,
                                             dataset_list = self.datasets)
            if self.spectrum_query.info["status"]["success"]:
                self.spectra = self.spectrum_query.records
            else:
                print("Something went wrong")
            toc = time.perf_counter()
            if verbose:
                print(f"Retrieving spectrum required {toc-tic} seconds")
        else:
             print("No target specId provided ")
        return None
    
    def get_coordinates_spectrum(self):
        """For cutout plottings"""
        if self.spectra is not None:
            self.coordinates_spectrum = SkyCoord(self.spectrum.ra, self.spectrum.dec, units = "deg",
                                                 frame = "icrs" )
    
    def get_smoothed_spectrum(self, kernel = "Box1DKernel", window = 10):

        if isinstance(kernel, str):
            kernel_dict = {"box1dkernel" : lambda : Box1DKernel(window),
                           "gaussian1dkernel" : lambda : Gaussian1DKernel(window)}
            kernel = kernel_dict[kernel.casefold()]()
        else:
            kernel = kernel(window)
        
        self.smoothed_fluxes  = [convolve(spectrum.flux, kernel) for spectrum in self.spectra]
        
        return None 
    
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
            self.absline_table = self.absline_table[self.emline_table["primary"]==1]



    def plot_spectrum(self, ax, plot_model = True, plot_emlines = True, plot_abslines = True):
        
        self.spectrum = self.spectra[0]
        self.smoothed_flux = self.smoothed_fluxes[0]


        wavlen = self.spectrum.wavelength
        redshift = self.spectrum.redshift

        ax.plot(wavlen, self.spectrum.flux, c = 'grey', lw = 0.1)
        ax.plot(wavlen, self.smoothed_flux, c = 'k', lw = 1)
        if plot_model:
            ax.plot(wavlen, self.spectrum.model, c= "r", lw = 2)
        
        ymin, ymax = np.min(self.smoothed_flux),  np.max(self.smoothed_flux)*1.5
        #sometimes ymin is < 0 so by dividing we are cutting out part of the spectrum
        ymin = ymin/3 if ymin >=0 else ymin*1.5
        xmin, xmax = np.min(wavlen), np.max(wavlen) 
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax*1.02)
        ax.set_xscale('log')
    
        if plot_emlines:
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
                ax.text(obs_wav, 0.8, name, rotation = 90, transform = transform, fontsize = 13)
        
        if plot_abslines:
            if not hasattr(self, "absline_table"):
                self.get_absline_table()
            transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            for name, wav in zip(self.absline_table["Name"],self.absline_table["wave_vac"]):
                obs_wav = wav*(redshift+1)
                if (obs_wav > xmax) or (obs_wav < xmin):
                    continue
                ax.axvline(obs_wav, c = 'b', lw = 0.5, ls = ':')
                ax.text(obs_wav, 0.2, name, rotation = 90, transform = transform, fontsize = 13)
        
        
        ax.set_xlabel(r'$\lambda_{obs}~[\AA]$', fontsize = 15)
        ax.set_ylabel(r'$F_{\lambda}~[10^{-17}~ergs~s^{-1}~cm^{-2}~{\AA}^{-1}]$', fontsize =15)
    
    

    def plot_spectrum_hv(self, plot_model = True, plot_emlines=True, plot_abslines=True):

        wavlen = self.spectrum.wavelength
        redshift = self.spectrum.redshift

        flux_curve = hv.Curve((wavlen, self.spectrum.flux)).opts(color='grey', line_width=0.1)
        smoothed_curve = hv.Curve((wavlen, self.smoothed_flux)).opts(color='black', line_width=1)
        
        overlays = [flux_curve, smoothed_curve]
        if plot_model:
            model_curve = hv.Curve((wavlen, self.spectrum.model)).opts(color='red', line_width=2)
            overlays.append(model_curve)
        ymin, ymax = np.min(self.smoothed_flux), np.max(self.smoothed_flux)*1.5
        ymin = ymin / 3 if ymin >= 0 else ymin * 1.5
        xmin, xmax =  xmin, xmax = np.min(wavlen), np.max(wavlen)
        
        if plot_emlines:
            if not hasattr(self, "emline_table"):
                self.get_emline_table()
            for name, wav in zip(self.emline_table["Name"], self.emline_table["wave_vac"]):
                obs_wav = wav * (redshift + 1)
                if obs_wav > xmax:
                    break
                if obs_wav < xmin:
                    continue
                line = hv.VLine(obs_wav).opts(color='red', line_width=0.5, line_dash='dotted')
                label = hv.Text(obs_wav, ymax*0.8, name).opts(yrotation=90, text_font_size='13pt')
                overlays.extend([line, label])
        if plot_abslines:
            if not hasattr(self, "absline_table"):
                self.get_absline_table()
            for name, wav in zip(self.absline_table["Name"], self.absline_table["wave_vac"]):
                obs_wav = wav * (redshift + 1)
                if (obs_wav > xmax) or (obs_wav < xmin):
                    continue
                line = hv.VLine(obs_wav).opts(color='blue', line_width=0.5, line_dash='dotted')
                label = hv.Text(obs_wav, ymax*0.2, name).opts(yrotation=90, text_font_size='13pt')
                overlays.extend([line, label])

        # Overlay all components
        spectrum_overlay = hv.Overlay(overlays).opts(
            opts.Curve(tools=['hover'], width=900, height=400),
            opts.Text(text_baseline='bottom'),
            opts.Overlay(
                 xaxis='bottom',
                 yaxis='left',
                 xlabel=r'$$\lambda_{obs}~[\AA\]$$',
                 ylabel=r'$$F_{\lambda}~[10^{-17}~ergs~s^{-1}~cm^{-2}~{\AA\}^{-1}]$$',
                 logx=True,
                 xlim=(xmin, xmax*1.02),
                 ylim=(ymin, ymax),
                 )
             )
        return spectrum_overlay


def LoTSS_cutout(ra, dec, radius = 10):
    """radius in arcsec"""
    has_coverage = check_isin_survey(ra, dec, survey = "LoTSS")
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
   


def VLASS_cutout(ra, dec, radius = 10, verbose = False):
    has_coverage = check_isin_survey(ra, dec, survey = "VLASS")
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
         
