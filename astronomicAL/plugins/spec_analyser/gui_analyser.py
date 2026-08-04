import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import os
import configparser
import logging
import threading
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Tuple, List
from functools import lru_cache

#Matplotlib and Astropy/Scipy Imports
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector

from astropy.table import Table
from astropy.modeling.models import Gaussian1D, Lorentz1D, Voigt1D
from astropy.modeling import fitting
from astropy import constants as const
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='astropy')
warnings.filterwarnings('ignore', message=".*contains multiple slashes.*")

# Constants and Configuration
EMISSION_LINES = {
    'Lyalpha': 1215.67,
    '[OII]': 3727.09,
    'Hbeta': 4861.32,
    '[OIII]4959': 4958.91,
    '[OIII]5007': 5006.84,
    'Halpha': 6562.80,
    '[SII]6716': 6716.44,
    '[SII]6731': 6730.82
}

CLASSIFICATION_COLOURS = {
    '[OII]': 'deep sky blue',
    '[OIII]5007': 'green',
    'Halpha': 'blue',
    'Unclear': 'orange',
    'Noisy/Bad': 'red',
    'Unclassified': 'black'
}

class AnalysisDefaults:
    SIGNAL_WINDOW_WIDTH = 50
    NOISE_OFFSET = 100
    NOISE_WINDOW_WIDTH = 150
    PEAK_HEIGHT_THRESHOLD_SIGMA = 2
    PEAK_MIN_DISTANCE = 30
    CONTINUUM_BUFFER = 20
    UPDATE_DEBOUNCE_MS = 250
    # Euclid-specific: typical resolution R~380 at 1.1-2.0 microns
    MIN_LINE_WIDTH_ANGSTROM = 2.0  # Minimum physical line width
    MAX_LINE_WIDTH_ANGSTROM = 100.0  # Maximum to catch broad lines

# Logging Setup
logging.basicConfig(
    filename='analyser.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Data Classes
@dataclass
class AppConfig:
    """Application configuration"""
    master_csv_path: str = ''
    spectra_folder: str = ''
    output_csv_path: str = './classified_spectra.csv'
    wavelength_col: str = 'WAVELENGTH'
    flux_col: str = 'SIGNAL'
    flux_error_col: str = ''
    lsf_variance_col: str = 'VAR'
    default_redshift_col: str = 'spe_z'
    show_tutorial: bool = True

    @classmethod
    def from_configparser(cls, config: configparser.ConfigParser):
        """Create AppConfig from ConfigParser"""
        return cls(
            master_csv_path=config.get('PATHS', 'master_csv_path', fallback=''),
            spectra_folder=config.get('PATHS', 'spectra_folder', fallback=''),
            output_csv_path=config.get('PATHS', 'output_csv_path', fallback='./classified_spectra.csv'),
            wavelength_col=config.get('FITS_COLUMNS', 'wavelength', fallback='WAVELENGTH'),
            flux_col=config.get('FITS_COLUMNS', 'flux', fallback='SIGNAL'),
            flux_error_col=config.get('FITS_COLUMNS', 'flux_error', fallback=''),
            lsf_variance_col=config.get('FITS_COLUMNS', 'lsf_variance', fallback='VAR'),
            default_redshift_col=config.get('ANALYSIS', 'default_redshift_col', fallback='spe_z'),
            show_tutorial=config.getboolean('USER_PREFERENCES', 'show_tutorial', fallback=True)
        )

    def to_configparser(self) -> configparser.ConfigParser:
        """Convert to ConfigParser format"""
        config = configparser.ConfigParser()
        config['PATHS'] = {
            'master_csv_path': self.master_csv_path,
            'spectra_folder': self.spectra_folder,
            'output_csv_path': self.output_csv_path
        }
        config['FITS_COLUMNS'] = {
            'wavelength': self.wavelength_col,
            'flux': self.flux_col,
            'flux_error': self.flux_error_col,
            'lsf_variance': self.lsf_variance_col
        }
        config['ANALYSIS'] = {
            'default_redshift_col': self.default_redshift_col
        }
        config['USER_PREFERENCES'] = {
            'show_tutorial': str(self.show_tutorial).lower()
        }
        return config

@dataclass
class AnalysisRegions:
    """Analysis region boundaries"""
    signal_start: float = 4000.0
    signal_end: float = 5000.0
    noise_start: float = 5100.0
    noise_end: float = 6000.0

    def is_valid(self) -> bool:
        return (self.signal_start < self.signal_end and
                self.noise_start < self.noise_end)

@dataclass
class PlotState:
    """Plotting display options"""
    show_original: bool = True
    show_continuum: bool = False
    show_continuum_sub: bool = True

@dataclass
class SpectrumData:
    """Container for spectrum data"""
    wv: np.ndarray
    flux: np.ndarray
    flux_err: Optional[np.ndarray] = None
    lsf_var: Optional[np.ndarray] = None
    source_id: str = ''
    continuum: Optional[np.ndarray] = None
    corrected_flux: Optional[np.ndarray] = None
    g_fit_model: Optional[object] = None
    error: Optional[str] = None

# Custom Exceptions
class AnalysisError(Exception):
    """Custom exception for analysis failures"""
    pass

# Utility Functions
def get_param_value(param):
    """Extract value from astropy parameter"""
    return param.value if hasattr(param, 'value') else param

def get_fwhm(model):
    """Calculate FWHM for different model types"""
    if isinstance(model, Gaussian1D):
        return get_param_value(model.fwhm)
    elif isinstance(model, Lorentz1D):
        return get_param_value(model.fwhm)
    elif isinstance(model, Voigt1D):
        fwhm_g = get_param_value(model.fwhm_G)
        fwhm_l = get_param_value(model.fwhm_L)
        return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)
    return np.nan

# Spectrum Analysis Classes
class ContinuumFitter:
    """Handles continuum fitting with sigma clipping"""

    @staticmethod
    def fit_with_sigma_clip(wv: np.ndarray, flux: np.ndarray,
                           mask: np.ndarray, iterations: int = 3,
                           sigma: float = 3.0, poly_degree: int = 1) -> np.poly1d:
        """Fit continuum using iterative sigma clipping"""
        current_mask = mask.copy()

        for _ in range(iterations):
            if np.sum(current_mask) < poly_degree + 1:
                return np.poly1d([0, np.median(flux)])

            coeffs = np.polyfit(wv[current_mask], flux[current_mask], poly_degree)
            continuum_model = np.poly1d(coeffs)
            residuals = flux - continuum_model(wv)
            std_dev = np.std(residuals[current_mask])

            if std_dev == 0:
                break

            outliers_mask = np.abs(residuals) > sigma * std_dev
            current_mask[outliers_mask] = False

        if np.sum(current_mask) > poly_degree:
            final_coeffs = np.polyfit(wv[current_mask], flux[current_mask], poly_degree)
            return np.poly1d(final_coeffs)
        else:
            return np.poly1d([0, np.median(flux)])

class LineFitter:
    """Handles emission line fitting with physical bounds for Euclid data"""

    def __init__(self, c_kms: float):
        self.c_kms = c_kms

    def fit_line(self, wv: np.ndarray, flux: np.ndarray,
                 corrected_flux: np.ndarray, regions: AnalysisRegions,
                 model_type: str, lsf_var: Optional[np.ndarray] = None) -> Dict:
        """Fit emission line and calculate properties with physical bounds"""

        # Create masks
        mask_signal = (wv >= regions.signal_start) & (wv <= regions.signal_end)
        mask_noise = (wv >= regions.noise_start) & (wv <= regions.noise_end)
        mask_noise = mask_noise & ~mask_signal

        x_fit = wv[mask_signal]
        y_fit = corrected_flux[mask_signal]

        if len(x_fit) < 5:
            raise AnalysisError("Insufficient data points in signal region")

        # Calculate noise
        stddev_noise = np.std(corrected_flux[mask_noise]) if np.any(mask_noise) else 0

        # Initial guesses
        mean_guess = x_fit[np.argmax(y_fit)]
        amp_guess = np.max(y_fit)

        # Estimate initial width from data (in Angstroms)
        initial_width = 15.0

        # Set up model with PHYSICAL BOUNDS
        if model_type == 'Gaussian':
            sigma_min = AnalysisDefaults.MIN_LINE_WIDTH_ANGSTROM / 2.355
            sigma_max = AnalysisDefaults.MAX_LINE_WIDTH_ANGSTROM / 2.355

            g_init = Gaussian1D(
                amplitude=amp_guess,
                mean=mean_guess,
                stddev=initial_width / 2.355,
                bounds={
                    'amplitude': (0, None),
                    'mean': (regions.signal_start, regions.signal_end),
                    'stddev': (sigma_min, sigma_max)
                }
            )
        elif model_type == 'Lorentzian':
            fwhm_min = AnalysisDefaults.MIN_LINE_WIDTH_ANGSTROM
            fwhm_max = AnalysisDefaults.MAX_LINE_WIDTH_ANGSTROM

            g_init = Lorentz1D(
                amplitude=amp_guess,
                x_0=mean_guess,
                fwhm=initial_width,
                bounds={
                    'amplitude': (0, None),
                    'x_0': (regions.signal_start, regions.signal_end),
                    'fwhm': (fwhm_min, fwhm_max)
                }
            )
        else:  # Voigt
            fwhm_min = AnalysisDefaults.MIN_LINE_WIDTH_ANGSTROM / 2
            fwhm_max = AnalysisDefaults.MAX_LINE_WIDTH_ANGSTROM / 2

            g_init = Voigt1D(
                x_0=mean_guess,
                amplitude_L=amp_guess,
                fwhm_L=initial_width / 2,
                fwhm_G=initial_width / 2,
                bounds={
                    'amplitude_L': (0, None),
                    'x_0': (regions.signal_start, regions.signal_end),
                    'fwhm_L': (fwhm_min, fwhm_max),
                    'fwhm_G': (fwhm_min, fwhm_max)
                }
            )

        fitter = fitting.LevMarLSQFitter()

        try:
            g_fit = fitter(g_init, x_fit, y_fit, maxiter=1000)
        except Exception as e:
            raise AnalysisError(f"Fitting failed: {str(e)}")

        if not fitter.fit_info.get('ierr', 0) in [1, 2, 3, 4]:
            raise AnalysisError("Fit did not converge properly")

        fwhm_obs = get_fwhm(g_fit)
        integral = np.sum(g_fit(x_fit)) * (x_fit[1] - x_fit[0]) if len(x_fit) > 1 else 0

        if stddev_noise > 0:
            fwhm_pixels = fwhm_obs / np.median(np.diff(x_fit)) if len(x_fit) > 1 else 1
            n_effective = max(1, fwhm_pixels)
            snr = integral / (stddev_noise * np.sqrt(n_effective))
        else:
            snr = 0

        lambda_center = (get_param_value(g_fit.mean) if hasattr(g_fit, 'mean')
                        else get_param_value(g_fit.x_0))

        fwhm_kms = self.c_kms * (fwhm_obs / lambda_center) if lambda_center > 0 else 0

        fwhm_int = np.nan
        if lsf_var is not None:
            lsf_fit_data = np.sqrt(lsf_var[mask_signal])
            int_fit_res = self._fit_with_lsf(x_fit, flux[mask_signal], lsf_fit_data)
            if int_fit_res:
                fwhm_int = int_fit_res.get('fwhm_int_A', np.nan)

        residuals = y_fit - g_fit(x_fit)
        n_params = 3
        dof = len(x_fit) - n_params
        chi_squared_red = np.sum(residuals**2) / dof if dof > 0 else np.inf

        return {
            'model': g_fit,
            'fwhm_obs_A': fwhm_obs,
            'fwhm_int_A': fwhm_int,
            'fwhm_kms': fwhm_kms,
            'flux_integral': integral,
            'snr': snr,
            'lambda_center': lambda_center,
            'chi_squared_red': chi_squared_red
        }

    def _fit_with_lsf(self, w: np.ndarray, f: np.ndarray,
                      sigma_lsf: np.ndarray) -> Optional[Dict]:
        """Fit line accounting for LSF"""
        if len(w) < 5:
            return None

        mu0 = w[np.argmax(f)]
        A0 = max(1e-6, float(np.max(f) - np.median(f)))
        c0_0 = float(np.median(f))
        c1_0 = 0.0
        sig0 = float(np.median(sigma_lsf)) if len(sigma_lsf) > 0 else 2.0

        def model(w_fit, A, mu, sigma_int, c0, c1):
            sigma_tot = np.sqrt(np.maximum(sigma_int, 0.0)**2 +
                              np.maximum(sigma_lsf, 0.0)**2)
            line = A * np.exp(-0.5 * ((w_fit - mu) / sigma_tot)**2)
            cont = c0 + c1 * (w_fit - mu)
            return cont + line

        bounds = (
            [0, np.min(w), 0.0, -np.inf, -np.inf],
            [np.inf, np.max(w), np.inf, np.inf, np.inf]
        )
        p0 = [A0, mu0, sig0, c0_0, c1_0]

        try:
            popt, _ = curve_fit(model, w, f, p0=p0, bounds=bounds, maxfev=5000)
            _, _, sigma_int, _, _ = popt
            fwhm_intrinsic = 2 * np.sqrt(2 * np.log(2)) * sigma_int
            return {"fwhm_int_A": fwhm_intrinsic}
        except (RuntimeError, ValueError):
            return None

class SpectrumAnalyser:
    """Main analysis coordinator for Euclid slitless spectra"""

    def __init__(self):
        self.c_kms = const.c.to('km/s').value
        self.continuum_fitter = ContinuumFitter()
        self.line_fitter = LineFitter(self.c_kms)

    def analyse(self, spectrum_data: SpectrumData, regions: AnalysisRegions,
                model_type: str, continuum_model: np.poly1d, flux_to_analyse: np.ndarray) -> Dict:
        """Perform full spectrum analysis on potentially residual flux"""

        if not regions.is_valid():
            raise AnalysisError("Invalid region definitions")

        wv = spectrum_data.wv

        try:
            fit_results = self.line_fitter.fit_line(
                wv, flux_to_analyse, flux_to_analyse, regions, model_type, spectrum_data.lsf_var
            )

            ew = (fit_results['flux_integral'] /
                  continuum_model(fit_results['lambda_center'])
                  if continuum_model(fit_results['lambda_center']) != 0 else 0)

            return {
                'source_id': spectrum_data.source_id,
                'signal_range': (regions.signal_start, regions.signal_end),
                'noise_range': (regions.noise_start, regions.noise_end),
                'flux_integral': fit_results['flux_integral'],
                'snr': fit_results['snr'],
                'fwhm_obs_A': fit_results['fwhm_obs_A'],
                'fwhm_int_A': fit_results['fwhm_int_A'],
                'fwhm_kms': fit_results['fwhm_kms'],
                'ew': ew,
                'chi_squared_red': fit_results['chi_squared_red'],
                'lambda_center': fit_results['lambda_center'],
                'fit_failed': False,
                'model': fit_results['model']
            }

        except Exception as e:
            logging.warning(f"Fitting failed: {e}")
            raise AnalysisError(f"Fit failed: {str(e)}")

# Data Management Classes
class SpectrumLoader:
    """Handles loading spectra from various sources"""

    def __init__(self, config: AppConfig):
        self.config = config

    def load_from_fits(self, filepath: str, column_map: Dict[str, str]) -> SpectrumData:
        """Load a single spectrum from a FITS file"""
        try:
            dataframe = Table.read(filepath, format='fits').to_pandas()
            return self._extract_data(dataframe, column_map, os.path.basename(filepath))
        except Exception as e:
            raise IOError(f"Failed to load FITS file: {e}")

    def load_spectra_from_folder(self, folder_path: str) -> pd.DataFrame:
        """Load all FITS files from a specified folder into a DataFrame."""
        try:
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.fits', '.fit'))]
            if not files:
                raise AnalysisError("No FITS files found in the selected folder.")

            df = pd.DataFrame({
                'filename': files,
                'full_path': [os.path.join(folder_path, f) for f in files]
            })
            return df
        except Exception as e:
            raise AnalysisError(f"Failed to process folder '{folder_path}': {e}")

    def load_spectra_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load a master CSV catalogue."""
        try:
            df = pd.read_csv(os.path.expanduser(csv_path))
            if 'full_path' not in df.columns:
                 raise AnalysisError("Master CSV must contain a 'full_path' column pointing to the FITS files.")
            return df
        except Exception as e:
            raise AnalysisError(f"Failed to load or parse CSV file '{csv_path}': {e}")

    def _extract_data(self, dataframe: pd.DataFrame,
                     column_map: Dict[str, str], source_id: str) -> SpectrumData:
        """Extract and validate data from dataframe"""
        try:
            wv = dataframe[column_map['wavelength']].astype(float).values
            flux = dataframe[column_map['flux']].astype(float).values

            err_col = column_map.get('flux_error', '')
            lsf_col = column_map.get('lsf_variance', '')

            flux_err = (dataframe[err_col].astype(float).values
                       if err_col and err_col in dataframe.columns else None)
            lsf_var = (dataframe[lsf_col].astype(float).values
                      if lsf_col and lsf_col in dataframe.columns else None)

            finite_mask = np.isfinite(wv) & np.isfinite(flux)

            return SpectrumData(
                wv=wv[finite_mask],
                flux=flux[finite_mask],
                flux_err=flux_err[finite_mask] if flux_err is not None else None,
                lsf_var=lsf_var[finite_mask] if lsf_var is not None else None,
                source_id=source_id
            )

        except KeyError as e:
            raise ValueError(f"Column not found: {e}")

class ResultsManager:
    """Manages classification results"""

    def __init__(self):
        self.results_dict: Dict[str, Dict] = {}
        self.undo_stack: List[Dict] = []

    def add_result(self, result: Dict):
        """Add or update a classification result"""
        source_id = result.get('source_id')
        if source_id:
            # Ensure any additional metadata from df_master is preserved if needed
            self.results_dict[source_id] = result
            self.undo_stack.append(result.copy())

    def undo_last(self) -> Optional[Dict]:
        """Undo last classification"""
        if not self.undo_stack:
            return None

        last_result = self.undo_stack.pop()
        source_id = last_result.get('source_id')
        if source_id and source_id in self.results_dict:
            del self.results_dict[source_id]

        return last_result

    def get_results_list(self) -> List[Dict]:
        """Get all results as list"""
        return list(self.results_dict.values())

    def export_to_csv(self, filepath: str):
        """Export results to CSV"""
        if not self.results_dict:
            raise ValueError("No results to export")

        df = pd.DataFrame(self.get_results_list())

        column_order = ['index', 'source_id', 'classification', 'z',
                       'flux_integral', 'snr', 'fwhm_obs_A', 'fwhm_int_A',
                       'fwhm_kms', 'ew', 'chi_squared_red', 'signal_range',
                       'noise_range', 'comment']
        
        # Add any other columns that might have been carried over from a master CSV
        other_cols = [col for col in df.columns if col not in column_order]
        final_column_order = column_order + sorted(other_cols)

        df = df.reindex(columns=[c for c in final_column_order if c in df.columns])
        df.to_csv(filepath, index=False)

        return len(df)

# Plotting Class
class SpectrumPlotter:
    """Handles all matplotlib plotting"""
    def __init__(self, fig, ax1, ax2, canvas):
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.canvas = canvas
        self.emission_lines = EMISSION_LINES.copy()

    def update_emission_lines(self, lines: Dict[str, float]):
        self.emission_lines = lines.copy()

    def plot_spectrum(self, spectrum_data: SpectrumData,
                     plot_state: PlotState, regions: AnalysisRegions,
                     z: float, analysis_results: Optional[Dict] = None,
                     locked_fits: List = None):
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.set_title("Full Spectrum - Drag to zoom below", fontsize=10)
        self.ax2.set_title("Zoomed View", fontsize=10)

        if spectrum_data.error:
            self._plot_error(spectrum_data.error)
            return
        
        if not spectrum_data.wv.size:
             self.canvas.draw()
             return

        wv = spectrum_data.wv
        flux = spectrum_data.flux

        if plot_state.show_original:
            self.ax1.plot(wv, flux, color='lightgrey', label='Original Spectrum', linewidth=0.8)

        total_locked_model = None
        if locked_fits:
            total_locked_model = np.zeros_like(wv)
            for locked_fit in locked_fits:
                model_name = locked_fit.get('model_name', 'Gaussian')
                locked_model = None
                if model_name == 'Gaussian':
                    locked_model = Gaussian1D(
                        amplitude=locked_fit['amplitude'],
                        mean=locked_fit['lambda_center'],
                        stddev=locked_fit.get('stddev_A', locked_fit['fwhm_obs_A']/2.355)
                    )
                elif model_name == 'Lorentzian':
                    locked_model = Lorentz1D(
                        amplitude=locked_fit['amplitude'],
                        x_0=locked_fit['lambda_center'],
                        fwhm=locked_fit['fwhm_obs_A']
                    )
                elif model_name == 'Voigt':
                    locked_model = Voigt1D(
                        amplitude_L=locked_fit.get('amplitude_L', locked_fit.get('amplitude')),
                        x_0=locked_fit['lambda_center'],
                        fwhm_L=locked_fit.get('fwhm_L', 5),
                        fwhm_G=locked_fit.get('fwhm_G', 5)
                    )
                if locked_model:
                    total_locked_model += locked_model(wv)
            self.ax1.plot(wv, total_locked_model, color='purple', lw=2,
                          label='Locked Fits', alpha=0.7)

        if plot_state.show_continuum_sub and spectrum_data.corrected_flux is not None:
            self.ax1.plot(wv, spectrum_data.corrected_flux,
                         color='black', lw=0.8, label='Continuum Subtracted')

        if plot_state.show_continuum and spectrum_data.continuum is not None:
            self.ax1.plot(wv, spectrum_data.continuum,
                         color='blue', linestyle='--', label='Continuum Fit', lw=1.5)

        self._plot_emission_lines(self.ax1, wv, z)

        self.ax1.set_ylabel("Flux", fontsize=10)
        self.ax1.legend(loc='best', fontsize=8)
        self.ax1.grid(True, alpha=0.3)

        flux_for_ax2 = spectrum_data.corrected_flux if spectrum_data.corrected_flux is not None else flux
        if locked_fits and total_locked_model is not None:
             flux_for_ax2 = flux_for_ax2 - total_locked_model

        self.ax2.plot(wv, flux_for_ax2, color='black', lw=0.8)

        self.ax2.axvspan(regions.signal_start, regions.signal_end,
                        color='green', alpha=0.2, label='Signal')
        self.ax2.axvspan(regions.noise_start, regions.noise_end,
                        color='gray', alpha=0.2, label='Noise')

        if analysis_results and not analysis_results.get('fit_failed'):
            if analysis_results.get('model') is not None:
                mask_signal = (wv >= regions.signal_start) & (wv <= regions.signal_end)
                chi2_str = f", χ²ᵣ={analysis_results.get('chi_squared_red', 0):.2f}"
                fit_label = (f"Fit (SNR={analysis_results.get('snr', 0):.1f}, "
                           f"FWHM={analysis_results.get('fwhm_obs_A', 0):.2f}Å{chi2_str})")
                self.ax2.plot(wv[mask_signal],
                            analysis_results['model'](wv[mask_signal]),
                            color='orange', lw=2, ls='--', label=fit_label)
        elif analysis_results and analysis_results.get('fit_failed'):
            self._plot_fit_failed(self.ax2, analysis_results.get('fail_reason', 'Unknown error'))

        ylabel_ax2 = "Residual Flux" if locked_fits else "Continuum Subtracted Flux"
        self.ax2.set(xlabel="Wavelength (Å)", ylabel=ylabel_ax2)
        self.ax2.legend(loc='best', fontsize=8)
        self.ax2.grid(True, alpha=0.3)

        if self.ax2.get_xlim() == (0.0, 1.0):
            self.ax2.set_xlim(regions.signal_start - 100, regions.signal_end + 100)

        self.canvas.draw()

    def _plot_emission_lines(self, ax, wv, z):
        wv_min, wv_max = wv.min(), wv.max()
        y_lim = ax.get_ylim()
        y_max = y_lim[1] if y_lim[1] != 1.0 else np.max(wv) * 0.1

        for name, rest_wl in self.emission_lines.items():
            obs_wl = rest_wl * (1 + z)
            if wv_min <= obs_wl <= wv_max:
                ax.axvline(obs_wl, color='red', linestyle='--', alpha=0.5, lw=0.8)
                ax.text(obs_wl + 10, y_max * 0.8, name,
                       rotation=90, color='red', ha='center', va='bottom',
                       fontsize=7, alpha=0.7)

    def _plot_error(self, error_msg):
        self.ax1.text(0.5, 0.5, f"Error loading spectrum:\n{error_msg}",
                     ha='center', va='center', color='red',
                     transform=self.ax1.transAxes, wrap=True)
        self.canvas.draw()

    def _plot_fit_failed(self, ax, reason):
        ax.text(0.5, 0.5, f"FIT FAILED\n({reason})",
               ha='center', va='center', color='red',
               transform=ax.transAxes, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', linewidth=2))
# Dialog Classes
class ColumnMappingDialog(tk.Toplevel):
    """Dialog for mapping FITS columns"""
    
    def __init__(self, parent, available_columns: List[str], current_mapping: Dict[str, str]):
        super().__init__(parent)
        self.title("Map FITS Columns")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.available_columns = available_columns
        self.current_mapping = current_mapping
        self.result = None
        
        self.vars = {
            'wavelength': tk.StringVar(),
            'flux': tk.StringVar(),
            'flux_error': tk.StringVar(),
            'lsf_variance': tk.StringVar()
        }
        self.save_to_config = tk.BooleanVar(value=True)
        
        self._create_widgets()
        self._pre_select_columns()
    
    def _create_widgets(self):
        frame = ttk.Frame(self, padding="10")
        frame.pack(expand=True, fill="both")
        
        ttk.Label(frame, text="Please map the required columns from your file:").grid(
            row=0, column=0, columnspan=2, pady=(0, 10))
        
        labels = {
            'wavelength': "Wavelength Column (Required):",
            'flux': "Flux Column (Required):",
            'flux_error': "Flux Error Column (Optional):",
            'lsf_variance': "LSF Variance Column (Optional):"
        }
        
        optional_columns = [''] + self.available_columns
        
        for i, key in enumerate(self.vars.keys()):
            ttk.Label(frame, text=labels[key]).grid(
                row=i+1, column=0, sticky="w", padx=5, pady=5)
            
            options = (self.available_columns if key in ['wavelength', 'flux'] 
                      else optional_columns)
            combo = ttk.Combobox(frame, textvariable=self.vars[key], 
                               values=options, width=30)
            combo.grid(row=i+1, column=1, sticky="ew", padx=5)
        
        ttk.Checkbutton(frame, text="Save these selections to config.ini",
                       variable=self.save_to_config).grid(
            row=len(self.vars) + 1, column=0, columnspan=2, pady=10)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=len(self.vars) + 2, columnspan=2)
        ttk.Button(btn_frame, text="Apply", command=self._on_apply).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(side="left", padx=10)
    
    def _pre_select_columns(self):
        """Intelligently guess correct columns"""
        guess_patterns = {
            'wavelength': ['wave', 'wl', 'lambda'],
            'flux': ['flux', 'signal', 'intensity'],
            'flux_error': ['err', 'ivar', 'error', 'uncertainty'],
            'lsf_variance': ['var', 'lsf', 'psf']
        }
        
        for key, var in self.vars.items():
            current_val = self.current_mapping.get(key, '')
            if current_val in self.available_columns:
                var.set(current_val)
                continue
            
            patterns = guess_patterns.get(key, [])
            for col in self.available_columns:
                if any(pattern in col.lower() for pattern in patterns):
                    var.set(col)
                    break
    
    def _on_apply(self):
        if not self.vars['wavelength'].get() or not self.vars['flux'].get():
            messagebox.showerror("Error", 
                               "Wavelength and Flux columns must be selected.", 
                               parent=self)
            return
        
        self.result = {key: var.get() for key, var in self.vars.items()}
        self.destroy()
    
    def _on_cancel(self):
        self.result = None
        self.destroy()

# Tutorial 
class TutorialWizard:
    """Welcome tutorial for first-time users"""
    
    def __init__(self, master, config_path: str):
        self.master = master
        self.config_path = config_path
        self.top = tk.Toplevel(master)
        self.top.title("Welcome to the Euclid Spectrum Analyser!")
        self.top.geometry("550x500")
        self.top.resizable(False, False)
        self.top.transient(master)
        self.top.grab_set()
        
        self.current_page = 0
        self.pages = [
            {
                "title": "Step 1: Setup & Load Data",
                "text": ("Welcome to the Euclid Slitless Spectra Analyser!\n\n"
                        "First, go to 'Settings > Configure Paths & Columns...' "
                        "to ensure all paths and FITS column names are correct for your data.\n\n"
                        "Then, use the 'File' menu to load your dataset. "
                        "If column names don't match, a mapping window will appear.\n\n"
                        "Note: This tool is optimized for Euclid's low-resolution "
                        "slitless spectra with typical FWHM ~20-30Å.")
            },
            {
                "title": "Step 2: Navigating the Spectrum",
                "text": ("The GUI is designed for rapid interaction:\n\n"
                        "- To zoom in, CLICK and DRAG a box on the TOP plot. "
                        "The bottom plot will update to show your selected region.\n\n"
                        "- Use the 'Go To:' buttons on the right to automatically jump to "
                        "the location of common emission lines based on the current redshift.\n\n"
                        "- The redshift slider allows fine-tuning of z.")
            },
            {
                "title": "Step 3: Defining Regions for Fitting",
                "text": ("This is the most important step for analysis!\n\n"
                        "Use the 'Analysis Regions' sliders in the right-hand panel to define "
                        "the green SIGNAL region and the grey NOISE region.\n\n"
                        "These sliders precisely control the areas used for line fitting "
                        "and SNR calculation.\n\n"
                        "For Euclid data, ensure the signal region captures the full "
                        "line profile (typically 50-100Å wide).")
            },
            {
                "title": "Step 4: Review Results & Classify",
                "text": ("The results of the fit (SNR, FWHM, χ²) will appear in the "
                        "'Derived Properties' box.\n\n"
                        "Physical bounds are enforced: line widths between 2-100Å.\n\n"
                        "Use the coloured buttons or number keys (1-5) to classify the spectrum. "
                        "Classifying automatically moves to the next object.\n\n"
                        "Use 'File > Export to CSV...' to save your work.\n\n"
                        "Happy analysing!")
            }
        ]
        
        self._create_widgets()
        self.update_page()
    
    def _create_widgets(self):
        self.title_label = ttk.Label(self.top, text="", 
                                    font=("Helvetica", 16, "bold"))
        self.title_label.pack(pady=15)
        
        self.text_label = ttk.Label(self.top, text="", wraplength=500,
                                   justify=tk.CENTER, font=("Helvetica", 10))
        self.text_label.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.show_again_var = tk.BooleanVar(value=False)
        checkbox = ttk.Checkbutton(self.top, text="Don't show this again",
                                  variable=self.show_again_var)
        checkbox.pack(pady=(10, 0))
        
        nav_frame = ttk.Frame(self.top)
        nav_frame.pack(pady=15, fill="x")
        
        self.prev_button = ttk.Button(nav_frame, text="< Previous",
                                     command=self.go_prev)
        self.prev_button.pack(side="left", padx=20)
        
        self.next_button = ttk.Button(nav_frame, text="Next >",
                                     command=self.go_next)
        self.next_button.pack(side="right", padx=20)
    
    def update_page(self):
        page_content = self.pages[self.current_page]
        self.title_label.config(text=page_content["title"])
        self.text_label.config(text=page_content["text"])
        
        self.prev_button['state'] = tk.NORMAL if self.current_page > 0 else tk.DISABLED
        
        if self.current_page == len(self.pages) - 1:
            self.next_button.config(text="Finish", command=self.close)
        else:
            self.next_button.config(text="Next >", command=self.go_next)
    
    def go_next(self):
        if self.current_page < len(self.pages) - 1:
            self.current_page += 1
            self.update_page()
    
    def go_prev(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_page()
    
    def close(self):
        if self.show_again_var.get():
            self._update_config_file()
        self.top.destroy()
    
    def _update_config_file(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        if 'USER_PREFERENCES' not in config:
            config['USER_PREFERENCES'] = {}
        config.set('USER_PREFERENCES', 'show_tutorial', 'false')
        with open(self.config_path, 'w') as f:
            config.write(f)

# Main Application Class
class SpectrumAnalyserGUI:
    """Main GUI application for Euclid slitless spectra"""

    def __init__(self, master):
        self.master = master
        self.master.title("Euclid Spectrum Analyser (Full Featured)")
        self.master.geometry("1400x950")

        self.config_path = 'config.ini'
        self.app_config = self._load_config()

        self.analyser = SpectrumAnalyser()
        self.loader = SpectrumLoader(self.app_config)
        self.results_manager = ResultsManager()

        self.df_master = None
        self.current_spectrum_data = SpectrumData(wv=np.array([]), flux=np.array([]))
        self.current_analysis_results = {}
        self.current_index = 0
        self.continuum_model = None

        self.regions = AnalysisRegions()
        self.plot_state = PlotState()
        self.emission_lines = EMISSION_LINES.copy()

        # UI variables
        self.z_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Welcome! Open a file or folder to begin.")
        self.comment_var = tk.StringVar()
        self.fit_model_var = tk.StringVar(value='Gaussian')
        self.z_finder_mode = tk.BooleanVar(value=False)
        self.multi_fit_mode_var = tk.BooleanVar(value=False)
        self.line_name_var = tk.StringVar()
        self.locked_fits = []
        
        self.goto_line_var = tk.StringVar()

        self.signal_start_var = tk.DoubleVar(value=self.regions.signal_start)
        self.signal_end_var = tk.DoubleVar(value=self.regions.signal_end)
        self.noise_start_var = tk.DoubleVar(value=self.regions.noise_start)
        self.noise_end_var = tk.DoubleVar(value=self.regions.noise_end)

        self.signal_start_var.trace_add('write', self._update_regions_from_vars)
        self.signal_end_var.trace_add('write', self._update_regions_from_vars)
        self.noise_start_var.trace_add('write', self._update_regions_from_vars)
        self.noise_end_var.trace_add('write', self._update_regions_from_vars)

        self.show_continuum_var = tk.BooleanVar(value=self.plot_state.show_continuum)
        self.show_original_var = tk.BooleanVar(value=self.plot_state.show_original)
        self.show_continuum_sub_var = tk.BooleanVar(value=self.plot_state.show_continuum_sub)

        self.show_continuum_var.trace_add('write', self._update_plot_state_from_vars)
        self.show_original_var.trace_add('write', self._update_plot_state_from_vars)
        self.show_continuum_sub_var.trace_add('write', self._update_plot_state_from_vars)

        self.analysis_thread = None
        self._update_job_id = None

        self._create_menu()
        self._create_widgets()
        self._bind_shortcuts()

        self.plotter = SpectrumPlotter(self.fig, self.ax1, self.ax2, self.canvas)

        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)
        logging.info("Application started.")
        self._update_status()

    def _create_menu(self):
        """Create menu bar"""
        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)

        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Single Spectrum...",
                            command=self._open_single_spectrum_dialog)
        file_menu.add_command(label="Open Spectra Folder...",
                            command=self._open_spectra_folder_dialog)
        file_menu.add_command(label="Open Master CSV...",
                            command=self._open_master_csv_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Load Emission Line List...",
                            command=self._load_emission_lines_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Export Classifications to CSV...", command=self._on_export)
        file_menu.add_command(label="Export Multi-Fitting Results...",
                              command=self._on_export_multifit)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)

        settings_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Configure Paths & Columns...",
                                 command=self._open_settings_dialog)

    def _create_widgets(self):
        """Create all GUI widgets"""
        main_frame = ttk.Frame(self.master, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(10, 8), constrained_layout=True,
            gridspec_kw={'height_ratios': [1, 2]}
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        zoom_props = dict(facecolor='red', edgecolor='red', alpha=0.1, fill=True)
        self.zoom_selector = RectangleSelector(
            self.ax1, self._on_zoom_select, button=[1],
            minspanx=5, minspany=5, spancoords='pixels',
            interactive=True, props=zoom_props
        )
        toolbar = NavigationToolbar2Tk(self.canvas, left_frame)
        toolbar.update()

        bottom_frame = ttk.Frame(left_frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        nav_frame = ttk.Frame(bottom_frame)
        nav_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.prev_button = ttk.Button(nav_frame, text="◀ Previous", command=self._on_prev)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.idx_spinbox = ttk.Spinbox(nav_frame, from_=1, to=1, command=self._on_goto, width=8)
        self.idx_spinbox.pack(side=tk.LEFT, padx=5)
        self.total_label = ttk.Label(nav_frame, text="/ 1")
        self.total_label.pack(side=tk.LEFT)
        self.next_button = ttk.Button(nav_frame, text="Next ▶", command=self._on_next)
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.classify_frame = ttk.Frame(bottom_frame)
        self.classify_frame.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        self.classify_frame.columnconfigure((0, 1, 2, 3, 4), weight=1)
        btn_opts = {'sticky': 'ew', 'padx': 2, 'pady': 2}
        tk.Button(self.classify_frame, text="[OII] (1)", bg="#5bc0de", fg="white",
                 command=lambda: self._classify_spectrum('[OII]')).grid(row=0, column=0, **btn_opts)
        tk.Button(self.classify_frame, text="[OIII] (2)", bg="#5cb85c", fg="white",
                 command=lambda: self._classify_spectrum('[OIII]5007')).grid(row=0, column=1, **btn_opts)
        tk.Button(self.classify_frame, text="Halpha (3)", bg="#0275d8", fg="white",
                 command=lambda: self._classify_spectrum('Halpha')).grid(row=0, column=2, **btn_opts)
        tk.Button(self.classify_frame, text="Unclear (4)", bg="#f0ad4e", fg="white",
                 command=lambda: self._classify_spectrum('Unclear')).grid(row=0, column=3, **btn_opts)
        tk.Button(self.classify_frame, text="Noisy/Bad (5)", bg="#d9534f", fg="white",
                 command=lambda: self._classify_spectrum('Noisy/Bad')).grid(row=0, column=4, **btn_opts)

        z_frame = ttk.LabelFrame(right_frame, text="Redshift (z)", padding=10)
        z_frame.pack(fill=tk.X, pady=5)
        ttk.Scale(z_frame, from_=0, to=7, orient=tk.HORIZONTAL,
                 variable=self.z_var, command=self._schedule_update).pack(fill=tk.X)
        ttk.Spinbox(z_frame, from_=-0.1, to=7, increment=0.001,
                   textvariable=self.z_var, command=self._schedule_update,
                   width=10).pack(side=tk.LEFT, pady=5, expand=True)
        ttk.Checkbutton(z_frame, text="Finder Mode",
                       variable=self.z_finder_mode,
                       command=self._toggle_z_finder_mode).pack(side=tk.RIGHT)

        plot_opts_frame = ttk.LabelFrame(right_frame, text="Plotting Options", padding=10)
        plot_opts_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(plot_opts_frame, text="Show Original Spectrum",
                       variable=self.show_original_var).pack(anchor=tk.W)
        ttk.Checkbutton(plot_opts_frame, text="Show Continuum Subtracted",
                       variable=self.show_continuum_sub_var).pack(anchor=tk.W)
        ttk.Checkbutton(plot_opts_frame, text="Show Continuum Fit",
                       variable=self.show_continuum_var).pack(anchor=tk.W)

        fit_frame = ttk.LabelFrame(right_frame, text="Fitting Model", padding=10)
        fit_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fit_frame, text="Line Profile:").pack(side=tk.LEFT, padx=(0, 5))
        model_combo = ttk.Combobox(fit_frame, textvariable=self.fit_model_var,
                                   values=['Gaussian', 'Lorentzian', 'Voigt'], width=10)
        model_combo.pack(side=tk.LEFT, expand=True, fill=tk.X)
        model_combo.bind('<<ComboboxSelected>>', self._schedule_update)

        multi_fit_frame = ttk.LabelFrame(right_frame, text="Multi-Line Fitting", padding=10)
        multi_fit_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(multi_fit_frame, text="Enable Multi-Line Fit Mode",
                       variable=self.multi_fit_mode_var,
                       command=self._toggle_multi_fit_mode).pack(anchor=tk.W)
        self.multi_fit_controls = ttk.Frame(multi_fit_frame)
        self.multi_fit_controls.pack(fill=tk.X, pady=(5,0))
        ttk.Label(self.multi_fit_controls, text="Line Name:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(self.multi_fit_controls, textvariable=self.line_name_var, width=20).grid(
            row=0, column=1, sticky='ew', pady=2, padx=(5,0))
        multi_btn_frame = ttk.Frame(self.multi_fit_controls)
        multi_btn_frame.grid(row=1, column=0, columnspan=2, pady=(5,0))
        ttk.Button(multi_btn_frame, text="Fit & Lock Line",
                  command=self._on_lock_fit).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(multi_btn_frame, text="Reset All Fits",
                  command=self._on_reset_fits).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))
        self.multi_fit_controls.columnconfigure(1, weight=1)

        sliders_frame = ttk.LabelFrame(right_frame, text="Analysis Regions (Å)", padding=10)
        sliders_frame.pack(fill=tk.X, pady=5)
        
        # Create a dictionary to hold the spinbox widgets for easy access later
        self.region_spinboxes = {}

        def create_slider_spinbox_pair(parent, label, var):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=1)
            ttk.Label(frame, text=label, width=15).pack(side=tk.LEFT)
            spinbox = ttk.Spinbox(frame, from_=3000, to=10000, textvariable=var, width=8, format="%.1f")
            spinbox.pack(side=tk.RIGHT, padx=5)
            slider = ttk.Scale(frame, from_=3000, to=10000, orient=tk.HORIZONTAL, variable=var)
            slider.pack(fill=tk.X, expand=True)
            return slider, spinbox

        self.signal_start_scale, self.region_spinboxes['signal_start'] = create_slider_spinbox_pair(sliders_frame, "Signal Start:", self.signal_start_var)
        self.signal_end_scale, self.region_spinboxes['signal_end'] = create_slider_spinbox_pair(sliders_frame, "Signal End:", self.signal_end_var)
        self.noise_start_scale, self.region_spinboxes['noise_start'] = create_slider_spinbox_pair(sliders_frame, "Noise Start:", self.noise_start_var)
        self.noise_end_scale, self.region_spinboxes['noise_end'] = create_slider_spinbox_pair(sliders_frame, "Noise End:", self.noise_end_var)
        
        self.line_frame = ttk.LabelFrame(right_frame, text="Emission Line Positions", padding=10)
        self.line_frame.pack(fill=tk.X, pady=5)
        self._build_line_widgets()

        results_frame = ttk.LabelFrame(right_frame, text="Derived Properties (Current Fit)", padding=10)
        results_frame.pack(fill=tk.BOTH, pady=5, expand=True)

        self.results_tree = ttk.Treeview(results_frame, columns=('prop', 'val'), show='', height=7)
        self.results_tree.column('prop', width=140, anchor='w')
        self.results_tree.column('val', width=100, anchor='e')
        self.results_tree.pack(fill=tk.BOTH, expand=True)

        locked_fits_frame = ttk.LabelFrame(right_frame, text="Locked Fits", padding=10)
        locked_fits_frame.pack(fill=tk.BOTH, pady=5, expand=True)

        self.locked_fits_tree = ttk.Treeview(
            locked_fits_frame,
            columns=('name', 'wav', 'fwhm', 'snr'),
            show='headings',
            height=4
        )
        self.locked_fits_tree.heading('name', text='Name')
        self.locked_fits_tree.column('name', width=80)
        self.locked_fits_tree.heading('wav', text='λ (Å)')
        self.locked_fits_tree.column('wav', width=70, anchor='center')
        self.locked_fits_tree.heading('fwhm', text='FWHM')
        self.locked_fits_tree.column('fwhm', width=60, anchor='center')
        self.locked_fits_tree.heading('snr', text='SNR')
        self.locked_fits_tree.column('snr', width=50, anchor='center')
        self.locked_fits_tree.pack(fill=tk.BOTH, expand=True)

        comment_frame = ttk.LabelFrame(right_frame, text="Comment", padding=10)
        comment_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(comment_frame, textvariable=self.comment_var).pack(fill=tk.X)

        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        action_frame.columnconfigure((0, 1), weight=1)
        ttk.Button(action_frame, text="Undo Last",
                  command=self._on_undo).grid(row=0, column=0, sticky='ew', padx=2)
        ttk.Button(action_frame, text="Export CSV",
                  command=self._on_export).grid(row=0, column=1, sticky='ew', padx=2)

        self.status_label = ttk.Label(self.master, textvariable=self.status_var,
                                     relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        self._toggle_multi_fit_mode()

    def _build_line_widgets(self):
        """Build emission line widgets"""
        for widget in self.line_frame.winfo_children():
            widget.destroy()

        self.line_info_text = tk.Text(self.line_frame, height=8, width=20,
                                     state=tk.DISABLED, font=("Courier", 9))
        self.line_info_text.pack(fill=tk.Y, expand=False, side=tk.LEFT)
        
        go_to_frame = ttk.Frame(self.line_frame)
        go_to_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(go_to_frame, text="Go To Line:").pack(anchor='w')

        line_combo = ttk.Combobox(go_to_frame, textvariable=self.goto_line_var, 
                                  values=list(self.emission_lines.keys()),
                                  state='readonly')
        line_combo.pack(fill=tk.X, expand=True)
        if list(self.emission_lines.keys()): # Set a default value
            line_combo.set(list(self.emission_lines.keys())[0])

        def go_to_selected_line():
            line_name = self.goto_line_var.get()
            if line_name:
                self._go_to_line(line_name)

        go_button = ttk.Button(go_to_frame, text="Go", command=go_to_selected_line)
        go_button.pack(fill=tk.X, expand=True, pady=(5,0))
    

    def _update_slider_ranges(self, wv_min: float, wv_max: float):
        """Update slider and spinbox ranges based on wavelength range"""
        for scale in [self.signal_start_scale, self.signal_end_scale,
                     self.noise_start_scale, self.noise_end_scale]:
            scale.config(from_=wv_min, to=wv_max)
        
        for name, spinbox in self.region_spinboxes.items():
            spinbox.config(from_=wv_min, to=wv_max)

    def _update_results_display(self):
        """Update analysis results display in the Treeview"""
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        res = self.current_analysis_results

        if res.get('fit_failed'):
            self.results_tree.insert('', tk.END, values=("Fit Status", "FAILED"))
            self.results_tree.insert('', tk.END, values=("Reason", res.get('fail_reason', 'Unknown')))
        elif not res:
            self.results_tree.insert('', tk.END, values=("Status", "No analysis performed."))
        else:
            fwhm_int = res.get('fwhm_int_A', np.nan)
            fwhm_int_str = f"{fwhm_int:.2f}" if not np.isnan(fwhm_int) else "N/A"
            chi2 = res.get('chi_squared_red', np.nan)
            chi2_str = f"{chi2:.2f}" if not np.isnan(chi2) else "N/A"

            results_data = {
                "SNR": f"{res.get('snr', 0):.2f}",
                "Flux Integral": f"{res.get('flux_integral', 0):.3e}",
                "FWHM (obs, Å)": f"{res.get('fwhm_obs_A', 0):.2f}",
                "FWHM (int, Å)": fwhm_int_str,
                "FWHM (km/s)": f"{res.get('fwhm_kms', 0):.2f}",
                "EW (Å)": f"{res.get('ew', 0):.2f}",
                "χ² reduced": chi2_str,
            }

            for prop, val in results_data.items():
                self.results_tree.insert('', tk.END, values=(prop, val))

    def _load_config(self) -> AppConfig:
        config = configparser.ConfigParser()
        if not os.path.exists(self.config_path):
            app_config = AppConfig()
            self._save_config(app_config)
            return app_config
        config.read(self.config_path)
        return AppConfig.from_configparser(config)

    def _save_config(self, app_config: AppConfig):
        config = app_config.to_configparser()
        with open(self.config_path, 'w') as f:
            config.write(f)
        logging.info("Configuration saved.")

    def _update_regions_from_vars(self, *args):
        self.regions.signal_start = self.signal_start_var.get()
        self.regions.signal_end = self.signal_end_var.get()
        self.regions.noise_start = self.noise_start_var.get()
        self.regions.noise_end = self.noise_end_var.get()
        self._schedule_update()

    def _update_plot_state_from_vars(self, *args):
        self.plot_state.show_continuum = self.show_continuum_var.get()
        self.plot_state.show_original = self.show_original_var.get()
        self.plot_state.show_continuum_sub = self.show_continuum_sub_var.get()
        self._schedule_update()

    def _bind_shortcuts(self):
        self.master.bind('<Right>', lambda e: self._on_next())
        self.master.bind('<Left>', lambda e: self._on_prev())
        self.master.bind('1', lambda e: self._classify_spectrum('[OII]'))
        self.master.bind('2', lambda e: self._classify_spectrum('[OIII]5007'))
        self.master.bind('3', lambda e: self._classify_spectrum('Halpha'))
        self.master.bind('4', lambda e: self._classify_spectrum('Unclear'))
        self.master.bind('5', lambda e: self._classify_spectrum('Noisy/Bad'))

    def _toggle_multi_fit_mode(self):
        is_multi_mode = self.multi_fit_mode_var.get()
        state = 'normal' if is_multi_mode else 'disabled'
        for child in self.multi_fit_controls.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                if hasattr(child, 'winfo_children'):
                    for sub_child in child.winfo_children():
                        try: sub_child.configure(state=state)
                        except tk.TclError: pass
        for child in self.classify_frame.winfo_children():
            try:
                child.configure(state='disabled' if is_multi_mode else 'normal')
            except tk.TclError:
                pass
        self._schedule_update()

    def _on_lock_fit(self):
        if not self.current_analysis_results or self.current_analysis_results.get('fit_failed'):
            messagebox.showwarning("Lock Failed", "No valid fit to lock.")
            return
        fit_data = self.current_analysis_results.copy()
        fit_data['name'] = self.line_name_var.get().strip()
        if not fit_data['name']:
            messagebox.showwarning("Lock Failed", "Please provide a name for the line.")
            return
        self.locked_fits.append(fit_data)
        logging.info(f"Locked fit for '{fit_data['name']}' at {fit_data.get('lambda_center', 0):.2f}Å")
        self._update_locked_fits_display()
        self.line_name_var.set('')
        self._schedule_update()

    def _on_reset_fits(self):
        if not self.locked_fits: return
        if messagebox.askyesno("Confirm Reset", "Clear all locked fits for this spectrum?"):
            self.locked_fits = []
            logging.info(f"Reset all locked fits for {self.current_spectrum_data.source_id}")
            self._update_locked_fits_display()
            self._schedule_update()

    def _update_locked_fits_display(self):
        for item in self.locked_fits_tree.get_children():
            self.locked_fits_tree.delete(item)
        for fit in self.locked_fits:
            self.locked_fits_tree.insert('', tk.END, values=(
                fit.get('name', 'N/A'),
                f"{fit.get('lambda_center', 0):.2f}",
                f"{fit.get('fwhm_obs_A', 0):.2f}",
                f"{fit.get('snr', 0):.1f}"
            ))

    def _guess_line_name(self, fitted_wavelength: float, z: float) -> str:
        if not self.emission_lines: return ""
        min_diff, best_name = float('inf'), ""
        for name, rest_wl in self.emission_lines.items():
            diff = abs(fitted_wavelength - (rest_wl * (1 + z)))
            if diff < min_diff:
                min_diff, best_name = diff, name
        return best_name if min_diff < 20 else ""

    def _on_export_multifit(self):
        if not self.multi_fit_mode_var.get():
            messagebox.showinfo("Info", "This export option is only for Multi-Line Fit Mode.")
            return
        if not self.locked_fits:
            messagebox.showwarning("Warning", "No locked fits to export.")
            return
        default_filename = f"multifit_{self.current_spectrum_data.source_id}.csv"
        save_path = filedialog.asksaveasfilename(
            initialfile=default_filename,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not save_path: return
        df = pd.DataFrame(self.locked_fits)
        cols = ['name', 'lambda_center', 'model_name', 'snr', 'fwhm_obs_A',
                'fwhm_int_A', 'fwhm_kms', 'flux_integral', 'ew', 'chi_squared_red']
        df = df.reindex(columns=[c for c in cols if c in df.columns])
        df.to_csv(save_path, index=False)
        messagebox.showinfo("Success", f"Saved {len(df)} locked fits to '{os.path.basename(save_path)}'")
        logging.info(f"Exported {len(df)} locked fits to {save_path}.")

    def _open_single_spectrum_dialog(self):
        path = filedialog.askopenfilename(
            title="Select Single Spectrum File",
            filetypes=[("FITS files", "*.fits *.fit"), ("All files", "*.*")]
        )
        if not path: return
        try:
            self.df_master = pd.DataFrame({'filename': [os.path.basename(path)], 'full_path': [path]})
            self.master.title(f"Euclid Spectrum Analyser - {os.path.basename(path)}")
            self._initialise_session()
            logging.info(f"Loaded single spectrum: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load spectrum:\n{e}")
            logging.error(f"Failed to load single spectrum: {e}")

    def _open_spectra_folder_dialog(self):
        folder_path = filedialog.askdirectory(title="Select Folder Containing Spectra")
        if not folder_path: return
        try:
            self.df_master = self.loader.load_spectra_from_folder(folder_path)
            self.master.title(f"Euclid Spectrum Analyser - {os.path.basename(folder_path)}")
            self._initialise_session(is_folder_mode=True)
            logging.info(f"Loaded {len(self.df_master)} spectra from folder: {folder_path}")
        except AnalysisError as e:
            messagebox.showwarning("Warning", str(e))
            logging.warning(f"Problem loading folder '{folder_path}': {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
            logging.error(f"Failed to process folder: {e}")

    def _open_master_csv_dialog(self):
        path = filedialog.askopenfilename(title="Select Master CSV File", filetypes=[("CSV files", "*.csv")])
        if not path: return
        try:
            self.df_master = self.loader.load_spectra_from_csv(path)
            self.master.title(f"Euclid Spectrum Analyser - {os.path.basename(path)}")
            self._initialise_session()
            logging.info(f"Loaded master CSV: {path}")
        except AnalysisError as e:
            messagebox.showerror("Error", str(e))
            self.df_master = None
            logging.error(f"Failed to load master CSV: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
            self.df_master = None
            logging.error(f"Failed to process CSV file: {e}")

    def _load_emission_lines_dialog(self):
        path = filedialog.askopenfilename(
            title="Select Emission Line File",
            filetypes=[("Text files", "*.txt *.dat"), ("All files", "*.*")]
        )
        if not path: return
        try:
            new_lines = {}
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            new_lines[parts[0]] = float(parts[1])
            if not new_lines:
                messagebox.showwarning("Warning", "No valid lines found in file.")
                return
            self.emission_lines = new_lines
            self.plotter.update_emission_lines(new_lines)
            self._build_line_widgets()
            self._update_line_info()
            messagebox.showinfo("Success", f"Loaded {len(new_lines)} emission lines.")
            logging.info(f"Loaded custom emission line list from {path}.")
        except Exception as e:
            messagebox.showerror("Error",
                f"Could not parse emission line file.\nEnsure format is: Name Wavelength\n\nError: {e}")
            logging.error(f"Failed to parse emission line file: {e}")

    def _initialise_session(self, is_folder_mode: bool = False):
        self.results_manager = ResultsManager()
        start_index = 0
        output_path = self.app_config.output_csv_path
        if os.path.exists(output_path) and not is_folder_mode and len(self.df_master) > 1:
            if messagebox.askyesno("Resume Session?",
                f"Found existing results file:\n{output_path}\nLoad it and resume?"):
                try:
                    saved_df = pd.read_csv(output_path)
                    for _, row in saved_df.iterrows():
                        self.results_manager.add_result(row.to_dict())
                    last_index = saved_df['index'].max() if not saved_df.empty else -1
                    start_index = int(last_index + 1)
                    logging.info(f"Resuming session from {output_path}, starting at index {start_index}.")
                except Exception as e:
                    logging.error(f"Failed to load previous results: {e}")
        if start_index >= len(self.df_master):
            messagebox.showinfo("Session Complete", "All sources have been classified! Starting at the last source.")
            start_index = len(self.df_master) - 1
        self.idx_spinbox.config(from_=1, to=len(self.df_master))
        self.total_label.config(text=f"/ {len(self.df_master)}")
        self.current_index = start_index
        self._change_source(self.current_index)

    def _change_source(self, new_index: int):
        if self.df_master is None or not (0 <= new_index < len(self.df_master)):
            return
        self.locked_fits = []
        self._update_locked_fits_display()
        self.current_index = new_index
        self.idx_spinbox.set(self.current_index + 1)
        self._start_analysis_thread()

    def _start_analysis_thread(self):
        self._set_ui_state('disabled')
        self._update_status_transient("Loading and analysing...")
        self.analysis_thread = threading.Thread(
            target=self._worker_load_and_analyse,
            args=(self.current_index, self.z_var.get())
        )
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def _worker_load_and_analyse(self, index_to_load: int, z_guess: float):
        try:
            row = self.df_master.iloc[index_to_load]
            path = row['full_path']
            source_id = row.get('filename', os.path.basename(path))
            dataframe = Table.read(path, format='fits').to_pandas()
            available_columns = list(dataframe.columns)
            z_col = self.app_config.default_redshift_col
            initial_z = row.get(z_col, z_guess)
            self.master.after(0, self._on_load_complete, {
                'dataframe': dataframe,
                'columns': available_columns,
                'source_id': source_id,
                'initial_z': initial_z,
                'error': None
            })
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error in worker thread for index {index_to_load}: {error_msg}")
            self.master.after(0, self._on_load_complete, {'error': error_msg})

    def _on_load_complete(self, result: Dict):
        if result.get('error'):
            messagebox.showerror("Error", f"Failed to load spectrum:\n{result['error']}")
            self._set_ui_state('normal')
            self._update_status()
            return
        dataframe = result['dataframe']
        available_columns = result['columns']
        source_id = result['source_id']
        initial_z = result['initial_z']
        column_map = {
            'wavelength': self.app_config.wavelength_col, 'flux': self.app_config.flux_col,
            'flux_error': self.app_config.flux_error_col, 'lsf_variance': self.app_config.lsf_variance_col,
        }
        if (column_map['wavelength'] not in available_columns or column_map['flux'] not in available_columns):
            dialog = ColumnMappingDialog(self.master, available_columns, column_map)
            self.master.wait_window(dialog)
            if dialog.result is None:
                self._set_ui_state('normal')
                self._update_status()
                return
            column_map = dialog.result
            if dialog.save_to_config.get():
                self.app_config.wavelength_col = column_map['wavelength']
                self.app_config.flux_col = column_map['flux']
                self.app_config.flux_error_col = column_map.get('flux_error', '')
                self.app_config.lsf_variance_col = column_map.get('lsf_variance', '')
                self._save_config(self.app_config)
        try:
            spectrum_data = self.loader._extract_data(dataframe, column_map, source_id)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract data after mapping:\n{e}")
            self._set_ui_state('normal')
            self._update_status()
            return
        if spectrum_data.error:
            messagebox.showerror("Error", spectrum_data.error)
            self._set_ui_state('normal')
            return
        self.current_spectrum_data = spectrum_data
        if not self.current_spectrum_data.wv.size:
            messagebox.showwarning("Warning", "Loaded spectrum has no valid data points.")
            self._set_ui_state('normal')
            return
        wv_min, wv_max = self.current_spectrum_data.wv.min(), self.current_spectrum_data.wv.max()
        self._update_slider_ranges(wv_min, wv_max)
        self.z_var.set(initial_z)
        self._set_smart_regions(initial_z)
        self._update_display()
        self._set_ui_state('normal')
        self._update_status()

    def _set_smart_regions(self, z):
        if not self.current_spectrum_data.wv.size: return
        wv, flux = self.current_spectrum_data.wv, self.current_spectrum_data.flux
        wv_min, wv_max = wv.min(), wv.max()
        median_flux, std_flux = np.median(flux), np.std(flux)
        peaks, props = find_peaks(flux, height=median_flux + AnalysisDefaults.PEAK_HEIGHT_THRESHOLD_SIGMA * std_flux, distance=AnalysisDefaults.PEAK_MIN_DISTANCE)
        center_wv = wv[peaks[np.argmax(props['peak_heights'])]] if len(peaks) > 0 else np.mean(wv)
        self.signal_start_var.set(max(wv_min, center_wv - AnalysisDefaults.SIGNAL_WINDOW_WIDTH))
        self.signal_end_var.set(min(wv_max, center_wv + AnalysisDefaults.SIGNAL_WINDOW_WIDTH))
        self.noise_start_var.set(min(wv_max, center_wv + AnalysisDefaults.NOISE_OFFSET))
        self.noise_end_var.set(min(wv_max, center_wv + AnalysisDefaults.NOISE_OFFSET + AnalysisDefaults.NOISE_WINDOW_WIDTH))

    def _schedule_update(self, event=None):
        if self._update_job_id:
            self.master.after_cancel(self._update_job_id)
        self._update_job_id = self.master.after(AnalysisDefaults.UPDATE_DEBOUNCE_MS, self._update_display)

    def _update_display(self):
        self._analyse_spectrum()
        self._update_plots()
        self._update_status()

    def _analyse_spectrum(self):
        if not self.current_spectrum_data.wv.size:
            self.current_analysis_results = {}
            return
        try:
            wv, flux = self.current_spectrum_data.wv, self.current_spectrum_data.flux
            continuum_mask = ((wv < self.regions.signal_start - AnalysisDefaults.CONTINUUM_BUFFER) |
                              (wv > self.regions.signal_end + AnalysisDefaults.CONTINUUM_BUFFER))
            self.continuum_model = self.analyser.continuum_fitter.fit_with_sigma_clip(wv, flux, continuum_mask)
            corrected_flux = flux - self.continuum_model(wv)
            self.current_spectrum_data.continuum = self.continuum_model(wv)
            self.current_spectrum_data.corrected_flux = corrected_flux
            flux_to_analyse = corrected_flux.copy()
            if self.multi_fit_mode_var.get() and self.locked_fits:
                for locked_fit in self.locked_fits:
                    model_name = locked_fit.get('model_name', 'Gaussian')
                    locked_model = None
                    if model_name == 'Gaussian':
                        locked_model = Gaussian1D(amplitude=locked_fit['amplitude'], mean=locked_fit['lambda_center'], stddev=locked_fit.get('stddev_A', locked_fit['fwhm_obs_A']/2.355))
                    elif model_name == 'Lorentzian':
                        locked_model = Lorentz1D(amplitude=locked_fit['amplitude'], x_0=locked_fit['lambda_center'], fwhm=locked_fit['fwhm_obs_A'])
                    elif model_name == 'Voigt':
                        locked_model = Voigt1D(amplitude_L=locked_fit.get('amplitude_L', locked_fit['amplitude']), x_0=locked_fit['lambda_center'], fwhm_L=locked_fit.get('fwhm_L', 5), fwhm_G=locked_fit.get('fwhm_G', 5))
                    if locked_model:
                        flux_to_analyse -= locked_model(wv)
            results = self.analyser.analyse(self.current_spectrum_data, self.regions, self.fit_model_var.get(), self.continuum_model, flux_to_analyse)
            self.current_spectrum_data.g_fit_model = results['model']
            results['z'] = self.z_var.get()
            results['index'] = self.current_index
            extra_cols = {c: self.df_master.iloc[self.current_index].get(c) for c in self.df_master.columns if c not in ['filename', 'full_path']}
            results.update(extra_cols)
            self.current_analysis_results = results
            g_fit = self.current_spectrum_data.g_fit_model
            model_name = self.fit_model_var.get()
            fit_params = {'model_name': model_name}
            if isinstance(g_fit, Gaussian1D):
                fit_params.update({'amplitude': get_param_value(g_fit.amplitude), 'stddev_A': get_param_value(g_fit.stddev)})
            elif isinstance(g_fit, Lorentz1D):
                fit_params.update({'amplitude': get_param_value(g_fit.amplitude)})
            elif isinstance(g_fit, Voigt1D):
                fit_params.update({'amplitude_L': get_param_value(g_fit.amplitude_L), 'fwhm_L': get_param_value(g_fit.fwhm_L), 'fwhm_G': get_param_value(g_fit.fwhm_G)})
            lambda_center = results['lambda_center']
            suggested_name = self._guess_line_name(lambda_center, self.z_var.get())
            if suggested_name and not self.line_name_var.get():
                self.line_name_var.set(suggested_name)
            self.current_analysis_results.update(fit_params)
        except AnalysisError as e:
            self.current_analysis_results = {'fit_failed': True, 'fail_reason': str(e), 'source_id': self.current_spectrum_data.source_id}
            logging.warning(f"Analysis failed: {e}")
        except Exception as e:
            self.current_analysis_results = {'fit_failed': True, 'fail_reason': 'Unexpected error', 'source_id': self.current_spectrum_data.source_id}
            logging.error(f"Unexpected error in analysis: {e}", exc_info=True)

    def _update_plots(self):
        self.plotter.plot_spectrum(self.current_spectrum_data, self.plot_state, self.regions, self.z_var.get(), self.current_analysis_results, self.locked_fits)
        self._update_line_info()
        self._update_results_display()

    def _update_line_info(self):
        line_info_str = "".join([f"{name:<12}: {rest_wl * (1 + self.z_var.get()):.2f}\n" for name, rest_wl in self.emission_lines.items()])
        self.line_info_text.config(state=tk.NORMAL)
        self.line_info_text.delete('1.0', tk.END)
        self.line_info_text.insert(tk.END, line_info_str)
        self.line_info_text.config(state=tk.DISABLED)

    def _update_status(self):
        if self.df_master is None:
            self.status_var.set("No data loaded. Open a file or folder from the 'File' menu.")
            return
        total = len(self.df_master)
        source_id = self.current_spectrum_data.source_id or 'N/A'
        classification = next((r['classification'] for r in self.results_manager.get_results_list() if str(r.get('source_id')) == str(source_id)), "Unclassified")
        self.status_label.config(foreground=CLASSIFICATION_COLOURS.get(classification, 'black'))
        status_msg = (f"Viewing {self.current_index + 1}/{total} | "
                      f"ID: {source_id} ({classification}) | "
                      f"Classified: {len(self.results_manager.get_results_list())}")
        if self.multi_fit_mode_var.get():
            status_msg += f" | Locked Fits: {len(self.locked_fits)}"
        self.status_var.set(status_msg)
        self.prev_button['state'] = tk.NORMAL if self.current_index > 0 else tk.DISABLED
        self.next_button['state'] = tk.NORMAL if self.current_index < total - 1 else tk.DISABLED

    def _update_status_transient(self, message):
        self.status_label.config(foreground='blue')
        self.status_var.set(message)

    def _set_ui_state(self, state):
        for widget in [self.prev_button, self.next_button, self.idx_spinbox]:
            widget.config(state=state)
        self.zoom_selector.set_active(state == 'normal')

    def _on_next(self):
        if self.df_master is not None and self.current_index < len(self.df_master) - 1:
            self._change_source(self.current_index + 1)

    def _on_prev(self):
        if self.current_index > 0:
            self._change_source(self.current_index - 1)

    def _on_goto(self):
        try:
            req_idx = int(self.idx_spinbox.get()) - 1
            if 0 <= req_idx < len(self.df_master):
                self._change_source(req_idx)
        except ValueError: pass

    def _go_to_line(self, line_name):
        if not self.current_spectrum_data.wv.size: return
        obs_wl = self.emission_lines[line_name] * (1 + self.z_var.get())
        self.signal_start_var.set(obs_wl - AnalysisDefaults.SIGNAL_WINDOW_WIDTH)
        self.signal_end_var.set(obs_wl + AnalysisDefaults.SIGNAL_WINDOW_WIDTH)
        logging.info(f"Jumping to {line_name} at {obs_wl:.2f} Å.")
        self.ax2.set_xlim(obs_wl - 200, obs_wl + 200)
        self._schedule_update()

    def _on_zoom_select(self, eclick, erelease):
        x1, x2 = sorted((eclick.xdata, erelease.xdata))
        self.ax2.set_xlim(x1, x2)
        self.canvas.draw()

    def _toggle_z_finder_mode(self):
        if self.z_finder_mode.get():
            self.z_finder_cid = self.canvas.mpl_connect('button_press_event', self._on_plot_click_for_z)
            self._update_status_transient("Redshift Finder Mode ON: Click a peak on the top plot.")
        else:
            if hasattr(self, 'z_finder_cid') and self.z_finder_cid:
                self.canvas.mpl_disconnect(self.z_finder_cid)
            self._update_status()

    def _on_plot_click_for_z(self, event):
        if event.inaxes != self.ax1: return
        obs_wave = event.xdata
        popup = tk.Toplevel(self.master)
        popup.title("Identify Line")
        ttk.Label(popup, text=f"Identify the line at ~{obs_wave:.2f} Å:").pack(padx=20, pady=10)
        def set_z(line_name):
            new_z = (obs_wave / self.emission_lines[line_name]) - 1
            self.z_var.set(round(new_z, 4))
            logging.info(f"Redshift updated to {new_z:.4f} by identifying {line_name} at {obs_wave:.2f} Å.")
            popup.destroy()
            self._schedule_update()
        for name in self.emission_lines.keys():
            ttk.Button(popup, text=name, command=lambda n=name: set_z(n)).pack(fill=tk.X, padx=10, pady=2)
        popup.transient(self.master)
        popup.grab_set()

    def _classify_spectrum(self, label):
        if self.multi_fit_mode_var.get():
            messagebox.showwarning("Warning", "Cannot classify in Multi-Line Fit Mode. Use 'Fit & Lock Line' instead.")
            return
        if not self.current_analysis_results: return
        source_id = self.current_analysis_results.get('source_id')
        if not source_id: return
        result = self.current_analysis_results.copy() if label != 'Noisy/Bad' else {'index': self.current_index, 'source_id': source_id}
        result.update({'classification': label, 'comment': self.comment_var.get()})
        self.results_manager.add_result(result)
        logging.info(f"Classified {source_id} as '{label}'.")
        self.comment_var.set('')
        self._on_next()

    def _on_undo(self):
        last_result = self.results_manager.undo_last()
        if not last_result: return
        undo_index = last_result.get('index', self.current_index - 1)
        self._change_source(undo_index)
        logging.info(f"Undid classification for {last_result.get('source_id')}.")

    def _on_export(self):
        try:
            save_path = filedialog.asksaveasfilename(
                initialfile=os.path.basename(self.app_config.output_csv_path),
                defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if not save_path: return
            num_results = self.results_manager.export_to_csv(save_path)
            messagebox.showinfo("Success", f"Saved {num_results} results to '{os.path.basename(save_path)}'")
            logging.info(f"Exported {num_results} results to {save_path}.")
        except ValueError as e:
            messagebox.showwarning("Warning", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export:\n{e}")
            logging.error(f"Export failed: {e}")

    def _open_settings_dialog(self):
        settings_win = tk.Toplevel(self.master)
        settings_win.title("Settings")
        settings_vars = {
            'master_csv_path': tk.StringVar(value=self.app_config.master_csv_path),
            'spectra_folder': tk.StringVar(value=self.app_config.spectra_folder),
            'output_csv_path': tk.StringVar(value=self.app_config.output_csv_path),
            'wavelength_col': tk.StringVar(value=self.app_config.wavelength_col),
            'flux_col': tk.StringVar(value=self.app_config.flux_col),
            'flux_error_col': tk.StringVar(value=self.app_config.flux_error_col),
            'lsf_variance_col': tk.StringVar(value=self.app_config.lsf_variance_col),
            'default_redshift_col': tk.StringVar(value=self.app_config.default_redshift_col)
        }
        frame = ttk.Frame(settings_win, padding="10")
        frame.pack(expand=True, fill="both")
        labels = {
            'master_csv_path': "Master CSV Path:", 'spectra_folder': "Spectra Folder:",
            'output_csv_path': "Output CSV Path:", 'wavelength_col': "FITS Wavelength Column:",
            'flux_col': "FITS Flux Column:", 'flux_error_col': "FITS Flux Error Column (optional):",
            'lsf_variance_col': "FITS LSF Variance Column (optional):", 'default_redshift_col': "CSV Redshift Column:"
        }
        for i, (key, label) in enumerate(labels.items()):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(frame, textvariable=settings_vars[key], width=50).grid(row=i, column=1, sticky="ew")
        def save_and_close():
            self.app_config.master_csv_path = settings_vars['master_csv_path'].get()
            self.app_config.spectra_folder = settings_vars['spectra_folder'].get()
            self.app_config.output_csv_path = settings_vars['output_csv_path'].get()
            self.app_config.wavelength_col = settings_vars['wavelength_col'].get()
            self.app_config.flux_col = settings_vars['flux_col'].get()
            self.app_config.flux_error_col = settings_vars['flux_error_col'].get()
            self.app_config.lsf_variance_col = settings_vars['lsf_variance_col'].get()
            self.app_config.default_redshift_col = settings_vars['default_redshift_col'].get()
            self._save_config(self.app_config)
            settings_win.destroy()
            messagebox.showinfo("Settings Saved", "Configuration has been saved to config.ini.")
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=len(labels), columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Save", command=save_and_close).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=settings_win.destroy).pack(side="left", padx=5)
        settings_win.transient(self.master)
        settings_win.grab_set()

    def _on_closing(self):
        if messagebox.askokcancel("Quit", "Do you really want to quit?"):
            logging.info("Application closed.")
            self.master.quit()
            self.master.destroy()

def show_tutorial_if_needed(root, config_path: str):
    """Show tutorial on first run"""
    config = configparser.ConfigParser()
    config.read(config_path)
    if config.getboolean('USER_PREFERENCES', 'show_tutorial', fallback=True):
        wizard = TutorialWizard(root, config_path)
        root.wait_window(wizard.top)

def main():
    """Main entry point"""
    root = tk.Tk()
    app = SpectrumAnalyserGUI(root)
    root.after(100, lambda: show_tutorial_if_needed(root, app.config_path))
    root.mainloop()

if __name__ == "__main__":
    main()
