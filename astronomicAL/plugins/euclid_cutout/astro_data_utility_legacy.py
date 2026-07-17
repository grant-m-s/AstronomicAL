from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.esa.euclid import EuclidClass
import mocpy

from astronomicAL.utils.error_tracker import ErrorTracker


DEFAULT_EUCLID_FILTERS = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]
DEFAULT_SAVE_DIR = "data/cutouts"
DEFAULT_MOC_PATH = "data/mocs"


class EuclidCutoutsClass:
    """Retrieve raw Euclid image cutouts from the ESA archive.

    This class is intentionally data-only. It performs archive access, FITS
    download, FITS reading, and WCS extraction. It does not stretch images,
    reproject bands, build RGB images, scale/clip arrays, convert coordinates
    for plotting, or create HoloViews objects. Those steps belong in
    ``image_visualization.ImageVisualizationClass`` and the panel layer.

    The constructor keeps the plugin-version signature so the runtime can
    instantiate it as::

        EuclidCutoutsClass(ra=ra, dec=dec, euclid_filters=filters,
                           save_dir=save_dir, context=context)
    """

    def __init__(
        self,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        *,
        client: Any = None,
        euclid_filters: Optional[Iterable[str]] = None,
        save_dir: str = DEFAULT_SAVE_DIR,
        context: Any = None,
        check_moc_coverage: bool = False,
        moc_survey: str = "Euclid_Q1",
        moc_path: str = DEFAULT_MOC_PATH,
    ) -> None:
        self.context = context
        self.save_dir = str(Path(save_dir).expanduser())
        os.makedirs(self.save_dir, exist_ok=True)

        self.error_tracker = ErrorTracker()
        self.check_moc_coverage = bool(check_moc_coverage)
        self.moc_survey = moc_survey
        self.moc_path = moc_path
        self.moc = None
        if self.check_moc_coverage:
            try:
                self.moc = load_moc(self.moc_survey, path=self.moc_path)
            except Exception as exc:
                # Missing local MOC files should not make the whole plugin
                # unusable. The archive query can still fail gracefully later.
                self.error_tracker.log_error(exc, "Could not load Euclid MOC coverage file")
                self.moc = None

        self.euclid_filters = list(euclid_filters or DEFAULT_EUCLID_FILTERS)

        self.client = client if client is not None else self._get_or_create_client()
        self.coordinates: Optional[SkyCoord] = None
        if ra is not None and dec is not None:
            self.set_coordinates(ra=ra, dec=dec)

    # ------------------------------------------------------------------
    # Client / context helpers
    # ------------------------------------------------------------------
    def _get_or_create_client(self) -> Any:
        services = getattr(self.context, "services", None) if self.context is not None else None
        key = "euclid.client"
        if services is not None:
            try:
                if services.has(key):
                    return services.get(key)
            except Exception:
                pass

        client = EuclidClass(environment=self.environment)
        print("Initialized EuclidClass")

        if services is not None:
            for method_name in ("set", "register", "put"):
                method = getattr(services, method_name, None)
                if callable(method):
                    try:
                        method(key, client)
                        break
                    except Exception:
                        pass
        return client

    def change_environment(
        self,
        environment: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        credentials_filepath: Optional[str] = None,
    ) -> None:
        """Switch the Euclid archive environment and optionally log in."""
        assert environment in ("PDR", "IDR", "OTF", "REG"), (
            "environment must be 'PDR', 'IDR', 'OTF', or 'REG'"
        )

        self.client = EuclidClass(environment=environment)
 
        if environment != "PDR":
            if credentials_filepath is not None:
                self.client.login(user=None, password=None, credentials_file=credentials_filepath)
            else:
                self.client.login(user=user, password=password, credentials_file=None)

        services = getattr(self.context, "services", None) if self.context is not None else None
        if services is not None:
            for method_name in ("set", "register", "put"):
                method = getattr(services, method_name, None)
                if callable(method):
                    try:
                        method("euclid.client", self.client)
                        break
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # State / coordinates
    # ------------------------------------------------------------------
    def reset_data(self, ra: Optional[float] = None, dec: Optional[float] = None) -> None:
        """Reset source-specific attributes, optionally changing coordinates."""
        if ra is not None and dec is not None:
            self.set_coordinates(ra=ra, dec=dec)
        self.error_tracker.reset()
        self._remove_source_attributes()

    def _remove_source_attributes(self) -> None:
        """Remove attributes generated for a specific source."""
        attributes = [
            "cone_results",
            "cutout_radius",
            "cutouts_paths",
            "data",
            "wcs",
            "headers",
        ]
        for attribute in attributes:
            if hasattr(self, attribute):
                delattr(self, attribute)

    def set_coordinates(
        self,
        coordinates: Optional[SkyCoord] = None,
        *,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
    ) -> None:
        if coordinates is not None:
            self.coordinates = coordinates
        elif (ra is not None) and (dec is not None):
            self.coordinates = SkyCoord(ra, dec, unit="deg", frame="icrs")
        else:
            raise ValueError("Either coordinates or ra and dec must be provided")

    # ------------------------------------------------------------------
    # Archive query / download / FITS read
    # ------------------------------------------------------------------
    def get_cone(
        self,
        initial_radius: u.Quantity = 0.5 * u.degree,
        async_job: bool = False,
        verbose: bool = True,
        Nattempts_max: int = 2,
    ) -> None:
        """Run a cone search on the Euclid mosaic-product table."""
        if self.coordinates is None:
            raise ValueError("No coordinates set — provide ra and dec first")

        try:
            tic = time.perf_counter()
            self.cone_results = None
            for attempt in range(Nattempts_max):
                radius = initial_radius * (1 + attempt)
                job = self.client.cone_search(
                    self.coordinates,
                    radius,
                    table_name="sedm.mosaic_product",
                    ra_column_name="ra",
                    dec_column_name="dec",
                    columns="*",
                    async_job=async_job,
                )
                self.cone_results = job.get_results()
                if len(self.cone_results) > 0:
                    break

            if verbose:
                toc = time.perf_counter()
                print(f"Cone search required {toc - tic:.3f} seconds")
                if self.cone_results is not None and len(self.cone_results) == 0:
                    print(
                        f"No cone-search results found after {Nattempts_max} attempts "
                        f"(max radius: {initial_radius * Nattempts_max})"
                    )

        except ConnectionError as exc:
            self.error_tracker.log_error(exc, "Failed to connect to ESA Science Archive")
        except Exception as exc:
            self.error_tracker.log_error(exc, "Euclid cone search failed")

    @staticmethod
    def get_info_cutout(cone_results: Any, filter_name: str) -> Tuple[str, Any, Any]:
        """Return archive file path, instrument, and observation/tile id for one band."""
        matching = cone_results[cone_results["filter_name"] == filter_name]
        if len(matching) == 0:
            raise KeyError(f"No Euclid cone-search product found for filter {filter_name!r}")
        try:
            matching.sort("processing_mode" ) #First DEEP then WIDE in DR1
        except ValueError:
            #No processing_mode column in Q1 table
            pass
        line = matching[0]
        file_path = os.path.join(str(line["file_path"]), str(line["file_name"]))
        instrument = line["instrument_name"]
        obs_id = line["tile_index"]
        return file_path, instrument, obs_id

    def get_band_cutout(self, band: str, fname: Optional[str] = None) -> Optional[str]:
        """Download one band and return the local FITS path."""
        if self.coordinates is None:
            raise ValueError("No coordinates set — provide ra and dec first")
        if not hasattr(self, "cone_results"):
            raise RuntimeError("Run get_cone before downloading cutouts")
        if not hasattr(self, "cutout_radius"):
            raise RuntimeError("cutout_radius is not set; call download_cutouts")

        try:
            file_path, instrument, obs_id = self.get_info_cutout(self.cone_results, band)
            stem = f"{obs_id}_{band}" if fname is None else f"{fname}_{band}"
            output_file = os.path.join(self.save_dir, f"{stem}.fits")

            result = self.client.get_cutout(
                file_path=file_path,
                instrument=instrument,
                id=obs_id,
                coordinate=self.coordinates,
                radius=self.cutout_radius,
                output_file=output_file,
            )
            return result[0] if result else output_file

        except ConnectionError as exc:
            self.error_tracker.log_error(exc, "Failed to connect to ESA Science Archive")
        except Exception as exc:
            self.error_tracker.log_error(exc, f"Failed to download Euclid {band} cutout")
        return None


    def download_cutouts(
        self,
        radius: float,
        verbose: bool = False,
        bands_to_retrieve: Optional[Iterable[str]] = None,
    ) -> Dict[str, str]:
        """Download FITS cutouts for the requested bands."""
        self.cutout_radius = float(radius) * u.arcsec
        self.cutouts_paths: Dict[str, str] = {}

        bands = list(bands_to_retrieve or self.euclid_filters)
        tic = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.get_band_cutout, band, fname="tmp"): band
                for band in bands
            }
            for future in concurrent.futures.as_completed(futures):
                band = futures[future]
                try:
                    save_path = future.result()
                except Exception as exc:
                    self.error_tracker.log_error(exc, f"Failed to download Euclid {band} cutout")
                    continue
                if save_path is not None:
                    self.cutouts_paths[band] = str(save_path)

        if verbose:
            toc = time.perf_counter()
            print(f"Retrieving all cutouts required {toc - tic:.3f} seconds")
            # BUG: Seems like being held by scatter...

        return self.cutouts_paths

    def read_cutouts(self) -> Tuple[Dict[str, np.ndarray], Dict[str, WCS]]:
        """Read downloaded FITS cutouts into raw arrays and WCS objects."""
        self.data: Dict[str, np.ndarray] = {}
        self.wcs: Dict[str, WCS] = {}
        self.headers: Dict[str, Any] = {}

        for band, path in getattr(self, "cutouts_paths", {}).items():
            try:
                with fits.open(path) as hdul:
                    hdu = hdul[0]
                    self.data[band] = np.asarray(hdu.data)
                    self.headers[band] = hdu.header.copy()
                    self.wcs[band] = WCS(hdu.header)

            except OSError as exc:
                self.error_tracker.log_error(exc, f"Downloaded corrupted FITS file for {band}")
            except Exception as exc:
                self.error_tracker.log_error(exc, f"Could not read Euclid {band} FITS cutout")

        return self.data, self.wcs

    def get_cutouts(
        self,
        radius: float = 5,
        *,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        bands_to_retrieve: Optional[Iterable[str]] = None,
        verbose: bool = False,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, WCS]]]:
        """Download and read raw Euclid cutouts, returning ``(data, wcs)``.

        This is the only high-level public retrieval method. It performs:

        1. optional coordinate update;
        2. optional MOC coverage check;
        3. archive cone search;
        4. FITS cutout download;
        5. FITS reading into raw arrays and WCS objects.

        It intentionally does not stretch, clip, scale, reproject for display,
        build RGB images, or create plotting objects.
        """
        self.error_tracker.reset()

        if ra is not None or dec is not None:
            if ra is None or dec is None:
                raise ValueError("Both ra and dec must be provided together")
            self.set_coordinates(ra=ra, dec=dec)

        if self.coordinates is None:
            raise ValueError("No coordinates set — provide ra and dec")

        if self.check_moc_coverage and self.moc is not None:
            inside = check_isin_survey(
                ra=self.coordinates.ra.value,
                dec=self.coordinates.dec.value,
                moc=self.moc,
            )
            if not inside:
                self.error_tracker.log_error(
                    "Source not in the survey",
                    "The selected source is outside the survey coverage area",
                )
                return None, None

        self.get_cone(verbose=verbose, async_job=False)
        if self.error_tracker.has_error:
            return None, None

        if not hasattr(self, "cone_results") or self.cone_results is None or len(self.cone_results) <= 0:
            self.error_tracker.log_error(
                "Cone search failed",
                "No Euclid mosaic products found within the search radius",
            )
            return None, None

        self.download_cutouts(
            radius=radius,
            verbose=verbose,
            bands_to_retrieve=bands_to_retrieve,
        )
        if self.error_tracker.has_error:
            return None, None

        self.read_cutouts()
        if self.error_tracker.has_error:
            return None, None

        return self.data, self.wcs

    # ------------------------------------------------------------------
    # Export / cleanup helpers
    # ------------------------------------------------------------------
    def export_cutouts_to_fits(
        self,
        bands_to_export: Iterable[str],
        directory_path: str = "data/saved_sources",
    ) -> None:
        """Copy downloaded cutouts to a target directory as FITS files."""
        os.makedirs(directory_path, exist_ok=True)

        for band in bands_to_export:
            try:
                path = self.cutouts_paths[band]
                with fits.open(path) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    hdu = fits.PrimaryHDU(data=data, header=header)
                    filename = f"{band}_cutout.fits"
                    fits.HDUList([hdu]).writeto(
                        os.path.join(directory_path, filename),
                        overwrite=True,
                    )
            except KeyError:
                print(f"The required band is not available: {band}")
            except OSError as exc:
                print(exc)
            except FileNotFoundError as exc:
                missing = getattr(self, "cutouts_paths", {}).get(band, "<unknown>")
                print(f"I could not find {missing}\n{exc}")

    def clean_space(self) -> None:
        """Free Euclid archive async-job quota."""
        try:
            joblist = self.client.list_async_jobs()
            to_remove = [job.jobid for job in joblist]
            if to_remove:
                self.client.remove_jobs(to_remove)
        except Exception:
            return


# ----------------------------------------------------------------------
# Small module helpers
# ----------------------------------------------------------------------
def load_moc(survey: str, path: str = DEFAULT_MOC_PATH) -> mocpy.MOC:
    surveys = {
        "Euclid_Q1": "Euclid_Q1_color.fits",
        "Euclid_DR1": "Euclid_Q1_color.fits",
        "DESI": "DESI_from_query.fits",
        "SDSS": "SDSS_color.fits",
        "VLASS": "VLASS_QL.fits",
        "LoTSS": "LoTSS_dr2.fits",
    }
    assert survey in surveys, f"No MOC file available for {survey}"
    return mocpy.MOC.from_fits(os.path.join(path, surveys[survey]))


def check_isin_survey(ra: float, dec: float, moc: mocpy.MOC) -> bool:
    value = moc.contains_lonlat(ra * u.deg, dec * u.deg)
    return bool(np.asarray(value).item())
