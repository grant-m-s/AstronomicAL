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

from uuid import uuid4

from astronomicAL.utils.error_tracker import ErrorTracker


DEFAULT_EUCLID_FILTERS = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]
DEFAULT_SAVE_DIR = "data/cutouts"
DEFAULT_MOC_PATH = "data/mocs"


def _cancel_requested(cancel_token: Any) -> bool:
    if cancel_token is None:
        return False

    for name in ("cancelled", "is_cancelled", "is_cancelled_requested"):
        value = getattr(cancel_token, name, None)

        if callable(value):
            try:
                if bool(value()):
                    return True
            except Exception:
                pass
        elif value is not None:
            try:
                if bool(value):
                    return True
            except Exception:
                pass

    return False


def _raise_if_cancelled(cancel_token: Any) -> None:
    if _cancel_requested(cancel_token):
        raise concurrent.futures.CancelledError(
            "Euclid cutout request was superseded."
        )

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
        check_moc_coverage: bool = True,
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

        client = EuclidClass(environment="PDR")
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
        cancel_token: Any = None,
    ) -> None:
        """Run a cone search on the Euclid mosaic-product table."""

        if self.coordinates is None:
            raise ValueError("No coordinates set — provide ra and dec first")

        try:
            _raise_if_cancelled(cancel_token)

            tic = time.perf_counter()
            self.cone_results = None

            for attempt in range(Nattempts_max):
                _raise_if_cancelled(cancel_token)

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

                _raise_if_cancelled(cancel_token)
                self.cone_results = job.get_results()
                _raise_if_cancelled(cancel_token)

                if len(self.cone_results) > 0:
                    break

            if verbose:
                print(
                    "Cone search required "
                    f"{time.perf_counter() - tic:.3f} seconds"
                )

            if self.cone_results is not None and len(self.cone_results) == 0:
                print(
                    f"No cone-search results found after {Nattempts_max} "
                    f"attempts (max radius: "
                    f"{initial_radius * Nattempts_max})"
                )

        except concurrent.futures.CancelledError:
            raise
        except ConnectionError as exc:
            self.error_tracker.log_error(
                exc,
                "Failed to connect to ESA Science Archive",
            )
        except Exception as exc:
            self.error_tracker.log_error(
                exc,
                "Euclid cone search failed",
            )

    @staticmethod
    def get_info_cutout(cone_results: Any, filter_name: str) -> Tuple[str, Any, Any]:
        """Return archive file path, instrument, and observation/tile id for one band."""
        matching = cone_results[cone_results["filter_name"] == filter_name]
        if len(matching) == 0:
            raise KeyError(f"No Euclid cone-search product found for filter {filter_name!r}")

        line = matching[0]
        file_path = os.path.join(str(line["file_path"]), str(line["file_name"]))
        instrument = line["instrument_name"]
        obs_id = line["tile_index"]
        return file_path, instrument, obs_id

    def get_band_cutout(
        self,
        band: str,
        fname: Optional[str] = None,
        cancel_token: Any = None,
    ) -> Optional[str]:
        """Download one band and return its isolated local FITS path."""

        if self.coordinates is None:
            raise ValueError("No coordinates set — provide ra and dec first")
        if not hasattr(self, "cone_results"):
            raise RuntimeError("Run get_cone before downloading cutouts")
        if not hasattr(self, "cutout_radius"):
            raise RuntimeError(
                "cutout_radius is not set; call download_cutouts"
            )

        output_file: Optional[str] = None
        downloaded_path: Optional[str] = None

        try:
            _raise_if_cancelled(cancel_token)

            file_path, instrument, obs_id = self.get_info_cutout(
                self.cone_results,
                band,
            )

            stem = f"{obs_id}_{band}" if fname is None else f"{fname}_{band}"
            output_file = os.path.join(
                self.save_dir,
                f"{stem}.fits",
            )

            result = self.client.get_cutout(
                file_path=file_path,
                instrument=instrument,
                id=obs_id,
                coordinate=self.coordinates,
                radius=self.cutout_radius,
                output_file=output_file,
            )

            if isinstance(result, (list, tuple)) and result:
                downloaded_path = str(result[0])
            elif result:
                downloaded_path = str(result)
            else:
                downloaded_path = output_file

            _raise_if_cancelled(cancel_token)
            return downloaded_path

        except concurrent.futures.CancelledError:
            for path in {output_file, downloaded_path}:
                if not path:
                    continue
                try:
                    Path(path).unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            raise

        except ConnectionError as exc:
            self.error_tracker.log_error(
                exc,
                "Failed to connect to ESA Science Archive",
            )
        except Exception as exc:
            self.error_tracker.log_error(
                exc,
                f"Failed to download Euclid {band} cutout",
            )

        return None


    def download_cutouts(
        self,
        radius: float,
        verbose: bool = False,
        bands_to_retrieve: Optional[Iterable[str]] = None,
        cancel_token: Any = None,
    ) -> Dict[str, str]:
        """Download requested bands into the request's isolated directory."""

        self.cutout_radius = float(radius) * u.arcsec
        self.cutouts_paths = {}

        bands = list(
            dict.fromkeys(
                bands_to_retrieve or self.euclid_filters
            )
        )
        if not bands:
            return self.cutouts_paths

        tic = time.perf_counter()
        _raise_if_cancelled(cancel_token)

        # Avoid a nested executor for the normal selected-band path.
        if len(bands) == 1:
            band = bands[0]
            save_path = self.get_band_cutout(
                band,
                cancel_token=cancel_token,
            )
            if save_path is not None:
                self.cutouts_paths[band] = str(save_path)

        else:
            max_workers = min(4, len(bands))

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
            ) as executor:
                futures = {
                    executor.submit(
                        self.get_band_cutout,
                        band,
                        cancel_token=cancel_token,
                    ): band
                    for band in bands
                }

                try:
                    for future in concurrent.futures.as_completed(futures):
                        _raise_if_cancelled(cancel_token)
                        band = futures[future]

                        try:
                            save_path = future.result()
                        except concurrent.futures.CancelledError:
                            raise
                        except Exception as exc:
                            self.error_tracker.log_error(
                                exc,
                                f"Failed to download Euclid {band} cutout",
                            )
                            continue

                        if save_path is not None:
                            self.cutouts_paths[band] = str(save_path)

                except concurrent.futures.CancelledError:
                    for future in futures:
                        future.cancel()
                    raise

        _raise_if_cancelled(cancel_token)

        if verbose:
            print(
                "Retrieving all cutouts required "
                f"{time.perf_counter() - tic:.3f} seconds"
            )

        return self.cutouts_paths

    def read_cutouts(
        self,
        cancel_token: Any = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, WCS]]:
        """Read FITS files into arrays that no longer reference those files."""

        self.data = {}
        self.wcs = {}
        self.headers = {}

        for band, path in getattr(self, "cutouts_paths", {}).items():
            try:
                _raise_if_cancelled(cancel_token)

                with fits.open(path, memmap=False) as hdul:
                    hdu = next(
                        (
                            candidate
                            for candidate in hdul
                            if getattr(candidate, "data", None) is not None
                        ),
                        None,
                    )
                    if hdu is None:
                        raise ValueError(
                            "FITS file contains no image HDU"
                        )

                    data = np.array(hdu.data, copy=True)
                    header = hdu.header.copy()

                while data.ndim > 2:
                    data = data[0]

                if data.ndim != 2 or data.size == 0:
                    raise ValueError(
                        f"Unexpected Euclid {band} image shape: "
                        f"{data.shape}"
                    )

                _raise_if_cancelled(cancel_token)

                self.data[band] = data
                self.headers[band] = header
                self.wcs[band] = WCS(header)

            except concurrent.futures.CancelledError:
                raise
            except OSError as exc:
                self.error_tracker.log_error(
                    exc,
                    f"Downloaded corrupted FITS file for {band}",
                )
            except Exception as exc:
                self.error_tracker.log_error(
                    exc,
                    f"Could not read Euclid {band} FITS cutout",
                )

        return self.data, self.wcs

    def get_cutouts(
        self,
        radius: float = 5,
        *,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        bands_to_retrieve: Optional[Iterable[str]] = None,
        cancel_token: Any = None,
        verbose: bool = False,
    ) -> Tuple[
        Optional[Dict[str, np.ndarray]],
        Optional[Dict[str, WCS]],
    ]:
        """Download and read raw Euclid cutouts."""

        self.error_tracker.reset()
        _raise_if_cancelled(cancel_token)

        if ra is not None or dec is not None:
            if ra is None or dec is None:
                raise ValueError(
                    "Both ra and dec must be provided together"
                )
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

        self.get_cone(
            verbose=verbose,
            async_job=False,
            cancel_token=cancel_token,
        )
        _raise_if_cancelled(cancel_token)

        # Cone-search failure is fatal because no band can be downloaded.
        if self.error_tracker.has_error:
            return None, None

        if (
            not hasattr(self, "cone_results")
            or self.cone_results is None
            or len(self.cone_results) <= 0
        ):
            self.error_tracker.log_error(
                "Cone search failed",
                "No Euclid mosaic products found within the search radius",
            )
            return None, None

        paths = self.download_cutouts(
            radius=radius,
            verbose=verbose,
            bands_to_retrieve=bands_to_retrieve,
            cancel_token=cancel_token,
        )
        _raise_if_cancelled(cancel_token)

        # Individual-band failures are not fatal when another requested band
        # completed successfully.
        if not paths:
            return None, None

        data, wcs = self.read_cutouts(
            cancel_token=cancel_token,
        )
        _raise_if_cancelled(cancel_token)

        if not data:
            return None, None

        return data, wcs

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
