from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


DEFAULT_EUCLID_FILTERS = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]
DEFAULT_SAVE_DIR = "data/cutouts"


def _cancelled(cancel_token: Any) -> bool:
    """Best-effort cancellation check across possible token implementations."""
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


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        out = float(value)
        if not np.isfinite(out):
            return None
        return out
    except Exception:
        return None


@dataclass
class EuclidCutoutResult:
    """Raw runtime result returned after a Euclid cutout request.

    This object deliberately contains raw images and WCS only. Stretching,
    clipping, RGB composition, coordinate conversion and plotting are handled
    by ``image_visualization.ImageVisualizationClass`` in the panel layer.
    """

    cutout: Any
    images: Dict[str, np.ndarray]
    wcs: Dict[str, Any]
    filters: list[str]
    ra: float
    dec: float
    radius_arcsec: float
    environment: str
    save_dir: str
    fits_paths: Dict[str, str]

    def artifact_payload(self, *, include_pixels: bool = False) -> Dict[str, Any]:
        """Return a platform-friendly artifact payload.

        By default this stores serialisable metadata and FITS references only.
        Raw image arrays and WCS objects are large/non-serialisable runtime
        objects and should not become the default artifact convention.

        Set ``include_pixels=True`` only for temporary in-memory debugging or
        compatibility artifacts.
        """
        payload: Dict[str, Any] = {
            "source": "Euclid",
            "ra": self.ra,
            "dec": self.dec,
            "radius_arcsec": self.radius_arcsec,
            "environment": self.environment,
            "save_dir": self.save_dir,
            "fits_paths": dict(self.fits_paths),
            "filters": list(self.filters),
            "payload_kind": "metadata+fits",
        }

        if include_pixels:
            payload["images"] = dict(self.images)
            payload["wcs"] = dict(self.wcs)
            payload["payload_kind"] = "in_memory_pixels"

        return payload


class EuclidCutoutRuntime:
    """Plugin runtime for Euclid cutout retrieval.

    The service retrieves pixels from the archive and reads FITS/WCS data. It
    intentionally does not normalise, stretch, reproject for plotting, build RGB
    images, create HoloViews objects, or convert sky coordinates to pixels.
    """

    def __init__(self, context: Any = None) -> None:
        self.context = context
        self._last_client: Any = None

    def _register_client_alias(self, client: Any) -> None:
        """Expose the raw Euclid client under the legacy key if possible."""
        self._last_client = client
        services = getattr(self.context, "services", None)
        if services is None:
            return
        for method_name in ("set", "register", "put"):
            method = getattr(services, method_name, None)
            if callable(method):
                try:
                    method("euclid.client", client)
                    return
                except Exception:
                    pass

    def fetch_cutout(
        self,
        *,
        ra: float,
        dec: float,
        radius_arcsec: float,
        filter_name: str = "Color",  # kept for old panel-call compatibility; ignored here
        stretch: str = "Linear",  # kept for compatibility; ignored here
        stretch_scale: Optional[float] = None,  # kept for compatibility; ignored here
        environment: str = "PDR",
        user: Optional[str] = None,
        password: Optional[str] = None,
        credentials_filepath: Optional[str] = None,
        save_dir: str = DEFAULT_SAVE_DIR,
        euclid_filters: Optional[Iterable[str]] = None,
        cancel_token: Any = None,
        verbose: bool = False,
    ) -> EuclidCutoutResult:
        """Fetch raw Euclid cutouts for one sky position."""
        if _cancelled(cancel_token):
            raise RuntimeError("Euclid cutout request was cancelled before it started.")

        ra_value = _safe_float(ra)
        dec_value = _safe_float(dec)
        radius_value = _safe_float(radius_arcsec)
        if ra_value is None or dec_value is None:
            raise ValueError(f"Invalid Euclid coordinates: ra={ra!r}, dec={dec!r}")
        if radius_value is None or radius_value <= 0:
            raise ValueError(f"Invalid Euclid cutout radius: {radius_arcsec!r}")

        from .astro_data_utility_legacy import EuclidCutoutsClass

        filters = list(euclid_filters or DEFAULT_EUCLID_FILTERS)
        save_path = Path(save_dir).expanduser()
        save_path.mkdir(parents=True, exist_ok=True)

        cutout = EuclidCutoutsClass(
            ra=ra_value,
            dec=dec_value,
            euclid_filters=filters,
            save_dir=str(save_path),
            context=self.context,
        )

        if environment and environment != "PDR":
            cutout.change_environment(
                environment=environment,
                user=user,
                password=password,
                credentials_filepath=credentials_filepath,
            )
        else:
            cutout.change_environment(environment="PDR")

        self._register_client_alias(getattr(cutout, "client", None))

        if _cancelled(cancel_token):
            raise RuntimeError("Euclid cutout request was cancelled before archive retrieval.")

        self._retrieve_raw_cutouts(cutout, radius_arcsec=radius_value, verbose=verbose)

        if _cancelled(cancel_token):
            raise RuntimeError("Euclid cutout request was cancelled after archive retrieval.")

        tracker = getattr(cutout, "error_tracker", None)
        if tracker is not None and getattr(tracker, "has_error", False):
            message = getattr(tracker, "error_message", None) or "Euclid cutout request failed."
            raise RuntimeError(str(message))

        raw_images = dict(getattr(cutout, "data", {}) or {})
        raw_wcs = dict(getattr(cutout, "wcs", {}) or {})
        available_filters = [band for band in filters if band in raw_images and band in raw_wcs]
        if not available_filters:
            # Be permissive in case the legacy object uses a slightly different
            # filter list than the requested one.
            available_filters = [band for band in raw_images.keys() if band in raw_wcs]

        if not available_filters:
            raise RuntimeError("No Euclid cutout images were loaded.")

        fits_paths = {
            str(key): str(value)
            for key, value in getattr(cutout, "cutouts_paths", {}).items()
            if value is not None
        }

        return EuclidCutoutResult(
            cutout=cutout,
            images={band: np.asarray(raw_images[band]) for band in available_filters},
            wcs={band: raw_wcs[band] for band in available_filters},
            filters=list(available_filters),
            ra=ra_value,
            dec=dec_value,
            radius_arcsec=radius_value,
            environment=environment or "PDR",
            save_dir=str(save_path),
            fits_paths=fits_paths,
        )

    def _retrieve_raw_cutouts(self, cutout: Any, *, radius_arcsec: float, verbose: bool) -> None:
        """Run the single high-level raw retrieval method."""
        get_cutouts = getattr(cutout, "get_cutouts", None)
        if not callable(get_cutouts):
            raise RuntimeError("EuclidCutoutsClass has no get_cutouts method.")

        data, wcs = get_cutouts(radius=radius_arcsec, verbose=verbose)
        if data is not None:
            cutout.data = data
        if wcs is not None:
            cutout.wcs = wcs

    def clean_async_jobs(self) -> None:
        """Clear Euclid archive async jobs for the active client, when supported."""
        client = self._last_client
        if client is None:
            services = getattr(self.context, "services", None)
            if services is not None:
                try:
                    if services.has("euclid.client"):
                        client = services.get("euclid.client")
                except Exception:
                    client = None
        if client is None:
            return
        try:
            joblist = client.list_async_jobs()
            to_remove = [job.jobid for job in joblist]
            if to_remove:
                client.remove_jobs(to_remove)
        except Exception:
            return


def create_euclid_runtime(context: Any = None, **_: Any) -> EuclidCutoutRuntime:
    return EuclidCutoutRuntime(context=context)
