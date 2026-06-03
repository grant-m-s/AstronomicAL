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
    """Runtime result returned to the panel after a Euclid cutout request."""

    cutout: Any
    image: Any
    filter_name: str
    ra: float
    dec: float
    radius_arcsec: float
    environment: str
    stretch: str
    stretch_scale: Optional[float]
    save_dir: str
    fits_paths: Dict[str, str]

    def artifact_payload(self) -> Dict[str, Any]:
        """Return a payload suitable for an in-memory ArtifactStore entry."""

        plot_data = {}
        plot_data_info = {}

        try:
            for key, value in getattr(self.cutout, "plot_data", {}).items():
                plot_data[key] = value
        except Exception:
            pass

        try:
            for key, value in getattr(self.cutout, "plot_data_info", {}).items():
                plot_data_info[key] = value
        except Exception:
            pass

        arcsec_per_pix = {}
        try:
            for key, value in getattr(self.cutout, "arcsec_per_pix", {}).items():
                try:
                    arcsec_per_pix[key] = float(value)
                except Exception:
                    arcsec_per_pix[key] = value
        except Exception:
            pass

        return {
            "source": "Euclid",
            "ra": self.ra,
            "dec": self.dec,
            "radius_arcsec": self.radius_arcsec,
            "environment": self.environment,
            "filter_name": self.filter_name,
            "stretch": self.stretch,
            "stretch_scale": self.stretch_scale,
            "save_dir": self.save_dir,
            "fits_paths": dict(self.fits_paths),
            "image": self.image,
            "plot_data": plot_data,
            "plot_data_info": plot_data_info,
            "arcsec_per_pix": arcsec_per_pix,
        }


class EuclidCutoutRuntime:
    """Plugin runtime for Euclid cutout retrieval.

    This service is intentionally thin. The actual archive/WCS/reprojection work
    remains in ``astro_data_utility_legacy.EuclidCutoutsClass`` so the old, tested
    Euclid code can be reused while the panel uses the new platform services.
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
        filter_name: str = "Color",
        stretch: str = "Linear",
        stretch_scale: Optional[float] = None,
        environment: str = "PDR",
        user: Optional[str] = None,
        password: Optional[str] = None,
        credentials_filepath: Optional[str] = None,
        save_dir: str = DEFAULT_SAVE_DIR,
        euclid_filters: Optional[Iterable[str]] = None,
        cancel_token: Any = None,
        verbose: bool = False,
    ) -> EuclidCutoutResult:
        """Fetch, reproject and stretch the Euclid cutouts for one sky position."""

        if _cancelled(cancel_token):
            raise RuntimeError("Euclid cutout request was cancelled before it started.")

        ra_value = _safe_float(ra)
        dec_value = _safe_float(dec)
        radius_value = _safe_float(radius_arcsec)

        if ra_value is None or dec_value is None:
            raise ValueError(f"Invalid Euclid coordinates: ra={ra!r}, dec={dec!r}")
        if radius_value is None or radius_value <= 0:
            raise ValueError(f"Invalid Euclid cutout radius: {radius_arcsec!r}")

        # Import lazily so the generic platform can be imported without pulling
        # the astronomy stack into non-astro deployments.
        from .astro_data_utility_legacy import EuclidCutoutsClass

        save_path = Path(save_dir).expanduser()
        save_path.mkdir(parents=True, exist_ok=True)

        cutout = EuclidCutoutsClass(
            ra=ra_value,
            dec=dec_value,
            euclid_filters=list(euclid_filters or DEFAULT_EUCLID_FILTERS),
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
        elif environment == "PDR":
            # Keep the legacy object and service registry consistent when the
            # user switches back from a protected environment.
            cutout.change_environment(environment="PDR")

        self._register_client_alias(getattr(cutout, "client", None))

        if _cancelled(cancel_token):
            raise RuntimeError("Euclid cutout request was cancelled before archive retrieval.")

        image = cutout.get_final_cutout(
            radius=radius_value,
            stretch=stretch,
            filtro=filter_name,
            reference="VIS",
            stretch_scale=stretch_scale,
            verbose=verbose,
            return_object=True,
        )

        if _cancelled(cancel_token):
            raise RuntimeError("Euclid cutout request was cancelled after archive retrieval.")

        tracker = getattr(cutout, "error_tracker", None)
        if tracker is not None and getattr(tracker, "has_error", False):
            message = getattr(tracker, "error_message", None) or "Euclid cutout request failed."
            raise RuntimeError(str(message))

        if image is None:
            image = getattr(cutout, "plot_data", {}).get(filter_name)
        if image is None:
            raise RuntimeError("Euclid cutout did not produce displayable image data.")

        fits_paths = {
            str(key): str(value)
            for key, value in getattr(cutout, "cutouts_paths", {}).items()
            if value is not None
        }

        return EuclidCutoutResult(
            cutout=cutout,
            image=image,
            filter_name=filter_name,
            ra=ra_value,
            dec=dec_value,
            radius_arcsec=radius_value,
            environment=environment or "PDR",
            stretch=stretch,
            stretch_scale=stretch_scale,
            save_dir=str(save_path),
            fits_paths=fits_paths,
        )

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
            # Cleaning ESA jobs is a convenience action. It should never break
            # the UI if the client is logged out or the archive is unreachable.
            return


def create_euclid_runtime(context: Any = None, **_: Any) -> EuclidCutoutRuntime:
    return EuclidCutoutRuntime(context=context)