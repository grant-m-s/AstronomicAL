from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .storage import (
    EuclidPanelStorageLease,
    EuclidRequestScratch,
    EuclidScratchManager,
)


DEFAULT_EUCLID_FILTERS = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]
DEFAULT_SAVE_DIR = "data/cutouts"
COLOR_BAND_SETS = [
    ["NIR_H", "NIR_Y", "VIS"],
    ["NIR_H", "NIR_J", "VIS"],
    ["NIR_J", "NIR_Y", "VIS"],
]


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


def _raise_if_cancelled(cancel_token: Any, stage: str) -> None:
    if _cancelled(cancel_token):
        raise CancelledError(f"Euclid cutout request cancelled {stage}.")


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


def _requested_filters(
    filter_name: str,
    configured_filters: Optional[Iterable[str]],
) -> list[str]:
    filters = list(dict.fromkeys(configured_filters or DEFAULT_EUCLID_FILTERS))
    selected = str(filter_name or "Color")
    if selected == "Color":
        return filters
    if selected not in filters:
        raise ValueError(f"Unknown Euclid filter: {selected!r}")
    return [selected]


def _available_color_bands(available: Iterable[str]) -> Optional[list[str]]:
    available_set = set(available)
    for candidate in COLOR_BAND_SETS:
        if all(band in available_set for band in candidate):
            return list(candidate)
    return None


@dataclass
class EuclidCutoutResult:
    """Raw runtime result returned after a Euclid cutout request.

    The arrays are detached from their FITS files. ``scratch`` remains alive
    only until the panel has accepted, promoted, or discarded the result.
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
    scratch: Optional[EuclidRequestScratch] = field(default=None, repr=False)
    owned_panel_lease: Optional[EuclidPanelStorageLease] = field(
        default=None,
        repr=False,
    )

    def cleanup(self) -> None:
        scratch = self.scratch
        self.scratch = None
        if scratch is not None:
            scratch.close()

        owned_lease = self.owned_panel_lease
        self.owned_panel_lease = None
        if owned_lease is not None:
            owned_lease.release()

    def artifact_payload(self, *, include_pixels: bool = False) -> Dict[str, Any]:
        """Return a platform-friendly artifact payload."""

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
    """Plugin service for Euclid retrieval and request-storage ownership."""

    def __init__(self, context: Any = None) -> None:
        self.context = context
        self._last_client: Any = None
        self._scratch = EuclidScratchManager()

    def acquire_panel_storage(self, owner_id: str) -> EuclidPanelStorageLease:
        return self._scratch.acquire_panel(owner_id)

    @staticmethod
    def release_panel_storage(lease: Optional[EuclidPanelStorageLease]) -> None:
        if lease is not None:
            lease.release()

    @staticmethod
    def cleanup_result(result: Any) -> None:
        cleanup = getattr(result, "cleanup", None)
        if callable(cleanup):
            cleanup()

    def promote_result(self, result: EuclidCutoutResult) -> EuclidCutoutResult:
        """Retain an explicitly requested artifact using a deterministic key."""

        promoted = self._scratch.promote_files(
            fits_paths=result.fits_paths,
            base_dir=result.save_dir,
            identity={
                "source": "Euclid",
                "environment": result.environment,
                "ra": result.ra,
                "dec": result.dec,
                "radius_arcsec": result.radius_arcsec,
                "filters": sorted(result.filters),
            },
        )
        if promoted:
            result.fits_paths = promoted
            cutout = getattr(result, "cutout", None)
            if cutout is not None:
                try:
                    cutout.cutouts_paths = dict(promoted)
                except Exception:
                    pass
        return result

    def dispose(self) -> None:
        self._scratch.close()

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
        stretch: str = "Linear",  # compatibility; display concern
        stretch_scale: Optional[float] = None,  # compatibility; display concern
        environment: str = "PDR",
        user: Optional[str] = None,
        password: Optional[str] = None,
        credentials_filepath: Optional[str] = None,
        save_dir: str = DEFAULT_SAVE_DIR,
        euclid_filters: Optional[Iterable[str]] = None,
        panel_storage: Optional[EuclidPanelStorageLease] = None,
        request_id: Optional[int] = None,
        cancel_token: Any = None,
        verbose: bool = False,
    ) -> EuclidCutoutResult:
        """Fetch raw Euclid cutouts for one sky position.

        A single selected band retrieves only that band. ``Color`` retrieves the
        configured set so the panel can choose a valid RGB combination.
        """

        del stretch, stretch_scale
        _raise_if_cancelled(cancel_token, "before it started")

        ra_value = _safe_float(ra)
        dec_value = _safe_float(dec)
        radius_value = _safe_float(radius_arcsec)
        if ra_value is None or dec_value is None:
            raise ValueError(f"Invalid Euclid coordinates: ra={ra!r}, dec={dec!r}")
        if radius_value is None or radius_value <= 0:
            raise ValueError(f"Invalid Euclid cutout radius: {radius_arcsec!r}")

        filters = _requested_filters(filter_name, euclid_filters)
        base_path = Path(save_dir or DEFAULT_SAVE_DIR).expanduser()
        base_path.mkdir(parents=True, exist_ok=True)

        owned_lease: Optional[EuclidPanelStorageLease] = None
        lease = panel_storage
        if lease is None:
            owned_lease = self.acquire_panel_storage("euclid.runtime.request")
            lease = owned_lease
        effective_request_id = (
            int(request_id)
            if request_id is not None
            else self._scratch.next_request_id()
        )
        scratch = lease.begin_request(
            base_dir=str(base_path),
            request_id=effective_request_id,
        )

        try:
            from .astro_data_utility_legacy import EuclidCutoutsClass

            cutout = EuclidCutoutsClass(
                ra=ra_value,
                dec=dec_value,
                euclid_filters=filters,
                save_dir=str(scratch.root),
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
            _raise_if_cancelled(cancel_token, "before archive retrieval")
            self._retrieve_raw_cutouts(
                cutout,
                radius_arcsec=radius_value,
                filters=filters,
                cancel_token=cancel_token,
                verbose=verbose,
            )
            _raise_if_cancelled(cancel_token, "after archive retrieval")

            raw_images = dict(getattr(cutout, "data", {}) or {})
            raw_wcs = dict(getattr(cutout, "wcs", {}) or {})
            available_filters = [
                band for band in filters if band in raw_images and band in raw_wcs
            ]
            tracker = getattr(cutout, "error_tracker", None)
            detail = (
                getattr(tracker, "error_message", None)
                if tracker is not None
                else None
            )
            selected_filter = str(filter_name or "Color")
            if selected_filter == "Color":
                if _available_color_bands(available_filters) is None:
                    message = (
                        "Could not load enough Euclid bands to build a Color "
                        "cutout"
                    )
                    if detail:
                        message = f"{message}: {detail}"
                    raise RuntimeError(message)
            elif selected_filter not in available_filters:
                message = (
                    f"Could not load requested Euclid {selected_filter} FITS "
                    "cutout"
                )
                if detail:
                    message = f"{message}: {detail}"
                raise RuntimeError(message)

            fits_paths = {
                str(key): str(value)
                for key, value in getattr(cutout, "cutouts_paths", {}).items()
                if value is not None and str(key) in available_filters
            }
            return EuclidCutoutResult(
                cutout=cutout,
                images={
                    band: np.array(raw_images[band], copy=True)
                    for band in available_filters
                },
                wcs={band: raw_wcs[band] for band in available_filters},
                filters=list(available_filters),
                ra=ra_value,
                dec=dec_value,
                radius_arcsec=radius_value,
                environment=environment or "PDR",
                save_dir=str(base_path),
                fits_paths=fits_paths,
                scratch=scratch,
                owned_panel_lease=owned_lease,
            )
        except BaseException:
            scratch.close()
            if owned_lease is not None:
                owned_lease.release()
            raise

    @staticmethod
    def _retrieve_raw_cutouts(
        cutout: Any,
        *,
        radius_arcsec: float,
        filters: list[str],
        cancel_token: Any,
        verbose: bool,
    ) -> None:
        get_cutouts = getattr(cutout, "get_cutouts", None)
        if not callable(get_cutouts):
            raise RuntimeError("EuclidCutoutsClass has no get_cutouts method.")
        data, wcs = get_cutouts(
            radius=radius_arcsec,
            bands_to_retrieve=filters,
            cancel_token=cancel_token,
            verbose=verbose,
        )
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