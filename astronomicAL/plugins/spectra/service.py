from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


DESI_DATASETS = ["DESI-DR1"]
SDSS_DATASETS = ["BOSS-DR17", "SDSS-DR17"]


def _cancelled(cancel_token: Any) -> bool:
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


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return int(value)
    except Exception:
        return None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    try:
        arr = np.asarray(value)
        if arr.ndim == 0:
            item = arr.item()
            if isinstance(item, float) and not np.isfinite(item):
                return [None]
            return [item]
        out = arr.tolist()
        return out if isinstance(out, list) else [out]
    except Exception:
        try:
            return list(value)
        except Exception:
            return [value]


def _jsonish_scalar(value: Any) -> Any:
    try:
        if value is None:
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    except Exception:
        return str(value)

def _spectrum_plot_color(index: int, total: int, cmap: str = "gist_rainbow") -> str:
    """Return the colour used for a spectrum index in the legacy spectrum plot.

    The legacy plotting code uses:

        plt.get_cmap(cmap, max(N, 2))(idx)

    for each spectrum. This helper mirrors that exactly and converts the result
    to a hex colour so other panels, especially Euclid Cutout, can reuse it.
    """

    fallback = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
    ]

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex

        colour = plt.get_cmap(cmap, max(int(total), 2))(int(index))
        return to_hex(colour, keep_alpha=False)
    except Exception:
        return fallback[int(index) % len(fallback)]

@dataclass
class SpectraResult:
    """Result returned by SpectraRuntime after one retrieval."""

    source: str
    spectra_object: Any
    spectra: List[Any]
    ra: Optional[float]
    dec: Optional[float]
    source_id: Optional[Any]
    max_separation_arcsec: float
    datasets: List[str]
    retrieval_mode: str
    smooth_kernel: str
    smooth_window: int

    @property
    def available_spectra(self) -> int:
        return len(self.spectra or [])

    def plot_hv(
        self,
        *,
        plot_model: bool = True,
        plot_lines: Any = "class",
        responsive: bool = True,
        cmap: str = "gist_rainbow",
        **kwargs: Any,
    ) -> Any:
        obj = self.spectra_object
        if obj is None:
            raise RuntimeError("No spectrum object is available.")

        if not hasattr(obj, "available_spectra"):
            try:
                obj.available_spectra = len(getattr(obj, "spectra", []) or [])
            except Exception:
                pass

        return obj.plot_all_spectra_hv(
            plot_model=plot_model,
            plot_lines=plot_lines,
            responsive=responsive,
            cmap=cmap,
            **kwargs,
        )

    def coordinates_payload(self, cmap: str = "gist_rainbow") -> Dict[str, Any]:
        """Return sky coordinates plus the matching spectrum plot colours.

        The Euclid cutout panel consumes this artifact. Each coordinate entry is
        colour-coded with the same colour used by the corresponding spectrum
        line/model line in the spectra plot.
        """

        ra_values: List[float] = []
        dec_values: List[float] = []
        colours: List[str] = []
        labels: List[str] = []
        indices: List[int] = []
        points: List[Dict[str, Any]] = []

        spectra = list(self.spectra or [])
        total = len(spectra)

        for idx, spectrum in enumerate(spectra):
            ra = _safe_float(getattr(spectrum, "ra", None))
            dec = _safe_float(getattr(spectrum, "dec", None))

            if ra is None or dec is None:
                continue

            colour = _spectrum_plot_color(idx, total, cmap=cmap)
            source_id = getattr(
                spectrum,
                "sourceid",
                getattr(spectrum, "sourceId", getattr(spectrum, "specid", None)),
            )
            data_release = getattr(spectrum, "data_release", None)

            label_parts = [self.source, f"#{idx + 1}"]
            if data_release:
                label_parts.append(str(data_release))
            if source_id is not None:
                label_parts.append(str(source_id))
            label = " ".join(label_parts)

            ra_values.append(ra)
            dec_values.append(dec)
            colours.append(colour)
            labels.append(label)
            indices.append(idx)

            points.append(
                {
                    "index": idx,
                    "ra": ra,
                    "dec": dec,
                    "color": colour,
                    "colour": colour,
                    "label": label,
                    "source": self.source,
                    "source_id": _jsonish_scalar(source_id),
                    "data_release": _jsonish_scalar(data_release),
                }
            )

        return {
            "ra": ra_values,
            "dec": dec_values,
            "colors": colours,
            "colours": colours,
            "labels": labels,
            "indices": indices,
            "points": points,
            "cmap": cmap,
            "source": self.source,
        }

    def artifact_payload(self) -> Dict[str, Any]:
        records = []
        smoothed = getattr(self.spectra_object, "smoothed_fluxes", None) or []
        spectra = list(self.spectra or [])
        total = len(spectra)

        for idx, spectrum in enumerate(spectra):
            smoothed_flux = smoothed[idx] if idx < len(smoothed) else None
            colour = _spectrum_plot_color(idx, total)

            records.append(
                {
                    "index": idx,
                    "plot_color": colour,
                    "plot_colour": colour,
                    "sourceid": _jsonish_scalar(
                        getattr(
                            spectrum,
                            "sourceid",
                            getattr(spectrum, "sourceId", getattr(spectrum, "specid", None)),
                        )
                    ),
                    "sparcl_id": _jsonish_scalar(getattr(spectrum, "sparcl_id", None)),
                    "specid": _jsonish_scalar(getattr(spectrum, "specid", None)),
                    "data_release": _jsonish_scalar(getattr(spectrum, "data_release", None)),
                    "ra": _jsonish_scalar(getattr(spectrum, "ra", None)),
                    "dec": _jsonish_scalar(getattr(spectrum, "dec", None)),
                    "redshift": _jsonish_scalar(getattr(spectrum, "redshift", None)),
                    "spectype": _jsonish_scalar(getattr(spectrum, "spectype", "")),
                    "wavelength": _as_list(getattr(spectrum, "wavelength", None)),
                    "flux": _as_list(getattr(spectrum, "flux", None)),
                    "smoothed_flux": _as_list(smoothed_flux),
                    "model": _as_list(getattr(spectrum, "model", None)),
                    "mask": _as_list(getattr(spectrum, "mask", None)),
                }
            )

        return {
            "type": "astro.spectra",
            "source": self.source,
            "ra": self.ra,
            "dec": self.dec,
            "source_id": self.source_id,
            "max_separation_arcsec": self.max_separation_arcsec,
            "datasets": list(self.datasets or []),
            "retrieval_mode": self.retrieval_mode,
            "smooth_kernel": self.smooth_kernel,
            "smooth_window": self.smooth_window,
            "available_spectra": len(records),
            "colors": [record["plot_color"] for record in records],
            "colours": [record["plot_color"] for record in records],
            "coordinates": self.coordinates_payload(),
            "spectra": records,
        }


class SpectraRuntime:
    """Runtime service for DESI/SDSS/BOSS/Euclid spectra.

    This service is intentionally thin. It keeps the old, domain-specific
    retrieval code in ``astro_data_utility_legacy.py`` but exposes it through a
    platform-safe API for panels and actions.
    """

    def __init__(self, context: Any = None) -> None:
        self.context = context

    def fetch_spectra(
        self,
        *,
        source: str,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        source_id: Optional[Any] = None,
        max_separation_arcsec: float = 1.0,
        datasets: Optional[Iterable[str]] = None,
        smooth_kernel: str = "Box1DKernel",
        smooth_window: int = 10,
        redshift_override: Optional[float] = None,
        spectype_override: Optional[str] = None,
        query_euclid_redshift: bool = False,
        cancel_token: Any = None,
    ) -> SpectraResult:
        if _cancelled(cancel_token):
            raise RuntimeError("Spectrum request was cancelled before it started.")

        source = str(source)
        max_sep = _safe_float(max_separation_arcsec)
        if max_sep is None or max_sep <= 0:
            raise ValueError(f"Invalid spectrum search radius: {max_separation_arcsec!r}")

        source_id_int = _safe_int(source_id)
        retrieval_mode = "target_id" if source_id is not None else "cone"

        if source_id is None:
            ra_value = _safe_float(ra)
            dec_value = _safe_float(dec)
            if ra_value is None or dec_value is None:
                raise ValueError("Cone-search spectrum retrieval requires numeric RA/Dec.")
        else:
            ra_value = _safe_float(ra)
            dec_value = _safe_float(dec)

        if source == "EuclidSpec":
            result = self._fetch_euclid(
                ra=ra_value,
                dec=dec_value,
                source_id=source_id_int if source_id_int is not None else source_id,
                max_separation_arcsec=max_sep,
                smooth_kernel=smooth_kernel,
                smooth_window=smooth_window,
                redshift_override=redshift_override,
                spectype_override=spectype_override,
                query_redshift=query_euclid_redshift,
                cancel_token=cancel_token,
            )
            result.retrieval_mode = retrieval_mode
            return result

        if source == "DESI":
            dataset_list = list(datasets or DESI_DATASETS)
        elif source == "SDSS":
            dataset_list = list(datasets or SDSS_DATASETS)
        else:
            raise ValueError(f"Unknown spectrum source: {source!r}")

        result = self._fetch_sparcl(
            source=source,
            ra=ra_value,
            dec=dec_value,
            source_id=source_id_int if source_id_int is not None else source_id,
            max_separation_arcsec=max_sep,
            datasets=dataset_list,
            smooth_kernel=smooth_kernel,
            smooth_window=smooth_window,
            redshift_override=redshift_override,
            spectype_override=spectype_override,
            cancel_token=cancel_token,
        )
        result.retrieval_mode = retrieval_mode
        return result

    def _fetch_sparcl(
        self,
        *,
        source: str,
        ra: Optional[float],
        dec: Optional[float],
        source_id: Optional[Any],
        max_separation_arcsec: float,
        datasets: List[str],
        smooth_kernel: str,
        smooth_window: int,
        redshift_override: Optional[float],
        spectype_override: Optional[str],
        cancel_token: Any,
    ) -> SpectraResult:
        from .astro_data_utility_legacy import DESISpectraClass

        obj = DESISpectraClass(
            ra if ra is not None else np.nan,
            dec if dec is not None else np.nan,
            datasets=datasets,
            max_separation=max_separation_arcsec,
            sourceId=source_id,
            context=self.context,
        )

        if _cancelled(cancel_token):
            raise RuntimeError("Spectrum request was cancelled before archive retrieval.")

        obj.get_spectra(max_separation=max_separation_arcsec, return_object=True)

        if _cancelled(cancel_token):
            raise RuntimeError("Spectrum request was cancelled after archive retrieval.")

        self._check_error_tracker(obj)

        spectra = list(getattr(obj, "spectra", []) or [])
        if not spectra:
            raise RuntimeError(f"No {source} spectra were returned.")

        try:
            obj.get_smoothed_spectra(kernel=smooth_kernel, window=int(smooth_window))
        except Exception:
            pass

        self._apply_info_overrides(
            obj,
            redshift_override=redshift_override,
            spectype_override=spectype_override,
        )

        return SpectraResult(
            source=source,
            spectra_object=obj,
            spectra=spectra,
            ra=ra,
            dec=dec,
            source_id=source_id,
            max_separation_arcsec=max_separation_arcsec,
            datasets=datasets,
            retrieval_mode="cone",
            smooth_kernel=smooth_kernel,
            smooth_window=int(smooth_window),
        )

    def _fetch_euclid(
        self,
        *,
        ra: Optional[float],
        dec: Optional[float],
        source_id: Optional[Any],
        max_separation_arcsec: float,
        smooth_kernel: str,
        smooth_window: int,
        redshift_override: Optional[float],
        spectype_override: Optional[str],
        query_redshift: bool,
        cancel_token: Any,
    ) -> SpectraResult:
        from .astro_data_utility_legacy import EuclidSpectraClass

        obj = EuclidSpectraClass(
            ra if ra is not None else np.nan,
            dec if dec is not None else np.nan,
            max_separation=max_separation_arcsec,
            sourceId=source_id,
            context=self.context,
        )

        # The legacy Euclid sourceId path expects table_results to exist.
        # Populate the minimal shape needed for exact-source retrieval.
        if source_id is not None:
            try:
                import pandas as pd

                obj.table_results = pd.DataFrame({"source_id": [source_id]})
                obj.available_spectra = 1
            except Exception:
                pass

        if _cancelled(cancel_token):
            raise RuntimeError("Euclid spectrum request was cancelled before archive retrieval.")

        obj.get_spectra(
            max_separation=max_separation_arcsec,
            return_object=True,
            smooth_kernel=smooth_kernel,
            smooth_window=int(smooth_window),
        )

        if _cancelled(cancel_token):
            raise RuntimeError("Euclid spectrum request was cancelled after archive retrieval.")

        self._check_error_tracker(obj)

        spectra = list(getattr(obj, "spectra", []) or [])
        if not spectra:
            raise RuntimeError("No Euclid spectra were returned.")

        if query_redshift:
            self.query_euclid_redshift(obj)

        self._apply_info_overrides(
            obj,
            redshift_override=redshift_override,
            spectype_override=spectype_override,
        )

        return SpectraResult(
            source="EuclidSpec",
            spectra_object=obj,
            spectra=spectra,
            ra=ra,
            dec=dec,
            source_id=source_id,
            max_separation_arcsec=max_separation_arcsec,
            datasets=["Euclid-Q1"],
            retrieval_mode="cone",
            smooth_kernel=smooth_kernel,
            smooth_window=int(smooth_window),
        )

    def query_euclid_redshift(self, spectra_object: Any) -> None:
        if spectra_object is None:
            return
        if not hasattr(spectra_object, "query_specz_table"):
            return
        spectra_object.query_specz_table(verbose=True)
        if hasattr(spectra_object, "update_info_from_query"):
            spectra_object.update_info_from_query()

    @staticmethod
    def _check_error_tracker(obj: Any) -> None:
        tracker = getattr(obj, "error_tracker", None)
        if tracker is None:
            return
        if getattr(tracker, "has_error", False):
            message = getattr(tracker, "error_message", None) or "Spectrum request failed."
            raise RuntimeError(str(message))

    @staticmethod
    def _apply_info_overrides(
        obj: Any,
        *,
        redshift_override: Optional[float],
        spectype_override: Optional[str],
    ) -> None:
        if obj is None or not hasattr(obj, "_update_info_spectra"):
            return

        redshift = _safe_float(redshift_override)
        if redshift is not None:
            obj._update_info_spectra("redshift", redshift)
            if not spectype_override:
                spectype_override = "galaxy" if redshift > 0 else "star"

        if spectype_override:
            obj._update_info_spectra("spectype", str(spectype_override))


def create_spectra_runtime(context: Any = None, **_: Any) -> SpectraRuntime:
    return SpectraRuntime(context=context)