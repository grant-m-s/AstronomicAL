from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import json
import math

import pandas as pd

BandConfig = Dict[str, Dict[str, Any]]


@dataclass
class SEDBuildResult:
    dataframe: pd.DataFrame
    skipped: List[Dict[str, str]]


SED_UNIT_OPTIONS = ["ABmag", "Jy", "mJy", "µJy", "nJy"]

SED_UNIT_ALIASES = {
    "ab": "ABmag",
    "abmag": "ABmag",
    "ab_mag": "ABmag",
    "ab magnitude": "ABmag",
    "mag": "ABmag",
    "magnitude": "ABmag",
    "jy": "Jy",
    "jansky": "Jy",
    "janskys": "Jy",
    "mjy": "mJy",
    "millijy": "mJy",
    "millijansky": "mJy",
    "millijanskys": "mJy",
    "ujy": "µJy",
    "µjy": "µJy",
    "μjy": "µJy",
    "microjy": "µJy",
    "microjansky": "µJy",
    "microjanskys": "µJy",
    "njy": "nJy",
    "nanojy": "nJy",
    "nanojansky": "nJy",
    "nanojanskys": "nJy",
}


def normalise_sed_unit(unit: Any) -> str:
    """Return the canonical SED unit name used by the plugin."""
    raw = str(unit or "ABmag").strip()
    return SED_UNIT_ALIASES.get(raw.lower(), raw)


def flux_value_to_microjy(value: float, unit: Any) -> float:
    """Convert one flux-like value to microJy.

    For AB magnitudes, this uses the standard relation in which
    m_AB = 23.9 corresponds to 1 microJy.
    """
    unit = normalise_sed_unit(unit)

    if unit == "ABmag":
        return 10 ** ((23.9 - value) / 2.5)

    if unit == "Jy":
        return value * 1.0e6

    if unit == "mJy":
        return value * 1.0e3

    if unit in {"uJy", "µJy"}:
        return value

    if unit == "nJy":
        return value * 1.0e-3

    raise ValueError(f"Unsupported SED unit: {unit!r}")


def flux_error_to_microjy(error_value: Optional[float], *, unit: Any, flux_uJy: float) -> float:
    """Convert an error value to microJy.

    The plugin assumes the error column has the same unit as the value column.
    Therefore, when the value unit is ABmag, the error is interpreted as a
    magnitude error and propagated as sigma_F = ln(10) / 2.5 * F * sigma_mag.
    """
    if error_value is None:
        return 0.0

    unit = normalise_sed_unit(unit)

    if unit == "ABmag":
        return math.log(10.0) / 2.5 * flux_uJy * error_value

    return flux_value_to_microjy(error_value, unit)


class SEDRuntime:
    """Shared runtime for the Broadband SED plugin.

    The SED JSON file is treated as filter metadata only.  Its top-level keys
    are filter names and each filter stores wavelength/FWHM information.  The
    dataset-specific mappings, error columns, and units are provided by the
    Panel UI and persisted in the plugin state/artifact payload.
    """

    def __init__(self, sed_data_dir: str | Path = "data/sed_data") -> None:
        self.sed_data_dir = Path(sed_data_dir)

    # ------------------------------------------------------------------
    # Band-file helpers
    # ------------------------------------------------------------------

    def ensure_sed_data_dir(self) -> Path:
        self.sed_data_dir.mkdir(parents=True, exist_ok=True)
        return self.sed_data_dir

    def list_band_files(self) -> List[str]:
        root = self.ensure_sed_data_dir()
        return sorted(str(path) for path in root.glob("*.json"))

    def load_band_file(self, path: str | Path) -> BandConfig:
        """Load the SED filter-definition JSON file.

        Supported format:

        {
          "euclid_VIS": {"wavelength": 0.75, "FWHM": 0.35},
          "lsst_g": {"wavelength": 0.48, "FWHM": 0.14}
        }

        A list of dictionaries is also accepted for convenience, provided each
        entry contains either ``filter_name`` or ``name``.
        """
        band_path = Path(path)

        if not band_path.is_file():
            raise FileNotFoundError(f"SED photometry-band file not found: {band_path}")

        with band_path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)

        bands: BandConfig = {}

        if isinstance(raw, list):
            iterable = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                name = item.get("filter_name") or item.get("name") or item.get("band")
                if name:
                    iterable.append((str(name), item))
        elif isinstance(raw, dict):
            iterable = list(raw.items())
        else:
            raise ValueError(f"SED photometry-band file must contain a JSON object or list: {band_path}")

        for filter_name, spec in iterable:
            if not isinstance(spec, dict):
                continue

            bands[str(filter_name)] = {
                "filter_name": str(spec.get("filter_name") or filter_name),
                "wavelength": spec.get("wavelength", -99),
                "FWHM": spec.get("FWHM", 0),
            }

        return bands

    def create_photometry_band_file(self, columns: Sequence[str]) -> str:
        """Create a template SED filter JSON file.

        This keeps the legacy behaviour: every dataset column is written as a
        possible filter and starts disabled with ``wavelength = -99``.  You can
        then edit the JSON so that only real filters have wavelength/FWHM.
        """
        root = self.ensure_sed_data_dir()

        bands_dict = {
            str(col): {
                "filter_name": str(col),
                "wavelength": -99,
                "FWHM": 0,
            }
            for col in columns
        }

        base_path = root / "photometry_bands.json"

        if not base_path.exists():
            output_path = base_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = root / f"photometry_bands_{timestamp}.json"

        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(bands_dict, fp, indent=2)

        return str(output_path)

    # ------------------------------------------------------------------
    # Band/schema helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_enabled_band(spec: Mapping[str, Any]) -> bool:
        return spec.get("wavelength", -99) != -99

    def referenced_column_tokens(self, bands: BandConfig) -> List[str]:
        """Return string tokens that may refer to dataset columns.

        The filter name itself is the value-column reference.  Wavelength/FWHM
        are usually numeric filter properties, but string-valued entries are
        kept supported for backwards compatibility.
        """
        tokens: List[str] = []

        for filter_name, spec in bands.items():
            if not self._is_enabled_band(spec):
                continue

            tokens.append(str(filter_name))

            for key in ("wavelength", "FWHM"):
                value = spec.get(key)
                if isinstance(value, str):
                    tokens.append(value)

        return list(dict.fromkeys(tokens))

    # ------------------------------------------------------------------
    # Row / SED construction
    # ------------------------------------------------------------------

    @staticmethod
    def row_to_mapping(row: Any) -> Dict[str, Any]:
        if row is None:
            return {}

        if isinstance(row, dict):
            return dict(row)

        if hasattr(row, "empty") and hasattr(row, "iloc"):
            if row.empty:
                return {}
            return dict(row.iloc[0].to_dict())

        if hasattr(row, "to_dict"):
            try:
                raw = row.to_dict()
                if isinstance(raw, dict):
                    return dict(raw)
            except Exception:
                pass

        return {}

    @staticmethod
    def _normalise_number(value: Any) -> Optional[float]:
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        try:
            out = float(value)
        except Exception:
            return None

        if not math.isfinite(out):
            return None

        return out

    def _resolve_reference(
        self,
        value: Any,
        *,
        row: Mapping[str, Any],
        column_overrides: Mapping[str, str],
    ) -> Any:
        """Resolve either a numeric literal or a dataset-column token."""
        if not isinstance(value, str):
            return value

        if value in row:
            return row[value]

        mapped = column_overrides.get(value)
        if mapped and mapped in row:
            return row[mapped]

        raise KeyError(value)

    def build_sed_table(
        self,
        *,
        row: Any,
        bands: BandConfig,
        column_overrides: Optional[Mapping[str, str]] = None,
        error_column_overrides: Optional[Mapping[str, str]] = None,
        unit_overrides: Optional[Mapping[str, str]] = None,
    ) -> SEDBuildResult:
        """Build the broadband SED dataframe for one focused row.

        ``column_overrides`` maps filter names to value columns.
        ``error_column_overrides`` maps filter names to error columns.
        ``unit_overrides`` maps filter names to the unit of the value/error
        columns.  The output flux and flux error are always in microJy.
        """
        row_map = self.row_to_mapping(row)
        column_overrides = dict(column_overrides or {})
        error_column_overrides = dict(error_column_overrides or {})
        unit_overrides = dict(unit_overrides or {})

        records: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []

        for filter_name, spec in bands.items():
            filter_name = str(filter_name)

            if not self._is_enabled_band(spec):
                continue

            value_column = column_overrides.get(filter_name, filter_name)
            error_column = error_column_overrides.get(filter_name, "")
            unit = normalise_sed_unit(unit_overrides.get(filter_name, "ABmag"))

            try:
                raw_value = self._resolve_reference(
                    filter_name,
                    row=row_map,
                    column_overrides=column_overrides,
                )
            except KeyError:
                skipped.append(
                    {
                        "band": filter_name,
                        "reason": f"value column for filter {filter_name!r} is not mapped",
                    }
                )
                continue

            value = self._normalise_number(raw_value)
            if value is None or value == -99:
                continue

            try:
                wavelength_raw = self._resolve_reference(
                    spec.get("wavelength", -99),
                    row=row_map,
                    column_overrides=column_overrides,
                )
                fwhm_raw = self._resolve_reference(
                    spec.get("FWHM", 0),
                    row=row_map,
                    column_overrides=column_overrides,
                )
            except KeyError as exc:
                skipped.append(
                    {
                        "band": filter_name,
                        "reason": f"referenced filter property {exc.args[0]!r} is not mapped",
                    }
                )
                continue

            wavelength = self._normalise_number(wavelength_raw)
            fwhm = self._normalise_number(fwhm_raw)

            if wavelength is None or wavelength == -99 or wavelength <= 0:
                continue

            if fwhm is None:
                fwhm = 0.0

            error_value = 0.0
            if error_column:
                if error_column in row_map:
                    error_value = self._normalise_number(row_map[error_column]) or 0.0
                else:
                    skipped.append(
                        {
                            "band": filter_name,
                            "reason": f"error column {error_column!r} is not available for this row",
                        }
                    )
                    error_value = 0.0

            try:
                flux_uJy = flux_value_to_microjy(value, unit)
                flux_error_uJy = flux_error_to_microjy(
                    error_value,
                    unit=unit,
                    flux_uJy=flux_uJy,
                )
            except ValueError as exc:
                skipped.append({"band": filter_name, "reason": str(exc)})
                continue

            if flux_uJy <= 0 or not math.isfinite(flux_uJy):
                continue

            # Preserve negative errors: in the legacy SED panel they denote
            # upper limits and are drawn as downward arrows.  Non-finite errors
            # are not useful for plotting, so they become zero.
            if not math.isfinite(flux_error_uJy):
                flux_error_uJy = 0.0

            is_upper_limit = flux_error_uJy < 0
            has_larger_error = flux_error_uJy > flux_uJy

            records.append(
                {
                    "band": filter_name,
                    "wavelength (µm)": wavelength,
                    "flux_uJy": flux_uJy,
                    "flux_error_uJy": flux_error_uJy,
                    "FWHM": fwhm,
                    "input_value": value,
                    "input_error": error_value,
                    "input_unit": unit,
                    "value_column": value_column,
                    "error_column": error_column,
                    "is_upper_limit": is_upper_limit,
                    "has_larger_error": has_larger_error,
                }
            )

        sed_df = pd.DataFrame(
            records,
            columns=[
                "band",
                "wavelength (µm)",
                "flux_uJy",
                "flux_error_uJy",
                "FWHM",
                "input_value",
                "input_error",
                "input_unit",
                "value_column",
                "error_column",
                "is_upper_limit",
                "has_larger_error",
            ],
        )

        return SEDBuildResult(dataframe=sed_df, skipped=skipped)

    def build_artifact_payload(
        self,
        *,
        sed_df: pd.DataFrame,
        dataset_id: str,
        row_id: str,
        sed_file: str,
        column_overrides: Mapping[str, str],
        error_column_overrides: Optional[Mapping[str, str]] = None,
        unit_overrides: Optional[Mapping[str, str]] = None,
        skipped: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> Dict[str, Any]:
        return {
            "dataset_id": dataset_id,
            "row_id": str(row_id),
            "sed_file": str(sed_file),
            "records": sed_df.to_dict(orient="records"),
            "column_overrides": dict(column_overrides),
            "error_column_overrides": dict(error_column_overrides or {}),
            "unit_overrides": dict(unit_overrides or {}),
            "output_unit": "uJy",
            "skipped": [dict(item) for item in (skipped or [])],
        }


def create_sed_runtime(context=None, **kwargs) -> SEDRuntime:
    return SEDRuntime()
