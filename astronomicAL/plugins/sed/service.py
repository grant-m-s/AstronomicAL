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


class SEDRuntime:
    """Shared runtime for the Broadband SED plugin.

    The legacy custom plot stored SED band definitions in JSON files under
    ``data/sed_data``. This service preserves that file format but removes the
    dependency on global ``astronomicAL.config`` state.
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
        band_path = Path(path)
        if not band_path.is_file():
            raise FileNotFoundError(f"SED photometry-band file not found: {band_path}")

        with band_path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)

        if not isinstance(raw, dict):
            raise ValueError(f"SED photometry-band file must contain a JSON object: {band_path}")

        bands: BandConfig = {}
        for band_name, spec in raw.items():
            if not isinstance(spec, dict):
                continue
            bands[str(band_name)] = {
                "wavelength": spec.get("wavelength", -99),
                "FWHM": spec.get("FWHM", 0),
                "error": spec.get("error", 0),
            }
        return bands

    def create_photometry_band_file(self, columns: Sequence[str]) -> str:
        """Create a legacy-compatible photometry-band JSON file.

        Each dataset column is written as a possible magnitude band and starts
        disabled with ``wavelength = -99``. This mirrors the legacy
        ``SEDPlot.create_photometry_band_file`` behaviour.
        """

        root = self.ensure_sed_data_dir()
        bands_dict = {
            str(col): {
                "wavelength": -99,
                "FWHM": 0,
                "error": 0,
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
        """Return all string tokens in the band file that may refer to dataset columns.

        This includes enabled band names, plus string-valued wavelength/FWHM/error
        entries. The panel uses this list to provide legacy-style dynamic column
        mapping when a token is not directly present in the active dataset.
        """

        tokens: List[str] = []
        for band_name, spec in bands.items():
            if not self._is_enabled_band(spec):
                continue
            tokens.append(str(band_name))
            for key in ("wavelength", "FWHM", "error"):
                value = spec.get(key)
                if isinstance(value, str):
                    tokens.append(value)

        # Stable de-duplication.
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
    ) -> SEDBuildResult:
        """Build the broadband SED dataframe for one focused row."""

        row_map = self.row_to_mapping(row)
        column_overrides = dict(column_overrides or {})

        records: List[Dict[str, float | str]] = []
        skipped: List[Dict[str, str]] = []

        for band_name, spec in bands.items():
            if not self._is_enabled_band(spec):
                continue

            try:
                mag_value = self._resolve_reference(
                    band_name,
                    row=row_map,
                    column_overrides=column_overrides,
                )
            except KeyError:
                skipped.append(
                    {
                        "band": str(band_name),
                        "reason": f"magnitude column {band_name!r} is not mapped",
                    }
                )
                continue

            mag = self._normalise_number(mag_value)
            if mag is None or mag == -99:
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
                error_raw = self._resolve_reference(
                    spec.get("error", 0),
                    row=row_map,
                    column_overrides=column_overrides,
                )
            except KeyError as exc:
                skipped.append(
                    {
                        "band": str(band_name),
                        "reason": f"referenced column {exc.args[0]!r} is not mapped",
                    }
                )
                continue

            wavelength = self._normalise_number(wavelength_raw)
            fwhm = self._normalise_number(fwhm_raw)
            mag_error = self._normalise_number(error_raw)

            if wavelength is None or wavelength == -99 or wavelength <= 0:
                continue

            if fwhm is None:
                fwhm = 0.0
            if mag_error is None:
                mag_error = 0.0

            records.append(
                {
                    "band": str(band_name),
                    "wavelength (µm)": wavelength,
                    "magnitude": mag,
                    "FWHM": fwhm,
                    "error": mag_error,
                }
            )

        sed_df = pd.DataFrame(
            records,
            columns=["band", "wavelength (µm)", "magnitude", "FWHM", "error"],
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
        skipped: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> Dict[str, Any]:
        return {
            "dataset_id": dataset_id,
            "row_id": str(row_id),
            "sed_file": str(sed_file),
            "records": sed_df.to_dict(orient="records"),
            "column_overrides": dict(column_overrides),
            "skipped": [dict(item) for item in (skipped or [])],
        }


def create_sed_runtime(context=None, **kwargs) -> SEDRuntime:
    return SEDRuntime()