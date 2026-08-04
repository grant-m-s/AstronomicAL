from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import param
import time

from .constants import (
    DEFAULT_DATASHADE_THRESHOLD,
    DEFAULT_INTERACTIVE_SAMPLE_LIMIT,
    DEFAULT_MAX_SELECTION_IDS,
)
from .utils import (
    _active_dataset_id,
    _dataset_columns,
    _dataset_dtypes,
    _default_colour_map,
    _get_dataset_view_for_columns,
    _mapped_column,
    _numeric_columns_from_dataset,
)


NONE_COLOUR_OPTION = "None"
AUTO_COLOUR_MODE = "auto"
CATEGORICAL_COLOUR_MODE = "categorical"
CONTINUOUS_COLOUR_MODE = "continuous"


class VisualisationState(param.Parameterized):
    """Live UI state for visualisation plugin panels."""

    x = param.Selector(objects=[], default=None, allow_None=True)
    y = param.Selector(objects=[], default=None, allow_None=True)

    # Now means:
    #   "None"       -> no point colouring
    #   "<column>"   -> colour points by this dataset column
    #
    # Backwards compatibility:
    #   old saved state value "Labels" is mapped to label_col during restore.
    color_by = param.Selector(objects=[NONE_COLOUR_OPTION], default=NONE_COLOUR_OPTION)

    # Automatic mode is the default:
    #   string/category/bool/low-cardinality integer -> categorical legend
    #   float/high-cardinality numeric               -> continuous colourbar
    color_mode = param.Selector(
        objects=[AUTO_COLOUR_MODE, CATEGORICAL_COLOUR_MODE, CONTINUOUS_COLOUR_MODE],
        default=AUTO_COLOUR_MODE,
    )
    color_cmap = param.Selector(
        objects=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo"],
        default="Viridis",
    )

    # For categorical colour columns this filters the displayed classes.
    # For continuous colour columns this is kept as ["All"] and ignored.
    label_filter = param.ListSelector(objects=["All"], default=["All"])

    # Interactive is the default because range-aware sampling now works well:
    # it shows up to interactive_sample_limit points in the current view, and
    # zooming in can reveal all points if the visible region is below the limit.
    render_mode = param.Selector(
        objects=["auto", "interactive", "datashader"],
        default="interactive",
    )
    datashade_threshold = param.Integer(
        default=DEFAULT_DATASHADE_THRESHOLD,
        bounds=(1_000, 20_000_000),
    )
    interactive_sample_limit = param.Integer(
        default=DEFAULT_INTERACTIVE_SAMPLE_LIMIT,
        bounds=(1_000, 2_000_000),
    )
    density_interactive_sample_limit = param.Integer(
        default=100_000_000,
        bounds=(1_000, 100_000_000),
    )
    max_selection_ids = param.Integer(
        default=DEFAULT_MAX_SELECTION_IDS,
        bounds=(1_000, 5_000_000),
    )

    point_size = param.Number(default=5, bounds=(1, 20))
    point_alpha = param.Number(default=0.65, bounds=(0.05, 1.0))
    log_x = param.Boolean(default=False)
    log_y = param.Boolean(default=False)

    # Histogram bins.
    bins = param.Integer(default=35, bounds=(2, 300))

    # Density-specific bins. Default requested baseline is 50.
    density_bins = param.Integer(default=50, bounds=(2, 500))

    # Optional data-space density limits. None means derive from finite data.
    x_min = param.Number(default=None, allow_None=True)
    x_max = param.Number(default=None, allow_None=True)
    y_min = param.Number(default=None, allow_None=True)
    y_max = param.Number(default=None, allow_None=True)

    density = param.Boolean(default=False)
    cumulative = param.Boolean(default=False)
    log_density = param.Boolean(default=False)

    # Used both for legacy labels and generic categorical colour columns.
    max_label_values = param.Integer(default=200, bounds=(2, 10_000))

    def __init__(self, context, **params):
        super().__init__(**params)
        self.context = context
        self.dataset_id: Optional[str] = None

        self.numeric_columns: List[Any] = []
        self.colour_columns: List[Any] = []

        self.record_id_col: Optional[Any] = None
        self.label_col: Optional[Any] = None

        # Legacy label settings remain supported.
        self.labels_to_strings: Dict[Any, str] = {}
        self.strings_to_labels: Dict[str, Any] = {}
        self.label_colours: Dict[Any, str] = {}

        # Generic categorical colour settings for the currently selected colour column.
        self.colour_values_to_strings: Dict[Any, str] = {}
        self.strings_to_colour_values: Dict[str, Any] = {}
        self.colour_value_colours: Dict[Any, str] = {}

        self._colour_column_kind: Dict[str, str] = {}
        self._suppress_colour_state_refresh = False

        self._colour_watchers = []
        for name in ("color_by", "color_mode"):
            try:
                watcher = self.param.watch(self._on_colour_param_changed, name)
                self._colour_watchers.append(watcher)
            except Exception:
                pass

        self.refresh_from_context()

    def refresh_from_context(self) -> None:
        t0 = time.perf_counter()
        print(
            "[AstronomicAL visualisation] state.refresh_from_context start",
            flush=True,
        )

        dataset_id = _active_dataset_id(self.context)
        self.dataset_id = dataset_id
        columns = _dataset_columns(self.context, dataset_id)

        if dataset_id is None or not columns:
            self.numeric_columns = []
            self.colour_columns = []
            self.param.x.objects = []
            self.param.y.objects = []
            self.param.color_by.objects = [NONE_COLOUR_OPTION]
            self.x = None
            self.y = None
            self.color_by = NONE_COLOUR_OPTION
            self.record_id_col = None
            self.label_col = None
            self._reset_labels()
            self._reset_colour_values()
            return

        self.record_id_col = _mapped_column(
            self.context,
            dataset_id,
            "record_id",
            allow_index=True,
        )
        self.label_col = _mapped_column(
            self.context,
            dataset_id,
            "target_label",
            allow_index=False,
        )

        t_numeric = time.perf_counter()
        numeric_columns = _numeric_columns_from_dataset(self.context, dataset_id)
        print(
            "[AstronomicAL visualisation] numeric column discovery "
            f"columns={len(columns):,} "
            f"numeric={len(numeric_columns):,} "
            f"duration={time.perf_counter() - t_numeric:.2f}s",
            flush=True,
        )

        # Last-resort fallback. This avoids a blank UI if dtype inference fails.
        # It lets the user choose columns manually; prepare_plot_frame() will
        # still filter non-finite/non-numeric values later.
        if not numeric_columns:
            numeric_columns = [
                column for column in columns if column != self.record_id_col
            ]

        self.numeric_columns = list(numeric_columns)

        if not self.numeric_columns:
            self.param.x.objects = []
            self.param.y.objects = []
            self.x = None
            self.y = None
            self._refresh_colour_columns(dataset_id, columns)
            self._refresh_label_state_for_dataset(dataset_id)
            return

        self.param.x.objects = list(self.numeric_columns)
        self.param.y.objects = list(self.numeric_columns)

        preferred = [
            column for column in self.numeric_columns if column != self.record_id_col
        ] or list(self.numeric_columns)

        if self.x not in self.numeric_columns:
            self.x = preferred[0]
        if self.y not in self.numeric_columns:
            self.y = preferred[1] if len(preferred) > 1 else preferred[0]
        if len(preferred) > 1 and self.x == self.y:
            self.y = next((column for column in preferred if column != self.x), self.y)

        self._refresh_colour_columns(dataset_id, columns)
        self._refresh_label_state_for_dataset(dataset_id)
        self._refresh_colour_state_for_dataset(dataset_id)

        print(
            "[AstronomicAL visualisation] state.refresh_from_context end "
            f"duration={time.perf_counter() - t0:.2f}s "
            f"x={self.x!r} y={self.y!r} color_by={self.color_by!r}",
            flush=True,
        )

    def _on_colour_param_changed(self, _event=None) -> None:
        if self._suppress_colour_state_refresh:
            return
        self._refresh_colour_state_for_dataset(self.dataset_id)

    def _reset_labels(self) -> None:
        self.labels_to_strings = {}
        self.strings_to_labels = {}
        self.label_colours = {}

    def _reset_colour_values(self) -> None:
        self.colour_values_to_strings = {}
        self.strings_to_colour_values = {}
        self.colour_value_colours = {}
        self.param.label_filter.objects = ["All"]
        self.label_filter = ["All"]

    def _refresh_colour_columns(self, dataset_id: Optional[str], columns: List[Any]) -> None:
        if dataset_id is None:
            self.colour_columns = []
            self.param.color_by.objects = [NONE_COLOUR_OPTION]
            self.color_by = NONE_COLOUR_OPTION
            self._colour_column_kind = {}
            return

        excluded = {
            str(self.record_id_col or ""),
            "Use Index",
            "No Labels",
            "",
        }

        colour_columns = [
            column
            for column in columns
            if str(column) not in excluded
        ]

        self.colour_columns = list(colour_columns)
        self._colour_column_kind = self._infer_colour_column_kinds(
            dataset_id,
            colour_columns,
        )

        colour_options = [NONE_COLOUR_OPTION] + list(colour_columns)
        self.param.color_by.objects = colour_options

        if self.color_by == "Labels":
            # Legacy saved layout value.
            self.color_by = self.label_col if self.label_col in colour_columns else NONE_COLOUR_OPTION

        if self.color_by not in colour_options:
            # Preserve old default behaviour: if a target-label mapping exists,
            # colour by it automatically.
            if self.label_col in colour_columns:
                self.color_by = self.label_col
            else:
                self.color_by = NONE_COLOUR_OPTION

    def _infer_colour_column_kinds(
        self,
        dataset_id: Optional[str],
        columns: List[Any],
    ) -> Dict[str, str]:
        dtypes = _dataset_dtypes(self.context, dataset_id)
        numeric = set(_numeric_columns_from_dataset(self.context, dataset_id))
        kinds: Dict[str, str] = {}

        for column in columns:
            column_str = str(column)
            dtype_name = str(dtypes.get(column_str, "")).lower()

            if column_str not in numeric:
                kinds[column_str] = CATEGORICAL_COLOUR_MODE
                continue

            if "bool" in dtype_name:
                kinds[column_str] = CATEGORICAL_COLOUR_MODE
                continue

            # Integer columns are often class IDs. Treat genuinely low-cardinality
            # integer-like columns as categorical, otherwise continuous.
            if any(marker in dtype_name for marker in ("int", "integer", "uint", "bigint", "smallint", "tinyint")):
                try:
                    sample = _get_dataset_view_for_columns(
                        self.context,
                        dataset_id,
                        [column],
                        limit=100_000,
                    )
                    values = pd.unique(_safe_series(sample, column).dropna())
                    if 0 < len(values) <= int(self.max_label_values):
                        kinds[column_str] = CATEGORICAL_COLOUR_MODE
                    else:
                        kinds[column_str] = CONTINUOUS_COLOUR_MODE
                except Exception:
                    kinds[column_str] = CONTINUOUS_COLOUR_MODE
                continue

            kinds[column_str] = CONTINUOUS_COLOUR_MODE

        return kinds

    def effective_colour_mode(self) -> str:
        colour_col = self.colour_column()
        if colour_col is None:
            return "none"

        requested = str(self.color_mode or AUTO_COLOUR_MODE).lower()
        if requested in {CATEGORICAL_COLOUR_MODE, CONTINUOUS_COLOUR_MODE}:
            return requested

        return self._colour_column_kind.get(str(colour_col), CATEGORICAL_COLOUR_MODE)

    def colour_column(self) -> Optional[Any]:
        value = self.color_by
        if value is None:
            return None
        if str(value) in {"", NONE_COLOUR_OPTION, "No Labels"}:
            return None
        if str(value) == "Labels":
            return self.label_col
        return value

    def has_colour_column(self) -> bool:
        return self.colour_column() is not None

    def _refresh_colour_state_for_dataset(self, dataset_id: Optional[str]) -> None:
        colour_col = self.colour_column()
        mode = self.effective_colour_mode()

        if dataset_id is None or colour_col is None or mode != CATEGORICAL_COLOUR_MODE:
            self._reset_colour_values()
            return

        try:
            colour_df = _get_dataset_view_for_columns(
                self.context,
                dataset_id,
                [colour_col],
                limit=100_000,
            )
        except Exception:
            colour_df = pd.DataFrame(columns=[colour_col])

        raw_values = _safe_unique_labels(
            colour_df,
            colour_col,
            self.max_label_values,
        )

        if raw_values is None:
            self._reset_colour_values()
            return

        # If this is the mapped target-label column, respect user label settings.
        if self.label_col is not None and str(colour_col) == str(self.label_col) and self.labels_to_strings:
            values_to_strings = dict(self.labels_to_strings)
            strings_to_values = dict(self.strings_to_labels) or {
                str(display): raw for raw, display in values_to_strings.items()
            }
            value_colours = dict(self.label_colours) or _default_colour_map(
                list(values_to_strings.keys())
            )
        else:
            values_to_strings = {raw: str(raw) for raw in raw_values}
            strings_to_values = {
                str(display): raw for raw, display in values_to_strings.items()
            }
            value_colours = _default_colour_map(raw_values)

        self.colour_values_to_strings = values_to_strings
        self.strings_to_colour_values = strings_to_values
        self.colour_value_colours = value_colours

        label_options = ["All"] + sorted(
            [str(key) for key in self.strings_to_colour_values.keys()]
        )

        old_suppress = self._suppress_colour_state_refresh
        self._suppress_colour_state_refresh = True
        try:
            self.param.label_filter.objects = label_options
            current = [
                value for value in list(self.label_filter or []) if value in label_options
            ]
            self.label_filter = current or ["All"]
        finally:
            self._suppress_colour_state_refresh = old_suppress

    def _refresh_label_state(self, df: pd.DataFrame) -> None:
        cfg = getattr(self.context, "config", None)
        settings = getattr(cfg, "settings", {}) if cfg is not None else {}
        labels_to_strings = dict(settings.get("labels_to_strings", {}) or {})
        strings_to_labels = dict(settings.get("strings_to_labels", {}) or {})
        label_colours = dict(settings.get("label_colours", {}) or {})

        if self.label_col and self.label_col in df.columns:
            raw_labels = _safe_unique_labels(df, self.label_col, self.max_label_values)
            if raw_labels is None:
                self._reset_labels()
                return

            if not labels_to_strings:
                labels_to_strings = {raw: str(raw) for raw in raw_labels}
            if not strings_to_labels:
                strings_to_labels = {
                    str(display): raw for raw, display in labels_to_strings.items()
                }
            if not label_colours:
                label_colours = _default_colour_map(raw_labels)

        if not labels_to_strings and not strings_to_labels:
            self._reset_labels()
            return

        self.labels_to_strings = labels_to_strings
        self.strings_to_labels = strings_to_labels or {
            str(display): raw for raw, display in labels_to_strings.items()
        }
        self.label_colours = label_colours or _default_colour_map(
            list(self.labels_to_strings.keys())
        )

    def _refresh_label_state_for_dataset(self, dataset_id: Optional[str]) -> None:
        if not self.label_col or dataset_id is None:
            self._reset_labels()
            return

        try:
            label_df = _get_dataset_view_for_columns(
                self.context,
                dataset_id,
                [self.label_col],
                limit=100_000,
            )
        except Exception:
            label_df = pd.DataFrame(columns=[self.label_col])

        self._refresh_label_state(label_df)

    def apply_label_settings(self, payload: Optional[Dict[str, Any]]) -> None:
        if not isinstance(payload, dict):
            return

        dataset_id = _active_dataset_id(self.context)
        columns = _dataset_columns(self.context, dataset_id)
        label_col = (
            payload.get("label_col")
            or payload.get("label_column")
            or payload.get("active_label_col")
            or payload.get("active_label_column")
            or payload.get("target_label")
        )

        if label_col and label_col in columns:
            self.label_col = label_col
            if self.color_by == "Labels":
                self.color_by = label_col

        labels_to_strings = (
            payload.get("labels_to_strings")
            or payload.get("label_to_string")
            or payload.get("label_strings")
            or payload.get("labels")
            or {}
        )
        strings_to_labels = (
            payload.get("strings_to_labels")
            or payload.get("string_to_label")
            or payload.get("reverse_labels")
            or {}
        )
        label_colours = (
            payload.get("label_colours")
            or payload.get("label_colors")
            or payload.get("colours")
            or payload.get("colors")
            or {}
        )

        if labels_to_strings:
            self.labels_to_strings = dict(labels_to_strings)
        if strings_to_labels:
            self.strings_to_labels = dict(strings_to_labels)
        elif self.labels_to_strings:
            self.strings_to_labels = {
                str(display): raw for raw, display in self.labels_to_strings.items()
            }

        if label_colours:
            self.label_colours = dict(label_colours)
        elif self.labels_to_strings:
            self.label_colours = _default_colour_map(list(self.labels_to_strings.keys()))

        if not self.strings_to_labels:
            self._reset_labels()

        self._refresh_colour_state_for_dataset(dataset_id)

    def label_display(self, raw: Any) -> str:
        return str(self.labels_to_strings.get(raw, self.labels_to_strings.get(str(raw), raw)))

    def label_colour(self, raw: Any) -> str:
        return str(self.label_colours.get(raw, self.label_colours.get(str(raw), "#1f77b4")))

    def selected_raw_labels(self) -> List[Any]:
        # Legacy compatibility. Now delegates to the active categorical colour column.
        return self.selected_colour_raw_values()

    def colour_display(self, raw: Any) -> str:
        return str(
            self.colour_values_to_strings.get(
                raw,
                self.colour_values_to_strings.get(str(raw), raw),
            )
        )

    def colour_value_colour(self, raw: Any) -> str:
        return str(
            self.colour_value_colours.get(
                raw,
                self.colour_value_colours.get(str(raw), "#1f77b4"),
            )
        )

    def selected_colour_raw_values(self) -> List[Any]:
        if "All" in (self.label_filter or []):
            return list(self.strings_to_colour_values.values())
        return [
            self.strings_to_colour_values[name]
            for name in self.label_filter
            if name in self.strings_to_colour_values
        ]

    def colour_status_text(self) -> str:
        colour_col = self.colour_column()
        if colour_col is None:
            return "plain"
        mode = self.effective_colour_mode()
        if mode == CATEGORICAL_COLOUR_MODE:
            return f"coloured by {colour_col} categorical"
        if mode == CONTINUOUS_COLOUR_MODE:
            return f"coloured by {colour_col} continuous"
        return f"coloured by {colour_col}"

    def get_state(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "color_by": self.color_by,
            "color_mode": self.color_mode,
            "color_cmap": self.color_cmap,
            "label_filter": list(self.label_filter or []),
            "render_mode": self.render_mode,
            "datashade_threshold": self.datashade_threshold,
            "interactive_sample_limit": self.interactive_sample_limit,
            "max_selection_ids": self.max_selection_ids,
            "point_size": self.point_size,
            "point_alpha": self.point_alpha,
            "log_x": self.log_x,
            "log_y": self.log_y,
            "bins": self.bins,
            "density_bins": self.density_bins,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "density": self.density,
            "cumulative": self.cumulative,
            "log_density": self.log_density,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        self.refresh_from_context()

        restored = dict(state)

        # Old layouts saved color_by="Labels". Map that to the mapped target-label
        # column where possible.
        if restored.get("color_by") == "Labels":
            restored["color_by"] = self.label_col if self.label_col else NONE_COLOUR_OPTION

        old_suppress = self._suppress_colour_state_refresh
        self._suppress_colour_state_refresh = True
        try:
            for key, value in restored.items():
                if key not in self.param:
                    continue
                try:
                    setattr(self, key, value)
                except Exception:
                    pass
        finally:
            self._suppress_colour_state_refresh = old_suppress

        self._refresh_colour_state_for_dataset(self.dataset_id)


def _safe_series(df: pd.DataFrame, column: Any) -> pd.Series:
    values = df[column]
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0]
    return values


def _safe_unique_labels(
    df: pd.DataFrame,
    label_col: Any,
    max_values: int,
) -> Optional[List[Any]]:
    try:
        series = _safe_series(df, label_col).dropna()
        values = list(pd.unique(series))
    except Exception:
        return []

    if len(values) > int(max_values):
        return None

    return sorted(values, key=lambda value: str(value))