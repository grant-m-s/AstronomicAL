from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    _default_colour_map,
    _mapped_column,
    _numeric_columns_from_dataset,
    _get_dataset_view_for_columns,
)


class VisualisationState(param.Parameterized):
    """Live UI state for visualisation plugin panels."""

    x = param.Selector(objects=[], default=None, allow_None=True)
    y = param.Selector(objects=[], default=None, allow_None=True)

    label_filter = param.ListSelector(objects=["All"], default=["All"])
    color_by = param.Selector(objects=["None", "Labels"], default="Labels")

    # Interactive is the default because range-aware sampling now works well:
    # it shows up to interactive_sample_limit points in the current view, and
    # zooming in can reveal all points if the visible region is below the limit.
    render_mode = param.Selector(objects=["auto", "interactive", "datashader"], default="interactive")

    datashade_threshold = param.Integer(
        default=DEFAULT_DATASHADE_THRESHOLD,
        bounds=(1_000, 20_000_000),
    )

    interactive_sample_limit = param.Integer(
        default=DEFAULT_INTERACTIVE_SAMPLE_LIMIT,
        bounds=(1_000, 2_000_000),
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

    density = param.Boolean(default=False)
    cumulative = param.Boolean(default=False)
    log_density = param.Boolean(default=False)

    max_label_values = param.Integer(default=200, bounds=(2, 10_000))

    def __init__(self, context, **params):
        super().__init__(**params)
        self.context = context

        self.dataset_id: Optional[str] = None
        self.numeric_columns: List[Any] = []

        self.record_id_col: Optional[Any] = None
        self.label_col: Optional[Any] = None

        self.labels_to_strings: Dict[Any, str] = {}
        self.strings_to_labels: Dict[str, Any] = {}
        self.label_colours: Dict[Any, str] = {}

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
            self.param.x.objects = []
            self.param.y.objects = []
            self.x = None
            self.y = None
            self.record_id_col = None
            self.label_col = None
            self._reset_labels()
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

        numeric_columns: List[Any] = []

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
        # It lets the user choose columns manually; prepare_plot_frame() will still
        # filter non-finite/non-numeric values later.
        if not numeric_columns:
            numeric_columns = [
                column for column in columns
                if column != self.record_id_col
            ]

        self.numeric_columns = list(numeric_columns)

        if not self.numeric_columns:
            self.param.x.objects = []
            self.param.y.objects = []
            self.x = None
            self.y = None
            self._refresh_label_state_for_dataset(dataset_id)
            return

        self.param.x.objects = list(self.numeric_columns)
        self.param.y.objects = list(self.numeric_columns)

        preferred = [
            column for column in self.numeric_columns
            if column != self.record_id_col
        ] or list(self.numeric_columns)

        if self.x not in self.numeric_columns:
            self.x = preferred[0]

        if self.y not in self.numeric_columns:
            self.y = preferred[1] if len(preferred) > 1 else preferred[0]

        if len(preferred) > 1 and self.x == self.y:
            self.y = next((column for column in preferred if column != self.x), self.y)

        self._refresh_label_state_for_dataset(dataset_id)

        print(
            "[AstronomicAL visualisation] state.refresh_from_context end "
            f"duration={time.perf_counter() - t0:.2f}s "
            f"x={self.x!r} y={self.y!r}",
            flush=True,
        )

    def _reset_labels(self) -> None:
        self.labels_to_strings = {}
        self.strings_to_labels = {}
        self.label_colours = {}

        self.param.label_filter.objects = ["All"]
        self.label_filter = ["All"]

        self.param.color_by.objects = ["None"]
        self.color_by = "None"

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

        label_options = ["All"] + sorted([str(key) for key in self.strings_to_labels.keys()])
        self.param.label_filter.objects = label_options

        current = [
            value for value in list(self.label_filter or []) if value in label_options
        ]
        self.label_filter = current or ["All"]

        colour_options = ["None"] + (["Labels"] if len(label_options) > 1 else [])
        self.param.color_by.objects = colour_options

        if self.color_by not in colour_options:
            self.color_by = "Labels" if "Labels" in colour_options else "None"

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
            return

        label_options = ["All"] + sorted([str(key) for key in self.strings_to_labels.keys()])
        self.param.label_filter.objects = label_options

        current = [
            value for value in list(self.label_filter or []) if value in label_options
        ]
        self.label_filter = current or ["All"]

        colour_options = ["None"] + (["Labels"] if len(label_options) > 1 else [])
        self.param.color_by.objects = colour_options

        if "Labels" in colour_options:
            self.color_by = "Labels"
        elif self.color_by not in colour_options:
            self.color_by = "None"

    def label_display(self, raw: Any) -> str:
        return str(self.labels_to_strings.get(raw, self.labels_to_strings.get(str(raw), raw)))

    def label_colour(self, raw: Any) -> str:
        return str(self.label_colours.get(raw, self.label_colours.get(str(raw), "#1f77b4")))

    def selected_raw_labels(self) -> List[Any]:
        if "All" in (self.label_filter or []):
            return list(self.strings_to_labels.values())

        return [
            self.strings_to_labels[name]
            for name in self.label_filter
            if name in self.strings_to_labels
        ]

    def get_state(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "label_filter": list(self.label_filter or []),
            "color_by": self.color_by,
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
            "density": self.density,
            "cumulative": self.cumulative,
            "log_density": self.log_density,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        self.refresh_from_context()

        for key, value in state.items():
            if key not in self.param:
                continue

            try:
                setattr(self, key, value)
            except Exception:
                pass


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