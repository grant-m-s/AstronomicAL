from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import traceback
import uuid

import numpy as np
import panel as pn
import holoviews as hv

try:
    hv.extension("bokeh")
except Exception:
    pass

from .service import DESI_DATASETS, SDSS_DATASETS, SpectraResult, SpectraRuntime


PLUGIN_ID = "astro.spectra"
RUNTIME_SERVICE_KEY = f"{PLUGIN_ID}.runtime"
SETTINGS_HEIGHT = 118


SOURCE_LABELS = {
    "DESI": "DESI",
    "SDSS": "SDSS/BOSS",
    "EuclidSpec": "Euclid",
}


def _style_widget(
    widget: Any,
    *,
    width: int = 145,
    height: int = 40,
    margin: Tuple[int, int, int, int] = (0, 6, 2, 6),
) -> Any:
    try:
        widget.width = width
        widget.height = height
        widget.sizing_mode = "fixed"
        widget.margin = margin
    except Exception:
        pass
    return widget


def _settings_box(*controls: Any) -> pn.FlexBox:
    return pn.FlexBox(
        *controls,
        sizing_mode="stretch_width",
        height_policy="fit",
        margin=(0, 0, 0, 0),
        styles={
            "overflow": "visible",
            "align-content": "flex-start",
            "align-items": "flex-start",
            "gap": "2px 6px",
            "padding": "4px 6px 4px 6px",
            "border-top": "1px solid #ddd",
            "border-bottom": "1px solid #eee",
            "background": "#fafafa",
            "box-sizing": "border-box",
        },
    )


def _small_label(text: str, *, width: int = 76) -> pn.pane.HTML:
    return pn.pane.HTML(
        f"<div style='font-size:11px;font-weight:600;color:#555;"
        f"padding-top:11px;white-space:nowrap'>{text}</div>",
        width=width,
        height=34,
        sizing_mode="fixed",
        margin=(0, 2, 0, 6),
    )


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


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    try:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return [arr.item()]
        out = arr.tolist()
        return out if isinstance(out, list) else [out]
    except Exception:
        try:
            return list(value)
        except Exception:
            return [value]


@dataclass
class _ResolvedTarget:
    dataset_id: str
    row_id: Optional[str]
    row: Dict[str, Any]
    ra: Optional[float]
    dec: Optional[float]
    source_id: Optional[Any]
    id_column: Optional[str]
    ra_column: Optional[str]
    dec_column: Optional[str]
    target_id_column: Optional[str]
    retrieval_mode: str


class SpectraPanel:
    """Plugin-native DESI/SDSS/Euclid spectra panel."""

    def __init__(
        self,
        *,
        context: Any,
        source: str,
        data: Any = None,
        state: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        self.context = context
        self.source = source
        self.source_label = SOURCE_LABELS.get(source, source)
        self.data = data
        self.panel_id = f"{PLUGIN_ID}.{source}.{uuid.uuid4().hex}"

        self._subscriptions: List[Any] = []
        self._job_handle: Any = None
        self._disposed = False
        self._initial_load_started = False
        self._settings_built = False
        self.settings_visible = False

        self._current_target: Optional[_ResolvedTarget] = None
        self._spectra_result: Optional[SpectraResult] = None

        self._build_widgets()
        if state:
            self.restore_state(state)
        self._build_layout()
        self._bind_events()

    # ------------------------------------------------------------------
    # Plugin controller API
    # ------------------------------------------------------------------

    def view(self) -> pn.viewable.Viewable:
        self._schedule_initial_load()
        return self.layout

    def panel(self) -> pn.viewable.Viewable:
        return self.view()

    def dispose(self) -> None:
        self._disposed = True
        self._cancel_job()

        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in list(self._subscriptions):
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions.clear()

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "settings_visible": self.settings_visible,
            "retrieval_mode": self.retrieve_mode.value,
            "max_separation_arcsec": self.max_separation_input.value,
            "target_id_column": self.target_id_column.value,
            "plot_lines": self.plot_lines_checkbox.value,
            "plot_model": self.plot_model_checkbox.value,
            "smoothing_function": self.smoothing_function_input.value,
            "smoothing_window": self.smoothing_window_input.value,
            "redshift": self.redshift_input.value,
            "redshift_column": self.redshift_column_selector.value,
            "auto_reload": self.auto_reload.value,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        self.settings_visible = bool(state.get("settings_visible", False))

        mapping = {
            "retrieval_mode": self.retrieve_mode,
            "max_separation_arcsec": self.max_separation_input,
            "target_id_column": self.target_id_column,
            "plot_lines": self.plot_lines_checkbox,
            "plot_model": self.plot_model_checkbox,
            "smoothing_function": self.smoothing_function_input,
            "smoothing_window": self.smoothing_window_input,
            "redshift": self.redshift_input,
            "redshift_column": self.redshift_column_selector,
            "auto_reload": self.auto_reload,
        }

        for key, widget in mapping.items():
            if key in state:
                try:
                    if key in {"target_id_column", "redshift_column"}:
                        current_options = list(widget.options)
                        if state[key] not in current_options:
                            widget.options = current_options + [state[key]]
                    widget.value = state[key]
                except Exception:
                    pass

        try:
            self._apply_settings_visibility()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        self.status = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            margin=(0, 8, 0, 8),
            styles={
                "font-size": "12px",
                "line-height": "1.25",
                "max-height": "42px",
                "overflow": "auto",
            },
        )

        self.target_status = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            height=34,
            margin=(0, 8, 0, 8),
            styles={
                "font-size": "12px",
                "line-height": "1.2",
                "overflow": "hidden",
                "white-space": "nowrap",
                "text-overflow": "ellipsis",
                "color": "#333",
            },
        )

        self.figure = pn.Column(
            self._empty_message("No spectrum loaded."),
            sizing_mode="stretch_both",
            min_height=320,
            margin=(0, 6, 6, 6),
            styles={"overflow": "hidden"},
        )

        self.title_pane = pn.pane.HTML(
            f"<div style='font-size:14px;font-weight:700;padding-top:10px'>{self.source_label} Spectra</div>",
            width=120,
            height=44,
            sizing_mode="fixed",
            margin=(0, 2, 0, 0),
        )

        self.retrieve_mode = pn.widgets.RadioButtonGroup(
            name="Retrieval",
            options=["Cone Search", "Use TargetId"],
            value="Cone Search",
            button_type="default",
            width=210,
            height=34,
            sizing_mode="fixed",
            margin=(8, 2, 0, 0),
        )

        self.max_separation_input = pn.widgets.FloatInput(
            name="Radius [arcsec]",
            value=1.0 if self.source != "EuclidSpec" else 0.5,
            start=0.01,
            step=0.5,
            width=122,
            height=44,
            sizing_mode="fixed",
            margin=(0, 2, 0, 0),
        )

        self.load_button = pn.widgets.Button(
            name="Load",
            button_type="primary",
            width=74,
            height=34,
            sizing_mode="fixed",
            margin=(8, 0, 0, 0),
        )

        self.settings_button = pn.widgets.Button(
            name="⚙",
            width=32,
            height=32,
            button_type="default",
            sizing_mode="fixed",
            margin=(8, 0, 0, 0),
        )

        self.target_id_column = _style_widget(
            pn.widgets.Select(
                name="Target ID column",
                options=["Auto"],
                value="Auto",
            ),
            width=180,
        )

        self.plot_lines_checkbox = _style_widget(
            pn.widgets.Checkbox(
                name="Line markers",
                value=self.source != "EuclidSpec",
            ),
            width=110,
            height=28,
            margin=(10, 8, 0, 6),
        )

        self.plot_model_checkbox = _style_widget(
            pn.widgets.Checkbox(
                name="Model",
                value=self.source != "EuclidSpec",
            ),
            width=85,
            height=28,
            margin=(10, 8, 0, 6),
        )

        self.auto_reload = _style_widget(
            pn.widgets.Checkbox(name="Auto reload", value=True),
            width=110,
            height=28,
            margin=(10, 8, 0, 6),
        )

        self.smoothing_function_input = _style_widget(
            pn.widgets.Select(
                name="Smoothing",
                options={"Box": "Box1DKernel", "Gaussian": "Gaussian1DKernel"},
                value="Box1DKernel",
            ),
            width=135,
        )

        self.smoothing_window_input = _style_widget(
            pn.widgets.IntInput(
                name="Window",
                value=10 if self.source != "EuclidSpec" else 5,
                start=1,
                end=100,
                step=1,
            ),
            width=95,
        )

        self.redshift_input = _style_widget(
            pn.widgets.FloatInput(
                name="Assign redshift",
                value=None,
                start=0.0,
                end=15.0,
                step=0.001,
            ),
            width=130,
        )

        self.redshift_column_selector = _style_widget(
            pn.widgets.Select(
                name="Redshift column",
                options=["None"],
                value="None",
            ),
            width=170,
        )

        self.query_redshift_button = pn.widgets.Button(
            name="Query Euclid redshift",
            button_type="primary",
            width=160,
            height=34,
            sizing_mode="fixed",
            margin=(6, 6, 0, 6),
            disabled=self.source != "EuclidSpec",
        )

        self.refresh_plot_button = pn.widgets.Button(
            name="Refresh plot",
            width=110,
            height=34,
            sizing_mode="fixed",
            margin=(6, 6, 0, 6),
        )

        if self.source == "EuclidSpec":
            self.plot_lines_checkbox.disabled = True
            self.plot_model_checkbox.disabled = True

        self.load_button.on_click(lambda _event: self.load_spectra(reason="button.load"))
        self.settings_button.on_click(self._toggle_settings)
        self.query_redshift_button.on_click(self._query_euclid_redshift)
        self.refresh_plot_button.on_click(lambda _event: self._render_existing_result())
        self.retrieve_mode.param.watch(self._retrieve_mode_changed, "value")

        for widget in [
            self.plot_lines_checkbox,
            self.plot_model_checkbox,
            self.smoothing_function_input,
            self.smoothing_window_input,
            self.redshift_input,
            self.redshift_column_selector,
        ]:
            widget.param.watch(lambda _event: self._render_existing_result(), "value")

    def _header(self) -> pn.GridBox:
        return pn.GridBox(
            self.title_pane,
            self.retrieve_mode,
            self.max_separation_input,
            self.load_button,
            self.settings_button,
            ncols=5,
            sizing_mode="stretch_width",
            height=48,
            margin=(0, 6, 0, 6),
            styles={
                "display": "grid",
                "grid-template-columns": "126px 216px 126px 78px 34px",
                "gap": "4px",
                "align-items": "start",
                "box-sizing": "border-box",
            },
        )

    def _settings_controls(self) -> pn.FlexBox:
        return _settings_box(
            _small_label("Request", width=58),
            self.target_id_column,
            self.auto_reload,
            _small_label("Plot", width=38),
            self.plot_lines_checkbox,
            self.plot_model_checkbox,
            self.refresh_plot_button,
            _small_label("Smooth", width=54),
            self.smoothing_function_input,
            self.smoothing_window_input,
            _small_label("Redshift", width=62),
            self.redshift_input,
            self.redshift_column_selector,
            self.query_redshift_button,
        )

    def _ensure_settings_built(self) -> None:
        if self._settings_built:
            return
        self.settings_pane[:] = [self._settings_controls()]
        self._settings_built = True

    def _apply_settings_visibility(self) -> None:
        self._ensure_settings_built()
        self.settings_pane.visible = self.settings_visible
        try:
            self.settings_button.button_type = "primary" if self.settings_visible else "default"
        except Exception:
            pass

    def _toggle_settings(self, _event: Any = None) -> None:
        self.settings_visible = not self.settings_visible
        self._apply_settings_visibility()

    def _build_layout(self) -> None:
        self.settings_pane = pn.Column(
            sizing_mode="stretch_width",
            height=SETTINGS_HEIGHT,
            min_height=SETTINGS_HEIGHT,
            max_height=SETTINGS_HEIGHT,
            height_policy="fixed",
            visible=False,
            margin=(0, 0, 0, 0),
            styles={
                "height": f"{SETTINGS_HEIGHT}px",
                "min-height": f"{SETTINGS_HEIGHT}px",
                "max-height": f"{SETTINGS_HEIGHT}px",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "box-sizing": "border-box",
            },
        )

        self._ensure_settings_built()
        self._apply_settings_visibility()

        self.layout = pn.Column(
            self._header(),
            self.settings_pane,
            self.target_status,
            self.status,
            self.figure,
            sizing_mode="stretch_both",
            height_policy="max",
            min_height=0,
            margin=(0, 0, 0, 0),
            styles={
                "min-height": "0",
                "overflow": "hidden",
                "box-sizing": "border-box",
            },
        )

    @staticmethod
    def _empty_message(text: str) -> pn.pane.Markdown:
        return pn.pane.Markdown(
            f"### {text}",
            sizing_mode="stretch_both",
            margin=(16, 16, 16, 16),
            styles={"color": "#666"},
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _bind_events(self) -> None:
        if getattr(self.context, "events", None) is None:
            return

        self._subscribe("selection.focus.changed", self._selection_changed)
        self._subscribe("selection.focus.cleared", self._selection_cleared)
        self._subscribe("dataset.active.changed", self._dataset_changed)
        self._subscribe("dataset.mapping_updated", self._dataset_changed)
        self._subscribe("astro.euclid.radius.changed", self._euclid_radius_changed)

    def _subscribe(self, topic: str, callback: Any) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        try:
            sub = events.subscribe(
                topic,
                callback,
                owner_id=self.panel_id,
                owner_label=f"{self.source_label} Spectra",
                owner_kind="panel",
            )
        except TypeError:
            sub = events.subscribe(topic, callback)

        self._subscriptions.append(sub)

    def _publish(self, topic: str, payload: Optional[Dict[str, Any]] = None) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        try:
            events.publish(topic, payload or {})
        except Exception:
            traceback.print_exc()

    def _selection_changed(self, topic: str, payload: Any) -> None:
        self._cancel_job()
        self._current_target = None
        self._spectra_result = None
        self.figure[:] = [self._empty_message("Loading new selection…")]
        self._refresh_column_options()
        self._update_target_status()

        if self.auto_reload.value:
            self.load_spectra(reason=str(topic or "selection.focus.changed"))

    def _selection_cleared(self, topic: str, payload: Any) -> None:
        self._cancel_job()
        self._current_target = None
        self._spectra_result = None
        self.status.object = "No focused row selected."
        self.target_status.object = ""
        self.figure[:] = [self._empty_message("No focused row selected.")]

    def _dataset_changed(self, topic: str, payload: Any) -> None:
        self._cancel_job()
        self._current_target = None
        self._spectra_result = None
        self.figure[:] = [self._empty_message("Dataset changed.")]
        self._refresh_column_options()
        self._update_target_status()

        if self.auto_reload.value:
            self.load_spectra(reason=str(topic or "dataset.changed"))

    def _euclid_radius_changed(self, topic: str, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        radius = _safe_float(payload.get("radius"))
        if radius is None:
            return
        if self.source == "EuclidSpec" and self.max_separation_input.value != radius:
            self.max_separation_input.value = radius

    # ------------------------------------------------------------------
    # Dataset / selection helpers
    # ------------------------------------------------------------------

    def _active_dataset_id(self) -> str:
        datasets = getattr(self.context, "datasets", None)
        if datasets is not None:
            for name in ("active_id", "get_active_id"):
                method = getattr(datasets, name, None)
                if callable(method):
                    try:
                        active = method()
                        if active:
                            return str(active)
                    except Exception:
                        pass
        return "default"

    def _focus_state(self) -> Any:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None

        method = getattr(selection, "get_focus", None)
        if callable(method):
            try:
                return method()
            except Exception:
                return None

        return None

    @staticmethod
    def _get_from_obj(obj: Any, *names: str) -> Any:
        if obj is None:
            return None

        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]
            return None

        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)

        return None

    def _mapping(self, dataset_id: str, *roles: str) -> Optional[str]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return None

        for role in roles:
            for method_name in ("get_mapping", "mapping", "get_column_mapping"):
                method = getattr(datasets, method_name, None)
                if not callable(method):
                    continue

                for args in ((dataset_id, role), (role,), (dataset_id, role, None)):
                    try:
                        value = method(*args)
                    except TypeError:
                        continue
                    except Exception:
                        value = None

                    if value:
                        return str(value)

        config = getattr(self.context, "config", None)
        settings = getattr(config, "settings", {}) if config is not None else {}

        aliases = {
            "record_id": ["id_col", "id", "ID", "source_id", "object_id"],
            "coords.ra": ["ra_dec", "ra", "RA", "Right Ascension"],
            "coords.dec": ["ra_dec", "dec", "DEC", "Declination"],
        }

        for role in roles:
            for key in aliases.get(role, [role]):
                value = settings.get(key)
                if isinstance(value, str):
                    if role == "coords.ra" and "," in value:
                        return value.split(",", 1)[0].strip()
                    if role == "coords.dec" and "," in value:
                        return value.split(",", 1)[1].strip()
                    return value

        return None

    def _columns(self, dataset_id: str) -> List[str]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return []

        for method_name in ("list_columns", "columns"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    return list(method(dataset_id))
                except TypeError:
                    try:
                        return list(method())
                    except Exception:
                        pass
                except Exception:
                    pass

        try:
            df = datasets.get_df(dataset_id)
            return list(df.columns)
        except Exception:
            return []

    def _guess_column(self, dataset_id: str, candidates: Iterable[str]) -> Optional[str]:
        columns = self._columns(dataset_id)
        lower_map = {str(col).lower(): str(col) for col in columns}

        for candidate in candidates:
            if candidate in columns:
                return candidate
            found = lower_map.get(str(candidate).lower())
            if found:
                return found

        return None

    def _target_column_candidates(self) -> List[str]:
        if self.source == "DESI":
            return [
                "DESI_TargetID",
                "DESI_targetid",
                "DESI_specid",
                "specid",
                "targetid",
                "TARGETID",
                "TARGET_ID",
            ]

        if self.source == "SDSS":
            return [
                "SDSS_TargetID",
                "SDSS_specid",
                "specid",
                "specObjID",
                "specobjid",
                "plate_mjd_fiberid",
            ]

        return [
            "EuclidSpec_TargetID",
            "Euclid_source_id",
            "euclid_source_id",
            "source_id",
            "sourceId",
            "object_id",
            "SOURCE_ID",
        ]

    def _refresh_column_options(self) -> None:
        dataset_id = self._active_dataset_id()
        columns = self._columns(dataset_id)

        current_target = self.target_id_column.value
        target_options = ["Auto"] + columns
        if current_target not in target_options:
            target_options.append(current_target)
        self.target_id_column.options = target_options
        self.target_id_column.value = current_target if current_target in target_options else "Auto"

        current_redshift = self.redshift_column_selector.value
        redshift_options = ["None"] + columns
        if current_redshift not in redshift_options:
            redshift_options.append(current_redshift)
        self.redshift_column_selector.options = redshift_options
        self.redshift_column_selector.value = (
            current_redshift if current_redshift in redshift_options else "None"
        )

    def _current_row_id(self) -> Optional[str]:
        if self._current_target is not None:
            return self._current_target.row_id

        focus = self._focus_state()
        value = self._get_from_obj(focus, "row_id", "record_id", "id", "source_id")
        return None if value is None else str(value)

    def _target_matches_current_focus(self, target: _ResolvedTarget) -> bool:
        focus = self._focus_state()
        if focus is None:
            return True

        focus_dataset = self._get_from_obj(focus, "dataset_id", "dataset")
        focus_row_id = self._get_from_obj(focus, "row_id", "record_id", "id", "source_id")

        if focus_dataset is not None and str(focus_dataset) != str(target.dataset_id):
            return False

        if focus_row_id is not None:
            if target.row_id is None:
                return False
            if str(focus_row_id) != str(target.row_id):
                return False

        return True

    def _resolve_target(self) -> _ResolvedTarget:
        dataset_id = self._active_dataset_id()
        focus = self._focus_state()

        focus_dataset = self._get_from_obj(focus, "dataset_id", "dataset")
        if focus_dataset:
            dataset_id = str(focus_dataset)

        metadata = self._get_from_obj(focus, "metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        id_column = self._mapping(dataset_id, "record_id", "id", "row_id")
        ra_column = self._mapping(dataset_id, "coords.ra", "ra") or self._guess_column(
            dataset_id,
            ["ra", "RA", "right_ascension", "alpha", "source_ra"],
        )
        dec_column = self._mapping(dataset_id, "coords.dec", "dec") or self._guess_column(
            dataset_id,
            ["dec", "DEC", "declination", "delta", "source_dec"],
        )

        if id_column is None:
            id_column = self._guess_column(
                dataset_id,
                ["id", "ID", "source_id", "object_id", "row_id"],
            )

        row_id = self._get_from_obj(focus, "row_id", "record_id", "id", "source_id")
        row_pos = self._get_from_obj(focus, "row_pos", "row_index", "position", "index")

        if row_pos is None:
            row_pos = (
                metadata.get("row_pos")
                or metadata.get("row_index")
                or metadata.get("position")
                or metadata.get("index")
            )

        required_columns = [col for col in [ra_column, dec_column, id_column] if col]
        target_column = self._target_id_column(dataset_id)
        if target_column and target_column not in required_columns:
            required_columns.append(target_column)

        redshift_column = self.redshift_column_selector.value
        if redshift_column and redshift_column != "None" and redshift_column not in required_columns:
            required_columns.append(redshift_column)

        row = None

        for key in ("row", "record", "row_data", "data"):
            candidate = metadata.get(key)
            if isinstance(candidate, dict):
                if row_id is None or id_column is None:
                    row = dict(candidate)
                    break
                candidate_id = candidate.get(id_column)
                if candidate_id is not None and str(candidate_id) == str(row_id):
                    row = dict(candidate)
                    break

        if row is None:
            row = self._fetch_row(
                dataset_id,
                id_column=id_column,
                row_id=row_id,
                row_pos=row_pos,
                required_columns=required_columns,
            )

        if row is None:
            row = self._row_from_legacy_data(
                id_column=id_column,
                row_id=row_id,
                allow_first_row=(row_id is None),
            )

        if row is None:
            raise RuntimeError("No focused row is available for the spectra panel.")

        if row_id is None and id_column and id_column != "Use Index" and id_column in row:
            row_id = row.get(id_column)

        row_id_str = None if row_id is None else str(row_id)

        retrieval_mode = self.retrieve_mode.value
        source_id = None
        ra = None
        dec = None

        if retrieval_mode == "Use TargetId":
            source_id = self._target_id_from_row(dataset_id, row, target_column)
            if source_id is None:
                raise RuntimeError(
                    f"{self.source_label} target-id mode requires a target/spec ID column."
                )

        if ra_column and dec_column and ra_column in row and dec_column in row:
            ra = _safe_float(row.get(ra_column))
            dec = _safe_float(row.get(dec_column))

        if retrieval_mode == "Cone Search":
            if ra is None or dec is None:
                raise RuntimeError(
                    "Cone-search spectrum retrieval requires mapped numeric `coords.ra` and `coords.dec`."
                )

        return _ResolvedTarget(
            dataset_id=dataset_id,
            row_id=row_id_str,
            row=dict(row),
            ra=ra,
            dec=dec,
            source_id=source_id,
            id_column=id_column,
            ra_column=ra_column,
            dec_column=dec_column,
            target_id_column=target_column,
            retrieval_mode=retrieval_mode,
        )

    def _target_id_column(self, dataset_id: str) -> Optional[str]:
        selected = self.target_id_column.value
        if selected and selected != "Auto":
            return selected

        semantic_role = {
            "DESI": "spectra.desi_target_id",
            "SDSS": "spectra.sdss_target_id",
            "EuclidSpec": "spectra.euclid_source_id",
        }.get(self.source)

        if semantic_role:
            mapped = self._mapping(dataset_id, semantic_role)
            if mapped:
                return mapped

        return self._guess_column(dataset_id, self._target_column_candidates())

    def _target_id_from_row(
        self,
        dataset_id: str,
        row: Dict[str, Any],
        target_column: Optional[str],
    ) -> Optional[Any]:
        if target_column and target_column in row:
            value = row.get(target_column)
            if value is not None and str(value).strip():
                return value

        for candidate in self._target_column_candidates():
            if candidate in row:
                value = row.get(candidate)
                if value is not None and str(value).strip():
                    return value

        return None

    def _row_from_legacy_data(
        self,
        *,
        id_column: Optional[str],
        row_id: Any,
        allow_first_row: bool = False,
    ) -> Optional[Dict[str, Any]]:
        data = self.data
        if data is None:
            return None

        try:
            if hasattr(data, "iloc") and len(data) > 0:
                if row_id is not None and id_column:
                    if id_column == "Use Index":
                        matches = data.loc[data.index.astype(str) == str(row_id)]
                    elif id_column in data.columns:
                        matches = data[data[id_column].astype(str) == str(row_id)]
                    else:
                        return None

                    if len(matches) > 0:
                        return matches.iloc[0].to_dict()
                    return None

                if allow_first_row:
                    return data.iloc[0].to_dict()

            if isinstance(data, dict):
                if row_id is not None and id_column and id_column in data:
                    if str(data[id_column]) != str(row_id):
                        return None
                return dict(data)
        except Exception:
            return None

        return None

    def _fetch_row(
        self,
        dataset_id: str,
        *,
        id_column: Optional[str],
        row_id: Any,
        row_pos: Any,
        required_columns: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return None

        columns: List[str] = []
        for col in required_columns or []:
            if col and col != "Use Index" and col not in columns:
                columns.append(col)

        source = None
        for method_name in ("get_source", "source"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    source = method(dataset_id)
                    break
                except Exception:
                    pass

        if source is not None:
            row = self._fetch_row_from_source(
                source,
                id_column=id_column,
                row_id=row_id,
                row_pos=row_pos,
                columns=columns,
            )
            if row is not None:
                return row

        try:
            df = datasets.get_df(dataset_id)

            if row_id is not None and id_column:
                if id_column == "Use Index":
                    matches = df.loc[df.index.astype(str) == str(row_id)]
                    if len(matches) > 0:
                        return matches.iloc[0].to_dict()

                elif id_column in df.columns:
                    matches = df[df[id_column].astype(str) == str(row_id)]
                    if len(matches) > 0:
                        return matches.iloc[0].to_dict()

            if row_pos is not None:
                return df.iloc[int(row_pos)].to_dict()

            return None
        except Exception:
            traceback.print_exc()
            return None

    @staticmethod
    def _normalise_rows(result: Any) -> Optional[Dict[str, Any]]:
        if result is None:
            return None

        try:
            if hasattr(result, "to_pandas"):
                result = result.to_pandas()
        except Exception:
            pass

        try:
            if hasattr(result, "iloc") and len(result) > 0:
                return result.iloc[0].to_dict()
        except Exception:
            pass

        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return first

        if isinstance(result, dict):
            return result

        return None

    def _fetch_row_from_source(
        self,
        source: Any,
        *,
        id_column: Optional[str],
        row_id: Any,
        row_pos: Any,
        columns: List[str],
    ) -> Optional[Dict[str, Any]]:
        if row_id is not None and id_column:
            method = getattr(source, "get_row_by_id", None)
            if callable(method):
                try:
                    row = self._normalise_rows(
                        method(row_id, id_column=id_column, columns=columns or None)
                    )
                    if row is not None:
                        return row
                except Exception:
                    pass

            method = getattr(source, "get_rows_by_ids", None)
            if callable(method):
                try:
                    row = self._normalise_rows(
                        method([row_id], id_column=id_column, columns=columns or None)
                    )
                    if row is not None:
                        return row
                except Exception:
                    pass

            attempts = [
                ("get_rows_by_id", {"row_ids": [row_id], "id_column": id_column, "columns": columns}),
                ("read_rows_by_id", {"row_ids": [row_id], "id_column": id_column, "columns": columns}),
                ("rows_by_id", {"row_ids": [row_id], "id_column": id_column, "columns": columns}),
                ("get_rows", {"row_ids": [row_id], "id_column": id_column, "columns": columns}),
            ]

            for name, kwargs in attempts:
                method = getattr(source, name, None)
                if callable(method):
                    for call_kwargs in (
                        kwargs,
                        {k: v for k, v in kwargs.items() if k != "columns"},
                    ):
                        try:
                            row = self._normalise_rows(method(**call_kwargs))
                            if row is not None:
                                return row
                        except TypeError:
                            continue
                        except Exception:
                            continue

        if row_pos is not None:
            method = getattr(source, "get_row_by_position", None)
            if callable(method):
                try:
                    row = self._normalise_rows(method(int(row_pos), columns=columns or None))
                    if row is not None:
                        return row
                except Exception:
                    pass

            attempts = [
                ("get_rows", {"row_positions": [row_pos], "columns": columns}),
                ("read_rows", {"row_positions": [row_pos], "columns": columns}),
                ("take", {"indices": [row_pos], "columns": columns}),
            ]

            for name, kwargs in attempts:
                method = getattr(source, name, None)
                if callable(method):
                    for call_kwargs in (
                        kwargs,
                        {k: v for k, v in kwargs.items() if k != "columns"},
                    ):
                        try:
                            row = self._normalise_rows(method(**call_kwargs))
                            if row is not None:
                                return row
                        except TypeError:
                            continue
                        except Exception:
                            continue

        return None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _runtime(self) -> SpectraRuntime:
        services = getattr(self.context, "services", None)
        if services is not None:
            try:
                if services.has(RUNTIME_SERVICE_KEY):
                    return services.get(RUNTIME_SERVICE_KEY)
            except Exception:
                pass

        return SpectraRuntime(context=self.context)

    def _schedule_initial_load(self) -> None:
        if self._initial_load_started:
            return

        self._initial_load_started = True

        def _run() -> None:
            self._refresh_column_options()
            self._update_target_status()
            if self.auto_reload.value:
                self.load_spectra(reason="initial")

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(_run)
            else:
                _run()
        except Exception:
            _run()

    def _datasets_for_source(self) -> List[str]:
        if self.source == "DESI":
            return DESI_DATASETS
        if self.source == "SDSS":
            return SDSS_DATASETS
        return ["Euclid-Q1"]

    def _redshift_from_row(self, target: _ResolvedTarget) -> Optional[float]:
        column = self.redshift_column_selector.value
        if not column or column == "None":
            return _safe_float(self.redshift_input.value)

        value = _safe_float(target.row.get(column))
        if value is not None:
            return value

        return _safe_float(self.redshift_input.value)

    def load_spectra(self, *, reason: str = "manual") -> None:
        if self._disposed:
            return

        try:
            target = self._resolve_target()
            self._current_target = target
        except Exception as exc:
            self.status.object = f"**Spectrum unavailable:** {exc}"
            self.figure[:] = [self._empty_message("Spectrum unavailable.")]
            return

        self._cancel_job()
        self._spectra_result = None
        self.figure[:] = [self._empty_message("Loading spectrum…")]
        self.status.object = "Loading spectrum…"
        self.target_status.object = self._target_html(target)

        self._publish(
            "astro.spectra.running",
            {
                "source": self.source,
                "running": True,
                "panel_id": self.panel_id,
                "dataset_id": target.dataset_id,
                "selected_id": target.row_id,
                "reason": reason,
            },
        )

        runtime = self._runtime()
        redshift = self._redshift_from_row(target)

        def _worker(cancel_token: Any = None) -> SpectraResult:
            return runtime.fetch_spectra(
                source=self.source,
                ra=target.ra,
                dec=target.dec,
                source_id=target.source_id,
                max_separation_arcsec=float(self.max_separation_input.value),
                datasets=self._datasets_for_source(),
                smooth_kernel=self.smoothing_function_input.value,
                smooth_window=int(self.smoothing_window_input.value),
                redshift_override=redshift,
                query_euclid_redshift=False,
                cancel_token=cancel_token,
            )

        def _done(result: SpectraResult) -> None:
            self._on_spectra_loaded(result, target=target, reason=reason)

        def _error(exc: BaseException) -> None:
            self._on_spectra_error(exc, target=target, reason=reason)

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                _done(_worker(cancel_token=None))
            except BaseException as exc:
                _error(exc)
            return

        key = (
            f"{self.panel_id}:{target.dataset_id}:{target.row_id}:"
            f"{target.retrieval_mode}:{target.source_id}:{target.ra}:{target.dec}:"
            f"{self.max_separation_input.value}:{self.source}"
        )

        self._job_handle = jobs.submit(
            _worker,
            title=f"Fetch {self.source_label} spectra",
            key=key,
            on_done=_done,
            on_error=_error,
        )

    def _cancel_job(self) -> None:
        handle = self._job_handle
        self._job_handle = None
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass

    def _on_spectra_loaded(
        self,
        result: SpectraResult,
        *,
        target: _ResolvedTarget,
        reason: str,
    ) -> None:
        if self._disposed:
            return

        if not self._target_matches_current_focus(target):
            return

        self._job_handle = None
        self._spectra_result = result
        self.status.object = ""
        self._render_existing_result()

        spectrum_artifact_id, coords_artifact_id = self._publish_spectrum_artifacts(
            result,
            target=target,
        )

        self._publish(
            "astro.spectra.updated",
            {
                "source": self.source,
                "artifact_id": spectrum_artifact_id,
                "coords_artifact_id": coords_artifact_id,
                "panel_id": self.panel_id,
                "dataset_id": target.dataset_id,
                "selected_id": target.row_id,
                "ra": target.ra,
                "dec": target.dec,
                "reason": reason,
                "available_spectra": result.available_spectra,
            },
        )

        self._publish(
            "astro.spectra.running",
            {
                "source": self.source,
                "running": False,
                "panel_id": self.panel_id,
                "dataset_id": target.dataset_id,
                "selected_id": target.row_id,
                "reason": reason,
            },
        )

    def _on_spectra_error(
        self,
        exc: BaseException,
        *,
        target: _ResolvedTarget,
        reason: str,
    ) -> None:
        if self._disposed:
            return

        if not self._target_matches_current_focus(target):
            return

        self._job_handle = None
        self.status.object = f"**Spectrum unavailable:** {exc}"
        self.figure[:] = [self._empty_message("Spectrum unavailable.")]

        self._publish(
            "astro.spectra.running",
            {
                "source": self.source,
                "running": False,
                "panel_id": self.panel_id,
                "dataset_id": target.dataset_id,
                "selected_id": target.row_id,
                "reason": reason,
                "error": str(exc),
            },
        )

    # ------------------------------------------------------------------
    # Rendering / artifacts
    # ------------------------------------------------------------------

    def _apply_redshift_controls(self) -> None:
        result = self._spectra_result
        target = self._current_target
        if result is None or target is None:
            return

        redshift = self._redshift_from_row(target)
        if redshift is None:
            return

        obj = result.spectra_object
        if obj is None or not hasattr(obj, "_update_info_spectra"):
            return

        try:
            obj._update_info_spectra("redshift", redshift)
            obj._update_info_spectra("spectype", "galaxy" if redshift > 0 else "star")
        except Exception:
            pass

    def _render_existing_result(self) -> None:
        result = self._spectra_result
        if result is None:
            return

        try:
            self._apply_redshift_controls()

            plot_lines = "class" if self.plot_lines_checkbox.value else False
            plot = result.plot_hv(
                plot_model=bool(self.plot_model_checkbox.value),
                plot_lines=plot_lines,
                responsive=True,
            )

            self.figure[:] = [
                pn.pane.HoloViews(
                    plot,
                    sizing_mode="stretch_both",
                    min_height=0,
                    margin=(0, 0, 0, 0),
                )
            ]
            self.status.object = ""
        except Exception as exc:
            self.status.object = f"**Could not render spectrum:** {exc}"
            self.figure[:] = [self._empty_message("Could not render spectrum.")]

    def _publish_spectrum_artifacts(
        self,
        result: SpectraResult,
        *,
        target: _ResolvedTarget,
    ) -> Tuple[Optional[str], Optional[str]]:
        artifacts = getattr(self.context, "artifacts", None)
        if artifacts is None:
            return None, None

        spectrum_artifact_id = None
        coords_artifact_id = None

        spectrum_payload = result.artifact_payload()
        coords = result.coordinates_payload()
        coordinate_count = min(len(coords.get("ra", [])), len(coords.get("dec", [])))

        try:
            spectrum_artifact_id = artifacts.put(
                "astro.spectra",
                spectrum_payload,
                dataset_id=target.dataset_id,
                row_ids=[target.row_id] if target.row_id is not None else None,
                params={
                    "source": self.source,
                    "selected_id": target.row_id,
                    "retrieval_mode": target.retrieval_mode,
                    "target_id_column": target.target_id_column,
                    "source_id": target.source_id,
                },
                persist=False,
            )
        except Exception:
            traceback.print_exc()

        if coordinate_count:
            try:
                coords_artifact_id = artifacts.put(
                    "astro.coords",
                    coords,
                    dataset_id=target.dataset_id,
                    row_ids=[target.row_id] if target.row_id is not None else None,
                    params={
                        "source": self.source,
                        "selected_id": target.row_id,
                        "coordinate_count": coordinate_count,
                    },
                    persist=False,
                )

                self._publish(
                    "astro.coords.updated",
                    {
                        "source": self.source,
                        "artifact_id": coords_artifact_id,
                        "spectrum_artifact_id": spectrum_artifact_id,
                        "dataset_id": target.dataset_id,
                        "selected_id": target.row_id,
                        "coordinate_count": coordinate_count,
                        "colors": coords.get("colors", []),
                        "colours": coords.get("colours", []),
                        "labels": coords.get("labels", []),
                        "points": coords.get("points", []),
                    },
                )
            except Exception:
                traceback.print_exc()

        return spectrum_artifact_id, coords_artifact_id

    # ------------------------------------------------------------------
    # Actions/callbacks
    # ------------------------------------------------------------------

    def _retrieve_mode_changed(self, event: Any) -> None:
        is_target_mode = event.new == "Use TargetId"
        self.max_separation_input.disabled = is_target_mode
        self.target_id_column.disabled = not is_target_mode

        if self.auto_reload.value:
            self.load_spectra(reason="spectrum.retrieve_mode.changed")

    def _query_euclid_redshift(self, _event: Any = None) -> None:
        if self.source != "EuclidSpec" or self._spectra_result is None:
            return

        obj = self._spectra_result.spectra_object
        if obj is None:
            return

        self.status.object = "Querying Euclid redshift table…"

        def _worker(cancel_token: Any = None) -> SpectraResult:
            if hasattr(obj, "query_specz_table"):
                obj.query_specz_table(verbose=True)
            if hasattr(obj, "update_info_from_query"):
                obj.update_info_from_query()
            try:
                obj.get_smoothed_spectra(
                    kernel=self.smoothing_function_input.value,
                    window=int(self.smoothing_window_input.value),
                )
            except Exception:
                pass
            return self._spectra_result

        def _done(result: SpectraResult) -> None:
            self.status.object = ""
            self._render_existing_result()
            if self._current_target is not None:
                self._publish_spectrum_artifacts(result, target=self._current_target)

        def _error(exc: BaseException) -> None:
            self.status.object = f"**Could not query Euclid redshift:** {exc}"

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                _done(_worker(cancel_token=None))
            except BaseException as exc:
                _error(exc)
            return

        self._job_handle = jobs.submit(
            _worker,
            title="Query Euclid spectrum redshift",
            key=f"{self.panel_id}:euclid-redshift:{self._current_row_id()}",
            on_done=_done,
            on_error=_error,
        )

    def _target_html(self, target: _ResolvedTarget) -> str:
        pieces = [
            f"<b>Source:</b> <code>{self.source_label}</code>",
            f"<b>Dataset:</b> <code>{target.dataset_id}</code>",
            f"<b>Record:</b> <code>{target.row_id or 'unmapped'}</code>",
            f"<b>Mode:</b> <code>{target.retrieval_mode}</code>",
        ]

        if target.source_id is not None:
            pieces.append(f"<b>Target:</b> <code>{target.source_id}</code>")

        if target.ra is not None and target.dec is not None:
            pieces.append(f"<b>RA/Dec:</b> <code>{target.ra:.6f}, {target.dec:.6f}</code>")

        return "<div>" + " &nbsp; ".join(pieces) + "</div>"

    def _update_target_status(self) -> None:
        try:
            target = self._resolve_target()
            self._current_target = target
            self.target_status.object = self._target_html(target)
        except Exception as exc:
            self.target_status.object = f"<div style='color:#8a5a00'>⚠️ {exc}</div>"

    # ------------------------------------------------------------------
    # Artifact viewer helpers
    # ------------------------------------------------------------------

def _fallback_artifact_colour(index: int) -> str:
    colours = [
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
    return colours[int(index) % len(colours)]

def spectra_payload_to_hv(payload: Dict[str, Any]) -> Any:
    spectra = payload.get("spectra", []) if isinstance(payload, dict) else []
    if not spectra:
        return hv.Curve([]).opts(title="No spectra available")

    overlays = []

    for idx, spectrum in enumerate(spectra):
        wavelength = np.asarray(spectrum.get("wavelength", []), dtype=float)
        flux = np.asarray(spectrum.get("flux", []), dtype=float)
        smoothed = np.asarray(spectrum.get("smoothed_flux", []), dtype=float)
        model = np.asarray(spectrum.get("model", []), dtype=float)

        if wavelength.size == 0 or flux.size == 0:
            continue

        colour = (
            spectrum.get("plot_color")
            or spectrum.get("plot_colour")
            or _fallback_artifact_colour(idx)
        )

        label = str(
            spectrum.get("data_release")
            or spectrum.get("sourceid")
            or f"Spectrum {idx + 1}"
        )

        overlays.append(
            hv.Curve((wavelength, flux), label=f"{label} flux").opts(
                color="grey",
                line_width=0.4,
                alpha=0.65,
            )
        )

        if smoothed.size == wavelength.size:
            overlays.append(
                hv.Curve((wavelength, smoothed), label=f"{label} smoothed").opts(
                    color=colour,
                    line_width=1.2,
                )
            )

        if model.size == wavelength.size and np.isfinite(model).any():
            overlays.append(
                hv.Curve((wavelength, model), label=f"{label} model").opts(
                    color=colour,
                    line_width=1.1,
                    line_dash="dashed",
                )
            )

    if not overlays:
        return hv.Curve([]).opts(title="No displayable spectra available")

    wavelength_arrays = [
        np.asarray(spec.get("wavelength", []), dtype=float)
        for spec in spectra
        if len(spec.get("wavelength", []))
    ]

    xmin = min(np.nanmin(arr) for arr in wavelength_arrays)
    xmax = max(np.nanmax(arr) for arr in wavelength_arrays)

    return hv.Overlay(overlays).opts(
        responsive=True,
        xlabel="Observed wavelength [Å]",
        ylabel="Flux",
        logx=True,
        xlim=(xmin, xmax),
        show_legend=True,
        legend_position="bottom_left",
        active_tools=[],
    )


# ----------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------


def make_spectra_panel_factory(source: str):
    def _factory(
        context: Any,
        data: Any = None,
        state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[pn.viewable.Viewable, SpectraPanel]:
        panel = SpectraPanel(
            context=context,
            source=source,
            data=data,
            state=state,
            **kwargs,
        )
        return panel.view(), panel

    return _factory


def create_spectra_artifact_viewer(
    context: Any,
    artifact_id: str,
    **_: Any,
) -> Tuple[pn.viewable.Viewable, Any]:
    artifacts = getattr(context, "artifacts", None)
    if artifacts is None:
        view = pn.pane.Markdown("Artifact store is not available.")
        return view, None

    try:
        payload = artifacts.get(artifact_id)
    except Exception as exc:
        view = pn.pane.Markdown(f"Could not load spectrum artifact `{artifact_id}`: {exc}")
        return view, None

    try:
        plot = spectra_payload_to_hv(payload)
        pane = pn.pane.HoloViews(
            plot,
            sizing_mode="stretch_both",
            min_height=420,
        )

        meta = pn.pane.Markdown(
            f"**Spectrum artifact:** `{artifact_id}`  \n"
            f"**Source:** `{payload.get('source')}`  \n"
            f"**Available spectra:** `{payload.get('available_spectra')}`  \n"
            f"**Retrieval mode:** `{payload.get('retrieval_mode')}`",
            sizing_mode="stretch_width",
        )

        return pn.Column(meta, pane, sizing_mode="stretch_both"), None
    except Exception as exc:
        return pn.pane.Markdown(f"Could not render spectrum artifact `{artifact_id}`: {exc}"), None