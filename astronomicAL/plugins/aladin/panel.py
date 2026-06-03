from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import html
import json
import traceback
import uuid

import numpy as np
import panel as pn


PLUGIN_ID = "astro.aladin"
SETTINGS_HEIGHT = 72


SURVEY_GROUPS: Dict[str, Dict[str, str]] = {
    "X-rays": {
        "eROSITA (rate, color)": "erosita/dr1/rate/rgb",
        "eROSITA 0.2-0.6 keV (count)": "erosita/dr1/count/021",
        "eROSITA 0.6-2.3 keV (count)": "erosita/dr1/count/022",
        "eROSITA 2.3-5 keV (count)": "erosita/dr1/count/023",
        "Swift XRT (exposure)": "nasa.heasarc/P/Swift/XRT/exp",
        "XMM (color)": "xcatdb/P/XMM/PN/color",
        "XMM (0.5-1 keV)": "xcatdb/P/XMM/PN/eb2",
        "XMM (1-2 keV)": "xcatdb/P/XMM/PN/eb3",
        "XMM (2-4.5 keV)": "xcatdb/P/XMM/PN/eb4",
    },
    "Optical/UV": {
        "SDSS (color)": "CDS/P/SDSS9/color",
        "DSS2 (color)": "P/DSS2/color",
        "Pan-STARRS (color)": "P/PanSTARRS/DR1/color-z-zg-g",
        "GALEX (color)": "P/GALEXGR6/AIS/color",
        "DESI Legacy Survey (color)": "CDS/P/DESI-Legacy-Surveys/DR10/color",
        "DES (color)": "CDS/P/DES-DR2/ColorIRG",
    },
    "IR": {
        "AllWISE (color)": "P/allWISE/color",
        "2MASS (color)": "P/2MASS/color",
        "Herschel SPIRE (color)": "ESAVO/P/HERSCHEL/SPIRE-color",
        "Spitzer IRAC (color)": "CDS/P/SPITZER/color",
        "Euclid Q1 (color)": "CDS/P/Euclid/Q1/color",
    },
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


def make_srcdoc_aladin_lite(
    *,
    survey_id: str,
    ra: float,
    dec: float,
    fov: float = 0.08,
    show_reticle: bool = True,
    show_simbad: bool = False,
    show_ned: bool = False,
    show_grid: bool = False,
) -> str:
    """Return a self-contained Aladin Lite HTML document.

    The legacy implementation used an iframe srcdoc. This keeps the same model
    while avoiding any dependency on the old custom-plot shell.
    """

    survey_json = json.dumps(str(survey_id))
    target_json = json.dumps(f"{float(ra)} {float(dec)}")
    ra_json = json.dumps(float(ra))
    dec_json = json.dumps(float(dec))
    fov_json = json.dumps(float(fov))
    show_reticle_json = json.dumps(bool(show_reticle))
    show_simbad_json = json.dumps(bool(show_simbad))
    show_ned_json = json.dumps(bool(show_ned))
    show_grid_json = json.dumps(bool(show_grid))

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta
    name="viewport"
    content="width=device-width, initial-scale=1, maximum-scale=1"
  />
  <link
    rel="stylesheet"
    href="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.min.css"
  />
  <script
    type="text/javascript"
    src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"
    charset="utf-8"
  ></script>
  <style>
    html, body {{
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: #111;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #aladin-lite-div {{
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
    }}
    #error {{
      display: none;
      color: #f7f7f7;
      background: #5b1a1a;
      padding: 10px;
      font-size: 13px;
      line-height: 1.35;
    }}
  </style>
</head>
<body>
  <div id="error"></div>
  <div id="aladin-lite-div"></div>
  <script>
    const surveyId = {survey_json};
    const target = {target_json};
    const ra = {ra_json};
    const dec = {dec_json};
    const fov = {fov_json};
    const showReticle = {show_reticle_json};
    const showSimbad = {show_simbad_json};
    const showNed = {show_ned_json};
    const showGrid = {show_grid_json};

    function showError(message) {{
      const error = document.getElementById("error");
      error.style.display = "block";
      error.textContent = message;
    }}

    function addCentreMarker(aladin) {{
      try {{
        const overlay = A.graphicOverlay({{name: "Selected source"}});
        aladin.addOverlay(overlay);
        overlay.add(A.marker(ra, dec, {{
          popupTitle: "Selected source",
          popupDesc: "RA: " + ra.toFixed(6) + ", Dec: " + dec.toFixed(6)
        }}));
      }} catch (err) {{
        console.warn("Could not add centre marker", err);
      }}
    }}

    function addCatalogueLayers(aladin) {{
      if (showSimbad) {{
        try {{
          aladin.addCatalog(A.catalogFromSimbad(target, 0.05, {{
            name: "Simbad",
            sourceSize: 12
          }}));
        }} catch (err) {{
          console.warn("Could not add Simbad layer", err);
        }}
      }}

      if (showNed) {{
        try {{
          aladin.addCatalog(A.catalogFromNED(target, 0.05, {{
            name: "NED",
            sourceSize: 12
          }}));
        }} catch (err) {{
          console.warn("Could not add NED layer", err);
        }}
      }}
    }}

    try {{
      A.init.then(() => {{
        const aladin = A.aladin("#aladin-lite-div", {{
          survey: surveyId,
          target: target,
          fov: fov,
          cooFrame: "ICRSd",
          showReticle: showReticle,
          showCooGrid: showGrid,
          showSimbadPointerControl: true,
          showShareControl: false,
          showFullscreenControl: false,
          showFrame: true,
          showCooGridControl: true,
          showProjectionControl: true,
          showLayerBox: true
        }});

        addCentreMarker(aladin);
        addCatalogueLayers(aladin);
      }}).catch((err) => {{
        showError("Could not initialise Aladin Lite: " + err);
      }});
    }} catch (err) {{
      showError("Could not initialise Aladin Lite: " + err);
    }}
  </script>
</body>
</html>"""


def make_iframe_html(
    *,
    survey_id: str,
    ra: float,
    dec: float,
    fov: float,
    show_reticle: bool,
    show_simbad: bool,
    show_ned: bool,
    show_grid: bool,
) -> str:
    srcdoc = make_srcdoc_aladin_lite(
        survey_id=survey_id,
        ra=ra,
        dec=dec,
        fov=fov,
        show_reticle=show_reticle,
        show_simbad=show_simbad,
        show_ned=show_ned,
        show_grid=show_grid,
    )
    srcdoc_escaped = html.escape(srcdoc, quote=True)

    return f"""
<iframe
  srcdoc="{srcdoc_escaped}"
  style="
    width: 100%;
    height: 100%;
    border: 0;
    display: block;
    margin: 0;
    padding: 0;
    background: #111;
  "
  referrerpolicy="no-referrer-when-downgrade"
  allow="fullscreen"
></iframe>
"""


@dataclass
class _ResolvedTarget:
    dataset_id: str
    row_id: Optional[str]
    row: Dict[str, Any]
    ra: float
    dec: float
    ra_column: str
    dec_column: str
    id_column: Optional[str]


class AladinPanel:
    """Platform-native Aladin Lite panel."""

    def __init__(
        self,
        *,
        context: Any,
        data: Any = None,
        state: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        self.context = context
        self.data = data
        self.panel_id = f"{PLUGIN_ID}.{uuid.uuid4().hex}"
        self._subscriptions: List[Any] = []
        self._disposed = False
        self._initial_load_started = False
        self._settings_built = False
        self.settings_visible = False
        self._current_target: Optional[_ResolvedTarget] = None

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
            "survey": self.survey_selector.value,
            "fov_deg": self.fov_input.value,
            "show_reticle": self.show_reticle.value,
            "show_simbad": self.show_simbad.value,
            "show_ned": self.show_ned.value,
            "show_grid": self.show_grid.value,
            "auto_reload": self.auto_reload.value,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        self.settings_visible = bool(state.get("settings_visible", False))

        mapping = {
            "survey": self.survey_selector,
            "fov_deg": self.fov_input,
            "show_reticle": self.show_reticle,
            "show_simbad": self.show_simbad,
            "show_ned": self.show_ned,
            "show_grid": self.show_grid,
            "auto_reload": self.auto_reload,
        }

        for key, widget in mapping.items():
            if key not in state:
                continue
            try:
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

        self.figure = pn.pane.HTML(
            "",
            sizing_mode="stretch_both",
            min_height=320,
            margin=(0, 6, 6, 6),
            styles={
                "width": "100%",
                "height": "100%",
                "min-height": "0",
                "overflow": "hidden",
                "box-sizing": "border-box",
            },
        )

        self.title_pane = pn.pane.HTML(
            "<div style='font-size:14px;font-weight:700;padding-top:10px'>Aladin Lite</div>",
            width=110,
            height=44,
            sizing_mode="fixed",
            margin=(0, 2, 0, 0),
        )

        self.survey_selector = pn.widgets.Select(
            name="Survey",
            value="P/DSS2/color",
            groups=SURVEY_GROUPS,
            sizing_mode="stretch_width",
            height=44,
            margin=(0, 2, 0, 0),
        )

        self.fov_input = pn.widgets.FloatInput(
            name="FoV [deg]",
            value=0.08,
            start=0.001,
            end=180,
            step=0.01,
            width=105,
            height=44,
            sizing_mode="fixed",
            margin=(0, 2, 0, 0),
        )

        self.reload_button = pn.widgets.Button(
            name="Reload",
            button_type="primary",
            width=76,
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

        self.show_reticle = _style_widget(
            pn.widgets.Checkbox(name="Reticle", value=True),
            width=90,
            height=28,
            margin=(10, 8, 0, 6),
        )

        self.show_grid = _style_widget(
            pn.widgets.Checkbox(name="Grid", value=False),
            width=80,
            height=28,
            margin=(10, 8, 0, 6),
        )

        self.show_simbad = _style_widget(
            pn.widgets.Checkbox(name="Simbad layer", value=False),
            width=115,
            height=28,
            margin=(10, 8, 0, 6),
        )

        self.show_ned = _style_widget(
            pn.widgets.Checkbox(name="NED layer", value=False),
            width=100,
            height=28,
            margin=(10, 8, 0, 6),
        )

        self.auto_reload = _style_widget(
            pn.widgets.Checkbox(name="Auto reload", value=True),
            width=110,
            height=28,
            margin=(10, 8, 0, 6),
        )

        self.reload_button.on_click(lambda _event: self.reload(reason="button.reload"))
        self.settings_button.on_click(self._toggle_settings)

        for widget in [
            self.survey_selector,
            self.fov_input,
            self.show_reticle,
            self.show_grid,
            self.show_simbad,
            self.show_ned,
        ]:
            widget.param.watch(lambda _event: self.reload(reason="setting.changed"), "value")

    def _header(self) -> pn.GridBox:
        return pn.GridBox(
            self.title_pane,
            self.survey_selector,
            self.fov_input,
            self.reload_button,
            self.settings_button,
            ncols=5,
            sizing_mode="stretch_width",
            height=48,
            margin=(0, 6, 0, 6),
            styles={
                "display": "grid",
                "grid-template-columns": "114px minmax(180px, 1fr) 108px 80px 34px",
                "gap": "4px",
                "align-items": "start",
                "box-sizing": "border-box",
            },
        )

    def _settings_controls(self) -> pn.FlexBox:
        return _settings_box(
            _small_label("Display", width=58),
            self.show_reticle,
            self.show_grid,
            self.show_simbad,
            self.show_ned,
            self.auto_reload,
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

    def _subscribe(self, topic: str, callback: Any) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        try:
            sub = events.subscribe(
                topic,
                callback,
                owner_id=self.panel_id,
                owner_label="Aladin Lite",
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
        self._current_target = None
        self._update_target_status()

        if self.auto_reload.value:
            self.reload(reason=str(topic or "selection.focus.changed"))

    def _selection_cleared(self, topic: str, payload: Any) -> None:
        self._current_target = None
        self.target_status.object = ""
        self.status.object = "No focused row selected."
        self.figure.object = self._empty_html("No focused row selected.")

    def _dataset_changed(self, topic: str, payload: Any) -> None:
        self._current_target = None
        self._update_target_status()

        if self.auto_reload.value:
            self.reload(reason=str(topic or "dataset.changed"))

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

        if not ra_column or not dec_column:
            raise RuntimeError(
                "Aladin Lite requires mapped coordinate columns `coords.ra` and `coords.dec`."
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
                required_columns=[ra_column, dec_column],
            )

        if row is None:
            row = self._row_from_legacy_data(
                id_column=id_column,
                row_id=row_id,
                allow_first_row=(row_id is None),
            )

        if row is None:
            raise RuntimeError("No focused row is available for the Aladin Lite panel.")

        if row_id is None and id_column and id_column != "Use Index" and id_column in row:
            row_id = row.get(id_column)

        row_id_str = None if row_id is None else str(row_id)

        try:
            ra = float(row[ra_column])
            dec = float(row[dec_column])
        except Exception as exc:
            raise RuntimeError(
                f"Could not read numeric RA/Dec from columns {ra_column!r}/{dec_column!r}."
            ) from exc

        return _ResolvedTarget(
            dataset_id=dataset_id,
            row_id=row_id_str,
            row=dict(row),
            ra=ra,
            dec=dec,
            ra_column=ra_column,
            dec_column=dec_column,
            id_column=id_column,
        )

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

        for col in [id_column, *(required_columns or [])]:
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
    # Rendering
    # ------------------------------------------------------------------

    def _schedule_initial_load(self) -> None:
        if self._initial_load_started:
            return

        self._initial_load_started = True

        def _run() -> None:
            self._update_target_status()
            if self.auto_reload.value:
                self.reload(reason="initial")

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(_run)
            else:
                _run()
        except Exception:
            _run()

    def _target_html(self, target: _ResolvedTarget) -> str:
        return (
            "<div>"
            f"<b>Dataset:</b> <code>{target.dataset_id}</code> &nbsp; "
            f"<b>Record:</b> <code>{target.row_id or 'unmapped'}</code> &nbsp; "
            f"<b>RA/Dec:</b> <code>{target.ra:.6f}, {target.dec:.6f}</code> &nbsp; "
            f"<b>Survey:</b> <code>{self.survey_selector.value}</code>"
            "</div>"
        )

    def _update_target_status(self) -> None:
        try:
            target = self._resolve_target()
            self._current_target = target
            self.target_status.object = self._target_html(target)
        except Exception as exc:
            self.target_status.object = f"<div style='color:#8a5a00'>⚠️ {exc}</div>"

    @staticmethod
    def _empty_html(message: str) -> str:
        message = html.escape(str(message))
        return f"""
<div style="
  width: 100%;
  height: 100%;
  min-height: 280px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #666;
  background: #f7f7f7;
  font-size: 14px;
  box-sizing: border-box;
">
  {message}
</div>
"""

    def reload(self, *, reason: str = "manual") -> None:
        if self._disposed:
            return

        try:
            target = self._resolve_target()
            self._current_target = target
        except Exception as exc:
            self.status.object = f"**Aladin Lite unavailable:** {exc}"
            self.figure.object = self._empty_html("Aladin Lite unavailable.")
            return

        fov = _safe_float(self.fov_input.value)
        if fov is None or fov <= 0:
            fov = 0.08
            self.fov_input.value = fov

        self.status.object = ""
        self.target_status.object = self._target_html(target)

        self.figure.object = make_iframe_html(
            survey_id=self.survey_selector.value,
            ra=target.ra,
            dec=target.dec,
            fov=fov,
            show_reticle=bool(self.show_reticle.value),
            show_simbad=bool(self.show_simbad.value),
            show_ned=bool(self.show_ned.value),
            show_grid=bool(self.show_grid.value),
        )

        self._publish(
            "astro.aladin.updated",
            {
                "panel_id": self.panel_id,
                "dataset_id": target.dataset_id,
                "selected_id": target.row_id,
                "ra": target.ra,
                "dec": target.dec,
                "survey": self.survey_selector.value,
                "fov_deg": fov,
                "reason": reason,
            },
        )


# ----------------------------------------------------------------------
# Plugin factory
# ----------------------------------------------------------------------


def create_aladin_panel(
    context: Any,
    data: Any = None,
    state: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[pn.viewable.Viewable, AladinPanel]:
    panel = AladinPanel(
        context=context,
        data=data,
        state=state,
        **kwargs,
    )
    return panel.view(), panel