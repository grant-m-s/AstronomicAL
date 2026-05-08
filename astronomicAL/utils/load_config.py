from __future__ import annotations

import panel as pn
import json
import os
import time
import uuid


from typing import Optional, Tuple

from datetime import datetime
import numpy as np
import pandas as pd
from astropy.table import Table
from astronomicAL.config import get_save_layout_button, get_save_logbook_button, get_save_panel_data_button
from astronomicAL.dashboard.dashboard import Dashboard
from astronomicAL.extensions.dynamic_react_layout import DynamicReactGrid
from astronomicAL.extensions.extension_plots import get_plot_dict
from astronomicAL.extensions.custom_plots import get_customplot_dict
from astronomicAL.extensions.feature_generation import get_oper_dict
from astronomicAL.extensions.models import get_classifiers
from astronomicAL.extensions.query_strategies import get_strategy_dict
from astronomicAL.settings.data_selection import DataSelection
from astronomicAL.platform.mapping_header import MappingAlertController
from astronomicAL.platform.dataset_header import DatasetHeaderController
from astronomicAL.platform.modal_utils import ensure_template_modal_host

import os
import json
import time
from datetime import datetime

import pandas as pd
import panel as pn

def create_header(react, grid, context):
    """
    Build and attach the app header row.

    Parameters
    ----------
    react: pn.template.ReactTemplate (or compatible)
    grid: the dynamic grid instance (react._dynamic_grid)
    context: AppContext (contains workspace + config, etc.)
    """
    if context is None:
        raise ValueError("create_header requires a non-null context")

    # Keep backward compatibility: still rely on config-provided button factories
    config = context.config
    if config is None:
        raise ValueError("context.config is required for current header buttons")

    if not hasattr(react, "_header_box"):
        react._header_box = pn.Row(sizing_mode="stretch_width")
        react.header.append(react._header_box)

    export_fits_file_button = pn.widgets.Button(name="Export Labelled Data to Fits File")

    # Close handling: delegate to WorkspaceManager (single owner of grid mutation)
    def _close_from_js(event):
        tile_id = event.new
        if not tile_id:
            return
        context.workspace.remove_panel(str(tile_id))
        # Make sure the JS signal is cleared to avoid repeated triggers
        try:
            grid.close_key = ""
        except Exception:
            pass

    # # Attach watcher only once (avoid duplicate watchers)
    # if not getattr(grid, "_close_watcher_attached", False):
    #     grid.param.watch(_close_from_js, "close_key")
    #     grid._close_watcher_attached = True

    def export_fits_file_cb(_event):
        list_ids: list[str] = []
        list_labels: list[str] = []

        settings = getattr(config, "settings", {}) or {}

        if settings.get("confirmed"):
            # include labels from classifiers in settings
            classifiers = settings.get("classifiers") or {}
            for label, entry in classifiers.items():
                if isinstance(entry, dict) and ("id" in entry) and ("y" in entry):
                    list_ids.extend(entry["id"])
                    list_labels.extend(entry["y"])

            # include test-set labels (if configured)
            orig_labelled_data = {}
            test_set_file = settings.get("test_set_file")
            if test_set_file:
                # NOTE: existing behavior loads from a fixed path; keep that identical
                if os.path.exists("data/test_set.json"):
                    with open("data/test_set.json", "r", encoding="utf-8") as f:
                        orig_labelled_data = json.load(f)

                for _id, _lab in orig_labelled_data.items():
                    list_ids.append(_id)
                    list_labels.append(_lab)

        if list_ids:
            exported_labels = pd.DataFrame({"id": list_ids, "label": list_labels}, dtype="string")

            from astronomicAL.utils.save_config import save_dataframe_to_fits

            dt_string = datetime.now().strftime("%Y%m%d_%H:%M:%S")
            path = f"data/labelled_data_{dt_string}.fits"
            save_dataframe_to_fits(exported_labels, path)

            export_fits_file_button.disabled = True
            export_fits_file_button.name = f"{len(list_ids)} labelled sources saved to '{path}'"
            time.sleep(3)
            export_fits_file_button.name = "Export Labelled Data to Fits File"
            export_fits_file_button.disabled = False
        else:
            export_fits_file_button.disabled = True
            export_fits_file_button.name = "No Labelled Data Found"
            time.sleep(3)
            export_fits_file_button.name = "Export Labelled Data to Fits File"
            export_fits_file_button.disabled = False

    export_fits_file_button.on_click(export_fits_file_cb)

    add_menu_btn = pn.widgets.Button(name="+", button_type="default", width=38, height=34)
    add_menu_btn.styles = {
        "font-size": "26px",
        "font-weight": "700",
        "line-height": "1",
        "padding": "0",
    }
    add_menu_btn.css_classes = ["al-add-menu-btn"]
    add_menu_btn.description = "Add Panel"

    def _on_add_menu(_):
        try:
            add_menu_panel(grid, context=context)
        except Exception as e:
            import traceback
            print("[add_menu_panel] ERROR:", e)
            traceback.print_exc()

    add_menu_btn.on_click(_on_add_menu)

    confirmed = bool(getattr(config, "settings", {}).get("confirmed", False))

    react.config.raw_css.append("""
    #pn-Modal {
    background: transparent !important;
    }

    #pn-Modal .pn-modal-content {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
    width: auto !important;
    max-width: none !important;
    overflow: visible !important;
    display: flex !important;
    justify-content: center !important;
    align-items: flex-start !important;
    }


    #pn-Modal .pn-modalclose {
    display: none !important;
    }
    """)

    ensure_template_modal_host(react)
    
    dataset_header = DatasetHeaderController(context=context, template=react)
    mapping_alert = MappingAlertController(context=context, template=react)

    # Keep references on the template so the controllers are not garbage-collected.
    react._dataset_header = dataset_header
    react._mapping_alert = mapping_alert

    header_row = pn.Row(
        get_save_layout_button(confirmed, True, context=context),
        dataset_header.view,
        # export_fits_file_button,
        # get_save_panel_data_button(confirmed),
        # get_save_logbook_button(confirmed),
        mapping_alert.view,
        add_menu_btn,
        sizing_mode="stretch_width",
    )

    # IMPORTANT: replace contents, don’t append
    react._header_box[:] = [header_row]
    return react


def bind_controller(view, controller):
    """
    Attach controller to a Panel view, and expose dispose on the view so
    WorkspaceManager can dispose it even if controller isn't passed explicitly.
    """
    if view is None or controller is None:
        return view
    try:
        setattr(view, "_al_controller", controller)
        # Provide a dispose method on the view that forwards to controller.dispose()
        if hasattr(controller, "dispose"):
            setattr(view, "dispose", controller.dispose)
    except Exception:
        pass
    return view

import uuid

def add_menu_panel(grid, context):
    grid = context.workspace.grid  # override any stale reference
    
    def _bp_geom(bp: str):
        if bp == "lg":
            return 4, 4   # w,h
        if bp == "md":
            return 6, 4
        return 12, 4

    def _overlaps(a, b) -> bool:
        return not (
            a["x"] + a["w"] <= b["x"] or
            b["x"] + b["w"] <= a["x"] or
            a["y"] + a["h"] <= b["y"] or
            b["y"] + b["h"] <= a["y"]
        )

    def find_first_fit(layout_items, cols: int, w: int, h: int) -> tuple[int, int]:
        """
        Find first free (x,y) for a w×h tile, scanning row-major:
        y=0.., x=0..cols-w
        """
        items = [
            {"x": int(it.get("x", 0)), "y": int(it.get("y", 0)),
             "w": int(it.get("w", 1)), "h": int(it.get("h", 1))}
            for it in (layout_items or [])
            if it is not None
        ]

        max_y = 0
        for it in items:
            max_y = max(max_y, it["y"] + it["h"])

        # search existing rows first, then a little beyond current max
        for y in range(0, max_y + 100):  # 100 rows headroom; cheap scan
            for x in range(0, cols - w + 1):
                cand = {"x": x, "y": y, "w": w, "h": h}
                if not any(_overlaps(cand, it) for it in items):
                    return x, y

        # fallback: append at bottom-left
        return 0, max_y

    if context is None or context.workspace is None:
        raise ValueError("add_menu_panel requires context with workspace")

    config = context.config
    if config is None:
        raise ValueError("add_menu_panel requires context.config (for Dashboard src/config)")

    # Ensure counter starts above any existing numeric ids
    settings = context.config.settings
    if settings is None:
        context.config.settings = {}
        settings = context.config.settings

    # compute current max numeric id in the live grid
    numeric = [int(str(x)) for x in (context.workspace.grid.keys or []) if str(x).isdigit()]
    current_max = max(numeric) if numeric else 0

    # initialise counter only once (or bump it if grid has moved ahead)
    counter = int(settings.get("_panel_id_counter", 0))
    counter = max(counter, current_max)

    # increment and persist
    counter += 1
    settings["_panel_id_counter"] = counter

    new_id = str(counter)
    
    dash = Dashboard(src=config.source, contents="Menu", context=context)
    try:
        view = dash.panel(in_grid=True)
    except TypeError:
        view = dash.panel()
    view = bind_controller(view, dash)

    # Compute a layout item for *one* breakpoint. WorkspaceManager will replicate
    # it across breakpoints if you coded it that way; if not, we can pass per-bp.
    #
    # To keep behavior closest to your original, we compute a bp layout item for each bp.
    # WorkspaceManager.add_panel currently takes one layout_item; so we pick lg if present,
    # otherwise first available breakpoint. (If you want exact per-bp placement, see note below.)
    new_layouts = (grid.layouts or {})
    cols_by_bp = (grid.cols_by_breakpoint or {})

    # Choose a "primary" breakpoint to compute placement from
    primary_bp = "lg" if "lg" in cols_by_bp else (next(iter(cols_by_bp.keys()), "lg"))
    cols = int(cols_by_bp.get(primary_bp, 12))
    w, h = _bp_geom(primary_bp)
    bp_layout = list(new_layouts.get(primary_bp, []))
    x, y = find_first_fit(bp_layout, cols=cols, w=w, h=h)

    layout_item = {"x": x, "y": y, "w": w, "h": h}
    
    context.workspace.add_panel(new_id, view, title="Menu", layout_item=layout_item)

def verify_import_config(curr_config_file, context):

    config = context.config

    has_error = False
    error_message = ""

    if (config.settings["config_load_level"] > 2) or (
        config.settings["config_load_level"] < 0
    ):
        has_error = True
        error_message += f"**Unable to import file due to the following errors:**\n\n\n\nconfig_load_level = {config.settings['config_load_level']} **[config_load_level should be 0,1 or 2]**\n"

        return has_error, error_message

    if config.settings["config_load_level"] > 0:
        
        columns_needed = [ "dataset_filepath",
                           "optimise_data",
                           "layout",
                           "id_col",
                           "label_col",
                           "labels",
                           "label_colours",
                           "labels_to_strings",
                           "strings_to_labels",
                            ]
        
        if curr_config_file.get("layout", {}).get("0", {}).get("contents") != "Exploring":
            columns_needed.extend(["extra_image_cols",
                                    "extra_info_cols",
                                    "feature_generation",
                                    "default_vars",
                                    "labels_to_train",
                                    "features_for_training",
                                    "exclude_labels",
                                    "exclude_unknown_labels",
                                    "unclassified_labels",
                                    "scale_data",
                                    "test_set_file",])

        missing_settings = list(
            set(columns_needed).difference(list(curr_config_file.keys()))
        )

        if len(missing_settings) > 0:
            has_error = True
            error_message += f"**Unable to import file due to the following errors:**\n\n\n\nThe config file is missing these settings: \n\n{missing_settings} \n\n **[Rerun astronomicAL and assign the settings yourself or manually edit `{layout_file}`, to include the missing settings]**\n\n\n"
            return has_error, error_message

        if "classifiers" not in list(curr_config_file.keys()):
            if config.settings["config_load_level"] == 2:
                config.settings["config_load_level"] = 1
                print(
                    "\n Switching to load level 1 as classifier data missing from imported config file\n"
                )

        update_config_settings(curr_config_file, config=config)
        filename = config.settings["dataset_filepath"]

        if not os.path.exists(filename):
            has_error = True
            error_message += f"**Unable to import file due to the following errors:**\n\n\n\nFile: {filename} does not exist. **[Check you have downloaded the correct dataset, with the correct name and placed it in your `data/` directory (symlinks are accepted)]**\n"

            return has_error, error_message

        try:
            ext = filename[filename.rindex(".") + 1 :]
        except:
            has_error = True
            error_message += f"**Unable to import file due to the following errors:**\n\n\n\nFile: `{filename}` has no extension. **[Extensions are required to load the data properly]**\n"

            return has_error, error_message
        try:
            table = Table.read(
                filename,
                format=f"{ext}",
            )
        except:
            has_error = True
            error_message += f"**Unable to import file due to the following errors:**\n\n\n\nExtension: {ext} is not a filetype that can be imported. **[See astropy documentation to see acceptable filetypes]**\n"

            return has_error, error_message

        columns_used = []

        for setting in [
            "id_col",
            "label_col",
            "features_for_training",
            "extra_info_cols",
            "extra_image_cols",
        ]:
            value = config.settings.get(setting) # In exploring mode this returns None for settings not required
            if value is None:
                continue 
            if isinstance(value, str):
                columns_used.append(value)

            elif isinstance(value, list):
                columns_used.extend(value)

        missing_cols = []
        for col in columns_used:
            if col not in table.colnames:
                missing_cols.append(col)

        if len(missing_cols) > 0:
            allowed_missing = ["Use Index", "No Labels"]
            if any(col not in allowed_missing for col in missing_cols):
                wrong_cols = [col for col in missing_cols if col not in  allowed_missing]
                has_error = True
                error_message += f"The dataset is missing these columns:\n\n{wrong_cols}\n\n **[Rerun astronomicAL and assign the settings yourself or manually edit `{filename}`, replacing the missing columns]**\n\n\n"
                error_message += "\n\n-------------------------------\n\n"
        
        if "feature_generation" not in missing_settings:
            opers = list(get_oper_dict().keys())
            missing_opers = []
            for oper in config.settings.get("feature_generation", []):
                if oper[0] not in opers:
                    missing_opers.append(oper[0])
            if len(missing_opers) > 0:
                has_error = True
                error_message += f"AstronomicAL is missing the following operations in `extensions/feature_generation.py`:\n\n{missing_opers}\n\n **[If they have not been uploaded to the astronomicAL repo you may need to contact the researcher who uploaded the config for the correct code]**\n\n\n"
                error_message += "\n\n-------------------------------\n\n"
        if "layout" in curr_config_file:
            plots = list(get_plot_dict().keys()) + list(get_customplot_dict(context=context).keys())
            contents = [
                "Settings",
                "Menu",
                "Active Learning",
                "Basic Plot",
                "Histogram Plot",
                "Density Plot",
                "Labelling",
                'Exploring',
                "Selected Source Info",
            ] + plots

            missing_contents = []

            for i in curr_config_file["layout"]:
                if "contents" in curr_config_file["layout"][i]:
                    if curr_config_file["layout"][i]["contents"] not in contents:
                        missing_contents.append(
                            curr_config_file["layout"][i]["contents"]
                        )

            if len(missing_contents) > 0:
                has_error = True
                error_message += f"AstronomicAL is missing the following plots in `extensions/extension_plots.py`:\n\n{missing_contents}\n\n **[If they have not been uploaded to the astronomicAL repo you may need to contact the researcher who uploaded the config for the correct code]**\n\n\n"
                error_message += "\n\n-------------------------------\n\n"
        
        if "classifiers" in curr_config_file:
            clfs = list(get_classifiers().keys())

            missing_clfs = []

            for i in curr_config_file["classifiers"]:
                if "classifier" in curr_config_file["classifiers"][i]:
                    for clf in curr_config_file["classifiers"][i]["classifier"]:
                        if clf not in clfs:
                            missing_clfs.append(clf)

            if len(missing_clfs) > 0:
                has_error = True
                error_message += f"AstronomicAL is missing the following classifiers in `extensions/models.py`:\n\n{missing_clfs}\n\n **[If they have not been uploaded to the astronomicAL repo you may need to contact the researcher who uploaded the config for the correct code]**\n\n\n"
                error_message += "\n\n-------------------------------\n\n"

        if "classifiers" in list(curr_config_file.keys()):
            qrys = list(get_strategy_dict().keys())

            missing_qrys = []

            for i in curr_config_file["classifiers"]:
                if "query" in curr_config_file["classifiers"][i]:
                    for qry in curr_config_file["classifiers"][i]["query"]:
                        if qry not in qrys:
                            missing_qrys.append(qry)

            if len(missing_qrys) > 0:
                has_error = True
                error_message += f"AstronomicAL is missing the following classifiers in `extensions/query_strategies.py`:\n\n{missing_qrys}\n\n **[If they have not been uploaded to the astronomicAL repo you may need to contact the researcher who uploaded the config for the correct code]**\n\n\n"
                error_message += "\n\n-------------------------------\n\n"

        if "test_set_file" in curr_config_file:
            if curr_config_file["test_set_file"]:
                if not os.path.isfile("data/test_set.json"):
                    has_error = True
                    error_message += f"AstronomicAL is missing the following test set file:\n\n `data/test_set.json` \n\n **[Your configuration file states it uses this file to create a verified test set. Change flag `test_file_set` to `false` in your config file to create a test set from the data (Classifier performance may be affected from previously stated results)]**\n\n\n"
                    error_message += "\n\n-------------------------------\n\n"
        
        has_error, error_message = verify_SED_config(curr_config_file, table,  has_error, error_message)   #checking in separate function for code readibility
        has_error, error_message = verify_euclid_cutout_config(curr_config_file, has_error, error_message)
        has_error, error_message = verify_plot_config(curr_config_file, table, has_error, error_message)
        
    if has_error:
        error_message = (
            "**Unable to import file due to the following errors:**\n\n\n\n"
            + error_message
        )

    return has_error, error_message


def update_config_settings(imported_config, config):

    ignore_keys = ["Author", "doi", "layout"]
    for key in imported_config.keys():
        if key in ignore_keys:
            continue
        elif key == "label_colours":
            label_colours = {}
            for i in imported_config["label_colours"]:
                label_colours[int(i)] = imported_config["label_colours"][i]
            config.settings[key] = label_colours
        else:
            config.settings[key] = imported_config[key]
    config.settings["confirmed"] = True

# keep your existing imports: Dashboard, DataSelection, update_config_settings, config, etc.

def create_layout_skeleton(react: pn.template.ReactTemplate, *, return_grid: bool=False):
    """
    Creates the DynamicReactGrid and attaches it to react.main,
    but does NOT create header/menu/dashboards.
    This lets main() construct AppContext first.
    """
    grid = DynamicReactGrid(keys=[], objects=[], layouts={})
    react._dynamic_grid = grid
    react.main[:12,:12] = grid
    if return_grid:
        return react, grid
    return react

def create_layout_from_file(
        react: pn.template.ReactTemplate,
        context=None,
        *,
        return_grid: bool = False,
    ):

    config = context.config

    with open(config.layout_file) as lf:
        print(f"Loading layout file: {lf}")
        curr_config_file = json.load(lf)


    # ---- keep your existing config/data loading exactly as-is ----
    if len(curr_config_file.keys()) > 1:
        if config.settings["config_load_level"] > 0:
            update_config_settings(curr_config_file, config=config)
            load_data = DataSelection(config.source, mode=config.mode, context = context)
            config.main_df = load_data.get_dataframe_from_fits_file(
                curr_config_file["dataset_filepath"],
                optimise_data=curr_config_file["optimise_data"],
            )

            src = {f"{col}": [] for col in config.main_df}
            if config.settings["id_col"] not in src:
                src[config.settings["id_col"]] = []
            config.source.data = src

    curr_layout = curr_config_file["layout"]

    # ---- NEW: create the DynamicReactGrid and populate it ----
    grid = DynamicReactGrid(
        sizing_mode="stretch_both",
        height=900,
        # choose your “3 across / 2 across / 1 across” thresholds
        breakpoints={"lg": 1350, "md": 900, "sm": 0},
        cols_by_breakpoint={"lg": 12, "md": 12, "sm": 12},
        resize_handles=["s","w","e","n","sw","nw","se"],
        compact_type="vertical",
    )

    keys = []
    objects = []
    colors = {}
    lg_layout = []

    # helper: generate md/sm from lg (simple packing)
    def _pack(ids, per_row, w, default_h=4):
        out = []
        for n, tid in enumerate(ids):
            base = next((it for it in lg_layout if it["i"] == tid), None) or {}
            h = int(base.get("h", default_h))
            out.append({"i": tid, "x": (n % per_row) * w, "y": (n // per_row) * h, "w": w, "h": h})
        return out

    # ---- same loop, but target the grid instead of react.main slices ----
    for p, panel in curr_layout.items():

        # existing geometry from file (assumed 12-col grid coords)
        x = int(panel.get("x", 0))
        y = int(panel.get("y", 0))
        w = int(panel.get("w", 4))
        h = int(panel.get("h", 4))

        # existing contents logic
        contents = panel.get("contents", "Menu")

        # The old code compares int(p)==0; keep that behaviour
        is_main = False
        try:
            is_main = (int(p) == 0)
        except Exception:
            is_main = (str(p) == "0")

        if is_main:
            if (contents == "Menu") or (config.settings["config_load_level"] == 0):
                contents = "Settings"
            elif config.mode == "Labelling":
                contents = "Labelling"
            elif config.mode == "AL":
                contents = "Active Learning"
            elif config.mode == "Exploring":
                contents = "Exploring"

        else:
            if "config_load_level" in config.settings and config.settings["config_load_level"] == 0:
                contents = "Menu"

        dash = Dashboard(src=config.source, contents=contents, context=context)
        config.dashboards[p] = dash

        # restore “Basic Plot” axis selections (your existing behaviour)
        if contents == "Basic Plot":
            pc = panel.get("panel_contents", None)
            if isinstance(pc, (list, tuple)) and len(pc) >= 2:
                x_axis, y_axis = pc[0], pc[1]
                if x_axis in list(config.source.data.keys()):
                    dash.panel_contents.X_variable = x_axis
                if y_axis in list(config.source.data.keys()):
                    dash.panel_contents.Y_variable = y_axis

        # ---- NEW: store into DynamicReactGrid ----
        tile_id = str(p)  # keep ids identical to existing file keys
        keys.append(tile_id)
        objects.append(dash.panel())

        # optional color/state if present in new schema
        if "color" in panel:
            colors[tile_id] = panel["color"]

        lg_layout.append({"i": tile_id, "x": x, "y": y, "w": w, "h": h})

        # optional: restore slider value if you later add it
        # if "value" in panel: set it here via your tile factory/state applier

    grid.keys = keys
    grid.objects = objects


    ids_in_layout_order = [it["i"] for it in sorted(lg_layout, key=lambda it: (it["y"], it["x"]))]

    md_layout = _pack(ids_in_layout_order, per_row=2, w=6)
    sm_layout = _pack(ids_in_layout_order, per_row=1, w=12)

    grid.breakpoints = {"lg": 1500, "md": 1050, "sm": 0}
    grid.cols_by_breakpoint = {"lg": 12, "md": 12, "sm": 12}
    grid.layouts = {"lg": lg_layout, "md": md_layout, "sm": sm_layout}

    # Put it into the template (single component)
    react.main[:12, :12] = grid   # 2D assignment
    react._dynamic_grid = grid

    react = create_header(react, grid, context=context)

    if return_grid:
        return react, grid
    return react


def create_default_layout(
    react: pn.template.ReactTemplate,
    context=None,
    *,
    return_grid: bool = False,
        ):
    print("No Layout File Found. Reverting to default dashboard layout (DynamicReactGrid).")

    grid = DynamicReactGrid(
        sizing_mode="stretch_both",
        height=800,
        # responsive behaviour you wanted:
        breakpoints={"lg": 1350, "md": 900, "sm": 0},
        cols_by_breakpoint={"lg": 12, "md": 12, "sm": 12},
        resize_handles=["s","w","e","n","sw","nw","se"],
        compact_type="vertical",
    )

    # ---- Build the same dashboards as before ----
    items = []

    # 1) Settings (top-left)
    main_plot = Dashboard(src=config.source, contents="Settings", context=context)
    config.dashboards[0] = main_plot
    items.append(("settings", main_plot.panel()))

    # 2) Top-right
    num = 0
    new_plot = Dashboard(src=config.source, context=context)
    config.dashboards[f"{num}"] = new_plot
    items.append((f"plot-{num}", new_plot.panel()))
    num += 1

    # 3) Bottom row: three plots
    for _ in [0, 4, 8]:
        new_plot = Dashboard(src=config.source, context=context)
        config.dashboards[f"{num}"] = new_plot
        items.append((f"plot-{num}", new_plot.panel()))
        num += 1

    # ---- Populate grid state ----
    grid.keys = [k for k, _ in items]
    grid.objects = [obj for _, obj in items]

    # ---- Default layout (lg): match your old template geometry ----
    # ReactTemplate was 12 cols, your slices:
    # settings: 0:6, 0:5  -> w=6, h=5
    # top-right: 6:12,0:5 -> w=6, h=5
    # bottom row: 0:4, 4:8, 8:12, rows 5:9 -> each w=4, h=4
    lg_layout = [
        {"i": "settings", "x": 0, "y": 0, "w": 6, "h": 12},
        {"i": "plot-0",   "x": 6, "y": 0, "w": 6, "h": 12},
        {"i": "plot-1",   "x": 0, "y": 12, "w": 4, "h": 8},
        {"i": "plot-2",   "x": 4, "y": 12, "w": 4, "h": 8},
        {"i": "plot-3",   "x": 8, "y": 12, "w": 4, "h": 8},
    ]

    # Generate md/sm from lg (3 across / 2 across / 1 across)
    # If you already have generator helpers, call them here.
    def pack(ids, per_row, w, h_default=4):
        out = []
        for n, tid in enumerate(ids):
            # keep each tile's original height if present
            base = next((it for it in lg_layout if it["i"] == tid), None) or {}
            h = int(base.get("h", h_default))
            out.append({"i": tid, "x": (n % per_row) * w, "y": (n // per_row) * h, "w": w, "h": h})
        return out

    ids = [it["i"] for it in lg_layout]
    md_layout = pack(ids, per_row=2, w=6)
    sm_layout = pack(ids, per_row=1, w=12)

    grid.breakpoints = {"lg": 1500, "md": 1050, "sm": 0}
    grid.cols_by_breakpoint = {"lg": 12, "md": 12, "sm": 12}
    grid.layouts = {"lg": lg_layout, "md": md_layout, "sm": sm_layout}

    react.main[:12, :12] = grid   # 2D assignment

    react._dynamic_grid = grid

    react = create_header(react, grid, context=context)

    if return_grid:
        return react, grid
    return react

def verify_column_properties(table, col_name, config_dict_name):
    """Checks that a column in the table contains numeric values"""
    has_error = False
    error_message = ""
    print(col_name)
    try:
        col = table[col_name]
        if col.dtype.kind not in {"i", "f"}:
            has_error = True
            error_message = f""""{col_name} in `{config_dict_name}` is not of numeric type in the dataset \n\n 
                                     **[Please replace it with a column storing numeric type objects]**\n\n\n"""
            error_message += "\n\n-------------------------------\n\n"

    except KeyError:
        #catch it somewhere else
        pass
    return has_error, error_message


def verify_config_dict(config_dict, validation_rules, config_dict_name):
    """General function to verify and validate a dictionary in the config file
       ----
       config_dict : dict, the dictionary to validate

       validation_rules : dict. It is a dictionary where the KEYS are all the allowed keys that can be found
                        in config_dict (it checks that user did not input some random key).
                        The VALUES are dictionaries which store the information to validate the input.
                        VALUES dictionaryes have as keys ['check', 'valid' 'error']. It is not required to pass always 
                        both 'check' and 'valid'.
                        'check'  (logic) function used to validate the config value (e.g. >0 AND float)
                        'valid' list of allowd values (e.g. [true, false], [VIS, NISP_Y, NISP_J])
                        'error' the error message to print if the 'check' or 'valid' condition are not satsfied.

        config_dict_name : The name of the config_dict, used in the Error messages               
                         
    """
    has_error = False
    error_message = ""
    if not isinstance(config_dict, dict):
                has_error = True
                error_message += f"""Wrong format for \n\n '{config_dict_name}' \n\n 
                                     **[It needs to be a dictionary]**\n\n\n"""
                error_message += "\n\n-------------------------------\n\n"
                return has_error, error_message
    
    wrong_keys = [i for i in config_dict if i not in validation_rules]
    if wrong_keys:
        has_error = True
        error_message += f""""`{config_dict_name}` has the following wrong keys {wrong_keys}  \n\n 
                                     **[Allowed keys are {list(validation_rules.keys())}]**\n\n\n"""
        error_message += "\n\n-------------------------------\n\n"
            
    for key, rule in validation_rules.items():
        if key not in config_dict:
            continue
        value = config_dict[key]
        if "valid" in rule and value not in rule["valid"]:
            has_error = True
            error_message += (
            f"Wrong {key} specified in {config_dict_name}: {value}\n\n**[{rule['error']}]**\n\n\n"
            "\n\n-------------------------------\n\n")

        if "check" in rule and not rule["check"](value):
            has_error = True
            error_message += (
            f"Wrong {key} specified in {config_dict_name}: {value}\n\n**[{rule['error']}]**\n\n\n"
            "\n\n-------------------------------\n\n")
    
    return has_error, error_message
  

def verify_SED_config(curr_config_file, table,  has_error, error_message):
    if "SED_bands" in curr_config_file:
        config_dict = curr_config_file["SED_bands"]
        if config_dict:
            if not isinstance(config_dict, dict):
                has_error = True
                error_message += f"""Wrong format for \n\n 'SED_bands' \n\n 
                                     **[It needs to be a dictionary with bands as keys and assoictaed columns as values]**\n\n\n"""
                error_message += "\n\n-------------------------------\n\n"  
            else:
                try:
                    filepath =  "data/sed_data/photometric_bands.json"
                    with open(filepath, 'r') as f:
                        filter_data = json.load(f)
    
                    missing_bands, missing_cols = [], []
                    for band, col in config_dict.items():
                        if ("err_" not in band) and (band not in filter_data):
                            missing_bands.append(band)
                        if col not in table.colnames:
                            missing_cols.append(col)
                    if len(missing_cols) > 0:
                        has_error = True
                        error_message += f"The dataset is missing these columns:\n\n{missing_cols}\n\n **[Rerun astronomicAL and assign the settings yourself or manually edit the config file, replacing the missing columns]**\n\n\n"
                        error_message += "\n\n-------------------------------\n\n"
                    if len(missing_bands) > 0:
                            has_error = True
                            error_message += f"The photometric file is missing these bands:\n\n{missing_bands}\n\n **[Rerun astronomicAL and assign the settings yourself or manually edit `data/sed_data/photometric_bands.json`, adding the missing bands]**\n\n\n"
                            error_message += "\n\n-------------------------------\n\n"
                            
                except FileNotFoundError:
                    has_error = True
                    error_message += f"""AstronomicAL is missing the following file:\n\n `data/sed_data/photometric_bands.json` \n\n 
                                     **[This file is needed to load information about filters in SED plot]**\n\n\n"""
                    error_message += "\n\n-------------------------------\n\n"
                    return has_error, error_message
    
    if "SED_units" in curr_config_file:
        config_dict = curr_config_file["SED_units"]
        if config_dict:
            if not isinstance(config_dict, dict):
                error_message += f"""Wrong format for \n\n 'SED_units' \n\n 
                                     **[It needs to be a dictionary with bands as keys and assoictaed units as values]**\n\n\n"""
                error_message += "\n\n-------------------------------\n\n"
            else:
                available_units = ["AB magnitudes", "milliJy", "microJy" , 
                                   "nanoJy", "cgs (erg/s/Hz/cm2)"]
                wrong_units = [u for u in config_dict.values() if u not in available_units]
                if wrong_units:
                    string_to_write = "'" + "', '".join(available_units) + "'"
                    has_error = True
                    error_message += f"Wrong units in SED_units:\n\n{wrong_units}\n\n **[Allowed units are: {string_to_write}]**\n\n\n"
                    error_message += "\n\n-------------------------------\n\n"
    return has_error, error_message


def verify_euclid_cutout_config(curr_config_file, has_error, error_message):
    if "Euclid_cutout_settings" in curr_config_file:
        config_dict = curr_config_file["Euclid_cutout_settings"]
        if config_dict:
            validation_rules = {
                "filter": {
                          "valid": {"VIS", "NIR_Y", "NIR_J", "NIR_H", "Color"},
                          "error": "Available filters: `VIS`, `NIR_Y`, `NIR_J`, `NIR_H`, or `Color`"
                          },
                "radius": {
                          "check": lambda r: isinstance(r, (int, float)) and 1 < r <= 100,
                          "error": "radius must be a number with `1 < radius ≤ 100`"
                          },
                "stretching":{
                               "valid" : {"Linear", "Sqrt", "Log", "Asinh", "PowerLaw"},
                               "error": "Available stretchings: `Linear`, `Sqrt`, `Log`, `Asinh`, `PowerLaw`"
                            },
                "scale": {
                          "valid": {"MinMax", "Expand"},
                          "error": "Available scale: `MinMax`, `Expand`"
                          },
                "clipping": {  
                          "check": lambda x: (isinstance(x, (list, tuple))
                                              and len(x) == 2
                                              and all(isinstance(v, (int, float)) for v in x)
                                              and 0 <= x[0] < x[1] <= 1),
                          "error": "clipping must be a list or tuple of `two ordered numbers between 0 and 1`"
                           },
                "source_coordinates": {
                                       "valid" : {True, False},
                                       "error" : "Available source_coordinates values: `true`, `false`"
                                       },
                "levels": {
                          "check": lambda l: isinstance(l, int) and 0 <= l <= 10,
                          "error": "levels must be a positive int number with `l ≤ 10`"
                          },
                
                "gamma": {  
                          "check": lambda x: (isinstance(x, (list, tuple))
                                              and len(x) == 3
                                              and all(isinstance(v, (int, float)) for v in x)
                                              and all(0 < v <= 5 for v in x)),
                          "error": "gamma must be a list or tuple of `three numbers between 0 and 5`"
                           },

                }
            
            new_error, new_message = verify_config_dict(config_dict, validation_rules, "Euclid_cutout_settings")
            has_error = has_error or new_error
            error_message += new_message
    
    return has_error, error_message


def verify_plot_config(curr_config_file, table, has_error, error_message):
    plot_names = ["Scatter_plot_settings", "Histogram_plot_settings"]
    if not any(key in curr_config_file for key in plot_names):
        return has_error, error_message
    
    general_validation_rules = {
                "X_variable": {
                          "valid": set(table.colnames),
                          "error": "Select a column in the dataset"
                          },
                "Y_variable": {
                          "valid": set(table.colnames),
                          "error": "Select a column in the dataset"
                          },
                
                "log_x": {
                         "valid" : {True, False},
                         "error" : "Available log_x values: `true`, `false`"
                         },
                
                "log_y": {
                         "valid" : {True, False},
                         "error" : "Available log_y values: `true`, `false`"
                         },

                "cumulative": {
                         "valid" : {True, False},
                         "error" : "Available cumulative values: `true`, `false`"
                         },
                
                "density": {
                         "valid" : {True, False},
                         "error" : "Available log_x values: `true`, `false`"
                         },

                "Nbins": {
                         "check" : lambda x: (isinstance(x, int)
                                              and  2 <= x <= 200),
                         "error" : "Nbins must be an integer value between 2 and 200"
                         },

                "range" : {  
                          "check": lambda x: (isinstance(x, (list, tuple))
                                              and len(x) == 2
                                              and all(isinstance(v, (int, float)) for v in x)
                                              and -np.inf <= x[0] < x[1] <= np.inf),
                          "error": "range must be a list or tuple of two ordered numbers between -np.inf and np.inf`"
                           },

                "labels": {
                           "check" : lambda x: isinstance(x, (list, tuple)),
                            "error" : "labels must be a list or a tuple"
                          },

                "mode": {
                         "valid" : {"tap", "rasterized"},
                         "error" : "Available mode values: `tap`, `rasterized`"
                         },         
                }
    for name in plot_names:
        if name in curr_config_file:
            config_dict = curr_config_file[name]
            if not config_dict:
                continue
            if name == "Histogram_plot_settings":
                validation_rules = {key : general_validation_rules[key] for key in general_validation_rules
                                   if key not in ["Y_variable", "mode"]}
            else:
                validation_rules = {key : general_validation_rules[key] for key in 
                                    ["X_variable", "Y_variable", "labels", "log_x", "log_y", "mode"]}
                
            new_error, new_message = verify_config_dict(config_dict, validation_rules, name)
            has_error = has_error or new_error
            error_message += new_message
            for key in ["X_variable", "Y_variable"]:
                try:
                    col_name = config_dict.get(key, None)
                    if col_name:
                        new_error, new_message =  verify_column_properties(table, col_name, name)
                        has_error = has_error or new_error
                        error_message += new_message
                except AttributeError:
                    ##config_dict is not a dict, but this is already catched verify_config_dict
                    continue
            try:
                labels = config_dict.get("labels", None)
                if labels:
                    wrong_labels = [lab for lab in labels if lab not in ["All"] + list(curr_config_file.get("string_to_labels", {}).keys())]
                    if wrong_labels:
                        has_error = True
                        error_message += f"Wrong labels:\n\n{wrong_labels}\n\n **[Please replace the wrong labels in {name}]**\n\n\n"
                        error_message += "\n\n-------------------------------\n\n"

            except AttributeError:
                    ##config_dict is not a dict, but this is already catched verify_config_dict
                    pass
    return has_error, error_message