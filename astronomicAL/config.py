from multiprocessing import Process
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, TextAreaInput
from functools import partial
import time
import datetime
import os
from astronomicAL.utils import save_logbook

initial_setup = True


settings = {"confirmed": False}


def get_save_layout_button(enable_button, from_main, context=None):
    """
    Build the workspace save button.

    This uses the new context.persistence save path instead of scraping the
    React layout with JavaScript.
    """
    import panel as pn

    button_key = "save_button"

    if (button_key not in settings) or from_main:
        button = pn.widgets.Button(
            name="Save Workspace",
            disabled=not bool(enable_button),
            button_type="primary",
            width=150,
        )

        status = pn.pane.Markdown(
            "",
            visible=False,
            width=260,
            margin=(6, 0, 0, 8),
        )

        def _save(_event):
            if context is None:
                status.object = "Save failed: no context."
                status.visible = True
                button.name = "Save Workspace"
                return

            try:
                from astronomicAL.utils.save_config import save_workspace

                config_obj = getattr(context, "config", None)
                path = getattr(config_obj, "layout_file", None) or "configs/workspace.json"

                save_workspace(context, path)

                button.name = "Saved"
                status.object = f"Saved: `{path}`"
                status.visible = True

            except Exception as exc:
                button.name = "Save failed"
                status.object = f"Save failed: `{exc}`"
                status.visible = True

        button.on_click(_save)

        settings[button_key] = pn.Row(
            button,
            status,
            sizing_mode="fixed",
        )

    if not from_main:
        try:
            settings[button_key][0].disabled = not bool(enable_button)
        except Exception:
            pass

    return settings[button_key]


def _save_layout_button_rename(context=None):
    """
    Removed old asynchronous rename behaviour.

    Kept as a harmless no-op in case anything still imports it while the
    surrounding UI is being cleaned up.
    """
    return None


def _save_layout_button_cb(event=None, context=None):
    """
    Removed old JS-triggered save callback.

    Saving is now handled directly by get_save_layout_button().
    """
    return None

def get_save_panel_data_button(enable_button):
    settings["save_panel_button"] = pn.widgets.Button(name="Export Panel Data", disabled = not enable_button, button_type = "default")
    settings["save_panel_button"].on_click(save_panel_data_button_cb)
    return settings["save_panel_button"]

def get_save_logbook_button(enable_button):
    settings["save_logbook_button"] = pn.widgets.Button(name="Export Logbook", disabled = not enable_button, button_type = "default")
    settings["save_logbook_button"].on_click(save_logbook_button_cb)
    return settings["save_logbook_button"]

def save_panel_data_button_cb(event):
     """ Call the _save_panel method for all the panels which allow to save their stored plots and  fits file.
         Currently it relies on Exploring or Labeling dashboards to being the first ones in order to create a folder with the sourceid name """
     print("Calling the save button callback")
     save_dir = "data/saved_sources"
     sourceid = None                 
     for dashboard_number, dashboard in dashboards.items():
        if sourceid is None:
            if hasattr(dashboard.panel_contents, "_get_selected_id"):
                sourceid = str(dashboard.panel_contents._get_selected_id())
                main_dir = os.path.join(save_dir, sourceid)
                os.makedirs(main_dir, exist_ok=True)
        if hasattr(dashboard.panel_contents, "_save_panel"):
            _ = dashboard.panel_contents._save_panel(directory_path = main_dir,
                                                     save_fits_files = True)

def save_logbook_button_cb(event):
    
    logbook_directory = settings.get("logbook_directory", None)
    if logbook_directory is None:
        logbook_directory = "data/logbook_" + datetime.date.today().strftime("%Y_%m_%d")
        os.makedirs(logbook_directory, exist_ok = True)
    else:
        os.makedirs(logbook_directory, exist_ok = True)
    
    filename = os.path.join(logbook_directory, "text.tex")
    
    if not os.path.isfile(filename):
       save_logbook.initialize_latex(filename)
                          
    sourceid = None
    figure_paths = []
    text_notes = ""
    source_dataframe = pd.DataFrame()
    for dashboard_number, dashboard in dashboards.items():
        if sourceid is None:
            if hasattr(dashboard.panel_contents, "_get_selected_id"):
                sourceid = dashboard.panel_contents._get_selected_id()
                if sourceid is not None:
                     sourceid = str(sourceid)

        if hasattr(dashboard.panel_contents, "_save_panel"):
            path = dashboard.panel_contents._save_panel(directory_path = logbook_directory, 
                                                        save_fits_files = False,
                                                        prefix = sourceid)
            figure_path = path.get("figure", None)
            if figure_path is not None:
                figure_paths.append(os.path.relpath(figure_path, logbook_directory))
            
            text = path.get("text", None)
            if text is not None:
                text_notes += (text + "\\")

        
        elif hasattr(dashboard.panel_contents, "_save_extra_info_df"):
            source_dataframe = dashboard.panel_contents._save_extra_info_df()

    string = save_logbook.export_source(sourceid = sourceid,
                               source_dataframe = source_dataframe,
                               text_notes = text_notes,
                               figure_paths = figure_paths)
    save_logbook.append_latex(filename = filename, new_content = string)
    
    
       
                   
                                          

dashboards = {}

source = ColumnDataSource()

main_df = pd.DataFrame()

ml_data = {}
