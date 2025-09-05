from multiprocessing import Process
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, TextAreaInput
from functools import partial
import time
import os

initial_setup = True


settings = {"confirmed": False}


def get_save_layout_button(enable_button, from_main):
    from astronomicAL.utils import save_config

    if ("save_button" not in settings.keys()) or from_main:
        settings["save_button"] = pn.widgets.Button(
            name="Save Current Configuration", disabled=not (enable_button)
        )
        #layout_dict = {}
        text_area_input = TextAreaInput(value="")
        text_area_input.on_change(
            "value",
            partial(
                save_config.save_config_file_cb,
                trigger_text=text_area_input,
                autosave=False,
            ),
        )

        settings["save_button"].jscallback(
            clicks=save_config.save_layout_js_cb,
            args=dict(text_area_input=text_area_input),
        )

        settings["save_button"].on_click(_save_layout_button_cb)

        return settings["save_button"]
    if not from_main:
        settings["save_button"].disabled = not (enable_button)
        return settings["save_button"]


def _save_layout_button_rename():
    get_save_layout_button(settings["confirmed"], True).disabled = True
    get_save_layout_button(
        settings["confirmed"], True
    ).name = "Configuration saved to configs folder with current timestamp."
    time.sleep(3)
    get_save_layout_button(
        settings["confirmed"], True
    ).name = "Save Current Configuration"
    if settings["confirmed"]:
        get_save_layout_button(settings["confirmed"], True).disabled = False


def _save_layout_button_cb(event):
    Process(target=_save_layout_button_rename).start()


def get_save_panel_data_button(enable_button):
    save_panel_button = pn.widgets.Button(name="Export Panel Data", disabled = not enable_button, button_type = "default")
    save_panel_button.on_click(save_panel_data_button_cb)
    return save_panel_button


def save_panel_data_button_cb(event):
     """ Call the _save_panel method for all the panels which allow to save their stored plots and  fits file.
         Currently it relies on Exploring or Labeling dashboards to being the first ones in order to create a folder with the sourceid name """
     print("Calling the save button callback")
     save_dir = "data/saved_sources"
     sourceid = None                 
     for dashboard_number, dashboard in dashboards.items():
        if hasattr(dashboard.panel_contents, "_get_selected_id"):
            sourceid = str(dashboard.panel_contents._get_selected_id())
            main_dir = os.path.join(save_dir, sourceid)
            os.makedirs(main_dir, exist_ok=True)
        elif hasattr(dashboard.panel_contents, "_save_panel"):
            dashboard.panel_contents._save_panel(directory_path = main_dir)
                                          

layout_file = "astronomicAL/layout.json"
dashboards = {}

source = ColumnDataSource()

main_df = pd.DataFrame()

ml_data = {}
