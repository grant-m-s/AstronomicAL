import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../"))
from astronomicAL.utils import load_config
import astronomicAL.config as config
import holoviews as hv
import json
import panel as pn
import pandas as pd
import time

hv.extension("bokeh")
hv.renderer("bokeh").webgl = True


def export_fits_file_cb(event):

    list_ids = []
    list_labels = []

    if config.settings["confirmed"]:

        if "classifiers" in config.settings.keys():
            for label in config.settings["classifiers"]:
                if ("id" in config.settings["classifiers"][label].keys()) and (
                    "y" in config.settings["classifiers"][label].keys()
                ):
                    for i in range(len(config.settings["classifiers"][label]["id"])):
                        list_ids.append(config.settings["classifiers"][label]["id"][i])
                        list_labels.append(
                            config.settings["classifiers"][label]["y"][i]
                        )
        orig_labelled_data = {}
        if "test_set_file" in config.settings.keys():
            if config.settings["test_set_file"]:
                if os.path.exists("data/test_set.json"):
                    with open("data/test_set.json", "r") as json_file:
                        orig_labelled_data = json.load(json_file)

                for id in list(orig_labelled_data.keys()):
                    list_ids.append(id)
                    list_labels.append(orig_labelled_data[id])

    if len(list_ids) != 0:
        exported_labels = pd.DataFrame(
            {"id": list_ids, "label": list_labels}, dtype="string"
        )

        from astronomicAL.utils.save_config import save_dataframe_to_fits
        from datetime import datetime

        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H:%M:%S")
        save_dataframe_to_fits(exported_labels, f"data/labelled_data_{dt_string}.fits")

        export_fits_file_button.disabled = True
        export_fits_file_button.name = f"{len(list_ids)} labelled sources saved to 'data/labelled_data_{dt_string}.fits'"
        print(
            f"{len(list_ids)} labelled sources saved to 'data/labelled_data_{dt_string}.fits'"
        )
        time.sleep(3)
        export_fits_file_button.name = "Export Labelled Data to Fits File"
        export_fits_file_button.disabled = False
    else:

        export_fits_file_button.disabled = True
        export_fits_file_button.name = "No Labelled Data Found"
        time.sleep(3)
        export_fits_file_button.name = "Export Labelled Data to Fits File"
        export_fits_file_button.disabled = False


files = pn.widgets.FileInput()

react = pn.template.ReactTemplate(title="astronomicAL")

pn.config.sizing_mode = "stretch_both"

if os.path.isfile(config.layout_file):
    print("loading from file...")
    react = load_config.create_layout_from_file(react)
    print("loaded from file...")

else:
    react = load_config.create_default_layout(react)

# export_fits_file_button = pn.widgets.Button(
#     name="Export Labelled Data to Fits File", disabled=False
# )

# export_fits_file_button.on_click(export_fits_file_cb)


# react.header.append(
#     pn.Row(
#         config.get_save_layout_button(config.settings["confirmed"], True),
#         export_fits_file_button,
#     )
# )

react.servable()

# TODO :: http://holoviews.org/reference/apps/bokeh/player.html#apps-bokeh-gallery-player
# TODO :: https://panel.holoviz.org/gallery/simple/save_filtered_df.html#simple-gallery-save-filtered-df

# CHANGED :: Keep axis the same on refresh (stop zooming out - outliers)
# CHANGED :: Save React Layout (likely need to wait for Panel update)
# CHANGED :: Fix Datashader Colours to match Bokeh
