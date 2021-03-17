import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../"))
from astronomicAL.dashboard.dashboard import Dashboard
from bokeh.models import ColumnDataSource, TextAreaInput

import astronomicAL.config as config
import holoviews as hv
import json
import pandas as pd
import panel as pn
import numpy as np

hv.extension("bokeh")
hv.renderer("bokeh").webgl = True


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_layout_cb(attr, old, new):
    print("json updated")
    layout = json.loads(new)
    with open("astronomicAL/layout.json", "w") as fp:
        json.dump(layout, fp)

    export_config = {}

    export_config["Author"] = ""
    export_config["doi"] = ""
    export_config["id_col"] = config.settings["id_col"]
    export_config["label_col"] = config.settings["label_col"]
    export_config["default_vars"] = config.settings["default_vars"]
    export_config["labels"] = config.settings["labels"]
    export_config["label_colours"] = config.settings["label_colours"]
    export_config["labels_to_strings"] = config.settings["labels_to_strings"]
    export_config["strings_to_labels"] = config.settings["strings_to_labels"]
    export_config["extra_info_cols"] = config.settings["extra_info_cols"]
    export_config["labels_to_train"] = config.settings["labels_to_train"]
    export_config["features_for_training"] = config.settings["features_for_training"]
    export_config["exclude_labels"] = config.settings["exclude_labels"]
    export_config["unclassified_labels"] = config.settings["unclassified_labels"]
    export_config["scale_data"] = config.settings["scale_data"]
    export_config["feature_generation"] = config.settings["feature_generation"]

    with open("astronomicAL/export_config.json", "w") as fp:
        json.dump(export_config, fp, cls=NumpyEncoder)


config.source = ColumnDataSource()

config.main_df = pd.DataFrame()

config.ml_data = {}

config.settings = {"confirmed": False}

files = pn.widgets.FileInput()

react = pn.template.ReactTemplate(title="astronomicAL")

pn.config.sizing_mode = "stretch_both"


if config.initial_setup:
    config.initial_setup = False
    if os.path.isfile(config.layout_file):
        with open(config.layout_file) as layout_file:
            curr_layout = json.load(layout_file)
            for p in curr_layout["layout"]:
                start_row = curr_layout["layout"][p]["y"]
                end_row = curr_layout["layout"][p]["y"] + curr_layout["layout"][p]["h"]
                start_col = curr_layout["layout"][p]["x"]
                end_col = curr_layout["layout"][p]["x"] + curr_layout["layout"][p]["w"]

                if int(p) == 0:
                    main_plot = Dashboard(
                        name="Main Plot", src=config.source, contents="Settings"
                    )
                    react.main[start_row:end_row, start_col:end_col] = main_plot.panel()
                else:
                    new_plot = Dashboard(name=f"{p}", src=config.source)
                    react.main[start_row:end_row, start_col:end_col] = new_plot.panel()

    else:
        main_plot = Dashboard(name="Main Plot", src=config.source, contents="Settings")
        react.main[:5, :6] = main_plot.panel()

        num = 0
        for i in [6]:
            new_plot = Dashboard(name=f"{num}", src=config.source)
            react.main[:5, 6:] = new_plot.panel()
            num += 1

        for i in [0, 4, 8]:
            new_plot = Dashboard(name=f"{num}", src=config.source)
            react.main[5:9, i : i + 4] = new_plot.panel()
            num += 1

else:
    if os.path.isfile(config.layout_file):
        with open(config.layout_file) as layout_file:
            curr_layout = json.load(layout_file)
            for p in curr_layout["layout"]:
                start_row = curr_layout["layout"][p]["y"]
                end_row = curr_layout["layout"][p]["y"] + curr_layout["layout"][p]["h"]
                start_col = curr_layout["layout"][p]["x"]
                end_col = curr_layout["layout"][p]["x"] + curr_layout["layout"][p]["w"]

                if int(p) == 0:
                    main_plot = Dashboard(
                        name="Main Plot", src=config.source, contents="Settings"
                    )
                    react.main[start_row:end_row, start_col:end_col] = main_plot.panel()
                else:
                    new_plot = Dashboard(
                        name=f"{p}", src=config.source, contents="Basic Plot"
                    )
                    react.main[start_row:end_row, start_col:end_col] = new_plot.panel()

save_layout_button = pn.widgets.Button(name="Save current layout")

layout_dict = {}
text_area_input = TextAreaInput(value="")
text_area_input.on_change("value", save_layout_cb)

save_layout_button.jscallback(
    clicks="""
function FindReact(dom, traverseUp = 0) {
const key = Object.keys(dom).find(key=>key.startsWith("__reactInternalInstance$"));
const domFiber = dom[key];
if (domFiber == null) return null;

// react 16+
const GetCompFiber = fiber=>{
//return fiber._debugOwner; // this also works, but is __DEV__ only
let parentFiber = fiber.return;
while (typeof parentFiber.type == "string") {
parentFiber = parentFiber.return;
}
return parentFiber;
};
let compFiber = GetCompFiber(domFiber);
for (let i = 0; i < traverseUp; i++) {
compFiber = GetCompFiber(compFiber);
}
return compFiber.stateNode;
}
var react_layout = document.getElementById("responsive-grid")
const someElement = react_layout.children[0];
const myComp = FindReact(someElement);
var layout_dict = {};

for(var i = 0; i < myComp["props"]["layout"].length; i++) {

layout_dict[i] = {
"x":myComp["state"]["layout"][i]["x"],
"y":myComp["state"]["layout"][i]["y"],
"w":myComp["state"]["layout"][i]["w"],
"h":myComp["state"]["layout"][i]["h"],
}
}

console.log(layout_dict)

text_area_input.value = JSON.stringify(layout_dict)

""",
    args=dict(text_area_input=text_area_input),
)

react.header.append(pn.Row(save_layout_button))

react.servable()

# TODO :: http://holoviews.org/reference/apps/bokeh/player.html#apps-bokeh-gallery-player
# TODO :: https://panel.holoviz.org/gallery/simple/save_filtered_df.html#simple-gallery-save-filtered-df

# CHANGED :: Keep axis the same on refresh (stop zooming out - outliers)
# CHANGED :: Save React Layout (likely need to wait for Panel update)
# CHANGED :: Fix Datashader Colours to match Bokeh
