import astronomicAL.config as config
from astronomicAL.dashboard.dashboard import Dashboard
from astronomicAL.settings.data_selection import DataSelection
import json
import numpy as np

save_layout_js_cb = """
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

"""


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


def save_config_file_cb(attr, old, new, trigger_text, autosave):
    save_config_file(new, trigger_text=trigger_text, autosave=autosave)


def save_config_file(layout_from_js, trigger_text, autosave=False):
    print("json updated")
    print(f"LAYOUT VALUE:{layout_from_js}")

    if layout_from_js == "":
        return

    layout = json.loads(layout_from_js)
    trigger_text.value = ""
    print(layout)
    for i in layout:
        curr_contents = config.dashboards[i].contents
        layout[i]["contents"] = curr_contents
        if curr_contents == "Basic Plot":
            layout[i]["panel_contents"] = [
                config.dashboards[i].panel_contents.X_variable,
                config.dashboards[i].panel_contents.Y_variable,
            ]

    print(f"FINAL CONFIG SETTINGS: {config.settings}")

    export_config = {}

    export_config["Author"] = ""
    export_config["doi"] = ""
    export_config["dataset_filepath"] = ""
    export_config["dataset_filepath"] = config.settings["dataset_filepath"]
    export_config["optimise_data"] = config.settings["optimise_data"]
    export_config["layout"] = layout
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
    export_config["classifiers"] = config.settings["classifiers"]

    print(f"FINAL Export CONFIG SETTINGS: {export_config}")

    if autosave:
        with open("configs/autosave.json", "w") as fp:
            json.dump(export_config, fp, cls=NumpyEncoder)
    else:
        with open("configs/export_config.json", "w") as fp:
            json.dump(export_config, fp, cls=NumpyEncoder)


def update_config_settings(imported_config):

    ignore_keys = ["Author", "doi", "layout"]
    for key in imported_config.keys():
        if key in ignore_keys:
            continue
        elif key == "label_colours":
            label_colours = {}
            for i in imported_config["label_colours"]:
                label_colours[int(i)] = imported_config["label_colours"][i]
            config.settings[key] = label_colours
            print(config.settings["label_colours"])
        else:
            print(key)
            config.settings[key] = imported_config[key]

        config.settings["confirmed"] = True


def create_layout_from_file(react):

    with open(config.layout_file) as layout_file:
        curr_config_file = json.load(layout_file)
        print(curr_config_file)
        if len(curr_config_file.keys()) > 1:
            print("IS LOADING NEW")
            update_config_settings(curr_config_file)
            load_data = DataSelection(config.source)
            config.main_df = load_data.get_dataframe_from_fits_file(
                curr_config_file["dataset_filepath"],
                optimise_data=curr_config_file["optimise_data"],
            )

            src = {}
            for col in config.main_df:
                src[f"{col}"] = []

            config.source.data = src

        curr_layout = curr_config_file["layout"]

        for p in curr_layout:
            start_row = curr_layout[p]["y"]
            end_row = curr_layout[p]["y"] + curr_layout[p]["h"]
            start_col = curr_layout[p]["x"]
            end_col = curr_layout[p]["x"] + curr_layout[p]["w"]

            if "contents" in curr_layout[p].keys():
                print("HAS CONTENTS")
                contents = curr_layout[p]["contents"]
            else:
                contents = "Menu"

            if int(p) == 0:
                if contents == "Menu":
                    contents = "Settings"
                main_plot = Dashboard(
                    name="Main Plot", src=config.source, contents=contents
                )
                config.dashboards[p] = main_plot
                react.main[start_row:end_row, start_col:end_col] = main_plot.panel()
            else:
                new_plot = Dashboard(name=f"{p}", src=config.source, contents=contents)
                config.dashboards[p] = new_plot
                if contents == "Basic Plot":
                    new_plot.panel_contents.X_variable = curr_layout[p][
                        "panel_contents"
                    ][0]
                    new_plot.panel_contents.Y_variable = curr_layout[p][
                        "panel_contents"
                    ][1]
                react.main[start_row:end_row, start_col:end_col] = new_plot.panel()

    return react


def create_default_layout(react):

    print(
        "No Layout File Found. Reverting to default found in astronomicAL/utils/save_config.py"
    )

    main_plot = Dashboard(name="Main Plot", src=config.source, contents="Settings")
    config.dashboards[0] = main_plot
    react.main[:5, :6] = main_plot.panel()

    num = 0
    for i in [6]:
        new_plot = Dashboard(name=f"{num}", src=config.source)
        config.dashboards[f"{num}"] = new_plot

        react.main[:5, 6:] = new_plot.panel()
        num += 1

    for i in [0, 4, 8]:
        new_plot = Dashboard(name=f"{num}", src=config.source)
        config.dashboards[f"{num}"] = new_plot

        react.main[5:9, i : i + 4] = new_plot.panel()
        num += 1

    return react
