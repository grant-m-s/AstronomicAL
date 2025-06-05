from datetime import datetime

import astronomicAL.config as config
from astropy.table import Table
import json
import numpy as np

save_layout_js_cb = """
function FindReact(dom, traverseUp = 0) {
const key = Object.keys(dom).find(key=>key.startsWith("__reactFiber$"));
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


def save_config_file(layout_from_js, trigger_text, autosave=False, test=False):

    if layout_from_js == "":
        return

    layout = json.loads(layout_from_js)
    trigger_text.value = ""
    for i in layout:
        curr_contents = config.dashboards[i].contents
        layout[i]["contents"] = curr_contents
        if curr_contents == "Basic Plot":
            layout[i]["panel_contents"] = [
                config.dashboards[i].panel_contents.X_variable,
                config.dashboards[i].panel_contents.Y_variable,
            ]

    export_config = {}

    export_config["Author"] = ""
    export_config["doi"] = ""
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
    export_config["extra_image_cols"] = config.settings["extra_image_cols"]
    export_config["labels_to_train"] = config.settings["labels_to_train"]
    export_config["features_for_training"] = config.settings["features_for_training"]
    export_config["exclude_labels"] = config.settings["exclude_labels"]
    export_config["exclude_unknown_labels"] = config.settings["exclude_unknown_labels"]
    export_config["unclassified_labels"] = config.settings["unclassified_labels"]
    export_config["scale_data"] = config.settings["scale_data"]
    export_config["feature_generation"] = config.settings["feature_generation"]
    export_config["test_set_file"] = config.settings["test_set_file"]

    if "classifiers" not in config.settings.keys():
        config.settings["classifiers"] = {}

    export_config["classifiers"] = config.settings["classifiers"]

    if autosave:
        print("AUTOSAVING...")
        with open("configs/autosave.json", "w") as fp:
            json.dump(export_config, fp, cls=NumpyEncoder, indent=4)
    elif test:
        with open(f"configs/config_export.json", "w") as fp:
            json.dump(export_config, fp, cls=NumpyEncoder, indent=4)
    else:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        with open(f"configs/config_{dt_string}.json", "w") as fp:
            json.dump(export_config, fp, cls=NumpyEncoder, indent=4)

        print(f"Final Export Config Settings: {export_config}")
        print(f"Config File saved to: configs/config_{dt_string}.json")


def save_dataframe_to_fits(df, filename, overwrite=True):
    assert (
        len(df.columns) <= 999
    ), f"FITS Files only allow up to 999 columns, dataframe contains {len(df.columns)}"
    t = Table.from_pandas(df)
    t.write(filename, overwrite=overwrite)
