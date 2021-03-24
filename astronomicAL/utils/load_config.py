import astronomicAL.config as config
from astronomicAL.dashboard.dashboard import Dashboard
from astronomicAL.settings.data_selection import DataSelection
import json


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

    print(config.settings)


def create_layout_from_file(react):

    with open(config.layout_file) as layout_file:
        curr_config_file = json.load(layout_file)
        print(curr_config_file)
        if len(curr_config_file.keys()) > 1:
            print("IS LOADING NEW")

            print(config.settings.keys())

            if config.settings["config_load_level"] > 0:

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
                contents = curr_layout[p]["contents"]
            else:
                contents = "Menu"

            if int(p) == 0:
                if (contents == "Menu") or (config.settings["config_load_level"] == 0):
                    contents = "Settings"
                main_plot = Dashboard(src=config.source, contents=contents)
                config.dashboards[p] = main_plot
                react.main[start_row:end_row, start_col:end_col] = main_plot.panel()
            else:
                if "config_load_level" in list(config.settings.keys()):
                    if config.settings["config_load_level"] == 0:
                        contents = "Menu"
                new_plot = Dashboard(src=config.source, contents=contents)
                config.dashboards[p] = new_plot
                if contents == "Basic Plot":

                    x_axis = curr_layout[p]["panel_contents"][0]
                    y_axis = curr_layout[p]["panel_contents"][1]

                    if x_axis in list(config.source.data.keys()):

                        new_plot.panel_contents.X_variable = curr_layout[p][
                            "panel_contents"
                        ][0]
                    if y_axis in list(config.source.data.keys()):
                        new_plot.panel_contents.Y_variable = curr_layout[p][
                            "panel_contents"
                        ][1]
                react.main[start_row:end_row, start_col:end_col] = new_plot.panel()

    return react


def create_default_layout(react):

    print(
        "No Layout File Found. Reverting to default found in astronomicAL/utils/save_config.py"
    )

    main_plot = Dashboard(src=config.source, contents="Settings")
    config.dashboards[0] = main_plot
    react.main[:5, :6] = main_plot.panel()

    num = 0
    for i in [6]:
        new_plot = Dashboard(src=config.source)
        config.dashboards[f"{num}"] = new_plot

        react.main[:5, 6:] = new_plot.panel()
        num += 1

    for i in [0, 4, 8]:
        new_plot = Dashboard(src=config.source)
        config.dashboards[f"{num}"] = new_plot

        react.main[5:9, i : i + 4] = new_plot.panel()
        num += 1

    return react
