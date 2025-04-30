import astronomicAL.config as config
from astronomicAL.dashboard.dashboard import Dashboard
from astronomicAL.extensions.extension_plots import get_plot_dict
from astronomicAL.extensions.feature_generation import get_oper_dict
from astronomicAL.extensions.models import get_classifiers
from astronomicAL.extensions.query_strategies import get_strategy_dict
from astronomicAL.settings.data_selection import DataSelection
from astropy.table import Table
import json
import os


def verify_import_config(curr_config_file):

    has_error = False
    error_message = ""

    if (config.settings["config_load_level"] > 2) or (
        config.settings["config_load_level"] < 0
    ):
        has_error = True
        error_message += f"**Unable to import file due to the following errors:**\n\n\n\nconfig_load_level = {config.settings['config_load_level']} **[config_load_level should be 0,1 or 2]**\n"

        return has_error, error_message

    if config.settings["config_load_level"] > 0:

        columns_needed = [
            "dataset_filepath",
            "optimise_data",
            "layout",
            "id_col",
            "label_col",
            "default_vars",
            "labels",
            "label_colours",
            "labels_to_strings",
            "strings_to_labels",
            "extra_info_cols",
            "extra_image_cols",
            "labels_to_train",
            "features_for_training",
            "exclude_labels",
            "exclude_unknown_labels",
            "unclassified_labels",
            "scale_data",
            "feature_generation",
            "test_set_file",
        ]

        missing_settings = list(
            set(columns_needed).difference(list(curr_config_file.keys()))
        )

        if len(missing_settings) > 0:
            has_error = True
            error_message += f"**Unable to import file due to the following errors:**\n\n\n\nThe config file is missing these settings: \n\n{missing_settings} \n\n **[Rerun astronomicAL and assign the settings yourself or manually edit `{config.layout_file}`, to include the missing settings]**\n\n\n"
            return has_error, error_message

        if "classifiers" not in list(curr_config_file.keys()):
            if config.settings["config_load_level"] == 2:
                config.settings["config_load_level"] = 1
                print(
                    "\n Switching to load level 1 as classifier data missing from imported config file\n"
                )

        update_config_settings(curr_config_file)
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
            if type(config.settings[setting]) is str:
                columns_used.append(config.settings[setting])

            elif type(config.settings[setting]) is list:
                for col in config.settings[setting]:
                    columns_used.append(col)

        missing_cols = []
        for col in columns_used:
            if col not in table.colnames:
                missing_cols.append(col)

        if len(missing_cols) > 0:
            has_error = True
            error_message += f"The dataset is missing these columns:\n\n{missing_cols}\n\n **[Rerun astronomicAL and assign the settings yourself or manually edit `{filename}`, replacing the missing columns]**\n\n\n"
            error_message += "\n\n-------------------------------\n\n"
        if "feature_generation" not in missing_settings:
            opers = list(get_oper_dict().keys())
            missing_opers = []
            for oper in config.settings["feature_generation"]:
                if oper[0] not in opers:
                    missing_opers.append(oper[0])
            if len(missing_opers) > 0:
                has_error = True
                error_message += f"AstronomicAL is missing the following operations in `extensions/feature_generation.py`:\n\n{missing_opers}\n\n **[If they have not been uploaded to the astronomicAL repo you may need to contact the researcher who uploaded the config for the correct code]**\n\n\n"
                error_message += "\n\n-------------------------------\n\n"
        if "layout" in list(curr_config_file.keys()):
            plots = list(get_plot_dict().keys())
            contents = [
                "Settings",
                "Menu",
                "Active Learning",
                "Basic Plot",
                "Labelling",
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
        if "classifiers" in list(curr_config_file.keys()):
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

        if "test_set_file" in list(curr_config_file.keys()):
            if curr_config_file["test_set_file"]:
                if not os.path.isfile("data/test_set.json"):
                    has_error = True
                    error_message += f"AstronomicAL is missing the following test set file:\n\n `data/test_set.json` \n\n **[Your configuration file states it uses this file to create a verified test set. Change flag `test_file_set` to `false` in your config file to create a test set from the data (Classifier performance may be affected from previously stated results)]**\n\n\n"
                    error_message += "\n\n-------------------------------\n\n"
    if has_error:
        error_message = (
            "**Unable to import file due to the following errors:**\n\n\n\n"
            + error_message
        )

    return has_error, error_message


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
        else:
            config.settings[key] = imported_config[key]

    config.settings["confirmed"] = True


def create_layout_from_file(react):

    with open(config.layout_file) as layout_file:
        curr_config_file = json.load(layout_file)

    if len(curr_config_file.keys()) > 1:

        if config.settings["config_load_level"] > 0:

            update_config_settings(curr_config_file)
            load_data = DataSelection(config.source, mode=config.mode)
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
        print("curr_layout: ", p)
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
            elif config.mode == "Labelling":
                contents = "Labelling"
            elif config.mode == "AL":
                contents = "Active Learning"
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
