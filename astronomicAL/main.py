from astropy.table import Table
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    TableColumn,
    TextAreaInput
)
from bokeh.plotting import figure
from datetime import datetime
from functools import partial
from holoviews.operation.datashader import (
    datashade,
    dynspread,
)
from itertools import combinations
from joblib import dump
from modAL.uncertainty import (
    entropy_sampling,
    margin_sampling,
    uncertainty_sampling
)
from modAL.models import ActiveLearner, Committee
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import datashader as ds
import holoviews as hv
import json
import numpy as np
import os
import pandas as pd
import panel as pn
import param
import time

hv.extension("bokeh")
hv.renderer("bokeh").webgl = True

############################### SETTINGS PIPELINE ##############################


class SettingsDashboard(param.Parameterized):

    def __init__(self, main, src, df, ** params):
        super(SettingsDashboard, self).__init__(**params)
        self.row = pn.Row(pn.pane.Str("loading"))

        self.src = src

        self.df = None

        self.pipeline_stage = 0

        self.close_settings_button = pn.widgets.Button(
            name="Close Settings",
            max_width=100,
            disabled=True
        )
        self.close_settings_button.on_click(
            partial(self.close_settings_cb, main=main))

        self.pipeline = pn.pipeline.Pipeline()
        self.pipeline.add_stage(
            "Select Your Data", DataSelection(src), ready_parameter="ready"
        ),
        self.pipeline.add_stage(
            "Assign Parameters", ParameterAssignment(df),
            ready_parameter="ready"
        ),
        self.pipeline.add_stage(
            "Active Learning Settings", ActiveLearningSettings(
                src, self.close_settings_button)
        )

        self.pipeline.layout[0][0][0].sizing_mode = "fixed"

        self.pipeline.layout[0][0][0].max_height = 75

        self.pipeline.layout[0][2][0].sizing_mode = "fixed"
        self.pipeline.layout[0][2][1].sizing_mode = "fixed"
        self.pipeline.layout[0][2][0].height = 30
        self.pipeline.layout[0][2][1].height = 30
        self.pipeline.layout[0][2][0].width = 100
        self.pipeline.layout[0][2][1].width = 100

        self.pipeline.layout[0][2][0].on_click(self.stage_previous_cb)

        self.pipeline.layout[0][2][1].button_type = 'success'
        self.pipeline.layout[0][2][1].on_click(self.stage_next_cb)

        print(self.pipeline)
        print(self.pipeline["Assign Parameters"].get_id_column())

    def get_settings(self):

        updated_settings = {}
        updated_settings["id_col"] = self.pipeline["Assign Parameters"].get_id_column(
        )
        updated_settings["label_col"] = self.pipeline[
            "Assign Parameters"
        ].get_label_column()
        updated_settings["default_vars"] = self.pipeline[
            "Assign Parameters"
        ].get_default_variables()
        updated_settings["label_colours"] = self.pipeline[
            "Assign Parameters"
        ].get_label_colours()

        return updated_settings

    def close_settings_cb(self, event, main):
        print("closing settings")

        self.df = self.pipeline["Active Learning Settings"].get_df()

        global main_df

        main_df = self.df

        src = {}
        for col in self.df.columns:
            src[f"{col}"] = []

        self.src.data = src
        print("\n\n\n\n")
        print(len(list(self.src.data.keys())))

        self.close_settings_button.disabled = True
        self.close_settings_button.name = "Setting up training panels..."

        main.set_contents(updated="Active Learning")

    def stage_previous_cb(self, event):

        self.pipeline_stage -= 1

    def stage_next_cb(self, event):

        if self.df is None:
            print("updating Settings df")
            self.df = self.pipeline["Select Your Data"].get_df()

        pipeline_list = list(self.pipeline._stages)
        print("STAGE:")
        current_stage = pipeline_list[self.pipeline_stage]

        next_stage = pipeline_list[self.pipeline_stage + 1]
        self.pipeline[next_stage].update_data(dataframe=self.df)

        self.pipeline_stage += 1

    def panel(self):
        if self.pipeline["Active Learning Settings"].is_complete():
            self.close_settings_button.disabled = False

        self.row[0] = pn.Card(
            pn.Column(
                pn.Row(
                    self.pipeline.title,
                    self.pipeline.buttons,
                ),
                self.pipeline.stage,
            ),
            header=pn.Row(pn.widgets.StaticText(
                name="Settings Panel",
                value="Please choose the appropriate settings for your data",
                ),
                self.close_settings_button,
            ),
            collapsible=False,
        )

        return self.row


class MenuDashboard(param.Parameterized):

    contents = param.String()

    def __init__(self, main, **params):
        super(MenuDashboard, self).__init__(**params)

        self.row = pn.Row(pn.pane.Str("loading"))

        self.add_plot_button = pn.widgets.Button(name="Add Plot")
        self.add_plot_button.on_click(
            partial(self.update_main_contents, main=main,
                    updated="Plot", button=self.add_plot_button))

        self.add_selected_info_button = pn.widgets.Button(
            name="Add Selected Source Info"
        )
        self.add_selected_info_button.on_click(
            partial(self.update_main_contents,
                    main=main,
                    updated="Selected Info",
                    button=self.add_selected_info_button)
                    )

    def update_main_contents(self, event, main, updated, button):
        print(updated)
        button.name = "Loading..."
        main.set_contents(updated)

    def panel(self):

        self.row[0] = pn.Column(
            self.add_plot_button,
            self.add_selected_info_button,
        )
        return self.row


class PlotDashboard(param.Parameterized):

    X_variable = param.Selector(
        objects=["0"],
        default="0",
        doc="Selection box for the X axis of the plot.")

    Y_variable = param.Selector(
        objects=["1"],
        default="1",
        doc="Selection box for the Y axis of the plot.")

    def __init__(self, src, **params):
        super(PlotDashboard, self).__init__(**params)

        global main_df

        self.df = main_df
        self.update_variable_lists()
        self.src = src
        self.src.on_change("data", self.panel_cb)
        # self.src.selected.on_change("indices", self.panel_cb)
        self.update_variable_lists()
        self.row = pn.Row(pn.pane.Str("loading"))

    def update_variable_lists_cb(self, attr, old, new):
        self.update_variable_lists()

    def update_variable_lists(self):

        print(f"x_var currently is: {self.X_variable}")

        cols = list(self.df.columns)

        if settings["id_col"] in cols:
            cols.remove(settings["id_col"])
        if settings["label_col"] in cols:
            cols.remove(settings["label_col"])

        self.param.X_variable.objects = cols
        self.param.Y_variable.objects = cols
        self.param.X_variable.default = settings["default_vars"][1]
        self.param.Y_variable.default = settings["default_vars"][0]
        self.Y_variable = settings["default_vars"][1]
        self.X_variable = settings["default_vars"][0]

        print(f"len of cols {len(cols)}")
        print(f"cols {cols}")

        print(f"len of cols {len(cols)}")
        print(f"cols {cols}")

        print(f"x_var has now changed to: {self.X_variable}")

    def panel_cb(self, attr, old, new):
        self.panel()

    @ param.depends("X_variable", "Y_variable")
    def plot(self):

        print(self.df.columns)

        p = hv.Points(
            self.df,
            [self.X_variable, self.Y_variable],
        ).opts(active_tools=["pan", "wheel_zoom"])

        print("problem after p")

        keys = list(self.src.data.keys())

        if len(self.src.data[keys[0]]) == 1:
            selected = pd.DataFrame(
                self.src.data, columns=keys, index=[0])
        else:
            selected = pd.DataFrame(columns=keys)

        print(selected)

        selected_plot = hv.Scatter(
            selected,
            self.X_variable,
            self.Y_variable,
        ).opts(
            fill_color="black",
            marker="circle",
            size=10,
            tools=["box_select"],
            active_tools=["pan", "wheel_zoom"],
        )

        color_key = settings["label_colours"]

        color_points = hv.NdOverlay(
            {
                settings['labels_to_strings'][f"{n}"]:
                hv.Points([0, 0],
                          label=settings['labels_to_strings'][f"{n}"]).opts(
                    style=dict(color=color_key[n], size=0)
                )
                for n in color_key
            }
        )

        max_x = np.max(self.df[self.X_variable])
        min_x = np.min(self.df[self.X_variable])

        max_y = np.max(self.df[self.Y_variable])
        min_y = np.min(self.df[self.Y_variable])

        x_sd = np.std(self.df[self.X_variable])
        x_mu = np.mean(self.df[self.X_variable])
        y_sd = np.std(self.df[self.Y_variable])
        y_mu = np.mean(self.df[self.Y_variable])

        max_x = np.min([x_mu + 4*x_sd, max_x])
        min_x = np.max([x_mu - 4*x_sd, min_x])

        max_y = np.min([y_mu + 4*y_sd, max_y])
        min_y = np.max([y_mu - 4*y_sd, min_y])

        print(f"shape: {selected.shape}")

        if selected.shape[0] > 0:

            print(f"shape: {selected.shape}")
            print(
                f"np.max(selected[self.X_variable]): {np.max(selected[self.X_variable])}")
            print(f"max_x: {max_x}")
            max_x = np.max([max_x, np.max(selected[self.X_variable])])
            min_x = np.min([min_x, np.min(selected[self.X_variable])])

            max_y = np.max([max_y, np.max(selected[self.Y_variable])])
            min_y = np.min([min_y, np.min(selected[self.Y_variable])])

        plot = (
            dynspread(
                datashade(
                    p,
                    color_key=color_key,
                    aggregator=ds.by(settings["label_col"], ds.count()),
                ).opts(xlim=(min_x, max_x), ylim=(min_y, max_y), responsive=True),
                threshold=0.75,
                how="saturate",
            )
            * selected_plot
            * color_points
        )

        return plot

    def panel(self):

        print("Panel Plot")

        self.row[0] = pn.Card(
            pn.Row(self.plot, sizing_mode="stretch_both"),
            header=pn.Row(
                pn.Spacer(width=25, sizing_mode="fixed"),
                self.param.X_variable,
                self.param.Y_variable,
                width=200,
                sizing_mode="fixed",
            ),
            collapsible=False,
            sizing_mode="stretch_both",
        )

        return self.row


class DataSelection(param.Parameterized):

    dataset = param.FileSelector(path="data/*.fits")

    ready = param.Boolean(default=False)

    def __init__(self, src, **params):
        super(DataSelection, self).__init__(**params)
        self.src = src

        self.load_data_button = pn.widgets.Button(
            name="Load File", max_height=30, margin=(45, 0, 0, 0)
        )
        self.load_data_button.on_click(self.load_data_cb)

    def get_dataframe_from_fits_file(self, filename):
        fits_table = Table.read(filename, format="fits")
        names = [name for name in fits_table.colnames if len(
            fits_table[name].shape) <= 1]
        df = fits_table[names].to_pandas()

        for col, dtype in df.dtypes.items():
            if dtype == np.object:  # Only process byte object columns.
                df[col] = df[col].apply(lambda x: x.decode("utf-8"))

        return df

    def load_data_cb(self, event):
        self.load_data_button.disabled = True
        self.load_data_button.name = "Loading File..."
        print("loading new dataset")
        global main_df
        main_df = self.get_dataframe_from_fits_file(self.dataset)
        self.df = main_df
        self.src.data = dict(pd.DataFrame())

        print(f" dataset shape: {self.df.shape}")
        print(f"Self.df: {self.df}")
        # FIXME :: Remove update_src
        self.update_src(self.df)
        self.ready = True
        self.load_data_button.name = "File Loaded."

    def get_df(self):
        return self.df

    def update_src(self, df):
        new_df = pd.DataFrame([[0, 0], [0, 0]], columns=["test", "test"])
        self.src.data = dict(new_df)

    def panel(self):
        return pn.Row(self.param.dataset, self.load_data_button, max_width=300)


class ParameterAssignment(param.Parameterized):

    label_column = param.ObjectSelector(objects=["default"], default="default")
    id_column = param.ObjectSelector(objects=["default"], default="default")
    default_x_variable = param.ObjectSelector(
        objects=["default"], default="default")
    default_y_variable = param.ObjectSelector(
        objects=["default"], default="default")

    completed = param.Boolean(
        default=False,
    )

    initial_update = True

    ready = param.Boolean(default=False)

    colours_param = {}

    label_strings_param = {}

    def __init__(self, df, **params):
        super(ParameterAssignment, self).__init__(**params)
        self.column = pn.Column(pn.pane.Str("loading"))
        # FIXME :: Remove src
        self.src = ColumnDataSource()
        # self.src.on_change("data", self.parameter_update_cb)

        self.df = None

        self.confirm_settings_button = pn.widgets.Button(
            name="Confirm Settings", max_height=30, margin=(25, 0, 0, 0)
        )
        self.confirm_settings_button.on_click(self.confirm_settings_cb)

        self.extra_info_selector = pn.widgets.MultiChoice(
            name="Extra Columns to display when inspecting a source:",
            value=[],
            options=[],
        )

    def update_data(self, dataframe=None):

        if dataframe is not None:
            self.df = dataframe

        if (self.initial_update) and (self.df is not None):
            print("Not None")
            cols = list(self.df.columns)

            if cols == []:
                return
            self.initial_update = False

            self.param.id_column.objects = cols
            self.param.id_column.default = cols[0]
            self.id_column = cols[0]

            self.param.label_column.objects = cols
            self.param.label_column.default = cols[0]
            self.label_column = cols[0]

            self.param.default_x_variable.objects = cols
            self.param.default_x_variable.default = cols[0]
            self.default_x_variable = cols[0]

            self.param.default_y_variable.objects = cols
            self.param.default_y_variable.default = cols[1]
            self.default_y_variable = cols[1]

            self.extra_info_selector.options = cols

    @param.depends("label_column", watch=True)
    def update_labels(self):
        self.labels = sorted(self.df[self.label_column].unique())
        settings["labels"] = self.labels

        self.update_colours()
        self.update_label_strings()
        self.panel()

    def update_colours(self):
        print("updating colours...")
        labels = settings["labels"]

        colour_list = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        for i, data_label in enumerate(labels):
            self.colours_param[f"{data_label}"] = pn.widgets.ColorPicker(
                name=f"{data_label}", value=colour_list[(i % 10)]
            )

        print(self.colours_param)

    def update_label_strings(self):
        print("Updating Label Strings...")
        labels = settings["labels"]
        for i, data_label in enumerate(labels):
            self.label_strings_param[f"{data_label}"] = pn.widgets.TextInput(
                name=f"{data_label}", placeholder=f"{data_label}")

    def confirm_settings_cb(self, event):
        print("Saving settings...")
        self.confirm_settings_button.name = "Assigning parameters..."
        self.confirm_settings_button.disabled = True
        global settings
        settings = self.get_settings()

        labels = self.df[settings["label_col"]]
        # print("start colour copy")
        # colours = labels.copy()
        # print("finish copy")
        # print("start colour loop")
        # for key in settings["label_colours"].keys():
        #     colours.loc[colours == key] = settings["label_colours"][key]
        # print("finish loop")

        settings["extra_info_cols"] = self.extra_info_selector.value
        self.confirm_settings_button.name = "Confirmed"
        self.ready = True

    def get_default_variables(self):
        return (self.default_x_variable, self.default_y_variable)

    def get_id_column(self):
        return self.id_column

    def get_label_column(self):
        return self.label_column

    def get_label_colours(self):
        colours = {}

        for key in self.colours_param.keys():
            colours[key] = self.colours_param[key].value

        is_int = True

        for key in colours.keys():
            try:
                i = int(key)
                print(f"{i} is int")
            except:
                is_int = False
                print(f"{key} is not int")
                break

        if is_int:
            new_colours = {}

            for key in colours.keys():
                new_colours[int(key)] = colours[key]

            colours = new_colours

        return colours

    def get_label_strings(self):
        labels_to_strings = {}
        strings_to_labels = {}

        for label in settings["labels"]:
            value = self.label_strings_param[f"{label}"].value
            if value == "":
                value = str(label)
            labels_to_strings[f"{label}"] = value
            strings_to_labels[f"{value}"] = label

        return labels_to_strings, strings_to_labels

    def get_settings(self):

        updated_settings = {}
        updated_settings["id_col"] = self.get_id_column()
        updated_settings["label_col"] = self.get_label_column()
        updated_settings["default_vars"] = self.get_default_variables()
        updated_settings["labels"] = self.labels
        updated_settings["label_colours"] = self.get_label_colours()
        (updated_settings["labels_to_strings"],
         updated_settings["strings_to_labels"]) = self.get_label_strings()

        return updated_settings

    def is_complete(self):
        return self.completed

    @param.depends("completed", watch=True)
    def panel(self):
        print("ASSIGN PARAMETERS IS RENDERING...")
        if self.completed:
            self.column[0] = pn.pane.Str("Settings Saved.")
        else:
            print(self.param.id_column)
            layout = pn.Column(
                pn.Row(
                    self.param.id_column,
                    self.param.label_column,
                    self.param.default_x_variable,
                    self.param.default_y_variable,
                )
            )

            if len(self.colours_param.keys()) > 0:
                colour_row = pn.Row(
                    pn.pane.Markdown("**Choose colours:**", max_height=50)
                )
                for widget in self.colours_param:
                    colour_row.append(self.colours_param[widget])

                label_strings_row = pn.Row(
                    pn.pane.Markdown("**Custom label names:**", max_height=50)
                )

                for widget in self.label_strings_param:
                    label_strings_row.append(self.label_strings_param[widget])

                layout.append(colour_row)

                layout.append(label_strings_row)

                layout.append(self.extra_info_selector)

                layout.append(self.confirm_settings_button)

            self.column[0] = layout

        return self.column


class ActiveLearningSettings(param.Parameterized):
    def __init__(self, src, close_button, **params):
        super(ActiveLearningSettings, self).__init__(**params)

        self.df = None

        self.close_button = close_button

        self.column = pn.Column("Loading")

        self.label_selector = pn.widgets.CrossSelector(
            name="Labels", value=[], options=[], max_height=100
        )

        self.label_selector._search[True].max_height = 20
        self.label_selector._search[False].max_height = 20

        self.feature_selector = pn.widgets.CrossSelector(
            name="Features", value=[], options=[], max_height=100
        )

        self.feature_selector._search[True].max_height = 20
        self.feature_selector._search[False].max_height = 20

        self.confirm_settings_button = pn.widgets.Button(
            name="Confirm Settings")
        self.confirm_settings_button.on_click(self.confirm_settings_cb)

        self.exclude_labels_checkbox = pn.widgets.Checkbox(
            name="Should remaining labels be removed from Active Learning datasets?", value=True
        )

        self.exclude_labels_tooltip = pn.pane.HTML(
            "<span data-toggle='tooltip' title='If enabled, this will remove all instances of labels without a classifier from the Active Learning train, validation and test sets (visualisations outside the AL panel are unaffected). This is useful for labels representing an unknown classification which would not be compatible with scoring functions.' style='border-radius: 15px;padding: 5px; background: #5e5e5e; ' >❔</span> ",
            max_width=5,
        )

        self.scale_features_checkbox = pn.widgets.Checkbox(
            name="Should features be scaled during training?"
        )

        self.scale_features_tooltip = pn.pane.HTML(
            "<span data-toggle='tooltip' title='If enabled, this can improve the performance of your model, however will require you to scale all new data with the produced scaler. This scaler will be saved in your model directory.' style='border-radius: 15px;padding: 5px; background: #5e5e5e; ' >❔</span> ",
            max_width=5,
        )

        self.completed = False

    def update_data(self, dataframe=None):

        if dataframe is not None:
            self.df = dataframe

        if (self.df is not None):

            labels = settings["labels"]
            options = []
            for label in labels:
                options.append(settings["labels_to_strings"][f"{label}"])
            self.label_selector.options = options

            self.feature_selector.options = list(self.df.columns)

    def confirm_settings_cb(self, event):
        print("Saving settings...")
        global settings

        settings["labels_to_train"] = self.label_selector.value
        settings["features_for_training"] = self.feature_selector.value
        settings["exclude_labels"] = self.exclude_labels_checkbox.value

        unclassified_labels = []
        for label in self.label_selector.options:
            if label not in self.label_selector.value:
                unclassified_labels.append(label)

        settings["unclassified_labels"] = unclassified_labels
        settings["scale_data"] = self.scale_features_checkbox.value

        self.completed = True

        self.close_button.disabled = False
        self.close_button.button_type = "success"

        self.panel()

    def get_df(self):
        return self.df

    def is_complete(self):
        return self.completed

    def panel(self):
        print("Rendering AL Settings Panel")
        global settings
        print(settings)

        if self.completed:
            self.column[0] = pn.pane.Str("Settings Saved.")

        else:

            self.column[0] = pn.Column(
                pn.Row(
                    self.label_selector.name, self.feature_selector.name, max_height=30
                ),
                pn.Row(
                    self.label_selector,
                    self.feature_selector,
                    height=200,
                    sizing_mode="stretch_width",
                ),
                pn.Row(self.exclude_labels_checkbox,
                       self.exclude_labels_tooltip),
                pn.Row(self.scale_features_checkbox,
                       self.scale_features_tooltip),
                pn.Row(self.confirm_settings_button, max_height=30),
            )

        return self.column


#########################  ACTIVE LEARNING RENDERING  #########################


class ActiveLearningTab(param.Parameterized):
    def create_results_plot(self, set):
        print("create_results_plot")
        fig = figure(
            plot_width=350,
            plot_height=350,
            output_backend="webgl",
            x_range=[-3, 4],
            y_range=[-2, 2.5],
            x_axis_label=settings["default_vars"][0],
            y_axis_label=settings["default_vars"][1],
            toolbar_location=None,
        )

        fig.circle(settings["default_vars"][0], settings["default_vars"][1])

    def __init__(self, src, df, label, **params):
        super(ActiveLearningTab, self).__init__(**params)
        print("init")

        self.df = df

        # FIXME :: Remove src
        self.src = src
        self.label = settings["strings_to_labels"][label]
        self.label_string = label

        self.last_label = str(label)
        self.current_tab = 0

        self.training = False
        self.assigned_label = False

        self.accuracy_list = {
            "train": {"score": [], "num_points": []},
            "val": {"score": [], "num_points": []},
        }
        self.f1_list = {
            "train": {"score": [], "num_points": []},
            "val": {"score": [], "num_points": []},
        }
        self.precision_list = {
            "train": {"score": [], "num_points": []},
            "val": {"score": [], "num_points": []},
        }
        self.recall_list = {
            "train": {"score": [], "num_points": []},
            "val": {"score": [], "num_points": []},
        }

        self.train_scores = {"acc": 0.00,
                             "prec": 0.00, "rec": 0.00, "f1": 0.00}
        self.val_scores = {"acc": 0.00, "prec": 0.00, "rec": 0.00, "f1": 0.00}

        # FIXME :: Think this will remove evrything but trained features
        self.df, self.data = self.generate_features(self.df)

        self.x, self.y = self.split_x_y(self.data)

        if settings["exclude_labels"]:
            for label in settings["unclassified_labels"]:
                self.x, self.y, _, _ = self.exclude_unclassified_labels(
                    self.x, self.y, label)

        (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        ) = self.train_dev_test_split(self.x, self.y, 0.6, 0.2)

        self.x_cols = self.x_train.columns
        self.y_cols = self.y_train.columns

        if settings["scale_data"]:

            (self.x_train, self.x_val, self.x_test,) = self.scale_data(
                self.x_train,
                self.y_train,
                self.x_val,
                self.y_val,
                self.x_test,
                self.y_test,
                self.x_cols,
                self.y_cols,
            )

        (
            self.y_train,
            self.id_train,
            self.y_val,
            self.id_val,
            self.y_test,
            self.id_test,
        ) = self.split_y_ids(self.y_train, self.y_val, self.y_test)

        self.convert_to_one_vs_rest()

        self.model_output_data_tr = {
            f'{settings["default_vars"][0]}': [],
            f'{settings["default_vars"][1]}': [],
            "metric": [],
            "y": [],
            "pred": [],
        }

        self.model_output_data_val = {
            f'{settings["default_vars"][0]}': [],
            f'{settings["default_vars"][1]}': [],
            "y": [],
            "pred": [],
        }

        # CHANGED :: Make labels dynamic
        options = []
        for label in settings["labels_to_train"]:
            options.append(label)
        options.append("Unsure")
        self.assign_label_group = pn.widgets.RadioButtonGroup(
            name="Label button group",
            options=options,
        )

        self.assign_label_button = pn.widgets.Button(
            name="Assign Label", button_type='primary')
        self.assign_label_button.on_click(self.assign_label_cb)

        self.checkpoint_button = pn.widgets.Button(name="Checkpoint")
        self.checkpoint_button.on_click(self.checkpoint_cb)

        self.show_queried_button = pn.widgets.Button(name="Show Queried")
        self.show_queried_button.on_click(self.show_queried_point)

        self.classifier_dropdown = pn.widgets.Select(
            name="Classifier",
            options=["KNN", "DTree", "RForest", "AdaBoost", "GBTrees"],
        )
        self.query_strategy_dropdown = pn.widgets.Select(
            name="Query Strategy",
            options=["Uncertainty Sampling",
                     "Margin Sampling", "Entropy Sampling"],
        )
        self.starting_num_points = pn.widgets.IntInput(
            name="How many initial points?", value=5, step=1, start=3
        )

        self.classifier_table_source = ColumnDataSource(
            dict(classifier=[], query=[]))
        table_column = [
            TableColumn(field="classifier", title="classifier"),
            TableColumn(field="query", title="query"),
        ]

        self.classifier_table = DataTable(
            source=self.classifier_table_source,
            columns=table_column,
        )

        self.add_classifier_button = pn.widgets.Button(
            name=">>", max_height=40)
        self.remove_classifier_button = pn.widgets.Button(
            name="<<", max_height=40)

        self.add_classifier_button.on_click(self.add_classifier_cb)
        self.remove_classifier_button.on_click(self.remove_classifier_cb)

        self.start_training_button = pn.widgets.Button(name="Start Training")
        self.start_training_button.on_click(self.start_training_cb)

        self.next_iteration_button = pn.widgets.Button(name="Next Iteration")
        self.next_iteration_button.on_click(self.next_iteration_cb)

        self.setup_row = pn.Row("Loading")
        self.panel_row = pn.Row("Loading")

        self.conf_mat_tr_tn = "TN"
        self.conf_mat_tr_fn = "FN"
        self.conf_mat_tr_fp = "FP"
        self.conf_mat_tr_tp = "TP"
        self.conf_mat_val_tn = "TN"
        self.conf_mat_val_fn = "FN"
        self.conf_mat_val_fp = "FP"
        self.conf_mat_val_tp = "TP"

        self.corr_train = ColumnDataSource(self.empty_data())
        self.incorr_train = ColumnDataSource(self.empty_data())
        self.corr_val = ColumnDataSource(self.empty_data())
        self.incorr_val = ColumnDataSource(self.empty_data())
        self.metric_values = ColumnDataSource(
            {
                f'{settings["default_vars"][0]}': [],
                f'{settings["default_vars"][1]}': [],
                "metric": [],
            }
        )
        self.queried_points = ColumnDataSource(self.empty_data())

        self.id_al_train = []

        print(self.corr_train.data)
        print(self.incorr_train.data)
        print(self.metric_values.data)

        x_sd = np.std(self.x_train[settings["default_vars"][0]])
        x_mu = np.mean(self.x_train[settings["default_vars"][0]])
        y_sd = np.std(self.x_train[settings["default_vars"][1]])
        y_mu = np.mean(self.x_train[settings["default_vars"][1]])

        x_max = x_mu + 4*x_sd
        x_min = x_mu - 4*x_sd

        y_max = y_mu + 4*y_sd
        y_min = y_mu - 4*y_sd

        self.max_x = np.min(
            [(x_max), np.max(self.x_train[settings["default_vars"][0]])])
        self.min_x = np.max(
            [(x_min), np.min(self.x_train[settings["default_vars"][0]])])

        self.max_y = np.min(
            [(y_max), np.max(self.x_train[settings["default_vars"][1]])])
        self.min_y = np.max(
            [(y_min), np.min(self.x_train[settings["default_vars"][1]])])

    def remove_from_pool(self):
        print("remove_from_pool")

        self.x_pool = np.delete(
            self.x_pool,
            self.query_index,
            0,
        )

        self.y_pool = np.delete(
            self.y_pool,
            self.query_index,
            0,
        )

        self.id_pool = self.id_pool.drop(self.id_pool.index[self.query_index])

    def save_model(self, event=None, checkpoint=False):

        list_c1 = self.classifier_table_source.data["classifier"]
        list_c2 = self.classifier_table_source.data["query"]

        clfs_shortened = ""

        for i in range(len(list(list_c1))):
            clf = f"{list_c1[i][:4]}_{list_c2[i][:4]}_"
            clfs_shortened += clf

        clfs_shortened = clfs_shortened[:-1]
        iteration = self.curr_num_points
        val_f1 = int(float(self.val_scores["f1"]) * 100)

        dir = "models/"
        filename = f"{dir}{self.label}-{clfs_shortened}"

        if self.committee:
            now = datetime.now()
            dt_string = now.strftime("%Y%m-%d_%H:%M:%S")
            for i, clf in enumerate(self.learner):
                model = clf
                print(clf.__class__.__name__)
                if checkpoint:
                    mod_dir = f"{filename}-{iteration}-{val_f1}-{dt_string}"
                    if not os.path.isdir(mod_dir):
                        os.mkdir(mod_dir)
                    scaler_dir = f"{mod_dir}/SCALER"
                    mod_dir = f"{mod_dir}/{list_c1[i][:6]}_{i}"
                    dump(model, f"{mod_dir}.joblib")

                    if (not os.path.isfile(f"{scaler_dir}.joblib") and (settings["scale_data"])):
                        dump(self.scaler, f"{scaler_dir}.joblib")

                else:
                    if not os.path.isdir(filename):
                        os.mkdir(filename)
                    dump(model, f"{filename}/{list_c1[i][:6]}_{i}.joblib")
                    scaler_dir = f"{filename}/SCALER"
                    if (not os.path.isfile(f"{scaler_dir}.joblib") and (settings["scale_data"])):
                        dump(self.scaler, f"{scaler_dir}.joblib")

        else:
            model = self.learner.estimator
            if checkpoint:
                now = datetime.now()
                dt_string = now.strftime("%Y%m-%d_%H:%M:%S")
                dump(
                    model, f"{filename}-{iteration}-{val_f1}-{dt_string}.joblib")
                scaler_dir = f"{filename}-{iteration}-{val_f1}-{dt_string}-SCALER"
                if (not os.path.isfile(f"{scaler_dir}.joblib") and (settings["scale_data"])):
                    dump(self.scaler, f"{scaler_dir}.joblib")
            else:
                dump(model, f"{filename}.joblib")
                scaler_dir = f"{filename}-SCALER"
                if (not os.path.isfile(f"{scaler_dir}.joblib") and (settings["scale_data"])):
                    dump(self.scaler, f"{scaler_dir}.joblib")

    def checkpoint_cb(self, event):

        self.checkpoint_button.disabled = True
        self.checkpoint_button.name = "Saved!"
        self.save_model(checkpoint=True)

    def show_queried_point(self, event=None):

        print("show_queried_point")
        start = time.time()
        query_instance = self.query_instance
        query_idx = self.query_index
        end = time.time()
        print(f"queries {end - start}")

        # FIXME :: Change to df
        data = self.df

        start = time.time()
        queried_id = self.id_pool.iloc[query_idx][settings["id_col"]]
        end = time.time()
        print(f"queried_id {end - start}")

        print(f"\n\n\n id is {queried_id.values} \n\n\n")
        start = time.time()
        sel_idx = np.where(
            data[f'{settings["id_col"]}'] == queried_id.values[0])
        end = time.time()
        print(f"np.where {end - start}")
        act_label = self.y_pool[query_idx]
        print(f"Should be a {act_label}")
        # FIXME :: Change how selected
        selected_source = self.df[self.df[settings["id_col"]]
                                  == queried_id.values[0]]
        selected_dict = selected_source.set_index(
            settings["id_col"]).to_dict('list')
        self.src.data = selected_dict

        print(f"\n\n\n src.data: {self.src.data} \n id:{queried_id.values[0]}")

        start = time.time()
        # self.src.data = dict(self.src.data)
        end = time.time()
        print(f"dict data {end - start}")
        plot_idx = [
            list(self.df.columns).index(settings["default_vars"][0]),
            list(self.df.columns).index(settings["default_vars"][1]),
        ]

        q = {
            f'{settings["default_vars"][0]}': query_instance[:, plot_idx[0]],
            f'{settings["default_vars"][1]}': [query_instance[:, plot_idx[1]]],
        }

        self.queried_points.data = q

    def iterate_AL(self):
        print("iterate al")

        # self.assign_label = False

        print("remove_from_pool")
        self.remove_from_pool()

        self.curr_num_points = self.x_al_train.shape[0]

        print("fitting")
        self.learner.fit(self.x_al_train, self.y_al_train)
        print("fitted")

        self.save_model(checkpoint=False)

        print("getting predictions")
        self.get_predictions()
        print("got predictions")

        self.query_new_point()

        start = time.time()
        self.show_queried_point()
        end = time.time()
        print(f"show_queried_point {end - start}")

        self.assign_label_button.name = "Assign"

    def query_new_point(self):

        self.query_index, self.query_instance = self.learner.query(self.x_pool)
        print("queried")

    def assign_label_cb(self, event):

        selected_label = self.assign_label_group.value
        self.last_label = selected_label

        if not selected_label == "Unsure":

            self.assigned_label = True
            self.next_iteration_button.name = "Next Iteration"

            if self.assign_label_button.name == "Assigned!":
                self.panel()
                return

            self.assign_label_button.name = "Assigned!"

            if int(settings["strings_to_labels"][selected_label]) == self.label:
                selected_label = 1
            else:
                selected_label = 0

            selected_label = np.array([selected_label])

            query = self.query_instance
            query_idx = self.query_index

            new_train = np.vstack((self.x_al_train, query))

            new_label = np.concatenate((self.y_al_train, selected_label))

            new_id = self.id_al_train.append(self.id_pool.iloc[query_idx])

            print(new_id)

            self.x_al_train = new_train
            self.y_al_train = new_label
            self.id_al_train = new_id

        else:
            self.assign_label_button.name = "Querying..."
            self.assign_label_button.disabled = True
            self.remove_from_pool()
            self.query_new_point()
            self.show_queried_point()
            self.assign_label_button.name = "Assign"
            self.assign_label_button.disabled = False

        assert (
            self.x_al_train.shape[0] == self.y_al_train.shape[0]
        ), "AL_TRAIN & LABELS NOT EQUAL"

        assert (
            self.y_al_train.shape[0] == self.id_al_train.shape[0]
        ), "AL_LABELS & IDs NOT EQUAL"

        self.panel()

    def empty_data(self):

        print("empty_data")

        empty = {
            f'{settings["default_vars"][0]}': [],
            f'{settings["default_vars"][1]}': [],
        }

        return empty

    def next_iteration_cb(self, event):
        print("next_iter_cb")

        if self.next_iteration_button.name == "Training...":
            return

        self.next_iteration_button.name = "Training..."

        self.iterate_AL()

        self.checkpoint_button.name = "Checkpoint"
        self.checkpoint_button.disabled = False
        self.assigned_label = False

        self.panel()

    def start_training_cb(self, event):
        print("start_training_cb")

        self.training = True
        self.num_points_list = []
        self.curr_num_points = self.starting_num_points.value

        self.panel()
        self.setup_learners()
        query_idx, query_instance = self.learner.query(self.x_pool)

        self.query_instance = query_instance
        self.query_index = query_idx

        self.show_queried_point()

        self.panel()

    def add_classifier_cb(self, event):
        print("add_classifier_cb")

        clf = self.classifier_dropdown.value
        qs = self.query_strategy_dropdown.value
        list_c1 = self.classifier_table_source.data["classifier"]
        list_c2 = self.classifier_table_source.data["query"]

        list_c1.append(clf)
        list_c2.append(qs)

        self.classifier_table_source.data = {
            "classifier": list_c1,
            "query": list_c2,
        }

    def remove_classifier_cb(self, event):

        print("remove_classifier_cb")

        list_c1 = self.classifier_table_source.data["classifier"]
        list_c2 = self.classifier_table_source.data["query"]

        list_c1 = list_c1[:-1]
        list_c2 = list_c2[:-1]

        self.classifier_table_source.data = {
            "classifier": list_c1,
            "query": list_c2,
        }

    def split_x_y(self, df_data):

        print("split_x_y")

        # print(df_data)

        df_data_y = df_data[[settings["label_col"], settings["id_col"]]]
        df_data_x = df_data.drop(
            columns=[settings["label_col"], settings["id_col"]])
        assert (
            df_data_y.shape[0] == df_data_x.shape[0]
        ), f"df_data_y has different number of rows than df_data_x, {df_data_y.shape[0]} != {df_data_x.shape[0]}"

        return df_data_x, df_data_y

    def exclude_unclassified_labels(self, df_data_x, df_data_y, excluded):
        excluded_label = settings["strings_to_labels"][excluded]
        unknowns_x = df_data_x[df_data_y[settings["label_col"]]
                               == excluded_label]
        unknowns_y = df_data_y[df_data_y[settings["label_col"]]
                               == excluded_label]

        data_x = df_data_x[df_data_y[settings["label_col"]] != excluded_label]
        data_y = df_data_y[df_data_y[settings["label_col"]] != excluded_label]

        return data_x, data_y, unknowns_x, unknowns_y

    def train_dev_test_split(self, df_data_x, df_data_y, train_ratio, val_ratio):

        print("train_dev_split")

        # print(df_data_x)
        np.random.seed(0)
        rng = np.random.RandomState(seed=0)

        test_ratio = 1 - train_ratio - val_ratio
        x_train, x_temp, y_train, y_temp = train_test_split(
            df_data_x,
            df_data_y,
            test_size=1 - train_ratio,
            stratify=df_data_y[settings["label_col"]],
            random_state=rng,
        )

        print()

        print("Expected:")
        for i in settings["labels_to_train"]:
            i_raw = settings["strings_to_labels"][i]
            current = df_data_y[df_data_y[settings["label_col"]] == i_raw]
            print(
                f"{i}    {y_train[settings['label_col']].value_counts()[i_raw] - current[settings['label_col']].value_counts()[i_raw] * train_ratio}"
            )

        print("\n")

        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=test_ratio / (test_ratio + val_ratio),
            stratify=y_temp[settings["label_col"]],
            random_state=rng,
        )

        print()

        print("Expected:")
        for i in settings["labels_to_train"]:
            i_raw = settings["strings_to_labels"][i]
            current = df_data_y[df_data_y[settings["label_col"]] == i_raw]
            print(
                f"{i}    {y_val[settings['label_col']].value_counts()[i_raw]- current[settings['label_col']].value_counts()[i_raw] * val_ratio}"
            )

        print("\n")

        print("Expected:")
        for i in settings["labels_to_train"]:
            i_raw = settings["strings_to_labels"][i]
            current = df_data_y[df_data_y[settings["label_col"]] == i_raw]
            print(
                f"{i} {y_test[settings['label_col']].value_counts()[i_raw] - current[settings['label_col']].value_counts()[i_raw] * test_ratio}"
            )

        print(f"train: {y_train[settings['label_col']].value_counts()}")
        print(f"val: {y_val[settings['label_col']].value_counts()}")
        print(f"test: {y_test[settings['label_col']].value_counts()}")

        return x_train, y_train, x_val, y_val, x_test, y_test

    def scale_data(
        self, x_train, y_train, x_val, y_val, x_test, y_test, x_cols, y_cols
    ):

        print("scale_data")

        self.scaler = RobustScaler()
        data_x_tr = self.scaler.fit_transform(x_train)

        data_x_val = self.scaler.transform(x_val)
        data_x_test = self.scaler.transform(x_test)

        data_x_tr = pd.DataFrame(data_x_tr, columns=x_cols)

        data_x_val = pd.DataFrame(data_x_val, columns=x_cols)

        data_x_test = pd.DataFrame(data_x_test, columns=x_cols)

        return (
            data_x_tr,
            data_x_val,
            data_x_test,
        )

    def split_y_ids(self, y_train, y_val, y_test):

        print("split_y_ids")

        data_y_tr = pd.DataFrame(
            y_train[settings["label_col"]], columns=[settings["label_col"]]
        )
        data_id_tr = pd.DataFrame(
            y_train[settings["id_col"]], columns=[settings["id_col"]]
        )
        data_y_val = pd.DataFrame(
            y_val[settings["label_col"]], columns=[settings["label_col"]]
        )
        data_id_val = pd.DataFrame(
            y_val[settings["id_col"]], columns=[settings["id_col"]]
        )
        data_y_test = pd.DataFrame(
            y_test[settings["label_col"]], columns=[settings["label_col"]]
        )
        data_id_test = pd.DataFrame(
            y_test[settings["id_col"]], columns=[settings["id_col"]]
        )

        return data_y_tr, data_id_tr, data_y_val, data_id_val, data_y_test, data_id_test

    def convert_to_one_vs_rest(self):

        y_tr = self.y_train.copy()
        y_val = self.y_val.copy()
        y_test = self.y_test.copy()

        is_label = y_tr[settings["label_col"]] == self.label
        isnt_label = y_tr[settings["label_col"]] != self.label

        # y_tr.loc[isnt_label, settings["label_col"]] = 1000000
        y_tr.loc[is_label, settings["label_col"]] = 1
        y_tr.loc[isnt_label, settings["label_col"]] = 0

        is_label = y_val[settings["label_col"]] == self.label
        isnt_label = y_val[settings["label_col"]] != self.label

        y_val.loc[isnt_label, settings["label_col"]] = 1000000
        y_val.loc[is_label, settings["label_col"]] = 1
        y_val.loc[isnt_label, settings["label_col"]] = 0

        is_label = y_test[settings["label_col"]] == self.label
        isnt_label = y_test[settings["label_col"]] != self.label

        y_test.loc[isnt_label, settings["label_col"]] = 1000000
        y_test.loc[is_label, settings["label_col"]] = 1
        y_test.loc[isnt_label, settings["label_col"]] = 0

        self.y_train = y_tr
        self.y_val = y_val
        self.y_test = y_test

    def get_blank_classifiers(self):

        classifiers = {
            "KNN": KNeighborsClassifier(3, n_jobs=-1),
            "DTree": DecisionTreeClassifier(
                random_state=0,
            ),
            "RForest": RandomForestClassifier(
                n_jobs=-1, random_state=0, n_estimators=1000
            ),
            "AdaBoost": AdaBoostClassifier(random_state=0, n_estimators=500),
            "GBTrees": GradientBoostingClassifier(random_state=0, n_estimators=1000),
        }

        return classifiers

    def get_predictions(self, first=False):

        print("get_predicitions")

        tr_pred = self.learner.predict(self.x_train).reshape((-1, 1))

        temp = self.y_train.to_numpy().reshape((-1, 1))

        is_correct = tr_pred == temp

        g_j = self.x_train[settings["default_vars"]
                           [0]].to_numpy().reshape((-1, 1))
        y_w1 = self.x_train[settings["default_vars"]
                            [1]].to_numpy().reshape((-1, 1))

        corr_data = {
            f'{settings["default_vars"][0]}': g_j[is_correct],
            f'{settings["default_vars"][1]}': y_w1[is_correct],
        }
        incorr_data = {
            f'{settings["default_vars"][0]}': g_j[~is_correct],
            f'{settings["default_vars"][1]}': y_w1[~is_correct],
        }

        self.corr_train.data = corr_data
        self.incorr_train.data = incorr_data

        curr_tr_acc = self.learner.score(self.x_train, self.y_train)
        curr_tr_f1 = f1_score(self.y_train, tr_pred)
        curr_tr_prec = precision_score(self.y_train, tr_pred)
        curr_tr_rec = recall_score(self.y_train, tr_pred)

        self.train_scores = {
            "acc": "%.3f" % round(curr_tr_acc, 3),
            "prec": "%.3f" % round(curr_tr_prec, 3),
            "rec": "%.3f" % round(curr_tr_rec, 3),
            "f1": "%.3f" % round(curr_tr_f1, 3),
        }

        print(self.train_scores)

        self.accuracy_list["train"]["score"].append(curr_tr_acc)
        self.f1_list["train"]["score"].append(curr_tr_f1)
        self.precision_list["train"]["score"].append(curr_tr_prec)
        self.recall_list["train"]["score"].append(curr_tr_rec)

        t_conf = confusion_matrix(self.y_train, tr_pred)

        val_pred = self.learner.predict(self.x_val).reshape((-1, 1))

        temp = self.y_val.to_numpy().reshape((-1, 1))

        is_correct = val_pred == temp

        g_j = self.x_val[settings["default_vars"]
                         [0]].to_numpy().reshape((-1, 1))
        y_w1 = self.x_val[settings["default_vars"]
                          [1]].to_numpy().reshape((-1, 1))

        corr_data = {
            f'{settings["default_vars"][0]}': g_j[is_correct],
            f'{settings["default_vars"][1]}': y_w1[is_correct],
        }
        incorr_data = {
            f'{settings["default_vars"][0]}': g_j[~is_correct],
            f'{settings["default_vars"][1]}': y_w1[~is_correct],
        }

        self.corr_val.data = corr_data
        self.incorr_val.data = incorr_data

        curr_val_acc = self.learner.score(self.x_val, self.y_val)
        curr_val_f1 = f1_score(self.y_val, val_pred)
        curr_val_prec = precision_score(self.y_val, val_pred)
        curr_val_rec = recall_score(self.y_val, val_pred)

        self.val_scores = {
            "acc": "%.3f" % round(curr_val_acc, 3),
            "prec": "%.3f" % round(curr_val_prec, 3),
            "rec": "%.3f" % round(curr_val_rec, 3),
            "f1": "%.3f" % round(curr_val_f1, 3),
        }
        self.accuracy_list["val"]["score"].append(curr_val_acc)
        print(f'APPENDED VAR - {self.accuracy_list["val"]["score"]}')
        self.f1_list["val"]["score"].append(curr_val_f1)
        self.precision_list["val"]["score"].append(curr_val_prec)
        self.recall_list["val"]["score"].append(curr_val_rec)

        v_conf = confusion_matrix(self.y_val, val_pred)

        self.num_points_list.append(self.curr_num_points)

        self.accuracy_list["train"]["num_points"] = self.num_points_list
        self.f1_list["train"]["num_points"] = self.num_points_list
        self.precision_list["train"]["num_points"] = self.num_points_list
        self.recall_list["train"]["num_points"] = self.num_points_list
        self.accuracy_list["val"]["num_points"] = self.num_points_list
        self.f1_list["val"]["num_points"] = self.num_points_list
        self.precision_list["val"]["num_points"] = self.num_points_list
        self.recall_list["val"]["num_points"] = self.num_points_list

        print(self.y_val.shape)
        print(val_pred.shape)

        self.conf_mat_tr_tn = str(t_conf[0][0])
        self.conf_mat_tr_fp = str(t_conf[0][1])
        self.conf_mat_tr_fn = str(t_conf[1][0])
        self.conf_mat_tr_tp = str(t_conf[1][1])

        self.conf_mat_val_tn = str(v_conf[0][0])
        self.conf_mat_val_fp = str(v_conf[0][1])
        self.conf_mat_val_fn = str(v_conf[1][0])
        self.conf_mat_val_tp = str(v_conf[1][1])

        proba = self.learner.predict_proba(self.x_train)

        print(proba.shape)

        proba = 1 - np.max(proba, axis=1)

        x_axis = self.x_train[settings["default_vars"][0]].to_numpy()
        y_axis = self.x_train[settings["default_vars"][1]].to_numpy()

        print(f"tr_pred:{tr_pred.shape}")
        print(f"y_train:{self.y_train.shape}")
        print(f"metric:{proba.shape}")

        self.model_output_data_tr[settings["default_vars"][0]] = x_axis
        self.model_output_data_tr[settings["default_vars"][1]] = y_axis
        self.model_output_data_tr["pred"] = tr_pred.flatten()
        self.model_output_data_tr["y"] = self.y_train.to_numpy().flatten()
        self.model_output_data_tr["metric"] = proba

        x_axis = self.x_val[settings["default_vars"][0]].to_numpy()
        y_axis = self.x_val[settings["default_vars"][1]].to_numpy()

        self.model_output_data_val[settings["default_vars"][0]] = x_axis
        self.model_output_data_val[settings["default_vars"][1]] = y_axis
        self.model_output_data_val["pred"] = val_pred.flatten()
        self.model_output_data_val["y"] = self.y_val.to_numpy().flatten()

        # print(self.model_output_data_tr)

    def create_pool(self):

        print("create_pool")

        np.random.seed(0)

        initial_points = int(self.starting_num_points.value)

        if initial_points >= len(self.x_train.index):
            self.starting_num_points.value = len(self.x_train.index)

            initial_points = len(self.x_train.index)

        y_tr = self.y_train.copy()

        # y_tr = y_tr.to_numpy()

        X_pool = self.x_train.to_numpy()
        y_pool = self.y_train.to_numpy().ravel()
        id_pool = self.id_train.to_numpy()

        print(X_pool.shape)

        train_idx = list(
            np.random.choice(
                range(X_pool.shape[0]), size=initial_points - 2, replace=False
            )
        )
        print(f"y_tr = {y_tr}")
        print(f"y_tr==0: {np.where(y_tr == 0)}")
        print(f"y_tr==1: {np.where(y_tr == 1)}")
        c0 = np.random.choice(np.where(y_tr == 0)[0])
        c1 = np.random.choice(np.where(y_tr == 1)[0])

        train_idx = train_idx + [c0] + [c1]

        X_train = X_pool[train_idx]
        y_train = y_pool[train_idx]
        id_train = self.id_train.iloc[train_idx]

        print(y_train)
        print(id_train)

        X_pool = np.delete(X_pool, train_idx, axis=0)
        y_pool = np.delete(y_pool, train_idx)
        id_pool = self.id_train.drop(self.id_train.index[train_idx])

        print(X_train.shape)
        print(X_pool.shape)

        self.x_pool = X_pool
        self.y_pool = y_pool
        self.id_pool = id_pool
        self.x_al_train = X_train
        self.y_al_train = y_train
        self.id_al_train = id_train

    def setup_learners(self):

        print("setup_learners")

        table = self.classifier_table_source.data

        if len(table["classifier"]) == 0:
            return

        # TODO :: Move this
        qs_dict = {
            "Uncertainty Sampling": uncertainty_sampling,
            "Margin Sampling": margin_sampling,
            "Entropy Sampling": entropy_sampling,
        }

        classifier_dict = self.get_blank_classifiers()

        self.create_pool()

        if len(table["classifier"]) == 1:
            self.committee = False
            learner = ActiveLearner(
                estimator=clone(classifier_dict[table["classifier"][0]]),
                query_strategy=qs_dict[table["query"][0]],
                X_training=self.x_al_train,
                y_training=self.y_al_train,
            )

            self.learner = learner

        else:
            learners = []
            self.committee = True
            for i in range(len(table["classifier"])):
                learner = ActiveLearner(
                    estimator=clone(classifier_dict[table["classifier"][i]]),
                    query_strategy=qs_dict[table["query"][i]],
                    X_training=self.x_al_train,
                    y_training=self.y_al_train,
                )
                learners.append(learner)

            self.learner = Committee(learner_list=learners)

        self.get_predictions(first=True)

    def panel_cb(self, attr, old, new):
        print("panel_cb")
        self.panel()

    # TODO :: Add bool to see if user wants this step
    def generate_features(self, df):
        print("generate_features")
        np.random.seed(0)

        # CHANGED :: Change this to selected["AL_Features"]
        bands = settings["features_for_training"]

        print(bands)

        features = bands + [settings["label_col"], settings["id_col"]]
        print(features[-5:])
        df_data = df[features]

        shuffled = np.random.permutation(list(df_data.index.values))

        print("shuffled...")

        df_data = df_data.reindex(shuffled)

        print("reindexed")

        df_data = df_data.reset_index()

        print("reset index")

        combs = list(combinations(bands, 2))

        print(combs)

        cols = list(df.columns)

        updated_columns = False
        for i, j in combs:
            df_data[f"{i}-{j}"] = df_data[i] - df_data[j]
            # print(df_data[f"{i}-{j}"])
            if f"{i}-{j}" not in cols:
                df[f"{i}-{j}"] = df[i] - df[j]

        print("Feature generations complete")

        return df, df_data

    # CHANGED :: Remove static declarations
    def combine_data(self):

        print("combine_data")
        data = np.array(
            self.model_output_data_tr[settings["default_vars"][0]]).reshape((-1, 1))
        print(data)
        print(data.shape)
        data = np.concatenate(
            (data, np.array(self.model_output_data_tr[settings["default_vars"][1]]).reshape((-1, 1))), axis=1
        )
        data = np.concatenate(
            (data, np.array(
                self.model_output_data_tr["metric"]).reshape((-1, 1))),
            axis=1,
        )
        data = np.concatenate(
            (data, np.array(self.model_output_data_tr["y"]).reshape((-1, 1))), axis=1
        )
        data = np.concatenate(
            (data, np.array(self.model_output_data_tr["pred"]).reshape((-1, 1))), axis=1
        )
        data = np.concatenate(
            (data, np.array(self.model_output_data_tr["acc"]).reshape((-1, 1))), axis=1
        )

        print(list(self.model_output_data_tr.keys()))

        data = pd.DataFrame(data, columns=list(
            self.model_output_data_tr.keys()))

        return data

    def train_tab(self):
        print("train_tab")
        start = time.time()
        self.model_output_data_tr["acc"] = np.equal(
            np.array(self.model_output_data_tr["pred"]),
            np.array(self.model_output_data_tr["y"]),
        )
        end = time.time()
        print(f"equal {end - start}")

        start = time.time()

        df = pd.DataFrame(
            self.model_output_data_tr, columns=list(
                self.model_output_data_tr.keys())
        )

        end = time.time()
        print(f"DF {end - start}")

        start = time.time()
        p = hv.Points(
            df,
            [settings["default_vars"][0], settings["default_vars"][1]],
        ).opts(toolbar=None, default_tools=[])
        end = time.time()
        print(f"P {end - start}")

        if hasattr(self, "x_al_train"):
            start = time.time()
            x_al_train = pd.DataFrame(
                self.x_al_train, columns=self.x_train.columns)
            end = time.time()
            print(f"hasattr {end - start}")
        else:
            start = time.time()
            x_al_train = self.empty_data()
            end = time.time()
            print(f"x_al_train {end - start}")

        start = time.time()
        x_al_train_plot = hv.Scatter(
            x_al_train,
            settings["default_vars"][0],
            settings["default_vars"][1],
            # sizing_mode="stretch_width",
        ).opts(
            fill_color="black",
            marker="circle",
            size=10, toolbar=None, default_tools=[]
        )
        end = time.time()
        print(f"x_al_train_plot {end - start}")

        if hasattr(self, "query_instance"):
            start = time.time()
            # FIXME :: Useful
            query_point = pd.DataFrame(
                self.query_instance, columns=self.x_train.columns
            )
            end = time.time()
            print(f"query_point {end - start}")
        else:
            start = time.time()
            query_point = self.empty_data()
            end = time.time()
            print(f"query_point {end - start}")

        start = time.time()
        query_point_plot = hv.Scatter(
            query_point,
            settings["default_vars"][0],
            settings["default_vars"][1],
            # sizing_mode="stretch_width",
        ).opts(
            fill_color="yellow",
            marker="circle",
            size=10, toolbar=None, default_tools=[]
        )
        end = time.time()
        print(f"query_point_plot {end - start}")
        # print(self.model_output_data_tr)
        color_key = {1: "#2eb800", 0: "#c20000", "q": "yellow", "t": "black"}
        # label_key = {1: "Correct", 0: "Incorrect",
        #              "q": "Queried", "t": "Trained"}

        # color_points = hv.NdOverlay(
        #     {
        #         label_key[n]: hv.Points([0, 0], label=label_key[n]).opts(
        #             style=dict(color=color_key[n], size=0, fontsize={"legend": 5}))
        #         for n in color_key
        #     }
        # )

        if len(x_al_train[settings["default_vars"][0]]) > 0:

            max_x_temp = np.max(
                [np.max(query_point[settings["default_vars"][0]]),
                 np.max(x_al_train[settings["default_vars"][0]])]
                )

            max_x = np.max([self.max_x, max_x_temp])

            min_x_temp = np.min(
                [np.min(query_point[settings["default_vars"][0]]),
                 np.min(x_al_train[settings["default_vars"][0]])]
                )

            min_x = np.min([self.min_x, min_x_temp])

            max_y_temp = np.max(
                [np.max(query_point[settings["default_vars"][1]]),
                 np.max(x_al_train[settings["default_vars"][1]])]
                )

            max_y = np.max([self.max_y, max_y_temp])

            min_y_temp = np.min(
                [np.min(query_point[settings["default_vars"][1]]),
                 np.min(x_al_train[settings["default_vars"][1]])]
                )

            min_y = np.min([self.min_y, min_y_temp])
        else:
            max_x, min_x, max_y, min_y = (
                self.max_x, self.min_x, self.max_y, self.min_y)

        start = time.time()
        plot = (
            dynspread(
                # datashade(p).opts(responsive=True),
                datashade(
                    p,
                    color_key=color_key,
                    aggregator=ds.by("acc", ds.count()),
                ).opts(xlim=(min_x, max_x), ylim=(min_y, max_y), responsive=True, shared_axes=False, toolbar=None, default_tools=[]),
                threshold=0.75,
                how="saturate",
            )
        )
        end = time.time()
        print(f"plot {end - start}")
        full_plot = (plot * x_al_train_plot
                     * query_point_plot).opts(toolbar=None, default_tools=[])  # * color_points

        return full_plot

    def val_tab(self):
        print("val_tab")
        start = time.time()
        self.model_output_data_val["acc"] = np.equal(
            np.array(self.model_output_data_val["pred"]),
            np.array(self.model_output_data_val["y"]),
        )
        end = time.time()
        print(f"output_data_val {end - start}")
        start = time.time()

        df = pd.DataFrame(
            self.model_output_data_val, columns=list(
                self.model_output_data_val.keys())
        )

        end = time.time()
        print(f"df-val {end - start}")
        start = time.time()
        p = hv.Points(
            df,
            [settings["default_vars"][0], settings["default_vars"][1]],
        ).opts(active_tools=["pan", "wheel_zoom"])
        end = time.time()
        print(f"p-val {end - start}")
        # print(self.model_output_data_tr)

        color_key = {1: "#2eb800", 0: "#c20000"}
        start = time.time()
        plot = dynspread(
            # datashade(p).opts(responsive=True),
            datashade(
                p,
                color_key=color_key,
                aggregator=ds.by("acc", ds.count()),
            ).opts(xlim=(self.min_x, self.max_x),
                   ylim=(self.min_y, self.max_y),
                   shared_axes=False, responsive=True,
                   active_tools=["pan", "wheel_zoom"]),
            threshold=0.75,
            how="saturate",
        )
        end = time.time()
        print(f"plot-val {end - start}")
        return plot

    def metric_tab(self):

        print("metric_tab")
        start = time.time()

        if "acc" not in self.model_output_data_tr.keys():
            self.model_output_data_tr["acc"] = np.equal(
                np.array(self.model_output_data_tr["pred"]),
                np.array(self.model_output_data_tr["y"]),
            )

        elif len(self.model_output_data_tr["acc"]) == 0:
            self.model_output_data_tr["acc"] = np.equal(
                np.array(self.model_output_data_tr["pred"]),
                np.array(self.model_output_data_tr["y"]),
            )

        for i in self.model_output_data_tr.keys():
            print(f"{i}:{len(self.model_output_data_tr[i])}")

        df = pd.DataFrame(
            self.model_output_data_tr, columns=list(
                self.model_output_data_tr.keys())
        )

        end = time.time()
        print(f"df {end - start}")
        # print(df.columns)

        start = time.time()
        p = hv.Points(
            df, [settings["default_vars"][0], settings["default_vars"][1]]
        ).opts(active_tools=["pan", "wheel_zoom"])

        end = time.time()
        print(f"p {end - start}")
        # print(self.model_output_data_tr)

        start = time.time()

        plot = dynspread(
            # datashade(p).opts(responsive=True),
            datashade(
                p, cmap="RdYlGn_r", aggregator=ds.max("metric"),
                normalization="linear"
            ).opts(active_tools=["pan", "wheel_zoom"],
                   xlim=(self.min_x, self.max_x),
                   ylim=(self.min_y, self.max_y),
                   shared_axes=False, responsive=True),
            threshold=0.75,
            how="saturate",
        )

        end = time.time()
        print(f"plot {end - start}")
        return plot

    def scores_tab(self):
        print("scores_tab")

        print(f'accnum_points{len(self.accuracy_list["train"]["num_points"])}')
        print(f'accscore{len(self.accuracy_list["train"]["score"])}')
        print(f'recnum_points{len(self.recall_list["train"]["num_points"])}')
        print(f'recscore{len(self.recall_list["train"]["score"])}')
        print(
            f'precnum_points{len(self.precision_list["train"]["num_points"])}')
        print(f'precscore{len(self.precision_list["train"]["score"])}')
        print(f'f1num_points{len(self.f1_list["train"]["num_points"])}')
        print(f'f1score{len(self.f1_list["train"]["score"])}')

        return (
            hv.Path(self.accuracy_list["train"], ["num_points", "score"])
            * hv.Path(self.recall_list["train"], ["num_points", "score"])
            * hv.Path(self.precision_list["train"], ["num_points", "score"])
            * hv.Path(self.f1_list["train"], ["num_points", "score"])
        )

    def add_conf_matrices(self):
        print("add_conf_matrices")
        return pn.Column(
            pn.pane.Markdown("Training Set:", sizing_mode="fixed"),
            pn.pane.Markdown(
                f"Acc: {self.train_scores['acc']}, Prec: {self.train_scores['prec']}, Rec: {self.train_scores['rec']}, F1: {self.train_scores['f1']}",
                sizing_mode="fixed",
            ),
            pn.Row(
                pn.Column(
                    pn.Row("", max_height=30),
                    pn.Row("Actual 0", min_height=50),
                    pn.Row("Actual 1", min_height=50),
                ),
                pn.Column(
                    pn.Row("Predicted 0", max_height=30),
                    pn.Row(pn.pane.Str(self.conf_mat_tr_tn), min_height=50),
                    pn.Row(pn.pane.Str(self.conf_mat_tr_fn), min_height=50),
                ),
                pn.Column(
                    pn.Row("Predicted 1", max_height=30),
                    pn.Row(pn.pane.Str(self.conf_mat_tr_fp), min_height=50),
                    pn.Row(pn.pane.Str(self.conf_mat_tr_tp), min_height=50),
                ),
            ),
            pn.pane.Markdown("Validation Set:", sizing_mode="fixed"),
            pn.pane.Markdown(
                f"Acc: {self.val_scores['acc']}, Prec: {self.val_scores['prec']}, Rec: {self.val_scores['rec']}, F1: {self.val_scores['f1']}",
                sizing_mode="fixed",
            ),
            pn.Row(
                pn.Column(
                    pn.Row("", max_height=30),
                    pn.Row("Actual 0", min_height=50),
                    pn.Row("Actual 1", min_height=50),
                ),
                pn.Column(
                    pn.Row("Predicted 0", max_height=30),
                    pn.Row(pn.pane.Str(self.conf_mat_val_tn), min_height=50),
                    pn.Row(pn.pane.Str(self.conf_mat_val_fn), min_height=50),
                ),
                pn.Column(
                    pn.Row("Predicted 1", max_height=30),
                    pn.Row(pn.pane.Str(self.conf_mat_val_fp), min_height=50),
                    pn.Row(pn.pane.Str(self.conf_mat_val_tp), min_height=50),
                ),
            ),
        )

    def setup_panel(self):
        print("setup_panel")

        if not self.training:
            self.setup_row[0] = pn.Row(
                pn.Column(
                    pn.Row(
                        self.classifier_dropdown,
                        self.query_strategy_dropdown,
                        max_height=55,
                    ),
                    self.starting_num_points,
                    max_height=110,
                ),
                pn.Column(
                    self.add_classifier_button,
                    self.remove_classifier_button,
                    self.start_training_button,
                ),
                pn.Column(self.classifier_table, max_height=125),
            )
        else:

            self.setup_row[0] = pn.Column(
                pn.widgets.StaticText(
                    name="Number of points trained on",
                    value=f"{self.curr_num_points}",
                ),
            )

    def panel(self):
        print("panel")

        self.assign_label_group.value = self.last_label
        print(f"PANEL - {self.training}")
        start = time.time()
        self.setup_panel()
        end = time.time()
        print(f"setup {end - start}")
        start = time.time()

        buttons_row = pn.Row(max_height=30)
        if self.training:
            if self.assigned_label:
                buttons_row.append(self.next_iteration_button)
            else:
                buttons_row = pn.Row(self.assign_label_group,
                                     pn.Row(self.assign_label_button,
                                            self.show_queried_button,
                                            self.checkpoint_button,
                                            max_height=30),
                                     max_height=30,)

        self.panel_row[0] = pn.Column(
            pn.Row(self.setup_row),
            pn.Row(
                pn.Tabs(
                    ("Train", self.train_tab),
                    ("Metric", self.metric_tab),
                    ("Val", self.val_tab),
                    ("Scores", self.scores_tab),
                    dynamic=True,
                ),
                self.add_conf_matrices(),
            ),
            pn.Row(max_height=20),
            buttons_row,
        )
        end = time.time()
        print(f"panel_row {end - start}")
        print("\n====================\n")
        return self.panel_row


class ActiveLearningDashboard(param.Parameterized):
    def __init__(self, src, df, **params):
        super(ActiveLearningDashboard, self).__init__(**params)

        self.df = df
        self.src = src
        self.row = pn.Row(pn.pane.Str("loading"))

        self.add_active_learning()

    def add_active_learning(self):
        self.active_learning = []
        # CHANGED :: Add to AL settings
        for label in settings["labels_to_train"]:
            print(f"Label is {label} with type: {type(label)}")
            raw_label = settings["strings_to_labels"][label]
            print(f"Raw Label is {raw_label} with type: {type(raw_label)}")
            self.active_learning.append(
                ActiveLearningTab(df=self.df, src=self.src, label=label)
            )
        self.al_tabs = pn.Tabs(dynamic=True)
        for i, al_tab in enumerate(self.active_learning):
            self.al_tabs.append((al_tab.label_string, al_tab.panel()))
        self.panel()

    def panel(self):
        self.row[0] = pn.Card(self.al_tabs)
        return self.row


class SelectedInfoDashboard(param.Parameterized):

    optical_image = pn.pane.JPG(
        alt_text="Image Unavailable",
        min_width=200,
        min_height=200,
        sizing_mode="scale_height",
    )

    url_optical_image = ""

    def __init__(self, src, df, **params):
        super(SelectedInfoDashboard, self).__init__(**params)

        #  FIXME :: remove src
        self.src = src
        self.df = df
        self.row = pn.Row(pn.pane.Str("loading"))

        self.selected_history = []

        self.image_zoom = 1.0

        self.add_selected_info()

    def add_selected_info(self):
        self.contents = "Selected Info"
        self.search_id = pn.widgets.AutocompleteInput(
            name="Select ID",
            options=list(self.src.data[settings["id_col"]].values),
            placeholder="Select a source by ID",
            max_height=50,
        )

        self.search_id.param.watch(self.change_selected, "value")

        self.image_zoom = 0.2
        # FIXME :: update this
        self.src.on_change("data", self.panel_cb)
        self.src.selected.on_change("indices", self.panel_cb)
        self.panel()

    def change_selected(self, event):

        # FIXME :: Change this
        selected_index = self.src.data[settings["id_col"]][
            self.src.data[settings["id_col"]] == event.new
        ].index[0]

        self.src.selected.indices = [selected_index]

    def panel_cb(self, attr, old, new):

        if len(self.src.selected.indices) > 0:
            self.selected_history = [
                self.src.data[settings["id_col"]][self.src.selected.indices[0]]
            ] + self.selected_history

        self.panel()

    def empty_selected(self, event):
        self.src.selected.indices = []

    def panel(self):
        # CHANGED :: Remove need to rerender with increases + decreases
        def change_zoom_cb(event, oper):
            if oper == "-":
                self.image_zoom += 0.1
                self.image_zoom = round(self.image_zoom, 1)
            if oper == "+":
                if self.image_zoom == 0.1:
                    pass
                else:
                    self.image_zoom -= 0.1
                    self.image_zoom = round(self.image_zoom, 1)
            try:
                index = self.url_optical_image.rfind("&")
                self.url_optical_image = (
                    f"{self.url_optical_image[:index]}&scale={self.image_zoom}"
                )
                self.optical_image.object = self.url_optical_image
            except:

                print("\n\n\n IMAGE ERROR: \n\n\n")
                print(f"index:{self.url_optical_image.rfind('&')}")
                print(
                    f"new url_optical_image: {self.url_optical_image[:self.url_optical_image.rfind('&')]}&scale={self.image_zoom}")

        def generate_radio_url(ra, dec):
            # TODO :: Verify
            ra = float(ra)
            dec = float(dec)
            print(f"ra:{ra}, dec:{dec}")

            h = np.floor(ra / 15.0)
            d = ra - h * 15
            m = np.floor(d / 0.25)
            d = d - m * 0.25
            s = d / (0.25 / 60.0)
            ra_conv = f"{h} {m} {s}"

            sign = 1
            if dec < 0:
                sign = -1

            g = np.abs(dec)
            d = np.floor(g) * sign
            g = g - np.floor(g)
            m = np.floor(g * 60.0)
            g = g - m / 60.0
            s = g * 3600.0

            dec_conv = f"{d} {m} {s}"

            print(f"ra_conv: {ra_conv}, dec_conv: {dec_conv}")

            url1 = "https://third.ucllnl.org/cgi-bin/firstimage?RA="
            url2 = "&Equinox=J2000&ImageSize=2.5&MaxInt=200&GIF=1"
            url = f"{url1}{ra_conv} {dec_conv}{url2}"

            return url

        if len(self.src.selected.indices) > 0:
            index = self.src.selected.indices[0]

            url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra="
            # TODO :: Set Ra Dec Columns
            ra_dec = self.src.data["ra_dec"][index]
            ra = ra_dec[: ra_dec.index(",")]
            print(f"RA_DEC:{ra_dec}")
            dec = ra_dec[ra_dec.index(",") + 1:]

            try:
                self.url_optical_image = (
                    f"{url}{ra}&dec={dec}&opt=G&scale={self.image_zoom}"
                )
                self.optical_image.object = self.url_optical_image
            except:
                print("\n\n\n\n Optical Image Timeout \n\n\n\n")

            try:
                url_radio_image = generate_radio_url(ra, dec)
            except:
                print("\n\n\n\n Radio Image Timeout \n\n\n\n")

            zoom_increase = pn.widgets.Button(
                name="Zoom In", max_height=30, max_width=50
            )
            zoom_increase.on_click(partial(change_zoom_cb, oper="+"))
            zoom_decrease = pn.widgets.Button(
                name="Zoom Out", max_height=30, max_width=50
            )
            zoom_decrease.on_click(partial(change_zoom_cb, oper="-"))

            print(len(self.src.data["png_path_DR16"][index]))
            print(self.src.data["png_path_DR16"][index])
            print(type(self.src.data["png_path_DR16"][index]))

            # CHANGED :: Slow after this...
            # FIXME :: update this
            if not self.src.data["png_path_DR16"][index].isspace():
                print("beginning if")
                try:
                    spectra_image = pn.pane.PNG(
                        self.src.data["png_path_DR16"][index],
                        alt_text="Image Unavailable",
                        min_width=400,
                        min_height=400,
                        sizing_mode="scale_height",
                    )
                except:
                    print("\n\n\n\n Radio Image Timeout \n\n\n\n")
                print("leaving if")

            else:
                print("beginning else")
                spectra_image = "No spectra available"
                print("leaving else")

            print("creating deselect")
            deselect_buttton = pn.widgets.Button(name="Deselect")
            deselect_buttton.on_click(self.empty_selected)

            print("setting extra row")
            extra_data_row = pn.Row()
            for col in settings["extra_info_cols"]:
                print(f"inside for: {col}")
                info = f"**{col}**: {str(self.src.data[f'{col}'][index])}"
                extra_data_row.append(
                    pn.pane.Markdown(info, max_width=12
                                     * len(info), max_height=10)
                )
            print("setting row")
            self.row[0] = pn.Card(
                pn.Column(
                    pn.pane.Markdown(
                        f'**Source ID**: {self.src.data[settings["id_col"]][index]}',
                        max_height=10,
                    ),
                    extra_data_row,
                    pn.Column(
                        pn.Row(
                            self.optical_image,
                            pn.pane.GIF(
                                url_radio_image,
                                alt_text="Image Unavailable",
                                min_width=200,
                                min_height=200,
                                sizing_mode="scale_height",
                            ),
                        ),
                        pn.Row(
                            zoom_increase,
                            zoom_decrease,
                        ),
                        spectra_image,
                    ),
                ),
                collapsible=False,
                header=deselect_buttton,
            )

            print("row set")

        else:
            self.row[0] = pn.Card(
                pn.Column(
                    self.search_id,
                    pd.DataFrame(self.selected_history,
                                 columns=["Selected IDs"]),
                )
            )

        return self.row

############################# MAIN PANEL RENDERING #############################


class DataDashboard(param.Parameterized):

    src = ColumnDataSource(data={"0": [], "1": []})

    contents = param.String()

    def __init__(self, src, contents="Menu", main_plot=None, **params):
        super(DataDashboard, self).__init__(**params)

        self.src = src
        self.row = pn.Row(pn.pane.Str("loading"))
        global main_df
        self.df = main_df

        self.contents = contents

    # FIXME :: update this
    def update_panel_contents_src(self, event):
        self.panel_contents.src.data = self.src.data

    @param.depends("contents", watch=True)
    def update_contents(self):

        print("Updating contents")

        if self.contents == "Settings":

            self.panel_contents = SettingsDashboard(
                self, self.src, self.df)

        elif self.contents == "Menu":

            self.panel_contents = MenuDashboard(self)

        elif self.contents == "Plot":

            self.panel_contents = PlotDashboard(self.src)

        elif self.contents == "Active Learning":

            global main_df

            self.df = main_df
            self.panel_contents = ActiveLearningDashboard(self.src, self.df)

        elif self.contents == "Selected Info":

            self.panel_contents = SelectedInfoDashboard(self.src)

        self.panel()

    def set_contents(self, updated):
        self.contents = updated

    def panel(self):

        self.row[0] = self.panel_contents.panel()

        return self.row


source = ColumnDataSource()

main_df = pd.DataFrame()

############################### CREATE TEMPLATE ###############################

files = pn.widgets.FileInput()

react = pn.template.ReactTemplate(title="astronomicAL")

pn.config.sizing_mode = "stretch_both"

settings = {}

if os.path.isfile("astronomicAL/layout.json"):
    with open("astronomicAL/layout.json") as layout_file:
        data = json.load(layout_file)
        for p in data:
            start_row = data[p]["y"]
            end_row = data[p]["y"] + data[p]["h"]
            start_col = data[p]["x"]
            end_col = data[p]["x"] + data[p]["w"]

            if int(p) == 0:
                main_plot = DataDashboard(
                    name="Main Plot", src=source, contents="Settings"
                )
                react.main[start_row:end_row,
                           start_col:end_col] = main_plot.panel()
            else:
                new_plot = DataDashboard(name=f"{p}", src=source)
                react.main[start_row:end_row,
                           start_col:end_col] = new_plot.panel()

else:
    main_plot = DataDashboard(
        name="Main Plot", src=source, contents="Settings")
    react.main[:5, :6] = main_plot.panel()

    num = 0
    for i in [6]:
        new_plot = DataDashboard(name=f"{num}", src=source)
        react.main[:5, 6:] = new_plot.panel()
        num += 1

    for i in [0, 4, 8]:
        new_plot = DataDashboard(name=f"{num}", src=source)
        react.main[5:9, i: i + 4] = new_plot.panel()
        num += 1


save_layout_button = pn.widgets.Button(name="Save current layout")


def save_layout_cb(attr, old, new):
    print("json updated")
    layout = json.loads(new)
    with open('panel_test/layout.json', 'w') as fp:
        json.dump(layout, fp)


layout_dict = {}
text_area_input = TextAreaInput(value="")
text_area_input.on_change("value", save_layout_cb)

save_layout_button.jscallback(clicks="""
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

""", args=dict(text_area_input=text_area_input))

react.header.append(pn.Row(save_layout_button))

react.servable()

# TODO :: http://holoviews.org/reference/apps/bokeh/player.html#apps-bokeh-gallery-player
# TODO :: https://panel.holoviz.org/gallery/simple/save_filtered_df.html#simple-gallery-save-filtered-df

# CHANGED :: Keep axis the same on refresh (stop zooming out - outliers)
# CHANGED :: Save React Layout (likely need to wait for Panel update)
# CHANGED :: Fix Datashader Colours to match Bokeh
# CHANGED :: Fix Datashader Colours to match Bokeh
# CHANGED :: Fix Datashader Colours to match Bokeh
# CHANGED :: Fix Datashader Colours to match Bokeh
# CHANGED :: Fix Datashader Colours to match Bokeh
# CHANGED :: Fix Datashader Colours to match Bokeh
# CHANGED :: Fix Datashader Colours to match Bokeh
# CHANGED :: Fix Datashader Colours to match Bokeh
# CHANGED :: Fix Datashader Colours to match Bokeh
# CHANGED :: Fix Datashader Colours to match Bokeh
