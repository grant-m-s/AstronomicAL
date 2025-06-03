import panel as pn
pn.extension('tabulator')
import astronomicAL.config as config
from astronomicAL.dashboard.plot import PlotDashboard
from astronomicAL.active_learning.active_learning import ActiveLearningModel
import numpy as np
import datashader as ds
import holoviews as hv
import param
import pandas as pd
import random
import os
import json

from functools import partial

from holoviews.operation.datashader import (
    datashade,
    dynspread,
)
hv.extension('bokeh', logo=False)


#This serves to avoid the rangeupdate error which occurs every time the selected source
#changes. The problem is probably arising from datashade + dynspread but i could not solve it if not
#by directly removing the calls to datashade. I just copied this
#snippet of code and it should be deleted and the problem fixed, but for the moment it is useful 
#as it avoids having the terminal filled with error messages
import holoviews.plotting.bokeh.callbacks
original_initialize = holoviews.plotting.bokeh.callbacks.Callback.initialize

def safe_initialize(self, plot_id=None):
    try:
        return original_initialize(self, plot_id)
    except KeyError as e:
        if 'rangesupdate' in str(e):
            print(f"Warning: Skipping unsupported event: {e}")
            return
        else:
            raise e

holoviews.plotting.bokeh.callbacks.Callback.initialize = safe_initialize



class LabellingDashboard(param.Parameterized):
    """A dashboard for .

    Parameters
    ----------


    Attributes
    ----------

    """

    X_variable = param.Selector(
        objects=["0"], default="0", doc="Selection box for the X axis of the plot."
    )

    Y_variable = param.Selector(
        objects=["1"], default="1", doc="Selection box for the Y axis of the plot."
    )

    def __init__(self, src, df):
        super(LabellingDashboard, self).__init__()

        self.row = pn.Row(pn.pane.Str("loading"))
        self.df = df
        self.sample_region = df
        self.region_criteria_df = pd.DataFrame([], columns=["column", "oper", "value"])
        self.region_message = ""
        self.src = src
        self.src.on_change("data", self._panel_cb)

        self.labels = self.get_previous_labels()
        self._construct_panel()

        ActiveLearningModel(self.src, df, config.settings["labels_to_train"][0])

        self._update_variable_lists()
        self.select_random_point()

    def _construct_panel(self):

        options = []

        all_labels = list(config.main_df[config.settings["label_col"]].unique())

        all_labels.sort()

        if -1 in all_labels:
            all_labels.remove(-1)

        if config.settings["exclude_labels"]:
            for i in config.settings["unclassified_labels"]:
                all_labels.remove(config.settings["strings_to_labels"][f"{i}"])

        for i in all_labels:
            options.append(config.settings["labels_to_strings"][f"{i}"])

        options.append("Unsure")
        self.assign_label_group = pn.widgets.RadioButtonGroup(
            name="Label button group",
            options=options,
        )

        self.assign_label_button = pn.widgets.Button(
            name="Assign Label", button_type="primary"
        )
        self.assign_label_button.on_click(self._assign_label_cb)

        self.first_labelled_button = pn.widgets.Button(name="First", max_height=35)
        self.first_labelled_button.on_click(
            partial(self.update_selected_point_from_buttons, button="First")
        )
        self.prev_labelled_button = pn.widgets.Button(name="<", max_height=35)
        self.prev_labelled_button.on_click(
            partial(self.update_selected_point_from_buttons, button="<")
        )
        self.next_labelled_button = pn.widgets.Button(name=">", max_height=35)
        self.next_labelled_button.on_click(
            partial(self.update_selected_point_from_buttons, button=">")
        )
        self.new_labelled_button = pn.widgets.Button(
            name="New", max_height=35, max_width=20
        )
        self.new_labelled_button.on_click(
            partial(self.update_selected_point_from_buttons, button="New")
        )
        self.column_dropdown = pn.widgets.Select(
            name="Column",
            options=list(self.df.columns),
            max_width=100,
            width=100,
        )
        self.operation_dropdown = pn.widgets.Select(
            name="Operation", options=[">", ">=", "==", "!=", "<=", "<"], max_width=75
        )
        self.input_value = pn.widgets.TextInput(name="Value", max_width=50)

        self.add_sample_criteria_button = pn.widgets.Button(
            name="Add Criterion", max_height=30, max_width=300
        )
        self.add_sample_criteria_button.on_click(
            partial(self.update_sample_region, button="ADD")
        )

        self.remove_sample_criteria_button = pn.widgets.Button(
            name="Remove Criterion", max_height=30, max_width=300
        )
        self.remove_sample_criteria_button.on_click(
            partial(self.update_sample_region, button="REMOVE")
        )

        self.criteria_dict = {}
        self.remove_sample_selection_dropdown = pn.widgets.Select(
            name="Criterion to Remove",
            options=[""],
            max_width=300,
            width=300,
        )

    def _update_variable_lists(self):
        """Update the list of options used inside `X_variable` and `Y_variable`.

        This method retrieves an up-to-date list of columns inside `df` and
        assigns them to both Selector objects.

        Returns
        -------
        None

        """

        cols = list(config.main_df.columns)

        if config.settings["id_col"] in cols:
            cols.remove(config.settings["id_col"])
        if config.settings["label_col"] in cols:
            cols.remove(config.settings["label_col"])

        self.column_dropdown.options = cols

        self.param.X_variable.objects = cols
        self.param.Y_variable.objects = cols
        self.param.X_variable.default = config.settings["default_vars"][0]
        self.param.Y_variable.default = config.settings["default_vars"][1]
        self.X_variable = config.settings["default_vars"][0]
        self.Y_variable = config.settings["default_vars"][1]

    def update_sample_region(self, event=None, button="ADD"):

        if button == "ADD":
            if self.input_value.value == "":
                return

            updated_df = pd.DataFrame(
                [
                    [
                        self.column_dropdown.value,
                        self.operation_dropdown.value,
                        self.input_value.value,
                    ]
                ],
                columns=["column", "oper", "value"],
            )

            self.criteria_dict[
                f"{self.column_dropdown.value} {self.operation_dropdown.value} {self.input_value.value}"
            ] = [
                self.column_dropdown.value,
                self.operation_dropdown.value,
                self.input_value.value,
            ]

            if len(self.region_criteria_df) == 0:
                self.region_criteria_df = updated_df

            else:
                exists = self.region_criteria_df[
                    (self.region_criteria_df["column"] == updated_df["column"][0])
                    & (self.region_criteria_df["oper"] == updated_df["oper"][0])
                    & (self.region_criteria_df["value"] == updated_df["value"][0])
                ]
                if len(exists) == 0:
                    self.region_criteria_df = self.region_criteria_df.append(
                        updated_df, ignore_index=True
                    )
                else:
                    return

        elif button == "REMOVE":
            if len(self.region_criteria_df) == 0:
                return
            else:
                col = self.criteria_dict[self.remove_sample_selection_dropdown.value][0]
                oper = self.criteria_dict[self.remove_sample_selection_dropdown.value][
                    1
                ]
                val = self.criteria_dict[self.remove_sample_selection_dropdown.value][2]

                exists = self.region_criteria_df[
                    (self.region_criteria_df["column"] == col)
                    & (self.region_criteria_df["oper"] == oper)
                    & (self.region_criteria_df["value"] == val)
                ]
                self.criteria_dict.pop(
                    self.remove_sample_selection_dropdown.value, None
                )
                self.region_criteria_df.drop([exists.index[0]], inplace=True)

        all_bools = ""

        for i in range(len(self.region_criteria_df)):
            row = self.region_criteria_df.iloc[i]
            oper = row["oper"]
            col = row["column"]
            value = float(row["value"])
            if oper == ">":
                bool = f"{col} > {value}"
            elif oper == ">=":
                bool = f"{col} >= {value}"
            elif oper == "==":
                bool = f"{col} == {value}"
            elif oper == "!=":
                bool = f"{col} != {value}"
            elif oper == "<=":
                bool = f"{col} <= {value}"
            elif oper == "<":
                bool = f"{col} < {value}"

            if i == 0:
                all_bools = bool
            else:
                all_bools = f"{all_bools} & {bool}"

        if all_bools == "":
            self.sample_region = self.df
        else:
            self.sample_region = self.df.query(all_bools)

        if len(self.sample_region) == 0:
            self.region_message = "No Matching Sources!"
        elif len(self.sample_region) == len(self.df):
            self.region_message = f"All Sources Matching ({len(self.sample_region)})"
        else:
            self.region_message = f"{len(self.sample_region)} Matching Sources"

        self.remove_sample_selection_dropdown.options = list(self.criteria_dict.keys())

        self.panel()

    @param.depends("X_variable", "Y_variable")
    def plot(self, x_var=None, y_var=None):
        """Create a basic scatter plot of the data with the selected axis.

        The data is represented as a Holoviews Datashader object allowing for
        large numbers of points to be rendered at once. Plotted using a Bokeh
        renderer, the user has full manuverabilty of the data in the plot.

        Returns
        -------
        plot : Holoviews Object
            A Holoviews plot

        """

        if x_var is None:
            x_var = self.X_variable

        if y_var is None:
            y_var = self.Y_variable

        p = hv.Points(
            self.df,
            [x_var, y_var],
        ).opts(
            #active_tools=["pan", "wheel_zoom"])
        )
        sample_region = hv.Points(
            self.sample_region,
            [x_var, y_var],
        ).opts(
            #active_tools=["pan", "wheel_zoom"])
        )
        cols = list(self.df.columns)

        if len(self.src.data[cols[0]]) == 1:
            selected = pd.DataFrame(self.src.data, columns=cols, index=[0])
        else:
            selected = pd.DataFrame(columns=cols)

        selected_plot = hv.Scatter(selected, x_var, y_var,).opts(
            fill_color="black",
            marker="circle",
            size=10,
            #active_tools=["pan", "wheel_zoom"],
        )

        color_key = config.settings["label_colours"]

        # color_points = hv.NdOverlay(
        #     {
        #         config.settings["labels_to_strings"][f"{n}"]: hv.Points(
        #             [0, 0], label=config.settings["labels_to_strings"][f"{n}"]
        #         ).opts(style=dict(color=color_key[n], size=0))
        #         for n in color_key
        #     }
        # )

        max_x = np.max(self.df[x_var])
        min_x = np.min(self.df[x_var])

        max_y = np.max(self.df[y_var])
        min_y = np.min(self.df[y_var])

        x_sd = np.std(self.df[x_var])
        x_mu = np.mean(self.df[x_var])
        y_sd = np.std(self.df[y_var])
        y_mu = np.mean(self.df[y_var])

        max_x = np.min([x_mu + 4 * x_sd, max_x])
        min_x = np.max([x_mu - 4 * x_sd, min_x])

        max_y = np.min([y_mu + 4 * y_sd, max_y])
        min_y = np.max([y_mu - 4 * y_sd, min_y])

        if selected.shape[0] > 0:

            max_x = np.max([max_x, np.max(selected[x_var])])
            min_x = np.min([min_x, np.min(selected[x_var])])

            max_y = np.max([max_y, np.max(selected[y_var])])
            min_y = np.min([min_y, np.min(selected[y_var])])

        new_key = {}

        for k in list(color_key.keys()):
            new_key[k] = "#333333"

        all_points = dynspread(
            datashade(
                p,
                color_key=new_key,
                aggregator=ds.by(config.settings["label_col"], ds.count()),
            ).opts(
                xlim=(min_x, max_x),
                ylim=(min_y, max_y),
                #responsive=True,
                alpha=0.5,
                #shared_axes=False,
                framewise=False,        
                axiswise=False,         
            ),
            threshold=0.3,
            how="over",
        )

        sample_region_plot = dynspread(
            datashade(
                sample_region,
                color_key=color_key,
                aggregator=ds.by(config.settings["label_col"], ds.count()),
                min_alpha=70,
                alpha=100,
            ).opts(
                xlim=(min_x, max_x),
                ylim=(min_y, max_y),
                #responsive=True,
                #shared_axes=False,
                framewise=False,        
                axiswise=False, 
            ),
            threshold=0.7,
            how="saturate",
        )
        plot = (all_points * sample_region_plot * selected_plot).opts(
            #shared_axes=False,
        )

        return plot
    
    
    def _assign_label_cb(self, event):
        print("_assign_label_cb")

        selected_label = self.assign_label_group.value
        id = self.src.data[config.settings["id_col"]][0]

        self.assign_label_button.disabled = True

        if selected_label != "Unsure":
            raw_label = config.settings["strings_to_labels"][selected_label]
        else:
            raw_label = -1

        self.save_label(id, raw_label)

        self.select_random_point()

    def get_previous_labels(self):

        labels = {}

        if os.path.exists("data/test_set.json"):
            with open("data/test_set.json", "r") as json_file:
                labels = json.load(json_file)

        return labels

    def save_label(self, id, label):

        labels = self.get_previous_labels()

        labels[id] = int(label)

        self.labels = labels

        with open("data/test_set.json", "w+") as outfile:
            json.dump(self.labels, outfile)

    def select_random_point(self):

        inside_region = list(self.sample_region[config.settings["id_col"]].values)

        if len(inside_region) == 0:
            self.region_message = "No Matching Sources!"
            return
        elif len(inside_region) == len(self.df):
            self.region_message = f"All Sources Matching ({len(self.sample_region)})"
        else:
            self.region_message = f"{len(inside_region)} Matching Sources"

        selected = random.choice(
            list(self.sample_region[config.settings["id_col"]].values)
        )
        selected_source = self.df[self.df[config.settings["id_col"]] == selected]
        selected_dict = selected_source.to_dict("list")

        self.src.data = selected_dict
        self.assign_label_group.value = "Unsure"

    def update_selected_point_from_buttons(self, event, button):

        index = self.get_current_index_in_labelled_data()

        updated = None

        if button == "<":

            updated = list(self.labels.keys())[index - 1]

        elif button == ">":

            updated = list(self.labels.keys())[index + 1]

        elif button == "First":

            updated = list(self.labels.keys())[0]

        elif button == "New":
            self.select_random_point()
            return

        if updated is not None:

            selected_source = self.df[self.df[config.settings["id_col"]] == updated]
            selected_dict = selected_source.to_dict("list")
            self.assign_label_group.value = config.settings["labels_to_strings"][
                f"{self.labels[updated]}"
            ]

            self.src.data = selected_dict

    def get_current_index_in_labelled_data(self):

        total = len(self.labels.keys())

        if len(self.src.data[config.settings["id_col"]]) > 0:
            if self.src.data[config.settings["id_col"]][0] in list(self.labels.keys()):
                index = list(self.labels.keys()).index(
                    self.src.data[config.settings["id_col"]][0]
                )
            else:
                index = total
        else:
            index = "-"

        return index

    def _reset_index_buttons(self):
        self.first_labelled_button.disabled = False
        self.prev_labelled_button.disabled = False
        self.next_labelled_button.disabled = False
        self.new_labelled_button.disabled = False

    def _panel_cb(self, attr, old, new):
        print("_panel_cb callback")
        self.panel()

    def _apply_format(self, plot, element):
        plot.handles["table"].autosize_mode = "none"
        plot.handles["table"].index_position = None  # hide index
        plot.handles["table"].columns[0].width = 80
        plot.handles["table"].columns[1].width = 50
        plot.handles["table"].columns[2].width = 50

    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which can then be rendered by the
            parent Dashboard.

        """

        col_names = self.region_criteria_df.columns.tolist()


        pn.extension('tabulator')

        df_pane = pn.widgets.Tabulator(self.region_criteria_df, widths={col_names[0]:80,col_names[1]:50,col_names[2]:50})

        buttons_row = pn.Row(
            self.assign_label_group,
            pn.Row(
                self.assign_label_button,
                max_height=30,
            ),
            max_height=30,
            max_width=600,
        )

        plot = pn.Row(self.plot, height=400, width=500, sizing_mode="fixed")
        total = len(self.labels.keys())

        index = self.get_current_index_in_labelled_data()

        print("current index: ", index, type(index))

        self._reset_index_buttons()

        if (index == 0) or (total == 0):
            self.first_labelled_button.disabled = True
            self.prev_labelled_button.disabled = True
        if type(index) == int:
            if index >= (total - 1):
                self.next_labelled_button.disabled = True
        else:
            print("\n\ncurrent index is not int: ", index, type(index), "\n\n")

        if len(self.sample_region) == 0:
            self.new_labelled_button.disabled = True

        if self.src.data[config.settings["id_col"]][0] in list(self.labels.keys()):

            raw_label = self.labels[self.src.data[config.settings["id_col"]][0]]

            label = config.settings["labels_to_strings"][f"{raw_label}"]

            previous_label = pn.widgets.StaticText(
                name="Current Label",
                value=f"{label}",
            )
        else:
            previous_label = pn.widgets.StaticText(
                name="Current Label",
                value=f"Unlabelled",
            )

        dataset_raw_label = self.src.data[config.settings["label_col"]][0]
        dataset_label = config.settings["labels_to_strings"][f"{dataset_raw_label}"]

        if (index + 1) > total:
            index_tally = f"NEW ({total} Labelled)"
        else:
            index_tally = f"{index+1}/{total}"

        labelling_info_col = pn.Column(
            pn.Column(
                pn.Row(self.region_message, max_height=50),
                pn.Row(
                    df_pane, max_height=130, sizing_mode="stretch_width", scroll=True
                ),
                max_height=100,
                margin=(0, 0, 50, 0),
                max_width=400,
            ),
            pn.Row(
                self.column_dropdown,
                self.operation_dropdown,
                self.input_value,
            ),
            self.add_sample_criteria_button,
            self.remove_sample_selection_dropdown,
            self.remove_sample_criteria_button,
            pn.widgets.StaticText(name="Labelled Point", value=index_tally),
            pn.widgets.StaticText(
                name="Source ID", value=f"{self.src.data[config.settings['id_col']][0]}"
            ),
            pn.widgets.StaticText(
                name="Original Dataset Label",
                value=f"{dataset_label}",
            ),
            pn.Row(previous_label, max_height=25),
            pn.Row(
                self.first_labelled_button,
                self.prev_labelled_button,
                self.next_labelled_button,
                self.new_labelled_button,
            ),
        )

        self.assign_label_button.disabled = False

        print("row before", self.row[0])

        self.row[0] = pn.Card(
            pn.Row(
                plot,
                labelling_info_col,
                margin=(0, 20),
            ),
            buttons_row,
            header=pn.Row(
                pn.Spacer(width=25,
                        #    sizing_mode="fixed"
                           ),
                pn.Row(self.param.X_variable, max_width=100),
                pn.Row(self.param.Y_variable, max_width=100),
                max_width=100,
                # sizing_mode="fixed",
            ),
            collapsible=False,
            sizing_mode="stretch_both",
        )
        return self.row
