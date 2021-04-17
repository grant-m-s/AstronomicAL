import astronomicAL.config as config
from astronomicAL.dashboard.plot import PlotDashboard
import panel as pn
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
        self.src = src
        self.src.on_change("data", self._panel_cb)

        self.labels = self.get_previous_labels()
        self._construct_panel()
        self._update_variable_lists()
        self.select_random_point()

    def _construct_panel(self):

        options = []
        for label in config.settings["labels_to_train"]:
            options.append(label)
        options.append("Unsure")
        self.assign_label_group = pn.widgets.RadioButtonGroup(
            name="Label button group",
            options=options,
        )

        self.assign_label_button = pn.widgets.Button(
            name="Assign Label", button_type="primary"
        )
        self.assign_label_button.on_click(self._assign_label_cb)

        self.first_labelled_button = pn.widgets.Button(name="First")
        self.first_labelled_button.on_click(
            partial(self.update_selected_point_from_buttons, button="First")
        )
        self.prev_labelled_button = pn.widgets.Button(name="<")
        self.prev_labelled_button.on_click(
            partial(self.update_selected_point_from_buttons, button="<")
        )
        self.next_labelled_button = pn.widgets.Button(name=">")
        self.next_labelled_button.on_click(
            partial(self.update_selected_point_from_buttons, button=">")
        )
        self.new_labelled_button = pn.widgets.Button(name="New")
        self.new_labelled_button.on_click(
            partial(self.update_selected_point_from_buttons, button="New")
        )

    def _update_variable_lists(self):
        """Update the list of options used inside `X_variable` and `Y_variable`.

        This method retrieves an up-to-date list of columns inside `df` and
        assigns them to both Selector objects.

        Returns
        -------
        None

        """
        print(f"x_var currently is: {self.X_variable}")

        cols = list(config.main_df.columns)

        if config.settings["id_col"] in cols:
            cols.remove(config.settings["id_col"])
        if config.settings["label_col"] in cols:
            cols.remove(config.settings["label_col"])

        self.param.X_variable.objects = cols
        self.param.Y_variable.objects = cols
        self.param.X_variable.default = config.settings["default_vars"][0]
        self.param.Y_variable.default = config.settings["default_vars"][1]
        self.X_variable = config.settings["default_vars"][0]
        self.Y_variable = config.settings["default_vars"][1]

        print(f"x_var has now changed to: {self.X_variable}")

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
        plot_db = PlotDashboard(self.src, None)
        return plot_db.plot(x_var=self.X_variable, y_var=self.Y_variable)

    def _assign_label_cb(self, event):

        selected_label = self.assign_label_group.value
        id = self.src.data[config.settings["id_col"]][0]

        self.assign_label_button.disabled = True

        if selected_label is not "Unsure":
            raw_label = config.settings["strings_to_labels"][selected_label]
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

        selected = random.choice(list(self.df[config.settings["id_col"]].values))
        selected_source = self.df[self.df[config.settings["id_col"]] == selected]
        selected_dict = selected_source.set_index(config.settings["id_col"]).to_dict(
            "list"
        )

        self.src.data = selected_source

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
            selected_dict = selected_source.set_index(
                config.settings["id_col"]
            ).to_dict("list")

            self.src.data = selected_source

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
        self.panel()

    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which can then be rendered by the
            parent Dashboard.

        """

        print("Labelling panel rendering...")

        buttons_row = pn.Row(
            self.assign_label_group,
            pn.Row(
                self.assign_label_button,
                max_height=30,
            ),
            max_height=30,
        )

        plot = pn.Row(self.plot, height=400, width=500, sizing_mode="fixed")
        total = len(self.labels.keys())

        index = self.get_current_index_in_labelled_data()

        self._reset_index_buttons()

        if (index == 0) or (total == 0):
            self.first_labelled_button.disabled = True
            self.prev_labelled_button.disabled = True

        if index >= (total - 1):
            self.next_labelled_button.disabled = True

        if index >= total:
            self.new_labelled_button.disabled = True
            self.next_labelled_button.disabled = True

        if self.src.data[config.settings["id_col"]][0] in list(self.labels.keys()):

            raw_label = self.labels[self.src.data[config.settings["id_col"]][0]]

            label = config.settings["labels_to_strings"][f"{raw_label}"]

            previous_label = pn.widgets.StaticText(
                name="You previously labelled this",
                value=f"{label}",
            )
        else:
            previous_label = "You have not yet labelled this source."

        dataset_raw_label = self.src.data[config.settings["label_col"]][0]
        dataset_label = config.settings["labels_to_strings"][f"{dataset_raw_label}"]

        labelling_info_col = pn.Column(
            pn.Row(
                self.first_labelled_button,
                self.prev_labelled_button,
                self.next_labelled_button,
                self.new_labelled_button,
            ),
            pn.widgets.StaticText(name="Index", value=f"{index+1}/{total}"),
            pn.widgets.StaticText(
                name="Source ID", value=f"{self.src.data[config.settings['id_col']][0]}"
            ),
            pn.widgets.StaticText(
                name="Original Dataset Label",
                value=f"{dataset_label}",
            ),
            previous_label,
        )

        self.assign_label_button.disabled = False

        self.row[0] = pn.Card(
            pn.Row(
                plot,
                labelling_info_col,
            ),
            buttons_row,
            header=pn.Row(
                pn.Spacer(width=25, sizing_mode="fixed"),
                pn.Row(self.param.X_variable, max_width=200),
                pn.Row(self.param.Y_variable, max_width=200),
                max_width=400,
                sizing_mode="fixed",
            ),
            collapsible=False,
            sizing_mode="stretch_both",
        )
        return self.row
