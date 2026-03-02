import panel as pn
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

    def __init__(self, src, df, context=None):
        super(LabellingDashboard, self).__init__()

        self.row = pn.Row(pn.pane.Str("loading"))
        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        self.df = self.config.main_df

        self._sub = self.context.events.subscribe("dataset.main.updated", self._on_df_updated)

        self.sample_region = self.df
        self.region_criteria_df = pd.DataFrame([], columns=["column", "oper", "value"])
        self.region_message = ""
        self.src = src
        self.watch_bokeh(self.src, "data", self._panel_cb)
        self.labels = self.get_previous_labels()
        self._construct_panel()

        ActiveLearningModel(self.src, self.df, self.config.settings["labels_to_train"][0], context=self.context)

        self._update_variable_lists()
        self.select_random_point()

    def _on_df_updated(self, topic, payload):

        self.df = self.config.main_df
        self._update_variable_lists()

    def get_toolbar(self):
        return pn.Spacer(height=1)

    def watch_bokeh(self, model, attr: str, callback):
        """Register and track Bokeh model.on_change callbacks for unified disposal."""
        if model is None:
            return
        try:
            model.on_change(attr, callback)
            self._bokeh_on_change.append((model, attr, callback))
        except Exception:
            pass

    def unwatch_all_bokeh(self):
        """Remove all tracked Bokeh callbacks (idempotent)."""
        for model, attr, callback in list(getattr(self, "_bokeh_on_change", [])):
            try:
                model.remove_on_change(attr, callback)
            except Exception as e:
                # Ignore double-remove noise
                if "list.remove(x): x not in list" in str(e):
                    pass
            # continue regardless
        self._bokeh_on_change = []

    def _dispose_impl(self):
        """Subclass-specific cleanup hook (override if needed)."""
        return

    def dispose(self):
        if getattr(self, "_disposed", False):
            return
        self._disposed = True

        # 1) subclass cleanup first
        try:
            self._dispose_impl()
        except Exception:
            pass

        # 2) unwatch bokeh callbacks (including src.on_change if registered via watch_bokeh)
        try:
            self.unwatch_all_bokeh()
        except Exception:
            pass

        # 3) Unsubscribe EventBus
        if self.context and getattr(self.context, "events", None):
            for sub in list(getattr(self, "_event_subs", [])):
                try:
                    self.context.events.unsubscribe(sub)
                except Exception:
                    pass
        self._event_subs = []

        # 4) Stop periodic callbacks
        for cb in list(getattr(self, "_periodic_cbs", [])):
            try:
                cb.stop()
            except Exception:
                pass
        self._periodic_cbs = []

    def _construct_panel(self):

        options = []

        all_labels = list(self.config.main_df[self.config.settings["label_col"]].unique())

        all_labels.sort()

        if -1 in all_labels:
            all_labels.remove(-1)

        if self.config.settings["exclude_labels"]:
            for i in self.config.settings["unclassified_labels"]:
                all_labels.remove(self.config.settings["strings_to_labels"][f"{i}"])

        for i in all_labels:
            options.append(self.config.settings["labels_to_strings"][f"{i}"])

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

        cols = list(self.config.main_df.columns)

        if self.config.settings["id_col"] in cols:
            cols.remove(self.config.settings["id_col"])
        if self.config.settings["label_col"] in cols:
            cols.remove(self.config.settings["label_col"])

        self.column_dropdown.options = cols

        self.param.X_variable.objects = cols
        self.param.Y_variable.objects = cols
        self.param.X_variable.default = self.config.settings["default_vars"][0]
        self.param.Y_variable.default = self.config.settings["default_vars"][1]
        self.X_variable = self.config.settings["default_vars"][0]
        self.Y_variable = self.config.settings["default_vars"][1]

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
                    self.region_criteria_df = pd.concat([self.region_criteria_df,updated_df], 
                                                        ignore_index=True
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

        color_key = self.config.settings["label_colours"]

        # color_points = hv.NdOverlay(
        #     {
        #         self.config.settings["labels_to_strings"][f"{n}"]: hv.Points(
        #             [0, 0], label=self.config.settings["labels_to_strings"][f"{n}"]
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
                aggregator=ds.by(self.config.settings["label_col"], ds.count()),
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
                aggregator=ds.by(self.config.settings["label_col"], ds.count()),
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
        id = self.src.data[self.config.settings["id_col"]][0]

        self.assign_label_button.disabled = True

        if selected_label != "Unsure":
            raw_label = self.config.settings["strings_to_labels"][selected_label]
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

        inside_region = list(self.get_id(df =self.sample_region).values)

        if len(inside_region) == 0:
            self.region_message = "No Matching Sources!"
            return
        elif len(inside_region) == len(self.df):
            self.region_message = f"All Sources Matching ({len(self.sample_region)})"
        else:
            self.region_message = f"{len(inside_region)} Matching Sources"

        selected = random.choice(inside_region)
        
        
        selected_source = self.df[self.get_id() == selected]
        selected_dict = selected_source.to_dict("list")
        if self.config.settings["id_col"] not in selected_dict:
            selected_dict[self.config.settings["id_col"]] = selected

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

            selected_source = self.df[self.get_id() == updated]
            selected_dict = selected_source.to_dict("list")
            if self.config.settings["id_col"] not in selected_dict:
                selected_dict[self.config.settings["id_col"]] = updated

            self.assign_label_group.value = self.config.settings["labels_to_strings"][
                f"{self.labels[updated]}"
            ]

            self.src.data = selected_dict

    def get_current_index_in_labelled_data(self):
        id_col = self.config.settings["id_col"]
        labelled_keys = list(self.labels.keys())
        total = len(labelled_keys)
        if id_col in self.src.data and len(self.src.data[id_col]) > 0:
            if self.src.data[id_col][0] in list(self.labels.keys()):
                index = labelled_keys .index(
                    self.src.data[id_col][0]
                )
            else:
                index = total
        else:
            index = "-"
        return index
    
    def get_id(self, df = None):
        id_col = self.config.settings["id_col"]
        target_df = self.df if df is None else df

        if id_col == "Use Index":
            return pd.Series(target_df.index, index=target_df.index)
        else:
            if id_col not in target_df.columns:
                raise KeyError(f"ID column '{id_col}' not found in DataFrame.")
            return target_df[id_col]

    def _reset_index_buttons(self):
        self.first_labelled_button.disabled = False
        self.prev_labelled_button.disabled = False
        self.next_labelled_button.disabled = False
        self.new_labelled_button.disabled = False

    def _panel_cb(self, attr, old, new):

        self.sample_region = self.df
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

        # ---- Table ----
        col_names = self.region_criteria_df.columns.tolist()
        widths = {}
        if len(col_names) > 0:
            widths[col_names[0]] = 140
        if len(col_names) > 1:
            widths[col_names[1]] = 70
        if len(col_names) > 2:
            widths[col_names[2]] = 70

        df_pane = pn.widgets.Tabulator(
            self.region_criteria_df,
            sizing_mode="stretch_both",   # <-- allow it to grow vertically
            widths=widths,
        )

        # Compact assign button
        try:
            self.assign_label_button.sizing_mode = "fixed"
            self.assign_label_button.width = 130
        except Exception:
            pass

        try:
            self.assign_label_group.sizing_mode = "fixed"
        except Exception:
            pass

        # A centered cluster that wraps if the window gets narrow
        buttons_row = pn.FlexBox(
            self.assign_label_group,
            self.assign_label_button,
            flex_wrap="wrap",
            justify_content="center",   # <-- center in the available width
            align_items="center",
            sizing_mode="stretch_width",
            height=44,
            margin=(6, 0, 0, 0),
        )

        # ---- Your indexing/enable/disable logic (kept) ----
        total = len(self.labels.keys())
        index = self.get_current_index_in_labelled_data()
        self._reset_index_buttons()

        if (index == 0) or (total == 0):
            self.first_labelled_button.disabled = True
            self.prev_labelled_button.disabled = True

        if isinstance(index, (int, np.integer)):
            if index >= (total - 1):
                self.next_labelled_button.disabled = True
        else:
            print("\n\ncurrent index is not int: ", index, type(index), "\n\n")

        if len(self.sample_region) == 0:
            self.new_labelled_button.disabled = True

        # ---- Labels info ----
        src_id = self.src.data[self.config.settings["id_col"]][0]

        if src_id in self.labels:
            raw_label = self.labels[src_id]
            label = self.config.settings["labels_to_strings"][f"{raw_label}"]
            previous_label = pn.widgets.StaticText(name="Current Label", value=str(label))
        else:
            previous_label = pn.widgets.StaticText(name="Current Label", value="Unlabelled")

        dataset_raw_label = self.src.data[self.config.settings["label_col"]][0]
        dataset_label = self.config.settings["labels_to_strings"][f"{dataset_raw_label}"]

        if (index + 1) > total:
            index_tally = f"NEW ({total} Labelled)"
        else:
            index_tally = f"{index+1}/{total}"

        # ---- Sidebar (fixed width, scrolls) ----
        sidebar_width = 450

        # ---- Header + message (compact) ----
        title = pn.pane.HTML(
            "<div style='margin:0; padding:0; line-height:1; font-weight:600;'>All Sources Matching</div>",
            margin=(0, 0, 0, 0),
        )

        # Force the message component to have no margin (works for most panes/widgets)
        try:
            self.region_message.margin = (0, 0, 0, 0)
        except Exception:
            pass

        criteria_header = pn.Column(
            title,
            self.region_message,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            max_height=50,
        )

        # ---- Growable table container (this is what fills height) ----
        table_box = pn.Column(
            df_pane,
            sizing_mode="stretch_both",
            min_height=130,          # keeps it usable in small windows
            margin=(0, 0, 6, 0),
        )

        df_pane.row_height = 28  # try 28–34 if needed

        filter_row = pn.Row(
            self.column_dropdown,
            self.operation_dropdown,
            self.input_value,
            sizing_mode="stretch_width",
            margin=(0, 0, 5, 0),
        )

        # ---- Compact controls + info (bunched) ----
        compact_info = pn.Column(
            filter_row,
            self.add_sample_criteria_button,
            self.remove_sample_selection_dropdown,
            self.remove_sample_criteria_button,
            pn.widgets.StaticText(name="Labelled Point", value=index_tally),
            pn.widgets.StaticText(name="Source ID", value=str(src_id)),
            pn.widgets.StaticText(name="Original Dataset Label", value=str(dataset_label)),
            previous_label,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        # ---- Nav row (bottom pinned) ----
        nav_row = pn.Row(
            self.first_labelled_button,
            self.prev_labelled_button,
            self.next_labelled_button,
            self.new_labelled_button,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        # ---- Sidebar: table grows, nav sticks to bottom ----
        labelling_info_col = pn.Column(
            criteria_header,
            table_box,              # <-- only this grows
            compact_info,           # <-- stays compact
            pn.Spacer(),            # <-- pushes nav_row to the bottom
            nav_row,                # <-- glued to bottom
            sizing_mode="stretch_height",
            width=sidebar_width,
            min_width=sidebar_width,
            max_width=sidebar_width,
            scroll=False,
            margin=(0, 0, 0, 10),
        )

        try:
            self.plot = self.plot.opts(responsive=True)
        except Exception:
            pass
        
        # ---- Plot (DynamicMap via HoloViews pane) ----
        plot_pane = pn.pane.HoloViews(
            self.plot,
            sizing_mode="stretch_both",
            min_width=0,
            min_height=380,
        )

        # ---- Toolbar (stretch width; no tiny max_width) ----
        toolbar = pn.Row(
            self.param.X_variable,
            self.param.Y_variable,
            pn.Spacer(),
            sizing_mode="stretch_width",
            height=50,          # was 50
            margin=(0, 0, 5, 0) # was (0, 0, 5, 0)
        )

        # ---- Main body ----
        body = pn.Row(
            plot_pane,
            labelling_info_col,
            sizing_mode="stretch_both",
            min_height=0,
            min_width=0,
            margin=(0, 0, 0, 0),
        )

        # Enable label assignment
        self.assign_label_button.disabled = False

        self.row[0] = pn.Column(
            toolbar,
            body,
            buttons_row,
            sizing_mode="stretch_both",
            min_height=0,
            margin=(0, 0, 0, 0),
        )
        return self.row