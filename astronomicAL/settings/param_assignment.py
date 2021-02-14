from bokeh.models import ColumnDataSource

import config
import panel as pn
import param


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
        self.src = ColumnDataSource()

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
        config.settings["labels"] = self.labels

        self.update_colours()
        self.update_label_strings()
        self.panel()

    def update_colours(self):
        print("updating colours...")
        labels = config.settings["labels"]

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
        labels = config.settings["labels"]
        for i, data_label in enumerate(labels):
            self.label_strings_param[f"{data_label}"] = pn.widgets.TextInput(
                name=f"{data_label}", placeholder=f"{data_label}")

    def confirm_settings_cb(self, event):
        print("Saving settings...")
        self.confirm_settings_button.name = "Assigning parameters..."
        self.confirm_settings_button.disabled = True

        config.settings = self.get_settings()

        labels = self.df[config.settings["label_col"]]

        config.settings["extra_info_cols"] = self.extra_info_selector.value
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

        for label in config.settings["labels"]:
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
