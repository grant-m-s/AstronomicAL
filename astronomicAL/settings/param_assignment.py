from bokeh.models import ColumnDataSource
import panel as pn
import param

from astronomicAL.utils import load_config
from astronomicAL.utils.optimise import matches_type, get_series_type


class ParameterAssignment(param.Parameterized):
    """The Parameter Assignment Stage used in the settings pipeline.

    Parameters
    ----------
    df : DataFrame
        The shared dataframe which holds all the data.

    Attributes
    ----------
    df : DataFrame
        The shared dataframe which holds all the data.
    label_column : Panel ObjectSelector
        Dropdown for choosing which of the dataset columns is the label column.
    id_column : Panel ObjectSelector
        Dropdown for choosing which of the dataset columns is the id column.
    completed : bool
        Flag indicating all active learning settings have been chosen and
        assigned.

    colours_param : dict
        Dictionary holding the colours to render each of the labels.
    label_strings_param : dict
        Dictionary holding the string aliases of the labels that are displayed
        throughout the ui.

    """

    label_column = param.ObjectSelector(objects=["default"], default="default")
    id_column = param.ObjectSelector(objects=["default"], default="default")

    completed = param.Boolean(
        default=False,
    )

    initial_update = True



    colours_param = {}

    label_strings_param = {}

    def __init__(self, mode = "Labelling", context = None):
        super(ParameterAssignment, self).__init__()

        self.context = context

        self.column = pn.Column(pn.pane.Str("loading"), sizing_mode = "stretch_both")
        self.src = ColumnDataSource()
        self.df = None
        self.mode = mode

    def update_data(self, dataframe=None):
        """Update the local copy of the data and update widgets accordingly.

        Parameters
        ----------
        dataframe : DataFrame, default = None
            The updated data to be used.

        Returns
        -------
        None

        """
        if dataframe is not None:
            self.df = dataframe

        if (self.initial_update) and (self.df is not None):
            cols = list(self.df.columns)

            if cols == []:
                return
            self.initial_update = False
        
        if self.mode.lower() == "exploring":
            id_cols = ["Use Index"] + cols
            label_cols = ["No Labels"] + cols
        else:
            id_cols = cols
            label_cols = [col for col in cols if matches_type(self.df[col].dtype, ["int"])]

        self.param.id_column.objects = id_cols
        self.param.id_column.default = id_cols[0]
        self.id_column = id_cols[0]

        self.param.label_column.objects = label_cols
        self.param.label_column.default = label_cols[0]
        self.label_column = label_cols[0]


    def _confirm_settings_cb(self, event):
        print("Saving settings...")
        if (self.label_column == "") or (self.id_column == ""):
            print("Select valid label and ID columns")
            return
        self.confirm_settings_button.name = "Assigning parameters..."
        self.confirm_settings_button.disabled = True

        updated_settings = self.get_settings()
        for key in updated_settings.keys():
            self.config.settings[key] = updated_settings[key]
        self.config.settings["confirmed"] = False
        self.confirm_settings_button.name = "Confirmed"


    @param.depends("label_column", watch=True)
    def _update_labels_cb(self):
        """Update label settings when the user changes the label column.

        Returns
        -------
        None

        """
        self.label_strings_param = {}
        self.colours_param = {}

        if (self.label_column not in self.df.columns) or (self.label_column == "No Labels"):
            self.labels = []
            self.config.settings["labels"] = self.labels

        else:
            self.label_type = get_series_type(self.df[self.label_column])
            if self.label_type == "mixed": # strings and numbers are in the column
                self.df[self.label_column] = self.df[self.label_column].astype(str)
                self.label_type = "string"
            elif self.label_type == "bool":
                self.df[self.label_column] = self.df[self.label_column].astype(int)
                self.label_type = "int"
            if len(self.df[self.label_column].unique()) > 20:
                print(
                """You have chosen a column with too many unique values (possibly continous) please choose a column with a smaller set of labels (<=20)"""
                )
                self.panel()
                return
            
            self.labels = sorted(self.df[self.label_column].unique())
            self.config.settings["labels"] = self.labels
            self.update_colours()
            self._initialise_label_strings_input()

        self.panel()

    def update_colours(self):
        """Update the colours used for rendering.

        Returns
        -------
        None

        """
        print("updating colours...")
        labels = self.config.settings["labels"]

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
                name=f"{data_label}",
                value=colour_list[(i % 10)],
                max_width=int(500 / len(labels)),
            )

    def _initialise_label_strings_input(self):
        labels = self.config.settings["labels"]
        for i, data_label in enumerate(labels):
            self.label_strings_param[f"{data_label}"] = pn.widgets.TextInput(
                name=f"{data_label}", placeholder=f"{data_label}"
            )

    def get_id_column(self):
        """Return the name of the id column.

        Returns
        -------
        id_column : str
            The column name of the id column of the data.

        """
        return self.id_column

    def get_label_column(self):
        """Return the name of the column containing the labels in the data.

        Returns
        -------
        label_column : str
            The name of the column containing the labels in the data.

        """
        return self.label_column

    def get_label_colours(self):
        """Return the colours chosen to represent labels in plots.

        Returns
        -------
        colours : dict
            Dictionary containing the labels as keys and the corresponding
            colours as values.

        """
        colours = {key : param_obj.value for key, param_obj in self.colours_param.items()}

        return colours
    
    def get_df(self):
        """Return the dataframe that has been loaded in from a file.

        Returns
        -------
        df : DataFrame
            DataFrame containing the dataset loaded in by a file chosen by the
            user.

        """
        return self.df

    def get_label_strings(self):
        """Return the string aliases and the corresponding conversions of the
        labels.

        Returns
        -------
        labels_to_strings : dict
            Dictionary containing labels as keys and the corresponding aliases
            as values.
        strings_to_labels : dict
            Dictionary containing aliases as keys and the corresponding labels
            as values.
        """
        labels_to_strings = {}
        strings_to_labels = {}

        for label in self.labels:
            value = self.label_strings_param[f"{label}"].value
            if value == "":
                value = str(label)
            labels_to_strings[f"{label}"] = value
            strings_to_labels[f"{value}"] = label

        return labels_to_strings, strings_to_labels
    
    def get_settings(self):
        """Return all the saved parameter assignment settings.

        Returns
        -------
        updated_settings : dict
            Dictionary containing the setting saved throughout the parameter
            assignment stage.

        """

        updated_settings = {}
        updated_settings["id_col"] = self.get_id_column()
        updated_settings["label_col"] = self.get_label_column()
        updated_settings["labels"] = self.labels
        updated_settings["label_colours"] = self.get_label_colours()
        (
            updated_settings["labels_to_strings"],
            updated_settings["strings_to_labels"],
        ) = self.get_label_strings()


        assert updated_settings is not None, f"\n Updated Settings in settings/Param_assignment.py is None..."

        return updated_settings


class ParameterAssignment_ML(ParameterAssignment):
    """    
    Attributes:
    ready : bool
        Flag for whether the data has been loaded in and ready to progress to
        the next settings stage."""

    ready = param.Boolean(default=False)

    def __init__(self, context = None):
        super(ParameterAssignment_ML, self).__init__(mode = "Labelling")

        self.context = context
        self.config = context.config

        self._initialise_widgets()

    def _initialise_widgets(self):
        self.confirm_settings_button = pn.widgets.Button(
            name="Confirm Settings",
            max_height=30,
            margin=(25, 0, 0, 0),
            button_type="primary",
        )
        self.confirm_settings_button.on_click(self._confirm_settings_cb)

        self.extra_info_selector = pn.widgets.MultiChoice(
            name="Columns shown when inspecting a source:",
            value=[],
            options=[],
            max_width=300,
            sizing_mode = "stretch_both",
        )

        self.extra_images_selector = pn.widgets.MultiChoice(
            name="Columns containing image URLs shown when inspecting a source:",
            value=[],
            options=[],
            max_width=300,
            sizing_mode = "stretch_both",
        )

    def _confirm_settings_cb(self, event):
        super()._confirm_settings_cb(event)
        self.ready = True

    def get_settings(self):
        updated_settings = super().get_settings()
        updated_settings["extra_info_cols"] = self.extra_info_selector.value
        updated_settings["extra_image_cols"] = self.extra_images_selector.value
        return updated_settings

    def update_data(self, dataframe=None):
        super().update_data(dataframe)
        if self.df is not None:
            cols = list(self.df.columns)
            self.extra_info_selector.options = cols
            self.extra_images_selector.options = cols


    @param.depends("completed", watch=True)
    def panel(self):
        """Render the current settings view.

        Returns
        -------
        column : Panel Column
            The panel is housed in a column which can then be rendered by the
            settings Dashboard.

        """

        if self.completed:
            self.column[0] = pn.pane.Str("Settings Saved.")
        else:
            layout = pn.Column(
                pn.Row(
                    pn.Row(self.param.id_column, max_width=150, max_height = 30),
                    pn.Row(self.param.label_column, max_width=150, max_height = 30),
                    max_width=600,
                ),
                sizing_mode = "stretch_both",
                min_height = 700,
                scroll = True,
            )

            if len(self.colours_param.keys()) > 0:
                colour_row = pn.Row(
                    pn.pane.Markdown(
                        "**Choose colours:**", max_height=50, max_width=150
                    ),
                    max_width=750,
                )
                for widget in self.colours_param:
                    colour_row.append(self.colours_param[widget])

                label_strings_row = pn.Row(
                    pn.pane.Markdown(
                        "**Custom label names:**", max_height=50, max_width=150
                    ),
                    max_width=750,
                )

                for widget in self.label_strings_param:
                    label_strings_row.append(self.label_strings_param[widget])

                layout.append(colour_row)

                layout.append(label_strings_row)
            
            layout.append(
                    pn.pane.Markdown(
                        "**Choose which extra information you want to view when inspecting each source:**",
                        margin=0,
                        max_height=20,
                        max_width=300,
                    )
                )
            layout.append(pn.Spacer(height=50))
            layout.append(
                pn.layout.Tabs(
                    (
                    "Extra Table Data",
                    self.extra_info_selector,
                    ),
                    (
                    "Extra Image Data",
                    self.extra_images_selector,
                    ),
                    tabs_location="left",
                    )
                )

            layout.append(pn.Spacer(height=30))

            layout.append(
                pn.Row(
                    self.confirm_settings_button,
                    )
                )

            self.column[0] = layout

        return self.column
    

class ParameterAssignment_Exploring(ParameterAssignment):
    
    ra_column = param.ObjectSelector(objects=["default"],  default="default")
    dec_column = param.ObjectSelector(objects=["default"],  default="default")

    labels = []


    def __init__(self, close_settings_button, context = None):
        super(ParameterAssignment_Exploring, self).__init__(mode = "Exploring")

        self.context = context
        self.config = context.config

        self.close_settings_button = close_settings_button
        self._initialise_widgets()
    
    
    
    def _initialise_widgets(self):
        self.confirm_settings_button = pn.widgets.Button(
            name="Confirm Settings",
            max_height=30,
            margin=(25, 0, 0, 0),
            button_type="primary",
        )
        self.confirm_settings_button.on_click(self._confirm_settings_cb)
 
    def update_data(self, dataframe=None):
        super().update_data(dataframe)
        if self.df is not None:
            coords_columns = [col for col in self.df.columns if matches_type(self.df[col].dtype, ["float"])]
            self.param.ra_column.objects  = coords_columns
            self.param.dec_column.objects = coords_columns
            self.param.ra_column.default = coords_columns[0]
            self.ra_column = coords_columns[0]
            self.param.dec_column.default = coords_columns[0]
            self.dec_column = coords_columns[0]
    
    
    def get_settings(self):
        updated_settings = super().get_settings()
        updated_settings["ra_col_name"] = self.ra_column
        updated_settings["dec_col_name"] = self.dec_column
        return updated_settings


    def _confirm_settings_cb(self, event):
        super()._confirm_settings_cb(event)
        self.config.settings["confirmed"] = True
        
        for save_name in ["save_button", "save_panel_button", "save_logbook_button"]:
            if save_name in self.config.settings:
                self.config.settings[save_name].disabled = False

        self.close_settings_button.disabled = False
        self.close_settings_button.button_type = "success"

    @param.depends("completed", watch=True)
    def panel(self):
        """Render the current settings view.

        Returns
        -------
        column : Panel Column
            The panel is housed in a column which can then be rendered by the
            settings Dashboard.

        """

        if self.completed:
            self.column[0] = pn.pane.Str("Settings Saved.")
        else:
            layout = pn.Column(
                pn.Row(
                    self.param.id_column,
                    self.param.label_column,
                    max_width=600,
                ),
                sizing_mode = "stretch_both",
            )

            if len(self.colours_param.keys()) > 0:
                colour_row = pn.Row(
                    pn.pane.Markdown(
                        "**Choose colours:**", max_height=50, max_width=150
                    ),
                    max_width=750,
                )
                for widget in self.colours_param:
                    colour_row.append(self.colours_param[widget])

                label_strings_row = pn.Row(
                    pn.pane.Markdown(
                        "**Custom label names:**", max_height=50, max_width=150
                    ),
                    max_width=750,
                )

                for widget in self.label_strings_param:
                    label_strings_row.append(self.label_strings_param[widget])

                layout.append(colour_row)

                layout.append(label_strings_row)
            layout.append(pn.Spacer(height=50))
            layout.append(
                pn.Row(self.param.ra_column,
                    self.param.dec_column,
                    max_width=600,
                    ),
            )
            layout.append(pn.Spacer(height=30))

            layout.append(
                pn.Row(
                    self.confirm_settings_button,
                    )
                )

            self.column[0] = layout

        return self.column   