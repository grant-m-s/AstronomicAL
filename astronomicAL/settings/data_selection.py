from astronomicAL.utils.optimise import optimise
from astropy.table import Table
from bokeh.models import TextAreaInput
from bokeh.models.callbacks import CustomJS
from functools import partial

import astronomicAL.config as config

import glob
import json
import numpy as np
import pandas as pd
import panel as pn
import param
import time


class DataSelection(param.Parameterized):
    """The Data Selection Stage used in the settings pipeline.

    Parameters
    ----------
    src : ColumnDataSource
        The shared data source which holds the current selected source.

    Attributes
    ----------
    dataset : FileSelector
        Dropdown list of .FITS files within `data/` folder.
    memory_optimisation_check : Panel Checkbox
        Flag for whether to memory optimise the loaded in DataFrame.
    ready : bool
        Flag for whether the data has been loaded in and ready to progress to
        the next settings stage.

    """

    dataset = param.ObjectSelector(default="", objects=[""])
    config_file = param.ObjectSelector(default="", objects=[""])
    load_config_select = param.ObjectSelector(default="", objects=[""])

    load_layout_check = param.Boolean(False, label="Load Custom Configuration?")

    ready = param.Boolean(default=False)

    def __init__(self, src, mode):
        super(DataSelection, self).__init__()

        self.mode = mode
        self.src = src
        self.error_message = ""

        self._initialise_widgets()

        self.panel_col = pn.Column("Loading...")

    def _initialise_widgets(self):

        data_files = self._get_data_files()

        if len(data_files) == 0:
            data_files = [""]

        config_files = self._get_config_files()
        load_config_options = self._init_load_config_options()

        config_files = [""] + config_files
        load_config_options = [""] + load_config_options

        self.param.dataset.objects = data_files
        self.param.load_config_select.objects = load_config_options
        self.param.config_file.objects = config_files
        self.param.dataset.default = data_files[0]
        self.param.load_config_select.default = load_config_options[0]
        self.param.config_file.default = config_files[0]
        self.dataset = data_files[0]
        self.load_config_select = load_config_options[0]
        self.config_file = config_files[0]

        print("CONFIG:")
        print(config.layout_file)
        self.memory_optimisation_check = pn.widgets.Checkbox(
            name="Optimise for memory?", value=True
        )
        self._memory_opt_tooltip = pn.pane.HTML(
            "<span data-toggle='tooltip' title='(up to 0.5x memory consumption, however initial loading of data can take up to 10x longer).' styles='border-radius: 15px;padding: 5px; background: #5e5e5e; ' >❔</span> ",
            max_width=5,
        )

        self.load_data_button = pn.widgets.Button(
            name="Load Data File", max_height=30, margin=(45, 0, 0, 0)
        )
        self.load_data_button.on_click(self._load_data_cb)

        self.load_data_button_js = pn.widgets.Button(
            name="Select values from dropdown to continue",
            max_height=30,
            margin=(45, 0, 0, 0),
            button_type="success",
            disabled=True,
        )

        self.load_data_button_js.jscallback(
            clicks="""

                        console.log('Reloading...')

                        button.label = 'Loading New Layout - Please Wait, This may take a minute.'
                        button.disabled = true

                        var i;
                        for (i = 0; i < 40; i++) {
                          if (i % 3 === 0) {
                              setTimeout(() => {  button.label = 'Loading New Layout - Please Wait, This may take a minute.'; }, 500*i);
                           } else if (i % 3 === 1) {
                              setTimeout(() => {  button.label = 'Loading New Layout - Please Wait, This may take a minute..'; }, 500*i);
                           } else {
                              setTimeout(() => {  button.label = 'Loading New Layout - Please Wait, This may take a minute...'; }, 500*i);
                           }
                        }
                        setTimeout(() => {  location.reload(); }, 2000);
                    """,
            args=dict(button=self.load_data_button_js),
        )

    def _get_data_files(self):
        files = glob.glob("data/*.*")
        return files

    def _get_config_files(self):
        files = glob.glob("configs/*.json")
        return files

    def _init_load_config_options(self):
        if self.mode == "AL":
            options = [
                "Only load layout. Let me choose all my own settings",
                "Load all settings but let me train the model from scratch.",
                "Load all settings and train model with provided labels.",
            ]

        elif self.mode == "Labelling":
            options = [
                "Only load layout. Let me choose all my own settings",
                "Load all settings and begin labelling data.",
            ]
        else:
            options = []

        return options

    @param.depends("load_config_select", "config_file", watch=True)
    def _update_layout_file_cb(self):

        if (self.load_config_select == "") or (self.config_file == ""):
            self.error_message = ""
            self.load_data_button_js.name = "Select values from dropdown to continue"
            self.load_data_button_js.disabled = True
            return

        self.load_data_button_js.name = "Verifying Config..."

        config.layout_file = self.config_file
        config.settings["config_load_level"] = (
            list(self.param.load_config_select.objects).index(self.load_config_select)
            - 1
        )

        with open(config.layout_file) as layout_file:
            curr_config_file = json.load(layout_file)

        from astronomicAL.utils.load_config import (
            verify_import_config,
        )  # causes circular import error at top

        has_error, error_message = verify_import_config(curr_config_file)

        if has_error:
            print(f"has error - {error_message}")
            self.error_message = error_message
            self.load_data_button_js.name = "Unable to load config"
            self.load_data_button_js.disabled = True
        else:
            self.error_message = "verified"
            self.error_message = ""
            self.load_data_button_js.name = "Load Data"
            self.load_data_button_js.disabled = False

        self.panel_col = self.panel()

        print(f"Config load level: {config.settings['config_load_level']}")

    def get_dataframe_from_fits_file(self, filename, optimise_data=None):
        """Load data from FITS file into dataframe.

        Parameters
        ----------
        filename : str
            Path of file to be loaded.

        Returns
        -------
        df : DataFrame
            DataFrame containing the loaded in data from `filename`.

        """
        ext = filename[filename.rindex(".") + 1 :]
        fits_table = Table.read(filename, format=f"{ext}")

        names = [
            name for name in fits_table.colnames if len(fits_table[name].shape) <= 1
        ]

        if optimise_data is None:
            config.settings["optimise_data"] = self.memory_optimisation_check.value
            if self.memory_optimisation_check.value:
                approx_time = np.ceil(
                    (np.array(fits_table).shape[0] * len(names)) / 200000000
                )
                self.load_data_button.name = (
                    f"Optimising... Approx time: {int(approx_time)} minute(s)"
                )
                df = optimise(fits_table[names].to_pandas())
            else:
                df = fits_table[names].to_pandas()
        elif optimise_data:
            approx_time = np.ceil(
                (np.array(fits_table).shape[0] * len(names)) / 200000000
            )
            self.load_data_button.name = (
                f"Optimising... Approx time: {int(approx_time)} minute(s)"
            )
            df = optimise(fits_table[names].to_pandas())
        else:
            df = fits_table[names].to_pandas()

        if ext == "fits":
            for col, dtype in df.dtypes.items():
                if dtype == object:  # Only process byte object columns.
                    df[col] = df[col].apply(lambda x: x.decode("utf-8"))

        df = self.add_ra_dec_col(df)

        return df

    def _load_data_cb(self, event):
        self.load_data_button.disabled = True
        self.load_data_button.name = "Loading File..."
        print("loading new dataset")

        config.settings["dataset_filepath"] = self.dataset

        print(config.settings)

        config.main_df = self.get_dataframe_from_fits_file(self.dataset)
        self.df = config.main_df
        self.src.data = dict(pd.DataFrame())

        print(f" dataset shape: {self.df.shape}")

        self._initialise_src()
        self.ready = True
        self.load_data_button.name = "File Loaded."

    def add_ra_dec_col(self, df):

        new_df = df

        #we should add this option at some point in the config/settings pipeline
        ra_col_name = config.settings.get("ra_col_name", "ra")
        dec_col_name = config.settings.get("dec_col_name", "dec")

        has_loc = (ra_col_name in list(new_df.columns)) and (dec_col_name in list(new_df.columns))
 
        if has_loc:
           new_df["ra_dec"] = df[ra_col_name].astype(str) + "," + df[dec_col_name].astype(str)
        else:
            print("Columns selected as Ra and Dec are not in the table")

        return new_df

    def get_df(self):
        """Return the dataframe that has been loaded in from a file.

        Returns
        -------
        df : DataFrame
            DataFrame containing the dataset loaded in by a file chosen by the
            user.

        """
        return self.df

    def _initialise_src(self):

        new_df = pd.DataFrame([[0, 0], [0, 0]], columns=["test", "test"])
        self.src.data = dict(new_df)

    @param.depends("load_layout_check", watch=True)
    def _panel_cb(self):
        self.panel_col = self.panel()

    def panel(self):
        """Render the current settings view.

        Returns
        -------
        column : Panel Column
            The panel is housed in a column which can then be rendered by the
            settings Dashboard.

        """
        if (self.error_message == "") or (self.error_message == "verified"):
            message = pn.pane.Markdown(
                """
                Welcome to AstronomicAL, an interactive dashboard for visualisation, integration and classification of data using active learning methods.

                For tutorials and API reference documents, please visit our documentation [here](https://astronomical.readthedocs.io).

                AstronomicAL provides both an example dataset and an example configuration file to allow you to jump right into the software and give it a test run.

                To begin training you simply have to select **Load Custom Configuration** checkbox and select your config file. Here we have chosen to use the `example_config.json` file.

                The **Load Config Select** option allows use to choose the extent to which to reload the configuration.

                #### Referencing the Package

                Please remember to cite our software and user guide whenever relevant. See the [Citing page](https://astronomical.readthedocs.io/en/latest/content/other/citing.html) in the documentation for instructions about referencing and citing the astronomicAL software.
                """,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )
        else:
            message = pn.pane.Markdown(self.error_message)

        if self.load_layout_check:
            self.panel_col[0] = pn.Row(
                pn.Column(
                    pn.layout.VSpacer(max_height=10),
                    pn.Row(self.param.load_layout_check, max_height=30),
                    pn.Row(self.param.config_file, max_width=300),
                    pn.Row(self.param.load_config_select),
                    pn.Row(self.load_data_button_js),
                    pn.layout.VSpacer(),
                    pn.layout.VSpacer(),
                    pn.layout.VSpacer(),
                ),
                pn.Row(message, scroll=True, sizing_mode="stretch_width"),
                pn.layout.VSpacer(),
                pn.layout.VSpacer(),
                pn.layout.VSpacer(),
                sizing_mode="stretch_width",
            )

        else:
            self.panel_col[0] = pn.Row(
                pn.Column(
                    pn.layout.VSpacer(max_height=10),
                    pn.Row(self.param.load_layout_check, max_height=30),
                    pn.Row(self.param.dataset, max_width=300),
                    pn.Row(
                        self.memory_optimisation_check,
                        self._memory_opt_tooltip,
                        max_width=300,
                    ),
                    pn.Row(self.load_data_button, max_width=300),
                    pn.layout.VSpacer(max_width=300),
                    pn.layout.VSpacer(max_width=300),
                    pn.layout.VSpacer(max_width=300),
                ),
                pn.Row(
                    pn.pane.Markdown(
                        """
                    Welcome to AstronomicAL, an interactive dashboard for visualisation, integration and classification of data using active learning methods.

                    For tutorials and API reference documents, please visit our documentation [here](https://astronomical.readthedocs.io).

                    AstronomicAL provides both an example dataset and an example configuration file to allow you to jump right into the software and give it a test run.

                    To begin training you simply have to select **Load Custom Configuration** checkbox and select your config file. Here we have chosen to use the `example_config.json` file.

                    The **Load Config Select** option allows use to choose the extent to which to reload the configuration.

                    #### Referencing the Package

                    Please remember to cite our software and user guide whenever relevant. See the [Citing page](https://astronomical.readthedocs.io/en/latest/content/other/citing.html) in the documentation for instructions about referencing and citing the astronomicAL software.
                    """,
                        sizing_mode="stretch_width",
                        margin=(0, 0, 0, 0),
                    )
                ),
            )

        return self.panel_col
