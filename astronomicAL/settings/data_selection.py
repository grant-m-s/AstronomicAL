from astronomicAL.utils.optimise import optimise
from astronomicAL.utils import load_config
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

    dataset = param.FileSelector(path="data/*")
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

        config_files = self._get_config_files()
        load_config_options = self._init_load_config_options()

        config_files = [""] + config_files
        load_config_options = [""] + load_config_options

        self.param.load_config_select.objects = load_config_options
        self.param.config_file.objects = config_files
        self.param.load_config_select.default = load_config_options[0]
        self.param.config_file.default = config_files[0]
        self.load_config_select = load_config_options[0]
        self.config_file = config_files[0]

        print("CONFIG:")
        print(config.layout_file)
        self.memory_optimisation_check = pn.widgets.Checkbox(
            name="Optimise for memory?", value=True
        )
        self._memory_opt_tooltip = pn.pane.HTML(
            "<span data-toggle='tooltip' title='(up to 0.5x memory consumption, however initial loading of data can take up to 10x longer).' style='border-radius: 15px;padding: 5px; background: #5e5e5e; ' >❔</span> ",
            max_width=5,
        )

        self.load_data_button = pn.widgets.Button(
            name="Load Data File", max_height=30, margin=(45, 0, 0, 0)
        )
        self.load_data_button.on_click(self._load_data_cb)

        self.load_data_button_js = pn.widgets.Button(
            name="Load Configuration File",
            max_height=30,
            margin=(45, 0, 0, 0),
            button_type="success",
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
            self.load_data_button_js.label = "Select values from dropdown to continue"
            self.load_data_button_js.disabled = True
            return

        config.layout_file = self.config_file
        config.settings["config_load_level"] = (
            list(self.param.load_config_select.objects).index(self.load_config_select)
            - 1
        )

        with open(config.layout_file) as layout_file:
            curr_config_file = json.load(layout_file)

        has_error, error_message = load_config.verify_import_config(curr_config_file)

        if has_error:
            print(f"has error - {error_message}")
            self.error_message = error_message
            self.load_data_button_js.label = "Unable to load config"
            self.load_data_button_js.disabled = True
        else:
            print(f"no error - {error_message}")
            self.error_message = "verified"
            self.error_message = ""
            self.load_data_button_js.label = "Load Data"
            self.load_data_button_js.disabled = False

        print(f"Config load level: {config.settings['config_load_level']}")
        print(f"INSIDE CB: {config.layout_file}")

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
        start = time.time()

        ext = filename[filename.rindex(".") + 1 :]
        fits_table = Table.read(filename, format=f"{ext}")

        end = time.time()
        print(f"Loading FITS Table {end - start}")
        start = time.time()
        names = [
            name for name in fits_table.colnames if len(fits_table[name].shape) <= 1
        ]
        end = time.time()
        print(f"names list {end - start}")

        start = time.time()
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
        end = time.time()
        print(f"Convert to Pandas {end - start}")

        start = time.time()
        if ext == "fits":
            for col, dtype in df.dtypes.items():
                if dtype == np.object:  # Only process byte object columns.
                    df[col] = df[col].apply(lambda x: x.decode("utf-8"))
        end = time.time()
        print(f"Pandas object loop {end - start}")

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

    def _panel_change_cb(self, attr, old, new):
        print("PANEL CB RAN")
        self.panel_col = self.panel()

    @param.depends("load_layout_check", watch=True)
    def _panel_cb(self):
        print("PANEL CB RAN")
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
            message = "Welcome!"
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
                pn.Row(message),
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
                    max_width=300,
                ),
                pn.Row("Welcome!"),
            )

        return self.panel_col
