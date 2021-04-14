from astronomicAL.utils.optimise import optimise
from astropy.table import Table
from bokeh.models import TextAreaInput
from bokeh.models.callbacks import CustomJS

import astronomicAL.config as config
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

    dataset = param.FileSelector(path="data/*.fits")
    config_file = param.FileSelector(path="configs/*.json")

    load_layout_check = param.Boolean(False, label="Load Custom Configuration?")

    ready = param.Boolean(default=False)

    def __init__(self, src, mode):
        super(DataSelection, self).__init__()

        self.mode = mode
        self.src = src

        self._initialise_widgets()

        self.panel_col = pn.Column("Loading...")

    def _initialise_widgets(self):
        print("CONFIG:")
        print(config.layout_file)
        self.memory_optimisation_check = pn.widgets.Checkbox(
            name="Optimise for memory?", value=True
        )
        self._memory_opt_tooltip = pn.pane.HTML(
            "<span data-toggle='tooltip' title='(up to 0.5x memory consumption, however initial loading of data can take up to 10x longer).' style='border-radius: 15px;padding: 5px; background: #5e5e5e; ' >❔</span> ",
            max_width=5,
        )

        if self.mode == "AL":
            self.load_config_select = pn.widgets.Select(
                name="How Much Would You Like To Load?",
                options=[
                    "Only load layout. Let me choose all my own settings",
                    "Load all settings but let me train the model from scratch.",
                    "Load all settings and train model with provided labels.",
                ],
                max_width=400,
            )

        elif self.mode == "Labelling":
            self.load_config_select = pn.widgets.Select(
                name="How Much Would You Like To Load?",
                options=[
                    "Only load layout. Let me choose all my own settings",
                    "Load all settings and begin labelling data.",
                ],
                max_width=400,
            )

        self.load_data_button = pn.widgets.Button(
            name="Load Data File", max_height=30, margin=(45, 0, 0, 0)
        )
        self.load_data_button.on_click(self._load_data_cb)

        self.load_data_button_js = pn.widgets.Button(
            name="Load Configuration File", max_height=30, margin=(45, 0, 0, 0)
        )

        self.load_data_button_js.js_on_click(
            args={"button": self.load_data_button_js},
            code="""
                console.log('Reloading...')
                button.label = 'Loading New Layout - Please Wait, This may take a minute.'
                button.disabled = true

                var i;
                for (i = 0; i < 25; i++) {
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
        )

        self.load_data_button_js.on_click(self._update_layout_file_cb)

    def _update_layout_file_cb(self, event):
        config.layout_file = self.config_file
        config.settings["config_load_level"] = list(
            self.load_config_select.options
        ).index(self.load_config_select.value)
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
        fits_table = Table.read(filename, format="fits")
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
                df = optimise(fits_table[names].to_pandas())
            else:
                df = fits_table[names].to_pandas()
        elif optimise_data:
            df = optimise(fits_table[names].to_pandas())
        else:
            df = fits_table[names].to_pandas()
        end = time.time()
        print(f"Convert to Pandas {end - start}")

        start = time.time()
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
        if self.load_layout_check:
            self.panel_col[0] = pn.Column(
                pn.layout.VSpacer(max_height=10),
                pn.Row(self.param.load_layout_check, max_height=30),
                pn.Row(self.param.config_file, max_width=300),
                pn.Row(self.load_config_select),
                pn.Row(self.load_data_button_js),
                pn.layout.VSpacer(),
                pn.layout.VSpacer(),
                pn.layout.VSpacer(),
            )

        else:
            self.panel_col[0] = pn.Column(
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
            )

        return self.panel_col
