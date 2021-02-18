from astropy.table import Table
from utils.optimise import optimise

import config
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

    ready = param.Boolean(default=False)

    def __init__(self, src, **params):
        super(DataSelection, self).__init__(**params)

        self._initialise_widgets()
        self.src = src

    def _initialise_widgets(self):
        self.memory_optimisation_check = pn.widgets.Checkbox(
            name="Optimise for memory?", value=True
        )
        self._memory_opt_tooltip = pn.pane.HTML(
            "<span data-toggle='tooltip' title='(up to 0.5x memory consumption, however initial loading of data can take up to 10x longer).' style='border-radius: 15px;padding: 5px; background: #5e5e5e; ' >❔</span> ",
            max_width=5,
        )

        self.load_data_button = pn.widgets.Button(
            name="Load File", max_height=30, margin=(45, 0, 0, 0)
        )
        self.load_data_button.on_click(self._load_data_cb)

    def get_dataframe_from_fits_file(self, filename):
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
        names = [name for name in fits_table.colnames if len(
            fits_table[name].shape) <= 1]
        end = time.time()
        print(f"names list {end - start}")

        start = time.time()
        if self.memory_optimisation_check.value:
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

    def panel(self):
        """Render the current settings view.

        Returns
        -------
        column : Panel Column
            The panel is housed in a column which can then be rendered by the
            settings Dashboard.

        """
        return pn.Column(
            pn.Row(self.param.dataset, self.load_data_button, max_width=300),
            pn.Row(
                pn.Row(self.memory_optimisation_check),
                self._memory_opt_tooltip,
                max_width=300)
            )
