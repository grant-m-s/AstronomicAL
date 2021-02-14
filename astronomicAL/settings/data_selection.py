from astropy.table import Table
import numpy as np
import pandas as pd
import panel as pn

import config
import param
import time


class DataSelection(param.Parameterized):

    dataset = param.FileSelector(path="data/*.fits")

    memory_optimisation_check = pn.widgets.Checkbox(
        name="Optimise for memory?", value=True
    )
    memory_opt_tooltip = pn.pane.HTML(
        "<span data-toggle='tooltip' title='(up to 0.5x memory consumption, however initial loading of data can take up to 10x longer).' style='border-radius: 15px;padding: 5px; background: #5e5e5e; ' >❔</span> ",
        max_width=5,
    )
    ready = param.Boolean(default=False)

    def __init__(self, src, **params):
        super(DataSelection, self).__init__(**params)
        self.src = src

        self.load_data_button = pn.widgets.Button(
            name="Load File", max_height=30, margin=(45, 0, 0, 0)
        )
        self.load_data_button.on_click(self.load_data_cb)

########################### dataframe optimisations ###########################
    # Following optimisation functions credited to:
    # https://medium.com/bigdatarepublic/advanced-pandas-optimize-speed-and-memory-a654b53be6c2
    def optimise_floats(self, df):
        start = time.time()
        floats = df.select_dtypes(include=['float64']).columns.tolist()
        df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
        end = time.time()
        print(f"optimise_floats {end - start}")
        return df

    def optimise_ints(self, df):
        start = time.time()
        ints = df.select_dtypes(include=['int64']).columns.tolist()
        df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
        end = time.time()
        print(f"optimise_ints {end - start}")
        return df

    def optimise_objects(self, df):
        start = time.time()
        for col in df.select_dtypes(include=['object']):
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        end = time.time()
        print(f"optimise objects {end - start}")
        return df

    def optimise(self, df):
        return self.optimise_floats(self.optimise_ints(self.optimise_objects(df)))

################################################################################

    def get_dataframe_from_fits_file(self, filename):
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
            df = self.optimise(fits_table[names].to_pandas())
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

    def load_data_cb(self, event):
        self.load_data_button.disabled = True
        self.load_data_button.name = "Loading File..."
        print("loading new dataset")

        config.main_df = self.get_dataframe_from_fits_file(self.dataset)
        self.df = config.main_df
        self.src.data = dict(pd.DataFrame())

        print(f" dataset shape: {self.df.shape}")

        self.update_src(self.df)
        self.ready = True
        self.load_data_button.name = "File Loaded."

    def get_df(self):
        return self.df

    def update_src(self, df):
        new_df = pd.DataFrame([[0, 0], [0, 0]], columns=["test", "test"])
        self.src.data = dict(new_df)

    def panel(self):
        return pn.Column(pn.Row(self.param.dataset, self.load_data_button, max_width=300), pn.Row(pn.Row(self.memory_optimisation_check), self.memory_opt_tooltip, max_width=300))
