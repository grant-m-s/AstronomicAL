from astronomicAL.utils.optimise import (
    PrintProgress,
    optimise_streaming,
    _fits_to_df_streaming,
    _decode_bytes_columns,
    df_mem_gib,
    rss_gib,
)
from astropy.table import Table

import glob
import json
import os
import pandas as pd
import panel as pn
import param


class DataSelection(param.Parameterized):
    """The Data Selection Stage used in the settings pipeline."""

    dataset = param.ObjectSelector(default="", objects=[""])
    config_file = param.ObjectSelector(default="", objects=[""])
    load_config_select = param.ObjectSelector(default="", objects=[""])
    load_layout_check = param.Boolean(False, label="Load Custom Configuration?")
    ready = param.Boolean(default=False)

    def __init__(self, src, mode, context, close_settings_button):
        super(DataSelection, self).__init__()

        self.context = context
        self.mode = mode
        self.src = src
        self.error_message = ""
        self.close_settings_button = close_settings_button
        self.df = pd.DataFrame()

        if context is not None and getattr(context, "config", None) is not None:
            self.config = context.config
        else:
            print("config doesn't exist")
            print(context)
            self.config = None

        if self.config is not None and getattr(self.config, "settings", None) is None:
            self.config.settings = {}

        self._initialise_widgets()

        self.view = pn.Column(sizing_mode="stretch_width")
        self._refresh_layout()

    def _initialise_widgets(self):
        data_files = self._get_data_files()
        if not data_files:
            data_files = [""]

        config_files = self._get_config_files()
        load_config_options = self._init_load_config_options()

        config_files = [""] + config_files
        load_config_options = [""] + load_config_options

        self.param.dataset.objects = data_files
        self.param.config_file.objects = config_files
        self.param.load_config_select.objects = load_config_options

        self.dataset = data_files[0]
        self.config_file = config_files[0]
        self.load_config_select = load_config_options[0]

        self.dataset_widget = pn.widgets.Select(
            options=data_files,
            value=self.dataset,
            width=320,
            margin=0,
            height=34,
        )

        self.config_file_widget = pn.widgets.Select(
            options=config_files,
            value=self.config_file,
            width=320,
            margin=0,
            height=30,
        )

        self.load_config_select_widget = pn.widgets.Select(
            options=load_config_options,
            value=self.load_config_select,
            width=320,
            margin=0,
            height=30,
        )

        self.load_data_button = pn.widgets.Button(
            name="Load data file",
            button_type="primary",
            width=220,
            height=38,
            margin=0,
        )

        self.load_layout_widget = pn.widgets.Checkbox(
            name="Load custom configuration?",
            value=self.load_layout_check,
            margin=0,
            height=20,
        )

        self.load_data_button_js = pn.widgets.Button(
            name="Select values from dropdown to continue",
            button_type="primary",
            disabled=True,
            width=220,
            height=38,
            margin=0,
        )

        self.memory_optimisation_check = pn.widgets.Checkbox(
            name="Optimise for memory?",
            value=True,
            margin=(0, 0, 0, 0),
        )

        self._memory_opt_tooltip = pn.pane.HTML(
            """
            <span title="Up to around 0.5x memory consumption, but initial loading may take much longer."
                style="display:inline-block; border-radius:12px; padding:2px 6px; background:#5e5e5e; color:white; cursor:help;">
                ?
            </span>
            """,
            width=24,
            height=24,
            margin=(0, 0, 0, 1),
        )

        self.dataset_widget.param.watch(self._sync_dataset, "value")
        self.config_file_widget.param.watch(self._sync_config_file, "value")
        self.load_config_select_widget.param.watch(self._sync_load_config_select, "value")
        self.load_layout_widget.param.watch(self._sync_load_layout_check, "value")

        self.load_data_button.on_click(self._load_data_cb)

        self.load_data_button_js.jscallback(
            clicks="""
                button.label = 'Loading New Layout - Please Wait...'
                button.disabled = true
                setTimeout(() => { location.reload(); }, 2000);
            """,
            args=dict(button=self.load_data_button_js),
        )

    def _sync_load_layout_check(self, event):
        self.load_layout_check = event.new
        self._refresh_layout()

    def _sync_dataset(self, event):
        self.dataset = event.new

    def _sync_config_file(self, event):
        self.config_file = event.new
        self.update_available_loading_options()
        self._update_layout_file_cb()

    def _sync_load_config_select(self, event):
        self.load_config_select = event.new
        self._update_layout_file_cb()

    def _get_data_files(self):
        return glob.glob("data/*.*")

    def _get_config_files(self):
        files = glob.glob("configs/*.json")
        exploring_file = "configs/exploring_default.json"
        files = [f for f in files if f != exploring_file]
        if self.mode == "Exploring" and os.path.exists(exploring_file):
            files = [exploring_file] + sorted(files)
        return files

    def _init_load_config_options(self):
        if self.mode == "AL":
            return [
                "Only load layout. Let me choose all my own settings",
                "Load all settings but let me train the model from scratch.",
                "Load all settings and train model with provided labels.",
            ]
        elif self.mode == "Labelling":
            return [
                "Only load layout. Let me choose all my own settings",
                "Load all settings and begin labelling data.",
            ]
        elif self.mode == "Exploring":
            return [
                "Only load layout. Let me choose all my own settings",
                "Load all settings and begin exploring data.",
            ]
        return []

    def _update_layout_file_cb(self, *events):
        if not self.load_layout_check:
            self.error_message = ""
            self._refresh_layout()
            return

        if (self.load_config_select == "") or (self.config_file == ""):
            self.error_message = ""
            self.load_data_button_js.name = "Select values from dropdown to continue"
            self.load_data_button_js.disabled = True
            self._refresh_layout()
            return

        self.load_data_button_js.name = "Verifying Config..."

        self.config.layout_file = self.config_file
        self.config.settings["config_load_level"] = (
            list(self.load_config_select_widget.options).index(self.load_config_select) - 1
        )

        with open(self.config.layout_file) as layout_file:
            curr_config_file = json.load(layout_file)

        from astronomicAL.utils.load_config import verify_import_config

        has_error, error_message = verify_import_config(
            curr_config_file, context=self.context
        )

        if has_error:
            self.config.settings = {}
            self.error_message = error_message
            self.load_data_button_js.name = "Unable to load config"
            self.load_data_button_js.disabled = True
        else:
            self.error_message = ""
            self.load_data_button_js.name = "Load Data"
            self.load_data_button_js.disabled = False

        self._refresh_layout()

    def update_available_loading_options(self):
        if self.config_file.endswith("exploring_default.json"):
            options = ["", "Only load layout. Let me choose all my own settings"]
        else:
            options = [""] + self._init_load_config_options()

        current = self.load_config_select if self.load_config_select in options else options[0]
        self.load_config_select_widget.options = options
        self.load_config_select_widget.value = current
        self.param.load_config_select.objects = options
        self.load_config_select = current

    def get_dataframe_from_fits_file(self, filename, optimise_data=None):
        p = PrintProgress(every=2.0)
        ext = filename[filename.rindex(".") + 1 :].lower()

        if optimise_data is None:
            val = bool(self.memory_optimisation_check.value)
            self.config.settings["optimise_data"] = val
            optimise_data = val
        else:
            optimise_data = bool(optimise_data)

        p.log(f"Start load: ext={ext}, optimise_data={optimise_data}")

        if ext in ("fits", "fit", "fts"):
            df = _fits_to_df_streaming(filename, hdu=1, p=p, cast_float32=False)
            p.log(f"After read: mem≈{df_mem_gib(df):.2f} GiB, RSS≈{rss_gib():.2f} GiB")
            df = _decode_bytes_columns(df, p=p)
            p.log(f"After decode: mem≈{df_mem_gib(df):.2f} GiB, RSS≈{rss_gib():.2f} GiB")
        else:
            p.log(f"Reading non-FITS table via astropy: {ext}")
            fits_table = Table.read(filename, format=ext)
            names = [name for name in fits_table.colnames if len(fits_table[name].shape) <= 1]
            df = fits_table[names].to_pandas()
            p.log(f"DataFrame built: rows={len(df):,}, cols={df.shape[1]:,}")
            p.log(f"After table built: mem≈{df_mem_gib(df):.2f} GiB, RSS≈{rss_gib():.2f} GiB")

        if optimise_data:
            df = optimise_streaming(df, p=p, log_every=10)
            p.log(f"After optimise: mem≈{df_mem_gib(df):.2f} GiB, RSS≈{rss_gib():.2f} GiB")

        p.log(f"After add_ra_dec: mem≈{df_mem_gib(df):.2f} GiB, RSS≈{rss_gib():.2f} GiB")
        p.log("All done")
        return df

    def _load_data_cb(self, event):
        self.load_data_button.disabled = True
        self.load_data_button.name = "Loading File..."

        self.config.settings["dataset_filepath"] = self.dataset
        self.config.main_df = self.get_dataframe_from_fits_file(self.dataset)
        self.df = self.config.main_df
        self.src.data = dict(pd.DataFrame())

        self._initialise_src()
        self.ready = True
        self.load_data_button.name = "File Loaded."
        self.close_settings_button.disabled = False
        self.close_settings_button.button_type = "success"

    def get_df(self):
        return self.df

    def _initialise_src(self):
        new_df = pd.DataFrame([[0, 0], [0, 0]], columns=["test", "test"])
        self.src.data = dict(new_df)

    def _welcome_message(self):
        if self.error_message == "":
            return pn.pane.Markdown(
                """
    Welcome to AstronomicAL, an interactive dashboard for visualisation,
    integration and classification of data using active learning methods.

    For tutorials and API reference documents, please visit our
    documentation [here](https://astronomical.readthedocs.io).

    AstronomicAL provides both an example dataset and an example
    configuration file to allow you to jump right into the software and
    give it a test run.

    To begin training you simply have to select **Load custom configuration**
    and choose your config file.

    The **Load config select** option allows you to choose the extent to
    which to reload the configuration.
                """,
                sizing_mode="stretch_width",
                styles={
                    "font-size": "14px",
                    "line-height": "1.6",
                    "color": "#253858",
                },
                margin=0,
            )

        return pn.pane.Alert(
            self.error_message,
            alert_type="danger",
            sizing_mode="stretch_width",
            margin=0,
        )

    def _build_controls(self):
        memory_row = pn.Row(
            self.memory_optimisation_check,
            self._memory_opt_tooltip,
            sizing_mode="stretch_width",
            margin=0,
        )

        label_styles = {
            "font-size": "13px",
            "color": "#44546A",
        }

        dataset_block = pn.Column(
            pn.pane.Markdown(
                "**Data file**",
                margin=(0, 0, 2, 0),
                styles=label_styles,
            ),
            self.dataset_widget,
            sizing_mode="stretch_width",
            margin=0,
            min_height=42,
        )

        config_file_block = pn.Column(
            pn.pane.HTML(
                """
                <div style="
                    font-size:13px;
                    color:#44546A;
                    font-weight:600;
                    line-height:13px;
                    margin:0;
                    padding:0;
                    display:inline-block;
                ">Configuration file</div>
                """,
                margin=0,
                height=16,
            ),
            self.config_file_widget,
            sizing_mode="stretch_width",
            margin=0,
        )

        load_option_block = pn.Column(
            pn.pane.HTML(
                """
                <div style="
                    font-size:13px;
                    color:#44546A;
                    font-weight:600;
                    line-height:13px;
                    margin:0;
                    padding:0;
                    display:inline-block;
                ">Load config options</div>
                """,
                margin=0,
                height=16,
            ),
            self.load_config_select_widget,
            sizing_mode="stretch_width",
            margin=0,
        )

        button_block = pn.Row(
            self.load_data_button if not self.load_layout_check else self.load_data_button_js,
            sizing_mode="stretch_width",
            margin=0,
            min_height=38,
        )

        if self.load_layout_check:
            return pn.Column(
                self.load_layout_widget,
                config_file_block,
                pn.Spacer(height=1),
                load_option_block,
                pn.Spacer(height=6),
                button_block,
                sizing_mode="stretch_width",
                max_width=380,
                margin=(0, 20, 10, 20),
            )

        return pn.Column(
            self.load_layout_widget,
            memory_row,
            pn.Spacer(height=2),
            dataset_block,
            pn.Spacer(height=8),
            button_block,
            sizing_mode="stretch_width",
            max_width=380,
            margin=(0, 20, 10, 20),
        )

    def _build_info(self):
        return pn.Column(
            pn.pane.Markdown(
                "### Information",
                styles={
                    "line-height": "1.2",
                    "color": "#172B4D",
                },
                margin=(0, 0, 8, 0),
            ),
            self._welcome_message(),
            sizing_mode="stretch_width",
            margin=(0, 20, 16, 20),
        )

    def _refresh_layout(self):
        self.view.objects = [
            self._build_controls(),
            pn.layout.Divider(margin=(8, 20, 12, 20)),
            self._build_info(),
        ]

    def panel(self):
        return self.view