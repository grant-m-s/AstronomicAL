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
from pathlib import Path
import re
import pandas as pd
import panel as pn
import param

class DataSelection(param.Parameterized):
    """The Data Selection Stage used in the settings pipeline."""

    dataset = param.ObjectSelector(default="", objects=[""])
    workspace_file = param.ObjectSelector(default="", objects=[""])
    load_layout_check = param.Boolean(False, label="Load Custom Workspace?")
    ready = param.Boolean(default=False)

    def __init__(self, src, mode, context, close_settings_button):
        super(DataSelection, self).__init__()

        self.context = context
        self.mode = mode
        self.src = src
        self.error_message = ""
        self.close_settings_button = close_settings_button
        self.df = pd.DataFrame()

        if context is None:
            raise ValueError("DataSelection requires context.")
        if getattr(context, "datasets", None) is None:
            raise ValueError("DataSelection requires context.datasets.")

        self._initialise_widgets()

        self.view = pn.Column(sizing_mode="stretch_width")
        self._refresh_layout()

    def _initialise_widgets(self):
        data_files = self._get_data_files()
        if not data_files:
            data_files = [""]

        workspace_files = [""] + self._get_workspace_files()

        self.param.dataset.objects = data_files
        self.param.workspace_file.objects = workspace_files

        self.dataset = data_files[0]
        self.workspace_file = workspace_files[0]

        self.dataset_widget = pn.widgets.Select(
            options=data_files,
            value=self.dataset,
            width=320,
            margin=0,
            height=34,
        )

        self.workspace_file_widget = pn.widgets.Select(
            options=workspace_files,
            value=self.workspace_file,
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
            name="Load custom workspace?",
            value=self.load_layout_check,
            margin=0,
            height=20,
        )

        self.load_workspace_button = pn.widgets.Button(
            name="Select a workspace file to continue",
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
        self.workspace_file_widget.param.watch(
            self._sync_workspace_file,
            "value",
        )
        self.load_layout_widget.param.watch(self._sync_load_layout_check, "value")

        self.load_data_button.on_click(self._load_data_cb)
        self.load_workspace_button.on_click(self._load_workspace_cb)

    def _sync_load_layout_check(self, event):
        self.load_layout_check = event.new
        self._update_layout_file_cb()

    def _sync_dataset(self, event):
        self.dataset = event.new

    def _sync_workspace_file(self, event):
        self.workspace_file = event.new
        self._update_layout_file_cb()

    def _get_data_files(self):
        return glob.glob("data/*.*")

    def _get_workspace_files(self):
        root = Path(
            getattr(self.context, "layout_directory", "layouts")
        ).expanduser()
        files = sorted(str(path) for path in root.glob("*.json") if path.is_file())

        current = Path(getattr(self.context, "layout_file", "")).expanduser()
        if current.is_file():
            current_text = str(current)
            files = [current_text, *[path for path in files if path != current_text]]
        return files

    def _update_layout_file_cb(self, *events):
        del events
        if not self.load_layout_check:
            self.error_message = ""
            self.load_workspace_button.name = "Select a workspace file to continue"
            self.load_workspace_button.disabled = True
            self._refresh_layout()
            return

        if not self.workspace_file:
            self.error_message = ""
            self.load_workspace_button.name = "Select a workspace file to continue"
            self.load_workspace_button.disabled = True
            self._refresh_layout()
            return

        path = Path(self.workspace_file).expanduser()
        self.load_workspace_button.name = "Verifying workspace..."
        self.load_workspace_button.disabled = True

        try:
            persistence = getattr(self.context, "persistence", None)
            if persistence is None:
                raise RuntimeError("context.persistence is not configured.")
            persistence.load(path)
        except Exception as exc:
            self.error_message = f"Unable to load workspace: `{exc}`"
            self.load_workspace_button.name = "Unable to load workspace"
            self.load_workspace_button.disabled = True
        else:
            self.error_message = ""
            self.load_workspace_button.name = "Load workspace"
            self.load_workspace_button.disabled = False

        self._refresh_layout()

    def _load_workspace_cb(self, event):
        del event
        path = Path(self.workspace_file).expanduser()
        self.load_workspace_button.disabled = True
        self.load_workspace_button.name = "Loading workspace..."

        try:
            persistence = getattr(self.context, "persistence", None)
            if persistence is None:
                raise RuntimeError("context.persistence is not configured.")
            snapshot = persistence.load(path)
            if hasattr(persistence, "reconcile"):
                issues = persistence.reconcile(snapshot, strict=False)
            else:
                issues = persistence.restore(snapshot, strict=False)

            self.context.layout_file = path
            self.ready = True
            self.error_message = ""
            self.load_workspace_button.name = (
                "Workspace loaded"
                if not issues
                else f"Loaded with {len(issues)} issue(s)"
            )
            self._publish(
                "workspace.loaded",
                {
                    "path": str(path),
                    "issues": issues,
                    "origin": "settings.data_selection",
                },
            )
            self._enable_close_buttons()
        except Exception as exc:
            self.error_message = f"Unable to load workspace: `{exc}`"
            self.load_workspace_button.name = "Unable to load workspace"
            self.load_workspace_button.disabled = False
            self._refresh_layout()

    def get_dataframe_from_fits_file(self, filename, optimise_data=None):
        p = PrintProgress(every=2.0)
        ext = filename[filename.rindex(".") + 1 :].lower()

        if optimise_data is None:
            optimise_data = bool(self.memory_optimisation_check.value)
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
        del event
        self.load_data_button.disabled = True
        self.load_data_button.name = "Loading File..."

        try:
            filename = str(self.dataset)
            optimise_data = bool(self.memory_optimisation_check.value)
            self.df = self.get_dataframe_from_fits_file(
                filename,
                optimise_data=optimise_data,
            )
            dataset_id = self._unique_dataset_id(Path(filename).stem)
            dataset_name = (
                Path(filename).stem.replace("_", " ").replace("-", " ").title()
            )
            self.context.datasets.register(
                dataset_id,
                self.df,
                name=dataset_name,
                source_path=filename,
                loader_id="settings.data_selection",
                optimise_data=optimise_data,
                rows=len(self.df),
                columns=list(self.df.columns),
            )
            self.context.datasets.set_active(
                dataset_id,
                origin="settings.data_selection",
            )
            self.src.data = {}
            self._initialise_src()
            self.ready = True
            self.load_data_button.name = "File Loaded."
            self._publish(
                "dataset.loaded",
                {
                    "dataset_id": dataset_id,
                    "name": dataset_name,
                    "rows": len(self.df),
                    "columns": list(self.df.columns),
                    "source_path": filename,
                    "loader_id": "settings.data_selection",
                    "optimise_data": optimise_data,
                    "backend": "pandas",
                },
            )
            self._enable_close_buttons()
        except Exception:
            self.load_data_button.name = "Unable to load file"
            raise
        finally:
            self.load_data_button.disabled = False

    def _unique_dataset_id(self, base: str) -> str:
        base = str(base or "dataset").strip().lower()
        base = re.sub(r"[^a-z0-9_]+", "_", base)
        base = re.sub(r"_+", "_", base).strip("_") or "dataset"
        existing = set(self.context.datasets.list_ids())
        if base not in existing:
            return base
        index = 2
        while f"{base}_{index}" in existing:
            index += 1
        return f"{base}_{index}"

    def _publish(self, topic: str, payload: dict) -> None:
        events = getattr(self.context, "events", None)
        if events is not None:
            events.publish(topic, payload)

    def _enable_close_buttons(self) -> None:
        self.close_settings_button.disabled = False
        self.close_settings_button.button_type = "success"
        try:
            modal_close_button = getattr(
                self.context,
                "_dataset_header_modal_close_button",
                None,
            )
            if modal_close_button is not None:
                modal_close_button.disabled = False
                modal_close_button.button_type = "success"
        except Exception:
            pass

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

    AstronomicAL can load datasets into the current session or restore a
    previously saved plugin workspace.

    Select **Load custom workspace** to restore panels, mappings, selection,
    and layout from a workspace JSON file.
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

        workspace_file_block = pn.Column(
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
                ">Workspace file</div>
                """,
                margin=0,
                height=16,
            ),
            self.workspace_file_widget,
            sizing_mode="stretch_width",
            margin=0,
        )

        button_block = pn.Row(
            (
                self.load_data_button
                if not self.load_layout_check
                else self.load_workspace_button
            ),
            sizing_mode="stretch_width",
            margin=0,
            min_height=38,
        )

        if self.load_layout_check:
            return pn.Column(
                self.load_layout_widget,
                workspace_file_block,
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