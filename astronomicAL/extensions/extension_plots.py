from datetime import datetime
import uuid
from holoviews.operation.datashader import (
    datashade,
    dynspread,
)

import datashader as ds
import holoviews as hv
from holoviews import opts

import numpy as np
import pandas as pd
import panel as pn
import glob
import json
import os


import uuid
from dataclasses import dataclass
from typing import Any, Callable, List

try:
    from astronomicAL.platform.events import Subscription
except Exception:  # pragma: no cover
    Subscription = Any  # type: ignore


@dataclass
class _ManagedJob:
    key: str
    handle: Any  # JobHandle from JobManager



def get_plot_dict():


    plot_dict = {
        #"Debug publish" : CustomPlot(debug_plot_publisher, []),

        #"Debug subscribe" : CustomPlot(debug_plot_subscriber, []),

        #"Euclid Cutout" : CustomPlot(euclid_cutout_plot, []),
       
        #"DESI Spectra from Coords" : CustomPlot(spectrum_plot, [], dataset="DESI", from_sourceId=False),

        #"DESI Spectrum from ID" : CustomPlot(spectrum_plot, ["DESI_TargetID"], dataset="DESI", from_sourceId=True),

        #"Euclid Spectra from Coords" : CustomPlot(spectrum_plot, [], dataset="EuclidSpec", from_sourceId=False),

        #"Euclid Spectrum from ID" : CustomPlot(spectrum_plot, ["Euclid_TargetID"], dataset="EuclidSpec", from_sourceId=True),

        #"SDSS Spectra from Coords" : CustomPlot(spectrum_plot, [], dataset="SDSS", from_sourceId=False),

        #"SDSS Spectrum from ID" : CustomPlot(spectrum_plot, ["SDSS_TargetID"], dataset="SDSS", from_sourceId=True),

        #"Mateos 2012 Wedge": CustomPlot(
        #    mateos_2012_wedge, ["Log10(W3_Flux/W2_Flux)", "Log10(W2_Flux/W1_Flux)"]
        #),
        "BPT Plots" : lambda context : CustomPlot(
            bpt_plot,
            [
                "Log10(NII_6584_FLUX/H_ALPHA_FLUX)",
                "Log10(SII_6717_FLUX/H_ALPHA_FLUX)",
                "Log10(OI_6300_FLUX/H_ALPHA_FLUX)",
                "Log10(OIII_5007_FLUX/H_BETA_FLUX)",
            ], context = context,
        ),
        #"SED Plot": SEDPlot(sed_plot, []),

        #"VLA-VLASS Cutout" : CustomPlot(vlass_cutout_plot, []),

        #"LOFAR-LoTSS Cutout" : CustomPlot(lotss_cutout_plot, []),



        "Stored Image"  : lambda context : CustomPlot(
            local_stored_plot, 
            ["Local_image_path"], 
            context=context)
    }

    return plot_dict


class CustomPlot:
    def __init__(self, plot_fn, extra_features, context, **plot_fn_kwargs):

        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        # Lifecycle tracking
        self._subscriptions: List[Subscription] = []
        self._jobs: List[_ManagedJob] = []
        self._bokeh_on_change: list[tuple[Any, str, Callable]] = []

        self._disposed = False
        self._event_subs = []
        self._periodic_cbs = []

        self.plot_fn = plot_fn
        self.extra_features = extra_features
        self.row = pn.Row("Loading...")
        self.plot_fn_kwargs = plot_fn_kwargs
        self.panel_id = str(uuid.uuid4()) 

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
        print(f"[dispose] CustomPlot panel_id={self.panel_id} plot_fn={getattr(self.plot_fn, '__name__', str(self.plot_fn))}")
        # 1) subclass cleanup first
        try:
            self._dispose_impl()
        except Exception:
            pass

        # remove any temp config keys set during unknown column selection
        try:
            self.remove_column_selection()
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
    
    def create_settings(self, unknown_cols):
        self.waiting = True
        main_columns = list(self.config.main_df.columns)
        settings_column = pn.Column()
        for i, col in enumerate(unknown_cols):

            if i % 3 == 0:
                settings_row = pn.Row()
            
            options = main_columns
            select_widget = pn.widgets.Select(name=col, options=options, max_height=120)
            settings_row.append(select_widget)
           

            if (i % 3 == 2) or (i == len(unknown_cols) - 1):
                settings_column.append(settings_row)

            if i == len(unknown_cols) - 1:
                settings_column.append(self.submit_button)

        return settings_column

    def render(self, data, selected=None):
        self.data = data
        self.selected = selected
        self.row[0] = self.col_selection
        return self.row

    def plot(self, submit_button):
        self.submit_button = submit_button

        current_cols = self.config.main_df.columns
        self.unknown_cols = []

        for col in self.extra_features:
            if col not in list(self.config.settings.keys()):
                if col not in current_cols:
                    self.unknown_cols.append(col)
                else:
                    self.config.settings[col] = col

        if len(self.unknown_cols) > 0:
            self.col_selection = self.create_settings(self.unknown_cols)
            return self.render
        else:
            def plot_with_instance(*args, **kwargs):
                view = self.plot_fn(context=self.context, *args, plot_instance=self, **kwargs, **self.plot_fn_kwargs)
                # put the view inside our controller-owned container
                self.row[0] = view
                return self.row
            return plot_with_instance

    def remove_column_selection(self):
        if hasattr(self, "unknown_cols"):
            for col in self.unknown_cols:
                if col in self.config.settings:
                    del self.config.settings[col]
            print(f"[{self.panel_id}] unknown columns selected removed from config")

    def panel(self):
        return self.row
    

def create_plot(
    data,
    x,
    y,
    plot_type="scatter",
    selected=None,
    show_selected=True,
    slow_render=False,
    legend=True,
    colours=True,
    smaller_axes_limits=False,
    bounds=None,
    legend_position=None,
    context = None
):
    assert x in list(data.columns), f"Column {x} is not a column in your dataframe."
    assert y in list(data.columns), f"Column {y} is not a column in your dataframe."

    if (context is not None and getattr(context, "config", None) is not None):
        config = context.config

    if bounds is not None:
        data = data[data[x] >= bounds[0]]
        data = data[data[y] <= bounds[1]]
        data = data[data[x] <= bounds[2]]
        data = data[data[y] >= bounds[3]]

    if plot_type == "scatter":
        p = hv.Points(
            data,
            [x, y],
        ).opts(active_tools=["pan", "wheel_zoom"])
    elif plot_type == "line":
        p = hv.Path(
            data,
            [x, y],
        ).opts(active_tools=["pan", "wheel_zoom"])
    if show_selected:

        if selected is not None:
            cols = list(data.columns)

            if len(selected.data[cols[0]]) == 1:
                selected = pd.DataFrame(selected.data, columns=cols, index=[0])
                if bounds is not None:
                    if (
                        (selected[x][0] < bounds[0])
                        or (selected[y][0] > bounds[1])
                        or (selected[x][0] > bounds[2])
                        or (selected[y][0] < bounds[3])
                    ):
                        selected = pd.DataFrame(columns=cols)
            else:
                selected = pd.DataFrame(columns=cols)

            selected_plot = hv.Scatter(selected, x, y,).opts(
                fill_color="black",
                marker="circle",
                size=10,
                active_tools=["pan", "wheel_zoom"],
            )

    if colours:
        color_key = config.settings["label_colours"]

        # color_points = hv.NdOverlay(
        #     {
        #         config.settings["labels_to_strings"][f"{n}"]: hv.Points(
        #             [0, 0], label=config.settings["labels_to_strings"][f"{n}"]
        #         ).opts(style=dict(color=color_key[n], size=0))
        #         for n in color_key
        #     }
        # )

    if smaller_axes_limits:

        max_x = np.max(data[x])
        min_x = np.min(data[x])

        max_y = np.max(data[y])
        min_y = np.min(data[y])

        x_sd = np.std(data[x])
        x_mu = np.mean(data[x])
        y_sd = np.std(data[y])
        y_mu = np.mean(data[y])

        max_x = np.min([x_mu + 4 * x_sd, max_x])
        min_x = np.max([x_mu - 4 * x_sd, min_x])

        max_y = np.min([y_mu + 4 * y_sd, max_y])
        min_y = np.max([y_mu - 4 * y_sd, min_y])

        if show_selected:
            if selected is not None:
                if selected.shape[0] > 0:

                    max_x = np.max([max_x, np.max(selected[x])])
                    min_x = np.min([min_x, np.min(selected[x])])

                    max_y = np.max([max_y, np.max(selected[y])])
                    min_y = np.min([min_y, np.min(selected[y])])

    if colours:

        label_col = (context.config.settings.get("label_col") if hasattr(context, "config") else config.settings.get("label_col"))
        has_label = bool(label_col) and (label_col in data.columns)

        agg = ds.by(label_col, ds.count()) if has_label else ds.count()


        if smaller_axes_limits:
            plot = dynspread(
                datashade(
                    p,
                    color_key=color_key,
                    aggregator=agg,
                ).opts(xlim=(min_x, max_x), ylim=(min_y, max_y), responsive=True),
                threshold=0.75,
                how="saturate",
            )
        else:
            plot = dynspread(
                datashade(
                    p,
                    color_key=color_key,
                    aggregator=agg,
                ).opts(responsive=True),
                threshold=0.75,
                how="saturate",
            )

    else:
        if smaller_axes_limits:
            plot = dynspread(
                datashade(
                    p,
                ).opts(xlim=(min_x, max_x), ylim=(min_y, max_y), responsive=True),
                threshold=0.75,
                how="saturate",
            ).redim.range(xdim=(min_x, max_x), ydim=(min_y, max_y))
        else:
            plot = dynspread(
                datashade(
                    p,
                ).opts(responsive=True),
                threshold=0.75,
                how="saturate",
            )

    if slow_render:
        plot = p

    if show_selected and (selected is not None):
        plot = plot * selected_plot

    if legend and colours:
        plot = plot #* color_points

    if legend_position is not None:
        plot = plot.opts(legend_position=legend_position)

    return plot


def bpt_plot(data, selected=None, plot_instance=None, context = None):

    config = context.config

    plot_NII = create_plot(
        data,
        config.settings["Log10(NII_6584_FLUX/H_ALPHA_FLUX)"],
        config.settings["Log10(OIII_5007_FLUX/H_BETA_FLUX)"],
        plot_type="scatter",
        legend=True,
        selected=selected,
        bounds=[-1.8, 1.25, 1, -2.2],
        legend_position="bottom_right",
        context=context
    )

    x1 = np.linspace(-1.6, -0.2, 60)
    x2 = np.linspace(-1.6, 0.2, 60)
    y1 = (0.61 / (x1 - 0.05)) + 1.3
    y2 = (0.61 / (x2 - 0.47)) + 1.19

    l1 = pd.DataFrame(np.array([x1, y1]).T, columns=["x", "y"])
    l2 = pd.DataFrame(np.array([x2, y2]).T, columns=["x", "y"])

    NII_line1 = create_plot(l1, "x", "y", plot_type="line", legend=False, colours=False, context=context)

    NII_line2 = create_plot(l2, "x", "y", plot_type="line", legend=False, colours=False, context=context)

    plot_NII = plot_NII * NII_line1 * NII_line2

    plot_SII = create_plot(
        data,
        config.settings["Log10(SII_6717_FLUX/H_ALPHA_FLUX)"],
        config.settings["Log10(OIII_5007_FLUX/H_BETA_FLUX)"],
        plot_type="scatter",
        legend=True,
        selected=selected,
        bounds=[-2.1, 1.2, 0.9, -2.1],
        legend_position="bottom_right",
        context=context
    )

    x1 = np.linspace(-2, 0.1, 60)
    y1 = (0.72 / (x1 - 0.32)) + 1.30

    l1 = pd.DataFrame(np.array([x1, y1]).T, columns=["x", "y"])

    SII_line1 = create_plot(l1, "x", "y", plot_type="line", legend=False, colours=False, context=context)

    plot_SII = plot_SII * SII_line1

    plot_OI = create_plot(
        data,
        config.settings["Log10(OI_6300_FLUX/H_ALPHA_FLUX)"],
        config.settings["Log10(OIII_5007_FLUX/H_BETA_FLUX)"],
        plot_type="scatter",
        legend=True,
        selected=selected,
        bounds=[-3.3, 1.25, 1.65, -2.3],
        legend_position="bottom_right",
        context=context
    )

    x1 = np.linspace(-3, -0.8, 60)
    y1 = (0.73 / (x1 + 0.59)) + 1.33

    l1 = pd.DataFrame(np.array([x1, y1]).T, columns=["x", "y"])

    OI_line1 = create_plot(l1, "x", "y", plot_type="line", legend=False, colours=False, context=context)

    plot_OI = plot_OI * OI_line1

    tabs = pn.Tabs(
        ("NII", plot_NII.opts(legend_position="bottom_right", shared_axes=False)),
        ("SII", plot_SII.opts(legend_position="bottom_right", shared_axes=False)),
        ("OI", plot_OI.opts(legend_position="bottom_right", shared_axes=False)),
    )
    return tabs


def mateos_2012_wedge(data, selected=None, plot_instance=None, context=None):

    if (context is not None and getattr(context, "config", None) is not None):
        config = context.config

    plot = create_plot(
        data,
        config.settings["Log10(W3_Flux/W2_Flux)"],
        config.settings["Log10(W2_Flux/W1_Flux)"],
        plot_type="scatter",
        legend=True,
        selected=selected,
        legend_position="bottom_right",
        context = context
    )

    x = data[config.settings["Log10(W3_Flux/W2_Flux)"]]

    top_y_orig = (0.315 * x) + 0.297
    bottom_y_orig = (0.315 * x) - 0.110

    threshold_w = (-3.172 * x) + 0.436

    top_x = x[top_y_orig > threshold_w]
    top_y = top_y_orig[top_y_orig > threshold_w]

    bottom_x = x[bottom_y_orig > threshold_w]
    bottom_y = bottom_y_orig[bottom_y_orig > threshold_w]

    threshold_y = threshold_w[
        (top_y_orig > threshold_w) & (bottom_y_orig < threshold_w)
    ]
    threshold_x = x[(top_y_orig > threshold_w) & (bottom_y_orig < threshold_w)]

    top_x = np.array([np.min(top_x), np.max(top_x)])
    top_y = (0.315 * top_x) + 0.297
    top = pd.DataFrame(np.array([top_x, top_y]).transpose(), columns=["x", "y"])

    bottom_x = np.array([np.min(bottom_x), np.max(bottom_x)])
    bottom_y = (0.315 * bottom_x) - 0.110
    bottom = pd.DataFrame(
        np.array([bottom_x, bottom_y]).transpose(), columns=["x", "y"]
    )

    threshold_x = np.array([np.min(threshold_x), np.max(threshold_x)])
    threshold_y = (-3.172 * threshold_x) + 0.436
    threshold = pd.DataFrame(
        np.array([threshold_x, threshold_y]).transpose(), columns=["x", "y"]
    )

    p1 = create_plot(top, "x", "y", plot_type="line", legend=False, colours=False, context = context)
    p2 = create_plot(bottom, "x", "y", plot_type="line", legend=False, colours=False, context = context)
    p3 = create_plot(threshold, "x", "y", plot_type="line", legend=False, colours=False, context = context)

    plot = plot * p1 * p2 * p3

    plot.opts(legend_position="bottom_left")

    return plot


class SEDPlot(CustomPlot):
    def __init__(self, plot_fn, extra_features, context = None):

        self.context = context
        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        self.plot_fn = plot_fn
        self.extra_features = extra_features
        self.row = pn.Row("Loading...")

    def create_settings(self, unknown_cols):

        self.waiting = True
        settings_column = pn.Column()
        for i, col in enumerate(unknown_cols):

            if i % 3 == 0:
                settings_row = pn.Row()

            settings_row.append(
                pn.widgets.Select(
                    name=col, options=list(self.config.main_df.columns), max_height=120
                )
            )

            if (i % 3 == 2) or (i == len(unknown_cols) - 1):
                settings_column.append(settings_row)

            if i == len(unknown_cols) - 1:
                settings_column.append(self.submit_button)

        return settings_column

    def create_photometry_band_file(self, event):

        bands_dict = {}

        for col in list(self.config.main_df.columns):
            bands_dict[col] = {"wavelength": -99, "FWHM": 0, "error": 0}

        if not os.path.isdir("data/sed_data"):
            os.mkdir("data/sed_data")

        if not os.path.isfile("data/sed_data/photometry_bands.json"):
            with open("data/sed_data/photometry_bands.json", "w") as fp:
                json.dump(bands_dict, fp, indent=2)

        else:
            now = datetime.now()
            dt_string = now.strftime("%Y%m%d_%H:%M:%S")
            with open(f"data/sed_data/photometry_bands_{dt_string}.json", "w") as fp:
                json.dump(bands_dict, fp, indent=2)

        files = glob.glob("data/sed_data/*.json")
        self.files_selection.options = [""] + files

        self.files_selection.value = f"data/sed_data/photometry_bands_{dt_string}.json"

        self.plot(self.submit_button)

    def _get_unknown_features(self):

        unknown_cols = []
        df_columns = list(self.config.main_df.columns)

        with open(self.config.settings["sed_file"], "r") as fp:
            bands = json.load(fp)

        for i in bands:
            if bands[i]["wavelength"] != -99:
                if i not in df_columns:
                    if i not in list(self.config.settings.keys()):
                        unknown_cols.append(i)
                    elif self.config.settings[i] not in df_columns:
                        unknown_cols.append(i)
            if type(bands[i]["wavelength"]) == str:
                if bands[i]["wavelength"] not in self.config.main_df.columns:
                    if bands[i]["wavelength"] not in self.config.settings.keys():
                        unknown_cols.append(bands[i]["wavelength"])
            if type(bands[i]["FWHM"]) == str:
                if bands[i]["FWHM"] not in self.config.main_df.columns:
                    if bands[i]["FWHM"] not in self.config.settings.keys():
                        unknown_cols.append(bands[i]["FWHM"])
            if type(bands[i]["error"]) == str:
                if bands[i]["error"] not in self.config.main_df.columns:
                    if bands[i]["error"] not in self.config.settings.keys():
                        unknown_cols.append(bands[i]["error"])

                else:
                    continue

        return unknown_cols

    def render(self, data, selected=None):
        self.data = data
        self.selected = selected
        self.row[0] = self.col_selection
        return self.row

    def _load_file(self):
        selected = self.files_selection.value

        if selected != "":
            self.config.settings["sed_file"] = selected
        else:
            self.config.settings["sed_file"] = None

    def _load_file_menu(self, data, selected=None):

        files = glob.glob("data/sed_data/*.json")
        self.files_selection = pn.widgets.Select(name="Select", options=[""] + files)

        self.create_new_file_button = pn.widgets.Button(name="Create new SED data file")
        self.create_new_file_button.on_click(self.create_photometry_band_file)

        load_column = pn.Column(
            self.files_selection, self.submit_button, self.create_new_file_button
        )

        self.row[0] = load_column

        return self.row

    def plot(self, submit_button):
        self.submit_button = submit_button

        if self.submit_button.disabled:
            pass

        elif "sed_file" not in self.config.settings.keys():
            return self._load_file_menu

        elif self.config.settings["sed_file"] is None:
            return self._load_file_menu

        elif not os.path.isfile(self.config.settings["sed_file"]):
            print("Wrong file")
            self.config.settings["sed_file"] = None
            return self._load_file_menu

        with open(self.config.settings["sed_file"], "r") as fp:
            self.extra_columns = json.load(fp)

        unknown_cols = self._get_unknown_features()

        if len(unknown_cols) > 0:
            self.col_selection = self.create_settings(unknown_cols)
            return self.render
        else:
            return self.plot_fn


def sed_plot(data, selected=None, context=None):

    if (context is not None and getattr(context, "config", None) is not None):
        config = context.config
    
    df_columns = list(config.main_df.columns)

    with open(config.settings["sed_file"], "r") as fp:
        bands = json.load(fp)
    new_data = []
    for i in bands:
        mag = -99
        if i in df_columns:
            if len(selected.data[i]) > 0:
                if bands[i]["wavelength"] == -99:
                    continue
                else:
                    mag = selected.data[i][0]

        elif i in list(config.settings.keys()):
            if config.settings[i] in df_columns:
                if len(selected.data[config.settings[i]]) > 0:
                    if bands[i]["wavelength"] == -99:
                        continue
                    else:
                        mag = selected.data[config.settings[i]][0]

        wavelength = bands[i]["wavelength"]
        if len(selected.data[f"{config.settings['id_col']}"]) > 0:
            if type(bands[i]["wavelength"]) == str:
                if wavelength in config.main_df.columns:
                    wavelength = selected.data[wavelength][0]
                elif config.settings[wavelength] in config.main_df.columns:
                    wavelength = selected.data[config.settings[wavelength]][0]
                else:
                    continue

        fwhm = bands[i]["FWHM"]
        if len(selected.data[f"{config.settings['id_col']}"]) > 0:
            if type(bands[i]["FWHM"]) == str:
                if fwhm in config.main_df.columns:
                    fwhm = selected.data[fwhm][0]
                elif config.settings[fwhm] in config.main_df.columns:
                    fwhm = selected.data[config.settings[fwhm]][0]
                else:
                    continue
        mag_err = bands[i]["error"]
        if len(selected.data[f"{config.settings['id_col']}"]) > 0:
            if type(bands[i]["error"]) == str:
                if mag_err in config.main_df.columns:
                    mag_err = selected.data[mag_err][0]
                elif config.settings[mag_err] in config.main_df.columns:
                    mag_err = selected.data[config.settings[mag_err]][0]
                else:
                    continue

        if mag == -99:
            continue

        new_data.append(
            [
                wavelength,
                mag,
                fwhm,
                mag_err,
            ]
        )

    new_data = pd.DataFrame(
        new_data, columns=["wavelength (µm)", "magnitude", "FWHM", "error"]
    )

    new_data = new_data[new_data["magnitude"] != -99]

    if len(new_data) > 0:
        plot = create_plot(
            new_data,
            "wavelength (µm)",
            "magnitude",
            plot_type="line",
            colours=False,
            legend=False,
            show_selected=False,
            slow_render=True,
            context = context
        )
        points = hv.Scatter(new_data, kdims=["wavelength (µm)"],).opts(
            fill_color="black",
            marker="circle",
            alpha=0.5,
            size=4,
            active_tools=["pan", "wheel_zoom"],
        )
        error_data_x = [
            (
                new_data["wavelength (µm)"].values[i],
                new_data["magnitude"].values[i],
                new_data["error"].values[i],
            )
            for i in range(len(new_data))
        ]

        error_data_y = [
            (
                new_data["wavelength (µm)"].values[i],
                new_data["magnitude"].values[i],
                new_data["FWHM"].values[i] * 0.5,
            )
            for i in range(len(new_data))
        ]

        errors_x = hv.Spread(error_data_x, horizontal=True)
        errors_y = hv.ErrorBars(error_data_y, horizontal=True)
        plot = plot * points * errors_x * errors_y
        plot.opts(invert_yaxis=True, logx=True)

    else:
        plot = hv.Scatter(
            pd.DataFrame({"wavelength (µm)": [], "magnitude": []}),
            vdims=["wavelength (µm)", "magnitude"],
            kdims=["magnitude"],
        )

    return plot


################## Ivano

def get_selected_source(data, selected):
        if selected is None:
            return None
        cols = list(data.columns)
        if not len(selected.data[cols[0]]) == 1:
            return None 
        return pd.DataFrame(selected.data, columns=cols, index=[0])

def check_required_column(df, column):
    return column in list(df.columns)


def empty_panel(message = "Loading error"):
    return pn.pane.Markdown(message)


def local_stored_plot(data, selected, plot_instance=None, context = None):
    selected_source = get_selected_source(data=data, selected = selected)
    path = int(selected_source[context.config.settings["Local_image_path"]].iloc[0])
    try:
        return pn.pane.Image(path, width = 500)
    except Exception as e:
        print(e)
        return empty_panel()