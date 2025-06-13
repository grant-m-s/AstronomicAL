from datetime import datetime
import uuid
from holoviews.operation.datashader import (
    datashade,
    dynspread,
)

import datashader as ds
import holoviews as hv
from holoviews import opts
from functools import partial

import astronomicAL.config as config
import numpy as np
import pandas as pd
import panel as pn
import glob
import json
import os
import param

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import concurrent.futures 
from astronomicAL.extensions.shared_data import shared_data
from astronomicAL.extensions.astro_data_utility import DESISpectraClass, EuclidCutoutsClass, EuclidSpectraClass
from astronomicAL.extensions.astro_data_utility import VLASS_cutout, LoTSS_cutout


def get_plot_dict():

    plot_dict = {
        #"Debug publish" : CustomPlot(debug_plot_publisher, []),

        #"Debug subscribe" : CustomPlot(debug_plot_subscriber, []),

        "Euclid Cutout" : CustomPlot(euclid_cutout_plot, []),

        "DESI Spectra from Coords" : CustomPlot(spectrum_plot, [], dataset="DESI", from_sourceId=False),

        "DESI Spectrum from ID" : CustomPlot(spectrum_plot, ["DESI_TargetID"], dataset="DESI", from_sourceId=True),

        "Euclid Spectra from Coords" : CustomPlot(spectrum_plot, [], dataset="EuclidSpec", from_sourceId=False),

        "Euclid Spectrum from ID" : CustomPlot(spectrum_plot, ["Euclid_TargetID"], dataset="EuclidSpec", from_sourceId=True),

        "SDSS Spectra from Coords" : CustomPlot(spectrum_plot, [], dataset="SDSS", from_sourceId=False),

        "SDSS Spectrum from ID" : CustomPlot(spectrum_plot, ["SDSS_TargetID"], dataset="SDSS", from_sourceId=True),

        "Mateos 2012 Wedge": CustomPlot(
            mateos_2012_wedge, ["Log10(W3_Flux/W2_Flux)", "Log10(W2_Flux/W1_Flux)"]
        ),
        "BPT Plots": CustomPlot(
            bpt_plot,
            [
                "Log10(NII_6584_FLUX/H_ALPHA_FLUX)",
                "Log10(SII_6717_FLUX/H_ALPHA_FLUX)",
                "Log10(OI_6300_FLUX/H_ALPHA_FLUX)",
                "Log10(OIII_5007_FLUX/H_BETA_FLUX)",
            ],
        ),
        "SED Plot": SEDPlot(sed_plot, []),

        "VLA-VLASS Cutout" : CustomPlot(vlass_cutout_plot, []),

        "LOFAR-LoTSS Cutout" : CustomPlot(lotss_cutout_plot, []),

        "Stored Image"  : CustomPlot(local_stored_plot, ["Local_image_path"])
    }

    return plot_dict


class CustomPlot:
    def __init__(self, plot_fn, extra_features, **plot_fn_kwargs):

        self.plot_fn = plot_fn
        self.extra_features = extra_features
        self.row = pn.Row("Loading...")
        self.plot_fn_kwargs = plot_fn_kwargs
        self.panel_id = str(uuid.uuid4()) 

    
    def create_settings(self, unknown_cols):
        self.waiting = True
        main_columns = list(config.main_df.columns)
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
        
        current_cols = config.main_df.columns
        
        unknown_cols = []
        
        for col in self.extra_features:
            if col not in list(config.settings.keys()):
                if col not in current_cols:
                    unknown_cols.append(col)
                else:
                    config.settings[col] = col
        if len(unknown_cols) > 0:
            self.col_selection = self.create_settings(unknown_cols)
            return self.render
        
        else:
            def plot_with_instance(*args, **kwargs):
                return self.plot_fn(*args, plot_instance = self, **kwargs, **self.plot_fn_kwargs)
            
            return plot_with_instance
    
    def cleanup_subscriptions(self):
        """Removes subscriptions to the shared data
           Used in a previous version"""
        shared_data.unsubscribe_panel(self.panel_id)
        print(f"CustomPlot {self.panel_id} removed from subscriptions")
    
    def cleanup_shared_data(self):
        """Removes subscriptions and published data from the shared data"""
        shared_data.cleanup_extension_panel(self.panel_id)
        print(f"CustomPlot {self.panel_id} removed from shared data")
    

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
):
    assert x in list(data.columns), f"Column {x} is not a column in your dataframe."
    assert y in list(data.columns), f"Column {y} is not a column in your dataframe."

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
        if smaller_axes_limits:
            plot = dynspread(
                datashade(
                    p,
                    color_key=color_key,
                    aggregator=ds.by(config.settings["label_col"], ds.count()),
                ).opts(xlim=(min_x, max_x), ylim=(min_y, max_y), responsive=True),
                threshold=0.75,
                how="saturate",
            )
        else:
            plot = dynspread(
                datashade(
                    p,
                    color_key=color_key,
                    aggregator=ds.by(config.settings["label_col"], ds.count()),
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


def bpt_plot(data, selected=None, plot_instance=None):

    plot_NII = create_plot(
        data,
        config.settings["Log10(NII_6584_FLUX/H_ALPHA_FLUX)"],
        config.settings["Log10(OIII_5007_FLUX/H_BETA_FLUX)"],
        plot_type="scatter",
        legend=True,
        selected=selected,
        bounds=[-1.8, 1.25, 1, -2.2],
        legend_position="bottom_right",
    )

    x1 = np.linspace(-1.6, -0.2, 60)
    x2 = np.linspace(-1.6, 0.2, 60)
    y1 = (0.61 / (x1 - 0.05)) + 1.3
    y2 = (0.61 / (x2 - 0.47)) + 1.19

    l1 = pd.DataFrame(np.array([x1, y1]).T, columns=["x", "y"])
    l2 = pd.DataFrame(np.array([x2, y2]).T, columns=["x", "y"])

    NII_line1 = create_plot(l1, "x", "y", plot_type="line", legend=False, colours=False)

    NII_line2 = create_plot(l2, "x", "y", plot_type="line", legend=False, colours=False)

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
    )

    x1 = np.linspace(-2, 0.1, 60)
    y1 = (0.72 / (x1 - 0.32)) + 1.30

    l1 = pd.DataFrame(np.array([x1, y1]).T, columns=["x", "y"])

    SII_line1 = create_plot(l1, "x", "y", plot_type="line", legend=False, colours=False)

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
    )

    x1 = np.linspace(-3, -0.8, 60)
    y1 = (0.73 / (x1 + 0.59)) + 1.33

    l1 = pd.DataFrame(np.array([x1, y1]).T, columns=["x", "y"])

    OI_line1 = create_plot(l1, "x", "y", plot_type="line", legend=False, colours=False)

    plot_OI = plot_OI * OI_line1

    tabs = pn.Tabs(
        ("NII", plot_NII.opts(legend_position="bottom_right", shared_axes=False)),
        ("SII", plot_SII.opts(legend_position="bottom_right", shared_axes=False)),
        ("OI", plot_OI.opts(legend_position="bottom_right", shared_axes=False)),
    )
    return tabs


def mateos_2012_wedge(data, selected=None, plot_instance=None):

    plot = create_plot(
        data,
        config.settings["Log10(W3_Flux/W2_Flux)"],
        config.settings["Log10(W2_Flux/W1_Flux)"],
        plot_type="scatter",
        legend=True,
        selected=selected,
        legend_position="bottom_right",
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

    p1 = create_plot(top, "x", "y", plot_type="line", legend=False, colours=False)
    p2 = create_plot(bottom, "x", "y", plot_type="line", legend=False, colours=False)
    p3 = create_plot(threshold, "x", "y", plot_type="line", legend=False, colours=False)

    plot = plot * p1 * p2 * p3

    plot.opts(legend_position="bottom_left")

    return plot


class SEDPlot(CustomPlot):
    def __init__(self, plot_fn, extra_features):

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
                    name=col, options=list(config.main_df.columns), max_height=120
                )
            )

            if (i % 3 == 2) or (i == len(unknown_cols) - 1):
                settings_column.append(settings_row)

            if i == len(unknown_cols) - 1:
                settings_column.append(self.submit_button)

        return settings_column

    def create_photometry_band_file(self, event):

        bands_dict = {}

        for col in list(config.main_df.columns):
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
        df_columns = list(config.main_df.columns)

        with open(config.settings["sed_file"], "r") as fp:
            bands = json.load(fp)

        for i in bands:
            if bands[i]["wavelength"] != -99:
                if i not in df_columns:
                    if i not in list(config.settings.keys()):
                        unknown_cols.append(i)
                    elif config.settings[i] not in df_columns:
                        unknown_cols.append(i)
            if type(bands[i]["wavelength"]) == str:
                if bands[i]["wavelength"] not in config.main_df.columns:
                    if bands[i]["wavelength"] not in config.settings.keys():
                        unknown_cols.append(bands[i]["wavelength"])
            if type(bands[i]["FWHM"]) == str:
                if bands[i]["FWHM"] not in config.main_df.columns:
                    if bands[i]["FWHM"] not in config.settings.keys():
                        unknown_cols.append(bands[i]["FWHM"])
            if type(bands[i]["error"]) == str:
                if bands[i]["error"] not in config.main_df.columns:
                    if bands[i]["error"] not in config.settings.keys():
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
            config.settings["sed_file"] = selected
        else:
            config.settings["sed_file"] = None

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

        elif "sed_file" not in config.settings.keys():
            return self._load_file_menu

        elif config.settings["sed_file"] is None:
            return self._load_file_menu

        elif not os.path.isfile(config.settings["sed_file"]):
            print("Wrong file")
            config.settings["sed_file"] = None
            return self._load_file_menu

        with open(config.settings["sed_file"], "r") as fp:
            self.extra_columns = json.load(fp)

        unknown_cols = self._get_unknown_features()

        if len(unknown_cols) > 0:
            self.col_selection = self.create_settings(unknown_cols)
            return self.render
        else:
            return self.plot_fn


def sed_plot(data, selected=None):

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
  

def get_ra_dec(selected_source):
    if check_required_column(selected_source, "ra_dec"):
        ra_dec = selected_source["ra_dec"][0]
        ra = float(ra_dec[: ra_dec.index(",")])
        dec = float(ra_dec[ra_dec.index(",") + 1 :])
    else:
        print("No ra and dec available for this source")
        ra, dec = None, None
    return ra, dec



def spectrum_plot(data, selected = None, plot_instance = None, dataset = "DESI", from_sourceId = False):

    if not hasattr(plot_instance, "container"):
        plot_instance.container = pn.Column(scroll = True)
    
    def update_plot(new_radius, panel_id = None):

        selected_source = get_selected_source(data=data, selected = selected)
    
        if from_sourceId:
            if config.settings[f"{dataset}_TargetID"] in selected_source.columns:
                sourceId = int(selected_source[config.settings[f"{dataset}_TargetID"]].iloc[0])
                ra, dec = None, None
            else:
                print("Missing column with target ID")
                plot_instance.container.objects = [pn.pane.Markdown("Missing Target ID")]
                return
        elif not from_sourceId:
            ra, dec = get_ra_dec(selected_source)
            if (ra is None) or (dec is None):
                plot_instance.container.objects = [pn.pane.Markdown("Missing RA and Dec")]
                return 
            sourceId = None 
    
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_spectrum, 
                                     ra = ra, dec = dec, dataset = dataset,
                                     sourceId = sourceId,
                                     max_separation = new_radius,
                                     check_coverage = not sourceId,
                                     )
        
        spectrum_object = future.result()   
    
        if spectrum_object is not None:
            _add_coordinates_to_shared(*spectrum_object.get_coordinates(), panel_id = panel_id,  key_name = dataset)
            plot_model = True if dataset != "EuclidSpec" else False
            kwargs = {"width" : 950,  "height" : 250 if spectrum_object.available_spectra > 1 else 300}
            plot = spectrum_object.plot_all_spectra_hv(plot_model = plot_model, **kwargs)
            plot_instance.container.objects = [plot] 

        else:
            plot_instance.container.objects = [pn.pane.Markdown("No spectrum found.")]
        
        
    initial_radius = shared_data.get_data("Euclid_radius", 0.5)
    update_plot(initial_radius, panel_id = plot_instance.panel_id)
        
    if not from_sourceId:
        if not shared_data.is_subscribed(plot_instance.panel_id, "Euclid_radius"):
            shared_data.subscribe(plot_instance.panel_id, "Euclid_radius", partial(update_plot, panel_id = plot_instance.panel_id))
    return plot_instance.container
   

def run_spectrum(ra, dec, dataset = "DESI",
            sourceId= None, max_separation = 0.5, check_coverage = True):
 
    if dataset == "EuclidSpec":
        spectrum_object = EuclidSpectraClass(ra, dec, max_separation = max_separation,
                                             sourceId = sourceId)
    else:
        datasets = (["DESI-DR1"] if dataset == "DESI"
            else ["BOSS-DR16", "SDSS-DR16"] if dataset == "SDSS"
            else None)
        spectrum_object = DESISpectraClass(ra, dec, datasets = datasets ,
                                        sourceId = sourceId, max_separation = max_separation,
                                        client = shared_data.get_data("Sparcl_client", None))
    
    spectrum_object.get_spectra()
  
    if spectrum_object.spectra is not None:
        spectrum_object.get_smoothed_spectra(kernel = "Box1dkernel", 
                                             window = 5 if dataset == "EuclidSpec" else 10)
        return spectrum_object
    return None

def _add_coordinates_to_shared(ra, dec, panel_id, key_name):
    """
    ra and dec are lists
    """
    shared_data.publish(panel_id, f"{key_name}_coordinates", {"ra": ra, "dec": dec})
    return 


def euclid_cutout_plot(data, selected = None, plot_instance = None):

    if not hasattr(plot_instance, "container"):
        plot_instance.container = pn.Column()


    selected_source = get_selected_source(data=data, selected = selected)
    ra, dec = get_ra_dec(selected_source)
    if ra is None or dec is None:
        return empty_panel(message = "Missing Ra and Dec")
    
    initial_radius = shared_data.get_data("Euclid_radius", 5.0)
    
    
    plot_instance.euclid_panel_manager = EuclidPanelManager(ra, dec, radius = initial_radius,
                                                       panel_id = plot_instance.panel_id)
    
    plot_instance.euclid_panel_manager._subscribe_to_shared()

    plot_instance.container.objects = [plot_instance.euclid_panel_manager.panel()]

    
    return plot_instance.container


class EuclidPanelManager:

    """A class that handles the retrieve of Euclid cutouts and its manipulation with 
    the widgets"""
    
    def __init__(self, ra, dec, radius = 5, panel_id = None):
        """radius in arcsec"""
        self.ra = ra
        self.dec = dec
        self.radius = radius
        self.panel_id = panel_id 
        self.overplotted_coordinates = []
        self.euclid_pane = pn.pane.HoloViews(width=400, height=400)
        
        self._initialise_radius_scaling_widgets()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.run_euclid, initialize = True)
        try:
            future.result()  # eventually raise an exception???
        except Exception as e:
            print(f"Error occurred: {e}")

    def _initialise_radius_scaling_widgets(self):
        
        self.radius_input = pn.widgets.FloatInput(name = "Radius [arcsec]", value = self.radius,
                                                  step = 0.5, start = 1, end = 100, 
                                                  sizing_mode="scale_width")
        self.radius_input.param.watch(self._update_radius, "value")

        
        self.stretching_input = pn.widgets.Select(name = "Stretching function", 
                                                options=  ['Linear', 'Sqrt', 'Log', 'Asinh', 'PowerLaw'],
                                                sizing_mode = "scale_width")
        
        self.stretching_input.param.watch(self._update_stretching, "value")

        
        self.contrast_scaler = pn.widgets.RangeSlider(name = "Image scaling", 
                                                    start = 0, end = 1,value = (0,1), step = 0.004, 
                                                    sizing_mode = "scale_both")
        
        self.contrast_scaler.param.watch(self._update_intensity_scaling, "value")  

        self.overplot_coords_widget = pn.widgets.Checkbox(name = "Spectrum Coordinates")
        self.overplot_coords_widget.param.watch(self._overplot_coordinates_callback, "value")

     
    def _update_radius(self, event):
        if event.new: #avoid passing None
            self.radius = event.new
            shared_data.publish(self.panel_id, "Euclid_radius", self.radius)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.run_euclid, initialize = False)
            try:
                future.result()  # This would raise the exception
            except Exception as e:
                print(f"Error occurred: {e}")
            self._update_image()
        else:
            print("Input a valid value for radius")

    @staticmethod
    def change_intensity_range(image, low, high):
        image = np.clip(image, low, high)
        image = (image-low)/(high-low)
        return np.clip(image, 0,1)

    def _update_intensity_scaling(self, event):
        low, high = event.new
        scaled_image = self.change_intensity_range(self.euclid_object.reprojected_data["stacked"], 
                                                   low, high)
        
        self.get_euclid_figure(scaled_image)

        self._update_image()
    
    def _update_stretching(self, event):
        stretch = event.new
        self.euclid_object.stack_cutouts(stretch = stretch)
        self.get_euclid_figure(self.euclid_object.reprojected_data["stacked"])
        self._update_image()

    
    def _update_image(self): 
        try:
             self.euclid_pane.object = hv.Overlay(self.euclid_fig+ self.overplotted_coordinates)
        except Exception as e:         #too generic
            print("Euclid image unavailable")
            print(e)

    def _add_coordinates(self, coordinates, dataset):
        """Storing Coordinates from DESI/SDSS
           coordinates : dict : {"ra" : [...], "dec" : [...]} 
           dataset : string, key of the dictionary storing the coordinates
        """

        if not coordinates or "ra" not in coordinates or "dec" not in coordinates:
            print("Wrong passed coordinates")
            return
        ra, dec  = coordinates["ra"], coordinates["dec"]
        if not hasattr(self, "stored_spectrum_coordinates"):
            self.stored_spectrum_coordinates = {}
        self.stored_spectrum_coordinates[dataset] = {"ra" : ra, "dec" : dec}

        self.overplot_coords_widget.name = "Spectrum Coordinates"
        if self.overplot_coords_widget.value:
            self._show_overplot_coordinates()
    
    def _show_overplot_coordinates(self):

        if hasattr(self, "stored_spectrum_coordinates"):
            self.overplot_coords_widget.name = "Spectrum Coordinates"
            if self.overplot_coords_widget.value:
                for dataset in self.stored_spectrum_coordinates:
                    print(f"overplotting coordinates for {dataset}")
                    N = len(self.stored_spectrum_coordinates[dataset]["ra"])
                    colors = plt.get_cmap("gist_rainbow", max(N,2))
                    marker = "+" if dataset == "DESI" else "*" #TODO improve
                    self.overplotted_coordinates = []
                    for i, (x, y) in enumerate(self.euclid_object.world_2_pix(ra =  self.stored_spectrum_coordinates[dataset]["ra"],
                                                                              dec = self.stored_spectrum_coordinates[dataset]["dec"],
                                                                              filtro = "stacked" )):

                        if (0 <= x < self.image_width) and (0 <= y < self.image_height):
                            self.overplotted_coordinates.append(hv.Points([(x,y)]).opts(
                                                               color = colors(i),
                                                                 marker = marker, 
                                                                 size = 20,
                                                                 ))

    def _overplot_coordinates_callback(self, event):
        if event.new:
            if not hasattr(self, "stored_spectrum_coordinates"):
                print("No spectrum coordinates available")
                event.obj.name = "Spectrum Coordinates [Not Currently Avaliable]"
                self.overplotted_coordinates = []
                return None
            
            event.obj.name = "Spectrum Coordinates"
            self._show_overplot_coordinates()

        elif not event.new:
            self.overplotted_coordinates = []

        self._update_image()
        

        
    def get_plot_scale(self):
        bar_length_arcsecond = self.bar_length_pixels * self.euclid_object.arcsec_per_pix["stacked"]
        return bar_length_arcsecond

    
    def get_euclid_figure(self, data, show_scale = True):
        
        self.image_height, self.image_width = data.shape[:2]
        bounds = (0, 0, self.image_height, self.image_width)

        image = hv.RGB(data[::-1,...], bounds=bounds).opts(
                                    active_tools =[], toolbar=None,
                                    padding = 0,
                                    border = 0,
                                    framewise = True,
                                    xaxis=None, 
                                    yaxis=None,
                                    )

        self.euclid_fig = [image]
        
        if show_scale:
            self.bar_length_pixels = self.image_width * 0.2  #always shows a bar 1/5 of the plot 
            x0, y0 = 0.1*self.image_width, 0.1*self.image_height
            x1 = x0 + self.bar_length_pixels
            scale_bar = hv.Curve(([x0, x1], [y0, y0])).opts(color='red', line_width=3)
            scale_text = hv.Text(x=(x0 + x1)/2, y=y0 + y0/2,
                            text=f'{self.get_plot_scale():.1f}"').opts(
                            text_color='red', text_align='center',
                            text_baseline='bottom', fontsize=14
                            )
            self.euclid_fig.extend([scale_bar, scale_text])
            
        if self.overplot_coords_widget.value:
            self._show_overplot_coordinates()
    
    
    def run_euclid(self, initialize = False):
        """Wrapper for multithreading"""
        if initialize:
            self.euclid_object = EuclidCutoutsClass(self.ra, self.dec, 
                             euclid_filters= ["VIS", "NIR_Y", "NIR_H"])
        
        self.euclid_object.get_final_cutout(radius = self.radius, 
                                            stretch = "Linear", reference = "VIS", 
                                            verbose = True)
          
        if not initialize:
            self.contrast_scaler.value = (0,1)
        
        self.overplot_coords_widget.value = False
        self.get_euclid_figure(self.euclid_object.reprojected_data["stacked"])                                                                        
 

    def _subscribe_to_shared(self):
        """It manages all the subscriptions to the shared dictionary. not very flexible but it works"""
        desi_callback = lambda coords: self._add_coordinates(coords, "DESI")
        sdss_callback = lambda coords: self._add_coordinates(coords, "SDSS")
        euclid_callback = lambda coords: self._add_coordinates(coords, "EuclidSpec")
        shared_data.replace_subscribe(self.panel_id, "DESI_coordinates", desi_callback)
        shared_data.replace_subscribe(self.panel_id, "SDSS_coordinates", sdss_callback)
        shared_data.replace_subscribe(self.panel_id, "EuclidSpec_coordinates", euclid_callback)
        
        #If DESI/SDSS panel are already initialized, I need to pass the coordinates directly
        if shared_data.get_data("DESI_coordinates"):
            self._add_coordinates(shared_data.get_data("DESI_coordinates"), "DESI")
        if shared_data.get_data("SDSS_coordinates"):
            self._add_coordinates(shared_data.get_data("SDSS_coordinates"), "SDSS")
        if shared_data.get_data("EuclidSpec_coordinates"):
            self._add_coordinates(shared_data.get_data("EuclidSpec_coordinates"), "EuclidSpec")

    
    def panel(self):
            self.panel_column = pn.Column(pn.Row(self.euclid_pane, 
                                                pn.Column(self.stretching_input, 
                                                        self.radius_input,
                                                        self.contrast_scaler,
                                                        self.overplot_coords_widget)
                                                ),
                                        )
            self._update_image()
            return self.panel_column


def vlass_cutout_plot(data, selected): 
    selected_source = get_selected_source(data=data, selected = selected)
    ra, dec = get_ra_dec(selected_source)
    if ra is None or dec is None:
        return empty_panel(message = "Missing Ra and Dec")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(VLASS_cutout, ra=ra, dec=dec, 
                                 radius=shared_data.get_data("Euclid_radius", 10), verbose = True)
        vlass_image = future.result()
    if vlass_image is not None:
        fig = Figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_axis_off()
        ax.imshow(vlass_image, origin = "lower")
        return pn.pane.Matplotlib(fig)
    else:
        return empty_panel(message = "Missing VLASS image")
    

def lotss_cutout_plot(data, selected): 
    selected_source = get_selected_source(data=data, selected = selected)
    ra, dec = get_ra_dec(selected_source)
    if ra is None or dec is None:
        return empty_panel(message = "Missing Ra and Dec")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(LoTSS_cutout, ra=ra, dec=dec, 
                                 radius=shared_data.get_data("Euclid_radius", 10))
        lotss_image = future.result()
    if lotss_image is not None:
        fig = Figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_axis_off()
        ax.imshow(lotss_image, origin = "lower")
        return pn.pane.Matplotlib(fig)
    else:
        return empty_panel(message = "Missing LoTSS image")

def local_stored_plot(data, selected):
    selected_source = get_selected_source(data=data, selected = selected)
    path = int(selected_source[config.settings["Local_image_path"]].iloc[0])
    try:
        return pn.pane.Image(path, width = 500)
    except Exception as e:
        print(e)
        return empty_panel()
    

def debug_plot_publisher(data, selected = None, plot_instance = None):
    if not hasattr(plot_instance, "container"):
        plot_instance.container = pn.Column()

    print(f"DEBUG: debug_plot_publisher called with plot_instance: {plot_instance}")
    print(f"DEBUG: plot_instance has panel_id: {hasattr(plot_instance, 'panel_id') if plot_instance else False}")

    input_widget = pn.widgets.IntInput(name='Debug input', value=5, step=1, start=0, end=2000)
    
    def input_widget_cb(event):
        value = event.new
        shared_data.publish(plot_instance.panel_id, "Debug_value", value)
        print(f"Publishing new value = {shared_data.get_data('Debug_value', 'ERROR')}")
    
    input_widget.param.watch(input_widget_cb, "value_throttled")
    plot_instance.container.objects = [input_widget]
    
    
    if not shared_data.is_subscribed(plot_instance.panel_id, "Definitely_Not_a_Key"):
           shared_data.subscribe(plot_instance.panel_id, "Definitely_Not_a_Key", lambda : print("Something has gone wrong"))  
    shared_data.publish(plot_instance.panel_id, "Debug_value", input_widget.value)
    print(f"Publishing new value = {shared_data.get_data('Debug_value', 'ERROR')}")
    return plot_instance.container


def debug_plot_subscriber(data, selected = None, plot_instance = None):
    if not hasattr(plot_instance, "container"):
        plot_instance.container = pn.Column()

    def update_plot(value, panel_id = None):
        text = f"## The value on the screen is {value}"
        text = text + f" and panel id = {panel_id}"
        print("we call the callback")
        plot_instance.container.objects = [pn.pane.Markdown(text)]

        
    initial_value = shared_data.get_data("Debug_value", 1000)
    update_plot(initial_value, panel_id = plot_instance.panel_id)
    

    if not shared_data.is_subscribed(plot_instance.panel_id, "Debug_value"):
        print("Suscribing to Debug panel")
        shared_data.subscribe(plot_instance.panel_id, "Debug_value", partial(update_plot, panel_id = plot_instance.panel_id))
    return plot_instance.container


""" def get_grid_shape(N):
    Returns the number of rows and column to add to the figure hosting
    the spectra depending on the number N of spectra to be plotted
    if N == 1:
        return (1, 1)
    elif N == 2:
        return (1, 2)
    elif N <= 10:
        ncols = 2
    else:
        ncols = 3
    nrows = np.ceil(N / ncols).astype(int)
    return (nrows, ncols) """

""" def get_desi_figure(desi_object, plot_emlines= True, plot_abslines = True):
     
    N = desi_object.available_spectra
    nrows, ncols = get_grid_shape(N)

    fig_width = 6.5 * ncols
    fig_height = 3 * nrows

    colors = plt.get_cmap("gist_rainbow", max(N,2)) #one spectrum-->red
 
    desi_fig = Figure(figsize = (fig_width, fig_height))
    for i in range(N):
        row = i // ncols
        col = i % ncols
        ax_idx = row * ncols + col + 1  
        ax = desi_fig.add_subplot(nrows, ncols, ax_idx)
        desi_object.plot_spectrum(ax, idx=i, plot_emlines = plot_emlines, plot_abslines = plot_abslines, 
                                  annotate_emlines = True, annotate_abslines = True,
                                  set_ylabel= (N==1), model_kwargs = {"lw" : 2, "color" : colors(i)})
        if row < nrows - 1:
            ax.tick_params(labelbottom=False)
        if ncols == 2 and col == 1:
            ax.tick_params(labelleft=False)

    if N == 1:
        #TODO: include information also for multiple spectra
        ax.text(0, 1.02, f"Dataset = {desi_object.spectra[0].data_release}", fontsize = 11, transform = ax.transAxes)
        ax.text(0.33, 1.02, f"SpecType = {desi_object.spectra[0].spectype}", fontsize = 11, transform = ax.transAxes)
        ax.text(0.66, 1.02, f"z = {np.round(desi_object.spectra[0].redshift,4)}", fontsize = 11,transform = ax.transAxes)
    
    elif N > 1:
        desi_fig.subplots_adjust(hspace=0, wspace = 0.01)
        desi_fig.text(0.04, 0.5, r'$F_{\lambda}~[10^{-17}~ergs~s^{-1}~cm^{-2}~{\AA}^{-1}]$', fontsize =15, 
                      va='center', rotation='vertical');
    return desi_fig """