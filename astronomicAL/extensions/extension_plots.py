from datetime import datetime
from holoviews.operation.datashader import (
    datashade,
    dynspread,
)

import datashader as ds
import holoviews as hv

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
import astronomicAL.extensions.extension_plots_shared as shared
from astronomicAL.extensions.astro_data_utility import DESISpectraClass, EuclidCutoutsClass
from astronomicAL.extensions.astro_data_utility import VLASS_cutout, LoTSS_cutout


def get_plot_dict():

    plot_dict = {
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

        "Euclid Cutout" : CustomPlot(euclid_cutout_plot, []),

        "DESI Spectra from Coords" : CustomPlot(spectrum_plot, [], dataset="DESI", from_specid=False),

        "DESI Spectrum from ID" : CustomPlot(spectrum_plot, ["DESI_TargetID"], dataset="DESI", from_specid=True),

        "Debug" : CustomPlot(debug_shared_dataframe, []),

        "SDSS Spectra from Coords" : CustomPlot(spectrum_plot, [], dataset="SDSS", from_specid=False),

        "SDSS Spectrum from ID" : CustomPlot(spectrum_plot, ["SDSS_TargetID"], dataset="SDSS", from_specid=True),

        "VLA-VLASS Cutout" : CustomPlot(vlass_cutout_plot, []),

        "LOFAR-LoTSS Cutout" : CustomPlot(lotss_cutout_plot, []),

        "Stored Image"  : CustomPlot(local_stored_plot, ["Local_image_path"])
    }

    return plot_dict


class CustomPlot:
    def __init__(self, plot_fn, extra_features, *plot_fn_args, **plot_fn_kwargs):

        self.plot_fn = plot_fn
        self.extra_features = extra_features
        self.row = pn.Row("Loading...")
        self.plot_fn_args = plot_fn_args
        self.plot_fn_kwargs = plot_fn_kwargs

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
            return lambda *args, **kwargs: self.plot_fn(*args, *self.plot_fn_args, **kwargs, **self.plot_fn_kwargs)


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


def bpt_plot(data, selected=None):

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


def mateos_2012_wedge(data, selected=None):

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
    return pn.MarkDown(message)
  

def get_ra_dec(selected_source):
    if check_required_column(selected_source, "ra_dec"):
        ra_dec = selected_source["ra_dec"][0]
        ra = float(ra_dec[: ra_dec.index(",")])
        dec = float(ra_dec[ra_dec.index(",") + 1 :])
    else:
        print("No ra and dec available for this source")
        ra, dec = None, None
    return ra, dec




def spectrum_plot(data, selected = None, dataset = "DESI", from_specid = False):
    selected_source = get_selected_source(data=data, selected = selected)
    
    if from_specid:
        if config.settings[f"{dataset}_TargetID"] in selected_source.columns:
            specId = int(selected_source[config.settings[f"{dataset}_TargetID"]].iloc[0])
            ra, dec = None, None
        else:
            print("Missing column with target ID")
            return empty_panel(message = "Missing Target ID")
    else:
        ra, dec = get_ra_dec(selected_source)
        if (ra is None) or (dec is None):
            return empty_panel(message = "Missing Ra and Dec")
        specId = None 
    
    if dataset == "DESI":
        datasets = ["DESI-DR1"]
    elif dataset == "SDSS":
        datasets = ["BOSS-DR16", "SDSS-DR16"]
    else:
        datasets = None
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_desi, ra = ra, dec = dec, datasets = datasets,
                                 specId = specId, 
                                 max_separation = shared.shared_data.get("Euclid_radius", 0.5),
                                 client = shared.shared_data.get("Sparcl_client", None))
        
    desi_object = future.result()   ###everything is named desi but works also for SDSS
    
    if desi_object is not None:
        _add_coordinates_to_shared(*desi_object.get_coordinates(), key_name = dataset)
        is_star = (desi_object.available_spectra ==1) and (desi_object.spectra[0].spectype == "STAR")
        plot = get_desi_figure(desi_object, plot_emlines = (not is_star), plot_abslines = is_star)
        #plot = get_desi_figure_hv(desi_object, plot_emlines = ~is_star, plot_abslines=is_star)
        return plot
    else:
        return None


def run_desi(ra, dec, datasets = ["DESI-DR1", "DESI-EDR", "BOSS-DR16", "SDSS-DR16"],
            specId = None, max_separation = 0.5, 
            client = None):
    desi_object = DESISpectraClass(ra, dec, datasets = datasets ,
                                   specId = specId, max_separation = max_separation,
                                   client = client)
    desi_object.get_spectra()
    
    if desi_object.spectra is not None:
        desi_object.get_smoothed_spectra(kernel = "Box1dkernel", window = 10)
        return desi_object
    return None

def get_desi_figure(desi_object, plot_emlines= True, plot_abslines = True):
     
    N = desi_object.available_spectra
    ax_height = 3 if N <= 3 else 2  #if too many spectra, make the subplots smaller
    colors = plt.get_cmap("gist_rainbow", N)
    if N > 4:
        ncols = 2                           #### two columns layout if many spectra
        nrows = np.ceil(N/2).astype(int)   
    else:
        ncols = 1
        nrows = N


    desi_fig = Figure(figsize = (7.5 * ncols, ax_height * nrows))
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
    return desi_fig

def get_desi_figure_hv(desi_object):
    desi_fig = desi_object.plot_spectrum_hv(plot_model = True, plot_emlines = True)
    return desi_fig

def _add_coordinates_to_shared(ra, dec, key_name):
    """
    ra and dec are lists
    """
    shared.shared_data[f"{key_name}_ra"] = ra
    shared.shared_data[f"{key_name}_dec"] = dec


def euclid_cutout_plot(data, selected = None, radius =5):
    selected_source = get_selected_source(data=data, selected = selected)
    ra, dec = get_ra_dec(selected_source)
    if ra is None or dec is None:
        return empty_panel(message = "Missing Ra and Dec")
    euclid_panel_manager = EuclidPanelManager(ra, dec, radius = radius, data = data)
    return euclid_panel_manager.panel()


class EuclidPanelManager:

    """A class to be consistent with what I wrote in selected_source so 
    that i can copy/paste that code"""
    
    def __init__(self, ra, dec, radius = 5, width = 300, data = None):
        """radius in arcsec"""
        self.ra = ra
        self.dec = dec
        self.radius = radius
        shared.shared_data["Euclid_radius"] = self.radius
        self._initialise_radius_scaling_widgets(width = int(width))
        self.data = data
        self.euclid_pane = pn.pane.Matplotlib(dpi = 144,
                                alt_text="Image Unavailable",
                                sizing_mode="stretch_both",
                                interactive = False,
                                tight = True,
                                width = int(width))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.run_init)
        try:
            future.result()  # eventually raise an exception???
        except Exception as e:
            print(f"Error occurred: {e}")

        return None
        

    def _initialise_radius_scaling_widgets(self, width = 300):
        
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
        self.overplot_coords_widget.param.watch(self._add_coordinates, "value") 

     
    def _update_radius(self, event):
        self.radius = event.new
        shared.shared_data["Euclid_radius"] = self.radius
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.run_radius)
        try:
            future.result()  # This would raise the exception
        except Exception as e:
            print(f"Error occurred: {e}")
        #self.add_desi_coordinates()
        self._update_image()

    @staticmethod
    def change_intensity_range(image, low, high):
        image = np.clip(image, low, high)
        image = (image-low)/(high-low)
        return np.clip(image, 0,1)

    def _update_intensity_scaling(self, event):
        low, high = event.new
        scaled_image = self.change_intensity_range(self.euclid_object.reprojected_data["stacked"], 
                                                   low, high)
        self.euclid_image.set_data(scaled_image)
        self.euclid_fig.canvas.draw()
        self._update_image()
    
    def _update_stretching(self, event):
        stretch = event.new
        self.euclid_object.stack_cutouts(stretch = stretch)
        self.euclid_image.set_data(self.euclid_object.reprojected_data["stacked"])
        self.euclid_fig.canvas.draw()
        self._update_image()

    def _update_image(self): 
        try:
             self.euclid_pane.object = self.euclid_fig
        except Exception as e:         #too generic
            print("Euclid image unavailable")
            print(e)
    
    def _add_coordinates_old(self, event, column_names = ["DESI_target", "SDSS_target"], colors = ["red", "blue"]):
        #TODO add color/marker information to differentiate surveys/spectra
        #event.new = self.overplot_coords.value
        if event.new and (self.data is not None) and hasattr(self, "ax"):
            for i, column_name in enumerate(column_names):
                ra_col_name = f"RA_{column_name}"
                dec_col_name = f"Dec_{column_name}"
                if (ra_col_name in self.data.columns) and (dec_col_name in self.data.columns):
                    ra, dec = self.data[ra_col_name].iloc[0],  self.data[dec_col_name].iloc[0]
                    self.euclid_object._add_overplot_coordinates(ra, dec)
                    self.overplot_coordinates(overplot = event)
        elif not event.new:
            self.overplot_coordinates(overplot = event.new)
        return None
    
    def _add_coordinates(self, event):
        #event.new = self.overplot_coords.value
        if event.new and hasattr(self, "ax"):
            ra, dec = [], []
            
            if shared.shared_data.get("DESI_ra"):
                ra += list(shared.shared_data.get("DESI_ra"))
                dec += list(shared.shared_data.get("DESI_dec"))
 
            if shared.shared_data.get("SDSS_ra"): 
                ra += list(shared.shared_data.get("SDSS_ra")) 
                dec += list(shared.shared_data.get("SDSS_dec"))
              
            self.euclid_object._add_overplot_coordinates(ra, dec)
            self.overplot_coordinates(overplot = event)
        
        elif not event.new:
            self.overplot_coordinates(overplot = event.new)
        return None



    def get_euclid_object(self):
        
        self.euclid_object = EuclidCutoutsClass(self.ra, self.dec, 
                            euclid_filters= ["VIS", "NIR_Y", "NIR_H"])
        
        self.euclid_object.get_cone(verbose = True)
        
        return None
    
    def get_euclid_cutout(self, stretch = "Linear", verbose = True):
        
        if not hasattr(self, "euclid_object"):
            self.get_euclid_object()
        
        if len(self.euclid_object.cone_results) > 2:
            self.euclid_object.get_cutouts(radius = self.radius, verbose = verbose)
            self.euclid_object.read_cutouts()
            self.euclid_object.reproject_cutouts(reference = "VIS")
            self.euclid_object.stack_cutouts(stretch = stretch)
        else:
            print(f"No sources in Euclid dataset with {self.ra}, {self.dec} coordinates")
            return None 
        
    def overplot_coordinates(self, overplot = False):

        if overplot:
            image_height, image_width = self.euclid_image.get_size()
            self.overplotted_coordinates = []
            N = len(self.euclid_object.overplot_coordinates["stacked"])
            colors = plt.get_cmap("gist_rainbow", N)
            for i, (x, y) in enumerate(self.euclid_object.overplot_coordinates["stacked"]):
                if (0 <= x < image_width) and (0 <= y < image_height):
                    sc = self.ax.scatter(x, y, color = colors(i), marker = "+", s = 300)
                    self.overplotted_coordinates.append(sc)
        else:
            if isinstance(self.overplotted_coordinates, list):
                for sc in self.overplotted_coordinates:
                    sc.remove()
                self.overplotted_coordinates =[]
        self._update_image()

        
    def get_plot_scale(self):
        bar_length_arcsecond = self.bar_length_pixels * self.euclid_object.arcsec_per_pix["stacked"]
        return bar_length_arcsecond

    
    def get_euclid_figure(self, show_scale = True, figsize = (5,5)):
        
        if hasattr(self, "euclid_image"):
            self.euclid_image.remove()
            self.euclid_image = self.ax.imshow(self.euclid_object.reprojected_data["stacked"], origin = "lower")


        self.euclid_fig = Figure(figsize=figsize)
        self.ax = self.euclid_fig.add_subplot(1,1,1)
        self.ax.set_axis_off()
        self.euclid_image = self.ax.imshow(self.euclid_object.reprojected_data["stacked"], origin = "lower")
        
        if show_scale:
            image_height, image_width = self.euclid_image.get_size()
            self.bar_length_pixels = image_width * 0.2  #always show a bar 1/5 of the plot 
            x0, y0 = 0.1*image_width, 0.1*image_height
            x1 = x0 + self.bar_length_pixels
            self.ax.plot([x0, x1], [y0, y0], color='red', lw=3)
            self.scale_text =  self.ax.text((x0 + x1) / 2, y0 + y0/2, f'{self.get_plot_scale():.1f}"',
                                        color='red', ha='center', va='bottom', fontsize=14, fontweight='bold')
        return None
    
    def run_init(self):
        """Wrapper for multithreading in __init__"""
        self.get_euclid_object()
        self.get_euclid_cutout()
        self.get_euclid_figure()
        self.overplot_coords_widget.value = False
    
    def run_radius(self):
        """Wrapper for multithreading in __update__radius"""
        self.get_euclid_cutout()
        self.contrast_scaler.value = (0,1)
        self.overplot_coords_widget.value = False
        self.get_euclid_figure()
        
    def panel(self):
            self.panel_column = pn.Column(self.euclid_pane,
                                          pn.WidgetBox(pn.Row(self.radius_input, self.stretching_input),
                                                     self.contrast_scaler,
                                                     self.overplot_coords_widget)
                                          )
            self._update_image()
            return self.panel_column


def vlass_cutout_plot(data, selected): 
    selected_source = get_selected_source(data=data, selected = selected)
    ra, dec = get_ra_dec(selected_source)
    if ra is None or dec is None:
        return empty_panel(message = "Missing Ra and Dec")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(VLASS_cutout, ra=ra, dec=dec, radius=10, verbose = True)
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
        future = executor.submit(LoTSS_cutout, ra=ra, dec=dec, radius=10)
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
    
def debug_shared_dataframe(data, selected):
    debug_dict = dict(shared.shared_data)
    if debug_dict.get("Sparcl_client"):
        del  debug_dict["Sparcl_client"]  ###cannot be rendered into a df
    df = pd.DataFrame.from_dict(debug_dict, orient = "index")
    return pn.widgets.DataFrame(df)
