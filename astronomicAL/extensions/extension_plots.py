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
import param


def get_plot_dict():

    plot_dict = {
        "AGN Wedge": CustomPlot(
            agn_wedge, ["Log10(W3_Flux/W2_Flux)", "Log10(W2_Flux/W1_Flux)"]
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
    }

    return plot_dict


class CustomPlot:
    def __init__(self, plot_fn, extra_features):

        self.plot_fn = plot_fn
        self.extra_features = extra_features
        self.row = pn.Row("Loading...")

    def create_settings(self, unknown_cols):
        print("creating settings...")
        print(unknown_cols)
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
            return self.plot_fn


def create_plot(
    data,
    x,
    y,
    plot_type="scatter",
    selected=None,
    label_plot=True,
    colours=True,
    smaller_axes_limits=False,
    bounds=None,
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
            tools=["box_select"],
            active_tools=["pan", "wheel_zoom"],
        )

    if colours:
        color_key = config.settings["label_colours"]

        color_points = hv.NdOverlay(
            {
                config.settings["labels_to_strings"][f"{n}"]: hv.Points(
                    [0, 0], label=config.settings["labels_to_strings"][f"{n}"]
                ).opts(style=dict(color=color_key[n], size=0))
                for n in color_key
            }
        )
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

    if selected is not None:
        plot = plot * selected_plot

    if label_plot:
        plot = plot * color_points

    return plot


def bpt_plot(data, selected=None, **kwargs):

    plot_NII = create_plot(
        data,
        config.settings["Log10(NII_6584_FLUX/H_ALPHA_FLUX)"],
        config.settings["Log10(OIII_5007_FLUX/H_BETA_FLUX)"],
        plot_type="scatter",
        label_plot=True,
        selected=selected,
        bounds=[-1.8, 1.25, 1, -2.2],
    )

    x1 = np.linspace(-1.6, -0.2, 60)
    x2 = np.linspace(-1.6, 0.2, 60)
    y1 = (0.61 / (x1 - 0.05)) + 1.3
    y2 = (0.61 / (x2 - 0.47)) + 1.19

    l1 = pd.DataFrame(np.array([x1, y1]).T, columns=["x", "y"])
    l2 = pd.DataFrame(np.array([x2, y2]).T, columns=["x", "y"])

    NII_line1 = create_plot(
        l1, "x", "y", plot_type="line", label_plot=False, colours=False
    )

    NII_line2 = create_plot(
        l2, "x", "y", plot_type="line", label_plot=False, colours=False
    )

    plot_NII = plot_NII * NII_line1 * NII_line2

    plot_SII = create_plot(
        data,
        config.settings["Log10(SII_6717_FLUX/H_ALPHA_FLUX)"],
        config.settings["Log10(OIII_5007_FLUX/H_BETA_FLUX)"],
        plot_type="scatter",
        label_plot=True,
        selected=selected,
        bounds=[-2.1, 1.2, 0.9, -2.1],
    )

    x1 = np.linspace(-2, 0.1, 60)
    y1 = (0.72 / (x1 - 0.32)) + 1.30

    l1 = pd.DataFrame(np.array([x1, y1]).T, columns=["x", "y"])

    SII_line1 = create_plot(
        l1, "x", "y", plot_type="line", label_plot=False, colours=False
    )

    plot_SII = plot_SII * SII_line1

    plot_OI = create_plot(
        data,
        config.settings["Log10(OI_6300_FLUX/H_ALPHA_FLUX)"],
        config.settings["Log10(OIII_5007_FLUX/H_BETA_FLUX)"],
        plot_type="scatter",
        label_plot=True,
        selected=selected,
        bounds=[-3.3, 1.25, 1.65, -2.3],
    )

    x1 = np.linspace(-3, -0.8, 60)
    y1 = (0.73 / (x1 + 0.59)) + 1.33

    l1 = pd.DataFrame(np.array([x1, y1]).T, columns=["x", "y"])

    OI_line1 = create_plot(
        l1, "x", "y", plot_type="line", label_plot=False, colours=False
    )

    plot_OI = plot_OI * OI_line1

    tabs = pn.Tabs(
        ("NII", plot_NII),
        ("SII", plot_SII),
        ("OI", plot_OI),
    )

    return tabs


def agn_wedge(data, selected=None, **kwargs):

    plot = create_plot(
        data,
        config.settings["Log10(W3_Flux/W2_Flux)"],
        config.settings["Log10(W2_Flux/W1_Flux)"],
        plot_type="scatter",
        label_plot=True,
        selected=selected,
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

    p1 = create_plot(top, "x", "y", plot_type="line", label_plot=False, colours=False)
    p2 = create_plot(
        bottom, "x", "y", plot_type="line", label_plot=False, colours=False
    )
    p3 = create_plot(
        threshold, "x", "y", plot_type="line", label_plot=False, colours=False
    )

    plot = plot * p1 * p2 * p3

    return plot
