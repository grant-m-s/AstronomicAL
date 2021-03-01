from holoviews.operation.datashader import (
    datashade,
    dynspread,
)

import datashader as ds
import holoviews as hv

import config
import numpy as np
import pandas as pd
import panel as pn
import param


def get_plot_dict():

    plot_dict = {
        "AGN Wedge": agn_wedge,
    }

    return plot_dict


def create_plot(
    data,
    x,
    y,
    plot_type="scatter",
    selected=None,
    label_plot=True,
    colours=True,
    smaller_axes_limits=False,
):

    print(type(data))
    print(data)
    print(data.columns)
    assert x in list(data.columns), f"Column {x} is not a column in your dataframe."
    assert y in list(data.columns), f"Column {y} is not a column in your dataframe."

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
            )
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


def agn_wedge(data, selected=None, **kwargs):

    plot = create_plot(
        data,
        "Log10(W3_Flux/W2_Flux)",
        "Log10(W2_Flux/W1_Flux)",
        plot_type="scatter",
        label_plot=True,
        selected=selected,
    )

    x = data["Log10(W3_Flux/W2_Flux)"]

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
