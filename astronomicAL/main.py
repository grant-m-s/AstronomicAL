import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../"))
from astronomicAL.utils import load_config
import astronomicAL.config as config
import holoviews as hv
import panel as pn

hv.extension("bokeh")
hv.renderer("bokeh").webgl = True

files = pn.widgets.FileInput()

react = pn.template.ReactTemplate(title="astronomicAL")

pn.config.sizing_mode = "stretch_both"

if os.path.isfile(config.layout_file):
    react = load_config.create_layout_from_file(react)
else:
    react = load_config.create_default_layout(react)

export_fits_file = pn.widgets.Button(
    name="Export Labelled Data to Fits File", disabled=True
)


react.header.append(
    pn.Row(
        config.get_save_layout_button(config.settings["confirmed"], True),
        export_fits_file,
    )
)

react.servable()

# TODO :: http://holoviews.org/reference/apps/bokeh/player.html#apps-bokeh-gallery-player
# TODO :: https://panel.holoviz.org/gallery/simple/save_filtered_df.html#simple-gallery-save-filtered-df

# CHANGED :: Keep axis the same on refresh (stop zooming out - outliers)
# CHANGED :: Save React Layout (likely need to wait for Panel update)
# CHANGED :: Fix Datashader Colours to match Bokeh
