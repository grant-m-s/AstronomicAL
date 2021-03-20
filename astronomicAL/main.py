import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../"))
from astronomicAL.utils import save_config
from bokeh.models import ColumnDataSource, TextAreaInput

import astronomicAL.config as config
import holoviews as hv
import pandas as pd
import panel as pn
from functools import partial

hv.extension("bokeh")
hv.renderer("bokeh").webgl = True

files = pn.widgets.FileInput()

react = pn.template.ReactTemplate(title="astronomicAL")

pn.config.sizing_mode = "stretch_both"

if os.path.isfile(config.layout_file):
    react = save_config.create_layout_from_file(react)
else:
    react = save_config.create_default_layout(react)

save_layout_button = pn.widgets.Button(name="Save current layout")

layout_dict = {}
text_area_input = TextAreaInput(value="")
text_area_input.on_change(
    "value",
    partial(
        save_config.save_config_file_cb, trigger_text=text_area_input, autosave=False
    ),
)

save_layout_button.jscallback(
    clicks=save_config.save_layout_js_cb,
    args=dict(text_area_input=text_area_input),
)

react.header.append(pn.Row(save_layout_button))

react.servable()

# TODO :: http://holoviews.org/reference/apps/bokeh/player.html#apps-bokeh-gallery-player
# TODO :: https://panel.holoviz.org/gallery/simple/save_filtered_df.html#simple-gallery-save-filtered-df

# CHANGED :: Keep axis the same on refresh (stop zooming out - outliers)
# CHANGED :: Save React Layout (likely need to wait for Panel update)
# CHANGED :: Fix Datashader Colours to match Bokeh
