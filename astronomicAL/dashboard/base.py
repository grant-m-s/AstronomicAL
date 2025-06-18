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
import threading
import time
import param
import uuid



class CustomPlotClass(param.Parameterized):

    stage = param.ObjectSelector(default="column_selection", objects=["column_selection", "plot"])
    
    def __init__(self, src, extra_features, close_button):
        super().__init__()
        self.src = src
        self.extra_features = extra_features
        self.close_button = close_button
        self.get_unknown_columns()
        self.figure = pn.pane.HoloViews(sizing_mode="stretch_both", min_height = 400)
        self.loading_pane = pn.pane.Markdown("", sizing_mode="stretch_both", max_height = 30)
        
    
    def _confirm_button_cb(self, event):
        for col, widget in self.select_widgets.items():
            print(col, widget.value)
        self.stage = "plot"
    

    def column_selection_panel(self):
        settings_grid = pn.GridBox(ncols=3, sizing_mode = "stretch_width", scroll = True)  
        options = list(config.main_df.columns)
        submit_button = pn.widgets.Button(name='Confirm', button_type='primary', max_height=120)
        submit_button.on_click(self._confirm_button_cb)
        for col in self.unknown_columns:
            select_widget = pn.widgets.Select(name= col, options=options, max_height=120, sizing_mode = "stretch_width")
            settings_grid.append(select_widget)
            self.select_widgets[col] = select_widget
        return pn.Card(settings_grid, header = pn.Row(pn.Spacer(width=25), self.close_button, submit_button),
                                sizing_mode="stretch_both", scroll=True, collapsible = False, min_height = 300 )
    
    def get_unknown_columns(self):
        current_cols = config.main_df.columns
        self.unknown_columns = []
        for col in self.extra_features:
            if col not in list(config.settings.keys()):
                print(f"{col} not in config")
                if col not in current_cols:
                    print(f"{col} not in df")
                    self.unknown_columns.append(col)
                else:
                    config.settings[col] = col
        if len(self.unknown_columns) > 0:
            self.select_widgets = {}
        else:
            self.stage = "plot"
    
    def plot(self, N=20):
        time.sleep(5)
        coords = [(i, np.random.random()) for i in range(N)]
        scatter = hv.Scatter(coords).opts(color='black', marker='+')
        self.figure.object = scatter
        self.loading_pane.visible = False

    
    def get_layout(self):
        points_input = pn.widgets.IntInput(name="Number of points", value=20, start=1, sizing_mode = "stretch_width" )
        def update_points(event):
            N = points_input.value
            self.plot_async()
        points_input.param.watch(update_points, 'value')
        self.plot_async(points_input.value)
        return pn.Column(self.loading_pane, self.figure, points_input, sizing_mode="stretch_both", min_height = 450,
                          styles={'background': 'lightgreen'})
    
    def plot_async(self, N=20):
        self.loading_pane.object = "### Loading plot, please wait..."
        self.loading_pane.visible = True
        threading.Thread(target=self.plot, args=(N,), daemon=True).start()


    def plot_panel(self):
        self.layout = self.get_layout()
        return pn.Card(self.layout, header = pn.Row(pn.Spacer(width=25,),self.close_button),
                       collapsible = False, sizing_mode="stretch_both", min_height =450,
                       styles={'background': 'lightblue'})
    
    @param.depends("stage")                        
    def mypanel(self):
        if self.stage == "column_selection":
            return self.column_selection_panel()
        else:
            return self.plot_panel()