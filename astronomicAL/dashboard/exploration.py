import panel as pn
pn.extension('tabulator')
import param
import pandas as pd
import os
import uuid

from functools import partial


import astronomicAL.config as config
from astronomicAL.extensions.shared_data import shared_data


class ExplorationDashboard(param.Parameterized):

    index = param.Integer(default=0, bounds=(0, 0))

    def __init__(self, src, df, switch_mode_button, **params):
        super().__init__(**params)
        self.src = src
        self.df = df
        self.visited_indices = [self.index]  
        self.panel_id = str(uuid.uuid4()) 
        self.current_position = 0 
        self.param.index.bounds = (0, len(self.df) - 1)
        
        self._switch_mode_button = switch_mode_button
        self.prev_button = pn.widgets.Button(name="Previous", button_type="primary", max_height = 80, max_width=150)
        self.next_button = pn.widgets.Button(name="Next", button_type="primary",  max_height = 80, max_width=150)
        self.objectid_input = pn.widgets.TextInput(name = "SourceId", value = "")

        self.extra_info_pane = pn.pane.DataFrame(self._get_extra_info_df(), index = False, header = False,
                                                 sizing_mode="stretch_both")
        
        self.prev_button.on_click(self._go_previous)
        self.next_button.on_click(self._go_next)
        self.objectid_input.param.watch(self._find_from_id, "value")
        self._update_navigation_flags()
        self._subscribe_to_shared()


    @param.depends('index', watch=True)
    def _update_history(self):
        if self.visited_indices[self.current_position] == self.index:
            return
        if self.index != self.visited_indices[-1]:
            self.visited_indices.append(self.index)

        self.current_position = len(self.visited_indices) - 1
        self._update_navigation_flags()
        self.objectid_input.value =""


    def _go_previous(self, event):
        if self.current_position > 0:
            self.current_position -= 1
            self.index = self.visited_indices[self.current_position]
        self._update_navigation_flags()


    def _go_next(self, event):
        if self.current_position < len(self.visited_indices) - 1:
            self.current_position += 1
            self.index = self.visited_indices[self.current_position]
        else:
            new_index = self.index + 1
            if new_index <= len(self.df) - 1:
                self.index = new_index
            else:
                self.index = 0
        self._update_navigation_flags()

    
    def _find_from_id(self, event):
        sourceid = event.new
        sourceid = sourceid.strip() 
        if not sourceid:
            return
        try:
            matches = self.df[config.settings["id_col"]].str.contains(sourceid, case = True)
        except AttributeError:
            matches = self.df[config.settings["id_col"]].astype(str).str.contains(sourceid, case = True)
        
        N_matches = matches.sum()
        if N_matches == 1:
            self.index = self.df[matches].index[0]
        elif N_matches == 0:
            print("No matches found")
        else:
            #avoid cases where the exact Id is also contained in some other ids
            exact_matches = self.df[config.settings["id_col"]].astype(str) == sourceid
            N_exact = exact_matches.sum()
            if N_exact == 1:
                self.index = self.df[exact_matches].index[0]
            elif N_exact > 1:
                print(f"There are {N_exact} sources which exactly match the provided sourceId")
            else:
               print(f"There are {N_matches} sources containing the provided sourceId, be more specific") 

    
    def _multithread_running_cb(self, is_running):
        self.prev_button.disabled = is_running
        self.next_button.disabled = is_running
       
    def _subscribe_to_shared(self):
        panels_using_multithread = ["EuclidCutout", "EuclidSpec", "DESI", "SDSS"]
        for panel in panels_using_multithread:
            shared_data.replace_subscribe(self.panel_id, f"{panel}_running", self._multithread_running_cb)
         
    def _update_navigation_flags(self):
        self.prev_button.disabled = self.current_position == 0
        self.next_button.disabled = False

    def _get_extra_info_df(self):
        extra_data_list = [["SourceId", str(self.src.data[config.settings["id_col"]][0])]]
        for col in config.settings["extra_info_cols"]:
            value = self.src.data[f"{col}"][0]
            if isinstance(value, float) and value < 1e4:
                value = float(f"{value:.6g}")
            extra_data_list.append([col, value])
        return  pd.DataFrame(extra_data_list, columns=["Column", "Value"])

    
    @param.depends('index', watch=True)
    def _update_selected_src(self):
        self.src.data = self.df.iloc[[self.index]].to_dict("list")
        self.extra_info_pane.object = self._get_extra_info_df()


    def get_layout(self):
        return pn.Column(pn.Param(self, parameters = ["index"], widgets={"index": pn.widgets.IntInput}),
                         self.extra_info_pane,
                         self.objectid_input,
                         pn.Row(self.prev_button,self.next_button), sizing_mode = "stretch_both")


    def mypanel(self):
        layout = self.get_layout()
        return pn.Card(layout, 
                      header = pn.Row(pn.Spacer(width=25,), self._switch_mode_button),
                        collapsible = False, sizing_mode="stretch_both")
    

    def remove_shared_data(self):
        """Removes subscriptions and published data from the shared data"""
        shared_data.cleanup_extension_panel(self.panel_id)
        print(f"[{self.panel_id}] removed from shared data")

    #def remove_src_listener(self):
    #    """Removes the callback to a change in the selected source"""
    #    if self.src is not None and hasattr(self, "_src_callback"):
    #        try:
    #            self.src.remove_on_change("data", self._src_callback)
    #            print(f"[{self.panel_id}] Listener removed")
    #        except Exception as e:
    #            print(f"[{self.panel_id}] Error removing src listener: {e}")

    #def remove_column_selection(self):
    #    if hasattr(self, "unknown_columns"):
    #        for col in self.unknown_columns:
    #            if col in config.settings:
    #                del config.settings[col]
    #        print(f"[{self.panel_id}] unknown columns selected removed from config")
            
    def cleanup_panel_plot(self):
        self.remove_shared_data()


