import panel as pn
pn.extension('tabulator')
import param
import os
import uuid
import pandas as pd
import numpy as np

from functools import partial


import astronomicAL.config as config
from astronomicAL.extensions.shared_data import shared_data
from astronomicAL.extensions import feature_generation

class ExplorationDashboard(param.Parameterized):

    index = param.Integer(default=0, bounds=(0, 0))

    def __init__(self, src, df, **params):
        super().__init__(**params)
        self.src = src
        self.df = df
        self.visited_indices = [self.index]  
        self.panel_id = str(uuid.uuid4()) 
        self.current_position = 0 
        self.param.index.bounds = (0, len(self.df) - 1)
        self._preprocess_data()
        self._update_selected_src()
        

        self.prev_button = pn.widgets.Button(name="Previous", button_type="primary", max_height = 50, max_width=100)
        self.next_button = pn.widgets.Button(name="Next", button_type="primary",  max_height = 50, max_width=100)
        self.prev_button = pn.widgets.Button(name="Previous", button_type="primary", max_height = 50, max_width=100)
        self.search_button = pn.widgets.Button(name= "Search", button_type="primary",  max_height = 50, max_width=100)
        self.sourceid_input = pn.widgets.TextInput(name = "SourceId", value = "",  max_height = 50)

        self.extra_info_pane = pn.pane.DataFrame(self._get_extra_info_df(), index = False, header = False,
                                                 sizing_mode="stretch_both" )
        self.prev_button.on_click(self._go_previous)
        self.next_button.on_click(self._go_next)
        self.search_button.on_click(self._search_button_cb)
        self.sourceid_watcher = self.sourceid_input.param.watch(self._sourceid_input_cb, "value", onlychanged=False)
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
        self.sourceid_input.value =""


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
    
    def _search_button_cb(self, event):
        sourceid = self.sourceid_input.value
        if sourceid:
            self._find_from_id(sourceid)


    def _sourceid_input_cb(self, event):
        sourceid = event.new
        if sourceid:
             self._find_from_id(self, sourceid)

    def get_id(self):
        id_col = config.settings["id_col"]
        if id_col == "Use Index":
            return pd.Series(self.df.index, index=self.df.index)  # Ensures it's a Series
        else:
            return self.df[id_col]

    def _find_from_id(self, sourceid):
        sourceid = sourceid.strip() 
        try:
            matches = self.get_id().str.contains(sourceid, case = True)
        except AttributeError:
            matches = self.get_id().astype(str).str.contains(sourceid, case = True)
        
        N_matches = matches.sum()
        if N_matches == 1:
            self.index = self.df[matches].index[0]
        elif N_matches == 0:
            print("No matches found")
        else:
            #avoid cases where the exact Id is also contained in some other ids
            exact_matches = self.get_id().astype(str) == sourceid
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
        panels_using_multithread = ["EuclidCutout", "EuclidSpec", "DESI", "SDSS", "VLASS", "LoTSS"]
        for panel in panels_using_multithread:
            shared_data.replace_subscribe(self.panel_id, f"{panel}_running", self._multithread_running_cb)
        shared_data.replace_subscribe(self.panel_id, "selected_sourceid", self._plot_selected_src_cb)
         
    def _update_navigation_flags(self):
        self.prev_button.disabled = self.current_position == 0
        self.next_button.disabled = False

    def _get_extra_info_df(self):
        id_col = config.settings["id_col"]
        if len(self.src.data[id_col]) > 0:
            source_id = str(self.src.data[id_col][0])
            extra_data_list = [["SourceId", source_id]]
            for col in config.settings["extra_info_cols"]:
                value = self.src.data[f"{col}"][0]
                if isinstance(value, float) and value < 1e4:
                    value = float(f"{value:.6g}")
                extra_data_list.append([col, value])
            return  pd.DataFrame(extra_data_list, columns=["Column", "Value"])
        else:
            cols = ["SourceId"] + config.settings["extra_info_cols"]
            return  pd.DataFrame(cols, columns=["Column"])
    

    def _plot_selected_src_cb(self, sourceid):
        self.sourceid_input.param.unwatch(self.sourceid_watcher)
        self.sourceid_input.value = sourceid
        self.sourceid_watcher = self.sourceid_input.param.watch(self._sourceid_input_cb, "value", onlychanged=False)
        
 
    @param.depends('index', watch=True)
    def _update_src_cb(self):
        self._update_selected_src()
        self.extra_info_pane.object = self._get_extra_info_df()

    def _update_selected_src(self):
        selected_dict = self.df.iloc[[self.index]].to_dict("list")
        if config.settings["id_col"] not in selected_dict:
                selected_dict[config.settings["id_col"]] = [self.index]
        self.src.data = selected_dict

    def _generate_features(self, df):
        """Create the feature combinations that the user specified.
        Parameters
        ----------
        df : DataFrame
            A dataframe containing all of the dataset.

        Returns
        -------
        df : DataFrame
            An expanding dataframe of `df` with the inclusion of the feature
            combinations.
        """

        bands = config.settings["features_for_training"]

        features = bands + [config.settings["label_col"], config.settings["id_col"]]

        oper_dict = feature_generation.get_oper_dict()

        if "feature_generation" in list(config.settings.keys()):
            for generator in config.settings["feature_generation"]:
                oper = generator[0]
                n = generator[1]
                df, generated_features = oper_dict[oper](df, n)
                features = features + generated_features
        return df
    
    def _generate_fake_label_column(self, df):
        """This is not very elegant but allows to keep the code as it is.
           If No labels is selected, it creates a column No labels with all Nan"""
        if config.settings["label_col"] not in df.columns and config.settings["label_col"]=="No Labels":
            df[config.settings["label_col"]] = np.nan
        return df
    

    def _add_ra_dec_col(self, df):

        new_df = df

        ra_col_name = config.settings["ra_col_name"]
        dec_col_name = config.settings["dec_col_name"]

        new_df["ra_dec"] = df[ra_col_name].astype(str) + "," + df[dec_col_name].astype(str)
      
        return new_df


    def _preprocess_data(self):
        """Process all the data according to the config file. In exploring panel it just"
           compute combination of features and creating ra and dec column"""
        self.df = self._generate_fake_label_column(self.df)
        self.df = self._generate_features(self.df)
        self.df = self._add_ra_dec_col(self.df)


    def get_layout(self):
        return pn.Column(pn.Param(self, parameters = ["index"], widgets={"index": pn.widgets.IntInput}),
                         pn.Column(self.extra_info_pane, sizing_mode = "stretch_both", scroll = True, max_height = 200),
                         self.sourceid_input,
                         pn.Row(self.prev_button,self.next_button, self.search_button), sizing_mode = "stretch_both")


    def mypanel(self):
        layout = self.get_layout()
        return pn.Card(layout, 
                      header = pn.Row(pn.Spacer(width=25,),),
                        collapsible = False, sizing_mode="stretch_both")
    

    def remove_shared_data(self):
        """Removes subscriptions and published data from the shared data"""
        shared_data.cleanup_extension_panel(self.panel_id)
        print(f"[{self.panel_id}] removed from shared data")
            
    def cleanup_panel_plot(self):
        self.remove_shared_data()


