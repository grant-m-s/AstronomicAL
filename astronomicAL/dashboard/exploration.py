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
from astronomicAL.utils.optimise import matches_type, get_series_type

class ExplorationDashboard(param.Parameterized):

    index = param.Integer(default=0, bounds=(0, 0))

    def __init__(self, src, df, **params):
        super().__init__(**params)
        self.src = src
        self.df = df
        self._running_panels = set() #elements of the set indicate the panel currently using multithreading
        self.visited_indices = [self.index]  
        self.panel_id = str(uuid.uuid4()) 
        self.current_position = 0 
        self.param.index.bounds = (0, len(self.df) - 1)
        self._preprocess_data()
        self._create_extra_info_cols_list()
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
        self._initialise_add_remove_columns_widgets()
        self._initialise_label_selector()
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
             self._find_from_id(sourceid)

    def _get_id(self):
        """Returns all ids in the table"""
        id_col = config.settings["id_col"]
        if id_col == "Use Index":
            return pd.Series(self.df.index, index=self.df.index)  # Ensures it's a Series
        else:
            return self.df[id_col]
        
    def _get_selected_id(self):
        """Returns the sourceID for the selected source"""
        id_col = config.settings["id_col"]
        if len(self.src.data[id_col]) > 0:
            return str(self.src.data[id_col][0])
        
    
    def _find_from_id(self, sourceid):
        sourceid = sourceid.strip() 
        try:
            matches = self._get_id().str.contains(sourceid, case = True)
        except AttributeError:
            matches = self._get_id().astype(str).str.contains(sourceid, case = True)
        
        N_matches = matches.sum()
        if N_matches == 1:
            self.index = self.df[matches].index[0]
        elif N_matches == 0:
            print("No matches found")
        else:
            #avoid cases where the exact Id is also contained in some other ids
            exact_matches = self._get_id().astype(str) == sourceid
            N_exact = exact_matches.sum()
            if N_exact == 1:
                self.index = self.df[exact_matches].index[0]
            elif N_exact > 1:
                print(f"There are {N_exact} sources which exactly match the provided sourceId")
            else:
               print(f"There are {N_matches} sources containing the provided sourceId, be more specific") 

    
    def _multithread_running_cb(self, is_running, panel_name):
        if is_running:
            self._running_panels.add(panel_name)
        else:
            self._running_panels.discard(panel_name)
        any_running = bool(self._running_panels)
        self.prev_button.disabled = any_running
        self.next_button.disabled = any_running
       
    def _subscribe_to_shared(self):
        panels_using_multithread = ["EuclidCutout", "EuclidSpec", "DESI", "SDSS", "VLASS", "LoTSS"]
        for panel in panels_using_multithread:
            shared_data.replace_subscribe(self.panel_id, f"{panel}_running", 
                                          lambda is_running, panel_name=panel: self._multithread_running_cb(is_running, panel_name))
        shared_data.replace_subscribe(self.panel_id, "selected_sourceid", self._selected_src_from_plot_cb)
         
    def _update_navigation_flags(self):
        self.prev_button.disabled = self.current_position == 0
        self.next_button.disabled = False

    def _get_extra_info_df(self):
        id_col = config.settings["id_col"]
        if len(self.src.data[id_col]) > 0:
            source_id = str(self.src.data[id_col][0])
            extra_data_list = [["SourceId", source_id]]
            for col in config.settings["extra_info_cols"]:
                try:
                    value = self.src.data[f"{col}"][0]
                    if isinstance(value, float) and value < 1e4:
                        value = float(f"{value:.6g}")
                    extra_data_list.append([col, value])
                except KeyError:
                    continue
            return  pd.DataFrame(extra_data_list, columns=["Column", "Value"])
        else:
            cols = ["SourceId"] + config.settings["extra_info_cols"]
            return  pd.DataFrame(cols, columns=["Column"])
        
    
    def _save_extra_info_df(self):
        return self._get_extra_info_df()
    
    def _initialise_add_remove_columns_widgets(self):
        self.add_column_button = pn.widgets.Button(name = "Add Col", max_height = 60, max_width =90, sizing_mode = "scale_both")
        self.remove_column_button = pn.widgets.Button(name = "Remove Col", max_height = 60, max_width =90, sizing_mode = "scale_both")
        self.column_selector = pn.widgets.Select(options = [], max_height = 60, max_width =120, visible = False)
        self.selector_watcher = None
        self.add_column_button.on_click(self._add_column_callback)
        self.remove_column_button.on_click(self._remove_column_callback)                      
    
    def _add_column_callback(self, event):
        self.column_selector.value = None
        options = [""] + [i for i in self.df.columns if i not in config.settings["extra_info_cols"]]
        self.column_selector.visible = True
        self.column_selector.options = options
        if self.selector_watcher is not None:
            self.column_selector.param.unwatch(self.selector_watcher)
        self.selector_watcher = self.column_selector.param.watch(self._add_extra_feature, "value")
    
    def _remove_column_callback(self, event):
        self.column_selector.value = None
        options = [""] + list(config.settings["extra_info_cols"])
        self.column_selector.visible = True
        self.column_selector.options = options
        if self.selector_watcher is not None:
            self.column_selector.param.unwatch(self.selector_watcher)
        self.selector_watcher = self.column_selector.param.watch(self._remove_extra_feature, "value")

    def _add_extra_feature(self, event):
        column = event.new
        if column and column not in config.settings["extra_info_cols"]:
            config.settings["extra_info_cols"].append(column)
            self.extra_info_pane.object = self._get_extra_info_df()
            self.column_selector.visible = False

    def _remove_extra_feature(self, event):
        column = event.new
        if column and column in config.settings["extra_info_cols"]:
            config.settings["extra_info_cols"].remove(column)
            self.extra_info_pane.object = self._get_extra_info_df()
            self.column_selector.visible = False

    
    def _selected_src_from_plot_cb(self, sourceid):
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
        if (config.settings["label_col"] not in df.columns) and (config.settings["label_col"] == "No Labels"):
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
        #self.df = self._generate_fake_label_column(self.df)
        #self.df = self._generate_features(self.df)
        self.df = self._add_ra_dec_col(self.df)

    def _create_extra_info_cols_list(self):  
        if "extra_info_cols" not in  config.settings:
            config.settings["extra_info_cols"] = []

    
    def _initialise_label_selector(self):
        label_column_options =  ["No Labels"] + list(self.df.columns)
        self.label_selector = pn.widgets.Select(options = label_column_options, 
                                                value = config.settings["label_col"],
                                                max_height = 60, max_width =120, 
                                                visible = True)
        self.label_selector.param.watch(self._update_labels_cb, "value")
        self.colorpickers_layout = pn.Column()
        self.strings_to_label_layout = pn.Column()
        self.confirm_label_button = pn.widgets.Button(name = "Confirm", button_type = "success",
                                                      max_height = 40, max_width = 80,
                                                       visible = False)
        self.confirm_label_button.on_click(self._confirm_labels_change_cb)
    
    def _update_labels_cb(self, event):
        selected_label_column = event.new
        if (selected_label_column not in self.df.columns) or (selected_label_column == "No Labels"):
            self.labels = []
        else:
            self.label_type = get_series_type(self.df[selected_label_column])
            if self.label_type == "mixed": # strings and numbers are in the column
                self.df[selected_label_column] = self.df[selected_label_column].astype(str)
                self.label_type = "string"
            elif self.label_type == "bool":
                self.df[selected_label_column] = self.df[selected_label_column].astype(int)
                self.label_type = "int"
            if len(self.df[selected_label_column].unique()) > 20:
                print(
                """You have chosen a column with too many unique values (possibly continous) please choose a column with a smaller set of labels (<=20)"""
                )
                return
            
            self.labels = sorted(self.df[selected_label_column].unique())
            self._update_label_strings_input()
            self._update_colours_input()
        self.confirm_label_button.visible = True

    def _confirm_labels_change_cb(self, event):
        self.strings_to_label_layout.clear()
        self.colorpickers_layout.clear()
        config.settings["label_col"] = self.label_selector.value
        config.settings["labels"] = self.labels
        config.settings["labels_to_strings"], config.settings["strings_to_labels"] = self.get_label_strings()
        config.settings["label_colours"] = self.get_label_colours()
        self.confirm_label_button.visible = False
            

    def _update_label_strings_input(self):
        self.strings_to_label_layout.clear()
        self.label_to_strings_param ={}
        for i, data_label in enumerate(self.labels):
            text_input = pn.widgets.TextInput(name=f"{data_label}", placeholder=f"{data_label}")
            self.label_to_strings_param[f"{data_label}"] = text_input
        self.strings_to_label_layout.extend(self.label_to_strings_param.values())


    def _update_colours_input(self):
        self.colorpickers_layout.clear()
        self.colours_param = {}

        colour_list = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        for i, label in enumerate(self.labels):
            picker = pn.widgets.ColorPicker(name= str(label),
                    value=colour_list[i % len(colour_list)],
                    max_width=int(200 / len(self.labels)),)
            self.colours_param[label] = picker
        self.colorpickers_layout.extend(self.colours_param.values())

    def get_label_strings(self):
        labels_to_strings = {}
        strings_to_labels = {}

        for label in self.labels:
            value = self.label_to_strings_param[f"{label}"].value
            if value == "":
                value = str(label)
            labels_to_strings[f"{label}"] = value
            strings_to_labels[f"{value}"] = label
        return labels_to_strings, strings_to_labels
    

    def get_label_colours(self):
        colours = {key : param_obj.value for key, param_obj in self.colours_param.items()}
        return colours


    
    def get_layout(self):
        return pn.Column(pn.Param(self, parameters = ["index"], widgets={"index": pn.widgets.IntInput}),
                         pn.Column(self.extra_info_pane, sizing_mode = "stretch_both", scroll = True, max_height = 200),
                         pn.Row(self.add_column_button,
                                self.remove_column_button,
                                pn.Spacer(width = 30),
                                self.column_selector,
                                ),
                         self.sourceid_input,
                         pn.Row(self.prev_button,self.next_button, self.search_button), 
                         pn.Row(self.label_selector, pn.Column(self.colorpickers_layout, self.strings_to_label_layout), self.confirm_label_button),
                         sizing_mode = "stretch_both")
    

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



