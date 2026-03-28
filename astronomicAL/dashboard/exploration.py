import panel as pn
import param
import os
import uuid
import pandas as pd
import numpy as np

from typing import Any, Dict, List

from functools import partial

from astronomicAL.extensions import feature_generation
from astronomicAL.utils.optimise import matches_type, get_series_type

class ExplorationDashboard(param.Parameterized):

    index = param.Integer(default=0, bounds=(0, 0))
    def __init__(self, src, df, context=None, **params):
        super().__init__(**params)

        self.src = src
        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        self.df = self.config.main_df
        self.panel_id = str(uuid.uuid4())

        self._running_panels = set()
        self._event_subs = []
        self._mapping_requests_sent = set()
        self._built = False

        self.visited_indices = [self.index]
        self.current_position = 0

        self._root = pn.Column(sizing_mode="stretch_both")

        self._ensure_dataset_registered()
        self._subscribe_to_mapping_events()
        self._try_build_dashboard()


    def _dataset_id(self) -> str:
        return "main"

    def _ensure_dataset_registered(self) -> None:
        if getattr(self, "context", None) is None or getattr(self.context, "datasets", None) is None:
            return

        self.context.datasets.ensure_registered(
            self._dataset_id(),
            self.df,
            name="Main Dataset",
        )
        self.context.datasets.set_active(self._dataset_id())

    def _exploration_mapping_specs(self) -> List[Dict[str, Any]]:
        columns = list(self.df.columns)

        return [
            {
                "semantic_name": "record_id",
                "config_key": "id_col",
                "display_name": "ID column",
                "description": "Needed by Exploration Mode for source lookup and navigation.",
                "required": True,
                "candidates": ["Use Index"] + columns,
                "suggested": self._guess_column(["source_id", "id", "objid", "object_id"]) or "Use Index",
            },
            {
                "semantic_name": "coords.ra",
                "config_key": "ra_col_name",
                "display_name": "RA column",
                "description": "Needed by Exploration Mode to construct the combined RA/DEC coordinate string.",
                "required": True,
                "candidates": columns,
                "suggested": self._guess_column(["ra", "raj2000", "ra_deg"]),
            },
            {
                "semantic_name": "coords.dec",
                "config_key": "dec_col_name",
                "display_name": "DEC column",
                "description": "Needed by Exploration Mode to construct the combined RA/DEC coordinate string.",
                "required": True,
                "candidates": columns,
                "suggested": self._guess_column(["dec", "dej2000", "dec_deg"]),
            },
            {
                "semantic_name": "target_label",
                "config_key": "label_col",
                "display_name": "Label column",
                "description": "Optional in Exploration Mode. Used for label selection and colour assignment.",
                "required": False,
                "candidates": ["No Labels"] + columns,
                "suggested": self._guess_column(["label", "class", "target", "y"]) or "No Labels",
            },
        ]

    def _flex_row(self, *objects, gap="8px", margin=(0, 0, 8, 0)):
        return pn.FlexBox(
            *objects,
            flex_direction="row",
            flex_wrap="wrap",
            gap=gap,
            align_items="center",
            sizing_mode="stretch_width",
            margin=margin,
        )

    def _guess_column(self, names: List[str]):
        lowered = {col.lower(): col for col in self.df.columns}
        for name in names:
            if name.lower() in lowered:
                return lowered[name.lower()]
        return None

    def _sync_config_from_dataset_mappings(self) -> None:
        dataset_id = self._dataset_id()

        for spec in self._exploration_mapping_specs():
            semantic_name = spec["semantic_name"]
            config_key = spec["config_key"]

            mapped = self.context.datasets.get_mapping(dataset_id, semantic_name)
            existing = self.config.settings.get(config_key)

            # Compatibility: if an old config value already exists, seed the DatasetManager from it.
            if mapped is None and existing in spec["candidates"]:
                self.context.datasets.set_mapping(dataset_id, semantic_name, existing)
                mapped = existing

            if mapped is not None:
                self.config.settings[config_key] = mapped

        # Exploration should still run even if the optional label is not mapped yet.
        self.config.settings.setdefault("label_col", "No Labels")

    def _publish_mapping_request(self, spec: Dict[str, Any]) -> None:
        key = (self._dataset_id(), spec["semantic_name"])
        if key in self._mapping_requests_sent:
            return

        payload = {
            "source": "exploration",
            "panel_id": self.panel_id,
            "dataset_id": self._dataset_id(),
            "semantic_name": spec["semantic_name"],
            "display_name": spec["display_name"],
            "description": spec["description"],
            "required": spec["required"],
            "config_key": spec["config_key"],
            "candidates": spec["candidates"],
            "suggested": spec["suggested"],
        }

        self.context.events.publish("mapping.requested", payload)
        self._mapping_requests_sent.add(key)

    def _request_missing_mappings(self) -> bool:
        """
        Returns True if any REQUIRED mapping is still missing.
        """
        self._sync_config_from_dataset_mappings()

        missing_required = False
        dataset_id = self._dataset_id()

        for spec in self._exploration_mapping_specs():
            mapped = self.context.datasets.get_mapping(dataset_id, spec["semantic_name"])

            if mapped is None:
                self._publish_mapping_request(spec)
                if spec["required"]:
                    missing_required = True

        return missing_required

    def _subscribe_to_mapping_events(self) -> None:
        if not getattr(self, "context", None) or not getattr(self.context, "events", None):
            return

        bus = self.context.events

        def _sub(topic, fn):
            sub = bus.subscribe(topic, fn)
            self._event_subs.append(sub)

        def _mapping_updated(_topic, payload):
            if not payload:
                return

            if payload.get("dataset_id") != self._dataset_id():
                return

            semantic_name = payload.get("semantic_name")
            column_name = payload.get("column_name")
            config_key = payload.get("config_key")

            if semantic_name is None or column_name is None:
                return

            if config_key:
                self.config.settings[config_key] = column_name

            # Optional live-update for label mapping after build.
            if semantic_name == "target_label" and self._built:
                self.config.settings["label_col"] = column_name
                if hasattr(self, "label_selector"):
                    options = ["No Labels"] + list(self.df.columns)
                    self.label_selector.options = options
                    if column_name in options:
                        self.label_selector.value = column_name

            self._try_build_dashboard()

        _sub("dataset.mapping_updated", _mapping_updated)

    def _try_build_dashboard(self) -> None:
        missing_required = self._request_missing_mappings()

        if missing_required:
            self._root[:] = [
                pn.Column(
                    self.get_toolbar(),
                    pn.pane.Alert(
                        "Exploration Mode needs dataset mappings before it can open. "
                        "Use the header alert to map the ID, RA and DEC columns.",
                        alert_type="warning",
                    ),
                    sizing_mode="stretch_width",
                )
            ]
            return

        # At this point the required mappings exist.
        self._sync_config_from_dataset_mappings()

        if not self._built:
            self._build_dashboard_ui()
            self._built = True

        self._root[:] = [
            pn.Column(
                self.get_toolbar(),
                self.get_layout(),
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )
        ]

    def _build_dashboard_ui(self) -> None:
        self.param.index.bounds = (0, len(self.df) - 1)

        self._preprocess_data()
        self._create_extra_info_cols_list()
        self._update_selected_src()

        self.prev_button = pn.widgets.Button(
            name="Previous",
            button_type="primary",
            width=84,
            height=34,
        )
        self.next_button = pn.widgets.Button(
            name="Next",
            button_type="primary",
            width=84,
            height=34,
        )
        self.search_button = pn.widgets.Button(
            name="Search",
            button_type="primary",
            width=84,
            height=34,
        )
        self.sourceid_input = pn.widgets.TextInput(
            name="SourceId",
            value="",
            width=180,
        )

        self.extra_info_pane = pn.pane.DataFrame(
            self._get_extra_info_df(),
            index=False,
            header=False,
            sizing_mode="stretch_both",
        )

        self.prev_button.on_click(self._go_previous)
        self.next_button.on_click(self._go_next)
        self.search_button.on_click(self._search_button_cb)
        self.sourceid_watcher = self.sourceid_input.param.watch(
            self._sourceid_input_cb,
            "value",
            onlychanged=False,
        )

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
        id_col = self.config.settings["id_col"]
        if id_col == "Use Index":
            return pd.Series(self.df.index, index=self.df.index)  # Ensures it's a Series
        else:
            return self.df[id_col]
        
    def _get_selected_id(self):
        """Returns the sourceID for the selected source"""
        id_col = self.config.settings["id_col"]
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
        """
        Replacement for shared_data subscriptions using EventBus.

        Expects events:
        - astro.spectra.running: {"source": "...", "running": bool, "panel_id": "..."}
        - astro.cutout.running: {"source": "Euclid", "running": bool, "panel_id": "..."}
        - astro.radio.running:  {"source": "VLASS"/"LoTSS"/..., "running": bool, "panel_id": "..."}
        - astro.sdss.running:   {"running": bool, "panel_id": "..."}   (or include source)
        - selection.sourceid.changed: {"sourceId": <id>, "origin": "..."}
        """
        if not getattr(self, "context", None) or not getattr(self.context, "events", None):
            return

        bus = self.context.events

        # Track subscriptions so you can unsubscribe if needed (or rely on dashboard dispose)
        self._event_subs = getattr(self, "_event_subs", [])

        def _sub(topic, fn):
            sub = bus.subscribe(topic, fn)
            self._event_subs.append(sub)

        # --- running status aggregator ---
        def _handle_running(payload, panel_name: str):
            if not payload:
                return
            running = payload.get("running")
            if running is None:
                return
            self._multithread_running_cb(bool(running), panel_name)

        # # spectra running: source will be DESI/SDSS/EuclidSpec
        # def _spectra_running(_topic, payload):
        #     src = (payload or {}).get("source", "Spectra")
        #     _handle_running(payload, panel_name=str(src))

        # _sub("astro.spectra.running", _spectra_running)

        # # euclid cutout running
        # def _cutout_running(_topic, payload):
        #     src = (payload or {}).get("source", "EuclidCutout")
        #     # normalize to your old names if you want:
        #     panel_name = "EuclidCutout" if str(src).lower().startswith("euclid") else str(src)
        #     _handle_running(payload, panel_name=panel_name)

        # _sub("astro.cutout.running", _cutout_running)

        # # radio running (VLASS/LoTSS)
        # def _radio_running(_topic, payload):
        #     src = (payload or {}).get("source", "Radio")
        #     _handle_running(payload, panel_name=str(src))

        # _sub("astro.radio.running", _radio_running)

        # # SDSS cutout running (if you publish it separately)
        # def _sdss_running(_topic, payload):
        #     _handle_running(payload, panel_name="SDSS")

        # _sub("astro.sdss.running", _sdss_running)

        # --- selected source id (from plots) ---
        def _selected_sourceid(_topic, payload):
            if not payload:
                return
            source_id = payload.get("sourceId", None)
            if source_id is None:
                return
            self._selected_src_from_plot_cb(source_id)

        _sub("selection.sourceid.changed", _selected_sourceid)

    def _update_navigation_flags(self):
        self.prev_button.disabled = self.current_position == 0
        self.next_button.disabled = False

    def _get_extra_info_df(self):
        id_col = self.config.settings["id_col"]
        if len(self.src.data[id_col]) > 0:
            source_id = str(self.src.data[id_col][0])
            extra_data_list = [["SourceId", source_id]]
            for col in self.config.settings["extra_info_cols"]:
                try:
                    value = self.src.data[f"{col}"][0]
                    if isinstance(value, float) and value < 1e4:
                        value = float(f"{value:.6g}")
                    extra_data_list.append([col, value])
                except KeyError:
                    continue
            return  pd.DataFrame(extra_data_list, columns=["Column", "Value"])
        else:
            cols = ["SourceId"] + self.config.settings["extra_info_cols"]
            return  pd.DataFrame(cols, columns=["Column"])
        
    
    def _save_extra_info_df(self):
        return self._get_extra_info_df()
    
    def _initialise_add_remove_columns_widgets(self):
        self.add_column_button = pn.widgets.Button(
            name="Add",
            button_type="light",
            width=64,
            height=32,
        )
        self.remove_column_button = pn.widgets.Button(
            name="Remove",
            button_type="light",
            width=82,
            height=32,
        )
        self.column_selector = pn.widgets.Select(
            options=[],
            width=130,
            visible=False,
        )
        self.selector_watcher = None
        self.add_column_button.on_click(self._add_column_callback)
        self.remove_column_button.on_click(self._remove_column_callback)       
    
    def _add_column_callback(self, event):
        self.column_selector.value = None
        options = [""] + [i for i in self.df.columns if i not in self.config.settings["extra_info_cols"]]
        self.column_selector.visible = True
        self.column_selector.options = options
        if self.selector_watcher is not None:
            self.column_selector.param.unwatch(self.selector_watcher)
        self.selector_watcher = self.column_selector.param.watch(self._add_extra_feature, "value")
    
    def _remove_column_callback(self, event):
        self.column_selector.value = None
        options = [""] + list(self.config.settings["extra_info_cols"])
        self.column_selector.visible = True
        self.column_selector.options = options
        if self.selector_watcher is not None:
            self.column_selector.param.unwatch(self.selector_watcher)
        self.selector_watcher = self.column_selector.param.watch(self._remove_extra_feature, "value")

    def _add_extra_feature(self, event):
        column = event.new
        if column and column not in self.config.settings["extra_info_cols"]:
            self.config.settings["extra_info_cols"].append(column)
            self.extra_info_pane.object = self._get_extra_info_df()
            self.column_selector.visible = False

    def _remove_extra_feature(self, event):
        column = event.new
        if column and column in self.config.settings["extra_info_cols"]:
            self.config.settings["extra_info_cols"].remove(column)
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
        if self.config.settings["id_col"] not in selected_dict:
                selected_dict[self.config.settings["id_col"]] = [self.index]
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

        bands = self.config.settings["features_for_training"]

        features = bands + [self.config.settings["label_col"], self.config.settings["id_col"]]

        oper_dict = feature_generation.get_oper_dict()

        if "feature_generation" in list(self.config.settings.keys()):
            for generator in self.config.settings["feature_generation"]:
                oper = generator[0]
                n = generator[1]
                df, generated_features = oper_dict[oper](df, n, context = self.context)
                features = features + generated_features
        return df
    
    def _generate_fake_label_column(self, df):
        """This is not very elegant but allows to keep the code as it is.
           If No labels is selected, it creates a column No labels with all Nan"""
        if (self.config.settings["label_col"] not in df.columns) and (self.config.settings["label_col"] == "No Labels"):
            df[self.config.settings["label_col"]] = np.nan
        return df
    

    def _add_ra_dec_col(self, df):
        new_df = df
        ra_col_name = self.config.settings["ra_col_name"]
        dec_col_name = self.config.settings["dec_col_name"]
        new_df["ra_dec"] = df[ra_col_name].astype(str) + "," + df[dec_col_name].astype(str)
      
        return new_df

    def _preprocess_data(self):
        """Process all the data according to the config file. In exploring panel it just"
           compute combination of features and creating ra and dec column"""
        #self.df = self._generate_fake_label_column(self.df)
        #self.df = self._generate_features(self.df)
        self.df = self._add_ra_dec_col(self.df)

    def _create_extra_info_cols_list(self):  
        if "extra_info_cols" not in  self.config.settings:
            self.config.settings["extra_info_cols"] = []

    
    def _initialise_label_selector(self):
        label_column_options = ["No Labels"] + list(self.df.columns)

        self.label_selector = pn.widgets.Select(
            options=label_column_options,
            value=self.config.settings["label_col"],
            width=110,
            visible=True,
        )
        self.label_selector.param.watch(self._update_labels_cb, "value")

        self.colorpickers_layout = pn.FlexBox(
            flex_direction="row",
            flex_wrap="wrap",
            gap="6px",
            align_items="center",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.strings_to_label_layout = pn.FlexBox(
            flex_direction="row",
            flex_wrap="wrap",
            gap="6px",
            align_items="center",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.confirm_label_button = pn.widgets.Button(
            name="Confirm",
            button_type="success",
            width=82,
            height=34,
            visible=False,
        )
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
        self.config.settings["label_col"] = self.label_selector.value
        self.config.settings["labels"] = self.labels
        self.config.settings["labels_to_strings"], self.config.settings["strings_to_labels"] = self.get_label_strings()
        self.config.settings["label_colours"] = self.get_label_colours()
        self.confirm_label_button.visible = False
            

    def _update_label_strings_input(self):
        self.strings_to_label_layout.clear()
        self.label_to_strings_param = {}

        widgets = []
        for data_label in self.labels:
            text_input = pn.widgets.TextInput(
                name=str(data_label),
                placeholder=str(data_label),
                width=80,
            )
            self.label_to_strings_param[f"{data_label}"] = text_input
            widgets.append(text_input)

        self.strings_to_label_layout[:] = widgets

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

        pickers = []
        for i, label in enumerate(self.labels):
            picker = pn.widgets.ColorPicker(
                name=str(label),
                value=colour_list[i % len(colour_list)],
                width=78,
            )
            self.colours_param[label] = picker
            pickers.append(picker)

        self.colorpickers_layout[:] = pickers

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
        index_widget = pn.Param(
            self,
            parameters=["index"],
            widgets={"index": {"type": pn.widgets.IntInput, "width": 80}},
            show_name=False,
            sizing_mode="fixed",
        )

        extra_info_section = pn.Column(
            self.extra_info_pane,
            height=100,
            sizing_mode="stretch_width",
            scroll=True,
            margin=(0, 0, 6, 0),
        )

        column_controls = self._flex_row(
            self.add_column_button,
            self.remove_column_button,
            self.column_selector,
            margin=(0, 0, 6, 0),
        )

        nav_controls = self._flex_row(
            self.sourceid_input,
            self.search_button,
            self.prev_button,
            self.next_button,
            margin=(0, 0, 6, 0),
        )

        label_controls = self._flex_row(
            self.label_selector,
            self.colorpickers_layout,
            self.confirm_label_button,
            margin=(0, 0, 4, 0),
        )

        label_string_controls = pn.FlexBox(
            self.strings_to_label_layout,
            flex_direction="row",
            flex_wrap="wrap",
            gap="6px",
            align_items="center",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        return pn.Column(
            index_widget,
            extra_info_section,
            column_controls,
            nav_controls,
            label_controls,
            label_string_controls,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )


    def get_toolbar(self):
        return pn.Spacer(height=1)

    def panel(self):
        return self._root

    def dispose(self):
        if getattr(self, "context", None) and getattr(self.context, "events", None):
            for sub in getattr(self, "_event_subs", []):
                try:
                    self.context.events.unsubscribe(sub)
                except Exception:
                    pass
        self._event_subs = []


