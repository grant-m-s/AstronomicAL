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
        self.main_layout = None
        self.labels_expanded = False

        self.visited_indices = [self.index]
        self.current_position = 0

        self._root = pn.Column(sizing_mode="stretch_both", scroll=True)

        self._ensure_dataset_registered()
        self._subscribe_to_mapping_events()
        self._subscribe_to_dataset_events()
        self._try_build_dashboard()

    def _dataset_id(self) -> str:
        if getattr(self, "context", None) is not None and getattr(self.context, "datasets", None) is not None:
            try:
                active = self.context.datasets.active_id()
                if active:
                    return active
            except Exception:
                pass
        return "main"

    def _ensure_dataset_registered(self) -> None:
        if getattr(self, "context", None) is None or getattr(self.context, "datasets", None) is None:
            return

        try:
            active = self.context.datasets.active_id()
        except Exception:
            active = None

        if active:
            try:
                self.df = self.context.datasets.get_df(active).copy()
                if getattr(self, "config", None) is not None:
                    self.config.main_df = self.df
                return
            except Exception:
                pass

        self.context.datasets.ensure_registered(
            "main",
            self.df,
            name="Main Dataset",
        )
        self.context.datasets.set_active("main")

    def _subscribe_to_dataset_events(self) -> None:
        if not getattr(self, "context", None) or not getattr(self.context, "events", None):
            return

        bus = self.context.events

        def _sub(topic, fn):
            sub = bus.subscribe(topic, fn)
            self._event_subs.append(sub)

        def _dataset_active_changed(_topic, payload):
            dataset_id = payload.get("dataset_id") if payload else None
            if dataset_id is not None and dataset_id != self._dataset_id():
                return
            self._refresh_from_active_dataset(reset_history=True)

        def _dataset_updated(_topic, payload):
            dataset_id = payload.get("dataset_id") if payload else None
            if dataset_id is not None and dataset_id != self._dataset_id():
                return
            self._refresh_from_active_dataset(reset_history=False)

        _sub("dataset.active.changed", _dataset_active_changed)
        _sub("dataset.updated", _dataset_updated)

    def _refresh_from_active_dataset(self, reset_history=True):
        if getattr(self, "context", None) is not None and getattr(self.context, "datasets", None) is not None:
            try:
                self.df = self.context.datasets.get_df(self._dataset_id()).copy()
                if getattr(self, "config", None) is not None:
                    self.config.main_df = self.df
            except Exception:
                self.df = self.config.main_df.copy()
        else:
            self.df = self.config.main_df.copy()

        missing_required = self._request_missing_mappings()
        if missing_required:
            self._root[:] = [
                pn.Column(
                    pn.pane.Alert(
                        "Exploration Mode needs dataset mappings before it can open. "
                        "Use the header alert to map the ID, RA and DEC columns.",
                        alert_type="warning",
                    ),
                    sizing_mode="stretch_width",
                    margin=(0, 0, 0, 0),
                )
            ]
            return

        self._sync_config_from_dataset_mappings()
        self._preprocess_data()
        self._create_extra_info_cols_list()

        self.config.settings["extra_info_cols"] = [
            c for c in self.config.settings["extra_info_cols"] if c in self.df.columns
        ]

        max_index = max(0, len(self.df) - 1)
        self.param.index.bounds = (0, max_index)
        self.index = min(self.index, max_index)

        if hasattr(self, "index_input"):
            self.index_input.start = 0
            self.index_input.end = max_index
            self.index_input.value = self.index

        if hasattr(self, "sourceid_input"):
            self.sourceid_input.value = ""

        if hasattr(self, "column_selector"):
            self.column_selector.value = None
            self.column_selector.visible = False

        if hasattr(self, "label_selector"):
            label_options = ["No Labels"] + list(self.df.columns)
            self.label_selector.options = label_options

            current_label_col = self.config.settings.get("label_col", "No Labels")
            if current_label_col not in label_options:
                current_label_col = "No Labels"
                self.config.settings["label_col"] = current_label_col

            self.label_selector.value = current_label_col
            self._sync_label_editor_state(current_label_col, keep_button_visible=True)

        if reset_history:
            self.visited_indices = [self.index]
            self.current_position = 0

        if not self._built:
            self._build_dashboard_ui()
            self._root[:] = [self.main_layout]
            self._built = True
            return

        self._update_selected_src()

        if hasattr(self, "extra_info_html"):
            self._refresh_extra_info_view()

        self._update_navigation_flags()
        self._rerender_main_layout()

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
                "suggested": self._guess_column(["ra", "raj2000", "ra_deg", "right_ascension"]),
            },
            {
                "semantic_name": "coords.dec",
                "config_key": "dec_col_name",
                "display_name": "DEC column",
                "description": "Needed by Exploration Mode to construct the combined RA/DEC coordinate string.",
                "required": True,
                "candidates": columns,
                "suggested": self._guess_column(["dec", "dej2000", "dec_deg", "declination"]),
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

    def _escape_html(self, value):
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _get_extra_info_html(self):
        df = self._get_extra_info_df()

        if df.empty:
            return "<div style='color:#666;'>No record information available.</div>"

        rows = []
        for _, row in df.iterrows():
            key = self._escape_html(row.iloc[0])
            value = self._escape_html(row.iloc[1] if len(row) > 1 else "")
            rows.append(
                f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600; white-space:nowrap; vertical-align:top; border-bottom:1px solid #eee;">{key}</td>
                    <td style="padding:6px 10px; vertical-align:top; border-bottom:1px solid #eee; word-break:break-word;">{value}</td>
                </tr>
                """
            )

        return f"""
        <div style="border:1px solid #ddd; border-radius:6px; overflow:hidden; background:white;">
        <table style="width:100%; border-collapse:collapse; font-size:13px;">
            <tbody>
            {''.join(rows)}
            </tbody>
        </table>
        </div>
        """

    def _refresh_extra_info_view(self):
        self.extra_info_html.object = self._get_extra_info_html()

    def _flex_row(self, *objects, gap="8px", margin=(0, 0, 8, 0), justify_content="flex-start"):
        return pn.FlexBox(
            *objects,
            flex_direction="row",
            flex_wrap="wrap",
            gap=gap,
            align_items="center",
            justify_content=justify_content,
            sizing_mode="stretch_width",
            margin=margin,
        )

    def _vspace(self, height=8):
        return pn.Spacer(height=height)

    def _field_block(self, label, widget, help_text=None, width=None):
        items = [pn.pane.Markdown(f"**{label}**", margin=(0, 0, 4, 0)), widget]
        if help_text:
            items.append(
                pn.pane.Markdown(
                    f"<div style='color:#666; font-size:0.9em;'>{help_text}</div>",
                    margin=(4, 0, 0, 0),
                )
            )
        return pn.Column(*items, width=width, margin=(0, 0, 0, 0))

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

            if mapped is None and existing in spec["candidates"]:
                self.context.datasets.set_mapping(dataset_id, semantic_name, existing)
                mapped = existing

            if mapped is not None:
                self.config.settings[config_key] = mapped

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

            if semantic_name == "target_label":
                self.config.settings["label_col"] = column_name

            self._refresh_from_active_dataset(reset_history=False)

        _sub("dataset.mapping_updated", _mapping_updated)

    def _publish_label_settings(self):
        if not getattr(self, "context", None) or not getattr(self.context, "events", None):
            return

        payload = {
            "dataset_id": self._dataset_id(),
            "label_col": self.config.settings.get("label_col", "No Labels"),
            "labels": list(getattr(self, "labels", [])),
            "labels_to_strings": dict(self.config.settings.get("labels_to_strings", {})),
            "strings_to_labels": dict(self.config.settings.get("strings_to_labels", {})),
            "label_colours": dict(self.config.settings.get("label_colours", {})),
            "source": "ExplorationDashboard",
            "panel_id": self.panel_id,
        }

        self.context.events.publish("labels.settings.updated", payload)

    def _try_build_dashboard(self) -> None:
        missing_required = self._request_missing_mappings()

        if missing_required:
            self._root[:] = [
                pn.Column(
                    pn.pane.Alert(
                        "Exploration Mode needs dataset mappings before it can open. "
                        "Use the header alert to map the ID, RA and DEC columns.",
                        alert_type="warning",
                    ),
                    sizing_mode="stretch_width",
                    margin=(0, 0, 0, 0),
                )
            ]
            return

        self._sync_config_from_dataset_mappings()

        if not self._built:
            self._build_dashboard_ui()
            self._root[:] = [self.main_layout]
            self._built = True

    def _rerender_main_layout(self):
        self.main_layout = self._build_main_layout()
        self._root[:] = [self.main_layout]

    def _toggle_labels_section(self, event=None):
        self.labels_expanded = not self.labels_expanded
        if self.main_layout is not None:
            self._rerender_main_layout()

    def _build_labels_card(self, body_fn, field_label_fn, card_fn):
        self.labels_toggle_button = pn.widgets.Button(
            name="Labels ▾" if self.labels_expanded else "Labels ▸",
            button_type="light",
            sizing_mode="stretch_width",
            height=30,
            margin=(0, 0, 0, 0),
        )
        self.labels_toggle_button.on_click(self._toggle_labels_section)

        collapsed_styles = {
            "border": "1px solid #d9d9d9",
            "border-radius": "8px",
            "background": "#ffffff",
            "padding": "4px 8px",
            "box-sizing": "border-box",
            "width": "100%",
        }

        expanded_styles = {
            "border": "1px solid #d9d9d9",
            "border-radius": "8px",
            "background": "#ffffff",
            "padding": "10px 12px",
            "box-sizing": "border-box",
            "width": "100%",
        }

        if not self.labels_expanded:
            return pn.Column(
                self.labels_toggle_button,
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 8, 0),
                styles=collapsed_styles,
            )

        label_items = [
            body_fn("Choose a label column, then optionally rename labels and set their colours."),
            pn.Spacer(height=6),
            pn.Column(
                field_label_fn("Label column"),
                self.label_selector,
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 0, 0),
            ),
        ]

        if len(self.label_editor_layout.objects) > 0:
            label_items.extend([
                pn.Spacer(height=6),
                pn.Column(
                    field_label_fn("Label display settings"),
                    self.label_editor_layout,
                    sizing_mode="stretch_width",
                    min_width=0,
                    margin=(0, 0, 0, 0),
                ),
            ])

        if self.confirm_label_button.visible:
            label_items.extend([
                pn.Spacer(height=6),
                pn.Row(
                    self.confirm_label_button,
                    sizing_mode="stretch_width",
                    min_width=0,
                    margin=(0, 0, 0, 0),
                ),
            ])

        labels_body = pn.Column(
            *label_items,
            sizing_mode="stretch_width",
            min_width=0,
            height=220,
            scroll=True,
            styles={
                "overflow-x": "hidden",
                "padding-right": "2px",
                "box-sizing": "border-box",
            },
            margin=(0, 0, 0, 0),
        )

        return pn.Column(
            self.labels_toggle_button,
            pn.Spacer(height=6),
            labels_body,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(0, 0, 8, 0),
            styles=expanded_styles,
        )

    def _extra_info_height(self):
        n_rows = len(self._get_extra_info_df())
        header = 30
        row_h = 28
        min_h = 56
        max_h = 180
        return max(min_h, min(max_h, header + n_rows * row_h))

    def _index_input_cb(self, event):
        if event.new != self.index:
            self.index = event.new

    def _sync_index_widget_cb(self, event):
        if hasattr(self, "index_input") and self.index_input.value != event.new:
            self.index_input.value = event.new

    def _build_dashboard_ui(self) -> None:
        self.param.index.bounds = (0, len(self.df) - 1)

        self._preprocess_data()
        self._create_extra_info_cols_list()
        self._update_selected_src()

        self.index_input = pn.widgets.IntInput(
            name="",
            value=self.index,
            start=0,
            end=len(self.df) - 1,
            width=88,
            height=28,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        )

        self.sourceid_input = pn.widgets.TextInput(
            name="",
            placeholder="Enter full or partial ID",
            value="",
            sizing_mode="stretch_width",
            max_width=320,
            margin=(0, 0, 0, 0),
        )

        self.index_input.param.watch(self._index_input_cb, "value")
        self.param.watch(self._sync_index_widget_cb, "index")

        self.prev_button = pn.widgets.Button(
            name="Previous",
            button_type="default",
            width=80,
            height=30,
            margin=(0, 0, 0, 0),
        )

        self.next_button = pn.widgets.Button(
            name="Next",
            button_type="primary",
            width=80,
            height=30,
            margin=(0, 0, 0, 0),
        )

        self.search_button = pn.widgets.Button(
            name="Find",
            button_type="primary",
            width=80,
            height=30,
            margin=(0, 0, 0, 0),
        )

        self.extra_info_html = pn.pane.HTML(
            self._get_extra_info_html(),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.prev_button.on_click(self._go_previous)
        self.next_button.on_click(self._go_next)
        self.search_button.on_click(self._search_button_cb)

        self._initialise_add_remove_columns_widgets()
        self._initialise_label_selector()
        self._update_navigation_flags()
        self._subscribe_to_shared()

        self.main_layout = self._build_main_layout()

    @param.depends('index', watch=True)
    def _update_history(self):
        if self.visited_indices[self.current_position] == self.index:
            return
        if self.index != self.visited_indices[-1]:
            self.visited_indices.append(self.index)

        self.current_position = len(self.visited_indices) - 1
        self._update_navigation_flags()
        self.sourceid_input.value = ""

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
        id_col = self.config.settings["id_col"]
        if id_col == "Use Index":
            return pd.Series(self.df.index, index=self.df.index)
        else:
            return self.df[id_col]

    def _get_selected_id(self):
        id_col = self.config.settings["id_col"]
        if len(self.src.data[id_col]) > 0:
            return str(self.src.data[id_col][0])
        
        
    

    
    def _find_from_id(self, sourceid):
        sourceid = sourceid.strip()
        try:
            matches = self._get_id().str.contains(sourceid, case=True)
        except AttributeError:
            matches = self._get_id().astype(str).str.contains(sourceid, case=True)

        N_matches = matches.sum()
        if N_matches == 1:
            self.index = self.df[matches].index[0]
        elif N_matches == 0:
            print("No matches found")
        else:
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
        self.prev_button.disabled = any_running or self.current_position == 0
        self.next_button.disabled = any_running

    def _subscribe_to_shared(self):
        if not getattr(self, "context", None) or not getattr(self.context, "events", None):
            return

        bus = self.context.events
        self._event_subs = getattr(self, "_event_subs", [])

        def _sub(topic, fn):
            sub = bus.subscribe(topic, fn)
            self._event_subs.append(sub)

        def _selected_sourceid(_topic, payload):
            if not payload:
                return
            source_id = payload.get("sourceId", None)
            if source_id is None:
                return
            self._selected_src_from_plot_cb(source_id)

        _sub("selection.sourceid.changed", _selected_sourceid)

    def _update_navigation_flags(self):
        if not hasattr(self, "prev_button") or not hasattr(self, "next_button"):
            return

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
            return pd.DataFrame(extra_data_list, columns=["Column", "Value"])
        else:
            cols = ["SourceId"] + self.config.settings["extra_info_cols"]
            return pd.DataFrame(cols, columns=["Column"])

    def _save_extra_info_df(self):
        return self._get_extra_info_df()

    def _initialise_add_remove_columns_widgets(self):

        self.add_column_button = pn.widgets.Button(
            name="Add field",
            button_type="light",
            width=92,
            height=30,
            margin=(0, 0, 0, 0),
        )

        self.remove_column_button = pn.widgets.Button(
            name="Remove field",
            button_type="light",
            width=112,
            height=30,
            margin=(0, 0, 0, 0),
        )

        self.column_selector = pn.widgets.Select(
            name="",
            options=[],
            sizing_mode="stretch_width",
            max_width=320,
            visible=False,
            margin=(0, 0, 0, 0),
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
        self._rerender_main_layout()

    def _remove_column_callback(self, event):
        self.column_selector.value = None
        options = [""] + list(self.config.settings["extra_info_cols"])
        self.column_selector.visible = True
        self.column_selector.options = options
        if self.selector_watcher is not None:
            self.column_selector.param.unwatch(self.selector_watcher)
        self.selector_watcher = self.column_selector.param.watch(self._remove_extra_feature, "value")
        self._rerender_main_layout()

    def _add_extra_feature(self, event):
        column = event.new
        if column and column not in self.config.settings["extra_info_cols"]:
            self.config.settings["extra_info_cols"].append(column)
            self._refresh_extra_info_view()
            self.column_selector.visible = False
            self._rerender_main_layout()

    def _remove_extra_feature(self, event):
        column = event.new
        if column and column in self.config.settings["extra_info_cols"]:
            self.config.settings["extra_info_cols"].remove(column)
            self._refresh_extra_info_view()
            self.column_selector.visible = False
            self._rerender_main_layout()

    def _selected_src_from_plot_cb(self, sourceid):
        self.sourceid_input.value = str(sourceid)

    @param.depends("index", watch=True)
    def _update_src_cb(self):
        self._update_selected_src()
        self._refresh_extra_info_view()

    def _update_selected_src(self):
        selected_dict = self.df.iloc[[self.index]].to_dict("list")
        if self.config.settings["id_col"] not in selected_dict:
            selected_dict[self.config.settings["id_col"]] = [self.index]
        self.src.data = selected_dict

    def _generate_features(self, df):
        bands = self.config.settings["features_for_training"]
        features = bands + [self.config.settings["label_col"], self.config.settings["id_col"]]
        oper_dict = feature_generation.get_oper_dict()

        if "feature_generation" in list(self.config.settings.keys()):
            for generator in self.config.settings["feature_generation"]:
                oper = generator[0]
                n = generator[1]
                df, generated_features = oper_dict[oper](df, n, context=self.context)
                features = features + generated_features
        return df

    def _generate_fake_label_column(self, df):
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
        self.df = self._add_ra_dec_col(self.df)

    def _create_extra_info_cols_list(self):
        if "extra_info_cols" not in self.config.settings:
            self.config.settings["extra_info_cols"] = []

    def _initialise_label_selector(self):
        label_column_options = ["No Labels"] + list(self.df.columns)

        self.label_selector = pn.widgets.Select(
            name="",
            options=label_column_options,
            value=self.config.settings["label_col"],
            sizing_mode="stretch_width",
            max_width=320,
            visible=True,
            margin=(0, 0, 0, 0),
        )

        self.label_editor_layout = pn.Column(
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.confirm_label_button = pn.widgets.Button(
            name="Apply label settings",
            button_type="success",
            width=130,
            height=30,
            visible=False,
            margin=(0, 0, 0, 0),
        )
        self.confirm_label_button.on_click(self._confirm_labels_change_cb)

        self.label_selector.param.watch(self._update_labels_cb, "value")

        self.labels = []
        self._sync_label_editor_state(self.label_selector.value, keep_button_visible=True)

    def _update_labels_cb(self, event):
        self._sync_label_editor_state(event.new, keep_button_visible=True)
        self._rerender_main_layout()

    def _confirm_labels_change_cb(self, event):
        self.config.settings["label_col"] = self.label_selector.value
        self.config.settings["labels"] = self.labels
        (
            self.config.settings["labels_to_strings"],
            self.config.settings["strings_to_labels"],
        ) = self.get_label_strings()
        self.config.settings["label_colours"] = self.get_label_colours()

        self.confirm_label_button.visible = bool(self.labels)

        self._publish_label_settings()
        self._rerender_main_layout()

    def _label_editor_height(self):
        n = len(getattr(self, "labels", []))
        if n == 0:
            return 0

        header_h = 16
        row_h = 30
        gap_h = 3
        padding_h = 8

        total = header_h + padding_h + n * row_h + max(0, n - 1) * gap_h
        return min(118, total)

    def _sync_label_editor_state(self, selected_label_column, keep_button_visible=True):
        if (selected_label_column not in self.df.columns) or (selected_label_column == "No Labels"):
            self.labels = []
            self.label_editor_layout[:] = []
            self.confirm_label_button.visible = False
            return

        self.label_type = get_series_type(self.df[selected_label_column])

        if self.label_type == "mixed":
            self.df[selected_label_column] = self.df[selected_label_column].astype(str)
            self.label_type = "string"
        elif self.label_type == "bool":
            self.df[selected_label_column] = self.df[selected_label_column].astype(int)
            self.label_type = "int"

        if len(self.df[selected_label_column].unique()) > 20:
            print(
                "You have chosen a column with too many unique values (possibly continuous); "
                "please choose a column with a smaller set of labels (<=20)"
            )
            self.labels = []
            self.label_editor_layout[:] = []
            self.confirm_label_button.visible = False
            return

        self.labels = sorted(self.df[selected_label_column].unique())
        self._build_label_editor_rows()
        self.confirm_label_button.visible = keep_button_visible


    def _build_label_editor_rows(self):
        self.label_to_strings_param = {}
        self.colours_param = {}

        colour_list = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        raw_w = 64
        colour_w = 88

        def header_cell(text, align="left"):
            return pn.pane.HTML(
                f"""
                <div style="
                    font-size:10px;
                    font-weight:600;
                    color:#666;
                    white-space:nowrap;
                    overflow:hidden;
                    text-overflow:ellipsis;
                    text-align:{align};
                    line-height:12px;
                    padding:0;
                    margin:0;
                ">
                    {text}
                </div>
                """,
                margin=(0, 0, 0, 0),
                sizing_mode="stretch_width",
                height=12,
            )

        rows = [
            pn.Row(
                pn.Column(
                    header_cell("Raw label"),
                    width=raw_w,
                    min_width=raw_w,
                    max_width=raw_w,
                    margin=(0, 0, 0, 0),
                ),
                pn.Column(
                    header_cell("Display name"),
                    sizing_mode="stretch_width",
                    min_width=0,
                    margin=(0, 0, 0, 0),
                ),
                pn.Column(
                    header_cell("Colour", align="center"),
                    width=colour_w,
                    min_width=colour_w,
                    max_width=colour_w,
                    margin=(0, 0, 0, 0),
                ),
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 6, 0),
            )
        ]

        for i, label in enumerate(self.labels):
            text_input = pn.widgets.TextInput(
                name="",
                value=str(label),
                placeholder="Display name",
                sizing_mode="stretch_width",
                min_width=0,
                height=26,
                margin=(0, 0, 0, 0),
            )

            picker = pn.widgets.ColorPicker(
                name="",
                value=colour_list[i % len(colour_list)],
                width=58,
                min_width=58,
                max_width=58,
                height=26,
                margin=(0, 0, 0, 0),
            )

            self.label_to_strings_param[f"{label}"] = text_input
            self.colours_param[label] = picker

            raw_chip = pn.pane.HTML(
                f"""
                <div style="
                    display:inline-block;
                    max-width:52px;
                    padding:4px 8px;
                    border:1px solid #d8d8d8;
                    border-radius:999px;
                    background:#f7f7f7;
                    color:#333;
                    font-size:12px;
                    font-weight:600;
                    white-space:nowrap;
                    overflow:hidden;
                    text-overflow:ellipsis;
                    box-sizing:border-box;
                    line-height:16px;
                ">
                    {self._escape_html(label)}
                </div>
                """,
                width=raw_w,
                height=24,
                margin=(0, 0, 0, 0),
            )

            colour_cell = pn.Row(
                pn.Spacer(sizing_mode="stretch_width"),
                picker,
                pn.Spacer(sizing_mode="stretch_width"),
                width=colour_w,
                min_width=colour_w,
                max_width=colour_w,
                margin=(0, 0, 0, 0),
                align="center",
            )

            row = pn.Row(
                pn.Column(
                    raw_chip,
                    width=raw_w,
                    min_width=raw_w,
                    max_width=raw_w,
                    margin=(0, 0, 0, 0),
                ),
                pn.Column(
                    text_input,
                    sizing_mode="stretch_width",
                    min_width=0,
                    margin=(0, 0, 0, 0),
                ),
                pn.Column(
                    colour_cell,
                    width=colour_w,
                    min_width=colour_w,
                    max_width=colour_w,
                    margin=(0, 0, 0, 0),
                ),
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 4, 0),
                styles={
                    "width": "100%",
                    "max-width": "100%",
                    "border": "1px solid #e8e8e8",
                    "border-radius": "7px",
                    "padding": "6px 6px",
                    "background": "#ffffff",
                    "box-sizing": "border-box",
                    "align-items": "center",
                    "overflow": "hidden",
                },
            )

            rows.append(row)

        self.label_editor_layout[:] = rows

    def _build_main_layout(self):
        def heading(text, size=15, weight=700, color="#2f2f2f", margin_bottom=0):
            return pn.pane.HTML(
                (
                    f"<div style='font-size:{size}px; font-weight:{weight}; "
                    f"color:{color}; margin:0 0 {margin_bottom}px 0;'>{text}</div>"
                ),
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )

        def body(text):
            return pn.pane.HTML(
                f"<div style='font-size:12px; line-height:1.35; color:#4a4a4a; margin:0;'>{text}</div>",
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )

        def field_label(text):
            return pn.pane.HTML(
                f"<div style='font-size:11px; font-weight:600; color:#2f2f2f; margin:0 0 4px 0;'>{text}</div>",
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )

        def divider():
            return pn.pane.HTML(
                "<div style='height:1px; background:#e8e8e8; margin:0;'></div>",
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )

        card_styles = {
            "border": "1px solid #d9d9d9",
            "border-radius": "8px",
            "background": "#ffffff",
            "padding": "10px 12px",
            "box-sizing": "border-box",
            "width": "100%",
            "overflow": "hidden",
        }

        def subsection(title, description, *content):
            items = [
                heading(title, size=14, weight=700, margin_bottom=4),
                body(description),
            ]
            for obj in content:
                items.extend([pn.Spacer(height=5), obj])

            return pn.Column(
                *items,
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 0, 0),
            )

        def card(*objects):
            return pn.Column(
                *objects,
                sizing_mode="stretch_width",
                min_width=0,
                styles=card_styles,
                margin=(0, 0, 8, 0),
            )

        browse_controls = pn.Row(
            self.search_button,
            self.prev_button,
            self.next_button,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(0, 0, 0, 0),
        )

        metadata_controls = pn.Row(
            self.add_column_button,
            self.remove_column_button,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(0, 0, 0, 0),
        )

        index_block = pn.Column(
            field_label("Record index"),
            pn.Row(
                self.index_input,
                sizing_mode="fixed",
                width=88,
                height=28,
                margin=(0, 0, 0, 0),
            ),
            sizing_mode="fixed",
            width=110,
            min_height=46,
            max_height=46,
            margin=(0, 0, 0, 0),
        )

        current_record_block = subsection(
            "Current record",
            "Review the selected record.",
            index_block,
            pn.Column(
                field_label("Visible record information"),
                self.extra_info_html,
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 0, 0),
            ),
        )

        browse_block = subsection(
            "Browse records",
            "Search for a record or move backward and forward through your navigation history.",
            pn.Column(
                field_label("Find record"),
                self.sourceid_input,
                sizing_mode="stretch_width",
                min_width=0,
                margin=(0, 0, 0, 0),
            ),
            browse_controls,
        )

        record_card = card(
            current_record_block,
            pn.Spacer(height=5),
            divider(),
            pn.Spacer(height=5),
            browse_block,
        )

        metadata_items = [metadata_controls]
        if self.column_selector.visible:
            metadata_items.extend([
                pn.Spacer(height=5),
                pn.Column(
                    field_label("Field"),
                    self.column_selector,
                    sizing_mode="stretch_width",
                    min_width=0,
                    margin=(0, 0, 0, 0),
                )
            ])

        metadata_block = subsection(
            "Visible metadata",
            "Choose which extra fields appear in the summary above.",
            *metadata_items,
        )

        settings_card = card(metadata_block)

        labels_card = self._build_labels_card(body, field_label, card)

        return pn.Column(
            record_card,
            settings_card,
            labels_card,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(0, 0, 0, 0),
        )

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
        return {key: param_obj.value for key, param_obj in self.colours_param.items()}

    def get_layout(self):
        return self.main_layout

    def get_toolbar(self):
        return None

    def panel(self):
        self._root.sizing_mode = "stretch_both"
        self._root.scroll = True
        self._root.margin = (0, 0, 0, 0)
        self._root.min_width = 0
        self._root.styles = {
            "padding": "4px",
            "box-sizing": "border-box",
        }
        return self._root

    def dispose(self):
        if getattr(self, "context", None) and getattr(self.context, "events", None):
            for sub in getattr(self, "_event_subs", []):
                try:
                    self.context.events.unsubscribe(sub)
                except Exception:
                    pass
        self._event_subs = []