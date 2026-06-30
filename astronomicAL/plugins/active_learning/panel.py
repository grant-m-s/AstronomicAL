from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import panel as pn

from astronomicAL.platform.plugins.specs import ActionRequest

from . import actions
from . import analytics as al_analytics
from . import contracts as al_contracts
from . import state as al_state

_AL_LAYOUT_CSS_INSTALLED = False

AL_OWNED_RECIPE_PARAM_NAMES = {
    "dataset_id",
    "recipe_id",
    "train_dataset_id",
    "session_artifact_id",
    "al_session_id",
    "al_session_artifact_id",
    "al_training_artifact_id",
    "al_round",
    "target",
    "label",
    "target_column",
    "label_column",
    "record_id_column",
    "id_column",
    "image_column",
    "image_path_column",
    "image_uri_column",
    "mask_column",
    "mask_path_column",
    "feature_columns",
    "input_columns",
    "features",
    "x_columns",
    "label_options",
    "class_labels",
    "classes",
    "known_classes",
    "target_classes",
    "validation_dataset_id",
    "test_dataset_id",
    "protocol_validation_source",
    "protocol_validation_dataset_id",
    "protocol_validation_size",
    "protocol_test_source",
    "protocol_test_dataset_id",
    "protocol_test_size",
    "protocol_random_state",
    "al_protocol",
}

class ActiveLearningPanel:
    state_version = 3

    def __init__(
        self,
        context: Any,
        restore_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.context = context
        self._install_layout_css_once()

        self._current_session_artifact_id: Optional[str] = None
        self._active_job_handle: Any = None
        self._job_running = False

        self._disposed = False
        self._subscriptions: List[Any] = []
        self.recipe_param_widgets: Dict[str, Any] = {}
        self.recipe_param_fields: Dict[str, Any] = {}
        self.protocol_widgets: Dict[str, Any] = {}
        self.protocol_fields: Dict[str, Any] = {}
        self._latest_session_contract: Dict[str, Any] = {}
        self._latest_preflight_errors: List[str] = []
        self._latest_preflight_warnings: List[str] = []

        restore_state = dict(restore_state or {})
        self._current_session_artifact_id = restore_state.get(
            "session_artifact_id"
        )

        self.dataset_select = pn.widgets.Select(
            name="Pool dataset",
            options=[],
        )
        self.session_select = pn.widgets.Select(
            name="Session",
            options={},
        )
        self.predictions_select = pn.widgets.Select(
            name="Predictions artifact",
            options={},
        )
        self.strategy_select = pn.widgets.Select(
            name="Query strategy",
            options={},
        )

        self.recipe_select = pn.widgets.Select(
            name="core.ml recipe",
            options={},
        )
        self.recipe_card = pn.pane.Markdown(
            "No recipe selected.",
            sizing_mode="stretch_width",
        )
        self.recipe_params_area = pn.Column(
            sizing_mode="stretch_width"
        )

        self.label_options_select = pn.widgets.MultiChoice(
            name="Allowed labels",
            options=[],
            value=[],
            solid=False,
        )
        self.target_column = pn.widgets.Select(
            name="Training label column",
            options=[],
            value=None,
        )

        self.image_column = pn.widgets.Select(
            name="Image input column",
            options={
                "Auto-detect from mappings / recipe": "",
            },
            value=str(
                restore_state.get("image_column")
                or ""
            ),
        )
        self.mask_column = pn.widgets.Select(
            name="Mask input column",
            options={
                "Auto-detect from mappings / recipe": "",
            },
            value=str(
                restore_state.get("mask_column")
                or ""
            ),
        )
        self.inspect_data_button = pn.widgets.Button(
            name="Inspect selected datasets",
            button_type="light",
        )
        self.data_contract_summary = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            styles={
                "font-size": "12px",
                "line-height": "1.35",
                "overflow-wrap": "anywhere",
                "border": "1px solid rgba(0,0,0,0.10)",
                "border-radius": "4px",
                "padding": "8px",
                "background": "#fafafa",
            },
        )

        self.feature_filter = pn.widgets.TextInput(
            name="Feature filter / prefix",
            value="",
            placeholder=(
                "e.g. flux_, mag_, err_ — comma/newline separates "
                "multiple filters"
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.feature_replace_matching_button = pn.widgets.Button(
            name="Replace with matches",
            button_type="light",
            width=175,
            margin=(0, 8, 8, 0),
        )
        self.feature_add_matching_button = pn.widgets.Button(
            name="Add matches",
            button_type="light",
            width=120,
            margin=(0, 8, 8, 0),
        )
        self.feature_select_numeric_button = pn.widgets.Button(
            name="Select numeric features",
            button_type="light",
            width=170,
            margin=(0, 8, 8, 0),
        )
        self.feature_clear_button = pn.widgets.Button(
            name="Clear features",
            button_type="light",
            width=120,
            margin=(0, 0, 8, 0),
        )

        self.feature_columns = pn.widgets.TextAreaInput(
            name="Selected feature columns",
            value="\n".join(
                str(value)
                for value in list(
                    restore_state.get("feature_columns")
                    or []
                )
                if value is not None
            ),
            placeholder=(
                "Selected feature columns, one per line.\n"
                "Use the filter above and buttons below to bulk-add columns."
            ),
            height=130,
            min_height=130,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.feature_match_preview = pn.pane.Markdown(
            "",
            height=46,
            min_height=46,
            sizing_mode="stretch_width",
            margin=(0, 0, 10, 0),
            styles={
                "font-size": "12px",
                "color": "#666",
                "line-height": "1.25",
                "overflow": "hidden",
                "text-overflow": "ellipsis",
                "border": "1px solid rgba(0,0,0,0.08)",
                "border-radius": "4px",
                "padding": "6px",
                "background": "#fafafa",
            },
        )

        self.validation_fraction = pn.widgets.FloatInput(
            name="Validation random split fraction",
            value=float(
                restore_state.get(
                    "validation_fraction",
                    0.1,
                )
            ),
            start=0.0,
            end=0.8,
            step=0.01,
        )
        self.test_fraction = pn.widgets.FloatInput(
            name="Test random split fraction",
            value=float(
                restore_state.get(
                    "test_fraction",
                    0.2,
                )
            ),
            start=0.0,
            end=0.8,
            step=0.01,
        )

        self.validation_dataset_select = pn.widgets.Select(
            name="Validation dataset",
            options={
                "Split from AL training rows": "",
            },
            value="",
        )
        self.test_dataset_select = pn.widgets.Select(
            name="Test dataset",
            options={
                "Split from AL training rows": "",
            },
            value="",
        )
        self.al_protocol = pn.widgets.Select(
            name="AL protocol",
            options={
                "Review mode: train on verified labels": "review",
                "Benchmark mode: fixed holdout datasets": "benchmark",
            },
            value=str(
                restore_state.get("al_protocol")
                or "review"
            ),
        )

        self.seed = pn.widgets.IntInput(
            name="Initialisation/random seed",
            value=int(
                restore_state.get(
                    "seed",
                    42,
                )
            ),
            start=0,
        )
        self.initial_k = pn.widgets.IntInput(
            name="Initial random points",
            value=int(
                restore_state.get(
                    "initial_k",
                    20,
                )
            ),
            start=0,
        )
        self.query_k = pn.widgets.IntInput(
            name="Query batch size",
            value=int(
                restore_state.get(
                    "query_k",
                    200,
                )
            ),
            start=1,
        )

        self.label_select = pn.widgets.Select(
            name="Label current focus",
            options={},
        )
        self.advance_after_label = pn.widgets.Checkbox(
            name="Advance to next selected row after labelling",
            value=bool(
                restore_state.get(
                    "advance_after_label",
                    True,
                )
            ),
        )
        self.auto_label_n = pn.widgets.IntInput(
            name="Auto-label next N points",
            value=int(
                restore_state.get(
                    "auto_label_n",
                    50,
                )
            ),
            start=1,
        )

        self.auto_label_button = pn.widgets.Button(
            name="Label next N most informative points",
            button_type="primary",
        )

        self.auto_label_hint = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            styles={
                "font-size": "12px",
                "color": "#666",
                "overflow-wrap": "anywhere",
            },
        )

        self.auto_label_source_column = pn.widgets.Select(
            name="Auto-label from dataset column",
            options=[],
            value=restore_state.get(
                "auto_label_source_column"
            ),
        )

        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="light",
        )
        self.start_button = pn.widgets.Button(
            name="Start / random sample",
            button_type="primary",
        )
        self.query_button = pn.widgets.Button(
            name="Query next batch",
            button_type="primary",
        )
        self.label_button = pn.widgets.Button(
            name="Record label / verification",
            button_type="success",
        )
        self.unsure_button = pn.widgets.Button(
            name="Mark current as Unsure",
            button_type="warning",
        )
        self.train_button = pn.widgets.Button(
            name="Train round + predict pool + query next batch",
            button_type="danger",
        )
        self.auto_predict_after_train = pn.widgets.Checkbox(
            name="Predict over the original pool after training",
            value=bool(restore_state.get("auto_predict_after_train", True)),
        )
        self.auto_query_after_train = pn.widgets.Checkbox(
            name="Create and select the next query batch after prediction",
            value=bool(restore_state.get("auto_query_after_train", True)),
        )

        # Short, muted hints shown next to gated buttons explaining why they
        # are disabled, for example "label a point before training".
        self.query_hint = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
        )
        self.train_hint = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
        )

        self.status = pn.pane.Alert(
            "",
            alert_type="info",
            visible=False,
        )
        self.summary = pn.pane.Markdown("")
        self.batch_table = _make_table()
        self.labels_table = _make_table()

        self.session_json = pn.pane.JSON(
            {},
            depth=2,
            sizing_mode="stretch_width",
            height=260,
            styles={
                "max-width": "100%",
                "width": "100%",
                "overflow": "auto",
                "box-sizing": "border-box",
            },
        )

        self._analytics_frame_index = int(
            restore_state.get(
                "analytics_frame_index",
                0,
            )
        )
        self._analytics_frames: List[Dict[str, Any]] = []
        self._analytics_dirty = True
        self._source_label_cache: Dict[Any, Any] = {}

        self.analytics_x_column = pn.widgets.Select(
            name="Analytics X column",
            options=[],
            value=restore_state.get(
                "analytics_x_column"
            ),
        )
        self.analytics_y_column = pn.widgets.Select(
            name="Analytics Y column",
            options=[],
            value=restore_state.get(
                "analytics_y_column"
            ),
        )

        self.analytics_first_button = pn.widgets.Button(
            name="<<",
            button_type="light",
            width=44,
        )
        self.analytics_prev_button = pn.widgets.Button(
            name="<",
            button_type="light",
            width=44,
        )
        self.analytics_next_button = pn.widgets.Button(
            name=">",
            button_type="light",
            width=44,
        )
        self.analytics_last_button = pn.widgets.Button(
            name=">>",
            button_type="light",
            width=44,
        )
        self.analytics_refresh_button = pn.widgets.Button(
            name="Refresh analytics",
            button_type="light",
        )

        self.analytics_frame_label = pn.pane.Markdown(
            "_No active-learning analytics frame selected._",
            sizing_mode="stretch_width",
        )
        self.analytics_summary = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
        )

        self.analytics_scatter = pn.pane.Bokeh(
            al_analytics.make_empty_figure(),
            sizing_mode="stretch_width",
            height=450,
        )
        self.analytics_performance_plot = pn.pane.Bokeh(
            al_analytics.make_empty_figure(
                "No performance timeline yet."
            ),
            sizing_mode="stretch_width",
            height=300,
        )
        self.analytics_informativeness_plot = pn.pane.Bokeh(
            al_analytics.make_empty_figure(
                "No query informativeness timeline yet."
            ),
            sizing_mode="stretch_width",
            height=300,
        )
        self.analytics_points_table = _make_table()

        self.training_contract_summary = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            styles={
                "font-size": "12px",
                "line-height": "1.35",
                "overflow-wrap": "anywhere",
                "border": "1px solid rgba(0,0,0,0.10)",
                "border-radius": "4px",
                "padding": "8px",
                "background": "#fafafa",
            },
        )

        self._normalise_widget_layout()

        self.refresh_button.on_click(
            self._refresh_clicked
        )
        self.inspect_data_button.on_click(
            self._inspect_data_clicked
        )
        self.start_button.on_click(
            self._start_clicked
        )
        self.query_button.on_click(
            self._query_clicked
        )
        self.label_button.on_click(
            self._label_clicked
        )
        self.unsure_button.on_click(
            self._unsure_clicked
        )
        self.train_button.on_click(
            self._train_clicked
        )
        self.auto_label_button.on_click(
            self._auto_label_next_clicked
        )
        self.analytics_first_button.on_click(
            self._analytics_first_frame_clicked
        )
        self.analytics_prev_button.on_click(
            self._analytics_prev_frame_clicked
        )
        self.analytics_next_button.on_click(
            self._analytics_next_frame_clicked
        )
        self.analytics_last_button.on_click(
            self._analytics_last_frame_clicked
        )
        self.analytics_refresh_button.on_click(
            self._analytics_refresh_clicked
        )

        self.feature_add_matching_button.on_click(
            self._feature_add_matching_clicked
        )
        self.feature_replace_matching_button.on_click(
            self._feature_replace_matching_clicked
        )
        self.feature_select_numeric_button.on_click(
            self._feature_select_numeric_clicked
        )
        self.feature_clear_button.on_click(
            self._feature_clear_clicked
        )

        self.session_select.param.watch(
            self._session_selected,
            "value",
        )
        self.dataset_select.param.watch(
            self._dataset_changed,
            "value",
        )
        self.target_column.param.watch(
            self._target_column_changed,
            "value",
        )
        self.image_column.param.watch(
            self._input_binding_changed,
            "value",
        )
        self.mask_column.param.watch(
            self._input_binding_changed,
            "value",
        )

        self.auto_label_source_column.param.watch(
            self._auto_label_source_changed,
            "value",
        )
        self.feature_filter.param.watch(
            lambda *_: self._refresh_feature_match_preview(),
            "value",
        )
        self.feature_columns.param.watch(
            self._feature_columns_changed,
            "value",
        )
        self.validation_fraction.param.watch(
            lambda *_: self._refresh_training_contract_summary(),
            "value",
        )
        self.test_fraction.param.watch(
            lambda *_: self._refresh_training_contract_summary(),
            "value",
        )
        self.validation_dataset_select.param.watch(
            self._evaluation_source_changed,
            "value",
        )
        self.test_dataset_select.param.watch(
            self._evaluation_source_changed,
            "value",
        )

        self.al_protocol.param.watch(
            self._protocol_changed,
            "value",
        )

        self.recipe_select.param.watch(
            self._recipe_changed,
            "value",
        )
        self.predictions_select.param.watch(
            self._predictions_changed,
            "value",
        )
        self.label_options_select.param.watch(
            lambda *_: self._set_label_select_options(),
            "value",
        )

        self.analytics_x_column.param.watch(
            self._analytics_control_changed,
            "value",
        )
        self.analytics_y_column.param.watch(
            self._analytics_control_changed,
            "value",
        )

        self._syncing_auto_label_source = False
        self._auto_label_source_user_set = bool(
            restore_state.get(
                "auto_label_source_column"
            )
        )

        self._subscribe_refresh_events()
        self.refresh()
        self._restore_widget_state(
            restore_state
        )

    def panel(self):
        feature_selector_section = pn.Column(
            self.feature_filter,
            pn.Row(
                self.feature_replace_matching_button,
                self.feature_add_matching_button,
                sizing_mode="stretch_width",
                min_height=42,
                margin=(0, 0, 4, 0),
                styles={
                    "overflow": "visible",
                    "clear": "both",
                },
            ),
            pn.Row(
                self.feature_select_numeric_button,
                self.feature_clear_button,
                sizing_mode="stretch_width",
                min_height=42,
                margin=(0, 0, 4, 0),
                styles={
                    "overflow": "visible",
                    "clear": "both",
                },
            ),
            self.feature_columns,
            self.feature_match_preview,
            sizing_mode="stretch_width",
            min_height=300,
            height_policy="min",
            margin=(0, 0, 0, 0),
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "visible",
                "clear": "both",
                "padding": "0",
            },
        )

        self.image_column_field = self._start_field(self.image_column)
        self.mask_column_field = self._start_field(self.mask_column)
        self.feature_selector_field = self._start_field(
            feature_selector_section,
            min_height=330,
            margin=(0, 0, 24, 0),
        )
        self.validation_fraction_field = self._start_field(
            self.validation_fraction
        )
        self.test_fraction_field = self._start_field(
            self.test_fraction
        )

        start_section = self._tab_body(
            pn.Column(
                self._start_field(self.dataset_select),
                self._start_field(self.session_select),
                self._start_field(self.al_protocol),
                self._start_field(self.recipe_select),
                self._start_field(
                    self.recipe_card,
                    min_height=155,
                    margin=(0, 0, 18, 0),
                ),
                self._start_field(
                    self.label_options_select,
                    min_height=82,
                ),
                self._start_field(self.target_column),
                self.image_column_field,
                self.mask_column_field,
                self.feature_selector_field,
                self._start_field(
                    self.data_contract_summary,
                    min_height=150,
                ),
                self._start_field(
                    self.inspect_data_button,
                    min_height=48,
                ),
                self._start_field(self.seed),
                self._start_field(self.initial_k),
                self._start_field(
                    self.validation_dataset_select
                ),
                self.validation_fraction_field,
                self._start_field(
                    self.test_dataset_select
                ),
                self.test_fraction_field,
                self._start_field(
                    self.start_button,
                    min_height=48,
                    margin=(4, 0, 18, 0),
                ),
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
                styles={
                    "max-width": "100%",
                    "width": "100%",
                    "box-sizing": "border-box",
                    "overflow": "visible",
                    "clear": "both",
                },
            ),
        )

        query_section = self._tab_body(
            pn.Column(
                self.predictions_select,
                self.strategy_select,
                self.query_k,
                self.query_hint,
                self.query_button,
                sizing_mode="stretch_width",
            ),
        )

        review_section = self._tab_body(
            pn.Column(
                self.label_select,
                self.advance_after_label,
                self.label_button,
                self.unsure_button,
                pn.layout.Divider(
                    margin=(8, 0, 8, 0)
                ),
                self.auto_label_source_column,
                self.auto_label_n,
                self.auto_label_button,
                self.auto_label_hint,
                sizing_mode="stretch_width",
            ),
        )

        params_scroller = pn.Column(
            self.recipe_params_area,
            height=340,
            scroll=True,
            sizing_mode="stretch_width",
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "border": "1px solid rgba(0,0,0,0.10)",
                "border-radius": "4px",
                "padding": "6px",
            },
        )

        train_section = self._tab_body(
            pn.Column(
                self.training_contract_summary,
                pn.layout.Divider(
                    margin=(8, 0, 8, 0)
                ),
                params_scroller,
                pn.layout.Divider(
                    margin=(8, 0, 8, 0)
                ),
                self.auto_predict_after_train,
                self.auto_query_after_train,
                self.train_hint,
                self.train_button,
                sizing_mode="stretch_width",
            ),
        )

        controls_tabs = pn.Tabs(
            ("Start", start_section),
            ("Query", query_section),
            ("Review", review_section),
            ("Train", train_section),
            dynamic=True,
            sizing_mode="stretch_width",
            height_policy="auto",
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "visible",
            },
            css_classes=["al-controls-tabs"],
        )

        # Fixed-height scroll viewport: lower sections never start until below
        # the whole top workflow area. Tall tabs scroll inside this viewport.
        controls_frame = pn.Column(
            controls_tabs,
            sizing_mode="stretch_width",
            height=700,
            min_height=700,
            max_height=700,
            scroll=True,
            margin=(0, 0, 10, 0),
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "position": "relative",
                "border-bottom": "1px solid rgba(0,0,0,0.20)",
                "padding-bottom": "8px",
            },
            css_classes=["al-controls-frame"],
        )

        analytics_section = pn.Column(
            pn.pane.Markdown(
                "### AL analytics",
                sizing_mode="stretch_width",
                margin=(0, 0, 4, 0),
            ),
            pn.pane.Markdown(
                (
                    "Scatter frame shows all points that have been trained on up "
                    "to the selected AL round/frame. Marker colour is the stored "
                    "query-strategy value / informativeness score from "
                    "`ml.active_learning_batch` artifacts."
                ),
                sizing_mode="stretch_width",
            ),
            pn.Row(
                self.analytics_x_column,
                self.analytics_y_column,
                sizing_mode="stretch_width",
            ),
            pn.Row(
                self.analytics_first_button,
                self.analytics_prev_button,
                self.analytics_next_button,
                self.analytics_last_button,
                self.analytics_refresh_button,
                sizing_mode="stretch_width",
            ),
            self.analytics_frame_label,
            self.analytics_scatter,
            pn.layout.Divider(
                margin=(8, 0, 8, 0)
            ),
            self.analytics_performance_plot,
            pn.layout.Divider(
                margin=(8, 0, 8, 0)
            ),
            self.analytics_informativeness_plot,
            pn.layout.Divider(
                margin=(8, 0, 8, 0)
            ),
            self.analytics_summary,
            self.analytics_points_table,
            sizing_mode="stretch_width",
            scroll=True,
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow-y": "auto",
                "overflow-x": "hidden",
            },
        )

        results_tabs = pn.Tabs(
            (
                "Current query batch",
                self.batch_table,
            ),
            (
                "Labelled / verified / unsure",
                self.labels_table,
            ),
            (
                "Session JSON",
                self.session_json,
            ),
            (
                "Analytics",
                analytics_section,
            ),
            dynamic=True,
            sizing_mode="stretch_width",
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
            },
            css_classes=["al-results-tabs"],
        )

        self._controls_tabs = controls_tabs
        self._results_tabs = results_tabs
        self._refresh_input_contract_ui()
        self._refresh_evaluation_widget_visibility()

        try:
            results_tabs.param.watch(
                self._results_tab_changed,
                "active",
            )
        except Exception:
            pass

        return pn.Column(
            pn.pane.Markdown(
                "### Active Learning",
                sizing_mode="stretch_width",
                margin=(0, 0, 6, 0),
                styles={
                    "font-size": "15px",
                    "font-weight": "600",
                    "line-height": "1.2",
                    "overflow-wrap": "anywhere",
                },
            ),
            self.status,
            self.refresh_button,
            controls_frame,
            pn.layout.Divider(
                margin=(8, 0, 8, 0)
            ),
            self.summary,
            results_tabs,
            sizing_mode="stretch_width",
            scroll=True,
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "padding": "8px",
            },
            css_classes=["al-panel-root"],
        )

    def _switch_results_tab(
        self,
        tab_name: str,
    ) -> None:
        tabs = getattr(
            self,
            "_results_tabs",
            None,
        )

        if tabs is None:
            return

        tab_indices = {
            "Current query batch": 0,
            "Labelled / verified / unsure": 1,
            "Session JSON": 2,
            "Analytics": 3,
        }

        index = tab_indices.get(
            str(tab_name)
        )

        if index is None:
            return

        try:
            tabs.active = index
        except Exception:
            pass

    def _switch_controls_tab(
        self,
        tab_name: str,
    ) -> None:
        tabs = getattr(
            self,
            "_controls_tabs",
            None,
        )

        if tabs is None:
            return

        tab_indices = {
            "Start": 0,
            "Query": 1,
            "Review": 2,
            "Train": 3,
        }

        index = tab_indices.get(
            str(tab_name)
        )

        if index is None:
            return

        try:
            tabs.active = index
        except Exception:
            pass

    def refresh(
        self,
        *,
        include_analytics: bool = False,
    ) -> None:
        if self._disposed:
            return

        self._refresh_datasets()
        self._refresh_sessions()
        self._refresh_predictions()
        self._refresh_strategies()
        self._refresh_recipes()
        self._refresh_column_widgets()
        self._refresh_auto_label_source_column()
        self._refresh_validation_test_datasets()
        self._refresh_recipe_params()
        self._apply_recipe_inferred_defaults()
        self._refresh_labels_from_available_context()
        self._set_label_select_options()
        self._refresh_review_label_for_focus()
        self._refresh_session_views()

        if include_analytics:
            self._maybe_refresh_analytics_views(
                force=True
            )
        else:
            self._mark_analytics_dirty()

        self._update_action_gating()

    def dispose(self) -> None:
        self._disposed = True
        handle = self._active_job_handle
        self._active_job_handle = None

        cancel = getattr(
            handle,
            "cancel",
            None,
        )

        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass

        events = getattr(
            self.context,
            "events",
            None,
        )

        unsubscribe = getattr(
            events,
            "unsubscribe",
            None,
        )

        if callable(unsubscribe):
            for sub in list(
                self._subscriptions
            ):
                try:
                    unsubscribe(sub)
                except Exception:
                    pass

        self._subscriptions = []

    def get_state(self) -> Dict[str, Any]:
        return {
            "state_version": self.state_version,
            "session_artifact_id": self._current_session_artifact_id,
            "dataset_id": self.dataset_select.value,
            "recipe_id": self.recipe_select.value,
            "label_options": list(
                self.label_options_select.value
                or []
            ),
            "target_column": self.target_column.value,
            "image_column": self.image_column.value,
            "mask_column": self.mask_column.value,
            "feature_columns": self._selected_feature_columns(),
            "session_contract": dict(
                self._latest_session_contract
                or {}
            ),
            "validation_dataset_id": self.validation_dataset_select.value,
            "validation_fraction": float(
                self.validation_fraction.value
                or 0.0
            ),
            "test_dataset_id": self.test_dataset_select.value,
            "test_fraction": float(
                self.test_fraction.value
                or 0.0
            ),
            "al_protocol": self.al_protocol.value,
            "seed": int(
                self.seed.value
                or 0
            ),
            "initial_k": int(
                self.initial_k.value
                or 0
            ),
            "query_k": int(
                self.query_k.value
                or 1
            ),
            "auto_predict_after_train": bool(
                self.auto_predict_after_train.value
            ),
            "auto_query_after_train": bool(
                self.auto_query_after_train.value
            ),
            "strategy_id": self.strategy_select.value,
            "recipe_params": self._recipe_params(),
            "advance_after_label": bool(
                self.advance_after_label.value
            ),
            "auto_label_n": int(
                self.auto_label_n.value
                or 1
            ),
            "auto_label_source_column": self.auto_label_source_column.value,
            "analytics_x_column": self.analytics_x_column.value,
            "analytics_y_column": self.analytics_y_column.value,
            "analytics_frame_index": int(
                self._analytics_frame_index
                or 0
            ),
        }

    def _refresh_clicked(
        self,
        *_: Any,
    ) -> None:
        self.refresh(
            include_analytics=True
        )
        self._set_status(
            "Refreshed.",
            "info",
        )

    def _inspect_data_clicked(
        self,
        *_: Any,
    ) -> None:
        try:
            dataset_id = self._require_dataset()
            recipe_id = str(
                self.recipe_select.value
                or ""
            ).strip()

            if not recipe_id:
                raise ValueError(
                    "Choose a core.ml recipe before inspecting data."
                )

            request = ActionRequest(
                dataset_id=dataset_id,
                params=self._profile_request_params(
                    inspect_images=True
                ),
                origin="core.active_learning.panel",
            )

            manager = getattr(
                self.context,
                "plugins",
                None,
            )

            run_action = getattr(
                manager,
                "run_action",
                None,
            )

            self._set_running(True)
            self._set_status(
                "Inspecting recipe inputs and a small image sample...",
                "info",
            )

            if callable(run_action):
                self._active_job_handle = run_action(
                    "core.active_learning.profile_data_contract",
                    self.context,
                    request,
                    on_done=self._on_profile_done,
                    on_error=self._on_profile_error,
                )
                return

            submit = getattr(
                getattr(
                    self.context,
                    "jobs",
                    None,
                ),
                "submit",
                None,
            )

            if callable(submit):
                self._active_job_handle = submit(
                    actions.profile_data_contract_action,
                    title="Inspect active-learning data contract",
                    key=(
                        "core.active_learning.profile:"
                        f"{dataset_id}:{recipe_id}"
                    ),
                    on_done=self._on_profile_done,
                    on_error=self._on_profile_error,
                    context=self.context,
                    request=request,
                )
                return

            self._on_profile_done(
                actions.profile_data_contract_action(
                    self.context,
                    request,
                )
            )

        except Exception as exc:
            self._set_running(False)
            self._set_error(
                "Could not inspect active-learning data",
                exc,
            )

    def _on_profile_done(
        self,
        result: Any,
    ) -> None:
        if self._disposed:
            return

        self._active_job_handle = None
        self._set_running(False)

        if not isinstance(
            result,
            Mapping,
        ):
            self._set_status(
                "Data inspection returned an unexpected result.",
                "warning",
            )
            return

        contract = dict(
            result.get("contract")
            or {}
        )

        report = al_contracts.PreflightReport(
            contract=contract,
            errors=list(
                result.get("errors")
                or []
            ),
            warnings=list(
                result.get("warnings")
                or []
            ),
        )

        self._remember_preflight(
            report
        )

        if report.errors:
            self._set_status(
                "Data inspection completed with blocking compatibility errors.",
                "danger",
            )
        elif report.warnings:
            self._set_status(
                "Data inspection completed with warnings. Review the data contract.",
                "warning",
            )
        else:
            self._set_status(
                "Data inspection completed successfully.",
                "success",
            )

    def _on_profile_error(
        self,
        error: Any,
    ) -> None:
        if self._disposed:
            return

        self._active_job_handle = None
        self._set_running(False)
        self._set_error(
            "Could not inspect active-learning data",
            error,
        )

    def _start_clicked(
        self,
        *_: Any,
    ) -> None:
        try:
            dataset_id = self._require_dataset()
            labels = self._current_label_options()

            if not labels:
                raise ValueError(
                    "No label options are available. Choose a populated label column, "
                    "or choose a recipe/predictions artifact that exposes class labels."
                )

            preflight = self._build_session_contract(
                inspect_images=False
            )
            preflight.raise_for_errors()
            self._remember_preflight(
                preflight
            )

            start_params = self._start_owned_recipe_params()

            request = ActionRequest(
                dataset_id=dataset_id,
                params={
                    "dataset_id": dataset_id,
                    "pool_dataset_id": dataset_id,
                    "recipe_id": (
                        self.recipe_select.value
                        or ""
                    ),
                    "label_options": labels,
                    "target_column": (
                        self.target_column.value
                        or "al_label"
                    ),
                    "validation_dataset_id": (
                        self.validation_dataset_select.value
                        or ""
                    ),
                    "test_dataset_id": (
                        self.test_dataset_select.value
                        or ""
                    ),
                    "al_protocol": (
                        self.al_protocol.value
                        or "review"
                    ),
                    "initial_k": int(
                        self.initial_k.value
                        or 0
                    ),
                    "seed": int(
                        self.seed.value
                        or 0
                    ),
                    "make_selection": True,
                    "session_contract": preflight.contract,
                    **start_params,
                },
                origin="core.active_learning.panel",
            )

            result = actions.start_session_action(
                self.context,
                request,
            )

            self._use_action_session_result(
                result
            )
            self._switch_controls_tab(
                "Review"
            )
            self._switch_results_tab(
                "Current query batch"
            )
            self._set_status(
                (
                    "Started session with "
                    f"{len(result.get('row_ids') or [])} "
                    "random rows."
                ),
                "success",
            )

        except Exception as exc:
            self._set_error(
                "Could not start active-learning session",
                exc,
            )

    def _apply_session_contract_to_widgets(
        self,
        session: Mapping[str, Any],
    ) -> None:
        contract = dict(session.get("contract") or {})
        if not contract:
            return

        recipe = dict(contract.get("recipe") or {})
        bindings = dict(contract.get("bindings") or {})
        pool = dict(bindings.get("pool") or {})
        protocol = dict(contract.get("protocol") or {})

        self._restore_select_value(
            self.dataset_select,
            str(pool.get("dataset_id") or session.get("dataset_id") or ""),
        )
        self._restore_select_value(
            self.recipe_select,
            str(recipe.get("id") or session.get("recipe_id") or ""),
        )

        self._refresh_column_widgets()
        self._refresh_recipe_params()

        target = str(
            pool.get("target_column")
            or session.get("target_column")
            or "al_label"
        )
        self._restore_select_value(self.target_column, target)
        self._restore_select_value(
            self.image_column,
            str(pool.get("image_column") or ""),
        )
        self._restore_select_value(
            self.mask_column,
            str(pool.get("mask_column") or ""),
        )
        self._set_feature_columns_value(pool.get("feature_columns") or [])

        self._restore_select_value(
            self.validation_dataset_select,
            str(session.get("validation_dataset_id") or ""),
        )
        self._restore_select_value(
            self.test_dataset_select,
            str(session.get("test_dataset_id") or ""),
        )
        self._restore_select_value(
            self.al_protocol,
            str(session.get("al_protocol") or "review"),
        )

        if str(protocol.get("protocol_validation_source") or "") == "split":
            try:
                self.validation_fraction.value = float(
                    protocol.get("protocol_validation_size")
                    if protocol.get("protocol_validation_size") is not None
                    else self.validation_fraction.value
                )
            except Exception:
                pass

        if str(protocol.get("protocol_test_source") or "") == "split":
            try:
                self.test_fraction.value = float(
                    protocol.get("protocol_test_size")
                    if protocol.get("protocol_test_size") is not None
                    else self.test_fraction.value
                )
            except Exception:
                pass

        labels = al_state.parse_label_options(session.get("label_options"))
        if labels:
            self.label_options_select.options = list(
                dict.fromkeys(
                    [
                        *list(self.label_options_select.options or []),
                        *labels,
                    ]
                )
            )
            self.label_options_select.value = labels

        recipe_params = dict(recipe.get("params") or {})
        for name, widget in self.recipe_param_widgets.items():
            if name in recipe_params:
                self._restore_widget_value(widget, recipe_params[name])

        self._latest_session_contract = contract
        self._latest_preflight_errors = list(
            (contract.get("preflight") or {}).get("errors") or []
        )
        self._latest_preflight_warnings = list(
            (contract.get("preflight") or {}).get("warnings") or []
        )

        self._refresh_input_contract_ui()
        self._refresh_evaluation_widget_visibility()
        self._refresh_data_contract_summary()

    def _label_options_changed(self, *_: Any) -> None:
        self._set_label_select_options()

    def _refresh_datasets(self) -> None:
        current = self.dataset_select.value
        ids = self._dataset_ids()
        self.dataset_select.options = ids

        active = None
        try:
            active = self.context.datasets.active_id()
        except Exception:
            active = None

        if current in ids:
            self.dataset_select.value = current
        elif active in ids:
            self.dataset_select.value = active
        elif ids:
            self.dataset_select.value = ids[0]
        else:
            self.dataset_select.value = None

    def _refresh_sessions(self) -> None:
        current = self._current_session_artifact_id or self.session_select.value
        latest: Dict[str, Dict[str, Any]] = {}

        try:
            refs = self.context.artifacts.find(type=al_state.ARTIFACT_SESSION)
        except Exception:
            refs = []

        for ref in refs:
            try:
                payload = al_state.coerce_session(
                    self.context.artifacts.get(ref.artifact_id)
                )
            except Exception:
                continue

            session_id = str(payload.get("session_id") or ref.artifact_id)
            updated_at = float(payload.get("updated_at") or 0.0)
            revision = int(payload.get("revision") or 0)
            existing = latest.get(session_id)

            if existing is None or (updated_at, revision) > (
                existing["updated_at"],
                existing["revision"],
            ):
                latest[session_id] = {
                    "artifact_id": ref.artifact_id,
                    "payload": payload,
                    "updated_at": updated_at,
                    "revision": revision,
                }

        options: Dict[str, str] = {}

        for session_id, item in sorted(
            latest.items(),
            key=lambda kv: float(kv[1]["updated_at"]),
            reverse=True,
        ):
            payload = item["payload"]
            counts = al_state.counts(payload)
            dataset_id = payload.get("dataset_id", "")

            label = (
                f"{session_id} | {dataset_id} | r{payload.get('round')} "
                f"rev{payload.get('revision', 0)} | "
                f"train={counts['labelled_or_verified']} "
                f"unsure={counts['unsure']}"
            )
            options[label] = item["artifact_id"]

        self.session_select.options = options
        values = set(options.values())

        if current and current in values:
            self.session_select.value = current
            self._current_session_artifact_id = current
        elif options:
            first_value = next(iter(options.values()))
            self.session_select.value = first_value
            self._current_session_artifact_id = first_value
        else:
            self.session_select.value = None
            self._current_session_artifact_id = None

    def _refresh_predictions(self) -> None:
        current = self.predictions_select.value
        options: Dict[str, str] = {}

        session = self._load_current_session()
        expected_dataset_id = self._session_pool_dataset_id(session)

        total_prediction_artifacts = 0
        skipped_wrong_dataset = 0

        try:
            refs = self.context.artifacts.find(type="ml.predictions")
        except Exception:
            refs = []

        for ref in refs:
            artifact_id = str(
                getattr(ref, "artifact_id", "")
                or ""
            )
            if not artifact_id:
                continue

            total_prediction_artifacts += 1

            try:
                payload = self.context.artifacts.get(artifact_id)
                if not isinstance(payload, Mapping):
                    payload = {}
            except Exception:
                payload = {}

            dataset_id = str(
                payload.get("dataset_id")
                or getattr(ref, "dataset_id", "")
                or ""
            ).strip()

            # Once a session exists, only offer predictions made over the AL
            # pool dataset. Do not offer predictions over derived datasets such
            # as "...__al_train_r1_...".
            if expected_dataset_id and dataset_id != expected_dataset_id:
                skipped_wrong_dataset += 1
                continue

            label = self._prediction_option_label(
                artifact_id=artifact_id,
                payload=payload,
            )
            options[label] = artifact_id

        self.predictions_select.options = options

        latest_prediction = (
            al_state.latest_reference(session, "predictions_artifact_id")
            if session
            else None
        )

        if latest_prediction in options.values():
            self.predictions_select.value = latest_prediction
        elif current in options.values():
            self.predictions_select.value = current
        elif options:
            self.predictions_select.value = next(iter(options.values()))
        else:
            self.predictions_select.value = None

        if (
            expected_dataset_id
            and not options
            and total_prediction_artifacts
        ):
            self.query_hint.object = (
                "_No pool prediction is available. Run the Train workflow with "
                "automatic prediction enabled, or provide a prediction artifact manually._"
            )
        elif expected_dataset_id and skipped_wrong_dataset:
            self.query_hint.object = (
                f"_Filtered out {skipped_wrong_dataset} prediction artifact(s) "
                "because they were not made on the AL pool dataset._"
            )

    def _refresh_strategies(self) -> None:
        current = self.strategy_select.value
        options: Dict[str, str] = {}

        try:
            registry = self.context.services.get(
                "core.active_learning.query_strategy_registry"
            )

            for info in registry.list():
                options[f"{info.title} ({info.id})"] = info.id

        except Exception:
            options = {
                "Least confidence (least_confidence)": "least_confidence",
                "Smallest margin (margin)": "margin",
                "Entropy (entropy)": "entropy",
                "Random (random)": "random",
            }

        self.strategy_select.options = options

        if current in options.values():
            self.strategy_select.value = current
        elif options:
            self.strategy_select.value = next(iter(options.values()))

    def _refresh_session_views(self) -> None:
        session = self._load_current_session()

        if not session:
            self.summary.object = ""
            _set_table_value(self.batch_table, pd.DataFrame())
            _set_table_value(self.labels_table, pd.DataFrame())
            self.session_json.object = {}
            return

        counts = al_state.counts(session)
        last_batch = session.get("last_batch") or {}

        session_contract = self._contract_with_labelled_count(
            dict(session.get("contract") or {}),
            counts["labelled_or_verified"],
        )

        contract_preflight = dict(
            session_contract.get("preflight")
            or {}
        )

        contract_markdown = (
            self._contract_markdown(
                session_contract,
                errors=list(
                    contract_preflight.get("errors")
                    or []
                ),
                warnings=list(
                    contract_preflight.get("warnings")
                    or []
                ),
            )
            if session_contract
            else "_This older session has no persisted recipe/data contract._"
        )

        self.summary.object = (
            f"### Session `{session.get('session_id')}`\n\n"
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| Dataset | `{session.get('dataset_id')}` |\n"
            f"| Pool | "
            f"`{session.get('pool_dataset_id') or session.get('dataset_id')}` |\n"
            f"| Recipe | "
            f"`{session.get('recipe_id') or (session_contract.get('recipe') or {}).get('id') or ''}` |\n"
            f"| Round | `{session.get('round')}` |\n"
            f"| Verified training labels | "
            f"**{counts['labelled_or_verified']}** |\n"
            f"| Unsure / ignored only | **{counts['unsure']}** |\n"
            f"| Total ignored from pool | **{counts['ignored']}** |\n"
            f"| Last batch | "
            f"`{last_batch.get('strategy_id') or 'none'}` "
            f"({last_batch.get('count') or 0} rows) |\n\n"
            f"{contract_markdown}"
        )

        labels_df = pd.DataFrame(
            al_state.label_rows(session)
        )
        _set_table_value(
            self.labels_table,
            labels_df,
        )

        batch_df = pd.DataFrame()
        batch_artifact_id = last_batch.get(
            "batch_artifact_id"
        )

        if batch_artifact_id:
            try:
                batch_payload = self.context.artifacts.get(
                    batch_artifact_id
                )
                records = list(
                    batch_payload.get("records")
                    or []
                )
                batch_df = pd.DataFrame(
                    records[:1000]
                )
            except Exception:
                batch_df = pd.DataFrame()

        _set_table_value(
            self.batch_table,
            batch_df,
        )
        self.session_json.object = session

    def _contract_with_labelled_count(
        self,
        contract: Mapping[str, Any],
        labelled_count: int,
    ) -> Dict[str, Any]:
        payload = dict(contract or {})

        if not payload:
            return payload

        protocol = dict(
            payload.get("protocol")
            or {}
        )
        profiles = dict(
            payload.get("profiles")
            or {}
        )

        validation_source = str(
            protocol.get("protocol_validation_source")
            or "split"
        )
        test_source = str(
            protocol.get("protocol_test_source")
            or "split"
        )
        validation_fraction = float(
            protocol.get("protocol_validation_size")
            or 0.0
        )
        test_fraction = float(
            protocol.get("protocol_test_size")
            or 0.0
        )
        labelled_count = max(
            0,
            int(labelled_count or 0),
        )

        validation_rows = (
            int(
                round(
                    labelled_count
                    * validation_fraction
                )
            )
            if validation_source == "split"
            else (
                profiles.get("validation")
                or {}
            ).get("row_count")
        )

        test_rows = (
            int(
                round(
                    labelled_count
                    * test_fraction
                )
            )
            if test_source == "split"
            else (
                profiles.get("test")
                or {}
            ).get("row_count")
        )

        internal_validation = (
            validation_rows
            if validation_source == "split"
            else 0
        )
        internal_test = (
            test_rows
            if test_source == "split"
            else 0
        )

        payload["split_estimates"] = {
            "labelled_total": labelled_count,
            "train": max(
                0,
                labelled_count
                - int(internal_validation or 0)
                - int(internal_test or 0),
            ),
            "validation": validation_rows,
            "test": test_rows,
        }

        return payload

    def _use_action_session_result(
        self,
        result: Mapping[str, Any],
        *,
        include_analytics: bool = False,
    ) -> None:
        session_artifact_id = result.get(
            "session_artifact_id"
        )

        if session_artifact_id:
            self._current_session_artifact_id = str(
                session_artifact_id
            )

        self.refresh(
            include_analytics=include_analytics
        )

    def _mark_analytics_dirty(self) -> None:
        self._analytics_dirty = True

    def _analytics_tab_is_active(self) -> bool:
        tabs = getattr(
            self,
            "_results_tabs",
            None,
        )

        if tabs is None:
            return False

        try:
            return int(
                getattr(
                    tabs,
                    "active",
                    -1,
                )
            ) == 3
        except Exception:
            return False

    def _refresh_analytics_if_visible(
        self,
        *,
        force: bool = False,
    ) -> None:
        if self._analytics_tab_is_active():
            self._maybe_refresh_analytics_views(
                force=force
            )
        else:
            self._mark_analytics_dirty()

    def _maybe_refresh_analytics_views(
        self,
        *,
        force: bool = False,
    ) -> None:
        if self._disposed:
            return

        if (
            not force
            and not bool(
                getattr(
                    self,
                    "_analytics_dirty",
                    True,
                )
            )
        ):
            return

        self._refresh_analytics_columns()
        self._refresh_analytics_views()
        self._analytics_dirty = False

    def _analytics_control_changed(
        self,
        *_: Any,
    ) -> None:
        self._mark_analytics_dirty()
        self._refresh_analytics_if_visible(
            force=True
        )

    def _results_tab_changed(
        self,
        event: Any,
    ) -> None:
        try:
            active = int(
                getattr(
                    event,
                    "new",
                    -1,
                )
            )
        except Exception:
            active = -1

        if active == 3:
            self._maybe_refresh_analytics_views()

    def _set_label_select_options(self) -> None:
        labels = al_state.parse_label_options(
            self.label_options_select.value
        )

        options = {
            label: label
            for label in labels
        }
        options[
            al_state.UNSURE_DISPLAY
        ] = al_state.UNSURE_LABEL

        current = self.label_select.value
        self.label_select.options = options

        if current in options.values():
            self.label_select.value = current
        elif options:
            self.label_select.value = (
                al_state.UNSURE_LABEL
            )

    def _next_review_row_id_after(
        self,
        current_row_id: Any,
    ) -> Optional[str]:
        """Find the next unlabelled row in the current ordered AL batch.

        This is the critical Review hot-path lookup. It must not call
        context.datasets.get_df(), get_rows_by_ids(), analytics helpers, or any
        visualisation code.
        """

        current_row_id = str(current_row_id or "").strip()
        row_ids = self._current_ranked_review_row_ids()

        if not row_ids:
            return None

        skip = self._session_labelled_or_ignored_row_ids()

        if current_row_id:
            skip.add(current_row_id)

        start_index = 0

        if current_row_id in row_ids:
            start_index = row_ids.index(current_row_id) + 1

        # Prefer rows after the current row.
        for row_id in row_ids[start_index:]:
            row_id = str(row_id)

            if row_id and row_id not in skip:
                return row_id

        # Wrap around only if needed.
        for row_id in row_ids[:start_index]:
            row_id = str(row_id)

            if row_id and row_id not in skip:
                return row_id

        return None

    def _set_focus_next_tick(
        self,
        *,
        dataset_id: str,
        row_id: str,
        origin: str,
    ) -> None:
        """Set platform focus outside the label-click hot path.

        selection.set_focus publishes selection.focus.changed synchronously.
        Some subscribers, especially visualisation panels, may need to resolve
        row data. Scheduling this avoids making the label button wait for those
        subscribers.
        """

        def apply() -> None:
            if self._disposed:
                return

            selection = getattr(self.context, "selection", None)

            if selection is None:
                return

            try:
                if hasattr(selection, "set_focus"):
                    selection.set_focus(
                        dataset_id=dataset_id,
                        row_id=str(row_id),
                        origin=origin,
                    )

                elif hasattr(selection, "focus"):
                    selection.focus(
                        dataset_id=dataset_id,
                        row_id=str(row_id),
                        origin=origin,
                    )

            except TypeError:
                try:
                    selection.set_focus(dataset_id, str(row_id))

                except Exception:
                    pass

            except Exception:
                pass

        try:
            curdoc = getattr(pn.state, "curdoc", None)

            if curdoc is not None:
                curdoc.add_next_tick_callback(apply)
            else:
                apply()

        except Exception:
            apply()

    def _load_current_session(self) -> Optional[Dict[str, Any]]:
        if not self._current_session_artifact_id:
            return None

        try:
            return al_state.coerce_session(
                self.context.artifacts.get(
                    self._current_session_artifact_id
                )
            )

        except Exception:
            return None

    def _session_pool_dataset_id(
        self,
        session: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Dataset that AL should query from.

        This is the original pool/source dataset, not the derived AL training
        dataset created for a particular round.
        """

        if session is None:
            session = self._load_current_session() or {}

        return str(
            session.get("pool_dataset_id")
            or session.get("dataset_id")
            or self.dataset_select.value
            or ""
        ).strip()

    def _prediction_dataset_id(
        self,
        artifact_id: str,
    ) -> str:
        artifact_id = str(artifact_id or "").strip()

        if not artifact_id:
            return ""

        try:
            payload = self.context.artifacts.get(artifact_id)

            if isinstance(payload, Mapping):
                return str(
                    payload.get("dataset_id")
                    or ""
                ).strip()

        except Exception:
            pass

        try:
            refs = self.context.artifacts.find(
                type="ml.predictions"
            )

            for ref in refs:
                if str(
                    getattr(ref, "artifact_id", "")
                ) == artifact_id:
                    return str(
                        getattr(ref, "dataset_id", "")
                        or ""
                    ).strip()

        except Exception:
            pass

        return ""

    def _prediction_matches_current_session(
        self,
        artifact_id: str,
    ) -> bool:
        artifact_id = str(artifact_id or "").strip()

        if not artifact_id:
            return False

        expected_dataset_id = self._session_pool_dataset_id()
        actual_dataset_id = self._prediction_dataset_id(artifact_id)

        if not expected_dataset_id:
            return True

        return bool(
            actual_dataset_id
            and actual_dataset_id == expected_dataset_id
        )

    def _prediction_option_label(
        self,
        *,
        artifact_id: str,
        payload: Mapping[str, Any],
    ) -> str:
        dataset_id = str(
            payload.get("dataset_id")
            or ""
        )

        count = len(
            payload.get("records")
            or []
        )

        model = (
            payload.get("model")
            if isinstance(
                payload.get("model"),
                Mapping,
            )
            else {}
        )

        recipe_id = str(
            payload.get("recipe_id")
            or model.get("recipe_id")
            or ""
        )

        run_id = str(
            payload.get("run_id")
            or ""
        )

        bits = [artifact_id[:8]]

        if dataset_id:
            bits.append(dataset_id)

        if count:
            bits.append(f"{count} rows")

        if recipe_id:
            bits.append(recipe_id)

        if run_id:
            bits.append(f"run {run_id[:8]}")

        return " | ".join(bits)

    def _label_for_current_focus(self) -> Optional[str]:
        dataset_id, row_id = self._current_focus_ref()

        if not row_id:
            return al_state.UNSURE_LABEL

        session = self._load_current_session() or {}

        session_dataset_id = str(
            session.get("dataset_id")
            or ""
        )

        if (
            dataset_id
            and session_dataset_id
            and dataset_id != session_dataset_id
        ):
            return al_state.UNSURE_LABEL

        session_label = self._session_label_for_row(
            session,
            row_id,
        )

        if session_label:
            return session_label

        source_label = self._source_label_for_row(
            dataset_id=dataset_id or session_dataset_id,
            row_id=row_id,
        )

        if source_label:
            return source_label

        return al_state.UNSURE_LABEL

    def _session_label_for_row(
        self,
        session: Mapping[str, Any],
        row_id: Any,
    ) -> Optional[str]:
        row_id = str(row_id)
        labels = dict(session.get("labels") or {})
        entry = labels.get(row_id)

        if not isinstance(entry, Mapping):
            return None

        label = entry.get("label")

        if label is None:
            return None

        label = al_state.normalise_label(label)

        return label or None

    def _source_label_for_row(
        self,
        *,
        dataset_id: Optional[str],
        row_id: Any,
    ) -> Optional[str]:
        dataset_id = str(dataset_id or "").strip()
        row_id = str(row_id or "").strip()

        if not dataset_id or not row_id:
            return None

        column = str(
            self.auto_label_source_column.value
            or ""
        ).strip()

        if not column:
            column = str(
                self.target_column.value
                or ""
            ).strip()

        if not column or column == "al_label":
            return None

        columns = set(
            self._physical_columns_for_dataset(
                dataset_id
            )
        )

        if column not in columns:
            # The widget value may occasionally be a semantic mapping name
            # rather than the physical column. Resolve it defensively.
            try:
                mapped = self.context.datasets.get_mapping(
                    dataset_id,
                    column,
                )

            except Exception:
                mapped = None

            mapped = str(mapped or "").strip()

            if mapped and mapped in columns:
                column = mapped
            else:
                return None

        value = self._dataset_value_for_row(
            dataset_id=dataset_id,
            row_id=row_id,
            column=column,
        )

        if value is None:
            return None

        label = self._normalise_auto_label_value(
            value,
            label_options=self._current_label_options(),
        )

        if label is None:
            return None

        label = al_state.normalise_label(label)

        return label or None

    def _dataset_value_for_row(
        self,
        *,
        dataset_id: str,
        row_id: str,
        column: str,
    ) -> Optional[Any]:
        dataset_id = str(dataset_id or "").strip()
        row_id = str(row_id or "").strip()
        column = str(column or "").strip()

        if not dataset_id or not row_id or not column:
            return None

        id_column = self._dataset_mapping(
            dataset_id,
            "record_id",
        )

        if not id_column:
            columns = self._dataset_columns(dataset_id)

            for candidate in (
                "record_id",
                "id",
                "ID",
                "source_id",
                "object_id",
                "row_id",
            ):
                if candidate in columns:
                    id_column = candidate
                    break

        requested_key_map = self._requested_row_id_key_map([row_id])

        if id_column:
            try:
                df = self.context.datasets.get_rows_by_ids(
                    dataset_id,
                    [row_id],
                    id_column=id_column,
                )
                if column in df.columns and not df.empty:
                    return df.iloc[0][column]
            except Exception:
                pass

        try:
            try:
                needed = [column]
                if id_column:
                    needed.append(id_column)
                df = self.context.datasets.get_df(
                    dataset_id,
                    columns=list(dict.fromkeys(needed)),
                )
            except TypeError:
                df = self.context.datasets.get_df(dataset_id)

            if column not in df.columns:
                return None

            if id_column and id_column in df.columns:
                mask = self._row_id_series_mask(df[id_column], requested_key_map)
                match = df[mask]
                if not match.empty:
                    return match.iloc[0][column]

            index_matches = []
            for idx in df.index:
                if self._row_id_match_keys(idx) & set(requested_key_map.keys()):
                    index_matches.append(idx)
                    break

            if index_matches:
                return df.loc[index_matches[0], column]

        except Exception:
            return None

        return None

    def _current_focus_ref(self) -> tuple[Optional[str], Optional[str]]:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None, None

        focus = selection.get_focus()
        dataset_id = getattr(focus, "dataset_id", None)
        row_id = getattr(focus, "row_id", None)

        return (
            None if dataset_id is None else str(dataset_id),
            None if row_id is None else str(row_id),
        )

    def _current_focus_row_id(self) -> Optional[str]:
        dataset_id, row_id = self._current_focus_ref()
        session = self._load_current_session() or {}
        session_dataset_id = str(session.get("dataset_id") or "")

        if (
            dataset_id
            and session_dataset_id
            and dataset_id != session_dataset_id
        ):
            raise ValueError(
                f"Focused row belongs to dataset {dataset_id!r}, but the AL session "
                f"belongs to {session_dataset_id!r}."
            )

        return row_id

    def _subscribe_refresh_events(self) -> None:
        events = getattr(self.context, "events", None)
        subscribe = getattr(events, "subscribe", None)

        if not callable(subscribe):
            return

        topics = [
            "plugin.enabled",
            "plugin.disabled",
            "artifact.created",
            "artifact.updated",
            "dataset.active.changed",
            "dataset.registered",
            "dataset.loaded",
            "dataset.updated",
            "dataset.*",
            "selection.focus.changed",
            "selection.focus.cleared",
            "ml.recipe.registered",
            "ml.recipe.unregistered",
            "ml.model_definition.created",
            "ml.model.created",
            "ml.run.created",
            "ml.predictions.created",
            "al.session.saved",
            "al.query_batch.created",
            "al.round.training_started",
            "al.round.training_finished",
            "al.labels.recorded_bulk",
        ]

        for topic in topics:
            try:
                self._subscriptions.append(
                    subscribe(
                        topic,
                        self._on_external_refresh_event,
                    )
                )
            except Exception:
                pass

    def _on_external_refresh_event(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """React to platform events.

        The platform EventBus calls subscribers as cb(topic, payload). This
        method accepts multiple callback shapes so event handling never breaks
        the panel.

        Important for Review performance:
        single-label actions publish AL/session events. Do not respond to those
        events with a full refresh or eager analytics rebuild.
        """

        if self._disposed:
            return

        topic = None
        payload: Any = None

        if len(args) >= 2:
            topic = args[0]
            payload = args[1]

        elif len(args) == 1:
            event = args[0]

            if isinstance(event, Mapping):
                topic = event.get("topic")
                payload = event
            else:
                topic = getattr(event, "topic", None)
                payload = getattr(event, "payload", None)

        if topic is None:
            topic = kwargs.get("topic")

        if payload is None:
            payload = kwargs.get(
                "payload",
                kwargs,
            )

        topic = str(topic or "")
        payload = (
            payload
            if isinstance(payload, Mapping)
            else {}
        )

        def apply() -> None:
            if self._disposed:
                return

            if topic in {
                "selection.focus.changed",
                "selection.focus.cleared",
            }:
                self._refresh_review_label_for_focus()
                return

            if (
                topic
                in {
                    "dataset.active.changed",
                    "dataset.registered",
                    "dataset.loaded",
                    "dataset.updated",
                }
                or topic.startswith("dataset.")
                or topic.startswith("mapping.")
            ):
                self._source_label_cache.clear()
                self._refresh_dataset_dependent_widgets()
                self._mark_analytics_dirty()
                self._refresh_analytics_if_visible()
                return

            if topic == "al.session.saved":
                session_artifact_id = str(
                    payload.get("session_artifact_id")
                    or ""
                ).strip()

                if session_artifact_id:
                    self._current_session_artifact_id = (
                        session_artifact_id
                    )

                self._refresh_session_views()
                self._update_action_gating()
                self._mark_analytics_dirty()
                self._refresh_analytics_if_visible()
                return

            if topic == "al.labels.recorded_bulk":
                session_artifact_id = str(
                    payload.get("session_artifact_id")
                    or ""
                ).strip()

                if session_artifact_id:
                    self._current_session_artifact_id = (
                        session_artifact_id
                    )

                self._refresh_session_views()
                self._update_action_gating()
                self._mark_analytics_dirty()
                self._refresh_analytics_if_visible()
                return

            if topic in {
                "al.query_batch.created",
                "al.round.training_started",
                "al.round.training_finished",
                "ml.model_definition.created",
                "ml.model.created",
                "ml.run.created",
                "ml.predictions.created",
            }:
                self._refresh_predictions()
                self._refresh_session_views()
                self._update_action_gating()
                self._mark_analytics_dirty()
                self._refresh_analytics_if_visible()
                return

            if topic in {
                "plugin.enabled",
                "plugin.disabled",
                "ml.recipe.registered",
                "ml.recipe.unregistered",
            }:
                self.refresh(
                    include_analytics=False
                )
                return

            # Conservative fallback: keep widgets fresh, but do not rebuild
            # analytics unless the user is actually viewing Analytics.
            self.refresh(
                include_analytics=False
            )
            self._refresh_analytics_if_visible()

        try:
            curdoc = getattr(
                pn.state,
                "curdoc",
                None,
            )

            if curdoc is not None:
                curdoc.add_next_tick_callback(
                    apply
                )
            else:
                apply()

        except Exception:
            apply()

    def _analytics_refresh_clicked(
        self,
        *_: Any,
    ) -> None:
        self._mark_analytics_dirty()
        self._switch_results_tab(
            "Analytics"
        )
        self._maybe_refresh_analytics_views(
            force=True
        )
        self._set_status(
            "Analytics refreshed from AL/ML artifacts.",
            "info",
        )

    def _analytics_first_frame_clicked(
        self,
        *_: Any,
    ) -> None:
        self._analytics_frame_index = 0
        self._mark_analytics_dirty()
        self._switch_results_tab(
            "Analytics"
        )
        self._maybe_refresh_analytics_views(
            force=True
        )

    def _analytics_prev_frame_clicked(
        self,
        *_: Any,
    ) -> None:
        self._analytics_frame_index = max(
            0,
            int(
                self._analytics_frame_index
                or 0
            )
            - 1,
        )
        self._mark_analytics_dirty()
        self._switch_results_tab(
            "Analytics"
        )
        self._maybe_refresh_analytics_views(
            force=True
        )

    def _analytics_next_frame_clicked(
        self,
        *_: Any,
    ) -> None:
        frame_count = len(
            self._analytics_frames
            or []
        )

        if frame_count:
            self._analytics_frame_index = min(
                frame_count - 1,
                int(
                    self._analytics_frame_index
                    or 0
                )
                + 1,
            )

        self._mark_analytics_dirty()
        self._switch_results_tab(
            "Analytics"
        )
        self._maybe_refresh_analytics_views(
            force=True
        )

    def _analytics_last_frame_clicked(
        self,
        *_: Any,
    ) -> None:
        frame_count = len(
            self._analytics_frames
            or []
        )

        if frame_count:
            self._analytics_frame_index = (
                frame_count - 1
            )

        self._mark_analytics_dirty()
        self._switch_results_tab(
            "Analytics"
        )
        self._maybe_refresh_analytics_views(
            force=True
        )

    def _refresh_analytics_columns(self) -> None:
        session = self._load_current_session()
        dataset_id = self._session_pool_dataset_id(
            session
        )

        if not dataset_id:
            dataset_id = str(
                self.dataset_select.value
                or ""
            ).strip()

        columns = al_analytics.dataset_columns(
            self.context,
            dataset_id,
        )

        options = {
            column: column
            for column in columns
        }

        current_x = self.analytics_x_column.value
        current_y = self.analytics_y_column.value

        self.analytics_x_column.options = options
        self.analytics_y_column.options = options

        guessed_x, guessed_y = (
            al_analytics.guess_xy_columns(
                self.context,
                dataset_id,
            )
        )

        if current_x in options.values():
            self.analytics_x_column.value = (
                current_x
            )
        elif guessed_x in options.values():
            self.analytics_x_column.value = (
                guessed_x
            )
        elif columns:
            self.analytics_x_column.value = (
                columns[0]
            )
        else:
            self.analytics_x_column.value = None

        if current_y in options.values():
            self.analytics_y_column.value = (
                current_y
            )
        elif guessed_y in options.values():
            self.analytics_y_column.value = (
                guessed_y
            )
        elif len(columns) >= 2:
            self.analytics_y_column.value = (
                columns[1]
            )
        elif columns:
            self.analytics_y_column.value = (
                columns[0]
            )
        else:
            self.analytics_y_column.value = None

    def _update_analytics_nav_buttons(self) -> None:
        frame_count = len(
            self._analytics_frames
            or []
        )
        has_frames = frame_count > 0

        for button in (
            self.analytics_first_button,
            self.analytics_prev_button,
            self.analytics_next_button,
            self.analytics_last_button,
        ):
            try:
                button.disabled = not has_frames
            except Exception:
                pass

        if has_frames:
            is_first = int(
                self._analytics_frame_index
                or 0
            ) <= 0
            is_last = int(
                self._analytics_frame_index
                or 0
            ) >= frame_count - 1

            self.analytics_first_button.disabled = is_first
            self.analytics_prev_button.disabled = is_first
            self.analytics_next_button.disabled = is_last
            self.analytics_last_button.disabled = is_last

    def _analytics_summary_markdown(
        self,
        *,
        session: Mapping[str, Any],
        performance_df: pd.DataFrame,
        informativeness_df: pd.DataFrame,
        scatter_df: pd.DataFrame,
        frame: Mapping[str, Any],
    ) -> str:
        counts = al_state.counts(session)
        frame_count = len(
            self._analytics_frames
            or []
        )
        scored_count = 0
        unscored_count = 0

        if (
            scatter_df is not None
            and not scatter_df.empty
        ):
            try:
                scored_count = int(
                    scatter_df["has_score"].sum()
                )
                unscored_count = int(
                    (
                        ~scatter_df["has_score"]
                    ).sum()
                )
            except Exception:
                scored_count = 0
                unscored_count = len(
                    scatter_df
                )

        latest_perf = ""

        if (
            performance_df is not None
            and not performance_df.empty
        ):
            metric_rows = performance_df[
                pd.to_numeric(
                    performance_df.get(
                        "metric_value"
                    ),
                    errors="coerce",
                ).notna()
            ]

            if not metric_rows.empty:
                row = metric_rows.iloc[-1]
                latest_perf = (
                    f"`{row.get('metric_name')}` = "
                    f"**{row.get('metric_value'):.4g}** "
                    f"at {int(row.get('trained_count') or 0)} "
                    "trained images"
                )

        latest_query = ""

        if (
            informativeness_df is not None
            and not informativeness_df.empty
        ):
            row = informativeness_df.iloc[-1]

            latest_query = (
                f"batch {int(row.get('batch_index') or 0)} · "
                f"strategy `{row.get('strategy_id')}` · "
                f"mean={row.get('mean_informativeness'):.4g}"
                if pd.notna(
                    row.get(
                        "mean_informativeness"
                    )
                )
                else (
                    f"batch {int(row.get('batch_index') or 0)} · "
                    f"strategy `{row.get('strategy_id')}`"
                )
            )

        return (
            "#### Analytics source summary\n\n"
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| Session | `{session.get('session_id')}` |\n"
            f"| Pool dataset | "
            f"`{session.get('pool_dataset_id') or session.get('dataset_id')}` |\n"
            f"| Current AL round | `{session.get('round')}` |\n"
            f"| Verified training labels | "
            f"**{counts.get('labelled_or_verified', 0)}** |\n"
            f"| Query batch frames | **{frame_count}** |\n"
            f"| Points visible in scatter frame | "
            f"**{len(scatter_df) if scatter_df is not None else 0}** |\n"
            f"| Points with query value | **{scored_count}** |\n"
            f"| Points without query value | **{unscored_count}** |\n"
            f"| Latest performance | "
            f"{latest_perf or '_No numeric metric found yet._'} |\n"
            f"| Latest informativeness | "
            f"{latest_query or '_No query timeline yet._'} |\n"
        )

    def _current_label_options(self) -> List[str]:
        labels = al_state.parse_label_options(
            self.label_options_select.value
        )

        if labels:
            return labels

        labels = (
            self._labels_from_selected_target_column()
        )

        if labels:
            self.label_options_select.options = labels
            self.label_options_select.value = labels
            self._set_label_select_options()
            return labels

        labels = (
            self._labels_from_selected_recipe()
        )

        if labels:
            self.label_options_select.options = labels
            self.label_options_select.value = labels
            self._set_label_select_options()
            return labels

        labels = (
            self._labels_from_selected_predictions()
        )

        if labels:
            self.label_options_select.options = labels
            self.label_options_select.value = labels
            self._set_label_select_options()
            return labels

        return []

    def _labels_from_selected_target_column(
        self,
    ) -> List[str]:
        dataset_id = str(
            self.dataset_select.value
            or ""
        ).strip()
        column = str(
            self.target_column.value
            or ""
        ).strip()

        if (
            not dataset_id
            or not column
            or column == "al_label"
        ):
            return []

        values = self._unique_values_from_column(
            dataset_id,
            column,
            max_values=200,
        )

        return al_state.parse_label_options(
            values
        )

    def _unique_values_from_column(
        self,
        dataset_id: str,
        column: str,
        *,
        max_values: int = 200,
    ) -> List[str]:
        values: List[str] = []

        try:
            source = None
            get_source = getattr(
                self.context.datasets,
                "get_source",
                None,
            )

            if callable(get_source):
                source = get_source(
                    dataset_id
                )

            for method_name in (
                "unique_values",
                "distinct_values",
                "get_column_values",
            ):
                method = (
                    getattr(
                        source,
                        method_name,
                        None,
                    )
                    if source is not None
                    else None
                )

                if callable(method):
                    try:
                        raw = method(
                            column,
                            limit=max_values,
                        )
                    except TypeError:
                        raw = method(column)

                    values = [
                        str(item)
                        for item in raw
                        if (
                            item is not None
                            and str(item).strip()
                        )
                    ]

                    return list(
                        dict.fromkeys(values)
                    )[:max_values]

        except Exception:
            pass

        try:
            try:
                df = self.context.datasets.get_df(
                    dataset_id,
                    columns=[column],
                )
            except TypeError:
                df = self.context.datasets.get_df(
                    dataset_id
                )

            if column not in df.columns:
                return []

            series = df[column].dropna()

            for value in series.tolist():
                text = str(value).strip()

                if not text:
                    continue

                if text not in values:
                    values.append(text)

                if len(values) >= max_values:
                    break

        except Exception:
            return []

        return values

    def _restore_select_value(
        self,
        widget: Any,
        value: Any,
    ) -> None:
        if value is None:
            return

        try:
            options = widget.options
            valid_values = (
                set(options.values())
                if isinstance(
                    options,
                    dict,
                )
                else set(options)
            )

            if value in valid_values:
                widget.value = value

        except Exception:
            pass

    def _restore_widget_value(
        self,
        widget: Any,
        value: Any,
    ) -> None:
        try:
            if isinstance(
                widget,
                pn.widgets.Select,
            ):
                options = widget.options
                valid_values = (
                    set(options.values())
                    if isinstance(
                        options,
                        dict,
                    )
                    else set(options)
                )

                if value in valid_values:
                    widget.value = value
            else:
                widget.value = value

        except Exception:
            pass

    def _refresh_dataset_dependent_widgets(
        self,
    ) -> None:
        """Refresh controls affected by dataset registration or mapping changes."""

        if self._disposed:
            return

        previous_dataset = (
            self.dataset_select.value
        )
        previous_validation = (
            self.validation_dataset_select.value
        )
        previous_test = (
            self.test_dataset_select.value
        )
        previous_target = (
            self.target_column.value
        )
        previous_prediction = (
            self.predictions_select.value
        )

        previous_recipe_values: Dict[
            str,
            Any,
        ] = {}

        for name, widget in (
            self.recipe_param_widgets.items()
        ):
            try:
                previous_recipe_values[
                    str(name)
                ] = widget.value
            except Exception:
                pass

        previous_protocol_values: Dict[
            str,
            Any,
        ] = {}

        for name, widget in (
            self.protocol_widgets.items()
        ):
            try:
                previous_protocol_values[
                    str(name)
                ] = widget.value
            except Exception:
                pass

        self._refresh_datasets()
        self._refresh_validation_test_datasets()
        self._refresh_predictions()
        self._refresh_column_widgets()
        self._refresh_recipe_params()

        self._restore_select_value(
            self.dataset_select,
            previous_dataset,
        )
        self._restore_select_value(
            self.validation_dataset_select,
            previous_validation,
        )
        self._restore_select_value(
            self.test_dataset_select,
            previous_test,
        )
        self._restore_select_value(
            self.target_column,
            previous_target,
        )
        self._restore_select_value(
            self.predictions_select,
            previous_prediction,
        )

        for name, value in (
            previous_recipe_values.items()
        ):
            widget = (
                self.recipe_param_widgets.get(
                    name
                )
            )

            if widget is not None:
                self._restore_widget_value(
                    widget,
                    value,
                )

        for name, value in (
            previous_protocol_values.items()
        ):
            widget = (
                self.protocol_widgets.get(
                    name
                )
            )

            if widget is not None:
                self._restore_widget_value(
                    widget,
                    value,
                )

        self._apply_recipe_inferred_defaults()
        self._refresh_labels_from_available_context()
        self._set_label_select_options()
        self._refresh_review_label_for_focus()
        self._update_action_gating()

    def _refresh_recipes(self) -> None:
        current = self.recipe_select.value
        options: Dict[str, str] = {}
        registry = self._recipe_registry()

        if registry is not None:
            try:
                for spec in registry.list():
                    label = (
                        f"{getattr(spec, 'title', getattr(spec, 'id', 'recipe'))} "
                        f"({getattr(spec, 'id', '')})"
                    )
                    options[label] = str(
                        getattr(
                            spec,
                            "id",
                            "",
                        )
                    )
            except Exception:
                options = {}

        self.recipe_select.options = options

        if current in options.values():
            self.recipe_select.value = current
        elif options:
            self.recipe_select.value = (
                next(
                    iter(
                        options.values()
                    )
                )
            )
        else:
            self.recipe_select.value = None
            self.recipe_card.object = (
                "No core.ml recipes are registered. "
                "Enable core.ml or add a recipe."
            )

    def _recipe_registry(self) -> Any:
        try:
            return self.context.services.get(
                "core.ml.recipe_registry"
            )
        except Exception:
            return None

    def _selected_recipe_spec(self) -> Any:
        registry = self._recipe_registry()
        recipe_id = str(
            self.recipe_select.value
            or ""
        ).strip()

        if (
            registry is None
            or not recipe_id
        ):
            return None

        try:
            return registry.get(
                recipe_id
            )
        except Exception:
            return None

    def _form_row(
        self,
        label: str,
        widget: Any,
        *,
        height: int = 46,
    ):
        return self._field(
            label,
            widget,
        )

    def _field(
        self,
        label: str,
        widget: Any,
    ):
        # Use the widget's own native label instead of stacking a separate
        # Markdown pane above it.
        try:
            widget.name = label
        except Exception:
            pass

        self._fit_widget(widget)

        # Reserve vertical space with min_height rather than a fixed height.
        try:
            if isinstance(
                widget,
                pn.widgets.TextAreaInput,
            ):
                widget.min_height = 120
            elif isinstance(
                widget,
                pn.widgets.Checkbox,
            ):
                widget.min_height = 28
            else:
                widget.min_height = 54

            widget.margin = (
                0,
                0,
                12,
                0,
            )

        except Exception:
            pass

        return widget

    def _install_layout_css_once(self) -> None:
        global _AL_LAYOUT_CSS_INSTALLED

        if _AL_LAYOUT_CSS_INSTALLED:
            return

        css = """
        .al-panel-root,
        .al-panel-root * {
            box-sizing: border-box;
        }

        .al-panel-root {
            max-width: 480px;
            width: 100%;
            margin: 0 auto;
            overflow-x: hidden !important;
        }

        .al-controls-tabs,
        .al-results-tabs,
        .al-tab-body,
        .al-section,
        .al-field {
            max-width: 100%;
            width: 100%;
            box-sizing: border-box;
        }

        .al-tab-body {
            overflow: visible;
        }

        .al-panel-root button,
        .al-panel-root .bk-btn {
            min-height: 32px !important;
            height: auto !important;
            white-space: normal !important;
        }

        .al-panel-root .markdown,
        .al-panel-root .bk-HTML,
        .al-panel-root code {
            max-width: 100%;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        .al-panel-root .tabulator {
            max-width: 100%;
            width: 100%;
            overflow: auto;
        }

        .al-panel-root .bk-menu,
        .al-panel-root .choices__list--dropdown {
            z-index: 10000 !important;
        }

        .al-hint p {
            margin: 0 0 6px 0;
            font-size: 12px;
            color: #9a6a00;
        }
        """

        try:
            pn.extension(raw_css=[css])
            _AL_LAYOUT_CSS_INSTALLED = True
        except Exception:
            pass

    def _fit_widget(self, widget: Any) -> Any:
        try:
            widget.sizing_mode = "stretch_width"
        except Exception:
            pass

        try:
            styles = dict(getattr(widget, "styles", {}) or {})
            styles.update(
                {
                    "max-width": "100%",
                    "width": "100%",
                    "box-sizing": "border-box",
                }
            )
            widget.styles = styles
        except Exception:
            pass

        return widget

    def _fit_form_widget(self, widget: Any) -> Any:
        """Fit recipe/protocol widgets.

        We now rely on each widget's native label (set via ``_field``), so this
        must NOT blank out the name as an earlier version did.
        """
        return self._fit_widget(widget)

    def _normalise_widget_layout(self) -> None:
        button_names = {
            self.refresh_button: "Refresh",
            self.start_button: "Start / random sample",
            self.query_button: "Query top-k from predictions",
            self.label_button: "Record label / verification",
            self.unsure_button: "Mark current as Unsure",
            self.train_button: (
                "Add verified labels to training set + train from scratch"
            ),
        }

        widgets = [
            self.dataset_select,
            self.session_select,
            self.predictions_select,
            self.strategy_select,
            self.recipe_select,
            self.label_options_select,
            self.target_column,
            self.validation_dataset_select,
            self.test_dataset_select,
            self.al_protocol,
            self.seed,
            self.initial_k,
            self.query_k,
            self.label_select,
            self.advance_after_label,
            self.refresh_button,
            self.start_button,
            self.query_button,
            self.label_button,
            self.unsure_button,
            self.train_button,
        ]

        for widget in widgets:
            self._fit_widget(widget)

        for button, name in button_names.items():
            try:
                button.name = name
                button.min_width = 160
                button.align = "start"
            except Exception:
                pass

        for pane in (
            self.status,
            self.summary,
            self.recipe_card,
        ):
            try:
                pane.sizing_mode = "stretch_width"
                pane.styles = {
                    "max-width": "100%",
                    "width": "100%",
                    "overflow-wrap": "anywhere",
                    "word-break": "break-word",
                    "box-sizing": "border-box",
                }
            except Exception:
                pass

        for hint in (
            self.query_hint,
            self.train_hint,
        ):
            try:
                hint.sizing_mode = "stretch_width"
                hint.css_classes = ["al-hint"]
                hint.styles = {
                    "max-width": "100%",
                    "width": "100%",
                    "box-sizing": "border-box",
                }
            except Exception:
                pass

        try:
            self.recipe_params_area.sizing_mode = "stretch_width"
            self.recipe_params_area.styles = {
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "visible",
                "clear": "both",
            }
        except Exception:
            pass

    def _tab_body(
        self,
        body: Any,
        *,
        height: int = 430,
    ):
        # No fixed height / inner scroll: the whole panel scrolls as one
        # region, so a button at the bottom of a tab is always reachable.
        return pn.Column(
            body,
            sizing_mode="stretch_width",
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "visible",
                "padding": "8px 6px 12px 6px",
            },
            css_classes=["al-tab-body"],
        )

    def _start_field(
        self,
        obj: Any,
        *,
        min_height: int = 70,
        margin: tuple[int, int, int, int] = (0, 0, 16, 0),
    ) -> pn.Column:
        """Wrap a Start-tab widget so Panel reserves enough vertical space."""

        return pn.Column(
            obj,
            sizing_mode="stretch_width",
            min_height=min_height,
            height_policy="min",
            margin=margin,
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "visible",
                "clear": "both",
                "padding": "0",
            },
        )

    def _start_button_row(
        self,
        *objects: Any,
    ) -> pn.Column:
        row = pn.Row(
            *objects,
            sizing_mode="stretch_width",
            min_height=42,
            margin=(0, 0, 0, 0),
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "visible",
                "clear": "both",
            },
        )

        return self._start_field(
            row,
            min_height=50,
            margin=(0, 0, 16, 0),
        )

    def _section(
        self,
        title: str,
        body: Any,
    ):
        return pn.Column(
            pn.pane.Markdown(
                f"**{title}**",
                sizing_mode="stretch_width",
                margin=(0, 0, 6, 0),
                styles={
                    "font-size": "12px",
                    "line-height": "1.2",
                    "font-weight": "600",
                    "overflow-wrap": "anywhere",
                },
            ),
            body,
            sizing_mode="stretch_width",
            styles={
                "max-width": "100%",
                "box-sizing": "border-box",
                "overflow": "visible",
                "padding-bottom": "4px",
            },
            css_classes=["al-section"],
        )

    def _protocol_params(
        self,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}

        for key, widget in self.protocol_widgets.items():
            try:
                params[key] = widget.value
            except Exception:
                pass

        validation_dataset_id = (
            self.validation_dataset_select.value
            or ""
        )

        test_dataset_id = (
            self.test_dataset_select.value
            or ""
        )

        validation_fraction = float(
            self.validation_fraction.value
            or 0.0
        )

        test_fraction = float(
            self.test_fraction.value
            or 0.0
        )

        if validation_dataset_id:
            params[
                "protocol_validation_source"
            ] = "dataset"

            params[
                "protocol_validation_dataset_id"
            ] = validation_dataset_id
        else:
            params["protocol_validation_source"] = "split"
            params["protocol_validation_dataset_id"] = ""
            params["protocol_validation_size"] = validation_fraction

        if test_dataset_id:
            params["protocol_test_source"] = "dataset"
            params["protocol_test_dataset_id"] = test_dataset_id
        elif test_fraction > 0:
            params["protocol_test_source"] = "split"
            params["protocol_test_dataset_id"] = ""
            params["protocol_test_size"] = test_fraction
        else:
            params["protocol_test_source"] = "none"
            params["protocol_test_dataset_id"] = ""
            params["protocol_test_size"] = 0.0

        params["validation_dataset_id"] = validation_dataset_id
        params["test_dataset_id"] = test_dataset_id
        params["al_protocol"] = self.al_protocol.value or "review"

        return params

    def _refresh_recipe_params(self) -> None:
        previous_values: Dict[str, Any] = {}
        for name, widget in self.recipe_param_widgets.items():
            try:
                previous_values[str(name)] = widget.value
            except Exception:
                pass

        previous_protocol_values: Dict[str, Any] = {}
        for name, widget in self.protocol_widgets.items():
            try:
                previous_protocol_values[str(name)] = widget.value
            except Exception:
                pass

        spec = self._selected_recipe_spec()
        self.recipe_param_widgets = {}
        self.recipe_param_fields = {}
        self.protocol_widgets = {}
        self.protocol_fields = {}

        if spec is None:
            self.recipe_card.object = "No recipe selected."
            self.recipe_params_area.objects = []
            self._refresh_training_contract_summary()
            return

        recipe_cls = getattr(spec, "recipe_cls", None)
        execution_mode = (
            getattr(recipe_cls, "execution_mode", None)
            or getattr(spec, "execution_mode", "freeform")
        )
        managed = str(execution_mode) == "managed"

        required_mappings = getattr(spec, "required_mappings", []) or []

        self.recipe_card.object = (
            f"`{getattr(spec, 'id', '')}` v{getattr(spec, 'version', '')}\n\n"
            f"{getattr(spec, 'description', '')}\n\n"
            f"- Task: `{getattr(spec, 'task', '')}`\n"
            f"- Modality: `{getattr(spec, 'modality', '')}`\n"
            f"- Mode: `{'managed' if managed else 'freeform'}`\n"
            f"- Required mappings: `{', '.join(required_mappings) or 'none'}`"
        )

        properties = (
            (getattr(spec, "params_schema", {}) or {})
            .get("properties", {})
            or {}
        )

        new_objects: List[Any] = []
        skipped_owned: List[str] = []

        if properties:
            for name, schema in properties.items():
                name = str(name)

                # Start tab owns these. Do not ask again in Train.
                if name.lower() in AL_OWNED_RECIPE_PARAM_NAMES:
                    skipped_owned.append(name)
                    continue

                widget = self._widget_for_schema(name, schema)

                if name in previous_values:
                    try:
                        widget.value = previous_values[name]
                    except Exception:
                        pass

                field = self._form_row(
                    str(schema.get("title") or name),
                    widget,
                )

                self.recipe_param_widgets[name] = widget
                self.recipe_param_fields[name] = field
                new_objects.append(field)

        else:
            new_objects.append(
                pn.pane.Markdown(
                    "This recipe exposes no extra parameters.",
                    sizing_mode="stretch_width",
                    styles={
                        "overflow-wrap": "anywhere",
                    },
                )
            )

        if skipped_owned:
            new_objects.insert(
                0,
                pn.pane.Markdown(
                    (
                        "_Dataset, recipe, target, image/mask/feature inputs, labels, "
                        "validation, and test settings are controlled by the "
                        "Start tab and passed into the recipe automatically._"
                    ),
                    sizing_mode="stretch_width",
                    styles={
                        "font-size": "12px",
                        "color": "#666",
                        "overflow-wrap": "anywhere",
                    },
                ),
            )

        protocol_block = self._build_protocol_section(
            managed=managed
        )

        for name, value in previous_protocol_values.items():
            widget = self.protocol_widgets.get(name)

            if widget is not None:
                try:
                    widget.value = value
                except Exception:
                    pass

        if protocol_block is not None:
            new_objects.append(protocol_block)

        self.recipe_params_area.objects = new_objects

        est = 0

        for widget in self.recipe_param_widgets.values():
            try:
                est += (
                    int(
                        getattr(
                            widget,
                            "min_height",
                            54,
                        )
                        or 54
                    )
                    + 14
                )
            except Exception:
                est += 68

        if managed:
            est += 90

        try:
            self.recipe_params_area.min_height = max(
                est,
                60,
            )
        except Exception:
            pass

        self._refresh_training_contract_summary()

    def _refresh_validation_test_datasets(self) -> None:
        dataset_options = {
            "Split from AL training rows": "",
        }

        dataset_options.update(
            {
                dataset_id: dataset_id
                for dataset_id in self._dataset_ids()
            }
        )

        for widget in (
            self.validation_dataset_select,
            self.test_dataset_select,
        ):
            current = widget.value
            widget.options = dataset_options
            widget.value = (
                current
                if current in dataset_options.values()
                else ""
            )

        self._refresh_evaluation_widget_visibility()

    def _refresh_labels_from_available_context(self) -> None:
        current = al_state.parse_label_options(
            self.label_options_select.value
        )

        labels: List[str] = []
        labels.extend(
            self._labels_from_selected_target_column()
        )
        labels.extend(
            self._labels_from_selected_recipe()
        )
        labels.extend(
            self._labels_from_selected_predictions()
        )
        labels.extend(current)

        labels = al_state.parse_label_options(
            labels
        )

        self.label_options_select.options = labels

        if current:
            self.label_options_select.value = [
                label
                for label in current
                if label in labels
            ]
        else:
            self.label_options_select.value = labels

        self._set_label_select_options()

    def _labels_from_selected_recipe(self) -> List[str]:
        spec = self._selected_recipe_spec()

        if spec is None:
            return []

        labels: List[str] = []
        schema = getattr(
            spec,
            "params_schema",
            {},
        ) or {}

        properties = (
            schema.get("properties", {})
            if isinstance(schema, Mapping)
            else {}
        )

        for key in (
            "labels",
            "class_labels",
            "classes",
            "label_options",
        ):
            prop = (
                properties.get(key, {})
                if isinstance(properties, Mapping)
                else {}
            )

            enum = (
                prop.get("enum")
                if isinstance(prop, Mapping)
                else None
            )

            default = (
                prop.get("default")
                if isinstance(prop, Mapping)
                else None
            )

            if enum:
                labels.extend(
                    str(item)
                    for item in enum
                )

            if isinstance(default, list):
                labels.extend(
                    str(item)
                    for item in default
                )

        for attr in (
            "labels",
            "class_labels",
            "classes",
        ):
            value = (
                getattr(spec, attr, None)
                or getattr(
                    getattr(spec, "recipe_cls", None),
                    attr,
                    None,
                )
            )

            if isinstance(
                value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                labels.extend(
                    str(item)
                    for item in value
                )

        return al_state.parse_label_options(
            labels
        )

    def _labels_from_selected_predictions(self) -> List[str]:
        artifact_id = str(
            self.predictions_select.value
            or ""
        ).strip()

        if not artifact_id:
            return []

        try:
            payload = self.context.artifacts.get(
                artifact_id
            )

            return actions._infer_label_options(
                payload
            )

        except Exception:
            return []

    def _apply_recipe_inferred_defaults(self) -> None:
        spec = self._selected_recipe_spec()
        dataset_id = self.dataset_select.value

        if spec is None or not dataset_id:
            self._refresh_input_contract_ui()
            return

        inferred: Dict[str, Any] = {}
        registry = self._recipe_registry()
        infer = getattr(
            registry,
            "infer_recipe_params",
            None,
        )

        if callable(infer):
            try:
                inferred = dict(
                    infer(
                        self.context,
                        dataset_id,
                        spec,
                    )
                    or {}
                )
            except Exception:
                inferred = {}

        for name, value in inferred.items():
            widget = self.recipe_param_widgets.get(
                str(name)
            )

            if widget is None:
                continue

            try:
                if widget.value in (
                    None,
                    "",
                    [],
                    {},
                ):
                    widget.value = value
            except Exception:
                pass

        contract = self._selected_recipe_input_contract()

        binding = al_contracts.resolve_dataset_role_binding(
            self.context,
            role="pool",
            dataset_id=str(dataset_id),
            recipe_contract=contract,
            explicit={
                "target_column": self.target_column.value,
                "feature_columns": self._selected_feature_columns(),
                "image_column": self.image_column.value,
                "mask_column": self.mask_column.value,
            },
        )

        inferred_image = (
            inferred.get("image_column")
            or inferred.get("image_path_column")
            or inferred.get("image_uri_column")
            or binding.image_column
        )

        inferred_mask = (
            inferred.get("mask_column")
            or inferred.get("mask_path_column")
            or binding.mask_column
        )

        inferred_features = (
            inferred.get("feature_columns")
            or inferred.get("input_columns")
            or inferred.get("features")
            or binding.feature_columns
        )

        if (
            contract.needs_image
            and not self.image_column.value
            and inferred_image
        ):
            self._restore_select_value(
                self.image_column,
                str(inferred_image),
            )

        if (
            contract.needs_mask
            and not self.mask_column.value
            and inferred_mask
        ):
            self._restore_select_value(
                self.mask_column,
                str(inferred_mask),
            )

        if (
            contract.needs_features
            and not self._selected_feature_columns()
            and inferred_features
        ):
            self._set_feature_columns_value(
                self._normalise_column_list(
                    inferred_features
                )
            )

        self._refresh_input_contract_ui()
        self._refresh_data_contract_summary()

    def _selected_recipe_input_contract(
        self,
    ) -> al_contracts.RecipeInputContract:
        return (
            al_contracts.RecipeInputContract
            .from_recipe_spec(
                self._selected_recipe_spec()
            )
        )

    def _refresh_binding_widgets(self) -> None:
        columns = self._dataset_columns(
            self.dataset_select.value
        )

        options = {
            "Auto-detect from mappings / recipe": "",
        }

        options.update(
            {
                column: column
                for column in columns
            }
        )

        for widget in (
            self.image_column,
            self.mask_column,
        ):
            current = str(
                widget.value
                or ""
            )

            widget.options = options
            widget.value = (
                current
                if current in options.values()
                else ""
            )

    def _refresh_input_contract_ui(self) -> None:
        contract = self._selected_recipe_input_contract()

        for field_name, visible in (
            (
                "image_column_field",
                contract.needs_image,
            ),
            (
                "mask_column_field",
                contract.needs_mask,
            ),
            (
                "feature_selector_field",
                contract.needs_features,
            ),
        ):
            field = getattr(
                self,
                field_name,
                None,
            )

            if field is not None:
                try:
                    field.visible = bool(
                        visible
                    )
                except Exception:
                    pass

        self._refresh_evaluation_widget_visibility()

    def _refresh_evaluation_widget_visibility(
        self,
    ) -> None:
        contract = self._selected_recipe_input_contract()

        validation_external = bool(
            self.validation_dataset_select.value
        )

        test_external = bool(
            self.test_dataset_select.value
        )

        field = getattr(
            self,
            "validation_fraction_field",
            None,
        )

        if field is not None:
            field.visible = bool(
                not validation_external
                and contract.supports_internal_validation_split
            )

        field = getattr(
            self,
            "test_fraction_field",
            None,
        )

        if field is not None:
            field.visible = bool(
                not test_external
                and contract.supports_internal_test_split
            )

    def _profile_request_params(
        self,
        *,
        inspect_images: bool,
    ) -> Dict[str, Any]:
        session = (
            self._load_current_session()
            or {}
        )

        return {
            "dataset_id": (
                self.dataset_select.value
                or ""
            ),
            "recipe_id": (
                self.recipe_select.value
                or ""
            ),
            "validation_dataset_id": (
                self.validation_dataset_select.value
                or ""
            ),
            "test_dataset_id": (
                self.test_dataset_select.value
                or ""
            ),
            "target_column": (
                self.target_column.value
                or "al_label"
            ),
            "feature_columns": (
                self._selected_feature_columns()
            ),
            "image_column": (
                self.image_column.value
                or ""
            ),
            "mask_column": (
                self.mask_column.value
                or ""
            ),
            "recipe_params": (
                self._recipe_params()
            ),
            "protocol_params": (
                self._protocol_params()
            ),
            "labelled_count": (
                self._labelled_count(session)
            ),
            "inspect_images": bool(
                inspect_images
            ),
            "image_sample_size": 8,
        }

    def _build_session_contract(
        self,
        *,
        inspect_images: bool = False,
        labelled_count: Optional[int] = None,
    ) -> al_contracts.PreflightReport:
        spec = self._selected_recipe_spec()

        if spec is None:
            return al_contracts.PreflightReport(
                contract={},
                errors=[
                    "Choose a core.ml recipe.",
                ],
            )

        if labelled_count is None:
            labelled_count = (
                self._labelled_count(
                    self._load_current_session()
                )
            )
        report = al_contracts.build_session_contract(
            self.context,
            recipe_spec=spec,
            pool_dataset_id=str(
                self.dataset_select.value
                or ""
            ),
            validation_dataset_id=str(
                self.validation_dataset_select.value
                or ""
            ),
            test_dataset_id=str(
                self.test_dataset_select.value
                or ""
            ),
            target_column=str(
                self.target_column.value
                or "al_label"
            ),
            feature_columns=(
                self._selected_feature_columns()
            ),
            image_column=(
                str(
                    self.image_column.value
                    or ""
                )
                or None
            ),
            mask_column=(
                str(
                    self.mask_column.value
                    or ""
                )
                or None
            ),
            recipe_params=(
                self._recipe_params()
            ),
            protocol_params=(
                self._protocol_params()
            ),
            labelled_count=int(
                labelled_count
                or 0
            ),
            label_counts=self._label_counts_from_session(
                self._load_current_session()
            ),
            inspect_images=inspect_images,
            include_column_counts=False,
        )

        # Preserve a previously inspected image sample while rebuilding the
        # lightweight contract, but only when the dataset and resolved image
        # binding still match.
        if (
            not inspect_images
            and self._latest_session_contract
        ):
            old_profiles = dict(
                self._latest_session_contract.get(
                    "profiles"
                )
                or {}
            )

            old_bindings = dict(
                self._latest_session_contract.get(
                    "bindings"
                )
                or {}
            )

            new_profiles = dict(
                report.contract.get(
                    "profiles"
                )
                or {}
            )

            new_bindings = dict(
                report.contract.get(
                    "bindings"
                )
                or {}
            )

            for role, new_profile in (
                new_profiles.items()
            ):
                old_profile = dict(
                    old_profiles.get(role)
                    or {}
                )

                old_binding = dict(
                    old_bindings.get(role)
                    or {}
                )

                new_binding = dict(
                    new_bindings.get(role)
                    or {}
                )

                if (
                    old_profile.get(
                        "image_profile"
                    )
                    and old_binding.get(
                        "dataset_id"
                    )
                    == new_binding.get(
                        "dataset_id"
                    )
                    and old_binding.get(
                        "image_column"
                    )
                    == new_binding.get(
                        "image_column"
                    )
                ):
                    new_profile[
                        "image_profile"
                    ] = old_profile.get(
                        "image_profile"
                    )

                    if old_profile.get(
                        "non_null"
                    ):
                        new_profile[
                            "non_null"
                        ] = old_profile.get(
                            "non_null"
                        )

            report.contract[
                "profiles"
            ] = new_profiles

        return report

    def _remember_preflight(
        self,
        report: al_contracts.PreflightReport,
    ) -> None:
        self._latest_session_contract = dict(
            report.contract
            or {}
        )

        self._latest_preflight_errors = list(
            report.errors
            or []
        )

        self._latest_preflight_warnings = list(
            report.warnings
            or []
        )

        self.data_contract_summary.object = (
            self._contract_markdown(
                self._latest_session_contract,
                errors=(
                    self._latest_preflight_errors
                ),
                warnings=(
                    self._latest_preflight_warnings
                ),
            )
        )

        self._refresh_training_contract_summary()

    def _refresh_data_contract_summary(
        self,
    ) -> None:
        if not hasattr(
            self,
            "data_contract_summary",
        ):
            return

        if (
            not self.dataset_select.value
            or self._selected_recipe_spec()
            is None
        ):
            self.data_contract_summary.object = (
                "_Choose a pool dataset and recipe "
                "to resolve image/tabular inputs._"
            )
            return

        try:
            self._remember_preflight(
                self._build_session_contract(
                    inspect_images=False
                )
            )

        except Exception as exc:
            self.data_contract_summary.object = (
                "**Could not resolve data contract:** "
                f"{exc}"
            )

    def _contract_markdown(
        self,
        contract: Mapping[str, Any],
        *,
        errors: Sequence[str],
        warnings: Sequence[str],
    ) -> str:
        recipe = dict(
            contract.get("recipe")
            or {}
        )

        bindings = dict(
            contract.get("bindings")
            or {}
        )

        profiles = dict(
            contract.get("profiles")
            or {}
        )

        split = dict(
            contract.get("split_estimates")
            or {}
        )

        recipe_params = dict(
            recipe.get("params")
            or {}
        )

        transform_parts: List[str] = []

        for key in (
            "input_size",
            "image_size",
            "crop_size",
            "resize",
        ):
            value = recipe_params.get(key)

            if value not in (
                None,
                "",
                [],
                {},
            ):
                transform_parts.append(
                    f"{key}=`{value}`"
                )

        lines = [
            "### Recipe-driven data contract",
            "",
            (
                f"Recipe: `{recipe.get('id') or ''}` · "
                f"task `{recipe.get('task') or ''}` · "
                f"modality `{recipe.get('modality') or ''}`"
            ),
            *(
                [
                    "Recipe image input/transform: "
                    f"{' · '.join(transform_parts)}"
                ]
                if transform_parts
                else []
            ),
            "",
            "| Role | Dataset | Shape | Inputs | Target |",
            "|---|---|---:|---|---|",
        ]

        for role in (
            "pool",
            "validation",
            "test",
        ):
            binding = dict(
                bindings.get(role)
                or {}
            )

            if not binding:
                continue

            profile = dict(
                profiles.get(role)
                or {}
            )

            shape = (
                profile.get("shape")
                or [
                    None,
                    None,
                ]
            )

            shape_text = (
                f"{shape[0]:,} × {shape[1]:,}"
                if (
                    len(shape) >= 2
                    and shape[0] is not None
                )
                else "unknown"
            )

            input_bits: List[str] = []

            if binding.get("image_column"):
                input_bits.append(
                    "image "
                    f"`{binding.get('image_column')}`"
                )

            if binding.get("mask_column"):
                input_bits.append(
                    "mask "
                    f"`{binding.get('mask_column')}`"
                )

            features = list(
                binding.get("feature_columns")
                or []
            )

            if features:
                preview = ", ".join(
                    f"`{value}`"
                    for value in features[:4]
                )

                if len(features) > 4:
                    preview += (
                        f", +{len(features) - 4} more"
                    )

                input_bits.append(
                    f"{len(features)} features: "
                    f"{preview}"
                )

            if not input_bits:
                input_bits.append(
                    "_unresolved_"
                )

            lines.append(
                f"| {role.title()} | "
                f"`{binding.get('dataset_id') or ''}` | "
                f"{shape_text} | "
                f"{'<br>'.join(input_bits)} | "
                f"`{binding.get('target_column') or ''}` |"
            )

            image_profile = profile.get("image_profile")
            if isinstance(image_profile, Mapping):
                sizes = dict(image_profile.get("sizes") or {})
                modes = dict(image_profile.get("modes") or {})
                formats = dict(image_profile.get("formats") or {})
                lines.extend(
                    [
                        "",
                        f"**{role.title()} image sample:** "
                        f"{image_profile.get('readable', 0)} readable, "
                        f"{image_profile.get('unreadable', 0)} unreadable, "
                        f"{image_profile.get('remote_skipped', 0)} remote URI(s) skipped.",
                        f"Sizes: `{sizes or 'not available'}` · modes: `{modes or 'not available'}` · formats: `{formats or 'not available'}`",
                    ]
                )

        if split.get("labelled_total"):
            lines.extend(
                [
                    "",
                    "**Estimated labelled split:** "
                    f"train `{split.get('train')}`, validation `{split.get('validation')}`, "
                    f"test `{split.get('test')}` from `{split.get('labelled_total')}` verified rows.",
                ]
            )

        if errors:
            lines.extend(["", "**Blocking issues**", *[f"- {value}" for value in errors]])
        if warnings:
            lines.extend(["", "**Warnings**", *[f"- {value}" for value in warnings]])
        if not errors and not warnings:
            lines.extend(["", "_Data bindings are compatible with the selected recipe._"])
        return "\n".join(lines)

    def _dataset_ids(self) -> List[str]:
        try:
            return list(self.context.datasets.list_ids())
        except Exception:
            return []

    def _dataset_columns(self, dataset_id: Any) -> List[str]:
        dataset_id = str(dataset_id or "").strip()
        if not dataset_id:
            return []
        try:
            return [str(col) for col in self.context.datasets.list_columns(dataset_id)]
        except Exception:
            try:
                return [str(col) for col in self.context.datasets.get_df(dataset_id).columns]
            except Exception:
                return []

    def _normalise_column_list(self, value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []

            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    value = parsed
                elif isinstance(parsed, str):
                    value = [parsed]
                else:
                    value = [text]
            except Exception:
                value = [
                    part.strip()
                    for chunk in text.splitlines()
                    for part in chunk.split(",")
                    if part.strip()
                ]

        elif isinstance(value, Mapping):
            value = value.keys()

        elif isinstance(value, (int, float)):
            value = [value]

        columns: List[str] = []
        for item in value or []:
            if item is None:
                continue
            text = str(item).strip()
            if not text or text == "Use Index":
                continue
            if text not in columns:
                columns.append(text)

        return columns

    def _selected_feature_columns(self) -> List[str]:
        """Return selected feature columns from the editable text area."""
        if not hasattr(self, "feature_columns"):
            return []
        return self._normalise_column_list(self.feature_columns.value)

    def _set_feature_columns_value(self, columns: Iterable[Any]) -> None:
        """Set selected feature columns into the editable text area.

        feature_columns is now a TextAreaInput, not a CrossSelector, so never
        use .options here.
        """
        valid = set(self._dataset_columns(self.dataset_select.value))
        clean: List[str] = []

        for value in columns or []:
            if value is None:
                continue

            column = str(value).strip()
            if not column or column == "Use Index":
                continue

            if valid and column not in valid:
                continue

            if column not in clean:
                clean.append(column)
        self.feature_columns.value = "\n".join(clean)
        self._refresh_feature_match_preview()
        self._refresh_training_contract_summary()

    def _feature_select_numeric_clicked(self, *_: Any) -> None:
        """Bulk-select numeric dataset columns.

        This replaces the older CrossSelector implementation that used
        self.feature_columns.options.
        """
        numeric = self._numeric_feature_columns()
        self._set_feature_columns_value(numeric)

    def _feature_clear_clicked(self, *_: Any) -> None:
        self._set_feature_columns_value([])

    def _feature_add_matching_clicked(self, *_: Any) -> None:
        current = self._selected_feature_columns()
        matches = self._feature_filter_matches()
        self._set_feature_columns_value([*current, *matches])

    def _feature_replace_matching_clicked(self, *_: Any) -> None:
        matches = self._feature_filter_matches()
        self._set_feature_columns_value(matches)

    def _row_id_match_keys(self, value: Any) -> set[str]:
        """Build tolerant match keys for exact int64 and scientific-notation ids.

        This is a defensive fallback for UI paths that display large object IDs
        as floats, e.g. -5.6309557250480134e+17. Exact strings are always used
        first; float keys are only fallback aliases.
        """
        if value is None:
            return set()

        try:
            if pd.isna(value):
                return set()
        except Exception:
            pass

        text = str(value).strip()
        if not text:
            return set()

        keys = {text}

        # Strip a harmless trailing .0 for small integer-like values.
        if text.endswith(".0"):
            keys.add(text[:-2])

        try:
            as_float = float(text)
            if not pd.isna(as_float):
                keys.add(str(as_float))
                keys.add(repr(as_float))
                keys.add(format(as_float, ".17g"))
                keys.add(format(as_float, ".16g"))
                keys.add(format(as_float, ".15g"))
                if as_float.is_integer():
                    keys.add(str(int(as_float)))
        except Exception:
            pass

        try:
            as_int = int(text)
            keys.add(str(as_int))
            as_float = float(as_int)
            keys.add(str(as_float))
            keys.add(repr(as_float))
            keys.add(format(as_float, ".17g"))
            keys.add(format(as_float, ".16g"))
            keys.add(format(as_float, ".15g"))
        except Exception:
            pass

        return {key for key in keys if key}

    def _requested_row_id_key_map(self, row_ids: List[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for row_id in row_ids:
            original = str(row_id)
            for key in self._row_id_match_keys(original):
                out.setdefault(key, original)
        return out

    def _row_id_series_mask(
        self,
        series: pd.Series,
        requested_key_map: Mapping[str, str],
    ) -> pd.Series:
        if series is None or series.empty or not requested_key_map:
            return pd.Series(False, index=getattr(series, "index", []))

        requested_keys = set(str(key) for key in requested_key_map.keys())

        try:
            exact = series.astype(str).isin(requested_keys)
        except Exception:
            exact = pd.Series(False, index=series.index)

        # Fallback for large int64 ids that have been converted to scientific
        # notation somewhere in UI/JS. This is slower, so only use it if exact
        # matching did not find everything.
        try:
            if int(exact.sum()) >= len(set(requested_key_map.values())):
                return exact
        except Exception:
            pass

        try:
            fuzzy = series.map(
                lambda value: bool(self._row_id_match_keys(value) & requested_keys)
            )
            return exact | fuzzy
        except Exception:
            return exact

    def _auto_label_column_should_follow_target(self) -> bool:
        if not bool(getattr(self, "_auto_label_source_user_set", False)):

            return True

        current = str(self.auto_label_source_column.value or "").strip()
        return not current

    def _set_auto_label_source_value(self, value: Any) -> None:
        self._syncing_auto_label_source = True
        try:
            self.auto_label_source_column.value = value
        finally:
            self._syncing_auto_label_source = False

    def _feature_filter_tokens(self) -> List[str]:
        text = str(self.feature_filter.value or "").strip()
        if not text:
            return []

        tokens: List[str] = []
        for chunk in text.splitlines():
            for part in chunk.split(","):
                token = part.strip()
                if token and token not in tokens:
                    tokens.append(token)

        return tokens

    def _feature_filter_matches(self) -> List[str]:
        columns = self._dataset_columns(self.dataset_select.value)
        tokens = self._feature_filter_tokens()

        target = str(self.target_column.value or "").strip()
        excluded = {
            "",
            "Use Index",
            target,
            "al_label",
            "label",
            "labels",
            "target",
            "target_label",
            "class",
            "class_label",
            "prediction",
            "predicted_label",
            "prediction_confidence",
            "entropy",
            "least_confidence",
            "margin",
            "margin_uncertainty",
            "selection_rank",
            "rank",
        }

        if not tokens:
            return []

        matches: List[str] = []

        for column in columns:
            if column in excluded:
                continue

            lower_column = str(column).lower()

            for token in tokens:
                lower_token = token.lower()

                # Useful cases:
                #   flux_      -> all columns containing/starting flux_
                #   flux_*     -> prefix match
                #   *flux*     -> contains match
                #   flux       -> contains match
                if lower_token.endswith("*") and not lower_token.startswith("*"):
                    ok = lower_column.startswith(lower_token[:-1])
                elif lower_token.startswith("*") and lower_token.endswith("*") and len(lower_token) > 2:
                    ok = lower_token[1:-1] in lower_column
                elif lower_token.startswith("*"):
                    ok = lower_column.endswith(lower_token[1:])
                else:
                    ok = lower_token in lower_column

                if ok:
                    matches.append(str(column))
                    break

        return list(dict.fromkeys(matches))

    def _refresh_feature_match_preview(self) -> None:
        if not hasattr(self, "feature_match_preview"):
            return

        selected = self._selected_feature_columns()
        matches = self._feature_filter_matches()
        tokens = self._feature_filter_tokens()

        if not tokens:
            self.feature_match_preview.object = (
                f"Selected {len(selected)} feature column(s). "
                "Type a prefix or substring above, then use Add/Replace matches."
            )
            return

        if matches:
            preview = ", ".join(str(column) for column in matches[:6])
            if len(matches) > 6:
                preview += f", … +{len(matches) - 6} more"
        else:
            preview = "no matching columns"

        self.feature_match_preview.object = (
            f"Filter matches {len(matches)} column(s). "
            f"Selected {len(selected)}. Preview: {preview}"
        )

    def _feature_columns_from_recipe_params(self, params: Mapping[str, Any]) -> List[str]:
        params = dict(params or {})
        for key in ("feature_columns", "input_columns", "features", "x_columns"):
            columns = self._normalise_column_list(params.get(key))

            if columns:
                return columns
        return []

    def _start_owned_recipe_params(self) -> Dict[str, Any]:
        feature_columns = self._selected_feature_columns()
        target_column = self.target_column.value or "al_label"
        validation_dataset_id = self.validation_dataset_select.value or ""
        test_dataset_id = self.test_dataset_select.value or ""

        params: Dict[str, Any] = {
            "dataset_id": self.dataset_select.value or "",
            "recipe_id": self.recipe_select.value or "",
            "target_column": target_column,
            "label_column": target_column,
            "label_options": self._current_label_options(),
            "class_labels": self._current_label_options(),
            "classes": self._current_label_options(),
            "known_classes": self._current_label_options(),
            "target_classes": self._current_label_options(),
            "validation_dataset_id": validation_dataset_id,
            "test_dataset_id": test_dataset_id,
            "al_protocol": self.al_protocol.value or "review",
            "seed": int(self.seed.value or 0),
            "random_seed": int(self.seed.value or 0),
            "protocol_random_state": int(self.seed.value or 0),
        }

        input_contract = self._selected_recipe_input_contract()

        if input_contract.needs_features and feature_columns:
            params["feature_columns"] = feature_columns
            params["input_columns"] = feature_columns
            params["features"] = feature_columns
            params["x_columns"] = feature_columns

        image_column = str(self.image_column.value or "").strip()
        if input_contract.needs_image and image_column:
            params["image_column"] = image_column
            params["image_path_column"] = image_column
            params["image_uri_column"] = image_column

        mask_column = str(self.mask_column.value or "").strip()
        if input_contract.needs_mask and mask_column:
            params["mask_column"] = mask_column
            params["mask_path_column"] = mask_column

        params.update(self._protocol_params())
        return params

    def _refresh_feature_column_options(self) -> None:
        """Keep selected feature text valid for the current dataset/target."""
        columns = set(self._dataset_columns(self.dataset_select.value))
        target = str(self.target_column.value or "").strip()

        current = []
        for column in self._selected_feature_columns():
            if column not in columns:
                continue
            if target and column == target:
                continue
            current.append(column)

        self._set_feature_columns_value(current)
        self._refresh_feature_match_preview()

    def _numeric_feature_columns(self) -> List[str]:
        dataset_id = self.dataset_select.value
        columns = self._dataset_columns(dataset_id)

        target = str(self.target_column.value or "").strip()
        excluded = {
            "",
            "Use Index",
            target,
            "al_label",
            "label",
            "labels",
            "target",
            "target_label",
            "class",
            "class_label",
            "prediction",
            "predicted_label",
            "prediction_confidence",
            "entropy",
            "least_confidence",
            "margin",
            "margin_uncertainty",
            "selection_rank",
            "rank",
        }

        try:
            id_column = actions._resolve_record_id_column(self.context, dataset_id)
            if id_column:
                excluded.add(str(id_column))
        except Exception:
            pass

        candidate_columns = [
            column
            for column in columns
            if column not in excluded
        ]

        if not candidate_columns:
            return []

        try:
            try:
                df = self.context.datasets.get_df(dataset_id, columns=candidate_columns)
            except TypeError:
                df = self.context.datasets.get_df(dataset_id)

            numeric = [
                str(column)
                for column in candidate_columns
                if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
            ]
            return numeric
        except Exception:
            return candidate_columns

    def _refresh_training_contract_summary(self) -> None:
        if not hasattr(self, "training_contract_summary"):
            return

        input_contract = self._selected_recipe_input_contract()
        features = self._selected_feature_columns()
        validation_dataset_id = self.validation_dataset_select.value or ""
        test_dataset_id = self.test_dataset_select.value or ""

        validation_text = (
            f"dataset `{validation_dataset_id}`"
            if validation_dataset_id
            else f"random split `{float(self.validation_fraction.value or 0.0):.3g}`"
        )
        test_text = (
            f"dataset `{test_dataset_id}`"
            if test_dataset_id
            else (
                f"random split `{float(self.test_fraction.value or 0.0):.3g}`"
                if float(self.test_fraction.value or 0.0) > 0
                else "none"
            )
        )

        input_lines: List[str] = []
        if input_contract.needs_image:
            image_value = str(self.image_column.value or "").strip()
            input_lines.append(
                f"- Image column: `{image_value}`"
                if image_value
                else "- Image column: _auto-detect / unresolved_"
            )
        if input_contract.needs_mask:
            mask_value = str(self.mask_column.value or "").strip()
            input_lines.append(
                f"- Mask column: `{mask_value}`"
                if mask_value
                else "- Mask column: _auto-detect / unresolved_"
            )
        if input_contract.needs_features:
            feature_preview = ", ".join(f"`{column}`" for column in features[:10])
            if len(features) > 10:
                feature_preview += f", … +{len(features) - 10} more"
            if not feature_preview:
                feature_preview = "_none selected_"
            input_lines.append(
                f"- Input features ({len(features)}): {feature_preview}"
            )

        if not input_lines:
            input_lines.append("- Inputs: _recipe-managed_" )

        issues = []
        if self._latest_preflight_errors:
            issues.append(
                f"- Blocking compatibility issues: **{len(self._latest_preflight_errors)}**"
            )
        if self._latest_preflight_warnings:
            issues.append(
                f"- Compatibility warnings: **{len(self._latest_preflight_warnings)}**"
            )

        self.training_contract_summary.object = "\n".join(
            [
                "### Training contract",
                "",
                f"- Dataset: `{self.dataset_select.value or ''}`",
                f"- Recipe: `{self.recipe_select.value or ''}`",
                f"- Task/modality: `{input_contract.task}` / `{input_contract.modality}`",
                f"- Target/label column: `{self.target_column.value or 'al_label'}`",
                *input_lines,
                f"- Validation: {validation_text}",
                f"- Test: {test_text}",
                *issues,
                "",
                "_Only controls required by the selected recipe are shown on the Start tab._",
            ]
        )

    def _guess_label_column(self, columns: List[str]) -> str:
        lowered = {str(col).lower(): str(col) for col in columns}

        for candidate in (
            "target_label",
            "label",
            "labels",
            "class",
            "class_label",
            "target",
            "y",
        ):
            if candidate in lowered:
                return lowered[candidate]

        return "al_label"

    def _dataset_mapping(
        self,
        dataset_id: Any,
        semantic_name: str,
    ) -> Optional[str]:
        try:
            value = self.context.datasets.get_mapping(
                str(dataset_id),
                semantic_name,
            )
            return None if value is None else str(value)
        except Exception:
            return None

    def _recipe_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        spec = self._selected_recipe_spec()
        properties = (
            (getattr(spec, "params_schema", {}) or {}).get("properties", {})
            if spec
            else {}
        )

        for name, widget in self.recipe_param_widgets.items():
            value = widget.value
            schema = (
                properties.get(name, {})
                if isinstance(properties, Mapping)
                else {}
            )
            kind = str(schema.get("type", "string"))

            if kind in {"array", "object"} and isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    pass

            params[name] = value

        # Start tab is canonical for AL recipe contract settings.
        params.update(self._start_owned_recipe_params())

        return params

    # ------------------------------------------------------------------
    # Workflow gating
    # ------------------------------------------------------------------
    def _labelled_count(
        self,
        session: Optional[Mapping[str, Any]],
    ) -> int:
        if not session:
            return 0

        try:
            return int(
                al_state.counts(session).get(
                    "labelled_or_verified",
                    0,
                )
            )
        except Exception:
            return 0

    def _label_counts_from_session(
        self,
        session: Optional[Mapping[str, Any]],
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if not session:
            return counts
        for item in al_state.labelled_training_items(session):
            label = al_state.normalise_label(item.get("label"))
            if label and label != al_state.UNSURE_LABEL:
                counts[label] = counts.get(label, 0) + 1
        return counts

    def _session_has_been_trained(
        self,
        session: Optional[Mapping[str, Any]],
    ) -> bool:
        """True once the session has at least one completed training round.

        Primary signal is the round counter (incremented by training). A few
        alternative keys are checked too so the gate still works if a given
        platform records training differently. If your `start` action sets
        `round` to 1, adjust the round check below to `>= 2`.
        """
        if not session:
            return False

        try:
            if int(session.get("round") or 0) >= 1:
                return True
        except Exception:
            pass

        for key in (
            "model_artifact_id",
            "last_training",
            "trained_at",
            "training_history",
        ):
            if session.get(key):
                return True

        return False

    def _update_action_gating(self) -> None:
        if self._disposed:
            return

        if self._job_running:
            return

        session = self._load_current_session()
        has_session = bool(session)
        labelled = self._labelled_count(session)
        trained = self._session_has_been_trained(session)

        preflight_errors: List[str] = []
        try:
            preflight = self._build_session_contract(
                inspect_images=False,
                labelled_count=labelled,
            )
            preflight_errors = list(
                preflight.errors
                or []
            )
        except Exception as exc:
            preflight_errors = [
                str(exc)
            ]

        has_labels = bool(
            al_state.parse_label_options(
                self.label_options_select.value
            )
        )
        can_start = bool(
            self.dataset_select.value
            and self.recipe_select.value
            and has_labels
            and not preflight_errors
        )

        try:
            self.start_button.disabled = not can_start
        except Exception:
            pass

        can_train = (
            has_session
            and labelled >= 1
            and not preflight_errors
        )

        try:
            self.train_button.disabled = not can_train
        except Exception:
            pass

        if not has_session:
            self.train_hint.object = (
                "_Start or select a session first._"
            )
        elif labelled < 1:
            self.train_hint.object = (
                "_Label at least one point in the Review tab before training._"
            )
        elif preflight_errors:
            self.train_hint.object = (
                "_Resolve the recipe/data contract first: "
                + "; ".join(preflight_errors[:3])
                + "_"
            )
        else:
            self.train_hint.object = ""

        selected_prediction = str(
            self.predictions_select.value
            or al_state.latest_reference(session or {}, "predictions_artifact_id")
            or ""
        ).strip()
        prediction_ok = bool(
            selected_prediction
            and self._prediction_matches_current_session(
                selected_prediction
            )
        )

        can_query = (
            has_session
            and trained
            and prediction_ok
        )

        try:
            self.query_button.disabled = not can_query
        except Exception:
            pass

        if not has_session:
            self.query_hint.object = (
                "_Start or select a session first._"
            )
        elif not trained:
            self.query_hint.object = (
                "_Train at least once before querying from predictions._"
            )
        elif not selected_prediction:
            expected_dataset_id = (
                self._session_pool_dataset_id(
                    session
                )
            )
            self.query_hint.object = (
                "_No matching pool prediction exists. Train with automatic prediction "
                "enabled or select a compatible prediction artifact._"
            )
        elif not prediction_ok:
            expected_dataset_id = (
                self._session_pool_dataset_id(
                    session
                )
            )
            actual_dataset_id = (
                self._prediction_dataset_id(
                    selected_prediction
                )
            )
            self.query_hint.object = (
                "_Selected predictions do not match the AL pool dataset. "
                f"Expected `{expected_dataset_id}`, got `{actual_dataset_id}`._"
            )
        else:
            self.query_hint.object = ""

        auto_label_column = str(
            self.auto_label_source_column.value
            or ""
        ).strip()
        benchmark_mode = str(self.al_protocol.value or "review") == "benchmark"
        can_auto_label = (
            benchmark_mode
            and has_session
            and bool(auto_label_column)
        )
        for widget in (
            self.auto_label_source_column,
            self.auto_label_n,
            self.auto_label_button,
            self.auto_label_hint,
        ):
            try:
                widget.visible = benchmark_mode
            except Exception:
                pass

        try:
            selection = getattr(
                self.context,
                "selection",
                None,
            )
            active_set = (
                selection.get_active_set()
                if selection is not None
                else None
            )

            if active_set is None:
                can_auto_label = False
        except Exception:
            can_auto_label = False

        try:
            self.auto_label_button.disabled = (
                not can_auto_label
            )
        except Exception:
            pass

        if not benchmark_mode:
            self.auto_label_hint.object = ""
        elif not has_session:
            self.auto_label_hint.object = (
                "_Start or select a benchmark session before auto-labelling._"
            )
        elif not auto_label_column:
            self.auto_label_hint.object = (
                "_Choose an auto-label source column._"
            )
        elif not can_auto_label:
            self.auto_label_hint.object = (
                "_Create or select a query batch before auto-labelling._"
            )
        else:
            self.auto_label_hint.object = ""

    def _require_dataset(self) -> str:
        dataset_id = str(
            self.dataset_select.value
            or ""
        ).strip()

        if not dataset_id:
            raise ValueError(
                "Choose a dataset."
            )

        return dataset_id

    def _require_session(self) -> str:
        session_artifact_id = str(
            self._current_session_artifact_id
            or self.session_select.value
            or ""
        ).strip()

        if not session_artifact_id:
            raise ValueError(
                "Start or choose an active-learning session."
            )

        return session_artifact_id

    def _require_labelled(self) -> None:
        session = self._load_current_session()

        if self._labelled_count(session) < 1:
            raise ValueError(
                "Label at least one point in the Review tab before training."
            )

    def _require_trained(self) -> None:
        session = self._load_current_session()

        if not self._session_has_been_trained(
            session
        ):
            raise ValueError(
                "Train at least once before querying a top-k batch from predictions."
            )

    def _require_prediction_artifact(self) -> str:
        session = self._load_current_session() or {}
        artifact_id = str(
            self.predictions_select.value
            or al_state.latest_reference(session, "predictions_artifact_id")
            or ""
        ).strip()
        expected_dataset_id = (
            self._session_pool_dataset_id()
        )

        if not artifact_id:
            if expected_dataset_id:
                raise ValueError(
                    "No prediction exists for the AL pool dataset. Train with automatic "
                    "prediction enabled or select a compatible ml.predictions artifact."
                )

            raise ValueError(
                "Choose an ml.predictions artifact."
            )

        actual_dataset_id = (
            self._prediction_dataset_id(
                artifact_id
            )
        )

        if (
            expected_dataset_id
            and actual_dataset_id
            != expected_dataset_id
        ):
            raise ValueError(
                "The selected prediction artifact was not produced for this AL "
                "session's pool dataset.\n\n"
                f"Expected pool dataset: {expected_dataset_id!r}\n"
                f"Prediction dataset: {actual_dataset_id!r}\n\n"
                "For AL querying, use predictions over the original pool/source "
                "dataset, not over the derived AL training dataset. Run "
                "`core.ml.predict` on the pool dataset with the trained model."
            )

        return artifact_id

    def _set_status(
        self,
        message: str,
        alert_type: str = "info",
    ) -> None:
        """Update the status alert shown at the top of the AL panel."""
        try:
            self.status.object = str(
                message
                or ""
            )
            self.status.alert_type = str(
                alert_type
                or "info"
            )
            self.status.visible = bool(
                message
            )
        except Exception:
            print(
                f"[ActiveLearningPanel STATUS] {alert_type}: {message}",
                flush=True,
            )

    def _set_error(
        self,
        message: str,
        error: Any = None,
    ) -> None:
        """Display and log an error without letting the error handler crash.

        Many panel callbacks call this from except blocks, so this method must
        be defensive. If _set_status/status itself is temporarily broken, fall
        back to printing the error rather than raising another exception.
        """
        details = ""

        if error is not None:
            try:
                details = str(
                    error
                ).strip()
            except Exception:
                details = repr(error)

        display_message = str(
            message
            or "Error"
        ).strip()

        if details:
            display_message = (
                f"{display_message}: {details}"
            )

        try:
            tb = traceback.format_exc()

            if (
                tb
                and "NoneType: None"
                not in tb
            ):
                print(
                    (
                        "[ActiveLearningPanel ERROR] "
                        f"{display_message}\n{tb}"
                    ),
                    flush=True,
                )
            else:
                print(
                    (
                        "[ActiveLearningPanel ERROR] "
                        f"{display_message}"
                    ),
                    flush=True,
                )
        except Exception:
            pass

        try:
            self._set_status(
                display_message,
                "danger",
            )
            return
        except Exception:
            pass

        # Last-resort UI fallback if _set_status has also been removed/broken.
        try:
            self.status.object = (
                display_message
            )
            self.status.alert_type = (
                "danger"
            )
            self.status.visible = True
        except Exception:
            print(
                display_message,
                flush=True,
            )

    def _set_running(
        self,
        running: bool,
    ) -> None:
        self._job_running = running
        self.train_button.disabled = running
        self.start_button.disabled = running
        self.query_button.disabled = running
        self.label_button.disabled = running
        self.unsure_button.disabled = running
        self.auto_label_button.disabled = running
        self.inspect_data_button.disabled = running

        for button in (
            getattr(
                self,
                "analytics_refresh_button",
                None,
            ),
            getattr(
                self,
                "analytics_first_button",
                None,
            ),
            getattr(
                self,
                "analytics_prev_button",
                None,
            ),
            getattr(
                self,
                "analytics_next_button",
                None,
            ),
            getattr(
                self,
                "analytics_last_button",
                None,
            ),
            getattr(
                self,
                "feature_select_numeric_button",
                None,
            ),
            getattr(
                self,
                "feature_clear_button",
                None,
            ),
        ):
            if button is None:
                continue

            try:
                button.disabled = running
            except Exception:
                pass

        if not running:
            self._update_action_gating()
            self._update_analytics_nav_buttons()

    def _advance_focus_after(self, current_row_id: Any) -> None:
            import time

            t0 = time.perf_counter()

            try:
                next_row_id = self._next_review_row_id_after(current_row_id)
                t1 = time.perf_counter()

                if not next_row_id:
                    self._set_status(
                        "Recorded label. No more unlabelled rows remain in the current batch.",
                        "success",
                    )
                    t2 = time.perf_counter()
                    print(
                        "[AL advance timing]",
                        {
                            "find_next": round(t1 - t0, 4),
                            "no_next_status": round(t2 - t1, 4),
                            "total": round(t2 - t0, 4),
                        },
                        flush=True,
                    )
                    return

                dataset_id = self._session_pool_dataset_id()
                if not dataset_id:
                    session = self._load_current_session()
                    dataset_id = str(
                        session.get("pool_dataset_id")
                        or session.get("dataset_id")
                        or self.dataset_select.value
                        or ""
                    ).strip()

                t2 = time.perf_counter()

                if not dataset_id:
                    self._set_status(
                        f"Recorded label. Next row is {next_row_id}, but no dataset is active.",
                        "warning",
                    )
                    t3 = time.perf_counter()
                    print(
                        "[AL advance timing]",
                        {
                            "find_next": round(t1 - t0, 4),
                            "resolve_dataset": round(t2 - t1, 4),
                            "missing_dataset_status": round(t3 - t2, 4),
                            "total": round(t3 - t0, 4),
                        },
                        flush=True,
                    )
                    return

                self._set_focus_next_tick(
                    dataset_id=dataset_id,
                    row_id=next_row_id,
                    origin="core.active_learning.panel.advance",
                )

                t3 = time.perf_counter()

                print(
                    "[AL advance timing]",
                    {
                        "find_next": round(t1 - t0, 4),
                        "resolve_dataset": round(t2 - t1, 4),
                        "schedule_focus": round(t3 - t2, 4),
                        "total": round(t3 - t0, 4),
                        "next_row_id": str(next_row_id),
                    },
                    flush=True,
                )

            except Exception as exc:
                t_err = time.perf_counter()
                print(
                    "[AL advance timing]",
                    {
                        "error_after": round(t_err - t0, 4),
                        "error": str(exc),
                    },
                    flush=True,
                )
                self._set_error("Could not advance to the next review row", exc)

    def _auto_label_next_clicked(self, *_: Any) -> None:
            try:
                session_artifact_id = self._require_session()
                session = self._load_current_session() or {}

                dataset_id = str(
                    session.get("pool_dataset_id")
                    or session.get("dataset_id")
                    or self.dataset_select.value
                    or ""
                ).strip()
                if not dataset_id:
                    raise ValueError("Could not determine the active-learning pool dataset.")

                source_label_column = self._resolve_auto_label_source_column(
                    dataset_id=dataset_id,
                    session=session,
                )
                label_options = self._current_label_options()

                n = max(1, int(self.auto_label_n.value or 1))

                selection = getattr(self.context, "selection", None)
                if selection is None or not hasattr(selection, "get_active_set"):
                    raise ValueError("No active selection set is available.")

                active_set = selection.get_active_set()
                if active_set is None:
                    raise ValueError("No active selection set is available.")

                active_dataset_id = str(getattr(active_set, "dataset_id", "") or "")
                actions._assert_dataset_ids_match(
                    dataset_id,
                    active_dataset_id,
                    action="auto_label_next active selection",
                )

                ordered_row_ids = [
                    str(row_id)
                    for row_id in (getattr(active_set, "row_ids", []) or [])
                    if row_id is not None
                ]
                if not ordered_row_ids:
                    raise ValueError("The active selection set has no rows.")

                focus_row_id = self._current_focus_row_id()
                start_index = 0
                if focus_row_id is not None:
                    try:
                        start_index = ordered_row_ids.index(str(focus_row_id))
                    except ValueError:
                        start_index = 0

                already_seen = set(
                    str(row_id)
                    for row_id in (session.get("ignored_row_ids", []) or [])
                )
                already_seen.update(
                    str(row_id)
                    for row_id in ((session.get("labels") or {}).keys())
                )

                to_label: List[str] = []
                for row_id in ordered_row_ids[start_index:]:
                    row_id = str(row_id)
                    if row_id in already_seen and row_id != str(focus_row_id):
                        continue
                    to_label.append(row_id)
                    if len(to_label) >= n:
                        break

                if len(to_label) < n:
                    for row_id in ordered_row_ids[:start_index]:
                        row_id = str(row_id)
                        if row_id in already_seen:
                            continue
                        to_label.append(row_id)
                        if len(to_label) >= n:
                            break

                if not to_label:
                    self._set_status(
                        "No unlabelled rows remain in the active query batch.",
                        "warning",
                    )
                    return

                next_focus = None
                labelled_set = set(to_label)
                for row_id in ordered_row_ids:
                    row_id = str(row_id)
                    if row_id not in labelled_set and row_id not in already_seen:
                        next_focus = row_id
                        break

                selection_set_id = getattr(active_set, "selection_set_id", None)

                self._set_running(True)
                self.auto_label_button.name = f"Labelling {len(to_label)}..."
                self.auto_label_hint.object = (
                    f"_Auto-labelling {len(to_label)} row(s) from "
                    f"`{source_label_column}`. The button is disabled until this finishes._"
                )
                self._set_status(
                    f"Auto-labelling {len(to_label)} row(s) from `{source_label_column}`...",
                    "info",
                )

                worker_kwargs = {
                    "session_artifact_id": session_artifact_id,
                    "dataset_id": dataset_id,
                    "row_ids": to_label,
                    "source_label_column": source_label_column,
                    "label_options": label_options,
                    "requested_n": n,
                    "next_focus": next_focus,
                    "selection_set_id": selection_set_id,
                }

                submit = getattr(getattr(self.context, "jobs", None), "submit", None)
                if callable(submit):
                    key = (
                        f"core.active_learning.auto_label_next:"
                        f"{session_artifact_id}:{source_label_column}:{len(to_label)}"
                    )
                    self._active_job_handle = submit(
                        self._auto_label_next_worker,
                        title=f"Auto-label {len(to_label)} active-learning rows",
                        key=key,
                        on_done=self._on_auto_label_done,
                        on_error=self._on_auto_label_error,
                        **worker_kwargs,
                    )
                else:
                    result = self._auto_label_next_worker(**worker_kwargs)
                    self._on_auto_label_done(result)

            except Exception as exc:
                self._set_running(False)
                self.auto_label_button.name = "Label next N most informative points"
                self._active_job_handle = None
                self._set_error("Could not auto-label next points", exc)

    def _auto_label_next_worker(
            self,
            *,
            session_artifact_id: str,
            dataset_id: str,
            row_ids: List[str],
            source_label_column: str,
            label_options: List[str],
            requested_n: int,
            next_focus: Optional[str],
            selection_set_id: Any = None,
            cancel_token: Any = None,
        ) -> Dict[str, Any]:
            """Background worker for Review-tab benchmark auto-labelling.

            This deliberately performs a bulk session update and writes one session
            artifact at the end. Calling actions.record_label_action once per row is
            correct but very slow for large batches because each call saves a new
            artifact and publishes refresh events.
            """
            actions._check_cancelled(cancel_token)

            session = al_state.coerce_session(
                self.context.artifacts.get(session_artifact_id)
            )

            labels_by_row_id = self._read_existing_labels_for_rows(
                dataset_id=dataset_id,
                row_ids=row_ids,
                target_column=source_label_column,
            )

            updated = al_state.coerce_session(session)
            labels = dict(updated.get("labels") or {})

            ignored: List[str] = [
                str(row_id)
                for row_id in (updated.get("ignored_row_ids", []) or [])
                if row_id is not None
            ]
            ignored_seen = set(ignored)

            training: List[str] = [
                str(row_id)
                for row_id in (updated.get("training_row_ids", []) or [])
                if row_id is not None
            ]
            training_seen = set(training)

            history = list(updated.get("history") or [])
            round_index = int(updated.get("round", 0))

            recorded_rows: List[Dict[str, Any]] = []
            skipped_missing: List[str] = []

            for row_id in row_ids:
                actions._check_cancelled(cancel_token)

                row_id = str(row_id)
                raw_label = labels_by_row_id.get(row_id)
                label = self._normalise_auto_label_value(
                    raw_label,
                    label_options=label_options,
                )

                if label is None or str(label).strip() == "":
                    skipped_missing.append(row_id)
                    continue

                label_value = al_state.normalise_label(label)
                if not label_value:
                    skipped_missing.append(row_id)
                    continue

                timestamp = al_state.now()
                is_unsure = label_value == al_state.UNSURE_LABEL

                entry = {
                    "row_id": row_id,
                    "label": label_value,
                    "display_label": al_state.display_label(label_value),
                    "status": "unsure" if is_unsure else "verified",
                    "source": "active_learning_panel.auto_label_next",
                    "round": round_index,
                    "timestamp": timestamp,
                }
                labels[row_id] = entry

                if row_id not in ignored_seen:
                    ignored.append(row_id)
                    ignored_seen.add(row_id)

                # A row may have been labelled before. Remove stale training status,
                # then add it back only if the new value is a verified label.
                if row_id in training_seen:
                    training = [item for item in training if item != row_id]
                    training_seen.discard(row_id)

                if not is_unsure:
                    training.append(row_id)
                    training_seen.add(row_id)

                history.append(
                    {
                        "event": "label_recorded",
                        "row_id": row_id,
                        "label": label_value,
                        "status": entry["status"],
                        "timestamp": timestamp,
                    }
                )

                recorded_rows.append(
                    {
                        "row_id": row_id,
                        "label": label_value,
                        "display_label": al_state.display_label(label_value),
                        "status": entry["status"],
                    }
                )

            updated["labels"] = labels
            updated["ignored_row_ids"] = list(dict.fromkeys(ignored))
            updated["training_row_ids"] = list(dict.fromkeys(training))
            updated["history"] = history
            updated["updated_at"] = al_state.now()

            if recorded_rows:
                new_session_artifact_id = actions._put_session(
                    self.context,
                    updated,
                    previous_artifact_id=session_artifact_id,
                )
            else:
                new_session_artifact_id = session_artifact_id

            counts = al_state.counts(updated)

            if recorded_rows:
                actions._publish(
                    self.context,
                    "al.labels.recorded_bulk",
                    {
                        "session_artifact_id": new_session_artifact_id,
                        "previous_session_artifact_id": session_artifact_id,
                        "session_id": updated.get("session_id"),
                        "dataset_id": updated.get("dataset_id"),
                        "source_dataset_id": dataset_id,
                        "source_label_column": source_label_column,
                        "recorded_count": len(recorded_rows),
                        "skipped_missing_count": len(skipped_missing),
                        "requested_n": requested_n,
                        "row_ids": [row["row_id"] for row in recorded_rows],
                        "counts": counts,
                    },
                )

            return {
                "ok": True,
                "session_artifact_id": new_session_artifact_id,
                "previous_session_artifact_id": session_artifact_id,
                "session_id": updated.get("session_id"),
                "dataset_id": updated.get("dataset_id"),
                "source_dataset_id": dataset_id,
                "source_label_column": source_label_column,
                "recorded": len(recorded_rows),
                "recorded_rows": recorded_rows[:25],
                "skipped_missing": skipped_missing,
                "skipped_missing_count": len(skipped_missing),
                "requested_n": requested_n,
                "next_focus": next_focus,
                "selection_set_id": selection_set_id,
                "counts": counts,
            }

    def _auto_label_source_changed(self, event: Any) -> None:
            if bool(getattr(self, "_syncing_auto_label_source", False)):
                return

            value = str(getattr(event, "new", "") or "").strip()
            self._auto_label_source_user_set = bool(value)

    def _build_protocol_section(self, *, managed: bool):
            """Build the advanced protocol block.

            Start tab owns validation/test datasets and random split fractions.
            This block only exposes genuinely advanced protocol settings.
            """
            if not managed:
                return None

            column_options = [""] + self._dataset_columns(self.dataset_select.value)

            advanced_toggle = pn.widgets.Checkbox(
                name="Show advanced validation/test protocol",
                value=False,
                sizing_mode="stretch_width",
            )
            self._fit_widget(advanced_toggle)

            self.protocol_widgets = {
                "protocol_split_strategy": pn.widgets.Select(
                    name="",
                    options={
                        "Random split": "random",
                        "Group-aware split": "by_group",
                        "Temporal split": "temporal",
                        "Predefined split column": "predefined",
                    },
                    value="random",
                ),
                "protocol_group_column": pn.widgets.Select(
                    name="",
                    options=column_options,
                    value="",
                ),
                "protocol_split_column": pn.widgets.Select(
                    name="",
                    options=column_options,
                    value="",
                ),
                "protocol_selection_metric": pn.widgets.Select(
                    name="",
                    options={
                        "Validation accuracy": "val_accuracy",
                        "Validation macro F1": "val_f1_macro",
                        "Validation loss": "val_loss",
                    },
                    value="val_accuracy",
                ),
                "protocol_random_state": pn.widgets.IntInput(
                    name="",
                    value=int(self.seed.value or 42),
                ),
            }

            advanced_body = pn.Column(
                self._field("Split method", self.protocol_widgets["protocol_split_strategy"]),
                self._field("Group/time column", self.protocol_widgets["protocol_group_column"]),
                self._field("Predefined split column", self.protocol_widgets["protocol_split_column"]),
                self._field("Best-epoch metric", self.protocol_widgets["protocol_selection_metric"]),
                self._field("Protocol random seed", self.protocol_widgets["protocol_random_state"]),
                visible=False,
                sizing_mode="stretch_width",
                styles={
                    "max-width": "100%",
                    "width": "100%",
                    "box-sizing": "border-box",
                    "overflow": "visible",
                    "padding-top": "6px",
                },
            )

            def _toggle_advanced(event: Any) -> None:
                advanced_body.visible = bool(getattr(event, "new", False))

            advanced_toggle.param.watch(_toggle_advanced, "value")

            return pn.Column(
                pn.layout.Divider(margin=(8, 0, 6, 0)),
                advanced_toggle,
                advanced_body,
                sizing_mode="stretch_width",
                styles={
                    "max-width": "100%",
                    "width": "100%",
                    "box-sizing": "border-box",
                    "overflow": "visible",
                    "clear": "both",
                },
            )

    def _current_ranked_review_row_ids(self) -> List[str]:
            """Return the ordered row ids for the current review batch.

            This must stay cheap. Do not query datasets here. The Review hot path
            should use only session/batch/selection metadata that is already in
            memory or stored in small artifacts.
            """
            session = self._load_current_session()
            if not isinstance(session, Mapping):
                return []

            last_batch = session.get("last_batch") or {}
            row_ids: List[str] = []

            if isinstance(last_batch, Mapping):
                for row_id in last_batch.get("row_ids") or []:
                    if row_id is not None:
                        row_ids.append(str(row_id))

            if row_ids:
                return list(dict.fromkeys(row_ids))

            batch_artifact_id = ""
            if isinstance(last_batch, Mapping):
                batch_artifact_id = str(last_batch.get("batch_artifact_id") or "").strip()

            if not batch_artifact_id:
                batch_artifact_id = str(session.get("last_batch_artifact_id") or "").strip()

            if batch_artifact_id:
                try:
                    batch = self.context.artifacts.get(batch_artifact_id)
                except Exception:
                    batch = None

                if isinstance(batch, Mapping):
                    for row_id in batch.get("row_ids") or []:
                        if row_id is not None:
                            row_ids.append(str(row_id))

                    if not row_ids:
                        for record in batch.get("records") or []:
                            if not isinstance(record, Mapping):
                                continue
                            row_id = record.get("row_id")
                            if row_id is not None:
                                row_ids.append(str(row_id))

            if row_ids:
                return list(dict.fromkeys(row_ids))

            selection = getattr(self.context, "selection", None)
            if selection is None:
                return []

            # Best-effort fallback for platform selection-set APIs. Keep this broad
            # because SelectionManager has changed a few times during the plugin
            # system migration.
            for method_name in (
                "get_selection_set",
                "get_active_selection_set",
                "get_current_selection_set",
            ):
                method = getattr(selection, method_name, None)
                if not callable(method):
                    continue

                try:
                    selection_set = method()
                except TypeError:
                    try:
                        selection_set = method(None)
                    except Exception:
                        continue
                except Exception:
                    continue

                if selection_set is None:
                    continue

                values = getattr(selection_set, "row_ids", None)
                if values is None and isinstance(selection_set, Mapping):
                    values = selection_set.get("row_ids")

                for row_id in values or []:
                    if row_id is not None:
                        row_ids.append(str(row_id))

                if row_ids:
                    return list(dict.fromkeys(row_ids))

            return []

    def _dataset_changed(self, *_: Any) -> None:
            self._source_label_cache.clear()
            self._refresh_column_widgets()
            self._refresh_validation_test_datasets()
            self._refresh_recipe_params()
            self._apply_recipe_inferred_defaults()
            self._refresh_input_contract_ui()
            self._refresh_data_contract_summary()
            self._refresh_labels_from_available_context()
            self._mark_analytics_dirty()
            self._refresh_analytics_if_visible()

    def _evaluation_source_changed(self, *_: Any) -> None:
            self._refresh_evaluation_widget_visibility()
            self._refresh_training_contract_summary()
            self._refresh_data_contract_summary()
            self._update_action_gating()

    def _feature_columns_changed(self, *_: Any) -> None:
            self._refresh_feature_match_preview()
            self._refresh_training_contract_summary()

    def _input_binding_changed(self, *_: Any) -> None:
            self._refresh_training_contract_summary()
            self._refresh_data_contract_summary()
            self._update_action_gating()

    def _label_clicked(self, *_: Any) -> None:
            self._record_current_label(self.label_select.value)

    def _normalise_auto_label_value(
            self,
            value: Any,
            *,
            label_options: List[str],
        ) -> Optional[str]:
            """Convert source label values into AL label values.

            Handles the common CIFAR/HuggingFace case where labels are integer class
            IDs and label_options contains class names.
            """
            if value is None or pd.isna(value):
                return None

            # Numeric class index -> label option.
            try:
                if label_options:
                    index = int(value)
                    if 0 <= index < len(label_options):
                        return str(label_options[index])
            except Exception:
                pass

            text = str(value).strip()
            return text or None

    def _on_auto_label_done(self, result: Any) -> None:
            if self._disposed:
                return

            self._active_job_handle = None
            self.auto_label_button.name = "Label next N most informative points"
            self.auto_label_hint.object = ""

            if not isinstance(result, Mapping):
                self._set_running(False)
                self._set_status(
                    "Auto-labelling finished, but returned an unexpected result shape.",
                    "warning",
                )
                return

            session_artifact_id = str(result.get("session_artifact_id") or "").strip()
            if session_artifact_id:
                self._current_session_artifact_id = session_artifact_id

            self._set_running(False)

            dataset_id = str(
                result.get("source_dataset_id")
                or result.get("dataset_id")
                or self.dataset_select.value
                or ""
            ).strip()
            next_focus = result.get("next_focus")

            if dataset_id and next_focus is not None:
                selection = getattr(self.context, "selection", None)
                set_focus = getattr(selection, "set_focus", None)
                if callable(set_focus):
                    try:
                        set_focus(
                            dataset_id=dataset_id,
                            row_id=str(next_focus),
                            origin="core.active_learning.panel.auto_label_next",
                            selection_set_id=result.get("selection_set_id"),
                            metadata={
                                "reason": "after_auto_label_next",
                                "auto_labelled_count": int(result.get("recorded") or 0),
                                "requested_n": int(result.get("requested_n") or 0),
                            },
                        )
                    except Exception:
                        pass

            self.refresh()
            self._switch_controls_tab("Train" if int(result.get("recorded") or 0) else "Review")
            self._switch_results_tab("Labelled / verified / unsure")

            recorded = int(result.get("recorded") or 0)
            skipped_missing = int(result.get("skipped_missing_count") or 0)
            source_label_column = result.get("source_label_column") or "selected column"

            message = f"Auto-labelled {recorded} row(s) from `{source_label_column}`."
            if skipped_missing:
                message += f" Skipped {skipped_missing} row(s) with missing labels."
            if next_focus is not None:
                message += f" Focus moved to row `{next_focus}`."

            self._set_status(message, "success" if recorded else "warning")

    def _on_auto_label_error(self, error: Any) -> None:
            if self._disposed:
                return
            self._set_running(False)
            self.auto_label_button.name = "Label next N most informative points"
            self.auto_label_hint.object = ""
            self._active_job_handle = None
            self._set_error("Could not auto-label next points", error)

    def _on_query_done(self, result: Any) -> None:
            if self._disposed:
                return

            self._set_running(False)
            self._active_job_handle = None

            if not isinstance(result, Mapping):
                self._set_status(
                    "Query finished, but returned an unexpected result shape.",
                    "warning",
                )
                return

            self._use_action_session_result(result, include_analytics=False)

            # Query creates a new batch, so analytics is stale. Refresh it only
            # because we are explicitly switching to the Analytics tab here.
            self._mark_analytics_dirty()
            self._switch_results_tab("Analytics")
            self._refresh_analytics_if_visible(force=True)

            stats = result.get("rank_stats") or {}
            self._set_status(
                (
                    f"Created ordered query batch with {len(result.get('row_ids') or [])} rows. "
                    f"Seen={stats.get('total_records_seen', 'unknown')}, "
                    f"excluded={stats.get('excluded_count', 'unknown')}, "
                    f"unscorable={stats.get('unscorable_count', 'unknown')}."
                ),
                "success",
            )

    def _on_query_error(self, error: Any) -> None:
            if self._disposed:
                return
            self._set_running(False)
            self._active_job_handle = None
            self._switch_controls_tab("Query")
            self._set_error("Could not create active-learning query batch", error)

    def _on_train_done(self, result: Any) -> None:
            if self._disposed:
                return

            self._set_running(False)
            self._active_job_handle = None

            if not isinstance(result, Mapping):
                self._set_status(
                    "Training finished, but returned an unexpected result shape.",
                    "warning",
                )
                return

            self._use_action_session_result(result, include_analytics=False)
            self._refresh_predictions()
            self._mark_analytics_dirty()
            self._switch_results_tab("Analytics")
            self._refresh_analytics_if_visible(force=True)

            query_result = dict(result.get("query_result") or {})
            prediction_result = dict(result.get("prediction_result") or {})
            workflow_errors = list(result.get("workflow_errors") or [])
            queried = len(query_result.get("row_ids") or [])
            predicted = prediction_result.get("count")

            if workflow_errors:
                details = "; ".join(
                    f"{item.get('stage')}: {item.get('error')}"
                    for item in workflow_errors
                )
                self._set_status(
                    (
                        f"Training round {result.get('round')} completed with "
                        f"{result.get('labelled_count')} verified labels, but the "
                        f"automatic workflow was only partial: {details}"
                    ),
                    "warning",
                )
            else:
                self._switch_controls_tab("Review" if queried else "Query")
                self._set_status(
                    (
                        f"Round {result.get('round')} complete: trained on "
                        f"{result.get('labelled_count')} verified labels, predicted "
                        f"{predicted if predicted is not None else 'the pool'}, and "
                        f"queued {queried} row(s) for review."
                    ),
                    "success",
                )

    def _on_train_error(self, error: Any) -> None:
            if self._disposed:
                return
            self._set_running(False)
            self._active_job_handle = None
            self._switch_controls_tab("Train")
            self._set_error("Scratch training failed", error)

    def _physical_columns_for_dataset(self, dataset_id: Any) -> List[str]:
            dataset_id = str(dataset_id or "").strip()
            if not dataset_id:
                return []

            try:
                return [str(col) for col in self.context.datasets.list_columns(dataset_id)]
            except Exception:
                pass

            try:
                df = self.context.datasets.get_df(dataset_id)
                return [str(col) for col in df.columns]
            except Exception:
                return []

    def _predictions_changed(self, *_: Any) -> None:
            self._refresh_labels_from_available_context()
            self._update_action_gating()

    def _query_clicked(self, *_: Any) -> None:
            try:
                session_artifact_id = self._require_session()
                self._require_trained()
                predictions_artifact_id = self._require_prediction_artifact()

                request = ActionRequest(
                    artifact_id=predictions_artifact_id,
                    params={
                        "session_artifact_id": session_artifact_id,
                        "predictions_artifact_id": predictions_artifact_id,
                        "strategy_id": self.strategy_select.value or "least_confidence",
                        "k": int(self.query_k.value or 200),
                        "seed": int(self.seed.value or 0),
                        "make_selection": True,
                    },
                    origin="core.active_learning.panel",
                )

                self._switch_controls_tab("Query")

                manager = getattr(self.context, "plugins", None)
                run_action = getattr(manager, "run_action", None)
                if callable(run_action):
                    self._set_running(True)
                    self._set_status("Querying prediction artifact...", "info")
                    self._active_job_handle = run_action(
                        "core.active_learning.query_batch",
                        self.context,
                        request,
                        on_done=self._on_query_done,
                        on_error=self._on_query_error,
                    )
                else:
                    result = actions.query_batch_action(self.context, request)
                    self._on_query_done(result)

            except Exception as exc:
                self._set_running(False)
                self._set_error("Could not create active-learning query batch", exc)

    def _read_existing_labels_for_rows(
            self,
            *,
            dataset_id: str,
            row_ids: List[str],
            target_column: str,
        ) -> Dict[str, Any]:
            """Read existing labels for specific row IDs from a physical dataset column.

            Handles exact int64 ids and fallback scientific-notation ids produced by
            UI/table display paths.
            """
            if not row_ids:
                return {}

            dataset_id = str(dataset_id or "").strip()
            target_column = str(target_column or "").strip()
            row_ids = [str(row_id) for row_id in row_ids]
            requested_key_map = self._requested_row_id_key_map(row_ids)

            if not dataset_id:
                raise ValueError("No dataset_id supplied for auto-labelling.")
            if not target_column:
                raise ValueError("No auto-label source column supplied.")

            id_column = actions._resolve_record_id_column(self.context, dataset_id)

            def get_df_compat(columns: Optional[List[str]] = None) -> pd.DataFrame:
                if columns:
                    projected = list(
                        dict.fromkeys(
                            str(col)
                            for col in columns
                            if col and str(col).strip()
                        )
                    )
                    try:
                        return self.context.datasets.get_df(dataset_id, columns=projected)
                    except TypeError:
                        return self.context.datasets.get_df(dataset_id)
                    except Exception:
                        return self.context.datasets.get_df(dataset_id)

                return self.context.datasets.get_df(dataset_id)

            def available_columns_debug(frame: Optional[pd.DataFrame] = None) -> List[str]:
                if frame is not None:
                    try:
                        return [str(col) for col in frame.columns]
                    except Exception:
                        pass

                try:
                    return [str(col) for col in self.context.datasets.list_columns(dataset_id)]
                except Exception:
                    pass

                try:
                    return [str(col) for col in get_df_compat().columns]
                except Exception:
                    return []

            def extract_by_id_column(frame: pd.DataFrame) -> Dict[str, Any]:
                if id_column not in frame.columns:
                    raise ValueError(
                        f"record_id column {id_column!r} is not present in dataset "
                        f"{dataset_id!r}. Available columns: {list(frame.columns)}"
                    )

                if target_column not in frame.columns:
                    raise ValueError(
                        f"Auto-label source column {target_column!r} is not present in "
                        f"dataset {dataset_id!r}. Available columns: {list(frame.columns)}"
                    )

                out: Dict[str, Any] = {}

                for _, row in frame.iterrows():
                    source_row_id = row[id_column]
                    matching_requested_id = None

                    for key in self._row_id_match_keys(source_row_id):
                        if key in requested_key_map:
                            matching_requested_id = requested_key_map[key]
                            break

                    if matching_requested_id is None:
                        continue

                    value = row[target_column]
                    if pd.notna(value):
                        out[str(matching_requested_id)] = value

                return out

            def extract_by_index(frame: pd.DataFrame) -> Dict[str, Any]:
                if target_column not in frame.columns:
                    raise ValueError(
                        f"Auto-label source column {target_column!r} is not present in "
                        f"dataset {dataset_id!r}. Available columns: {list(frame.columns)}"
                    )

                out: Dict[str, Any] = {}

                for idx, row in frame.iterrows():
                    matching_requested_id = None
                    for key in self._row_id_match_keys(idx):
                        if key in requested_key_map:
                            matching_requested_id = requested_key_map[key]
                            break

                    if matching_requested_id is None:
                        continue

                    value = row[target_column]
                    if pd.notna(value):
                        out[str(matching_requested_id)] = value

                return out

            if id_column:
                row_lookup_df: Optional[pd.DataFrame] = None

                try:
                    row_lookup_df = self.context.datasets.get_rows_by_ids(
                        dataset_id,
                        row_ids,
                        id_column=id_column,
                    )
                    row_lookup_df = row_lookup_df.copy()
                except Exception:
                    row_lookup_df = None

                if (
                    row_lookup_df is not None
                    and not row_lookup_df.empty
                    and id_column in row_lookup_df.columns
                    and target_column in row_lookup_df.columns
                ):
                    found = extract_by_id_column(row_lookup_df)
                    if found:
                        return found

                try:
                    source_df = get_df_compat([id_column, target_column])
                except Exception as exc:
                    returned_columns = available_columns_debug(row_lookup_df)
                    raise ValueError(
                        f"Could not read columns {id_column!r} and {target_column!r} "
                        f"from dataset {dataset_id!r}. Row lookup returned columns: "
                        f"{returned_columns}"
                    ) from exc

                if id_column not in source_df.columns or target_column not in source_df.columns:
                    returned_columns = available_columns_debug(source_df)
                    row_lookup_columns = (
                        list(row_lookup_df.columns)
                        if row_lookup_df is not None
                        else []
                    )
                    raise ValueError(
                        f"Could not read auto-label source column {target_column!r} "
                        f"from dataset {dataset_id!r} using id column {id_column!r}.\n"
                        f"Projected/full dataframe columns: {returned_columns}\n"
                        f"get_rows_by_ids dataframe columns: {row_lookup_columns}"
                    )

                mask = self._row_id_series_mask(source_df[id_column], requested_key_map)
                filtered = source_df[mask].copy()

                return extract_by_id_column(filtered)

            source_df = get_df_compat([target_column])

            if target_column not in source_df.columns:
                raise ValueError(
                    f"Auto-label source column {target_column!r} is not present in "
                    f"dataset {dataset_id!r}. Available columns: {list(source_df.columns)}"
                )

            return extract_by_index(source_df)

    def _recipe_changed(self, *_: Any) -> None:
            self._refresh_recipe_params()
            self._apply_recipe_inferred_defaults()
            self._refresh_input_contract_ui()
            self._refresh_data_contract_summary()
            self._refresh_labels_from_available_context()
            self._update_action_gating()

    def _record_current_label(self, label: Any) -> None:
            import time

            try:
                t0 = time.perf_counter()

                session_artifact_id = self._require_session()
                row_id = self._current_focus_row_id()
                if row_id is None:
                    raise ValueError("No focused row is active. Select a row first.")

                request = ActionRequest(
                    params={
                        "session_artifact_id": session_artifact_id,
                        "row_id": row_id,
                        "label": label,
                        "source": "active_learning_panel",
                    },
                    origin="core.active_learning.panel",
                )

                result = actions.record_label_action(self.context, request)
                t1 = time.perf_counter()

                new_session_artifact_id = result.get("session_artifact_id")
                if new_session_artifact_id:
                    self._current_session_artifact_id = str(new_session_artifact_id)

                display = al_state.display_label(result.get("label"))
                self._set_status(f"Recorded {display!r} for row {row_id}.", "success")

                t2 = time.perf_counter()

                if bool(self.advance_after_label.value):
                    self._advance_focus_after(row_id)

                t3 = time.perf_counter()

                self._mark_analytics_dirty()
                self._schedule_light_review_refresh()

                t4 = time.perf_counter()

                print(
                    "[AL label timing]",
                    {
                        "record_label_action": round(t1 - t0, 4),
                        "status_update": round(t2 - t1, 4),
                        "advance_focus": round(t3 - t2, 4),
                        "schedule_light_refresh": round(t4 - t3, 4),
                        "total_hot_path": round(t4 - t0, 4),
                    },
                    flush=True,
                )

            except Exception as exc:
                self._set_error("Could not record label", exc)

    def _refresh_analytics_views(self) -> None:
            if self._disposed:
                return

            session = self._load_current_session()
            if not session:
                self._analytics_frames = []
                self.analytics_frame_label.object = "_Start or select an AL session to see analytics._"
                self.analytics_scatter.object = al_analytics.make_empty_figure(
                    "Start or select an AL session to see trained-point analytics."
                )
                self.analytics_performance_plot.object = al_analytics.make_empty_figure(
                    "No active-learning session selected."
                )
                self.analytics_informativeness_plot.object = al_analytics.make_empty_figure(
                    "No active-learning session selected."
                )
                self.analytics_summary.object = ""
                _set_table_value(self.analytics_points_table, pd.DataFrame())
                self._update_analytics_nav_buttons()
                return

            self._analytics_frames = al_analytics.query_batches_for_session(
                self.context,
                session,
            )

            if self._analytics_frames:
                self._analytics_frame_index = max(
                    0,
                    min(
                        int(self._analytics_frame_index or 0),
                        len(self._analytics_frames) - 1,
                    ),
                )
            else:
                self._analytics_frame_index = 0

            performance_df = al_analytics.performance_dataframe(
                self.context,
                session,
            )
            informativeness_df = al_analytics.informativeness_dataframe(
                self.context,
                session,
            )

            self.analytics_performance_plot.object = al_analytics.make_performance_figure(
                performance_df,
            )
            self.analytics_informativeness_plot.object = al_analytics.make_informativeness_figure(
                informativeness_df,
            )

            x_column = str(self.analytics_x_column.value or "").strip()
            y_column = str(self.analytics_y_column.value or "").strip()

            if not x_column or not y_column:
                self.analytics_scatter.object = al_analytics.make_empty_figure(
                    "Choose X and Y columns for the trained-point scatter plot."
                )
                self.analytics_frame_label.object = "_Choose X and Y columns to render the scatter frame._"
                self.analytics_summary.object = self._analytics_summary_markdown(
                    session=session,
                    performance_df=performance_df,
                    informativeness_df=informativeness_df,
                    scatter_df=pd.DataFrame(),
                    frame={},
                )
                _set_table_value(self.analytics_points_table, pd.DataFrame())
                self._update_analytics_nav_buttons()
                return

            try:
                scatter_df, frame = al_analytics.training_scatter_dataframe(
                    self.context,
                    session,
                    frame_index=int(self._analytics_frame_index or 0),
                    x_column=x_column,
                    y_column=y_column,
                )
            except Exception as exc:
                scatter_df = pd.DataFrame()
                frame = {}
                self.analytics_scatter.object = al_analytics.make_empty_figure(
                    f"Could not build AL scatter frame: {exc}"
                )
            else:
                self.analytics_scatter.object = al_analytics.make_training_scatter_figure(
                    scatter_df,
                    frame=frame,
                    x_column=x_column,
                    y_column=y_column,
                )

            frame_index = int(frame.get("frame_index", self._analytics_frame_index) or 0)
            frame_count = int(frame.get("frame_count", len(self._analytics_frames)) or len(self._analytics_frames))
            round_index = frame.get("round", session.get("round", 0))
            strategy_id = frame.get("strategy_id") or frame.get("strategy") or "unknown"
            batch_artifact_id = frame.get("artifact_id") or "none"

            if frame_count:
                self.analytics_frame_label.object = (
                    f"**Frame {frame_index + 1}/{frame_count}** · "
                    f"round `{round_index}` · strategy `{strategy_id}` · "
                    f"batch `{batch_artifact_id}`"
                )
            else:
                self.analytics_frame_label.object = (
                    "_No query-batch artifacts found yet. The scatter falls back to "
                    "currently verified/trained session labels where possible._"
                )

            self.analytics_summary.object = self._analytics_summary_markdown(
                session=session,
                performance_df=performance_df,
                informativeness_df=informativeness_df,
                scatter_df=scatter_df,
                frame=frame,
            )

            table_df = scatter_df.copy()
            if not table_df.empty:
                columns = [
                    col
                    for col in (
                        "row_id",
                        "trained_round",
                        "queried_round",
                        "strategy_id",
                        "informativeness_score",
                        "x",
                        "y",
                        "source_batch_artifact_id",
                    )
                    if col in table_df.columns
                ]
                table_df = table_df[columns].head(1000)

            _set_table_value(self.analytics_points_table, table_df)
            self._update_analytics_nav_buttons()

    def _refresh_auto_label_source_column(self) -> None:
            """Refresh physical dataset columns usable for benchmark auto-labelling.

            Default follows the Start tab's training label column unless the user
            has manually chosen a different Review auto-label source column.
            """
            current = self.auto_label_source_column.value
            dataset_id = self.dataset_select.value

            columns = self._physical_columns_for_dataset(dataset_id)
            options = {col: col for col in columns}

            self.auto_label_source_column.options = options

            target_column = str(self.target_column.value or "").strip()

            if (
                self._auto_label_column_should_follow_target()
                and target_column
                and target_column != "al_label"
                and target_column in columns
            ):
                self._set_auto_label_source_value(target_column)
                return

            if current in options.values():
                self._set_auto_label_source_value(current)
                return

            mapped_target = self._dataset_mapping(dataset_id, "target_label")
            if mapped_target and mapped_target in columns:
                self._set_auto_label_source_value(mapped_target)
                return

            for candidate in (
                target_column,
                "label_name",
                "target_label_name",
                "class_name",
                "target_name",
                "class_label",
                "target_label",
                "label",
                "class",
                "target",
                "fine_label",
                "coarse_label",
            ):
                if candidate and candidate in columns:
                    self._set_auto_label_source_value(candidate)
                    return

            self._set_auto_label_source_value(columns[0] if columns else None)

    def _refresh_column_widgets(self) -> None:
            """Refresh column-backed dropdowns after dataset or mappings change."""
            dataset_id = self.dataset_select.value
            columns = self._dataset_columns(dataset_id)

            target_options = list(dict.fromkeys(["al_label", *columns]))
            current_target = self.target_column.value

            self.target_column.options = target_options

            mapped_target = self._dataset_mapping(dataset_id, "target_label")
            if current_target in target_options:
                self.target_column.value = current_target
            elif mapped_target in target_options:
                self.target_column.value = mapped_target
            else:
                guessed = self._guess_label_column(columns)
                self.target_column.value = guessed if guessed in target_options else "al_label"

            self._refresh_feature_column_options()
            self._refresh_binding_widgets()

            column_options = [""] + columns

            for name, widget in list(self.recipe_param_widgets.items()):
                lname = str(name).lower()

                if lname in {
                    "feature_columns",
                    "input_columns",
                    "features",
                    "x_columns",
                }:
                    current_values = self._normalise_column_list(getattr(widget, "value", None))
                    current_values = [
                        str(value)
                        for value in current_values
                        if str(value) in columns
                    ]

                    if isinstance(widget, pn.widgets.TextAreaInput):
                        widget.value = "\n".join(current_values)
                        continue

                    if hasattr(widget, "options"):
                        try:
                            widget.options = columns
                        except Exception:
                            pass

                    try:
                        widget.value = current_values
                    except Exception:
                        pass

                    continue

                if not isinstance(widget, pn.widgets.Select):
                    continue

                looks_like_column_param = (
                    lname.endswith("_column")
                    or lname
                    in {
                        "target",
                        "label",
                        "label_column",
                        "target_column",
                        "image_column",
                        "image_path_column",
                        "image_uri_column",
                        "mask_column",
                        "record_id_column",
                        "id_column",
                        "group_column",
                        "split_column",
                    }
                )

                if not looks_like_column_param:
                    continue

                current = widget.value
                widget.options = column_options
                widget.value = current if current in column_options else ""

            for key in ("protocol_group_column", "protocol_split_column"):
                widget = self.protocol_widgets.get(key)
                if widget is None or not isinstance(widget, pn.widgets.Select):
                    continue

                current = widget.value
                widget.options = column_options
                widget.value = current if current in column_options else ""

            self._refresh_labels_from_available_context()
            self._refresh_training_contract_summary()

    def _refresh_review_label_for_focus(self) -> None:
            """Set the Review label dropdown from session label, source column, or Unsure."""
            if self._disposed:
                return

            label = self._label_for_current_focus()
            if label is None:
                label = al_state.UNSURE_LABEL

            label = al_state.normalise_label(label)

            # Ensure the inferred label is present in both the allowed-label widget
            # and the review dropdown. Unsure is always allowed as a review action.
            if label != al_state.UNSURE_LABEL:
                allowed = al_state.parse_label_options(self.label_options_select.value)
                if label not in allowed:
                    allowed.append(label)
                    self.label_options_select.options = list(
                        dict.fromkeys([*self.label_options_select.options, label])
                    )
                    self.label_options_select.value = allowed

            self._set_label_select_options()

            valid_values = set(self.label_select.options.values()) if isinstance(self.label_select.options, dict) else set(self.label_select.options)
            if label in valid_values:
                self.label_select.value = label
            else:
                self.label_select.value = al_state.UNSURE_LABEL

    def _resolve_auto_label_source_column(
            self,
            *,
            dataset_id: str,
            session: Mapping[str, Any],
        ) -> str:
            """Resolve the physical source column to read benchmark labels from.

            Prefer the Review source column, then the Start tab's target column.
            """
            columns = set(self._physical_columns_for_dataset(dataset_id))

            candidates: List[str] = []

            explicit = str(self.auto_label_source_column.value or "").strip()
            target = str(self.target_column.value or session.get("target_column") or "").strip()

            if explicit:
                candidates.append(explicit)
            if target and target != explicit:
                candidates.append(target)

            candidates.extend(
                [
                    "target_label",
                    "label",
                    "class_label",
                    "class",
                    "target",
                    "label_name",
                    "target_label_name",
                    "class_name",
                    "target_name",
                    "fine_label",
                    "coarse_label",
                ]
            )

            for candidate in candidates:
                if candidate and candidate in columns:
                    return candidate

                if not candidate:
                    continue

                try:
                    mapped = self.context.datasets.get_mapping(dataset_id, candidate)
                except Exception:
                    mapped = None

                mapped = str(mapped or "").strip()
                if mapped and mapped in columns:
                    return mapped

            raise ValueError(
                "Could not find a physical source label column for auto-labelling. "
                "Choose one in 'Auto-label from dataset column'. "
                f"Available columns: {sorted(columns)}"
            )

    def _restore_widget_state(self, state: Mapping[str, Any]) -> None:
            state = dict(state or {})

            for widget_name, key in (
                ("dataset_select", "dataset_id"),
                ("recipe_select", "recipe_id"),
                ("target_column", "target_column"),
                ("image_column", "image_column"),
                ("mask_column", "mask_column"),
                ("validation_dataset_select", "validation_dataset_id"),
                ("test_dataset_select", "test_dataset_id"),
                ("strategy_select", "strategy_id"),
                ("al_protocol", "al_protocol"),
                ("analytics_x_column", "analytics_x_column"),
                ("analytics_y_column", "analytics_y_column"),
            ):
                widget = getattr(self, widget_name, None)
                value = state.get(key)
                if widget is None or value is None:
                    continue

                try:
                    options = widget.options
                    valid_values = set(options.values()) if isinstance(options, dict) else set(options)
                    if value in valid_values:
                        widget.value = value
                except Exception:
                    pass

            try:
                self._analytics_frame_index = int(state.get("analytics_frame_index") or 0)
            except Exception:
                self._analytics_frame_index = 0

            restored_contract = state.get("session_contract")
            if isinstance(restored_contract, Mapping):
                self._latest_session_contract = dict(restored_contract)

            feature_columns = [
                str(value)
                for value in list(state.get("feature_columns") or [])
                if value is not None
            ]
            if feature_columns and hasattr(self, "feature_columns"):
                try:
                    self._set_feature_columns_value(feature_columns)
                except Exception:
                    pass

            # Restore visible random split sizes from the Start tab.
            if hasattr(self, "validation_fraction"):
                try:
                    self.validation_fraction.value = float(
                        state.get("validation_fraction", self.validation_fraction.value)
                        or 0.0
                    )
                except Exception:
                    pass

            if hasattr(self, "test_fraction"):
                try:
                    self.test_fraction.value = float(
                        state.get("test_fraction", self.test_fraction.value)
                        or 0.0
                    )
                except Exception:
                    pass

            self._refresh_input_contract_ui()
            self._refresh_data_contract_summary()

            labels = al_state.parse_label_options(state.get("label_options"))
            if labels:
                self.label_options_select.options = list(
                    dict.fromkeys([*labels, *self.label_options_select.options])
                )
                self.label_options_select.value = labels

            params = state.get("recipe_params") or {}
            if isinstance(params, Mapping):
                for name, value in params.items():
                    widget = self.recipe_param_widgets.get(str(name)) or self.protocol_widgets.get(str(name))
                    if widget is not None:
                        try:
                            widget.value = value
                        except Exception:
                            pass

            # Start tab owns the AL training contract. Recompute the summary after
            # all restored values have landed.
            try:
                self._refresh_training_contract_summary()
            except Exception:
                pass

            self._mark_analytics_dirty()
            self._refresh_analytics_if_visible()
            self._update_action_gating()

    def _schedule_light_review_refresh(self) -> None:
            """Refresh cheap Review/session UI after the focus has already moved.

            This intentionally does not rebuild analytics. It is scheduled for the
            next UI tick when possible so the user-visible refocus is not blocked by
            tables, summaries, or action-gating updates.
            """

            def apply() -> None:
                if self._disposed:
                    return

                try:
                    self._refresh_session_views()
                    self._set_label_select_options()
                    self._refresh_review_label_for_focus()
                    self._update_action_gating()
                except Exception:
                    # A failed auxiliary refresh should not break the Review hot path.
                    pass

            try:
                curdoc = getattr(pn.state, "curdoc", None)
                if curdoc is not None:
                    curdoc.add_next_tick_callback(apply)
                else:
                    apply()
            except Exception:
                apply()

    def _session_labelled_or_ignored_row_ids(self) -> set[str]:
            """Rows that Review navigation should skip.

            This function intentionally does not inspect the source dataset. It only
            reads the AL session payload, so it is safe to call on every label click.
            """
            session = self._load_current_session()
            if not isinstance(session, Mapping):
                return set()

            skip: set[str] = set()

            def add_many(values: Any) -> None:
                if values is None:
                    return

                if isinstance(values, Mapping):
                    values = values.keys()
                elif isinstance(values, (str, bytes, bytearray, int, float)):
                    values = [values]

                for value in values or []:
                    if value is None:
                        continue
                    text = str(value).strip()
                    if text:
                        skip.add(text)

            add_many(session.get("ignored_row_ids") or [])
            add_many(session.get("training_row_ids") or [])
            add_many(session.get("labels") or {})

            return skip

    def _session_selected(self, event: Any) -> None:
            value = getattr(event, "new", None)
            if value:
                self._current_session_artifact_id = str(value)
                self._analytics_frame_index = 0
                session = self._load_current_session() or {}
                self._apply_session_contract_to_widgets(session)
                self._refresh_predictions()
                self._refresh_review_label_for_focus()
                self._refresh_session_views()
                self._mark_analytics_dirty()
                self._refresh_analytics_if_visible()
                self._update_action_gating()

    def _protocol_changed(self, *_: Any) -> None:
            self._refresh_training_contract_summary()
            self._update_action_gating()

    def _target_column_changed(self, *_: Any) -> None:
            # Changing the Start-tab label column should make Review auto-labelling
            # follow it again unless the user subsequently picks another source.
            self._auto_label_source_user_set = False

            self._refresh_feature_column_options()
            self._refresh_auto_label_source_column()
            self._refresh_labels_from_available_context()
            self._set_label_select_options()
            self._refresh_review_label_for_focus()
            self._refresh_training_contract_summary()

    def _train_clicked(self, *_: Any) -> None:
            try:
                session_artifact_id = self._require_session()
                self._require_labelled()
                recipe_id = str(self.recipe_select.value or "").strip()
                if not recipe_id:
                    raise ValueError("Choose a core.ml recipe.")

                preflight = self._build_session_contract(
                    inspect_images=False,
                    labelled_count=self._labelled_count(self._load_current_session()),
                )
                preflight.raise_for_errors()
                self._remember_preflight(preflight)
                recipe_params = self._recipe_params()

                params = {
                    "session_artifact_id": session_artifact_id,
                    "dataset_id": self.dataset_select.value or "",
                    "recipe_id": recipe_id,
                    "recipe_params": recipe_params,
                    "target_column": self.target_column.value or "al_label",
                    "validation_dataset_id": self.validation_dataset_select.value or "",
                    "test_dataset_id": self.test_dataset_select.value or "",
                    "al_protocol": self.al_protocol.value or "review",
                    "seed": int(self.seed.value or 0),
                    "session_contract": preflight.contract,
                    "auto_predict": bool(self.auto_predict_after_train.value),
                    "auto_query": bool(self.auto_query_after_train.value),
                    "query_strategy_id": self.strategy_select.value or "least_confidence",
                    "query_k": int(self.query_k.value or 200),
                    "make_selection": True,
                }
                params.update(self._protocol_params())

                request = ActionRequest(
                    params=params,
                    origin="core.active_learning.panel",
                )

                self._switch_controls_tab("Train")
                self._set_running(True)
                self._set_status(
                    "Training started. The workflow will train from scratch, predict over the pool, and optionally select the next query batch.",
                    "info",
                )

                manager = getattr(self.context, "plugins", None)
                run_action = getattr(manager, "run_action", None)
                if callable(run_action):
                    self._active_job_handle = run_action(
                        "core.active_learning.train_from_session",
                        self.context,
                        request,
                        on_done=self._on_train_done,
                        on_error=self._on_train_error,
                    )
                else:
                    submit = getattr(getattr(self.context, "jobs", None), "submit", None)
                    if callable(submit):
                        key = f"core.active_learning.train:{session_artifact_id}:{recipe_id}"
                        self._active_job_handle = submit(
                            actions.train_from_session_action,
                            title="Active-learning scratch training",
                            key=key,
                            on_done=self._on_train_done,
                            on_error=self._on_train_error,
                            context=self.context,
                            request=request,
                        )
                    else:
                        result = actions.train_from_session_action(self.context, request)
                        self._on_train_done(result)

            except Exception as exc:
                self._set_running(False)
                self._set_error("Could not start scratch training", exc)

    def _unsure_clicked(self, *_: Any) -> None:
            self._record_current_label(al_state.UNSURE_LABEL)

    def _widget_for_schema(self, name: str, schema: Mapping[str, Any]):
            schema = dict(schema or {})
            kind = str(schema.get("type", "string"))
            default = schema.get("default", "")
            widget_kind = str(schema.get("x-widget") or schema.get("widget") or "")
            lower_name = str(name or "").lower()

            if widget_kind in {"dataset_select", "dataset"} or lower_name.endswith("_dataset_id"):
                options = [""] + self._dataset_ids()
                value = default if default in options else ""
                return pn.widgets.Select(
                    name="",
                    options=options,
                    value=value,
                    sizing_mode="stretch_width",
                )

            if (
                widget_kind
                in {
                    "column_multichoice",
                    "column_multi_choice",
                    "column_multiselect",
                    "column_multi_select",
                    "feature_columns",
                }
                or lower_name
                in {
                    "feature_columns",
                    "input_columns",
                    "features",
                    "x_columns",
                }
            ):
                columns = self._dataset_columns(self.dataset_select.value)
                selected = [
                    column
                    for column in self._normalise_column_list(default)
                    if column in columns
                ]
                return pn.widgets.TextAreaInput(
                    name="",
                    value="\n".join(selected),
                    placeholder="Feature columns, one per line.",
                    height=120,
                    min_height=120,
                    sizing_mode="stretch_width",
                )

            if (
                widget_kind in {"column_select", "dataset_column", "column"}
                or lower_name.endswith("_column")
                or lower_name
                in {
                    "target",
                    "label",
                    "label_column",
                    "target_column",
                    "image_column",
                    "image_path_column",
                    "image_uri_column",
                    "mask_column",
                    "record_id_column",
                    "id_column",
                    "group_column",
                    "split_column",
                }
            ):
                columns = [""] + self._dataset_columns(self.dataset_select.value)
                value = str(default or "")
                return pn.widgets.Select(
                    name="",
                    options=columns,
                    value=value if value in columns else "",
                    sizing_mode="stretch_width",
                )

            if "enum" in schema:
                values = list(schema.get("enum") or [])
                options = {str(v): v for v in values}
                value = default if default in values else (values[0] if values else None)
                return pn.widgets.Select(
                    name="",
                    options=options,
                    value=value,
                    sizing_mode="stretch_width",
                )

            if kind in {"integer", "int"}:
                return pn.widgets.IntInput(
                    name="",
                    value=int(default or 0),
                    start=schema.get("minimum"),
                    end=schema.get("maximum"),
                    sizing_mode="stretch_width",
                )

            if kind in {"number", "float"}:
                return pn.widgets.FloatInput(
                    name="",
                    value=float(default or 0.0),
                    start=schema.get("minimum"),
                    end=schema.get("maximum"),
                    sizing_mode="stretch_width",
                )

            if kind in {"boolean", "bool"}:
                return pn.widgets.Checkbox(
                    name="",
                    value=bool(default),
                    sizing_mode="stretch_width",
                )

            if kind in {"array", "object"}:
                text = json.dumps(
                    default if default not in ("", None) else ([] if kind == "array" else {})
                )
                return pn.widgets.TextAreaInput(
                    name="",
                    value=text,
                    height=100,
                    sizing_mode="stretch_width",
                )

            return pn.widgets.TextInput(
                name="",
                value="" if default is None else str(default),
                placeholder=str(schema.get("description") or ""),
                sizing_mode="stretch_width",
            )

def _make_table():
    empty = pd.DataFrame()

    try:
        return pn.widgets.Tabulator(
            empty,
            pagination="remote",
            page_size=10,
            disabled=True,
            sizing_mode="stretch_width",
            height=240,
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "auto",
            },
        )
    except Exception:
        return pn.widgets.DataFrame(
            empty,
            disabled=True,
            sizing_mode="stretch_width",
            height=240,
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "auto",
            },
        )

def _set_table_value(
    widget: Any,
    df: pd.DataFrame,
) -> None:
    try:
        widget.value = df
    except Exception:
        try:
            widget.object = df
        except Exception:
            pass
