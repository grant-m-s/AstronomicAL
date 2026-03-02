from astronomicAL.extensions import models, query_strategies, feature_generation
from astronomicAL.utils import save_config
from bokeh.models import (
    ColumnDataSource,
    CheckboxButtonGroup,
    DataTable,
    TableColumn,
    TextAreaInput,
)
from datetime import datetime
from functools import partial
from holoviews.operation.datashader import (
    datashade,
    dynspread,
)
from joblib import dump
from modAL.models import ActiveLearner, Committee
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import datashader as ds
import holoviews as hv
import numpy as np
import gc
import os
import pandas as pd
import panel as pn
import json
import sys
import time

class ActiveLearningModel:
    """This class handles the Machine Learning aspect of the codebase.

    Based on the users settings, the required features will be extracted from
    the data and split into train, validation and test sets. The user can
    then specify which classifiers and query functions they would like to
    use in the Active Learning Process. The results at each stage will be
    displayed in various widgets and plots, allowing the user to select the
    correct label for the newly queried source. Each instance will train a
    separate one-vs-rest classifier.

    Parameters
    ----------
    src : ColumnDataSource
        The shared data source which holds the current selected source.
    df : DataFrame
        The shared dataframe which holds all the data.
    label : str
        The string alias of the label that will be the positive case in the
        one-vs-rest classifier.

    Attributes
    ----------
    df : Dataframe
        The shared dataframe which holds all the data.
    src : ColumnDataSource
        The shared data source which holds the current selected source.
    _label : int
        The label that will be the positive case in the one-vs-rest classifier.
    _label_alias : str
        The string alias of `_label`
    _training : bool
        Flag for whether the training process has begun.
    _assigned : bool
        Flag for whether the user has assigned a label to the current queried
        source.
    retrain : bool
        Flag for whether the class is retraining a previous model from within `config.settings["classifiers"]`
    scaler : sklearn.preprocessing.RobustScaler
        The scaler used to standardise features across the train, val and test sets according to the training set.
        NOTE: Only initialised if `config.settings["scale_data"]` is `True`.
    _show_test_results : bool
        Flag for whether to render the test set results column in `panel` method.
    _seen_test_results : bool
        Flag for indicating whether the user has viewed the test results of the classifier.
    _show_caution : bool
        Flag for whether to show the test set caution column to the user when trying to view the test set results.
    _seen_caution : bool
        Flag for indicating whether the user has viewed the test results caution page.
    _max_x : float
        The maximum value of the x axis of the train, val and metric plots. This is set as the `min(μ(x)+4*σ(x), max(x))`.
    _max_y : float
        The maximum value of the y axis of the train, val and metric plots. This is set as the `min(μ(y)+4*σ(y), max(y))`.
    _min_x : float
        The minimum value of the x axis of the train, val and metric plots. This is set as the `max(μ(x)-4*σ(x), min(x))`.
    _min_y : float
        The minimum value of the y axis of the train, val and metric plots. This is set as the `max(μ(y)-4*σ(y), min(y))`.
    _model_output_data_tr : dict
        Dictionary containing the plotting data [`config.settings["default_vars"][0]`,`config.settings["default_vars"][1]`,`metric`,`y`,`pred`] required for the train and metric plots.
    _model_output_data_val : dict
        Dictionary containing the plotting data [`config.settings["default_vars"][0]`,`config.settings["default_vars"][1]`,`y`,`pred`] required for the val plot.
    _accuracy_list : dict
        Dictionary containing the train and validation vs number of points accuracy scores.
    _f1_list : dict
        Dictionary containing the train and validation vs number of points f1 scores.
    _precision_list : dict
        Dictionary containing the train and validation vs number of points precision scores.
    _recall_list : dict
        Dictionary containing the train and validation vs number of points recall scores.
    _train_scores : dict
        Dictionary containing the current scores for the training set.
    _val_scores : dict
        Dictionary containing the current scores for the validation set.
    _test_scores : dict
        Dictionary containing the current scores for the testing set.
    corr_train : ColumnDataSource
        The `config.settings["default_vars"][0]` and `config.settings["default_vars"][1]` values of all the training sources that are currently predicted correctly.
    incorr_train : ColumnDataSource
        The `config.settings["default_vars"][0]` and `config.settings["default_vars"][1]` values of all the training sources that are currently predicted incorrectly.
    corr_val : ColumnDataSource
        The `config.settings["default_vars"][0]` and `config.settings["default_vars"][1]` values of all the validation sources that are currently predicted correctly.
    incorr_val : ColumnDataSource
        The `config.settings["default_vars"][0]` and `config.settings["default_vars"][1]` values of all the validation sources that are currently predicted incorrectly.
    queried_points : ColumnDataSource
        The `config.settings["default_vars"][0]` and `config.settings["default_vars"][1]` values of the current queried points.
    full_labelled_data : dict
        Dictionary containing the `id` and `y` values of all labelled points during training.
    assign_label_group : Panel RadioButtonGroup Widget
        The group of buttons containing the possible labels for labelling during training.
    assign_label_button : Panel Button Widget
        The button for assigning the selected label from `assign_label_group` to the currently queried source.
    show_queried_button : Panel Button Widget
        The button for making the current queried point the current selected point.
    classifier_dropdown : Panel Select Widget
        A dropdown menu showing all the classifiers initialised in `astronomicAL.extensions.models`.
    query_strategy_dropdown : Panel Select Widget
        A dropdown menu showing all the query strategies initialised in `astronomicAL.extensions.query_strategies`.
    starting_num_points : Panel IntInput Widget
        Set the number of initial randomly selected points to train on.
    classifier_table_source : ColumnDataSource
        The collection of all the currently selected classifier and query strategy pairs.
    classifier_table : DataTable
        The table for visualising `classifier_table_source`
    add_classifier_button : Panel Button Widget
        The button for appending the currently selected values from `classifier_dropdown` and `query_strategy_dropdown` to `classifier_table_source`.
    remove_classifier_button : Panel Button Widget
        The button for removing the last entry from `classifier_table_source`.
    start_training_button : Panel Button Widget
        The button for beginning the training of a classifier using the selected parameters from `classifier_table_source` and `starting_num_points`.
    next_interation_button : Panel Button Widget
        The button to begin the next iteration of the Active Learning process. Only visible after assigning a label to the currently queried point.
    checkpoint_button : Panel Button Widget
        The button to save the current model and parameters required to recreate current set up.
    request_test_results_button : Panel Button Widget
        The button to request the current classifiers results for the test set.
    _return_to_train_view_button : Panel Button Widget
        The button displayed in the test set caution window, allowing the user to return to the train and validation results without seeing the test set results.
    _stop_caution_show_checkbox : Panel Checkbox Widget
        A checkbox for whether the user wants to disable the test set caution window from appearing when they want to view the test set.
    _view_test_results_button : Panel Button Widget
        The button to show the test set results to the user. If `_show_caution` is `True`, this button will show the test set caution window instead.
    _queried_is_selected : bool
        Flag for whether the current queried point is also the current selected point.
    setup_row : Panel Row
        A row containing all the classifier setup settings required before the training process has begun.
    panel_row : Panel Row
        A row containing all the visualisation aspects of the ActiveLearningModel view.
    conf_mat_tr_tn : str
        The current number of true negatives in the classifiers current prediction of the training set.
    conf_mat_tr_fn : str
        The current number of false negatives in the classifiers current prediction of the training set.
    conf_mat_tr_fp : str
        The current number of false positives in the classifiers current prediction of the training set.
    conf_mat_tr_tp : str
        The current number of true positives in the classifiers current prediction of the training set.
    conf_mat_val_tn : str
        The current number of true negatives in the classifiers current prediction of the validation set.
    conf_mat_val_fn : str
        The current number of false negatives in the classifiers current prediction of the validation set.
    conf_mat_val_fp : str
        The current number of false positives in the classifiers current prediction of the validation set.
    conf_mat_val_tp : str
        The current number of true positives in the classifiers current prediction of the validation set.
    conf_mat_test_tn : str
        The current number of true negatives in the classifiers current prediction of the test set.
    conf_mat_test_fn : str
        The current number of false negatives in the classifiers current prediction of the test set.
    conf_mat_test_fp : str
        The current number of false positives in the classifiers current prediction of the test set.
    conf_mat_test_tp : str
        The current number of true positives in the classifiers current prediction of the test set.
    all_al_data : DataFrame
        A dataframe containing a subset of `df` with only the required features for training.
    x_train : DataFrame
        A dataframe containing all the training input data.
    y_train : DataFrame
        A dataframe containing all the training labels.
    id_train : DataFrame
        A dataframe containing all the training ids.
    x_val : DataFrame
        A dataframe containing all the validation input data.
    y_val : DataFrame
        A dataframe containing all the validation labels.
    id_val : DataFrame
        A dataframe containing all the validation ids.
    x_test : DataFrame
        A dataframe containing all the test input data.
    y_test : DataFrame
        A dataframe containing all the test labels.
    id_test : DataFrame
        A dataframe containing all the test ids.
    x_al_train : Numpy Array
        The data that the classifier is training on.
    y_al_train : Numpy Array
        The labels for the data the classifier is training on.
    id_al_train : DataFrame
        The ids of the data the classifier is training on.
    x_pool : Numpy Array
        The data of the sources in the pool that are available to query from.
    y_pool : Numpy Array
        The labels of the sources in the pool that are available to query from.
    id_pool : DataFrame
        The ids of the sources in the pool that are available to query from.
    query_index : int
        The current index of `x_pool` that contains the current queried point.
    learner : ModAL ActiveLearner
        The current classifier that is being trained. If multiple classifiers exist in `classifier_table_source`, then `learner` will be a ModAL Committee.
    """

    def __init__(self, src, df, label, context=None):

        self.context = context
        
        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config
        self.df = self.config.main_df

        self.src = src
        self.watch_bokeh(self.src, "data", self._panel_cb)

        self._label = self.config.settings["strings_to_labels"][label]
        self._label_alias = label

        self._training = False
        self._assigned = False
        self.retrain = False

        self._ui_built = False

        if "config_load_level" in list(self.config.settings.keys()):
            if (self.config.settings["config_load_level"] == 2) and (
                f"{self._label}" in self.config.settings["classifiers"]
            ):
                keys = list(self.config.settings["classifiers"][f"{self._label}"].keys())
                if ("y" in keys) and ("id" in keys):
                    self.retrain = True

        if len(self.config.ml_data.keys()) == 0:
            print("preprocessing...")
            self._preprocess_data()

            self.x_train_without_unknowns = self.config.ml_data["x_train_without_unknowns"]

        elif len(self.config.ml_data["x_train_with_unknowns"].columns) == 0:
            print("preprocessing...")
            self._preprocess_data()

            self.x_train_without_unknowns = self.config.ml_data["x_train_without_unknowns"]
        else:

            self.x_train_without_unknowns = self.config.ml_data["x_train_without_unknowns"]

            self.y_train_without_unknowns = self.config.ml_data["y_train_without_unknowns"]
            self.y_train_with_unknowns = self.config.ml_data["y_train_with_unknowns"]
            self.y_val = self.config.ml_data["y_val"]
            self.y_test = self.config.ml_data["y_test"]

            self.id_train_with_unknowns = self.config.ml_data["id_train_with_unknowns"]
            self.id_train_without_unknowns = self.config.ml_data["id_train_without_unknowns"]
            self.id_val = self.config.ml_data["id_val"]
            self.id_test = self.config.ml_data["id_test"]

            if self.config.settings["scale_data"]:

                self.scaler = self.config.ml_data["scaler"]

            self.df = self.config.main_df

        self._convert_to_one_vs_rest()

        self._construct_panel()

        self._initialise_placeholders()

        self._resize_plot_scales()

        self._show_test_results = False
        self._seen_test_results = False
        self._show_caution = True
        self._seen_caution = False

        self._build_ui_once()
        
        if self.retrain:
            self._start_training_cb(None)


    def _dispose_impl(self):
        """Subclass-specific cleanup hook (override if needed)."""
        return

    def dispose(self):
        if getattr(self, "_disposed", False):
            return
        self._disposed = True

        # 1) subclass cleanup first
        try:
            self._dispose_impl()
        except Exception:
            pass

        # 2) unwatch bokeh callbacks (including src.on_change if registered via watch_bokeh)
        try:
            self.unwatch_all_bokeh()
        except Exception:
            pass

        # 3) Unsubscribe EventBus
        if self.context and getattr(self.context, "events", None):
            for sub in list(getattr(self, "_event_subs", [])):
                try:
                    self.context.events.unsubscribe(sub)
                except Exception:
                    pass
        self._event_subs = []

        # 4) Stop periodic callbacks
        for cb in list(getattr(self, "_periodic_cbs", [])):
            try:
                cb.stop()
            except Exception:
                pass
        self._periodic_cbs = []

    def watch_bokeh(self, model, attr: str, callback):
        """Register and track Bokeh model.on_change callbacks for unified disposal."""
        if model is None:
            return
        try:
            model.on_change(attr, callback)
            self._bokeh_on_change.append((model, attr, callback))
        except Exception:
            pass

    def unwatch_all_bokeh(self):
        """Remove all tracked Bokeh callbacks (idempotent)."""
        for model, attr, callback in list(getattr(self, "_bokeh_on_change", [])):
            try:
                model.remove_on_change(attr, callback)
            except Exception as e:
                # Ignore double-remove noise
                if "list.remove(x): x not in list" in str(e):
                    pass
            # continue regardless
        self._bokeh_on_change = []

    def _resize_plot_scales(self):

        x_axis = self.config.settings["default_vars"][0]
        y_axis = self.config.settings["default_vars"][1]

        x_sd = np.std(self.config.ml_data["x_train_without_unknowns"][x_axis])
        x_mu = np.mean(self.config.ml_data["x_train_without_unknowns"][x_axis])
        y_sd = np.std(self.config.ml_data["x_train_without_unknowns"][y_axis])
        y_mu = np.mean(self.config.ml_data["x_train_without_unknowns"][y_axis])

        x_max = x_mu + 4 * x_sd
        x_min = x_mu - 4 * x_sd

        y_max = y_mu + 4 * y_sd
        y_min = y_mu - 4 * y_sd

        self._max_x = np.min(
            [(x_max), np.max(self.config.ml_data["x_train_without_unknowns"][x_axis])]
        )
        self._min_x = np.max(
            [(x_min), np.min(self.config.ml_data["x_train_without_unknowns"][x_axis])]
        )

        self._max_y = np.min(
            [(y_max), np.max(self.config.ml_data["x_train_without_unknowns"][y_axis])]
        )
        self._min_y = np.max(
            [(y_min), np.min(self.config.ml_data["x_train_without_unknowns"][y_axis])]
        )

    def _initialise_placeholders(self):

        if (
            not self.config.settings["default_vars"][0]
            in self.config.ml_data["x_train_without_unknowns"].keys()
        ):
            self.config.settings["default_vars"] = (
                self.config.settings["features_for_training"][0],
                self.config.settings["default_vars"][1],
            )

        if (
            not self.config.settings["default_vars"][1]
            in self.config.ml_data["x_train_without_unknowns"].keys()
        ):
            self.config.settings["default_vars"] = (
                self.config.settings["default_vars"][0],
                self.config.settings["features_for_training"][1],
            )

        self._model_output_data_tr = {
            f'{self.config.settings["default_vars"][0]}': [],
            f'{self.config.settings["default_vars"][1]}': [],
            "metric": [],
            "y": [],
            "pred": [],
        }

        self._model_output_data_val = {
            f'{self.config.settings["default_vars"][0]}': [],
            f'{self.config.settings["default_vars"][1]}': [],
            "y": [],
            "pred": [],
        }

        self._accuracy_list = {
            "train": {"score": [], "num_points": []},
            "val": {"score": [], "num_points": []},
        }
        self._f1_list = {
            "train": {"score": [], "num_points": []},
            "val": {"score": [], "num_points": []},
        }
        self._precision_list = {
            "train": {"score": [], "num_points": []},
            "val": {"score": [], "num_points": []},
        }
        self._recall_list = {
            "train": {"score": [], "num_points": []},
            "val": {"score": [], "num_points": []},
        }

        self._train_scores = {"acc": 0.00, "prec": 0.00, "rec": 0.00, "f1": 0.00}
        self._val_scores = {"acc": 0.00, "prec": 0.00, "rec": 0.00, "f1": 0.00}
        self._test_scores = {"acc": 0.00, "prec": 0.00, "rec": 0.00, "f1": 0.00}

        self.corr_train = ColumnDataSource(self._empty_data())
        self.incorr_train = ColumnDataSource(self._empty_data())
        self.corr_val = ColumnDataSource(self._empty_data())
        self.incorr_val = ColumnDataSource(self._empty_data())

        self.queried_points = ColumnDataSource(self._empty_data())

        self.id_al_train = []

        self.full_labelled_data = {"id": [], "y": []}

    def _construct_panel(self):

        options = []

        all_labels = list(self.config.main_df[self.config.settings["label_col"]].unique())

        all_labels.sort()

        if -1 in all_labels:
            all_labels.remove(-1)

        if self.config.settings["exclude_labels"]:
            for i in self.config.settings["unclassified_labels"]:
                all_labels.remove(self.config.settings["strings_to_labels"][f"{i}"])

        for i in all_labels:
            options.append(self.config.settings["labels_to_strings"][f"{i}"])

        options.append("Unsure")
        self.assign_label_group = pn.widgets.RadioButtonGroup(
            name="Label button group",
            options=options,
            sizing_mode="stretch_width",
        )

        self.assign_label_group.value = "Unsure"

        self.assign_label_group.css_classes = ["al-labels"]

        self.assign_label_button = pn.widgets.Button(
            name="Assign Label",
            button_type="primary",
            width=125,
            sizing_mode="fixed",
        )
        self.assign_label_button.on_click(self._assign_label_cb)

        self.show_queried_button = pn.widgets.Button(
            name="Select Current Queried Point", sizing_mode="stretch_width"
        )
        self.show_queried_button.on_click(self._show_queried_point_cb)

        self.classifier_dropdown = pn.widgets.Select(
            name="Classifier",
            options=list(models.get_classifiers().keys()),
        )
        self.query_strategy_dropdown = pn.widgets.Select(
            name="Query Strategy",
            options=list(query_strategies.get_strategy_dict().keys()),
        )
        self.starting_num_points = pn.widgets.IntInput(
            name="How many initial points?", value=5, step=1, start=3
        )

        self.classifier_table_source = ColumnDataSource(dict(classifier=[], query=[]))
        table_column = [
            TableColumn(field="classifier", title="classifier"),
            TableColumn(field="query", title="query"),
        ]

        self.classifier_table = DataTable(
            source=self.classifier_table_source,
            columns=table_column,
            height=120,
        )

        if "classifiers" in self.config.settings.keys():
            if str(self._label) in self.config.settings["classifiers"].keys():
                list_c1 = self.classifier_table_source.data["classifier"]
                list_c2 = self.classifier_table_source.data["query"]

                imported_classifiers = self.config.settings["classifiers"][f"{self._label}"][
                    "classifier"
                ]
                imported_querys = self.config.settings["classifiers"][f"{self._label}"][
                    "query"
                ]

                assert len(imported_classifiers) == len(imported_querys)

                for i in range(len(imported_classifiers)):

                    list_c1.append(imported_classifiers[i])
                    list_c2.append(imported_querys[i])

                self.classifier_table_source.data = {
                    "classifier": list_c1,
                    "query": list_c2,
                }

        self.add_classifier_button = pn.widgets.Button(name=">>", max_height=30)
        self.remove_classifier_button = pn.widgets.Button(name="<<", max_height=30)

        self.add_classifier_button.on_click(self._add_classifier_cb)
        self.remove_classifier_button.on_click(self._remove_classifier_cb)

        self.start_training_button = pn.widgets.Button(
            name="Start Training", max_height=30
        )
        self.start_training_button.on_click(self._start_training_cb)

        self.next_iteration_button = pn.widgets.Button(name="Next Iteration")
        self.next_iteration_button.on_click(self._next_iteration_cb)

        trigger_for_autosave = TextAreaInput(value="")
        trigger_for_autosave.on_change(
            "value",
            partial(
                save_config.save_config_file_cb,
                trigger_text=trigger_for_autosave,
                autosave=True,
                context=self.context
            ),
        )

        trigger_for_checkout = TextAreaInput(value="")
        trigger_for_checkout.on_change(
            "value",
            partial(
                save_config.save_config_file_cb,
                trigger_text=trigger_for_checkout,
                autosave=False,
                context=self.context
            ),
        )

        self.next_iteration_button.jscallback(
            clicks=save_config.save_layout_js_cb,
            args=dict(text_area_input=trigger_for_autosave),
        )

        self.checkpoint_button = pn.widgets.Button(
            name="Checkpoint", sizing_mode="stretch_width"
        )
        self.checkpoint_button.on_click(self._checkpoint_cb)
        self.checkpoint_button.jscallback(
            clicks=save_config.save_layout_js_cb,
            args=dict(text_area_input=trigger_for_checkout),
        )

        self.request_test_results_button = pn.widgets.Button(
            name="View Test Results", button_type="warning", max_width=125
        )
        self.request_test_results_button.on_click(self._request_test_results_cb)

        self._return_to_train_view_button = pn.widgets.Button(name="<< Go Back")
        self._return_to_train_view_button.on_click(self._return_to_train_cb)

        self._stop_caution_show_checkbox = pn.widgets.Checkbox(
            name="Don't show this message again for this classifier."
        )

        self._view_test_results_button = pn.widgets.Button(
            name="Continue to Test Data", button_type="danger"
        )
        self._view_test_results_button.on_click(self._show_test_results_cb)

        self._queried_is_selected = False

        self._train_tab_colour_switch = CheckboxButtonGroup(
            labels=["Show Incorrect", "Show Correct"],
            active=[0, 1],
            max_height=35,
            height=35,
            # sizing_mode="fixed",
        )

        self._train_tab_colour_switch.on_change("active", self._update_tab_plots_cb)

        self._val_tab_colour_switch = CheckboxButtonGroup(
            labels=["Show Incorrect", "Show Correct"],
            active=[0, 1],
            max_height=35,
            height=35,
            # sizing_mode="fixed",
        )
        self._val_tab_colour_switch.on_change("active", self._update_tab_plots_cb)

        self.active_tab = 0

        self.setup_row = pn.Column(sizing_mode="stretch_width", margin=(0,0,0,0))
        self.panel_row = pn.Column(sizing_mode="stretch_both", margin=(0,0,0,0))

        self.setup_row.sizing_mode = "stretch_width"
        self.setup_row.height_policy = "min"
        self.setup_row.margin = (0, 0, 0, 0)

        self.conf_mat_tr_tn = "TN"
        self.conf_mat_tr_fn = "FN"
        self.conf_mat_tr_fp = "FP"
        self.conf_mat_tr_tp = "TP"
        self.conf_mat_val_tn = "TN"
        self.conf_mat_val_fn = "FN"
        self.conf_mat_val_fp = "FP"
        self.conf_mat_val_tp = "TP"
        self.conf_mat_test_tn = "TN"
        self.conf_mat_test_fn = "FN"
        self.conf_mat_test_fp = "FP"
        self.conf_mat_test_tp = "TP"

    def _preprocess_data(self):

        self.df, self.all_al_data = self.generate_features(self.df)

        x, y = self.split_x_y_ids(self.all_al_data)

        excluded_x = {}
        excluded_y = {}
        if self.config.settings["exclude_labels"]:
            for label in self.config.settings["unclassified_labels"]:
                (
                    x,
                    y,
                    excluded_x[f"{label}"],
                    excluded_y[f"{label}"],
                ) = self.exclude_unclassified_labels(x, y, label)

        if "-1" in self.config.settings["labels_to_strings"].keys():
            print("removing -1")
            label = self.config.settings["labels_to_strings"]["-1"]
            (
                x,
                y,
                excluded_x[f"{label}"],
                excluded_y[f"{label}"],
            ) = self.exclude_unclassified_labels(x, y, label)

        (
            self.x_train,
            y_train,
            self.x_val,
            y_val,
            self.x_test,
            y_test,
        ) = self.train_val_test_split(x, y, excluded_x, excluded_y, 0.6, 0.2)

        if "exclude_unknown_labels" in self.config.settings.keys():
            if not self.config.settings["exclude_unknown_labels"]:
                if "-1" in self.config.settings["labels_to_strings"].keys():
                    if self.config.settings["labels_to_strings"]["-1"] in excluded_x.keys():
                        self.x_train = self.x_train.append(
                            excluded_x[self.config.settings["labels_to_strings"]["-1"]],
                            ignore_index=True,
                        )
                        y_train = y_train.append(
                            excluded_y[self.config.settings["labels_to_strings"]["-1"]],
                            ignore_index=True,
                        )

        x_cols = list(self.x_train.columns)
        y_cols = list(y_train.columns)

        if "index" in x_cols:
            x_cols.remove("index")

        if "index" in y_cols:
            y_cols.remove("index")

        if "level_0" in x_cols:
            x_cols.remove("level_0")

        if "level_0" in y_cols:
            y_cols.remove("level_0")

        assert not "index" in x_cols
        assert not "index" in y_cols
        assert not "level_0" in x_cols
        assert not "level_0" in y_cols

        self.x_cols = x_cols
        self.y_cols = y_cols

        print(f"train: {y_train[self.config.settings['label_col']].value_counts()}")
        print(f"val: {y_val[self.config.settings['label_col']].value_counts()}")
        print(f"test: {y_test[self.config.settings['label_col']].value_counts()}")

        if self.config.settings["scale_data"]:

            (self.x_train, self.x_val, self.x_test,) = self.scale_data(
                self.x_train,
                self.x_val,
                self.x_test,
                self.x_cols,
            )

        (
            self.y_train,
            self.id_train,
            self.y_val,
            self.id_val,
            self.y_test,
            self.id_test,
        ) = self.split_y_ids(y_train, y_val, y_test)

        self.assign_global_data()


    def assign_global_data(self):
        """Assign the current train, validation and test sets to the shared
         `ml_data` dictionary so that it can be used by other classifiers.

        Returns
        -------
        None

        """

        self.x_train = self.x_train[self.x_cols]
        self.x_val = self.x_val[self.x_cols]
        self.x_test = self.x_test[self.x_cols]

        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.y_train[self.config.settings["label_col"]] != -1)

        self.config.ml_data["x_train_without_unknowns"] = self.x_train[
            self.y_train[self.config.settings["label_col"]] != -1
        ]

        self.x_train_without_unknowns = self.x_train[
            self.y_train[self.config.settings["label_col"]] != -1
        ]

        self.config.ml_data["x_train_with_unknowns"] = self.x_train
        self.config.ml_data["x_val"] = self.x_val
        self.config.ml_data["x_test"] = self.x_test

        self.config.ml_data["y_train_without_unknowns"] = self.y_train[
            self.y_train[self.config.settings["label_col"]] != -1
        ]
        self.config.ml_data["y_train_with_unknowns"] = self.y_train
        self.config.ml_data["y_val"] = self.y_val
        self.config.ml_data["y_test"] = self.y_test

        self.config.ml_data["id_train_without_unknowns"] = self.id_train[
            self.y_train[self.config.settings["label_col"]] != -1
        ]
        self.config.ml_data["id_train_with_unknowns"] = self.id_train
        self.config.ml_data["id_val"] = self.id_val
        self.config.ml_data["id_test"] = self.id_test

        if self.config.settings["scale_data"]:
            self.config.ml_data["scaler"] = self.scaler

        self.config.main_df = self.df

        if self.context and self.context.events:
            self.context.events.publish(
                "dataset.main.updated",
                {"reason": "preprocess", "ncols": self.df.shape[1]},
            )

    def remove_from_pool(self, id=None):
        """Remove the current queried source from the active learning pool.

        Returns
        -------
        None

        """

        if id is None:
            index = self.query_index
        else:
            ind_list = list(self.id_pool.index.values)
            df_index = self.id_pool.index[
                self.id_pool[self.config.settings["id_col"]] == id
            ].to_list()[0]
            index = ind_list.index(df_index)

        self.x_pool = np.delete(
            self.x_pool,
            index,
            0,
        )

        self.y_pool = np.delete(
            self.y_pool,
            index,
            0,
        )

        self.id_pool = self.id_pool.drop(self.id_pool.index[index])

    def save_model(self, checkpoint=False):
        """Save the current classifier(s) as a joblib file to the models/
        directory. The classifier filename will include the classifier(s) used
        and corresponding query function(s). If training a committee, a new
        directory will be created where each of the committee modals will be
        saved.

        Parameters
        ----------
        checkpoint : bool, default = False
            Flag whether or not the model is saving a checkpoint. If `True` the
            filename will include the current size of the training set, the
            current validation F1-score as well as the time and date to allow
            for easy tracking and organisation of models.

        Returns
        -------
        None

        """
        list_c1 = self.classifier_table_source.data["classifier"]
        list_c2 = self.classifier_table_source.data["query"]

        clfs_shortened = ""

        for i in range(len(list(list_c1))):
            clf = f"{list_c1[i][:4]}_{list_c2[i][:4]}_"
            clfs_shortened += clf

        clfs_shortened = clfs_shortened[:-1]
        iteration = self.curr_num_points
        val_f1 = int(float(self._val_scores["f1"]) * 100)

        dir = "models/"
        if not os.path.isdir(dir):
            os.mkdir(dir)

        filename = f"{dir}{self._label}-{clfs_shortened}"

        if self.committee:
            now = datetime.now()
            dt_string = now.strftime("%Y%m%d_%H:%M:%S")
            for i, clf in enumerate(self.learner):
                model = clf

                if checkpoint:
                    mod_dir = f"{filename}-{iteration}-{val_f1}-{dt_string}"
                    if not os.path.isdir(mod_dir):
                        os.mkdir(mod_dir)
                    scaler_dir = f"{mod_dir}/SCALER"
                    mod_dir = f"{mod_dir}/{list_c1[i][:6]}_{i}"
                    dump(model, f"{mod_dir}.joblib")

                    if not os.path.isfile(f"{scaler_dir}.joblib") and (
                        self.config.settings["scale_data"]
                    ):
                        dump(self.scaler, f"{scaler_dir}.joblib")

                else:
                    if not os.path.isdir(filename):
                        os.mkdir(filename)
                    dump(model, f"{filename}/{list_c1[i][:6]}_{i}.joblib")
                    scaler_dir = f"{filename}/SCALER"
                    if not os.path.isfile(f"{scaler_dir}.joblib") and (
                        self.config.settings["scale_data"]
                    ):
                        dump(self.scaler, f"{scaler_dir}.joblib")

        else:
            model = self.learner.estimator
            if checkpoint:
                now = datetime.now()
                dt_string = now.strftime("%Y%m%d_%H:%M:%S")
                dump(model, f"{filename}-{iteration}-{val_f1}-{dt_string}.joblib")
                scaler_dir = f"{filename}-{iteration}-{val_f1}-{dt_string}-SCALER"
                if not os.path.isfile(f"{scaler_dir}.joblib") and (
                    self.config.settings["scale_data"]
                ):
                    dump(self.scaler, f"{scaler_dir}.joblib")
            else:
                dump(model, f"{filename}.joblib")
                scaler_dir = f"{filename}-SCALER"
                if not os.path.isfile(f"{scaler_dir}.joblib") and (
                    self.config.settings["scale_data"]
                ):
                    dump(self.scaler, f"{scaler_dir}.joblib")

    def _checkpoint_cb(self, event):

        self.checkpoint_button.disabled = True
        self.checkpoint_button.name = "Saved!"
        self.save_model(checkpoint=True)

    def _show_queried_point_cb(self, event):
        self.show_queried_point()

    def show_queried_point(self):
        """Assign the classifier's current queried point as the current selected
        source.

        Returns
        -------
        None

        """

        query_instance = self.query_instance
        query_idx = self.query_index

        data = self.df

        queried_id = self.id_pool.iloc[query_idx][self.config.settings["id_col"]].copy()

        act_label = self.y_pool[query_idx]

        selected_source = self.df[
            self.df[self.config.settings["id_col"]] == queried_id.values[0]
        ]

        selected_dict = selected_source.set_index(self.config.settings["id_col"]).to_dict(
            "list"
        )

        selected_dict[self.config.settings["id_col"]] = [queried_id.values[0]]

        try:
            self.src.data = selected_dict
        except ValueError:
            print(
                "Please make sure that your data does not contain any Inf or NaN values."
            )

        plot_idx = [
            list(self.config.ml_data["x_train_without_unknowns"].columns).index(
                self.config.settings["default_vars"][0]
            ),
            list(self.config.ml_data["x_train_without_unknowns"].columns).index(
                self.config.settings["default_vars"][1]
            ),
        ]

        q = {
            f'{self.config.settings["default_vars"][0]}': query_instance[:, plot_idx[0]],
            f'{self.config.settings["default_vars"][1]}': [query_instance[:, plot_idx[1]]],
        }

        self.queried_points.data = q

    def iterate_AL(self):
        """Iterate through one iteration of active learning.

        Returns
        -------
        None

        """

        # self.assign_label = False

        self.remove_from_pool()

        self.curr_num_points = self.x_al_train.shape[0]

        print("fitting...")
        self.learner.fit(self.x_al_train, self.y_al_train)

        self.save_model(checkpoint=False)

        self._update_predictions()

        self.query_new_point()

        self.show_queried_point()

        self.assign_label_button.name = "Assign"

    def query_new_point(self):
        """Query the most informative point from the training pool based off the
        chosen query metric.

        Returns
        -------
        None

        """
        self.query_index, self.query_instance = self.learner.query(self.x_pool)

    def _assign_label_cb(self, event):

        selected_label = self.assign_label_group.value

        query = self.query_instance
        query_idx = self.query_index

        if not selected_label == "Unsure":

            self.full_labelled_data["id"].append(
                list(self.id_pool.iloc[query_idx].values)[0][0]
            )
            self.full_labelled_data["y"].append(
                self.config.settings["strings_to_labels"][selected_label]
            )

            self._assigned = True
            self.next_iteration_button.name = "Next Iteration"

            if self.assign_label_button.name == "Assigned!":
                self._refresh_ui()
                return

            self.assign_label_button.name = "Assigned!"

            if int(self.config.settings["strings_to_labels"][selected_label]) == self._label:
                selected_label = 1
            else:
                selected_label = 0

            selected_label = np.array([selected_label])

            new_train = np.vstack((self.x_al_train, query))

            new_label = np.concatenate((self.y_al_train, selected_label))

            new_id = self.id_al_train.append(self.id_pool.iloc[query_idx])

            self.x_al_train = new_train
            self.y_al_train = new_label
            self.id_al_train = new_id

        else:

            self.full_labelled_data["id"].append(
                list(self.id_pool.iloc[query_idx].values)[0][0]
            )
            self.full_labelled_data["y"].append(-1)

            self.assign_label_button.name = "Querying..."
            self.assign_label_button.disabled = True

            self.remove_from_pool()
            self.query_new_point()
            self.show_queried_point()
            self.assign_label_button.name = "Assign"
            self.assign_label_button.disabled = False
            self._refresh_ui()

        self.config.settings["classifiers"][f"{self._label}"][
            "id"
        ] = self.full_labelled_data["id"]
        self.config.settings["classifiers"][f"{self._label}"]["y"] = self.full_labelled_data[
            "y"
        ]

        assert self.x_al_train.shape[0] == len(
            self.y_al_train
        ), f"AL_TRAIN & LABELS NOT EQUAL - {self.x_al_train.shape[0]}|{len(self.y_al_train)}"

        assert len(self.y_al_train) == len(
            self.id_al_train
        ), f"AL_LABELS & IDs NOT EQUAL - {len(self.y_al_train)}|{len(self.id_al_train)}"

        self._refresh_ui()

    def _empty_data(self):

        empty = {
            f'{self.config.settings["default_vars"][0]}': [],
            f'{self.config.settings["default_vars"][1]}': [],
        }

        return empty

    def _next_iteration_cb(self, event):

        self.assign_label_group.value = "Unsure"

        self.next_iteration_button.disabled = True

        self.next_iteration_button.name = "Training..."

        self.iterate_AL()

        self.next_iteration_button.name = "Querying..."

        self.checkpoint_button.name = "Checkpoint"
        self._assigned = False

        self.next_iteration_button.disabled = False
        self._refresh_ui_full()
        self.checkpoint_button.disabled = False

    def _start_training_cb(self, event):
        table = self.classifier_table_source.data

        if len(table["classifier"]) == 0:
            print("No Classifiers Selected.")
            self.start_training_button.disabled = True
            self.start_training_button.name = "No Classifiers Selected"
            time.sleep(1.5)
            self.start_training_button.disabled = False
            self.start_training_button.name = "Start Training"
            return

        self._training = True
        self.start_training_button.name = "Beginning Training..."
        self.start_training_button.disabled = True
        self.add_classifier_button.disabled = True
        self.remove_classifier_button.disabled = True
        self.num_points_list = []
        self.curr_num_points = self.starting_num_points.value

        if "classifiers" not in self.config.settings.keys():
            self.config.settings["classifiers"] = {}

        if self.retrain:
            self.curr_num_points = len(
                self.config.settings["classifiers"][f"{self._label}"]["y"]
            )

        if f"{self._label}" not in self.config.settings["classifiers"]:
            self.config.settings["classifiers"][f"{self._label}"] = {}

        self.config.settings["classifiers"][f"{self._label}"]["classifier"] = table[
            "classifier"
        ]
        self.config.settings["classifiers"][f"{self._label}"]["query"] = table["query"]

        self.setup_learners()

        self.query_index, self.query_instance = self.learner.query(self.x_pool)

        self.show_queried_point()

        if self.retrain:
            empty = {}
            for key in list(self.src.data.keys()):
                empty[key] = []

            self.src.data = empty

        self._refresh_ui_full()

    def _add_classifier_cb(self, event):

        clf = self.classifier_dropdown.value
        qs = self.query_strategy_dropdown.value
        list_c1 = self.classifier_table_source.data["classifier"]
        list_c2 = self.classifier_table_source.data["query"]

        list_c1.append(clf)
        list_c2.append(qs)

        self.classifier_table_source.data = {
            "classifier": list_c1,
            "query": list_c2,
        }

    def _remove_classifier_cb(self, event):

        list_c1 = self.classifier_table_source.data["classifier"]
        list_c2 = self.classifier_table_source.data["query"]

        list_c1 = list_c1[:-1]
        list_c2 = list_c2[:-1]

        self.classifier_table_source.data = {
            "classifier": list_c1,
            "query": list_c2,
        }

    def split_x_y_ids(self, df_data):
        """Separate the data into X and [y,ids] dataframes.

        Parameters
        ----------
        df_data : DataFrame
            A dataframe containing all the training features, the label column
            and the id column.

        Returns
        -------
        df_data_x : DataFrame
            A dataframe containing only the features used for machine learning.
        df_data_y_ids : DataFrame
            A dataframe containing only the label and id columns corresponding
            to `df_data_x`.

        """

        df_data_y_ids = df_data[
            [self.config.settings["label_col"], self.config.settings["id_col"]]
        ]
        df_data_x = df_data.drop(
            columns=[self.config.settings["label_col"], self.config.settings["id_col"]]
        )
        assert (
            df_data_y_ids.shape[0] == df_data_x.shape[0]
        ), f"df_data_y_ids has different number of rows than df_data_x, {df_data_y_ids.shape[0]} != {df_data_x.shape[0]}"

        return df_data_x, df_data_y_ids

    def exclude_unclassified_labels(self, df_data_x, df_data_y, excluded):
        """Remove any sources that have a label that is not being trained on.

        Parameters
        ----------
        df_data_x : DataFrame
            A dataframe containing only the features used for machine learning.
        df_data_y : DataFrame
            A dataframe containing the label corresponding to `df_data_x`.
        excluded : str
            The label which should be removed from `df_data_x` and `df_data_y`.

        Returns
        -------
        data_x : DataFrame
            A subset of `df_data_x` which has had all rows with label `excluded`
            removed.
        data_y : DataFrame
            A subset of `df_data_y` which has had all rows with label `excluded`
            removed.
        excluded_x : DataFrame
            A subset of `df_data_x` which only has rows with label `excluded`.
        excluded_y : DataFrame
            A subset of `df_data_y` which only has rows with label `excluded`.

        """
        excluded_label = self.config.settings["strings_to_labels"][excluded]
        excluded_x = df_data_x[
            df_data_y[self.config.settings["label_col"]] == excluded_label
        ]
        excluded_y = df_data_y[
            df_data_y[self.config.settings["label_col"]] == excluded_label
        ]

        data_x = df_data_x[df_data_y[self.config.settings["label_col"]] != excluded_label]
        data_y = df_data_y[df_data_y[self.config.settings["label_col"]] != excluded_label]

        return data_x, data_y, excluded_x, excluded_y

    def train_val_test_split(
        self, df_data_x, df_data_y, excluded_x, excluded_y, train_ratio, val_ratio
    ):
        """Split data into train, validation and test sets.
        The method uses stratified sampling to ensure each set has the correct
        distribution of points.

        Parameters
        ----------
        df_data_x : DataFrame
            A dataframe containing only the features used for machine learning.
        df_data_y : DataFrame
            A dataframe containing the labels corresponding to `df_data_x`.
        train_ratio : float
            The ratio of all the total dataset that should be used for the
            training set.
        val_ratio : float
            The ratio of all the total dataset that should be used for the
            validation set.

        Returns
        -------
        x_train : DataFrame
            A subset of `df_data_x` which will be used for training a model.
        y_train : DataFrame
            A dataframe containing the labels corresponding to `x_train`.
        x_val : DataFrame
            A subset of `df_data_x` which will be used for validating a model.
        y_val : DataFrame
            A dataframe containing the labels corresponding to `x_val`.
        x_test : DataFrame
            A subset of `df_data_x` which will be used for testing a model.
        y_test : DataFrame
            A dataframe containing the labels corresponding to `x_test`.

        """

        np.random.seed(0)
        rng = np.random.RandomState(seed=0)

        include_test_file = True

        if "test_set_file" not in list(self.config.settings.keys()):
            include_test_file = False

        elif not self.config.settings["test_set_file"]:
            include_test_file = False

        test_ratio = 1 - train_ratio - val_ratio
        x_train, x_temp, y_train, y_temp = train_test_split(
            df_data_x,
            df_data_y,
            test_size=1 - train_ratio,
            stratify=df_data_y[self.config.settings["label_col"]],
            random_state=rng,
        )

        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=test_ratio / (test_ratio + val_ratio),
            stratify=y_temp[self.config.settings["label_col"]],
            random_state=rng,
        )

        if include_test_file:
            (
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
            ) = self.reconstruct_tailored_sets(
                x_train, y_train, x_val, y_val, x_test, y_test, excluded_x, excluded_y
            )

        return x_train, y_train, x_val, y_val, x_test, y_test

    def reconstruct_tailored_sets(
        self, x_train, y_train, x_val, y_val, x_test, y_test, excluded_x, excluded_y
    ):

        labels = {}
        if os.path.exists("data/test_set.json"):
            with open("data/test_set.json", "r") as json_file:
                labels = json.load(json_file)

        ids_test = []

        for id_key in list(labels.keys()):
            if labels[id_key] != -1:
                ids_test.append(id_key)

        ids_trained_on = []

        if self.retrain:
            for i in self.config.settings["classifiers"]:
                if "id" in list(self.config.settings["classifiers"][i].keys()):
                    ids_trained_on += self.config.settings["classifiers"][i]["id"]

        ids_trained_on = list(dict.fromkeys(ids_trained_on))

        isin_test = y_train[self.config.settings["id_col"]].isin(ids_test)
        isin_trained_on = y_train[self.config.settings["id_col"]].isin(ids_trained_on)

        new_x_test = x_train[(isin_test) & (~isin_trained_on)]
        new_y_test = y_train[(isin_test) & (~isin_trained_on)]

        new_x_train = x_train[~((isin_test) & (~isin_trained_on))]
        new_y_train = y_train[~((isin_test) & (~isin_trained_on))]

        isin_test = y_val[self.config.settings["id_col"]].isin(ids_test)
        isin_trained_on = y_val[self.config.settings["id_col"]].isin(ids_trained_on)

        new_x_test_temp = x_val[(isin_test) & (~isin_trained_on)]
        new_y_test_temp = y_val[(isin_test) & (~isin_trained_on)]

        new_x_test = new_x_test.append(new_x_test_temp)
        new_y_test = new_y_test.append(new_y_test_temp)

        new_x_val = x_val[~((isin_test) & (~isin_trained_on))]
        new_y_val = y_val[~((isin_test) & (~isin_trained_on))]

        isin_test = y_test[self.config.settings["id_col"]].isin(ids_test)
        isin_trained_on = y_test[self.config.settings["id_col"]].isin(ids_trained_on)

        new_x_test_temp = x_test[(isin_test) & (~isin_trained_on)]
        new_y_test_temp = y_test[(isin_test) & (~isin_trained_on)]

        new_x_test = new_x_test.append(new_x_test_temp)
        new_y_test = new_y_test.append(new_y_test_temp)

        new_x_val_temp = x_test[~((isin_test) & (~isin_trained_on))]
        new_y_val_temp = y_test[~((isin_test) & (~isin_trained_on))]

        new_x_val = new_x_val.append(new_x_val_temp)
        new_y_val = new_y_val.append(new_y_val_temp)

        for label in list(excluded_x.keys()):

            curr_x = excluded_x[label]
            curr_y = excluded_y[label]

            inc_x = curr_x[curr_y[self.config.settings["id_col"]].isin(ids_test)]
            inc_y = curr_y[curr_y[self.config.settings["id_col"]].isin(ids_test)]

            new_x_test = new_x_test.append(inc_x)
            new_y_test = new_y_test.append(inc_y)

        y_test_temp = []

        for id in list(new_y_test[self.config.settings["id_col"]].values):

            y_test_temp.append(labels[id])

        new_y_test[self.config.settings["label_col"]] = y_test_temp

        assert len(new_x_test) == len(
            new_y_test
        ), f"new_x_test len:{len(new_x_test)}, new_y_test len:{len(new_y_test)}"

        assert len(y_test_temp) == len(
            ids_test
        ), f"y_test_temp len:{len(y_test_temp)}, ids_test len:{len(ids_test)}"

        assert len(new_x_val) == len(
            new_y_val
        ), f"new_x_val len:{len(new_x_val)}, new_y_val len:{len(new_y_val)}"
        assert len(new_x_train) == len(
            new_y_train
        ), f"new_x_train len:{len(new_x_train)}, new_y_train len:{len(new_y_train)}"

        return new_x_train, new_y_train, new_x_val, new_y_val, new_x_test, new_y_test

    def scale_data(self, x_train, x_val, x_test, x_cols):
        """Scale the features of the data according to the training set.

        A RobustScaler is used to limit the impact of outliers on the data.

        Parameters
        ----------
        x_train : DataFrame
            A dataframe containing the training set. All subsequent data will
            be scaled according to this data.
        x_val : DataFrame
            A dataframe containing the validation set.
        x_test : DataFrame
            A dataframe containing the testing set.
        x_cols : list of str
            List containing all the column names in `x_train`,`x_val` and
            `x_test`.

        Returns
        -------
        data_x_tr : DataFrame
            A dataframe containing the normalised training set.
        data_x_val : DataFrame
            A dataframe containing the normalised validation set.
        data_x_test : DataFrame
            A dataframe containing the normalised testing set.

        """

        self.scaler = RobustScaler()

        x_tr = x_train[x_cols]
        x_tr = x_train.to_numpy()
        x_tr = x_tr[:, 1:]

        data_x_tr = self.scaler.fit_transform(x_tr)

        x_v = x_val[x_cols]
        x_v = x_val.to_numpy()
        x_v = x_v[:, 1:]
        data_x_val = self.scaler.transform(x_v)

        x_te = x_test[x_cols]
        x_te = x_test.to_numpy()
        x_te = x_te[:, 1:]
        data_x_test = self.scaler.transform(x_te)

        data_x_tr = pd.DataFrame(data_x_tr, columns=x_cols, index=x_train.index)

        data_x_val = pd.DataFrame(data_x_val, columns=x_cols, index=x_val.index)

        data_x_test = pd.DataFrame(data_x_test, columns=x_cols, index=x_test.index)

        return data_x_tr, data_x_val, data_x_test

    def split_y_ids(self, y_id_train, y_id_val, y_id_test):
        """Split label and id columns into separate dataframes.

        Parameters
        ----------
        y_id_train : DataFrame
            Dataframe containing label and id columns of the training set.
        y_id_val : DataFrame
            Dataframe containing label and id columns of the validation set.
        y_id_test : DataFrame
            Dataframe containing label and id columns of the test set.

        Returns
        -------
        data_y_tr : DataFrame
            Dataframe containing only the label column of `y_id_train`.
        data_id_tr : DataFrame
            Dataframe containing only the id column of `y_id_train`.
        data_y_val : DataFrame
            Dataframe containing only the label column of `y_id_val`.
        data_id_val : DataFrame
            Dataframe containing only the id column of `y_id_val`.
        data_y_test : DataFrame
            Dataframe containing only the label column of `y_id_test`.
        data_id_test : DataFrame
            Dataframe containing only the id column of `y_id_test`.

        """

        data_y_tr = pd.DataFrame(
            y_id_train[self.config.settings["label_col"]],
            columns=[self.config.settings["label_col"]],
        )
        data_id_tr = pd.DataFrame(
            y_id_train[self.config.settings["id_col"]], columns=[self.config.settings["id_col"]]
        )
        data_y_val = pd.DataFrame(
            y_id_val[self.config.settings["label_col"]],
            columns=[self.config.settings["label_col"]],
        )
        data_id_val = pd.DataFrame(
            y_id_val[self.config.settings["id_col"]], columns=[self.config.settings["id_col"]]
        )
        data_y_test = pd.DataFrame(
            y_id_test[self.config.settings["label_col"]],
            columns=[self.config.settings["label_col"]],
        )
        data_id_test = pd.DataFrame(
            y_id_test[self.config.settings["id_col"]], columns=[self.config.settings["id_col"]]
        )

        return data_y_tr, data_id_tr, data_y_val, data_id_val, data_y_test, data_id_test

    def _convert_to_one_vs_rest(self):

        y_tr = self.config.ml_data["y_train_without_unknowns"].copy()
        y_val = self.y_val.copy()
        y_test = self.y_test.copy()

        is_label = y_tr[self.config.settings["label_col"]] == self._label
        isnt_label = y_tr[self.config.settings["label_col"]] != self._label

        y_tr.loc[is_label, self.config.settings["label_col"]] = 1
        y_tr.loc[isnt_label, self.config.settings["label_col"]] = 0

        is_label = y_val[self.config.settings["label_col"]] == self._label
        isnt_label = y_val[self.config.settings["label_col"]] != self._label

        y_val.loc[is_label, self.config.settings["label_col"]] = 1
        y_val.loc[isnt_label, self.config.settings["label_col"]] = 0

        is_label = y_test[self.config.settings["label_col"]] == self._label
        isnt_label = y_test[self.config.settings["label_col"]] != self._label

        y_test.loc[is_label, self.config.settings["label_col"]] = 1
        y_test.loc[isnt_label, self.config.settings["label_col"]] = 0

        self.y_train_without_unknowns = y_tr
        self.y_train_with_unknowns = self.config.ml_data["y_train_with_unknowns"].copy()
        self.y_val = y_val
        self.y_test = y_test

    def _get_blank_classifiers(self):
        classifiers = models.get_classifiers()
        return classifiers

    def _update_predictions(self):

        proba = self.learner.predict_proba(self.x_train_without_unknowns)

        tr_pred = np.argmax(proba, axis=1).reshape((-1, 1))

        temp = self.y_train_without_unknowns.to_numpy().reshape((-1, 1))
        is_correct = tr_pred == temp

        default_x = (
            self.config.ml_data["x_train_without_unknowns"][
                self.config.settings["default_vars"][0]
            ]
            .to_numpy()
            .reshape((-1, 1))
        )
        default_y = (
            self.config.ml_data["x_train_without_unknowns"][
                self.config.settings["default_vars"][1]
            ]
            .to_numpy()
            .reshape((-1, 1))
        )

        corr_data = {
            f'{self.config.settings["default_vars"][0]}': default_x[is_correct],
            f'{self.config.settings["default_vars"][1]}': default_y[is_correct],
        }
        incorr_data = {
            f'{self.config.settings["default_vars"][0]}': default_x[~is_correct],
            f'{self.config.settings["default_vars"][1]}': default_y[~is_correct],
        }

        self.corr_train.data = corr_data
        self.incorr_train.data = incorr_data

        curr_tr_acc = accuracy_score(self.y_train_without_unknowns, tr_pred)
        curr_tr_f1 = f1_score(self.y_train_without_unknowns, tr_pred)
        curr_tr_prec = precision_score(self.y_train_without_unknowns, tr_pred)
        curr_tr_rec = recall_score(self.y_train_without_unknowns, tr_pred)

        self._train_scores = {
            "acc": "%.3f" % round(curr_tr_acc, 3),
            "prec": "%.3f" % round(curr_tr_prec, 3),
            "rec": "%.3f" % round(curr_tr_rec, 3),
            "f1": "%.3f" % round(curr_tr_f1, 3),
        }

        self._accuracy_list["train"]["score"].append(curr_tr_acc)
        self._f1_list["train"]["score"].append(curr_tr_f1)
        self._precision_list["train"]["score"].append(curr_tr_prec)
        self._recall_list["train"]["score"].append(curr_tr_rec)

        t_conf = confusion_matrix(self.y_train_without_unknowns, tr_pred)

        val_pred = self.learner.predict(self.config.ml_data["x_val"]).reshape((-1, 1))

        temp = self.y_val.to_numpy().reshape((-1, 1))

        is_correct = val_pred == temp

        default_x = (
            self.config.ml_data["x_val"][self.config.settings["default_vars"][0]]
            .to_numpy()
            .reshape((-1, 1))
        )
        default_y = (
            self.config.ml_data["x_val"][self.config.settings["default_vars"][1]]
            .to_numpy()
            .reshape((-1, 1))
        )

        corr_data = {
            f'{self.config.settings["default_vars"][0]}': default_x[is_correct],
            f'{self.config.settings["default_vars"][1]}': default_y[is_correct],
        }
        incorr_data = {
            f'{self.config.settings["default_vars"][0]}': default_x[~is_correct],
            f'{self.config.settings["default_vars"][1]}': default_y[~is_correct],
        }

        self.corr_val.data = corr_data
        self.incorr_val.data = incorr_data

        curr_val_acc = accuracy_score(self.y_val, val_pred)
        curr_val_f1 = f1_score(self.y_val, val_pred)
        curr_val_prec = precision_score(self.y_val, val_pred)
        curr_val_rec = recall_score(self.y_val, val_pred)

        self._val_scores = {
            "acc": "%.3f" % round(curr_val_acc, 3),
            "prec": "%.3f" % round(curr_val_prec, 3),
            "rec": "%.3f" % round(curr_val_rec, 3),
            "f1": "%.3f" % round(curr_val_f1, 3),
        }
        self._accuracy_list["val"]["score"].append(curr_val_acc)

        self._f1_list["val"]["score"].append(curr_val_f1)
        self._precision_list["val"]["score"].append(curr_val_prec)
        self._recall_list["val"]["score"].append(curr_val_rec)

        v_conf = confusion_matrix(self.y_val, val_pred)

        test_pred = self.learner.predict(self.config.ml_data["x_test"]).reshape((-1, 1))

        temp = self.y_test.to_numpy().reshape((-1, 1))

        curr_test_acc = accuracy_score(self.y_test, test_pred)
        curr_test_f1 = f1_score(self.y_test, test_pred)
        curr_test_prec = precision_score(self.y_test, test_pred)
        curr_test_rec = recall_score(self.y_test, test_pred)

        self._test_scores = {
            "acc": "%.3f" % round(curr_test_acc, 3),
            "prec": "%.3f" % round(curr_test_prec, 3),
            "rec": "%.3f" % round(curr_test_rec, 3),
            "f1": "%.3f" % round(curr_test_f1, 3),
        }

        test_conf = confusion_matrix(self.y_test, test_pred)

        self.num_points_list.append(self.curr_num_points)

        self._accuracy_list["train"]["num_points"] = self.num_points_list
        self._f1_list["train"]["num_points"] = self.num_points_list
        self._precision_list["train"]["num_points"] = self.num_points_list
        self._recall_list["train"]["num_points"] = self.num_points_list
        self._accuracy_list["val"]["num_points"] = self.num_points_list
        self._f1_list["val"]["num_points"] = self.num_points_list
        self._precision_list["val"]["num_points"] = self.num_points_list
        self._recall_list["val"]["num_points"] = self.num_points_list

        self.conf_mat_tr_tn = str(t_conf[0][0])
        self.conf_mat_tr_fp = str(t_conf[0][1])
        self.conf_mat_tr_fn = str(t_conf[1][0])
        self.conf_mat_tr_tp = str(t_conf[1][1])

        self.conf_mat_val_tn = str(v_conf[0][0])
        self.conf_mat_val_fp = str(v_conf[0][1])
        self.conf_mat_val_fn = str(v_conf[1][0])
        self.conf_mat_val_tp = str(v_conf[1][1])

        try:
            self.conf_mat_test_tn = str(test_conf[0][0])
            self.conf_mat_test_fp = str(test_conf[0][1])
            self.conf_mat_test_fn = str(test_conf[1][0])
            self.conf_mat_test_tp = str(test_conf[1][1])
        except:
            self.conf_mat_test_tn = str("N/A")
            self.conf_mat_test_fp = str("N/A")
            self.conf_mat_test_fn = str("N/A")
            self.conf_mat_test_tp = str("N/A")
            print(
                f"Your test set either contains all {self._label_alias} labels or does not contain any {self._label_alias} labels at all."
            )
            print(
                "Please return to Labelling Mode and add some of the missing labels to your test set."
            )
        proba = 1 - np.max(proba, axis=1)

        x_axis = self.config.ml_data["x_train_without_unknowns"][
            self.config.settings["default_vars"][0]
        ].to_numpy()
        y_axis = self.config.ml_data["x_train_without_unknowns"][
            self.config.settings["default_vars"][1]
        ].to_numpy()

        self._model_output_data_tr[self.config.settings["default_vars"][0]] = x_axis
        self._model_output_data_tr[self.config.settings["default_vars"][1]] = y_axis
        self._model_output_data_tr["pred"] = tr_pred.flatten()
        self._model_output_data_tr[
            "y"
        ] = self.y_train_without_unknowns.to_numpy().flatten()
        self._model_output_data_tr["metric"] = proba

        x_axis = self.config.ml_data["x_val"][self.config.settings["default_vars"][0]].to_numpy()
        y_axis = self.config.ml_data["x_val"][self.config.settings["default_vars"][1]].to_numpy()

        self._model_output_data_val[self.config.settings["default_vars"][0]] = x_axis
        self._model_output_data_val[self.config.settings["default_vars"][1]] = y_axis
        self._model_output_data_val["pred"] = val_pred.flatten()
        self._model_output_data_val["y"] = self.y_val.to_numpy().flatten()

    def create_pool(self, preselected=None):
        """Create the pool used for query points during active learning.
        The training set will be split into the pool and the classifier's
        training set. The number in the classifier's training set has already
        been set by the user and these points will be chosen randomly from
        the pool.

        Returns
        -------
        None

        """

        np.random.seed(0)

        if preselected is None:
            initial_points = int(self.starting_num_points.value)

            if initial_points >= len(self.config.ml_data["x_train_without_unknowns"].index):
                self.starting_num_points.value = len(
                    self.config.ml_data["x_train_without_unknowns"].index
                )

                initial_points = len(self.config.ml_data["x_train_without_unknowns"].index)

            y_tr = self.y_train_without_unknowns.copy()

            X_pool = self.config.ml_data["x_train_with_unknowns"].to_numpy()
            y_pool = self.y_train_with_unknowns.to_numpy().ravel()

            self.id_train = self.config.ml_data["id_train_with_unknowns"].copy()

            id_pool = self.id_train.to_numpy()

            train_idx = list(
                np.random.choice(
                    range(len(self.config.ml_data["x_train_without_unknowns"])),
                    size=initial_points - 2,
                    replace=False,
                )
            )

            c0 = train_idx[0]
            c1 = train_idx[1]

            while c0 in train_idx:
                c0 = np.random.choice(np.where(y_tr == 0)[0])
            while c1 in train_idx:
                c1 = np.random.choice(np.where(y_tr == 1)[0])

            train_idx = train_idx + [c0, c1]

            self.x_al_train = X_pool[train_idx]
            self.y_al_train = self.y_train_without_unknowns.iloc[
                train_idx
            ].values.ravel()
            self.id_al_train = self.id_train.iloc[train_idx]

            self.x_pool = np.delete(X_pool, train_idx, axis=0)
            self.y_pool = np.delete(y_pool, train_idx)
            self.id_pool = self.id_train.drop(self.id_train.index[train_idx])

            self.config.settings["classifiers"][f"{self._label}"]["id"] = self.id_al_train[
                self.config.settings["id_col"]
            ].values.tolist()
            self.full_labelled_data["id"] = self.id_al_train[
                self.config.settings["id_col"]
            ].values.tolist()


            self.config.settings["classifiers"][f"{self._label}"][
                "y"
            ] = self.y_train_with_unknowns.iloc[train_idx]

            flat_list = [
                item
                for sublist in self.y_train_with_unknowns.iloc[
                    train_idx
                ].values.tolist()
                for item in sublist
            ]

            self.full_labelled_data["y"] = flat_list

        else:

            new_y = preselected[0]
            new_id = preselected[1]

            y_tr = self.y_train_without_unknowns.copy()
            self.id_train = self.config.ml_data["id_train_with_unknowns"].copy()

            X_pool = self.config.ml_data["x_train_with_unknowns"].to_numpy()
            y_pool = self.y_train_with_unknowns.to_numpy().ravel()
            id_pool = self.id_train.to_numpy()

            train_idx = []

            for id in new_id:
                train_idx.append(np.where(id_pool == id)[0][0])

            self.x_al_train = X_pool[train_idx]
            self.y_al_train = new_y
            self.id_al_train = self.id_train.iloc[train_idx]

            self.x_pool = np.delete(X_pool, train_idx, axis=0)
            self.y_pool = np.delete(y_pool, train_idx)
            self.id_pool = self.id_train.drop(self.id_train.index[train_idx])

            self.config.settings["classifiers"][f"{self._label}"][
                "y"
            ] = self.full_labelled_data["y"]
            self.config.settings["classifiers"][f"{self._label}"][
                "id"
            ] = self.full_labelled_data["id"]

    def setup_learners(self):
        """Initialise the classifiers used during active learning.

        The classifiers used have already been chosen by the user.

        Returns
        -------
        None

        """

        table = self.classifier_table_source.data

        if len(table["classifier"]) == 0:
            return

        qs_dict = query_strategies.get_strategy_dict()

        classifier_dict = self._get_blank_classifiers()

        setup = False

        if self.retrain:

            new_y_with_unknowns = self.config.settings["classifiers"][f"{self._label}"]["y"]
            new_id_with_unknowns = self.config.settings["classifiers"][f"{self._label}"][
                "id"
            ]

            new_y_converted = new_y_with_unknowns.copy()

            for i in range(len(new_y_with_unknowns)):
                if new_y_with_unknowns[i] == self._label:
                    new_y_converted[i] = 1

                elif str(new_y_with_unknowns[i]) == "-1":
                    new_y_converted[i] = -1

                else:
                    new_y_converted[i] = 0

            self.full_labelled_data["y"] = new_y_with_unknowns
            self.full_labelled_data["id"] = new_id_with_unknowns

            new_y = [x for x in new_y_converted if x != -1]
            new_id = [
                x
                for i, x in enumerate(new_id_with_unknowns)
                if new_y_converted[i] != -1
            ]

            unknowns_id = [
                x
                for i, x in enumerate(new_id_with_unknowns)
                if new_y_with_unknowns[i] == -1
            ]

            preselected = [new_y, new_id]

            self.create_pool(preselected=preselected)

            for id in unknowns_id:
                self.remove_from_pool(id=id)

            setup = True

        if not setup:
            self.create_pool()

        if len(table["classifier"]) == 1:
            self.committee = False
            learner = ActiveLearner(
                estimator=clone(classifier_dict[table["classifier"][0]]),
                query_strategy=qs_dict[table["query"][0]],
                X_training=self.x_al_train,
                y_training=self.y_al_train,
            )

            self.learner = learner

        else:
            learners = []
            self.committee = True
            for i in range(len(table["classifier"])):
                learner = ActiveLearner(
                    estimator=clone(classifier_dict[table["classifier"][i]]),
                    query_strategy=qs_dict[table["query"][i]],
                    X_training=self.x_al_train,
                    y_training=self.y_al_train,
                )
                learners.append(learner)

            self.learner = Committee(learner_list=learners)

        self._update_predictions()

    # TODO :: Add bool to see if user wants this step
    def generate_features(self, df):
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
        df_al : DataFrame
            A dataframe containing a subset of `df` with only the required
            features for training.

        """
        np.random.seed(0)

        bands = self.config.settings["features_for_training"]

        features = bands + [self.config.settings["label_col"], self.config.settings["id_col"]]

        oper_dict = feature_generation.get_oper_dict()

        print("number of features before: ", len(features))

        if "feature_generation" in list(self.config.settings.keys()):
            for generator in self.config.settings["feature_generation"]:

                oper = generator[0]
                n = generator[1]

                df, generated_features = oper_dict[oper](df, n, context = self.context)
                features = features + generated_features

        print("number of features after: ", len(features))
        
        df_al = df[features]

        shuffled = np.random.permutation(list(df_al.index.values))

        df_al = df_al.reindex(shuffled)
        df_al = df_al.reset_index()

        return df, df_al

    # CHANGED :: Remove static declarations
    def _combine_data(self):

        data = np.array(
            self._model_output_data_tr[self.config.settings["default_vars"][0]]
        ).reshape((-1, 1))

        data = np.concatenate(
            (
                data,
                np.array(
                    self._model_output_data_tr[self.config.settings["default_vars"][1]]
                ).reshape((-1, 1)),
            ),
            axis=1,
        )
        data = np.concatenate(
            (data, np.array(self._model_output_data_tr["metric"]).reshape((-1, 1))),
            axis=1,
        )
        data = np.concatenate(
            (data, np.array(self._model_output_data_tr["y"]).reshape((-1, 1))), axis=1
        )
        data = np.concatenate(
            (data, np.array(self._model_output_data_tr["pred"]).reshape((-1, 1))),
            axis=1,
        )
        data = np.concatenate(
            (data, np.array(self._model_output_data_tr["acc"]).reshape((-1, 1))), axis=1
        )

        data = pd.DataFrame(data, columns=list(self._model_output_data_tr.keys()))

        return data

    def _train_tab(self):

        self._model_output_data_tr["acc"] = 1 * np.equal(
            np.array(self._model_output_data_tr["pred"]),
            np.array(self._model_output_data_tr["y"]),
        )

        df = pd.DataFrame(
            self._model_output_data_tr, columns=list(self._model_output_data_tr.keys())
        )

        df = df[df["acc"].isin(self._train_tab_colour_switch.active)]

        p = hv.Points(
            df,
            [self.config.settings["default_vars"][0], self.config.settings["default_vars"][1]],
        ).opts(toolbar=None, default_tools=[])

        if hasattr(self, "x_al_train"):
            x_al_train = pd.DataFrame(
                self.x_al_train, columns=self.config.ml_data["x_train_with_unknowns"].columns
            )

        else:

            x_al_train = self._empty_data()

        x_al_train_plot = hv.Scatter(
            x_al_train,
            self.config.settings["default_vars"][0],
            self.config.settings["default_vars"][1],
            label="Trained On",
            # sizing_mode="stretch_width",
        ).opts(
            fill_color="black",
            marker="circle",
            size=10,
            toolbar=None,
            default_tools=[],
            show_legend=True,
        )

        if hasattr(self, "query_instance"):

            query_point = pd.DataFrame(
                self.query_instance,
                columns=self.config.ml_data["x_train_with_unknowns"].columns,
            )

        else:

            query_point = self._empty_data()

        query_point_plot = hv.Scatter(
            query_point,
            self.config.settings["default_vars"][0],
            self.config.settings["default_vars"][1],
            label="Queried"
            # sizing_mode="stretch_width",
        ).opts(
            fill_color="yellow",
            marker="circle",
            size=10,
            toolbar=None,
            default_tools=[],
            show_legend=True,
        )

        if len(x_al_train[self.config.settings["default_vars"][0]]) > 0:

            max_x_temp = np.max(
                [
                    np.max(query_point[self.config.settings["default_vars"][0]]),
                    np.max(x_al_train[self.config.settings["default_vars"][0]]),
                ]
            )

            max_x = np.max([self._max_x, max_x_temp])

            min_x_temp = np.min(
                [
                    np.min(query_point[self.config.settings["default_vars"][0]]),
                    np.min(x_al_train[self.config.settings["default_vars"][0]]),
                ]
            )

            min_x = np.min([self._min_x, min_x_temp])

            max_y_temp = np.max(
                [
                    np.max(query_point[self.config.settings["default_vars"][1]]),
                    np.max(x_al_train[self.config.settings["default_vars"][1]]),
                ]
            )

            max_y = np.max([self._max_y, max_y_temp])

            min_y_temp = np.min(
                [
                    np.min(query_point[self.config.settings["default_vars"][1]]),
                    np.min(x_al_train[self.config.settings["default_vars"][1]]),
                ]
            )

            min_y = np.min([self._min_y, min_y_temp])
        else:
            max_x, min_x, max_y, min_y = (
                self._max_x,
                self._min_x,
                self._max_y,
                self._min_y,
            )

        plot = dynspread(
            datashade(
                p,
                aggregator=ds.max("acc"),
                cmap="RdYlGn",
                normalization="linear",
                clims=(-0.2, 1.2),
            ).opts(
                xlim=(min_x, max_x),
                ylim=(min_y, max_y),
                responsive=True,
                shared_axes=False,
                toolbar=None,
                default_tools=[],
            ),
            threshold=0.75,
            how="saturate",
        )

        full_plot = (plot * x_al_train_plot * query_point_plot).opts(
            toolbar=None, default_tools=[]
        )  # * color_points

        plot_col = pn.Column(
            pn.Row(
                self._train_tab_colour_switch,
                max_height=35,
                sizing_mode="stretch_width",
            ),
            full_plot,
            sizing_mode="stretch_both",
            name="Training Set",
        )
        return plot_col

    def _val_tab(self):

        self._model_output_data_val["acc"] = 1 * np.equal(
            np.array(self._model_output_data_val["pred"]),
            np.array(self._model_output_data_val["y"]),
        )

        df = pd.DataFrame(
            self._model_output_data_val,
            columns=list(self._model_output_data_val.keys()),
        )

        max_x = np.max(df[f"{self.config.settings['default_vars'][0]}"])
        min_x = np.min(df[f"{self.config.settings['default_vars'][0]}"])
        max_y = np.max(df[f"{self.config.settings['default_vars'][1]}"])
        min_y = np.min(df[f"{self.config.settings['default_vars'][1]}"])

        df = df[df["acc"].isin(self._val_tab_colour_switch.active)]
        p = hv.Points(
            df,
            [self.config.settings["default_vars"][0], self.config.settings["default_vars"][1]],
        ).opts(toolbar=None, default_tools=[])

        plot = dynspread(
            datashade(
                p,
                aggregator=ds.max("acc"),
                cmap="RdYlGn",
                normalization="linear",
                clims=(-0.2, 1.2),
            ).opts(
                xlim=(min_x, max_x),
                ylim=(min_y, max_y),
                responsive=True,
                shared_axes=False,
                toolbar=None,
                default_tools=[],
            ),
            threshold=0.75,
            how="saturate",
        )

        full_plot = plot.opts(toolbar=None, default_tools=[])

        plot_col = pn.Column(
            pn.Row(
                self._val_tab_colour_switch,
                max_height=35,
                max_width=500,
                sizing_mode="stretch_width",
            ),
            pn.Row(full_plot, sizing_mode="stretch_both"),
            sizing_mode="stretch_both",
            name="Validation Set",
        )
        return plot_col

    def _metric_tab(self):

        if "acc" not in self._model_output_data_tr.keys():
            self._model_output_data_tr["acc"] = np.equal(
                np.array(self._model_output_data_tr["pred"]),
                np.array(self._model_output_data_tr["y"]),
            )

        elif len(self._model_output_data_tr["acc"]) == 0:
            self._model_output_data_tr["acc"] = np.equal(
                np.array(self._model_output_data_tr["pred"]),
                np.array(self._model_output_data_tr["y"]),
            )

        df = pd.DataFrame(
            self._model_output_data_tr, columns=list(self._model_output_data_tr.keys())
        )

        p = hv.Points(
            df, [self.config.settings["default_vars"][0], self.config.settings["default_vars"][1]]
        ).opts(toolbar=None, default_tools=[])

        plot = dynspread(
            datashade(
                p,
                cmap="RdYlGn_r",
                aggregator=ds.max("metric"),
                normalization="linear",
                clims=(0, 0.5),
            ).opts(
                xlim=(self._min_x, self._max_x),
                ylim=(self._min_y, self._max_y),
                shared_axes=False,
                responsive=True,
            ),
            threshold=0.75,
            how="saturate",
        )

        return plot.opts(toolbar=None, default_tools=[])

    def _scores_tab(self):

        return (
            hv.Path(
                self._accuracy_list["train"], ["num_points", "score"], label="Acc"
            ).options(show_legend=True)
            * hv.Path(
                self._recall_list["train"], ["num_points", "score"], label="Recall"
            ).options(show_legend=True)
            * hv.Path(
                self._precision_list["train"],
                ["num_points", "score"],
                label="Precision",
            ).options(show_legend=True)
            * hv.Path(
                self._f1_list["train"], ["num_points", "score"], label="F1"
            ).options(show_legend=True)
        ).opts(legend_position="bottom_right", toolbar=None, default_tools=[])


    def _cm_block(self, title, scores_text, tn, fp, fn, tp):
        html = f"""
        <div class="al-cm">
        <h4>{title}</h4>
        <div class="metrics">{scores_text}</div>
        <table>
            <tr>
            <th></th>
            <th>Predicted 0</th>
            <th>Predicted 1</th>
            </tr>
            <tr>
            <td class="rowhdr">Actual 0</td>
            <td class="diag">{tn}</td>
            <td class="offd">{fp}</td>
            </tr>
            <tr>
            <td class="rowhdr">Actual 1</td>
            <td class="offd">{fn}</td>
            <td class="diag">{tp}</td>
            </tr>
        </table>
        </div>
        """
        return pn.pane.HTML(html, sizing_mode="stretch_width", margin=(0,0,0,0))

    def _add_conf_matrices(self):
        if not self._show_test_results:
            tr_scores = f"Acc: {self._train_scores['acc']}, Prec: {self._train_scores['prec']}, Rec: {self._train_scores['rec']}, F1: {self._train_scores['f1']}"
            val_scores = f"Acc: {self._val_scores['acc']}, Prec: {self._val_scores['prec']}, Rec: {self._val_scores['rec']}, F1: {self._val_scores['f1']}"

            return pn.Column(
                self._cm_block("Training Set", tr_scores,
                        self.conf_mat_tr_tn, self.conf_mat_tr_fp,
                        self.conf_mat_tr_fn, self.conf_mat_tr_tp),
                pn.Spacer(height=12),
                self._cm_block("Validation Set", val_scores,
                        self.conf_mat_val_tn, self.conf_mat_val_fp,
                        self.conf_mat_val_fn, self.conf_mat_val_tp),
                sizing_mode="stretch_width",
                margin=(0,0,0,0),
            )
        else:
            test_scores = f"Acc: {self._test_scores['acc']}, Prec: {self._test_scores['prec']}, Rec: {self._test_scores['rec']}, F1: {self._test_scores['f1']}"

            if (not self._show_caution) or (self._seen_caution):
                return pn.Column(
                    self._cm_block("Test Set", test_scores,
                        self.conf_mat_test_tn, self.conf_mat_test_fp,
                        self.conf_mat_test_fn, self.conf_mat_test_tp),
                    pn.Spacer(height=12),
                    pn.pane.Markdown(
                        """
                        Please remember to cite our software if you publish these results. See the [Citing page](https://astronomical.readthedocs.io/en/latest/content/other/citing.html) in the documentation for instructions about referencing and citing the astronomicAL software.
                        """,
                        sizing_mode="stretch_width",
                        margin=(0, 0, 0, 0),
                    ),
                    pn.Row(self._return_to_train_view_button, max_height=20),
                    min_height=400,
                )
            else:
                return self._show_caution_message()

    def _show_caution_message(self):

        return pn.Column(
            pn.Row(
                pn.layout.HSpacer(width=75),
                pn.pane.PNG(
                    "images/caution_sign.png",
                    max_height=90,
                    max_width=90,
                ),
                pn.layout.HSpacer(height=5),
            ),
            pn.pane.Markdown(
                """
            ### Caution

            You are about to view the final publishable results for this classifier on the test set.

            It is extremely poor Machine Learning practice to continue training your model after viewing these results.

            If you do continue to train after viewing these results you risk invalidating the generalisability of your classifier by removing the unbias nature of your test set.
            """,
                min_height=200,
            ),
            pn.Row(
                self._return_to_train_view_button,
                self._view_test_results_button,
                max_height=30,
            ),
            self._stop_caution_show_checkbox,
            min_height=400,
        )

    def _request_test_results_cb(self, event):
        self._show_test_results = True
        self._refresh_ui()
        self._seen_caution = True

    def _return_to_train_cb(self, event):
        self._show_test_results = False
        self._seen_caution = False

        if self._stop_caution_show_checkbox.value:
            self._show_caution = False

        self._refresh_ui()

    def _show_test_results_cb(self, event):
        self._seen_test_results = True
        self._show_test_results = True
        self._refresh_ui()
        self._seen_caution = False

        if self._stop_caution_show_checkbox.value:
            self._show_caution = False

    def setup_panel(self):
        """Create the panel which will house all the classifier setup options.

        Returns
        -------
        self.panel_row : Panel Row
            The panel is housed in a row which can then be rendered by the
            respective Dashboard.

        """

        content = None

        if not self._training:

            left = pn.Column(
                pn.Row(self.classifier_dropdown, self.query_strategy_dropdown, max_height=55, height_policy="min"),
                self.starting_num_points,
                sizing_mode="stretch_width",
                height_policy="min",
                max_height=110,
            )
            right = pn.Column(
                self.add_classifier_button,
                self.remove_classifier_button,
                self.start_training_button,
                sizing_mode="fixed",
                height_policy="min",
            )

            table = pn.Column(
                self.classifier_table,
                sizing_mode="stretch_width",
                height=120,
                height_policy="fixed",
                margin=(0,0,0,0),
            )

            content = pn.Row(left, right, table, height_policy="min", sizing_mode="stretch_width")
        else:
            content = pn.widgets.StaticText(
                name="Number of points trained on",
                value=f"{self.curr_num_points}",
                sizing_mode="stretch_width",
                height_policy="min",
                margin=(0, 0, 0, 0),
            )

        if len(self.setup_row) == 0:
            self.setup_row.append(content)
        else:
            self.setup_row[0] = content

    def _update_tab_plots_cb(self, attr, old, new):
        if getattr(self, "_ui_built", False):
            self._train_container[:] = [self._train_tab()]
            self._val_container[:] = [self._val_tab()]

    def _footer(self, selected_message):
        # Row 1: label group + assign
        row1 = pn.Row(
            self.assign_label_group,
            self.assign_label_button,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
        self.assign_label_button.width = 140
        self.assign_label_button.sizing_mode = "fixed"

        # Row 2: status + actions
        self.show_queried_button.width = 220
        self.show_queried_button.sizing_mode = "fixed"

        self.checkpoint_button.width = 140
        self.checkpoint_button.sizing_mode = "fixed"

        self.request_test_results_button.width = 160
        self.request_test_results_button.sizing_mode = "fixed"

        status = pn.pane.Markdown(
            selected_message.object if hasattr(selected_message, "object") else str(selected_message),
            styles=selected_message.styles if hasattr(selected_message, "styles") else {},
            sizing_mode="stretch_width",
            margin=(6, 10, 0, 0),
            css_classes=["al-status"],
        )

        row2 = pn.Row(
            status,
            self.show_queried_button,
            self.checkpoint_button,
            self.request_test_results_button,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        return pn.Column(row1, row2, sizing_mode="stretch_width", css_classes=["al-footer"])

    def _refresh_active_plot(self):
        a = self.tabs_view.active
        if a == 0:
            self._train_pane.object = self._train_tab()
        elif a == 1:
            self._metric_pane.object = self._metric_tab()
        elif a == 2:
            self._val_pane.object = self._val_tab()
        elif a == 3:
            self._scores_pane.object = self._scores_tab()

    def _panel_cb(self, attr, old, new):
        if self._training:
            try:
                query_idx = self.query_index
                queried_id = self.id_pool.iloc[query_idx][self.config.settings["id_col"]].copy()
                self._queried_is_selected = (self.src.data[self.config.settings["id_col"]] == list(queried_id))
            except Exception:
                self._queried_is_selected = False

        if getattr(self, "_ui_built", False):
            self._refresh_footer_only()

    def _build_ui_once(self):
        # --- persistent tab containers (can hold Panel layouts) ---
        self._train_container  = pn.Column(sizing_mode="stretch_both")
        self._metric_container = pn.Column(sizing_mode="stretch_both")
        self._val_container    = pn.Column(sizing_mode="stretch_both")
        self._scores_container = pn.Column(sizing_mode="stretch_both")

        # initial fill
        self._train_container[:]  = [self._train_tab()]
        self._metric_container[:] = [self._metric_tab()]
        self._val_container[:]    = [self._val_tab()]
        self._scores_container[:] = [self._scores_tab()]

        self.tabs_view = pn.Tabs(
            ("Training Set", self._train_container),
            ("Metric", self._metric_container),
            ("Validation Set", self._val_container),
            ("Scores", self._scores_container),
            active=self.active_tab,
            sizing_mode="stretch_both",
            min_height=500,
        )
        self.tabs_view.param.watch(self._on_tab_change, "active")

        self._cm_col = pn.Column(sizing_mode="stretch_both", scroll=True)
        self._footer_col = pn.Column(sizing_mode="stretch_width")

        self._middle_row = pn.Row(
            self.tabs_view,
            self._cm_col,
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
            css_classes=["al-middle-grow"],
        )

        self._middle_row.sizing_mode = "stretch_both"
        self._middle_row.height_policy = "max"

        self.panel_row.css_classes = ["al-root-col"]

        self.setup_panel()
        self._refresh_cm()
        self._refresh_footer_only()

        self.panel_row[:] = [self.setup_row, self._middle_row, self._footer_col]


    def _make_selected_message(self):
        if self._queried_is_selected:
            return pn.pane.Markdown("**The queried source is currently selected**",
                                    styles={"color": "#558855"}, margin=(6,10,0,0))
        return pn.pane.Markdown("**The queried source is not currently selected**",
                                styles={"color": "#ff5555"}, margin=(6,10,0,0))
    
    def _refresh_cm(self):
        # Replace children in-place (fast)
        self._cm_col[:] = [self._add_conf_matrices()]


    def _refresh_cm(self):
        self._cm_col[:] = [self._add_conf_matrices()]

    def _refresh_footer(self):
        selected_message = self._make_selected_message()
        if self._training and (not self._assigned):
            footer = self._footer(selected_message)
        elif self._training and self._assigned:
            footer = pn.Row(self.next_iteration_button, sizing_mode="stretch_width")
        else:
            footer = pn.Row()
        self._footer_col[:] = [footer]

    def _refresh_ui(self, plots=True, cm=True, footer=True):
        self.setup_panel()

        if plots:
            self._train_container[:]  = [self._train_tab()]
            self._metric_container[:] = [self._metric_tab()]
            self._val_container[:]    = [self._val_tab()]
            self._scores_container[:] = [self._scores_tab()]

        if cm:
            self._refresh_cm()
        if footer:
            self._refresh_footer()

    def _on_tab_change(self, event):
        self.active_tab = int(event.new)

    def _refresh_footer_only(self):
        if self._queried_is_selected:
            selected_message = pn.pane.Markdown(
                "**The queried source is currently selected**",
                styles={"color": "#558855"},
                max_width=250,
            )
        else:
            selected_message = pn.pane.Markdown(
                "**The queried source is not currently selected**",
                styles={"color": "#ff5555"},
                max_width=250,
            )

        if self._training and (not self._assigned):
            footer = self._footer(selected_message)
        elif self._training and self._assigned:
            footer = pn.Row(self.next_iteration_button, sizing_mode="stretch_width")
        else:
            footer = pn.Row()

        self._footer_col[:] = [footer]

    def _refresh_ui_full(self):

        self.setup_panel()

        # update tab contents (swap the one child in each container)
        self._train_container[:]  = [self._train_tab()]
        self._metric_container[:] = [self._metric_tab()]
        self._val_container[:]    = [self._val_tab()]
        self._scores_container[:] = [self._scores_tab()]

        self._cm_col[:] = [self._add_conf_matrices()]

        self._refresh_footer_only()

    def panel(self, button_update=False):

        """Create the active learning tab panel.

        Returns
        -------
        panel_row : Panel Row
            The panel is housed in a row which can then be rendered by the
            respective Dashboard.

        """

        if not getattr(self, "_ui_built", False):
            self._build_ui_once()
            self._ui_built = True

        # Update only what is needed
        if button_update:
            self._refresh_footer_only()
        else:
            self._refresh_ui_full()

        return self.panel_row