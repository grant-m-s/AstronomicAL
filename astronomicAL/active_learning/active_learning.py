from bokeh.models import (
    ColumnDataSource,
    DataTable,
    TableColumn,
)
from datetime import datetime
from holoviews.operation.datashader import (
    datashade,
    dynspread,
)
from itertools import combinations
from joblib import dump
from modAL.uncertainty import (
    entropy_sampling,
    margin_sampling,
    uncertainty_sampling
)
from modAL.models import ActiveLearner, Committee
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from utils.optimise import optimise

import config
import datashader as ds
import holoviews as hv
import numpy as np
import os
import pandas as pd
import panel as pn
import param
import sys
import time


class ActiveLearningTab(param.Parameterized):
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
    _label : int
        The label that will be the positive case in the one-vs-rest classifier.
    _last_label : str
        The string alias of the last assigned label in the Active Learing
        process. Used for visual improvements.
    _training : bool
        Flag for whether the training process has begun.
    _assigned : bool
        Flag for whether the user has assigned a label to the current queried
        source.
    y_train : DataFrame
        The labels for the training set.
    y_val : DataFrame
        The labels for the validation set.
    y_test : DataFrame
        The labels for the test set.
    id_train : DataFrame
        The ids of the sources in the training set.
    id_val : DataFrame
        The ids of the sources in the validation set.
    id_test : DataFrame
        The ids of the sources in the test set.
    _model_output_data_tr : type
        Description of attribute `_model_output_data_tr`.

    """

    def __init__(self, src, df, label, **params):
        super(ActiveLearningTab, self).__init__(**params)
        print("init")

        self.df = df

        print(self.df.columns)

        self.src = src
        self._label = config.settings["strings_to_labels"][label]
        self._label_alias = label

        self._last_label = str(label)

        self._training = False
        self._assigned = False

        if len(config.ml_data.keys()) == 0:

            self._preprocess_data()

        else:

            self.y_train = config.ml_data["y_train"]
            self.y_val = config.ml_data["y_val"]
            self.y_test = config.ml_data["y_test"]

            self.id_train = config.ml_data["id_train"]
            self.id_val = config.ml_data["id_val"]
            self.id_test = config.ml_data["id_test"]

            if config.settings["scale_data"]:

                self.scaler = config.ml_data["scaler"]

            self.df = config.main_df

        self._convert_to_one_vs_rest()

        self._construct_panel()

        self._initialise_placeholders()

        self._resize_plot_scales()

    def _resize_plot_scales(self):

        x_axis = config.settings["default_vars"][0]
        y_axis = config.settings["default_vars"][1]

        assert x_axis in config.ml_data["x_train"].keys(
        ), f"Your Default x axis variable doesn't seem to be in your set of features. (MISSING: {x_axis})"
        assert y_axis in config.ml_data["x_train"].keys(
        ), f"Your Default y axis variable doesn't seem to be in your set of features. (MISSING: {y_axis})"

        x_sd = np.std(config.ml_data["x_train"][x_axis])
        x_mu = np.mean(config.ml_data["x_train"][x_axis])
        y_sd = np.std(config.ml_data["x_train"][y_axis])
        y_mu = np.mean(config.ml_data["x_train"][y_axis])

        x_max = x_mu + 4*x_sd
        x_min = x_mu - 4*x_sd

        y_max = y_mu + 4*y_sd
        y_min = y_mu - 4*y_sd

        self._max_x = np.min(
            [(x_max), np.max(config.ml_data["x_train"][x_axis])])
        self._min_x = np.max(
            [(x_min), np.min(config.ml_data["x_train"][x_axis])])

        self._max_y = np.min(
            [(y_max), np.max(config.ml_data["x_train"][y_axis])])
        self._min_y = np.max(
            [(y_min), np.min(config.ml_data["x_train"][y_axis])])

    def _initialise_placeholders(self):

        self._model_output_data_tr = {
            f'{config.settings["default_vars"][0]}': [],
            f'{config.settings["default_vars"][1]}': [],
            "metric": [],
            "y": [],
            "pred": [],
        }

        self._model_output_data_val = {
            f'{config.settings["default_vars"][0]}': [],
            f'{config.settings["default_vars"][1]}': [],
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

        self._train_scores = {"acc": 0.00,
                              "prec": 0.00, "rec": 0.00, "f1": 0.00}
        self._val_scores = {"acc": 0.00, "prec": 0.00, "rec": 0.00, "f1": 0.00}

        self.corr_train = ColumnDataSource(self._empty_data())
        self.incorr_train = ColumnDataSource(self._empty_data())
        self.corr_val = ColumnDataSource(self._empty_data())
        self.incorr_val = ColumnDataSource(self._empty_data())
        self.metric_values = ColumnDataSource(
            {
                f'{config.settings["default_vars"][0]}': [],
                f'{config.settings["default_vars"][1]}': [],
                "metric": [],
            }
        )

        self.queried_points = ColumnDataSource(self._empty_data())

        self.id_al_train = []

    def _construct_panel(self):

        options = []
        for label in config.settings["labels_to_train"]:
            options.append(label)
        options.append("Unsure")
        self.assign_label_group = pn.widgets.RadioButtonGroup(
            name="Label button group",
            options=options,
        )

        self.assign_label_button = pn.widgets.Button(
            name="Assign Label", button_type='primary')
        self.assign_label_button.on_click(self._assign_label_cb)

        self.checkpoint_button = pn.widgets.Button(name="Checkpoint")
        self.checkpoint_button.on_click(self._checkpoint_cb)

        self.show_queried_button = pn.widgets.Button(name="Show Queried")
        self.show_queried_button.on_click(self._show_queried_point_cb)

        self.classifier_dropdown = pn.widgets.Select(
            name="Classifier",
            options=["KNN", "DTree", "RForest", "AdaBoost", "GBTrees"],
        )
        self.query_strategy_dropdown = pn.widgets.Select(
            name="Query Strategy",
            options=["Uncertainty Sampling",
                     "Margin Sampling", "Entropy Sampling"],
        )
        self.starting_num_points = pn.widgets.IntInput(
            name="How many initial points?", value=5, step=1, start=3
        )

        self.classifier_table_source = ColumnDataSource(
            dict(classifier=[], query=[]))
        table_column = [
            TableColumn(field="classifier", title="classifier"),
            TableColumn(field="query", title="query"),
        ]

        self.classifier_table = DataTable(
            source=self.classifier_table_source,
            columns=table_column,
        )

        self.add_classifier_button = pn.widgets.Button(
            name=">>", max_height=40)
        self.remove_classifier_button = pn.widgets.Button(
            name="<<", max_height=40)

        self.add_classifier_button.on_click(self._add_classifier_cb)
        self.remove_classifier_button.on_click(self._remove_classifier_cb)

        self.start_training_button = pn.widgets.Button(name="Start Training")
        self.start_training_button.on_click(self._start_training_cb)

        self.next_iteration_button = pn.widgets.Button(name="Next Iteration")
        self.next_iteration_button.on_click(self._next_iteration_cb)

        self.setup_row = pn.Row("Loading")
        self.panel_row = pn.Row("Loading")

        self.conf_mat_tr_tn = "TN"
        self.conf_mat_tr_fn = "FN"
        self.conf_mat_tr_fp = "FP"
        self.conf_mat_tr_tp = "TP"
        self.conf_mat_val_tn = "TN"
        self.conf_mat_val_fn = "FN"
        self.conf_mat_val_fp = "FP"
        self.conf_mat_val_tp = "TP"

    def _preprocess_data(self):

        self.df, self.data = self.generate_features(self.df)
        print(f"df:{sys.getsizeof(self.df)}")
        print(f"data df:{sys.getsizeof(self.data)}")

        x, y = self.split_x_y_ids(self.data)

        if config.settings["exclude_labels"]:
            for label in config.settings["unclassified_labels"]:
                x, y, _, _ = self.exclude_unclassified_labels(
                    x, y, label)

        (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        ) = self.train_val_test_split(x, y, 0.6, 0.2)

        self.x_cols = self.x_train.columns
        self.y_cols = self.y_train.columns

        if config.settings["scale_data"]:

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
        ) = self.split_y_ids(self.y_train, self.y_val, self.y_test)

        self.assign_global_data()

        total = 0
        total += sys.getsizeof(config.ml_data)
        for dataframe in config.ml_data.keys():
            total += sys.getsizeof(config.ml_data[dataframe])
        print(f"config.ml_data :{total}")

        for key in config.ml_data.keys():
            if isinstance(config.ml_data[key], pd.DataFrame):
                config.ml_data[key] = optimise(config.ml_data[key])

        total = 0
        total += sys.getsizeof(config.ml_data)
        for dataframe in config.ml_data.keys():
            total += sys.getsizeof(config.ml_data[dataframe])
        print(f"optimised ml_data :{total}")

    def assign_global_data(self):
        """Assign the current train, validation and test sets to the shared
         `ml_data` dictionary so that it can be used by other classifiers.

        Returns
        -------
        None

        """

        config.ml_data["x_train"] = self.x_train
        config.ml_data["x_val"] = self.x_val
        config.ml_data["x_test"] = self.x_test

        config.ml_data["y_train"] = self.y_train
        config.ml_data["y_val"] = self.y_val
        config.ml_data["y_test"] = self.y_test

        config.ml_data["id_train"] = self.id_train
        config.ml_data["id_val"] = self.id_val
        config.ml_data["id_test"] = self.id_test

        if config.settings["scale_data"]:
            config.ml_data["scaler"] = self.scaler

    def remove_from_pool(self):
        """Remove the current queried source from the active learning pool.

        Returns
        -------
        None

        """
        print("remove_from_pool")

        self.x_pool = np.delete(
            self.x_pool,
            self.query_index,
            0,
        )

        self.y_pool = np.delete(
            self.y_pool,
            self.query_index,
            0,
        )

        self.id_pool = self.id_pool.drop(self.id_pool.index[self.query_index])

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
            dt_string = now.strftime("%Y%m-%d_%H:%M:%S")
            for i, clf in enumerate(self.learner):
                model = clf

                if checkpoint:
                    mod_dir = f"{filename}-{iteration}-{val_f1}-{dt_string}"
                    if not os.path.isdir(mod_dir):
                        os.mkdir(mod_dir)
                    scaler_dir = f"{mod_dir}/SCALER"
                    mod_dir = f"{mod_dir}/{list_c1[i][:6]}_{i}"
                    dump(model, f"{mod_dir}.joblib")

                    if (not os.path.isfile(f"{scaler_dir}.joblib") and (config.settings["scale_data"])):
                        dump(self.scaler, f"{scaler_dir}.joblib")

                else:
                    if not os.path.isdir(filename):
                        os.mkdir(filename)
                    dump(model, f"{filename}/{list_c1[i][:6]}_{i}.joblib")
                    scaler_dir = f"{filename}/SCALER"
                    if (not os.path.isfile(f"{scaler_dir}.joblib") and (config.settings["scale_data"])):
                        dump(self.scaler, f"{scaler_dir}.joblib")

        else:
            model = self.learner.estimator
            if checkpoint:
                now = datetime.now()
                dt_string = now.strftime("%Y%m-%d_%H:%M:%S")
                dump(
                    model, f"{filename}-{iteration}-{val_f1}-{dt_string}.joblib")
                scaler_dir = f"{filename}-{iteration}-{val_f1}-{dt_string}-SCALER"
                if (not os.path.isfile(f"{scaler_dir}.joblib") and (config.settings["scale_data"])):
                    dump(self.scaler, f"{scaler_dir}.joblib")
            else:
                dump(model, f"{filename}.joblib")
                scaler_dir = f"{filename}-SCALER"
                if (not os.path.isfile(f"{scaler_dir}.joblib") and (config.settings["scale_data"])):
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

        print("show_queried_point")
        start = time.time()
        query_instance = self.query_instance
        query_idx = self.query_index
        end = time.time()
        print(f"queries {end - start}")

        data = self.df

        start = time.time()
        queried_id = self.id_pool.iloc[query_idx][config.settings["id_col"]]
        end = time.time()
        print(f"queried_id {end - start}")

        print(f"\n\n\n id is {queried_id.values} \n\n\n")
        start = time.time()
        sel_idx = np.where(
            data[f'{config.settings["id_col"]}'] == queried_id.values[0])
        end = time.time()
        print(f"np.where {end - start}")
        act_label = self.y_pool[query_idx]
        print(f"Should be a {act_label}")
        selected_source = self.df[self.df[config.settings["id_col"]]
                                  == queried_id.values[0]]
        selected_dict = selected_source.set_index(
            config.settings["id_col"]).to_dict('list')
        selected_dict[config.settings["id_col"]] = [queried_id.values[0]]
        self.src.data = selected_dict

        plot_idx = [
            list(self.df.columns).index(config.settings["default_vars"][0]),
            list(self.df.columns).index(config.settings["default_vars"][1]),
        ]

        q = {
            f'{config.settings["default_vars"][0]}': query_instance[:, plot_idx[0]],
            f'{config.settings["default_vars"][1]}': [query_instance[:, plot_idx[1]]],
        }

        self.queried_points.data = q

    def iterate_AL(self):
        """Iterate through one iteration of active learning.

        Returns
        -------
        None

        """
        print("iterate al")

        # self.assign_label = False

        print("remove_from_pool")
        self.remove_from_pool()

        self.curr_num_points = self.x_al_train.shape[0]

        print("fitting")
        self.learner.fit(self.x_al_train, self.y_al_train)
        print("fitted")

        self.save_model(checkpoint=False)

        print("getting predictions")
        self._update_predictions()
        print("got predictions")

        self.query_new_point()

        start = time.time()
        self.show_queried_point()
        end = time.time()
        print(f"show_queried_point {end - start}")

        self.assign_label_button.name = "Assign"

    def query_new_point(self):
        """Query the most informative point from the training pool based off the
        chosen query metric.

        Returns
        -------
        None

        """

        self.query_index, self.query_instance = self.learner.query(self.x_pool)
        print("queried")

    def _assign_label_cb(self, event):

        selected_label = self.assign_label_group.value
        self._last_label = selected_label

        if not selected_label == "Unsure":

            self._assigned = True
            self.next_iteration_button.name = "Next Iteration"

            if self.assign_label_button.name == "Assigned!":
                self.panel()
                return

            self.assign_label_button.name = "Assigned!"

            if int(config.settings["strings_to_labels"][selected_label]) == self._label:
                selected_label = 1
            else:
                selected_label = 0

            selected_label = np.array([selected_label])

            query = self.query_instance
            query_idx = self.query_index

            new_train = np.vstack((self.x_al_train, query))

            new_label = np.concatenate((self.y_al_train, selected_label))

            new_id = self.id_al_train.append(self.id_pool.iloc[query_idx])

            print(new_id)

            self.x_al_train = new_train
            self.y_al_train = new_label
            self.id_al_train = new_id

        else:
            self.assign_label_button.name = "Querying..."
            self.assign_label_button.disabled = True
            self.remove_from_pool()
            self.query_new_point()
            self.show_queried_point()
            self.assign_label_button.name = "Assign"
            self.assign_label_button.disabled = False

        assert (
            self.x_al_train.shape[0] == self.y_al_train.shape[0]
        ), "AL_TRAIN & LABELS NOT EQUAL"

        assert (
            self.y_al_train.shape[0] == self.id_al_train.shape[0]
        ), "AL_LABELS & IDs NOT EQUAL"

        self.panel()

    def _empty_data(self):

        print("empty_data")

        empty = {
            f'{config.settings["default_vars"][0]}': [],
            f'{config.settings["default_vars"][1]}': [],
        }

        return empty

    def _next_iteration_cb(self, event):
        print("next_iter_cb")

        if self.next_iteration_button.name == "Training...":
            return

        self.next_iteration_button.name = "Training..."

        self.iterate_AL()

        self.checkpoint_button.name = "Checkpoint"
        self.checkpoint_button.disabled = False
        self._assigned = False

        self.panel()

    def _start_training_cb(self, event):
        print("start_training_cb")

        self._training = True
        self.start_training_button.name = "Beginning Training..."
        self.start_training_button.disabled = True
        self.add_classifier_button.disabled = True
        self.remove_classifier_button.disabled = True
        self.num_points_list = []
        self.curr_num_points = self.starting_num_points.value

        self.setup_learners()
        query_idx, query_instance = self.learner.query(self.x_pool)

        self.query_instance = query_instance
        self.query_index = query_idx

        self.show_queried_point()

        self.panel()

    def _add_classifier_cb(self, event):
        print("add_classifier_cb")

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

        print("remove_classifier_cb")

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

        print("split_x_y_ids")

        df_data_y_ids = df_data[[config.settings["label_col"],
                                 config.settings["id_col"]]]
        df_data_x = df_data.drop(
            columns=[config.settings["label_col"], config.settings["id_col"]])
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
        excluded_label = config.settings["strings_to_labels"][excluded]
        excluded_x = df_data_x[df_data_y[config.settings["label_col"]]
                               == excluded_label]
        excluded_y = df_data_y[df_data_y[config.settings["label_col"]]
                               == excluded_label]

        data_x = df_data_x[df_data_y[config.settings["label_col"]]
                           != excluded_label]
        data_y = df_data_y[df_data_y[config.settings["label_col"]]
                           != excluded_label]

        return data_x, data_y, excluded_x, excluded_y

    def train_val_test_split(self, df_data_x, df_data_y, train_ratio, val_ratio):
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

        print("train_dev_split")

        np.random.seed(0)
        rng = np.random.RandomState(seed=0)

        test_ratio = 1 - train_ratio - val_ratio
        x_train, x_temp, y_train, y_temp = train_test_split(
            df_data_x,
            df_data_y,
            test_size=1 - train_ratio,
            stratify=df_data_y[config.settings["label_col"]],
            random_state=rng,
        )

        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=test_ratio / (test_ratio + val_ratio),
            stratify=y_temp[config.settings["label_col"]],
            random_state=rng,
        )

        print(f"train: {y_train[config.settings['label_col']].value_counts()}")
        print(f"val: {y_val[config.settings['label_col']].value_counts()}")
        print(f"test: {y_test[config.settings['label_col']].value_counts()}")

        return x_train, y_train, x_val, y_val, x_test, y_test

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

        print("scale_data")

        self.scaler = RobustScaler()
        data_x_tr = self.scaler.fit_transform(x_train)

        data_x_val = self.scaler.transform(x_val)
        data_x_test = self.scaler.transform(x_test)

        data_x_tr = pd.DataFrame(data_x_tr, columns=x_cols)

        data_x_val = pd.DataFrame(data_x_val, columns=x_cols)

        data_x_test = pd.DataFrame(data_x_test, columns=x_cols)

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

        print("split_y_id_ids")

        data_y_tr = pd.DataFrame(
            y_id_train[config.settings["label_col"]], columns=[
                config.settings["label_col"]]
        )
        data_id_tr = pd.DataFrame(
            y_id_train[config.settings["id_col"]], columns=[
                config.settings["id_col"]]
        )
        data_y_val = pd.DataFrame(
            y_id_val[config.settings["label_col"]], columns=[
                config.settings["label_col"]]
        )
        data_id_val = pd.DataFrame(
            y_id_val[config.settings["id_col"]], columns=[
                config.settings["id_col"]]
        )
        data_y_test = pd.DataFrame(
            y_id_test[config.settings["label_col"]], columns=[
                config.settings["label_col"]]
        )
        data_id_test = pd.DataFrame(
            y_id_test[config.settings["id_col"]], columns=[
                config.settings["id_col"]]
        )

        return data_y_tr, data_id_tr, data_y_val, data_id_val, data_y_test, data_id_test

    def _convert_to_one_vs_rest(self):

        y_tr = self.y_train.copy()
        y_val = self.y_val.copy()
        y_test = self.y_test.copy()

        is_label = y_tr[config.settings["label_col"]] == self._label
        isnt_label = y_tr[config.settings["label_col"]] != self._label

        y_tr.loc[is_label, config.settings["label_col"]] = 1
        y_tr.loc[isnt_label, config.settings["label_col"]] = 0

        is_label = y_val[config.settings["label_col"]] == self._label
        isnt_label = y_val[config.settings["label_col"]] != self._label

        y_val.loc[is_label, config.settings["label_col"]] = 1
        y_val.loc[isnt_label, config.settings["label_col"]] = 0

        is_label = y_test[config.settings["label_col"]] == self._label
        isnt_label = y_test[config.settings["label_col"]] != self._label

        y_test.loc[is_label, config.settings["label_col"]] = 1
        y_test.loc[isnt_label, config.settings["label_col"]] = 0

        self.y_train = y_tr
        self.y_val = y_val
        self.y_test = y_test

    def _get_blank_classifiers(self):

        classifiers = {
            "KNN": KNeighborsClassifier(3, n_jobs=-1),
            "DTree": DecisionTreeClassifier(
                random_state=0,
            ),
            "RForest": RandomForestClassifier(
                n_jobs=-1, random_state=0, n_estimators=1000
            ),
            "AdaBoost": AdaBoostClassifier(random_state=0, n_estimators=500),
            "GBTrees": GradientBoostingClassifier(random_state=0, n_estimators=1000),
        }

        return classifiers

    def _update_predictions(self):

        start = time.time()
        proba = self.learner.predict_proba(config.ml_data["x_train"])
        end = time.time()
        print(f"pred proba {end - start}")

        print("get_predicitions")
        start = time.time()
        tr_pred = np.argmax(proba, axis=1).reshape((-1, 1))
        print(tr_pred.reshape)
        print(tr_pred)
        end = time.time()
        print(f"predict {end - start}")
        temp = self.y_train.to_numpy().reshape((-1, 1))
        start = time.time()
        is_correct = tr_pred == temp
        end = time.time()
        print(f"is_correct {end - start}")

        start = time.time()
        default_x = config.ml_data["x_train"][config.settings["default_vars"]
                                              [0]].to_numpy().reshape((-1, 1))
        default_y = config.ml_data["x_train"][config.settings["default_vars"]
                                              [1]].to_numpy().reshape((-1, 1))
        end = time.time()
        print(f"def x_y {end - start}")
        start = time.time()
        corr_data = {
            f'{config.settings["default_vars"][0]}': default_x[is_correct],
            f'{config.settings["default_vars"][1]}': default_y[is_correct],
        }
        incorr_data = {
            f'{config.settings["default_vars"][0]}': default_x[~is_correct],
            f'{config.settings["default_vars"][1]}': default_y[~is_correct],
        }
        end = time.time()
        print(f"corr and incorr {end - start}")
        self.corr_train.data = corr_data
        self.incorr_train.data = incorr_data
        start = time.time()
        curr_tr_acc = accuracy_score(self.y_train, tr_pred)
        curr_tr_f1 = f1_score(self.y_train, tr_pred)
        curr_tr_prec = precision_score(self.y_train, tr_pred)
        curr_tr_rec = recall_score(self.y_train, tr_pred)
        end = time.time()
        print(f"scores {end - start}")
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
        start = time.time()
        t_conf = confusion_matrix(self.y_train, tr_pred)
        end = time.time()
        print(f"conf matrix {end - start}")
        start = time.time()
        val_pred = self.learner.predict(
            config.ml_data["x_val"]).reshape((-1, 1))
        end = time.time()
        print(f"pred val {end - start}")
        temp = self.y_val.to_numpy().reshape((-1, 1))

        is_correct = val_pred == temp

        default_x = config.ml_data["x_val"][config.settings["default_vars"]
                                            [0]].to_numpy().reshape((-1, 1))
        default_y = config.ml_data["x_val"][config.settings["default_vars"]
                                            [1]].to_numpy().reshape((-1, 1))

        corr_data = {
            f'{config.settings["default_vars"][0]}': default_x[is_correct],
            f'{config.settings["default_vars"][1]}': default_y[is_correct],
        }
        incorr_data = {
            f'{config.settings["default_vars"][0]}': default_x[~is_correct],
            f'{config.settings["default_vars"][1]}': default_y[~is_correct],
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

        start = time.time()
        proba = 1 - np.max(proba, axis=1)
        end = time.time()
        print(f"1-proba {end - start}")

        x_axis = config.ml_data["x_train"][config.settings["default_vars"][0]].to_numpy(
        )
        y_axis = config.ml_data["x_train"][config.settings["default_vars"][1]].to_numpy(
        )

        print(f"tr_pred:{tr_pred.shape}")
        print(f"y_train:{self.y_train.shape}")
        print(f"metric:{proba.shape}")

        self._model_output_data_tr[config.settings["default_vars"][0]] = x_axis
        self._model_output_data_tr[config.settings["default_vars"][1]] = y_axis
        self._model_output_data_tr["pred"] = tr_pred.flatten()
        self._model_output_data_tr["y"] = self.y_train.to_numpy().flatten()
        self._model_output_data_tr["metric"] = proba

        x_axis = config.ml_data["x_val"][config.settings["default_vars"][0]].to_numpy(
        )
        y_axis = config.ml_data["x_val"][config.settings["default_vars"][1]].to_numpy(
        )

        self._model_output_data_val[config.settings["default_vars"][0]] = x_axis
        self._model_output_data_val[config.settings["default_vars"][1]] = y_axis
        self._model_output_data_val["pred"] = val_pred.flatten()
        self._model_output_data_val["y"] = self.y_val.to_numpy().flatten()

    def create_pool(self):
        """Create the pool used for query points during active learning.
        The training set will be split into the pool and the classifier's
        training set. The number in the classifier's training set has already
        been set by the user and these points will be chosen randomly from
        the pool.

        Returns
        -------
        None

        """

        print("create_pool")

        np.random.seed(0)

        initial_points = int(self.starting_num_points.value)

        if initial_points >= len(config.ml_data["x_train"].index):
            self.starting_num_points.value = len(
                config.ml_data["x_train"].index)

            initial_points = len(config.ml_data["x_train"].index)

        y_tr = self.y_train.copy()

        # y_tr = y_tr.to_numpy()

        X_pool = config.ml_data["x_train"].to_numpy()
        y_pool = self.y_train.to_numpy().ravel()
        id_pool = self.id_train.to_numpy()

        print(X_pool.shape)

        train_idx = list(
            np.random.choice(
                range(X_pool.shape[0]), size=initial_points - 2, replace=False
            )
        )

        c0 = np.random.choice(np.where(y_tr == 0)[0])
        c1 = np.random.choice(np.where(y_tr == 1)[0])

        train_idx = train_idx + [c0] + [c1]

        X_train = X_pool[train_idx]
        y_train = y_pool[train_idx]
        id_train = self.id_train.iloc[train_idx]

        X_pool = np.delete(X_pool, train_idx, axis=0)
        y_pool = np.delete(y_pool, train_idx)
        id_pool = self.id_train.drop(self.id_train.index[train_idx])

        self.x_pool = X_pool
        self.y_pool = y_pool
        self.id_pool = id_pool
        self.x_al_train = X_train
        self.y_al_train = y_train
        self.id_al_train = id_train

    def setup_learners(self):
        """Initialise the classifiers used during active learning.
        The classifiers used have already been chosen by the user.

        Returns
        -------
        None

        """

        print("setup_learners")

        table = self.classifier_table_source.data

        if len(table["classifier"]) == 0:
            return

        # TODO :: Move this
        qs_dict = {
            "Uncertainty Sampling": uncertainty_sampling,
            "Margin Sampling": margin_sampling,
            "Entropy Sampling": entropy_sampling,
        }

        classifier_dict = self._get_blank_classifiers()

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
        print("generate_features")
        np.random.seed(0)

        # CHANGED :: Change this to selected["AL_Features"]
        bands = config.settings["features_for_training"]

        print(bands)

        features = bands + [config.settings["label_col"],
                            config.settings["id_col"]]
        print(features[-5:])
        df_al = df[features]

        shuffled = np.random.permutation(list(df_al.index.values))

        print("shuffled...")

        df_al = df_al.reindex(shuffled)

        print("reindexed")

        df_al = df_al.reset_index()

        print("reset index")

        combs = list(combinations(bands, 2))

        print(combs)

        cols = list(df.columns)

        for i, j in combs:
            df_al[f"{i}-{j}"] = df_al[i] - df_al[j]

            if f"{i}-{j}" not in cols:
                df[f"{i}-{j}"] = df[i] - df[j]

        print("Feature generations complete")

        return df, df_al

    # CHANGED :: Remove static declarations
    def _combine_data(self):

        print("combine_data")
        data = np.array(
            self._model_output_data_tr[config.settings["default_vars"][0]]).reshape((-1, 1))

        data = np.concatenate(
            (data, np.array(self._model_output_data_tr[config.settings["default_vars"][1]]).reshape((-1, 1))), axis=1
        )
        data = np.concatenate(
            (data, np.array(
                self._model_output_data_tr["metric"]).reshape((-1, 1))),
            axis=1,
        )
        data = np.concatenate(
            (data, np.array(self._model_output_data_tr["y"]).reshape((-1, 1))), axis=1
        )
        data = np.concatenate(
            (data, np.array(self._model_output_data_tr["pred"]).reshape((-1, 1))), axis=1
        )
        data = np.concatenate(
            (data, np.array(self._model_output_data_tr["acc"]).reshape((-1, 1))), axis=1
        )

        data = pd.DataFrame(data, columns=list(
            self._model_output_data_tr.keys()))

        return data

    def _train_tab(self):

        print("train_tab")
        start = time.time()
        self._model_output_data_tr["acc"] = np.equal(
            np.array(self._model_output_data_tr["pred"]),
            np.array(self._model_output_data_tr["y"]),
        )
        end = time.time()
        print(f"equal {end - start}")

        start = time.time()

        df = pd.DataFrame(
            self._model_output_data_tr, columns=list(
                self._model_output_data_tr.keys())
        )

        end = time.time()
        print(f"DF {end - start}")

        start = time.time()
        p = hv.Points(
            df,
            [config.settings["default_vars"][0],
                config.settings["default_vars"][1]],
        ).opts(toolbar=None, default_tools=[])
        end = time.time()
        print(f"P {end - start}")

        if hasattr(self, "x_al_train"):
            start = time.time()
            x_al_train = pd.DataFrame(
                self.x_al_train, columns=config.ml_data["x_train"].columns)
            end = time.time()
            print(f"hasattr {end - start}")
        else:
            start = time.time()
            x_al_train = self._empty_data()
            end = time.time()
            print(f"x_al_train {end - start}")

        start = time.time()
        x_al_train_plot = hv.Scatter(
            x_al_train,
            config.settings["default_vars"][0],
            config.settings["default_vars"][1],
            # sizing_mode="stretch_width",
        ).opts(
            fill_color="black",
            marker="circle",
            size=10, toolbar=None, default_tools=[]
        )
        end = time.time()
        print(f"x_al_train_plot {end - start}")

        if hasattr(self, "query_instance"):
            start = time.time()
            query_point = pd.DataFrame(
                self.query_instance, columns=config.ml_data["x_train"].columns
            )
            end = time.time()
            print(f"query_point {end - start}")
        else:
            start = time.time()
            query_point = self._empty_data()
            end = time.time()
            print(f"query_point {end - start}")

        start = time.time()
        query_point_plot = hv.Scatter(
            query_point,
            config.settings["default_vars"][0],
            config.settings["default_vars"][1],
            # sizing_mode="stretch_width",
        ).opts(
            fill_color="yellow",
            marker="circle",
            size=10, toolbar=None, default_tools=[]
        )
        end = time.time()
        print(f"query_point_plot {end - start}")

        color_key = {1: "#2eb800", 0: "#c20000", "q": "yellow", "t": "black"}

        if len(x_al_train[config.settings["default_vars"][0]]) > 0:

            max_x_temp = np.max(
                [np.max(query_point[config.settings["default_vars"][0]]),
                 np.max(x_al_train[config.settings["default_vars"][0]])]
                )

            max_x = np.max([self._max_x, max_x_temp])

            min_x_temp = np.min(
                [np.min(query_point[config.settings["default_vars"][0]]),
                 np.min(x_al_train[config.settings["default_vars"][0]])]
                )

            min_x = np.min([self._min_x, min_x_temp])

            max_y_temp = np.max(
                [np.max(query_point[config.settings["default_vars"][1]]),
                 np.max(x_al_train[config.settings["default_vars"][1]])]
                )

            max_y = np.max([self._max_y, max_y_temp])

            min_y_temp = np.min(
                [np.min(query_point[config.settings["default_vars"][1]]),
                 np.min(x_al_train[config.settings["default_vars"][1]])]
                )

            min_y = np.min([self._min_y, min_y_temp])
        else:
            max_x, min_x, max_y, min_y = (
                self._max_x, self._min_x, self._max_y, self._min_y)

        start = time.time()
        plot = (
            dynspread(
                datashade(
                    p,
                    color_key=color_key,
                    aggregator=ds.by("acc", ds.count()),
                ).opts(xlim=(min_x, max_x), ylim=(min_y, max_y), responsive=True, shared_axes=False, toolbar=None, default_tools=[]),
                threshold=0.75,
                how="saturate",
            )
        )
        end = time.time()
        print(f"plot {end - start}")
        full_plot = (plot * x_al_train_plot
                     * query_point_plot).opts(toolbar=None, default_tools=[])  # * color_points

        return full_plot

    def _val_tab(self):
        print("_val_tab")
        start = time.time()
        self._model_output_data_val["acc"] = np.equal(
            np.array(self._model_output_data_val["pred"]),
            np.array(self._model_output_data_val["y"]),
        )
        end = time.time()
        print(f"output_data_val {end - start}")
        start = time.time()

        df = pd.DataFrame(
            self._model_output_data_val, columns=list(
                self._model_output_data_val.keys())
        )

        end = time.time()
        print(f"df-val {end - start}")
        start = time.time()
        p = hv.Points(
            df,
            [config.settings["default_vars"][0],
                config.settings["default_vars"][1]],
        ).opts(active_tools=["pan", "wheel_zoom"])
        end = time.time()
        print(f"p-val {end - start}")

        color_key = {1: "#2eb800", 0: "#c20000"}
        start = time.time()
        plot = dynspread(
            datashade(
                p,
                color_key=color_key,
                aggregator=ds.by("acc", ds.count()),
            ).opts(xlim=(self._min_x, self._max_x),
                   ylim=(self._min_y, self._max_y),
                   shared_axes=False, responsive=True,
                   active_tools=["pan", "wheel_zoom"]),
            threshold=0.75,
            how="saturate",
        )
        end = time.time()
        print(f"plot-val {end - start}")
        return plot

    def _metric_tab(self):

        print("_metric_tab")
        start = time.time()

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
            self._model_output_data_tr, columns=list(
                self._model_output_data_tr.keys())
        )

        end = time.time()
        print(f"df {end - start}")

        start = time.time()
        p = hv.Points(
            df, [config.settings["default_vars"][0],
                 config.settings["default_vars"][1]]
        ).opts(active_tools=["pan", "wheel_zoom"])

        end = time.time()
        print(f"p {end - start}")

        start = time.time()

        plot = dynspread(
            datashade(
                p, cmap="RdYlGn_r", aggregator=ds.max("metric"),
                normalization="linear"
            ).opts(active_tools=["pan", "wheel_zoom"],
                   xlim=(self._min_x, self._max_x),
                   ylim=(self._min_y, self._max_y),
                   shared_axes=False, responsive=True),
            threshold=0.75,
            how="saturate",
        )

        end = time.time()
        print(f"plot {end - start}")
        return plot

    def _scores_tab(self):
        print("_scores_tab")

        return (
            hv.Path(self._accuracy_list["train"], ["num_points", "score"])
            * hv.Path(self._recall_list["train"], ["num_points", "score"])
            * hv.Path(self._precision_list["train"], ["num_points", "score"])
            * hv.Path(self._f1_list["train"], ["num_points", "score"])
        )

    def _add_conf_matrices(self):
        print("_add_conf_matrices")
        return pn.Column(
            pn.pane.Markdown("Training Set:", sizing_mode="fixed"),
            pn.pane.Markdown(
                f"Acc: {self._train_scores['acc']}, Prec: {self._train_scores['prec']}, Rec: {self._train_scores['rec']}, F1: {self._train_scores['f1']}",
                sizing_mode="fixed",
            ),
            pn.Row(
                pn.Column(
                    pn.Row("", max_height=30),
                    pn.Row("Actual 0", min_height=50),
                    pn.Row("Actual 1", min_height=50),
                ),
                pn.Column(
                    pn.Row("Predicted 0", max_height=30),
                    pn.Row(pn.pane.Str(self.conf_mat_tr_tn), min_height=50),
                    pn.Row(pn.pane.Str(self.conf_mat_tr_fn), min_height=50),
                ),
                pn.Column(
                    pn.Row("Predicted 1", max_height=30),
                    pn.Row(pn.pane.Str(self.conf_mat_tr_fp), min_height=50),
                    pn.Row(pn.pane.Str(self.conf_mat_tr_tp), min_height=50),
                ),
            ),
            pn.pane.Markdown("Validation Set:", sizing_mode="fixed"),
            pn.pane.Markdown(
                f"Acc: {self._val_scores['acc']}, Prec: {self._val_scores['prec']}, Rec: {self._val_scores['rec']}, F1: {self._val_scores['f1']}",
                sizing_mode="fixed",
            ),
            pn.Row(
                pn.Column(
                    pn.Row("", max_height=30),
                    pn.Row("Actual 0", min_height=50),
                    pn.Row("Actual 1", min_height=50),
                ),
                pn.Column(
                    pn.Row("Predicted 0", max_height=30),
                    pn.Row(pn.pane.Str(self.conf_mat_val_tn), min_height=50),
                    pn.Row(pn.pane.Str(self.conf_mat_val_fn), min_height=50),
                ),
                pn.Column(
                    pn.Row("Predicted 1", max_height=30),
                    pn.Row(pn.pane.Str(self.conf_mat_val_fp), min_height=50),
                    pn.Row(pn.pane.Str(self.conf_mat_val_tp), min_height=50),
                ),
            ),
        )

    def setup_panel(self):
        """Create the panel which will house all the classifier setup options.

        Returns
        -------
        self.panel_row : Panel Row
            The panel is housed in a row which can then be rendered by the
            respective Dashboard.

        """
        print("setup_panel")

        if not self._training:
            self.setup_row[0] = pn.Row(
                pn.Column(
                    pn.Row(
                        self.classifier_dropdown,
                        self.query_strategy_dropdown,
                        max_height=55,
                    ),
                    self.starting_num_points,
                    max_height=110,
                ),
                pn.Column(
                    self.add_classifier_button,
                    self.remove_classifier_button,
                    self.start_training_button,
                ),
                pn.Column(self.classifier_table, max_height=125),
            )
        else:

            self.setup_row[0] = pn.Column(
                pn.widgets.StaticText(
                    name="Number of points trained on",
                    value=f"{self.curr_num_points}",
                ),
            )

    def panel(self):
        """Create the active learning tab panel.

        Returns
        -------
        panel_row : Panel Row
            The panel is housed in a row which can then be rendered by the
            respective Dashboard.

        """
        print("panel")

        self.assign_label_group.value = self._last_label

        start = time.time()
        self.setup_panel()
        end = time.time()
        print(f"setup {end - start}")
        start = time.time()

        buttons_row = pn.Row(max_height=30)
        if self._training:
            if self._assigned:
                buttons_row.append(self.next_iteration_button)
            else:
                buttons_row = pn.Row(self.assign_label_group,
                                     pn.Row(self.assign_label_button,
                                            self.show_queried_button,
                                            self.checkpoint_button,
                                            max_height=30),
                                     max_height=30,)

        self.panel_row[0] = pn.Column(
            pn.Row(self.setup_row),
            pn.Row(
                pn.Tabs(
                    ("Train", self._train_tab),
                    ("Metric", self._metric_tab),
                    ("Val", self._val_tab),
                    ("Scores", self._scores_tab),
                    dynamic=True,
                ),
                self._add_conf_matrices(),
            ),
            pn.Row(max_height=20),
            buttons_row,
        )
        end = time.time()
        print(f"panel_row {end - start}")
        print("\n====================\n")
        return self.panel_row
