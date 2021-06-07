import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../"))

import inspect
import numpy as np
import pandas as pd
import panel as pn
from astronomicAL.active_learning.active_learning import ActiveLearningModel
from astronomicAL.dashboard.active_learning import ActiveLearningDashboard
from astronomicAL.dashboard.dashboard import Dashboard
from astronomicAL.dashboard.labelling import LabellingDashboard
from astronomicAL.dashboard.menu import MenuDashboard
from astronomicAL.dashboard.plot import PlotDashboard
from astronomicAL.dashboard.selected_source import SelectedSourceDashboard
from astronomicAL.dashboard.settings_dashboard import SettingsDashboard
from astronomicAL.extensions.extension_plots import (
    CustomPlot,
    get_plot_dict,
    create_plot,
    bpt_plot,
    mateos_2012_wedge,
)
from astronomicAL.extensions import feature_generation
from astronomicAL.extensions import models
from astronomicAL.extensions import query_strategies
from astronomicAL.settings.param_assignment import ParameterAssignment
from astronomicAL.settings.active_learning import ActiveLearningSettings
from astronomicAL.settings.data_selection import DataSelection
from astronomicAL.utils import load_config
from astronomicAL.utils import save_config
from astronomicAL.utils import optimise
from bokeh.models import ColumnDataSource
from bokeh.models import TextAreaInput
from bokeh.document import Document
from pyviz_comms import Comm
import astronomicAL.config as config
import pytest
import json


@pytest.fixture
def document():
    return Document()


@pytest.fixture
def comm():
    return Comm()


def check_folder_exists(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)


class Event:
    def __init__(self, new, old):
        self.new = new
        self.old = old


class TestClass:
    def func(self, x):
        return x + 1

    def test_answer(self):
        assert self.func(3) == 4


class TestSettings:
    def _create_test_df(self):

        data = []

        for i in range(100):
            data.append([str(i), i % 3, i, i, i])

        df = pd.DataFrame(data, columns=list("ABCDE"))

        return df

    def test_param_assignment_id_column_contains_correct_columns_on_init(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        assert paramAssignment.param.id_column.objects == ["default"]

    def test_param_assignment_id_column_contains_correct_columns_on_update_no_df(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data()

        assert paramAssignment.param.id_column.objects == ["default"]

    def test_param_assignment_id_column_contains_correct_columns_on_update_with_df(
        self,
    ):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        assert paramAssignment.param.id_column.objects == [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]

    def test_param_assignment_label_column_contains_correct_columns_on_init(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        assert paramAssignment.param.label_column.objects == ["default"]

    def test_param_assignment_label_column_contains_correct_columns_on_update_no_df(
        self,
    ):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data()

        assert paramAssignment.param.label_column.objects == ["default"]

    def test_param_assignment_label_column_contains_correct_columns_on_update_with_df(
        self,
    ):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        assert paramAssignment.param.label_column.objects == [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]

    # def test_param_assignment_default_x_variable_contains_correct_columns_on_init(self):
    #
    #     df = self._create_test_df()
    #
    #     paramAssignment = ParameterAssignment()
    #
    #     assert paramAssignment.param.default_x_variable.objects == ["default"]
    #
    # def test_param_assignment_default_x_variable_contains_correct_columns_on_update_no_df(
    #     self,
    # ):
    #
    #     df = self._create_test_df()
    #
    #     paramAssignment = ParameterAssignment()
    #
    #     paramAssignment.update_data()
    #
    #     assert paramAssignment.param.default_x_variable.objects == ["default"]
    #
    # def test_param_assignment_default_x_variable_contains_correct_columns_on_update_with_df(
    #     self,
    # ):
    #
    #     df = self._create_test_df()
    #
    #     paramAssignment = ParameterAssignment()
    #
    #     paramAssignment.update_data(df)
    #
    #     assert paramAssignment.param.default_x_variable.objects == [
    #         "A",
    #         "B",
    #         "C",
    #         "D",
    #         "E",
    #     ]
    #
    # def test_param_assignment_default_y_variable_contains_correct_columns_on_init(self):
    #
    #     df = self._create_test_df()
    #
    #     paramAssignment = ParameterAssignment()
    #
    #     assert paramAssignment.param.default_y_variable.objects == ["default"]
    #
    # def test_param_assignment_default_y_variable_contains_correct_columns_on_update_no_df(
    #     self,
    # ):
    #
    #     df = self._create_test_df()
    #
    #     paramAssignment = ParameterAssignment()
    #
    #     paramAssignment.update_data()
    #
    #     assert paramAssignment.param.default_y_variable.objects == ["default"]
    #
    # def test_param_assignment_default_y_variable_contains_correct_columns_on_update_with_df(
    #     self,
    # ):
    #
    #     df = self._create_test_df()
    #
    #     paramAssignment = ParameterAssignment()
    #
    #     paramAssignment.update_data(df)
    #
    #     assert paramAssignment.param.default_y_variable.objects == [
    #         "A",
    #         "B",
    #         "C",
    #         "D",
    #         "E",
    #     ]

    def test_param_assignment_greater_than_20_labels_dont_proceed(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "E"

        paramAssignment._update_labels_cb()

        assert paramAssignment.label_strings_param == {}

    def test_param_assignment_less_than_20_labels_proceed(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "B"

        paramAssignment._update_labels_cb()

        assert paramAssignment.label_strings_param != {}

    def test_param_assignment_strings_to_labels_check_save_without_inputs(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "B"

        paramAssignment._update_labels_cb()

        paramAssignment._confirm_settings_cb(None)

        assert config.settings["strings_to_labels"] == {"0": 0, "1": 1, "2": 2}

    def test_param_assignment_strings_to_labels_check_save_with_some_inputs(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "B"

        paramAssignment._update_labels_cb()

        paramAssignment.label_strings_param["0"].value = "string_label_for_0"
        paramAssignment.label_strings_param["2"].value = "string_label_for_2"

        paramAssignment._confirm_settings_cb(None)

        assert config.settings["strings_to_labels"] == {
            "string_label_for_0": 0,
            "1": 1,
            "string_label_for_2": 2,
        }

    def test_param_assignment_strings_to_labels_check_save_with_all_inputs(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "B"

        paramAssignment._update_labels_cb()

        paramAssignment.label_strings_param["0"].value = "string_label_for_0"
        paramAssignment.label_strings_param["1"].value = "string_label_for_1"
        paramAssignment.label_strings_param["2"].value = "string_label_for_2"

        paramAssignment._confirm_settings_cb(None)

        assert config.settings["strings_to_labels"] == {
            "string_label_for_0": 0,
            "string_label_for_1": 1,
            "string_label_for_2": 2,
        }

    def test_param_assignment_labels_to_strings_check_save_without_inputs(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "B"

        paramAssignment._update_labels_cb()

        paramAssignment._confirm_settings_cb(None)

        assert config.settings["labels_to_strings"] == {"0": "0", "1": "1", "2": "2"}

    def test_param_assignment_labels_to_strings_check_save_with_some_inputs(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "B"

        paramAssignment._update_labels_cb()

        paramAssignment.label_strings_param["0"].value = "string_label_for_0"
        paramAssignment.label_strings_param["2"].value = "string_label_for_2"

        paramAssignment._confirm_settings_cb(None)

        assert config.settings["labels_to_strings"] == {
            "0": "string_label_for_0",
            "1": "1",
            "2": "string_label_for_2",
        }

    def test_param_assignment_labels_to_strings_check_save_with_all_inputs(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "B"

        paramAssignment._update_labels_cb()

        paramAssignment.label_strings_param["0"].value = "string_label_for_0"
        paramAssignment.label_strings_param["1"].value = "string_label_for_1"
        paramAssignment.label_strings_param["2"].value = "string_label_for_2"

        paramAssignment._confirm_settings_cb(None)

        assert config.settings["labels_to_strings"] == {
            "0": "string_label_for_0",
            "1": "string_label_for_1",
            "2": "string_label_for_2",
        }

    def test_param_assignment_selected_columns_saved_to_config_correctly(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "B"

        paramAssignment._update_labels_cb()

        paramAssignment._confirm_settings_cb(None)

        assert config.settings["id_col"] == "A"
        assert config.settings["label_col"] == "B"
        # assert config.settings["default_vars"] == ("C", "D")

    def test_AL_settings_correct_labels_on_init(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        assert alSettings.label_selector.options == []
        assert alSettings.label_selector.value == []

    def test_AL_settings_correct_features_on_init(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        assert alSettings.feature_selector.options == []
        assert alSettings.feature_selector.value == []

    def test_AL_settings_correct_labels_on_update_no_df(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(None)

        assert alSettings.label_selector.options == []
        assert alSettings.label_selector.value == []

    def test_AL_settings_correct_features_on_update_no_df(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(None)

        assert alSettings.feature_selector.options == []
        assert alSettings.feature_selector.value == []

    def test_AL_settings_correct_labels_on_update_df(self):

        df = self._create_test_df()

        labels = df[config.settings["label_col"]].astype(str).unique()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(df)

        assert alSettings.label_selector.options == list(labels)
        assert alSettings.label_selector.value == []

    def test_AL_settings_correct_features_on_update_df(self):

        df = self._create_test_df()

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "extra_image_cols": [],
        }

        features = list(df.columns)

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(df)

        features.remove("A")
        features.remove("B")

        assert alSettings.feature_selector.options == list(features)
        assert alSettings.feature_selector.value == []

    def test_AL_settings_check_is_complete_on_init(self):

        alSettings = ActiveLearningSettings(None, mode="AL")

        assert not alSettings.is_complete()

    def test_AL_settings_check_is_complete_on_update_no_df(self):

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(None)

        assert not alSettings.is_complete()

    def test_AL_settings_check_is_complete_on_update_df(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(df)

        assert not alSettings.is_complete()

    def test_AL_settings_get_df_on_init(self):

        alSettings = ActiveLearningSettings(None, mode="AL")

        assert alSettings.get_df() is None

    def test_AL_settings_get_df_on_update_no_df(self):

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(None)

        assert alSettings.get_df() is None

    def test_AL_settings_get_df_on_update_df(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(df)

        assert alSettings.get_df() is df

    def test_AL_settings_correct_feature_generator_on_init(self):

        alSettings = ActiveLearningSettings(None, mode="AL")

        ans = feature_generation.get_oper_dict().keys()

        assert alSettings.feature_generator.options == list(ans)

    def test_AL_settings_correct_feature_generator_on_update_no_df(self):

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(None)

        ans = feature_generation.get_oper_dict().keys()

        assert alSettings.feature_generator.options == list(ans)

    def test_AL_settings_correct_feature_generator_on_update_df(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(df)

        ans = feature_generation.get_oper_dict().keys()
        assert alSettings.feature_generator.options == list(ans)

    def test_AL_settings_add_feature_cb_test(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(df)

        alSettings.feature_generator.options = ["a", "b", "c", "d"]
        alSettings.feature_generator.value = "b"
        alSettings.feature_generator_number.value = 4

        alSettings._add_feature_selector_cb(None)

        assert alSettings.feature_generator_selected == [["b", 4]]
        pd.testing.assert_frame_equal(
            alSettings._feature_generator_dataframe.value,
            pd.DataFrame([["b", 4]], columns=["oper", "n"]),
        )

    def test_AL_settings_add_multiple_feature_cb_test(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(df)

        alSettings.feature_generator.options = ["a", "b", "c", "d"]
        alSettings.feature_generator.value = "b"
        alSettings.feature_generator_number.value = 4

        alSettings._add_feature_selector_cb(None)

        assert alSettings.feature_generator_selected == [["b", 4]]
        pd.testing.assert_frame_equal(
            alSettings._feature_generator_dataframe.value,
            pd.DataFrame([["b", 4]], columns=["oper", "n"]),
        )

        alSettings.feature_generator.value = "a"
        alSettings.feature_generator_number.value = 2

        alSettings._add_feature_selector_cb(None)

        assert alSettings.feature_generator_selected == [["b", 4], ["a", 2]]
        pd.testing.assert_frame_equal(
            alSettings._feature_generator_dataframe.value,
            pd.DataFrame([["b", 4], ["a", 2]], columns=["oper", "n"]),
        )

    def test_AL_settings_add_duplicate_feature_cb_test(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(df)

        alSettings.feature_generator.options = ["a", "b", "c", "d"]
        alSettings.feature_generator.value = "b"
        alSettings.feature_generator_number.value = 4

        alSettings._add_feature_selector_cb(None)

        assert alSettings.feature_generator_selected == [["b", 4]]
        pd.testing.assert_frame_equal(
            alSettings._feature_generator_dataframe.value,
            pd.DataFrame([["b", 4]], columns=["oper", "n"]),
        )

        alSettings._add_feature_selector_cb(None)

        assert alSettings.feature_generator_selected == [["b", 4]]
        pd.testing.assert_frame_equal(
            alSettings._feature_generator_dataframe.value,
            pd.DataFrame([["b", 4]], columns=["oper", "n"]),
        )

    def test_AL_settings_add_multiple_and_duplicate_feature_cb_test(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(df)

        alSettings.feature_generator.options = ["a", "b", "c", "d"]
        alSettings.feature_generator.value = "b"
        alSettings.feature_generator_number.value = 4

        alSettings._add_feature_selector_cb(None)

        assert alSettings.feature_generator_selected == [["b", 4]]
        pd.testing.assert_frame_equal(
            alSettings._feature_generator_dataframe.value,
            pd.DataFrame([["b", 4]], columns=["oper", "n"]),
        )

        alSettings.feature_generator.value = "a"
        alSettings.feature_generator_number.value = 2

        alSettings._add_feature_selector_cb(None)

        assert alSettings.feature_generator_selected == [["b", 4], ["a", 2]]
        pd.testing.assert_frame_equal(
            alSettings._feature_generator_dataframe.value,
            pd.DataFrame([["b", 4], ["a", 2]], columns=["oper", "n"]),
        )

        alSettings.feature_generator.value = "b"
        alSettings.feature_generator_number.value = 4

        alSettings._add_feature_selector_cb(None)

        assert alSettings.feature_generator_selected == [["b", 4], ["a", 2]]
        pd.testing.assert_frame_equal(
            alSettings._feature_generator_dataframe.value,
            pd.DataFrame([["b", 4], ["a", 2]], columns=["oper", "n"]),
        )

    def test_AL_settings_remove_feature_cb_test(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(df)

        alSettings.feature_generator_selected = [["b", 4]]
        alSettings._feature_generator_dataframe.value = pd.DataFrame(
            [["b", 4]], columns=["oper", "n"]
        )

        alSettings._remove_feature_selector_cb(None)

        assert alSettings.feature_generator_selected == []

        pd.testing.assert_frame_equal(
            alSettings._feature_generator_dataframe.value,
            pd.DataFrame([], columns=["oper", "n"]),
        )

    def test_AL_settings_remove_feature_empty_cb_test(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(df)

        alSettings._remove_feature_selector_cb(None)

        assert alSettings.feature_generator_selected == []

        pd.testing.assert_frame_equal(
            alSettings._feature_generator_dataframe.value,
            pd.DataFrame([], columns=["oper", "n"]),
        )

    def test_AL_settings_add_remove_multiple_duplicate_feature_cb_test(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None, mode="AL")

        alSettings.update_data(df)

        alSettings.feature_generator.options = ["a", "b", "c", "d"]
        alSettings.feature_generator.value = "b"
        alSettings.feature_generator_number.value = 4

        alSettings._add_feature_selector_cb(None)

        alSettings.feature_generator.value = "a"
        alSettings.feature_generator_number.value = 2

        alSettings._add_feature_selector_cb(None)

        alSettings._remove_feature_selector_cb(None)

        alSettings.feature_generator.value = "b"
        alSettings.feature_generator_number.value = 4

        alSettings._add_feature_selector_cb(None)

        alSettings._remove_feature_selector_cb(None)

        assert alSettings.feature_generator_selected == []
        pd.testing.assert_frame_equal(
            alSettings._feature_generator_dataframe.value,
            pd.DataFrame([], columns=["oper", "n"]),
        )

    def test_AL_settings_confirm_settings_labels_to_train(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button, mode="AL")

        alSettings.update_data(df)

        alSettings.label_selector.options = ["a", "b", "c", "d"]
        alSettings.label_selector.value = ["a", "b", "c"]

        alSettings._confirm_settings_cb(None)

        assert config.settings["labels_to_train"] == ["a", "b", "c"]

    def test_AL_settings_confirm_settings_features_for_training(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button, mode="AL")

        alSettings.update_data(df)

        alSettings.feature_selector.options = ["a", "b", "c", "d"]
        alSettings.feature_selector.value = ["a", "b", "c"]

        alSettings._confirm_settings_cb(None)

        assert config.settings["features_for_training"] == ["a", "b", "c"]

    def test_AL_settings_confirm_settings_exclude_labels_true(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button, mode="AL")

        alSettings.update_data(df)

        alSettings.exclude_labels_checkbox.value = True

        alSettings.label_selector.value = ["0", "1"]

        alSettings._verify_valid_selection_cb(None)

        print(alSettings.exclude_labels_checkbox.disabled)

        alSettings._confirm_settings_cb(None)

        assert config.settings["exclude_labels"]

    def test_AL_settings_confirm_settings_exclude_labels_false(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button, mode="AL")

        alSettings.update_data(df)

        alSettings.label_selector.value = ["0", "1"]

        alSettings._verify_valid_selection_cb(None)

        alSettings.exclude_labels_checkbox.value = False

        alSettings._confirm_settings_cb(None)

        assert not config.settings["exclude_labels"]

    def test_AL_settings_confirm_settings_unclassified_labels(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button, mode="AL")

        alSettings.update_data(df)

        alSettings.label_selector.options = ["a", "b", "c", "d"]
        alSettings.label_selector.value = ["a", "b", "c"]

        alSettings._confirm_settings_cb(None)

        assert config.settings["unclassified_labels"] == ["d"]

    def test_AL_settings_confirm_settings_scale_data_true(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button, mode="AL")

        alSettings.update_data(df)

        alSettings.scale_features_checkbox.value = True

        alSettings._confirm_settings_cb(None)

        assert config.settings["scale_data"]

    def test_AL_settings_confirm_settings_scale_data_false(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button, mode="AL")

        alSettings.update_data(df)

        alSettings.scale_features_checkbox.value = False

        alSettings._confirm_settings_cb(None)

        assert not config.settings["scale_data"]

    def test_AL_settings_confirm_settings_feature_generation(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button, mode="AL")

        alSettings.update_data(df)

        alSettings.feature_generator.options = ["a", "b", "c", "d"]
        alSettings.feature_generator.value = "b"
        alSettings.feature_generator_number.value = 4

        alSettings._add_feature_selector_cb(None)

        alSettings._confirm_settings_cb(None)

        assert config.settings["feature_generation"] == [["b", 4]]

    def test_data_selection_check_config_load_level_no_load_config(self):

        from astropy.table import Table

        t = Table([[1, 2], [4, 5], [7, 8]], names=("a", "b", "c"))

        check_folder_exists("data")
        check_folder_exists("configs")
        t.write("data/table1.fits", format="fits")

        src = ColumnDataSource()
        ds = DataSelection(src, "AL")

        ds.load_layout_check = False

        ds.dataset = "data/table1.fits"

        ds._load_data_cb(None)

        os.remove("data/table1.fits")

        assert "config_load_level" not in list(config.settings.keys())

    def test_data_selection_check_config_load_level_layout_only(self, document, comm):

        os.makedirs("configs/", exist_ok=True)
        if not os.path.exists("configs/test.json"):
            os.system("""echo '{"testing":true}' > configs/test.json""")

        src = ColumnDataSource()
        ds = DataSelection(src, "AL")

        ds.load_layout_check = True

        ds.load_config_select = "Only load layout. Let me choose all my own settings"
        ds.config_file = "configs/test.json"

        button = ds.load_data_button_js

        widget = ds.load_data_button_js.get_root(document, comm=comm)

        button._server_click(document, widget.ref["id"], None)

        button._process_events({"clicks": 1})

        assert config.settings["config_load_level"] == 0

    def test_data_selection_check_config_load_level_0_settings_only(self):

        check_folder_exists("configs")
        if not os.path.exists("configs/test.json"):
            os.system("""echo '{"testing":true}' > configs/test.json""")

        src = ColumnDataSource()
        ds = DataSelection(src, "AL")

        ds.load_layout_check = True

        ds.load_config_select = "Only load layout. Let me choose all my own settings"
        ds.config_file = "configs/test.json"

        ds._update_layout_file_cb()

        assert config.settings["config_load_level"] == 0

    def test_data_selection_check_config_load_level_1_settings_only(self):

        check_folder_exists("configs")
        if not os.path.exists("configs/test.json"):
            os.system("""echo '{"testing":true}' > configs/test.json""")

        src = ColumnDataSource()
        ds = DataSelection(src, "AL")

        ds.load_layout_check = True

        ds.load_config_select = (
            "Load all settings but let me train the model from scratch."
        )

        ds.config_file = "configs/test.json"

        ds._update_layout_file_cb()

        assert config.settings["config_load_level"] == 1

    def test_data_selection_check_config_load_level_2_settings_only(self):

        check_folder_exists("configs")
        if not os.path.exists("configs/test.json"):
            os.system("""echo '{"testing":true}' > configs/test.json""")

        src = ColumnDataSource()
        ds = DataSelection(src, "AL")

        ds.load_layout_check = True

        ds.load_config_select = (
            "Load all settings and train model with provided labels."
        )
        ds.config_file = "configs/test.json"

        ds._update_layout_file_cb()

        assert config.settings["config_load_level"] == 2

    def test_data_selection_get_dataframe_from_fits_non_optimised_parameter(self):

        from astropy.table import Table

        t = Table([[1, 2], [4, 5], [7, 8]], names=("a", "b", "c"))

        check_folder_exists("data")
        t.write("data/table1.fits", format="fits")

        src = ColumnDataSource()
        ds = DataSelection(src, "AL")

        ds.memory_optimisation_check.value = False

        df = ds.get_dataframe_from_fits_file("data/table1.fits", optimise_data=False)

        os.remove("data/table1.fits")

        pd.testing.assert_frame_equal(
            df, pd.DataFrame([[1, 4, 7], [2, 5, 8]], columns=["a", "b", "c"])
        )

    def test_data_selection_get_dataframe_from_fits_optimised_parameter(self):

        from astropy.table import Table

        t = Table([[1, 2], [4, 5], [7, 8]], names=("a", "b", "c"))

        check_folder_exists("data")
        t.write("data/table1.fits", format="fits")

        src = ColumnDataSource()
        ds = DataSelection(src, "AL")

        ds.memory_optimisation_check.value = False

        df = ds.get_dataframe_from_fits_file("data/table1.fits", optimise_data=True)

        os.remove("data/table1.fits")

        new_df = pd.DataFrame([[1, 4, 7], [2, 5, 8]], columns=["a", "b", "c"])

        new_df = new_df.astype("int8")

        pd.testing.assert_frame_equal(df, new_df)

    def test_data_selection_get_dataframe_from_fits_optimised_checkbox(self):

        from astropy.table import Table

        t = Table([[1, 2], [4, 5], [7, 8]], names=("a", "b", "c"))

        check_folder_exists("data")
        t.write("data/table1.fits", format="fits")

        src = ColumnDataSource()
        ds = DataSelection(src, "AL")

        ds.memory_optimisation_check.value = True

        df = ds.get_dataframe_from_fits_file("data/table1.fits")

        os.remove("data/table1.fits")

        new_df = pd.DataFrame([[1, 4, 7], [2, 5, 8]], columns=["a", "b", "c"])

        new_df = new_df.astype("int8")

        pd.testing.assert_frame_equal(df, new_df)
        assert config.settings["optimise_data"]

    def test_data_selection_get_dataframe_from_fits_non_optimised_checkbox(self):

        from astropy.table import Table

        t = Table([[1, 2], [4, 5], [7, 8]], names=("a", "b", "c"))

        check_folder_exists("data")
        t.write("data/table1.fits", format="fits")

        src = ColumnDataSource()
        ds = DataSelection(src, "AL")

        ds.memory_optimisation_check.value = False

        df = ds.get_dataframe_from_fits_file("data/table1.fits")

        os.remove("data/table1.fits")

        new_df = pd.DataFrame([[1, 4, 7], [2, 5, 8]], columns=["a", "b", "c"])

        pd.testing.assert_frame_equal(df, new_df)
        assert not config.settings["optimise_data"]

    def test_data_selection_get_df_after_load_non_optimised(self):

        from astropy.table import Table

        t = Table([[1, 2], [4, 5], [7, 8]], names=("a", "b", "c"))

        check_folder_exists("data")
        t.write("data/table1.fits", format="fits")

        src = ColumnDataSource()
        ds = DataSelection(src, "AL")

        ds.memory_optimisation_check.value = False

        ds.dataset = "data/table1.fits"

        ds._load_data_cb(None)

        os.remove("data/table1.fits")

        df = ds.get_df()

        new_df = pd.DataFrame([[1, 4, 7], [2, 5, 8]], columns=["a", "b", "c"])

        pd.testing.assert_frame_equal(df, new_df)
        assert ds.ready
        assert not config.settings["optimise_data"]

    def test_data_selection_get_df_after_load_optimised(self):

        from astropy.table import Table

        t = Table([[1, 2], [4, 5], [7, 8]], names=("a", "b", "c"))

        check_folder_exists("data")
        t.write("data/table1.fits", format="fits")

        src = ColumnDataSource()
        ds = DataSelection(src, "AL")

        ds.memory_optimisation_check.value = True

        ds.dataset = "data/table1.fits"

        ds._load_data_cb(None)

        os.remove("data/table1.fits")

        df = ds.get_df()

        new_df = pd.DataFrame([[1, 4, 7], [2, 5, 8]], columns=["a", "b", "c"])
        new_df = new_df.astype("int8")

        pd.testing.assert_frame_equal(df, new_df)
        assert ds.ready
        assert config.settings["optimise_data"]


class TestDashboards:
    def _create_test_df(self):

        data = []

        for i in range(100):
            data.append([str(i), i % 3, i, i, i])

        df = pd.DataFrame(data, columns=list("ABCDE"))

        return df

    def _create_test_df_with_image_data(self):

        data = []

        for i in range(100):
            if i % 2 == 0:
                data.append([str(i), i % 3, i, i, i, "178.52904,2.1655949", ""])
            else:
                data.append(
                    [
                        i,
                        i % 3,
                        i,
                        i,
                        i,
                        "178.52904,2.1655949",
                        "https://dr15.sdss.org/sas/dr15/sdss/spectro/redux/images/v5_10_0/8125-56955/spec-image-8125-56955-0534.png",
                    ]
                )

        df = pd.DataFrame(data, columns=list("ABCDE") + ["ra_dec", "png_path_DR16"])

        return df

    def test_dashboard_init(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        dashboard = Dashboard(src=src)

        assert dashboard.src.data == src.data
        assert dashboard.contents == "Menu"

    @pytest.mark.parametrize(
        "contents_name",
        [
            "Menu",
            "Settings",
            "Menu",
            "Basic Plot",
            "Selected Source Info",
        ],
        ids=str,
    )
    def test_dashboard_closing_button_check_contents(self, contents_name):
        data = self._create_test_df()
        config.main_df = data

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["1"],
            "features_for_training": ["C", "D"],
            "exclude_unknown_labels": False,
            "exclude_labels": True,
            "unclassified_labels": ["0", "2"],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
            "extra_image_cols": [],
            "confirmed": False,
        }

        src = ColumnDataSource()
        dashboard = Dashboard(src=src, contents=contents_name)
        dashboard._close_button_cb(None)

        assert dashboard.contents == "Menu"

    def test_selected_source_init_no_selected(self):

        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == []
        assert selected_source._url_optical_image == ""
        assert selected_source._search_status == ""
        assert selected_source._image_zoom == 0.2

    def test_selected_source_init_selected(self):

        data = self._create_test_df()
        config.main_df = data
        data = data.iloc[5]
        src = ColumnDataSource({str(c): [v] for c, v in data.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == ["5"]
        assert selected_source._url_optical_image == ""
        assert selected_source._search_status == ""
        assert selected_source._image_zoom == 0.2

        data = self._create_test_df()
        data = data.iloc[15]
        src = ColumnDataSource({str(c): [v] for c, v in data.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == ["15"]
        assert selected_source._url_optical_image == ""
        assert selected_source._search_status == ""
        assert selected_source._image_zoom == 0.2

        data = self._create_test_df()
        data = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == ["72"]
        assert selected_source._url_optical_image == ""
        assert selected_source._search_status == ""
        assert selected_source._image_zoom == 0.2

    def test_selected_source_check_history_from_none_to_selected(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == []

        data_selected = data.iloc[72]
        src.data = {str(c): [v] for c, v in data_selected.items()}

        assert selected_source.selected_history == ["72"]

    def test_selected_source_check_history_from_selected_to_selected_unique(self):
        data = self._create_test_df()
        data_selected = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == ["72"]

        data_selected = data.iloc[30]
        src.data = {str(c): [v] for c, v in data_selected.items()}

        assert selected_source.selected_history == ["30", "72"]

    def test_selected_source_check_history_from_selected_to_selected_same_id_head(self):
        data = self._create_test_df()
        config.main_df = data
        data_selected = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == ["72"]

        data_selected = data.iloc[72]
        src.data = {str(c): [v] for c, v in data_selected.items()}

        assert selected_source.selected_history == ["72"]

    def test_selected_source_check_history_from_selected_to_selected_no_id(self):
        data = self._create_test_df()
        config.main_df = data
        data_selected = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == ["72"]

        data_selected = data.iloc[53].copy()
        data_selected["A"] = ""
        src.data = {str(c): [v] for c, v in data_selected.items()}

        assert selected_source.selected_history == ["72"]

    def test_selected_source_check_history_from_selected_to_selected_same_id_throughout(
        self,
    ):
        data = self._create_test_df()
        config.main_df = data
        data_selected = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        data_selected = data.iloc[30]
        src.data = {str(c): [v] for c, v in data_selected.items()}

        data_selected = data.iloc[30]
        src.data = {str(c): [v] for c, v in data_selected.items()}

        data_selected = data.iloc[72]
        src.data = {str(c): [v] for c, v in data_selected.items()}

        assert selected_source.selected_history == ["72", "30", "72"]

    def test_selected_source_empty_selected_from_non_selected(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        selected_source.empty_selected()

        empty = {}

        assert selected_source.src.data == empty

    def test_selected_source_empty_selected_from_selected(self):
        data = self._create_test_df()
        config.main_df = data
        data_selected = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        selected_source.empty_selected()

        empty = {"A": [], "B": [], "C": [], "D": [], "E": []}

        assert selected_source.src.data == empty

    def test_selected_source_check_valid_select_from_non_selected(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        valid = selected_source._check_valid_selected()

        assert not valid

    def test_selected_source_check_valid_select_from_selected_is_valid(self):
        data = self._create_test_df()
        config.main_df = data
        data_selected = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        valid = selected_source._check_valid_selected()

        assert valid

    def test_selected_source_check_valid_select_from_selected_is_not_valid(self):
        data = self._create_test_df()
        config.main_df = data
        data_selected = data.iloc[72].copy()
        data_selected["A"] = 500

        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        valid = selected_source._check_valid_selected()

        assert not valid

    def test_selected_source_check_ra_dec_conv(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        new_url = selected_source._generate_radio_url(121.5042, -34.1172)

        url1 = "https://third.ucllnl.org/cgi-bin/firstimage?RA="
        url2 = "&Equinox=J2000&ImageSize=2.5&MaxInt=200&GIF=1"
        url = f"{url1}8.0 6.0 1.0 -34.0 7.0 2.0{url2}"
        print(new_url)
        print(url)

        assert new_url == url

        new_url = selected_source._generate_radio_url(183.0500, 45.7625)

        url1 = "https://third.ucllnl.org/cgi-bin/firstimage?RA="
        url2 = "&Equinox=J2000&ImageSize=2.5&MaxInt=200&GIF=1"
        url = f"{url1}12.0 12.0 12.0 45.0 45.0 45.0{url2}"
        print(new_url)
        print(url)

        assert new_url == url

        new_url = selected_source._generate_radio_url(182.2667, 1.1358)

        url1 = "https://third.ucllnl.org/cgi-bin/firstimage?RA="
        url2 = "&Equinox=J2000&ImageSize=2.5&MaxInt=200&GIF=1"
        url = f"{url1}12.0 9.0 4.0 1.0 8.0 9.0{url2}"
        print(new_url)
        print(url)

        assert new_url == url

        new_url = selected_source._generate_radio_url(15.2542, 1.0169)

        url1 = "https://third.ucllnl.org/cgi-bin/firstimage?RA="
        url2 = "&Equinox=J2000&ImageSize=2.5&MaxInt=200&GIF=1"
        url = f"{url1}1.0 1.0 1.0 1.0 1.0 1.0{url2}"
        print(new_url)
        print(url)

        assert new_url == url

        new_url = selected_source._generate_radio_url(15.2542, -1.0169)

        url1 = "https://third.ucllnl.org/cgi-bin/firstimage?RA="
        url2 = "&Equinox=J2000&ImageSize=2.5&MaxInt=200&GIF=1"
        url = f"{url1}1.0 1.0 1.0 -1.0 1.0 1.0{url2}"
        print(new_url)
        print(url)

        assert new_url == url

        new_url = selected_source._generate_radio_url(167.7958, 11.1864)

        url1 = "https://third.ucllnl.org/cgi-bin/firstimage?RA="
        url2 = "&Equinox=J2000&ImageSize=2.5&MaxInt=200&GIF=1"
        url = f"{url1}11.0 11.0 11.0 11.0 11.0 11.0{url2}"
        print(new_url)
        print(url)

        assert new_url == url

        new_url = selected_source._generate_radio_url(167.7958, -11.1864)

        url1 = "https://third.ucllnl.org/cgi-bin/firstimage?RA="
        url2 = "&Equinox=J2000&ImageSize=2.5&MaxInt=200&GIF=1"
        url = f"{url1}11.0 11.0 11.0 -11.0 11.0 11.0{url2}"
        print(new_url)
        print(url)

        assert new_url == url

    def test_selected_source_change_zoom_in(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        selected_source._change_zoom_cb(None, "in")
        assert selected_source._image_zoom == 0.1

    def test_selected_source_change_zoom_in_check_stop(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        selected_source._change_zoom_cb(None, "in")
        selected_source._change_zoom_cb(None, "in")
        selected_source._change_zoom_cb(None, "in")
        selected_source._change_zoom_cb(None, "in")
        assert selected_source._image_zoom == 0.1

    def test_selected_source_change_zoom_out(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        selected_source._change_zoom_cb(None, "out")
        assert selected_source._image_zoom == 0.3

    def test_selected_source_change_zoom_exception(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)
        selected_source._url_optical_image = "INVALID_URL&"

        selected_source._change_zoom_cb(None, "out")

        assert selected_source._image_zoom == 0.3

    def test_selected_source_search_empty(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        event = Event("", "")
        selected_source._change_selected(event)

        assert selected_source._search_status == ""

    def test_selected_source_search_invalid_id(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        event = Event("DOESNT_EXIST", "")
        selected_source._change_selected(event)

        assert selected_source._search_status == "ID not found in dataset"

    def test_selected_source_search_valid_id(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        event = Event("5", "")
        selected_source._change_selected(event)

        print(selected_source.src.data)

        assert selected_source._search_status == "Searching..."
        assert selected_source.src.data == src.data
        assert selected_source.src.data["A"] == ["5"]

    def test_selected_source_check_required_column_has_column(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        has_column = selected_source.check_required_column("A")

        assert has_column

    def test_selected_source_check_required_column_missing_column(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        has_column = selected_source.check_required_column("F")

        assert not has_column

    def test_selected_source_search_deselect(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        event = Event("5", "")
        selected_source._change_selected(event)

        selected_source._deselect_source_cb(None)

        assert selected_source.search_id.value == ""
        assert selected_source._search_status == ""
        assert selected_source.src.data == {"A": [], "B": [], "C": [], "D": [], "E": []}

    def test_selected_source_update_default_images(self):
        data = self._create_test_df_with_image_data()
        config.main_df = data
        data_selected = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)
        selected_source._update_default_images()

        ra_dec = selected_source.src.data["ra_dec"][0]
        ra = ra_dec[: ra_dec.index(",")]
        dec = ra_dec[ra_dec.index(",") + 1 :]

        url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra="
        _url_optical_image = (
            f"{url}{ra}&dec={dec}&opt=G&scale={selected_source._image_zoom}"
        )

        assert selected_source._url_optical_image == _url_optical_image

    def test_selected_source_update_custom_images(self):
        data = self._create_test_df_with_image_data()
        config.main_df = data

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["1"],
            "features_for_training": ["C", "D"],
            "exclude_unknown_labels": False,
            "exclude_labels": True,
            "unclassified_labels": ["0", "2"],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
            "extra_image_cols": ["png_path_DR16"],
        }

        data_selected = data.iloc[71]
        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        image_tab = selected_source._create_image_tab()

        assert len(image_tab) == 1
        assert (
            image_tab[0].object
            == "https://dr15.sdss.org/sas/dr15/sdss/spectro/redux/images/v5_10_0/8125-56955/spec-image-8125-56955-0534.png"
        )

        data_selected = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        image_tab = selected_source._create_image_tab()

        assert len(image_tab) == 1
        assert image_tab[0].object == "No Image available for this source."

    def test_settings_dashboard_init(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        settings_db = SettingsDashboard(None, src)

        assert settings_db._pipeline_stage == 0

    def test_settings_dashboard_close_settings(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        main = Dashboard(src)

        settings_db = SettingsDashboard(None, src)

        settings_db._create_pipeline_cb(None, "AL", main)

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["1"],
            "features_for_training": ["C", "D"],
            "exclude_unknown_labels": False,
            "exclude_labels": True,
            "unclassified_labels": ["0", "2"],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
            "extra_image_cols": ["png_path_DR16"],
        }

        settings_db.pipeline["Active Learning Settings"].df = data
        settings_db._close_settings_cb(None, main)

        pd.testing.assert_frame_equal(config.main_df, settings_db.df)
        assert settings_db.src.data == {"A": [], "B": [], "C": [], "D": [], "E": []}
        assert main.contents == "Active Learning"

    def test_settings_dashboard_pipeline_previous(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        settings_db = SettingsDashboard(None, src)
        settings_db.create_pipeline(mode="AL")

        settings_db._pipeline_stage = 3

        settings_db._stage_previous_cb(None)

        assert settings_db._pipeline_stage == 2

    def test_settings_dashboard_pipeline_next(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        main = Dashboard(src)
        settings_db = SettingsDashboard(None, src)
        settings_db.create_pipeline(mode="AL")

        settings_db.pipeline["Select Your Data"].df = data

        settings_db._stage_next_cb(None)

        assert settings_db._pipeline_stage == 1

    def test_settings_dashboard_close_button_check_disabled(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()
        main = Dashboard(src)

        settings_db = SettingsDashboard(None, src)
        settings_db.create_pipeline(mode="AL")

        settings_db.pipeline["Active Learning Settings"].completed = False
        settings_db.panel()
        assert settings_db._close_settings_button.disabled

        settings_db.pipeline["Active Learning Settings"].completed = True
        settings_db.panel()
        assert not settings_db._close_settings_button.disabled

    def test_settings_dashboard_get_settings(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        settings_db = SettingsDashboard(None, src)
        settings_db.create_pipeline(mode="AL")

        settings_db.pipeline["Assign Parameters"].update_data(dataframe=data)
        settings_db.pipeline["Assign Parameters"].id_column = "A"
        settings_db.pipeline["Assign Parameters"].label_column = "B"
        settings_db.pipeline["Assign Parameters"].update_colours()
        settings_db.pipeline["Active Learning Settings"].default_x_variable.options = [
            "C",
            "D",
        ]
        settings_db.pipeline["Active Learning Settings"].default_y_variable.options = [
            "C",
            "D",
        ]
        settings_db.pipeline["Active Learning Settings"].default_x_variable.value = "C"
        settings_db.pipeline["Active Learning Settings"].default_y_variable.value = "D"

        updated_settings = settings_db.get_settings()

        settings = {
            "id_col": "A",
            "label_col": "B",
            "label_colours": {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"},
            "default_vars": ("C", "D"),
        }

        assert updated_settings == settings

    def test_AL_dashboard_init(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": [],
            "features_for_training": ["C", "D"],
            "exclude_unknown_labels": False,
            "exclude_labels": True,
            "unclassified_labels": ["0", "1", "2"],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
            "extra_image_cols": ["png_path_DR16"],
        }

        al_db = ActiveLearningDashboard(src, data)

        pd.testing.assert_frame_equal(al_db.df, data)
        assert al_db.src.data == src.data
        assert al_db.active_learning == []

    def test_AL_dashboard_labels_to_train_empty(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        config.settings["labels_to_train"] = []

        al_db = ActiveLearningDashboard(src, data)

        assert al_db.active_learning == []

    def test_AL_dashboard_labels_to_train_single(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["1"],
            "features_for_training": ["C", "D"],
            "exclude_unknown_labels": False,
            "exclude_labels": True,
            "unclassified_labels": ["0", "2"],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
        }

        al_db = ActiveLearningDashboard(src, data)

        assert len(al_db.active_learning) == 1
        assert len(al_db.al_tabs) == 1

    def test_AL_dashboard_labels_to_train_multiple(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "exclude_unknown_labels": False,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
        }

        al_db = ActiveLearningDashboard(src, data)

        assert len(al_db.active_learning) == 3
        assert len(al_db.al_tabs) == 3

    def test_plot_dashboard_init(self):
        data = self._create_test_df()
        config.main_df = data
        # data_selected = data.iloc[72]
        src = ColumnDataSource()

        plot_db = PlotDashboard(src, None)

        pd.testing.assert_frame_equal(plot_db.df, data)

    def test_plot_dashboard_update_df(self):
        data = self._create_test_df()
        config.main_df = data
        # data_selected = data.iloc[72]
        src = ColumnDataSource()

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
        }

        plot_db = PlotDashboard(src, None)

        data = self._create_test_df_with_image_data()
        config.main_df = data

        plot_db.update_df()

        pd.testing.assert_frame_equal(plot_db.df, data)

    def test_plot_dashboard_update_variable_lists_id_col_removed(self):
        data = self._create_test_df()
        config.main_df = data
        # data_selected = data.iloc[72]
        src = ColumnDataSource()

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
        }

        plot_db = PlotDashboard(src, None)

        assert "A" not in list(plot_db.param.X_variable.objects)
        assert "A" not in list(plot_db.param.Y_variable.objects)

    def test_plot_dashboard_update_variable_lists_non_id_col_not_removed(self):
        data = self._create_test_df()
        config.main_df = data
        # data_selected = data.iloc[72]
        src = ColumnDataSource()

        config.settings = {
            "id_col": "B",
            "label_col": "E",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
        }

        plot_db = PlotDashboard(src, None)

        assert "A" in list(plot_db.param.Y_variable.objects)
        assert "A" in list(plot_db.param.X_variable.objects)

    def test_plot_dashboard_update_variable_lists_label_col_removed(self):
        data = self._create_test_df()
        config.main_df = data
        # data_selected = data.iloc[72]
        src = ColumnDataSource()

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
        }

        plot_db = PlotDashboard(src, None)

        assert "B" not in list(plot_db.param.X_variable.objects)
        assert "B" not in list(plot_db.param.Y_variable.objects)

    def test_plot_dashboard_update_variable_lists_non_label_col_not_removed(self):
        data = self._create_test_df()
        config.main_df = data
        # data_selected = data.iloc[72]
        src = ColumnDataSource()

        config.settings = {
            "id_col": "A",
            "label_col": "E",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
        }

        plot_db = PlotDashboard(src, None)

        assert "B" in list(plot_db.param.Y_variable.objects)
        assert "B" in list(plot_db.param.X_variable.objects)

    def test_plot_dashboard_update_variable_lists_correct_columns(self):
        data = self._create_test_df()
        config.main_df = data
        # data_selected = data.iloc[72]
        src = ColumnDataSource()

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
        }

        plot_db = PlotDashboard(src, None)

        assert list(plot_db.param.X_variable.objects) == ["C", "D", "E"]
        assert list(plot_db.param.Y_variable.objects) == ["C", "D", "E"]

    def test_plot_dashboard_update_variable_lists_correct_cols_after_update(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
        }

        plot_db = PlotDashboard(src, None)

        data = self._create_test_df_with_image_data()
        config.main_df = data

        plot_db.update_variable_lists()

        assert list(plot_db.param.X_variable.objects) == [
            "C",
            "D",
            "E",
            "ra_dec",
            "png_path_DR16",
        ]
        assert list(plot_db.param.Y_variable.objects) == [
            "C",
            "D",
            "E",
            "ra_dec",
            "png_path_DR16",
        ]

    def test_plot_dashboard_check_correct_defaults_selected(self):
        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
        }

        plot_db = PlotDashboard(src, None)

        assert plot_db.param.X_variable.default == "C"
        assert plot_db.param.Y_variable.default == "D"
        assert plot_db.X_variable == "C"
        assert plot_db.Y_variable == "D"

    def _create_test_test_set(self):

        if os.path.exists("data/test_set.json"):
            os.system("mv data/test_set.json data/test_set_cp.json")

        test_data = """{ "2":0,"5":1,"7":0 }"""

        os.system(f"echo '{test_data}' > data/test_set.json")

    def test_labelling_dashboard_init(self):

        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        self._create_test_test_set()

        labelling = LabellingDashboard(src=src, df=data)

        os.remove("data/test_set.json")

        pd.testing.assert_frame_equal(config.main_df, labelling.sample_region)
        assert list(labelling.region_criteria_df.columns) == ["column", "oper", "value"]
        assert (
            labelling.region_message
            == f"All Sources Matching ({len(labelling.sample_region)})"
        )
        first_key = list(labelling.src.data.keys())[0]
        assert len(labelling.src.data[first_key]) == 1

        assert labelling.criteria_dict == {}

    def test_labelling_dashboard_check_single_matching_value_in_region(self):

        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        self._create_test_test_set()

        labelling = LabellingDashboard(src=src, df=data)

        os.remove("data/test_set.json")

        labelling.column_dropdown.value = "C"
        labelling.operation_dropdown.value = "=="
        labelling.input_value.value = "1"

        labelling.update_sample_region(None, button="ADD")

        assert len(labelling.sample_region) == 1
        assert list(labelling.criteria_dict.keys()) == ["C == 1"]
        assert labelling.criteria_dict["C == 1"] == ["C", "==", "1"]
        assert labelling.region_message == "1 Matching Sources"

    def test_labelling_dashboard_check_multiple_matching_values_in_region(self):

        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        self._create_test_test_set()

        labelling = LabellingDashboard(src=src, df=data)

        os.remove("data/test_set.json")

        labelling.column_dropdown.value = "C"
        labelling.operation_dropdown.value = "<"
        labelling.input_value.value = "5"

        labelling.update_sample_region(None, button="ADD")

        assert len(labelling.sample_region) == 5
        assert list(labelling.criteria_dict.keys()) == ["C < 5"]
        assert labelling.criteria_dict["C < 5"] == ["C", "<", "5"]
        assert labelling.region_message == "5 Matching Sources"

    def test_labelling_dashboard_check_combined_matching_values_in_region(self):

        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        self._create_test_test_set()

        labelling = LabellingDashboard(src=src, df=data)

        os.remove("data/test_set.json")

        labelling.column_dropdown.value = "C"
        labelling.operation_dropdown.value = "<"
        labelling.input_value.value = "5"

        labelling.update_sample_region(None, button="ADD")

        labelling.column_dropdown.value = "C"
        labelling.operation_dropdown.value = "=="
        labelling.input_value.value = "1"

        labelling.update_sample_region(None, button="ADD")

        assert len(labelling.sample_region) == 1
        assert list(labelling.criteria_dict.keys()) == ["C < 5", "C == 1"]
        assert labelling.criteria_dict["C == 1"] == ["C", "==", "1"]
        assert labelling.region_message == "1 Matching Sources"

        assert labelling.remove_sample_selection_dropdown.options == ["C < 5", "C == 1"]

    def test_labelling_dashboard_removing_sample_criteria(self):

        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        self._create_test_test_set()

        labelling = LabellingDashboard(src=src, df=data)

        os.remove("data/test_set.json")

        labelling.column_dropdown.value = "C"
        labelling.operation_dropdown.value = "<"
        labelling.input_value.value = "5"

        labelling.update_sample_region(None, button="ADD")

        labelling.column_dropdown.value = "C"
        labelling.operation_dropdown.value = "=="
        labelling.input_value.value = "1"

        labelling.update_sample_region(None, button="ADD")

        labelling.remove_sample_selection_dropdown.value = "C == 1"

        labelling.update_sample_region(None, button="REMOVE")

        assert len(labelling.sample_region) == 5
        assert list(labelling.criteria_dict.keys()) == ["C < 5"]
        assert labelling.criteria_dict["C < 5"] == ["C", "<", "5"]
        assert labelling.region_message == "5 Matching Sources"

        labelling.remove_sample_selection_dropdown.value = "C < 5"

        labelling.update_sample_region(None, button="REMOVE")

        assert len(labelling.sample_region) == len(config.main_df)
        assert list(labelling.criteria_dict.keys()) == []
        assert (
            labelling.region_message == f"All Sources Matching ({len(config.main_df)})"
        )

    def test_labelling_dashboard_check_looping_through_labelled(self):

        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
        }

        self._create_test_test_set()

        labelling = LabellingDashboard(src=src, df=data)

        os.remove("data/test_set.json")

        assert len(labelling.src.data[config.settings["id_col"]]) == 1

        labelling.update_selected_point_from_buttons(None, button="<")

        first_key = list(labelling.src.data.keys())[0]
        assert len(labelling.src.data[first_key]) == 1

        assert labelling.src.data["A"][0] == "7"
        assert labelling.labels[labelling.src.data[config.settings["id_col"]][0]] == 0

        labelling.update_selected_point_from_buttons(None, button="<")

        first_key = list(labelling.src.data.keys())[0]
        assert len(labelling.src.data[first_key]) == 1

        assert labelling.src.data["A"][0] == "5"
        assert labelling.labels[labelling.src.data[config.settings["id_col"]][0]] == 1
        assert labelling.next_labelled_button.disabled == False
        assert labelling.prev_labelled_button.disabled == False

        labelling.update_selected_point_from_buttons(None, button="<")

        first_key = list(labelling.src.data.keys())[0]
        assert len(labelling.src.data[first_key]) == 1

        assert labelling.src.data["A"][0] == "2"
        assert labelling.labels[labelling.src.data[config.settings["id_col"]][0]] == 0
        assert labelling.next_labelled_button.disabled == False
        assert labelling.prev_labelled_button.disabled == True

        assert len(labelling.src.data[config.settings["id_col"]]) == 1

        labelling.update_selected_point_from_buttons(None, button=">")

        first_key = list(labelling.src.data.keys())[0]
        assert len(labelling.src.data[first_key]) == 1

        assert labelling.src.data["A"][0] == "5"
        assert labelling.next_labelled_button.disabled == False
        assert labelling.prev_labelled_button.disabled == False

        labelling.update_selected_point_from_buttons(None, button=">")

        first_key = list(labelling.src.data.keys())[0]
        assert len(labelling.src.data[first_key]) == 1

        assert labelling.src.data["A"][0] == "7"

        labelling.update_selected_point_from_buttons(None, button="First")

        first_key = list(labelling.src.data.keys())[0]
        assert len(labelling.src.data[first_key]) == 1

        assert labelling.src.data["A"][0] == "2"
        assert labelling.labels[labelling.src.data[config.settings["id_col"]][0]] == 0
        assert labelling.next_labelled_button.disabled == False
        assert labelling.prev_labelled_button.disabled == True

        assert len(labelling.src.data[config.settings["id_col"]]) == 1

    def test_labelling_dashboard_check_new_button_with_single_match(self):

        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        self._create_test_test_set()

        labelling = LabellingDashboard(src=src, df=data)

        os.remove("data/test_set.json")

        labelling.column_dropdown.value = "C"
        labelling.operation_dropdown.value = "=="
        labelling.input_value.value = "1"

        labelling.update_sample_region(None, button="ADD")

        labelling.update_selected_point_from_buttons(None, button="New")

        assert len(labelling.sample_region) == 1
        assert list(labelling.criteria_dict.keys()) == ["C == 1"]
        assert labelling.criteria_dict["C == 1"] == ["C", "==", "1"]
        assert labelling.region_message == "1 Matching Sources"

    def test_labelling_dashboard_check_zero_matching(self):

        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        self._create_test_test_set()

        labelling = LabellingDashboard(src=src, df=data)

        os.remove("data/test_set.json")

        labelling.column_dropdown.value = "C"
        labelling.operation_dropdown.value = "=="
        labelling.input_value.value = "1"

        labelling.update_sample_region(None, button="ADD")

        labelling.column_dropdown.value = "C"
        labelling.operation_dropdown.value = "=="
        labelling.input_value.value = "2"

        labelling.update_sample_region(None, button="ADD")

        assert len(labelling.sample_region) == 0
        assert labelling.region_message == "No Matching Sources!"

        labelling.update_selected_point_from_buttons(None, button="New")

        assert labelling.new_labelled_button.disabled == True

    def test_labelling_dashboard_save_assigned_label(self):

        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        config.settings = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
        }

        self._create_test_test_set()

        labelling = LabellingDashboard(src=src, df=data)

        selected = labelling.src.data["A"][0]

        labelling.assign_label_group.value = "0"

        labelling._assign_label_cb(None)

        with open("data/test_set.json", "r") as json_file:
            labels = json.load(json_file)

        should_be = {"2": 0, "5": 1, "7": 0}
        should_be[selected] = 0

        assert labels == should_be

        os.remove("data/test_set.json")


class TestExtensions:
    def _create_test_df(self):

        data = []

        for i in range(100):
            data.append([str(i), i % 3, i, i, i])

        df = pd.DataFrame(data, columns=list("ABCDE"))

        return df

    def test_feature_extension_subtract(self):

        config.settings = {"features_for_training": ["C", "D", "E"]}

        data = self._create_test_df()

        data_df = data.loc[:, ["C", "D", "E"]]

        new_data, gen_features = feature_generation.subtract(data_df, 2)

        expect_features = ["C", "D", "E", "C-D", "C-E", "D-E"]
        assert len(expect_features) == len(new_data.columns)
        for feature in expect_features:
            assert feature in list(new_data.columns)

        for col in ["C-D", "C-E", "D-E"]:
            assert list(new_data[col].unique()) == [0]

    def test_feature_extension_add(self):

        config.settings = {"features_for_training": ["C", "D", "E"]}

        data = self._create_test_df()

        data_df = data.loc[:, ["C", "D", "E"]]

        new_data, gen_features = feature_generation.add(data_df, 2)

        expect_features = ["C", "D", "E", "C+D", "C+E", "D+E"]
        assert len(expect_features) == len(new_data.columns)
        for feature in expect_features:
            assert feature in list(new_data.columns)

        for col in ["C+D", "C+E", "D+E"]:
            assert list(new_data[col].unique()) == [2 * x for x in range(len(data))]

    def test_feature_extension_multiply(self):

        config.settings = {"features_for_training": ["C", "D", "E"]}

        data = self._create_test_df()

        data_df = data.loc[:, ["C", "D", "E"]]

        new_data, gen_features = feature_generation.multiply(data_df, 2)

        expect_features = ["C", "D", "E", "C*D", "C*E", "D*E"]
        assert len(expect_features) == len(new_data.columns)
        for feature in expect_features:
            assert feature in list(new_data.columns)

        for col in ["C*D", "C*E", "D*E"]:
            assert list(new_data[col].unique()) == [x ** 2 for x in range(len(data))]

    def test_feature_extension_divide(self):

        config.settings = {"features_for_training": ["C", "D", "E"]}

        data = self._create_test_df()

        data_df = data.loc[:, ["C", "D", "E"]]

        new_data, gen_features = feature_generation.divide(data_df, 2)

        expect_features = ["C", "D", "E", "C/D", "C/E", "D/E"]
        assert len(expect_features) == len(new_data.columns)
        for feature in expect_features:
            assert feature in list(new_data.columns)

        for col in ["C/D", "C/E", "D/E"]:
            assert np.isnan(list(new_data[col].unique())[0])
            assert list(new_data[col].unique())[1] == 1.0
            assert len(list(new_data[col].unique())) == 2


class TestUtils:
    def _create_test_df(self):

        data = []

        for i in range(100):
            data.append([str(i), i % 3, i, i, i])

        df = pd.DataFrame(data, columns=list("ABCDE"))

        return df

    def test_load_config_update_config_settings(self):

        config.settings = {}

        imported_config = {
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {0: "#ffad0e", 1: "#0057ff", 2: "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
            "extra_image_cols": ["png_path_DR16"],
        }

        load_config.update_config_settings(imported_config)

        del config.settings["confirmed"]

        assert config.settings == imported_config, print(
            f"{config.settings},\n\n {imported_config}"
        )

    def test_save_config_save_config_file(self):

        config.settings = {
            "Author": "",
            "doi": "",
            "dataset_filepath": "",
            "layout": {
                "0": {"x": 0, "y": 0, "w": 6, "h": 6, "contents": "Menu"},
                "1": {
                    "x": 0,
                    "y": 6,
                    "w": 6,
                    "h": 6,
                    "contents": "Menu",
                },
                "2": {"x": 6, "y": 0, "w": 6, "h": 6, "contents": "Settings"},
            },
            "id_col": "A",
            "label_col": "B",
            "default_vars": ["C", "D"],
            "labels": [0, 1, 2],
            "label_colours": {"0": "#ffad0e", "1": "#0057ff", "2": "#a2a2a2"},
            "labels_to_strings": {"0": "0", "1": "1", "2": "2"},
            "strings_to_labels": {"0": 0, "1": 1, "2": 2},
            "extra_info_cols": [
                "C",
            ],
            "labels_to_train": ["0", "1", "2"],
            "features_for_training": ["C", "D"],
            "exclude_labels": True,
            "unclassified_labels": [],
            "scale_data": False,
            "feature_generation": [["subtract (a-b)", 2]],
            "extra_image_cols": ["png_path_DR16"],
            "confirmed": False,
            "optimise_data": True,
            "exclude_unknown_labels": True,
            "classifiers": [],
        }

        example_layout = """{"0":{"x":0,"y":0,"w":6,"h":6},"1":{"x":0,"y":6,"w":6,"h":6},"2":{"x":6,"y":0,"w":6,"h":6}}"""

        data = self._create_test_df()
        config.main_df = data
        src = ColumnDataSource()

        config.dashboards = {
            "0": Dashboard(src, contents="Menu"),
            "1": Dashboard(src, contents="Menu"),
            "2": Dashboard(src, contents="Settings"),
        }

        save_config.save_config_file(example_layout, TextAreaInput(value=""), test=True)

        with open("configs/config_export.json") as layout_file:
            created_file = json.load(layout_file)

        for k in list(created_file.keys()):
            assert (
                created_file[k] == config.settings[k]
            ), f"{k}: {created_file[k]} != {config.settings[k]}"

        os.remove("configs/config_export.json")
