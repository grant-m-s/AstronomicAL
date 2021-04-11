import os, sys, inspect

sys.path.insert(1, os.path.join(sys.path[0], "../"))
import pandas as pd
import panel as pn

from astronomicAL.active_learning.active_learning import ActiveLearningTab
from astronomicAL.dashboard.active_learning import ActiveLearningDashboard
from astronomicAL.dashboard.dashboard import Dashboard
from astronomicAL.dashboard.menu import MenuDashboard
from astronomicAL.dashboard.plot import PlotDashboard
from astronomicAL.dashboard.selected_source import SelectedSourceDashboard
from astronomicAL.dashboard.settings_dashboard import SettingsDashboard
from astronomicAL.extensions.extension_plots import (
    CustomPlot,
    get_plot_dict,
    create_plot,
    bpt_plot,
    agn_wedge,
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
from bokeh.document import Document
from pyviz_comms import Comm

import astronomicAL.config as config
import pytest


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
            data.append([i, i % 3, i, i, i])

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

    def test_param_assignment_default_x_variable_contains_correct_columns_on_init(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        assert paramAssignment.param.default_x_variable.objects == ["default"]

    def test_param_assignment_default_x_variable_contains_correct_columns_on_update_no_df(
        self,
    ):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data()

        assert paramAssignment.param.default_x_variable.objects == ["default"]

    def test_param_assignment_default_x_variable_contains_correct_columns_on_update_with_df(
        self,
    ):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        assert paramAssignment.param.default_x_variable.objects == [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]

    def test_param_assignment_default_y_variable_contains_correct_columns_on_init(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        assert paramAssignment.param.default_y_variable.objects == ["default"]

    def test_param_assignment_default_y_variable_contains_correct_columns_on_update_no_df(
        self,
    ):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data()

        assert paramAssignment.param.default_y_variable.objects == ["default"]

    def test_param_assignment_default_y_variable_contains_correct_columns_on_update_with_df(
        self,
    ):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        assert paramAssignment.param.default_y_variable.objects == [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]

    def test_param_assignment_greater_than_20_labels_dont_proceed(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "E"
        paramAssignment.default_x_variable = "C"
        paramAssignment.default_y_variable = "D"

        paramAssignment._update_labels_cb()

        assert paramAssignment.label_strings_param == {}

    def test_param_assignment_less_than_20_labels_proceed(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "B"
        paramAssignment.default_x_variable = "C"
        paramAssignment.default_y_variable = "D"

        paramAssignment._update_labels_cb()

        assert paramAssignment.label_strings_param != {}

    def test_param_assignment_strings_to_labels_check_save_without_inputs(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "B"
        paramAssignment.default_x_variable = "C"
        paramAssignment.default_y_variable = "D"

        paramAssignment._update_labels_cb()

        paramAssignment._confirm_settings_cb(None)

        assert config.settings["strings_to_labels"] == {"0": 0, "1": 1, "2": 2}

    def test_param_assignment_strings_to_labels_check_save_with_some_inputs(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "B"
        paramAssignment.default_x_variable = "C"
        paramAssignment.default_y_variable = "D"

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
        paramAssignment.default_x_variable = "C"
        paramAssignment.default_y_variable = "D"

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
        paramAssignment.default_x_variable = "C"
        paramAssignment.default_y_variable = "D"

        paramAssignment._update_labels_cb()

        paramAssignment._confirm_settings_cb(None)

        assert config.settings["labels_to_strings"] == {"0": "0", "1": "1", "2": "2"}

    def test_param_assignment_labels_to_strings_check_save_with_some_inputs(self):

        df = self._create_test_df()

        paramAssignment = ParameterAssignment()

        paramAssignment.update_data(df)

        paramAssignment.id_column = "A"
        paramAssignment.label_column = "B"
        paramAssignment.default_x_variable = "C"
        paramAssignment.default_y_variable = "D"

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
        paramAssignment.default_x_variable = "C"
        paramAssignment.default_y_variable = "D"

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
        paramAssignment.default_x_variable = "C"
        paramAssignment.default_y_variable = "D"

        paramAssignment._update_labels_cb()

        paramAssignment._confirm_settings_cb(None)

        assert config.settings["id_col"] == "A"
        assert config.settings["label_col"] == "B"
        assert config.settings["default_vars"] == ("C", "D")

    def test_AL_settings_correct_labels_on_init(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None)

        assert alSettings.label_selector.options == []
        assert alSettings.label_selector.value == []

    def test_AL_settings_correct_features_on_init(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None)

        assert alSettings.feature_selector.options == []
        assert alSettings.feature_selector.value == []

    def test_AL_settings_correct_labels_on_update_no_df(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None)

        alSettings.update_data(None)

        assert alSettings.label_selector.options == []
        assert alSettings.label_selector.value == []

    def test_AL_settings_correct_features_on_update_no_df(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None)

        alSettings.update_data(None)

        assert alSettings.feature_selector.options == []
        assert alSettings.feature_selector.value == []

    def test_AL_settings_correct_labels_on_update_df(self):

        df = self._create_test_df()

        labels = df[config.settings["label_col"]].astype(str).unique()

        alSettings = ActiveLearningSettings(None)

        alSettings.update_data(df)

        assert alSettings.label_selector.options == list(labels)
        assert alSettings.label_selector.value == []

    def test_AL_settings_correct_features_on_update_df(self):

        df = self._create_test_df()

        features = df.columns

        alSettings = ActiveLearningSettings(None)

        alSettings.update_data(df)

        assert alSettings.feature_selector.options == list(features)
        assert alSettings.feature_selector.value == []

    def test_AL_settings_check_is_complete_on_init(self):

        alSettings = ActiveLearningSettings(None)

        assert not alSettings.is_complete()

    def test_AL_settings_check_is_complete_on_update_no_df(self):

        alSettings = ActiveLearningSettings(None)

        alSettings.update_data(None)

        assert not alSettings.is_complete()

    def test_AL_settings_check_is_complete_on_update_df(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None)

        alSettings.update_data(df)

        assert not alSettings.is_complete()

    def test_AL_settings_get_df_on_init(self):

        alSettings = ActiveLearningSettings(None)

        assert alSettings.get_df() is None

    def test_AL_settings_get_df_on_update_no_df(self):

        alSettings = ActiveLearningSettings(None)

        alSettings.update_data(None)

        assert alSettings.get_df() is None

    def test_AL_settings_get_df_on_update_df(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None)

        alSettings.update_data(df)

        assert alSettings.get_df() is df

    def test_AL_settings_correct_feature_generator_on_init(self):

        alSettings = ActiveLearningSettings(None)

        ans = feature_generation.get_oper_dict().keys()

        assert alSettings.feature_generator.options == list(ans)

    def test_AL_settings_correct_feature_generator_on_update_no_df(self):

        alSettings = ActiveLearningSettings(None)

        alSettings.update_data(None)

        ans = feature_generation.get_oper_dict().keys()

        assert alSettings.feature_generator.options == list(ans)

    def test_AL_settings_correct_feature_generator_on_update_df(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None)

        alSettings.update_data(df)

        ans = feature_generation.get_oper_dict().keys()
        assert alSettings.feature_generator.options == list(ans)

    def test_AL_settings_add_feature_cb_test(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None)

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

        alSettings = ActiveLearningSettings(None)

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

        alSettings = ActiveLearningSettings(None)

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

        alSettings = ActiveLearningSettings(None)

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

        alSettings = ActiveLearningSettings(None)

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

        alSettings = ActiveLearningSettings(None)

        alSettings.update_data(df)

        alSettings._remove_feature_selector_cb(None)

        assert alSettings.feature_generator_selected == []

        pd.testing.assert_frame_equal(
            alSettings._feature_generator_dataframe.value,
            pd.DataFrame([], columns=["oper", "n"]),
        )

    def test_AL_settings_add_remove_multiple_duplicate_feature_cb_test(self):

        df = self._create_test_df()

        alSettings = ActiveLearningSettings(None)

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
        alSettings = ActiveLearningSettings(button)

        alSettings.update_data(df)

        alSettings.label_selector.options = ["a", "b", "c", "d"]
        alSettings.label_selector.value = ["a", "b", "c"]

        alSettings._confirm_settings_cb(None)

        assert config.settings["labels_to_train"] == ["a", "b", "c"]

    def test_AL_settings_confirm_settings_features_for_training(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button)

        alSettings.update_data(df)

        alSettings.feature_selector.options = ["a", "b", "c", "d"]
        alSettings.feature_selector.value = ["a", "b", "c"]

        alSettings._confirm_settings_cb(None)

        assert config.settings["features_for_training"] == ["a", "b", "c"]

    def test_AL_settings_confirm_settings_exclude_labels_true(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button)

        alSettings.update_data(df)

        alSettings.exclude_labels_checkbox.value = True

        alSettings._confirm_settings_cb(None)

        assert config.settings["exclude_labels"]

    def test_AL_settings_confirm_settings_exclude_labels_false(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button)

        alSettings.update_data(df)

        alSettings.exclude_labels_checkbox.value = False

        alSettings._confirm_settings_cb(None)

        assert not config.settings["exclude_labels"]

    def test_AL_settings_confirm_settings_unclassified_labels(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button)

        alSettings.update_data(df)

        alSettings.label_selector.options = ["a", "b", "c", "d"]
        alSettings.label_selector.value = ["a", "b", "c"]

        alSettings._confirm_settings_cb(None)

        assert config.settings["unclassified_labels"] == ["d"]

    def test_AL_settings_confirm_settings_scale_data_true(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button)

        alSettings.update_data(df)

        alSettings.scale_features_checkbox.value = True

        alSettings._confirm_settings_cb(None)

        assert config.settings["scale_data"]

    def test_AL_settings_confirm_settings_scale_data_false(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button)

        alSettings.update_data(df)

        alSettings.scale_features_checkbox.value = False

        alSettings._confirm_settings_cb(None)

        assert not config.settings["scale_data"]

    def test_AL_settings_confirm_settings_feature_generation(self):

        df = self._create_test_df()

        button = pn.widgets.Button()
        alSettings = ActiveLearningSettings(button)

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
        ds = DataSelection(src)

        ds.load_layout_check = False

        ds.dataset = "data/table1.fits"

        ds._load_data_cb(None)

        os.remove("data/table1.fits")

        assert "config_load_level" not in list(config.settings.keys())

    def test_data_selection_check_config_load_level_layout_only(self, document, comm):

        src = ColumnDataSource()
        ds = DataSelection(src)

        ds.load_layout_check = True

        ds.load_config_select.value = (
            "Only load layout. Let me choose all my own settings"
        )

        button = ds.load_data_button_js

        widget = ds.load_data_button_js.get_root(document, comm=comm)

        button._server_click(document, widget.ref["id"], None)

        button._process_events({"clicks": 1})

        assert config.settings["config_load_level"] == 0

    def test_data_selection_check_config_load_level_layout_only(self):

        src = ColumnDataSource()
        ds = DataSelection(src)

        check_folder_exists("configs")

        ds.load_layout_check = True

        ds.load_config_select.value = (
            "Only load layout. Let me choose all my own settings"
        )

        ds._update_layout_file_cb(None)

        assert config.settings["config_load_level"] == 0

    def test_data_selection_check_config_load_level_settings_only(self):

        src = ColumnDataSource()
        ds = DataSelection(src)

        check_folder_exists("configs")

        ds.load_layout_check = True

        ds.load_config_select.value = (
            "Load all settings but let me train the model from scratch."
        )

        ds._update_layout_file_cb(None)

        assert config.settings["config_load_level"] == 1

    def test_data_selection_check_config_load_level_settings_only(self):

        src = ColumnDataSource()
        ds = DataSelection(src)

        check_folder_exists("configs")

        ds.load_layout_check = True

        ds.load_config_select.value = (
            "Load all settings and train model with provided labels."
        )

        ds._update_layout_file_cb(None)

        assert config.settings["config_load_level"] == 2

    def test_data_selection_get_dataframe_from_fits_non_optimised_parameter(self):

        from astropy.table import Table

        t = Table([[1, 2], [4, 5], [7, 8]], names=("a", "b", "c"))

        check_folder_exists("data")
        t.write("data/table1.fits", format="fits")

        src = ColumnDataSource()
        ds = DataSelection(src)

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
        ds = DataSelection(src)

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
        ds = DataSelection(src)

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
        ds = DataSelection(src)

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
        ds = DataSelection(src)

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
        ds = DataSelection(src)

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
            data.append([i, i % 3, i, i, i])

        df = pd.DataFrame(data, columns=list("ABCDE"))

        return df

    def _create_test_df_with_image_data(self):

        data = []

        for i in range(100):
            data.append([i, i % 3, i, i, i, "178.52904,2.1655949"])

        df = pd.DataFrame(data, columns=list("ABCDE") + ["ra_dec"])

        return df

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

        assert selected_source.selected_history == [5]
        assert selected_source._url_optical_image == ""
        assert selected_source._search_status == ""
        assert selected_source._image_zoom == 0.2

        data = self._create_test_df()
        data = data.iloc[15]
        src = ColumnDataSource({str(c): [v] for c, v in data.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == [15]
        assert selected_source._url_optical_image == ""
        assert selected_source._search_status == ""
        assert selected_source._image_zoom == 0.2

        data = self._create_test_df()
        data = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == [72]
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

        assert selected_source.selected_history == [72]

    def test_selected_source_check_history_from_selected_to_selected_unique(self):
        data = self._create_test_df()
        data_selected = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == [72]

        data_selected = data.iloc[30]
        src.data = {str(c): [v] for c, v in data_selected.items()}

        assert selected_source.selected_history == [30, 72]

    def test_selected_source_check_history_from_selected_to_selected_same_id_head(self):
        data = self._create_test_df()
        config.main_df = data
        data_selected = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == [72]

        data_selected = data.iloc[72]
        src.data = {str(c): [v] for c, v in data_selected.items()}

        assert selected_source.selected_history == [72]

    def test_selected_source_check_history_from_selected_to_selected_no_id(self):
        data = self._create_test_df()
        config.main_df = data
        data_selected = data.iloc[72]
        src = ColumnDataSource({str(c): [v] for c, v in data_selected.items()})
        selected_source = SelectedSourceDashboard(src=src, close_button=None)

        assert selected_source.selected_history == [72]

        data_selected = data.iloc[53].copy()
        data_selected["A"] = ""
        src.data = {str(c): [v] for c, v in data_selected.items()}

        assert selected_source.selected_history == [72]

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

        assert selected_source.selected_history == [72, 30, 72]

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

        event = Event(5, "")
        selected_source._change_selected(event)

        print(selected_source.src.data)

        assert selected_source._search_status == "Searching..."
        assert selected_source.src.data == src.data
        assert selected_source.src.data["A"] == [5]

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

        event = Event(5, "")
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

        ra_dec = selected_source.src.data["ra_dec"][0]
        ra = ra_dec[: ra_dec.index(",")]
        dec = ra_dec[ra_dec.index(",") + 1 :]

        url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra="
        _url_optical_image = (
            f"{url}{ra}&dec={dec}&opt=G&scale={selected_source._image_zoom}"
        )

        assert selected_source._url_optical_image == _url_optical_image
