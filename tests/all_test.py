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

# from astronomicAL.dashboard.active_learning import ActiveLearningDashboard
import astronomicAL.config as config


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
