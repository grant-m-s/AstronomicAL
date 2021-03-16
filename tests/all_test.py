import os, sys, inspect

sys.path.insert(1, os.path.join(sys.path[0], "../"))
import pandas as pd
from astronomicAL.dashboard.dashboard import Dashboard
from astronomicAL.settings.param_assignment import ParameterAssignment
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
