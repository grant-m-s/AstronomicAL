import os, sys, inspect

sys.path.insert(1, os.path.join(sys.path[0], "../"))
import pandas as pd
from astronomicAL.dashboard.dashboard import Dashboard
from astronomicAL.settings.param_assignment import ParameterAssignment


class TestClass:
    def func(self, x):
        return x + 1

    def test_answer(self):
        assert self.func(3) == 4


class TestSettings:
    def _create_test_df(self):

        data = []

        for i in range(100):
            data.append([i, i, i, i, i])

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
