import os, sys, inspect

sys.path.insert(1, os.path.join(sys.path[0], "../astronomicAL"))

from dashboard.dashboard import Dashboard

# import config

# from bokeh.models import ColumnDataSource


class TestClass:
    def func(self, x):
        return x + 1

    def test_answer(self):
        assert self.func(3) == 4
