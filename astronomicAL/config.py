import pandas as pd
from bokeh.models import ColumnDataSource


initial_setup = True
layout_file = "astronomicAL/layout.json"
dashboards = {}

source = ColumnDataSource()

main_df = pd.DataFrame()

ml_data = {}

settings = {"confirmed": False}
