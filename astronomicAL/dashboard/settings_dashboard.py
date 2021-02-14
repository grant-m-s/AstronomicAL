from functools import partial
from settings.active_learning import ActiveLearningSettings
from settings.data_selection import DataSelection
from settings.param_assignment import ParameterAssignment

import config
import panel as pn
import param


class SettingsDashboard(param.Parameterized):

    def __init__(self, main, src, df, **params):
        super(SettingsDashboard, self).__init__(**params)
        self.row = pn.Row(pn.pane.Str("loading"))

        self.src = src

        self.df = None

        self.pipeline_stage = 0

        self.close_settings_button = pn.widgets.Button(
            name="Close Settings",
            max_width=100,
            disabled=True
        )
        self.close_settings_button.on_click(
            partial(self.close_settings_cb, main=main))

        self.pipeline = pn.pipeline.Pipeline()
        self.pipeline.add_stage(
            "Select Your Data", DataSelection(src), ready_parameter="ready"
        ),
        self.pipeline.add_stage(
            "Assign Parameters", ParameterAssignment(df),
            ready_parameter="ready"
        ),
        self.pipeline.add_stage(
            "Active Learning Settings", ActiveLearningSettings(
                src, self.close_settings_button)
        )

        self.pipeline.layout[0][0][0].sizing_mode = "fixed"

        self.pipeline.layout[0][0][0].max_height = 75

        self.pipeline.layout[0][2][0].sizing_mode = "fixed"
        self.pipeline.layout[0][2][1].sizing_mode = "fixed"
        self.pipeline.layout[0][2][0].height = 30
        self.pipeline.layout[0][2][1].height = 30
        self.pipeline.layout[0][2][0].width = 100
        self.pipeline.layout[0][2][1].width = 100

        self.pipeline.layout[0][2][0].on_click(self.stage_previous_cb)

        self.pipeline.layout[0][2][1].button_type = 'success'
        self.pipeline.layout[0][2][1].on_click(self.stage_next_cb)

        # print(self.pipeline)
        # print(self.pipeline["Assign Parameters"].get_id_column())

    def get_settings(self):

        updated_settings = {}
        updated_settings["id_col"] = self.pipeline["Assign Parameters"].get_id_column(
        )
        updated_settings["label_col"] = self.pipeline[
            "Assign Parameters"
        ].get_label_column()
        updated_settings["default_vars"] = self.pipeline[
            "Assign Parameters"
        ].get_default_variables()
        updated_settings["label_colours"] = self.pipeline[
            "Assign Parameters"
        ].get_label_colours()

        return updated_settings

    def close_settings_cb(self, event, main):
        print("closing settings")

        self.df = self.pipeline["Active Learning Settings"].get_df()

        config.main_df = self.df

        src = {}
        for col in self.df.columns:
            src[f"{col}"] = []

        self.src.data = src
        # print("\n\n\n\n")
        # print(len(list(self.src.data.keys())))

        self.close_settings_button.disabled = True
        self.close_settings_button.name = "Setting up training panels..."

        main.set_contents(updated="Active Learning")

    def stage_previous_cb(self, event):

        self.pipeline_stage -= 1

    def stage_next_cb(self, event):

        if self.df is None:
            print("updating Settings df")
            self.df = self.pipeline["Select Your Data"].get_df()

        pipeline_list = list(self.pipeline._stages)
        print("STAGE:")
        current_stage = pipeline_list[self.pipeline_stage]

        next_stage = pipeline_list[self.pipeline_stage + 1]
        self.pipeline[next_stage].update_data(dataframe=self.df)

        self.pipeline_stage += 1

    def panel(self):
        if self.pipeline["Active Learning Settings"].is_complete():
            self.close_settings_button.disabled = False

        self.row[0] = pn.Card(
            pn.Column(
                pn.Row(
                    self.pipeline.title,
                    self.pipeline.buttons,
                ),
                self.pipeline.stage,
            ),
            header=pn.Row(pn.widgets.StaticText(
                name="Settings Panel",
                value="Please choose the appropriate settings for your data",
                ),
                self.close_settings_button,
            ),
            collapsible=False,
        )

        return self.row
