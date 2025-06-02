from astronomicAL.settings.active_learning import ActiveLearningSettings
from astronomicAL.settings.data_selection import DataSelection
from astronomicAL.settings.param_assignment import ParameterAssignment
from functools import partial

import astronomicAL.config as config
import panel as pn


class SettingsDashboard:
    """A Dashboard used to display configuration settings for the user.

    Parameters
    ----------
    main : Dashboard
        The parent Dashboard view required for updating which dashboard is
        rendered.
    src : ColumnDataSource
        The shared data source which holds the current selected source.

    Attributes
    ----------
    row : Panel Row
        The panel is housed in a row which can then be rendered by the
        Panel layout.
    pipeline : Panel Pipeline
        A pipeline of stages for the user to assign key parameters.
    """

    def __init__(self, main, src):
        self.row = pn.Row(pn.pane.Str("loading"))

        self.src = src

        self.df = None

        self._pipeline_stage = 0

        self._initialise_widgets(main)

        self.create_mode_selection_menu()

        self.pipeline_initialised = False

    def _initialise_widgets(self, main):
        self._close_settings_button = pn.widgets.Button(
            name="Close Settings", max_width=150, max_height=50, disabled=True
        )
        self._close_settings_button.on_click(
            partial(self._close_settings_cb, main=main)
        )

        self.select_AL_mode_button = pn.widgets.Button(name="Active Learning Mode")

        self.select_AL_mode_button.on_click(
            partial(self._create_pipeline_cb, mode="AL", main=main)
        )

        self.select_labelling_mode_button = pn.widgets.Button(name="Labelling Mode")

        self.select_labelling_mode_button.on_click(
            partial(self._create_pipeline_cb, mode="Labelling", main=main)
        )

    def _create_pipeline_cb(self, event, mode, main):
        main.mode = mode
        config.mode = mode
        self.create_pipeline(mode=mode)

    def create_mode_selection_menu(self):
        layout = pn.Card(
            pn.Row(
                pn.Column(
                    pn.pane.PNG(
                        "images/classification.png",
                        width=250,
                        height=250,
                        margin=(0, 0, 0, 50),
                    ),
                    pn.Row(self.select_labelling_mode_button, max_height=30),
                ),
                pn.Column(
                    pn.pane.PNG(
                        "images/cluster.png",
                        width=250,
                        height=250,
                        margin=(0, 0, 0, 50),
                    ),
                    pn.Row(self.select_AL_mode_button, max_height=30),
                ),
            )
        )

        return layout

    def create_pipeline(self, mode):
        """Create the pipeline of setting stages.

        Parameters
        ----------

        Returns
        -------
        None

        """
        self.pipeline = pn.pipeline.Pipeline()

        valid_mode = True

        if mode == "AL":
            self.pipeline.add_stage(
                "Select Your Data",
                DataSelection(self.src, mode=mode),
                ready_parameter="ready",
            ),
            self.pipeline.add_stage(
                "Assign Parameters", ParameterAssignment(), ready_parameter="ready"
            ),
            self.pipeline.add_stage(
                "Active Learning Settings",
                ActiveLearningSettings(self._close_settings_button, mode=mode),
            )
        elif mode == "Labelling":
            self.pipeline.add_stage(
                "Select Your Data",
                DataSelection(self.src, mode=mode),
                ready_parameter="ready",
            ),
            self.pipeline.add_stage(
                "Assign Parameters", ParameterAssignment(), ready_parameter="ready"
            ),
            self.pipeline.add_stage(
                "Active Learning Settings",
                ActiveLearningSettings(self._close_settings_button, mode=mode),
            )
        else:
            valid_mode = False

        if valid_mode:
            self._adjust_pipeline_widgets()
            self.pipeline_initialised = True

        self.panel()

    def _adjust_pipeline_widgets(self):

        self.pipeline.layout[0][0][0].sizing_mode = "fixed"

        self.pipeline.layout[0][0][0].max_height = 75

        self.pipeline.layout[0][2][0].sizing_mode = "fixed"
        self.pipeline.layout[0][2][1].sizing_mode = "fixed"
        self.pipeline.layout[0][2][0].height = 30
        self.pipeline.layout[0][2][1].height = 30
        self.pipeline.layout[0][2][0].max_width = 150
        self.pipeline.layout[0][2][1].max_width = 150

        self.pipeline.layout[0][2][0].on_click(self._stage_previous_cb)

        self.pipeline.layout[0][2][1].button_type = "success"
        self.pipeline.layout[0][2][1].on_click(self._stage_next_cb)

    def get_settings(self):
        """Get the settings assigned during the pipeline stages.

        Returns
        -------
        updated_settings : dict
            A dictionary of assigned parameters.

        """
        updated_settings = {}
        updated_settings["id_col"] = self.pipeline["Assign Parameters"].get_id_column()
        updated_settings["label_col"] = self.pipeline[
            "Assign Parameters"
        ].get_label_column()
        updated_settings["default_vars"] = self.pipeline[
            "Active Learning Settings"
        ].get_default_variables()
        updated_settings["label_colours"] = self.pipeline[
            "Assign Parameters"
        ].get_label_colours()

        return updated_settings

    def _close_settings_cb(self, event, main):
        print("closing settings")

        self.df = self.pipeline["Active Learning Settings"].get_df()

        config.main_df = self.df

        src = {}
        for col in self.df.columns:
            src[f"{col}"] = []

        self.src.data = src

        self._close_settings_button.disabled = True
        self._close_settings_button.name = "Setting up training panels..."

        if config.mode == "AL":
            main.set_contents(updated="Active Learning")
        elif config.mode == "Labelling":
            main.set_contents(updated="Labelling")

    def _stage_previous_cb(self, event):

        self._pipeline_stage -= 1
        self.panel()

    def _stage_next_cb(self, event):

        if self.df is None:
            self.df = config.main_df

        pipeline_list = list(self.pipeline._stages)

        next_stage = pipeline_list[self._pipeline_stage + 1]

        self.pipeline[next_stage].update_data(dataframe=self.df)

        self._pipeline_stage += 1
        self.panel()

    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which can then be rendered by the
            parent Dashboard.

        """

        if not self.pipeline_initialised:
            self.row[0] = self.create_mode_selection_menu()
            return self.row

        else:

            if self.pipeline["Active Learning Settings"].is_complete():
                self._close_settings_button.disabled = False

            self.row[0] = pn.Card(
                pn.Column(
                    pn.Row(self.pipeline.stage),
                    pn.Row(
                        pn.layout.HSpacer(),
                        pn.layout.HSpacer(),
                        self.pipeline.buttons[1],
                        max_height=50,
                        # max_width=500,
                    ),
                ),
                header=pn.Row(
                    pn.widgets.StaticText(
                        name="Settings Panel",
                        value="Please choose the appropriate settings for your data",
                    ),
                    pn.pane.Markdown(
                        "### " + list(self.pipeline._stages)[self._pipeline_stage],
                    ),
                    self._close_settings_button,
                ),
                collapsible=False,
            )

            return self.row
