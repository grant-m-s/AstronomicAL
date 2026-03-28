from astronomicAL.settings.active_learning import ActiveLearningSettings
from astronomicAL.settings.data_selection import DataSelection
from astronomicAL.settings.param_assignment import ParameterAssignment_ML, ParameterAssignment_Exploring
from functools import partial

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

    def __init__(self, main, src, context=None):
        self.row = pn.Row(pn.pane.Str("loading"), sizing_mode = "stretch_both")

        self.src = src

        self.context = context
        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

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

        self.select_AL_mode_button = pn.widgets.Button(name="Active Learning Mode", sizing_mode = "stretch_width", max_width =150,
                                                       align="center")

        self.select_AL_mode_button.on_click(
            partial(self._create_pipeline_cb, mode="AL", main=main)
        )

        self.select_labelling_mode_button = pn.widgets.Button(name="Labelling Mode", sizing_mode = "stretch_width", max_width =150,
                                                              align="center")

        self.select_labelling_mode_button.on_click(
            partial(self._create_pipeline_cb, mode="Labelling", main=main)
        )

        self.select_exploring_mode_button = pn.widgets.Button(name="Exploring Mode", sizing_mode = "stretch_width", max_width =150,
                                                              align="center")

        self.select_exploring_mode_button.on_click(
            partial(self._create_pipeline_cb, mode="Exploring", main=main)
        )

    def get_toolbar(self):
        return pn.Spacer(height=1)

    def _create_pipeline_cb(self, event, mode, main):
        main.mode = mode
        self.config.mode = mode
        self.create_pipeline(mode=mode)

    def create_mode_selection_menu(self):
        return pn.FlexBox(
            self._make_mode_column("images/classification.png", self.select_labelling_mode_button),
            self._make_mode_column("images/cluster.png", self.select_AL_mode_button),
            self._make_mode_column("images/exploration.png", self.select_exploring_mode_button),
            flex_direction="row",
            flex_wrap="wrap",
            justify_content="center",
            align_items="flex-start",
            gap="24px",
            sizing_mode="stretch_width",
            margin=(20, 0, 0, 0),
        )
    
    # def _make_mode_row(self, image_path, button):
    #     return pn.Row(pn.pane.PNG(
    #                                image_path,
    #                                width = 200,
    #                                height = 200,          
    #                                margin=(0, 0, 5, 0),
    #                                ),
    #                                button,)      
           
    def _make_mode_column(self, image_path, button):
        button.width = 150
        button.height = 36
        button.sizing_mode = "fixed"
        button.margin = (0, 0, 0, 0)

        image = pn.pane.PNG(
            image_path,
            width=160,
            height=160,
            sizing_mode="fixed",
            margin=(0, 0, 12, 0),
        )

        image_box = pn.Column(
            image,
            width=200,
            height=200,
            sizing_mode="fixed",
            styles={
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
            },
            margin=(0, 0, 0, 0),
        )

        button_row = pn.Row(
            pn.layout.HSpacer(),
            button,
            pn.layout.HSpacer(),
            width=220,
            height=40,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        )

        return pn.Column(
            image_box,
            button_row,
            width=200,
            height=250,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        )


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
                DataSelection(self.src, mode=mode, context=self.context),
                ready_parameter="ready",
            ),
            self.pipeline.add_stage(
                "Assign Parameters", ParameterAssignment_ML(context=self.context), ready_parameter="ready",
            ),
            self.pipeline.add_stage(
                "Features Settings",
                ActiveLearningSettings(self._close_settings_button, mode=mode, context=self.context),
            )
        elif mode == "Labelling":
            self.pipeline.add_stage(
                "Select Your Data",
                DataSelection(self.src, mode=mode, context=self.context),
                ready_parameter="ready",
            ),
            self.pipeline.add_stage(
                "Assign Parameters", ParameterAssignment_ML(context=self.context), ready_parameter="ready"
            ),
            self.pipeline.add_stage(
                "Features Settings",
                ActiveLearningSettings(self._close_settings_button, mode=mode, context=self.context),
            )

        elif mode == "Exploring":
            self.pipeline.add_stage(
                "Select Your Data",
                DataSelection(self.src, mode=mode, context=self.context, close_settings_button=self._close_settings_button),
                # ready_parameter="ready",
            ),
            # self.pipeline.add_stage(
            #     "Assign Parameters", ParameterAssignment_Exploring(self._close_settings_button, context=self.context), 
            # ),

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
            "Features Settings"
        ].get_default_variables()
        updated_settings["label_colours"] = self.pipeline[
            "Assign Parameters"
        ].get_label_colours()

        return updated_settings

    def _close_settings_cb(self, event, main):
        
        print("closing settings")

        stage_name =  list(self.pipeline._stages.keys())[self._pipeline_stage]
        self.df = self.pipeline[stage_name].get_df()
        self.config.main_df = self.df

        src = {}
        for col in self.df.columns:
            src[f"{col}"] = []

        self.src.data = src

        self._close_settings_button.disabled = True
        self._close_settings_button.name = "Setting up panels..."

        if self.config.mode == "AL":
            main.set_contents(updated="Active Learning")
        elif self.config.mode == "Labelling":
            main.set_contents(updated="Labelling")
        elif self.config.mode == "Exploring":
            main.set_contents(updated="Exploring")

    def _stage_previous_cb(self, event):

        self._pipeline_stage -= 1
        self.panel()

    def _stage_next_cb(self, event):

        if self.df is None:
            self.df = self.config.main_df

        pipeline_list = list(self.pipeline._stages)

        next_stage = pipeline_list[self._pipeline_stage + 1]

        self.pipeline[next_stage].update_data(dataframe=self.df)

        self._pipeline_stage += 1
        self.panel()

    def panel(self):
        """Render the current view."""

        if not self.pipeline_initialised:
            self.row[0] = self.create_mode_selection_menu()
            return self.row

        if "Features Settings" in self.pipeline._stages:
            if self.pipeline["Features Settings"].is_complete():
                self._close_settings_button.disabled = False

        current_stage_title = list(self.pipeline._stages)[self._pipeline_stage]

        self._close_settings_button.width = 150
        self._close_settings_button.height = 44
        self._close_settings_button.min_height = 44

        close_row = pn.Row(
            pn.layout.HSpacer(),
            self._close_settings_button,
            sizing_mode="stretch_width",
            margin=(6, 12, 0, 12),
        )

        settings_text = pn.pane.Markdown(
            "**Settings panel:** Choose the appropriate settings for your data",
            sizing_mode="stretch_width",
            styles={
                "font-size": "14px",
                "color": "#5E6C84",
            },
            margin=(2, 12, 4, 12),
        )

        stage_title = pn.pane.Markdown(
            f"## {current_stage_title}",
            sizing_mode="stretch_width",
            styles={
                "line-height": "1.2",
                "color": "#172B4D",
            },
            margin=(0, 12, 8, 12),
        )

        stage_body = pn.Column(
            self.pipeline.stage,
            sizing_mode="stretch_width",
            margin=(0, 12, 16, 12),
        )

        self.row[0] = pn.Column(
            close_row,
            settings_text,
            stage_title,
            stage_body,
            sizing_mode="stretch_both",
            styles={
                "background": "#F7F8FA",
            },
        )

        return self.row