import config
import panel as pn
import param


class ActiveLearningSettings(param.Parameterized):

    def __init__(self, src, close_button, **params):
        super(ActiveLearningSettings, self).__init__(**params)

        self.df = None

        self.close_button = close_button

        self.column = pn.Column("Loading")

        self.label_selector = pn.widgets.CrossSelector(
            name="Labels", value=[], options=[], max_height=100
        )

        self.label_selector._search[True].max_height = 20
        self.label_selector._search[False].max_height = 20

        self.feature_selector = pn.widgets.CrossSelector(
            name="Features", value=[], options=[], max_height=100
        )

        self.feature_selector._search[True].max_height = 20
        self.feature_selector._search[False].max_height = 20

        self.confirm_settings_button = pn.widgets.Button(
            name="Confirm Settings")
        self.confirm_settings_button.on_click(self.confirm_settings_cb)

        self.exclude_labels_checkbox = pn.widgets.Checkbox(
            name="Should remaining labels be removed from Active Learning datasets?", value=True
        )

        self.exclude_labels_tooltip = pn.pane.HTML(
            "<span data-toggle='tooltip' title='If enabled, this will remove all instances of labels without a classifier from the Active Learning train, validation and test sets (visualisations outside the AL panel are unaffected). This is useful for labels representing an unknown classification which would not be compatible with scoring functions.' style='border-radius: 15px;padding: 5px; background: #5e5e5e; ' >❔</span> ",
            max_width=5,
        )

        self.scale_features_checkbox = pn.widgets.Checkbox(
            name="Should features be scaled during training?"
        )

        self.scale_features_tooltip = pn.pane.HTML(
            "<span data-toggle='tooltip' title='If enabled, this can improve the performance of your model, however will require you to scale all new data with the produced scaler. This scaler will be saved in your model directory.' style='border-radius: 15px;padding: 5px; background: #5e5e5e; ' >❔</span> ",
            max_width=5,
        )

        self.completed = False

    def update_data(self, dataframe=None):

        if dataframe is not None:
            self.df = dataframe

        if (self.df is not None):

            labels = config.settings["labels"]
            options = []
            for label in labels:
                options.append(
                    config.settings["labels_to_strings"][f"{label}"])
            self.label_selector.options = options

            self.feature_selector.options = list(self.df.columns)

    def confirm_settings_cb(self, event):
        print("Saving settings...")

        config.settings["labels_to_train"] = self.label_selector.value
        config.settings["features_for_training"] = self.feature_selector.value
        config.settings["exclude_labels"] = self.exclude_labels_checkbox.value

        unclassified_labels = []
        for label in self.label_selector.options:
            if label not in self.label_selector.value:
                unclassified_labels.append(label)

        config.settings["unclassified_labels"] = unclassified_labels
        config.settings["scale_data"] = self.scale_features_checkbox.value

        self.completed = True

        self.close_button.disabled = False
        self.close_button.button_type = "success"

        self.panel()

    def get_df(self):
        return self.df

    def is_complete(self):
        return self.completed

    def panel(self):
        print("Rendering AL Settings Panel")

        if self.completed:
            self.column[0] = pn.pane.Str("Settings Saved.")

        else:

            self.column[0] = pn.Column(
                pn.Row(
                    self.label_selector.name, self.feature_selector.name, max_height=30
                ),
                pn.Row(
                    self.label_selector,
                    self.feature_selector,
                    height=200,
                    sizing_mode="stretch_width",
                ),
                pn.Row(self.exclude_labels_checkbox,
                       self.exclude_labels_tooltip),
                pn.Row(self.scale_features_checkbox,
                       self.scale_features_tooltip),
                pn.Row(self.confirm_settings_button, max_height=30),
            )

        return self.column
