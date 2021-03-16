from astronomicAL.extensions import models, query_strategies, feature_generation

import astronomicAL.config as config
import pandas as pd
import panel as pn
import param


class ActiveLearningSettings(param.Parameterized):
    """The Active Learning Settings Stage used in the settings pipeline.

    Parameters
    ----------
    src : ColumnDataSource
        The shared data source which holds the current selected source.
    close_button : Panel Button
        Close button widget from the parent settings dashboard to allow the
        button to be updated when all settings have been completed.

    Attributes
    ----------
    df : DataFrame
        The shared dataframe which holds all the data.
    label_selector : Panel CrossSelector
        CrossSelector widget for choosing which of the dataset labels should
        have a classifier created.
    feature_selector : Panel CrossSelector
        CrossSelector widget for choosing which of the data columns should be
        used during the machine learning steps.
    exclude_labels_checkbox : Panel Checkbox
        Checkbox widget for choosing whether the labels which do not have a
        classifier should be removed completely from the data used for the
        machine learning steps.
    scale_features_checkbox : Panel Checkbox
        Checkbox widget for choosing whether the chosen features should be
        normalised in the mahchine learning preprocessing.
    completed : bool
        Flag indicating all active learning settings have been chosen and
        assigned.

    """

    def __init__(self, src, close_button, **params):
        super(ActiveLearningSettings, self).__init__(**params)

        self.df = None

        self.feature_generator_selected = []

        self.close_button = close_button

        self.column = pn.Column("Loading")

        self.completed = False

        self._initialise_widgets()

        self._adjust_widget_layouts()

    def _adjust_widget_layouts(self):

        self.label_selector._search[True].max_height = 20
        self.label_selector._search[False].max_height = 20
        self.feature_selector._search[True].max_height = 20
        self.feature_selector._search[False].max_height = 20

    def _initialise_widgets(self):

        self.label_selector = pn.widgets.CrossSelector(
            name="**Which Labels would you like to create a classifier for?**",
            value=[],
            options=[],
            max_height=100,
        )

        self.feature_selector = pn.widgets.CrossSelector(
            name="**Which Features should be used for training?**",
            value=[],
            options=[],
            max_height=100,
        )

        self.feature_generator = pn.widgets.Select(
            name="Create Feature Combinations?",
            options=list(feature_generation.get_oper_dict().keys()),
        )

        self.feature_generator_number = pn.widgets.IntInput(
            name="How many features to combine?", value=2, step=1, start=2, end=5
        )

        self._add_feature_generator_button = pn.widgets.Button(name=">>")
        self._add_feature_generator_button.on_click(self._add_feature_selector_cb)

        self._remove_feature_generator_button = pn.widgets.Button(name="Remove")
        self._remove_feature_generator_button.on_click(self._remove_feature_selector_cb)

        self._feature_generator_dataframe = pn.widgets.DataFrame(
            pd.DataFrame(self.feature_generator_selected, columns=["oper", "n"]),
            name="",
            show_index=False,
        )

        self.confirm_settings_button = pn.widgets.Button(
            name="Confirm Settings", button_type="primary"
        )
        self.confirm_settings_button.on_click(self._confirm_settings_cb)

        self.exclude_labels_checkbox = pn.widgets.Checkbox(
            name="Should remaining labels be removed from Active Learning datasets?",
            value=True,
        )

        self._exclude_labels_tooltip = pn.pane.HTML(
            "<span data-toggle='tooltip' title='If enabled, this will remove all instances of labels without a classifier from the Active Learning train, validation and test sets (visualisations outside the AL panel are unaffected). This is useful for labels representing an unknown classification which would not be compatible with scoring functions.' style='border-radius: 15px;padding: 5px; background: #5e5e5e; ' >❔</span> ",
            max_width=5,
        )

        self.scale_features_checkbox = pn.widgets.Checkbox(
            name="Should features be scaled during training?"
        )

        self._scale_features_tooltip = pn.pane.HTML(
            "<span data-toggle='tooltip' title='If enabled, this can improve the performance of your model, however will require you to scale all new data with the produced scaler. This scaler will be saved in your model directory.' style='border-radius: 15px;padding: 5px; background: #5e5e5e; ' >❔</span> ",
            max_width=5,
        )

    def update_data(self, dataframe=None):
        """Update the classes local copy of the dataset.

        Parameters
        ----------
        dataframe : DataFrame, default = None
            An up to date version of the dataset.

        Returns
        -------
        None

        """
        if dataframe is not None:
            self.df = dataframe

        if self.df is not None:

            labels = config.settings["labels"]
            options = []
            for label in labels:
                options.append(config.settings["labels_to_strings"][f"{label}"])
            self.label_selector.options = options

            self.feature_selector.options = list(self.df.columns)

    def _add_feature_selector_cb(self, event):

        new_feature_generator = [
            self.feature_generator.value,
            self.feature_generator_number.value,
        ]

        if new_feature_generator not in self.feature_generator_selected:
            self.feature_generator_selected.append(
                [self.feature_generator.value, self.feature_generator_number.value]
            )
            self._feature_generator_dataframe.value = pd.DataFrame(
                self.feature_generator_selected, columns=["oper", "n"]
            )

    def _remove_feature_selector_cb(self, event):
        self.feature_generator_selected = self.feature_generator_selected[:-1]
        self._feature_generator_dataframe.value = pd.DataFrame(
            self.feature_generator_selected, columns=["oper", "n"]
        )

    def _confirm_settings_cb(self, event):
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
        config.settings["feature_generation"] = self.feature_generator_selected
        config.settings["confirmed"] = True

        self.completed = True

        self.close_button.disabled = False
        self.close_button.button_type = "success"

        self.panel()

    def get_df(self):
        """Return the dataframe the active learning settings has generated.

        Returns
        -------
        df : DataFrame
            Original DataFrame with any generated features added.

        """
        return self.df

    def is_complete(self):
        """Return whether the settings page has been completed.

        Returns
        -------
        completed : bool
            Flag for whether the settings have been completed and assigned.

        """
        return self.completed

    def panel(self):
        """Render the current settings view.

        Returns
        -------
        column : Panel Column
            The panel is housed in a column which can then be rendered by the
            settings Dashboard.

        """
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
                pn.Row(self.exclude_labels_checkbox, self._exclude_labels_tooltip),
                pn.Row(self.scale_features_checkbox, self._scale_features_tooltip),
                pn.Row(
                    self.feature_generator,
                    self.feature_generator_number,
                    pn.Column(
                        self._add_feature_generator_button,
                        self._remove_feature_generator_button,
                    ),
                    self._feature_generator_dataframe,
                    sizing_mode="stretch_width",
                ),
                pn.layout.Divider(max_width=30),
                pn.Row(self.confirm_settings_button, max_height=30),
                pn.Row(pn.Spacer(height=50)),
            )

        return self.column
