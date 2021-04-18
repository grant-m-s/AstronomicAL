from astronomicAL.extensions import models, query_strategies, feature_generation

import astronomicAL.config as config
import pandas as pd
import panel as pn
import json
import os
import param


class ActiveLearningSettings(param.Parameterized):
    """The Active Learning Settings Stage used in the settings pipeline.

    Parameters
    ----------
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

    def __init__(self, close_button, mode):

        self.df = None

        self.feature_generator_selected = []

        self.close_button = close_button

        self.column = pn.Column("Loading")

        self.completed = False

        self._initialise_widgets()

        self._adjust_widget_layouts()

        self._verify_valid_selection_cb(None)

    def _adjust_widget_layouts(self):

        self.label_selector.size = 5
        self.feature_selector.size = 5

        self.label_selector._search[True].max_height = 20
        self.label_selector._search[False].max_height = 20
        self.feature_selector._search[True].max_height = 20
        self.feature_selector._search[False].max_height = 20
        self.label_selector._search[True].max_width = 300
        self.label_selector._search[False].max_width = 300
        self.feature_selector._search[True].max_width = 300
        self.feature_selector._search[False].max_width = 300

        self.label_selector._lists[True].max_width = 300
        self.label_selector._lists[False].max_width = 300
        self.feature_selector._lists[True].width = 500
        self.feature_selector._lists[False].width = 500

        self.label_selector._buttons[True].max_width = 50
        self.label_selector._buttons[False].max_width = 50
        self.feature_selector._buttons[True].max_width = 50
        self.feature_selector._buttons[False].max_width = 50

        self.label_selector._buttons[True].max_height = 30
        self.label_selector._buttons[False].max_height = 30
        self.feature_selector._buttons[True].max_height = 30
        self.feature_selector._buttons[False].max_height = 30

        self.label_selector._buttons[True].margin = (50, 20, 0, 20)
        self.label_selector._buttons[False].margin = (10, 20, 0, 20)
        self.feature_selector._buttons[True].margin = (50, 20, 0, 20)
        self.feature_selector._buttons[False].margin = (10, 20, 0, 20)

        self.label_selector._composite[:] = [
            self.label_selector._unselected,
            pn.Column(
                self.label_selector._buttons[True],
                self.label_selector._buttons[False],
            ),
            self.label_selector._selected,
        ]

        self.feature_selector._composite[:] = [
            self.feature_selector._unselected,
            pn.Column(
                self.feature_selector._buttons[True],
                self.feature_selector._buttons[False],
            ),
            self.feature_selector._selected,
        ]

    def _initialise_widgets(self):

        self.label_selector = pn.widgets.CrossSelector(
            name="**Which Labels would you like to create a classifier for?**",
            value=[],
            options=[],
        )

        self.label_selector._buttons[True].on_click(self._verify_valid_selection_cb)
        self.label_selector._buttons[False].on_click(self._verify_valid_selection_cb)

        self.feature_selector = pn.widgets.CrossSelector(
            name="**Which Features should be used for training?**",
            value=[],
            options=[],
            width=500,
            max_width=500,
            sizing_mode="fixed",
        )

        self.feature_selector._buttons[True].on_click(self._verify_valid_selection_cb)
        self.feature_selector._buttons[False].on_click(self._verify_valid_selection_cb)

        self.feature_generator = pn.widgets.Select(
            name="Create Feature Combinations?",
            options=list(feature_generation.get_oper_dict().keys()),
        )

        self.feature_generator_number = pn.widgets.IntInput(
            name="How many features to combine?", value=2, step=1, start=2, end=5
        )

        self._add_feature_generator_button = pn.widgets.Button(name=">>", max_width=80)
        self._add_feature_generator_button.on_click(self._add_feature_selector_cb)

        self._remove_feature_generator_button = pn.widgets.Button(
            name="Remove", max_width=80
        )
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

        labelled_data = {}

        if os.path.exists("data/test_set.json"):
            with open("data/test_set.json", "r") as json_file:
                labelled_data = json.load(json_file)

        total_labelled = len(list(labelled_data.keys()))

        if total_labelled == 0:
            self.test_set_checkbox = pn.widgets.Checkbox(
                name=f"Should data/test_set.json be used as the test set? [DISABLED: 0 LABELLED DATA]",
                disabled=True,
            )

        if total_labelled < 7:
            self.test_set_checkbox = pn.widgets.Checkbox(
                name=f"Should data/test_set.json be used as the test set? [DISABLED: currently labelled: {total_labelled} - REQUIRED >7]",
                disabled=True,
            )
        elif total_labelled < 500:
            self.test_set_checkbox = pn.widgets.Checkbox(
                name=f"Should data/test_set.json be the used test set? [currently labelled: {total_labelled} - RECOMMENDED >500]"
            )
        else:
            self.test_set_checkbox = pn.widgets.Checkbox(
                name=f"Should data/test_set.json be the used test set? [currently labelled: {total_labelled}]"
            )

    def _verify_valid_selection_cb(self, event):

        selected_labels = self.label_selector.value
        selected_features = self.feature_selector.value

        print(selected_features)
        print(selected_labels)

        exclude_labels = False
        confirm_settings = False

        if len(selected_labels) < 2:
            self.exclude_labels_checkbox.name = (
                "Should remaining labels be removed from Active Learning datasets? "
                + " [DISABLED: Atleast 2 labels must be selected]"
            )
            self.confirm_settings_button.name = "Atleast 2 labels must be selected"
            exclude_labels = True
            confirm_settings = True

        if len(selected_features) < 2:
            confirm_settings = True
            self.confirm_settings_button.name = "Atleast 2 features must be selected"

        if not exclude_labels:
            self.exclude_labels_checkbox.name = (
                "Should remaining labels be removed from Active Learning datasets?"
            )

        if not confirm_settings:
            self.confirm_settings_button.name = "Confirm Settings"

        self.confirm_settings_button.disabled = confirm_settings
        self.exclude_labels_checkbox.disabled = exclude_labels

        self.panel()

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

        if not self.exclude_labels_checkbox.disabled:
            config.settings["exclude_labels"] = self.exclude_labels_checkbox.value
        else:
            config.settings["exclude_labels"] = False

        unclassified_labels = []
        for label in self.label_selector.options:
            if label not in self.label_selector.value:
                unclassified_labels.append(label)

        config.settings["unclassified_labels"] = unclassified_labels
        config.settings["scale_data"] = self.scale_features_checkbox.value
        config.settings["feature_generation"] = self.feature_generator_selected
        config.settings["test_set_file"] = self.test_set_checkbox.value
        config.settings["confirmed"] = True

        self.completed = True

        self.close_button.disabled = False
        self.close_button.button_type = "success"

        self.panel()

    def get_df(self):
        """Return the active learning settings dataframe.

        Returns
        -------
        df : DataFrame
            Data collected up to and including the Active Learning settings panel.

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
                    sizing_mode="stretch_width",
                ),
                pn.Row(self.exclude_labels_checkbox, self._exclude_labels_tooltip),
                pn.Row(self.scale_features_checkbox, self._scale_features_tooltip),
                pn.Row(self.test_set_checkbox),
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
                pn.Row(self.confirm_settings_button, max_height=30),
                pn.Row(pn.Spacer(height=10)),
            )

        return self.column
