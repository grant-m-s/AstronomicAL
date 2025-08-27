from astronomicAL.extensions import feature_generation

import astronomicAL.config as config
import pandas as pd
import panel as pn
import json
import os
import param


class ExploringSettings(param.Parameterized):
    """The Exploring Settings Stage used in the settings pipeline. It is a simplified version 
    of the Active Learning settings stage.

    Parameters
    ----------
    close_button : Panel Button
        Close button widget from the parent settings dashboard to allow the
        button to be updated when all settings have been completed.

    Attributes
    ----------
    df : DataFrame
        The shared dataframe which holds all the data.
    feature_selector : Panel CrossSelector
        CrossSelector widget for choosing which of the data columns should be
        used during the machine learning steps.
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

        self.feature_selector.size = 5

        self.feature_selector._search[True].max_height = 20
        self.feature_selector._search[False].max_height = 20

        self.feature_selector._search[True].max_width = 300
        self.feature_selector._search[False].max_width = 300

        self.feature_selector._lists[True].width = 600
        self.feature_selector._lists[False].width = 600

        self.feature_selector._buttons[True].max_width = 50
        self.feature_selector._buttons[False].max_width = 50

        self.feature_selector._buttons[True].max_height = 30
        self.feature_selector._buttons[False].max_height = 30

        self.feature_selector._buttons[True].margin = (50, 10, 0, 10)
        self.feature_selector._buttons[False].margin = (10, 10, 0, 10)

        self.feature_selector._composite[:] = [
            self.feature_selector._unselected,
            pn.Column(
                self.feature_selector._buttons[True],
                self.feature_selector._buttons[False],
            ),
            self.feature_selector._selected,
        ]

    def _initialise_widgets(self):

        self.feature_selector = pn.widgets.CrossSelector(
            name="**On which features do you want to apply the operations?**",
            value=[],
            options=[],
            width=500,
            max_width=600,
            # sizing_mode="fixed",
        )

        self.feature_selector._buttons[True].on_click(self._verify_valid_selection_cb)
        self.feature_selector._buttons[False].on_click(self._verify_valid_selection_cb)

        self.feature_generator = pn.widgets.Select(
            name="Create Feature Combinations?",
            options=list(feature_generation.get_oper_dict().keys()),
            max_height=30,
        )

        self._add_feature_generator_button = pn.widgets.Button(name=">>", max_width=80)
        self._add_feature_generator_button.on_click(self._add_feature_selector_cb)

        self._remove_feature_generator_button = pn.widgets.Button(
            name="Remove", max_width=80
        )
        self._remove_feature_generator_button.on_click(self._remove_feature_selector_cb)

        self._feature_generator_dataframe = pn.pane.DataFrame(
            pd.DataFrame(self.feature_generator_selected, columns=["oper"]),
            name="",
            index=False,
        )
        self.ra_column_selector = pn.widgets.Select(
            name="Column with RA values", options=[], max_height = 30
        )
        self.dec_column_selector = pn.widgets.Select(
            name="Column with Dec values", options=[], max_height = 30
        )
        
        self.confirm_settings_button = pn.widgets.Button(
            name="Confirm Settings", button_type="primary"
        )
        self.confirm_settings_button.on_click(self._confirm_settings_cb)




    def _verify_valid_selection_cb(self, event):

        selected_features = self.feature_selector.value

        confirm_settings = False

        if len(selected_features) < 2:
            confirm_settings = True
            self.confirm_settings_button.name = "At least 2 features must be selected"

        if not confirm_settings:
            self.confirm_settings_button.name = "Confirm Settings"

        self.confirm_settings_button.disabled = confirm_settings
       
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
            features = list(self.df.columns)
            try:
                features.remove(config.settings["id_col"])
            except ValueError:
                pass 
            try:
                features.remove(config.settings["label_col"])
            except ValueError:
                pass 

            self.feature_selector.options = features
            self.ra_column_selector.options = features
            self.dec_column_selector.options = features
    


    def _add_feature_selector_cb(self, event):

        new_feature_generator = [self.feature_generator.value, 2]

        if new_feature_generator not in self.feature_generator_selected:
            self.feature_generator_selected.append(new_feature_generator)
            self._feature_generator_dataframe.object = pd.DataFrame(
                [i[0] for i in self.feature_generator_selected], columns=["oper"])



    def _remove_feature_selector_cb(self, event):
        self.feature_generator_selected = self.feature_generator_selected[:-1]
        self._feature_generator_dataframe.object = pd.DataFrame(
             [i[0] for i in self.feature_generator_selected], columns=["oper"])
        


    def get_default_variables(self):
        x_var = self.feature_selector.value[0]
        y_var = self.feature_selector.value[1]
        return (x_var, y_var)


    def _confirm_settings_cb(self, event):
        print("Saving settings...")

        config.settings["default_vars"] = self.get_default_variables()
        config.settings["labels_to_train"] = config.settings["labels"]
        config.settings["features_for_training"] = self.feature_selector.value

        config.settings["exclude_labels"] = False

        config.settings["exclude_unknown_labels"] = False

        config.settings["unclassified_labels"] = []
        config.settings["scale_data"] = False
        config.settings["feature_generation"] = self.feature_generator_selected
        config.settings["test_set_file"] = False
        config.settings["confirmed"] = True
        config.settings["ra_col_name"] = self.ra_column_selector.value
        config.settings["dec_col_name"] = self.dec_column_selector.value

        if "save_button" in config.settings.keys():
            config.settings["save_button"].disabled = False

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
                    self.feature_selector,
                    sizing_mode="stretch_width",
                    max_height=150,
                ),

                pn.Row(
                     self.feature_selector.name, max_height=30
                ),
                pn.Row(
                    self.feature_generator,
                    pn.Column(
                        self._add_feature_generator_button,
                        self._remove_feature_generator_button,
                    ),
                    self._feature_generator_dataframe,
                   sizing_mode="stretch_width",
                ),
                pn.Row(self.ra_column_selector,
                       self.dec_column_selector),
                pn.Row(self.confirm_settings_button, max_height=30),
                pn.Row(pn.Spacer(height=30)),
            )

        return self.column