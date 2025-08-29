from functools import partial
from requests.exceptions import ConnectionError
from multiprocessing import Process
import astronomicAL.config as config
import numpy as np
import pandas as pd
import panel as pn
import requests


class SelectedSourceDashboard:
    """A Dashboard used for showing detailed information about a source.

    Optical and Radio images of the source are provided free when the user has
    Right Ascension (RA) and Declination (Dec) information. The user can specify
    what extra information they want to display when a source is selected.

    Parameters
    ----------
    src : ColumnDataSource
        The shared data source which holds the current selected source.

    Attributes
    ----------
    df : DataFrame
        The shared dataframe which holds all the data.
    row : Panel Row
        The panel is housed in a row which can then be rendered by the
        parent Dashboard.
    selected_history : List of str
        List of source ids that have been selected.
    optical_image : Panel Pane JPG
        Widget for holding the JPG image of the selected source based its RADEC.
        The image is pulled from the SDSS SkyServer DR16 site.
    radio_image : Panel Pane GIF
        Widget for holding the GIF image of the selected source based its RADEC.
        The image is pulled from the FIRST cutout server.
    _image_zoom : float
        A float containing the current zoom level of the `optical_image`. The
        zoom is controlled by the zoom in and out buttons on the dashboard
    src : ColumnDataSource
        The shared data source which holds the current selected source.

    """

    
    def __init__(self, src, close_button):

        self.df = config.main_df

        self.src = src
        self.src.on_change("data", self._panel_cb)

        self.close_button = close_button

        self.row = pn.Row(pn.pane.Str("loading"))

        self.selected_history = []

        self._search_status = ""

        self._add_selected_info()


    def _add_selected_info(self):

        self.contents = "Selected Source"
        self.search_id = pn.widgets.TextInput(
            name="Select ID",
            placeholder="Select a source by ID",
            max_height=50,
        )

        self.search_id.param.watch(self._change_selected, "value")

        self.panel()

    def _change_selected(self, event):

        if event.new == "":
            self._search_status = ""
            self.panel()
            return

        self._search_status = "Searching..."

        if event.new not in list(self.df[config.settings["id_col"]].values):
            self._search_status = "ID not found in dataset"
            self.panel()
            return

        selected_source = self.df[self.df[config.settings["id_col"]] == event.new]

        selected_dict = selected_source.set_index(config.settings["id_col"]).to_dict(
            "list"
        )
        selected_dict[config.settings["id_col"]] = [event.new]
        self.src.data = selected_dict

        self.panel()

    def _panel_cb(self, attr, old, new):
        self._image_updated = False
        self._image_zoom = 0.2
        self.panel()

    def _check_valid_selected(self):
        selected = False

        if config.settings["id_col"] in list(self.src.data.keys()):
            if len(self.src.data[config.settings["id_col"]]) > 0:
                if self.src.data[config.settings["id_col"]][0] in list(
                    self.df[config.settings["id_col"]].values
                ):
                    selected = True

        return selected

    def _add_selected_to_history(self):

        add_source_to_list = True

        if len(self.selected_history) > 0:
            selected_id = self.src.data[config.settings["id_col"]][0]
            top_of_history = self.selected_history[0]
            if selected_id == top_of_history:
                add_source_to_list = False
            elif selected_id == "":
                add_source_to_list = False

        if add_source_to_list:
            self.selected_history = [
                self.src.data[config.settings["id_col"]][0]
            ] + self.selected_history

    def _deselect_source_cb(self, event):
        self.deselect_button.disabled = True
        self.deselect_button.name = "Deselecting..."
        print("deselecting...")
        self.empty_selected()
        print("deselected...")
        self.search_id.value = ""
        self._search_status = ""
        print("blank...")
        self.deselect_button.disabled = False
        self.deselect_button.name = "Deselect"

    def empty_selected(self):
        """Deselect sources by emptying `src.data`.

        Returns
        -------
        None

        """
        empty = {}
        for key in list(self.src.data.keys()):
            empty[key] = []

        self.src.data = empty


    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which can then be rendered by the
            parent Dashboard.

        """
        # CHANGED :: Remove need to rerender with increases + decreases

        selected = self._check_valid_selected()

        if selected:

            self._add_selected_to_history()

            button_row = pn.Row()


            self.deselect_button = pn.widgets.Button(name="Deselect")
            self.deselect_button.on_click(self._deselect_source_cb)

            extra_data_list = [
                ["Source ID", self.src.data[config.settings["id_col"]][0]]
            ]

            for i, col in enumerate(config.settings["extra_info_cols"]):

                extra_data_list.append([col, self.src.data[f"{col}"][0]])

            extra_data_df = pd.DataFrame(extra_data_list, columns=["Column", "Value"])
            extra_data_pn = pn.pane.DataFrame(
                extra_data_df, index=False,
            )
            self.row[0] = pn.Card(
                    pn.Row(extra_data_pn, max_height=250, max_width=300),
                    collapsible=False,
                    header=pn.Row(self.close_button, self.deselect_button, max_width=300),
                )

   

        else:
            self.row[0] = pn.Card(
                pn.Column(
                    self.search_id,
                    self._search_status,
                    pn.Row(
                        pn.widgets.DataFrame(
                            pd.DataFrame(
                                self.selected_history, columns=["Selected IDs"]
                            ),
                            show_index=False,
                        ),
                        max_width=300,
                    ),
                ),
                header=pn.Row(self.close_button, max_width=300),
            )

        return self.row