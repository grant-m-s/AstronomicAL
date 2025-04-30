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

    optical_image = pn.pane.JPG(
        alt_text="Image Unavailable",
        min_width=200,
        min_height=200,
        sizing_mode="scale_height",
    )

    radio_image = pn.pane.GIF(
        alt_text="Image Unavailable",
        min_width=200,
        min_height=200,
        sizing_mode="scale_height",
    )

    def __init__(self, src, close_button):

        self.df = config.main_df

        self.src = src
        self.src.on_change("data", self._panel_cb)

        self.close_button = close_button

        self.row = pn.Row(pn.pane.Str("loading"))

        self.selected_history = []

        self._url_optical_image = ""

        self._search_status = ""

        self._image_zoom = 0.2

        self._image_updated = False

        self._initialise_optical_zoom_buttons()

        self._add_selected_info()

    def _create_image_tab(self):

        if len(config.settings["extra_image_cols"]) == 0:
            return

        else:
            tab = pn.Tabs(
                sizing_mode="scale_height",
                tabs_location="left",
                max_height=1000,
                max_width=1000,
            )
            for col in config.settings["extra_image_cols"]:
                url = f"{self.src.data[col][0]}"
                if len(url) <= 4:
                    pane = "No Image available for this source."
                elif "." in url:
                    ext = url[url.rindex(".") + 1 :]

                    if ext.lower() in ["jpg", "jpeg"]:
                        pane = pn.pane.JPG(
                            url,
                            alt_text="Image Unavailable",
                            min_width=325,
                            min_height=325,
                            max_width=1000,
                            max_height=1000,
                            sizing_mode="scale_height",
                        )
                    elif ext.lower() == "png":
                        pane = pn.pane.PNG(
                            url,
                            alt_text="Image Unavailable",
                            min_width=325,
                            min_height=325,
                            max_width=1000,
                            max_height=1000,
                            sizing_mode="scale_height",
                        )
                    elif ext.lower() == "svg":
                        pane = pn.pane.SVG(
                            url,
                            alt_text="Image Unavailable",
                            min_width=325,
                            min_height=325,
                            max_width=1000,
                            max_height=1000,
                            sizing_mode="scale_height",
                        )
                    else:
                        pane = f"Unsupported extension {ext}."
                else:
                    pane = "Image url does not contain extension."

                tab.append((col, pane))
            return tab

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
        self.deselect_buttton.disabled = True
        self.deselect_buttton.name = "Deselecting..."
        print("deselecting...")
        self.empty_selected()
        print("deselected...")
        self.search_id.value = ""
        self._search_status = ""
        print("blank...")
        self.deselect_buttton.disabled = False
        self.deselect_buttton.name = "Deselect"

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

    def _generate_radio_url(self, ra, dec):

        ra = float(ra)
        dec = float(dec)

        h = np.floor(ra / 15.0)
        d = ra - h * 15
        m = np.floor(d / 0.25)
        d = d - m * 0.25
        s = d / (0.25 / 60.0)
        s = np.round(s)
        ra_conv = f"{h} {m} {s}"

        sign = 1
        if dec < 0:
            sign = -1

        g = np.abs(dec)
        d = np.floor(g) * sign
        g = g - np.floor(g)
        m = np.floor(g * 60.0)
        g = g - m / 60.0
        s = g * 3600.0

        s = np.round(s)

        dec_conv = f"{d} {m} {s}"

        url1 = "https://third.ucllnl.org/cgi-bin/firstimage?RA="
        url2 = "&Equinox=J2000&ImageSize=2.5&MaxInt=200&GIF=1"
        url = f"{url1}{ra_conv} {dec_conv}{url2}"

        print(url)

        return url

    def _change_zoom_cb(self, event, oper):

        self.zoom_increase.disabled = False
        if oper == "out":
            self._image_zoom += 0.1
            self._image_zoom = round(self._image_zoom, 1)
        if oper == "in":
            if self._image_zoom <= 0.1:
                self._image_zoom = 0.1
                self.zoom_increase.disabled = True
            else:
                self._image_zoom -= 0.1
                self._image_zoom = round(self._image_zoom, 1)

        optical_url = self._get_optical_url()

        if len(optical_url) > 0:

            try:
                index = self._url_optical_image.rfind("&")
                self.optical_image = pn.pane.JPG(
                    alt_text="Image Unavailable",
                    min_width=200,
                    min_height=200,
                    sizing_mode="scale_height",
                )
                self._url_optical_image = f"{optical_url}{self._image_zoom}"
                self.optical_image.object = self._url_optical_image
                # Process(target=self._update_default_images).start()
                self.row[0][0][0][0][1][0] = pn.Row(self.optical_image)

            except:

                print("\n\n\n IMAGE ERROR: \n\n\n")
                print(f"orig:{self._url_optical_image}")
                print(f"index:{self._url_optical_image.rfind('&')}")
                print(
                    f"new url_optical_image: {self._url_optical_image[:self._url_optical_image.rfind('&')]}&scale={self._image_zoom}"
                )

            self._update_default_images()

    def _get_optical_url(self):
        url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra="

        if self.check_required_column("ra_dec"):
            ra_dec = self.src.data["ra_dec"][0]
            ra = ra_dec[: ra_dec.index(",")]
            dec = ra_dec[ra_dec.index(",") + 1 :]

            url = f"{url}{ra}&dec={dec}&opt=G&scale="

            return url

        else:
            return ""

    def _update_default_images(self):

        optical_url = self._get_optical_url()
        if len(optical_url) > 0:
            try:
                self._url_optical_image = f"{optical_url}{self._image_zoom}"
                r = requests.get(f"{self._url_optical_image}", timeout=20.0)
                self.optical_image.object = self._url_optical_image
                print(self.optical_image.object)
            except ConnectionError as e:
                print("optical image unavailable")
                print(e)

            try:
                ra_dec = self.src.data["ra_dec"][0]
                ra = ra_dec[: ra_dec.index(",")]
                dec = ra_dec[ra_dec.index(",") + 1 :]
                self._url_radio_image = self._generate_radio_url(ra, dec)
                r = requests.get(f"{self._url_radio_image}", timeout=20.0)
                self.radio_image.object = self._url_radio_image
            except ConnectionError as e:
                print("radio image unavailable")
                print(e)

            self._initialise_optical_zoom_buttons()

            self.row[0][0][0][0][1] = pn.Row(
                pn.Row(self.optical_image),
                self.radio_image,
            )

            # self.panel()

    def _initialise_optical_zoom_buttons(self):

        self.zoom_increase = pn.widgets.Button(
            name="Zoom In", max_height=30, max_width=50
        )
        self.zoom_increase.on_click(partial(self._change_zoom_cb, oper="in"))
        self.zoom_decrease = pn.widgets.Button(
            name="Zoom Out", max_height=30, max_width=50
        )
        self.zoom_decrease.on_click(partial(self._change_zoom_cb, oper="out"))

    def check_required_column(self, column):
        """Check `df` has the required column.

        Parameters
        ----------
        column : str
            Check whether this column is in `df`

        Returns
        -------
        has_required : bool
            Whether the column is in `df`.

        """
        has_required = False
        if column in list(self.df.columns):
            has_required = True

        return has_required

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

            if self.check_required_column("ra_dec"):
                button_row.append(self.zoom_increase)
                button_row.append(self.zoom_decrease)

            self.deselect_buttton = pn.widgets.Button(name="Deselect")
            self.deselect_buttton.on_click(self._deselect_source_cb)

            extra_data_list = [
                ["Source ID", self.src.data[config.settings["id_col"]][0]]
            ]

            extra_data_string = "" 
            for i, col in enumerate(config.settings["extra_info_cols"]):

                extra_data_string += f"{col}: {self.src.data[f'{col}'][0]} \n"

                # extra_data_list.append([col, self.src.data[f"{col}"][0]])

            # extra_data_df = pd.DataFrame(extra_data_list, columns=["Column", "Value"])
            # extra_data_pn = pn.widgets.DataFrame(
            #     extra_data_df, show_index=False, autosize_mode="fit_viewport"
            # )

            extra_data_pn = extra_data_string
            self.row[0] = pn.Card(
                pn.Column(
                    pn.Row(
                        pn.Column(
                            button_row,
                            pn.Row(
                                pn.Row(self.optical_image),
                                self.radio_image,
                            ),
                        ),
                        pn.Row(extra_data_pn), max_height=250, max_width=300),
                    ),
                    self._create_image_tab(),
                collapsible=False,
                header=pn.Row(self.close_button, self.deselect_buttton, max_width=300),
            )

            if not self._image_updated:
                self._image_updated = True
                Process(target=self._update_default_images).start()

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
