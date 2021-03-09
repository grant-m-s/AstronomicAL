from functools import partial

import astronomicAL.config as config
import numpy as np
import pandas as pd
import panel as pn
import param


class SelectedSourceDashboard(param.Parameterized):
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
    spectra_image : Panel Pane PNG
        Widget for holding the PNG image of the selected source. The spectra
        requires a "png_path_DR16" column.
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

    spectra_image = pn.pane.PNG(
        alt_text="Image Unavailable",
        min_width=400,
        min_height=400,
        sizing_mode="scale_height",
    )

    def __init__(self, src, **params):
        super(SelectedSourceDashboard, self).__init__(**params)

        self.df = config.main_df

        self.src = src
        self.src.on_change("data", self._panel_cb)

        self.row = pn.Row(pn.pane.Str("loading"))

        self.selected_history = []

        self._url_optical_image = ""

        self._image_zoom = 0.2

        self._initialise_optical_zoom_buttons()

        self._add_selected_info()

    def _add_selected_info(self):

        self.contents = "Selected Source"
        self.search_id = pn.widgets.AutocompleteInput(
            name="Select ID",
            options=list(self.df[config.settings["id_col"]].values),
            placeholder="Select a source by ID",
            max_height=50,
        )

        # TODO :: Fix the weird printouts of the autocomplete search
        # self.search_id.param.watch(self._change_selected, "value")

        self.panel()

    def _change_selected(self, event):

        if event.new == "":
            return

        selected_source = self.df[self.df[config.settings["id_col"]] == event.new]

        selected_dict = selected_source.set_index(config.settings["id_col"]).to_dict(
            "list"
        )
        selected_dict[config.settings["id_col"]] = [event.new]
        self.src.data = selected_dict

        print("Changed selected")

        self.panel()

    def _panel_cb(self, attr, old, new):
        print("_panel_cb")
        self.panel()

    def _check_valid_selected(self):
        selected = False

        print("checking valid selection")

        if config.settings["id_col"] in list(self.src.data.keys()):
            if len(self.src.data[config.settings["id_col"]]) > 0:
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
        self.search_id.value = ""
        self.empty_selected()

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
        # TODO :: Verify
        ra = float(ra)
        dec = float(dec)
        print(f"ra:{ra}, dec:{dec}")

        h = np.floor(ra / 15.0)
        d = ra - h * 15
        m = np.floor(d / 0.25)
        d = d - m * 0.25
        s = d / (0.25 / 60.0)
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

        dec_conv = f"{d} {m} {s}"

        print(f"ra_conv: {ra_conv}, dec_conv: {dec_conv}")

        url1 = "https://third.ucllnl.org/cgi-bin/firstimage?RA="
        url2 = "&Equinox=J2000&ImageSize=2.5&MaxInt=200&GIF=1"
        url = f"{url1}{ra_conv} {dec_conv}{url2}"

        print(url)

        return url

    def _change_zoom_cb(self, event, oper):
        if oper == "-":
            self._image_zoom += 0.1
            self._image_zoom = round(self._image_zoom, 1)
        if oper == "+":
            if self._image_zoom == 0.1:
                pass
            else:
                self._image_zoom -= 0.1
                self._image_zoom = round(self._image_zoom, 1)
        try:
            index = self._url_optical_image.rfind("&")
            self._url_optical_image = (
                f"{self._url_optical_image[:index]}&scale={self._image_zoom}"
            )
            self.optical_image.object = self._url_optical_image
        except:

            print("\n\n\n IMAGE ERROR: \n\n\n")
            print(f"index:{self._url_optical_image.rfind('&')}")
            print(
                f"new url_optical_image: {self._url_optical_image[:self._url_optical_image.rfind('&')]}&scale={self._image_zoom}"
            )

    def _update_default_images(self):
        url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra="
        # TODO :: Set Ra Dec Columns

        if self.check_required_column("ra_dec"):
            ra_dec = self.src.data["ra_dec"][0]
            ra = ra_dec[: ra_dec.index(",")]
            print(f"RA_DEC:{ra_dec}")
            dec = ra_dec[ra_dec.index(",") + 1 :]

            self._url_optical_image = (
                f"{url}{ra}&dec={dec}&opt=G&scale={self._image_zoom}"
            )
            self.optical_image.object = self._url_optical_image

            self.url_radio_image = self._generate_radio_url(ra, dec)
            self.radio_image.object = self.url_radio_image

            self._initialise_optical_zoom_buttons()

    def _initialise_optical_zoom_buttons(self):

        self.zoom_increase = pn.widgets.Button(
            name="Zoom In", max_height=30, max_width=50
        )
        self.zoom_increase.on_click(partial(self._change_zoom_cb, oper="+"))
        self.zoom_decrease = pn.widgets.Button(
            name="Zoom Out", max_height=30, max_width=50
        )
        self.zoom_decrease.on_click(partial(self._change_zoom_cb, oper="-"))

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

            self._update_default_images()

            if self.check_required_column("png_path_DR16"):

                if not self.src.data["png_path_DR16"][0].isspace():
                    print("beginning if")

                    self.spectra_image.object = self.src.data["png_path_DR16"][0]
                    spectra = self.spectra_image

                else:
                    print("beginning else")
                    spectra = "No Spectra Image Available."
                    print("leaving else")

            else:
                spectra = ""

            button_row = pn.Row()
            if self.check_required_column("ra_dec"):
                button_row.append(self.zoom_increase)
                button_row.append(self.zoom_decrease)

            print("creating deselect")
            deselect_buttton = pn.widgets.Button(name="Deselect")
            deselect_buttton.on_click(self._deselect_source_cb)

            print("setting extra row")
            extra_data_row = pn.Row()
            for col in config.settings["extra_info_cols"]:

                info = f"**{col}**: {str(self.src.data[f'{col}'][0])}"
                extra_data_row.append(
                    pn.pane.Markdown(info, max_width=12 * len(info), max_height=10)
                )
            print("setting row")
            self.row[0] = pn.Card(
                pn.Column(
                    pn.pane.Markdown(
                        f'**Source ID**: {self.src.data[config.settings["id_col"]][0]}',
                        max_height=10,
                    ),
                    extra_data_row,
                    pn.Column(
                        pn.Row(
                            self.optical_image,
                            self.radio_image,
                        ),
                        button_row,
                        spectra,
                    ),
                ),
                collapsible=False,
                header=deselect_buttton,
            )

            print("row set")

        else:
            print("Nothing is selected!")
            # print(self.src.data)
            self.row[0] = pn.Card(
                pn.Column(
                    # self.search_id,
                    pn.widgets.DataFrame(
                        pd.DataFrame(self.selected_history, columns=["Selected IDs"]),
                        show_index=False,
                    ),
                )
            )

        print("selected source rendered...")

        return self.row
