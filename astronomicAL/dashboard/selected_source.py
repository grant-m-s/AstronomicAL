from functools import partial

import config
import numpy as np
import pandas as pd
import panel as pn
import param


class SelectedSourceDashboard(param.Parameterized):

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
        self.src.on_change("data", self.panel_cb)

        self.row = pn.Row(pn.pane.Str("loading"))

        self.selected_history = []

        self.url_optical_image = ""

        self.image_zoom = 0.2

        self.add_selected_info()

    def add_selected_info(self):

        self.contents = "Selected Source"
        self.search_id = pn.widgets.AutocompleteInput(
            name="Select ID",
            options=list(self.df[config.settings["id_col"]].values),
            placeholder="Select a source by ID",
            max_height=50,
        )

        self.search_id.param.watch(self.change_selected, "value")

        self.panel()

    def change_selected(self, event):

        if event.new == "":
            return

        selected_source = self.df[self.df[config.settings["id_col"]] == event.new]

        selected_dict = selected_source.set_index(
            config.settings["id_col"]).to_dict('list')
        selected_dict[config.settings["id_col"]] = [event.new]
        self.src.data = selected_dict

        print("Changed selected")

        self.panel()

    def panel_cb(self, attr, old, new):
        print("panel_cb")
        self.panel()

    def check_valid_selected(self):
        selected = False

        print("checking valid selection")

        if config.settings["id_col"] in list(self.src.data.keys()):
            if len(self.src.data[config.settings["id_col"]]) > 0:
                selected = True

        return selected

    def add_selected_to_history(self):

        add_source_to_list = True

        if len(self.selected_history) > 0:
            selected_id = self.src.data[config.settings["id_col"]][0]
            top_of_history = self.selected_history[0]
            if (selected_id == top_of_history):
                add_source_to_list = False
            elif selected_id == "":
                add_source_to_list = False

        if add_source_to_list:
            self.selected_history = [
                self.src.data[config.settings["id_col"]][0]
            ] + self.selected_history

    def deselect_source_cb(self, event):
        self.search_id.value = ""
        self.empty_selected()

    def empty_selected(self):

        empty = {}
        for key in list(self.src.data.keys()):
            empty[key] = []

        self.src.data = empty

    def panel(self):
        # CHANGED :: Remove need to rerender with increases + decreases
        def change_zoom_cb(event, oper):
            if oper == "-":
                self.image_zoom += 0.1
                self.image_zoom = round(self.image_zoom, 1)
            if oper == "+":
                if self.image_zoom == 0.1:
                    pass
                else:
                    self.image_zoom -= 0.1
                    self.image_zoom = round(self.image_zoom, 1)
            try:
                index = self.url_optical_image.rfind("&")
                self.url_optical_image = (
                    f"{self.url_optical_image[:index]}&scale={self.image_zoom}"
                )
                self.optical_image.object = self.url_optical_image
            except:

                print("\n\n\n IMAGE ERROR: \n\n\n")
                print(f"index:{self.url_optical_image.rfind('&')}")
                print(
                    f"new url_optical_image: {self.url_optical_image[:self.url_optical_image.rfind('&')]}&scale={self.image_zoom}")

        def generate_radio_url(ra, dec):
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

            return url

        selected = self.check_valid_selected()

        if selected:

            self.add_selected_to_history()
            try:
                url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra="
                # TODO :: Set Ra Dec Columns
                ra_dec = self.src.data["ra_dec"][0]
                ra = ra_dec[: ra_dec.index(",")]
                print(f"RA_DEC:{ra_dec}")
                dec = ra_dec[ra_dec.index(",") + 1:]

                self.url_optical_image = (
                    f"{url}{ra}&dec={dec}&opt=G&scale={self.image_zoom}"
                )
                self.optical_image.object = self.url_optical_image
            except:
                print("\n\n\n\n Optical Image Timeout \n\n\n\n")

            try:
                url_radio_image = generate_radio_url(ra, dec)
            except:
                url_radio_image = "No Radio Image Available"
                print("\n\n\n\n Radio Image Timeout \n\n\n\n")

            zoom_increase = pn.widgets.Button(
                name="Zoom In", max_height=30, max_width=50
            )
            zoom_increase.on_click(partial(change_zoom_cb, oper="+"))
            zoom_decrease = pn.widgets.Button(
                name="Zoom Out", max_height=30, max_width=50
            )
            zoom_decrease.on_click(partial(change_zoom_cb, oper="-"))
            try:
                # CHANGED :: Slow after this...
                if not self.src.data["png_path_DR16"][0].isspace():
                    print("beginning if")

            except:
                print("\n\n\n\n Radio Image Timeout \n\n\n\n")
                print("leaving if")
                spectra_image = "No spectra available"

            else:
                print("beginning else")
                spectra_image = "No spectra available"
                print("leaving else")

            print("creating deselect")
            deselect_buttton = pn.widgets.Button(name="Deselect")
            deselect_buttton.on_click(self.deselect_source_cb)

            print("setting extra row")
            extra_data_row = pn.Row()
            for col in config.settings["extra_info_cols"]:

                info = f"**{col}**: {str(self.src.data[f'{col}'][0])}"
                extra_data_row.append(
                    pn.pane.Markdown(info, max_width=12
                                     * len(info), max_height=10)
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
                        pn.Row(
                            zoom_increase,
                            zoom_decrease,
                        ),
                        spectra_image,
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
                    self.search_id,
                    pd.DataFrame(self.selected_history,
                                 columns=["Selected IDs"]),
                )
            )

        print("selected source rendered...")

        return self.row
