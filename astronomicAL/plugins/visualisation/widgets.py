from __future__ import annotations

import panel as pn


def header_select(parameter, *, name: str):
    return pn.widgets.Select.from_param(
        parameter,
        name=name,
        sizing_mode="stretch_width",
        height=44,
        margin=(0, 2, 0, 0),
    )


def settings_select(parameter, *, name: str, width: int = 145):
    return pn.widgets.Select.from_param(
        parameter,
        name=name,
        width=width,
        height=40,
        sizing_mode="fixed",
        margin=(0, 6, 2, 6),
    )


def settings_multichoice(parameter, *, name: str, width: int = 200):
    widget = pn.widgets.MultiChoice.from_param(
        parameter,
        name=name,
        width=width,
        height=60,
        sizing_mode="fixed",
        margin=(0, 6, 2, 6),
    )

    if "allow_html" in widget.param:
        try:
            widget.allow_html = False
        except Exception:
            pass

    return widget


def settings_int_input(parameter, *, name: str, width: int = 140):
    return pn.widgets.IntInput.from_param(
        parameter,
        name=name,
        width=width,
        height=50,
        sizing_mode="fixed",
        margin=(0, 6, 2, 6),
    )


def settings_float_slider(parameter, *, name: str, width: int = 180):
    return pn.widgets.FloatSlider.from_param(
        parameter,
        name=name,
        width=width,
        height=40,
        sizing_mode="fixed",
        margin=(0, 8, 2, 6),
    )


def settings_int_slider(parameter, *, name: str, width: int = 180):
    return pn.widgets.IntSlider.from_param(
        parameter,
        name=name,
        width=width,
        height=40,
        sizing_mode="fixed",
        margin=(0, 8, 2, 6),
    )


def settings_checkbox(parameter, *, name: str):
    return pn.widgets.Checkbox.from_param(
        parameter,
        name=name,
        width=120,
        height=28,
        sizing_mode="fixed",
        margin=(10, 8, 0, 6),
    )


def settings_box(*controls):
    """Compact settings row/box.

    The parent settings pane owns the fixed height and scrolling. This box
    should fit its content naturally and should not try to stretch with the
    outer plot panel.
    """

    return pn.FlexBox(
        *controls,
        sizing_mode="stretch_width",
        height_policy="fit",
        margin=(0, 0, 0, 0),
        styles={
            "overflow": "visible",
            "align-content": "flex-start",
            "align-items": "flex-start",
            "gap": "2px 6px",
            "padding": "4px 6px 4px 6px",
            "border-top": "1px solid #ddd",
            "border-bottom": "1px solid #eee",
            "background": "#fafafa",
            "box-sizing": "border-box",
        },
    )