from __future__ import annotations

from typing import Any

import panel as pn


_MODAL_HOST_ATTR = "_al_modal_host"


def ensure_template_modal_host(template: Any) -> pn.Column:
    """Ensure the template has one stable AstronomicAL modal host.

    Panel templates can be unreliable if different controllers replace
    ``template.modal`` after the app has rendered. Instead, mount one stable
    Column into ``template.modal`` once, then swap that Column's children.
    """

    host = getattr(template, _MODAL_HOST_ATTR, None)

    if host is None:
        host = pn.Column(
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            name="AstronomicAL Modal Host",
        )
        setattr(template, _MODAL_HOST_ATTR, host)

    # Make sure the host, not a controller-specific modal root, is mounted.
    try:
        current = list(template.modal)
    except Exception:
        current = []

    if current != [host]:
        try:
            template.modal[:] = [host]
        except Exception:
            try:
                template.modal.clear()
                template.modal.append(host)
            except Exception:
                template.modal.objects = [host]

    return host


def mount_template_modal(template: Any, content: Any) -> None:
    """Mount modal content into the shared AstronomicAL modal host."""

    host = ensure_template_modal_host(template)

    try:
        host[:] = [content]
    except Exception:
        try:
            host.objects = [content]
        except Exception:
            host.clear()
            host.append(content)


def open_template_modal(template: Any, content: Any) -> None:
    """Mount content into the shared modal host and open the template modal."""

    mount_template_modal(template, content)
    template.open_modal()