from __future__ import annotations

from typing import Any, Callable, Optional

import panel as pn


_MODAL_HOST_ATTR = "_al_overlay_modal_host"
_MODAL_CLOSE_CALLBACKS_ATTR = "_al_overlay_modal_close_callbacks"
_MODAL_TOP_OFFSET_PX = 42


def _install_modal_css() -> None:
    css = f"""
    /*
    AstronomicAL overlay modal system.

    This intentionally does NOT use template.open_modal().
    Native Panel/template modal backdrops are too template-dependent and can
    create opaque overlays. Instead, AstronomicAL mounts a fixed-position
    overlay host into the rendered template header/main area.
    */

    .al-overlay-modal-host {{
        width: 0px !important;
        height: 0px !important;
        min-width: 0px !important;
        min-height: 0px !important;
        overflow: visible !important;
        position: static !important;
        margin: 0 !important;
        padding: 0 !important;
        flex: 0 0 0px !important;
    }}

    .al-template-modal-shell {{
        position: fixed !important;
        inset: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        pointer-events: none !important;
        overflow: visible !important;
        z-index: 100000 !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
    }}

    .al-template-modal-backdrop-button {{
        position: fixed !important;
        inset: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        min-width: 100vw !important;
        min-height: 100vh !important;

        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        outline: none !important;

        cursor: default !important;
        pointer-events: auto !important;
        z-index: 1 !important;

        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
    }}

    .al-template-modal-backdrop-button button,
    .al-template-modal-backdrop-button .bk-btn,
    .al-template-modal-backdrop-button .bk-btn-default {{
        width: 100vw !important;
        height: 100vh !important;
        min-width: 100vw !important;
        min-height: 100vh !important;

        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        outline: none !important;
        color: transparent !important;

        cursor: default !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
    }}

    .al-template-modal-backdrop-button:hover,
    .al-template-modal-backdrop-button:focus,
    .al-template-modal-backdrop-button:active,
    .al-template-modal-backdrop-button button:hover,
    .al-template-modal-backdrop-button button:focus,
    .al-template-modal-backdrop-button button:active,
    .al-template-modal-backdrop-button .bk-btn:hover,
    .al-template-modal-backdrop-button .bk-btn:focus,
    .al-template-modal-backdrop-button .bk-btn:active,
    .al-template-modal-backdrop-button .bk-btn-default:hover,
    .al-template-modal-backdrop-button .bk-btn-default:focus,
    .al-template-modal-backdrop-button .bk-btn-default:active {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        color: transparent !important;
        opacity: 0 !important;
    }}

    .al-template-modal-card {{
        position: fixed !important;
        top: {_MODAL_TOP_OFFSET_PX}px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;

        z-index: 2 !important;
        pointer-events: auto !important;

        max-width: calc(100vw - 48px) !important;
        max-height: calc(100vh - {_MODAL_TOP_OFFSET_PX + 24}px) !important;

        overflow: visible !important;
        margin: 0 !important;
    }}

    .al-modal-card {{
        background: #f3f5f7;
        border: 1px solid rgba(15, 23, 42, 0.24);
        border-radius: 14px;
        padding: 16px 16px 22px 16px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.22);
        color: #111827;
        box-sizing: border-box;
        overflow: hidden;
    }}

    .al-modal-titlebar {{
        width: 100%;
        padding: 10px 14px;
        border-radius: 10px;
        background: #1f2933;
        color: #ffffff;
        box-sizing: border-box;
    }}

    .al-modal-heading {{
        font-size: 16px;
        font-weight: 700;
        line-height: 1.2;
        margin: 0;
        padding: 0;
    }}

    .al-modal-subtitle {{
        font-size: 12px;
        line-height: 1.35;
        opacity: 0.78;
        margin-top: 3px;
    }}

    .al-modal-body {{
        border: 1px solid rgba(15, 23, 42, 0.14);
        border-radius: 10px;
        background: #ffffff;
        color: #111827;
        box-sizing: border-box;
        overflow-y: auto;
        overflow-x: hidden;
    }}

    .al-modal-footer {{
        border-top: 1px solid rgba(15, 23, 42, 0.12);
        background: transparent;
        box-sizing: border-box;
        padding-top: 12px;
        padding-bottom: 6px;
        overflow: visible;
    }}

    .al-modal-section {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px;
        box-sizing: border-box;
    }}

    .al-modal-field-card {{
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        box-sizing: border-box;
    }}

    .al-modal-muted {{
        color: #64748b;
        font-size: 12px;
    }}

    .al-dataset-modal-card {{
        padding-bottom: 26px !important;
    }}

    .al-dataset-modal-body {{
        max-height: calc(100vh - 230px) !important;
    }}

    /*
    Neutralise native template modal layers in case older code opened one.
    */

    .modal-backdrop,
    .modal-backdrop.show,
    .bk-modal-backdrop,
    .bk-Modal-backdrop,
    .pn-modal-backdrop,
    .MuiBackdrop-root,
    .MuiBackdrop-root.MuiModal-backdrop {{
        background: transparent !important;
        background-color: transparent !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }}

    .bk-modal,
    .bk-Modal,
    .modal,
    .pn-modal {{
        background: transparent !important;
        background-color: transparent !important;
        overflow: visible !important;
    }}

    .bk-modal-content,
    .bk-Modal-content,
    .modal-content,
    .pn-modal-content {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        overflow: visible !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    .modal-dialog,
    .bk-modal-dialog,
    .pn-modal-dialog {{
        margin: 0 !important;
        max-width: none !important;
        width: auto !important;
        transform: none !important;
    }}

    body.modal-open {{
        overflow: auto !important;
        padding-right: 0 !important;
    }}
    """

    try:
        if css not in pn.config.raw_css:
            pn.config.raw_css.append(css)
    except Exception:
        pass


def _get_close_callbacks(template: Any) -> list[Callable[[], None]]:
    callbacks = getattr(template, _MODAL_CLOSE_CALLBACKS_ATTR, None)
    if callbacks is None:
        callbacks = []
        setattr(template, _MODAL_CLOSE_CALLBACKS_ATTR, callbacks)
    return callbacks


def _set_close_callbacks(
    template: Any,
    callbacks: list[Callable[[], None]],
) -> None:
    setattr(template, _MODAL_CLOSE_CALLBACKS_ATTR, callbacks)


def _run_close_callbacks(template: Any) -> None:
    """
    Notify the owner of the currently open modal that it has closed.

    This is required for modals opened from persistent UI controls, such as the
    runtime diagnostics button, so those controls can unlock themselves when
    the modal is closed by backdrop click.
    """
    callbacks = list(_get_close_callbacks(template))
    _set_close_callbacks(template, [])

    for callback in callbacks:
        try:
            callback()
        except Exception as exc:
            print(
                "[AstronomicAL modal] close callback failed:",
                repr(exc),
                flush=True,
            )


def _clear_native_template_modal(template: Any) -> None:
    """
    Make sure no native template modal/backdrop is active.

    This does not run AstronomicAL close callbacks. It only neutralises the
    template-native modal system.
    """
    try:
        template.close_modal()
    except Exception:
        pass

    try:
        template.modal[:] = []
    except Exception:
        try:
            template.modal.objects = []
        except Exception:
            try:
                template.modal.clear()
            except Exception:
                pass


def _append_once(area: Any, obj: Any) -> bool:
    try:
        if obj in list(area):
            return True
    except Exception:
        pass

    try:
        area.append(obj)
        return True
    except Exception:
        pass

    try:
        current = list(area)
        area[:] = current + [obj]
        return True
    except Exception:
        pass

    try:
        current = list(getattr(area, "objects", []))
        area.objects = current + [obj]
        return True
    except Exception:
        pass

    return False


def ensure_template_modal_host(template: Any) -> pn.Column:
    """
    Ensure the template has one stable AstronomicAL overlay modal host.
    """
    _install_modal_css()

    host = getattr(template, _MODAL_HOST_ATTR, None)

    if host is None:
        host = pn.Column(
            width=0,
            height=0,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
            name="AstronomicAL Overlay Modal Host",
            css_classes=["al-overlay-modal-host"],
        )
        setattr(template, _MODAL_HOST_ATTR, host)

    mounted = False

    header = getattr(template, "header", None)
    if header is not None:
        mounted = _append_once(header, host)

    if not mounted:
        main = getattr(template, "main", None)
        if main is not None:
            mounted = _append_once(main, host)

    if not mounted:
        sidebar = getattr(template, "sidebar", None)
        if sidebar is not None:
            mounted = _append_once(sidebar, host)

    _clear_native_template_modal(template)

    return host


def close_template_modal(
    template: Any,
    *,
    clear: bool = True,
    notify: bool = True,
) -> None:
    """
    Close the shared AstronomicAL overlay modal.

    notify=True runs callbacks registered by open_template_modal(..., on_close=...).
    Use notify=False only for internal cleanup where the caller explicitly does
    not want to signal modal closure.
    """
    host = getattr(template, _MODAL_HOST_ATTR, None)

    if clear and host is not None:
        try:
            host[:] = []
        except Exception:
            try:
                host.objects = []
            except Exception:
                try:
                    host.clear()
                except Exception:
                    pass

    _clear_native_template_modal(template)

    if notify:
        _run_close_callbacks(template)


def mount_template_modal(
    template: Any,
    content: Any,
    *,
    close_on_backdrop: bool = True,
    on_close: Optional[Callable[[], None]] = None,
) -> None:
    """
    Mount modal content into AstronomicAL's fixed overlay host.

    The surroundings are fully transparent. If close_on_backdrop=True, an
    invisible full-screen Panel button is placed behind the card to capture
    outside clicks.
    """
    # If another AstronomicAL overlay modal is already open, close it first and
    # notify its owner before replacing it.
    existing_host = getattr(template, _MODAL_HOST_ATTR, None)
    if existing_host is not None:
        try:
            has_existing_content = bool(list(existing_host))
        except Exception:
            has_existing_content = bool(getattr(existing_host, "objects", []))

        if has_existing_content:
            close_template_modal(template, clear=True, notify=True)

    host = ensure_template_modal_host(template)

    callbacks: list[Callable[[], None]] = []
    if on_close is not None:
        callbacks.append(on_close)
    _set_close_callbacks(template, callbacks)

    objects: list[Any] = []

    if close_on_backdrop:
        backdrop_button = pn.widgets.Button(
            name="",
            button_type="default",
            sizing_mode="fixed",
            width=1,
            height=1,
            margin=(0, 0, 0, 0),
            css_classes=["al-template-modal-backdrop-button"],
        )

        def _close_from_backdrop(_event: Any = None) -> None:
            close_template_modal(template, clear=True, notify=True)

        backdrop_button.on_click(_close_from_backdrop)

        try:
            backdrop_button.styles = {
                "position": "fixed",
                "inset": "0",
                "width": "100vw",
                "height": "100vh",
                "min-width": "100vw",
                "min-height": "100vh",
                "background": "transparent",
                "background-color": "transparent",
                "border": "none",
                "box-shadow": "none",
                "outline": "none",
                "opacity": "0",
                "pointer-events": "auto",
                "z-index": "1",
                "margin": "0",
                "padding": "0",
            }
        except Exception:
            pass

        objects.append(backdrop_button)

    card = pn.Column(
        content,
        sizing_mode="fixed",
        margin=(0, 0, 0, 0),
        css_classes=["al-template-modal-card"],
    )

    try:
        card.styles = {
            "position": "fixed",
            "top": f"{_MODAL_TOP_OFFSET_PX}px",
            "left": "50%",
            "transform": "translateX(-50%)",
            "z-index": "2",
            "pointer-events": "auto",
            "max-width": "calc(100vw - 48px)",
            "max-height": f"calc(100vh - {_MODAL_TOP_OFFSET_PX + 24}px)",
            "overflow": "visible",
            "margin": "0",
        }
    except Exception:
        pass

    objects.append(card)

    shell = pn.Column(
        *objects,
        width=0,
        height=0,
        sizing_mode="fixed",
        margin=(0, 0, 0, 0),
        css_classes=["al-template-modal-shell"],
    )

    try:
        shell.styles = {
            "position": "fixed",
            "inset": "0",
            "width": "100vw",
            "height": "100vh",
            "pointer-events": "none",
            "overflow": "visible",
            "z-index": "100000",
            "margin": "0",
            "padding": "0",
            "background": "transparent",
            "background-color": "transparent",
        }
    except Exception:
        pass

    try:
        host[:] = [shell]
    except Exception:
        try:
            host.objects = [shell]
        except Exception:
            host.clear()
            host.append(shell)


def open_template_modal(
    template: Any,
    content: Any,
    *,
    close_on_backdrop: bool = True,
    on_close: Optional[Callable[[], None]] = None,
) -> None:
    """
    Open an AstronomicAL overlay modal.

    This deliberately does not call template.open_modal().
    """
    mount_template_modal(
        template,
        content,
        close_on_backdrop=close_on_backdrop,
        on_close=on_close,
    )