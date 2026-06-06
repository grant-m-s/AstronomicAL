from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Callable

import panel as pn

from astronomicAL.utils.save_config import (
    default_layout_directory,
    list_workspace_layouts,
    save_workspace_as,
    save_workspace_timestamped,
)


def _layout_options(context: Any) -> dict[str, str]:
    paths = list_workspace_layouts(context=context)
    options: dict[str, str] = {}

    layout_dir = default_layout_directory(context)
    for path in paths:
        try:
            label = str(path.relative_to(layout_dir))
        except Exception:
            label = path.name
        options[label] = str(path)

    return options


def _load_snapshot_from_upload(file_input: pn.widgets.FileInput) -> dict[str, Any] | None:
    if not file_input.value:
        return None

    value = file_input.value
    if isinstance(value, bytes):
        text = value.decode("utf-8")
    else:
        text = str(value)

    snapshot = json.loads(text)
    if not isinstance(snapshot, dict):
        raise TypeError("Uploaded layout JSON must contain an object at the top level.")

    return snapshot


def _load_layout_snapshot(
    *,
    context: Any,
    snapshot: dict[str, Any],
    source_label: str,
    notify_success: Callable[[str], None],
    notify_warning: Callable[[str], None],
) -> None:
    persistence = getattr(context, "persistence", None)
    if persistence is None:
        raise RuntimeError("context.persistence is not configured.")

    if hasattr(persistence, "reconcile"):
        issues = persistence.reconcile(snapshot, strict=False)
    else:
        issues = persistence.restore(snapshot, strict=False)

    issue_count = len(issues or [])
    if issue_count:
        notify_warning(f"Loaded layout with {issue_count} issue(s): {source_label}")
    else:
        notify_success(f"Loaded layout: {source_label}")


def create_layout_controls(
    *,
    context: Any,
    template: Any | None = None,
) -> pn.Row:
    """
    Header controls for layout save/load.

    Save As and Load use a fixed-position responsive overlay. The overlay is
    mounted in the header with a zero-size anchor so it does not alter the
    header layout.
    """

    toast_pane = pn.pane.HTML(
        "",
        width=0,
        height=0,
        margin=0,
        sizing_mode="fixed",
        styles={
            "width": "0px",
            "height": "0px",
            "overflow": "visible",
            "padding": "0",
            "margin": "0",
        },
    )

    drawer = pn.Column(
        visible=False,
        width=560,
        max_width=560,
        sizing_mode="fixed",
        margin=0,
        styles={
            "position": "fixed",
            "top": "64px",
            "right": "16px",
            "width": "min(560px, calc(100vw - 32px))",
            "max-width": "calc(100vw - 32px)",
            "max-height": "calc(100vh - 96px)",
            "overflow-y": "auto",
            "box-sizing": "border-box",
            "background": "white",
            "color": "#222",
            "border": "1px solid rgba(0, 0, 0, 0.20)",
            "border-radius": "10px",
            "box-shadow": "0 10px 30px rgba(0, 0, 0, 0.28)",
            "padding": "14px",
            "z-index": "2147483646",
        },
    )

    drawer_anchor = pn.Column(
        drawer,
        width=0,
        height=0,
        margin=0,
        sizing_mode="fixed",
        styles={
            "width": "0px",
            "height": "0px",
            "overflow": "visible",
            "padding": "0",
            "margin": "0",
        },
    )

    toast_counter = {"value": 0}
    button_flash_state: dict[int, dict[str, Any]] = {}
    button_original_names: dict[int, str] = {}

    def _toast_html(message: str, *, kind: str) -> str:
        safe_message = html.escape(str(message))

        colours = {
            "success": {
                "background": "#d1e7dd",
                "border": "#badbcc",
                "text": "#0f5132",
            },
            "error": {
                "background": "#f8d7da",
                "border": "#f5c2c7",
                "text": "#842029",
            },
            "warning": {
                "background": "#fff3cd",
                "border": "#ffecb5",
                "text": "#664d03",
            },
            "info": {
                "background": "#cff4fc",
                "border": "#b6effb",
                "text": "#055160",
            },
        }

        colour = colours.get(kind, colours["info"])

        return f"""
<div
  style="
    position: fixed;
    top: clamp(56px, 8vh, 92px);
    left: 16px;
    right: 16px;
    max-width: 420px;
    margin-left: auto;
    z-index: 2147483647;
    box-sizing: border-box;
    pointer-events: none;
  "
>
  <div
    style="
      width: 100%;
      box-sizing: border-box;
      display: block;
      padding: 12px 14px;
      border-radius: 8px;
      border: 1px solid {colour["border"]};
      background: {colour["background"]};
      color: {colour["text"]};
      font-size: 14px;
      line-height: 1.35;
      font-weight: 500;
      white-space: normal;
      overflow-wrap: anywhere;
      word-break: break-word;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
      pointer-events: auto;
    "
  >
    {safe_message}
  </div>
</div>
"""

    def _hide_toast(expected_counter: int) -> None:
        if toast_counter["value"] != expected_counter:
            return
        toast_pane.object = ""

    def _notify(
        message: str,
        *,
        kind: str = "info",
        duration: int = 4500,
    ) -> None:
        toast_counter["value"] += 1
        current_counter = toast_counter["value"]

        toast_pane.object = _toast_html(message, kind=kind)

        doc = getattr(pn.state, "curdoc", None)
        if doc is not None and hasattr(doc, "add_timeout_callback"):
            try:
                doc.add_timeout_callback(
                    lambda: _hide_toast(current_counter),
                    duration,
                )
            except Exception:
                pass

        print(f"[layout:{kind}] {message}")

    def _flash_button(
        button: pn.widgets.Button,
        temporary_name: str,
        *,
        duration: int = 2500,
    ) -> None:
        button_key = id(button)
        original_name = button_original_names.get(button_key, button.name)

        state = button_flash_state.setdefault(
            button_key,
            {
                "counter": 0,
            },
        )

        state["counter"] += 1
        current_counter = state["counter"]

        button.name = temporary_name

        def _restore_button_name() -> None:
            latest_state = button_flash_state.get(button_key)
            if latest_state is None:
                return
            if latest_state.get("counter") != current_counter:
                return

            button.name = original_name

        doc = getattr(pn.state, "curdoc", None)
        if doc is not None and hasattr(doc, "add_timeout_callback"):
            try:
                doc.add_timeout_callback(_restore_button_name, duration)
                return
            except Exception:
                pass

        button.name = original_name

    quick_save_button = pn.widgets.Button(
        name="Quick Save Layout",
        button_type="default",
        disabled=context is None,
        width=155,
        height=34,
        margin=(0, 0, 0, 0),
    )
    save_as_button = pn.widgets.Button(
        name="Save Layout As...",
        button_type="default",
        disabled=context is None,
        width=145,
        height=34,
        margin=(0, 0, 0, 0),
    )
    load_button = pn.widgets.Button(
        name="Load Layout...",
        button_type="default",
        disabled=context is None,
        width=125,
        height=34,
        margin=(0, 0, 0, 0),
    )

    button_original_names[id(quick_save_button)] = quick_save_button.name
    button_original_names[id(save_as_button)] = save_as_button.name
    button_original_names[id(load_button)] = load_button.name

    def _close_drawer() -> None:
        drawer.visible = False
        drawer[:] = []

    def _show_drawer(*objects: Any) -> None:
        drawer[:] = list(objects)
        drawer.visible = True

    def _drawer_header(title: str) -> pn.Row:
        close_button = pn.widgets.Button(
            name="×",
            button_type="default",
            width=34,
            height=30,
            margin=(0, 0, 0, 8),
        )

        close_button.on_click(lambda _event: _close_drawer())

        return pn.Row(
            pn.pane.Markdown(
                f"### {title}",
                sizing_mode="stretch_width",
                margin=(2, 0, 0, 0),
            ),
            close_button,
            sizing_mode="stretch_width",
        )

    def _quick_save(_event: Any) -> None:
        if context is None:
            _notify("Quick save failed: no context.", kind="error", duration=7000)
            return

        quick_save_button.disabled = True
        try:
            path = save_workspace_timestamped(context)
            _notify(f"Saved layout: {path}", kind="success", duration=5000)
            _flash_button(quick_save_button, "Saved ✓")
        except Exception as exc:
            _notify(f"Quick save failed: {exc}", kind="error", duration=8000)
            _flash_button(quick_save_button, "Save failed")
        finally:
            quick_save_button.disabled = False

    def _save_as(_event: Any) -> None:
        if context is None:
            _notify("Save failed: no context.", kind="error", duration=7000)
            return

        name_input = pn.widgets.TextInput(
            name="Layout filename",
            placeholder="my-layout",
            sizing_mode="stretch_width",
        )
        save_button = pn.widgets.Button(
            name="Save",
            button_type="primary",
            width=90,
            height=34,
        )
        cancel_button = pn.widgets.Button(
            name="Cancel",
            button_type="default",
            width=90,
            height=34,
        )
        message = pn.pane.Markdown("", visible=False, sizing_mode="stretch_width")

        def _do_save(_save_event: Any = None) -> None:
            if save_button.disabled:
                return

            save_button.disabled = True
            try:
                path = save_workspace_as(context, name_input.value)
                _notify(f"Saved layout: {path}", kind="success", duration=5000)
                _flash_button(save_as_button, "Saved ✓")
                _close_drawer()
            except Exception as exc:
                message.object = f"Save failed: `{exc}`"
                message.visible = True
                _notify(f"Save layout failed: {exc}", kind="error", duration=8000)
                _flash_button(save_as_button, "Save failed")
            finally:
                save_button.disabled = False

        def _do_cancel(_cancel_event: Any) -> None:
            _close_drawer()

        save_button.on_click(_do_save)
        cancel_button.on_click(_do_cancel)

        if "enter_pressed" in name_input.param:
            def _save_on_enter(event: Any) -> None:
                if bool(event.new):
                    _do_save(event)

            name_input.param.watch(_save_on_enter, "enter_pressed")

        _show_drawer(
            _drawer_header("Save Layout As"),
            pn.pane.Markdown(
                "Enter a layout filename. `.json` will be added automatically if missing.",
                sizing_mode="stretch_width",
            ),
            name_input,
            pn.Row(save_button, cancel_button),
            message,
        )

    def _load(_event: Any) -> None:
        if context is None:
            _notify("Load failed: no context.", kind="error", duration=7000)
            return

        options = _layout_options(context)

        saved_select = pn.widgets.Select(
            name="Saved layout",
            options=options,
            value=next(iter(options.values()), None),
            disabled=not bool(options),
            sizing_mode="stretch_width",
        )
        path_input = pn.widgets.TextInput(
            name="Or layout path",
            placeholder=str(default_layout_directory(context) / "layout.json"),
            sizing_mode="stretch_width",
        )
        file_input = pn.widgets.FileInput(
            name="Or upload layout JSON",
            accept=".json,application/json",
            sizing_mode="stretch_width",
        )
        refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="default",
            width=95,
            height=34,
        )
        load_selected_button = pn.widgets.Button(
            name="Load",
            button_type="primary",
            width=95,
            height=34,
        )
        cancel_button = pn.widgets.Button(
            name="Cancel",
            button_type="default",
            width=95,
            height=34,
        )
        message = pn.pane.Markdown("", visible=False, sizing_mode="stretch_width")

        def _refresh(_refresh_event: Any) -> None:
            refreshed = _layout_options(context)
            saved_select.options = refreshed
            saved_select.value = next(iter(refreshed.values()), None)
            saved_select.disabled = not bool(refreshed)

            _notify(
                f"Found {len(refreshed)} saved layout(s).",
                kind="info",
                duration=3000,
            )

        def _do_load(_load_event: Any) -> None:
            load_selected_button.disabled = True
            try:
                uploaded_snapshot = _load_snapshot_from_upload(file_input)
                if uploaded_snapshot is not None:
                    _load_layout_snapshot(
                        context=context,
                        snapshot=uploaded_snapshot,
                        source_label=file_input.filename or "uploaded JSON",
                        notify_success=lambda msg: _notify(
                            msg,
                            kind="success",
                            duration=5000,
                        ),
                        notify_warning=lambda msg: _notify(
                            msg,
                            kind="warning",
                            duration=7000,
                        ),
                    )
                    _flash_button(load_button, "Loaded ✓")
                    _close_drawer()
                    return

                raw_path = (path_input.value or "").strip()
                if raw_path:
                    path = Path(raw_path).expanduser()
                elif saved_select.value:
                    path = Path(saved_select.value).expanduser()
                else:
                    raise ValueError(
                        "Choose a saved layout, enter a path, or upload a JSON file."
                    )

                persistence = getattr(context, "persistence", None)
                if persistence is None:
                    raise RuntimeError("context.persistence is not configured.")

                snapshot = persistence.load(path)
                _load_layout_snapshot(
                    context=context,
                    snapshot=snapshot,
                    source_label=path.name,
                    notify_success=lambda msg: _notify(
                        msg,
                        kind="success",
                        duration=5000,
                    ),
                    notify_warning=lambda msg: _notify(
                        msg,
                        kind="warning",
                        duration=7000,
                    ),
                )
                _flash_button(load_button, "Loaded ✓")
                _close_drawer()
            except Exception as exc:
                message.object = f"Load failed: `{exc}`"
                message.visible = True
                _notify(f"Load layout failed: {exc}", kind="error", duration=8000)
                _flash_button(load_button, "Load failed")
            finally:
                load_selected_button.disabled = False

        def _do_cancel(_cancel_event: Any) -> None:
            _close_drawer()

        refresh_button.on_click(_refresh)
        load_selected_button.on_click(_do_load)
        cancel_button.on_click(_do_cancel)

        _show_drawer(
            _drawer_header("Load Layout"),
            pn.pane.Markdown(
                "Loading reconciles the current workspace: extra panels are closed, "
                "missing panels are opened, matching panels are kept, and the saved "
                "grid geometry is restored.",
                sizing_mode="stretch_width",
            ),
            saved_select,
            path_input,
            file_input,
            pn.Row(refresh_button, load_selected_button, cancel_button),
            message,
        )

    quick_save_button.on_click(_quick_save)
    save_as_button.on_click(_save_as)
    load_button.on_click(_load)

    return pn.Row(
        quick_save_button,
        save_as_button,
        load_button,
        toast_pane,
        drawer_anchor,
        sizing_mode="fixed",
        height=40,
        margin=(0, 16, 0, 0),
        align="center",
        styles={
            "display": "flex",
            "align-items": "center",
            "gap": "8px",
        },
    )