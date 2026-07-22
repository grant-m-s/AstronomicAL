from __future__ import annotations

from dataclasses import asdict, is_dataclass
import html
import json
from typing import Any, Optional

import math
import time

import panel as pn

try:
    from .assets import ImagePreview, get_image_resolver
except ImportError:
    from assets import ImagePreview, get_image_resolver

FIT_MODES = {
    "Fit inside": "contain",
    "Fill and crop": "cover",
    "Actual size": "none",
}


class ImageViewerPanel:
    """
    Selection-aware single-image viewer.

    The viewer is intentionally focused on one job: render the currently focused
    record's image as large as possible. Record navigation belongs to the record
    browser or other selection-owning panels.
    """

    state_version = 3

    def __init__(
        self,
        context: Any,
        *,
        max_size: int = 1024,
        image_height: int = 360,
    ) -> None:
        self.context = context
        self.max_size = int(max_size or 2048)
        self.image_height = int(image_height or 360)

        self._subscriptions: list[Any] = []
        self._job_handles: list[Any] = []
        self._current_dataset_id: Optional[str] = None
        self._current_row_id: Optional[str] = None
        self._last_preview: Optional[ImagePreview] = None
        self._request_seq = 0

        self._load_delay_ms = 360
        self._load_scheduled = False
        self._load_due_at = 0.0
        self._pending_load: Optional[tuple[str, str, int]] = None

        self._title = pn.pane.Markdown("### Image Viewer", sizing_mode="stretch_width")
        self._status = pn.pane.Alert(
            "Select a record with an image mapping to preview it.",
            alert_type="info",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self._image = pn.pane.HTML(
            self._empty_image_html("No image selected"),
            sizing_mode="stretch_width",
            height=self.image_height,
            styles={
                "width": "100%",
                "height": f"{self.image_height}px",
                "min-height": f"{min(self.image_height, 260)}px",
                "overflow": "hidden",
                "background": "#111",
                "box-sizing": "border-box",
            },
            margin=(0, 0, 8, 0),
        )

        self.fit_mode = pn.widgets.Select(
            name="Fit mode",
            value="Fit inside",
            options=list(FIT_MODES.keys()),
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )
        self.fit_mode.param.watch(lambda event: self._rerender_last_preview(), "value")

        self.refresh_button = pn.widgets.Button(
            name="Refresh image",
            button_type="default",
            sizing_mode="stretch_width",
            margin=(0, 0, 6, 0),
        )
        self.refresh_button.on_click(lambda event: self._refresh_from_current_focus())

        self._meta = pn.pane.Markdown("", sizing_mode="stretch_width", margin=(0, 0, 6, 0))
        self._uri_tools = pn.pane.HTML("", sizing_mode="stretch_width", margin=(0, 0, 0, 0))

        self._controls = pn.Column(
            self.fit_mode,
            self.refresh_button,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self._view = pn.Column(
            self._title,
            self._status,
            self._image,
            self._controls,
            self._meta,
            self._uri_tools,
            sizing_mode="stretch_width",
            styles={
                "height": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "padding": "8px",
            },
        )

        self._subscribe()
        self._refresh_from_current_focus()
        self._schedule(self._refresh_from_current_focus)

    def panel(self) -> pn.Column:
        return self._view

    def dispose(self) -> None:
        self._request_seq += 1
        self._pending_load = None
        self._cancel_image_jobs()

        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in self._subscriptions:
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions.clear()

    def get_state(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "max_size": self.max_size,
            "image_height": self.image_height,
            "dataset_id": self._current_dataset_id,
            "row_id": self._current_row_id,
            "fit_mode": self.fit_mode.value,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        try:
            self.max_size = int(state.get("max_size", self.max_size))
            self.image_height = int(state.get("image_height", self.image_height))
        except Exception:
            pass
        if state.get("fit_mode") in FIT_MODES:
            self.fit_mode.value = state["fit_mode"]

    def _subscribe(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        for topic, callback in (
            ("selection.focus.changed", self.on_selection_focus_changed),
            ("selection.focus.cleared", self.on_selection_focus_cleared),
            ("dataset.active.changed", self.on_dataset_active_changed),
            ("dataset.mapping.updated", self.on_dataset_mapping_updated),
            ("mapping.resolved", self.on_dataset_mapping_updated),
        ):
            self._subscriptions.append(events.subscribe(topic, callback))

    def _start_pending_load(self) -> None:
        remaining = self._load_due_at - time.monotonic()
        if remaining > 0:
            self._schedule(
                self._start_pending_load,
                delay_ms=max(1, math.ceil(remaining * 1000)),
            )
            return

        self._load_scheduled = False
        request = self._pending_load
        self._pending_load = None
        if request is None:
            return

        dataset_id, row_id, request_seq = request
        self._cancel_image_jobs()
        self._submit_image_load(dataset_id, row_id, request_seq)

    def _submit_image_load(
        self,
        dataset_id: str,
        row_id: str,
        request_seq: int,
    ) -> None:
        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            self._set_error("The platform job service is unavailable.")
            return

        resolver = get_image_resolver(self.context)
        self._set_loading(f"Loading image for `{html.escape(row_id)}`…")

        handle = None

        def done(preview: ImagePreview) -> None:
            self._discard_job_handle(handle)
            self._render_preview_if_current(preview, request_seq)

        def failed(exc: BaseException) -> None:
            self._discard_job_handle(handle)
            if request_seq == self._request_seq:
                self._set_error(str(exc))

        handle = jobs.submit(
            resolver.load_preview_for_row,
            title="Load image preview",
            key=f"core.image.viewer:{id(self)}:{request_seq}",
            on_done=done,
            on_error=failed,
            dataset_id=dataset_id,
            row_id=row_id,
            max_size=self.max_size,
            prefer_thumbnail=False,
        )
        self._job_handles.append(handle)

    def _cancel_image_jobs(self) -> None:
        for handle in list(self._job_handles):
            try:
                handle.cancel()
            except Exception:
                pass
        self._job_handles.clear()

    def _discard_job_handle(self, handle: Any) -> None:
        if handle is None:
            return
        try:
            self._job_handles.remove(handle)
        except ValueError:
            pass

    def on_selection_focus_changed(self, topic: str, payload: dict[str, Any]) -> None:
        dataset_id = payload.get("dataset_id")
        row_id = payload.get("row_id")
        if not dataset_id or row_id is None:
            return

        self._current_dataset_id = str(dataset_id)
        self._current_row_id = str(row_id)
        self._last_preview = None
        self._request_seq += 1
        self._cancel_image_jobs()

        self._pending_load = (
            self._current_dataset_id,
            self._current_row_id,
            self._request_seq,
        )
        self._load_due_at = time.monotonic() + self._load_delay_ms / 1000

        if not self._load_scheduled:
            self._load_scheduled = True
            self._schedule(self._start_pending_load, delay_ms=self._load_delay_ms)

    def on_selection_focus_cleared(self, topic: str, payload: dict[str, Any]) -> None:
        self._request_seq += 1
        self._pending_load = None
        self._cancel_image_jobs()
        self._current_dataset_id = None
        self._current_row_id = None
        self._last_preview = None
        self._image.object = self._empty_image_html("No image selected")
        self._meta.object = ""
        self._uri_tools.object = ""
        self._set_info("Selection cleared.")

    def on_dataset_active_changed(self, topic: str, payload: dict[str, Any]) -> None:
        self._refresh_from_current_focus(
            reason="Active dataset changed. Select a record to load its image."
        )

    def on_dataset_mapping_updated(self, topic: str, payload: dict[str, Any]) -> None:
        focus = self._get_current_focus()
        if not focus:
            self._set_info("Image mapping updated. Select a record to load its image.")
            return

        focus_dataset_id = str(focus.get("dataset_id", ""))
        event_dataset_id = str(payload.get("dataset_id", "") or payload.get("dataset", ""))
        if event_dataset_id and focus_dataset_id and event_dataset_id != focus_dataset_id:
            return

        self._set_loading("Image mapping updated. Reloading focused image…")
        self.on_selection_focus_changed("selection.focus.changed", focus)

    def _render_preview_if_current(self, preview: ImagePreview, request_seq: int) -> None:
        if request_seq != self._request_seq:
            return
        if (
            str(preview.asset.dataset_id) != str(self._current_dataset_id)
            or str(preview.asset.row_id) != str(self._current_row_id)
        ):
            return
        self._last_preview = preview
        self._render_preview(preview)

    def _render_preview(self, preview: ImagePreview) -> None:
        self._status.alert_type = "success"
        self._status.object = (
            f"Loaded `{html.escape(preview.asset.row_id)}` "
            f"({preview.original_width}×{preview.original_height})."
        )

        alt = html.escape(preview.asset.row_id, quote=True)
        fit_css = FIT_MODES.get(self.fit_mode.value, "contain")
        if fit_css == "none":
            img_style = "max-width:none; max-height:none; width:auto; height:auto; object-fit:contain; display:block;"
            outer_overflow = "auto"
        else:
            img_style = f"width:100%; height:100%; object-fit:{fit_css}; display:block;"
            outer_overflow = "hidden"

        self._image.object = f"""
        <div style="width:100%; height:100%; min-height:0; display:flex; align-items:center; justify-content:center; overflow:{outer_overflow}; background:#111;">
          <img src="{preview.data_uri}" alt="{alt}" style="{img_style}" />
        </div>
        """

        self._meta.object = self._metadata_markdown(preview)
        self._uri_tools.object = self._uri_tools_html(preview)
        self._publish_preview_artifact(preview)

    def _metadata_markdown(self, preview: ImagePreview) -> str:
        metadata = preview.asset.metadata or {}
        lines = [
            f"**Dataset:** `{preview.asset.dataset_id}`  ",
            f"**Record:** `{preview.asset.row_id}`  ",
        ]

        label = str(metadata.get("target_label") or "").strip()
        label_column = str(metadata.get("target_label_column") or "").strip()
        if label:
            line = f"**Label:** `{html.escape(label)}`"
            if label_column:
                line += f" from `{html.escape(label_column)}`"
            lines.append(line + "  ")

        prediction = str(metadata.get("prediction") or "").strip()
        probability = str(metadata.get("probability") or "").strip()
        uncertainty = str(metadata.get("uncertainty") or "").strip()
        label_state = str(metadata.get("label_state") or "").strip()
        if prediction:
            line = f"**Prediction:** `{html.escape(prediction)}`"
            if probability:
                line += f" · probability `{html.escape(probability)}`"
            lines.append(line + "  ")
        if uncertainty:
            lines.append(f"**Uncertainty:** `{html.escape(uncertainty)}`  ")
        if label_state:
            lines.append(f"**State:** `{html.escape(label_state)}`  ")

        lines.extend(
            [
                f"**Preview:** {preview.width}×{preview.height} `{preview.media_type}`  ",
                f"**Original:** {preview.original_width}×{preview.original_height}  ",
                f"**Source:** `{html.escape(str(metadata.get('source_semantic') or ''))}` / `{html.escape(str(metadata.get('source_column') or ''))}`  ",
                f"**Cache:** `{html.escape(preview.cache_path)}`",
            ]
        )
        return "\n".join(lines)

    def _uri_tools_html(self, preview: ImagePreview) -> str:
        uri = str(preview.asset.uri or "")
        safe_uri = html.escape(uri, quote=True)
        js_uri = json.dumps(uri)
        return f"""
        <div style="display:block; width:100%; box-sizing:border-box; line-height:1.8;">
          <a href="{safe_uri}" target="_blank" rel="noopener noreferrer">Open image URI</a><br>
          <button type="button" onclick='navigator.clipboard && navigator.clipboard.writeText({js_uri})'>Copy URI</button>
          <span style="display:block; width:100%; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#777;" title="{safe_uri}">{safe_uri}</span>
        </div>
        """

    def _publish_preview_artifact(self, preview: ImagePreview) -> None:
        artifacts = getattr(self.context, "artifacts", None)
        events = getattr(self.context, "events", None)
        if artifacts is None:
            return
        try:
            artifact_id = artifacts.put(
                "image.preview",
                preview.to_dict(),
                dataset_id=preview.asset.dataset_id,
                row_ids=[preview.asset.row_id],
                params={"max_size": self.max_size, "fit_mode": self.fit_mode.value},
            )
            if events is not None:
                events.publish(
                    "artifact.created",
                    {
                        "artifact_id": artifact_id,
                        "type": "image.preview",
                        "dataset_id": preview.asset.dataset_id,
                        "row_ids": [preview.asset.row_id],
                        "origin": "core.image",
                    },
                )
                events.publish(
                    "image.preview",
                    {
                        "artifact_id": artifact_id,
                        "dataset_id": preview.asset.dataset_id,
                        "row_id": preview.asset.row_id,
                        "origin": "core.image.viewer",
                    },
                )
        except Exception:
            pass

    def _refresh_from_current_focus(self, *, reason: str = "") -> None:
        focus = self._get_current_focus()
        if focus:
            self.on_selection_focus_changed("selection.focus.changed", focus)
            return
        self._set_info(reason or "Select a record with an image mapping to preview it.")

    def _rerender_last_preview(self) -> None:
        if self._last_preview is not None:
            self._render_preview(self._last_preview)

    def _get_current_focus(self) -> Optional[dict[str, Any]]:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None
        try:
            focus = self._state_to_dict(selection.get_focus())
        except Exception:
            return None
        if not focus or not focus.get("dataset_id") or focus.get("row_id") is None:
            return None
        return {
            "dataset_id": str(focus.get("dataset_id")),
            "row_id": str(focus.get("row_id")),
            "origin": focus.get("origin"),
            "panel_id": focus.get("panel_id"),
            "metadata": dict(focus.get("metadata") or {}),
        }

    @staticmethod
    def _state_to_dict(value: Any) -> Optional[dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, dict):
            return dict(value)
        if is_dataclass(value):
            return asdict(value)
        result = {}
        for key in (
            "dataset_id",
            "row_id",
            "row_ids",
            "selection_set_id",
            "origin",
            "panel_id",
            "mode",
            "artifact_id",
            "metadata",
            "timestamp",
        ):
            if hasattr(value, key):
                result[key] = getattr(value, key)
        return result or None

    @staticmethod
    def _empty_image_html(message: str) -> str:
        safe = html.escape(message)
        return f"""
        <div style="width:100%; height:100%; display:flex; align-items:center; justify-content:center; background:#111; color:#aaa; text-align:center; padding:12px; box-sizing:border-box;">
          {safe}
        </div>
        """

    @staticmethod
    def _schedule(fn: Any, *, delay_ms: int = 0) -> None:
        try:
            doc = pn.state.curdoc
        except Exception:
            doc = None

        if doc is None:
            if delay_ms <= 0:
                try:
                    fn()
                except Exception:
                    pass
            return

        try:
            if delay_ms > 0:
                doc.add_timeout_callback(fn, int(delay_ms))
            else:
                doc.add_next_tick_callback(fn)
        except Exception:
            if delay_ms <= 0:
                try:
                    fn()
                except Exception:
                    pass

    def _set_info(self, message: str) -> None:
        self._status.alert_type = "info"
        self._status.object = message

    def _set_loading(self, message: str) -> None:
        self._status.alert_type = "primary"
        self._status.object = message

    def _set_error(self, message: str) -> None:
        self._status.alert_type = "danger"
        self._status.object = f"Image preview failed: {message}"
        self._image.object = self._empty_image_html("Image preview failed")
        self._meta.object = ""
        self._uri_tools.object = ""