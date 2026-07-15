from __future__ import annotations

from dataclasses import asdict, is_dataclass
import html
from typing import Any, Mapping, Optional

import panel as pn

try:
    from .assets import ImagePreview, get_image_resolver
except ImportError:
    from assets import ImagePreview, get_image_resolver


class ImageSelectionGalleryPanel:
    """
    Fast thumbnail gallery for the active platform selection set.

    This implementation deliberately avoids Panel ReactiveHTML/custom Bokeh
    models. The gallery itself is one pn.pane.HTML object containing a CSS grid,
    so large active-learning batches do not create hundreds of Panel/Bokeh card
    models. Row focus is controlled through a lightweight Select + Button rather
    than per-card click callbacks.
    """

    state_version = 6

    def __init__(
        self,
        context: Any,
        *,
        thumb_size: int = 80,
        max_items: int = 2000,
        show_badges: bool = False,
        max_in_flight: int = 4,
        batch_size: int = 24,
        render_interval_ms: int = 180,
    ) -> None:
        self.context = context
        self.thumb_size = int(thumb_size or 112)
        self.max_items = int(max_items or 240)
        self.max_in_flight = max(1, int(max_in_flight or 4))
        self.batch_size = max(1, int(batch_size or 24))
        self.render_interval_ms = max(40, int(render_interval_ms or 180))

        self._subscriptions: list[Any] = []
        self._job_handles: list[Any] = []
        self._request_seq = 0
        self._render_scheduled = False
        self._reload_scheduled = False

        self._dataset_id: Optional[str] = None
        self._row_ids: list[str] = []
        self._focused_row_id: Optional[str] = None
        self._previews: dict[str, ImagePreview] = {}
        self._preview_cache: dict[tuple[str, str, int], ImagePreview] = {}
        self._errors: dict[str, str] = {}
        self._pending_row_ids: list[str] = []
        self._in_flight: set[str] = set()

        self._title = pn.pane.Markdown("### Image Selection Gallery", sizing_mode="stretch_width")
        self._status = pn.pane.Alert(
            "Select or lasso records to show their images.",
            alert_type="info",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.thumb_size_widget = pn.widgets.IntSlider(
            name="Thumbnail size",
            start=72,
            end=256,
            step=8,
            value=self.thumb_size,
            sizing_mode="stretch_width",
        )
        self.max_items_widget = pn.widgets.IntInput(
            name="Max items",
            value=self.max_items,
            start=1,
            end=5000,
            width=110,
            height=50,
            sizing_mode="fixed",
        )
        self.fit_mode = pn.widgets.Select(
            name="Card fit",
            value="Crop",
            options=["Crop", "Contain"],
            width=110,
            height=34,
            sizing_mode="fixed",
        )
        self.show_badges = pn.widgets.Checkbox(
            name="Show badges",
            value=bool(show_badges),
            width=120,
            height=34,
            sizing_mode="fixed",
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="default",
            width=90,
            height=34,
            sizing_mode="fixed",
        )
        self.focus_select = pn.widgets.Select(
            name="Focus row",
            value=None,
            options=[],
            sizing_mode="stretch_width",
        )
        self.focus_button = pn.widgets.Button(
            name="Focus selected row",
            button_type="primary",
            width=150,
            height=34,
            sizing_mode="fixed",
        )

        self.thumb_size_widget.param.watch(self._on_load_options_changed, "value")
        self.max_items_widget.param.watch(self._on_load_options_changed, "value")
        self.fit_mode.param.watch(lambda event: self._schedule_render(), "value")
        self.show_badges.param.watch(lambda event: self._schedule_render(), "value")
        self.refresh_button.on_click(lambda event: self._load_current_rows(force=True))
        self.focus_button.on_click(lambda event: self._focus_selected_row())

        self._toolbar = pn.Column(
            self.thumb_size_widget,
            pn.Row(
                self.max_items_widget,
                self.fit_mode,
                self.show_badges,
                self.refresh_button,
                sizing_mode="stretch_width",
                height=60,
                styles={"align-items": "center", "gap": "12px", "overflow": "visible"},
            ),
            pn.Row(
                self.focus_select,
                self.focus_button,
                sizing_mode="stretch_width",
                styles={"align-items": "end", "gap": "10px", "overflow": "visible"},
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self._grid = pn.pane.HTML(
            self._empty_html("No active selection set."),
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
        self._scroller = pn.Column(
            self._grid,
            sizing_mode="stretch_both",
            styles={
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "height": "100%",
                "min-height": "240px",
                "padding": "8px",
                "box-sizing": "border-box",
                "min-height": "0",
                "flex": "1 1 0",
                "max-height": "100%",
                "background": "#0f1115",
                "border-radius": "12px",
                "border": "1px solid rgba(255,255,255,0.08)",
            },
        )

        self._view = pn.Column(
            self._title,
            self._status,
            self._toolbar,
            self._scroller,
            sizing_mode="stretch_both",
            styles={
                "height": "100%",
                "width": "100%",
                "display": "flex",
                "flex-direction": "column",
                "min-height": "0",
                "overflow": "hidden",
                "box-sizing": "border-box",
                "padding": "8px",
            },
        )

        self._subscribe()
        self._refresh_from_selection()
        self._schedule(self._refresh_from_selection)

    def panel(self) -> pn.Column:
        return self._view

    def dispose(self) -> None:
        self._cancel_current_load()
        events = getattr(self.context, "events", None)
        unsubscribe = getattr(events, "unsubscribe", None)
        if callable(unsubscribe):
            for sub in self._subscriptions:
                try:
                    unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions.clear()

    def get_state(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "thumb_size": int(self.thumb_size_widget.value),
            "max_items": int(self.max_items_widget.value),
            "fit_mode": self.fit_mode.value,
            "show_badges": bool(self.show_badges.value),
            "max_in_flight": int(self.max_in_flight),
            "batch_size": int(self.batch_size),
            "render_interval_ms": int(self.render_interval_ms),
            "dataset_id": self._dataset_id,
            "row_ids": list(self._row_ids),
            "focused_row_id": self._focused_row_id,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        try:
            self.thumb_size_widget.value = int(state.get("thumb_size", self.thumb_size_widget.value))
            self.max_items_widget.value = int(state.get("max_items", self.max_items_widget.value))
            self.max_in_flight = max(1, int(state.get("max_in_flight", self.max_in_flight)))
            self.batch_size = max(1, int(state.get("batch_size", self.batch_size)))
            self.render_interval_ms = max(40, int(state.get("render_interval_ms", self.render_interval_ms)))
        except Exception:
            pass
        if state.get("fit_mode") in {"Crop", "Contain"}:
            self.fit_mode.value = state["fit_mode"]
        if "show_badges" in state:
            self.show_badges.value = bool(state["show_badges"])

    def _subscribe(self) -> None:
        events = getattr(self.context, "events", None)
        subscribe = getattr(events, "subscribe", None)
        if not callable(subscribe):
            return

        for topic, callback in (
            ("selection.set.changed", self.on_selection_set_changed),
            ("selection.set.cleared", self.on_selection_set_cleared),
            ("selection.focus.changed", self.on_selection_focus_changed),
            ("selection.focus.cleared", self.on_selection_focus_cleared),
            ("dataset.active.changed", self.on_dataset_changed),
            ("dataset.mapping.updated", self.on_dataset_mapping_updated),
            ("mapping.resolved", self.on_dataset_mapping_updated),
        ):
            try:
                sub = subscribe(
                    topic,
                    callback,
                    owner_id="core.image.gallery",
                    owner_label="Image Selection Gallery",
                    owner_kind="panel",
                )
            except TypeError:
                sub = subscribe(topic, callback)
            self._subscriptions.append(sub)

    def on_selection_set_changed(self, topic: str, payload: dict[str, Any]) -> None:
        self._dataset_id = str(payload.get("dataset_id") or "") or None
        self._row_ids = [str(row_id) for row_id in payload.get("row_ids") or []]
        focus_row_id = payload.get("focus_row_id")
        if focus_row_id is not None:
            self._focused_row_id = str(focus_row_id)
        self._load_current_rows()

    def on_selection_set_cleared(self, topic: str, payload: dict[str, Any]) -> None:
        self._cancel_current_load()
        self._dataset_id = None
        self._row_ids = []
        self._focused_row_id = None
        self._previews.clear()
        self._errors.clear()
        self._refresh_focus_options()
        self._grid.object = self._empty_html("Selection cleared.")
        self._status.alert_type = "info"
        self._status.object = "Selection cleared."

    def on_selection_focus_changed(self, topic: str, payload: dict[str, Any]) -> None:
        row_id = payload.get("row_id")
        if row_id is None:
            return
        self._focused_row_id = str(row_id)
        dataset_id = payload.get("dataset_id")
        if dataset_id and self._dataset_id and str(dataset_id) != self._dataset_id:
            self._refresh_from_selection()
            return
        self._refresh_focus_options()
        self._schedule_render()

    def on_selection_focus_cleared(self, topic: str, payload: dict[str, Any]) -> None:
        self._focused_row_id = None
        self._refresh_focus_options()
        self._schedule_render()

    def on_dataset_changed(self, topic: str, payload: dict[str, Any]) -> None:
        self._refresh_from_selection()

    def on_dataset_mapping_updated(self, topic: str, payload: dict[str, Any]) -> None:
        dataset_id = str(payload.get("dataset_id") or payload.get("dataset") or "")
        if dataset_id and self._dataset_id and dataset_id != self._dataset_id:
            return
        self._load_current_rows(force=True)

    def _refresh_from_selection(self) -> None:
        focus = self._get_current_focus()
        active_set = self._get_active_set()
        if focus:
            self._focused_row_id = str(focus.get("row_id"))

        if active_set:
            self._dataset_id = str(active_set.get("dataset_id") or "") or None
            self._row_ids = [str(row_id) for row_id in active_set.get("row_ids") or []]
        elif focus:
            self._dataset_id = str(focus.get("dataset_id") or "") or None
            self._row_ids = [str(focus.get("row_id"))] if focus.get("row_id") is not None else []
        else:
            self._dataset_id = None
            self._row_ids = []

        self._load_current_rows()

    def _load_current_rows(self, *, force: bool = False) -> None:
        self.thumb_size = int(self.thumb_size_widget.value or self.thumb_size)
        self.max_items = int(self.max_items_widget.value or self.max_items)
        self._cancel_current_load()
        self._request_seq += 1
        request_seq = self._request_seq

        if not self._dataset_id or not self._row_ids:
            self._previews.clear()
            self._errors.clear()
            self._refresh_focus_options()
            self._grid.object = self._empty_html("No active selection set. Select or lasso records to populate the gallery.")
            self._status.alert_type = "info"
            self._status.object = "No active selection set. Select or lasso records to populate the gallery."
            return

        visible_row_ids = self._visible_row_ids()
        visible_set = set(visible_row_ids)
        cache_keys = [(self._dataset_id, row_id, self.thumb_size) for row_id in visible_row_ids]

        if force:
            for key in cache_keys:
                self._preview_cache.pop(key, None)

        self._previews = {
            row_id: self._preview_cache[(self._dataset_id, row_id, self.thumb_size)]
            for row_id in visible_row_ids
            if (self._dataset_id, row_id, self.thumb_size) in self._preview_cache
        }
        self._errors = {
            row_id: error
            for row_id, error in self._errors.items()
            if row_id in visible_set and row_id not in self._previews
        }
        self._pending_row_ids = [row_id for row_id in visible_row_ids if row_id not in self._previews]

        omitted = max(0, len(self._row_ids) - len(visible_row_ids))
        cached = len(self._previews)
        total = len(visible_row_ids)
        self._status.alert_type = "primary" if self._pending_row_ids else "success"
        self._status.object = f"Loaded {cached}/{total} thumbnails"
        if self._pending_row_ids:
            self._status.object += f"; loading {len(self._pending_row_ids)}"
        if omitted:
            self._status.object += f"; {omitted} omitted"
        self._status.object += "."

        self._refresh_focus_options()
        self._render_grid()
        self._launch_more(request_seq)

    def _visible_row_ids(self) -> list[str]:
        return [str(row_id) for row_id in self._row_ids[: max(1, int(self.max_items))]]

    def _launch_more(self, request_seq: int) -> None:
        if request_seq != self._request_seq:
            return

        jobs = getattr(self.context, "jobs", None)
        resolver = get_image_resolver(self.context)
        batch_loader = getattr(resolver, "load_previews_for_rows", None)

        while self._pending_row_ids and len(self._in_flight) < self.max_in_flight:
            batch: list[str] = []
            while self._pending_row_ids and len(batch) < self.batch_size:
                row_id = self._pending_row_ids.pop(0)
                if row_id in self._previews:
                    continue
                batch.append(row_id)

            if not batch:
                continue

            for row_id in batch:
                self._in_flight.add(row_id)

            if jobs is None:
                try:
                    result = self._load_batch_sync(resolver, batch, batch_loader=batch_loader)
                    self._on_preview_batch_loaded(batch, result, request_seq)
                except Exception as exc:
                    self._on_preview_batch_error(batch, exc, request_seq)
                continue

            if callable(batch_loader):
                first_row = batch[0] if batch else "empty"
                handle = None

                def done(result: Any, batch=batch, handle_getter=lambda: handle) -> None:
                    current_handle = handle_getter()
                    if current_handle is not None:
                        self._discard_job_handle(current_handle)
                    self._on_preview_batch_loaded(batch, result, request_seq)

                def failed(exc: BaseException, batch=batch, handle_getter=lambda: handle) -> None:
                    current_handle = handle_getter()
                    if current_handle is not None:
                        self._discard_job_handle(current_handle)
                    self._on_preview_batch_error(batch, exc, request_seq)

                handle = jobs.submit(
                    batch_loader,
                    title=f"Load {len(batch)} image thumbnails",
                    key=(
                        f"core.image.thumbs:{self._dataset_id}:{self.thumb_size}:"
                        f"{request_seq}:{first_row}:{len(batch)}"
                    ),
                    on_done=done,
                    on_error=failed,
                    dataset_id=self._dataset_id,
                    row_ids=batch,
                    role="thumbnail",
                    max_size=self.thumb_size,
                    prefer_thumbnail=True,
                )
                self._job_handles.append(handle)
            else:
                for row_id in list(batch):
                    handle = None

                    def done(preview: ImagePreview, row_id=row_id, handle_getter=lambda: handle) -> None:
                        current_handle = handle_getter()
                        if current_handle is not None:
                            self._discard_job_handle(current_handle)
                        self._on_preview_batch_loaded([row_id], {"previews": {row_id: preview}, "errors": {}}, request_seq)

                    def failed(exc: BaseException, row_id=row_id, handle_getter=lambda: handle) -> None:
                        current_handle = handle_getter()
                        if current_handle is not None:
                            self._discard_job_handle(current_handle)
                        self._on_preview_batch_error([row_id], exc, request_seq)

                    handle = jobs.submit(
                        resolver.load_preview_for_row,
                        title="Load image thumbnail",
                        key=f"core.image.thumb:{self._dataset_id}:{row_id}:{self.thumb_size}:{request_seq}",
                        on_done=done,
                        on_error=failed,
                        dataset_id=self._dataset_id,
                        row_id=row_id,
                        role="thumbnail",
                        max_size=self.thumb_size,
                        prefer_thumbnail=True,
                    )
                    self._job_handles.append(handle)

    def _load_batch_sync(self, resolver: Any, batch: list[str], *, batch_loader: Any) -> dict[str, Any]:
        if callable(batch_loader):
            return batch_loader(
                self._dataset_id,
                batch,
                role="thumbnail",
                max_size=self.thumb_size,
                prefer_thumbnail=True,
            )

        previews: dict[str, ImagePreview] = {}
        errors: dict[str, str] = {}
        for row_id in batch:
            try:
                previews[row_id] = resolver.load_preview_for_row(
                    self._dataset_id,
                    row_id,
                    role="thumbnail",
                    max_size=self.thumb_size,
                    prefer_thumbnail=True,
                )
            except Exception as exc:
                errors[row_id] = str(exc)
        return {"previews": previews, "errors": errors}

    def _on_preview_batch_loaded(self, row_ids: list[str], result: Any, request_seq: int) -> None:
        if request_seq != self._request_seq:
            return

        row_ids = [str(row_id) for row_id in row_ids]
        for row_id in row_ids:
            self._in_flight.discard(row_id)

        payload = result if isinstance(result, Mapping) else {}
        previews = payload.get("previews") if isinstance(payload.get("previews"), Mapping) else {}
        errors = payload.get("errors") if isinstance(payload.get("errors"), Mapping) else {}

        for row_id, preview in previews.items():
            row_id = str(row_id)
            if preview is None:
                continue
            self._previews[row_id] = preview
            if self._dataset_id:
                self._preview_cache[(self._dataset_id, row_id, self.thumb_size)] = preview
            self._errors.pop(row_id, None)

        for row_id, error in errors.items():
            row_id = str(row_id)
            if row_id not in self._previews:
                self._errors[row_id] = str(error)

        for row_id in row_ids:
            if row_id not in self._previews and row_id not in self._errors:
                self._errors[row_id] = "No preview returned."

        self._update_progress_status()
        self._schedule_render()
        self._launch_more(request_seq)

    def _on_preview_batch_error(self, row_ids: list[str], exc: BaseException, request_seq: int) -> None:
        if request_seq != self._request_seq:
            return

        message = str(exc)
        for row_id in row_ids:
            row_id = str(row_id)
            self._in_flight.discard(row_id)
            if row_id not in self._previews:
                self._errors[row_id] = message

        self._update_progress_status()
        self._schedule_render()
        self._launch_more(request_seq)

    def _discard_job_handle(self, handle: Any) -> None:
        try:
            self._job_handles.remove(handle)
        except ValueError:
            pass

    def _update_progress_status(self) -> None:
        total = len(self._visible_row_ids())
        if total <= 0:
            return
        visible = self._visible_row_ids()
        loaded = len([row_id for row_id in visible if row_id in self._previews])
        failed = len([row_id for row_id in visible if row_id in self._errors])
        remaining = max(0, total - loaded - failed)
        omitted = max(0, len(self._row_ids) - total)

        if remaining:
            self._status.alert_type = "primary"
            self._status.object = f"Loaded {loaded}/{total} thumbnails; {remaining} loading"
        else:
            self._status.alert_type = "success" if failed == 0 else "warning"
            self._status.object = f"Loaded {loaded}/{total} thumbnails"
            if failed:
                self._status.object += f"; {failed} failed"
        if omitted:
            self._status.object += f"; {omitted} omitted"
        self._status.object += "."

    def _render_grid(self) -> None:
        if not self._row_ids:
            self._grid.object = self._empty_html("No active selection set.")
            return

        visible = self._visible_row_ids()
        css = self._css()
        cards = "".join(self._card_html(row_id) for row_id in visible)
        omitted = max(0, len(self._row_ids) - len(visible))
        omitted_html = ""
        if omitted:
            omitted_html = (
                f"<div class='aical-gallery-omitted'>"
                f"Showing {len(visible):,} of {len(self._row_ids):,} selected rows; "
                f"{omitted:,} omitted by Max items."
                f"</div>"
            )
        self._grid.object = f"{css}<div class='aical-gallery-grid'>{cards}</div>{omitted_html}"

    def _css(self) -> str:
        size = max(48, int(self.thumb_size))
        fit = "cover" if self.fit_mode.value == "Crop" else "contain"
        return f"""
<style>
.aical-gallery-grid {{
  --thumb-size: {size}px;
  --image-fit: {fit};
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(calc(var(--thumb-size) + 30px), 1fr));
  gap: 14px;
  align-items: start;
  padding: 4px 4px 18px 4px;
  box-sizing: border-box;
  width: 100%;
}}
.aical-gallery-card {{
  position: relative;
  min-width: 0;
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.035));
  box-shadow: 0 12px 30px rgba(0,0,0,0.24);
  overflow: hidden;
}}
.aical-gallery-card.is-focused {{
  border-color: rgba(86,166,255,0.95);
  box-shadow: 0 0 0 2px rgba(86,166,255,0.42), 0 18px 34px rgba(0,0,0,0.32);
}}
.aical-gallery-card.is-error {{
  border-color: rgba(255,107,107,0.75);
}}
.aical-gallery-image-frame {{
  width: 100%;
  aspect-ratio: 1 / 1;
  background:
    radial-gradient(circle at 30% 20%, rgba(255,255,255,0.13), transparent 30%),
    linear-gradient(135deg, #1f2631, #11141b 55%, #080a0f);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}}
.aical-gallery-image-frame img {{
  width: 100%;
  height: 100%;
  object-fit: var(--image-fit);
  display: block;
}}
.aical-gallery-placeholder {{
  width: 46%;
  height: 46%;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(255,255,255,0.08), rgba(255,255,255,0.20), rgba(255,255,255,0.08));
  animation: aical-gallery-pulse 1.2s ease-in-out infinite;
}}
@keyframes aical-gallery-pulse {{
  0%, 100% {{ opacity: 0.45; transform: scale(0.96); }}
  50% {{ opacity: 0.95; transform: scale(1.03); }}
}}
.aical-gallery-error {{
  color: #ffb4b4;
  font-size: 12px;
  line-height: 1.3;
  text-align: center;
  padding: 12px;
  word-break: break-word;
}}
.aical-gallery-meta {{
  padding: 9px 10px 10px 10px;
  box-sizing: border-box;
}}
.aical-gallery-row-id {{
  color: rgba(255,255,255,0.88);
  font-size: 12px;
  font-weight: 650;
  line-height: 1.25;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.aical-gallery-subtitle {{
  color: rgba(255,255,255,0.50);
  font-size: 10.5px;
  line-height: 1.25;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-top: 3px;
}}
.aical-gallery-badges {{
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 8px;
  max-height: 48px;
  overflow: hidden;
}}
.aical-gallery-badge {{
  max-width: 100%;
  border-radius: 999px;
  padding: 2px 7px;
  background: rgba(255,255,255,0.10);
  color: rgba(255,255,255,0.78);
  font-size: 10px;
  line-height: 1.35;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.aical-gallery-focus-pill {{
  position: absolute;
  top: 8px;
  right: 8px;
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 10px;
  font-weight: 700;
  color: white;
  background: rgba(38,132,255,0.92);
  box-shadow: 0 4px 16px rgba(0,0,0,0.24);
}}
.aical-gallery-omitted {{
  margin: 4px 4px 18px 4px;
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.66);
  font-size: 12px;
}}
</style>
"""

    def _empty_html(self, message: str) -> str:
        safe_message = html.escape(str(message))
        return f"""
{self._css()}
<div class='aical-gallery-omitted'>{safe_message}</div>
"""

    def _card_html(self, row_id: str) -> str:
        focused = self._focused_row_id is not None and row_id == str(self._focused_row_id)
        preview = self._previews.get(row_id)
        error = self._errors.get(row_id)
        safe_row = html.escape(str(row_id), quote=True)
        card_classes = ["aical-gallery-card"]
        if focused:
            card_classes.append("is-focused")
        if error:
            card_classes.append("is-error")

        focus_html = "<div class='aical-gallery-focus-pill'>Focused</div>" if focused else ""
        subtitle = "Loading thumbnail…"
        image_html = "<div class='aical-gallery-placeholder'></div>"
        badges = ""
        title = safe_row

        if preview is not None:
            metadata = preview.asset.metadata or {}
            title = html.escape(self._title_for_preview(preview), quote=True)
            src = html.escape(preview.data_uri, quote=True)
            alt = html.escape(str(preview.asset.row_id), quote=True)
            subtitle = self._subtitle_for_metadata(metadata)
            image_html = f"<img src='{src}' alt='{alt}' loading='lazy' decoding='async'>"
            badges = self._badges_html(metadata) if bool(self.show_badges.value) else ""
        elif error:
            subtitle = "Preview failed"
            safe_error = html.escape(str(error), quote=True)
            title = f"{safe_row}\n{safe_error}"
            image_html = f"<div class='aical-gallery-error'>Failed<br>{safe_row}</div>"

        return f"""
<div class='{' '.join(card_classes)}' title='{title}'>
  {focus_html}
  <div class='aical-gallery-image-frame'>{image_html}</div>
  <div class='aical-gallery-meta'>
    <div class='aical-gallery-row-id'>{safe_row}</div>
    <div class='aical-gallery-subtitle'>{html.escape(subtitle)}</div>
    {badges}
  </div>
</div>
"""

    def _badges_html(self, metadata: Mapping[str, Any]) -> str:
        parts: list[str] = []
        for label, key in (
            ("Label", "target_label"),
            ("Pred", "prediction"),
            ("P", "probability"),
            ("Unc", "uncertainty"),
            ("State", "label_state"),
        ):
            value = str(metadata.get(key) or "").strip()
            if value:
                parts.append(
                    "<span class='aical-gallery-badge'>"
                    f"{html.escape(label)}: {html.escape(value)}"
                    "</span>"
                )
        return f"<div class='aical-gallery-badges'>{''.join(parts)}</div>" if parts else ""

    def _subtitle_for_metadata(self, metadata: Mapping[str, Any]) -> str:
        for key in ("target_label", "prediction", "label_state", "filename"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
        return "Loaded"

    def _title_for_preview(self, preview: ImagePreview) -> str:
        metadata = preview.asset.metadata or {}
        pieces = [f"Record: {preview.asset.row_id}"]
        for label, key in (
            ("Label", "target_label"),
            ("Prediction", "prediction"),
            ("Probability", "probability"),
            ("Uncertainty", "uncertainty"),
            ("State", "label_state"),
        ):
            value = str(metadata.get(key) or "").strip()
            if value:
                pieces.append(f"{label}: {value}")
        return "\n".join(pieces)

    def _refresh_focus_options(self) -> None:
        visible = self._visible_row_ids() if self._row_ids else []
        current = self._focused_row_id if self._focused_row_id in visible else None
        try:
            self.focus_select.options = visible
            if current is not None:
                self.focus_select.value = current
            elif visible and self.focus_select.value not in visible:
                self.focus_select.value = visible[0]
            elif not visible:
                self.focus_select.value = None
        except Exception:
            pass

    def _focus_selected_row(self) -> None:
        row_id = self.focus_select.value
        if row_id is not None:
            self._focus_row(str(row_id))

    def _focus_row(self, row_id: str) -> None:
        if not self._dataset_id:
            return
        selection = getattr(self.context, "selection", None)
        if selection is not None:
            try:
                selection.set_focus(
                    dataset_id=self._dataset_id,
                    row_id=str(row_id),
                    origin="core.image.gallery",
                    panel_id="core.image.gallery",
                )
                return
            except Exception:
                pass
        self._focused_row_id = str(row_id)
        self._refresh_focus_options()
        self._schedule_render()

    def _cancel_current_load(self) -> None:
        self._request_seq += 1
        self._pending_row_ids = []
        self._in_flight.clear()
        for handle in self._job_handles:
            try:
                handle.cancel()
            except Exception:
                pass
        self._job_handles.clear()

    def _on_load_options_changed(self, event: Any) -> None:
        self.thumb_size = int(self.thumb_size_widget.value or self.thumb_size)
        self.max_items = int(self.max_items_widget.value or self.max_items)
        self._cancel_current_load()
        self._status.alert_type = "primary"
        self._status.object = "Updating gallery…"
        self._schedule_reload()

    def _schedule_reload(self) -> None:
        if self._reload_scheduled:
            return
        self._reload_scheduled = True

        def run() -> None:
            self._reload_scheduled = False
            self._load_current_rows()

        self._schedule(run, delay_ms=120)

    def _schedule_render(self) -> None:
        if self._render_scheduled:
            return
        self._render_scheduled = True

        def run() -> None:
            self._render_scheduled = False
            self._render_grid()

        self._schedule(run, delay_ms=int(self.render_interval_ms))

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
        return focus

    def _get_active_set(self) -> Optional[dict[str, Any]]:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None
        try:
            active_set = self._state_to_dict(selection.get_active_set())
        except Exception:
            return None
        if not active_set or not active_set.get("dataset_id"):
            return None
        return active_set

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
    def _schedule(fn: Any, *, delay_ms: int = 0) -> None:
        try:
            doc = pn.state.curdoc
        except Exception:
            doc = None
        if doc is None:
            try:
                fn()
            except Exception:
                pass
            return
        if delay_ms:
            try:
                doc.add_timeout_callback(fn, delay_ms)
                return
            except Exception:
                pass
        try:
            doc.add_next_tick_callback(fn)
        except Exception:
            try:
                fn()
            except Exception:
                pass