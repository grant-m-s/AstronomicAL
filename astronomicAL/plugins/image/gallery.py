
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import html
from typing import Any, Optional

import panel as pn

try:
    from .assets import ImagePreview, get_image_resolver
except ImportError:
    from assets import ImagePreview, get_image_resolver


class ImageSelectionGalleryPanel:
    """
    Scrollable thumbnail gallery for the active platform selection set.

    The gallery is tuned for compact square-ish panels: controls wrap, cards have
    fixed outer dimensions, rows do not overlap, and thumbnail loading is
    progressive/cancellable so top rows become usable first.
    """

    state_version = 4

    def __init__(
        self,
        context: Any,
        *,
        thumb_size: int = 112,
        max_items: int = 500,
        show_badges: bool = False,
        max_in_flight: int = 8,
    ) -> None:
        self.context = context
        self.thumb_size = int(thumb_size or 112)
        self.max_items = int(max_items or 500)
        self.max_in_flight = max(1, int(max_in_flight or 8))

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
            start=80,
            end=256,
            step=16,
            value=self.thumb_size,
            sizing_mode="stretch_width",
        )

        self.max_items_widget = pn.widgets.IntInput(
            name="Max items", value=self.max_items, start=1, end=5000,
            width=110, height=50, sizing_mode="fixed",
        )
        self.fit_mode = pn.widgets.Select(
            name="Card fit", value="Crop", options=["Crop", "Contain"],
            width=110, height=34, sizing_mode="fixed",
        )
        self.show_badges = pn.widgets.Checkbox(
            name="Show badges", value=bool(show_badges),
            width=120, height=34, sizing_mode="fixed",
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh", button_type="default",
            width=90, height=34, sizing_mode="fixed",
        )

        self.thumb_size_widget.param.watch(self._on_load_options_changed, "value")
        self.max_items_widget.param.watch(self._on_load_options_changed, "value")
        self.fit_mode.param.watch(lambda event: self._schedule_render(), "value")
        self.show_badges.param.watch(lambda event: self._schedule_render(), "value")
        self.refresh_button.on_click(lambda event: self._load_current_rows(force=True))

        self._toolbar = pn.Column(
            self.thumb_size_widget,
            pn.Row(
                self.max_items_widget,
                self.fit_mode,
                self.show_badges,
                self.refresh_button,
                pn.Spacer(height=5,sizing_mode='fixed'),
                sizing_mode="stretch_width",
                height=60,
                styles={
                    "align-items": "center",
                    "gap": "12px",
                    "overflow": "visible",
                },
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self._grid = pn.FlexBox(
            sizing_mode="stretch_width",
            styles={
                "gap": "10px",
                "row-gap": "10px",
                "align-content": "flex-start",
                "align-items": "flex-start",
                "overflow": "visible",
                "box-sizing": "border-box",
                "padding-bottom": "8px",
            },
        )
        # Generous trailing space is intentional. In compact gridstack/Panel
        # layouts, wrapped FlexBox rows can be painted lower than the scroll
        # container's measured content height. Keeping a large spacer in the
        # same scrollable column ensures the final row can always be scrolled
        # fully into view instead of stopping mid-row.
        self._bottom_spacer = pn.Spacer(height=self._scroll_slack_px(), sizing_mode="stretch_width")
        self._scroller = pn.Column(
            self._grid,
            self._bottom_spacer,
            sizing_mode="stretch_both",
            styles={
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "height": "100%",
                "min-height": "240px",
                "padding": "8px",
                "scroll-padding-bottom": f"{self._scroll_slack_px()}px",
                "box-sizing": "border-box",
                "min-height": "0",
                "flex": "1 1 0",
                "max-height": "100%",
                "background": "#111",
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
            "thumb_size": int(self.thumb_size_widget.value),
            "max_items": int(self.max_items_widget.value),
            "fit_mode": self.fit_mode.value,
            "show_badges": bool(self.show_badges.value),
            "max_in_flight": int(self.max_in_flight),
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
        except Exception:
            pass
        if state.get("fit_mode") in {"Crop", "Contain"}:
            self.fit_mode.value = state["fit_mode"]
        if "show_badges" in state:
            self.show_badges.value = bool(state["show_badges"])

    def _subscribe(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
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
            self._subscriptions.append(events.subscribe(topic, callback))

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
        self._grid.objects = []
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
        self._schedule_render()

    def on_selection_focus_cleared(self, topic: str, payload: dict[str, Any]) -> None:
        self._focused_row_id = None
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
        self._sync_scroll_spacer()
        self._cancel_current_load()
        self._request_seq += 1
        request_seq = self._request_seq

        if not self._dataset_id or not self._row_ids:
            self._previews.clear()
            self._errors.clear()
            self._grid.objects = []
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
        self._errors = {row_id: error for row_id, error in self._errors.items() if row_id in visible_set and row_id not in self._previews}

        omitted = max(0, len(self._row_ids) - len(visible_row_ids))
        cached = len(self._previews)
        total = len(visible_row_ids)
        missing_row_ids = [row_id for row_id in visible_row_ids if row_id not in self._previews]
        self._pending_row_ids = list(missing_row_ids)

        self._status.alert_type = "primary" if missing_row_ids else "success"
        self._status.object = f"Loaded {cached}/{total} thumbnails"
        if missing_row_ids:
            self._status.object += f"; loading {len(missing_row_ids)}"
        if omitted:
            self._status.object += f"; {omitted} omitted"
        self._status.object += "."

        self._sync_scroll_spacer()
        self._render_grid()
        self._launch_more(request_seq)

    def _visible_row_ids(self) -> list[str]:
        return [str(row_id) for row_id in self._row_ids[: max(1, int(self.max_items))]]

    def _launch_more(self, request_seq: int) -> None:
        if request_seq != self._request_seq:
            return

        jobs = getattr(self.context, "jobs", None)
        resolver = get_image_resolver(self.context)

        while self._pending_row_ids and len(self._in_flight) < self.max_in_flight:
            row_id = self._pending_row_ids.pop(0)
            if row_id in self._previews:
                continue
            self._in_flight.add(row_id)

            if jobs is None:
                try:
                    preview = resolver.load_preview_for_row(
                        self._dataset_id,
                        row_id,
                        role="thumbnail",
                        max_size=self.thumb_size,
                        prefer_thumbnail=True,
                    )
                    self._on_preview_loaded(row_id, preview, request_seq)
                except Exception as exc:
                    self._on_preview_error(row_id, exc, request_seq)
                continue

            handle = jobs.submit(
                resolver.load_preview_for_row,
                title="Load image thumbnail",
                key=f"core.image.thumb:{self._dataset_id}:{row_id}:{self.thumb_size}:{request_seq}",
                on_done=lambda preview, row_id=row_id: self._on_preview_loaded(row_id, preview, request_seq),
                on_error=lambda exc, row_id=row_id: self._on_preview_error(row_id, exc, request_seq),
                dataset_id=self._dataset_id,
                row_id=row_id,
                role="thumbnail",
                max_size=self.thumb_size,
                prefer_thumbnail=True,
            )
            self._job_handles.append(handle)

    def _on_preview_loaded(self, row_id: str, preview: ImagePreview, request_seq: int) -> None:
        row_id = str(row_id)
        if request_seq != self._request_seq:
            return
        self._in_flight.discard(row_id)
        self._previews[row_id] = preview
        if self._dataset_id:
            self._preview_cache[(self._dataset_id, row_id, self.thumb_size)] = preview
        self._errors.pop(row_id, None)
        self._update_progress_status()
        self._schedule_render()
        self._launch_more(request_seq)

    def _on_preview_error(self, row_id: str, exc: BaseException, request_seq: int) -> None:
        row_id = str(row_id)
        if request_seq != self._request_seq:
            return
        self._in_flight.discard(row_id)
        self._errors[row_id] = str(exc)
        self._update_progress_status()
        self._schedule_render()
        self._launch_more(request_seq)

    def _update_progress_status(self) -> None:
        total = len(self._visible_row_ids())
        if total <= 0:
            return
        loaded = len([row_id for row_id in self._visible_row_ids() if row_id in self._previews])
        failed = len([row_id for row_id in self._visible_row_ids() if row_id in self._errors])
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
            self._grid.objects = []
            return
        self._sync_scroll_spacer()
        self._grid.objects = [self._card_for_row(row_id) for row_id in self._visible_row_ids()]

    def _card_for_row(self, row_id: str) -> pn.Column:
        focused = self._focused_row_id is not None and row_id == str(self._focused_row_id)
        preview = self._previews.get(row_id)
        error = self._errors.get(row_id)
        size = int(self.thumb_size)
        show_badges = bool(self.show_badges.value)
        label_height = 24
        badge_height = 44 if show_badges else 0
        body_height = size + label_height + badge_height
        button_height = 30
        inner_width = size
        card_padding = 6
        border_allowance = 6
        card_width = inner_width + (2 * card_padding) + border_allowance
        card_height = body_height + button_height + 22
        object_fit = "cover" if self.fit_mode.value == "Crop" else "contain"
        safe_row = html.escape(row_id)
        border = self._border_for_row(preview, error, focused)

        if preview is not None:
            metadata = preview.asset.metadata or {}
            badges = self._badges_html(metadata) if show_badges else ""
            title = html.escape(self._title_for_preview(preview), quote=True)
            image_html = f"""
            <div title="{title}" style="width:{size}px; height:{body_height}px; box-sizing:border-box; overflow:hidden;">
              <div style="width:{size}px; height:{size}px; background:#000; display:flex; align-items:center; justify-content:center; overflow:hidden;">
                <img src="{preview.data_uri}" alt="{safe_row}" style="width:100%; height:100%; object-fit:{object_fit}; display:block;" />
              </div>
              <div style="height:{label_height}px; line-height:{label_height}px; padding:0 2px; color:#ddd; font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; box-sizing:border-box;" title="{safe_row}">{safe_row}</div>
              {badges}
            </div>
            """
        elif error:
            safe_error = html.escape(error, quote=True)
            image_html = f"""
            <div title="{safe_error}" style="width:{size}px; height:{body_height}px; background:#220; color:#f99; display:flex; align-items:center; justify-content:center; text-align:center; padding:8px; box-sizing:border-box; font-size:11px; overflow:hidden;">
              Failed<br>{safe_row}
            </div>
            """
        else:
            image_html = f"""
            <div style="width:{size}px; height:{body_height}px; background:#222; color:#aaa; display:flex; align-items:center; justify-content:center; text-align:center; padding:8px; box-sizing:border-box; font-size:11px; overflow:hidden;">
              Loading<br>{safe_row}
            </div>
            """

        body = pn.pane.HTML(
            image_html,
            sizing_mode="fixed",
            width=size,
            height=body_height,
            margin=(0, 0, 6, 0),
        )

        focus_button = pn.widgets.Button(
            name="Focused" if focused else "Focus",
            button_type="primary" if focused else "default",
            width=size,
            height=button_height,
            margin=(0, 0, 0, 0),
        )
        focus_button.on_click(lambda event, row_id=row_id: self._focus_row(row_id))

        return pn.Column(
            body,
            focus_button,
            sizing_mode="fixed",
            width=card_width,
            height=card_height,
            margin=(0, 0, 0, 0),
            styles={
                "border": border,
                "border-radius": "6px",
                "padding": f"{card_padding}px",
                "background": "#181818",
                "box-sizing": "border-box",
                "overflow": "hidden",
                "flex": f"0 0 {card_width}px",
                "width": f"{card_width}px",
                "min-width": f"{card_width}px",
                "max-width": f"{card_width}px",
                "height": f"{card_height}px",
                "min-height": f"{card_height}px",
                "max-height": f"{card_height}px",
            },
        )

    def _badges_html(self, metadata: dict[str, Any]) -> str:
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
                    f"<span style='display:block; max-width:100%; margin:1px 0; padding:1px 4px; border-radius:4px; background:#333; color:#ddd; font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; box-sizing:border-box;'>{html.escape(label)}: {html.escape(value)}</span>"
                )
        return f"<div style='height:44px; overflow:hidden; line-height:1.2;'>{''.join(parts)}</div>" if parts else ""

    def _border_for_row(
        self,
        preview: Optional[ImagePreview],
        error: Optional[str],
        focused: bool,
    ) -> str:
        if focused:
            return "3px solid #4da3ff"
        if error:
            return "2px solid #bb4444"
        if preview is not None:
            state = str((preview.asset.metadata or {}).get("label_state") or "").lower()
            if state in {"unsure", "uncertain"}:
                return "2px dashed #d1a21b"
            if state in {"labelled", "labeled", "reviewed"}:
                return "2px solid #4a8f4a"
        return "1px solid #555"

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
        self._schedule_render()

    def _scroll_slack_px(self) -> int:
        # Needs to be substantially larger than one card row because Panel
        # gridstack containers can under-measure wrapped FlexBox content. The
        # slack is only empty scroll room after the final card, not visual space
        # between rows.
        return max(720, int(self.thumb_size) * 5)

    def _sync_scroll_spacer(self) -> None:
        slack = self._scroll_slack_px()
        try:
            self._bottom_spacer.height = slack
        except Exception:
            pass
        try:
            self._scroller.styles["scroll-padding-bottom"] = f"{slack}px"
        except Exception:
            pass

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
        self._sync_scroll_spacer()
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

        self._schedule(run, delay_ms=40)

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
