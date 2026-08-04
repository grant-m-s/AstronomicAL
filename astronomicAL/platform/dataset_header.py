from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

import panel as pn

from astronomicAL.platform.application_chrome_styles import (
    APPLICATION_CHROME_CSS,
    HEADER_BUTTON_STYLESHEET,
    HEADER_PRIMARY_BUTTON_STYLESHEET,
    HEADER_SELECT_STYLESHEET,
)
from astronomicAL.platform.dataset_loader import DatasetLoaderController
from astronomicAL.platform.modal_utils import open_template_modal


def _install_application_chrome_css() -> None:
    marker = "--al-application-chrome-v7"
    if any(marker in css for css in pn.config.raw_css):
        return
    pn.config.raw_css.append(APPLICATION_CHROME_CSS)


class DatasetHeaderController:
    """Global active-dataset switcher and entry point to the dataset loader."""

    def __init__(self, context: Any, template: Any) -> None:
        if context is None:
            raise ValueError("DatasetHeaderController requires an AppContext.")
        if getattr(context, "datasets", None) is None:
            raise ValueError("DatasetHeaderController requires context.datasets.")

        self.context = context
        self.template = template
        self._subs: list[Any] = []
        self._updating_select = False
        self._disposed = False

        _install_application_chrome_css()

        self.context_label = pn.pane.HTML(
            '<span aria-hidden="true">Dataset</span>',
            width=52,
            height=30,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-context-label"],
        )

        self.active_dataset_select = pn.widgets.Select(
            name="",
            options=OrderedDict({"No dataset loaded": ""}),
            value="",
            width=270,
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-header-dataset-select"],
            stylesheets=[HEADER_SELECT_STYLESHEET],
        )
        self.active_dataset_select.description = "Switch the active dataset"
        self._active_dataset_watcher = self.active_dataset_select.param.watch(
            self._on_active_dataset_select_changed,
            "value",
        )

        self.button = pn.widgets.Button(
            name="Add data",
            icon="database-plus",
            button_type="default",
            width=98,
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-header-add-data"],
            stylesheets=[HEADER_BUTTON_STYLESHEET],
        )
        self.button.description = "Register another FITS or Parquet dataset"
        self.button.on_click(self._open_modal)

        self.dataset_stats = pn.pane.HTML(
            "",
            width=178,
            height=30,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
            visible=False,
            css_classes=["al-dataset-stats"],
        )

        self.view = pn.Row(
            self.context_label,
            self.active_dataset_select,
            self.button,
            self.dataset_stats,
            sizing_mode="fixed",
            height=30,
            margin=(0, 0, 0, 0),
            css_classes=["al-dataset-header"],
            styles={"overflow": "visible", "min-width": "0"},
        )

        self.loader = DatasetLoaderController(
            context=self.context,
            template=self.template,
        )
        self.modal_root = self.loader.view

        self._subscribe()
        self._refresh_header()

    # ------------------------------------------------------------------
    # Event wiring and lifecycle
    # ------------------------------------------------------------------

    def _subscribe(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return

        for topic in (
            "dataset.loaded",
            "dataset.active.changed",
            "dataset.updated",
            "dataset.removed",
            "dataset.open_requested",
        ):
            try:
                subscription = events.subscribe(
                    topic,
                    self._on_dataset_event,
                    owner_id="platform.dataset_header",
                    owner_label="Dataset Header",
                    owner_kind="platform-header",
                )
            except TypeError:
                subscription = events.subscribe(topic, self._on_dataset_event)
            self._subs.append(subscription)

    def _on_dataset_event(self, topic: str, payload: Any) -> None:
        del payload
        if self._disposed:
            return
        if topic == "dataset.open_requested":
            self._open_modal()
            return
        self._refresh_header()

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True

        try:
            self.active_dataset_select.param.unwatch(self._active_dataset_watcher)
        except Exception:
            pass

        events = getattr(self.context, "events", None)
        if events is not None:
            for subscription in list(self._subs):
                try:
                    events.unsubscribe(subscription)
                except Exception:
                    pass
        self._subs.clear()
        self.loader.dispose()

    # ------------------------------------------------------------------
    # Header state
    # ------------------------------------------------------------------

    def _refresh_header(self) -> None:
        dataset_ids = self._dataset_ids()
        active_id = self._active_dataset_id_or_none()

        self._updating_select = True
        try:
            if not dataset_ids:
                self.active_dataset_select.options = OrderedDict(
                    {"No dataset loaded": ""}
                )
                self.active_dataset_select.value = ""
                self.active_dataset_select.disabled = True
                self.active_dataset_select.description = "No dataset is loaded"
                self.dataset_stats.object = ""
                self.dataset_stats.visible = False
                self.button.button_type = "primary"
                self.button.stylesheets = [HEADER_PRIMARY_BUTTON_STYLESHEET]
                self.button.description = "Load the first FITS or Parquet dataset"
                return

            options = self._dataset_option_labels(dataset_ids)
            selected_id = active_id if active_id in dataset_ids else dataset_ids[0]

            self.active_dataset_select.options = options
            self.active_dataset_select.disabled = False
            self.active_dataset_select.value = selected_id

            name = self._dataset_name(selected_id)
            dimensions = self._dataset_dimensions(selected_id)
            self.active_dataset_select.description = (
                f"Active dataset: {name}"
                + (f" ({dimensions})" if dimensions else "")
            )
            self.dataset_stats.object = (
                dimensions
                if dimensions
                else '<span aria-hidden="true">&nbsp;</span>'
            )
            self.dataset_stats.visible = True
            self.button.button_type = "default"
            self.button.stylesheets = [HEADER_BUTTON_STYLESHEET]
            self.button.description = "Register another FITS or Parquet dataset"
        finally:
            self._updating_select = False

    def _on_active_dataset_select_changed(self, event: Any) -> None:
        if self._updating_select or self._disposed:
            return
        dataset_id = str(event.new or "")
        if not dataset_id:
            return
        self._set_active_dataset(dataset_id)

    def _dataset_option_labels(self, dataset_ids: list[str]) -> OrderedDict[str, str]:
        names = [self._dataset_name(dataset_id) for dataset_id in dataset_ids]
        counts: dict[str, int] = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1

        options: OrderedDict[str, str] = OrderedDict()
        for dataset_id, name in zip(dataset_ids, names):
            label = name if counts[name] == 1 else f"{name} · {dataset_id}"
            options[label] = dataset_id
        return options

    def _dataset_name(self, dataset_id: str) -> str:
        try:
            dataset = self.context.datasets.get(dataset_id)
            return str(getattr(dataset, "name", None) or dataset_id)
        except Exception:
            return str(dataset_id)

    def _dataset_dimensions(self, dataset_id: str) -> str:
        try:
            rows = self.context.datasets.row_count(dataset_id)
        except Exception:
            rows = None
        try:
            columns = len(self.context.datasets.list_columns(dataset_id))
        except Exception:
            columns = None

        if rows is not None and columns is not None:
            return f"{int(rows):,} rows · {int(columns):,} cols"
        if rows is not None:
            return f"{int(rows):,} rows"
        if columns is not None:
            return f"{int(columns):,} cols"
        return ""

    def _set_active_dataset(self, dataset_id: str) -> None:
        self._clear_selection_for_dataset_switch()
        self.context.datasets.set_active(
            dataset_id,
            origin="platform.dataset_header",
        )
        self._refresh_header()

    # ------------------------------------------------------------------
    # Modal and helpers
    # ------------------------------------------------------------------

    def _open_modal(self, _event: Any = None) -> None:
        if self._disposed:
            return
        self.loader.prepare()
        open_template_modal(
            self.template,
            self.modal_root,
            close_on_backdrop=True,
        )

    def _clear_selection_for_dataset_switch(self) -> None:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return
        try:
            selection.clear_focus(origin="dataset.active.changed")
        except Exception:
            pass
        try:
            selection.clear_selection_set(origin="dataset.active.changed")
        except Exception:
            pass

    def has_active_dataset(self) -> bool:
        dataset_id = self._active_dataset_id_or_none()
        if not dataset_id:
            return False
        try:
            return dataset_id in self.context.datasets.list_ids()
        except Exception:
            return False

    def _dataset_ids(self) -> list[str]:
        try:
            return [str(dataset_id) for dataset_id in self.context.datasets.list_ids()]
        except Exception:
            return []

    def _active_dataset_id_or_none(self) -> Optional[str]:
        try:
            value = self.context.datasets.active_id()
        except Exception:
            return None
        return str(value) if value not in (None, "") else None


__all__ = ["DatasetHeaderController"]