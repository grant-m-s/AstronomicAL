from __future__ import annotations

from copy import deepcopy
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd
import pytest

from astronomicAL.platform.artifacts import ArtifactStore
from astronomicAL.platform.context import AppContext
from astronomicAL.platform.datasets import DatasetManager
from astronomicAL.platform.events import EventBus
from astronomicAL.platform.jobs import JobManager
from astronomicAL.platform.plugins.manager import PluginManager
from astronomicAL.platform.selection import SelectionManager
from astronomicAL.platform.services import ServiceRegistry


def _safe_dispose(obj: Any) -> None:
    dispose = getattr(obj, "dispose", None)
    if callable(dispose):
        dispose()


def _controller_state(controller: Any) -> dict[str, Any]:
    if controller is None:
        return {}

    get_state = getattr(controller, "get_state", None)
    if callable(get_state):
        state = get_state()
        if state is None:
            return {}
        if isinstance(state, dict):
            return dict(state)
        return {"value": state}

    return {}


def _controller_state_version(controller: Any, default: int = 1) -> int:
    if controller is None:
        return int(default)

    get_state_version = getattr(controller, "get_state_version", None)
    if callable(get_state_version):
        try:
            return int(get_state_version())
        except Exception:
            return int(default)

    return int(getattr(controller, "state_version", default))


@dataclass
class FakePanelRecord:
    panel_id: str
    title: str
    view: Any
    controller: Any | None = None
    kind: str = "plugin_panel"
    plugin_id: str | None = None
    registration_id: str | None = None
    plugin_version: str | None = None
    state_version: int = 1
    persistent: bool = True
    open_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeWorkspace:
    """
    Headless workspace double.

    These tests are about platform/plugin contracts, not Panel/DynamicReactGrid
    rendering. The persistence tests need enough WorkspaceManager-compatible
    snapshot/restore behaviour to exercise WorkspacePersistence without a
    browser-backed grid.
    """

    def __init__(self) -> None:
        self._panels: dict[str, FakePanelRecord] = {}
        self.removed: list[str] = []
        self._grid: dict[str, Any] = {
            "keys": [],
            "layouts": {"lg": []},
            "breakpoints": {},
            "cols_by_breakpoint": {"lg": 12},
            "current_breakpoint": "lg",
            "current_layout": [],
            "row_height": 30,
            "margin": [10, 10],
            "compact_type": None,
            "resize_handles": ["se"],
            "titles": {},
        }

    def _default_layout_item(self, panel_id: str) -> dict[str, Any]:
        return {
            "i": str(panel_id),
            "x": 0,
            "y": len(self._panels) * 4,
            "w": 4,
            "h": 4,
        }

    def _set_layout_for_panel(
        self,
        panel_id: str,
        *,
        layout_item: dict[str, Any] | None = None,
        layout_items: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        panel_id = str(panel_id)

        layouts = deepcopy(dict(self._grid.get("layouts") or {}))
        if not layouts:
            layouts = {"lg": []}

        if layout_items:
            for breakpoint, item in layout_items.items():
                item_copy = dict(item or {})
                item_copy["i"] = panel_id

                existing = [
                    dict(layout)
                    for layout in layouts.get(str(breakpoint), [])
                    if str(layout.get("i")) != panel_id
                ]
                existing.append(item_copy)
                layouts[str(breakpoint)] = existing
        else:
            item = dict(layout_item or self._default_layout_item(panel_id))
            item["i"] = panel_id

            existing = [
                dict(layout)
                for layout in layouts.get("lg", [])
                if str(layout.get("i")) != panel_id
            ]
            existing.append(item)
            layouts["lg"] = existing

        self._grid["layouts"] = layouts
        self._grid["current_layout"] = deepcopy(
            layouts.get(self._grid.get("current_breakpoint", "lg"), [])
            or layouts.get("lg", [])
            or []
        )

    def _attach_metadata_to_view(self, record: FakePanelRecord) -> None:
        attrs = {
            "_al_panel_id": record.panel_id,
            "_al_title": record.title,
            "_al_kind": record.kind,
            "_al_plugin_id": record.plugin_id,
            "_al_registration_id": record.registration_id,
            "_al_plugin_version": record.plugin_version,
            "_al_state_version": record.state_version,
            "_al_persistent": record.persistent,
            "_al_open_kwargs": dict(record.open_kwargs),
            "_al_metadata": dict(record.metadata),
            "_al_panel_record": record,
        }

        targets = [record.view]
        if record.controller is not None and record.controller is not record.view:
            targets.append(record.controller)

        for target in targets:
            if target is None:
                continue
            for key, value in attrs.items():
                try:
                    setattr(target, key, value)
                except Exception:
                    pass

        try:
            setattr(record.view, "_al_controller", record.controller)
        except Exception:
            pass

    def add_panel(
        self,
        panel_id: str,
        view: Any,
        *,
        title: str | None = None,
        controller: Any | None = None,
        layout_item: dict[str, Any] | None = None,
        layout_items: dict[str, dict[str, Any]] | None = None,
        kind: str = "plugin_panel",
        plugin_id: str | None = None,
        registration_id: str | None = None,
        plugin_version: str | None = None,
        state_version: int = 1,
        persistent: bool = True,
        open_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **extra_metadata: Any,
    ) -> FakePanelRecord:
        panel_id = str(panel_id)

        if panel_id in self._panels:
            self.remove_panel(panel_id)

        merged_metadata = dict(metadata or {})
        merged_metadata.update(extra_metadata)

        # Existing integration helpers inspect kind through metadata because the
        # first FakeWorkspace stored unknown kwargs there. Keep that compatibility
        # while also storing kind on the record.
        merged_metadata.setdefault("kind", kind)

        record = FakePanelRecord(
            panel_id=panel_id,
            title=title or panel_id,
            view=view,
            controller=controller,
            kind=kind,
            plugin_id=plugin_id,
            registration_id=registration_id,
            plugin_version=plugin_version,
            state_version=int(state_version or 1),
            persistent=bool(persistent),
            open_kwargs=dict(open_kwargs or {}),
            metadata=merged_metadata,
        )

        self._panels[panel_id] = record
        self._attach_metadata_to_view(record)

        keys = [str(key) for key in self._grid.get("keys", [])]
        if panel_id not in keys:
            keys.append(panel_id)
        self._grid["keys"] = keys

        titles = dict(self._grid.get("titles") or {})
        titles[panel_id] = record.title
        self._grid["titles"] = titles

        self._set_layout_for_panel(
            panel_id,
            layout_item=layout_item,
            layout_items=layout_items,
        )

        return record

    def remove_panel(self, panel_id: str) -> None:
        panel_id = str(panel_id)
        record = self._panels.pop(panel_id, None)
        if record is None:
            return

        self.removed.append(panel_id)

        if record.controller is not None:
            _safe_dispose(record.controller)

        if record.view is not record.controller:
            _safe_dispose(record.view)

        self._grid["keys"] = [
            key for key in self._grid.get("keys", []) if str(key) != panel_id
        ]

        titles = dict(self._grid.get("titles") or {})
        titles.pop(panel_id, None)
        self._grid["titles"] = titles

        layouts = {}
        for breakpoint, layout in (self._grid.get("layouts") or {}).items():
            layouts[breakpoint] = [
                item for item in layout if str(item.get("i")) != panel_id
            ]
        self._grid["layouts"] = layouts
        self._grid["current_layout"] = deepcopy(
            layouts.get(self._grid.get("current_breakpoint", "lg"), [])
            or layouts.get("lg", [])
            or []
        )

    def list_panels(self) -> list[FakePanelRecord]:
        return list(self._panels.values())

    def get_panel_record(self, panel_id: str) -> FakePanelRecord | None:
        return self._panels.get(str(panel_id))

    def clear(self) -> None:
        for panel_id in list(self._panels):
            self.remove_panel(panel_id)

        self._panels.clear()
        self._grid["keys"] = []
        self._grid["layouts"] = {"lg": []}
        self._grid["current_layout"] = []
        self._grid["titles"] = {}

    def snapshot_grid(self) -> dict[str, Any]:
        return deepcopy(self._grid)

    def apply_grid_snapshot(self, grid_snapshot: dict[str, Any]) -> None:
        snapshot = deepcopy(dict(grid_snapshot or {}))
        self._grid.update(snapshot)

        self._grid.setdefault("keys", [])
        self._grid.setdefault("layouts", {"lg": []})
        self._grid.setdefault("current_breakpoint", "lg")
        self._grid.setdefault("current_layout", [])
        self._grid.setdefault("titles", {})

    def snapshot_panels(self) -> list[dict[str, Any]]:
        panels: list[dict[str, Any]] = []

        for panel_id, record in self._panels.items():
            if not record.persistent:
                continue

            controller = record.controller
            if controller is None:
                controller = getattr(record.view, "_al_controller", None)

            panels.append(
                {
                    "instance_id": panel_id,
                    "kind": record.kind,
                    "plugin_id": record.plugin_id,
                    "registration_id": record.registration_id,
                    "plugin_version": record.plugin_version,
                    "title": record.title,
                    "state_version": _controller_state_version(
                        controller,
                        default=record.state_version,
                    ),
                    "state": _controller_state(controller),
                    "open_kwargs": dict(record.open_kwargs or {}),
                    "metadata": dict(record.metadata or {}),
                }
            )

        return panels

    def snapshot(self) -> dict[str, Any]:
        return {
            "grid": self.snapshot_grid(),
            "panels": self.snapshot_panels(),
        }


def _build_test_app_context(tmp_path: Path) -> AppContext:
    events = EventBus(trace=True)
    artifacts = ArtifactStore(cache_dir=str(tmp_path / "artifacts"))
    datasets = DatasetManager()
    jobs = JobManager(max_workers=2)
    selection = SelectionManager(events=events, artifacts=artifacts)
    services = ServiceRegistry()
    workspace = FakeWorkspace()

    context = AppContext(
        events=events,
        jobs=jobs,
        artifacts=artifacts,
        datasets=datasets,
        workspace=workspace,
        selection=selection,
        services=services,
        config=SimpleNamespace(),
    )

    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "value": [1.0, 2.0, 3.0],
            "other": [10, 20, 30],
            "name": ["alpha", "beta", "gamma"],
            "label": [0, 1, -1],
        }
    )

    datasets.register("main", df, name="Main test dataset")
    datasets.set_mapping("main", "record_id", "id")
    datasets.set_mapping("main", "target_label", "label")

    return context


@pytest.fixture
def make_app_context(tmp_path: Path) -> Callable[[str], AppContext]:
    contexts: list[AppContext] = []

    def _make_app_context(name: str = "context") -> AppContext:
        context = _build_test_app_context(tmp_path / name)
        contexts.append(context)
        return context

    yield _make_app_context

    for context in contexts:
        try:
            context.workspace.clear()
        except Exception:
            pass
        try:
            context.jobs.shutdown(wait=False)
        except Exception:
            pass


@pytest.fixture
def app_context(make_app_context: Callable[[str], AppContext]) -> AppContext:
    return make_app_context("app_context")


@pytest.fixture
def write_plugin(tmp_path: Path) -> Callable[[str, str], Path]:
    """
    Create a local plugin directory accepted by PluginManager(local_plugin_dirs=[...]).
    """

    root = tmp_path / "local_plugins"
    root.mkdir(exist_ok=True)

    def _write_plugin(package_name: str, code: str) -> Path:
        plugin_dir = root / package_name
        plugin_dir.mkdir(parents=True, exist_ok=True)

        plugin_file = plugin_dir / "plugin.py"
        plugin_file.write_text(textwrap.dedent(code), encoding="utf-8")

        return root

    return _write_plugin


@pytest.fixture
def make_manager(
    app_context: AppContext,
    write_plugin: Callable[[str, str], Path],
) -> Callable[[str, str], PluginManager]:
    def _make_manager(package_name: str, code: str) -> PluginManager:
        plugin_root = write_plugin(package_name, code)
        manager = PluginManager(local_plugin_dirs=[plugin_root])
        app_context.plugins = manager
        manager.discover()

        errors = manager.list_discovery_errors()
        if errors:
            raise AssertionError(f"Plugin discovery errors: {errors}")

        return manager

    return _make_manager


def recent_event_topics(context: AppContext) -> list[str]:
    return [topic for _, topic, _ in context.events.recent_events(500)]


def recent_events_for(context: AppContext, topic_name: str) -> list[dict[str, Any]]:
    return [
        payload
        for _, topic, payload in context.events.recent_events(500)
        if topic == topic_name
    ]