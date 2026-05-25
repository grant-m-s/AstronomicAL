from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure tests can import the local package even before editable install.
# This must happen before importing from astronomicAL.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from astronomicAL.platform.artifacts import ArtifactStore
from astronomicAL.platform.context import AppContext
from astronomicAL.platform.datasets import DatasetManager
from astronomicAL.platform.events import EventBus
from astronomicAL.platform.jobs import JobManager
from astronomicAL.platform.selection import SelectionManager
from astronomicAL.platform.services import ServiceRegistry


class ConfigWriteGuard:
    """
    Test helper for new plugin/platform code.

    context.config is allowed as a migration bridge, but new runtime state
    should not be written there.
    """

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(
            f"Attempted to read context.config.{name!s}. "
            "New code should use datasets, selection, artifacts, events, jobs, "
            "workspace, or services instead."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AssertionError(
            f"Attempted to write context.config.{name!s}. "
            "New runtime state must not be stored in context.config."
        )


@dataclass
class DummyWorkspace:
    """
    Lightweight workspace stand-in for platform service tests.
    """

    panels: dict[str, dict[str, Any]] = field(default_factory=dict)
    removed: list[str] = field(default_factory=list)

    def add_panel(
        self,
        panel_id: str,
        view: Any,
        *,
        title: str | None = None,
        controller: Any = None,
        **metadata: Any,
    ) -> None:
        self.panels[str(panel_id)] = {
            "view": view,
            "title": title or str(panel_id),
            "controller": controller,
            "metadata": metadata,
        }

    def remove_panel(self, panel_id: str) -> None:
        panel_id = str(panel_id)
        record = self.panels.pop(panel_id, None)
        self.removed.append(panel_id)

        if record is None:
            return

        controller = record.get("controller")
        view = record.get("view")

        for target in (controller, view):
            if target is None:
                continue
            dispose = getattr(target, "dispose", None)
            if callable(dispose):
                dispose()

    def snapshot(self) -> dict[str, Any]:
        return {
            "panels": [
                {
                    "panel_id": panel_id,
                    "title": record["title"],
                    "metadata": record["metadata"],
                }
                for panel_id, record in self.panels.items()
            ]
        }


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["r1", "r2", "r3", "r4"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [4.0, 5.0, 6.0, 7.0],
            "label": [0, 1, -1, 1],
            "ra": [10.1, 10.2, 10.3, 10.4],
            "dec": [-1.1, -1.2, -1.3, -1.4],
            "note": ["a", "b", "c", "d"],
        }
    )


@pytest.fixture
def events() -> EventBus:
    return EventBus(trace=True)


@pytest.fixture
def artifacts(tmp_path) -> ArtifactStore:
    return ArtifactStore(cache_dir=str(tmp_path / "artifact_cache"))


@pytest.fixture
def datasets(sample_df: pd.DataFrame) -> DatasetManager:
    manager = DatasetManager()
    manager.register("main", sample_df.copy(), name="Main test dataset")
    manager.set_active("main")
    manager.set_mapping("main", "record_id", "id")
    manager.set_mapping("main", "target_label", "label")
    return manager


@pytest.fixture
def jobs():
    manager = JobManager(max_workers=4)
    try:
        yield manager
    finally:
        manager.shutdown(wait=True)


@pytest.fixture
def services() -> ServiceRegistry:
    return ServiceRegistry()


@pytest.fixture
def workspace() -> DummyWorkspace:
    return DummyWorkspace()


@pytest.fixture
def context(
    events: EventBus,
    jobs: JobManager,
    artifacts: ArtifactStore,
    datasets: DatasetManager,
    services: ServiceRegistry,
    workspace: DummyWorkspace,
) -> AppContext:
    return AppContext(
        events=events,
        jobs=jobs,
        artifacts=artifacts,
        datasets=datasets,
        workspace=workspace,
        selection=SelectionManager(events, artifacts=artifacts),
        services=services,
        config=ConfigWriteGuard(),
    )


# ---------------------------------------------------------------------------
# Plugin-quality reporting
# ---------------------------------------------------------------------------

def _plugin_name_from_path(path: Path) -> str | None:
    """
    Convert a path such as:

        astronomicAL/plugins/record_browser/plugin.py

    into:

        record_browser

    This intentionally uses the plugin directory name rather than importing the
    plugin manifest, so the summary does not add extra import-time cost.
    """

    parts = path.parts

    try:
        plugins_index = parts.index("plugins")
    except ValueError:
        return None

    try:
        return parts[plugins_index + 1]
    except IndexError:
        return None


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def pytest_configure(config):
    """
    Storage for the plugin-quality terminal summary.

    This is intentionally attached to pytest's config object rather than kept in
    module globals so it behaves better under pytest internals and future plugin
    use.
    """

    config._astronomical_plugin_quality_results = {}


def pytest_collection_modifyitems(config, items):
    """
    Attach plugin metadata to parametrized plugin-quality tests.

    The static plugin-quality tests parametrize a variable named `path`. When
    that path points into astronomicAL/plugins/<plugin_name>/..., we record the
    plugin name and file path as test user properties. Later, the terminal
    summary can report exactly which plugins were tested.
    """

    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue

        params = getattr(callspec, "params", {})
        raw_path = params.get("path")
        if raw_path is None:
            continue

        path = Path(raw_path)
        plugin_name = _plugin_name_from_path(path)
        if not plugin_name:
            continue

        rel_path = _relative_to_project(path)

        item.user_properties.append(("astronomical_plugin", plugin_name))
        item.user_properties.append(("astronomical_plugin_file", rel_path))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Record pass/fail/skip/xfail status for plugin-quality tests.

    We use makereport rather than print() inside tests because pytest captures
    normal print output by default. The final terminal summary is visible in
    ordinary pytest output.
    """

    outcome = yield
    report = outcome.get_result()

    if report.when != "call":
        return

    properties = dict(item.user_properties)
    plugin_name = properties.get("astronomical_plugin")
    plugin_file = properties.get("astronomical_plugin_file")

    if not plugin_name:
        return

    results = item.config._astronomical_plugin_quality_results

    entry = results.setdefault(
        plugin_name,
        {
            "files": set(),
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
            "tests": [],
        },
    )

    if plugin_file:
        entry["files"].add(plugin_file)

    if getattr(report, "wasxfail", None):
        if report.outcome == "passed":
            status = "xpassed"
        else:
            status = "xfailed"
    else:
        status = report.outcome

    if status not in entry:
        entry[status] = 0

    entry[status] += 1

    entry["tests"].append(
        {
            "nodeid": report.nodeid,
            "status": status,
            "file": plugin_file,
        }
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Print a clear list of plugins covered by plugin-quality tests.

    This only prints when plugin-quality tests actually ran. Running the default
    fast platform suite will stay clean.
    """

    results = getattr(config, "_astronomical_plugin_quality_results", {})

    if not results:
        return

    terminalreporter.section("AstronomicAL plugin-quality summary")

    for plugin_name in sorted(results):
        entry = results[plugin_name]

        passed = entry.get("passed", 0)
        failed = entry.get("failed", 0)
        skipped = entry.get("skipped", 0)
        xfailed = entry.get("xfailed", 0)
        xpassed = entry.get("xpassed", 0)

        total = passed + failed + skipped + xfailed + xpassed

        if failed:
            symbol = "FAILED"
        elif xpassed:
            symbol = "XPASS"
        elif skipped and not passed:
            symbol = "SKIPPED"
        else:
            symbol = "PASSED"

        terminalreporter.write_line(
            f"{symbol:7} {plugin_name}: "
            f"{total} checks "
            f"({passed} passed, {failed} failed, {skipped} skipped, "
            f"{xfailed} xfailed, {xpassed} xpassed)"
        )

        for file_path in sorted(entry["files"]):
            terminalreporter.write_line(f"          - {file_path}")