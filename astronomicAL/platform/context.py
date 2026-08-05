from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from astronomicAL.platform.artifacts import ArtifactStore
from astronomicAL.platform.datasets import DatasetManager
from astronomicAL.platform.events import EventBus
from astronomicAL.platform.jobs import JobManager
from astronomicAL.platform.record_navigation import RecordNavigationManager
from astronomicAL.platform.runtime_status import RuntimeStatus
from astronomicAL.platform.selection import SelectionManager
from astronomicAL.platform.services import ServiceRegistry
from astronomicAL.platform.workspace import WorkspaceManager

@dataclass
class AppContext:
    """
    Runtime dependency object for platform services and application paths.

    Runtime state lives in the explicit platform services below. Application
    paths are carried directly rather than through a process-global config
    module.
    """

    events: EventBus
    jobs: JobManager
    artifacts: ArtifactStore
    datasets: DatasetManager
    workspace: WorkspaceManager
    selection: SelectionManager
    services: ServiceRegistry

    layout_file: Path = Path("astronomicAL/layout.json")
    layout_directory: Path = Path("layouts")

    navigation: Optional[RecordNavigationManager] = None
    plugins: Optional[Any] = None
    plugin_state: Optional[Any] = None
    plugin_activation: Optional[Any] = None
    installed_plugins: Optional[Any] = None
    plugin_installer: Optional[Any] = None
    marketplace: Optional[Any] = None
    marketplace_planner: Optional[Any] = None
    marketplace_updates: Optional[Any] = None
    marketplace_installer: Optional[Any] = None
    persistence: Optional[Any] = None
    runtime_status: Optional[RuntimeStatus] = None