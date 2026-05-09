from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from astronomicAL.platform.artifacts import ArtifactStore
from astronomicAL.platform.datasets import DatasetManager
from astronomicAL.platform.events import EventBus
from astronomicAL.platform.jobs import JobManager
from astronomicAL.platform.selection import SelectionManager
from astronomicAL.platform.services import ServiceRegistry
from astronomicAL.platform.workspace import WorkspaceManager


@dataclass
class AppContext:
    """
    Runtime dependency object for platform services.

    config remains temporarily available during the transition, but new runtime
    state should live in datasets, selection, artifacts, events, jobs, workspace,
    services, plugins, or persistence.
    """

    events: EventBus
    jobs: JobManager
    artifacts: ArtifactStore
    datasets: DatasetManager
    workspace: WorkspaceManager
    selection: SelectionManager
    services: ServiceRegistry
    config: Optional[Any] = None
    plugins: Optional[Any] = None
    persistence: Optional[Any] = None