from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from astronomicAL.platform.events import EventBus
from astronomicAL.platform.jobs import JobManager
from astronomicAL.platform.artifacts import ArtifactStore
from astronomicAL.platform.datasets import DatasetManager
from astronomicAL.platform.workspace import WorkspaceManager
from astronomicAL.platform.selection import SelectionManager
from astronomicAL.platform.services import ServiceRegistry


@dataclass
class AppContext:
    """Single object containing platform services.

    ``config`` is included temporarily to avoid a large Phase 1 refactor.
    Long term, most config/global uses should move into datasets, selection,
    events, artifacts, jobs, workspace, services, or plugins.
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