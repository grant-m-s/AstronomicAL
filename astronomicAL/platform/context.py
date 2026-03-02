# astronomicAL/platform/context.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from astronomicAL.platform.events import EventBus
from astronomicAL.platform.jobs import JobManager
from astronomicAL.platform.artifacts import ArtifactStore
from astronomicAL.platform.datasets import DatasetManager
from astronomicAL.platform.workspace import WorkspaceManager
from astronomicAL.platform.services import ServiceRegistry

@dataclass
class AppContext:
    """
    Single object containing platform services.

    NOTE: `config` is included temporarily to avoid massive refactors in Phase 1.
    Long term you should replace most config/global uses with datasets/events/artifacts.
    """
    events: EventBus
    jobs: JobManager
    artifacts: ArtifactStore
    datasets: DatasetManager
    workspace: WorkspaceManager
    services: ServiceRegistry
    config: Optional[Any] = None