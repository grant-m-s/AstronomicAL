from __future__ import annotations

from astronomicAL.platform.mapping_requirements import MappingRequirement
from .api import PluginAPI
from .errors import (
    PluginDiscoveryError,
    PluginError,
    PluginExecutionError,
    PluginLoadError,
    PluginRegistrationError,
    PluginValidationError,
)
from .manifest import PluginManifest
from .manager import PluginManager
from .specs import (
    ActionRegistration,
    ActionRequest,
    ActionResult,
    ArtifactResult,
    ArtifactViewerRegistration,
    DatasetResult,
    EventResult,
    InputSpec,
    OutputSpec,
    PanelRegistration,
    PluginInfo,
    PluginStatus,
    ProcessedActionResult,
    ServiceRegistration,
    ValidationResult,
    WorkflowRegistration,
)

__all__ = [
    "PluginAPI",
    "PluginError",
    "PluginDiscoveryError",
    "PluginExecutionError",
    "PluginLoadError",
    "PluginRegistrationError",
    "PluginValidationError",
    "PluginManifest",
    "PluginManager",
    "ActionRegistration",
    "ActionRequest",
    "ActionResult",
    "ArtifactResult",
    "ArtifactViewerRegistration",
    "DatasetResult",
    "EventResult",
    "InputSpec",
    "OutputSpec",
    "MappingRequirement",
    "PanelRegistration",
    "PluginInfo",
    "PluginStatus",
    "ProcessedActionResult",
    "ServiceRegistration",
    "ValidationResult",
    "WorkflowRegistration",
]