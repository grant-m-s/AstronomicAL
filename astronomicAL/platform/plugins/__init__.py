from __future__ import annotations

from astronomicAL.platform.mapping_requirements import MappingRequirement
from .api import PluginAPI
from .errors import (
    PluginDiscoveryError,
    PluginError,
    PluginExecutionError,
    PluginInstallError,
    PluginLoadError,
    PluginPackageError,
    PluginRegistrationError,
    PluginValidationError,
    InstalledPluginStoreError,
)
from .manifest import PluginManifest, PluginRequirement, parse_plugin_requirement
from .manager import PluginManager
from .activation import PluginActivationService
from .installed import InstalledPluginRecord, InstalledPluginStore
from .installer import PluginInstallResult, PluginInstaller, detect_astronomical_version
from .package import PluginPackageInspection, inspect_plugin_package
from .sources import PluginOrigin, PluginSearchPath
from .state import PluginStateStore
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
    "PluginInstallError",
    "PluginLoadError",
    "PluginPackageError",
    "PluginRegistrationError",
    "PluginValidationError",
    "InstalledPluginStoreError",
    "PluginManifest",
    "PluginRequirement",
    "parse_plugin_requirement",
    "PluginManager",
    "PluginActivationService",
    "InstalledPluginRecord",
    "InstalledPluginStore",
    "PluginInstallResult",
    "PluginInstaller",
    "detect_astronomical_version",
    "PluginPackageInspection",
    "inspect_plugin_package",
    "PluginOrigin",
    "PluginSearchPath",
    "PluginStateStore",
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