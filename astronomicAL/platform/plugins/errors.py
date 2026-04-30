from __future__ import annotations


class PluginError(Exception):
    """Base class for plugin-system errors."""


class PluginDiscoveryError(PluginError):
    """Raised when a plugin candidate cannot be discovered or inspected."""


class PluginLoadError(PluginError):
    """Raised when a plugin module cannot be imported or loaded."""


class PluginValidationError(PluginError):
    """Raised when a plugin manifest, dependency set, or runtime request is invalid."""


class PluginRegistrationError(PluginError):
    """Raised when a plugin registers an invalid contribution."""


class PluginExecutionError(PluginError):
    """Raised when a registered plugin action/panel/workflow fails."""