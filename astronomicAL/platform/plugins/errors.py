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

class PluginPackageError(PluginError):
    """Raised when an .alplugin archive is malformed, unsafe, or invalid."""

class PluginInstallError(PluginError):
    """Raised when a plugin install, update, or uninstall transaction cannot complete."""

class InstalledPluginStoreError(PluginError):
    """Raised when persistent installed-plugin metadata cannot be read or changed."""

class PluginPythonEnvironmentError(PluginInstallError):
    """Raised when managed community-plugin Python dependencies cannot be reconciled."""