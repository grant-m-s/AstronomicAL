"""Aladin Lite plugin.

This package provides a lightweight platform-native Aladin Lite panel. It keeps
Aladin-specific browser/embed code out of the core platform while using the
platform dataset, mapping and selection services.
"""

from .plugin import manifest, register

__all__ = ["manifest", "register"]