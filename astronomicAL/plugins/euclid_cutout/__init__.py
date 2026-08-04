"""Euclid cutout plugin.

This package keeps Euclid-specific code out of the generic AstronomicAL core.
The plugin exposes a panel, a runtime service and an artifact viewer through the
platform plugin API.
"""

from .plugin import manifest, register

__all__ = ["manifest", "register"]