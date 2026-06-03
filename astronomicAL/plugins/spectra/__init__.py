"""Astronomy spectra plugin.

This package hosts DESI, SDSS/BOSS and Euclid spectrum panels as optional
astronomy-domain plugins. The heavy legacy retrieval/plotting classes are kept
in ``astro_data_utility_legacy.py``; the new files provide the platform-native
plugin wrapper.
"""

from .plugin import manifest, register

__all__ = ["manifest", "register"]