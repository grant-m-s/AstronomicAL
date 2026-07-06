"""Active-learning plugin for AstronomicAL.

The package is intentionally thin at import time.  ``plugin.py`` exposes the
platform manifest/register contract; the implementation is split into small
modules so query-strategy authors do not need to touch panel or core.ml code.
"""
