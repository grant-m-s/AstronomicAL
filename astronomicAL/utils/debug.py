from __future__ import annotations

import os
from typing import Any


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


DEBUG_BOOT = _truthy_env("ASTRONOMICAL_DEBUG_BOOT")
DEBUG_PLUGINS = _truthy_env("ASTRONOMICAL_DEBUG_PLUGINS") or DEBUG_BOOT


def boot_print(*args: Any, **kwargs: Any) -> None:
    """Print boot-order diagnostics only when explicitly enabled."""

    if DEBUG_BOOT:
        print("[BOOT]", *args, **kwargs)


def plugin_debug_print(*args: Any, **kwargs: Any) -> None:
    """Print plugin diagnostics only when explicitly enabled."""

    if DEBUG_PLUGINS:
        print("[plugins]", *args, **kwargs)