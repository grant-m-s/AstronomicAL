from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def make_json_safe(value: Any) -> Any:
    """
    Convert common Python objects into JSON-safe values.

    This is deliberately conservative. Panel authors should return JSON-safe
    state from get_state(), but this prevents one bad value from breaking the
    entire workspace save.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return make_json_safe(asdict(value))

    if isinstance(value, dict):
        return {
            str(make_json_safe(key)): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def get_controller_state(controller: Any) -> dict[str, Any]:
    """
    Optional panel persistence hook.

    Plugin authors do not have to implement anything. If they do implement
    get_state() or snapshot_state(), the returned dict is persisted.
    """
    if controller is None:
        return {}

    state_getter = getattr(controller, "get_state", None)
    if callable(state_getter):
        state = state_getter()
        return make_json_safe(state or {})

    snapshot_getter = getattr(controller, "snapshot_state", None)
    if callable(snapshot_getter):
        state = snapshot_getter()
        return make_json_safe(state or {})

    return {}


def restore_controller_state(controller: Any, state: dict[str, Any] | None) -> None:
    """
    Optional panel restore hook.

    Plugin authors only implement restore_state(state) when their panel has
    internal widget/display state worth restoring.
    """
    if controller is None:
        return

    if not state:
        return

    restore = getattr(controller, "restore_state", None)
    if callable(restore):
        restore(dict(state))


def get_controller_state_version(controller: Any, default: int = 1) -> int:
    if controller is None:
        return default

    for attr in ("state_version", "persistence_version"):
        value = getattr(controller, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return default

    return default