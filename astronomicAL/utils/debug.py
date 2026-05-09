from __future__ import annotations

import os
from typing import Any, Mapping, Sequence


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


DEBUG_ALL = _truthy_env("ASTRONOMICAL_DEBUG")
DEBUG_BOOT = _truthy_env("ASTRONOMICAL_DEBUG_BOOT") or DEBUG_ALL
DEBUG_PLUGINS = _truthy_env("ASTRONOMICAL_DEBUG_PLUGINS") or DEBUG_BOOT or DEBUG_ALL
DEBUG_WORKSPACE = _truthy_env("ASTRONOMICAL_DEBUG_WORKSPACE") or DEBUG_BOOT or DEBUG_ALL
DEBUG_PERSISTENCE = _truthy_env("ASTRONOMICAL_DEBUG_PERSISTENCE") or DEBUG_WORKSPACE or DEBUG_ALL
DEBUG_DATASETS = _truthy_env("ASTRONOMICAL_DEBUG_DATASETS") or DEBUG_ALL
DEBUG_MAPPING = _truthy_env("ASTRONOMICAL_DEBUG_MAPPING") or DEBUG_DATASETS or DEBUG_ALL
DEBUG_SELECTION = _truthy_env("ASTRONOMICAL_DEBUG_SELECTION") or DEBUG_ALL


def _debug_print(enabled: bool, prefix: str, *args: Any, **kwargs: Any) -> None:
    if enabled:
        print(prefix, *args, **kwargs)


def boot_print(*args: Any, **kwargs: Any) -> None:
    """Print boot-order diagnostics only when explicitly enabled."""
    _debug_print(DEBUG_BOOT, "[BOOT]", *args, **kwargs)


def plugin_debug_print(*args: Any, **kwargs: Any) -> None:
    """Print plugin diagnostics only when explicitly enabled."""
    _debug_print(DEBUG_PLUGINS, "[plugins]", *args, **kwargs)


def workspace_debug_print(*args: Any, **kwargs: Any) -> None:
    """Print workspace/layout diagnostics only when explicitly enabled."""
    _debug_print(DEBUG_WORKSPACE, "[workspace]", *args, **kwargs)


def persistence_debug_print(*args: Any, **kwargs: Any) -> None:
    """Print save/load workspace persistence diagnostics only when explicitly enabled."""
    _debug_print(DEBUG_PERSISTENCE, "[persistence]", *args, **kwargs)


def dataset_debug_print(*args: Any, **kwargs: Any) -> None:
    """Print DatasetManager diagnostics only when explicitly enabled."""
    _debug_print(DEBUG_DATASETS, "[datasets]", *args, **kwargs)


def mapping_debug_print(*args: Any, **kwargs: Any) -> None:
    """Print semantic mapping diagnostics only when explicitly enabled."""
    _debug_print(DEBUG_MAPPING, "[mapping]", *args, **kwargs)


def selection_debug_print(*args: Any, **kwargs: Any) -> None:
    """Print selection/focus diagnostics only when explicitly enabled."""
    _debug_print(DEBUG_SELECTION, "[selection]", *args, **kwargs)


def compact_list(values: Sequence[Any] | None, *, max_items: int = 12) -> list[Any]:
    """
    Return a compact preview of a list-like object for debug output.
    """
    if values is None:
        return []

    values = list(values)

    if len(values) <= max_items:
        return values

    return [
        *values[:max_items],
        f"... +{len(values) - max_items} more",
    ]


def compact_mapping(mapping: Mapping[str, Any] | None, *, max_items: int = 12) -> dict[str, Any]:
    """
    Return a compact preview of a mapping for debug output.
    """
    if not mapping:
        return {}

    items = list(mapping.items())

    if len(items) <= max_items:
        return dict(items)

    preview = dict(items[:max_items])
    preview["..."] = f"+{len(items) - max_items} more"
    return preview


def summarize_dataset_snapshot(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Compact summary of the dataset section of a workspace snapshot.

    Avoids printing hundreds of column names during normal debugging.
    """
    if not snapshot:
        return {}

    items = []

    for item in snapshot.get("items", []) or []:
        meta = dict(item.get("meta") or {})
        columns = item.get("columns") or meta.get("columns") or []

        items.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "mappings": dict(item.get("mappings") or {}),
                "source_path": meta.get("source_path"),
                "rows": meta.get("rows"),
                "column_count": len(columns),
                "columns_preview": compact_list(columns),
            }
        )

    return {
        "active_id": snapshot.get("active_id"),
        "item_count": len(items),
        "items": items,
    }


def summarize_workspace_snapshot(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Compact summary of the workspace section of a workspace snapshot.
    """
    if not snapshot:
        return {}

    panels = snapshot.get("panels", []) or []
    grid = snapshot.get("grid", {}) or {}

    return {
        "panel_count": len(panels),
        "panels": [
            {
                "instance_id": panel.get("instance_id"),
                "kind": panel.get("kind"),
                "plugin_id": panel.get("plugin_id"),
                "registration_id": panel.get("registration_id"),
                "title": panel.get("title"),
            }
            for panel in panels
        ],
        "grid_keys": list(grid.get("keys") or []),
        "breakpoints": list((grid.get("layouts") or {}).keys()),
    }


def summarize_pending_restore_items(items: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """
    Compact summary of pending dataset restore items.
    """
    if not items:
        return []

    summary = []

    for item in items:
        meta = dict(item.get("meta") or {})
        columns = item.get("columns") or meta.get("columns") or []

        summary.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "mappings": dict(item.get("mappings") or {}),
                "source_path": meta.get("source_path"),
                "rows": meta.get("rows"),
                "column_count": len(columns),
                "columns_preview": compact_list(columns),
            }
        )

    return summary