from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astronomicAL.platform.missing_panel import create_missing_panel
from astronomicAL.platform.panel_state import make_json_safe
from astronomicAL.utils.debug import (
    persistence_debug_print,
    summarize_dataset_snapshot,
    summarize_pending_restore_items,
    summarize_workspace_snapshot,
)


SCHEMA_NAME = "astronomical.workspace"
SCHEMA_VERSION = 1


class WorkspacePersistence:
    """
    Save/load service for the new plugin workspace.

    This service owns the save/load orchestration. It does not require plugin
    authors to do anything for basic layout restore. Plugin authors only add
    get_state() / restore_state() to their controllers when they want internal
    widget state preserved.
    """

    def __init__(self, context: Any) -> None:
        self.context = context

    def snapshot(self) -> dict[str, Any]:
        return make_json_safe(
            {
                "schema": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "plugins": self._snapshot_plugins(),
                "datasets": self._snapshot_datasets(),
                "selection": self._snapshot_selection(),
                "workspace": self._snapshot_workspace(),
            }
        )

    def save(self, path: str | Path) -> dict[str, Any]:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        snapshot = self.snapshot()

        with path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2, sort_keys=True)

        events = getattr(self.context, "events", None)
        if events is not None:
            events.publish(
                "workspace.saved",
                {
                    "path": str(path),
                    "schema": snapshot.get("schema"),
                    "schema_version": snapshot.get("schema_version"),
                },
            )

        return snapshot

    def load(self, path: str | Path) -> dict[str, Any]:
        path = Path(path).expanduser()

        with path.open("r", encoding="utf-8") as handle:
            snapshot = json.load(handle)

        self._validate_snapshot(snapshot)

        return snapshot

    def restore(self, snapshot: dict[str, Any], *, strict: bool = False) -> list[dict[str, Any]]:
        self._validate_snapshot(snapshot)

        issues: list[dict[str, Any]] = []

        persistence_debug_print(
            "restore start",
            {
                "top_level_keys": list(snapshot.keys()),
                "datasets": summarize_dataset_snapshot(snapshot.get("datasets")),
                "workspace": summarize_workspace_snapshot(snapshot.get("workspace")),
            },
        )

        # Important:
        # Queue restored dataset mappings before plugin panels are restored.
        # Plugin panels may be mapping-gated and need those mappings later when
        # the dataset is actually loaded.
        self._restore_datasets(snapshot.get("datasets", {}) or {})

        issues.extend(self._restore_plugins(snapshot.get("plugins", {}) or {}, strict=strict))

        issues.extend(
            self._restore_workspace(
                snapshot.get("workspace", {}) or {},
                strict=strict,
            )
        )

        self._restore_selection(snapshot.get("selection", {}) or {})

        events = getattr(self.context, "events", None)
        if events is not None:
            events.publish(
                "workspace.restore.completed",
                {
                    "schema": snapshot.get("schema"),
                    "schema_version": snapshot.get("schema_version"),
                    "issues": issues,
                },
            )

        if strict and issues:
            issue_text = "; ".join(issue.get("message", str(issue)) for issue in issues)
            raise RuntimeError(f"Workspace restore completed with issues: {issue_text}")

        return issues

    def _validate_snapshot(self, snapshot: dict[str, Any]) -> None:
        if not isinstance(snapshot, dict):
            raise TypeError("Workspace snapshot must be a dictionary.")

        schema = snapshot.get("schema")
        if schema != SCHEMA_NAME:
            raise ValueError(f"Unsupported workspace schema: {schema!r}")

        version = int(snapshot.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Workspace schema version {version} is newer than supported "
                f"version {SCHEMA_VERSION}."
            )

    def _snapshot_plugins(self) -> dict[str, Any]:
        plugins = getattr(self.context, "plugins", None)
        workspace = getattr(self.context, "workspace", None)

        enabled = []
        plugin_info = {}

        if plugins is not None and hasattr(plugins, "list_plugins"):
            for info in plugins.list_plugins():
                status = getattr(info, "status", None)
                status_value = getattr(status, "value", status)
                plugin_id = getattr(info, "id", None)
                if not plugin_id:
                    continue

                plugin_info[plugin_id] = {
                    "id": plugin_id,
                    "name": getattr(info, "name", None),
                    "version": getattr(info, "version", None),
                    "status": status_value,
                    "source": getattr(info, "source", None),
                }

                if status_value == "enabled":
                    enabled.append(plugin_id)

        required = {}

        if workspace is not None:
            for panel in workspace.snapshot_panels():
                plugin_id = panel.get("plugin_id")
                if not plugin_id:
                    continue

                required.setdefault(
                    plugin_id,
                    {
                        "id": plugin_id,
                        "version": panel.get("plugin_version"),
                        "panels": [],
                    },
                )
                registration_id = panel.get("registration_id")
                if registration_id:
                    required[plugin_id]["panels"].append(registration_id)

        return {
            "enabled": sorted(set(enabled)),
            "required": list(required.values()),
            "info": plugin_info,
        }

    def _snapshot_datasets(self) -> dict[str, Any]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return {}

        snapshot_method = getattr(datasets, "snapshot", None)
        if callable(snapshot_method):
            return snapshot_method()

        return {}

    def _snapshot_selection(self) -> dict[str, Any]:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return {}

        snapshot_method = getattr(selection, "snapshot", None)
        if callable(snapshot_method):
            return snapshot_method()

        return {}

    def _snapshot_workspace(self) -> dict[str, Any]:
        workspace = getattr(self.context, "workspace", None)
        if workspace is None:
            return {}

        snapshot_method = getattr(workspace, "snapshot", None)
        if callable(snapshot_method):
            return snapshot_method()

        return {
            "grid": workspace.snapshot_grid(),
            "panels": workspace.snapshot_panels(),
        }

    def _restore_plugins(
        self,
        plugin_snapshot: dict[str, Any],
        *,
        strict: bool,
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []

        plugins = getattr(self.context, "plugins", None)
        if plugins is None:
            return issues

        if hasattr(plugins, "discover"):
            try:
                plugins.discover()
            except Exception as exc:
                issue = {
                    "type": "plugin_discovery_failed",
                    "message": str(exc),
                }
                issues.append(issue)
                if strict:
                    return issues

        required = plugin_snapshot.get("required", [])
        enabled = set(plugin_snapshot.get("enabled", []))

        required_ids = {
            item.get("id")
            for item in required
            if isinstance(item, dict) and item.get("id")
        }
        plugin_ids_to_enable = sorted(enabled.union(required_ids))

        currently_enabled = set()
        if hasattr(plugins, "list_plugins"):
            for info in plugins.list_plugins():
                status = getattr(info, "status", None)
                status_value = getattr(status, "value", status)
                if status_value == "enabled":
                    currently_enabled.add(getattr(info, "id", None))

        for plugin_id in plugin_ids_to_enable:
            if plugin_id in currently_enabled:
                continue

            try:
                plugins.enable(plugin_id, self.context, validate=False)
            except Exception as exc:
                issue = {
                    "type": "plugin_enable_failed",
                    "plugin_id": plugin_id,
                    "message": str(exc),
                }
                issues.append(issue)
                if strict:
                    return issues

        return issues

    def _restore_datasets(self, dataset_snapshot: dict[str, Any]) -> None:
        datasets = getattr(self.context, "datasets", None)

        persistence_debug_print(
            "restore datasets start",
            {
                "datasets_available": datasets is not None,
                "snapshot": summarize_dataset_snapshot(dataset_snapshot),
            },
        )

        if datasets is None:
            persistence_debug_print("restore datasets skipped: no DatasetManager on context")
            return

        if not isinstance(dataset_snapshot, dict):
            persistence_debug_print("restore datasets skipped: dataset snapshot is not a dict")
            return

        items = [
            dict(item)
            for item in (dataset_snapshot.get("items") or [])
            if isinstance(item, dict)
        ]
        active_id = dataset_snapshot.get("active_id")

        if not hasattr(datasets, "_pending_restore_items"):
            datasets._pending_restore_items = []

        if not hasattr(datasets, "_pending_active_id"):
            datasets._pending_active_id = None

        datasets._pending_active_id = active_id

        restore_method = getattr(datasets, "restore_metadata_snapshot", None)

        persistence_debug_print(
            "restore datasets method",
            {
                "restore_method_available": callable(restore_method),
            },
        )

        if callable(restore_method):
            try:
                restore_method(dataset_snapshot)
            except Exception as exc:
                persistence_debug_print("restore_metadata_snapshot failed", exc)

        existing_dataset_ids = set(getattr(datasets, "_datasets", {}).keys())
        pending = list(getattr(datasets, "_pending_restore_items", []) or [])

        persistence_debug_print(
            "restore datasets before safety net",
            {
                "existing_dataset_ids": sorted(existing_dataset_ids),
                "pending": summarize_pending_restore_items(pending),
            },
        )

        for item in items:
            item_id = item.get("id")
            mappings = dict(item.get("mappings") or {})

            if not mappings and not item.get("meta"):
                continue

            applied = False

            if item_id and str(item_id) in existing_dataset_ids:
                apply_method = getattr(datasets, "_apply_restore_item_to_dataset", None)
                if callable(apply_method):
                    try:
                        applied = bool(apply_method(str(item_id), item))
                        persistence_debug_print(
                            "restore datasets applied immediately",
                            {
                                "dataset_id": item_id,
                                "mappings": mappings,
                            },
                        )
                    except Exception as exc:
                        persistence_debug_print(
                            "restore datasets immediate apply failed",
                            {
                                "dataset_id": item_id,
                                "error": str(exc),
                            },
                        )

            if applied:
                continue

            already_pending = False
            for pending_item in getattr(datasets, "_pending_restore_items", []) or []:
                if (
                    pending_item.get("id") == item.get("id")
                    and pending_item.get("mappings") == item.get("mappings")
                    and pending_item.get("meta") == item.get("meta")
                ):
                    already_pending = True
                    break

            if not already_pending:
                persistence_debug_print(
                    "restore datasets queueing via safety net",
                    {
                        "dataset_id": item.get("id"),
                        "mappings": mappings,
                    },
                )
                datasets._pending_restore_items.append(dict(item))

        persistence_debug_print(
            "restore datasets complete",
            {
                "pending": summarize_pending_restore_items(
                    getattr(datasets, "_pending_restore_items", None)
                ),
            },
        )

    def _restore_workspace(
        self,
        workspace_snapshot: dict[str, Any],
        *,
        strict: bool,
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []

        workspace = getattr(self.context, "workspace", None)
        plugins = getattr(self.context, "plugins", None)

        if workspace is None:
            return issues

        workspace.clear()

        grid_snapshot = workspace_snapshot.get("grid", {})
        panel_snapshots = workspace_snapshot.get("panels", [])

        workspace.apply_grid_snapshot(grid_snapshot)

        for panel_snapshot in panel_snapshots:
            if not isinstance(panel_snapshot, dict):
                continue

            instance_id = str(panel_snapshot.get("instance_id") or "")
            if not instance_id:
                issue = {
                    "type": "panel_restore_failed",
                    "message": "Saved panel is missing instance_id.",
                    "panel": panel_snapshot,
                }
                issues.append(issue)
                if strict:
                    return issues
                continue

            kind = panel_snapshot.get("kind") or "plugin_panel"
            layout_items = self._layout_items_for_panel(grid_snapshot, instance_id)

            if kind != "plugin_panel":
                self._restore_missing_panel(
                    panel_snapshot=panel_snapshot,
                    reason=f"Unsupported panel kind: {kind}",
                    layout_items=layout_items,
                )
                issues.append(
                    {
                        "type": "unsupported_panel_kind",
                        "instance_id": instance_id,
                        "kind": kind,
                        "message": f"Unsupported panel kind: {kind}",
                    }
                )
                continue

            registration_id = panel_snapshot.get("registration_id")
            if not registration_id:
                self._restore_missing_panel(
                    panel_snapshot=panel_snapshot,
                    reason="Saved plugin panel is missing registration_id.",
                    layout_items=layout_items,
                )
                issues.append(
                    {
                        "type": "missing_registration_id",
                        "instance_id": instance_id,
                        "message": "Saved plugin panel is missing registration_id.",
                    }
                )
                continue

            if plugins is None:
                self._restore_missing_panel(
                    panel_snapshot=panel_snapshot,
                    reason="Plugin manager is unavailable.",
                    layout_items=layout_items,
                )
                issues.append(
                    {
                        "type": "plugin_manager_unavailable",
                        "instance_id": instance_id,
                        "registration_id": registration_id,
                        "message": "Plugin manager is unavailable.",
                    }
                )
                continue

            try:
                plugins.open_panel(
                    registration_id,
                    context=self.context,
                    instance_id=instance_id,
                    title=panel_snapshot.get("title"),
                    layout_items=layout_items,
                    restore_state=panel_snapshot.get("state") or {},
                    restore_metadata={
                        "state_version": panel_snapshot.get("state_version", 1),
                        "plugin_version": panel_snapshot.get("plugin_version"),
                        "panel_snapshot": panel_snapshot,
                    },
                    open_kwargs=panel_snapshot.get("open_kwargs") or {},
                )
            except Exception as exc:
                self._restore_missing_panel(
                    panel_snapshot=panel_snapshot,
                    reason=str(exc),
                    layout_items=layout_items,
                )
                issues.append(
                    {
                        "type": "panel_restore_failed",
                        "instance_id": instance_id,
                        "registration_id": registration_id,
                        "message": str(exc),
                    }
                )
                if strict:
                    return issues

        return issues

    def _restore_missing_panel(
        self,
        *,
        panel_snapshot: dict[str, Any],
        reason: str,
        layout_items: dict[str, dict[str, Any]],
    ) -> None:
        workspace = self.context.workspace
        instance_id = str(panel_snapshot.get("instance_id"))
        title = panel_snapshot.get("title") or "Missing panel"
        view, controller = create_missing_panel(reason=reason, snapshot=panel_snapshot)

        workspace.add_panel(
            instance_id,
            view,
            title=title,
            controller=controller,
            layout_items=layout_items,
            kind="missing_panel",
            plugin_id=panel_snapshot.get("plugin_id"),
            registration_id=panel_snapshot.get("registration_id"),
            plugin_version=panel_snapshot.get("plugin_version"),
            state_version=panel_snapshot.get("state_version", 1),
            persistent=True,
            open_kwargs=panel_snapshot.get("open_kwargs") or {},
            metadata={
                "restore_reason": reason,
                "original_panel_snapshot": panel_snapshot,
            },
        )

    def _restore_selection(self, selection_snapshot: dict[str, Any]) -> None:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return

        restore_method = getattr(selection, "restore_snapshot", None)
        if callable(restore_method):
            restore_method(selection_snapshot, publish=True)

    def _layout_items_for_panel(
        self,
        grid_snapshot: dict[str, Any],
        instance_id: str,
    ) -> dict[str, dict[str, Any]]:
        layout_items: dict[str, dict[str, Any]] = {}

        for breakpoint, breakpoint_layout in (grid_snapshot.get("layouts") or {}).items():
            for item in breakpoint_layout or []:
                if str(item.get("i")) != str(instance_id):
                    continue

                item_copy = dict(item)
                item_copy["i"] = str(instance_id)
                layout_items[str(breakpoint)] = item_copy
                break

        return layout_items