from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

class PluginStateStore:
    """Persistent user activation state for non-bundled plugins."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._lock = threading.RLock()
        self._data = self._default_data()
        self._load_error: str | None = None
        self.reload()

    @classmethod
    def _default_data(cls) -> Dict[str, Any]:
        return {
            "schema_version": cls.SCHEMA_VERSION,
            "community_plugins_enabled": False,
            "plugins": {},
        }

    @property
    def load_error(self) -> str | None:
        with self._lock:
            return self._load_error

    @property
    def community_plugins_enabled(self) -> bool:
        with self._lock:
            return bool(self._data.get("community_plugins_enabled", False))

    def set_community_plugins_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._data["community_plugins_enabled"] = bool(enabled)
            self._save_locked()

    def is_enabled(self, plugin_id: str) -> bool:
        plugin_id = self._normalise_plugin_id(plugin_id)
        with self._lock:
            plugin_state = self._data.get("plugins", {}).get(plugin_id, {})
            return bool(plugin_state.get("enabled", False))

    def set_enabled(self, plugin_id: str, enabled: bool) -> None:
        plugin_id = self._normalise_plugin_id(plugin_id)
        with self._lock:
            plugins = self._data.setdefault("plugins", {})
            plugin_state = plugins.setdefault(plugin_id, {})
            plugin_state["enabled"] = bool(enabled)
            self._save_locked()

    def enabled_plugin_ids(self) -> set[str]:
        with self._lock:
            plugins = self._data.get("plugins", {})
            return {
                str(plugin_id)
                for plugin_id, plugin_state in plugins.items()
                if isinstance(plugin_state, dict) and plugin_state.get("enabled") is True
            }

    def remove_plugin(self, plugin_id: str) -> None:
        """Remove persisted activation preference for an uninstalled plugin."""

        plugin_id = self._normalise_plugin_id(plugin_id)
        with self._lock:
            plugins = self._data.setdefault("plugins", {})
            if plugin_id not in plugins:
                return
            plugins.pop(plugin_id, None)
            self._save_locked()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data))

    def reload(self) -> None:
        with self._lock:
            self._data = self._default_data()
            self._load_error = None

            if not self.path.exists():
                return

            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)

                if not isinstance(data, dict):
                    raise ValueError("Plugin state root must be a JSON object.")

                version = int(data.get("schema_version", 0))
                if version != self.SCHEMA_VERSION:
                    raise ValueError(
                        f"Unsupported plugin state schema version {version}; "
                        f"expected {self.SCHEMA_VERSION}."
                    )

                plugins = data.get("plugins", {})
                if not isinstance(plugins, dict):
                    raise ValueError("Plugin state 'plugins' value must be a JSON object.")

                self._data = {
                    "schema_version": self.SCHEMA_VERSION,
                    "community_plugins_enabled": bool(
                        data.get("community_plugins_enabled", False)
                    ),
                    "plugins": {
                        str(plugin_id): dict(plugin_state)
                        for plugin_id, plugin_state in plugins.items()
                        if isinstance(plugin_state, dict)
                    },
                }
            except Exception as exc:
                self._data = self._default_data()
                self._load_error = str(exc)

    @staticmethod
    def _normalise_plugin_id(plugin_id: str) -> str:
        plugin_id = str(plugin_id or "").strip()
        if not plugin_id:
            raise ValueError("plugin_id cannot be empty.")
        return plugin_id

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")

        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self._data, handle, indent=2, sort_keys=True)
                handle.write("\n")
            temporary.replace(self.path)
            self._load_error = None
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass