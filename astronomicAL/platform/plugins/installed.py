from __future__ import annotations

from dataclasses import dataclass
import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .errors import InstalledPluginStoreError


@dataclass(frozen=True)
class InstalledPluginRecord:
    """Persistent metadata for one plugin installed by AstronomicAL."""

    id: str
    name: str
    version: str
    source: str
    sha256: str
    installed_at: str
    directory: str
    archive_name: str
    manifest: Dict[str, Any]
    updated_at: Optional[str] = None
    source_id: Optional[str] = None
    release_url: Optional[str] = None

    @property
    def marketplace_managed(self) -> bool:
        return self.source == "marketplace"

    @classmethod
    def from_dict(
        cls,
        plugin_id: str,
        data: Dict[str, Any],
    ) -> "InstalledPluginRecord":
        plugin_id = str(plugin_id or "").strip()
        if not plugin_id:
            raise InstalledPluginStoreError("Installed plugin id cannot be empty.")
        if not isinstance(data, dict):
            raise InstalledPluginStoreError(
                f"Installed plugin record for {plugin_id!r} must be a JSON object."
            )

        manifest = data.get("manifest", {})
        if not isinstance(manifest, dict):
            raise InstalledPluginStoreError(
                f"Installed plugin manifest for {plugin_id!r} must be a JSON object."
            )

        source = str(data.get("source", "") or "file")
        source_id = cls._optional_string(data.get("source_id"))
        release_url = cls._optional_string(data.get("release_url"))
        cls._validate_provenance(
            plugin_id=plugin_id,
            source=source,
            source_id=source_id,
            release_url=release_url,
        )

        return cls(
            id=plugin_id,
            name=str(data.get("name", "") or plugin_id),
            version=str(data.get("version", "") or ""),
            source=source,
            sha256=str(data.get("sha256", "") or ""),
            installed_at=str(data.get("installed_at", "") or ""),
            directory=str(data.get("directory", "") or plugin_id),
            archive_name=str(data.get("archive_name", "") or ""),
            manifest=dict(manifest),
            updated_at=(
                str(data.get("updated_at"))
                if data.get("updated_at") not in (None, "")
                else None
            ),
            source_id=source_id,
            release_url=release_url,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "sha256": self.sha256,
            "installed_at": self.installed_at,
            "updated_at": self.updated_at,
            "directory": self.directory,
            "archive_name": self.archive_name,
            "manifest": dict(self.manifest),
        }
        if self.source_id is not None:
            data["source_id"] = self.source_id
        if self.release_url is not None:
            data["release_url"] = self.release_url
        return data

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        value = str(value).strip()
        return value or None

    @staticmethod
    def _validate_provenance(
        *,
        plugin_id: str,
        source: str,
        source_id: Optional[str],
        release_url: Optional[str],
    ) -> None:
        if source != "marketplace":
            return
        if source_id is None:
            raise InstalledPluginStoreError(
                f"Marketplace plugin {plugin_id!r} is missing source_id."
            )
        if release_url is None:
            raise InstalledPluginStoreError(
                f"Marketplace plugin {plugin_id!r} is missing release_url."
            )


class InstalledPluginStore:
    """Persistent inventory of plugins installed by AstronomicAL.

    This store intentionally tracks installation state only. Whether a plugin is
    configured or allowed to run belongs to PluginStateStore.
    """

    SCHEMA_VERSION = 2
    LEGACY_SCHEMA_VERSION = 1

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
            "plugins": {},
        }

    @property
    def load_error(self) -> str | None:
        with self._lock:
            return self._load_error

    def require_healthy(self) -> None:
        with self._lock:
            if self._load_error:
                raise InstalledPluginStoreError(
                    "Installed plugin database could not be loaded and will not be "
                    f"overwritten: {self._load_error}"
                )

    def get(self, plugin_id: str) -> InstalledPluginRecord | None:
        plugin_id = self._normalise_plugin_id(plugin_id)
        with self._lock:
            data = self._data.get("plugins", {}).get(plugin_id)
            if not isinstance(data, dict):
                return None
            return InstalledPluginRecord.from_dict(plugin_id, data)

    def contains(self, plugin_id: str) -> bool:
        return self.get(plugin_id) is not None

    def list(self) -> list[InstalledPluginRecord]:
        with self._lock:
            plugins = self._data.get("plugins", {})
            records = [
                InstalledPluginRecord.from_dict(str(plugin_id), data)
                for plugin_id, data in plugins.items()
                if isinstance(data, dict)
            ]
        return sorted(records, key=lambda record: record.id)

    def set(self, record: InstalledPluginRecord) -> None:
        if not isinstance(record, InstalledPluginRecord):
            raise TypeError("record must be an InstalledPluginRecord.")

        plugin_id = self._normalise_plugin_id(record.id)
        InstalledPluginRecord._validate_provenance(
            plugin_id=plugin_id,
            source=record.source,
            source_id=record.source_id,
            release_url=record.release_url,
        )
        with self._lock:
            self._require_healthy_locked()
            plugins = self._data.setdefault("plugins", {})
            plugins[plugin_id] = record.to_dict()
            self._save_locked()

    def set_many(self, records: Iterable[InstalledPluginRecord]) -> None:
        records = list(records)
        normalised: list[tuple[str, InstalledPluginRecord]] = []
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, InstalledPluginRecord):
                raise TypeError("records must contain InstalledPluginRecord values.")
            plugin_id = self._normalise_plugin_id(record.id)
            if plugin_id in seen:
                raise ValueError(f"Duplicate installed plugin id {plugin_id!r}.")
            seen.add(plugin_id)
            InstalledPluginRecord._validate_provenance(
                plugin_id=plugin_id,
                source=record.source,
                source_id=record.source_id,
                release_url=record.release_url,
            )
            normalised.append((plugin_id, record))

        if not normalised:
            return

        with self._lock:
            self._require_healthy_locked()
            previous = json.loads(json.dumps(self._data))
            try:
                plugins = self._data.setdefault("plugins", {})
                for plugin_id, record in normalised:
                    plugins[plugin_id] = record.to_dict()
                self._save_locked()
            except Exception:
                self._data = previous
                raise

    def remove(self, plugin_id: str) -> InstalledPluginRecord | None:
        plugin_id = self._normalise_plugin_id(plugin_id)
        with self._lock:
            self._require_healthy_locked()
            plugins = self._data.setdefault("plugins", {})
            previous = plugins.pop(plugin_id, None)
            if previous is None:
                return None
            self._save_locked()
            return InstalledPluginRecord.from_dict(plugin_id, previous)

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
                    raise ValueError("Installed plugin database root must be a JSON object.")

                version = int(data.get("schema_version", 0))
                supported_versions = {
                    self.LEGACY_SCHEMA_VERSION,
                    self.SCHEMA_VERSION,
                }
                if version not in supported_versions:
                    raise ValueError(
                        f"Unsupported installed plugin schema version {version}; "
                        f"expected one of {sorted(supported_versions)}."
                    )

                plugins = data.get("plugins", {})
                if not isinstance(plugins, dict):
                    raise ValueError(
                        "Installed plugin database 'plugins' value must be a JSON object."
                    )

                normalised: Dict[str, Dict[str, Any]] = {}
                for plugin_id, plugin_data in plugins.items():
                    record = InstalledPluginRecord.from_dict(
                        str(plugin_id),
                        plugin_data,
                    )
                    normalised[record.id] = record.to_dict()

                self._data = {
                    "schema_version": self.SCHEMA_VERSION,
                    "plugins": normalised,
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

    def _require_healthy_locked(self) -> None:
        if self._load_error:
            raise InstalledPluginStoreError(
                "Installed plugin database could not be loaded and will not be "
                f"overwritten: {self._load_error}"
            )

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")

        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self._data, handle, indent=2, sort_keys=True)
                handle.write("\n")
            temporary.replace(self.path)
            self._load_error = None
        except Exception as exc:
            raise InstalledPluginStoreError(
                f"Could not save installed plugin database {self.path}: {exc}"
            ) from exc
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass