from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .manifest import PluginManifest, coerce_manifest

MARKETPLACE_SCHEMA_VERSION = 1

_MARKETPLACE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

class MarketplaceCatalogueError(ValueError):
    """Raised when marketplace catalogue data is invalid."""

@dataclass(frozen=True)
class MarketplaceInfo:
    """Identity and presentation metadata for one marketplace source."""

    id: str
    name: str
    homepage: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not _MARKETPLACE_ID_RE.match(self.id):
            raise MarketplaceCatalogueError(
                "Marketplace id must contain only letters, numbers, '.', '_' or '-'."
            )
        if not str(self.name or "").strip():
            raise MarketplaceCatalogueError("Marketplace name is required.")
        if not isinstance(self.metadata, dict):
            raise MarketplaceCatalogueError("Marketplace metadata must be a JSON object.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketplaceInfo":
        if not isinstance(data, dict):
            raise MarketplaceCatalogueError("Marketplace metadata must be a JSON object.")
        return cls(
            id=str(data.get("id", "") or "").strip(),
            name=str(data.get("name", "") or "").strip(),
            homepage=_optional_string(data.get("homepage")),
            metadata=_object_field(data, "metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "homepage": self.homepage,
            "metadata": dict(self.metadata),
        }

@dataclass(frozen=True)
class MarketplaceRelease:
    """Static metadata for one downloadable .alplugin release.

    The embedded manifest is a catalogue snapshot used for compatibility and
    dependency planning. The downloaded archive must still be inspected and its
    manifest compared with this snapshot before installation.
    """

    version: str
    url: str
    sha256: str
    manifest: PluginManifest
    size: Optional[int] = None
    published_at: Optional[str] = None
    yanked: bool = False
    yank_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        version = str(self.version or "").strip()
        url = str(self.url or "").strip()
        sha256 = str(self.sha256 or "").strip().lower()

        if not version:
            raise MarketplaceCatalogueError("Marketplace release version is required.")
        if not url:
            raise MarketplaceCatalogueError("Marketplace release URL is required.")
        if not _SHA256_RE.fullmatch(sha256):
            raise MarketplaceCatalogueError(
                "Marketplace release sha256 must be exactly 64 hexadecimal characters."
            )
        if self.manifest.version != version:
            raise MarketplaceCatalogueError(
                "Marketplace release version does not match embedded manifest version: "
                f"{version!r} != {self.manifest.version!r}."
            )
        if self.size is not None:
            if isinstance(self.size, bool) or not isinstance(self.size, int):
                raise MarketplaceCatalogueError(
                    "Marketplace release size must be an integer when provided."
                )
            if self.size < 0:
                raise MarketplaceCatalogueError(
                    "Marketplace release size must be zero or greater."
                )
        if self.published_at is not None:
            _validate_timestamp(self.published_at)
        if not isinstance(self.yanked, bool):
            raise MarketplaceCatalogueError("Marketplace release yanked must be a boolean.")
        if not isinstance(self.metadata, dict):
            raise MarketplaceCatalogueError(
                "Marketplace release metadata must be a JSON object."
            )

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "sha256", sha256)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketplaceRelease":
        if not isinstance(data, dict):
            raise MarketplaceCatalogueError("Marketplace release must be a JSON object.")

        manifest_data = data.get("manifest")
        if not isinstance(manifest_data, dict):
            raise MarketplaceCatalogueError(
                "Marketplace release manifest must be a JSON object."
            )
        try:
            manifest = coerce_manifest(manifest_data)
        except Exception as exc:
            raise MarketplaceCatalogueError(
                f"Invalid marketplace release manifest: {exc}"
            ) from exc

        size = data.get("size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int)):
            raise MarketplaceCatalogueError(
                "Marketplace release size must be an integer when provided."
            )

        yanked = data.get("yanked", False)
        if not isinstance(yanked, bool):
            raise MarketplaceCatalogueError("Marketplace release yanked must be a boolean.")

        url = str(data.get("url", "") or "").strip()
        _validate_catalogue_release_url(url)

        return cls(
            version=str(data.get("version", "") or "").strip(),
            url=url,
            sha256=str(data.get("sha256", "") or "").strip(),
            manifest=manifest,
            size=size,
            published_at=_optional_string(data.get("published_at")),
            yanked=yanked,
            yank_reason=_optional_string(data.get("yank_reason")),
            metadata=_object_field(data, "metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "url": self.url,
            "sha256": self.sha256,
            "size": self.size,
            "published_at": self.published_at,
            "yanked": self.yanked,
            "yank_reason": self.yank_reason,
            "manifest": self.manifest.to_dict(),
            "metadata": dict(self.metadata),
        }

@dataclass(frozen=True)
class MarketplacePlugin:
    """Marketplace-specific metadata and releases for one plugin id."""

    id: str
    releases: tuple[MarketplaceRelease, ...]
    repository: Optional[str] = None
    documentation: Optional[str] = None
    license: Optional[str] = None
    icon: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise MarketplaceCatalogueError("Marketplace plugin id is required.")
        if not self.releases:
            raise MarketplaceCatalogueError(
                f"Marketplace plugin {self.id!r} must contain at least one release."
            )
        if not isinstance(self.metadata, dict):
            raise MarketplaceCatalogueError(
                f"Marketplace plugin {self.id!r} metadata must be a JSON object."
            )

        seen_versions: set[str] = set()
        for release in self.releases:
            if release.manifest.id != self.id:
                raise MarketplaceCatalogueError(
                    f"Marketplace plugin {self.id!r} contains release "
                    f"{release.version!r} for manifest id {release.manifest.id!r}."
                )
            if release.version in seen_versions:
                raise MarketplaceCatalogueError(
                    f"Marketplace plugin {self.id!r} contains duplicate release "
                    f"version {release.version!r}."
                )
            seen_versions.add(release.version)

    @classmethod
    def from_dict(
        cls,
        plugin_id: str,
        data: Dict[str, Any],
    ) -> "MarketplacePlugin":
        plugin_id = str(plugin_id or "").strip()
        if not plugin_id:
            raise MarketplaceCatalogueError("Marketplace plugin id cannot be empty.")
        if not isinstance(data, dict):
            raise MarketplaceCatalogueError(
                f"Marketplace plugin {plugin_id!r} must be a JSON object."
            )

        release_data = data.get("releases")
        if not isinstance(release_data, list):
            raise MarketplaceCatalogueError(
                f"Marketplace plugin {plugin_id!r} releases must be a JSON array."
            )

        releases: list[MarketplaceRelease] = []
        for index, value in enumerate(release_data):
            try:
                releases.append(MarketplaceRelease.from_dict(value))
            except MarketplaceCatalogueError as exc:
                raise MarketplaceCatalogueError(
                    f"Invalid release {index} for marketplace plugin {plugin_id!r}: {exc}"
                ) from exc

        return cls(
            id=plugin_id,
            releases=tuple(releases),
            repository=_optional_string(data.get("repository")),
            documentation=_optional_string(data.get("documentation")),
            license=_optional_string(data.get("license")),
            icon=_optional_string(data.get("icon")),
            metadata=_object_field(data, "metadata"),
        )

    def get_release(self, version: str) -> MarketplaceRelease:
        version = str(version or "").strip()
        for release in self.releases:
            if release.version == version:
                return release
        raise KeyError(
            f"Marketplace plugin {self.id!r} has no release version {version!r}."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repository": self.repository,
            "documentation": self.documentation,
            "license": self.license,
            "icon": self.icon,
            "releases": [release.to_dict() for release in self.releases],
            "metadata": dict(self.metadata),
        }

@dataclass(frozen=True)
class MarketplaceCatalogue:
    """Parsed, non-executing marketplace catalogue,"""

    marketplace: MarketplaceInfo
    plugins: tuple[MarketplacePlugin, ...]
    schema_version: int = MARKETPLACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MARKETPLACE_SCHEMA_VERSION:
            raise MarketplaceCatalogueError(
                f"Unsupported marketplace schema version {self.schema_version}; "
                f"expected {MARKETPLACE_SCHEMA_VERSION}."
            )

        seen_ids: set[str] = set()
        for plugin in self.plugins:
            if plugin.id in seen_ids:
                raise MarketplaceCatalogueError(
                    f"Marketplace catalogue contains duplicate plugin id {plugin.id!r}."
                )
            seen_ids.add(plugin.id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketplaceCatalogue":
        if not isinstance(data, dict):
            raise MarketplaceCatalogueError(
                "Marketplace catalogue root must be a JSON object."
            )

        schema_version = data.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise MarketplaceCatalogueError(
                "Marketplace catalogue schema_version must be an integer."
            )
        if schema_version != MARKETPLACE_SCHEMA_VERSION:
            raise MarketplaceCatalogueError(
                f"Unsupported marketplace schema version {schema_version}; "
                f"expected {MARKETPLACE_SCHEMA_VERSION}."
            )

        marketplace = MarketplaceInfo.from_dict(data.get("marketplace"))

        plugin_data = data.get("plugins")
        if not isinstance(plugin_data, dict):
            raise MarketplaceCatalogueError(
                "Marketplace catalogue plugins must be a JSON object keyed by plugin id."
            )

        plugins: list[MarketplacePlugin] = []
        for plugin_id, value in plugin_data.items():
            try:
                plugins.append(MarketplacePlugin.from_dict(str(plugin_id), value))
            except MarketplaceCatalogueError as exc:
                raise MarketplaceCatalogueError(
                    f"Invalid marketplace plugin {plugin_id!r}: {exc}"
                ) from exc

        plugins.sort(key=lambda plugin: plugin.id)
        return cls(
            schema_version=schema_version,
            marketplace=marketplace,
            plugins=tuple(plugins),
        )

    def get_plugin(self, plugin_id: str) -> MarketplacePlugin:
        plugin_id = str(plugin_id or "").strip()
        for plugin in self.plugins:
            if plugin.id == plugin_id:
                return plugin
        raise KeyError(f"Unknown marketplace plugin id: {plugin_id!r}")

    def get_release(self, plugin_id: str, version: str) -> MarketplaceRelease:
        return self.get_plugin(plugin_id).get_release(version)

    def list_plugins(self) -> list[MarketplacePlugin]:
        return list(self.plugins)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "marketplace": self.marketplace.to_dict(),
            "plugins": {
                plugin.id: plugin.to_dict()
                for plugin in self.plugins
            },
        }

def _validate_catalogue_release_url(value: str) -> None:
    """Validate a release URL while parsing untrusted catalogue metadata.

    ``MarketplaceRelease`` itself remains constructible with arbitrary URLs so
    lower-level transport tests and programmatic callers can exercise the
    downloader's own security boundary. External catalogue ingestion is stricter:
    release URLs advertised by a marketplace must be absolute HTTPS URLs.
    """

    parsed = urlparse(str(value or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise MarketplaceCatalogueError(
            "Marketplace release URL must be an absolute HTTPS URL."
        )


def _object_field(data: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    value = data.get(field_name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise MarketplaceCatalogueError(
            f"Marketplace field {field_name!r} must be a JSON object."
        )
    return dict(value)

def _optional_string(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise MarketplaceCatalogueError(
            f"Marketplace optional string fields must be strings, got {type(value)!r}."
        )
    value = value.strip()
    return value or None

def _validate_timestamp(value: str) -> None:
    value = str(value or "").strip()
    if not value:
        raise MarketplaceCatalogueError(
            "Marketplace release published_at cannot be empty when provided."
        )

    normalised = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(normalised)
    except ValueError as exc:
        raise MarketplaceCatalogueError(
            f"Marketplace release published_at must be an ISO-8601 timestamp: {value!r}."
        ) from exc