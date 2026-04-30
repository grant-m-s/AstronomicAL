from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re

_PLUGIN_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


@dataclass(frozen=True)
class PluginManifest:
    """Small, cheap-to-import description of a plugin.

    Plugin modules should expose a module-level ``manifest`` object of this type.
    Keep this object declarative: do not import heavyweight optional libraries just
    to build the manifest.
    """

    id: str
    name: str
    version: str
    description: str = ""
    author: Optional[str] = None
    homepage: Optional[str] = None
    package: Optional[str] = None

    # AstronomicAL/plugin compatibility.
    min_astronomical: Optional[str] = None
    max_astronomical: Optional[str] = None

    # PEP 508 requirement strings, e.g. ["scikit-learn>=1.4", "astropy"].
    requires: List[str] = field(default_factory=list)
    optional_requires: List[str] = field(default_factory=list)

    # AstronomicAL plugin dependencies.
    requires_plugins: List[str] = field(default_factory=list)
    optional_plugins: List[str] = field(default_factory=list)

    # Free-form discoverability metadata.
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not _PLUGIN_ID_RE.match(self.id):
            raise ValueError(
                "PluginManifest.id must be a stable id containing only letters, "
                "numbers, '.', '_' or '-'."
            )
        if not self.name:
            raise ValueError("PluginManifest.name is required.")
        if not self.version:
            raise ValueError("PluginManifest.version is required.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginManifest":
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "homepage": self.homepage,
            "package": self.package,
            "min_astronomical": self.min_astronomical,
            "max_astronomical": self.max_astronomical,
            "requires": list(self.requires),
            "optional_requires": list(self.optional_requires),
            "requires_plugins": list(self.requires_plugins),
            "optional_plugins": list(self.optional_plugins),
            "capabilities": list(self.capabilities),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


def coerce_manifest(obj: Any) -> PluginManifest:
    """Convert supported manifest forms into a PluginManifest."""

    if isinstance(obj, PluginManifest):
        return obj
    if isinstance(obj, dict):
        return PluginManifest.from_dict(obj)
    raise TypeError(
        "Plugin manifest must be a PluginManifest or dict compatible with "
        "PluginManifest."
    )