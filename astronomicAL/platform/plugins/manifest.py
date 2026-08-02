from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import re

_PLUGIN_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_PLUGIN_SPECIFIER_START = frozenset("<>=!~")


@dataclass(frozen=True)
class PluginRequirement:
    """Parsed AstronomicAL-to-AstronomicAL plugin dependency.

    ``specifier`` intentionally uses the same syntax as ``packaging.specifiers``
    (for example ``">=1.4,<2"``), but parsing the plugin id itself does not import
    ``packaging``. That keeps static manifest discovery cheap and lets the runtime
    perform the actual version comparison only when validation is requested.
    """

    plugin_id: str
    specifier: str = ""

    def __post_init__(self) -> None:
        if not self.plugin_id or not _PLUGIN_ID_RE.match(self.plugin_id):
            raise ValueError(
                "Plugin dependency id must contain only letters, numbers, '.', '_' or '-'."
            )

        specifier = str(self.specifier or "").strip()
        if specifier and specifier[0] not in _PLUGIN_SPECIFIER_START:
            raise ValueError(
                "Plugin dependency version constraints must start with one of "
                "'<', '>', '=', '!', or '~'."
            )

        object.__setattr__(self, "specifier", specifier)

    def __str__(self) -> str:
        return f"{self.plugin_id}{self.specifier}"


def parse_plugin_requirement(value: Any) -> PluginRequirement:
    """Parse a plugin dependency such as ``core.ml>=1.4,<2``.

    Bare plugin ids remain fully supported. Version specifier syntax is validated
    later by the runtime with ``packaging.specifiers.SpecifierSet`` so importing a
    static manifest does not require importing optional runtime libraries.
    """

    if not isinstance(value, str):
        raise ValueError("Plugin dependency requirements must be strings.")

    text = value.strip()
    if not text:
        raise ValueError("Plugin dependency requirement cannot be empty.")

    split_at: int | None = None
    for index, character in enumerate(text):
        if character in _PLUGIN_SPECIFIER_START:
            split_at = index
            break

    if split_at is None:
        plugin_id = text
        specifier = ""
    else:
        plugin_id = text[:split_at].strip()
        specifier = text[split_at:].strip()

    return PluginRequirement(plugin_id=plugin_id, specifier=specifier)


def _validate_plugin_requirement_list(
    field_name: str,
    values: Iterable[Any],
) -> None:
    for value in values:
        try:
            parse_plugin_requirement(value)
        except ValueError as exc:
            raise ValueError(
                f"PluginManifest.{field_name} contains an invalid plugin requirement "
                f"{value!r}: {exc}"
            ) from exc


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

    # AstronomicAL plugin dependencies. Bare ids remain backwards-compatible;
    # versioned requirements use packaging-style specifiers, e.g.
    # ["core.ml>=1.4,<2", "core.visualisation>=1.2"].
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

        if isinstance(self.requires_plugins, str):
            raise ValueError("PluginManifest.requires_plugins must be a list of strings.")
        if isinstance(self.optional_plugins, str):
            raise ValueError("PluginManifest.optional_plugins must be a list of strings.")

        _validate_plugin_requirement_list("requires_plugins", self.requires_plugins)
        _validate_plugin_requirement_list("optional_plugins", self.optional_plugins)

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