from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class PluginOrigin(str, Enum):
    """Trust/source classification assigned by the AstronomicAL host."""

    BUNDLED = "bundled"
    USER = "user"
    DEVELOPMENT = "development"
    ENTRY_POINT = "entry_point"
    RUNTIME = "runtime"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PluginSearchPath:
    """A local plugin root together with host-owned discovery policy."""

    path: Path
    origin: PluginOrigin = PluginOrigin.DEVELOPMENT
    require_static_manifest: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser())

    @classmethod
    def from_any(cls, value: Any) -> "PluginSearchPath":
        if isinstance(value, cls):
            return value
        return cls(path=Path(value).expanduser())