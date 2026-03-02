# astronomicAL/platform/services.py
from __future__ import annotations

from typing import Any, Dict


class ServiceRegistry:
    """
    Small in-memory registry for non-serializable shared services (clients, auth sessions, etc.).
    This is NOT an ArtifactStore. It’s for live objects.
    """

    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._services.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._services[key] = value

    def has(self, key: str) -> bool:
        return key in self._services

    def clear(self) -> None:
        self._services.clear()