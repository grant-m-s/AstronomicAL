from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import logging
import threading


logger = logging.getLogger(__name__)


@dataclass
class ServiceFactory:
    factory: Callable[[], Any]
    lazy: bool = True
    value: Any = None
    initialized: bool = False
    initializing: bool = False
    owner: Optional[str] = None


class ServiceRegistry:
    """Registry for non-serializable shared runtime services.

    Use this for live objects such as API clients, authenticated sessions,
    database connections, filesystem adapters, or remote-service handles.

    It is intentionally not an ArtifactStore. Data products and computed results
    should live in artifacts, not services.
    """

    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, ServiceFactory] = {}
        self._owners: Dict[str, Optional[str]] = {}
        self._lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        factory: Optional[Callable[[], Any]] = None
        reg: Optional[ServiceFactory] = None

        with self._lock:
            if key in self._services:
                return self._services[key]

            reg = self._factories.get(key)
            if reg is None:
                return default

            if reg.initialized:
                self._services[key] = reg.value
                return reg.value

            if reg.initializing:
                raise RuntimeError(
                    f"Service {key!r} is already initializing; "
                    "this usually indicates a circular service dependency."
                )

            reg.initializing = True
            factory = reg.factory

        try:
            value = factory()
        except Exception:
            with self._lock:
                if reg is not None:
                    reg.initializing = False
            raise

        with self._lock:
            if reg is not None:
                reg.value = value
                reg.initialized = True
                reg.initializing = False
            self._services[key] = value
            return value

    def require(self, key: str) -> Any:
        sentinel = object()
        value = self.get(key, default=sentinel)
        if value is sentinel:
            raise KeyError(f"Unknown service: {key}")
        return value

    def set(
        self,
        key: str,
        value: Any,
        *,
        replace: bool = True,
        owner: Optional[str] = None,
    ) -> None:
        with self._lock:
            if not replace and self.has(key):
                raise KeyError(f"Service already exists: {key}")
            self._services[key] = value
            self._factories.pop(key, None)
            self._owners[key] = owner

    def set_factory(
        self,
        key: str,
        factory: Callable[[], Any],
        *,
        lazy: bool = True,
        replace: bool = False,
        owner: Optional[str] = None,
    ) -> None:
        """Register a lazy or eager service factory.

        Parameters
        ----------
        key:
            Stable service key, e.g. ``"euclid.client"``.
        factory:
            Zero-argument callable that returns the live service object.
        lazy:
            If true, construct on first ``get``. If false, construct now.
        replace:
            If false, raise when an existing service/factory uses the same key.
        owner:
            Optional plugin id that owns this service registration.
        """

        with self._lock:
            if not replace and self.has(key):
                raise KeyError(f"Service already exists: {key}")
            self._services.pop(key, None)
            self._factories[key] = ServiceFactory(
                factory=factory,
                lazy=lazy,
                owner=owner,
            )
            self._owners[key] = owner

        if not lazy:
            self.get(key)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._services or key in self._factories

    def owner(self, key: str) -> Optional[str]:
        with self._lock:
            return self._owners.get(key)

    def is_initialized(self, key: str) -> bool:
        with self._lock:
            if key in self._services:
                return True
            reg = self._factories.get(key)
            return bool(reg and reg.initialized)

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(set(self._services) | set(self._factories))

    def remove(
        self,
        key: str,
        *,
        dispose: bool = True,
        owner: Optional[str] = None,
    ) -> Any:
        """Remove a service and optionally dispose/close it.

        If ``owner`` is supplied, the service is removed only if the current owner
        matches. This prevents plugin rollback from deleting a service it did not
        install.

        Returns the removed service object if it had been initialized, otherwise
        returns ``None``.
        """

        with self._lock:
            current_owner = self._owners.get(key)
            if owner is not None and current_owner != owner:
                return None

            value = self._services.pop(key, None)
            reg = self._factories.pop(key, None)
            self._owners.pop(key, None)

            if value is None and reg is not None and reg.initialized:
                value = reg.value

        if dispose and value is not None:
            self._safe_dispose(value)
        return value

    def clear(self, *, dispose: bool = True) -> None:
        with self._lock:
            values = list(self._services.values())
            values.extend(reg.value for reg in self._factories.values() if reg.initialized)
            self._services.clear()
            self._factories.clear()
            self._owners.clear()

        if dispose:
            seen: set[int] = set()
            for value in values:
                marker = id(value)
                if marker in seen:
                    continue
                seen.add(marker)
                self._safe_dispose(value)

    @staticmethod
    def _safe_dispose(value: Any) -> None:
        for method_name in ("dispose", "close", "shutdown"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception as exc:
                    logger.warning("Failed to dispose service %r: %s", value, exc)
                return