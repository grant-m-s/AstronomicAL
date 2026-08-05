from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Iterable, Optional

from .marketplace import MarketplaceCatalogue
from .marketplace_client import (
    MarketplaceCache,
    MarketplaceClient,
    MarketplaceRefreshResult,
    MarketplaceSource,
)
from .marketplace_planner import InstallPlan, PluginInstallPlanner
from .marketplace_updates import MarketplaceUpdateInfo, MarketplaceUpdateService


class MarketplaceServiceError(RuntimeError):
    """Raised when a configured marketplace source cannot satisfy a request."""


@dataclass(frozen=True)
class MarketplaceSourceState:
    source_id: str
    url: str
    enabled: bool
    catalogue_loaded: bool
    last_status: Optional[str] = None
    fetched_at: Optional[float] = None
    error: Optional[str] = None


MarketplaceClientFactory = Callable[
    [MarketplaceSource, MarketplaceCache],
    MarketplaceClient,
]


class MarketplaceService:
    """Host-owned catalogue/cache coordinator.

    Construction and ``load_cached()`` perform no network access. Remote refresh
    is explicit so the application can submit it through JobManager instead of
    blocking startup or the UI thread.
    """

    def __init__(
        self,
        *,
        sources: Iterable[MarketplaceSource],
        cache_root: str | Path,
        client_factory: Optional[MarketplaceClientFactory] = None,
    ) -> None:
        self.cache = MarketplaceCache(cache_root)
        self._lock = threading.RLock()
        self._sources: dict[str, MarketplaceSource] = {}
        self._clients: dict[str, MarketplaceClient] = {}
        self._catalogues: dict[str, MarketplaceCatalogue] = {}
        self._results: dict[str, MarketplaceRefreshResult] = {}
        self._errors: dict[str, str] = {}

        factory = client_factory or self._default_client_factory

        for source in sources:
            if not isinstance(source, MarketplaceSource):
                raise TypeError("sources must contain MarketplaceSource values.")
            if source.id in self._sources:
                raise MarketplaceServiceError(
                    f"Duplicate marketplace source id {source.id!r}."
                )
            self._sources[source.id] = source
            self._clients[source.id] = factory(source, self.cache)

    @staticmethod
    def _default_client_factory(
        source: MarketplaceSource,
        cache: MarketplaceCache,
    ) -> MarketplaceClient:
        return MarketplaceClient(source=source, cache=cache)

    def sources(self) -> list[MarketplaceSource]:
        with self._lock:
            return list(self._sources.values())

    def source(self, source_id: str) -> MarketplaceSource:
        source_id = self._normalise_source_id(source_id)
        with self._lock:
            source = self._sources.get(source_id)
        if source is None:
            raise MarketplaceServiceError(
                f"Unknown marketplace source {source_id!r}."
            )
        return source

    def catalogues(self) -> dict[str, MarketplaceCatalogue]:
        with self._lock:
            return dict(self._catalogues)

    def catalogue(self, source_id: str) -> Optional[MarketplaceCatalogue]:
        source_id = self._normalise_source_id(source_id)
        with self._lock:
            return self._catalogues.get(source_id)

    def require_catalogue(self, source_id: str) -> MarketplaceCatalogue:
        catalogue = self.catalogue(source_id)
        if catalogue is None:
            raise MarketplaceServiceError(
                f"Marketplace source {source_id!r} has no loaded catalogue, "
                "Load its cache or refresh it first."
            )
        return catalogue

    def load_cached(self) -> dict[str, MarketplaceCatalogue]:
        """Load all enabled last-known-good catalogues without network access."""

        with self._lock:
            source_ids = [
                source_id
                for source_id, source in self._sources.items()
                if source.enabled
            ]

        for source_id in source_ids:
            client = self._clients[source_id]
            try:
                catalogue = client.load_cached()
            except Exception as exc:
                with self._lock:
                    self._errors[source_id] = str(exc)
                continue

            with self._lock:
                self._errors.pop(source_id, None)
                if catalogue is not None:
                    self._catalogues[source_id] = catalogue

        return self.catalogues()

    def refresh(
        self,
        source_id: str,
        *,
        allow_stale_cache: bool = True,
    ) -> MarketplaceRefreshResult:
        """Refresh one source explicitly and publish its validated catalogue,"""

        source = self.source(source_id)
        if not source.enabled:
            raise MarketplaceServiceError(
                f"Marketplace source {source.id!r} is disabled."
            )

        client = self._clients[source.id]
        try:
            result = client.refresh(allow_stale_cache=allow_stale_cache)
        except Exception as exc:
            with self._lock:
                self._errors[source.id] = str(exc)
            raise

        with self._lock:
            self._catalogues[source.id] = result.catalogue
            self._results[source.id] = result
            if result.error:
                self._errors[source.id] = result.error
            else:
                self._errors.pop(source.id, None)
        return result

    def states(self) -> list[MarketplaceSourceState]:
        with self._lock:
            states = []
            for source in self._sources.values():
                result = self._results.get(source.id)
                states.append(
                    MarketplaceSourceState(
                        source_id=source.id,
                        url=source.url,
                        enabled=source.enabled,
                        catalogue_loaded=source.id in self._catalogues,
                        last_status=(
                            result.status
                            if result is not None
                            else ("cached" if source.id in self._catalogues else None)
                        ),
                        fetched_at=(
                            result.fetched_at if result is not None else None
                        ),
                        error=self._errors.get(source.id),
                    )
                )
            return states

    @staticmethod
    def _normalise_source_id(source_id: str) -> str:
        source_id = str(source_id or "").strip()
        if not source_id:
            raise MarketplaceServiceError("source_id cannot be empty.")
        return source_id


class MarketplacePlanningService:
    """Create plans from the currently loaded catalogue snapshot."""

    def __init__(
        self,
        *,
        marketplace: MarketplaceService,
        installed_store: Any,
        manager: Any | None = None,
        astronomical_version: str | None = None,
    ) -> None:
        self.marketplace = marketplace
        self.installed_store = installed_store
        self.manager = manager
        self.astronomical_version = astronomical_version

    def plan_install(
        self,
        source_id: str,
        plugin_id: str,
        *,
        version: str | None = None,
        allow_source_replacement: bool = False,
        allow_downgrade: bool = False,
    ) -> InstallPlan:
        planner = PluginInstallPlanner(
            catalogue=self.marketplace.require_catalogue(source_id),
            installed_store=self.installed_store,
            manager=self.manager,
            astronomical_version=self.astronomical_version,
        )
        return planner.plan_install(
            plugin_id,
            version=version,
            allow_source_replacement=allow_source_replacement,
            allow_downgrade=allow_downgrade,
        )


class MarketplaceUpdatesFacade:
    """Dynamic update view over the marketplace service's current catalogues."""

    def __init__(
        self,
        *,
        marketplace: MarketplaceService,
        installed_store: Any,
        manager: Any | None = None,
        astronomical_version: str | None = None,
    ) -> None:
        self.marketplace = marketplace
        self.installed_store = installed_store
        self.manager = manager
        self.astronomical_version = astronomical_version

    def list_updates(self) -> list[MarketplaceUpdateInfo]:
        return self._service().list_updates()

    def check(self, plugin_id: str) -> MarketplaceUpdateInfo:
        return self._service().check(plugin_id)

    def plan_update(self, plugin_id: str) -> InstallPlan:
        return self._service().plan_update(plugin_id)

    def _service(self) -> MarketplaceUpdateService:
        return MarketplaceUpdateService(
            catalogues=self.marketplace.catalogues(),
            installed_store=self.installed_store,
            manager=self.manager,
            astronomical_version=self.astronomical_version,
        )