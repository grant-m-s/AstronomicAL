from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

from .marketplace import MarketplaceCatalogue, MarketplaceRelease
from .marketplace_planner import (
    InstallPlan,
    InstallPlanningError,
    PluginInstallPlanner,
)

try:
    from packaging.version import Version
except Exception:
    Version = None  # type: ignore[assignment]


class MarketplaceUpdateError(RuntimeError):
    """Raised when marketplace update state cannot be evaluated safely."""


class MarketplaceUpdateStatus(str, Enum):
    UP_TO_DATE = "up_to_date"
    UPDATE_AVAILABLE = "update_available"
    NO_COMPATIBLE_UPDATE = "no_compatible_update"
    SOURCE_UNAVAILABLE = "source_unavailable"
    PLUGIN_REMOVED = "plugin_removed"
    CURRENT_RELEASE_YANKED = "current_release_yanked"


@dataclass(frozen=True)
class MarketplaceUpdateInfo:
    """Read-only update state for one marketplace-managed plugin."""

    plugin_id: str
    installed_version: str
    source_id: str
    status: MarketplaceUpdateStatus
    current_release: Optional[MarketplaceRelease] = None
    newest_release: Optional[MarketplaceRelease] = None
    target_release: Optional[MarketplaceRelease] = None
    plan: Optional[InstallPlan] = None
    detail: Optional[str] = None

    @property
    def update_available(self) -> bool:
        return self.target_release is not None and self.plan is not None

    @property
    def target_version(self) -> Optional[str]:
        if self.target_release is None:
            return None
        return self.target_release.version

    @property
    def current_release_yanked(self) -> bool:
        return bool(self.current_release and self.current_release.yanked)


class MarketplaceUpdateService:
    """Discover updates for marketplace-managed installed plugins.

    Discovery is read-only. It does not download packages, change the installed
    plugin store, reconcile Python dependencies or enable/disable plugins.

    Each installed plugin is evaluated only against the marketplace recorded in
    its provenance. Dependencies may still be satisfied by already installed or
    discovered plugins through PluginInstallPlanner's normal inventory.
    """

    def __init__(
        self,
        *,
        catalogues: Mapping[str, MarketplaceCatalogue] | Iterable[MarketplaceCatalogue],
        installed_store: Any,
        manager: Any | None = None,
        astronomical_version: str | None = None,
    ) -> None:
        self.catalogues = self._normalise_catalogues(catalogues)
        self.installed_store = installed_store
        self.manager = manager
        self.astronomical_version = (
            str(astronomical_version).strip()
            if astronomical_version not in (None, "")
            else None
        )
        if Version is None:
            raise MarketplaceUpdateError(
                "Cannot evaluate marketplace updates because packaging is not installed."
            )

    def list_updates(self) -> list[MarketplaceUpdateInfo]:
        """Return update state for every marketplace-managed installation."""

        try:
            records = self.installed_store.list()
        except Exception as exc:
            raise MarketplaceUpdateError(
                f"Could not read installed plugin inventory: {exc}"
            ) from exc

        results = [
            self._check_record(record)
            for record in records
            if str(getattr(record, "source", "") or "") == "marketplace"
        ]
        return sorted(results, key=lambda item: item.plugin_id)

    def check(self, plugin_id: str) -> MarketplaceUpdateInfo:
        """Return current marketplace update state for one managed plugin."""

        plugin_id = str(plugin_id or "").strip()
        if not plugin_id:
            raise MarketplaceUpdateError("plugin_id cannot be empty.")

        try:
            record = self.installed_store.get(plugin_id)
        except Exception as exc:
            raise MarketplaceUpdateError(
                f"Could not read installed plugin {plugin_id!r}: {exc}"
            ) from exc

        if record is None:
            raise MarketplaceUpdateError(
                f"Plugin {plugin_id!r} is not an AstronomicAL-managed install."
            )
        if str(getattr(record, "source", "") or "") != "marketplace":
            raise MarketplaceUpdateError(
                f"Plugin {plugin_id!r} was not installed from a marketplace."
            )

        return self._check_record(record)

    def plan_update(self, plugin_id: str) -> InstallPlan:
        """Return the exact read-only install plan for the recommended update."""

        info = self.check(plugin_id)
        if info.plan is None or info.target_release is None:
            raise MarketplaceUpdateError(
                f"No executable marketplace update is available for "
                f"{plugin_id!r}; status is {info.status.value!r}."
            )
        return info.plan

    def _check_record(self, record: Any) -> MarketplaceUpdateInfo:
        plugin_id = str(getattr(record, "id", "") or "").strip()
        installed_version = str(
            getattr(record, "version", "") or ""
        ).strip()
        source_id = str(
            getattr(record, "source_id", "") or ""
        ).strip()

        if not plugin_id:
            raise MarketplaceUpdateError(
                "Marketplace-installed record has an empty plugin id."
            )
        if not installed_version:
            raise MarketplaceUpdateError(
                f"Marketplace-installed plugin {plugin_id!r} has no version."
            )
        if not source_id:
            raise MarketplaceUpdateError(
                f"Marketplace-installed plugin {plugin_id!r} has no source_id."
            )

        catalogue = self.catalogues.get(source_id)
        if catalogue is None:
            return MarketplaceUpdateInfo(
                plugin_id=plugin_id,
                installed_version=installed_version,
                source_id=source_id,
                status=MarketplaceUpdateStatus.SOURCE_UNAVAILABLE,
                detail=(
                    f"Marketplace source {source_id!r} is not currently configured "
                    "or its catalogue is unavailable."
                ),
            )

        try:
            marketplace_plugin = catalogue.get_plugin(plugin_id)
        except KeyError:
            return MarketplaceUpdateInfo(
                plugin_id=plugin_id,
                installed_version=installed_version,
                source_id=source_id,
                status=MarketplaceUpdateStatus.PLUGIN_REMOVED,
                detail=(
                    f"Plugin {plugin_id!r} is no longer present in marketplace "
                    f"{source_id!r}."
                ),
            )

        try:
            current_version = Version(installed_version)
        except Exception as exc:
            return MarketplaceUpdateInfo(
                plugin_id=plugin_id,
                installed_version=installed_version,
                source_id=source_id,
                status=MarketplaceUpdateStatus.NO_COMPATIBLE_UPDATE,
                detail=(
                    f"Installed plugin version {installed_version!r} is not "
                    f"PEP 440-compatible: {exc}"
                ),
            )

        current_release = self._find_release(
            marketplace_plugin.releases,
            installed_version,
        )
        newest_release = self._newest_parseable_release(
            marketplace_plugin.releases,
        )
        newer_release_exists = self._has_newer_release(
            marketplace_plugin.releases,
            current_version,
        )

        planner = PluginInstallPlanner(
            catalogue=catalogue,
            installed_store=self.installed_store,
            manager=self.manager,
            astronomical_version=self.astronomical_version,
        )

        plan: Optional[InstallPlan] = None
        target_release: Optional[MarketplaceRelease] = None
        planning_error: Optional[str] = None

        try:
            candidate_plan = planner.plan_install(plugin_id)
            root_item = next(
                item
                for item in candidate_plan.items
                if item.plugin_id == plugin_id
            )
            if root_item.action == "update" and root_item.release is not None:
                plan = candidate_plan
                target_release = root_item.release
        except (InstallPlanningError, StopIteration) as exc:
            planning_error = str(exc)

        if current_release is not None and current_release.yanked:
            detail = (
                current_release.yank_reason
                or f"Installed release {plugin_id}=={installed_version} is yanked."
            )
            if planning_error:
                detail = f"{detail} No replacement plan is currently available: {planning_error}"
            return MarketplaceUpdateInfo(
                plugin_id=plugin_id,
                installed_version=installed_version,
                source_id=source_id,
                status=MarketplaceUpdateStatus.CURRENT_RELEASE_YANKED,
                current_release=current_release,
                newest_release=newest_release,
                target_release=target_release,
                plan=plan,
                detail=detail,
            )

        if target_release is not None and plan is not None:
            return MarketplaceUpdateInfo(
                plugin_id=plugin_id,
                installed_version=installed_version,
                source_id=source_id,
                status=MarketplaceUpdateStatus.UPDATE_AVAILABLE,
                current_release=current_release,
                newest_release=newest_release,
                target_release=target_release,
                plan=plan,
            )

        if newer_release_exists:
            detail = (
                planning_error
                or "Newer marketplace releases exist, but none produce a safe "
                "compatible install plan for the current AstronomicAL environment."
            )
            return MarketplaceUpdateInfo(
                plugin_id=plugin_id,
                installed_version=installed_version,
                source_id=source_id,
                status=MarketplaceUpdateStatus.NO_COMPATIBLE_UPDATE,
                current_release=current_release,
                newest_release=newest_release,
                detail=detail,
            )

        if planning_error and current_release is None:
            return MarketplaceUpdateInfo(
                plugin_id=plugin_id,
                installed_version=installed_version,
                source_id=source_id,
                status=MarketplaceUpdateStatus.NO_COMPATIBLE_UPDATE,
                current_release=None,
                newest_release=newest_release,
                detail=(
                    "The installed release is no longer listed and no safe "
                    f"replacement plan is available: {planning_error}"
                ),
            )

        return MarketplaceUpdateInfo(
            plugin_id=plugin_id,
            installed_version=installed_version,
            source_id=source_id,
            status=MarketplaceUpdateStatus.UP_TO_DATE,
            current_release=current_release,
            newest_release=newest_release,
        )

    @staticmethod
    def _normalise_catalogues(
        catalogues: Mapping[str, MarketplaceCatalogue] | Iterable[MarketplaceCatalogue],
    ) -> dict[str, MarketplaceCatalogue]:
        if isinstance(catalogues, Mapping):
            values = list(catalogues.items())
        else:
            values = [
                (catalogue.marketplace.id, catalogue)
                for catalogue in catalogues
            ]

        normalised: dict[str, MarketplaceCatalogue] = {}
        for source_id, catalogue in values:
            if not isinstance(catalogue, MarketplaceCatalogue):
                raise TypeError(
                    "catalogues must contain MarketplaceCatalogue values."
                )
            source_id = str(source_id or "").strip()
            if not source_id:
                raise MarketplaceUpdateError(
                    "Marketplace catalogue source id cannot be empty."
                )
            if source_id != catalogue.marketplace.id:
                raise MarketplaceUpdateError(
                    f"Marketplace catalogue key {source_id!r} does not match "
                    f"catalogue id {catalogue.marketplace.id!r}."
                )
            if source_id in normalised:
                raise MarketplaceUpdateError(
                    f"Duplicate marketplace catalogue id {source_id!r}."
                )
            normalised[source_id] = catalogue
        return normalised

    @staticmethod
    def _find_release(
        releases: Iterable[MarketplaceRelease],
        version: str,
    ) -> Optional[MarketplaceRelease]:
        for release in releases:
            if release.version == version:
                return release
        return None

    @staticmethod
    def _newest_parseable_release(
        releases: Iterable[MarketplaceRelease],
    ) -> Optional[MarketplaceRelease]:
        parsed: list[tuple[Any, str, MarketplaceRelease]] = []
        for release in releases:
            try:
                version = Version(release.version)
            except Exception:
                continue
            parsed.append((version, release.url, release))

        if not parsed:
            return None
        parsed.sort(key=lambda value: (value[0], value[1]), reverse=True)
        return parsed[0][2]

    @staticmethod
    def _has_newer_release(
        releases: Iterable[MarketplaceRelease],
        current_version: Any,
    ) -> bool:
        for release in releases:
            try:
                if Version(release.version) > current_version:
                    return True
            except Exception:
                continue
        return False