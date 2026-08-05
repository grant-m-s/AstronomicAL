from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .installer import (
    PluginBatchInstallRequest,
    PluginBatchPreflightRequest,
    PluginBatchPreflightResult,
    PluginInstallResult,
)
from .marketplace_download import MarketplacePackageDownloader, VerifiedMarketplacePackage
from .marketplace_planner import InstallPlan, InstallPlanItem


class MarketplaceInstallError(RuntimeError):
    """Raised when an approved marketplace install plan cannot be executed."""


@dataclass(frozen=True)
class MarketplaceInstallExecution:
    plan: InstallPlan
    results: tuple[PluginInstallResult, ...]


class MarketplaceInstallOrchestrator:
    """Execute an already-resolved marketplace install plan.

    The orchestrator translates marketplace plan metadata into installer-owned
    preflight and filesystem transactions. It never imports or enables plugin
    runtime code. All changing releases are downloaded and verified before
    PluginInstaller is asked to commit anything.
    """

    def __init__(
        self,
        *,
        downloader: MarketplacePackageDownloader,
        installer: Any,
    ) -> None:
        self.downloader = downloader
        self.installer = installer

    def preflight(self, plan: InstallPlan) -> PluginBatchPreflightResult:
        """Validate an approved plan before downloading package archives."""

        changes = self._validated_changes(plan)
        if not changes:
            return PluginBatchPreflightResult(python_environment_change=False)

        preflight_batch = getattr(self.installer, "preflight_batch", None)
        if not callable(preflight_batch):
            raise MarketplaceInstallError(
                "PluginInstaller does not support marketplace batch preflight."
            )

        requests = tuple(
            PluginBatchPreflightRequest(
                manifest=item.release.manifest,
                operation=item.action,
                expected_existing_version=(
                    item.existing_version if item.action == "update" else None
                ),
                expected_existing_source=(
                    item.existing_source if item.action == "update" else None
                ),
                expected_existing_source_id=(
                    item.existing_source_id if item.action == "update" else None
                ),
                expected_existing_sha256=(
                    item.existing_sha256 if item.action == "update" else None
                ),
                verify_existing_provenance=item.action == "update",
                allow_downgrade=False,
            )
            for item in changes
        )
        return preflight_batch(requests)

    def execute(self, plan: InstallPlan) -> MarketplaceInstallExecution:
        changes = self._validated_changes(plan)
        if not changes:
            return MarketplaceInstallExecution(plan=plan, results=())

        preflight = self.preflight(plan)
        if not preflight.ok:
            raise MarketplaceInstallError(
                "Marketplace plan is currently blocked. Disable these managed "
                "plugins before continuing: " + ", ".join(preflight.blockers)
            )

        verified: list[tuple[InstallPlanItem, VerifiedMarketplacePackage]] = []
        try:
            for item in changes:
                package = self.downloader.download_and_verify(
                    item.release,
                    expected_plugin_id=item.plugin_id,
                )
                verified.append((item, package))

            requests = tuple(
                PluginBatchInstallRequest(
                    archive_path=package.path,
                    operation=item.action,
                    source="marketplace",
                    source_id=plan.marketplace_id,
                    release_url=item.release.url,
                    expected_existing_version=(
                        item.existing_version if item.action == "update" else None
                    ),
                    expected_existing_source=(
                        item.existing_source if item.action == "update" else None
                    ),
                    expected_existing_source_id=(
                        item.existing_source_id if item.action == "update" else None
                    ),
                    expected_existing_sha256=(
                        item.existing_sha256 if item.action == "update" else None
                    ),
                    verify_existing_provenance=item.action == "update",
                    allow_downgrade=False,
                )
                for item, package in verified
            )
            apply_batch = getattr(self.installer, "apply_batch", None)
            if not callable(apply_batch):
                raise MarketplaceInstallError(
                    "PluginInstaller does not support atomic batch installation."
                )
            results = tuple(apply_batch(requests))
            return MarketplaceInstallExecution(plan=plan, results=results)
        finally:
            for _item, package in reversed(verified):
                package.cleanup()

    @staticmethod
    def _validated_changes(plan: InstallPlan) -> tuple[InstallPlanItem, ...]:
        if not isinstance(plan, InstallPlan):
            raise TypeError("plan must be an InstallPlan.")

        changes = tuple(plan.changes)
        for item in changes:
            if item.action not in {"install", "update"}:
                raise MarketplaceInstallError(
                    f"Unsupported marketplace plan action {item.action!r}."
                )
            if item.release is None:
                raise MarketplaceInstallError(
                    f"Marketplace plan item {item.plugin_id!r} has no release."
                )
            if item.release.manifest.id != item.plugin_id:
                raise MarketplaceInstallError(
                    f"Marketplace plan item {item.plugin_id!r} does not match "
                    f"release manifest id {item.release.manifest.id!r}."
                )
            if item.release.version != item.version:
                raise MarketplaceInstallError(
                    f"Marketplace plan item {item.plugin_id!r} version "
                    f"{item.version!r} does not match release version "
                    f"{item.release.version!r}."
                )
        return changes