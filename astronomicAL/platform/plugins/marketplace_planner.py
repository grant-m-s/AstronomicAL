from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from .manifest import (
    PluginManifest,
    coerce_manifest,
    parse_plugin_requirement,
)
from .marketplace import (
    MarketplaceCatalogue,
    MarketplacePlugin,
    MarketplaceRelease,
)

try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except Exception:
    SpecifierSet = None  # type: ignore[assignment]
    Version = None  # type: ignore[assignment]

class InstallPlanningError(RuntimeError):
    """Raised when no safe, compatible marketplace install plan exists."""

@dataclass(frozen=True)
class InstallPlanItem:
    """One plugin participating in a resolved install plan."""

    plugin_id: str
    version: str
    action: str
    source: str
    required_by: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    python_requirements: tuple[str, ...] = ()
    existing_version: Optional[str] = None
    existing_source: Optional[str] = None
    existing_source_id: Optional[str] = None
    existing_sha256: Optional[str] = None
    release: Optional[MarketplaceRelease] = None

    @property
    def changes_installation(self) -> bool:
        return self.action in {"install", "update"}

@dataclass(frozen=True)
class InstallPlan:
    """Deterministic, read-only resolution of one marketplace request."""

    marketplace_id: str
    requested_plugin_id: str
    requested_version: Optional[str]
    items: tuple[InstallPlanItem, ...]

    @property
    def changes(self) -> tuple[InstallPlanItem, ...]:
        return tuple(item for item in self.items if item.changes_installation)

    @property
    def satisfied(self) -> tuple[InstallPlanItem, ...]:
        return tuple(item for item in self.items if item.action == "satisfied")

@dataclass(frozen=True)
class _RequirementConstraint:
    owner_id: Optional[str]
    text: str
    specifier: str

@dataclass(frozen=True)
class _ExistingPlugin:
    id: str
    version: str
    manifest: PluginManifest
    source: str
    managed: bool
    source_id: Optional[str] = None
    sha256: Optional[str] = None

@dataclass(frozen=True)
class _Candidate:
    id: str
    version: str
    manifest: PluginManifest
    source: str
    release: Optional[MarketplaceRelease] = None
    existing: Optional[_ExistingPlugin] = None

@dataclass
class _SolverState:
    constraints: Dict[str, list[_RequirementConstraint]]
    required_by: Dict[str, set[str]]
    assignments: Dict[str, _Candidate]

    def clone(self) -> "_SolverState":
        return _SolverState(
            constraints={
                plugin_id: list(values)
                for plugin_id, values in self.constraints.items()
            },
            required_by={
                plugin_id: set(values)
                for plugin_id, values in self.required_by.items()
            },
            assignments=dict(self.assignments),
        )

class PluginInstallPlanner:
    """Resolve marketplace plugin dependencies without changing runtime state.

    The planner operates only on catalogue metadata, installed records and
    discovered plugin metadata. It never downloads packages, mutates the plugin
    store, changes the Python environment or enables/disables plugins.
    """

    def __init__(
        self,
        *,
        catalogue: MarketplaceCatalogue,
        installed_store: Any | None = None,
        manager: Any | None = None,
        astronomical_version: str | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.installed_store = installed_store
        self.manager = manager
        self.astronomical_version = (
            str(astronomical_version).strip()
            if astronomical_version not in (None, "")
            else None
        )
        self._require_packaging()
        self._existing = self._build_existing_inventory()

    def plan_install(
        self,
        plugin_id: str,
        *,
        version: str | None = None,
        allow_source_replacement: bool = False,
        allow_downgrade: bool = False,
    ) -> InstallPlan:
        plugin_id = str(plugin_id or "").strip()
        requested_version = (
            str(version).strip()
            if version not in (None, "")
            else None
        )
        if not plugin_id:
            raise InstallPlanningError("plugin_id cannot be empty.")

        try:
            self.catalogue.get_plugin(plugin_id)
        except KeyError as exc:
            raise InstallPlanningError(
                f"Plugin {plugin_id!r} is not available from marketplace "
                f"{self.catalogue.marketplace.id!r}."
            ) from exc

        state = _SolverState(
            constraints={
                plugin_id: [
                    _RequirementConstraint(
                        owner_id=None,
                        text=(
                            f"{plugin_id}=={requested_version}"
                            if requested_version
                            else plugin_id
                        ),
                        specifier=(
                            f"=={requested_version}"
                            if requested_version
                            else ""
                        ),
                    )
                ]
            },
            required_by={plugin_id: set()},
            assignments={},
        )

        solved = self._search(
            state,
            root_plugin_id=plugin_id,
            requested_version=requested_version,
            allow_source_replacement=allow_source_replacement,
            allow_downgrade=allow_downgrade,
        )
        if solved is None:
            raise self._resolution_error(
                state,
                root_plugin_id=plugin_id,
                requested_version=requested_version,
                allow_source_replacement=allow_source_replacement,
                allow_downgrade=allow_downgrade,
            )

        order = self._dependency_order(solved.assignments)
        items = tuple(
            self._plan_item(
                solved,
                solved.assignments[current_id],
            )
            for current_id in order
        )
        return InstallPlan(
            marketplace_id=self.catalogue.marketplace.id,
            requested_plugin_id=plugin_id,
            requested_version=requested_version,
            items=items,
        )

    def _search(
        self,
        state: _SolverState,
        *,
        root_plugin_id: str,
        requested_version: Optional[str],
        allow_source_replacement: bool,
        allow_downgrade: bool,
    ) -> Optional[_SolverState]:
        if not self._assigned_constraints_are_valid(state):
            return None

        unresolved = sorted(
            plugin_id
            for plugin_id in state.constraints
            if plugin_id not in state.assignments
        )
        if not unresolved:
            return state

        # Prefer the most constrained dependency first; tie-break by id so
        # catalogue insertion order never changes the selected plan.
        ranked: list[tuple[int, str, list[_Candidate]]] = []
        for plugin_id in unresolved:
            candidates = self._candidates_for(
                plugin_id,
                state.constraints.get(plugin_id, []),
                is_root=(plugin_id == root_plugin_id),
                requested_version=(
                    requested_version
                    if plugin_id == root_plugin_id
                    else None
                ),
                allow_source_replacement=allow_source_replacement,
                allow_downgrade=allow_downgrade,
            )
            if not candidates:
                return None
            ranked.append((len(candidates), plugin_id, candidates))

        _, plugin_id, candidates = min(
            ranked,
            key=lambda value: (value[0], value[1]),
        )

        for candidate in candidates:
            branch = state.clone()
            branch.assignments[plugin_id] = candidate

            try:
                self._add_candidate_dependencies(branch, candidate)
            except InstallPlanningError:
                continue

            solved = self._search(
                branch,
                root_plugin_id=root_plugin_id,
                requested_version=requested_version,
                allow_source_replacement=allow_source_replacement,
                allow_downgrade=allow_downgrade,
            )
            if solved is not None:
                return solved

        return None

    def _candidates_for(
        self,
        plugin_id: str,
        constraints: Iterable[_RequirementConstraint],
        *,
        is_root: bool,
        requested_version: Optional[str],
        allow_source_replacement: bool,
        allow_downgrade: bool,
    ) -> list[_Candidate]:
        constraints = list(constraints)
        existing = self._existing.get(plugin_id)

        if is_root:
            releases = self._marketplace_releases(
                plugin_id,
                constraints,
                requested_version=requested_version,
                allow_yanked=bool(requested_version),
            )
            return [
                self._marketplace_candidate(release)
                for release in releases
                if self._release_can_replace_existing(
                    release,
                    existing,
                    allow_source_replacement=allow_source_replacement,
                    allow_downgrade=allow_downgrade,
                )
            ]

        candidates: list[_Candidate] = []
        if existing is not None:
            candidate = _Candidate(
                id=existing.id,
                version=existing.version,
                manifest=existing.manifest,
                source=existing.source,
                existing=existing,
            )
            if self._candidate_satisfies(candidate, constraints):
                candidates.append(candidate)

            # A non-managed discovery source cannot be replaced by the managed
            # marketplace installer. A managed plugin may be upgraded only when
            # marketplace provenance is compatible or replacement is explicit.
            if not existing.managed:
                return candidates
            if not self._may_replace_existing_source(
                existing,
                allow_source_replacement=allow_source_replacement,
            ):
                return candidates

        releases = self._marketplace_releases(
            plugin_id,
            constraints,
            requested_version=None,
            allow_yanked=False,
        )
        for release in releases:
            if existing is not None and release.version == existing.version:
                continue
            if (
                existing is not None
                and not allow_downgrade
                and self._is_downgrade(release.version, existing.version)
            ):
                continue
            candidates.append(self._marketplace_candidate(release))

        return candidates

    def _marketplace_releases(
        self,
        plugin_id: str,
        constraints: Iterable[_RequirementConstraint],
        *,
        requested_version: Optional[str],
        allow_yanked: bool,
    ) -> list[MarketplaceRelease]:
        try:
            plugin = self.catalogue.get_plugin(plugin_id)
        except KeyError:
            return []

        constraints = list(constraints)
        releases: list[MarketplaceRelease] = []
        invalid_versions: list[str] = []

        for release in plugin.releases:
            if requested_version is not None and release.version != requested_version:
                continue
            if release.yanked and not allow_yanked:
                continue
            if not self._manifest_host_compatible(release.manifest):
                continue

            candidate = self._marketplace_candidate(release)
            if not self._candidate_satisfies(candidate, constraints):
                continue

            try:
                Version(release.version)
            except Exception:
                invalid_versions.append(release.version)
                continue
            releases.append(release)

        if requested_version is not None and invalid_versions:
            raise InstallPlanningError(
                f"Requested marketplace release {plugin_id}=={requested_version} "
                "does not use a PEP 440-compatible version."
            )

        releases.sort(
            key=lambda release: (
                Version(release.version),
                release.url,
            ),
            reverse=True,
        )
        return releases

    def _add_candidate_dependencies(
        self,
        state: _SolverState,
        candidate: _Candidate,
    ) -> None:
        for raw_requirement in candidate.manifest.requires_plugins:
            try:
                requirement = parse_plugin_requirement(raw_requirement)
            except ValueError as exc:
                raise InstallPlanningError(
                    f"Plugin {candidate.id!r} declares invalid required plugin "
                    f"requirement {raw_requirement!r}: {exc}"
                ) from exc

            # Validate the specifier using the same packaging syntax as the
            # existing PluginInstaller before it participates in resolution.
            self._validate_specifier(
                requirement.specifier,
                owner_id=candidate.id,
                requirement_text=str(requirement),
            )

            state.constraints.setdefault(
                requirement.plugin_id,
                [],
            ).append(
                _RequirementConstraint(
                    owner_id=candidate.id,
                    text=str(requirement),
                    specifier=requirement.specifier,
                )
            )
            state.required_by.setdefault(
                requirement.plugin_id,
                set(),
            ).add(candidate.id)

    def _assigned_constraints_are_valid(
        self,
        state: _SolverState,
    ) -> bool:
        for plugin_id, candidate in state.assignments.items():
            if not self._candidate_satisfies(
                candidate,
                state.constraints.get(plugin_id, []),
            ):
                return False
        return True

    def _candidate_satisfies(
        self,
        candidate: _Candidate,
        constraints: Iterable[_RequirementConstraint],
    ) -> bool:
        for constraint in constraints:
            if not constraint.specifier:
                continue
            self._validate_specifier(
                constraint.specifier,
                owner_id=constraint.owner_id,
                requirement_text=constraint.text,
            )
            try:
                if Version(candidate.version) not in SpecifierSet(
                    constraint.specifier
                ):
                    return False
            except Exception as exc:
                raise InstallPlanningError(
                    f"Cannot compare plugin version {candidate.version!r} "
                    f"against requirement {constraint.text!r}: {exc}"
                ) from exc
        return True

    def _manifest_host_compatible(
        self,
        manifest: PluginManifest,
    ) -> bool:
        if not manifest.min_astronomical and not manifest.max_astronomical:
            return True
        if not self.astronomical_version:
            return False

        try:
            current = Version(self.astronomical_version)
            minimum = (
                Version(manifest.min_astronomical)
                if manifest.min_astronomical
                else None
            )
            maximum = (
                Version(manifest.max_astronomical)
                if manifest.max_astronomical
                else None
            )
        except Exception as exc:
            raise InstallPlanningError(
                f"Invalid AstronomicAL compatibility version for plugin "
                f"{manifest.id!r}: {exc}"
            ) from exc

        if minimum is not None and current < minimum:
            return False
        if maximum is not None and current > maximum:
            return False
        return True

    def _release_can_replace_existing(
        self,
        release: MarketplaceRelease,
        existing: Optional[_ExistingPlugin],
        *,
        allow_source_replacement: bool,
        allow_downgrade: bool,
    ) -> bool:
        if existing is None or existing.version == release.version:
            return True
        if not existing.managed:
            return False
        if not allow_downgrade and self._is_downgrade(
            release.version,
            existing.version,
        ):
            return False
        return self._may_replace_existing_source(
            existing,
            allow_source_replacement=allow_source_replacement,
        )

    @staticmethod
    def _is_downgrade(candidate_version: str, existing_version: str) -> bool:
        try:
            return Version(candidate_version) < Version(existing_version)
        except Exception as exc:
            raise InstallPlanningError(
                f"Cannot compare plugin versions {existing_version!r} and "
                f"{candidate_version!r}: {exc}"
            ) from exc

    def _may_replace_existing_source(
        self,
        existing: _ExistingPlugin,
        *,
        allow_source_replacement: bool,
    ) -> bool:
        if allow_source_replacement:
            return True
        return (
            existing.source == "marketplace"
            and existing.source_id == self.catalogue.marketplace.id
        )

    def _build_existing_inventory(self) -> Dict[str, _ExistingPlugin]:
        inventory: Dict[str, _ExistingPlugin] = {}

        store = self.installed_store
        if store is not None:
            try:
                records = store.list()
            except Exception as exc:
                raise InstallPlanningError(
                    f"Could not read installed plugin inventory: {exc}"
                ) from exc

            for record in records:
                plugin_id = str(getattr(record, "id", "") or "").strip()
                if not plugin_id:
                    continue
                try:
                    manifest = coerce_manifest(record.manifest)
                except Exception as exc:
                    raise InstallPlanningError(
                        f"Installed plugin {plugin_id!r} has invalid stored "
                        f"manifest metadata: {exc}"
                    ) from exc

                inventory[plugin_id] = _ExistingPlugin(
                    id=plugin_id,
                    version=str(record.version),
                    manifest=manifest,
                    source=str(getattr(record, "source", "") or "file"),
                    managed=True,
                    source_id=(
                        str(record.source_id).strip()
                        if getattr(record, "source_id", None)
                        else None
                    ),
                    sha256=(
                        str(record.sha256).strip().lower()
                        if getattr(record, "sha256", None)
                        else None
                    ),
                )

        manager = self.manager
        if manager is not None:
            try:
                infos = manager.list_plugins()
            except Exception as exc:
                raise InstallPlanningError(
                    f"Could not read discovered plugin inventory: {exc}"
                ) from exc

            for info in infos:
                plugin_id = str(getattr(info, "id", "") or "").strip()
                if not plugin_id or plugin_id in inventory:
                    continue

                version = str(getattr(info, "version", "") or "").strip()
                name = str(getattr(info, "name", "") or plugin_id)
                if not version:
                    continue

                try:
                    manifest = PluginManifest(
                        id=plugin_id,
                        name=name,
                        version=version,
                        description=str(
                            getattr(info, "description", "") or ""
                        ),
                        requires=[
                            str(value)
                            for value in (
                                getattr(info, "requires", None) or []
                            )
                        ],
                        optional_requires=[
                            str(value)
                            for value in (
                                getattr(info, "optional_requires", None) or []
                            )
                        ],
                        requires_plugins=[
                            str(value)
                            for value in (
                                getattr(info, "requires_plugins", None) or []
                            )
                        ],
                        capabilities=[
                            str(value)
                            for value in (
                                getattr(info, "capabilities", None) or []
                            )
                        ],
                        tags=[
                            str(value)
                            for value in (
                                getattr(info, "tags", None) or []
                            )
                        ],
                    )
                except Exception as exc:
                    raise InstallPlanningError(
                        f"Discovered plugin {plugin_id!r} has invalid metadata: {exc}"
                    ) from exc

                origin = getattr(info, "origin", None)
                origin_value = getattr(origin, "value", origin)
                source = (
                    str(origin_value).strip()
                    if origin_value not in (None, "")
                    else "discovered"
                )
                inventory[plugin_id] = _ExistingPlugin(
                    id=plugin_id,
                    version=version,
                    manifest=manifest,
                    source=source,
                    managed=False,
                )

        return inventory

    def _dependency_order(
        self,
        assignments: Dict[str, _Candidate],
    ) -> list[str]:
        graph: Dict[str, set[str]] = {
            plugin_id: set()
            for plugin_id in assignments
        }
        for plugin_id, candidate in assignments.items():
            for raw_requirement in candidate.manifest.requires_plugins:
                requirement = parse_plugin_requirement(raw_requirement)
                if requirement.plugin_id in assignments:
                    graph[plugin_id].add(requirement.plugin_id)

        visiting: list[str] = []
        visited: set[str] = set()
        order: list[str] = []

        def visit(plugin_id: str) -> None:
            if plugin_id in visited:
                return
            if plugin_id in visiting:
                start = visiting.index(plugin_id)
                cycle = visiting[start:] + [plugin_id]
                raise InstallPlanningError(
                    "Plugin dependency cycle detected: "
                    + " -> ".join(cycle)
                )

            visiting.append(plugin_id)
            try:
                for dependency_id in sorted(graph[plugin_id]):
                    visit(dependency_id)
            finally:
                visiting.pop()

            visited.add(plugin_id)
            order.append(plugin_id)

        for plugin_id in sorted(graph):
            visit(plugin_id)
        return order

    def _plan_item(
        self,
        state: _SolverState,
        candidate: _Candidate,
    ) -> InstallPlanItem:
        existing = self._existing.get(candidate.id)

        if existing is None:
            action = "install"
            source = "marketplace"
            existing_version = None
            existing_source = None
            existing_source_id = None
            existing_sha256 = None
        elif existing.version == candidate.version:
            action = "satisfied"
            source = existing.source
            existing_version = existing.version
            existing_source = existing.source
            existing_source_id = existing.source_id
            existing_sha256 = existing.sha256
        else:
            action = "update"
            source = "marketplace"
            existing_version = existing.version
            existing_source = existing.source
            existing_source_id = existing.source_id
            existing_sha256 = existing.sha256

        return InstallPlanItem(
            plugin_id=candidate.id,
            version=candidate.version,
            action=action,
            source=source,
            required_by=tuple(
                sorted(state.required_by.get(candidate.id, set()))
            ),
            requirements=tuple(
                constraint.text
                for constraint in state.constraints.get(candidate.id, [])
                if constraint.owner_id is not None
            ),
            python_requirements=tuple(candidate.manifest.requires),
            existing_version=existing_version,
            existing_source=existing_source,
            existing_source_id=existing_source_id,
            existing_sha256=existing_sha256,
            release=candidate.release,
        )

    def _resolution_error(
        self,
        state: _SolverState,
        *,
        root_plugin_id: str,
        requested_version: Optional[str],
        allow_source_replacement: bool,
        allow_downgrade: bool,
    ) -> InstallPlanningError:
        root = self.catalogue.get_plugin(root_plugin_id)
        if requested_version is not None:
            try:
                release = root.get_release(requested_version)
            except KeyError:
                return InstallPlanningError(
                    f"Marketplace plugin {root_plugin_id!r} has no release "
                    f"version {requested_version!r}."
                )

            if not self._manifest_host_compatible(release.manifest):
                if (
                    release.manifest.min_astronomical
                    or release.manifest.max_astronomical
                ) and not self.astronomical_version:
                    return InstallPlanningError(
                        f"Cannot evaluate AstronomicAL compatibility for "
                        f"{root_plugin_id}=={requested_version} because the host "
                        "version is unknown."
                    )
                return InstallPlanningError(
                    f"Marketplace release {root_plugin_id}=={requested_version} "
                    "is not compatible with this AstronomicAL version."
                )

        existing = self._existing.get(root_plugin_id)
        if (
            requested_version is not None
            and existing is not None
            and existing.managed
            and not allow_downgrade
            and self._is_downgrade(requested_version, existing.version)
        ):
            return InstallPlanningError(
                f"Refusing to plan a downgrade of plugin {root_plugin_id!r} "
                f"from {existing.version!r} to {requested_version!r}. "
                "Pass allow_downgrade=True only if intentional."
            )
        if (
            existing is not None
            and not allow_source_replacement
            and (
                not existing.managed
                or existing.source != "marketplace"
                or existing.source_id != self.catalogue.marketplace.id
            )
        ):
            return InstallPlanningError(
                f"Plugin {root_plugin_id!r} already exists at version "
                f"{existing.version!r} from source {existing.source!r}. "
                "Marketplace replacement requires explicit "
                "allow_source_replacement=True."
            )

        details: list[str] = []
        for plugin_id, constraints in sorted(state.constraints.items()):
            if plugin_id == root_plugin_id:
                continue
            requirements = ", ".join(
                constraint.text
                for constraint in constraints
            )
            details.append(f"{plugin_id}: {requirements}")

        suffix = (
            " Required constraints: " + "; ".join(details)
            if details
            else ""
        )
        return InstallPlanningError(
            f"No compatible dependency plan exists for marketplace plugin "
            f"{root_plugin_id!r}.{suffix}"
        )

    @staticmethod
    def _marketplace_candidate(
        release: MarketplaceRelease,
    ) -> _Candidate:
        return _Candidate(
            id=release.manifest.id,
            version=release.version,
            manifest=release.manifest,
            source="marketplace",
            release=release,
        )

    @staticmethod
    def _validate_specifier(
        specifier: str,
        *,
        owner_id: Optional[str],
        requirement_text: str,
    ) -> None:
        if not specifier:
            return
        try:
            SpecifierSet(specifier)
        except Exception as exc:
            owner = (
                f"Plugin {owner_id!r}"
                if owner_id
                else "Requested plugin"
            )
            raise InstallPlanningError(
                f"{owner} declares invalid plugin requirement "
                f"{requirement_text!r}: {exc}"
            ) from exc

    @staticmethod
    def _require_packaging() -> None:
        if Version is None or SpecifierSet is None:
            raise InstallPlanningError(
                "Marketplace dependency planning requires the packaging library."
            )