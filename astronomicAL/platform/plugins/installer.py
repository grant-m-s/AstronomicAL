from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata as importlib_metadata
from pathlib import Path
import shutil
import threading
from typing import Any, Dict, Iterable, Optional, Sequence
import uuid

from .errors import (
    PluginInstallError,
    PluginPackageError,
    PluginPythonEnvironmentError,
    PluginValidationError,
)
from .installed import InstalledPluginRecord, InstalledPluginStore
from .manifest import PluginManifest, coerce_manifest, parse_plugin_requirement
from .package import PluginPackageInspection, extract_plugin_package, inspect_plugin_package

try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except Exception:
    SpecifierSet = None  # type: ignore[assignment]
    Version = None  # type: ignore[assignment]

@dataclass(frozen=True)
class PluginInstallResult:
    """Result of one committed plugin filesystem transaction."""

    operation: str
    plugin_id: str
    version: str
    path: Path
    sha256: str = ""
    warnings: tuple[str, ...] = ()

@dataclass(frozen=True)
class _DependencyNode:
    id: str
    version: str
    requires_plugins: tuple[str, ...]

class _NoopPythonEnvironmentTransaction:
    changed = False

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def finalize(self) -> None:
        pass


def detect_astronomical_version() -> str | None:
    """Best-effort version lookup without hard-coding a distribution name.

    Installed distributions are mapped back to the top-level ``astronomicAL``
    package. Source checkouts that are not installed as a distribution may not
    have a detectable version; callers can pass an explicit version instead.
    """

    try:
        distributions = importlib_metadata.packages_distributions().get("astronomicAL", [])
    except Exception:
        distributions = []

    for distribution_name in distributions:
        try:
            return importlib_metadata.version(distribution_name)
        except Exception:
            continue

    return None

class PluginInstaller:
    """Transactional local installer for static-manifest community plugins.

    Installation never imports ``plugin.py`` and never enables a plugin. Runtime
    activation remains owned by PluginActivationService / PluginManager.
    """

    def __init__(
        self,
        *,
        store: InstalledPluginStore,
        plugin_dir: str | Path,
        manager: Any | None = None,
        python_environment: Any | None = None,
        staging_dir: str | Path | None = None,
        backup_dir: str | Path | None = None,
        astronomical_version: str | None = None,
    ) -> None:
        self.store = store
        self.plugin_dir = Path(plugin_dir).expanduser()
        base_dir = self.plugin_dir.parent
        self.staging_dir = Path(
            staging_dir or (base_dir / "plugin-staging")
        ).expanduser()
        self.backup_dir = Path(
            backup_dir or (base_dir / "plugin-backups")
        ).expanduser()
        self.manager = manager
        self.python_environment = python_environment
        self.astronomical_version = (
            str(astronomical_version).strip()
            if astronomical_version not in (None, "")
            else detect_astronomical_version()
        )
        self._lock = threading.RLock()

    def inspect(self, archive_path: str | Path) -> PluginPackageInspection:
        """Inspect package structure and compatibility without changing disk state."""

        inspection = inspect_plugin_package(archive_path)
        self._validate_host_compatibility(inspection.manifest)
        self._validate_dependency_graph(inspection.manifest)
        self._validate_python_requirement_declarations(inspection.manifest)
        return inspection

    def install(self, archive_path: str | Path) -> PluginInstallResult:
        """Install a new .alplugin package, leaving it disabled."""

        with self._lock:
            self.store.require_healthy()
            inspection = self.inspect(archive_path)
            manifest = inspection.manifest
            plugin_id = manifest.id
            target = self._target_path(plugin_id)

            if self.store.contains(plugin_id):
                raise PluginInstallError(
                    f"Plugin {plugin_id!r} is already managed by AstronomicAL. "
                    "Use update() to replace it."
                )
            if target.exists():
                raise PluginInstallError(
                    f"Plugin destination already exists but is not registered as an "
                    f"AstronomicAL-managed install: {target}"
                )

            requirements = self._managed_python_requirements(candidate=manifest)
            python_changed = self._python_environment_needs_reconcile(requirements)
            if python_changed:
                self._assert_python_environment_change_safe()
            python_tx = self._prepare_python_environment(requirements)

            stage = self._new_transaction_path(self.staging_dir, plugin_id, "install")
            target_created = False
            success = False

            try:
                extract_plugin_package(inspection, stage)
                if python_tx.changed:
                    self._purge_python_environment_modules()

                self.plugin_dir.mkdir(parents=True, exist_ok=True)
                stage.replace(target)
                target_created = True

                python_tx.commit()

                now = _utc_now()
                record = InstalledPluginRecord(
                    id=plugin_id,
                    name=manifest.name,
                    version=manifest.version,
                    source="file",
                    sha256=inspection.sha256,
                    installed_at=now,
                    updated_at=None,
                    directory=plugin_id,
                    archive_name=inspection.archive_path.name,
                    manifest=manifest.to_dict(),
                )
                self.store.set(record)
                success = True

            except Exception as exc:
                python_tx.rollback()
                if target_created and target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                if isinstance(exc, PluginInstallError):
                    raise
                raise PluginInstallError(
                    f"Failed to install plugin {plugin_id!r}: {exc}"
                ) from exc
            finally:
                if stage.exists():
                    shutil.rmtree(stage, ignore_errors=True)
                if success:
                    python_tx.finalize()

            warnings = self._refresh_manager_after_install()
            return PluginInstallResult(
                operation="install",
                plugin_id=plugin_id,
                version=manifest.version,
                path=target,
                sha256=inspection.sha256,
                warnings=tuple(warnings),
            )

    def update(
        self,
        archive_path: str | Path,
        *,
        allow_downgrade: bool = False,
    ) -> PluginInstallResult:
        """Replace an existing managed plugin transactionally.

        The plugin must not be running. Python dependencies are reconciled as one
        shared host-compatible overlay before the update is committed.
        """

        with self._lock:
            self.store.require_healthy()
            inspection = inspect_plugin_package(archive_path)
            manifest = inspection.manifest
            plugin_id = manifest.id

            existing = self.store.get(plugin_id)
            if existing is None:
                raise PluginInstallError(
                    f"Plugin {plugin_id!r} is not an AstronomicAL-managed install. "
                    "Use install() for a new plugin."
                )

            self._assert_not_runtime_enabled(plugin_id)
            self._validate_host_compatibility(manifest)
            self._validate_dependency_graph(manifest)
            self._validate_python_requirement_declarations(manifest)
            self._validate_reverse_dependencies(
                plugin_id,
                proposed_version=manifest.version,
            )
            self._validate_update_version(
                existing.version,
                manifest.version,
                allow_downgrade=allow_downgrade,
            )

            target = self._target_path(plugin_id)
            if not target.is_dir():
                raise PluginInstallError(
                    f"Managed plugin directory is missing: {target}"
                )

            requirements = self._managed_python_requirements(candidate=manifest)
            python_changed = self._python_environment_needs_reconcile(requirements)
            if python_changed:
                self._assert_python_environment_change_safe()
            python_tx = self._prepare_python_environment(requirements)

            stage = self._new_transaction_path(self.staging_dir, plugin_id, "update")
            backup = self._new_transaction_path(self.backup_dir, plugin_id, "backup")
            backup_created = False
            replacement_created = False
            success = False

            try:
                extract_plugin_package(inspection, stage)
                if python_tx.changed:
                    self._purge_python_environment_modules()

                backup.parent.mkdir(parents=True, exist_ok=True)
                target.replace(backup)
                backup_created = True

                stage.replace(target)
                replacement_created = True

                python_tx.commit()

                record = InstalledPluginRecord(
                    id=plugin_id,
                    name=manifest.name,
                    version=manifest.version,
                    source="file",
                    sha256=inspection.sha256,
                    installed_at=existing.installed_at,
                    updated_at=_utc_now(),
                    directory=plugin_id,
                    archive_name=inspection.archive_path.name,
                    manifest=manifest.to_dict(),
                )
                self.store.set(record)
                success = True

            except Exception as exc:
                python_tx.rollback()
                if replacement_created and target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                if backup_created and backup.exists() and not target.exists():
                    try:
                        backup.replace(target)
                    except Exception:
                        pass
                if isinstance(exc, PluginInstallError):
                    raise
                raise PluginInstallError(
                    f"Failed to update plugin {plugin_id!r}: {exc}"
                ) from exc
            finally:
                if stage.exists():
                    shutil.rmtree(stage, ignore_errors=True)
                if success:
                    python_tx.finalize()

            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)

            warnings = self._refresh_manager_after_replace(plugin_id)
            return PluginInstallResult(
                operation="update",
                plugin_id=plugin_id,
                version=manifest.version,
                path=target,
                sha256=inspection.sha256,
                warnings=tuple(warnings),
            )

    def uninstall(
        self,
        plugin_id: str,
        *,
        context: Any | None = None,
    ) -> PluginInstallResult:
        """Uninstall managed plugin code while preserving separate plugin data."""

        plugin_id = str(plugin_id or "").strip()
        if not plugin_id:
            raise PluginInstallError("plugin_id cannot be empty.")

        with self._lock:
            self.store.require_healthy()
            existing = self.store.get(plugin_id)
            if existing is None:
                raise PluginInstallError(
                    f"Plugin {plugin_id!r} is not an AstronomicAL-managed install."
                )

            self._validate_reverse_dependencies(plugin_id, proposed_version=None)

            requirements = self._managed_python_requirements(remove_plugin_id=plugin_id)
            python_changed = self._python_environment_needs_reconcile(requirements)
            if python_changed:
                # The target may still be enabled here; uninstall() can disable it
                # safely below. Other managed plugins must not be running while the
                # shared dependency overlay changes underneath the process.
                self._assert_python_environment_change_safe(
                    exclude_plugin_ids={plugin_id}
                )
            python_tx = self._prepare_python_environment(requirements)

            if self._is_runtime_enabled(plugin_id):
                if context is None:
                    python_tx.rollback()
                    raise PluginInstallError(
                        f"Plugin {plugin_id!r} is currently enabled. Disable it before "
                        "uninstalling, or provide AppContext so the installer can disable it."
                    )

                activation = getattr(context, "plugin_activation", None)
                disable = getattr(activation, "disable", None)
                if not callable(disable):
                    python_tx.rollback()
                    raise PluginInstallError(
                        "AppContext is missing PluginActivationService; cannot safely "
                        f"disable {plugin_id!r} before uninstall."
                    )
                disable(plugin_id, context=context)

            target = self._target_path(plugin_id)
            if not target.is_dir():
                python_tx.rollback()
                raise PluginInstallError(
                    f"Managed plugin directory is missing: {target}"
                )

            backup = self._new_transaction_path(self.backup_dir, plugin_id, "uninstall")
            backup.parent.mkdir(parents=True, exist_ok=True)
            moved = False
            success = False

            try:
                if python_tx.changed:
                    self._purge_python_environment_modules()

                target.replace(backup)
                moved = True

                python_tx.commit()

                removed = self.store.remove(plugin_id)
                if removed is None:
                    raise PluginInstallError(
                        f"Installed plugin record disappeared during uninstall: {plugin_id}"
                    )
                success = True

            except Exception as exc:
                python_tx.rollback()
                if moved and backup.exists() and not target.exists():
                    try:
                        backup.replace(target)
                    except Exception:
                        pass
                if isinstance(exc, PluginInstallError):
                    raise
                raise PluginInstallError(
                    f"Failed to uninstall plugin {plugin_id!r}: {exc}"
                ) from exc
            finally:
                if success:
                    python_tx.finalize()

            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)

            plugin_state = getattr(context, "plugin_state", None) if context is not None else None
            remove_state = getattr(plugin_state, "remove_plugin", None)
            if callable(remove_state):
                try:
                    remove_state(plugin_id)
                except Exception:
                    # The code uninstall is already committed. Do not pretend it rolled back
                    # because stale activation preference is safe: the plugin no longer exists.
                    pass

            warnings = self._refresh_manager_after_replace(plugin_id)
            return PluginInstallResult(
                operation="uninstall",
                plugin_id=plugin_id,
                version=existing.version,
                path=target,
                sha256=existing.sha256,
                warnings=tuple(warnings),
            )

    def _validate_python_requirement_declarations(
        self,
        manifest: PluginManifest,
    ) -> None:
        if not manifest.requires:
            return
        environment = self.python_environment
        validator = getattr(environment, "validate_requirements", None)
        if not callable(validator):
            raise PluginInstallError(
                f"Plugin {manifest.id!r} declares Python dependencies, but AstronomicAL's "
                "managed plugin Python environment is not configured."
            )
        validator(manifest.id, manifest.requires)

    def _managed_python_requirements(
        self,
        *,
        candidate: PluginManifest | None = None,
        remove_plugin_id: str | None = None,
    ) -> Dict[str, tuple[str, ...]]:
        requirements: Dict[str, tuple[str, ...]] = {}
        remove_plugin_id = str(remove_plugin_id or "").strip()

        for record in self.store.list():
            if remove_plugin_id and record.id == remove_plugin_id:
                continue
            try:
                manifest = coerce_manifest(record.manifest)
            except Exception as exc:
                raise PluginInstallError(
                    f"Cannot reconcile Python dependencies because installed plugin "
                    f"{record.id!r} has invalid stored manifest metadata: {exc}"
                ) from exc
            requirements[manifest.id] = tuple(str(item) for item in manifest.requires)

        if candidate is not None:
            requirements[candidate.id] = tuple(
                str(item) for item in candidate.requires
            )
        return requirements

    def _python_environment_needs_reconcile(
        self,
        requirements: Dict[str, tuple[str, ...]],
    ) -> bool:
        environment = self.python_environment
        if environment is None:
            return any(requirements.values())
        checker = getattr(environment, "needs_reconcile", None)
        if not callable(checker):
            return any(requirements.values())
        return bool(checker(requirements))

    def _prepare_python_environment(
        self,
        requirements: Dict[str, tuple[str, ...]],
    ) -> Any:
        environment = self.python_environment
        if environment is None:
            if any(requirements.values()):
                raise PluginInstallError(
                    "Managed community-plugin Python dependencies are not configured."
                )
            return _NoopPythonEnvironmentTransaction()

        prepare = getattr(environment, "prepare", None)
        if not callable(prepare):
            if any(requirements.values()):
                raise PluginInstallError(
                    "Managed community-plugin Python dependency service does not "
                    "support reconciliation."
                )
            return _NoopPythonEnvironmentTransaction()
        return prepare(requirements)

    def _assert_python_environment_change_safe(
        self,
        *,
        exclude_plugin_ids: set[str] | None = None,
    ) -> None:
        if self.manager is None:
            return
        excluded = set(exclude_plugin_ids or ())
        managed_ids = {record.id for record in self.store.list()}
        blockers: list[str] = []
        try:
            infos = self.manager.list_plugins()
        except Exception:
            infos = []

        for info in infos:
            plugin_id = str(getattr(info, "id", "") or "").strip()
            if not plugin_id or plugin_id in excluded or plugin_id not in managed_ids:
                continue
            status = getattr(info, "status", None)
            if getattr(status, "value", status) == "enabled":
                blockers.append(plugin_id)

        if blockers:
            raise PluginInstallError(
                "The shared community-plugin Python dependency environment must not "
                "change while managed plugins are running. Disable these managed "
                "plugins first: " + ", ".join(sorted(blockers))
            )

    def _purge_python_environment_modules(self) -> None:
        environment = self.python_environment
        purge = getattr(environment, "purge_loaded_overlay_modules", None)
        if callable(purge):
            purge()

    def _target_path(self, plugin_id: str) -> Path:
        target = self.plugin_dir / plugin_id
        if target.parent != self.plugin_dir:
            raise PluginInstallError(f"Unsafe plugin id for installation path: {plugin_id!r}")
        return target

    @staticmethod
    def _new_transaction_path(root: Path, plugin_id: str, operation: str) -> Path:
        token = uuid.uuid4().hex
        return root / f"{plugin_id}.{operation}.{token}"

    def _validate_host_compatibility(self, manifest: PluginManifest) -> None:
        if not manifest.min_astronomical and not manifest.max_astronomical:
            return

        if not self.astronomical_version:
            raise PluginInstallError(
                f"Plugin {manifest.id!r} declares AstronomicAL compatibility bounds, "
                "but the host application version could not be detected. Pass "
                "astronomical_version explicitly when constructing PluginInstaller."
            )
        if Version is None:
            raise PluginInstallError(
                "Cannot validate AstronomicAL/plugin compatibility because packaging "
                "is not installed."
            )

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
            raise PluginInstallError(
                f"Invalid AstronomicAL compatibility version for plugin "
                f"{manifest.id!r}: {exc}"
            ) from exc

        if minimum is not None and current < minimum:
            raise PluginInstallError(
                f"Plugin {manifest.id!r} requires AstronomicAL >= {minimum}; "
                f"current version is {current}."
            )
        if maximum is not None and current > maximum:
            raise PluginInstallError(
                f"Plugin {manifest.id!r} requires AstronomicAL <= {maximum}; "
                f"current version is {current}."
            )

    def _dependency_inventory(
        self,
        candidate: PluginManifest,
    ) -> Dict[str, _DependencyNode]:
        nodes: Dict[str, _DependencyNode] = {}

        for record in self.store.list():
            try:
                manifest = coerce_manifest(record.manifest)
            except Exception:
                continue
            nodes[manifest.id] = _DependencyNode(
                id=manifest.id,
                version=manifest.version,
                requires_plugins=tuple(manifest.requires_plugins),
            )

        manager = self.manager
        if manager is not None:
            try:
                infos = manager.list_plugins()
            except Exception:
                infos = []

            for info in infos:
                plugin_id = str(getattr(info, "id", "") or "").strip()
                if not plugin_id:
                    continue
                nodes[plugin_id] = _DependencyNode(
                    id=plugin_id,
                    version=str(getattr(info, "version", "") or ""),
                    requires_plugins=tuple(
                        str(item)
                        for item in (getattr(info, "requires_plugins", None) or [])
                    ),
                )

        nodes[candidate.id] = _DependencyNode(
            id=candidate.id,
            version=candidate.version,
            requires_plugins=tuple(candidate.requires_plugins),
        )
        return nodes

    def _validate_dependency_graph(self, candidate: PluginManifest) -> None:
        nodes = self._dependency_inventory(candidate)
        visiting: list[str] = []
        visited: set[str] = set()

        def visit(plugin_id: str) -> None:
            if plugin_id in visited:
                return
            if plugin_id in visiting:
                start = visiting.index(plugin_id)
                cycle = visiting[start:] + [plugin_id]
                raise PluginInstallError(
                    "Plugin dependency cycle detected: " + " -> ".join(cycle)
                )

            node = nodes.get(plugin_id)
            if node is None:
                raise PluginInstallError(
                    f"Missing required AstronomicAL plugin: {plugin_id}"
                )

            visiting.append(plugin_id)
            try:
                for raw_requirement in node.requires_plugins:
                    try:
                        requirement = parse_plugin_requirement(raw_requirement)
                    except ValueError as exc:
                        raise PluginInstallError(
                            f"Plugin {plugin_id!r} declares invalid required plugin "
                            f"requirement {raw_requirement!r}: {exc}"
                        ) from exc

                    required = nodes.get(requirement.plugin_id)
                    if required is None:
                        raise PluginInstallError(
                            f"Plugin {plugin_id!r} requires {str(requirement)!r}, "
                            f"but {requirement.plugin_id!r} is not installed/discovered."
                        )

                    self._assert_version_satisfies(
                        owner_plugin_id=plugin_id,
                        requirement_text=str(requirement),
                        required_plugin_id=requirement.plugin_id,
                        discovered_version=required.version,
                        specifier=requirement.specifier,
                    )
                    visit(requirement.plugin_id)
            finally:
                visiting.pop()

            visited.add(plugin_id)

        visit(candidate.id)

    def _validate_reverse_dependencies(
        self,
        plugin_id: str,
        *,
        proposed_version: str | None,
    ) -> None:
        owners: Dict[str, Sequence[str]] = {}

        for record in self.store.list():
            if record.id == plugin_id:
                continue
            try:
                manifest = coerce_manifest(record.manifest)
            except Exception:
                continue
            owners[record.id] = tuple(manifest.requires_plugins)

        if self.manager is not None:
            try:
                infos = self.manager.list_plugins()
            except Exception:
                infos = []
            for info in infos:
                owner_id = str(getattr(info, "id", "") or "").strip()
                if not owner_id or owner_id == plugin_id:
                    continue
                owners[owner_id] = tuple(
                    str(item)
                    for item in (getattr(info, "requires_plugins", None) or [])
                )

        blockers: list[str] = []
        incompatibilities: list[str] = []

        for owner_id, requirements in sorted(owners.items()):
            for raw_requirement in requirements:
                try:
                    requirement = parse_plugin_requirement(raw_requirement)
                except ValueError:
                    continue
                if requirement.plugin_id != plugin_id:
                    continue

                if proposed_version is None:
                    blockers.append(f"{owner_id} requires {str(requirement)}")
                    continue

                if requirement.specifier:
                    try:
                        compatible = self._version_matches(
                            proposed_version,
                            requirement.specifier,
                        )
                    except PluginInstallError as exc:
                        raise PluginInstallError(
                            f"Cannot validate dependent plugin {owner_id!r}: {exc}"
                        ) from exc
                    if not compatible:
                        incompatibilities.append(
                            f"{owner_id} requires {str(requirement)}"
                        )

        if blockers:
            raise PluginInstallError(
                f"Cannot uninstall plugin {plugin_id!r}; installed/discovered "
                "plugins still depend on it: " + "; ".join(blockers)
            )

        if incompatibilities:
            raise PluginInstallError(
                f"Cannot update plugin {plugin_id!r} to version {proposed_version}; "
                "the proposed version would break: " + "; ".join(incompatibilities)
            )

    def _assert_version_satisfies(
        self,
        *,
        owner_plugin_id: str,
        requirement_text: str,
        required_plugin_id: str,
        discovered_version: str,
        specifier: str,
    ) -> None:
        if not specifier:
            return

        if not self._version_matches(discovered_version, specifier):
            raise PluginInstallError(
                f"Plugin {owner_plugin_id!r} requires {requirement_text!r}, "
                f"but discovered/installed plugin {required_plugin_id!r} has "
                f"version {discovered_version!r}."
            )

    @staticmethod
    def _version_matches(version: str, specifier: str) -> bool:
        if Version is None or SpecifierSet is None:
            raise PluginInstallError(
                "Cannot validate versioned plugin requirements because packaging "
                "is not installed."
            )
        try:
            parsed_version = Version(str(version))
            parsed_specifier = SpecifierSet(str(specifier))
        except Exception as exc:
            raise PluginInstallError(
                f"Invalid plugin version/specifier ({version!r}, {specifier!r}): {exc}"
            ) from exc
        return parsed_version in parsed_specifier

    @staticmethod
    def _validate_update_version(
        current_version: str,
        new_version: str,
        *,
        allow_downgrade: bool,
    ) -> None:
        if current_version == new_version:
            raise PluginInstallError(
                f"Plugin version {new_version} is already installed."
            )

        if allow_downgrade:
            return

        if Version is None:
            raise PluginInstallError(
                "Cannot determine whether this package is an upgrade because packaging "
                "is not installed. Pass allow_downgrade=True only if intentional."
            )

        try:
            current = Version(current_version)
            new = Version(new_version)
        except Exception as exc:
            raise PluginInstallError(
                f"Cannot compare plugin versions {current_version!r} and "
                f"{new_version!r}: {exc}"
            ) from exc

        if new < current:
            raise PluginInstallError(
                f"Refusing to downgrade plugin from {current} to {new}. "
                "Pass allow_downgrade=True to override."
            )

    def _is_runtime_enabled(self, plugin_id: str) -> bool:
        if self.manager is None:
            return False
        try:
            info = self.manager.plugin_info(plugin_id)
        except Exception:
            return False
        status = getattr(info, "status", None)
        return getattr(status, "value", status) == "enabled"

    def _assert_not_runtime_enabled(self, plugin_id: str) -> None:
        if self._is_runtime_enabled(plugin_id):
            raise PluginInstallError(
                f"Plugin {plugin_id!r} is currently enabled. Disable it before updating."
            )

    def _refresh_manager_after_install(self) -> list[str]:
        if self.manager is None:
            return []
        try:
            self.manager.discover()
            return []
        except Exception as exc:
            return [
                "Plugin files were installed successfully, but live discovery refresh "
                f"failed. Restart AstronomicAL to pick up the plugin: {exc}"
            ]

    def _refresh_manager_after_replace(self, plugin_id: str) -> list[str]:
        if self.manager is None:
            return []

        warnings: list[str] = []
        forget = getattr(self.manager, "forget_user_plugin", None)
        if callable(forget):
            try:
                forget(plugin_id)
            except KeyError:
                pass
            except Exception as exc:
                warnings.append(
                    f"Could not clear the previous live plugin record: {exc}"
                )

        try:
            self.manager.discover()
        except Exception as exc:
            warnings.append(
                "Filesystem transaction succeeded, but live discovery refresh failed. "
                f"Restart AstronomicAL to reconcile plugin discovery: {exc}"
            )

        return warnings

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()