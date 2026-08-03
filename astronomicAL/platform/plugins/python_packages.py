from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.machinery
import importlib.metadata as importlib_metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import threading
from typing import Any, Dict, Iterable, Mapping, Sequence
import uuid

from .errors import PluginPythonEnvironmentError

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version
except Exception:
    Requirement = None  # type: ignore[assignment]
    canonicalize_name = None  # type: ignore[assignment]
    Version = None  # type: ignore[assignment]


_STATE_NAME = ".astronomical-python-packages.json"
_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PythonRequirementCheck:
    ok: bool
    message: str = ""
    skipped: bool = False
    source: str = ""
    installed_version: str | None = None


class PluginPythonEnvironmentTransaction:
    """Prepared, reversible replacement of the shared plugin package overlay."""

    def __init__(
        self,
        environment: "PluginPythonEnvironment",
        *,
        changed: bool,
        transaction_root: Path | None = None,
        stage: Path | None = None,
    ) -> None:
        self.environment = environment
        self.changed = bool(changed)
        self.transaction_root = transaction_root
        self.stage = stage
        self.backup: Path | None = None
        self.committed = False
        self.finished = False

    def commit(self) -> None:
        if self.finished:
            raise PluginPythonEnvironmentError(
                "Python dependency transaction has already been finalized."
            )
        if self.committed or not self.changed:
            self.committed = True
            return
        if self.stage is None or not self.stage.is_dir():
            raise PluginPythonEnvironmentError(
                "Prepared Python dependency staging directory is missing."
            )

        env = self.environment
        live = env.root
        live.parent.mkdir(parents=True, exist_ok=True)
        backup = live.with_name(f".{live.name}.backup.{uuid.uuid4().hex}")

        try:
            if live.exists():
                live.replace(backup)
                self.backup = backup
            self.stage.replace(live)
            self.committed = True
            env._activate_live_root()
        except Exception as exc:
            if live.exists():
                shutil.rmtree(live, ignore_errors=True)
            if backup.exists() and not live.exists():
                try:
                    backup.replace(live)
                except Exception:
                    pass
            env._activate_live_root()
            if isinstance(exc, PluginPythonEnvironmentError):
                raise
            raise PluginPythonEnvironmentError(
                f"Could not activate prepared Python plugin dependencies: {exc}"
            ) from exc

    def rollback(self) -> None:
        if self.finished:
            return

        env = self.environment
        live = env.root

        if self.changed and self.committed:
            try:
                if live.exists():
                    shutil.rmtree(live, ignore_errors=True)
                if self.backup is not None and self.backup.exists():
                    self.backup.replace(live)
            finally:
                self.committed = False
                env._activate_live_root()

        self._cleanup_stage()

    def finalize(self) -> None:
        if self.finished:
            return
        if self.backup is not None and self.backup.exists():
            shutil.rmtree(self.backup, ignore_errors=True)
        self._cleanup_stage()
        self.finished = True

    def _cleanup_stage(self) -> None:
        if self.transaction_root is not None and self.transaction_root.exists():
            shutil.rmtree(self.transaction_root, ignore_errors=True)


class PluginPythonEnvironment:
    """Shared, host-compatible Python dependency overlay for managed plugins.

    Community plugins execute in AstronomicAL's interpreter. This service therefore
    never attempts to shadow a distribution already present in the host environment.
    Host distributions always win; only additional compatible distributions are
    materialized under ``~/.astronomical/python-packages``.

    Dependency resolution/building happens in a disposable virtualenv that is given
    read-only import visibility of the distribution roots used by the running host.
    AstronomicAL ships the ``virtualenv`` package as a core
    dependency so resolver creation does not depend on the operating system providing
    stdlib ``venv``/``ensurepip`` support. The resolver gets its own seeded pip and
    never depends on or mutates a distro/global pip installation. The resulting
    plugin-only wheel files are copied into a fresh staging directory and the live
    overlay is replaced atomically.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        build_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        pip_timeout: float = 900.0,
    ) -> None:
        if Requirement is None or canonicalize_name is None or Version is None:
            raise PluginPythonEnvironmentError(
                "Managed plugin Python dependencies require the 'packaging' package."
            )

        self.root = Path(root).expanduser().resolve()
        self.build_root = Path(
            build_root or (self.root.parent / "python-package-builds")
        ).expanduser().resolve()
        # Do not resolve() the Python executable. Virtual environments commonly
        # expose their interpreter as a symlink to the base interpreter; resolving
        # that symlink would turn e.g. ``.venv/bin/python`` into ``/usr/bin/python``
        # and discard the virtualenv's sys.prefix/site-packages context.
        python_path = Path(python_executable or sys.executable).expanduser()
        if not python_path.is_absolute():
            python_path = Path.cwd() / python_path
        self.python_executable = Path(os.path.abspath(str(python_path)))
        self.pip_timeout = float(pip_timeout)
        self._lock = threading.RLock()
        self._load_error: str | None = None
        self._state: Dict[str, Any] | None = None
        self._overlay_inventory: Dict[str, str] = {}
        self._host_inventory = self._capture_host_inventory()
        self._host_distribution_paths = self._capture_host_distribution_paths()
        self._host_fingerprint = self._inventory_fingerprint(self._host_inventory)

    @property
    def load_error(self) -> str | None:
        with self._lock:
            return self._load_error

    @property
    def host_inventory(self) -> Dict[str, str]:
        return dict(self._host_inventory)

    @property
    def overlay_inventory(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._overlay_inventory)

    def activate(self) -> None:
        """Activate a previously reconciled overlay at lower priority than the host."""
        with self._lock:
            self._activate_live_root()

    def check_requirement(self, requirement: str) -> PythonRequirementCheck:
        """Validate one PEP 508 requirement against host first, then the overlay."""
        try:
            req = self._parse_requirement(requirement)
        except PluginPythonEnvironmentError as exc:
            return PythonRequirementCheck(ok=False, message=str(exc))

        if req.marker is not None and not req.marker.evaluate():
            return PythonRequirementCheck(ok=True, skipped=True)

        name = self._normalise_name(req.name)
        host_version = self._host_inventory.get(name)
        if host_version is not None:
            if req.specifier and Version(host_version) not in req.specifier:
                return PythonRequirementCheck(
                    ok=False,
                    source="host",
                    installed_version=host_version,
                    message=(
                        f"Host dependency {req.name} has version {host_version}, but "
                        f"plugin requires {req.specifier}. Community plugins may not "
                        "replace AstronomicAL's host packages."
                    ),
                )
            return PythonRequirementCheck(
                ok=True,
                source="host",
                installed_version=host_version,
            )

        with self._lock:
            overlay_version = self._overlay_inventory.get(name)
            overlay_error = self._load_error

        if overlay_version is None:
            if overlay_error:
                return PythonRequirementCheck(
                    ok=False,
                    source="overlay",
                    message=(
                        f"Missing dependency: {requirement}. The managed plugin Python "
                        f"environment is unavailable: {overlay_error}"
                    ),
                )
            return PythonRequirementCheck(
                ok=False,
                message=f"Missing dependency: {requirement}",
            )

        if req.specifier and Version(overlay_version) not in req.specifier:
            return PythonRequirementCheck(
                ok=False,
                source="overlay",
                installed_version=overlay_version,
                message=(
                    f"Dependency {req.name} has managed version {overlay_version}, "
                    f"but plugin requires {req.specifier}."
                ),
            )

        return PythonRequirementCheck(
            ok=True,
            source="overlay",
            installed_version=overlay_version,
        )

    def validate_requirements(
        self,
        plugin_id: str,
        requirements: Sequence[str],
    ) -> None:
        """Validate declarations and fail early on direct host-package conflicts."""
        plugin_id = str(plugin_id or "").strip() or "<unknown>"
        for raw in requirements:
            req = self._parse_requirement(raw)
            if req.marker is not None and not req.marker.evaluate():
                continue
            name = self._normalise_name(req.name)
            host_version = self._host_inventory.get(name)
            if host_version is None or not req.specifier:
                continue
            if Version(host_version) not in req.specifier:
                raise PluginPythonEnvironmentError(
                    f"Plugin {plugin_id!r} requires {str(req)!r}, but AstronomicAL's "
                    f"host environment provides {req.name} {host_version}. Community "
                    "plugins may add packages but may not replace host packages."
                )

    def needs_reconcile(
        self,
        requirements_by_plugin: Mapping[str, Sequence[str]],
    ) -> bool:
        normalized, active = self._normalise_requirement_map(requirements_by_plugin)
        if not active and not self.root.exists():
            return False
        fingerprint = self._requirements_fingerprint(normalized, active)
        with self._lock:
            state = self._read_live_state(raise_on_error=False)
            if state is None:
                return True
            return str(state.get("fingerprint", "")) != fingerprint

    def prepare(
        self,
        requirements_by_plugin: Mapping[str, Sequence[str]],
    ) -> PluginPythonEnvironmentTransaction:
        """Build a complete replacement overlay without changing live imports."""
        with self._lock:
            normalized, active = self._normalise_requirement_map(requirements_by_plugin)
            if not active and not self.root.exists():
                return PluginPythonEnvironmentTransaction(self, changed=False)
            fingerprint = self._requirements_fingerprint(normalized, active)
            state = self._read_live_state(raise_on_error=False)
            if (
                state is not None
                and str(state.get("fingerprint", "")) == fingerprint
                and self.root.is_dir()
            ):
                return PluginPythonEnvironmentTransaction(self, changed=False)

            transaction_root = self.build_root / f"reconcile.{uuid.uuid4().hex}"
            stage = transaction_root / "python-packages"
            build_venv = transaction_root / "resolver-venv"
            stage.mkdir(parents=True, exist_ok=False)

            try:
                distributions: Dict[str, str] = {}
                if self._requires_plugin_layer(active):
                    distributions = self._build_overlay(
                        active,
                        build_venv=build_venv,
                        stage=stage,
                    )

                state_data = {
                    "schema_version": _STATE_SCHEMA_VERSION,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "python_version": self._python_version_key(),
                    "host_fingerprint": self._host_fingerprint,
                    "fingerprint": fingerprint,
                    "requirements": {
                        plugin_id: list(requirements)
                        for plugin_id, requirements in normalized.items()
                    },
                    "distributions": dict(sorted(distributions.items())),
                }
                (stage / _STATE_NAME).write_text(
                    json.dumps(state_data, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except Exception:
                shutil.rmtree(transaction_root, ignore_errors=True)
                raise

            return PluginPythonEnvironmentTransaction(
                self,
                changed=True,
                transaction_root=transaction_root,
                stage=stage,
            )

    def loaded_overlay_modules(self) -> list[str]:
        """Return modules currently loaded from the managed package directory."""
        root = self.root
        result: list[str] = []
        for name, module in list(sys.modules.items()):
            if module is None:
                continue
            candidates: list[str] = []
            file_value = getattr(module, "__file__", None)
            if file_value:
                candidates.append(str(file_value))
            package_paths = getattr(module, "__path__", None)
            if package_paths:
                try:
                    candidates.extend(str(item) for item in package_paths)
                except Exception:
                    pass
            for candidate in candidates:
                try:
                    path = Path(candidate).resolve()
                except Exception:
                    continue
                if self._is_relative_to(path, root):
                    result.append(name)
                    break
        return sorted(set(result))

    def purge_loaded_overlay_modules(self) -> list[str]:
        """Remove pure-Python overlay modules from sys.modules before a safe swap.

        Native extension modules cannot be safely replaced in-process; if one is
        loaded, the operation is rejected and AstronomicAL must be restarted first.
        """
        loaded = self.loaded_overlay_modules()
        native: list[str] = []
        extension_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
        for name in loaded:
            module = sys.modules.get(name)
            file_value = str(getattr(module, "__file__", "") or "")
            if file_value.endswith(extension_suffixes):
                native.append(name)
        if native:
            preview = ", ".join(native[:8])
            if len(native) > 8:
                preview += f", … (+{len(native) - 8} more)"
            raise PluginPythonEnvironmentError(
                "Managed Python dependencies include native modules that are already "
                f"loaded ({preview}). Restart AstronomicAL before changing the plugin "
                "Python dependency environment."
            )

        # Children first so package parents cannot retain stale submodule attributes.
        for name in sorted(loaded, key=lambda value: value.count("."), reverse=True):
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        return loaded

    def _build_overlay(
        self,
        active: Sequence[tuple[str, str, Any]],
        *,
        build_venv: Path,
        stage: Path,
    ) -> Dict[str, str]:
        requirements = [raw for _, raw, _ in active]
        self._create_resolver_environment(build_venv)

        venv_python = self._venv_python(build_venv)
        self._require_resolver_pip(venv_python)
        self._expose_host_distributions(venv_python)
        report_path = build_venv.parent / "pip-resolution-report.json"

        env = os.environ.copy()
        # The live plugin overlay is derived state and must not participate in the
        # resolver. Otherwise an old overlay could incorrectly satisfy a fresh
        # rebuild and then disappear when the staged directory replaces it.
        env.pop("PYTHONPATH", None)
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

        # Resolve first *without* --target. The resolver virtualenv sees the actual
        # distribution roots used by the running AstronomicAL interpreter through a
        # generated .pth file, so pip can correctly reuse a compatible host dependency
        # instead of needlessly selecting a newer copy
        # for the plugin overlay. pip's --target mode does not reliably make that
        # distinction and can otherwise propose packages such as typing-extensions
        # even when the host already satisfies the transitive requirement.
        resolve_command = [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--report",
            str(report_path),
            "--disable-pip-version-check",
            "--no-input",
            "--only-binary=:all:",
            *requirements,
        ]
        completed = self._run_pip(
            resolve_command,
            env=env,
            action="resolving plugin Python dependencies",
        )
        if completed.returncode != 0:
            detail = self._pip_error_detail(completed.stdout, completed.stderr)
            raise PluginPythonEnvironmentError(
                "Could not resolve plugin Python dependencies using binary wheels "
                f"only.{detail}"
            )

        planned = self._read_resolution_report(report_path)

        host_conflicts: list[tuple[str, str, str]] = []
        overlay_pins: list[str] = []
        expected_overlay: Dict[str, str] = {}
        for name, version in sorted(planned.items()):
            host_version = self._host_inventory.get(name)
            if host_version is not None:
                try:
                    same_version = Version(host_version) == Version(version)
                except Exception:
                    same_version = host_version == version
                if not same_version:
                    host_conflicts.append((name, version, host_version))
                # Whether equal or conflicting, a host-owned distribution is never
                # materialized into the lower-priority plugin overlay.
                continue

            expected_overlay[name] = version
            overlay_pins.append(f"{name}=={version}")

        if host_conflicts:
            details = "; ".join(
                f"{name} would install {candidate} over host {host}"
                for name, candidate, host in host_conflicts
            )
            raise PluginPythonEnvironmentError(
                "Plugin Python dependencies would replace packages already owned by "
                f"AstronomicAL's host environment: {details}."
            )

        if not overlay_pins:
            return {}

        # Materialize exactly the versions selected by the host-aware resolver.
        # --no-deps is intentional: all transitive dependencies were already
        # resolved above. This prevents --target from independently re-resolving
        # dependencies and installing unnecessary duplicates of host packages.
        install_command = [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--only-binary=:all:",
            "--no-deps",
            "--target",
            str(stage),
            *overlay_pins,
        ]
        installed = self._run_pip(
            install_command,
            env=env,
            action="materializing plugin Python dependencies",
        )
        if installed.returncode != 0:
            detail = self._pip_error_detail(installed.stdout, installed.stderr)
            raise PluginPythonEnvironmentError(
                "Plugin Python dependencies resolved successfully, but AstronomicAL "
                "could not materialize the resolved binary wheels into the managed "
                f"plugin package overlay.{detail}"
            )

        actual_overlay = self._distribution_inventory([stage])
        mismatches: list[str] = []
        for name, expected_version in sorted(expected_overlay.items()):
            actual_version = actual_overlay.get(name)
            if actual_version != expected_version:
                mismatches.append(
                    f"{name} expected {expected_version}, found {actual_version or 'missing'}"
                )
        unexpected = sorted(set(actual_overlay) - set(expected_overlay))
        if unexpected:
            mismatches.append(
                "unexpected distributions: " + ", ".join(unexpected)
            )
        if mismatches:
            raise PluginPythonEnvironmentError(
                "Managed plugin Python dependency materialization did not match the "
                "resolver plan: " + "; ".join(mismatches)
            )

        return dict(sorted(actual_overlay.items()))

    def _run_pip(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        action: str,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                list(command),
                env=dict(env),
                text=True,
                capture_output=True,
                timeout=self.pip_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PluginPythonEnvironmentError(
                f"Timed out after {self.pip_timeout:g}s while {action}."
            ) from exc
        except Exception as exc:
            raise PluginPythonEnvironmentError(
                f"Could not run pip while {action}: {exc}"
            ) from exc

    def _read_resolution_report(self, report_path: Path) -> Dict[str, str]:
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PluginPythonEnvironmentError(
                "pip completed dependency resolution but did not produce a readable "
                f"resolution report: {exc}"
            ) from exc

        installs = data.get("install", []) if isinstance(data, dict) else []
        if not isinstance(installs, list):
            raise PluginPythonEnvironmentError(
                "pip returned an invalid plugin dependency resolution report."
            )

        planned: Dict[str, str] = {}
        for item in installs:
            if not isinstance(item, dict):
                raise PluginPythonEnvironmentError(
                    "pip returned an invalid entry in the plugin dependency "
                    "resolution report."
                )
            if item.get("is_direct"):
                raise PluginPythonEnvironmentError(
                    "Plugin Python dependency resolution selected a direct URL/VCS "
                    "artifact. Community plugin dependencies must resolve through the "
                    "configured Python package index."
                )
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                raise PluginPythonEnvironmentError(
                    "pip returned plugin dependency metadata in an invalid format."
                )
            raw_name = str(metadata.get("name", "") or "").strip()
            version = str(metadata.get("version", "") or "").strip()
            if not raw_name or not version:
                raise PluginPythonEnvironmentError(
                    "pip returned a resolved plugin dependency without a package "
                    "name/version."
                )
            name = self._normalise_name(raw_name)
            previous = planned.get(name)
            if previous is not None and previous != version:
                raise PluginPythonEnvironmentError(
                    f"pip resolved multiple versions of {raw_name}: {previous}, {version}."
                )
            planned[name] = version

        return planned

    def _copy_added_distributions(
        self,
        *,
        site_paths: Sequence[Path],
        baseline: Mapping[str, str],
        added: Mapping[str, str],
        stage: Path,
    ) -> None:
        copied: set[Path] = set()
        for site_path in site_paths:
            for dist in importlib_metadata.distributions(path=[str(site_path)]):
                raw_name = str(dist.metadata.get("Name", "") or "").strip()
                if not raw_name:
                    continue
                name = self._normalise_name(raw_name)
                version = str(dist.version or "")
                if name not in added or baseline.get(name) == version:
                    continue

                files = dist.files or []
                for entry in files:
                    source = Path(dist.locate_file(entry))
                    try:
                        source_resolved = source.resolve()
                    except Exception:
                        continue
                    base = next(
                        (
                            candidate
                            for candidate in site_paths
                            if self._is_relative_to(source_resolved, candidate)
                        ),
                        None,
                    )
                    if base is None or not source_resolved.is_file():
                        # Console scripts and other files outside site-packages are not
                        # part of the import overlay.
                        continue
                    relative = source_resolved.relative_to(base)
                    target = stage / relative
                    if target in copied:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_resolved, target)
                    copied.add(target)

    def _normalise_requirement_map(
        self,
        requirements_by_plugin: Mapping[str, Sequence[str]],
    ) -> tuple[Dict[str, tuple[str, ...]], list[tuple[str, str, Any]]]:
        normalized: Dict[str, tuple[str, ...]] = {}
        active: list[tuple[str, str, Any]] = []

        for plugin_id in sorted(requirements_by_plugin):
            raw_values = requirements_by_plugin[plugin_id]
            if isinstance(raw_values, str):
                raise PluginPythonEnvironmentError(
                    f"Plugin {plugin_id!r} Python requirements must be a list of strings."
                )
            cleaned: list[str] = []
            for raw in raw_values:
                if not isinstance(raw, str) or not raw.strip():
                    raise PluginPythonEnvironmentError(
                        f"Plugin {plugin_id!r} contains an invalid Python requirement: {raw!r}."
                    )
                text = raw.strip()
                req = self._parse_requirement(text)
                cleaned.append(text)
                if req.marker is not None and not req.marker.evaluate():
                    continue
                name = self._normalise_name(req.name)
                host_version = self._host_inventory.get(name)
                if (
                    host_version is not None
                    and req.specifier
                    and Version(host_version) not in req.specifier
                ):
                    raise PluginPythonEnvironmentError(
                        f"Plugin {plugin_id!r} requires {text!r}, but AstronomicAL's "
                        f"host environment provides {req.name} {host_version}. Community "
                        "plugins may not replace host packages."
                    )
                active.append((plugin_id, text, req))
            normalized[str(plugin_id)] = tuple(cleaned)

        return normalized, active

    def _requires_plugin_layer(self, active: Sequence[tuple[str, str, Any]]) -> bool:
        for _, _, req in active:
            name = self._normalise_name(req.name)
            if name not in self._host_inventory or bool(req.extras):
                return True
        return False

    def _parse_requirement(self, value: str) -> Any:
        if not isinstance(value, str) or not value.strip():
            raise PluginPythonEnvironmentError(
                f"Python dependency requirement must be a non-empty string: {value!r}."
            )
        try:
            req = Requirement(value.strip())
        except Exception as exc:
            raise PluginPythonEnvironmentError(
                f"Invalid Python dependency requirement {value!r}: {exc}"
            ) from exc
        if req.url:
            raise PluginPythonEnvironmentError(
                f"Direct URL/VCS Python dependencies are not allowed for community "
                f"plugins: {value!r}. Publish a versioned package to the configured "
                "Python package index instead."
            )
        return req

    def _requirements_fingerprint(
        self,
        normalized: Mapping[str, Sequence[str]],
        active: Sequence[tuple[str, str, Any]],
    ) -> str:
        active_by_plugin: Dict[str, list[str]] = {}
        for plugin_id, raw, _ in active:
            active_by_plugin.setdefault(plugin_id, []).append(raw)
        payload = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "python_version": self._python_version_key(),
            "host_fingerprint": self._host_fingerprint,
            "requirements": {
                plugin_id: sorted(values)
                for plugin_id, values in sorted(active_by_plugin.items())
                if values
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _activate_live_root(self) -> None:
        root_text = str(self.root)
        sys.path[:] = [entry for entry in sys.path if str(entry) != root_text]
        self._state = None
        self._overlay_inventory = {}
        self._load_error = None

        if not self.root.exists():
            importlib.invalidate_caches()
            return

        state = self._read_live_state(raise_on_error=False)
        if state is None:
            importlib.invalidate_caches()
            return

        sys.path.append(root_text)
        self._state = state
        self._overlay_inventory = self._distribution_inventory([self.root])
        importlib.invalidate_caches()

    def _read_live_state(self, *, raise_on_error: bool) -> Dict[str, Any] | None:
        if not self.root.exists():
            return None
        state_path = self.root / _STATE_NAME
        try:
            if not state_path.is_file():
                raise ValueError(
                    f"Managed Python package directory is missing {_STATE_NAME}."
                )
            data = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Managed Python package state must be a JSON object.")
            if int(data.get("schema_version", 0)) != _STATE_SCHEMA_VERSION:
                raise ValueError(
                    "Unsupported managed Python package state schema version."
                )
            if str(data.get("python_version", "")) != self._python_version_key():
                raise ValueError(
                    "Managed plugin Python packages were built for a different Python "
                    "major/minor version. Reconcile plugin dependencies before enabling "
                    "community plugins."
                )
            if str(data.get("host_fingerprint", "")) != self._host_fingerprint:
                raise ValueError(
                    "AstronomicAL's host Python environment changed after the managed "
                    "plugin dependency set was built. Reconcile plugin dependencies "
                    "before enabling community plugins."
                )
            self._load_error = None
            return data
        except Exception as exc:
            self._load_error = str(exc)
            if raise_on_error:
                raise PluginPythonEnvironmentError(str(exc)) from exc
            return None

    def _capture_host_inventory(self) -> Dict[str, str]:
        inventory: Dict[str, str] = {}
        for dist in importlib_metadata.distributions():
            raw_name = str(dist.metadata.get("Name", "") or "").strip()
            if not raw_name:
                continue
            try:
                location = Path(dist.locate_file("")).resolve()
            except Exception:
                location = None
            if location is not None and self._is_relative_to(location, self.root):
                continue
            name = self._normalise_name(raw_name)
            inventory.setdefault(name, str(dist.version or ""))
        return inventory

    def _capture_host_distribution_paths(self) -> tuple[Path, ...]:
        """Return distribution roots visible to the running AstronomicAL process.

        A resolver created from a virtualenv interpreter does not inherit the parent
        virtualenv's site-packages merely because the child uses
        ``--system-site-packages``; that flag refers to the base interpreter's global
        site-packages. Capture the actual distribution roots from the running host so
        the disposable resolver can mirror AstronomicAL's real package inventory.
        """
        paths: set[Path] = set()
        for dist in importlib_metadata.distributions():
            try:
                raw_location = Path(dist.locate_file(""))
                location = Path(os.path.abspath(str(raw_location.expanduser())))
            except Exception:
                continue
            try:
                root_cmp = location.resolve()
            except Exception:
                root_cmp = location
            if self._is_relative_to(root_cmp, self.root):
                continue
            if location.is_dir():
                paths.add(location)
        return tuple(sorted(paths, key=lambda value: str(value)))

    def _distribution_inventory(self, paths: Sequence[Path]) -> Dict[str, str]:
        inventory: Dict[str, str] = {}
        for path in paths:
            if not path.exists():
                continue
            for dist in importlib_metadata.distributions(path=[str(path)]):
                raw_name = str(dist.metadata.get("Name", "") or "").strip()
                if not raw_name:
                    continue
                inventory[self._normalise_name(raw_name)] = str(dist.version or "")
        return inventory

    def _require_resolver_pip(self, python_path: Path) -> None:
        """Verify the disposable resolver owns a usable pip installation."""
        try:
            completed = subprocess.run(
                [str(python_path), "-m", "pip", "--version"],
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except Exception as exc:
            raise PluginPythonEnvironmentError(
                "Could not verify pip in the temporary plugin dependency resolver: "
                f"{exc}"
            ) from exc

        if completed.returncode == 0:
            return

        detail = self._pip_error_detail(completed.stdout, completed.stderr)
        raise PluginPythonEnvironmentError(
            "The temporary plugin dependency resolver was created by virtualenv, but "
            "its private seeded pip installation is unusable. AstronomicAL does not "
            "fall back to the system/global pip because doing so can use a pip build "
            "incompatible with the running Python interpreter."
            f"{detail}"
        )

    def _create_resolver_environment(self, build_venv: Path) -> None:
        """Create a disposable resolver using AstronomicAL's bundled virtualenv.

        ``virtualenv`` seeds pip from its own wheel bundle/cache and therefore does not
        require the host operating system to provide ``pythonX.Y-venv`` or a working
        stdlib ``ensurepip`` installation. The resolver starts isolated; after its
        private pip is verified, AstronomicAL explicitly exposes the distribution roots
        visible to the *running host virtualenv*. This is more accurate than
        ``--system-site-packages``, which can point at the base/system interpreter and
        omit packages installed in AstronomicAL's parent virtualenv.
        """
        command = [
            str(self.python_executable),
            "-m",
            "virtualenv",
            str(build_venv),
        ]
        env = os.environ.copy()
        # Resolver creation must not inherit the live plugin overlay.
        env.pop("PYTHONPATH", None)

        try:
            completed = subprocess.run(
                command,
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PluginPythonEnvironmentError(
                "Timed out while creating the temporary Python dependency resolver "
                "with AstronomicAL's virtualenv package."
            ) from exc
        except Exception as exc:
            raise PluginPythonEnvironmentError(
                f"Could not launch AstronomicAL's virtualenv resolver: {exc}"
            ) from exc

        if completed.returncode == 0:
            return

        detail = self._pip_error_detail(completed.stdout, completed.stderr)
        raise PluginPythonEnvironmentError(
            "Could not create the temporary plugin dependency resolver using "
            "AstronomicAL's bundled 'virtualenv' package. Ensure requirements-core.txt "
            "has been installed for the same Python interpreter that runs AstronomicAL. "
            "No operating-system python3-venv package or sudo installation should be "
            "required."
            f"{detail}"
        )

    def _expose_host_distributions(self, resolver_python: Path) -> None:
        """Expose AstronomicAL's actual host distribution roots to resolver pip.

        The .pth file is written only after the resolver's private pip has been
        verified, so host paths cannot replace the resolver's own pip. Paths are added
        after the resolver site-packages directory and are used for dependency
        satisfaction/metadata only; the live plugin overlay itself is excluded.
        """
        site_paths = self._venv_site_paths(resolver_python)
        if not site_paths:
            raise PluginPythonEnvironmentError(
                "Could not determine the temporary plugin resolver site-packages path."
            )

        target_site = site_paths[0]
        target_site.mkdir(parents=True, exist_ok=True)
        pth_path = target_site / "_astronomical_host_environment.pth"
        lines = [str(path) for path in self._host_distribution_paths if path.is_dir()]
        try:
            pth_path.write_text(
                "".join(f"{line}\n" for line in lines),
                encoding="utf-8",
            )
        except Exception as exc:
            raise PluginPythonEnvironmentError(
                "Could not expose AstronomicAL's host Python environment to the "
                f"temporary plugin dependency resolver: {exc}"
            ) from exc

    def _venv_site_paths(self, python_path: Path) -> list[Path]:
        code = (
            "import json,sysconfig; "
            "p=sysconfig.get_paths(); "
            "print(json.dumps([p.get('purelib'),p.get('platlib')]))"
        )
        completed = subprocess.run(
            [str(python_path), "-c", code],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            raise PluginPythonEnvironmentError(
                "Could not determine temporary resolver site-packages paths."
            )
        try:
            raw_paths = json.loads(completed.stdout.strip())
        except Exception as exc:
            raise PluginPythonEnvironmentError(
                "Temporary resolver returned invalid site-packages path data."
            ) from exc
        result: list[Path] = []
        for value in raw_paths:
            if not value:
                continue
            path = Path(value).resolve()
            if path not in result:
                result.append(path)
        return result

    @staticmethod
    def _venv_python(root: Path) -> Path:
        if os.name == "nt":
            return root / "Scripts" / "python.exe"
        return root / "bin" / "python"

    @staticmethod
    def _python_version_key() -> str:
        return f"{sys.version_info.major}.{sys.version_info.minor}"

    @staticmethod
    def _inventory_fingerprint(inventory: Mapping[str, str]) -> str:
        encoded = json.dumps(
            dict(sorted(inventory.items())),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _normalise_name(value: str) -> str:
        return str(canonicalize_name(str(value)))

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _pip_error_detail(stdout: str, stderr: str) -> str:
        text = "\n".join(
            part.strip()
            for part in (stdout, stderr)
            if part and part.strip()
        )
        if not text:
            return ""
        # Keep diagnostics useful without flooding Plugin Manager with full pip logs.
        text = text[-5000:]
        return "\n\n" + text