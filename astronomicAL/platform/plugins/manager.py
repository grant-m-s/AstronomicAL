from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import inspect
import json
import logging
import re
import sys
import traceback
import uuid

import html
import time

import panel as pn

import threading

from astronomicAL.platform.panel_state import restore_controller_state
from astronomicAL.utils.debug import boot_print
from .mapping_gate import MappingGatedPanel

from .api import PluginAPI
from .errors import (
    PluginDiscoveryError,
    PluginExecutionError,
    PluginLoadError,
    PluginRegistrationError,
    PluginValidationError,
)
from .manifest import PluginManifest, coerce_manifest
from .specs import (
    ActionRegistration,
    ActionRequest,
    ActionResult,
    ArtifactResult,
    ArtifactViewerRegistration,
    CreatedPanel,
    DatasetResult,
    EventResult,
    InputSpec,
    PanelRegistration,
    PluginInfo,
    PluginStatus,
    ProcessedActionResult,
    ServiceRegistration,
    ValidationResult,
    WorkflowRegistration,
)

try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except Exception:
    Requirement = None  # type: ignore[assignment]
    SpecifierSet = None  # type: ignore[assignment]
    Version = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

def plugin_open_debug(label: str, **values: Any) -> None:
    try:
        parts = " ".join(f"{key}={value!r}" for key, value in values.items())
        print(
            f"[AL_DEBUG][PluginManager][{label}] "
            f"thread={threading.current_thread().name} {parts}",
            flush=True,
        )
    except Exception:
        print(f"[AL_DEBUG][PluginManager][{label}] <print failed>", flush=True)


class LoadingPanelController:
    """Temporary controller shown while a plugin panel is being constructed."""

    state_version = 1

    def __init__(self, title: str, detail: str = "") -> None:
        self.title = str(title or "Panel")
        self.detail = str(detail or "Preparing panel...")
        self.started_at = time.time()

    def dispose(self) -> None:
        pass

    def get_state(self) -> dict[str, Any]:
        return {
            "loading": True,
            "title": self.title,
            "detail": self.detail,
            "started_at": self.started_at,
        }


def make_loading_panel(title: str, detail: str = "") -> tuple[Any, Any]:
    safe_title = html.escape(str(title or "Panel"), quote=True)
    safe_detail = html.escape(str(detail or "Preparing panel..."), quote=True)

    view = pn.Column(
        pn.Spacer(height=8),
        pn.indicators.LoadingSpinner(
            value=True,
            width=44,
            height=44,
            sizing_mode="fixed",
            margin=(8, 0, 8, 0),
        ),
        pn.pane.HTML(
            f"""
            <div style="text-align: center; padding: 0 12px;">
                <div style="font-weight: 700; font-size: 15px; margin-bottom: 6px;">
                    Loading {safe_title}
                </div>
                <div style="font-size: 12px; color: #666;">
                    {safe_detail}
                </div>
            </div>
            """,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_both",
        align="center",
        margin=(0, 0, 0, 0),
        styles={
            "height": "100%",
            "width": "100%",
            "box-sizing": "border-box",
            "display": "flex",
            "align-items": "center",
            "justify-content": "center",
            "overflow": "hidden",
            "background": "#fafafa",
        },
    )

    return view, LoadingPanelController(title=title, detail=detail)


class PanelLoadErrorController:
    state_version = 1

    def __init__(self, title: str, error: str) -> None:
        self.title = str(title or "Panel")
        self.error = str(error)

    def dispose(self) -> None:
        pass

    def get_state(self) -> dict[str, Any]:
        return {
            "error": True,
            "title": self.title,
            "message": self.error,
        }


def make_panel_load_error_panel(title: str, error: BaseException | str) -> tuple[Any, Any]:
    safe_title = html.escape(str(title or "Panel"), quote=True)
    safe_error = html.escape(str(error), quote=True)

    view = pn.Column(
        pn.pane.Alert(
            f"Could not load **{safe_title}**.",
            alert_type="danger",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        ),
        pn.pane.HTML(
            f"""
            <div style="font-size: 12px; color: #555; white-space: pre-wrap;
                        border: 1px solid #e1e1e1; border-radius: 6px;
                        padding: 8px; background: #fff;">
                {safe_error}
            </div>
            """,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_both",
        margin=(0, 0, 0, 0),
        styles={
            "height": "100%",
            "width": "100%",
            "box-sizing": "border-box",
            "padding": "10px",
            "overflow": "auto",
        },
    )

    return view, PanelLoadErrorController(title=title, error=str(error))


def schedule_panel_callback(callback) -> None:
    """Run callback shortly after the current UI callback returns.

    This gives the browser a chance to render the loading placeholder before
    expensive panel construction starts.
    """

    try:
        doc = pn.state.curdoc
    except Exception:
        doc = None

    if doc is not None:
        try:
            doc.add_timeout_callback(callback, 50)
            return
        except Exception:
            try:
                doc.add_next_tick_callback(callback)
                return
            except Exception:
                pass

    callback()

@dataclass
class PluginCandidate:
    """A discoverable plugin before it is enabled."""

    source: str
    module_name: str
    path: Optional[Path] = None
    entry_point_name: Optional[str] = None
    entry_point_object: Optional[str] = None
    manifest_path: Optional[Path] = None
    manifest: Optional[PluginManifest] = None
    module: Optional[ModuleType] = None
    error: Optional[str] = None


@dataclass
class PluginRecord:
    manifest: PluginManifest
    candidate: PluginCandidate
    status: PluginStatus = PluginStatus.DISCOVERED
    module: Optional[ModuleType] = None
    error: Optional[str] = None
    settings_schema: Dict[str, Any] = field(default_factory=dict)


class PluginSettingsStore:
    """In-memory settings store for plugin configuration.

    This intentionally starts simple. A later persistence layer can load/save this
    dictionary from user config without changing plugin-facing APIs.
    """

    def __init__(self) -> None:
        self._settings: Dict[str, Dict[str, Any]] = {}

    def get(self, plugin_id: str, key: str, default: Any = None) -> Any:
        return self._settings.get(plugin_id, {}).get(key, default)

    def set(self, plugin_id: str, key: str, value: Any) -> None:
        self._settings.setdefault(plugin_id, {})[key] = value

    def update(self, plugin_id: str, values: Dict[str, Any]) -> None:
        self._settings.setdefault(plugin_id, {}).update(values)

    def get_all(self, plugin_id: str) -> Dict[str, Any]:
        return dict(self._settings.get(plugin_id, {}))

    def clear(self, plugin_id: Optional[str] = None) -> None:
        if plugin_id is None:
            self._settings.clear()
        else:
            self._settings.pop(plugin_id, None)


class PluginManager:
    """Discovers, validates, enables, and exposes AstronomicAL plugins.

    The manager intentionally owns only plugin metadata and registrations. Runtime
    data still belongs in platform services: datasets, selection, events,
    artifacts, jobs, workspace, and services.
    """

    ENTRY_POINT_GROUP = "astronomical.plugins"
    STATIC_MANIFEST_ENTRY_POINT_GROUP = "astronomical.plugin_manifests"

    def __init__(
        self,
        *,
        entry_point_group: str = ENTRY_POINT_GROUP,
        static_manifest_entry_point_group: str = STATIC_MANIFEST_ENTRY_POINT_GROUP,
        local_plugin_dirs: Optional[Sequence[str | Path]] = None,
        auto_discover: bool = False,
    ) -> None:
        self.entry_point_group = entry_point_group
        self.static_manifest_entry_point_group = static_manifest_entry_point_group
        self.local_plugin_dirs = [Path(p).expanduser() for p in (local_plugin_dirs or [])]

        self.settings = PluginSettingsStore()

        self._records: Dict[str, PluginRecord] = {}
        self._discovery_errors: Dict[str, str] = {}
        self._candidates_by_module: Dict[str, PluginCandidate] = {}

        self._panels: Dict[str, PanelRegistration] = {}
        self._actions: Dict[str, ActionRegistration] = {}
        self._workflows: Dict[str, WorkflowRegistration] = {}
        self._services: Dict[str, ServiceRegistration] = {}
        self._artifact_viewers: Dict[str, List[ArtifactViewerRegistration]] = {}

        self._installed_service_keys_by_plugin: Dict[str, set[str]] = {}
        self._workspace_panels_by_plugin: Dict[str, set[str]] = {}
        self._job_handles_by_plugin: Dict[str, list[Any]] = {}

        if auto_discover:
            self.discover()

    def discover(self) -> List[PluginInfo]:
        """Discover installed and local plugins.

        Static manifest candidates are read without importing the plugin runtime
        module. Import-based discovery remains available for simple local
        development and backwards compatibility.
        """

        boot_print("PluginManager.discover: start")
        boot_print(f"PluginManager.discover: local_plugin_dirs={self.local_plugin_dirs}")

        candidates: List[PluginCandidate] = []
        candidates.extend(self._discover_entry_points())
        candidates.extend(self._discover_static_manifest_entry_points())
        candidates.extend(self._discover_local_dirs())

        boot_print(f"PluginManager.discover: candidates count={len(candidates)}")
        for candidate in candidates:
            boot_print(
                "PluginManager.discover: candidate "
                f"source={candidate.source} "
                f"module={candidate.module_name} "
                f"path={candidate.path} "
                f"manifest_path={candidate.manifest_path}"
            )
            try:
                if candidate.manifest_path is not None:
                    module = None
                    manifest = self._read_static_manifest(candidate.manifest_path)
                    
                else:
                    module = self._load_module(candidate)
                    candidate.module = module
                    manifest = self._read_manifest(module, candidate=candidate)

                candidate.manifest = manifest
                boot_print(
                    "PluginManager.discover: manifest read "
                    f"id={manifest.id} name={manifest.name} version={manifest.version}"
                )
                if manifest.id in self._records:
                    existing = self._records[manifest.id]
                    if self._same_candidate(existing.candidate, candidate):
                        continue

                    message = (
                        f"Duplicate plugin id {manifest.id!r}. Existing source is "
                        f"{existing.candidate.source}; duplicate source is {candidate.source}."
                    )
                    candidate.error = message
                    self._record_discovery_error(candidate, message)
                    continue

                self._records[manifest.id] = PluginRecord(
                    manifest=manifest,
                    candidate=candidate,
                    module=module,
                    status=PluginStatus.DISABLED,
                )
                boot_print(
                    "PluginManager.discover: registered record "
                    f"id={manifest.id} status={self._records[manifest.id].status}"
                )
                if candidate.module_name:
                    self._candidates_by_module[candidate.module_name] = candidate
            except Exception as exc:
                candidate.error = self._format_exception(exc)
                self._record_discovery_error(candidate, candidate.error)

        boot_print(
            f"PluginManager.discover: complete records={list(self._records.keys())}"
        )
        return self.list_plugins()

    def _discover_entry_points(self) -> List[PluginCandidate]:
        candidates: List[PluginCandidate] = []
        try:
            eps = importlib_metadata.entry_points()
            if hasattr(eps, "select"):
                selected = eps.select(group=self.entry_point_group)
            else:
                selected = eps.get(self.entry_point_group, [])  # type: ignore[attr-defined]

            for ep in selected:
                module_name, object_name = self._split_entry_point_value(ep.value)
                manifest_path = None
                if object_name and self._looks_like_manifest_file(object_name):
                    manifest_path = self._resolve_entry_point_manifest_path(
                        ep,
                        module_name=module_name,
                        manifest_ref=object_name,
                    )

                candidates.append(
                    PluginCandidate(
                        source="entry_point_static_manifest" if manifest_path else "entry_point",
                        module_name=module_name,
                        entry_point_name=ep.name,
                        entry_point_object=object_name,
                        manifest_path=manifest_path,
                    )
                )
        except Exception as exc:
            raise PluginDiscoveryError(f"Failed to discover entry points: {exc}") from exc
        return candidates

    def _discover_static_manifest_entry_points(self) -> List[PluginCandidate]:
        candidates: List[PluginCandidate] = []
        if self.static_manifest_entry_point_group == self.entry_point_group:
            return candidates

        try:
            eps = importlib_metadata.entry_points()
            if hasattr(eps, "select"):
                selected = eps.select(group=self.static_manifest_entry_point_group)
            else:
                selected = eps.get(self.static_manifest_entry_point_group, [])  # type: ignore[attr-defined]

            for ep in selected:
                module_name, object_name = self._split_entry_point_value(ep.value)
                manifest_ref = object_name or module_name
                manifest_path = self._resolve_entry_point_manifest_path(
                    ep,
                    module_name=module_name if object_name else "",
                    manifest_ref=manifest_ref,
                )
                candidates.append(
                    PluginCandidate(
                        source="static_manifest_entry_point",
                        module_name=module_name if object_name else "",
                        entry_point_name=ep.name,
                        entry_point_object=object_name,
                        manifest_path=manifest_path,
                    )
                )
        except Exception as exc:
            raise PluginDiscoveryError(
                f"Failed to discover static manifest entry points: {exc}"
            ) from exc
        return candidates

    def _discover_local_dirs(self) -> List[PluginCandidate]:
        candidates: List[PluginCandidate] = []
        for root in self.local_plugin_dirs:
            if not root.exists():
                continue
            for item in root.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_file() and item.name.endswith(".py"):
                    candidates.append(
                        PluginCandidate(
                            source="local_file",
                            module_name=self._local_module_name(item, item.stem),
                            path=item,
                        )
                    )
                elif item.is_dir():
                    plugin_py = item / "plugin.py"
                    manifest_path = self._find_local_manifest(item)
                    if plugin_py.exists():
                        candidates.append(
                            PluginCandidate(
                                source="local_dir_static_manifest" if manifest_path else "local_dir",
                                module_name=self._local_module_name(plugin_py, item.name),
                                path=plugin_py,
                                manifest_path=manifest_path,
                            )
                        )
        return candidates

    def add_local_plugin_dir(self, path: str | Path) -> None:
        p = Path(path).expanduser()
        if p not in self.local_plugin_dirs:
            self.local_plugin_dirs.append(p)

    def list_discovery_errors(self) -> Dict[str, str]:
        return dict(self._discovery_errors)

    def _record_discovery_error(self, candidate: PluginCandidate, error: str) -> None:
        key = (
            candidate.entry_point_name
            or str(candidate.manifest_path)
            or str(candidate.path)
            or candidate.module_name
            or f"candidate:{len(self._discovery_errors)}"
        )
        self._discovery_errors[key] = error

    @staticmethod
    def _split_entry_point_value(value: str) -> Tuple[str, Optional[str]]:
        module_name, sep, object_name = value.partition(":")
        return module_name, object_name if sep else None

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"\W+", "_", value).strip("_")
        return slug or "plugin"

    def _local_module_name(self, path: Path, stem: str) -> str:
        digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
        return f"astronomical_local_plugin_{self._slug(stem)}_{digest}"

    @staticmethod
    def _looks_like_manifest_file(value: str) -> bool:
        lowered = value.lower()
        return lowered.endswith(".json") or lowered.endswith(".toml")

    def _find_local_manifest(self, root: Path) -> Optional[Path]:
        for name in (
            "astronomical-plugin.json",
            "astronomical_plugin.json",
            "plugin.json",
            "manifest.json",
            "astronomical-plugin.toml",
            "astronomical_plugin.toml",
            "plugin.toml",
            "manifest.toml",
        ):
            candidate = root / name
            if candidate.exists():
                return candidate
        return None

    def _resolve_entry_point_manifest_path(
        self,
        ep: Any,
        *,
        module_name: str,
        manifest_ref: str,
    ) -> Optional[Path]:
        ref_path = Path(manifest_ref)
        if ref_path.is_absolute() and ref_path.exists():
            return ref_path

        candidates: List[str] = [manifest_ref]
        if module_name:
            parts = module_name.split(".")
            if len(parts) > 1:
                candidates.append("/".join(parts[:-1] + [manifest_ref]))
            candidates.append("/".join(parts + [manifest_ref]))
            candidates.append(f"{parts[0]}/{manifest_ref}")

        dist = getattr(ep, "dist", None)
        if dist is not None:
            for candidate in candidates:
                try:
                    path = Path(dist.locate_file(candidate))
                except Exception:
                    continue
                if path.exists():
                    return path

        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                return path

        return None

    def _read_static_manifest(self, path: Optional[Path]) -> PluginManifest:
        if path is None:
            raise PluginDiscoveryError("Static plugin manifest path could not be resolved.")

        path = Path(path)
        if not path.exists():
            raise PluginDiscoveryError(f"Static plugin manifest does not exist: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            data = self._read_toml_manifest(path)
        else:
            raise PluginDiscoveryError(f"Unsupported static manifest format: {path}")

        if isinstance(data, dict):
            data = data.get("plugin", data.get("astronomical", data))
        return coerce_manifest(data)

    @staticmethod
    def _read_toml_manifest(path: Path) -> Dict[str, Any]:
        try:
            import tomllib  # type: ignore[import-not-found]
        except Exception:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except Exception as exc:
                raise PluginDiscoveryError(
                    "TOML plugin manifests require Python 3.11+ or tomli."
                ) from exc

        with path.open("rb") as handle:
            data = tomllib.load(handle)

        if "tool" in data and "astronomical" in data["tool"]:
            return data["tool"]["astronomical"].get("plugin", data["tool"]["astronomical"])
        return data

    @staticmethod
    def _same_candidate(left: PluginCandidate, right: PluginCandidate) -> bool:
        return (
            left.source == right.source
            and left.module_name == right.module_name
            and left.entry_point_name == right.entry_point_name
            and left.entry_point_object == right.entry_point_object
            and str(left.path or "") == str(right.path or "")
            and str(left.manifest_path or "") == str(right.manifest_path or "")
        )

    def validate(
        self,
        plugin_id: str,
        *,
        astronomical_version: Optional[str] = None,
    ) -> ValidationResult:
        record = self._require_record(plugin_id)
        manifest = record.manifest

        errors: List[str] = []
        warnings: List[str] = []
        missing: List[str] = []

        if astronomical_version:
            compat = self._check_astronomical_version(manifest, astronomical_version)
            errors.extend(compat.errors)
            warnings.extend(compat.warnings)

        required_result = self._validate_requirements(manifest.requires)
        errors.extend(required_result.errors)
        warnings.extend(required_result.warnings)
        missing.extend(required_result.missing_dependencies)

        optional_result = self._validate_requirements(manifest.optional_requires, optional=True)
        warnings.extend(optional_result.warnings)

        for required_plugin in manifest.requires_plugins:
            other = self._records.get(required_plugin)
            if other is None:
                errors.append(f"Missing required AstronomicAL plugin: {required_plugin}")
            elif other.status != PluginStatus.ENABLED:
                errors.append(f"Required AstronomicAL plugin is not enabled: {required_plugin}")

        return ValidationResult(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            missing_dependencies=missing,
        )

    def _validate_requirements(
        self,
        requirements: Sequence[str],
        *,
        optional: bool = False,
    ) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        missing: List[str] = []

        for req in requirements:
            ok, message, skipped = self._check_requirement(req)
            if skipped:
                continue
            if not ok:
                if optional:
                    warnings.append(f"Optional dependency unavailable: {message}")
                else:
                    missing.append(req)
                    errors.append(message)

        return ValidationResult(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            missing_dependencies=missing,
        )

    def _check_astronomical_version(
        self,
        manifest: PluginManifest,
        version: str,
    ) -> ValidationResult:
        if Version is None or SpecifierSet is None:
            return ValidationResult.success(
                warnings=["Cannot check AstronomicAL version: packaging is not installed."]
            )

        errors: List[str] = []
        v = Version(version)
        if manifest.min_astronomical and v < Version(manifest.min_astronomical):
            errors.append(
                f"Plugin requires AstronomicAL >= {manifest.min_astronomical}; current is {version}."
            )
        if manifest.max_astronomical and v > Version(manifest.max_astronomical):
            errors.append(
                f"Plugin requires AstronomicAL <= {manifest.max_astronomical}; current is {version}."
            )
        return ValidationResult(ok=not errors, errors=errors)

    def _check_requirement(self, requirement: str) -> Tuple[bool, str, bool]:
        """Return ``(ok, message, skipped_by_marker)``."""

        if Requirement is None:
            name = requirement.split("=")[0].split("<")[0].split(">")[0].strip()
            try:
                importlib_metadata.version(name)
                return True, "", False
            except importlib_metadata.PackageNotFoundError:
                return False, f"Missing dependency: {requirement}", False

        try:
            req = Requirement(requirement)
        except Exception:
            return False, f"Invalid requirement string: {requirement}", False

        if req.marker is not None and not req.marker.evaluate():
            return True, "", True

        try:
            installed = importlib_metadata.version(req.name)
        except importlib_metadata.PackageNotFoundError:
            return False, f"Missing dependency: {requirement}", False

        if req.specifier and Version is not None:
            if Version(installed) not in req.specifier:
                return False, (
                    f"Dependency {req.name} has version {installed}, "
                    f"but plugin requires {req.specifier}."
                ), False
        return True, "", False

    def enable(
        self,
        plugin_id: str,
        context: Any,
        *,
        astronomical_version: Optional[str] = None,
        validate: bool = True,
    ) -> None:
        boot_print(f"PluginManager.enable: start plugin_id={plugin_id}")
        record = self._require_record(plugin_id)
        if record.status == PluginStatus.ENABLED:
            return

        if validate:
            boot_print(f"PluginManager.enable: validate plugin_id={plugin_id}")
            result = self.validate(plugin_id, astronomical_version=astronomical_version)
            if not result.ok:
                record.status = PluginStatus.ERROR
                record.error = "; ".join(result.errors)
                raise PluginValidationError(record.error)
            
            boot_print(f"PluginManager.enable: validation ok plugin_id={plugin_id}")

        installed_before_error = self._installed_service_keys_by_plugin.setdefault(plugin_id, set())

        try:
            boot_print(f"PluginManager.enable: loading module plugin_id={plugin_id}")
            module = record.module or self._load_module(record.candidate)
            boot_print(
                "PluginManager.enable: module loaded "
                f"plugin_id={plugin_id} module={module.__name__}"
            )
            record.module = module
            api = PluginAPI(self, plugin_id)

            register = getattr(module, "register", None)
            if not callable(register):
                raise PluginLoadError(f"Plugin {plugin_id!r} does not define register(api).")
            
            boot_print(f"PluginManager.enable: calling register(api) plugin_id={plugin_id}")
            register(api)
            boot_print(
                "PluginManager.enable: register(api) complete "
                f"plugin_id={plugin_id} "
                f"panels={[r.id for r in self._panels.values() if r.plugin_id == plugin_id]} "
                f"actions={[r.id for r in self._actions.values() if r.plugin_id == plugin_id]} "
                f"services={[r.key for r in self._services.values() if r.plugin_id == plugin_id]}"
            )
            installed_before_error = self._install_services_for_plugin(plugin_id, context)
            boot_print(
                "PluginManager.enable: services installed "
                f"plugin_id={plugin_id} keys={sorted(installed_before_error)}"
            )
            on_enable = getattr(module, "on_enable", None)
            if callable(on_enable):
                self._call_with_supported_args(on_enable, context=context, manager=self)

            record.status = PluginStatus.ENABLED
            record.error = None
            boot_print(f"PluginManager.enable: publishing plugin.enabled plugin_id={plugin_id}")
            if hasattr(context, "events"):
                context.events.publish(
                    "plugin.enabled",
                    {"plugin_id": plugin_id, "name": record.manifest.name},
                )
            boot_print(f"PluginManager.enable: complete plugin_id={plugin_id}")
        except Exception as exc:
            self._remove_registrations_for_plugin(
                plugin_id,
                context=context,
                service_keys=installed_before_error,
            )
            record.status = PluginStatus.ERROR
            record.error = self._format_exception(exc)
            raise PluginLoadError(f"Failed to enable plugin {plugin_id}: {exc}") from exc

    def disable(
        self,
        plugin_id: str,
        context: Any | None = None,
        *,
        remove_panels: bool = True,
        cancel_jobs: bool = True,
    ) -> None:
        record = self._require_record(plugin_id)
        module = record.module

        if module is not None:
            on_disable = getattr(module, "on_disable", None)
            if callable(on_disable):
                try:
                    self._call_with_supported_args(on_disable, context=context, manager=self)
                except Exception:
                    traceback.print_exc()

        if context is not None:
            if cancel_jobs:
                self._cancel_jobs_for_plugin(plugin_id)
            if remove_panels:
                self._remove_workspace_panels_for_plugin(plugin_id, context)

        self._remove_registrations_for_plugin(plugin_id, context=context)
        record.status = PluginStatus.DISABLED
        record.error = None

        if context is not None and hasattr(context, "events"):
            context.events.publish("plugin.disabled", {"plugin_id": plugin_id})

    def reload(self, plugin_id: str, context: Any) -> None:
        record = self._require_record(plugin_id)
        self.disable(plugin_id, context=context)
        if record.module is not None:
            try:
                record.module = importlib.reload(record.module)
                if record.candidate.manifest_path is not None:
                    record.manifest = self._read_static_manifest(record.candidate.manifest_path)
                else:
                    record.manifest = self._read_manifest(record.module, candidate=record.candidate)
            except Exception as exc:
                record.status = PluginStatus.ERROR
                record.error = self._format_exception(exc)
                raise PluginLoadError(f"Failed to reload plugin {plugin_id}: {exc}") from exc
        self.enable(plugin_id, context, validate=False)

        if hasattr(context, "events"):
            context.events.publish("plugin.reloaded", {"plugin_id": plugin_id})

    def _register_panel(self, registration: PanelRegistration) -> None:
        self._ensure_unique(self._panels, registration.id, "panel")
        self._panels[registration.id] = registration

    def _register_action(self, registration: ActionRegistration) -> None:
        self._ensure_unique(self._actions, registration.id, "action")
        self._actions[registration.id] = registration

    def _register_workflow(self, registration: WorkflowRegistration) -> None:
        self._ensure_unique(self._workflows, registration.id, "workflow")
        self._workflows[registration.id] = registration

    def _register_service(self, registration: ServiceRegistration) -> None:
        if registration.key in self._services and not registration.replace:
            raise PluginRegistrationError(
                f"Service key {registration.key!r} is already registered."
            )
        self._services[registration.key] = registration

    def _register_artifact_viewer(self, registration: ArtifactViewerRegistration) -> None:
        regs = self._artifact_viewers.setdefault(registration.artifact_type, [])
        if registration.id and any(r.id == registration.id for r in regs):
            raise PluginRegistrationError(
                f"Duplicate artifact viewer id: {registration.id!r}"
            )
        regs.append(registration)
        regs.sort(key=lambda r: (not r.default, r.priority, r.title or "", r.id or ""))

    def _register_settings_schema(self, plugin_id: str, schema: Dict[str, Any]) -> None:
        record = self._require_record(plugin_id)
        record.settings_schema = schema

    def _ensure_unique(self, registry: Dict[str, Any], id: str, kind: str) -> None:
        if id in registry:
            raise PluginRegistrationError(f"Duplicate {kind} registration id: {id}")

    def get_plugin_setting(self, plugin_id: str, key: str, default: Any = None) -> Any:
        self._require_record(plugin_id)
        return self.settings.get(plugin_id, key, default)

    def set_plugin_setting(self, plugin_id: str, key: str, value: Any) -> None:
        self._require_record(plugin_id)
        self.settings.set(plugin_id, key, value)

    def get_plugin_settings(self, plugin_id: str) -> Dict[str, Any]:
        self._require_record(plugin_id)
        return self.settings.get_all(plugin_id)

    def update_plugin_settings(self, plugin_id: str, values: Dict[str, Any]) -> None:
        self._require_record(plugin_id)
        self.settings.update(plugin_id, values)

    def _registration_plugin_ids(self) -> set[str]:
        """Return plugin ids that have registered contributions.

        This catches orphan registrations where panels/actions/services exist
        but the plugin does not currently have a discovered record in
        ``self._records``. That should not be the normal long-term path, but it
        is useful during migration and prevents Plugin Manager diagnostics from
        hiding registered contributions.
        """
        plugin_ids: set[str] = set()

        for registry in (
            self._panels,
            self._actions,
            self._workflows,
            self._services,
        ):
            for registration in registry.values():
                plugin_id = getattr(registration, "plugin_id", None)
                if plugin_id:
                    plugin_ids.add(str(plugin_id))

        for registrations in self._artifact_viewers.values():
            for registration in registrations:
                plugin_id = getattr(registration, "plugin_id", None)
                if plugin_id:
                    plugin_ids.add(str(plugin_id))

        return plugin_ids

    def _synthetic_plugin_info(self, plugin_id: str) -> PluginInfo:
        """Build PluginInfo for a plugin id that has registrations but no record.

        This should mainly happen during transitional code paths where a module
        registers contributions directly without going through full discovery.
        The status is marked enabled because contributions are already live.
        """
        display_name = plugin_id.replace("_", " ").replace(".", " / ").title()

        return PluginInfo(
            id=plugin_id,
            name=display_name,
            version="unknown",
            status=PluginStatus.ENABLED,
            description=(
                "Runtime-registered plugin contributions were found, but no "
                "discovered plugin record exists. This usually means the plugin "
                "was registered through a transitional/direct-registration path."
            ),
            source="runtime_registration",
            path=None,
            error=None,
            capabilities=[],
            tags=[],
            requires=[],
            optional_requires=[],
            requires_plugins=[],
            panels=[r.id for r in self._panels.values() if r.plugin_id == plugin_id],
            actions=[r.id for r in self._actions.values() if r.plugin_id == plugin_id],
            workflows=[r.id for r in self._workflows.values() if r.plugin_id == plugin_id],
            services=[r.key for r in self._services.values() if r.plugin_id == plugin_id],
            artifact_viewers=[
                r.artifact_type
                for regs in self._artifact_viewers.values()
                for r in regs
                if r.plugin_id == plugin_id
            ],
        )


    def list_plugins(self) -> List[PluginInfo]:
        plugin_ids = set(self._records)
        plugin_ids.update(self._registration_plugin_ids())

        return [self.plugin_info(plugin_id) for plugin_id in sorted(plugin_ids)]

    def plugin_info(self, plugin_id: str) -> PluginInfo:
        record = self._records.get(plugin_id)

        if record is None:
            if plugin_id in self._registration_plugin_ids():
                return self._synthetic_plugin_info(plugin_id)
            raise KeyError(f"Unknown AstronomicAL plugin: {plugin_id}")

        manifest = record.manifest

        return PluginInfo(
            id=manifest.id,
            name=manifest.name,
            version=manifest.version,
            status=record.status,
            description=manifest.description,
            source=record.candidate.source,
            path=str(record.candidate.path) if record.candidate.path else None,
            error=record.error,
            capabilities=list(manifest.capabilities),
            tags=list(manifest.tags),
            requires=list(manifest.requires),
            optional_requires=list(manifest.optional_requires),
            requires_plugins=list(manifest.requires_plugins),
            panels=[r.id for r in self._panels.values() if r.plugin_id == plugin_id],
            actions=[r.id for r in self._actions.values() if r.plugin_id == plugin_id],
            workflows=[r.id for r in self._workflows.values() if r.plugin_id == plugin_id],
            services=[r.key for r in self._services.values() if r.plugin_id == plugin_id],
            artifact_viewers=[
                r.artifact_type
                for regs in self._artifact_viewers.values()
                for r in regs
                if r.plugin_id == plugin_id
            ],
        )

    def list_panels(self) -> List[PanelRegistration]:
        return sorted(self._panels.values(), key=lambda r: (r.category or "", r.title, r.id))

    def list_actions(self) -> List[ActionRegistration]:
        return sorted(self._actions.values(), key=lambda r: (r.category or "", r.title, r.id))

    def list_workflows(self) -> List[WorkflowRegistration]:
        return sorted(self._workflows.values(), key=lambda r: (r.category or "", r.title, r.id))

    def list_services(self) -> List[ServiceRegistration]:
        return sorted(self._services.values(), key=lambda r: r.key)

    def list_artifact_viewers(
        self,
        artifact_type: Optional[str] = None,
    ) -> List[ArtifactViewerRegistration]:
        if artifact_type is not None:
            return list(self._artifact_viewers.get(artifact_type, []))
        return [r for regs in self._artifact_viewers.values() for r in regs]

    def get_panel(self, panel_id: str) -> PanelRegistration:
        try:
            return self._panels[panel_id]
        except KeyError as exc:
            raise KeyError(f"Unknown plugin panel registration: {panel_id}") from exc

    def get_action(self, action_id: str) -> ActionRegistration:
        return self._actions[action_id]

    def get_workflow(self, workflow_id: str) -> WorkflowRegistration:
        return self._workflows[workflow_id]


    def create_panel(
        self,
        panel_id: str,
        context: Any,
        *,
        instance_id: Optional[str] = None,
        restore_state: Optional[Dict[str, Any]] = None,
        restore_metadata: Optional[Dict[str, Any]] = None,
        open_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """
        Backwards-friendly panel factory.

        Existing callers can still do:

            view, controller = manager.create_panel(panel_id, context)

        New workspace restore code should usually call open_panel() so panel
        metadata and layout are recorded in WorkspaceManager.
        """
        reg = self.get_panel(panel_id)

        merged_kwargs = dict(getattr(reg, "default_open_kwargs", {}) or {})
        merged_kwargs.update(open_kwargs or {})
        merged_kwargs.update(kwargs)

        dependency_validation = self._validate_registration_dependencies(
            reg.requires,
            optional_requires=reg.optional_requires,
        )
        if not dependency_validation.ok:
            raise PluginValidationError("; ".join(dependency_validation.errors))

        if getattr(reg, "required_mappings", None) or getattr(reg, "optional_mappings", None):
            gate = MappingGatedPanel(
                context=context,
                manager=self,
                registration=reg,
                kwargs=merged_kwargs,
                instance_id=instance_id,
                restore_state=restore_state or {},
                restore_metadata=restore_metadata or {},
            )
            return gate.view, gate

        return self._create_panel_now(
            reg,
            context,
            instance_id=instance_id,
            restore_state=restore_state or {},
            restore_metadata=restore_metadata or {},
            **merged_kwargs,
        )

    def create_panel_instance(
        self,
        panel_id: str,
        context: Any,
        *,
        instance_id: Optional[str] = None,
        restore_state: Optional[Dict[str, Any]] = None,
        restore_metadata: Optional[Dict[str, Any]] = None,
        open_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CreatedPanel:
        reg = self.get_panel(panel_id)
        resolved_instance_id = instance_id or f"{reg.id}:{uuid.uuid4().hex[:10]}"

        view, controller = self.create_panel(
            panel_id,
            context,
            instance_id=resolved_instance_id,
            restore_state=restore_state,
            restore_metadata=restore_metadata,
            open_kwargs=open_kwargs,
            **kwargs,
        )

        return CreatedPanel(
            view=view,
            controller=controller,
            registration=reg,
            instance_id=resolved_instance_id,
            title=reg.title,
        )

    def _create_panel_now(
        self,
        reg: PanelRegistration,
        context: Any,
        *,
        instance_id: Optional[str] = None,
        restore_state: Optional[Dict[str, Any]] = None,
        restore_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """
        Create a plugin panel immediately.

        MappingGatedPanel calls this once semantic column requirements are
        satisfied.
        """
        try:
            result = self._call_with_supported_args(
                reg.factory,
                context=context,
                manager=self,
                instance_id=instance_id,
                restore_state=restore_state or {},
                restore_metadata=restore_metadata or {},
                **kwargs,
            )
            view, controller = self._normalise_panel_result(result)

            try:
                setattr(view, "_al_plugin_id", reg.plugin_id)
                setattr(view, "_al_registration_id", reg.id)
                setattr(view, "_al_instance_id", instance_id)
                setattr(view, "_al_plugin_version", self._plugin_version(reg.plugin_id))
                setattr(view, "_al_state_version", reg.state_version)
            except Exception:
                pass

            if getattr(reg, "persist_state", True):
                restore_controller_state(controller, restore_state or {})

            return view, controller

        except Exception as exc:
            raise PluginExecutionError(f"Failed to create panel {reg.id}: {exc}") from exc

    def _workspace_panel_exists(self, context: Any, workspace_id: str) -> bool:

        workspace = getattr(context, "workspace", None)
        if workspace is None:
            return False

        try:
            return str(workspace_id) in workspace.list_panels()
        except Exception:
            try:
                keys = [str(key) for key in (workspace.grid.keys or [])]
                return str(workspace_id) in keys
            except Exception:
                return True


    def add_panel_to_workspace(
        self,
        panel_id: str,
        context: Any,
        *,
        instance_id: Optional[str] = None,
        title: Optional[str] = None,
        layout_item: Optional[Dict[str, Any]] = None,
        layout_items: Optional[Dict[str, Dict[str, Any]]] = None,
        restore_state: Optional[Dict[str, Any]] = None,
        restore_metadata: Optional[Dict[str, Any]] = None,
        open_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:

        reg = self.get_panel(panel_id)

        plugin_open_debug(
            "add_panel_to_workspace START",
            requested_panel_id=panel_id,
            registration_id=reg.id,
            plugin_id=reg.plugin_id,
            title=title or reg.title,
        )

        workspace_id = instance_id or f"{reg.id}:{uuid.uuid4().hex[:10]}"
        visible_title = title or reg.title

        loading_view, loading_controller = make_loading_panel(
            visible_title,
            detail="Preparing panel...",
        )

        context.workspace.add_panel(
            panel_id=workspace_id,
            title=visible_title,
            view=loading_view,
            controller=loading_controller,
            layout_item=layout_item if layout_item is not None else reg.default_layout,
            layout_items=layout_items,
            kind="loading_panel",
            plugin_id=reg.plugin_id,
            registration_id=reg.id,
            plugin_version=self._plugin_version(reg.plugin_id),
            state_version=1,
            persistent=False,
            open_kwargs=open_kwargs or {},
            metadata={
                "loading_for": reg.id,
                "panel_registration_title": reg.title,
                "panel_category": reg.category,
                "restore_policy": reg.restore_policy,
            },
        )

        plugin_open_debug(
            "loading_panel ADDED",
            workspace_id=workspace_id,
            registration_id=reg.id,
            plugin_id=reg.plugin_id,
            workspace_keys=list(getattr(context.workspace.grid, "keys", []) or []),
        )

        self._workspace_panels_by_plugin.setdefault(reg.plugin_id, set()).add(workspace_id)

        def _build_panel_job(*, cancel_token) -> CreatedPanel:
            if cancel_token is not None and cancel_token.cancelled():
                raise RuntimeError("Panel opening was cancelled.")

            plugin_open_debug(
                "build_panel_job START",
                workspace_id=workspace_id,
                registration_id=reg.id,
                plugin_id=reg.plugin_id,
            )

            created = self.create_panel_instance(
                panel_id,
                context,
                instance_id=workspace_id,
                restore_state=restore_state,
                restore_metadata=restore_metadata,
                open_kwargs=open_kwargs,
                **kwargs,
            )

            plugin_open_debug(
                "build_panel_job CREATED",
                workspace_id=workspace_id,
                registration_id=reg.id,
                created_type=type(created).__name__,
                view_type=type(getattr(created, "view", None)).__name__,
                controller_type=type(getattr(created, "controller", None)).__name__,
            )

            if cancel_token is not None and cancel_token.cancelled():
                raise RuntimeError("Panel opening was cancelled.")

            return created

        def _on_panel_ready(created: CreatedPanel) -> None:
            if not self._workspace_panel_exists(context, workspace_id):
                # The user closed the loading tile before the panel finished.
                controller = getattr(created, "controller", None)
                view = getattr(created, "view", None)
                try:
                    if controller is not None and hasattr(controller, "dispose"):
                        controller.dispose()
                except Exception:
                    pass
                try:
                    if view is not None and view is not controller and hasattr(view, "dispose"):
                        view.dispose()
                except Exception:
                    pass
                return

            context.workspace.add_panel(
                panel_id=workspace_id,
                title=visible_title,
                view=created.view,
                controller=created.controller,
                layout_item=layout_item if layout_item is not None else reg.default_layout,
                layout_items=layout_items,
                kind="plugin_panel",
                plugin_id=reg.plugin_id,
                registration_id=reg.id,
                plugin_version=self._plugin_version(reg.plugin_id),
                state_version=reg.state_version,
                persistent=reg.persist_layout,
                open_kwargs=open_kwargs or {},
                metadata={
                    "panel_registration_title": reg.title,
                    "panel_category": reg.category,
                    "restore_policy": reg.restore_policy,
                },
            )

            plugin_open_debug(
                "plugin_panel ADDED",
                workspace_id=workspace_id,
                registration_id=reg.id,
                plugin_id=reg.plugin_id,
                workspace_keys=list(getattr(context.workspace.grid, "keys", []) or []),
            )

            if hasattr(context, "events"):
                try:
                    context.events.publish(
                        "plugin.panel.opened",
                        {
                            "panel_id": workspace_id,
                            "registration_id": reg.id,
                            "plugin_id": reg.plugin_id,
                            "title": visible_title,
                        },
                    )
                except Exception:
                    pass

        def _on_panel_error(exc: BaseException) -> None:

            plugin_open_debug(
                "on_panel_error ENTER",
                workspace_id=workspace_id,
                registration_id=reg.id,
                plugin_id=reg.plugin_id,
                error=repr(exc),
            )
            traceback.print_exception(type(exc), exc, exc.__traceback__)

            if not self._workspace_panel_exists(context, workspace_id):
                return

            error_view, error_controller = make_panel_load_error_panel(
                visible_title,
                exc,
            )

            context.workspace.add_panel(
                panel_id=workspace_id,
                title=f"{visible_title} failed",
                view=error_view,
                controller=error_controller,
                layout_item=layout_item if layout_item is not None else reg.default_layout,
                layout_items=layout_items,
                kind="error_panel",
                plugin_id=reg.plugin_id,
                registration_id=reg.id,
                plugin_version=self._plugin_version(reg.plugin_id),
                state_version=1,
                persistent=False,
                open_kwargs=open_kwargs or {},
                metadata={
                    "failed_panel_registration": reg.id,
                    "panel_registration_title": reg.title,
                    "panel_category": reg.category,
                    "restore_policy": reg.restore_policy,
                    "error": str(exc),
                },
            )

            logger.exception("Failed to open plugin panel %s", reg.id, exc_info=exc)

            if hasattr(context, "events"):
                try:
                    context.events.publish(
                        "plugin.panel.open_failed",
                        {
                            "panel_id": workspace_id,
                            "registration_id": reg.id,
                            "plugin_id": reg.plugin_id,
                            "title": visible_title,
                            "error": str(exc),
                        },
                    )
                except Exception:
                    pass

        jobs = getattr(context, "jobs", None)

        if jobs is not None:

            plugin_open_debug(
                "jobs.submit PANEL_OPEN",
                workspace_id=workspace_id,
                registration_id=reg.id,
                plugin_id=reg.plugin_id,
            )
            handle = jobs.submit(
                _build_panel_job,
                title=f"Open panel: {visible_title}",
                key=f"plugin-panel-open:{workspace_id}",
                on_done=_on_panel_ready,
                on_error=_on_panel_error,
            )

            plugin_open_debug(
                "jobs.submit RETURNED",
                workspace_id=workspace_id,
                job_id=getattr(handle, "job_id", None),
                title=getattr(handle, "title", None),
            )

            self._job_handles_by_plugin.setdefault(reg.plugin_id, []).append(handle)
        else:
            # Fallback for tests or contexts without JobManager.
            try:
                _on_panel_ready(_build_panel_job(cancel_token=None))
            except BaseException as exc:
                _on_panel_error(exc)

        return workspace_id



    def open_panel(
        self,
        panel_id: str,
        *,
        context: Any,
        instance_id: Optional[str] = None,
        title: Optional[str] = None,
        layout_item: Optional[Dict[str, Any]] = None,
        layout_items: Optional[Dict[str, Dict[str, Any]]] = None,
        restore_state: Optional[Dict[str, Any]] = None,
        restore_metadata: Optional[Dict[str, Any]] = None,
        open_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Preferred workspace-facing panel opener.

        This is intentionally thin over add_panel_to_workspace(), but the name
        reads better in persistence and menu code.
        """

        plugin_open_debug(
            "open_panel ENTER",
            panel_id=panel_id,
            instance_id=instance_id,
            title=title,
            open_kwargs=open_kwargs,
        )


        return self.add_panel_to_workspace(
            panel_id,
            context,
            instance_id=instance_id,
            title=title,
            layout_item=layout_item,
            layout_items=layout_items,
            restore_state=restore_state,
            restore_metadata=restore_metadata,
            open_kwargs=open_kwargs,
            **kwargs,
        )
    
    def _plugin_version(self, plugin_id: str) -> Optional[str]:
        record = self._records.get(plugin_id)
        if record is None:
            return None
        return getattr(record.manifest, "version", None)

    def required_plugins_for_workspace(self, workspace_snapshot: Dict[str, Any]) -> List[str]:
        required: set[str] = set()

        for panel in workspace_snapshot.get("panels", []) or []:
            if not isinstance(panel, dict):
                continue
            plugin_id = panel.get("plugin_id")
            if plugin_id:
                required.add(str(plugin_id))

        return sorted(required)

    def run_action(
        self,
        action_id: str,
        context: Any,
        request: Optional[ActionRequest | Dict[str, Any]] = None,
        *,
        on_done: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
        return_processed: bool = False,
    ) -> Any:
        reg = self.get_action(action_id)

        dependency_validation = self._validate_registration_dependencies(
            reg.requires,
            optional_requires=reg.optional_requires,
        )
        if not dependency_validation.ok:
            raise PluginValidationError("; ".join(dependency_validation.errors))

        req = request if isinstance(request, ActionRequest) else ActionRequest.from_dict(request)
        self._fill_default_request(context, req)
        req.params = self._prepare_params(
            reg.params_schema,
            req.params,
            action_id=reg.id,
        )

        validation = self._validate_action_request(reg, context, req)
        if not validation.ok:
            raise PluginValidationError("; ".join(validation.errors))

        def _run(*, cancel_token=None):
            raw = self._call_with_supported_args(
                reg.handler,
                context=context,
                request=req,
                manager=self,
                cancel_token=cancel_token,
            )
            processed = self._process_action_result(
                raw,
                context=context,
                request=req,
                registration=reg,
            )
            return processed if return_processed else processed.legacy_return()

        if not reg.run_in_job:
            return _run(cancel_token=None)

        key = self._action_job_key(reg, req)
        handle_holder: Dict[str, Any] = {"handle": None, "pending_cleanup": False}

        def _cleanup_handle() -> None:
            handle = handle_holder.get("handle")
            if handle is None:
                handle_holder["pending_cleanup"] = True
                return
            self._remove_job_handle(reg.plugin_id, handle)

        def _on_done(result: Any) -> None:
            _cleanup_handle()
            if on_done is not None:
                on_done(result)

        def _on_error(exc: BaseException) -> None:
            _cleanup_handle()
            if on_error is not None:
                on_error(exc)

        handle = context.jobs.submit(
            _run,
            title=reg.title,
            key=key,
            on_done=_on_done,
            on_error=_on_error,
        )
        handle_holder["handle"] = handle
        self._job_handles_by_plugin.setdefault(reg.plugin_id, []).append(handle)
        if handle_holder.get("pending_cleanup"):
            self._remove_job_handle(reg.plugin_id, handle)
        return handle

    def build_workflow(
        self,
        workflow_id: str,
        context: Any,
        request: Optional[Dict[str, Any]] = None,
    ) -> Any:
        reg = self.get_workflow(workflow_id)

        dependency_validation = self._validate_registration_dependencies(
            reg.requires,
            optional_requires=reg.optional_requires,
        )
        if not dependency_validation.ok:
            raise PluginValidationError("; ".join(dependency_validation.errors))

        return self._call_with_supported_args(
            reg.builder,
            context=context,
            request=request or {},
            manager=self,
        )

    def create_artifact_viewer(
        self,
        artifact_type: str,
        context: Any,
        artifact_id: str,
        *,
        viewer_id: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        viewers = self._artifact_viewers.get(artifact_type, [])
        if viewer_id is not None:
            viewers = [viewer for viewer in viewers if viewer.id == viewer_id]
        if not viewers:
            raise KeyError(f"No artifact viewer registered for {artifact_type!r}")

        reg = viewers[0]
        dependency_validation = self._validate_registration_dependencies(
            reg.requires,
            optional_requires=reg.optional_requires,
        )
        if not dependency_validation.ok:
            raise PluginValidationError("; ".join(dependency_validation.errors))

        result = self._call_with_supported_args(
            reg.viewer_factory,
            context=context,
            artifact_id=artifact_id,
            manager=self,
        )
        return self._normalise_panel_result(result)

    def _load_module(self, candidate: PluginCandidate) -> ModuleType:
        if candidate.module is not None:
            return candidate.module

        try:
            if candidate.path is not None:
                return self._load_module_from_path(candidate.module_name, candidate.path)
            return importlib.import_module(candidate.module_name)
        except Exception as exc:
            raise PluginLoadError(f"Failed to import {candidate.module_name}: {exc}") from exc

    def _load_module_from_path(self, module_name: str, path: Path) -> ModuleType:
        path = path.resolve()
        if not path.exists():
            raise PluginLoadError(f"Plugin file does not exist: {path}")
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Could not create import spec for {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    def _read_manifest(
        self,
        module: ModuleType,
        *,
        candidate: Optional[PluginCandidate] = None,
    ) -> PluginManifest:
        if (
            candidate is not None
            and candidate.entry_point_object
            and self._looks_like_manifest_file(candidate.entry_point_object)
        ):
            if candidate.manifest_path is not None:
                return self._read_static_manifest(candidate.manifest_path)
            raise PluginLoadError(
                f"Static manifest {candidate.entry_point_object!r} could not be resolved."
            )

        if candidate is not None and candidate.entry_point_object:
            obj = getattr(module, candidate.entry_point_object, None)
            if obj is None:
                raise PluginLoadError(
                    f"Entry point object {candidate.entry_point_object!r} was not found "
                    f"in module {module.__name__!r}."
                )
            if callable(obj) and candidate.entry_point_object.startswith("get_"):
                obj = obj()
            return coerce_manifest(obj)

        if hasattr(module, "manifest"):
            return coerce_manifest(getattr(module, "manifest"))
        if hasattr(module, "PLUGIN_MANIFEST"):
            return coerce_manifest(getattr(module, "PLUGIN_MANIFEST"))
        get_manifest = getattr(module, "get_manifest", None)
        if callable(get_manifest):
            return coerce_manifest(get_manifest())
        raise PluginLoadError(
            f"Plugin module {module.__name__!r} must expose manifest, "
            "PLUGIN_MANIFEST, or get_manifest()."
        )

    def _install_services_for_plugin(self, plugin_id: str, context: Any) -> set[str]:
        installed = self._installed_service_keys_by_plugin.setdefault(plugin_id, set())
        newly_installed: set[str] = set()

        try:
            for reg in list(self._services.values()):
                if reg.plugin_id != plugin_id:
                    continue

                dependency_validation = self._validate_registration_dependencies(
                    reg.requires,
                    optional_requires=reg.optional_requires,
                )
                if not dependency_validation.ok:
                    raise PluginValidationError("; ".join(dependency_validation.errors))

                def _factory(reg=reg, context=context):
                    return self._call_with_supported_args(
                        reg.factory,
                        context=context,
                        manager=self,
                        settings=self.settings.get_all(plugin_id),
                    )

                if hasattr(context.services, "set_factory"):
                    context.services.set_factory(
                        reg.key,
                        _factory,
                        lazy=reg.lazy,
                        replace=reg.replace,
                        owner=plugin_id,
                    )
                    installed.add(reg.key)
                    newly_installed.add(reg.key)
                else:
                    if not reg.lazy:
                        context.services.set(reg.key, _factory(), owner=plugin_id)
                        installed.add(reg.key)
                        newly_installed.add(reg.key)
                    elif not context.services.has(reg.key):
                        context.services.set(
                            reg.key,
                            _LazyServiceProxy(reg.key, _factory),
                            owner=plugin_id,
                        )
                        installed.add(reg.key)
                        newly_installed.add(reg.key)

            return set(installed)
        except Exception:
            self._remove_installed_services(
                plugin_id,
                context=context,
                service_keys=newly_installed,
            )
            raise

    def _remove_installed_services(
        self,
        plugin_id: str,
        *,
        context: Any,
        service_keys: set[str],
    ) -> None:
        for key in list(service_keys):
            if context is None or not hasattr(context, "services"):
                continue

            remove = getattr(context.services, "remove", None)
            if callable(remove):
                try:
                    remove(key, dispose=True, owner=plugin_id)
                except TypeError:
                    remove(key, dispose=True)

            installed = self._installed_service_keys_by_plugin.get(plugin_id)
            if installed is not None:
                installed.discard(key)

    def _remove_registrations_for_plugin(
        self,
        plugin_id: str,
        context: Any | None = None,
        service_keys: Optional[set[str]] = None,
    ) -> None:
        self._panels = {k: v for k, v in self._panels.items() if v.plugin_id != plugin_id}
        self._actions = {k: v for k, v in self._actions.items() if v.plugin_id != plugin_id}
        self._workflows = {k: v for k, v in self._workflows.items() if v.plugin_id != plugin_id}

        installed_keys = set(
            service_keys
            if service_keys is not None
            else self._installed_service_keys_by_plugin.get(plugin_id, set())
        )

        for key in list(installed_keys):
            if context is not None and hasattr(context, "services"):
                remove = getattr(context.services, "remove", None)
                if callable(remove):
                    try:
                        remove(key, dispose=True, owner=plugin_id)
                    except TypeError:
                        remove(key, dispose=True)

        self._services = {k: v for k, v in self._services.items() if v.plugin_id != plugin_id}
        self._installed_service_keys_by_plugin.pop(plugin_id, None)

        for artifact_type, regs in list(self._artifact_viewers.items()):
            kept = [r for r in regs if r.plugin_id != plugin_id]
            if kept:
                self._artifact_viewers[artifact_type] = kept
            else:
                self._artifact_viewers.pop(artifact_type, None)

    def _validate_registration_dependencies(
        self,
        requires: Sequence[str],
        *,
        optional_requires: Sequence[str] = (),
    ) -> ValidationResult:
        required_result = self._validate_requirements(requires)
        optional_result = self._validate_requirements(optional_requires, optional=True)

        return ValidationResult(
            ok=required_result.ok,
            errors=required_result.errors,
            warnings=required_result.warnings + optional_result.warnings,
            missing_dependencies=required_result.missing_dependencies,
        )

    def _validate_action_request(
        self,
        reg: ActionRegistration,
        context: Any,
        request: ActionRequest,
    ) -> ValidationResult:
        errors: List[str] = []
        inputs: InputSpec = reg.inputs

        if inputs.dataset and not request.dataset_id:
            errors.append(f"Action {reg.id!r} requires a dataset.")

        if inputs.selection == "required" and not request.row_ids:
            errors.append(f"Action {reg.id!r} requires a focused row or selection set.")

        if inputs.columns == "one" and len(request.columns) != 1:
            errors.append(f"Action {reg.id!r} requires exactly one column.")
        elif inputs.columns == "many" and len(request.columns) < 1:
            errors.append(f"Action {reg.id!r} requires one or more columns.")

        if inputs.numeric_columns in {"one", "many"}:
            if inputs.numeric_columns == "one" and len(request.columns) != 1:
                errors.append(f"Action {reg.id!r} requires exactly one numeric column.")
            elif inputs.numeric_columns == "many" and len(request.columns) < 1:
                errors.append(f"Action {reg.id!r} requires one or more numeric columns.")
            else:
                errors.extend(self._validate_numeric_columns(context, request))

        if request.artifact_id and inputs.accepts_artifact_types:
            artifact_type = self._artifact_type(context, request.artifact_id)
            if artifact_type and artifact_type not in inputs.accepts_artifact_types:
                errors.append(
                    f"Action {reg.id!r} does not accept artifact type {artifact_type!r}."
                )

        for mapping in inputs.required_mappings:
            if not self._mapping_exists(context, request.dataset_id, mapping):
                errors.append(f"Action {reg.id!r} requires dataset mapping {mapping!r}.")

        return ValidationResult(ok=not errors, errors=errors)

    def _validate_numeric_columns(self, context: Any, request: ActionRequest) -> List[str]:
        if not request.dataset_id or not request.columns:
            return []

        try:
            df = context.datasets.get_df(request.dataset_id)
        except Exception:
            return []

        errors: List[str] = []
        for column in request.columns:
            if column not in getattr(df, "columns", []):
                errors.append(f"Column {column!r} does not exist in dataset {request.dataset_id!r}.")
                continue

            try:
                dtype = df[column].dtype
                if not _is_numeric_dtype(dtype):
                    errors.append(f"Column {column!r} is not numeric.")
            except Exception:
                pass
        return errors

    def _artifact_type(self, context: Any, artifact_id: str) -> Optional[str]:
        store = getattr(context, "artifacts", None)
        if store is None:
            return None

        for attr in ("metadata", "get_metadata", "info"):
            method = getattr(store, attr, None)
            if callable(method):
                try:
                    meta = method(artifact_id)
                    if isinstance(meta, dict):
                        return meta.get("type")
                    return getattr(meta, "type", None)
                except Exception:
                    pass

        return None

    def _mapping_exists(self, context: Any, dataset_id: Optional[str], mapping: str) -> bool:
        datasets = getattr(context, "datasets", None)
        if datasets is None or not dataset_id:
            return False

        for method_name in ("get_mapping", "mapping", "get_column_mapping"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    value = method(dataset_id, mapping)
                    if value:
                        return True
                except Exception:
                    pass

        return False

    def _prepare_params(
        self,
        schema: Dict[str, Any],
        params: Optional[Dict[str, Any]],
        *,
        action_id: str,
    ) -> Dict[str, Any]:
        prepared = dict(params or {})
        if not schema:
            return prepared

        if schema.get("type") and not self._json_type_matches(prepared, schema["type"]):
            raise PluginValidationError(
                f"Action {action_id!r} params must match schema type {schema['type']!r}."
            )

        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for name, prop_schema in properties.items():
                if (
                    isinstance(prop_schema, dict)
                    and name not in prepared
                    and "default" in prop_schema
                ):
                    prepared[name] = _json_safe(prop_schema["default"])

        errors = self._validate_params_against_schema(
            prepared,
            schema,
            path="params",
        )
        if errors:
            raise PluginValidationError(
                f"Invalid params for action {action_id!r}: " + "; ".join(errors)
            )

        return prepared

    def _validate_params_against_schema(
        self,
        value: Any,
        schema: Dict[str, Any],
        *,
        path: str,
    ) -> List[str]:
        errors: List[str] = []

        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path} must be one of {schema['enum']!r}.")
            return errors

        expected_type = schema.get("type")
        if expected_type is not None and not self._json_type_matches(value, expected_type):
            errors.append(f"{path} must be of type {expected_type!r}.")
            return errors

        if value is None:
            return errors

        if isinstance(value, dict):
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
                for key in required:
                    if key not in value:
                        errors.append(f"{path}.{key} is required.")

            if schema.get("additionalProperties") is False and isinstance(properties, dict):
                allowed = set(properties)
                for key in value:
                    if key not in allowed:
                        errors.append(f"{path}.{key} is not allowed.")

            if isinstance(properties, dict):
                for key, prop_schema in properties.items():
                    if key in value and isinstance(prop_schema, dict):
                        errors.extend(
                            self._validate_params_against_schema(
                                value[key],
                                prop_schema,
                                path=f"{path}.{key}",
                            )
                        )

        if isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                errors.append(f"{path} must contain at least {schema['minItems']} item(s).")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                errors.append(f"{path} must contain at most {schema['maxItems']} item(s).")

            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    errors.extend(
                        self._validate_params_against_schema(
                            item,
                            item_schema,
                            path=f"{path}[{index}]",
                        )
                    )

        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{path} must be at least {schema['minLength']} character(s).")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{path} must be at most {schema['maxLength']} character(s).")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{path} must be >= {schema['minimum']}.")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{path} must be <= {schema['maximum']}.")

        return errors

    @staticmethod
    def _json_type_matches(value: Any, expected: Any) -> bool:
        if isinstance(expected, list):
            return any(PluginManager._json_type_matches(value, item) for item in expected)

        if expected == "null":
            return value is None
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "string":
            return isinstance(value, str)
        if expected == "array":
            return isinstance(value, list)
        if expected == "object":
            return isinstance(value, dict)

        return True

    def _process_action_result(
        self,
        raw: Any,
        *,
        context: Any,
        request: ActionRequest,
        registration: ActionRegistration,
    ) -> ProcessedActionResult:
        if isinstance(raw, ArtifactResult):
            result = ActionResult(artifacts=[raw])
        elif isinstance(raw, DatasetResult):
            result = ActionResult(datasets=[raw])
        elif isinstance(raw, EventResult):
            result = ActionResult(events=[raw])
        elif isinstance(raw, ActionResult):
            result = raw
        else:
            if raw is not None and len(registration.outputs) == 1:
                result = ActionResult(
                    artifacts=[
                        ArtifactResult(
                            type=registration.outputs[0].type,
                            payload=raw,
                            dataset_id=request.dataset_id,
                            row_ids=request.row_ids,
                            params=dict(request.params),
                        )
                    ],
                    value=raw,
                )
            else:
                return ProcessedActionResult(raw=raw, value=raw)

        artifact_ids: List[str] = []
        dataset_ids: List[str] = []
        published_events: List[EventResult] = []

        for dataset_result in result.datasets:
            metadata = dict(dataset_result.metadata)
            try:
                context.datasets.register(
                    dataset_result.id,
                    dataset_result.dataframe,
                    name=dataset_result.name or dataset_result.id,
                    **metadata,
                )
            except TypeError:
                context.datasets.register(dataset_result.id, dataset_result.dataframe)

            dataset_ids.append(dataset_result.id)

            if dataset_result.set_active and hasattr(context.datasets, "set_active"):
                context.datasets.set_active(dataset_result.id)

        for artifact_result in result.artifacts:
            dataset_id = artifact_result.dataset_id or request.dataset_id
            artifact_id = artifact_result.artifact_id

            params = dict(artifact_result.params)
            provenance = dict(params.get("provenance", {}))
            provenance.setdefault("plugin_id", registration.plugin_id)
            provenance.setdefault("action_id", registration.id)
            params["provenance"] = provenance

            if artifact_id is None:
                artifact_id = context.artifacts.put(
                    artifact_result.type,
                    artifact_result.payload,
                    dataset_id=dataset_id,
                    row_ids=artifact_result.row_ids or request.row_ids,
                    params=params,
                )
                artifact_result.artifact_id = artifact_id

            artifact_ids.append(artifact_id)

            if artifact_result.publish and hasattr(context, "events"):
                event = EventResult(
                    "artifact.created",
                    {
                        "artifact_id": artifact_id,
                        "type": artifact_result.type,
                        "dataset_id": dataset_id,
                        "origin": registration.id,
                        "plugin_id": registration.plugin_id,
                    },
                )
                context.events.publish(event.topic, event.payload)
                published_events.append(event)

        for event in result.events:
            if hasattr(context, "events"):
                context.events.publish(event.topic, event.payload)
            published_events.append(event)

        return ProcessedActionResult(
            raw=raw,
            value=result.value,
            artifact_ids=artifact_ids,
            dataset_ids=dataset_ids,
            events=published_events,
            result=result,
        )

    def _is_action_result_like(self, value: Any) -> bool:
        return isinstance(value, (ActionResult, ArtifactResult, DatasetResult, EventResult))

    def _fill_default_request(self, context: Any, request: ActionRequest) -> None:
        if request.dataset_id is None and hasattr(context, "datasets"):
            try:
                request.dataset_id = context.datasets.active_id()
            except Exception:
                pass

        if request.row_ids is None and hasattr(context, "selection"):
            try:
                active = context.selection.get_active_set()
            except Exception:
                active = None

            if active is not None:
                request.row_ids = list(
                    active.row_ids if hasattr(active, "row_ids") else active["row_ids"]
                )
                return

            try:
                focus = context.selection.get_focus()
            except Exception:
                focus = None

            if focus is not None:
                row_id = focus.row_id if hasattr(focus, "row_id") else focus.get("row_id")
                if row_id is not None:
                    request.row_ids = [row_id]

    def _action_job_key(self, reg: ActionRegistration, req: ActionRequest) -> str:
        if reg.key_fn is not None:
            try:
                return str(self._call_with_supported_args(reg.key_fn, request=req))
            except TypeError:
                return str(reg.key_fn(req))

        payload = {
            "action_id": reg.id,
            "dataset_id": req.dataset_id,
            "row_ids": list(req.row_ids or []),
            "columns": list(req.columns or []),
            "params": _json_safe(req.params),
            "artifact_id": req.artifact_id,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return f"plugin-action:{reg.id}:{digest}"

    def _remove_workspace_panels_for_plugin(self, plugin_id: str, context: Any) -> None:
        panel_ids = list(self._workspace_panels_by_plugin.get(plugin_id, set()))
        for panel_id in panel_ids:
            remove = getattr(context.workspace, "remove_panel", None)
            if callable(remove):
                try:
                    remove(panel_id)
                except Exception as exc:
                    logger.warning("Failed to remove plugin panel %s: %s", panel_id, exc)
        self._workspace_panels_by_plugin.pop(plugin_id, None)

    def _cancel_jobs_for_plugin(self, plugin_id: str) -> None:
        handles = self._job_handles_by_plugin.pop(plugin_id, [])
        for handle in handles:
            cancel = getattr(handle, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception as exc:
                    logger.warning("Failed to cancel plugin job for %s: %s", plugin_id, exc)

    def _remove_job_handle(self, plugin_id: str, handle: Any) -> None:
        handles = self._job_handles_by_plugin.get(plugin_id)
        if not handles:
            return
        try:
            handles.remove(handle)
        except ValueError:
            pass
        if not handles:
            self._job_handles_by_plugin.pop(plugin_id, None)

    def _require_record(self, plugin_id: str) -> PluginRecord:
        if plugin_id not in self._records:
            raise KeyError(f"Unknown plugin_id: {plugin_id}")
        return self._records[plugin_id]

    def _normalise_panel_result(self, result: Any) -> Tuple[Any, Any]:
        if isinstance(result, tuple) and len(result) == 2:
            view, controller = result
        else:
            controller = result
            if hasattr(result, "panel") and callable(result.panel):
                view = result.panel()
            else:
                view = result

        try:
            setattr(view, "_al_controller", controller)
        except Exception:
            pass
        return view, controller

    def _call_with_supported_args(self, func: Callable[..., Any], **kwargs: Any) -> Any:
        """Call func with only the keyword arguments it accepts.

        This keeps plugin author ergonomics high: factories can be written as
        ``factory(context)`` or ``factory(context, manager=...)`` without ceremony.
        """

        sig = inspect.signature(func)

        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return func(**kwargs)

        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}

        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect.Parameter.empty
        ]

        if len(positional) == 1 and not supported:
            name = positional[0].name
            if name in kwargs:
                return func(kwargs[name])

        return func(**supported)

    @staticmethod
    def _format_exception(exc: BaseException) -> str:
        return "".join(traceback.format_exception_only(type(exc), exc)).strip()


class _LazyServiceProxy:
    """Compatibility fallback for older ServiceRegistry implementations.

    Prefer adding set_factory/remove to ServiceRegistry. This proxy exists only to
    avoid crashing if the manager is used before that patch lands.
    """

    def __init__(self, key: str, factory: Callable[[], Any]) -> None:
        self._key = key
        self._factory = factory
        self._value: Any = None
        self._loaded = False

    def _get(self) -> Any:
        if not self._loaded:
            self._value = self._factory()
            self._loaded = True
        return self._value

    def __getattr__(self, item: str) -> Any:
        return getattr(self._get(), item)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _is_numeric_dtype(dtype: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.api.types.is_numeric_dtype(dtype))
    except Exception:
        text = str(dtype).lower()
        return any(token in text for token in ("int", "float", "double", "decimal", "number"))