from __future__ import annotations

from pathlib import Path
import os
import sys
import traceback

import holoviews as hv
import panel as pn

# ---------------------------------------------------------------------
# Panel / HoloViews setup
# ---------------------------------------------------------------------

pn.extension("tabulator")

pn.extension(
    raw_css=[
        r"""
        .al-cm { font-size: 12px; }
        .al-cm h4 { margin: 6px 0 4px 0; }
        .al-cm .metrics { margin: 0 0 8px 0; color: #333; }
        .al-cm table {
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;
        }
        .al-cm th, .al-cm td {
            border: 1px solid rgba(0,0,0,0.15);
            padding: 6px 8px;
            text-align: center;
            vertical-align: middle;
        }
        .al-cm th {
            background: rgba(0,0,0,0.04);
            font-weight: 600;
        }
        .al-cm .rowhdr {
            background: rgba(0,0,0,0.02);
            font-weight: 600;
            text-align: left;
        }
        .al-cm .diag { background: rgba(60, 180, 75, 0.10); }
        .al-cm .offd { background: rgba(230, 25, 75, 0.06); }

        .al-labels .bk-btn-group .bk-btn {
            padding: 4px 10px;
            font-size: 12px;
            line-height: 1.2;
        }

        .al-footer { padding-top: 4px; }
        .al-status { font-size: 12px; }

        .al-middle-grow {
            flex: 1 1 auto !important;
            min-height: 0 !important;
        }

        .al-middle-grow > .bk {
            height: 100% !important;
        }

        .al-root-col {
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            height: 100% !important;
            min-height: 0 !important;
        }

        .bk-modal .bk.modal-body {
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            min-height: 0 !important;
        }
        """
    ]
)

hv.extension("bokeh")
hv.renderer("bokeh").webgl = True
pn.config.sizing_mode = "stretch_both"

# Keep existing relative import behaviour.
sys.path.insert(1, os.path.join(sys.path[0], "../"))

# ---------------------------------------------------------------------
# AstronomicAL imports
# ---------------------------------------------------------------------

from astronomicAL.utils import load_config
from astronomicAL.utils.debug import boot_print, plugin_debug_print

from astronomicAL.platform.context import AppContext
from astronomicAL.platform.events import EventBus
from astronomicAL.platform.jobs import JobManager
from astronomicAL.platform.artifacts import ArtifactStore
from astronomicAL.platform.datasets import DatasetManager
from astronomicAL.platform.workspace import WorkspaceManager
from astronomicAL.platform.selection import SelectionManager
from astronomicAL.platform.record_navigation import RecordNavigationManager
from astronomicAL.platform.services import ServiceRegistry
from astronomicAL.platform.plugins import (
    InstalledPluginStore,
    PluginActivationService,
    PluginInstaller,
    PluginManager,
    PluginOrigin,
    PluginSearchPath,
    PluginStateStore,
)
from astronomicAL.platform.persistence import WorkspacePersistence
from astronomicAL.platform.runtime_status import RuntimeStatus

def _plugin_dirs() -> list[PluginSearchPath]:
    """Return local plugin roots together with host-owned activation policy."""

    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parent

    sources = [
        PluginSearchPath(
            package_dir / "plugins",
            origin=PluginOrigin.BUNDLED,
        ),
        PluginSearchPath(
            project_dir / "plugins",
            origin=PluginOrigin.DEVELOPMENT,
        ),
        PluginSearchPath(
            Path.cwd() / "plugins",
            origin=PluginOrigin.DEVELOPMENT,
        ),
        PluginSearchPath(
            Path.home() / ".astronomical" / "plugins",
            origin=PluginOrigin.USER,
            require_static_manifest=True,
        ),
    ]

    extra = os.environ.get("ASTRONOMICAL_PLUGIN_PATH", "")
    for item in extra.split(os.pathsep):
        if item.strip():
            sources.append(
                PluginSearchPath(
                    Path(item).expanduser(),
                    origin=PluginOrigin.DEVELOPMENT,
                )
            )

    # Preserve order while removing duplicate roots. The first declaration owns
    # the origin/policy for a path, so bundled roots cannot be downgraded later.
    seen: set[Path] = set()
    unique: list[PluginSearchPath] = []
    for source in sources:
        path = source.path.expanduser()
        if path in seen:
            continue
        seen.add(path)
        unique.append(source)

    return unique

def _discover_and_enable_plugins(context: AppContext) -> None:
    boot_print("main.py: plugin discovery start")
    manager = context.plugins
    activation = context.plugin_activation

    if manager is None:
        print("[plugins] no PluginManager on context")
        return
    if activation is None:
        raise RuntimeError("AppContext is missing plugin activation policy.")

    plugin_debug_print("local_plugin_sources:")
    for source in manager.local_plugin_sources:
        plugin_debug_print(
            "  - "
            f"{source.path} "
            f"origin={source.origin.value} "
            f"static_manifest_required={source.require_static_manifest} "
            f"exists={source.path.exists()}"
        )

    try:
        discovered = manager.discover()
        boot_print(f"main.py: plugin discovery complete count={len(discovered)}")
        for info in discovered:
            boot_print(
                "main.py: discovered plugin "
                f"id={info.id} "
                f"status={info.status} "
                f"source={info.source} "
                f"origin={info.origin.value} "
                f"path={info.path}"
            )
    except Exception as exc:
        print("[plugins] discovery failed:", exc)
        traceback.print_exc()
        return

    plugin_debug_print("discovered plugin infos:")
    for info in discovered:
        plugin_debug_print(
            "  - "
            f"{info.id} "
            f"status={info.status} "
            f"source={info.source} "
            f"origin={info.origin.value} "
            f"path={info.path}"
        )

    errors = manager.list_discovery_errors()
    if errors:
        print("[plugins] discovery errors:")
        for key, error in errors.items():
            print(f"  - {key}: {error}")

    for info in discovered:
        if not activation.should_enable(info):
            boot_print(
                "main.py: leaving plugin disabled "
                f"id={info.id} "
                f"origin={info.origin.value} "
                f"reason={activation.blocked_reason(info)}"
            )
            continue

        try:
            boot_print(f"main.py: enabling plugin id={info.id}")
            manager.enable(info.id, context=context)
            enabled_info = manager.plugin_info(info.id)
            boot_print(
                f"main.py: enabled plugin id={enabled_info.id} "
                f"panels={enabled_info.panels} "
                f"actions={enabled_info.actions} "
                f"services={enabled_info.services} "
                f"workflows={enabled_info.workflows}"
            )
        except Exception as exc:
            print(f"[plugins] failed to enable {info.id}: {exc}")
            traceback.print_exc()

# ---------------------------------------------------------------------
# Template and platform service construction
# ---------------------------------------------------------------------

react = pn.template.ReactTemplate(
    title="AstronomicAL",
    compact="vertical",
    prevent_collision=False,
)

# Create empty layout/grid first.
react, grid = load_config.create_layout_skeleton(react, return_grid=True)

runtime_status = RuntimeStatus(history_limit=1000)

events = EventBus(
    trace=True,
    trace_limit=1000,
    diagnostics_limit=1000,
)
jobs = JobManager(
    max_workers=16,
    history_limit=1000,
)
artifacts = ArtifactStore(cache_dir="data/cache_artifacts")
datasets = DatasetManager(events=events)
workspace = WorkspaceManager(react_template=react, grid=grid)
selection = SelectionManager(events=events, artifacts=artifacts)
navigation = RecordNavigationManager(datasets=datasets, selection=selection, events=events)
services = ServiceRegistry()
astronomical_home = Path.home() / ".astronomical"
plugin_state = PluginStateStore(
    astronomical_home / "plugin-state.json"
)
installed_plugins = InstalledPluginStore(
    astronomical_home / "installed-plugins.json"
)
plugin_activation = PluginActivationService(plugin_state)

# Make the status service available both directly on context and through the
# generic service registry for plugins that want to read diagnostics.
services.set(
    "platform.runtime_status",
    runtime_status,
    owner="platform",
)
services.set(
    "platform.record_navigation",
    navigation,
    owner="platform",
)

services.set(
    "platform.plugin_state",
    plugin_state,
    owner="platform",
)
services.set(
    "platform.plugin_activation",
    plugin_activation,
    owner="platform",
)
services.set(
    "platform.installed_plugins",
    installed_plugins,
    owner="platform",
)

boot_print("main.py: platform services created")
boot_print(f"main.py: events={type(events).__name__}")
boot_print(f"main.py: jobs={type(jobs).__name__}")
boot_print(f"main.py: artifacts={type(artifacts).__name__}")
boot_print(f"main.py: datasets={type(datasets).__name__}")
boot_print(f"main.py: workspace={type(workspace).__name__}")
boot_print(f"main.py: selection={type(selection).__name__}")
boot_print(f"main.py: services={type(services).__name__}")
boot_print(f"main.py: navigation={type(navigation).__name__}")
boot_print(f"main.py: plugin_state={type(plugin_state).__name__}")
boot_print(f"main.py: plugin_activation={type(plugin_activation).__name__}")
boot_print(f"main.py: installed_plugins={type(installed_plugins).__name__}")
boot_print(f"main.py: runtime_status={type(runtime_status).__name__}")

if plugin_state.load_error:
    print(
        "[plugins] plugin state could not be loaded; "
        "community plugins will remain disabled:",
        plugin_state.load_error,
    )

if installed_plugins.load_error:
    print(
        "[plugins] installed plugin database could not be loaded; "
        "install/update/uninstall operations will remain blocked:",
        installed_plugins.load_error,
    )

plugins = PluginManager(
    local_plugin_dirs=_plugin_dirs(),
    auto_discover=False,
)

plugin_installer = PluginInstaller(
    store=installed_plugins,
    plugin_dir=astronomical_home / "plugins",
    manager=plugins,
)
services.set(
    "platform.plugin_installer",
    plugin_installer,
    owner="platform",
)

boot_print("main.py: plugin manager created")
boot_print(f"main.py: plugin_installer={type(plugin_installer).__name__}")
boot_print("main.py: plugin local sources:")
for source in plugins.local_plugin_sources:
    boot_print(
        "  - "
        f"{source.path} "
        f"origin={source.origin.value} "
        f"static_manifest_required={source.require_static_manifest} "
        f"exists={source.path.exists()}"
    )

context = AppContext(
    events=events,
    jobs=jobs,
    artifacts=artifacts,
    datasets=datasets,
    workspace=workspace,
    selection=selection,
    services=services,
    navigation=navigation,
    layout_file=Path("astronomicAL/layout.json"),
    layout_directory=Path("layouts"),
    plugins=plugins,
    plugin_state=plugin_state,
    plugin_activation=plugin_activation,
    installed_plugins=installed_plugins,
    plugin_installer=plugin_installer,
    runtime_status=runtime_status,
)

boot_print("main.py: AppContext created")
boot_print(f"main.py: context.plugins={type(context.plugins).__name__}")

context.persistence = WorkspacePersistence(context)

boot_print(f"main.py: context.persistence={type(context.persistence).__name__}")

react._app_context = context

required = [
    "events",
    "jobs",
    "artifacts",
    "datasets",
    "workspace",
    "selection",
    "navigation",
    "services",
    "plugins",
    "plugin_state",
    "plugin_activation",
    "installed_plugins",
    "plugin_installer",
    "persistence",
    "runtime_status",
    "layout_file",
    "layout_directory",
]

missing = [name for name in required if getattr(context, name, None) is None]
if missing:
    raise RuntimeError(f"AppContext missing services: {missing}")

# IMPORTANT:
# Enable plugins before restoring/defaulting the layout, because Dashboard and
# get_customplot_dict need to see plugin panel registrations.
_discover_and_enable_plugins(context)

# ---------------------------------------------------------------------
# Layout creation
# ---------------------------------------------------------------------

boot_print("main.py: layout creation start")

boot_print(f"main.py: layout_directory={context.layout_directory}")

boot_print(
    "main.py: plugin panels before layout="
    f"{[p.id for p in plugins.list_panels()]}"
)

boot_print("main.py: calling create_default_layout")

load_config.create_default_layout(react, context)

boot_print("main.py: returned from create_default_layout")

boot_print("main.py: workspace.register_existing start")
workspace.register_existing()
boot_print("main.py: workspace.register_existing complete")

boot_print("main.py: react.servable")
react.servable()