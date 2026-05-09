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

import astronomicAL.config as config
from astronomicAL.utils import load_config
from astronomicAL.utils.debug import boot_print, plugin_debug_print

from astronomicAL.platform.context import AppContext
from astronomicAL.platform.events import EventBus
from astronomicAL.platform.jobs import JobManager
from astronomicAL.platform.artifacts import ArtifactStore
from astronomicAL.platform.datasets import DatasetManager
from astronomicAL.platform.workspace import WorkspaceManager
from astronomicAL.platform.selection import SelectionManager
from astronomicAL.platform.services import ServiceRegistry
from astronomicAL.platform.plugins import PluginManager
from astronomicAL.platform.persistence import WorkspacePersistence

def _plugin_dirs() -> list[Path]:
    """Return plugin roots scanned by PluginManager.

    PluginManager expects each root directory to contain plugin folders, where
    each plugin folder contains a plugin.py file, for example:

        astronomicAL/plugins/event_monitor/plugin.py
        astronomicAL/plugins/table_tools/plugin.py

    A plain .py file directly inside one of these roots is also supported by the
    supplied PluginManager.
    """

    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parent

    paths = [
        package_dir / "plugins",
        project_dir / "plugins",
        Path.cwd() / "plugins",
        Path.home() / ".astronomical" / "plugins",
    ]

    extra = os.environ.get("ASTRONOMICAL_PLUGIN_PATH", "")
    for item in extra.split(os.pathsep):
        if item.strip():
            paths.append(Path(item).expanduser())

    # Preserve order while removing duplicates.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique

def _discover_and_enable_plugins(context: AppContext) -> None:
    boot_print("main.py: plugin discovery start")
    manager = context.plugins
    if manager is None:
        print("[plugins] no PluginManager on context")
        return

    plugin_debug_print("local_plugin_dirs:")
    for path in manager.local_plugin_dirs:
        plugin_debug_print(f"  - {path} exists={path.exists()}")

    try:
        discovered = manager.discover()
        boot_print(f"main.py: plugin discovery complete count={len(discovered)}")
        for info in discovered:
            boot_print(
                "main.py: discovered plugin "
                f"id={info.id} status={info.status} source={info.source} path={info.path}"
            )
    except Exception as exc:
        print("[plugins] discovery failed:", exc)
        traceback.print_exc()
        return

    plugin_debug_print("discovered plugin infos:")
    for info in discovered:
        plugin_debug_print(
            f"  - {info.id} status={info.status} source={info.source} path={info.path}"
        )

    errors = manager.list_discovery_errors()
    if errors:
        print("[plugins] discovery errors:")
        for key, error in errors.items():
            print(f"  - {key}: {error}")

    for info in discovered:
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

events = EventBus(trace=True, trace_limit=1000)
jobs = JobManager(max_workers=16)
artifacts = ArtifactStore(cache_dir="data/cache_artifacts")
datasets = DatasetManager()
workspace = WorkspaceManager(react_template=react, grid=grid)
selection = SelectionManager(events=events, artifacts=artifacts)
services = ServiceRegistry()

boot_print("main.py: platform services created")
boot_print(f"main.py: events={type(events).__name__}")
boot_print(f"main.py: jobs={type(jobs).__name__}")
boot_print(f"main.py: artifacts={type(artifacts).__name__}")
boot_print(f"main.py: datasets={type(datasets).__name__}")
boot_print(f"main.py: workspace={type(workspace).__name__}")
boot_print(f"main.py: selection={type(selection).__name__}")
boot_print(f"main.py: services={type(services).__name__}")

plugins = PluginManager(
    local_plugin_dirs=_plugin_dirs(),
    auto_discover=False,
)

boot_print("main.py: plugin manager created")
boot_print("main.py: plugin local dirs:")
for path in plugins.local_plugin_dirs:
    boot_print(f"  - {path} exists={path.exists()}")

context = AppContext(
    events=events,
    jobs=jobs,
    artifacts=artifacts,
    datasets=datasets,
    workspace=workspace,
    selection=selection,
    services=services,
    config=config,
    plugins=plugins,
)

boot_print("main.py: AppContext created")
boot_print(f"main.py: context.plugins={type(context.plugins).__name__}")
boot_print(f"main.py: context.config={type(context.config).__name__}")

context.persistence = WorkspacePersistence(context)

boot_print(f"main.py: context.persistence={type(context.persistence).__name__}")


context.config.layout_file = getattr(
    context.config,
    "layout_file",
    "astronomicAL/layout.json",
)

# Optional compatibility handles for older code that reaches into config.
context.config.plugins = plugins
context.config.app_context = context

react._app_context = context

required = [
    "config",
    "events",
    "jobs",
    "artifacts",
    "datasets",
    "workspace",
    "selection",
    "services",
    "plugins",
    "persistence",
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
boot_print(f"main.py: layout_file={config.layout_file}")
boot_print(f"main.py: layout_file_exists={os.path.isfile(config.layout_file)}")
boot_print(
    "main.py: plugin panels before layout="
    f"{[p.id for p in plugins.list_panels()]}"
)

if os.path.isfile(config.layout_file):
    boot_print("main.py: calling create_layout_from_file")
    load_config.create_layout_from_file(react, context)
    boot_print("main.py: returned from create_layout_from_file")
else:
    boot_print("main.py: calling create_default_layout")
    load_config.create_default_layout(react, context)
    boot_print("main.py: returned from create_default_layout")

boot_print("main.py: workspace.register_existing start")
workspace.register_existing()
boot_print("main.py: workspace.register_existing complete")

boot_print("main.py: react.servable")
react.servable()