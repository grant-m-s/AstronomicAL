import panel as pn
pn.extension("tabulator")

print("pn.config.raw_css type:", type(pn.config.raw_css))
print("pn.state._extensions:", getattr(pn.state, "_extensions", None))
print("pn.state._loaded_extensions:", getattr(pn.state, "_loaded_extensions", None))

pn.extension(raw_css=[r"""
.al-cm { font-size: 12px; }
.al-cm h4 { margin: 6px 0 4px 0; }
.al-cm .metrics { margin: 0 0 8px 0; color: #333; }

.al-cm table { border-collapse: collapse; width: 100%; table-layout: fixed; }
.al-cm th, .al-cm td {
  border: 1px solid rgba(0,0,0,0.15);
  padding: 6px 8px;
  text-align: center;
  vertical-align: middle;
}
.al-cm th { background: rgba(0,0,0,0.04); font-weight: 600; }
.al-cm .rowhdr { background: rgba(0,0,0,0.02); font-weight: 600; text-align: left; }

.al-cm .diag { background: rgba(60, 180, 75, 0.10); }   /* optional: diagonal highlight */
.al-cm .offd { background: rgba(230, 25, 75, 0.06); }  /* optional: off-diagonal highlight */
                      
/* Compact the label selector buttons */
.al-labels .bk-btn-group .bk-btn {
  padding: 4px 10px;
  font-size: 12px;
  line-height: 1.2;
}
.al-footer { padding-top: 4px; }
.al-status { font-size: 12px; }
                      
.al-middle-grow {
  flex: 1 1 auto !important;   /* grow to fill vertical space */
  min-height: 0 !important;    /* allow children to shrink/scroll */
}
.al-middle-grow > .bk { 
  height: 100% !important;
}
                      

.al-root-col {
  display: flex !important;
  flex-direction: column !important;
  justify-content: flex-start !important; /* stop space-between behavior */
  height: 100% !important;
  min-height: 0 !important;
}
                      
.bk-modal .bk.modal-body {
  display: flex !important;
  flex-direction: column !important;
  justify-content: flex-start !important;
  min-height: 0 !important;
}
"""])

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../"))
from astronomicAL.utils import load_config
import astronomicAL.config as config

from astronomicAL.platform.context import AppContext
from astronomicAL.platform.events import EventBus
from astronomicAL.platform.jobs import JobManager
from astronomicAL.platform.artifacts import ArtifactStore
from astronomicAL.platform.datasets import DatasetManager
from astronomicAL.platform.workspace import WorkspaceManager
from astronomicAL.platform.selection import SelectionManager
from astronomicAL.platform.services import ServiceRegistry

import holoviews as hv
hv.extension("bokeh")
hv.renderer("bokeh").webgl = True

pn.config.sizing_mode = "stretch_both"

react = pn.template.ReactTemplate(title="AstronomicAL", compact="vertical", prevent_collision=False)#

# Create empty layout/grid first
react, grid = load_config.create_layout_skeleton(react, return_grid=True)

# Construct platform services
events = EventBus(trace=True, trace_limit=1000)
jobs = JobManager(max_workers=16)
artifacts = ArtifactStore(cache_dir="data/cache_artifacts")
datasets = DatasetManager()
workspace = WorkspaceManager(react_template=react, grid=grid)
selection = SelectionManager(events=events, artifacts=artifacts)
services = ServiceRegistry()

context = AppContext(
    events=events,
    jobs=jobs,
    artifacts=artifacts,
    datasets=datasets,
    workspace=workspace,
    selection=selection,
    services=services,
    config=config,
)

context.config.layout_file = getattr(context.config, "layout_file", "astronomicAL/layout.json")

react._app_context = context

required = ["config", "events", "jobs", "artifacts", "datasets", "workspace"]
missing = [name for name in required if getattr(context, name, None) is None]
if missing:
    raise RuntimeError(f"AppContext missing services: {missing}")

# Now finalize layout using context (header + dashboards + restore layouts)
if os.path.isfile(config.layout_file):
    print("Has layout File")
    load_config.create_layout_from_file(react, context)
else:
    print("Create Default")
    load_config.create_default_layout(react, context)

workspace.register_existing()

react.servable()
