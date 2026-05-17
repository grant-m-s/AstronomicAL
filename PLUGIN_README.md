# AstronomicAL Plugin System

AstronomicAL is moving toward a small generic platform plus composable plugins.

Plugins are the preferred way to add panels, actions, services, artifact viewers, workflows, and domain-specific functionality without adding more assumptions to the core application.

The goal is to keep the platform small and reusable while allowing domain-specific, workflow-specific, and user-specific functionality to live in separate modules.

In this model:

```text
AstronomicAL platform
    Generic runtime services and plugin host.

Core plugins
    Plugins shipped with the main AstronomicAL repository.

Domain plugins
    Plugins for a specific scientific, industrial, or data domain.

Workflow plugins
    Plugins that provide reusable workflows such as active learning, review, triage, or quality control.

Plugin bundles
    Installable collections of plugins that can be added or enabled together.

User plugins
    Local project-specific plugins.

Third-party plugins
    Plugins distributed independently of the main repository.
```

The platform should stay domain-neutral. Domain assumptions should live in plugins.

---

## Current status

The plugin system is under active development on the `plugin_system` branch.

Several features that were previously part of the main application are now being moved into plugins:

* **Record Browser** replaces the generic browsing part of the old Exploration mode.
* **Visualisation** replaces the old BasicPlots-style plotting functionality.
* **Selection Tools** provides selection-set inspection as a plugin.
* **Table Tools** provides table transforms and subset creation as a plugin.
* **Annotations** provides record-level notes, review state, label suggestions, and annotation summaries as a plugin.
* **Event Monitor** provides platform and event diagnostics as a plugin.
* **Plugin Manager** provides plugin inspection and management as a plugin.
* Dataset loading and switching now live in a platform-level dataset header.
* Column mapping is platform-managed and is automatically applied to plugins that declare required or optional semantic columns.
* Workspace persistence can save and restore plugin-based layouts.

Plugin authors should target the platform services and plugin API, not older dashboard modes.

---

## Terminology

### Platform

The platform is the generic host layer under:

```text
astronomicAL/platform/
```

It owns runtime services such as datasets, selection, events, artifacts, jobs, workspace, services, mapping, persistence, and plugin management.

The platform should not know what kind of records a user is analysing. A record might be a source, patient, transaction, sample, machine part, observation, product, document, or anything else represented in a dataset.

### Plugin framework

The plugin framework lives under:

```text
astronomicAL/platform/plugins/
```

This is infrastructure for discovering, validating, enabling, disabling, registering, and running plugins.

Do not put normal plugins in this directory.

### Core plugins

Core plugins are plugins that ship with the main AstronomicAL repository.

They live under:

```text
astronomicAL/plugins/
```

Core plugins are still real plugins. They use the same plugin API as external plugins, but they are maintained alongside the platform because they provide broadly useful functionality.

### Plugin bundles

A plugin bundle is a collection of plugins that can eventually be installed or enabled together.

### Domain plugins

Domain plugins contain field-specific assumptions.

Examples:

```text
astro.cutouts
astro.spectra
medical.scan_viewer
medical.patient_review
factory.sensor_review
factory.defect_images
ecology.audio_review
materials.microscopy
finance.transaction_review
```

Domain plugins should not be required for users who do not work in that domain.

### Workflow plugins

Workflow plugins assemble panels, actions, services, and artifacts into a reusable workflow.

Examples:

```text
active_learning.core
review.catalogue
quality_control.inspect
anomaly_triage.core
reporting.summary
```

### User plugins

User plugins are local plugins written for a specific project, dataset, team, or experiment.

---

## Directory structure

There are two different plugin-related directories in the codebase.

### Plugin framework directory

```text
astronomicAL/platform/plugins/
```

This directory contains the plugin framework itself:

```text
astronomicAL/platform/plugins/
├── __init__.py
├── api.py
├── errors.py
├── manager.py
├── manifest.py
├── mapping_gate.py
└── specs.py
```

This area contains:

* `PluginManager`
* `PluginAPI`
* `PluginManifest`
* plugin registration specs
* plugin errors
* mapping-gated panel support
* action registration
* panel registration
* service registration
* artifact-viewer registration
* workflow registration

Do not put normal plugins here.

### Core plugin directory

```text
astronomicAL/plugins/
```

This directory contains Core plugins that ship with AstronomicAL.

Current Core plugins include:

```text
astronomicAL/plugins/
├── annotations/
├── event_monitor/
├── plugin_manager/
├── record_browser/
├── selection_tools/
├── table_tools/
└── visualisation/
```

Each plugin should live in its own folder and expose a `plugin.py` module.

Example:

```text
astronomicAL/plugins/my_plugin/
└── plugin.py
```

### User and development plugin directories

Local user plugins can also be loaded from:

```text
~/.astronomical/plugins/
```

A development checkout may also scan:

```text
plugins/
```

at the repository root, depending on configured plugin paths.

Installed third-party packages can expose plugins through Python entry points.

---

## Platform services

Plugins receive a platform `context` object. This is the main host API.

```python
context.datasets
context.selection
context.events
context.artifacts
context.jobs
context.workspace
context.services
context.plugins
context.persistence
```

Use the named platform services instead of importing global runtime state.

`context.config` is transitional. New runtime state should not be added there.

`context.config` exists only as a migration bridge for older code. New plugins should avoid it unless they are reading legacy settings that have not yet been moved into platform services.

---

## Where state belongs

Use this guide when deciding where something should live.

| Need | Use |
| --- | --- |
| Source dataframe or loaded table | `context.datasets` |
| Active dataset id | `context.datasets` |
| Current focused row | `context.selection` |
| Active multi-row selection | `context.selection` |
| Derived output or cached computed result | `context.artifacts` |
| Notification that something changed | `context.events` |
| Slow or cancellable work | `context.jobs` |
| Visible panels and layout | `context.workspace` |
| Live API client, session, connector, or runtime backend | `context.services` |
| Plugin metadata, registrations, enable/disable state | `context.plugins` |
| Workspace save/load | `context.persistence` |
| Temporary compatibility with old application code | `context.config` |

Summary rule:

```text
Datasets store source data.
Selection stores live focus and selected row sets.
Artifacts store derived data.
Events announce changes.
Jobs run slow work.
Workspace owns visible panels.
Services hold live capabilities.
Plugins contribute behaviour.
Persistence saves and restores workspace state.
Config is only a temporary compatibility bridge.
```

---

## Core platform services

### `datasets`

`context.datasets` owns canonical source data.

Use it for:

* loaded dataframes
* active dataset tracking
* dataset metadata
* column lists
* semantic column mappings
* derived datasets that become first-class working tables

Do not use it for temporary computed outputs. Those usually belong in `artifacts`.

Example:

```python
context.datasets.register(
    "main_catalog",
    df,
    name="Main Catalog",
    source="data/catalog.parquet",
)

context.datasets.set_active("main_catalog")

active_df = context.datasets.get_df()
```

---

### `selection`

`context.selection` owns the current platform selection state.

It separates two related concepts:

* **focus**: one current row or record
* **selection set**: a multi-row selected subset

Use focus when a table, plot, browser, or workflow is pointing at one row.

```python
context.selection.set_focus(
    dataset_id=context.datasets.active_id(),
    row_id=row_id,
    origin="my_plugin.table",
)
```

Use a selection set when a lasso, filter, review queue, or batch operation selects multiple rows.

```python
context.selection.set_selection_set(
    dataset_id=context.datasets.active_id(),
    row_ids=row_ids,
    origin="my_plugin.lasso",
)
```

The selection service publishes canonical events such as:

```text
selection.focus.changed
selection.focus.cleared
selection.set.changed
selection.set.cleared
```

Plugins should use `selection` as the source of truth and use events only as notifications.

---

### `events`

`context.events` is the platform publish/subscribe system.

Use it to decouple plugins and panels.

A panel should not call another panel directly. It should update platform state or publish an event, and interested subscribers should react independently.

Example:

```python
sub = context.events.subscribe(
    "selection.focus.changed",
    on_focus_changed,
    owner_id="my_plugin.detail_panel",
    owner_label="Detail Panel",
    owner_kind="panel",
)
```

Always unsubscribe during cleanup:

```python
context.events.unsubscribe(sub)
```

Good event names use dotted topics:

```text
dataset.loaded
dataset.active.changed
dataset.updated
dataset.mapping_updated
mapping.requested
mapping.resolved
mapping.open_requested
selection.focus.changed
selection.focus.cleared
selection.set.changed
selection.set.cleared
artifact.created
plugin.enabled
plugin.disabled
workspace.saved
workflow.stage.changed
```

Events should be lightweight. Do not put large dataframes, images, spectra, model outputs, or reports directly on the event bus.

Bad:

```python
context.events.publish("scores.updated", huge_dataframe)
```

Better:

```python
artifact_id = context.artifacts.put(
    "classifier.scores",
    scores,
    dataset_id=context.datasets.active_id(),
)

context.events.publish(
    "artifact.created",
    {
        "artifact_id": artifact_id,
        "type": "classifier.scores",
    },
)
```

---

### `artifacts`

`context.artifacts` stores derived outputs.

Use artifacts for results that are computed from source data or fetched from external services and may be reused by other plugins.

Examples:

* classifier scores
* trained model metadata
* filtered table results
* embeddings
* reports
* remote API responses
* image cutouts
* spectra
* transformed coordinates
* selection materialisations

Example:

```python
artifact_id = context.artifacts.put(
    "table.filtered",
    filtered_df.to_dict(orient="records"),
    dataset_id=context.datasets.active_id(),
    row_ids=row_ids,
    params={"filter": "score > 0.9"},
)

context.events.publish(
    "artifact.created",
    {
        "artifact_id": artifact_id,
        "type": "table.filtered",
    },
)
```

Artifacts are for data products. Services are for live clients. Datasets are for source or promoted working tables.

---

### `jobs`

`context.jobs` runs slow or cancellable work without blocking the UI.

Use jobs for:

* remote API fetches
* expensive table operations
* model training
* model inference
* feature generation
* file export
* report generation
* long-running visualisation preparation

Example:

```python
def fetch_remote_data(*, cancel_token, row_id):
    if cancel_token and cancel_token.cancelled():
        return None

    client = context.services.require("my_plugin.client")
    result = client.fetch(row_id)

    if cancel_token and cancel_token.cancelled():
        return None

    return result


def handle_result(result):
    if result is None:
        return

    artifact_id = context.artifacts.put(
        "remote.payload",
        result,
        dataset_id=context.datasets.active_id(),
    )

    context.events.publish(
        "artifact.created",
        {
            "artifact_id": artifact_id,
            "type": "remote.payload",
        },
    )


handle = context.jobs.submit(
    fetch_remote_data,
    title="Fetch remote data",
    key=f"remote:{row_id}",
    on_done=handle_result,
    row_id=row_id,
)
```

Keep job handles if the panel should cancel them when closed.

---

### `workspace`

`context.workspace` owns visible panels and layout.

Use it to add or remove panels. Do not mutate the underlying template or grid directly from plugin code.

Example:

```python
context.workspace.add_panel(
    panel_id="my_plugin.summary",
    title="Summary",
    view=view,
    controller=controller,
)
```

A controller should implement `dispose()` when it owns subscriptions, jobs, periodic callbacks, or widget watchers.

```python
class SummaryController:
    def __init__(self, context):
        self.context = context
        self.subscriptions = []

    def dispose(self):
        for sub in self.subscriptions:
            self.context.events.unsubscribe(sub)
        self.subscriptions.clear()
```

Workspace persistence uses panel metadata and controller state to restore layouts where possible.

---

### `services`

`context.services` stores shared live runtime capabilities.

Use services for:

* API clients
* authenticated sessions
* database connections
* filesystem adapters
* external analysis backends
* shared plugin runtime capabilities

Example:

```python
def create_client(context, **kwargs):
    return MyClient(api_key="...")


api.register_service(
    key="client",
    factory=create_client,
    lazy=True,
    replace=True,
)
```

A plugin service key is automatically namespaced by the plugin id.

For example, if the plugin id is:

```text
example.remote
```

and the service key is:

```text
client
```

the canonical service key becomes:

```text
example.remote.client
```

Services are not for storing data products.

Bad:

```python
context.services.set("latest_scores", scores)
```

Better:

```python
context.artifacts.put(
    "classifier.scores",
    scores,
    dataset_id=context.datasets.active_id(),
)
```

---

### `plugins`

`context.plugins` gives access to the plugin manager.

Use it for:

* introspection
* opening plugin panels
* running plugin actions
* building plugin-driven workflows

Common manager responsibilities include:

* discovering plugins
* validating manifests and dependencies
* enabling plugins
* disabling plugins
* registering plugin contributions
* opening plugin panels
* running plugin actions
* installing plugin services
* listing panels, actions, workflows, services, and artifact viewers

Plugin code should usually interact through the `PluginAPI` during registration and through `context` at runtime. Direct use of manager internals should be avoided.

---

### `persistence`

`context.persistence` owns workspace save/load behaviour.

The persistence layer may record:

* schema name and version
* enabled plugin information
* dataset metadata and pending mappings
* selection state
* workspace panels
* panel layout
* panel restore metadata
* panel state where controllers expose it

Plugin authors do not need to do anything for basic layout restore.

To preserve internal widget or controller state, implement `get_state()` and restore support in the panel or controller.

Panel state must be JSON-safe.

Do not store these in panel state:

* dataframes
* API clients
* file handles
* Bokeh documents
* jobs
* threads
* model objects
* large binary payloads

Use the platform services instead:

```text
dataframes              -> context.datasets
derived outputs         -> context.artifacts
API clients/sessions    -> context.services
slow work               -> context.jobs
current focus/subsets   -> context.selection
```

---

## Minimal plugin

A minimal plugin is a folder containing `plugin.py`.

```text
astronomicAL/plugins/example_plugin/
└── plugin.py
```

The plugin module should expose:

1. a module-level `manifest`
2. a `register(api)` function

Example:

```python
from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="example.hello",
    name="Hello Plugin",
    version="0.1.0",
    description="A minimal example plugin.",
    capabilities=["panel"],
    tags=["example"],
)


def register(api):
    api.register_panel(
        id="panel",
        title="Hello Panel",
        factory=create_panel,
        description="A minimal plugin panel.",
        category="Examples",
    )


def create_panel(context, **kwargs):
    import panel as pn

    view = pn.pane.Markdown("## Hello from a plugin")
    controller = None
    return view, controller
```

The registered panel id becomes:

```text
example.hello.panel
```

Registration ids are automatically namespaced by the plugin id.

---

## Plugin manifests

Every plugin should define a `PluginManifest`.

Example:

```python
from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="example.remote",
    name="Remote Data Tools",
    version="0.1.0",
    description="Example panels and services for a remote data source.",
    author="Your Name",
    homepage="https://example.org",
    package="astronomicAL.plugins.example_remote",
    min_astronomical="0.1.0",
    max_astronomical=None,
    capabilities=["panel", "service", "artifacts"],
    tags=["example", "remote"],
    requires=[
        "requests",
    ],
    optional_requires=[],
    requires_plugins=[],
    optional_plugins=[],
    metadata={},
)
```

`requires` and `optional_requires` are Python package dependencies.

Examples:

```text
requests
astropy>=6
pydicom
```

`requires_plugins` and `optional_plugins` are AstronomicAL plugin dependencies.

Examples:

```text
core.record_browser
core.visualisation
astro.spectra
```

Keep manifests cheap to import.

Do not import heavy optional libraries just to create a manifest. Heavy imports should happen inside factories, handlers, service constructors, or plugin enable hooks.

Good:

```python
from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="domain.image_tools",
    name="Domain Image Tools",
    version="0.1.0",
    requires=["pillow"],
)


def register(api):
    api.register_panel(
        id="viewer",
        title="Image Viewer",
        factory=create_image_viewer,
    )


def create_image_viewer(context, **kwargs):
    from PIL import Image

    return build_viewer(context, image_module=Image)
```

Avoid:

```python
from PIL import Image
from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="domain.image_tools",
    name="Domain Image Tools",
    version="0.1.0",
)
```

This matters because plugins may be discovered even when they are not enabled.

---

## What plugins can contribute

A plugin can contribute:

* panels
* actions
* dataframe actions
* services
* artifact viewers
* workflows
* settings schemas

---

## Panels

Panels are visible UI components that can be added to the workspace.

Example:

```python
def register(api):
    api.register_panel(
        id="summary",
        title="Summary Panel",
        factory=create_summary_panel,
        description="Display a summary of the active dataset.",
        category="Data",
        icon="table",
        tags=["dataset", "summary"],
        default_layout={"x": 0, "y": 0, "w": 4, "h": 4},
        default_open_kwargs={},
        state_version=1,
        persist_layout=True,
        persist_state=True,
    )


def create_summary_panel(context, **kwargs):
    controller = SummaryPanel(context, **kwargs)
    return controller.panel(), controller
```

Panel factories should usually return:

```python
return view, controller
```

where:

* `view` is a Panel object or compatible view object
* `controller` is optional, but should implement `dispose()` if it owns runtime resources

A controller should clean up:

* event subscriptions
* jobs
* widget watchers
* periodic callbacks
* scheduled callbacks
* child controllers or child panels
* remote connections it owns
* temporary resources it created

Panel factories should accept `**kwargs` so restore metadata can be passed through safely.

---

## Mapping-aware panels

Plugins should not hard-code column names when they need semantic fields.

Instead, declare required and optional mappings.

Example:

```python
def register(api):
    api.register_panel(
        id="scatter",
        title="Scatter Plot",
        factory=create_scatter_panel,
        required_mappings=["record_id"],
        optional_mappings=["target_label"],
    )
```

If a required mapping is missing, the platform opens a mapping-gated placeholder instead of constructing the real panel. The user can resolve mappings through the column-mapping header. Once the required mappings exist, the real panel is constructed.

Use:

```text
required_mappings
    Semantic columns the panel cannot function without.

optional_mappings
    Semantic columns that enable extra behaviour if available.
```

Common semantic mappings include:

```text
record_id
target_label
```

Domain plugins may define their own semantic mappings.

Examples:

```text
coords.ra
coords.dec
patient.id
sample.id
transaction.id
machine.serial_number
observation.site
document.id
```

A string mapping is enough for common cases.

For richer mapping requests, use dictionaries:

```python
api.register_panel(
    id="domain_view",
    title="Domain View",
    factory=create_domain_view,
    required_mappings=[
        {
            "semantic_name": "domain.primary_id",
            "display_name": "Primary domain identifier",
            "description": "Unique identifier used by this domain plugin.",
            "aliases": ["id", "object_id", "record_id", "source_id"],
        },
        {
            "semantic_name": "domain.measurement",
            "display_name": "Primary measurement",
            "description": "Measurement used by this panel.",
            "aliases": ["measurement", "value", "score"],
        },
    ],
)
```

Use semantic mappings for domain-specific requirements too.

Example:

```python
api.register_panel(
    id="patient_summary",
    title="Patient Summary",
    factory=create_patient_summary,
    required_mappings=[
        {
            "semantic_name": "patient.id",
            "display_name": "Patient ID",
            "description": "Unique patient identifier.",
            "aliases": ["patient_id", "patient", "subject_id"],
        },
    ],
)
```

Example:

```python
api.register_panel(
    id="sky_view",
    title="Sky View",
    factory=create_sky_view,
    required_mappings=[
        {
            "semantic_name": "coords.ra",
            "display_name": "RA column",
            "description": "Right ascension in degrees.",
            "aliases": ["ra", "RA", "ra_deg", "raj2000"],
        },
        {
            "semantic_name": "coords.dec",
            "display_name": "DEC column",
            "description": "Declination in degrees.",
            "aliases": ["dec", "DEC", "dec_deg", "dej2000"],
        },
    ],
)
```

The platform should know that a mapping exists, but it does not need to understand the domain meaning.

---

## Panel creation and loading

Panel factories should return quickly.

If a panel needs remote data, large preprocessing, model inference, expensive plotting, or heavy domain setup, it should create a lightweight initial view and move slow work into `context.jobs`.

The platform may show a temporary loading panel while a heavier panel is being constructed. If panel creation fails, the platform should show an error panel rather than crashing the workspace.

Do not fetch remote data, train models, or build expensive payloads directly inside the panel factory.

Use the factory to build the controller/view, then use `context.jobs` for slow work.

---

## Actions

Actions are callable operations registered by plugins.

They can be used by panels, workflows, buttons, menus, or future automation layers.

Example:

```python
from astronomicAL.platform.plugins.specs import ArtifactResult, InputSpec

def register(api):
    api.register_action(
        id="profile_numeric",
        title="Profile numeric columns",
        handler=profile_numeric,
        inputs=InputSpec(
            dataset=True,
            selection="optional",
            numeric_columns="many",
            columns="many",
        ),
        outputs=["table.numeric_profile"],
        run_in_job=True,
    )


def profile_numeric(context, request, cancel_token=None):
    df = context.datasets.get_df(request.dataset_id)

    if request.columns:
        df = df[request.columns]

    result = df.describe().reset_index().to_dict(orient="records")

    return ArtifactResult(
        type="table.numeric_profile",
        payload=result,
        dataset_id=request.dataset_id,
        row_ids=request.row_ids,
        params=request.params,
    )
```

Action handlers may receive some or all of:

```text
context
request
manager
cancel_token
```

The platform calls handlers with the arguments they support.

---

## Dataframe actions

For common dataframe-oriented operations, use `register_dataframe_action`.

Example:

```python
def register(api):
    api.register_dataframe_action(
        id="count_rows",
        title="Count rows",
        handler=count_rows,
        output_type="table.row_count",
        selection="optional",
        columns="none",
        numeric_columns="none",
        run_in_job=False,
    )


def count_rows(df, dataset_id=None, row_ids=None, params=None):
    return {
        "dataset_id": dataset_id,
        "row_count": len(df),
        "selection_count": len(row_ids or []),
    }
```

The platform resolves:

* active dataframe
* dataset id
* selected rows
* selected columns
* numeric columns
* action params
* cancellation token

and passes only the arguments the handler accepts.

---

## Action inputs and outputs

Action inputs are described with `InputSpec`.

Useful fields include:

```python
InputSpec(
    dataset=True,
    selection="optional",
    numeric_columns="many",
    columns="optional",
    required_mappings=["record_id"],
    optional_mappings=["target_label"],
    accepts_artifact_types=["table.filtered"],
)
```

Selection values should be one of:

```text
none
optional
required
```

Column and numeric-column values should be one of:

```text
none
optional
one
many
```

Actions can return structured results.

### Artifact result

```python
from astronomicAL.platform.plugins.specs import ArtifactResult

return ArtifactResult(
    type="classifier.scores",
    payload=scores,
    dataset_id=dataset_id,
    row_ids=row_ids,
    params={"model": "rf"},
    publish=True,
)
```

### Dataset result

```python
from astronomicAL.platform.plugins.specs import DatasetResult

return DatasetResult(
    id="high_score_subset",
    dataframe=filtered_df,
    name="High Score Subset",
    metadata={"derived_from": dataset_id},
    set_active=True,
)
```

### Event result

```python
from astronomicAL.platform.plugins.specs import EventResult

return EventResult(
    topic="review.status.changed",
    payload={
        "dataset_id": dataset_id,
        "row_id": row_id,
        "status": "approved",
    },
)
```

### Combined action result

```python
from astronomicAL.platform.plugins.specs import (
    ActionResult,
    ArtifactResult,
    DatasetResult,
    EventResult,
)

return ActionResult(
    value={"status": "complete"},
    artifacts=[
        ArtifactResult(
            type="report.summary",
            payload=summary,
            dataset_id=dataset_id,
        )
    ],
    datasets=[
        DatasetResult(
            id="review_queue",
            dataframe=review_df,
            name="Review Queue",
            set_active=False,
        )
    ],
    events=[
        EventResult(
            topic="workflow.stage.changed",
            payload={"stage": "review_ready"},
        )
    ],
)
```

When actions return structured results, the platform can store artifacts, register datasets, publish events, and return processed identifiers to the caller.

---

## Services

Services are shared live runtime objects.

Use services for clients, connectors, sessions, and backends.

Example:

```python
def register(api):
    api.register_service(
        key="client",
        factory=create_client,
        lazy=True,
        replace=True,
        description="Shared remote API client.",
    )


def create_client(context, **kwargs):
    return RemoteClient()
```

Access the service from another panel or action:

```python
client = context.services.require("example.remote.client")
result = client.fetch("123")
```

Prefer namespaced service keys for cross-plugin access.

Example service keys:

```text
example.remote.client
database.primary
storage.cache
medical.fhir.client
factory.sensor_store
astro.archive.client
```

---

## Artifact viewers

Artifact viewers display derived results created by panels or actions.

Example:

```python
def register(api):
    api.register_artifact_viewer(
        artifact_type="table.filtered",
        viewer_factory=create_table_viewer,
        title="Filtered Table Viewer",
        default=True,
    )


def create_table_viewer(context, artifact_id=None, **kwargs):
    payload = context.artifacts.get(artifact_id)
    return build_table_view(payload), None
```

Artifact viewers let one plugin produce a result and another plugin display it.

Examples:

```text
table.filtered
table.numeric_profile
classifier.scores
classifier.model
image.cutout
scan.medical
spectra.desi
report.summary
```

Artifact viewers should retrieve data from `context.artifacts`.

They should not assume that an artifact payload is always a local file, dataframe, or specific domain object unless that is part of the artifact type contract.

---

## Workflows

Workflows assemble panels, services, and actions into reusable workspaces.

Example:

```python
def register(api):
    api.register_workflow(
        id="review",
        title="Review Workflow",
        builder=build_review_workspace,
        description="Open a record browser, visualisation panel, and review tools.",
    )


def build_review_workspace(context, **kwargs):
    context.plugins.open_panel("core.record_browser.panel", context=context)
    context.plugins.open_panel("core.visualisation.explorer", context=context)
    context.plugins.open_panel("core.selection_tools.selection_set", context=context)
```

A workflow should use platform APIs such as:

```python
context.workspace.add_panel(...)
context.plugins.open_panel(...)
context.plugins.run_action(...)
```

Do not directly mutate the template, grid, or another panel.

---

## Plugin settings

Plugins can register a settings schema and access plugin-scoped settings through the API.

Example:

```python
def register(api):
    api.register_settings_schema(
        {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "title": "API endpoint",
                },
                "timeout": {
                    "type": "number",
                    "title": "Timeout",
                    "default": 30,
                },
            },
        }
    )

    api.register_service(
        key="client",
        factory=create_client,
        lazy=True,
    )


def create_client(context, **kwargs):
    endpoint = context.plugins.get_plugin_setting(
        "example.remote",
        "endpoint",
        "https://example.invalid",
    )
    return RemoteClient(endpoint=endpoint)
```

Plugin settings storage is intentionally simple at this stage and may be extended by the persistence or configuration layer later.

---

## Selection-aware panel pattern

A detail panel should read the current focus when it starts and subscribe to future focus changes.

```python
class DetailPanel:
    def __init__(self, context):
        self.context = context
        self.subscriptions = []

        sub = context.events.subscribe(
            "selection.focus.changed",
            self.on_focus_changed,
            owner_id="example.detail",
            owner_label="Example Detail Panel",
            owner_kind="panel",
        )
        self.subscriptions.append(sub)

        current_focus = context.selection.get_focus()
        if current_focus:
            self.on_focus_changed("selection.focus.changed", current_focus)

    def on_focus_changed(self, topic, payload):
        dataset_id = payload["dataset_id"]
        row_id = payload["row_id"]

        df = self.context.datasets.get_df(dataset_id)
        id_col = self.context.datasets.get_mapping(dataset_id, "record_id")

        if id_col:
            row = df[df[id_col].astype(str) == str(row_id)]
        else:
            row = df.loc[[row_id]]

        self.render(row)

    def render(self, row):
        pass

    def panel(self):
        return self.view

    def dispose(self):
        for sub in self.subscriptions:
            self.context.events.unsubscribe(sub)
        self.subscriptions.clear()
```

A publisher panel should update canonical selection state rather than calling detail panels directly.

```python
class CatalogTable:
    def __init__(self, context):
        self.context = context

    def on_row_clicked(self, row_id):
        self.context.selection.set_focus(
            dataset_id=self.context.datasets.active_id(),
            row_id=row_id,
            origin="example.catalog_table",
        )
```

---

## Event plus artifact pattern

Use this pattern whenever a result may be reused by other plugins.

```python
class ScoreProducer:
    def __init__(self, context):
        self.context = context

    def compute_scores(self, row_ids, scores):
        artifact_id = self.context.artifacts.put(
            "classifier.scores",
            scores,
            dataset_id=self.context.datasets.active_id(),
            row_ids=row_ids,
            params={"model": "rf_v2"},
        )

        self.context.events.publish(
            "artifact.created",
            {
                "artifact_id": artifact_id,
                "type": "classifier.scores",
                "dataset_id": self.context.datasets.active_id(),
            },
        )
```

Consumer:

```python
class ScoreConsumer:
    def __init__(self, context):
        self.context = context
        self.subscriptions = []

        sub = context.events.subscribe(
            "artifact.created",
            self.on_artifact_created,
            owner_id="example.score_consumer",
            owner_label="Score Consumer",
            owner_kind="panel",
        )
        self.subscriptions.append(sub)

    def on_artifact_created(self, topic, payload):
        if payload.get("type") != "classifier.scores":
            return

        artifact_id = payload["artifact_id"]
        scores = self.context.artifacts.get(artifact_id)
        self.render(scores)

    def render(self, scores):
        pass

    def dispose(self):
        for sub in self.subscriptions:
            self.context.events.unsubscribe(sub)
        self.subscriptions.clear()
```

---

## Slow external fetch pattern

Use `jobs` for remote or expensive work.

```python
class RemoteDetailPanel:
    def __init__(self, context):
        self.context = context
        self._disposed = False
        self.subscriptions = []
        self.job_handles = []

        sub = context.events.subscribe(
            "selection.focus.changed",
            self.on_focus_changed,
            owner_id="example.remote_detail",
            owner_label="Remote Detail Panel",
            owner_kind="panel",
        )
        self.subscriptions.append(sub)

    def on_focus_changed(self, topic, payload):
        if self._disposed:
            return

        dataset_id = payload["dataset_id"]
        row_id = payload["row_id"]

        handle = self.context.jobs.submit(
            self.fetch_remote_detail,
            title="Fetch remote detail",
            key=f"remote-detail:{dataset_id}:{row_id}",
            on_done=lambda result: self.on_loaded(dataset_id, row_id, result),
            row_id=row_id,
        )
        self.job_handles.append(handle)

    def fetch_remote_detail(self, *, cancel_token, row_id):
        if cancel_token and cancel_token.cancelled():
            return None

        client = self.context.services.require("example.remote.client")
        result = client.fetch(row_id)

        if cancel_token and cancel_token.cancelled():
            return None

        return result

    def on_loaded(self, dataset_id, row_id, result):
        if self._disposed:
            return

        if result is None:
            return

        artifact_id = self.context.artifacts.put(
            "remote.detail",
            result,
            dataset_id=dataset_id,
            row_ids=[row_id],
        )

        self.context.events.publish(
            "artifact.created",
            {
                "artifact_id": artifact_id,
                "type": "remote.detail",
                "dataset_id": dataset_id,
            },
        )

    def dispose(self):
        if self._disposed:
            return

        self._disposed = True

        for sub in self.subscriptions:
            self.context.events.unsubscribe(sub)
        self.subscriptions.clear()

        for handle in self.job_handles:
            handle.cancel()
        self.job_handles.clear()
```

---

## Lifecycle rules

Plugins must clean up after themselves.

If a panel subscribes to events, it must unsubscribe.

If a panel starts jobs, it should cancel them when closed where appropriate.

If a panel creates widget watchers, periodic callbacks, scheduled callbacks, child controllers, or other handles, it should release them in `dispose()`.

`dispose()` must be idempotent. It should be safe to call more than once.

Event handlers and job callbacks should check `_disposed` before touching state or UI.

Panels that schedule refreshes with `add_next_tick_callback`, `add_timeout_callback`, periodic callbacks, or similar mechanisms must no-op if the panel has been disposed before the callback runs.

Recommended controller shape:

```python
class MyController:
    def __init__(self, context):
        self.context = context
        self._disposed = False
        self.subscriptions = []
        self.job_handles = []
        self.watchers = []
        self.periodic_callbacks = []

    def dispose(self):
        if self._disposed:
            return

        self._disposed = True

        for sub in list(self.subscriptions):
            try:
                self.context.events.unsubscribe(sub)
            except Exception:
                pass
        self.subscriptions.clear()

        for handle in list(self.job_handles):
            try:
                handle.cancel()
            except Exception:
                pass
        self.job_handles.clear()

        for widget, watcher in list(self.watchers):
            try:
                widget.param.unwatch(watcher)
            except Exception:
                pass
        self.watchers.clear()

        for callback in list(self.periodic_callbacks):
            try:
                callback.stop()
            except Exception:
                pass
        self.periodic_callbacks.clear()
```

The workspace manager calls `dispose()` when a managed panel is removed.

---

## Scheduled refresh pattern

If a panel schedules refresh work, guard against stale callbacks after panel closure.

```python
def schedule_refresh(self):
    if self._disposed or self._refresh_pending:
        return

    self._refresh_pending = True

    def _run():
        self._refresh_pending = False

        if self._disposed:
            return

        self.refresh()

    pn.state.curdoc.add_next_tick_callback(_run)
```

This prevents callbacks from updating closed panels.

---

## Workspace persistence

The plugin workspace can be saved and restored.

The persistence layer may record:

* schema name and version
* enabled plugin information
* dataset metadata and pending mappings
* selection state
* workspace panels
* panel layout
* panel restore metadata
* panel state where controllers expose it

Plugin authors do not need to do anything for basic layout restore.

To preserve internal widget state, implement `get_state()` and restore support in the panel or controller.

Example:

```python
class MyController:
    state_version = 1

    def __init__(self, context, restore_state=None, **kwargs):
        self.context = context
        restore_state = restore_state or {}
        self.selected_column = restore_state.get("selected_column")

    def get_state(self):
        return {
            "selected_column": self.selected_column,
        }
```

Panel factories should accept `**kwargs` so restore metadata can be passed through safely.

```python
def create_panel(context, **kwargs):
    controller = MyController(context, **kwargs)
    return controller.panel(), controller
```

Mapping-gated panels can wait for datasets and mappings before constructing the real panel. This is important when restoring workspaces before a dataset has been loaded.

---

## Platform headers

### Dataset header

Dataset loading and switching are platform-level concerns.

Plugins should not own the global dataset-loading workflow unless they are specifically implementing a dataset import plugin.

A plugin that needs data should:

1. read the active dataset from `context.datasets`
2. show an empty or waiting state if no dataset exists
3. optionally publish `dataset.open_requested` to ask the platform UI to open the dataset loader

Example:

```python
context.events.publish(
    "dataset.open_requested",
    {
        "source": "my_plugin",
    },
)
```

### Column-mapping header

Column mapping is also platform-level.

A plugin should declare `required_mappings` and `optional_mappings` during registration. It should not create its own generic mapping UI.

The platform listens for mapping requests, lets the user resolve them, stores mappings in `context.datasets`, and publishes mapping events.

Useful mapping events include:

```text
mapping.requested
mapping.resolved
mapping.open_requested
dataset.mapping_updated
```

---

## Core plugins

Core plugins ship with the main AstronomicAL repository.

They are still real plugins and should follow the same rules as external plugins.

### `core.record_browser`

Location:

```text
astronomicAL/plugins/record_browser/plugin.py
```

Provides:

* generic active-dataset record browsing
* replacement for the generic browsing part of old Exploration mode
* record navigation
* record ID search
* visible metadata fields
* label display settings
* focus publication through `context.selection`
* focus consumption through selection events

Plugin id:

```text
core.record_browser
```

Registered panel:

```text
core.record_browser.panel
```

Important mappings:

```text
required: record_id
optional: target_label
```

This plugin intentionally does not:

* load datasets
* own column mapping setup
* require domain-specific columns
* generate domain-specific fields
* create training features
* create placeholder labels

---

### `core.visualisation`

Location:

```text
astronomicAL/plugins/visualisation/plugin.py
```

Provides:

* generic visualisation panels for the active dataset
* shared visualisation state service
* scatter plot
* histogram
* 2D density plot
* linked plot explorer
* focus publication from plot interactions
* multi-row selection publication from plot selections
* optional label-aware rendering
* datashader fallback for large datasets

Plugin id:

```text
core.visualisation
```

Registered service:

```text
core.visualisation.state
```

Registered panels:

```text
core.visualisation.scatter
core.visualisation.histogram
core.visualisation.density
core.visualisation.explorer
```

Important mappings:

```text
required: record_id
optional: target_label
```

This plugin is intended for generic tabular visualisation. Domain-specific visualisations should live in domain plugins or domain-specific visualisation plugins.

---

### `core.table_tools`

Location:

```text
astronomicAL/plugins/table_tools/plugin.py
```

Provides:

* Table Transform panel
* pandas-style expression preview
* derived-column creation
* boolean subset preview
* subset dataset creation
* dataset update events

Plugin id:

```text
core.table_tools
```

Registered panel:

```text
core.table_tools.transform_panel
```

Registered actions:

```text
core.table_tools.add_column
core.table_tools.create_subset
```

Typical events:

```text
dataset.updated
dataset.loaded
dataset.active.changed
```

Use this plugin as a model for turning table operations into actions.

---

### `core.selection_tools`

Location:

```text
astronomicAL/plugins/selection_tools/plugin.py
```

Provides:

* Selection Set panel
* focused-row inspection
* active multi-row selection inspection
* selected-row preview
* selection clearing
* focus stepping through selected rows
* derived dataset creation from the active selection

Plugin id:

```text
core.selection_tools
```

Registered panel:

```text
core.selection_tools.selection_set
```

Important mappings:

```text
optional: record_id
```

If `record_id` is unmapped, Selection Tools can fall back to the dataframe index where possible.

---

### `core.annotations`

Location:

```text
astronomicAL/plugins/annotations/plugin.py
```

Provides:

* record-level notes
* review state
* label suggestions
* annotation records
* annotation summary tooling
* annotation-related artifacts or events
* integration points for review and active-learning workflows

Plugin id:

```text
core.annotations
```

Typical panels and actions may include:

```text
core.annotations.panel
core.annotations.summary
core.annotations.build_summary
```

Annotations should use `context.selection` to follow the current focused row. They should publish lightweight events when review state or labels change and store reusable annotation outputs as artifacts or datasets depending on durability.

---

### `core.event_monitor`

Location:

```text
astronomicAL/plugins/event_monitor/plugin.py
```

Provides:

* Event Monitor panel
* EventBus diagnostics
* recent event trace
* subscription inspection
* owner/topic visibility
* dataset-event coverage
* periodic refresh
* plugin lifecycle cleanup

Plugin id:

```text
core.event_monitor
```

Use this plugin when debugging plugin communication, missing subscriptions, unexpected event storms, or stale listeners.

---

### `core.plugin_manager`

Location:

```text
astronomicAL/plugins/plugin_manager/plugin.py
```

Provides:

* Plugin Manager panel
* discovered plugin inspection
* enabled/disabled plugin inspection
* registered panel/action/service/workflow/artifact-viewer lists
* open panel instance inspection
* basic plugin diagnostics

Plugin id:

```text
core.plugin_manager
```

Use this plugin when checking whether a plugin was discovered, enabled, or registered correctly.

---

## Menus and panel discovery

Plugin panels should be registered through `api.register_panel(...)`.

Do not add new panels directly to legacy custom-plot dictionaries unless you are temporarily bridging old code.

The menu system can discover registered plugin panels from `context.plugins`. Categories should be used to keep the UI organised.

Good categories:

```text
Core
Core / Visualisation
Data tools
Selection
Diagnostics
Domain
Domain / Inspection
Domain / External data
Workflow
Active Learning
Review
Quality Control
Examples
```

The current direction is to organise menus by plugin, domain, application, or category rather than hard-coding every panel into one monolithic list.

---

## Legacy Choose Plot bridge

Plugin panels should be opened through the plugin and workspace system.

During migration, some plugin panels may also appear in the legacy Choose Plot menu through compatibility code. New plugins should not target that bridge directly.

The long-term direction is for panels to be registered and opened through the plugin system.

---

## Dependency guidance

Keep heavyweight, optional, and domain-specific dependencies out of the platform core where possible.

Core plugins should avoid depending on libraries that only make sense for one domain.

Domain-specific tools should live in domain plugins and declare their own requirements.

Example:

```python
manifest = PluginManifest(
    id="medical.scan_viewer",
    name="Medical Scan Viewer",
    version="0.1.0",
    description="Panels and viewers for medical scan review.",
    capabilities=["panel", "artifacts"],
    tags=["medical", "imaging"],
    requires=[
        "pydicom",
    ],
)
```

Example:

```python
manifest = PluginManifest(
    id="factory.sensor_review",
    name="Factory Sensor Review",
    version="0.1.0",
    description="Tools for reviewing sensor streams and machine events.",
    capabilities=["panel", "service", "artifacts"],
    tags=["manufacturing", "sensors"],
    requires=[
        "pyarrow",
    ],
)
```

Example:

```python
manifest = PluginManifest(
    id="astro.spectra",
    name="Astronomy Spectra",
    version="0.1.0",
    description="Panels and services for astronomical spectra.",
    capabilities=["panel", "service", "artifacts"],
    tags=["astronomy", "spectra"],
    requires=[
        "astropy",
    ],
)
```

Do not force users to install dependencies for domains they do not use.

---

## Suggested plugin id conventions

Use stable dotted identifiers.

Good:

```text
core.table_tools
core.record_browser
core.visualisation
astro.euclid
medical.scan_viewer
factory.sensor_review
active_learning.core
example.demo
```

Avoid:

```text
test
plugin
my stuff
newplugin
```

Plugin ids should be stable because they are used in:

* registration ids
* service keys
* workspace persistence
* settings
* debugging
* dependency declarations

---

## Suggested artifact type conventions

Use dotted artifact types.

Good:

```text
classifier.scores
classifier.model
table.filtered
table.numeric_profile
selection.ids
image.cutout
scan.medical
spectra.desi
sensor.trace
report.summary
review.queue
```

Avoid vague names:

```text
result
data
output
thing
```

Artifact types are interoperability points. Other plugins may use them to decide which viewers or workflows are available.

---

## Suggested event conventions

Use dotted event topics.

Common platform topics:

```text
dataset.loaded
dataset.active.changed
dataset.updated
dataset.mapping_updated
mapping.requested
mapping.resolved
mapping.open_requested
selection.focus.changed
selection.focus.cleared
selection.set.changed
selection.set.cleared
artifact.created
plugin.enabled
plugin.disabled
workspace.saved
workflow.stage.changed
```

Domain plugins should namespace their events.

Examples:

```text
domain.external.loaded
review.status.changed
active_learning.iteration.completed
active_learning.query.selected
quality_control.item.approved
medical.scan.loaded
factory.sensor.loaded
astro.spectrum.loaded
```

Keep event payloads small and JSON-friendly where possible.

---

## Suggested service key conventions

Use stable namespaced keys.

Examples:

```text
core.visualisation.state
example.remote.client
database.primary
storage.cache
medical.fhir.client
factory.sensor_store
astro.archive.client
```

The plugin API automatically namespaces service keys when needed.

---

## Suggested dataset id conventions

Use durable identifiers rather than display labels.

Good:

```text
main_catalog
main_dataset
training_set
test_set
review_queue
high_score_subset
selected_rows
```

Avoid:

```text
My Dataset!!!
new data
temp
```

Dataset ids may appear in workspace snapshots, artifact metadata, selection state, and event payloads.

---

## Migration guide for old features

When moving existing AstronomicAL code into the plugin system, use this mapping.

| Old pattern | New pattern |
| --- | --- |
| Global dataframe in config | `context.datasets` |
| Current record stored in config or panel state | `context.selection.set_focus(...)` |
| Multi-row selected ids stored in ad hoc variables | `context.selection.set_selection_set(...)` |
| Shared computed state | `context.artifacts` |
| One panel directly calling another | `context.selection` plus `context.events` |
| Ad hoc background thread | `context.jobs` |
| Direct template/grid mutation | `context.workspace` |
| Global API client | `context.services` |
| Hard-coded column names | `required_mappings` or `optional_mappings` |
| Monolithic mode-specific workflow | plugin workflow |
| Domain-specific panel in core | domain plugin |

Recommended migration order:

1. Inject `context`.
2. Move source dataframe access to `context.datasets`.
3. Move focused row and selected rows to `context.selection`.
4. Replace direct panel calls with events.
5. Move derived results to `context.artifacts`.
6. Move slow work to `context.jobs`.
7. Register clients/connectors as `context.services`.
8. Add `required_mappings` and `optional_mappings`.
9. Register the feature as a panel, action, service, artifact viewer, or workflow.
10. Remove new writes to `context.config` unless they are strictly compatibility bridges.

---

## Adapting domain-specific features

Domain-specific functionality should live in domain plugins rather than the platform core.

A domain plugin may provide:

* semantic column mappings
* domain-specific panels
* domain-specific actions
* external service clients
* artifact viewers
* workflow templates
* import/export tools
* validation or review tools

The platform should not need to understand the meaning of a domain concept. It should only provide the generic services that let plugins coordinate.

Examples:

| Domain feature | Plugin-system shape |
| --- | --- |
| External archive, API, or database client | service |
| Remote image, scan, document, trace, or cutout fetch | action or selection-aware panel using jobs |
| Remote or computed data product | artifact |
| Image, scan, spectrum, trace, document, or report viewer | artifact viewer or panel |
| Domain-specific plot | panel |
| Domain-specific coordinate, identifier, or measurement columns | required or optional mappings |
| Review or triage layout | workflow |
| Batch processing operation | action |
| Export or report generation | action, artifact, or workflow |

Example domain panel registration:

```python
api.register_panel(
    id="detail_viewer",
    title="Detail Viewer",
    factory=create_detail_viewer,
    category="Domain / Inspection",
    required_mappings=[
        "record_id",
        "domain.primary_measurement",
    ],
    optional_mappings=[
        "target_label",
    ],
    produces=[
        "artifact.created",
    ],
)
```

The generic platform should not need to know what the mapped columns mean.

A medical plugin may define:

```text
patient.id
scan.accession_number
```

A manufacturing plugin may define:

```text
part.serial_number
machine.id
sensor.timestamp
```

An ecology plugin may define:

```text
observation.id
species.name
recording.path
```

An astronomy plugin may define:

```text
coords.ra
coords.dec
source.id
```

Those meanings belong in plugins.

---

## Adapting active learning

Active learning should become a reusable workflow or plugin bundle rather than the built-in identity of the application.

The active-learning plugin should remain domain-agnostic where possible. It should operate on datasets, labels, features, selections, models, and artifacts.

Domain-specific inspection tools should be added as separate panels in the same workflow.

For example:

* an astronomy workflow may combine active learning with image cutouts and spectra
* a medical workflow may combine active learning with scan viewers and patient metadata
* a manufacturing workflow may combine active learning with defect images and sensor traces
* an ecology workflow may combine active learning with images, audio clips, or observation records
* a finance workflow may combine active learning with transaction details and entity summaries

A clean active-learning split would be:

| Capability | Plugin-system shape |
| --- | --- |
| classifier setup UI | panel |
| train model | action, usually job-backed |
| query next examples | action |
| model scores | artifact |
| trained model | artifact |
| selected query candidate | `selection.focus.changed` |
| labelled training data | dataset or artifact |
| test/validation set | dataset |
| training metrics | artifact |
| active-learning dashboard | workflow |

Example artifact types:

```text
active_learning.query
classifier.scores
classifier.model
classifier.metrics
labels.updated
```

The active-learning plugin should not need to know which domain-specific panels are open. It should publish selections, artifacts, and events that other plugins can react to.

---

## Development checklist

Before committing a plugin, check:

* [ ] The plugin lives outside `astronomicAL/platform/plugins/`.
* [ ] `plugin.py` exposes a module-level `manifest`.
* [ ] `plugin.py` defines `register(api)`.
* [ ] The manifest is cheap to import.
* [ ] Heavy optional imports happen inside factories, handlers, or services.
* [ ] Panel factories accept `context`.
* [ ] Panel factories accept `**kwargs`.
* [ ] Panels return `(view, controller)` where possible.
* [ ] Controllers implement `dispose()` when needed.
* [ ] `dispose()` is idempotent.
* [ ] Event handlers and job callbacks guard against `_disposed`.
* [ ] Scheduled callbacks no-op after disposal.
* [ ] Source data is read from `context.datasets`.
* [ ] Current focus and multi-row selections use `context.selection`.
* [ ] Derived outputs use `context.artifacts`.
* [ ] Notifications use `context.events`.
* [ ] Slow work uses `context.jobs`.
* [ ] Runtime clients use `context.services`.
* [ ] Visible panels are added through `context.workspace` or plugin manager helpers.
* [ ] Column-dependent plugins declare `required_mappings` or `optional_mappings`.
* [ ] The plugin does not add new runtime state to `context.config` unless unavoidable.
* [ ] Events have stable dotted names.
* [ ] Artifact types have stable dotted names.
* [ ] Service keys are stable and namespaced.
* [ ] Panel category and title are useful in menus.
* [ ] The plugin can be discovered without breaking app startup.
* [ ] The plugin can be disabled without leaving subscriptions, jobs, or panels behind.
* [ ] Any workspace-restorable internal state is exposed through `get_state()`.
* [ ] Persistent state is JSON-safe.

---

## Debugging plugin startup

Plugin and boot debug logs are normally quiet.

To enable boot-order diagnostics:

```bash
ASTRONOMICAL_DEBUG_BOOT=1 panel serve astronomicAL --show
```

To enable plugin-specific diagnostics:

```bash
ASTRONOMICAL_DEBUG_PLUGINS=1 panel serve astronomicAL --show
```

To debug mapping behaviour:

```bash
ASTRONOMICAL_DEBUG_MAPPING=1 panel serve astronomicAL --show
```

To debug workspace persistence:

```bash
ASTRONOMICAL_DEBUG_PERSISTENCE=1 panel serve astronomicAL --show
```

A healthy startup should show that plugins were discovered, enabled, registered, and made available to the menu or workspace layer.

Useful diagnostic plugins:

```text
core.event_monitor
core.plugin_manager
```

---

## Common mistakes

### Putting plugins in the framework directory

Wrong:

```text
astronomicAL/platform/plugins/my_plugin/plugin.py
```

Right:

```text
astronomicAL/plugins/my_plugin/plugin.py
```

or a configured user/development plugin directory.

---

### Importing heavy domain dependencies at manifest import time

Wrong:

```python
from PIL import Image
from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="domain.image_tools",
    name="Image Tools",
    version="0.1.0",
)
```

Right:

```python
from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="domain.image_tools",
    name="Image Tools",
    version="0.1.0",
    requires=["pillow"],
)


def create_panel(context, **kwargs):
    from PIL import Image

    return build_panel(context, image_module=Image)
```

---

### Storing data products in services

Wrong:

```python
context.services.set("latest_scores", scores)
```

Right:

```python
context.artifacts.put(
    "classifier.scores",
    scores,
    dataset_id=context.datasets.active_id(),
)
```

---

### Using events as storage

Wrong:

```python
context.events.publish("table.filtered", filtered_dataframe)
```

Right:

```python
artifact_id = context.artifacts.put(
    "table.filtered",
    filtered_dataframe.to_dict(orient="records"),
    dataset_id=context.datasets.active_id(),
)

context.events.publish(
    "artifact.created",
    {
        "artifact_id": artifact_id,
        "type": "table.filtered",
    },
)
```

---

### Letting panels call each other directly

Wrong:

```python
detail_panel.show_row(row_id)
plot_panel.highlight_row(row_id)
document_panel.load(row_id)
```

Right:

```python
context.selection.set_focus(
    dataset_id=context.datasets.active_id(),
    row_id=row_id,
    origin="my_plugin",
)
```

---

### Hard-coding column names

Wrong:

```python
record_id = df["record_id"]
value = df["value"]
```

Right:

```python
id_col = context.datasets.get_mapping(dataset_id, "record_id")
value_col = context.datasets.get_mapping(dataset_id, "domain.primary_measurement")

record_id = df[id_col]
value = df[value_col]
```

and declare the mappings when registering the panel:

```python
api.register_panel(
    id="domain_view",
    title="Domain View",
    factory=create_domain_view,
    required_mappings=[
        "record_id",
        "domain.primary_measurement",
    ],
)
```

---

### Bypassing workspace lifecycle

Wrong:

```python
template.main.append(view)
```

Right:

```python
context.workspace.add_panel(
    panel_id="my_panel",
    title="My Panel",
    view=view,
    controller=controller,
)
```

or:

```python
context.plugins.open_panel(
    "example.plugin.panel",
    context=context,
)
```

---

### Forgetting cleanup

Wrong:

```python
class MyPanel:
    def __init__(self, context):
        context.events.subscribe("selection.focus.changed", self.on_focus)
```

Right:

```python
class MyPanel:
    def __init__(self, context):
        self.context = context
        self._disposed = False
        self.subscriptions = [
            context.events.subscribe(
                "selection.focus.changed",
                self.on_focus,
                owner_id="my_plugin.my_panel",
                owner_label="My Panel",
                owner_kind="panel",
            )
        ]

    def on_focus(self, topic, payload):
        if self._disposed:
            return

    def dispose(self):
        if self._disposed:
            return

        self._disposed = True

        for sub in self.subscriptions:
            self.context.events.unsubscribe(sub)
        self.subscriptions.clear()
```

---

### Blocking the UI in a panel factory

Wrong:

```python
def create_panel(context, **kwargs):
    data = remote_client.fetch_large_payload()
    return build_panel(data), None
```

Right:

```python
def create_panel(context, **kwargs):
    controller = RemotePanel(context)
    controller.start_fetch()
    return controller.panel(), controller
```

with the slow fetch run through `context.jobs`.

---

## Design principles

### Keep the platform small

If something is domain-specific, workflow-specific, or optional, it should usually be a plugin.

### Treat `context` as the host API

Do not import global runtime state when a platform service exists.

### Prefer semantic mappings over column names

A plugin should describe what it needs, not assume how every dataset names it.

### Prefer events over direct references

Panels should communicate through platform state and events.

### Prefer artifacts over large event payloads

Events announce that something happened. Artifacts store reusable outputs.

### Prefer jobs over ad hoc threads

Long-running work should be cancellable, deduplicated, and visible to the platform.

### Prefer workspace lifecycle over direct layout mutation

Panels should be opened, restored, and disposed by the workspace layer.

### Keep domain logic in domain plugins

Domain-specific logic should live in domain plugins. Other domains should be able to use AstronomicAL without installing dependencies or accepting assumptions they do not need.

### Keep Core plugins generic

Core plugins should provide broadly useful platform functionality. They should not become a hidden place for domain-specific code.

---

## Future direction

The plugin system is intended to make AstronomicAL a generic analysis host.

The long-term direction is:

```text
small generic platform
+ generic Core plugins
+ optional workflow plugins
+ optional domain plugins
+ optional plugin bundles
+ optional user plugins
+ optional third-party plugins
```

Core plugins should provide broadly useful functionality such as record browsing, visualisation, table tools, selection tools, annotations, diagnostics, and plugin management.

Domain plugins should provide field-specific functionality. Examples include astronomy archive tools, medical scan viewers, manufacturing sensor tools, ecology media review tools, materials microscopy tools, and finance transaction review tools.

Workflow plugins should provide reusable workflows such as active learning, catalogue curation, anomaly triage, quality control, review queues, or report generation.

Plugin bundles should eventually allow users to install or enable a collection of related plugins together.

For example:

* an active-learning bundle could include model training, query strategy, labelling, metrics, and report plugins
* a review bundle could include record browsing, selection tools, notes, labels, and export plugins
* a domain bundle could include the panels, services, artifact viewers, and workflows needed by a specific field
* a diagnostics bundle could include event monitoring, plugin inspection, job monitoring, and artifact inspection

The platform itself should remain neutral. It should not assume what kind of records the user is analysing.