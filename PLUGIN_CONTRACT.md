# AstronomicAL Plugin Contract v0

**Status:** Draft baseline
**Audience:** AstronomicAL core contributors, bundled plugin authors, domain plugin authors, workflow plugin authors
**Scope:** The plugin-facing contract for the `plugin_system` architecture

This contract defines how plugins interact with the AstronomicAL platform. Its purpose is to keep the core small, generic, and stable while allowing astronomy, active learning, visualisation, annotation, table tooling, and future non-astro workflows to grow as composable plugins.

The contract follows the direction laid out in the platform, migration, and plugin examples: source data belongs in `datasets`, focus/selection belongs in `selection`, derived outputs belong in `artifacts`, communication goes through `events`, slow work goes through `jobs`, visible UI goes through `workspace`, live clients belong in `services`, and `config` is only a migration bridge.   

---

# 1. Core principle

AstronomicAL core is a **plugin host**, not a domain application.

A plugin may be highly domain-specific. The core must remain domain-neutral.

```text
Core responsibility:
  Provide runtime services, plugin registration, lifecycle, persistence,
  mapping gates, and workspace composition.

Plugin responsibility:
  Declare capabilities, declare requirements, use context services,
  clean up after itself, and keep domain/workflow assumptions outside core.
```

Astronomy, active learning, annotations, table transforms, visualisation, and record browsing should all be expressible as plugins or plugin bundles.

---

# 2. Terminology

## Platform

The reusable runtime layer that owns:

```text
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

## Plugin

A package or module that contributes panels, actions, workflows, services, artifact viewers, or other extension points through the plugin API.

## Bundled plugin

A plugin distributed with AstronomicAL.

Examples may include:

```text
record_browser
visualisation
selection_tools
table_tools
annotations
event_monitor
plugin_manager
```

## Domain plugin

A plugin that contains domain-specific logic.

Examples:

```text
astro.euclid
astro.desi
astro.sdss
medical.imaging
materials.assay
geo.raster
```

## Workflow plugin

A plugin that assembles multiple panels/actions/services into a coherent workflow.

Examples:

```text
active_learning
catalogue_review
model_audit
annotation_workflow
```

## Panel

A visible UI component managed by `context.workspace`.

## Action

A callable operation registered by a plugin.

## Artifact

A reusable derived result produced by a plugin or workflow.

## Service

A live runtime capability such as an API client, database connection, or authenticated session.

---

# 3. Normative language

This contract uses:

```text
MUST       required for v0 compatibility
SHOULD     strongly recommended; deviations need justification
MAY        optional
MUST NOT   forbidden for new plugin code
```

---

# 4. Plugin shape

A v0 plugin MUST expose:

```python
manifest = {
    "id": "example.plugin",
    "name": "Example Plugin",
    "version": "0.1.0",
    "description": "Short human-readable description",
}

def register(api):
    ...
```

A plugin MAY expose additional module-level metadata, but `manifest` and `register(api)` are the v0 entry points.

---

# 5. Manifest contract

## Required manifest fields

```python
manifest = {
    "id": "example.plugin",
    "name": "Example Plugin",
    "version": "0.1.0",
    "description": "Adds an example panel or workflow",
}
```

### `id`

The plugin id MUST be globally stable.

It SHOULD use a dotted namespace:

```text
core.record_browser
core.visualisation
core.annotations
astro.desi
astro.euclid
workflow.active_learning
```

Plugin ids MUST NOT contain spaces.

### `name`

A short display name.

### `version`

A semantic version string.

For v0, semantic versioning SHOULD be followed:

```text
MAJOR.MINOR.PATCH
```

### `description`

A human-readable explanation of the plugin.

---

## Optional manifest fields

A plugin MAY declare:

```python
manifest = {
    "id": "astro.desi",
    "name": "DESI Tools",
    "version": "0.1.0",
    "description": "DESI spectra panels and services",
    "author": "AstronomicAL contributors",
    "homepage": "...",
    "license": "...",
    "requires": ["core.record_browser"],
    "python_requires": ">=3.10",
    "dependencies": ["sparclclient", "astropy"],
    "tags": ["astronomy", "spectra"],
}
```

## Dependency rule

Domain-specific or heavyweight dependencies SHOULD be declared by the plugin that needs them.

They MUST NOT be required by AstronomicAL core unless the core genuinely cannot function without them.

This is especially important for astronomy dependencies such as survey clients, FITS/WCS tooling, remote cutout libraries, MOC tooling, or domain-specific visualisation packages.

---

# 6. Registration contract

A plugin registers all contributions inside:

```python
def register(api):
    ...
```

The plugin MUST NOT mutate global application state during import.

Importing a plugin module SHOULD be cheap.

Heavy setup MUST be deferred until registration, service initialization, panel creation, or job execution.

---

# 7. Supported contribution types

A v0 plugin MAY register:

```text
panels
actions
dataframe actions
workflows
services
artifact viewers
```

The plugin API is the only supported route for these contributions.

A plugin MUST NOT directly patch AstronomicAL menus, mutate the workspace grid, or append views to templates outside `context.workspace`.

---

# 8. Context contract

Every plugin contribution that needs runtime access MUST use `context`.

A plugin SHOULD treat `context` as its host API.

```python
class MyPanel:
    def __init__(self, context):
        self.context = context
        self.datasets = context.datasets
        self.selection = context.selection
        self.events = context.events
        self.artifacts = context.artifacts
        self.jobs = context.jobs
        self.services = context.services
```

## Stable v0 services

The following services are part of the v0 contract:

```text
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

## Transitional service

```text
context.config
```

`context.config` exists only for migration.

New plugins MUST NOT use `context.config` for:

```text
current row
selection set
derived results
service clients
cached data products
panel-to-panel communication
active dataset ownership
```

`context.config` MAY be used temporarily to read legacy configuration that has not yet been migrated.

The migration guide explicitly treats `config` as a compatibility bridge, not a permanent home for new state. 

---

# 9. Data ownership contract

## Source data

Canonical source data MUST live in `context.datasets`.

Examples:

```text
loaded catalogue
training table
review queue
test set
user-imported dataframe
promoted filtered subset
```

A plugin SHOULD register source-like data as a dataset:

```python
context.datasets.register(
    "main_catalog",
    df,
    name="Main Catalog",
    source="catalog.parquet"
)
```

## Derived data

Computed or cached outputs MUST live in `context.artifacts`.

Examples:

```text
classifier scores
model outputs
spectra
cutouts
embeddings
filtered table previews
reports
remote API responses
feature-generation outputs
```

A plugin SHOULD store derived outputs as artifacts:

```python
artifact_id = context.artifacts.put(
    "classifier.scores",
    scores,
    dataset_id=context.datasets.active_id(),
    row_ids=row_ids,
    params={"model": "rf_v2"}
)
```

## Durable derived tables

A derived table SHOULD become a dataset only when users are expected to treat it as a new working input.

```python
context.datasets.register(
    "high_score_subset",
    filtered_df,
    name="High Score Subset",
    derived_from=context.datasets.active_id(),
    filter="score > 0.9"
)
```

Otherwise, it SHOULD remain an artifact.

This distinction is central to the platform model. 

---

# 10. Selection contract

Plugins MUST use `context.selection` for shared row focus and multi-row selection state.

## Focus

Focus means one current row/object.

```python
context.selection.set_focus(
    dataset_id=context.datasets.active_id(),
    row_id=row_id,
    origin="catalog_table"
)
```

Consumers SHOULD subscribe to:

```text
selection.focus.changed
selection.focus.cleared
```

A newly opened panel SHOULD read current focus immediately:

```python
current_focus = context.selection.get_focus()
if current_focus:
    self.on_selection_focus_changed(
        "selection.focus.changed",
        current_focus
    )
```

## Selection set

A selection set means multiple rows.

```python
context.selection.set_selection_set(
    dataset_id=context.datasets.active_id(),
    row_ids=row_ids,
    origin="scatter_lasso"
)
```

Consumers SHOULD subscribe to:

```text
selection.set.changed
selection.set.cleared
```

## Required distinction

Plugins MUST NOT confuse single-row focus with multi-row selection.

A tap/click on one object SHOULD normally update focus.

A lasso/brush/multi-select SHOULD update the active selection set.

The platform docs explicitly separate focus from selection sets so panels can query current workflow state at any time. 

---

# 11. Event contract

Plugins MUST use `context.events` for notifications and loose coupling.

A plugin SHOULD publish events when something relevant changes:

```python
context.events.publish(
    "artifact.created",
    {
        "artifact_id": artifact_id,
        "type": "classifier.scores",
        "dataset_id": context.datasets.active_id(),
    }
)
```

## Events are notifications

Events MUST NOT be used as bulk data storage.

Bad:

```python
context.events.publish("scores.updated", huge_dataframe)
```

Good:

```python
artifact_id = context.artifacts.put(
    "classifier.scores",
    scores,
    dataset_id=context.datasets.active_id()
)

context.events.publish(
    "artifact.created",
    {
        "artifact_id": artifact_id,
        "type": "classifier.scores"
    }
)
```

The event-plus-artifact pattern is a core part of the platform direction.  

## Canonical v0 event topics

Plugins SHOULD use dotted event names.

Recommended v0 topics:

```text
dataset.loaded
dataset.active.changed
dataset.mapping.updated

selection.focus.changed
selection.focus.cleared
selection.set.changed
selection.set.cleared

artifact.created
artifact.updated
artifact.deleted

labels.updated
review.status.changed

workflow.stage.changed
workflow.started
workflow.completed
workflow.failed

plugin.enabled
plugin.disabled
plugin.error
```

Plugins MAY introduce additional namespaced topics:

```text
astro.desi.spectrum.loaded
active_learning.query.created
annotations.note.created
table_tools.subset.created
```

Custom topics SHOULD be prefixed by plugin or workflow namespace.

---

# 12. Artifact contract

Plugins MUST use `context.artifacts` for reusable computed outputs.

Artifact types SHOULD use namespaced strings:

```text
classifier.scores
classifier.model
active_learning.query_batch
selection.ids
table.filtered
table.transform_preview
cutout.euclid
spectra.desi
spectra.sdss
coords.icrs
report.summary
annotation.record
```

## Artifact creation pattern

```python
artifact_id = context.artifacts.put(
    "spectra.desi",
    result,
    dataset_id=dataset_id,
    row_ids=[row_id],
    params={"source": "sparcl"}
)

context.events.publish(
    "artifact.created",
    {
        "artifact_id": artifact_id,
        "type": "spectra.desi",
        "dataset_id": dataset_id,
        "row_ids": [row_id],
    }
)
```

## Artifact payloads

Artifact payloads MAY be:

```text
small dicts
lists
dataframe-like records
model metadata
paths to persisted files
remote response summaries
serializable analysis outputs
```

Plugins SHOULD avoid storing huge non-serializable objects directly unless the artifact store explicitly supports that use case.

Long-lived live objects belong in `services`, not artifacts.

---

# 13. Job contract

Plugins MUST use `context.jobs` for slow, cancellable, or remote work.

Examples:

```text
remote API fetch
large preprocessing task
model training
model inference over many rows
image cutout retrieval
spectra download
report generation
file export
```

A plugin MUST NOT block the UI thread with expensive work.

## Job pattern

```python
def fetch_spectrum(*, cancel_token, row_id):
    if cancel_token and cancel_token.cancelled():
        return None

    client = context.services.get("desi_client")
    result = client.fetch_spectrum(row_id)

    if cancel_token and cancel_token.cancelled():
        return None

    return result


context.jobs.submit(
    fetch_spectrum,
    title="Fetch spectrum",
    key=f"spectrum:{dataset_id}:{row_id}",
    on_done=lambda result: self.on_loaded(dataset_id, row_id, result),
    row_id=row_id,
)
```

## Job keys

Job keys SHOULD be stable and deduplicate repeated work:

```text
spectrum:main_catalog:12345
cutout:euclid:main_catalog:12345
model_train:rf_v2:training_set
```

## Cancellation

Long-running jobs SHOULD check `cancel_token`.

Panels that launch jobs SHOULD cancel them in `dispose()` when appropriate.

The platform guides identify `jobs` as the shared mechanism for slow work, cancellation, and deduplication. 

---

# 14. Service contract

Plugins MUST use `context.services` for live runtime capabilities.

Examples:

```text
API clients
database connections
authenticated sessions
filesystem adapters
remote service handles
model-serving clients
cache backends
```

Example:

```python
def register(api):
    api.register_service(
        id="desi_client",
        factory=create_desi_client,
        lazy=True,
    )
```

Usage:

```python
client = context.services.get("astro.desi.desi_client")
```

## Services are not data storage

Plugins MUST NOT store computed outputs in services.

Bad:

```python
context.services.set("latest_scores", scores)
```

Good:

```python
context.artifacts.put(
    "classifier.scores",
    scores,
    dataset_id=context.datasets.active_id()
)
```

The platform guide is explicit that services are for live capabilities, not source tables or derived data products. 

---

# 15. Workspace and panel lifecycle contract

Plugins MUST open visible UI through `context.workspace` or the plugin manager.

Plugins MUST NOT directly mutate the underlying template/grid from arbitrary modules.

## Panel registration

A plugin panel SHOULD be registered through the plugin API:

```python
def register(api):
    api.register_panel(
        id="record_browser",
        title="Record Browser",
        factory=RecordBrowserPanel,
        required_mappings=["record_id"],
        persistent=True,
    )
```

## Panel factory

A panel factory MUST accept `context` either directly or through a supported factory signature.

Recommended:

```python
class RecordBrowserPanel:
    def __init__(self, context, **kwargs):
        self.context = context
```

## Panel disposal

A panel/controller that subscribes, watches, or launches jobs MUST implement `dispose()`.

```python
def dispose(self):
    for sub in self.subscriptions:
        self.context.events.unsubscribe(sub)

    for handle in self.job_handles:
        handle.cancel()

    self.subscriptions.clear()
    self.job_handles.clear()
```

The plugin examples treat cleanup as essential for a stable plugin workspace. 

## Platform promises

For registered panels, the platform SHOULD:

```text
validate requirements before creation
apply mapping gates when needed
construct the panel through the registered factory
track the panel in workspace
associate panel with plugin id and registration id
call dispose() when the panel closes
include persistent panels in workspace snapshots
restore panel state when supported
remove plugin-owned panels when plugin is disabled
```

## Plugin responsibilities

A panel plugin MUST:

```text
avoid hidden globals
avoid direct calls to other panels
use context services
clean up subscriptions, jobs, and watchers
return JSON-safe state from get_state()
handle missing data gracefully
handle dataset switches where relevant
```

---

# 16. Mapping contract

Plugins MUST declare semantic column requirements instead of assuming hard-coded column names.

Example:

```python
api.register_panel(
    id="spectra",
    title="DESI Spectra",
    factory=SpectrumPanel,
    required_mappings=["record_id"],
    optional_mappings=["coords.ra", "coords.dec"],
)
```

## Required mappings

A required mapping means the panel/action cannot work without it.

The platform SHOULD gate panel creation until required mappings are resolved.

## Optional mappings

An optional mapping means the plugin can provide enhanced behaviour when available.

The plugin MUST still work without optional mappings.

## Common semantic mappings

Recommended v0 mapping names:

```text
record_id
target_label
label
features
coords.ra
coords.dec
image.path
image.url
time
group_id
```

Domain plugins MAY introduce domain-specific semantic mappings:

```text
astro.redshift
astro.object_type
medical.patient_id
medical.scan_path
materials.compound_id
geo.latitude
geo.longitude
```

## Plugin behaviour

A plugin MUST resolve actual dataframe columns through the dataset mapping system.

It MUST NOT assume that a catalogue column is literally named `id`, `ra`, `dec`, `class`, or `label`.

The migration docs emphasize this shift from hard-coded domain assumptions toward semantic mappings. 

---

# 17. Action contract

Plugins MAY register actions.

Actions are non-panel operations that can be invoked by menus, buttons, workflows, or other plugins.

Example:

```python
def create_subset(context, request):
    selection = context.selection.get_active_set()
    if not selection:
        return None

    df = context.datasets.get_df(selection["dataset_id"])
    subset = df[df["id"].isin(selection["row_ids"])]

    return DatasetResult(
        dataset_id="selected_subset",
        dataframe=subset,
        name="Selected Subset",
    )
```

## Supported v0 action result types

Actions SHOULD return one of:

```text
DatasetResult
ArtifactResult
EventResult
ActionResult
None
```

## Result handling

The platform SHOULD:

```text
register DatasetResult outputs in context.datasets
store ArtifactResult outputs in context.artifacts
publish EventResult events through context.events
surface errors consistently
```

## Action rules

Actions MUST NOT directly mutate unrelated panels.

Actions SHOULD communicate through datasets, artifacts, events, selection, or workspace.

---

# 18. Dataframe action contract

A dataframe action is an action intended to operate on a dataset or selected dataframe subset.

Examples:

```text
create filtered subset
add derived column
preview transform
export selected rows
compute feature statistics
```

A dataframe action MUST declare whether it operates on:

```text
active dataset
focused row
active selection set
explicit dataframe input
```

A dataframe action that creates a durable table SHOULD return/register a dataset.

A dataframe action that creates a temporary preview SHOULD return/store an artifact.

---

# 19. Workflow contract

A workflow is an assembled set of panels, services, actions, and conventions.

Example:

```python
def build_active_learning_workspace(context):
    context.workspace.add_panel(...)
    context.workspace.add_panel(...)
    context.workspace.add_panel(...)
```

A workflow plugin MAY register:

```text
default panels
default layout
required services
required mappings
workflow actions
artifact conventions
event conventions
```

## Workflow examples

```text
active_learning
catalogue_review
annotation_review
model_audit
astro_exploration
```

## Workflow rule

A workflow MUST be assembled through platform services.

It MUST NOT become a hidden second application inside the plugin.

The plugin examples show active learning and non-ML review workflows as compositions of panels rather than built-in app identity. 

---

# 20. Artifact viewer contract

Plugins MAY register artifact viewers.

An artifact viewer declares which artifact types it can render.

Example:

```python
api.register_artifact_viewer(
    id="desi_spectrum_viewer",
    artifact_types=["spectra.desi"],
    factory=DESISpectrumViewer,
)
```

Viewer plugins SHOULD:

```text
retrieve artifact payloads through context.artifacts
avoid assuming local file paths unless declared
handle missing or stale artifacts gracefully
clean up watchers/subscriptions/jobs
```

---

# 21. Persistence contract

Persistent plugin panels SHOULD support state save/restore.

## Basic persistence

A panel marked persistent MAY be restored by workspace layout alone.

## Rich persistence

A controller/panel SHOULD implement:

```python
def get_state(self):
    return {
        "selected_tab": self.selected_tab,
        "display_mode": self.display_mode,
    }

def restore_state(self, state):
    self.selected_tab = state.get("selected_tab", "default")
    self.display_mode = state.get("display_mode", "summary")
```

## State requirements

Panel state MUST be JSON-safe or safely convertible.

Panel state MUST NOT include:

```text
raw dataframe objects
open file handles
API clients
thread objects
Bokeh document objects
large binary payloads
non-serializable model objects
```

Store those elsewhere:

```text
dataframe → datasets
derived output → artifacts
client/session → services
slow work → jobs
```

## Versioning

Persistent panels SHOULD declare a `state_version`.

A plugin that changes its state schema SHOULD handle old state gracefully.

---

# 22. Dependency contract

Plugins SHOULD declare their dependencies explicitly.

Core MUST NOT import plugin-only dependencies at startup.

Domain plugins MUST isolate domain-specific dependencies.

Examples:

```text
astro.desi may depend on sparclclient
astro.euclid may depend on astropy/reproject/mocpy
medical.imaging may depend on pydicom
geo.raster may depend on rasterio
```

A plugin that cannot load because dependencies are missing SHOULD fail gracefully.

The platform SHOULD surface a clear error:

```text
Plugin astro.desi is unavailable because dependency sparclclient is missing.
Install astronomical-astro[desi] to enable it.
```

---

# 23. Error handling contract

Plugins SHOULD fail locally, not destabilize the platform.

## Registration errors

If plugin registration fails, the plugin SHOULD be marked unavailable or errored.

Other plugins SHOULD continue loading.

## Panel errors

If panel creation fails, the workspace SHOULD show a clear error panel rather than crashing the whole app.

## Event subscriber errors

One failing event subscriber MUST NOT prevent other subscribers from receiving the event.

## Job errors

Job failures SHOULD be reported through job status and/or an event.

Recommended event:

```text
job.failed
plugin.error
```

---

# 24. Inter-plugin communication contract

Plugins SHOULD communicate through platform services, not direct references.

Allowed communication paths:

```text
selection state
events
artifacts
datasets
services
workspace actions
plugin manager metadata
```

Forbidden for new plugins:

```python
other_panel.update(...)
some_global.current_row = row_id
template.main.append(view)
context.config.latest_result = result
```

Preferred pattern:

```python
context.selection.set_focus(...)
```

or:

```python
artifact_id = context.artifacts.put(...)
context.events.publish("artifact.created", {"artifact_id": artifact_id})
```

---

# 25. Naming conventions

## Plugin ids

```text
core.record_browser
core.visualisation
core.annotations
core.table_tools
workflow.active_learning
astro.desi
astro.euclid
```

## Panel ids

Panel ids SHOULD be unique within the plugin.

Fully qualified panel ids SHOULD be formed by the platform:

```text
core.record_browser.browser
astro.desi.spectra
workflow.active_learning.query_queue
```

## Event topics

Use dotted names:

```text
selection.focus.changed
artifact.created
workflow.stage.changed
```

## Artifact types

Use dotted names:

```text
classifier.scores
table.filtered
spectra.desi
cutout.euclid
```

## Service keys

Use stable descriptive names:

```text
desi_client
euclid_client
database.primary
storage.cache
model_backend
```

The platform MAY namespace service keys by plugin id internally.

## Dataset ids

Use durable machine-readable ids:

```text
main_catalog
training_set
test_set
review_queue
high_score_subset
```

---

# 26. Built-in plugin categories

## Core utility plugins

These SHOULD remain domain-neutral.

Examples:

```text
record browser
visualisation
selection tools
table tools
event monitor
plugin manager
annotations
```

## Workflow plugins

These MAY depend on core utility plugins.

Examples:

```text
active learning
labelling
review queue
model audit
```

## Domain plugins

These SHOULD contain domain assumptions and domain dependencies.

Examples:

```text
astro.desi
astro.euclid
astro.sdss
astro.aladin
astro.sed
astro.radio
```

---

# 27. Adapting current AstronomicAL features

## Record browsing

Old exploration/source-browser behaviour SHOULD become a record browser plugin.

It SHOULD:

```text
read active dataset from context.datasets
publish focused row through context.selection
avoid knowing which detail panels exist
support dataset switching
use mappings for record_id
```

## Visualisation

Old BasicPlots/custom plotting SHOULD become visualisation plugins.

They SHOULD:

```text
read source data from datasets
publish focus on tap/click
publish selection set on lasso/brush
store derived plot products as artifacts when needed
keep plot configuration in panel state
```

## Selection Set panel

Selection-set UI SHOULD live in a selection tools plugin.

It SHOULD:

```text
read active selection from context.selection
materialize selection.ids artifacts when needed
avoid storing selected ids in config
```

## Table transforms

Table transformation tools SHOULD live in a table tools plugin.

They SHOULD:

```text
preview temporary outputs as artifacts
promote durable outputs to datasets
publish dataset/artifact events
```

## Annotations

Annotations SHOULD live in an annotations plugin.

They SHOULD:

```text
subscribe to focus changes
store annotation records as artifacts or datasets depending on durability
publish labels.updated or review.status.changed where appropriate
integrate with active-learning workflows through events/artifacts/datasets
```

## Active learning

Active learning SHOULD become a workflow plugin.

It SHOULD register:

```text
model controls
query queue
label controls
training/test-set panels
score/metric panels
model export actions
```

It SHOULD use:

```text
datasets       training_set, test_set, review_queue
artifacts      classifier.model, classifier.scores, active_learning.query_batch
events         labels.updated, workflow.stage.changed, artifact.created
jobs           training, inference, querying
services       model backend, feature backend
```

## Astronomy

Astronomy-specific tools SHOULD move into optional domain plugins.

Examples:

```text
astro.euclid.cutout
astro.desi.spectra
astro.sdss.spectra
astro.aladin.viewer
astro.sed.viewer
astro.radio.cutouts
```

They SHOULD:

```text
declare astronomy dependencies in their plugin manifests
use mappings for record_id and coordinates
fetch remote data through jobs
store cutouts/spectra/SEDs as artifacts
register clients through services
avoid importing astronomy dependencies into core
```

---

# 28. Anti-patterns

## 1. Recreating globals

Bad:

```python
context.services.set("shared_state", huge_mutable_blob)
```

Better:

```text
source data → datasets
derived data → artifacts
live clients → services
```

## 2. Using events as storage

Bad:

```python
context.events.publish("data.ready", huge_dataframe)
```

Better:

```python
artifact_id = context.artifacts.put("data.ready", payload)
context.events.publish("artifact.created", {"artifact_id": artifact_id})
```

## 3. Writing runtime state into config

Bad:

```python
context.config.current_row = row_id
context.config.latest_scores = scores
context.config.client = client
```

Better:

```text
current row → selection
scores → artifacts
client → services
```

## 4. Direct panel coupling

Bad:

```python
detail_panel.show_row(row_id)
plot_panel.highlight_row(row_id)
spectra_panel.load(row_id)
```

Better:

```python
context.selection.set_focus(
    dataset_id=context.datasets.active_id(),
    row_id=row_id,
    origin="record_browser"
)
```

## 5. Bypassing workspace

Bad:

```python
template.main.append(view)
```

Better:

```python
context.workspace.add_panel(...)
```

## 6. Blocking UI work

Bad:

```python
result = remote_fetch(row_id)
self.render(result)
```

Better:

```python
context.jobs.submit(...)
```

---

# 29. Minimum plugin quality checklist

A plugin is v0-compatible when:

```text
[ ] It exposes manifest and register(api)
[ ] It uses context instead of importing global runtime state
[ ] It declares required mappings instead of assuming fixed column names
[ ] It uses datasets for source data
[ ] It uses selection for focus and multi-row selection
[ ] It uses artifacts for derived outputs
[ ] It uses events for notifications
[ ] It uses jobs for slow/cancellable work
[ ] It uses services for live clients/connectors
[ ] It opens panels through the plugin/workspace system
[ ] It implements dispose() when it subscribes, watches, or starts jobs
[ ] It does not write new runtime state into context.config
[ ] It handles missing mappings gracefully
[ ] It handles missing optional dependencies gracefully
[ ] It returns JSON-safe state from get_state(), if persistent
[ ] It avoids domain-specific dependencies unless it is a domain plugin
```

---

# 30. Platform guarantees for v0 plugins

The AstronomicAL platform SHOULD guarantee:

```text
context services are available to registered plugins
plugin ids are namespaced and tracked
panel registrations are tracked by plugin
mapping requirements are checked before panel creation
workspace owns panel lifecycle
dispose() is called on panel removal where available
plugin-owned panels can be removed when plugin is disabled
events are delivered to subscribers without one failure stopping all
jobs support cancellation/deduplication where available
artifacts can be stored and retrieved by id
datasets can be registered and made active
workspace snapshots can include persistent plugin panels
```

---

# 31. Non-guarantees in v0

The following SHOULD be considered unstable unless explicitly documented elsewhere:

```text
private PluginManager methods
internal workspace record structure
exact implementation of the layout/grid backend
raw Panel/Bokeh internals
temporary context.config bridge
experimental bundled plugin ids
unfinished persistence schema internals
non-public helper modules
```

Plugins SHOULD depend on public registration APIs and `context` services, not internal implementation details.

---

# 32. Versioning and compatibility

## Contract version

This document defines:

```text
AstronomicAL Plugin Contract: v0
```

## Platform compatibility

A plugin MAY declare:

```python
manifest = {
    "id": "astro.desi",
    "version": "0.1.0",
    "astronomical_plugin_contract": "v0",
}
```

## Breaking changes

Breaking changes to this contract SHOULD require a new contract version:

```text
v0 → v1
```

Examples of breaking changes:

```text
renaming context services
changing register(api) entry point
removing supported contribution types
changing panel factory requirements
changing persistence hooks incompatibly
changing action result processing incompatibly
```

## Non-breaking changes

The platform MAY add:

```text
new optional manifest fields
new event topics
new artifact types
new contribution types
new optional metadata
new helper APIs
```

without breaking v0.

---

# 33. Test expectations

Each bundled plugin SHOULD have at least a smoke test covering:

```text
plugin discovery
manifest validation
registration
panel creation
required mapping behaviour
panel disposal
basic persistence, if persistent
```

Complex plugins SHOULD additionally test:

```text
job cancellation
artifact creation
dataset switching
selection focus updates
selection set updates
missing dependency behaviour
workspace save/load
disable/re-enable cleanup
```

Workflow plugins SHOULD test the full workflow path.

For active learning, that means:

```text
load dataset
map id/label/features
create training/test set
train model through jobs
store classifier artifacts
query next item
update label
refresh scores
save/reload workspace
```

For astronomy plugins, that means:

```text
load catalogue
map record_id/ra/dec
focus source
fetch remote/local cutout or spectrum through jobs
store artifact
render artifact viewer
close panel
disable plugin
confirm cleanup
```

---

# 34. Migration rule

When migrating old AstronomicAL code, contributors SHOULD use this sequence:

```text
1. Inject context
2. Move source dataframe access to datasets
3. Move current row/current subset to selection
4. Move derived products to artifacts
5. Replace direct panel calls with events
6. Move slow work to jobs
7. Move panel creation to workspace/plugin registrations
8. Keep only unavoidable compatibility reads in config
```

This sequence matches the migration guide and allows incremental conversion without rewriting everything at once. 

---

# 35. Reference plugin skeleton

```python
manifest = {
    "id": "example.detail",
    "name": "Example Detail Panel",
    "version": "0.1.0",
    "description": "Shows details for the currently focused row",
    "astronomical_plugin_contract": "v0",
}


def register(api):
    api.register_panel(
        id="detail",
        title="Detail",
        factory=DetailPanel,
        required_mappings=["record_id"],
        persistent=True,
        default_layout={
            "width": 4,
            "height": 4,
        },
    )


class DetailPanel:
    def __init__(self, context, **kwargs):
        self.context = context
        self.datasets = context.datasets
        self.selection = context.selection
        self.events = context.events
        self.subscriptions = []

        sub = self.events.subscribe(
            "selection.focus.changed",
            self.on_focus_changed
        )
        self.subscriptions.append(sub)

        current_focus = self.selection.get_focus()
        if current_focus:
            self.on_focus_changed("selection.focus.changed", current_focus)

    def on_focus_changed(self, topic, payload):
        dataset_id = payload["dataset_id"]
        row_id = payload["row_id"]

        df = self.datasets.get_df(dataset_id)

        record_col = self.datasets.resolve_mapping(
            dataset_id,
            "record_id"
        )

        row = df[df[record_col] == row_id]
        self.render(row)

    def render(self, row):
        # Build or update panel view here.
        pass

    def get_state(self):
        return {
            "version": 1,
        }

    def restore_state(self, state):
        pass

    def dispose(self):
        for sub in self.subscriptions:
            self.events.unsubscribe(sub)
        self.subscriptions.clear()
```

---

# 36. Reference slow-data plugin skeleton

```python
manifest = {
    "id": "astro.desi",
    "name": "DESI Tools",
    "version": "0.1.0",
    "description": "DESI spectra service and viewer",
    "astronomical_plugin_contract": "v0",
    "dependencies": ["sparclclient"],
}


def register(api):
    api.register_service(
        id="client",
        factory=create_desi_client,
        lazy=True,
    )

    api.register_panel(
        id="spectra",
        title="DESI Spectra",
        factory=DESISpectraPanel,
        required_mappings=["record_id"],
        optional_mappings=["coords.ra", "coords.dec"],
        persistent=True,
    )

    api.register_artifact_viewer(
        id="spectra_viewer",
        artifact_types=["spectra.desi"],
        factory=DESISpectraViewer,
    )


def create_desi_client(context):
    return DESIClient()


class DESISpectraPanel:
    def __init__(self, context, **kwargs):
        self.context = context
        self.subscriptions = []
        self.job_handles = []

        sub = context.events.subscribe(
            "selection.focus.changed",
            self.on_focus_changed
        )
        self.subscriptions.append(sub)

        current_focus = context.selection.get_focus()
        if current_focus:
            self.on_focus_changed("selection.focus.changed", current_focus)

    def on_focus_changed(self, topic, payload):
        dataset_id = payload["dataset_id"]
        row_id = payload["row_id"]

        handle = self.context.jobs.submit(
            self.fetch_spectrum,
            title="Fetch DESI spectrum",
            key=f"astro.desi.spectrum:{dataset_id}:{row_id}",
            on_done=lambda result: self.on_loaded(dataset_id, row_id, result),
            row_id=row_id,
        )
        self.job_handles.append(handle)

    def fetch_spectrum(self, *, cancel_token, row_id):
        if cancel_token and cancel_token.cancelled():
            return None

        client = self.context.services.get("astro.desi.client")
        result = client.fetch_spectrum(row_id)

        if cancel_token and cancel_token.cancelled():
            return None

        return result

    def on_loaded(self, dataset_id, row_id, result):
        if result is None:
            return

        artifact_id = self.context.artifacts.put(
            "spectra.desi",
            result,
            dataset_id=dataset_id,
            row_ids=[row_id],
        )

        self.context.events.publish(
            "artifact.created",
            {
                "artifact_id": artifact_id,
                "type": "spectra.desi",
                "dataset_id": dataset_id,
                "row_ids": [row_id],
            }
        )

    def dispose(self):
        for sub in self.subscriptions:
            self.context.events.unsubscribe(sub)
        self.subscriptions.clear()

        for handle in self.job_handles:
            handle.cancel()
        self.job_handles.clear()
```

---

# 37. Final contract statement

A v0-compatible AstronomicAL plugin:

```text
declares itself through a manifest
registers capabilities through the plugin API
uses context as the host API
keeps source data in datasets
keeps live focus and selected subsets in selection
keeps derived results in artifacts
uses events only for lightweight notifications
uses jobs for slow or cancellable work
uses services for live runtime integrations
lets workspace own panel lifecycle
declares semantic mappings instead of assuming column names
cleans up after itself
keeps domain assumptions out of core
```

That is the baseline needed for AstronomicAL to become a stable generic analysis platform rather than a monolithic astronomy-heavy application.
