# AstronomicAL

## A plugin-based platform for interactive visualisation, inspection, labelling, and classification of scientific data

[![Documentation Status](https://readthedocs.org/projects/astronomical/badge/?version=latest)](https://astronomical.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03635/status.svg)](https://doi.org/10.21105/joss.03635)

> [!WARNING]
> This branch is under active development and the codebase is changing frequently. Some parts of the platform are still incomplete, unstable, or not yet fully functional.


AstronomicAL is a local, human-in-the-loop analysis platform for working with tabular scientific datasets.

It helps researchers inspect records, combine contextual information, curate reliable labels, create review workflows, and build machine-learning workflows such as active learning. AstronomicAL was originally developed and validated in astronomy, but the platform is being redesigned so that astronomy is one optional domain bundle rather than the built-in identity of the whole application.

The long-term goal is:

```text
small generic platform
+ generic Core plugins
+ optional workflow plugins
+ optional domain plugins
+ optional user plugins
+ optional third-party plugins
```

This lets AstronomicAL support astronomy workflows while remaining useful for other domains that need interactive data inspection, expert review, labelling, triage, quality control, or model-guided analysis.

---

## What AstronomicAL is for

AstronomicAL is designed for situations where:

* datasets are too large to inspect manually record by record
* labels are missing, noisy, inconsistent, or expensive to create
* expert judgement is needed to decide difficult or ambiguous cases
* users need to inspect each record with multiple sources of context
* workflows need to combine tables, plots, annotations, external services, derived products, and models
* active learning or other model-assisted workflows can reduce manual labelling effort
* researchers need a local, reproducible, configurable workspace

Although astronomy motivated the original project, the underlying problem is common across many domains: expert-labelled data is valuable, but expensive, and users often need rich context before assigning or correcting a label.

---

## Statement of need

Modern scientific and industrial datasets are increasingly large, heterogeneous, and difficult to label reliably. Supervised machine-learning systems depend on the quality of their training labels, but manual labelling can be slow, expensive, inconsistent, or impossible to complete exhaustively.

Active learning helps address this by focusing expert attention on informative examples, often near class boundaries or regions of model uncertainty. However, active learning is only useful in practice when experts can inspect each queried item in enough context to make a reliable decision.

AstronomicAL provides an interactive environment for that human-in-the-loop process.

It allows users to:

* browse and inspect records from a dataset
* visualise tabular features
* select individual records or groups of records
* add labels, notes, review status, and annotations
* create derived datasets and filtered subsets
* run slow or remote operations without blocking the interface
* store reusable derived outputs as artifacts
* build workflows from composable panels and plugins
* integrate domain-specific tools without bloating the platform core

The original AstronomicAL application focused heavily on active learning for astronomical datasets. The current platform direction keeps active learning as an important workflow, but generalises the software into a plugin-based analysis host.

---

## Key capabilities

AstronomicAL provides a generic platform for building interactive analysis workspaces.

Current and planned capabilities include:

* **Dataset loading and switching**
  * Work with tabular datasets through a central dataset service.
  * Track the active dataset and associated metadata.

* **Column mapping**
  * Map semantic requirements such as `record_id`, `target_label`, or domain-specific fields to actual dataset columns.
  * Allow plugins to declare what columns they need without hard-coding column names.

* **Record browsing**
  * Browse records in the active dataset.
  * Search by record identifier.
  * Publish focused-row changes to the rest of the workspace.

* **Visualisation**
  * Explore tabular data with generic visualisation panels.
  * Use scatter, histogram, density, and linked explorer views.
  * Coordinate plots with the platform selection state.

* **Selection tools**
  * Track one focused row separately from multi-row selection sets.
  * Inspect selected subsets.
  * Create derived datasets from selected rows.

* **Annotations and review**
  * Attach notes, review status, and label suggestions to records.
  * Build review and curation workflows.
  * Store annotation outputs as reusable platform data.

* **Artifacts**
  * Store derived outputs such as model scores, filtered tables, reports, spectra, cutouts, embeddings, or other computed products.
  * Let one plugin produce an output and another plugin display or reuse it.

* **Background jobs**
  * Run slow work such as remote fetches, model training, feature generation, exports, or large transforms without blocking the interface.
  * Support cancellation and deduplication where possible.

* **Plugin-based extension**
  * Add panels, actions, services, artifact viewers, and workflows through plugins.
  * Keep optional and domain-specific code outside the platform core.

* **Workspace persistence**
  * Save and restore plugin-based layouts and panel state where supported.

* **Active learning workflows**
  * Active learning remains a major use case.
  * The future direction is for active learning to be implemented as a workflow/plugin bundle that can be combined with domain-specific inspection tools.

---

## Architecture overview

AstronomicAL is being organised around a small platform layer and a plugin system.

The platform exposes shared runtime services through an `AppContext` object.

```text
context.datasets      canonical source data
context.selection     focused row and active multi-row selection sets
context.events        lightweight notifications
context.artifacts     reusable derived outputs
context.jobs          slow or cancellable background work
context.workspace     visible panel lifecycle and layout
context.services      live runtime clients, sessions, and connectors
context.plugins       plugin discovery, registration, and execution
context.persistence   workspace save/load behaviour
context.config        temporary compatibility bridge for older code
```

A useful rule of thumb is:

```text
Datasets store source data.
Selection stores current focus and selected row sets.
Artifacts store derived data.
Events announce changes.
Jobs run slow work.
Workspace owns visible panels.
Services hold live capabilities.
Plugins contribute behaviour.
Persistence saves and restores workspace state.
Config is only a temporary compatibility bridge.
```

This structure is intended to make the core smaller, more maintainable, and less domain-specific.

---

## Why plugins?

Earlier versions of AstronomicAL were designed to be domain-flexible, but the main codebase gradually accumulated more astronomy-specific panels, services, and dependencies.

Those astronomy features are valuable, but not every user needs them. A non-astronomy user should not need to install astronomy-specific clients, cutout tools, spectra tools, or survey integrations just to use the platform.

The plugin system solves this by separating:

```text
generic platform functionality
from
workflow-specific functionality
from
domain-specific functionality
```

For example:

```text
Core plugins
    Record browser, visualisation, table tools, selection tools, annotations,
    event monitor, plugin manager.

Workflow plugins
    Active learning, review queues, model audit, quality control, reporting.

Domain plugins
    Astronomy cutouts, spectra, SEDs, Aladin views, medical scan viewers,
    manufacturing sensor tools, ecology media review, materials microscopy,
    finance transaction inspection.

User plugins
    Local project-specific tools and panels.
```

The platform should not need to know what a domain-specific artifact means. It should only provide the generic services that allow plugins to coordinate safely.

---

## Core plugins

Core plugins ship with the main AstronomicAL repository. They are still real plugins and should follow the same rules as external plugins.

Current Core plugins include:

```text
core.record_browser
core.visualisation
core.selection_tools
core.table_tools
core.annotations
core.event_monitor
core.plugin_manager
```

### `core.record_browser`

Provides generic active-dataset record browsing.

It is responsible for:

* browsing rows in the active dataset
* searching by record id
* showing record metadata
* publishing focused-row changes through `context.selection`
* consuming platform selection events

It does not own dataset loading, column mapping, or domain-specific inspection.

### `core.visualisation`

Provides generic visualisation panels for tabular datasets.

It includes:

* scatter plots
* histograms
* density plots
* linked explorer panels
* focus publication from plot interactions
* multi-row selection publication from plot selections

Domain-specific plots should live in domain plugins.

### `core.selection_tools`

Provides selection-set inspection and selected-subset workflows.

It includes:

* focused-row display
* active selection-set display
* selected-row preview
* clearing selection state
* stepping through selected rows
* creating derived datasets from selected rows

### `core.table_tools`

Provides table transformation and subset creation tools.

It includes:

* expression preview
* derived-column creation
* boolean subset preview
* subset dataset creation
* dataset update events

### `core.annotations`

Provides record-level annotation and review support.

It can support:

* notes
* review status
* label suggestions
* annotation summaries
* review workflow integration
* future active-learning integration

### `core.event_monitor`

Provides EventBus diagnostics and event observability.

It is useful for debugging:

* event topics
* event payloads
* subscribers
* owner metadata
* stale listeners
* plugin communication

### `core.plugin_manager`

Provides plugin inspection and management UI.

It is useful for checking:

* discovered plugins
* enabled plugins
* registered panels
* registered actions
* registered services
* registered workflows
* open plugin panels

---

## Example workflows

AstronomicAL is intended to support many workflows, not just one application mode.

### Expert review

```text
load dataset
map record id and optional label fields
open record browser
open visualisation panel
open annotation panel
review records
store notes, review status, and label suggestions
export or materialise reviewed subset
```

### Active learning

```text
load dataset
map record id, labels, and model features
open active-learning workflow
train model through background jobs
query uncertain or informative records
inspect each queried record with domain-specific panels
assign or correct labels
store model scores and metrics as artifacts
repeat
```

### Domain-specific inspection

```text
load dataset
map domain-specific identifiers
open generic browsing and visualisation panels
open domain plugin panels
fetch external context through jobs
store fetched products as artifacts
review, annotate, label, or export results
```

For astronomy, this might include spectra, cutouts, SEDs, and external survey services.

For other domains, the same pattern could support scans, images, sensor traces, documents, assays, transactions, or other contextual evidence.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/grant-m-s/AstronomicAL.git
cd AstronomicAL
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, using Conda:

```bash
conda create -n astronomical python
conda activate astronomical
pip install -r requirements.txt
```

The current requirements are still being refined as domain-specific functionality is moved out of the platform core. In the long term, heavyweight or domain-specific dependencies should belong to optional plugin bundles rather than the base install.

---

## Running AstronomicAL

Start the application with:

```bash
panel serve astronomicAL --show
```

This should open AstronomicAL in your browser.

If it does not open automatically, visit the local URL printed by Panel in the terminal.

---

## Debugging startup

Plugin and boot debug logs are normally quiet.

To enable boot-order diagnostics:

```bash
ASTRONOMICAL_DEBUG_BOOT=1 panel serve astronomicAL --show
```

To enable plugin diagnostics:

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

Useful diagnostic plugins include:

```text
core.event_monitor
core.plugin_manager
```

---

## Documentation

The main documentation is available at:

```text
https://astronomical.readthedocs.io
```

Plugin-system documentation is being expanded as the platform stabilises.

Important project documents include:

```text
PLUGIN_README.md
PLUGIN_CONTRACT.md
docs/source/
```

`PLUGIN_README.md` explains how to write plugins.

`PLUGIN_CONTRACT.md`, when present, defines the supported baseline for plugin authors and contributors.

---

## Developing plugins

A minimal plugin lives in its own folder and exposes a `plugin.py` file.

```text
astronomicAL/plugins/example_plugin/
└── plugin.py
```

A plugin should expose:

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

Plugin registration ids are automatically namespaced by the plugin id.

For a full guide, see:

```text
PLUGIN_README.md
```

---

## Plugin design principles

When writing plugins, follow these rules.

### Treat `context` as the host API

Use platform services instead of importing global runtime state.

### Keep source data and derived data separate

Use:

```text
context.datasets   for source or promoted working datasets
context.artifacts  for reusable derived outputs
```

### Use selection for focus and selected subsets

Use:

```text
context.selection.set_focus(...)
context.selection.set_selection_set(...)
```

Do not hide shared selection state in widgets, config, or panel-local variables.

### Use events for notifications

Events should announce that something happened.

They should not carry huge payloads.

### Use artifacts for reusable outputs

Model scores, filtered tables, reports, remote payloads, cutouts, spectra, and other computed products should become artifacts.

### Use jobs for slow work

Remote fetches, training, inference, exports, and large transforms should not block the UI.

### Use services for live clients

API clients, sessions, database handles, and runtime backends belong in `context.services`.

### Use workspace for panel lifecycle

Panels should be opened, tracked, restored, and disposed by the workspace/plugin system.

### Clean up

Panels that subscribe to events, start jobs, create watchers, schedule callbacks, or own child controllers must clean up in `dispose()`.

`dispose()` should be idempotent.

---

## Plugin quality checklist

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

## Migration guide for older AstronomicAL code

Older AstronomicAL code often used global config, shared mutable state, direct panel coupling, and ad hoc threads.

The migration direction is:

| Older pattern | New platform pattern |
| --- | --- |
| Global dataframe in config/shared module | `context.datasets` |
| Current focused row in config or panel state | `context.selection.set_focus(...)` |
| Multi-row selected ids in ad hoc variables | `context.selection.set_selection_set(...)` |
| Shared computed state | `context.artifacts` |
| One panel directly calling another | `context.selection` plus `context.events` |
| Ad hoc background thread | `context.jobs` |
| Direct template/grid mutation | `context.workspace` |
| Global API client | `context.services` |
| Hard-coded column names | `required_mappings` or `optional_mappings` |
| Monolithic mode-specific workflow | plugin workflow |
| Domain-specific panel in core | domain plugin |
| Temporary compatibility | `context.config` |

Recommended migration order:

1. Inject `context`.
2. Move source dataframe access to `context.datasets`.
3. Move focused row and selected rows to `context.selection`.
4. Replace direct panel calls with events.
5. Move derived results to `context.artifacts`.
6. Move slow work to `context.jobs`.
7. Register clients and connectors as `context.services`.
8. Add `required_mappings` and `optional_mappings`.
9. Register the feature as a panel, action, service, artifact viewer, or workflow.
10. Remove new writes to `context.config` unless strictly needed for compatibility.

---

## Active learning direction

Active learning remains one of AstronomicAL's most important use cases.

The future direction is for active learning to be implemented as a reusable workflow or plugin bundle rather than the built-in identity of the entire application.

A clean active-learning plugin should be able to use:

| Capability | Platform shape |
| --- | --- |
| classifier setup UI | panel |
| train model | action or job-backed action |
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

Domain-specific inspection tools should be separate plugins that can be opened alongside the active-learning workflow.

For example:

* astronomy active learning may use cutout and spectra plugins
* medical active learning may use scan and patient-summary plugins
* manufacturing active learning may use defect-image and sensor-trace plugins
* ecology active learning may use image, audio, or observation plugins

The active-learning workflow should not need to know which domain-specific panels are open. It should publish selections, artifacts, and events that other plugins can react to.

---

## Domain plugin direction

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

Examples:

```text
astro.euclid
astro.desi
astro.sdss
astro.aladin
medical.scan_viewer
medical.patient_review
factory.sensor_review
ecology.audio_review
materials.microscopy
finance.transaction_review
```

A domain plugin should declare its own optional dependencies.

For example:

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

## Testing

Run tests with:

```bash
pytest
```

Plugin-related tests should cover:

* plugin discovery
* manifest validation
* registration
* panel creation
* required mapping behaviour
* panel disposal
* job cancellation where relevant
* artifact creation where relevant
* dataset switching where relevant
* workspace save/load where relevant
* disable/re-enable cleanup

Lifecycle tests are especially important. A plugin should not leave behind stale subscriptions, orphan jobs, widget watchers, callbacks, or panels after disposal.

---

## Contributing

Contributions are welcome.

Good areas for contribution include:

* improving the platform services
* migrating legacy features into plugins
* improving workspace persistence
* improving plugin lifecycle tests
* writing new Core plugins
* building optional domain plugins
* building workflow plugins such as active learning
* improving documentation and examples
* reporting bugs and usability issues

When contributing, please keep the main design principle in mind:

```text
The platform should stay small and generic.
Domain and workflow assumptions should live in plugins.
```

### Reporting bugs

Please report bugs using the GitHub issues page.

Include:

* steps to reproduce
* expected behaviour
* actual behaviour
* relevant traceback or logs
* dataset/config details where possible
* plugin list if the issue involves plugins

### Submitting plugins

If you have created a plugin that may be useful to others, open a pull request or issue describing:

* what the plugin does
* what platform services it uses
* what mappings it requires
* what dependencies it adds
* what artifacts, events, services, actions, or panels it contributes
* whether it is generic, workflow-specific, or domain-specific

---

## Project history

AstronomicAL was originally introduced as an interactive environment for visualisation, integration, labelling, and classification of scientific data with active learning.

The original project was developed and validated using astronomy datasets, where large survey catalogues, imbalanced classes, ambiguous class definitions, missing labels, and noisy ground truth are common.

Those problems remain central to AstronomicAL's motivation.

The current platform direction generalises the architecture so that astronomy-specific functionality can continue to grow while the platform remains usable by other domains.

---

## Referencing AstronomicAL

Please cite AstronomicAL when it supports your research or software work.

The original software paper is available through JOSS:

```text
https://doi.org/10.21105/joss.03635
```

The documentation also includes citation guidance:

```text
https://astronomical.readthedocs.io/en/latest/content/other/citing.html
```

---

## License

See the repository license file for licensing information.

---

## Summary

AstronomicAL is becoming a generic, plugin-based analysis platform.

It is designed to support:

* interactive data exploration
* expert review
* annotation and labelling
* active learning
* model-assisted workflows
* domain-specific inspection tools
* reusable analysis workspaces

The platform provides the shared runtime services. Plugins provide the behaviour.

This keeps AstronomicAL useful for astronomy while allowing it to grow into a broader tool for many domains that need human-in-the-loop analysis of structured data.