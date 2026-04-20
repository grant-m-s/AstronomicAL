# AstronomicAL (`context-core-api`)

> [!WARNING]
> This branch is under active development and the codebase is changing frequently. Some parts of the platform are still incomplete, unstable, or not yet fully functional.
>
> For now, if you are testing panels or exploring the current architecture, use **Exploration Mode**. Other modes are still undergoing significant work and should not yet be considered reliable.

AstronomicAL is evolving from a domain-flexible but astronomy-heavy application into a smaller, generic platform for building interactive, panel-based analysis workspaces.

The `context-core-api` branch introduces a platform layer built around shared runtime services such as datasets, selection state, events, artifacts, jobs, workspace management, and runtime services. The aim is to keep the core small and generic while allowing astronomy-specific and other domain-specific functionality to be built as composable panels and plugins.

This branch is intended for contributors and early adopters exploring the new architecture. Migration from older config-driven patterns is still in progress, and `context.config` remains as a temporary bridge for legacy code. New code should prefer the newer platform services. 

## Why this branch exists

AstronomicAL originally aimed to support interactive analysis of tabular scientific data in a domain-agnostic way. Over time, more astronomy-specific capabilities and workflow assumptions accumulated in the main application, making the project heavier and less clearly generic.

This branch addresses that by moving shared runtime behavior into a small platform core and pushing domain-specific and workflow-specific logic outward into composable modules. The long-term goal is for astronomy to be one plugin/workflow bundle among many, rather than the built-in identity of the whole application. 

## What this branch is

This branch is:

- a platform refactor in progress
- a host for building analysis workspaces out of panels and services
- useful for contributors building new panels or migrating old ones

This branch is not:

- a finished replacement for the original application
- only an astronomy dashboard
- only an active-learning tool

Active learning remains an important use case, but the new architecture is intended to support many kinds of workflows, including non-ML workflows. 

## Core concepts

The platform is organized around a small set of services exposed through `AppContext`:

- `context` — host dependency access
- `datasets` — canonical source data
- `selection` — current focus and active multi-row selection sets
- `events` — decoupled notifications
- `artifacts` — derived outputs and reusable intermediate results
- `jobs` — slow or cancellable background work
- `workspace` — panel lifecycle and layout management
- `services` — shared live runtime capabilities
- `config` — temporary migration bridge for older code 

A good rule of thumb is:

- source data goes in `datasets`
- current user focus goes in `selection`
- computed results go in `artifacts`
- notifications go through `events`
- slow work goes through `jobs`
- visible panels are managed by `workspace`
- API clients and similar live objects go in `services` 

## Install

    git clone https://github.com/grant-m-s/AstronomicAL.git
    cd AstronomicAL
    git checkout context-core-api
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Run

    panel serve astronomicAL --show

This branch is transitional, so some older config-driven paths may still exist alongside the newer platform services. The recommended direction for new work is the `context`-based model described below. 

## Build your own panel

New panels should treat `context` as the host API.

Minimal example:

    class DetailPanel:
        def __init__(self, context):
            self.context = context
            self.events = context.events
            self.datasets = context.datasets
            self.selection = context.selection

            self.sub = self.events.subscribe(
                "selection.focus.changed",
                self.on_selection_focus_changed
            )

            current_focus = self.selection.get_focus()
            if current_focus:
                self.on_selection_focus_changed("selection.focus.changed", current_focus)

        def on_selection_focus_changed(self, topic, payload):
            dataset_id = payload["dataset_id"]
            row_id = payload["row_id"]

            df = self.datasets.get_df(dataset_id)
            row = df[df["id"] == row_id]
            self.render(row)

        def render(self, row):
            print("Render row:", row)

        def dispose(self):
            self.events.unsubscribe(self.sub)

This is the recommended pattern:

- read source data from `datasets`
- read and publish selection through `selection`
- react through `events`
- use `jobs` for slow work
- store reusable outputs in `artifacts`
- open and close panels through `workspace`
- clean up subscriptions and jobs in `dispose()` 

### Publish focus from a table

    class CatalogTable:
        def __init__(self, context):
            self.context = context

        def on_row_clicked(self, row_id):
            self.context.selection.set_focus(
                dataset_id=self.context.datasets.active_id(),
                row_id=row_id,
                origin="catalog_table"
            )

### Run slow work through `jobs`

    class SpectrumPanel:
        def __init__(self, context):
            self.context = context
            self.sub = context.events.subscribe(
                "selection.focus.changed",
                self.on_selection_focus_changed
            )

        def on_selection_focus_changed(self, topic, payload):
            dataset_id = payload["dataset_id"]
            row_id = payload["row_id"]

            self.context.jobs.submit(
                self.fetch_spectrum,
                title="Fetch spectrum",
                key=f"spectrum:{dataset_id}:{row_id}",
                on_done=lambda result: self.on_loaded(dataset_id, row_id, result),
                row_id=row_id,
            )

        def fetch_spectrum(self, *, cancel_token, row_id):
            if cancel_token and cancel_token.cancelled():
                return None
            client = self.context.services.get("desi_client")
            return client.fetch_spectrum(row_id)

        def on_loaded(self, dataset_id, row_id, result):
            if result is None:
                return

            artifact_id = self.context.artifacts.put(
                "spectra.desi",
                result,
                dataset_id=dataset_id,
                row_ids=[row_id]
            )

            self.context.events.publish(
                "artifact.created",
                {
                    "artifact_id": artifact_id,
                    "type": "spectra.desi"
                }
            )

The common pattern is:

1. react to selection or another event
2. run slow work through `jobs`
3. store the result in `artifacts`
4. publish a lightweight event announcing it 

## Migration notes

Older AstronomicAL code often relied on global config, shared mutable state, direct panel coupling, and ad hoc threads.

The migration direction for this branch is:

| Older pattern | New direction |
|---|---|
| Global dataframe in config/shared module | `context.datasets` |
| Current focused row or selected subset | `context.selection` |
| Shared computed state | `context.artifacts` |
| Panel calling another panel directly | `context.events` |
| Ad hoc background thread | `context.jobs` |
| Direct layout mutation | `context.workspace` |
| Global client or connector | `context.services` |
| Temporary compatibility | `context.config` | :contentReference[oaicite:9]{index=9}

For new contributions, prefer the platform model over extending older global/config-based patterns. 

## Contributor guidance

When building new features:

- keep the core small and generic
- keep domain-specific logic out of the core where possible
- prefer events over direct panel references
- use artifacts for derived outputs instead of shared mutable fields
- use jobs for slow work
- always clean up subscriptions, watchers, and jobs 

## Further reading

The branch direction is explained in more detail in the platform documents:

- plugin and panel examples
- platform overview
- migration guide 

## Background

The original AstronomicAL project was introduced as an interactive environment for visualization, integration, and classification of scientific data with active learning. That remains an important part of the project history and a major use case, but the `context-core-api` branch is focused on a broader platform architecture.