# AstronomicAL Plugins

This directory contains bundled plugins for the new AstronomicAL platform.

Plugins are the preferred way to add new panels, actions, services, artifact viewers, and workflows without adding more domain-specific code to the core application.

The main goal is to keep the core platform small and generic, while allowing astronomy-specific, workflow-specific, and user-specific functionality to live in separate composable modules.

---

## Directory structure

There are two different plugin-related directories in the codebase, and they serve different purposes.

```text
astronomicAL/platform/plugins/
```

Contains the plugin framework itself:

- `PluginManager`
- `PluginAPI`
- plugin manifests
- registration specs
- plugin errors
- action, panel, service, artifact-viewer, and workflow registration types

Do not put normal plugins in this directory.

```text
astronomicAL/plugins/
```

Contains bundled plugins that ship with AstronomicAL.

Each plugin should live in its own folder:

```text
astronomicAL/plugins/
├── event_monitor/
│   └── plugin.py
├── table_tools/
│   └── plugin.py
└── my_plugin/
    └── plugin.py
```

Local user plugins can also be loaded from:

```text
~/.astronomical/plugins/
```

A development checkout may also scan:

```text
plugins/
```

at the repository root, depending on the configured plugin paths.

---

## Minimal plugin layout

A minimal plugin is a folder containing `plugin.py`:

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

When registered, the panel id becomes:

```text
example.hello.panel
```

because registration ids are automatically namespaced by the plugin id.

---

## What plugins can contribute

A plugin can register several kinds of contributions.

### Panels

Panels are visible UI components that can be added to the workspace or selected from the legacy Choose Plot menu.

```python
api.register_panel(
    id="summary",
    title="Summary Panel",
    factory=create_summary_panel,
    description="Display a summary of the active dataset.",
    category="Data",
)
```

Panel factories should usually return:

```python
return view, controller
```

where:

- `view` is a Panel object
- `controller` is an optional object with `dispose()`

If the panel subscribes to events, starts jobs, or creates watchers, the controller should implement `dispose()`.

---

### Actions

Actions are callable operations registered by plugins.

They can be used by panels, workflows, buttons, menus, or future automation layers.

```python
api.register_action(
    id="add_column",
    title="Add derived column",
    handler=add_column_action,
    run_in_job=False,
)
```

The handler can receive `context`, `request`, `manager`, and optionally `cancel_token`, depending on what arguments it declares.

---

### DataFrame actions

For common table operations, plugins can register dataframe-oriented actions:

```python
api.register_dataframe_action(
    id="profile_numeric",
    title="Profile numeric columns",
    handler=profile_numeric_columns,
    output_type="table.numeric_profile",
    selection="optional",
    numeric_columns="many",
    columns="many",
)
```

The platform resolves the active dataframe, selected rows, columns, and parameters before calling the handler.

---

### Services

Services are shared live runtime objects.

Use services for things like:

- API clients
- authenticated sessions
- database connections
- filesystem adapters
- external analysis backends

```python
api.register_service(
    key="client",
    factory=create_client,
    lazy=True,
)
```

A plugin service key is automatically namespaced:

```text
my_plugin.client
```

Services are not for storing data products. Derived results should go into artifacts.

---

### Artifact viewers

Artifact viewers display derived results created by actions or panels.

```python
api.register_artifact_viewer(
    artifact_type="table.filtered",
    viewer_factory=create_table_viewer,
    title="Filtered Table Viewer",
    default=True,
)
```

Artifact viewers allow one plugin to produce a result and another panel or workflow to display it.

---

### Workflows

Workflows assemble panels, services, and actions into a reusable workspace.

```python
api.register_workflow(
    id="review",
    title="Review Workflow",
    builder=build_review_workspace,
)
```

A workflow should use `context.workspace.add_panel(...)` rather than directly mutating the template or grid.

---

## The `context` object

Plugin factories and handlers receive the platform `context`.

This is the main host API for plugins.

```python
context.datasets
context.selection
context.events
context.artifacts
context.jobs
context.workspace
context.services
context.plugins
context.config
```

Use the named platform services rather than importing global state.

---

## Where state belongs

Use this guide when deciding where to put data or state:

| Need | Use |
|---|---|
| Source dataframe or loaded table | `context.datasets` |
| Current focused row | `context.selection` |
| Active multi-row selection | `context.selection` |
| Derived result or cached output | `context.artifacts` |
| Notification that something changed | `context.events` |
| Slow or cancellable work | `context.jobs` |
| Visible panels and layout | `context.workspace` |
| Live API client, session, or connector | `context.services` |
| Temporary legacy compatibility | `context.config` |

Avoid putting new runtime state into `context.config` unless there is no migrated platform service for it yet.

---

## Recommended event pattern

Events should be lightweight notifications.

Do not put large dataframes, images, spectra, or model outputs directly on the event bus.

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

Use artifacts for reusable results. Use events to announce that those results exist.

---

## Selection pattern

A table, plot, or browser that changes the focused row should call:

```python
context.selection.set_focus(
    dataset_id=context.datasets.active_id(),
    row_id=row_id,
    origin="my_plugin.table",
)
```

Panels that care about the focused row should subscribe to:

```text
selection.focus.changed
```

Example:

```python
class DetailPanel:
    def __init__(self, context):
        self.context = context
        self.sub = context.events.subscribe(
            "selection.focus.changed",
            self.on_focus_changed,
        )

        current_focus = context.selection.get_focus()
        if current_focus:
            self.on_focus_changed("selection.focus.changed", current_focus)

    def on_focus_changed(self, topic, payload):
        dataset_id = payload["dataset_id"]
        row_id = payload["row_id"]

        df = self.context.datasets.get_df(dataset_id)
        row = df[df["id"] == row_id]
        self.render(row)

    def dispose(self):
        self.context.events.unsubscribe(self.sub)
```

A lasso, filter, or batch-selection tool should use selection sets:

```python
context.selection.set_selection_set(
    dataset_id=context.datasets.active_id(),
    row_ids=row_ids,
    origin="my_plugin.lasso",
)
```

---

## Job pattern

Use `context.jobs` for slow work.

This keeps the UI responsive and allows cancellation or deduplication.

```python
def fetch_remote_data(*, cancel_token, row_id):
    if cancel_token and cancel_token.cancelled():
        return None

    client = context.services.require("my_plugin.client")
    result = client.fetch(row_id)

    if cancel_token and cancel_token.cancelled():
        return None

    return result


context.jobs.submit(
    fetch_remote_data,
    title="Fetch remote data",
    key=f"remote:{row_id}",
    on_done=handle_result,
    row_id=row_id,
)
```

Store reusable results in `context.artifacts` and publish an event after the job completes.

---

## Lifecycle rules

Plugin panels should clean up after themselves.

If a panel subscribes to events:

```python
self.sub = context.events.subscribe("selection.focus.changed", self.on_focus)
```

then it should unsubscribe:

```python
def dispose(self):
    context.events.unsubscribe(self.sub)
```

If a panel starts jobs, cancel them on disposal where appropriate.

If a panel creates periodic callbacks or widget watchers, stop or unwatch them in `dispose()`.

A good controller shape is:

```python
class MyPanel:
    def __init__(self, context):
        self.context = context
        self.subscriptions = []
        self.job_handles = []
        self.watchers = []

    def dispose(self):
        for sub in self.subscriptions:
            self.context.events.unsubscribe(sub)
        self.subscriptions.clear()

        for handle in self.job_handles:
            handle.cancel()
        self.job_handles.clear()

        for widget, watcher in self.watchers:
            widget.param.unwatch(watcher)
        self.watchers.clear()
```

---

## Legacy Choose Plot bridge

During migration, plugin panels can appear in the existing Choose Plot menu.

The bridge is handled through:

```text
astronomicAL/extensions/custom_plots.py
```

`get_customplot_dict(context=...)` includes both:

- legacy custom plot entries
- plugin panel registrations from `context.plugins.list_panels()`

New panels should not be added directly to `get_customplot_dict` unless they are temporary legacy code.

Prefer creating a plugin under this directory.

---

## Bundled plugins

### `event_monitor`

Location:

```text
astronomicAL/plugins/event_monitor/plugin.py
```

Provides:

- Event Monitor panel
- EventBus subscription diagnostics
- recent event trace
- dataset-event coverage table
- owner/topic graph
- periodic refresh
- plugin lifecycle cleanup

Plugin id:

```text
core.event_monitor
```

Registered panel:

```text
core.event_monitor.panel
```

---

### `table_tools`

Location:

```text
astronomicAL/plugins/table_tools/plugin.py
```

Provides:

- Table Transform panel
- derived-column preview
- derived-column creation
- boolean subset preview
- subset dataset creation
- `dataset.updated`
- `dataset.loaded`
- `dataset.active.changed`

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

---

## Naming conventions

Use stable, namespaced names.

### Plugin ids

Good:

```text
core.table_tools
astro.euclid
astro.desi
samp.integration
example.demo
```

Avoid:

```text
test
plugin
my stuff
```

### Event topics

Use dotted topic names:

```text
dataset.loaded
dataset.active.changed
dataset.updated
dataset.mapping_updated
selection.focus.changed
selection.set.changed
artifact.created
plugin.enabled
plugin.disabled
workflow.stage.changed
```

### Artifact types

Use dotted artifact types:

```text
classifier.scores
classifier.model
table.filtered
table.numeric_profile
spectra.desi
cutout.euclid
report.summary
```

### Service keys

Use stable descriptive names:

```text
astro.euclid.client
astro.desi.client
samp.client
database.primary
storage.cache
```

The plugin API automatically namespaces service keys when needed.

---

## Dependency guidance

A plugin should declare required dependencies in its manifest:

```python
manifest = PluginManifest(
    id="astro.euclid",
    name="Euclid Tools",
    version="0.1.0",
    requires=[
        "astropy",
        "astroquery",
        "reproject",
    ],
)
```

Keep heavyweight, domain-specific dependencies out of the core platform where possible.

For example, astronomy-specific tools should live in an astronomy plugin bundle rather than being imported by the base application.

---

## Development checklist

Before committing a plugin, check:

- [ ] The plugin lives outside `astronomicAL/platform/plugins/`.
- [ ] `plugin.py` exposes `manifest`.
- [ ] `plugin.py` defines `register(api)`.
- [ ] Panel factories accept `context`.
- [ ] Panels return `(view, controller)` where possible.
- [ ] Controllers clean up subscriptions, watchers, callbacks, and jobs.
- [ ] Source data is read from `context.datasets`.
- [ ] Focus and multi-row selections use `context.selection`.
- [ ] Derived outputs use `context.artifacts`.
- [ ] Notifications use `context.events`.
- [ ] Slow work uses `context.jobs`.
- [ ] Runtime clients use `context.services`.
- [ ] New runtime state is not added to `context.config` unless unavoidable.
- [ ] The plugin appears in the Choose Plot menu if it registers a panel.
- [ ] The plugin can be enabled without breaking app startup.

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

A healthy startup should show:

```text
plugins discovered
plugins enabled
plugin panels before layout
custom_plots sees plugin panels
MenuDashboard sees plugin panels
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

---

### Passing close buttons into plugin panels

Plugin panels should not render the legacy close button themselves.

The legacy dashboard shell owns the close button. Plugin panels should render only their own content.

---

### Storing data products in services

Wrong:

```python
context.services.set("latest_scores", scores)
```

Right:

```python
context.artifacts.put("classifier.scores", scores, dataset_id=...)
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

## Future direction

The plugin system is intended to make AstronomicAL a generic analysis host.

The long-term direction is:

```text
small generic core
+ optional bundled plugins
+ optional domain plugins
+ optional user plugins
```

Astronomy-specific tools such as Euclid cutouts, DESI spectra, SED panels, and Aladin views should eventually live in astronomy plugin bundles rather than the core application.

Active learning should also become a workflow or plugin bundle rather than the only built-in identity of the app.

This allows AstronomicAL to support astronomy workflows while remaining usable by non-astronomy domain experts.