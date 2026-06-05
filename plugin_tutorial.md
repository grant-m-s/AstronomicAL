# Building Your First AstronomicAL Plugin

This tutorial is a progressive plugin walkthrough for the `plugin_system` architecture. It starts with the smallest possible plugin and gradually adds the platform features that real AstronomicAL plugins use: panels, datasets, events, selection, artifacts, actions, jobs, services, mapping requirements, workspace state, and workflows.

The goal is not to build a polished science plugin. The goal is to show where each kind of functionality belongs, so new contributors know where to start and how the pieces fit together.

---

## Mental model

AstronomicAL is becoming a plugin host. Plugins contribute behaviour; the platform owns shared runtime infrastructure.

Use this rule of thumb:

| Need | Use |
| --- | --- |
| Source tables or working datasets | `context.datasets` |
| Current focused row | `context.selection` |
| Multi-row selected set | `context.selection` |
| Derived results, computed outputs, fetched products | `context.artifacts` |
| Lightweight notifications | `context.events` |
| Slow or cancellable work | `context.jobs` |
| Visible UI panels and layout | `context.workspace` |
| API clients, sessions, database handles, caches | `context.services` |
| Plugin discovery, actions, workflows, registered panels | `context.plugins` |
| Saved workspace and panel state | `context.persistence` |
| Legacy bridge only | `context.config` |

A plugin should not directly call another panel. It should update platform state or publish an event, and other plugins can react.

---

## Where to put the tutorial plugin

For a bundled/core-style plugin during development:

```text
astronomicAL/plugins/tutorial_plugin/
├── __init__.py
├── plugin.py
└── README.md
```

Local user plugins may also be loaded from configured plugin paths such as:

```text
~/.astronomical/plugins/
```

This tutorial uses one file, `plugin.py`, for clarity. A real plugin can split panels, services, and actions into separate modules once it grows.

---

## Stage 1: register the plugin

Start with a manifest and a `register(api)` function. This is the minimum plugin shape.

```python
# astronomicAL/plugins/tutorial_plugin/plugin.py

from __future__ import annotations

from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="example.tutorial",
    name="Tutorial Plugin",
    version="0.1.0",
    description="A progressive example plugin for learning AstronomicAL plugins.",
    capabilities=[],
    tags=["example", "tutorial"],
)


def register(api) -> None:
    # Nothing is contributed yet.
    # This plugin should still be discoverable and inspectable.
    pass
```

### What this demonstrates

The plugin can now be discovered by the plugin manager. It does not add UI, actions, services, or data yet.

### Important rules

Keep import-time work cheap. Do not load large files, import heavy optional packages, create clients, train models, or touch global application state while the module is imported.

---

## Stage 2: register a simple panel

A panel is visible UI. Register it through `api.register_panel(...)` and return a `(view, controller)` pair from the factory.

```python
from __future__ import annotations

from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="example.tutorial",
    name="Tutorial Plugin",
    version="0.1.0",
    description="A progressive example plugin for learning AstronomicAL plugins.",
    capabilities=["panel"],
    tags=["example", "tutorial"],
)


def register(api) -> None:
    api.register_panel(
        id="hello",
        title="Tutorial: Hello",
        factory=create_hello_panel,
        description="The smallest useful tutorial panel.",
        category="Examples",
        icon="school",
        tags=["example", "tutorial"],
        default_layout={"x": 0, "y": 0, "w": 4, "h": 3},
    )


def create_hello_panel(context, **kwargs):
    import panel as pn

    view = pn.Column(
        pn.pane.Markdown("# Hello from a plugin"),
        pn.pane.Markdown(
            "This panel was registered by `example.tutorial`."
        ),
        sizing_mode="stretch_width",
    )

    # No controller is needed because this panel owns no subscriptions,
    # jobs, watchers, or state.
    return view, None
```

The registered panel id becomes:

```text
example.tutorial.hello
```

Registration ids are automatically namespaced by the plugin id unless they are already fully qualified.

---

## Stage 3: publish a lightweight event

Events are for notifications, not bulk data. Use them when something happened and other plugins may care.

```python
def register(api) -> None: ## All new panels need to be added to the same register function

    api.register_panel( ## you can remove this panel now if you want
        id="hello",
        title="Tutorial: Hello",
        factory=create_hello_panel,
        description="The smallest useful tutorial panel.",
        category="Examples",
        icon="school",
        tags=["example", "tutorial"],
        default_layout={"x": 0, "y": 0, "w": 4, "h": 3},
    )

    api.register_panel( ## Our new panel
        id="event_publisher",
        title="Tutorial: Event Publisher",
        factory=create_event_publisher_panel,
        description="Publishes a simple tutorial event.",
        category="Examples",
    )


def create_event_publisher_panel(context, **kwargs): ## New factory
    import panel as pn

    button = pn.widgets.Button(name="Publish tutorial event", button_type="primary")
    status = pn.pane.Markdown("No event published yet.")

    def publish_event(_event):
        context.events.publish(
            "example.tutorial.ping",
            {
                "message": "Hello from Tutorial Plugin",
                "origin": "example.tutorial.event_publisher",
            },
        )
        status.object = "Published `example.tutorial.ping`."

    button.on_click(publish_event)

    return pn.Column(button, status), None
```

Use dotted topic names. For custom plugin events, prefix with the plugin or workflow namespace.

Good:

```text
example.tutorial.ping
astro.desi.spectrum.loaded
active_learning.query.created
```

Avoid:

```text
updated
new_data
thing happened
```

---

## Stage 4: subscribe to an event

Panels that subscribe to events must clean up in `dispose()`. This prevents closed panels from continuing to receive messages.

```python
def register(api) -> None:

    api.register_panel( ## Previous publisher
        id="event_publisher",
        title="Tutorial: Event Publisher",
        factory=create_event_publisher_panel,
        description="Publishes a simple tutorial event.",
        category="Examples",
    )

    api.register_panel( ## New Listener
        id="event_listener",
        title="Tutorial: Event Listener",
        factory=create_event_listener_panel,
        description="Listens for tutorial events.",
        category="Examples",
    )


def create_event_listener_panel(context, **kwargs):
    controller = EventListenerPanel(context)
    return controller.panel(), controller


class EventListenerPanel:
    def __init__(self, context):
        import panel as pn

        self.context = context
        self._disposed = False
        self.subscriptions = []

        self.log = pn.pane.Markdown("Waiting for `example.tutorial.ping`...")
        self.view = pn.Column(
            pn.pane.Markdown("## Tutorial Event Listener"),
            self.log,
            sizing_mode="stretch_width",
        )

        sub = context.events.subscribe(
            "example.tutorial.ping",
            self.on_ping,
            owner_id="example.tutorial.event_listener",
            owner_label="Tutorial Event Listener",
            owner_kind="panel",
        )
        self.subscriptions.append(sub)

    def on_ping(self, topic, payload):
        if self._disposed:
            return
        self.log.object = (
            f"Received `{topic}`\n\n"
            f"```python\n{payload!r}\n```"
        )

    def panel(self):
        return self.view

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        for sub in list(self.subscriptions):
            self.context.events.unsubscribe(sub)
        self.subscriptions.clear()
```

### Pattern

Publisher panels do not know which panels are listening. Listener panels do not know which panel published the event. The event bus decouples them.

---

## Stage 5: read the active dataset

Most useful plugins read the active dataset. New code should prefer source-aware dataset access where possible, because large datasets may be backed by Parquet/DuckDB instead of an in-memory pandas dataframe.

```python
def register(api) -> None:
    
    ⋮
    ⋮ ## Previous panel registrations will still be here
    ⋮
     
    api.register_panel( 
        id="dataset_summary",
        title="Tutorial: Dataset Summary",
        factory=create_dataset_summary_panel,
        description="Shows basic information about the active dataset.",
        category="Examples",
        tags=["dataset", "summary"],
    )


def create_dataset_summary_panel(context, **kwargs):
    controller = DatasetSummaryPanel(context)
    return controller.panel(), controller


class DatasetSummaryPanel:
    def __init__(self, context):
        import panel as pn

        self.context = context
        self._disposed = False
        self.subscriptions = []

        self.output = pn.pane.Markdown("No active dataset. Load a dataset first.")
        self.refresh_button = pn.widgets.Button(name="Refresh", button_type="primary")
        self.refresh_button.on_click(lambda _event: self.refresh())

        self.view = pn.Column(
            pn.pane.Markdown("## Active Dataset"),
            self.refresh_button,
            self.output,
            sizing_mode="stretch_width",
        )

        for topic in [
            "dataset.loaded",
            "dataset.active.changed",
        ]:
            sub = context.events.subscribe(
                topic,
                self.on_dataset_changed,
                owner_id="example.tutorial.dataset_summary",
                owner_label="Tutorial Dataset Summary",
                owner_kind="panel",
            )
            self.subscriptions.append(sub)

        self.refresh()

    def _active_dataset_id(self):
        """
        Return the active dataset id, or None if no active dataset exists yet.

        Some DatasetManager implementations raise when active_id() is called
        before a dataset is loaded, so this helper must be defensive.
        """
        active_id = getattr(self.context.datasets, "active_id", None)

        if callable(active_id):
            try:
                return active_id()
            except Exception:
                return None

        value = getattr(self.context.datasets, "active_dataset_id", None)
        if value:
            return value

        return None

    def refresh(self):
        dataset_id = self._active_dataset_id()

        if not dataset_id:
            self.output.object = "No active dataset. Load a dataset first."
            return

        try:
            source = self.context.datasets.get_source(dataset_id)
        except Exception as exc:
            self.output.object = (
                f"Dataset `{dataset_id}` is active, but could not be read.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        columns = source.columns()
        row_count = source.row_count()
        preview = source.head(5)

        self.output.object = (
            f"**Dataset:** `{dataset_id}`\n\n"
            f"**Rows:** `{row_count}`\n\n"
            f"**Columns:** `{len(columns)}`\n\n"
            f"**First columns:** `{columns[:10]}`\n\n"
            "### Preview\n\n"
            f"```python\n{preview!r}\n```"
        )

    def on_dataset_changed(self, topic, payload):
        if not self._disposed:
            self.refresh()

    def panel(self):
        return self.view

    def dispose(self):
        if self._disposed:
            return

        self._disposed = True

        for sub in list(self.subscriptions):
            self.context.events.unsubscribe(sub)

        self.subscriptions.clear()
```

### Why not always use pandas?

Older code often does this:

```python
df = context.datasets.get_df(dataset_id)
```

That can force a large source-backed dataset into memory. Prefer:

```python
source = context.datasets.get_source(dataset_id)
columns = source.columns()
row_count = source.row_count()
preview = source.head(100, columns=["id", "score"])
```

Use pandas materialization only when the operation truly requires it and the dataset is small enough.

---

## Stage 6: declare required semantic mappings

Plugin panels should not assume that the row id column is named `id`, `source_id`, `object_id`, or anything else. Ask for semantic mappings.

A panel can declare mapping requirements during registration:

```python
def register(api) -> None:

    ⋮
    ⋮ ## Previous panel registrations will still be here
    ⋮

    api.register_panel(
        id="mapped_dataset_summary",
        title="Tutorial: Mapped Dataset Summary",
        factory=create_mapped_dataset_summary_panel,
        description="Shows dataset information using semantic column mappings.",
        category="Examples",
        required_mappings=["record_id"],
        optional_mappings=["target_label", "ra", "dec"],
    )
```

Then implement the factory and panel:

```python
def create_mapped_dataset_summary_panel(context, **kwargs):
    controller = MappedDatasetSummaryPanel(context)
    return controller.panel(), controller


class MappedDatasetSummaryPanel:
    def __init__(self, context):
        import panel as pn

        self.context = context
        self._disposed = False
        self.subscriptions = []

        self.status = pn.pane.Markdown(
            "No active dataset. Load a dataset first.",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.preview = pn.pane.DataFrame(
            None,
            sizing_mode="stretch_width",
            height=200,
            margin=(0, 0, 0, 0),
        )

        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="primary",
            width=90,
            height=30,
            sizing_mode="fixed",
            margin=(0, 0, 0, 12),
        )
        self.refresh_button.on_click(lambda _event: self.refresh())

        self.header = pn.Row(
            pn.pane.HTML(
                "<h2 style='margin: 0;'>Mapped Dataset Summary</h2>",
                sizing_mode="stretch_width",
            ),
            pn.Spacer(sizing_mode="stretch_width"),
            self.refresh_button,
            sizing_mode="stretch_width",
            align="center",
            margin=(0, 0, 12, 0),
        )

        self.view = pn.Column(
            self.header,
            self.status,
            pn.pane.HTML(
                "<h3 style='margin: 8px 0 6px 0;'>Preview</h3>",
                sizing_mode="stretch_width",
            ),
            self.preview,
            sizing_mode="stretch_width",
            margin=(16, 24),
        )

        for topic in [
            "dataset.loaded",
            "dataset.active.changed",
            "dataset.mapping.updated",
        ]:
            sub = context.events.subscribe(
                topic,
                self.on_dataset_changed,
                owner_id="example.tutorial.mapped_dataset_summary",
                owner_label="Tutorial Mapped Dataset Summary",
                owner_kind="panel",
            )
            self.subscriptions.append(sub)

        self.refresh()

    def _active_dataset_id(self):
        """
        Return the active dataset id, or None if no active dataset exists yet.

        Some DatasetManager implementations raise when active_id() is called
        before a dataset is loaded, so this helper must be defensive.
        """
        active_id = getattr(self.context.datasets, "active_id", None)

        if callable(active_id):
            try:
                return active_id()
            except Exception:
                return None

        value = getattr(self.context.datasets, "active_dataset_id", None)
        if value:
            return value

        return None

    def refresh(self):
        dataset_id = self._active_dataset_id()

        if not dataset_id:
            self.status.object = "No active dataset. Load a dataset first."
            self.preview.object = None
            return

        try:
            id_col = self.context.datasets.get_mapping(dataset_id, "record_id")
            label_col = self.context.datasets.get_mapping(dataset_id, "target_label")
            ra_col = self.context.datasets.get_mapping(dataset_id, "ra")
            dec_col = self.context.datasets.get_mapping(dataset_id, "dec")
        except Exception as exc:
            self.status.object = (
                "The active dataset exists, but its mappings could not be read.\n\n"
                f"```text\n{exc}\n```"
            )
            self.preview.object = None
            return

        try:
            source = self.context.datasets.get_source(dataset_id)
        except Exception as exc:
            self.status.object = (
                f"Dataset `{dataset_id}` is active, but could not be read.\n\n"
                f"```text\n{exc}\n```"
            )
            self.preview.object = None
            return

        mapped_columns = [
            column
            for column in [id_col, label_col, ra_col, dec_col]
            if column
        ]

        try:
            preview = source.head(10, columns=mapped_columns or None)
        except TypeError:
            # Fallback for DatasetSource implementations whose head()
            # does not accept a columns argument yet.
            preview = source.head(10)
            if mapped_columns:
                preview = preview[mapped_columns]

        self.status.object = (
            f"**Dataset:** `{dataset_id}`  \n"
            f"**Rows:** `{source.row_count()}`  \n"
            f"**record_id:** `{id_col}`  \n"
            f"**target_label:** `{label_col}`  \n"
            f"**ra:** `{ra_col}`  \n"
            f"**dec:** `{dec_col}`"
        )

        self.preview.object = preview

    def on_dataset_changed(self, topic, payload):
        if not self._disposed:
            self.refresh()

    def panel(self):
        return self.view

    def dispose(self):
        if self._disposed:
            return

        self._disposed = True

        for sub in list(self.subscriptions):
            self.context.events.unsubscribe(sub)

        self.subscriptions.clear()
```

### What the mapping gate does

If a panel declares required mappings, the platform should show mapping UI instead of constructing the real panel until the required mappings are available. This keeps plugin code simpler: the panel can assume required mappings exist once it is created.

However, tutorial code should still be defensive because users may open panels before loading a dataset, or while mappings are being edited.

---

## Stage 7: publish focused row selection

The current focused row belongs in `context.selection`. This lets record browsers, plots, annotation panels, and domain plugins all react to the same selected object.

```python
def register(api) -> None:

    ⋮
    ⋮ ## Previous panel registrations will still be here
    ⋮

    api.register_panel(
        id="focus_first_row",
        title="Tutorial: Focus First Row",
        factory=create_focus_first_row_panel,
        description="Publishes the first row as the focused record.",
        category="Examples",
        required_mappings=["record_id"],
    )


def create_focus_first_row_panel(context, **kwargs):
    import panel as pn

    status = pn.pane.Markdown(
        "No focused row yet.",
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )

    button = pn.widgets.Button(
        name="Focus first row",
        button_type="primary",
        width=130,
        height=30,
        sizing_mode="fixed",
        margin=(0, 0, 0, 0),
    )

    def active_dataset_id():
        active_id = getattr(context.datasets, "active_id", None)

        if callable(active_id):
            try:
                return active_id()
            except Exception:
                return None

        return getattr(context.datasets, "active_dataset_id", None)

    def focus_first(_event):
        dataset_id = active_dataset_id()
        if not dataset_id:
            status.object = "No active dataset. Load a dataset first."
            return

        try:
            id_col = context.datasets.get_mapping(dataset_id, "record_id")
            source = context.datasets.get_source(dataset_id)
            preview = source.head(1, columns=[id_col])
        except TypeError:
            # Fallback for DatasetSource implementations whose head()
            # does not accept a columns argument yet.
            source = context.datasets.get_source(dataset_id)
            preview = source.head(1)
        except Exception as exc:
            status.object = (
                "Could not read the first row from the active dataset.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        if preview is None or len(preview) == 0:
            status.object = "Active dataset is empty."
            return

        try:
            row_id = preview.iloc[0][id_col]
        except Exception as exc:
            status.object = (
                f"The `record_id` mapping points to `{id_col}`, but that column "
                "was not available in the preview.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        context.selection.set_focus(
            dataset_id=dataset_id,
            row_id=row_id,
            origin="example.tutorial.focus_first_row",
        )

        status.object = (
            f"Focused row `{row_id}` in dataset `{dataset_id}`."
        )

    button.on_click(focus_first)

    view = pn.Column(
        pn.pane.HTML(
            "<h2 style='margin: 0 0 12px 0;'>Focus First Row</h2>",
            sizing_mode="stretch_width",
        ),
        pn.Row(
            button,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        ),
        status,
        sizing_mode="stretch_width",
        margin=(16, 24),
    )

    return view, None
```

Other plugins can now subscribe to:

```text
selection.focus.changed
selection.focus.cleared
```

They should read the focus state from `context.selection`, not rely only on the event payload.

---

## Stage 8: subscribe to focused row changes

```python
def register(api) -> None:
    
    ⋮
    ⋮ ## Previous panel registrations will still be here
    ⋮

    api.register_panel(
        id="focus_listener",
        title="Tutorial: Focus Listener",
        factory=create_focus_listener_panel,
        description="Shows the current focused row.",
        category="Examples",
    )


def create_focus_listener_panel(context, **kwargs):
    controller = FocusListenerPanel(context)
    return controller.panel(), controller


class FocusListenerPanel:
    def __init__(self, context):
        import panel as pn

        self.context = context
        self._disposed = False
        self.subscriptions = []

        self.output = pn.pane.Markdown("No focused row.")
        self.view = pn.Column(
            pn.pane.Markdown("## Current Focus"),
            self.output,
            sizing_mode="stretch_width",
        )

        sub = context.events.subscribe(
            "selection.focus.changed",
            self.on_focus_changed,
            owner_id="example.tutorial.focus_listener",
            owner_label="Tutorial Focus Listener",
            owner_kind="panel",
        )
        self.subscriptions.append(sub)

        clear_sub = context.events.subscribe(
            "selection.focus.cleared",
            self.on_focus_changed,
            owner_id="example.tutorial.focus_listener",
            owner_label="Tutorial Focus Listener",
            owner_kind="panel",
        )
        self.subscriptions.append(clear_sub)

        self.refresh()

    def refresh(self):
        focus = self.context.selection.get_focus()
        if not focus:
            self.output.object = "No focused row."
            return

        self.output.object = (
            f"**Dataset:** `{focus.dataset_id}`\n\n"
            f"**Row id:** `{focus.row_id}`\n\n"
            f"**Origin:** `{focus.origin}`"
        )

    def on_focus_changed(self, topic, payload):
        if not self._disposed:
            self.refresh()

    def panel(self):
        return self.view

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        for sub in list(self.subscriptions):
            self.context.events.unsubscribe(sub)
        self.subscriptions.clear()
```

### Pattern

Events tell the panel to refresh. Platform state tells the panel what the current value is.

This is more robust than trusting event payloads alone, because panels may be opened after an event has already happened.

---

## Stage 9: publish a multi-row selection set

Selection sets are for a group of row ids. A plugin might create them from a lasso selection, a filter, a model query, or a table selection.

```python
def register(api) -> None:

    ⋮
    ⋮ ## Previous panel registrations will still be here
    ⋮

    api.register_panel(
        id="select_first_rows",
        title="Tutorial: Select First Rows",
        factory=create_select_first_rows_panel,
        description="Publishes the first N rows as the active selection set.",
        category="Examples",
        required_mappings=["record_id"],
    )


def create_select_first_rows_panel(context, **kwargs):
    import panel as pn

    count = pn.widgets.IntSlider(
        name="Rows",
        start=1,
        end=100,
        value=10,
        width=240,
        sizing_mode="fixed",
        margin=(0, 0, 8, 0),
    )

    button = pn.widgets.Button(
        name="Select first rows",
        button_type="primary",
        width=140,
        height=30,
        sizing_mode="fixed",
        margin=(0, 0, 0, 0),
    )

    status = pn.pane.Markdown(
        "No selection set created yet.",
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )

    def active_dataset_id():
        active_id = getattr(context.datasets, "active_id", None)

        if callable(active_id):
            try:
                return active_id()
            except Exception:
                return None

        return getattr(context.datasets, "active_dataset_id", None)

    def select_rows(_event):
        dataset_id = active_dataset_id()
        if not dataset_id:
            status.object = "No active dataset. Load a dataset first."
            return

        try:
            id_col = context.datasets.get_mapping(dataset_id, "record_id")
            source = context.datasets.get_source(dataset_id)
            preview = source.head(count.value, columns=[id_col])
        except TypeError:
            # Fallback for DatasetSource implementations whose head()
            # does not accept a columns argument yet.
            try:
                source = context.datasets.get_source(dataset_id)
                preview = source.head(count.value)
                id_col = context.datasets.get_mapping(dataset_id, "record_id")
                preview = preview[[id_col]]
            except Exception as exc:
                status.object = (
                    "Could not read rows from the active dataset.\n\n"
                    f"```text\n{exc}\n```"
                )
                return
        except Exception as exc:
            status.object = (
                "Could not read rows from the active dataset.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        if preview is None or len(preview) == 0:
            status.object = "Active dataset is empty."
            return

        try:
            row_ids = list(preview[id_col])
        except Exception as exc:
            status.object = (
                f"The `record_id` mapping points to `{id_col}`, but that column "
                "was not available in the preview.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        selection = context.selection.set_selection_set(
            dataset_id=dataset_id,
            row_ids=row_ids,
            origin="example.tutorial.select_first_rows",
            mode="replace",
            metadata={
                "name": "Tutorial first rows",
                "requested_count": count.value,
            },
        )

        status.object = (
            f"Created selection set `{selection.selection_set_id}` with "
            f"`{len(row_ids)}` rows from dataset `{dataset_id}`."
        )

    button.on_click(select_rows)

    view = pn.Column(
        pn.pane.HTML(
            "<h2 style='margin: 0 0 12px 0;'>Select First Rows</h2>",
            sizing_mode="stretch_width",
        ),
        pn.Row(
            count,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        ),
        pn.Row(
            button,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        ),
        status,
        sizing_mode="stretch_width",
        margin=(16, 24),
    )

    return view, None
```

Other plugins can subscribe to:

```text
selection.set.changed
selection.set.cleared
```

A good pattern is to use events to trigger refresh, then call `context.selection.get_active_selection_set()` or equivalent selection manager method to get current state.

You can visualise the points being selected by opening a scatter plot from: Core->Visualisation->Scatter Plot

---

## Stage 10: store a derived output as an artifact

Artifacts are for derived products that other plugins may inspect, display, save, or use later. Examples include model scores, profile summaries, reports, spectra, cutouts, plots, predictions, or fetched metadata.

```python
def register(api) -> None:

    ⋮
    ⋮ ## Previous panel registrations will still be here
    ⋮

    api.register_panel(
        id="profile_artifact",
        title="Tutorial: Profile Artifact",
        factory=create_profile_artifact_panel,
        description="Creates a small artifact describing the active dataset.",
        category="Examples",
    )


def create_profile_artifact_panel(context, **kwargs):
    import panel as pn

    button = pn.widgets.Button(
        name="Create profile artifact",
        button_type="primary",
        width=170,
        height=30,
        sizing_mode="fixed",
        margin=(0, 0, 0, 0),
    )

    status = pn.pane.Markdown(
        "No artifact created yet.",
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )

    def active_dataset_id():
        active_id = getattr(context.datasets, "active_id", None)

        if callable(active_id):
            try:
                return active_id()
            except Exception:
                return None

        return getattr(context.datasets, "active_dataset_id", None)

    def create_artifact(_event):
        dataset_id = active_dataset_id()
        if not dataset_id:
            status.object = "No active dataset. Load a dataset first."
            return

        try:
            source = context.datasets.get_source(dataset_id)
        except Exception as exc:
            status.object = (
                f"Dataset `{dataset_id}` is active, but could not be read.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        try:
            payload = {
                "dataset_id": dataset_id,
                "rows": source.row_count(),
                "columns": source.columns(),
                "dtypes": source.dtypes(),
            }
        except Exception as exc:
            status.object = (
                "Could not create a profile for the active dataset.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        artifact_id = context.artifacts.put(
            "example.tutorial.dataset_profile",
            payload,
            dataset_id=dataset_id,
        )

        context.events.publish(
            "artifact.created",
            {
                "artifact_id": artifact_id,
                "type": "example.tutorial.dataset_profile",
                "dataset_id": dataset_id,
                "origin": "example.tutorial.profile_artifact",
            },
        )

        status.object = (
            f"Created dataset profile artifact `{artifact_id}` for "
            f"dataset `{dataset_id}`."
        )

    button.on_click(create_artifact)

    view = pn.Column(
        pn.pane.HTML(
            "<h2 style='margin: 0 0 12px 0;'>Profile Artifact</h2>",
            sizing_mode="stretch_width",
        ),
        pn.Row(
            button,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        ),
        status,
        sizing_mode="stretch_width",
        margin=(16, 24),
    )

    return view, None
```

### Artifact rule

Store data in artifacts. Publish event metadata about the artifact.

Avoid:

```python
context.events.publish("example.tutorial.profile", huge_profile_payload)
```

Prefer:

```python
artifact_id = context.artifacts.put("example.tutorial.dataset_profile", payload)
context.events.publish(
    "artifact.created",
    {
        "artifact_id": artifact_id,
        "type": "example.tutorial.dataset_profile",
    },
)
```

---

## Stage 11: subscribe to artifacts

A panel can react to artifact events and then read the actual artifact from `context.artifacts`.

```python
def register(api) -> None:

    ⋮
    ⋮ ## Previous panel registrations will still be here
    ⋮

    api.register_panel(
        id="artifact_feed",
        title="Tutorial: Artifact Feed",
        factory=create_artifact_feed_panel,
        description="Shows profile artifacts created by the tutorial plugin.",
        category="Examples",
    )


def create_artifact_feed_panel(context, **kwargs):
    controller = ArtifactFeedPanel(context)
    return controller.panel(), controller


class ArtifactFeedPanel:
    def __init__(self, context):
        import panel as pn

        self.context = context
        self._disposed = False
        self.subscriptions = []
        self.artifact_ids = []

        self.output = pn.pane.Markdown("No tutorial profile artifacts seen yet.")
        self.view = pn.Column(
            pn.pane.Markdown("## Tutorial Artifact Feed"),
            self.output,
            sizing_mode="stretch_width",
        )

        sub = context.events.subscribe(
            "artifact.created",
            self.on_artifact_created,
            owner_id="example.tutorial.artifact_feed",
            owner_label="Tutorial Artifact Feed",
            owner_kind="panel",
        )
        self.subscriptions.append(sub)

    def on_artifact_created(self, topic, payload):
        if self._disposed:
            return

        if payload.get("type") != "example.tutorial.dataset_profile":
            return

        artifact_id = payload.get("artifact_id")
        self.artifact_ids.append(artifact_id)

        artifact = self.context.artifacts.get(artifact_id)

        self.output.object = (
            f"Latest tutorial artifact: `{artifact_id}`\n\n"
            f"```python\n{artifact!r}\n```"
        )

    def panel(self):
        return self.view

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        for sub in list(self.subscriptions):
            self.context.events.unsubscribe(sub)
        self.subscriptions.clear()
```

---

## Stage 12: register an action

Actions are reusable plugin operations. They can be invoked from panels, menus, workflows, or other platform UI.

```python
## Ensure these are now imported
from astronomicAL.platform.plugins import PluginManifest
from astronomicAL.platform.plugins.specs import InputSpec, ArtifactResult


def register(api) -> None:

    ⋮
    ⋮ ## Previous registrations will still be here
    ⋮

    api.register_action(
        id="profile_dataset",
        title="Tutorial: Profile Dataset",
        handler=profile_dataset,
        description="Creates a dataset profile artifact.",
        inputs=InputSpec(dataset=True, selection="optional", columns="optional"),
        outputs=["example.tutorial.dataset_profile"],
        run_in_job=True,
        category="Examples",
        tags=["tutorial", "profile"],
    )


def profile_dataset(context, request, cancel_token=None):
    dataset_id = request.dataset_id
    if not dataset_id:
        dataset_id = context.datasets.active_id()

    if not dataset_id:
        raise ValueError("No dataset selected.")

    source = context.datasets.get_source(dataset_id)
    columns = request.columns or source.columns()

    if cancel_token and cancel_token.cancelled():
        return None

    payload = {
        "dataset_id": dataset_id,
        "row_count": source.row_count(),
        "columns": columns,
        "dtypes": source.dtypes(),
        "row_ids": request.row_ids,
        "params": request.params,
    }

    return ArtifactResult(
        type="example.tutorial.dataset_profile",
        payload=payload,
        dataset_id=dataset_id,
        row_ids=request.row_ids,
        params=request.params,
    )
```

### Why actions matter

A panel button is useful only from that panel. An action can be reused by command palettes, workflows, menus, tests, or other plugins.

---

## Stage 13: call an action from a panel

Now that `profile_dataset` is registered as an action, update the Stage 10 panel so it has two buttons:

1. **Create artifact directly** — the panel calls `context.artifacts.put(...)` itself.
2. **Create artifact through action** — the panel calls `context.plugins.run_action(...)`, and the plugin manager processes the returned `ArtifactResult`.

This lets you see why actions are useful: the panel-specific version and the reusable action version produce similar outputs, but the action can also be reused by workflows, command palettes, menus, or other plugins.

Replace the Stage 10 `create_profile_artifact_panel` function with this version:

```python
def create_profile_artifact_panel(context, **kwargs):
    import panel as pn

    direct_button = pn.widgets.Button(
        name="Create directly",
        button_type="primary",
        width=130,
        height=30,
        sizing_mode="fixed",
        margin=(0, 8, 0, 0),
    )

    action_button = pn.widgets.Button(
        name="Create via action",
        button_type="success",
        width=140,
        height=30,
        sizing_mode="fixed",
        margin=(0, 0, 0, 0),
    )

    status = pn.pane.Markdown(
        "No artifact created yet.",
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )

    def active_dataset_id():
        active_id = getattr(context.datasets, "active_id", None)

        if callable(active_id):
            try:
                return active_id()
            except Exception:
                return None

        return getattr(context.datasets, "active_dataset_id", None)

    def create_directly(_event):
        dataset_id = active_dataset_id()
        if not dataset_id:
            status.object = "No active dataset. Load a dataset first."
            return

        try:
            source = context.datasets.get_source(dataset_id)
            payload = {
                "dataset_id": dataset_id,
                "rows": source.row_count(),
                "columns": source.columns(),
                "dtypes": source.dtypes(),
                "created_by": "direct panel code",
            }
        except Exception as exc:
            status.object = (
                "Could not create a direct dataset profile.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        artifact_id = context.artifacts.put(
            "example.tutorial.dataset_profile",
            payload,
            dataset_id=dataset_id,
        )

        context.events.publish(
            "artifact.created",
            {
                "artifact_id": artifact_id,
                "type": "example.tutorial.dataset_profile",
                "dataset_id": dataset_id,
                "origin": "example.tutorial.profile_artifact.direct",
            },
        )

        status.object = (
            f"Created artifact directly: `{artifact_id}`\n\n"
            f"Dataset: `{dataset_id}`"
        )

    def create_via_action(_event):
        dataset_id = active_dataset_id()
        if not dataset_id:
            status.object = "No active dataset. Load a dataset first."
            return

        status.object = "Submitting `example.tutorial.profile_dataset` action..."

        def on_done(result):
            artifact_ids = getattr(result, "artifact_ids", None)

            if artifact_ids:
                status.object = (
                    "Action completed.\n\n"
                    f"Created artifact through action: `{artifact_ids[0]}`\n\n"
                    f"Dataset: `{dataset_id}`"
                )
            else:
                status.object = (
                    "Action completed, but no artifact id was returned.\n\n"
                    f"```python\n{result!r}\n```"
                )

        def on_error(exc):
            status.object = (
                "Action failed.\n\n"
                f"```text\n{exc}\n```"
            )

        try:
            handle_or_result = context.plugins.run_action(
                "example.tutorial.profile_dataset",
                context,
                {
                    "dataset_id": dataset_id,
                    "params": {
                        "origin": "example.tutorial.profile_artifact.action",
                    },
                },
                on_done=on_done,
                on_error=on_error,
                return_processed=True,
            )
        except Exception as exc:
            status.object = (
                "Could not start action.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        # If the action is registered with run_in_job=True, run_action returns a
        # job handle and on_done will update the status later.
        #
        # If the action is not job-backed, run_action returns the processed result
        # immediately, so handle that case too.
        if hasattr(handle_or_result, "artifact_ids"):
            on_done(handle_or_result)
        else:
            status.object = (
                "Action submitted. Waiting for it to finish..."
            )

    direct_button.on_click(create_directly)
    action_button.on_click(create_via_action)

    view = pn.Column(
        pn.pane.HTML(
            "<h2 style='margin: 0 0 12px 0;'>Profile Artifact</h2>",
            sizing_mode="stretch_width",
        ),
        pn.Row(
            direct_button,
            action_button,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        ),
        status,
        sizing_mode="stretch_width",
        margin=(16, 24),
    )

    return view, None
```

The important comparison is:

```python
artifact_id = context.artifacts.put(...)
context.events.publish("artifact.created", ...)
```

versus:

```python
context.plugins.run_action(
    "example.tutorial.profile_dataset",
    context,
    {"dataset_id": dataset_id},
    return_processed=True,
)
```

The first approach is fine for simple panel-local behaviour. The second approach is better when the operation should be reusable outside that one panel.

Depending on how the action was registered, the plugin manager may run it directly or submit it as a job.

---

## Stage 14: register a service

Services are for shared, live runtime capabilities: clients, caches, sessions, connections, or stateful helpers. They are not for storing derived outputs.

```python
class TutorialClient:
    def __init__(self, base_url=None):
        self.base_url = base_url or "https://example.invalid"

    def fetch_metadata(self, row_id):
        # Replace with a real network call in a real plugin.
        return {
            "row_id": row_id,
            "source": self.base_url,
            "message": "Example metadata",
        }

    def close(self):
        pass


def create_tutorial_client(context, **kwargs):
    settings = kwargs.get("settings") or {}
    return TutorialClient(base_url=settings.get("base_url"))


def register(api) -> None:

    ⋮
    ⋮ ## Previous panel registrations will still be here
    ⋮

    api.register_service(
        key="client",
        factory=create_tutorial_client,
        description="Example client shared by tutorial panels/actions.",
        lazy=True,
        replace=False,
    )
```

The registered service id becomes:

```text
example.tutorial.client
```

Using the service:

```python
client = context.services.get("example.tutorial.client")
metadata = client.fetch_metadata(row_id)
```

### Service rule

Use a service when the object is a live capability. Use an artifact when the object is a computed result.

---

## Stage 15: use a service from a focused-row panel

```python
def register(api) -> None:

    ⋮
    ⋮ ## Previous panel registrations will still be here
    ⋮

    api.register_panel(
        id="metadata_lookup",
        title="Tutorial: Metadata Lookup",
        factory=create_metadata_lookup_panel,
        description="Uses a registered service to fetch metadata for the focused row.",
        category="Examples",
    )

    api.register_service(
        key="client",
        factory=create_tutorial_client,
        description="Example client shared by tutorial panels/actions.",
        lazy=True,
        replace=False,
    )


def create_metadata_lookup_panel(context, **kwargs):
    controller = MetadataLookupPanel(context)
    return controller.panel(), controller


class MetadataLookupPanel:
    def __init__(self, context):
        import panel as pn

        self.context = context
        self._disposed = False
        self.subscriptions = []

        self.button = pn.widgets.Button(name="Fetch metadata", button_type="primary")
        self.output = pn.pane.Markdown("Focus a row, then fetch metadata.")
        self.button.on_click(self.fetch)

        self.view = pn.Column(
            pn.pane.Markdown("## Tutorial Metadata Lookup"),
            self.button,
            self.output,
            sizing_mode="stretch_width",
        )

        sub = context.events.subscribe(
            "selection.focus.changed",
            self.on_focus_changed,
            owner_id="example.tutorial.metadata_lookup",
            owner_label="Tutorial Metadata Lookup",
            owner_kind="panel",
        )
        self.subscriptions.append(sub)

    def on_focus_changed(self, topic, payload):
        if not self._disposed:
            focus = self.context.selection.get_focus()
            self.output.object = (
                f"Focused row `{focus.row_id}`. Click **Fetch metadata**."
                if focus
                else "No focused row."
            )

    def fetch(self, _event):
        focus = self.context.selection.get_focus()
        if not focus:
            self.output.object = "No focused row."
            return

        client = self.context.services.get("example.tutorial.client")
        metadata = client.fetch_metadata(focus.row_id)

        artifact_id = self.context.artifacts.put(
            "example.tutorial.metadata",
            metadata,
            dataset_id=focus.dataset_id,
            row_ids=[focus.row_id],
        )

        self.context.events.publish(
            "artifact.created",
            {
                "artifact_id": artifact_id,
                "type": "example.tutorial.metadata",
                "dataset_id": focus.dataset_id,
                "row_ids": [focus.row_id],
                "origin": "example.tutorial.metadata_lookup",
            },
        )

        self.output.object = (
            f"Created metadata artifact `{artifact_id}`.\n\n"
            f"```python\n{metadata!r}\n```"
        )

    def panel(self):
        return self.view

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        for sub in list(self.subscriptions):
            self.context.events.unsubscribe(sub)
        self.subscriptions.clear()
```

---

## Stage 16: submit slow work as a job

If work may block the UI, run it through the job manager. This includes network requests, model inference, expensive table transforms, large file IO, image generation, remote queries, and long plotting preparation.

```python
def register(api) -> None:

    ⋮
    ⋮ ## Previous panel registrations will still be here
    ⋮

    api.register_panel(
        id="slow_profile",
        title="Tutorial: Slow Profile Job",
        factory=create_slow_profile_panel,
        description="Creates a dataset profile using the shared job manager.",
        category="Examples",
        tags=["tutorial", "jobs"],
    )

def create_slow_profile_panel(context, **kwargs):
    import panel as pn

    start_button = pn.widgets.Button(
        name="Start profile job",
        button_type="primary",
        width=140,
        height=30,
        sizing_mode="fixed",
        margin=(0, 8, 0, 0),
    )

    cancel_button = pn.widgets.Button(
        name="Cancel",
        button_type="default",
        width=90,
        height=30,
        sizing_mode="fixed",
        margin=(0, 0, 0, 0),
        disabled=True,
    )

    status = pn.pane.Markdown(
        "No job running.",
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )

    current_job = {"handle": None}

    def active_dataset_id():
        active_id = getattr(context.datasets, "active_id", None)

        if callable(active_id):
            try:
                return active_id()
            except Exception:
                return None

        return getattr(context.datasets, "active_dataset_id", None)

    def start_job(_event):
        dataset_id = active_dataset_id()
        if not dataset_id:
            status.object = "No active dataset. Load a dataset first."
            return

        start_button.disabled = True
        cancel_button.disabled = False
        status.object = f"Starting profile job for dataset `{dataset_id}`..."

        def work(cancel_token=None):
            source = context.datasets.get_source(dataset_id)

            columns = source.columns()
            dtypes = source.dtypes()
            row_count = source.row_count()

            profile = {
                "dataset_id": dataset_id,
                "row_count": row_count,
                "column_count": len(columns),
                "columns": {},
            }

            for index, column in enumerate(columns):
                if cancel_token and cancel_token.cancelled():
                    return {
                        "dataset_id": dataset_id,
                        "cancelled": True,
                        "processed_columns": index,
                    }

                profile["columns"][column] = {
                    "dtype": str(dtypes.get(column)),
                    "position": index,
                }

            return profile

        def on_done(result):
            start_button.disabled = False
            cancel_button.disabled = True
            current_job["handle"] = None

            if result is None:
                status.object = "Job finished with no result."
                return

            if result.get("cancelled"):
                status.object = (
                    "Job cancelled.\n\n"
                    f"Processed `{result.get('processed_columns', 0)}` columns."
                )
                return

            artifact_id = context.artifacts.put(
                "example.tutorial.slow_profile",
                result,
                dataset_id=dataset_id,
            )

            context.events.publish(
                "artifact.created",
                {
                    "artifact_id": artifact_id,
                    "type": "example.tutorial.slow_profile",
                    "dataset_id": dataset_id,
                    "origin": "example.tutorial.slow_profile",
                },
            )

            status.object = (
                f"Job completed.\n\n"
                f"Created slow profile artifact `{artifact_id}`."
            )

        def on_error(exc):
            start_button.disabled = False
            cancel_button.disabled = True
            current_job["handle"] = None

            status.object = (
                "Job failed.\n\n"
                f"```text\n{exc}\n```"
            )

        try:
            handle = context.jobs.submit(
                work,
                title="Tutorial slow profile",
                key=f"example.tutorial.slow_profile:{dataset_id}",
                on_done=on_done,
                on_error=on_error,
            )
        except Exception as exc:
            start_button.disabled = False
            cancel_button.disabled = True
            status.object = (
                "Could not submit job.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        current_job["handle"] = handle
        status.object = f"Submitted job `{handle.job_id}`."

    def cancel_job(_event):
        handle = current_job.get("handle")
        if handle is None:
            status.object = "No active job to cancel."
            return

        handle.cancel()
        status.object = f"Cancellation requested for job `{handle.job_id}`."

    start_button.on_click(start_job)
    cancel_button.on_click(cancel_job)

    view = pn.Column(
        pn.pane.HTML(
            "<h2 style='margin: 0 0 12px 0;'>Slow Profile Job</h2>",
            sizing_mode="stretch_width",
        ),
        pn.Row(
            start_button,
            cancel_button,
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        ),
        status,
        sizing_mode="stretch_width",
        margin=(16, 24),
    )

    return view, None
```

### Job guidance

Use jobs for expensive work. Keep panel construction fast. Check cancellation in long loops. Store large or reusable results as artifacts, and publish a lightweight event after the artifact is created.

---

## Stage 17: register and use an artifact viewer

Stage 10 created an artifact of type:

```text
example.tutorial.dataset_profile
```

That artifact stores data, but the platform does not automatically know the best way to display it. An artifact viewer solves that.

A viewer is useful because it lets any generic artifact browser, workflow, or panel say:

> “I have an artifact id. Which plugin knows how to render this artifact type?”

The plugin that created the artifact type can provide the renderer.

```python
def register(api) -> None:

    ⋮
    ⋮ ## Previous panel registrations will still be here
    ⋮

    api.register_artifact_viewer(
        artifact_type="example.tutorial.dataset_profile",
        viewer_factory=create_dataset_profile_viewer,
        id="dataset_profile_viewer",
        title="Tutorial Dataset Profile Viewer",
        description="Displays tutorial dataset profile artifacts.",
        priority=10,
        default=True,
    )

    api.register_panel(
        id="profile_artifact_browser",
        title="Tutorial: Profile Artifact Browser",
        factory=create_profile_artifact_browser_panel,
        description="Finds tutorial profile artifacts and renders them with the registered viewer.",
        category="Examples",
        tags=["tutorial", "artifacts"],
    )


def create_dataset_profile_viewer(context, artifact_id=None, **kwargs):
    import pandas as pd
    import panel as pn

    if not artifact_id:
        return pn.pane.Markdown("No artifact selected."), None

    try:
        payload = context.artifacts.get(artifact_id)
        ref = context.artifacts.ref(artifact_id)
    except Exception as exc:
        return pn.Column(
            pn.pane.HTML(
                "<h2 style='margin: 0 0 12px 0;'>Dataset Profile Artifact</h2>",
                sizing_mode="stretch_width",
            ),
            pn.pane.Alert(
                f"Could not load artifact `{artifact_id}`.\n\n{exc}",
                alert_type="danger",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            margin=(16, 24),
        ), None

    columns = payload.get("columns", [])
    dtypes = payload.get("dtypes", {})

    rows = []
    for column in columns:
        rows.append(
            {
                "column": column,
                "dtype": str(dtypes.get(column, "")),
            }
        )

    table = pd.DataFrame(rows)

    summary = pn.pane.Markdown(
        f"**Artifact id:** `{artifact_id}`  \n"
        f"**Artifact type:** `{ref.type}`  \n"
        f"**Dataset:** `{ref.dataset_id}`  \n"
        f"**Rows:** `{payload.get('rows', payload.get('row_count', 'unknown'))}`  \n"
        f"**Columns:** `{len(columns)}`",
        sizing_mode="stretch_width",
        margin=(0, 0, 12, 0),
    )

    view = pn.Column(
        pn.pane.HTML(
            "<h2 style='margin: 0 0 12px 0;'>Dataset Profile Artifact</h2>",
            sizing_mode="stretch_width",
        ),
        summary,
        pn.pane.HTML(
            "<h3 style='margin: 8px 0 6px 0;'>Column dtypes</h3>",
            sizing_mode="stretch_width",
        ),
        pn.pane.DataFrame(
            table,
            sizing_mode="stretch_width",
            height=260,
        ),
        pn.pane.DataFrame(
            table,
            sizing_mode="stretch_width",
        ),
        margin=(16, 24),
    )

    return view, None
```

Now add a small panel that proves why the viewer is useful. It finds the latest tutorial profile artifact, asks the plugin manager for a viewer registered for that artifact type, and renders it without hard-coding the display logic into the browser panel.

```python
def create_profile_artifact_browser_panel(context, **kwargs):
    controller = ProfileArtifactBrowserPanel(context)
    return controller.panel(), controller


class ProfileArtifactBrowserPanel:
    def __init__(self, context):
        import panel as pn

        self.context = context
        self._disposed = False
        self.subscriptions = []
        self.latest_artifact_id = None

        self.refresh_button = pn.widgets.Button(
            name="Find latest profile",
            button_type="primary",
            width=150,
            height=30,
            sizing_mode="fixed",
            margin=(0, 8, 0, 0),
        )

        self.render_button = pn.widgets.Button(
            name="Render with viewer",
            button_type="success",
            width=150,
            height=30,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
            disabled=True,
        )

        self.status = pn.pane.Markdown(
            "Create a profile artifact first, then click **Find latest profile**.",
            sizing_mode="stretch_width",
            margin=(0, 0, 12, 0),
        )

        self.viewer_slot = pn.Column(
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

        self.refresh_button.on_click(self.find_latest)
        self.render_button.on_click(self.render_latest)

        self.view = pn.Column(
            pn.pane.HTML(
                "<h2 style='margin: 0 0 12px 0;'>Profile Artifact Browser</h2>",
                sizing_mode="stretch_width",
            ),
            pn.Row(
                self.refresh_button,
                self.render_button,
                sizing_mode="stretch_width",
                margin=(0, 0, 12, 0),
            ),
            self.status,
            self.viewer_slot,
            sizing_mode="stretch_width",
            margin=(16, 24),
        )

        sub = context.events.subscribe(
            "artifact.created",
            self.on_artifact_created,
            owner_id="example.tutorial.profile_artifact_browser",
            owner_label="Tutorial Profile Artifact Browser",
            owner_kind="panel",
        )
        self.subscriptions.append(sub)

        self.find_latest()

    def find_latest(self, _event=None):
        try:
            refs = self.context.artifacts.find(
                type="example.tutorial.dataset_profile"
            )
        except Exception as exc:
            self.status.object = (
                "Could not search artifacts.\n\n"
                f"```text\n{exc}\n```"
            )
            self.render_button.disabled = True
            return

        if not refs:
            self.latest_artifact_id = None
            self.status.object = (
                "No `example.tutorial.dataset_profile` artifacts found yet. "
                "Create one from the Profile Artifact panel."
            )
            self.render_button.disabled = True
            self.viewer_slot.clear()
            return

        latest = refs[0]
        self.latest_artifact_id = latest.artifact_id
        self.render_button.disabled = False

        self.status.object = (
            f"Latest profile artifact: `{latest.artifact_id}`  \n"
            f"Dataset: `{latest.dataset_id}`"
        )

    def render_latest(self, _event=None):
        if not self.latest_artifact_id:
            self.status.object = "No profile artifact selected."
            return

        try:
            viewers = self.context.plugins.list_artifact_viewers(
                "example.tutorial.dataset_profile"
            )
        except Exception as exc:
            self.status.object = (
                "Could not list artifact viewers.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        if not viewers:
            self.status.object = (
                "No viewer is registered for "
                "`example.tutorial.dataset_profile`."
            )
            return

        viewer = viewers[0]

        try:
            result = viewer.viewer_factory(
                context=self.context,
                artifact_id=self.latest_artifact_id,
            )
        except Exception as exc:
            self.status.object = (
                "The artifact viewer failed.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        if isinstance(result, tuple):
            view, _controller = result
        else:
            view = result

        self.viewer_slot.clear()
        self.viewer_slot.append(view)

        self.status.object = (
            f"Rendered artifact `{self.latest_artifact_id}` using "
            f"`{viewer.id or viewer.title}`."
        )

    def on_artifact_created(self, topic, payload):
        if self._disposed:
            return

        if payload.get("type") != "example.tutorial.dataset_profile":
            return

        self.latest_artifact_id = payload.get("artifact_id")
        self.render_button.disabled = False
        self.status.object = (
            f"New profile artifact created: `{self.latest_artifact_id}`. "
            "Click **Render with viewer** to display it."
        )

    def panel(self):
        return self.view

    def dispose(self):
        if self._disposed:
            return

        self._disposed = True

        for sub in list(self.subscriptions):
            self.context.events.unsubscribe(sub)

        self.subscriptions.clear()
```

### What this demonstrates

The browser panel only knows this:

```python
refs = context.artifacts.find(type="example.tutorial.dataset_profile")
viewers = context.plugins.list_artifact_viewers("example.tutorial.dataset_profile")
```

It does not know how to format a dataset profile. The registered viewer owns that display logic.

That is the key reason artifact viewers are useful: they let producers of artifact types also provide the rendering logic, while generic platform UI can remain generic.

---

## Stage 18: save and restore small panel state

Panels may implement `get_state()` and `restore_state(state)` so the workspace can preserve small UI state.

```python
class StatefulTutorialPanel:
    def __init__(self, context):
        import panel as pn

        self.context = context
        self.limit = pn.widgets.IntInput(name="Preview rows", value=10, start=1, end=1000)
        self.output = pn.pane.Markdown("")
        self.view = pn.Column(self.limit, self.output)

    def panel(self):
        return self.view

    def get_state(self):
        return {
            "limit": int(self.limit.value),
        }

    def restore_state(self, state):
        if not state:
            return
        self.limit.value = int(state.get("limit", self.limit.value))

    def dispose(self):
        pass
```

### Persistence rule

Persist small JSON-safe UI state only.

Good:

```python
{"limit": 20, "selected_tab": "summary"}
```

Avoid:

```python
{"dataframe": df, "client": client, "large_array": array}
```

Put data in datasets/artifacts/services, and store only ids in panel state.


---

## Stage 19: add plugin settings schema

Settings schemas let plugins describe configurable values without manually building all settings UI themselves.

```python
def register(api) -> None:
    api.register_settings_schema(
        {
            "type": "object",
            "properties": {
                "base_url": {
                    "type": "string",
                    "title": "Base URL",
                    "default": "https://example.invalid",
                },
                "preview_rows": {
                    "type": "integer",
                    "title": "Default preview rows",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 1000,
                },
            },
        }
    )
```

Panels, services, and actions can then read plugin settings through the platform configuration or service construction path used by the plugin manager.

---

## Stage 20: create a complete tutorial plugin

This example combines the common pieces into one plugin file. It registers:

- a service
- a mapped summary panel
- an action
- an artifact viewer

```python
# astronomicAL/plugins/tutorial_plugin/plugin.py

from __future__ import annotations

from astronomicAL.platform.plugins import PluginManifest
from astronomicAL.platform.plugins.specs import ArtifactResult, InputSpec


manifest = PluginManifest(
    id="example.tutorial",
    name="Tutorial Plugin",
    version="0.1.0",
    description="A progressive example plugin for learning AstronomicAL plugins.",
    capabilities=[
        "panel",
        "action",
        "service",
        "artifact_viewer",
    ],
    tags=["example", "tutorial"],
)


def register(api) -> None:
    api.register_settings_schema(
        {
            "type": "object",
            "properties": {
                "base_url": {
                    "type": "string",
                    "title": "Base URL",
                    "default": "https://example.invalid",
                },
            },
        }
    )

    api.register_service(
        key="client",
        factory=create_toy_client,
        description="Example shared client for the tutorial plugin.",
        lazy=True,
        replace=False,
    )

    api.register_panel(
        id="summary",
        title="Tutorial: Summary",
        factory=create_summary_panel,
        description="Shows mapped dataset and focused-row information.",
        category="Examples",
        required_mappings=["record_id"],
        optional_mappings=["target_label"],
        default_layout={"x": 0, "y": 0, "w": 5, "h": 4},
    )

    api.register_action(
        id="profile_dataset",
        title="Tutorial: Profile Dataset",
        handler=profile_dataset,
        inputs=InputSpec(dataset=True, selection="optional", columns="optional"),
        outputs=["example.tutorial.profile"],
        run_in_job=True,
        category="Examples",
    )

    api.register_artifact_viewer(
        artifact_type="example.tutorial.profile",
        viewer_factory=create_profile_viewer,
        id="profile_viewer",
        title="Tutorial Profile Viewer",
        default=True,
    )


class ToyClient:
    def fetch(self, row_id):
        return {"row_id": row_id, "detail": f"Detail for {row_id}"}


def create_toy_client(context, **kwargs):
    return ToyClient()


def create_summary_panel(context, **kwargs):
    controller = SummaryPanel(context)
    return controller.panel(), controller


class SummaryPanel:
    def __init__(self, context):
        import panel as pn

        self.context = context
        self._disposed = False
        self.subscriptions = []

        self.output = pn.pane.Markdown(
            "Loading...",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.button = pn.widgets.Button(
            name="Create profile artifact",
            button_type="primary",
            width=170,
            height=30,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        )
        self.button.on_click(self.create_profile_artifact)

        self.view = pn.Column(
            pn.pane.HTML(
                "<h2 style='margin: 0 0 12px 0;'>Tutorial Summary</h2>",
                sizing_mode="stretch_width",
            ),
            pn.Row(
                self.button,
                sizing_mode="stretch_width",
                margin=(0, 0, 12, 0),
            ),
            self.output,
            sizing_mode="stretch_width",
            margin=(16, 24),
        )

        for topic in [
            "dataset.loaded",
            "dataset.active.changed",
            "dataset.mapping.updated",
            "selection.focus.changed",
            "selection.focus.cleared",
        ]:
            sub = context.events.subscribe(
                topic,
                self.on_runtime_changed,
                owner_id="example.tutorial.summary",
                owner_label="Tutorial Summary Panel",
                owner_kind="panel",
            )
            self.subscriptions.append(sub)

        self.refresh()

    def active_dataset_id(self):
        active_id = getattr(self.context.datasets, "active_id", None)

        if callable(active_id):
            try:
                return active_id()
            except Exception:
                return None

        return getattr(self.context.datasets, "active_dataset_id", None)

    def refresh(self):
        dataset_id = self.active_dataset_id()
        focus = self.context.selection.get_focus()

        if not dataset_id:
            self.output.object = "No active dataset. Load a dataset first."
            return

        try:
            source = self.context.datasets.get_source(dataset_id)
            id_col = self.context.datasets.get_mapping(dataset_id, "record_id")
            label_col = self.context.datasets.get_mapping(dataset_id, "target_label")
        except Exception as exc:
            self.output.object = (
                "Could not read the active dataset or its mappings.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        self.output.object = (
            f"**Dataset:** `{dataset_id}`  \n"
            f"**Rows:** `{source.row_count()}`  \n"
            f"**record_id column:** `{id_col}`  \n"
            f"**target_label column:** `{label_col}`  \n"
            f"**Focus:** `{getattr(focus, 'row_id', None)}`"
        )

    def on_runtime_changed(self, topic, payload):
        if not self._disposed:
            self.refresh()

    def create_profile_artifact(self, _event):
        dataset_id = self.active_dataset_id()

        if not dataset_id:
            self.output.object = "No active dataset. Load a dataset first."
            return

        try:
            source = self.context.datasets.get_source(dataset_id)
            payload = {
                "dataset_id": dataset_id,
                "rows": source.row_count(),
                "columns": source.columns(),
                "dtypes": source.dtypes(),
            }
        except Exception as exc:
            self.output.object = (
                "Could not create a dataset profile artifact.\n\n"
                f"```text\n{exc}\n```"
            )
            return

        artifact_id = self.context.artifacts.put(
            "example.tutorial.profile",
            payload,
            dataset_id=dataset_id,
        )

        self.context.events.publish(
            "artifact.created",
            {
                "artifact_id": artifact_id,
                "type": "example.tutorial.profile",
                "dataset_id": dataset_id,
                "origin": "example.tutorial.summary",
            },
        )

        self.output.object = (
            f"Created profile artifact `{artifact_id}`.\n\n"
            f"**Dataset:** `{dataset_id}`  \n"
            f"**Rows:** `{payload['rows']}`  \n"
            f"**Columns:** `{len(payload['columns'])}`"
        )

    def panel(self):
        return self.view

    def dispose(self):
        if self._disposed:
            return

        self._disposed = True

        for sub in list(self.subscriptions):
            self.context.events.unsubscribe(sub)

        self.subscriptions.clear()


def profile_dataset(context, request, cancel_token=None):
    dataset_id = request.dataset_id
    if not dataset_id:
        active_id = getattr(context.datasets, "active_id", None)
        dataset_id = active_id() if callable(active_id) else None
    if not dataset_id:
        raise ValueError("No dataset selected.")

    source = context.datasets.get_source(dataset_id)
    columns = request.columns or source.columns()

    if cancel_token and cancel_token.cancelled():
        return None

    return ArtifactResult(
        type="example.tutorial.profile",
        payload={
            "dataset_id": dataset_id,
            "row_count": source.row_count(),
            "columns": columns,
            "dtypes": source.dtypes(),
        },
        dataset_id=dataset_id,
        row_ids=request.row_ids,
        params=request.params,
    )


def create_profile_viewer(context, artifact_id=None, **kwargs):
    import panel as pn

    if not artifact_id:
        return pn.pane.Markdown("No profile artifact selected."), None

    payload = context.artifacts.get(artifact_id)
    return pn.Column(
        pn.pane.Markdown(f"## Profile artifact `{artifact_id}`"),
        pn.pane.Markdown(f"```python\n{payload!r}\n```"),
    ), None
```

---

## Migration guide for old AstronomicAL features

When moving existing AstronomicAL functionality into plugins, use this checklist.

### Replace global or legacy state

| Old habit | New plugin-system pattern |
| --- | --- |
| Read or mutate global config state for current row | `context.selection.get_focus()` / `set_focus(...)` |
| Store selected rows in panel instance only | `context.selection.set_selection_set(...)` |
| Store computed outputs in config/service globals | `context.artifacts.put(...)` |
| Pass data directly between panels | Publish event + store artifact/dataset/selection state |
| Load full dataframe for simple metadata | `context.datasets.get_source(...).columns()` / `row_count()` |
| Assume `id`, `ra`, `dec`, `class` column names | Declare required/optional mappings |
| Do long work in panel constructor | Create lightweight UI, then submit `context.jobs` work |
| Keep API client on a panel if several panels need it | Register `context.services` service |
| Rebuild UI state manually after reload | Use workspace persistence and `get_state()` |

### Decide where outputs belong

Ask what the output is:

- Is it a new source-like table users should continue working with? Register a dataset.
- Is it a derived result, model output, cutout, spectrum, score table, report, or preview? Store an artifact.
- Is it a notification that something happened? Publish an event.
- Is it current UI/workflow state about rows? Use selection.
- Is it a live client/session/connection? Use a service.

---

## Common plugin author mistakes

### Mistake: using events as data storage

Events should carry identifiers and metadata, not large payloads.

```python
# Avoid
context.events.publish("example.scores", huge_dataframe)

# Prefer
artifact_id = context.artifacts.put("example.scores", scores, dataset_id=dataset_id)
context.events.publish("artifact.created", {"artifact_id": artifact_id, "type": "example.scores"})
```

### Mistake: direct panel-to-panel calls

```python
# Avoid
other_panel.update(row_id)

# Prefer
context.selection.set_focus(dataset_id=dataset_id, row_id=row_id, origin="my.panel")
```

### Mistake: hard-coded column names

```python
# Avoid
row_id = row["source_id"]

# Prefer
id_col = context.datasets.get_mapping(dataset_id, "record_id")
row_id = row[id_col]
```

### Mistake: materializing large datasets accidentally

```python
# Avoid for large datasets
df = context.datasets.get_df(dataset_id)

# Prefer
source = context.datasets.get_source(dataset_id)
preview = source.head(100, columns=[id_col, score_col])
```

### Mistake: forgetting cleanup

Any panel that subscribes to events, launches jobs, registers watchers, or starts callbacks should implement idempotent `dispose()`.

---

## Recommended development order for a real plugin

1. Create `manifest` and empty `register(api)`.
2. Register one tiny panel with no dependencies.
3. Read the active dataset and show a waiting state when no dataset is loaded.
4. Declare required and optional semantic mappings.
5. Publish focus or selection through `context.selection`.
6. Subscribe to platform events and clean up in `dispose()`.
7. Store derived outputs as artifacts.
8. Add actions for reusable operations.
9. Move slow operations into jobs.
10. Register services for shared clients or runtime capabilities.
11. Add artifact viewers if the plugin produces reusable outputs.
12. Add workflow builders only after the component panels/actions are stable.
13. Add persistence state only for small JSON-safe UI state.

---

## Quick reference: contribution types

```python
api.register_panel(...)
api.register_action(...)
api.register_dataframe_action(...)
api.register_service(...)
api.register_artifact_viewer(...)
api.register_workflow(...)
api.register_settings_schema(...)
```

---

## Quick reference: common event topics

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
plugin.enabled
plugin.disabled
plugin.error
workflow.started
workflow.stage.changed
workflow.completed
workflow.failed
```

Custom topics should be namespaced:

```text
example.tutorial.something_changed
astro.desi.spectrum.loaded
active_learning.query.created
annotations.note.created
```

---

## Final rule

When in doubt, ask: “Am I adding generic runtime infrastructure, or am I adding behaviour?”

Generic runtime infrastructure belongs in `astronomicAL/platform/`.

Behaviour belongs in a plugin.