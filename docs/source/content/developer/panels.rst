Creating Panels
===============

Panel Registration
------------------

Register panels through :code:`api.register_panel(...)`. The registration owns
the user-facing title, mapping and dependency requirements, default layout and
workspace-persistence policy.

.. code-block:: python

   api.register_panel(
       id="example",
       title="Example Panel",
       factory=create_panel,
       category="Examples",
       required_mappings=["record_id"],
       optional_mappings=["target_label"],
       state_version=1,
       persist_layout=True,
       persist_state=True,
   )

Registration IDs are automatically namespaced by the plugin ID unless they are
already fully qualified.

Panel Factory
-------------

A panel factory receives :code:`context` and should accept :code:`**kwargs` so
the platform can pass lifecycle metadata without breaking older factories.

The platform may also provide values such as :code:`instance_id`,
:code:`restore_state` and :code:`restore_metadata`.

A normal factory should construct the controller and return a
:code:`(view, controller)` pair:

.. code-block:: python

   def create_panel(context, instance_id=None, **kwargs):
       controller = ExamplePanel(
           context=context,
           instance_id=instance_id,
       )
       return controller.panel(), controller

For panels with :code:`persist_state=True`, PluginManager applies saved state to
the returned controller after the factory completes. Do not normally restore the
same state manually inside the factory.

Controller and View
-------------------

The controller owns panel-specific lifecycle state such as:

* event subscriptions;
* Param watchers;
* periodic callbacks;
* panel-owned job handles;
* small persistent UI state;
* child controllers.

The returned view is the object placed in the workspace.

A simple static panel can return :code:`(view, None)` when it owns no resources
that require cleanup or persistence.

Panel Opening
-------------

The workspace-facing opener is :code:`context.plugins.open_panel(...)`.

Opening a panel first creates a temporary loading tile. When a JobManager is
available, panel construction runs through a background panel-open job and the
completed view is then installed into the workspace.

Each opening receives a separate workspace instance ID, so multiple instances
of the same registered panel can coexist.

.. caution::

   Panel construction should remain lightweight even though the current opener
   uses a background job. Do not start remote queries, large scans or expensive
   computations merely to construct the initial UI.

Mappings
--------

Declare semantic requirements on the panel registration.

Required mappings gate construction of the real panel until they are resolved.
Optional mappings do not block opening and should only enable additional
behaviour when available.

.. code-block:: python

   api.register_panel(
       id="coordinates",
       title="Coordinate Inspector",
       factory=create_coordinates_panel,
       required_mappings=[
           "record_id",
           "coords.ra",
           "coords.dec",
       ],
       optional_mappings=["target_label"],
   )

Do not construct a half-working domain panel and then duplicate the platform's
required-mapping flow inside the plugin.

Dependencies and Services
-------------------------

A panel can also declare required or optional Python packages and service keys:

.. code-block:: python

   api.register_panel(
       id="remote_viewer",
       title="Remote Viewer",
       factory=create_remote_viewer,
       uses_services=["astro.example.client"],
       requires=["astropy>=6"],
       optional_requires=["mocpy"],
   )

Missing required packages prevent construction. Optional requirements can be
used to enable additional behaviour without making the complete panel
unavailable.

Event Subscriptions
-------------------

Keep every subscription handle owned by the controller and provide ownership
metadata for diagnostics.

.. code-block:: python

   subscription = context.events.subscribe(
       "selection.focus.changed",
       self._on_focus_changed,
       owner_id=self.instance_id,
       owner_label="Example Panel",
       owner_kind="panel",
   )

   self._subscriptions.append(subscription)

Event callbacks should normally use the event as a refresh signal and then read
the current canonical state from :code:`context.selection`,
:code:`context.datasets` or another platform service.

UI Thread
---------

Panel and Bokeh objects should only be mutated from the document/UI thread once
the view is live.

A workspace panel factory can currently be constructed in the panel-open worker
before the view has been mounted. Avoid assuming :code:`pn.state.curdoc` is a
live document during construction.

If a controller needs a document-bound periodic callback or other live-session
resource, attach it after the panel is mounted, for example in response to the
matching :code:`plugin.panel.opened` event.

Jobs submitted from normal UI callbacks can use JobManager
:code:`on_done`/:code:`on_error` handlers; the manager schedules those callbacks
onto the Panel document captured at submission time.

Do not mutate visible widgets from inside the worker function itself.

Stale Results
-------------

Asynchronous completion may arrive after the user changes dataset, focus,
settings or request parameters.

Capture enough request identity to detect this, for example:

* dataset ID;
* row ID;
* request generation;
* service or query parameters.

Before replacing visible state, verify that the completion still belongs to the
current request.

Cancellation helps reduce wasted work but is not a substitute for stale-result
checks.

Persistence
-----------

A restorable controller can implement:

.. code-block:: python

   state_version = 2

   def get_state(self):
       return {
           "selected_tab": self.tabs.active,
           "limit": int(self.limit.value),
       }

   def restore_state(self, state):
       self.tabs.active = int(state.get("selected_tab", 0))
       self.limit.value = int(state.get("limit", 20))

:code:`snapshot_state()` is also recognised as a state getter, and
:code:`persistence_version` is accepted as an alternative controller version
attribute.

Persist only small JSON-safe UI state. Keep datasets, artifacts, clients and
large arrays in the platform services designed for them.

The panel registration also provides:

* :code:`state_version`;
* :code:`persist_layout`;
* :code:`persist_state`;
* :code:`restore_policy`.

Setting :code:`persist_layout=False` makes the workspace panel transient and
excludes it from saved panel snapshots.

:code:`persist_state=False` prevents PluginManager from applying saved
controller state when the panel is reconstructed.

Disposal
--------

:code:`dispose()` must be idempotent and release every resource owned by the
controller.

Typical cleanup includes:

* unsubscribe event handles;
* stop periodic callbacks;
* remove Param watchers;
* request cancellation of panel-owned jobs;
* close panel-owned resources;
* dispose child controllers.

.. code-block:: python

   def dispose(self):
       if self._disposed:
           return
       self._disposed = True

       for subscription in self._subscriptions:
           self.context.events.unsubscribe(subscription)
       self._subscriptions.clear()

       if self._job_handle is not None:
           self._job_handle.cancel()
           self._job_handle = None

When a workspace panel is removed, WorkspaceManager calls :code:`dispose()` on
the controller and, where separate, on a disposable view. Plugin disablement
also closes the plugin's open workspace panels.
