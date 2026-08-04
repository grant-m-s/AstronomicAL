Plugin Persistence
==================

Panel State
-----------

Persistent panels can expose small controller state through
:code:`get_state()` or :code:`snapshot_state()`.

Return a dictionary containing only values that represent restorable UI or
workflow state:

.. code-block:: python

   def get_state(self):
       return {
           "x_column": self.x_column.value,
           "show_labels": bool(self.show_labels.value),
       }

The workspace applies a JSON-safety conversion before writing the snapshot, but
plugins should still return deliberately JSON-safe state rather than relying on
that conversion to serialize arbitrary objects.

Registration Controls
---------------------

Panel persistence is configured when the panel is registered:

.. code-block:: python

   api.register_panel(
       id="example",
       title="Example",
       factory=create_panel,
       state_version=2,
       persist_layout=True,
       persist_state=True,
       restore_policy="best_effort",
   )

:code:`persist_layout=False` makes the workspace panel transient, so the panel
instance is not included in saved panel snapshots.

:code:`persist_state=False` prevents PluginManager from applying saved
controller state when a new panel instance is constructed.

State Versions
--------------

Increment :code:`state_version` when the meaning or structure of saved state
changes.

The saved panel snapshot records the state version. A controller can also expose
:code:`state_version` or :code:`persistence_version`; WorkspaceManager uses that
value when snapshotting controller state.

.. caution::

   The current persistence layer records state versions but does not
   automatically migrate, reject or transform panel state when versions differ.
   Version-aware compatibility must currently be implemented by the plugin.

Restoring
---------

During workspace restore, the saved state and restore metadata are passed into
the panel-opening path.

The panel factory may receive:

.. code-block:: python

   def create_panel(
       context,
       restore_state=None,
       restore_metadata=None,
       **kwargs,
   ):
       ...

For panels with :code:`persist_state=True`, PluginManager also calls the
controller's :code:`restore_state(state)` after the factory returns.

A normal panel should therefore let :code:`restore_state()` own state
application and avoid applying the same saved state twice inside the factory.

.. code-block:: python

   def restore_state(self, state):
       if not state:
           return

       self.x_column.value = state.get(
           "x_column",
           self.x_column.value,
       )
       self.show_labels.value = bool(
           state.get(
               "show_labels",
               self.show_labels.value,
           )
       )

If construction genuinely depends on saved metadata, the factory can inspect
:code:`restore_state` or :code:`restore_metadata`, but controller restoration
must remain safe when PluginManager subsequently invokes
:code:`restore_state()`.

First Asynchronous Request
--------------------------

Do not start an asynchronous request in the constructor if it depends on state
that may still be restored.

Prefer this order:

#. construct lightweight widgets and controller state;
#. let PluginManager call :code:`restore_state()`;
#. attach live-session callbacks after the panel is opened;
#. start the first request using the restored values.

This avoids launching one request with defaults and another immediately after
workspace restoration.

Best-Effort Restore
-------------------

Panel registrations currently default to:

.. code-block:: python

   restore_policy="best_effort"

The policy is retained in workspace panel metadata. The current restore path
does not automatically interpret it as a state-migration strategy, so
:code:`restore_state()` should itself be tolerant of compatible older state.

Good restore methods should:

* ignore unknown fields;
* provide defaults for newly added fields;
* tolerate missing optional values;
* validate values before assigning them to widgets;
* avoid failing the complete workspace because one optional field is stale.

For a structural state change, inspect the saved version supplied through
:code:`restore_metadata` when custom migration logic is required.

Do Not Persist
--------------

Do not put live or large runtime objects in panel state, including:

* Panel or Bokeh widgets;
* DataFrames or dataset contents;
* futures, job handles or cancellation tokens;
* network clients or sessions;
* event subscription handles;
* callbacks or Param watchers;
* large arrays;
* arbitrary trained model objects.

Keep those objects in their appropriate platform boundary and persist only the
identifier or small setting required to reconstruct the panel.

For example, store an artifact ID instead of the artifact payload, or a dataset
ID instead of a DataFrame.

Workspace Scope
---------------

Panel state is only one part of a workspace snapshot. The workspace persistence
service separately records persistent panel instances and geometry, dataset
metadata and mappings, plugin information, and focus/selection state.

A panel should not duplicate those platform-owned values inside its own state
unless the value has panel-specific meaning.
