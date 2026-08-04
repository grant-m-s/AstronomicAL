Using Events
============

Topic Names
-----------

Use stable dotted names for plugin-specific events:

.. code-block:: text

   example.plugin.started
   example.plugin.finished

Prefer an existing canonical platform topic when it already describes the
change, such as :code:`dataset.mapping.updated`,
:code:`selection.focus.changed` or :code:`artifact.created`.

Custom topics should normally be namespaced by the plugin or workflow that owns
their meaning.

Publishing
----------

Publish lightweight notifications through :code:`context.events`:

.. code-block:: python

   context.events.publish(
       "example.result.created",
       {
           "artifact_id": artifact_id,
           "dataset_id": dataset_id,
       },
   )

Store large or reusable data in datasets or artifacts and publish identifiers
that let subscribers find the current result.

Subscribing
-----------

Register callbacks with :code:`context.events.subscribe(...)`.

Provide ownership metadata so Event Monitor and runtime diagnostics can identify
the subscriber:

.. code-block:: python

   subscription = context.events.subscribe(
       "example.result.created",
       on_result_created,
       owner_id=panel_id,
       owner_label="Example Results",
       owner_kind="panel",
   )

Panels should normally read canonical platform state again after receiving an
event rather than treating the event payload as the only source of truth.

Payloads
--------

Payloads should remain small and identifier-based. Dictionaries containing IDs,
counts, status values and other lightweight metadata are a good default.

Avoid using the EventBus as storage for complete DataFrames, images, model
objects or other large products.

Callback Duration
-----------------

Event delivery is synchronous. A slow callback delays later subscribers and the
publisher itself.

Callbacks should therefore perform only quick state updates, lightweight reads
or job submission. Expensive IO, computation or remote access belongs in
:code:`context.jobs`.

Callback errors are recorded by the EventBus diagnostics and do not prevent
later subscribers from being called.

Diagnostics
-----------

The EventBus records current subscriptions, active publishes, publish timings,
slow callbacks and callback errors. When tracing is enabled it can also retain a
bounded trace of recent publishes.

Use Event Monitor or Runtime Diagnostics to inspect those records rather than
adding diagnostic wildcard subscriptions to normal plugins.

Cleanup
-------

Store every subscription handle owned by a panel or controller and unsubscribe
it during idempotent :code:`dispose()`.

.. code-block:: python

   def dispose(self):
       if self._disposed:
           return
       self._disposed = True

       for subscription in self._subscriptions:
           self.context.events.unsubscribe(subscription)

       self._subscriptions.clear()