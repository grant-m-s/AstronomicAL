.. _events-concept:

Events
======

What Events are For
-------------------

Events announce that something changed.

Examples include:

* :code:`dataset.active.changed`;
* :code:`dataset.mapping.updated`;
* :code:`selection.focus.changed`;
* :code:`artifact.created`;
* plugin-specific workflow events.

Custom plugin topics should use a plugin or workflow namespace.

Events are not Data Storage
---------------------------

An event should normally contain identifiers and small pieces of metadata.

.. code-block:: python

   {
       "dataset_id": "catalogue",
       "row_id": "source-42",
       "origin": "core.record_browser",
   }

Large tables, images and model objects should be stored as datasets, artifacts
or services and referenced by identifier.

Delivery
--------

Event callbacks run synchronously. A slow callback delays later subscribers, so
expensive work should be submitted as a job.

A callback exception is recorded by the EventBus and does not stop delivery to
later callbacks.

Diagnostics
-----------

The EventBus records subscription metadata, active publishes, publish timings,
slow callbacks and callback errors. It can also keep an optional trace of recent
raw publishes when tracing is enabled.

The Event Monitor plugin reads the diagnostic information to show event health,
recent publish activity, topic traffic and current subscriptions without
modifying the EventBus.
