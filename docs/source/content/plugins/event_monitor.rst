Event Monitor
=============

Overview
--------

:code:`core.event_monitor` is a diagnostics plugin for understanding how panels
and plugins communicate through the platform EventBus. It is mainly intended
for development, support and investigating linked views that do not update as
expected.

What You Do
-----------

1. Open **Event Monitor** while reproducing the behaviour you want to inspect.
2. Check the health summary for callback errors, slow callbacks or stalled
   publishes.
3. Filter by topic group or search for a topic, owner, callback or payload
   summary.
4. Inspect recent activity to confirm that expected events were published.
5. Check topic traffic for unusually frequent or slow publishes.
6. Inspect subscriptions to confirm which owners and callbacks are registered.

This can reveal whether an event was never published, used an unexpected topic,
triggered a slow or failing callback, or is still connected to a panel that
should have been disposed.

.. TODO: Add an image of the health summary and recent event activity.

Panel
-----

**Event Monitor**

The panel contains views for:

* overall event health, throughput and publish latency;
* stalled publishes, callback errors and slow callbacks;
* the most active event topics;
* recent completed publishes and payload summaries;
* active subscriptions and their owners.

Controls
--------

The display can be filtered by topic group and search text. The history window
controls how much recent diagnostic history is loaded. Live updates refresh the
panel every two seconds and can be paused, while **Refresh now** requests an
immediate snapshot from the EventBus.

Typical Use Cases
-----------------

Use this plugin when:

* a linked panel does not respond to focus, selection or dataset changes;
* one action appears to trigger the same update more than once;
* an event topic may be misspelt or namespaced incorrectly;
* a disposed panel still appears in the subscription list;
* an event callback is failing or taking too long;
* an EventBus publish appears to be stalled.

What It Does Not Do
-------------------

The monitor is read-only. It does not enable tracing, publish test events,
modify subscriptions, replay application state or replace general runtime and
job diagnostics.

.. caution::

   Event diagnostics may contain dataset, artifact, record or plugin
   information, and callback errors may include traceback text. Review copied
   diagnostics before sharing them outside the project.