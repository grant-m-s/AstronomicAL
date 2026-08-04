Sharing Data and State Between Plugins
======================================

Plugins should not communicate by reaching into each other's panels or private
state. The platform provides several shared mechanisms instead.

Choosing the Right Mechanism
----------------------------

Use an artifact
    when a result should be reusable, inspectable, or persisted.

Use a service
    when plugins need access to a live object such as a registry, cache,
    session, client, or model catalogue.

Use an event
    when other components only need to know that something happened.

Use an action
    when a reusable user or workflow operation should be callable outside the
    panel that presents it.

Artifacts
---------

Artifacts are typed results stored by the platform.

They can include dataset association, parameters, row identity, provenance, and
references to durable payloads. Examples include selections, model outputs,
predictions, and workflow results.

Artifacts are preferable to panel-local dictionaries when another plugin may
need to consume the result later.

Services
--------

The Service Registry stores live runtime objects.

Services are useful for objects that do not naturally behave like saved
results, such as API clients, strategy registries, caches, or long-lived
sessions.

Services have owners and can be removed when the owning plugin is disabled.

Events
------

The Event Bus provides lightweight notifications between otherwise independent
components.

Delivery is synchronous. Subscribers should therefore do only small amounts of
work. Expensive processing should be submitted to the Job Manager instead.

Callbacks are isolated so one failing subscriber does not prevent later
subscribers from running.

Events can also carry ownership and tracing information used by runtime
diagnostics.
