Architecture Overview
=====================

What the Platform Does
----------------------

AstronomicAL separates shared application infrastructure from feature code.

The platform owns common runtime state and services. Plugins use those services
to provide visualisation, astronomy tools, machine learning, annotations, data
integrations, and other workflows.

A useful rule is:

* if unrelated plugins need a capability to work together, it probably belongs
  in the platform;
* if it performs a scientific or user workflow, it probably belongs in a
  plugin.

AppContext
----------

:code:`AppContext` is the main dependency boundary between the application and
plugins. It provides access to shared services such as:

* datasets and semantic mappings;
* focused records and selections;
* artifacts and events;
* background jobs;
* services registered by plugins;
* the workspace and persistence layer;
* plugin management and runtime status.

Plugins should use these shared services instead of creating their own copies of
application-wide state.

Ownership
---------

Each kind of runtime state should have one clear owner.

For example, the Dataset Manager owns loaded datasets, the Selection Manager
owns the current selection, and the Workspace Manager owns live panel
instances. This avoids different panels maintaining conflicting versions of the
same state.

How Plugins Fit In
------------------

Plugins register capabilities through the plugin API. Common contributions
include panels, actions, services, workflows, and artifact viewers.

Reusable operations should normally be exposed as actions rather than hidden
inside panel callbacks. Results that need to be shared with other features
should use datasets, artifacts, events, or registered services.

Threading
---------

UI callbacks and event delivery are synchronous. Expensive or blocking work
belongs in the Job Manager. Work that updates the visible interface must return
to the correct Panel/Bokeh document before changing UI state.

Startup
-------

The host creates the platform services before discovering and enabling plugins.
Plugins are enabled before saved layouts are restored so persisted plugin panels
can be reconstructed where possible.
