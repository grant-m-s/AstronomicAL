Workspace and Persistence
=========================

Panel Specifications and Instances
----------------------------------

A panel registration describes a capability contributed by a plugin.

A panel instance is one live controller and view created from that registration.

The Workspace Manager owns those live instances and their placement in the
application grid.

Panel Records
-------------

The workspace records enough information to identify and restore an instance,
including its plugin, panel registration, geometry, state version, and restore
metadata.

Geometry
--------

The workspace owns panel geometry.

A plugin's default layout is a starting size or placement hint for a newly
opened panel. It must not overwrite the geometry restored from a saved layout.

New panels are placed into available workspace space where possible.

Panel State
-----------

A controller may expose a small persistence interface such as
:code:`get_state()` and :code:`restore_state()`.

Persisted panel state should contain serialisable user/workflow state, not live
runtime objects such as clients, threads, or caches.

Saving and Loading Layouts
--------------------------

Workspace persistence can store information such as:

* dataset registrations and mappings;
* enabled plugins;
* panel instances;
* panel geometry;
* supported controller state.

Normal layout loading reconciles the saved workspace with the current one:
matching instances can be kept, missing instances opened, and surplus instances
closed.

Missing Plugins or Panels
-------------------------

A missing plugin should not make an otherwise valid workspace unusable.

The platform can preserve the missing panel's place with a placeholder so the
rest of the layout still loads.

Disposal
--------

Closing a panel must dispose its controller before the workspace record is
removed. Disabling a plugin also closes panel instances owned by that plugin.
