.. _workspaces-concept:

Workspaces and Layouts
======================

Workspace Composition
---------------------

A workspace contains panel instances arranged on a responsive grid. Persistent
plugin-panel instances record which plugin and panel registration created them,
together with their instance-specific state.

Panel Instances
---------------

Two instances of the same registered panel can have different state. For
example, two Scatter Plot panels can show different axes.

Each opened plugin panel receives its own workspace instance ID.

Saving a Layout
---------------

A saved workspace may include:

* enabled and required plugin information;
* persistent open panel instances;
* panel positions and sizes for the responsive grid;
* JSON-safe controller state and panel metadata;
* dataset IDs, names, metadata and column mappings;
* the active dataset ID;
* focused-record and active-selection state.

Dataset contents themselves are not embedded in the workspace file.

Restoring a Layout
------------------

Dataset metadata and mappings are restored or queued first, required plugins are
discovered and enabled where possible, and panel instances are then recreated.
Focus and selection state are restored after the workspace panels.

User-facing layout loading reconciles the current workspace: matching panels
are retained, surplus panels are closed, missing panels are opened and the saved
geometry is applied.

A mapping-gated panel waits until its dataset exists and its required mappings
can be resolved.

Missing Plugins
---------------

If a saved panel cannot be recreated because its plugin or registration is
unavailable, AstronomicAL can place a missing-panel placeholder in the saved
workspace slot and report the restore issue.

.. note::

   A workspace file does not contain live clients, running jobs, complete
   DataFrames, dataset source contents or arbitrary Python objects. Panel state
   must be JSON-safe.
