Platform Architecture
=====================

The platform is the shared runtime used by AstronomicAL plugins. It owns the
application-wide services that plugins use to load data, communicate, run work,
manage panels, and restore a workspace.

Most user-facing or domain-specific features should be implemented as plugins,
not added directly to :code:`astronomicAL/platform`.

This section is intended for contributors working on the platform itself or
building plugins that need to understand its contracts.

.. toctree::
   :maxdepth: 1

   overview
   datasets
   selection_navigation
   artifacts_services_events
   runtime
   workspace
   plugin_manager
   compatibility
