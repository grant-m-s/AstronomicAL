.. _services-concept:

Services
========

What Belongs in a Service?
--------------------------

A service is a live, usually non-serializable runtime capability shared through
the application context.

Examples include:

* an authenticated archive client;
* a SAMP bridge;
* an image asset resolver;
* a machine-learning recipe registry;
* a resource sampler.

Lazy Services
-------------

A service factory can be created only when the service is first requested. This
keeps startup fast and avoids opening clients or connections that are never
used.

Ownership
---------

Plugin services are installed with the owning plugin ID. When the plugin is
disabled, its installed services are removed and initialized service objects are
disposed where possible using :code:`dispose()`, :code:`close()` or
:code:`shutdown()`.

The platform can also register host-owned services, so not every entry in the
service registry belongs to a plugin.

Services, Datasets and Artifacts
--------------------------------

Use a service for a live capability.

Use a dataset for source or working tabular data.

Use an artifact for a generated result.

.. caution::

   The service registry is not a replacement for global configuration or a
   generic dictionary of plugin state. Persist small UI state through workspace
   persistence and keep reusable data products in datasets or artifacts.
