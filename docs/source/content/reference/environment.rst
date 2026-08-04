Environment Variables and Filesystem Locations
==============================================

Debugging Variables
-------------------

.. code-block:: text

   ASTRONOMICAL_DEBUG_BOOT
   ASTRONOMICAL_DEBUG_PLUGINS
   ASTRONOMICAL_DEBUG_MAPPING
   ASTRONOMICAL_DEBUG_PERSISTENCE

Set a value of :code:`1` before starting AstronomicAL to enable the corresponding
diagnostics.

Plugin Search Path
------------------

:envvar:`ASTRONOMICAL_PLUGIN_PATH` adds one or more local plugin locations.

.. todo::

   Confirm the path separator behaviour on Windows and Unix systems.

Workspace Files
---------------

Workspace JSON contains layouts, plugin IDs, dataset registrations, mappings and
panel state.

Cache and Artifact Files
------------------------

Parquet caches, model sidecars and remote products may be written outside the
workspace file.

.. todo::

   Document the final default cache, workspace, artifact and user-plugin
   directories for each operating system.

Cleaning Local State
--------------------

Do not delete a cache directory until checking whether it contains the only copy
of a model, imported dataset or remote artifact.
