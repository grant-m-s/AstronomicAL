Developing Plugins
==================

AstronomicAL is a plugin host. Plugins add panels, actions, workflows,
artifact viewers and services; the platform supplies shared datasets,
selection, navigation, events, jobs, artifacts, workspace management and
persistence. Domain-specific behaviour belongs in plugins rather than in
``astronomicAL/platform/``.

Start with :doc:`first_plugin` for a progressive, bundled/development-plugin
tutorial. If you are creating a **separately distributed plugin**, use the
`AstronomicAL Hello World template
<https://github.com/grant-m-s/AstronomicAL-Hello-World>`_ and read
:doc:`manifests`, :doc:`package_structure` and :doc:`packaging`.

The :doc:`plugin_api` page is the canonical registration contract, while
:doc:`app_context` explains ownership of shared runtime state. The other
pages describe each extension surface in more detail. The tutorials and API
pages refer to the implementation at commit ``90a912c``; when developing
against a later commit, verify changed signatures in the source code.

.. note::

   A bundled/development plugin normally exposes a Python
   ``PluginManifest`` and ``register(api)``. A managed community plugin
   distributed as a ``.alplugin`` uses a static
   ``astronomical-plugin.json`` and ``register(api)``. Do not copy the
   bundled manifest example into a managed archive without adapting the
   packaging format.

.. toctree::
   :maxdepth: 1

   first_plugin
   package_structure
   manifests
   plugin_api
   app_context
   panels
   dataset_programming
   actions
   workflows
   events
   jobs
   artifacts
   services
   mappings
   persistence
   packaging
