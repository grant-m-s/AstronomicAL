Plugin API
==========

The :class:`PluginAPI` passed to :code:`register(api)` is the supported
author-facing registration surface for plugin contributions.

Plugins should register through this object rather than mutating
:class:`PluginManager` registries directly.

Registering a Panel
-------------------

.. code-block:: python

   api.register_panel(
       id="panel",
       title="Example",
       factory=create_panel,
       category="Examples",
       required_mappings=["record_id"],
   )

Panel registrations can also declare optional mappings, services, produced
artifact types, Python requirements, default layout, default open arguments and
workspace-persistence settings.

Registering an Action
---------------------

.. code-block:: python

   api.register_action(
       id="calculate",
       title="Calculate Result",
       handler=calculate_action,
       run_in_job=True,
   )

Actions can declare an :class:`InputSpec`, output artifact types, parameter and
settings schemas, Python requirements and a custom job-key function.

The API also provides the :code:`api.action(...)` decorator as an alternative
registration style.

DataFrame Actions
-----------------

:code:`api.register_dataframe_action(...)` and
:code:`api.dataframe_action(...)` are convenience adapters for operations that
genuinely want a pandas DataFrame.

The adapter resolves the selected dataset and rows, materializes the dataset
through :code:`context.datasets.get_df(...)`, and passes only the arguments
accepted by the handler.

.. caution::

   DataFrame actions are convenient but are not the preferred path for
   source-backed large datasets. Use a normal action with DatasetSource-aware or
   batched access when full pandas materialization is unnecessary.

Other Contribution Types
------------------------

A plugin may also register:

* workflows with :code:`api.register_workflow(...)`;
* services with :code:`api.register_service(...)`;
* artifact viewers with :code:`api.register_artifact_viewer(...)`;
* one plugin settings schema with :code:`api.register_settings_schema(...)`.

The API also exposes :code:`get_setting(...)` and :code:`set_setting(...)` for
plugin-scoped runtime settings.

Namespacing
-----------

Local panel, action and workflow IDs are automatically joined to the plugin ID.

For a plugin named :code:`example.plugin`:

.. code-block:: python

   api.register_action(
       id="calculate",
       title="Calculate",
       handler=calculate,
   )

becomes:

.. code-block:: text

   example.plugin.calculate

The same automatic namespacing applies to service keys and explicit artifact
viewer IDs.

Already-qualified IDs beginning with the current plugin ID are left unchanged.

Registration IDs may contain letters, numbers, dots, underscores and hyphens,
and must begin with a letter or number.

Artifact Types
--------------

Artifact types are validated but are **not** automatically namespaced by the
Plugin API.

Custom artifact types should therefore normally include their own stable plugin
or domain namespace:

.. code-block:: text

   example.plugin.result
   astro.spectra

This avoids collisions between unrelated plugins.

Ownership
---------

Each panel, action, workflow, service and artifact-viewer registration records
the plugin that created it.

When a plugin fails during enablement or is disabled, PluginManager removes that
plugin's registrations. Installed services are removed and disposed where
possible, open plugin panels can be closed, and PluginManager-managed jobs can
be cancelled.

Settings schemas are stored on the plugin record rather than in a separate
global contribution registry.

Duplicate Registration
----------------------

Panel, action and workflow IDs must be unique. Registering the same canonical ID
twice raises :class:`PluginRegistrationError`.

Service keys must also be unique unless the service registration explicitly
uses:

.. code-block:: python

   api.register_service(
       key="client",
       factory=create_client,
       replace=True,
   )

Use service replacement deliberately; normal plugin services should prefer
their own namespaced keys.

Artifact viewers are grouped by artifact type. Multiple viewers may exist for
one artifact type, but two viewers with the same explicit viewer ID are an
error. Viewer ordering uses the :code:`default` flag and :code:`priority`.

Settings Schemas
----------------

:code:`api.register_settings_schema(schema)` stores the plugin-level settings
schema on the current plugin record.

Calling it again replaces the previous schema for that plugin rather than
creating a second independently named contribution.

Registration-Time Rules
-----------------------

Keep :code:`register(api)` declarative and fast.

Registration should describe contributions, not:

* open network connections;
* run dataset scans;
* construct workspace panels;
* submit background jobs;
* create long-lived service instances directly.

Services should be registered with factories, panels with panel factories, and
actions with handlers. The platform controls when those contributions are
constructed or executed.
