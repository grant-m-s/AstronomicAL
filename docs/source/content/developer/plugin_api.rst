Plugin API and Authoring Contract
=================================

This is the canonical reference for the plugin registration surface at
AstronomicAL commit ``90a912c``. It consolidates the former root-level
``PLUGIN_CONTRACT.md`` and the duplicate developer
``plugin_contract.rst``. It describes the *implemented* interfaces in
``astronomicAL/platform/plugins/api.py``, ``specs.py`` and ``manifest.py``.

A plugin is a contributor of behaviour; the host owns cross-plugin runtime
state. Normal plugins belong under ``astronomicAL/plugins/`` (bundled), a
configured development search path, or an independently distributed package.
They do **not** belong in ``astronomicAL/platform/plugins/``: that directory
implements the extension framework itself.

Author-facing interfaces
------------------------

A plugin's runtime entry point is ``register(api)``. The argument is a
``PluginAPI`` instance scoped to the plugin's identifier. Registration is
responsible for describing contributions, not for constructing live panels
or running long tasks.

The supported registration methods are:

* ``api.register_panel(...)``;
* ``api.register_action(...)`` and ``api.action(...)``;
* ``api.register_dataframe_action(...)`` and ``api.dataframe_action(...)``;
* ``api.register_workflow(...)``;
* ``api.register_service(...)``;
* ``api.register_artifact_viewer(...)``;
* ``api.register_settings_schema(schema)``.

``api.get_setting(key, default=None)`` and ``api.set_setting(key, value)``
access plugin-scoped runtime settings. See `Plugin settings`_ for the
persistence limitation.

Use ``api`` to register contributions, rather than modifying
``PluginManager`` registries directly or patching the application's
underlying Panel template or workspace grid.

For the two supported manifest patterns, see :doc:`manifests`. For a
progressive implementation example, see :doc:`first_plugin`.

Registration IDs and namespaces
-------------------------------

The plugin API automatically qualifies **panel**, **action**,
**workflow**, **service**, and explicit **artifact-viewer** IDs with the
current plugin ID. Given the plugin ID ``example.plugin``:

.. code-block:: python

   api.register_action(
       id="calculate",
       title="Calculate",
       handler=calculate,
   )

The registered action is named ``example.plugin.calculate``. An ID already
prefixed by ``example.plugin.`` is kept unchanged.

Registration IDs must begin with an alphanumeric character and may contain
letters, numbers, dots, underscores and hyphens. Reusing an ID where a
unique registration is required raises ``PluginRegistrationError``. Do not
rely on implicit replacement of another plugin's contribution.

**Artifact types are different:** they are validated but *not* automatically
namespaced. A plugin should use a stable explicit type such as
``example.plugin.report`` for its outputs and viewers. Other plugins can
then consume that type without importing the producing plugin's internals.

Registering a panel
-------------------

.. code-block:: python

   def create_panel(context, **kwargs):
       import panel as pn
       view = pn.Column(pn.pane.Markdown("Hello from this plugin"))
       return view, None

   def register(api):
       api.register_panel(
           id="viewer",
           title="Example Viewer",
           factory=create_panel,
           description="A small example panel.",
           category="Examples",
           required_mappings=["record_id"],
           optional_mappings=["target_label"],
           default_layout={"x": 0, "y": 0, "w": 4, "h": 3},
           persist_layout=True,
           persist_state=False,
       )

A factory receives ``context`` and should accept ``**kwargs``. Return a
``(view, controller)`` pair. Returning ``None`` as the controller is valid
for a static panel with no subscriptions, jobs or state that need cleanup.
Other panels should provide a controller with an idempotent ``dispose()``.

``register_panel`` accepts the following keyword arguments:

.. list-table:: Panel registration fields
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``id``, ``title``, ``factory``
     - Required local ID, title and callable factory.
   * - ``description``, ``category``, ``icon``, ``tags``
     - Presentation and discoverability metadata.
   * - ``required_mappings``, ``optional_mappings``
     - Semantic dataset mappings; required entries gate real panel creation.
   * - ``uses_services``, ``produces``
     - Declared service keys and produced artifact/event descriptions.
   * - ``default_layout``, ``default_open_kwargs``
     - Initial workspace geometry and factory-opening defaults.
   * - ``requires``, ``optional_requires``
     - Contribution-level Python requirements (not plugin IDs).
   * - ``state_version``
     - Saved controller-state schema version (default ``1``).
   * - ``persist_layout``, ``persist_state``
     - Whether the panel instance and controller state should be saved;
       both default to ``True``.
   * - ``restore_policy``
     - Recorded restore policy; currently defaults to ``"best_effort"``.

There is **no** ``persistent=`` keyword in this registration method. Use
``persist_layout`` and ``persist_state``. Register semantic requirements
instead of assuming source columns literally use names such as ``id``,
``ra``, ``dec`` or ``label``. A required mapping prevents the functional
panel from opening until the user resolves it; an optional mapping must not
be necessary for the panel's basic operation.

The supported workspace-facing opener is:

.. code-block:: python

   context.plugins.open_panel(
       "example.plugin.viewer",
       context=context,
   )

``context`` is **keyword-only** for ``open_panel``. The opener returns the
workspace instance ID and can construct a loading tile while the actual
panel is being created. Panel instance IDs differ from registered panel IDs;
multiple instances can coexist.

Controller state and cleanup are described in :doc:`panels` and
:doc:`persistence`. Do not mutate live Panel/Bokeh widgets from a
background worker. Schedule changes on the relevant document/UI thread,
check that asynchronous results still belong to the latest request, and
release subscriptions and jobs in ``dispose()``.

Registering an action
---------------------

Actions are reusable operations that can be invoked without embedding them
in a particular panel. A handler is called with the supported arguments
``context``, ``request``, optionally ``cancel_token`` and ``manager``.

.. code-block:: python

   from astronomicAL.platform.plugins.specs import ArtifactResult

   def list_column_names(context, request, cancel_token=None):
       dataset_id = request.dataset_id or context.datasets.active_id()
       columns = context.datasets.list_columns(dataset_id)
       return ArtifactResult(
           type="example.plugin.column_list",
           payload={"columns": list(columns)},
           dataset_id=dataset_id,
       )

   def register(api):
       api.register_action(
           id="list_columns",
           title="List Dataset Columns",
           handler=list_column_names,
           inputs={"dataset": True, "selection": "none", "columns": "none"},
           outputs=["example.plugin.column_list"],
           run_in_job=True,
       )

The complete ``register_action`` keyword surface is:

* Required: ``id``, ``title``, ``handler``.
* Input/output declaration: ``inputs``, ``outputs``, ``params_schema``,
  ``settings_schema``.
* Execution: ``run_in_job=True`` (default), ``key_fn``.
* Presentation: ``description``, ``category``, ``icon``, ``tags``.
* Contribution dependencies: ``requires``, ``optional_requires``.

``inputs`` accepts ``InputSpec`` or an equivalent dictionary. An
``InputSpec`` declares ``dataset``, ``selection``, ``numeric_columns``,
``columns``, ``required_mappings``, ``optional_mappings`` and
``accepts_artifact_types``. Selection values are ``none``, ``optional`` or
``required``. Column requirements accept ``none``, ``one``, ``many`` or
``optional``. ``outputs`` accepts strings, ``OutputSpec`` values or
corresponding dictionaries.

``ActionRequest`` carries ``dataset_id``, ``row_ids``, ``columns``,
``params``, ``artifact_id`` and ``origin``. The manager fills suitable
request defaults, validates declared inputs and prepares parameters before
calling the handler.

The supported action result types include:

* ``ArtifactResult(type=..., payload=..., dataset_id=...)``;
* ``DatasetResult(id=..., dataframe=...)`` or
  ``DatasetResult(id=..., source=...)`` (provide exactly one data source);
* ``EventResult(topic=..., payload=...)``;
* ``ActionResult(value=..., artifacts=[...], datasets=[...], events=[...])``;
* a direct value or ``None``.

The manager processes declared artifact, dataset and event results against
the shared platform services. Treat a ``DatasetResult`` as a new working
dataset and an ``ArtifactResult`` as a derived output, not interchangeably.
See :doc:`actions` and :doc:`artifacts`.

Actions can also be registered with the ``api.action(...)`` decorator; it
accepts the same action metadata except ``handler``, which is supplied by
the decorated function.

To execute an action, use the manager instead of calling another plugin's
handler directly:

.. code-block:: python

   context.plugins.run_action(
       "example.plugin.list_columns",
       context,
       request={"dataset_id": dataset_id},
   )

The execution path supports ``on_done``, ``on_error`` and
``return_processed``. With ``run_in_job=True`` work may complete
asynchronously; do not assume the immediate return value is the result.
Do not assume that declaring a schema automatically creates a complete
interactive action form: the schemas presently support registration and
validation, but there is no general documented third-party form-rendering
API at this baseline.

DataFrame action convenience adapter
------------------------------------

``api.register_dataframe_action(...)`` and ``api.dataframe_action(...)``
wrap handlers that deliberately operate on pandas data. The adapter uses
bounded row-ID lookup when a selection is supplied. If no rows are
selected, **it rejects implicit whole-dataset materialization for a lazy
backend**. For whole-dataset operations on Parquet/DuckDB sources, register
a normal action and use ``DatasetSource`` or batched manager operations.

DataFrame actions support ``output_type``, ``selection``,
``numeric_columns``, ``columns``, required/optional mappings, schemas,
``run_in_job`` and the common presentation/dependency fields. See
:doc:`dataset_programming` for scalable access patterns.

Workflows
---------

``api.register_workflow(id=..., title=..., builder=...)`` registers a
workflow builder. Optional parameters are ``settings_schema``,
``description``, ``category``, ``icon``, ``tags``, ``requires`` and
``optional_requires``. A workflow can compose previously registered panels
and actions; it should not bypass the plugin/workspace management boundary.

When opening registered panels from a workflow, pass the fully qualified
panel ID and ``context=context`` to ``context.plugins.open_panel``.
See :doc:`workflows`.

Services
--------

``api.register_service`` accepts the required ``key`` and ``factory`` plus
``lazy=True``, ``replace=False``, ``description``, ``requires`` and
``optional_requires``. Service keys are plugin-qualified automatically:

.. code-block:: python

   api.register_service(
       key="client",
       factory=create_client,
       lazy=True,
   )

   # From a panel or handler whose context is available:
   client = context.services.get("example.plugin.client")

Services are live capabilities: API clients, database handles, caches and
resolvers. Derived data belongs in artifacts, not services. Treat service
replacement as an exceptional operation rather than a shortcut for
registration collisions. Services installed with plugin ownership can be
removed/disposed on disablement; a plugin's lifecycle hooks may need to
release additional resources explicitly. See :doc:`services`.

Artifact viewers
----------------

``api.register_artifact_viewer`` accepts ``artifact_type`` and
``viewer_factory`` plus optional ``id``, ``title``, ``description``,
``priority=100``, ``default=False``, ``requires`` and
``optional_requires``. Several viewers may support one artifact type, but
an explicit viewer ID cannot be duplicated. An explicit viewer ID is
plugin-qualified; the *artifact type* is not.

.. code-block:: python

   api.register_artifact_viewer(
       artifact_type="example.plugin.column_list",
       viewer_factory=create_column_list_viewer,
       id="column_list_viewer",
       title="Column List",
       default=True,
   )

A viewer factory can use ``context`` and an ``artifact_id`` to retrieve a
stored artifact. Consult :doc:`artifacts` and :doc:`first_plugin` for a
complete example.

Plugin settings
---------------

``api.register_settings_schema(schema)`` registers or replaces the one
settings schema associated with the plugin record. ``api.get_setting`` and
``api.set_setting`` operate on the plugin-scoped settings store. At
``90a912c`` the inspected ``PluginSettingsStore`` is in-memory; **do not
assume arbitrary plugin settings are durable across restarts**. This is
separate from persistent plugin activation preferences and workspace
controller snapshots. See :doc:`persistence`.

Lifecycle, ownership and failures
---------------------------------

Plugins should keep import-time and registration-time work cheap. They
should not open network connections, scan large datasets, launch long jobs,
or construct workspace panels during import or ``register(api)``. Register
factories, handlers and descriptions instead.

A plugin can define optional module hooks ``on_enable(...)`` and
``on_disable(...)``. Registrations belong to the registering plugin.
PluginManager rolls back registrations after failed enablement and removes
owned registrations during disablement. It can close owned panels, remove
managed services and cancel tracked jobs; plugin controllers must still
clean up their own observers, periodic callbacks and independently owned
resources. Cleanup must be safe when invoked more than once.

A plugin should not:

* Keep its own authoritative copy of the platform's active dataset,
  focused record or selection set.
* Send large tables, media frames or model objects through EventBus.
* Rely on direct imports of another plugin's private viewer or controller.
* Depend on a writable installed plugin source directory.
* Assume every dataset can be converted to an unrestricted pandas DataFrame.
* Treat successfully installed external Python code as sandboxed.

Use ``context.datasets`` for datasets, ``context.selection`` for shared
focus and selection sets, ``context.events`` for notifications,
``context.artifacts`` for reusable results, ``context.jobs`` for expensive
work, ``context.workspace`` for visible instances, and ``context.services``
for live runtime facilities. See :doc:`app_context` for ownership rules.

Versioning and compatibility
----------------------------

At this commit there is no dedicated plugin-contract-version manifest
field. ``PluginManifest`` provides host-version bounds, and plugins should
version their own interfaces and saved state. ``state_version`` is recorded
for a panel but state migration is **not** automatic. Compatibility with
other plugins' actions/services must be handled explicitly; optional
integrations should degrade gracefully when missing or disabled.

The static/community package installer performs validation and integrity
checks before enabling code, but enabled plugins execute Python in the
AstronomicAL host process. They are not sandboxed. See :doc:`manifests`,
:doc:`packaging` and :doc:`persistence`.

Before publishing
-----------------

Test discovery, validation, enable/disable/re-enable, missing dependencies,
failed registration rollback, panel opening and disposal, async completion
after closure, semantic mapping gaps, lazy dataset access, and workspace
save/restore. A plugin that restores correctly should also tolerate
missing or incompatible optional integrations.

For end-to-end packaging and Marketplace release verification, use the
`AstronomicAL Hello World template
<https://github.com/grant-m-s/AstronomicAL-Hello-World>`_.
