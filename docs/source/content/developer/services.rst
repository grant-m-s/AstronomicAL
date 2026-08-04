Creating Services
=================

Registering a Service
---------------------

Register shared runtime capabilities through :code:`api.register_service(...)`:

.. code-block:: python

   api.register_service(
       key="client",
       factory=create_client,
       lazy=True,
       description="Example API client.",
   )

For a plugin with ID :code:`example.plugin`, the local key becomes:

.. code-block:: text

   example.plugin.client

Service keys are automatically namespaced unless they are already qualified
with the current plugin ID.

Service Factories
-----------------

A service factory should construct and return the live capability:

.. code-block:: python

   def create_client(context, **kwargs):
       return ExampleClient()

Keep registration itself declarative. Do not create the client inside
:code:`register(api)`.

Factories are the right place to construct objects such as:

* API or archive clients;
* authenticated sessions;
* database connections;
* caches;
* resource samplers;
* shared domain helpers.

Lazy Services
-------------

With :code:`lazy=True`, the factory is not called until the service is first
requested.

This keeps application startup cheap and avoids opening connections or importing
heavy optional dependencies when the service is never used.

Use eager construction only when the capability genuinely must exist as soon as
the plugin is enabled.

Using a Service
---------------

Retrieve a registered service through :code:`context.services`:

.. code-block:: python

   client = context.services.get("example.plugin.client")
   result = client.fetch(...)

Panels and actions should depend on the service key rather than importing the
service implementation from another plugin.

A panel can also declare service dependencies in its registration with
:code:`uses_services=[...]` when those services are required for construction.

Optional Integrations
---------------------

A plugin can integrate with another plugin through its public service key
without importing that plugin's implementation module.

Optional integrations must still handle the service being unavailable. A
missing optional plugin or disabled provider should reduce functionality
gracefully rather than breaking unrelated parts of the panel.

For required cross-plugin relationships, declare the plugin dependency in the
manifest and the service requirement on the contribution that uses it.

Ownership
---------

Services registered through :class:`PluginAPI` are associated with the plugin
that registered them.

When the plugin is disabled, PluginManager removes its installed services from
the shared registry. Initialized service objects are disposed where possible.

The platform can also register host-owned services, so the registry is not
limited to plugin-provided entries.

Cleanup
-------

A service that owns resources should expose one of the standard teardown methods
recognised by the registry:

.. code-block:: python

   class ExampleClient:
       def close(self):
           ...

The registry attempts common lifecycle methods such as:

* :code:`dispose()`;
* :code:`close()`;
* :code:`shutdown()`.

Cleanup should be idempotent where practical.

This is especially important for services that own:

* network sessions or sockets;
* background threads;
* temporary directories;
* file handles;
* database connections;
* native resources.

Replacement
-----------

Service keys are unique by default.

A registration can explicitly request replacement:

.. code-block:: python

   api.register_service(
       key="client",
       factory=create_client,
       lazy=True,
       replace=True,
   )

Use replacement deliberately. Normal plugins should prefer stable namespaced
keys rather than replacing another provider's service.

If an existing initialized service is replaced or removed, its teardown path is
run before the registry discards it.

Authentication
--------------

Credentials should come from an appropriate user environment, external secret
store or other explicit secure configuration path.

Do not place access tokens, passwords or other secrets in workspace panel state
or workspace JSON.

Plugin settings schemas are useful for ordinary configuration, but should not be
treated as a dedicated secret vault unless the surrounding application provides
a secure storage mechanism for that value.

Services and Persistence
------------------------

Service objects are live process state and are not persisted as part of a
workspace.

Persist only the small information needed to recreate behaviour, such as a
selected endpoint name or service mode. Reconstruct the live client through the
service factory after the plugin is enabled.

Services and Artifacts
----------------------

Use a service for a live capability.

Use an artifact for a generated result.

.. code-block:: text

   API client/session        -> service
   downloaded spectrum       -> artifact
   prediction result         -> artifact
   database connection       -> service

.. caution::

   Do not use the service registry as a generic dictionary for computed outputs,
   panel state or datasets. Generated reusable results belong in
   :code:`context.artifacts`, and tabular working inputs belong in
   :code:`context.datasets`.
