Application Context
===================

The :code:`AppContext` object is the runtime boundary between a plugin and the
host.

Available Services
------------------

.. code-block:: python

   context.datasets
   context.selection
   context.events
   context.artifacts
   context.jobs
   context.workspace
   context.services
   context.plugins
   context.persistence
   context.navigation
   context.runtime_status

Accessing Data
--------------

Use :code:`context.datasets` to find the active dataset or retrieve a source.

Accessing Focus
---------------

Use :code:`context.selection` rather than reading another panel.

Submitting Work
---------------

Use :code:`context.jobs` for slow or cancellable work.

Looking Up a Service
--------------------

.. code-block:: python

   client = context.services.get("astro.example.client")