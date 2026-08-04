Event Catalogue
===============

Canonical Topics
----------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Topic
     - Meaning
   * - :code:`dataset.loaded`
     - A dataset was registered.
   * - :code:`dataset.active.changed`
     - The active dataset changed.
   * - :code:`dataset.updated`
     - A registered dataset changed.
   * - :code:`dataset.mapping.updated`
     - Semantic mappings changed.
   * - :code:`selection.focus.changed`
     - The focused row changed.
   * - :code:`selection.set.changed`
     - The active multi-row selection changed.
   * - :code:`artifact.created`
     - A new artifact was registered.
   * - :code:`plugin.enabled`
     - A plugin completed enablement.
   * - :code:`plugin.disabled`
     - A plugin completed disablement.

Payload Guidance
----------------

Payloads should contain identifiers, origin and lightweight metadata. Consumers
should retrieve the full dataset, artifact or service from the context.

Plugin Events
-------------

Plugin-specific topics are documented on the plugin page or in the plugin
manifest.

.. todo::

   Replace this hand-written list with a generated event schema catalogue.
