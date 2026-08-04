Declaring Semantic Mappings
===========================

Required Mappings
-----------------

.. code-block:: python

   api.register_panel(
       ...,
       required_mappings=["record_id", "coords.ra", "coords.dec"],
   )

The panel is constructed only after these values are available.

Optional Mappings
-----------------

Optional mappings can enable additional overlays or metadata.

Rich Requirements
-----------------

A mapping declaration may include a display name, description, aliases and
whether index fallback is allowed.

Naming
------

Use stable dotted semantic names for domain concepts:

.. code-block:: text

   coords.ra
   coords.dec
   image.uri
   spectra.desi_target_id

Dataset Changes
---------------

A mapped panel must handle active-dataset changes. The mapping gate may dispose
and rebuild the controller with restored state.