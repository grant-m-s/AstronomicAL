.. _semantic-mappings:

Semantic Column Mappings
========================

Why Mappings are Required
-------------------------

Different datasets use different column names for the same concept. One table
may use :code:`source_id`, while another uses :code:`object_id`.

A plugin therefore requests a semantic name such as :code:`record_id` and lets
the dataset mapping identify the matching source column.

Common Mappings
---------------

The platform provides common aliases for:

* :code:`record_id` -- a stable identifier for each row;
* :code:`coords.ra` -- right ascension;
* :code:`coords.dec` -- declination;
* :code:`target_label` -- a known or provisional label.

Plugins can declare additional semantic names for their own capabilities, such
as :code:`image.path`, :code:`image.uri` or survey-specific
:code:`spectra.*` identifiers.

For :code:`record_id`, mapping requirements can also allow **Use Index** where a
dataset does not contain a suitable ID column.

Required and Optional Mappings
------------------------------

A required mapping must be resolved to a valid dataset column before the real
panel can be constructed.

An absent optional mapping does not prevent the panel from opening. It can be
used to enable extra behaviour when the mapping is available.

.. Add GIF: resolving a mapping request from the header.

Changing a Mapping
------------------

Mappings belong to a dataset. Changing the active dataset may therefore expose
a different mapping state.

When the application mapping flow changes a mapping it publishes
:code:`dataset.mapping.updated`. Panels that depend on mappings can then
re-resolve their columns or refresh their display.

.. caution::

   A mapping describes the meaning of a column, not merely its data type. Two
   numeric columns are not interchangeable just because both contain
   floating-point values.
