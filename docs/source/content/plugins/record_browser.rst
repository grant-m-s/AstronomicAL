Record Browser
==============

Overview
--------

:code:`core.record_browser` is the general one-record-at-a-time view of the
active dataset. It provides a readable inspection panel while also acting as a
simple way to change the shared focused record.

What You Do
-----------

1. Open **Record Browser** for the active dataset.
2. Search or move to the record that you want to inspect.
3. Review its values and the configured label information.
4. Make that row the focused record.
5. Keep Image, Aladin, Spectra, SED, Annotations and other linked panels open so
   they update at the same time.
6. Focus a point in a plot or gallery and the browser follows it in return.

This makes the browser useful as the central details panel in an exploration,
review or active-learning workspace.

.. TODO: Add a GIF showing two-way focus between Record Browser and a plot.

Required Mappings
-----------------

* :code:`record_id`

The stable ID is used to find the row and publish focus consistently across
plugins.

Panel
-----

**Record Browser**

The panel browses the active dataset and presents one row in a form intended for
human inspection. It does not own dataset loading or mapping configuration.
Those remain platform responsibilities.

The browser publishes :code:`selection.focus.changed` when the user chooses a
new record. It also listens for that event, so the displayed row can originate
from:

* a scatter-plot tap;
* an image-gallery card;
* selection navigation;
* an active-learning query;
* another record-aware plugin.

Label Display
-------------

The plugin includes label-related browsing support and publishes
:code:`labels.settings.updated` when its label-display settings change. This
allows other compatible views to present label information consistently without
making the browser the owner of the labels themselves.

Typical Uses
------------

Use Record Browser when:

* the complete row contains more detail than can be shown in a plot tooltip;
* a review workflow needs a stable central source view;
* several domain panels should follow the same object;
* a selected set should be inspected one record at a time.

.. note::

   Focus is one record. A selection is a set of records. Browsing a focused row
   does not automatically add it to or remove it from the active selection.
