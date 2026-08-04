Aladin Lite
===========

Overview
--------

:code:`astro.aladin` adds an interactive sky view for the currently focused
record. It is useful for checking the surrounding field, nearby objects, and
sky context without leaving the AstronomicAL workspace.

The panel embeds the `Aladin Lite <https://aladin.cds.unistra.fr/AladinLite/>`_
web application, allowing images from a wide range of astronomical surveys to
be visualised interactively.

A predefined selection of surveys covering different wavelength ranges is
provided. Additional surveys from the
`HiPS catalogue <https://aladin.cds.unistra.fr/hips/list>`_ can be added by
including their identifiers in :code:`SURVEY_DATA` within the plugin's
:code:`panel.py`.

What You Do
-----------

#. Open the **Aladin Lite** panel.
#. Map the record ID, right ascension, and declination columns if you have not
   already done so.
#. Focus a record from a plot, the navigation bar, a gallery, or Record Browser.
#. Use the controls in the embedded Aladin viewer to inspect the surrounding
   sky.
#. Focus another record and the viewer recentres on the new source
   automatically.

The panel does not choose or own the current record. It follows the same
platform focus used by other linked panels.

.. todo::

   Add a GIF showing focus changing in a plot and Aladin recentring.

Required Mappings
-----------------

The panel requires:

* :code:`record_id`
* :code:`coords.ra`
* :code:`coords.dec`

The coordinate values must contain valid sky coordinates for the viewer to move
to the source.

Panel
-----

**Aladin Lite**

The panel embeds Aladin Lite and centres it on the mapped coordinates of the
focused record. Normal Aladin interactions remain available inside the viewer,
while AstronomicAL handles the link between dataset focus and sky position.

The panel publishes :code:`astro.aladin.updated` when its displayed source is
updated. This can be useful for diagnostics or for plugins that need to react
to changes in the sky viewer.

Saved State
-----------

Supported viewer settings and the panel layout are restored on a best-effort
basis.

The focused record remains platform state and is not stored as an independent
Aladin selection.

Troubleshooting
---------------

If the viewer does not move:

* confirm that a record is focused;
* check the :code:`coords.ra` and :code:`coords.dec` mappings;
* inspect the focused record for missing or invalid coordinates;
* confirm that the browser can load the Aladin web resources.

.. note::

   Aladin provides a sky view around a source. It does not add catalogue
   measurements to the active dataset or automatically save viewed survey data.