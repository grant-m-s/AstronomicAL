Broadband SED
=============

Overview
--------

:code:`astro.sed` turns photometry stored across multiple dataset columns into
a broadband spectral energy distribution (SED) for the focused record.

The SED can be displayed either as flux density, :math:`F_\nu`, or as
:math:`\nu F_\nu`. Photometric measurements can be supplied in AB magnitudes,
μJy, or nJy, with optional error columns for each band.

Measurements with negative errors are interpreted as upper limits and are
displayed as downward-pointing arrows.

What You Do
-----------

#. Focus a source in Record Browser, a plot, or another linked panel.
#. Select or load the appropriate photometric-band definition.
#. Map the dataset columns corresponding to each photometric band.
#. Specify the units and, where available, the associated error columns.
#. Inspect the extracted photometry as a table and SED plot.
#. Move to another source and the view updates automatically.
#. Reopen a completed result later through its artifact viewer.

.. todo::

   Add an image of the band-mapping controls beside a completed SED.

Band Definitions
----------------

A band-definition file describes the photometric bands available to the panel.
Each band includes a name and effective wavelength, with an optional FWHM.

The definition is independent of the literal column names in a catalogue.
After loading it, the corresponding columns in the active dataset can be mapped
to each band.

An example definition is provided containing Euclid bands together with
external ground-based bands relevant to the northern hemisphere.

Keeping the band-definition file alongside a dataset or workspace makes it
easier to reproduce the same SED configuration later.

Mappings
--------

The panel requires:

* :code:`record_id`

The following mapping is optional:

* :code:`redshift`

Each photometric band can additionally be mapped to:

* a measurement column;
* a measurement unit;
* an optional error column.

Unresolved band columns can be mapped from within the panel.

Panel
-----

**Broadband SED**

The panel extracts the configured photometric measurements from the focused
record and presents them as both a structured table and an SED plot.

Changing the focused record updates the displayed measurements using the same
band configuration and mappings.

Where applicable, the optional redshift mapping can be used by displays or
interpretations that require it.

Requirements
------------

The plugin uses:

* :code:`numpy`
* :code:`pandas`
* :code:`panel`
* :code:`holoviews`

Artifact Viewer
---------------

Completed results are stored as :code:`astro.sed.broadband` artifacts.

The **Broadband SED Viewer** can display a saved result independently of the
live SED panel, without repeating the original extraction.

Troubleshooting
---------------

An incomplete SED usually indicates that:

* one or more photometric-band columns are unmapped;
* the focused record contains missing measurements;
* the selected band definition does not match the active dataset;
* no record is currently focused.

.. note::

   The panel visualises the measurements available for the focused record. It
   does not infer missing bands or automatically fit a physical SED model.