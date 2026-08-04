Astronomy Spectra
=================

Overview
--------

:code:`astro.spectra` links the focused catalogue record to supported survey
spectra. Separate panels cover DESI, SDSS/BOSS, and Euclid so that each
survey's archive, identifiers, and retrieval rules remain clear.

Euclid Spectra
**************

The **Euclid Spectra** panel retrieves and displays Euclid Red Grism spectra,
showing both the raw data and a box-filter-smoothed version.

Spectra can be retrieved in two ways:

Identifier Query
   Retrieves a single spectrum using an identifier column from the main table.

Cone Search
   Retrieves all spectra within a user-defined radius around the source
   coordinates and displays them in order of increasing distance from the
   source.

The cone-search radius can also be linked to the Euclid cutout size, allowing
all spectra within the cutout region to be retrieved. The positions of the
selected spectra can then be overlaid on the Euclid image.

Common emission-line positions can be displayed on the spectrum when a
redshift is available. The redshift can be:

* queried from the Euclid pipeline using
  :code:`catalogue.spectro_zcatalog_spe_classification`, with the most reliable
  redshift solution returned;
* read from a mapped column in the main table;
* entered manually.

DESI and SDSS Spectra
*********************

The **DESI Spectra** and **SDSS Spectra** panels provide similar functionality
to the Euclid Spectra panel, but retrieve spectra from the DESI and SDSS/eBOSS
surveys. Data are accessed using the SPARCL package (Juneau et al. 2024).

For these panels, the source redshift is retrieved together with the spectrum
and can be used directly to display common emission-line positions.

What You Do
-----------

#. Map the record ID and sky-coordinate columns.
#. Optionally map a survey-specific target ID for direct lookup.
#. Open the panel for the required survey.
#. Focus a source in another panel.
#. Use automatic reload or manually load a spectrum by coordinates or target
   ID.
#. Adjust the matching radius, smoothing, and plot options where required.
#. Open produced :code:`astro.spectra` artifacts with the **Spectrum Viewer**.

.. todo::

   Add an image comparing the DESI, SDSS, and Euclid spectrum panels.

Requirements
------------

The plugin uses:

* :code:`numpy`
* :code:`pandas`
* :code:`panel`
* :code:`holoviews`
* :code:`matplotlib`
* :code:`astropy`
* :code:`astroquery`
* :code:`sparcl`
* :code:`mocpy`
* :code:`requests`

Individual panels also declare the smaller subset of packages required by their
survey backend.

Required Mappings
-----------------

All survey panels require:

* :code:`record_id`
* :code:`coords.ra`
* :code:`coords.dec`

Optional survey identifiers allow direct target-ID retrieval:

* :code:`spectra.desi_target_id`
* :code:`spectra.sdss_target_id`
* :code:`spectra.euclid_source_id`

Panels
------

DESI Spectra
************

Retrieves DESI DR1 spectra through SPARCL. The panel can use a mapped target ID
or perform a coordinate search within the selected matching radius.

SDSS Spectra
************

Retrieves BOSS DR17 and SDSS DR17 spectra through SPARCL. As with DESI, lookup
can use either the mapped survey identifier or the focused sky coordinates.

Euclid Spectra
**************

Retrieves Euclid Q1 spectra through the shared Euclid client. The panel supports
source-ID or coordinate lookup and can query Euclid redshift information for
the focused result.

All three panels support automatic reload on focus changes, configurable
smoothing, and plot-refresh controls. DESI and SDSS can also show spectral
models and line markers where available.

Jobs and Focus Changes
----------------------

Archive retrieval runs through the platform :code:`JobManager`. Rapid focus
changes are debounced, previous requests are cancelled where possible, and
generation checks prevent a late result for an old source from replacing the
current display.

The panels listen to selection focus, active-dataset, and mapping changes. With
automatic reload disabled, focus changes update the target information without
starting a new archive request.

Artifacts
---------

Successful retrievals produce :code:`astro.spectra` artifacts and, where
coordinates are available, :code:`astro.coords` artifacts.

The plugin also publishes spectrum-running, spectrum-updated, and
coordinate-updated events for other platform components.

The **Spectrum Viewer** is the default viewer for :code:`astro.spectra`
artifacts and displays a retrieved result independently of the live survey
panel.

.. note::

   Survey footprint, target matching, archive availability, and access rules
   differ between DESI, SDSS/BOSS, and Euclid. A source with no returned
   spectrum is not necessarily an application error.