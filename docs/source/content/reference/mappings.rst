Semantic Mapping Catalogue
==========================

The following mappings are used by bundled plugins.

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Mapping
     - Meaning
     - Common Users
   * - :code:`record_id`
     - Stable identifier for one row
     - Most linked panels
   * - :code:`target_label`
     - Known or provisional target label
     - ML and review workflows
   * - :code:`coords.ra`
     - Right ascension
     - Aladin, cutouts and spectra
   * - :code:`coords.dec`
     - Declination
     - Aladin, cutouts and spectra
   * - :code:`redshift`
     - Astronomical redshift
     - SED and domain panels
   * - :code:`image.uri`
     - Local path, relative path or URL for an image
     - Image Viewer and Gallery
   * - :code:`spectra.desi_target_id`
     - Optional DESI identifier
     - DESI Spectra
   * - :code:`spectra.sdss_target_id`
     - Optional SDSS/BOSS identifier
     - SDSS Spectra
   * - :code:`spectra.euclid_source_id`
     - Optional Euclid identifier
     - Euclid Spectra

.. note::

   Plugins may define additional mappings. Their page should document the
   meaning, expected units and whether the mapping is required.

.. todo::

   Generate this table from registered plugin specifications during the
   documentation build.
