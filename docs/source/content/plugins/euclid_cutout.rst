Euclid Cutout
=============

.. raw:: html

   <video width="100%" autoplay loop muted playsinline>
     <source src="_static/images/euclid_cutout_example_short.mp4" type="video/mp4">
   </video>


Overview
--------

:code:`astro.euclid_cutout` retrieves image cutouts centred on supplied
coordinates from the Euclid Science Archive using :code:`astroquery`.

The currently available bands are:

* :code:`VIS`
* :code:`NISP.Y`
* :code:`NISP.J`
* :code:`NISP.H`

Ground-based EXT data are not currently retrieved.

By default, the panel accesses publicly available Euclid data, currently Q1.
Euclid Consortium members can log in to access Internal Data Releases.

What You Do
-----------

#. Map the record ID, right ascension, and declination columns.
#. Focus a source in another panel.
#. Open **Euclid Cutout** and choose the available cutout options.
#. Request the relevant VIS or NIR data.
#. Inspect the image, light profile, or bounded surface.
#. Reopen the retained result later through its artifact viewer.

When focus changes quickly, older requests are treated as stale and are not allowed to replace the result for the newest source.

Settings
--------

Radius
   Sets the cutout size from 1 to 100 arcsec.

Filter
   Selects the displayed band.

   The **Colour** option creates an RGB image using :code:`NISP.H`,
   :code:`NISP.Y`, and :code:`VIS` as the red, green, and blue channels,
   respectively.

Stretching
   Controls the image intensity scaling.

   Available options are:

   * Linear
   * Asinh
   * Logarithmic
   * Square Root
   * Power Law

   The scale parameter corresponds to the :math:`a` parameter used by
   :code:`astropy.visualization`. When set to :code:`None`, the default value
   for the selected stretch is used.

Clipping
   Sets the minimum and maximum displayed values on a relative scale from 0 to 1.

   For colour images, clipping can be adjusted independently for each channel.
   Separate gamma values can also be applied to the red, green, and blue
   channels, giving the transformation

   .. math::

      (R, G, B)
      \rightarrow
      \left(
      R^{\gamma_R},
      G^{\gamma_G},
      B^{\gamma_B}
      \right).

Contour Levels
   Overlays contours on the image at selected intensity levels.

Light Profiles and 3D View
   Displays light profiles along the :math:`x` and :math:`y` directions at a selected pixel.

   A 3D surface view is also available, with adjustable grid size and
   smoothing.

Requirements
------------

The plugin uses astronomy archive and WCS packages, including:

* :code:`astroquery`
* :code:`astropy`
* :code:`reproject`
* :code:`mocpy`

:code:`matplotlib` is used by supported visual products where available.

Required Mappings
-----------------

The panel requires the following semantic mappings:

* :code:`record_id`
* :code:`coords.ra`
* :code:`coords.dec`

Panel
-----

**Euclid Cutout**

The panel follows the platform focus and submits archive retrieval through the Job Manager. This keeps network and image-processing work away from normal UI callbacks and makes the current request status visible.

The available views include:

* the retrieved cutout;
* a light-profile view;
* a bounded surface view.

The exact bands and products available depend on archive coverage and the plugin runtime.

Artifacts
---------

Successful results are stored as :code:`astro.cutout.euclid` artifacts.

The artifact retains the relationship to the source record together with the output required by the Euclid cutout artifact viewer.

Troubleshooting
---------------

No result may mean that:

* the coordinate is outside the available Euclid coverage;
* the archive service is unavailable;
* authentication or additional access is required;
* the mapped coordinates are missing or invalid;
* an earlier request was discarded because the focused record changed.

.. note::

   Remote archive coverage is independent of the AstronomicAL dataset. A valid record can therefore have no matching Euclid product without indicating an application error.