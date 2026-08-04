Supported Formats and External Services
=======================================

Dataset Formats
---------------

The platform currently includes paths for:

* Parquet;
* FITS binary tables converted to Parquet;
* image manifest tables;
* VOTable/FITS exchange through SAMP;
* plugin-provided imports such as Hugging Face datasets.

Export Formats
--------------

Export support depends on the dataset and plugin.

.. todo::

   Add the final release table for CSV, Parquet, FITS, VOTable and artifact
   exports.

External Services
-----------------

Optional astronomy plugins may connect to:

* Euclid archive services;
* DESI/SPARCL;
* SDSS/BOSS services;
* HiPS/Aladin resources;
* SAMP hubs.

The Hugging Face integration connects to the Hugging Face Hub and Datasets
service.

Authentication
--------------

Each plugin should document whether credentials are optional, required or read
from an existing client configuration.

Offline Behaviour
-----------------

Generic dataset, visualisation, selection and annotation plugins can be used
without remote services. Remote panels should show an unavailable state rather
than block the rest of the workspace.
