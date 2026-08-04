Hugging Face Datasets
=====================

Overview
--------

:code:`integrations.huggingface` connects AstronomicAL to image datasets on the
Hugging Face Hub. It can search for a repository, inspect its structure,
preview examples and register selected splits as AstronomicAL image-manifest
datasets.

What You Do
-----------

1. Search for a dataset or enter an exact :code:`owner/repository` ID.
2. Select a result and let AstronomicAL inspect its configurations, splits and
   image structure.
3. Preview sample images and check the detected image, label and record-ID
   fields.
4. Select the splits to import and optionally limit the rows per split.
5. Import the selected splits as separate AstronomicAL datasets.
6. Choose which imported split becomes active, or leave the active dataset
   unchanged.
7. Continue in Image Viewer, Image Selection Gallery, Record Browser,
   Annotations or an ML workflow.

Existing Hugging Face caches and matching AstronomicAL registrations are reused
where possible before new data is downloaded.

.. TODO: Add a GIF showing repository search, inspection, preview and import.

Requirements
------------

The integration uses :code:`huggingface_hub>=0.24`, :code:`datasets>=2.20` and
:code:`pillow>=10`. The :doc:`Image plugin <image>` is optional but is the
normal way to inspect imported images.

Panel
-----

**Hugging Face Dataset Importer**

The importer supports:

* dataset search by type, popularity or recent updates;
* automatic configuration, split and column detection;
* local Hugging Face cache and existing-registration reuse;
* previewing selected examples before import;
* importing one or more selected splits;
* optional row limits and Parquet-backed manifests;
* advanced overrides for unusual repository structures.

Automatic inspection prefers structured Hugging Face dataset rows when they
provide a usable image column. It can fall back to image files and folders in
the repository, where labels may be inferred from paths. Labels are optional.

Action
------

**Import Hugging Face Image Dataset** provides a job-backed programmatic import
for a single split. Parameters include repository, configuration, split,
dataset identity, image, label and ID columns, row limit, access token,
:code:`trust_remote_code`, Parquet storage and whether the imported dataset
becomes active.

Mappings and Output
-------------------

Each imported split is registered as its own image-manifest dataset. The
importer publishes mappings for:

* :code:`record_id`;
* :code:`image.path`;
* :code:`image.uri`;
* :code:`target_label`.

If no suitable source ID is selected, record IDs are generated. Labels may come
from a selected dataset column, be inferred from repository paths, or remain
empty.

By default the manifest is written to Parquet and registered through the
platform's Parquet backend where available, with a pandas registration used as
a fallback.

Security
--------

Remote dataset code is not trusted by default. Private or gated repositories
may require a Hugging Face access token; when no token is entered, the normal
Hugging Face authentication and cache configuration can be used.

.. caution::

   Only enable :code:`trust_remote_code` for a repository whose code and owner
   you have reviewed and trust. Large imports may download many image files, so
   preview the dataset or use a row limit before importing a full split where
   practical.