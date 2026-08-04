Image Assets
============

Overview
--------

:code:`core.image` provides the complete image-dataset workflow: create a
manifest from files, display the image belonging to the focused record, and
browse the current selection as a thumbnail gallery.

What You Do
-----------

For a folder of images:

1. Open **Image Manifest Builder** and choose the root folder.
2. Decide whether subfolders should be scanned and whether parent folder names
   should be used as labels.
3. Build and register the manifest dataset.
4. Map :code:`record_id` and :code:`image.uri` if they were not mapped
   automatically.
5. Open **Image Viewer** to follow one focused record.
6. Create a selection in another panel and open **Image Selection Gallery** to
   compare many selected images.
7. Click any gallery card to make that record the shared focus.

The same viewer and gallery can be used with a compatible manifest imported by
the Hugging Face plugin or another data loader.

.. TODO: Add an image showing Manifest Builder, Viewer and Selection Gallery.

Requirements
------------

Pillow is required.

Mappings
--------

Image viewing requires:

* :code:`record_id`;
* :code:`image.uri`.

The image value may resolve to a local path, a path relative to the manifest, or
a supported URI. The resolver handles loading separately from the dataset table.

Panels
------

Image Manifest Builder
**********************

The builder scans supported image files and creates a tabular manifest. It can:

* scan recursively;
* recognise common formats such as JPEG, PNG, WebP, BMP, GIF and TIFF;
* store relative paths for a more portable project;
* derive a label from the parent folder;
* write the manifest to Parquet;
* register it and optionally make it the active dataset.

Image Viewer
************

The viewer displays the image associated with the focused record. Display
options include fitting and bounded image sizing. A successfully resolved image
is also exposed as an :code:`image.preview` artifact.

Image Selection Gallery
***********************

The gallery loads thumbnails for the active selection rather than the entire
dataset. It can show the focused card and, where available, label, prediction or
uncertainty information. Clicking a card changes platform focus so every other
linked panel follows it.

The gallery uses bounded batches and concurrency to avoid starting an unlimited
number of image requests at once.

Action
------

**Build Image Manifest** runs the folder scan as a background job, allowing the
same operation to be used outside the panel.

Artifacts
---------

* :code:`image.preview`

.. note::

   A manifest stores image references and metadata, not a copy of every image
   byte. Keep relative files with the project, or ensure that shared URLs remain
   accessible to other users.
