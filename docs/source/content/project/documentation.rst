Contributing to the Documentation
=================================

Documentation Style
-------------------

Keep the original AstronomicAL style:

* explain the purpose before the controls;
* use short sections;
* prefer a practical example;
* add notes and cautions where users may make a costly mistake;
* leave room for screenshots and GIFs.

Page Types
----------

Use:

* tutorials for a complete learning journey;
* user guides for a task;
* concept pages for understanding;
* plugin pages for one bundled plugin;
* API pages for exact interfaces.

Images and GIFs
---------------

Store images in the documentation image directory and use descriptive
filenames.

Add alternative text where supported and avoid relying on colour alone.

Code Examples
-------------

Examples should be complete enough to copy and should use public APIs only.

Building Locally
----------------

.. code-block:: bash

   cd docs/source
   make html

Link Checking
-------------

Run the Sphinx link checker before release.

.. code-block:: bash

   make linkcheck

Review Checklist
----------------

Check that:

* headings match the navigation;
* cross-references resolve;
* commands match the current release;
* optional features are labelled;
* screenshots use the current interface;
* no private paths, tokens or dataset values are visible.
