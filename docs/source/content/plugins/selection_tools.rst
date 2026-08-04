Selection Tools
===============

Overview
--------

:code:`core.selection_tools` makes the platform's active selection visible and
usable. It bridges the gap between selecting rows in another panel and turning
those rows into a dataset for continued analysis.

What You Do
-----------

1. Create a selection using a scatter plot, image gallery, active-learning query
   or another plugin.
2. Open **Selection Set** to preview the selected records.
3. Move focus through the selected IDs while linked detail panels remain open.
4. Remove the selection when it is no longer needed, or materialise it as a new
   dataset.
5. Optionally make the new subset the active dataset and continue exploring it.

.. TODO: Add a GIF showing lasso selection, preview and dataset creation.

Mappings
--------

The :code:`record_id` mapping is optional. When unavailable, the plugin can fall
back to a dataframe index where the source supports it.

A stable record ID is strongly preferred because it survives sorting, filtered
views and exchange with other plugins more reliably than an index position.

Panel
-----

**Selection Set**

The panel provides:

* the current selection count;
* a preview of selected rows;
* previous and next focus navigation within the set;
* a clear-selection control;
* controls for naming and creating a derived dataset;
* an option to activate the new dataset immediately.

Changing focus through the panel does not alter the membership of the set.

Action
------

**Create Dataset From Active Selection** looks up the selected rows, registers
a new dataset and can make it active. The action allows the same operation to
be used by a workflow without opening the panel.

Typical Uses
------------

Selection Tools is useful for:

* turning a lassoed region into a working dataset;
* inspecting an active-learning query one row at a time;
* comparing selected images or spectra;
* exporting or transforming a scientifically interesting subset.

.. caution::

   Index fallback is less portable than a stable identifier. Map
   :code:`record_id` whenever possible, especially before saving or sharing a
   selection-derived result.
