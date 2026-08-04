.. _focus-selection:

Focus, Selection and Navigation
===============================

Focused Record
--------------

The focus is the one row currently being inspected. Clicking a point, choosing
a table row or using record navigation can update the focus.

Panels such as Record Browser, Image Viewer and Spectra can react to the same
focus state through :code:`context.selection`.

Selection Set
-------------

The active selection set contains a group of row identifiers from one dataset.
It is commonly created with a box, lasso, table selection, filter or workflow.

The set can then be:

* inspected by selection-aware panels and tools;
* used as the input to an action;
* converted into a derived dataset;
* passed to review or active-learning workflows.

By default, creating a selection set also creates a :code:`selection.ids`
artifact describing that selection.

Why They are Separate
---------------------

You may want to inspect one row inside a larger selected group. Changing focus
does not clear the active selection set.

.. code-block:: text

   active selection = 250 rows
   current focus    = one row

When a new selection set is created with the default focus policy, AstronomicAL
preserves the existing focus if it is already inside that set; otherwise focus
moves to the first selected row.

Navigation
----------

Record navigation can step through the complete active dataset or through the
active selection set. The application toolbar also lets the focused row be
added to or removed from the active selection.

Clearing State
--------------

Clearing focus removes the currently inspected row. Clearing the selection set
removes the group. These are separate operations and publish separate selection
events.
