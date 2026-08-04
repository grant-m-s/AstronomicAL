Selection and Record Navigation
===============================

Focus and Selection Are Different
---------------------------------

AstronomicAL keeps the currently focused record separate from the current
selection.

The focused record is the record being inspected now.

A selection is a set of records collected for navigation or later operations.

Keeping these concepts separate allows an image viewer, spectrum panel, table,
and annotation tool to follow the same focused record without each maintaining
its own selection state.

Selection Manager
-----------------

The Selection Manager owns the canonical focus and active selection set.

Changes publish platform events so interested plugins can react without
depending directly on the panel that caused the change.

Record Navigation
-----------------

Record navigation reads the current focus and moves through either:

* all records in the active dataset; or
* only the records in the active selection.

This is the behaviour exposed by the application navigation controls.

Dataset Switching
-----------------

When the active dataset changes, focus and selection state must be reconciled
with the new dataset. State that no longer applies should not be carried over
blindly.

Persisting a Selection
----------------------

A selection can be stored as an artifact when it needs to survive beyond the
current interaction or be consumed by another plugin.

The focused record is usually treated as transient workspace state.
