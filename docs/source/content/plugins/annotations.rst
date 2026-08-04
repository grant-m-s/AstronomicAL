Annotations
===========

Overview
--------

:code:`core.annotations` adds a lightweight review layer to any dataset. Notes,
review status, confidence, tags and optional label suggestions are attached to a
record without changing the original source table.

Label suggestions are free-form annotation metadata. They do not require a
label column to exist in the dataset and are not restricted to values from an
existing label column.

What You Do
-----------

1. Focus a record using Record Browser, a plot, a gallery or another workflow.
2. Write a note describing anything important about that record.
3. Set its review status and, where useful, a confidence value, tags or a
   free-form label suggestion.
4. Save the note or review state and continue to another record.
5. Open the summary panel to review completed work or create an exportable
   summary dataset or CSV artifact.

This is useful for data-quality review, label verification, collaborative
catalogue inspection and recording decisions made during active learning.

.. TODO: Add an image showing one focused record and its annotation form.

Required Mappings
-----------------

* :code:`record_id`

No label mapping is required. A dataset does not need an existing label column
to use annotations, and label suggestions entered by a reviewer are stored as
annotation metadata rather than being constrained by the source dataset.

Annotations are matched using both the dataset ID and row ID. The same row ID
in two datasets is therefore treated as two separate records.

Panels
------

Annotations
***********

The **Annotations** panel edits the currently focused record. Its interface is
split into **Annotate**, **Record** and **History** tabs.

The **Annotate** tab can store:

* free-text notes;
* review status, including unreviewed, in review, approved, rejected, unsure
  and needs follow-up;
* confidence;
* tags;
* an optional free-form label suggestion.

Notes and review-state changes are stored as artifacts. Saving either also
records the current status, confidence, tags and label suggestion so that the
latest review state can be reconstructed from annotation history.

The **Record** tab shows a compact preview of the focused source record. It does
not require or give special meaning to a label column.

The **History** tab shows the notes and review decisions previously recorded for
the focused record and can be hidden when it is not needed.

Changing focus loads the annotation state associated with the new row. The
panel can also reload the current record directly from stored artifacts.

Annotation Summary
******************

The **Annotation Summary** panel combines :code:`annotation.note` and
:code:`review.status` artifacts for the active dataset.

It shows one summary row per annotated record, including the latest review
state, confidence, tags, label suggestion, note and review counts, latest note
and relevant timestamps.

The summary is rebuilt from stored artifacts rather than treated as an
independent persistent copy. It refreshes when annotation or review artifacts
are created for the active dataset.

From the summary panel you can:

* refresh the current summary;
* create a derived **Annotation Summary** dataset;
* create a CSV artifact containing the summary.

Action
------

**Build Annotation Summary** creates a :code:`table.annotations_summary`
artifact containing the latest review state and annotation-history summary for
each annotated record in the active dataset.

Creating a derived dataset or CSV export is handled separately by the
**Annotation Summary** panel.

Artifacts
---------

* :code:`annotation.note`
* :code:`review.status`
* :code:`table.annotations_summary`
* :code:`annotation.summary.csv`

Persistence
-----------

Artifacts are the canonical shared record of completed annotations and review
decisions.

The **Annotations** panel persists only small UI state needed to restore the
workspace, including the current draft, history visibility and active tab.
Annotation and review history is reconstructed from artifacts rather than
stored as a second persistent copy inside the panel.

The **Annotation Summary** panel does not persist its generated table. The
summary is rebuilt from the canonical annotation and review artifacts when the
panel is opened or refreshed.

.. caution::

   Changing a dataset ID or its stable row identifiers can separate existing
   annotations from the replacement dataset. Preserve both when annotations
   must remain linked.