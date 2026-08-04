.. _datasets-concept:

Datasets and Dataset Sources
============================

What is a Dataset?
------------------

A dataset is source data, or a derived table that has been promoted into a new
working input. Every registered dataset has a stable identifier, a display name
and a source object.

The Active Dataset
------------------

Only one dataset is active at a time. Panels normally read from this dataset
unless a workflow explicitly points them to another one.

Changing the active dataset does not delete the previous dataset.

Dataset Sources
---------------

A dataset source hides the details of the storage backend. A source may be:

* an in-memory pandas DataFrame;
* a lazy Parquet or DuckDB relation;
* a manifest describing image assets;
* another backend implementing the dataset-source contract.

Plugins should ask the source for columns, batches or row identifiers rather
than assuming that the entire table is already in memory.

Scans and Batches
-----------------

Large datasets can be processed in batches. This lets a plugin calculate a
result without materialising every row at once.

.. note::

   Calling a full DataFrame conversion may be convenient, but it can remove the
   main benefit of a lazy dataset source.

Derived Datasets
----------------

A filtered table, transformed table or selected subset may be registered as a
new dataset. Its metadata should record the source dataset and the operation
that created it.

Dataset or Artifact?
--------------------

Use a dataset when the result will become an input to later exploration.

Use an artifact when the result is a generated product, such as a model,
prediction table, spectrum, cutout or report.

An artifact containing tabular data can later be promoted to a dataset when
continued exploration is required.
