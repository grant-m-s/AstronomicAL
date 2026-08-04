Datasets and Semantic Mappings
==============================

Dataset Manager
---------------

The Dataset Manager owns loaded dataset registrations, the active dataset,
metadata, and semantic column mappings.

Plugins should normally ask the platform for the dataset they need rather than
keeping their own application-wide copy.

Dataset Sources
---------------

:code:`DatasetSource` is the shared data-access contract.

A source can expose schema information, bounded reads, row lookup, sampling, and
other operations without requiring the whole dataset to be loaded into memory.

Current sources include:

* pandas-backed sources for compatibility and smaller in-memory data;
* Parquet/DuckDB-backed sources for lazy access to larger datasets.

Full DataFrame materialisation is still useful in some workflows, but new code
should avoid treating it as the default data-access pattern.

Large Datasets
--------------

When possible, request only the rows and columns needed for the operation.

For large catalogues this keeps memory use predictable and allows backends such
as Parquet and DuckDB to perform projection, filtering, sampling, and
aggregation efficiently.

Semantic Mappings
-----------------

Plugins should depend on the meaning of a column, not a specific column name.

For example, a plugin can request mappings such as:

.. code-block:: text

   record_id
   coords.ra
   coords.dec
   target_label

The platform resolves those concepts against the active dataset.

Required and Optional Mappings
------------------------------

A required mapping prevents a panel from becoming active until it can be
resolved. An optional mapping may improve a feature without blocking it.

When a required mapping is missing, the platform can request that the user
assign an appropriate dataset column.

Mapping Gates
-------------

Panels with mapping requirements may be wrapped by a mapping gate.

The gate waits until the active dataset satisfies the requirements, then creates
the real panel. It also protects against stale builds when datasets change
quickly and disposes the old child controller when rebuilding.

Restoration
-----------

A saved workspace may restore dataset and mapping information before the data
source itself is ready. Mapping requirements are re-evaluated once the dataset
becomes available rather than trusting stale state.
