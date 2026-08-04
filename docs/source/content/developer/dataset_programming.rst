Dataset Programming
===================

Dataset Manager
---------------

Use :code:`context.datasets` to list, register and activate datasets, resolve
semantic mappings and access backend-neutral dataset operations.

New code should prefer :code:`register_source()` or :code:`register_parquet()`.

Dataset Source
--------------

:code:`DatasetSource` is the canonical dataset payload. It provides
backend-neutral access to operations such as:

* column names and dtypes;
* row counts and filtered counts where supported;
* bounded pandas materialisation;
* row lookup by position or record ID;
* multi-row lookup by record ID;
* distinct values and column statistics;
* batched scans;
* backend-specific sampling or aggregation capabilities where implemented.

Use :code:`context.datasets.get_source(dataset_id)` when code needs direct
source capabilities, and prefer DatasetManager wrappers for common access.

Batch Processing
----------------

Use the manager-owned batch boundary for large scans:

.. code-block:: python

   for batch in context.datasets.iter_batches(
       dataset_id,
       columns=["record_id", "score"],
       batch_size=8192,
   ):
       process(batch)

The same API can accept a :code:`DatasetScan` object, or individual scan
arguments such as columns, filters, limits and shard information.

Avoid Full Materialisation
--------------------------

Do not call :code:`get_df()` or request an unbounded pandas conversion unless
the operation genuinely requires a complete DataFrame and the dataset is known
to be safe to materialise.

For normal inspection, prefer methods such as :code:`list_columns()`,
:code:`row_count()`, :code:`head()`, :code:`get_row_by_id()`,
:code:`distinct_values()` and :code:`column_statistics()`.

:code:`get_df()` remains a compatibility API and records materialisation
telemetry, including whether the request materialised the full dataset.

Registering Derived Data
------------------------

Register a new dataset when a result should become a continuing working input.

A plugin can register a :code:`DatasetSource` directly with
:code:`register_source()`, or register a Parquet result lazily with
:code:`register_parquet()`.

Actions can also return a :code:`DatasetResult`; the plugin manager processes
that result and registers the returned DataFrame as a dataset.

For replaceable derived columns that belong on an existing dataset, the
DatasetManager also supports named column overlays without creating a second
dataset identity.

Parquet Compatibility Path
--------------------------

Code that already produces a pandas DataFrame can write the result to Parquet
and call :code:`register_parquet()` so later access uses the lazy DuckDB-backed
source rather than retaining a large DataFrame as the canonical dataset.

Where row-count or column metadata is already known, pass those hints during
registration to avoid unnecessary inspection of large Parquet files.
