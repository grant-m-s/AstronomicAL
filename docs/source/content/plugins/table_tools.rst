Table Tools
===========

Overview
--------

:code:`core.table_tools` performs two common table operations without requiring
a separate notebook: calculate a new column from an expression, or create a new
dataset from rows matching a boolean condition.

What You Do
-----------

To add a derived column:

1. Open **Table Transform** and review the available columns.
2. Enter an expression and preview the result on a bounded number of rows.
3. Choose a new column name.
4. Apply the expression and register the transformed dataset.

To create a subset:

1. Enter a boolean expression.
2. Preview how many rows match and inspect a sample.
3. Choose a dataset ID and display name.
4. Register the subset and optionally make it active.

.. TODO: Add an image showing expression preview and the resulting dataset.

Panel
-----

**Table Transform**

The panel brings together column inspection, expression preview and the two
registered transformations. Previewing first helps identify missing columns,
invalid syntax and unexpectedly broad filters before creating a full result.

Actions
-------

Add Derived Column
******************

**Add Derived Column** evaluates an expression and adds its result under a new
column name. Existing columns are protected from accidental overwrite.

DuckDB-backed sources can retain a lazy relational expression. Sources that
require pandas materialisation are written through the Parquet compatibility
path so that the resulting dataset can be accessed lazily afterwards.

Create Subset Dataset
*********************

**Create Subset Dataset** evaluates a boolean condition and registers only the
matching rows. Existing semantic mappings are copied to the derived dataset
where their columns remain available.

Expression Examples
-------------------

Create a colour-like derived value:

.. code-block:: text

   mag_g - mag_r

Create a high-score, low-redshift subset:

.. code-block:: text

   (score > 0.9) & (redshift < 1.0)

Use explicit parentheses when combining conditions so that the intended order
is clear.

Outputs and Provenance
----------------------

Both operations produce registered datasets rather than changing another
plugin's private dataframe. This allows the result to appear in the global
dataset selector and participate in mappings, selections, persistence and later
actions.

.. caution::

   Expressions are powerful and a syntactically valid result can still be
   scientifically wrong. Always review the preview, null behaviour, data type
   and output row count before using a transformed dataset for training or
   publication.
