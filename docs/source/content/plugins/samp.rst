SAMP
====

Overview
--------

:code:`astro.samp` exchanges tables with TOPCAT and other applications that use
the Simple Application Messaging Protocol. It supports both directions: send
an AstronomicAL dataset or selection, and receive an external table as a new
lazy dataset.

What You Do
-----------

To send data:

1. Connect AstronomicAL to the running SAMP hub.
2. Open **SAMP Send**.
3. Choose the active dataset or only the active selection.
4. Choose the columns and a safe row limit.
5. Send the table to the selected SAMP client.

To receive data:

1. Open **SAMP Receive** and keep AstronomicAL connected to the hub.
2. Send or broadcast a VOTable from TOPCAT or another client.
3. Review the incoming table information.
4. Import it as a locally mirrored, Parquet-backed AstronomicAL dataset.
5. Resolve any required semantic mappings and continue working normally.

.. TODO: Add a GIF showing a selection sent to TOPCAT and a table returned.

Requirements
------------

The plugin uses Astropy, pandas, NumPy, PyArrow and DuckDB. A SAMP hub must be
available, commonly through TOPCAT or another VO application.

Panels
------

SAMP Receive
************

The receive panel listens for supported table messages. Incoming URLs are
mirrored locally before the table is parsed and registered. Large tables are
not parsed inside the SAMP callback, which keeps the hub connection responsive.

A received table can become a lazy Parquet dataset and can publish registration,
mapping and active-dataset events like any other loader.

SAMP Send
*********

The send panel transfers either the active dataset or the active selection. It
supports:

* choosing only required columns;
* applying a row limit;
* filtering to selected record IDs;
* using source-side projection where the dataset supports it;
* FITS output by default for larger transfers.

Service
-------

The SAMP bridge is a long-lived plugin service. It owns the hub connection and
send/receive callbacks, while the panels provide the user-facing controls.

Large Tables
------------

Column projection, selection filtering and row limits should be applied before
materialising a transfer. This is both faster and safer than sending every row
and column by default.

.. caution::

   SAMP sends data to another application on the connected hub. Confirm the
   recipient, selected columns and row limit before transferring confidential
   or very large data.
