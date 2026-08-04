Creating Artifacts
==================

Artifact Types
--------------

Use stable dotted names:

.. code-block:: text

   example.measurement
   example.report
   example.preview

Metadata and Provenance
-----------------------

Record the dataset, row identifiers, parameters and software information needed
to understand the output.

Payload or URI
--------------

Small results may be stored inline. Large arrays, tables and models should use a
file-backed payload or URI.

Creating an Artifact
--------------------

Prefer returning an artifact result from an action so the plugin manager can
register and publish it consistently.

Artifact Viewers
----------------

Register a viewer when the artifact has a useful visual representation.

Missing Payloads
----------------

A viewer should handle a missing or moved sidecar file with a clear message.

Promoting Tables
----------------

When a table artifact becomes a working input, register it as a dataset and keep
the artifact ID in the dataset provenance.
