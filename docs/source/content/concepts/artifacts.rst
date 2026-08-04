.. _artifacts-concept:

Artifacts
=========

What is an Artifact?
--------------------

An artifact is a reusable result created by a plugin or workflow.

Examples include:

* a trained model;
* a prediction table;
* a training curve;
* an active-learning query batch;
* an image cutout;
* a spectrum or SED;
* an annotation summary;
* an exported report.

Artifact Metadata
-----------------

An artifact may record:

* its type and identifier;
* the source dataset;
* related row identifiers;
* parameters used to create it;
* a small inline payload or a file location;
* additional provenance metadata.

Viewing Artifacts
-----------------

Plugins can register viewers for artifact types. For example, a cutout artifact
can be opened without repeating the original archive request.

Promoting an Artifact
---------------------

A tabular artifact may be promoted into a dataset when it becomes a working
input for later analysis.

.. note::

   Artifacts are not live services. An API client belongs in
   :code:`context.services`, while the result returned by that client normally
   belongs in :code:`context.artifacts`.

Persistence
-----------

The artifact interface is designed for reusable outputs, but the exact
persistence guarantees may depend on the current storage backend. Important
research products should also be exported to a known project location.
