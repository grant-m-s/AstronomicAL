.. _home:

Welcome to AstronomicAL's Documentation!
========================================

A plugin-based platform for interactive visualisation, inspection, labelling,
and classification of scientific data.

AstronomicAL is a local, human-in-the-loop analysis platform for working with
scientific datasets. It can be used to inspect records, combine contextual
information, create reliable labels, and build model-assisted workflows such
as active learning.

Although AstronomicAL has been heavily used for astronomy applications, the
plugin-based system allows domain-specific tools to be created for other areas
of scientific research. Astronomy remains an important part of the project,
but astronomy-specific tools no longer need to be installed or used by every
researcher.

.. Add image: overview of a complete AstronomicAL workspace.

Statement of Need
-----------------

Modern datasets are often too large to inspect manually, while their labels may
be incomplete, noisy, or expensive to create. A model can only be as reliable
as the data used to train and test it.

AstronomicAL brings the dataset, model, and expert into one workspace. Users
can inspect difficult records alongside the information they need, record a
decision, and immediately continue the workflow.

.. note::

   Active learning remains a major use case, but it is now one workflow built
   from plugins rather than the fixed identity of the application.

What Has Changed?
-----------------

Earlier versions of AstronomicAL used a fixed dashboard and a shared configuration object. The new application is assembled from plugins.

The platform provides shared infrastructure for:

* datasets and semantic column mappings;
* focused records and selection sets;
* background jobs;
* reusable artifacts and services;
* events between independent components;
* workspace layout and persistence;
* plugin discovery and lifecycle management.

Plugins provide the visible panels, integrations, and research workflows.

Where Should I Start?
---------------------

If you are new to AstronomicAL, start with
:doc:`content/getting_started/installation` and then read
:doc:`content/getting_started/interface`.

For some quick-start example dataset and layout, start here:
:doc:`content/getting_started/examples`.

To understand the main ideas used throughout AstronomicAL, see
:doc:`content/concepts/index`.

For documentation on the plugins included with AstronomicAL, see
:doc:`content/plugins/index`.

Plugin authors should begin with
:doc:`content/developer/first_plugin` and then read
:doc:`content/developer/plugin_contract`.

Contributors working on the shared application infrastructure should see
:doc:`content/platform/index`.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   content/getting_started/index
   content/concepts/index
   content/plugins/index

.. toctree::
   :maxdepth: 2

   content/developer/index
   content/platform/index

.. toctree::
   :maxdepth: 2

   content/reference/index
   content/troubleshooting/index

.. toctree::
   :maxdepth: 2

   content/project/index