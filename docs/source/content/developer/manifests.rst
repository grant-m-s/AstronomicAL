Plugin Manifests
================

Required Fields
---------------

A :class:`PluginManifest` requires:

* :code:`id`;
* :code:`name`;
* :code:`version`.

:code:`description` is optional but strongly recommended for user-facing
plugins.

.. code-block:: python

   from astronomicAL.platform.plugins import PluginManifest

   manifest = PluginManifest(
       id="astro.example",
       name="Example Astronomy Tool",
       version="0.1.0",
       description="Adds an example domain panel.",
   )

Keep the module-level manifest cheap to import. Do not import heavyweight
optional libraries merely to construct it.

Plugin IDs
----------

Use a stable identifier containing only letters, numbers, dots, underscores or
hyphens. A dotted namespace is recommended:

.. code-block:: text

   core.record_browser
   workflow.review
   astro.example
   integrations.example

The first character must be alphanumeric. Plugin IDs must also be unique across
the discovered plugin set.

Versions
--------

Use semantic-style versions where possible.

The manifest can declare AstronomicAL compatibility with
:code:`min_astronomical` and :code:`max_astronomical`. When the host version is
provided during validation, plugins outside those bounds are rejected.

Requirements
------------

Required Python packages belong in :code:`requires` as PEP 508 requirement
strings:

.. code-block:: python

   requires=[
       "astropy>=6",
       "requests",
   ]

Optional Python packages belong in :code:`optional_requires`. Missing required
packages prevent enablement; missing optional packages are reported as warnings.

Environment markers and version specifiers are supported where the Python
:mod:`packaging` library is available.

Heavy astronomy, image or ML packages should remain dependencies of the plugin
or individual contribution that needs them rather than being moved into the
platform core for convenience.

Plugin Dependencies
-------------------

Use :code:`requires_plugins` when the plugin cannot function unless another
AstronomicAL plugin is present and enabled:

.. code-block:: python

   requires_plugins=[
       "core.image",
   ]

A missing or disabled required plugin prevents enablement.

The manifest also exposes :code:`optional_plugins` for describing optional
relationships. The current PluginManager does not treat those entries as
required enablement dependencies, so code using an optional integration must
still detect its availability and degrade gracefully.

Capabilities and Tags
---------------------

:code:`capabilities` and :code:`tags` are free-form discoverability metadata.

Capabilities normally describe the kinds of contributions or platform features
the plugin provides, while tags help Plugin Manager and other UI group or
describe the plugin.

.. code-block:: python

   capabilities=["panel", "service", "artifact_viewer"]
   tags=["astronomy", "spectra"]

They are descriptive metadata rather than permission grants.

Other Metadata
--------------

A manifest may also provide:

* :code:`author`;
* :code:`homepage`;
* :code:`package`;
* :code:`metadata` for additional JSON-like descriptive values.

Managed or community plugin discovery may read the same manifest information
from a static manifest before importing the plugin's Python code.

Validation Errors
-----------------

Plugin validation and registration can report problems such as:

* invalid or duplicate plugin IDs;
* missing or incompatible required Python packages;
* invalid PEP 508 requirement strings;
* incompatible AstronomicAL versions;
* missing or disabled entries in :code:`requires_plugins`;
* duplicate panel, action, workflow, service or artifact-viewer registrations;
* import or :code:`register(api)` failures.

Optional Python requirements produce warnings rather than blocking enablement.
