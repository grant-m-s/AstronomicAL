Plugin Manifests
================

This page describes the manifest formats implemented at AstronomicAL
commit ``90a912c``. The old root-level ``PLUGIN_CONTRACT.md`` showed an
untyped Python dictionary with fields such as ``dependencies`` and
``python_requires``. That is **not** the authoritative manifest schema.
The current runtime format is ``PluginManifest``; managed community
packages use a static manifest read without executing plugin code.

Choose the manifest for your distribution path
----------------------------------------------

.. list-table:: Manifest and entry-point forms
   :header-rows: 1
   :widths: 30 35 35

   * - Plugin origin
     - Discovery metadata
     - Runtime entry
   * - Bundled / development directory
     - Usually a Python ``PluginManifest`` object in ``plugin.py``;
       can also supply a static manifest
     - ``register(api)``
   * - Installed Python package
     - Python entry point, optionally an entry point for static metadata
     - Python module providing ``manifest`` and ``register(api)``
   * - Managed community ``.alplugin``
     - Static ``astronomical-plugin.json`` at archive root
     - ``plugin.py`` implementing ``register(api)``; no module-level
       Python manifest is necessary for this path

A local development plugin may be discovered by importing ``plugin.py``;
therefore import-time code must remain cheap and side-effect free. Managed
community plugin metadata can be inspected without importing untrusted
plugin code. See :doc:`package_structure` and :doc:`packaging`.

Python ``PluginManifest``
-------------------------

Bundled and development plugins normally use:

.. code-block:: python

   from astronomicAL.platform.plugins import PluginManifest

   manifest = PluginManifest(
       id="astro.example",
       name="Example Plugin",
       version="0.1.0",
       description="Adds an example panel and action.",
       requires=["pillow>=10"],
       optional_requires=["av>=12"],
       requires_plugins=[],
       optional_plugins=["core.video>=0.2"],
       capabilities=["panel", "action"],
       tags=["example"],
   )

   def register(api):
       ...

The ``PluginManifest`` dataclass requires ``id``, ``name`` and ``version``.
``description`` is optional but strongly recommended. Supported fields:

.. list-table:: Current manifest fields
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``id``, ``name``, ``version``
     - Stable ID, display name and plugin version (required).
   * - ``description``
     - User-facing description.
   * - ``author``, ``homepage``, ``package``
     - Attribution, project URL and package metadata.
   * - ``min_astronomical``, ``max_astronomical``
     - Optional host-version compatibility bounds.
   * - ``requires``, ``optional_requires``
     - Required and optional **Python package requirements** as PEP 508
       strings, such as ``"pillow>=10"``.
   * - ``requires_plugins``, ``optional_plugins``
     - Required and optional **AstronomicAL plugin** dependencies. May use
       version specifiers such as ``"core.video>=0.2"``.
   * - ``capabilities``, ``tags``
     - Informational feature and discovery labels.
   * - ``metadata``
     - Optional dictionary for additional descriptive metadata.

Do not confuse ``requires`` (Python packages) with ``requires_plugins``
(AstronomicAL extensions). An optional plugin is not guaranteed to be
installed or enabled; check for its action/service before using it.

Plugin IDs may contain letters, digits, dots, underscores and hyphens,
and must start with an alphanumeric character. Use stable dotted IDs, for
example ``community.example`` or ``astro.spectra``. IDs must be unique
within the discovered plugin set. Do not change a published ID merely to
rename the display title; layouts and other integrations can refer to it.

Use semantic-style release versions and explicit host bounds where needed.
The host can validate Python and plugin dependency versions, but a plugin
must still handle incompatible optional integrations gracefully.

Static manifest for managed plugins
-----------------------------------

The independently distributed
`Hello World template <https://github.com/grant-m-s/AstronomicAL-Hello-World>`_
uses this structure:

.. code-block:: text

   astronomical-plugin.json
   plugin.py
   README.md

Its manifest has the following form:

.. code-block:: json

   {
     "id": "community.example",
     "name": "Example Plugin",
     "version": "1.0.0",
     "description": "An example external AstronomicAL plugin.",
     "author": "Plugin Author",
     "requires": [],
     "optional_requires": [],
     "requires_plugins": [],
     "optional_plugins": [],
     "capabilities": ["panel"],
     "tags": ["example"]
   }

The runtime ``plugin.py`` for this managed-package pattern need only expose
``register(api)`` and any optional lifecycle hooks. The static manifest is
the source of discovery metadata; its plugin ID and version must match the
release metadata and package identity.

Development and local directory discovery also recognise static JSON or
TOML manifest filenames, including ``astronomical-plugin.json``,
``astronomical-plugin.toml``, ``plugin.json`` and ``plugin.toml``. The
managed ``.alplugin`` release template uses the root
``astronomical-plugin.json`` convention; use that for Marketplace packages
rather than relying on a more permissive development-path alias.

Compatibility and dependencies
------------------------------

Required Python requirements go in ``requires`` as PEP 508 strings. Put
optional packages in ``optional_requires``. The application reports
missing/incompatible required packages and can reject enablement;
optional packages do not make an entire plugin unavailable.

Use ``requires_plugins`` only for hard dependencies. It supports bare IDs
or versioned specifiers, for example:

.. code-block:: python

   requires_plugins=["core.image", "core.video>=0.2,<1"]

``optional_plugins`` is metadata for possible integrations, not a promise
that those plugins will be active. At the contribution level, the API can
also declare its own Python requirements with ``requires`` and
``optional_requires``; see :doc:`plugin_api`.

Avoid importing heavyweight optional libraries in order to create the
manifest. Import them only when the relevant action, panel or service is
actually used.

Discovery, validation and activation
------------------------------------

A plugin can be *discovered* without being *enabled*. This distinction is
particularly important for installed community plugins, whose Python code
must not execute merely for metadata inspection. Installation and
activation are separate operations. Community activation is subject to
global and per-plugin preferences.

The system may report:

* invalid IDs or duplicate plugin identities;
* missing or incompatible Python requirements;
* invalid dependency specifiers;
* an incompatible host version;
* absent or disabled required plugins;
* duplicate contribution registrations;
* import, registration or lifecycle-hook failures.

A static manifest, successful archive verification and Marketplace listing
do not constitute a code security audit. When enabled, the plugin executes
Python inside the AstronomicAL host process; see :doc:`packaging`.

Versioning and author responsibilities
--------------------------------------

There is no independent ``plugin_contract_version`` field in the current
manifest dataclass. Publish compatibility assumptions in the plugin's
release notes and use ``min_astronomical`` / ``max_astronomical`` for
host-version compatibility when needed. Version saved panel state
independently with the registration's ``state_version``; automatic state
migration is not implemented at this baseline. See :doc:`persistence`.

The static manifest and Python runtime manifest (where a plugin has both)
should describe the same plugin identity, version, dependencies and
compatibility information. Test the exact discovery and installation path
used by your users, not just import-based local discovery.
