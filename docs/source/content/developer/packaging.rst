Packaging and Distributing Plugins
==================================

AstronomicAL supports normal Python distributions with plugin entry points,
managed ``.alplugin`` archives installed through Plugin Manager, and local
development search paths. The external
`Hello World repository <https://github.com/grant-m-s/AstronomicAL-Hello-World>`_
is the maintained starter template for the managed Marketplace workflow.

.. important::

   The managed ``.alplugin`` format and the normal Python-package
   entry-point route use different discovery/packaging conventions. Choose
   one release path and test it end to end. See :doc:`manifests`.

Local development
-----------------

During development, place a directory containing ``plugin.py`` in a
configured local plugin search path. A bundled plugin lives under
``astronomicAL/plugins/``; a normal third-party plugin should not need to
edit the main repository. A local development plugin can use a
module-level ``PluginManifest`` and ``register(api)``; a static JSON/TOML
manifest is also supported by the discovery layer.

Development plugins may be imported to inspect their metadata. Keep
import-time code cheap, avoid network calls and only create long-lived
objects when the platform enables or instantiates the relevant feature.

Python package distribution
---------------------------

Installable Python packages are discovered from the
``astronomical.plugins`` entry-point group. Example project metadata:

.. code-block:: toml

   [project]
   name = "astronomical-example-plugin"
   version = "0.1.0"
   dependencies = ["AstronomicAL"]

   [project.entry-points."astronomical.plugins"]
   example = "astronomical_example_plugin.plugin"

The referenced module should expose a ``PluginManifest`` and
``register(api)``:

.. code-block:: python

   from astronomicAL.platform.plugins import PluginManifest

   manifest = PluginManifest(
       id="astro.example",
       name="Example Tool",
       version="0.1.0",
   )

   def register(api):
       ...

The host also recognises an ``astronomical.plugin_manifests`` entry-point
group for static-manifest discovery. These are **Python packaging entry
points**, not files expected in a managed ``.alplugin`` release. A Python
distribution should declare its dependencies in standard packaging
metadata as well as the plugin manifest where applicable.

Managed ``.alplugin`` release layout
------------------------------------

The managed plugin archive is a ZIP with a static root manifest and the
plugin's Python entry module. An example minimal archive contains:

.. code-block:: text

   astronomical-plugin.json
   plugin.py
   README.md

The root ``astronomical-plugin.json`` supplies plugin identity,
compatibility and dependency metadata for static validation. The runtime
``plugin.py`` implements ``register(api)`` and may implement
``on_enable(...)`` and ``on_disable(...)``. A separate Python
``PluginManifest`` is not required by the Hello World managed-plugin
pattern. See :doc:`manifests` for the current field names.

Installing a managed archive validates its structure and records the
package without automatically enabling its Python code. Managed updates
are staged and verify the target plugin identity; enabled plugins are
disabled before replacement rather than being silently restarted.
Uninstalling plugin code should not remove user data written to an
appropriate location outside the installed plugin directory.

Building a Marketplace release
------------------------------

The
`Hello World template <https://github.com/grant-m-s/AstronomicAL-Hello-World>`_
provides ``astronomical-plugin.json``, ``plugin.py``, a release build script
and a GitHub Actions workflow. For a new external plugin:

#. Create a repository from the template.
#. Update ``astronomical-plugin.json`` with your stable plugin ID, version,
   Python requirements, required/optional plugins and descriptive metadata.
#. Implement ``register(api)`` in ``plugin.py``. Keep local tests and any
   needed assets in the repository.
#. Verify discovery, enablement, panel/action operation, disablement,
   cleanup and workspace restoration against the supported host version.
#. Build and inspect a local archive before tagging a release.
#. Tag a version consistent with the manifest, and publish the GitHub
   release assets.
#. Submit the GitHub *release page URL* to the Marketplace's plugin-release
   submission process.

With the template's release script, a local build uses:

.. code-block:: console

   python tools/build_release.py --repository YOUR-USERNAME/YOUR-PLUGIN

For a manifest with ID ``community.example`` and version ``1.0.0``, the
script produces an ``.alplugin`` archive and ``marketplace-release.json``
release metadata in ``dist/``. The metadata records the versioned release
asset URL, SHA-256 digest, archive size and plugin manifest. Do not
manually substitute unverified values in this metadata.

The template's release workflow is triggered by a version tag such as
``v1.0.0`` and publishes the distributable archive and release metadata as
GitHub release assets. The Marketplace submission workflow validates the
release, compares the package and manifest to the metadata, and prepares a
catalogue change for review. Submission verification does not execute the
plugin's Python code.

For the authoritative template steps, see its
`README <https://github.com/grant-m-s/AstronomicAL-Hello-World/blob/master/README.md>`_
and the
`Marketplace repository <https://github.com/grant-m-s/AstronomicAL-Marketplace>`_.

Python and plugin dependencies
------------------------------

Required Python requirements go in the manifest's ``requires`` list as
PEP 508 strings, and optional packages in ``optional_requires``. Host-owned
packages must not be silently replaced by an external plugin's dependency
installation. The managed dependency environment can supply compatible
additional packages, but host dependency conflicts can reject the install.

Declare AstronomicAL plugin dependencies separately with
``requires_plugins`` and ``optional_plugins``. The latter must be handled
at runtime when absent or disabled. Heavy astronomy, image and
machine-learning packages should stay with the feature that uses them,
not be added to the platform for convenience.

Compatibility
-------------

Specify host-version bounds if a plugin relies on a particular platform
release:

.. code-block:: python

   manifest = PluginManifest(
       id="astro.example",
       name="Example Tool",
       version="0.1.0",
       min_astronomical="0.8.0",
       max_astronomical="0.9.99",
   )

These numbers are illustrative, not an assertion of compatibility with
those host versions. Document support ranges, migration notes, required
plugins, optional integrations and large-data requirements in release
notes. There is no separate plugin-contract-version manifest field at
``90a912c``.

Trust and permissions
---------------------

Installed third-party plugins run Python in the AstronomicAL application
process. The static manifest and checksum checks protect parts of the
installation pipeline; **they do not sandbox or audit enabled code**.

A third-party plugin should document its filesystem access, network
requests, authentication/credential use, native dependencies and any
ability to execute user-provided or remote code. Community plugin
activation is separately controlled by the host's global community-plugin
setting and individual enablement preference.

Release checklist
-----------------

Before publishing or updating an external plugin:

* Verify the manifest ID and version match the release tag and metadata.
* Test the exact built archive, not only source imports.
* Test enable, disable, re-enable and cleanup of subscriptions, jobs and
  owned services.
* Test missing required and optional Python/plugin dependencies.
* Test lazy dataset backends if the plugin reads tables.
* Test workspace restore with and without the plugin installed.
* Verify that generated data and caches are outside the installed plugin
  code directory.
* Test inactive community-plugin discovery (metadata inspection should not
  execute code) and user-controlled activation.
* Publish a clear changelog and installation instructions.

Updates involving already imported Python packages or native extensions
may require an application restart before all loaded modules reflect the
new versions.
