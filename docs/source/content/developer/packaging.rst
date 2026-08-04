Packaging and Distributing Plugins
==================================

Distribution Options
--------------------

AstronomicAL currently supports two main distribution paths:

* a normal installed Python package discovered through Python entry points;
* an AstronomicAL-managed :code:`.alplugin` archive installed through
  **Plugin Manager**.

Development plugins can also be loaded directly from configured local plugin
paths, but that is normally an authoring workflow rather than a release format.

Package Metadata
----------------

For a Python distribution, use normal package metadata with an explicit package
version and licence. Keep the AstronomicAL plugin manifest version aligned with
the released plugin version where practical.

The runtime plugin module must expose a :class:`PluginManifest` and
:code:`register(api)`.

Entry Points
------------

Installed Python packages are discovered through the
:code:`astronomical.plugins` entry-point group.

For example, with :file:`pyproject.toml`:

.. code-block:: toml

   [project]
   name = "astronomical-example-plugin"
   version = "0.1.0"
   dependencies = [
       "AstronomicAL",
   ]

   [project.entry-points."astronomical.plugins"]
   example = "astronomical_example_plugin.plugin"

The referenced module can expose the normal runtime entry points:

.. code-block:: python

   manifest = PluginManifest(
       id="astro.example",
       name="Example Astronomy Tool",
       version="0.1.0",
   )

   def register(api):
       ...

AstronomicAL also recognises the
:code:`astronomical.plugin_manifests` entry-point group for static-manifest
discovery. Static manifests let the host inspect plugin metadata without first
executing the plugin runtime.

Static Manifests
----------------

A static JSON or TOML manifest is especially important for community or managed
plugins, where AstronomicAL may need to inspect the plugin before allowing its
Python code to run.

Supported local manifest names include:

.. code-block:: text

   astronomical-plugin.json
   astronomical_plugin.json
   plugin.json
   manifest.json
   astronomical-plugin.toml
   astronomical_plugin.toml
   plugin.toml
   manifest.toml

The static manifest should describe the same plugin identity, version,
requirements and compatibility information as the runtime manifest.

Managed ``.alplugin`` Packages
------------------------------

Plugin Manager can install a local :code:`.alplugin` archive into AstronomicAL's
managed user-plugin area.

Installation validates and records the package without enabling or executing the
plugin. Managed updates verify that the archive belongs to the selected plugin;
a running plugin is disabled before replacement and is not restarted
automatically.

Managed uninstall removes the installed plugin code while preserving plugin
data managed outside that code directory.

Use the managed package format when distribution should be handled by
AstronomicAL itself. Use a normal Python package and entry point when the plugin
should be installed and upgraded through the Python packaging environment.

Dependencies
------------

Put required Python dependencies in the plugin manifest's :code:`requires`
field using PEP 508 requirement strings. Optional packages belong in
:code:`optional_requires`.

A separately distributed Python package should also declare the dependencies
needed to import and run that package in its normal package metadata.

For AstronomicAL-managed plugins, compatible Python dependencies can be placed
in the managed plugin dependency environment. The host environment takes
precedence, and dependency conflicts with host-owned packages are rejected
rather than silently replacing the application's versions.

Heavy astronomy, image and machine-learning dependencies should stay with the
plugin that requires them rather than being added to the platform core.

Compatibility
-------------

Declare host-version bounds in the plugin manifest where required:

.. code-block:: python

   manifest = PluginManifest(
       id="astro.example",
       name="Example Astronomy Tool",
       version="0.1.0",
       min_astronomical="0.8.0",
       max_astronomical="0.9.99",
   )

Release documentation should also describe:

* supported AstronomicAL versions;
* required and optional Python packages;
* required AstronomicAL plugins;
* any workspace, settings or artifact migrations;
* optional external integrations and services.

There is no separate manifest field for a plugin-contract version in the
current API, so document contract assumptions in the release notes when they
matter.

Trust
-----

A plugin executes Python code inside the AstronomicAL process. It is not a
sandbox.

Distribution pages should clearly state:

* files and directories the plugin reads or writes;
* network services it contacts;
* credentials, tokens or authentication it can use;
* external executables or native libraries it invokes;
* whether remote code or user-provided code can be executed.

Community plugins remain subject to AstronomicAL's activation policy even when
they have already been discovered or installed.

Release Checklist
-----------------

Before release:

* validate the runtime and static manifests;
* test discovery from the intended distribution format;
* test enablement, disablement and cleanup;
* test required and optional dependency handling;
* test with community-plugin execution disabled;
* test installation or entry-point discovery in a clean environment;
* test workspace restore with the plugin available and unavailable;
* verify generated files are written outside the installed source directory;
* build the documentation and changelog;
* publish a minimal installation and usage example.

.. caution::

   Updating Python packages that contain native extensions may require an
   application restart before every loaded module reflects the new version.
   Do not rely on in-process plugin reload as a general replacement for process
   restart after dependency changes.
