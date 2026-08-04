Plugin Package Structure
========================

Minimal Structure
-----------------

For a directory-based local plugin, the discovery anchor is
:code:`plugin.py`:

.. code-block:: text

   example_plugin/
   └── plugin.py

An :code:`__init__.py` file may be included for conventional packaging, but it
is not required by AstronomicAL's local directory loader.

Larger plugins should split implementation code into sibling modules:

.. code-block:: text

   example_plugin/
   ├── plugin.py
   ├── panel.py
   ├── actions.py
   ├── service.py
   ├── assets/
   └── tests/

The Entry Module
----------------

:code:`plugin.py` is the runtime entry module. It should expose the plugin
manifest and :code:`register(api)` function.

Optional lifecycle hooks such as :code:`on_enable(...)` and
:code:`on_disable(...)` also belong at the plugin entry boundary when they are
needed.

Keep :code:`plugin.py` cheap to import. Development and bundled plugins without
a static manifest may be imported during discovery, before the plugin is
enabled.

Relative Imports
----------------

Local directory plugins are loaded as synthetic Python packages. AstronomicAL
loads :code:`plugin.py` as the package root and places the plugin directory on
its package path.

Normal relative imports therefore work:

.. code-block:: python

   from . import panel
   from .actions import build_result
   from .service import create_client

Do not depend on the value of :code:`__name__` or assume that the directory
name becomes the top-level Python module name. Local plugins receive a generated
synthetic package name.

Static Manifests
----------------

AstronomicAL can discover a plugin from a static JSON or TOML manifest without
importing its Python runtime first. Supported local manifest filenames include
forms such as:

.. code-block:: text

   astronomical-plugin.json
   astronomical-plugin.toml
   plugin.json
   plugin.toml

The managed user-plugin directory requires static-manifest discovery so
third-party Python code does not need to execute merely to appear in Plugin
Manager.

Development plugin paths can still use import-based discovery when no static
manifest is present.

Heavy Imports
-------------

Import optional or expensive dependencies inside the panel factory, action,
service factory or implementation module that actually needs them.

For example, a plugin with an optional astronomy client should avoid importing
that client merely to expose its manifest.

.. caution::

   Importing or discovering a plugin must not open a network connection, scan a
   large directory, start a background thread, submit jobs or mutate global
   application state.

Local Discovery
---------------

AstronomicAL searches configured plugin locations for directory plugins
containing :code:`plugin.py`. Development search paths may also contain simple
single-file :code:`.py` plugins.

The application can additionally discover installed Python packages through the
AstronomicAL plugin entry-point groups.

For multi-file local plugins, prefer the directory structure above so sibling
modules can use ordinary relative imports.

Assets
------

Keep read-only static assets, templates and small example resources inside the
plugin package where practical. Resolve them relative to the plugin package
rather than from the process working directory.

Do not write generated results, caches or user data into the installed plugin
source directory. Managed plugin updates or uninstall operations may replace or
remove that code directory.

Generated tabular working data belongs in datasets, reusable derived results
belong in artifacts, and other persistent files should be written to an
explicit user, project or configured cache location.
