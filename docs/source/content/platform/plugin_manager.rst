Plugin Manager
==============

Purpose
-------

The Plugin Manager discovers plugins, validates their metadata, manages their
lifecycle, and owns the contribution registries used by the application.

Discovery
---------

Plugins can be discovered from bundled, development, or user/community
locations.

Static manifests allow the host to inspect basic plugin metadata before
importing and executing the plugin itself.

Contributions
-------------

A plugin can register capabilities such as:

* panels;
* actions and workflows;
* services;
* artifact viewers.

The Plugin Manager tracks which plugin owns each contribution so it can clean up
correctly during disablement.

Dependencies
------------

Required plugin dependencies are resolved before enablement.

Dependency ordering is deterministic, version requirements are checked, and
cycles or missing requirements are reported instead of being ignored.

Enablement is transactional across newly enabled dependencies: if part of the
activation fails, contributions created by that activation are rolled back.

Disablement
-----------

A plugin cannot be treated as a collection of permanent globals.

When it is disabled, owned jobs are cancelled where possible, panel instances
are closed, services and contribution records are removed, and lifecycle
cleanup is run.

Activation Policy
-----------------

Bundled and development plugins may be enabled automatically according to host
policy. Community plugins are also subject to the application's community-plugin
setting and persisted per-plugin activation state.

A dependency does not bypass those host trust decisions.

Python Dependencies
-------------------

Community plugins can use AstronomicAL's managed Python package overlay for
additional compatible dependencies.

The host environment remains authoritative: a plugin must not replace packages
owned by AstronomicAL. Dependency resolution is performed before the managed
overlay is replaced.

Some native-package changes may require an application restart if extension
modules are already loaded.

Trust and Security
------------------

Plugins are not sandboxed.

Enabling a plugin executes Python code with the permissions of the AstronomicAL
process. Packaging, validation, and activation controls reduce accidental or
unsupported installation states, but they do not isolate a plugin from the
host.

Third-party plugins should therefore be treated as code the user has chosen to
trust. Credentials and sensitive values should not be written into workspace
or artifact metadata unless they are explicitly protected.
