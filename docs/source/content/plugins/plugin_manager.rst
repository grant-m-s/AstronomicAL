Plugin Manager
==============

Overview
--------

:code:`core.plugin_manager` is the main management surface for AstronomicAL's
plugin system. It shows plugin health, lifecycle state, open panels,
contributions, dependencies, discovery problems and managed package information
from one place.

What You Do
-----------

1. Use **Scan for plugins** when new local or installed plugins have been added.
2. Search or filter the plugin list by status, capability or plugin ID.
3. Select a plugin and review its status, source, contributions, open panels and
   dependency checks.
4. Enable or disable plugins that support lifecycle management.
5. Reload development plugins after changing their code.
6. Install, update or uninstall AstronomicAL-managed :code:`.alplugin`
   packages.
7. Review discovery issues when a plugin candidate could not be inspected.

.. TODO: Add an image of the plugin list and selected-plugin details.

Panel
-----

**Plugin Manager**

The panel contains:

* summary cards for enabled, available and problematic plugins and open panels;
* a searchable, filterable plugin table;
* community-plugin controls;
* installation from local :code:`.alplugin` packages;
* selected-plugin details, contributions and technical information;
* lifecycle and managed-package controls;
* discovery issues from the latest plugin scan.

The selected-plugin view shows panels, actions, workflows, open instances,
readiness checks and technical details such as capabilities, tags, package
requirements, plugin dependencies, services, artifact viewers and settings.

Management Controls
-------------------

* **Refresh** redraws the current plugin state.
* **Scan for plugins** runs plugin discovery again.
* **Community plugins** controls whether third-party plugins may execute.
* **Enable plugin** activates an available plugin.
* **Disable plugin** disables it and closes owned panels where required.
* **Reload development plugin** reloads enabled development plugins.
* **Install** validates and installs a local :code:`.alplugin` package without
  enabling or executing it.
* **Update selected plugin** replaces an AstronomicAL-managed package. A running
  plugin is disabled first and is not restarted automatically.
* **Uninstall installed plugin** removes AstronomicAL-managed plugin code after
  confirmation while preserving plugin data.

Lifecycle Rules
---------------

Plugin Manager cannot disable or reload itself from inside its own panel.
Runtime registrations do not expose lifecycle or package-management controls.

Reload is available only for development plugins. Disabling or reloading a
plugin with open panels closes those panel instances and cancels its tracked
jobs.

Community plugins also require the global **Community plugins** switch. A
plugin may be configured as enabled but remain blocked while that switch is
off.

.. caution::

   Community plugins execute third-party Python code inside AstronomicAL. Only
   enable plugins you trust. Updating, disabling or reloading a plugin can also
   discard in-memory controller or service state.

.. note::

   A discovered plugin is not necessarily ready to run. Missing Python
   packages, required plugins, incompatible versions, registration failures or
   plugin-state errors can prevent activation. The selected-plugin readiness
   checks and discovery issues explain the relevant failure where available.