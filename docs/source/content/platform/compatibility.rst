Compatibility and Migration
============================

Why Compatibility Exists
------------------------

AstronomicAL is moving older features onto the shared plugin and platform
contracts. Some compatibility paths remain so existing code can continue to
work during that migration.

They should not be treated as the preferred architecture for new features.

Preferred Direction
-------------------

When changing legacy code, prefer:

* :code:`AppContext` over process-global application state;
* :code:`DatasetSource` and bounded reads over automatic full DataFrame
  materialisation;
* semantic mapping requirements over hard-coded dataset column names;
* the Selection Manager and Record Navigation Manager over panel-local focus
  state;
* artifacts, services, and events over direct panel-to-panel communication;
* registered actions over reusable logic hidden inside UI callbacks;
* the Job Manager over plugin-owned background executors;
* workspace panel state over ad-hoc layout persistence.

Pandas Compatibility
--------------------

Pandas-backed dataset sources remain useful for smaller data and migrated code.

Code that requests a fully materialised DataFrame may still be supported, but
new large-data features should use the source contract where practical.

Import and API Compatibility
----------------------------

Compatibility aliases may remain while modules are reorganised. New code should
import from the current platform/plugin APIs rather than depending on legacy
paths solely because they still resolve.

Removing Compatibility Code
----------------------------

A compatibility path should only be removed once the code that depends on it
has been migrated and the relevant behaviour is covered by tests.

The goal is gradual migration without maintaining two competing architectures.
