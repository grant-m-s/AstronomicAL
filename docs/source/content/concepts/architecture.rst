.. _platform-concepts:

Platform and Plugin Architecture
================================

AstronomicAL is a Plugin Host
-----------------------------

The application core owns shared runtime infrastructure. Plugins contribute
panels, actions, workflows, services, artifact viewers and integrations without
owning the global application state.

.. code-block:: text

   Plugin panel, action or workflow
                |
                v
            AppContext
                |
                +-- datasets
                +-- selection
                +-- artifacts
                +-- events
                +-- jobs
                +-- services
                +-- workspace
                +-- navigation
                +-- plugins
                +-- persistence
                +-- runtime_status

Application paths such as the layout file and layout directory are also carried
on :code:`AppContext`.

Platform Responsibilities
-------------------------

The platform is responsible for:

* plugin discovery, activation and lifecycle;
* dataset registration, source access and mappings;
* focused rows, selection sets and record navigation;
* background jobs;
* artifact storage;
* service registration and ownership;
* workspace panel lifecycle, layout and persistence;
* event delivery and runtime diagnostics.

Plugin Responsibilities
-----------------------

A plugin declares what it contributes and which mappings, services or optional
packages it needs. Panel code should keep construction lightweight and clean up
subscriptions, callbacks, watchers and panel-owned job handles in
:code:`dispose()`.

Plugins should use the shared platform services rather than mutating the
workspace, another panel or process-global runtime state directly.

Plugin Types
------------

The following are useful roles rather than separate platform classes:

**Core plugins** provide generic tools such as record browsing, visualisation
and annotations.

**Workflow plugins** combine several operations into a larger process, such as
active learning.

**Domain plugins** provide specialist tools such as astronomical spectra,
cutouts or sky viewers.

**User plugins** provide locally installed or project-specific behaviour without
requiring that functionality to live in the platform core.

.. caution::

   Plugins should not communicate by directly calling each other's panels.
   Shared focus and selection belong in :code:`context.selection`, reusable
   derived results belong in :code:`context.artifacts`, and lightweight
   notifications belong in :code:`context.events`.
