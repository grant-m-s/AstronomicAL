Application Context and Shared-State Ownership
==============================================

``AppContext`` is the runtime dependency object passed to plugin panel
factories, action handlers, service factories and workflow builders. It is
implemented in ``astronomicAL/platform/context.py``. Plugins should use
this explicit object instead of accessing another plugin's private UI or
keeping duplicate process-global state.

The platform is a domain-neutral plugin host. A source record can represent
an astronomical object, video, transaction, document or another entity.
Domain-specific interpretations belong in plugins.

Context attributes at ``90a912c``
---------------------------------

The ``AppContext`` dataclass has these primary services:

.. list-table:: Shared runtime services
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Responsibility
   * - ``context.datasets``
     - Authoritative datasets, active dataset, column mappings and bounded
       data access; supports pandas and lazy ``DatasetSource`` backends.
   * - ``context.selection``
     - Authoritative current focused record and selected row-ID sets.
   * - ``context.navigation``
     - Optional record-navigation manager for moving through the current
       dataset or selected-record scope.
   * - ``context.events``
     - Notifications between loosely coupled features and plugins.
   * - ``context.artifacts``
     - Reusable computed outputs, their metadata and identifiers.
   * - ``context.jobs``
     - Background work, cancellation and completion callbacks.
   * - ``context.workspace``
     - Panel instances and workspace composition.
   * - ``context.services``
     - Live runtime capabilities, such as API clients and media resolvers.
   * - ``context.plugins``
     - Optional plugin manager: registered panels, actions, workflows and
       plugin lifecycle.
   * - ``context.persistence``
     - Optional workspace save/restore facility.
   * - ``context.runtime_status``
     - Optional runtime-status/diagnostics service.

The context also contains application paths (``layout_file``,
``layout_directory``) and host-managed plugin facilities such as
``plugin_state``, ``plugin_activation``, ``installed_plugins``,
``plugin_installer``, ``marketplace``, ``marketplace_planner``,
``marketplace_updates`` and ``marketplace_installer``. These additional
fields are optional and should only be used for features that explicitly
need the corresponding host capability.

.. important::

   ``AppContext`` at this baseline has **no** ``context.config`` field.
   Older documents describing it as a transitional plugin API are obsolete.
   New plugins must use the explicit services above. Do not add a
   ``context.config`` dependency when migrating legacy code.

Ownership rules
---------------

The division of responsibilities is intentional:

.. list-table:: Where state belongs
   :header-rows: 1
   :widths: 37 63

   * - Information
     - Owner
   * - Source table or newly promoted working table
     - ``context.datasets``
   * - Current row and multi-row selection
     - ``context.selection``
   * - Navigation order and scope
     - ``context.navigation`` when available
   * - Scores, reports, annotations exported as reusable results
     - ``context.artifacts`` or an explicitly persisted resource referenced
       by an artifact
   * - Change notifications
     - ``context.events``
   * - Remote fetch, preprocessing or model training
     - ``context.jobs``
   * - API client, live database connection, shared resolver
     - ``context.services``
   * - Individual panel geometry and instance lifecycle
     - ``context.workspace``
   * - Panel-specific controls and their compact saved values
     - The owning panel controller and :doc:`persistence`

A derived table becomes a dataset when it is intended to serve as a new
working input. An intermediate result or report usually remains an
artifact. Do not use live services as a dumping ground for cached analysis
outputs. See :doc:`dataset_programming` and :doc:`artifacts`.

Dataset access
--------------

Prefer backend-neutral, bounded operations. For example:

.. code-block:: python

   dataset_id = context.datasets.active_id()
   if dataset_id is not None:
       columns = context.datasets.list_columns(dataset_id)
       source = context.datasets.get_source(dataset_id)
       # Use source capabilities or bounded manager reads for actual rows.

Do not assume ``get_df()`` can materialize every dataset. Whole-dataframe
reads can be blocked for lazy Parquet/DuckDB sources. Read only the needed
columns or rows, stream batches where appropriate and resolve record IDs
through semantic mappings. See :doc:`dataset_programming`.

Focus and selection
-------------------

Single-record focus and multi-row selection are separate states. A click
that identifies one record should normally change focus; lasso/brush tools
should update the selection set. Other plugins should read the canonical
state from ``context.selection`` rather than importing the originating
panel.

.. code-block:: python

   focus = context.selection.get_focus()
   # Focus is a FocusState object, not an event payload dict.
   if focus is not None and focus.dataset_id is not None:
       dataset_id = focus.dataset_id
       row_id = focus.row_id

Subscriptions to ``selection.focus.changed`` and
``selection.focus.cleared`` should prompt a fresh read of canonical state.
This pattern also avoids relying on event payloads containing large
selected-row sets. Selection-related event payloads may be bounded or
abbreviated; read the authoritative state when full details are needed.
See :doc:`events` and :doc:`panels`.

Mappings
--------

Semantic mappings describe what a source column *means*, rather than
requiring every dataset to use identical names. For instance a plugin can
require ``record_id`` or ``video.uri`` without expecting the dataset to
have those exact column labels.

Declare required and optional mappings on the panel or action registration.
The host gates functional panels on unresolved required mappings. The
plugin should resolve actual source column names through the dataset
mapping system, not hard-code columns such as ``id`` or ``ra``. See
:doc:`mappings`.

Events, jobs and document-thread safety
---------------------------------------

Events communicate that a change occurred; they are not a transport for
large tables, frames or model objects. Store reusable outputs in
``context.artifacts`` and broadcast identifiers or small summaries.
Prefer namespaced topics for plugin-specific notifications.

Use ``context.jobs`` for slow, cancellable operations. If a job is owned by
a panel, cancel it during disposal. Cancellation is cooperative; callbacks
must verify that a result still matches the current dataset, row and
request generation before replacing any visible state.

Panel/Bokeh widgets belong on the document/UI thread once attached to a
live session. Worker functions must not mutate widgets directly. The
manager's UI completion callbacks can arrange document-thread updates when
a document was captured at submission. See :doc:`jobs` and :doc:`panels`.

Cross-plugin integration
------------------------

Prefer registered actions, services, events and artifacts over imports of
another plugin's internal view or controller implementation.

For example, the bundled ``core.video`` plugin owns temporal video state
and exposes a provider-neutral overlay registry. Other plugins can provide
annotation overlays through that contract while the platform continues to
own the focused *video record*. The host does not need to know how the
annotation provider stores masks or tracks.

A cross-plugin integration should specify the required plugin/service,
check availability when optional, and handle disablement. Plugin IDs,
action IDs, service keys, artifact types and semantic mappings should be
stable enough for independently released consumers. See :doc:`plugin_api`
and :doc:`manifests`.

Workspace and plugin persistence
--------------------------------

The platform stores panel identities, instances, layout and relevant
workspace information. Each controller should save only compact JSON-safe
state that the platform does not own: selected columns, view mode, display
preferences or references to stored artifacts.

Do not persist DataFrames, Panel widgets, JobManager handles, event
subscriptions or live service objects in controller state. Panel state
versions are recorded, but migration between versions is not automatic.
See :doc:`persistence`.

A plugin author's migration checklist
-------------------------------------

When moving legacy AstronomicAL functionality into a plugin:

#. Inject ``context`` instead of reading process-global application state.
#. Register source and derived working tables with ``context.datasets``.
#. Replace local authoritative current-record state with ``context.selection``.
#. Publish reusable calculated outputs via ``context.artifacts``.
#. Replace direct cross-panel calls with explicit services, actions or events.
#. Move slow processing to ``context.jobs``.
#. Register panels and let ``context.workspace`` own their instances.
#. Add an idempotent disposal path and compact, versioned controller state.
#. Test with lazy datasets and missing optional plugins, not just small
   in-memory data.
