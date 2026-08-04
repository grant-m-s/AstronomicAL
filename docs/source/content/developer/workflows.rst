Workflow Plugins
================

What is a Workflow?
-------------------

A workflow combines existing platform capabilities into a coherent research
task.

Examples include:

* active-learning loops;
* review queues;
* model audit and comparison;
* staged data-preparation or inspection flows.

A workflow should orchestrate panels, actions, services, datasets, selections
and artifacts rather than introducing a second runtime architecture.

Registering a Workflow
----------------------

Register a workflow through :code:`api.register_workflow(...)`:

.. code-block:: python

   api.register_workflow(
       id="review",
       title="Example Review Workflow",
       builder=build_review_workflow,
       description="Opens the panels used for a review task.",
       category="Workflows",
       settings_schema={
           "type": "object",
           "properties": {
               "batch_size": {
                   "type": "integer",
                   "default": 50,
               },
           },
       },
   )

The local ID is namespaced by the plugin ID in the same way as panel and action
registrations.

Workflow Builder
----------------

The registered builder is called through
:code:`context.plugins.build_workflow(...)`.

The manager can pass :code:`context`, the request dictionary and itself when the
builder accepts those arguments:

.. code-block:: python

   def build_review_workflow(context, request=None, manager=None):
       request = request or {}

       context.plugins.open_panel(
           "core.record_browser.browser",
           context,
       )

       return {
           "workflow": "review",
           "dataset_id": request.get("dataset_id"),
       }

The builder's return value is not automatically interpreted as a special
workflow-state object. It is simply returned to the caller.

.. caution::

   Workflow builders run synchronously. Keep the builder itself lightweight.
   Expensive computation, remote access or model work should be delegated to
   registered actions or :code:`context.jobs`.

What a Workflow Registration Declares
-------------------------------------

The current workflow registration can declare:

* ID and title;
* description, category, icon and tags;
* a builder function;
* a settings schema;
* required Python packages;
* optional Python packages.

For example:

.. code-block:: python

   api.register_workflow(
       id="audit",
       title="Model Audit",
       builder=build_audit,
       requires=["scikit-learn>=1.5"],
       optional_requires=["shap"],
   )

Missing required Python packages prevent the workflow from being built.
Unavailable optional packages are reported as warnings and should disable only
the optional integration.

Plugin Dependencies
-------------------

Dependencies on other AstronomicAL plugins belong in the plugin manifest's
:code:`requires_plugins` field rather than on
:code:`register_workflow(...)`.

.. code-block:: python

   manifest = PluginManifest(
       id="workflow.review",
       name="Review Workflow",
       version="0.1.0",
       requires_plugins=[
           "core.record_browser",
           "core.visualisation",
       ],
   )

Required plugins are validated when the workflow plugin itself is enabled.

Mappings
--------

:class:`WorkflowRegistration` does not currently contain
:code:`required_mappings` or :code:`optional_mappings`.

Declare semantic mappings on the panels and actions that actually consume those
columns. This keeps the requirement close to the component that understands its
meaning.

For example, a review workflow might open one panel that requires
:code:`record_id` and another astronomy panel that requires
:code:`coords.ra` and :code:`coords.dec`. The normal panel mapping gate then
handles those requirements.

If the builder itself needs a mapping before it can decide what to construct, it
should resolve or validate that mapping explicitly through
:code:`context.datasets`.

Expected Datasets
-----------------

Workflow registrations do not currently declare a formal list of expected
datasets.

Use the request dictionary and :code:`context.datasets` to resolve the working
dataset. Component actions can express dataset requirements through their
:class:`InputSpec`, and panels should show appropriate empty states when no
active dataset exists.

Do not create a private dataset registry inside the workflow.

Panels and Geometry
-------------------

Default workflow panels and geometry are currently orchestration decisions, not
fields on :class:`WorkflowRegistration`.

A workflow builder can open registered panels through
:code:`context.plugins.open_panel(...)` and supply workspace layout information
where required.

Prefer registered panels over constructing hidden workflow-only UI objects.
Registered panels participate in the normal workspace lifecycle, persistence,
mapping gate and plugin ownership rules.

Actions and Jobs
----------------

Reusable operations should normally be registered as actions rather than
implemented only inside the workflow builder.

This allows the same operation to be invoked from:

* the workflow;
* another panel;
* tests;
* command or menu UI;
* another compatible plugin.

Use job-backed actions or :code:`context.jobs` for slow work. Do not block the
workflow builder with model training, remote archive requests or large dataset
transforms.

Artifacts
---------

Workflow registrations do not currently declare formal input or output artifact
types.

Actions should declare their output artifact types, and panels can declare
produced artifact types where useful. The workflow can then connect those
components by storing results in :code:`context.artifacts` and passing artifact
IDs rather than large payloads.

Use stable, namespaced artifact types such as:

.. code-block:: text

   workflow.review.query
   workflow.review.summary
   model.audit.report

Avoid a Hidden Second Application
---------------------------------

A workflow should use the existing platform services:

.. code-block:: text

   source and working tables  -> context.datasets
   focused row and row sets   -> context.selection
   generated products         -> context.artifacts
   lightweight notifications  -> context.events
   expensive work             -> context.jobs
   live clients               -> context.services
   visible panels             -> context.workspace / context.plugins
   saved panel/layout state   -> context.persistence

Do not create a private dataset manager, event bus, selection model, job pool,
service registry or layout system inside a workflow plugin.

Domain Independence
-------------------

A generic workflow should communicate through semantic mappings, datasets,
focus, selections, actions and artifacts rather than importing domain-panel
implementations directly.

For example, a review workflow can update the shared focused record while an
astronomy plugin independently reacts by showing a spectrum or image for that
record.

This lets domain-specific panels participate without changing the generic
workflow.

Resume Behaviour
----------------

There is no separate automatic workflow-state persistence layer in the current
workflow registration API.

Build resumability from the platform state that already has a defined lifecycle:

* persistent panel instances and their small JSON-safe state;
* dataset identities, metadata and semantic mappings;
* focus and active-selection state;
* durable external result references where the workflow creates them.

Store only the identifiers and parameters required to reconstruct the task.

Do not serialize:

* running jobs or cancellation tokens;
* futures;
* live service objects or clients;
* Panel or Bokeh widgets;
* complete DataFrames;
* arbitrary model objects.

If a workflow requires durable research outputs across application sessions,
persist those products to an appropriate project or external location rather
than relying only on in-memory workflow objects or the runtime artifact
registry.
