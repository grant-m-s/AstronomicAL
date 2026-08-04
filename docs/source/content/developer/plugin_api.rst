Plugin API
==========

The Plugin API is the only supported registration surface.

Registering a Panel
-------------------

.. code-block:: python

   api.register_panel(
       id="panel",
       title="Example",
       factory=create_panel,
       category="Examples",
       required_mappings=["record_id"],
   )

Registering an Action
---------------------

.. code-block:: python

   api.register_action(
       id="calculate",
       title="Calculate Result",
       handler=calculate_action,
       run_in_job=True,
   )

Other Contribution Types
------------------------

A plugin may also register:

* dataframe actions;
* workflows;
* services;
* artifact viewers.

Namespacing
-----------

Local contribution IDs are joined to the plugin ID. The action above becomes:

.. code-block:: text

   example.plugin.calculate

Ownership
---------

Every contribution is owned by the registering plugin. This allows the manager
to remove it safely during rollback or disablement.

Duplicate Registration
----------------------

Duplicate IDs are errors unless the relevant registration explicitly supports
replacement.
