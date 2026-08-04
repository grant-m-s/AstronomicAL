Troubleshooting
===============

AstronomicAL Will Not Start
---------------------------

Run the start command without :code:`--show` and review the terminal output.

Enable boot diagnostics:

.. code-block:: bash

   ASTRONOMICAL_DEBUG_BOOT=1 panel serve astronomicAL

A Plugin Cannot Be Enabled
--------------------------

Open Plugin Manager and review its requirements and registration error.

A Panel is Waiting for Mappings
-------------------------------

Resolve the request in the application header. Confirm that the mapping belongs
to the active dataset.

A Saved Panel is Missing
------------------------

Install or enable the owning plugin. The placeholder should preserve the panel's
grid location.

A Job Failed
------------

Open runtime status and review the job error. Remote failures may be temporary;
validation errors normally require changing the request.

A Panel Shows an Older Record
-----------------------------

Change focus once more and check the Event Monitor. Report the issue if a stale
job completion replaced a newer request.

Memory Use is High
------------------

Check whether a plugin materialised a lazy dataset. Reduce selected columns,
batch size or image-gallery limits.

Remote Data is Unavailable
--------------------------

Confirm network access, credentials, archive coverage and service status.

Workspace Restore Failed
------------------------

Try loading into a fresh workspace and review each reported plugin, dataset or
state error separately.

Debugging Variables
-------------------

.. code-block:: bash

   ASTRONOMICAL_DEBUG_PLUGINS=1 panel serve astronomicAL --show
   ASTRONOMICAL_DEBUG_MAPPING=1 panel serve astronomicAL --show
   ASTRONOMICAL_DEBUG_PERSISTENCE=1 panel serve astronomicAL --show
