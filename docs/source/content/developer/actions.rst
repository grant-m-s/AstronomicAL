Actions and DataFrame Actions
=============================

Action Specifications
---------------------

An action declares its title, handler, input requirements, parameters, outputs
and whether it runs as a job.

Action Request
--------------

The request may contain:

* dataset ID;
* row IDs;
* selected columns;
* parameters;
* source artifact ID;
* origin.

Action Result
-------------

An action can return:

* a direct value;
* artifacts to register;
* datasets to register;
* events to publish.

.. code-block:: python

   return ActionResult(
       value={"rows": count},
       artifacts=[artifact],
       events=[EventResult("example.finished", payload)],
   )

Validation
----------

Validate required parameters before performing expensive work. Error messages
should explain what the user can change.

Asynchronous Actions
--------------------

Set :code:`run_in_job=True` for slow work and check the cancellation token during
long loops.

Idempotency
-----------

Where possible, use stable identifiers or job keys so repeated clicks do not
produce duplicate remote requests or conflicting outputs.
