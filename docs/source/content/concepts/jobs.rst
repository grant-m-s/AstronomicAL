.. _jobs-concept:

Background Jobs
===============

What is a Job?
--------------

A job is work that should not block the interface. Typical jobs include model
training, remote archive requests, large imports and dataset exports.

The platform JobManager runs jobs through a shared thread pool and returns a
:code:`JobHandle`.

Job Status
----------

The JobManager records jobs as :code:`queued`, :code:`running`,
:code:`finished`, :code:`error` or :code:`cancelled`. Runtime diagnostics can
show active jobs and a bounded history of recently completed jobs.

Plugins may expose their own progress UI, but the core job snapshot does not
define a generic percentage-progress field.

Cancelling Work
---------------

Cancellation is cooperative. :code:`JobHandle.cancel()` requests cancellation,
sets the job's cancellation token and also tries to cancel the underlying
future before it starts.

A running function must check :code:`cancel_token.cancelled()` at suitable
points and return promptly when cancellation is requested.

.. caution::

   Closing a panel does not make arbitrary Python code stop instantly. A panel
   that owns job handles must request cancellation during disposal where that
   work should not outlive the panel.

Duplicate Work
--------------

A plugin can submit a stable job key. If another submission uses the same key
while that job is still active, the JobManager reuses the existing job and can
attach the new completion callbacks instead of starting duplicate work.

Stale Results
-------------

A completed job may refer to an older record, dataset or request generation.
Panels should compare the result with their current request before replacing
visible state.
