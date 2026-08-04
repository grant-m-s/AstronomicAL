Using Background Jobs
=====================

Submitting a Job
----------------

Use :code:`context.jobs` for work that should not block the interface, including
remote requests, large transforms, training, prediction and expensive file IO.

A directly submitted job function should accept the reserved
:code:`cancel_token` keyword:

.. code-block:: python

   def work(*, dataset_id, cancel_token):
       if cancel_token.cancelled():
           return None

       return run_expensive_operation(dataset_id)

   handle = context.jobs.submit(
       work,
       title="Build result",
       key=f"example.build:{dataset_id}",
       dataset_id=dataset_id,
       on_done=on_done,
       on_error=on_error,
   )

The manager runs jobs through a shared :class:`~concurrent.futures.ThreadPoolExecutor`
and returns a :code:`JobHandle`.

Stable Keys
-----------

Use a stable :code:`key` when only one copy of the same request should run at a
time.

If the same key is submitted while a job is still active, the JobManager returns
the existing handle instead of starting duplicate work. New completion callbacks
are attached to that existing job.

Do not reuse one key for requests whose inputs or expected outputs differ.

Cancellation
------------

Call :code:`handle.cancel()` to request cancellation. This always sets the
cooperative cancellation token and also attempts to cancel the underlying future
if it has not started.

Running work must check the token at useful boundaries, for example:

* between dataset batches;
* between remote pages or requests;
* between training or inference stages;
* before expensive post-processing;
* before writing final output.

Cancellation is cooperative. Python code that is already running is not stopped
forcibly.

Completion
----------

Use :code:`on_done` and :code:`on_error` for completion handling.

The JobManager captures the current Panel/Bokeh document when each callback is
registered and schedules those callbacks back onto that document's next UI tick.
This makes normal widget and pane updates safe from job completion callbacks.

When no Panel document exists, such as in tests or non-UI use, callbacks run
immediately.

Do not update Panel or Bokeh objects from inside the worker function itself.

Ownership
---------

Jobs started by PluginManager-managed operations, such as job-backed actions and
panel construction, are tracked against their owning plugin and can be cancelled
when that plugin is disabled.

A panel that calls :code:`context.jobs.submit()` directly should retain its own
job handles and request cancellation from :code:`dispose()` when that work
should not outlive the panel.

.. code-block:: python

   def dispose(self):
       if self._disposed:
           return
       self._disposed = True

       if self._job_handle is not None:
           self._job_handle.cancel()
           self._job_handle = None

Stale Results
-------------

Cancellation alone is not enough for rapidly changing UI state. A remote request
may finish just as focus, dataset or settings change.

Compare completion results with the current request identity, generation or
dataset before replacing visible state.

CPU-Bound Work
--------------

The current JobManager uses threads. Native libraries that release the GIL can
still benefit from the shared thread pool, but pure-Python CPU-heavy work may
need a process-based or remote execution backend if it becomes a bottleneck.

.. caution::

   Do not create unmanaged threads inside panel callbacks. Use the JobManager so
   work participates in shared scheduling, diagnostics, deduplication and
   cooperative cancellation.
