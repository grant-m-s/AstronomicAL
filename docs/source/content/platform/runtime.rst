Runtime and Background Work
===========================

Job Manager
-----------

The Job Manager is the shared place for expensive or blocking work.

It currently uses a thread pool and tracks job ownership, timing, results,
failures, cancellation, and recent history.

Plugins should prefer it over creating their own unmanaged executors.

Cancellation
------------

Cancellation is cooperative once work has started. Long-running tasks should
check the provided cancellation signal at sensible points.

Deduplication
-------------

A stable job key can prevent the same expensive request from being queued or run
more than once.

This is useful for repeated data preparation, archive requests, or other work
triggered by rapid UI changes.

UI Completion
-------------

Worker threads must not update Panel/Bokeh models directly.

The Job Manager captures the relevant document and routes visible completion
callbacks back to the UI thread where required.

Runtime Diagnostics
-------------------

The platform records recent jobs, errors, event traces, slow callbacks, and
interface status.

The application header provides a compact runtime summary with access to more
detail when something is slow or fails.

Performance Guidelines
----------------------

Keep UI callbacks short.

Avoid loading an entire large dataset when a bounded read will do.

Use bounded caches and invalidate them by dataset identity or fingerprint.

For repeated remote requests, prefer cancellation, deduplication, and cached
results where appropriate.
