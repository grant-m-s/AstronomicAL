Resource Monitor
================

Overview
--------

:code:`core.resources` provides a top-like view of the machine running the
AstronomicAL server. It helps explain whether slow imports, training runs,
remote processing or several open panels are limited by CPU, memory, disk,
network or GPU resources.

What You Do
-----------

1. Open **Resource Monitor** before or during a demanding operation.
2. Watch overall CPU and memory use as well as the AstronomicAL process.
3. Check disk activity when importing, converting or materialising datasets.
4. Check network activity during archive, SAMP or remote-image work.
5. Inspect NVIDIA GPU information when a compatible GPU and tools are present.
6. Compare the measurements with running jobs shown by Runtime Status.

The panel is diagnostic only. It does not automatically stop a job or change a
recipe's resource settings.

.. TODO: Add an image of the monitor during a model-training job.

Optional Requirements
---------------------

The :code:`psutil` package enables detailed process and system measurements.
NVIDIA GPU information is shown only where the relevant system tools and device
are available.

Panel
-----

**Resource Monitor**

Depending on the host machine, the panel may display:

* total and per-process CPU use;
* system and AstronomicAL process memory;
* disk capacity and activity;
* network traffic;
* process information;
* NVIDIA GPU utilisation and memory.

The panel state and layout can be saved with the workspace.

Service and Snapshots
---------------------

A lazy resource-sampler service collects measurements only when required. The
panel can publish :code:`resources.snapshot` so that a point-in-time diagnostic
record can be observed by platform tooling.

Understanding the Values
------------------------

A high value is not always a fault. For example, model training may be expected
to use most CPU or GPU capacity. The monitor is most useful for identifying
unexpected behaviour such as steadily growing memory, no GPU use during a GPU
recipe, or heavy disk access caused by repeated materialisation.

.. note::

   The values describe the machine running the Panel server. In a remote or
   shared deployment this is not the laptop or browser used to view
   AstronomicAL.
