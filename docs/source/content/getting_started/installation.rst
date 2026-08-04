.. _installation:

Installation
============

System Requirements
-------------------

AstronomicAL runs locally and is displayed in a web browser. A normal desktop
computer is sufficient for the example datasets, although large tables, image
collections and machine-learning workflows may require additional memory,
storage or a GPU.

.. todo::

   Confirm the supported Python versions and operating systems before the first
   plugin-system release.

Installing From Source
----------------------

Clone the repository and enter the project directory:

.. code-block:: bash

   git clone https://github.com/grant-m-s/AstronomicAL.git
   cd AstronomicAL

Create an isolated environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate

Install the current requirements:

.. code-block:: bash

   pip install -r requirements-core.txt


Other application-specific requirement docs are also available:

.. code-block:: bash

      pip install -r

with any of the following files:

- requirements-astro.txt
- requirements-dev.txt
- requirements-huggingface.txt
- requirements-ml.txt

or install all at once with

.. code-block:: bash

      pip install -r requirements-all.txt



Running AstronomicAL
--------------------

Start the application with:

.. code-block:: bash

   panel serve astronomicAL --show

Panel will print a local address in the terminal. Open this address manually if the browser does not open automatically.

Verifying the Installation
--------------------------

Once AstronomicAL opens:

1. Check that the application header is visible.
2. Open the **Plugin Manager** panel **(Core > Platform > Plugin Manager)**.
3. Confirm that the bundled plugins have been discovered.
4. Review any of the plugins that **Need Attention** for missing optional dependencies or other errors.

.. Add image: Plugin Manager showing discovered plugins.

Updating
--------

Before updating the repository, save any important layouts and keep a copy of
the datasets and artifacts used by your project.

.. code-block:: bash

   git pull
   pip install -r requirements.txt

Common Problems
---------------

**The application does not open**

Use the URL printed by Panel and check that the selected port is not already in
use.

**A plugin is unavailable**

Open the Plugin Manager and inspect its missing requirements.

**The interface opens with an error**

Start AstronomicAL with the debugging variables described in
:doc:`../troubleshooting/troubleshooting`.
