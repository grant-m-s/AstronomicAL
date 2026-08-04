Artifact Type Catalogue
=======================

Core and Workflow Types
-----------------------

.. list-table::
   :header-rows: 1

   * - Type
     - Producer
     - Purpose
   * - :code:`annotation.note`
     - Annotations
     - Record-level note
   * - :code:`review.status`
     - Annotations
     - Review decision and confidence
   * - :code:`image.preview`
     - Image Assets
     - Focused image preview
   * - :code:`ml.model`
     - ML Core
     - Trained model and manifest
   * - :code:`ml.predictions`
     - ML Core
     - Predictions, probabilities and uncertainty
   * - :code:`ml.training_log`
     - ML Core
     - Training progress and metrics
   * - :code:`al.session`
     - Active Learning
     - Session contract and workflow state
   * - :code:`al.labels`
     - Active Learning
     - Verified labels
   * - :code:`ml.active_learning_batch`
     - Active Learning
     - Ranked review batch

Astronomy Types
---------------

.. list-table::
   :header-rows: 1

   * - Type
     - Producer
     - Purpose
   * - :code:`astro.cutout.euclid`
     - Euclid Cutout
     - Retrieved cutout product
   * - :code:`astro.spectra`
     - Astronomy Spectra
     - Retrieved spectrum
   * - :code:`astro.sed.broadband`
     - Broadband SED
     - Broadband SED table and plot data

.. todo::

   Generate the complete table from bundled plugin specifications.
