Active Learning
===============

Overview
--------

:code:`core.active_learning` manages a complete active-learning workflow rather
than only selecting the next uncertain point. It keeps the session definition,
labels, training membership, query batches, model references and round history
together so that the process can be paused, inspected and continued.

What You Do
-----------

A typical workflow is:

1. Create a session for an active dataset and choose the label information to
   use.
2. Define the training pool and keep fixed validation and test data.
3. Add a small initial set of labels, either manually or from an existing label
   column.
4. Train a model through the ML Core plugin.
5. Choose a query strategy and request the next batch of useful records.
6. Review those records, assign labels and retrain the model.
7. Repeat the query, review and train cycle until the result is suitable.

The panel also lets you inspect how many records have been reviewed, which rows
are in training, which query batch is waiting, and how model performance changes
between rounds.

.. TODO: Add an image of the Active Learning panel and its main tabs.

Required Mappings
-----------------

* :code:`record_id`

A label column is normally selected when the session is created. ML Core is
optional for creating sessions and recording labels, but is required for the
managed training and prediction workflow.

The Active Learning Panel
-------------------------

The **Active Learning** panel brings the whole session into one place. Its
controls cover four closely related tasks.

Reviewing Records
*****************

The review area displays the current query rows and lets you record a label,
verification state or an unsure decision. Existing labels can also be copied in
batches to initialise a session or accelerate review of a known dataset.

Training a Round
****************

Training is delegated to :doc:`ML Core <core_ml>`. A saved ML recipe profile can
be reused so that preprocessing, features and model settings remain consistent
between rounds.

Starting a new training run invalidates any queued query batch produced by the
previous model. Those scores no longer describe the newly trained model.

Creating a Query Batch
**********************

After prediction, choose a query strategy and the number of records to review.
The plugin scores candidate rows and retains the highest-ranked results. It can
also place the batch into the active selection so that Record Browser, Image,
Spectra and other linked panels immediately follow the same records.

Inspecting Progress
*******************

Session summaries show the current round, labelled totals, reviewed and unsure
records, queued rows, and the model and prediction artifacts used for the
latest query. Performance and diagnostic views help compare rounds and inspect
where labelled, trained and queried records lie in the data.

Main Actions
------------

The panel uses the same registered actions that are available to workflows and
other plugins:

* **Start Active-Learning Session** creates the durable session and data
  protocol.
* **Inspect Active-Learning Data Contract** checks how the session can be passed
  to ML Core.
* **Create Active-Learning Query Batch** ranks candidates for review.
* **Calculate Query-Strategy Scores** produces a durable score table for the
  complete pool.
* **Record Active-Learning Label** saves a label or verification decision.
* **Bulk Label Next Review Rows** copies labels from a source column.
* **Prepare Active-Learning Training Rows** materialises the current training
  membership and verified-label overlay.
* **Train Active-Learning Round Through ML Core** launches managed training and
  can optionally run prediction afterwards.

Artifacts and Outputs
---------------------

Important artifact types include:

* :code:`al.session` -- the session definition and current round;
* :code:`al.labels` -- labels and review decisions;
* :code:`al.memberships` -- pool, training, validation and test membership;
* :code:`al.training_set` -- a prepared training view;
* :code:`ml.active_learning_batch` -- the current query batch;
* :code:`ml.active_learning_scores` -- query scores across the pool;
* model, prediction and metric artifacts produced through ML Core.

Performance
-----------

Prediction rows and strategy scores are streamed where possible. Query batches
keep only the highest-ranked candidates rather than loading every score into
the interface. Long-running operations use platform jobs and can make use of
cooperative pause or cancellation controls.

.. caution::

   Keep validation and test rules fixed for the lifetime of a session. Changing
   them between rounds makes performance scores difficult to compare and can
   lead to misleading conclusions.
