ML Core
=======

Overview
--------

:code:`core.ml` provides the common machine-learning layer used by AstronomicAL
workflows. It does more than start a classifier: it manages reusable recipes,
profiles, data splits, resource checks, training progress, checkpoints, model
catalogues, compatible prediction and result artifacts.

What You Do
-----------

A normal ML workflow is:

1. Choose or create a code-backed recipe.
2. Select its feature, target, preprocessing and model settings.
3. Save those settings as a reusable recipe profile if they will be used again.
4. Define the training, validation and test protocol.
5. Run training and monitor its progress or curves.
6. Pause, resume or cancel where the recipe supports it.
7. Inspect the evaluation and saved model artifacts.
8. Use **ML Predictor** to apply a compatible model to another dataset.
9. Register row-keyed predictions as a dataset when they need to be explored in
   plots or joined to another workflow.

.. TODO: Add a workflow image covering Recipe Launcher, Curves and Predictor.

Requirements
------------

Scikit-learn is required. Optional recipes and viewers may use PyTorch,
torchvision, Pillow, Matplotlib, Optuna, joblib, timm or XGBoost. A missing
optional package should affect only the recipes that require it.

Required Mappings
-----------------

* :code:`record_id`

Individual recipes can require additional feature, target, image or domain
mappings.

Panels
------

ML Recipe Launcher
******************

The **ML Recipe Launcher** configures and starts code-backed recipes. The
managed harness owns the dataset protocol, split information, provenance,
progress and standard outputs, while a recipe supplies the model-specific
training behaviour.

Saved profiles let the same recipe configuration be reused by Active Learning
or another dataset without repeatedly entering every option.

ML Predictor
************

The **ML Predictor** selects a saved model, checks that the active dataset is
compatible and runs inference. It can expose predictions as artifacts and, for
row-based results, create a prediction dataset suitable for visualisation or
further table operations.

ML Models and Checkpoints
*************************

The **ML Models and Checkpoints** panel is the catalogue for completed models,
imported manifests and resumable runs. It can inspect storage locations, load a
trusted model manifest and send a paused run back to the launcher.

ML Training Curves
******************

The **ML Training Curves** panel displays the metrics recorded by a run. The
available curves depend on the recipe and can include loss, accuracy, F1 and
regression scores.

Actions and Profiles
--------------------

The main actions include:

* **Run ML Recipe**;
* **Save Recipe Profile**;
* **Load Saved Model Manifest**;
* **Predict With Trained Model**.

Recipes describe executable ML behaviour. Profiles store user-selected values
for a recipe. Keeping these separate means one recipe can be run with several
repeatable configurations.

Artifacts
---------

A managed run may produce:

* :code:`ml.model`;
* :code:`ml.predictions`;
* :code:`ml.training_log`;
* :code:`ml.split_spec`;
* split datasets where requested;
* :code:`ml.evaluation_report`;
* :code:`ml.run` provenance;
* :code:`ml.resume_checkpoint`.

Resource Preflight
------------------

Recipes declare how they access data. Before training, the managed harness can
check source capabilities, data types and estimated memory use. Unsafe full
materialisation may be blocked rather than allowing a large dataset to exhaust
the server.

An explicit unsafe-memory override can permit a known memory risk, but it does
not bypass an incompatible dataset source or unsupported data type.

.. caution::

   A model file alone is not a complete reproducible result. Keep it with its
   recipe, feature order, preprocessing, split specification, class information
   and run provenance.

.. caution::

   Loading an external model manifest can execute or deserialize trusted model
   formats. Only import files from a source you trust and keep external access
   restricted to the configured model roots where possible.
