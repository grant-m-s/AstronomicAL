Using Your Models Outside of AstronomicAL
=============================================

Loading your model
---------------------------

The other tutorials show how you can use AstronomicAL to create accurate and robust classifiers using significantly less training data than normally required. This tutorial will explain the steps that will allow you to use your trained classifiers outside of AstronomicAL.

After every iteration of active learning or whenever you press the :code:`checkpoint` button during training, your model is saved inside :code:`models/`. As the models are created using `Scikit Learn`_ it is a quick and straightforward process to reload the models and use them on any new data.

.. _`SciKit Learn`: https://scikit-learn.org/stable/

To load your model in another python file or jupyter notebook run the following code:

.. code-block:: python
  :linenos:

  from joblib import load
  clf = load('models/your_classifier_model.joblib')

You can then simple run :code:`clf.predict(some_new_data)` to generate new predictions. If you want the probability outputs of the predictions you can run :code:`clf.predict_proba(some_new_data)` instead.

.. note::

	You must use data that contains the same number of features that you trained with your model. This includes any feature combinations that you generated in the settings.

    If you do not use the same amount of features, you will receive the following error:

        **ValueError: X has # features, but Classifier is expecting # features as input.**

Loading a Committee
---------------------------

If you created a committee of classifiers during training, you would need to continue using the committee as a whole to get the same performance. When your committee is saved, rather than just saving as a single model, each individual model is saved and bundled into a directory within :code:`data/`.

By default, the committees in AstronomicAL return the class probabilities averaged across each learner (known as *consensus probabilities*) and then the maximum probability is chosen to assign the prediction. It is recommended to recreate this behaviour outside of AstronomicAL to ensure that the model performs as expected.

.. code-block:: python
    :linenos:

    from joblib import load
    import glob
    import numpy as np

    classifiers = []

    committee_dir = "models/committee_dir"

    classifiers = {} # Dictionary to hold all the individual classifiers in the committee
    for i, filename in enumerate(glob.iglob(f"{committee_dir}/*.joblib")):

        # This is only required if you are using a scalar during training
        if "SCALER" in filename:
            continue

        classifiers[f"{i}"] = load(filename)

    for i, clf in enumerate(classifiers):
        pred_clf = classifiers[clf].predict_proba(some_new_data) # get the probability output for each classifier
        if i == 0:
            predictions_total = pred_clf
        else:
            predictions_total = predictions_total + pred_clf

    predictions_avg = predictions_total / len(classifiers.keys()) # This is the averaged probabilities for each class
    predictions = np.argmax(predictions_avg, axis=1) # This is the final predictions made

Scaling New Data
---------------------------
During the :ref:`settings <settings>` assignment, you had the option to scale your features. If this option was selected, you must scale any new data with the same scaler. Whenever your model is saved, the scaler (if used) is saved along with your updated models. The code below explains how to apply this scaler to your new data.

.. code-block:: python

    scaler = load("models/your_classifier_model-SCALER.joblib")

    new_data_scaled = scaler.transform(some_new_data)
