.. _preparing-data:
Preparing Your Data For Use in AstronomicAL
============================================

AstronomicAL has been designed so that researchers from any discipline can take advantage of the benefits of active learning with any tabular dataset. For this reason, we have placed very few requirements on the pre-processing required for your dataset.

Dataset Columns
**************************************

AstronomicAL only requires two columns to be included in your dataset:
  - **ID Column**: This column must contain unique identifiers for each data point.
  - **Label Column**: This column must contain integer labels for each data point.

.. note::
    We put no naming requirements for these columns (or any of the other columns in your dataset) as we ask you to select these corresponding columns in the settings panel.

*Which columns should I include?*
##########################################

As well as the two columns above, you should include:
    - All columns that you want to input as features to your classifiers.
    - Any columns that you require for axes in domain-specific plots.
    - Any columns that feature key information that would improve your ability to classify data points.

.. note::

	Active learning removes the requirement for large amounts of labelled data; this opens up possibilities for working with unlabelled datasets that are much larger. Although AstronomicAL can render millions of points, such large datasets can be demanding for memory consumption. **It is recommended that any columns not explicitly used should not be included in your dataset**.

Dataset Labels
**************************************
Each data point within your **Label Column** should have an integer assigned that corresponds to a particular class label.

In our :code:`example_dataset.fits` file, we use the following labels:
    - :code:`0` is a Star
    - :code:`1` is a Galaxy
    - :code:`2` is a QSO

AstronomicAL interprets the label :code:`-1` as an **unknown label**, and so you should only assign this value to data points that you do not yet know the correct label for (see below for more details).

For the known labels, it does not matter which value you assign to which class, as long as you're consistent across your dataset.

.. note::
    To interpret these labels easier, we allow you to assign name aliases for these values in the settings. These names will then be shown to you in plots and any labelling buttons.

Unknown Labels
-----------------
Unknown labels are handled differently from the other labels in the following ways:

    - None of the unknown labels are passed to the performance metrics as the models would interpret :code:`-1` as just another label leading to incorrect classifications due to your model never predicting :code:`-1`.
    - As AstronomicAL allows you to relabel any queried point, unknown labels are only valuable when used in the training pool. Therefore all unlabelled points are always assigned to the training set.

*What if I don't have any labels?*
#####################

One of the main benefits of active learning is that you don't need large amounts of labelled data to produce high accuracy classifiers. Due to AstronomicAL's labelling framework, all that's required is one instance of every label in your classification task for you to start the rest of your labelling in either :ref:`active learning mode <active-learning>` or :ref:`labelling mode <labelling>`.
