Welcome to astronomicAL's documentation!
========================================
An Interactive Dashboard for Active Learning in Astronomy.
--------------------------------------------------------------------

.. image:: https://travis-ci.com/grant-m-s/astronomicAL.svg?token=upRGxrMseZqj7kT3bSGx&branch=master
    :target: https://travis-ci.com/grant-m-s/astronomicAL


.. image:: https://codecov.io/gh/grant-m-s/astronomicAL/branch/master/graph/badge.svg?token=TCO9J2AD1Z
    :target: https://codecov.io/gh/grant-m-s/astronomicAL

astronomicAL is an iteractive dashboard to enable Astronomy-based researchers to gain valuable insight into their data by showing them the key information they require to make accurate classifications of each source. Combining the use of the Panel_, Bokeh_, modAL_ and `SciKit Learn`_ packages, astronomicAL enables researchers to take full advantage of the benefits of Active Learning, high accuracy models using just a fraction of the total data, without the requirement of being well versed in the Machine Learning theory or implementations.

.. _Panel: https://panel.holoviz.org/
.. _Bokeh: https://docs.bokeh.org/en/latest/index.html
.. _modAL: https://github.com/modAL-python/modAL
.. _`SciKit Learn`: https://scikit-learn.org/stable/

Statement of Need
*****************
With the influx of billions of sources incoming from future surveys, automated classification methods are becoming critical. To produce accurate models, it is often required to have large amounts of labelled data; however, many datasets only consider a narrow window of the electromagnetic spectrum, resulting in key features of a classification going unseen, leading to inaccurate and unreliable labels. Active Learning, a method that automatically and adaptively selects the most informative datapoints to improve a model's performance, has repeatedly been shown to be a valuable tool to address the constraint of large labelled datasets; however, the issue of unreliable labels still exists. AstronomicAL, an interactive dashboard for training and labelling, has been developed to enable domain experts to take advantage of Active Learning whilst ensuring that they are presented with as complete a picture as possible when deciding on a source's classification, resulting in more accurate and reliable labels whilst requiring substantially less labelled training data.


Installation
------------------
To install astronomicAL and all its dependencies, the user can simply clone the repository and from within the repo folder run :code:`pip install -r requirements.txt`. It is recommended that the user creates a virtual environment using tools such as Virtualenv_ or Conda_, to prevent any conflicting package versions.

.. _Virtualenv: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv
.. _Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Quickstart Instructions
-----------------------
To begin using the software, run :code:`bokeh serve astronomicAL --show` and your browser should automatically open to `localhost:5006/astronomicAL
<localhost:5006/astronomicAL>`_.

You can make use of the provided test fits files to enable you to explore the framework, or if you'd like to dive in with your own fits file simply place your file (or a symlink to the file) in the :code:`data/` directory.

To make the integration of your data with the system as seamless as possible, we put very few requirements on what columns your data must contain. Whenever there is a default column required, the user is able to select a corresponding column from their dataset that matches that requirement. This prevents the tedius task of needing to rename column names. We only require the user to have the following columns:
  - :code:`ID` column - containing unique identifiers for each source
  - :code:`Labels` column - A set of predefined labels which are solely used in scoring metrics in the training process.
  - :code:`RA` and :code:`DEC` columns - The location of your source will enable you to gain access to optical and radio images of your sources.
  - :code:`default_x` and :code:`default_y` columns - These are simply the default axes used for the active learning based plots.

For a thorough tutorial on the training process, see :ref:`Training a Classifier: From Start to Finish <Training a Classifier: From Start to Finish>`.

.. raw:: html

   <hr>

Referencing the Package
-------------------------

Please remember to cite our software and user guide whenever relevant.

See the :ref:`citing <citing>` page for instructions about referencing and citing the astronomicAL software.


.. raw:: html

   <hr>

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: API reference

    content/apireference/active_learning.rst
    content/apireference/dashboard.rst
    content/apireference/extensions.rst
    content/apireference/settings.rst
    content/apireference/utils.rst

.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    content/tutorials/active_learning.rst
    content/tutorials/plots.rst
    content/tutorials/feature_generation.rst

.. toctree::
    :maxdepth: 1
    :caption: Other

    content/other/contributors.rst
    content/other/citing.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. image:: images/CDT-UOB-logo.png
   :width: 38%
   :target: http://www.bristol.ac.uk/cdt/interactive-ai/


.. image:: images/EPSRC+logo.png
   :width: 56%
   :target: https://gtr.ukri.org/projects?ref=studentship-2466020
