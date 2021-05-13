# astronomicAL
[![Build Status](https://travis-ci.com/grant-m-s/astronomicAL.svg?token=upRGxrMseZqj7kT3bSGx&branch=master)](https://travis-ci.com/grant-m-s/astronomicAL) [![codecov](https://codecov.io/gh/grant-m-s/astronomicAL/branch/master/graph/badge.svg?token=TCO9J2AD1Z)](https://codecov.io/gh/grant-m-s/astronomicAL)

# AstronomicAL
## An Interactive Dashboard for Active Learning in Astronomy.

AstronomicAL is an iteractive dashboard to enable Astronomy-based researchers to gain valuable insight into their data by showing them the key information they require to make accurate classifications of each source. Combining the use of the [Panel](https://panel.holoviz.org/), [Bokeh](https://docs.bokeh.org/en/latest/index.html), [modAL](https://github.com/modAL-python/modAL) and [SciKit Learn](https://scikit-learn.org/stable/) packages, astronomicAL enables researchers to take full advantage of the benefits of Active Learning, high accuracy models using just a fraction of the total data, without the requirement of being well versed in the Machine Learning theory or implementations.

### Statement of Need

With the influx of billions of sources incoming from future surveys, automated classification methods are becoming critical. To produce accurate models, it is often required to have large amounts of labelled data; however, many datasets only consider a narrow window of the electromagnetic spectrum, resulting in key features of a classification going unseen, leading to inaccurate and unreliable labels. Active Learning, a method that automatically and adaptively selects the most informative datapoints to improve a model's performance, has repeatedly been shown to be a valuable tool to address the constraint of large labelled datasets; however, the issue of unreliable labels still exists. AstronomicAL, an interactive dashboard for training and labelling, has been developed to enable domain experts to take advantage of Active Learning whilst ensuring that they are presented with as complete a picture as possible when deciding on a source's classification, resulting in more accurate and reliable labels whilst requiring substantially less labelled training data.

### Documentation
The documentation for AstronomicAL can be found [here](https://astronomical.readthedocs.io).

### Installation

To install astronomicAL and all its dependencies, the user can simply clone the repository and from within the repo folder run `pip install -r requirements.txt`. It is recommended that the user creates a virtual environment using tools such as [Virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv) or [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), to prevent any conflicting package versions.

### Quickstart Instructions

To begin using the software, run `bokeh serve astronomicAL --show` and your browser should automatically open to [localhost:5006/astronomicAL](localhost:5006/astronomicAL>`)

AstronomicAL provides both an example dataset and an example configuration file to allow you to jump right into the software and give it a test run.

![Load Configuration](docs/source/images/Load_config_AL.gif)

To begin training you simply have to select **Load Custom Configuration** checkbox and select your config file. Here we have chosen to use the `example_config.json` file.

The **Load Config Select** option allows use to choose the extent to which to reload the configuration.

### Referencing the Package
-------------------------

Please remember to cite our software and user guide whenever relevant.

See the [Citing page](https://astronomical.readthedocs.io/en/latest/content/other/citing.html) in the documentation for instructions about referencing and citing the astronomicAL software.
