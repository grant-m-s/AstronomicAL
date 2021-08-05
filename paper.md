---
title: 'AstronomicAL: an interactive dashboard for visualisation, integration and classification of data with Active Learning'
tags:
 - Python
 - astronomy
 - machine learning
 - active learning
 - labelling
 - classification
 - software
authors:
 - name: Grant Stevens
   orcid: 0000-0002-8885-4443
   affiliation: 1
 - name: Sotiria Fotopoulou
   orcid: 0000-0002-9686-254X
   affiliation: 2
 - name: Malcolm N. Bremer
   orcid:
   affiliation: 2
 - name: Oliver Ray
   orcid:
   affiliation: 1

affiliations:
- name: Department of Computer Science, Merchant Venturers Building, University of Bristol, Woodland Road, Bristol, BS8 1UB
  index: 1
- name: School of Physics, HH Wills Physics Laboratory, University of Bristol, Tyndall Avenue, Bristol, BS8 1TL
  index: 2
date: 5 August 2021
bibliography: paper/paper.bib

---

# Summary

AstronomicAL is a human-in-the-loop interactive labelling and training dashboard that allows users to create reliable datasets and robust classifiers using active learning. This technique prioritises data that offer high information gain, leading to improved performance using substantially less data. The system allows users to visualise and integrate data from different sources and deal with incorrect or missing labels and imbalanced class sizes. AstronomicAL enables experts to visualise domain-specific plots and key information relating both to broader context and details of a point of interest drawn from a variety of data sources, ensuring reliable labels. In addition, AstronomicAL provides functionality to explore all aspects of the training process, including custom models and query strategies. This makes the software a tool for experimenting with both domain-specific classifications and more general-purpose machine learning strategies. We illustrate using the system with an astronomical dataset due to the field’s immediate need; however, AstronomicAL has been designed for datasets from any discipline. Finally, by exporting a simple configuration file, entire layouts, models, and assigned labels can be shared with the community. This allows for complete transparency and ensures that the process of reproducing results is effortless.

![AstronomicAL workflow from the perspective of an astronomy user. Each part of the workflow improves the reliability of labels, leading to enhanced performance of any classifiers produced. AstronomicAL allows the user to tailor this workflow specifically to their domain with its modular and extensible design.  \label{fig:workflow}](paper/astronomicAL_diagram_portrait.png){ width=86% }

# Statement of Need

Active learning [@settles2012active] removes the requirement for large amounts of labelled training data whilst still producing high accuracy models. This is extremely important as with ever-growing datasets; it is becoming impossible to manually inspect and verify ground truth used to train machine learning systems. The reliability of the training data limits the performance of any supervised learning model, so consistent classifications become more problematic as data sizes increase. The problem is exacerbated when a dataset does not contain any labelled data, preventing supervised learning techniques entirely. AstronomicAL has been developed to tackle these issues head-on and provide a solution for any large scientific dataset.

It is common for active learning to query areas of high uncertainty; these are often in the boundaries between classes where the expert’s knowledge is required. To facilitate this human-in-the-loop process, AstronomicAL provides users with the functionality to fully explore each data point chosen. This allows them to inject their domain expertise directly into the training process, ensuring that assigned labels are both accurate and reliable.

AstronomicAL has been extensively validated on astronomy datasets. These are highly representative of the issues that we anticipate will be found in other domains for which the tool is designed to be easily customisable. Such issues include the volume of data (millions of sources per survey), vastly imbalanced classes and ambiguous class definitions leading to inconsistent labelling. AstronomicAL has been developed to be sufficiently general for any tabular data and can be customised for any domain. For example, we provide the functionality for data fusion of catalogued data and online cutout services for astronomical datasets.

Using its modular and extensible design, researchers can quickly adapt AstronomicAL for their research to allow for domain-specific plots, novel query strategies, and improved models. Furthermore, there is no requirement to be well-versed in the underlying libraries that the software uses. This is due to large parts of the complexity being abstracted whilst allowing more experienced users to access full customisability.

As the software runs entirely locally on the user’s system, AstronomicAL provides a private space to experiment whilst providing a public mechanism to share results. By sharing only the configuration file, users remain in charge of distributing their potentially sensitive data, enabling collaboration whilst respecting privacy.

![The dashboard is set up in 3 main areas. A) Active learning and labelling panel which controls the machine learning functionality. Users can train separate models for each class and see how each performs as additional points are added to the model. B) User-defined information shown for the currently active data point, such as specific columns, local images and online cutout services. C) Customisable domain-specific plots that can render millions of data points. \label{fig:full_layout}](paper/full_layout_browser_box.png)

# Active Learning and Classification

Active learning utilises methods for predicting which part of the search space of unseen examples would be most likely to improve classification accuracy. By querying these key examples and presenting them directly to the user for inspection and labelling, active learning removes the reliance on non-verified data. In the optimal case, it provides users with simple classifications that have a high impact on model performance, leading to a dramatic reduction in the number of data points required for training a classifier.

By making use of the modAL [@modAL2018] active learning framework, AstronomicAL allows users to take full advantage of active learning techniques, leading to improved classifiers. Scikit-Learn [@scikit-learn] models are fully supported in AstronomicAL, and the software will be extended to support Pytorch [@pytorch] and Tensorflow [@tensorflow2015-whitepaper] in the future.

Astronomy datasets are frequently the result of the fusion of multiple heterogeneous data sources. Often in machine learning, only information that is available across all data points is usable during training. However, in astronomy, where *any* available information could be critical in determining an object’s true class, it must be used whenever this information is available. With the flexibility to choose which information is essential for classification, AstronomicAL ensures that users are presented with this critical information whenever available - directly leading to improved confidence and justifiability for any assigned labels. Although this extra information remains unseen by the model, by using AstronomicAL, we inject this information into the training process by labelling the data based on all the available information rather than just the features the model sees.

In \autoref{fig:full_layout} we show an example of a classification task using an extract of the astronomical dataset used in @Fotopoulou2018CPz. \autoref{fig:full_layout} (A) shows the benefits of active learning where with only 30 data points given as training data to the model, we can classify Stars with 97.5% accuracy. \autoref{fig:full_layout} (B) shows the extra information chosen by the user to improve confidence in the assigned labels. This information can be retrieved from web services or local files. In this example, we are showing the SDSS^[http://skyserver.sdss.org/dr16/en/help/docs/api.aspx#imgcutout] [@Ahumada2020DR16] and FIRST^[https://third.ucllnl.org/cgi-bin/firstcutout] [@Becker1995Radio] cutout services. \autoref{fig:full_layout} (C) shows how domain-specific plots can be created and visualised to gain extra insight into the data, aiding the researcher in their ability and confidence in labelling data points.


![AstronomicAL allows users to view all key information during training. Top Left: Training set showing points that have been trained on, current queried point and correct and incorrect predictions. Top Right: Visualisation of the model’s confidence of each data point. Bottom Left: Validation set showing correct and incorrect predictions of the model. Bottom Right: The performance scores of the model as new points are added to the training set. \label{fig:active_learning}](paper/active_learning_1.png){ width=65% }

# Interactivity

Using the Panel dashboard library [@philipp_rudiger_panel_2021_4573728], paired with Holoviews [@philipp_rudiger_holoviews_2021_4581995], Bokeh [@Bokeh2018] and Datashader [@james_a_bednar_datashader_2020_3987379] visualisation libraries, AstronomicAL provides users with interactive plots, enabling zooming and panning of millions of data points in real-time whilst also giving the ability to rearrange and resize plots dynamically to optimise screen layout.

Plots are also interconnected, with any updates projected to all panels simultaneously, allowing analysis of how the properties of a particular point fit in the context of the whole dataset. This enables researchers to gain insight and identify trends - crucial for reliable labelling.

![AstronomicAL enables the creation of curated test sets. The sample is selected according to user-defined criteria and verified visually. This test set can then be used during the active learning training process to ensure that the model will be sufficiently generalisable to future data. \label{fig:labelling_mode}](paper/labelling_mode.png){ width=85% }

The interactive labelling functionality is not limited to only the training stage; as shown in \autoref{fig:labelling_mode}, users can curate a labelled test set to sufficiently demonstrate the validity and generalisability of their model. Furthermore, to facilitate and encourage only assigning labels that the user trusts, AstronomicAL allows users to mark any example as *unsure*. Such examples are excluded from the training set, ensuring that all training data are of high quality.

In summary, all design decisions have been in response to user feedback and were explicitly tailored to improve the visualisation of data, irrespective of the data’s specific nature, to ensure that AstronomicAL is a tool for any discipline.

# Acknowledgements

Grant Stevens acknowledges financial support from the UKRI for a Centre for Doctoral Training studentship in Interactive Artificial Intelligence at the University of Bristol. (EP/S022937/1)

# References
