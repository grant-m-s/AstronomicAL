Adding custom features to your model
====================================================

The features that are given to the model can often be the deciding factor for how well a model is able to produce accurate predictions. This is arguably even more so when approaching the problem using a method such as Active Learning, where you may only being using a tiny fraction of your entire dataset.

Creating Colours
----------------------------------------------------
Given the prevalence of photometry data, the most common additional features to create are colours. In astronomicAL, these are provided with the default `subtract (a-b)` with a combination value of 2.
