Frequently Asked Questions
==========================

Can I Use Non-Astronomical Data?
--------------------------------

Yes. Generic dataset, visualisation, selection, annotation, image and
machine-learning plugins are not tied to astronomy.

Do I Need Labels?
-----------------

No. Exploration and annotation can begin without labels. Supervised training
requires suitable target information.

Can I Use Data Larger Than Memory?
----------------------------------

Yes, when the dataset source and selected plugins support lazy or streamed
access.

What is the Difference Between Focus and Selection?
---------------------------------------------------

Focus is one row. A selection set is a group of rows.

What is the Difference Between a Dataset and an Artifact?
---------------------------------------------------------

A dataset is a working input. An artifact is a generated result. A tabular
artifact can be promoted to a dataset.

Are Plugins Trusted?
--------------------

Enabled plugins execute Python code with the application's permissions. Install
only plugins you trust.

Where are Layouts Stored?
-------------------------

Layouts are currently saved in astronomicAL/layouts

Can I Use a Model Outside AstronomicAL?
---------------------------------------

Yes, when the model format and recipe support export. Keep the model manifest
and preprocessing requirements with the saved model.