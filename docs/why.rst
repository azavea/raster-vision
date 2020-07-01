Why Raster Vision?
==================

Why another deep learning library?
-----------------------------------

Most machine learning libraries implement the core functionality needed to build and train models, but leave the "plumbing" to users to figure out.
This plumbing is the work of implementing a repeatable, configurable workflow that creates training data, trains models, makes predictions, and computes evaluations, and runs locally and in the cloud. Not giving this work the engineering effort it deserves often results in a bunch of hacky, one-off scripts that are not reusable.

In addition, most machine learning libraries cannot work out-of-the-box with massive, geospatial imagery. This is because of the format of the data (eg. GeoTIFF and GeoJSON), the massive size of each scene (eg. 10,000 x 10,000 pixels), the use of map coordinates (eg. latitude and longitude), the use of more than three channels (eg. infrared), patches of missing data (eg. NODATA), and the need to focus on irregularly-shaped AOIs (areas of interest) within larger images.

What are the benefits of using Raster Vision?
----------------------------------------------

* Programmatically configure pipelines in a concise, modifiable, and reusable way, using abstractions such as :ref:`pipelines <rv pipelines>`, :ref:`backends <backend>`, :ref:`datasets <dataset>`, and :ref:`scenes <scene>`.
* Let the framework handle the challenges and idiosyncrasies of doing machine learning on massive, geospatial imagery.
* Run pipelines and individual commands from the command line that execute in parallel, locally or on AWS Batch.
* Read files from HTTP, S3, the local filesystem, or anywhere with the pluggable :ref:`filesystem` architecture.
* Make predictions and build inference pipelines using a single "model bundle" which includes the trained model and associated metadata.
* :ref:`Customize <customizing rv>` Raster Vision using the :ref:`plugins <pipelines plugins>` architecture.

Who is Raster Vision for?
--------------------------

* Developers **new to deep learning** who want to get spun up on applying deep learning to imagery quickly or who want to leverage existing deep learning libraries like PyTorch for their projects simply.
* People who are **already applying deep learning** to problems and want to make their processes more robust, faster and scalable.
* Machine Learning engineers who are **developing new deep learning capabilities** they want to plug into a framework that allows them to focus on the interesting problems.
* **Teams building models collaboratively** that are in need of ways to share model configurations and create repeatable results in a consistent and maintainable way.
