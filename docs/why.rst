Why Raster Vision?
==================

Why do we need yet another deep learning library?
-------------------------------------------------

Machine learning libraries generally implement the algorithms and other core functionality needed to build models. The workflow of creating training data in a format that the machine learning library understands, running training in a highly configurable way, making predictions on validation images and performing evaluations on models is usually up to the user to figure out. This often results in a bunch of one-off scripts that are assembled per project, and not engineered to be reusable. Raster Vision is a framework that allows you to state configurations in modifiable and reusable ways, and keeps track of all the files through each step of the machine learning model building workflow. This means you can focus on running experiments to see which machine learning techniques apply best to your problems, and leave the data munging and repeatable workflow processes to Raster Vision.

In addition, the current libraries in the deep learning ecosystem don't usually work well out of the box with large imagery sets, and especially not geospatial imagery (e.g. satellite, aerial, and drone imagery). For example, in traditional object detection, each image is a small PNG file and contains a few objects. In contrast, when working with satellite and aerial imagery, each image is a set of very large GeoTIFF files and contains hundreds of objects that are sparsely distributed. In addition, annotations and predictions are represented in geospatial coordinates using GeoJSON files.



What are the benefits of Raster Vision?
---------------------------------------

* Configure :ref:`task`, :ref:`backend`, and other components of deep learning :ref:`experiment` using a flexible and readable pattern that sets up all the information needed to run a machine learning workflow.
* Run :ref:`commands` from the command line that execute locally or on AWS Batch. With AWS Batch, you can fire off jobs that run through the entire workflow on a GPU spot instance that is created for the workload and terminates immediately afterwards, saving not only money in EC2 instance hours, but also time usually spent ssh'ing into machines or babysitting processes.
* Read files from HTTP, S3, the local filesystem, or anywhere with the pluggable :ref:`filesystem` architecture.
* Make predictions and build inference pipelines using a single file as output of the Raster Vision workflow for any experiment, which includes the trained model and configuration.

Who is Raster Vision for?
-------------------------

Raster Vision is for:

* Developers **new to deep learning** who want to get spun up on applying deep learning to imagery quickly or who want to leverage existing deep learning libraries like Tensorflow and Keras for their projects simply.
* People who are **already applying deep learning** to problems and want to make their processes more robust, faster and scalable.
* Machine Learning engineers who are **developing new deep learning capabilities** they want to plug into a framework that allows them to focus on the hard problems.
* **Teams building models collaboratively** that are in need of ways to share model configurations and create repeatable results in a consistent and maintainable way.
