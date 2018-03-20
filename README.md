# Raster Vision

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

*Note: this project is under development and may be difficult to use at the moment.*

The overall goal of Raster Vision is to make it easy to train and run deep learning models over aerial and satellite imagery. At the moment, it includes functionality for making training data, training models, making predictions, and evaluating models for the task of object detection implemented via the Tensorflow Object Detection API.  It also supports running experimental workflows using AWS Batch. The library is designed to be easy to extend to new data sources, machine learning tasks, and machine learning implementation.

Our future work includes:
* more documentation and unit tests
* scalable predictions using parallelism
* classification and segmentation

We are shooting for a first release in Summer 2018.

Why do we need yet another deep learning library? In traditional object detection, each image is a small PNG file and contains a few objects. In contrast, when working with satellite and aerial imagery, each image is a set of very large GeoTIFF files and contains hundreds of objects that are sparsely distributed. In addition, annotations and predictions are represented in geospatial coordinates using GeoJSON files.

### Documentation

* [Vagrant / Docker / AWS Setup](docs/setup.md)
* [Object Detection Tutorial](docs/object-detection.md)

### Contact and Support

You can find more information and talk to developers (let us know what you're working on!) at:
* [Gitter](https://gitter.im/azavea/raster-vision)
* [Mailing List](https://groups.google.com/forum/#!forum/raster-vision)

### Previous work on semantic segmentation and tagging

In the past, we developed prototypes for semantic segmentation and tagging in this repo, which were discussed in our [segmentation ](https://www.azavea.com/blog/2017/05/30/deep-learning-on-aerial-imagery/), and [tagging](https://www.azavea.com/blog/2018/01/03/amazon-deep-learning/) blog posts. However, this functionality is no longer being maintained, and has been removed from the `develop` branch, but can still be found at [this tag](https://github.com/azavea/raster-vision/releases/tag/old-semseg-tagging).
