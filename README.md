# Raster Vision

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Join the chat at https://gitter.im/azavea/raster-vision](https://badges.gitter.im/azavea/raster-vision.svg)](https://gitter.im/azavea/raster-vision?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

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

In the past, we developed prototypes for semantic segmentation and tagging in this repo, which were discussed in our [segmentation ](https://www.azavea.com/blog/2017/05/30/deep-learning-on-aerial-imagery/), and [tagging](https://www.azavea.com/blog/2018/01/03/amazon-deep-learning/) blog posts. This implementation has been removed from the `develop` branch and is unsupported, but can still be found at [this tag](https://github.com/azavea/raster-vision/releases/tag/old-semseg-tagging).
Similarly, an outdated prototype of object detection can be found at [this tag](https://github.com/azavea/raster-vision/releases/tag/old-object-detection) under the `rv` module.

### Contributing

We are happy to take contributions! It is best to get in touch with the maintainers
about larger features or design changes *before* starting the work,
as it will make the process of accepting changes smoother.

Everyone who contributes code to Raster Vision will be asked to sign the
Azavea CLA, which is based off of the Apache CLA.

1. Download a copy of the [Raster Vision Individual Contributor License
   Agreement](docs/cla/2018_04_17-Raster-Vision-Open-Source-Contributor-Agreement-Individual.pdf)
   or the [Raster Vision Corporate Contributor License
   Agreement](docs/cla/2018_04_18-Raster-Vision-Open-Source-Contributor-Agreement-Corporate.pdf)

2. Print out the CLAs and sign them, or use PDF software that allows placement of a signature image.

3. Send the CLAs to Azavea by one of:
  - Scanning and emailing the document to cla@azavea.com
  - Faxing a copy to +1-215-925-2600.
  - Mailing a hardcopy to:
    Azavea, 990 Spring Garden Street, 5th Floor, Philadelphia, PA 19107 USA
