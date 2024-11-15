![Raster Vision Logo](docs/img/raster-vision-logo.png)
&nbsp;

[![Pypi](https://img.shields.io/pypi/v/rastervision.svg)](https://pypi.org/project/rastervision/)
[![Documentation Status](https://readthedocs.org/projects/raster-vision/badge/?version=latest)](https://docs.rastervision.io/en/stable/?badge=stable)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://github.com/azavea/raster-vision/actions/workflows/release.yml/badge.svg)](https://github.com/azavea/raster-vision/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/azavea/raster-vision/branch/master/graph/badge.svg)](https://codecov.io/gh/azavea/raster-vision)

Raster Vision is an open source Python **library** and **framework** for building computer vision models on satellite, aerial, and other large imagery sets (including oblique drone imagery).

It has built-in support for chip classification, object detection, and semantic segmentation with backends using PyTorch.

<div align="center">
    <img src="docs/img/cv-tasks.png" alt="Examples of chip classification, object detection and semantic segmentation" width="60%">
</div>

**As a library**, Raster Vision provides a full suite of utilities for dealing with all aspects of a geospatial deep learning workflow: reading geo-referenced data, training models, making predictions, and writing out predictions in geo-referenced formats.

**As a low-code framework**, Raster Vision allows users (who don't need to be experts in deep learning!) to quickly and repeatably configure experiments that execute a machine learning pipeline including: analyzing training data, creating training chips, training models, creating predictions, evaluating models, and bundling the model files and configuration for easy deployment.

For more details, see the [documentation](https://docs.rastervision.io/en/stable/).

---

### Prerequisites
- Python 3.6+
- Docker (optional but recommended for running pre-built images)

For cloud experiments, ensure AWS CLI and AWS Batch or AWS Sagemaker are configured.

---

### Installation Steps

#### Option 1: Install via `pip`
You can install Raster Vision directly via pip:
```sh
pip install rastervision
```

#### Option 2: Use Pre-built Docker Image
Alternatively, use a pre-built Docker image. Docker images are published to [quay.io](https://quay.io/repository/azavea/raster-vision).

To pull the latest version:
```sh
docker pull quay.io/azavea/raster-vision:pytorch-latest
```

#### Option 3: Build Docker Image
You can also build a Docker image from scratch. Clone the repo and run the following commands:
```sh
docker/build
docker/run
```

---

### Usage Examples and Tutorials

#### Non-developers
Refer to the [*Quickstart guide*](https://docs.rastervision.io/en/stable/framework/quickstart.html) for a simple entry point into using Raster Vision.

#### Developers
Developers can start with [*Usage Overview*](https://docs.rastervision.io/en/stable/usage/overview.html), followed by [*Basic Concepts*](https://docs.rastervision.io/en/stable/usage/basics.html) and [*Tutorials*](https://docs.rastervision.io/en/stable/usage/tutorials/index.html).

---

### Contributing

We are happy to take contributions! It is best to get in touch with the maintainers about larger features or design changes *before* starting the work. See [Contributing](https://docs.rastervision.io/en/stable/CONTRIBUTING.html) for instructions.

---

### Licenses

Raster Vision is licensed under the Apache 2 license. See license [here](./LICENSE).

3rd party licenses for all dependencies used by Raster Vision can be found [here](./THIRD_PARTY_LICENSES.txt).

---

