# Raster Vision

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Note: this project is under development and may be difficult to use at the moment.

## Object Detection on Aerial and Satellite Imagery

This project provides a set of scripts for training and running object detection models on aerial and satellite imagery. In traditional object detection, each image is a small PNG file and contains a few objects. In contrast, when working with satellite and aerial imagery, each image is a set of very large GeoTIFF files and contains hundreds of objects that are sparsely distributed. In addition, annotations and predictions are represented in geospatial coordinates using GeoJSON files.

The [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) provides good core object detection functionality using deep learning, but cannot natively handle satellite and aerial imagery. To leverage this API for our use case, we use [Rasterio](https://github.com/mapbox/rasterio) to convert data into and out of a form that the API can handle. For prediction, we split images into small chips, make predictions on the chips, and then aggregate the predictions back together. For training, we crop out windows around objects to create training chips. This project also includes code for building Docker containers and AMIs, and running jobs on AWS EC2 using [AWS Batch](https://aws.amazon.com/batch/).

#### Previous work on semantic segmentation and tagging

In the past, we explored semantic segmentation and tagging in this repo. Some of this work is discussed in our [blog post](https://www.azavea.com/blog/2017/05/30/deep-learning-on-aerial-imagery/).
However, this functionality is no longer being maintained, and has been removed from the `develop` branch, but can still be found at [this tag](https://github.com/azavea/raster-vision/releases/tag/old-semseg-tagging). Someday we may add this functionality back in a refactored form.

## Usage

### Requirements

- Vagrant 1.8+
- VirtualBox 4.3+
- Python 2.7
- Ansible 2.1+

### Scripts

| Name     | Description                              |
| -------- | ---------------------------------------- |
| `cipublish`  | Publish docker image to ECR |
| `clean`  | Remove build outputs inside virtual machine |
| `test`   | Run unit tests and lint on source code |
| `run` | Run container locally or remotely |
| `jupyter` | Run container with Juptyer notebook with mounted data and notebook directory from `RASTER_VISION_NOTEBOOK_DIR` |
| `setup`  | Bring up the virtual machine and install dependent software on it |
| `setup_aws_batch`  | Setup AWS Batch |
| `update` | Install dependent software inside virtual machine |

### Initial setup

First, set the `RASTER_VISION_DATA_DIR` environment variable on your host machine. All data including datasets and results should be stored in a single directory outside of the repo. The `Vagrantfile` maps the `RASTER_VISION_DATA_DIR` environment variable on the host machine to `/opt/data` on the guest machine. Within the project root, execute the following commands to setup and then log into the VM.

```bash
$ ./scripts/setup
$ vagrant ssh
```

If you get an error message about the Docker daemon not being started, you
may want to run `vagrant provision`.

## Running locally on CPUs

### Running the Docker container

You can build the Docker container and then get into the Bash console for it as follows.
```shell
vagrant ssh
vagrant@raster-vision:/vagrant$ ./scripts/update --cpu
vagrant@raster-vision:/vagrant$ ./scripts/run --cpu
```

### Running an object detection workflow

See the [object detection README](docs/object-detection.md).

### Running a Jupyter notebook

You can run a Juptyer notebook that has the data from `RASTER_VISION_DATA_DIR` mounted to `/opt/data`
and `RASTER_VISION_NOTEBOOK_DIR` mounted to `/opt/notebooks` and set as the Juptyer notebook directory.

```shell
vagrant ssh
vagrant@raster-vision:/vagrant$ ./scripts/update --jupyter
vagrant@raster-vision:/vagrant$ ./scripts/jupyter
```

## Running remotely using AWS Batch

In order to run scripts on GPUs and in parallel, we use [AWS Batch](https://aws.amazon.com/batch/).

### Publishing the container to ECR

The latest Docker image should be stored in ECR so that it can be used by Batch. To build and publish the container, run `./scripts/cipublish`.

### Submit jobs to AWS Batch

#### Setup Batch

To setup the AWS Batch stack, which should only be done once per AWS account, run `./scripts/setup_aws_batch`.

#### Updating the Batch AMI

Use `deployment/batch_amis.py` to update the Batch environment AMI. This requires your `raster-vision` AWS profile to be configured.

```bash
$ aws --profile raster-vision configure
$ cd deployment
$ pip install -r requirements.txt
$ ./batch_amis.py  build-amis --aws-profile raster-vision
...
==> Builds finished. The artifacts of successful builds are:
--> raster-vision-gpu: AMIs were created:

us-east-1: ami-fb5c7980
```

Use the AMI ID provided above, to update the ComputeResources > imageId field in `deployment/batch/compute_environment_{gpu,cpu}.json`. To apply these changes, delete the existing Batch environments using the AWS Console, and then re-run the steps in the section above.

Prune any old AMIs by using the `prune-amis` command to `batch_amis.py`
```bash
$ ./batch_amis.py  prune-amis --keep 10
```

#### Submitting a job

To submit a job to Batch, use the `batch_submit` script inside the Docker container as follows.

```
src/detection/scripts/batch_submit.py <branch_name> "<command_to_run>" --attempts <# of attempts>
```

The `branch_name` should be the name of the branch with the code to run. If you are testing a job to see if it might fail, you should run it with `--attempts 1` so that it won't be retried if it fails. After submitting a job, AWS Batch will start an EC2 instance, run the command inside a Docker container, and will shut it down when finished.
