# Raster Vision

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction

This goal of this project is to implement a variety of deep learning architectures for analyzing aerial and satellite imagery. At the moment, we have implemented approaches for semantic segmentation, and are about to begin work on tagging/recognition. In the future we may add support for object detection. There is code for building Docker containers, running experiments on AWS EC2, loading and processing data, and constructing, training and evaluating models
using the [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) libraries.

⚠️ 🚧 This project is under construction. Some things are poorly documented, may change drastically without notice, and are tied to the particular experiments that we are running.

### Semantic Segmentation
The goal of semantic segmentation is to infer a meaningful label such as "road" or "building for each pixel in an image. Here is an example of an aerial image segmented using a model learned by our system.

![Example segmentation](results/unet/img/good1.png)

The following datasets and model architectures are implemented.

#### Datasets
* [ISPRS Potsdam 2D dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)
* [ISPRS Vaihingen 2D dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html)

#### Model architectures
* [FCN](https://arxiv.org/abs/1411.4038) (Fully Convolutional Networks) using [ResNets](https://arxiv.org/abs/1512.03385)
* [U-Net](https://arxiv.org/abs/1505.04597)
* [Fully Convolutional DenseNets](https://arxiv.org/abs/1611.09326) (aka the 100 Layer Tiramisu)

### Tagging / Recognition

The goal of tagging is to infer a set of labels for each image. The following datasets and model architectures are implemented.

#### Datasets
* [Planet: Understanding the Amazon from Space (Kaggle Competition)](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

#### Model architectures
* [ResNet](https://arxiv.org/abs/1512.03385)

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
| `infra`  | Execute Terraform subcommands            |
| `test`   | Run unit tests and lint on source code |
| `run` | Run container locally or remotely |
| `setup`  | Bring up the virtual machine and install dependent software on it |
| `update` | Install dependent software inside virtual machine |
| `upload_code` | Upload code to EC2 instance for rapid debugging |

### Initial setup

First, set the `RASTER_VISION_DATA_DIR` environment variable on your host machine. If you are not at Azavea and want to use the deployment code, you will need to set the `RASTER_VISION_S3_BUCKET` environment variable and update various values in
[deployment/terraform/variables.tf](deployment/terraform/variables.tf).

From within the project root, execute the following commands.

```bash
$ ./scripts/setup
$ vagrant ssh
```

You will be prompted to enter credentials for AWS, along with a default region. These credentials will be used to authenticate calls to the AWS API when using the AWS CLI and Terraform.

## Running locally on CPUs

### Data directory

All data including datasets and results are stored in a single directory outside of the repo. The `Vagrantfile` maps the `RASTER_VISION_DATA_DIR` environment variable on the host machine to `/opt/data` on the guest machine. The datasets are stored in `/opt/data/datasets` and results are stored in `/opt/data/results`.

### Running the Docker container

You can get into the bash console for the Docker container which has Keras and Tensorflow installed with
```shell
vagrant ssh
vagrant@raster-vision:/vagrant$ ./scripts/update --cpu
vagrant@raster-vision:/vagrant$ ./scripts/run --cpu
```

### Preparing datasets
These are instructions for replicating our setup from scratch. If you are at Azavea, the prepared data is already on S3 and you can download it from there onto your local machine.

#### ISPRS
Before running any experiments locally, the data needs to be prepared so that Keras can consume it. For the
[ISPRS 2D Semantic Labeling Potsdam dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html), you can download the data after filling out the [request form](http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html).
After following the link to the Potsdam dataset, download
`1_DSM_normalisation.zip`, `4_Ortho_RGBIR.zip`, `5_Labels_for_participants.zip`, and `5_Labels_for_participants_no_Boundary.zip`. Then unzip the files into
`<RASTER_VISION_DATA_DIR>/datasets/isprs/potsdam`, resulting in `<RASTER_VISION_DATA_DIR>/datasets/isprs/potsdam/1_DSM_normalisation/`, etc.

For the [ISPRS 2D Semantic Labeling Vaihingen dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) dataset, download `ISPRS_semantic_labeling_Vaihingen.zip` and `ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_for_participants.zip`. Then unzip the files into `<RASTER_VISION_DATA_DIR>/datasets/isprs/vaihingen`, resulting in
`<RASTER_VISION_DATA_DIR>/datasets/isprs/vaihingen/dsm`, `<RASTER_VISION_DATA_DIR>/datasets/isprs/vaihingen/gts_for_participants`, etc.

Then run `python -m rastervision.semseg.data.factory all all all`. This takes about 30 minutes and will generate `<RASTER_VISION_DATA_DIR>/datasets/isprs/processed_potsdam` and `<RASTER_VISION_DATA_DIR>/datasets/isprs/processed_vaihingen`, as well as PDF files that visualize samples produced by the data generator in  `<RASTER_VISION_DATA_DIR>/results/gen_samples/`. You can also run the command for a specific task, dataset, and generator. For instance, you can run `python -m rastervision.semseg.data.factory isprs/potsdam numpy plot`.
 To make the processed data available for use on EC2, upload a zip file of `<RASTER_VISION_DATA_DIR>/datasets/isprs/processed_potsdam` named `processed_potsdam.zip` (and similar for Vaihingen) to `s3://<RASTER_VISION_S3_BUCKET>/datasets/isprs/`.

#### Planet Kaggle
The data is available from the Kaggle [website](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data) and requires logging in. For running locally, you will need to create a directory `<RASTER_VISION_DATA_DIR>/datasets/planet_kaggle`, and place the  `train-tif-v2/` and `test-tif-v2/` directories in it, as well as the `train_v2.csv` file. To save disk space, you can just keep a small sample of the data in these directories for testing purposes. You will then need to convert the corresponding 7zip files into zip files and upload them to `s3://<RASTER_VISION_S3_BUCKET>/datasets/planet_kaggle/`. To test that the generator works for this dataset, you can run `python -m rastervision.semseg.taging.factory planet_kaggle tiff plot`, which will generate debug plots in `<RASTER_VISION_DATA_DIR>/results/gen_samples/`.

### Running experiments

An experiment consists of training a model on a dataset using a particular set of hyperparameters. Each experiment is defined using an options `json` file,
which contains fields such as `problem_type` (`semseg` or `tagging`), `dataset_name`, `model_type`, and `run_name` (which is used as an ID for the experiment, and should be unique).
An example can be found in [potsdam_quick_test.json](src/experiments/tests/semseg/potsdam_quick_test.json), and this
can be used as a quick integration test.
In order to run an experiment, you invoke `python -m rastervision.run <experiment.json> <list of tasks>`. The list of available tasks varies based on the `problem_type`. The format of experiment files and available tasks is evolving rapidly, so you should look at the source code and existing experiment files for guidance.

Here are some examples of how to use the `run` command.
```shell
# Run all tasks by default
python -m rastervision.run experiments/tests/semseg/potsdam_quick_test.json
# Only run the plot_curves tasks which requires that train_model were previously run
python -m rastervision.run experiments/tests/semseg/potsdam_quick_test.json plot_curves
```
This will generate a directory structure in `<RASTER_VISION_DATA_DIR>/results/<run_name>/` which contains the options file, the learned model, and various metrics and visualization files.

## Running remotely on AWS EC2 GPUs

Support for Amazon EC2 GPU instances is provided through a combination of the AWS CLI and Terraform. The AWS CLI is used to produce a local AWS profile with valid credentials, and Terraform is used to bring up the appropriate EC2 instances using spot pricing.

### Spot Fleet

Once an AWS profile exists, use the `infra` script to interact with Amazon EC2 Spot Fleet API, which will request the lowest priced P2 GPU instance across all availability zones. The following command will start 2 instances and print their
public DNS names.
```bash
vagrant@vagrant-ubuntu-trusty-64:/vagrant$ ./scripts/infra start 2
```

Afterwards, navigate to the [spot request](https://console.aws.amazon.com/ec2sp/v1/spot/home?region=us-east-1#) section of the EC2 console to monitor the request's progress. Once fulfilled, a running instance will visible in the [instances](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#Instances:sort=instanceId) section.

After initializing, the instance should have `nvidia-docker`, the GPU-enabled Docker image for this repo, and this repo + datasets in the home directory.

### Running Models on GPUs

After starting an instance, `ssh` into it with
```shell
ssh-add ~/.aws/raster-vision.pem
ssh ec2-user@<public dns>
```

Then you can train a model with
```shell
cd raster-vision
./scripts/run --gpu
root@230fb62d8ecd:/opt/src# python -m rastervision.run experiments/tests/semseg/potsdam_quick_test.json
```

When running on EC2, the results will be saved to the S3 bucket after each epoch and the final evaluation. If a run is terminated for any reason and you would like to resume it,
simply run the above command with the same options file, and it should pick up where it left off.

⚠️️ When finished with all the instances, you should shut them down with
```shell
./scripts/infra destroy
```

### Publishing the Docker container to ECR

To enable fast bootup, we publish the latest Docker image to ECR and then download it when initializing the EC2 instance. To build and publish the container, run `./scripts/cipublish`.
