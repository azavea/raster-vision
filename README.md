# keras-semantic-segmentation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction
A key task in computer vision is semantic segmentation, which attempts to simultaneously answer the questions of what is in an image, and where it is located. More formally, the task is to assign to each pixel a meaningful label such as "road" or "building."

This repo contains code for semantic segmentation using convolutional neural networks built on top of the [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) libraries.
There is code for building Docker containers, running experiments on AWS EC2, loading data, training models, and evaluating models on validation and test data.
Here is an example of an aerial image segmented using a model learned by our system.

![Example segmentation](results/unet/img/good1.png)

The following datasets and model architectures are implemented.

### Datasets
* [ISPRS Potsdam 2D dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html) ✈️
* [ISPRS Vaihingen 2D dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) ✈️

### Model architectures
* [FCN](https://arxiv.org/abs/1411.4038) (Fully Convolutional Networks) using [ResNets](https://arxiv.org/abs/1512.03385)
* [U-Net](https://arxiv.org/abs/1505.04597)
* [Fully Convolutional DenseNets](https://arxiv.org/abs/1611.09326) (aka the 100 Layer Tiramisu)

⚠️ 🚧 This project is under construction. Some things are poorly documented, may change drastically without notice, and are tied to the particular experiments that we are running.

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

From within the project root, execute the following commands.

```bash
$ ./scripts/setup
$ vagrant ssh
```

You will be prompted to enter the OpenTreeID AWS credentials, along with a default region. These credentials will be used to authenticate calls to the AWS API when using the AWS CLI and Terraform. Note that if you are not at Azavea, deployment code will need to be manually updated or ignored entirely.

## Running locally on CPUs

### Data directory

All data including datasets and results are stored in a single directory outside of the repo. The `Vagrantfile` maps `~/data` on the host machine to `/opt/data` on the guest machine. The datasets are stored in `/opt/data/datasets` and results are stored in `/opt/data/results`.

### Running the Docker container

You can get into the bash console for the Docker container which has Keras and Tensorflow installed with
```shell
vagrant ssh
vagrant@otid:/vagrant$ ./scripts/update --cpu
vagrant@otid:/vagrant$ ./scripts/run --cpu
```

### Preparing datasets

Before running any experiments locally, the data needs to be prepared so that Keras can consume it. For the
[ISPRS 2D Semantic Labeling Potsdam dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html), you can download the data after filling out the [request form](http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html).
After following the link to the Potsdam dataset, download
`1_DSM_normalisation.zip`, `4_Ortho_RGBIR.zip`, `5_Labels_for_participants.zip`, and `5_Labels_for_participants_no_Boundary.zip`. Then unzip the files into
`/opt/data/datasets/potsdam`, resulting in `/opt/data/datasets/potsdam/1_DSM_normalisation/`, etc.

For the [ISPRS 2D Semantic Labeling Vaihingen dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) dataset, download `ISPRS_semantic_labeling_Vaihingen.zip` and `ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_for_participants.zip`. Then unzip the files into `/opt/data/datasets/vaihingen`, resulting in
`/opt/data/datasets/vaihingen/dsm`, `/opt/data/datasets/vaihingen/gts_for_participants`, etc.

Then run `python -m semseg.data.factory --preprocess`. This takes about 15 minutes and will generate `/opt/data/datasets/processed_potsdam` and `/opt/data/datasets/processed_vaihingen`. As a test, you may want to run `python -m semseg.data.factory --plot` which will generate PDF files that visualize samples produced by the data generator in  `/opt/data/results/gen_samples/`.
 To make the processed data available for use on EC2, upload a zip file of `/opt/data/datasets/processed_potsdam` named `processed_potsdam.zip` (and similar for Vaihingen) to the `otid-data` bucket.

### Running experiments

An experiment consists of training a model on a dataset using a set of hyperparameters. Each experiment is defined using an options `json` file.
An example can be found in [src/experiments/quick_test.json](src/experiments/quick_test.json), and this
can be used as a quick integration test.
In order to run an experiment, you must also provide a list of tasks to perform. These tasks
include `setup_run`, `train_model`, `plot_curves`, `validation_eval`, `test_eval`. More details about these can be found in [src/semseg/run.py](src/semseg/run.py).

Here are some examples of how to use the `run` command.
```shell
# Run all tasks by default
python -m semseg.run experiments/quick_test.json
# Only run the plot_curves tasks which requires that setup_run and train_model were previously run
python -m semseg.run experiments/quick_test.json plot_curves
```
This will generate a directory structure in `/opt/data/results/<run_name>/` which contains the options file, the learned model, and various metrics and visualization files.

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
ssh-add ~/.aws/open-tree-id.pem
ssh ec2-user@<public dns>
```

Then you can train a model with
```shell
cd keras-semantic-segmentation
./scripts/run --gpu
root@230fb62d8ecd:/opt/src# python -m semseg.run experiments/quick_test.json
```

When running on EC2, the results will be saved to the `otid-data` S3 bucket after each epoch and the final evaluation. If a run is terminated for any reason and you would like to resume it,
simply run the above command with the same options file, and it should pick up where it left off.

⚠️️ When finished with all the instances, you should shut them down with
```shell
./scripts/infra destroy
```

### Publishing the Docker container to ECR

To enable fast bootup, we publish the latest Docker image to ECR and then download it when initializing the EC2 instance. To build and publish the container, run `./scripts/cipublish`.
