# Raster Vision Setup

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

See the [object detection tutorial](object-detection.md).

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

To manually submit a job to Batch, use the `batch_submit` script inside the Docker container as follows.

```
python -m rv2.utils.batch <branch_name> "<command_to_run>" --attempts <# of attempts>
```

The `branch_name` should be the name of the Git branch with the code to run. If you are testing a job to see if it might fail, you should run it with `--attempts 1` so that it won't be retried if it fails. After submitting a job, AWS Batch will start an EC2 instance, run the command inside a Docker container, and will shut it down when finished. You can also add the `--gpu` option to run it on a GPU enabled instance.
