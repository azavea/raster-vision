# keras-semantic-segmentation

Experiments in using deep learning with Keras/Tensorflow to perform semantic segmentation.

## Usage

### Requirements

- Vagrant 1.8+
- VirtualBox 4.3+
- Pip 8.1+

### Scripts

| Name     | Description                              |
| -------- | ---------------------------------------- |
| `cipublish`  | Publish docker image to ECR |
| `clean`  | Remove build outputs inside virtual machine |
| `infra`  | Execute Terraform subcommands            |
| `lint`   | Run flake8 on source code |
| `run` | Run container locally or remotely |
| `setup`  | Bring up the virtual machine and install dependent software on it |
| `update` | Install dependent software inside virtual machine |
| `upload_code` | Upload code to EC2 instance for rapid debugging |

### Initial Setup

From within the project root, execute the following commands.

```bash
$ ./scripts/setup
$ vagrant ssh
```

You will be prompted to enter the OpenTreeID AWS credentials, along with a default region. These credentials will be used to authenticate calls to the AWS API when using the AWS CLI and Terraform.

## Running locally on CPUs

To run an experiment locally, invoke
```shell
vagrant ssh
vagrant@otid:/vagrant$ ./scripts/update --cpu
vagrant@otid:/vagrant$ ./scripts/run --cpu
root@230fb62d8ecd:/opt/src# python -m model_training.run experiments/2_28_17/conv_logistic_test.json setup train eval
```

⚠️️ See [model_training/README.md](src/model_training/README.md) for more information on preparing data and running experiments.

## Running remotely on AWS EC2 GPUs

Support for Amazon EC2 GPU instances is provided through a combination of the AWS CLI and Terraform. The AWS CLI is used to produce a local AWS profile with valid credentials, and Terraform is used to bring up the appropriate EC2 instances using spot pricing.

### Spot Fleet

Once an AWS profile exists, use the `infra` script to interact with Amazon EC2 Spot Fleet API, which will request the lowest priced P2 GPU instance across all availability zones. The following command will start 1 instance and print their
public DNS names. More than 1 instance can be specified if needed.
```bash
vagrant@vagrant-ubuntu-trusty-64:/vagrant$ ./scripts/infra start 1
```

Afterwards, navigate to the [spot request](https://console.aws.amazon.com/ec2sp/v1/spot/home?region=us-east-1#) section of the EC2 console to monitor the request's progress. Once fulfilled, a running instance will visible in the [instances](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#Instances:sort=instanceId) section.

After initializing, the instance should have `nvidia-docker`, the GPU-enabled Docker image for this repo, and this repo and the datasets in the home directory.

### Running Models On A GPU Instance

After starting an instance, ssh into it with
```shell
ssh-add ~/.aws/open-tree-id.pem
ssh ubuntu@<public dns>
```

Then you can train a model with
```shell
cd keras-semantic-segmentation
./scripts/run --gpu
root@230fb62d8ecd:/opt/src# python -m model_training.run experiments/2_28_17/conv_logistic_test.json setup train eval
```

When running on EC2, the results will be saved to the `otid-data` S3 bucket after each epoch and the evaluation. If a run is terminated for any reason and you would like to resume it,
simply run the above command with the same options file, and it should pick up where it left off.

⚠️️ When done with all the instances, you should shut them down with
```shell
./scripts/infra destroy
```

### Publishing the Docker container to ECR

To enable fast bootup, we publish the latest Docker image to ECR and then download it when initializing the EC2 instance. To build and publish the container, run `./scripts/cipublish`.
