# keras-image-segmentation

Learning to segment images using Keras

## Table of Contents

- [Usage](#usage)
  - [Requirements](#requirements)
  - [Quick Setup](#quick-setup)
  - [Scripts](#scripts)

* [Amazon EC2 GPU Instances](#amazon-ec2-gpu-instances)
  * [Credentials](#credentials)
  * [Spot Fleet](#spot-fleet)
  * [NVIDIA Docker](#nvidia-docker)

## Usage

### Requirements

- Vagrant 1.8+
- VirtualBox 4.3+
- Pip 8.1+

### Quick Setup

From within the project root, execute the following commands:

```bash
$ ./scripts/setup
$ vagrant ssh
```

### Scripts

| Name     | Description                              |
| -------- | ---------------------------------------- |
| `setup`  | Bring up the virtual machine and install dependent software on it |
| `infra`  | Execute Terraform subcommands            |
| `update` | Install dependent software inside virtual machine |
| `clean`  | Remove build outputs inside virtual machine |

### Running Models In Development

From within the VM, use `update` to build the model runner container,
then run it with a model script and an attached output volume.

```shell
$ mkdir -p output
$ sudo su
# ./scripts/update
...container build output...
# docker run --rm \
    -v /vagrant/output:/opt/model_training/output \
    otid-model-training
```

To run a model other than the default

```shell
# docker run --rm \
    -v /vagrant/output:/opt/model_training/output \
    otid-model-training python experiment.py
```
## Amazon EC2 GPU Instances

Support for Amazon EC2 GPU instances is provided through a combination of the AWS CLI and Terraform. The AWS CLI is used to produce a local AWS profile with valid credentials, and Terraform is used to bring up the appropriate EC2 instances using spot pricing.

### Credentials

Using the AWS CLI, create an AWS profile named `open-tree-id`:

```bash
vagrant@vagrant-ubuntu-trusty-64:/vagrant$ unset AWS_PROFILE
vagrant@vagrant-ubuntu-trusty-64:/vagrant$ aws --profile open-tree-id configure
AWS Access Key ID [****************F2DQ]:
AWS Secret Access Key [****************TLJ/]:
Default region name [us-east-1]: us-east-1
Default output format [None]:
```

You will be prompted to enter the OpenTreeID AWS credentials, along with a default region. These credentials will be used to authenticate calls to the AWS API when using the AWS CLI and Terraform.

### Spot Fleet

Once an AWS profile exists, use the `infra` script to interact with Amazon EC2 Spot Fleet API, which will request the lowest priced GPU instance across all availability zones:

```bash
vagrant@vagrant-ubuntu-trusty-64:/vagrant$ ./scripts/infra plan
vagrant@vagrant-ubuntu-trusty-64:/vagrant$ ./scripts/infra apply
```

**Note**: Currently only the `g2.2xlarge` and `p2.xlarge` instance types are part of the Spot Fleet request.

Afterwards, navigate to the [spot request](https://console.aws.amazon.com/ec2sp/v1/spot/home?region=us-east-1#) section of the EC2 console to monitor the request's progress. Once fulfilled, a running instance will visible in the [instances](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#Instances:sort=instanceId) section.

### NVIDIA Docker

After a GPU instance is up-and-running, use SSH to connect to it via its public hostname. Once logged in, the NVIDIA drivers and `nvidia-docker` tooling will be available.

Below is an example of using the `nvidia-docker`tooling to confirm that the GPUs can be accessed from within the running container:

```bash
$ ssh -i ~/.ssh/open-tree-id.pem -l ec2-user ec2-52-201-251-205.compute-1.amazonaws.com
[ec2-user@ip-172-31-31-168 ~]$ nvidia-docker run --rm nvidia/cuda:7.5-runtime nvidia-smi
7.5-runtime: Pulling from nvidia/cuda
04c996abc244: Pull complete
d394d3da86fe: Pull complete
bac77aae22d4: Pull complete
b48b86b78e97: Pull complete
09b3dd842bf5: Pull complete
6800a2eb7cad: Pull complete
9cddad8d6231: Pull complete
8f8b080957aa: Pull complete
5fc28fb955da: Pull complete
Digest: sha256:bbb67fd482c06bc9159209abd3f0aed3fc80643f01cd1d27acc988fc5830ec6c
Status: Downloaded newer image for nvidia/cuda:7.5-runtime
Thu Oct 20 21:28:46 2016
+------------------------------------------------------+
| NVIDIA-SMI 352.99     Driver Version: 352.99         |
|-------------------------------|----------------------|----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           On   | 0000:00:1E.0     Off |                    0 |
| N/A   32C    P8    30W / 149W |     55MiB / 11519MiB |      0%      Default |
+-------------------------------|----------------------|----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```


### Running Models On A GPU Instance

You can use the `run` script to bring up a GPU-enabled AWS instance, copy code and data to it, and build and run the docker image. This requires adding the `open-tree-id.pem` key to your `ssh-agent`.

```shell
./scripts/run --remote
```

The script leaves the EC2 instance running, so you can tweak changes locally and call `run` again to rerun the docker image with your latest changes.

When done, you can shut down the EC2 instance with:

```shell
./scripts/infra destroy
```
