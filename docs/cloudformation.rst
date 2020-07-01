.. _cloudformation setup:

Setup AWS Batch using CloudFormation
=====================================

This describes the deployment code that sets up the necessary AWS resources to utilize the AWS Batch runner. Using Batch is advantageous because it starts and stops instances automatically and runs jobs sequentially or in parallel according to the dependencies between them. In addition, this deployment sets up distinct CPU and GPU resources and utilizes spot instances, which is more cost-effective than always using a GPU on-demand instance. Deployment is driven via the AWS console using a `CloudFormation template <https://aws.amazon.com/cloudformation/aws-cloudformation-templates/>`_.

This AWS Batch setup is an "advanced" option that assumes some familiarity with `Docker <https://docs.docker.com/>`_, AWS `IAM <https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html>`_, `named profiles <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html>`_, `availability zones <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html>`_, `EC2 <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html>`_, `ECR  <https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html>`_, `CloudFormation <https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/Welcome.html>`_, and `Batch <https://docs.aws.amazon.com/batch/latest/userguide/what-is-batch.html>`_.

AWS Account Setup
-------------------

In order to setup Batch using this repo, you will need to setup your AWS account so that:

* you have either root access to your AWS account, or an IAM user with admin permissions. It is probably possible with less permissions, but we haven't figured out how to do this yet after some experimentation.
* you have the ability to launch P2 or P3 instances which have GPUs.
* you have requested permission from AWS to use availability zones outside the USA if you would like to use them. (New AWS accounts can't launch EC2 instances in other AZs by default.) If you are in doubt, just use ``us-east-1``.

AWS Credentials
----------------

Using the AWS CLI, create an AWS profile for the target AWS environment. An example, naming the profile ``raster-vision``:

.. code-block:: terminal

    $ aws --profile raster-vision configure
    AWS Access Key ID [****************F2DQ]:
    AWS Secret Access Key [****************TLJ/]:
    Default region name [us-east-1]: us-east-1
    Default output format [None]:

You will be prompted to enter your AWS credentials, along with a default region. The Access Key ID and Secret Access Key can be retrieved from the IAM console. These credentials will be used to authenticate calls to the AWS API when using Packer and the AWS CLI.

Deploying Batch resources
--------------------------

To deploy AWS Batch resources using AWS CloudFormation, start by logging into your AWS console. Then, follow the steps below:

- Navigate to ``CloudFormation > Create Stack``
- In the ``Choose a template field``, select ``Upload a template to Amazon S3`` and upload the template in `cloudformation/template.yml <https://github.com/azavea/raster-vision/tree/0.12/cloudformation/template.yml>`_. **Warning:** Some versions of Chrome fail at this step without an explanation. As a workaround, try a different version of Chrome, or Firefox. See `this thread <https://forums.aws.amazon.com/thread.jspa?messageID=946331&tstart=0>`_ for more details.
- ``Prefix``: If you are setting up multiple RV stacks within an AWS account, you need to set a prefix for namespacing resources. Otherwise, there will be name collisions with any resources that were created as part of another stack.
- Specify the following required parameters:
    - ``Stack Name``: The name of your CloudFormation stack
    - ``VPC``: The ID of the Virtual Private Cloud in which to deploy your resource. Your account should have at least one by default.
    - ``Subnets``: The ID of any subnets that you want to deploy your resources into. Your account should have at least two by default; make sure that the subnets you select are in the VPC that you chose by using the AWS VPC console, or else CloudFormation will throw an error. (Subnets are tied to availability zones, and so affect spot prices.) In addition, you need to choose subnets that are available for the instance type you have chosen. To find which subnets are available, go to Spot Pricing History in the EC2 console and select the instance type. Then look up the availability zones that are present in the VPC console to find the corresponding subnets. Your spot requests will be more likely to be successful and your savings will be greater if you have subnets in more availability zones.

    .. image:: img/spot-azs.png
        :width: 500
        :alt: Spot availability zones for P3 instances

    - ``SSH Key Name``: The name of the SSH key pair you want to be able to use to shell into your Batch instances. If you've created an EC2 instance before, you should already have one you can use; otherwise, you can create one in the EC2 console. *Note: If you decide to create a new one, you will need to log out and then back in to the console before creating a Cloudformation stack using this key.*
    - ``Instance Types``: Provide the instance types you would like to use. (For GPUs, ``p3.2xlarge`` is approximately 4 times the speed for 4 times the price.)
- Adjust any preset parameters that you want to change (the defaults should be fine for most users) and click ``Next``.
    - Advanced users: If you plan on modifying Raster Vision and would like to publish a custom image to run on Batch, you will need to specify an ECR repo name and a tag name. Note that the repo names cannot be the same as the Stack name (the first field in the UI) and cannot be the same as any existing ECR repo names. If you are in a team environment where you are sharing the AWS account, the repo names should contain an identifier such as your username.
- Accept all default options on the ``Options`` page and click ``Next``
- Accept "I acknowledge that AWS CloudFormation might create IAM resources with custom names" on the ``Review`` page and click ``Create``
- Watch your resources get deployed!

Publish local Raster Vision images to ECR
-------------------------------------------

If you setup ECR repositories during the CloudFormation setup (the "advanced user" option), then you will need to follow this step, which publishes local Raster Vision images to those ECR repositories. Every time you make a change to your local Raster Vision images and want to use those on Batch, you will need to run these steps:

* Run ``./docker/build`` in the Raster Vision repo to build a local copy of the Docker image.
* Run ``./docker/ecr_publish`` in the Raster Vision repo to publish the Docker images to ECR. Note that this requires setting the ``RV_ECR_IMAGE`` environment variable to be set to ``<ecr_repo_name>:<tag_name>``.

Update Raster Vision configuration
-----------------------------------

Finally, make sure to update your :ref:`aws batch setup` with the Batch resources that were created.

.. _cloudformation jobdefs:

Deploy new job definitions
-----------------------------

When a user starts working on a new RV-based project (or a new user starts working on an existing RV-based project), they will often want to publish a custom Docker image to ECR and use it when running on Batch. To facilitate this, there is a separate `cloudformation/job_def_template.yml <https://github.com/azavea/raster-vision/tree/0.12/cloudformation/job_def_template.yml>`_. The idea is that for each user/project pair which is identified by a ``Namespace`` string, a CPU and GPU job definition is created which point to a specified ECR repo using that ``Namespace`` as the tag. After creating these new resources, the image should be published to ``<repo>:<namespace>`` on ECR, and the new job definitions should be placed in a project-specific RV profile file.
