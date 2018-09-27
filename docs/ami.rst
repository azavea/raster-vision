Vagrant / Packer Instructions
=============================

These instructions are for getting Vagrant running and building and publishing a new AMI using Packer. We plan to eventually move this process to a Docker container and eliminate the use of Vagrant and Ansible for simplicity.

Requirements
------------

* Vagrant 1.8+

* VirtualBox 4.3+
* Python 2.7
* Ansible 2.1+

Scripts
-------

+------------+-------------------------------------------------------------------+
| Name       | Description                                                       |
+============+===================================================================+
| ``setup``  | Bring up the virtual machine and install dependent software on it |
+------------+-------------------------------------------------------------------+

Get Vagrant box running
-----------------------

First, set the ``RASTER_VISION_DATA_DIR`` environment variable on your host machine. All data including datasets and results should be stored in a single directory outside of the repo. The ``Vagrantfile`` maps the ``RASTER_VISION_DATA_DIR`` environment variable on the host machine to ``/opt/data`` on the guest machine. Within the project root, execute the following commands to setup and then log into the VM.

.. code-block:: console

                $ ./scripts/setup
                $ vagrant ssh


If you get an error message about the Docker daemon not being started, you
may want to run ``vagrant provision``.

Updating the Batch AMI
----------------------

In order to run scripts on GPUs and in parallel, we use `AWS Batch <https://aws.amazon.com/batch/>`_.

Use ``deployment/batch_amis.py`` to update the Batch environment AMI. This requires your ``raster-vision`` AWS profile to be configured.

.. code-block:: console

                $ aws --profile raster-vision configure
                $ cd deployment
                $ pip install -r requirements.txt
                $ ./batch_amis.py  build-amis --aws-profile raster-vision
                ...
                ==> Builds finished. The artifacts of successful builds are:
                --> raster-vision-gpu: AMIs were created:

                us-east-1: ami-fb5c7980


Use the AMI ID provided above, to update the ComputeResources > imageId field in ``deployment/batch/compute_environment_{gpu,cpu}.json``. To apply these changes, delete the existing Batch environments using the AWS Console, and then re-run the steps in the section above.

Prune any old AMIs by using the ``prune-amis`` command to ``batch_amis.py``

.. code-block:: console

                $ ./batch_amis.py  prune-amis --keep 10
