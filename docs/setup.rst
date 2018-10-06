Setup
=====

.. _docker containers:

Docker Containers
-----------------

Using the docker containers published for Raster Vision allows
you to use a fully set up environment.

.. _aws batch setup:

Setting up AWS Batch
--------------------

If you want to run code against AWS, you'll have to have a Raster Vision AWS Batch setup on your account, which you can accomplish through the instructions at the  `Raster Vision AWS repository <https://github.com/azavea/raster-vision-aws>`_.

Make sure to set the appropriate configuration in your ``$HOME/.rastervision/default`` configuration, e.g.

.. code:: ini

   [AWS_BATCH]
   job_queue=rasterVisionQueue
   job_definition=raster-vision-gpu

.. seealso::
   For more information about how Raster Vision uses AWS Batch, see the section: :ref:`aws batch`.
