.. _bootstrap:

Bootstrap new projects with a template
=======================================

When using Raster Vision on a new project, the best practice is to create a new repo with its own Docker image based on the Raster Vision image. This involves a fair amount of boilerplate code which has a few things that vary between projects. To facilitate bootstrapping new projects, there is a `cookiecutter <https://cookiecutter.readthedocs.io/>`_ `template <https://github.com/azavea/raster-vision/tree/0.12/cookiecutter_template>`_. Assuming that you cloned the Raster Vision repo and ran ``pip install cookiecutter==1.7.0``, you can instantiate the template as follows (after adjusting paths appropriately for your particular setup).

.. code-block:: terminal

    [lfishgold@monoshone ~/projects]
    $ cookiecutter raster-vision/cookiecutter_template/
    caps_project_name [MY_PROJECT]:
    project_name [my_project]:
    docker_image [my_project]:
    parent_docker_image [quay.io/azavea/raster-vision:pytorch-0.12]:
    version [0.12]:
    description [A Raster Vision plugin]:
    url [https://github.com/azavea/raster-vision]:
    author [Azavea]:
    author_email [info@azavea.com]:

    [lfishgold@monoshone ~/projects]
    $ tree my_project/
    my_project/
    ├── Dockerfile
    ├── README.md
    ├── docker
    │   ├── build
    │   ├── ecr_publish
    │   └── run
    └── rastervision_my_project
        ├── rastervision
        │   └── my_project
        │       ├── __init__.py
        │       ├── configs
        │       │   ├── __init__.py
        │       │   └── test.py
        │       ├── test_pipeline.py
        │       └── test_pipeline_config.py
        ├── requirements.txt
        └── setup.py

    5 directories, 12 files

The output is a repo structure with the skeleton of a Raster Vision plugin that can be pip installed, and everything needed to build, run, and publish a Docker image with the plugin. The resulting ``README.md`` file contains setup and usage information for running locally and on Batch, which makes use of the :ref:`CloudFormation setup <cloudformation jobdefs>` for creating new user/project-specific job defs.
