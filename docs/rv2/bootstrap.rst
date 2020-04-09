.. _rv2_bootstrap:

Bootstrap new projects with a template
=======================================

When using Raster Vision on a new project, the best practice is to create a new repo with its own Docker image based on the Raster Vision image. This involves a certain amount of boilerplate code which has a few things that vary between projects. To facilitate bootstrapping new projects, there is a `cookiecutter <https://cookiecutter.readthedocs.io/>`_ template. Assuming that you have cloned the Raster Vision repo and have run ``pip install cookiecutter==1.7.0``, you can run it as follows (after adjusting paths appropriately for your particular setup).

.. code-block:: console

    $ cookiecutter /Users/lfishgold/projects/raster-vision/rastervision2/examples/cookiecutter_template
    project_name [my_project]:
    docker_image [my_project]:
    parent_docker_image [quay.io/azavea/raster-vision:pytorch-latest]:
    [lfishgold@monoshone ~/projects]
    $ tree my_project/
    my_project/
    ├── Dockerfile
    ├── README.md
    ├── configs
    │   ├── __init__.py
    │   └── test.py
    ├── docker
    │   ├── build
    │   └── run
    ├── rastervision2
    │   └── my_project
    │       ├── __init__.py
    │       ├── test_pipeline.py
    │       └── test_pipeline_config.py
    └── scripts
        └── debug

    5 directories, 10 files
