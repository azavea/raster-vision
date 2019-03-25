.. _release:

Release Process
===============

This is a guide to the process of creating a new release, and is meant for the maintainers of Raster Vision. It describes how to create a new bug fix release, using incrementing from 0.8.0 to 0.8.1 as an example. The process for minor and major releases are somewhat different, and will be documented in the future.

.. note:: The following instructions assume that Python 3 is the default Python on your local system. Using Python 2 will not work.

Prepare branch
---------------

This assumes that there is already a branch for a minor release called ``0.8``. To create a bug fix release (version 0.8.1), we need to backport all the bug fix commits on the ``master`` branch into the ``0.8`` branch that have been added since the last bug fix release. For each bug fix PR on ``master`` we need to create a PR against ``0.8`` based on a branch of ``0.8`` that has cherry-picked the commits from the original PR. The title of the PR should start with [BACKPORT]. Our goal is to create and merge each backport PR immediately after each bug fix PR is merged, so hopefully the preceding is already done by the time we are creating a bug fix release.

Make and merge a PR against ``0.8`` (but not ``master``) that increments ``version.py`` to ``0.8.1``.
Then wait for the ``0.8`` branch to be built by Travis and the ``0.8`` Docker images to be published to Quay. If that is successful, we can proceed to the next steps of actually publishing a release.

Make Github release
----------------------
Using the Github UI, make a new release. Use ``0.8.1`` as the tag, and ``0.8`` as the target.

Make Docker image
-------------------
The image for ``0.8`` is created automatically by Travis, but we need to manually create images for ``0.8.1``. For this you will need an account on Quay.io under the Azavea organization.

.. code-block:: console

    docker login quay.io

    docker pull quay.io/azavea/raster-vision:cpu-0.8
    docker tag quay.io/azavea/raster-vision:cpu-0.8 quay.io/azavea/raster-vision:cpu-0.8.1
    docker push quay.io/azavea/raster-vision:cpu-0.8.1

    docker pull quay.io/azavea/raster-vision:gpu-0.8
    docker tag quay.io/azavea/raster-vision:gpu-0.8 quay.io/azavea/raster-vision:gpu-0.8.1
    docker push quay.io/azavea/raster-vision:gpu-0.8.1

Make release on PyPI
---------------------
Once a release is created on PyPI it can't be deleted, so be careful. This step requires ``twine`` which you can install with ``pip install twine``. To store settings for PyPI you can setup a ``~/.pypirc`` file containing:

.. code-block:: shell

    [pypi]
    username = azavea

To create the release distribution, navigate to the ``raster-vision`` repo on your local filesystem on an up-to-date branch ``0.8.``. Then run

.. code-block:: console

    python setup.py sdist bdist_wheel

The contents of the distribution will be in ``dist/``. When you are ready to upload to PyPI, run:

.. code-block:: console

    twine upload dist/*

Announcement
------------

Let people in the Gitter channel know there is a new version.
