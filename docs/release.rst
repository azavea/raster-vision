Release Process
===============

This is a guide to the process of creating a new release, and is meant for the maintainers of Raster Vision.

.. note:: The following instructions assume that Python 3 is the default Python on your local system. Using Python 2 will not work.

Minor or Major Version Release
------------------------------

#.  It's a good idea to update any major dependencies before the release.
#.  Update THIRD_PARTY_LICENSES.txt:

    #. Use ``third_party_license_file_generator`` (``pip install third_party_license_file_generator``) to auto generate the file. Update ``third_party_license_file_generator_overrides.yml``, if necessary.

       .. code-block:: console

           cat `find **/requirements.txt` requirements.txt requirements-dev.txt > requirements-all.txt;

           python -m third_party_license_file_generator \
            -r requirements-all.txt \
            -p `which python` \
            -o THIRD_PARTY_LICENSES.txt \
            -l third_party_license_file_generator_overrides.yml;

#.  Test examples:

    #.  Checkout the ``master`` branch, re-build the docker image (``docker/build``), and push it to ECR (``docker/ecr_publish``).
    #. Follow the instructions in `this README <{{ repo_examples }}/README.md>`__ to do the following:
        
       #.  Run all :ref:`rv examples` and check that evaluation metrics are close to the scores from the last release. (For each example, there should be a link to a JSON file with the evaluation metrics from the last release.) This stage often uncovers bugs, and is the most time consuming part of the release process.
       #.  Collect all model bundles, and check that they work with the ``predict`` command and sanity check output in QGIS.
       #.  Update the :ref:`model zoo` by uploading model bundles and sample images to the right place on S3. If you use the ``collect`` command (`see <{{ repo_examples }}/README.md>`__), you should be able to sync the ``collect_dir`` to ``s3://azavea-research-public-data/raster-vision/examples/model-zoo-<version>``.
       #. Screenshot the outputs of the ``compare`` command (for each example) and include them in the PR described below.

#.  Test notebooks:

    #.  Update the `tutorial notebooks <{{ repo }}/docs/usage/tutorials/>`__ that use models from the model zoo so that they use the latest version.
    #.  Execute all `tutorial notebooks <{{ repo }}/docs/usage/tutorials/>`__ and make sure they work correctly. Do not commit output changes unless code behavior has changed.

#. Test/update docs:

   #.  Update the docs if needed. See the `docs README <{{ repo }}/docs/README.md>`__ for instructions.
   #.  Update `tiny_spacenet.py <{{ repo_examples }}/tiny_spacenet.py>`__ if needed and ensure the line numbers in every ``literalinclude`` of that file are correct. Tip: you can find all instances by searching the repo using the regex: ``\.\. literalinclude:: .+tiny_spacenet\.py$``.
   #.  Test :ref:`setup` and :ref:`quickstart` instructions and make sure they work.
   #.  Test examples from :ref:`bootstrap`.

       .. code-block:: console

           cookiecutter /opt/src/cookiecutter_template

   #.  Update the `the changelog <{{ repo }}/docs/changelog.rst>`__, and point out API changes.
   #.  Fix any broken badges on the GitHub repo readme.

#.  Update the version number. This occurs in several places, so it's best to do this with a find-and-replace over the entire repo.
#.  Make a PR to the ``master`` branch with the preceding updates. In the PR, there should be a link to preview the docs. Check that they are building and look correct.
#.  Make a git branch with the version as the name, and push to GitHub.
#.  Ensure that the docs are building correctly for the new version branch on `readthedocs <https://readthedocs.org/projects/raster-vision/>`_. You will need to have admin access on your RTD account. Once the branch is building successfully, Under *Versions -> Activate a Version*, you can activate the version to add it to the sidebar of the docs for the latest version. (This might require manually triggering a rebuild of the docs.) Then, under *Admin -> Advanced Settings*, change the default version to the new version.
#.  GitHub Actions is supposed to publish an image whenever there is a push to a branch with a version number as the name. If this doesn't work or you want to publish it immediately, then you can manually make a Docker image for the new version and push to Quay. For this you will need an account on Quay.io under the Azavea organization.

    .. code-block:: console

        ./docker/build
        docker login quay.io
        docker tag raster-vision-pytorch:latest quay.io/azavea/raster-vision:pytorch-<version>
        docker push quay.io/azavea/raster-vision:pytorch-<version>

#.  Make a GitHub `tag <https://github.com/azavea/raster-vision/tags>`_ and `release <https://github.com/azavea/raster-vision/releases>`_ using the previous release as a template.
#. Remove artifacts from previous builds. From the repo root:

    .. code-block:: console

        rm -rf build/ dist/ *.egg-info
        rm -rf rastervision_*/build rastervision_*/dist rastervision_*/*.egg-info

#.  Publish all packages to PyPI. This step requires `twine <https://twine.readthedocs.io/en/stable/>`__ which you can install with

    .. code-block:: console

        pip install twine

    To store settings for PyPI you can set up a ``~/.pypirc`` file containing:

    .. code-block:: console

        [pypi]
        username = azavea

        [testpypi]
        username = azavea

    Once packages are published they cannot be changed, so be careful. (It's possible to practice using TestPyPI.) Navigate to the repo's root directory on your local filesystem. With the version branch checked out, run the following scripts to build packages and publish to PyPI. 
    
    Build:

    .. code-block:: console

        scripts/pypi_build

    Publish to TestPyPI. (You will be prompted for the PyPI password multiple times--once for each package.)

    .. code-block:: console

        scripts/pypi_publish --test

    You can then test it with ``pip`` like so:

    .. code-block:: console

        pip install --index-url https://test.pypi.org/simple/ rastervision

    Finally, if everything looks okay, publish to Pypi.  (You will be prompted for the PyPI password multiple times--once for each package.)

    .. code-block:: console

        scripts/pypi_publish

#.  Announce the new release in our `forum <https://github.com/azavea/raster-vision/discussions>`_, and with a blog post if it's a big release.
#.  Make a PR to the master branch that updates the version number to the next development version, ``X.Y.Z-dev``. For example, if the last release was ``0.20.1``, update the version to ``0.20.2-dev``.

Patch Release
-----------------

This describes how to create a new patch release (AKA a bug-fix release), using an increment from 0.8.0 to 0.8.1 as an example. This assumes that there is already a branch for a minor release called ``0.8``.

#. Backport changes to the ``0.8`` branch. To create a patch release (version 0.8.1), we need to backport all the commits on the ``master`` branch that have been added since the last patch release onto the ``0.8`` branch. To do this:

   #. Create a new branch from the ``0.8`` branch. Let's call it ``backport``.
   #. Cherry-pick each commit that we want to include from the ``master`` branch onto the ``backport`` branch.
   #. Make a PR against the ``0.8`` branch from the ``backport`` branch. The title of the PR should start with ``[BACKPORT]``.
#. Update changelog and version on the ``0.8`` branch. Make and merge a PR against ``0.8`` (but not ``master``) that adds a changelog for the new release and increments the version to ``0.8.1`` throughout the repo. Wait for the ``0.8`` branch to be built by GitHub Actions and the ``0.8`` Docker images to be published to Quay. If that is successful, we can proceed to the next steps of actually publishing a release.
#. Publish the new version to PyPI. Follow the same instructions for PyPI as listed above for minor/major version releases.
#. Using the GitHub UI, make a new release. Use ``v0.8.1`` as the tag, and the ``0.8`` branch as the target.
#. Update changelog and version on the ``master`` branch. Make and merge a PR against ``master`` that 

   * includes the cherry-picked commit that updates the changelog for ``0.8.1`` and 
   * increments the version to ``0.8.2-dev`` throughout the repo.
