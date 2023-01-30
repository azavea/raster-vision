# flake8: noqa

from os import path as op
import os
import json
import io
import re
from setuptools import (setup, find_namespace_packages)
from imp import load_source

here = op.abspath(op.dirname(__file__))
__version__ = '0.20.2'

# get the dependencies and installs
with io.open(op.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

# The RTD build environment fails with the reqs in bad_reqs.
if 'READTHEDOCS' in os.environ:
    bad_reqs = ['pyproj', 'h5py']
    all_reqs = list(
        filter(lambda r: r.split('==')[0] not in bad_reqs, all_reqs))

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]


def replace_images(readme):
    """Replaces image links in the README with static links to
    the GitHub release branch."""
    release_branch = '.'.join(__version__.split('.')[:2])
    r = r'\((docs/)(.*\.png)'
    rep = (r'(https://raw.githubusercontent.com/azavea/raster-vision/'
           '{}/docs/\g<2>'.format(release_branch))

    return re.sub(r, rep, readme)


# del extras_require['feature-extraction']

setup(
    name='rastervision',
    version=__version__,
    description='An open source framework for deep learning '
    'on satellite and aerial imagery',
    long_description=replace_images(open('README.md').read()),
    long_description_content_type='text/markdown',
    url='https://github.com/azavea/raster-vision',
    author='Azavea',
    author_email='info@azavea.com',
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    keywords=
    'raster deep-learning ml computer-vision earth-observation geospatial geospatial-processing',
    packages=[],
    include_package_data=True,
    install_requires=install_requires)
