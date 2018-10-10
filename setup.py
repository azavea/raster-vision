# flake8: noqa

from os import path as op
import json
import io
import re
from setuptools import (setup, find_packages)
from imp import load_source

__version__ = load_source('rastervision.version',
                          'rastervision/version.py').__version__

here = op.abspath(op.dirname(__file__))

# get the dependencies and installs
with io.open(op.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [
    x.strip().replace('git+', '') for x in all_reqs if 'git+' not in x
]

def replace_images(readme):
    """Replaces image links in the README with static links to
    the GitHub release branch."""
    release_branch = '.'.join(__version__.split('.')[:2])
    r = r'\((docs/)(.*\.png)'
    rep = (r'(https://raw.githubusercontent.com/azavea/raster-vision/'
           '{}/docs/\g<2>'.format(release_branch))

    return re.sub(r, rep, readme)

# Dependencies for extras, which pertain to installing specific backends.
with io.open(op.join(here, 'extras_requirements.json'), encoding='utf-8') as f:
    extras_require = json.loads(f.read())

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
    packages=find_packages(exclude=['integration_tests*', 'tests*']),
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    dependency_links=dependency_links,
    entry_points='''
        [console_scripts]
        rastervision=rastervision.cli.main:main
    ''',
)
