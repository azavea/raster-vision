# flake8: noqa

from os.path import abspath, dirname, join
from setuptools import setup, find_namespace_packages
import re

__version__ = '0.31.1'
requirement_constraints = {}

here = abspath(dirname(__file__))


def parse_requirements(requirements_path: str) -> list[str]:
    requirements = []
    with open(requirements_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'git+' in line:
                continue
            # match package name, ignoring version constraints
            match = re.match(r'^\s*([^\s<=>]+)', line)
            if not match:
                continue
            package_name = match.group(1)
            if package_name in requirement_constraints:
                constraint = requirement_constraints[package_name]
                package_name = f'{package_name}{constraint}'
            requirements.append(package_name)
    return requirements


def replace_images(readme) -> str:
    """Replaces image links in the README with static links to
    the GitHub release branch."""
    release_branch = '.'.join(__version__.split('.')[:2])
    r = r'\((docs/)(.*\.png)'
    rep = rf'(https://raw.githubusercontent.com/azavea/raster-vision/{release_branch}/docs/\g<2>'
    return re.sub(r, rep, readme)


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
    install_requires=parse_requirements(join(here, 'requirements.txt')))
