# flake8: noqa

from os.path import abspath, dirname, join
from setuptools import setup, find_namespace_packages
import re

name = 'rastervision_aws_s3'
version = '0.31.1'
description = 'A rastervision plugin that adds an AWS S3 file system'
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


setup(
    name=name,
    version=version,
    description=description,
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
    packages=find_namespace_packages(exclude=['integration_tests*', 'tests*']),
    install_requires=parse_requirements(join(here, 'requirements.txt')),
    zip_safe=False)
