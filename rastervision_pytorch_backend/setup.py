from os import path as op
import io
from setuptools import (setup, find_namespace_packages)

here = op.abspath(op.dirname(__file__))
with io.open(op.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')
install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

setup(
    name='rastervision_pytorch_backend',
    version='0.12',
    packages=find_namespace_packages(exclude=['integration_tests*', 'tests*']),
    install_requires=install_requires,
    zip_safe=False,
)
