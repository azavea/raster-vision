import setuptools

setuptools.setup(
    name="rastervision",
    version="0.8.0",
    description='An open source framework for deep learning'
    'on satellite and aerial imagery',
    long_description=open('../README.md').read(),
    url='https://github.com/azavea/raster-vision',
    author='Raster Vision',
    author_email='info@azavea.com',
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    keywords='raster deep-learning ml computer-vision earth-observation geospatial geospatial-processing',
    packages=setuptools.find_packages(exclude=['tests', 'tests.*', 'integration_tests', 'integration_tests.*']),
    package_data={'rastervision.backend': ['*.json']},
    install_requires=[
        'networkx >= 2.1',
        'everett >= 0.9',
        'pluginbase >= 0.7',
        'npstreams >= 1.4.*',
        'lxml >= 4.2.*',
        'shapely >= 1.6.*',
        'pyproj >= 1.9.5.*',
        'imageio >= 2.3.*',
        'scikit-learn >= 0.19.*',
        'six >= 1.11.*',
        'h5py >= 2.7.*',
        'matplotlib >= 2.1.*',
        'pillow >= 5.0.*',
        'click >= 6.*'
    ],
    tests_require=[],
    entry_points='''
        [console_scripts]
        rastervision=rastervision.cli.main:main
    ''',
)
