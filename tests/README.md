# Unit tests

## Adding tests
In general, tests for code in the file
```
rastervision_some_plugin/rastervision/some_plugin/some_module/some_submodule/some_file.py
```
should go in 
```
tests/rastervision_some_plugin/some_module/some_submodule/test_some_file.py
```

Every directory in the path of the test file _must_ have an `__init__.py` file for it to be discoverable by `unittest`.

## Running tests

### Run all unit tests
```sh
scripts/unit_tests
```
Or (from the repo root):
```sh
python -m unittest discover -t . tests -vf
```

### Run specific unit tests

A file:
```sh
python -m unittest tests/core/data/utils/test_geojson.py -vf
```

A `unittest.TestCase` class:
```sh
python -m unittest tests.core.data.utils.test_geojson.TestGeojsonUtils -vf
```

A single test:
```sh
python -m unittest tests.core.data.utils.test_geojson.TestGeojsonUtils.test_merge_geojsons -vf
```

### Generate code coverage report
Produce a `.coverage` file (from the repo root):
```sh
coverage run -m unittest discover -t . tests -vf
```

Generate an HTML report from the `.coverage` file:
```sh
coverage html
```

See `coverage help` for other available report formats.
