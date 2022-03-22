import os

# It is important for some tests that this import runs before unittest imports
# any other tests so that the registry is set up correctly.
# To ensure this file gets executed first, unit tests should be run using:
# python -m unittest discover -t /opt/src tests -vf
import rastervision.pipeline  # noqa


def data_file_path(rel_path):
    data_dir = os.path.join(os.path.dirname(__file__), 'data_files')
    return os.path.join(data_dir, rel_path)
