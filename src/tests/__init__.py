import os


def data_file_path(rel_path):
    data_dir = os.path.join(os.path.dirname(__file__), 'data-files')
    return os.path.join(data_dir, rel_path)
