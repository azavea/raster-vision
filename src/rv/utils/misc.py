import signal
import ctypes
import os
import json

import numpy as np
import scipy

from rv.utils.files import download_if_needed

# Constant taken from http://linux.die.net/include/linux/prctl.h
PR_SET_PDEATHSIG = 1


class PrCtlError(Exception):
    pass


# From http://evans.io/legacy/posts/killing-child-processes-on-parent-exit-prctl/  # noqa
def on_parent_exit(signame):
    """
    Return a function to be run in a child process which will trigger
    SIGNAME to be sent when the parent process dies
    """
    signum = getattr(signal, signame)

    def set_parent_exit_signal():
        # http://linux.die.net/man/2/prctl
        result = ctypes.cdll['libc.so.6'].prctl(PR_SET_PDEATHSIG, signum)
        if result != 0:
            raise PrCtlError('prctl failed with error code %s' % result)
    return set_parent_exit_signal


def save_img(path, arr):
    scipy.misc.imsave(path, arr)


def add_blank_chips(blank_count, chip_size, chip_dir):
    blank_im = np.zeros((chip_size, chip_size, 3))
    for blank_neg_ind in range(blank_count):
        chip_path = os.path.join(
            chip_dir, 'blank-{}.png'.format(blank_neg_ind))
        save_img(chip_path, blank_im)


def load_projects(projects_path, temp_dir):
    image_paths_list = []
    annotations_paths = []
    project_ids = []
    with open(projects_path, 'r') as projects_file:
        projects = json.load(projects_file)
        for project_ind, project in enumerate(projects):
            project_ids.append(project.get('id', project_ind))
            image_uris = project['images']
            image_paths = [download_if_needed(image_uri, temp_dir)
                           for image_uri in image_uris]
            image_paths_list.append(image_paths)
            annotations_uri = project['annotations']
            annotations_path = download_if_needed(annotations_uri, temp_dir)
            annotations_paths.append(annotations_path)

    return project_ids, image_paths_list, annotations_paths
