import io
from math import ceil

from PIL import Image
import numpy as np
import imageio
import atexit
import logging

log = logging.getLogger(__name__)


def save_img(im_array, output_path):
    imageio.imwrite(output_path, im_array)


def numpy_to_png(array: np.ndarray) -> str:
    """Get a PNG string from a Numpy array.

    Args:
         array: A Numpy array of shape (w, h, 3) or (w, h), where the
               former is meant to become a three-channel image and the
               latter a one-channel image.  The dtype of the array
               should be uint8.

    Returns:
         str

    """
    im = Image.fromarray(array)
    output = io.BytesIO()
    im.save(output, 'png')
    return output.getvalue()


def png_to_numpy(png: str, dtype=np.uint8) -> np.ndarray:
    """Get a Numpy array from a PNG string.

    Args:
         png: A str containing a PNG-formatted image.

    Returns:
         numpy.ndarray

    """
    incoming = io.BytesIO(png)
    im = Image.open(incoming)
    return np.array(im)


def terminate_at_exit(process):
    def terminate():
        log.debug('Terminating {}...'.format(process.pid))
        process.terminate()

    atexit.register(terminate)


def grouped(lst, size):
    """Returns a list of lists of length 'size'.
    The last list will have size <= 'size'.
    """
    return [lst[n:n + size] for n in range(0, len(lst), size)]


def split_into_groups(lst, num_groups):
    """Attempts to split a list into a given number of groups.
    The number of groups will be at least 1 and at most
    num_groups.

    Args:
       lst:             The list to split
       num_groups:      The number of groups to create.
    Returns:
       A list of size between 1 and num_groups containing lists
       of items of l."""
    group_sz = max(int(ceil((len(lst)) / num_groups)), 1)

    return grouped(lst, group_sz)
