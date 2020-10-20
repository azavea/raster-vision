import io
from pydantic import confloat

from PIL import Image
import numpy as np
import imageio
import logging

Proportion = confloat(ge=0, le=1)

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
