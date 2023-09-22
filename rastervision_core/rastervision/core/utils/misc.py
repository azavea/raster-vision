from typing import TYPE_CHECKING
from pydantic import confloat

from skimage.io import imsave

if TYPE_CHECKING:
    import numpy as np

Proportion = confloat(ge=0, le=1)


def save_img(im_array: 'np.ndarray', output_path: str):
    imsave(output_path, im_array)
