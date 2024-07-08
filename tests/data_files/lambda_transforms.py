from typing import TYPE_CHECKING

import albumentations as A

if TYPE_CHECKING:
    import numpy as np


def ndvi(rgb_nir: 'np.ndarray', **kwargs) -> 'np.ndarray':
    red = rgb_nir[..., 0]
    nir = rgb_nir[..., 3]
    ndvi = (nir - red) / (nir + red)
    return ndvi


def swap(image: 'np.ndarray', **kwargs) -> 'np.ndarray':
    return image[..., [3, 4, 5, 0, 1, 2]]


lambda_transforms = {
    'ndvi': A.Lambda(name='ndvi', image=ndvi, always_apply=True),
    'swap': A.Lambda(name='swap', image=swap, always_apply=True)
}
