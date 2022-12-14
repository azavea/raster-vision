from typing import Dict, Iterable, List, Optional, Tuple
from os import PathLike
from os.path import splitext
from pathlib import Path
from itertools import chain

import numpy as np
from torchvision.datasets.folder import (IMG_EXTENSIONS, DatasetFolder)
from PIL import Image

IMG_EXTENSIONS = tuple([*IMG_EXTENSIONS, '.npy'])


class DatasetError(Exception):
    pass


class ImageDatasetError(DatasetError):
    pass


class GeoDatasetError(DatasetError):
    pass


def discover_images(dir: PathLike,
                    extensions: Iterable[str] = IMG_EXTENSIONS) -> List[Path]:
    """Find all images with the given ``extensions`` in ``dir``."""
    dir = Path(dir)
    img_paths = chain.from_iterable(
        (dir.glob(f'*{ext}') for ext in extensions))
    return list(img_paths)


def load_image(path: PathLike) -> np.ndarray:
    """Read in image from path and return as a (H, W, C) numpy array."""
    ext = splitext(path)[-1]
    if ext == '.npy':
        img = np.load(path)
    else:
        img = np.array(Image.open(path))

    if img.ndim == 2:
        # (h, w) --> (h, w, 1)
        img = img[..., np.newaxis]

    return img


def make_image_folder_dataset(data_dir: str,
                              classes: Optional[Iterable[str]] = None
                              ) -> DatasetFolder:
    """Initializes and returns an ImageFolder.

    If classes is specified, ImageFolder's default class-to-index mapping
    behavior is overriden to use the indices of classes instead.
    """
    if classes is None:
        return DatasetFolder(
            data_dir, loader=load_image, extensions=IMG_EXTENSIONS)

    class ImageFolder(DatasetFolder):
        def find_classes(self,
                         directory: str) -> Tuple[List[str], Dict[str, int]]:
            """Override to force mapping from class name to class index."""
            return classes, {c: i for (i, c) in enumerate(classes)}

    return ImageFolder(data_dir, loader=load_image, extensions=IMG_EXTENSIONS)
