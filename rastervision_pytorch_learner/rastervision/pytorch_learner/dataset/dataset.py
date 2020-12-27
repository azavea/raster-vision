from typing import Union, Optional, Tuple, Any, TypeVar

import numpy as np
import albumentations as A

import torch
from torch.utils.data import Dataset

from rastervision.core.box import Box
from rastervision.core.data import Scene
from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt
from rastervision.pytorch_learner.dataset.transform import (TransformType,
                                                            TF_TYPE_TO_TF_FUNC)


class AlbumentationsDataset(Dataset):
    """An adapter to use arbitrary datasets with albumentations transforms."""

    def __init__(self,
                 orig_dataset: Any,
                 transform: Optional[A.BasicTransform] = None,
                 transform_type: TransformType = TransformType.noop,
                 normalize=True,
                 to_pytorch=True):
        """Constructor.

        Args:
            orig_dataset (Any): An object with a __getitem__ and __len__.
            transform (A.BasicTransform, optional): An Albumentations
                transform. Defaults to None.
            transform_type (TransformType): The type of transform so that its
                inputs and outputs can be handled correctly. Defaults to
                TransformType.noop.
            normalize (bool, optional): If True, x is normalized to [0, 1]
                based on its data type. Defaults to True.
            to_pytorch (bool, optional): If True, x and y are converted to
                pytorch tensors. Defaults to True.
        """
        self.orig_dataset = orig_dataset
        self.normalize = normalize
        self.to_pytorch = to_pytorch

        tf_func = TF_TYPE_TO_TF_FUNC[transform_type]
        self.transform = lambda inp: tf_func(inp, transform)

        if transform_type == TransformType.object_detection:
            self.normalize = False
            self.to_pytorch = False

    def __getitem__(self, key) -> Tuple[torch.Tensor, torch.Tensor]:
        val = self.orig_dataset[key]
        x, y = self.transform(val)

        if self.normalize and np.issubdtype(x.dtype, np.unsignedinteger):
            max_val = np.iinfo(x.dtype).max
            x = x.astype(np.float32) / max_val

        if self.to_pytorch:
            # (H, W, C) --> (C, H, W)
            x = torch.from_numpy(x).permute(2, 0, 1).float()
            y = torch.from_numpy(y)

        return x, y

    def __len__(self):
        return len(self.orig_dataset)


class ImageDataset(AlbumentationsDataset):
    """ Dataset that reads from image files. """
    pass


class GeoDataset(AlbumentationsDataset):
    """ Dataset that reads directly from a Scene
        (i.e. a raster source and a label source).
    """

    def __init__(self,
                 scene: Scene,
                 transform: Optional[A.BasicTransform] = None,
                 transform_type: Optional[TransformType] = None,
                 normalize: bool = True,
                 to_pytorch: bool = True):
        """Constructor.

        Args:
            scene (Scene): A Scene object.
            transform (Optional[A.BasicTransform], optional): Albumentations
                transform to apply to the windows. Defaults to None.
            transform_type (Optional[TransformType], optional): Type of
                transform. Defaults to None.
            normalize (bool, optional): If True, x is normalized to [0, 1]
                based on its data type. Defaults to True.
            to_pytorch (bool, optional): If True, x and y are converted to
                pytorch tensors. Defaults to True.
        """
        self.scene = scene
        # Scene requires its raster and label sources to be activated before
        # they can be used. However, activating and deactivating for every read
        # could have a large overhead.
        #
        # Instead, we would like the source to stay activated for the entire
        # lifespan of this dataset. To achieve this, we set stay_activated=True
        # when activating the scene, so that once the sources are activated,
        # they will stay activated even when the context manager exits.
        with self.scene.activate(stay_activated=True):
            pass

        super().__init__(
            orig_dataset=scene,
            transform=transform,
            transform_type=transform_type,
            normalize=normalize,
            to_pytorch=to_pytorch)

    def __len__(self):
        raise NotImplementedError()


T = TypeVar('T')


def _to_tuple(x: T, n: int = 2) -> Tuple[T, T]:
    """Convert to n-tuple if not already an n-tuple."""
    if isinstance(x, tuple):
        if len(x) != n:
            raise ValueError()
        return x
    return tuple([x] * n)


class SlidingWindowGeoDataset(GeoDataset):
    """Read the scene left-to-right, top-to-bottom, using a sliding window.
    """

    def __init__(self,
                 scene: Scene,
                 size: Union[PosInt, Tuple[PosInt, PosInt]],
                 stride: Union[PosInt, Tuple[PosInt, PosInt]],
                 padding: Optional[Union[NonNegInt, Tuple[NonNegInt,
                                                          NonNegInt]]] = None,
                 transform: Optional[A.BasicTransform] = None,
                 transform_type: Optional[TransformType] = None):
        """Constructor.

        Args:
            scene (Scene): A Scene object.
            size (Union[PosInt, Tuple[PosInt, PosInt]]): Window size.
            stride (Union[PosInt, Tuple[PosInt, PosInt]]): Step size between
                windows.
            padding (Optional[Union[NonNegInt, Tuple[NonNegInt, NonNegInt]]]):
                How many pixels the windows are allowed to overflow the sides
                of the raster source. If None, padding = size.
                Defaults to None.
            transform (Optional[A.BasicTransform], optional): Albumentations
                transform to apply to the windows. Defaults to None.
            transform_type (Optional[TransformType], optional): Type of
                transform. Defaults to None.
        """
        super().__init__(scene, transform, transform_type)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)
        self.padding = padding
        self.init_windows()

    def init_windows(self) -> None:
        """Pre-compute windows."""
        windows = self.scene.raster_source.get_extent().get_windows(
            chip_sz=self.size, stride=self.stride, padding=self.padding)
        if len(self.scene.aoi_polygons) > 0:
            windows = Box.filter_by_aoi(windows, self.scene.aoi_polygons)
        self.windows = windows

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise StopIteration()
        window = self.windows[idx]
        return super().__getitem__(window)

    def __len__(self):
        return len(self.windows)


class RandomWindowGeoDataset(GeoDataset):
    """Read the scene by sampling random window sizes and locations.
    """

    def __init__(self,
                 scene: Scene,
                 size_lims: Optional[Tuple[PosInt, PosInt]] = None,
                 h_lims: Optional[Tuple[PosInt, PosInt]] = None,
                 w_lims: Optional[Tuple[PosInt, PosInt]] = None,
                 out_size: Union[PosInt, Tuple[PosInt, PosInt]] = None,
                 padding: Optional[Union[NonNegInt, Tuple[NonNegInt,
                                                          NonNegInt]]] = None,
                 max_windows: Optional[NonNegInt] = None,
                 transform: Optional[A.BasicTransform] = None,
                 transform_type: Optional[TransformType] = None,
                 max_sample_attempts: PosInt = 100,
                 return_window: bool = False):
        """Constructor.

        Will sample square windows if size_lims is specified. Otherwise, will
        sample rectangular windows with height and width sampled according to
        h_lims and w_lims.

        Args:
            scene (Scene): A Scene object.
            size_lims (Optional[Tuple[PosInt, PosInt]]): Interval from which to
                sample window size.
            h_lims (Optional[Tuple[PosInt, PosInt]]): Interval from which to
                sample window height.
            w_lims (Optional[Tuple[PosInt, PosInt]]): Interval from which to
                sample window width.
            out_size (Union[PosInt, Tuple[PosInt, PosInt]], optional): Resize
                windows to this size before returning. This is to aid in
                collating the windows into a batch. If None, windows are
                returned without being normalized or converted to pytorch, and
                will be of different sizes in successive reads.
                Defaults to None.
            padding (Optional[Union[NonNegInt, Tuple[NonNegInt, NonNegInt]]]):
                How many pixels the windows are allowed to overflow the sides
                of the raster source. If None, padding = size.
                Defaults to None.
            max_windows (Optional[NonNegInt]): Max allowed reads. Will raise
                StopIteration on further read attempts. If None, will be set to
                np.inf. Defaults to None.
            transform (Optional[A.BasicTransform], optional): Albumentations
                transform to apply to the windows. Defaults to None.
            transform_type (Optional[TransformType], optional): Type of
                transform. Defaults to None.
            return_window (bool, optional): Make __getitem__ return the window
                coordinates used to generate the image. Defaults to False.
            max_sample_attempts (NonNegInt, optional): Max attempts when trying
                to find a window within the AOI of the scene. Only used if the
                scene has aoi_polygons specified. StopIteratioin is raised if
                this is exceeded. Defaults to 100.
        """
        has_size_lims = size_lims is not None
        has_h_lims = h_lims is not None
        has_w_lims = w_lims is not None
        if has_size_lims == (has_w_lims or has_h_lims):
            raise ValueError('Specify either size_lims or h and w lims.')
        if has_h_lims != has_w_lims:
            raise ValueError('h_lims and w_lims must both be specified')

        if out_size is not None:
            normalize, to_pytorch = True, True
            out_size = _to_tuple(out_size)
            transform = self.get_resize_transform(transform, out_size)
        else:
            normalize, to_pytorch = False, False

        super().__init__(
            scene,
            transform,
            transform_type,
            normalize=normalize,
            to_pytorch=to_pytorch)

        if padding is None:
            if size_lims is not None:
                max_size = size_lims[1]
                padding = (max_size, max_size)
            else:
                max_h, max_w = h_lims[1], w_lims[1]
                padding = (max_h, max_w)
        padding = _to_tuple(padding)

        if max_windows is None:
            max_windows = np.iinfo('int').max

        self.size_lims = size_lims
        self.h_lims = h_lims
        self.w_lims = w_lims
        self.padding = padding
        self.return_window = return_window
        self.max_windows = max_windows
        self.max_sample_attempts = max_sample_attempts

        # include padding in the extent
        ymin, xmin, ymax, xmax = scene.raster_source.get_extent()
        h_padding, w_padding = self.padding
        self.extent = (ymin, xmin, ymax + h_padding, xmax + w_padding)

    def get_resize_transform(
            self, transform: Optional[A.BasicTransform],
            out_size: Tuple[PosInt, PosInt]) -> Union[A.Resize, A.Compose]:
        """Get transform to use for resizing windows to out_size."""
        resize_tf = A.Resize(*out_size, always_apply=True)
        if transform is None:
            transform = resize_tf
        else:
            transform = A.Compose([transform, resize_tf])
        return transform

    @property
    def min_size(self):
        if self.size_lims is not None:
            return self.size_lims[0], self.size_lims[0]
        return self.h_lims[0], self.w_lims[0]

    @property
    def max_size(self):
        if self.size_lims is not None:
            return self.size_lims[1], self.size_lims[1]
        return self.h_lims[1], self.w_lims[1]

    def sample_window_size(self) -> Tuple[int, int]:
        """Randomly sample the window size."""
        if self.size_lims is not None:
            hmin, hmax = self.size_lims
            size = torch.randint(low=hmin, high=hmax, size=(1, )).item()
            return size, size
        hmin, hmax = self.h_lims
        wmin, wmax = self.w_lims
        h = torch.randint(low=hmin, high=hmax, size=(1, )).item()
        w = torch.randint(low=wmin, high=wmax, size=(1, )).item()
        return h, w

    def sample_window_loc(self, h: int, w: int) -> Tuple[int, int]:
        """Randomly sample coordinates of the top left corner of the window."""
        ymin, xmin, ymax, xmax = self.extent
        y = torch.randint(low=ymin, high=ymax - h, size=(1, )).item()
        x = torch.randint(low=xmin, high=xmax - w, size=(1, )).item()
        return x, y

    def _sample_window(self) -> Box:
        """Randomly sample a window with random size and location."""
        h, w = self.sample_window_size()
        x, y = self.sample_window_loc(h, w)
        window = Box(y, x, y + h, x + w)
        return window

    def sample_window(self) -> Box:
        """If scene has AOI polygons, try to find a random window that is
        within the AOI. Otherwise, just return the first sampled window.

        Raises:
            StopIteration: If unable to find a valid window within
                self.max_sample_attempts attempts.

        Returns:
            Box: The sampled window.
        """
        if not self.scene.aoi_polygons:
            window = self._sample_window()
            return window

        for _ in range(self.max_sample_attempts):
            window = self._sample_window()
            if Box.within_aoi(window, self.scene.aoi_polygons):
                return window
        raise StopIteration('Failed to find random window within scene AOI.')

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise StopIteration()
        window = self.sample_window()
        if self.return_window:
            return (super().__getitem__(window), window)
        return super().__getitem__(window)

    def __len__(self):
        return self.max_windows
