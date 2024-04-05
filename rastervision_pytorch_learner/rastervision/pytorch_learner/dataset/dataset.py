from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, TypeVar, Union
import logging

import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset
from shapely.ops import unary_union

from rastervision.core.box import Box
from rastervision.core.data import Scene
from rastervision.core.data.utils import AoiSampler
from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt
from rastervision.pytorch_learner.dataset.transform import (TransformType,
                                                            TF_TYPE_TO_TF_FUNC)

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon

log = logging.getLogger(__name__)

T = TypeVar('T')


def _to_tuple(x: T, n: int = 2) -> Tuple[T, ...]:
    """Convert to n-tuple if not already an n-tuple."""
    if isinstance(x, tuple):
        if len(x) != n:
            raise ValueError()
        return x
    return tuple([x] * n)


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
            transform (A.BasicTransform, optional): Albumentations
                transform to apply to the windows. Defaults to None.
                Each transform in Albumentations takes images of type uint8, and
                sometimes other data types. The data type requirements can be
                seen at https://albumentations.ai/docs/api_reference/augmentations/transforms/ # noqa
                If there is a mismatch between the data type of imagery and the
                transform requirements, a RasterTransformer should be set
                on the RasterSource that converts to uint8, such as
                MinMaxTransformer or StatsTransformer.
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
        self.transform_type = transform_type

        tf_func = TF_TYPE_TO_TF_FUNC[transform_type]
        self.transform = lambda inp: tf_func(inp, transform)

        if transform_type == TransformType.object_detection:
            self.normalize = False
            self.to_pytorch = False

    def __getitem__(self, key) -> Tuple[torch.Tensor, torch.Tensor]:
        val = self.orig_dataset[key]

        try:
            x, y = self.transform(val)
        except Exception as exc:
            log.warning(
                'Many albumentations transforms require uint8 input. Therefore, we '
                'recommend passing a MinMaxTransformer or StatsTransformer to the '
                'RasterSource so the input will be converted to uint8.')
            raise exc

        if self.normalize and np.issubdtype(x.dtype, np.unsignedinteger):
            max_val = np.iinfo(x.dtype).max
            x = x.astype(float) / max_val

        if self.to_pytorch:
            x = torch.from_numpy(x).float()
            # (..., H, W, C) --> (..., C, H, W)
            x = x.transpose_(-2, -1).transpose_(-3, -2)
            if y is not None:
                y = torch.from_numpy(y)

        if y is None:
            # Ideally, y should be None to semantically convey the absence of
            # any label, but PyTorch's default collate function doesn't handle
            # None values.
            y = torch.tensor(np.nan)

        return x, y

    def __len__(self):
        return len(self.orig_dataset)


class ImageDataset(AlbumentationsDataset):
    """ Dataset that reads from image files. """


class GeoDataset(AlbumentationsDataset):
    """ Dataset that reads directly from a Scene
        (i.e. a raster source and a label source).
    """

    def __init__(
            self,
            scene: Scene,
            out_size: Optional[Union[PosInt, Tuple[PosInt, PosInt]]] = None,
            within_aoi: bool = True,
            transform: Optional[A.BasicTransform] = None,
            transform_type: Optional[TransformType] = None,
            normalize: bool = True,
            to_pytorch: bool = True,
            return_window: bool = False):
        """Constructor.

        Args:
            scene (Scene): A Scene object.
            out_size: Resize chips to this size before returning.
            within_aoi: If True and if the scene has an AOI, only sample
                windows that lie fully within the AOI. If False, windows only
                partially intersecting the AOI will also be allowed.
                Defaults to True.
            transform (Optional[A.BasicTransform], optional): Albumentations
                transform to apply to the windows. Defaults to None.
                Each transform in Albumentations takes images of type uint8, and
                sometimes other data types. The data type requirements can be
                seen at https://albumentations.ai/docs/api_reference/augmentations/transforms/ # noqa
                If there is a mismatch between the data type of imagery and the
                transform requirements, a RasterTransformer should be set
                on the RasterSource that converts to uint8, such as
                MinMaxTransformer or StatsTransformer.
            transform_type (Optional[TransformType], optional): Type of
                transform. Defaults to None.
            normalize (bool, optional): If True, x is normalized to [0, 1]
                based on its data type. Defaults to True.
            to_pytorch (bool, optional): If True, x and y are converted to
                pytorch tensors. Defaults to True.
            return_window (bool, optional): Make __getitem__ return the window
                coordinates used to generate the image. Defaults to False.
        """
        self.scene = scene
        self.within_aoi = within_aoi
        self.return_window = return_window
        self.out_size = None

        if out_size is not None:
            self.out_size = _to_tuple(out_size)
            transform = self.append_resize_transform(transform, self.out_size)

        super().__init__(
            orig_dataset=scene,
            transform=transform,
            transform_type=transform_type,
            normalize=normalize,
            to_pytorch=to_pytorch)

    def append_resize_transform(
            self, transform: A.BasicTransform | None,
            out_size: tuple[PosInt, PosInt]) -> A.Resize | A.Compose:
        """Get transform to use for resizing windows to out_size."""
        resize_tf = A.Resize(*out_size, always_apply=True)
        if transform is None:
            transform = resize_tf
        else:
            transform = A.Compose([transform, resize_tf])
        return transform

    def __len__(self):
        raise NotImplementedError()

    @classmethod
    def from_uris(cls, *args, **kwargs) -> 'GeoDataset':
        raise NotImplementedError()


class SlidingWindowGeoDataset(GeoDataset):
    """Read the scene left-to-right, top-to-bottom, using a sliding window.
    """

    def __init__(
            self,
            scene: Scene,
            size: Union[PosInt, Tuple[PosInt, PosInt]],
            stride: Union[PosInt, Tuple[PosInt, PosInt]],
            out_size: Optional[Union[PosInt, Tuple[PosInt, PosInt]]] = None,
            padding: Optional[Union[NonNegInt, Tuple[NonNegInt,
                                                     NonNegInt]]] = None,
            pad_direction: Literal['both', 'start', 'end'] = 'end',
            within_aoi: bool = True,
            transform: Optional[A.BasicTransform] = None,
            transform_type: Optional[TransformType] = None,
            normalize: bool = True,
            to_pytorch: bool = True,
            return_window: bool = False):
        """Constructor.

        Args:
            scene (Scene): A Scene object.
            size (Union[PosInt, Tuple[PosInt, PosInt]]): Window size.
            stride (Union[PosInt, Tuple[PosInt, PosInt]]): Step size between
                windows.
            out_size: Resize chips to this size before returning. Defaults to
                ``None``.
            padding (Optional[Union[NonNegInt, Tuple[NonNegInt, NonNegInt]]]):
                How many pixels the windows are allowed to overflow the sides
                of the raster source. If None, padding is set to size // 2.
                Defaults to None.
            pad_direction (Literal['both', 'start', 'end']): If 'end', only pad
                ymax and xmax (bottom and right). If 'start', only pad ymin and
                xmin (top and left). If 'both', pad all sides. Has no effect if
                paddiong is zero. Defaults to 'end'.
            within_aoi: If True and if the scene has an AOI, only sample
                windows that lie fully within the AOI. If False, windows only
                partially intersecting the AOI will also be allowed.
                Defaults to True.
            transform (Optional[A.BasicTransform], optional): Albumentations
                transform to apply to the windows. Defaults to None.
                Each transform in Albumentations takes images of type uint8, and
                sometimes other data types. The data type requirements can be
                seen at https://albumentations.ai/docs/api_reference/augmentations/transforms/ # noqa
                If there is a mismatch between the data type of imagery and the
                transform requirements, a RasterTransformer should be set
                on the RasterSource that converts to uint8, such as
                MinMaxTransformer or StatsTransformer.
            transform_type (Optional[TransformType], optional): Type of
                transform. Defaults to None.
            normalize (bool, optional): If True, x is normalized to [0, 1]
                based on its data type. Defaults to True.
            to_pytorch (bool, optional): If True, x and y are converted to
                pytorch tensors. Defaults to True.
            return_window (bool, optional): Make __getitem__ return the window
                coordinates used to generate the image. Defaults to False.
        """
        super().__init__(
            scene=scene,
            out_size=out_size,
            within_aoi=within_aoi,
            transform=transform,
            transform_type=transform_type,
            normalize=normalize,
            to_pytorch=to_pytorch,
            return_window=return_window)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)
        self.padding = padding
        self.pad_direction = pad_direction
        self.init_windows()

    def init_windows(self) -> None:
        """Pre-compute windows."""
        windows = self.scene.extent.get_windows(
            self.size,
            stride=self.stride,
            padding=self.padding,
            pad_direction=self.pad_direction)
        if len(self.scene.aoi_polygons_bbox_coords) > 0:
            windows = Box.filter_by_aoi(
                windows,
                self.scene.aoi_polygons_bbox_coords,
                within=self.within_aoi)
        self.windows = windows

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise StopIteration()
        window = self.windows[idx]
        out = super().__getitem__(window)
        if self.return_window:
            return (out, window)
        return out

    def __len__(self):
        return len(self.windows)


class RandomWindowGeoDataset(GeoDataset):
    """Read the scene by sampling random window sizes and locations.
    """

    def __init__(self,
                 scene: Scene,
                 out_size: Optional[Union[PosInt, Tuple[PosInt, PosInt]]],
                 size_lims: Optional[Tuple[PosInt, PosInt]] = None,
                 h_lims: Optional[Tuple[PosInt, PosInt]] = None,
                 w_lims: Optional[Tuple[PosInt, PosInt]] = None,
                 padding: Optional[Union[NonNegInt, Tuple[NonNegInt,
                                                          NonNegInt]]] = None,
                 max_windows: Optional[NonNegInt] = None,
                 max_sample_attempts: PosInt = 100,
                 efficient_aoi_sampling: bool = True,
                 within_aoi: bool = True,
                 transform: Optional[A.BasicTransform] = None,
                 transform_type: Optional[TransformType] = None,
                 normalize: bool = True,
                 to_pytorch: bool = True,
                 return_window: bool = False):
        """Constructor.

        Will sample square windows if size_lims is specified. Otherwise, will
        sample rectangular windows with height and width sampled according to
        h_lims and w_lims.

        Args:
            scene (Scene): A Scene object.
            out_size (Optional[Union[PosInt, Tuple[PosInt, PosInt]]]]): Resize
                windows to this size before returning. This is to aid in
                collating the windows into a batch. If None, windows are
                returned without being normalized or converted to pytorch, and
                will be of different sizes in successive reads.
            size_lims (Optional[Tuple[PosInt, PosInt]]): Interval from which to
                sample window size.
            h_lims (Optional[Tuple[PosInt, PosInt]]): Interval from which to
                sample window height.
            w_lims (Optional[Tuple[PosInt, PosInt]]): Interval from which to
                sample window width.
            padding (Optional[Union[NonNegInt, Tuple[NonNegInt, NonNegInt]]]):
                How many pixels the windows are allowed to overflow the sides
                of the raster source. If None, padding = size.
                Defaults to None.
            max_windows (Optional[NonNegInt]): Max allowed reads. Will raise
                StopIteration on further read attempts. If None, will be set to
                np.inf. Defaults to None.
            transform (Optional[A.BasicTransform], optional): Albumentations
                transform to apply to the windows. Defaults to None.
                Each transform in Albumentations takes images of type uint8, and
                sometimes other data types. The data type requirements can be
                seen at https://albumentations.ai/docs/api_reference/augmentations/transforms/
                If there is a mismatch between the data type of imagery and the
                transform requirements, a RasterTransformer should be set
                on the RasterSource that converts to uint8, such as
                MinMaxTransformer or StatsTransformer.
            transform_type (Optional[TransformType], optional): Type of
                transform. Defaults to None.
            max_sample_attempts (NonNegInt, optional): Max attempts when trying
                to find a window within the AOI of the scene. Only used if the
                scene has aoi_polygons specified. StopIteratioin is raised if
                this is exceeded. Defaults to 100.
            efficient_aoi_sampling (bool, optional): If the scene has AOIs,
                sampling windows at random anywhere in the extent and then
                checking if they fall within any of the AOIs can be very
                inefficient. This flag enables the use of an alternate
                algorithm that only samples window locations inside the AOIs.
                Defaults to True.
            within_aoi: If True and if the scene has an AOI, only sample
                windows that lie fully within the AOI. If False, windows only
                partially intersecting the AOI will also be allowed.
                Defaults to True.
            transform (Optional[A.BasicTransform], optional): Albumentations
                transform to apply to the windows. Defaults to None.
            transform_type (Optional[TransformType], optional): Type of
                transform. Defaults to None.
            normalize (bool, optional): If True, x is normalized to [0, 1]
                based on its data type. Defaults to True.
            to_pytorch (bool, optional): If True, x and y are converted to
                pytorch tensors. Defaults to True.
            return_window (bool, optional): Make __getitem__ return the window
                coordinates used to generate the image. Defaults to False.
        """ # noqa
        has_size_lims = size_lims is not None
        has_h_lims = h_lims is not None
        has_w_lims = w_lims is not None
        if has_size_lims == (has_w_lims or has_h_lims):
            raise ValueError('Specify either size_lims or h and w lims.')
        if has_h_lims != has_w_lims:
            raise ValueError('h_lims and w_lims must both be specified')

        if out_size is None:
            log.warning(f'out_size is None, chips will not be normalized or '
                        'converted to PyTorch Tensors.')
            normalize, to_pytorch = False, False

        super().__init__(
            scene=scene,
            out_size=out_size,
            within_aoi=within_aoi,
            transform=transform,
            transform_type=transform_type,
            normalize=normalize,
            to_pytorch=to_pytorch,
            return_window=return_window)

        if padding is None:
            if has_size_lims:
                max_size = size_lims[1]
                padding = (max_size // 2, max_size // 2)
            else:
                max_h, max_w = h_lims[1], w_lims[1]
                padding = (max_h // 2, max_w // 2)
        padding = _to_tuple(padding)

        if max_windows is None:
            max_windows = np.iinfo('int').max

        self.size_lims = size_lims
        self.h_lims = h_lims
        self.w_lims = w_lims
        self.padding = padding
        self.max_windows = max_windows
        self.max_sample_attempts = max_sample_attempts

        # include padding in the extent
        ymin, xmin, ymax, xmax = scene.extent
        h_padding, w_padding = self.padding
        self.extent = Box(ymin - h_padding, xmin - w_padding, ymax + h_padding,
                          xmax + w_padding)

        self.aoi_sampler = None
        aoi_polygons = self.scene.aoi_polygons_bbox_coords
        self.has_aoi_polygons = len(aoi_polygons) > 0
        if self.has_aoi_polygons:
            extent_polygon = self.extent.to_shapely()
            aoi: 'Polygon' | 'MultiPolygon' = unary_union(aoi_polygons)
            # only sample from polygons that intersect w/ the extent
            self.aoi = aoi.intersection(extent_polygon)
            if efficient_aoi_sampling:
                try:
                    self.aoi_sampler = AoiSampler([self.aoi])
                except ModuleNotFoundError:
                    log.info('Ignoring efficient_aoi_sampling since triangle '
                             'is not installed.')

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
            sz_min, sz_max = self.size_lims
            if sz_max == sz_min + 1:
                return sz_min, sz_min
            size = torch.randint(low=sz_min, high=sz_max, size=(1, )).item()
            return size, size
        hmin, hmax = self.h_lims
        wmin, wmax = self.w_lims
        h = torch.randint(low=hmin, high=hmax, size=(1, )).item()
        w = torch.randint(low=wmin, high=wmax, size=(1, )).item()
        return h, w

    def sample_window_loc(self, h: int, w: int) -> Tuple[int, int]:
        """Randomly sample coordinates of the top left corner of the window."""
        if not self.aoi_sampler:
            ymin, xmin, ymax, xmax = self.extent
            y = torch.randint(low=ymin, high=ymax - h, size=(1, )).item()
            x = torch.randint(low=xmin, high=xmax - w, size=(1, )).item()
        else:
            x, y = self.aoi_sampler.sample().round().T
            x, y = int(x.item()), int(y.item())
        return x, y

    def _sample_window(self) -> Box:
        """Randomly sample a window with random size and location."""
        h, w = self.sample_window_size()
        x, y = self.sample_window_loc(h, w)
        window = Box(y, x, y + h, x + w)
        return window

    def sample_window(self) -> Box:
        """Sample a window with random size and location within the AOI.

        If the scene has AOI polygons, try to find a random window that is
        within the AOI. Otherwise, just return the first sampled window.

        Raises:
            StopIteration: If unable to find a valid window within
                self.max_sample_attempts attempts.

        Returns:
            Box: The sampled window.
        """
        if not self.has_aoi_polygons:
            window = self._sample_window()
            return window

        for _ in range(self.max_sample_attempts):
            window = self._sample_window()
            if self.within_aoi:
                if Box.within_aoi(window, self.aoi):
                    return window
            else:
                if Box.intersects_aoi(window, self.aoi):
                    return window
        raise StopIteration('Failed to find valid window within scene AOI in '
                            f'{self.max_sample_attempts} attempts.')

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise StopIteration()
        window = self.sample_window()
        out = super().__getitem__(window)
        if self.return_window:
            return (out, window)
        return out

    def __len__(self):
        return self.max_windows
