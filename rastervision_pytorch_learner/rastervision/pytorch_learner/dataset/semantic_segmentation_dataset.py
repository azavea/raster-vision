from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from pathlib import Path
import logging

import numpy as np
from torch.utils.data import Dataset

from rastervision.pytorch_learner.dataset import (
    ImageDataset, TransformType, SlidingWindowGeoDataset,
    RandomWindowGeoDataset, load_image, discover_images, ImageDatasetError)
from rastervision.core.data.utils import make_ss_scene

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig

log = logging.getLogger(__name__)


class SemanticSegmentationDataReader(Dataset):
    """Reads semantic segmentatioin images and labels from files."""

    def __init__(self, img_dir: str, label_dir: str):
        """Constructor.

        Args:
            img_dir (str): Directory containing images.
            label_dir (str): Directory containing segmentation masks.
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)

        # collect image and label paths, match them based on filename
        img_paths = discover_images(img_dir)
        label_paths = discover_images(label_dir)
        self.img_paths = sorted(img_paths, key=lambda p: p.stem)
        self.label_paths = sorted(label_paths, key=lambda p: p.stem)
        self.validate_paths()

    def validate_paths(self) -> None:
        if len(self.img_paths) != len(self.label_paths):
            raise ImageDatasetError(
                'There should be a label file for every image file. '
                f'Found {len(self.img_paths)} image files and '
                f'{len(self.label_paths)} label files.')
        for img_path, label_path in zip(self.img_paths, self.label_paths):
            if img_path.stem != label_path.stem:
                raise ImageDatasetError(
                    f'Name mismatch between image file {img_path.stem} '
                    f'and label file {label_path.stem}.')

    def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.img_paths[ind]
        label_path = self.label_paths[ind]

        x = load_image(img_path)
        y = load_image(label_path).squeeze()

        return x, y

    def __len__(self):
        return len(self.img_paths)


class SemanticSegmentationImageDataset(ImageDataset):
    """Reads semantic segmentatioin images and labels from files.

    Uses :class:`.SemanticSegmentationDataReader` to read the data.
    """

    def __init__(self, img_dir: str, label_dir: str, *args, **kwargs):
        """Constructor.

        Args:
            img_dir (str): Directory containing images.
            label_dir (str): Directory containing segmentation masks.
            *args: See :meth:`.ImageDataset.__init__`.
            **kwargs: See :meth:`.ImageDataset.__init__`.
        """

        ds = SemanticSegmentationDataReader(img_dir, label_dir)
        super().__init__(
            ds,
            *args,
            **kwargs,
            transform_type=TransformType.semantic_segmentation)


def make_ss_geodataset(
        cls,
        image_uri: Union[str, List[str]],
        label_raster_uri: Optional[Union[str, List[str]]] = None,
        label_vector_uri: Optional[str] = None,
        class_config: Optional['ClassConfig'] = None,
        aoi_uri: Union[str, List[str]] = [],
        label_vector_default_class_id: Optional[int] = None,
        image_raster_source_kw: dict = {},
        label_raster_source_kw: dict = {},
        label_vector_source_kw: dict = {},
        **kwargs):
    """Create an instance of this class from image and label URIs.

    This is a convenience method. For more fine-grained control, it is
    recommended to use the default constructor.

    Args:
        image_uri (Union[str, List[str]]): URI or list of URIs of GeoTIFFs to
            use as the source of image data.
        label_raster_uri (Optional[Union[str, List[str]]], optional): URI or
            list of URIs of GeoTIFFs to use as the source of segmentation label
            data. If the labels are in the form of GeoJSONs, use
            label_vector_uri instead. Defaults to None.
        label_vector_uri (Optional[str], optional):  URI of GeoJSON file to use
            as the source of segmentation label data. If the labels are in the
            form of GeoTIFFs, use label_raster_uri instead. Defaults to None.
        class_config (Optional['ClassConfig']): The ClassConfig. Can be None if
            not using any labels.
        aoi_uri (Union[str, List[str]], optional): URI or list of URIs of
            GeoJSONs that specify the area-of-interest. If provided, the
            dataset will only access data from this area. Defaults to [].
        label_vector_default_class_id (Optional[int], optional): If using
            label_vector_uri and all polygons in that file belong to the same
            class and they do not contain a `class_id` property, then use this
            argument to map all of the polygons to the appropriate class ID.
            See docs for ClassInferenceTransformer for more details.
            Defaults to None.
        image_raster_source_kw (dict, optional): Additional arguments to pass
            to the RasterioSource used for image data. See docs for
            RasterioSource for more details. Defaults to {}.
        label_raster_source_kw (dict, optional): Additional arguments to pass
            to the RasterioSource used for label data, if label_raster_uri is
            used. See docs for RasterioSource for more details. Defaults to {}.
        label_vector_source_kw (dict, optional): Additional arguments to pass
            to the GeoJSONVectorSource used for label data, if label_vector_uri
            is used. See docs for GeoJSONVectorSource for more details.
            Defaults to {}.
        **kwargs: All other keyword args are passed to the default constructor
            for this class.

    Raises:
        ValueError: If both label_raster_uri and label_vector_uri are
            specified.

    Returns:
        An instance of this GeoDataset subclass.
    """
    scene = make_ss_scene(
        image_uri=image_uri,
        label_raster_uri=label_raster_uri,
        label_vector_uri=label_vector_uri,
        class_config=class_config,
        aoi_uri=aoi_uri,
        label_vector_default_class_id=label_vector_default_class_id,
        image_raster_source_kw=image_raster_source_kw,
        label_raster_source_kw=label_raster_source_kw,
        label_vector_source_kw=label_vector_source_kw)
    ds = cls(scene, **kwargs)
    return ds


class SemanticSegmentationSlidingWindowGeoDataset(SlidingWindowGeoDataset):
    from_uris = classmethod(make_ss_geodataset)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            transform_type=TransformType.semantic_segmentation)


class SemanticSegmentationRandomWindowGeoDataset(RandomWindowGeoDataset):
    from_uris = classmethod(make_ss_geodataset)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            transform_type=TransformType.semantic_segmentation)
