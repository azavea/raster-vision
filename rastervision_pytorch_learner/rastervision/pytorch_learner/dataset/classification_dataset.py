from typing import TYPE_CHECKING, Iterable, List, Optional, Union
import logging

from rastervision.pytorch_learner.dataset import (
    ImageDataset, TransformType, SlidingWindowGeoDataset,
    RandomWindowGeoDataset, make_image_folder_dataset)
from rastervision.core.data.utils import make_cc_scene

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig

log = logging.getLogger(__name__)


class ClassificationImageDataset(ImageDataset):
    def __init__(self, data_dir: str, class_names: Iterable[str], *args,
                 **kwargs):
        ds = make_image_folder_dataset(data_dir, classes=class_names)
        super().__init__(
            ds, *args, **kwargs, transform_type=TransformType.classification)


def make_cc_geodataset(cls,
                       class_config: 'ClassConfig',
                       image_uri: Union[str, List[str]],
                       label_vector_uri: Optional[str] = None,
                       aoi_uri: Union[str, List[str]] = [],
                       label_vector_default_class_id: Optional[int] = None,
                       image_raster_source_kw: dict = {},
                       label_vector_source_kw: dict = {},
                       label_source_kw: dict = {},
                       **kwargs):
    """Create an instance of this class from image and label URIs.

    This is a convenience method. For more fine-grained control, it is
    recommended to use the default constructor.

    Args:
        class_config (ClassConfig): The ClassConfig.
        image_uri (Union[str, List[str]]): URI or list of URIs of GeoTIFFs to
            use as the source of image data.
        label_vector_uri (Optional[str], optional):  URI of GeoJSON file to use
            as the source of segmentation label data. Defaults to None.
        aoi_uri (Union[str, List[str]], optional): URI or list of URIs of
            GeoJSONs that specify the area-of-interest. If provided, the
            dataset will only access data from this area. Defaults to [].
        label_vector_default_class_id (Optional[int], optional): If using
            label_vector_uri and all polygons in that file belong to the same
            class and they do not contain a `class_id` property, then use this
            argument to map all of the polgons to the appropriate class ID.
            See docs for ClassInferenceTransformer for more details.
            Defaults to None.
        image_raster_source_kw (dict, optional): Additional arguments to pass
            to the RasterioSource used for image data. See docs for
            RasterioSource for more details. Defaults to {}.
        label_vector_source_kw (dict, optional): Additional arguments to pass
            to the GeoJSONVectorSourceConfig used for label data, if
            label_vector_uri is used. See docs for GeoJSONVectorSourceConfig
            for more details. Defaults to {}.
        label_source_kw (dict, optional): Additional arguments to pass
            to the ChipClassificationLabelSourceConfig used for label data, if
            label_vector_uri is used. See docs for
            ChipClassificationLabelSourceConfig for more details.
            Defaults to {}.
        **kwargs: All other keyword args are passed to the default constructor
            for this class.

    Returns:
        An instance of this GeoDataset subclass.
    """
    scene = make_cc_scene(
        class_config=class_config,
        image_uri=image_uri,
        label_vector_uri=label_vector_uri,
        aoi_uri=aoi_uri,
        label_vector_default_class_id=label_vector_default_class_id,
        image_raster_source_kw=image_raster_source_kw,
        label_vector_source_kw=label_vector_source_kw,
        label_source_kw=label_source_kw)
    ds = cls(scene, **kwargs)
    return ds


class ClassificationSlidingWindowGeoDataset(SlidingWindowGeoDataset):
    from_uris = classmethod(make_cc_geodataset)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.classification)

    def init_windows(self):
        super().init_windows()
        if self.scene.label_source is not None:
            self.scene.label_source.populate_labels(cells=self.windows)


class ClassificationRandomWindowGeoDataset(RandomWindowGeoDataset):
    from_uris = classmethod(make_cc_geodataset)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.classification)
