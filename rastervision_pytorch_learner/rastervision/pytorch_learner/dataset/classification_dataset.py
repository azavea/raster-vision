from typing import TYPE_CHECKING, Iterable
import logging

from rastervision.pytorch_learner.dataset import (
    ImageDataset, TransformType, SlidingWindowGeoDataset,
    RandomWindowGeoDataset, make_image_folder_dataset)
from rastervision.core.data.utils import make_cc_scene

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig, ChipClassificationLabelSource

log = logging.getLogger(__name__)


class ClassificationImageDataset(ImageDataset):
    """Read images and class labels from images stored in class folders.

    I.e., all images for a class "A" are stored in directory ``A/``, all images
    for a class "B" are stored in directory ``B/``, and so on. And all class
    directories are located in the same parent directory.
    """

    def __init__(self, data_dir: str, class_names: Iterable[str] | None, *args,
                 **kwargs):
        """Constructor.

        Args:
            data_dir: Root directory containing class dirs.
            class_names: Class names. Should match class dir names.
            *args: See :meth:`.ImageDataset.__init__`.
            **kwargs: See :meth:`.ImageDataset.__init__`.
        """
        ds = make_image_folder_dataset(data_dir, classes=class_names)
        super().__init__(
            ds, *args, **kwargs, transform_type=TransformType.classification)


def make_cc_geodataset(cls,
                       image_uri: str | list[str],
                       label_vector_uri: str | None = None,
                       class_config: 'ClassConfig | None' = None,
                       aoi_uri: str | list[str] = [],
                       label_vector_default_class_id: int | None = None,
                       image_raster_source_kw: dict = {},
                       label_vector_source_kw: dict = {},
                       label_source_kw: dict = {},
                       **kwargs):
    """Create an instance of this class from image and label URIs.

    This is a convenience method. For more fine-grained control, it is
    recommended to use the default constructor.

    Args:
        class_config: The ``ClassConfig``.
        image_uri: URI or list of URIs of GeoTIFFs to use as the source of
            image data.
        label_vector_uri:  URI of GeoJSON file to use as the source of
            segmentation label data. Defaults to ``None``.
        class_config: The ``ClassConfig``. Can be ``None`` if not using any
            labels.
        aoi_uri: URI or list of URIs of GeoJSONs that specify the
            area-of-interest. If provided, the dataset will only access data
            from this area. Defaults to ``[]``.
        label_vector_default_class_id: If using ``label_vector_uri`` and all
            polygons in that file belong to the same class and they do not
            contain a `class_id` property, then use this argument to map all of
            the polygons to the appropriate class ID. See docs for
            :class:`.ClassInferenceTransformer` for more details.
            Defaults to ``None``.
        image_raster_source_kw: Additional arguments to pass
            to the RasterioSource used for image data. See docs for
            RasterioSource for more details. Defaults to ``{}``.
        label_vector_source_kw: Additional arguments to pass to the
            :class:`.GeoJSONVectorSourceConfig` used for label data, if
            ``label_vector_uri`` is set. See docs for
            :class:`.GeoJSONVectorSourceConfig` for more details.
            Defaults to ``{}``.
        label_source_kw: Additional arguments to pass
            to the :class:`.ChipClassificationLabelSourceConfig` used for
            label data, if ``label_vector_uri`` is set. See docs for
            :class:`.ChipClassificationLabelSourceConfig` for more details.
            Defaults to ``{}``.
        **kwargs: All other keyword args are passed to the default constructor
            for this class.

    Returns:
        An instance of this GeoDataset subclass.
    """
    scene = make_cc_scene(
        image_uri=image_uri,
        label_vector_uri=label_vector_uri,
        class_config=class_config,
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
            ls: 'ChipClassificationLabelSource' = self.scene.label_source
            ls.populate_labels(cells=self.windows)


class ClassificationRandomWindowGeoDataset(RandomWindowGeoDataset):
    from_uris = classmethod(make_cc_geodataset)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.classification)
