from typing import TYPE_CHECKING, Optional, Tuple, List, Dict, Union
from os.path import join
from collections import defaultdict
import logging

import albumentations as A
import numpy as np
from torch.utils.data import Dataset

from rastervision.pipeline.file_system import file_to_json
from rastervision.core.box import Box
from rastervision.core.data import ObjectDetectionLabels
from rastervision.pytorch_learner.dataset import (
    TransformType, ImageDataset, SlidingWindowGeoDataset,
    RandomWindowGeoDataset, load_image)
from rastervision.core.data.utils import make_od_scene

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig, ObjectDetectionLabelSource
log = logging.getLogger(__name__)


class CocoDataset(Dataset):
    """Read Object Detection data in the COCO format."""

    def __init__(self, img_dir: str, annotation_uri: str):
        """Constructor.

        Args:
            img_dir (str): Directory containing the images. Image filenames
                must match the image IDs in the annotations file.
            annotation_uri (str): URI to a JSON file containing annotations in
                the COCO format.
        """
        self.annotation_uri = annotation_uri
        ann_json = file_to_json(annotation_uri)

        self.img_ids: List[str] = [img['id'] for img in ann_json['images']]
        self.img_paths = {
            img['id']: join(img_dir, img['file_name'])
            for img in ann_json['images']
        }
        self.img_anns = {id: defaultdict(list) for id in self.img_ids}
        for ann in ann_json['annotations']:
            img_ann = self.img_anns[ann['image_id']]
            img_ann['bboxes'].append(ann['bbox'])
            img_ann['category_id'].append(ann['category_id'])

    def __getitem__(self, ind: int
                    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, str]]:
        img_id = self.img_ids[ind]
        path = self.img_paths[img_id]
        ann: Dict[str, list] = self.img_anns[img_id]

        x = load_image(path)
        bboxes = np.array(ann['bboxes'])
        class_ids = np.array(ann['category_id'], dtype=np.int64)

        if len(bboxes) == 0:
            bboxes = np.empty((0, 4))
            class_ids = np.empty((0, ), dtype=np.int64)
        return x, (bboxes, class_ids, 'xywh')

    def __len__(self):
        return len(self.img_anns)


class ObjectDetectionImageDataset(ImageDataset):
    """Read Object Detection data in the COCO format.

    Uses :class:`.CocoDataset` to read the data.
    """

    def __init__(self, img_dir: str, annotation_uri: str, *args, **kwargs):
        """Constructor.

        Args:
            img_dir (str): Directory containing the images. Image filenames
                must match the image IDs in the annotations file.
            annotation_uri (str): URI to a JSON file containing annotations in
                the COCO format.
            *args: See :meth:`.ImageDataset.__init__`.
            **kwargs: See :meth:`.ImageDataset.__init__`.
        """
        ds = CocoDataset(img_dir, annotation_uri)
        super().__init__(
            ds, *args, **kwargs, transform_type=TransformType.object_detection)


def make_od_geodataset(cls,
                       image_uri: Union[str, List[str]],
                       label_vector_uri: Optional[str] = None,
                       class_config: Optional['ClassConfig'] = None,
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
        image_uri (Union[str, List[str]]): URI or list of URIs of GeoTIFFs to
            use as the source of image data.
        label_vector_uri (Optional[str], optional):  URI of GeoJSON file to use
            as the source of segmentation label data. Defaults to None.
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
        label_vector_source_kw (dict, optional): Additional arguments to pass
            to the GeoJSONVectorSourceConfig used for label data, if
            label_vector_uri is set. See docs for GeoJSONVectorSourceConfig
            for more details. Defaults to {}.
        label_source_kw (dict, optional): Additional arguments to pass
            to the ObjectDetectionLabelSourceConfig used for label data, if
            label_vector_uri is set. See docs for
            ObjectDetectionLabelSourceConfig for more details.
            Defaults to {}.
        **kwargs: All other keyword args are passed to the default constructor
            for this class.

    Returns:
        An instance of this GeoDataset subclass.
    """
    scene = make_od_scene(
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


class ObjectDetectionSlidingWindowGeoDataset(SlidingWindowGeoDataset):
    from_uris = classmethod(make_od_geodataset)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.object_detection)


class ObjectDetectionRandomWindowGeoDataset(RandomWindowGeoDataset):
    from_uris = classmethod(make_od_geodataset)

    def __init__(self, *args, **kwargs):
        """Constructor.

        Args:
            *args: See :meth:`.RandomWindowGeoDataset.__init__`.

        Keyword Args:
            bbox_params (Optional[A.BboxParams], optional): Optional
                bbox_params to use when resizing windows. Defaults to None.
            ioa_thresh (float, optional): Minimum IoA of a bounding box with a
                given window for it to be included in the labels for that
                window. Defaults to 0.9.
            clip (bool, optional): Clip bounding boxes to window limits when
                retrieving labels for a window. Defaults to False.
            neg_ratio (Optional[float], optional): Ratio of sampling
                probabilities of negative windows (windows w/o bboxes) vs
                positive windows (windows w/ at least 1 bbox). E.g. neg_ratio=2
                means 2/3 probability of sampling a negative window.
                If None, the default sampling behavior of
                RandomWindowGeoDataset is used, without taking bboxes into
                account. Defaults to None.
            neg_ioa_thresh (float, optional): A window will be considered
                negative if its max IoA with any bounding box is less than this
                threshold. Defaults to 0.2.
            **kwargs: See :meth:`.RandomWindowGeoDataset.__init__`.
        """
        from rastervision.pytorch_learner import DEFAULT_BBOX_PARAMS
        self.bbox_params: Optional[A.BboxParams] = kwargs.pop(
            'bbox_params', DEFAULT_BBOX_PARAMS)
        ioa_thresh: float = kwargs.pop('ioa_thresh', 0.9)
        clip: bool = kwargs.pop('clip', False)
        neg_ratio: Optional[float] = kwargs.pop('neg_ratio', None)
        neg_ioa_thresh: float = kwargs.pop('neg_ioa_thresh', 0.2)

        super().__init__(
            *args, **kwargs, transform_type=TransformType.object_detection)

        label_source: Optional[
            'ObjectDetectionLabelSource'] = self.scene.label_source
        if label_source is not None:
            label_source.ioa_thresh = ioa_thresh
            label_source.clip = clip

        if neg_ratio is not None:
            if label_source is None:
                raise ValueError(
                    'Scene must have a LabelSource if neg_ratio is set.')
            self.neg_probability = neg_ratio / (neg_ratio + 1)
            self.neg_ioa_thresh: float = neg_ioa_thresh

            # Get labels for the AOI. clip=True here to ensure that it is
            # possible to draw a window (that lies within the extent) around
            # each bbox.
            self.labels = label_source.get_labels(
                ioa_thresh=ioa_thresh, clip=True)
            num_bboxes_in_scene = len(self.labels)
            if num_bboxes_in_scene == 0:
                raise ValueError(
                    'neg_ratio specified, but no bboxes found in scene.')

            if self.has_aoi_polygons:
                self.labels = self.labels.filter_by_aoi(
                    self.scene.aoi_polygons)
                num_bboxes_in_aoi = len(self.labels)
                if num_bboxes_in_aoi == 0:
                    raise ValueError(
                        'neg_ratio specified, but no bboxes found in AOI. '
                        'Total bboxes in scene (ignoring AOI):'
                        f'{num_bboxes_in_scene}.')

            self.bboxes = self.labels.get_boxes()
        else:
            self.neg_probability = None

    def append_resize_transform(self, transform: A.BasicTransform,
                                out_size: tuple[int, int]) -> A.BasicTransform:
        resize_tf = A.Resize(*out_size, always_apply=True)
        if transform is None:
            transform = resize_tf
        else:
            transform = A.Compose(
                [transform, resize_tf], bbox_params=self.bbox_params)
        return transform

    def _sample_pos_window(self) -> Box:
        """Sample a window containing at least one bounding box.

        This is done by randomly sampling one of the bounding boxes in the
        scene and drawing a random window around it.
        """
        bbox: Box = np.random.choice(self.bboxes)
        box_h, box_w = bbox.size

        # check if it is possible to sample a containing window
        hmax, wmax = self.max_size
        if box_h > hmax or box_w > wmax:
            raise ValueError(
                f'Cannot sample containing window because bounding box {bbox}'
                f'is larger than self.max_size ({self.max_size}).')

        # try to sample a window size that is larger than the box's size
        for _ in range(self.max_sample_attempts):
            h, w = self.sample_window_size()
            if h >= box_h and w >= box_w:
                window = bbox.make_random_box_container(h, w)
                return window
        log.warning('ObjectDetectionRandomWindowGeoDataset: Failed to find '
                    'suitable (h, w) for positive window. '
                    f'Using (hmax, wmax) = ({hmax}, {wmax}) instead.')
        window = bbox.make_random_box_container(hmax, wmax)
        return window

    def _sample_neg_window(self) -> Box:
        """Attempt to sample a window containing no bounding boxes.

        If not found within self.max_sample_attempts, just return the last
        sampled window.
        """
        for _ in range(self.max_sample_attempts):
            window = super()._sample_window()
            labels = ObjectDetectionLabels.get_overlapping(
                self.labels, window, ioa_thresh=self.neg_ioa_thresh)
            if len(labels) == 0:
                return window

        log.warning('ObjectDetectionRandomWindowGeoDataset: Failed to find '
                    'negative window. Returning last sampled window.')
        return window

    def _sample_window(self) -> Box:
        """Sample negative or positive window based on neg_probability, if set.

        If neg_probability is not set, use
        :meth:`.RandomWindowGeoDataset._sample_window`.
        """
        if self.neg_probability is None:
            return super()._sample_window()

        if np.random.sample() < self.neg_probability:
            return self._sample_neg_window()
        return self._sample_pos_window()
