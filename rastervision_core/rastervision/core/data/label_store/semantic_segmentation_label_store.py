from typing import TYPE_CHECKING, Sequence, overload
from os.path import join
import logging

import numpy as np
import rasterio as rio
import rasterio.windows as rio_windows
from tqdm.auto import tqdm

from rastervision.pipeline.file_system import (
    get_local_path, json_to_file, make_dir, sync_to_dir, file_exists,
    download_if_needed, NotReadableError, get_tmp_dir)
from rastervision.core.box import Box
from rastervision.core.data import (CRSTransformer, ClassConfig)
from rastervision.core.data.label import (SemanticSegmentationLabels,
                                          SemanticSegmentationSmoothLabels)
from rastervision.core.data.label_store import LabelStore
from rastervision.core.data.label_source import SemanticSegmentationLabelSource
from rastervision.core.data.raster_transformer import RGBClassTransformer
from rastervision.core.data.raster_source import RasterioSource
from rastervision.core.data.utils import write_window

if TYPE_CHECKING:
    from rastervision.core.data import (VectorOutputConfig,
                                        SemanticSegmentationDiscreteLabels)

log = logging.getLogger(__name__)


class SemanticSegmentationLabelStore(LabelStore):
    """Storage for semantic segmentation predictions.

    Can store predicted class ID raster and class scores raster as GeoTIFFs,
    and can optionally vectorize predictions and store them as GeoJSON files.
    """

    def __init__(self,
                 uri: str,
                 crs_transformer: CRSTransformer,
                 class_config: ClassConfig,
                 bbox: Box | None = None,
                 tmp_dir: str | None = None,
                 vector_outputs: 'Sequence[VectorOutputConfig] | None' = None,
                 save_as_rgb: bool = False,
                 discrete_output: bool = True,
                 smooth_output: bool = False,
                 smooth_as_uint8: bool = False,
                 rasterio_block_size: int = 512):
        """Constructor.

        Args:
            uri: Path to directory where the predictions are/will be
                stored. Smooth scores will be saved as ``"uri/scores.tif"``,
                discrete labels will be stored as ``"uri/labels.tif"``, and
                vector outputs will be saved in ``"uri/vector_outputs/"``.
            crs_transformer: CRS transformer for correctly mapping from pixel
                coords to map coords.
            class_config: Class config.
            bbox: User-specified crop of the extent.
                If provided, only labels falling inside it are returned by
                :meth:`.SemanticSegmentationLabelStore.get_labels`. Must be
                provided if the corresponding ``RasterSource`` has
                ``bbox != extent``.
            tmp_dir: Temporary directory to use. If ``None``, will be
                auto-generated. Defaults to ``None``.
            vector_outputs:
                List of VectorOutputConfig's containing vectorization
                configuration information. Only classes for which a
                ``VectorOutputConfig`` is specified will be saved as vectors.
                If ``None``, no vector outputs will be produced.
                Defaults to ``None``.
            save_as_rgb: If ``True``, saves labels as an RGB
                image, using the class-color mapping in the ``class_config``.
                Defaults to ``False``.
            discrete_output: If ``True``, saves labels as a raster
                of class IDs (one band). Defaults to ``False``.
            smooth_output: If ``True``, saves labels as a raster
                of class scores (one band for each class). Defaults to ``False``.
            smooth_as_uint8: If ``True``, stores smooth class
                scores as ``np.uint8`` (0-255) values rather than as
                ``np.float32`` discrete labels, to help save memory/disk space.
                Defaults to ``False``.
            rasterio_block_size: Value to set ``blockxsize`` and ``blockysize``
                to. Defaults to ``512``.
        """
        self.root_uri = uri

        self.tmp_dir = tmp_dir
        if self.tmp_dir is None:
            self._tmp_dir = get_tmp_dir()
            self.tmp_dir = self._tmp_dir.name

        self.vector_outputs = vector_outputs
        self._bbox = bbox
        self._crs_transformer = crs_transformer
        self.class_config = class_config
        self.discrete_output = discrete_output
        self.smooth_output = smooth_output
        self.smooth_as_uint8 = smooth_as_uint8
        self.rasterio_block_size = rasterio_block_size

        self.class_transformer = None
        if save_as_rgb:
            self.class_transformer = RGBClassTransformer(class_config)

        self.label_source = None
        self.score_source = None

        if file_exists(self.label_uri):
            if self.class_transformer is not None:
                tfs = [self.class_transformer]
            else:
                tfs = []
            label_raster_source = RasterioSource(
                self.label_uri, raster_transformers=tfs, bbox=self._bbox)
            self.label_source = SemanticSegmentationLabelSource(
                label_raster_source, class_config)

        if self.smooth_output and file_exists(self.score_uri):
            num_classes = len(class_config)
            dtype = np.uint8 if self.smooth_as_uint8 else np.float32
            score_raster_source = RasterioSource(
                self.score_uri, bbox=self._bbox)

            bands_equal = (score_raster_source.num_channels == num_classes)
            dtypes_equal = (score_raster_source.dtype == dtype)
            if not (bands_equal and dtypes_equal):
                raise FileExistsError(f'{self.score_uri} already exists '
                                      'and is incompatible.')

            self.score_source = score_raster_source

    def get_labels(self) -> SemanticSegmentationLabels:
        """Get all labels.

        Returns:
            SemanticSegmentationLabels
        """
        if self.smooth_output:
            return self.get_scores()
        else:
            return self.get_discrete_labels()

    def get_discrete_labels(self) -> 'SemanticSegmentationDiscreteLabels':
        """Get all labels.

        Returns:
            SemanticSegmentationDiscreteLabels
        """
        if self.label_source is None:
            raise FileNotFoundError(
                f'Raster source at {self.label_uri} does not exist.')

        return self.label_source.get_labels()

    def get_scores(self) -> 'SemanticSegmentationSmoothLabels':
        """Get all scores.

        Returns:
            SemanticSegmentationSmoothLabels
        """
        if self.score_source is None:
            raise Exception(
                f'Raster source at {self.score_uri} does not exist '
                'or is not consistent with the current params.')

        log.info('Loading scores...')

        extent = self.score_source.extent
        try:
            hits_uri_local = download_if_needed(self.hits_uri)
            hits_arr = np.load(hits_uri_local)
        except NotReadableError:
            log.warning(f'Pixel hits array not found at {self.hits_uri}.'
                        'Setting all pixels hits to 1.')
            hits_arr = np.ones(extent.size, dtype=np.uint8)

        score_arr = self.score_source.get_chip(extent)
        # (H, W, C) --> (C, H, W)
        score_arr = score_arr.transpose(2, 0, 1)
        # convert to float
        if score_arr.dtype == np.uint8:
            score_arr = score_arr.astype(np.float16)
            score_arr /= 255

        _, h, w = score_arr.shape
        labels = SemanticSegmentationSmoothLabels(
            extent=Box(0, 0, h, w), num_classes=len(self.class_config))
        labels.pixel_scores = score_arr * hits_arr
        labels.pixel_hits = hits_arr
        return labels

    def save(self,
             labels: SemanticSegmentationLabels,
             profile: dict | None = None) -> None:
        """Save labels to disk.

        More info on rasterio IO:
        - https://github.com/mapbox/rasterio/blob/master/docs/quickstart.rst
        - https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html

        Args:
            labels - (SemanticSegmentationLabels) labels to be saved
        """
        make_dir(self.root_uri_local)

        out_profile = self.build_rasterio_gtiff_profile(labels)
        if profile is not None:
            out_profile.update(profile)

        if self.discrete_output:
            labels_path = get_local_path(self.label_uri, self.tmp_dir)
            self.write_discrete_raster_output(out_profile, labels_path, labels)

        if self.smooth_output:
            # if old scores exist, combine them with the new ones
            if self.score_source is not None:
                log.info('Old scores found. '
                         'Attempting to merge with current scores.')
                labels = self.merge_with_old_scores(labels)
            scores_path = get_local_path(self.score_uri, self.tmp_dir)
            hits_path = get_local_path(self.hits_uri, self.tmp_dir)
            self.write_smooth_raster_output(out_profile, scores_path,
                                            hits_path, labels)

        if self.vector_outputs:
            self.write_vector_outputs(labels, self.vector_output_dir_local)

        sync_to_dir(self.root_uri_local, self.root_uri)

    def build_rasterio_gtiff_profile(
            self, labels: SemanticSegmentationLabels) -> dict:
        height, width = labels.extent.size
        if self.bbox is not None:
            bbox_rio_window = self.bbox.rasterio_format()
            transform = rio_windows.transform(bbox_rio_window,
                                              self.crs_transformer.transform)
        else:
            transform = self.crs_transformer.transform
        out_profile = dict(
            driver='GTiff',
            height=height,
            width=width,
            transform=transform,
            crs=self.crs_transformer.image_crs,
            blockxsize=min(self.rasterio_block_size, width),
            blockysize=min(self.rasterio_block_size, height))
        return out_profile

    def merge_with_old_scores(self, labels: SemanticSegmentationSmoothLabels
                              ) -> SemanticSegmentationSmoothLabels:
        old_extent = self.score_source.extent
        new_extent = labels.extent
        if old_extent != new_extent:
            raise ValueError('Cannot merge with old sores. '
                             'Non-identical extents:\n'
                             f'old extent: {old_extent}\n'
                             f'new extent: {new_extent}')
        old_labels = self.get_scores()
        labels += old_labels
        return labels

    def write_smooth_raster_output(
            self, out_profile: dict, scores_path: str, hits_path: str,
            labels: SemanticSegmentationSmoothLabels) -> None:
        num_bands = labels.num_classes
        dtype = np.uint8 if self.smooth_as_uint8 else np.float32
        out_profile.update(dict(count=num_bands, dtype=dtype))

        extent = labels.extent

        with rio.open(scores_path, 'w', **out_profile) as ds:
            windows = [Box.from_rasterio(w) for _, w in ds.block_windows(1)]
            with tqdm(windows, desc='Saving pixel scores') as bar:
                for window in bar:
                    window, _ = self._clip_to_extent(extent, window)
                    score_arr = labels.get_score_arr(window)
                    if dtype == np.uint8:
                        score_arr = self._scores_to_uint8(score_arr)
                    else:
                        score_arr = score_arr.astype(dtype)
                    score_arr = score_arr.transpose(1, 2, 0)
                    write_window(ds, score_arr, window)

        # save pixel hits too
        np.save(hits_path, labels.pixel_hits)

    def write_discrete_raster_output(
            self, out_profile: dict, path: str,
            labels: SemanticSegmentationLabels) -> None:
        num_bands = 1 if self.class_transformer is None else 3
        dtype = np.uint8
        out_profile.update(dict(count=num_bands, dtype=dtype))

        extent = labels.extent
        null_class_id = self.class_config.null_class_id
        with rio.open(path, 'w', **out_profile) as ds:
            windows = [Box.from_rasterio(w) for _, w in ds.block_windows(1)]
            with tqdm(windows, desc='Saving pixel labels') as bar:
                for window in bar:
                    label_arr = labels.get_label_arr(
                        window, null_class_id).astype(dtype)
                    window, label_arr = self._clip_to_extent(
                        extent, window, label_arr)
                    if self.class_transformer is not None:
                        label_arr = self.class_transformer.class_to_rgb(
                            label_arr)
                    write_window(ds, label_arr, window)

    def write_vector_outputs(self, labels: SemanticSegmentationLabels,
                             vector_output_dir: str) -> None:
        """Write vectorized outputs for all configs in self.vector_outputs."""
        log.info('Writing vector outputs to disk.')

        with tqdm(self.vector_outputs, desc='Vectorizing predictions') as bar:
            for vo in bar:
                bar.set_postfix(vo.dict())
                class_mask = labels.get_class_mask(labels.extent, vo.class_id,
                                                   vo.threshold)
                self.write_vector_output(vo, class_mask, vector_output_dir)

    def write_vector_output(self, vo: 'VectorOutputConfig', mask: np.ndarray,
                            vector_output_dir: str) -> None:
        """Write vector output for a single ``VectorOutputConfig``."""
        from rastervision.core.data.utils import geoms_to_geojson

        mask = mask.astype(np.uint8)
        polys = vo.vectorize(mask)
        polys = [
            self.crs_transformer.pixel_to_map(p, bbox=self.bbox) for p in polys
        ]
        geojson = geoms_to_geojson(polys)
        out_uri = vo.get_uri(vector_output_dir, self.class_config)
        json_to_file(geojson, out_uri)

    @overload
    def _clip_to_extent(self, extent: Box, window: Box,
                        arr: np.ndarray) -> tuple[Box, np.ndarray]:
        ...

    @overload
    def _clip_to_extent(self, extent: Box, window: Box,
                        arr: None = ...) -> tuple[Box, None]:
        ...

    def _clip_to_extent(self, extent, window, arr=None):
        clipped_window = window.intersection(extent)
        if arr is not None:
            h, w = clipped_window.size
            arr = arr[:h, :w]
        return clipped_window, arr

    def _scores_to_uint8(self, score_arr: np.ndarray) -> np.ndarray:
        """Quantize scores to uint8 (0-255)."""
        score_arr *= 255
        score_arr = np.around(score_arr, out=score_arr)
        score_arr = score_arr.astype(np.uint8)
        return score_arr

    @property
    def bbox(self) -> 'Box':
        return self._bbox

    @property
    def crs_transformer(self) -> 'CRSTransformer':
        return self._crs_transformer

    @property
    def label_uri(self) -> str:
        return join(self.root_uri, 'labels.tif')

    @property
    def score_uri(self) -> str:
        return join(self.root_uri, 'scores.tif')

    @property
    def hits_uri(self) -> str:
        return join(self.root_uri, 'pixel_hits.npy')

    @property
    def vector_output_uri(self) -> str:
        return join(self.root_uri, 'vector_output')

    @property
    def root_uri_local(self) -> str:
        return get_local_path(self.root_uri, self.tmp_dir)

    @property
    def vector_output_dir_local(self) -> str:
        return get_local_path(self.vector_output_uri, self.tmp_dir)

    def set_bbox(self, bbox: 'Box') -> None:
        self._bbox = bbox
