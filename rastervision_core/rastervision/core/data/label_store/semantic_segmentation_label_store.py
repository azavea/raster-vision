from typing import TYPE_CHECKING, Optional, Sequence, Tuple
from os.path import join
import logging

import numpy as np
import rasterio as rio
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

if TYPE_CHECKING:
    from rastervision.core.data import (VectorOutputConfig,
                                        SemanticSegmentationDiscreteLabels)

log = logging.getLogger(__name__)


class SemanticSegmentationLabelStore(LabelStore):
    """Storage for semantic segmentation predictions.

    Can store predicted class ID raster and class scores raster as GeoTIFFs,
    and can optionally vectorize predictions and store them as GeoJSON files.
    """

    def __init__(
            self,
            uri: str,
            extent: Box,
            crs_transformer: CRSTransformer,
            class_config: ClassConfig,
            tmp_dir: Optional[str] = None,
            vector_outputs: Optional[Sequence['VectorOutputConfig']] = None,
            save_as_rgb: bool = False,
            discrete_output: bool = True,
            smooth_output: bool = False,
            smooth_as_uint8: bool = False,
            rasterio_block_size: int = 512):
        """Constructor.

        Args:
            uri (str): Path to directory where the predictions are/will be
                stored. Smooth scores will be saved as "uri/scores.tif",
                discrete labels will be stored as "uri/labels.tif", and vector
                outputs will be saved in "uri/vector_outputs/".
            extent (Box): The extent of the scene.
            crs_transformer (CRSTransformer): CRS transformer for correctly
                mapping from pixel coords to map coords.
            tmp_dir (Optional[str], optional): Temporary directory to use. If
                None, will be auto-generated. Defaults to None.
            vector_outputs (Optional[Sequence[VectorOutputConfig]], optional):
                List of VectorOutputConfig's containing vectorization
                configuration information. Only classes for which a
                VectorOutputConfig is specified will be saved as vectors.
                If None, no vector outputs will be produced. Defaults to None.
            class_config (ClassConfig): Class config.
            save_as_rgb (bool, optional): If True, saves labels as an RGB
                image, using the class-color mapping in the class_config.
                Defaults to False.
            discrete_output (bool, optional): If True, saves labels as a raster
                of class IDs (one band). Defaults to False.
            smooth_output (bool, optional): If True, saves labels as a raster
                of class scores (one band for each class). Defaults to False.
            smooth_as_uint8 (bool, optional): If True, stores smooth class
                scores as np.uint8 (0-255) values rather than as np.float32
                discrete labels, to help save memory/disk space.
                Defaults to False.
            rasterio_block_size (int, optional): Value to set blockxsize and
                blockysize to. Defaults to 512.
        """
        self.root_uri = uri
        self.label_uri = join(uri, 'labels.tif')
        self.score_uri = join(uri, 'scores.tif')
        self.hits_uri = join(uri, 'pixel_hits.npy')

        self.tmp_dir = tmp_dir
        if self.tmp_dir is None:
            self._tmp_dir = get_tmp_dir()
            self.tmp_dir = self._tmp_dir.name

        self.vector_outputs = vector_outputs
        self.extent = extent
        self.crs_transformer = crs_transformer
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
                self.label_uri, raster_transformers=tfs)
            self.label_source = SemanticSegmentationLabelSource(
                label_raster_source, class_config)

        if self.smooth_output:
            if file_exists(self.score_uri):
                raster_source = RasterioSource(self.score_uri)
                extents_equal = raster_source.extent == self.extent
                bands_equal = raster_source.num_channels == len(class_config)
                self_dtype = np.uint8 if self.smooth_as_uint8 else np.float32
                dtypes_equal = raster_source.dtype == self_dtype

                if extents_equal and bands_equal and dtypes_equal:
                    self.score_source = raster_source
                else:
                    raise FileExistsError(f'{self.score_uri} already exists '
                                          'and is incompatible.')

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
            log.warn(f'Pixel hits array not found at {self.hits_uri}.'
                     'Setting all pixels hits to 1.')
            hits_arr = np.ones(extent.size, dtype=np.uint8)

        score_arr = self.score_source.get_chip(extent)
        # (H, W, C) --> (C, H, W)
        score_arr = score_arr.transpose(2, 0, 1)
        # convert to float
        if score_arr.dtype == np.uint8:
            score_arr = score_arr.astype(np.float16)
            score_arr /= 255

        labels = self.empty_labels()
        assert isinstance(labels, SemanticSegmentationSmoothLabels)
        labels.pixel_scores = score_arr * hits_arr
        labels.pixel_hits = hits_arr
        return labels

    def save(self,
             labels: SemanticSegmentationLabels,
             profile: Optional[dict] = None) -> None:
        """Save labels to disk.

        More info on rasterio IO:
        - https://github.com/mapbox/rasterio/blob/master/docs/quickstart.rst
        - https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html

        Args:
            labels - (SemanticSegmentationLabels) labels to be saved
        """
        local_root = get_local_path(self.root_uri, self.tmp_dir)
        make_dir(local_root)

        height, width = self.extent.size
        out_profile = dict(
            driver='GTiff',
            height=height,
            width=width,
            transform=self.crs_transformer.transform,
            crs=self.crs_transformer.image_crs,
            blockxsize=min(self.rasterio_block_size, width),
            blockysize=min(self.rasterio_block_size, height))
        if profile is not None:
            out_profile.update(profile)

        # if old scores exist, combine them with the new ones
        if self.score_source is not None:
            log.info('Old scores found. Merging with current scores.')
            old_labels = self.get_scores()
            labels += old_labels

        if self.discrete_output:
            labels_path = get_local_path(self.label_uri, self.tmp_dir)
            self.write_discrete_raster_output(out_profile, labels_path, labels)

        if self.smooth_output:
            scores_path = get_local_path(self.score_uri, self.tmp_dir)
            hits_path = get_local_path(self.hits_uri, self.tmp_dir)
            self.write_smooth_raster_output(out_profile, scores_path,
                                            hits_path, labels)

        if self.vector_outputs is not None:
            self.write_vector_outputs(labels)

        sync_to_dir(local_root, self.root_uri)

    def write_smooth_raster_output(
            self, out_profile: dict, scores_path: str, hits_path: str,
            labels: SemanticSegmentationSmoothLabels) -> None:
        num_bands = labels.num_classes
        dtype = np.uint8 if self.smooth_as_uint8 else np.float32
        out_profile.update(dict(count=num_bands, dtype=dtype))

        with rio.open(scores_path, 'w', **out_profile) as ds:
            windows = [Box.from_rasterio(w) for _, w in ds.block_windows(1)]
            with tqdm(windows, desc='Saving pixel scores') as bar:
                for window in bar:
                    window, _ = self._clip_to_extent(self.extent, window)
                    score_arr = labels.get_score_arr(window)
                    if dtype == np.uint8:
                        score_arr = self._scores_to_uint8(score_arr)
                    else:
                        score_arr = score_arr.astype(dtype)
                    self._write_array(ds, window, score_arr)

        # save pixel hits too
        np.save(hits_path, labels.pixel_hits)

    def write_discrete_raster_output(
            self, out_profile: dict, path: str,
            labels: SemanticSegmentationLabels) -> None:
        num_bands = 1 if self.class_transformer is None else 3
        dtype = np.uint8
        out_profile.update(dict(count=num_bands, dtype=dtype))

        null_class_id = self.class_config.null_class_id
        with rio.open(path, 'w', **out_profile) as ds:
            windows = [Box.from_rasterio(w) for _, w in ds.block_windows(1)]
            with tqdm(windows, desc='Saving pixel labels') as bar:
                for window in bar:
                    label_arr = labels.get_label_arr(
                        window, null_class_id).astype(dtype)
                    window, label_arr = self._clip_to_extent(
                        self.extent, window, label_arr)
                    if self.class_transformer is not None:
                        label_arr = self.class_transformer.class_to_rgb(
                            label_arr)
                        label_arr = label_arr.transpose(2, 0, 1)
                    self._write_array(ds, window, label_arr)

    def write_vector_outputs(self, labels: SemanticSegmentationLabels) -> None:
        """Write vectorized outputs for all configs in self.vector_outputs."""
        from rastervision.core.data.utils import (denoise, geoms_to_geojson,
                                                  mask_to_building_polygons,
                                                  mask_to_polygons)

        log.info('Writing vector output to disk.')

        label_arr = labels.get_label_arr(self.extent,
                                         self.class_config.null_class_id)

        with tqdm(self.vector_outputs, desc='Vectorizing predictions') as bar:
            for i, vo in enumerate(bar):
                bar.set_postfix(
                    dict(
                        class_id=vo.class_id,
                        mode=vo.get_mode(),
                        denoise_radius=vo.denoise))

                if vo.uri is None:
                    log.info(f'Skipping VectorOutputConfig at index {i} '
                             'due to missing uri.')
                    continue

                class_mask = (label_arr == vo.class_id).astype(np.uint8)

                if vo.denoise > 0:
                    class_mask = denoise(class_mask, radius=vo.denoise)

                mode = vo.get_mode()
                if mode == 'polygons':
                    polys = mask_to_polygons(class_mask)
                elif mode == 'buildings':
                    polys = mask_to_building_polygons(
                        mask=class_mask,
                        min_area=vo.min_area,
                        width_factor=vo.element_width_factor,
                        thickness=vo.element_thickness)
                else:
                    raise NotImplementedError()

                polys = [self.crs_transformer.pixel_to_map(p) for p in polys]
                geojson = geoms_to_geojson(polys)
                json_to_file(geojson, vo.uri)

    def empty_labels(self, **kwargs) -> SemanticSegmentationLabels:
        """Returns an empty SemanticSegmentationLabels object."""
        args = dict(
            extent=self.extent,
            num_classes=len(self.class_config),
            smooth=self.smooth_output)
        args.update(**kwargs)
        labels = SemanticSegmentationLabels.make_empty(**args)
        return labels

    def _write_array(self, dataset: rio.DatasetReader, window: Box,
                     arr: np.ndarray) -> None:
        """Write array out to a rasterio dataset. Array must be of shape
        (C, H, W).
        """
        rio_window = window.rasterio_format()
        if len(arr.shape) == 2:
            dataset.write_band(1, arr, window=rio_window)
        else:
            for i, band in enumerate(arr, start=1):
                dataset.write_band(i, band, window=rio_window)

    def _clip_to_extent(self,
                        extent: Box,
                        window: Box,
                        arr: Optional[np.ndarray] = None
                        ) -> Tuple[Box, Optional[np.ndarray]]:
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
