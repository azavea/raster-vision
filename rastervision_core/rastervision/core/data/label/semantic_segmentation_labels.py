from typing import (TYPE_CHECKING, Any, Iterable, Sequence)
from abc import abstractmethod

import numpy as np
from rasterio.features import rasterize
from shapely.ops import transform

from rastervision.core.box import Box
from rastervision.core.data.label import Labels
from rastervision.core.data.label.utils import discard_prediction_edges

if TYPE_CHECKING:
    from typing import Self
    from shapely.geometry import Polygon
    from rastervision.core.data import (ClassConfig, CRSTransformer,
                                        VectorOutputConfig)


class SemanticSegmentationLabels(Labels):
    """Representation of Semantic Segmentation labels."""

    def __init__(self, extent: Box, num_classes: int, dtype: np.dtype):
        """Constructor.

        Args:
            extent (Box): The extent of the region to which the labels belong,
                in global coordinates.
            num_classes (int): Number of classes.
        """
        self.extent = extent
        self.num_classes = num_classes
        self.ymin, self.xmin, self.width, self.height = extent.to_xywh()
        self.dtype = dtype

    @abstractmethod
    def __add__(self, other: 'Self') -> 'Self':
        """Merge self with other labels."""

    def __setitem__(self, window: Box, values: np.ndarray) -> None:
        """Set labels for the given window."""
        self.add_window(window, values)

    @abstractmethod
    def __delitem__(self, window: Box) -> None:
        """Delete labels for the given window."""

    @abstractmethod
    def __getitem__(self, window: Box) -> np.ndarray:
        """Get labels for the given window."""

    @abstractmethod
    def add_window(self, window: Box, values: np.ndarray) -> list[Box]:
        """Set labels for the given window."""

    @abstractmethod
    def get_label_arr(self, window: Box,
                      null_class_id: int = -1) -> np.ndarray:
        """Get labels as a 2D array of class IDs.

        Note: The returned array is not guaranteed to be the same size as the
        input window.
        """

    @abstractmethod
    def get_score_arr(self, window: Box,
                      null_class_id: int = -1) -> np.ndarray:
        """Get (C, H, W) array of pixel scores."""

    def get_class_mask(self,
                       window: Box,
                       class_id: int,
                       threshold: float | None = None) -> np.ndarray:
        """Get a binary mask representing all pixels of a class."""
        scores = self.get_score_arr(window)
        if threshold is None:
            threshold = (1 / self.num_classes)
        mask = scores[class_id] >= threshold
        return mask

    def get_windows(self, **kwargs) -> list[Box]:
        """Generate sliding windows over the local extent.

        The keyword args are passed to :meth:`.Box.get_windows` and can
        therefore be used to control the specifications of the windows.

        If the keyword args do not contain size, a list of length 1,
        containing the full extent is returned.

        Args:
            **kwargs: Extra args for :meth:`.Box.get_windows`.
        """
        size: int | None = kwargs.pop('size', None)
        if size is None:
            return [self.extent]
        return self.extent.get_windows(size, size, **kwargs)

    def filter_by_aoi(self, aoi_polygons: list['Polygon'], null_class_id: int,
                      **kwargs) -> 'Self':
        """Keep only the values that lie inside the AOI.

        This is an inplace operation.

        Args:
            aoi_polygons (list[Polygon]): AOI polygons to filter by, in pixel
                coordinates.
            null_class_id (int): Class ID to assign to pixels falling outside
                the AOI polygons.
            **kwargs: Extra args for
                :meth:`.SemanticSegmentationLabels.get_windows`.
        """
        if not aoi_polygons:
            return self
        for window in self.get_windows(**kwargs):
            self._filter_window_by_aoi(window, aoi_polygons, null_class_id)
        return self

    @abstractmethod
    def mask_fill(self, window: Box, mask: np.ndarray,
                  fill_value: Any) -> None:
        """Given a window and a binary mask, set all the pixels in the window
        for which the mask is ON to the fill_value.
        """

    def _filter_window_by_aoi(self, window: Box, aoi_polygons: list['Polygon'],
                              null_class_id: int) -> None:
        window_geom = window.to_shapely()
        label_arr = self[window]

        # For each aoi_polygon, intersect with window, and
        # put in window frame of reference.
        window_aois = []
        for aoi in aoi_polygons:
            window_aoi = aoi.intersection(window_geom)
            if not window_aoi.is_empty:

                def transform_shape(x, y, z=None):
                    return (x - window.xmin, y - window.ymin)

                window_aoi = transform(transform_shape, window_aoi)
                window_aois.append(window_aoi)

        # If window doesn't overlap with any AOI, then it won't be in
        # new_labels.
        if window_aois:
            # If window intersects with AOI, set pixels outside the
            # AOI polygon to 0 so they are ignored during eval.
            mask = rasterize(
                [(p, 0) for p in window_aois],
                out_shape=label_arr.shape[-2:],
                fill=1,
                dtype=np.uint8)
            mask = mask.astype(bool)
            self.mask_fill(window, mask, null_class_id)
        else:
            del self[window]

    @classmethod
    def make_empty(cls, extent: Box, num_classes: int,
                   smooth: bool = False) -> 'Self':
        """Instantiate an empty instance.

        Args:
            extent (Box): The extent of the region to which the labels belong,
                in global coordinates.
            num_classes (int): Number of classes.
            smooth (bool): If True, creates a
                SemanticSegmentationSmoothLabels object. If False, creates a
                SemanticSegmentationDiscreteLabels object. Defaults to False.

        Returns:
            If smooth=True, returns a SemanticSegmentationSmoothLabels.
            Otherwise, a SemanticSegmentationDiscreteLabels.

        Raises:
            ValueError: if num_classes and extent are not specified, but
                smooth=True.
        """
        if not smooth:
            return SemanticSegmentationDiscreteLabels.make_empty(
                extent=extent, num_classes=num_classes)
        else:
            return SemanticSegmentationSmoothLabels.make_empty(
                extent=extent, num_classes=num_classes)

    @classmethod
    def from_predictions(cls,
                         windows: Iterable['Box'],
                         predictions: Iterable[Any],
                         extent: Box,
                         num_classes: int,
                         smooth: bool = False,
                         crop_sz: int | None = None) -> 'Self':
        """Instantiate from windows and their corresponding predictions.

        Args:
            windows: Boxes in pixel coords, specifying chips in the raster.
            predictions: The model predictions for each chip specified by the
                windows.
            extent: The extent of the region to which the labels belong, in
                global coordinates.
            num_classes: Number of classes.
            smooth: If ``True``, creates a ``SemanticSegmentationSmoothLabels``
                object. If ``False``, creates a
                ``SemanticSegmentationDiscreteLabels`` object.
                Defaults to ``False``.
            crop_sz: Number of rows/columns of pixels from the
                edge of prediction windows to discard. This is useful because
                predictions near edges tend to be lower quality and can result
                in very visible artifacts near the edges of chips. This should
                only be used if the given windows represent a sliding-window
                grid over the scene extent with overlap between adjacent
                windows. Defaults to None.

        Returns:
            If smooth=True, returns a SemanticSegmentationSmoothLabels.
            Otherwise, a SemanticSegmentationDiscreteLabels.
        """
        labels = cls.make_empty(extent, num_classes, smooth=smooth)
        labels.add_predictions(windows, predictions, crop_sz=crop_sz)
        return labels

    def add_predictions(self,
                        windows: Iterable['Box'],
                        predictions: Iterable[Any],
                        crop_sz: int | None = None) -> None:
        """Populate predictions.

        Args:
            windows: Boxes in pixel coords, specifying chips in the raster.
            predictions: The model predictions for each chip specified by the
                windows.
            crop_sz: Number of rows/columns of pixels from the
                edge of prediction windows to discard. This is useful because
                predictions near edges tend to be lower quality and can result
                in very visible artifacts near the edges of chips. This should
                only be used if the given windows represent a sliding-window
                grid over the scene extent with overlap between adjacent
                windows. Defaults to None.
        """
        if crop_sz is not None:
            windows, predictions = discard_prediction_edges(
                windows, predictions, crop_sz)
        # If predictions is tqdm-wrapped, it needs to be the first arg to zip()
        # or the progress bar won't terminate with the correct count.
        for prediction, window in zip(predictions, windows):
            self[window] = prediction


class SemanticSegmentationDiscreteLabels(SemanticSegmentationLabels):
    """Vote-counts for each pixel belonging to each class.

    Maintains a num_classes x H x W array where value_{ijk} represents how
    many times pixel_{jk} has been classified as class i. A label array can be
    obtained from this by argmax'ing along the class dimension. Can also be
    turned into a score converting counts to probabilities.
    """

    def __init__(self, extent: Box, num_classes: int, dtype: Any = np.uint8):
        """Constructor.

        Args:
            extent: The extent of the region to which the labels belong, in
                global coordinates.
            num_classes: Number of classes.
            dtype: dtype of the counts array. Defaults to np.uint8.
        """
        super().__init__(extent, num_classes, dtype)

        self.pixel_counts = np.zeros(
            (self.num_classes, self.height, self.width), dtype=self.dtype)
        # track which pixels have been hit at all
        self.hit_mask = np.zeros((self.height, self.width), dtype=bool)

    def __add__(self, other: 'Self') -> 'Self':
        """Merge self with other labels by adding the pixel counts."""
        if self.extent != other.extent:
            raise ValueError('Cannot add labels with unqeual extents.')

        self.pixel_counts += other.pixel_counts
        return self

    def __eq__(self, other: 'Self') -> bool:
        if not isinstance(other, SemanticSegmentationDiscreteLabels):
            return False
        if self.extent != other.extent:
            return False
        mask_equal = np.all(self.hit_mask == other.hit_mask)
        if not mask_equal:
            return False
        counts_equal = np.all(self.pixel_counts == other.pixel_counts)
        return counts_equal

    def __delitem__(self, window: Box) -> None:
        """Reset counts to zero for pixels in the window."""
        y0, x0, y1, x1 = window.intersection(self.extent)
        self.pixel_counts[..., y0:y1, x0:x1] = 0
        self.hit_mask[y0:y1, x0:x1] = False

    def __getitem__(self, window: Box) -> np.ndarray:
        return self.get_label_arr(window)

    def add_window(self, window: Box, pixel_class_ids: np.ndarray) -> None:
        # sub-window in self.extent coords to write to
        window_dst = window.intersection(self.extent)

        # sub-window in pixel_class_ids coords to read from
        window_src = window_dst.to_global_coords(
            self.extent).to_local_coords(window)

        # read sub-window from source array
        src_yslice, src_xslice = window_src.to_slices()
        pixel_class_ids = pixel_class_ids.astype(self.dtype)
        pixel_class_ids = pixel_class_ids[..., src_yslice, src_xslice]

        # write sub-window in destination array
        dst_yslice, dst_xslice = window_dst.to_slices()
        window_pixel_counts = self.pixel_counts[:, dst_yslice, dst_xslice]
        for ch_class_id, ch in enumerate(window_pixel_counts):
            ch[pixel_class_ids == ch_class_id] += 1
        self.hit_mask[dst_yslice, dst_xslice] = True

    def get_label_arr(self, window: Box,
                      null_class_id: int = -1) -> np.ndarray:
        """Get labels as array of class IDs.

        Returns null_class_id for pixels for which there is no data.
        """
        y0, x0, y1, x1 = window.intersection(self.extent)
        label_arr = self.pixel_counts[..., y0:y1, x0:x1].argmax(axis=0)
        hit_mask = self.hit_mask[y0:y1, x0:x1]
        return np.where(hit_mask, label_arr, null_class_id)

    def get_score_arr(self, window: Box) -> np.ndarray:
        """Get array of pixel scores."""
        y0, x0, y1, x1 = window.intersection(self.extent)
        class_counts = self.pixel_counts[..., y0:y1, x0:x1]
        scores = class_counts / class_counts.sum(axis=0)
        return scores

    def mask_fill(self, window: Box, mask: np.ndarray,
                  fill_value: Any) -> None:
        """Set fill_value'th class ID's count to 1 and all others to zero."""
        class_id = fill_value
        y0, x0, y1, x1 = window.intersection(self.extent)
        h, w = y1 - y0, x1 - x0
        mask = mask[:h, :w]
        self.pixel_counts[:, y0:y1, x0:x1][..., mask] = 0
        self.pixel_counts[class_id, y0:y1, x0:x1][mask] = 1

    @classmethod
    def make_empty(cls, extent: Box, num_classes: int) -> 'Self':
        """Instantiate an empty instance."""
        return cls(extent=extent, num_classes=num_classes)

    @classmethod
    def from_predictions(cls,
                         windows: Iterable['Box'],
                         predictions: Iterable[Any],
                         extent: Box,
                         num_classes: int,
                         crop_sz: int | None = None) -> 'Self':
        labels = cls.make_empty(extent, num_classes)
        labels.add_predictions(windows, predictions, crop_sz=crop_sz)
        return labels

    def save(self,
             uri: str,
             crs_transformer: 'CRSTransformer',
             class_config: 'ClassConfig',
             bbox: Box | None = None,
             tmp_dir: str | None = None,
             save_as_rgb: bool = False,
             raster_output: bool = True,
             rasterio_block_size: int = 512,
             vector_outputs: 'Sequence[VectorOutputConfig] | None' = None,
             profile_overrides: dict | None = None) -> None:
        """Save labels as a raster and/or vectors.

        If URI is remote, all files will be first written locally and then
        uploaded to the URI.

        Args:
            uri: URI of directory in which to save all output files.
            crs_transformer: CRSTransformer to configure CRS and affine
                transform of the output GeoTiff.
            class_config: The ClassConfig.
            bbox: User-specified crop of the extent. Must be provided if the
                corresponding RasterSource has ``bbox != extent``.
            tmp_dir: Temporary directory to use. If None, will be
                auto-generated. Defaults to ``None``.
            save_as_rgb: If ``True``, Saves labels as an RGB image, using the
                class-color mapping in the class_config. Defaults to ``False``.
            raster_output: If ``True``, saves labels as a raster of class IDs
                (one band). Defaults to ``True``.
            rasterio_block_size: Value to set `blockxsize` and ``blockysize``
                to. Defaults to ``512``.
            vector_outputs: List of VectorOutputConfig's containing
                vectorization configuration information. Only classes for which
                a ``VectorOutputConfig`` is specified will be saved as vectors.
                If ``None``, no vector outputs will be produced.
                Defaults to ``None``.
            profile_overrides: This can be used to arbitrarily override
                properties in the profile used to create the output GeoTiff.
                Defaults to ``None``.
        """
        from rastervision.core.data import SemanticSegmentationLabelStore

        label_store = SemanticSegmentationLabelStore(
            uri=uri,
            crs_transformer=crs_transformer,
            class_config=class_config,
            bbox=bbox,
            tmp_dir=tmp_dir,
            save_as_rgb=save_as_rgb,
            discrete_output=raster_output,
            smooth_output=False,
            rasterio_block_size=rasterio_block_size,
            vector_outputs=vector_outputs)
        label_store.save(self, profile=profile_overrides)


class SemanticSegmentationSmoothLabels(SemanticSegmentationLabels):
    """Membership-scores for each pixel for each class.

    Maintains a num_classes x H x W array where value_{ijk} represents the
    probability (or some other measure) of pixel_{jk} belonging to class i.
    A discrete label array can be obtained from this by argmax'ing along the
    class dimension.
    """

    def __init__(self,
                 extent: Box,
                 num_classes: int,
                 dtype: Any = np.float16,
                 dtype_hits: Any = np.uint8):
        """Constructor.

        Args:
            extent: The extent of the region to which the labels belong,
                in global coordinates.
            num_classes: Number of classes.
            dtype: ``dtype`` of the scores array. Defaults to ``np.float16``.
            dtype_hits: ``dtype`` of the hits array. Defaults to ``np.uint8``.
        """
        super().__init__(extent, num_classes, dtype)

        self.pixel_scores = np.zeros(
            (self.num_classes, self.height, self.width), dtype=self.dtype)
        self.pixel_hits = np.zeros((self.height, self.width), dtype=dtype_hits)

    def __add__(self, other: 'Self') -> 'Self':
        """Merge self with other by adding pixel scores and hits."""
        if self.extent != other.extent:
            raise ValueError('Cannot add labels with unqeual extents.')

        self.pixel_scores += other.pixel_scores
        self.pixel_hits += other.pixel_hits
        return self

    def __eq__(self, other: 'Self') -> bool:
        if not isinstance(other, SemanticSegmentationSmoothLabels):
            return False
        if self.extent != other.extent:
            return False
        scores_equal = np.allclose(self.pixel_scores, other.pixel_scores)
        hits_equal = np.array_equal(self.pixel_hits, other.pixel_hits)
        return (scores_equal and hits_equal)

    def __delitem__(self, window: Box) -> None:
        """Reset scores and hits to zero for pixels in the window."""
        y0, x0, y1, x1 = window.intersection(self.extent)
        self.pixel_scores[..., y0:y1, x0:x1] = 0
        self.pixel_hits[..., y0:y1, x0:x1] = 0

    def __getitem__(self, window: Box) -> np.ndarray:
        return self.get_score_arr(window)

    def add_window(self, window: Box, pixel_class_scores: np.ndarray) -> None:
        # sub-window in self.extent coords to write to
        window_dst = window.intersection(self.extent)

        # sub-window in pixel_class_scores coords to read from
        window_src = window_dst.to_global_coords(
            self.extent).to_local_coords(window)

        # read sub-window from source array
        src_yslice, src_xslice = window_src.to_slices()
        pixel_class_scores = pixel_class_scores.astype(self.dtype)
        pixel_class_scores = pixel_class_scores[..., src_yslice, src_xslice]

        # write sub-window in destination array
        dst_yslice, dst_xslice = window_dst.to_slices()
        self.pixel_scores[..., dst_yslice, dst_xslice] += pixel_class_scores
        self.pixel_hits[dst_yslice, dst_xslice] += 1

    def get_score_arr(self, window: Box) -> np.ndarray:
        """Get array of pixel scores."""
        y0, x0, y1, x1 = window.intersection(self.extent)
        scores = self.pixel_scores[..., y0:y1, x0:x1]
        hits = self.pixel_hits[y0:y1, x0:x1]
        avg_scores = scores / hits
        return avg_scores

    def get_label_arr(self, window: Box,
                      null_class_id: int = -1) -> np.ndarray:
        """Get labels as array of class IDs.

        Returns null_class_id for pixels for which there is no data.
        """
        avg_scores = self.get_score_arr(window)
        label_arr = np.argmax(avg_scores, axis=0)
        mask = np.isnan(avg_scores[0])
        return np.where(mask, null_class_id, label_arr)

    def mask_fill(self, window: Box, mask: np.ndarray,
                  fill_value: Any) -> None:
        """Set fill_value'th class ID's score to 1 and all others to zero."""
        class_id = fill_value
        y0, x0, y1, x1 = window.intersection(self.extent)
        h, w = y1 - y0, x1 - x0
        mask = mask[:h, :w]
        self.pixel_scores[..., y0:y1, x0:x1][..., mask] = 0
        self.pixel_scores[class_id, y0:y1, x0:x1][mask] = 1
        self.pixel_hits[y0:y1, x0:x1][mask] = 1

    @classmethod
    def make_empty(cls, extent: Box, num_classes: int) -> 'Self':
        """Instantiate an empty instance."""
        return cls(extent=extent, num_classes=num_classes)

    @classmethod
    def from_predictions(cls,
                         windows: Iterable['Box'],
                         predictions: Iterable[Any],
                         extent: Box,
                         num_classes: int,
                         crop_sz: int | None = None) -> 'Self':
        labels = cls.make_empty(extent, num_classes)
        labels.add_predictions(windows, predictions, crop_sz=crop_sz)
        return labels

    def save(self,
             uri: str,
             crs_transformer: 'CRSTransformer',
             class_config: 'ClassConfig',
             bbox: Box | None = None,
             tmp_dir: str | None = None,
             save_as_rgb: bool = False,
             discrete_output: bool = True,
             smooth_output: bool = True,
             smooth_as_uint8: bool = False,
             rasterio_block_size: int = 512,
             vector_outputs: 'Sequence[VectorOutputConfig] | None' = None,
             profile_overrides: dict | None = None) -> None:
        """Save labels as rasters and/or vectors.

        If URI is remote, all files will be first written locally and then
        uploaded to the URI.

        Args:
            uri: URI of directory in which to save all output files.
            crs_transformer: CRSTransformer to configure CRS and affine
                transform of the output GeoTiff(s).
            class_config: The ClassConfig.
            bbox: User-specified crop of the extent. Must be provided if the
                corresponding RasterSource has bbox != extent.
            tmp_dir: Temporary directory to use. If ``None``, will be
                auto-generated. Defaults to ``None``.
            save_as_rgb: If True, saves labels as an RGB image, using the
                class-color mapping in the ``class_config``.
                Defaults to ``False``.
            discrete_output: If ``True``, saves labels as a raster of class IDs
                (one band). Defaults to ``True``.
            smooth_output: If ``True``, saves labels as a raster of class
                scores (one band for each class). Defaults to ``True``.
            smooth_as_uint8: If ``True``, stores smooth class scores as
                ``np.uint8`` (0-255) values rather than as ``np.float32``
                discrete labels, to help save memory/disk space.
                Defaults to ``False``.
            rasterio_block_size: Value to set ``blockxsize`` and ``blockysize``
                to. Defaults to ``512``.
            vector_outputs: List of VectorOutputConfig's containing
                vectorization configuration information. Only classes for which
                a ``VectorOutputConfig`` is specified will be saved as vectors.
                If ``None``, no vector outputs will be produced.
                Defaults to ``None``.
            profile_overrides: This can be used to arbitrarily override
                properties in the profile used to create the output GeoTiff(s).
                Defaults to ``None``.
        """
        from rastervision.core.data import SemanticSegmentationLabelStore

        label_store = SemanticSegmentationLabelStore(
            uri=uri,
            crs_transformer=crs_transformer,
            class_config=class_config,
            bbox=bbox,
            tmp_dir=tmp_dir,
            save_as_rgb=save_as_rgb,
            discrete_output=discrete_output,
            smooth_output=smooth_output,
            smooth_as_uint8=smooth_as_uint8,
            rasterio_block_size=rasterio_block_size,
            vector_outputs=vector_outputs)
        label_store.save(self, profile=profile_overrides)
