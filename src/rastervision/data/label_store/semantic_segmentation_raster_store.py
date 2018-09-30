import numpy as np
import rasterio

import rastervision as rv
from rastervision.utils.files import (get_local_path, make_dir, upload_or_copy)
from rastervision.data.label import SemanticSegmentationLabels
from rastervision.data.label_store import LabelStore
from rastervision.data.utils import SegmentationClassTransformer


class SemanticSegmentationRasterStore(LabelStore):
    """A prediction label store for segmentation raster files.
    """

    def __init__(self, uri, crs_transformer, class_map, tmp_dir):
        """Constructor.

        Args:
            uri: (str) URI of GeoTIFF file used for storing predictions as RGB values
            crs_transformer: (CRSTransformer)
            class_map: (ClassMap) with color values used to convert class ids to
                RGB values
            tmp_dir: (str) temp directory to use
        """
        self.uri = uri
        self.crs_transformer = crs_transformer
        # Note: can't name this class_transformer due to Python using that attribute
        self.class_trans = SegmentationClassTransformer(class_map)
        self.tmp_dir = tmp_dir

    def get_labels(self):
        """Get all labels.

        Returns:
             np.ndarray of shape [height, width] with class ids as values
        """
        source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                      .with_uri(self.uri) \
                                      .build() \
                                      .create_source(self.tmp_dir)
        rgb_labels = source.get_raw_image_array()
        return self.class_trans.rgb_to_class(rgb_labels)

    def save(self, labels):
        """Save.

        Args:
            labels - (SemanticSegmentationLabels) labels to be saved
        """
        local_path = get_local_path(self.uri, self.tmp_dir)
        make_dir(local_path, use_dirname=True)

        # TODO: this only works if crs_transformer is RasterioCRSTransformer.
        # Need more general way of computing transform for the more general case.
        transform = self.crs_transformer.image_dataset.transform
        extent = labels.get_extent()
        crs = self.crs_transformer.get_image_crs()

        # https://github.com/mapbox/rasterio/blob/master/docs/quickstart.rst
        # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
        with rasterio.open(
                local_path,
                'w',
                driver='GTiff',
                height=extent.ymax,
                width=extent.xmax,
                count=3,
                dtype=np.uint8,
                transform=transform,
                crs=crs) as dataset:
            for (window, class_labels) in labels.get_label_pairs():
                window = (window.ymin, window.ymax), (window.xmin, window.xmax)
                rgb_labels = self.class_trans.class_to_rgb(class_labels)
                for chan in range(3):
                    dataset.write_band(
                        chan + 1, rgb_labels[:, :, chan], window=window)

        upload_or_copy(local_path, self.uri)

    def empty_labels(self):
        """Returns an empty SemanticSegmentationLabels object."""
        return SemanticSegmentationLabels()
