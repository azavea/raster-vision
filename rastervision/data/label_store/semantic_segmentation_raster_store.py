import numpy as np
import rasterio

import rastervision as rv
from rastervision.utils.files import (get_local_path, make_dir, upload_or_copy)
from rastervision.data.label import SemanticSegmentationLabels
from rastervision.data.label_store import LabelStore
from rastervision.data.label_source import SegmentationClassTransformer


class SemanticSegmentationRasterStore(LabelStore):
    """A prediction label store for segmentation raster files.
    """

    def __init__(self,
                 uri,
                 extent,
                 crs_transformer,
                 tmp_dir,
                 vector_output=None,
                 class_map=None):
        """Constructor.

        Args:
            uri: (str) URI of GeoTIFF file used for storing predictions as RGB values
            extent: (Box) The extent of the scene
            crs_transformer: (CRSTransformer)
            tmp_dir: (str) temp directory to use
            vector_output: (str or None) URI of GeoJSON polygons derived from the mask
            class_map: (ClassMap) with color values used to convert class ids to
                RGB values

        """
        self.uri = uri
        self.vector_output = vector_output
        self.extent = extent
        self.crs_transformer = crs_transformer
        self.tmp_dir = tmp_dir
        # Note: can't name this class_transformer due to Python using that attribute
        if class_map:
            self.class_trans = SegmentationClassTransformer(class_map)
        else:
            self.class_trans = None

    def get_labels(self):
        """Get all labels.

        Returns:
            SemanticSegmentationLabels
        """
        source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                      .with_uri(self.uri) \
                                      .build() \
                                      .create_source(self.tmp_dir)
        with source.activate():
            raw_labels = source.get_raw_image_array()
            if self.class_trans:
                labels = self.class_trans.rgb_to_class(raw_labels)
            else:
                labels = np.squeeze(raw_labels)
            return SemanticSegmentationLabels.from_array(labels)

    def save(self, labels):
        """Save.

        Args:
            labels - (SemanticSegmentationLabels) labels to be saved
        """
        local_path = get_local_path(self.uri, self.tmp_dir)
        make_dir(local_path, use_dirname=True)

        # TODO: this only works if crs_transformer is RasterioCRSTransformer.
        # Need more general way of computing transform for the more general case.
        transform = self.crs_transformer.transform
        crs = self.crs_transformer.get_image_crs()
        clipped_labels = labels.get_clipped_labels(self.extent)

        band_count = 1
        dtype = np.uint8
        if self.class_trans:
            band_count = 3
            dtype = np.uint8

        if self.vector_output:
            mask = np.zeros(
                (self.extent.ymax, self.extent.xmax), dtype=np.uint8)
        else:
            mask = None

        # https://github.com/mapbox/rasterio/blob/master/docs/quickstart.rst
        # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
        with rasterio.open(
                local_path,
                'w',
                driver='GTiff',
                height=self.extent.ymax,
                width=self.extent.xmax,
                count=band_count,
                dtype=dtype,
                transform=transform,
                crs=crs) as dataset:
            for (window, class_labels) in clipped_labels.get_label_pairs():
                window = (window.ymin, window.ymax), (window.xmin, window.xmax)
                if mask is not None:
                    mask[window[0][0]:window[0][1], window[1][0]:window[1][
                        1]] = class_labels
                if self.class_trans:
                    rgb_labels = self.class_trans.class_to_rgb(class_labels)
                    for chan in range(3):
                        dataset.write_band(
                            chan + 1, rgb_labels[:, :, chan], window=window)
                else:
                    img = class_labels.astype(dtype)
                    dataset.write_band(1, img, window=window)

        upload_or_copy(local_path, self.uri)

        if self.vector_output:
            import mask_to_polygons.vectorification as vectorification
            import mask_to_polygons.processing.denoise as denoise

            for vo in self.vector_output:
                denoise_radius = vo['denoise']
                uri = vo['uri']
                mode = vo['mode']
                class_id = vo['class_id']
                class_mask = np.array(mask == class_id, dtype=np.uint8)
                local_geojson_path = get_local_path(uri, self.tmp_dir)

                transform = self.crs_transformer.get_affine_transform()

                if denoise_radius > 0:
                    class_mask = denoise.denoise(class_mask, denoise_radius)

                if uri and mode == 'buildings':
                    geojson = vectorification.geojson_from_mask(
                        mask=class_mask, transform=transform, mode=mode)
                elif uri and mode == 'polygons':
                    geojson = vectorification.geojson_from_mask(
                        mask=class_mask, transform=transform, mode=mode)

                if local_geojson_path:
                    with open(local_geojson_path, 'w') as file_out:
                        file_out.write(geojson)
                        upload_or_copy(local_geojson_path, uri)

    def empty_labels(self):
        """Returns an empty SemanticSegmentationLabels object."""
        return SemanticSegmentationLabels()
