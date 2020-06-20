import numpy as np
import rasterio

from rastervision.pipeline.file_system import (
    get_local_path, make_dir, upload_or_copy, file_exists, str_to_file)
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.label_store import LabelStore
from rastervision.core.data.label_source import SegmentationClassTransformer
from rastervision.core.data.raster_source import RasterioSourceConfig


class SemanticSegmentationLabelStore(LabelStore):
    """Storage for semantic segmentation predictions.

    Stores class raster as GeoTIFF, and can optionally vectorizes predictions and stores
    them in GeoJSON files.
    """

    def __init__(self,
                 uri,
                 extent,
                 crs_transformer,
                 tmp_dir,
                 vector_output=None,
                 class_config=None):
        """Constructor.

        Args:
            uri: (str) URI of GeoTIFF file used for storing predictions as RGB values
            extent: (Box) The extent of the scene
            crs_transformer: (CRSTransformer)
            tmp_dir: (str) temp directory to use
            vector_output: (None or array of VectorOutputConfig) containing
                vectorifiction configuration information
            class_config: (ClassConfig) with color values used to convert
                class ids to RGB value
        """
        self.uri = uri
        self.vector_output = vector_output
        self.extent = extent
        self.crs_transformer = crs_transformer
        self.tmp_dir = tmp_dir
        # Note: can't name this class_transformer due to Python using that attribute
        if class_config:
            self.class_trans = SegmentationClassTransformer(class_config)
        else:
            self.class_trans = None

        self.source = None
        if file_exists(uri):
            self.source = RasterioSourceConfig(uris=[uri]).build(tmp_dir)

    def _subcomponents_to_activate(self):
        if self.source is not None:
            return [self.source]
        return []

    def get_labels(self):
        """Get all labels.

        Returns:
            SemanticSegmentationLabels
        """
        if self.source is None:
            raise Exception('Raster source at {} does not exist'.format(
                self.uri))

        labels = SemanticSegmentationLabels()
        extent = self.source.get_extent()
        raw_labels = self.source.get_raw_chip(extent)
        label_arr = (np.squeeze(raw_labels) if self.class_trans is None else
                     self.class_trans.rgb_to_class(raw_labels))
        labels.set_label_arr(extent, label_arr)
        return labels

    def save(self, labels):
        """Save.

        Args:
            labels - (SemanticSegmentationLabels) labels to be saved
        """
        local_path = get_local_path(self.uri, self.tmp_dir)
        make_dir(local_path, use_dirname=True)

        transform = self.crs_transformer.get_affine_transform()
        crs = self.crs_transformer.get_image_crs()

        band_count = 1
        dtype = np.uint8
        if self.class_trans:
            band_count = 3

        mask = (np.zeros((self.extent.ymax, self.extent.xmax), dtype=np.uint8)
                if self.vector_output else None)

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
            for window in labels.get_windows():
                label_arr = labels.get_label_arr(window)
                window = window.intersection(self.extent)
                label_arr = label_arr[0:window.get_height(), 0:
                                      window.get_width()]

                if mask is not None:
                    mask[window.ymin:window.ymax, window.xmin:
                         window.xmax] = label_arr

                window = window.rasterio_format()
                if self.class_trans:
                    rgb_labels = self.class_trans.class_to_rgb(label_arr)
                    for chan in range(3):
                        dataset.write_band(
                            chan + 1, rgb_labels[:, :, chan], window=window)
                else:
                    img = label_arr.astype(dtype)
                    dataset.write_band(1, img, window=window)

        upload_or_copy(local_path, self.uri)

        if self.vector_output:
            import mask_to_polygons.vectorification as vectorification
            import mask_to_polygons.processing.denoise as denoise

            for vo in self.vector_output:
                denoise_radius = vo.denoise
                uri = vo.uri
                mode = vo.get_mode()
                class_id = vo.class_id
                class_mask = np.array(mask == class_id, dtype=np.uint8)

                def transform(x, y):
                    return self.crs_transformer.pixel_to_map((x, y))

                if denoise_radius > 0:
                    class_mask = denoise.denoise(class_mask, denoise_radius)

                if uri and mode == 'buildings':
                    geojson = vectorification.geojson_from_mask(
                        mask=class_mask,
                        transform=transform,
                        mode=mode,
                        min_aspect_ratio=vo.min_aspect_ratio,
                        min_area=vo.min_area,
                        width_factor=vo.element_width_factor,
                        thickness=vo.element_thickness)
                elif uri and mode == 'polygons':
                    geojson = vectorification.geojson_from_mask(
                        mask=class_mask, transform=transform, mode=mode)
                str_to_file(geojson, uri)

    def empty_labels(self):
        """Returns an empty SemanticSegmentationLabels object."""
        return SemanticSegmentationLabels()
