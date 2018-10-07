import rasterio

from rastervision.data.raster_source.rasterio_source import (
    RasterioRasterSource)
from rastervision.data.crs_transformer.identity_crs_transformer import (
    IdentityCRSTransformer)
from rastervision.utils.files import download_if_needed


class ImageSource(RasterioRasterSource):
    def __init__(self, uri, raster_transformers, temp_dir, channel_order=None):
        self.uri = uri
        super().__init__(raster_transformers, temp_dir, channel_order)

    def build_image_dataset(self, temp_dir):
        imagery_path = download_if_needed(self.uri, self.temp_dir)
        return rasterio.open(imagery_path)

    def get_crs_transformer(self):
        return IdentityCRSTransformer()
