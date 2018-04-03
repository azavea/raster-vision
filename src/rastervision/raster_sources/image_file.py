import rasterio

from rastervision.raster_sources.rasterio_raster_source import (
    RasterioRasterSource)
from rastervision.crs_transformers.identity_crs_transformer import (
    IdentityCRSTransformer)
from rastervision.utils.files import download_if_needed


class ImageFile(RasterioRasterSource):
    def __init__(self, raster_transformer, uri):
        self.uri = uri
        super().__init__(raster_transformer)

    def build_image_dataset(self):
        imagery_path = download_if_needed(self.uri, self.temp_dir.name)
        return rasterio.open(imagery_path)

    def get_crs_transformer(self):
        return IdentityCRSTransformer()
