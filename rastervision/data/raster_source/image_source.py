from rastervision.data.raster_source.rasterio_source import (
    RasterioRasterSource)
from rastervision.data.crs_transformer.identity_crs_transformer import (
    IdentityCRSTransformer)
from rastervision.utils.files import download_if_needed


class ImageSource(RasterioRasterSource):
    def __init__(self, uri, raster_transformers, temp_dir, channel_order=None):
        self.uri = uri
        super().__init__(raster_transformers, temp_dir, channel_order)

    def _download_data(self, temp_dir):
        return download_if_needed(self.uri, self.temp_dir)

    def _set_crs_transformer(self):
        self.crs_transformer = IdentityCRSTransformer()
