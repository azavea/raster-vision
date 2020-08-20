from typing import List, Optional, Tuple

from rastervision.pipeline.config import Config, register_config, Field
from rastervision.core.data.raster_transformer import RasterTransformerConfig
from rastervision.core.data.raster_source import RasterSourceConfig, MultiRasterSource


@register_config('raster_source')
class MultiRasterSourceConfig(RasterSourceConfig):
	# TODO add descriptions
	raster_source_configs: List[Tuple[RasterSourceConfig, tuple]] = Field([], description='')
	raw_channel_order: Optional[List[int]] = Field(None, description='')

	def update(self):
		self.raw_channel_order = sum(list(cfg[1]) for cfg in self.raster_source_configs], [])

    def build(self, tmp_dir, use_transformers=True):
        raster_transformers = ([rt.build() for rt in self.transformers]
                               if use_transformers else [])

		raster_sources = [cfg.build(tmp_dir, use_transformers) for cfg in self.raster_source_configs]
		multi_raster_source = MultiRasterSource(raster_sources, raster_transformers)
		return multi_raster_source

    def update(self, pipeline=None, scene=None):
        for t in self.transformers:
            t.update(pipeline, scene)
