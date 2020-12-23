from typing import Optional, List
from os.path import join

from rastervision.core.data.label_store import (LabelStoreConfig,
                                                SemanticSegmentationLabelStore)
from rastervision.pipeline.config import register_config, Config, Field


@register_config('vector_output')
class VectorOutputConfig(Config):
    """Config for vectorized semantic segmentation predictions."""
    uri: Optional[str] = Field(
        None,
        description=
        ('URI of vector output. If None, and this Config is part of a SceneConfig and '
         'RVPipeline, this field will be auto-generated.'))
    class_id: int = Field(
        ...,
        description='The prediction class that is to turned into vectors.')
    denoise: int = Field(
        0,
        description=
        ('Radius of the structural element used to remove high-frequency signals from '
         'the image.'))

    def update(self, pipeline=None, scene=None):
        if pipeline and scene:
            mode = self.get_mode()
            class_id = self.class_id
            filename = f'{mode}-{class_id}.json'
            self.uri = join(pipeline.predict_uri, scene.id, 'vector_output',
                            filename)

    def get_mode(self):
        raise NotImplementedError()


@register_config('polygon_vector_output')
class PolygonVectorOutputConfig(VectorOutputConfig):
    """Config for vectorized semantic segmentation predictions."""

    def get_mode(self):
        return 'polygons'


@register_config('building_vector_output')
class BuildingVectorOutputConfig(VectorOutputConfig):
    """Config for vectorized semantic segmentation predictions.

    Intended to break up clusters of buildings.
    """
    min_aspect_ratio: float = Field(
        1.618,
        description=
        ('Ratio between length and height (or height and length) of anything that can '
         'be considered to be a cluster of buildings. The goal is to distinguish between '
         'rows of buildings and (say) a single building.'))
    min_area: float = Field(
        0.0,
        description=
        ('Minimum area of anything that can be considered to be a cluster of buildings. '
         'The goal is to distinguish between buildings and artifacts.'))
    element_width_factor: float = Field(
        0.5,
        description=
        ('Width of the structural element used to break building clusters as a fraction '
         'of the width of the cluster.'))
    element_thickness: float = Field(
        0.001,
        description=
        ('Thickness of the structural element that is used to break building clusters.'
         ))

    def get_mode(self):
        return 'buildings'


@register_config('semantic_segmentation_label_store')
class SemanticSegmentationLabelStoreConfig(LabelStoreConfig):
    """Config for storage for semantic segmentation predictions.

    Stores class raster as GeoTIFF, and can optionally vectorizes predictions and stores
    them in GeoJSON files.
    """
    uri: Optional[str] = Field(
        None,
        description=(
            'URI of file with predictions. If None, and this Config is part of '
            'a SceneConfig inside an RVPipelineConfig, this fiend will be '
            'auto-generated.'))
    vector_output: List[VectorOutputConfig] = []
    rgb: bool = Field(
        False,
        description=
        ('If True, save prediction class_ids in RGB format using the colors in '
         'class_config.'))
    smooth_output: bool = Field(
        False,
        description='If True, expects labels to be continuous values '
        'representing class scores and stores both scores and discrete '
        'labels.')
    smooth_as_uint8: bool = Field(
        False,
        description='If True, stores smooth scores as uint8, resulting in '
        'loss of precision, but reduced file size. Only used if '
        'smooth_output=True.')
    rasterio_block_size: int = Field(
        256,
        description='blockxsize and blockysize params in rasterio.open() will '
        'be set to this.')

    def build(self, class_config, crs_transformer, extent, tmp_dir):
        class_config.ensure_null_class()

        label_store = SemanticSegmentationLabelStore(
            self.uri,
            extent,
            crs_transformer,
            tmp_dir,
            vector_outputs=self.vector_output,
            class_config=class_config,
            save_as_rgb=self.rgb,
            smooth_output=self.smooth_output,
            smooth_as_uint8=self.smooth_as_uint8,
            rasterio_block_size=self.rasterio_block_size)

        return label_store

    def update(self, pipeline=None, scene=None):
        if pipeline is not None and scene is not None:
            if self.uri is None:
                self.uri = join(pipeline.predict_uri, f'{scene.id}')

        for vo in self.vector_output:
            vo.update(pipeline, scene)
