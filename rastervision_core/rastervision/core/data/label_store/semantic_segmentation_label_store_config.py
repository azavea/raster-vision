from typing import TYPE_CHECKING, Iterator, List, Optional
from os.path import join

from rastervision.pipeline.config import register_config, Config, Field
from rastervision.core.data.label_store import (LabelStoreConfig,
                                                SemanticSegmentationLabelStore)
from rastervision.core.data.utils import (denoise, mask_to_building_polygons,
                                          mask_to_polygons)

if TYPE_CHECKING:
    import numpy as np
    from shapely.geometry.base import BaseGeometry

    from rastervision.core.box import Box
    from rastervision.core.data import (ClassConfig, CRSTransformer,
                                        SceneConfig)
    from rastervision.core.rv_pipeline import RVPipelineConfig


def vo_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 8:
        try:
            # removed in version 9
            del cfg_dict['uri']
        except KeyError:
            pass
    return cfg_dict


@register_config('vector_output', upgrader=vo_config_upgrader)
class VectorOutputConfig(Config):
    """Config for vectorized semantic segmentation predictions."""
    class_id: int = Field(
        ...,
        description='The prediction class that is to turned into vectors.')
    denoise: int = Field(
        8,
        description='Diameter of the circular structural element used to '
        'remove high-frequency signals from the image. Smaller values will '
        'reduce less noise and make vectorization slower and more memory '
        'intensive (especially for large images). Larger values will remove '
        'more noise and make vectorization faster but might also remove '
        'legitimate detections.')

    def vectorize(self, mask: 'np.ndarray') -> Iterator['BaseGeometry']:
        """Vectorize binary mask representing the target class into polygons.
        """
        raise NotImplementedError()

    def get_uri(self, root: str,
                class_config: Optional['ClassConfig'] = None) -> str:
        if class_config is not None:
            class_name = class_config.get_name(self.class_id)
            uri = join(root, f'class-{self.class_id}-{class_name}.json')
        else:
            uri = join(root, f'class-{self.class_id}.json')
        return uri


@register_config('polygon_vector_output')
class PolygonVectorOutputConfig(VectorOutputConfig):
    """Config for vectorized semantic segmentation predictions."""

    def vectorize(self, mask: 'np.ndarray') -> Iterator['BaseGeometry']:
        if self.denoise > 0:
            mask = denoise(mask, self.denoise)
        return mask_to_polygons(mask)


def building_vo_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 6:
        try:
            # removed in version 7
            del cfg_dict['min_aspect_ratio']
        except KeyError:
            pass
    return cfg_dict


@register_config(
    'building_vector_output', upgrader=building_vo_config_upgrader)
class BuildingVectorOutputConfig(VectorOutputConfig):
    """Config for vectorized semantic segmentation predictions.

    Intended to break up clusters of buildings.
    """
    min_area: float = Field(
        0.0,
        description='Minimum area (in pixels^2) of anything that can be '
        'considered to be a building or a cluster of buildings. The goal is '
        'to distinguish between buildings and artifacts.')
    element_width_factor: float = Field(
        0.5,
        description='Width of the structural element used to break building '
        'clusters as a fraction of the width of the cluster.')
    element_thickness: float = Field(
        0.001,
        description='Thickness of the structural element that is used to '
        'break building clusters.')

    def vectorize(self, mask: 'np.ndarray') -> Iterator['BaseGeometry']:
        if self.denoise > 0:
            mask = denoise(mask, self.denoise)
        polygons = mask_to_building_polygons(
            mask=mask,
            min_area=self.min_area,
            width_factor=self.element_width_factor,
            thickness=self.element_thickness)
        return polygons


@register_config('semantic_segmentation_label_store')
class SemanticSegmentationLabelStoreConfig(LabelStoreConfig):
    """Configure a :class:`.SemanticSegmentationLabelStore`.

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

    def build(self,
              class_config: 'ClassConfig',
              crs_transformer: 'CRSTransformer',
              bbox: 'Box',
              tmp_dir: Optional[str] = None) -> SemanticSegmentationLabelStore:
        class_config.ensure_null_class()

        label_store = SemanticSegmentationLabelStore(
            uri=self.uri,
            crs_transformer=crs_transformer,
            class_config=class_config,
            bbox=bbox,
            tmp_dir=tmp_dir,
            vector_outputs=self.vector_output,
            save_as_rgb=self.rgb,
            smooth_output=self.smooth_output,
            smooth_as_uint8=self.smooth_as_uint8,
            rasterio_block_size=self.rasterio_block_size)

        return label_store

    def update(self,
               pipeline: Optional['RVPipelineConfig'] = None,
               scene: Optional['SceneConfig'] = None):
        if pipeline is not None and scene is not None:
            if self.uri is None:
                self.uri = join(pipeline.predict_uri, f'{scene.id}')
