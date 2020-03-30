# flake8: noqa

from os.path import join

from rastervision2.core.rv_pipeline import *
from rastervision2.core.backend import *
from rastervision2.core.data import *
from rastervision2.pytorch_backend import *
from rastervision2.pytorch_learner import *


def get_config(runner):
    root_uri = '/opt/data/output/'
    base_uri = ('https://s3.amazonaws.com/azavea-research-public-data/'
                'raster-vision/examples/spacenet')
    train_image_uri = '{}/RGB-PanSharpen_AOI_2_Vegas_img205.tif'.format(base_uri)
    train_label_uri = '{}/buildings_AOI_2_Vegas_img205.geojson'.format(base_uri)
    val_image_uri = '{}/RGB-PanSharpen_AOI_2_Vegas_img25.tif'.format(base_uri)
    val_label_uri = '{}/buildings_AOI_2_Vegas_img25.geojson'.format(base_uri)
    channel_order = [0, 1, 2]
    class_config = ClassConfig(
        names=['building', 'background'],
        colors=['red', 'black'])

    def make_scene(scene_id, image_uri, label_uri):
        """
        - StatsTransformer is used to convert uint16 values to uint8.
        - The GeoJSON does not have a class_id property for each geom,
          so it is inferred as 0 (ie. building) because the default_class_id
          is set to 0.
        - The labels are in the form of GeoJSON which needs to be rasterized
          to use as label for semantic segmentation, so we use a RasterizedSource.
        - The rasterizer set the background (as opposed to foreground) pixels
          to 1 because background_class_id is set to 1.
        """
        raster_source = RasterioSourceConfig(
            uris=[image_uri], channel_order=channel_order,
            transformers=[StatsTransformerConfig()])
        label_source = SemanticSegmentationLabelSourceConfig(
            raster_source=RasterizedSourceConfig(
                vector_source=GeoJSONVectorSourceConfig(
                    uri=label_uri, default_class_id=0),
                rasterizer_config=RasterizerConfig(background_class_id=1)
            ))
        return SceneConfig(
            id=scene_id,
            raster_source=raster_source,
            label_source=label_source)

    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene('scene_205', train_image_uri, train_label_uri)],
        validation_scenes=[make_scene('scene_25', val_image_uri, val_label_uri)])

    # Use the PyTorch backend for the SemanticSegmentation pipeline.
    train_chip_sz = 300
    backend = PyTorchSemanticSegmentationConfig(
        model=SemanticSegmentationModelConfig(backbone='resnet50'),
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=1,
            batch_sz=2))
    chip_options = SemanticSegmentationChipOptions(
        window_method='random_sample', chips_per_scene=10)

    return SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=train_chip_sz,
        chip_options=chip_options,
        debug=False)
