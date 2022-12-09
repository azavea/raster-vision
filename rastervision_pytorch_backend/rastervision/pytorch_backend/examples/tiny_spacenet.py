# flake8: noqa

from os.path import join
from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *


def get_config(runner) -> SemanticSegmentationConfig:
    output_root_uri = '/opt/data/output/'
    class_config = ClassConfig(
        names=['building', 'background'], colors=['red', 'black'])

    base_uri = ('https://s3.amazonaws.com/azavea-research-public-data/'
                'raster-vision/examples/spacenet')
    train_image_uri = join(base_uri, 'RGB-PanSharpen_AOI_2_Vegas_img205.tif')
    train_label_uri = join(base_uri, 'buildings_AOI_2_Vegas_img205.geojson')
    val_image_uri = join(base_uri, 'RGB-PanSharpen_AOI_2_Vegas_img25.tif')
    val_label_uri = join(base_uri, 'buildings_AOI_2_Vegas_img25.geojson')

    train_scene = make_scene('scene_205', train_image_uri, train_label_uri,
                             class_config)
    val_scene = make_scene('scene_25', val_image_uri, val_label_uri,
                           class_config)
    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[train_scene],
        validation_scenes=[val_scene])

    # Use the PyTorch backend for the SemanticSegmentation pipeline.
    chip_sz = 300

    backend = PyTorchSemanticSegmentationConfig(
        data=SemanticSegmentationGeoDataConfig(
            scene_dataset=scene_dataset,
            window_opts=GeoDataWindowConfig(
                # randomly sample training chips from scene
                method=GeoDataWindowMethod.random,
                # ... of size chip_sz x chip_sz
                size=chip_sz,
                # ... and at most 10 chips per scene
                max_windows=10)),
        model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(lr=1e-4, num_epochs=1, batch_sz=2))

    return SemanticSegmentationConfig(
        root_uri=output_root_uri,
        dataset=scene_dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz)


def make_scene(scene_id: str, image_uri: str, label_uri: str,
               class_config: ClassConfig) -> SceneConfig:
    """Define a Scene with image and labels from the given URIs."""

    raster_source = RasterioSourceConfig(
        uris=image_uri,
        # use only the first 3 bands
        channel_order=[0, 1, 2],
    )

    # configure GeoJSON reading
    vector_source = GeoJSONVectorSourceConfig(
        uri=label_uri,
        # This assumes the CRS is WGS-84 and ignores whatever the CRS specified
        # in the file is.
        ignore_crs_field=True,
        # The geoms in the label GeoJSON do not have a "class_id" property, so
        # classes must be inferred. Since all geoms are for the building class,
        # this is easy to do: we just assing the building class ID to all of
        # them.
        transformers=[
            ClassInferenceTransformerConfig(
                default_class_id=class_config.get_class_id('building'))
        ])
    # configure transformation of vector data into semantic segmentation labels
    label_source = SemanticSegmentationLabelSourceConfig(
        # semantic segmentation labels must be rasters, so rasterize the geoms
        raster_source=RasterizedSourceConfig(
            vector_source=vector_source,
            rasterizer_config=RasterizerConfig(
                # What about pixels outside of any geoms? Mark them as
                # background.
                background_class_id=class_config.get_class_id('background'))))

    return SceneConfig(
        id=scene_id,
        raster_source=raster_source,
        label_source=label_source,
    )
