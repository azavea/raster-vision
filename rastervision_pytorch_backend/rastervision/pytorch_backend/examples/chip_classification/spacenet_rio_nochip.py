import os
from os.path import join

from rastervision.core.rv_pipeline import (ChipClassificationConfig)
from rastervision.core.data import (
    ClassConfig, RasterioSourceConfig, ChipClassificationLabelSourceConfig,
    GeoJSONVectorSourceConfig, SceneConfig, DatasetConfig)
from rastervision.pytorch_backend import (PyTorchChipClassificationConfig)
from rastervision.pytorch_learner import (
    SolverConfig, GeoDataWindowConfig, GeoDataWindowMethod,
    ClassificationGeoDataConfig, ClassificationModelConfig, Backbone)
from rastervision.pytorch_backend.examples.utils import get_scene_info

AOI_PATH = 'AOIs/AOI_1_Rio/srcData/buildingLabels/Rio_OUTLINE_Public_AOI.geojson'


def get_config(runner,
               raw_uri: str,
               processed_uri: str,
               root_uri: str,
               test: bool = False) -> ChipClassificationConfig:
    train_scene_info = get_scene_info(join(processed_uri, 'train-scenes.csv'))
    val_scene_info = get_scene_info(join(processed_uri, 'val-scenes.csv'))

    class_config = ClassConfig(names=['no_building', 'building'])

    if test:
        train_scene_info = train_scene_info[:1]
        val_scene_info = val_scene_info[:1]
    img_sz = 256

    def make_scene(scene_info) -> SceneConfig:
        (raster_uri, label_uri) = scene_info
        raster_uri = join(raw_uri, raster_uri)
        label_uri = join(processed_uri, label_uri)
        aoi_uri = join(raw_uri, AOI_PATH)

        id = os.path.splitext(os.path.basename(raster_uri))[0]
        raster_source = RasterioSourceConfig(
            channel_order=[0, 1, 2], uris=[raster_uri], persist=True)
        label_source = ChipClassificationLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=label_uri, default_class_id=1, ignore_crs_field=True),
            ioa_thresh=0.5,
            use_intersection_over_cell=False,
            pick_min_class_id=False,
            background_class_id=0,
            infer_cells=True,
            cell_sz=img_sz,
            lazy=True)

        return SceneConfig(
            id=id,
            raster_source=raster_source,
            label_source=label_source,
            aoi_uris=[aoi_uri])

    train_scenes = [make_scene(info) for info in train_scene_info]
    val_scenes = [make_scene(info) for info in val_scene_info]
    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=val_scenes)

    window_opts = {}
    for s in train_scenes:
        window_opts[s.id] = GeoDataWindowConfig(
            # method=GeoDataWindowMethod.sliding,
            method=GeoDataWindowMethod.random,
            size=img_sz,
            # stride=img_sz,
            size_lims=(200, 300),
            # h_lims=(200, 300),
            # w_lims=(200, 300),
            max_windows=3514,
        )
    for s in val_scenes:
        window_opts[s.id] = GeoDataWindowConfig(
            method=GeoDataWindowMethod.sliding, size=img_sz, stride=img_sz)

    data = ClassificationGeoDataConfig(
        scene_dataset=scene_dataset,
        window_opts=window_opts,
        img_sz=img_sz,
        num_workers=4)
    model = ClassificationModelConfig(backbone=Backbone.resnet18)
    solver = SolverConfig(lr=1e-4, num_epochs=10, batch_sz=32)

    backend = PyTorchChipClassificationConfig(
        data=data, model=model, solver=solver)

    pipeline_config = ChipClassificationConfig(
        root_uri=root_uri, dataset=scene_dataset, backend=backend)
    return pipeline_config
