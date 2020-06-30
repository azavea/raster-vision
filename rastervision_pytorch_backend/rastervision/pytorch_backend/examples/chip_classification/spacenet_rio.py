# flake8: noqa

import os
from os.path import join

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import get_scene_info, save_image_crop

aoi_path = 'AOIs/AOI_1_Rio/srcData/buildingLabels/Rio_OUTLINE_Public_AOI.geojson'


def get_config(runner, raw_uri, processed_uri, root_uri, test=False):
    debug = False
    train_scene_info = get_scene_info(join(processed_uri, 'train-scenes.csv'))
    val_scene_info = get_scene_info(join(processed_uri, 'val-scenes.csv'))
    log_tensorboard = True
    run_tensorboard = True
    class_config = ClassConfig(names=['no_building', 'building'])

    if test:
        debug = True
        train_scene_info = train_scene_info[0:1]
        val_scene_info = val_scene_info[0:1]

    def make_scene(scene_info):
        (raster_uri, label_uri) = scene_info
        raster_uri = join(raw_uri, raster_uri)
        label_uri = join(processed_uri, label_uri)
        aoi_uri = join(raw_uri, aoi_path)

        if test:
            crop_uri = join(processed_uri, 'crops',
                            os.path.basename(raster_uri))
            label_crop_uri = join(processed_uri, 'crops',
                                  os.path.basename(label_uri))

            save_image_crop(
                raster_uri,
                crop_uri,
                label_uri=label_uri,
                label_crop_uri=label_crop_uri,
                size=600,
                min_features=20,
                class_config=class_config)
            raster_uri = crop_uri
            label_uri = label_crop_uri

        id = os.path.splitext(os.path.basename(raster_uri))[0]
        raster_source = RasterioSourceConfig(
            channel_order=[0, 1, 2], uris=[raster_uri])
        label_source = ChipClassificationLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=label_uri, default_class_id=1, ignore_crs_field=True),
            ioa_thresh=0.5,
            use_intersection_over_cell=False,
            pick_min_class_id=False,
            background_class_id=0,
            infer_cells=True)

        return SceneConfig(
            id=id,
            raster_source=raster_source,
            label_source=label_source,
            aoi_uris=[aoi_uri])

    chip_sz = 200
    train_scenes = [make_scene(info) for info in train_scene_info]
    val_scenes = [make_scene(info) for info in val_scene_info]
    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=val_scenes)

    model = ClassificationModelConfig(backbone=Backbone.resnet50)
    solver = SolverConfig(
        lr=1e-4, num_epochs=20, test_num_epochs=3, batch_sz=32, one_cycle=True)
    backend = PyTorchChipClassificationConfig(
        model=model,
        solver=solver,
        log_tensorboard=log_tensorboard,
        run_tensorboard=run_tensorboard,
        test_mode=test)

    config = ChipClassificationConfig(
        root_uri=root_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz)
    return config
