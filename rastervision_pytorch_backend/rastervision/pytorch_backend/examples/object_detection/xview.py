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


def get_config(runner, raw_uri, processed_uri, root_uri, test=False):
    train_scene_info = get_scene_info(join(processed_uri, 'train-scenes.csv'))
    val_scene_info = get_scene_info(join(processed_uri, 'val-scenes.csv'))
    if test:
        train_scene_info = train_scene_info[0:1]
        val_scene_info = val_scene_info[0:1]

    def make_scene(scene_info):
        (raster_uri, label_uri) = scene_info
        raster_uri = join(raw_uri, raster_uri)
        label_uri = join(processed_uri, label_uri)

        if test:
            crop_uri = join(processed_uri, 'crops',
                            os.path.basename(raster_uri))
            save_image_crop(raster_uri, crop_uri, size=600, min_features=5)
            raster_uri = crop_uri

        id = os.path.splitext(os.path.basename(raster_uri))[0]

        raster_source = RasterioSourceConfig(
            uris=[raster_uri], channel_order=[0, 1, 2])

        label_source = ObjectDetectionLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=label_uri, default_class_id=0, ignore_crs_field=True))

        return SceneConfig(
            id=id, raster_source=raster_source, label_source=label_source)

    train_scenes = [make_scene(info) for info in train_scene_info]
    val_scenes = [make_scene(info) for info in val_scene_info]
    class_config = ClassConfig(names=['vehicle'], colors=['red'])
    chip_sz = 300
    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=val_scenes)
    chip_options = ObjectDetectionChipOptions(neg_ratio=1.0, ioa_thresh=0.8)
    predict_options = ObjectDetectionPredictOptions(
        merge_thresh=0.1, score_thresh=0.5)

    backend = PyTorchObjectDetectionConfig(
        model=ObjectDetectionModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=10,
            test_num_epochs=2,
            batch_sz=16,
            one_cycle=True),
        log_tensorboard=True,
        run_tensorboard=False,
        test_mode=test)

    return ObjectDetectionConfig(
        root_uri=root_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options,
        predict_options=predict_options)
