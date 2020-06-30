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
    train_ids = [
        '2_10', '2_11', '2_12', '2_14', '3_11', '3_13', '4_10', '5_10', '6_7',
        '6_9'
    ]
    val_ids = ['2_13', '6_8', '3_10']

    if test:
        train_ids = train_ids[0:1]
        val_ids = val_ids[0:1]

    def make_scene(id):
        raster_uri = join(raw_uri,
                          '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(id))
        label_uri = join(processed_uri, 'labels', 'all',
                         'top_potsdam_{}_RGBIR.json'.format(id))

        if test:
            crop_uri = join(processed_uri, 'crops',
                            os.path.basename(raster_uri))
            save_image_crop(
                raster_uri,
                crop_uri,
                label_uri=label_uri,
                size=1000,
                min_features=5)
            raster_uri = crop_uri

        raster_source = RasterioSourceConfig(
            uris=[raster_uri], channel_order=[0, 1, 2])

        label_source = ObjectDetectionLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=label_uri, default_class_id=0, ignore_crs_field=True))

        return SceneConfig(
            id=id, raster_source=raster_source, label_source=label_source)

    class_config = ClassConfig(names=['vehicle'])
    chip_sz = 300
    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene(id) for id in train_ids],
        validation_scenes=[make_scene(id) for id in val_ids])
    chip_options = ObjectDetectionChipOptions(neg_ratio=5.0, ioa_thresh=0.9)
    predict_options = ObjectDetectionPredictOptions(
        merge_thresh=0.5, score_thresh=0.9)

    backend = PyTorchObjectDetectionConfig(
        model=ObjectDetectionModelConfig(backbone=Backbone.resnet18),
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
