# flake8: noqa

import os
from os.path import join

from rastervision2.core.rv_pipeline import *
from rastervision2.core.backend import *
from rastervision2.core.data import *
from rastervision2.core.analyzer import *
from rastervision2.pytorch_backend import *
from rastervision2.pytorch_learner import *
from rastervision2.examples.utils import get_scene_info, save_image_crop


def get_config(runner, test=False):
    if runner in ['inprocess']:
        raw_uri = '/opt/data/raw-data/isprs-potsdam'
        processed_uri = '/opt/data/examples/cowc-potsdam/processed-data'
        root_uri = '/opt/data/examples/cowc-potsdam/local-output'
    else:
        raw_uri = 's3://raster-vision-raw-data/isprs-potsdam'
        processed_uri = 's3://raster-vision-lf-dev/examples/cowc-potsdam/processed-data'
        root_uri = 's3://raster-vision-lf-dev/examples/cowc-potsdam/remote-output'

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
                uri=label_uri, default_class_id=0))

        return SceneConfig(
            id=id, raster_source=raster_source, label_source=label_source)

    class_config = ClassConfig(names=['vehicle'], colors=['red'])
    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene(id) for id in train_ids],
        validation_scenes=[make_scene(id) for id in val_ids])
    chip_options = ObjectDetectionChipOptions(neg_ratio=5.0, ioa_thresh=0.9)
    predict_options = ObjectDetectionPredictOptions(
        merge_thresh=0.5, score_thresh=0.9)

    backend = PyTorchObjectDetectionConfig(
        model=ObjectDetectionModelConfig(backbone='resnet18'),
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=10,
            test_num_epochs=2,
            batch_sz=16,
            one_cycle=True),
        log_tensorboard=True,
        run_tensorboard=False)

    return ObjectDetectionConfig(
        root_uri=root_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=300,
        chip_options=chip_options,
        predict_options=predict_options,
        debug=test)
