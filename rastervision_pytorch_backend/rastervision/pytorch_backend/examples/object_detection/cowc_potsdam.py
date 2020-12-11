# flake8: noqa

import os
from os.path import join

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import (get_scene_info,
                                                         save_image_crop)

TRAIN_IDS = [
    '2_10', '2_11', '2_12', '2_14', '3_11', '3_13', '4_10', '5_10', '6_7',
    '6_9'
]
VAL_IDS = ['2_13', '6_8', '3_10']


def get_config(runner,
               raw_uri: str,
               processed_uri: str,
               root_uri: str,
               nochip: bool,
               test: bool = False) -> ObjectDetectionConfig:
    """Generate the pipeline config for this task. This function will be called
    by RV, with arguments from the command line, when this example is run.

    Args:
        runner (Runner): Runner for the pipeline. Will be provided by RV.
        raw_uri (str): Directory where the raw data resides
        processed_uri (str): Directory for storing processed data. 
                             E.g. crops for testing.
        root_uri (str): Directory where all the output will be written.
        nochip (bool, optional): If True, read directly from the TIFF during
            training instead of from pre-generated chips. The analyze and chip
            commands should not be run, if this is set to True. Defaults to
            False.
        test (bool, optional): If True, does the following simplifications:
            (1) Uses only the first 2 scenes
            (2) Uses only a 2000x2000 crop of the scenes
            (3) Enables test mode in the learner, which makes it use the
                test_batch_sz and test_num_epochs, and also halves the img_sz.
            Defaults to False.

    Returns:
        ObjectDetectionConfig: A pipeline config.
    """
    train_ids = TRAIN_IDS
    val_ids = VAL_IDS

    if test:
        train_ids = train_ids[:2]
        val_ids = val_ids[:2]

    chip_sz = 600
    img_sz = 256
    if nochip:
        chip_options = ObjectDetectionChipOptions()
    else:
        chip_options = ObjectDetectionChipOptions(
            neg_ratio=5.0, ioa_thresh=0.9)

    def make_scene(id: str) -> SceneConfig:
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
                size=2000,
                min_features=5)
            raster_uri = crop_uri

        raster_source = RasterioSourceConfig(
            uris=[raster_uri], channel_order=[0, 1, 2])

        vector_source = GeoJSONVectorSourceConfig(
            uri=label_uri, default_class_id=0, ignore_crs_field=True)
        label_source = ObjectDetectionLabelSourceConfig(
            vector_source=vector_source)

        return SceneConfig(
            id=id, raster_source=raster_source, label_source=label_source)

    class_config = ClassConfig(names=['vehicle'])
    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene(id) for id in train_ids],
        validation_scenes=[make_scene(id) for id in val_ids])

    if nochip:
        window_opts = {}
        # set window configs for training scenes
        for s in scene_dataset.train_scenes:
            window_opts[s.id] = ObjectDetectionGeoDataWindowConfig(
                # method=GeoDataWindowMethod.sliding,
                method=GeoDataWindowMethod.random,
                size=img_sz,
                # stride=img_sz // 2,
                size_lims=(200, 300),
                # h_lims=(200, 300),
                # w_lims=(200, 300),
                max_windows=500,
                max_sample_attempts=100,
                ioa_thresh=0.9,
                clip=True,
                neg_ratio=5.0,
                neg_ioa_thresh=0.2,
            )
        # set window configs for validation scenes
        for s in scene_dataset.validation_scenes:
            window_opts[s.id] = GeoDataWindowConfig(
                method=GeoDataWindowMethod.sliding,
                size=img_sz,
                stride=img_sz // 2)

        data = ObjectDetectionGeoDataConfig(
            scene_dataset=scene_dataset,
            window_opts=window_opts,
            img_sz=img_sz,
            num_workers=4)
    else:
        data = ObjectDetectionImageDataConfig(img_sz=img_sz, num_workers=4)

    backend = PyTorchObjectDetectionConfig(
        data=data,
        model=ObjectDetectionModelConfig(backbone=Backbone.resnet18),
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=10,
            test_num_epochs=2,
            batch_sz=16,
            one_cycle=True),
        log_tensorboard=False,
        run_tensorboard=False,
        test_mode=test)

    predict_options = ObjectDetectionPredictOptions(
        merge_thresh=0.5, score_thresh=0.9)

    pipeline = ObjectDetectionConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options,
        predict_options=predict_options)

    return pipeline
