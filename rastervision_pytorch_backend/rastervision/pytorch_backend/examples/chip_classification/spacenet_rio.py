# flake8: noqa

import os
from os.path import join

import albumentations as A

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import (get_scene_info,
                                                         save_image_crop)

aoi_path = 'AOIs/AOI_1_Rio/srcData/buildingLabels/Rio_OUTLINE_Public_AOI.geojson'

CLASS_NAMES = ['no_building', 'building']


def get_config(runner,
               raw_uri: str,
               processed_uri: str,
               root_uri: str,
               external_model: bool = False,
               external_loss: bool = False,
               nochip: bool = True,
               test: bool = False) -> ChipClassificationConfig:
    """Generate the pipeline config for this task. This function will be called
    by RV, with arguments from the command line, when this example is run.

    Args:
        runner (Runner): Runner for the pipeline. Will be provided by RV.
        raw_uri (str): Directory where the raw data resides
        processed_uri (str): Directory for storing processed data.
                             E.g. crops for testing.
        root_uri (str): Directory where all the output will be written.
        external_model (bool, optional): If True, use an external model defined
            by the ExternalModuleConfig. Defaults to False.
        external_loss (bool, optional): If True, use an external loss defined
            by the ExternalModuleConfig. Defaults to False.
        augment (bool, optional): If True, use custom data augmentation
            transforms. Some basic data augmentation is done even if this is
            False. To completely disable, specify augmentors=[] is the dat
            config. Defaults to False.
        nochip (bool, optional): If True, read directly from the TIFF during
            training instead of from pre-generated chips. The analyze and chip
            commands should not be run, if this is set to True. Defaults to
            True.
        test (bool, optional): If True, does the following simplifications:
            (1) Uses only the first 1 scene
            (2) Uses only a 600x600 crop of the scenes
            (3) Enables test mode in the learner, which makes it use the
                test_batch_sz and test_num_epochs, among other things.
            Defaults to False.

    Returns:
        ChipClassificationConfig: A pipeline config.
    """
    train_scene_info = get_scene_info(join(processed_uri, 'train-scenes.csv'))
    val_scene_info = get_scene_info(join(processed_uri, 'val-scenes.csv'))
    class_config = ClassConfig(names=CLASS_NAMES)

    if test:
        train_scene_info = train_scene_info[:1]
        val_scene_info = val_scene_info[:1]

    chip_sz = 200
    img_sz = chip_sz

    def make_scene(scene_info) -> SceneConfig:
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
                uri=label_uri,
                ignore_crs_field=True,
                transformers=[
                    ClassInferenceTransformerConfig(default_class_id=1)
                ]),
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

    train_scenes = [make_scene(info) for info in train_scene_info]
    val_scenes = [make_scene(info) for info in val_scene_info]
    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=val_scenes)

    if nochip:
        window_opts = {}
        for s in train_scenes:
            window_opts[s.id] = GeoDataWindowConfig(
                method=GeoDataWindowMethod.sliding,
                size=chip_sz,
                stride=chip_sz // 2)
        for s in val_scenes:
            window_opts[s.id] = GeoDataWindowConfig(
                method=GeoDataWindowMethod.sliding,
                size=chip_sz,
                stride=chip_sz // 2)

        data = ClassificationGeoDataConfig(
            scene_dataset=scene_dataset,
            window_opts=window_opts,
            img_sz=img_sz,
            num_workers=4)
    else:
        data = ClassificationImageDataConfig(img_sz=img_sz, num_workers=4)

    if external_model:
        model = ClassificationModelConfig(
            external_def=ExternalModuleConfig(
                github_repo='lukemelas/EfficientNet-PyTorch',
                name='efficient_net',
                entrypoint='efficientnet_b0',
                force_reload=False,
                entrypoint_kwargs={
                    'num_classes': len(class_config.names),
                    'pretrained': 'imagenet'
                }))
    else:
        model = ClassificationModelConfig(backbone=Backbone.resnet50)

    if external_loss:
        external_loss_def = ExternalModuleConfig(
            github_repo='AdeelH/pytorch-multi-class-focal-loss',
            name='focal_loss',
            entrypoint='focal_loss',
            force_reload=False,
            entrypoint_kwargs={
                'alpha': [.75, .25],
                'gamma': 2
            })
    else:
        external_loss_def = None

    solver = SolverConfig(
        lr=1e-4,
        num_epochs=20,
        test_num_epochs=4,
        batch_sz=32,
        one_cycle=True,
        external_loss_def=external_loss_def)

    backend = PyTorchChipClassificationConfig(
        data=data,
        model=model,
        solver=solver,
        test_mode=test,
        log_tensorboard=True,
        run_tensorboard=False)

    pipeline = ChipClassificationConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz)

    return pipeline
