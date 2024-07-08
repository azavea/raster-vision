import os
from os.path import join

from rastervision.core.rv_pipeline import (
    ObjectDetectionConfig, ObjectDetectionChipOptions,
    ObjectDetectionPredictOptions, ObjectDetectionWindowSamplingConfig,
    WindowSamplingMethod)
from rastervision.core.data import (
    ClassConfig, ClassInferenceTransformerConfig, DatasetConfig,
    GeoJSONVectorSourceConfig, ObjectDetectionLabelSourceConfig,
    RasterioSourceConfig, SceneConfig)
from rastervision.pytorch_backend import PyTorchObjectDetectionConfig
from rastervision.pytorch_learner import (
    Backbone, ObjectDetectionGeoDataConfig, ObjectDetectionImageDataConfig,
    ObjectDetectionModelConfig, SolverConfig)
from rastervision.pytorch_backend.examples.utils import (get_scene_info,
                                                         save_image_crop)


def get_config(runner,
               raw_uri: str,
               processed_uri: str,
               root_uri: str,
               nochip: bool = True,
               test: bool = False) -> ObjectDetectionConfig:
    """Generate the pipeline config for this task. This function will be called
    by RV, with arguments from the command line, when this example is run.

    Args:
        runner (Runner): Runner for the pipeline. Will be provided by RV.
        raw_uri (str): Directory where the raw data resides
        processed_uri (str): Directory for storing processed data.
                             E.g. crops for testing.
        root_uri (str): Directory where all the output will be written.
        nochip (bool): If True, read directly from the TIFF during
            training instead of from pre-generated chips. The analyze and chip
            commands should not be run, if this is set to True. Defaults to
            False.
        test (bool): If True, does the following simplifications:
            (1) Uses only the first 2 scenes.
            (2) Uses only a 2000x2000 crop of the scenes.
            (3) Trains for only 2 epochs.
            Defaults to False.

    Returns:
        ObjectDetectionConfig: A pipeline config.
    """
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
            save_image_crop(raster_uri, crop_uri, size=2000, min_features=5)
            raster_uri = crop_uri

        id = os.path.splitext(os.path.basename(raster_uri))[0]

        raster_source = RasterioSourceConfig(
            uris=[raster_uri], channel_order=[0, 1, 2])

        label_source = ObjectDetectionLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uris=label_uri,
                transformers=[
                    ClassInferenceTransformerConfig(default_class_id=0)
                ]))

        return SceneConfig(
            id=id, raster_source=raster_source, label_source=label_source)

    train_scenes = [make_scene(info) for info in train_scene_info]
    val_scenes = [make_scene(info) for info in val_scene_info]
    class_config = ClassConfig(names=['vehicle'], colors=['red'])
    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=val_scenes)

    chip_sz = 300
    img_sz = chip_sz

    chip_options = ObjectDetectionChipOptions(
        sampling=ObjectDetectionWindowSamplingConfig(
            method=WindowSamplingMethod.random,
            size=chip_sz,
            size_lims=(chip_sz, chip_sz + 1),
            max_windows=200,
            clip=True,
            neg_ratio=1.0,
            ioa_thresh=0.8))

    if nochip:
        data = ObjectDetectionGeoDataConfig(
            scene_dataset=scene_dataset,
            sampling=chip_options.sampling,
            img_sz=img_sz,
            augmentors=[])
    else:
        data = ObjectDetectionImageDataConfig(img_sz=img_sz, num_workers=4)

    backend = PyTorchObjectDetectionConfig(
        data=data,
        model=ObjectDetectionModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=10 if not test else 2,
            batch_sz=16,
            one_cycle=True,
        ),
        log_tensorboard=True,
        run_tensorboard=False,
    )

    predict_options = ObjectDetectionPredictOptions(
        chip_sz=chip_sz, merge_thresh=0.1, score_thresh=0.5)

    return ObjectDetectionConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        chip_options=chip_options,
        predict_options=predict_options)
