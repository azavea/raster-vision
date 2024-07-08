import os
from os.path import join

from rastervision.core.rv_pipeline import (
    ObjectDetectionConfig, ObjectDetectionChipOptions,
    ObjectDetectionPredictOptions, WindowSamplingMethod,
    ObjectDetectionWindowSamplingConfig)
from rastervision.core.data import (
    ClassConfig, ClassInferenceTransformerConfig, DatasetConfig,
    GeoJSONVectorSourceConfig, ObjectDetectionLabelSourceConfig,
    RasterioSourceConfig, SceneConfig)
from rastervision.pytorch_backend import PyTorchObjectDetectionConfig
from rastervision.pytorch_learner import (
    Backbone, ExternalModuleConfig, ObjectDetectionGeoDataConfig,
    ObjectDetectionImageDataConfig, ObjectDetectionModelConfig, PlotOptions,
    SolverConfig)
from rastervision.pytorch_backend.examples.utils import save_image_crop

TRAIN_IDS = [
    '2_10', '2_11', '2_12', '2_14', '3_11', '3_13', '4_10', '5_10', '6_7',
    '6_9'
]
VAL_IDS = ['2_13', '6_8', '3_10']


def get_config(runner,
               raw_uri: str,
               processed_uri: str,
               root_uri: str,
               nochip: bool = True,
               multiband: bool = False,
               external_model: bool = False,
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
        multiband (bool): If True, all 4 channels (R, G, B, & IR)
            available in the raster source will be used. If False, only
            IR, R, G (in that order) will be used. Defaults to False.
        external_model (bool): If True, use an external model defined
            by the ExternalModuleConfig. Defaults to True.
        test (bool): If True, does the following simplifications:
            (1) Uses only the first 2 scenes
            (2) Uses only a 2000x2000 crop of the scenes
            (3) Trains for only 2 epochs.
            Defaults to False.

    Returns:
        ObjectDetectionConfig: A pipeline config.
    """
    train_ids = TRAIN_IDS
    val_ids = VAL_IDS

    chip_sz = 300
    img_sz = chip_sz
    chips_per_scene = 100

    if test:
        train_ids = train_ids[:2]
        val_ids = val_ids[:2]
        chips_per_scene = 10

    if multiband:
        channel_order = [0, 1, 2, 3]
        channel_display_groups = {'RGB': [0, 1, 2], 'IR': [3]}
    else:
        channel_order = [0, 1, 2]
        channel_display_groups = None

    def make_scene(id: str) -> SceneConfig:
        raster_uri = join(raw_uri, f'4_Ortho_RGBIR/top_potsdam_{id}_RGBIR.tif')
        label_uri = join(processed_uri, 'labels', 'all',
                         f'top_potsdam_{id}_RGBIR.json')

        if test:
            crop_uri = join(processed_uri, 'crops',
                            os.path.basename(raster_uri))
            save_image_crop(
                raster_uri,
                crop_uri,
                label_uri=label_uri,
                vector_labels=True,
                size=2000,
                min_features=5,
                default_class_id=0)
            raster_uri = crop_uri

        raster_source = RasterioSourceConfig(
            uris=[raster_uri], channel_order=channel_order)

        vector_source = GeoJSONVectorSourceConfig(
            uris=label_uri,
            transformers=[ClassInferenceTransformerConfig(default_class_id=0)])
        label_source = ObjectDetectionLabelSourceConfig(
            vector_source=vector_source)

        return SceneConfig(
            id=id, raster_source=raster_source, label_source=label_source)

    class_config = ClassConfig(names=['vehicle'], colors=['red'])
    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene(id) for id in train_ids],
        validation_scenes=[make_scene(id) for id in val_ids])

    window_sampling_opts = ObjectDetectionWindowSamplingConfig(
        method=WindowSamplingMethod.random,
        size=chip_sz,
        size_lims=(chip_sz, chip_sz + 1),
        max_windows=chips_per_scene,
        max_sample_attempts=100,
        clip=True,
        neg_ratio=5.0,
        ioa_thresh=0.9,
        neg_ioa_thresh=0.2)

    chip_options = ObjectDetectionChipOptions(sampling=window_sampling_opts)
    if nochip:
        data = ObjectDetectionGeoDataConfig(
            scene_dataset=scene_dataset,
            sampling=window_sampling_opts,
            img_sz=img_sz,
            num_workers=4,
            plot_options=PlotOptions(
                channel_display_groups=channel_display_groups))
    else:
        data = ObjectDetectionImageDataConfig(
            img_sz=img_sz,
            num_workers=4,
            plot_options=PlotOptions(
                channel_display_groups=channel_display_groups))

    if external_model:
        """This demonstrates how to use an external model for object detection,
        but to successfully use this functionality with different settings, the
        following things should be kept in mind:

          (1) Torchvision does not expose its object detection models via
            torch hub (https://github.com/pytorch/vision/issues/1945). So, to
            use those, you will need to fork the torchvision repo and manually
            add those models or corresponding factory functions to hubconf.py.
            Example: github.com/AdeelH/vision/blob/det_hubconf_0.10/hubconf.py.
            Further, you should ensure that the branch of the fork is the same
            version as the version in Raster Vision's Docker image; or, if
            using outside Docker, it should match the version of the local
            torchvision installation.
          (2) The external model should behave exactly like torchvision
            object detection models. This includes, but might not be limited
            to:
                - Accepting targets as dicts with keys: 'boxes' and 'labels'.
                - Accepting 1-indexed class labels.
                - Computing losses internally and returning them in a dict
                during training.
                - Returning predictions as dicts with keys: 'boxes', 'labels',
                and 'scores'.
        """

        model = ObjectDetectionModelConfig(
            external_def=ExternalModuleConfig(
                github_repo='AdeelH/vision:det_hubconf_0.12',
                name='ssd',
                entrypoint='ssd300_vgg16',
                force_reload=True,
                entrypoint_kwargs={
                    # torchvision OD models need add an additional null class,
                    # so +1 is needed here
                    'num_classes': len(class_config.names) + 1,
                    'pretrained': False,
                    'pretrained_backbone': True
                }))
    else:
        model = ObjectDetectionModelConfig(backbone=Backbone.resnet18)

    backend = PyTorchObjectDetectionConfig(
        data=data,
        model=model,
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=10 if not test else 2,
            batch_sz=16,
            one_cycle=True,
        ),
        log_tensorboard=False,
        run_tensorboard=False,
    )

    predict_options = ObjectDetectionPredictOptions(
        chip_sz=chip_sz, merge_thresh=0.5, score_thresh=0.9)

    pipeline = ObjectDetectionConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        chip_options=chip_options,
        predict_options=predict_options)

    return pipeline
