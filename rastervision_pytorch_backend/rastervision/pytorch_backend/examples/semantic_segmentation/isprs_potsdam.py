from typing import Optional
from os.path import join, basename

import albumentations as A

from rastervision.core.rv_pipeline import (
    SemanticSegmentationConfig, SemanticSegmentationChipOptions,
    SemanticSegmentationPredictOptions, WindowSamplingConfig,
    WindowSamplingMethod)
from rastervision.core.data import (
    ClassConfig, DatasetConfig, PolygonVectorOutputConfig,
    RasterioSourceConfig, RGBClassTransformerConfig, SceneConfig,
    SemanticSegmentationLabelSourceConfig,
    SemanticSegmentationLabelStoreConfig)
from rastervision.pytorch_backend import PyTorchSemanticSegmentationConfig
from rastervision.pytorch_learner import (
    Backbone, ExternalModuleConfig, PlotOptions, SolverConfig,
    SemanticSegmentationGeoDataConfig, SemanticSegmentationImageDataConfig,
    SemanticSegmentationModelConfig)
from rastervision.pytorch_backend.examples.utils import save_image_crop
from rastervision.pytorch_backend.examples.semantic_segmentation.utils import (
    example_multiband_transform, example_rgb_transform, imagenet_stats,
    Unnormalize)

TRAIN_IDS = [
    '2_10', '2_11', '3_10', '3_11', '4_10', '4_11', '4_12', '5_10', '5_11',
    '5_12', '6_10', '6_11', '6_7', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8',
    '7_9'
]
VAL_IDS = ['2_12', '3_12', '6_12']

CLASS_NAMES = [
    'Car', 'Building', 'Low Vegetation', 'Tree', 'Impervious', 'Clutter'
]
CLASS_COLORS = [
    '#ffff00', '#0000ff', '#00ffff', '#00ff00', '#ffffff', '#ff0000'
]


def get_config(runner,
               raw_uri: str,
               root_uri: str,
               processed_uri: Optional[str] = None,
               multiband: bool = False,
               external_model: bool = True,
               augment: bool = False,
               nochip: bool = True,
               num_epochs: int = 10,
               batch_sz: int = 8,
               test: bool = False) -> SemanticSegmentationConfig:
    """Generate the pipeline config for this task. This function will be called
    by RV, with arguments from the command line, when this example is run.

    Args:
        runner (Runner): Runner for the pipeline. Will be provided by RV.
        raw_uri (str): Directory where the raw data resides
        root_uri (str): Directory where all the output will be written.
        processed_uri (str): Directory for storing processed data.
                             E.g. crops for testing. Defaults to None.
        multiband (bool, optional): If True, all 4 channels (R, G, B, & IR)
            available in the raster source will be used. If False, only
            IR, R, G (in that order) will be used. Defaults to False.
        external_model (bool, optional): If True, use an external model defined
            by the ExternalModuleConfig. Defaults to True.
        augment (bool, optional): If True, use custom data augmentation
            transforms. Some basic data augmentation is done even if this is
            False. To completely disable, specify augmentors=[] is the dat
            config. Defaults to False.
        nochip (bool, optional): If True, read directly from the TIFF during
            training instead of from pre-generated chips. The analyze and chip
            commands should not be run, if this is set to True. Defaults to
            True.
        num_epochs (int): Number of epochs to train for.
        batch_sz (int): Batch size.
        test (bool, optional): If True, does the following simplifications:
            (1) Uses only the first 2 scenes
            (2) Uses only a 600x600 crop of the scenes
            (3) Trains for only 2 epochs and uses a batch size of 2.
            Defaults to False.

    Returns:
        SemanticSegmentationConfig: A pipeline config.
    """
    train_ids = TRAIN_IDS
    val_ids = VAL_IDS

    chip_sz = 300
    img_sz = chip_sz
    num_epochs = int(num_epochs)
    batch_sz = int(batch_sz)

    if test:
        train_ids = train_ids[:2]
        val_ids = val_ids[:2]
        num_epochs = 2
        batch_sz = 2

    if multiband:
        # use all 4 channels
        channel_order = [0, 1, 2, 3]
        channel_display_groups = {'RGB': (0, 1, 2), 'IR': (3, )}
        aug_transform = example_multiband_transform
    else:
        # use infrared, red, & green channels only
        channel_order = [3, 0, 1]
        channel_display_groups = None
        aug_transform = example_rgb_transform

    if augment:
        mu, std = imagenet_stats['mean'], imagenet_stats['std']
        mu, std = mu[channel_order], std[channel_order]

        base_transform = A.Normalize(mean=mu.tolist(), std=std.tolist())
        plot_transform = Unnormalize(mean=mu, std=std)

        aug_transform = A.to_dict(aug_transform)
        base_transform = A.to_dict(base_transform)
        plot_transform = A.to_dict(plot_transform)
    else:
        aug_transform = None
        base_transform = None
        plot_transform = None

    class_config = ClassConfig(names=CLASS_NAMES, colors=CLASS_COLORS)

    def make_scene(id) -> SceneConfig:
        id = id.replace('-', '_')
        raster_uri = join(raw_uri, f'4_Ortho_RGBIR/top_potsdam_{id}_RGBIR.tif')
        label_uri = join(
            raw_uri, f'5_Labels_for_participants/top_potsdam_{id}_label.tif')

        if test:
            crop_uri = join(processed_uri, 'crops', basename(raster_uri))
            label_crop_uri = join(processed_uri, 'crops', basename(label_uri))
            save_image_crop(
                raster_uri,
                crop_uri,
                label_uri=label_uri,
                label_crop_uri=label_crop_uri,
                size=600,
                vector_labels=False)
            raster_uri = crop_uri
            label_uri = label_crop_uri

        raster_source = RasterioSourceConfig(
            uris=[raster_uri], channel_order=channel_order)

        # Using with_rgb_class_map because label TIFFs have classes encoded as
        # RGB colors.
        label_source = SemanticSegmentationLabelSourceConfig(
            raster_source=RasterioSourceConfig(
                uris=[label_uri],
                transformers=[
                    RGBClassTransformerConfig(class_config=class_config)
                ]))

        # URI will be injected by scene config.
        # Using rgb=True because we want prediction TIFFs to be in
        # RGB format.
        label_store = SemanticSegmentationLabelStoreConfig(
            rgb=True, vector_output=[PolygonVectorOutputConfig(class_id=0)])

        scene = SceneConfig(
            id=id,
            raster_source=raster_source,
            label_source=label_source,
            label_store=label_store)

        return scene

    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene(id) for id in train_ids],
        validation_scenes=[make_scene(id) for id in val_ids])

    window_sampling_opts = {}
    # set window configs for training scenes
    for s in scene_dataset.train_scenes:
        window_sampling_opts[s.id] = WindowSamplingConfig(
            method=WindowSamplingMethod.sliding, size=chip_sz, stride=chip_sz)

    # set window configs for validation scenes
    for s in scene_dataset.validation_scenes:
        window_sampling_opts[s.id] = WindowSamplingConfig(
            method=WindowSamplingMethod.sliding, size=chip_sz, stride=chip_sz)

    chip_options = SemanticSegmentationChipOptions(
        sampling=window_sampling_opts)

    if nochip:
        data = SemanticSegmentationGeoDataConfig(
            scene_dataset=scene_dataset,
            sampling=window_sampling_opts,
            img_sz=img_sz,
            img_channels=len(channel_order),
            num_workers=4,
            base_transform=base_transform,
            aug_transform=aug_transform,
            plot_options=PlotOptions(
                transform=plot_transform,
                channel_display_groups=channel_display_groups))
    else:
        data = SemanticSegmentationImageDataConfig(
            img_sz=img_sz,
            num_workers=4,
            base_transform=base_transform,
            aug_transform=aug_transform,
            plot_options=PlotOptions(
                transform=plot_transform,
                channel_display_groups=channel_display_groups))

    if external_model:
        class_config.ensure_null_class()
        num_classes = len(class_config)
        model = SemanticSegmentationModelConfig(
            external_def=ExternalModuleConfig(
                github_repo='AdeelH/pytorch-fpn:0.3',
                name='fpn',
                entrypoint='make_fpn_resnet',
                entrypoint_kwargs={
                    'name': 'resnet50',
                    'fpn_type': 'panoptic',
                    'num_classes': num_classes,
                    'fpn_channels': 256,
                    'in_channels': len(channel_order),
                    'out_size': (img_sz, img_sz)
                }))
    else:
        model = SemanticSegmentationModelConfig(backbone=Backbone.resnet50)

    solver = SolverConfig(
        lr=1e-4, num_epochs=num_epochs, batch_sz=batch_sz, one_cycle=True)

    backend = PyTorchSemanticSegmentationConfig(
        data=data,
        model=model,
        solver=solver,
        log_tensorboard=True,
        run_tensorboard=False,
    )

    predict_options = SemanticSegmentationPredictOptions(chip_sz=chip_sz)

    pipeline = SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        chip_options=chip_options,
        predict_options=predict_options)

    return pipeline
