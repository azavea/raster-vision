from rastervision.core.rv_pipeline import (SemanticSegmentationConfig)
from rastervision.core.data import (
    SceneConfig, ClassConfig, DatasetConfig, RasterioSourceConfig,
    SemanticSegmentationLabelSourceConfig,
    SemanticSegmentationLabelStoreConfig, PolygonVectorOutputConfig)
from rastervision.pytorch_backend import (PyTorchSemanticSegmentationConfig)
from rastervision.pytorch_learner import (
    ExternalModuleConfig, SemanticSegmentationGeoDataConfig,
    SemanticSegmentationModelConfig, GeoDataWindowMethod, GeoDataWindowConfig,
    SolverConfig)

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
               multiband: bool = False,
               test: bool = False) -> SemanticSegmentationConfig:
    train_ids = TRAIN_IDS
    val_ids = VAL_IDS

    if test:
        train_ids = train_ids[:1]
        val_ids = val_ids[:1]

    class_config = ClassConfig(names=CLASS_NAMES, colors=CLASS_COLORS)
    class_config.ensure_null_class()

    if multiband:
        # use all 4 channels
        channel_order = [0, 1, 2, 3]
        channel_display_groups = {'RGB': (0, 1, 2), 'IR': (3, )}
    else:
        # use infrared, red, & green channels only
        channel_order = [3, 0, 1]
        channel_display_groups = None

    def make_scene(id) -> SceneConfig:
        id = str(id)
        id = id.replace('-', '_')
        raster_uri = f'{raw_uri}/4_Ortho_RGBIR/top_potsdam_{id}_RGBIR.tif'
        label_uri = f'{raw_uri}/5_Labels_for_participants/top_potsdam_{id}_label.tif'

        raster_source = RasterioSourceConfig(
            uris=[raster_uri], channel_order=channel_order, persist=True)

        # Using with_rgb_class_map because label TIFFs have classes encoded as
        # RGB colors.
        label_source = SemanticSegmentationLabelSourceConfig(
            rgb_class_config=class_config,
            raster_source=RasterioSourceConfig(uris=[label_uri], persist=True))

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

    img_sz = 256
    model_config = SemanticSegmentationModelConfig(
        external_def=ExternalModuleConfig(
            github_repo='AdeelH/pytorch-fpn',
            name='fpn',
            entrypoint='make_segm_fpn_resnet',
            entrypoint_kwargs={
                'name': 'resnet18',
                'fpn_type': 'panoptic',
                'num_classes': len(class_config.names) + 1,
                'fpn_channels': 128,
                'in_channels': len(channel_order),
                'out_size': (img_sz, img_sz)
            }))

    window_opts = {}
    for s in scene_dataset.train_scenes:
        window_opts[s.id] = GeoDataWindowConfig(
            # method=GeoDataWindowMethod.sliding,
            method=GeoDataWindowMethod.random,
            size=img_sz,
            # size_lims=(200, 300),
            h_lims=(200, 300),
            w_lims=(200, 300),
            max_windows=2209,
        )
    for s in scene_dataset.validation_scenes:
        window_opts[s.id] = GeoDataWindowConfig(
            method=GeoDataWindowMethod.sliding,
            size=img_sz,
            stride=img_sz // 2)

    data_config = SemanticSegmentationGeoDataConfig(
        scene_dataset=scene_dataset,
        window_opts=window_opts,
        img_sz=img_sz,
        img_channels=len(channel_order),
        num_workers=4,
        channel_display_groups=channel_display_groups)

    backend = PyTorchSemanticSegmentationConfig(
        data=data_config,
        model=model_config,
        solver=SolverConfig(lr=1e-4, num_epochs=10, batch_sz=8))

    pipeline_config = SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        predict_chip_sz=img_sz)

    return pipeline_config
