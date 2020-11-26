from os.path import join, dirname, basename

from rastervision.core.rv_pipeline import ChipClassificationConfig
from rastervision.core.data import (
    ClassConfig, ChipClassificationLabelSourceConfig,
    GeoJSONVectorSourceConfig, RasterioSourceConfig, StatsTransformerConfig,
    SceneConfig, DatasetConfig)
from rastervision.pytorch_backend import PyTorchChipClassificationConfig
from rastervision.pytorch_learner import (Backbone, SolverConfig,
                                          ClassificationModelConfig,
                                          ClassificationImageDataConfig)


def get_config(runner, root_uri, data_uri=None, full_train=False):
    def get_path(part):
        if full_train:
            return join(data_uri, part)
        else:
            return join(dirname(__file__), part)

    class_config = ClassConfig(
        names=['car', 'building', 'background'],
        colors=['red', 'blue', 'black'])

    def make_scene(img_path, label_path):
        id = basename(img_path)
        label_source = ChipClassificationLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=label_path, default_class_id=None, ignore_crs_field=True),
            ioa_thresh=0.5,
            use_intersection_over_cell=False,
            pick_min_class_id=True,
            background_class_id=2,
            infer_cells=True)

        raster_source = RasterioSourceConfig(
            channel_order=[0, 1, 2],
            uris=[img_path],
            transformers=[StatsTransformerConfig()])

        return SceneConfig(
            id=id, raster_source=raster_source, label_source=label_source)

    scenes = [
        make_scene(get_path('scene/image.tif'), get_path('scene/labels.json')),
        make_scene(
            get_path('scene/image2.tif'), get_path('scene/labels2.json'))
    ]
    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=scenes,
        validation_scenes=scenes)

    chip_sz = 200
    img_sz = chip_sz
    data = ClassificationImageDataConfig(img_sz=img_sz, augmentors=[])

    if full_train:
        model = ClassificationModelConfig(backbone=Backbone.resnet18)
        solver = SolverConfig(
            lr=1e-4,
            num_epochs=300,
            batch_sz=8,
            one_cycle=True,
            sync_interval=300)
    else:
        pretrained_uri = (
            'https://github.com/azavea/raster-vision-data/releases/download/v0.12/'
            'chip-classification.pth')
        model = ClassificationModelConfig(
            backbone=Backbone.resnet18, init_weights=pretrained_uri)
        solver = SolverConfig(
            lr=1e-9,
            num_epochs=1,
            batch_sz=2,
            one_cycle=True,
            sync_interval=200)
    backend = PyTorchChipClassificationConfig(
        data=data,
        model=model,
        solver=solver,
        log_tensorboard=False,
        run_tensorboard=False)

    config = ChipClassificationConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz)

    return config
