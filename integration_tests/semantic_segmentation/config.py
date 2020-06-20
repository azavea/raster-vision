from os.path import join, dirname

from rastervision2.core.data import (
    ClassConfig, SemanticSegmentationLabelSourceConfig,
    SemanticSegmentationLabelStoreConfig, RasterioSourceConfig, SceneConfig,
    PolygonVectorOutputConfig, DatasetConfig, BuildingVectorOutputConfig)
from rastervision2.core.rv_pipeline import (
    SemanticSegmentationChipOptions, SemanticSegmentationWindowMethod,
    SemanticSegmentationConfig)
from rastervision2.pytorch_backend import PyTorchSemanticSegmentationConfig
from rastervision2.pytorch_learner import (
    Backbone, SolverConfig, SemanticSegmentationModelConfig)


def get_config(runner, root_uri, data_uri=None, full_train=False):
    def get_path(part):
        if full_train:
            return join(data_uri, part)
        else:
            return join(dirname(__file__), part)

    class_config = ClassConfig(
        names=['red', 'green'],
        colors=['red', 'green'])

    def make_scene(id, img_path, label_path):
        raster_source = RasterioSourceConfig(
            channel_order=[0, 1, 2], uris=[img_path])
        label_source = SemanticSegmentationLabelSourceConfig(
            rgb_class_config=class_config,
            raster_source=RasterioSourceConfig(uris=[label_path]))
        label_store = SemanticSegmentationLabelStoreConfig(
            rgb=True, vector_output=[
                PolygonVectorOutputConfig(class_id=0),
                BuildingVectorOutputConfig(class_id=1)])

        return SceneConfig(
            id=id,
            raster_source=raster_source,
            label_source=label_source,
            label_store=label_store)

    if full_train:
        model = SemanticSegmentationModelConfig(backbone=Backbone.resnet50)
        solver = SolverConfig(
            lr=1e-4, num_epochs=300, batch_sz=8, one_cycle=True,
            sync_interval=300)
    else:
        pretrained_uri = (
            'https://github.com/azavea/raster-vision-data/releases/download/v0.12/'
            'semantic-segmentation.pth')
        model = SemanticSegmentationModelConfig(
            backbone=Backbone.resnet50, init_weights=pretrained_uri)
        solver = SolverConfig(
            lr=1e-9, num_epochs=1, batch_sz=2, one_cycle=True, sync_interval=200)
    backend = PyTorchSemanticSegmentationConfig(
        model=model,
        solver=solver,
        log_tensorboard=False,
        run_tensorboard=False,
        augmentors=[])

    scenes = [
        make_scene(
            'test-scene', get_path('scene/image.tif'), get_path('scene/labels.tif')),
        make_scene(
            'test-scene2', get_path('scene/image2.tif'), get_path('scene/labels2.tif'))]
    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=scenes,
        validation_scenes=scenes)

    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.sliding, stride=300)

    return SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=300,
        predict_chip_sz=300,
        chip_options=chip_options)
