from os.path import join, dirname

from rastervision2.core.rv_pipeline import (
    ObjectDetectionConfig, ObjectDetectionChipOptions, ObjectDetectionPredictOptions)
from rastervision2.core.data import (
    ClassConfig, ObjectDetectionLabelSourceConfig, GeoJSONVectorSourceConfig,
    RasterioSourceConfig, SceneConfig, DatasetConfig)
from rastervision2.pytorch_backend import PyTorchObjectDetectionConfig
from rastervision2.pytorch_learner import (
    Backbone, SolverConfig, ObjectDetectionModelConfig)


def get_config(runner, root_uri, data_uri=None, full_train=False):
    def get_path(part):
        if full_train:
            return join(data_uri, part)
        else:
            return join(dirname(__file__), part)

    class_config = ClassConfig(
        names=['car', 'building'],
        colors=['blue', 'red'])

    def make_scene(scene_id, img_path, label_path):
        raster_source = RasterioSourceConfig(
            channel_order=[0, 1, 2], uris=[img_path])
        label_source = ObjectDetectionLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=label_path, default_class_id=None))
        return SceneConfig(
            id=scene_id,
            raster_source=raster_source,
            label_source=label_source)

    if full_train:
        model = ObjectDetectionModelConfig(backbone=Backbone.resnet18)
        solver = SolverConfig(
            lr=1e-4, num_epochs=300, batch_sz=8, one_cycle=True,
            sync_interval=300)
    else:
        pretrained_uri = (
            'https://github.com/azavea/raster-vision-data/releases/download/v0.12/'
            'object-detection.pth')
        model = ObjectDetectionModelConfig(
            backbone=Backbone.resnet18, init_weights=pretrained_uri)
        solver = SolverConfig(
            lr=1e-9, num_epochs=1, batch_sz=2, one_cycle=True, sync_interval=200)
    backend = PyTorchObjectDetectionConfig(
        model=model,
        solver=solver,
        log_tensorboard=False,
        run_tensorboard=False,
        augmentors=[])

    scenes = [
        make_scene(
            'od_test', get_path('scene/image.tif'), get_path('scene/labels.json')),
        make_scene(
            'od_test-2', get_path('scene/image2.tif'), get_path('scene/labels2.json'))]
    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=scenes,
        validation_scenes=scenes)

    chip_options = ObjectDetectionChipOptions(neg_ratio=1.0, ioa_thresh=1.0)
    predict_options = ObjectDetectionPredictOptions(
        merge_thresh=0.1, score_thresh=0.5)

    return ObjectDetectionConfig(
        root_uri=root_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=300,
        predict_chip_sz=300,
        chip_options=chip_options,
        predict_options=predict_options)
