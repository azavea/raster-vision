import os
from os.path import join

from rastervision.v2.rv.task import ChipClassificationConfig
from rastervision.v2.rv.backend import PyTorchChipClassificationConfig
from rastervision.v2.rv.data import (
    DatasetConfig, ClassConfig, SceneConfig, ChipClassificationLabelSourceConfig,
    GeoJSONVectorSourceConfig, RasterioSourceConfig)
from rastervision.v2.learner import (
    ClassificationLearnerConfig, ClassificationDataConfig,
    SolverConfig, ModelConfig)
from rastervision.v2.examples.utils import (
    get_scene_info, str_to_bool, save_image_crop)


aoi_path = 'AOIs/AOI_1_Rio/srcData/buildingLabels/Rio_OUTLINE_Public_AOI.geojson'


def get_config(runner, test=False):
    if runner in ['inprocess']:
        raw_uri = '/opt/data/raw-data/spacenet-dataset'
        processed_uri = '/opt/data/examples/spacenet/rio/processed-data'
        root_uri = '/opt/data/examples/test-output/rio-chip-classification'
    else:
        raw_uri = 's3://spacenet-dataset/'
        processed_uri = 's3://raster-vision-lf-dev/examples/spacenet/rio/processed-data'
        root_uri = 's3://raster-vision-lf-dev/examples/test-output/rio-chip-classification'

    debug = False
    train_scene_info = get_scene_info(join(processed_uri, 'train-scenes.csv'))
    val_scene_info = get_scene_info(join(processed_uri, 'val-scenes.csv'))
    if test:
        debug = True
        train_scene_info = train_scene_info[0:1]
        val_scene_info = val_scene_info[0:1]

    def make_scene(scene_info):
        (raster_uri, label_uri) = scene_info
        raster_uri = join(raw_uri, raster_uri)
        label_uri = join(processed_uri, label_uri)
        aoi_uri = join(raw_uri, aoi_path)

        if test:
            crop_uri = join(
                processed_uri, 'crops', os.path.basename(raster_uri))
            save_image_crop(raster_uri, crop_uri, label_uri=label_uri,
                            size=600, min_features=20)
            raster_uri = crop_uri

        id = os.path.splitext(os.path.basename(raster_uri))[0]
        raster_source = RasterioSourceConfig(
            channel_order=[0, 1, 2], uris=[raster_uri])
        label_source = ChipClassificationLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(uri=label_uri),
            ioa_thresh=0.5,
            use_intersection_over_cell=False,
            pick_min_class_id=True,
            background_class_id=2,
            infer_cells=True)

        return SceneConfig(
            id=id,
            raster_source=raster_source,
            label_source=label_source,
            aoi_uris=[aoi_uri])

    train_scenes = [make_scene(info) for info in train_scene_info]
    val_scenes = [make_scene(info) for info in val_scene_info]
    class_config = ClassConfig(
        names=['building', 'no_building'], colors=['red', 'black'])
    dataset = DatasetConfig(
        class_config=class_config, train_scenes=train_scenes,
        validation_scenes=val_scenes)

    model = ModelConfig(backbone='resnet50')
    solver = SolverConfig(lr=2e-4, num_epochs=3, batch_sz=8, one_cycle=True)
    data = ClassificationDataConfig(
        data_format='image_folder',
        labels=['building', 'no_building'])
    learner = ClassificationLearnerConfig(
        model=model, solver=solver, data=data, test_mode=test)
    backend = PyTorchChipClassificationConfig(learner=learner)

    config = ChipClassificationConfig(
        root_uri=root_uri, dataset=dataset, backend=backend,
        train_chip_size=200, debug=debug)
    return config
