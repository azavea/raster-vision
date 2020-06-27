# flake8: noqa

import re
import random
import os
from abc import abstractmethod

from rastervision.pipeline.file_system import list_paths
from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *

BUILDINGS = 'buildings'
ROADS = 'roads'


class SpacenetConfig(object):
    def __init__(self, raw_uri):
        self.raw_uri = raw_uri

    @staticmethod
    def create(raw_uri, target):
        if target.lower() == BUILDINGS:
            return VegasBuildings(raw_uri)
        elif target.lower() == ROADS:
            return VegasRoads(raw_uri)
        else:
            raise ValueError('{} is not a valid target.'.format(target))

    def get_raster_source_uri(self, id):
        return os.path.join(self.raw_uri, self.base_dir, self.raster_dir,
                            '{}{}.tif'.format(self.raster_fn_prefix, id))

    def get_geojson_uri(self, id):
        return os.path.join(self.raw_uri, self.base_dir, self.label_dir,
                            '{}{}.geojson'.format(self.label_fn_prefix, id))

    def get_scene_ids(self):
        label_dir = os.path.join(self.raw_uri, self.base_dir, self.label_dir)
        label_paths = list_paths(label_dir, ext='.geojson')
        label_re = re.compile(r'.*{}(\d+)\.geojson'.format(
            self.label_fn_prefix))
        scene_ids = [
            label_re.match(label_path).group(1) for label_path in label_paths
        ]
        return scene_ids

    @abstractmethod
    def get_class_config(self):
        pass

    @abstractmethod
    def get_class_id_to_filter(self):
        pass


class VegasRoads(SpacenetConfig):
    def __init__(self, raw_uri):
        self.base_dir = 'spacenet/SN3_roads/train/AOI_2_Vegas/'
        self.raster_dir = 'PS-RGB/'
        self.label_dir = 'geojson_roads/'
        self.raster_fn_prefix = 'SN3_roads_train_AOI_2_Vegas_PS-RGB_img'
        self.label_fn_prefix = 'SN3_roads_train_AOI_2_Vegas_geojson_roads_img'
        super().__init__(raw_uri)

    def get_class_config(self):
        return ClassConfig(
            names=['road', 'background'], colors=['orange', 'black'])

    def get_class_id_to_filter(self):
        return {0: ['has', 'highway']}


class VegasBuildings(SpacenetConfig):
    def __init__(self, raw_uri):
        self.base_dir = 'spacenet/SN2_buildings/train/AOI_2_Vegas'
        self.raster_dir = 'PS-RGB'
        self.label_dir = 'geojson_buildings'
        self.raster_fn_prefix = 'SN2_buildings_train_AOI_2_Vegas_PS-RGB_img'
        self.label_fn_prefix = 'SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img'
        super().__init__(raw_uri)

    def get_class_config(self):
        return ClassConfig(
            names=['building', 'background'], colors=['orange', 'black'])

    def get_class_id_to_filter(self):
        return {0: ['has', 'building']}


def build_scene(spacenet_cfg, id, channel_order=None):
    image_uri = spacenet_cfg.get_raster_source_uri(id)
    label_uri = spacenet_cfg.get_geojson_uri(id)

    # Need to use stats_transformer because imagery is uint16.
    raster_source = RasterioSourceConfig(
        uris=[image_uri],
        channel_order=channel_order,
        transformers=[StatsTransformerConfig()])

    # Set a line buffer to convert line strings to polygons.
    vector_source = GeoJSONVectorSourceConfig(
        uri=label_uri,
        default_class_id=0,
        ignore_crs_field=True,
        line_bufs={0: 15})
    label_source = SemanticSegmentationLabelSourceConfig(
        raster_source=RasterizedSourceConfig(
            vector_source=vector_source,
            rasterizer_config=RasterizerConfig(background_class_id=1)))

    # Generate polygon output for segmented buildings.
    label_store = None
    if isinstance(spacenet_cfg, VegasBuildings):
        label_store = SemanticSegmentationLabelStoreConfig(
            vector_output=[PolygonVectorOutputConfig(class_id=0, denoise=3)])

    return SceneConfig(
        id=id,
        raster_source=raster_source,
        label_source=label_source,
        label_store=label_store)


def get_config(runner, raw_uri, root_uri, target=BUILDINGS, test=False):
    spacenet_cfg = SpacenetConfig.create(raw_uri, target)
    scene_ids = spacenet_cfg.get_scene_ids()
    if len(scene_ids) == 0:
        raise ValueError(
            'No scenes found. Something is configured incorrectly.')

    random.seed(5678)
    scene_ids = sorted(scene_ids)
    random.shuffle(scene_ids)

    # Workaround to handle scene 1000 missing on S3.
    if '1000' in scene_ids:
        scene_ids.remove('1000')
    split_ratio = 0.8
    num_train_ids = round(len(scene_ids) * split_ratio)
    train_ids = scene_ids[0:num_train_ids]
    val_ids = scene_ids[num_train_ids:]

    num_train_scenes = len(train_ids)
    num_val_scenes = len(val_ids)
    if test:
        num_train_scenes = 16
        num_val_scenes = 4
    train_ids = train_ids[0:num_train_scenes]
    val_ids = val_ids[0:num_val_scenes]
    channel_order = [0, 1, 2]

    class_config = spacenet_cfg.get_class_config()
    train_scenes = [
        build_scene(spacenet_cfg, id, channel_order) for id in train_ids
    ]
    val_scenes = [
        build_scene(spacenet_cfg, id, channel_order) for id in val_ids
    ]
    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=val_scenes)

    train_chip_sz = 325
    predict_chip_sz = 650
    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.sliding,
        stride=train_chip_sz)
    num_epochs = 5

    backend = PyTorchSemanticSegmentationConfig(
        model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=num_epochs,
            test_num_epochs=2,
            batch_sz=8,
            one_cycle=True),
        log_tensorboard=True,
        run_tensorboard=False,
        test_mode=test)

    return SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=train_chip_sz,
        predict_chip_sz=predict_chip_sz,
        chip_options=chip_options)
