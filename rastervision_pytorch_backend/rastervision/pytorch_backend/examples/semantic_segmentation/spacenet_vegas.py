# flake8: noqa

from typing import Optional
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


def build_scene(spacenet_cfg: SpacenetConfig,
                id: str,
                channel_order: Optional[list] = None) -> SceneConfig:
    image_uri = spacenet_cfg.get_raster_source_uri(id)
    label_uri = spacenet_cfg.get_geojson_uri(id)

    raster_source = RasterioSourceConfig(
        uris=[image_uri], channel_order=channel_order)

    # Set a line buffer to convert line strings to polygons.
    vector_source = GeoJSONVectorSourceConfig(
        uri=label_uri,
        ignore_crs_field=True,
        transformers=[
            ClassInferenceTransformerConfig(default_class_id=0),
            BufferTransformerConfig(
                geom_type='LineString', class_bufs={0: 15}),
            BufferTransformerConfig(geom_type='Point'),
        ])
    label_source = SemanticSegmentationLabelSourceConfig(
        raster_source=RasterizedSourceConfig(
            vector_source=vector_source,
            rasterizer_config=RasterizerConfig(background_class_id=1)))

    label_store = SemanticSegmentationLabelStoreConfig(
        vector_output=[PolygonVectorOutputConfig(class_id=0, denoise=3)])

    return SceneConfig(
        id=id,
        raster_source=raster_source,
        label_source=label_source,
        label_store=label_store)


def get_config(runner,
               raw_uri: str,
               root_uri: str,
               target: str = BUILDINGS,
               nochip: bool = True,
               test: bool = False) -> SemanticSegmentationConfig:
    """Generate the pipeline config for this task. This function will be called
    by RV, with arguments from the command line, when this example is run.

    Args:
        runner (Runner): Runner for the pipeline. Will be provided by RV.
        raw_uri (str): Directory where the raw data resides
        root_uri (str): Directory where all the output will be written.
        target (str): "buildings" | "roads". Defaults to "buildings".
        nochip (bool, optional): If True, read directly from the TIFF during
            training instead of from pre-generated chips. The analyze and chip
            commands should not be run, if this is set to True. Defaults to
            True.
        test (bool, optional): If True, does the following simplifications:
            (1) Uses only a small subset of training and validation scenes.
            (2) Enables test mode in the learner, which makes it use the
                test_batch_sz and test_num_epochs, among other things.
            Defaults to False.

    Returns:
        SemanticSegmentationConfig: An pipeline config.
    """

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
    train_ids = scene_ids[:num_train_ids]
    val_ids = scene_ids[num_train_ids:]

    if test:
        train_ids = train_ids[:16]
        val_ids = val_ids[:4]

    channel_order = [0, 1, 2]

    class_config = spacenet_cfg.get_class_config()
    train_scenes = [
        build_scene(spacenet_cfg, id, channel_order) for id in train_ids
    ]
    val_scenes = [
        build_scene(spacenet_cfg, id, channel_order) for id in val_ids
    ]
    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=val_scenes)

    chip_sz = 325
    img_sz = chip_sz

    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.sliding, stride=chip_sz)

    if nochip:
        data = SemanticSegmentationGeoDataConfig(
            scene_dataset=scene_dataset,
            window_opts=GeoDataWindowConfig(
                method=GeoDataWindowMethod.sliding,
                size=chip_sz,
                stride=chip_options.stride),
            img_sz=img_sz,
            num_workers=4)
    else:
        data = SemanticSegmentationImageDataConfig(
            img_sz=img_sz, num_workers=4)

    backend = PyTorchSemanticSegmentationConfig(
        data=data,
        model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=5,
            test_num_epochs=2,
            batch_sz=8,
            one_cycle=True),
        log_tensorboard=True,
        run_tensorboard=False,
        test_mode=test)

    return SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options)
