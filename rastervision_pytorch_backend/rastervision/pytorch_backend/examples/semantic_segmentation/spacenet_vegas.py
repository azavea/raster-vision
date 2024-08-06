import re
import random
import os
from abc import abstractmethod

from rastervision.pipeline.file_system.utils import list_paths
from rastervision.core.rv_pipeline import (
    SemanticSegmentationConfig, SemanticSegmentationChipOptions,
    SemanticSegmentationPredictOptions, WindowSamplingConfig,
    WindowSamplingMethod)
from rastervision.core.data import (
    BufferTransformerConfig, ClassConfig, ClassInferenceTransformerConfig,
    DatasetConfig, GeoJSONVectorSourceConfig, PolygonVectorOutputConfig,
    RasterioSourceConfig, RasterizedSourceConfig, RasterizerConfig,
    SceneConfig, SemanticSegmentationLabelSourceConfig,
    SemanticSegmentationLabelStoreConfig, StatsTransformerConfig)
from rastervision.pytorch_backend import PyTorchSemanticSegmentationConfig
from rastervision.pytorch_learner import (
    Backbone, SolverConfig, SemanticSegmentationGeoDataConfig,
    SemanticSegmentationImageDataConfig, SemanticSegmentationModelConfig)

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
            raise ValueError(f'{target} is not a valid target.')

    def get_raster_source_uri(self, id):
        filename = f'{self.raster_fn_prefix}{id}.tif'
        return os.path.join(self.raw_uri, self.base_dir, self.raster_dir,
                            filename)

    def get_geojson_uri(self, id):
        filename = f'{self.label_fn_prefix}{id}.geojson'
        return os.path.join(self.raw_uri, self.base_dir, self.label_dir,
                            filename)

    def get_scene_ids(self):
        label_dir = os.path.join(self.raw_uri, self.base_dir, self.label_dir)
        label_paths = list_paths(label_dir, ext='.geojson')
        label_re = re.compile(rf'.*{self.label_fn_prefix}(\d+)\.geojson')
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
            names=['road', 'background'],
            colors=['orange', 'black'],
            null_class='background')

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
            names=['building', 'background'],
            colors=['orange', 'black'],
            null_class='background')

    def get_class_id_to_filter(self):
        return {0: ['has', 'building']}


def build_scene(spacenet_cfg: SpacenetConfig,
                id: str,
                channel_order: list | None = None) -> SceneConfig:
    image_uri = spacenet_cfg.get_raster_source_uri(id)
    label_uri = spacenet_cfg.get_geojson_uri(id)

    raster_source = RasterioSourceConfig(
        uris=[image_uri],
        channel_order=channel_order,
        transformers=[StatsTransformerConfig()])

    # Set a line buffer to convert line strings to polygons.
    vector_source = GeoJSONVectorSourceConfig(
        uris=label_uri,
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
        nochip (bool): If True, read directly from the TIFF during
            training instead of from pre-generated chips. The analyze and chip
            commands should not be run, if this is set to True. Defaults to
            True.
        test (bool): If True, does the following simplifications:
            (1) Uses only a small subset of training and validation scenes.
            (2) Trains for only 2 epochs.
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
    chip_sz = 325
    img_sz = chip_sz

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

    chip_options = SemanticSegmentationChipOptions(
        sampling=WindowSamplingConfig(
            method=WindowSamplingMethod.sliding, size=chip_sz, stride=chip_sz))

    if nochip:
        data = SemanticSegmentationGeoDataConfig(
            scene_dataset=scene_dataset,
            sampling=chip_options.sampling,
            img_sz=img_sz,
            num_workers=4)
    else:
        data = SemanticSegmentationImageDataConfig(
            img_sz=img_sz, num_workers=4)

    backend = PyTorchSemanticSegmentationConfig(
        data=data,
        model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(lr=1e-4, num_epochs=5, batch_sz=8, one_cycle=True),
        log_tensorboard=True,
        run_tensorboard=False,
    )

    predict_options = SemanticSegmentationPredictOptions(chip_sz=chip_sz)

    return SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        chip_options=chip_options,
        predict_options=predict_options)
