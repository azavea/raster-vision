from functools import partial
from typing import Tuple, Union

from rastervision.core.rv_pipeline import (
    SceneConfig, DatasetConfig, SemanticSegmentationChipOptions,
    SemanticSegmentationWindowMethod, SemanticSegmentationConfig)

from rastervision.core.data import (
    ClassConfig, RasterioSourceConfig, MultiRasterSourceConfig,
    SemanticSegmentationLabelSourceConfig,
    SemanticSegmentationLabelStoreConfig, PolygonVectorOutputConfig,
    RGBClassTransformerConfig)

from rastervision.pytorch_backend import (PyTorchSemanticSegmentationConfig,
                                          SemanticSegmentationModelConfig)
from rastervision.pytorch_backend.examples.utils import (save_image_crop)
from rastervision.pytorch_learner import (
    Backbone, SolverConfig, SemanticSegmentationImageDataConfig,
    SemanticSegmentationGeoDataConfig, GeoDataWindowConfig,
    GeoDataWindowMethod, PlotOptions)

# -----------------------
# Input files and paths
# -----------------------
RGBIR_DIR = '4_Ortho_RGBIR'
ELEVATION_DIR = 'elevation'
LABEL_DIR = '5_Labels_for_participants'

RGBIR_FNAME = lambda scene_id: f'top_potsdam_{scene_id}_RGBIR.tif'  # noqa
ELEVATION_FNAME = lambda scene_id: f'{scene_id}.jpg'  # noqa
LABEL_FNAME = lambda scene_id: f'top_potsdam_{scene_id}_label.tif'  # noqa

TRAIN_IDS = [
    '2_10', '2_11', '3_10', '3_11', '4_10', '4_11', '4_12', '5_10', '5_11',
    '5_12', '6_10', '6_11', '6_7', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8',
    '7_9'
]
VAL_IDS = ['2_12', '3_12', '6_12']

# -----------------
# Data prep
# -----------------
CLASS_NAMES = [
    'Car', 'Building', 'Low Vegetation', 'Tree', 'Impervious', 'Clutter'
]
CLASS_COLORS = [
    '#ffff00', '#0000ff', '#00ffff', '#00ff00', '#ffffff', '#ff0000'
]
CHIP_SIZE = 300
CHANNEL_ORDER = [0, 1, 2, 3, 4]

# -----------------
# Training
# -----------------
LR = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 8
ONE_CYCLE = True
LOG_TENSORBOARD = True
RUN_TENSORBOARD = False
CHANNEL_DISPLAY_GROUPS = {'RGB': [0, 1, 2], 'IR': [3], 'Elevation': [4]}

# -----------------
# Test mode settings
# -----------------
TEST_MODE_TRAIN_IDS = TRAIN_IDS[:2]
TEST_MODE_VAL_IDS = VAL_IDS[:2]
TEST_MODE_NUM_EPOCHS = 2
TEST_MODE_BATCH_SIZE = 2
TEST_CROP_SIZE = 600
TEST_CROP_DIR = 'crops'


################
# Config
################
def get_config(runner,
               raw_uri: str,
               processed_uri: str,
               root_uri: str,
               nochip: bool = True,
               test: bool = False) -> SemanticSegmentationConfig:
    """Generate the pipeline config for this task. This function will be called
    by RV, with arguments from the command line, when this example is run.

    Args:
        runner (Runner): Runner for the pipeline.
        raw_uri (str): Directory where the raw data resides
        processed_uri (str): Directory for storing processed data.
                             E.g. crops for testing.
        root_uri (str): Directory where all the output will be written.
        nochip (bool, optional): If True, read directly from the TIFF during
            training instead of from pre-generated chips. The analyze and chip
            commands should not be run, if this is set to True. Defaults to
            True.
        test (bool, optional): If True, does the following simplifications:
            (1) Uses only the first 2 scenes
            (2) Uses only a 600x600 crop of the scenes
            (3) Enables test mode in the learner, which makes it use the
                test_batch_sz and test_num_epochs, among other things.
            Defaults to False.

    Returns:
        SemanticSegmentationConfig: A pipeline config.
    """
    if not test:
        train_ids, val_ids = TRAIN_IDS, VAL_IDS
    else:
        train_ids, val_ids = TEST_MODE_TRAIN_IDS, TEST_MODE_VAL_IDS

    raw_uri = UriPath(raw_uri)
    processed_uri = UriPath(processed_uri)

    # -------------------------------
    # Configure dataset generation
    # -------------------------------
    class_config = ClassConfig(names=CLASS_NAMES, colors=CLASS_COLORS)

    _make_scene = partial(
        make_scene, raw_uri, processed_uri, class_config, test_mode=test)

    dataset_config = DatasetConfig(
        class_config=class_config,
        train_scenes=[_make_scene(scene_id) for scene_id in train_ids],
        validation_scenes=[_make_scene(scene_id) for scene_id in val_ids])

    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.sliding,
        stride=CHIP_SIZE)

    if nochip:
        window_opts = GeoDataWindowConfig(
            method=GeoDataWindowMethod.sliding,
            size=CHIP_SIZE,
            stride=chip_options.stride)

        data = SemanticSegmentationGeoDataConfig(
            scene_dataset=dataset_config,
            window_opts=window_opts,
            img_sz=CHIP_SIZE,
            num_workers=4,
            plot_options=PlotOptions(
                channel_display_groups=CHANNEL_DISPLAY_GROUPS))
    else:
        data = SemanticSegmentationImageDataConfig(
            img_sz=CHIP_SIZE,
            num_workers=4,
            plot_options=PlotOptions(
                channel_display_groups=CHANNEL_DISPLAY_GROUPS))

    # --------------------------------------------
    # Configure PyTorch backend and training
    # --------------------------------------------
    model_config = SemanticSegmentationModelConfig(backbone=Backbone.resnet50)

    solver_config = SolverConfig(
        lr=LR,
        num_epochs=NUM_EPOCHS,
        batch_sz=BATCH_SIZE,
        test_num_epochs=TEST_MODE_NUM_EPOCHS,
        test_batch_sz=TEST_MODE_BATCH_SIZE,
        one_cycle=ONE_CYCLE)

    backend_config = PyTorchSemanticSegmentationConfig(
        data=data,
        model=model_config,
        solver=solver_config,
        log_tensorboard=LOG_TENSORBOARD,
        run_tensorboard=RUN_TENSORBOARD,
        test_mode=test)

    # -----------------------------------------------
    # Pass configurations to the pipeline config
    # -----------------------------------------------
    pipeline_config = SemanticSegmentationConfig(
        root_uri=root_uri,
        train_chip_sz=CHIP_SIZE,
        predict_chip_sz=CHIP_SIZE,
        chip_options=chip_options,
        dataset=dataset_config,
        backend=backend_config)

    return pipeline_config


####################
# Utils
####################
class UriPath(object):
    """ Workaround for pathlib.Path converting "s3://abc to s3:/abc" """

    def __init__(self, s):
        from pathlib import Path
        self._path = Path(s)

    @property
    def name(self):
        return self._path.name

    @property
    def stem(self):
        return self._path.stem

    def __truediv__(self, val):
        return UriPath(self._path / val)

    def __repr__(self):
        import re
        s = str(self._path)
        # s3:/abc --> s3://abc
        s = re.sub(r'^([^/]+):(?:/([^/]|$))', r'\1://\2', s)
        return s


def make_scene(raw_uri: UriPath,
               processed_uri: UriPath,
               class_config: ClassConfig,
               scene_id: str,
               test_mode=False) -> SceneConfig:
    rgbir_raster_uri = raw_uri / RGBIR_DIR / RGBIR_FNAME(scene_id)
    elevation_raster_uri = raw_uri / ELEVATION_DIR / ELEVATION_FNAME(scene_id)
    label_uri = raw_uri / LABEL_DIR / LABEL_FNAME(scene_id)

    if test_mode:
        label_uri_orig = label_uri
        rgbir_raster_uri, _ = make_crop(processed_uri, rgbir_raster_uri, None)

        elevation_raster_uri, label_uri = make_crop(
            processed_uri, elevation_raster_uri, label_uri_orig)

    raster_source = make_multi_raster_source(rgbir_raster_uri,
                                             elevation_raster_uri)

    label_source, label_store = make_label_source(class_config, label_uri)

    scene = SceneConfig(
        id=scene_id,
        raster_source=raster_source,
        label_source=label_source,
        label_store=label_store)

    return scene


def make_multi_raster_source(
        rgbir_raster_uri: Union[UriPath, str],
        elevation_raster_uri: Union[UriPath, str]) -> MultiRasterSourceConfig:
    """ Create multi raster source by combining rgbir and elevation sources. """
    rgbir_raster_uri = str(rgbir_raster_uri)
    elevation_raster_uri = str(elevation_raster_uri)

    rgbir_source = RasterioSourceConfig(
        uris=[rgbir_raster_uri], channel_order=[0, 1, 2, 3])

    elevation_source = RasterioSourceConfig(
        uris=[elevation_raster_uri], channel_order=[0])

    raster_source = MultiRasterSourceConfig(
        raster_sources=[rgbir_source, elevation_source])

    return raster_source


def make_crop(processed_uri: UriPath,
              raster_uri: UriPath,
              label_uri: UriPath = None) -> Tuple[UriPath, UriPath]:
    crop_uri = processed_uri / TEST_CROP_DIR / raster_uri.name
    if label_uri is not None:
        label_crop_uri = processed_uri / TEST_CROP_DIR / label_uri.name
    else:
        label_crop_uri = None

    save_image_crop(
        str(raster_uri),
        str(crop_uri),
        label_uri=str(label_uri) if label_uri else None,
        label_crop_uri=str(label_crop_uri) if label_uri else None,
        size=TEST_CROP_SIZE,
        vector_labels=False)

    return crop_uri, label_crop_uri


def make_label_source(class_config: ClassConfig, label_uri: Union[UriPath, str]
                      ) -> Tuple[SemanticSegmentationLabelSourceConfig,
                                 SemanticSegmentationLabelStoreConfig]:
    label_uri = str(label_uri)
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

    return label_source, label_store
