# flake8: noqa

import os
from os.path import join
from pathlib import Path
from functools import partial

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import get_scene_info, save_image_crop

# -----------------
# Input files
# -----------------
RGBIR_DIR = '4_Ortho_RGBIR'
ELEVATION_DIR = 'elevation'
LABEL_DIR = '5_Labels_for_participants'
CROP_DIR = 'crops'

RGBIR_FNAME = lambda scene_id: f'top_potsdam_{scene_id}_RGBIR.tif'
ELEVATION_FNAME = lambda scene_id: f'{scene_id}.jpg'
LABEL_FNAME = lambda scene_id: f'top_potsdam_{scene_id}_label.tif'

TRAIN_IDS = [
    '2-10', '2-11', '3-10', '3-11', '4-10', '4-11', '4-12', '5-10', '5-11',
    '5-12', '6-10', '6-11', '6-7', '6-9', '7-10', '7-11', '7-12', '7-7', '7-8',
    '7-9'
]
VAL_IDS = ['2-12', '3-12', '6-12']

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

# -----------------
# Training
# -----------------
LR = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 8
ONE_CYCLE = True
LOG_TENSORBOARD = True
RUN_TENSORBOARD = False
CHANNEL_DISPLAY_GROUPS = {'RGB': (0, 1, 2), 'IR': (3, ), 'Elevation': (4, )}

# -----------------
# Test mode settings
# -----------------
TEST_MODE_TRAIN_IDS = TRAIN_IDS[:2]
TEST_MODE_VAL_IDS = VAL_IDS[:2]
TEST_MODE_NUM_EPOCHS = 2
TEST_MODE_BATCH_SIZE = 2
TEST_CROP_SIZE = 600


####################
# Utils
####################
def make_crop(processed_uri, raster_uri, label_uri):
    crop_uri = processed_uri / CROP_DIR / raster_uri.name
    label_crop_uri = processed_uri / CROP_DIR / label_uri.name

    save_image_crop(
        str(raster_uri),
        str(crop_uri),
        label_uri=str(label_uri),
        label_crop_uri=str(label_crop_uri),
        size=TEST_CROP_SIZE,
        vector_labels=False)

    return crop_uri, label_crop_uri


def make_label_source(class_config, label_uri):
    label_uri = str(label_uri)
    # Using with_rgb_class_map because label TIFFs have classes encoded as
    # RGB colors.
    label_source = SemanticSegmentationLabelSourceConfig(
        rgb_class_config=class_config,
        raster_source=RasterioSourceConfig(uris=[label_uri]))

    # URI will be injected by scene config.
    # Using rgb=True because we want prediction TIFFs to be in
    # RGB format.
    label_store = SemanticSegmentationLabelStoreConfig(
        rgb=True, vector_output=[PolygonVectorOutputConfig(class_id=0)])

    return label_source, label_store


def make_multi_raster_source(rgbir_raster_uri, elevation_raster_uri):
    rgbir_raster_uri = str(rgbir_raster_uri)
    elevation_raster_uri = str(elevation_raster_uri)

    # create multi raster source by combining rgbir and elevation sources
    rgbir_source = RasterioSourceConfig(uris=[rgbir_raster_uri])
    elevation_source = RasterioSourceConfig(uris=[elevation_raster_uri])

    raster_source = MultiRasterSourceConfig(
        raster_sources=[
            (rgbir_source, (0, 1, 2, 3)),
            (elevation_source, (4, ))])

    return raster_source


def make_scene(raw_uri, processed_uri, class_config, test, scene_id):
    scene_id = scene_id.replace('-', '_')
    rgbir_raster_uri = raw_uri / RGBIR_DIR / RGBIR_FNAME(scene_id)
    elevation_raster_uri = raw_uri / ELEVATION_DIR / ELEVATION_FNAME(scene_id)
    label_uri = raw_uri / LABEL_DIR / LABEL_FNAME(scene_id)

    if test:
        rgbir_raster_uri, _ = make_crop(processed_uri, rgbir_raster_uri,
                                        label_uri)

        elevation_raster_uri, label_uri = make_crop(
            processed_uri, elevation_raster_uri, label_uri)

    raster_source = make_multi_raster_source(rgbir_raster_uri,
                                             elevation_raster_uri)

    label_source, label_store = make_label_source(class_config, label_uri)

    scene = SceneConfig(
        id=scene_id,
        raster_source=raster_source,
        label_source=label_source,
        label_store=label_store)

    return scene


################
# Config
################
def get_config(runner, raw_uri, processed_uri, root_uri, test=False):

    if test:
        train_ids = TEST_MODE_TRAIN_IDS
        val_ids = TEST_MODE_VAL_IDS
    else:
        train_ids = TRAIN_IDS
        val_ids = VAL_IDS

    raw_uri = Path(raw_uri)
    processed_uri = Path(processed_uri)

    # ----------------------------
    # Configure chip generation
    # ----------------------------
    chip_sz = CHIP_SIZE
    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.sliding, stride=chip_sz)

    # -------------------------------
    # Configure dataset generation
    # -------------------------------
    class_config = ClassConfig(names=CLASS_NAMES, colors=CLASS_COLORS)

    _make_scene = partial(make_scene, raw_uri, processed_uri, class_config,
                          test)
    dataset_config = DatasetConfig(
        class_config=class_config,
        train_scenes=[_make_scene(id) for id in train_ids],
        validation_scenes=[_make_scene(id) for id in val_ids])

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
        model=model_config,
        solver=solver_config,
        log_tensorboard=LOG_TENSORBOARD,
        run_tensorboard=RUN_TENSORBOARD,
        test_mode=test)

    # -----------------------------------------------
    # Pass configurations to the pipeline config
    # -----------------------------------------------
    return SemanticSegmentationConfig(
        root_uri=root_uri,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options,
        dataset=dataset_config,
        backend=backend_config,
        channel_display_groups=CHANNEL_DISPLAY_GROUPS,
    )
