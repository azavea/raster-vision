# flake8: noqa

import os
from os.path import join

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import get_scene_info, save_image_crop


def get_config(runner, raw_uri, processed_uri, root_uri, test=False):
    train_ids = [
        '2-10', '2-11', '3-10', '3-11', '4-10', '4-11', '4-12', '5-10', '5-11',
        '5-12', '6-10', '6-11', '6-7', '6-9', '7-10', '7-11', '7-12', '7-7',
        '7-8', '7-9'
    ]
    val_ids = ['2-12', '3-12', '6-12']

    if test:
        train_ids = train_ids[0:1]
        val_ids = val_ids[0:1]

    class_config = ClassConfig(
        names=[
            'Car', 'Building', 'Low Vegetation', 'Tree', 'Impervious',
            'Clutter'
        ],
        colors=[
            '#ffff00', '#0000ff', '#00ffff', '#00ff00', '#ffffff', '#ff0000'
        ])

    def make_scene(id):
        id = id.replace('-', '_')
        raster_uri = '{}/4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(
            raw_uri, id)
        label_uri = '{}/5_Labels_for_participants/top_potsdam_{}_label.tif'.format(
            raw_uri, id)

        if test:
            crop_uri = join(processed_uri, 'crops',
                            os.path.basename(raster_uri))
            label_crop_uri = join(processed_uri, 'crops',
                                  os.path.basename(label_uri))
            save_image_crop(
                raster_uri,
                crop_uri,
                label_uri=label_uri,
                label_crop_uri=label_crop_uri,
                size=600,
                vector_labels=False)
            raster_uri = crop_uri
            label_uri = label_crop_uri

        # infrared, red, green
        channel_order = [3, 0, 1]
        raster_source = RasterioSourceConfig(
            uris=[raster_uri], channel_order=channel_order)

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

        scene = SceneConfig(
            id=id,
            raster_source=raster_source,
            label_source=label_source,
            label_store=label_store)

        return scene

    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene(id) for id in train_ids],
        validation_scenes=[make_scene(id) for id in val_ids])
    chip_sz = 300
    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.sliding, stride=chip_sz)

    backend = PyTorchSemanticSegmentationConfig(
        model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=10,
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
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options)
