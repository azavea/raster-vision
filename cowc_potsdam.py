from os.path import join

from rastervision.config.utils import (CL, OD, TFOD, KERAS, MOBILENET,
                                       RESNET50)
from rastervision.config.experiment import (ExperimentPaths, ExperimentHelper,
                                            Experiment)

raw_root = "s3://raster-vision-raw-data"
rv_root = "s3://raster-vision-lf-dev"

train_scene_ids = [
    '2_10', '2_11', '2_12', '2_14', '3_11', '3_13', '4_10', '5_10', '6_7',
    '6_9'
]

validation_scene_ids = ['2_13', '6_8', '3_10']

def make_image_uri(scene_id):
    return join(raw_root,
                '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(scene_id))

def make_label_uri(scene_id):
    return join(rv_root,
                'processed-data/labels/all/top_potsdam_{}_RGBIR.json'.format(scene_id))

def create_experiments():
    m = TaskConfig.builder(OD) \
                   .with_classes(['car']) \
                   .with_chip_size(300) \
                   .build()

    b = BackendConfig.builder(KERAS) \
                     .for_task(m) \
                     .with_image_gen({
                         "horizontal_flip": True,
                         "vertical_flip": True,
                         "zca_whitening": True
                     }) \
                     .build()

    e = ExperimentConfig.builder() \
                        .with_model(m)  \
                        .with_backend(b)

    def make_scene(scene_id):
        return SceneConfig.builder() \
                          .with_raster_source(make_image_uri(scene_id)) \
                          .with_ground_truth(make_label_uri(scene_id)) \
                          .build()

    for scene in map(make_scene, train_scene_ids):
        e.add_scene('train', scene)

    for scene in map(make_scene, validation_scene_ids):
        e.add_scene('validation', scene)

    e.with_name("COWC-potsdam-object-detection")

    experiment = e.build()

    return [experiment]


if __name__ == "__main__":
    for e in create_experiments():
        e.save("experiment-{}-config.json".format(e.name))
else:
    experiments = create_experiments()
