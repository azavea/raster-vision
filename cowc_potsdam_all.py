from os.path import join

from rastervision.config.utils import (CL, OD, TFOD, KERAS, MOBILENET,
                                       RESNET50)
from rastervision.config.experiment import (ExperimentPaths, ExperimentHelper,
                                            Experiment)

# Should be able to point at json config, python file, or repository

raw_root = "s3://raster-vision-raw-data"
rv_root = "s3://raster-vision-lf-dev"

train_scene_ids = [
    '2_10', '2_11', '2_12', '2_14', '3_11',
    '3_13', '4_10', '5_10', '6_7', '6_9'
]

validation_scene_ids = ['2_13', '6_8', '3_10']

def make_image_uri(scene_id):
    return join(raw_root,
                '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(scene_id))

def make_label_uri(scene_id):
    return join(rv_root,
                'processed-data/labels/all/top_potsdam_{}_RGBIR.json'.format(scene_id))

def create_experiments(**kwargs):
    t = TaskConfig.builder(OD) \
                   .with_classes(['car']) \
                   .with_chip_size(300) \
                   .build()

    # BackendConfig consists of a ModelConfig and TrainingConfig

    b = BackendConfig.builder(KERAS) \
                     .for_task(t) \
                     .with_model(RESNET50) \
                     .with_model_args({ "activation": "softmax" }) \
                     .with_image_gen({
                         "horizontal_flip": True,
                         "vertical_flip": True,
                         "zca_whitening": True
                     }) \
                     .build()

    e = ExperimentConfig.builder() \
                        .with_task(t) \
                        .with_backend(b)

    def make_scene(scene_id):
        return SceneConfig.builder() \
                          .with_raster_source(make_image_uri(scene_id)) \
                          .with_ground_truth(make_label_uri(scene_id)) \
                          .build()

    for scene in map(make_scene, train_scene_ids):
        e = e.with_scene('train', scene)

    for scene in map(make_scene, validation_scene_ids):
        e = e.with_scene('validation', scene)

    e = e.with_name("COWC-potsdam-object-detection") \
         .with_root("s3://raster-vision-rob-dev/cowc") \
         .with_raw_dataset_key("potsdam") \
         .with_processed_dataset_key("classification-chips") \
         .with_model_key("resnet50") \
         .with_predict_key("predction") \
         .with_eval_key("eval")

    experiment = e.build()

    return [experiment]

if __name__ == "__main__":
    for e in create_experiments():
        e.save("experiment-{}-config.json".format(e.name))
