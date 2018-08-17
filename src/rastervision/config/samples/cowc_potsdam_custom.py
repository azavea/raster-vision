from os.path import join

import click

from rastervision.config.utils import (CL, OD, TFOD, KERAS, MOBILENET,
                                       RESNET50, ClassificationGeoJSONOptions,
                                       ObjectDetectionPredictOptions)
from rastervision.config.experiment import (ExperimentPaths, ExperimentHelper,
                                            Experiment)


class CowcPotsdamPaths():
    def __init__(self, isprs_potsdam_uri, cowc_potsdam_uri):
        self.isprs_potsdam_uri = isprs_potsdam_uri
        self.cowc_potsdam_uri = cowc_potsdam_uri

    def make_image_uri(self, id):
        return join(
            self.isprs_potsdam_uri,
            '4_Ortho_RGBIR/top_potsdam_{id}_RGBIR.tif'.format(id=id))

    def make_label_uri(self, id):
        return join(
            self.cowc_potsdam_uri,
            'labels/all/top_potsdam_{id}_RGBIR.json'.format(id=id))


class TinyCowcPotsdamPaths():
    def __init__(self, cowc_potsdam_uri):
        self.cowc_potsdam_uri = cowc_potsdam_uri

    def make_image_uri(self, id):
        return join(self.cowc_potsdam_uri, 'tiny/{id}.tif'.format(id=id))

    def make_label_uri(self, id):
        return join(self.cowc_potsdam_uri, 'tiny/{id}.json'.format(id=id))


def get_hyperparams(model_config, is_tiny_dataset):
    if is_tiny_dataset:
        batch_size = 1
        num_iters = 1
    elif (model_config.backend == TFOD
          and model_config.model_type == MOBILENET):
        batch_size = 8
        num_iters = 50000
    elif (model_config.backend == KERAS
          and model_config.model_type == RESNET50):
        batch_size = 8
        num_iters = 20

    return batch_size, num_iters


def generate_experiment(experiment_uri,
                        isprs_potsdam_uri,
                        cowc_potsdam_uri,
                        use_tiny_dataset=False,
                        task=OD):
    class_names = ['car']
    train_scene_ids = [
        '2_10', '2_11', '2_12', '2_14', '3_11', '3_13', '4_10', '5_10', '6_7',
        '6_9'
    ]
    validation_scene_ids = ['2_13', '6_8', '3_10']
    cowc_potsdam_paths = CowcPotsdamPaths(isprs_potsdam_uri, cowc_potsdam_uri)

    if use_tiny_dataset:
        train_scene_ids = train_scene_ids[0:1]
        validation_scene_ids = validation_scene_ids[0:1]
        cowc_potsdam_paths = TinyCowcPotsdamPaths(cowc_potsdam_uri)

    paths = ExperimentPaths(experiment_uri)
    # Custom chip_size
    chip_size = 400
    helper = ExperimentHelper(paths, chip_size=chip_size)
    # Custom backend for classification (there's only one option right now
    # so this is silly, but it's illustrative.)
    backend = None
    if task == CL:
        backend = KERAS
    model_config = helper.make_model_config(
        class_names, task, backend=backend)

    # Custom way of inferring classification cells.
    task_options = task
    if task == CL:
        task_options = ClassificationGeoJSONOptions(
            ioa_thresh=0.1, pick_min_class_id=False)

    def make_scene(id):
        return helper.make_scene(
            id,
            model_config, [cowc_potsdam_paths.make_image_uri(id)],
            task_options,
            ground_truth_labels_uri=cowc_potsdam_paths.make_label_uri(id))

    train_scenes = [make_scene(id) for id in train_scene_ids]
    validation_scenes = [make_scene(id) for id in validation_scene_ids]

    batch_size, num_iters = get_hyperparams(model_config, use_tiny_dataset)
    # Could swap in custom pretrained model or backend_config_uri if you
    # wanted.
    backend_config_uri = helper.get_backend_config_uri(
        model_config, batch_size, num_iters)
    pretrained_model_uri = helper.get_pretrained_model_uri(model_config)

    # Custom predict options for object detection
    task_predict_options = None
    if task == OD:
        task_predict_options = ObjectDetectionPredictOptions(
            merge_thresh=0.1)
    experiment = Experiment(paths, model_config, train_scenes,
                            validation_scenes, backend_config_uri,
                            pretrained_model_uri,
                            task_predict_options=task_predict_options)
    experiment.save()


@click.command()
@click.argument('isprs_potsdam_uri')
@click.argument('cowc_potsdam_uri')
@click.argument('output_uri')
def main(isprs_potsdam_uri, cowc_potsdam_uri, output_uri):
    """Generate and save configs for experiments on COWC-Potsdam dataset.

    Args:
        isprs_potsdam_uri: URI of directory with raw data from ISPRS Potsdam
            dataset
        cowc_potsdam_uri: URI of directory with COWC-Potsdam data
        output_uri: URI of root directory of all output
    """
    use_tiny_dataset_vals = [True, False]
    tasks = [OD, CL]
    for task in tasks:
        for use_tiny_dataset in use_tiny_dataset_vals:
            name = 'classification' if task == CL else 'object-detection'
            if use_tiny_dataset:
                name += '-tiny'

            experiment_uri = join(output_uri, name)
            generate_experiment(experiment_uri, isprs_potsdam_uri,
                                cowc_potsdam_uri, use_tiny_dataset, task)


if __name__ == '__main__':
    main()
