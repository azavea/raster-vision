from os.path import join
from copy import deepcopy

from rastervision.experiment_generator import (
    ExperimentGenerator, get_parent_dir)


class TestExperimentGenerator(ExperimentGenerator):
    def generate_experiments(self):
        base_exp = {
            'batch_size': 8,
            'problem_type': 'tagging',
            'dataset_name': 'planet_kaggle',
            'generator_name': 'jpg',
            'active_input_inds': [0, 1, 2],
            'use_pretraining': True,
            'optimizer': 'adam',
            'lr_schedule': [[0, 0.0001], [10, 0.00001]],
            'model_type': 'densenet121',
            'train_ratio': 0.8,
            'epochs': 15,
            'nb_eval_plot_samples': 100,
            'validation_steps': 600,
            'run_name': 'tagging/7_18_17/80/densenet_transform',
            'steps_per_epoch': 2400,
            'augment_methods': ['hflip', 'vflip', 'rotate', 'zoom',
                                'translate']
        }

        exps = []
        nb_exps = 2
        for exp_ind in range(nb_exps):
            exp = deepcopy(base_exp)
            exp['run_name'] = join(exp['run_name'], str(exp_ind))
            exps.append(exp)

        return exps


if __name__ == '__main__':
    path = get_parent_dir(__file__)
    gen = TestExperimentGenerator().run(path)
