from os.path import join
from copy import deepcopy

from rastervision.experiment_generator import (
    ExperimentGenerator, get_parent_dir)


class TestExperimentGenerator(ExperimentGenerator):
    def generate_experiments(self):
        base_exp = {
            'batch_size': 32,
            'problem_type': 'tagging',
            'dataset_name': 'planet_kaggle',
            'generator_name': 'jpg',
            'active_input_inds': [0, 1, 2],
            'use_pretraining': True,
            'optimizer': 'sgd',
            'lr_schedule': [[0, 1e-2], [15, 1e-3]],
            'nesterov': True,
            'momentum': 0.9,
            'train_ratio': 0.99,
            'nb_eval_plot_samples': 100,
            'use_best_model': False,
            'epochs': 30,
            'validation_steps': 4,
            'run_name': 'tagging/7_19_17/99/inception/',
            'steps_per_epoch': 800,
            'augment_methods': ['hflip', 'vflip', 'rotate90'],
            'model_type': 'inception_v3'
        }
        exps = []
        exp_count = 0
        for run_ind in range(5):
            exp = deepcopy(base_exp)
            exp['run_name'] = join(exp['run_name'], str(exp_count))
            exps.append(exp)
            exp_count += 1

        return exps


if __name__ == '__main__':
    path = get_parent_dir(__file__)
    gen = TestExperimentGenerator().run(path)
