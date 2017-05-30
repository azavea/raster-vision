from os.path import join
from copy import deepcopy

from rastervision.experiment_generator import (
    ExperimentGenerator, get_parent_dir)


class TestExperimentGenerator(ExperimentGenerator):
    def generate_experiments(self):
        base_exp = {
            'batch_size': 1,
            'git_commit': None,
            'problem_type': 'semseg',
            'dataset_name': 'isprs/potsdam',
            'generator_name': 'numpy',
            'active_input_inds': [0, 1, 2, 4, 5],
            'optimizer': 'adam',
            'init_lr': 1e-3,
            'target_size': [256, 256],
            'eval_target_size': [2000, 2000],
            'model_type': 'conv_logistic',
            'train_ratio': 0.8,
            'kernel_size': [5, 5],
            'epochs': 2,
            'nb_labels': 6,
            'validation_steps': 1,
            'nb_eval_samples': 1,
            'run_name': 'tests/semseg/potsdam_generator_quick_test',
            'steps_per_epoch': 2
        }

        init_lrs = [1e-3, 1e-4]

        exps = []
        exp_count = 0
        for init_lr in init_lrs:
            exp = deepcopy(base_exp)
            exp['init_lr'] = init_lr
            exp['run_name'] = join(exp['run_name'], str(exp_count))
            exps.append(exp)
            exp_count += 1

        return exps


if __name__ == '__main__':
    path = get_parent_dir(__file__)
    gen = TestExperimentGenerator().run(path)
