from os.path import join
from copy import deepcopy

from rastervision.experiment_generator import (
    ExperimentGenerator, get_parent_dir)


class TestExperimentGenerator(ExperimentGenerator):
    def generate_experiments(self):
        base_exp = {
            'batch_size': 1,
            'problem_type': 'tagging',
            'dataset_name': 'planet_kaggle',
            'generator_name': 'jpg',
            'active_input_inds': [0, 1, 2],
            'use_pretraining': True,
            'optimizer': 'adam',
            'init_lr': 1e-3,
            'model_type': 'baseline_resnet',
            'train_ratio': 0.8,
            'epochs': 2,
            'nb_eval_samples': 10,
            'nb_eval_plot_samples': 3,
            'validation_steps': 1,
            'run_name': 'tagging/tests/quick_test',
            'steps_per_epoch': 2,
            'augment_types': ['hflip', 'vflip', 'rotate', 'translate']
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

        agg_exp = {
            'problem_type': base_exp['problem_type'],
            'run_name': join(base_exp['run_name'], str(exp_count)),
            'aggregate_run_names': [exp['run_name'] for exp in exps]
        }
        exps.append(agg_exp)
        exp_count += 1

        return exps


if __name__ == '__main__':
    path = get_parent_dir(__file__)
    gen = TestExperimentGenerator().run(path)
