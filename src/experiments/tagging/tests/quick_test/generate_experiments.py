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
            'augment_methods': ['hflip', 'vflip', 'rotate', 'translate'],
            'rare_sample_prob': 0.5
        }

        model_types = ['baseline_resnet', 'densenet121']
        freeze_base = [False, True]

        exps = []
        exp_count = 0
        for model_type, freeze_base in zip(model_types, freeze_base):
            exp = deepcopy(base_exp)
            exp['run_name'] = join(exp['run_name'], str(exp_count))
            exp['model_type'] = model_type
            exp['freeze_base'] = freeze_base
            exps.append(exp)
            exp_count += 1

        agg_exp = deepcopy(base_exp)
        agg_exp['run_name'] = join(base_exp['run_name'], str(exp_count))
        agg_exp['aggregate_run_names'] = [exp['run_name'] for exp in exps]
        agg_exp['aggregate_type'] = 'agg_ensemble'
        exps.append(agg_exp)
        exp_count += 1

        return exps


if __name__ == '__main__':
    path = get_parent_dir(__file__)
    gen = TestExperimentGenerator().run(path)
