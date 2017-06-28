from os.path import join
from copy import deepcopy

from rastervision.experiment_generator import (
    ExperimentGenerator, get_parent_dir)
from rastervision.tagging.data.planet_kaggle import Dataset


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
            'run_name': 'tagging/tests/agg_concat_test',
            'steps_per_epoch': 2,
            'augment_methods': ['hflip', 'vflip', 'rotate', 'translate'],
            'active_tags_prob': 0.75
        }

        dataset = Dataset()
        active_tags_list = [
            dataset.atmos_tags, dataset.rare_tags, dataset.common_tags]

        exps = []
        exp_count = 0
        for active_tags in active_tags_list:
            exp = deepcopy(base_exp)
            exp['active_tags'] = active_tags
            exp['run_name'] = join(exp['run_name'], str(exp_count))
            exps.append(exp)
            exp_count += 1

        agg_exp = deepcopy(base_exp)
        agg_exp['run_name'] = join(base_exp['run_name'], str(exp_count))
        agg_exp['aggregate_run_names'] = [exp['run_name'] for exp in exps]
        agg_exp['aggregate_type'] = 'agg_concat'
        agg_exp['active_tags'] = dataset.all_tags
        exps.append(agg_exp)
        exp_count += 1

        return exps


if __name__ == '__main__':
    path = get_parent_dir(__file__)
    gen = TestExperimentGenerator().run(path)
