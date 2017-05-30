from os.path import dirname, abspath, join
import json

from rastervision.options import make_options

from rastervision.common.utils import _makedirs


def get_parent_dir(f):
    return dirname(abspath(f))


class ExperimentGenerator():
    def run(self, path):
        exps = self.generate_experiments()
        self.save_to_dir(exps, path)

    def save_to_dir(self, experiments, path):
        if not self.has_unique_run_names(experiments):
            raise ValueError('Each run_name needs to be unique.')

        for exp_ind, exp in enumerate(experiments):
            self.parse_experiment(exp)

            json_str = json.dumps(exp, sort_keys=True, indent=4)
            exp_path = join(path, 'experiments', '{}.json'.format(exp_ind))
            _makedirs(dirname(exp_path))
            with open(exp_path, 'w') as exp_file:
                exp_file.write(json_str)

    def parse_experiment(self, experiment):
        # Will raise exception if it can't be parsed.
        make_options(experiment)

    def has_unique_run_names(self, experiments):
        run_names = [exp['run_name'] for exp in experiments]
        return len(run_names) == len(set(run_names))

    def generate_experiments(self):
        raise NotImplementedError()
