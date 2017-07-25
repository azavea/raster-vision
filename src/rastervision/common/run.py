from os.path import join, isdir, isfile
import json
import sys

from rastervision.common.utils import (
    Logger, make_sync_results, _makedirs, save_json, s3_download)
from rastervision.common.settings import results_path


class Runner():
    # Needs to be set/overridden by derived classes:
    # self.valid_tasks = None
    # self.model_factory_class = None
    # self.data_generator_factory_class = None
    # self.train_model_class = None
    # self.options_class = None
    # self.agg_file_names = None

    def is_valid_task(self, task):
        if task not in self.valid_tasks:
            return False

        return True

    def get_options(self, options_path):
        with open(options_path) as options_file:
            options_dict = json.load(options_file)
            options = self.options_class(options_dict)
            return options

    def setup_run(self):
        """Setup path for the results of a run.

        Creates directory if doesn't exist, downloads results from cloud, and
        write the options to <run_path>/options.json
        """
        if not isdir(self.run_path):
            self.sync_results(download=True)

        _makedirs(self.run_path)

        options_path = join(self.run_path, 'options.json')
        save_json(self.options.__dict__, options_path)

    def run_tasks(self, options_path, tasks):
        """Run tasks specified on command line.

        This creates the RunOptions object from the json file specified on the
        command line, creates a data generator, and then runs the tasks.
        """
        self.options = self.get_options(options_path)
        self.tasks = tasks
        if len(self.tasks) == 0:
            self.tasks = self.valid_tasks

        self.run_path = join(results_path, self.options.run_name)
        self.sync_results = make_sync_results(self.options.run_name)
        self.model_factory = self.model_factory_class()
        self.setup_run()
        sys.stdout = Logger(self.run_path)

        self.model = None
        self.generator = self.data_generator_factory_class() \
                             .get_data_generator(self.options)

        if self.options.aggregate_type is None:
            self.model = self.model_factory.get_model(
                self.run_path, self.options, self.generator,
                use_best=self.options.use_best_model)
        else:
            for run_name in self.options.aggregate_run_names:
                for file_name in self.agg_file_names:
                    if not isfile(join(results_path, run_name, file_name)):
                        s3_download(run_name, file_name)

        for task in self.tasks:
            if self.is_valid_task(task):
                print('Running task: {}'.format(task))
                self.run_task(task)
            else:
                print('{} is not a valid task'.format(task))

            self.sync_results()
