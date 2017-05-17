from os.path import join
import sys

from rastervision.common.utils import Logger, make_sync_results, setup_run
from rastervision.common.tasks.plot_curves import plot_curves, PLOT_CURVES
from rastervision.common.tasks.train_model import TRAIN_MODEL
from rastervision.common.settings import results_path, datasets_path

from .options import TaggingOptions
from .data.factory import get_data_generator
from .models.factory import TaggingModelFactory
from .tasks.train_model import TaggingTrainModel

valid_tasks = [TRAIN_MODEL, PLOT_CURVES]


def run_tasks(options_dict, tasks):
    """Run tasks specified on command line.

    This creates the RunOptions object from the json file specified on the
    command line, creates a data generator, and then runs the tasks.
    """
    options = TaggingOptions(options_dict)
    model_factory = TaggingModelFactory()
    generator = get_data_generator(options)
    run_path = join(results_path, options.run_name)

    sync_results = make_sync_results(options.run_name)
    setup_run(run_path, options, sync_results)
    sys.stdout = Logger(run_path)

    if len(tasks) == 0:
        tasks = valid_tasks

    for task in tasks:
        if task not in valid_tasks:
            raise ValueError('{} is not a valid task'.format(task))

    for task in tasks:
        if task == TRAIN_MODEL:
            model = model_factory.get_model(
                run_path, options, generator, use_best=False)
            train_model = TaggingTrainModel(
                run_path, sync_results, options, generator, model)
            train_model.train_model()
        elif task == PLOT_CURVES:
            plot_curves(run_path)

        sync_results()
