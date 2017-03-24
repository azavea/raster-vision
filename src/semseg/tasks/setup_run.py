from os.path import join, isdir
import json

from ..data.utils import _makedirs

SETUP_RUN = 'setup_run'


def setup_run(run_path, options, sync_results):
    """Setup path for the results of a run.

    Creates directory if doesn't exist, downloads results from cloud, and
    copies the options.json file to run_path.

    # Arguments
        run_path: the path to the files for a run
        options: RunOptions object that specifies the run
        sync_results: function used to sync results with cloud
    """
    if not isdir(run_path):
        sync_results(download=True)

    _makedirs(run_path)

    # TODO just copy the file over
    # Read the options file and write it to the run directory.
    options_json = json.dumps(options.__dict__, sort_keys=True, indent=4)
    options_path = join(run_path, 'options.json')
    with open(options_path, 'w') as options_file:
        options_file.write(options_json)

    return run_path
