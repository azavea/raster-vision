from os.path import join
import json

from rastervision.common.settings import results_path
from rastervision.common.utils import save_json


def aggregate_scores(options, best_score_key):
    runs = []
    best_run_name = None
    best_score = -float('inf')

    for run_name in options.aggregate_run_names:
        run_path = join(results_path, run_name)
        scores_path = join(run_path, 'scores.json')
        with open(scores_path) as scores_file:
            scores_dict = json.load(scores_file)
            score = scores_dict[best_score_key]
            if score > best_score:
                best_score = score
                best_run_name = run_name

            runs.append({
                'run_name': run_name,
                'scores': scores_dict
            })

    agg_scores = {
        'best_run_name': best_run_name,
        'best_score': best_score,
        'runs': runs
    }

    scores_path = join(results_path, options.run_name, 'scores.json')
    save_json(agg_scores, scores_path)
