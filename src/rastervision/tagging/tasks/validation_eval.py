from os.path import join, isfile
import json

from rastervision.tagging.data.planet_kaggle import TagStore
from rastervision.tagging.utils import f2_score

VALIDATION_EVAL = 'validation_eval'


class Scores():
    """A set of scores for the performance of a model on a dataset."""
    def __init__(self):
        self.f2 = None

    def to_json(self):
        return json.dumps(self.__dict__, sort_keys=True, indent=4)

    def save(self, path):
        scores_json = self.to_json()
        with open(path, 'w') as scores_file:
            scores_file.write(scores_json)


def validation_eval(run_path, generator):
    y_true = generator.tag_store.get_tag_array(generator.validation_file_inds)

    predictions_path = join(run_path, 'validation_predictions.csv')
    if not isfile(predictions_path):
        raise Exception(
            'validation_predict needs to be run before validation_eval.')
    predictions_tag_store = TagStore(predictions_path)
    y_pred = predictions_tag_store.get_tag_array(
        generator.validation_file_inds)

    scores = Scores()
    scores.f2 = f2_score(y_true, y_pred)
    scores_path = join(run_path, 'scores.json')
    scores.save(scores_path)
