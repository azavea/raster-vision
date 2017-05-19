from os.path import join, isfile
import json

from sklearn.metrics import fbeta_score
import numpy as np

from rastervision.tagging.data.planet_kaggle import TagStore

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


# Copied from https://www.kaggle.com/anokas/fixed-f2-score-in-python
def f2_score(y_true, y_pred):
    """Compute F2 score.
       Note that with f2 score, your predictions need to be binary.
    """
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method
    # will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


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
