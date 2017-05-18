from sklearn.metrics import fbeta_score
import numpy as np


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
