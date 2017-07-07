import numpy as np


def compute_prediction(y_probs, dataset, tag_store, thresholds):
    return (y_probs > thresholds).astype(np.float32)
