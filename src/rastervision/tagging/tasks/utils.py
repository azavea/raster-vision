import numpy as np


def compute_prediction(y_probs, dataset, tag_store, thresholds=None):
    if thresholds is None:
        # If no thresholds, assume softmax loss.
        y_pred = np.zeros((y_probs.shape))
        y_pred[np.argmax(y_probs)] = 1
        return y_pred
    return (y_probs > thresholds).astype(np.float32)
