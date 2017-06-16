import numpy as np


def compute_prediction(y_probs, dataset, thresholds):
    atmos_inds = [dataset.get_tag_ind(tag)
                  for tag in dataset.atmos_tags]

    y_pred = (y_probs > thresholds).astype(np.float32)

    # TODO remove this post-processing step once our model
    # enforces the constraint that there is at least one atmospheric
    # tag.
    if np.sum(y_pred[atmos_inds]) == 0:
        max_ind = np.argmax(y_probs[atmos_inds])
        max_tag = dataset.atmos_tags[max_ind]
        max_ind = dataset.get_tag_ind(max_tag)
        y_pred[max_ind] = 1

    return y_pred
