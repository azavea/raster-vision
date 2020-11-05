# from typing import
import numpy as np


def nodata_below_threshold(chip: np.ndarray,
                           threshold: float,
                           nodata_val: int = 0) -> bool:
    """ Check if proportion of nodata pixels is below the threshold. """
    if len(chip.shape) == 3:
        chip = chip.sum(axis=-1)
    nodata_prop = (chip == 0).mean()
    return nodata_prop < threshold


def fill_no_data(img: np.ndarray, label_arr: np.ndarray,
                 null_class_id: int) -> None:
    """ If chip has null labels, fill in those pixels with nodata. """
    mask = label_arr == null_class_id
    if np.any(mask):
        img[mask, :] = 0
    return img
