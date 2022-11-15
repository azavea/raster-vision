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
