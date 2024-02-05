import numpy as np


def nodata_below_threshold(chip: np.ndarray,
                           threshold: float,
                           nodata_val: int = 0) -> bool:
    """Check if fraction of nodata pixels is below the threshold.

    Args:
        chip (np.ndarray): Raster as (..., H, W[, C]) numpy array.
        threshold (float): Threshold to check the fraction of NODATA pixels
            against.
        nodata_val (int, optional): Value that represents NODATA pixels.
            Defaults to 0.

    Returns:
        bool: Whether the fraction of NODATA pixels is below the given
        threshold.
    """
    if chip.ndim > 2:
        # (..., w, h, c) --> (..., w, h)
        chip = chip.sum(axis=-1)
    nodata_frac = (chip == nodata_val).mean()
    return nodata_frac < threshold
