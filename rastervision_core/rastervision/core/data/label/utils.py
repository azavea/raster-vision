from typing import (TYPE_CHECKING, Iterable, Iterator, List, Tuple)
if TYPE_CHECKING:
    import numpy as np
    from rastervision.core.box import Box


def discard_prediction_edges(
        windows: Iterable['Box'], predictions: Iterable['np.ndarray'],
        crop_sz: int) -> Tuple[List['Box'], Iterator['np.ndarray']]:
    """Discard the edges of predicted chips.

    Args:
        windows (Iterable[Box]): The windows corresponding to the chips.
        predictions (Iterable[np.ndarray]): The predicted chips.
        crop_sz (int): Number of pixel rows/cols to discard.

    Returns:
        Tuple[Iterator[Box], Iterator[np.ndarray]]: Cropped windows and chips.
    """
    windows_cropped = [w.center_crop(crop_sz, crop_sz) for w in windows]
    array_slices = [
        wc.to_offsets(w).to_slices() for w, wc in zip(windows, windows_cropped)
    ]
    predictions_cropped = (p[..., yslice, xslice]
                           for p, (xslice,
                                   yslice) in zip(predictions, array_slices))
    return windows_cropped, predictions_cropped
