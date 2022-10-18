###############################################################################
# Ported over from https://github.com/azavea/mask-to-polygons.
###############################################################################

from typing import TYPE_CHECKING, Iterator, Optional, Tuple
from itertools import chain

import numpy as np
import cv2
import rasterio as rio
from shapely.geometry import shape

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

RotatedRectange = Tuple[Tuple[float, float], Tuple[float, float], float]


def mask_to_polygons(mask: np.ndarray, transform: Optional[rio.Affine] = None
                     ) -> Iterator['BaseGeometry']:
    """Polygonize a raster mask. Wrapper around rasterio.features.shapes.

    Args:
        mask (np.ndarray): The mask containing buildings to polygonize.
        transform (Optional[rio.Affine]): Affine transform to use during
            polygonization. Deafults to None (i.e. identity transform).

    Returns:
        Iterator[BaseGeometry]: Generator of shapely polygons.
    """
    if transform is None:
        transform = rio.Affine.identity()
    shapes = rio.features.shapes(mask, mask=(mask == 1), transform=transform)
    polygons = (shape(s) for s, v in shapes if v == 1)
    return polygons


def mask_to_building_polygons(
        mask: np.ndarray,
        transform: Optional[rio.Affine] = None,
        min_area: float = 100,
        width_factor: float = 0.5,
        thickness: float = 0.001) -> Iterator['BaseGeometry']:
    """Try to break up building clusters and then convert to polygons.

    Perofrms the following steps:

    1.  Identify connected components in ``mask``.
    2.  For each connected component, if >= ``min_area``:

        a.  Generate a kernel based on its dimensions and orientation and
            the ``width_factor`` and ``thickness`` params.
        b.  Use the kernel to apply morphological erosion to component-
            mask.
        c.  Identify connected sub-components in component-mask.
        d.  For each connected sub-component, if >= ``min_area``:

            1. Apply morphological dilation using the kernel from above.
            2. Polygonize using ``mask_to_polygons()``.

    Args:
        mask (np.ndarray): The mask containing buildings to polygonize.
        transform (Optional[rio.Affine]): Affine transform to use during
            polygonization. Deafults to None (i.e. identity transform).
        min_area (float): Minimum area (in pixels^2) of anything that can be
            considered to be a building or cluster of buildings. The goal is to
            distinguish between buildings and artifacts. Components with area
            less than this value will be discarded. Defaults to 100.
        width_factor (float): Width of the structural element used to break
            building clusters as a fraction of the width of the cluster.
        thickness (float): Thickness of the structural element that is used to
            break building clusters. Defaults to 0.001.

    Returns:
        Iterator[BaseGeometry]: Generator of shapely polygons.
    """
    n, components = cv2.connectedComponents(mask)

    iterators = []
    for i in range(1, n):
        component = (components == i).astype(np.uint8)
        if component.sum() < min_area:
            continue

        rectangle = get_rectangle(component)
        kernel = get_kernel(rectangle, width_factor, thickness)
        if kernel is None:
            iterators.append(mask_to_polygons(component, transform))
            continue
        eroded = cv2.morphologyEx(
            component, cv2.MORPH_ERODE, kernel, iterations=1)
        m, sub_components = cv2.connectedComponents(eroded)

        for j in range(1, m):
            sub_component = (sub_components == j).astype(np.uint8)
            if sub_component.sum() < min_area:
                continue
            sub_component_dilated = cv2.morphologyEx(
                sub_component, cv2.MORPH_DILATE, kernel, iterations=1)
            iterators.append(
                mask_to_polygons(sub_component_dilated, transform))

    return chain.from_iterable(iterators)


def get_rectangle(buildings: np.ndarray) -> Optional[RotatedRectange]:
    contours, _ = cv2.findContours(buildings, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        rectangle = cv2.minAreaRect(contours[0])
        return rectangle
    else:
        return None


def get_kernel(rectangle: RotatedRectange,
               width_factor: float = 0.5,
               thickness: float = 0.001) -> Optional[np.ndarray]:
    ((cx, cy), (xwidth, ywidth), angle) = rectangle

    width = int(width_factor * min(xwidth, ywidth))

    kernel = np.zeros((width, width), dtype=np.uint8)
    try:
        kernel = cv2.cvtColor(kernel, cv2.COLOR_GRAY2BGR)
    except Exception:
        return None

    diagonal = width * np.sqrt(2)
    pos = (width // 2, width // 2)
    dim = (diagonal, thickness)
    if ywidth < xwidth:
        angle += 90
    element_rect: RotatedRectange = (pos, dim, angle)
    element_contour = cv2.boxPoints(box=element_rect)
    # https://stackoverflow.com/questions/48350693/what-is-numpy-method-int0
    element_contour = np.int0(element_contour)

    cv2.drawContours(
        image=kernel,
        contours=[element_contour],
        contourIdx=0,
        color=(1, 0, 0),
        # -1 means fill interior
        thickness=-1)
    kernel = kernel[:, :, 0]

    return kernel


# Adapted from https://github.com/mapbox/robosat/blob/a8e0e3d676b454b61df03897e43e003867b6ef48/robosat/features/core.py#L65-L77  # noqa
def denoise(mask: np.ndarray, radius: int) -> np.ndarray:
    """Apply morphological opening /w circular kernel to remove hi-freq noise.

    Args:
        mask (np.ndarray): the binary mask to remove noise from.
        radius (int): size in pixels of kernel for morphological op.

    Returns:
        np.ndarray: The mask after applying denoising.
    """
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_ELLIPSE, ksize=(radius, radius))
    out = cv2.morphologyEx(src=mask, op=cv2.MORPH_OPEN, kernel=kernel)
    return out
