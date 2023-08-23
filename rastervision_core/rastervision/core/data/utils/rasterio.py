from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union
import os
import subprocess
import logging

import numpy as np
import rasterio as rio
import rasterio.windows as rio_windows
from rasterio.transform import from_origin
from rasterio.enums import (ColorInterp, MaskFlags, Resampling)

from rastervision.pipeline.file_system.utils import (
    file_to_json, get_local_path, get_tmp_dir, make_dir, upload_or_copy,
    download_if_needed)
from rastervision.core.box import Box

if TYPE_CHECKING:
    from rasterio.io import DatasetReader

log = logging.getLogger(__name__)


def write_window(dataset: 'DatasetReader',
                 arr: np.ndarray,
                 window: Optional[Box] = None) -> None:
    """Write a (H, W[, C]) array out to a rasterio dataset.

    Args:
        dataset (DatasetReader): Rasterio dataset, opened for writing.
        arr (np.ndarray): Array to write.
        window (Optional[Box]): Window (in pixel coords) to write to.
            Defaults to None.
    """
    if window is not None:
        window = window.rasterio_format()
    if arr.ndim == 2:
        dataset.write_band(1, arr, window=window)
    else:
        arr_chw = arr.transpose(2, 0, 1)
        for i, band in enumerate(arr_chw, start=1):
            dataset.write_band(i, band, window=window)


def write_bbox(path: str, arr: np.ndarray, bbox: Box, crs_wkt: str, **kwargs):
    """Write a (H, W[, C]) array to a GeoTIFF, georeferenced to the given bbox.

    Args:
        path (str): GeoTiff path.
        arr (np.ndarray): (H, W[, C]) Array to write.
        bbox (Box): Bouding box in map coords to georeference the GeoTiff to.
        crs_wkt (str): CRS in WKT format.
    """
    if arr.ndim == 2:
        h_arr, w_arr = arr.shape
        num_channels = 1
    else:
        h_arr, w_arr, num_channels = arr.shape
    h_bbox, w_bbox = bbox.size
    resolution = h_bbox / h_arr, w_bbox / w_arr
    transform = from_origin(bbox.xmin, bbox.ymax, *resolution)
    out_profile = dict(
        driver='GTiff',
        height=h_arr,
        width=w_arr,
        crs=crs_wkt,
        count=num_channels,
        dtype=arr.dtype,
        transform=transform,
    )
    out_profile.update(kwargs)
    with rio.open(path, 'w', **out_profile) as ds:
        write_window(ds, arr)


def write_geotiff_like_geojson(path: str,
                               arr: np.ndarray,
                               geojson_path: str,
                               crs: Optional[str] = None,
                               **kwargs) -> None:
    """Write array to GeoTIFF, georeferenced to same bbox as the given GeoJSON.

    Args:
        path (str): GeoTiff path.
        arr (np.ndarray): (H, W[, C]) Array to write.
        geojson_path (str): GeoJSON path.
        crs (str): CRS name. If None, read from the GeoJSON. If not specified
            in the GeoJSON, use "EPSG:4326". Defaults to None.
    """
    from rastervision.core.data.utils.geojson import geojson_to_geoms
    import pyproj
    from shapely.ops import unary_union

    geojson = file_to_json(geojson_path)
    if crs is None:
        try:
            crs = geojson['crs']['properties']['name']
        except KeyError:
            crs = 'epsg:4326'
    crs_wkt = pyproj.CRS(crs).to_wkt()
    geoms = unary_union(list(geojson_to_geoms(geojson)))
    bbox = Box.from_shapely(geoms).normalize()
    write_bbox(path, arr, bbox=bbox, crs_wkt=crs_wkt, **kwargs)


def crop_geotiff(src_uri: str, window: Box, dst_uri: str):
    """Create a new GeoTIFF from a crop of an existing GeoTIFF.

    Args:
        src_uri (str): Source GeoTIFF URI to read from.
        window (Box): Window specifying the crop bounds.
        dst_uri (str): Crop GeoTIFF URI to write to.
    """
    rio_window = window.rasterio_format()

    with rio.open(src_uri) as src_ds, get_tmp_dir() as tmp_dir:
        crop_path = get_local_path(dst_uri, tmp_dir)
        make_dir(crop_path, use_dirname=True)

        meta = src_ds.meta
        colorinterp = src_ds.colorinterp
        img_cropped = src_ds.read(window=rio_window)

        meta['height'], meta['width'] = window.size
        meta['transform'] = rio_windows.transform(rio_window, src_ds.transform)

        with rio.open(crop_path, 'w', **meta) as dst_ds:
            dst_ds.colorinterp = colorinterp
            dst_ds.write(img_cropped)

        upload_or_copy(crop_path, dst_uri)


def build_vrt(vrt_path: str, image_uris: List[str]) -> None:
    """Build a VRT for a set of TIFF files.

    Args:
        vrt_path (str): Local path for the VRT to be created.
        image_uris (List[str]): Image URIs.
    """
    log.info('Building VRT...')
    cmd = ['gdalbuildvrt', vrt_path]
    cmd.extend(image_uris)
    subprocess.run(cmd)


def download_and_build_vrt(image_uris: List[str],
                           vrt_dir: str,
                           stream: bool = False) -> str:
    """Download images (if needed) and build a VRT for a set of TIFF files.

    Args:
        image_uris (List[str]): Image URIs.
        vrt_dir (str): Dir where the VRT will be created.
        stream (bool, optional): If true, do not download images.
            Defaults to False.

    Returns:
        str: The path to the created VRT file.
    """
    if not stream:
        image_uris = [download_if_needed(uri) for uri in image_uris]
    vrt_path = os.path.join(vrt_dir, 'index.vrt')
    build_vrt(vrt_path, image_uris)
    return vrt_path


def read_window(
        dataset: 'DatasetReader',
        bands: Optional[Union[int, Sequence[int]]] = None,
        window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        is_masked: bool = False,
        out_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Load a window of an image using Rasterio.

    Args:
        dataset: a Rasterio dataset.
        bands (Optional[Union[int, Sequence[int]]]): Band index or indices to
            read. Must be 1-indexed.
        window (Optional[Tuple[Tuple[int, int], Tuple[int, int]]]):
            ((row_start, row_stop), (col_start, col_stop)) or
            ((y_min, y_max), (x_min, x_max)). If None, reads the entire raster.
            Defaults to None.
        is_masked (bool): If True, read a masked array from rasterio.
            Defaults to False.
        out_shape (Optional[Tuple[int, int]]): (hieght, width) of the output
            chip. If None, no resizing is done. Defaults to None.

    Returns:
        np.ndarray: array of shape (height, width, channels).
    """
    if bands is not None:
        bands = tuple(bands)
    im = dataset.read(
        indexes=bands,
        window=window,
        boundless=True,
        masked=is_masked,
        out_shape=out_shape,
        resampling=Resampling.bilinear)

    if is_masked:
        im = np.ma.filled(im, fill_value=0)

    # Handle non-zero NODATA values by setting the data to 0.
    if bands is None:
        for channel, nodataval in enumerate(dataset.nodatavals):
            if nodataval is not None and nodataval != 0:
                im[channel, im[channel] == nodataval] = 0
    else:
        for channel, src_band in enumerate(bands):
            src_band_0_indexed = src_band - 1
            nodataval = dataset.nodatavals[src_band_0_indexed]
            if nodataval is not None and nodataval != 0:
                im[channel, im[channel] == nodataval] = 0

    im = np.transpose(im, axes=[1, 2, 0])
    return im


def get_channel_order_from_dataset(dataset: 'DatasetReader') -> List[int]:
    """Get channel order from rasterio image dataset.

    Accounts for dataset's ``colorinterp`` if defined.

    Args:
        dataset (DatasetReader): Rasterio image dataset.

    Returns:
        List[int]: List of channel indices.
    """
    colorinterp = dataset.colorinterp
    if colorinterp:
        channel_order = [
            i for i, color_interp in enumerate(colorinterp)
            if color_interp != ColorInterp.alpha
        ]
    else:
        channel_order = list(range(0, dataset.count))
    return channel_order


def is_masked(dataset: 'DatasetReader') -> bool:
    """Check if dataset has any masks defined."""
    mask_flags = dataset.mask_flag_enums
    is_masked = any(m for m in mask_flags if m != MaskFlags.all_valid)
    return is_masked
