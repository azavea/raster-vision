import tempfile

from PIL import Image
import numpy as np
import click
import mercantile
import rasterio
from rasterio.windows import Window
import pyproj

from rastervision.pipeline.file_system import (download_if_needed,
                                               get_local_path, upload_or_copy)
from rastervision.core.utils.cog import create_cog


def lnglat2merc(lng, lat):
    """Convert lng, lat point to x/y Web Mercator tuple."""
    return pyproj.transform(
        pyproj.Proj(init='epsg:4326'), pyproj.Proj(init='epsg:3857'), lng, lat)


def merc2lnglat(x, y):
    """Convert x, y Web Mercator point to lng/lat tuple."""
    return pyproj.transform(
        pyproj.Proj(init='epsg:3857'), pyproj.Proj(init='epsg:4326'), x, y)


def merc2pixel(tile_x, tile_y, zoom, merc_x, merc_y, tile_sz=256):
    """Convert Web Mercator point to pixel coordinates.

    This is within the coordinate frame of a single ZXY tile.

    Args:
        tile_x: (int) x coordinate of ZXY tile
        tile_y: (int) y coordinate of ZXY tile
        zoom: (int) zoom level of ZXY tile
        merc_x: (float) Web Mercator x axis of point
        merc_y: (float) Web Mercator y axis of point
        tile_sz: (int) size of ZXY tile
    """
    tile_merc_bounds = mercantile.xy_bounds(tile_x, tile_y, zoom)
    pix_y = int(
        round(tile_sz * ((tile_merc_bounds.top - merc_y) /
                         (tile_merc_bounds.top - tile_merc_bounds.bottom))))
    pix_x = int(
        round(tile_sz * ((merc_x - tile_merc_bounds.left) /
                         (tile_merc_bounds.right - tile_merc_bounds.left))))
    return (pix_x, pix_y)


def _zxy2geotiff(tile_schema, zoom, bounds, output_uri, make_cog=False):
    """Generates a GeoTIFF of a bounded region from a ZXY tile server.

    Args:
        tile_schema: (str) the URI schema for zxy tiles (ie. a slippy map tile server)
            of the form /tileserver-uri/{z}/{x}/{y}.png. If {-y} is used, the tiles
            are assumed to be indexed using TMS coordinates, where the y axis starts
            at the southernmost point. The URI can be for http, S3, or the local
            file system.
        zoom: (int) the zoom level to use when retrieving tiles
        bounds: (list) a list of length 4 containing min_lat, min_lng,
            max_lat, max_lng
        output_uri: (str) where to save the GeoTIFF. The URI can be for http, S3, or the
            local file system
    """
    min_lat, min_lng, max_lat, max_lng = bounds
    if min_lat >= max_lat:
        raise ValueError('min_lat must be < max_lat')
    if min_lng >= max_lng:
        raise ValueError('min_lng must be < max_lng')

    is_tms = False
    if '{-y}' in tile_schema:
        tile_schema = tile_schema.replace('{-y}', '{y}')
        is_tms = True

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name

    # Get range of tiles that cover bounds.
    output_path = get_local_path(output_uri, tmp_dir)
    tile_sz = 256
    t = mercantile.tile(min_lng, max_lat, zoom)
    xmin, ymin = t.x, t.y
    t = mercantile.tile(max_lng, min_lat, zoom)
    xmax, ymax = t.x, t.y

    # The supplied bounds are contained within the "tile bounds" -- ie. the
    # bounds of the set of tiles that covers the supplied bounds. Therefore,
    # we need to crop out the imagery that lies within the supplied bounds.
    # We do this by computing a top, bottom, left, and right offset in pixel
    # units of the supplied bounds against the tile bounds. Getting the offsets
    # in pixel units involves converting lng/lat to web mercator units since we
    # assume that is the CRS of the tiles. These offsets are then used to crop
    # individual tiles and place them correctly into the output raster.
    nw_merc_x, nw_merc_y = lnglat2merc(min_lng, max_lat)
    left_pix_offset, top_pix_offset = merc2pixel(xmin, ymin, zoom, nw_merc_x,
                                                 nw_merc_y)

    se_merc_x, se_merc_y = lnglat2merc(max_lng, min_lat)
    se_left_pix_offset, se_top_pix_offset = merc2pixel(xmax, ymax, zoom,
                                                       se_merc_x, se_merc_y)
    right_pix_offset = tile_sz - se_left_pix_offset
    bottom_pix_offset = tile_sz - se_top_pix_offset

    uncropped_height = tile_sz * (ymax - ymin + 1)
    uncropped_width = tile_sz * (xmax - xmin + 1)
    height = uncropped_height - top_pix_offset - bottom_pix_offset
    width = uncropped_width - left_pix_offset - right_pix_offset

    transform = rasterio.transform.from_bounds(nw_merc_x, se_merc_y, se_merc_x,
                                               nw_merc_y, width, height)
    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            crs='epsg:3857',
            transform=transform,
            dtype=rasterio.uint8) as dataset:
        out_x = 0
        for xi, x in enumerate(range(xmin, xmax + 1)):
            tile_xmin, tile_xmax = 0, tile_sz - 1
            if x == xmin:
                tile_xmin += left_pix_offset
            if x == xmax:
                tile_xmax -= right_pix_offset
            window_width = tile_xmax - tile_xmin + 1

            out_y = 0
            for yi, y in enumerate(range(ymin, ymax + 1)):
                tile_ymin, tile_ymax = 0, tile_sz - 1
                if y == ymin:
                    tile_ymin += top_pix_offset
                if y == ymax:
                    tile_ymax -= bottom_pix_offset
                window_height = tile_ymax - tile_ymin + 1

                # Convert from xyz to tms if needed.
                # https://gist.github.com/tmcw/4954720
                if is_tms:
                    y = (2**zoom) - y - 1
                tile_uri = tile_schema.format(x=x, y=y, z=zoom)
                tile_path = download_if_needed(tile_uri, tmp_dir)
                img = np.array(Image.open(tile_path))
                img = img[tile_ymin:tile_ymax + 1, tile_xmin:tile_xmax + 1, :]

                window = Window(out_x, out_y, window_width, window_height)
                dataset.write(
                    np.transpose(img[:, :, 0:3], (2, 0, 1)), window=window)
                out_y += window_height
            out_x += window_width

    if make_cog:
        create_cog(output_path, output_uri, tmp_dir)
    else:
        upload_or_copy(output_path, output_uri)


@click.command()
@click.argument('tile_schema')
@click.argument('zoom')
@click.argument('bounds')
@click.argument('output_uri')
@click.option('--make-cog', is_flag=True, default=False)
def zxy2geotiff(tile_schema, zoom, bounds, output_uri, make_cog):
    """Generates a GeoTIFF of a bounded region from a ZXY tile server.

    TILE_SCHEMA: the URI schema for zxy tiles (ie. a slippy map tile server) of
    the form /tileserver-uri/{z}/{x}/{y}.png. If {-y} is used, the tiles are
    assumed to be indexed using TMS coordinates, where the y axis starts at
    the southernmost point. The URI can be for http, S3, or the local file
    system.

    ZOOM: the zoom level to use when retrieving tiles

    BOUNDS: a space-separated string containing min_lat, min_lng, max_lat,
    max_lng

    OUTPUT_URI: where to save the GeoTIFF. The URI can be for http, S3, or the
    local file system.
    """
    bounds = [float(x) for x in bounds.split(' ')]
    _zxy2geotiff(tile_schema, int(zoom), bounds, output_uri, make_cog=make_cog)


if __name__ == '__main__':
    zxy2geotiff()
