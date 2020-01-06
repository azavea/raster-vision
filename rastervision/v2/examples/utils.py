import csv
from io import StringIO
import tempfile
import os

import rasterio
from shapely.strtree import STRtree
from shapely.geometry import shape

from rastervision.core import Box
from rastervision.data import RasterioCRSTransformer, GeoJSONVectorSource
from rastervision.utils.files import (
    file_to_str, file_exists, get_local_path, upload_or_copy, make_dir,
    file_to_json)
from rastervision.filesystem import S3FileSystem


def str_to_bool(x):
    if type(x) == str:
        if x.lower() == 'true':
            return True
        elif x.lower() == 'false':
            return False
        else:
            raise ValueError('{} is expected to be true or false'.format(x))
    return x


def get_scene_info(csv_uri):
    csv_str = file_to_str(csv_uri)
    reader = csv.reader(StringIO(csv_str), delimiter=',')
    return list(reader)


def save_image_crop(image_uri, crop_uri, label_uri=None, size=600,
                    min_features=10):
    """Save a crop of an image to use for testing.

    If label_uri is set, the crop needs to cover >= min_features.

    Args:
        image_uri: URI of original image
        crop_uri: URI of cropped image to save
        label_uri: optional URI of GeoJSON file
        size: height and width of crop

    Raises:
        ValueError if cannot find a crop satisfying min_features constraint.
    """
    if not file_exists(crop_uri):
        print('Saving test crop to {}...'.format(crop_uri))
        old_environ = os.environ.copy()
        try:
            request_payer = S3FileSystem.get_request_payer()
            if request_payer == 'requester':
                os.environ['AWS_REQUEST_PAYER'] = request_payer
            im_dataset = rasterio.open(image_uri)
            h, w = im_dataset.height, im_dataset.width

            extent = Box(0, 0, h, w)
            windows = extent.get_windows(size, size)
            if label_uri is not None:
                crs_transformer = RasterioCRSTransformer.from_dataset(im_dataset)
                vs = GeoJSONVectorSource(label_uri, crs_transformer)
                geojson = vs.get_geojson()
                geoms = []
                for f in geojson['features']:
                    g = shape(f['geometry'])
                    geoms.append(g)
                tree = STRtree(geoms)

            for w in windows:
                use_window = True
                if label_uri is not None:
                    w_polys = tree.query(w.to_shapely())
                    use_window = len(w_polys) >= min_features

                if use_window:
                    w = w.rasterio_format()
                    im = im_dataset.read(window=w)

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        crop_path = get_local_path(crop_uri, tmp_dir)
                        make_dir(crop_path, use_dirname=True)

                        meta = im_dataset.meta
                        meta['width'], meta['height'] = size, size
                        meta['transform'] = rasterio.windows.transform(
                            w, im_dataset.transform)

                        with rasterio.open(crop_path, 'w', **meta) as dst:
                            dst.colorinterp = im_dataset.colorinterp
                            dst.write(im)

                        upload_or_copy(crop_path, crop_uri)
                    break

            if not use_window:
                raise ValueError('Could not find a good crop.')
        finally:
            os.environ.clear()
            os.environ.update(old_environ)
