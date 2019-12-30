from os.path import isfile
from urllib.parse import urlparse
import subprocess
import sys

import boto3
from pystac import STAC_IO
from pystac import Catalog, LabelItem
from shapely.geometry import box

from rastervision.utils.misc import (terminate_at_exit)
from rastervision.utils.files import get_local_path, download_if_needed


def run_command(cmd, shell=False):
    process = subprocess.Popen(cmd, shell=shell)
    terminate_at_exit(process)
    exitcode = process.wait()
    if exitcode != 0:
        sys.exit(exitcode)


def setup_stac_s3():
    def my_read_method(uri):
        parsed = urlparse(uri)
        if parsed.scheme == 's3':
            bucket = parsed.netloc
            key = parsed.path[1:]
            s3 = boto3.resource('s3')
            obj = s3.Object(bucket, key)
            return obj.get()['Body'].read().decode('utf-8')
        else:
            return STAC_IO.default_read_text_method(uri)

    def my_write_method(uri, txt):
        parsed = urlparse(uri)
        if parsed.scheme == 's3':
            bucket = parsed.netloc
            key = parsed.path[1:]
            s3 = boto3.resource('s3')
            s3.Object(bucket, key).put(Body=txt)
        else:
            STAC_IO.default_write_text_method(uri, txt)

    STAC_IO.read_text_method = my_read_method
    STAC_IO.write_text_method = my_write_method


def parse_stac(stac_uri):
    setup_stac_s3()
    cat = Catalog.from_file(stac_uri)
    cat.make_all_asset_hrefs_absolute()
    labels_uri = None
    geotiff_uris = []
    for item in cat.get_all_items():
        if isinstance(item, LabelItem):
            labels_uri = list(item.assets.values())[0].href
            labels_box = box(*item.bbox)

    # only use geotiffs that intersect with bbox of labels
    for item in cat.get_all_items():
        if not isinstance(item, LabelItem):
            geotiff_uri = list(item.assets.values())[0].href
            geotiff_box = box(*item.bbox)
            if labels_box.intersects(geotiff_box):
                geotiff_uri = geotiff_uri.replace('%7C', '|')
                geotiff_uris.append(geotiff_uri)

    if not labels_uri:
        raise ValueError('Unable to read labels URI from STAC.')
    if not geotiff_uris:
        raise ValueError('Unable to read GeoTIFF URIs from STAC.')
    return labels_uri, labels_box, geotiff_uris


def cached_download(uris, data_dir):
    paths = []
    for uri in uris:
        path = get_local_path(uri, data_dir)
        paths.append(path)
        if not isfile(path):
            download_if_needed(uri, data_dir)
    return paths
