import csv
from io import StringIO
import os

import rasterio
from shapely.strtree import STRtree
from shapely.geometry import shape, mapping
from shapely.ops import transform

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core import Box
from rastervision.core.data import (RasterioCRSTransformer,
                                    GeoJSONVectorSourceConfig,
                                    ClassInferenceTransformerConfig)
from rastervision.pipeline.file_system import (file_to_str, file_exists,
                                               get_local_path, upload_or_copy,
                                               make_dir, json_to_file)
from rastervision.aws_s3 import S3FileSystem


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


def crop_image(image_uri: str, window: Box, crop_uri: str):
    im_dataset = rasterio.open(image_uri)
    rasterio_window = window.rasterio_format()
    im = im_dataset.read(window=rasterio_window)

    with get_tmp_dir() as tmp_dir:
        crop_path = get_local_path(crop_uri, tmp_dir)
        make_dir(crop_path, use_dirname=True)

        meta = im_dataset.meta
        meta['height'], meta['width'] = window.size
        meta['transform'] = rasterio.windows.transform(rasterio_window,
                                                       im_dataset.transform)

        with rasterio.open(crop_path, 'w', **meta) as dst:
            dst.colorinterp = im_dataset.colorinterp
            dst.write(im)

        upload_or_copy(crop_path, crop_uri)


def save_image_crop(image_uri,
                    image_crop_uri,
                    label_uri=None,
                    label_crop_uri=None,
                    size=600,
                    min_features=10,
                    vector_labels=True,
                    class_config=None):
    """Save a crop of an image to use for testing.

    If label_uri is set, the crop needs to cover >= min_features.

    Args:
        image_uri: URI of original image
        image_crop_uri: URI of cropped image to save
        label_uri: optional URI of label file
        label_crop_uri: optional URI of cropped labels to save
        size: height and width of crop

    Raises:
        ValueError if cannot find a crop satisfying min_features constraint.
    """
    if not file_exists(image_crop_uri):
        print('Saving test crop to {}...'.format(image_crop_uri))
        old_environ = os.environ.copy()
        try:
            request_payer = S3FileSystem.get_request_payer()
            if request_payer == 'requester':
                os.environ['AWS_REQUEST_PAYER'] = request_payer
            im_dataset = rasterio.open(image_uri)
            h, w = im_dataset.height, im_dataset.width

            extent = Box(0, 0, h, w)
            windows = extent.get_windows(size, size)
            if label_uri and vector_labels:
                crs_transformer = RasterioCRSTransformer.from_dataset(
                    im_dataset)
                geojson_vs_config = GeoJSONVectorSourceConfig(
                    uri=label_uri,
                    ignore_crs_field=True,
                    transformers=[
                        ClassInferenceTransformerConfig(default_class_id=0)
                    ])
                vs = geojson_vs_config.build(class_config, crs_transformer)
                geojson = vs.get_geojson()
                geoms = []
                for f in geojson['features']:
                    g = shape(f['geometry'])
                    geoms.append(g)
                tree = STRtree(geoms)

            def p2m(x, y, z=None):
                return crs_transformer.pixel_to_map((x, y))

            for w in windows:
                use_window = True
                if label_uri and vector_labels:
                    w_polys = tree.query(w.to_shapely())
                    use_window = len(w_polys) >= min_features
                    if use_window and label_crop_uri is not None:
                        print('Saving test crop labels to {}...'.format(
                            label_crop_uri))

                        label_crop_features = [
                            mapping(transform(p2m, wp)) for wp in w_polys
                        ]
                        label_crop_json = {
                            'type':
                            'FeatureCollection',
                            'features': [{
                                'geometry': f
                            } for f in label_crop_features]
                        }
                        json_to_file(label_crop_json, label_crop_uri)

                if use_window:
                    crop_image(image_uri, w, image_crop_uri)

                    if not vector_labels and label_uri and label_crop_uri:
                        crop_image(label_uri, w, label_crop_uri)

                    break

            if not use_window:
                raise ValueError('Could not find a good crop.')
        finally:
            os.environ.clear()
            os.environ.update(old_environ)
