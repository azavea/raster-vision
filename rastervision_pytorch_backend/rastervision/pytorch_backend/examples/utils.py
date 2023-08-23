from typing import TYPE_CHECKING, Optional
import os
import csv
from io import StringIO

from rastervision.core.data import (RasterioSource, GeoJSONVectorSource,
                                    ClassInferenceTransformer)
from rastervision.core.data.utils import geoms_to_geojson, crop_geotiff
from rastervision.pipeline.file_system import (file_to_str, json_to_file)
from rastervision.aws_s3 import S3FileSystem

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig


def get_scene_info(csv_uri: str) -> list:  # pragma: no cover
    csv_str = file_to_str(csv_uri)
    reader = csv.reader(StringIO(csv_str))
    return list(reader)


def save_image_crop(
        image_uri: str,
        image_crop_uri: str,
        label_uri: Optional[str] = None,
        label_crop_uri: Optional[str] = None,
        size: int = 600,
        min_features: int = 10,
        vector_labels: bool = True,
        default_class_id: int = 0,
        class_config: Optional['ClassConfig'] = None):  # pragma: no cover
    """Save a crop of an image to use for testing.

    If label_uri is set, the crop needs to cover >= min_features.

    Args:
        image_uri: URI of original image
        image_crop_uri: URI of cropped image to save
        label_uri: optional URI of label file
        label_crop_uri: optional URI of cropped labels to save
        size: height and width of crop
        min_features: min number of label polygons that the crop should have
        vector_labels: whether the labels are vector labels
        default_class_id: default class ID to use to infer labels
        class_config: ClassConfig to use to infer labels

    Raises:
        ValueError if cannot find a crop satisfying min_features constraint.
    """
    print(f'Saving test crop to {image_crop_uri}...')
    old_environ = os.environ.copy()
    try:
        request_payer = S3FileSystem.get_request_payer()
        if request_payer == 'requester':
            os.environ['AWS_REQUEST_PAYER'] = request_payer
        rs = RasterioSource(image_uri, allow_streaming=True)
        if label_uri and vector_labels:
            crs_tf = rs.crs_transformer
            vs = GeoJSONVectorSource(
                uris=label_uri,
                crs_transformer=crs_tf,
                ignore_crs_field=True,
                vector_transformers=[
                    ClassInferenceTransformer(
                        default_class_id=default_class_id,
                        class_config=class_config)
                ])
            labels_df = vs.get_dataframe()

        windows = rs.extent.get_windows(size, size)
        for w in windows:
            use_window = True
            if label_uri and vector_labels:
                w_geom = w.to_shapely()
                df_int = labels_df[labels_df.intersects(w_geom)]
                w_polys = df_int.geometry
                use_window = len(w_polys) >= min_features
                if use_window and label_crop_uri is not None:
                    print(f'Saving test crop labels to {label_crop_uri}...')
                    w_polys_map = [crs_tf.pixel_to_map(wp) for wp in w_polys]
                    label_crop_json = geoms_to_geojson(w_polys_map)
                    json_to_file(label_crop_json, label_crop_uri)
            if use_window:
                crop_geotiff(image_uri, w, image_crop_uri)
                if not vector_labels and label_uri and label_crop_uri:
                    crop_geotiff(label_uri, w, label_crop_uri)
                break
        if not use_window:
            raise ValueError('Could not find a good crop.')
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
