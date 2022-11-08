from typing import TYPE_CHECKING, Optional, Sequence, Union

from rastervision.core.data.utils.geojson import geoms_to_geojson

if TYPE_CHECKING:
    from rastervision.core.box import Box
    from rastervision.core.data import ClassConfig, CRSTransformer


def boxes_to_geojson(
        boxes: Sequence['Box'],
        class_ids: Sequence[int],
        crs_transformer: 'CRSTransformer',
        class_config: 'ClassConfig',
        scores: Optional[Sequence[Union[float, Sequence[float]]]] = None
) -> dict:
    """Convert boxes and associated data into a GeoJSON dict.

    Args:
        boxes (Sequence[Box]): List of Box in pixel row/col format.
        class_ids (Sequence[int]): List of int (one for each box)
        crs_transformer (CRSTransformer): CRSTransformer used to convert pixel
            coords to map coords in the GeoJSON.
        class_config (ClassConfig): ClassConfig
        scores (Optional[Sequence[Union[float, Sequence[float]]]], optional):
            Optional list of score or scores. If floats (one for each box),
            property name will be "score". If lists of floats, property name
            will be "scores". Defaults to None.

    Returns:
        dict: Serialized GeoJSON.
    """
    if len(boxes) != len(class_ids):
        raise ValueError(f'len(boxes) ({len(boxes)}) != '
                         f'len(class_ids) ({len(class_ids)})')
    if scores is not None and len(boxes) != len(scores):
        raise ValueError(f'len(boxes) ({len(boxes)}) != '
                         f'len(scores) ({len(scores)})')

    geoms = [crs_transformer.pixel_to_map(box.to_shapely()) for box in boxes]
    properties = [
        dict(class_id=id, class_name=class_config.get_name(id))
        for id in class_ids
    ]
    if scores is not None:
        for prop, box_score in zip(properties, scores):
            key = 'score' if isinstance(box_score, float) else 'scores'
            prop[key] = box_score

    geojson = geoms_to_geojson(geoms, properties)
    return geojson
