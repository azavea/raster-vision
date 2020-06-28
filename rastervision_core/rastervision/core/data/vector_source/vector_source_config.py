from typing import Dict, Optional, Union

from rastervision.pipeline.config import Config, register_config, Field


@register_config('vector_source')
class VectorSourceConfig(Config):
    default_class_id: Optional[int] = Field(
        ...,
        description=
        ('The default class_id to use if class cannot be inferred using other '
         'mechanisms. If a feature has an inferred class_id of None, then it '
         'will be deleted.'))
    class_id_to_filter: Optional[Dict] = Field(
        None,
        description=
        ('Map from class_id to JSON filter used to infer missing class_ids. '
         'Each key should be a class id, and its value should be a boolean '
         'expression which is run against the property field for each feature.'
         'This allows matching different features to different class ids based on '
         'its properties. The expression schema is that described by '
         'https://docs.mapbox.com/mapbox-gl-js/style-spec/other/#other-filter'
         ))
    line_bufs: Optional[Dict[int, Union[int, float, None]]] = Field(
        None,
        description=
        ('This is useful, for example, for buffering lines representing roads so that '
         'their width roughly matches the width of roads in the imagery. If None, uses '
         'default buffer value of 1. Otherwise, a map from class_id to '
         'number of pixels to buffer by. If the buffer value is None, then no buffering '
         'will be performed and the LineString or Point won\'t get converted to a '
         'Polygon. Not converting to Polygon is incompatible with the currently '
         'available LabelSources, but may be useful in the future.'))
    point_bufs: Optional[Dict[int, Union[int, float, None]]] = Field(
        None,
        description=
        'Same as above, but used for buffering Points into Polygons.')

    def has_null_class_bufs(self):
        if self.point_bufs is not None:
            for c, v in self.point_bufs.items():
                if v is None:
                    return True

        if self.line_bufs is not None:
            for c, v in self.line_bufs.items():
                if v is None:
                    return True

        return False

    def build(self, class_config, crs_transformer):
        raise NotImplementedError()

    def update(self, pipeline=None, scene=None):
        pass
