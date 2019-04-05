from abc import abstractmethod
from copy import deepcopy
import numbers

from google.protobuf import (json_format)

import rastervision as rv
from rastervision.core.config import Config, ConfigBuilder
from rastervision.protos.vector_source_pb2 import (VectorSourceConfig as
                                                   VectorSourceConfigMsg)


class VectorSourceConfig(Config):
    def __init__(self,
                 source_type,
                 class_id_to_filter=None,
                 default_class_id=1,
                 line_bufs=None,
                 point_bufs=None):
        self.source_type = source_type
        self.class_id_to_filter = class_id_to_filter
        self.default_class_id = default_class_id
        self.line_bufs = line_bufs
        self.point_bufs = point_bufs

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

    def to_proto(self):
        msg = VectorSourceConfigMsg(source_type=self.source_type)

        if self.class_id_to_filter is not None:
            # Convert class_ids to str to put into json format.
            class_id_to_filter = dict(
                [(str(class_id), filter)
                 for class_id, filter in self.class_id_to_filter.items()])
            d = {'class_id_to_filter': class_id_to_filter}
            msg.MergeFrom(json_format.ParseDict(d, VectorSourceConfigMsg()))

        if self.default_class_id is not None:
            msg.default_class_id = self.default_class_id

        if self.line_bufs is not None:
            # Convert class_ids to str to put into json format.
            line_bufs = dict([(str(c), v) for c, v in self.line_bufs.items()])
            d = {'line_bufs': line_bufs}
            msg.MergeFrom(json_format.ParseDict(d, VectorSourceConfigMsg()))

        if self.point_bufs is not None:
            # Convert class_ids to str to put into json format.
            point_bufs = dict(
                [(str(c), v) for c, v in self.point_bufs.items()])
            d = {'point_bufs': point_bufs}
            msg.MergeFrom(json_format.ParseDict(d, VectorSourceConfigMsg()))

        return msg

    @staticmethod
    def builder(source_type):
        return rv._registry.get_config_builder(rv.VECTOR_SOURCE, source_type)()

    def to_builder(self):
        return rv._registry.get_config_builder(rv.VECTOR_SOURCE,
                                               self.source_type)(self)

    @staticmethod
    def from_proto(msg):
        """Creates a from the specificed protobuf message.
        """
        return rv._registry.get_config_builder(rv.VECTOR_SOURCE, msg.source_type)() \
                           .from_proto(msg) \
                           .build()

    @abstractmethod
    def create_source(self, crs_transformer=None, extent=None, class_map=None):
        pass


class VectorSourceConfigBuilder(ConfigBuilder):
    def from_proto(self, msg):
        b = self
        d = json_format.MessageToDict(msg)

        class_id_to_filter = None
        if msg.HasField('class_id_to_filter'):
            # Convert class_ids from strs to ints.
            # Have to use camel case after parsing from json :(
            class_id_to_filter = dict(
                [(int(class_id), filter)
                 for class_id, filter in d['classIdToFilter'].items()])

        default_class_id = None
        if msg.HasField('default_class_id'):
            default_class_id = msg.default_class_id

        b = b.with_class_inference(
            class_id_to_filter=class_id_to_filter,
            default_class_id=default_class_id)

        line_bufs = msg.line_bufs if msg.HasField('line_bufs') else None
        point_bufs = msg.point_bufs if msg.HasField('point_bufs') else None
        b = b.with_buffers(line_bufs, point_bufs)

        return b

    def validate(self):
        def _validate(bufs):
            if bufs is not None:
                for c, v in bufs.items():
                    if not (v is None or isinstance(v, numbers.Number)):
                        raise rv.ConfigError(
                            'Buffer size {} must be a number or None.'.format(
                                v))

        _validate(self.config.get('point_bufs'))
        _validate(self.config.get('line_bufs'))
        super().validate()

    def with_class_inference(self, class_id_to_filter=None,
                             default_class_id=1):
        """Set options for inferring the class of each feature.

        For more info on how class inference works, see ClassInference.infer_class()

        Args:
            class_id_to_filter: (dict) map from class_id to JSON filter.
                The filter schema is according to
                https://github.com/mapbox/mapbox-gl-js/blob/c9900db279db776f493ce8b6749966cedc2d6b8a/src/style-spec/feature_filter/index.js  # noqa
            default_class_id: (int) the default class_id to use if class can't be
                inferred using other mechanisms. If a feature defaults to a class_id of
                None, then that feature will be deleted.
        """
        b = deepcopy(self)
        # Ensure class_ids are ints.
        if class_id_to_filter is not None:
            class_id_to_filter = dict(
                [(int(c), f) for c, f in class_id_to_filter.items()])
        b.config['class_id_to_filter'] = class_id_to_filter
        b.config['default_class_id'] = default_class_id
        return b

    def with_buffers(self, line_bufs=None, point_bufs=None):
        """Set options for buffering lines and points into polygons.

        For example, this is useful for buffering lines representing roads so that
        their width roughly matches the width of roads in the imagery.

        Args:
            line_bufs: (dict or None) If none, uses default buffer value of 1.
                Otherwise, a map from class_id to number of pixels to buffer by.
                If the buffer value is None, then no buffering will be performed
                and the LineString or Point won't get converted to a Polygon.
                Not converting to Polygon is incompatible with the currently
                available LabelSources, but may be useful in the future.
            point_bufs: (dict or None) same as above, but used for buffering
                Points into Polygons.
        """
        b = deepcopy(self)

        if line_bufs is not None:
            line_bufs = dict([(int(c), v) for c, v in line_bufs.items()])
        b.config['line_bufs'] = line_bufs

        if point_bufs is not None:
            point_bufs = dict([(int(c), v) for c, v in point_bufs.items()])
        b.config['point_bufs'] = point_bufs

        return b
