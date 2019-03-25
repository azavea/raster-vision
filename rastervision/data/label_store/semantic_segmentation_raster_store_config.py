import os
from copy import deepcopy

import rastervision as rv
from rastervision.data.label_store import (
    LabelStoreConfig, LabelStoreConfigBuilder, SemanticSegmentationRasterStore)
from rastervision.protos.label_store_pb2 import LabelStoreConfig as LabelStoreConfigMsg

VectorOutput = LabelStoreConfigMsg.SemanticSegmentationRasterStore.VectorOutput


class SemanticSegmentationRasterStoreConfig(LabelStoreConfig):
    def __init__(self, uri=None, vector_output=[], rgb=False):
        super().__init__(store_type=rv.SEMANTIC_SEGMENTATION_RASTER)
        self.uri = uri
        self.vector_output = vector_output
        self.rgb = rgb

    def to_proto(self):
        """Turn this configuration into a ProtoBuf message.

        The fields in the message are as follows:
            - `denoise` gives the radius of the structural element
              used to remove high-frequency signals from the image.
            - `uri` is the location where vector output should be
              written
            - `mode` is the vectorification mode (currently only
              "polygons" and "buildings" are acceptable values).
            - `class_id` specifies the predication class that is to
              turned into vectors
            - `building_options` communicates options useful for
              vectorification of building predictions (it is intended
              to break-up clusters of buildings):
                - `min_aspect_ratio` is the ratio between length and
                  height (or height and length) of anything that can
                  be considered to be a cluster of buildings.  The
                  goal is to distinguish between rows of buildings and
                  (say) a single building.
                - `min_area` is the minimum area of anything that can
                  be considered to be a cluster of buildings.  The
                  goal is to distinguish between buildings and
                  artifacts.
                - `element_width_factor` is the width of the
                  structural element used to break building clusters
                  as a fraction of the width of the cluster.
                - `element_thickness` is the thickness of the
                  structural element that is used to break building
                  clusters.
        """
        msg = super().to_proto()
        if self.uri:
            msg.semantic_segmentation_raster_store.uri = self.uri
        if self.vector_output:
            ar = []
            for vo in self.vector_output:
                vo_msg = VectorOutput()
                vo_msg.denoise = vo['denoise'] if 'denoise' in vo.keys() else 0
                vo_msg.uri = vo['uri'] if 'uri' in vo.keys() else ''
                vo_msg.mode = vo['mode']
                vo_msg.class_id = vo['class_id']
                if 'building_options' in vo.keys():
                    options = vo['building_options']
                else:
                    options = {}
                bldg_msg = vo_msg.building_options
                if 'min_aspect_ratio' in options.keys():
                    bldg_msg.min_aspect_ratio = options['min_aspect_ratio']
                if 'min_area' in options.keys() and options['min_area']:
                    bldg_msg.min_area = options['min_area']
                if 'element_width_factor' in options.keys():
                    bldg_msg.element_width_factor = options[
                        'element_width_factor']
                if 'element_thickness' in options.keys():
                    bldg_msg.element_thickness = options['element_thickness']
                ar.append(vo_msg)
            msg.semantic_segmentation_raster_store.vector_output.extend(ar)
        msg.semantic_segmentation_raster_store.rgb = self.rgb
        return msg

    def for_prediction(self, label_uri):
        return self.to_builder() \
                   .with_uri(label_uri) \
                   .build()

    def create_store(self, task_config, extent, crs_transformer, tmp_dir):
        class_map = None
        if self.rgb:
            class_map = task_config.class_map

        return SemanticSegmentationRasterStore(
            self.uri,
            extent,
            crs_transformer,
            tmp_dir,
            vector_output=self.vector_output,
            class_map=class_map)

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        if command_type == rv.PREDICT:
            if not self.uri:
                # Construct the URI for this prediction store,
                # using the scene ID.
                root = experiment_config.predict_uri
                uri = None
                for c in context:
                    if isinstance(c, rv.SceneConfig):
                        uri = os.path.join(root, '{}.tif'.format(c.id))
                if uri:
                    self.uri = uri
                else:
                    raise rv.ConfigError(
                        'SemanticSegmentationRasterStoreConfig has no '
                        'URI set, and is not associated with a SceneConfig.')

            # Construct URIs for vector predictions
            for vo in self.vector_output:
                for c in context:
                    if isinstance(c,
                                  rv.SceneConfig) and ('uri' not in vo.keys()
                                                       or not vo['uri']):
                        root = experiment_config.predict_uri
                        mode = vo['mode']
                        class_id = vo['class_id']
                        vo['uri'] = os.path.join(
                            root, '{}-{}-{}.geojson'.format(
                                c.id, class_id, mode))

    def report_io(self, command_type, io_def):
        if command_type == rv.PREDICT:
            # Construct URIs for vector predictions
            for vo in self.vector_output:
                io_def.add_output(vo['uri'])
            io_def.add_output(self.uri)

        if command_type == rv.EVAL:
            if self.uri:
                io_def.add_input(self.uri)
            else:
                msg = 'No URI set for SemanticSegmentationRasterStoreConfig'
                io_def.add_missing(msg)

            for vo in self.vector_output:
                io_def.add_input(vo['uri'])


class SemanticSegmentationRasterStoreConfigBuilder(LabelStoreConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'uri': prev.uri,
                'vector_output': prev.vector_output,
                'rgb': prev.rgb,
            }

        super().__init__(SemanticSegmentationRasterStoreConfig, config)
        self.valid_modes = set(['buildings', 'polygons'])

    def from_proto(self, msg):
        uri = msg.semantic_segmentation_raster_store.uri
        rgb = msg.semantic_segmentation_raster_store.rgb
        vo_msg = msg.semantic_segmentation_raster_store.vector_output

        return self.with_uri(uri) \
                   .with_vector_output(vo_msg) \
                   .with_rgb(rgb)

    def with_uri(self, uri):
        """Set URI for a GeoTIFF used to read/write predictions."""
        b = deepcopy(self)
        b.config['uri'] = uri
        return b

    def with_vector_output(self, vector_output):
        """Configure vector output for predictions.

            Args:
                vector_output: Either a list of dictionaries or a
                    protobuf object.  The dictionary or the object
                    contain (respectively) keys (attributes) called
                    'denoise', 'uri', 'class_id', and 'mode'.  The
                    value associated with the 'denoise' key specifies
                    the radius of the structural element used to
                    perform a low-pass filtering process on the mask
                    (see
                    https://en.wikipedia.org/wiki/Mathematical_morphology#Opening).
                    The value associated with the 'uri' key is either
                    a file where the GeoJSON prediction will be
                    written, or "" indicating that the filename should
                    be auto-generated.  'class_id' is the integer
                    prediction class that is of interest.  The 'mode'
                    key must be set to 'buildings' or 'polygons'.

        """
        b = deepcopy(self)
        ar = []

        if isinstance(vector_output, list):
            for vo in vector_output:
                ar.append(vo.copy())
        else:
            for vo_msg in vector_output:
                bldg_msg = vo_msg.building_options
                ar.append({
                    'denoise': vo_msg.denoise,
                    'uri': vo_msg.uri,
                    'mode': vo_msg.mode,
                    'class_id': vo_msg.class_id,
                    'building_options': {
                        'min_aspect_ratio':
                        bldg_msg.min_aspect_ratio,
                        'min_area':
                        bldg_msg.min_area if bldg_msg.min_area > 0 else None,
                        'element_width_factor':
                        bldg_msg.element_width_factor,
                        'element_thickness':
                        bldg_msg.element_thickness,
                    },
                })

        b.config['vector_output'] = ar
        return b

    def with_rgb(self, rgb):
        """Set flag for writing RGB data using the class map.

        Otherwise this method will write the class ID into a single band.
        """
        b = deepcopy(self)
        b.config['rgb'] = rgb
        return b

    def validate(self):
        vector_output = self.config.get('vector_output')

        if vector_output and not isinstance(vector_output, list):
            for vo in vector_output:
                if not hasattr(vo, 'mode'):
                    raise rv.ConfigError(
                        'The attribute vector_output of'
                        ' SemanticSegmentationRasterStoreConfig'
                        ' must be either trivial, a protobuf configuration'
                        ' object, or a list of'
                        ' appropriate dictionaries.')
                if vo.mode not in self.valid_modes:
                    raise rv.ConfigError(
                        'mode key in vector_output dictionary must be one of {}'
                        .format(self.valid_modes))
        elif vector_output and isinstance(vector_output, list):
            for vo in vector_output:
                if not isinstance(vo, dict):
                    raise rv.ConfigError(
                        'The attribute vector_output of'
                        ' SemanticSegmentationRasterStoreConfig'
                        ' must be either trivial, a protobuf configuration'
                        ' object, or a list of'
                        ' appropriate dictionaries.')
                if 'mode' not in vo.keys(
                ) or vo['mode'] not in self.valid_modes:
                    raise rv.ConfigError(
                        'mode key in vector_output dictionary must be one of {}'
                        .format(self.valid_modes))
