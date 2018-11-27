import os
from copy import deepcopy

import rastervision as rv
from rastervision.data.label_store import (
    LabelStoreConfig, LabelStoreConfigBuilder, SemanticSegmentationRasterStore)
from rastervision.protos.label_store_pb2 import LabelStoreConfig as LabelStoreConfigMsg


class SemanticSegmentationRasterStoreConfig(LabelStoreConfig):
    def __init__(self, uri=None, vector_output=[], rgb=False):
        super().__init__(store_type=rv.SEMANTIC_SEGMENTATION_RASTER)
        self.uri = uri
        self.vector_output = vector_output
        self.rgb = rgb

    def to_proto(self):
        msg = super().to_proto()
        if self.uri:
            msg.semantic_segmentation_raster_store.uri = self.uri
        if self.vector_output:
            ar = []
            for vo in self.vector_output:
                msg2 = LabelStoreConfigMsg.SemanticSegmentationRasterStore.VectorOutput(
                )
                msg2.uri = vo['uri']
                msg2.mode = vo['mode']
                ar.append(msg2)
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
        affine_transform = task_config.affine_transform

        return SemanticSegmentationRasterStore(
            self.uri,
            self.vector_output,
            extent,
            affine_transform,
            crs_transformer,
            tmp_dir,
            class_map=class_map)

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        io_def = io_def or rv.core.CommandIODefinition()

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
                    io_def.add_output(uri)
                else:
                    raise rv.ConfigError(
                        'SemanticSegmentationRasterStoreConfig has no '
                        'URI set, and is not associated with a SceneConfig.')

            # Construct URIs for vector predictions
            for vo in self.vector_output:
                for c in context:
                    if isinstance(c, rv.SceneConfig) and vo['uri'] == '*':
                        root = experiment_config.predict_uri
                        vo['uri'] = os.path.join(
                            root, '{}-{}.geojson'.format(c.id, vo['mode']))
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

        return io_def


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

    def from_proto(self, msg):
        uri = msg.semantic_segmentation_raster_store.uri
        rgb = msg.semantic_segmentation_raster_store.rgb
        vo = msg.semantic_segmentation_raster_store.vector_output

        return self.with_uri(uri) \
                   .with_vector_output(vo) \
                   .with_rgb(rgb)

    def with_uri(self, uri):
        """Set URI for a GeoTIFF used to read/write predictions."""
        b = deepcopy(self)
        b.config['uri'] = uri
        return b

    def with_vector_output(self, msg):
        """Vector output for predictions."""
        b = deepcopy(self)
        ar = []

        if isinstance(msg, list):
            for vo in msg:
                ar.append(vo.copy())
        else:
            for vo in msg:
                ar.append({'uri': vo.uri, 'mode': vo.mode})

        b.config['vector_output'] = ar
        return b

    def with_rgb(self, rgb):
        """Set flag for writing RGB data using the class map.

        Otherwise this method will write the class ID into a single band.
        """
        b = deepcopy(self)
        b.config['rgb'] = rgb
        return b
