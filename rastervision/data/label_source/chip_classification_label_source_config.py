from copy import deepcopy

import rastervision as rv
from rastervision.data.label_source import (
    LabelSourceConfig, LabelSourceConfigBuilder, ChipClassificationLabelSource)
from rastervision.data.vector_source import VectorSourceConfig
from rastervision.protos.label_source_pb2 import LabelSourceConfig as LabelSourceConfigMsg


class ChipClassificationLabelSourceConfig(LabelSourceConfig):
    def __init__(self,
                 vector_source,
                 ioa_thresh=None,
                 use_intersection_over_cell=False,
                 pick_min_class_id=False,
                 background_class_id=None,
                 cell_size=None,
                 infer_cells=False):
        super().__init__(source_type=rv.CHIP_CLASSIFICATION)
        self.vector_source = vector_source
        self.ioa_thresh = ioa_thresh
        self.use_intersection_over_cell = use_intersection_over_cell
        self.pick_min_class_id = pick_min_class_id
        self.background_class_id = background_class_id
        self.cell_size = cell_size
        self.infer_cells = infer_cells

    def to_proto(self):
        msg = super().to_proto()
        options = LabelSourceConfigMsg.ChipClassificationLabelSource(
            vector_source=self.vector_source.to_proto(),
            ioa_thresh=self.ioa_thresh,
            use_intersection_over_cell=self.use_intersection_over_cell,
            pick_min_class_id=self.pick_min_class_id,
            background_class_id=self.background_class_id,
            cell_size=self.cell_size,
            infer_cells=self.infer_cells)
        msg.chip_classification_label_source.CopyFrom(options)
        return msg

    def create_source(self, task_config, extent, crs_transformer, tmp_dir):
        vector_source = self.vector_source.create_source(
            crs_transformer=crs_transformer,
            extent=extent,
            class_map=task_config.class_map)
        return ChipClassificationLabelSource(
            vector_source, crs_transformer, task_config.class_map, extent,
            self.ioa_thresh, self.use_intersection_over_cell,
            self.pick_min_class_id, self.background_class_id, self.cell_size,
            self.infer_cells)

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        io_def = io_def or rv.core.CommandIODefinition()
        self.vector_source.update_for_command(command_type, experiment_config,
                                              context, io_def)

        if not self.cell_size:
            self.cell_size = experiment_config.task.chip_size

        return io_def


class ChipClassificationLabelSourceConfigBuilder(LabelSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'vector_source': prev.vector_source,
                'ioa_thresh': prev.ioa_thresh,
                'use_intersection_over_cell': prev.use_intersection_over_cell,
                'pick_min_class_id': prev.pick_min_class_id,
                'background_class_id': prev.background_class_id,
                'cell_size': prev.cell_size,
                'infer_cells': prev.infer_cells
            }

        super().__init__(ChipClassificationLabelSourceConfig, config)

    def validate(self):
        if self.config.get('vector_source') is None:
            raise rv.ConfigError(
                'You must set the vector_source for ChipClassificationLabelSourceConfig'
                ' Use "with_vector_source".')

    def from_proto(self, msg):
        b = ChipClassificationLabelSourceConfigBuilder()
        conf = msg.chip_classification_label_source

        return b \
            .with_vector_source(rv.VectorSourceConfig.from_proto(conf.vector_source)) \
            .with_ioa_thresh(conf.ioa_thresh) \
            .with_use_intersection_over_cell(conf.use_intersection_over_cell) \
            .with_pick_min_class_id(conf.pick_min_class_id) \
            .with_background_class_id(conf.background_class_id) \
            .with_cell_size(conf.cell_size) \
            .with_infer_cells(conf.infer_cells)

    def with_vector_source(self, vector_source):
        """Set the vector_source.

        Args:
            vector_source (str or VectorSource) if a string, assume it is
                a URI and use the default provider to construct a VectorSource.
        """
        if isinstance(vector_source, str):
            return self.with_uri(vector_source)

        b = deepcopy(self)
        if isinstance(vector_source, VectorSourceConfig):
            b.config['vector_source'] = vector_source
        else:
            raise rv.ConfigError(
                'vector_source must be of type str or VectorSource')

        return b

    def with_uri(self, uri):
        b = deepcopy(self)
        provider = rv._registry.get_vector_source_default_provider(uri)
        b.config['vector_source'] = provider.construct(uri)
        return b

    def with_ioa_thresh(self, ioa_thresh):
        """The minimum IOA of a polygon and cell."""
        b = deepcopy(self)
        b.config['ioa_thresh'] = ioa_thresh
        return b

    def with_use_intersection_over_cell(self, use_intersection_over_cell):
        """ Set this label source to use intersection over cell or not.

        If use_intersection_over_cell is true, then use the area of the
        cell as the denominator in the IOA. Otherwise, use the area of the
        polygon.
        """
        b = deepcopy(self)
        b.config['use_intersection_over_cell'] = use_intersection_over_cell
        return b

    def with_pick_min_class_id(self, pick_min_class_id):
        """Set this label source to pick min class ID

        If true, the class_id for a cell is the minimum class_id of the
        boxes in that cell. Otherwise, pick the class_id of the box
        covering the greatest area.
        """
        b = deepcopy(self)
        b.config['pick_min_class_id'] = pick_min_class_id
        return b

    def with_background_class_id(self, background_class_id):
        """Sets the background class ID.

        Optional class_id to use as the background class; ie. the one that
        is used when a window contains no boxes. If not set, empty windows
        have None set as their class_id.
        """
        b = deepcopy(self)
        b.config['background_class_id'] = background_class_id
        return b

    def with_infer_cells(self, infer_cells):
        """Set if this label source should infer cells.

        If true, the label source will infer the cell polygon and label
        from the polygons in the vector_source. If the labels are already
        cells and properly labeled, this can be False.
        """
        b = deepcopy(self)
        b.config['infer_cells'] = infer_cells
        return b

    def with_cell_size(self, cell_size):
        """Sets the cell size of the chips."""
        b = deepcopy(self)
        b.config['cell_size'] = cell_size
        return b
