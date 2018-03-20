from rv2.core.project import Project
from rv2.builders import label_source_builder, raster_source_builder


def build(config):
    raster_source = None
    crs_transformer = None
    ground_truth_label_source = None
    prediction_label_source = None

    raster_source = raster_source_builder.build(config.raster_source)
    crs_transformer = raster_source.get_crs_transformer()

    if config.HasField('ground_truth_label_source'):
        ground_truth_label_source = label_source_builder.build(
            config.ground_truth_label_source, crs_transformer,
            writable=False)

    if config.HasField('prediction_label_source'):
        prediction_label_source = label_source_builder.build(
            config.prediction_label_source, crs_transformer,
            writable=True)

    return Project(
        raster_source=raster_source,
        ground_truth_label_source=ground_truth_label_source,
        prediction_label_source=prediction_label_source)
