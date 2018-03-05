from rv2.core.project import Project
from rv2.builders import annotation_source_builder, raster_source_builder


def build(config):
    raster_source = None
    crs_transformer = None
    ground_truth_annotation_source = None
    prediction_annotation_source = None

    raster_source = raster_source_builder.build(config.raster_source)
    crs_transformer = raster_source.get_crs_transformer()

    if config.HasField('ground_truth_annotation_source'):
        ground_truth_annotation_source = annotation_source_builder.build(
            config.ground_truth_annotation_source, crs_transformer,
            writable=False)

    if config.HasField('prediction_annotation_source'):
        prediction_annotation_source = annotation_source_builder.build(
            config.prediction_annotation_source, crs_transformer,
            writable=True)

    return Project(
        raster_source=raster_source,
        ground_truth_annotation_source=ground_truth_annotation_source,
        prediction_annotation_source=prediction_annotation_source)
