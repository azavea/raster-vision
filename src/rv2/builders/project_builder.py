from rv2.core.project import Project
from rv2.builders import annotation_source_builder, raster_source_builder


def build(config, write_mode=False):
    raster_source = raster_source_builder.build(config.raster_source)
    crs_transformer = raster_source.get_crs_transformer()
    annotation_source = annotation_source_builder.build(
        config.annotation_source, crs_transformer, write_mode=write_mode)
    return Project(
        raster_source=raster_source, annotation_source=annotation_source)
