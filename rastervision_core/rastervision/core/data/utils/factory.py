from typing import TYPE_CHECKING, List, Optional, Union
from uuid import uuid4

from rastervision.core.data.utils import listify_uris, get_polygons_from_uris

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig, Scene


def make_ss_scene(image_uri: Union[str, List[str]],
                  label_raster_uri: Optional[Union[str, List[str]]] = None,
                  class_config: Optional['ClassConfig'] = None,
                  label_vector_uri: Optional[str] = None,
                  aoi_uri: Union[str, List[str]] = [],
                  label_vector_default_class_id: Optional[int] = None,
                  image_raster_source_kw: dict = {},
                  label_raster_source_kw: dict = {},
                  label_vector_source_kw: dict = {},
                  scene_id: Optional[str] = None) -> 'Scene':
    """Create a semantic segmentation scene from image and label URIs.

    This is a convenience method. For more fine-grained control, it is
    recommended to use the default constructor.

    Args:
        image_uri (Union[str, List[str]]): URI or list of URIs of GeoTIFFs to
            use as the source of image data.
        label_raster_uri (Optional[Union[str, List[str]]], optional): URI or
            list of URIs of GeoTIFFs to use as the source of segmentation label
            data. If the labels are in the form of GeoJSONs, use
            label_vector_uri instead. Defaults to None.
        label_vector_uri (Optional[str], optional):  URI of GeoJSON file to use
            as the source of segmentation label data. If the labels are in the
            form of GeoTIFFs, use label_raster_uri instead. Defaults to None.
        class_config (Optional[ClassConfig]): The ClassConfig. Must be
            non-None if creating a scene without a LabelSource.
            Defaults to None.
        aoi_uri (Union[str, List[str]], optional): URI or list of URIs of
            GeoJSONs that specify the area-of-interest. If provided, the
            dataset will only access data from this area. Defaults to [].
        label_vector_default_class_id (Optional[int], optional): If using
            label_vector_uri and all polygons in that file belong to the same
            class and they do not contain a `class_id` property, then use this
            argument to map all of the polgons to the appropriate class ID.
            See docs for ClassInferenceTransformer for more details.
            Defaults to None.
        image_raster_source_kw (dict, optional): Additional arguments to pass
            to the RasterioSource used for image data. See docs for
            RasterioSource for more details. Defaults to {}.
        label_raster_source_kw (dict, optional): Additional arguments to pass
            to the RasterioSource used for label data, if label_raster_uri is
            used. See docs for RasterioSource for more details. Defaults to {}.
        label_vector_source_kw (dict, optional): Additional arguments to pass
            to the GeoJSONVectorSource used for label data, if label_vector_uri
            is used. See docs for GeoJSONVectorSource for more details.
            Defaults to {}.
        scene_id (Optional[str]): Optional scene ID. If None, will be randomly
            generated. Defaults to None.

    Raises:
        ValueError: If both label_raster_uri and label_vector_uri are
            specified.

    Returns:
        Scene: A semantic segmentation scene.
    """
    # use local imports to avoid circular import problems
    from rastervision.core.data import (
        GeoJSONVectorSource, RasterioSource, RasterizedSource, Scene,
        SemanticSegmentationLabelSource, ClassInferenceTransformer)

    if label_raster_uri is not None and label_vector_uri is not None:
        raise ValueError('Specify either label_raster_uri or '
                         'label_vector_uri or neither, but not both.')

    if label_raster_uri is not None or label_vector_uri is not None:
        if class_config is None:
            raise ValueError('class_config is required if using labels.')
        class_config.ensure_null_class()

    image_uri = listify_uris(image_uri)
    raster_source = RasterioSource(uris=image_uri, **image_raster_source_kw)

    crs_transformer = raster_source.crs_transformer
    extent = raster_source.extent

    label_raster_source = None
    if label_raster_uri is not None:
        label_raster_uri = listify_uris(label_raster_uri)
        label_raster_source = RasterioSource(
            uris=label_raster_uri, **label_raster_source_kw)
    elif label_vector_uri is not None:
        if label_vector_default_class_id is not None:
            # add a ClassInferenceTransformer to the VectorSource
            class_inf_tf = ClassInferenceTransformer(
                default_class_id=label_vector_default_class_id)
            vector_tfs = label_vector_source_kw.get('vector_transformers', [])
            label_vector_source_kw['vector_transformers'] = (
                [class_inf_tf] + vector_tfs)
        vector_source = GeoJSONVectorSource(
            uri=label_vector_uri,
            ignore_crs_field=True,
            crs_transformer=crs_transformer,
            **label_vector_source_kw)
        label_raster_source = RasterizedSource(
            vector_source=vector_source,
            background_class_id=label_raster_source_kw.pop(
                'background_class_id', class_config.null_class_id),
            extent=extent,
            **label_raster_source_kw)

    label_source = None
    if label_raster_source is not None:
        label_source = SemanticSegmentationLabelSource(
            raster_source=label_raster_source, class_config=class_config)

    aoi_polygons = get_polygons_from_uris(aoi_uri, crs_transformer)
    scene = Scene(
        id=uuid4() if scene_id is None else scene_id,
        raster_source=raster_source,
        label_source=label_source,
        aoi_polygons=aoi_polygons)

    return scene


def make_cc_scene(image_uri: Union[str, List[str]],
                  label_vector_uri: Optional[str] = None,
                  class_config: Optional['ClassConfig'] = None,
                  aoi_uri: Union[str, List[str]] = [],
                  label_vector_default_class_id: Optional[int] = None,
                  image_raster_source_kw: dict = {},
                  label_vector_source_kw: dict = {},
                  label_source_kw: dict = {},
                  scene_id: Optional[str] = None) -> 'Scene':
    """Create a chip classification scene from image and label URIs.

    This is a convenience method. For more fine-grained control, it is
    recommended to use the default constructor.

    Args:
        image_uri (Union[str, List[str]]): URI or list of URIs of GeoTIFFs to
            use as the source of image data.
        label_vector_uri (Optional[str], optional):  URI of GeoJSON file to use
            as the source of segmentation label data. Defaults to None.
        class_config (Optional[ClassConfig]): The ClassConfig. Must be
            non-None if creating a scene without a LabelSource.
            Defaults to None.
        aoi_uri (Union[str, List[str]], optional): URI or list of URIs of
            GeoJSONs that specify the area-of-interest. If provided, the
            dataset will only access data from this area. Defaults to [].
        label_vector_default_class_id (Optional[int], optional): If using
            label_vector_uri and all polygons in that file belong to the same
            class and they do not contain a `class_id` property, then use this
            argument to map all of the polgons to the appropriate class ID.
            See docs for ClassInferenceTransformer for more details.
            Defaults to None.
        image_raster_source_kw (dict, optional): Additional arguments to pass
            to the RasterioSource used for image data. See docs for
            RasterioSource for more details. Defaults to {}.
        label_vector_source_kw (dict, optional): Additional arguments to pass
            to the GeoJSONVectorSourceConfig used for label data, if
            label_vector_uri is set. See docs for GeoJSONVectorSourceConfig
            for more details. Defaults to {}.
        label_source_kw (dict, optional): Additional arguments to pass
            to the ChipClassificationLabelSourceConfig used for label data, if
            label_vector_uri is set. See docs for
            ChipClassificationLabelSourceConfig for more details.
            Defaults to {}.
        scene_id (Optional[str]): Optional scene ID. If None, will be randomly
            generated. Defaults to None.

    Returns:
        Scene: A chip classification scene.
    """
    # use local imports to avoid circular import problems
    from rastervision.core.data import (
        RasterioSource, Scene, ClassInferenceTransformerConfig,
        ChipClassificationLabelSourceConfig, GeoJSONVectorSourceConfig)

    image_uri = listify_uris(image_uri)
    raster_source = RasterioSource(image_uri, **image_raster_source_kw)

    crs_transformer = raster_source.crs_transformer
    extent = raster_source.extent

    label_source = None
    if label_vector_uri is not None:
        if class_config is None:
            raise ValueError('class_config is required if using labels.')
        if label_vector_default_class_id is not None:
            # add a ClassInferenceTransformer to the VectorSource
            class_inf_tf = ClassInferenceTransformerConfig(
                default_class_id=label_vector_default_class_id)
            vector_tfs = label_vector_source_kw.get('transformers', [])
            label_vector_source_kw['transformers'] = (
                [class_inf_tf] + vector_tfs)
        geojson_cfg = GeoJSONVectorSourceConfig(
            uri=label_vector_uri,
            ignore_crs_field=True,
            **label_vector_source_kw)
        # use config to ensure required transformers are auto added
        label_source_cfg = ChipClassificationLabelSourceConfig(
            vector_source=geojson_cfg, **label_source_kw)
        label_source = label_source_cfg.build(
            class_config, crs_transformer, extent=extent)

    aoi_polygons = get_polygons_from_uris(aoi_uri, crs_transformer)
    scene = Scene(
        id=uuid4() if scene_id is None else scene_id,
        raster_source=raster_source,
        label_source=label_source,
        aoi_polygons=aoi_polygons)

    return scene


def make_od_scene(image_uri: Union[str, List[str]],
                  label_vector_uri: Optional[str] = None,
                  class_config: Optional['ClassConfig'] = None,
                  aoi_uri: Union[str, List[str]] = [],
                  label_vector_default_class_id: Optional[int] = None,
                  image_raster_source_kw: dict = {},
                  label_vector_source_kw: dict = {},
                  label_source_kw: dict = {},
                  scene_id: Optional[str] = None) -> 'Scene':
    """Create an object detection scene from image and label URIs.

    This is a convenience method. For more fine-grained control, it is
    recommended to use the default constructor.

    Args:
        image_uri (Union[str, List[str]]): URI or list of URIs of GeoTIFFs to
            use as the source of image data.
        label_vector_uri (Optional[str], optional):  URI of GeoJSON file to use
            as the source of segmentation label data. Defaults to None.
        class_config (Optional[ClassConfig]): The ClassConfig. Must be
            non-None if creating a scene without a LabelSource.
            Defaults to None.
        aoi_uri (Union[str, List[str]], optional): URI or list of URIs of
            GeoJSONs that specify the area-of-interest. If provided, the
            dataset will only access data from this area. Defaults to [].
        label_vector_default_class_id (Optional[int], optional): If using
            label_vector_uri and all polygons in that file belong to the same
            class and they do not contain a `class_id` property, then use this
            argument to map all of the polgons to the appropriate class ID.
            See docs for ClassInferenceTransformer for more details.
            Defaults to None.
        image_raster_source_kw (dict, optional): Additional arguments to pass
            to the RasterioSource used for image data. See docs for
            RasterioSource for more details. Defaults to {}.
        label_vector_source_kw (dict, optional): Additional arguments to pass
            to the GeoJSONVectorSourceConfig used for label data, if
            label_vector_uri is set. See docs for GeoJSONVectorSourceConfig
            for more details. Defaults to {}.
        label_source_kw (dict, optional): Additional arguments to pass
            to the ObjectDetectionLabelSourceConfig used for label data, if
            label_vector_uri is set. See docs for
            ObjectDetectionLabelSourceConfig for more details.
            Defaults to {}.
        scene_id (Optional[str]): Optional scene ID. If None, will be randomly
            generated. Defaults to None.

    Returns:
        Scene: An object detection scene.
    """
    # use local imports to avoid circular import problems
    from rastervision.core.data import (
        RasterioSource, Scene, ClassInferenceTransformerConfig,
        GeoJSONVectorSourceConfig, ObjectDetectionLabelSourceConfig)

    image_uri = listify_uris(image_uri)
    raster_source = RasterioSource(image_uri, **image_raster_source_kw)

    crs_transformer = raster_source.crs_transformer
    extent = raster_source.extent

    label_source = None
    if label_vector_uri is not None:
        if class_config is None:
            raise ValueError('class_config is required if using labels.')
        if label_vector_default_class_id is not None:
            # add a ClassInferenceTransformer to the VectorSource
            class_inf_tf = ClassInferenceTransformerConfig(
                default_class_id=label_vector_default_class_id)
            vector_tfs = label_vector_source_kw.get('transformers', [])
            label_vector_source_kw['transformers'] = (
                [class_inf_tf] + vector_tfs)
        geojson_cfg = GeoJSONVectorSourceConfig(
            uri=label_vector_uri,
            ignore_crs_field=True,
            **label_vector_source_kw)
        # use config to ensure required transformers are auto added
        label_source_cfg = ObjectDetectionLabelSourceConfig(
            vector_source=geojson_cfg, **label_source_kw)
        label_source = label_source_cfg.build(
            class_config, crs_transformer, extent=extent)

    aoi_polygons = get_polygons_from_uris(aoi_uri, crs_transformer)
    scene = Scene(
        id=uuid4() if scene_id is None else scene_id,
        raster_source=raster_source,
        label_source=label_source,
        aoi_polygons=aoi_polygons)

    return scene
