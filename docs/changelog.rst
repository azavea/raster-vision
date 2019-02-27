CHANGELOG
=========

Raster Vision 0.9
-----------------

Raster Vision 0.9.0
~~~~~~~~~~~~~~~~~~~
- Conserve disk space when dealing with raster files `#692 <https://github.com/azavea/raster-vision/pull/692>`_
- Optimize StatsAnalyzer `#690 <https://github.com/azavea/raster-vision/pull/690>`_
- Include per-scene eval metrics `#641 <https://github.com/azavea/raster-vision/pull/641>`_
- Make and save predictions and do eval chip-by-chip `#635 <https://github.com/azavea/raster-vision/pull/635>`_
- Decrease semseg memory usage `#630 <https://github.com/azavea/raster-vision/pull/630>`_
- Add support for vector tiles in .mbtiles files `#601 <https://github.com/azavea/raster-vision/pull/601>`_
- Add support for getting labels from zxy vector tiles `#532 <https://github.com/azavea/raster-vision/pull/532>`_
- Remove custom ``__deepcopy__`` implementation from ``ConfigBuilder``s. `#567 <https://github.com/azavea/raster-vision/pull/567>`_
- Add ability to shift raster images by given numbers of meters.  `#573 <https://github.com/azavea/raster-vision/pull/573>`_
- Add ability to generate GeoJSON segmentation predictions.  `#575 <https://github.com/azavea/raster-vision/pull/575>`_
- Add ability to run the DeepLab eval script.  `#653 <https://github.com/azavea/raster-vision/pull/653>`_
- Submit CPU-only stages to a CPU queue on Aws.  `#668 <https://github.com/azavea/raster-vision/pull/668>`_

Raster Vision 0.8
-----------------

Raster Vision 0.8.2
~~~~~~~~~~~~~~~~~~~

Bug Fixes
^^^^^^^^^
- Fixed issue with handling width > height in semseg eval `#627 <https://github.com/azavea/raster-vision/pull/627>`_
- Fixed issue with experiment configs not setting key names correctly `#523 <https://github.com/azavea/raster-vision/pull/576>`_
- Fixed issue with Raster Sources that have channel order `#503 <https://github.com/azavea/raster-vision/pull/576>`_

Raster Vision 0.8.1
~~~~~~~~~~~~~~~~~~~

Bug Fixes
^^^^^^^^^
- Allow multiploygon for chip classification `#523 <https://github.com/azavea/raster-vision/pull/523>`_
- Remove unused args for AWS Batch runner `#503 <https://github.com/azavea/raster-vision/pull/503>`_
- Skip over lines when doing chip classification, Use background_class_id for scenes with no polygons `#507 <https://github.com/azavea/raster-vision/pull/507>`_
- Fix issue where ``get_matching_s3_keys`` fails when ``suffix`` is ``None`` `#497 <https://github.com/azavea/raster-vision/pull/497>`_
