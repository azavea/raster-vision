CHANGELOG
=========

Raster Vision 0.9
-----------------

Raster Vision 0.9.0
~~~~~~~~~~~~~~~~~~~

- Remove custom ``__deepcopy__`` implementation from ``ConfiBuilder``s. `#567 <https://github.com/azavea/raster-vision/pull/567>`_
- Add ability to shift raster images by given numbers of meters.  `#573 <https://github.com/azavea/raster-vision/pull/573>`_

Raster Vision 0.8
-----------------

Raster Vision 0.8.2
~~~~~~~~~~~~~~~~~~~

Bug Fixes
^^^^^^^^^
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
