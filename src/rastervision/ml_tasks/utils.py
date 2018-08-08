def compare_window_to_aoi(window, aoi_polygons):

    if not aoi_polygons:
        return True

    window_shapely = window.get_shapely()
    for polygon in aoi_polygons:
        if window_shapely.within(polygon):
            return True

    return False
