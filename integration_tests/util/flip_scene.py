import click
import numpy as np
import rasterio
import pyproj
import json


def flip_geom(m, b, geom):
    """Flips a geom along a straight line y = mx + b.
    """

    def traverse_coords(coords, dst_coords):
        for p in coords:
            if type(p[0]) is list:
                lst = []
                traverse_coords(p, lst)
                dst_coords.append(lst)
            else:
                x, y = p[0], p[1]
                d = (x + (y - b) * m) / (1 + m * m)
                x2 = 2 * d - x
                y2 = 2 * d * m - y + 2 * b
                dst_coords.append((x2, y2))
        return dst_coords

    return {
        'type': geom['type'],
        'coordinates': traverse_coords(geom['coordinates'], [])
    }


@click.command()
@click.argument('src_tiff_path')
@click.argument('src_labels_path')
@click.argument('dst_tiff_path')
@click.argument('dst_labels_path')
def flip_scene(src_tiff_path, src_labels_path, dst_tiff_path, dst_labels_path):
    """Flips a scene and it's labels.

    Useful for generating multiple training scenes for integration test usage.
    """

    labels_are_tif = src_labels_path.endswith('.tif')

    with rasterio.open(src_tiff_path) as src:
        profile = src.profile
        bands = src.read()

        with rasterio.open(dst_tiff_path, 'w', **profile) as dst:
            fbands = np.flip(bands, 1)
            dst.write(fbands)

        if not labels_are_tif:

            img_crs = pyproj.Proj(init=src.crs['init'])
            map_crs = pyproj.Proj(init='epsg:4326')

            def t(x, y):
                return pyproj.transform(img_crs, map_crs, x, y)

            # Find the center horizontal line through the image.

            ll = (src.bounds.left, src.bounds.bottom)
            ul = (src.bounds.left, src.bounds.top)
            ur = (src.bounds.right, src.bounds.top)
            lr = (src.bounds.right, src.bounds.bottom)

            left = t(ul[0] - ((ul[0] - ll[0]) / 2),
                     ul[1] - ((ul[1] - ll[1]) / 2))

            right = t(ur[0] - ((ur[0] - lr[0]) / 2),
                      ur[1] - ((ur[1] - lr[1]) / 2))

            m = abs(left[1] - right[1]) / abs(left[0] - right[0])
            b = left[1] - (m * left[0])

    if labels_are_tif:
        with rasterio.open(src_labels_path) as src:
            profile = src.profile
            bands = src.read()

            with rasterio.open(dst_labels_path, 'w', **profile) as dst:
                fbands = np.flip(bands, 1)
                dst.write(fbands)
    else:

        def traverse_labels(src, dst):
            for key in src:
                e = src[key]
                if type(e) is dict:
                    if key == 'geometry':
                        dst[key] = flip_geom(m, b, src[key])
                    else:
                        dst[key] = {}
                        traverse_labels(e, dst[key])
                elif type(e) is list:
                    d_list = []
                    for x in e:
                        if type(x) is dict:
                            ne = {}
                            traverse_labels(x, ne)
                            d_list.append(ne)
                        else:
                            d_list.append(x)
                    dst[key] = d_list
                else:
                    dst[key] = e
            return dst

        with open(src_labels_path) as src_labels_file:
            source_labels = json.loads(src_labels_file.read())

        dst_labels = traverse_labels(source_labels, {})

        with open(dst_labels_path, 'w') as dst_labels_file:
            dst_labels_file.write(json.dumps(dst_labels, indent=4))

    print('done.')


if __name__ == '__main__':
    flip_scene()
