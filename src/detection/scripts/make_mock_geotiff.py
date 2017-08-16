import argparse
from os.path import splitext

from scipy.ndimage import imread
import rasterio
from rasterio.transform import from_origin


def make_geotiff(image_path):
    im = imread(image_path)
    height, width, nchannels = im.shape
    out_path = splitext(image_path)[0] + '.tif'

    transform = from_origin(
        -75.163506, 39.952536, 0.000001, 0.000001)

    with rasterio.open(out_path, 'w', driver='GTiff', height=height,
                       transform=transform, crs='EPSG:4326',
                       compression=rasterio.enums.Compression.none,
                       width=width, count=nchannels, dtype=im.dtype) as dst:
        for channel_ind in range(0, nchannels):
            dst.write(im[:, :, channel_ind], channel_ind + 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path')

    return parser.parse_args()


def run():
    args = parse_args()
    print('image_path: {}'.format(args.image_path))
    make_geotiff(args.image_path)


if __name__ == '__main__':
    run()
