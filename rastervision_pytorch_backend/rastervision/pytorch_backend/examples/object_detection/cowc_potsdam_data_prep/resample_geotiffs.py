import os
import glob
import subprocess

import click

from rv.utils import make_empty_dir


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option(
    '--resolution', default=0.3, help='Output resolution in meters/pixel')
def resample_geotiffs(input_dir, output_dir, resolution):
    input_paths = glob.glob(os.path.join(input_dir, '*.tif'))
    make_empty_dir(output_dir)

    for input_path in input_paths:
        input_filename = os.path.basename(input_path)
        subprocess.call([
            'gdalwarp', '-tr',
            str(resolution),
            str(-resolution), '-r', 'bilinear', '-setci', '-co', 'ALPHA=NO',
            input_path,
            os.path.join(output_dir, input_filename)
        ])


if __name__ == '__main__':
    resample_geotiffs()
