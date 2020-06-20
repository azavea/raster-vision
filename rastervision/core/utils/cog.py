import os
from subprocess import Popen

from rastervision.pipeline.file_system import (download_or_copy,
                                               upload_or_copy)

COGIFY = 'COGIFY'

DEFAULT_BLOCK_SIZE = 512
DEFAULT_RESAMPLE_METHOD = 'near'
DEFAULT_COMPRESSION = 'deflate'
DEFAULT_OVERVIEWS = [2, 4, 8, 16, 32]


def gdal_cog_commands(input_path,
                      tmp_dir,
                      block_size=DEFAULT_BLOCK_SIZE,
                      resample_method=DEFAULT_RESAMPLE_METHOD,
                      compression=DEFAULT_COMPRESSION,
                      overviews=None):
    """
    GDAL commands to create a COG from an input file.
    Returns a tuple (commands, output_path)
    """

    if not overviews:
        overviews = DEFAULT_OVERVIEWS

    def get_output_path(command):
        fname = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(tmp_dir, '{}-{}.tif'.format(fname, command))

    compression = compression.lower()

    def add_compression(cmd, overview=False):
        if compression != 'none':
            if not overview:
                return cmd[:1] + ['-co', 'compress={}'.format(compression)
                                  ] + cmd[1:]
            else:
                return cmd[:1] + [
                    '--config', 'COMPRESS_OVERVIEW', compression
                ] + cmd[1:]
        else:
            return cmd

    # Step 1: Translate to a GeoTiff.
    translate_path = get_output_path('translate')
    translate = add_compression([
        'gdal_translate', '-of', 'GTiff', '-co', 'tiled=YES', '-co',
        'BIGTIFF=IF_SAFER', input_path, translate_path
    ])

    # Step 2: Add overviews
    add_overviews = add_compression(
        ['gdaladdo', '-r', resample_method, translate_path] + list(
            map(lambda x: str(x), overviews)),
        overview=True)

    # Step 3: Translate to COG
    output_path = get_output_path('cog')

    create_cog = add_compression([
        'gdal_translate', '-co', 'TILED=YES', '-co', 'COPY_SRC_OVERVIEWS=YES',
        '-co', 'BLOCKXSIZE={}'.format(block_size), '-co',
        'BLOCKYSIZE={}'.format(block_size), '-co', 'BIGTIFF=IF_SAFER',
        '--config', 'GDAL_TIFF_OVR_BLOCKSIZE',
        str(block_size), translate_path, output_path
    ])

    return ([translate, add_overviews, create_cog], output_path)


def run_cmd(cmd):
    p = Popen(cmd)
    (out, err) = p.communicate(input)
    if p.returncode != 0:
        s = 'Command failed:\n'
        s += ' '.join(cmd) + '\n\n'
        if out:
            s += out + '\n\n'
        if err:
            s += err
        raise Exception(s)


def create_cog(source_uri,
               dest_uri,
               local_dir,
               block_size=DEFAULT_BLOCK_SIZE,
               resample_method=DEFAULT_RESAMPLE_METHOD,
               compression=DEFAULT_COMPRESSION,
               overviews=None):
    local_path = download_or_copy(source_uri, local_dir)

    commands, output_path = gdal_cog_commands(
        local_path,
        local_dir,
        block_size=block_size,
        resample_method=resample_method,
        compression=compression,
        overviews=overviews)
    for command in commands:
        run_cmd(command)

    upload_or_copy(output_path, dest_uri)
