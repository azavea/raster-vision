import os
import glob
import random

from PIL import Image
import numpy as np
import click


@click.command()
@click.argument('image_dir')
@click.option('--sample-size', default=100)
@click.option('--file-extension', default='png')
def get_image_stats(image_dir, sample_size, file_extension):
    paths = glob.glob(
        os.path.join(image_dir, '**', '*.{}'.format(file_extension)),
        recursive=True)
    random.shuffle(paths)
    ims = []
    for path in paths[0:sample_size]:
        im = np.array(Image.open(path))
        im = np.reshape(im, (-1, 3))
        ims.append(im)
    ims = np.concatenate(ims, axis=0) / 255.0
    means = np.mean(ims, axis=0)
    stds = np.std(ims, axis=0)
    print('Means: {}'.format(means))
    print('Stds: {}'.format(stds))


if __name__ == '__main__':
    get_image_stats()
