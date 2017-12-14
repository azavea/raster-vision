import os
import shutil
import glob
import random

import click

from rv.utils import make_empty_dir


@click.command()
@click.argument('dataset_dir')
@click.argument('split1_dir')
@click.argument('split2_dir')
@click.option('--split1-ratio', default=0.8)
def split_dataset(dataset_dir, split1_dir, split2_dir, split1_ratio):
    subdirs = [name for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name))]
    make_empty_dir(split1_dir)
    make_empty_dir(split2_dir)

    for subdir in subdirs:
        make_empty_dir(os.path.join(split1_dir, subdir))
        make_empty_dir(os.path.join(split2_dir, subdir))

    for subdir in subdirs:
        paths = glob.glob(os.path.join(dataset_dir, subdir, '*'))
        random.shuffle(paths)
        split1_size = int(len(paths) * split1_ratio)
        split1_paths = paths[0:split1_size]
        split2_paths = paths[split1_size:]

        for path in split1_paths:
            filename = os.path.basename(path)
            shutil.copyfile(
                path, os.path.join(split1_dir, subdir, filename))

        for path in split2_paths:
            filename = os.path.basename(path)
            shutil.copyfile(
                path, os.path.join(split2_dir, subdir, filename))


if __name__ == '__main__':
    split_dataset()
