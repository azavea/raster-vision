from os.path import join, isdir
import glob
from subprocess import call

import numpy as np

from rastervision.common.utils import _makedirs
from rastervision.common.settings import VALIDATION

from rastervision.semseg.tasks.utils import (
    make_prediction_img, plot_prediction, predict_x)
from rastervision.semseg.models.factory import SemsegModelFactory

MAKE_VIDEOS = 'make_videos'


def make_videos(run_path, options, generator):
    model_factory = SemsegModelFactory()
    videos_path = join(run_path, 'videos')
    _makedirs(videos_path)

    checkpoints_path = join(run_path, 'delta_model_checkpoints')
    if not isdir(checkpoints_path):
        print('Cannot make videos without delta_model_checkpoints.')
        return

    model_paths = glob.glob(join(checkpoints_path, '*.h5'))
    model_paths.sort()
    models = []
    for model_path in model_paths:
        model = model_factory.make_model(options, generator)
        model.load_weights(model_path, by_name=True)
        models.append(model)

    split_gen = generator.make_split_generator(
        VALIDATION, target_size=options.eval_target_size,
        batch_size=1, shuffle=False, augment_methods=None, normalize=True,
        only_xy=False)

    for video_ind, batch in \
            enumerate(split_gen):
        x = np.squeeze(batch.x, axis=0)
        y = np.squeeze(batch.y, axis=0)
        display_y = generator.dataset.one_hot_to_rgb_batch(y)
        all_x = np.squeeze(batch.all_x, axis=0)

        make_video(
            x, display_y, all_x, models, videos_path, video_ind,
            options, generator)

        if video_ind == options.nb_videos - 1:
            break


def make_video(x, y, all_x, models, videos_path, video_ind, options,
               generator):
    video_path = join(videos_path, str(video_ind))
    _makedirs(video_path)

    for frame_ind, model in enumerate(models):
        y_pred = make_prediction_img(
            x, options.target_size[0],
            lambda x: generator.dataset.one_hot_to_rgb_batch(
                predict_x(x, model)))
        print(video_ind)
        print(frame_ind)
        frame_path = join(
            video_path, 'frame_{:0>4}.png'.format(frame_ind))
        plot_prediction(generator, all_x, y, y_pred, frame_path)

    frames_path = join(video_path, 'frame_%04d.png')
    video_path = join(videos_path, '{}.mp4'.format(video_ind))
    call(['avconv',
          '-r', '2',
          '-i', frames_path,
          '-vf', 'scale=trunc(in_w/2)*2:trunc(in_h/2)*2',
          video_path])
