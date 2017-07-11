from os.path import join
import json
from json import encoder

from sklearn.metrics import fbeta_score
import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from rastervision.common.settings import VALIDATION
from rastervision.common.utils import plot_img_row, _makedirs

from rastervision.tagging.data.planet_kaggle import TagStore

VALIDATION_EVAL = 'validation_eval'

encoder.FLOAT_REPR = lambda o: format(o, '.5f')


class Scores():
    """A set of scores for the performance of a model on a dataset."""
    def __init__(self, y_true, y_pred, dataset, active_tags):
        self.f2_samples = fbeta_score(
            y_true, y_pred, beta=2, average='samples')
        self.f2_labels = fbeta_score(y_true, y_pred, beta=2,
                                     average='macro')
        f2_subscores = fbeta_score(y_true, y_pred, beta=2, average=None)

        self.atmos_scores, self.common_scores, self.rare_scores = {}, {}, {}
        for tag_ind, tag in enumerate(active_tags):
            f2_subscore = f2_subscores[tag_ind]
            if tag in dataset.atmos_tags:
                self.atmos_scores[tag] = f2_subscore
            if tag in dataset.common_tags:
                self.common_scores[tag] = f2_subscore
            if tag in dataset.rare_tags:
                self.rare_scores[tag] = f2_subscore

    def to_json(self):
        return json.dumps(self.__dict__, sort_keys=True, indent=4)

    def save(self, path):
        scores_json = self.to_json()
        with open(path, 'w') as scores_file:
            scores_file.write(scores_json)


def plot_prediction(generator, all_x, y_true, y_pred,
                    file_path):
    dataset = generator.dataset
    fig = plt.figure()

    nb_subplot_cols = 3 + len(generator.active_input_inds)
    grid_spec = mpl.gridspec.GridSpec(1, nb_subplot_cols)

    all_x = generator.calibrate_image(all_x)
    rgb_input_im = all_x[:, :, dataset.rgb_inds]
    imgs = [rgb_input_im]
    titles = ['RGB']

    if generator.dataset.nb_channels > 3:
        ir_im = all_x[:, :, dataset.ir_ind]
        imgs.append(ir_im)
        titles.append('IR')

        if dataset.ndvi_ind:
            ndvi_im = all_x[:, :, dataset.ndvi_ind]
            imgs.append(ndvi_im)
            titles.append('NDVI')

    plot_img_row(fig, grid_spec, 0, imgs, titles)

    add_pred_tags, remove_pred_tags = \
        generator.tag_store.get_tag_diff(y_true, y_pred)
    y_true_strs = sorted(generator.tag_store.binary_to_strs(y_true))

    y_true_strs = ', '.join(y_true_strs)
    add_pred_tags = ', '.join(add_pred_tags)
    remove_pred_tags = ', '.join(remove_pred_tags)
    tag_info = 'ground truth: {}\nfalse +: {}\nfalse -: {}'.format(
        y_true_strs, add_pred_tags, remove_pred_tags)
    fig.text(0.15, 0.35, tag_info, fontsize=5)

    plt.savefig(file_path, bbox_inches='tight', format='png', dpi=300)
    plt.close(fig)


def plot_predictions(run_path, options, generator):
    validation_pred_path = join(run_path, 'validation_preds.csv')

    validation_plot_path = join(run_path, 'validation_plots')
    _makedirs(validation_plot_path)

    validation_pred_tag_store = TagStore(
        tags_path=validation_pred_path, active_tags=options.active_tags)
    split_gen = generator.make_split_generator(
        VALIDATION, target_size=None,
        batch_size=options.batch_size, shuffle=False, augment_methods=None,
        normalize=True, only_xy=False)

    sample_count = 0
    plot_sample_count = 0
    y_trues = []
    y_preds = []
    for batch_ind, batch in enumerate(split_gen):
        for sample_ind in range(batch.x.shape[0]):
            file_ind = batch.file_inds[sample_ind]
            all_x = batch.all_x[sample_ind, :, :, :]

            y_true = generator.tag_store.get_tag_array([file_ind])
            y_trues.append(y_true)
            y_pred = validation_pred_tag_store.get_tag_array([file_ind])
            y_preds.append(y_pred)

            if (options.nb_eval_plot_samples is None or
                    plot_sample_count < options.nb_eval_plot_samples):
                is_mistake = not np.array_equal(y_true, y_pred)
                if is_mistake:
                    plot_sample_count += 1
                    plot_path = join(
                        validation_plot_path, '{}_debug.png'.format(file_ind))
                    plot_prediction(
                        generator, all_x, y_true[0, :], y_pred[0, :],
                        plot_path)

            sample_count += 1

            if (options.nb_eval_samples is not None and
                    sample_count >= options.nb_eval_samples):
                break

        if (options.nb_eval_samples is not None and
                sample_count >= options.nb_eval_samples):
            break

    y_true = np.concatenate(y_trues, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    if options.nb_eval_samples is not None:
        y_true = y_true[0:options.nb_eval_samples, :]
        y_pred = y_pred[0:options.nb_eval_samples, :]

    return y_true, y_pred


def validation_eval(run_path, options, generator):
    y_true, y_pred = plot_predictions(run_path, options, generator)

    scores = Scores(
        y_true, y_pred, generator.dataset, generator.tag_store.active_tags)
    scores_path = join(run_path, 'scores.json')
    scores.save(scores_path)
