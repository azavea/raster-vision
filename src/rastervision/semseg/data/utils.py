import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt


def plot_sample(file_path, batch_x, batch_y, generator):
    batch_x = generator.unnormalize(batch_x)
    dataset = generator.dataset

    fig = plt.figure()
    nb_input_inds = batch_x.shape[2]
    nb_output_inds = batch_y.shape[2]

    gs = mpl.gridspec.GridSpec(2, 7)

    def plot_img(plot_row, plot_col, im, is_rgb=False):
        a = fig.add_subplot(gs[plot_row, plot_col])
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)

        if is_rgb:
            a.imshow(im.astype(np.uint8))
        else:
            a.imshow(im, cmap='gray', vmin=0, vmax=255)

    plot_row = 0
    plot_col = 0
    im = batch_x[:, :, dataset.rgb_inds]
    plot_img(plot_row, plot_col, im, is_rgb=True)

    for channel_ind in range(nb_input_inds):
        plot_col += 1
        if channel_ind == dataset.ndvi_ind:
            im = (np.clip(batch_x[:, :, channel_ind], -1, 1) + 1) * 100
        else:
            im = batch_x[:, :, channel_ind]
        plot_img(plot_row, plot_col, im)

    plot_row = 1
    plot_col = 0
    rgb_batch_y = dataset.one_hot_to_rgb_batch(batch_y)
    plot_img(plot_row, plot_col, rgb_batch_y, is_rgb=True)

    for channel_ind in range(nb_output_inds):
        plot_col += 1
        im = batch_y[:, :, channel_ind] * 150
        plot_img(plot_row, plot_col, im)

    plt.savefig(file_path, bbox_inches='tight', format='pdf', dpi=600)
    plt.close(fig)
