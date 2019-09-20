import matplotlib.patches as patches


def plot_xy(ax, x, y=None, label_names=None):
    ax.imshow(x.permute(1, 2, 0))

    if y is not None:
        scores = y.get_field('scores')
        for box_ind, (box, label) in enumerate(
                zip(y.boxes, y.get_field('labels'))):
            rect = patches.Rectangle(
                (box[1], box[0]),
                box[3] - box[1],
                box[2] - box[0],
                linewidth=1,
                edgecolor='cyan',
                facecolor='none')
            ax.add_patch(rect)

            label_name = label_names[label]
            if scores is not None:
                score = scores[box_ind]
                label_name += ' {:.2f}'.format(score)

            h, w = x.shape[1:]
            label_height = h * 0.03
            label_width = w * 0.15
            rect = patches.Rectangle(
                (box[1], box[0] - label_height),
                label_width,
                label_height,
                color='cyan')
            ax.add_patch(rect)

            ax.text(
                box[1] + w * 0.003, box[0] - h * 0.003, label_name, fontsize=7)

    ax.axis('off')
