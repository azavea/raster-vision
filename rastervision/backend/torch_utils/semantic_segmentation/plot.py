from rastervision.data.label_source.utils import color_to_triple
import matplotlib
matplotlib.use('Agg')  # noqa


def plot_xy(ax, x, class_map, y=None):
    ax.axis('off')
    x = x.permute((1, 2, 0)).numpy()
    ax.imshow(x)

    if y is not None:
        colors = [class_map.get_by_id(i).color for i in range(len(class_map))]
        colors = [color_to_triple(c) for c in colors]
        colors = [tuple([_c / 255 for _c in c]) for c in colors]
        cmap = matplotlib.colors.ListedColormap(colors)

        y = y.numpy()
        ax.imshow(y, alpha=0.4, vmin=0, vmax=len(colors), cmap=cmap)
