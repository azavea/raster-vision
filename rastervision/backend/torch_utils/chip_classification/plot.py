def plot_xy(ax, x, y=None, label_names=None):
    ax.imshow(x.permute(1, 2, 0))
    if y is not None:
        ax.set_title(label_names[y])
    ax.axis('off')
