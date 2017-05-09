from os.path import join

import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

PLOT_CURVES = 'plot_curves'


def plot_curves(run_path):
    """Plot the training and validation accuracy over epochs.

    # Arguments
        run_path: the path to the files for a run
    """
    log_path = join(run_path, 'log.txt')

    log = np.genfromtxt(log_path, delimiter=',', skip_header=1)
    epochs = log[:, 0]
    acc = log[:, 1]
    val_acc = log[:, 3]

    plt.figure()
    plt.title('Training Log')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.grid()
    plt.plot(epochs, acc, '-', label='Training')
    plt.plot(epochs, val_acc, '--', label='Validation')

    plt.legend(loc='best')
    accuracy_path = join(run_path, 'accuracy.pdf')
    plt.savefig(accuracy_path, format='pdf', dpi=300)
