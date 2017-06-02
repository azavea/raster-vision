from os.path import join

import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
from matplotlib import cm

from rastervision.common.settings import results_path

PLOT_CURVES = 'plot_curves'


def plot_multi_curves(options):
    run_path = join(results_path, options.run_name)
    run_names = [run_name for run_name in options.aggregate_run_names]

    fig, (ax0, ax1) = plt.subplots(
        2, 1, sharex=True, sharey=True)
    ax0.grid()
    ax0.set_ylabel('Training Accuracy')

    ax1.set_ylabel('Validation Accuracy')
    ax1.grid()
    ax1.set_xlabel('Epochs')

    run_name_to_color = dict(zip(
        run_names, cm.jet(np.linspace(0, 1, len(run_names)))))
    handles = []

    for run_name in run_names:
        log_path = join(results_path, run_name, 'log.txt')
        log = np.genfromtxt(log_path, delimiter=',', skip_header=1)
        epochs = log[:, 0]
        acc = log[:, 1]
        val_acc = log[:, 3]

        handle, = ax0.plot(epochs, acc, color=run_name_to_color[run_name],
                           label=run_name)
        handles.append(handle)
        ax1.plot(epochs, val_acc,
                 color=run_name_to_color[run_name],
                 label=run_name)

    fig.legend(handles=handles, labels=run_names, loc='upper center')

    accuracy_path = join(run_path, 'accuracy.pdf')
    plt.savefig(accuracy_path, format='pdf')


def plot_single_curves(options):
    plt.figure()
    plt.grid()

    run_path = join(results_path, options.run_name)
    log_path = join(run_path, 'log.txt')
    log = np.genfromtxt(log_path, delimiter=',', skip_header=1)
    epochs = log[:, 0]
    acc = log[:, 1]
    val_acc = log[:, 3]

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs, acc, '-', label='Training')
    plt.plot(epochs, val_acc, '--', label='Validation')
    plt.legend(loc='best')

    accuracy_path = join(run_path, 'accuracy.pdf')
    plt.savefig(accuracy_path, format='pdf', dpi=300)


def plot_curves(options):
    """Plot the training and validation accuracy over epochs.

    # Arguments
        options: the options for a run
    """
    if options.aggregate_run_names is None:
        plot_single_curves(options)
    else:
        plot_multi_curves(options)
