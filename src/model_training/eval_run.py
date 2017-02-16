"""
Compute prediction images, learning curve graph and various metrics for a run.
"""
from os.path import join
import json

from keras.models import load_model
import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
from sklearn import metrics

from process_data import (results_path,
                          one_hot_to_rgb_batch,
                          one_hot_to_label_batch,
                          rgb_to_label_batch,
                          label_names,
                          make_input_output_generators,
                          make_data_generator,
                          proc_data_path,
                          VALIDATION,
                          RGB_INPUT,
                          OUTPUT)

np.random.seed(1337)


def plot_predictions(model, run_path, nb_prediction_images, include_depth):
    _, validation_generator = make_input_output_generators(
        nb_prediction_images, include_depth)

    inputs, _ = next(validation_generator)
    # Get unscaled RGB images for display
    display_inputs = next(
        make_data_generator(
            join(proc_data_path, VALIDATION, RGB_INPUT),
            batch_size=nb_prediction_images, shuffle=True, augment=True))
    display_outputs = next(
        make_data_generator(
            join(proc_data_path, VALIDATION, OUTPUT),
            batch_size=nb_prediction_images, shuffle=True, augment=True))

    predictions = one_hot_to_rgb_batch(
        model.predict(inputs, nb_prediction_images))

    fig = plt.figure()
    subplot_index = 0
    nb_subplot_cols = 3
    gs = mpl.gridspec.GridSpec(nb_prediction_images, nb_subplot_cols)
    gs.update(wspace=0.1, hspace=0.1, left=0.1, right=0.4, bottom=0.1, top=0.9)

    def plot_image(subplot_index, im, title):
        a = fig.add_subplot(gs[subplot_index])
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        a.imshow(im.astype(np.uint8))
        if subplot_index < nb_subplot_cols:
            a.set_title(title, fontsize=6)

    for i in range(nb_prediction_images):
        plot_image(subplot_index, display_inputs[i, :, :, :], 'Input')
        subplot_index += 1
        plot_image(subplot_index, display_outputs[i, :, :, :], 'Ground Truth')
        subplot_index += 1
        plot_image(subplot_index, predictions[i, :, :, :], 'Prediction')
        subplot_index += 1

    predictions_path = join(run_path, 'predictions.pdf')
    plt.savefig(predictions_path, bbox_inches='tight', format='pdf', dpi=300)


def get_samples(data_gen, batch_size, nb_samples):
    samples = []
    for _ in range(0, nb_samples, batch_size):
        samples.append(next(data_gen))
    return np.concatenate(samples, axis=0)[0:nb_samples, :, :, :]


def compute_scores(model, run_path, batch_size, nb_val_samples, include_depth):
    _, validation_generator = make_input_output_generators(
        batch_size, include_depth)

    input_gen = map(lambda x: x[0], validation_generator)
    output_gen = map(lambda x: x[1], validation_generator)

    inputs = get_samples(input_gen, batch_size, nb_val_samples)
    predictions = one_hot_to_label_batch(model.predict(inputs))
    outputs = rgb_to_label_batch(get_samples(
        output_gen, batch_size, nb_val_samples))

    # Treat each pixel as a separate data point so we can use metric functions.
    predictions = np.ravel(predictions)
    outputs = np.ravel(outputs)

    # See https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection#evaluation # NOQA
    jaccard = metrics.jaccard_similarity_score(outputs, predictions)
    avg_f1 = metrics.f1_score(outputs, predictions, average='macro')
    avg_accuracy = metrics.accuracy_score(outputs, predictions)
    precision, recall, f1, support = \
        metrics.precision_recall_fscore_support(outputs, predictions)
    confusion_mat = metrics.confusion_matrix(outputs, predictions)
    # Avoid divide by zero error by adding 0.1
    accuracy = confusion_mat.diagonal() / (support + 0.1)

    scores = {
        'label_names': label_names,
        'jaccard': jaccard,
        'avg_f1': avg_f1,
        'avg_accuracy': avg_accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist(),
        'confusion_mat': confusion_mat.tolist(),
        'accuracy': accuracy.tolist()
    }
    scores_json = json.dumps(scores, sort_keys=True, indent=4)
    print(scores_json)
    with open(join(run_path, 'scores.txt'), 'w') as scores_file:
        scores_file.write(scores_json)

    return scores


def plot_graphs(model, run_path):
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

    plt.legend(loc="best")
    accuracy_path = join(run_path, 'accuracy.pdf')
    plt.savefig(accuracy_path, format='pdf', dpi=300)


def eval_run(options):
    run_path = join(results_path, options.run_name)
    model = load_model(join(run_path, 'model.h5'))

    print('Plotting predictions...')
    plot_predictions(model, run_path, options.nb_prediction_images,
                     options.include_depth)

    print('Computing scores...')
    compute_scores(model, run_path, options.batch_size, options.nb_val_samples,
                   options.include_depth)

    print('Plotting graphs...')
    plot_graphs(model, run_path)
