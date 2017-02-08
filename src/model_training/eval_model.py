"""
Take the learned model and then make predictions for some validation data.
For each image, save the original input, the ground truth output, and the
predicted image. Also, compute various metrics on the model.
"""
from os.path import join
import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
np.random.seed(1337)
from sklearn import metrics
from keras.models import load_model

from process_data import (model_path, eval_path,
    one_hot_to_rgb_batch, one_hot_to_label_batch, rgb_to_label_batch,
    save_image, label_names,
    make_input_output_generators, make_data_generator,
    proc_data_path, VALIDATION, INPUT, OUTPUT)

def visualize_predictions(model, batch_size):
    _, validation_generator = make_input_output_generators(batch_size)
    inputs, _ = next(validation_generator)

    # Get unscaled images for display
    raw_inputs = next(make_data_generator(join(proc_data_path, VALIDATION, INPUT),
        batch_size=batch_size, shuffle=True, augment=True))
    raw_outputs = next(make_data_generator(join(proc_data_path, VALIDATION, OUTPUT),
        batch_size=batch_size, shuffle=True, augment=True))

    predictions = one_hot_to_rgb_batch(model.predict(inputs, batch_size))

    for i in range(batch_size):
        input_im, output_im, prediction_im = \
            raw_inputs[i, :, :, :], raw_outputs[i, :, :, :], predictions[i, :, :, :]

        save_image(join(eval_path, '{}-input.png'.format(i)), input_im)
        save_image(join(eval_path, '{}-output.png'.format(i)), output_im)
        save_image(join(eval_path, '{}-prediction.png'.format(i)), prediction_im)

def get_samples(data_gen, batch_size, nb_samples):
    samples = []
    for _ in range(0, nb_samples, batch_size):
        samples.append(next(data_gen))
    return np.concatenate(samples, axis=0)[0:nb_samples, :, :, :]

def compute_metrics(model, batch_size, val_samples):
    input_gen = make_data_generator(join(proc_data_path, VALIDATION, INPUT),
        shuffle=True, batch_size=batch_size, scale=True)

    output_gen = make_data_generator(join(proc_data_path, VALIDATION, OUTPUT),
        shuffle=True, batch_size=batch_size)

    inputs = get_samples(input_gen, batch_size, val_samples)
    predictions = one_hot_to_label_batch(model.predict(inputs))
    outputs = rgb_to_label_batch(get_samples(
        output_gen, batch_size, val_samples))

    predictions = np.ravel(predictions)
    outputs = np.ravel(outputs)

    # See https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection#evaluation #noqa
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
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_mat': confusion_mat,
        'accuracy': accuracy
    }

    return scores

if __name__ == '__main__':
    model_file_name = 'cl.h5'
    model = load_model(join(model_path, model_file_name))
    batch_size = 32
    val_samples = 300

    visualize_predictions(model, batch_size)
    scores = compute_metrics(model, batch_size, val_samples)
    pp.pprint(scores)
