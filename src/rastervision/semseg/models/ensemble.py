import numpy as np

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv2D, Activation)

CONCAT_ENSEMBLE = 'concat_ensemble'
AVG_ENSEMBLE = 'avg_ensemble'


class ConcatEnsemble(Model):
    """Combines several saved models into an ensemble."""

    def __init__(self, models, active_input_inds_list, input_shape, nb_labels):
        # See https://github.com/fchollet/keras/issues/2397
        self.graph = tf.get_default_graph()
        self.models = models
        self.active_input_inds_list = active_input_inds_list

        nb_channels = len(models) * nb_labels
        input_shape = (input_shape[0], input_shape[1], nb_channels)
        nb_labels = nb_labels

        nb_rows, nb_cols, _ = input_shape
        input_tensor = Input(shape=input_shape)

        x = Conv2D(
            nb_labels, 1, 1, name='conv_labels')(input_tensor)
        x = Activation('softmax', axis=3)(x)

        super().__init__(input=input_tensor, output=x)

    def make_ensemble_batch(self, batch_x):
        # See https://github.com/fchollet/keras/issues/2397
        with self.graph.as_default():
            ensemble_batch_x = []
            for model, active_input_inds in \
                    zip(self.models, self.active_input_inds_list):
                ensemble_batch_x.append(model.predict(
                    batch_x[:, :, :, active_input_inds]))
            ensemble_batch_x = np.concatenate(ensemble_batch_x, axis=3)
            return ensemble_batch_x

    def fit_generator(self, train_gen, **kwargs):
        validation_gen = kwargs['validation_data']

        def make_ensemble_gen(gen):
            for batch_x, batch_y in gen:
                ensemble_batch_x = self.make_ensemble_batch(batch_x)
                yield ensemble_batch_x, batch_y

        train_gen = make_ensemble_gen(train_gen)
        validation_gen = make_ensemble_gen(validation_gen)

        kwargs['validation_data'] = validation_gen
        return super().fit_generator(train_gen, **kwargs)

    def predict(self, batch_x):
        ensemble_batch_x = self.make_ensemble_batch(batch_x)
        return super().predict(ensemble_batch_x)


class AvgEnsemble():
    """Combines saved models into an ensemble by averaging their outputs."""

    def __init__(self, models, active_input_inds_list):
        # See https://github.com/fchollet/keras/issues/2397
        self.graph = tf.get_default_graph()
        self.models = models
        self.active_input_inds_list = active_input_inds_list

    def make_ensemble_batch_x(self, batch_x):
        # See https://github.com/fchollet/keras/issues/2397
        with self.graph.as_default():
            ensemble_batch_x = []
            for model, active_input_inds in \
                    zip(self.models, self.active_input_inds_list):
                ensemble_batch_x.append(model.predict(
                    batch_x[:, :, :, active_input_inds]))
            ensemble_batch_x = np.array(ensemble_batch_x)
            return ensemble_batch_x

    def fit_generator(self, train_gen, **kwargs):
        pass

    def compile(self, **kwargs):
        pass

    def load_weights(self, model_path, **kwargs):
        pass

    def summary(self):
        return 'avg'

    def predict(self, batch_x):
        ensemble_batch_x = self.make_ensemble_batch_x(batch_x)
        return np.mean(ensemble_batch_x, axis=0)
