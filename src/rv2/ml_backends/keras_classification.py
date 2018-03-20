from rv2.core.ml_backend import MLBackend


class KerasClassification(MLBackend):
    def convert_training_data(self, training_data, validation_data, class_map,
                              options):
        pass

    def train(self, options):
        pass

    def predict(self, chip, options):
        pass
