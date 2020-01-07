from rastervision.v2.rv.backend import Backend


class PyTorchChipClassification(Backend):
    def __init__(self, learner):
        self.learner = learner
        
    def process_scene_data(self, scene, data, tmp_dir):
        pass

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        pass

    def train(self, tmp_dir):
        pass

    def load_model(self, tmp_dir):
        pass

    def predict(self, chips, windows, tmp_dir):
        pass
