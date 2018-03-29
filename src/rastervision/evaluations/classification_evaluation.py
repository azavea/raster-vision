from rastervision.core.evaluation import Evaluation


class ClassificationEvaluation(Evaluation):
    def clear(self):
        pass

    def compute(ground_truth_label_store, prediction_label_store):
        pass

    def merge(self, evaluation):
        pass

    def save(self, output_uri):
        pass
