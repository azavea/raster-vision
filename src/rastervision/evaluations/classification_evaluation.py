from rastervision.core.evaluation import Evaluation


class ClassificationEvaluation(Evaluation):
    def clear(self):
        pass

    def compute(ground_truth_label_source, prediction_label_source):
        pass

    def merge(self, evaluation):
        pass

    def save(self, output_uri):
        pass
