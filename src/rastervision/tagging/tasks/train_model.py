from rastervision.common.tasks.train_model import TrainModel


class TaggingTrainModel(TrainModel):
    def __init__(self, run_path, sync_results, options,
                 generator, model):
        super().__init__(
            run_path, sync_results, options, generator, model)

        self.metrics = ['binary_accuracy']
        self.loss_function = 'binary_crossentropy'
