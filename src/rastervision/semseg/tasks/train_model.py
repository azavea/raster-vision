from rastervision.common.tasks.train_model import TrainModel


class SemsegTrainModel(TrainModel):
    def __init__(self, run_path, sync_results, options,
                 generator, model):
        super().__init__(
            run_path, sync_results, options, generator, model)

        self.metrics = ['accuracy']
        self.loss_function = 'categorical_crossentropy'
