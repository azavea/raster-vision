from rastervision.common.tasks.train_model import TrainModel


class TaggingTrainModel(TrainModel):
    def __init__(self, run_path, sync_results, options,
                 generator, model):
        super().__init__(
            run_path, sync_results, options, generator, model)

        self.metrics = options.metrics
        self.loss_function = options.loss_function
