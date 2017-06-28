from rastervision.common.data.generators import (
    all_augment_methods, safe_augment_methods)

AGG_SUMMARY = 'agg_summary'
AGG_ENSEMBLE = 'agg_ensemble'
AGG_CONCAT = 'agg_concat'

class Options():
    """Represents the options used to control an experimental run."""

    def __init__(self, options):
        self.problem_type = options['problem_type']
        self.run_name = options['run_name']
        self.aggregate_type = options.get('aggregate_type')
        self.aggregate_run_names = options.get('aggregate_run_names')
        self.batch_size = options.get('batch_size', 32)

        # Controls how many samples to use in eval tasks.
        # Setting this to a low value can be useful when testing
        # the code, since it will save time.
        self.nb_eval_samples = options.get('nb_eval_samples')
        # Controls how many samples to plot as part of validation_eval
        self.nb_eval_plot_samples = options.get('nb_eval_plot_samples')

        self.train_stages = options.get('train_stages')
        if self.train_stages is not None:
            options.update(self.train_stages[0])

        self.model_type = options['model_type']
        self.dataset_name = options['dataset_name']
        self.generator_name = options['generator_name']
        self.epochs = options['epochs']
        self.steps_per_epoch = options['steps_per_epoch']
        self.validation_steps = options['validation_steps']
        self.active_input_inds = options['active_input_inds']

        # Optional options
        self.git_commit = options.get('git_commit')
        # Size of the imgs used as input to the network
        # [nb_rows, nb_cols]
        self.target_size = options.get('target_size', (256, 256))
        self.optimizer = options.get('optimizer', 'adam')
        self.init_lr = options.get('init_lr', 1e-3)
        self.patience = options.get('patience')
        self.lr_schedule = options.get('lr_schedule')
        self.train_ratio = options.get('train_ratio')
        self.cross_validation = options.get('cross_validation')
        self.delta_model_checkpoint = options.get(
            'delta_model_checkpoint', None)
        self.augment_methods = options.get(
            'augment_methods', safe_augment_methods)
        if self.augment_methods is not None:
            invalid_augment_methods = \
                set(self.augment_methods) - set(all_augment_methods)
            if invalid_augment_methods:
                raise ValueError(
                    '{} are not valid augment_methods'.format(
                        str(invalid_augment_methods)))
