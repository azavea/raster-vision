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
        self.momentum = options.get('momentum')
        self.patience = options.get('patience')
        self.lr_schedule = options.get('lr_schedule')
        self.train_ratio = options.get('train_ratio')
        self.cross_validation = options.get('cross_validation')
        self.test_augmentation = options.get('test_augmentation', True)
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

        # for sgd
        self.momentum = options.get('momentum', 0)
        self.nesterov = options.get('nesterov', False)

        # for rmsprop
        self.rho = options.get('rho', 0.9)
        self.epsilon = options.get('epsilon', 1e-8)

        self.use_best_model = options.get('use_best_model', True)

        # decay options
        decay_set = False
        self.lr_step_decay = options.get('lr_step_decay', 0.0)
        if self.lr_step_decay != 0.0:
            decay_set = True
        self.lr_epoch_decay = options.get('lr_epoch_decay', 0.0)
        if self.lr_epoch_decay != 0.0:
            if decay_set:
                raise ValueError('Cannot set more than one decay option.')
            decay_set = True
        self.cyclic_lr = options.get('cyclic_lr')
        if self.cyclic_lr is not None:
            if decay_set:
                raise ValueError('Cannot set more than one decay option.')
            self.base_lr = options['cyclic_lr']['base_lr']
            self.max_lr = options['cyclic_lr']['max_lr']
            self.step_size = options['cyclic_lr']['step_size']
            self.cycle_mode = options['cyclic_lr']['cycle_mode']
