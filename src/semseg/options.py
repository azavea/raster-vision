import json

from .models.conv_logistic import CONV_LOGISTIC
from .models.fcn_resnet import FCN_RESNET
from .models.fc_densenet import FC_DENSENET
from .models.ensemble import CONCAT_ENSEMBLE, AVG_ENSEMBLE
from .data.potsdam import POTSDAM, PotsdamDataset
from .data.vaihingen import VAIHINGEN


class RunOptions():
    """Represents the options used to control an experimental run."""

    def __init__(self, options):
        if 'train_stages' in options and options['train_stages'] is not None:
            train_stages = options['train_stages']
            options.update(train_stages[0])
        self.train_stages = options.get('train_stages')

        # Required options
        self.model_type = options['model_type']
        self.run_name = options['run_name']
        self.dataset_name = options['dataset_name']
        self.generator_name = options['generator_name']
        self.batch_size = options['batch_size']
        self.nb_epoch = options['nb_epoch']
        self.samples_per_epoch = options['samples_per_epoch']
        self.nb_val_samples = options['nb_val_samples']
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

        # Controls how many samples to use in the final evaluation.
        # Setting this to a low value can be useful when testing
        # the code, since it will save time.
        self.nb_eval_samples = options.get('nb_eval_samples')

        # Model type dependent options
        if self.model_type == CONV_LOGISTIC:
            self.kernel_size = options['kernel_size']
        elif self.model_type == FC_DENSENET:
            self.growth_rate = options['growth_rate']
            self.drop_prob = options['drop_prob']
            self.weight_decay = options['weight_decay']
            self.down_blocks = options['down_blocks']
            self.up_blocks = options['up_blocks']
        elif self.model_type == FCN_RESNET:
            self.use_pretraining = options['use_pretraining']
            self.freeze_base = options['freeze_base']
            if self.use_pretraining and len(self.active_input_inds) != 3:
                raise ValueError(
                    'Can only use pretraining with 3 input channels')
            if self.freeze_base and not self.use_pretraining:
                raise ValueError(
                    'If freeze_base == True, then use_pretraining must be True'
                )
        elif self.model_type in [CONCAT_ENSEMBLE, AVG_ENSEMBLE]:
            # Names of the runs that should be combined together into an
            # ensemble. The outputs of each model will be concatenated
            # together and used as the input to this model.
            self.ensemble_run_names = options['ensemble_run_names']

        default_eval_target_size = (2000, 2000) \
            if self.dataset_name == POTSDAM else None

        # Size of the imgs evaluated at each iteration in validation_eval.
        # This should evenly divide the size of the original image.
        # None means to use the size of the original image.
        self.eval_target_size = options.get(
            'eval_target_size', default_eval_target_size)


def load_options(file_path):
    options = None
    with open(file_path) as options_file:
        options = json.load(options_file)
        options = RunOptions(options)

    return options
