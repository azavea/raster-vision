import json

from .models.conv_logistic import CONV_LOGISTIC
from .models.fcn_resnet import FCN_RESNET
from .models.fc_densenet import FC_DENSENET
from .data.potsdam import POTSDAM, PotsdamDataset
from .data.vaihingen import VAIHINGEN


class RunOptions():
    """Represents the options used to control an experimental run."""

    def __init__(self, options):
        if 'train_stages' in options:
            train_stages = options['train_stages']
            options.update(train_stages[0])
            self.train_stages = train_stages
        # Required options
        self.model_type = options['model_type']
        self.run_name = options['run_name']
        self.dataset_name = options['dataset_name']
        self.generator_name = options['generator_name']
        self.include_ir = options['include_ir']
        self.include_depth = options['include_depth']
        self.include_ndvi = options['include_ndvi']
        # Size of the imgs used as input to the network
        # [nb_rows, nb_cols]
        self.target_size = options['target_size']
        # Size of the imgs evaluated at each iteration in validation_eval.
        # This should evenly divide the size of the original image.
        # None means to use the size of the original image.
        self.eval_target_size = options['eval_target_size']

        self.batch_size = options['batch_size']
        self.nb_epoch = options['nb_epoch']
        self.samples_per_epoch = options['samples_per_epoch']
        self.nb_val_samples = options['nb_val_samples']
        self.optimizer = options['optimizer']
        self.init_lr = options['init_lr']

        self.git_commit = options['git_commit']

        # Optional options
        self.patience = options.get('patience')
        self.lr_schedule = options.get('lr_schedule')
        # Controls how many samples to use in the final evaluation.
        # Setting this to a low value can be useful when testing
        # the code, since it will save time.
        self.nb_eval_samples = options.get('nb_eval_samples')

        # TODO make all options optional and have datasets and models
        # check for options that they require
        # model_type dependent options
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

            not_three_channels = (
                self.include_ir or self.include_depth or self.include_ndvi)
            if self.use_pretraining and not_three_channels:
                raise ValueError(
                    'Can only use pretraining with 3 input channels')
            if self.freeze_base and not self.use_pretraining:
                raise ValueError(
                    'If freeze_base == True, then use_pretraining must be True'
                )

        # dataset dependent options
        if (self.dataset_name == POTSDAM and 'sharah_train_ratio' in options
                and options['sharah_train_ratio']):
            self.train_ratio = PotsdamDataset.sharah_train_ratio
        else:
            self.train_ratio = options['train_ratio']


def load_options(file_path):
    options = None
    with open(file_path) as options_file:
        options = json.load(options_file)
        options = RunOptions(options)

    return options
