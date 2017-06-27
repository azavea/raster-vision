from rastervision.common.options import Options
from rastervision.common.data.generators import ROTATE, TRANSLATE

from rastervision.semseg.models.conv_logistic import CONV_LOGISTIC
from rastervision.semseg.models.fcn_resnet import FCN_RESNET
from rastervision.semseg.models.dual_fcn_resnet import DUAL_FCN_RESNET
from rastervision.semseg.models.fc_densenet import FC_DENSENET
from rastervision.semseg.models.ensemble import CONCAT_ENSEMBLE, AVG_ENSEMBLE
from rastervision.semseg.data.potsdam import POTSDAM


class SemsegOptions(Options):
    """Represents the options used to control an experimental run."""

    def __init__(self, options):
        super().__init__(options)

        if (self.augment_methods is not None and
            (ROTATE in self.augment_methods or
             TRANSLATE in self.augment_methods)):
            raise ValueError('Cannot use rotate or translate with semseg.')

        self.nb_videos = options.get('nb_videos')

        if self.model_type == CONV_LOGISTIC:
            self.kernel_size = options['kernel_size']
        elif self.model_type == FC_DENSENET:
            self.growth_rate = options['growth_rate']
            self.drop_prob = options['drop_prob']
            self.weight_decay = options['weight_decay']
            self.down_blocks = options['down_blocks']
            self.up_blocks = options['up_blocks']
        elif self.model_type in [FCN_RESNET, DUAL_FCN_RESNET]:
            self.use_pretraining = options['use_pretraining']
            self.freeze_base = options['freeze_base']

        if self.model_type == DUAL_FCN_RESNET:
            self.dual_active_input_inds = options['dual_active_input_inds']
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
