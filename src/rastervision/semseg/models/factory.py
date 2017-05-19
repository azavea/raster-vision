from os.path import join

from rastervision.common.models.factory import ModelFactory

from rastervision.semseg.models.conv_logistic import (
    make_conv_logistic, CONV_LOGISTIC)
from rastervision.semseg.models.fcn_resnet import (
    make_fcn_resnet, FCN_RESNET)
from rastervision.semseg.models.dual_fcn_resnet import (
    make_dual_fcn_resnet, DUAL_FCN_RESNET)
from rastervision.semseg.models.unet import (
    make_unet, UNET)
from rastervision.semseg.models.fc_densenet import (
    make_fc_densenet, FC_DENSENET)
from rastervision.semseg.models.ensemble import (
    ConcatEnsemble, AvgEnsemble, CONCAT_ENSEMBLE, AVG_ENSEMBLE)


class SemsegModelFactory(ModelFactory):
    def __init__(self):
        super().__init__()

    def load_ensemble_models(self, options):
        from ..options import load_options
        from ..data.factory import get_data_generator

        models = []
        active_input_inds_list = []

        for run_name in options.ensemble_run_names:
            self.s3_download(run_name, 'options.json')
            self.s3_download(run_name, 'best_model.h5')

            run_path = join(self.results_path, run_name)
            options_path = join(run_path, 'options.json')
            options = load_options(options_path)
            generator = get_data_generator(options, self.datasets_path)
            active_input_inds = generator.active_input_inds
            model = self.load_model(
                run_path, options, generator, use_best=True)

            models.append(model)
            active_input_inds_list.append(active_input_inds)

        return models, active_input_inds_list

    def make_model(self, options, generator):
        """Make a new model."""
        model_type = options.model_type
        input_shape = (options.target_size[0], options.target_size[1],
                       len(options.active_input_inds))
        nb_labels = generator.dataset.nb_labels

        if model_type == CONV_LOGISTIC:
            model = make_conv_logistic(input_shape, nb_labels,
                                       options.kernel_size)
        elif model_type == FCN_RESNET:
            model = make_fcn_resnet(
                input_shape, nb_labels, options.use_pretraining,
                options.freeze_base)
        elif model_type == DUAL_FCN_RESNET:
            model = make_dual_fcn_resnet(
                input_shape, options.dual_active_input_inds,
                nb_labels, options.use_pretraining, options.freeze_base)
        elif model_type == UNET:
            model = make_unet(input_shape, nb_labels)
        elif model_type == FC_DENSENET:
            model = make_fc_densenet(
                input_shape, nb_labels, drop_prob=options.drop_prob,
                weight_decay=options.weight_decay,
                down_blocks=options.down_blocks,
                up_blocks=options.up_blocks)
        elif model_type in [CONCAT_ENSEMBLE, AVG_ENSEMBLE]:
            models, active_input_inds_list = self.load_ensemble_models(options)
            if model_type == CONCAT_ENSEMBLE:
                model = ConcatEnsemble(
                    models, active_input_inds_list, input_shape, nb_labels)
            elif model_type == AVG_ENSEMBLE:
                model = AvgEnsemble(models, active_input_inds_list)
        else:
            raise ValueError('{} is not a valid model_type'.format(model_type))

        return model
